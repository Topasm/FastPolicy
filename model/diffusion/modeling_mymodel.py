import math
from collections import deque
from typing import Callable, Optional
import os

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn
from pathlib import Path
import safetensors

from lerobot.common.constants import OBS_ENV, OBS_ROBOT, OBS_IMAGE
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.diffusion_modules import (
    DiffusionRgbEncoder,
    DiffusionTransformer,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)

from model.critic.multimodal_scorer import MultimodalTrajectoryScorer
from model.invdynamics.invdyn import MlpInvDynamic


class MYDiffusionPolicy(PreTrainedPolicy):
    """
    Diffusion Policy using a Diffusion Transformer (DiT) backbone, inspired by
    "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy)
    and DiT (https://arxiv.org/abs/2212.09748).
    """

    config_class = DiffusionConfig
    name = "mydiffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics for normalization.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalizers/Unnormalizers
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)

        # Inverse dynamics normalizes states and actions
        self.normalize_invdyn_state = Normalize(
            {"observation.state": config.robot_state_feature},
            config.normalization_mapping, dataset_stats
        )
        self.normalize_invdyn_action = Normalize(
            {"action": config.action_feature},
            config.normalization_mapping, dataset_stats
        )
        self.unnormalize_action_output = Unnormalize(
            {"action": config.action_feature},  # Only unnormalize action
            config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy
        self._queues = None
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        # Instantiate the diffusion model (now using DiT)
        self.diffusion = MyDiffusionModel(config)
        # Instantiate MlpInvDynamic
        self.inv_dyn_model = MlpInvDynamic(
            o_dim=self.state_dim * 2,
            a_dim=self.action_dim,
            hidden_dim=self.config.inv_dyn_hidden_dim,
            dropout=0.1,
            use_layernorm=True,
            out_activation=nn.Tanh(),
        )
        # Instantiate MultimodalTrajectoryScorer
        self.critic_scorer = MultimodalTrajectoryScorer(
            state_dim=self.state_dim,
            horizon=config.horizon,
            hidden_dim=config.critic_hidden_dim
        )

        # Determine and store the device AFTER loading weights
        self.diffusion.to(config.device)
        self.inv_dyn_model.to(config.device)
        self.critic_scorer.to(config.device)
        self.device = get_device_from_parameters(self.diffusion)

        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """
        Instantiate a policy from a pretrained checkpoint directory.
        Overrides the base method to load individual component checkpoints.

        Expects directory structure:
        - config.json
        - stats.safetensors
        - diffusion.pth (state_dict for MyDiffusionModel)
        - invdyn.pth (state_dict for MlpInvDynamic)
        - critic.pth (state_dict for MultimodalTrajectoryScorer, optional)
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        # 1. Load config
        config_path = pretrained_model_name_or_path / "config.json"
        if not config_path.is_file():
            raise OSError(
                f"config.json not found in {pretrained_model_name_or_path}")
        config = cls.config_class.from_json_file(config_path)

        # 2. Load dataset stats
        stats_path = pretrained_model_name_or_path / "stats.safetensors"
        if not stats_path.is_file():
            raise OSError(
                f"stats.safetensors not found in {pretrained_model_name_or_path}")
        with safetensors.safe_open(stats_path, framework="pt", device="cpu") as f:
            dataset_stats = {k: f.get_tensor(k) for k in f.keys()}

        # 3. Instantiate the policy
        policy = cls(config, dataset_stats)
        policy.eval()  # Set to eval mode by default after loading

        # 4. Load individual component state dicts
        device = config.device  # Use device from config

        diffusion_ckpt_path = pretrained_model_name_or_path / "diffusion.pth"
        if diffusion_ckpt_path.is_file():
            print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
            diff_state_dict = torch.load(
                diffusion_ckpt_path, map_location="cpu")
            policy.diffusion.load_state_dict(diff_state_dict)
        else:
            print(
                f"Warning: diffusion.pth not found in {pretrained_model_name_or_path}. Diffusion model not loaded.")

        invdyn_ckpt_path = pretrained_model_name_or_path / "invdyn.pth"
        if invdyn_ckpt_path.is_file():
            print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
            inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
            policy.inv_dyn_model.load_state_dict(inv_state_dict)
        else:
            print(
                f"Warning: invdyn.pth not found in {pretrained_model_name_or_path}. Inverse dynamics model not loaded.")

        critic_ckpt_path = pretrained_model_name_or_path / "critic.pth"
        if critic_ckpt_path.is_file():
            print(f"Loading critic state dict from: {critic_ckpt_path}")
            crit_state_dict = torch.load(critic_ckpt_path, map_location="cpu")
            policy.critic_scorer.load_state_dict(crit_state_dict)
        else:
            # Critic might be optional
            print(
                f"Info: critic.pth not found in {pretrained_model_name_or_path}. Critic model not loaded.")

        # Move policy to the correct device AFTER loading state dicts
        policy.to(device)
        policy.device = device  # Update device attribute

        return policy

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            # Use a single key for stacked images in the queue
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs
        norm_batch = self.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(norm_batch)
            processed_batch["observation.image"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            processed_batch = norm_batch  # Use the normalized batch

        # Populate queues with the latest *normalized* observation
        self._queues = populate_queues(self._queues, processed_batch)

        # Generate new action plan only when the action queue is empty

        if len(self._queues["action"]) == 0:
            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

            # Get the very last state (already normalized)
            current_state = model_input_batch["observation.state"][:, 0, :]
            num_samples = getattr(self.config, "num_inference_samples", 1)

            # Pass normalized batch and state to generation function
            actions = self.diffusion.generate_actions_via_inverse_dynamics(
                model_input_batch,  # Pass normalized batch
                current_state,     # Pass normalized state
                self.inv_dyn_model,
                num_samples=num_samples,
            )  # Returns normalized actions

            # Unnormalize actions
            actions_unnormalized = self.unnormalize_action_output(
                {"action": actions})["action"]

            self._queues["action"].extend(actions_unnormalized.transpose(0, 1))

        # Pop the next action from the queue
        action = self._queues["action"].popleft()
        return action


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances.
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class MyDiffusionModel(nn.Module):
    """
    Main model class coordinating the vision encoders, DiT, and noise scheduler.
    """

    def __init__(self, config: DiffusionConfig, dataset_stats: dict[str, dict[str, Tensor]] | None = None):
        super().__init__()
        self.config = config
        self.state_dim = config.robot_state_feature.shape[0]  # Store state dim
        self.action_dim = config.action_feature.shape[0]  # Store action dim

        # Add normalizers/unnormalizers
        if dataset_stats is not None:
            self.normalize_inputs = Normalize(
                config.input_features, config.normalization_mapping, dataset_stats)
            self.unnormalize_action_output = Unnormalize(
                {"action": config.action_feature},
                config.normalization_mapping, dataset_stats
            )
        else:
            # Provide dummy implementations if no stats available
            self.normalize_inputs = lambda x: x
            self.unnormalize_action_output = lambda x: x

        # Determine the dimension the diffusion model should predict
        self.diffusion_target_dim = self.state_dim  # Always predict state dim now

        # --- Observation Encoders ---
        global_cond_dim_per_step = 0  # Calculate conditioning dim per timestep first

        # Robot state dimension
        if config.robot_state_feature:
            global_cond_dim_per_step += config.robot_state_feature.shape[0]
        else:
            raise ValueError(
                "`observation.state` (robot_state_feature) is required for conditioning.")

        # Image features
        self.rgb_encoder = None
        if config.image_features:
            num_cameras = len(config.image_features)
            image_feature_dim_per_cam = config.transformer_dim
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList([
                    DiffusionRgbEncoder(config) for _ in range(num_cameras)
                ])
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)

            global_cond_dim_per_step += image_feature_dim_per_cam * num_cameras

        # Environment state features
        self.env_state_encoder = None
        if config.env_state_feature:
            env_state_dim = config.env_state_feature.shape[0]
            global_cond_dim_per_step += env_state_dim

        # Total global conditioning dimension after flattening history
        global_cond_dim_total = global_cond_dim_per_step * config.n_obs_steps

        # --- Diffusion Transformer ---
        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim_total,
            output_dim=self.diffusion_target_dim  # Predict state dimension
        )

        # --- Noise Scheduler ---
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        # --- Inference Steps ---
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _encode_images(self, images: Tensor) -> Tensor:
        """Encodes images using the appropriate encoder(s)."""
        # Handle various image tensor shapes (including the 6D tensor case)
        if len(images.shape) == 6 and images.shape[2] == 1:
            # Special case: Shape is [b, t, 1, c, h, w] - extra dimension
            B, T_obs = images.shape[:2]
            # Remove the extra dimension
            images = images.squeeze(2)
            N_cam = 1  # Set N_cam to 1 since we've removed that dimension
        else:
            # Standard case: Shape is [b, t, n, c, h, w]
            B, T_obs, N_cam = images.shape[:3]

        try:
            if self.config.use_separate_rgb_encoder_per_camera:
                images_reshaped = einops.rearrange(
                    images, "b t n c h w -> n (b t) c h w")
                features_list = []
                for i in range(N_cam):
                    features_list.append(
                        self.rgb_encoder[i](images_reshaped[i]))
                img_features = torch.stack(features_list, dim=0)
                img_features = einops.rearrange(
                    img_features, "n (b t) d -> b t (n d)", b=B, t=T_obs)
            else:
                images_reshaped = einops.rearrange(
                    images, "b t n c h w -> (b t n) c h w")
                img_features = self.rgb_encoder(images_reshaped)
                img_features = einops.rearrange(
                    img_features, "(b t n) d -> b t (n d)", b=B, t=T_obs, n=N_cam)
            return img_features
        except Exception as e:
            print(f"Error during image encoding: {e}")
            print(f"Image shape: {images.shape}")
            # Return a zero tensor as fallback
            return torch.zeros((B, T_obs, self.config.transformer_dim * N_cam),
                               device=images.device)

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector.
           Expects batch to have normalized 'observation.state' and potentially 'observation.image'.
        """
        # Check required keys exist
        if OBS_ROBOT not in batch:
            raise KeyError(
                f"Missing '{OBS_ROBOT}' in batch for _prepare_global_conditioning")

        batch_size = batch[OBS_ROBOT].shape[0]
        n_obs_steps = self.config.n_obs_steps

        if batch[OBS_ROBOT].shape[1] < n_obs_steps:
            raise ValueError(
                f"{OBS_ROBOT} sequence length ({batch[OBS_ROBOT].shape[1]}) "
                f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
            )
        cond_state = batch[OBS_ROBOT][:, :n_obs_steps, :]
        global_cond_feats = [cond_state]

        # Check if images are configured AND present in the batch before processing
        if self.config.image_features and OBS_IMAGE in batch:
            images = batch[OBS_IMAGE]
            _B = images.shape[0]
            n_img_steps = images.shape[1]

            # Handle different image tensor shapes
            if len(images.shape) == 6:  # Shape: [b, t, n_cam, c, h, w]
                # Special case for 6D tensor
                if images.shape[2] == 1:  # If there's just one camera, squeeze that dimension
                    images = images.squeeze(2)  # Convert to [b, t, c, h, w]
                else:
                    # Use the _encode_images method which can handle multi-camera input
                    try:
                        img_features = self._encode_images(images)
                        global_cond_feats.append(img_features)
                        # Skip the rest of the processing for this case
                        goto_next_condition = True
                    except Exception as e:
                        print(f"Error processing 6D image tensor: {e}")
                        print(f"Image shape: {images.shape}")
                        # Create a fallback feature tensor
                        img_features = torch.zeros((batch_size, n_obs_steps, self.config.transformer_dim),
                                                   device=batch["observation.state"].device)
                        global_cond_feats.append(img_features)
                        goto_next_condition = True

            # If we need to skip to the next condition
            if 'goto_next_condition' in locals() and goto_next_condition:
                pass
            # Standard processing for 5D tensor [b, t, c, h, w]
            elif len(images.shape) == 5:
                if n_img_steps != n_obs_steps:
                    raise ValueError(
                        f"Image sequence length ({n_img_steps}) in batch does not match "
                        f"configured n_obs_steps ({n_obs_steps}). Check dataset delta_timestamps "
                        f"and policy config."
                    )
                assert _B == batch_size

                try:
                    images_reshaped = einops.rearrange(
                        images, "b t c h w -> (b t) c h w")
                    img_features = self.rgb_encoder(images_reshaped)
                    img_features = einops.rearrange(
                        img_features, "(b t) d -> b t d", b=batch_size, t=n_obs_steps
                    )
                    global_cond_feats.append(img_features)
                except Exception as e:
                    print(f"Error during standard image processing: {e}")
                    print(f"Image shape: {images.shape}")
                    # Create a fallback feature tensor
                    img_features = torch.zeros((batch_size, n_obs_steps, self.config.transformer_dim),
                                               device=batch["observation.state"].device)
                    global_cond_feats.append(img_features)
            else:
                # Unknown image tensor format
                raise ValueError(
                    f"Unsupported image tensor shape: {images.shape}")

        elif self.config.image_features and "observation.image" not in batch:
            # If images configured but not provided in this specific batch, print warning
            print("Warning: image_features configured but 'observation.image' not found in batch for _prepare_global_conditioning.")
            # Continue without image features for this batch

        # Check if env state is configured AND present
        if self.config.env_state_feature and OBS_ENV in batch:
            if batch[OBS_ENV].shape[1] < n_obs_steps:
                raise ValueError(
                    f"{OBS_ENV} sequence length ({batch[OBS_ENV].shape[1]}) "
                    f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
                )
            cond_env_state = batch[OBS_ENV][:, :n_obs_steps, :]
            global_cond_feats.append(cond_env_state)
        elif self.config.env_state_feature and OBS_ENV not in batch:
            print(
                f"Warning: env_state_feature configured but '{OBS_ENV}' not found in batch for _prepare_global_conditioning.")
            # Continue without env state features for this batch

        concatenated_features = torch.cat(global_cond_feats, dim=-1)
        global_cond = concatenated_features.flatten(start_dim=1)
        return global_cond

    def compute_diffusion_loss(self, norm_batch: dict[str, Tensor]) -> Tensor:
        """Computes the diffusion loss.
        Expects norm_batch to be a fully normalized batch containing all keys
        loaded by the dataset (state seq, image history, padding mask).
        Handles internal slicing for conditioning and target.
        """
        n_obs_steps = self.config.n_obs_steps
        horizon = self.config.horizon

        # --- Prepare Conditioning Data (History) ---
        if "observation.state" not in norm_batch:
            raise KeyError(
                "Missing 'observation.state' in norm_batch for compute_diffusion_loss")

        full_state_sequence = norm_batch["observation.state"]
        # Ensure sequence is long enough for history
        if full_state_sequence.shape[1] < n_obs_steps:
            raise ValueError(
                f"Full state sequence too short for history. Need {n_obs_steps}, got {full_state_sequence.shape[1]}")
        # Shape (B, n_obs, D)
        history_state = full_state_sequence[:, :n_obs_steps]

        cond_batch = {"observation.state": history_state}

        # Check for image history
        if self.config.image_features:
            if OBS_IMAGE in norm_batch:
                image_history = norm_batch["observation.image"]
                # Check if image sequence length matches n_obs_steps
                if image_history.shape[1] != n_obs_steps:
                    # If dataset loaded more images than needed (e.g., full sequence), slice here
                    if image_history.shape[1] > n_obs_steps:
                        print(
                            f"Warning: Image sequence longer than n_obs_steps ({image_history.shape[1]} > {n_obs_steps}). Slicing.")
                        cond_batch["observation.image"] = image_history[:,
                                                                        :n_obs_steps]
                    else:  # If shorter, it's an error
                        raise ValueError(
                            f"Image history length mismatch. Expected {n_obs_steps}, got {image_history.shape[1]}")
                else:  # Length matches
                    cond_batch["observation.image"] = image_history
            else:
                # If images configured but not in batch, print warning. global_cond will be smaller.
                print(
                    "Warning: image_features configured but 'observation.image' not found in norm_batch.")

        # Add env state history if configured and present
        if self.config.env_state_feature:
            if OBS_ROBOT in norm_batch:
                full_env_state = norm_batch[OBS_ROBOT]
                if full_env_state.shape[1] < n_obs_steps:
                    raise ValueError(
                        f"Env state sequence too short for history. Need {n_obs_steps}, got {full_env_state.shape[1]}")
                cond_batch[OBS_ROBOT] = full_env_state[:, :n_obs_steps]
            else:
                print(
                    f"Warning: env_state_feature configured but '{OBS_ROBOT}' not found in norm_batch.")

        # Calculate global_cond based on available history features in cond_batch
        # This tensor's dimension MUST match the expected input dim of cond_embed layer
        global_cond = self._prepare_global_conditioning(cond_batch)

        # --- Prepare Target Data (Future States) ---
        # Assuming state prediction target is always this key
        target_key = "observation.state"
        # Ensure sequence is long enough for target horizon starting after history
        expected_target_end_idx = n_obs_steps + horizon
        if full_state_sequence.shape[1] < expected_target_end_idx:
            raise ValueError(
                f"Full state sequence too short for target. Need length {expected_target_end_idx}, got {full_state_sequence.shape[1]}")
        # Target states are s_1...s_H, which are at indices n_obs_steps to n_obs_steps+horizon-1
        # Shape (B, H, D)
        clean_targets = full_state_sequence[:,
                                            n_obs_steps-1:expected_target_end_idx-1, :]

        if clean_targets.shape[1] != horizon:
            # This check should be redundant if the length check above passes, but good for safety
            raise ValueError(
                f"Target state slicing failed. Expected horizon {horizon}, got {clean_targets.shape[1]}")

        B = clean_targets.shape[0]
        device = clean_targets.device

        # --- Diffusion Process ---
        noise = torch.randn(clean_targets.shape, device=device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,), device=device,
        ).long()
        noisy_targets = self.noise_scheduler.add_noise(
            clean_targets, noise, timesteps)

        # --- Transformer Prediction ---
        # This is the likely point of failure if global_cond dimension is wrong
        pred = self.transformer(noisy_targets, timesteps,
                                global_cond=global_cond)

        # --- Loss Calculation ---
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = clean_targets
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction="none")

        # Optional: Mask loss based on padding
        if self.config.do_mask_loss_for_padding and "action_is_pad" in norm_batch:
            padding_mask = norm_batch["action_is_pad"]
            # Ensure padding mask has at least horizon length
            if padding_mask.shape[1] < horizon:
                raise ValueError(
                    f"Padding mask too short. Expected {horizon}, got {padding_mask.shape[1]}")
            # Select the part of the mask corresponding to the target horizon
            mask = ~padding_mask[:, :horizon]  # Shape (B, H)
            mask_expanded = mask.unsqueeze(-1).expand_as(diffusion_loss)
            diffusion_loss = diffusion_loss * mask_expanded
            # Normalize loss by number of unmasked elements
            diffusion_loss = diffusion_loss.sum() / (mask_expanded.sum() + 1e-8)
        else:
            diffusion_loss = diffusion_loss.mean()  # Original mean loss

        return diffusion_loss

    def compute_invdyn_loss(self, invdyn_batch: dict[str, Tensor], inv_dyn_model: MlpInvDynamic) -> Tensor:
        """Computes only the inverse dynamics loss. Moved here for consistency."""
        n_obs_steps = self.config.n_obs_steps
        horizon = self.config.horizon

        expected_len = len(self.config.state_delta_indices)
        loaded_state_len = invdyn_batch["observation.state"].shape[1]

        if loaded_state_len != expected_len:
            raise ValueError(
                f"Loaded state tensor for invdyn has unexpected length {loaded_state_len}. "
                # Updated error message
                f"Expected {expected_len} based on config.state_delta_indices."
            )

        # Extract states s_{-1} through s_{H} (or s_{n_obs+H-1})
        # Shape: (B, expected_len, D)
        all_states = invdyn_batch["observation.state"]

        # Create previous and current states for pairs
        # s_prev: s_{-1} to s_{H-1} (or s_{n_obs+H-2}). Shape: (B, expected_len-1, D_state)
        s_prev = all_states[:, :-1, :]  # All states except the last one
        # s_curr: s_{0} to s_{H} (or s_{n_obs+H-1}). Shape: (B, expected_len-1, D_state)
        s_curr = all_states[:, 1:, :]  # All states except the first one

        # Shape: (B, expected_len-1, D_state * 2)
        state_pairs = torch.cat([s_prev, s_curr], dim=-1)

        # We only need the pairs corresponding to t=0..H-1 for loss calculation
        # These are pairs (s_0, s_1) ... (s_{H-1}, s_H)
        # The index for t=0 in state_pairs is n_obs_steps - 1
        start_pair_idx = n_obs_steps - 1
        # We need to make sure we don't go beyond available action data
        # Since actions are [0...15] but states go to [17], we need to limit the horizon
        action_horizon = min(horizon, len(self.config.action_delta_indices))
        end_pair_idx = start_pair_idx + action_horizon

        # Check if our indices would be in range
        if start_pair_idx >= state_pairs.shape[1] or end_pair_idx > state_pairs.shape[1]:
            raise ValueError(
                f"Invalid indices for state_pairs with shape {state_pairs.shape}. "
                f"start_pair_idx={start_pair_idx}, end_pair_idx={end_pair_idx}"
            )

        # Shape: (B, H, D_state * 2)
        state_pairs_for_loss = state_pairs[:, start_pair_idx:end_pair_idx, :]

        B, H, D_pair = state_pairs_for_loss.shape
        if H != action_horizon:  # Compare with action_horizon instead of horizon
            raise ValueError(
                f"Sliced state_pairs_for_loss has wrong horizon {H}, expected {action_horizon}")

        state_pairs_flat = state_pairs_for_loss.reshape(
            B * H, D_pair)  # Use reshape instead of view
        pred_actions_flat = inv_dyn_model(
            state_pairs_flat)  # Use the passed model
        pred_actions = pred_actions_flat.view(
            B, H, self.action_dim)  # Shape: (B, H, A)

        # Get the actions that correspond to the state pairs we're using for loss
        # These are actions a_0 to a_{H-1}
        # Make sure we're using the limited action_horizon here
        # Shape: (B, H, A)
        true_actions = invdyn_batch["action"][:,
                                              n_obs_steps:n_obs_steps+action_horizon, :]

        # Ensure shapes match exactly - this can happen if state horizon > action horizon
        if pred_actions.shape[1] != true_actions.shape[1]:
            # Adjust pred_actions to match true_actions
            pred_actions = pred_actions[:, :true_actions.shape[1], :]

        if self.config.do_mask_loss_for_padding and "action_is_pad" in invdyn_batch:
            # Make sure to use the correct slice of action_is_pad that corresponds to our true_actions
            pad_mask = invdyn_batch["action_is_pad"][:,
                                                     n_obs_steps-1:n_obs_steps-1+action_horizon]
            pad_mask = pad_mask[:, :true_actions.shape[1]]  # Ensure same shape
            # Invert the mask: 0 for padding, 1 for valid
            mask = ~pad_mask  # Shape: (B, H)
            loss_e = F.mse_loss(pred_actions, true_actions,
                                reduction="none")  # Shape: (B, H, A)
            loss_e = loss_e * mask.unsqueeze(-1)
            inv_dyn_loss = loss_e.sum() / (mask.sum() * self.action_dim + 1e-8)
        else:
            inv_dyn_loss = F.mse_loss(pred_actions, true_actions)

        return inv_dyn_loss

    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        """ Samples normalized states """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = torch.randn(
            size=(batch_size, self.config.horizon, self.diffusion_target_dim),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.transformer(
                sample,
                torch.full(sample.shape[:1], t,
                           dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )

            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample

        return sample  # Returns normalized states


class CombinedPolicy(nn.Module):
    """
    Combined policy class for the diffusion model and inverse dynamics model.
    This class is a simple torch.nn.Module that doesn't require config_class.
    """

    def __init__(self, diffusion_model: MyDiffusionModel, inv_dyn_model: MlpInvDynamic):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model
        self.config = diffusion_model.config
        self.device = get_device_from_parameters(diffusion_model)

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Selects an action using the combined policy."""
        # Normalize inputs
        norm_batch = self.diffusion_model.normalize_inputs(batch)

        # Prepare global conditioning
        global_cond = self.diffusion_model._prepare_global_conditioning(
            norm_batch)

        # Generate actions using the diffusion model
        actions = self.diffusion_model.conditional_sample(
            batch_size=batch["observation.state"].shape[0],
            global_cond=global_cond,
        )

        # Unnormalize actions
        actions_unnormalized = self.diffusion_model.unnormalize_action_output(
            {"action": actions})["action"]

        return actions_unnormalized

    @torch.no_grad()
    def generate_actions_via_inverse_dynamics(
        self,
        batch: dict[str, Tensor],  # Expects the raw batch
        num_samples: int = 1,
        action_queues: dict = None,
    ) -> Tensor:
        """Generates actions using diffusion for states and then iteratively applying
        an inverse dynamics model predicting a_t from (s_t, s_{t+1}).
        Returns `n_action_steps` (4) normalized actions.

        If action_queues is provided, will add predicted actions to the queues
        similar to how select_action works with populate_queues.
        """
        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs
        norm_batch = self.diffusion_model.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(norm_batch)
            processed_batch["observation.image"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            processed_batch = norm_batch  # Use the normalized batch

        # Populate queues with the latest *normalized* observation if provided
        if action_queues is not None:
            action_queues = populate_queues(action_queues, processed_batch)

        # Get current state
        norm_current_state = processed_batch["observation.state"][:, 0, :]
        batch_size = norm_current_state.shape[0]
        n_action_steps = self.config.n_action_steps  # This is 4

        # Prepare conditioning from the normalized batch
        global_cond = self._prepare_global_conditioning(norm_batch)

        # Repeat global_cond if num_samples > 1
        if num_samples > 1:
            global_cond = global_cond.repeat_interleave(num_samples, dim=0)

        # Sample normalized future states (Horizon=16)
        predicted_states_flat = self.conditional_sample(
            batch_size * num_samples, global_cond=global_cond
        )  # Output: (B*N, 16, D_state)

        predicted_states = einops.rearrange(
            predicted_states_flat, "(b n) h d -> b n h d", b=batch_size, n=num_samples
        )

        # Always select the first sample if multiple are generated
        selected_states = predicted_states[:, 0]  # (B, 16, D_state)

        # Iteratively predict actions
        predicted_actions = []
        current_s = norm_current_state  # s_0
        for i in range(n_action_steps):  # Iterate 4 times (i=0, 1, 2, 3)
            next_s = selected_states[:, i+1]  # s_1, s_2, s_3, s_4

            # Create state pair (s_i, s_{i+1})
            # Shape: (B, D_state * 2)
            state_pair = torch.cat([current_s, next_s], dim=-1)

            # Predict action a_i
            action_i = inv_dyn_model.predict(
                state_pair)  # Shape: (B, action_dim)
            predicted_actions.append(action_i)

            # Update current state for next iteration
            current_s = next_s

        # Stack the predicted actions
        actions_to_execute = torch.stack(
            predicted_actions, dim=1)  # Shape: (B, 4, action_dim)

        # If action_queues is provided, populate it with actions (like in select_action)
        if action_queues is not None and "action" in action_queues:
            # Clear the existing action queue to replace with new action plan
            while len(action_queues["action"]) > 0:
                action_queues["action"].popleft()

            # Add normalized actions to the queue
            action_batch = {"action": actions_to_execute}
            action_queues = populate_queues(action_queues, action_batch)

        return actions_to_execute  # Return normalized actions
