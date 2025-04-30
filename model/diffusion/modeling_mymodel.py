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

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
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
from model.critic.critic_model import CriticScorer  # Import CriticScorer
# Import your inverse dynamics model
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
        # Diffusion normalizes states
        self.normalize_diffusion_target = Normalize(
            {config.diffusion_target_key:
                config.output_features[config.diffusion_target_key]},
            config.normalization_mapping, dataset_stats
        )
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
            {"action": config.action_feature},
            config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy
        self._queues = None
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        # Instantiate the diffusion model (now using DiT)
        self.diffusion = MyDiffusionModel(config)

        self.inv_dyn_model = MlpInvDynamic(
            o_dim=self.state_dim,
            a_dim=self.action_dim,
            hidden_dim=config.inv_dyn_hidden_dim,  # Use config value
            # Assuming Tanh activation for actions based on MlpInvDynamic default
            out_activation=nn.Tanh()
        )
        self.critic_scorer = CriticScorer(
            state_dim=self.state_dim,
            # Critic likely scores state from diffusion
            horizon=config.horizon,
            hidden_dim=config.critic_hidden_dim
        )

        self.reset()

    def get_optim_params(self) -> dict:
        # Return parameters of the diffusion model for optimization
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            # Use a single key for stacked images in the queue
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations."""
        # Normalize inputs
        batch = self.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(batch)
            processed_batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            processed_batch = batch

        # Populate queues with the latest observation
        self._queues = populate_queues(self._queues, processed_batch)

        # Generate new action plan only when the action queue is empty
        if len(self._queues["action"]) == 0:
            # Prepare batch for the model by stacking history from queues
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    model_input_batch[key] = torch.stack(list(queue), dim=1)

            # Generate action sequence using the diffusion model

            current_state = model_input_batch["observation.state"][:, -1, :]
            num_samples = getattr(self.config, "num_inference_samples", 1)

            actions = self.diffusion.generate_actions_via_inverse_dynamics(
                model_input_batch,
                current_state,
                self.inv_dyn_model,
                num_samples=num_samples,
                critic_scorer=self.critic_scorer
            )
            # actions = self.diffusion.generate_actions(model_input_batch)

            # Unnormalize actions
            actions = self.unnormalize_outputs({"action": actions})["action"]

            # Add generated actions to the queue
            self._queues["action"].extend(actions.squeeze(0))

        # Pop the next action from the queue
        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """
        Run the batch through the model and compute the loss.
        Always returns a Tensor (possibly zero) so train.py never sees None.
        """
        # handy device and dummy‐loss creator
        device = next(self.parameters()).device

        def _dummy():
            return torch.tensor(0.0, device=device, requires_grad=True), None

        # 1) normalize all inputs and build the model_batch
        normalized = self.normalize_inputs(batch)
        model_batch: dict[str, Tensor] = {}
        for key in self.config.input_features:
            if key not in normalized:
                continue
            x = normalized[key]
            if key.startswith("observation."):
                # ensure we only take the first n_obs_steps
                if x.ndim > 1 and x.shape[1] < self.config.n_obs_steps:
                    print(
                        f"Warning: Input '{key}' has {x.shape[1]} time steps < {self.config.n_obs_steps}. Skipping batch."
                    )
                    return _dummy()
                # slice time dimension if it’s longer than needed
                model_batch[key] = x if x.ndim == 1 else x[:,
                                                           : self.config.n_obs_steps]
            else:
                model_batch[key] = x

        # 2) image‑stacking checks …
        if self.config.image_features:
            # Stack images if needed (using potentially normalized images)
            image_keys_present = [
                k for k in self.config.image_features if k in model_batch]
            if image_keys_present:
                # Stack only the observation steps
                # Ensure the image tensor also has enough time steps
                min_img_len = min(model_batch[key].shape[1]
                                  for key in image_keys_present)
                if min_img_len < self.config.n_obs_steps:
                    print(
                        f"Warning: Image data has insufficient time steps ({min_img_len} < {self.config.n_obs_steps}). Skipping batch.")
                    return _dummy()
                model_batch["observation.images"] = torch.stack(
                    [model_batch[key][:, :self.config.n_obs_steps] for key in image_keys_present], dim=-4
                )

        # 3) target checks …
        target_key = self.config.diffusion_target_key
        if target_key in batch:
            required_len = self.config.n_obs_steps + self.config.horizon
            if batch[target_key].ndim <= 1 or batch[target_key].shape[1] < required_len:
                print(
                    f"Warning: Target '{target_key}' has shape {batch[target_key].shape}, need at least length {required_len}. Skipping batch.")
                return _dummy()  # Skip batch

            if self.config.predict_state:
                # Target is future states: slice the original observation.state tensor
                target_data_slice = batch[target_key][:,
                                                      self.config.n_obs_steps: required_len]
            else:
                # Target is action
                # Check if action length matches horizon if predicting actions
                if batch[target_key].shape[1] < self.config.horizon:
                    print(
                        f"Warning: Action data length ({batch[target_key].shape[1]}) is less than horizon ({self.config.horizon}). Skipping batch.")
                    return _dummy()  # Skip if action length is insufficient
                # Assuming action target matches horizon length
                target_data_slice = batch[target_key][:, :self.config.horizon]

            # Normalize the target slice
            target_data_to_norm = {target_key: target_data_slice}
            normalized_target = self.normalize_targets(target_data_to_norm)
            model_batch[f"normalized_{target_key}"] = normalized_target[target_key]
        else:
            print(
                f"Warning: Target key '{target_key}' not found in batch. Skipping batch.")
            return _dummy()  # Skip if target key is missing

        # 4) compute real loss
        loss = self.diffusion.compute_loss(model_batch)
        return loss, None


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

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.state_dim = config.robot_state_feature.shape[0]  # Store state dim
        self.action_dim = config.action_feature.shape[0]  # Store action dim

        # Determine the dimension the diffusion model should predict
        self.diffusion_target_dim = self.state_dim if config.predict_state else self.action_dim

        # --- Observation Encoders ---
        global_cond_dim = 0

        # Robot state dimension
        if config.robot_state_feature:
            global_cond_dim += config.robot_state_feature.shape[0] * \
                config.n_obs_steps
        else:
            raise ValueError(
                "`observation.state` (robot_state_feature) is required.")

        # Image features
        self.rgb_encoder = None
        if config.image_features:
            num_cameras = len(config.image_features)
            image_feature_dim = config.transformer_dim
            if config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList([
                    DiffusionRgbEncoder(config) for _ in range(num_cameras)
                ])
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += image_feature_dim * num_cameras * config.n_obs_steps

        # Environment state features
        self.env_state_encoder = None
        if config.env_state_feature:
            env_state_dim = config.env_state_feature.shape[0]
            global_cond_dim += env_state_dim * config.n_obs_steps

        # --- Diffusion Transformer ---
        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim,
            output_dim=self.diffusion_target_dim  # Pass the correct target dimension
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
        B, T_obs, N_cam = images.shape[:3]

        if self.config.use_separate_rgb_encoder_per_camera:
            images_reshaped = einops.rearrange(
                images, "b t n c h w -> n (b t) c h w")
            features_list = []
            for i in range(N_cam):
                features_list.append(self.rgb_encoder[i](images_reshaped[i]))
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

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = [batch[OBS_ROBOT]]
        # Extract image features.
        if self.config.image_features:
            if self.config.use_separate_rgb_encoder_per_camera:
                # Combine batch and sequence dims while rearranging to make the camera index dimension first.
                images_per_camera = einops.rearrange(
                    batch["observation.images"], "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [
                        encoder(images)
                        for encoder, images in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                # Separate batch and sequence dims back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                # Combine batch, sequence, and "which camera" dims before passing to shared encoder.
                img_features = self.rgb_encoder(
                    einops.rearrange(
                        batch["observation.images"], "b s n ... -> (b s n) ...")
                )
                # Separate batch dim and sequence dim back out. The camera index dim gets absorbed into the
                # feature dim (effectively concatenating the camera features).
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            global_cond_feats.append(batch[OBS_ENV])

        # Concatenate features then flatten to (B, global_cond_dim).
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

    # ========= inference  ============
    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Sample prior.
        sample = torch.randn(
            size=(batch_size, self.config.horizon,
                  self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Predict model output.
            model_output = self.unet(
                sample,
                torch.full(sample.shape[:1], t,
                           dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample

        return sample

    @torch.no_grad()
    def generate_actions_via_inverse_dynamics(
        self,
        batch: dict[str, Tensor],
        current_state: Tensor,  # Add current_state (B, state_dim)
        inv_dyn_model: MlpInvDynamic,  # Add inverse dynamics model
        num_samples: int = 1,
        critic_scorer: Optional[CriticScorer] = None
    ) -> Tensor:
        """
        Generates future states, infers actions using inverse dynamics,
        and selects the best action sequence using a critic.
        """
        batch_size = current_state.shape[0]
        n_obs_steps = batch["observation.state"].shape[1]
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # 1. Generate multiple future STATE samples:
        predicted_states = self.conditional_sample(
            batch_size, global_cond=global_cond, num_samples=num_samples
        )  # (B, num_samples, horizon, state_dim)

        # 2. (Optional) pick best state‐trajectory via critic
        if num_samples > 1 and critic_scorer is not None:
            best_states = []
            for i in range(batch_size):
                # score each (horizon, state_dim) sample
                scores = critic_scorer.score(
                    predicted_states[i])  # (num_samples,)
                best_idx = torch.argmax(scores).item()
                best_states.append(predicted_states[i, best_idx])
            # selected_states: (B, horizon, state_dim)
            selected_states = torch.stack(best_states, dim=0)
        else:
            # just take the first sample
            selected_states = predicted_states[:, 0]

        # 3. Infer actions *only* for the selected trajectory
        # Prepare pairs (s_t, s_{t+1}) across the horizon
        # current_state: (B, state_dim) -> (B,1,state_dim)
        s_t0 = current_state.unsqueeze(1)
        # for t=0..h-2 use selected_states[:,t], last pair uses selected_states[:,h-1]
        # (B,horizon,state_dim)
        s_t_pairs = torch.cat([s_t0, selected_states[:, :-1]], dim=1)
        # (B,horizon,state_dim)
        s_tp1_pairs = selected_states

        # flatten to (B*horizon, state_dim)
        B, H, D = s_t_pairs.shape
        s_t_flat = s_t_pairs.reshape(B * H, D)
        s_tp1_flat = s_tp1_pairs.reshape(B * H, D)

        # Predict actions: (B*horizon, action_dim)
        actions_flat = inv_dyn_model.predict(s_t_flat, s_tp1_flat)

        # reshape back to (B, horizon, action_dim)
        final_actions_horizon = actions_flat.view(B, H, -1)

        # 4. extract first n_action_steps
        actions_to_execute = final_actions_horizon[:,
                                                   : self.config.n_action_steps]

        return actions_to_execute

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """

        assert set(batch).issuperset(
            {"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Prepare global conditioning using ONLY observation steps
        global_cond = self._prepare_global_conditioning(batch)

        # Forward diffusion.
        clean_targets = batch["observation.state"]
        B = clean_targets.shape[0]
        device = clean_targets.device

        noise = torch.randn(clean_targets.shape, device=device)

        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,), device=device,
        ).long()

        # Add noise to clean targets
        noisy_targets = self.noise_scheduler.add_noise(
            clean_targets, noise, timesteps)

        # Predict noise or clean target
        pred = self.transformer(noisy_targets, timesteps,
                                global_cond=global_cond)

        # Determine target for loss
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = clean_targets
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        # Compute MSE loss
        # (B, horizon, target_dim)
        loss = F.mse_loss(pred, target, reduction="none")

        # Mask loss wherever the action is padded with copies (edges of the dataset trajectory).
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError(
                    "You need to provide 'action_is_pad' in the batch when "
                    f"{self.config.do_mask_loss_for_padding=}."
                )
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()
