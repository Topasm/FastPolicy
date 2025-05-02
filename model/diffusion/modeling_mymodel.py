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
from model.invdynamics.invdyn import MlpInvDynamic, SeqInvDynamic, TemporalUNetInvDynamic


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
        # Switch back to MlpInvDynamic
        self.inv_dyn_model = MlpInvDynamic(
            o_dim=self.state_dim * config.horizon,  # MLP expects flattened state sequence
            # MLP predicts flattened action sequence
            a_dim=self.action_dim * config.horizon,
            hidden_dim=self.config.inv_dyn_hidden_dim,
            dropout=0.1,  # Example dropout, adjust if needed
            use_layernorm=True,  # Example, adjust if needed
            out_activation=nn.Tanh(),  # Keep Tanh for normalized actions
        )
        self.critic_scorer = CriticScorer(
            state_dim=self.state_dim,
            # Critic likely scores state from diffusion
            horizon=config.horizon,
            hidden_dim=config.critic_hidden_dim
        )

        # Determine and store the device
        self.diffusion.to(config.device)
        self.inv_dyn_model.to(config.device)
        self.critic_scorer.to(config.device)
        self.device = get_device_from_parameters(self.diffusion)

        self.reset()

    def get_optim_params(self) -> list:
        # Return parameters of both models for joint optimization
        # Ensure parameters from all relevant submodules are included
        return list(self.diffusion.parameters()) + list(self.inv_dyn_model.parameters()) + list(self.critic_scorer.parameters())

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
        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs
        norm_batch = self.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(norm_batch)
            processed_batch["observation.images"] = torch.stack(
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
            current_state = model_input_batch["observation.state"][:,
                                                                   0, :]
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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, None]:
        """
        Run the batch through the model and compute the loss.
        Always returns a Tensor (possibly zero) so train.py never sees None.
        """
        raw_batch = batch  # Keep original batch for inv dyn normalization

        # # --- Debug Print Raw Data Range Start ---
        # target_key = self.config.diffusion_target_key
        # if target_key in raw_batch:
        #     raw_targets = raw_batch[target_key][:, 0:self.config.horizon, :]
        #     if os.environ.get("LOCAL_RANK", "0") == "0":  # Basic check for multi-gpu
        #         if raw_targets.shape[-1] > 1:
        #             raw_y_min = torch.min(raw_targets[..., 1])
        #             raw_y_max = torch.max(raw_targets[..., 1])
        #             if raw_y_min is not None and raw_y_max is not None:
        #                 print(
        #                     f"DEBUG forward: Raw target y-axis range: [{raw_y_min.item():.4f}, {raw_y_max.item():.4f}]")
        #         else:
        #             raw_s_min = torch.min(raw_targets)
        #             raw_s_max = torch.max(raw_targets)
        #             if raw_s_min is not None and raw_s_max is not None:
        #                 print(
        #                     f"DEBUG forward: Raw target range (dim=1): [{raw_s_min.item():.4f}, {raw_s_max.item():.4f}]")
        # else:
        #     if os.environ.get("LOCAL_RANK", "0") == "0":
        #         print(
        #             f"DEBUG forward: Target key '{target_key}' not found in raw_batch.")
        # # --- Debug Print Raw Data Range End ---

        # 1. Normalize inputs
        norm_batch = self.normalize_inputs(batch)

        # 2. Stack images if necessary (using normalized batch)
        if self.config.image_features:
            norm_batch = dict(norm_batch)  # shallow copy
            norm_batch["observation.images"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )

        # 4. Separately normalize state and action for inverse dynamics (using raw batch)
        invdyn_batch = self.normalize_invdyn_state(raw_batch)
        invdyn_batch = self.normalize_invdyn_action(invdyn_batch)
        # Ensure invdyn_batch has necessary keys if they were modified/missing in raw_batch processing
        # This assumes raw_batch contains 'observation.state' and 'action' with correct sequence lengths
        # Copy padding mask
        invdyn_batch['action_is_pad'] = raw_batch['action_is_pad']

        # 5. Compute loss using appropriately normalized batches
        loss = self.diffusion.compute_loss(
            diffusion_batch=norm_batch,
            invdyn_batch=invdyn_batch,
            inv_dyn_model=self.inv_dyn_model
        )
        # no output_dict so returning None
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
        """Encode image features and concatenate them all together along with the state vector.
           Expects batch to have normalized 'observation.state' and potentially 'observation.images'.
        """
        batch_size = batch["observation.state"].shape[0]
        n_obs_steps = self.config.n_obs_steps

        if batch["observation.state"].shape[1] < n_obs_steps:
            raise ValueError(
                f"observation.state sequence length ({batch['observation.state'].shape[1]}) "
                f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
            )
        cond_state = batch["observation.state"][:, :n_obs_steps, :]
        global_cond_feats = [cond_state]

        if self.config.image_features:
            images = batch["observation.images"]
            _B, n_img_steps, n_cam = images.shape[:3]

            if n_img_steps != n_obs_steps:
                raise ValueError(
                    f"Image sequence length ({n_img_steps}) in batch does not match "
                    f"configured n_obs_steps ({n_obs_steps}). Check dataset delta_timestamps "
                    f"and policy config."
                )
            assert _B == batch_size

            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    images, "b s n ... -> n (b s) ...", s=n_obs_steps)
                img_features_list = torch.cat(
                    [
                        encoder(imgs)
                        for encoder, imgs in zip(self.rgb_encoder, images_per_camera, strict=True)
                    ]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                images_reshaped = einops.rearrange(
                    images, "b s n ... -> (b s n) ...", s=n_obs_steps
                )
                img_features = self.rgb_encoder(images_reshaped)
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps, n=n_cam
                )
            global_cond_feats.append(img_features)

        if self.config.env_state_feature:
            if batch[OBS_ENV].shape[1] < n_obs_steps:
                raise ValueError(
                    f"{OBS_ENV} sequence length ({batch[OBS_ENV].shape[1]}) "
                    f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
                )
            cond_env_state = batch[OBS_ENV][:, :n_obs_steps, :]
            global_cond_feats.append(cond_env_state)

        concatenated_features = torch.cat(global_cond_feats, dim=-1)
        global_cond = concatenated_features.flatten(start_dim=1)
        return global_cond

    @torch.no_grad()
    def generate_actions_via_inverse_dynamics(
        self,
        norm_batch: dict[str, Tensor],  # Expects normalized batch
        norm_current_state: Tensor,    # Expects normalized state
        inv_dyn_model: MlpInvDynamic,  # Updated type hint
        num_samples: int = 1,
    ) -> Tensor:
        """Generates actions using diffusion for states and then inverse dynamics.
           Expects inputs (batch, current_state) to be already normalized.
           Returns normalized actions.
        """
        batch_size = norm_current_state.shape[0]
        n_obs_steps = self.config.n_obs_steps
        assert norm_batch["observation.state"].shape[1] == n_obs_steps
        if self.config.image_features:
            assert norm_batch["observation.images"].shape[1] == n_obs_steps

        # Prepare conditioning from the normalized batch
        global_cond = self._prepare_global_conditioning(norm_batch)

        # Repeat global_cond if num_samples > 1
        if num_samples > 1:
            global_cond = global_cond.repeat_interleave(num_samples, dim=0)

        # Sample normalized future states
        predicted_states_flat = self.conditional_sample(
            batch_size * num_samples, global_cond=global_cond
        )  # Output is normalized states

        predicted_states = einops.rearrange(
            predicted_states_flat, "(b n) h d -> b n h d", b=batch_size, n=num_samples
        )

        # Always select the first sample if multiple are generated
        # (B, H, D_state) - Normalized
        selected_states = predicted_states[:, 0]

        # Prepare state pairs for inverse dynamics model (using normalized states)
        s_t0 = norm_current_state.unsqueeze(1)  # (B, 1, D_state) - Normalized
        # Concatenate current state with the first H-1 predicted states
        s_t_pairs = torch.cat(
            [s_t0, selected_states[:, :-1]], dim=1)  # (B, H, D_state) - Normalized

        # Flatten the state sequence for MLP input
        B, H, D_state = s_t_pairs.shape
        s_t_pairs_flat = s_t_pairs.view(B, -1)  # (B, H * D_state)

        # Predict actions for the sequence of *normalized* state pairs
        pred_actions_flat = inv_dyn_model.predict(
            s_t_pairs_flat)  # (B, H * action_dim) - Normalized actions

        # Reshape predicted actions back to sequence format
        pred_actions = pred_actions_flat.view(
            B, H, self.action_dim)  # (B, H, action_dim)

        # --- Apply the slicing logic from the provided example ---
        # Select actions using start = n_obs_steps - 1
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        # Ensure the slice indices are within the bounds of the predicted horizon (H)
        if end > H:
            print(
                f"Warning: Slicing end index ({end}) exceeds prediction horizon ({H}). Adjusting end index.")
            end = H
        if start >= H:
            raise ValueError(
                f"Slicing start index ({start}) is out of bounds for prediction horizon ({H}). Check n_obs_steps and horizon.")

        # (B, n_action_steps_effective, action_dim)
        actions_to_execute = pred_actions[:, start:end]
        # --- End of applied slicing logic ---

        # Pad if the slice is shorter than n_action_steps (e.g., if end was adjusted)
        actual_steps = actions_to_execute.shape[1]
        if actual_steps < self.config.n_action_steps:
            print(
                f"Warning: Only {actual_steps} actions could be sliced. Padding the rest.")
            padding_needed = self.config.n_action_steps - actual_steps
            # Pad with the last valid action
            # Keep dimension B, 1, D
            last_action = actions_to_execute[:, -1:, :]
            padding = last_action.repeat(1, padding_needed, 1)
            actions_to_execute = torch.cat(
                [actions_to_execute, padding], dim=1)

        return actions_to_execute  # Return normalized actions

    def compute_loss(self, diffusion_batch: dict[str, Tensor], invdyn_batch: dict[str, Tensor], inv_dyn_model: MlpInvDynamic) -> Tensor:
        """
        Computes the diffusion loss and the inverse dynamics loss.
        Expects diffusion_batch to be normalized for diffusion inputs and targets.
        Expects invdyn_batch to be normalized for inverse dynamics inputs (state) and targets (action).
        """
        n_obs_steps = self.config.n_obs_steps
        horizon = self.config.horizon

        # --- Diffusion Loss Part (uses diffusion_batch) ---
        if self.config.image_features:
            assert diffusion_batch["observation.images"].shape[1] == n_obs_steps, \
                f"Image sequence length ({diffusion_batch['observation.images'].shape[1]}) must match n_obs_steps ({n_obs_steps})"
        # Padding mask should be consistent across batches, get from invdyn_batch as it was copied from raw
        assert "action_is_pad" in invdyn_batch, "Need 'action_is_pad' to mask loss."
        assert invdyn_batch["action_is_pad"].shape[1] == horizon, \
            f"action_is_pad length ({invdyn_batch['action_is_pad'].shape[1]}) must match prediction horizon ({horizon})"

        # Prepare global conditioning using diffusion_batch (already normalized inputs)
        cond_batch = {
            "observation.state": diffusion_batch["observation.state"][:, :n_obs_steps],
        }
        if self.config.image_features:
            cond_batch["observation.images"] = diffusion_batch["observation.images"]
        if self.config.env_state_feature:
            # Assuming env_state is part of diffusion_batch if needed
            if diffusion_batch[OBS_ENV].shape[1] < n_obs_steps:
                raise ValueError(f"{OBS_ENV} sequence length issue.")
            cond_batch[OBS_ENV] = diffusion_batch[OBS_ENV][:, :n_obs_steps]

        global_cond = self._prepare_global_conditioning(cond_batch)

        # Extract target states from diffusion_batch (already normalized target)
        # Shape: (B, horizon, state_dim)
        # Use the specific target key
        clean_targets = diffusion_batch["observation.state"][:, 0:horizon, :]

        B = clean_targets.shape[0]
        device = clean_targets.device

        # Diffusion loss calculation (remains the same logic)
        noise = torch.randn(clean_targets.shape, device=device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,), device=device,
        ).long()
        noisy_targets = self.noise_scheduler.add_noise(
            clean_targets, noise, timesteps)
        pred = self.transformer(noisy_targets, timesteps,
                                global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = clean_targets
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        diffusion_loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            padding_mask = invdyn_batch["action_is_pad"]  # Shape: (B, horizon)
            in_episode_bound = ~padding_mask
            diffusion_loss = diffusion_loss * \
                in_episode_bound.unsqueeze(-1)

        diffusion_loss = diffusion_loss.mean()

        # --- Inverse Dynamics Loss Part (uses invdyn_batch) ---
        # Extract states normalized for inv dyn model
        s_t = invdyn_batch["observation.state"][
            :, :horizon, :
        ]                              # (B, H, D_state) - Normalized for InvDyn
        B, H, D_state = s_t.shape

        # Flatten the state sequence for MLP input
        s_t_flat = s_t.view(B, -1)  # (B, H * D_state)

        # Predict actions using inv dyn model with appropriately normalized states
        pred_actions_flat = inv_dyn_model(  # Use forward during training
            s_t_flat)  # (B, H * action_dim) - Normalized actions

        # Reshape predicted actions back to sequence format
        pred_actions = pred_actions_flat.view(
            B, H, self.action_dim)  # (B, H, action_dim)

        # Ground‐truth actions normalized for inv dyn model
        # (B, H, action_dim) - Normalized for InvDyn
        true_actions = invdyn_batch["action"][:, :horizon, :]

        # Inverse dynamics loss calculation (remains the same logic)
        if self.config.do_mask_loss_for_padding:
            mask = ~invdyn_batch["action_is_pad"]            # (B, H)
            loss_e = F.mse_loss(pred_actions, true_actions, reduction="none")
            loss_e = loss_e * mask.unsqueeze(-1)      # zero‐out padded
            inv_dyn_loss = loss_e.sum() / (mask.sum() * self.action_dim + 1e-8)
        else:
            inv_dyn_loss = F.mse_loss(pred_actions, true_actions)

        total_loss = diffusion_loss + self.config.inv_dyn_loss_weight * inv_dyn_loss

        return total_loss

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
