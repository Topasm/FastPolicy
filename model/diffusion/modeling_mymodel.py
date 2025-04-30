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
            actions = self.unnormalize_action_output(
                {"action": actions})["action"]

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

        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            # shallow copy so that adding a key doesn't modify the original
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        batch = self.normalize_diffusion_target(batch)
        loss = self.diffusion.compute_loss(batch)
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
        """Encode image features and concatenate them all together along with the state vector."""
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
        batch: dict[str, Tensor],
        current_state: Tensor,
        inv_dyn_model: MlpInvDynamic,
        num_samples: int = 1,
        critic_scorer: Optional[CriticScorer] = None
    ) -> Tensor:
        batch_size = current_state.shape[0]
        n_obs_steps = self.config.n_obs_steps
        assert batch["observation.state"].shape[1] == n_obs_steps
        if self.config.image_features:
            assert batch["observation.images"].shape[1] == n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        predicted_states_flat = self.conditional_sample(
            batch_size * num_samples, global_cond=global_cond.repeat_interleave(num_samples, dim=0)
        )

        predicted_states = einops.rearrange(
            predicted_states_flat, "(b n) h d -> b n h d", b=batch_size, n=num_samples
        )

        if num_samples > 1 and critic_scorer is not None:
            best_states = []
            for i in range(batch_size):
                scores = critic_scorer.score(
                    predicted_states[i])
                best_idx = torch.argmax(scores).item()
                best_states.append(predicted_states[i, best_idx])
            selected_states = torch.stack(best_states, dim=0)
        else:
            selected_states = predicted_states[:, 0]

        s_t0 = current_state.unsqueeze(1)
        s_t_pairs = torch.cat([s_t0, selected_states[:, :-1]], dim=1)
        s_tp1_pairs = selected_states

        B, H, D_state = s_t_pairs.shape
        s_t_flat = s_t_pairs.reshape(B * H, D_state)
        s_tp1_flat = s_tp1_pairs.reshape(B * H, D_state)

        actions_flat = inv_dyn_model.predict(s_t_flat, s_tp1_flat)

        final_actions_horizon = actions_flat.view(B, H, -1)

        actions_to_execute = final_actions_horizon[:,
                                                   : self.config.n_action_steps]

        return actions_to_execute

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        n_obs_steps = self.config.n_obs_steps
        horizon = self.config.horizon
        state_seq_len = batch["observation.state"].shape[1]

        # The state sequence must contain n_obs_steps for conditioning
        # PLUS horizon steps for the target prediction.
        # Corrected minimum length calculation
        expected_min_len = n_obs_steps + horizon
        if state_seq_len < expected_min_len:
            raise ValueError(
                f"observation.state sequence length ({state_seq_len}) is too short. "
                # Corrected error message to reflect the required length
                f"Need at least {expected_min_len} steps ({n_obs_steps} obs steps + {horizon} prediction horizon)."
            )
        if self.config.image_features:
            assert batch["observation.images"].shape[1] == n_obs_steps, \
                f"Image sequence length ({batch['observation.images'].shape[1]}) must match n_obs_steps ({n_obs_steps})"
        assert "action_is_pad" in batch, "Need 'action_is_pad' to mask loss for state prediction horizon."
        assert batch["action_is_pad"].shape[1] == horizon, \
            f"action_is_pad length ({batch['action_is_pad'].shape[1]}) must match prediction horizon ({horizon})"

        cond_batch = {
            "observation.state": batch["observation.state"][:, :n_obs_steps],
        }
        if self.config.image_features:
            cond_batch["observation.images"] = batch["observation.images"]
        if self.config.env_state_feature:
            # Ensure env state also has n_obs_steps length if provided
            if batch[OBS_ENV].shape[1] < n_obs_steps:
                raise ValueError(
                    f"{OBS_ENV} sequence length ({batch[OBS_ENV].shape[1]}) "
                    f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
                )
            cond_batch[OBS_ENV] = batch[OBS_ENV][:, :n_obs_steps]

        global_cond = self._prepare_global_conditioning(cond_batch)

        # Extract target states: sequence of length `horizon` starting after `n_obs_steps`
        clean_targets = batch["observation.state"][:,
                                                   n_obs_steps: n_obs_steps + horizon, :]  # Shape: (B, horizon, state_dim)

        # Ensure clean_targets has the expected horizon length after slicing
        if clean_targets.shape[1] != horizon:
            raise ValueError(
                f"Sliced clean_targets sequence length ({clean_targets.shape[1]}) does not match horizon ({horizon}). "
                f"Check original state_seq_len ({state_seq_len}) and slicing logic."
            )

        B = clean_targets.shape[0]
        device = clean_targets.device

        noise = torch.randn(clean_targets.shape, device=device)

        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,), device=device,
        ).long()

        noisy_targets = self.noise_scheduler.add_noise(
            clean_targets, noise, timesteps)  # Shape: (B, horizon, state_dim)

        # Pass noisy targets (shape B, horizon, state_dim) to transformer
        pred = self.transformer(noisy_targets, timesteps,
                                global_cond=global_cond)  # Output: (B, horizon, state_dim)

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = clean_targets
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        # Shape: (B, horizon, state_dim)
        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            padding_mask = batch["action_is_pad"]  # Shape: (B, horizon)
            in_episode_bound = ~padding_mask
            loss = loss * in_episode_bound.unsqueeze(-1)  # Apply mask

        return loss.mean()

    def conditional_sample(
        self, batch_size: int, global_cond: Tensor | None = None, generator: torch.Generator | None = None
    ) -> Tensor:
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

        return sample
