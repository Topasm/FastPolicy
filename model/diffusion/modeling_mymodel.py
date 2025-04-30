import math
from collections import deque
from typing import Callable, Optional

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

        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            # Use the dynamic target key
            {self.config.diffusion_target_key:
                self.config.output_features[self.config.diffusion_target_key]},
            config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # queues are populated during rollout of the policy
        self._queues = None

        # Instantiate the diffusion model (now using DiT)
        self.diffusion = MyDiffusionModel(config)

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
        """Run the batch through the model and compute the loss for training or validation."""
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )

        # Normalize the specific target data using the dynamic key
        target_key = self.config.diffusion_target_key
        target_data = {target_key: batch[target_key]}
        target_data = self.normalize_targets(target_data)
        batch[f"normalized_{target_key}"] = target_data[target_key]

        # Compute loss using the diffusion model
        loss = self.diffusion.compute_loss(batch)
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
        """Encodes observations and concatenates them into a flat global conditioning vector."""
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        global_cond_feats = []

        robot_state = batch[OBS_ROBOT].flatten(start_dim=1)
        global_cond_feats.append(robot_state)

        if self.config.image_features:
            img_features = self._encode_images(batch["observation.images"])
            global_cond_feats.append(img_features.flatten(start_dim=1))

        if self.config.env_state_feature:
            env_state = batch[OBS_ENV]
            global_cond_feats.append(env_state.flatten(start_dim=1))

        return torch.cat(global_cond_feats, dim=-1)

    @torch.no_grad()
    def conditional_sample(
        self,
        batch_size: int,
        global_cond: Tensor,
        num_samples: int = 1,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Samples future sequences (states or actions) using the DDPM/DDIM reverse process."""
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        expanded_global_cond = global_cond.repeat_interleave(
            num_samples, dim=0)
        effective_batch_size = batch_size * num_samples

        # Sample initial noise for future TARGETS (states or actions)
        noisy_targets = torch.randn(
            size=(effective_batch_size, self.config.horizon,
                  self.diffusion_target_dim),  # Use the target dimension
            dtype=dtype, device=device, generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            timesteps_batch = torch.full(
                (effective_batch_size,), t, dtype=torch.long, device=device)

            # Predict model output (noise or target sample)
            model_output = self.transformer(
                noisy_targets,  # Input noisy targets
                timesteps_batch,
                global_cond=expanded_global_cond,
            )

            scheduler_output = self.noise_scheduler.step(
                model_output, t, noisy_targets, generator=generator
            )
            noisy_targets = scheduler_output.prev_sample

        # Reshape the output targets
        # (B * num_samples, horizon, target_dim) -> (B, num_samples, horizon, target_dim)
        predicted_targets = noisy_targets.view(
            batch_size, num_samples, self.config.horizon, self.diffusion_target_dim
        )
        return predicted_targets

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
        # Shape: (B, num_samples, horizon, state_dim)
        predicted_states = self.conditional_sample(
            batch_size, global_cond=global_cond, num_samples=num_samples
        )

        # 2. Infer action sequences using Inverse Dynamics Model
        # Prepare state pairs for inverse dynamics model
        # current_state: (B, state_dim) -> (B, num_samples, 1, state_dim)
        s_t = current_state.unsqueeze(1).unsqueeze(
            1).expand(-1, num_samples, 1, -1)
        # predicted_states: (B, num_samples, horizon, state_dim)
        # Concatenate current state at the beginning for pairs:
        # Shape: (B, num_samples, horizon+1, state_dim)
        all_states = torch.cat([s_t, predicted_states], dim=2)

        # Get pairs (s_t, s_{t+1})
        # s_t_pairs shape: (B, num_samples, horizon, state_dim)
        s_t_pairs = all_states[:, :, :-1, :]
        # s_tplus1_pairs shape: (B, num_samples, horizon, state_dim)
        s_tplus1_pairs = all_states[:, :, 1:, :]

        # Reshape for batch processing by inv_dyn_model:
        # Shape: (B * num_samples * horizon, state_dim)
        s_t_flat = einops.rearrange(s_t_pairs, 'b ns h d -> (b ns h) d')
        s_tplus1_flat = einops.rearrange(
            s_tplus1_pairs, 'b ns h d -> (b ns h) d')

        # Predict actions: Shape: (B * num_samples * horizon, action_dim)
        inferred_actions_flat = inv_dyn_model.predict(s_t_flat, s_tplus1_flat)

        # Reshape back: (B, num_samples, horizon, action_dim)
        inferred_actions = einops.rearrange(
            inferred_actions_flat, '(b ns h) d -> b ns h d',
            b=batch_size, ns=num_samples, h=self.config.horizon
        )

        # 3. Select best action sequence using Critic (if available)
        if num_samples > 1 and critic_scorer is not None:
            best_actions_horizon = []
            for i in range(batch_size):
                # Get inferred action samples for the current batch item
                # Shape: (num_samples, horizon, action_dim)
                current_action_samples = inferred_actions[i]
                # Score the action samples
                scores = critic_scorer.score(
                    current_action_samples)  # (num_samples,)
                best_idx = torch.argmax(scores)
                best_actions_horizon.append(current_action_samples[best_idx])
            # Stack best actions: (B, horizon, action_dim)
            final_actions_horizon = torch.stack(best_actions_horizon, dim=0)
        else:
            # If only one sample or no critic, take the first sample
            # (B, horizon, action_dim)
            final_actions_horizon = inferred_actions[:, 0]

        # 4. Extract the required `n_action_steps` for execution
        # We need the first n_action_steps from the inferred sequence
        actions_to_execute = final_actions_horizon[:,
                                                   :self.config.n_action_steps]

        return actions_to_execute  # (B, n_action_steps, action_dim)

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """Computes the diffusion loss for the configured target (state or action)."""
        target_key = self.config.diffusion_target_key
        normalized_target_key = f"normalized_{target_key}"

        required_keys = {"observation.state", normalized_target_key}
        # Add padding mask key if needed
        # if self.config.do_mask_loss_for_padding:
        #     required_keys.add("target_is_pad") # Example key
        assert set(batch).issuperset(required_keys)
        assert self.config.image_features or self.config.env_state_feature

        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch[normalized_target_key].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        # Target is the sequence of clean future states or actions
        # (B, horizon, target_dim)
        clean_targets = batch[normalized_target_key]
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

        # Mask loss for padding if needed (adapt mask key/logic)
        # if self.config.do_mask_loss_for_padding:
        #     in_episode_mask = ~batch["target_is_pad"] # Example mask
        #     loss = loss * in_episode_mask.unsqueeze(-1)
        #     loss = loss.sum() / in_episode_mask.sum().clamp(min=1.0)
        # else:
        loss = loss.mean()  # Simple mean loss for now

        return loss
