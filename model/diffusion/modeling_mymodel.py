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
            config.output_features, config.normalization_mapping, dataset_stats
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
            actions = self.diffusion.generate_actions(model_input_batch)

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
        batch = self.normalize_targets(batch)

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
            config, global_cond_dim=global_cond_dim
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
        num_samples: int = 1,  # Add num_samples parameter
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Samples actions using the DDPM/DDIM reverse process.
           Generates `num_samples` candidates per batch item.
        """
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        # Expand global condition to match num_samples
        # (B, D_cond) -> (B * num_samples, D_cond)
        expanded_global_cond = global_cond.repeat_interleave(
            num_samples, dim=0)
        effective_batch_size = batch_size * num_samples

        # Sample initial noise for all samples
        noisy_actions = torch.randn(
            size=(effective_batch_size, self.config.horizon,
                  self.config.action_feature.shape[0]),
            dtype=dtype,
            device=device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            # Prepare timestep tensor for the expanded batch
            timesteps_batch = torch.full(
                (effective_batch_size,), t, dtype=torch.long, device=device)

            # Predict model output for the expanded batch
            model_output = self.transformer(
                noisy_actions,
                timesteps_batch,
                global_cond=expanded_global_cond,  # Use expanded condition
            )

            # Scheduler step
            scheduler_output = self.noise_scheduler.step(
                model_output, t, noisy_actions, generator=generator
            )
            noisy_actions = scheduler_output.prev_sample

        # Reshape the output to group samples per batch item
        # (B * num_samples, horizon, action_dim) -> (B, num_samples, horizon, action_dim)
        actions = noisy_actions.view(
            batch_size, num_samples, self.config.horizon, self.config.action_feature.shape[0]
        )

        return actions  # Return samples grouped by batch item

    @torch.no_grad()
    def generate_actions(
        self,
        batch: dict[str, Tensor],
        num_samples: int = 1,  # Add num_samples
        critic_scorer: Optional[CriticScorer] = None  # Add critic scorer
    ) -> Tensor:
        """
        Generates a sequence of actions based on the input observations.
        Optionally generates multiple samples and selects the best using a critic.
        """
        batch_size = batch["observation.state"].shape[0]
        n_obs_steps = batch["observation.state"].shape[1]
        assert n_obs_steps == self.config.n_obs_steps, \
            f"Expected {self.config.n_obs_steps} obs steps, got {n_obs_steps}"

        global_cond = self._prepare_global_conditioning(
            batch)  # (B, global_cond_dim)

        # Generate multiple action samples: (B, num_samples, horizon, action_dim)
        action_samples = self.conditional_sample(
            batch_size, global_cond=global_cond, num_samples=num_samples
        )

        # Select best action using critic if provided and num_samples > 1
        if num_samples > 1 and critic_scorer is not None:
            best_actions = []
            for i in range(batch_size):
                # Get samples for the current batch item: (num_samples, horizon, action_dim)
                current_samples = action_samples[i]
                # Score the samples using the critic
                # Assuming critic scores action sequences directly
                scores = critic_scorer.score(current_samples)  # (num_samples,)
                # Find the index of the best score
                best_idx = torch.argmax(scores)
                # Select the best action sequence
                best_actions.append(current_samples[best_idx])
            # Stack best actions for the batch: (B, horizon, action_dim)
            actions = torch.stack(best_actions, dim=0)
        else:
            # If only one sample or no critic, just take the first (or only) sample
            actions = action_samples[:, 0]  # (B, horizon, action_dim)

        # Extract the required `n_action_steps` for execution
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        # (B, n_action_steps, action_dim)
        actions_to_execute = actions[:, start:end]

        return actions_to_execute

    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Computes the diffusion loss for training.
        """
        required_keys = {"observation.state", "action"}
        if self.config.do_mask_loss_for_padding:
            required_keys.add("action_is_pad")
        assert set(batch).issuperset(required_keys)
        assert self.config.image_features or self.config.env_state_feature

        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        global_cond = self._prepare_global_conditioning(batch)

        clean_actions = batch["action"]
        B = clean_actions.shape[0]
        device = clean_actions.device

        noise = torch.randn(clean_actions.shape, device=device)

        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(B,),
            device=device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(
            clean_actions, noise, timesteps)

        pred = self.transformer(noisy_actions, timesteps,
                                global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "sample":
            target = clean_actions
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding:
            in_episode_mask = ~batch["action_is_pad"]
            loss = loss * in_episode_mask.unsqueeze(-1)
            loss = loss.sum() / in_episode_mask.sum().clamp(min=1.0)
        else:
            loss = loss.mean()

        return loss
