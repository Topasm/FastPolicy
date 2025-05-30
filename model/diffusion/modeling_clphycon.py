#!/usr/bin/env python


import math
from collections import deque
from typing import Callable, Optional  # Added Optional

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

# OBS_IMAGE was not used directly here
from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
# Ensure DiffusionTransformer and DenoisingTransformer are correctly imported
from model.diffusion.diffusion_modules import DiffusionTransformer
# Assuming this is your DenoisingTransformer
from model.diffusion.async_modules import DenoisingTransformer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    # populate_queues, # Not directly used in predict_action, but in original select_action
)


class CLDiffPhyConModel(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    Modified to accept plan conditioning.
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
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

        self._queues = None  # For its own select_action, not predict_action

        # Plan embedder for BidirectionalTransformer plan conditioning
        # The state_dim for the plan comes from robot_state_feature, assuming plan states match obs states
        # If BidirTransformer outputs a different state_dim for its plan, this needs to match.
        # For now, assume plan_states_normalized has elements of state_dim = config.robot_state_feature.shape[0]
        plan_input_dim = config.robot_state_feature.shape[0]
        # Check if config has a specific plan_state_dim, otherwise use robot_state_feature
        if hasattr(config, 'plan_state_dim_for_rt_diffusion') and config.plan_state_dim_for_rt_diffusion is not None:
            plan_input_dim = config.plan_state_dim_for_rt_diffusion

        # Embedding dimension for plan features
        plan_feature_dim = getattr(config, 'plan_feature_dim', 64)

        self.plan_embedder = nn.Sequential(
            # Input dim matches each state in the plan
            nn.Linear(plan_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, plan_feature_dim)
        )
        self.plan_feature_dim = plan_feature_dim  # Store for DiffusionModel

        # Initialize the diffusion model with the plan feature dimension
        self.diffusion = DiffusionModel(
            config, plan_feature_dim=self.plan_feature_dim
        )

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        # This reset is for the standalone policy's select_action,
        # BidirectionalRTDiffusionPolicy will have its own queues.
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            # For its own select_action
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:  # Check based on config.image_features being non-empty
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:  # Check based on config.env_state_feature being non-empty
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def predict_action(self,
                       obs_dict: dict[str, Tensor],
                       plan_states_normalized: Tensor,  # Plan from Bidirectional Transformer
                       previous_rt_diffusion_plan: Optional[Tensor] = None
                       ) -> Tensor:
        """
        Generate actions using the RT-Diffusion model, conditioned on observation history
        and the planned state trajectory from Bidirectional Transformer.

        Args:
            obs_dict: Dictionary containing normalized observation history tensors.
                      Expected keys: "observation.state" ([B, n_obs_steps, state_dim]),
                                     "observation.image" ([B, n_obs_steps, NumCameras, C,H,W]).
            plan_states_normalized: Normalized planned state path from Bidirectional Transformer
                                     Shape: [B, plan_horizon, plan_state_dim].
            previous_rt_diffusion_plan: Optional normalized action plan from the previous step
                                         for CL-DiffPhyCon style feedback. Shape: [B, action_horizon, action_dim].

        Returns:
            normalized_action_sequence: Predicted normalized action sequence.
                                         Shape: [B, action_horizon, action_dim].
        """
        batch_size = obs_dict["observation.state"].shape[0]
        device = get_device_from_parameters(self)

        # 1. Embed the plan_states_normalized
        # plan_states_normalized is [B, plan_horizon, plan_state_dim]
        # self.plan_embedder expects [N, plan_state_dim]
        # We can either process it step-wise or reshape. Reshaping is more efficient.
        original_shape = plan_states_normalized.shape
        reshaped_plan_states = plan_states_normalized.reshape(
            -1, original_shape[-1])
        embedded_plan_steps = self.plan_embedder(reshaped_plan_states)
        embedded_plan_sequence = embedded_plan_steps.reshape(
            original_shape[0], original_shape[1], -1)  # [B, plan_horizon, plan_feature_dim]

        # Create a fixed-size plan feature vector by averaging or taking the first step's embedding.
        # Averaging might be more robust.
        plan_features = embedded_plan_sequence.mean(
            dim=1)  # [B, plan_feature_dim]
        plan_features = plan_features.to(device)

        # 2. Prepare global conditioning from observation history + embedded plan
        # obs_dict should already contain features like OBS_ROBOT, "observation.images"
        # These are expected by _prepare_global_conditioning
        # Ensure keys in obs_dict match what _prepare_global_conditioning expects (OBS_ROBOT etc.)
        # The obs_dict passed from BidirectionalRTDiffusionPolicy uses "observation.state" and "observation.image"
        # We need to map them if _prepare_global_conditioning uses OBS_ROBOT.

        # Create processed_obs for _prepare_global_conditioning
        processed_obs_for_diffusion = {}
        if "observation.state" in obs_dict:
            processed_obs_for_diffusion[OBS_ROBOT] = obs_dict["observation.state"]
        if "observation.image" in obs_dict and self.config.image_features:
            # _prepare_global_conditioning expects "observation.images" if using images
            processed_obs_for_diffusion["observation.images"] = obs_dict["observation.image"]
        if "observation.environment_state" in obs_dict and self.config.env_state_feature:
            processed_obs_for_diffusion[OBS_ENV] = obs_dict["observation.environment_state"]

        global_cond_with_plan = self.diffusion._prepare_global_conditioning(
            processed_obs_for_diffusion,
            plan_features=plan_features
        )

        # 3. Conditional Sampling Logic
        action_sequence: Tensor
        if previous_rt_diffusion_plan is not None:
            print(
                f"[{self.__class__.__name__}] Refining plan with feedback (async)...")
            action_sequence = self.diffusion.async_conditional_sample(
                current_input_normalized=previous_rt_diffusion_plan,
                global_cond=global_cond_with_plan  # This now includes plan features
            )
        else:
            print(f"[{self.__class__.__name__}] Generating initial plan (sync)...")
            action_sequence = self.diffusion.conditional_sample(
                batch_size=batch_size,
                global_cond=global_cond_with_plan  # This now includes plan features
            )

        return action_sequence

    @torch.no_grad()
    def select_action(
        self,
        observation: dict[str, Tensor],
        episode_return: Optional[float] = None,
        episode_step: Optional[int] = None,
        training_progress: Optional[float] = None,
        deterministic: bool = False,
    ) -> Tensor:  # 반환 타입은 실제 반환하는 것에 맞게 조정할 수 있으나, Tensor로 우선 둡니다.
        """Select action for evaluation (when used as a standalone policy)."""
        from lerobot.common.policies.utils import populate_queues

        current_obs_for_norm = {}
        # input_features는 Dict[str, FeatureSpec] 형태이므로, key로 직접 순회합니다.
        for cfg_key in self.config.input_features:
            obs_part_key = cfg_key.split("observation.")[-1]
            if obs_part_key in observation:
                current_obs_for_norm[cfg_key] = observation[obs_part_key]

        current_obs_normalized = self.normalize_inputs(current_obs_for_norm)

        if self.config.image_features:
            current_obs_normalized_for_queue = dict(current_obs_normalized)
            current_obs_normalized_for_queue["observation.images"] = torch.stack(
                [current_obs_normalized[key] for key in self.config.image_features], dim=-4
            )
        else:
            current_obs_normalized_for_queue = current_obs_normalized

        self._queues = populate_queues(
            self._queues, current_obs_normalized_for_queue)

        if len(self._queues["action"]) == 0:
            if len(self._queues["observation.state"]) < self.config.n_obs_steps:
                action_dim = self.config.action_feature.shape[0]
                device = get_device_from_parameters(self)
                return torch.zeros((observation[list(observation.keys())[0]].shape[0], action_dim), device=device)

            obs_history_batch = {}
            for queue_key in self._queues:
                if queue_key.startswith("observation") and len(self._queues[queue_key]) > 0:
                    if queue_key == "observation.state":
                        obs_history_batch[OBS_ROBOT] = torch.stack(
                            list(self._queues[queue_key]), dim=1)
                    elif queue_key == "observation.images" and self.config.image_features:
                        obs_history_batch["observation.images"] = torch.stack(
                            list(self._queues[queue_key]), dim=1)
                    elif queue_key == "observation.environment_state" and self.config.env_state_feature:
                        obs_history_batch[OBS_ENV] = torch.stack(
                            list(self._queues[queue_key]), dim=1)

            actions_normalized = self.diffusion.generate_actions(
                obs_history_batch)
            actions_unnormalized = self.unnormalize_outputs(
                {"action": actions_normalized})["action"]
            self._queues["action"].extend(actions_unnormalized.transpose(0, 1))

        selected_action = self._queues["action"].popleft()
        return selected_action

    # This is the training method
    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the batch through the model and compute the loss for training or validation."""
        # Normalize inputs (observations)
        normalized_batch = self.normalize_inputs(batch)

        # Handle image features specifically if they exist
        if self.config.image_features:
            normalized_batch_with_images = dict(
                normalized_batch)  # shallow copy
            normalized_batch_with_images["observation.images"] = torch.stack(
                [normalized_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            normalized_batch_with_images = normalized_batch

        # Normalize targets (actions)
        # The original batch has 'action', normalize_targets will look for 'action' based on output_features
        final_normalized_batch = self.normalize_targets(
            normalized_batch_with_images)

        # The DiffusionModel.compute_loss expects global_cond to be prepared without plan_features
        # as plan_features are specific to predict_action for plan following.
        # For training, global_cond is based purely on observations.
        global_cond_train = self.diffusion._prepare_global_conditioning(
            final_normalized_batch, plan_features=None)

        loss = self.diffusion.compute_loss(
            final_normalized_batch, global_cond_override=global_cond_train)
        return loss

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the batch through the model with asynchronous diffusion training."""
        from model.diffusion.async_training import AsyncDiffusionTrainer  # Keep import local

        # Normalize inputs and targets
        normalized_batch = self.normalize_inputs(batch)
        if self.config.image_features:
            normalized_batch_with_images = dict(normalized_batch)
            normalized_batch_with_images["observation.images"] = torch.stack(
                [normalized_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            normalized_batch_with_images = normalized_batch

        final_normalized_batch = self.normalize_targets(
            normalized_batch_with_images)

        # Prepare global conditioning (without plan features for training async transformer)
        global_cond_train_async = self.diffusion._prepare_global_conditioning(
            final_normalized_batch, plan_features=None)

        actions = final_normalized_batch["action"]
        # Use config.horizon for async training target
        if actions.shape[1] < self.config.horizon:
            raise ValueError(
                f"Need at least {self.config.horizon} action frames for async training, got {actions.shape[1]}")

        clean_sequence = actions[:, :self.config.horizon, :]

        if not hasattr(self, '_async_trainer'):
            self._async_trainer = AsyncDiffusionTrainer(
                # from config or default
                gap_timesteps=getattr(self.config, 'async_gap_timesteps', 20),
                # from config or default
                gap=getattr(self.config, 'async_gap_value', 3),
                horizon=self.config.horizon  # from config
            )

        loss = self._async_trainer.compute_async_loss(
            clean_sequence=clean_sequence,
            denoising_model=self.diffusion.async_transformer,
            noise_scheduler=self.diffusion.noise_scheduler,
            global_cond=global_cond_train_async
        )
        return loss


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig, plan_feature_dim: int = 0):
        super().__init__()
        self.config = config
        self.plan_feature_dim = plan_feature_dim  # Store it

        # Build observation encoders
        # global_cond_dim here is for observation features only
        obs_only_cond_dim = 0
        if self.config.robot_state_feature:  # Check if robot_state_feature is defined
            obs_only_cond_dim += self.config.robot_state_feature.shape[0]

        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config)
                            for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                obs_only_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                obs_only_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.env_state_feature:  # Check if env_state_feature is defined
            obs_only_cond_dim += self.config.env_state_feature.shape[0]

        # Total dimension for observation part of global_cond, considering n_obs_steps
        global_cond_obs_part_dim_total = obs_only_cond_dim * config.n_obs_steps

        # Final global_cond_dim_total includes the plan features
        global_cond_dim_total_for_transformer = global_cond_obs_part_dim_total + \
            self.plan_feature_dim

        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim_total_for_transformer,  # Use updated dim
            output_dim=config.action_feature.shape[0]
        )

        self.async_transformer = DenoisingTransformer(  # This is your DenoisingTransformer from async_modules
            config,
            global_cond_dim=global_cond_dim_total_for_transformer,  # Use updated dim
            output_dim=config.action_feature.shape[0]
        )

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

        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor], plan_features: Optional[Tensor] = None) -> Tensor:
        """
        Encode observation features and concatenate them with optional plan features.
        batch: Expected to contain keys like OBS_ROBOT, "observation.images", OBS_ENV.
               All tensors in batch are [B, n_obs_steps, ...].
        plan_features: Optional tensor [B, plan_feature_dim].
        """
        batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:
                                                         2]  # Assuming OBS_ROBOT is always present

        global_cond_feats = []

        # Robot state
        if OBS_ROBOT in batch:
            # Flatten feature_dim part if multi-dim
            global_cond_feats.append(batch[OBS_ROBOT].flatten(start_dim=2))

        # Image features
        if self.config.image_features and "observation.images" in batch:
            # [B, n_obs_steps, NumCameras, C, H, W]
            images_data = batch["observation.images"]
            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    images_data, "b s n ... -> n (b s) ...")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(
                        self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(images_data, "b s n ... -> (b s n) ...")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        # Environment state
        if self.config.env_state_feature and OBS_ENV in batch:
            # Should also be [B, n_obs_steps, env_dim]
            global_cond_feats.append(batch[OBS_ENV])

        # Concatenate all observation features along the feature dimension
        # Each element in global_cond_feats is [B, n_obs_steps, D_i]
        concatenated_obs_feats = torch.cat(
            global_cond_feats, dim=-1)  # [B, n_obs_steps, D_obs_total]

        # Flatten the observation features across n_obs_steps
        flattened_obs_feats = concatenated_obs_feats.flatten(
            start_dim=1)  # [B, n_obs_steps * D_obs_total]

        # Now, prepare final conditioning list
        final_cond_list = [flattened_obs_feats]

        if plan_features is not None:
            # plan_features is [B, plan_feature_dim]
            final_cond_list.append(plan_features)

        # Concatenate flattened observation features and plan features
        # [B, (n_obs_steps * D_obs_total) + plan_feature_dim]
        final_global_cond = torch.cat(final_cond_list, dim=-1)

        return final_global_cond

    def conditional_sample(
        # plan_features is now part of global_cond
        self, batch_size: int, global_cond: Tensor, plan_features: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        device = get_device_from_parameters(self)
        dtype = get_dtype_from_parameters(self)

        sample = torch.randn(
            size=(batch_size, self.config.horizon,
                  self.config.action_feature.shape[0]),
            dtype=dtype, device=device, generator=generator,
        )
        self.noise_scheduler.set_timesteps(self.num_inference_steps)

        for t in self.noise_scheduler.timesteps:
            model_output = self.transformer(  # Synchronous transformer
                sample,
                torch.full((batch_size,), t, dtype=torch.long,
                           device=sample.device),
                global_cond=global_cond,  # This global_cond already includes plan_features
            )
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample
        return sample

    def async_conditional_sample(
        self, current_input_normalized: Tensor, global_cond: Tensor,
        # plan_features is now part of global_cond
        plan_features: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None
    ) -> Tensor:
        """
        Refines current_input_normalized using the async_transformer and global_cond (which includes plan features).
        This version calls self.async_transformer (DenoisingTransformer) which has its own async logic.
        """
        device = current_input_normalized.device
        batch_size = current_input_normalized.shape[0]
        horizon = self.config.horizon

        # Parameters for DenoisingTransformer's async behavior
        # These should ideally come from config if DenoisingTransformer is to be highly configurable
        # For now, using fixed values or those potentially in DenoisingTransformer's own config.
        # The DenoisingTransformer.async_conditional_sample should be the one generating async_ts

        # Call the DenoisingTransformer's own async_conditional_sample method
        # This assumes DenoisingTransformer has such a method, which was defined in async_modules.py
        if hasattr(self.async_transformer, 'async_conditional_sample'):
            refined_plan = self.async_transformer.async_conditional_sample(
                current_input_normalized=current_input_normalized,
                global_cond=global_cond  # This already includes plan features
            )
        else:
            # Fallback: if DenoisingTransformer does not have its own specialized async_conditional_sample,
            # use a generic few-step refinement with async_transformer (DenoisingTransformer)
            # This part is similar to the previous DiffusionModel.async_conditional_sample
            print("Warning: DenoisingTransformer.async_conditional_sample not found. Using generic refinement loop.")
            sample = current_input_normalized.clone()
            async_num_steps = getattr(self.config, 'async_refinement_steps', min(
                10, self.num_inference_steps // 10) or 1)

            # Use a small set of timesteps for refinement, starting from a low noise level
            # e.g. start from t=async_num_steps-1 down to 0
            # These are conceptual denoising steps for the refinement process
            refinement_timesteps = torch.arange(
                async_num_steps - 1, -1, -1, device=device, dtype=torch.long)

            for t_idx, conceptual_t in enumerate(refinement_timesteps):
                # Pass the current `sample` (which is x_t_feedback)
                # The `timesteps` argument to `async_transformer` should be the per-token async timesteps
                # if `async_mode=True` is intended for `async_transformer.forward`.
                # If `async_transformer.forward` is called with `async_mode=False`, then `conceptual_t` is used.

                # If async_transformer.forward expects per-token async timesteps even for refinement:
                # base_t_for_async = conceptual_t.expand(batch_size)
                # async_ts_for_transformer = asyn_t_seq(
                #    base_t_for_async,
                #    getattr(self.config, 'feedback_async_gap', 3), # from DenoisingTransformer
                #    horizon
                # )
                # model_input_timesteps = async_ts_for_transformer
                # use_async_mode_for_transformer = True

                # OR if async_transformer.forward uses conceptual_t with async_mode=False for this refinement:
                model_input_timesteps = conceptual_t.expand(batch_size)
                use_async_mode_for_transformer = False

                model_output = self.async_transformer(  # DenoisingTransformer instance
                    sample,
                    model_input_timesteps,
                    global_cond=global_cond,
                    async_mode=use_async_mode_for_transformer
                )
                # The DDPMScheduler step uses the conceptual_t of the *overall sample*
                sample = self.noise_scheduler.step(
                    model_output, conceptual_t, sample, generator=generator).prev_sample
            refined_plan = sample

        return refined_plan

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Generates actions based on observations (without explicit plan conditioning here,
        as this is for the standalone policy use-case).
        """
        batch_size = batch[OBS_ROBOT].shape[0]  # Assuming OBS_ROBOT is always present

        # Prepare global_cond without plan_features for standalone generation
        global_cond_obs_only = self._prepare_global_conditioning(
            batch, plan_features=None)

        actions = self.conditional_sample(
            batch_size, global_cond=global_cond_obs_only)

        # Extract `n_action_steps`
        start = self.config.n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]
        return actions

    def compute_loss(self, batch: dict[str, Tensor], global_cond_override: Optional[Tensor] = None) -> Tensor:
        """
        Computes diffusion loss.
        Can take global_cond_override if global_cond is pre-computed (e.g., in CLDiffPhyConModel.forward).
        """
        trajectory = batch["action"]  # Normalized actions
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],), device=trajectory.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, eps, timesteps)

        if global_cond_override is not None:
            current_global_cond = global_cond_override
        else:
            # Prepare global_cond without plan_features for standard training
            current_global_cond = self._prepare_global_conditioning(
                batch, plan_features=None)

        pred = self.transformer(
            noisy_trajectory, timesteps, global_cond=current_global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory  # batch["action"] is already normalized here
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")
        if self.config.do_mask_loss_for_padding:
            if "action_is_pad" not in batch:
                raise ValueError("Missing 'action_is_pad' for masking loss.")
            in_episode_bound = ~batch["action_is_pad"]
            loss = loss * in_episode_bound.unsqueeze(-1)
        return loss.mean()


# Remaining classes (SpatialSoftmax, DiffusionRgbEncoder, _replace_submodules, DiffusionSinusoidalPosEmb)
# are assumed to be correct and are not repeated here for brevity, but should be in the actual file.

class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape
        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w, dtype=np.float32),
            # Ensure float32 for consistency
            np.linspace(-1.0, 1.0, self._in_h, dtype=np.float32)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1))
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1))
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        features_flat = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features_flat, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config  # Store config
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(
                config.crop_shape)
            self.maybe_random_crop = torchvision.transforms.RandomCrop(
                config.crop_shape) if config.crop_is_random else self.center_crop
        else:
            self.do_crop = False

        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights)
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))

        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "Can't use GroupNorm with pretrained BatchNorm weights.")
            self.backbone = _replace_submodules(
                self.backbone,
                lambda x: isinstance(x, nn.BatchNorm2d),
                lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features)
            )

        # Determine feature_map_shape correctly
        images_shape = next(
            iter(config.image_features.values())).shape  # Get one C,H,W
        dummy_h, dummy_w = config.crop_shape if config.crop_shape else images_shape[1:]
        dummy_c = images_shape[0]
        dummy_input = torch.zeros(
            1, dummy_c, dummy_h, dummy_w)  # Batch, C, H, W

        feature_map_shape = get_output_shape(self.backbone, dummy_input.shape)[
            1:]  # Get C_feat, H_feat, W_feat

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        # feature_dim is num_keypoints * 2 (for x,y coords)
        self.feature_dim_from_pool = config.spatial_softmax_num_keypoints * 2

        # Output linear layer to potentially change feature dim or just pass through
        # If DiffusionConfig has a desired vision_feature_dim, use it. Else, use pooled dim.
        self.final_feature_dim = getattr(
            config, 'vision_feature_dim', self.feature_dim_from_pool)
        self.out = nn.Linear(self.feature_dim_from_pool,
                             self.final_feature_dim)
        self.relu = nn.ReLU()

        # Store the output dimension of this encoder
        self.feature_dim = self.final_feature_dim

    def forward(self, x: Tensor) -> Tensor:
        if self.do_crop:
            x = self.maybe_random_crop(
                x) if self.training else self.center_crop(x)
        x = self.backbone(x)
        x = self.pool(x)  # [B, NumKeypoints, 2]
        x = torch.flatten(x, start_dim=1)  # [B, NumKeypoints * 2]
        x = self.relu(self.out(x))  # [B, feature_dim]
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(
        remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if parents:  # Check if parents list is not empty
            parent_module = root_module.get_submodule(".".join(parents))

        current_module = parent_module
        # Handle cases where parent_module might be Sequential or direct attribute
        if isinstance(parent_module, nn.Sequential):
            src_module = current_module[int(k)]
            new_module = func(src_module)
            current_module[int(k)] = new_module
        else:
            src_module = getattr(current_module, k)
            new_module = func(src_module)
            setattr(current_module, k, new_module)

    assert not any(predicate(m)
                   for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device,
                        dtype=torch.float32) * -emb)  # Ensure float for arange
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)  # Ensure x is float
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
