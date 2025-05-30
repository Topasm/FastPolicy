#!/usr/bin/env python


import math
from collections import deque
from typing import Callable, Optional, Dict  # Added Dict

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torch import Tensor, nn

from lerobot.common.constants import OBS_ENV, OBS_ROBOT, OBS_IMAGE
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
from model.diffusion.diffusion_modules import DiffusionTransformer
from model.diffusion.async_modules import DenoisingTransformer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)


class CLDiffPhyConModel(PreTrainedPolicy):
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

        self._queues = None
        self.diffusion = DiffusionModel(config)
        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:  # Check based on config.image_features being non-empty
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:  # Check based on config.env_state_feature being non-empty
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """
        Selects an action based on the current raw observation.
        This method makes CLDiffPhyConModel a concrete class.
        It primarily supports action-prediction mode. If the model is configured
        to predict states (interpolate_state=True), this method will raise a
        NotImplementedError as it requires an Inverse Dynamics Model to convert
        predicted states to actions.
        """
        if self.config.diffusion_target_key != "action":
            raise NotImplementedError(
                f"select_action in CLDiffPhyConModel is not designed to directly output actions "
                f"when the diffusion model is trained to predict states (target: '{self.config.diffusion_target_key}'). "
                f"An Inverse Dynamics Model would be required. This instance was likely intended for training a state predictor "
                f"or as a component in a policy that handles IDM."
            )

        device = get_device_from_parameters(self)

        # Normalize raw observation.
        # normalize_inputs expects a batch, so add a batch dim and remove it after.
        # Or, ensure current_raw_observation is already batched (e.g., by a wrapper).
        # Assuming current_raw_observation is a single unbatched observation dict.
        # For simplicity, let's assume current_raw_observation might not have batch dim.
        # However, LeRobot policies usually expect batched inputs even for B=1.
        # Let's assume it comes with Batch=1 from an eval loop.
        normalized_obs_for_queue = self.normalize_inputs(
            current_raw_observation)

        # Update observation queues
        # Squeeze batch dimension if it was B=1 for queue storage (if queues store single timesteps)
        self._queues["observation.state"].append(
            normalized_obs_for_queue["observation.state"].squeeze(0))

        if self.config.image_features:
            # Stack images from possibly multiple cameras for the queue
            # normalized_obs_for_queue will have keys like "observation.image.camera_0"
            # These are already [B, C, H, W]. We need [NumCam, C, H, W] for the queue if B=1.
            current_images_stacked_for_queue = torch.stack(
                [normalized_obs_for_queue[key].squeeze(0) for key in self.config.image_features], dim=0
            )  # [NumCam, C, H, W]
            self._queues["observation.images"].append(
                current_images_stacked_for_queue)

        # Manage action execution queue from previous predictions
        if len(self._queues["action"]) > 0:
            return self._queues["action"].popleft()

        # Check if enough observation history is available
        if len(self._queues["observation.state"]) < self.config.n_obs_steps:
            # Default action if not enough history
            action_dim = self.config.output_features["action"].shape[0]
            # print(f"Warning: Not enough obs history for {self.name}. Returning zero action.")
            # Return [ActionDim] for B=1
            return torch.zeros(action_dim, device=device)

        # Prepare batched observations for the model from queues
        # Stack items from deques: list of [StateDim] -> [n_obs_steps, StateDim] -> [1, n_obs_steps, StateDim]
        obs_state_history = torch.stack(
            list(self._queues["observation.state"]), dim=0).unsqueeze(0)

        obs_dict_for_model = {
            OBS_ROBOT: obs_state_history  # Used by _prepare_global_conditioning
        }
        if self.config.image_features:
            # list(self._queues["observation.images"]) is a list of [NumCam, C, H, W]
            # Stack to [n_obs_steps, NumCam, C, H, W], then unsqueeze for batch_dim
            obs_image_history = torch.stack(
                list(self._queues["observation.images"]), dim=0).unsqueeze(0)
            # Used by _prepare_global_conditioning
            obs_dict_for_model["observation.images"] = obs_image_history
            # For predict_action, it might expect "observation.image" if single camera setup was different
            # The current _prepare_global_conditioning uses "observation.images" (plural)

        # Call the core diffusion model prediction
        # predict_action returns a normalized sequence [B, horizon, target_dim]
        # Here B=1.
        predicted_target_sequence_normalized = self.predict_action(
            obs_dict_for_model, previous_rt_diffusion_plan=None)

        # Unnormalize the output. This assumes predict_action returned actions.
        unnormalized_output = self.unnormalize_outputs({
            # Remove B=1 dim for unnormalizer
            "action": predicted_target_sequence_normalized.squeeze(0)
        })
        # [horizon, ActionDim]
        unnormalized_actions = unnormalized_output["action"]

        # Populate action queue for n_action_steps and return the first one
        for i in range(min(self.config.n_action_steps, unnormalized_actions.shape[0])):
            self._queues["action"].append(unnormalized_actions[i])

        if len(self._queues["action"]) > 0:
            return self._queues["action"].popleft()
        else:
            action_dim = self.config.output_features["action"].shape[0]
            return torch.zeros(action_dim, device=device)

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], previous_rt_diffusion_plan: Optional[Tensor] = None) -> Tensor:
        # obs_dict keys here are expected to be OBS_ROBOT, "observation.images", etc.
        # and values have shape [B, n_obs_steps, ...]
        batch_size = obs_dict[OBS_ROBOT].shape[0] if OBS_ROBOT in obs_dict else obs_dict["observation.images"].shape[0]
        # device = get_device_from_parameters(self) # Already on device if called from select_action

        # _prepare_global_conditioning takes the already history-batched obs_dict
        global_cond = self.diffusion._prepare_global_conditioning(obs_dict)

        output_sequence: Tensor
        if previous_rt_diffusion_plan is not None:
            output_sequence = self.diffusion.async_conditional_sample(
                current_input_normalized=previous_rt_diffusion_plan,
                global_cond=global_cond
            )
        else:
            output_sequence = self.diffusion.conditional_sample(
                batch_size=batch_size,
                global_cond=global_cond
            )
        return output_sequence

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        normalized_batch = self.normalize_inputs(batch)

        if self.config.image_features:
            normalized_batch_with_images = dict(normalized_batch)
            # Correct stacking for "observation.images" for _prepare_global_conditioning
            # Input batch[key] is [B, S, C, H, W], stack to [B, S, N_cam, C, H, W]
            # Assuming S is the sequence/history length (n_obs_steps)
            list_of_image_batches = [normalized_batch[key]
                                     for key in self.config.image_features]
            if list_of_image_batches:
                normalized_batch_with_images["observation.images"] = torch.stack(
                    list_of_image_batches, dim=2)  # dim=2 for NumCameras
        else:
            normalized_batch_with_images = normalized_batch

        final_normalized_batch = self.normalize_targets(
            normalized_batch_with_images)

        global_cond_train = self.diffusion._prepare_global_conditioning(
            final_normalized_batch)

        diffusion_target_key = self.config.diffusion_target_key
        if diffusion_target_key not in final_normalized_batch:
            raise KeyError(
                f"Diffusion target key '{diffusion_target_key}' not found in normalized batch. Available keys: {list(final_normalized_batch.keys())}")

        trajectory_to_diffuse = final_normalized_batch[diffusion_target_key]

        loss = self.diffusion.compute_loss(
            trajectory_to_diffuse,
            global_cond_train,
            batch_info_for_masking=final_normalized_batch
        )
        return loss

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        from model.diffusion.async_training import AsyncDiffusionTrainer

        normalized_batch = self.normalize_inputs(batch)
        if self.config.image_features:
            normalized_batch_with_images = dict(normalized_batch)
            list_of_image_batches_async = [
                normalized_batch[key] for key in self.config.image_features]
            if list_of_image_batches_async:
                normalized_batch_with_images["observation.images"] = torch.stack(
                    list_of_image_batches_async, dim=2)
        else:
            normalized_batch_with_images = normalized_batch

        final_normalized_batch = self.normalize_targets(
            normalized_batch_with_images)
        global_cond_train_async = self.diffusion._prepare_global_conditioning(
            final_normalized_batch)

        diffusion_target_key = self.config.diffusion_target_key
        if diffusion_target_key not in final_normalized_batch:
            raise KeyError(
                f"Diffusion target key '{diffusion_target_key}' not found in final_normalized_batch for async training.")

        target_full_sequence = final_normalized_batch[diffusion_target_key]

        if target_full_sequence.shape[1] < self.config.horizon:
            raise ValueError(
                f"Need at least {self.config.horizon} frames for async training target '{diffusion_target_key}', got {target_full_sequence.shape[1]}")

        clean_sequence = target_full_sequence[:, :self.config.horizon, :]

        if not hasattr(self, '_async_trainer'):
            self._async_trainer = AsyncDiffusionTrainer(
                gap_timesteps=getattr(self.config, 'async_gap_timesteps', 20),
                gap=getattr(self.config, 'async_gap_value', 3),
                horizon=self.config.horizon
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
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        obs_only_cond_dim = 0
        if self.config.robot_state_feature:
            obs_only_cond_dim += self.config.robot_state_feature.shape[0]

        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config)
                            for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                if encoders:  # ensure encoders list is not empty
                    obs_only_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                obs_only_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.env_state_feature:
            obs_only_cond_dim += self.config.env_state_feature.shape[0]

        global_cond_obs_part_dim_total = obs_only_cond_dim * config.n_obs_steps
        global_cond_dim_total_for_transformer = global_cond_obs_part_dim_total

        diffusion_target_key = config.diffusion_target_key
        if diffusion_target_key not in config.output_features:
            raise ValueError(
                f"Diffusion target key '{diffusion_target_key}' not found in config.output_features. "
                f"Ensure config.output_features includes the target. "
                f"Available output_features: {list(config.output_features.keys())}"
            )
        target_feature_spec = config.output_features[diffusion_target_key]
        if not hasattr(target_feature_spec, 'shape') or not target_feature_spec.shape:
            raise ValueError(
                f"Shape for diffusion target key '{diffusion_target_key}' is not properly defined in output_features."
            )
        diffusion_output_dim = target_feature_spec.shape[0]

        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim_total_for_transformer,
            output_dim=diffusion_output_dim
        )

        self.async_transformer = DenoisingTransformer(
            config,
            global_cond_dim=global_cond_dim_total_for_transformer,
            output_dim=diffusion_output_dim
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

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        # Determine batch_size and n_obs_steps from available keys
        # OBS_ROBOT: [B, S, StateDim]
        # "observation.images": [B, S, N_cam, C, H, W]
        # OBS_ENV: [B, S, EnvDim]
        if OBS_ROBOT in batch and batch[OBS_ROBOT] is not None:
            batch_size, n_obs_steps = batch[OBS_ROBOT].shape[:2]
        elif "observation.images" in batch and batch["observation.images"] is not None:
            batch_size, n_obs_steps = batch["observation.images"].shape[:2]
        elif OBS_ENV in batch and batch[OBS_ENV] is not None:
            batch_size, n_obs_steps = batch[OBS_ENV].shape[:2]
        else:
            # This case should ideally not be reached if config requires some observation.
            # If it can be reached (e.g. unconditional model), handle appropriately.
            # For now, assume at least one observation type is always present.
            first_valid_key = next(
                (k for k, v in batch.items() if isinstance(v, Tensor) and v.ndim >= 2), None)
            if first_valid_key:
                batch_size, n_obs_steps = batch[first_valid_key].shape[:2]
            else:  # Fallback, though this indicates an issue with input batch or config
                raise ValueError(
                    "Cannot determine batch_size and n_obs_steps from batch for global conditioning.")

        global_cond_feats = []

        if OBS_ROBOT in batch and self.config.robot_state_feature and batch[OBS_ROBOT] is not None:
            global_cond_feats.append(batch[OBS_ROBOT].flatten(start_dim=2))

        if self.config.image_features and "observation.images" in batch and batch["observation.images"] is not None:
            images_data = batch["observation.images"]
            if images_data.ndim != 6:
                if images_data.ndim == 5 and len(self.config.image_features) == 1:
                    images_data = images_data.unsqueeze(2)
                else:
                    raise ValueError(
                        f"Expected observation.images to have 6 dims (B,S,N,C,H,W), got {images_data.ndim}")

            if self.config.use_separate_rgb_encoder_per_camera:
                images_per_camera = einops.rearrange(
                    images_data, "b s n c h w -> n (b s) c h w")
                img_features_list = torch.cat(
                    [encoder(images) for encoder, images in zip(
                        self.rgb_encoder, images_per_camera, strict=True)]
                )
                img_features = einops.rearrange(
                    img_features_list, "(n b s) d -> b s (n d)", b=batch_size, s=n_obs_steps
                )
            else:
                img_features = self.rgb_encoder(
                    einops.rearrange(
                        images_data, "b s n c h w -> (b s n) c h w")
                )
                img_features = einops.rearrange(
                    img_features, "(b s n) d -> b s (n d)", b=batch_size, s=n_obs_steps
                )
            global_cond_feats.append(img_features)

        if OBS_ENV in batch and self.config.env_state_feature and batch[OBS_ENV] is not None:
            global_cond_feats.append(batch[OBS_ENV])

        if not global_cond_feats:
            # If global_cond_dim_total_for_transformer is 0 (e.g. time_embed only in DiffusionTransformer)
            # This means cond_embed in DiffusionTransformer takes only time_emb.
            # The DiffusionTransformer.cond_embed.in_features would be transformer_dim (from time_embed).
            # If transformer_dim (from config) for time_embed and cond_embed is non-zero.
            time_emb_dim = self.transformer.time_embed[-1].out_features
            if self.transformer.cond_embed.in_features == time_emb_dim:
                # This means global_cond for cat with time_emb should be zero-dim
                dummy_device = batch_size > 0 and global_cond_feats and global_cond_feats[0].device or torch.device(
                    "cpu")
                return torch.empty(batch_size, 0, device=dummy_device)
            else:
                raise ValueError(
                    "No global conditioning features found, but DiffusionTransformer expects them.")

        concatenated_obs_feats = torch.cat(global_cond_feats, dim=-1)
        flattened_obs_feats = concatenated_obs_feats.flatten(start_dim=1)
        final_global_cond = flattened_obs_feats
        return final_global_cond

    def conditional_sample(self, batch_size: int, global_cond: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        device = global_cond.device  # get_device_from_parameters(self)
        dtype = global_cond.dtype  # get_dtype_from_parameters(self)

        sample_output_dim = self.transformer.denoising_head.net[-1].out_features

        sample = torch.randn(
            size=(batch_size, self.config.horizon, sample_output_dim),
            dtype=dtype, device=device, generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=device)  # Add device here

        for t in self.noise_scheduler.timesteps:
            model_output = self.transformer(
                sample,
                torch.full((batch_size,), t, dtype=torch.long,
                           device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample
        return sample

    def async_conditional_sample(self, current_input_normalized: Tensor, global_cond: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        device = current_input_normalized.device
        sample = current_input_normalized.clone()
        async_num_steps = getattr(self.config, 'async_refinement_steps', min(
            10, self.num_inference_steps // 10) or 1)

        # Ensure num_inference_steps is set for the scheduler if not already
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=device)  # Ensure timesteps are set

        # For refinement, we typically use a small number of steps from a low noise level.
        # Let's use a subset of the scheduler's timesteps, e.g., the last `async_num_steps`.
        # Or, if `async_num_steps` is small, generate specific timesteps for refinement.
        if async_num_steps <= len(self.noise_scheduler.timesteps):
            refinement_timesteps = self.noise_scheduler.timesteps[-async_num_steps:]
        else:  # If async_num_steps is larger than total inference steps, use all.
            refinement_timesteps = self.noise_scheduler.timesteps

        # If refinement_timesteps needs to be custom (e.g. 0 to async_num_steps-1 for conceptual t)
        # Then the DDPMScheduler step needs to be used carefully.
        # For now, using actual scheduler timesteps for refinement.

        for t in refinement_timesteps:  # Iterate from higher noise to lower noise
            model_input_timesteps = t.expand(sample.shape[0])  # B
            use_async_mode_for_transformer = False

            model_output = self.async_transformer(
                sample,
                model_input_timesteps,
                global_cond=global_cond,
                async_mode=use_async_mode_for_transformer
            )
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample
        refined_plan = sample
        return refined_plan

    @torch.no_grad()
    def refine_state_path(self, initial_state_path: Tensor, observation_batch_for_cond: dict[str, Tensor], num_refinement_steps: Optional[int] = None, generator: Optional[torch.Generator] = None) -> Tensor:
        device = initial_state_path.device  # get_device_from_parameters(self)
        dtype = initial_state_path.dtype  # get_dtype_from_parameters(self)
        batch_size = initial_state_path.shape[0]

        if initial_state_path.shape[1] != self.config.horizon:
            print(
                f"Warning: refine_state_path input horizon {initial_state_path.shape[1]} != model config horizon {self.config.horizon}.")
            if initial_state_path.shape[1] > self.config.horizon:
                initial_state_path = initial_state_path[:,
                                                        :self.config.horizon]
            else:
                padding_needed = self.config.horizon - \
                    initial_state_path.shape[1]
                padding_shape = (batch_size, padding_needed,
                                 initial_state_path.shape[2])
                padding = torch.zeros(
                    padding_shape, device=device, dtype=dtype)
                initial_state_path = torch.cat(
                    [initial_state_path, padding], dim=1)

        global_cond = self._prepare_global_conditioning(
            observation_batch_for_cond)

        effective_num_refinement_steps = num_refinement_steps
        if effective_num_refinement_steps is None:
            effective_num_refinement_steps = min(
                20, self.noise_scheduler.config.num_train_timesteps // 10)
            if effective_num_refinement_steps == 0:
                effective_num_refinement_steps = 1

        self.noise_scheduler.set_timesteps(
            effective_num_refinement_steps, device=device)

        # Start with a bit of noise on initial_state_path if it's considered x0
        # Or, if initial_state_path is already x_t like, proceed directly.
        # Assuming initial_state_path is a "clean" proposed path (x0).
        noise = torch.randn_like(
            initial_state_path, device=device, dtype=dtype)

        # Start refinement from a relatively high noise level (first timestep in refinement schedule)
        start_timestep_for_refinement = self.noise_scheduler.timesteps[0]

        sample = self.noise_scheduler.add_noise(
            initial_state_path, noise,
            torch.full((batch_size,), start_timestep_for_refinement,
                       device=device, dtype=torch.long)
        )

        for t in self.noise_scheduler.timesteps:  # These are the refinement timesteps
            model_input_timesteps = torch.full(
                (batch_size,), t, dtype=torch.long, device=device)
            predicted_noise_or_sample = self.transformer(  # Use the main transformer for refinement
                sample, model_input_timesteps, global_cond=global_cond
            )
            sample = self.noise_scheduler.step(
                predicted_noise_or_sample, t, sample, generator=generator).prev_sample
        return sample

    def compute_loss(self, trajectory_to_diffuse: Tensor, global_cond: Tensor, batch_info_for_masking: Optional[dict[str, Tensor]] = None) -> Tensor:
        eps = torch.randn(trajectory_to_diffuse.shape,
                          device=trajectory_to_diffuse.device)
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory_to_diffuse.shape[0],
                  ), device=trajectory_to_diffuse.device,
        ).long()
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory_to_diffuse, eps, timesteps)

        pred = self.transformer(
            noisy_trajectory, timesteps, global_cond=global_cond)

        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory_to_diffuse
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        loss = F.mse_loss(pred, target, reduction="none")

        if self.config.do_mask_loss_for_padding and batch_info_for_masking is not None:
            pad_mask_key = None
            # Try to find a specific padding mask for the target key
            specific_pad_mask_key = f"{self.config.diffusion_target_key}_is_pad"
            if specific_pad_mask_key in batch_info_for_masking:
                pad_mask_key = specific_pad_mask_key
            elif "action_is_pad" in batch_info_for_masking:  # Fallback to action_is_pad
                pad_mask_key = "action_is_pad"
            # Add more fallbacks if necessary, e.g. a generic "is_pad"

            if pad_mask_key and pad_mask_key in batch_info_for_masking:
                is_pad_mask = batch_info_for_masking[pad_mask_key]
                # Ensure mask matches trajectory horizon used in loss
                # This is self.config.horizon for training
                current_horizon = trajectory_to_diffuse.shape[1]

                if is_pad_mask.shape[1] < current_horizon:
                    # This indicates an issue, mask is shorter than data. Pad mask with False (not_pad).
                    padding_amount = current_horizon - is_pad_mask.shape[1]
                    mask_padding = torch.zeros(
                        is_pad_mask.shape[0], padding_amount, dtype=torch.bool, device=is_pad_mask.device)
                    is_pad_mask_adjusted = torch.cat(
                        [is_pad_mask, mask_padding], dim=1)
                else:
                    is_pad_mask_adjusted = is_pad_mask[:, :current_horizon]

                in_episode_bound = ~is_pad_mask_adjusted
                loss = loss * in_episode_bound.unsqueeze(-1)
            else:
                print(
                    f"Warning: `do_mask_loss_for_padding` is True, but a suitable padding mask key was not found in `batch_info_for_masking` for target '{self.config.diffusion_target_key}'.")
        elif self.config.do_mask_loss_for_padding and batch_info_for_masking is None:
            print(f"Warning: `do_mask_loss_for_padding` is True, but `batch_info_for_masking` was not provided to compute_loss.")

        return loss.mean()


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
            np.linspace(-1.0, 1.0, self._in_h, dtype=np.float32)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1))
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1))
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        # Handle cases where features might be empty (e.g. B=0 due to dataloader drop_last=True and small last batch)
        if features.shape[0] == 0:
            return torch.empty(0, self._out_c, 2, device=features.device, dtype=features.dtype)

        features_flat = features.reshape(-1, self._in_h * self._in_w)
        attention = F.softmax(features_flat, dim=-1)
        expected_xy = attention @ self.pos_grid
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)
        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
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
                lambda x: nn.GroupNorm(num_groups=max(
                    1, x.num_features // 16 if x.num_features > 0 else 1), num_channels=x.num_features)
            )

        if not config.image_features:
            print("Warning: config.image_features is empty for DiffusionRgbEncoder. Using default (3,96,96) for dummy input shape.")
            images_shape_chw = (
                3, config.crop_shape[0] if config.crop_shape else 96, config.crop_shape[1] if config.crop_shape else 96)
        else:
            images_shape_chw = next(iter(config.image_features.values())).shape

        dummy_h, dummy_w = config.crop_shape if config.crop_shape else images_shape_chw[1:]
        dummy_c = images_shape_chw[0]
        # Ensure dummy_c, dummy_h, dummy_w are positive for dummy_input
        if not (dummy_c > 0 and dummy_h > 0 and dummy_w > 0):
            raise ValueError(
                f"Dummy input dimensions must be positive. Got C={dummy_c}, H={dummy_h}, W={dummy_w}")
        dummy_input = torch.zeros(1, dummy_c, dummy_h, dummy_w)

        with torch.no_grad():  # Ensure no_grad for shape inference
            feature_map_shape = get_output_shape(
                self.backbone, dummy_input.shape)[1:]

        if not (feature_map_shape[0] > 0 and feature_map_shape[1] > 0 and feature_map_shape[2] > 0):
            raise ValueError(
                f"Backbone output feature_map_shape has non-positive dimensions: {feature_map_shape}. Check vision_backbone and input shape.")

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim_from_pool = config.spatial_softmax_num_keypoints * 2
        self.final_feature_dim = getattr(
            config, 'vision_feature_dim', self.feature_dim_from_pool)
        # Avoid Linear(0,0)
        if self.feature_dim_from_pool == 0 and self.final_feature_dim == 0:
            self.out = nn.Identity()  # Or handle this case based on expected behavior
            print("Warning: feature_dim_from_pool and final_feature_dim are 0 in DiffusionRgbEncoder. Using nn.Identity for 'out' layer.")
        else:
            self.out = nn.Linear(self.feature_dim_from_pool,
                                 self.final_feature_dim)

        self.relu = nn.ReLU()
        self.feature_dim = self.final_feature_dim

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:  # Handle empty batch
            return torch.empty(0, self.feature_dim, device=x.device, dtype=x.dtype)
        if self.do_crop:
            x = self.maybe_random_crop(
                x) if self.training else self.center_crop(x)
        x = self.backbone(x)
        x = self.pool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.relu(self.out(x))
        return x


def _replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(
        remove_duplicate=True) if predicate(m)]
    for *parents, k_str in replace_list:  # Renamed k to k_str to avoid conflict
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))

        # current_module = parent_module # This was redundant
        if isinstance(parent_module, nn.Sequential):
            idx = int(k_str)
            src_module = parent_module[idx]
            new_module = func(src_module)
            parent_module[idx] = new_module
        else:
            src_module = getattr(parent_module, k_str)
            new_module = func(src_module)
            setattr(parent_module, k_str, new_module)
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
        if half_dim <= 0:  # Avoid division by zero if dim is 0 or 1
            if self.dim == 1:
                # Or some other handling for dim=1
                return torch.zeros_like(x.float().unsqueeze(-1))
            return torch.empty(*x.shape, self.dim, device=device, dtype=torch.float32)

        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(
            half_dim, device=device, dtype=torch.float32) * -emb)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
