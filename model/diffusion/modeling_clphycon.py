#!/usr/bin/env python


import math
from collections import deque
from typing import Callable, Optional, Dict  # Added Dict
from einops import rearrange  # Ensure einops is imported if used directly

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
# Assuming this is in the path
from model.diffusion.diffusion_modules import DiffusionTransformer
# Assuming this is in the path
from model.diffusion.async_modules import DenoisingTransformer
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
)
from collections import OrderedDict  # For _replace_submodules OrderedDict fix


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
        if self.config.image_features:
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        if self.config.diffusion_target_key != "action":
            raise NotImplementedError(
                f"select_action in CLDiffPhyConModel is not designed to directly output actions "
                f"when the diffusion model is trained to predict states (target: '{self.config.diffusion_target_key}'). "
                f"An Inverse Dynamics Model would be required. This instance was likely intended for training a state predictor "
                f"or as a component in a policy that handles IDM."
            )

        device = get_device_from_parameters(self)

        normalized_obs_for_queue = self.normalize_inputs(
            current_raw_observation)

        self._queues["observation.state"].append(
            normalized_obs_for_queue["observation.state"].squeeze(0))

        if self.config.image_features:
            current_images_stacked_for_queue = torch.stack(
                [normalized_obs_for_queue[key].squeeze(0) for key in self.config.image_features], dim=0
            )
            self._queues["observation.images"].append(
                current_images_stacked_for_queue)

        if len(self._queues["action"]) > 0:
            return self._queues["action"].popleft()

        if len(self._queues["observation.state"]) < self.config.n_obs_steps:
            action_dim = self.config.output_features["action"].shape[0]
            return torch.zeros(action_dim, device=device)

        obs_state_history = torch.stack(
            list(self._queues["observation.state"]), dim=0).unsqueeze(0)

        obs_dict_for_model = {
            OBS_ROBOT: obs_state_history
        }
        if self.config.image_features:
            obs_image_history = torch.stack(
                list(self._queues["observation.images"]), dim=0).unsqueeze(0)
            obs_dict_for_model["observation.images"] = obs_image_history

        predicted_target_sequence_normalized = self.predict_action(
            obs_dict_for_model, previous_rt_diffusion_plan=None)

        unnormalized_output = self.unnormalize_outputs({
            "action": predicted_target_sequence_normalized.squeeze(0)
        })
        unnormalized_actions = unnormalized_output["action"]

        for i in range(min(self.config.n_action_steps, unnormalized_actions.shape[0])):
            self._queues["action"].append(unnormalized_actions[i])

        if len(self._queues["action"]) > 0:
            return self._queues["action"].popleft()
        else:
            action_dim = self.config.output_features["action"].shape[0]
            return torch.zeros(action_dim, device=device)

    @torch.no_grad()
    def predict_action(self, obs_dict: dict[str, Tensor], previous_rt_diffusion_plan: Optional[Tensor] = None) -> Tensor:
        batch_size = -1
        if OBS_ROBOT in obs_dict and obs_dict[OBS_ROBOT] is not None:
            batch_size = obs_dict[OBS_ROBOT].shape[0]
        elif "observation.images" in obs_dict and obs_dict["observation.images"] is not None:
            batch_size = obs_dict["observation.images"].shape[0]
        elif OBS_ENV in obs_dict and obs_dict[OBS_ENV] is not None:
            batch_size = obs_dict[OBS_ENV].shape[0]
        else:
            first_key = next(iter(obs_dict.keys()))
            batch_size = obs_dict[first_key].shape[0]

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

    def _prepare_batch_for_global_cond(self, normalized_batch_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """
        Slices the necessary history from normalized_batch_inputs for global conditioning.
        Ensures all conditioning modalities have sequence length self.config.n_obs_steps.
        """
        batch_for_global_cond = {}
        cond_horizon = self.config.n_obs_steps

        # Process OBS_ROBOT (state) for conditioning
        if OBS_ROBOT in normalized_batch_inputs and self.config.robot_state_feature:
            # [B, S_loaded, Dim]
            state_input_full = normalized_batch_inputs[OBS_ROBOT]
            # Take the slice corresponding to observation_delta_indices, which are the first cond_horizon steps
            # if delta_timestamps was constructed as [-N+1..0, 1..H_target]
            if state_input_full.shape[1] < cond_horizon:
                raise ValueError(
                    f"Full state input seq len {state_input_full.shape[1]} < cond_horizon {cond_horizon}")
            batch_for_global_cond[OBS_ROBOT] = state_input_full[:,
                                                                :cond_horizon, :]

        # Process image features for conditioning
        if self.config.image_features:
            list_of_image_tensors_for_cond = []
            for key in self.config.image_features:
                if key in normalized_batch_inputs:
                    # [B, S_img_loaded, C, H, W]
                    img_input_full = normalized_batch_inputs[key]
                    if img_input_full.shape[1] < cond_horizon:
                        raise ValueError(
                            f"Full image input seq len {img_input_full.shape[1]} < cond_horizon {cond_horizon} for key {key}")
                    list_of_image_tensors_for_cond.append(
                        img_input_full[:, :cond_horizon, :, :, :])
            if list_of_image_tensors_for_cond:
                batch_for_global_cond["observation.images"] = torch.stack(
                    list_of_image_tensors_for_cond, dim=2)

        # Process OBS_ENV for conditioning
        if OBS_ENV in normalized_batch_inputs and self.config.env_state_feature:
            # [B, S_env_loaded, Dim]
            env_input_full = normalized_batch_inputs[OBS_ENV]
            if env_input_full.shape[1] < cond_horizon:
                raise ValueError(
                    f"Full env input seq len {env_input_full.shape[1]} < cond_horizon {cond_horizon}")
            batch_for_global_cond[OBS_ENV] = env_input_full[:,
                                                            :cond_horizon, :]

        return batch_for_global_cond

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        normalized_batch_inputs = self.normalize_inputs(batch)
        batch_for_global_cond = self._prepare_batch_for_global_cond(
            normalized_batch_inputs)

        normalized_targets_batch = self.normalize_targets(batch)

        final_normalized_batch = dict(batch_for_global_cond)
        for key, val in normalized_targets_batch.items():
            if key in self.config.output_features:
                final_normalized_batch[key] = val

        global_cond_train = self.diffusion._prepare_global_conditioning(
            batch_for_global_cond)

        diffusion_target_key = self.config.diffusion_target_key
        if diffusion_target_key not in final_normalized_batch:
            raise KeyError(
                f"Diffusion target key '{diffusion_target_key}' not found in final_normalized_batch. Available keys: {list(final_normalized_batch.keys())}")

        # trajectory_to_diffuse is the target sequence (e.g. states for t=1..H)
        # It should have length self.config.horizon
        # LeRobotDataset loads this based on cfg.target_delta_indices
        trajectory_to_diffuse = final_normalized_batch[diffusion_target_key]
        if trajectory_to_diffuse.shape[1] != self.config.horizon:
            raise ValueError(
                f"Target trajectory for '{diffusion_target_key}' has length {trajectory_to_diffuse.shape[1]}, expected horizon {self.config.horizon}")

        loss = self.diffusion.compute_loss(
            trajectory_to_diffuse,
            global_cond_train,
            batch_info_for_masking=final_normalized_batch
        )
        return loss

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        from model.diffusion.async_training import AsyncDiffusionTrainer

        normalized_batch_inputs = self.normalize_inputs(batch)
        batch_for_global_cond_async = self._prepare_batch_for_global_cond(
            normalized_batch_inputs)

        normalized_targets_batch_async = self.normalize_targets(batch)
        final_normalized_batch_async = dict(batch_for_global_cond_async)
        for key, val in normalized_targets_batch_async.items():
            if key in self.config.output_features:
                final_normalized_batch_async[key] = val

        global_cond_train_async = self.diffusion._prepare_global_conditioning(
            batch_for_global_cond_async)

        diffusion_target_key = self.config.diffusion_target_key
        if diffusion_target_key not in final_normalized_batch_async:
            raise KeyError(
                f"Diffusion target key '{diffusion_target_key}' not found in final_normalized_batch for async training.")

        target_full_sequence = final_normalized_batch_async[diffusion_target_key]

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
            # Instantiate DiffusionRgbEncoder correctly based on config
            if self.config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [DiffusionRgbEncoder(config) for _ in range(num_images)])
                if encoders:
                    # Use feature_dim from instance
                    obs_only_cond_dim += self.rgb_encoder[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(
                    config)  # Single encoder
                obs_only_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.env_state_feature:
            obs_only_cond_dim += self.config.env_state_feature.shape[0]

        global_cond_dim_per_step = obs_only_cond_dim
        global_cond_dim_total_for_transformer = global_cond_dim_per_step * config.n_obs_steps

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

    def _prepare_global_conditioning(self, batch_for_cond: dict[str, Tensor]) -> Tensor:
        """
        Prepares the global conditioning vector from a batch dictionary.
        `batch_for_cond` MUST contain tensors for conditioning (OBS_ROBOT, observation.images, OBS_ENV)
        where each tensor ALREADY has its sequence length equal to self.config.n_obs_steps.
        Slicing of longer sequences from dataloader should happen *before* calling this method
        (e.g., in CLDiffPhyConModel.forward/forward_async).
        """
        conditioning_horizon = self.config.n_obs_steps
        batch_size = -1
        first_valid_tensor_for_device = None  # For device info if list is empty

        # Determine batch_size from a reliable source in the batch
        # All conditioning inputs are expected to have B and S=conditioning_horizon dimensions.
        for key in [OBS_ROBOT, "observation.images", OBS_ENV]:  # Check in order of likelihood
            if key in batch_for_cond and batch_for_cond[key] is not None:
                batch_size = batch_for_cond[key].shape[0]
                first_valid_tensor_for_device = batch_for_cond[key]
                # CRITICAL SANITY CHECK: Ensure input sequence lengths match conditioning_horizon
                if batch_for_cond[key].shape[1] != conditioning_horizon:
                    raise ValueError(
                        f"INTERNAL ERROR: Sequence length for conditioning key '{key}' ({batch_for_cond[key].shape[1]}) "
                        f"in _prepare_global_conditioning's input batch does not match config.n_obs_steps ({conditioning_horizon}). "
                        f"Slicing should have occurred before calling this method."
                    )
                break  # Found batch_size

        if batch_size == -1:  # No standard conditioning keys found or all were None
            if not batch_for_cond:  # Empty input dict
                # This means global_cond_dim_total_for_transformer should be 0.
                # The DiffusionTransformer's cond_embed would only take time_embedding.
                time_emb_dim = self.transformer.time_embed[-1].out_features
                if self.transformer.cond_embed.in_features == time_emb_dim:
                    # Need a batch_size to return empty tensor of shape [B, 0]
                    # This path is problematic if batch_size cannot be inferred.
                    # Assuming if batch_for_cond is empty, it implies an unconditional call
                    # where global_cond is not used or is just time.
                    # If the calling code guarantees a batch_size (e.g. from target tensor), use that.
                    # For now, this path is ambiguous for determining batch_size.
                    # Let's assume if this is reached, it's an error unless model is purely unconditional on obs.
                    raise ValueError(
                        "Cannot determine batch_size for _prepare_global_conditioning as input batch_for_cond is effectively empty or missing standard keys.")
                else:
                    raise ValueError(
                        "No conditioning features found in batch_for_cond, but DiffusionTransformer expects them.")
            else:  # Fallback: try to get batch_size from any tensor if standard keys absent/None
                first_key_in_batch = next(iter(batch_for_cond))
                batch_size = batch_for_cond[first_key_in_batch].shape[0]
                first_valid_tensor_for_device = batch_for_cond[first_key_in_batch]

        processed_features_list = []

        # Process robot state for conditioning
        if OBS_ROBOT in batch_for_cond and self.config.robot_state_feature and batch_for_cond[OBS_ROBOT] is not None:
            state_data_cond = batch_for_cond[OBS_ROBOT]
            processed_features_list.append(state_data_cond)

        # Process image features for conditioning
        if self.config.image_features and "observation.images" in batch_for_cond and batch_for_cond["observation.images"] is not None:
            images_data_cond = batch_for_cond["observation.images"]

            s_img_cond = images_data_cond.shape[1]
            n_cam_cond = images_data_cond.shape[2]

            # This s_img_cond should be == conditioning_horizon due to checks/slicing before calling
            if self.config.use_separate_rgb_encoder_per_camera:
                img_features_all_cams = []
                for i in range(n_cam_cond):
                    cam_images = images_data_cond[:, :, i, :, :, :]
                    cam_images_flat = rearrange(
                        cam_images, "b s c h w -> (b s) c h w")
                    cam_features_encoded = self.rgb_encoder[i](cam_images_flat)
                    cam_features_reshaped = rearrange(cam_features_encoded, "(b s) d -> b s d",
                                                      b=batch_size, s=s_img_cond)
                    img_features_all_cams.append(cam_features_reshaped)
                img_features_processed = torch.cat(
                    img_features_all_cams, dim=-1)
            else:  # Single shared encoder
                images_flat_for_encoder = rearrange(
                    images_data_cond, "b s n c h w -> (b s n) c h w")
                img_features_encoded = self.rgb_encoder(
                    images_flat_for_encoder)
                img_features_processed = rearrange(
                    img_features_encoded, "(b s n) d -> b s (n d)",
                    b=batch_size, s=s_img_cond, n=n_cam_cond
                )
            processed_features_list.append(img_features_processed)

        # Process environment state for conditioning
        if OBS_ENV in batch_for_cond and self.config.env_state_feature and batch_for_cond[OBS_ENV] is not None:
            env_data_cond = batch_for_cond[OBS_ENV]
            processed_features_list.append(env_data_cond)

        if not processed_features_list:
            device_to_use = first_valid_tensor_for_device.device if first_valid_tensor_for_device is not None else torch.device(
                "cpu")
            return torch.empty(batch_size, 0, device=device_to_use)

        concatenated_per_step_feats = torch.cat(
            processed_features_list, dim=-1)
        flattened_global_cond = concatenated_per_step_feats.flatten(
            start_dim=1)

        return flattened_global_cond

    def conditional_sample(self, batch_size: int, global_cond: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        device = global_cond.device
        dtype = global_cond.dtype

        sample_output_dim = self.transformer.denoising_head.net[-1].out_features

        sample = torch.randn(
            size=(batch_size, self.config.horizon, sample_output_dim),
            dtype=dtype, device=device, generator=generator,
        )
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=device)

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

        async_num_steps = getattr(
            self.config, 'async_refinement_steps', self.num_inference_steps)
        if async_num_steps == 0 and self.num_inference_steps > 0:
            async_num_steps = 1
        elif self.num_inference_steps == 0:
            async_num_steps = 0

        if async_num_steps == 0:
            return sample

        self.noise_scheduler.set_timesteps(async_num_steps, device=device)

        timesteps_to_refine = self.noise_scheduler.timesteps

        if not len(timesteps_to_refine):
            return sample

        for t in timesteps_to_refine:
            model_input_timesteps = t.repeat(sample.shape[0])
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
        device = initial_state_path.device
        dtype = initial_state_path.dtype
        batch_size = initial_state_path.shape[0]

        current_horizon = initial_state_path.shape[1]
        target_horizon = self.config.horizon

        if current_horizon != target_horizon:
            if current_horizon > target_horizon:
                initial_state_path_adjusted = initial_state_path[:,
                                                                 :target_horizon]
            else:
                padding_needed = target_horizon - current_horizon
                padding_shape = (batch_size, padding_needed,
                                 initial_state_path.shape[2])
                last_frame = initial_state_path[:, -1:, :]
                padding = last_frame.repeat(1, padding_needed, 1)
                initial_state_path_adjusted = torch.cat(
                    [initial_state_path, padding], dim=1)
        else:
            initial_state_path_adjusted = initial_state_path

        global_cond = self._prepare_global_conditioning(
            observation_batch_for_cond)

        effective_num_refinement_steps = num_refinement_steps
        if effective_num_refinement_steps is None:
            effective_num_refinement_steps = getattr(self.config, 'num_refinement_steps_default',
                                                     min(10, self.noise_scheduler.config.num_train_timesteps // 20) or 1)

        if effective_num_refinement_steps == 0:
            return initial_state_path_adjusted

        self.noise_scheduler.set_timesteps(
            effective_num_refinement_steps, device=device)

        if not len(self.noise_scheduler.timesteps):
            return initial_state_path_adjusted

        noise = torch.randn_like(
            initial_state_path_adjusted, device=device, dtype=dtype)
        start_timestep_for_refinement = self.noise_scheduler.timesteps[0]

        sample = self.noise_scheduler.add_noise(
            initial_state_path_adjusted, noise,
            torch.full((batch_size,), start_timestep_for_refinement,
                       device=device, dtype=torch.long)
        )

        for t in self.noise_scheduler.timesteps:
            model_input_timesteps = torch.full(
                (batch_size,), t, dtype=torch.long, device=device)
            predicted_noise_or_sample = self.transformer(
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
            specific_pad_mask_key = f"{self.config.diffusion_target_key}_is_pad"

            if specific_pad_mask_key in batch_info_for_masking:
                pad_mask_key = specific_pad_mask_key
            elif self.config.diffusion_target_key == "action" and "action_is_pad" in batch_info_for_masking:
                pad_mask_key = "action_is_pad"
            elif "is_pad" in batch_info_for_masking and batch_info_for_masking["is_pad"].shape[1] == trajectory_to_diffuse.shape[1]:
                pad_mask_key = "is_pad"

            if pad_mask_key and pad_mask_key in batch_info_for_masking:
                is_pad_mask = batch_info_for_masking[pad_mask_key]
                current_horizon = trajectory_to_diffuse.shape[1]

                if is_pad_mask.shape[1] < current_horizon:
                    padding_amount = current_horizon - is_pad_mask.shape[1]
                    mask_padding = torch.zeros(
                        is_pad_mask.shape[0], padding_amount, dtype=torch.bool, device=is_pad_mask.device)
                    is_pad_mask_adjusted = torch.cat(
                        [is_pad_mask, mask_padding], dim=1)
                else:
                    is_pad_mask_adjusted = is_pad_mask[:, :current_horizon]

                in_episode_bound = ~is_pad_mask_adjusted
                if loss.ndim == 3 and in_episode_bound.ndim == 2:
                    loss = loss * in_episode_bound.unsqueeze(-1)
                elif loss.ndim == in_episode_bound.ndim:
                    loss = loss * in_episode_bound
                else:
                    print(
                        f"Warning: Loss shape {loss.shape} and padding mask shape {in_episode_bound.shape} are not compatible for broadcasting in masking.")

            elif self.config.do_mask_loss_for_padding:
                print(
                    f"Warning: `do_mask_loss_for_padding` is True, but a suitable padding mask key ('{specific_pad_mask_key}', 'action_is_pad', or 'is_pad') was not found in `batch_info_for_masking` for target '{self.config.diffusion_target_key}'. Loss will not be masked for padding.")

        return loss.mean()


class SpatialSoftmax(nn.Module):
    def __init__(self, input_shape, num_kp=None):
        super().__init__()
        assert len(
            input_shape) == 3, f"SpatialSoftmax input_shape must be 3D (C,H,W), got {input_shape}"
        self._in_c, self._in_h, self._in_w = input_shape

        self.is_valid = not (
            self._in_c <= 0 or self._in_h <= 0 or self._in_w <= 0)

        if not self.is_valid:
            self.nets = nn.Identity()
            self._out_c = self._in_c if num_kp is None else num_kp
            self.register_buffer(
                "pos_grid", torch.empty(0, 2, dtype=torch.float32))
            return

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
        if not self.is_valid:
            batch_size = features.shape[0]
            return torch.zeros(batch_size, self._out_c, 2, device=features.device, dtype=features.dtype)

        if features.shape[1] != self._in_c or features.shape[2] != self._in_h or features.shape[3] != self._in_w:
            # This means input feature's C,H,W doesn't match what SpatialSoftmax was configured for.
            # This can happen if the backbone output shape changed or was miscalculated.
            # print(f"Warning: SpatialSoftmax forward input feature shape {features.shape} mismatches init C={self._in_c}, H={self._in_h},W={self._in_w}")
            # Fallback to zeros, but this is an indicator of a deeper problem.
            return torch.zeros(features.shape[0], self._out_c, 2, device=features.device, dtype=features.dtype)

        # Apply Conv2D if it exists (to change num channels to num_kp)
        if self.nets is not None:
            processed_features = self.nets(features)
        else:
            # Use features directly if nets is None (num_kp was not set or _in_c was 0)
            processed_features = features

        # After nets, features are [B, self._out_c, H, W]
        current_h, current_w = processed_features.shape[2], processed_features.shape[3]

        if processed_features.shape[0] == 0:
            return torch.empty(0, self._out_c, 2, device=processed_features.device, dtype=processed_features.dtype)

        if current_h == 0 or current_w == 0:
            return torch.zeros(processed_features.shape[0], self._out_c, 2, device=processed_features.device, dtype=processed_features.dtype)

        # Ensure pos_grid matches the H, W of processed_features if they could change (they shouldn't after conv1x1)
        if current_h * current_w != self.pos_grid.shape[0]:
            # This should not happen if H,W are fixed by backbone and nets is 1x1 conv.
            # Recreate pos_grid if necessary (though it's a sign of an issue)
            # print(f"Warning: Recreating pos_grid in SpatialSoftmax due to shape mismatch. Feat H,W: ({current_h},{current_w}), Grid H,W: ({self._in_h},{self._in_w})")
            pos_x, pos_y = np.meshgrid(
                np.linspace(-1.0, 1.0, current_w, dtype=np.float32),
                np.linspace(-1.0, 1.0, current_h, dtype=np.float32)
            )
            pos_x_tensor = torch.from_numpy(pos_x.reshape(
                current_h * current_w, 1)).to(self.pos_grid.device)
            pos_y_tensor = torch.from_numpy(pos_y.reshape(
                current_h * current_w, 1)).to(self.pos_grid.device)
            current_pos_grid = torch.cat([pos_x_tensor, pos_y_tensor], dim=1)
        else:
            current_pos_grid = self.pos_grid

        features_flat = processed_features.reshape(-1, current_h * current_w)
        attention = F.softmax(features_flat, dim=-1)
        expected_xy = attention @ current_pos_grid
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

        # Determine dummy input shape for feature_map_shape calculation
        # Use a reliable way to get at least one image feature's shape
        images_shape_chw_iter_list = [feat.shape for feat_name, feat in config.image_features.items(
        ) if feat_name.startswith("observation.image")]
        if not images_shape_chw_iter_list:  # If no "observation.image" keys, but other image_features might exist
            images_shape_chw_iter_list = [
                feat.shape for feat in config.image_features.values()]

        if not images_shape_chw_iter_list:
            # print("Warning: config.image_features is empty for DiffusionRgbEncoder. Using default (3,96,96) for dummy input shape.")
            first_image_shape = (3, config.crop_shape[0] if config.crop_shape and config.crop_shape[0] > 0 else 96,
                                 config.crop_shape[1] if config.crop_shape and config.crop_shape[1] > 0 else 96)
        else:
            first_image_shape = images_shape_chw_iter_list[0]

        dummy_c_cfg = first_image_shape[0]
        # Use crop_shape if available, else use shape from feature spec
        crop_h, crop_w = (config.crop_shape if config.crop_shape else (
            first_image_shape[1], first_image_shape[2]))

        dummy_c = dummy_c_cfg if dummy_c_cfg > 0 else 3
        dummy_h = crop_h if crop_h > 0 else 96
        dummy_w = crop_w if crop_w > 0 else 96

        dummy_input = torch.zeros(1, dummy_c, dummy_h, dummy_w)

        with torch.no_grad():
            try:
                # Pass a clone of dummy_input to avoid in-place modification issues if any
                feature_map_shape_tuple = get_output_shape(
                    self.backbone, dummy_input.clone().shape)[1:]
            except Exception as e:
                # print(f"Error getting output shape from backbone for DiffusionRgbEncoder with dummy_input {dummy_input.shape}: {e}")
                # print("Defaulting feature_map_shape to (1,1,1) to allow SpatialSoftmax init, but this is likely an error.")
                feature_map_shape_tuple = (1, 1, 1)

        c_feat, h_feat, w_feat = feature_map_shape_tuple
        # Ensure dimensions passed to SpatialSoftmax are positive
        valid_feature_map_shape = (max(1, c_feat) if c_feat is not None else 1,
                                   max(1, h_feat) if h_feat is not None else 1,
                                   max(1, w_feat) if w_feat is not None else 1)
        # if feature_map_shape_tuple != valid_feature_map_shape:
        # print(f"Warning: Backbone output feature_map_shape {feature_map_shape_tuple} corrected to {valid_feature_map_shape} for SpatialSoftmax.")

        self.pool = SpatialSoftmax(
            valid_feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim_from_pool = config.spatial_softmax_num_keypoints * 2
        self.final_feature_dim = getattr(
            config, 'vision_feature_dim', self.feature_dim_from_pool)

        if self.feature_dim_from_pool <= 0:  # Check less than or equal to handle robustly
            #  print(f"Warning: feature_dim_from_pool is {self.feature_dim_from_pool} for DiffusionRgbEncoder. Output layer will be Identity if final_feature_dim is also 0, or Linear(0, D).")
            # If final_feature_dim is also 0 or less, Identity is safest. Otherwise, Linear(0, D) is an issue.
            if self.final_feature_dim <= 0:
                self.out = nn.Identity()
                self.final_feature_dim = 0  # Ensure final_feature_dim reflects Identity output
            else:  # feature_dim_from_pool is 0, but final_feature_dim > 0
                # This is problematic: Linear(0, D)
                self.out = nn.Linear(0, self.final_feature_dim)
        else:  # feature_dim_from_pool > 0
            self.out = nn.Linear(self.feature_dim_from_pool,
                                 self.final_feature_dim)

        self.relu = nn.ReLU()
        self.feature_dim = self.final_feature_dim  # Store the actual output dimension

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[0] == 0:
            return torch.empty(0, self.feature_dim, device=x.device, dtype=x.dtype)
        if self.do_crop:
            x = self.maybe_random_crop(
                x) if self.training else self.center_crop(x)

        backbone_out = self.backbone(x)

        if backbone_out.shape[2] <= 0 or backbone_out.shape[3] <= 0:
            # print(f"Warning: Backbone output has non-positive spatial dimensions ({backbone_out.shape}). Returning zeros from RGB encoder.")
            return torch.zeros(x.shape[0], self.feature_dim, device=x.device, dtype=x.dtype)

        pooled_out = self.pool(backbone_out)
        flattened_out = torch.flatten(pooled_out, start_dim=1)

        if isinstance(self.out, nn.Linear):
            if flattened_out.shape[1] != self.out.in_features:
                if self.out.in_features == 0:
                    # print(f"Warning: Input dim {flattened_out.shape[1]} to Linear(0, D) layer in DiffusionRgbEncoder. Returning zeros for safety.")
                    return torch.zeros(flattened_out.shape[0], self.out.out_features, device=flattened_out.device, dtype=flattened_out.dtype)
                else:  # self.out.in_features != 0 and mismatch
                    raise ValueError(
                        f"Dimension mismatch for Linear layer in DiffusionRgbEncoder. Expected {self.out.in_features}, got {flattened_out.shape[1]}")

        output = self.relu(self.out(flattened_out))
        return output


def _replace_submodules(root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    if predicate(root_module):
        return func(root_module)
    replace_list = [k.split(".") for k, m in root_module.named_modules(
        remove_duplicate=True) if predicate(m)]
    for *parents, k_str in replace_list:
        parent_module = root_module
        if parents:
            parent_module = root_module.get_submodule(".".join(parents))

        if isinstance(parent_module, nn.Sequential):
            try:
                idx = int(k_str)
                src_module = parent_module[idx]
                new_module = func(src_module)
                parent_module[idx] = new_module
            except ValueError:
                if hasattr(parent_module, k_str):
                    src_module = getattr(parent_module, k_str)
                    new_module = func(src_module)
                    # Attempt to replace by name if parent is an OrderedDict (like default nn.Sequential)
                    try:
                        parent_module._modules[k_str] = new_module
                    except AttributeError:
                        print(
                            f"Warning: Could not directly setattr or use _modules to replace {k_str} on Sequential {type(parent_module)}. Module replacement might be incomplete.")
                else:
                    print(
                        f"Warning: Could not find module name {k_str} in Sequential block {parent_module} to replace.")
        else:
            src_module = getattr(parent_module, k_str)
            new_module = func(src_module)
            setattr(parent_module, k_str, new_module)
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        if half_dim <= 0:
            if self.dim == 1:
                return torch.sin(x.float().unsqueeze(-1))
            return torch.zeros(*x.shape, self.dim, device=device, dtype=torch.float32)

        emb_val = math.log(10000) / (half_dim -
                                     1) if half_dim > 1 else math.log(10000)
        emb = torch.exp(torch.arange(half_dim, device=device,
                        dtype=torch.float32) * -emb_val)
        emb = x.float().unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if self.dim > 0 and self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1), "constant", 0)

        return emb
