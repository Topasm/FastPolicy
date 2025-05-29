#!/usr/bin/env python

# Copyright 2024 Columbia Artificial Intelligence, Robotics Lab,
# and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"

TODO(alexander-soare):
  - Remove reliance on diffusers for DDPMScheduler and LR scheduler.
"""

import math
from collections import deque
from typing import Callable

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
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)


class CLDiffPhyConModel(PreTrainedPolicy):
    """
    Diffusion Policy as per "Diffusion Policy: Visuomotor Policy Learning via Action Diffusion"
    (paper: https://arxiv.org/abs/2303.04137, code: https://github.com/real-stanford/diffusion_policy).
    """

    config_class = DiffusionConfig
    name = "diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
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

        # queues are populated during rollout of the policy, they contain the n latest observations and actions
        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.reset()

    def get_optim_params(self) -> dict:
        return self.diffusion.parameters()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
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

#     def reset(self):
#         """Clear observation and action queues. Should be called on `env.reset()`"""
#         self._queues = {
#             "observation.state": deque(maxlen=self.config.n_obs_steps),
#             "action": deque(maxlen=self.config.n_action_steps), # 이 큐는 이제 CL-DiffPhyCon 방식에서는 다른 용도
#         }
#         if self.config.image_features:
#             self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
#         if self.config.env_state_feature:
#             self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

#         self._u_pred_async = None # 에피소드 시작 시 이전 예측 초기화
#         # self._action_queue_for_execution = deque(maxlen=self.config.n_action_steps) # 필요하다면 별도 큐

    # def select_action(self, batch: dict[str, Tensor]) -> Tensor:
    #     """Select action using CL-DiffPhyCon asynchronous inference."""
    #     device = get_device_from_parameters(self)  # policy.device 대신 사용

    #     # 1. 입력 정규화 및 현재 관찰 준비
    #     normalized_batch_obs_only = {}
    #     # 현재 관찰만 사용 (batch는 현재 스텝의 관찰만 포함한다고 가정)
    #     for k_in, k_out in self.config.input_features.items():
    #         if k_in in batch:  # Check if the key from config exists in the batch
    #             normalized_batch_obs_only[k_out] = batch[k_in]  # Use mapping
    #         # else: # Handle missing keys if necessary, e.g. if state/image is optional
    #         #    print(f"Warning: Key {k_in} not found in current observation batch for select_action")

    #     normalized_batch_obs_only = self.normalize_inputs(
    #         normalized_batch_obs_only)

    #     if self.config.image_features:
    #         # input_features에 정의된 이미지 키들을 사용해야 함
    #         # 예: "observation.image_rgb", "observation.image_depth"
    #         # self.config.image_features는 단순 리스트가 아니라 Dict[str, FeatureSpec] 일 수 있음

    #         # 가정: self.config.image_features가 ["observation.image"] 형태의 키 리스트라고 가정
    #         # 실제로는 config.input_features 에서 image 관련 키들을 찾아야 함
    #         img_keys_in_batch = [
    #             k for k in normalized_batch_obs_only if "image" in k]  # 더 정확한 로직 필요
    #         if img_keys_in_batch:
    #             normalized_batch_obs_only["observation.images"] = torch.stack(
    #                 [normalized_batch_obs_only[key] for key in img_keys_in_batch], dim=-4
    #             )

    #     # populate_queues는 n_obs_steps 만큼의 히스토리를 만드는데 사용됨.
    #     # CL-DiffPhyCon은 현재 관찰 s_k를 사용하므로, 이 부분은 config.n_obs_steps=1 일때 간단해짐.
    #     # n_obs_steps > 1 이면, populate_queues로 히스토리 관찰을 만들고,
    #     # _prepare_global_conditioning에 전달.
    #     self._queues = populate_queues(
    #         self._queues, normalized_batch_obs_only)  # 현재 관찰을 큐에 추가/업데이트

    #     current_stacked_obs_batch = {
    #         # OBS_ROBOT, "observation.images" 등을 큐에서 가져와 스택
    #         k: torch.stack(list(self._queues[k]), dim=1)
    #         for k in self.config.input_features  # self._queues에 있는 키만 사용
    #         if k in self._queues and len(self._queues[k]) > 0
    #     }

    #     if not current_stacked_obs_batch:  # 초기 스텝, 큐가 비어있을 수 있음
    #         print(
    #             "Warning: current_stacked_obs_batch is empty in select_action. Using raw batch.")
    #         # 이 경우, batch에서 직접 global_cond를 만들어야 함 (n_obs_steps=1 이라고 가정)
    #         # 또는 populate_queues가 최소 1개의 데이터를 보장하도록 해야 함
    #         temp_global_cond_input = {}
    #         if OBS_ROBOT in normalized_batch_obs_only:
    #              temp_global_cond_input[OBS_ROBOT] = normalized_batch_obs_only[OBS_ROBOT].unsqueeze(
    #                  1)  # Add sequence dim
    #          if "observation.images" in normalized_batch_obs_only:  # 이미 stack 되어있다고 가정
    #              temp_global_cond_input["observation.images"] = normalized_batch_obs_only["observation.images"].unsqueeze(
    #                  1)  # Add sequence dim
    #          # ... 기타 환경 상태 등 ...
    #          if not temp_global_cond_input:
    #              raise ValueError(
    #                  "Cannot prepare global_cond, not enough observation data.")
    #          global_cond = self.diffusion._prepare_global_conditioning(
    #              temp_global_cond_input)
    #     else:
    #         global_cond = self.diffusion._prepare_global_conditioning(
    #             current_stacked_obs_batch)

    #     # 2. 비동기 추론 로직
    #     if self._u_pred_async is None:
    #         # 초기화 단계: 표준 동기 샘플링 (ddpm_init 역할)
    #         # self.diffusion.transformer (동기 모델) 사용
    #         print(f"[{self.__class__.__name__}] Generating initial plan (sync)...")
    #         # conditional_sample은 정규화된 액션을 반환한다고 가정
    #         current_plan_normalized = self.diffusion.conditional_sample(
    #             batch_size=global_cond.shape[0],
    #             global_cond=global_cond
    #         )  # (B, 16, action_dim)
    #     else:
    #         # 피드백 단계: 비동기 샘플링 (ddpm_feedback 역할)
    #         # self.diffusion.async_transformer 사용
    #         print(f"[{self.__class__.__name__}] Updating plan (async)...")

    #         # current_input_normalized 준비:
    #         # 현재 실제 상태 (u_init)와 이전 예측의 뒷부분 (u_pred_shifted)을 결합해야 함.
    #         # u_init은 current_stacked_obs_batch에서 파생될 수 있으나, 액션 차원으로 변환 필요.
    #         # 또는, CL-DiffPhyCon 논문처럼 u_pred를 그대로 사용하고,
    #         # DenoisingTransformer가 global_cond (현재 상태 s_k)를 통해 현재 상태를 반영하도록 함.
    #         # 여기서는 후자를 가정. u_pred를 바로 current_input_normalized로 사용.

    #         current_plan_normalized = self.diffusion.async_conditional_sample(
    #             current_input_normalized=self._u_pred_async,  # 이전 스텝의 정규화된 예측
    #             global_cond=global_cond,  # 현재 관찰 기반의 global_cond
    #             # async_t_seq_func, gap_timesteps 등은 async_conditional_sample 내부에서 처리되거나 config에서 가져옴
    #         )

    #     # 3. 다음 스텝을 위한 _u_pred_async 업데이트 (롤링/쉬프팅)
    #     #   다음 u_pred는 현재 plan에서 한 스텝 밀린 것.
    #     #   [p1, p2, ..., p16] -> 다음 u_pred는 [p2, ..., p16, new_noisy_frame] 형태
    #     next_u_pred = torch.zeros_like(current_plan_normalized, device=device)
    #     next_u_pred[:, :-1, :] = current_plan_normalized[:, 1:, :]
    #     # 마지막 프레임은 새롭게 노이즈를 추가하거나, 0으로 채우거나, 복제할 수 있음.
    #     # 가장 간단하게는 이전 프레임 복제 또는 0으로 채우기.
    #     # 또는 약간의 노이즈 추가:
    #     next_u_pred[:, -1,
    #                 :] = torch.randn_like(next_u_pred[:, -1, :]) * 0.1  # 예시
    #     # 아니면 그냥 마지막것 복사: next_u_pred[:, -1, :] = current_plan_normalized[:, -1, :]
    #     self._u_pred_async = next_u_pred

    #     # 4. 실제 환경에 적용할 액션 추출 및 비정규화
    #     # 현재 plan (current_plan_normalized)에서 첫 번째 액션을 사용
    #     # (B, action_dim)
    #     action_normalized_to_execute = current_plan_normalized[:, 0, :]

    #     action_unnormalized = self.unnormalize_outputs(
    #         # config.output_features에 "action" 키가 있다고 가정
    #         {"action": action_normalized_to_execute}
    #     )["action"]

    #     return action_unnormalized

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method handles caching a history of observations and an action trajectory generated by the
        underlying diffusion model. Here's how it works:
          - `n_obs_steps` steps worth of observations are cached (for the first steps, the observation is
            copied `n_obs_steps` times to fill the cache).
          - The diffusion model generates `horizon` steps worth of actions.
          - `n_action_steps` worth of actions are actually kept for execution, starting from the current step.
        Schematically this looks like:
            ----------------------------------------------------------------------------------------------
            (legend: o = n_obs_steps, h = horizon, a = n_action_steps)
            |timestep            | n-o+1 | n-o+2 | ..... | n     | ..... | n+a-1 | n+a   | ..... | n-o+h |
            |observation is used | YES   | YES   | YES   | YES   | NO    | NO    | NO    | NO    | NO    |
            |action is generated | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   | YES   |
            |action is used      | NO    | NO    | NO    | YES   | YES   | YES   | NO    | NO    | NO    |
            ----------------------------------------------------------------------------------------------
        Note that this means we require: `n_action_steps <= horizon - n_obs_steps + 1`. Also, note that
        "horizon" may not the best name to describe what the variable actually means, because this period is
        actually measured from the first observation which (if `n_obs_steps` > 1) happened in the past.
        """
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            # shallow copy so that adding a key doesn't modify the original
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        # Note: It's important that this happens after stacking the images into a single key.
        self._queues = populate_queues(self._queues, batch)

        if len(self._queues["action"]) == 0:
            # stack n latest observations from the queue
            batch = {k: torch.stack(
                list(self._queues[k]), dim=1) for k in batch if k in self._queues}
            actions = self.diffusion.generate_actions(batch)

            # TODO(rcadene): make above methods return output dictionary?
            actions = self.unnormalize_outputs({"action": actions})["action"]

            self._queues["action"].extend(actions.transpose(0, 1))

        action = self._queues["action"].popleft()
        return action

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the batch through the model and compute the loss for training or validation."""
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            # shallow copy so that adding a key doesn't modify the original
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)
        loss = self.diffusion.compute_loss(batch)
        # Just return the loss directly for training
        return loss

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        """Run the batch through the model with asynchronous diffusion training."""
        from model.diffusion.async_training import AsyncDiffusionTrainer

        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)

        # Prepare global conditioning
        global_cond = self.diffusion._prepare_global_conditioning(batch)

        # Extract 16-frame action sequence
        actions = batch["action"]  # (B, horizon, action_dim)
        if actions.shape[1] < 16:
            raise ValueError(
                f"Need at least 16 action frames for async training, got {actions.shape[1]}")

        # Take first 16 frames as ground truth sequence
        clean_sequence = actions[:, :16, :]  # (B, 16, action_dim)

        # Initialize async trainer if not exists
        if not hasattr(self, '_async_trainer'):
            # Calculate safe parameters to stay within num_train_timesteps=100
            # For horizon=16, gap=3: max_timestep = 19 + (16-1)*3 = 19 + 45 = 64 < 100 ✓
            self._async_trainer = AsyncDiffusionTrainer(
                # Low range for base timestep sampling (0-19)
                gap_timesteps=20,
                gap=3,             # Gap between consecutive timesteps
                horizon=16         # 16-frame horizon
            )

        # Compute asynchronous diffusion loss
        loss = self._async_trainer.compute_async_loss(
            clean_sequence=clean_sequence,
            denoising_model=self.diffusion.async_transformer,
            noise_scheduler=self.diffusion.noise_scheduler,
            global_cond=global_cond
        )

        return loss


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """
    Factory for noise scheduler instances of the requested type. All kwargs are passed
    to the scheduler.
    """
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

        # Build observation encoders (depending on which observations are provided).
        global_cond_dim = self.config.robot_state_feature.shape[0]
        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                encoders = [DiffusionRgbEncoder(config)
                            for _ in range(num_images)]
                self.rgb_encoder = nn.ModuleList(encoders)
                global_cond_dim += encoders[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                global_cond_dim += self.rgb_encoder.feature_dim * num_images
        if self.config.env_state_feature:
            global_cond_dim += self.config.env_state_feature.shape[0]

        # Calculate total global conditioning dimension
        global_cond_dim_total = global_cond_dim * config.n_obs_steps

        # Replace UNet with DiffusionTransformer
        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim_total,
            output_dim=config.action_feature.shape[0]
        )

        # Also create an async-capable transformer
        from model.diffusion.async_modules import DenoisingTransformer
        self.async_transformer = DenoisingTransformer(
            config,
            global_cond_dim=global_cond_dim_total,
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
            # Predict model output with transformer instead of UNet
            model_output = self.transformer(
                sample,
                torch.full(sample.shape[:1], t,
                           dtype=torch.long, device=sample.device),
                global_cond=global_cond,
            )
            # Compute previous image: x_t -> x_t-1
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample

        return sample

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

    def generate_actions(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have:
        {
            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)
        }
        """
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(
            batch)  # (B, global_cond_dim)

        # run sampling
        actions = self.conditional_sample(batch_size, global_cond=global_cond)

        # Extract `n_action_steps` steps worth of actions (from the current observation).
        start = n_obs_steps - 1
        end = start + self.config.n_action_steps
        actions = actions[:, start:end]

        return actions

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
        # Input validation.
        assert set(batch).issuperset(
            {"observation.state", "action", "action_is_pad"})
        assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1]
        horizon = batch["action"].shape[1]
        assert horizon == self.config.horizon
        assert n_obs_steps == self.config.n_obs_steps

        # Encode image features and concatenate them all together along with the state vector.
        global_cond = self._prepare_global_conditioning(
            batch)  # (B, global_cond_dim)

        # Forward diffusion.
        trajectory = batch["action"]
        # Sample noise to add to the trajectory.
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        # Sample a random noising timestep for each item in the batch.
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        # Add noise to the clean trajectories according to the noise magnitude at each timestep.
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, eps, timesteps)

        # Run the transformer instead of UNet for denoising
        pred = self.transformer(
            noisy_trajectory, timesteps, global_cond=global_cond)

        # Compute the loss.
        # The target is either the original trajectory, or the noise.
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = batch["action"]
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

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


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(
            self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(
                config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        # The dummy input should take the number of image channels from `config.image_features` and it should
        # use the height and width from `config.crop_shape` if it is provided, otherwise it should use the
        # height and width from `config.image_features`.

        # Note: we have a check in the config class to make sure all images have the same shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[
            1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        feature_map_shape = get_output_shape(self.backbone, dummy_shape)[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2
        self.out = nn.Linear(
            config.spatial_softmax_num_keypoints * 2, self.feature_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:  # noqa: SIM108
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        # Final linear layer with non-linearity.
        x = self.relu(self.out(x))
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(
        remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all BN are replaced
    assert not any(predicate(m)
                   for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
