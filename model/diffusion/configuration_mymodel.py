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
from dataclasses import dataclass, field
from typing import Optional

from lerobot.common.optim.optimizers import AdamConfig
from lerobot.common.optim.schedulers import DiffuserSchedulerConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode


@PreTrainedConfig.register_subclass("mydiffusion")
@dataclass
class DiffusionConfig(PreTrainedConfig):
    """Configuration class for DiffusionPolicy using a Transformer backbone (DiT).

    Can be configured to predict actions directly or predict future states and use
    an inverse dynamics model to infer actions.

    Defaults are configured for training with PushT providing proprioceptive and single camera observations.

    The parameters you will most likely need to change are the ones which depend on the environment / sensors.
    Those are: `input_shapes` and `output_shapes`.

    Notes on the inputs and outputs:
        - "observation.state" is required as an input key.
        - Either:
            - At least one key starting with "observation.image is required as an input.
              AND/OR
            - The key "observation.environment_state" is required as input.
        - If there are multiple keys beginning with "observation.image" they are treated as multiple camera
          views. Right now we only support all images having the same shape.
        - If `predict_state` is False (default):
            - "action" is required as an output key for the diffusion model target.
        - If `predict_state` is True:
            - "next_observation.state" (or similar key representing future states) is required as the diffusion model target.
            - `inv_dyn_model_path` must be provided for action generation during inference.
            - "action" is still required in `output_shapes` for the final policy output.

    Args:
        predict_state: If True, the diffusion model predicts future states. If False, it predicts actions directly.
        n_obs_steps: Number of environment steps worth of observations to pass to the policy.
        horizon: Diffusion model prediction size (length of the state/action sequence).
        n_action_steps: The number of action steps to run in the environment for one invocation of the policy.
        input_shapes: A dictionary defining the shapes of the input data for the policy.
        output_shapes: A dictionary defining the shapes of the output data for the policy.
        input_normalization_modes: A dictionary specifying normalization modes for input modalities.
        output_normalization_modes: Similar dictionary for output normalization/unnormalization.
        vision_backbone: Name of the torchvision resnet backbone for image encoding.
        crop_shape: (H, W) shape to crop images to. If None, no cropping.
        crop_is_random: Whether cropping is random during training.
        pretrained_backbone_weights: Pretrained weights for the vision backbone.
        use_group_norm: Whether to replace BatchNorm with GroupNorm in the vision backbone.
        spatial_softmax_num_keypoints: Number of keypoints for SpatialSoftmax pooling.
        use_separate_rgb_encoders_per_camera: Whether to use a separate RGB encoder per camera.
        transformer_dim: Hidden dimension of the Diffusion Transformer.
        transformer_num_layers: Number of layers in the Diffusion Transformer.
        transformer_num_heads: Number of attention heads in the Diffusion Transformer.
        noise_scheduler_type: Name of the noise scheduler ("DDPM", "DDIM").
        num_train_timesteps: Number of diffusion steps for training schedule.
        beta_schedule: Diffusion beta schedule name.
        beta_start: Initial beta value.
        beta_end: Final beta value.
        prediction_type: Type of prediction ("epsilon" or "sample").
        clip_sample: Whether to clip samples during inference.
        clip_sample_range: Clipping range magnitude.
        num_inference_steps: Number of steps for inference. Defaults to `num_train_timesteps`.
        num_inference_samples: Number of candidate sequences (states or actions) to generate during inference.
        inv_dyn_model_path: Path to the pretrained inverse dynamics model (required if `predict_state` is True).
        inv_dyn_hidden_dim: Hidden dimension for the inverse dynamics MLP.
        critic_model_path: Path to the pretrained critic model (optional, used for selecting best sample).
        critic_hidden_dim: Hidden dimension for the critic MLP.
        do_mask_loss_for_padding: Whether to mask loss for padded targets (actions or states).
    """

    # Prediction mode
    predict_state: bool = False

    # Inputs / output structure.
    n_obs_steps: int = 2
    horizon: int = 16
    n_action_steps: int = 8

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )

    drop_n_last_frames: int = 7

    # Architecture / modeling.
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False
    transformer_dim: int = 512
    transformer_num_layers: int = 6
    transformer_num_heads: int = 8
    noise_scheduler_type: str = "DDPM"
    num_train_timesteps: int = 100
    beta_schedule: str = "squaredcos_cap_v2"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    prediction_type: str = "epsilon"
    clip_sample: bool = True
    clip_sample_range: float = 1.0

    # Inference
    num_inference_steps: int | None = None
    num_inference_samples: int = 1

    # Auxiliary Models (Paths and Config)
    inv_dyn_model_path: Optional[str] = None
    inv_dyn_hidden_dim: int = 512
    critic_model_path: Optional[str] = None
    critic_hidden_dim: int = 128

    # Loss computation
    do_mask_loss_for_padding: bool = False

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple = (0.95, 0.999)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-6
    scheduler_name: str = "cosine"
    scheduler_warmup_steps: int = 500

    def __post_init__(self):
        super().__post_init__()
        # Validation logic
        if self.predict_state:
            # Check if a state key exists for prediction target
            # Allow 'observation.state' itself to be the target key
            state_keys = [
                k for k in self.output_features if k.endswith("state")]
            if not state_keys:
                raise ValueError(
                    "When predict_state is True, output_features must contain at least one key ending with 'state' (e.g., 'observation.state' or 'next_observation.state') to be used as the diffusion target."
                )
            # Ensure action is still in output_features for the final policy output
            if "action" not in self.output_features:
                raise ValueError(
                    "'action' must be included in output_features even when predict_state is True, as it is the final policy output."
                )
        else:
            # Ensure action is the only output if not predicting state
            # Allow other keys if needed, but action must be present
            if "action" not in self.output_features:
                raise ValueError(
                    "When predict_state is False, 'action' must be included in output_features."
                )
            # Original check (might be too strict depending on use case):
            # if list(self.output_features.keys()) != ["action"]:
            #     raise ValueError(
            #         "When predict_state is False, 'action' should be the only key in output_features."
            #     )

        # Validate horizon vs n_action_steps and n_obs_steps
        if self.transformer_dim % self.transformer_num_heads != 0:
            raise ValueError(
                f"{self.transformer_dim=} must be divisible by {self.transformer_num_heads=}"
            )

        supported_prediction_types = ["epsilon", "sample"]
        if self.prediction_type not in supported_prediction_types:
            raise ValueError(
                f"`prediction_type` must be one of {supported_prediction_types}. Got {self.prediction_type}."
            )
        supported_noise_schedulers = ["DDPM", "DDIM"]
        if self.noise_scheduler_type not in supported_noise_schedulers:
            raise ValueError(
                f"`noise_scheduler_type` must be one of {supported_noise_schedulers}. "
                f"Got {self.noise_scheduler_type}."
            )

        if self.num_inference_samples > 1 and self.critic_model_path is None:
            print(
                f"Warning: `num_inference_samples` is {self.num_inference_samples}, but no "
                "`critic_model_path` provided. The first sample will be chosen by default."
            )

    def get_optimizer_preset(self) -> AdamConfig:
        return AdamConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
        )

    def get_scheduler_preset(self) -> DiffuserSchedulerConfig:
        return DiffuserSchedulerConfig(
            name=self.scheduler_name,
            num_warmup_steps=self.scheduler_warmup_steps,
        )

    def validate_features(self) -> None:
        if len(self.image_features) == 0 and self.env_state_feature is None:
            raise ValueError(
                "You must provide at least one image or the environment state among the inputs."
            )

        if self.crop_shape is not None:
            for key, image_ft in self.image_features.items():
                if self.crop_shape[0] > image_ft.shape[1] or self.crop_shape[1] > image_ft.shape[2]:
                    raise ValueError(
                        f"`crop_shape` should fit within the images shapes. Got {self.crop_shape} "
                        f"for `crop_shape` and {image_ft.shape} for "
                        f"`{key}`."
                    )

        if len(self.image_features) > 0:
            first_image_key, first_image_ft = next(
                iter(self.image_features.items()))
            for key, image_ft in self.image_features.items():
                if image_ft.shape != first_image_ft.shape:
                    raise ValueError(
                        f"`{key}` does not match `{first_image_key}`, but we expect all image shapes to match."
                    )

        if "action" not in self.output_features:
            raise ValueError(
                "The key 'action' must be present in `output_features` for the final policy output."
            )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def target_delta_indices(self) -> list:
        """Indices relative to the start of the observation window for the target sequence."""
        # If predicting states, the target sequence starts right after the last observation step.
        # If predicting actions, the target sequence might include the action concurrent with the last obs step.
        # Assuming state prediction target starts at index 1 relative to last observation step.
        start_index = 1
        return list(range(start_index, start_index + self.horizon))

    @property
    def action_delta_indices(self) -> list:
        """Indices relative to the start of the observation window for the action sequence.
        Required by the base class, even if not directly used for state prediction targets.
        Assumes actions start from the last observation step (index 0).
        """
        start_index = 0
        return list(range(start_index, start_index + self.horizon))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def diffusion_target_key(self) -> str:
        if self.predict_state:
            # Find state keys in output_features
            state_keys = [
                k for k in self.output_features if k.endswith("state")]
            if not state_keys:
                # This should ideally be caught by __post_init__, but added for safety
                raise ValueError(
                    "No state key found in output_features for state prediction.")

            # Prioritize keys like 'next_observation.state' if they exist
            preferred_keys = [
                k for k in state_keys if k != "observation.state"]
            if preferred_keys:
                # If multiple non-'observation.state' keys exist, raise ambiguity or pick first
                if len(preferred_keys) > 1:
                    print(
                        f"Warning: Multiple potential state target keys found ({preferred_keys}). Using '{preferred_keys[0]}'.")
                return preferred_keys[0]
            elif "observation.state" in state_keys:
                # If only 'observation.state' is found, use it as the target
                return "observation.state"
            else:
                # This case should not be reachable if __post_init__ passed
                raise ValueError(
                    "Cannot determine diffusion target key for state prediction.")
        else:
            # If not predicting state, the target is action
            return "action"
