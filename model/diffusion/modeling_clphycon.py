#!/usr/bin/env python


from collections import deque
from typing import Optional, Dict
import torch
from torch import Tensor

from lerobot.common.constants import OBS_ENV_STATE, OBS_STATE
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
# Import the refactored diffusion modules
from model.diffusion.diffusion_modules import DiffusionModel
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters, populate_queues


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

        # Initialize normalization and model
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats)

        config.use_async_transformer = True
        self.model = DiffusionModel(config)
        self.diffusion = self.model

        self._queues = None
        self.reset()

    @torch.no_grad()
    def sample_cl_diffphycon_step(
        self,
        prev_noisy_sequence: torch.Tensor,
        current_env_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Single CL-DiffPhyCon algorithm step: denoise current and prepare next sequence.

        Args:
            prev_noisy_sequence: Noisy trajectory from previous step (B, H, target_dim)
            current_env_state: Current environmental state for conditioning

        Returns:
            current_output: Clean output for current step (B, target_dim)
            next_noisy_sequence: Prepared noisy sequence for next step (B, H, target_dim)
        """
        # Prepare conditioning from environment state
        if isinstance(current_env_state, dict):
            obs_dict = current_env_state
        else:
            batch_size = prev_noisy_sequence.shape[0]
            device = prev_noisy_sequence.device

            # Create observation dict with expanded state
            from lerobot.common.constants import OBS_STATE
            obs_state = current_env_state.unsqueeze(
                1).expand(-1, self.config.n_obs_steps, -1)
            obs_dict = {OBS_STATE: obs_state}

            # Add image observations if available
            if hasattr(self, '_queues') and self.config.image_features and 'observation.images' in self._queues:
                if len(self._queues['observation.images']) == self.config.n_obs_steps:
                    img_obs = torch.stack(
                        list(self._queues['observation.images']), dim=0)
                    if img_obs.dim() == 4:  # [S, C, H, W]
                        img_obs = img_obs.unsqueeze(0).expand(
                            batch_size, -1, -1, -1, -1).to(device)
                    obs_dict['observation.images'] = img_obs

        # Get global conditioning and timestep for denoising
        global_cond = self.diffusion._prepare_global_conditioning(obs_dict)
        timestep_T_div_H = int(
            self.diffusion.noise_scheduler.config.num_train_timesteps // self.config.horizon)

        # Denoise the sequence
        denoised_sequence = self.diffusion.sample_asynchronous_step(
            prev_noisy_sequence,
            timestep_T_div_H,
            global_cond,
            num_async_inference_steps=getattr(
                self.config, 'num_async_inference_steps', None)
        )

        # Get current output and prepare next sequence
        # Check if first token is -1.0, -1.0 or similar (sentinel value)
        first_token = denoised_sequence[:, 0, :]
        # Check if all values in the first token are close to -1.0
        is_first_token_sentinel = torch.all(torch.isclose(first_token, torch.tensor(
            [-1.0], device=first_token.device), rtol=0.05, atol=0.05), dim=1)

        # If first token seems to be a sentinel value, use the second token as the current output
        if torch.any(is_first_token_sentinel):
            print(
                "Detected sentinel value (-1.0) in first token, using the second token as current output")
            current_output = denoised_sequence[:, 1, :]
            # When using second token as output, shift sequence starts from third token
            shifted_sequence = denoised_sequence[:, 2:, :]
            # Create two new noise elements to maintain sequence length
            batch_size, _, target_dim = prev_noisy_sequence.shape
            new_noise = torch.randn(
                (batch_size, 2, target_dim),
                device=prev_noisy_sequence.device
            )
        else:
            # Normal case: use first token as output
            current_output = denoised_sequence[:, 0, :]
            # Shift sequence starts from second token
            shifted_sequence = denoised_sequence[:, 1:, :]
            # Create one new noise element
            batch_size, _, target_dim = prev_noisy_sequence.shape
            new_noise = torch.randn(
                (batch_size, 1, target_dim),
                device=prev_noisy_sequence.device
            )

        # Form sequence for next step
        next_noisy_sequence = torch.cat([shifted_sequence, new_noise], dim=1)

        return current_output, next_noisy_sequence

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
            OBS_STATE: obs_state_history
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
        if OBS_STATE in obs_dict and obs_dict[OBS_STATE] is not None:
            batch_size = obs_dict[OBS_STATE].shape[0]
        elif "observation.images" in obs_dict and obs_dict["observation.images"] is not None:
            batch_size = obs_dict["observation.images"].shape[0]
        elif OBS_ENV_STATE in obs_dict and obs_dict[OBS_ENV_STATE] is not None:
            batch_size = obs_dict[OBS_ENV_STATE].shape[0]
        else:
            if not obs_dict:
                raise ValueError("obs_dict is empty in predict_action")
            first_key = next(iter(obs_dict.keys()))
            batch_size = obs_dict[first_key].shape[0]

        global_cond = self.diffusion._prepare_global_conditioning(obs_dict)

        if previous_rt_diffusion_plan is not None:
            return self.diffusion.async_conditional_sample(
                current_input_normalized=previous_rt_diffusion_plan,
                global_cond=global_cond
            )
        return self.diffusion.conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond
        )

    def _prepare_batch_for_cond_logic(self, normalized_batch_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Prepare batch for global conditioning."""
        result = {}
        cond_horizon = self.config.n_obs_steps

        # Process robot state
        if OBS_STATE in normalized_batch_inputs and self.config.robot_state_feature:
            state_input = normalized_batch_inputs[OBS_STATE]
            if state_input.shape[1] < cond_horizon:
                raise ValueError(
                    f"State input length {state_input.shape[1]} < cond_horizon {cond_horizon}")
            result[OBS_STATE] = state_input[:, :cond_horizon, :]

        # Process image features
        if self.config.image_features:
            image_tensors = []
            for key in self.config.image_features:
                if key in normalized_batch_inputs:
                    img_input = normalized_batch_inputs[key]
                    if img_input.shape[1] < cond_horizon:
                        raise ValueError(
                            f"Image input length {img_input.shape[1]} < cond_horizon {cond_horizon}")
                    image_tensors.append(img_input[:, :cond_horizon, :, :, :])
            if image_tensors:
                result["observation.images"] = torch.stack(
                    image_tensors, dim=2)

        # Process environment state
        if OBS_ENV_STATE in normalized_batch_inputs and self.config.env_state_feature:
            env_input = normalized_batch_inputs[OBS_ENV_STATE]
            if env_input.shape[1] < cond_horizon:
                raise ValueError(
                    f"Environment input length {env_input.shape[1]} < cond_horizon {cond_horizon}")
            result[OBS_ENV_STATE] = env_input[:, :cond_horizon, :]

        return result

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        normalized_inputs = self.normalize_inputs(batch)
        batch_for_cond = self._prepare_batch_for_cond_logic(normalized_inputs)
        normalized_targets = self.normalize_targets(batch)

        # Prepare final batch with conditioning and targets
        final_batch = dict(batch_for_cond)
        for key, val in normalized_targets.items():
            if key in self.config.output_features:
                final_batch[key] = val

        global_cond = self.diffusion._prepare_global_conditioning(
            batch_for_cond)

        # Get and validate diffusion target
        target_key = self.config.diffusion_target_key
        if target_key not in final_batch:
            raise KeyError(
                f"Target key '{target_key}' not found. Available: {list(final_batch.keys())}")

        trajectory = final_batch[target_key]
        if trajectory.shape[1] != self.config.horizon:
            raise ValueError(
                f"Target trajectory length {trajectory.shape[1]} != horizon {self.config.horizon}")

        return self.diffusion.compute_loss(
            trajectory,
            global_cond,
            batch_info_for_masking=final_batch
        )

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        from model.diffusion.async_training import AsyncDiffusionTrainer

        normalized_inputs = self.normalize_inputs(batch)
        batch_for_cond = self._prepare_batch_for_cond_logic(normalized_inputs)
        normalized_targets = self.normalize_targets(batch)

        # Prepare final batch with conditioning and targets
        final_batch = dict(batch_for_cond)
        for key, val in normalized_targets.items():
            if key in self.config.output_features:
                final_batch[key] = val

        global_cond = self.diffusion._prepare_global_conditioning(
            batch_for_cond)

        # Get and validate diffusion target
        target_key = self.config.diffusion_target_key
        if target_key not in final_batch:
            raise KeyError(
                f"Target key '{target_key}' not found for async training")

        target_sequence = final_batch[target_key]
        if target_sequence.shape[1] < self.config.horizon:
            raise ValueError(
                f"Need at least {self.config.horizon} frames for async training, got {target_sequence.shape[1]}")

        clean_sequence = target_sequence[:, :self.config.horizon, :]

        # Initialize async trainer if needed
        if not hasattr(self, '_async_trainer'):
            self._async_trainer = AsyncDiffusionTrainer(
                gap_timesteps=getattr(self.config, 'async_gap_timesteps', 20),
                gap=getattr(self.config, 'async_gap_value', 3),
                horizon=self.config.horizon
            )

        return self._async_trainer.compute_async_loss(
            clean_sequence=clean_sequence,
            denoising_model=self.diffusion.async_transformer,
            noise_scheduler=self.diffusion.noise_scheduler,
            global_cond=global_cond
        )
