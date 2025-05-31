#!/usr/bin/env python


from collections import deque
from typing import Optional, Dict
import torch
from torch import Tensor

from lerobot.common.constants import OBS_ENV, OBS_ROBOT
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.common.policies.normalize import Normalize, Unnormalize
# Import the refactored diffusion modules
from model.diffusion.diffusion_modules import DiffusionModel
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters


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

        # Initialize normalization modules
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Initialize the diffusion model
        config.use_async_transformer = True  # Enable async transformer for CLDiffPhyCon
        self.model = DiffusionModel(config)

        self._queues = None
        self.diffusion = self.model  # Use the initialized model
        self.reset()

    @torch.no_grad()
    def initialize_cl_diffphycon_state(
        self,
        # Plan from BidirectionalARTransformer, shape (B, H, target_dim)
        initial_clean_plan: torch.Tensor
    ) -> torch.Tensor:
        """
        Initialize the CL-DiffPhyCon state from a clean plan, producing the initial
        noisy trajectory at SDE time T/H for asynchronous diffusion.

        This method implements the initialization step for CL-DiffPhyCon by adding noise
        according to a specific timestep schedule that varies across the horizon.

        Args:
            initial_clean_plan: Clean trajectory from which to initialize, shape (B, H, target_dim)

        Returns:
            torch.Tensor: Noisy trajectory at time T/H for each timestep in the sequence, shape (B, H, target_dim)
        """
        device = initial_clean_plan.device
        batch_size, horizon, _ = initial_clean_plan.shape

        # Verify that horizon matches config
        if horizon != self.config.horizon:
            raise ValueError(
                f"Input plan horizon {horizon} does not match config.horizon {self.config.horizon}")

        # Get total number of timesteps in the diffusion process
        T_total = self.diffusion.noise_scheduler.config.num_train_timesteps

        # For each token j (0 to H-1) in the plan, compute its target timestep
        # using the formula (j+1)*(T_total/H)
        # This creates timesteps [T/H, 2T/H, 3T/H, ..., T]
        target_sde_times = torch.linspace(
            T_total/horizon, T_total, horizon, device=device)

        # Convert to integer timesteps for the DDPM scheduler
        # Round to integers since DDPM timesteps are discrete
        target_ddpm_timesteps = target_sde_times.round().long()

        # Expand to match batch size: [B, H]
        target_ddpm_timesteps = target_ddpm_timesteps.unsqueeze(
            0).expand(batch_size, -1)

        # Generate random noise
        noise = torch.randn_like(initial_clean_plan)

        # Add noise according to the timestep schedule
        # For diffusers' add_noise, need to flatten the timesteps to match the batch dimension
        flattened_clean_plan = initial_clean_plan.reshape(
            -1, initial_clean_plan.shape[-1])
        flattened_noise = noise.reshape(-1, noise.shape[-1])
        flattened_timesteps = target_ddpm_timesteps.reshape(-1)

        # Add noise for each (sample, timestep) pair
        noised_flattened = self.diffusion.noise_scheduler.add_noise(
            flattened_clean_plan,
            flattened_noise,
            flattened_timesteps
        )

        # Reshape back to [B, H, D]
        async_noised_plan = noised_flattened.reshape(batch_size, horizon, -1)

        return async_noised_plan

    @torch.no_grad()
    def sample_cl_diffphycon_step(
        self,
        prev_async_noisy_sequence_at_T_div_H: torch.Tensor,  # From previous step or init
        current_env_state_feedback: torch.Tensor,  # Actual u_env,τ-1
    ) -> tuple[torch.Tensor, torch.Tensor]:  # (current_step_output_z_tau_at_0, noisy_sequence_for_next_step)
        """
        Perform a single step of the CL-DiffPhyCon algorithm.

        This implements Algorithm 1, lines 3-10 from the CL-DiffPhyCon paper. It:
        1. Denoises the current noisy sequence to get the clean output for the current step
        2. Prepares the noisy sequence for the next physical time step by shifting and adding new noise

        Args:
            prev_async_noisy_sequence_at_T_div_H: The noisy trajectory from previous step, at time T/H
                for each token, shape (B, H, target_dim)
            current_env_state_feedback: Current environmental state for conditioning, shape (B, state_dim)
                or dictionary of observation tensors

        Returns:
            tuple containing:
                - current_step_output_z_tau_at_0: Clean output for current step τ, shape (B, target_dim)
                - noisy_sequence_for_next_step: Prepared noisy sequence for step τ+1, shape (B, H, target_dim)
        """
        # Prepare global conditioning properly from environmental state feedback
        # Check if current_env_state_feedback is already a dictionary
        if isinstance(current_env_state_feedback, dict):
            # It's already a dictionary of observation tensors
            obs_dict_for_conditioning = current_env_state_feedback
        else:
            # It's a raw state tensor, convert to proper observation dict
            batch_size = prev_async_noisy_sequence_at_T_div_H.shape[0]
            device = prev_async_noisy_sequence_at_T_div_H.device

            # Create observation dict with proper sequence length for conditioning
            from lerobot.common.constants import OBS_ROBOT

            # Create observations with proper sequence dimension [B, n_obs_steps, D]
            # Repeat the current state for all observation steps
            obs_state = current_env_state_feedback.unsqueeze(1).expand(
                -1, self.config.n_obs_steps, -1)

            obs_dict_for_conditioning = {OBS_ROBOT: obs_state}

            # Add image observations if configured and available
            if hasattr(self, '_queues') and self.config.image_features and 'observation.images' in self._queues:
                if len(self._queues['observation.images']) == self.config.n_obs_steps:
                    # Stack image observations from queue
                    img_obs = torch.stack(
                        list(self._queues['observation.images']), dim=0)
                    # Add batch dimension if needed and move to correct device
                    if img_obs.dim() == 4:  # [S, C, H, W]
                        img_obs = img_obs.unsqueeze(0).expand(
                            batch_size, -1, -1, -1, -1).to(device)
                    obs_dict_for_conditioning['observation.images'] = img_obs

        # Process observations to get proper global conditioning
        global_cond = self.diffusion._prepare_global_conditioning(
            obs_dict_for_conditioning)

        # Get the time step corresponding to T/H
        timestep_T_div_H = int(
            self.diffusion.noise_scheduler.config.num_train_timesteps // self.config.horizon)

        # Get the fully denoised sequence for the current step
        # This performs the denoising from time T/H to time 0 for the entire sequence
        denoised_sequence_at_t0 = self.diffusion.sample_asynchronous_step(
            prev_async_noisy_sequence_at_T_div_H,  # x_t - noisy input
            timestep_T_div_H,  # timestep T/H
            global_cond,       # Properly prepared global conditioning
            num_async_inference_steps=getattr(
                self.config, 'num_async_inference_steps', None)
        )

        # Extract the first element which is the current step output z_τ(0)
        current_step_output_z_tau_at_0 = denoised_sequence_at_t0[:, 0, :]

        # Prepare noisy sequence for the next physical time step
        # 1. Shift left: take elements 1 to H-1 from denoised_sequence
        shifted_sequence = denoised_sequence_at_t0[:, 1:, :]

        # 2. Create new last element z_τ+H(T) with pure Gaussian noise
        batch_size = prev_async_noisy_sequence_at_T_div_H.shape[0]
        target_dim = prev_async_noisy_sequence_at_T_div_H.shape[2]
        new_last_element_noise = torch.randn(
            (batch_size, 1, target_dim),
            device=prev_async_noisy_sequence_at_T_div_H.device
        )

        # 3. Concatenate to form the sequence for the next step
        noisy_sequence_for_next_step = torch.cat(
            [shifted_sequence, new_last_element_noise],
            dim=1
        )

        return current_step_output_z_tau_at_0, noisy_sequence_for_next_step

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
        # Determine batch_size from a key that is expected to be in obs_dict for conditioning
        if OBS_ROBOT in obs_dict and obs_dict[OBS_ROBOT] is not None:
            batch_size = obs_dict[OBS_ROBOT].shape[0]
        elif "observation.images" in obs_dict and obs_dict["observation.images"] is not None:
            batch_size = obs_dict["observation.images"].shape[0]
        elif OBS_ENV in obs_dict and obs_dict[OBS_ENV] is not None:
            batch_size = obs_dict[OBS_ENV].shape[0]
        else:  # Fallback, assumes obs_dict is not empty
            if not obs_dict:
                raise ValueError("obs_dict is empty in predict_action")
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

    def _prepare_batch_for_cond_logic(self, normalized_batch_inputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Helper to prepare the batch specifically for _prepare_global_conditioning."""
        batch_for_global_cond = {}
        cond_horizon = self.config.n_obs_steps

        # Process OBS_ROBOT (state) for conditioning
        if OBS_ROBOT in normalized_batch_inputs and self.config.robot_state_feature:
            # [B, S_loaded, Dim]
            state_input_full = normalized_batch_inputs[OBS_ROBOT]
            # Slice to the required conditioning horizon.
            # Assumes the *first* cond_horizon steps of the loaded OBS_ROBOT are for conditioning.
            # This aligns with how LeRobotDataset typically loads based on observation_delta_indices.
            if state_input_full.shape[1] < cond_horizon:
                raise ValueError(
                    f"Full state input seq len {state_input_full.shape[1]} < cond_horizon {cond_horizon} for {OBS_ROBOT}")
            batch_for_global_cond[OBS_ROBOT] = state_input_full[:,
                                                                :cond_horizon, :]

        # Process image features for conditioning
        if self.config.image_features:
            list_of_image_tensors_for_cond = []
            # `self.config.image_features` is a dict like {"obs.image.cam1": FeatureSpec}
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
                # Stack to create NumCameras dimension: [B, cond_horizon, N_cam, C, H, W]
                batch_for_global_cond["observation.images"] = torch.stack(
                    list_of_image_tensors_for_cond, dim=2)

        # Process OBS_ENV for conditioning
        if OBS_ENV in normalized_batch_inputs and self.config.env_state_feature:
            # [B, S_env_loaded, Dim]
            env_input_full = normalized_batch_inputs[OBS_ENV]
            if env_input_full.shape[1] < cond_horizon:
                raise ValueError(
                    f"Full env input seq len {env_input_full.shape[1]} < cond_horizon {cond_horizon} for {OBS_ENV}")
            batch_for_global_cond[OBS_ENV] = env_input_full[:,
                                                            :cond_horizon, :]

        return batch_for_global_cond

    def forward(self, batch: dict[str, Tensor]) -> Tensor:
        # `normalize_inputs` operates on `batch` using `self.config.input_features`.
        # `LeRobotDataset` loads data for each key in `input_features` according to `cfg.observation_delta_indices` (length `n_obs_steps`).
        # And for keys in `output_features` according to `cfg.target_delta_indices` (length `horizon`).
        # If a key is in both (e.g. "observation.state"), `delta_timestamps` makes it load the union.
        # `normalize_inputs` will thus see "observation.state" with 18 steps.
        normalized_batch_inputs = self.normalize_inputs(batch)

        # This helper will slice inputs down to `config.n_obs_steps` for conditioning
        batch_for_cond = self._prepare_batch_for_cond_logic(
            normalized_batch_inputs)

        # `normalize_targets` operates on `batch` using `self.config.output_features`.
        # `batch[self.config.diffusion_target_key]` will have `horizon` steps.
        normalized_targets_batch = self.normalize_targets(batch)

        # `final_normalized_batch` is used for `batch_info_for_masking` and to get the diffusion target.
        # It needs the *original target sequence length* for the diffusion_target_key.
        # Start with correctly sliced conditioning data
        final_normalized_batch = dict(batch_for_cond)
        for key, val in normalized_targets_batch.items():  # Add normalized targets
            if key in self.config.output_features:
                final_normalized_batch[key] = val

        global_cond_train = self.diffusion._prepare_global_conditioning(
            batch_for_cond)

        diffusion_target_key = self.config.diffusion_target_key
        if diffusion_target_key not in final_normalized_batch:
            raise KeyError(
                f"Diffusion target key '{diffusion_target_key}' not found in final_normalized_batch. Available keys: {list(final_normalized_batch.keys())}")

        trajectory_to_diffuse = final_normalized_batch[diffusion_target_key]
        if trajectory_to_diffuse.shape[1] != self.config.horizon:
            # This check is important: ensures the target provided to diffusion model is of correct length
            raise ValueError(
                f"Target trajectory for '{diffusion_target_key}' has length {trajectory_to_diffuse.shape[1]}, expected horizon {self.config.horizon}. Check LeRobotDataset loading for output_features.")

        loss = self.diffusion.compute_loss(
            trajectory_to_diffuse,
            global_cond_train,
            batch_info_for_masking=final_normalized_batch
        )
        return loss

    def forward_async(self, batch: dict[str, Tensor]) -> Tensor:
        from model.diffusion.async_training import AsyncDiffusionTrainer

        normalized_batch_inputs = self.normalize_inputs(batch)
        batch_for_global_cond_async = self._prepare_batch_for_cond_logic(
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

        # Should be [B, config.horizon, Dim]
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
