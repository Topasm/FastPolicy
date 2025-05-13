from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
import einops

from lerobot.common.policies.utils import (
    get_device_from_parameters,
    populate_queues,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model.invdynamics.invdyn import MlpInvDynamic
from model.diffusion.modeling_mymodel import MyDiffusionModel


class CombinedPolicy(nn.Module):
    """
    Combined policy class for the diffusion model and inverse dynamics model.
    This class is a simple torch.nn.Module that doesn't require config_class.
    """

    def __init__(self, diffusion_model: MyDiffusionModel, inv_dyn_model: MlpInvDynamic):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model
        self.config = diffusion_model.config
        self.device = get_device_from_parameters(diffusion_model)

        # Create our own normalizers since they're not available in diffusion_model
        # Fetch dataset stats from LeRobot
        try:
            metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
            dataset_stats = {}
            for key, stat in metadataset_stats.stats.items():
                dataset_stats[key] = {
                    subkey: torch.as_tensor(
                        subval, dtype=torch.float32, device=self.device)
                    for subkey, subval in stat.items()
                }

            # Create normalizers
            self.normalize_inputs = Normalize(
                self.config.input_features, self.config.normalization_mapping, dataset_stats)
            self.unnormalize_action_output = Unnormalize(
                {"action": self.config.action_feature}, self.config.normalization_mapping, dataset_stats)

            print("Successfully created normalizers in CombinedPolicy")
        except Exception as e:
            print(f"Warning: Failed to create normalizers: {e}")
            # Fallback to empty normalizers that do nothing
            self.normalize_inputs = lambda x: x
            self.unnormalize_action_output = lambda x: x

        # Initialize queues
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def plan_future_states(self, batch: dict[str, Tensor]) -> Tensor:
        """
        Step 1: Planner (diffusion model) samples future states using the current observation

        Args:
            batch: Dictionary with current observations

        Returns:
            Predicted future states tensor [batch_size, horizon, state_dim]
        """
        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs
        try:
            # First try to get normalizer from self
            if hasattr(self, 'normalize_inputs'):
                norm_batch = self.normalize_inputs(batch)
            # Fallback if we don't have our own normalizer
            elif hasattr(self.diffusion_model, 'normalize_inputs'):
                norm_batch = self.diffusion_model.normalize_inputs(batch)
            else:
                # If neither has the normalizer, use batch as-is
                print("Warning: No normalizer found. Using raw batch.")
                norm_batch = batch
        except Exception as e:
            # If something goes wrong with normalization, provide diagnostic info
            print(f"Error during normalization: {e}")
            print(f"Batch keys: {batch.keys()}")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    print(f"{k} shape: {v.shape}, dtype: {v.dtype}")
            raise

        # Stack multiple camera views if necessary
        if self.config.image_features and all(key in norm_batch for key in self.config.image_features):
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(norm_batch)
            processed_batch["observation.image"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            # If image features configured but not all present in batch, just use what we have
            processed_batch = norm_batch

        # Populate queues with the latest *normalized* observation
        self._queues = populate_queues(self._queues, processed_batch)

        # Check if we have enough history to make a prediction
        required_obs = self.config.n_obs_steps
        if len(self._queues["observation.state"]) < required_obs:
            print(
                f"Warning: Not enough history in queues. Have {len(self._queues['observation.state'])}, need {required_obs}")
            # Return zeros if we don't have enough history yet
            return torch.zeros((batch["observation.state"].shape[0], self.config.horizon,
                               self.config.robot_state_feature.shape[0]), device=self.device)

        # Prepare batch for the model by stacking history from queues (already normalized)
        model_input_batch = {}
        for key, queue in self._queues.items():
            if key.startswith("observation"):
                # Ensure tensors are on the correct device before stacking
                queue_list = [item.to(self.device) if isinstance(
                    item, torch.Tensor) else item for item in queue]
                model_input_batch[key] = torch.stack(queue_list, dim=1)

        # Prepare global conditioning for the diffusion model
        batch_size = batch["observation.state"].shape[0]
        global_cond = self.diffusion_model._prepare_global_conditioning(
            model_input_batch)

        # Generate future states using the diffusion model
        predicted_states = self.diffusion_model.conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond,
        )

        start = self.config.n_obs_steps-1
        end = 8

        predicted_states = predicted_states[:, start:end]

        return predicted_states

    @torch.no_grad()
    def predict_action(self, current_state: Tensor, next_state: Tensor) -> Tensor:
        """
        Step 2: Predict action between current state and desired next state

        Args:
            current_state: Current observation state tensor [batch_size, state_dim] (MUST be normalized)
            next_state: Desired next state tensor [batch_size, state_dim] (already normalized)

        Returns:
            Predicted action tensor [batch_size, action_dim]
        """
        # Create state pair (s_t, s_{t+1})
        state_pair = torch.cat([current_state, next_state], dim=-1)

        # Predict action using inverse dynamics model
        action = self.inv_dyn_model(state_pair)

        # Unnormalize action if needed
        if hasattr(self, 'unnormalize_action_output'):
            action_unnormalized = self.unnormalize_action_output(
                {"action": action.unsqueeze(1)})["action"].squeeze(1)
        elif hasattr(self.diffusion_model, 'unnormalize_action_output'):
            action_unnormalized = self.diffusion_model.unnormalize_action_output(
                {"action": action.unsqueeze(1)})["action"].squeeze(1)
        else:
            # If no unnormalizer, use the action as is
            print("Warning: No unnormalizer found. Using raw action.")
            action_unnormalized = action

        return action_unnormalized
