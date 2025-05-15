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


class CombinedCriticPolicy(nn.Module):
    """
    Combined policy class for the diffusion model and inverse dynamics model.
    This class is a simple torch.nn.Module that doesn't require config_class.
    """

    def __init__(self, diffusion_model: MyDiffusionModel, inv_dyn_model: MlpInvDynamic, critic_model=None, num_samples=10):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model
        self.critic_model = critic_model
        self.num_samples = num_samples  # Number of trajectory samples to generate
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
    def select_action(self, curr_state=None, prev_state=None, image=None, batch=None) -> tuple[Tensor, list[Tensor]]:
        """
        Selects an action using the combined policy.

        Args:
            curr_state: Current state tensor (optional)
            prev_state: Previous state tensor (optional)
            image: Image observation tensor (optional)
            batch: Full batch dictionary (used if curr_state is not provided)

        Returns:
            Tuple of (action, trajectories) where:
                action: The selected action to take
                trajectories: List of trajectory sequences that were considered
        """
        # Use either the provided batch or construct one from curr_state, prev_state, and image
        if batch is None and curr_state is not None:
            batch = {"observation.state": curr_state}
            if image is not None:
                batch["observation.image"] = image

        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs using our own normalizer, not the diffusion model's
        try:
            # First try to get normalizer from self
            if hasattr(self, 'normalize_inputs'):
                norm_batch = self.normalize_inputs(batch)
            # Fallback if we don't have our own normalizer
            elif hasattr(self.diffusion_model, 'normalize_inputs'):
                norm_batch = self.diffusion_model.normalize_inputs(batch)
            else:
                # If neither has the normalizer, use batch as-is and print warning
                print("Warning: No normalizer found. Using raw batch.")
                norm_batch = batch
        except Exception as e:
            # If something goes wrong with normalization, try to provide diagnostic info
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

        # Generate new action plan and trajectories
        trajectories = []

        if len(self._queues["action"]) == 0:
            # Check if we have enough history to make a prediction
            required_obs = self.config.n_obs_steps
            if len(self._queues["observation.state"]) < required_obs:
                print(
                    f"Warning: Not enough history in queues. Have {len(self._queues['observation.state'])}, need {required_obs}")
                # Return zero action and empty trajectory list if we don't have enough history yet
                zero_action = torch.zeros(
                    (1, self.config.action_feature.shape[0]), device=self.device)
                return zero_action, []

            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

            try:
                # Use our helper method to generate predicted states and actions all at once
                actions, best_trajectory = self._generate_states_critic_actions(
                    model_input_batch=model_input_batch,
                    batch_size=batch["observation.state"].shape[0]
                )

                # Add best trajectory to list
                trajectories.append(best_trajectory)

                # Unnormalize actions using our own or diffusion model's unnormalizer
                if hasattr(self, 'unnormalize_action_output'):
                    actions_unnormalized = self.unnormalize_action_output(
                        {"action": actions})["action"]
                elif hasattr(self.diffusion_model, 'unnormalize_action_output'):
                    actions_unnormalized = self.diffusion_model.unnormalize_action_output(
                        {"action": actions})["action"]
                else:
                    # If no unnormalizer, just use the actions as they are
                    print("Warning: No unnormalizer found. Using raw actions.")
                    actions_unnormalized = actions

                # Add unnormalized actions to the queue
                for i in range(actions_unnormalized.shape[1]):
                    self._queues["action"].append(actions_unnormalized[:, i])

            except Exception as e:
                print(f"Error during action generation: {e}")
                # Return zero action and a default trajectory in case of an error
                zero_action = torch.zeros(
                    (1, self.config.action_feature.shape[0]), device=self.device)

                # Create a default trajectory with the correct shape based on critic model's expectations
                if self.critic_model is not None and hasattr(self.critic_model, 'horizon') and hasattr(self.critic_model, 'state_dim'):
                    default_horizon = self.critic_model.horizon
                    default_state_dim = self.critic_model.state_dim
                else:
                    default_horizon = 16  # Use the default horizon
                    default_state_dim = batch["observation.state"].shape[-1] if "observation.state" in batch else 2

                default_trajectory = torch.zeros(
                    (1, default_horizon, default_state_dim), device=self.device)

                return zero_action, [default_trajectory]

        # Pop the next action from the queue
        next_action = self._queues["action"].popleft()
        return next_action, trajectories

    @torch.no_grad()
    def _generate_states_and_actions(self, model_input_batch, batch_size):
        """
        Helper method to generate future states and corresponding actions

        Args:
            model_input_batch: Dict with observation history
            batch_size: Batch size for diffusion model

        Returns:
            Normalized actions tensor of shape [batch_size, n_action_steps, action_dim]
        """
        # Prepare global conditioning
        global_cond = self.diffusion_model._prepare_global_conditioning(
            model_input_batch)

        # Generate future states using diffusion
        predicted_states = self.diffusion_model.conditional_sample(
            batch_size=batch_size,
            global_cond=global_cond,
        )

        start = 0
        end = 8

        predicted_states = predicted_states[:, start:end]

        # Generate actions using inverse dynamics
        actions_normalized = []
        n_action_steps = predicted_states.shape[1]

        current_s = predicted_states[:, 0]

        for i in range(n_action_steps-1):
            next_s = predicted_states[:, i+1]
            # Create state pair (s_t, s_{t+1})
            state_pair = torch.cat([current_s, next_s], dim=-1)
            # Predict action
            action_i = self.inv_dyn_model(state_pair)
            actions_normalized.append(action_i)
            # Update current state
            current_s = next_s

        # Stack actions into a sequence
        return torch.stack(actions_normalized, dim=1)

    @torch.no_grad()
    def _generate_states_critic_actions(self, model_input_batch, batch_size):
        """
        Helper method to generate multiple future state sequences, evaluate them with critic model,
        and select the best one to generate actions with inverse dynamics.

        Args:
            model_input_batch: Dict with observation history
            batch_size: Batch size for diffusion model

        Returns:
            Tuple of (actions, best_trajectory) where:
                actions: Normalized actions tensor of shape [batch_size, n_action_steps, action_dim]
                best_trajectory: Best state trajectory selected by the critic
        """
        # Prepare global conditioning
        global_cond = self.diffusion_model._prepare_global_conditioning(
            model_input_batch)

        # Store all trajectories and their scores
        all_trajectories = []
        critic_scores = []

        # Generate multiple samples if we have a critic model
        num_samples = self.num_samples if self.critic_model is not None else 1

        for i in range(num_samples):
            # Generate future states using diffusion
            predicted_states = self.diffusion_model.conditional_sample(
                batch_size=batch_size,
                global_cond=global_cond,
            )

            start = 0
            # Use the critic model's horizon if available, otherwise use default
            if self.critic_model is not None and hasattr(self.critic_model, 'horizon'):
                end = self.critic_model.horizon
                print(f"Using critic model's horizon: {end}")
            else:
                end = 16  # Default to a larger horizon which is what the transformer critic appears to expect
                print(f"Using default horizon: {end}")

            # Ensure we don't exceed the available predicted states
            end = min(end, predicted_states.shape[1])
            print(f"Trajectory shape will be: [batch_size, {end}, state_dim]")

            trajectory = predicted_states[:, start:end]
            all_trajectories.append(trajectory)

            # Score trajectory with critic if available
            if self.critic_model is not None:
                # Get score from critic model
                with torch.no_grad():
                    try:
                        # Try to score the trajectory with the critic
                        score = self.critic_model(
                            trajectory_sequence=trajectory).squeeze()
                        critic_scores.append(score)
                    except ValueError as e:
                        # If there's a shape mismatch, print the error but continue with the sample
                        print(f"Critic scoring failed with error: {e}")
                        # Assign a neutral score
                        critic_scores.append(
                            torch.tensor(0.0, device=self.device))

        # Select best trajectory based on critic scores
        if self.critic_model is not None and len(critic_scores) > 0:
            critic_scores = torch.stack(critic_scores, dim=0)
            # Find index of trajectory with highest score
            best_idx = torch.argmax(critic_scores).item()
            best_trajectory = all_trajectories[best_idx]

            print(
                f"Selected trajectory {best_idx} with score {critic_scores[best_idx].item():.4f}")
        else:
            # If no critic model, just use the first trajectory
            best_trajectory = all_trajectories[0]

        # Generate actions using inverse dynamics on the best trajectory
        actions_normalized = []
        n_action_steps = best_trajectory.shape[1]

        current_s = best_trajectory[:, 0]

        for i in range(n_action_steps-1):
            next_s = best_trajectory[:, i+1]
            # Create state pair (s_t, s_{t+1})
            state_pair = torch.cat([current_s, next_s], dim=-1)
            # Predict action
            action_i = self.inv_dyn_model(state_pair)
            actions_normalized.append(action_i)
            # Update current state
            current_s = next_s

        # Stack actions into a sequence
        actions = torch.stack(actions_normalized, dim=1)

        return actions, best_trajectory
