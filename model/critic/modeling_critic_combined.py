from collections import deque
import torch
import torch.nn as nn
from torch import Tensor

from lerobot.common.policies.utils import (
    get_device_from_parameters,
    populate_queues,
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model.invdynamics.invdyn import MlpInvDynamic
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.critic.ciritic_modules import TransformerCritic
# We're accepting any critic model instance now, no need to import specific types


class CombinedCriticPolicy(nn.Module):
    """
    Combined policy class for the diffusion model and inverse dynamics model.
    This class is a simple torch.nn.Module that doesn't require config_class.
    """

    def __init__(self, diffusion_model: MyDiffusionModel, inv_dyn_model: MlpInvDynamic,
                 critic_model=TransformerCritic, num_samples=4):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model

        # Store the critic model - ensure we have an instance, not a class
        if isinstance(critic_model, torch.nn.Module):
            # If we're given an instance, use it
            self.critic_model = critic_model
        else:
            raise ValueError(
                "critic_model must be an instance of torch.nn.Module")

        # Check if the critic model supports half_horizon (for sequence splitting)
        self.supports_half_horizon = hasattr(self.critic_model, 'half_horizon')

        if self.supports_half_horizon:
            print(
                f"Detected critic model with half_horizon={self.critic_model.half_horizon}")

            # Ensure half_horizon attribute is correctly set
            if hasattr(self.critic_model, 'horizon') and not hasattr(self.critic_model, 'half_horizon'):
                # Set it automatically based on horizon
                self.critic_model.half_horizon = self.critic_model.horizon // 2
                print(
                    f"Setting half_horizon to {self.critic_model.half_horizon}")
        else:
            print("Using standard critic model without half-sequence capability")

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

        For ModernBertCritic models with next sequence prediction:
        - Generates full trajectories (16 steps)
        - Splits them into first half (0-7) and second half (8-15) for critic scoring
        - Uses only the first half (0-7) for action prediction via inverse dynamics

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

        # Process image features - required for TransformerCritic
        processed_batch = dict(norm_batch)

        # Handle different ways images may be provided
        if "observation.image" in norm_batch:
            # Image already present in the right format
            pass
        elif self.config.image_features and all(key in norm_batch for key in self.config.image_features):
            # Stack multiple camera views
            processed_batch["observation.image"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            # If no image is available but critic requires it, warn
            if self.critic_model is not None and hasattr(self.critic_model, 'use_image_context') and self.critic_model.use_image_context:
                print(
                    "WARNING: TransformerCritic requires image features, but none provided in input batch!")
                # What image keys are available vs what we need
                if self.config.image_features:
                    print(
                        f"Required image features: {self.config.image_features}")
                print(f"Available batch keys: {list(norm_batch.keys())}")

        # Populate queues with the latest *normalized* observation
        self._queues = populate_queues(self._queues, processed_batch)

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
                actions = self._generate_states_critic_actions(
                    model_input_batch=model_input_batch,
                    batch_size=batch["observation.state"].shape[0]
                )

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
        return next_action

    @torch.no_grad()
    def _generate_states_critic_actions(self, model_input_batch, batch_size):
        """
        Helper method to generate multiple future state sequences, evaluate them with critic model,
        and select the best one to generate actions with inverse dynamics.

        Args:
            model_input_batch: Dict with observation history (already normalized by select_action)
            batch_size: Batch size for diffusion model

        Returns:
            Tuple of (actions, best_trajectory) where:
                actions: Normalized actions tensor of shape [batch_size, n_action_steps, action_dim]
                best_trajectory: Best state trajectory selected by the critic
        """
        # Prepare global conditioning
        global_cond = self.diffusion_model._prepare_global_conditioning(
            model_input_batch)

        all_trajectories = []
        critic_scores = []
        num_samples_to_generate = self.num_samples if self.critic_model is not None else 1

        # Prepare raw images for the critic if needed
        critic_raw_images = model_input_batch["observation.image"][:, 0, :]

        for i in range(num_samples_to_generate):
            predicted_states = self.diffusion_model.conditional_sample(
                batch_size=batch_size,
                global_cond=global_cond,
            )

            trajectory = predicted_states
            # Add debugging information about trajectory shape
            if i == 0:  # Only print for the first sample to avoid spam
                print(f"Generated trajectory shape: {trajectory.shape}, "
                      f"Expected half_horizon: {self.critic_model.half_horizon if hasattr(self.critic_model, 'half_horizon') else 'N/A'}")

            all_trajectories.append(trajectory)

            if self.critic_model is not None:
                # Get trajectory dimensions
                full_horizon = trajectory.shape[1]

                # Determine if we should use half-horizon based scoring
                if self.supports_half_horizon:
                    half_horizon = self.critic_model.half_horizon

                    # Prepare the trajectory for scoring based on its length
                    if full_horizon >= half_horizon:
                        # We have enough steps for at least the first half
                        first_half = trajectory[:, :half_horizon]

                        # Print diagnostic info for trajectory structure
                        if i == 0:  # Only print for first sample
                            print(
                                f"Trajectory split: Full len={full_horizon}, Using first {half_horizon} steps")

                            # If we have a full trajectory, also check continuity
                            if full_horizon >= 2 * half_horizon:
                                second_half = trajectory[:,
                                                         half_horizon:2*half_horizon]
                                first_end = first_half[:, -1, 0].item()
                                second_start = second_half[:, 0, 0].item()
                                print(
                                    f"First half ends at: {first_end:.4f}, Second half starts at: {second_start:.4f}")

                        try:
                            # Use the score method if available, otherwise fallback to direct call
                            if hasattr(self.critic_model, 'score'):
                                score = self.critic_model.score(
                                    trajectory_sequence=first_half,
                                    raw_images=critic_raw_images,
                                    second_half=False
                                ).squeeze()
                                print(
                                    f"Using score method with first {half_horizon} steps")
                            else:
                                score = self.critic_model(
                                    trajectory_sequence=first_half,
                                    raw_images=critic_raw_images,
                                    second_half=False
                                ).squeeze()
                                print(
                                    f"Using direct call with first {half_horizon} steps")
                        except Exception as e:
                            # Handle any errors in the scoring process
                            print(f"Error in critic scoring: {e}")
                            # Fallback to standard scoring with full trajectory
                            score = self.critic_model(
                                trajectory_sequence=trajectory,
                                raw_images=critic_raw_images
                            ).squeeze()
                            print(
                                "Falling back to full trajectory scoring due to error")
                    else:
                        # Trajectory too short, use what we have
                        print(
                            f"Short trajectory ({full_horizon} steps), using entire trajectory")
                        score = self.critic_model(
                            trajectory_sequence=trajectory,
                            raw_images=critic_raw_images,
                            second_half=False  # Treat as first half
                        ).squeeze()
                else:
                    # Standard scoring for models without half-horizon support
                    if hasattr(self.critic_model, 'score'):
                        # Use score method if available
                        score = self.critic_model.score(
                            trajectory_sequence=trajectory,
                            raw_images=critic_raw_images
                        ).squeeze()
                    else:
                        # Fall back to direct call
                        score = self.critic_model(
                            trajectory_sequence=trajectory,
                            raw_images=critic_raw_images
                        ).squeeze()
                critic_scores.append(score)

        if self.critic_model is not None and len(critic_scores) > 0:
            # Print critic scores for debugging
            scores_for_print = [s.item() if isinstance(s, torch.Tensor) and s.numel() == 1
                                else s.mean().item() if isinstance(s, torch.Tensor)
                                else s for s in critic_scores]
            print(f"Critic scores: {[f'{s:.4f}' for s in scores_for_print]}")

            # Handle the case where critic scores are scalar tensors
            if critic_scores[0].dim() == 0:
                # Convert list of scalar tensors to a tensor of shape [num_samples]
                critic_scores_tensor = torch.stack(
                    critic_scores)  # (num_samples,)
                # Get the index of the best score
                best_index = torch.argmax(
                    critic_scores_tensor).item()  # scalar
                # Select the best trajectory (same for all items in batch)
                best_trajectory = all_trajectories[best_index]
                print(
                    f"Selected best trajectory at index {best_index} with score {critic_scores_tensor[best_index].item():.4f}")
            else:
                # Original logic for batched scores of shape [B, num_samples]
                critic_scores_tensor = torch.stack(
                    critic_scores, dim=1)  # (B, num_samples)
                best_indices = torch.argmax(
                    critic_scores_tensor, dim=1)  # (B,)
                # Select the best trajectory for each item in the batch
                best_trajectory_list = []
                for i in range(batch_size):
                    best_trajectory_list.append(
                        all_trajectories[best_indices[i]][i])
                    print(
                        f"Batch item {i}: Selected trajectory {best_indices[i]} with score {critic_scores_tensor[i, best_indices[i]].item():.4f}")
                best_trajectory = torch.stack(
                    best_trajectory_list, dim=0)  # (B, H, D_state)
        else:
            best_trajectory = all_trajectories[0]

        actions_normalized = []

        # Get the full trajectory length and check if we need to handle partial trajectories
        full_horizon = best_trajectory.shape[1]
        print(
            f"Best trajectory shape before processing: {best_trajectory.shape}")

        # Process the best trajectory for action prediction
        if self.supports_half_horizon:
            half_horizon = self.critic_model.half_horizon

            # Use the first half of the trajectory for action prediction
            if full_horizon >= half_horizon:
                print(
                    f"Using first {half_horizon} steps for action prediction")

                # For action prediction, we only need the first half
                action_trajectory = best_trajectory[:, :half_horizon]

                # For debugging, log trajectory information
                if full_horizon >= 2 * half_horizon:
                    # We have both halves, check for continuity
                    first_half = best_trajectory[:, :half_horizon]
                    second_half = best_trajectory[:,
                                                  half_horizon:2*half_horizon]
                    print(
                        f"First half: {first_half[:, 0, 0].item():.4f} to {first_half[:, -1, 0].item():.4f}")
                    print(
                        f"Second half: {second_half[:, 0, 0].item():.4f} to {second_half[:, -1, 0].item():.4f}")

                best_trajectory = action_trajectory
            else:
                # We have less than half_horizon, use what we have
                print(
                    f"Using all {full_horizon} steps (less than ideal {half_horizon})")
        else:
            # Standard processing - use a fixed range for action prediction (typically 8 steps)
            # Default to 8 steps if available
            action_horizon = min(8, full_horizon)
            best_trajectory = best_trajectory[:, :action_horizon]
            print(f"Using first {action_horizon} steps out of {full_horizon}")

        print(
            f"Processed trajectory shape for action generation: {best_trajectory.shape}")

        n_action_steps = best_trajectory.shape[1]
        current_s = best_trajectory[:, 0]

        # Predict action from s_t to s_{t+1}
        for i in range(n_action_steps - 1):
            next_s = best_trajectory[:, i + 1]
            # Inverse dynamics model expects concatenated (s_t, s_{t+1})
            state_pair = torch.cat([current_s, next_s], dim=-1)
            action_i = self.inv_dyn_model(state_pair)
            actions_normalized.append(action_i)
            current_s = next_s

        if not actions_normalized:  # Handle case where n_action_steps <= 1
            # Return zero actions or handle as per policy design for short horizons
            action_dim = self.config.action_feature["shape"][-1]
            actions = torch.zeros(
                (batch_size, 0, action_dim), device=self.device)
        else:
            actions = torch.stack(actions_normalized, dim=1)
        return actions

    def compute_critic_loss(self, batch: dict, positive_trajectories: torch.Tensor, negative_trajectories: torch.Tensor) -> tuple[Tensor, Tensor]:
        if self.critic_model is None:
            raise ValueError(
                "Critic model is not defined. Cannot compute critic loss.")

        # Normalize the input batch (contains raw observations)
        norm_batch = self.normalize_inputs(batch)

        # Get raw images for the critic (to be processed directly by the critic)
        critic_raw_images = None
        if hasattr(self.critic_model, 'use_image_context') and self.critic_model.use_image_context:
            # Extract current raw image observations
            # The critic usually conditions on the "current" image context.
            # If norm_batch['observation.image'] has a time dimension (from obs history), take the last one.
            image_key = "observation.image"  # Default image key
            # Try to get image_features dictionary if it exists
            if hasattr(self.critic_model, 'image_features'):
                # Get first key from the image_features dictionary
                image_keys = list(self.critic_model.image_features.keys())
                if image_keys:
                    image_key = image_keys[0]

            if image_key in norm_batch:
                img_tensor = norm_batch[image_key]
                # B, T, C, H, W
                if img_tensor.ndim == 5 and img_tensor.shape[1] > 0:
                    # Last image in the sequence
                    critic_raw_images = img_tensor[:, -1]  # B, C, H, W
                elif img_tensor.ndim == 4:  # B, C, H, W
                    critic_raw_images = img_tensor
                else:
                    raise ValueError(
                        f"Unexpected shape for image key '{image_key}': {img_tensor.shape} in norm_batch.")
            else:
                raise ValueError(
                    f"Image key '{image_key}' expected by critic model not in norm_batch.")

            if critic_raw_images is None:
                raise ValueError(
                    "No valid image data found in norm_batch for critic.")

        positive_trajectories = positive_trajectories.to(self.device)
        negative_trajectories = negative_trajectories.to(self.device)

        # Call compute_binary_classification_loss on the critic instance
        if hasattr(self.critic_model, 'compute_binary_classification_loss'):
            # Check if the critic model supports the norm_batch API (advanced critics)
            if hasattr(self.critic_model, 'half_horizon') and hasattr(self.critic_model, 'compute_binary_classification_loss'):
                # For advanced critics with half_horizon support, use norm_batch with raw images
                if critic_raw_images is not None:
                    # Prepare images in the format expected by the critic
                    norm_batch["observation.image"] = critic_raw_images

                # Verify the trajectory shape before passing to critic
                if "observation.state" in norm_batch:
                    traj = norm_batch["observation.state"]
                    print(f"Trajectory shape: {traj.shape}")

                    # Check if we have enough timesteps
                    if hasattr(self.critic_model, 'horizon') and traj.ndim == 3 and traj.shape[1] < self.critic_model.horizon:
                        print(
                            f"Warning: Trajectory has only {traj.shape[1]} steps, model expects {self.critic_model.horizon}")

                # Pass noise parameters for creating negative examples
                loss, accuracy = self.critic_model.compute_binary_classification_loss(
                    norm_batch=norm_batch,
                    noise_params={
                        "base_noise_scale": 0.05,
                        "noise_type": "progressive",
                        "noise_growth_factor": 1.2
                    }
                )
            else:
                # For standard critics, pass trajectories directly
                loss, accuracy = self.critic_model.compute_binary_classification_loss(
                    positive_trajectories=positive_trajectories,
                    negative_trajectories=negative_trajectories,
                    raw_images=critic_raw_images  # Pass raw images directly
                )
        else:
            # Fallback implementation
            raise ValueError(
                "Critic model missing compute_binary_classification_loss method")

        return loss, accuracy
