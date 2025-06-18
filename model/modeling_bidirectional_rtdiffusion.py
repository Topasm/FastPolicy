from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
import einops
import time
from typing import Dict

from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.constants import OBS_STATE, OBS_IMAGE, OBS_ENV_STATE
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import PolicyFeature

from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.invdyn.invdyn import MlpInvDynamic


class BidirectionalRTDiffusionPolicy(nn.Module):
    """
    Combined policy class that integrates:
    1. Bidirectional Transformer for state plan generation
    2. Inverse dynamics model for action prediction from states

    Note: The diffusion model is no longer used in the action generation pipeline.
    It is still passed in the constructor for compatibility with existing code.
    """

    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        state_diffusion_model: CLDiffPhyConModel,
        inverse_dynamics_model: MlpInvDynamic,
        dataset_stats: dict,
        all_dataset_features: Dict[str, any],
        n_obs_steps: int
    ):
        super().__init__()
        # Store the transformer - now there's only one version with integrated normalization
        self.bidirectional_transformer = bidirectional_transformer
        self.base_transformer = bidirectional_transformer

        self.state_diffusion_model = state_diffusion_model
        self.inverse_dynamics_model = inverse_dynamics_model
        self.config = state_diffusion_model.config
        self.device = get_device_from_parameters(bidirectional_transformer)

        # These diffusion-related parameters are kept for compatibility
        # but are not used in the modified _generate_state_plan method
        # Set to False since we're bypassing diffusion completely
        self.use_cl_diffphycon = False
        self.cl_diffphycon_latent_z_prev = None

        # Process dataset stats for normalization
        processed_stats = self._process_dataset_stats(dataset_stats)

        # Create normalizers
        self.normalize_inputs, self.unnormalize_action_output = self._create_normalizers(
            all_dataset_features, processed_stats)

        # Initialize queues
        self.n_obs_steps = n_obs_steps

        # Initialize RGB encoder if images are used
        if self.config.image_features:
            from model.diffusion.diffusion_modules import DiffusionRgbEncoder
            self.rgb_encoder = DiffusionRgbEncoder(self.config)
            self.rgb_encoder.to(self.device)

        self.reset()
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

    def _process_dataset_stats(self, dataset_stats):
        """Process raw dataset statistics into tensor format for normalizers."""
        processed_stats = {}
        if dataset_stats is None:
            print("Warning: No dataset_stats provided to BidirectionalRTDiffusionPolicy.")
            return processed_stats

        for key, stat_group in dataset_stats.items():
            processed_stats[key] = {}
            if isinstance(stat_group, dict):
                for subkey, subval in stat_group.items():
                    try:
                        processed_stats[key][subkey] = torch.as_tensor(
                            subval, dtype=torch.float32, device=self.device)
                    except Exception:
                        processed_stats[key][subkey] = subval
            else:
                processed_stats[key] = stat_group

        return processed_stats

    def _create_normalizers(self, all_features, processed_stats):
        """Create input normalizer and action unnormalizer."""
        # Default no-op normalizers
        def normalize_fn(x): return x
        def unnormalize_fn(x): return x.get(
            "action", x) if isinstance(x, dict) else x

        if not hasattr(self.config, 'input_features') or not hasattr(self.config, 'normalization_mapping'):
            print("Warning: Missing required config attributes for normalization.")
            return normalize_fn, unnormalize_fn

        # Store processed stats for later use (adding this for manual unnormalization)
        self.config.dataset_stats = processed_stats

        # Create normalizer for inputs
        try:
            # Prepare input features for normalizer
            valid_input_features = {}
            for k, v_feat in self.config.input_features.items():
                if isinstance(v_feat, dict) and 'type' in v_feat and 'shape' in v_feat:
                    try:
                        valid_input_features[k] = PolicyFeature(
                            type=FeatureType(v_feat['type']),
                            shape=tuple(v_feat['shape']))
                    except Exception as e:
                        print(f"Warning: Could not convert feature {k}: {e}")
                elif hasattr(v_feat, 'type') and hasattr(v_feat, 'shape'):
                    valid_input_features[k] = v_feat

            normalize_fn = Normalize(
                valid_input_features,
                self.config.normalization_mapping,
                processed_stats
            )
        except Exception as e:
            print(f"Error creating input normalizer: {e}")

        # Create unnormalizer for actions
        try:
            action_feature_data = all_features.get("action")

            # Debug info about action feature and stats
            if "action" in processed_stats:
                print(
                    f"Action stats available - mean shape: {processed_stats['action'].get('mean').shape if 'mean' in processed_stats['action'] else 'N/A'}")
                print(
                    f"Action stats available - std shape: {processed_stats['action'].get('std').shape if 'std' in processed_stats['action'] else 'N/A'}")
                print(f"Action mean: {processed_stats['action'].get('mean')}")
                print(f"Action std: {processed_stats['action'].get('std')}")
            else:
                print("WARNING: No 'action' key in processed_stats!")

            if action_feature_data:
                if isinstance(action_feature_data, dict) and 'type' in action_feature_data and 'shape' in action_feature_data:
                    # Create a PolicyFeature object from the action feature data
                    action_feature = PolicyFeature(
                        type=FeatureType(action_feature_data['type']),
                        shape=tuple(action_feature_data['shape'])
                    )
                    # Create the unnormalizer specifically for the action
                    unnormalize_fn = Unnormalize(
                        {"action": action_feature},
                        self.config.normalization_mapping,
                        processed_stats
                    )
                    print(
                        f"Successfully created action unnormalizer with shape {action_feature_data['shape']}")

                    # Test unnormalization with a dummy input to verify it's working
                    dummy_action = torch.tensor(
                        [[0.0, 0.0]], device=self.device)
                    try:
                        dummy_result = unnormalize_fn({"action": dummy_action})
                        if isinstance(dummy_result, dict) and torch.allclose(dummy_result["action"], dummy_action):
                            print(
                                "WARNING: Test unnormalization didn't change values!")
                        else:
                            print("Test unnormalization successful!")
                    except Exception as test_e:
                        print(
                            f"Warning: Test unnormalization failed: {test_e}")

                elif hasattr(self.config, 'action_feature'):
                    # Alternatively use action_feature from the config if available
                    unnormalize_fn = Unnormalize(
                        {"action": self.config.action_feature},
                        self.config.normalization_mapping,
                        processed_stats
                    )
                    print("Created action unnormalizer from config.action_feature")
        except Exception as e:
            print(f"Error creating action unnormalizer: {e}")
            print("Using identity function for action unnormalization")

        return normalize_fn, unnormalize_fn

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()
        self.cl_diffphycon_latent_z_prev = None

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """Select an action given the current observation."""
        # Move tensors to device
        raw_obs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in current_raw_observation.items()}

        # Process and normalize observation
        processed_obs = self._preprocess_observation(raw_obs)
        normalized_obs = self._normalize_observation(processed_obs)

        self._queues = populate_queues(
            self._queues, normalized_obs)
        # Generate new action plan only when the action queue is empty
        if len(self._queues["action"]) == 0:
            # Check if we have enough history to make a prediction
            required_obs = self.config.n_obs_steps
            if len(self._queues["observation.state"]) < required_obs:
                print(
                    f"Warning: Not enough history in queues. Have {len(self._queues['observation.state'])}, need {required_obs}")
                # Return zero action if we don't have enough history yet
                return torch.zeros((1, self.config.action_feature.shape[0]), device=self.device)

            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

        # If we already have actions queued, return the next one
        if self._action_execution_queue:
            return self._get_next_action()

        # Get the last state in the sequence
        norm_state = model_input_batch["observation.state"][:, -1, :]

        # Generate state plan using only the transformer (diffusion bypassed)
        transformer_state_plan = self._generate_state_plan(model_input_batch)

        # Log state plan details for debugging
        print(f"Transformer state plan: min={transformer_state_plan.min().item():.4f}, "
              f"max={transformer_state_plan.max().item():.4f}, "
              f"mean={transformer_state_plan.mean().item():.4f}")

        # Generate actions using inverse dynamics directly from transformer predictions
        actions = self._generate_actions_from_states(
            norm_state, transformer_state_plan)

        # Queue up actions for execution
        for i in range(actions.shape[1]):
            # Store each action as a tensor of shape [action_dim], not [batch_size, action_dim]
            # This way the queue contains simple action vectors
            self._action_execution_queue.append(
                actions[0, i, :])  # Assuming batch_size=1

        # Print queue size for debugging
        print(
            f"Queued {len(self._action_execution_queue)} actions for execution")

        # Return the first action
        if self._action_execution_queue:
            return self._get_next_action()
        else:
            # Create zero action directly
            batch_size = norm_state.shape[0]
            action_dim = getattr(self.inverse_dynamics_model, 'a_dim',
                                 getattr(self.config.action_feature, 'shape', [2])[0])
            return torch.zeros((batch_size, action_dim), device=self.device)

    def _preprocess_observation(self, raw_obs):
        """Preprocess raw observations, particularly adjusting image dimensions."""
        processed_obs = {}
        for k, v in raw_obs.items():
            if k == "observation.image" and v is not None:
                # Handle image format conversions
                if v.ndim == 4:  # [B, H, W, C]
                    # Convert uint8 to float32 and normalize to [0,1] range first
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0
                    processed_obs[k] = v.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                elif v.ndim == 3:  # [H, W, C]
                    # Convert uint8 to float32 and normalize to [0,1] range first
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0
                    processed_obs[k] = v.permute(
                        2, 0, 1).unsqueeze(0)  # -> [1, C, H, W]
                else:
                    processed_obs[k] = v
            else:
                processed_obs[k] = v

        return processed_obs

    def _normalize_observation(self, processed_obs):
        """Normalize preprocessed observations."""
        try:
            return self.normalize_inputs(processed_obs)
        except Exception as e:
            print(f"Error during normalization: {e}")
            return processed_obs  # Fallback to unnormalized

    def _get_next_action(self):
        """Get the next normalized action from the queue and unnormalize it."""
        next_action = self._action_execution_queue.popleft()
        print(f"Action shape before unsqueeze: {next_action.shape}")

        # Handle different shapes - ensure it's a batch
        if len(next_action.shape) == 1:
            next_action = next_action.unsqueeze(0)
        print(f"Action shape after unsqueeze: {next_action.shape}")

        # Use the standard unnormalizer
        try:
            # Make sure we're passing a properly formatted dictionary to unnormalize_action_output
            action_dict = {"action": next_action}
            unnorm_action = self.unnormalize_action_output(action_dict)

            # Check if the result is a dictionary or tensor
            if isinstance(unnorm_action, dict) and "action" in unnorm_action:
                unnorm_action_tensor = unnorm_action["action"]
                print(f"Unnormalized action: {unnorm_action_tensor}")
            else:
                unnorm_action_tensor = unnorm_action
                print(f"Unnormalizer returned type: {type(unnorm_action)}")

            return unnorm_action_tensor

        except Exception as e:
            print(f"Error using unnormalizer: {e}")
            print("Returning original action as fallback")
            return next_action

    def _generate_state_plan(self, model_input_batch):
        """
        Generates a state plan directly from the bidirectional transformer,
        bypassing any diffusion model refinement.
        The returned plan represents predicted future states (e.g., s_1, s_2, ..., s_K).

        Args:
            model_input_batch: Dict containing observation history with keys like "observation.state", "observation.image"
        """
        # 1. Extract current observations and prepare for Bidirectional Transformer
        # model_input_batch already contains normalized, batched, and device-mapped history
        # Current state s_0
        norm_state_current = model_input_batch["observation.state"][:, -1, :]
        norm_img_current = None
        if "observation.image" in model_input_batch and model_input_batch["observation.image"] is not None:
            # Extract last image from the sequence
            # Current image i_0
            norm_img_current = model_input_batch["observation.image"][:, -1, :, :, :]

        # Prepare state for bidirectional transformer
        bidir_state_dim = self.base_transformer.config.state_dim
        current_state_for_bidir_transformer = norm_state_current
        if norm_state_current.shape[-1] != bidir_state_dim:
            current_state_for_bidir_transformer = norm_state_current[:,
                                                                     :bidir_state_dim]

        # Check if image is required but missing
        bidir_uses_image = any(k.startswith("observation.image")
                               for k in getattr(self.base_transformer.config, 'input_features', {}))
        if bidir_uses_image and norm_img_current is None:
            raise ValueError(
                "BidirectionalARTransformer requires an image but none is available")

        # 2. Get state plan directly from Bidirectional Transformer
        print("Generating state plan directly from Bidirectional Transformer (diffusion bypassed)")
        transformer_predictions = self.base_transformer(
            initial_images=norm_img_current,  # This is current image i_0
            initial_states=current_state_for_bidir_transformer,  # This is current state s_0
            training=False
        )

        # 'predicted_forward_states' from transformer is [s_1_pred, s_2_pred, ..., s_{F-1}_pred]
        # This is the plan of future states that _generate_actions_from_states expects.
        transformer_predicted_future_states = transformer_predictions['predicted_forward_states']

        print(
            f"Generated state plan of shape {transformer_predicted_future_states.shape}")

        # Visualize the transformer predictions (only occasionally to avoid cluttering the output directory)
        if torch.rand(1).item() < 0.1:  # 10% chance to visualize
            self.visualize_transformer_predictions(
                transformer_predicted_future_states,
                current_state_for_bidir_transformer
            )

        return transformer_predicted_future_states

    def visualize_transformer_predictions(self, state_plan, current_state=None):
        """
        Visualize the transformer state predictions for debugging purposes.

        Args:
            state_plan: The state plan from transformer [batch_size, seq_len, state_dim]
            current_state: Optional current state for reference [batch_size, state_dim]
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            from pathlib import Path

            # Create output directory if needed
            output_dir = Path("outputs/viz")
            output_dir.mkdir(exist_ok=True, parents=True)

            # Convert tensors to numpy arrays
            plan_np = state_plan.detach().cpu().numpy()

            # Flatten batch dimension if it's 1
            if plan_np.shape[0] == 1:
                plan_np = plan_np.squeeze(0)  # [seq_len, state_dim]

            # Plot state dimensions over time
            plt.figure(figsize=(10, 5))

            # Get number of state dimensions to plot (limit to first 2-4 for clarity)
            state_dim = min(4, plan_np.shape[1])

            # Plot each state dimension over time
            for i in range(state_dim):
                plt.plot(plan_np[:, i], label=f'State dim {i}')

            if current_state is not None:
                current_np = current_state.detach().cpu().numpy()
                if current_np.shape[0] == 1:
                    current_np = current_np.squeeze(0)
                # Add points for current state
                for i in range(state_dim):
                    plt.scatter(0, current_np[i], marker='o', color=f'C{i}')

            plt.title('Transformer State Predictions')
            plt.xlabel('Time step')
            plt.ylabel('State value')
            plt.legend()
            plt.grid(True)

            # Save the plot
            timestamp = int(time.time())
            plt.savefig(output_dir / f'transformer_plan_{timestamp}.png')
            print(
                f"Saved state plan visualization to outputs/viz/transformer_plan_{timestamp}.png")
            plt.close()

        except Exception as e:
            print(f"Failed to visualize transformer predictions: {e}")

    def _generate_actions_from_states(self, norm_state, state_plan):
        """
        Generate actions from state plan using inverse dynamics.

        Args:
            norm_state: Current normalized state [batch_size, state_dim]
            state_plan: Future state predictions from transformer [batch_size, seq_len, state_dim]

        Returns:
            Tensor of shape [batch_size, seq_len-1, action_dim] with actions to transition between states
        """
        # Prepare state dimensions for inverse dynamics
        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim

        # Adjust current state if needed
        current_state = norm_state
        if current_state.shape[-1] != inv_dyn_state_dim:
            current_state = current_state[:, :inv_dyn_state_dim]

        # Handle transformer output (should already be [batch_size, seq_len, state_dim])
        plan_for_invdyn = state_plan
        print(f"State plan shape: {plan_for_invdyn.shape}")

        # Check if plan is just a single state rather than a sequence
        if len(plan_for_invdyn.shape) == 2:  # [batch_size, state_dim]
            print(
                f"Plan is single state shape {plan_for_invdyn.shape}, expanding to sequence with len=1")
            # Make it a sequence of length 1: [batch_size, 1, state_dim]
            plan_for_invdyn = plan_for_invdyn.unsqueeze(1)

        # Adjust dimensions if needed
        if plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            if plan_for_invdyn.shape[-1] < inv_dyn_state_dim:
                # Pad if too small
                padding_size = inv_dyn_state_dim - plan_for_invdyn.shape[-1]
                padding = torch.zeros(
                    plan_for_invdyn.shape[0],
                    plan_for_invdyn.shape[1],
                    padding_size,
                    device=self.device
                )
                plan_for_invdyn = torch.cat([plan_for_invdyn, padding], dim=-1)
            else:
                # Truncate if too large
                plan_for_invdyn = plan_for_invdyn[:, :, :inv_dyn_state_dim]

        # Check if we have a valid plan
        num_planned_states = plan_for_invdyn.shape[1]
        if num_planned_states == 0:
            print("No planned states available")
            return torch.zeros((norm_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)

        # Generate actions from state pairs
        actions_list = []
        current_state_t = current_state

        for i in range(num_planned_states):
            next_state = plan_for_invdyn[:, i, :]
            state_pair = torch.cat([current_state_t, next_state], dim=-1)
            action_i = self.inverse_dynamics_model(state_pair)
            actions_list.append(action_i)
            current_state_t = next_state

        # Stack actions into sequence
        if actions_list:
            actions = torch.stack(actions_list, dim=1)
            print(f"Generated actions shape: {actions.shape}")
            return actions
        else:
            return torch.zeros((norm_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)

    # Other helper methods and implementations...
