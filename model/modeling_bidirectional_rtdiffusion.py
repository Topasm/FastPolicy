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
from lerobot.configs.types import FeatureType, NormalizationMode
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
        inverse_dynamics_model: MlpInvDynamic,
        dataset_stats: dict,
        all_dataset_features: Dict[str, any],
        n_obs_steps: int,
        output_features: Dict[str, PolicyFeature] = None,
    ):
        super().__init__()
        # Store the transformer - now there's only one version with integrated normalization
        self.bidirectional_transformer = bidirectional_transformer
        self.base_transformer = bidirectional_transformer

        self.inverse_dynamics_model = inverse_dynamics_model
        self.config = bidirectional_transformer.config
        self.device = get_device_from_parameters(bidirectional_transformer)

        # Store the latest predicted image from the transformer
        self.latest_predicted_image = None

        # Convert string keys to FeatureType enum keys for normalization_mapping
        proper_normalization_mapping = {
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MIN_MAX,
            FeatureType.ACTION: NormalizationMode.MIN_MAX
        }

        # Create normalizers with only observation.state (no image)
        filtered_input_features = {}
        if hasattr(self.config, 'input_features') and 'observation.state' in self.config.input_features:
            # Create a proper PolicyFeature object with type attribute
            state_feature = self.config.input_features['observation.state']
            filtered_input_features['observation.state'] = PolicyFeature(
                type=FeatureType.STATE,
                shape=tuple(state_feature['shape']
                            ) if 'shape' in state_feature else None
            )

        # Override the config's input_features to use only observation.state
        self.config.input_features = filtered_input_features

        self.normalize_inputs = Normalize(
            filtered_input_features,
            proper_normalization_mapping,  # Use the fixed mapping
            dataset_stats
        )
        self.unnormalize_action_output = Unnormalize(
            output_features,
            proper_normalization_mapping,  # Use the fixed mapping
            dataset_stats
        )

        # Initialize queues
        self.n_obs_steps = n_obs_steps

        self.reset()

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")

        self._queues = {
            "observation.image": deque(maxlen=self.config.n_obs_steps),
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }

        self._action_execution_queue = deque()  # 이것은 액션 실행용 큐

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """Select an action given the current observation."""
        # Move tensors to device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in current_raw_observation.items()}

        # # Process and normalize observation
        normalized_obs = self.normalize_inputs(batch)
        # Update observation queues with new observation
        self._queues = populate_queues(self._queues, normalized_obs)

        if len(self._queues["action"]) == 0:
            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

            # Generate state plan using only the transformer (diffusion bypassed)
            transformer_state_plan, predicted_goal_image = self._generate_state_plan(
                model_input_batch)

            self.latest_predicted_image = predicted_goal_image
            print(
                f"Predicted goal image available: {predicted_goal_image is not None}")

            # Generate actions using inverse dynamics directly from transformer predictions
            actions = self._generate_actions_from_states(
                transformer_state_plan)

            # Print queue size for debugging
            print(
                f"Queued {len(self._action_execution_queue)} actions for execution")

            # Unnormalize actions
            actions_unnormalized = self.unnormalize_action_output(
                {"action": actions})["action"]

            self._queues["action"].extend(actions_unnormalized.transpose(0, 1))

        # Pop the next action from the queue
        action = self._queues["action"].popleft()
        return action

    def _generate_state_plan(self, model_input_batch):
        """
        Generates a state plan directly from the bidirectional transformer,
        bypassing any diffusion model refinement.
        The returned plan represents predicted future states (e.g., s_1, s_2, ..., s_K).

        Args:
            model_input_batch: Dict containing observation history with keys like "observation.state", "observation.image"
        """
        initial_images_input = model_input_batch["observation.image"]
        initial_states_input = model_input_batch["observation.state"]

        # 2. Get state plan directly from Bidirectional Transformer
        print("Generating state plan directly from Bidirectional Transformer (diffusion bypassed)")
        print("Generating state plan directly from Bidirectional Transformer (diffusion bypassed)")
        transformer_predictions = self.base_transformer(
            initial_images=initial_images_input,  # Temporal sequence or single image
            initial_states=initial_states_input,  # Temporal sequence or single state
            training=False
        )

        # 'predicted_forward_states' from transformer is [s_1_pred, s_2_pred, ..., s_{F-1}_pred]
        # This is the plan of future states that _generate_actions_from_states expects.
        transformer_predicted_future_states = transformer_predictions['predicted_forward_states']
        # Note: The key is 'predicted_goal_images' (plural) in the transformer output
        predicted_goal_image = transformer_predictions.get(
            'predicted_goal_images', None)

        print(
            f"Generated state plan of shape {transformer_predicted_future_states.shape}")

        return transformer_predicted_future_states, predicted_goal_image

    def _generate_actions_from_states(self, state_plan):
        """
        Generate actions from state plan using inverse dynamics.

        Args:
            state_plan: State predictions from transformer [batch_size, seq_len, state_dim]
                       where the first state (index 0) is the current state
            start: Starting index for state plan slicing (default: 0)
            end: Ending index for state plan slicing (default: 8)

        Returns:
            Tensor of shape [batch_size, seq_len-1, action_dim] with actions
        """

        start = 0
        end = 31
        # Slice the state plan to use only the specified range
        state_plan = state_plan[:, start:end, :]

        # Extract current state and future states
        current_state = state_plan[:, 0, :]
        future_states = state_plan[:, 1:, :]

        # Number of steps to generate actions for
        n_action_steps = future_states.shape[1]

        # Generate actions from state pairs
        actions_list = []
        current_s = current_state

        for i in range(n_action_steps):
            next_s = future_states[:, i]
            # Create state pair (s_t, s_{t+1})
            state_pair = torch.cat([current_s, next_s], dim=-1)
            # Predict action a_t that transitions from s_t to s_{t+1}
            action_i = self.inverse_dynamics_model(state_pair)
            actions_list.append(action_i)
            # Update current state for next iteration
            current_s = next_s

        # Stack actions into a sequence
        return torch.stack(actions_list, dim=1)
