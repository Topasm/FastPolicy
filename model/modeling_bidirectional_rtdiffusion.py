from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
import einops
import time
from typing import Dict

from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE, OBS_ENV
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

        # Convert string keys to FeatureType enum keys for normalization_mapping
        proper_normalization_mapping = {
            FeatureType.VISUAL: NormalizationMode.MEAN_STD,
            FeatureType.STATE: NormalizationMode.MIN_MAX,
            FeatureType.ACTION: NormalizationMode.MIN_MAX
        }

        # Create normalizers with the proper enum-based mapping
        self.normalize_inputs = Normalize(
            self.config.input_features,
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

        # Initialize queues
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        # if self.config.env_state_feature:
        #     self._queues["observation.environment_state"] = deque(
        #         maxlen=self.config.n_obs_steps)

        self.reset()

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")

        # __init__에서 사용하는 self.n_obs_steps와 self.config를 일관되게 사용
        # self.n_obs_steps는 __init__에서 인자로 받아 설정됨
        # self.config는 state_diffusion_model.config에서 옴
        obs_queue_len = self.n_obs_steps
        action_queue_len = self.config.n_action_steps  # DiffusionConfig의 n_action_steps

        self._queues = {
            "observation.state": deque(maxlen=obs_queue_len),
            "action": deque(maxlen=action_queue_len),  # 이 큐의 역할 재고 필요 (아래 참조)
        }
        # self.config는 DiffusionConfig이므로 image_features 존재 여부 확인 방식 변경 가능
        if hasattr(self.config, 'image_features') and self.config.image_features:
            self._queues["observation.image"] = deque(maxlen=obs_queue_len)
        # if hasattr(self.config, 'env_state_feature') and self.config.env_state_feature:
        #     self._queues["observation.environment_state"] = deque(
        #         maxlen=obs_queue_len)

        self._action_execution_queue = deque()  # 이것은 액션 실행용 큐

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """Select an action given the current observation."""
        # Move tensors to device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in current_raw_observation.items()}

        # Process and normalize observation
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
            transformer_state_plan = self._generate_state_plan(
                model_input_batch)

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
        # 1. Extract observations and prepare for Bidirectional Transformer
        # model_input_batch already contains normalized, batched, and device-mapped history

        # Check if transformer uses temporal encoding
        n_obs_steps = getattr(self.base_transformer.config, 'n_obs_steps', 1)

        if n_obs_steps > 1:
            # Use full temporal sequences for temporal transformer
            # [B, n_obs_steps, state_dim]
            norm_state_sequence = model_input_batch["observation.state"]
            norm_img_sequence = None
            if "observation.image" in model_input_batch and model_input_batch["observation.image"] is not None:
                # [B, n_obs_steps, C, H, W]
                norm_img_sequence = model_input_batch["observation.image"]

            # Ensure we have the right number of temporal steps
            if norm_state_sequence.shape[1] != n_obs_steps:
                # If we have more history than needed, take the last n_obs_steps
                if norm_state_sequence.shape[1] > n_obs_steps:
                    norm_state_sequence = norm_state_sequence[:, -
                                                              n_obs_steps:, :]
                    if norm_img_sequence is not None:
                        norm_img_sequence = norm_img_sequence[:, -
                                                              n_obs_steps:, :, :, :]
                else:
                    # If we have less history, pad with the first observation
                    # This should not happen in normal operation, but handle it gracefully
                    print(
                        f"Warning: Expected {n_obs_steps} temporal steps, got {norm_state_sequence.shape[1]}")
                    while norm_state_sequence.shape[1] < n_obs_steps:
                        norm_state_sequence = torch.cat(
                            [norm_state_sequence[:, :1, :], norm_state_sequence], dim=1)
                        if norm_img_sequence is not None:
                            norm_img_sequence = torch.cat(
                                [norm_img_sequence[:, :1, :, :, :], norm_img_sequence], dim=1)

            # Prepare state for bidirectional transformer
            bidir_state_dim = self.base_transformer.config.state_dim
            current_state_for_bidir_transformer = norm_state_sequence
            if norm_state_sequence.shape[-1] != bidir_state_dim:
                current_state_for_bidir_transformer = norm_state_sequence[:,
                                                                          :, :bidir_state_dim]

            # Use temporal sequences
            initial_images_input = norm_img_sequence
            initial_states_input = current_state_for_bidir_transformer
        else:
            # Use single-step observations for non-temporal transformer
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

            # Use single observations
            initial_images_input = norm_img_current
            initial_states_input = current_state_for_bidir_transformer

        # Check if image is required but missing
        bidir_uses_image = any(k.startswith("observation.image")
                               for k in getattr(self.base_transformer.config, 'input_features', {}))
        if bidir_uses_image and initial_images_input is None:
            raise ValueError(
                "BidirectionalARTransformer requires an image but none is available")

        # 2. Get state plan directly from Bidirectional Transformer
        print("Generating state plan directly from Bidirectional Transformer (diffusion bypassed)")
        transformer_predictions = self.base_transformer(
            initial_images=initial_images_input,  # Temporal sequence or single image
            initial_states=initial_states_input,  # Temporal sequence or single state
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

    def _generate_actions_from_states(self, state_plan):
        """
        Generate actions from state plan using inverse dynamics.

        Args:
            state_plan: State predictions from transformer [batch_size, seq_len, state_dim]
                       where the first state (index 0) is the current state

        Returns:
            Tensor of shape [batch_size, seq_len-1, action_dim] with actions to transition between states
        """
        # Extract current state from the first element of state_plan
        current_state = state_plan[:, 0, :]

        # The rest of state_plan (from index 1 onward) contains future states
        future_states = state_plan[:, 1:, :]

        # Prepare state dimensions for inverse dynamics
        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim

        # Adjust current state if needed
        if current_state.shape[-1] != inv_dyn_state_dim:
            current_state = current_state[:, :inv_dyn_state_dim]

        # Handle transformer output
        plan_for_invdyn = future_states
        print(f"Future states shape: {plan_for_invdyn.shape}")

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
            return torch.zeros((current_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)

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
            return torch.zeros((current_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)

    # Other helper methods and implementations...
