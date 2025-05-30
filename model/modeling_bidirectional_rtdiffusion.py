from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict

from lerobot.common.policies.utils import get_device_from_parameters
# Assuming populate_queues might be used or adapted from, but not strictly in this version
# from lerobot.common.policies.utils import populate_queues
from lerobot.common.policies.normalize import Normalize, Unnormalize
# For type hinting if needed, not for direct use here
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
from model.diffusion.modeling_clphycon import CLDiffPhyConModel  # state_diffusion_model
from model.invdyn.invdyn import MlpInvDynamic
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE


class BidirectionalRTDiffusionPolicy(nn.Module):
    """
    Refactored combined policy:
    1. BidirectionalARTransformer: current_obs -> initial_future_state_path
    2. StateDiffusionModel (CLDiffPhyConModel): initial_path + obs_history -> refined_future_state_path
    3. MlpInvDynamic: refined_path + current_state -> action_sequence
    Normalization is handled by instances created within this policy.
    """

    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        state_diffusion_model: CLDiffPhyConModel,
        inverse_dynamics_model: MlpInvDynamic,
        # Expects raw dataset_stats (e.g., from LeRobotDatasetMetadata.stats)
        dataset_stats: dict,
        n_obs_steps: int
    ):
        super().__init__()
        self.bidirectional_transformer = bidirectional_transformer
        self.state_diffusion_model = state_diffusion_model
        self.inverse_dynamics_model = inverse_dynamics_model
        # Use the config from state_diffusion_model as it's central to conditioning and action specs
        self.config = state_diffusion_model.config
        self.device = get_device_from_parameters(
            bidirectional_transformer)  # Or any main model component

        # Process dataset_stats to be on the correct device
        processed_dataset_stats = {}
        if dataset_stats is not None:
            for key, stat_group in dataset_stats.items():
                processed_dataset_stats[key] = {}
                for subkey, subval in stat_group.items():
                    try:
                        processed_dataset_stats[key][subkey] = torch.as_tensor(
                            subval, dtype=torch.float32, device=self.device
                        )
                    except Exception as e:
                        # print(f"Warning: Could not convert stat {key}.{subkey} to tensor: {e}. Using original value.")
                        processed_dataset_stats[key][subkey] = subval
        else:
            print("Warning: No dataset_stats provided to BidirectionalRTDiffusionPolicy. Normalization may be incorrect.")

        # Create normalizers using the processed stats and config from state_diffusion_model
        # Ensure config has necessary attributes like input_features, action_feature, normalization_mapping
        if hasattr(self.config, 'input_features') and \
           hasattr(self.config, 'normalization_mapping') and \
           hasattr(self.config, 'action_feature'):
            self.normalize_inputs = Normalize(
                self.config.input_features,
                self.config.normalization_mapping,
                processed_dataset_stats
            )
            self.unnormalize_action_output = Unnormalize(
                # Unnormalize only the action
                {"action": self.config.action_feature},
                self.config.normalization_mapping,
                processed_dataset_stats
            )
            # print("Successfully created normalizers in BidirectionalRTDiffusionPolicy")
        else:
            print(
                "Warning: Missing attributes in config for normalizer creation. Using identity normalizers.")
            self.normalize_inputs = lambda x: x  # Identity function
            self.unnormalize_action_output = lambda x: x.get(
                "action", x) if isinstance(x, dict) else x  # Identity

        self.n_obs_steps = n_obs_steps
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()

    def reset(self):
        """Clear all observation and action queues."""
        self._obs_image_queue.clear()
        self._obs_state_queue.clear()
        self._action_execution_queue.clear()

        if hasattr(self.state_diffusion_model, 'reset'):
            self.state_diffusion_model.reset()

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """
        Full pipeline with refined normalization handling.
        """
        # Ensure raw observation tensors are on the correct device before normalization
        # self.normalize_inputs expects a dictionary of tensors.
        # The keys in current_raw_observation should match those in self.config.input_features
        raw_obs_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in current_raw_observation.items()
        }

        # 1. Normalize Current Observation using self.normalize_inputs
        # This returns a dict with normalized tensors, e.g., normalized_obs["observation.state"]
        normalized_obs_batch = self.normalize_inputs(raw_obs_on_device)

        # Extract specific normalized observations for pipeline steps
        # Ensure keys match what `self.config.input_features` defined.
        # Example: if input_features = {"observation.state": ..., "observation.image.cam1": ...}
        # then normalized_obs_batch will contain these keys.

        # For BidirectionalARTransformer input:
        # It needs a single image tensor and a single state tensor.
        # If multiple cameras, decide how to handle (e.g. use first, or BidirectionalARTransformer handles multiple)
        # For simplicity, assume BidirARTransformer uses keys "observation.image" and "observation.state"
        # from its own config (bidirectional_transformer.config.input_features).
        # We need to provide these from our `normalized_obs_batch`.

        # Infer primary state and image keys from the policy's main config (from state_diffusion_model)
        # This assumes these keys are consistently used.
        main_state_key = OBS_ROBOT  # "observation.state"

        # For image, self.config.image_features is a dict. Pick the first one for BidirARTransformer.
        # This is a simplification; BidirARTransformer might have its own specific image input key.
        image_keys_in_config = list(
            self.config.image_features.keys()) if self.config.image_features else []
        main_image_key = image_keys_in_config[0] if image_keys_in_config else None

        if main_state_key not in normalized_obs_batch:
            raise KeyError(
                f"Normalized batch missing required state key: {main_state_key}")
        # Should be [B, StateDim]
        norm_state_current = normalized_obs_batch[main_state_key]

        norm_img_current = None
        if main_image_key and main_image_key in normalized_obs_batch:
            # Should be [B, C, H, W]
            norm_img_current = normalized_obs_batch[main_image_key]
        elif "observation.image" in normalized_obs_batch:  # Fallback to a generic key
            norm_img_current = normalized_obs_batch["observation.image"]
        else:
            # This case should be handled if BidirARTransformer requires an image
            if self.bidirectional_transformer.config.input_features.get("observation.image"):
                raise KeyError(
                    "Normalized batch missing required image key for BidirectionalARTransformer.")

        # Ensure batch dimension exists for queue items (B=1 for single step inference)
        if norm_state_current.ndim == 1:
            norm_state_current = norm_state_current.unsqueeze(0)
        if norm_img_current is not None and norm_img_current.ndim == 3:
            norm_img_current = norm_img_current.unsqueeze(0)

        # Update Observation Queues with current normalized observation
        # Queues store history for state_diffusion_model's conditioning
        self._obs_state_queue.append(
            norm_state_current.unsqueeze(1))  # Store as [B,1,D_state]

        # For image queue, stack all configured image features if multiple, then add to queue.
        # This part feeds into `state_diffusion_model`'s `_prepare_global_conditioning`.
        if self.config.image_features:
            current_all_norm_images_stacked = []
            for img_key in self.config.image_features:
                if img_key in normalized_obs_batch:
                    img_tensor = normalized_obs_batch[img_key]
                    if img_tensor.ndim == 3:
                        img_tensor = img_tensor.unsqueeze(
                            0)  # Ensure batch dim
                    current_all_norm_images_stacked.append(img_tensor)
            if current_all_norm_images_stacked:
                # Stack along a new "camera" dimension for `observation.images`
                # Each item in queue should be [B, 1, N_cam, C, H, W]
                # Here, current_all_norm_images_stacked is list of [B,C,H,W]
                # Stack them to [B, N_cam, C,H,W], then unsqueeze for seq_dim=1
                stacked_cams_for_step = torch.stack(
                    current_all_norm_images_stacked, dim=1)  # [B, N_cam, C, H, W]
                self._obs_image_queue.append(
                    stacked_cams_for_step.unsqueeze(1))  # [B, 1, N_cam, C, H, W]

        if self._action_execution_queue:
            return self.unnormalize_action_output({"action": self._action_execution_queue.popleft()})["action"]

        if len(self._obs_state_queue) < self.n_obs_steps:
            action_dim = self.config.action_feature.shape[0]
            # print(f"Warning: Not enough obs history ({len(self._obs_state_queue)}/{self.n_obs_steps}). Returning zero action.")
            # Return unnormalized zero action
            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]

        # STAGE 1: BidirectionalARTransformer for initial future state path
        bidir_config_state_dim = self.bidirectional_transformer.config.state_dim
        if norm_state_current.shape[-1] != bidir_config_state_dim:
            norm_state_for_bidir = norm_state_current[:,
                                                      :bidir_config_state_dim]
        else:
            norm_state_for_bidir = norm_state_current

        if norm_img_current is None and self.bidirectional_transformer.config.input_features.get("observation.image"):
            raise ValueError(
                "BidirectionalARTransformer requires an image input but it's missing from normalized_obs_batch.")

        transformer_predictions = self.bidirectional_transformer(
            initial_images=norm_img_current,
            initial_states=norm_state_for_bidir,
            training=False
        )
        norm_predicted_future_states = transformer_predictions['predicted_forward_states']
        initial_state_plan_normalized = torch.cat(
            [norm_state_for_bidir.unsqueeze(1), norm_predicted_future_states], dim=1
        )

        # STAGE 2A: State Diffusion Model for state path refinement
        # History from queues:
        # [B, n_obs_steps, StateDim]
        obs_history_state = torch.cat(list(self._obs_state_queue), dim=1)

        observation_batch_for_cond = {OBS_ROBOT: obs_history_state}
        if self.config.image_features and self._obs_image_queue:
            obs_history_img_stacked = torch.cat(
                list(self._obs_image_queue), dim=1)  # [B, n_obs_steps, N_cam, C,H,W]
            observation_batch_for_cond["observation.images"] = obs_history_img_stacked

        diffusion_horizon = self.state_diffusion_model.config.horizon
        initial_state_plan_for_diffusion = initial_state_plan_normalized[:,
                                                                         :diffusion_horizon, :]

        refined_state_plan_normalized = self.state_diffusion_model.diffusion.refine_state_path(
            initial_state_path=initial_state_plan_for_diffusion,
            observation_batch_for_cond=observation_batch_for_cond
        )

        # STAGE 2B: MlpInvDynamic for action generation
        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim
        # Use the *actual current full state* (norm_state_current) for the first state pair with invdyn.
        current_norm_state_for_invdyn = norm_state_current

        if current_norm_state_for_invdyn.shape[-1] != inv_dyn_state_dim:
            current_norm_state_for_invdyn = current_norm_state_for_invdyn[:,
                                                                          :inv_dyn_state_dim]

        refined_plan_for_invdyn = refined_state_plan_normalized
        if refined_plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            if refined_plan_for_invdyn.shape[-1] < inv_dyn_state_dim:
                padding_size = inv_dyn_state_dim - \
                    refined_plan_for_invdyn.shape[-1]
                padding = torch.zeros(
                    refined_plan_for_invdyn.shape[0],
                    refined_plan_for_invdyn.shape[1],
                    padding_size,
                    device=self.device
                )
                refined_plan_for_invdyn = torch.cat(
                    [refined_plan_for_invdyn, padding], dim=-1)
            else:
                refined_plan_for_invdyn = refined_plan_for_invdyn[:,
                                                                  :, :inv_dyn_state_dim]

        num_planned_actions = refined_plan_for_invdyn.shape[1]
        if num_planned_actions == 0:  # Should not happen if horizon > 0
            action_dim = self.config.action_feature.shape[0]
            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]

        # Create state pairs for inverse dynamics model
        # First pair: (current_actual_state, refined_plan_state_0)
        # Subsequent pairs: (refined_plan_state_i, refined_plan_state_{i+1})
        actions_normalized_list = []

        first_state_for_pair = current_norm_state_for_invdyn
        for i in range(num_planned_actions):  # Iterate up to H_diffusion-1 actions
            # This is s'_i
            second_state_for_pair = refined_plan_for_invdyn[:, i, :]

            # [B, 2*inv_dyn_state_dim]
            state_pair = torch.cat(
                [first_state_for_pair, second_state_for_pair], dim=-1)
            action_i_normalized = self.inverse_dynamics_model(
                state_pair)  # [B, action_dim]
            actions_normalized_list.append(action_i_normalized)

            # Update for next iteration: s'_i becomes s_t
            first_state_for_pair = second_state_for_pair

        actions_normalized_sequence = torch.stack(
            actions_normalized_list, dim=1)  # [B, num_planned_actions, action_dim]

        # Unnormalize and queue actions
        # The `unnormalize_action_output` expects a dict {"action": tensor}
        # where tensor is [B, H, ActionDim] or [H, ActionDim]
        # Our `actions_normalized_sequence` is [B, H_actions, ActionDim]
        # If B=1 (typical for eval step), squeeze batch dim for unnormalizer if it expects [H, Dim]
        # However, LeRobot Unnormalizer handles batched inputs.

        # Store normalized actions in queue, unnormalize when popping.
        for i in range(actions_normalized_sequence.shape[1]):
            self._action_execution_queue.append(
                actions_normalized_sequence[:, i, :])  # Store [B, ActionDim]

        if self._action_execution_queue:
            # Pop first normalized action, then unnormalize it
            # [B, ActionDim]
            next_normalized_action = self._action_execution_queue.popleft()
            return self.unnormalize_action_output({"action": next_normalized_action})["action"]
        else:  # Should not be reached if num_planned_actions > 0
            action_dim = self.config.action_feature.shape[0]
            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]
