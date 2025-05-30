from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict  # Added Dict for type hint

from lerobot.common.policies.utils import get_device_from_parameters
from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
# RT-Diffusion model (CLDiffPhyConModel) will be passed as state_diffusion_model
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.invdyn.invdyn import MlpInvDynamic  # Import MlpInvDynamic
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE  # Import constants


class BidirectionalRTDiffusionPolicy(nn.Module):
    """
    Combined policy class that uses:
    1. A bidirectional transformer to predict an initial future state path from images/state.
    2. A state diffusion model (e.g., CLDiffPhyConModel configured for state prediction)
       to refine the initial state path.
    3. An inverse dynamics model (MlpInvDynamic) to generate actions from the refined state path.
    """

    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        state_diffusion_model: CLDiffPhyConModel,  # Renamed for clarity
        inverse_dynamics_model: MlpInvDynamic,    # Added inverse dynamics model
        dataset_stats: dict,
        n_obs_steps: int
    ):
        super().__init__()
        self.bidirectional_transformer = bidirectional_transformer
        self.state_diffusion_model = state_diffusion_model
        self.inverse_dynamics_model = inverse_dynamics_model
        self.stats = dataset_stats
        self.n_obs_steps = n_obs_steps
        self.device = get_device_from_parameters(bidirectional_transformer)

        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()

        # For CL-DiffPhyCon style feedback mechanism (can be removed if not used by state_diffusion_model)
        # self._u_pred_async_rt_diffusion = None # This was for action diffusion

    def reset(self):
        """Clear all observation and action queues."""
        self._obs_image_queue.clear()
        self._obs_state_queue.clear()
        self._action_execution_queue.clear()
        # self._u_pred_async_rt_diffusion = None

        if hasattr(self.state_diffusion_model, 'reset'):
            self.state_diffusion_model.reset()

    def _normalize_observation(self, raw_img: Tensor, raw_state: Tensor) -> Dict[str, Tensor]:
        """Normalizes raw image and state observations."""
        norm_img = raw_img  # Assume raw_img is already [0,1] CHW or handled by BidirTransformer
        # B,C,H,W, 0-1
        if raw_img.max() <= 1.0 and raw_img.min() >= 0.0 and raw_img.ndim == 4 and raw_img.shape[1] == 3:
            # Assuming BidirectionalARTransformer expects images in [-1, 1]
            norm_img = raw_img * 2.0 - 1.0
        # B,C,H,W, but not 0-1, assume already correct
        elif raw_img.ndim == 4 and raw_img.shape[1] == 3:
            norm_img = raw_img
        else:  # HWC, 0-255
            norm_img = (raw_img.float() / 255.0).permute(0,
                                                         3, 1, 2)  # BHWC to BCHW
            norm_img = norm_img * 2.0 - 1.0

        if self.stats and "observation.state" in self.stats:
            state_mean = self.stats["observation.state"]["mean"].to(
                self.device)
            state_std = self.stats["observation.state"]["std"].to(self.device)
            norm_state = (raw_state.to(self.device) - state_mean) / state_std
        else:
            print(
                "Warning: State normalization stats not found. Using raw state for norm_state.")
            norm_state = raw_state.to(self.device)
        return {"image": norm_img.to(self.device), "state": norm_state}

    def _unnormalize_action(self, normalized_action_sequence: Tensor) -> Tensor:
        """Unnormalizes the action sequence."""
        if self.stats and "action" in self.stats:
            action_mean = self.stats["action"]["mean"].to(self.device)
            action_std = self.stats["action"]["std"].to(self.device)
            return normalized_action_sequence * action_std + action_mean
        else:
            print(
                "Warning: Action unnormalization stats not found. Returning normalized actions.")
            return normalized_action_sequence

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """
        Full pipeline:
        1. BidirectionalARTransformer: current_obs -> initial_future_state_path
        2. StateDiffusionModel: initial_future_state_path + obs_history -> refined_future_state_path
        3. MlpInvDynamic: refined_future_state_path + current_state -> action_sequence
        """
        # Ensure input tensors are on the correct device (if not already)
        raw_img_input = current_raw_observation["observation.image"].to(
            self.device)
        raw_state_input = current_raw_observation["observation.state"].to(
            self.device)

        # 1. Prepare Current Observation (Normalize & Update Queues)
        # Assuming current_raw_observation["observation.image"] is [B,C,H,W] or [B,H,W,C]
        # Assuming current_raw_observation["observation.state"] is [B, env_state_dim]
        normalized_obs = self._normalize_observation(
            raw_img_input, raw_state_input)
        # Expected by BidirARTransformer: [B,C,H,W]
        norm_img = normalized_obs["image"]
        # Expected by BidirARTransformer: [B,D_state]
        norm_state = normalized_obs["state"]

        self._obs_image_queue.append(
            norm_img.unsqueeze(1))  # Store as [B,1,C,H,W]
        self._obs_state_queue.append(
            norm_state.unsqueeze(1))  # Store as [B,1,D_state]

        if self._action_execution_queue:
            return self._action_execution_queue.popleft()

        if len(self._obs_state_queue) < self.n_obs_steps:
            action_dim_tensor = self.stats.get("action", {}).get(
                "mean", torch.zeros(2, device=self.device))
            action_dim = action_dim_tensor.shape[0]
            print(
                f"Warning: Not enough obs history ({len(self._obs_state_queue)}/{self.n_obs_steps}). Returning zero action.")
            return torch.zeros((raw_state_input.shape[0], action_dim), device=self.device)

        # STAGE 1: BidirectionalARTransformer for initial future state path
        # Ensure norm_state for bidir_transformer matches its expected state_dim
        bidir_config_state_dim = self.bidirectional_transformer.config.state_dim
        if norm_state.shape[-1] != bidir_config_state_dim:
            # Assuming the BidirARTransformer was trained on a specific part of the state
            # (e.g., first `bidir_config_state_dim` elements)
            norm_state_for_bidir = norm_state[:, :bidir_config_state_dim]
        else:
            norm_state_for_bidir = norm_state

        transformer_predictions = self.bidirectional_transformer(
            initial_images=norm_img,
            initial_states=norm_state_for_bidir,
            training=False
        )
        # norm_predicted_future_states: [B, F-1, D_state_bidir] (st_1 to st_{F-1})
        norm_predicted_future_states = transformer_predictions['predicted_forward_states']
        # initial_state_plan_normalized: [B, F, D_state_bidir] (st_0 to st_{F-1})
        initial_state_plan_normalized = torch.cat(
            [norm_state_for_bidir.unsqueeze(1), norm_predicted_future_states], dim=1
        )

        # STAGE 2A: State Diffusion Model for state path refinement
        obs_history_img = torch.cat(
            list(self._obs_image_queue), dim=1)    # [B, n_obs, C, H, W]
        obs_history_state = torch.cat(
            list(self._obs_state_queue), dim=1)  # [B, n_obs, D_state_full]

        # state_diffusion_model._prepare_global_conditioning expects keys like OBS_ROBOT, OBS_IMAGE
        observation_batch_for_cond = {
            OBS_ROBOT: obs_history_state,
            OBS_IMAGE: obs_history_img
            # Add OBS_ENV if used by state_diffusion_model's config
        }

        diffusion_horizon = self.state_diffusion_model.config.horizon
        # Ensure initial_state_plan_normalized matches the state dim expected by state_diffusion_model
        # The state_diffusion_model refines states in its own configured state space.
        # If bidir_state_dim is different from state_diffusion_model's state_dim,
        # a projection/padding might be needed. For now, assume they match or
        # state_diffusion_model.config.robot_state_feature.shape[0] is bidir_config_state_dim
        if initial_state_plan_normalized.shape[-1] != self.state_diffusion_model.config.robot_state_feature.shape[0]:
            print(
                f"Warning: State dim mismatch between BidirTransformer output ({initial_state_plan_normalized.shape[-1]}) and StateDiffusion input ({self.state_diffusion_model.config.robot_state_feature.shape[0]}). Ensure this is handled or dimensions match.")
            # Example: if state_diffusion expects more dims, pad with zeros or repeat
            # For now, we'll proceed assuming they match or the diffusion model handles it internally via its target key.

        initial_state_plan_for_diffusion = initial_state_plan_normalized[:,
                                                                         :diffusion_horizon, :]

        # refined_state_plan_normalized will be in the state space of state_diffusion_model
        refined_state_plan_normalized = self.state_diffusion_model.refine_state_path(
            initial_state_path=initial_state_plan_for_diffusion,
            observation_batch_for_cond=observation_batch_for_cond
        )  # Output: [B, diffusion_horizon, D_state_diffusion] (normalized)

        # STAGE 2B: MlpInvDynamic for action generation
        # refined_state_plan_normalized contains s'_0, s'_1, ..., s'_{H_diff-1}
        # We need to use the *current actual normalized state* (norm_state) as the first state for invdyn
        # The state dimension for inv_dyn_model is self.inverse_dynamics_model.o_dim
        # Ensure norm_state and refined_state_plan_normalized match this dimension.
        # Typically, MlpInvDynamic is trained on the full robot state.
        # So, norm_state (full dim) and refined_state_plan (potentially partial dim if Bidir was partial)

        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim

        # If refined_state_plan_normalized dim != inv_dyn_state_dim, projection/padding is needed.
        # For this example, let's assume refined_state_plan_normalized output by state_diffusion_model
        # already matches the inv_dyn_state_dim. If not, this is a point of careful handling.
        # And norm_state (full current state) should also be sliced if inv_dyn expects a specific sub-part.
        # However, MlpInvDynamic usually takes the full state.

        # Use the full current normalized state
        current_norm_state_for_invdyn = norm_state
        if current_norm_state_for_invdyn.shape[-1] != inv_dyn_state_dim:
            print(
                f"Warning: Current state dim ({current_norm_state_for_invdyn.shape[-1]}) mismatch for InvDyn ({inv_dyn_state_dim}). Slicing/Padding may be needed.")
            # Simplistic slice
            current_norm_state_for_invdyn = current_norm_state_for_invdyn[:,
                                                                          :inv_dyn_state_dim]

        refined_plan_for_invdyn = refined_state_plan_normalized
        if refined_plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            print(
                f"Warning: Refined plan dim ({refined_plan_for_invdyn.shape[-1]}) mismatch for InvDyn ({inv_dyn_state_dim}). Slicing/Padding may be needed.")
            # Example: if refined plan is 2D (x,y) from Bidir->StateDiff, but InvDyn needs 7D state
            # This would require mapping 2D plan to 7D plan, which is non-trivial and outside scope here.
            # For now, assume state_diffusion_model outputs states compatible with MlpInvDynamic.
            # Or, a simple slice/pad if dimensions are close:
            # Pad with zeros
            if refined_plan_for_invdyn.shape[-1] < inv_dyn_state_dim:
                padding = torch.zeros(refined_plan_for_invdyn.shape[0], refined_plan_for_invdyn.shape[1],
                                      inv_dyn_state_dim - refined_plan_for_invdyn.shape[-1], device=self.device)
                refined_plan_for_invdyn = torch.cat(
                    [refined_plan_for_invdyn, padding], dim=-1)
            else:  # Slice
                refined_plan_for_invdyn = refined_plan_for_invdyn[:,
                                                                  :, :inv_dyn_state_dim]

        num_planned_actions = refined_plan_for_invdyn.shape[1]

        s_prev_list = [current_norm_state_for_invdyn.unsqueeze(1)] + \
                      [refined_plan_for_invdyn[:, i, :].unsqueeze(
                          1) for i in range(num_planned_actions - 1)]
        s_prev_seq = torch.cat(s_prev_list, dim=1)

        s_curr_seq = refined_plan_for_invdyn[:, :num_planned_actions, :]

        inv_dyn_input_seq = torch.cat([s_prev_seq, s_curr_seq], dim=-1)

        B_inv, H_inv, D_inv_pair = inv_dyn_input_seq.shape
        inv_dyn_input_flat = inv_dyn_input_seq.reshape(
            B_inv * H_inv, D_inv_pair)

        actions_normalized_flat = self.inverse_dynamics_model(
            inv_dyn_input_flat)
        actions_normalized_sequence = actions_normalized_flat.reshape(
            B_inv, H_inv, -1)

        actions_unnormalized = self._unnormalize_action(
            actions_normalized_sequence)

        for i in range(actions_unnormalized.shape[1]):
            self._action_execution_queue.append(actions_unnormalized[:, i, :])

        if self._action_execution_queue:
            return self._action_execution_queue.popleft()
        else:
            action_dim_tensor = self.stats.get("action", {}).get(
                "mean", torch.zeros(2, device=self.device))
            action_dim = action_dim_tensor.shape[0]
            return torch.zeros((raw_state_input.shape[0], action_dim), device=self.device)
