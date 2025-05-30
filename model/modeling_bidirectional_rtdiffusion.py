from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional, Dict  # Added Dict for type hint

from lerobot.common.policies.utils import get_device_from_parameters
from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
# CLDiffPhyConModel will be used as the State Diffusion Model
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

    def reset(self):
        """Clear all observation and action queues."""
        self._obs_image_queue.clear()
        self._obs_state_queue.clear()
        self._action_execution_queue.clear()

        if hasattr(self.state_diffusion_model, 'reset'):
            self.state_diffusion_model.reset()

    def _normalize_observation(self, raw_img: Tensor, raw_state: Tensor) -> Dict[str, Tensor]:
        """Normalizes raw image and state observations."""
        norm_img = raw_img
        if raw_img.ndim == 4 and raw_img.shape[1] == 3:
            if raw_img.max() <= 1.0 and raw_img.min() >= 0.0:
                norm_img = raw_img * 2.0 - 1.0
        elif raw_img.ndim == 4 and raw_img.shape[3] == 3:
            norm_img_permuted = raw_img.permute(0, 3, 1, 2)
            if norm_img_permuted.max() <= 1.0 and norm_img_permuted.min() >= 0.0:
                norm_img = norm_img_permuted * 2.0 - 1.0
            elif raw_img.dtype == torch.uint8:
                norm_img = (norm_img_permuted.float() / 255.0) * 2.0 - 1.0
            else:
                norm_img = norm_img_permuted
        elif raw_img.ndim == 3 and raw_img.shape[2] == 3:  # HWC, no batch
            norm_img_permuted = raw_img.permute(
                2, 0, 1).unsqueeze(0)  # Add batch, to BCHW
            if norm_img_permuted.max() <= 1.0 and norm_img_permuted.min() >= 0.0:
                norm_img = norm_img_permuted * 2.0 - 1.0
            elif raw_img.dtype == torch.uint8:
                norm_img = (norm_img_permuted.float() / 255.0) * 2.0 - 1.0
            else:
                norm_img = norm_img_permuted
        else:
            # Attempt to handle B,H,W,C if not already caught
            if raw_img.ndim == 4 and raw_img.shape[-1] == 3:  # B,H,W,C
                norm_img_permuted = raw_img.permute(0, 3, 1, 2)  # B,C,H,W
                if norm_img_permuted.max() <= 1.0 and norm_img_permuted.min() >= 0.0:
                    norm_img = norm_img_permuted * 2.0 - 1.0
                elif raw_img.dtype == torch.uint8:
                    norm_img = (norm_img_permuted.float() / 255.0) * 2.0 - 1.0
                else:
                    norm_img = norm_img_permuted
            else:
                print(
                    f"Warning: Unsupported image shape or channel order for normalization: {raw_img.shape}. Passing as is.")
                norm_img = raw_img  # Pass as is if unsure

        if self.stats and "observation.state" in self.stats:
            state_mean = self.stats["observation.state"]["mean"].to(
                raw_state.device)
            state_std = self.stats["observation.state"]["std"].to(
                raw_state.device)
            norm_state = (raw_state - state_mean) / state_std
        else:
            # print("Warning: State normalization stats not found. Using raw state for norm_state.")
            norm_state = raw_state
        return {"image": norm_img.to(self.device), "state": norm_state.to(self.device)}

    def _unnormalize_action(self, normalized_action_sequence: Tensor) -> Tensor:
        """Unnormalizes the action sequence."""
        if self.stats and "action" in self.stats:
            action_mean = self.stats["action"]["mean"].to(self.device)
            action_std = self.stats["action"]["std"].to(self.device)
            return normalized_action_sequence * action_std + action_mean
        else:
            # print("Warning: Action unnormalization stats not found. Returning normalized actions.")
            return normalized_action_sequence

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """
        Full pipeline:
        1. BidirectionalARTransformer: current_obs -> initial_future_state_path
        2. StateDiffusionModel: initial_future_state_path + obs_history -> refined_future_state_path
        3. MlpInvDynamic: refined_future_state_path + current_state -> action_sequence
        """
        raw_img_input = current_raw_observation["observation.image"].to(
            self.device)
        raw_state_input = current_raw_observation["observation.state"].to(
            self.device)

        normalized_obs = self._normalize_observation(
            raw_img_input, raw_state_input)
        norm_img = normalized_obs["image"]
        norm_state = normalized_obs["state"]

        # Ensure norm_img is 4D (B,C,H,W) and norm_state is 2D (B,Dim) before adding to queue
        if norm_img.ndim == 3:
            norm_img = norm_img.unsqueeze(0)  # Add batch if missing
        if norm_state.ndim == 1:
            norm_state = norm_state.unsqueeze(0)  # Add batch if missing

        self._obs_image_queue.append(
            norm_img.unsqueeze(1))  # Store as [B,1,C,H,W]
        self._obs_state_queue.append(
            norm_state.unsqueeze(1))  # Store as [B,1,D_state]

        if self._action_execution_queue:
            return self._action_execution_queue.popleft()

        if len(self._obs_state_queue) < self.n_obs_steps:
            action_dim_tensor = self.stats.get("action", {}).get(
                "mean", torch.zeros(2, device=self.device))
            action_dim = action_dim_tensor.shape[0] if hasattr(
                action_dim_tensor, 'shape') else 2
            # print(f"Warning: Not enough obs history ({len(self._obs_state_queue)}/{self.n_obs_steps}). Returning zero action.")
            return torch.zeros((raw_state_input.shape[0], action_dim), device=self.device)

        bidir_config_state_dim = self.bidirectional_transformer.config.state_dim
        if norm_state.shape[-1] != bidir_config_state_dim:
            norm_state_for_bidir = norm_state[:, :bidir_config_state_dim]
        else:
            norm_state_for_bidir = norm_state

        transformer_predictions = self.bidirectional_transformer(
            initial_images=norm_img,  # norm_img should be [B,C,H,W]
            # norm_state_for_bidir should be [B, bidir_state_dim]
            initial_states=norm_state_for_bidir,
            training=False
        )
        norm_predicted_future_states = transformer_predictions['predicted_forward_states']
        initial_state_plan_normalized = torch.cat(
            [norm_state_for_bidir.unsqueeze(1), norm_predicted_future_states], dim=1
        )

        obs_history_img = torch.cat(list(self._obs_image_queue), dim=1)
        obs_history_state = torch.cat(list(self._obs_state_queue), dim=1)

        observation_batch_for_cond = {
            OBS_ROBOT: obs_history_state,
            # Renamed key for consistency if _prepare_global_conditioning expects "OBS_IMAGE"
            OBS_IMAGE: obs_history_img
            # However, CLDiffPhyConModel's _prepare_global_conditioning uses "observation.images" (plural)
            # Let's ensure the key matches what _prepare_global_conditioning expects.
            # The current CLDiffPhyConModel uses "observation.images"
        }
        if "OBS_IMAGE" in observation_batch_for_cond:  # Temporary fix if OBS_IMAGE was used
            observation_batch_for_cond["observation.images"] = observation_batch_for_cond.pop(
                "OBS_IMAGE")

        diffusion_horizon = self.state_diffusion_model.config.horizon

        initial_state_plan_for_diffusion = initial_state_plan_normalized[:,
                                                                         :diffusion_horizon, :]

        # --- MODIFICATION HERE ---
        refined_state_plan_normalized = self.state_diffusion_model.diffusion.refine_state_path(
            initial_state_path=initial_state_plan_for_diffusion,
            observation_batch_for_cond=observation_batch_for_cond
        )
        # --- END MODIFICATION ---

        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim
        current_norm_state_for_invdyn = norm_state

        if current_norm_state_for_invdyn.shape[-1] != inv_dyn_state_dim:
            # print(f"Warning: Current state dim ({current_norm_state_for_invdyn.shape[-1]}) mismatch for InvDyn ({inv_dyn_state_dim}). Using slice.")
            current_norm_state_for_invdyn = current_norm_state_for_invdyn[:,
                                                                          :inv_dyn_state_dim]

        refined_plan_for_invdyn = refined_state_plan_normalized
        if refined_plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            # print(f"Warning: Refined plan dim ({refined_plan_for_invdyn.shape[-1]}) mismatch for InvDyn ({inv_dyn_state_dim}). Adjusting.")
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
            action_dim = action_dim_tensor.shape[0] if hasattr(
                action_dim_tensor, 'shape') else 2
            return torch.zeros((raw_state_input.shape[0], action_dim), device=self.device)
