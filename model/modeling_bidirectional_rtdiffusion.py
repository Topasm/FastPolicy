from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional  # For Optional type hint

from lerobot.common.policies.utils import get_device_from_parameters
from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
# RT-Diffusion model (e.g., CLDiffPhyConModel) will be passed as nn.Module


class BidirectionalRTDiffusionPolicy(nn.Module):
    """
    Combined policy class that uses a bidirectional transformer to predict states from images,
    then uses an RT-Diffusion model for action generation.

    The bidirectional transformer generates forward states (e.g., st_0 to st_15) from images.
    The RT-Diffusion model generates actions based on current observation history and
    the predicted state path from the bidirectional transformer.
    """

    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        rt_diffusion_model: nn.Module,  # This will be an instance of CLDiffPhyConModel
        dataset_stats: dict,
        n_obs_steps: int
    ):
        super().__init__()
        self.bidirectional_transformer = bidirectional_transformer
        self.rt_diffusion_model = rt_diffusion_model
        self.stats = dataset_stats  # Store the full stats dictionary
        self.n_obs_steps = n_obs_steps
        self.device = get_device_from_parameters(bidirectional_transformer)

        # Initialize observation queues for RT-Diffusion
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)

        # Queue for executing actions planned by RT-Diffusion
        self._action_execution_queue = deque()

        # For CL-DiffPhyCon style feedback mechanism (optional)
        self._u_pred_async_rt_diffusion = None

    def reset(self):
        """Clear all observation and action queues, and reset state variables."""
        self._obs_image_queue.clear()
        self._obs_state_queue.clear()
        self._action_execution_queue.clear()
        self._u_pred_async_rt_diffusion = None

        if hasattr(self.rt_diffusion_model, 'reset'):
            self.rt_diffusion_model.reset()

    @torch.no_grad()
    def select_action(self, current_raw_observation: dict[str, Tensor]) -> Tensor:
        """
        Two-stage action generation pipeline:
        1. Bidirectional Transformer generates future states from current image.
        2. RT-Diffusion generates actions to achieve these states.
        """
        # Ensure input batch tensors are on the correct device
        current_raw_observation = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in current_raw_observation.items()
        }

        # 1. Prepare Current Observation (Normalize & Update Queues)
        # Expected: [B,C,H,W], pixels 0-1
        raw_img = current_raw_observation["observation.image"]
        # Expected: [B, env_state_dim]
        raw_state = current_raw_observation["observation.state"]

        # Normalize image to [-1, 1]
        if raw_img.max() <= 1.0 and raw_img.min() >= 0.0:
            norm_img = raw_img * 2.0 - 1.0
        else:
            norm_img = raw_img  # Assume already in desired range if not 0-1

        # Normalize state
        if self.stats and "observation.state" in self.stats:
            state_mean = self.stats["observation.state"]["mean"]
            state_std = self.stats["observation.state"]["std"]
            norm_state = (raw_state - state_mean) / state_std
        else:
            print("Warning: State normalization stats not found. Using raw state.")
            norm_state = raw_state

        # Update observation queues (store with sequence dimension for cat later)
        # Each element in queue is [B, 1, FeatureDim]
        self._obs_image_queue.append(norm_img.unsqueeze(1))
        self._obs_state_queue.append(norm_state.unsqueeze(1))

        # 2. Check Action Execution Queue
        if self._action_execution_queue:
            return self._action_execution_queue.popleft()

        # 3. Check if enough observation history is available for RT-Diffusion
        if len(self._obs_image_queue) < self.n_obs_steps:
            action_dim = self.stats.get("action", {}).get(
                "mean", torch.zeros(2)).shape[0]  # Default to 2D if no stats
            print(
                f"Warning: Not enough obs history for RT-Diffusion ({len(self._obs_image_queue)}/{self.n_obs_steps}). Returning zero action.")
            return torch.zeros((raw_state.shape[0], action_dim), device=self.device)

        try:
            # 4. Generate Future State Path (Bidirectional Transformer)
            # Bidir Transformer usually takes current single obs (not history)
            # The `norm_state` and `norm_img` here are the latest ones.

            # This is the full env state dim, normalized
            current_norm_state_for_bidir = norm_state
            current_norm_img_for_bidir = norm_img

            # Adjust state dimension for Bidirectional Transformer if its config differs
            bidir_state_dim = self.bidirectional_transformer.config.state_dim
            if current_norm_state_for_bidir.shape[-1] != bidir_state_dim:
                print(
                    f"Warning: BidirTransformer expects state_dim {bidir_state_dim}, got {current_norm_state_for_bidir.shape[-1]}. Slicing state.")
                # Assuming the first `bidir_state_dim` components are relevant (e.g., XY position)
                current_norm_state_for_bidir_input = current_norm_state_for_bidir[
                    :, :bidir_state_dim]
            else:
                current_norm_state_for_bidir_input = current_norm_state_for_bidir

            transformer_predictions = self.bidirectional_transformer(
                initial_images=current_norm_img_for_bidir,
                initial_states=current_norm_state_for_bidir_input,
                training=False
            )
            # norm_predicted_plan_segment: [B, forward_steps-1, bidir_state_dim] (e.g., st_1 to st_15)
            norm_predicted_plan_segment = transformer_predictions['predicted_forward_states']

            # The `plan_states_normalized` for RT-Diffusion should be what BidirTransformer outputs directly
            # If BidirTransformer outputs 2D states, this `plan_for_rtdiffusion` will be 2D.
            # The `current_norm_state_for_bidir_input` is st_0 for this plan.
            plan_for_rtdiffusion = torch.cat(
                [current_norm_state_for_bidir_input.unsqueeze(1), norm_predicted_plan_segment], dim=1
            )  # Shape: [B, forward_steps, bidir_state_dim]

            # 5. Generate Action Sequence (RT-Diffusion Model)
            # Prepare observation history for RT-Diffusion (using full-dimensional normalized states)
            obs_history_img = torch.cat(list(self._obs_image_queue), dim=1)
            obs_history_state = torch.cat(list(self._obs_state_queue), dim=1)

            rt_diffusion_input_obs = {
                # [B, n_obs_steps, C, H, W]
                "observation.image": obs_history_img,
                # [B, n_obs_steps, env_state_dim]
                "observation.state": obs_history_state
            }

            # Call RT-Diffusion model's predict_action method
            normalized_action_sequence = self.rt_diffusion_model.predict_action(
                obs_dict=rt_diffusion_input_obs,
                # Pass the direct BidirTransformer output plan
                plan_states_normalized=plan_for_rtdiffusion,
                # For CL-DiffPhyCon feedback
                previous_rt_diffusion_plan=self._u_pred_async_rt_diffusion
            )  # Expected output: [B, action_horizon, action_dim] (normalized)

            # 6. Optional: Update _u_pred_async_rt_diffusion for CL-DiffPhyCon feedback
            if self._u_pred_async_rt_diffusion is not None or True:  # Always update if using CL-DiffPhyCon style
                if normalized_action_sequence.shape[1] > 1:
                    next_u_pred = torch.roll(
                        normalized_action_sequence, shifts=-1, dims=1)
                    # Or re-noise, or zero
                    next_u_pred[:, -1,
                                :] = normalized_action_sequence[:, -1, :].clone()
                else:
                    next_u_pred = normalized_action_sequence.clone()
                self._u_pred_async_rt_diffusion = next_u_pred.detach()

            # 7. Unnormalize Actions and Store in Execution Queue
            if self.stats and "action" in self.stats:
                action_mean = self.stats["action"]["mean"]
                action_std = self.stats["action"]["std"]

                for i in range(normalized_action_sequence.shape[1]):
                    action_norm = normalized_action_sequence[:, i, :]
                    action_unnorm = action_norm * action_std + action_mean
                    self._action_execution_queue.append(action_unnorm)
            else:
                print(
                    "Warning: Action normalization stats not found. Using normalized actions.")
                for i in range(normalized_action_sequence.shape[1]):
                    self._action_execution_queue.append(
                        normalized_action_sequence[:, i, :])

            # Return the first action from the queue
            if self._action_execution_queue:
                return self._action_execution_queue.popleft()
            else:  # Should not happen if normalized_action_sequence has actions
                action_dim = self.stats.get("action", {}).get(
                    "mean", torch.zeros(2)).shape[0]
                return torch.zeros((raw_state.shape[0], action_dim), device=self.device)

        except Exception as e:
            print(f"Error during action generation: {e}")
            import traceback
            traceback.print_exc()
            action_dim = self.stats.get("action", {}).get(
                "mean", torch.zeros(2)).shape[0]
            return torch.zeros((raw_state.shape[0] if 'raw_state' in locals() else 1, action_dim), device=self.device)
