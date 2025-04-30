import os
from pathlib import Path
from collections import deque
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    populate_queues,
)
from model.diffusion.configuration_mymodel import DiffusionConfig
# Reuse diffusion components
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic  # Import the modified MLP
from model.critic.critic_model import CriticScorer  # Optional for inference


class JointDiffusionPolicy(PreTrainedPolicy):
    """
    Jointly trains a state diffusion model and an inverse dynamics model.
    Uses teacher forcing during training by default.
    """
    # TODO: Potentially create a JointDiffusionConfig inheriting from DiffusionConfig
    #       to add specific flags like teacher_forcing, loss weights, etc.
    config_class = DiffusionConfig
    name = "joint_diffusion"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
    ):
        super().__init__(config)
        # Enforce state prediction mode for this joint policy
        if not config.predict_state:
            raise ValueError(
                "JointDiffusionPolicy requires config.predict_state=True.")
        config.validate_features()
        self.config = config

        # Normalizers/Unnormalizers
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        # Diffusion normalizes states
        self.normalize_diffusion_target = Normalize(
            {config.diffusion_target_key:
                config.output_features[config.diffusion_target_key]},
            config.normalization_mapping, dataset_stats
        )
        # Inverse dynamics normalizes states and actions
        self.normalize_invdyn_state = Normalize(
            {"observation.state": config.robot_state_feature},
            config.normalization_mapping, dataset_stats
        )
        self.normalize_invdyn_action = Normalize(
            {"action": config.action_feature},
            config.normalization_mapping, dataset_stats
        )
        self.unnormalize_action_output = Unnormalize(
            {"action": config.action_feature},
            config.normalization_mapping, dataset_stats
        )

        # State and Action Dimensions
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        # --- Instantiate Models ---
        # 1. Diffusion Model (predicts states)
        self.diffusion_model = MyDiffusionModel(config)

        # 2. Inverse Dynamics Model (predicts actions from state transitions)
        self.inv_dyn_model = MlpInvDynamic(
            o_dim=self.state_dim,
            a_dim=self.action_dim,
            hidden_dim=config.inv_dyn_hidden_dim,  # Use config value
            # Assuming Tanh activation for actions based on MlpInvDynamic default
            out_activation=nn.Tanh()
        )

        # Optional: Critic for inference sample selection
        self.critic_scorer = None
        if config.critic_model_path and config.num_inference_samples > 1:
            self.critic_scorer = CriticScorer(
                model_path=config.critic_model_path,
                state_dim=self.state_dim,
                # Critic likely scores state trajectories from diffusion
                horizon=config.horizon,
                hidden_dim=config.critic_hidden_dim,
                device=self.device  # Assuming device is set later or inferred
            )

        # Queues for inference
        self._queues = None
        self.reset()

    def get_optim_params(self) -> list:
        # Return parameters of both models for joint optimization
        return list(self.diffusion_model.parameters()) + list(self.inv_dyn_model.parameters())

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.images"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Select a single action given environment observations."""
        device = get_device_from_parameters(self)
        batch = {k: v.to(device) for k, v in batch.items()}
        batch = self.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            processed_batch = dict(batch)
            processed_batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            processed_batch = batch

        # Populate queues with the latest observation
        self._queues = populate_queues(self._queues, processed_batch)

        # Generate new action plan only when the action queue is empty
        if len(self._queues["action"]) == 0:
            # Prepare batch for the model by stacking history from queues
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking
                    queue_list = [item.to(device) for item in queue]
                    if queue_list:  # Check if queue is not empty
                        model_input_batch[key] = torch.stack(queue_list, dim=1)
                    else:
                        # Handle empty queue case if necessary, maybe raise error or return default
                        raise ValueError(
                            f"Observation queue for '{key}' is empty during inference.")

            # Check if all required observation keys are present
            required_obs_keys = {"observation.state"}
            if self.config.image_features:
                required_obs_keys.add("observation.images")
            if self.config.env_state_feature:
                required_obs_keys.add("observation.environment_state")

            if not required_obs_keys.issubset(model_input_batch.keys()):
                raise ValueError(
                    f"Missing required observation keys for inference. Need {required_obs_keys}, got {model_input_batch.keys()}")

            # --- Inference Pipeline ---
            # 1. Prepare conditioning for diffusion
            global_cond = self.diffusion_model._prepare_global_conditioning(
                model_input_batch)
            batch_size = global_cond.shape[0]

            # 2. Sample future state trajectories from diffusion model
            # Shape: (B, num_samples, horizon, state_dim)
            predicted_states_normalized = self.diffusion_model.conditional_sample(
                batch_size,
                global_cond=global_cond,
                num_samples=self.config.num_inference_samples
            )

            # 3. Prepare states for inverse dynamics model
            # Get current state (normalized)
            # Shape: (B, state_dim)
            current_state_normalized = model_input_batch["observation.state"][:, -1, :]

            # Prepend current state to predicted state trajectories
            # Shape: (B, num_samples, 1, state_dim)
            s_t_normalized = current_state_normalized.unsqueeze(1).unsqueeze(1).expand(
                -1, self.config.num_inference_samples, 1, -1
            )
            # Shape: (B, num_samples, horizon + 1, state_dim)
            all_states_normalized = torch.cat(
                [s_t_normalized, predicted_states_normalized], dim=2)

            # Get state pairs (s_t, s_{t+1}) - these are normalized
            # Shape: (B, num_samples, horizon, state_dim)
            s_t_pairs_normalized = all_states_normalized[:, :, :-1, :]
            s_tplus1_pairs_normalized = all_states_normalized[:, :, 1:, :]

            # Reshape for batch processing by inv_dyn_model
            # Shape: (B * num_samples * horizon, state_dim)
            s_t_flat_normalized = einops.rearrange(
                s_t_pairs_normalized, 'b ns h d -> (b ns h) d')
            s_tplus1_flat_normalized = einops.rearrange(
                s_tplus1_pairs_normalized, 'b ns h d -> (b ns h) d')

            # 4. Predict action sequences using Inverse Dynamics Model
            # Output actions are normalized because inv_dyn_model output activation is Tanh
            # Shape: (B * num_samples * horizon, action_dim)
            inferred_actions_flat_normalized = self.inv_dyn_model.predict(
                s_t_flat_normalized, s_tplus1_flat_normalized
            )

            # Reshape back: (B, num_samples, horizon, action_dim)
            inferred_actions_normalized = einops.rearrange(
                inferred_actions_flat_normalized, '(b ns h) d -> b ns h d',
                b=batch_size, ns=self.config.num_inference_samples, h=self.config.horizon
            )

            # 5. Select best action sequence using Critic (if available)
            # Critic expects unnormalized states/actions? Check CriticScorer implementation.
            # Assuming critic works with normalized states/actions for now.
            if self.config.num_inference_samples > 1 and self.critic_scorer is not None:
                # TODO: Verify if critic needs normalized or unnormalized inputs
                # For now, assume it uses the normalized predicted states from diffusion
                scores = self.critic_scorer.score(
                    predicted_states_normalized.squeeze(
                        0)  # Assuming B=1 for rollout
                )  # (num_samples,)
                best_idx = torch.argmax(scores)
                # (1, horizon, action_dim)
                final_actions_normalized_horizon = inferred_actions_normalized[:, best_idx]
            else:
                # If only one sample or no critic, take the first sample
                # (B, horizon, action_dim)
                final_actions_normalized_horizon = inferred_actions_normalized[:, 0]

            # 6. Unnormalize the final action sequence
            # Shape: (B, horizon, action_dim)
            final_actions_unnormalized = self.unnormalize_action_output(
                {"action": final_actions_normalized_horizon}
            )["action"]

            # 7. Extract the required `n_action_steps` for execution
            # Shape: (B, n_action_steps, action_dim)
            actions_to_execute = final_actions_unnormalized[:,
                                                            :self.config.n_action_steps]

            # Add generated actions to the queue (assuming B=1 for rollout)
            self._queues["action"].extend(actions_to_execute.squeeze(0))

        # Pop the next action from the queue
        action = self._queues["action"].popleft()
        return action.unsqueeze(0)  # Return with batch dim

    def forward(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict]:
        """
        Run the batch through both models and compute the combined loss.
        Assumes the batch tensors are already on the correct device.
        """
        # --- 1. Diffusion Model Loss (State Prediction) ---
        # Normalize inputs required by the diffusion model's compute_loss
        # This typically includes observations and the target state.
        normalized_batch_for_diffusion = self.normalize_inputs(batch)
        normalized_batch_for_diffusion.update(
            self.normalize_diffusion_target(batch))

        # Pass the normalized batch to the diffusion model's loss computation
        diffusion_loss, _ = self.diffusion_model.compute_loss(
            normalized_batch_for_diffusion  # Use the normalized batch here
        )

        # --- 2. Inverse Dynamics Model Loss ---
        # Extract required data slices (use the original, unnormalized batch for slicing)
        n_obs = self.config.n_obs_steps
        horizon = self.config.horizon
        gt_states = batch["observation.state"]
        gt_actions = batch["action"]

        # Prepare state pairs (s_t, s_{t+1}) from GT states
        s_t_gt = gt_states[:, n_obs-1: n_obs+horizon-1]
        s_tplus1_gt = gt_states[:, n_obs: n_obs+horizon]
        gt_actions_horizon = gt_actions[:, n_obs-1: n_obs+horizon-1]

        # Normalize states and actions specifically for the inv dyn model
        s_t_gt_norm = self.normalize_invdyn_state({"observation.state": s_t_gt})[
            "observation.state"]
        s_tplus1_gt_norm = self.normalize_invdyn_state(
            {"observation.state": s_tplus1_gt})["observation.state"]
        gt_actions_norm = self.normalize_invdyn_action(
            {"action": gt_actions_horizon})["action"]

        # Predict actions using inv dyn model with GT states (Teacher Forcing)
        pred_actions_norm = self.inv_dyn_model(s_t_gt_norm, s_tplus1_gt_norm)

        # Compute Inverse Dynamics Loss (MSE)
        inv_dyn_loss = F.mse_loss(pred_actions_norm, gt_actions_norm)

        # --- 3. Combine Losses ---
        total_loss = diffusion_loss + inv_dyn_loss  # Simple sum for now

        loss_dict = {
            "loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "inv_dyn_loss": inv_dyn_loss,
        }
        return total_loss, loss_dict

    def save_pretrained(self, save_directory: str | Path):
        """Saves both diffusion and inverse dynamics models."""
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save_pretrained(save_directory)

        # Save diffusion model weights
        torch.save(self.diffusion_model.state_dict(),
                   save_directory / "diffusion_model.pth")
        # Save inverse dynamics model weights
        torch.save(self.inv_dyn_model.state_dict(),
                   save_directory / "inv_dyn_model.pth")

        # Save dataset stats if available (optional but recommended)
        # TODO: Need access to dataset_stats here or save them during init

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        # Allow passing stats during load
        dataset_stats: Optional[dict] = None,
        **kwargs,
    ):
        """Loads the config and weights for both models."""
        config_path = Path(pretrained_model_name_or_path)
        if not config_path.exists():
            # TODO: Add hub download logic if needed
            raise FileNotFoundError(
                f"Config directory not found at {config_path}")

        # Load config
        config = cls.config_class.from_pretrained(config_path, **kwargs)

        # Instantiate the policy
        policy = cls(config, dataset_stats)

        # Load diffusion model weights
        diffusion_weights_path = config_path / "diffusion_model.pth"
        if diffusion_weights_path.exists():
            state_dict = torch.load(
                diffusion_weights_path, map_location=policy.device)
            policy.diffusion_model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Diffusion model weights not found at {diffusion_weights_path}")

        # Load inverse dynamics model weights
        inv_dyn_weights_path = config_path / "inv_dyn_model.pth"
        if inv_dyn_weights_path.exists():
            state_dict = torch.load(
                inv_dyn_weights_path, map_location=policy.device)
            policy.inv_dyn_model.load_state_dict(state_dict)
        else:
            print(
                f"Warning: Inverse dynamics model weights not found at {inv_dyn_weights_path}")

        policy.to(policy.device)  # Ensure model is on the correct device
        policy.eval()  # Set to eval mode by default after loading
        return policy
