#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/eval_multimodal_diffusion.py
"""
This script evaluates a policy combining the autoregressive multimodal predictor
with the diffusion model for improved trajectory planning and execution.

It requires the installation of the 'gym_pusht' simulation environment:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path
import torch
import numpy as np
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import json
import time

# Import necessary components
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictorConfig
from model.predictor.multimodal_autoregressive_predictor import MultimodalAutoregressivePredictor
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.normalize import Normalize, Unnormalize


class MultimodalDiffusionPolicy(torch.nn.Module):
    """
    Combined policy that integrates the multimodal autoregressive predictor with
    the diffusion model for improved planning and control.

    The autoregressive predictor generates future state and image predictions,
    which are then used to guide the diffusion model for more accurate actions.
    """

    def __init__(
        self,
        diffusion_model: MyDiffusionModel,
        inv_dyn_model: MlpInvDynamic,
        autoregressive_model: MultimodalAutoregressivePredictor,
        use_images_from_autoregressive: bool = True,
        num_samples: int = 4
    ):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model
        self.autoregressive_model = autoregressive_model
        self.use_images_from_autoregressive = use_images_from_autoregressive
        self.num_samples = num_samples

        # Get configuration from the diffusion model
        self.config = diffusion_model.config
        self.device = next(diffusion_model.parameters()).device

        # Track observation history
        self._queues = {}
        self.reset()

        # Create normalizers
        try:
            metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
            dataset_stats = {}
            for key, stat in metadataset_stats.stats.items():
                dataset_stats[key] = {
                    subkey: torch.as_tensor(
                        subval, dtype=torch.float32, device=self.device)
                    for subkey, subval in stat.items()
                }

            # Create normalizers for diffusion model
            self.normalize_inputs = Normalize(
                self.config.input_features, self.config.normalization_mapping, dataset_stats)
            self.unnormalize_action_output = Unnormalize(
                {"action": self.config.action_feature}, self.config.normalization_mapping, dataset_stats)

            print("Successfully created normalizers in MultimodalDiffusionPolicy")
        except Exception as e:
            print(f"Warning: Failed to create normalizers: {e}")

    def reset(self):
        """Reset observation queues for a new episode."""
        self._queues = {}
        self._queues["observation.state"] = []
        if hasattr(self.config, "image_features") and self.config.image_features:
            self._queues["observation.image"] = []

    @torch.no_grad()
    def select_action(self, curr_state, image=None):
        """
        Select an action using the combined policy.

        1. Generate future trajectory with the autoregressive model
        2. Pass the predicted trajectory to the diffusion model
        3. Use inverse dynamics to convert to actions

        Args:
            curr_state: Current state tensor (B, state_dim)
            image: Optional current image tensor (B, C, H, W)

        Returns:
            Action tensor (B, action_dim)
        """
        # Update observation queues
        if len(curr_state.shape) == 1:
            curr_state = curr_state.unsqueeze(0)  # Add batch dimension

        # Append current state to queue
        self._queues["observation.state"].append(curr_state)

        # If we have image, append it too
        if image is not None:
            if len(image.shape) == 3:
                image = image.unsqueeze(0)  # Add batch dimension
            self._queues["observation.image"].append(image)

        # Prepare batch for diffusion model
        batch = {}

        # Need at least 2 states for the diffusion model (previous and current)
        if len(self._queues["observation.state"]) >= 2:
            # Get previous and current states
            prev_state = self._queues["observation.state"][-2]
            curr_state = self._queues["observation.state"][-1]

            # Stack them for observation.state key
            batch["observation.state"] = torch.cat(
                [prev_state, curr_state], dim=0).unsqueeze(0)

            if image is not None and "observation.image" in self._queues:
                # Format image for diffusion model: [B, T, C, H, W]
                # First, get the latest image and ensure proper dimensions
                latest_image = self._queues["observation.image"][-1]

                # Resize to 84x84 which is likely what the diffusion model expects
                if latest_image.shape[-1] != 84 or latest_image.shape[-2] != 84:
                    latest_image = torch.nn.functional.interpolate(
                        latest_image, size=(84, 84), mode='bilinear', align_corners=False
                    )

                # Create a sequence of images to match n_obs_steps
                # Duplicate the image for the required observation steps (typically 2)
                n_obs_steps = self.config.n_obs_steps if hasattr(self.config, "n_obs_steps") else 2
                image_sequence = latest_image.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)  # [B, n_obs_steps, C, H, W]
                
                batch["observation.image"] = image_sequence
        else:
            # Not enough states yet, use only the current state twice
            batch["observation.state"] = curr_state.repeat(2, 1).unsqueeze(0)
            if image is not None:
                # Resize to 84x84 which is likely what the diffusion model expects
                if image.shape[-1] != 84 or image.shape[-2] != 84:
                    image = torch.nn.functional.interpolate(
                        image, size=(84, 84), mode='bilinear', align_corners=False
                    )

                # Create a sequence of images to match n_obs_steps
                # Duplicate the image for the required observation steps (typically 2)
                n_obs_steps = self.config.n_obs_steps if hasattr(self.config, "n_obs_steps") else 2
                image_sequence = image.unsqueeze(1).repeat(1, n_obs_steps, 1, 1, 1)  # [B, n_obs_steps, C, H, W]

                batch["observation.image"] = image_sequence

        # Convert to correct device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}

        # Step 1: Generate future trajectory with the autoregressive model if available
        future_trajectory = None

        if self.autoregressive_model is not None:
            # Get context for the autoregressive model (all available states)
            context_states = torch.stack(
                self._queues["observation.state"], dim=1)

            # Ensure we have at least one image for the autoregressive model
            if image is not None:
                # Get original image from the queue (before any resizing for diffusion model)
                original_image = self._queues["observation.image"][-1]
                
                # Use the autoregressive model to predict future trajectory with the original image
                predicted_states, _, _, _ = self.autoregressive_model.predict_future_trajectory(
                    context_states, original_image, generate_noise=False
                )

                # Debug print to see shapes
                print(f"Autoregressive model predicted_states shape: {predicted_states.shape}")
                
                # The autoregressive model returns a 2D tensor (B, state_dim)
                # We need to reshape it to match the diffusion model's output format (B, horizon, state_dim)
                # We'll repeat the prediction to match the expected horizon length of the diffusion model
                # Default to a reasonable horizon length if we don't know it yet
                horizon_length = 10  # Default value, will be updated when we see diffusion model output
                future_trajectory = predicted_states.unsqueeze(1).repeat(1, horizon_length, 1)  # (B, horizon, state_dim)

        # Step 2: Get normalized batch and pass to diffusion model
        try:
            norm_batch = self.normalize_inputs(batch)
        except Exception as e:
            print(f"Normalization error: {e}, using unnormalized batch")
            norm_batch = batch

        # Step 3: Sample trajectories from diffusion model
        # If future trajectory is available, use it to guide sampling
        sampled_actions = []

        # Prepare global conditioning for the diffusion model
        global_cond = self.diffusion_model._prepare_global_conditioning(
            norm_batch)

        for _ in range(self.num_samples):
            # Sample from diffusion model using conditional_sample
            trajectories = self.diffusion_model.conditional_sample(
                batch_size=curr_state.shape[0],
                global_cond=global_cond
            )

            # If future trajectory from autoregressive model is available, use it
            if future_trajectory is not None:
                # Apply some weighting between diffusion and autoregressive predictions
                # This is a simple example - more sophisticated fusion methods could be used
                # Weight for diffusion model (0.7 diffusion, 0.3 autoregressive)
                alpha = 0.7

                # Only blend the future part (not conditioning states)
                # If we have future predictions (beyond conditioning)
                if trajectories.shape[-2] > 2:
                    # Extract future predictions (beyond the 2 conditioning states)
                    diffusion_future = trajectories[:, 2:, :]

                    # Resize future_trajectory if needed
                    print(f"Diffusion future shape: {diffusion_future.shape}, Future trajectory shape: {future_trajectory.shape}")
                    # Handle the case where future_trajectory is 2D (B, state_dim) or has different shape
                    if len(future_trajectory.shape) == 2:
                        # Expand to match diffusion_future shape by repeating the single prediction
                        future_trajectory_expanded = future_trajectory.unsqueeze(1).repeat(1, diffusion_future.shape[1], 1)
                        print(f"Expanded future trajectory from 2D to shape: {future_trajectory_expanded.shape}")
                        blended_future = alpha * diffusion_future + (1-alpha) * future_trajectory_expanded
                    elif diffusion_future.shape[-2] != future_trajectory.shape[-2]:
                        # If both are 3D but different sizes, use as much as we can
                        min_steps = min(
                            diffusion_future.shape[-2], future_trajectory.shape[-2])
                        blended_future = alpha * diffusion_future[:, :min_steps, :] + \
                            (1-alpha) * future_trajectory[:, :min_steps, :]
                    else:
                        blended_future = alpha * diffusion_future + \
                            (1-alpha) * future_trajectory

                    # Replace future part with blended prediction
                    trajectories[:, 2:, :] = blended_future

            # Use inverse dynamics to get action from first two states of trajectory
            if trajectories.shape[1] >= 2:
                # The inverse dynamics model expects state pairs as a single concatenated tensor
                # Create state pairs by concatenating consecutive states
                state_pairs = torch.cat([
                    trajectories[:, 0, :],  # Previous state
                    trajectories[:, 1, :]   # Next state
                ], dim=-1)  # Shape: (B, state_dim*2)
                
                # Call the inverse dynamics model with the concatenated state pair
                sampled_action = self.inv_dyn_model(state_pairs)
            else:
                print(
                    f"Warning: Trajectory too short ({trajectories.shape}), using zero action")
                sampled_action = torch.zeros((curr_state.shape[0], self.inv_dyn_model.a_dim),
                                             device=self.device)

            sampled_actions.append(sampled_action)

        # Stack all sampled actions
        # (B, num_samples, action_dim)
        stacked_actions = torch.stack(sampled_actions, dim=1)

        # Choose best action - for now, just take the first one
        # In a more advanced implementation, we could score them with a critic
        selected_action = stacked_actions[:, 0, :]

        # Unnormalize action if needed
        try:
            action = self.unnormalize_action_output(
                {"action": selected_action})["action"]
        except Exception as e:
            print(f"Unnormalization error: {e}, using normalized action")
            action = selected_action

        return action


def main():
    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path("outputs/train/diffusion_only")
    invdyn_output_dir = Path("outputs/train/invdyn_only")
    autoregressive_output_dir = Path("outputs/train/multimodal_autoregressive")

    # Assume config and stats are available in the diffusion directory
    config_stats_path = diffusion_output_dir

    # Output directory for evaluation results
    output_directory = Path("outputs/eval/multimodal_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Config and Stats ---
    # Load diffusion config
    diffusion_cfg = DiffusionConfig.from_pretrained(config_stats_path)
    diffusion_cfg.device = device  # Override device if needed

    # Load dataset stats
    metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
    dataset_stats = {}
    for key, stat in metadataset_stats.stats.items():
        dataset_stats[key] = {
            subkey: torch.as_tensor(subval, dtype=torch.float32, device=device)
            for subkey, subval in stat.items()
        }

    # --- Load Models ---
    # 1. Diffusion Model
    diffusion_ckpt_path = diffusion_output_dir / "diffusion_final.pth"
    if not diffusion_ckpt_path.is_file():
        raise OSError(
            f"Diffusion checkpoint not found at {diffusion_ckpt_path}")

    diffusion_model = MyDiffusionModel(diffusion_cfg)
    print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
    diff_state_dict = torch.load(diffusion_ckpt_path, map_location="cpu")
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.eval()
    diffusion_model.to(device)

    # 2. Inverse Dynamics Model
    invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"
    if not invdyn_ckpt_path.is_file():
        raise OSError(
            f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path}")

    inv_dyn_model = MlpInvDynamic(
        # State dim * 2 (current + prev)
        o_dim=diffusion_cfg.robot_state_feature.shape[0],
        a_dim=diffusion_cfg.action_feature.shape[0],
        hidden_dim=diffusion_cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh()
    )

    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # 3. Autoregressive Model
    autoregressive_ckpt_path = autoregressive_output_dir / \
        "multimodal_autoregressive_final.pt"
    if not autoregressive_ckpt_path.is_file():
        print(
            f"Warning: Autoregressive checkpoint not found at {autoregressive_ckpt_path}")
        print("Continuing without autoregressive model...")
        autoregressive_model = None
    else:
        # Load config from checkpoint
        checkpoint = torch.load(autoregressive_ckpt_path, map_location="cpu")

        if 'config' in checkpoint:
            print("Loading autoregressive config from checkpoint")
            autoreg_cfg_dict = checkpoint['config']
            autoreg_cfg = MultimodalFuturePredictorConfig(**autoreg_cfg_dict)
        else:
            print("Creating default autoregressive config")
            # Create a default config - adjust as needed
            autoreg_cfg = MultimodalFuturePredictorConfig(
                state_dim=diffusion_cfg.robot_state_feature.shape[0],
                horizon=8 + 1,  # Future steps + current
                hidden_dim=512,
                num_layers=6,
                num_heads=8,
                future_steps=8,
                predict_uncertainty=False
            )

        # Create and load the autoregressive model
        autoregressive_model = MultimodalAutoregressivePredictor(autoreg_cfg)

        print(
            f"Loading autoregressive state dict from: {autoregressive_ckpt_path}")
        if 'model_state_dict' in checkpoint:
            autoregressive_model.load_state_dict(
                checkpoint['model_state_dict'])
        else:
            autoregressive_model.load_state_dict(checkpoint)

        autoregressive_model.eval()
        autoregressive_model.to(device)

    # --- Create Combined Policy ---
    combined_model = MultimodalDiffusionPolicy(
        diffusion_model=diffusion_model,
        inv_dyn_model=inv_dyn_model,
        autoregressive_model=autoregressive_model,
        use_images_from_autoregressive=True,
        num_samples=4  # Generate 4 trajectory samples
    )

    # --- Environment Setup ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",  # Ensure this matches config expectations
        max_episode_steps=500,
    )

    # --- Evaluation Loop ---
    numpy_observation, info = env.reset(seed=42)
    rewards = []
    frames = []
    frames.append(env.render())

    step = 0
    done = False

    print("Starting evaluation rollout...")
    while not done:
        # --- Prepare Observation ---
        state_np = numpy_observation["agent_pos"]
        image_np = numpy_observation["pixels"]

        state = torch.from_numpy(state_np).to(torch.float32)
        image = torch.from_numpy(image_np).to(torch.float32) / 255
        image = image.permute(2, 0, 1)  # HWC to CHW

        # Add batch dim
        state = state.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        # Reset combined model if starting a new episode
        if step == 0:
            combined_model.reset()

        # Get action from the combined policy
        with torch.inference_mode():
            action = combined_model.select_action(
                curr_state=state,
                image=image
            )

        # Convert to numpy for environment step
        numpy_action = action.squeeze(0).cpu().numpy()

        # Step the environment
        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)

        print(f"Step {step}: Reward = {reward}, Terminated = {terminated}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated or truncated
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure or max steps reached!")

    # Get the speed of environment
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")

    # Save metrics to a JSON file
    metrics = {
        "returns": rewards,
        "mean_return": np.mean(rewards),
        "std_return": np.std(rewards),
        "num_steps": step,
        "success": terminated
    }

    # Save metrics to JSON
    metrics_file = output_directory / "metrics.json"
    with open(metrics_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = metrics.copy()
        for key, value in metrics_json.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
        json.dump(metrics_json, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
