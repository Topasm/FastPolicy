#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/eval_multimodal_combined.py
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for combining separately trained models with multimodal future predictor.

"""
This script demonstrates how to evaluate a policy composed of separately trained
diffusion, inverse dynamics, and multimodal future predictor models.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
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
import matplotlib.pyplot as plt
from collections import deque

# Import necessary components
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.model_adapter import create_diffusion_model_from_config
from model.invdynamics.invdyn import MlpInvDynamic
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictor, MultimodalFuturePredictorConfig
# Import dataset metadata for normalization
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.normalize import Normalize, Unnormalize


class CombinedMultimodalPolicy(torch.nn.Module):
    """
    Combined policy class for diffusion model, inverse dynamics model, and multimodal future predictor.
    This class integrates the future predictor to generate more accurate trajectories that are then
    refined by the diffusion model, followed by action generation with the inverse dynamics model.
    """

    def __init__(self, diffusion_model: MyDiffusionModel, inv_dyn_model: MlpInvDynamic,
                 future_predictor: MultimodalFuturePredictor, num_samples=4,
                 context_horizon=8, planning_horizon=8):
        super().__init__()
        self.diffusion_model = diffusion_model
        self.inv_dyn_model = inv_dyn_model
        self.future_predictor = future_predictor
        self.num_samples = num_samples
        self.context_horizon = context_horizon
        self.planning_horizon = planning_horizon

        # Get device from diffusion model
        self.device = next(diffusion_model.parameters()).device
        # Store config reference
        self.config = diffusion_model.config

        # Initialize state history for trajectory prediction
        self.state_history = deque(maxlen=self.context_horizon)

        # Create normalizers since they're not available in the models
        try:
            metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
            self.dataset_stats = {}
            for key, stat in metadataset_stats.stats.items():
                self.dataset_stats[key] = {
                    subkey: torch.as_tensor(
                        subval, dtype=torch.float32, device=self.device)
                    for subkey, subval in stat.items()
                }

            # Create normalizers for the diffusion model
            self.normalize_inputs = Normalize(
                self.config.input_features, self.config.normalization_mapping, self.dataset_stats)
            self.unnormalize_action_output = Unnormalize(
                {"action": self.config.action_feature}, self.config.normalization_mapping, self.dataset_stats)

            print("Successfully created normalizers in CombinedMultimodalPolicy")
        except Exception as e:
            print(f"Warning: Failed to create normalizers: {e}")
            # Fallback to empty normalizers that do nothing

    def reset(self):
        """Reset the policy internal state at the beginning of an episode"""
        self.state_history.clear()

    def select_action(self, curr_state, image=None):
        """
        Select an action using the combined models.
        Process:
        1. Store current state in history
        2. If we have enough history, use the future predictor to predict next states
        3. Feed predicted states to diffusion model to generate a trajectory
        4. Use inverse dynamics to convert the trajectory to an action

        Args:
            curr_state: Current robot state tensor [B, D]
            image: Current RGB image tensor [B, C, H, W]

        Returns:
            action: Action tensor [B, A]
            trajectories: Optional generated trajectories for visualization
        """
        # Add current state to history
        self.state_history.append(curr_state.detach().clone())

        # Create a batch dictionary with the observations
        batch = {"observation.state": curr_state}
        if image is not None:
            batch["observation.image"] = image

        # If we don't have enough history yet, use just the diffusion model directly
        if len(self.state_history) < self.context_horizon:
            try:
                # Prepare global conditioning from current observation
                global_cond = self.diffusion_model._prepare_global_conditioning(
                    batch)

                # Use the diffusion model's conditional_sample method
                traj = self.diffusion_model.conditional_sample(
                    batch_size=curr_state.shape[0],
                    global_cond=global_cond
                )

                # Get first step for action prediction
                next_state = traj[:, 0]

                # Create state pair for inverse dynamics model (current and next state)
                state_pair = torch.cat([curr_state, next_state], dim=-1)

                # Use inverse dynamics to predict action
                action = self.inv_dyn_model(state_pair)

                # Return without future predictions since we don't have enough history
                return action, traj, None
            except Exception as e:
                print(f"Error in diffusion model sampling: {e}")
                # Fallback to random action if diffusion fails
                random_action = torch.zeros_like(
                    curr_state[:, :2])  # Assuming action dim is 2
                random_traj = curr_state.unsqueeze(1).repeat(
                    1, 8, 1)  # Simple repeated trajectory
                print("Using fallback random action due to diffusion model error")
                return random_action, random_traj, None

        # We have enough history - use the multimodal future predictor
        # Stack history into a batch tensor [B, context_horizon, D]
        state_history_tensor = torch.stack(list(self.state_history), dim=1)

        # Predict future trajectory using multimodal model
        future_state_pred = None
        with torch.no_grad():
            try:
                # Get the prediction - returns (state, image, state_uncertainty, image_uncertainty)
                future_state_pred, future_image, state_uncertainty, _ = self.future_predictor.predict_future_trajectory(
                    state_history_tensor, image
                )

                # Check if multi-step prediction is available
                if self.future_predictor.multi_step_prediction:
                    print(
                        f"Using multi-step prediction with shape: {future_state_pred.shape}")
                else:
                    print(
                        f"Using single-step prediction with shape: {future_state_pred.shape}")

                # If future_state_pred is not None, we'll use it to guide the diffusion model
                if future_state_pred is not None:
                    print("Successfully generated future predictions")

            except Exception as e:
                print(f"Error in multimodal predictor: {e}")
                future_state_pred = None

        try:
            # Prepare global conditioning from current observation
            global_cond = self.diffusion_model._prepare_global_conditioning(
                batch)

            # Set up guidance from future predictor if available
            if future_state_pred is not None:
                # Create a guidance signal for the diffusion model using future_state_pred
                # For multi-step prediction, use the predicted trajectory directly
                if self.future_predictor.multi_step_prediction and future_state_pred.shape[1] > 1:
                    # If future state predictions match the diffusion model's horizon, use directly
                    if future_state_pred.shape[1] >= self.planning_horizon:
                        # Create an initial trajectory from the future prediction
                        # (might be different shape than diffusion model expects)
                        predicted_traj = future_state_pred[:,
                                                           :self.diffusion_model.config.horizon]

                        # Sample from the diffusion model starting from the predicted trajectory
                        # instead of pure noise, using the predicted trajectory as guidance
                        print(
                            f"Using predicted trajectory as guidance, shape: {predicted_traj.shape}")
                        traj = self.diffusion_model.conditional_sample(
                            batch_size=curr_state.shape[0],
                            global_cond=global_cond,
                            guidance_scale=0.5,  # Balance between diffusion noise and guidance
                            initial_guidance=predicted_traj  # Use predicted trajectory to guide diffusion
                        )
                    else:
                        # If prediction horizon is shorter, repeat last prediction to match diffusion horizon
                        print(
                            "Prediction horizon shorter than diffusion horizon, extending...")
                        last_pred = future_state_pred[:, -1:].repeat(1,
                                                                     self.diffusion_model.config.horizon -
                                                                     future_state_pred.shape[1],
                                                                     1)
                        predicted_traj = torch.cat(
                            [future_state_pred, last_pred], dim=1)
                        traj = self.diffusion_model.conditional_sample(
                            batch_size=curr_state.shape[0],
                            global_cond=global_cond,
                            guidance_scale=0.5,
                            initial_guidance=predicted_traj
                        )
                else:
                    # For single-step prediction, repeat to create a trajectory
                    # This is a fallback approach when we only have one step of prediction
                    print(
                        "Using single-step prediction, repeating to create trajectory")
                    predicted_traj = future_state_pred.unsqueeze(1).repeat(
                        1, self.diffusion_model.config.horizon, 1)
                    traj = self.diffusion_model.conditional_sample(
                        batch_size=curr_state.shape[0],
                        global_cond=global_cond,
                        guidance_scale=0.3,  # Lower guidance for single-step prediction
                        initial_guidance=predicted_traj
                    )
            else:
                # If no future prediction is available, fall back to standard diffusion sampling
                print(
                    "No future prediction available, using standard diffusion sampling")
                traj = self.diffusion_model.conditional_sample(
                    batch_size=curr_state.shape[0],
                    global_cond=global_cond
                )

            # Get first step for action prediction
            next_state = traj[:, 0]

            # Create state pair for inverse dynamics model (current and next state)
            state_pair = torch.cat([curr_state, next_state], dim=-1)

            # Use inverse dynamics to predict action
            action = self.inv_dyn_model(state_pair)

            return action, traj, future_state_pred

        except Exception as e:
            print(f"Error in final action generation: {e}")
            # Fallback to random action if process fails
            random_action = torch.zeros_like(
                curr_state[:, :2])  # Assuming action dim is 2
            random_traj = curr_state.unsqueeze(1).repeat(
                1, 8, 1)  # Simple repeated trajectory
            print("Using fallback random action due to error")
            return random_action, random_traj, None


def main():
    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path("outputs/train/diffusion_only")
    invdyn_output_dir = Path("outputs/train/invdyn_only")
    multimodal_output_dir = Path("outputs/train/multimodal_future")

    # Assume config.json is available in diffusion output directory
    config_stats_path = diffusion_output_dir

    # Output directory for evaluation results
    output_directory = Path("outputs/eval/multimodal_combined")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Context horizon for multimodal predictor
    context_horizon = 8
    planning_horizon = 8

    # --- Load Config and Stats ---
    cfg_path = Path(config_stats_path) / "config.json"
    data = json.loads(cfg_path.read_text())

    # Ensure we use the correct horizon that matches the saved model
    # The model was trained with horizon=16, which results in position embedding of size [1, 17, 512]
    # The "+1" is because position embedding includes positions for all horizon steps plus one for conditioning
    data["horizon"] = 16  # Set to match the trained model's horizon

    # Load dataset stats
    metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
    dataset_stats = {}
    for key, stat in metadataset_stats.stats.items():
        dataset_stats[key] = {
            subkey: torch.as_tensor(subval, dtype=torch.float32, device=device)
            for subkey, subval in stat.items()
        }

    # Get feature shapes and normalization mapping from dataset metadata
    features = metadataset_stats.features
    input_features = {}
    output_features = {}

    # Add input features - state and image
    for key, feature in features.items():
        if key.startswith("observation."):
            input_features[key] = {"shape": feature["shape"]}

    # Add output features - action and state (needed for interpolation)
    output_features["action"] = {"shape": features["action"]["shape"]}
    # Also add state to output features for interpolation target
    output_features["observation.state"] = {
        "shape": features["observation.state"]["shape"]}

    # Create config with the proper feature dictionaries
    # Match the exact parameters the saved model was trained with:
    # From the saved model state dict analysis:
    # - pos_embed shape [1, 17, 512] means horizon=16, transformer_dim=512
    # - cond_embed.weight shape [512, 1540] means conditioning_dim=1540
    # - 1540 ≈ (state_dim * n_obs_steps) + (transformer_dim * n_obs_steps)
    # - With state_dim=2, transformer_dim=512: 1540 ≈ (2*3) + (512*3) = 6 + 1536 = 1542
    # - So n_obs_steps = 3 is the correct value
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        # Match the saved model horizon (pos_embed has 17 positions for 16 horizon)
        horizon=16,
        device=device,
        interpolate_state=True,  # Explicitly set interpolation mode
        noise_scheduler_type="DDPM",  # Add required params
        # Based on conditioning dimension analysis: 1540 ≈ (2*3) + (512*3) = 1542
        n_obs_steps=3,
        # Match the saved model dimension (from all the weight shapes)
        transformer_dim=512,
        # Match pretrained model layers
        transformer_num_layers=6,
        transformer_num_heads=8,  # For 512 dim (512/64=8 heads)
        inv_dyn_hidden_dim=512,
        output_horizon=16  # Same as horizon
    )

    # --- Load Models ---
    # Diffusion Model
    diffusion_ckpt_path = diffusion_output_dir / "diffusion_final.pth"
    if not diffusion_ckpt_path.is_file():
        raise OSError(
            f"Diffusion checkpoint not found at {diffusion_ckpt_path}")

    # Use the adapter to create a diffusion model that handles our config format
    diffusion_model = create_diffusion_model_from_config(cfg, dataset_stats)
    print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
    diff_state_dict = torch.load(diffusion_ckpt_path, map_location=device)
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.eval()
    diffusion_model.to(device)

    # Inverse Dynamics Model
    invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"
    if not invdyn_ckpt_path.is_file():
        raise OSError(
            f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path}")

    # Find state dimension from the first state feature
    state_dim = None
    for feature_name, feature in cfg.input_features.items():
        if feature_name.endswith('state'):
            state_dim = feature['shape'][0]
            break

    # Find action dimension from the action feature
    action_dim = None
    for feature_name, feature in cfg.output_features.items():
        if feature_name == 'action':
            action_dim = feature['shape'][0]
            break

    if state_dim is None or action_dim is None:
        raise ValueError(
            "Could not determine state and action dimensions from config")

    inv_dyn_model = MlpInvDynamic(
        # State dim * 2 (current + prev)
        o_dim=state_dim * 2,
        a_dim=action_dim,
        hidden_dim=cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh()
    )
    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location=device)
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # Multimodal Future Predictor Model
    multimodal_ckpt_path = multimodal_output_dir / "multimodal_future_final.pth"
    if not multimodal_ckpt_path.is_file():
        raise OSError(
            f"Multimodal future checkpoint not found at {multimodal_ckpt_path}")

    print(f"Loading multimodal future predictor from: {multimodal_ckpt_path}")
    checkpoint = torch.load(multimodal_ckpt_path, map_location=device)

    # Load config from checkpoint
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        mm_config = MultimodalFuturePredictorConfig(**config_dict)
    else:
        # Fallback to manually creating config
        # Extract state dim from the config
        state_dim = None
        for feat_name, feat in cfg.input_features.items():
            if feat_name.endswith('state'):
                state_dim = feat['shape'][0]
                break

        if state_dim is None:
            raise ValueError("Could not find state dimension in config")

        mm_config = MultimodalFuturePredictorConfig(
            state_dim=state_dim,
            horizon=context_horizon,
            hidden_dim=768,  # Default from multimodal predictor
            num_layers=8,
            num_heads=12,
            predict_future_image=True,
            predict_future_state=True,
            multi_step_prediction=True,
            num_future_steps=planning_horizon
        )

    # Initialize model
    multimodal_model = MultimodalFuturePredictor(mm_config).to(device)
    multimodal_model.load_state_dict(checkpoint['model_state_dict'])
    multimodal_model.eval()

    print("Multimodal future predictor model loaded successfully")
    print(f"  - State dimension: {mm_config.state_dim}")
    print(f"  - Context horizon: {mm_config.horizon}")
    print(f"  - Multi-step prediction: {mm_config.multi_step_prediction}")
    if mm_config.multi_step_prediction:
        print(f"  - Future steps: {mm_config.num_future_steps}")

    # Create combined model
    combined_model = CombinedMultimodalPolicy(
        diffusion_model=diffusion_model,
        inv_dyn_model=inv_dyn_model,
        future_predictor=multimodal_model,
        num_samples=4,  # Generate 4 trajectory samples
        context_horizon=context_horizon,
        planning_horizon=planning_horizon
    )

    # --- Environment Setup ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",  # Ensure this matches config expectations
        max_episode_steps=500,
    )

    # --- Visualization Setup ---
    plt.ion()  # Interactive mode for live visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    plt.tight_layout()

    # --- Evaluation Loop ---
    numpy_observation, info = env.reset(seed=42)
    rewards = []
    frames = []
    frames.append(env.render())

    step = 0
    done = False
    trajectories_history = []

    print("Starting evaluation rollout...")
    while not done:
        # --- Prepare Observation ---
        state_np = numpy_observation["agent_pos"]
        image_np = numpy_observation["pixels"]

        state = torch.from_numpy(state_np).to(torch.float32)
        image = torch.from_numpy(image_np).to(torch.float32) / 255
        image = image.permute(2, 0, 1)  # HWC to CHW

        # Add batch dim and move to device
        state = state.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        # Reset combined model if starting a new episode
        if step == 0:
            combined_model.reset()

        # Get action from the combined policy
        with torch.inference_mode():
            # Call our updated select_action method that uses future predictor
            action, trajectories, future_predictions = combined_model.select_action(
                curr_state=state,
                image=image
            )

        # Store trajectories for visualization
        if trajectories is not None:
            trajectories_history.append(trajectories.detach().cpu().numpy())

        # Store future predictions in info dict for visualization
        info = info or {}
        info['future_predictions'] = future_predictions.detach(
        ).cpu() if future_predictions is not None else None

        # Convert to numpy for environment step
        numpy_action = action.squeeze(0).cpu().numpy()

        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)

        print(f"{step=} {reward=} {terminated=}")

        rewards.append(reward)
        frames.append(env.render())

        # Visualize current state and predicted trajectories
        if step % 5 == 0 and len(trajectories_history) > 0:
            # Clear previous plots
            for ax in axes:
                ax.clear()

            # Show current image
            axes[0].imshow(numpy_observation["pixels"])
            axes[0].set_title("Current Observation")
            axes[0].axis('off')

            # Show latest trajectories
            current_pos = numpy_observation["agent_pos"][:2]  # x, y position
            latest_trajs = trajectories_history[-1]

            # Plot trajectory samples
            for i in range(min(4, latest_trajs.shape[0])):
                traj = latest_trajs[i]
                axes[1].plot(traj[:, 0], traj[:, 1], 'o-', alpha=0.7,
                             linewidth=1, markersize=3, label=f"Diffusion {i+1}")

            # Add multimodal future prediction if available
            if 'future_predictions' in info and info['future_predictions'] is not None:
                future_traj = info['future_predictions']
                # For multi-step prediction
                if len(future_traj.shape) == 3:
                    axes[1].plot(future_traj[0, :, 0], future_traj[0, :, 1], 's--',
                                 color='g', alpha=0.8, linewidth=2,
                                 markersize=4, label="Multimodal")
                # For single-step prediction
                elif len(future_traj.shape) == 2:
                    axes[1].plot(future_traj[0, 0], future_traj[0, 1], 's',
                                 color='g', alpha=0.8, markersize=6,
                                 label="Multimodal")

            # Add current position marker
            axes[1].plot(current_pos[0], current_pos[1],
                         'ro', markersize=8, label="Current")
            axes[1].set_title("Predicted Trajectories")
            axes[1].legend(loc='upper right')
            axes[1].set_aspect('equal')

            # Plot rewards over time
            axes[2].plot(rewards, 'b-')
            axes[2].set_title(f"Rewards (Total: {sum(rewards):.2f})")
            axes[2].set_xlabel("Step")
            axes[2].set_ylabel("Reward")
            axes[2].grid(True)

            plt.tight_layout()
            plt.draw()
            plt.pause(0.001)

            # Save visualization
            vis_path = output_directory / f"vis_step_{step}.png"
            plt.savefig(vis_path)

        done = terminated or truncated or done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    plt.close()  # Close the interactive plot

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
        "sum_return": np.sum(rewards),
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

    # Create summary visualizations
    plt.figure(figsize=(12, 10))

    # Plot 1: Rewards over time
    plt.subplot(2, 1, 1)
    plt.plot(rewards, 'b-')
    plt.title('Rewards over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)

    # Plot 2: Cumulative reward
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(rewards), 'r-')
    plt.title(f'Cumulative Reward (Total: {sum(rewards):.2f})')
    plt.xlabel('Step')
    plt.ylabel('Cumulative Reward')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_directory / "reward_summary.png")
    plt.close()

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
