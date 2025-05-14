# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for combining separately trained models with noise critic.

"""
This script demonstrates how to evaluate a policy composed of separately trained
diffusion, inverse dynamics, and noise critic models.

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

# Import necessary components
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.modeling_critic_combined import CombinedCriticPolicy
from model.invdynamics.invdyn import MlpInvDynamic
from model.critic.noise_critic import create_noise_critic, NoiseCriticConfig
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from sklearn.metrics import roc_auc_score, accuracy_score


def main():
    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path("outputs/train/diffusion_only")
    invdyn_output_dir = Path("outputs/train/invdyn_only")
    # Using noise_critic for the critic model
    noise_critic_output_dir = Path("outputs/train/noise_critic")

    # Assume config.json is available in diffusion output directory
    config_stats_path = diffusion_output_dir

    # Output directory for evaluation results
    output_directory = Path("outputs/eval/noise_critic")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Noise levels to test
    noise_levels = [0.05, 0.1, 0.2]

    # --- Load Config and Stats ---
    cfg_path = Path(config_stats_path) / "config.json"
    data = json.loads(cfg_path.read_text())
    cfg = DiffusionConfig(**data)

    # Load config using the base class method
    cfg = DiffusionConfig.from_pretrained(config_stats_path)
    cfg.device = device  # Override device if needed

    # Load dataset stats
    metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
    dataset_stats = {}
    for key, stat in metadataset_stats.stats.items():
        dataset_stats[key] = {
            subkey: torch.as_tensor(subval, dtype=torch.float32, device=device)
            for subkey, subval in stat.items()
        }

    # --- Load Models ---
    # Diffusion Model
    diffusion_ckpt_path = diffusion_output_dir / "diffusion_final.pth"
    if not diffusion_ckpt_path.is_file():
        raise OSError(
            f"Diffusion checkpoint not found at {diffusion_ckpt_path}")
    diffusion_model = MyDiffusionModel(cfg)
    print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
    diff_state_dict = torch.load(diffusion_ckpt_path, map_location="cpu")
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.eval()
    diffusion_model.to(device)

    # Inverse Dynamics Model
    invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"
    if not invdyn_ckpt_path.is_file():
        raise OSError(
            f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path}")
    inv_dyn_model = MlpInvDynamic(
        # State dim * 2 (current + prev)
        o_dim=cfg.robot_state_feature.shape[0] * 2,
        a_dim=cfg.action_feature.shape[0],
        hidden_dim=cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh()
    )
    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # Noise Critic Model
    noise_critic_ckpt_path = noise_critic_output_dir / "noise_critic_final.pth"
    if not noise_critic_ckpt_path.is_file():
        raise OSError(
            f"Noise critic checkpoint not found at {noise_critic_ckpt_path}")

    # Create noise critic config
    critic_cfg = NoiseCriticConfig(
        state_dim=cfg.robot_state_feature.shape[0],
        horizon=cfg.horizon,
        hidden_dim=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        use_layernorm=cfg.use_layernorm,
        architecture="dv_horizon",  # Use the transformer-based critic
        use_image_context=cfg.use_image_observations
    )

    # Create and load critic model
    print(f"Loading noise critic state dict from: {noise_critic_ckpt_path}")
    noise_critic_model = create_noise_critic(critic_cfg)
    noise_critic_state_dict = torch.load(
        noise_critic_ckpt_path, map_location=device)
    noise_critic_model.load_state_dict(noise_critic_state_dict)
    noise_critic_model.eval()
    noise_critic_model.to(device)
    print("Noise critic model loaded successfully.")

    # Create combined model with critic
    combined_model = CombinedCriticPolicy(
        diffusion_model=diffusion_model,
        inv_dyn_model=inv_dyn_model,
        critic_model=noise_critic_model,
        num_samples=5  # Generate 5 trajectory samples
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

    # Store noise critic scores
    critic_scores = {level: [] for level in noise_levels}
    critic_scores["original"] = []

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

        # Add batch dim and move to device
        state = state.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        # Create observation dictionary for the model
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Reset combined model if starting a new episode
        if step == 0:
            combined_model.reset()

        # Get action from the combined policy
        # The select_action method now returns both action and trajectories
        with torch.inference_mode():
            # Call our updated select_action method that uses the critic model
            action, trajectories = combined_model.select_action(
                curr_state=observation["observation.state"],
                image=image
            )

        # Extract trajectory for noise critic (take first trajectory)
        normalized_trajectory = trajectories[0]  # Shape: [horizon, state_dim]

        # Score original trajectory with noise critic
        with torch.inference_mode():
            orig_score = noise_critic_model(
                trajectory_sequence=normalized_trajectory.unsqueeze(0)
            ).squeeze().item()
            critic_scores["original"].append(orig_score)

        # Score noisy trajectories
        for noise_level in noise_levels:
            # Apply noise to trajectory
            noisy_traj = normalized_trajectory.clone(
            ) + torch.randn_like(normalized_trajectory) * noise_level

            # Get critic score
            with torch.inference_mode():
                noise_score = noise_critic_model(
                    trajectory_sequence=noisy_traj.unsqueeze(0)
                ).squeeze().item()
                critic_scores[noise_level].append(noise_score)

        # Convert to numpy for environment step
        numpy_action = action.squeeze(0).cpu().numpy()

        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)

        print(f"{step=} {reward=} {terminated=}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated or truncated or done
        step += 1

    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")

    # Calculate basic metrics
    for k in critic_scores:
        critic_scores[k] = np.array(critic_scores[k])

    # Save metrics to a JSON file
    metrics = {
        "noise_levels": noise_levels,
        "returns": rewards,
        "mean_return": np.mean(rewards),
        "std_return": np.std(rewards),
        "auc": [],
        "accuracy": []
    }

    # Calculate classifier metrics for each noise level
    for noise_level in noise_levels:
        # Combine original and noisy scores
        all_scores = np.concatenate([
            critic_scores["original"],
            critic_scores[noise_level]
        ])

        # Create labels (1 for original, 0 for noisy)
        all_labels = np.concatenate([
            np.ones_like(critic_scores["original"]),
            np.zeros_like(critic_scores[noise_level])
        ])

        # Apply sigmoid to convert logits to probabilities
        all_probs = 1 / (1 + np.exp(-all_scores))

        # Calculate AUC and accuracy
        auc = roc_auc_score(all_labels, all_probs)
        preds = (all_probs > 0.5).astype(float)
        accuracy = accuracy_score(all_labels, preds)

        metrics["auc"].append(auc)
        metrics["accuracy"].append(accuracy)

        print(
            f"Noise level {noise_level}: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")

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
