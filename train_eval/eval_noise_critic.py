#!/usr/bin/env python3
"""
Evaluation script for the noise trajectory critic model in a real environment.
This script combines the diffusion model, inverse dynamics model, and the noise critic
to evaluate the effectiveness of the noise critic in a real environment setting.
"""

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union
import os

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

# Import the models
from model.critic.noise_critic import create_noise_critic, NoiseCriticConfig
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.modeling_combined import CombinedPolicy
from model.invdynamics.invdyn import MlpInvDynamic
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix
from pathlib import Path
import argparse
import einops
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import json
import time
from typing import Dict, List, Optional, Tuple, Any, Union

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

# Import the models
from model.critic.noise_critic import create_noise_critic, NoiseCriticConfig
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.modeling_combined import CombinedPolicy
from model.invdynamics.invdyn import MlpInvDynamic
from model.diffusion.diffusion_modules import DiffusionRgbEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, confusion_matrix


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate noise trajectory critic model with combined policy")
    parser.add_argument("--critic_path", type=str, required=True,
                        help="Path to the trained critic model checkpoint")
    parser.add_argument("--diffusion_path", type=str, required=True,
                        help="Path to the trained diffusion model checkpoint")
    parser.add_argument("--invdyn_path", type=str, required=True,
                        help="Path to the trained inverse dynamics model checkpoint")
    parser.add_argument("--config_path", type=str, required=True,
                        help="Path to the model configuration file")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/noise_critic",
                        help="Directory to save evaluation results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Dataset to use for evaluation")
    parser.add_argument("--noise_levels", type=str, default="0.01,0.05,0.1,0.2,0.5,1.0",
                        help="Comma-separated list of noise levels to evaluate")
    parser.add_argument("--num_episodes", type=int, default=5,
                        help="Number of episodes to run in the environment")
    parser.add_argument("--render", action="store_true",
                        help="Render the environment")
    parser.add_argument("--save_video", action="store_true",
                        help="Save videos of the episodes")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)

    # Load model configuration
    with open(args.config_path, "r") as f:
        config_dict = json.load(f)

    # Use DiffusionConfig for dataset parameters and load from the config path
    cfg = DiffusionConfig(**config_dict)
    cfg.device = device  # Override device if needed

    # Load dataset metadata
    print(f"Loading dataset metadata for: {args.dataset}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    print("Dataset metadata loaded.")

    # Setup image processing if needed
    image_key = None
    image_encoder = None

    if critic_cfg.use_image_context:
        # Find an available image key
        available_img_keys = [
            k for k in features if k.startswith("observation.images")]
        if not available_img_keys:
            print("Warning: No image keys found. Running without image context.")
            critic_cfg.use_image_context = False
        else:
            image_key = available_img_keys[0]
            print(f"Using images from key: {image_key}")

            # Update config with image input
            temp_diffusion_cfg = DiffusionConfig(
                input_features={
                    "observation.state": features["observation.state"],
                    image_key: features[image_key]
                },
                output_features={"action": features["action"]}
            )

            # Create image encoder
            image_encoder = DiffusionRgbEncoder(temp_diffusion_cfg).to(device)
            image_encoder.eval()

    # Create the critic model
    critic_model = create_noise_critic(critic_cfg)

    # Load trained weights
    critic_model.load_state_dict(torch.load(
        args.model_path, map_location=device))
    critic_model.to(device)
    critic_model.eval()

    # Setup normalization
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]},
        temp_diffusion_cfg.normalization_mapping,
        dataset_metadata.stats
    )

    # Move normalization stats to device
    stats_for_device = {}
    for k, v in dataset_metadata.stats.items():
        if isinstance(v, torch.Tensor):
            stats_for_device[k] = v.to(device)
        else:
            stats_for_device[k] = v
    normalize_state.stats = stats_for_device

    # Set up dataset
    state_indices = list(range(
        temp_diffusion_cfg.n_obs_steps,
        temp_diffusion_cfg.n_obs_steps + critic_cfg.horizon
    ))

    # Image indices if using images
    image_indices = []
    if critic_cfg.use_image_context and image_key:
        image_indices = list(range(temp_diffusion_cfg.n_obs_steps))

    # Setup delta timestamps
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in state_indices],
    }

    if critic_cfg.use_image_context and image_key:
        delta_timestamps[image_key] = [
            i / dataset_metadata.fps for i in image_indices]

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print("Dataset initialized.")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=4,
        pin_memory=device.type == "cuda"
    )

    # Parse noise levels
    noise_levels = [float(x) for x in args.noise_levels.split(",")]

    # Prepare results storage
    results = {
        "noise_levels": noise_levels,
        "scores": {level: [] for level in noise_levels},
        "labels": []
    }

    # Evaluate on different noise levels
    print("Starting evaluation...")

    total_samples = 0
    with torch.no_grad():
        for batch in dataloader:
            if total_samples >= args.num_samples:
                break

            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Normalize state trajectories
            state_batch = {"observation.state": batch["observation.state"]}
            norm_state_batch = normalize_state(state_batch)

            # Extract positive state trajectories
            # (B, horizon, state_dim)
            original_trajectory = norm_state_batch["observation.state"]
            B, H, D_state = original_trajectory.shape

            # Process images if using them
            image_features = None
            if critic_cfg.use_image_context and image_key in batch:
                with torch.no_grad():
                    images = batch[image_key]  # (B, T_img, C, H, W)
                    Bi, T_img = images.shape[:2]

                    # Reshape for encoder
                    images_flat = einops.rearrange(
                        images, "b t c h w -> (b t) c h w")
                    image_features_flat = image_encoder(
                        images_flat)  # (B*T_img, feature_dim)

                    # Reshape back and mean pool over time dimension
                    image_features_seq = einops.rearrange(
                        image_features_flat, "(b t) d -> b t d", b=Bi, t=T_img
                    )
                    image_features = image_features_seq.mean(
                        dim=1)  # (B, feature_dim)

            # Score original trajectories
            original_scores = critic_model(
                trajectory_sequence=original_trajectory,
                image_features=image_features
            ).squeeze(-1)

            # Original trajectories get label 1
            results["labels"].extend([1] * B)

            # Test with different noise levels
            for noise_level in noise_levels:
                # Apply uniform noise across all timesteps
                noisy_trajectory = original_trajectory.clone()
                noise = torch.randn_like(noisy_trajectory) * noise_level
                noisy_trajectory += noise

                # Score noisy trajectories
                noisy_scores = critic_model(
                    trajectory_sequence=noisy_trajectory,
                    image_features=image_features
                ).squeeze(-1)

                # Store scores
                results["scores"][noise_level].extend(
                    noisy_scores.cpu().numpy().tolist())

                # For the first noise level, also store the original scores
                if noise_level == noise_levels[0]:
                    results["scores"]["original"] = results["scores"].get(
                        "original", [])
                    results["scores"]["original"].extend(
                        original_scores.cpu().numpy().tolist())

            total_samples += B
            print(f"Processed {total_samples}/{args.num_samples} samples")

    # Convert to numpy arrays for easier analysis
    for k, v in results["scores"].items():
        results["scores"][k] = np.array(v)
    results["labels"] = np.array(results["labels"])

    # Calculate metrics
    metrics = {
        "noise_levels": noise_levels,
        "auc": [],
        "accuracy": []
    }

    for noise_level in noise_levels:
        # Combine original and noisy scores
        all_scores = np.concatenate([
            results["scores"]["original"],
            results["scores"][noise_level]
        ])

        # Create labels (1 for original, 0 for noisy)
        all_labels = np.concatenate([
            np.ones_like(results["scores"]["original"]),
            np.zeros_like(results["scores"][noise_level])
        ])

        # Calculate AUC
        from sklearn.metrics import roc_auc_score, accuracy_score

        # Apply sigmoid to convert logits to probabilities
        all_probs = 1 / (1 + np.exp(-all_scores))
        auc = roc_auc_score(all_labels, all_probs)

        # Calculate accuracy using 0.5 threshold
        preds = (all_probs > 0.5).astype(float)
        accuracy = accuracy_score(all_labels, preds)

        metrics["auc"].append(auc)
        metrics["accuracy"].append(accuracy)

        print(
            f"Noise level {noise_level}: AUC = {auc:.4f}, Accuracy = {accuracy:.4f}")

    # Plot results
    plt.figure(figsize=(12, 10))

    # Plot 1: Score distributions
    plt.subplot(2, 2, 1)
    plt.hist(results["scores"]["original"],
             bins=30, alpha=0.5, label="Original")
    for noise_level in noise_levels:
        plt.hist(results["scores"][noise_level], bins=30,
                 alpha=0.5, label=f"Noise {noise_level}")
    plt.xlabel("Score (logits)")
    plt.ylabel("Count")
    plt.title("Score Distributions")
    plt.legend()

    # Plot 2: ROC Curves
    plt.subplot(2, 2, 2)
    for i, noise_level in enumerate(noise_levels):
        from sklearn.metrics import roc_curve

        # Combine scores
        all_scores = np.concatenate([
            results["scores"]["original"],
            results["scores"][noise_level]
        ])

        # Create labels
        all_labels = np.concatenate([
            np.ones_like(results["scores"]["original"]),
            np.zeros_like(results["scores"][noise_level])
        ])

        # Calculate ROC
        all_probs = 1 / (1 + np.exp(-all_scores))
        fpr, tpr, _ = roc_curve(all_labels, all_probs)

        plt.plot(
            fpr, tpr, label=f"Noise {noise_level} (AUC = {metrics['auc'][i]:.3f})")

    plt.plot([0, 1], [0, 1], 'k--', label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()

    # Plot 3: Performance vs. Noise Level
    plt.subplot(2, 2, 3)
    plt.plot(noise_levels, metrics["auc"], 'o-', label="AUC")
    plt.plot(noise_levels, metrics["accuracy"], 'o--', label="Accuracy")
    plt.xlabel("Noise Level")
    plt.ylabel("Performance")
    plt.title("Performance vs. Noise Level")
    plt.legend()
    plt.grid(True)

    # Plot 4: Confusion matrix for mid noise level
    plt.subplot(2, 2, 4)
    mid_idx = len(noise_levels) // 2
    mid_noise = noise_levels[mid_idx]

    # Combine scores for mid noise level
    all_scores = np.concatenate([
        results["scores"]["original"],
        results["scores"][mid_noise]
    ])

    # Create labels
    all_labels = np.concatenate([
        np.ones_like(results["scores"]["original"]),
        np.zeros_like(results["scores"][mid_noise])
    ])

    # Calculate predictions
    all_probs = 1 / (1 + np.exp(-all_scores))
    preds = (all_probs > 0.5).astype(float)

    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, preds)

    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Noisy", "Original"],
                yticklabels=["Noisy", "Original"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (Noise Level = {mid_noise})")

    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_results.png")
    print(f"Saved evaluation plots to {output_dir / 'evaluation_results.png'}")

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {output_dir / 'metrics.json'}")

    print("Evaluation complete!")


if __name__ == "__main__":
    main()
