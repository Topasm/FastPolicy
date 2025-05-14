#!/usr/bin/env python3
"""
Training script for the noise trajectory critic model.
This model learns to distinguish between original trajectories and trajectories with progressively added noise.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import os
import safetensors.torch
import matplotlib.pyplot as plt
import einops

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

# Import the critic model
from model.critic.noise_critic import create_noise_critic, NoiseCriticConfig
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.diffusion_modules import DiffusionRgbEncoder


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train noise trajectory critic model")
    parser.add_argument("--output_dir", type=str, default="outputs/train/noise_critic",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--architecture", type=str, choices=["mlp", "transformer", "gru"], default="mlp",
                        help="Type of critic model architecture")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Logging frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=500,
                        help="Checkpoint saving frequency (steps)")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--dataset", type=str,
                        default="lerobot/pusht", help="Dataset to use for training")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for the model")
    parser.add_argument("--use_images", action="store_true",
                        help="Whether to use image features as context")
    parser.add_argument("--base_noise_scale", type=float,
                        default=0.05, help="Initial noise level")
    parser.add_argument("--noise_growth_factor", type=float, default=1.2,
                        help="Factor by which noise increases per timestep")
    parser.add_argument("--use_last_frame_only", action="store_true",
                        help="Whether to use only the last observation frame as context")
    parser.add_argument("--diffusion_steps", type=int, default=100,
                        help="Number of diffusion timesteps (if using diffusion-style noise)")
    parser.add_argument("--noise_type", type=str, choices=["progressive", "diffusion", "uniform"],
                        default="progressive", help="Type of noise to apply")
    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = torch.device(args.device)

    # Load dataset metadata
    print(f"Loading dataset metadata for: {args.dataset}")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)
    print("Dataset metadata loaded.")

    # Set up features similar to diffusion training for PushT dataset
    input_features = {
        "observation.state": features["observation.state"],
        # PushT uses this key
        "observation.image": features["observation.image"],
    }
    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]
    }

    # Use DiffusionConfig to get image encoder details and horizons
    temp_diffusion_cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        predict_state=True,  # Set to true for state prediction alignment
    )

    # Set up image processing if using images
    image_key = "observation.image"  # PushT dataset standard key
    image_encoder = None
    image_feature_dim = 0

    if args.use_images:
        if image_key not in features:
            print(
                f"Warning: '{image_key}' not found in dataset features. Running without image context.")
            args.use_images = False
        else:
            print(f"Using images from key: {image_key}")

            # Create image encoder
            image_encoder = DiffusionRgbEncoder(temp_diffusion_cfg).to(device)
            image_encoder.eval()
            image_feature_dim = temp_diffusion_cfg.transformer_dim

    # Create critic configuration
    critic_cfg = NoiseCriticConfig(
        state_dim=features["observation.state"].shape[0],
        horizon=temp_diffusion_cfg.horizon,
        hidden_dim=args.hidden_dim,
        use_image_context=args.use_images,
        image_feature_dim=image_feature_dim,
        architecture=args.architecture
    )

    # Create the critic model
    critic_model = create_noise_critic(critic_cfg)
    critic_model.to(device)
    critic_model.train()

    # Try to compile model if PyTorch version supports it
    try:
        if hasattr(torch, "compile"):
            print("Compiling critic model...")
            critic_model = torch.compile(critic_model)
            print("Model compiled successfully.")
        else:
            print("torch.compile not available. Skipping model compilation.")
    except Exception as e:
        print(f"Model compilation failed: {e}")

    # Setup normalization
    # We'll keep normalization on CPU to match the incoming data from dataloader
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]},
        temp_diffusion_cfg.normalization_mapping,
        dataset_metadata.stats
    )

    # Create a second normalizer for any device operations if needed
    normalize_device = Normalize(
        {"observation.state": features["observation.state"]},
        temp_diffusion_cfg.normalization_mapping,
        dataset_metadata.stats
    )

    # Move normalization stats to device for the device normalizer
    stats_for_device = {}
    for k, v in dataset_metadata.stats.items():
        if isinstance(v, torch.Tensor):
            stats_for_device[k] = v.to(device)
        else:
            stats_for_device[k] = v
    normalize_device.stats = stats_for_device

    # Set up dataset and dataloaders
    # For critic training, we need:
    # 1. State sequence to evaluate (as positive examples)
    # 2. Optional image context if using

    # Define time indices similar to diffusion training
    # Include both observation window and future states (like in train_diffusion.py)
    diffusion_state_indices = list(
        range(1 - temp_diffusion_cfg.n_obs_steps, temp_diffusion_cfg.horizon + 1))

    # Setup delta timestamps - matching the diffusion training approach
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in diffusion_state_indices],
    }

    # Add image timestamps if using images
    if args.use_images and image_key:
        # Use the observation window indices as in diffusion training
        delta_timestamps[image_key] = [
            i / dataset_metadata.fps for i in temp_diffusion_cfg.observation_delta_indices]

    # Include action for padding mask if needed
    delta_timestamps["action"] = [
        i / dataset_metadata.fps for i in temp_diffusion_cfg.action_delta_indices]

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print("Dataset initialized.")

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # Setup optimizer
    optimizer = torch.optim.AdamW(critic_model.parameters(), lr=args.lr)

    # Setup loss function
    criterion = torch.nn.BCEWithLogitsLoss()

    # Try to import tqdm for progress bars
    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        print("tqdm not found. Install with 'pip install tqdm' for progress bars.")
        use_tqdm = False

    # Training loop
    step = 0
    losses = []
    accuracies = []

    print(
        f"Starting training with {args.architecture} critic on {args.dataset} dataset...")
    print(f"Noise type: {args.noise_type}, Base scale: {args.base_noise_scale}" +
          (f", Growth factor: {args.noise_growth_factor}" if args.noise_type == "progressive" else ""))
    print(f"Using images: {args.use_images}" + (
        f" (last frame only: {args.use_last_frame_only})" if args.use_images else ""))
    print(
        f"Model dimensions: state_dim={critic_cfg.state_dim}, horizon={critic_cfg.horizon}, hidden_dim={critic_cfg.hidden_dim}")

    # Print dataset info
    print(f"Dataset size: {len(dataset)} samples")
    print(f"Batch size: {args.batch_size}, Steps: {args.steps}")

    # Setup progress tracking
    try:
        from tqdm import tqdm
        # Create progress bar for the total number of steps
        pbar = tqdm(total=args.steps, desc="Training")
    except ImportError:
        pbar = None

    while step < args.steps:
        for batch in dataloader:
            # First normalize on CPU (since the original batch is on CPU)
            # Create a copy of the batch that contains only the keys we want to normalize
            state_batch = {"observation.state": batch["observation.state"]}
            norm_state_batch = normalize_state(state_batch)

            # Move batch and normalized state to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Create a norm_batch dictionary with everything now on the device
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}

            # Replace with normalized state
            norm_batch["observation.state"] = norm_state_batch["observation.state"].to(
                device)

            # Add padding mask back if it exists in the original batch
            if "action_is_pad" in batch:
                norm_batch["action_is_pad"] = batch["action_is_pad"]

            # Extract positive state trajectories - use only the future states
            # (from n_obs_steps onwards, matching the horizon length)
            positive_state_trajectory = norm_batch["observation.state"][:,
                                                                        temp_diffusion_cfg.n_obs_steps:temp_diffusion_cfg.n_obs_steps + temp_diffusion_cfg.horizon]
            B, H, D_state = positive_state_trajectory.shape

            # Generate negative trajectories based on the noise type
            negative_state_trajectory = positive_state_trajectory.clone()

            if args.noise_type == "progressive":
                # Apply progressive noise (increasing with each timestep)
                current_noise_scale = args.base_noise_scale
                for t_step in range(1, H):  # Start from the second state
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * current_noise_scale
                    negative_state_trajectory[:, t_step] += noise
                    current_noise_scale *= args.noise_growth_factor

            elif args.noise_type == "diffusion":
                # Apply diffusion-like noise (stronger at the end of the sequence)
                # This simulates the noise schedule from diffusion model sampling
                for t_step in range(1, H):
                    # Calculate noise scale based on position in sequence (like diffusion timestep)
                    timestep_fraction = t_step / (H - 1)  # 0 to 1
                    noise_scale = args.base_noise_scale * \
                        (1.0 + 10 * timestep_fraction**2)
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * noise_scale
                    negative_state_trajectory[:, t_step] += noise

            elif args.noise_type == "uniform":
                # Apply uniform noise across all timesteps
                for t_step in range(1, H):
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * args.base_noise_scale
                    negative_state_trajectory[:, t_step] += noise

            # Process images if using them - use observation images only
            image_features = None
            if args.use_images and image_key in batch:
                try:
                    with torch.no_grad():
                        # Extract observation images (already moved to device)
                        images = batch[image_key]  # (B, T_img, C, H, W)
                        Bi, T_img = images.shape[:2]

                        # Safety check for image dimensions
                        if len(images.shape) < 5:
                            print(
                                f"Warning: Unexpected image shape: {images.shape}. Skipping image processing.")
                        else:
                            # Reshape for encoder (matches diffusion approach)
                            images_flat = einops.rearrange(
                                images, "b t c h w -> (b t) c h w")
                            image_features_flat = image_encoder(
                                images_flat)  # (B*T_img, feature_dim)

                            # Reshape back and use either the last frame features or mean pooling
                            # For PushT, often we just use the first/last frame
                            image_features_seq = einops.rearrange(
                                image_features_flat, "(b t) d -> b t d", b=Bi, t=T_img
                            )

                            # Use the last frame features for conditioning (or mean pool if preferred)
                            if args.use_last_frame_only and T_img > 0:
                                # Last frame: (B, feature_dim)
                                image_features = image_features_seq[:, -1]
                            else:
                                image_features = image_features_seq.mean(
                                    dim=1)  # Mean: (B, feature_dim)
                except Exception as e:
                    print(f"Error processing images: {e}")
                    image_features = None

            # Combine trajectories and create labels
            combined_trajectories = torch.cat(
                [positive_state_trajectory, negative_state_trajectory], dim=0
            )  # (2*B, H, D_state)

            # Duplicate image features if using them
            combined_image_features = None
            if image_features is not None:
                combined_image_features = torch.cat(
                    [image_features, image_features], dim=0
                )  # (2*B, feature_dim)

            # Create labels
            labels = torch.cat([
                torch.ones(B, device=device),    # Positive examples (original)
                torch.zeros(B, device=device)    # Negative examples (noisy)
            ]).float()

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            logits = critic_model(
                trajectory_sequence=combined_trajectories,
                image_features=combined_image_features
            )  # (2*B, 1)

            # Compute loss
            loss = criterion(logits.squeeze(-1), labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log progress
            if step % args.log_freq == 0:
                # Calculate accuracy
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                acc = (preds == labels).float().mean().item()

                # Store metrics
                losses.append(loss.item())
                accuracies.append(acc)

                # Update progress bar or print
                if pbar is not None:
                    pbar.set_postfix(
                        loss=f"{loss.item():.4f}", acc=f"{acc:.3f}")
                else:
                    print(
                        f"Step {step}/{args.steps} | Loss: {loss.item():.4f} | Acc: {acc:.3f}")

            # Save checkpoint
            if step % args.save_freq == 0 and step > 0:
                checkpoint_path = output_dir / f"noise_critic_{step}.pth"
                torch.save(critic_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

                # Save loss plot
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(losses)
                plt.title('Loss')
                plt.subplot(1, 2, 2)
                plt.plot(accuracies)
                plt.title('Accuracy')
                plt.tight_layout()
                plt.savefig(output_dir / "training_curve.png")
                plt.close()

            # Update step counter and progress bar
            step += 1
            if pbar is not None:
                pbar.update(1)

            if step >= args.steps:
                break

    # Close progress bar if it exists
    if pbar is not None:
        pbar.close()

    # Save final model
    final_path = output_dir / "noise_critic_final.pth"
    torch.save(critic_model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # Save config
    import json
    config_dict = {k: v for k, v in critic_cfg.__dict__.items()}
    with open(output_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # Save stats
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items()
        if isinstance(v, torch.Tensor)
    }
    safetensors.torch.save_file(
        stats_to_save, output_dir / "stats.safetensors")

    # Save final plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(accuracies)
    plt.title('Accuracy')
    plt.tight_layout()
    plt.savefig(output_dir / "final_training_curve.png")
    plt.close()

    print("All artifacts saved. Training complete!")


if __name__ == "__main__":
    main()
