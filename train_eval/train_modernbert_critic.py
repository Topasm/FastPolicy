#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/train_modernbert_critic.py

import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import safetensors if available
try:
    import safetensors.torch
    has_safetensors = True
except ImportError:
    has_safetensors = False
    print("safetensors not available, falling back to PyTorch format for stats saving.")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.critic.modernbert_critic import ModernBertCritic, ModernBertCriticConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ModernBERT Critic model for next sequence prediction")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Name of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension of the transformer")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")
    parser.add_argument("--base_noise_scale", type=float, default=0.02,
                        help="Base noise scale for negative samples (starting noise for t=0)")
    parser.add_argument("--noise_growth_factor", type=float, default=1.5,
                        help="Growth factor for noise per future timestep (t=7 gets more noise)")
    parser.add_argument("--final_noise_multiplier", type=float, default=1.0,
                        help="Global multiplier to adjust overall noise magnitude")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--output_dir", type=str, default="outputs/train/modernbert_critic",
                        help="Directory to save model and logs")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="How often to log training metrics")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="How often to save model checkpoints")
    # New arguments for next sequence prediction
    parser.add_argument("--half_horizon", type=int, default=8,
                        help="Length of half trajectory for next sequence prediction")
    parser.add_argument("--apply_gaussian_smoothing", action="store_true",
                        help="Apply Gaussian smoothing after adding noise to make trajectories smoother")
    parser.add_argument("--gaussian_sigma", type=float, default=1.0,
                        help="Sigma parameter for Gaussian smoothing (higher = smoother)")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset and Feature Setup ---
    print(f"Loading dataset metadata for {args.dataset}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Configure dataset based on features
    horizon = 16  # Default trajectory horizon length
    half_horizon = args.half_horizon  # Half-horizon for next sequence prediction

    # Get state dimension and set image feature dimension
    state_dim = features["observation.state"].shape[0]

    print(
        f"Using state dimension: {state_dim}, full horizon: {horizon}, half horizon: {half_horizon}")

    # Setup delta timestamps for trajectory features
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in range(horizon)],
        "observation.image": [i / dataset_metadata.fps for i in range(horizon)]
    }

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset initialized with {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # --- Normalization Setup ---
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]},
        {"observation.state": "standard"},  # Simple normalization mapping
        dataset_metadata.stats
    )

    # --- Model Setup ---
    # Create and configure ModernBert critic model
    print("Creating ModernBERT critic model with next sequence prediction capability...")
    config = ModernBertCriticConfig(
        state_dim=state_dim,
        horizon=horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        use_layernorm=True,
        swiglu_intermediate_factor=4,
        half_horizon=half_horizon,  # Set half-horizon for next sequence prediction
        # Whether to apply Gaussian smoothing
        apply_gaussian_smoothing=args.apply_gaussian_smoothing,
        gaussian_sigma=args.gaussian_sigma  # Sigma parameter for Gaussian filter
    )

    # Save config for future reference
    import json
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Initialize model and move to device
    model = ModernBertCritic(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created ModernBERT critic with {num_params:,} parameters")

    # Try to optimize with torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("Model compilation successful - performance should be improved")
        except Exception as e:
            print(f"Model compilation skipped: {e}")
            print("Continuing with standard model execution")

    # --- Optimizer Setup ---
    # Create optimizer and learning rate scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,  # Common weight decay for transformers
        betas=(0.9, 0.999)
    )

    # Use cosine annealing schedule for better convergence
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )

    # Train model
    model.train()

    print(
        f"Starting ModernBERT critic training for next sequence prediction ({args.steps} steps)...")
    print(
        f"Base noise scale: {args.base_noise_scale}, Growth factor: {args.noise_growth_factor}")

    # Configure simplified noise parameters
    noise_params = {
        "base_noise_scale": args.base_noise_scale,
        "noise_growth_factor": args.noise_growth_factor,
        "final_noise_multiplier": args.final_noise_multiplier
    }

    # Log the noise configuration
    print(
        f"Noise configuration: base_scale={args.base_noise_scale}, growth_factor={args.noise_growth_factor}, multiplier={args.final_noise_multiplier}")

    pbar = tqdm(range(args.steps), desc="Training ModernBERT Critic")
    step = 0

    # Main training loop
    while step < args.steps:
        for batch in dataloader:
            # Normalize state data
            norm_state_batch = normalize_state(
                {"observation.state": batch["observation.state"]})

            # Create normalized batch on device
            norm_batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in batch.items()}
            norm_batch["observation.state"] = norm_state_batch["observation.state"].to(
                device)

            # Compute loss and update model
            optimizer.zero_grad()
            loss, accuracy = model.compute_binary_classification_loss(
                norm_batch=norm_batch,
                noise_params=noise_params
            )
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Track metrics
            if not hasattr(main, 'metrics'):
                main.metrics = {
                    'losses': [],
                    'accuracies': [],
                    'lr': []
                }

            # Store current metrics
            main.metrics['losses'].append(loss.item())
            main.metrics['accuracies'].append(accuracy.item())
            main.metrics['lr'].append(lr_scheduler.get_last_lr()[0])

            # Log progress
            if step % args.log_freq == 0:
                # Calculate running averages
                recent_losses = main.metrics['losses'][-args.log_freq:]
                recent_accs = main.metrics['accuracies'][-args.log_freq:]
                avg_loss = sum(recent_losses) / len(recent_losses)
                avg_acc = sum(recent_accs) / len(recent_accs)
                current_lr = lr_scheduler.get_last_lr()[0]

                print(f"Step {step}/{args.steps}: Loss={avg_loss:.4f} (current={loss.item():.4f}), "
                      f"Acc={avg_acc:.4f}, LR={current_lr:.6f}")

            # Save checkpoint
            if step % args.save_freq == 0 and step > 0:
                # Save with metrics included
                checkpoint_path = output_dir / f"modernbert_critic_{step}.pth"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item(),
                    'accuracy': accuracy.item(),
                    'config': config.__dict__,
                }, checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            # Update step counter
            step += 1
            pbar.update(1)
            if step >= args.steps:
                break

    # --- Save Final Model ---
    final_path = output_dir / "modernbert_critic_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'step': step,
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # Also save just the model weights for easy loading
    torch.save(model.state_dict(), output_dir /
               "modernbert_critic_weights.pth")
    print(
        f"Model weights saved to: {output_dir / 'modernbert_critic_weights.pth'}")

    # --- Save Normalization Stats ---
    print("Saving dataset statistics for future normalization...")

    def process_stats_item(value):
        """Convert numpy arrays to tensors for saving"""
        if isinstance(value, np.ndarray):
            return torch.from_numpy(value)
        return value if isinstance(value, torch.Tensor) else None

    # Process and flatten stats dictionary
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, (np.ndarray, torch.Tensor)):
            stats_to_save[key] = process_stats_item(value)
        elif isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, (np.ndarray, torch.Tensor)):
                    stats_to_save[f"{key}.{k}"] = process_stats_item(v)

    # Save stats file using preferred format
    stats_path = output_dir / \
        ("stats.safetensors" if has_safetensors else "stats.pt")
    try:
        if has_safetensors:
            safetensors.torch.save_file(stats_to_save, stats_path)
        else:
            torch.save(stats_to_save, stats_path)
        print(f"Dataset statistics saved to {stats_path}")
    except Exception as e:
        fallback_path = output_dir / "stats.pt"
        print(f"Error saving stats: {e}")
        torch.save(stats_to_save, fallback_path)
        print(f"Stats saved using fallback format to {fallback_path}")


if __name__ == "__main__":
    main()
