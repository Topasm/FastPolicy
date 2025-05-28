#!/usr/bin/env python3
"""
Training script for the Bidirectional Autoregressive Transformer.

This script implements the complete training pipeline for image-conditioned 
bidirectional trajectory generation with autoregressive learning.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import time
from pathlib import Path
import safetensors.torch
from typing import Dict, Any

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig,
    compute_loss
)
from model.bidirectional_dataset import BidirectionalTrajectoryDataset


def compute_comprehensive_loss(
    model: BidirectionalARTransformer,
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    initial_image_latents: torch.Tensor
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive loss including latent consistency.

    Args:
        model: The bidirectional transformer model
        predictions: Model predictions
        targets: Ground truth targets  
        initial_image_latents: Initial image latents for consistency

    Returns:
        Dictionary of losses
    """
    losses = {}

    # State prediction losses (MSE)
    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        # Forward states: predict st_1 to st_15 from st_0
        losses['forward_state_loss'] = F.mse_loss(
            predictions['predicted_forward_states'],
            targets['forward_states'][:, 1:]  # Skip initial state st_0
        )

    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.mse_loss(
            predictions['predicted_backward_states'],
            targets['backward_states']
        )

    # Image reconstruction losses
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'],
            targets['goal_images']
        )

    if 'predicted_final_images' in predictions and 'initial_images' in targets:
        losses['image_reconstruction_loss'] = F.mse_loss(
            predictions['predicted_final_images'],
            targets['initial_images']
        )

    # Latent consistency losses
    if 'predicted_goal_latents' in predictions:
        # Goal latent should be consistent with goal image
        goal_image_latents = model.image_encoder(targets['goal_images'])
        losses['goal_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_goal_latents'],
            goal_image_latents
        )

    if 'predicted_final_latents' in predictions:
        # Final latent should match initial image latent
        losses['final_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_final_latents'],
            initial_image_latents
        )

    # Compute weighted total loss
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 2.0,
        'image_reconstruction_loss': 2.0,
        'goal_latent_consistency_loss': 1.0,
        'final_latent_consistency_loss': 2.0,  # Higher weight for cycle consistency
    }

    total_loss = 0.0
    for loss_name, loss_value in losses.items():
        weight = weights.get(loss_name, 1.0)
        total_loss += weight * loss_value

    losses['total_loss'] = total_loss

    return losses


def train_epoch(
    model: BidirectionalARTransformer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    epoch_losses = {}
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(device)

        optimizer.zero_grad()

        # Encode initial images for latent consistency
        initial_image_latents = model.image_encoder(batch['initial_images'])

        # Forward pass
        predictions = model(
            initial_images=batch['initial_images'],
            initial_states=batch['initial_states'],
            forward_states=batch['forward_states'],
            goal_images=batch['goal_images'],
            backward_states=batch['backward_states'],
            training=True
        )

        # Compute losses
        losses = compute_comprehensive_loss(
            model, predictions, batch, initial_image_latents
        )

        # Backward pass
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Accumulate losses
        for loss_name, loss_value in losses.items():
            if loss_name not in epoch_losses:
                epoch_losses[loss_name] = 0.0
            epoch_losses[loss_name] += loss_value.item()

        num_batches += 1

        # Log progress every 50 batches
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch}, Batch {batch_idx}/{len(dataloader)}, "
                  f"Total Loss: {losses['total_loss'].item():.6f}")

    # Average losses over epoch
    for loss_name in epoch_losses:
        epoch_losses[loss_name] /= num_batches

    return epoch_losses


def main():
    """Main training function."""
    # Configuration
    output_directory = Path("outputs/train/bidirectional_transformer")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    num_epochs = 100
    batch_size = 16  # Smaller batch size due to model complexity
    learning_rate = 1e-4
    log_freq = 10
    save_freq = 20

    print("=== Bidirectional Autoregressive Transformer Training ===")
    print(f"Device: {device}")
    print(f"Output directory: {output_directory}")

    # --- Dataset Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features for our model
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    state_dim = features["observation.state"].shape[-1]
    print(f"State dimension: {state_dim}")

    # Normalization
    normalize_inputs = Normalize(input_features, {}, dataset_metadata.stats)

    # Create base dataset
    lerobot_dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=None)

    # Create bidirectional dataset
    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        normalizer=normalize_inputs,
        forward_steps=16,
        backward_steps=16,
        min_episode_length=50,
        image_key="observation.image",
        state_key="observation.state"
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=BidirectionalTrajectoryDataset.collate_fn
    )

    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # --- Model Configuration ---
    config = BidirectionalARTransformerConfig(
        state_dim=state_dim,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        max_position_value=64,
        image_channels=3,
        image_size=96,  # Assuming 96x96 images
        image_latent_dim=256,
        forward_steps=16,
        backward_steps=16
    )

    # Initialize model
    model = BidirectionalARTransformer(config)
    model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # --- Optimizer and Scheduler ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.1
    )

    # --- Training Loop ---
    start_time = time.time()
    best_total_loss = float('inf')

    print("Starting training...")

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Train for one epoch
        epoch_losses = train_epoch(model, dataloader, optimizer, device, epoch)

        # Update scheduler
        scheduler.step()

        # Log progress
        if epoch % log_freq == 0:
            elapsed_time = time.time() - start_time
            epoch_time = time.time() - epoch_start_time

            print(f"\nEpoch {epoch}/{num_epochs}")
            print(
                f"Epoch time: {epoch_time:.2f}s, Total time: {elapsed_time:.2f}s")
            print(f"Learning rate: {optimizer.param_groups[0]['lr']:.6f}")

            for loss_name, loss_value in epoch_losses.items():
                print(f"{loss_name}: {loss_value:.6f}")
            print("-" * 50)

        # Save checkpoints
        if epoch % save_freq == 0 and epoch > 0:
            checkpoint_path = output_directory / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config.__dict__,
                'losses': epoch_losses,
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        if epoch_losses['total_loss'] < best_total_loss:
            best_total_loss = epoch_losses['total_loss']
            best_model_path = output_directory / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__,
                'losses': epoch_losses,
                'best_total_loss': best_total_loss,
            }, best_model_path)
            print(
                f"New best model saved with total loss: {best_total_loss:.6f}")

    # --- Save Final Model ---
    final_path = output_directory / "final_model.pt"
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'total_training_time': time.time() - start_time,
    }, final_path)

    print(f"\nTraining completed! Final model saved to: {final_path}")
    print(f"Total training time: {time.time() - start_time:.2f}s")

    # Save model weights separately for easy loading
    torch.save(model.state_dict(), output_directory / "model_weights.pt")

    # Save configuration and dataset stats
    config_dict = config.__dict__
    torch.save(config_dict, output_directory / "config.pt")

    # Save dataset statistics
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, type(None)):
            if isinstance(value, type(None)):
                continue
            stats_to_save[key] = value
        elif hasattr(value, '__array__'):  # numpy array
            stats_to_save[key] = torch.from_numpy(value)

    if stats_to_save:
        safetensors.torch.save_file(stats_to_save, str(
            output_directory / "stats.safetensors"))
        print(
            f"Dataset stats saved to: {output_directory / 'stats.safetensors'}")


if __name__ == "__main__":
    main()
