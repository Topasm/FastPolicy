#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/train_continuous_transformer.py
import torch
import numpy as np
from pathlib import Path
import safetensors.torch
from torch.utils.data import DataLoader
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.lerobot_continuous_dataset import LeRobotContinuousDataset
from model.predictor.continuous_ar_transformer import ContinuousARTransformer, ContinuousARTransformerConfig


def main():
    output_directory = Path("outputs/train/continuous_transformer")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 20000
    log_freq = 200
    save_freq = 2000
    batch_size = 64

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"  # Same dataset as diffusion model
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features needed for our model
    input_features = {
        "observation.state": features["observation.state"],
    }

    # Simplified configuration with full trajectory timestamps
    # Using None for delta_timestamps will give us the full trajectory data
    delta_timestamps = None
    state_dim = features["observation.state"].shape[-1]
    print(f"State dimension: {state_dim}")

    # --- Normalization ---
    normalize_inputs = Normalize(
        input_features, {}, dataset_metadata.stats)

    # --- Dataset ---
    # Create the base LeRobot dataset
    lerobot_dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # Create our continuous dataset wrapper
    dataset = LeRobotContinuousDataset(
        lerobot_dataset=lerobot_dataset,
        normalizer=normalize_inputs,
        state_dim=state_dim,        # Dimension of state vectors
        bidirectional=True,         # Support both directions
        direction_weight=0.5,       # 50% forward, 50% backward
        max_position_value=64,      # Maximum position value
        min_traj_len=2,            # Set to 2 to match dataset's episode length
        seq_len=2                  # Reduced to match available data points
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,
        collate_fn=LeRobotContinuousDataset.collate_fn
    )

    # --- Model Configuration ---
    config = ContinuousARTransformerConfig(
        state_dim=state_dim,        # Dimension of state vectors from the dataset
        hidden_dim=256,             # Reduced hidden dimension for simpler task
        num_layers=3,               # Reduced layers since we have less complexity
        num_heads=4,                # Reduced attention heads
        dropout=0.1,                # Dropout rate
        max_position_value=64,      # Maximum position value
        bidirectional=True,         # Support bidirectional generation
        image_channels=3,           # RGB images
        image_size=64               # 64x64 images
    )

    # Initialize model
    model = ContinuousARTransformer(config)
    model.to(device)

    # --- Optimizer ---
    # Modified optimizer settings for a simpler prediction task
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=5e-4, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps)

    # --- Loss Function ---
    # For state prediction, we use MSE loss
    loss_fn = torch.nn.MSELoss()

    # --- Training Loop ---
    step = 0
    epoch = 0
    start_time = time.time()
    print("Starting Continuous Bidirectional Transformer Training...")
    print("WARNING: Training with very short sequences (length=2) due to dataset limitations.")
    print("This will essentially train a simple state-to-state prediction model.")

    while step < training_steps:
        epoch += 1

        for batch in dataloader:
            # Move batch to device
            inputs = batch['inputs'].to(device)
            targets = batch['targets'].to(device)
            positions = batch['positions'].to(device)
            directions = batch['directions'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Forward pass
            optimizer.zero_grad()

            # Our wrapper model will handle any reshaping needed
            outputs = model(
                inputs=inputs,
                positions=positions,
                direction=directions,
                attention_mask=attention_mask
            )

            # Calculate loss - we want to predict the next state in the sequence
            if outputs.size(1) == 1:
                # For seq_len=2 case, outputs is [B, 1, D], so we can just squeeze
                last_token_preds = outputs.squeeze(1)
            else:
                # Standard case with multiple sequence positions
                # Get the prediction for the last token in each sequence
                last_token_idx = attention_mask.sum(dim=1).long() - 1

                # Select the prediction for the last token
                batch_indices = torch.arange(outputs.size(0), device=device)
                last_token_preds = outputs[batch_indices, last_token_idx]

            # Calculate MSE loss
            loss = loss_fn(last_token_preds, targets)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            # Log progress
            if step % log_freq == 0:
                elapsed_time = time.time() - start_time
                print(f"Step: {step}/{training_steps} | Epoch: {epoch} | "
                      f"Loss: {loss.item():.6f} | "
                      f"Time: {elapsed_time:.2f}s")

            # Save checkpoints
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'step': step,
                    'config': config.__dict__,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                break

    # --- Save Final Model ---
    final_path = output_directory / "continuous_transformer_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'step': step,
    }, final_path)
    print(f"Training finished. Final model saved to: {final_path}")

    # Also save just the model weights for easy loading
    torch.save(model.state_dict(), output_directory /
               "continuous_transformer_weights.pt")

    # --- Save Dataset Stats and Metadata ---
    # Save dataset statistics from the dataset metadata
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    # Save the stats
    safetensors.torch.save_file(stats_to_save, str(
        output_directory / "stats.safetensors"))
    print(f"Stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
