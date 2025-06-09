#!/usr/bin/env python3
"""
Training script for the Bidirectional Autore    # Create a normalizer that only handles the "observation.state" key
    # The bidirectional transformer model will internally map normalization
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, 
        cfg.normalization_mapping, 
        dataset_metadata.stats)ransformer.
"""

import torch
from torch.utils.data import DataLoader
from pathlib import Path
import safetensors.torch
import numpy as np  # Added for stats saving
from tqdm import tqdm  # Added for progress bar
import wandb  # Added for visualization
import torchvision.transforms as T  # Added for image processing
import matplotlib.pyplot as plt  # Added for plotting
from datetime import datetime  # Added for unique run names

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

from model.predictor.policy import (
    GoalConditionedAutoregressivePolicy
)

from model.predictor.config import BidirectionalARTransformerConfig
from model.predictor.bidirectional_dataset import BidirectionalTrajectoryDataset
from model.predictor.normalization_utils import KeyMappingNormalizer, KeyMappingUnnormalizer
from model.diffusion.configuration_mymodel import DiffusionConfig


def main():
    """Main training function."""
    # Configuration
    output_directory = Path("outputs/train/bidirectional_transformer32")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training hyperparameters
    training_steps = 20000  # Reduced for testing WandB integration
    batch_size = 64
    learning_rate = 1e-4
    log_freq = 100  # More frequent logging for testing
    save_freq = 500  # More frequent saving for testing
    n_obs_steps = 3  # Number of observation steps for temporal encoding

    # Initialize Weights & Biases
    run_name = f"bidirectional_transformer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    wandb.init(
        project="fastpolicy",
        name=run_name,
        config={
            "architecture": "BidirectionalARTransformer",
            "training_steps": training_steps,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "forward_steps": 32,
            "backward_steps": 32,
            "n_obs_steps": n_obs_steps,
        }
    )

    print("=== Bidirectional Autoregressive Transformer Training ===")
    print(f"Device: {device}")
    print(f"Output directory: {output_directory}")
    print(f"WandB run: {run_name}")

    # --- Dataset Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    cfg = DiffusionConfig(
        input_features={"observation.state": features["observation.state"]},
        output_features={"observation.image": features["observation.image"]}
    )  # Dummy config to access properties
    input_features = {
        "observation.state": features["observation.state"]
    }
    output_features = {
        # Define what the model is expected to output, matching keys in compute_loss
        "predicted_forward_states": features["observation.state"],
        "predicted_goal_images": features["observation.image"],
        # Assuming this is an output
        "predicted_backward_states": features["observation.state"],
        # Placeholder, actual is latent dim
        "predicted_goal_latents": features["observation.image"],
    }

    state_dim = features["observation.state"].shape[-1]
    print(f"State dimension: {state_dim}")

    # We'll create normalizers later when setting up the model
    lerobot_dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=None)  # Using base dataset

    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        forward_steps=32,
        backward_steps=32,
        image_key="observation.image",
        state_key="observation.state",
        n_obs_steps=n_obs_steps  # Enable temporal encoding
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True,  # Important for consistent batch sizes
        collate_fn=BidirectionalTrajectoryDataset.collate_fn
    )

    # --- Model Configuration ---
    config = BidirectionalARTransformerConfig(
        state_dim=state_dim,
        hidden_dim=512,  # Single dimension parameter used throughout the model
        num_layers=6,
        num_heads=8,
        dropout=0.1,
        image_channels=3,
        image_size=84,
        output_image_size=96,
        forward_steps=32,
        backward_steps=32,
        n_obs_steps=n_obs_steps,  # Enable temporal encoding
        input_features=input_features,  # Pass the actual FeatureSpec objects
        output_features=output_features,  # Pass the actual FeatureSpec objects
        max_sequence_length=512,  # Adjusted for forward/backward steps
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per iteration: {len(dataloader)}")

    # --- Create normalizers ---
    # Basic normalizer that works with "observation.state" key
    normalize_state_base = Normalize(
        {"observation.state": features["observation.state"]},
        config.normalization_mapping,
        dataset_metadata.stats)

    # Create a key mapping normalizer that maps from batch keys to normalizer keys
    key_mapping = {
        "initial_states": "observation.state",
        "forward_states": "observation.state",
        "backward_states": "observation.state"
    }

    # Wrap the base normalizer with our key mapper
    wrapped_normalizer = KeyMappingNormalizer(
        normalize_state_base, key_mapping)

    # Create the model without internal normalizers (we'll use external normalization)
    model = GoalConditionedAutoregressivePolicy(
        config=config
    )
    model.to(device)
    model.train()  # Set model to training mode

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.95)
    )

    # Scheduler can still be used if desired, adjusting T_max
    # For step-based training, CosineAnnealingLR might need T_max = training_steps
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps, eta_min=learning_rate * 0.1
    )

    # --- Training Loop (Step-based) ---
    print("Starting Bidirectional AR Transformer Training...")
    step = 0
    done = False
    # best_total_loss = float('inf') # Can still track best loss if needed for saving best model

    while not done:
        for batch in tqdm(dataloader, desc=f"Training Step: {step}/{training_steps}"):
            # Move batch to device
            batch = wrapped_normalizer(batch)  # Normalize the batch

            batch_device = {}

            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                else:
                    # Handle cases where batch items might not be tensors (e.g., metadata)
                    batch_device[key] = value

            optimizer.zero_grad()

            # Forward pass
            # We've already normalized the batch with wrapped_normalizer above
            total_loss = model(
                initial_images=batch_device['initial_images'],
                initial_states=batch_device['initial_states'],
                forward_states=batch_device['forward_states'],
                goal_images=batch_device['goal_images'],
                backward_states=batch_device['backward_states'],
                training=True
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()  # Step scheduler

            # Logging
            if step % log_freq == 0:
                # 훨씬 간결해진 로깅
                log_str = f"Step: {step}/{training_steps} | Autoregressive Loss: {total_loss.item():.4f}"
                print(log_str)

                wandb_log = {
                    'train/autoregressive_loss': total_loss.item(),
                    'train/learning_rate': optimizer.param_groups[0]['lr']
                }

                # Log to WandB
                wandb.log(wandb_log, step=step)

            # Checkpointing (similar to rtdiffusion)
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                # Also save the model to WandB
                artifact = wandb.Artifact(f"model-step-{step}", type="model")
                artifact.add_file(str(ckpt_path))
                wandb.log_artifact(artifact)

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "model_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining completed! Final model saved to: {final_path}")

    # Save configuration in lerobot style
    config.save_pretrained(output_directory)
    print(f"Configuration saved to: {output_directory / 'config.json'}")

    # Log the final model to WandB
    artifact = wandb.Artifact(f"model-final", type="model")
    artifact.add_file(str(final_path))
    wandb.log_artifact(artifact)

    # Close wandb run
    wandb.finish()

    # Save dataset statistics (similar to rtdiffusion)
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if isinstance(value, np.ndarray):
                # Ensure tensor and correct dtype
                value = torch.from_numpy(value).float()
            stats_to_save[key] = value
        # Add handling for other types if necessary, e.g. lists of numbers for bounds
        elif isinstance(value, dict) and "min" in value and "max" in value:
            # Example: convert min/max lists to tensors if they are numerical
            try:
                stats_to_save[key] = {
                    "min": torch.tensor(value["min"], dtype=torch.float32),
                    "max": torch.tensor(value["max"], dtype=torch.float32)
                }
            except Exception as e:
                print(
                    f"Warning: Could not convert stats for key '{key}' to tensor: {e}")
                stats_to_save[key] = value  # Save as is if conversion fails

    if stats_to_save:
        try:
            safetensors.torch.save_file(
                stats_to_save, output_directory / "stats.safetensors")
            print(
                f"Dataset stats saved to: {output_directory / 'stats.safetensors'}")
        except Exception as e:
            print(f"Error saving stats.safetensors: {e}")
            # Fallback or alternative saving if safetensors fails for complex structures
            import pickle
            with open(output_directory / "stats.pkl", "wb") as f_pkl:
                pickle.dump(stats_to_save, f_pkl)
            print(
                f"Dataset stats saved to: {output_directory / 'stats.pkl'} as fallback.")


if __name__ == "__main__":
    main()
