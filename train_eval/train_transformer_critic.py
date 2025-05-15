#!/usr/bin/env python
import torch
import numpy
from pathlib import Path
import safetensors.torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

# Import the necessary model modules
from model.critic.ciritic_modules import NoiseCriticConfig, TransformerCritic


def main():
    # --- Configuration ---
    output_directory = Path("outputs/train/transformer_critic")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Training parameters
    training_steps = 10000
    log_freq = 50
    save_freq = 500
    batch_size = 64
    learning_rate = 1e-4

    # Noise parameters for generating negative examples
    noise_type = "progressive"  # Options: "progressive", "diffusion", "uniform"
    base_noise_scale = 0.05
    noise_growth_factor = 1.2  # For progressive noise

    # Model parameters
    hidden_dim = 512
    num_layers = 4

    # Dataset parameters
    dataset_repo_id = "lerobot/pusht"

    # --- Dataset Metadata Setup ---
    print(f"Loading dataset metadata for {dataset_repo_id}...")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Check if image feature is available in the dataset
    image_key = "observation.image"
    if image_key not in features:
        raise ValueError(
            f"Image features ({image_key}) required but not found in dataset features.")

    # --- Image Feature Configuration ---
    # Define the structure of the image features
    if isinstance(features[image_key], tuple):
        image_features_dict = {
            "observation.image": features[image_key]
        }
    else:
        image_features_dict = {"observation.image": features[image_key]}

    # --- Critic Model Setup ---
    print("Initializing transformer critic model...")
    critic_cfg = NoiseCriticConfig(
        state_dim=features["observation.state"].shape[0],
        horizon=16,  # Default horizon
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        use_layernorm=True,
        use_image_context=True,
        image_feature_dim=hidden_dim,  # For ViT-like patching
        transformer_dim=hidden_dim,
        image_features=image_features_dict,
        n_heads=8
    )

    critic_model = TransformerCritic(critic_cfg)
    critic_model.to(device)
    print(
        f"Created transformer critic with {sum(p.numel() for p in critic_model.parameters())} parameters")

    # Try to use torch.compile if available
    try:
        if hasattr(torch, "compile"):
            print("Compiling transformer critic model...")
            critic_model = torch.compile(critic_model)
            print("Model compiled successfully.")
    except Exception as e:
        print(
            f"Model compilation failed: {e}. Continuing with standard model.")

    # --- Normalization Setup ---
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]},
        critic_cfg.normalization_mapping,
        dataset_metadata.stats
    )

    # --- Dataset Setup ---
    # Define time indices for state and image features
    critic_state_indices = list(range(0, critic_cfg.horizon))

    # Setup delta timestamps
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in critic_state_indices],
        # Just need the current image
        image_key: [i / dataset_metadata.fps for i in [0]]
    }

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    print(f"Dataset initialized with {len(dataset)} samples")

    # --- Training Setup ---
    optimizer = torch.optim.AdamW(critic_model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=training_steps)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # --- Training Loop ---
    step = 0
    losses = []
    accuracies = []

    print(f"Starting transformer critic training ({training_steps} steps)...")
    print(f"Noise type: {noise_type}, Base scale: {base_noise_scale}" +
          (f", Growth factor: {noise_growth_factor}" if noise_type == "progressive" else ""))

    while step < training_steps:
        for batch in dataloader:
            # First normalize the state data (on CPU)
            state_batch = {"observation.state": batch["observation.state"]}
            norm_state_batch = normalize_state(state_batch)

            # Move batch to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Create normalized batch on device
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in batch.items()}  # batch is already on device

            # Replace with normalized state (which was normalized on CPU)
            norm_batch["observation.state"] = norm_state_batch["observation.state"].to(
                device)

            # --- Compute Critic Loss ---
            optimizer.zero_grad()

            # Configure noise parameters
            noise_params = {
                "base_noise_scale": base_noise_scale,
                "noise_type": noise_type,
                "noise_growth_factor": noise_growth_factor
            }

            # Pass normalized batch directly to the critic model
            # The critic will handle image patching internally (ViT-like approach)
            loss, accuracy = critic_model.compute_binary_classification_loss(
                norm_batch=norm_batch,
                noise_params=noise_params,
                image_features=None  # No pre-encoded features, use raw images
            )

            # Backprop and optimize
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log progress
            if step % log_freq == 0:
                # Store metrics
                losses.append(loss.item())
                accuracies.append(accuracy.item())

                # Print metrics
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"Step {step}/{training_steps}: Loss={loss.item():.4f}, Acc={accuracy.item():.4f}, LR={current_lr:.6f}")

            # Save checkpoint
            if step % save_freq == 0 and step > 0:
                checkpoint_path = output_directory / \
                    f"transformer_critic_{step}.pth"
                torch.save(critic_model.state_dict(), checkpoint_path)
                print(f"Saved checkpoint to {checkpoint_path}")

            # Update step counter
            step += 1
            if step >= training_steps:
                break

    # --- Save Final Model ---
    final_path = output_directory / "transformer_critic_final.pth"
    torch.save(critic_model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # --- Save Config and Stats ---
    # Save the critic config - handle non-serializable objects
    config_dict = {}
    for k, v in critic_cfg.__dict__.items():
        # Handle special cases that aren't JSON serializable
        if k == "image_features":
            # Convert PolicyFeature objects to their string representations
            image_features_dict = {}
            for img_key, feature in v.items():
                # Store the shape info if it's a tuple
                if hasattr(feature, "shape"):
                    image_features_dict[img_key] = list(feature.shape)
                else:
                    # Just store the string representation as fallback
                    image_features_dict[img_key] = str(feature)
            config_dict[k] = image_features_dict
        elif k == "normalization_mapping":
            # Convert enum values to strings
            norm_dict = {}
            for norm_key, norm_val in v.items():
                norm_dict[norm_key] = str(norm_val)
            config_dict[k] = norm_dict
        else:
            # Handle basic types that are JSON serializable
            try:
                # Test if the value is JSON serializable
                import json
                json.dumps(v)
                config_dict[k] = v
            except (TypeError, OverflowError):
                # Fall back to string representation for non-serializable objects
                config_dict[k] = str(v)

    with open(output_directory / "config.json", "w") as f:
        import json
        json.dump(config_dict, f, indent=4)

    # Filter stats to only include tensors
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, numpy.ndarray):
            # Convert numpy arrays to tensors if needed
            if isinstance(value, numpy.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    # Save the stats using safetensors
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
