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
from model.diffusion.diffusion_modules import DiffusionRgbEncoder


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
    use_images = True  # Always use image features with TransformerCritic

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

    # --- Create Base Config ---
    cfg = NoiseCriticConfig(
        state_dim=features["observation.state"].shape[0],
        horizon=16,  # Default horizon
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        use_layernorm=True,
        use_image_context=True,
        transformer_dim=hidden_dim,
    )

    # --- Create Custom Feature Shape for DiffusionRgbEncoder ---
    # For DiffusionRgbEncoder, the image_features needs to have a shape attribute
    class FeatureShape:
        def __init__(self, shape):
            self.shape = shape

    # Create a modified config for the image encoder
    encoder_cfg = NoiseCriticConfig()
    if isinstance(features[image_key], tuple):
        encoder_cfg.image_features = {
            "observation.image": FeatureShape(features[image_key])
        }
    else:
        encoder_cfg.image_features = {"observation.image": features[image_key]}

    # --- Image Encoder Setup ---
    print("Initializing image encoder...")
    image_encoder = DiffusionRgbEncoder(encoder_cfg).to(device)
    image_encoder.eval()  # No need to train the image encoder
    image_feature_dim = encoder_cfg.transformer_dim

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
        image_feature_dim=image_feature_dim,
        transformer_dim=image_feature_dim,
        image_features=encoder_cfg.image_features,
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
        cfg.normalization_mapping,
        dataset_metadata.stats
    )

    # --- Dataset Setup ---
    # Define time indices for state and image features
    critic_state_indices = list(range(0, cfg.horizon))

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
                          for k, v in batch.items()}

            # Replace with normalized state
            norm_batch["observation.state"] = norm_state_batch["observation.state"].to(
                device)

            # Extract state trajectories
            all_state_trajectories = norm_batch["observation.state"][:, 0:cfg.horizon]
            B, H, D_state = all_state_trajectories.shape

            # Create positive state trajectories (original or slightly perturbed)
            positive_state_trajectory = all_state_trajectories.clone()

            # Apply minor data augmentation to positive examples (20% probability)
            if torch.rand(1).item() < 0.2:
                tiny_noise_scale = base_noise_scale * 0.1  # 10x smaller noise
                for t_step in range(1, H):
                    tiny_noise = torch.randn_like(
                        positive_state_trajectory[:, t_step]) * tiny_noise_scale
                    positive_state_trajectory[:, t_step] += tiny_noise

            # Create negative state trajectories (corrupted with noise)
            negative_state_trajectory = all_state_trajectories.clone()

            # Apply shuffling to some negative examples (30% of batch)
            shuffle_mask = torch.rand(B) < 0.3
            if shuffle_mask.any():
                # Get the number of examples to shuffle
                num_to_shuffle = shuffle_mask.sum().item()

                # Create a single set of shuffled indices
                shuffle_indices = torch.randperm(B)[:num_to_shuffle]

                # Get the indices of elements where shuffle_mask is True
                mask_indices = torch.where(shuffle_mask)[0]

                # For each index where shuffle_mask is True, assign a trajectory from a different batch element
                for i, idx in enumerate(mask_indices):
                    # Get a random index from the shuffle_indices
                    random_idx = shuffle_indices[i % len(shuffle_indices)]
                    # Make sure we're not copying the same trajectory
                    if random_idx == idx:
                        random_idx = (random_idx + 1) % B
                    # Copy the trajectory
                    negative_state_trajectory[idx] = all_state_trajectories[random_idx]

            # Apply noise to negative trajectories based on specified noise type
            if noise_type == "progressive":
                # Apply progressive noise (increasing with each timestep)
                current_noise_scale = base_noise_scale
                for t_step in range(1, H):
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * current_noise_scale
                    negative_state_trajectory[:, t_step] += noise
                    current_noise_scale *= noise_growth_factor

            elif noise_type == "diffusion":
                # Apply diffusion-like noise (stronger at the end of the sequence)
                for t_step in range(1, H):
                    # Calculate noise scale based on position in sequence
                    timestep_fraction = t_step / (H - 1)  # 0 to 1
                    noise_scale = base_noise_scale * \
                        (1.0 + 10 * timestep_fraction**2)
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * noise_scale
                    negative_state_trajectory[:, t_step] += noise

            elif noise_type == "uniform":
                # Apply uniform noise across all timesteps
                for t_step in range(1, H):
                    noise = torch.randn_like(
                        negative_state_trajectory[:, t_step]) * base_noise_scale
                    negative_state_trajectory[:, t_step] += noise

            # Apply temporal shifts to some negative examples (30% probability)
            if torch.rand(1).item() < 0.3:
                shift_mask = torch.rand(B) < 0.5
                if shift_mask.any():
                    # Get the indices of elements where shift_mask is True
                    mask_indices = torch.where(shift_mask)[0]

                    # Maximum shift is 1/4 of horizon length
                    max_shift = max(1, H//4)

                    for idx in mask_indices:
                        # Generate random shift amount
                        shift = torch.randint(1, max_shift + 1, (1,)).item()
                        shift = min(shift, H-1)  # Safety check

                        # Create a shifted version of this trajectory
                        trajectory = negative_state_trajectory[idx].clone()

                        # Apply the shift: move states ahead
                        negative_state_trajectory[idx,
                                                  :-shift] = trajectory[shift:]

                        # Repeat the last state for the remaining positions
                        last_state = trajectory[-1]
                        negative_state_trajectory[idx, -
                                                  shift:] = last_state.unsqueeze(0).repeat(shift, 1)

            # Process images to get image features
            with torch.no_grad():
                # Extract observation images
                images = batch[image_key]

                # Print shape for debugging
                print(f"Original image shape: {images.shape}")

                # Handle different image shapes
                if len(images.shape) == 5:  # (B, T_img, C, H, W)
                    B_i, T_img = images.shape[:2]

                    # Check if images need to be normalized to [0,1]
                    if images.max() > 1.0:
                        images = images / 255.0

                    # Flatten batch and time dimensions
                    # (B*T_img, C, H, W)
                    images_flat = images.reshape(-1, *images.shape[2:])

                elif len(images.shape) == 4:  # (B, C, H, W) - only one timestep
                    B_i = images.shape[0]
                    T_img = 1

                    # Check if images need to be normalized to [0,1]
                    if images.max() > 1.0:
                        images = images / 255.0

                    # Already in the right format for processing
                    images_flat = images

                # Print shape for debugging
                print(f"Processed image shape: {images_flat.shape}")

                # Extract features using the image encoder
                image_features_flat = image_encoder(images_flat)

                # Handle the reshaping based on original shape
                if len(images.shape) == 5:  # (B, T_img, C, H, W)
                    # Reshape back to sequence format
                    image_features_seq = image_features_flat.view(
                        B_i, T_img, -1)
                    # Last frame
                    image_features = image_features_seq[:, -1]
                else:  # (B, C, H, W)
                    # Already flat, no need to take the last frame
                    image_features = image_features_flat

            # Calculate loss using the simplified critic loss computation
            optimizer.zero_grad()
            loss, accuracy = critic_model.compute_critic_loss(
                positive_trajectories=positive_state_trajectory,
                negative_trajectories=negative_state_trajectory,
                image_features=image_features
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
    # Save the critic config
    config_dict = {k: v for k, v in critic_cfg.__dict__.items()}
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
