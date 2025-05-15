import torch
from pathlib import Path
import safetensors.torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

# Import the necessary model modules
from model.critic.noise_critic import NoiseCriticConfig, TransformerCritic
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

    # Noise parameters
    noise_type = "progressive"  # Options: "progressive", "diffusion", "uniform"
    base_noise_scale = 0.05
    noise_growth_factor = 1.2  # For progressive noise

    # Improved data augmentation parameters
    augment_positive_examples = True  # Apply small noise to positive examples
    augment_probability = 0.2         # Probability of augmenting positive examples
    use_shuffled_negatives = True     # Create some negatives by trajectory shuffling
    use_temporal_shifts = True        # Apply time shifts to some negative examples

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

    # Set up features similar to diffusion training
    input_features = {
        "observation.state": features["observation.state"],
    }

    # Add image features if needed
    image_key = "observation.image"
    if use_images and image_key in features:
        input_features[image_key] = features[image_key]
        print(f"Using images from key: {image_key}")

    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]
    }

    # Use DiffusionConfig to get consistent horizon and observation parameters
    cfg = NoiseCriticConfig()

    # --- Image Encoder Setup (required) ---
    # Check if image feature is available in the dataset
    if image_key not in features:
        raise ValueError(
            f"Image features ({image_key}) required but not found in dataset features.")

    # Debug print to understand the image feature structure
    print(f"Image feature type: {type(features[image_key])}")
    print(f"Image feature content: {features[image_key]}")

    # Create a modified NoiseCriticConfig for the DiffusionRgbEncoder
    # with the right image_features structure
    encoder_cfg = NoiseCriticConfig()

    # Create a custom feature shape that has the shape attribute
    if isinstance(features[image_key], tuple):
        # For DiffusionRgbEncoder, the image_features need to have a shape attribute
        # Create a class with a shape attribute
        class FeatureShape:
            def __init__(self, shape):
                self.shape = shape

        # Create the image_features dictionary with the right structure
        encoder_cfg.image_features = {
            "observation.image": FeatureShape(features[image_key])
        }
    else:
        encoder_cfg.image_features = {"observation.image": features[image_key]}

    # Initialize the image encoder with the modified config
    image_encoder = DiffusionRgbEncoder(encoder_cfg).to(device)
    image_encoder.eval()  # No need to train the image encoder
    image_feature_dim = encoder_cfg.transformer_dim

    # --- Model Setup ---
    # Create the transformer critic configuration
    critic_cfg = NoiseCriticConfig(
        state_dim=features["observation.state"].shape[0],
        horizon=cfg.horizon,  # Using horizon from diffusion config for consistency
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        use_layernorm=True,
        use_image_context=True,  # Always use image context
        image_feature_dim=image_feature_dim,
        transformer_dim=image_feature_dim,
        # Use the same image_features structure as the encoder_cfg
        image_features=encoder_cfg.image_features,
        n_heads=8
    )

    # Create transformer critic model directly
    critic_model = TransformerCritic(critic_cfg)
    critic_model.to(device)
    print(
        f"Created transformer critic model with {sum(p.numel() for p in critic_model.parameters())} parameters")

    # Try to use torch.compile if available (optional performance boost)
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

    # --- Dataset ---
    # Define time indices similar to diffusion training
    critic_state_indices = list(
        range(0, cfg.horizon))

    # Setup delta timestamps for state and image (both required)
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in critic_state_indices],
        image_key: [
            i / dataset_metadata.fps for i in cfg.observation_delta_indices]
    }

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    print(f"Dataset initialized with {len(dataset)} samples")

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(critic_model.parameters(), lr=learning_rate)
    # Optional: add a learning rate scheduler
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

    # --- Training Setup ---
    criterion = torch.nn.BCEWithLogitsLoss()  # Binary classification loss
    step = 0
    losses = []
    accuracies = []

    # --- Training Loop ---
    print(f"Starting transformer critic training ({training_steps} steps)...")
    print(f"Noise type: {noise_type}, Base scale: {base_noise_scale}" +
          (f", Growth factor: {noise_growth_factor}" if noise_type == "progressive" else ""))
    print("Using improved data augmentation:")
    print("  - Positive examples: Small random noise (20% probability)")
    print("  - Negative examples: Mix of techniques:")
    print("    * Inter-batch trajectory shuffling (30% of examples)")
    print("    * Noise-based corruption (all examples)")
    print("    * Temporal shifts (30% probability with 50% of batch affected)")
    print("  Note: Multiple augmentations may be applied to the same example")

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

            # Extract state trajectories - use only the future states
            all_state_trajectories = norm_batch["observation.state"][:,
                                                                     0:cfg.horizon]
            B, H, D_state = all_state_trajectories.shape

            # Create positive state trajectories
            positive_state_trajectory = all_state_trajectories.clone()

            # Apply optional minor data augmentation to positive examples
            if torch.rand(1).item() < 0.2:  # 20% probability
                # Apply minimal noise to positive examples
                tiny_noise_scale = base_noise_scale * 0.1  # 10x smaller noise
                for t_step in range(1, H):
                    tiny_noise = torch.randn_like(
                        positive_state_trajectory[:, t_step]) * tiny_noise_scale
                    positive_state_trajectory[:, t_step] += tiny_noise

            # Create base for negative trajectories - use original trajectories
            # For some examples (30%), create completely shuffled trajectories
            negative_state_trajectory = all_state_trajectories.clone()
            shuffle_mask = torch.rand(B) < 0.3
            if shuffle_mask.any():
                # Get the number of examples to shuffle
                num_to_shuffle = shuffle_mask.sum().item()

                # Create a single set of shuffled indices (not a stack)
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
                for t_step in range(1, H):  # Start from the second state
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

            # Apply random temporal shifts to some negative examples (30% probability)
            # This helps the model learn temporal coherence importance
            if torch.rand(1).item() < 0.3:
                shift_mask = torch.rand(B) < 0.5
                if shift_mask.any():
                    # Get the indices of elements where shift_mask is True
                    mask_indices = torch.where(shift_mask)[0]

                    # For elements where shift_mask is True, shift the trajectory in time
                    # Ensure minimum shift of 1 and maximum of 1/4 of horizon length
                    max_shift = max(1, H//4)

                    # For each index where shift_mask is True, apply a temporal shift
                    for idx in mask_indices:
                        # Generate a random shift amount for this example
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

            # Process images (required for TransformerCritic)
            if image_key not in batch:
                raise ValueError(
                    f"Image features ({image_key}) required but not found in batch.")

            with torch.no_grad():
                # Extract observation images
                # Can be (B, C, H, W) or (B, T_img, C, H, W)
                images = batch[image_key]
                print(f"Batch image shape: {images.shape}")

                # Handle both 4D and 5D tensors
                if len(images.shape) == 4:
                    # Single image per batch item: (B, C, H, W)
                    # Add a time dimension: (B, 1, C, H, W)
                    images = images.unsqueeze(1)
                    print(f"Added time dimension. New shape: {images.shape}")
                elif len(images.shape) != 5:
                    raise ValueError(
                        f"Image shape incorrect. Expected 4D or 5D tensor, got shape {images.shape}")

                # Process images with the encoder
                B_i, T_img = images.shape[:2]
                images_flat = images.flatten(0, 1)  # (B*T_img, C, H, W)
                print(f"Flattened image shape: {images_flat.shape}")
                image_features_flat = image_encoder(images_flat)

                # Reshape back and use the last frame features
                image_features_seq = image_features_flat.view(B_i, T_img, -1)
                # Last frame
                image_features = image_features_seq[:, -1]

            # Combine trajectories and create labels
            combined_trajectories = torch.cat(
                [positive_state_trajectory, negative_state_trajectory], dim=0
            )  # (2*B, H, D_state)

            # Duplicate image features for combined trajectories
            combined_image_features = torch.cat(
                [image_features, image_features], dim=0
            )  # (2*B, feature_dim)

            # Create labels (1=original, 0=noisy)
            labels = torch.cat([
                torch.ones(B, device=device),    # Positive examples (original)
                torch.zeros(B, device=device)    # Negative examples (noisy)
            ]).float()

            # Zero gradients and compute loss
            optimizer.zero_grad()

            # Forward pass
            logits = critic_model(
                trajectory_sequence=combined_trajectories,
                image_features=combined_image_features
            )  # (2*B, 1)

            # Compute loss and backpropagate
            loss = criterion(logits.squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log progress
            if step % log_freq == 0:
                # Calculate accuracy
                preds = (torch.sigmoid(logits.squeeze(-1)) > 0.5).float()
                acc = (preds == labels).float().mean().item()

                # Store metrics
                losses.append(loss.item())
                accuracies.append(acc)

                # Print metrics
                current_lr = lr_scheduler.get_last_lr()[0]
                print(
                    f"Step {step}/{training_steps}: Loss={loss.item():.4f}, Acc={acc:.4f}, LR={current_lr:.6f}")

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

    # Filter stats to include only tensors
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items()
        if isinstance(v, torch.Tensor)
    }
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
