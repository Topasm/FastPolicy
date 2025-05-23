import torch
from pathlib import Path
import safetensors.torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.configuration_mymodel import DiffusionConfig
import numpy
import random


def validate_dataset_indices(dataloader, all_state_indices, image_indices):
    """Validate that dataset has enough frames for the required indices."""
    sample_batch = next(iter(dataloader))
    state_seq_length = sample_batch["observation.state"].shape[1]
    image_seq_length = sample_batch["observation.image"].shape[1]

    print("\nDataset validation:")
    print(f"State sequence shape: {sample_batch['observation.state'].shape}")
    print(f"Image sequence shape: {sample_batch['observation.image'].shape}")

    # Check if we have enough state frames
    max_required_state_idx = max(all_state_indices)
    if max_required_state_idx >= state_seq_length:
        print(
            f"WARNING: Required max state index {max_required_state_idx} exceeds available states {state_seq_length-1}")
        return False

    # Check if we have enough image frames
    max_required_img_idx = max(image_indices)
    if max_required_img_idx >= image_seq_length:
        print(
            f"WARNING: Required max image index {max_required_img_idx} exceeds available images {image_seq_length-1}")
        return False

    print("Dataset validation successful")
    return True


def get_random_keyframe_config():
    """Generate a random keyframe configuration with variable horizon length and adaptive keyframes."""
    # Choose a random horizon length between 4 and 32
    horizon = random.randint(4, 32)

    # Select appropriate interpolation mode based on horizon length
    if horizon <= 8:
        mode = "dense"
    elif horizon <= 16:
        mode = "skip_even"
    else:
        mode = "sparse"

    # Generate adaptive keyframes based on horizon length
    keyframes = []

    # Always include the horizon as the furthest keyframe
    keyframes.append(horizon)

    # For medium to long horizons, add intermediate keyframes
    if horizon > 12:
        # Add a midpoint keyframe
        mid_point = horizon // 2
        keyframes.append(mid_point)

        # For very long horizons, add another intermediate point
        if horizon > 24:
            quarter_point = horizon // 4
            keyframes.append(quarter_point)

    # Sort keyframes to ensure they're in ascending order
    keyframes.sort()

    # Choose interpolation method with dynamic probability
    # For longer horizons, favor cubic interpolation for smoother transitions
    cubic_prob = 0.3  # Base probability
    if horizon > 16:
        cubic_prob = 0.5  # Increase probability for long horizons

    interp_method = "cubic" if random.random() < cubic_prob else "linear"

    return horizon, mode, keyframes, interp_method


def main():
    output_directory = Path("outputs/train/diffusion_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 20000
    log_freq = 100
    save_freq = 500

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features needed for diffusion model conditioning and target
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }
    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]
    }

    # Initial configuration (will be updated dynamically during training)
    horizon, interpolation_mode, keyframe_indices, interp_method = get_random_keyframe_config()

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        interpolate_state=True,
        horizon=horizon,
        interpolation_mode=interpolation_mode,
        keyframe_indices=keyframe_indices,
        interpolation_method=interp_method,
    )

    # --- Model ---
    diffusion_model = MyDiffusionModel(cfg)
    diffusion_model.train()
    diffusion_model.to(device)

    # --- Normalization ---
    normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    # Define keyframe indices and target indices for all possible configurations
    # Standard keyframe indices for conditioning
    keyframe_indices = [-1, 0]  # History keyframes

    # Add future keyframes for the largest possible horizon
    max_horizon = 32
    keyframe_indices.extend(list(range(1, max_horizon + 1)))

    # For target indices, we need all possible indices that could be used
    # by any interpolation mode for any horizon in the range 4-32
    target_indices = list(range(max_horizon + 1))

    # Combine all indices and remove duplicates
    all_state_indices = list(set(keyframe_indices + target_indices))
    all_state_indices.sort()

    # Create a range of state indices to request from the dataset
    min_state_index = min(all_state_indices)
    max_state_index = max(all_state_indices)
    state_range = list(range(min_state_index, max_state_index + 1))

    # For images, just use observation history images
    image_indices = [-1, 0]

    print(f"Using state indices range: {min_state_index} to {max_state_index}")
    print(f"Using image indices: {image_indices}")

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
        "observation.state": [i / dataset_metadata.fps for i in state_range],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    # --- Optimizer & Dataloader ---
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    optimizer = torch.optim.Adam(
        diffusion_model.parameters(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # Validate dataset has enough frames for the required indices
    if not validate_dataset_indices(dataloader, all_state_indices, image_indices):
        return  # Exit if validation fails

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Diffusion Model Training...")

    while not done:
        for batch in dataloader:
            # Randomly select configuration for this batch
            batch_horizon, batch_mode, batch_keyframes, batch_interp_method = get_random_keyframe_config()

            # Update the model configuration for this batch
            diffusion_model.config.horizon = batch_horizon
            diffusion_model.config.interpolation_mode = batch_mode
            diffusion_model.config.keyframe_indices = batch_keyframes
            diffusion_model.config.interpolation_method = batch_interp_method

            # Only print config at the start and when logging
            if step % log_freq == 0:
                print(
                    f"Batch config: horizon={batch_horizon}, mode={batch_mode}, keyframes={batch_keyframes}, interp={batch_interp_method}")

            # --- Normalize the batch and move to device ---
            norm_batch = normalize_inputs(batch)
            if "action_is_pad" in batch:
                norm_batch["action_is_pad"] = batch["action_is_pad"]
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # --- Compute Loss ---
            loss = diffusion_model.compute_diffusion_loss(norm_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"diffusion_step_{step}.pth"
                torch.save(diffusion_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "diffusion_final.pth"
    torch.save(diffusion_model.state_dict(), final_path)
    print(f"Training finished. Final diffusion model saved to: {final_path}")

    # --- Save Config and Stats ---
    cfg.save_pretrained(output_directory)

    # Save dataset statistics
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, numpy.ndarray):
            if isinstance(value, numpy.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
