#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/train_multimodal_autoregressive.py
import torch
from pathlib import Path
import numpy as np
import safetensors.torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictorConfig
from model.predictor.multimodal_autoregressive_predictor import MultimodalAutoregressivePredictor

# Temporarily enable anomaly detection for debugging gradient issues
torch.autograd.set_detect_anomaly(True)


def create_causal_mask(seq_len, device):
    """Create a causal mask for autoregressive generation"""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device)
                      * float('-inf'), diagonal=1)
    return mask


def main():
    output_directory = Path("outputs/train/multimodal_autoregressive")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 20000
    log_freq = 200
    save_freq = 5000
    batch_size = 128

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Simplified configuration
    # For autoregressive prediction from state 0-64
    # Images at 0, 8, 16, 24, 32
    state_indices = list(range(65))  # 0-64
    image_indices = [0, 8, 16, 24, 32, 64]  # Images at these specific indices

    print(f"Using state indices: {state_indices}")
    print(f"Using image indices: {image_indices}")

    # Convert indices to timestamps
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in state_indices],
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
    }

    # Features needed for future prediction model
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    # Initialize dataset with timestamps
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # We'll use 8-step windows for predictions (like the diffusion model)
    window_size = 8

    # Configure the model for 8-step prediction
    cfg = MultimodalFuturePredictorConfig(
        state_dim=features["observation.state"].shape[0],
        horizon=window_size + 1,  # +1 for the previous state
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        future_steps=window_size,
        predict_uncertainty=False
    )

    # --- Create Model ---
    model = MultimodalAutoregressivePredictor(cfg)
    model.train()
    model.to(device)

    # --- Normalization ---
    normalize_inputs = Normalize(
        input_features, {"observation.state": "standard",
                         "observation.image": "none"},
        dataset_metadata.stats
    )

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=device.type == "cuda", drop_last=True
    )

    # --- Loss functions ---
    state_loss_fn = torch.nn.MSELoss()
    image_loss_fn = torch.nn.L1Loss()

    step = 0
    done = False
    print("Starting Autoregressive Future Prediction Training...")

    # Settings for autoregressive training
    generate_noise = False  # Toggle between noise and image prediction

    while not done:
        for batch in dataloader:
            # Normalize data and move to device
            norm_batch = normalize_inputs(batch)
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # Extract states and images
            states = norm_batch["observation.state"]
            images = norm_batch["observation.image"]

            # Reset gradients
            optimizer.zero_grad()

            # Compute loss for each segment
            total_loss = 0.0

            # Train autoregressive prediction for each window
            # For multiple windows: 0-8, 8-16, 16-24, 24-32, etc.
            state_losses = []
            image_losses = []

            # Autoregressive segments from 0 to max window
            # This handles segments: (0-8), (8-16), (16-24), etc.
            # Avoid going beyond 64 states (65 total indices)
            max_segments = min(7, (len(state_indices) - 1) // window_size)

            # Initialize variables to store autoregressive predictions
            prev_predicted_states = None
            prev_predicted_images = {}  # Dictionary to store images by timestamp

            for segment in range(max_segments):
                start_idx = segment * window_size
                end_idx = (segment + 1) * window_size
                target_idx = end_idx

                # For the first segment, use ground truth data
                if segment == 0:
                    # Take states from 0 to window_size (inclusive)
                    context_states = states[:, :window_size + 1].clone()
                    # First image (at time 0)
                    current_image = images[:, 0].clone()
                else:
                    # For subsequent segments, use a mix of real and predicted data
                    # Get base context from the dataset
                    context_states = states[:, start_idx-1:end_idx + 1]

                    # Replace context states with predictions where available
                    if prev_predicted_states is not None:
                        # Create a clone of context_states to avoid in-place operation issues
                        modified_context_states = context_states.clone()
                        # Replace the first state in this context with our previous prediction
                        modified_context_states[:, 0] = prev_predicted_states
                        context_states = modified_context_states

                    # Find appropriate image for current context
                    # Use the most recent image timestamp less than or equal to start_idx
                    prev_image_idx = 0
                    prev_image_time = 0

                    # Find the most recent predicted or available image timestamp
                    for img_time in image_indices:
                        if img_time <= start_idx and img_time > prev_image_time:
                            prev_image_time = img_time
                            # If we have a predicted image at this time, use it
                            if img_time in prev_predicted_images:
                                # Use a clone to avoid in-place operation issues
                                current_image = prev_predicted_images[img_time].clone(
                                )
                                prev_image_idx = None  # Mark that we're using a prediction
                                break
                            # Otherwise note this as a potential image to use from ground truth
                            prev_image_idx = image_indices.index(img_time)

                    # If we didn't find a predicted image, use ground truth
                    if prev_image_idx is not None:
                        current_image = images[:, prev_image_idx].clone()

                # Target state is the last state in this window
                future_target = states[:, target_idx]
                future_image_target = None

                # If the target index is in image_indices, we also predict an image
                if target_idx in image_indices:
                    img_idx = image_indices.index(target_idx)
                    future_image_target = images[:, img_idx]

                # Run forward pass with full autoregressive causal masking
                # The MultimodalAutoregressivePredictor applies causal masks to:
                # 1. The transformer encoder - ensuring states only attend to previous states
                # 2. The image decoder - ensuring each patch only attends to previous patches
                predicted_states, predicted_future, _, _ = model.predict_future_trajectory(
                    context_states, current_image, generate_noise=generate_noise
                )

                # Store predictions for next autoregressive step
                # Make a detached clone to avoid in-place operation issues
                prev_predicted_states = predicted_states.clone().detach()

                # Store image prediction if not generating noise and target is available
                if future_image_target is not None and not generate_noise:
                    # Make a detached clone to avoid in-place operation issues
                    prev_predicted_images[target_idx] = predicted_future.clone(
                    ).detach()

                # Always compute state loss
                curr_state_loss = state_loss_fn(
                    predicted_states, future_target)
                state_losses.append(curr_state_loss)
                total_loss += curr_state_loss

                # Compute image loss if applicable
                if future_image_target is not None and not generate_noise:
                    # Ensure target image is the same size as predicted
                    if future_image_target.shape != predicted_future.shape:
                        future_image_target = torch.nn.functional.interpolate(
                            future_image_target,
                            size=predicted_future.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        )

                    # Normalize target image to match prediction range if using tanh
                    if future_image_target.min() >= 0 and future_image_target.max() <= 1:
                        future_image_target = future_image_target * 2 - 1

                    curr_image_loss = image_loss_fn(
                        predicted_future, future_image_target)
                    image_loss_weight = 0.1
                    image_losses.append(curr_image_loss)
                    total_loss += curr_image_loss * image_loss_weight

            # Toggle noise generation for next iteration
            generate_noise = not generate_noise

            # Backward pass and update
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Log progress
            if step % log_freq == 0:
                log_message = f"Step: {step}/{training_steps} Total Loss: {total_loss.item():.4f}"
                if state_losses:
                    avg_state_loss = sum(
                        loss_val.item() for loss_val in state_losses) / len(state_losses)
                    log_message += f", Avg State Loss: {avg_state_loss:.4f}"
                if image_losses:
                    avg_image_loss = sum(
                        loss_val.item() for loss_val in image_losses) / len(image_losses)
                    log_message += f", Avg Image Loss: {avg_image_loss:.4f}"
                print(log_message)

            # Save checkpoints
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step': step,
                    'config': cfg.__dict__,
                }, ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "multimodal_autoregressive_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': cfg.__dict__,
        'step': step,
    }, final_path)
    print(f"Training finished. Final model saved to: {final_path}")

    # Also save just the model weights for easy loading
    torch.save(model.state_dict(), output_directory /
               "multimodal_autoregressive_weights.pt")

    # --- Save Config and Stats ---
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    safetensors.torch.save_file(stats_to_save, str(
        output_directory / "stats.safetensors"))
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
