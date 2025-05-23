#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/eval_multimodal_future.py

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictor, MultimodalFuturePredictorConfig

# Add matplotlib settings for better visualizations
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained MultimodalFuturePredictor model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Name of the dataset to use for evaluation")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to evaluate on")
    parser.add_argument("--context_horizon", type=int, default=8,
                        help="Length of trajectory context used for prediction")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/multimodal_future",
                        help="Directory to save evaluation results and visualizations")

    return parser.parse_args()


def calculate_ssim(img1, img2):
    """
    Calculate Structural Similarity Index (SSIM) between two images
    Assumes images are in range [-1, 1] or [0, 1]
    """
    from skimage.metrics import structural_similarity as ssim

    # Convert to numpy arrays in range [0, 1]
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        if img1.min() < 0:  # If in range [-1, 1]
            img1 = (img1 + 1) / 2

    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
        if img2.min() < 0:  # If in range [-1, 1]
            img2 = (img2 + 1) / 2

    # Ensure images are in HWC format
    if img1.shape[0] == 3:  # If in CHW format
        img1 = np.transpose(img1, (1, 2, 0))

    if img2.shape[0] == 3:  # If in CHW format
        img2 = np.transpose(img2, (1, 2, 0))

    # Calculate SSIM
    return ssim(img1, img2, multichannel=True, data_range=1.0)


def calculate_psnr(img1, img2):
    """
    Calculate Peak Signal-to-Noise Ratio (PSNR) between two images
    Assumes images are in range [-1, 1] or [0, 1]
    """
    from skimage.metrics import peak_signal_noise_ratio as psnr

    # Convert to numpy arrays in range [0, 1]
    if torch.is_tensor(img1):
        img1 = img1.detach().cpu().numpy()
        if img1.min() < 0:  # If in range [-1, 1]
            img1 = (img1 + 1) / 2

    if torch.is_tensor(img2):
        img2 = img2.detach().cpu().numpy()
        if img2.min() < 0:  # If in range [-1, 1]
            img2 = (img2 + 1) / 2

    # Ensure images are in HWC format
    if img1.shape[0] == 3:  # If in CHW format
        img1 = np.transpose(img1, (1, 2, 0))

    if img2.shape[0] == 3:  # If in CHW format
        img2 = np.transpose(img2, (1, 2, 0))

    # Calculate PSNR
    return psnr(img1, img2, data_range=1.0)


def visualize_predictions(current_image, predicted_image, target_image, current_state, predicted_state,
                          target_state, state_uncertainty=None, image_uncertainty=None, output_path=None):
    """
    Visualize predictions alongside targets.

    Args:
        current_image: Current image tensor [C, H, W]
        predicted_image: Predicted future image tensor [C, H, W]
        target_image: Target future image tensor [C, H, W]
        current_state: Current state tensor [D]
        predicted_state: Predicted future state tensor [D] or [T, D] for multi-step
        target_state: Target future state tensor [D] or [T, D] for multi-step
        state_uncertainty: Predicted state uncertainty tensor [D] or [T, D] or None
        image_uncertainty: Predicted image uncertainty tensor [C, H, W] or None
        output_path: Path to save the visualization or None to display
    """
    # Convert tensors to numpy arrays for visualization
    def to_np(x): return x.detach().cpu().numpy() if torch.is_tensor(x) else x

    current_image_np = to_np(current_image)
    predicted_image_np = to_np(predicted_image)
    target_image_np = to_np(target_image)
    current_state_np = to_np(current_state)
    predicted_state_np = to_np(predicted_state)
    target_state_np = to_np(target_state)

    # Convert images from [-1, 1] to [0, 1] if needed
    if current_image_np.min() < 0:
        current_image_np = (current_image_np + 1) / 2
        predicted_image_np = (predicted_image_np + 1) / 2
        target_image_np = (target_image_np + 1) / 2

    # Determine if multi-step predictions
    multi_step = len(
        predicted_state_np.shape) > 1 and predicted_state_np.shape[0] > 1

    # Create a figure with two rows:
    # Row 1: Images (current, predicted, target)
    # Row 2: States (current, predicted, target)
    fig = plt.figure(figsize=(18, 10))

    # --- First row: Images ---
    ax1 = plt.subplot2grid((2, 6), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((2, 6), (0, 2), colspan=2)
    ax3 = plt.subplot2grid((2, 6), (0, 4), colspan=2)

    # Plot images
    ax1.imshow(np.transpose(current_image_np, (1, 2, 0)))
    ax1.set_title("Current Image")
    ax1.axis('off')

    ax2.imshow(np.transpose(predicted_image_np, (1, 2, 0)))
    ax2.set_title("Predicted Future Image")
    ax2.axis('off')

    ax3.imshow(np.transpose(target_image_np, (1, 2, 0)))
    ax3.set_title("Target Future Image")
    ax3.axis('off')

    # Calculate image metrics
    ssim_val = calculate_ssim(predicted_image_np, target_image_np)
    psnr_val = calculate_psnr(predicted_image_np, target_image_np)
    fig.suptitle(
        f"Image Metrics: SSIM = {ssim_val:.4f}, PSNR = {psnr_val:.2f} dB", fontsize=16)

    # --- Second row: States ---
    if multi_step:
        # For multi-step predictions
        ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=6)
        num_steps = predicted_state_np.shape[0]
        state_dim = predicted_state_np.shape[1]

        # Create x-axis for plotting
        x_current = np.array([-1])  # Current state at t-1
        x_future = np.arange(num_steps)  # Future states at t=0,1,2,...

        # Plot each state dimension
        for d in range(min(state_dim, 5)):  # Plot up to 5 dimensions to avoid clutter
            ax4.plot(
                x_current, current_state_np[d], 'o', color=f'C{d}', label=f'Current {d}')
            ax4.plot(x_future, predicted_state_np[:, d], '-',
                     marker='o', color=f'C{d}', label=f'Predicted {d}')
            ax4.plot(x_future, target_state_np[:, d], '--',
                     marker='x', color=f'C{d}', label=f'Target {d}')

            # Plot uncertainty if available
            if state_uncertainty is not None:
                uncertainty_np = to_np(state_uncertainty)
                ax4.fill_between(
                    x_future,
                    predicted_state_np[:, d] - 2*np.sqrt(uncertainty_np[:, d]),
                    predicted_state_np[:, d] + 2*np.sqrt(uncertainty_np[:, d]),
                    alpha=0.2, color=f'C{d}'
                )

        ax4.set_title("State Trajectory Prediction")
        ax4.set_xlabel("Future Steps")
        ax4.set_ylabel("State Value")
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True)
    else:
        # For single-step prediction
        ax4 = plt.subplot2grid((2, 6), (1, 0), colspan=3)
        ax5 = plt.subplot2grid((2, 6), (1, 3), colspan=3)

        # Bar chart of current and future states
        state_indices = np.arange(len(current_state_np))
        bar_width = 0.25

        # Current state
        ax4.bar(state_indices - bar_width, current_state_np,
                bar_width, label='Current')
        # Predicted future state
        ax4.bar(state_indices, predicted_state_np,
                bar_width, label='Predicted')
        # Target future state
        ax4.bar(state_indices + bar_width,
                target_state_np, bar_width, label='Target')

        ax4.set_title("State Comparison")
        ax4.set_xlabel("State Dimension")
        ax4.set_ylabel("State Value")
        ax4.set_xticks(state_indices)
        ax4.legend()

        # Calculate MSE
        mse = np.mean((predicted_state_np - target_state_np) ** 2)
        ax4.set_title(f"State Comparison (MSE = {mse:.4f})")

        # Plot state prediction error
        error = np.abs(predicted_state_np - target_state_np)
        ax5.bar(state_indices, error, color='red', alpha=0.7)

        # Plot uncertainty if available
        if state_uncertainty is not None:
            uncertainty_np = to_np(state_uncertainty)
            # Plot uncertainty bars
            ax5.errorbar(state_indices, error, yerr=2*np.sqrt(uncertainty_np),
                         fmt='none', capsize=5, color='black', alpha=0.5)

        ax5.set_title("Prediction Error by Dimension")
        ax5.set_xlabel("State Dimension")
        ax5.set_ylabel("Absolute Error")
        ax5.set_xticks(state_indices)

    plt.tight_layout()

    # Save or show the visualization
    if output_path is not None:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load Model ---
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)

    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = MultimodalFuturePredictorConfig(**config_dict)
    else:
        raise ValueError("Config not found in checkpoint")

    # Initialize model
    model = MultimodalFuturePredictor(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Model loaded successfully:")
    print(f"  - State dimension: {config.state_dim}")
    print(f"  - Hidden dimension: {config.hidden_dim}")
    print(f"  - Num layers: {config.num_layers}")
    print(f"  - Num heads: {config.num_heads}")
    print(f"  - Multi-step prediction: {config.multi_step_prediction}")
    if config.multi_step_prediction:
        print(f"  - Num future steps: {config.num_future_steps}")
    print(f"  - Predict uncertainty: {config.predict_uncertainty}")

    # --- Dataset Setup ---
    print(f"Loading dataset metadata for {args.dataset}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Configure dataset based on features
    context_horizon = args.context_horizon
    future_steps = config.future_steps

    # Total horizon depends on prediction type
    if config.multi_step_prediction and config.num_future_steps > 1:
        total_horizon = context_horizon + \
            max(future_steps, config.num_future_steps)
    else:
        total_horizon = context_horizon + future_steps

    # Setup delta timestamps for trajectory features
    timestamps = [i / dataset_metadata.fps for i in range(total_horizon)]

    delta_timestamps = {
        "observation.state": timestamps,
        "observation.image": [timestamps[0], timestamps[context_horizon], timestamps[-1]]
    }

    # Initialize dataset
    print("Initializing dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset initialized with {len(dataset)} samples")

    # Create a small dataloader for evaluation
    eval_loader = DataLoader(
        dataset,
        batch_size=1,  # Process one sample at a time for visualization
        shuffle=True,
        num_workers=1
    )

    # --- Normalization Setup ---
    normalize_inputs = Normalize(
        {"observation.state": features["observation.state"],
         "observation.image": features["observation.image"]},
        {"observation.state": "standard",
         "observation.image": "none"},
        dataset_metadata.stats
    )

    # --- Evaluation Metrics ---
    metrics = {
        'state_mse': [],
        'image_ssim': [],
        'image_psnr': []
    }

    # Evaluate on selected samples
    print(f"Evaluating on {args.num_samples} samples...")

    with torch.no_grad():
        for idx, batch in enumerate(eval_loader):
            if idx >= args.num_samples:
                break

            # Normalize the data
            norm_batch = normalize_inputs(batch)
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # Extract the states and images
            states = norm_batch["observation.state"]  # [B, H, D_state]
            images = norm_batch["observation.image"]  # [B, T_img, C, H, W]

            # Split into context and future
            # [B, context_horizon, D_state]
            context_states = states[:, :context_horizon]
            # [B, future_steps, D_state]
            future_states = states[:, context_horizon:]

            # Get current and target future images
            current_image = images[:, 0]  # [B, C, H, W]
            target_future_image = images[:, 2]  # [B, C, H, W]

            # Make predictions
            predicted_state, predicted_image, state_uncertainty, image_uncertainty = model.predict_future_trajectory(
                context_states, current_image)

            # Get target state based on prediction type
            if config.multi_step_prediction:
                # For multi-step, use multiple future states
                num_pred_steps = min(
                    config.num_future_steps, future_states.shape[1])
                target_state = future_states[:, :num_pred_steps]

                # If we predicted more steps than available, only evaluate on available ones
                if predicted_state.shape[1] > target_state.shape[1]:
                    predicted_state = predicted_state[:,
                                                      :target_state.shape[1]]
                    if state_uncertainty is not None:
                        state_uncertainty = state_uncertainty[:,
                                                              :target_state.shape[1]]
            else:
                # For single-step, use the last future state
                target_state = future_states[:, -1]

            # Calculate metrics
            if config.multi_step_prediction:
                # MSE across all predicted timesteps
                state_mse = torch.mean(
                    (predicted_state - target_state) ** 2).item()
            else:
                state_mse = torch.mean(
                    (predicted_state - target_state) ** 2).item()

            # Image metrics (if images are predicted)
            if predicted_image is not None:
                # Ensure the target image has the same size
                if target_future_image.shape[-2:] != predicted_image.shape[-2:]:
                    target_future_image = torch.nn.functional.interpolate(
                        target_future_image,
                        size=predicted_image.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Normalize target to match prediction range
                if target_future_image.min() >= 0 and target_future_image.max() <= 1:
                    target_future_image = target_future_image * 2 - 1

                # Calculate SSIM and PSNR
                ssim_val = calculate_ssim(
                    predicted_image[0], target_future_image[0])
                psnr_val = calculate_psnr(
                    predicted_image[0], target_future_image[0])

                metrics['image_ssim'].append(ssim_val)
                metrics['image_psnr'].append(psnr_val)

            metrics['state_mse'].append(state_mse)

            # Visualize predictions
            output_path = output_dir / f"sample_{idx}.png"

            # Current state for visualization - last state in context
            current_state_viz = context_states[0, -1].cpu()

            # Get states and images for visualization
            if config.multi_step_prediction:
                predicted_state_viz = predicted_state[0].cpu()  # [T, D]
                target_state_viz = target_state[0].cpu()  # [T, D]
                state_uncertainty_viz = state_uncertainty[0].cpu(
                ) if state_uncertainty is not None else None
            else:
                predicted_state_viz = predicted_state[0].cpu()  # [D]
                target_state_viz = target_state[0].cpu()  # [D]
                state_uncertainty_viz = state_uncertainty[0].cpu(
                ) if state_uncertainty is not None else None

            visualize_predictions(
                current_image=current_image[0].cpu(),
                predicted_image=predicted_image[0].cpu(
                ) if predicted_image is not None else None,
                target_image=target_future_image[0].cpu(),
                current_state=current_state_viz,
                predicted_state=predicted_state_viz,
                target_state=target_state_viz,
                state_uncertainty=state_uncertainty_viz,
                image_uncertainty=image_uncertainty[0].cpu(
                ) if image_uncertainty is not None else None,
                output_path=output_path
            )

            print(
                f"Processed sample {idx+1}/{args.num_samples}, MSE = {state_mse:.4f}")

    # Print overall metrics
    print("\n--- Evaluation Results ---")
    print(f"Average State MSE: {np.mean(metrics['state_mse']):.4f}")

    if metrics['image_ssim']:
        print(f"Average Image SSIM: {np.mean(metrics['image_ssim']):.4f}")
        print(f"Average Image PSNR: {np.mean(metrics['image_psnr']):.2f} dB")

    # Save metrics to file
    with open(output_dir / "metrics.txt", "w") as f:
        f.write("--- Evaluation Results ---\n")
        f.write(f"Average State MSE: {np.mean(metrics['state_mse']):.4f}\n")

        if metrics['image_ssim']:
            f.write(
                f"Average Image SSIM: {np.mean(metrics['image_ssim']):.4f}\n")
            f.write(
                f"Average Image PSNR: {np.mean(metrics['image_psnr']):.2f} dB\n")

    print(f"Visualizations and metrics saved to {output_dir}")


if __name__ == "__main__":
    main()
