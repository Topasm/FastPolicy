#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/eval_future_prediction.py

import argparse
import torch
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.critic.modernbert_critic import ModernBertCritic, ModernBertCriticConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a ModernBERT future prediction model")

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the trained model checkpoint")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Name of the dataset to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate and visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for evaluation")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/future_prediction",
                        help="Directory to save evaluation results and visualizations")

    return parser.parse_args()


def calculate_image_metrics(pred_img, target_img):
    """Calculate image quality metrics between predicted and target images."""
    # Convert tensors to numpy arrays if needed
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()

    # Make sure images are in the correct range for metrics
    if pred_img.min() < 0:  # If in [-1, 1] range, convert to [0, 1]
        pred_img = (pred_img + 1) / 2
    if target_img.min() < 0:
        target_img = (target_img + 1) / 2

    # Clip values to valid range
    pred_img = np.clip(pred_img, 0, 1)
    target_img = np.clip(target_img, 0, 1)

    # Move channel dimension to end for skimage compatibility
    if pred_img.shape[0] == 3:  # If [C, H, W]
        pred_img = np.transpose(pred_img, (1, 2, 0))
    if target_img.shape[0] == 3:
        target_img = np.transpose(target_img, (1, 2, 0))

    # Calculate metrics
    try:
        ssim_value = ssim(target_img, pred_img, channel_axis=2, data_range=1.0)
    except Exception as e:
        print(f"Error calculating SSIM: {e}")
        ssim_value = 0.0

    try:
        psnr_value = psnr(target_img, pred_img, data_range=1.0)
    except Exception as e:
        print(f"Error calculating PSNR: {e}")
        psnr_value = 0.0

    return {
        'ssim': ssim_value,
        'psnr': psnr_value
    }


def visualize_predictions(current_img, pred_img, target_img, current_state, pred_state, target_state,
                          metrics, save_path, sample_idx):
    """Visualize the predicted images and states compared to the targets."""
    # Convert tensors to numpy arrays
    if isinstance(current_img, torch.Tensor):
        current_img = current_img.detach().cpu().numpy()
    if isinstance(pred_img, torch.Tensor):
        pred_img = pred_img.detach().cpu().numpy()
    if isinstance(target_img, torch.Tensor):
        target_img = target_img.detach().cpu().numpy()
    if isinstance(current_state, torch.Tensor):
        current_state = current_state.detach().cpu().numpy()
    if isinstance(pred_state, torch.Tensor):
        pred_state = pred_state.detach().cpu().numpy()
    if isinstance(target_state, torch.Tensor):
        target_state = target_state.detach().cpu().numpy()

    # Ensure images are in [0, 1] range for visualization
    if current_img.min() < 0:
        current_img = (current_img + 1) / 2
    if pred_img.min() < 0:
        pred_img = (pred_img + 1) / 2
    if target_img.min() < 0:
        target_img = (target_img + 1) / 2

    # Create figure with two rows
    plt.figure(figsize=(15, 10))

    # Row 1: Images
    plt.subplot(2, 3, 1)
    plt.title("Current Image (t)")
    if current_img.shape[0] == 3:  # Channel-first format
        plt.imshow(np.transpose(current_img, (1, 2, 0)))
    else:
        plt.imshow(current_img)
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Predicted Future (t+8)")
    if pred_img.shape[0] == 3:
        plt.imshow(np.transpose(pred_img, (1, 2, 0)))
    else:
        plt.imshow(pred_img)
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Ground Truth Future (t+8)")
    if target_img.shape[0] == 3:
        plt.imshow(np.transpose(target_img, (1, 2, 0)))
    else:
        plt.imshow(target_img)
    plt.axis('off')

    # Row 2: States
    plt.subplot(2, 1, 2)
    state_dim = min(10, len(current_state))  # Show at most 10 dimensions
    x = np.arange(state_dim)
    width = 0.25

    plt.bar(x - width, current_state[:state_dim], width, label='Current State')
    plt.bar(x, pred_state[:state_dim], width, label='Predicted Future State')
    plt.bar(x + width, target_state[:state_dim],
            width, label='Ground Truth Future State')

    plt.xlabel('State Dimension')
    plt.ylabel('Value')
    plt.title('State Comparison')
    plt.xticks(x, [f"Dim {i}" for i in range(state_dim)])
    plt.legend()

    # Add metrics as text
    mse = np.mean((pred_state - target_state) ** 2)
    plt.figtext(0.5, 0.01, f"Image Metrics: SSIM={metrics['ssim']:.4f}, PSNR={metrics['psnr']:.2f}dB | State MSE={mse:.4f}",
                ha="center", fontsize=12, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})

    # Save the figure
    plt.tight_layout()
    plt.savefig(save_path / f"sample_{sample_idx}.png")
    plt.close()

    return mse


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)

    # Load model checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)

    # Load config from checkpoint or create one if not available
    if 'config' in checkpoint:
        config_dict = checkpoint['config']
        config = ModernBertCriticConfig(**config_dict)
    else:
        print("No config found in checkpoint, using default values")
        config = ModernBertCriticConfig(
            state_dim=10,  # Will be overridden from dataset
            horizon=16,
            predict_future=True,
            future_steps=8,
            predict_future_image=True,
            predict_future_state=True
        )

    # Load dataset metadata
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Update config with correct state dimension from dataset
    state_dim = features["observation.state"].shape[0]
    config.state_dim = state_dim

    # Configure dataset parameters
    context_horizon = 8  # Default context length
    future_steps = config.future_steps
    total_horizon = context_horizon + future_steps

    print(f"Using state dimension: {state_dim}")
    print(f"Context horizon: {context_horizon}, Future steps: {future_steps}")

    # Setup delta timestamps for trajectory features
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in range(total_horizon)],
        # Current, midpoint, future
        "observation.image": [i / dataset_metadata.fps for i in [0, context_horizon, total_horizon-1]]
    }

    # Initialize dataset and dataloader
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda"
    )

    # Setup normalization
    normalize_inputs = Normalize(
        {"observation.state": features["observation.state"],
         "observation.image": features["observation.image"]},
        {"observation.state": "standard",
         "observation.image": "none"},
        dataset_metadata.stats
    )

    # Initialize model
    model = ModernBertCritic(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Prepare for evaluation
    results = {
        'state_mse': [],
        'image_ssim': [],
        'image_psnr': []
    }

    # Evaluate on some samples
    print(f"Evaluating model on {args.num_samples} samples...")
    processed_samples = 0

    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Break if we've processed enough samples
            if processed_samples >= args.num_samples:
                break

            # Normalize data
            norm_batch = normalize_inputs(batch)
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # Extract states and images
            states = norm_batch["observation.state"]  # [B, H, D_state]
            images = norm_batch["observation.image"]  # [B, T_img, C, H, W]

            batch_size = states.shape[0]

            # Split into context and future
            context_states = states[:, :context_horizon]
            future_states = states[:,
                                   context_horizon:context_horizon+future_steps]

            # Get current and future images
            current_image = images[:, 0]
            target_future_image = images[:, 2]

            # Target is the last future state
            target_future_state = future_states[:, -1]

            # Make predictions
            predicted_future_state, predicted_future_image = model.predict_future_trajectory(
                context_states, current_image)

            # Evaluate predictions for each sample in batch
            for i in range(batch_size):
                if processed_samples >= args.num_samples:
                    break

                # Get individual sample
                curr_img = current_image[i]
                pred_img = predicted_future_image[i] if predicted_future_image is not None else None
                tgt_img = target_future_image[i]

                curr_state = context_states[i, -1]  # Last state of context
                pred_state = predicted_future_state[i] if predicted_future_state is not None else None
                tgt_state = target_future_state[i]

                # Calculate metrics
                if pred_img is not None and tgt_img is not None:
                    img_metrics = calculate_image_metrics(pred_img, tgt_img)
                    results['image_ssim'].append(img_metrics['ssim'])
                    results['image_psnr'].append(img_metrics['psnr'])
                else:
                    img_metrics = {'ssim': 0, 'psnr': 0}

                if pred_state is not None and tgt_state is not None:
                    state_mse = visualize_predictions(
                        curr_img, pred_img, tgt_img,
                        curr_state, pred_state, tgt_state,
                        img_metrics, vis_dir, processed_samples
                    )
                    results['state_mse'].append(state_mse)

                processed_samples += 1

    # Calculate and print aggregate metrics
    print("\nEvaluation Results:")
    print(f"Average State MSE: {np.mean(results['state_mse']):.6f}")
    print(
        f"Average Image SSIM: {np.mean(results['image_ssim']):.6f} (higher is better)")
    print(
        f"Average Image PSNR: {np.mean(results['image_psnr']):.2f} dB (higher is better)")

    # Save results to file
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump({
            'state_mse': float(np.mean(results['state_mse'])),
            'image_ssim': float(np.mean(results['image_ssim'])),
            'image_psnr': float(np.mean(results['image_psnr'])),
            'config': config.__dict__,
            'num_samples': processed_samples
        }, f, indent=2)

    print(f"Evaluation complete. Results saved to {output_dir}")
    print(f"Visualizations saved to {vis_dir}")


if __name__ == "__main__":
    main()
