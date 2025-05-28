#!/usr/bin/env python3
"""
Evaluation script for the Bidirectional Autoregressive Transformer.

This script loads a trained model and evaluates its performance on generating
bidirectional trajectories and reconstructing images.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import time

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)
from model.bidirectional_dataset import BidirectionalTrajectoryDataset


def denormalize_image(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor to displayable numpy array."""
    # Convert from [-1, 1] to [0, 1]
    image = (image_tensor + 1.0) / 2.0
    # Clamp to valid range
    image = torch.clamp(image, 0.0, 1.0)
    # Convert to numpy and move channels to last dimension
    image = image.cpu().numpy()
    if image.ndim == 3:  # [C, H, W] -> [H, W, C]
        image = np.transpose(image, (1, 2, 0))
    return image


def visualize_trajectory_generation(
    model: BidirectionalARTransformer,
    batch: dict,
    device: torch.device,
    save_path: Path,
    num_samples: int = 4
):
    """Visualize trajectory generation results."""
    model.eval()

    with torch.no_grad():
        # Move batch to device
        for key in batch:
            batch[key] = batch[key].to(device)

        # Run inference
        predictions = model(
            initial_images=batch['initial_images'][:num_samples],
            initial_states=batch['initial_states'][:num_samples],
            training=False
        )

        # Create visualization
        fig, axes = plt.subplots(num_samples, 4, figsize=(16, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)

        for i in range(num_samples):
            # Initial image
            initial_img = denormalize_image(batch['initial_images'][i])
            axes[i, 0].imshow(initial_img)
            axes[i, 0].set_title('Initial Image i_0')
            axes[i, 0].axis('off')

            # Predicted goal image
            if 'predicted_goal_images' in predictions:
                goal_img = denormalize_image(
                    predictions['predicted_goal_images'][i])
                axes[i, 1].imshow(goal_img)
                axes[i, 1].set_title('Predicted Goal Image i_n')
                axes[i, 1].axis('off')

            # Ground truth goal image
            gt_goal_img = denormalize_image(batch['goal_images'][i])
            axes[i, 2].imshow(gt_goal_img)
            axes[i, 2].set_title('Ground Truth Goal Image')
            axes[i, 2].axis('off')

            # Reconstructed initial image
            if 'predicted_final_images' in predictions:
                recon_img = denormalize_image(
                    predictions['predicted_final_images'][i])
                axes[i, 3].imshow(recon_img)
                axes[i, 3].set_title('Reconstructed Initial Image')
                axes[i, 3].axis('off')

        plt.tight_layout()
        plt.savefig(save_path / 'trajectory_images.png',
                    dpi=150, bbox_inches='tight')
        plt.close()

        # Plot state trajectories
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        for i in range(min(num_samples, 4)):
            row, col = i // 2, i % 2

            if 'predicted_forward_states' in predictions:
                # Forward trajectory
                gt_forward = batch['forward_states'][i].cpu().numpy()
                pred_forward = predictions['predicted_forward_states'][i].cpu(
                ).numpy()

                # Plot first 3 dimensions of state
                for dim in range(min(3, gt_forward.shape[1])):
                    axes[row, col].plot(gt_forward[:, dim],
                                        label=f'GT Forward Dim {dim}',
                                        linestyle='-', alpha=0.7)
                    axes[row, col].plot(pred_forward[:, dim],
                                        label=f'Pred Forward Dim {dim}',
                                        linestyle='--', alpha=0.7)

            if 'predicted_backward_states' in predictions:
                # Backward trajectory
                gt_backward = batch['backward_states'][i].cpu().numpy()
                pred_backward = predictions['predicted_backward_states'][i].cpu(
                ).numpy()

                for dim in range(min(3, gt_backward.shape[1])):
                    x_offset = len(
                        gt_forward) if 'predicted_forward_states' in predictions else 0
                    x_vals = np.arange(x_offset, x_offset + len(gt_backward))
                    axes[row, col].plot(x_vals, gt_backward[:, dim],
                                        label=f'GT Backward Dim {dim}',
                                        linestyle='-', alpha=0.7)
                    axes[row, col].plot(x_vals, pred_backward[:, dim],
                                        label=f'Pred Backward Dim {dim}',
                                        linestyle='--', alpha=0.7)

            axes[row, col].set_title(f'Sample {i+1} State Trajectories')
            axes[row, col].set_xlabel('Time Step')
            axes[row, col].set_ylabel('State Value')
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path / 'trajectory_states.png',
                    dpi=150, bbox_inches='tight')
        plt.close()


def evaluate_model(
    model: BidirectionalARTransformer,
    dataloader: DataLoader,
    device: torch.device,
    num_eval_batches: int = 50
) -> dict:
    """Evaluate model performance on test data."""
    model.eval()

    metrics = {
        'forward_state_mse': 0.0,
        'backward_state_mse': 0.0,
        'goal_image_mse': 0.0,
        'image_reconstruction_mse': 0.0,
        'goal_latent_consistency': 0.0,
        'final_latent_consistency': 0.0,
    }

    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= num_eval_batches:
                break

            # Move batch to device
            for key in batch:
                batch[key] = batch[key].to(device)

            # Get initial image latents
            initial_image_latents = model.image_encoder(
                batch['initial_images'])

            # Run inference
            predictions = model(
                initial_images=batch['initial_images'],
                initial_states=batch['initial_states'],
                training=False
            )

            # Compute metrics
            if 'predicted_forward_states' in predictions:
                forward_mse = F.mse_loss(
                    predictions['predicted_forward_states'],
                    batch['forward_states'][:, 1:]  # Skip initial state
                )
                metrics['forward_state_mse'] += forward_mse.item()

            if 'predicted_backward_states' in predictions:
                backward_mse = F.mse_loss(
                    predictions['predicted_backward_states'],
                    batch['backward_states']
                )
                metrics['backward_state_mse'] += backward_mse.item()

            if 'predicted_goal_images' in predictions:
                goal_image_mse = F.mse_loss(
                    predictions['predicted_goal_images'],
                    batch['goal_images']
                )
                metrics['goal_image_mse'] += goal_image_mse.item()

            if 'predicted_final_images' in predictions:
                recon_mse = F.mse_loss(
                    predictions['predicted_final_images'],
                    batch['initial_images']
                )
                metrics['image_reconstruction_mse'] += recon_mse.item()

            if 'predicted_goal_latents' in predictions:
                goal_image_latents = model.image_encoder(batch['goal_images'])
                goal_latent_mse = F.mse_loss(
                    predictions['predicted_goal_latents'],
                    goal_image_latents
                )
                metrics['goal_latent_consistency'] += goal_latent_mse.item()

            if 'predicted_final_latents' in predictions:
                final_latent_mse = F.mse_loss(
                    predictions['predicted_final_latents'],
                    initial_image_latents
                )
                metrics['final_latent_consistency'] += final_latent_mse.item()

            num_batches += 1

    # Average metrics
    for key in metrics:
        metrics[key] /= num_batches

    return metrics


def main():
    """Main evaluation function."""
    # Configuration
    model_path = Path("outputs/train/bidirectional_transformer/best_model.pt")
    output_dir = Path("outputs/eval/bidirectional_transformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Bidirectional Autoregressive Transformer Evaluation ===")
    print(f"Device: {device}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")

    # Load model
    if not model_path.exists():
        print(f"Model file not found: {model_path}")
        print("Please train the model first using train_bidirectional_transformer.py")
        return

    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']

    # Create model config
    config = BidirectionalARTransformerConfig(**config_dict)

    # Initialize and load model
    model = BidirectionalARTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Best training loss: {checkpoint.get('best_total_loss', 'N/A')}")

    # --- Dataset Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    # Normalization
    normalize_inputs = Normalize(input_features, {}, dataset_metadata.stats)

    # Create dataset (use subset for evaluation)
    lerobot_dataset = LeRobotDataset(dataset_repo_id, delta_timestamps=None)
    dataset = BidirectionalTrajectoryDataset(
        lerobot_dataset=lerobot_dataset,
        normalizer=normalize_inputs,
        forward_steps=config.forward_steps,
        backward_steps=config.backward_steps,
        min_episode_length=50,
        image_key="observation.image",
        state_key="observation.state"
    )

    # Create data loader
    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=2,
        collate_fn=BidirectionalTrajectoryDataset.collate_fn
    )

    print(f"Evaluation dataset size: {len(dataset)}")

    # --- Run Evaluation ---
    print("Running evaluation...")
    start_time = time.time()

    # Quantitative evaluation
    metrics = evaluate_model(model, dataloader, device, num_eval_batches=100)

    print(f"Evaluation completed in {time.time() - start_time:.2f}s")
    print("\n=== Evaluation Metrics ===")
    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value:.6f}")

    # Save metrics
    torch.save(metrics, output_dir / "evaluation_metrics.pt")

    # --- Qualitative Evaluation ---
    print("\nGenerating visualization...")

    # Get a batch for visualization
    eval_batch = next(iter(dataloader))

    # Generate visualizations
    visualize_trajectory_generation(
        model, eval_batch, device, output_dir, num_samples=4
    )

    print(f"Visualizations saved to: {output_dir}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
