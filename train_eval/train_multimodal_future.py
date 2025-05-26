#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/train_multimodal_future.py

import argparse
import torch
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import safetensors if available
try:
    import safetensors.torch
    has_safetensors = True
except ImportError:
    has_safetensors = False
    print("safetensors not available, falling back to PyTorch format for stats saving.")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictor, MultimodalFuturePredictorConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a MultimodalFuturePredictor for future trajectory prediction with uncertainty")

    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Name of the dataset to use")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension of the transformer")
    parser.add_argument("--num_layers", type=int, default=8,
                        help="Number of transformer layers")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--steps", type=int, default=10000,
                        help="Number of training steps")

    # Future prediction specific arguments
    parser.add_argument("--future_steps", type=int, default=8,
                        help="Number of steps into future to predict")
    parser.add_argument("--use_gpt2_style", action="store_true", default=True,
                        help="Whether to use GPT2-style architecture")
    parser.add_argument("--context_horizon", type=int, default=8,
                        help="Length of trajectory context used for prediction")
    parser.add_argument("--predict_image", action="store_true", default=True,
                        help="Whether to predict future images")
    parser.add_argument("--predict_state", action="store_true", default=True,
                        help="Whether to predict future states")
    parser.add_argument("--multi_step", action="store_true", default=False,
                        help="Whether to predict multiple future steps")
    parser.add_argument("--num_future_steps", type=int, default=3,
                        help="Number of future steps to predict if multi_step is enabled")
    parser.add_argument("--predict_uncertainty", action="store_true", default=False,
                        help="Whether to predict uncertainty in predictions")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--output_dir", type=str, default="outputs/train/multimodal_future",
                        help="Directory to save model and logs")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="How often to log training metrics")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="How often to save model checkpoints")

    return parser.parse_args()


def negative_log_likelihood_loss(pred, target, uncertainty):
    """
    Computes negative log likelihood loss with predicted uncertainty.
    Assumes uncertainty is variance (sigma^2).

    Args:
        pred: Predicted mean values [B, D]
        target: Target values [B, D]
        uncertainty: Predicted variance (sigma^2) [B, D]

    Returns:
        NLL loss
    """
    # Avoid division by zero
    eps = 1e-6
    variance = uncertainty + eps

    # NLL loss
    sq_diff = (pred - target) ** 2
    loss = 0.5 * torch.log(variance) + 0.5 * sq_diff / variance

    return loss.mean()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Dataset and Feature Setup ---
    print(f"Loading dataset metadata for {args.dataset}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Configure dataset based on features
    context_horizon = args.context_horizon
    future_steps = args.future_steps

    # Total steps needed = context_horizon + future_steps
    if args.multi_step and args.num_future_steps > 1:
        total_horizon = context_horizon + \
            max(future_steps, args.num_future_steps)
    else:
        total_horizon = context_horizon + future_steps

    # Get state dimension
    state_dim = features["observation.state"].shape[0]

    print(f"Using state dimension: {state_dim}")
    print(f"Context horizon: {context_horizon}, Future steps: {future_steps}")
    print(f"Total horizon for dataset: {total_horizon}")

    if args.multi_step:
        print(
            f"Multi-step prediction enabled with {args.num_future_steps} future steps")

    if args.predict_uncertainty:
        print("Uncertainty prediction enabled")

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

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # --- Normalization Setup ---
    normalize_inputs = Normalize(
        {"observation.state": features["observation.state"],
         "observation.image": features["observation.image"]},
        {"observation.state": "standard",
         "observation.image": "none"},
        dataset_metadata.stats
    )

    # --- Model Setup ---
    print("Creating MultimodalFuturePredictor model...")
    config = MultimodalFuturePredictorConfig(
        state_dim=state_dim,
        horizon=total_horizon,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=0.1,
        use_layernorm=True,
        mlp_intermediate_factor=4,
        use_gpt2_style=args.use_gpt2_style,
        predict_future=True,
        future_steps=args.future_steps,
        predict_future_image=args.predict_image,
        predict_future_state=args.predict_state,
        multi_step_prediction=args.multi_step,
        num_future_steps=args.num_future_steps,
        predict_uncertainty=args.predict_uncertainty
    )

    # Save config for future reference
    with open(output_dir / "config.json", "w") as f:
        json.dump(config.__dict__, f, indent=2)

    # Initialize model and move to device
    model = MultimodalFuturePredictor(config).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Created MultimodalFuturePredictor with {num_params:,} parameters")

    # Try to optimize with torch.compile if available (PyTorch 2.0+)
    if hasattr(torch, "compile"):
        try:
            print("Compiling model with torch.compile()...")
            model = torch.compile(model)
            print("Model compilation successful - performance should be improved")
        except Exception as e:
            print(f"Model compilation skipped: {e}")
            print("Continuing with standard model execution")

    # --- Optimizer Setup ---
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )

    # Use cosine annealing schedule for better convergence
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps
    )

    # Train model
    model.train()

    print(f"Starting training ({args.steps} steps)...")

    # Define loss functions
    state_loss_fn = torch.nn.MSELoss()
    image_loss_fn = torch.nn.L1Loss()

    # Create metrics dict to track progress
    metrics = {
        'total_losses': [],
        'state_losses': [],
        'image_losses': [],
        'uncertainty_losses': [],
        'lr': []
    }

    pbar = tqdm(range(args.steps), desc="Training Future Predictor")
    step = 0

    # Main training loop
    while step < args.steps:
        for batch in dataloader:
            # Normalize the data
            norm_batch = normalize_inputs(batch)
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # Extract the states and images
            states = norm_batch["observation.state"]
            images = norm_batch["observation.image"]

            B = states.shape[0]

            # Split into context and future
            context_states = states[:, :context_horizon]
            future_states = states[:,
                                   context_horizon:context_horizon+future_steps]

            # Get current and future images
            current_image = images[:, 0]
            target_future_image = images[:, 2]

            # Reset gradients
            optimizer.zero_grad()

            # Make predictions
            if args.multi_step:
                # For multi-step prediction
                predicted_states, predicted_future_image, state_uncertainties, image_uncertainty = model.predict_future_trajectory(
                    context_states, current_image)

                # Target states depend on how many future steps we're predicting
                if args.num_future_steps > future_states.shape[1]:
                    # If we're predicting more steps than available, pad with the last state
                    padding = args.num_future_steps - future_states.shape[1]
                    last_state = future_states[:, -1:].expand(-1, padding, -1)
                    padded_future_states = torch.cat(
                        [future_states, last_state], dim=1)
                    target_future_states = padded_future_states
                else:
                    target_future_states = future_states[:,
                                                         :args.num_future_steps]
            else:
                # For single-step prediction
                predicted_states, predicted_future_image, state_uncertainties, image_uncertainty = model.predict_future_trajectory(
                    context_states, current_image)

                target_future_states = future_states[:, -1]

            # Calculate losses
            total_loss = 0.0
            state_loss = None
            image_loss = None
            uncertainty_loss = None

            if args.predict_state:
                if args.multi_step:
                    if args.predict_uncertainty and state_uncertainties is not None:
                        uncertainty_loss = negative_log_likelihood_loss(
                            predicted_states.reshape(
                                B * args.num_future_steps, -1),
                            target_future_states.reshape(
                                B * args.num_future_steps, -1),
                            state_uncertainties.reshape(
                                B * args.num_future_steps, -1)
                        )
                        total_loss += uncertainty_loss
                        metrics['uncertainty_losses'].append(
                            uncertainty_loss.item())
                    else:
                        state_loss = state_loss_fn(
                            predicted_states, target_future_states)
                        total_loss += state_loss
                        metrics['state_losses'].append(state_loss.item())
                else:
                    if args.predict_uncertainty and state_uncertainties is not None:
                        uncertainty_loss = negative_log_likelihood_loss(
                            predicted_states, target_future_states, state_uncertainties
                        )
                        total_loss += uncertainty_loss
                        metrics['uncertainty_losses'].append(
                            uncertainty_loss.item())
                    else:
                        state_loss = state_loss_fn(
                            predicted_states, target_future_states)
                        total_loss += state_loss
                        metrics['state_losses'].append(state_loss.item())

            if args.predict_image and predicted_future_image is not None:
                # Ensure target image is the same size as predicted
                if target_future_image.shape != predicted_future_image.shape:
                    target_future_image = torch.nn.functional.interpolate(
                        target_future_image,
                        size=predicted_future_image.shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    )

                # Normalize target image to match prediction range if using tanh
                if target_future_image.min() >= 0 and target_future_image.max() <= 1:
                    target_future_image = target_future_image * 2 - 1

                if args.predict_uncertainty and image_uncertainty is not None:
                    pred_img_flat = predicted_future_image.view(B, -1)
                    target_img_flat = target_future_image.view(B, -1)
                    img_uncertainty_flat = image_uncertainty.view(B, -1)

                    img_uncertainty_loss = negative_log_likelihood_loss(
                        pred_img_flat, target_img_flat, img_uncertainty_flat
                    )
                    img_loss_weight = 0.05
                    total_loss += img_uncertainty_loss * img_loss_weight
                    metrics['uncertainty_losses'].append(
                        img_uncertainty_loss.item())
                else:
                    image_loss = image_loss_fn(
                        predicted_future_image, target_future_image)
                    image_loss_weight = 0.1
                    total_loss += image_loss * image_loss_weight
                    metrics['image_losses'].append(image_loss.item())

            # Backprop and update
            total_loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            lr_scheduler.step()

            # Track metrics
            metrics['total_losses'].append(total_loss.item())
            metrics['lr'].append(lr_scheduler.get_last_lr()[0])

            # Log progress
            if step % args.log_freq == 0:
                recent_total_losses = metrics['total_losses'][-args.log_freq:]
                avg_total_loss = sum(recent_total_losses) / \
                    len(recent_total_losses)
                current_lr = lr_scheduler.get_last_lr()[0]

                log_msg = f"Step {step}/{args.steps}: Total Loss={avg_total_loss:.4f}, LR={current_lr:.6f}"

                if args.predict_state and len(metrics['state_losses']) > 0:
                    recent_state_losses = metrics['state_losses'][-min(
                        args.log_freq, len(metrics['state_losses'])):]
                    if recent_state_losses:
                        avg_state_loss = sum(
                            recent_state_losses) / len(recent_state_losses)
                        log_msg += f", State Loss={avg_state_loss:.4f}"

                if args.predict_image and len(metrics['image_losses']) > 0:
                    recent_image_losses = metrics['image_losses'][-min(
                        args.log_freq, len(metrics['image_losses'])):]
                    if recent_image_losses:
                        avg_image_loss = sum(
                            recent_image_losses) / len(recent_image_losses)
                        log_msg += f", Image Loss={avg_image_loss:.4f}"

                if args.predict_uncertainty and len(metrics['uncertainty_losses']) > 0:
                    recent_uncertainty_losses = metrics['uncertainty_losses'][-min(
                        args.log_freq, len(metrics['uncertainty_losses'])):]
                    if recent_uncertainty_losses:
                        avg_uncertainty_loss = sum(
                            recent_uncertainty_losses) / len(recent_uncertainty_losses)
                        log_msg += f", Uncertainty Loss={avg_uncertainty_loss:.4f}"

                print(log_msg)

            # Save checkpoint
            if step % args.save_freq == 0 and step > 0:
                checkpoint_path = output_dir / f"model_step_{step}.pt"
                torch.save({
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': lr_scheduler.state_dict(),
                    'total_loss': total_loss.item(),
                    'config': config.__dict__,
                }, checkpoint_path)
                print(f"Saved checkpoint at step {step} to {checkpoint_path}")

            step += 1
            pbar.update(1)

            if step >= args.steps:
                break

    # --- Save Final Model ---
    final_path = output_dir / "multimodal_future_final.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': config.__dict__,
        'step': step,
    }, final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # Also save just the model weights for easy loading
    torch.save(model.state_dict(), output_dir / "multimodal_future_weights.pt")
    print(
        f"Model weights saved to: {output_dir / 'multimodal_future_weights.pt'}")

    # --- Save Normalization Stats ---
    print("Saving dataset statistics for future normalization...")
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    if has_safetensors:
        safetensors.torch.save_file(
            stats_to_save, str(output_dir / "stats.safetensors"))
        print(f"Stats saved to: {output_dir / 'stats.safetensors'}")
    else:
        torch.save(stats_to_save, output_dir / "stats.pth")
        print(f"Stats saved to: {output_dir / 'stats.pth'}")


if __name__ == "__main__":
    main()
