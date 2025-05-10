"""
This file contains integration code for using the enhanced inverse dynamics model.
It provides conversion and utility functions to integrate the enhanced model 
with the existing FastPolicy pipeline.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
from pathlib import Path
import sys
import logging

# Import the enhanced model
from model.invdynamics.enhanced_invdyn import EnhancedInvDynamic


def convert_to_enhanced_invdyn(
    model_path: str,
    state_dim: int,
    action_dim: int,
    hidden_dims: List[int] = [512, 512, 512, 512],
    use_probabilistic: bool = False,
    use_state_encoding: bool = True,
    temperature: float = 0.1,
    output_path: Optional[str] = None
) -> EnhancedInvDynamic:
    """
    Create an enhanced inverse dynamics model and optionally transfer weights from an existing model.

    Args:
        model_path: Path to the existing model checkpoint (MlpInvDynamic or SeqInvDynamic)
        state_dim: Dimension of the state
        action_dim: Dimension of the action
        hidden_dims: List of hidden dimensions for the enhanced model
        use_probabilistic: Whether to use probabilistic output
        use_state_encoding: Whether to encode current and next states separately
        temperature: Temperature for probabilistic sampling
        output_path: Optional path to save the converted model

    Returns:
        Enhanced inverse dynamics model
    """
    # Create enhanced model
    enhanced_model = EnhancedInvDynamic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        use_state_encoding=use_state_encoding,
        is_probabilistic=use_probabilistic,
        temperature=temperature,
        train_temp=1.0,
        eval_temp=0.1,  # Small but non-zero for slight exploration
        dropout=0.1,
        use_layernorm=True,
        out_activation=nn.Tanh(),
    )

    # If model path is provided, try to load and transfer compatible weights
    if model_path and Path(model_path).exists():
        try:
            # Load the source model state dict
            source_state_dict = torch.load(model_path, map_location='cpu')
            print(f"Loaded source model from {model_path}")

            # Debug model structure
            print("\n--- Model Structure Debug ---")
            print("Enhanced model parameter keys:")
            for name, _ in enhanced_model.named_parameters():
                print(f"  {name}")

            print("\nSource model state dict keys:")
            for key in source_state_dict.keys():
                print(f"  {key}")
            print("---------------------------\n")

            # Transfer weights for compatible layers
            # This is just a placeholder - you'll need to customize this based on
            # the actual structure of your models

            # If transferring is not feasible or beneficial, we just initialize from scratch
            print("NOTE: Only using the enhanced model's initialization")

            # Example (uncomment and modify as needed):
            # matched_keys = []
            # for name, param in enhanced_model.named_parameters():
            #     source_key = None
            #     # Map enhanced model keys to source model keys
            #     if "mean_net" in name and "net.16" in source_state_dict:
            #         source_key = name.replace("mean_net", "net.16")
            #     # Add more mapping rules as needed

            #     if source_key and source_key in source_state_dict:
            #         if param.shape == source_state_dict[source_key].shape:
            #             param.data.copy_(source_state_dict[source_key])
            #             matched_keys.append((name, source_key))

            # print(f"Transferred {len(matched_keys)} layers from source model")

        except Exception as e:
            print(f"Error loading or transferring weights: {e}")
            print("Using original initialization for enhanced model")

    # Save the model if output path is provided
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(enhanced_model.state_dict(), output_path)
        print(f"Saved enhanced model to {output_path}")

    return enhanced_model


def generate_actions_with_enhanced_invdyn(
    diffusion_model,
    inv_dyn_model,
    norm_batch,
    norm_current_state,
    num_inference_samples=1,
    temperature=None,
    use_sampling=False
):
    """
    Generate actions using diffusion model for state prediction and enhanced inverse dynamics for action prediction.

    Args:
        diffusion_model: MyDiffusionModel instance
        inv_dyn_model: EnhancedInvDynamic model instance
        norm_batch: Normalized batch with observation history
        norm_current_state: Current normalized state
        num_inference_samples: Number of future trajectories to sample
        temperature: Optional temperature override for sampling
        use_sampling: Whether to use stochastic sampling (if probabilistic)

    Returns:
        Predicted actions
    """
    device = next(inv_dyn_model.parameters()).device

    # Generate predictions with the diffusion model
    future_predictions = diffusion_model.generate(
        norm_batch, num_inference_samples=num_inference_samples
    )

    # Extract the first predicted future state
    # Shape: [batch, num_samples, 1, state_dim]
    future_state = future_predictions["observation.state"][:, :, 0, :]

    # If we have multiple samples, we need to handle them
    if num_inference_samples > 1:
        # Reshape to [batch*samples, state_dim]
        batch_size = norm_current_state.shape[0]
        future_state = future_state.reshape(-1, future_state.shape[-1])
        norm_current_state = norm_current_state.repeat_interleave(
            num_inference_samples, dim=0)

    # Use the enhanced model to predict actions
    if use_sampling and hasattr(inv_dyn_model, 'is_probabilistic') and inv_dyn_model.is_probabilistic:
        # Use stochastic sampling if available
        actions = inv_dyn_model.sample(
            norm_current_state, future_state, temperature=temperature)
    else:
        # Use deterministic prediction
        actions = inv_dyn_model.predict(
            torch.cat([norm_current_state, future_state], dim=-1))

    # Reshape actions back to [batch, num_samples, action_dim] if needed
    if num_inference_samples > 1:
        actions = actions.reshape(batch_size, num_inference_samples, -1)

    return actions


def train_enhanced_invdyn(
    train_dataset,
    model,
    batch_size=64,
    lr=1e-4,
    training_steps=5000,
    log_freq=10,
    save_freq=100,
    output_directory="outputs/train/enhanced_invdyn",
    normalizer_state=None,
    normalizer_action=None
):
    """
    Train the enhanced inverse dynamics model.

    Args:
        train_dataset: Training dataset
        model: EnhancedInvDynamic model instance
        batch_size: Batch size for training
        lr: Learning rate
        training_steps: Number of training steps
        log_freq: Frequency for printing logs
        save_freq: Frequency for saving checkpoints
        output_directory: Directory to save outputs
        normalizer_state: Optional Normalize instance for states
        normalizer_action: Optional Normalize instance for actions
    """
    from pathlib import Path
    import torch.utils.data
    import time

    # Create output directory
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Setup device
    device = next(model.parameters()).device
    model.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=4,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True
    )

    # Training loop
    step = 0
    done = False
    epoch = 0
    losses = []

    print(f"Starting training for {training_steps} steps...")
    start_time = time.time()

    while not done:
        epoch += 1
        for batch in dataloader:
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Get states and actions
            states = batch["observation.state"]  # Shape: [B, S, D]
            actions = batch["action"]  # Shape: [B, A, D_a]

            # Apply normalization if provided
            if normalizer_state is not None:
                states = normalizer_state({"observation.state": states})[
                    "observation.state"]
            if normalizer_action is not None:
                actions = normalizer_action({"action": actions})["action"]

            # Compute loss for each action using consecutive states
            total_loss = 0.0
            batch_info = {}

            for i in range(actions.shape[1]):
                # Get current state and next state
                if i + 1 < states.shape[1]:
                    curr_state = states[:, i, :]
                    next_state = states[:, i+1, :]
                    target_action = actions[:, i, :]

                    # Compute loss
                    loss, info = model.loss(
                        curr_state, next_state, target_action)
                    total_loss += loss

                    # Update batch info
                    for k, v in info.items():
                        if k not in batch_info:
                            batch_info[k] = []
                        batch_info[k].append(v)

            # Average losses
            for k, v in batch_info.items():
                batch_info[k] = sum(v) / len(v)

            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Store loss value
            losses.append(total_loss.item())

            # Logging
            if step % log_freq == 0:
                avg_loss = sum(losses[-log_freq:]) / min(log_freq, len(losses))
                elapsed = time.time() - start_time
                print(f"Step: {step}/{training_steps} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"Time: {elapsed:.2f}s | "
                      f"Epoch: {epoch}")

                # Additional info if available
                if batch_info:
                    info_str = " | ".join(
                        [f"{k}: {v:.4f}" for k, v in batch_info.items()])
                    print(f"  Details: {info_str}")

            # Save checkpoint
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / \
                    f"enhanced_invdyn_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            # Check if done
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final model
    final_path = output_directory / "enhanced_invdyn_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")

    # Save loss curve
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output_directory / "loss_curve.png")

    return model, losses
