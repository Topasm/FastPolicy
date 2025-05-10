#!/usr/bin/env python3
"""
Training script for the enhanced inverse dynamics model.
This script supports training both the sequential (GRU) based model and the 
new enhanced model with features from the reference code.
"""

import torch
import torch.nn.functional as F
from pathlib import Path
import argparse
import os
import safetensors.torch
import matplotlib.pyplot as plt

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

# Import the inverse dynamics models
from model.invdynamics.invdyn import MlpInvDynamic, SeqInvDynamic
from model.invdynamics.enhanced_invdyn import EnhancedInvDynamic
from model.diffusion.configuration_mymodel import DiffusionConfig


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Train enhanced inverse dynamics model")
    parser.add_argument("--output_dir", type=str, default="outputs/train/enhanced_invdyn",
                        help="Directory to save model checkpoints and logs")
    parser.add_argument("--model_type", type=str, choices=["enhanced", "seq", "mlp"], default="enhanced",
                        help="Type of inverse dynamics model to train")
    parser.add_argument("--batch_size", type=int,
                        default=64, help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--steps", type=int, default=2000,
                        help="Number of training steps")
    parser.add_argument("--log_freq", type=int, default=10,
                        help="Logging frequency (steps)")
    parser.add_argument("--save_freq", type=int, default=100,
                        help="Checkpoint saving frequency (steps)")
    parser.add_argument("--seq_length", type=int, default=3,
                        help="Sequence length for sequential models")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda, cpu)")
    parser.add_argument("--dataset", type=str,
                        default="lerobot/pusht", help="Dataset to use for training")
    parser.add_argument("--hidden_dim", type=int, default=512,
                        help="Hidden dimension for the model")
    parser.add_argument("--probabilistic", action="store_true",
                        help="Use probabilistic output for enhanced model")
    parser.add_argument("--temperature", type=float, default=0.1,
                        help="Temperature parameter for probabilistic model")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")

    args = parser.parse_args()

    # Create output directory
    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)

    # Save arguments
    with open(output_directory / "args.txt", "w") as f:
        f.write("\n".join([f"{k}: {v}" for k, v in vars(args).items()]))

    # Device setup
    device = torch.device(args.device)

    # --- Dataset and Config Setup ---
    dataset_repo_id = args.dataset
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Use DiffusionConfig to get parameters easily
    cfg = DiffusionConfig(
        input_features={"observation.state": features["observation.state"]},
        output_features={"action": features["action"]}
    )

    # --- Model Configuration ---
    print(
        f"Training with model_type: {args.model_type}, seq_length: {args.seq_length}")

    if args.model_type == "enhanced":
        # Create enhanced model
        invdyn_model = EnhancedInvDynamic(
            state_dim=features["observation.state"].shape[0],
            action_dim=features["action"].shape[0],
            # 4-layer network matching original
            hidden_dims=[args.hidden_dim] * 4,
            use_state_encoding=True,  # Encode current and next states separately
            is_probabilistic=args.probabilistic,
            temperature=args.temperature,
            train_temp=1.0,
            eval_temp=0.1,  # Small but non-zero for slight exploration
            dropout=0.1,
            use_layernorm=True,
            out_activation=torch.nn.Tanh(),
        )
    elif args.model_type == "seq":
        # Use GRU-based sequential model
        invdyn_model = SeqInvDynamic(
            state_dim=features["observation.state"].shape[0],
            action_dim=features["action"].shape[0],
            hidden_dim=args.hidden_dim,
            n_layers=2,  # Increased capacity
            dropout=0.1,
            out_activation=torch.nn.Tanh()
        )
    else:  # "mlp"
        # Original MLP model
        invdyn_model = MlpInvDynamic(
            o_dim=features["observation.state"].shape[0] * 2,  # s_t, s_{t+1}
            a_dim=features["action"].shape[0],
            hidden_dim=args.hidden_dim,
            dropout=0.1,
            use_layernorm=True,
            out_activation=torch.nn.Tanh(),
        )

    # If resuming from checkpoint
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        state_dict = torch.load(args.resume, map_location="cpu")

        # Debug model loading
        debug_model_loading = True
        if debug_model_loading:
            print("\n--- Model Structure Debug ---")
            print("Model Parameter Keys:")
            for name, _ in invdyn_model.named_parameters():
                print(f"  {name}")
            print("\nState Dict Keys:")
            for key in state_dict.keys():
                print(f"  {key}")
            print("---------------------------\n")

        # Try loading state dict
        try:
            invdyn_model.load_state_dict(state_dict, strict=False)
            print("Successfully loaded checkpoint (partial matching allowed)")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")

    # Move model to device
    invdyn_model.train()
    invdyn_model.to(device)

    # --- Normalization ---
    # Move stats to the target device to avoid device mismatches
    stats_for_device = {}
    for k, v in dataset_metadata.stats.items():
        if isinstance(v, torch.Tensor):
            stats_for_device[k] = v.to(device)
        else:
            stats_for_device[k] = v

    normalize_state = Normalize(
        {"observation.state": features["observation.state"]},
        cfg.normalization_mapping, stats_for_device).to(device)
    normalize_action = Normalize(
        {"action": features["action"]},
        cfg.normalization_mapping, stats_for_device).to(device)

    # --- Dataset ---
    # Set up appropriate delta timestamps based on model type
    if args.model_type == "enhanced":
        # For enhanced model, we just need current and next state for each action
        custom_state_indices = [0, 1]  # s_t, s_{t+1}
    else:
        # For sequential models, we need sequence history
        custom_state_indices = []
        for i in range(-args.seq_length+1, len(cfg.action_delta_indices)+1):
            custom_state_indices.append(i)

    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in custom_state_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    print(f"State indices: {custom_state_indices}")
    print(f"Action indices: {cfg.action_delta_indices}")

    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(invdyn_model.parameters(), lr=args.lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=args.batch_size,
        shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Training loop ---
    step = 0
    done = False
    epoch = 0
    losses = []

    print(f"Starting training for {args.steps} steps...")

    while not done:
        epoch += 1
        for batch in dataloader:
            # Move data to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # Get normalized states and actions
            # Make sure the tensors are on the correct device before normalization
            norm_batch = {
                "observation.state": normalize_state({"observation.state": batch["observation.state"].to(device)})["observation.state"],
                "action": normalize_action({"action": batch["action"].to(device)})["action"]
            }

            # Compute loss based on model type
            if args.model_type == "enhanced":
                # For enhanced model with current and next state
                total_loss = 0.0
                states = norm_batch["observation.state"]
                actions = norm_batch["action"]

                # Process each action in the sequence
                for i in range(actions.shape[1]):
                    # Check if we have the next state
                    if i + 1 < states.shape[1]:
                        curr_state = states[:, i, :]
                        next_state = states[:, i+1, :]
                        target_action = actions[:, i, :]

                        # Compute loss using the enhanced model's loss method
                        action_loss, _ = invdyn_model.loss(
                            curr_state, next_state, target_action)
                        total_loss += action_loss

                # Average loss over sequence length
                loss = total_loss / actions.shape[1]

            elif args.model_type == "seq":
                # For sequential GRU model
                states = norm_batch["observation.state"]
                actions = norm_batch["action"]
                total_loss = 0.0

                # For each action, create a sequence of states and predict
                for i in range(actions.shape[1]):
                    if i + args.seq_length <= states.shape[1]:
                        state_seq = states[:, i:i+args.seq_length, :]
                        target_action = actions[:, i, :]

                        # Run the sequential model
                        pred_action = invdyn_model(state_seq)
                        # Get the last prediction
                        pred_action = pred_action[:, -1, :]

                        # Compute MSE loss
                        action_loss = F.mse_loss(pred_action, target_action)
                        total_loss += action_loss

                # Average loss over sequence length
                loss = total_loss / actions.shape[1]

            else:  # MLP
                # For MLP model, use pairs of consecutive states
                states = norm_batch["observation.state"]
                actions = norm_batch["action"]
                total_loss = 0.0

                for i in range(actions.shape[1]):
                    if i + 1 < states.shape[1]:
                        # Concatenate current and next state
                        state_pair = torch.cat(
                            [states[:, i, :], states[:, i+1, :]], dim=1)
                        target_action = actions[:, i, :]

                        # Predict action
                        pred_action = invdyn_model(state_pair)

                        # Compute MSE loss
                        action_loss = F.mse_loss(pred_action, target_action)
                        total_loss += action_loss

                # Average loss over sequence length
                loss = total_loss / actions.shape[1]

            # Optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Store loss
            losses.append(loss.item())

            # Logging
            if step % args.log_freq == 0:
                avg_loss = sum(losses[-args.log_freq:]) / \
                    min(args.log_freq, len(losses))
                print(
                    f"Step: {step}/{args.steps} | Loss: {avg_loss:.4f} | Epoch: {epoch}")

            # Save checkpoint
            if step % args.save_freq == 0 and step > 0:
                ckpt_path = output_directory / \
                    f"invdyn_{args.model_type}_step_{step}.pth"
                torch.save(invdyn_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

                # Plot and save loss curve
                plt.figure(figsize=(10, 5))
                plt.plot(losses)
                plt.title("Training Loss")
                plt.xlabel("Step")
                plt.ylabel("Loss")
                plt.grid(True)
                plt.savefig(output_directory / f"loss_curve_{step}.png")
                plt.close()

            # Check if done
            step += 1
            if step >= args.steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / f"invdyn_{args.model_type}_final.pth"
    torch.save(invdyn_model.state_dict(), final_path)
    print(f"Training finished. Final model saved to: {final_path}")

    # --- Save Config and Stats ---
    cfg.save_pretrained(output_directory)
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items() if isinstance(v, torch.Tensor)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")

    # Plot final loss curve
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(output_directory / "loss_curve_final.png")
    plt.close()


if __name__ == "__main__":
    main()
