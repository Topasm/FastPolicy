#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/train_eval/eval_continuous_transformer.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import safetensors.torch

from model.predictor.continuous_ar_transformer import ContinuousARTransformer, ContinuousARTransformerConfig


def main():
    # Configuration
    model_path = Path(
        "outputs/train/continuous_transformer/continuous_transformer_final.pt")
    stats_path = Path("outputs/train/continuous_transformer/stats.safetensors")
    output_dir = Path("outputs/eval/continuous_transformer")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config_dict = checkpoint['config']

    # Create model config and initialize model
    config = ContinuousARTransformerConfig(
        state_dim=config_dict['state_dim'],
        hidden_dim=config_dict['hidden_dim'],
        num_layers=config_dict['num_layers'],
        num_heads=config_dict['num_heads'],
        dropout=config_dict['dropout'],
        max_position_value=config_dict['max_position_value'],
        bidirectional=config_dict['bidirectional'],
        image_channels=config_dict['image_channels'],
        image_size=config_dict['image_size']
    )

    model = ContinuousARTransformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # Load stats for denormalization
    stats = safetensors.torch.load_file(stats_path)
    state_mean = stats['state_mean'].numpy()
    state_std = stats['state_std'].numpy()

    print(f"Model loaded successfully from {model_path}")

    # Generate some test trajectories
    num_samples = 5
    num_points = 30  # Generate more points for smoother trajectories

    # Test forward generation, with goal conditioning and image decoding
    print("\nGenerating forward trajectories with goal conditioning...")
    for i in range(num_samples):
        # Create random start and goal states
        start_state = torch.randn(1, config.state_dim, device=device)
        goal_state = torch.randn(1, config.state_dim, device=device)

        # Generate trajectory with goal image
        with torch.no_grad():
            trajectory, goal_image = model.generate(
                start_z=start_state,
                goal_z=goal_state,
                direction=0,  # forward
                num_steps=num_points,
                decode_goal_image=True
            )

        # Denormalize
        trajectory_np = trajectory.cpu().numpy()
        denorm_trajectory = trajectory_np * state_std + state_mean

        # Generate evenly spaced position values for plotting
        positions = np.linspace(0, config.max_position_value, num_points)

        # Plot trajectory
        plt.figure(figsize=(12, 6))
        for dim in range(min(3, config.state_dim)):  # Plot first 3 dimensions
            plt.plot(positions, denorm_trajectory[0, :, dim],
                     label=f"Dim {dim}", marker='o')

        plt.xlabel("Relative Position")
        plt.ylabel("State Value")
        plt.title(f"Forward Generation Sample {i+1} (with Goal Conditioning)")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"forward_goal_sample_{i+1}.png")
        plt.close()

        # Save the goal image
        if goal_image is not None:
            # Normalize to [0, 1] range for visualization
            img = goal_image[0].cpu().permute(1, 2, 0)  # [H, W, C]
            img = (img + 1) / 2  # Convert from [-1, 1] to [0, 1]
            img = img.numpy()

            # Save the decoded image
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Decoded Goal Image for Sample {i+1}")
            plt.savefig(output_dir / f"goal_image_sample_{i+1}.png")
            plt.close()

    # Test backward generation, with goal conditioning
    print("\nGenerating backward trajectories with goal conditioning...")
    for i in range(num_samples):
        # Create random start and goal states
        goal_state = torch.randn(1, config.state_dim, device=device)
        start_state = torch.randn(1, config.state_dim, device=device)

        # Generate trajectory
        with torch.no_grad():
            trajectory, _ = model.generate(
                start_z=goal_state,  # In backward mode, we start from goal
                goal_z=start_state,  # In backward mode, goal is the start
                direction=1,  # backward
                num_steps=num_points,
                decode_goal_image=True
            )

        # Denormalize
        trajectory_np = trajectory.cpu().numpy()
        denorm_trajectory = trajectory_np * state_std + state_mean

        # Generate evenly spaced position values for plotting
        positions = np.linspace(0, config.max_position_value, num_points)

        # Plot trajectory
        plt.figure(figsize=(12, 6))
        for dim in range(min(3, config.state_dim)):  # Plot first 3 dimensions
            plt.plot(positions, denorm_trajectory[0, :, dim],
                     label=f"Dim {dim}", marker='o')

        plt.xlabel("Relative Position")
        plt.ylabel("State Value")
        plt.title(f"Backward Generation Sample {i+1} (with Goal Conditioning)")
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / f"backward_goal_sample_{i+1}.png")
        plt.close()

    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
