#!/usr/bin/env python3
"""
Test script to verify temporal functionality in the Bidirectional Autoregressive Transformer.
"""

import torch
from pathlib import Path
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features

from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig,
)
from model.predictor.bidirectional_dataset import BidirectionalTrajectoryDataset


def test_temporal_functionality():
    """Test temporal encoding with multiple observation steps."""
    print("=== Testing Temporal Functionality ===")

    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Test different n_obs_steps values
    n_obs_steps_values = [1, 2, 3, 4]

    for n_obs_steps in n_obs_steps_values:
        print(f"\n--- Testing with n_obs_steps = {n_obs_steps} ---")

        # Dataset setup
        dataset_repo_id = "lerobot/pusht"
        dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
        features = dataset_to_policy_features(dataset_metadata.features)
        state_dim = features["observation.state"].shape[-1]

        print(f"State dimension: {state_dim}")

        # Create dataset with temporal encoding
        lerobot_dataset = LeRobotDataset(
            dataset_repo_id, delta_timestamps=None)
        dataset = BidirectionalTrajectoryDataset(
            lerobot_dataset=lerobot_dataset,
            forward_steps=8,  # Smaller for testing
            backward_steps=8,
            min_episode_length=16,
            image_key="observation.image",
            state_key="observation.state",
            n_obs_steps=n_obs_steps
        )

        print(f"Dataset size: {len(dataset)}")

        # Test dataset sampling
        try:
            sample = dataset[0]
            print("Dataset sampling successful!")

            # Check tensor shapes
            print(f"Initial images shape: {sample['initial_images'].shape}")
            print(f"Initial states shape: {sample['initial_states'].shape}")
            print(f"Expected shapes:")
            if n_obs_steps == 1:
                print(f"  Images: [3, 96, 96]")
                print(f"  States: [{state_dim}]")
            else:
                print(f"  Images: [{n_obs_steps}, 3, 96, 96]")
                print(f"  States: [{n_obs_steps}, {state_dim}]")

        except Exception as e:
            print(f"Dataset sampling failed: {e}")
            continue

        # Model configuration
        input_features = {
            "observation.state": features["observation.state"],
            "observation.image": features["observation.image"],
        }
        output_features = {
            "predicted_forward_states": features["observation.state"],
            "predicted_goal_images": features["observation.image"],
            "predicted_backward_states": features["observation.state"],
            "predicted_goal_latents": features["observation.image"],
        }

        config = BidirectionalARTransformerConfig(
            state_dim=state_dim,
            hidden_dim=128,  # Smaller for testing
            num_layers=2,    # Smaller for testing
            num_heads=4,     # Smaller for testing
            dropout=0.1,
            max_position_value=32,
            image_channels=3,
            image_size=96,
            image_latent_dim=64,  # Smaller for testing
            forward_steps=8,
            backward_steps=8,
            n_obs_steps=n_obs_steps,
            input_features=input_features,
            output_features=output_features,
            use_diffusion_encoder=True,
            vision_backbone="resnet18",
            pretrained_backbone_weights="IMAGENET1K_V1",
            spatial_softmax_num_keypoints=16,  # Smaller for testing
            use_group_norm=False,
            crop_shape=None,
            crop_is_random=False,
        )

        # Create model
        try:
            model = BidirectionalARTransformer(
                config=config,
                state_key="observation.state",
                image_key="observation.image"
            )
            model.to(device)
            model.eval()

            total_params = sum(p.numel() for p in model.parameters())
            print(
                f"Model created successfully! Total parameters: {total_params:,}")

        except Exception as e:
            print(f"Model creation failed: {e}")
            continue

        # Test model forward pass
        try:
            # Create a small batch
            batch_size = 2
            batch = {}

            for key, value in sample.items():
                if isinstance(value, torch.Tensor):
                    # Repeat sample to create batch
                    batch[key] = value.unsqueeze(0).repeat(
                        batch_size, *([1] * value.dim()))
                else:
                    batch[key] = [value] * batch_size

            # Move to device
            batch_device = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch_device[key] = value.to(device)
                    # Normalize images to [-1, 1]
                    if 'images' in key:
                        batch_device[key] = batch_device[key] * 2.0 - 1.0
                else:
                    batch_device[key] = value

            # Forward pass in inference mode
            with torch.no_grad():
                predictions = model(
                    initial_images=batch_device['initial_images'],
                    initial_states=batch_device['initial_states'],
                    training=False
                )

            print("Model forward pass successful!")
            print("Prediction keys:", list(predictions.keys()))

            # Check output shapes
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor):
                    print(f"  {key}: {value.shape}")

        except Exception as e:
            print(f"Model forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Test training mode
        try:
            model.train()
            with torch.no_grad():
                predictions = model(
                    initial_images=batch_device['initial_images'],
                    initial_states=batch_device['initial_states'],
                    forward_states=batch_device['forward_states'],
                    goal_images=batch_device['goal_images'],
                    backward_states=batch_device['backward_states'],
                    training=True
                )

            print("Model training forward pass successful!")

        except Exception as e:
            print(f"Model training forward pass failed: {e}")
            import traceback
            traceback.print_exc()
            continue

        print(f"✅ All tests passed for n_obs_steps = {n_obs_steps}")

    print("\n=== Temporal Functionality Test Complete ===")


if __name__ == "__main__":
    test_temporal_functionality()
