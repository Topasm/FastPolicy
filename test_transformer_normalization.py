#!/usr/bin/env python3
"""
A simple script to test the normalization functionality in the BidirectionalARTransformer.
"""

from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)
from lerobot.common.datasets.utils import PolicyFeature
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.common.policies.normalize import Normalize, Unnormalize
from pathlib import Path
import torch
import sys
import os
print("Python Path:", sys.path)

# Ensure the current directory is in the path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)
    print(f"Added {current_dir} to Python Path")


# Add debug imports
print("Importing lerobot modules...")
print("Imported normalize")
print("Imported feature type")
print("Imported policy feature")


def test_transformer_normalization():
    # Create a simple dataset with well-defined values
    state_dim = 2
    batch_size = 1

    # Define features
    state_feature = PolicyFeature(
        type=FeatureType.STATE, shape=(state_dim,))
    image_feature = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, 96, 96))

    # Define features dictionary
    input_features = {
        "observation.state": state_feature,
        "observation.image": image_feature
    }
    output_features = {
        "observation.state": state_feature,
        "observation.image": image_feature
    }

    # Create synthetic stats for normalization
    # state will be normalized with mean 1 and std 0.1
    stats = {
        "observation.state": {
            "mean": torch.tensor([1.0, 1.0]),
            "std": torch.tensor([0.1, 0.1])
        }
    }

    # Create normalization mapping
    normalization_mapping = {
        FeatureType.STATE: NormalizationMode.MEAN_STD
    }

    # Create normalizer and unnormalizer
    normalizer = Normalize(input_features, normalization_mapping, stats)
    unnormalizer = Unnormalize(output_features, normalization_mapping, stats)

    # Create model config
    config = BidirectionalARTransformerConfig(
        state_dim=state_dim,
        hidden_dim=16,  # Small model for testing
        num_layers=1,
        num_heads=1,
        dropout=0.0,
        image_channels=3,
        image_size=96,
        image_latent_dim=8,
        forward_steps=3,
        backward_steps=3,
        input_features=input_features,
        output_features=output_features
    )

    # Create model with normalization
    model = BidirectionalARTransformer(
        config=config,
        normalizer=normalizer,
        unnormalizer=unnormalizer,
        state_key="observation.state",
        image_key="observation.image"
    )

    # Set model to eval mode
    model.eval()

    # Create test input: state [2.0, 2.0] should be normalized to [10.0, 10.0]
    # since (2.0-1.0)/0.1 = 10.0
    initial_state = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    initial_image = torch.zeros((batch_size, 3, 96, 96), dtype=torch.float32)

    # Print the normalization test
    test_batch = {"observation.state": initial_state}
    normalized_batch = normalizer(test_batch)
    normalized_state = normalized_batch["observation.state"]

    print(f"Original state: {initial_state}")
    print(f"Normalized state: {normalized_state}")
    print(f"Expected: tensor([[10.0, 10.0]])")

    # Test the forward pass
    with torch.no_grad():
        outputs = model(
            initial_images=initial_image,
            initial_states=initial_state,
            training=False
        )

    # Print shapes of outputs
    print("Model output keys:", list(outputs.keys()))
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"{key} shape: {value.shape}")

    # Create a simple test to verify the normalization is used in forward pass
    print("\nVerifying normalization in forward pass...")

    # Instead of comparing models, let's verify the internal normalization process
    # by examining the input to the underlying transformer

    # Create a simple subclass to track the normalized inputs
    class TrackingTransformer(BidirectionalARTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.normalized_state_seen = None

        def _forward_inference(self, initial_image_latents, initial_states, device):
            self.normalized_state_seen = initial_states.clone()
            return super()._forward_inference(initial_image_latents, initial_states, device)

    # Create a tracking model with normalization
    tracking_model = TrackingTransformer(
        config=config,
        normalizer=normalizer,
        unnormalizer=unnormalizer,
        state_key="observation.state",
        image_key="observation.image"
    )
    tracking_model.eval()

    # Forward pass with tracking model
    with torch.no_grad():
        # Normal forward pass that should internally normalize
        outputs_with_norm = tracking_model(
            initial_images=initial_image,
            initial_states=initial_state,
            training=False
        )

        # Get the normalized state that was passed to the transformer
        transformer_input = tracking_model.normalized_state_seen

        print(f"Input state to transformer: {transformer_input}")
        print(f"Expected normalized state: {normalized_state}")

        # Calculate match percentage
        match = torch.isclose(transformer_input, normalized_state, rtol=1e-4)
        match_percentage = match.float().mean() * 100
        print(f"Match percentage: {match_percentage:.2f}%")

    # Show some model outputs
    print("Predicted forward states with normalization:",
          outputs_with_norm['predicted_forward_states'][0, 0, :])

    # Verify if normalization is working by checking if the tracked normalized state
    # matches our expectation
    if match_percentage > 99.0:
        print("✓ Normalization verification successful!")
    else:
        print("✗ Normalization verification failed!")

    print("\nTest completed!")


if __name__ == "__main__":
    test_transformer_normalization()
