#!/usr/bin/env python3
"""
A comprehensive test for normalization and unnormalization in BidirectionalARTransformer.
"""

import torch
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.common.datasets.utils import PolicyFeature
from lerobot.common.policies.normalize import Normalize, Unnormalize
from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)
import sys
print("Python Path:", sys.path)


def test_transformer_normalization():
    """Test normalization in the BidirectionalARTransformer."""
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
        FeatureType.STATE: NormalizationMode.MEAN_STD,
        FeatureType.VISUAL: NormalizationMode.IDENTITY
    }
    print(f"Using normalization mapping: {normalization_mapping}")

    # Create normalizer and unnormalizer
    normalizer = Normalize(input_features, normalization_mapping, stats)
    unnormalizer = Unnormalize(output_features, normalization_mapping, stats)

    # Test basic normalization
    test_state = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    test_batch = {"observation.state": test_state}
    normalized_batch = normalizer(test_batch)
    normalized_state = normalized_batch["observation.state"]

    print(f"Original state: {test_state}")
    print(f"Normalized state: {normalized_state}")
    print(f"Expected: tensor([[10.0, 10.0]])  # Because (2.0-1.0)/0.1 = 10.0")

    # Test basic unnormalization
    unnormalized_batch = unnormalizer(normalized_batch)
    unnormalized_state = unnormalized_batch["observation.state"]

    print(f"Unnormalized state: {unnormalized_state}")
    print(f"Should match original: {test_state}")

    is_norm_correct = torch.allclose(
        normalized_state, torch.tensor([[10.0, 10.0]]), rtol=1e-4)
    is_unnorm_correct = torch.allclose(
        unnormalized_state, test_state, rtol=1e-4)

    print(f"Normalization correct: {is_norm_correct}")
    print(f"Unnormalization correct: {is_unnorm_correct}")

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

    # Create test input
    initial_state = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
    initial_image = torch.zeros((batch_size, 3, 96, 96), dtype=torch.float32)

    # Create a tracking model to verify normalization within the transformer
    class TrackingTransformer(BidirectionalARTransformer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.normalized_state_seen = None
            self.unnormalized_state_seen = None

        def _forward_inference(self, initial_image_latents, initial_states, device):
            self.normalized_state_seen = initial_states.clone()
            results = super()._forward_inference(
                initial_image_latents, initial_states, device)
            self.unnormalized_state_seen = results['predicted_forward_states'][0, 0].clone(
            )
            return results

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
        outputs_with_tracking = tracking_model(
            initial_images=initial_image,
            initial_states=initial_state,
            training=False
        )

        # Get the normalized state that was passed to the transformer
        transformer_input = tracking_model.normalized_state_seen

        print(f"\nVerifying transformer normalization...")
        print(f"Input state to transformer: {transformer_input}")
        print(f"Expected normalized state: {normalized_state}")

        # Calculate match percentage for normalization
        norm_match = torch.isclose(
            transformer_input, normalized_state, rtol=1e-4)
        norm_match_percentage = norm_match.float().mean() * 100
        print(f"Normalization match percentage: {norm_match_percentage:.2f}%")

        # Verify unnormalization (should be somewhere in the output)
        print(
            f"Sample output from transformer: {tracking_model.unnormalized_state_seen}")
        print(f"Should be unnormalized (values in original space)")

    if norm_match_percentage > 99.0 and is_norm_correct and is_unnorm_correct:
        print("\n✓ Normalization and unnormalization verification successful!")
    else:
        print("\n✗ Normalization and unnormalization verification failed!")

    print("\nTest completed!")


if __name__ == "__main__":
    test_transformer_normalization()
