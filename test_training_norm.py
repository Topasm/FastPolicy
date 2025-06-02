#!/usr/bin/env python3
"""
Script to test normalization in the training script.
"""

import torch
from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.common.datasets.utils import PolicyFeature
from lerobot.common.policies.normalize import Normalize, Unnormalize

# Create a similar setup to the training script
# Define features
state_dim = 7  # Example dimension
state_feature = PolicyFeature(
    type=FeatureType.STATE,
    shape=(state_dim,)
)
image_feature = PolicyFeature(
    type=FeatureType.VISUAL,
    shape=(3, 96, 96)
)

# Define features dictionary
input_features = {
    "observation.state": state_feature,
    "observation.image": image_feature
}

# Create synthetic stats for normalization
stats = {
    "observation.state": {
        "mean": torch.tensor([1.0] * state_dim),
        "std": torch.tensor([0.1] * state_dim)
    }
}

# Create normalization mapping
normalization_mapping = {
    FeatureType.STATE: NormalizationMode.MEAN_STD,
    FeatureType.VISUAL: NormalizationMode.IDENTITY
}
print(f"Using normalization mapping: {normalization_mapping}")

# Create normalizer
normalize_inputs = Normalize(input_features, normalization_mapping, stats)

# Test normalization
test_state = torch.tensor(
    [[2.0, 2.0] + [0.0] * (state_dim-2)], dtype=torch.float32)
test_batch = {"observation.state": test_state}
normalized_batch = normalize_inputs(test_batch)
normalized_state = normalized_batch["observation.state"]

print(f"Original state (first 2 dims): {test_state[0,:2]}")
print(f"Normalized state (first 2 dims): {normalized_state[0,:2]}")
print(f"Expected: tensor([10.0, 10.0])  # Because (2.0-1.0)/0.1 = 10.0")

# Create an unnormalizer using the same mapping
unnormalize_outputs = Unnormalize(
    input_features, normalization_mapping, stats
)

# Test unnormalization
unnormalized_batch = unnormalize_outputs(normalized_batch)
unnormalized_state = unnormalized_batch["observation.state"]

print(f"Unnormalized state (first 2 dims): {unnormalized_state[0,:2]}")
print(f"Should match original: {test_state[0,:2]}")

# Check if the values are close with a printed debug
is_close = torch.allclose(unnormalized_state, test_state, rtol=1e-4, atol=1e-4)
print(f"Values are close: {is_close}")
print(f"Difference: {torch.abs(unnormalized_state - test_state).max()}")

if is_close:
    print("✓ Normalization and unnormalization working correctly!")
else:
    print("✗ There's an issue with normalization/unnormalization.")
