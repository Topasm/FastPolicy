#!/usr/bin/env python3
"""
A simple test script to understand how normalization works in the lerobot library.
"""

from lerobot.configs.types import FeatureType, NormalizationMode
from lerobot.common.datasets.utils import PolicyFeature
from lerobot.common.policies.normalize import Normalize, Unnormalize
import torch
import sys
print("Python Path:", sys.path)

# Import lerobot normalization components

# Create a simple dataset with well-defined values
state_dim = 2

# Define a feature
state_feature = PolicyFeature(
    type=FeatureType.STATE, shape=(state_dim,))

# Define features dictionary
features = {
    "observation.state": state_feature,
}

# Create synthetic stats for normalization
# state will be normalized with mean 1 and std 0.1
stats = {
    "observation.state": {
        "mean": torch.tensor([1.0, 1.0]),
        "std": torch.tensor([0.1, 0.1])
    }
}

# Print the normalization mode available
# (inspect what modes are available in the Normalize class)
print("Available normalization modes:")
normalize_cls = Normalize
print([attr for attr in dir(normalize_cls) if attr.startswith('NORMALIZATION_')])

# Create proper normalization mapping using the enum
normalization_mapping = {
    FeatureType.STATE: NormalizationMode.MEAN_STD  # Using the enum directly
}

print("\nTesting normalization with MEAN_STD mode:")
normalizer = Normalize(features, normalization_mapping, stats)

# Test normalization
test_state = torch.tensor([[2.0, 2.0]], dtype=torch.float32)
test_batch = {"observation.state": test_state}

normalized_batch = normalizer(test_batch)
normalized_state = normalized_batch["observation.state"]

print(f"Original state: {test_state}")
print(f"Normalized state: {normalized_state}")
print(f"Expected: tensor([[10.0, 10.0]])  # Because (2.0-1.0)/0.1 = 10.0")

# Create unnormalizer
unnormalizer = Unnormalize(features, normalization_mapping, stats)
unnormalized_batch = unnormalizer(normalized_batch)
unnormalized_state = unnormalized_batch["observation.state"]

print(f"Unnormalized state: {unnormalized_state}")
print(f"Should match original: {test_state}")
