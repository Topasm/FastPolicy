#!/usr/bin/env python3
"""
Utility functions to handle normalization with different key mappings.
"""

import torch
from typing import Dict, Any, Optional
from lerobot.common.policies.normalize import Normalize, Unnormalize


class KeyMappingNormalizer:
    """
    Wrapper around Normalize to map between different key structures.
    This allows using a normalizer with statistics for "observation.state"
    but applying it to tensors with different keys like "initial_states".
    """

    def __init__(
        self,
        normalizer: Normalize,
        key_mapping: Dict[str, str]
    ):
        """
        Initialize the key mapping normalizer.

        Args:
            normalizer: The base normalizer to use
            key_mapping: Dict mapping from input keys to normalizer keys
                e.g. {"initial_states": "observation.state"}
        """
        self.normalizer = normalizer
        self.key_mapping = key_mapping

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply normalization with key mapping.

        Args:
            batch: Input batch with keys that might not match normalizer keys

        Returns:
            Normalized batch with original keys
        """
        # Create a copy of the input batch
        result_batch = dict(batch)

        # For each key in the input batch that has a mapping
        for input_key, norm_key in self.key_mapping.items():
            if input_key in batch:
                # Create a temporary batch with the correct key for the normalizer
                temp_batch = {norm_key: batch[input_key]}

                # Normalize using the base normalizer
                normalized_temp = self.normalizer(temp_batch)

                # Copy the result back to the output batch with the original key
                result_batch[input_key] = normalized_temp[norm_key]

        return result_batch


class KeyMappingUnnormalizer:
    """
    Wrapper around Unnormalize to map between different key structures.
    Similar to KeyMappingNormalizer but for unnormalizing outputs.
    """

    def __init__(
        self,
        unnormalizer: Unnormalize,
        key_mapping: Dict[str, str]
    ):
        """
        Initialize the key mapping unnormalizer.

        Args:
            unnormalizer: The base unnormalizer to use
            key_mapping: Dict mapping from input keys to unnormalizer keys
                e.g. {"predicted_forward_states": "observation.state"}
        """
        self.unnormalizer = unnormalizer
        self.key_mapping = key_mapping

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Apply unnormalization with key mapping.

        Args:
            batch: Input batch with keys that might not match unnormalizer keys

        Returns:
            Unnormalized batch with original keys
        """
        # Create a copy of the input batch
        result_batch = dict(batch)

        # For each key in the input batch that has a mapping
        for input_key, unnorm_key in self.key_mapping.items():
            if input_key in batch:
                # Create a temporary batch with the correct key for the unnormalizer
                temp_batch = {unnorm_key: batch[input_key]}

                # Unnormalize using the base unnormalizer
                unnormalized_temp = self.unnormalizer(temp_batch)

                # Copy the result back to the output batch with the original key
                result_batch[input_key] = unnormalized_temp[unnorm_key]

        return result_batch
