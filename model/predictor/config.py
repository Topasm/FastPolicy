#!/usr/bin/env python3
"""
Configuration classes for the Bidirectional Autoregressive Transformer.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from lerobot.configs.types import NormalizationMode


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7
    hidden_dim: int = 256  # Main dimension parameter used throughout the model
    num_layers: int = 8
    num_heads: int = 8  # Use 8 heads for even division of 256
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-5

    # Visual parameters
    image_channels: int = 3
    image_size: int = 84  # Default input image size
    output_image_size: int = 96  # Output size after upsampling

    # image_latent_dim property that returns hidden_dim for consistency
    @property
    def image_latent_dim(self) -> int:
        """Ensure image latent dimension always matches hidden_dim for consistency"""
        return self.hidden_dim
    crop_is_random: bool = True  # Whether to use random crop during training

    # Sequence parameters
    max_sequence_length: int = 128
    forward_steps: int = 32
    backward_steps: int = 32
    n_obs_steps: int = 3  # Number of observation steps for temporal encoding

    # Feature specifications
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    # Normalization mapping
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )

    def save_pretrained(self, output_dir):
        """Save the configuration to a JSON file."""
        import json
        from pathlib import Path

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Convert to serializable dict
        config_dict = {k: v for k, v in self.__dict__.items()}

        # Handle non-serializable objects
        if 'input_features' in config_dict:
            config_dict['input_features'] = {
                k: str(v) for k, v in config_dict['input_features'].items()}
        if 'output_features' in config_dict:
            config_dict['output_features'] = {
                k: str(v) for k, v in config_dict['output_features'].items()}
        if 'normalization_mapping' in config_dict:
            config_dict['normalization_mapping'] = {k: v.value if hasattr(v, 'value') else v
                                                    for k, v in config_dict['normalization_mapping'].items()}

        # Save to file
        with open(output_dir / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
