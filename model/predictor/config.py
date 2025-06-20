#!/usr/bin/env python3
"""
Configuration classes for the Bidirectional Autoregressive Transformer.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any
from lerobot.configs.types import NormalizationMode
from pathlib import Path
import json


@dataclass
class HierarchicalPolicyConfig:
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
    forward_steps: int = 64
    backward_steps: int = 64
    n_obs_steps: int = 3  # Number of observation steps for temporal encoding
    n_action_steps: int = 8  # Number of action steps to predict

    # Feature specifications
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        def feature_to_dict(feat):
            if hasattr(feat, 'to_dict'):
                return feat.to_dict()
            if hasattr(feat, '__dataclass_fields__'):
                return asdict(feat)
            return str(feat)
        d = asdict(self)
        d["input_features"] = {k: feature_to_dict(
            v) for k, v in self.input_features.items()}
        d["output_features"] = {k: feature_to_dict(
            v) for k, v in self.output_features.items()}
        return d

    def save_pretrained(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, output_dir: Path):

        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )
