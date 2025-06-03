#!/usr/bin/env python3
"""
Bidirectional Autoregressive Transformer for image-conditioned trajectory generation.

This model implements the following pipeline:
1. Input: initial image i_0 and state st_0
2. Generate forward states: st_0 → st_1 → ... → st_15
3. Generate goal image: i_n from st_15
4. Generate backward states: st_n → st_n-1 → ... → st_n-15

The model is trained autoregressively with proper causal masking.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json  # Added for config saving
from dataclasses import dataclass, asdict, field  # Ensure dataclass is imported
from typing import Optional, Dict, Any, Union
from pathlib import Path
from lerobot.configs.types import NormalizationMode

# Removed normalize imports as we handle normalization outside the model


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7                # Dimension of state vectors
    hidden_dim: int = 512             # Hidden dimension for transformer layers
    num_layers: int = 6               # Number of transformer layers
    num_heads: int = 8                # Number of attention heads
    dropout: float = 0.1              # Dropout rate
    max_position_value: int = 64      # Maximum position value for encoding
    layernorm_epsilon: float = 1e-5   # Epsilon for layer normalization
    image_channels: int = 3           # Number of image channels (RGB)
    image_size: int = 96              # Size of the input/output images
    image_latent_dim: int = 256       # Dimension of image latent representation
    forward_steps: int = 20
    backward_steps: int = 16
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        def feature_to_dict(feat):
            if hasattr(feat, 'to_dict'):
                return feat.to_dict()
            if hasattr(feat, '__dataclass_fields__'):  # Check if it's a dataclass
                return asdict(feat)
            return str(feat)

        d = asdict(self)
        # Ensure features are serializable
        d["input_features"] = {k: feature_to_dict(
            v) for k, v in self.input_features.items()}
        d["output_features"] = {k: feature_to_dict(
            v) for k, v in self.output_features.items()}
        return d

    def save_pretrained(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure path is a string for older python versions if json.dump requires it
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, output_dir: Path):
        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        # Deserialization of FeatureSpec might be needed here if they are complex
        # For now, assume they are simple dicts or basic types after to_dict
        return cls(**config_dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )


class ImageEncoder(nn.Module):
    """Encodes images into latent representations."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config

        # CNN encoder: 96x96x3 -> latent_dim
        self.encoder = nn.Sequential(
            # 96x96x3 -> 48x48x64
            nn.Conv2d(config.image_channels, 64,
                      kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 48x48x64 -> 24x24x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 24x24x128 -> 12x12x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 12x12x256 -> 6x6x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 6x6x512 -> 3x3x512
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Global average pooling: 3x3x512 -> 512
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            # Final projection to latent dimension
            nn.Linear(512, config.image_latent_dim),
            nn.ReLU()
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent representations.

        Args:
            images: [B, C, H, W] images

        Returns:
            latents: [B, image_latent_dim] latent representations
        """
        return self.encoder(images)


class ImageDecoder(nn.Module):
    """Decodes latent representations back into images."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config

        # Start from latent and go to 3x3x512
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 3 * 3),
            nn.ReLU()
        )

        # Transposed CNN decoder: latent -> 96x96x3
        self.decoder = nn.Sequential(
            # 3x3x512 -> 6x6x512
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # 6x6x512 -> 12x12x256
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # 12x12x256 -> 24x24x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # 24x24x128 -> 48x48x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # 48x48x64 -> 96x96x3
            nn.ConvTranspose2d(64, config.image_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representations to images.

        Args:
            latents: [B, image_latent_dim] latent representations

        Returns:
            images: [B, C, H, W] reconstructed images
        """
        # Project to initial spatial feature map
        x = self.initial_linear(latents)
        x = x.view(-1, 512, 3, 3)  # [B, 512, 3, 3]

        # Decode through transposed convolutions
        return self.decoder(x)


class BidirectionalARTransformer(nn.Module):
    """
    Bidirectional Autoregressive Transformer for image-conditioned trajectory generation.

    Pipeline:
    1. Encode initial image i_0 to latent representation
    2. Combine with initial state st_0
    3. Generate forward trajectory: st_0 → st_1 → ... → st_15 (autoregressive)
    4. Generate goal image i_n from final state st_15
    5. Generate backward trajectory: st_n → st_n-1 → ... → st_n-15 (autoregressive)

    Note: This transformer expects all inputs to be pre-normalized by calling code.
    Normalization should be handled externally before passing data to this model.
    """

    def __init__(self,
                 config: BidirectionalARTransformerConfig,
                 state_key: str = "observation.state",
                 image_key: str = "observation.image"):
        super().__init__()
        self.config = config

        # Keep track of state and image keys for reference
        self.state_key = state_key
        self.image_key = image_key

        # Import here to avoid circular import
        from lerobot.configs.types import FeatureType, NormalizationMode
        self.feature_type = FeatureType
        self.normalization_mode = NormalizationMode

        # Image encoder and decoder
        self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)

        # State and image latent projections to hidden dimension
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # Token type embeddings (0: image, 1: state, 2: goal_image, 3: forward_query, 4: goal_query, 5: backward_query)
        self.token_type_embedding = nn.Embedding(6, config.hidden_dim)

        # Position embeddings for the full sequence
        # We need positions for: 1 initial_image + 16 forward_states + 1 goal_image + 16 backward_states
        max_seq_len = 1 + config.forward_steps + 1 + config.backward_steps
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_dim)

        # Special readout tokens for non-autoregressive inference
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

        # Output prediction heads for the query tokens
        # Modified: Ensure we predict F-1 states for forward trajectory (as we have initial state)
        self.forward_state_head = nn.Linear(
            config.hidden_dim, (config.forward_steps-1) * config.state_dim)
        self.goal_image_latent_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)
        self.backward_state_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        # Output heads for different token types
        self.state_output_head = nn.Linear(config.hidden_dim, config.state_dim)
        self.image_latent_output_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights of the model."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create causal attention mask for autoregressive generation."""
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask

    def _create_query_based_mask(self, seq_len: int, device: torch.device, num_condition_tokens: int = 2) -> torch.Tensor:
        """
        Create attention mask for query-based non-autoregressive generation.

        In this mask, the query tokens can attend to the conditioning tokens (initial image and state)
        but not to each other, creating a parallelizable inference pattern.

        Args:
            seq_len: Total sequence length including condition and query tokens
            device: Device to create the mask on
            num_condition_tokens: Number of conditioning tokens (initial image + initial state)

        Returns:
            Attention mask of shape [seq_len, seq_len]
        """
        # Start with a fully masked tensor
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)

        # Allow all tokens to attend to the conditioning tokens (initial image, initial state)
        mask[:, :num_condition_tokens] = False

        # Allow regular attention pattern for conditioning tokens (can attend to previous tokens)
        for i in range(num_condition_tokens):
            mask[i, i+1:] = True  # Can't attend to future tokens

        return mask

    def _forward_training(
        self,
        initial_image_latents: torch.Tensor,
        initial_states: torch.Tensor,
        forward_states: torch.Tensor,
        goal_images: torch.Tensor,
        backward_states: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass using both autoregressive and query-based approaches."""
        batch_size = initial_image_latents.shape[0]

        # Encode goal images
        goal_image_latents = self.image_encoder(
            goal_images)  # [B, image_latent_dim]

        # === Autoregressive part (for better sequence learning) ===
        # Prepare sequence tokens for autoregressive learning
        ar_tokens = []
        ar_token_types = []

        # 1. Initial image latent
        ar_tokens.append(self.image_latent_projection(initial_image_latents))
        ar_token_types.append(0)  # image token

        # 2. Initial state (st_0)
        ar_tokens.append(self.state_projection(initial_states))
        ar_token_types.append(1)  # state token

        # 3. Forward trajectory states (st_1 to st_{F-1})
        future_forward_states_for_input = forward_states[:,
                                                         1:self.config.forward_steps]
        for i in range(self.config.forward_steps - 1):
            ar_tokens.append(self.state_projection(
                future_forward_states_for_input[:, i]))
            ar_token_types.append(1)  # state token

        # 4. Goal image latent
        ar_tokens.append(self.image_latent_projection(goal_image_latents))
        ar_token_types.append(2)  # goal image token

        # 5. Backward trajectory states
        for i in range(self.config.backward_steps):
            ar_tokens.append(self.state_projection(backward_states[:, i]))
            ar_token_types.append(1)  # state token

        # Stack tokens
        ar_sequence = torch.stack(ar_tokens, dim=1)  # [B, seq_len, hidden_dim]
        ar_seq_len = ar_sequence.shape[1]

        # Add token type embeddings
        ar_token_types_tensor = torch.tensor(
            ar_token_types, device=device).unsqueeze(0).expand(batch_size, -1)
        ar_type_embeddings = self.token_type_embedding(ar_token_types_tensor)
        ar_sequence = ar_sequence + ar_type_embeddings

        # Add position embeddings
        ar_positions = torch.arange(ar_seq_len, device=device).unsqueeze(
            0).expand(batch_size, -1)
        ar_pos_embeddings = self.position_embedding(ar_positions)
        ar_sequence = ar_sequence + ar_pos_embeddings

        # Create causal mask for autoregressive learning
        ar_causal_mask = self._create_causal_mask(ar_seq_len, device)

        # Pass through transformer for autoregressive learning
        ar_hidden_states = self.transformer(
            src=ar_sequence,
            mask=ar_causal_mask
        )

        # Extract predictions for different parts of the sequence (autoregressive)
        ar_results = {}
        F = self.config.forward_steps
        B = self.config.backward_steps

        # Forward states predictions (st_1 to st_{F-1})
        forward_hidden = ar_hidden_states[:, 1:F]
        ar_results['predicted_forward_states'] = self.state_output_head(
            forward_hidden)

        # Goal image prediction
        goal_image_predictor_hidden = ar_hidden_states[:, F]
        predicted_goal_latents = self.image_latent_output_head(
            goal_image_predictor_hidden)
        ar_results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)
        ar_results['predicted_goal_latents'] = predicted_goal_latents

        # Backward states predictions
        backward_hidden = ar_hidden_states[:, F + 1: F + 1 + B]
        ar_results['predicted_backward_states'] = self.state_output_head(
            backward_hidden)

        # === Query-based part (matching inference approach) ===
        # Project initial image latents and state to hidden dimension
        projected_initial_image = self.image_latent_projection(
            initial_image_latents)
        projected_initial_state = self.state_projection(initial_states)

        # Reshape to [B, 1, D] for sequence processing
        projected_initial_image = projected_initial_image.unsqueeze(1)
        projected_initial_state = projected_initial_state.unsqueeze(1)

        # Expand query tokens for the batch
        batch_forward_query = self.forward_seq_query_token.expand(
            batch_size, -1, -1)
        batch_goal_query = self.goal_image_query_token.expand(
            batch_size, -1, -1)
        batch_backward_query = self.backward_seq_query_token.expand(
            batch_size, -1, -1)

        # Construct the input sequence: [initial_image, initial_state, forward_query, goal_query, backward_query]
        query_sequence = torch.cat([
            projected_initial_image,
            projected_initial_state,
            batch_forward_query,
            batch_goal_query,
            batch_backward_query
        ], dim=1)  # [B, 5, D]

        # Define token types: 0=image, 1=state, 3=forward_query, 4=goal_query, 5=backward_query
        query_token_types_tensor = torch.tensor(
            [0, 1, 3, 4, 5], device=device).unsqueeze(0).expand(batch_size, -1)

        # Add token type embeddings
        query_type_embeddings = self.token_type_embedding(
            query_token_types_tensor)
        query_sequence = query_sequence + query_type_embeddings

        # Add position embeddings
        query_positions = torch.arange(
            5, device=device).unsqueeze(0).expand(batch_size, -1)
        query_pos_embeddings = self.position_embedding(query_positions)
        query_sequence = query_sequence + query_pos_embeddings

        # Create attention mask for query-based approach
        num_condition_tokens = 2  # initial_image and initial_state
        query_attn_mask = self._create_query_based_mask(
            5, device, num_condition_tokens)

        # Pass through transformer for query-based learning
        query_hidden_states = self.transformer(
            src=query_sequence,
            mask=query_attn_mask
        )

        # Extract hidden states for the query tokens
        forward_query_hidden = query_hidden_states[:, 2]  # [B, D]
        goal_query_hidden = query_hidden_states[:, 3]     # [B, D]
        backward_query_hidden = query_hidden_states[:, 4]  # [B, D]

        # Predict forward states sequence (F-1 states, as the initial state is given)
        query_predicted_fwd_states_flat = self.forward_state_head(
            forward_query_hidden)

        # Fix: Ensure the reshaping is consistent with the linear layer output
        # We're predicting (F-1) states, each of state_dim dimensions
        query_fwd_states = query_predicted_fwd_states_flat.view(
            batch_size, self.config.forward_steps-1, self.config.state_dim
        )

        # Predict goal image latent
        query_predicted_goal_latents = self.goal_image_latent_head(
            goal_query_hidden)
        query_goal_images = self.image_decoder(query_predicted_goal_latents)

        # Predict backward states sequence
        query_predicted_bwd_states_flat = self.backward_state_head(
            backward_query_hidden)
        query_bwd_states = query_predicted_bwd_states_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim
        )

        # Combine results from both approaches
        results = {
            # Autoregressive results
            'predicted_forward_states': ar_results['predicted_forward_states'],
            'predicted_goal_images': ar_results['predicted_goal_images'],
            'predicted_goal_latents': ar_results['predicted_goal_latents'],
            'predicted_backward_states': ar_results['predicted_backward_states'],

            # Query-based results (for training the query approach to match autoregressive)
            'query_predicted_forward_states': query_fwd_states,
            'query_predicted_goal_images': query_goal_images,
            'query_predicted_goal_latents': query_predicted_goal_latents,
            'query_predicted_backward_states': query_bwd_states
        }

        return results

    def _forward_inference(
        self,
        initial_image_latents: torch.Tensor,
        initial_states: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Non-autoregressive inference using special query tokens."""
        batch_size = initial_image_latents.shape[0]
        results = {}

        # Project initial image latents and state to hidden dimension
        projected_initial_image = self.image_latent_projection(
            initial_image_latents)  # [B, D]
        projected_initial_state = self.state_projection(
            initial_states)  # [B, D]

        # Reshape to [B, 1, D] for sequence processing
        projected_initial_image = projected_initial_image.unsqueeze(
            1)  # [B, 1, D]
        projected_initial_state = projected_initial_state.unsqueeze(
            1)  # [B, 1, D]

        # Expand query tokens for the batch
        batch_forward_query = self.forward_seq_query_token.expand(
            batch_size, -1, -1)  # [B, 1, D]
        batch_goal_query = self.goal_image_query_token.expand(
            batch_size, -1, -1)  # [B, 1, D]
        batch_backward_query = self.backward_seq_query_token.expand(
            batch_size, -1, -1)  # [B, 1, D]

        # Construct the input sequence: [initial_image, initial_state, forward_query, goal_query, backward_query]
        sequence = torch.cat([
            projected_initial_image,       # [B, 1, D]
            projected_initial_state,       # [B, 1, D]
            batch_forward_query,           # [B, 1, D]
            batch_goal_query,              # [B, 1, D]
            batch_backward_query           # [B, 1, D]
        ], dim=1)  # [B, 5, D]

        # Define token types: 0=image, 1=state, 3=forward_query, 4=goal_query, 5=backward_query
        token_types_tensor = torch.tensor(
            [0, 1, 3, 4, 5], device=device
        ).unsqueeze(0).expand(batch_size, -1)  # [B, 5]

        # Add token type embeddings
        type_embeddings = self.token_type_embedding(token_types_tensor)
        sequence = sequence + type_embeddings

        # Add position embeddings
        positions = torch.arange(5, device=device).unsqueeze(
            0).expand(batch_size, -1)
        pos_embeddings = self.position_embedding(positions)
        sequence = sequence + pos_embeddings

        # Create attention mask - query tokens can attend only to condition tokens
        num_condition_tokens = 2  # initial_image and initial_state
        attn_mask = self._create_query_based_mask(
            5, device, num_condition_tokens)

        # Pass through transformer
        hidden_states = self.transformer(
            src=sequence,
            mask=attn_mask
        )

        # Extract hidden states for the query tokens
        forward_query_hidden = hidden_states[:, 2]  # [B, D]
        goal_query_hidden = hidden_states[:, 3]     # [B, D]
        backward_query_hidden = hidden_states[:, 4]  # [B, D]

        # Predict forward states sequence
        predicted_fwd_states_flat = self.forward_state_head(
            forward_query_hidden)

        # Fix: Ensure the reshaping is consistent with the linear layer output
        results['predicted_forward_states'] = predicted_fwd_states_flat.view(
            batch_size, self.config.forward_steps-1, self.config.state_dim
        )

        # Predict goal image latent
        predicted_goal_latents = self.goal_image_latent_head(goal_query_hidden)
        results['predicted_goal_latents'] = predicted_goal_latents
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)

        # Predict backward states sequence
        predicted_bwd_states_flat = self.backward_state_head(
            backward_query_hidden)
        results['predicted_backward_states'] = predicted_bwd_states_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim
        )

        return results

    # Methods _normalize_if_needed and _unnormalize_if_needed have been replaced with direct
    # calls to normalizer and unnormalizer in the forward method for more reliable normalization

    def forward(
        self,
        initial_images: torch.Tensor,
        initial_states: torch.Tensor,
        forward_states: Optional[torch.Tensor] = None,
        goal_images: Optional[torch.Tensor] = None,
        backward_states: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the bidirectional autoregressive transformer.

        Note: All inputs should already be normalized before being passed to this method.
        Normalization is handled outside the model.

        Args:
            initial_images: [B, C, H, W] initial images
            initial_states: [B, state_dim] initial states (normalized)
            forward_states: [B, forward_steps, state_dim] forward trajectory states (normalized, training only)
            goal_images: [B, C, H, W] goal images (training only)
            backward_states: [B, backward_steps, state_dim] backward trajectory states (normalized, training only)
            training: Whether in training mode

        Returns:
            Dictionary containing predicted states, images, and latents (all outputs are in normalized space)
        """
        device = initial_images.device

        # Skip normalization since we're now doing it externally before passing to the model
        normalized_initial_states = initial_states
        normalized_forward_states = forward_states
        normalized_backward_states = backward_states

        # The normalizer is no longer needed here since we normalize the batch before model.forward()

        # Encode initial image
        initial_image_latents = self.image_encoder(
            initial_images)  # [B, image_latent_dim]

        if training:
            # Training mode: teacher forcing with full sequence
            results = self._forward_training(
                initial_image_latents, normalized_initial_states, normalized_forward_states,
                goal_images, normalized_backward_states, device
            )
        else:
            # Inference mode: non-autoregressive generation with query tokens
            results = self._forward_inference(
                initial_image_latents, normalized_initial_states, device)

        # No normalization or unnormalization needed here
        # All normalization happens outside the model

        return results


@classmethod
def from_pretrained(cls, path, device=None, **kwargs):
    """
    Load a transformer from pretrained files.

    Args:
        path: Path to the directory containing model checkpoint and config
        device: Device to load the model on
        **kwargs: Additional arguments for model initialization

    Returns:
        Initialized BidirectionalARTransformer
    """
    from pathlib import Path
    import torch

    path = Path(path)

    # Load config
    try:
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = BidirectionalARTransformerConfig(**config_dict)
    except Exception as e:
        print(f"Error loading config: {e}")
        raise

    # Create model
    model = cls(config=config, **kwargs)

    # Load weights
    try:
        model_path = path / "model_final.pth"
        if not model_path.exists():
            # Try alternative names
            candidates = list(path.glob("*.pth"))
            if candidates:
                model_path = candidates[0]
                print(f"Using model checkpoint: {model_path}")

        if device is not None:
            state_dict = torch.load(model_path, map_location=device)
        else:
            state_dict = torch.load(model_path)

        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model weights: {e}")
        raise

    # Move model to device if specified
    if device is not None:
        model = model.to(device)

    return model


def compute_loss(
    model: 'BidirectionalARTransformer',
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive loss for the bidirectional autoregressive model.

    Args:
        model: The BidirectionalARTransformer model instance.
        predictions: Model predictions (already normalized).
        targets: Ground truth targets (already normalized).

    Returns:
        Dictionary of losses.
    """
    losses = {}

    # State prediction losses (MSE) - for autoregressive outputs
    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        # Use the ground truth states from second state onward
        target_states = targets['forward_states'][:, 1:]
        losses['forward_state_loss'] = F.mse_loss(
            predictions['predicted_forward_states'],
            target_states
        )

    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.mse_loss(
            predictions['predicted_backward_states'],
            targets['backward_states']
        )

    # Image reconstruction losses
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'],
            targets['goal_images']
        )

    # Latent consistency losses
    if 'predicted_goal_latents' in predictions and 'goal_images' in targets:
        with torch.no_grad():  # Ensure encoder is not trained on this reconstruction
            # No need to normalize images as the image_encoder expects raw pixel values
            goal_image_latents = model.image_encoder(targets['goal_images'])
        losses['goal_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_goal_latents'],
            goal_image_latents
        )

    # Query-based prediction losses (to align with autoregressive outputs)
    if 'query_predicted_forward_states' in predictions and 'forward_states' in targets:
        # All values are already properly normalized at this point
        target_states = targets['forward_states'][:, 1:]
        losses['query_forward_state_loss'] = F.mse_loss(
            predictions['query_predicted_forward_states'],
            target_states
        )

    if 'query_predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['query_backward_state_loss'] = F.mse_loss(
            predictions['query_predicted_backward_states'],
            targets['backward_states']
        )

    if 'query_predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['query_goal_image_loss'] = F.mse_loss(
            predictions['query_predicted_goal_images'],
            targets['goal_images']
        )

    if 'query_predicted_goal_latents' in predictions and 'goal_images' in targets:
        with torch.no_grad():
            # No need to normalize images as the image_encoder expects raw pixel values
            goal_image_latents = model.image_encoder(targets['goal_images'])
        losses['query_goal_latent_consistency_loss'] = F.mse_loss(
            predictions['query_predicted_goal_latents'],
            goal_image_latents
        )

    # Compute weighted total loss
    weights = {
        # Original autoregressive weights
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 2.0,
        'goal_latent_consistency_loss': 1.0,
        # Query-based weights
        'query_forward_state_loss': 1.0,
        'query_backward_state_loss': 1.0,
        'query_goal_image_loss': 2.0,
        'query_goal_latent_consistency_loss': 1.0,
    }

    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)
    for loss_name, loss_value in losses.items():
        if loss_name in weights:  # only include weighted losses in the sum for total_loss
            weight = weights.get(loss_name, 1.0)
            total_loss += weight * loss_value

    losses['total_loss'] = total_loss
    return losses
