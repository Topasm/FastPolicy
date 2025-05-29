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
from typing import Optional, Dict, Any
from pathlib import Path


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
    forward_steps: int = 16
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
    """

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config

        # Image encoder and decoder
        self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)

        # State and image latent projections to hidden dimension
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # Token type embeddings (0: image, 1: state, 2: goal_image)
        self.token_type_embedding = nn.Embedding(3, config.hidden_dim)

        # Position embeddings for the full sequence
        # We need positions for: 1 initial_image + 16 forward_states + 1 goal_image + 16 backward_states
        max_seq_len = 1 + config.forward_steps + 1 + config.backward_steps
        self.position_embedding = nn.Embedding(max_seq_len, config.hidden_dim)

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

        Args:
            initial_images: [B, C, H, W] initial images
            initial_states: [B, state_dim] initial states
            forward_states: [B, forward_steps, state_dim] forward trajectory states (training only)
            goal_images: [B, C, H, W] goal images (training only)
            backward_states: [B, backward_steps, state_dim] backward trajectory states (training only)
            training: Whether in training mode

        Returns:
            Dictionary containing predicted states, images, and latents
        """
        device = initial_images.device

        # Encode initial image
        initial_image_latents = self.image_encoder(
            initial_images)  # [B, image_latent_dim]

        if training:
            # Training mode: teacher forcing with full sequence
            return self._forward_training(
                initial_image_latents, initial_states, forward_states,
                goal_images, backward_states, device
            )
        else:
            # Inference mode: autoregressive generation
            return self._forward_inference(initial_image_latents, initial_states, device)

    def _forward_training(
        self,
        initial_image_latents: torch.Tensor,
        initial_states: torch.Tensor,
        forward_states: torch.Tensor,
        goal_images: torch.Tensor,
        backward_states: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Training forward pass with teacher forcing."""
        batch_size = initial_image_latents.shape[0]

        # Encode goal images
        goal_image_latents = self.image_encoder(
            goal_images)  # [B, image_latent_dim]

        # Prepare sequence tokens
        tokens = []
        token_types = []

        # 1. Initial image latent
        tokens.append(self.image_latent_projection(initial_image_latents))
        token_types.append(0)  # image token

        # 2. Initial state
        tokens.append(self.state_projection(initial_states))
        token_types.append(1)  # state token

        # 3. Forward trajectory states (st_1 to st_15)
        for i in range(self.config.forward_steps - 1):
            tokens.append(self.state_projection(forward_states[:, i]))
            token_types.append(1)  # state token

        # 4. Goal image latent
        tokens.append(self.image_latent_projection(goal_image_latents))
        token_types.append(2)  # goal image token

        # 5. Backward trajectory states
        for i in range(self.config.backward_steps):
            tokens.append(self.state_projection(backward_states[:, i]))
            token_types.append(1)  # state token

        # Stack tokens
        sequence = torch.stack(tokens, dim=1)  # [B, seq_len, hidden_dim]
        seq_len = sequence.shape[1]

        # Add token type embeddings
        token_types_tensor = torch.tensor(
            token_types, device=device).unsqueeze(0).expand(batch_size, -1)
        type_embeddings = self.token_type_embedding(token_types_tensor)
        sequence = sequence + type_embeddings

        # Add position embeddings
        positions = torch.arange(seq_len, device=device).unsqueeze(
            0).expand(batch_size, -1)
        pos_embeddings = self.position_embedding(positions)
        sequence = sequence + pos_embeddings

        # Create causal mask
        causal_mask = self._create_causal_mask(seq_len, device)

        # Pass through transformer
        hidden_states = self.transformer(
            src=sequence,
            mask=causal_mask
        )

        # Extract predictions for different parts of the sequence
        results = {}

        # Forward states predictions (positions 2 to forward_steps)
        # [B, forward_steps-1, hidden_dim]
        forward_hidden = hidden_states[:, 2:2+self.config.forward_steps-1]
        results['predicted_forward_states'] = self.state_output_head(
            forward_hidden)

        # Goal image prediction (position forward_steps+1)
        # [B, 1, hidden_dim]
        goal_hidden = hidden_states[:, 1 +
                                    self.config.forward_steps:1+self.config.forward_steps+1]
        predicted_goal_latents = self.image_latent_output_head(
            goal_hidden.squeeze(1))
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)
        results['predicted_goal_latents'] = predicted_goal_latents

        # Backward states predictions
        backward_start = 2 + self.config.forward_steps
        backward_hidden = hidden_states[:,
                                        backward_start:backward_start+self.config.backward_steps]
        results['predicted_backward_states'] = self.state_output_head(
            backward_hidden)

        return results

    def _forward_inference(
        self,
        initial_image_latents: torch.Tensor,
        initial_states: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Inference forward pass with autoregressive generation."""
        batch_size = initial_image_latents.shape[0]

        # Start with initial image and state
        tokens = [
            self.image_latent_projection(initial_image_latents),
            self.state_projection(initial_states)
        ]
        token_types = [0, 1]  # image, state

        results = {
            'predicted_forward_states': [],
            'predicted_backward_states': []
        }

        # Generate forward trajectory autoregressively
        for step in range(self.config.forward_steps - 1):
            # [B, current_len, hidden_dim]
            current_seq = torch.stack(tokens, dim=1)
            current_len = current_seq.shape[1]

            # Add embeddings
            token_types_tensor = torch.tensor(
                token_types, device=device).unsqueeze(0).expand(batch_size, -1)
            type_embeddings = self.token_type_embedding(token_types_tensor)
            current_seq = current_seq + type_embeddings

            positions = torch.arange(current_len, device=device).unsqueeze(
                0).expand(batch_size, -1)
            pos_embeddings = self.position_embedding(positions)
            current_seq = current_seq + pos_embeddings

            # Create causal mask
            causal_mask = self._create_causal_mask(current_len, device)

            # Forward pass
            hidden_states = self.transformer(src=current_seq, mask=causal_mask)

            # Predict next state
            next_state = self.state_output_head(
                hidden_states[:, -1])  # [B, state_dim]
            results['predicted_forward_states'].append(next_state)

            # Add to sequence
            tokens.append(self.state_projection(next_state))
            token_types.append(1)  # state token

        # Generate goal image from final forward state
        current_seq = torch.stack(tokens, dim=1)
        current_len = current_seq.shape[1]

        # Add embeddings
        token_types_tensor = torch.tensor(
            token_types, device=device).unsqueeze(0).expand(batch_size, -1)
        type_embeddings = self.token_type_embedding(token_types_tensor)
        current_seq = current_seq + type_embeddings

        positions = torch.arange(current_len, device=device).unsqueeze(
            0).expand(batch_size, -1)
        pos_embeddings = self.position_embedding(positions)
        current_seq = current_seq + pos_embeddings

        causal_mask = self._create_causal_mask(current_len, device)
        hidden_states = self.transformer(src=current_seq, mask=causal_mask)

        # Predict goal image latent
        predicted_goal_latents = self.image_latent_output_head(
            hidden_states[:, -1])
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)
        results['predicted_goal_latents'] = predicted_goal_latents

        # Add goal image to sequence
        tokens.append(self.image_latent_projection(predicted_goal_latents))
        token_types.append(2)  # goal image token

        # Generate backward trajectory autoregressively
        for step in range(self.config.backward_steps):
            current_seq = torch.stack(tokens, dim=1)
            current_len = current_seq.shape[1]

            # Add embeddings
            token_types_tensor = torch.tensor(
                token_types, device=device).unsqueeze(0).expand(batch_size, -1)
            type_embeddings = self.token_type_embedding(token_types_tensor)
            current_seq = current_seq + type_embeddings

            positions = torch.arange(current_len, device=device).unsqueeze(
                0).expand(batch_size, -1)
            pos_embeddings = self.position_embedding(positions)
            current_seq = current_seq + pos_embeddings

            causal_mask = self._create_causal_mask(current_len, device)
            hidden_states = self.transformer(src=current_seq, mask=causal_mask)

            # Predict next backward state
            next_state = self.state_output_head(hidden_states[:, -1])
            results['predicted_backward_states'].append(next_state)

            # Add to sequence
            tokens.append(self.state_projection(next_state))
            token_types.append(1)  # state token

        # Stack the lists into tensors
        if results['predicted_forward_states']:
            results['predicted_forward_states'] = torch.stack(
                results['predicted_forward_states'], dim=1)
        if results['predicted_backward_states']:
            results['predicted_backward_states'] = torch.stack(
                results['predicted_backward_states'], dim=1)

        return results


def compute_loss(
    model: 'BidirectionalARTransformer',
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute comprehensive loss for the bidirectional autoregressive model.

    Args:
        model: The BidirectionalARTransformer model instance.
        predictions: Model predictions.
        targets: Ground truth targets.

    Returns:
        Dictionary of losses.
    """
    losses = {}

    # State prediction losses (MSE)
    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        losses['forward_state_loss'] = F.mse_loss(
            predictions['predicted_forward_states'],
            # Ground truth for st_1 to st_{T-1}
            targets['forward_states'][:, 1:]
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
            goal_image_latents = model.image_encoder(targets['goal_images'])
        losses['goal_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_goal_latents'],
            goal_image_latents
        )

    # Compute weighted total loss
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 2.0,
        'goal_latent_consistency_loss': 1.0,
    }

    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)
    for loss_name, loss_value in losses.items():
        if loss_name in weights:  # only include weighted losses in the sum for total_loss
            weight = weights.get(loss_name, 1.0)
            total_loss += weight * loss_value

    losses['total_loss'] = total_loss
    return losses
