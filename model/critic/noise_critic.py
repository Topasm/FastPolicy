import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class NoiseCriticConfig:
    """Configuration for the noise trajectory critic model."""
    state_dim: int  # Dimension of state vectors
    horizon: int  # Length of state sequence to evaluate
    hidden_dim: int = 512  # Hidden dimension of the network
    num_layers: int = 4  # Number of transformer layers or MLP blocks
    dropout: float = 0.1
    use_layernorm: bool = True
    use_image_context: bool = False  # Whether to use image features as context
    image_feature_dim: int = 0  # Dimension of image features (if used)
    # Parameters for transformer
    n_heads: int = 8  # Number of attention heads


class TransformerCritic(nn.Module):
    """
    Transformer-based critic that scores a sequence of states.
    Input: (B, H, D_state)
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: NoiseCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = config.use_image_context

        # State embedding layer (to transformer hidden dim)
        self.state_embedding = nn.Linear(config.state_dim, config.hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.horizon, config.hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,  # 8 heads is a common choice
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers
        )

        # Image context processing if needed
        if self.use_image_context:
            self.img_encoder = nn.Sequential(
                nn.Linear(config.image_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(
                    config.hidden_dim) if config.use_layernorm else nn.Identity(),
                nn.Dropout(config.dropout)
            )
            # We'll use image features as a prefix token

        # Output head (token aggregation + linear layer)
        self.output_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(
                config.hidden_dim) if config.use_layernorm else nn.Identity(),
            nn.Linear(config.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Use Xavier initialization for transformers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)

    def forward(self, trajectory_sequence: torch.Tensor, image_features=None) -> torch.Tensor:
        """
        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            image_features: Optional (B, D_img) tensor of image features.
        Returns:
            (B, 1) tensor of scores (logits).
        """
        B, H, D = trajectory_sequence.shape
        if H != self.horizon or D != self.state_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (B, {self.horizon}, {self.state_dim}), got {(B, H, D)}")

        # Embed states to hidden dimension
        state_embeddings = self.state_embedding(
            trajectory_sequence)  # (B, H, hidden_dim)

        # Add positional embeddings
        state_embeddings = state_embeddings + self.pos_embedding

        # Process image features and prepend to sequence if provided
        sequence_for_transformer = state_embeddings

        if self.use_image_context and image_features is not None:
            img_embedding = self.img_encoder(
                image_features).unsqueeze(1)  # (B, 1, hidden_dim)
            sequence_for_transformer = torch.cat(
                [img_embedding, state_embeddings], dim=1)  # (B, H+1, hidden_dim)

        # Through transformer
        transformer_output = self.transformer_encoder(sequence_for_transformer)

        # For classification, use the representation of the first token (CLS token approach)
        # If using image as first token, use that, otherwise use first state token
        first_token = transformer_output[:, 0, :]

        # Output head
        score = self.output_head(first_token)
        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, image_features=None) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        score = self.forward(trajectory_sequence, image_features)
        self.train()
        return score
