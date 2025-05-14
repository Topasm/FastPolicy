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
    architecture: str = "mlp"  # Options: "mlp", "transformer", "gru", "dv_horizon"
    use_image_context: bool = False  # Whether to use image features as context
    image_feature_dim: int = 0  # Dimension of image features (if used)
    # Additional parameters for DVHorizonCritic
    n_heads: int = 8  # Number of attention heads
    # Norm type for DVTransformerBlock ("pre" or "post")
    norm_type: str = "post"


class MLPCritic(nn.Module):
    """
    MLP-based critic that scores a sequence of states.
    Input: (B, H, D_state)
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: NoiseCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = config.use_image_context

        # Input dimensions
        input_dim = config.state_dim * config.horizon

        # Add image feature dimension if using image context
        if self.use_image_context:
            self.img_encoder = nn.Sequential(
                nn.Linear(config.image_feature_dim, config.hidden_dim),
                nn.GELU(),
                nn.LayerNorm(
                    config.hidden_dim) if config.use_layernorm else nn.Identity(),
                nn.Dropout(config.dropout),
                nn.Linear(config.hidden_dim, config.hidden_dim // 2)
            )
            input_dim += config.hidden_dim // 2

        # Build network
        layers = []
        current_dim = input_dim

        # Hidden layers
        for _ in range(config.num_layers - 1):
            layers.append(nn.Linear(current_dim, config.hidden_dim))
            if config.use_layernorm:
                layers.append(nn.LayerNorm(config.hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config.dropout))
            current_dim = config.hidden_dim

        # Output layer (no activation for logits)
        layers.append(nn.Linear(current_dim, 1))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

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

        # Flatten sequence
        state_flat = trajectory_sequence.view(B, -1)

        # Process image features if provided
        if self.use_image_context and image_features is not None:
            img_encoded = self.img_encoder(image_features)
            # Concatenate flattened state sequence with image features
            net_input = torch.cat([state_flat, img_encoded], dim=1)
        else:
            net_input = state_flat

        # Pass through network
        score = self.net(net_input.float())
        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, image_features=None) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        score = self.forward(trajectory_sequence, image_features)
        self.train()
        return score


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


class GRUCritic(nn.Module):
    """
    GRU-based critic that scores a sequence of states.
    Input: (B, H, D_state)
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: NoiseCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = config.use_image_context

        # GRU layer
        self.gru = nn.GRU(
            input_size=config.state_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0
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

        # Output head
        output_input_dim = config.hidden_dim
        if self.use_image_context:
            output_input_dim += config.hidden_dim  # Add image feature dimension

        self.output_head = nn.Sequential(
            nn.Linear(output_input_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(
                config.hidden_dim) if config.use_layernorm else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

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

        # Process sequence through GRU
        # h_n shape: (num_layers, B, hidden_dim)
        _, h_n = self.gru(trajectory_sequence)

        # Use the final hidden state from the last layer
        final_hidden = h_n[-1]  # (B, hidden_dim)

        # Process image features if provided
        if self.use_image_context and image_features is not None:
            img_encoded = self.img_encoder(image_features)  # (B, hidden_dim)
            # Concatenate with GRU output
            # (B, 2*hidden_dim)
            combined = torch.cat([final_hidden, img_encoded], dim=1)
        else:
            combined = final_hidden

        # Final score
        score = self.output_head(combined)
        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, image_features=None) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        score = self.forward(trajectory_sequence, image_features)
        self.train()
        return score


class SinusoidalEmbedding(nn.Module):
    """
    Sinusoidal Positional Embedding module.
    Creates position-dependent sinusoidal patterns for encoding sequence position information.
    """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        embeddings = torch.zeros(x.shape[0], self.dim, device=device)
        div_term = torch.exp(torch.arange(0, half_dim, 2, device=device).float() *
                             -(torch.log(torch.tensor(10000.0)) / half_dim))
        position = x.unsqueeze(1)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        if self.dim % 2 != 0:  # For odd dimensions
            embeddings[:, 1::2] = torch.cos(
                position * div_term)[:, 0:embeddings[:, 1::2].shape[1]]
        else:
            embeddings[:, 1::2] = torch.cos(position * div_term)
        return embeddings


class DVTransformerBlock(nn.Module):
    """
    Transformer block used in DVHorizonCritic.
    Implements both pre-layer normalization and post-layer normalization architectures.
    """

    def __init__(self, hidden_size: int, n_heads: int, dropout: float = 0.0, norm_type="post"):
        super().__init__()
        self.norm_type = norm_type

        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, n_heads, dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)

        def approx_gelu(): return nn.GELU(approximate="tanh")

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            approx_gelu(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size)
        )

    def forward(self, x: torch.Tensor):
        if self.norm_type == "post":
            # Post-LN: Sublayer -> Add -> Norm
            attn_output, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_output)
            x = self.norm2(x + self.mlp(x))
        elif self.norm_type == "pre":
            # Pre-LN: Norm -> Sublayer -> Add
            # Self-Attention part
            normed_x = self.norm1(x)
            attn_output, _ = self.attn(normed_x, normed_x, normed_x)
            x = x + attn_output
            # MLP part
            normed_x = self.norm2(x)
            mlp_output = self.mlp(normed_x)
            x = x + mlp_output
        else:
            raise NotImplementedError(
                f"norm_type {self.norm_type} not implemented.")
        return x


class DVHorizonCritic(nn.Module):
    """
    DVHorizonCritic: An advanced transformer-based critic for scoring state sequences.
    Uses sinusoidal positional embeddings and custom transformer blocks.

    Input: (B, H, D_state)
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: NoiseCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = config.use_image_context

        # Create model with parameters from config
        d_model = config.hidden_dim
        # Default to 8 if not specified
        n_heads = getattr(config, 'n_heads', 8)
        depth = config.num_layers
        dropout = config.dropout
        # Default to 'post' if not specified
        norm_type = getattr(config, 'norm_type', 'post')

        # Input projection
        self.x_proj = nn.Linear(config.state_dim, d_model)

        # Positional encoding
        self.pos_emb = SinusoidalEmbedding(d_model)
        self.pos_emb_cache = None  # To cache positional embeddings

        # Transformer blocks
        self.blocks = nn.ModuleList([
            DVTransformerBlock(d_model, n_heads, dropout, norm_type) for _ in range(depth)
        ])

        # Image context processing if needed
        if self.use_image_context:
            self.img_encoder = nn.Sequential(
                nn.Linear(config.image_feature_dim, d_model),
                nn.GELU(),
                nn.LayerNorm(
                    d_model) if config.use_layernorm else nn.Identity(),
                nn.Dropout(dropout)
            )

        # Final output layer
        self.final_layer = nn.Linear(d_model, 1)

        self.initialize_weights()

    def initialize_weights(self):
        """Initialize weights for better training stability."""
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

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

        # Cache positional embeddings if needed
        if self.pos_emb_cache is None or self.pos_emb_cache.shape[0] != H:
            positions = torch.arange(H, device=trajectory_sequence.device)
            self.pos_emb_cache = self.pos_emb(positions)

        # Project input features
        projected_x = self.x_proj(trajectory_sequence)  # (B, H, d_model)

        # Add positional embeddings
        x_with_pos = projected_x + self.pos_emb_cache

        # Process image features if provided
        if self.use_image_context and image_features is not None:
            img_emb = self.img_encoder(
                image_features).unsqueeze(1)  # (B, 1, d_model)
            # Prepend image token
            x_with_pos = torch.cat(
                [img_emb, x_with_pos], dim=1)  # (B, H+1, d_model)

        # Pass through transformer blocks
        processed_x = x_with_pos
        for block in self.blocks:
            processed_x = block(processed_x)

        # Get the first token's representation for the final score
        # If using image context, this will be the image token
        first_token = processed_x[:, 0, :]

        # Apply final layer
        score = self.final_layer(first_token)

        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, image_features=None) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        score = self.forward(trajectory_sequence, image_features)
        self.train()
        return score


def create_noise_critic(config: NoiseCriticConfig):
    """
    Factory function to create the appropriate critic model based on config.
    """
    if config.architecture.lower() == "mlp":
        return MLPCritic(config)
    elif config.architecture.lower() == "transformer":
        return TransformerCritic(config)
    elif config.architecture.lower() == "gru":
        return GRUCritic(config)
    elif config.architecture.lower() == "dv_horizon":
        return DVHorizonCritic(config)
    else:
        raise ValueError(
            f"Unknown architecture: {config.architecture}. Choose from 'mlp', 'transformer', 'gru', or 'dv_horizon'.")
