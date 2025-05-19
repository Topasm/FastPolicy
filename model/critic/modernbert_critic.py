import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class ModernBertCriticConfig:
    """Configuration for ModernBertCritic - inspired by ModernBERT architecture"""
    state_dim: int  # Dimension of state
    horizon: int    # Sequence length (time steps) for state trajectory
    hidden_dim: int = 768  # Hidden dimension for transformer layers
    # Dimension of image features (if pre-encoded)
    image_feature_dim: int = 2048
    dropout: float = 0.1  # Dropout rate
    use_layernorm: bool = True  # Use LayerNorm in output heads
    num_layers: int = 8  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    use_image_context: bool = True  # Whether to use image context
    swiglu_intermediate_factor: int = 4  # Factor for SwiGLU intermediate dim


# --- SwiGLU Activation ---
class SwiGLU(nn.Module):
    """ SwiGLU Activation Function - Modern Transformer architecture component """

    def __init__(self, dim: int, hidden_dim: int | None = None, bias: bool = False):
        super().__init__()
        if hidden_dim is None:
            # Default expansion factor for SwiGLU
            hidden_dim = int(dim * 4 * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w3 = nn.Linear(hidden_dim, dim, bias=bias)

    def forward(self, x):
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


# --- Simple Attention Pooling ---
class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = False):
        super().__init__()
        self.query = nn.Parameter(torch.randn(
            1, 1, output_dim))  # Learnable query
        self.key_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.value_proj = nn.Linear(input_dim, output_dim, bias=bias)
        self.scale = output_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, Seq, Dim)
            mask: Attention mask (B, Seq), True where tokens should be ignored.
        Returns:
            Pooled output (B, Dim_out)
        """
        B, _, _ = x.shape
        k = self.key_proj(x)    # (B, Seq, Dim_out)
        v = self.value_proj(x)  # (B, Seq, Dim_out)
        q = self.query.expand(B, -1, -1)  # (B, 1, Dim_out)

        # Attention scores (B, 1, Seq)
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            # Add a large negative number to masked positions
            attn_scores = attn_scores.masked_fill(
                mask.unsqueeze(1), float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, 1, Seq)
        # Weighted sum of values (B, 1, Dim_out)
        pooled_output = attn_weights @ v

        return pooled_output.squeeze(1)  # (B, Dim_out)


class ModernBertCritic(nn.Module):
    """
    ModernBERT-style Transformer critic that scores a sequence of states with image context.
    Implements modern transformer techniques from recent research:
    - SwiGLU activations instead of traditional GELU
    - Pre-normalization architecture (norm_first=True)
    - Attention pooling for feature aggregation
    - Proper position embeddings
    - No bias in linear projections (following modern BERT practices)

    Input: 
        - trajectory_sequence: (B, H, D_state) tensor of state sequences
        - raw_images: Raw images that will be processed with a ViT-like approach
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: ModernBertCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        use_bias = False  # ModernBERT: No bias in linear layers

        # Image parameters for ViT-like processing
        self.use_vit_patching = True
        self.patch_size = 16
        self.img_size = 96
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.in_channels = 3

        # State embedding layer (to transformer hidden dim)
        self.state_embedding = nn.Linear(
            config.state_dim, config.hidden_dim, bias=use_bias)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.horizon, config.hidden_dim))

        # Image patching and embedding (ViT-like approach)
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=config.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
                bias=use_bias
            ),
            nn.Flatten(2),
        )

        # Position embedding for image patches
        self.img_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, config.hidden_dim))

        # CLS token for image (following ViT approach)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))

        # Calculate SwiGLU hidden dimension
        swiglu_hidden_dim = config.hidden_dim * config.swiglu_intermediate_factor

        # Legacy image feature processing (for pre-encoded features)
        self.img_encoder = nn.Sequential(
            nn.Linear(config.image_feature_dim,
                      config.hidden_dim, bias=use_bias),
            nn.GELU(),  # Keep GELU for compatibility with older code
            nn.LayerNorm(
                config.hidden_dim) if config.use_layernorm else nn.Identity(),
            nn.Dropout(config.dropout)
        )

        # Transformer encoder with pre-normalization and modern design
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=swiglu_hidden_dim,
            dropout=config.dropout,
            activation='gelu',  # Keep GELU for compatibility with PyTorch's transformer
            batch_first=True,
            norm_first=True  # Pre-LN architecture for better training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers
        )

        # Attention pooling instead of simple averaging or first token extraction
        self.attention_pooler = AttentionPooling(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            bias=use_bias
        )

        # Output head with SwiGLU
        self.output_head = nn.Sequential(
            SwiGLU(config.hidden_dim, hidden_dim=swiglu_hidden_dim, bias=use_bias),
            nn.LayerNorm(
                config.hidden_dim) if config.use_layernorm else nn.Identity(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1, bias=use_bias)
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using modern initialization methods"""
        # Normal initialization for embeddings and parameters with std=0.02
        for param in [self.pos_embedding, self.img_pos_embedding, self.cls_token]:
            nn.init.normal_(param, mean=0.0, std=0.02)

        # Xavier uniform for linear and conv layers
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)

    def process_raw_images(self, images):
        """Process raw images with ViT-like patching approach."""
        # images: (B, C, H, W)
        B = images.shape[0]

        # Resize images if they don't match the expected size
        if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
            images = F.interpolate(images, size=(
                self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Patch embedding and transpose manually
        patch_embs = self.patch_embedding(
            images)  # (B, hidden_dim, num_patches)
        patch_embeddings = patch_embs.transpose(
            1, 2)  # (B, num_patches, hidden_dim)

        # Add position embedding
        patch_embeddings = patch_embeddings + self.img_pos_embedding

        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patch_embeddings_with_cls = torch.cat(
            [cls_tokens, patch_embeddings], dim=1)

        # Pass through transformer encoder
        img_features = self.transformer_encoder(patch_embeddings_with_cls)

        # Use attention pooling instead of just CLS token
        # Create mask where all positions except CLS are ignored
        cls_only_mask = torch.ones(
            B, self.num_patches + 1, dtype=torch.bool, device=images.device)
        cls_only_mask[:, 0] = False  # Don't mask the CLS token

        # Return attention-pooled representation
        return self.attention_pooler(img_features, mask=cls_only_mask)

    def forward(self, trajectory_sequence: torch.Tensor, raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            raw_images: Optional (B, C, H, W) tensor of raw images. 

        Returns:
            (B, 1) tensor of scores.
        """
        B, H, D = trajectory_sequence.shape
        if H != self.horizon or D != self.state_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (B, {self.horizon}, {self.state_dim}), got {(B, H, D)}")

        # Process images
        if raw_images is not None:
            img_embedding = self.process_raw_images(raw_images).unsqueeze(1)

        elif not self.use_image_context:
            # If image context is not used, create a learnable dummy token instead
            img_embedding = self.cls_token.expand(B, 1, -1)
        else:
            raise ValueError(
                "Either image_features or raw_images must be provided when use_image_context=True")

        # Embed states to hidden dimension
        state_embeddings = self.state_embedding(
            trajectory_sequence)  # (B, H, hidden_dim)

        # Add positional embeddings to state embeddings
        state_embeddings = state_embeddings + self.pos_embedding

        # Combine image embedding with state embeddings
        sequence_for_transformer = torch.cat(
            [img_embedding, state_embeddings], dim=1)  # (B, H+1, hidden_dim)

        # Through transformer
        transformer_output = self.transformer_encoder(sequence_for_transformer)

        # Use attention pooling instead of just taking first token
        # Create a mask that ignores padding if any (all real tokens are valid here)
        attention_mask = None  # No masking by default as we have no padding

        # Get pooled representation with attention
        pooled_output = self.attention_pooler(
            transformer_output, mask=attention_mask)

        # Output head
        score = self.output_head(pooled_output)
        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Inference entrypoint.

        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            raw_images: Optional (B, C, H, W) tensor of raw images.
        """
        self.eval()
        score = self.forward(trajectory_sequence, raw_images)
        return score

    def compute_binary_classification_loss(
        self,
        positive_trajectories: torch.Tensor = None,
        negative_trajectories: torch.Tensor = None,
        raw_images: torch.Tensor = None,
        noise_params: dict = None,
        norm_batch: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes binary classification loss with provided positive and negative trajectories.
        Positive trajectories are labeled 1, negative trajectories are labeled 0.

        Args:
            positive_trajectories: (B_pos, H, D_state) tensor of positive state sequences.
            negative_trajectories: (B_neg, H, D_state) tensor of negative state sequences.
            raw_images: (B, C, H, W) tensor of raw images.
            noise_params: Optional dictionary with noise parameters to apply:
                - base_noise_scale: Base scale of noise
                - noise_type: Type of noise ("progressive", "diffusion", "uniform")
                - noise_growth_factor: Factor for progressive noise
            norm_batch: Optional normalized batch dictionary. If provided, extracts trajectories and images from it.

        Returns:
            tuple with (loss, accuracy)
        """
        # Extract trajectories and images from norm_batch if provided
        if norm_batch is not None:
            if "observation.state" in norm_batch:
                # Extract trajectories from the normalized batch
                trajectories = norm_batch["observation.state"][:,
                                                               :self.horizon]

                # Use trajectories as both positive and negative (will add noise to negative)
                positive_trajectories = trajectories.clone()
                negative_trajectories = trajectories.clone()

                # Get raw images from norm_batch
                if raw_images is None and "observation.image" in norm_batch:
                    img_tensor = norm_batch["observation.image"]
                    # Handle different image tensor shapes
                    if img_tensor.ndim == 5:  # B, T_img, C, H, W
                        # Take the first image in sequence
                        raw_images = img_tensor[:, 0]  # B, C, H, W
                    elif img_tensor.ndim == 4:  # B, C, H, W
                        raw_images = img_tensor
            else:
                raise ValueError(
                    "norm_batch is provided but does not contain 'observation.state'")
        elif positive_trajectories is None or negative_trajectories is None:
            raise ValueError(
                "Either norm_batch or both positive_trajectories and negative_trajectories must be provided")

        device = positive_trajectories.device

        # Apply noise to negative trajectories if noise_params is provided
        if noise_params is not None:
            base_noise_scale = noise_params.get("base_noise_scale", 0.05)
            noise_type = noise_params.get("noise_type", "progressive")
            noise_growth_factor = noise_params.get("noise_growth_factor", 1.2)

            # Clone to ensure we don't modify the original
            negative_trajectories = negative_trajectories.clone()

            # Add progressive noise to negative trajectories (skip the first step)
            horizon = negative_trajectories.shape[1]
            for t_step in range(1, horizon):
                current_noise = base_noise_scale

                if noise_type == "progressive":
                    current_noise *= (noise_growth_factor ** (t_step - 1))
                elif noise_type == "diffusion":
                    timestep_fraction = t_step / \
                        (horizon - 1) if horizon > 1 else 0
                    current_noise *= (1.0 + 10 * timestep_fraction**2)

                noise = torch.randn_like(
                    negative_trajectories[:, t_step]) * current_noise
                negative_trajectories[:, t_step] += noise

        if raw_images is not None:
            positive_scores = self.forward(
                positive_trajectories, raw_images=raw_images)
            negative_scores = self.forward(
                negative_trajectories, raw_images=raw_images)
        else:
            raise ValueError(
                "Either raw_images must be provided")

        all_scores = torch.cat([positive_scores, negative_scores], dim=0)

        positive_labels = torch.ones_like(positive_scores, device=device)
        negative_labels = torch.zeros_like(negative_scores, device=device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(all_scores, all_labels)

        predictions = (all_scores > 0.0).float()
        accuracy = (predictions == all_labels).float().mean()

        return loss, accuracy
