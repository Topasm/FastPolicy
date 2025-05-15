import torch
import torch.nn as nn
from dataclasses import dataclass, field
from lerobot.configs.types import NormalizationMode


@dataclass
class NoiseCriticConfig:
    """Configuration for the noise trajectory critic model."""
    state_dim: int = 2  # Dimension of state vectors
    horizon: int = 16  # Length of state sequence to evaluate
    hidden_dim: int = 512  # Hidden dimension of the network
    num_layers: int = 4  # Number of transformer layers or MLP blocks
    dropout: float = 0.1
    use_layernorm: bool = True
    # Options: "mlp", "transformer", "gru", "dv_horizon"
    architecture: str = "transformer"
    use_image_context: bool = True  # Always use image features as context
    image_feature_dim: int = 512  # Dimension of image features (required)
    # Fields needed for DiffusionRgbEncoder
    image_features: dict = field(default_factory=lambda: {
                                 "observation.image": (3, 84, 84)})  # Image features dict
    transformer_dim: int = 512  # Dimension for the transformer
    # Additional parameters for DVHorizonCritic
    n_heads: int = 8  # Number of attention heads
    # Norm type for DVTransformerBlock ("pre" or "post")
    norm_type: str = "post"
    observation_delta_indices: list = field(default_factory=lambda: [0])
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )

    # Parameters for DiffusionRgbEncoder
    vision_backbone: str = "resnet18"
    crop_shape: tuple[int, int] | None = (84, 84)
    crop_is_random: bool = True
    pretrained_backbone_weights: str | None = None
    use_group_norm: bool = True
    spatial_softmax_num_keypoints: int = 32
    use_separate_rgb_encoder_per_camera: bool = False


class TransformerCritic(nn.Module):
    """
    Transformer-based critic that scores a sequence of states with required image context.
    Input: 
        - trajectory_sequence: (B, H, D_state) tensor of state sequences
        - image_features: (B, D_img) tensor of image features OR
        - raw_images: Raw images that will be processed with a ViT-like approach
    Output: (B, 1) score per sequence
    """

    def __init__(self, config: NoiseCriticConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.image_features = config.image_features

        # Image parameters for ViT-like processing
        self.use_vit_patching = True  # Use ViT-like image patching
        self.patch_size = 16  # Default patch size for ViT
        self.img_size = 96    # PushT dataset uses 96x96 images
        # 36 patches for 96x96 with 16x16 patches
        self.num_patches = (self.img_size // self.patch_size) ** 2

        # The expected input channels (assumed RGB)
        self.in_channels = 3

        # Always use image context - no longer optional
        config.use_image_context = True
        self.use_image_context = True

        # State embedding layer (to transformer hidden dim)
        self.state_embedding = nn.Linear(config.state_dim, config.hidden_dim)

        # Positional encoding
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, config.horizon, config.hidden_dim))

        # Image patching and embedding (ViT-like approach)
        # Projects patches to hidden_dim
        self.patch_embedding = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ),
            nn.Flatten(2),
        )
        # torch.nn doesn't have a Transpose module, so we'll handle this in the forward pass

        # Position embedding for image patches
        self.img_pos_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches, self.hidden_dim))

        # CLS token for image (optional but common in ViT)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim))

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

        # Legacy image feature processing (when pre-encoded features are provided)
        self.img_encoder = nn.Sequential(
            nn.Linear(config.image_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(
                config.hidden_dim) if config.use_layernorm else nn.Identity(),
            nn.Dropout(config.dropout)
        )

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
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                # Use Xavier initialization for transformers
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize positional embeddings
        nn.init.normal_(self.pos_embedding, std=0.02)
        nn.init.normal_(self.img_pos_embedding, std=0.02)
        nn.init.normal_(self.cls_token, std=0.02)

    def process_raw_images(self, images):
        """Process raw images with ViT-like patching approach."""
        # images: (B, C, H, W)
        B = images.shape[0]

        # Resize images if they don't match the expected size
        if images.shape[2] != self.img_size or images.shape[3] != self.img_size:
            import torch.nn.functional as F
            images = F.interpolate(images, size=(
                self.img_size, self.img_size), mode='bilinear')

        # Patch embedding and transpose manually
        patch_embs = self.patch_embedding(
            images)  # (B, hidden_dim, num_patches)
        patch_embeddings = patch_embs.transpose(
            1, 2)  # (B, num_patches, hidden_dim)

        # Add position embedding
        patch_embeddings = patch_embeddings + self.img_pos_embedding

        # Add CLS token (will be used as the image representation)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        patch_embeddings_with_cls = torch.cat(
            [cls_tokens, patch_embeddings], dim=1)

        # Pass through transformer encoder
        img_features = self.transformer_encoder(patch_embeddings_with_cls)

        # Use the CLS token as the image representation
        return img_features[:, 0]  # (B, hidden_dim)

    def forward(self, trajectory_sequence: torch.Tensor, image_features: torch.Tensor = None, raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            image_features: Optional (B, D_img) tensor of pre-encoded image features.
            raw_images: Optional (B, C, H, W) tensor of raw images. 
                        Either image_features or raw_images must be provided.
        Returns:
            (B, 1) tensor of scores (logits).
        """
        B, H, D = trajectory_sequence.shape
        if H != self.horizon or D != self.state_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (B, {self.horizon}, {self.state_dim}), got {(B, H, D)}")

        # Process images if raw_images is provided
        if raw_images is not None:
            img_embedding = self.process_raw_images(
                raw_images).unsqueeze(1)  # (B, 1, hidden_dim)
        elif image_features is not None:
            # Legacy mode: use pre-encoded image features
            img_embedding = self.img_encoder(
                image_features).unsqueeze(1)  # (B, 1, hidden_dim)
        else:
            raise ValueError(
                "Either image_features or raw_images must be provided")

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

        # For classification, use the representation of the first token (CLS token approach)
        # If using image as first token, use that, otherwise use first state token
        first_token = transformer_output[:, 0, :]

        # Output head
        score = self.output_head(first_token)
        return score

    @torch.no_grad()
    def score(self, trajectory_sequence: torch.Tensor, image_features: torch.Tensor = None, raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Inference entrypoint.

        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            image_features: Optional (B, D_img) tensor of pre-encoded image features.
            raw_images: Optional (B, C, H, W) tensor of raw images.
                       Either image_features or raw_images must be provided.
        """
        self.eval()
        score = self.forward(trajectory_sequence, image_features, raw_images)
        self.train()
        return score

    def compute_binary_classification_loss(
        self,
        norm_batch: dict,
        noise_params: dict = None,
        image_features: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the binary classification loss for training the critic.
        Automatically extracts trajectories from normalized batch and creates positive and negative trajectories.

        Positive trajectories are labeled 1, negative trajectories are labeled 0.

        Args:
            norm_batch: Dictionary containing normalized batch data. Must have "observation.state" key.
                       If image_features is None, must also contain raw image data.
            noise_params: Dictionary with noise parameters:
                - base_noise_scale: Base scale of noise
                - noise_type: Type of noise ("progressive", "diffusion", "uniform")
                - noise_growth_factor: Factor for progressive noise
                If None, uses default parameters.
            image_features: Optional pre-encoded image features. If None, uses raw images from norm_batch.

        Returns:
            A tuple containing:
                - loss: Scalar tensor representing the BCEWithLogitsLoss.
                - accuracy: Scalar tensor representing the classification accuracy.
        """
        # Extract trajectories from the normalized batch
        trajectories = norm_batch["observation.state"][:, 0:self.horizon]
        device = trajectories.device
        B, H, D_state = trajectories.shape

        # Process image features if not provided - extract raw images from norm_batch
        raw_images = None
        if image_features is None:
            # Try to get raw images from norm_batch
            for key in self.image_features.keys():  # e.g. "observation.image"
                if key in norm_batch:
                    img_tensor = norm_batch[key]  # Already on device
                    # Handle different image tensor shapes
                    if img_tensor.ndim == 5:  # B, T_img, C, H, W
                        # Take the last image in sequence
                        raw_images = img_tensor[:, -1]  # B, C, H, W
                    elif img_tensor.ndim == 4:  # B, C, H, W
                        raw_images = img_tensor
                    else:
                        raise ValueError(
                            f"Unexpected image tensor dimensions for key '{key}': {img_tensor.shape}")
                    break

            if raw_images is None:
                raise ValueError(
                    "No image data found in norm_batch and no image_features provided.")

        # Set default noise parameters if not provided
        if noise_params is None:
            noise_params = {
                "base_noise_scale": 0.05,
                "noise_type": "progressive",
                "noise_growth_factor": 1.2,
            }

        base_noise_scale = noise_params.get("base_noise_scale", 0.05)
        noise_type = noise_params.get("noise_type", "progressive")
        noise_growth_factor = noise_params.get("noise_growth_factor", 1.2)

        # --- Create Positive and Negative Trajectories ---
        positive_trajectories = trajectories.clone()

        # Optional: Augment positive trajectories slightly
        if torch.rand(1).item() < 0.2:  # 20% chance
            tiny_noise_scale = base_noise_scale * 0.1
            for t_step in range(1, H):  # Don't noise the initial state
                tiny_noise = torch.randn_like(
                    positive_trajectories[:, t_step]) * tiny_noise_scale
                positive_trajectories[:, t_step] += tiny_noise

        negative_trajectories = trajectories.clone()

        # Apply shuffling to a subset of negative examples
        shuffle_mask = torch.rand(B, device=device) < 0.3  # 30% of batch
        if shuffle_mask.any():
            num_to_shuffle = shuffle_mask.sum().item()
            shuffle_indices = torch.randperm(B, device=device)[:num_to_shuffle]
            mask_indices = torch.where(shuffle_mask)[0]
            for i, idx in enumerate(mask_indices):
                # Use num_to_shuffle for modulo
                random_idx = shuffle_indices[i % num_to_shuffle]
                if random_idx == idx:
                    random_idx = (random_idx + 1) % B
                negative_trajectories[idx] = trajectories[random_idx]

        # Apply noise to negative trajectories
        # Note: Noise is applied *after* potential shuffling
        for t_step in range(1, H):  # Don't noise the initial state
            current_noise_this_step = base_noise_scale
            if noise_type == "progressive":
                current_noise_this_step *= (noise_growth_factor **
                                            (t_step - 1))
            elif noise_type == "diffusion":
                timestep_fraction = t_step / (H - 1) if H > 1 else 0
                current_noise_this_step *= (1.0 + 10 * timestep_fraction**2)
            # 'uniform' uses base_noise_scale directly

            noise = torch.randn_like(
                negative_trajectories[:, t_step]) * current_noise_this_step
            negative_trajectories[:, t_step] += noise

        # Apply temporal shifts to a subset of negative examples
        if torch.rand(1).item() < 0.3:  # 30% chance
            shift_mask = torch.rand(B, device=device) < 0.5  # 50% of the 30%
            if shift_mask.any():
                mask_indices = torch.where(shift_mask)[0]
                max_shift = max(1, H // 4)
                for idx in mask_indices:
                    shift = torch.randint(
                        1, max_shift + 1, (1,), device=device).item()
                    shift = min(shift, H - 1)

                    original_traj_for_shift = negative_trajectories[idx].clone(
                    )
                    negative_trajectories[idx, :-
                                          shift] = original_traj_for_shift[shift:]
                    # Use the actual last state before shift
                    last_valid_state = original_traj_for_shift[H-1]
                    negative_trajectories[idx, -
                                          shift:] = last_valid_state.unsqueeze(0).repeat(shift, 1)

        # Calculate scores for positive and negative trajectories
        if raw_images is not None:
            # Use raw_images directly
            positive_scores = self.forward(
                positive_trajectories, raw_images=raw_images)
            negative_scores = self.forward(
                negative_trajectories, raw_images=raw_images)
        else:
            # Use pre-encoded image_features
            positive_scores = self.forward(
                positive_trajectories, image_features=image_features)
            negative_scores = self.forward(
                negative_trajectories, image_features=image_features)

        all_scores = torch.cat([positive_scores, negative_scores], dim=0)

        positive_labels = torch.ones_like(positive_scores, device=device)
        negative_labels = torch.zeros_like(negative_scores, device=device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(all_scores, all_labels)

        predictions = (all_scores > 0.0).float()
        accuracy = (predictions == all_labels).float().mean()

        return loss, accuracy

    def compute_binary_classification_loss_with_provided_trajectories(self, positive_trajectories: torch.Tensor, negative_trajectories: torch.Tensor, image_features: torch.Tensor, image_features_for_negative: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Legacy method that computes the binary classification loss using provided positive and negative trajectories.
        Positive trajectories are labeled 1, negative trajectories are labeled 0.

        Args:
            positive_trajectories: (B_pos, H, D_state) tensor of positive state sequences.
            negative_trajectories: (B_neg, H, D_state) tensor of negative state sequences.
            image_features: (B_pos, D_img) tensor of image features for positive trajectories.
            image_features_for_negative: (B_neg, D_img) tensor of image features for negative trajectories.
                                         If None, uses image_features for both.

        Returns:
            A tuple containing:
                - loss: Scalar tensor representing the BCEWithLogitsLoss.
                - accuracy: Scalar tensor representing the classification accuracy.
        """
        positive_scores = self.forward(positive_trajectories, image_features)

        if image_features_for_negative is None:
            # If negative trajectories share image features (e.g. same context, different futures)
            # and batch sizes match. This is common if image_features are for the "current" observation.
            if positive_trajectories.shape[0] == negative_trajectories.shape[0]:
                img_feat_neg = image_features
            else:
                # If batch sizes don't match and no specific negative image features are given, this is ambiguous.
                # Fallback: repeat the first positive image feature. This might not be ideal.
                # Consider requiring image_features_for_negative if B_pos != B_neg.
                img_feat_neg = image_features.expand(
                    negative_trajectories.shape[0], -1)

        else:
            img_feat_neg = image_features_for_negative

        negative_scores = self.forward(negative_trajectories, img_feat_neg)

        all_scores = torch.cat([positive_scores, negative_scores], dim=0)

        positive_labels = torch.ones_like(
            positive_scores, device=positive_scores.device)
        negative_labels = torch.zeros_like(
            negative_scores, device=negative_scores.device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(all_scores, all_labels)

        predictions = (all_scores > 0.0).float()
        accuracy = (predictions == all_labels).float().mean()

        return loss, accuracy


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
