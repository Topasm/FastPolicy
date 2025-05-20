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
    dropout: float = 0.1  # Dropout rate
    use_layernorm: bool = True  # Use LayerNorm in output heads
    num_layers: int = 8  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    swiglu_intermediate_factor: int = 4  # Factor for SwiGLU intermediate dim
    # Half-horizon for split sequence prediction (if None, uses horizon // 2)
    half_horizon: int = None


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
        # Set half_horizon to half of horizon if not provided
        self.half_horizon = config.half_horizon if config.half_horizon is not None else self.horizon // 2
        self.use_image_context = True  # Default to using image context
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

    def forward(self, trajectory_sequence: torch.Tensor, raw_images: torch.Tensor = None, second_half: bool = False) -> torch.Tensor:
        """
        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            raw_images: Optional (B, C, H, W) tensor of raw images or (B, 2, C, H, W) for first and second half images.
            second_half: Whether this is the second half of the sequence (for positional embedding indexing)

        Returns:
            (B, 1) tensor of scores.
        """
        B, H, D = trajectory_sequence.shape

        # When we're doing next sequence prediction, we need to handle both full sequences
        # and half sequences (half_horizon length)
        # Only validate the state dimension (D) not the sequence length (H)
        if D != self.state_dim:
            raise ValueError(
                f"Input state dimension mismatch. Expected state dim {self.state_dim}, got {D}")

        # Process images - handle different image formats
        if raw_images is not None:
            # Process the image and get embedding
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

        # Add positional embeddings to state embeddings - adjust for second half
        if second_half:
            # For second half, use position embeddings from the second half of the full sequence
            # When processing half_horizon length sequences, this would be positions half_horizon to horizon-1
            pos_embeddings_to_use = self.pos_embedding[:,
                                                       self.half_horizon:self.half_horizon+trajectory_sequence.shape[1], :]
            state_embeddings = state_embeddings + pos_embeddings_to_use
        else:
            # For first half or full sequence, use the beginning of the position embeddings
            state_embeddings = state_embeddings + \
                self.pos_embedding[:, :trajectory_sequence.shape[1], :]

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

    def score(self, trajectory_sequence: torch.Tensor, raw_images: torch.Tensor = None, second_half: bool = False) -> torch.Tensor:
        """
        Scoring entrypoint. This will maintain gradients in training mode or disable them in eval mode.

        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            raw_images: Optional (B, C, H, W) tensor of raw images or (B, 2, C, H, W) for first and second half images.
            second_half: Whether this is the second half of the sequence (for positional embedding indexing)
        """
        score = self.forward(trajectory_sequence, raw_images, second_half)
        return score

    def score_next_sequence(self, first_half: torch.Tensor, second_half: torch.Tensor,
                            raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Score how likely the second half sequence is a correct continuation of the first half.

        Args:
            first_half: (B, H/2, D_state) tensor of first half state sequences
            second_half: (B, H/2, D_state) tensor of second half state sequences
            raw_images: (B, 2, C, H, W) tensor with images for first and second half
                        or (B, C, H, W) tensor with a single image used for both halves

        Returns:
            (B, 1) tensor of scores, higher means more likely to be correct continuation
        """
        # Note: This function must maintain gradients for training

        # Prepare images if we only have one image per sequence
        if raw_images is not None and raw_images.ndim == 4:
            # Duplicate the image for both first and second half
            raw_images = torch.stack(
                [raw_images, raw_images], dim=1)  # B, 2, C, H, W        # Get scores for first half using first image - raw_images should be (B, 2, C, H, W)
        if raw_images is not None:
            if raw_images.ndim != 5:
                raise ValueError(
                    f"Expected raw_images to be (B, 2, C, H, W), got shape {raw_images.shape}")
            first_image = raw_images[:, 0]  # (B, C, H, W)
            second_image = raw_images[:, 1]  # (B, C, H, W)
        else:
            first_image = None
            second_image = None

        # Forward pass with gradient tracking (no @torch.no_grad decorator)
        first_scores = self.forward(
            first_half, raw_images=first_image, second_half=False)
        second_scores = self.forward(
            second_half, raw_images=second_image, second_half=True)

        # Combine scores (average)
        combined_scores = (first_scores + second_scores) / 2

        return combined_scores

    def extract_images_from_batch(self, norm_batch):
        """
        Extract images for first and second half from the normalized batch.

        Args:
            norm_batch: Batch dictionary containing observation.image

        Returns:
            Tuple of (first_half_img, second_half_img) - each is (B, C, H, W)
        """
        if "observation.image" not in norm_batch:
            return None, None

        img_tensor = norm_batch["observation.image"]

        # Extract images for first and second halves
        if img_tensor.ndim == 5:  # B, T_img, C, H, W
            # Take images at timestep 0 and timestep half_horizon
            # Make sure we're within bounds
            first_idx = 0
            second_idx = min(self.half_horizon, img_tensor.shape[1]-1)

            first_half_img = img_tensor[:,
                                        first_idx].contiguous()  # B, C, H, W
            second_half_img = img_tensor[:,
                                         second_idx].contiguous()  # B, C, H, W
        elif img_tensor.ndim == 4:  # B, C, H, W - only single image
            # Use the same image for both halves
            first_half_img = img_tensor
            second_half_img = img_tensor
        else:
            raise ValueError(
                f"Unsupported image tensor shape: {img_tensor.shape}")

        return first_half_img, second_half_img

    def compute_binary_classification_loss(
        self,
        positive_trajectories: torch.Tensor = None,
        negative_trajectories: torch.Tensor = None,
        raw_images: torch.Tensor = None,
        noise_params: dict = None,
        norm_batch: dict = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes binary classification loss for next sequence prediction task.
        First half sequences (0-7) are used to predict if second half sequences (8-16) are correct continuations.
        Positive trajectories are labeled 1 (correct next sequence), negative trajectories are labeled 0 (incorrect).

        Uses only two image frames (at timesteps 0 and 8) to reduce memory usage.

        Args:
            positive_trajectories: (B_pos, H, D_state) tensor of positive state sequences (for legacy support).
            negative_trajectories: (B_neg, H, D_state) tensor of negative state sequences (for legacy support).
            raw_images: (B, 2, C, H, W) tensor of raw images for first (idx 0) and second half (idx 1).
            noise_params: Optional dictionary with noise parameters to apply to create negative samples.
            norm_batch: Normalized batch dictionary. If provided, extracts trajectories and images from it.

        Returns:
            tuple with (loss, accuracy)
        """
        # Extract trajectories and images from norm_batch if provided
        if norm_batch is not None:
            if "observation.state" in norm_batch:
                # Extract full trajectories from the normalized batch
                trajectories = norm_batch["observation.state"][:,
                                                               :self.horizon]

                # Split into first half and second half
                first_half = trajectories[:, :self.half_horizon]
                second_half = trajectories[:, self.half_horizon:self.horizon]

                # Create positive samples - true second half sequences
                positive_first_half = first_half.clone()
                positive_second_half = second_half.clone()

                # For negative samples, we'll use the same first half but noised second half
                negative_first_half = first_half.clone()
                negative_second_half = second_half.clone()

                # Get images for first and second half
                if raw_images is None:
                    # Use extract_images_from_batch to handle image processing
                    first_img, second_img = self.extract_images_from_batch(
                        norm_batch)
                    if first_img is not None and second_img is not None:
                        # Stack for score_next_sequence which expects [B, 2, C, H, W]
                        raw_images = torch.stack(
                            [first_img, second_img], dim=1)  # B, 2, C, H, W

            else:
                raise ValueError(
                    "norm_batch is provided but does not contain 'observation.state'")
        else:
            raise ValueError(
                "norm_batch must be provided for next sequence prediction")

        device = positive_first_half.device
        B = positive_first_half.shape[0]

        # Apply noise to negative second half trajectories if noise_params is provided
        if noise_params is not None:
            base_noise_scale = noise_params.get("base_noise_scale", 0.05)
            noise_type = noise_params.get("noise_type", "progressive")
            noise_growth_factor = noise_params.get("noise_growth_factor", 1.2)

            # Add noise to each timestep in the negative second half
            for t_idx in range(negative_second_half.shape[1]):
                relative_step = t_idx + 1  # Start from 1 for noise progression
                current_noise = base_noise_scale

                if noise_type == "progressive":
                    current_noise *= (noise_growth_factor **
                                      (relative_step - 1))
                elif noise_type == "diffusion":
                    timestep_fraction = relative_step / \
                        self.half_horizon if self.half_horizon > 1 else 0
                    current_noise *= (1.0 + 10 * timestep_fraction**2)

                noise = torch.randn_like(
                    negative_second_half[:, t_idx]) * current_noise
                negative_second_half[:, t_idx] += noise

        # Randomly shuffle some of the negative second halves for more variety
        if torch.rand(1).item() < 0.5:  # 50% chance to apply shuffling
            # 70% of batch gets shuffled
            shuffle_mask = torch.rand(B, device=device) < 0.7

            if shuffle_mask.any():
                shuffle_indices = torch.randperm(B, device=device)
                for i in range(B):
                    if shuffle_mask[i]:
                        # Get a different index to swap with
                        j = shuffle_indices[i]
                        if j == i:  # Avoid self-swap
                            j = (j + 1) % B
                        negative_second_half[i] = second_half[j].clone()

        # Compute model predictions
        if raw_images is not None:
            # Score the positive pairs (first half -> true second half)
            positive_scores = self.score_next_sequence(
                positive_first_half, positive_second_half, raw_images=raw_images)

            # Score the negative pairs (first half -> noised/shuffled second half)
            negative_scores = self.score_next_sequence(
                negative_first_half, negative_second_half, raw_images=raw_images)
        else:
            raise ValueError(
                "raw_images must be provided for next sequence prediction")

        # Combine positive and negative scores and create labels
        all_scores = torch.cat([positive_scores, negative_scores], dim=0)

        positive_labels = torch.ones_like(positive_scores, device=device)
        negative_labels = torch.zeros_like(negative_scores, device=device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Calculate loss and accuracy
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(all_scores, all_labels)

        predictions = (all_scores > 0.0).float()
        accuracy = (predictions == all_labels).float().mean()

        return loss, accuracy
