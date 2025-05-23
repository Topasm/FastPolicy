import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
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
    # Future prediction parameters
    predict_future: bool = False  # Enable future prediction mode
    future_steps: int = 8  # Number of steps to predict into the future
    predict_future_image: bool = True  # Whether to predict future images
    predict_future_state: bool = True  # Whether to predict future states
    multi_step_prediction: bool = False  # Whether to predict multiple future steps
    # Number of future steps to predict if multi_step_prediction is True
    num_future_steps: int = 1
    predict_uncertainty: bool = False  # Whether to predict uncertainty in predictions


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

        # Add future prediction components
        if hasattr(config, "predict_future") and config.predict_future:
            self.predict_future = config.predict_future
            self.future_steps = config.future_steps
            self.predict_future_image = getattr(
                config, "predict_future_image", True)
            self.predict_future_state = getattr(
                config, "predict_future_state", True)
            self.multi_step_prediction = getattr(
                config, "multi_step_prediction", False)
            self.num_future_steps = getattr(
                config, "num_future_steps", 1)
            self.predict_uncertainty = getattr(
                config, "predict_uncertainty", False)

            # For future state prediction
            if self.predict_future_state:
                state_pred_hidden = config.hidden_dim // 2

                if self.multi_step_prediction:
                    # For multi-step prediction, output num_future_steps states
                    self.future_state_decoder = nn.Sequential(
                        SwiGLU(config.hidden_dim,
                               hidden_dim=swiglu_hidden_dim, bias=use_bias),
                        nn.LayerNorm(
                            config.hidden_dim) if config.use_layernorm else nn.Identity(),
                        nn.Linear(config.hidden_dim,
                                  state_pred_hidden, bias=use_bias),
                        nn.ReLU(),
                        nn.Linear(state_pred_hidden,
                                  config.state_dim * self.num_future_steps, bias=use_bias)
                    )
                else:
                    # Standard single-step prediction
                    self.future_state_decoder = nn.Sequential(
                        SwiGLU(config.hidden_dim,
                               hidden_dim=swiglu_hidden_dim, bias=use_bias),
                        nn.LayerNorm(
                            config.hidden_dim) if config.use_layernorm else nn.Identity(),
                        nn.Linear(config.hidden_dim,
                                  state_pred_hidden, bias=use_bias),
                        nn.ReLU(),
                        nn.Linear(state_pred_hidden,
                                  config.state_dim, bias=use_bias)
                    )

                # Add uncertainty estimation head if enabled
                if self.predict_uncertainty:
                    if self.multi_step_prediction:
                        # Predict uncertainty for each state dimension at each future step
                        self.state_uncertainty_head = nn.Sequential(
                            SwiGLU(config.hidden_dim,
                                   hidden_dim=swiglu_hidden_dim, bias=use_bias),
                            nn.LayerNorm(
                                config.hidden_dim) if config.use_layernorm else nn.Identity(),
                            nn.Linear(config.hidden_dim,
                                      state_pred_hidden, bias=use_bias),
                            nn.ReLU(),
                            nn.Linear(state_pred_hidden,
                                      config.state_dim * self.num_future_steps, bias=use_bias),
                            nn.Softplus()  # Ensure positive values for variance
                        )
                    else:
                        # Predict uncertainty for each state dimension
                        self.state_uncertainty_head = nn.Sequential(
                            SwiGLU(config.hidden_dim,
                                   hidden_dim=swiglu_hidden_dim, bias=use_bias),
                            nn.LayerNorm(
                                config.hidden_dim) if config.use_layernorm else nn.Identity(),
                            nn.Linear(config.hidden_dim,
                                      state_pred_hidden, bias=use_bias),
                            nn.ReLU(),
                            nn.Linear(state_pred_hidden,
                                      config.state_dim, bias=use_bias),
                            nn.Softplus()  # Ensure positive values for variance
                        )

            # For future image prediction
            if self.predict_future_image:
                # Image decoder components (ViT-style)
                img_decoder_dim = config.hidden_dim
                self.img_decoder_token = nn.Parameter(
                    torch.zeros(1, 1, img_decoder_dim))
                self.img_decoder_pos_embed = nn.Parameter(
                    torch.zeros(1, self.num_patches + 1, img_decoder_dim)
                )

                # Image decoder (simplified ViT structure)
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=img_decoder_dim,
                    nhead=config.num_heads // 2,  # Use fewer heads for the decoder
                    dim_feedforward=img_decoder_dim * 2,
                    dropout=config.dropout,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True
                )
                self.img_decoder = nn.TransformerDecoder(
                    decoder_layer=decoder_layer,
                    num_layers=4  # Use fewer layers for the decoder
                )

                # Image reconstruction head
                self.img_reconstruction_head = nn.Sequential(
                    nn.Linear(img_decoder_dim, img_decoder_dim),
                    nn.GELU(),
                    nn.Linear(img_decoder_dim, self.patch_size **
                              2 * self.in_channels)
                )

                # Add image uncertainty prediction if enabled
                if self.predict_uncertainty:
                    self.img_uncertainty_head = nn.Sequential(
                        nn.Linear(img_decoder_dim, img_decoder_dim),
                        nn.GELU(),
                        nn.Linear(img_decoder_dim, self.patch_size **
                                  2 * self.in_channels),
                        nn.Softplus()  # Ensure positive values for variance
                    )
        else:
            self.predict_future = False

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using modern initialization methods"""
        # Normal initialization for embeddings and parameters with std=0.02
        for param in [self.pos_embedding, self.img_pos_embedding, self.cls_token]:
            nn.init.normal_(param, mean=0.0, std=0.02)

        # Initialize future prediction parameters if present
        if hasattr(self, "predict_future") and self.predict_future:
            if self.predict_future_image:
                nn.init.normal_(self.img_decoder_token, mean=0.0, std=0.02)
                nn.init.normal_(self.img_decoder_pos_embed, mean=0.0, std=0.02)

            # Initialize uncertainty heads if present
            if hasattr(self, "predict_uncertainty") and self.predict_uncertainty:
                if hasattr(self, "state_uncertainty_head"):
                    for m in self.state_uncertainty_head.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

                if hasattr(self, "img_uncertainty_head"):
                    for m in self.img_uncertainty_head.modules():
                        if isinstance(m, nn.Linear):
                            nn.init.xavier_uniform_(m.weight)
                            if m.bias is not None:
                                nn.init.zeros_(m.bias)

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
            seq_len = trajectory_sequence.shape[1]

            # Ensure we don't exceed the available position embeddings
            start_idx = self.half_horizon
            end_idx = min(self.half_horizon + seq_len, self.horizon)

            if end_idx - start_idx < seq_len:
                print(
                    f"Warning: Position embeddings truncated. Need {seq_len} positions starting at {start_idx}, but only have {end_idx-start_idx}")

            # Use position embeddings from half_horizon onward
            pos_embeddings_to_use = self.pos_embedding[:, start_idx:end_idx, :]

            # Handle case where sequence is longer than available positions
            if pos_embeddings_to_use.shape[1] < seq_len:
                # Pad with the last position embedding
                last_pos = self.pos_embedding[:, -1:,
                                              :].expand(-1, seq_len - pos_embeddings_to_use.shape[1], -1)
                pos_embeddings_to_use = torch.cat(
                    [pos_embeddings_to_use, last_pos], dim=1)

            state_embeddings = state_embeddings + pos_embeddings_to_use
        else:
            # For first half or full sequence, use the beginning of the position embeddings
            seq_len = trajectory_sequence.shape[1]

            # Ensure we don't exceed available position embeddings
            if seq_len <= self.pos_embedding.shape[1]:
                # Standard case - enough position embeddings
                pos_embeddings_to_use = self.pos_embedding[:, :seq_len, :]
            else:
                # Sequence is longer than available positions - use what we have and pad
                available_pos = self.pos_embedding.shape[1]
                pos_embeddings_to_use = self.pos_embedding
                # Pad with the last position embedding
                last_pos = self.pos_embedding[:, -1:,
                                              :].expand(-1, seq_len - available_pos, -1)
                pos_embeddings_to_use = torch.cat(
                    [pos_embeddings_to_use, last_pos], dim=1)

            state_embeddings = state_embeddings + pos_embeddings_to_use

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

        # Ensure the score has shape [B, 1] for easier processing
        if score.ndim == 1:
            score = score.unsqueeze(-1)  # Convert [B] to [B, 1]

        return score

    def score(self, trajectory_sequence: torch.Tensor, raw_images: torch.Tensor = None, second_half: bool = False) -> torch.Tensor:
        """
        Scoring entrypoint. This maintains gradients for proper backpropagation during training.

        For inference use cases, uses only the first half of the trajectory for scoring, because:
        1. The second half contains noisy predictions that make it less reliable
        2. The previous half would use previous observations/states in a queue-like manner

        Args:
            trajectory_sequence: (B, H, D_state) tensor of state sequences.
            raw_images: Optional (B, C, H, W) tensor of raw images or (B, 2, C, H, W) for first and second half images.
            second_half: Whether this is the second half of the sequence (for positional embedding indexing)
        """
        # Check trajectory dimensions
        B, H, D = trajectory_sequence.shape

        # First verify that it's a valid trajectory
        if D != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {D}")

        # If we're in training mode, use both halves for next sequence prediction
        if self.training:
            # Check if this is a full sequence that should be split (exact match with horizon)
            if not second_half and H == self.horizon:
                # This is a full sequence, so we should use score_next_sequence for consistency
                first_half = trajectory_sequence[:, :self.half_horizon]
                second_half = trajectory_sequence[:,
                                                  self.half_horizon:2*self.half_horizon]

                # Prepare images for next sequence prediction if provided
                if raw_images is not None:
                    if raw_images.ndim == 4:  # B, C, H, W
                        raw_images_stacked = torch.stack(
                            [raw_images, raw_images], dim=1)
                    else:
                        raw_images_stacked = raw_images
                    return self.score_next_sequence(first_half, second_half, raw_images_stacked)
                else:
                    return self.score_next_sequence(first_half, second_half)
            # Check if it's potentially a full sequence that could be split (at least 2*half_horizon)
            elif not second_half and H >= 2 * self.half_horizon:
                # This might be a variable-length trajectory that's at least full length
                first_half = trajectory_sequence[:, :self.half_horizon]
                second_half = trajectory_sequence[:,
                                                  self.half_horizon:self.half_horizon*2]

                # Log that we're splitting a non-standard length trajectory
                print(
                    f"Splitting non-standard length trajectory: {H} steps into {self.half_horizon} + {self.half_horizon}")

                # Prepare images for next sequence prediction
                if raw_images is not None:
                    if raw_images.ndim == 4:  # B, C, H, W
                        raw_images_stacked = torch.stack(
                            [raw_images, raw_images], dim=1)
                    else:
                        raw_images_stacked = raw_images
                    return self.score_next_sequence(first_half, second_half, raw_images_stacked)
                else:
                    return self.score_next_sequence(first_half, second_half)
        # For inference, use only the first half regardless of whether this is a full trajectory
        elif not second_half and H >= self.half_horizon:
            # Use only the first half of the trajectory for scoring during inference
            first_half = trajectory_sequence[:, :self.half_horizon]
            print(
                f"Inference mode: Using only first half ({self.half_horizon} steps) for scoring")

            # Add more diagnostic info for debugging
            if H > self.half_horizon:
                print(
                    f"Truncating trajectory from {H} steps to {self.half_horizon} steps for inference scoring")
                # Use the first element of the batch for printing example values
                print(
                    f"Example from first batch element - First value: {trajectory_sequence[0, 0, 0].item():.4f}, Last value used: {trajectory_sequence[0, self.half_horizon-1, 0].item():.4f}")

            # Process this as a first half trajectory
            score = self.forward(first_half, raw_images, second_half=False)
            return score
        else:
            # For shorter trajectories or when explicitly marked as second half
            print(
                f"Using trajectory as-is (length {H}, second_half={second_half})")
            score = self.forward(trajectory_sequence, raw_images, second_half)

            # Ensure score has shape [B, 1] for consistent handling
            if score.ndim == 1:
                score = score.unsqueeze(-1)  # Convert [B] to [B, 1]

            return score

    def score_next_sequence(self, first_half: torch.Tensor, second_half: torch.Tensor,
                            raw_images: torch.Tensor = None) -> torch.Tensor:
        """
        Score how likely the second half sequence is a correct continuation of the first half.

        During training: Uses both halves and combines their scores.
        During inference: Uses only the first half for scoring.

        Args:
            first_half: (B, H/2, D_state) tensor of first half state sequences
            second_half: (B, H/2, D_state) tensor of second half state sequences
            raw_images: (B, 2, C, H, W) tensor with images for first and second half
                        or (B, C, H, W) tensor with a single image used for both halves

        Returns:
            (B, 1) tensor of scores, higher means more likely to be correct continuation
        """
        # Validate trajectory shapes
        if first_half.shape[1] != self.half_horizon:
            if not self.training:
                print(
                    f"Warning: First half trajectory with {first_half.shape[1]} steps (expected {self.half_horizon})")

            # For inference with non-standard lengths, make sure we're not exceeding half_horizon
            if not self.training and first_half.shape[1] > self.half_horizon:
                print(
                    f"Truncating first half from {first_half.shape[1]} to {self.half_horizon} steps")
                first_half = first_half[:, :self.half_horizon]

        # Prepare images if we only have one image per sequence
        if raw_images is not None and raw_images.ndim == 4:
            # Duplicate the image for both first and second half
            raw_images = torch.stack(
                [raw_images, raw_images], dim=1)  # B, 2, C, H, W

        if raw_images is not None:
            if raw_images.ndim != 5:
                raise ValueError(
                    f"Expected raw_images to be (B, 2, C, H, W), got shape {raw_images.shape}")
            first_image = raw_images[:, 0]  # (B, C, H, W)
            second_image = raw_images[:, 1]  # (B, C, H, W)
        else:
            first_image = None
            # For inference, use only the first half (more reliable)
            second_image = None
            if not self.training:
                print("Inference mode: Using only first half for scoring_next_sequence")
                first_scores = self.forward(
                    first_half, raw_images=first_image, second_half=False)
                return first_scores
            else:
                # For training, use both halves as before
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
            print("No 'observation.image' key found in norm_batch")
            # Try checking for other possible image keys
            possible_keys = [
                k for k in norm_batch.keys() if "image" in k.lower()]
            if possible_keys:
                print(f"Found potential image keys: {possible_keys}")
            return None, None

        img_tensor = norm_batch["observation.image"]
        print(f"Image tensor shape: {img_tensor.shape}")

        # Extract images for first and second halves
        if img_tensor.ndim == 5:  # B, T_img, C, H, W
            # Take images at timestep 0 and timestep half_horizon
            # Make sure we're within bounds
            first_idx = 0

            # For second half, try to use an image at half_horizon if available,
            # otherwise use the last available image
            if img_tensor.shape[1] > self.half_horizon:
                second_idx = self.half_horizon
            else:
                second_idx = img_tensor.shape[1] - 1
                print(
                    f"Warning: Image sequence too short for ideal second half index. Using index {second_idx} instead of {self.half_horizon}")

            first_half_img = img_tensor[:,
                                        first_idx].contiguous()  # B, C, H, W
            second_half_img = img_tensor[:,
                                         second_idx].contiguous()  # B, C, H, W

            print(
                f"Using image indices {first_idx} and {second_idx} from sequence of length {img_tensor.shape[1]}")
        elif img_tensor.ndim == 4:  # B, C, H, W - only single image
            # Use the same image for both halves
            first_half_img = img_tensor
            second_half_img = img_tensor
            print("Using single image for both halves")
        # B, V, T, C, H, W (multi-view)
        elif img_tensor.ndim == 6 and img_tensor.shape[1] > 0:
            # Handle multi-view images - take first view
            print(
                f"Detected multi-view image format with {img_tensor.shape[1]} views")
            view_img = img_tensor[:, 0]  # B, T, C, H, W (first view)

            # Now handle as regular time sequence
            first_idx = 0
            second_idx = min(self.half_horizon, view_img.shape[1]-1)

            first_half_img = view_img[:, first_idx].contiguous()  # B, C, H, W
            second_half_img = view_img[:,
                                       second_idx].contiguous()  # B, C, H, W
        else:
            print(
                f"Warning: Unexpected image tensor shape: {img_tensor.shape}, attempting to adapt")
            # Try to adapt the tensor to the expected format
            if img_tensor.ndim >= 3:
                # Take the first dimensions until we get to B, C, H, W
                while img_tensor.ndim > 4:
                    img_tensor = img_tensor[:, 0]

                if img_tensor.ndim == 4:  # Successfully reduced to B, C, H, W
                    first_half_img = img_tensor
                    second_half_img = img_tensor
                else:
                    raise ValueError(
                        f"Failed to adapt image tensor of shape {img_tensor.shape}")
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
                # Extract trajectories from the normalized batch
                trajectories = norm_batch["observation.state"]

                # Check if we received time-series data with a time dimension
                if trajectories.ndim == 3:  # [B, T, D]
                    # Use up to self.horizon steps
                    horizon_to_use = min(trajectories.shape[1], self.horizon)
                    trajectories = trajectories[:, :horizon_to_use]

                    # Log shape information for debugging
                    print(
                        f"Trajectory shape from batch: {trajectories.shape}, using horizon: {horizon_to_use}")
                    print(f"Half-horizon value: {self.half_horizon}")

                    # Check if we have enough steps for splitting
                    if trajectories.shape[1] < self.horizon:
                        print(
                            f"Warning: Received trajectory with {trajectories.shape[1]} steps, but expected {self.horizon}")

                # Ensure we have enough timesteps for splitting
                if trajectories.shape[1] >= 2 * self.half_horizon:
                    # Split into first half and second half
                    first_half = trajectories[:, :self.half_horizon]
                    second_half = trajectories[:,
                                               self.half_horizon:self.half_horizon*2]
                elif trajectories.shape[1] > self.half_horizon:
                    # We have more than half but less than full
                    first_half = trajectories[:, :self.half_horizon]
                    # Just use what we have for second half, will be shorter than expected
                    second_half = trajectories[:, self.half_horizon:]
                    # Pad second half if needed to match half_horizon length
                    if second_half.shape[1] < self.half_horizon:
                        pad_size = self.half_horizon - second_half.shape[1]
                        pad = torch.zeros((second_half.shape[0], pad_size, second_half.shape[2]),
                                          device=second_half.device, dtype=second_half.dtype)
                        second_half = torch.cat([second_half, pad], dim=1)
                else:
                    # Not enough steps for proper splitting
                    raise ValueError(
                        f"Trajectory too short ({trajectories.shape[1]} steps) for next sequence prediction task (needs at least {self.half_horizon+1} steps)")

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
        elif positive_trajectories is not None and negative_trajectories is not None:
            # Legacy support for explicitly provided trajectories
            # Verify shapes
            if positive_trajectories.shape[1] < 2 * self.half_horizon or negative_trajectories.shape[1] < 2 * self.half_horizon:
                raise ValueError(
                    f"Trajectory too short for next sequence prediction task (needs {2*self.half_horizon} steps)")

            # Split positive trajectories
            positive_first_half = positive_trajectories[:, :self.half_horizon]
            positive_second_half = positive_trajectories[:,
                                                         self.half_horizon:2*self.half_horizon]

            # Split negative trajectories
            negative_first_half = negative_trajectories[:, :self.half_horizon]
            negative_second_half = negative_trajectories[:,
                                                         self.half_horizon:2*self.half_horizon]
        else:
            raise ValueError(
                "Either norm_batch or positive_trajectories/negative_trajectories must be provided")

        device = positive_first_half.device
        B = positive_first_half.shape[0]

        # Apply noise to negative second half trajectories if noise_params is provided
        if noise_params is not None:
            base_noise_scale = noise_params.get("base_noise_scale", 0.05)
            noise_type = noise_params.get("noise_type", "progressive")
            noise_growth_factor = noise_params.get("noise_growth_factor", 1.2)

            # Get noise scheduling parameters
            noise_schedule = noise_params.get("noise_schedule", "none")
            initial_multiplier = noise_params.get(
                "initial_noise_multiplier", 2.0)
            final_multiplier = noise_params.get("final_noise_multiplier", 0.5)
            current_step = noise_params.get("current_step", 0)
            total_steps = noise_params.get("total_steps", 10000)

            # Calculate current noise multiplier based on training progress
            noise_multiplier = initial_multiplier

            if noise_schedule != "none" and total_steps > 0:
                progress = min(current_step / total_steps, 1.0)

                if noise_schedule == "linear":
                    # Linear decay from initial_multiplier to final_multiplier
                    noise_multiplier = initial_multiplier + progress * \
                        (final_multiplier - initial_multiplier)

                elif noise_schedule == "cosine":
                    # Cosine decay from initial_multiplier to final_multiplier
                    cosine_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    noise_multiplier = final_multiplier + \
                        (initial_multiplier - final_multiplier) * cosine_factor

                elif noise_schedule == "exponential":
                    # Exponential decay from initial_multiplier to final_multiplier
                    decay_rate = np.log(final_multiplier / initial_multiplier)
                    noise_multiplier = initial_multiplier * \
                        np.exp(decay_rate * progress)

            # Apply the noise multiplier to the base noise scale
            base_noise_scale *= noise_multiplier

            # Log the current noise level occasionally
            if torch.rand(1).item() < 0.01:  # Only log about 1% of the time
                print(
                    f"Current noise multiplier: {noise_multiplier:.4f}x, Effective base noise: {base_noise_scale:.6f}")

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
                        # Make sure we clone from the original second half, not the already noised one
                        negative_second_half[i] = second_half[j].clone()

        # Compute model predictions
        # Score the positive pairs (first half -> true second half)
        positive_scores = self.score_next_sequence(
            positive_first_half, positive_second_half, raw_images=raw_images)

        # Score the negative pairs (first half -> noised/shuffled second half)
        negative_scores = self.score_next_sequence(
            negative_first_half, negative_second_half, raw_images=raw_images)

        # Combine positive and negative scores and create labels
        all_scores = torch.cat([positive_scores, negative_scores], dim=0)

        # Create labels: 1 for positive pairs, 0 for negative pairs
        positive_labels = torch.ones_like(positive_scores, device=device)
        negative_labels = torch.zeros_like(negative_scores, device=device)
        all_labels = torch.cat([positive_labels, negative_labels], dim=0)

        # Calculate binary cross entropy loss
        # Ensure scores are properly shaped for loss calculation
        if all_scores.ndim > 2:
            all_scores = all_scores.view(all_scores.size(0), -1)
            print(
                f"Warning: Reshaped all_scores from complex shape to shape {all_scores.shape}")

        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(all_scores, all_labels)

        # Calculate accuracy
        # Threshold at 0 since we're using BCEWithLogitsLoss
        predictions = (all_scores > 0.0).float()
        accuracy = (predictions == all_labels).float().mean()

        # Print some diagnostic information during training
        if torch.rand(1).item() < 0.01:  # Only print occasionally
            print(f"Batch size: {B}, Pos mean score: {positive_scores.mean().item():.4f}, "
                  f"Neg mean score: {negative_scores.mean().item():.4f}, Accuracy: {accuracy.item():.4f}")

        return loss, accuracy

    def predict_future_trajectory(self, current_trajectory: torch.Tensor, current_image: torch.Tensor = None) -> tuple:
        """
        Predicts future state (t+future_steps) and image based on current trajectory and image.

        Args:
            current_trajectory: (B, H_current, D_state) tensor of current state trajectory
            current_image: (B, C, H, W) tensor of current image frame

        Returns:
            tuple containing prediction outputs:
            - If multi_step_prediction=False (default):
                (future_state, future_image, state_uncertainty, image_uncertainty)
                - future_state: (B, D_state) tensor of predicted state at t+future_steps
                - future_image: (B, C, H, W) tensor of predicted image at t+future_steps or None
                - state_uncertainty: (B, D_state) tensor of predicted state uncertainty or None
                - image_uncertainty: (B, C, H, W) tensor of predicted image uncertainty or None

            - If multi_step_prediction=True:
                (future_states, future_image, state_uncertainties, image_uncertainty)
                - future_states: (B, num_future_steps, D_state) tensor of predicted states
                - future_image: (B, C, H, W) tensor of predicted image at t+future_steps or None
                - state_uncertainties: (B, num_future_steps, D_state) tensor of predicted 
                  state uncertainties or None
                - image_uncertainty: (B, C, H, W) tensor of predicted image uncertainty or None
        """
        if not hasattr(self, "predict_future") or not self.predict_future:
            raise ValueError(
                "This model was not configured for future prediction. Set predict_future=True in config.")

        B, H, D = current_trajectory.shape

        # Validate input dimensions
        if D != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {D}")

        # Process current image
        if current_image is not None:
            img_embedding = self.process_raw_images(current_image).unsqueeze(1)
        elif not self.use_image_context:
            img_embedding = self.cls_token.expand(B, 1, -1)
        else:
            raise ValueError("Image must be provided for future prediction")

        # Embed current trajectory
        state_embeddings = self.state_embedding(current_trajectory)

        # Add positional embeddings
        seq_len = current_trajectory.shape[1]
        if seq_len <= self.pos_embedding.shape[1]:
            pos_embeddings = self.pos_embedding[:, :seq_len, :]
        else:
            # Handle case where sequence is longer than available positions
            available_pos = self.pos_embedding.shape[1]
            pos_embeddings = self.pos_embedding
            # Pad with the last position embedding
            last_pos = self.pos_embedding[:, -1:,
                                          :].expand(-1, seq_len - available_pos, -1)
            pos_embeddings = torch.cat([pos_embeddings, last_pos], dim=1)

        state_embeddings = state_embeddings + pos_embeddings

        # Combine image and state embeddings
        sequence_for_transformer = torch.cat(
            [img_embedding, state_embeddings], dim=1)  # (B, H+1, hidden_dim)

        # Pass through transformer
        transformer_output = self.transformer_encoder(sequence_for_transformer)

        # Get pooled representation for future prediction
        pooled_output = self.attention_pooler(transformer_output)

        # Initialize return values
        future_state = None
        future_image = None
        state_uncertainty = None
        image_uncertainty = None

        # Predict future state if enabled
        if self.predict_future_state:
            if hasattr(self, "multi_step_prediction") and self.multi_step_prediction:
                # For multi-step prediction
                future_state_flat = self.future_state_decoder(pooled_output)

                # Reshape to [B, num_future_steps, state_dim]
                future_state = future_state_flat.view(
                    B, self.num_future_steps, self.state_dim)

                # Predict uncertainty if enabled
                if hasattr(self, "predict_uncertainty") and self.predict_uncertainty and hasattr(self, "state_uncertainty_head"):
                    state_uncertainty_flat = self.state_uncertainty_head(
                        pooled_output)
                    state_uncertainty = state_uncertainty_flat.view(
                        B, self.num_future_steps, self.state_dim)
            else:
                # Standard single-step prediction
                future_state = self.future_state_decoder(pooled_output)

                # Predict uncertainty if enabled
                if hasattr(self, "predict_uncertainty") and self.predict_uncertainty and hasattr(self, "state_uncertainty_head"):
                    state_uncertainty = self.state_uncertainty_head(
                        pooled_output)

        # Predict future image if enabled
        if self.predict_future_image and current_image is not None:
            # Get image encoder memory
            B = current_image.shape[0]

            # Resize images if they don't match expected size
            if current_image.shape[2] != self.img_size or current_image.shape[3] != self.img_size:
                current_image = F.interpolate(
                    current_image,
                    size=(self.img_size, self.img_size),
                    mode='bilinear',
                    align_corners=False
                )

            # Get image encoder memory
            patch_embs = self.patch_embedding(current_image)
            patch_embeddings = patch_embs.transpose(1, 2)

            # Create decoder tokens with position embedding
            decoder_tokens = self.img_decoder_token.expand(
                B, self.num_patches + 1, -1)
            decoder_tokens = decoder_tokens + self.img_decoder_pos_embed

            # Add pooled feature as a condition
            decoder_tokens[:, 0] = pooled_output

            # No need for causal mask as this isn't autoregressive generation
            tgt_mask = None
            memory_mask = None

            # Decode future image patches
            decoded_patches = self.img_decoder(
                tgt=decoder_tokens,
                memory=patch_embeddings,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask
            )

            # Skip the first token (condition token) and reconstruct patches
            decoded_patches = decoded_patches[:, 1:, :]

            # Apply reconstruction head to get pixel values for each patch
            patch_pixels = self.img_reconstruction_head(decoded_patches)
            patch_pixels = patch_pixels.reshape(
                B, self.num_patches, self.patch_size, self.patch_size, self.in_channels
            )

            # Rearrange to image
            patches_per_side = int(np.sqrt(self.num_patches))
            future_image = patch_pixels.permute(0, 1, 4, 2, 3).reshape(
                B, self.in_channels,
                patches_per_side * self.patch_size,
                patches_per_side * self.patch_size
            )

            # Apply tanh to normalize pixel values to [-1, 1]
            future_image = torch.tanh(future_image)

            # Predict image uncertainty if enabled
            if hasattr(self, "predict_uncertainty") and self.predict_uncertainty and hasattr(self, "img_uncertainty_head"):
                patch_uncertainties = self.img_uncertainty_head(
                    decoded_patches)
                patch_uncertainties = patch_uncertainties.reshape(
                    B, self.num_patches, self.patch_size, self.patch_size, self.in_channels
                )

                # Rearrange to image
                image_uncertainty = patch_uncertainties.permute(0, 1, 4, 2, 3).reshape(
                    B, self.in_channels,
                    patches_per_side * self.patch_size,
                    patches_per_side * self.patch_size
                )

        # Return predictions and uncertainties
        return future_state, future_image, state_uncertainty, image_uncertainty
