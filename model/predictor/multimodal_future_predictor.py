#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/model/predictor/multimodal_future_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class MultimodalFuturePredictorConfig:
    """Configuration for MultimodalFuturePredictor - a transformer-based multimodal future prediction model"""
    state_dim: int  # Dimension of state
    horizon: int    # Sequence length (time steps) for state trajectory
    hidden_dim: int = 768  # Hidden dimension for transformer layers
    dropout: float = 0.1  # Dropout rate
    use_layernorm: bool = True  # Use LayerNorm in output heads
    num_layers: int = 8  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    swiglu_intermediate_factor: int = 4  # Factor for SwiGLU intermediate dim

    # Future prediction parameters
    # Enable future prediction mode (default is now True)
    predict_future: bool = True
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


class MultimodalFuturePredictor(nn.Module):
    """
    Transformer-based model for multimodal future prediction.

    Predicts future states and images based on current trajectory and image context.
    Can predict single or multiple future steps with optional uncertainty estimation.

    Input: 
        - trajectory_sequence: (B, H, D_state) tensor of state sequences
        - current_image: (B, C, H, W) tensor of current image
    Output: Future predictions, depending on configuration:
        - future_state: (B, D_state) or (B, num_future_steps, D_state)
        - future_image: (B, C, H, W) 
        - state_uncertainty: (B, D_state) or (B, num_future_steps, D_state)
        - image_uncertainty: (B, C, H, W)
    """

    def __init__(self, config: MultimodalFuturePredictorConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = True
        use_bias = False  # No bias in linear layers

        # Image parameters for ViT-like processing
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
            activation='gelu',
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

        # Future prediction settings
        self.predict_future = config.predict_future
        self.future_steps = config.future_steps
        self.predict_future_image = config.predict_future_image
        self.predict_future_state = config.predict_future_state
        self.multi_step_prediction = getattr(
            config, "multi_step_prediction", False)
        self.num_future_steps = getattr(config, "num_future_steps", 1)
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

        self._init_weights()

    def _init_weights(self):
        """Initialize weights using modern initialization methods"""
        # Normal initialization for embeddings and parameters with std=0.02
        for param in [self.pos_embedding, self.img_pos_embedding, self.cls_token]:
            nn.init.normal_(param, mean=0.0, std=0.02)

        # Initialize future prediction parameters if present
        if self.predict_future_image:
            nn.init.normal_(self.img_decoder_token, mean=0.0, std=0.02)
            nn.init.normal_(self.img_decoder_pos_embed, mean=0.0, std=0.02)

        # Initialize uncertainty heads if present
        if self.predict_uncertainty:
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

    def process_image(self, image):
        """Process image with ViT-like patching approach."""
        # images: (B, C, H, W)
        B = image.shape[0]

        # Resize images if they don't match the expected size
        if image.shape[2] != self.img_size or image.shape[3] != self.img_size:
            image = F.interpolate(image, size=(
                self.img_size, self.img_size), mode='bilinear', align_corners=False)

        # Patch embedding and transpose
        patch_embs = self.patch_embedding(
            image)  # (B, hidden_dim, num_patches)
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

        # Create mask where all positions except CLS are ignored
        cls_only_mask = torch.ones(
            B, self.num_patches + 1, dtype=torch.bool, device=image.device)
        cls_only_mask[:, 0] = False  # Don't mask the CLS token

        # Return attention-pooled representation
        return self.attention_pooler(img_features, mask=cls_only_mask)

    def predict_future_trajectory(self, current_trajectory: torch.Tensor, current_image: torch.Tensor = None) -> tuple:
        """
        Predicts future state and image based on current trajectory and image.

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
        B, H, D = current_trajectory.shape

        # Validate input dimensions
        if D != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {D}")

        # Process current image
        if current_image is not None:
            img_embedding = self.process_image(current_image).unsqueeze(1)
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
            if self.multi_step_prediction:
                # For multi-step prediction
                future_state_flat = self.future_state_decoder(pooled_output)

                # Reshape to [B, num_future_steps, state_dim]
                future_state = future_state_flat.view(
                    B, self.num_future_steps, self.state_dim)

                # Predict uncertainty if enabled
                if self.predict_uncertainty and hasattr(self, "state_uncertainty_head"):
                    state_uncertainty_flat = self.state_uncertainty_head(
                        pooled_output)
                    state_uncertainty = state_uncertainty_flat.view(
                        B, self.num_future_steps, self.state_dim)
            else:
                # Standard single-step prediction
                future_state = self.future_state_decoder(pooled_output)

                # Predict uncertainty if enabled
                if self.predict_uncertainty and hasattr(self, "state_uncertainty_head"):
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
            if self.predict_uncertainty and hasattr(self, "img_uncertainty_head"):
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
