#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/model/predictor/multimodal_future_predictor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from model.predictor.gpt2_blocks import GPT2Block, GPT2MLP


@dataclass
class MultimodalFuturePredictorConfig:
    """Configuration for MultimodalFuturePredictor - a transformer-based multimodal future prediction model
    that predicts future states and alternates between generating images and noise."""
    state_dim: int  # Dimension of state
    horizon: int    # Sequence length (time steps) for state trajectory
    hidden_dim: int = 768  # Hidden dimension for transformer layers
    dropout: float = 0.1  # Dropout rate
    use_layernorm: bool = True  # Use LayerNorm in output heads
    num_layers: int = 8  # Number of transformer layers
    num_heads: int = 12  # Number of attention heads
    mlp_intermediate_factor: int = 4  # Factor for MLP intermediate dimension

    # Future prediction parameters
    future_steps: int = 8  # Number of steps to predict into the future
    predict_uncertainty: bool = False  # Whether to predict uncertainty in predictions


# --- Using GPT2MLP instead of SwiGLU ---
# Importing from gpt2_blocks.py


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
    GPT2-style Transformer-based model for multimodal future prediction.

    Predicts future states, images and noise based on current trajectory and image context.
    Uses GPT2-style blocks with pre-normalization and residual connections.
    Always alternates between generating images and noise during training,
    which improves diversity and quality of generated outputs.

    Specifically designed to take an image at index 0 (current state) and predict 
    either an image or noise at index 8 (future state).

    Input: 
        - trajectory_sequence: (B, H, D_state) tensor of state sequences
        - current_image: (B, C, H, W) tensor of current image at index 0
        - generate_noise: Boolean flag to toggle between generating image or noise

    Output: Future predictions:
        - future_state: (B, D_state) - state at index 8
        - future_output: (B, C, H, W) - future image or noise at index 8
        - state_uncertainty: (B, D_state) (if uncertainty prediction is enabled)
        - output_uncertainty: (B, C, H, W) (if uncertainty prediction is enabled)
    """

    def __init__(self, config: MultimodalFuturePredictorConfig):
        super().__init__()
        self.state_dim = config.state_dim
        self.horizon = config.horizon
        self.hidden_dim = config.hidden_dim
        self.use_image_context = True
        use_bias = False  # No bias in linear layers

        # Always enable future prediction features
        self.predict_future_state = True
        self.predict_future_image = True

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

        # Calculate hidden dimension for MLP layers
        self.mlp_hidden_dim = config.hidden_dim * config.mlp_intermediate_factor

        # Use GPT2-style transformer blocks
        self.transformer_encoder = nn.ModuleList([
            GPT2Block(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])
        self.use_gpt2_style = True

        # Attention pooling instead of simple averaging or first token extraction
        self.attention_pooler = AttentionPooling(
            input_dim=config.hidden_dim,
            output_dim=config.hidden_dim,
            bias=use_bias
        )

        # Future prediction settings
        self.future_steps = config.future_steps
        self.predict_uncertainty = config.predict_uncertainty

        # For future state prediction (always enabled)
        state_pred_hidden = config.hidden_dim // 2

        # Standard single-step prediction
        self.future_state_decoder = nn.Sequential(
            GPT2MLP(config.hidden_dim,
                    hidden_dim=self.mlp_hidden_dim, dropout=config.dropout),
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
            # Predict uncertainty for each state dimension
            self.state_uncertainty_head = nn.Sequential(
                GPT2MLP(config.hidden_dim,
                        hidden_dim=self.mlp_hidden_dim, dropout=config.dropout),
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

            # Noise prediction head (always included)
            self.noise_prediction_head = nn.Sequential(
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

                # Noise uncertainty prediction (always included if using uncertainty)
                self.noise_uncertainty_head = nn.Sequential(
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

            # Initialize noise prediction head
            for m in self.noise_prediction_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

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

            # Initialize noise uncertainty head
            if hasattr(self, "noise_uncertainty_head"):
                for m in self.noise_uncertainty_head.modules():
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

        # Pass through transformer encoder - GPT2 style blocks
        img_features = patch_embeddings_with_cls
        for block in self.transformer_encoder:
            img_features = block(img_features)

        # Create mask where all positions except CLS are ignored
        cls_only_mask = torch.ones(
            B, self.num_patches + 1, dtype=torch.bool, device=image.device)
        cls_only_mask[:, 0] = False  # Don't mask the CLS token

        # Return attention-pooled representation
        return self.attention_pooler(img_features, mask=cls_only_mask)

    def predict_future_trajectory(self, current_trajectory: torch.Tensor, current_image: torch.Tensor = None, generate_noise: bool = False) -> tuple:
        """
        Predicts future state and image based on current trajectory and image.
        Always alternates between predicting images and noise.

        Args:
            current_trajectory: (B, H_current, D_state) tensor of current state trajectory
            current_image: (B, C, H, W) tensor of current image at index 0 (current state)
            generate_noise: Whether to generate noise instead of image

        Returns:
            tuple containing prediction outputs:
            (future_state, future_output, state_uncertainty, output_uncertainty)
            - future_state: (B, D_state) tensor of predicted state at t+future_steps 
            - future_output: (B, C, H, W) tensor of predicted image or noise
            - state_uncertainty: (B, D_state) tensor of predicted state uncertainty or None
            - output_uncertainty: (B, C, H, W) tensor of predicted uncertainty or None
        """
        B, H, D = current_trajectory.shape

        # Validate input dimensions
        if D != self.state_dim:
            raise ValueError(f"Expected state dim {self.state_dim}, got {D}")

        # Process current image - image is always required
        if current_image is not None:
            img_embedding = self.process_image(current_image).unsqueeze(1)
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

        # Pass through transformer - GPT2 style blocks
        transformer_output = sequence_for_transformer
        for block in self.transformer_encoder:
            transformer_output = block(transformer_output)

        # Get pooled representation for future prediction
        pooled_output = self.attention_pooler(transformer_output)

        # Initialize return values
        future_state = None
        future_image = None
        state_uncertainty = None
        image_uncertainty = None

        # Predict future state (always enabled)
        future_state = self.future_state_decoder(pooled_output)

        # Predict uncertainty if enabled
        if self.predict_uncertainty and hasattr(self, "state_uncertainty_head"):
            state_uncertainty = self.state_uncertainty_head(pooled_output)

        # Predict future image or noise if current image is provided
        if current_image is not None:
            # Use generate_noise parameter directly
            generate_noise_this_time = generate_noise

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

            # Skip the first token (condition token)
            decoded_patches = decoded_patches[:, 1:, :]

            # Apply reconstruction head to get pixel values for each patch
            if generate_noise_this_time:
                # Use noise prediction head
                patch_pixels = self.noise_prediction_head(decoded_patches)
            else:
                # Use image reconstruction head
                patch_pixels = self.img_reconstruction_head(decoded_patches)

            patch_pixels = patch_pixels.reshape(
                B, self.num_patches, self.patch_size, self.patch_size, self.in_channels
            )

            # Rearrange to image
            patches_per_side = int(np.sqrt(self.num_patches))
            output_tensor = patch_pixels.permute(0, 1, 4, 2, 3).reshape(
                B, self.in_channels,
                patches_per_side * self.patch_size,
                patches_per_side * self.patch_size
            )

            # Apply tanh to normalize pixel values to [-1, 1] for images but not for noise
            if not generate_noise_this_time:
                output_tensor = torch.tanh(output_tensor)
                future_image = output_tensor
            else:
                # For noise, we want values from a standard normal distribution
                future_image = output_tensor  # This is actually noise now

            # Predict uncertainty if enabled
            if self.predict_uncertainty:
                if generate_noise_this_time:
                    # Use noise uncertainty head
                    patch_uncertainties = self.noise_uncertainty_head(
                        decoded_patches)
                else:
                    # Use image uncertainty head
                    patch_uncertainties = self.img_uncertainty_head(
                        decoded_patches)

                if patch_uncertainties is not None:
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
