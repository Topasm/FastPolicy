import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.predictor.multimodal_future_predictor import MultimodalFuturePredictor, MultimodalFuturePredictorConfig
from model.predictor.causal_gpt2_blocks import CausalGPT2Block


class MultimodalAutoregressivePredictor(MultimodalFuturePredictor):
    """
    Extended version of MultimodalFuturePredictor with autoregressive prediction capabilities.

    This model adds causal masking to both the transformer encoder and decoder to enforce 
    autoregressive constraints in all predictions. It can be used for step-by-step 
    autoregressive prediction across multiple time windows.

    The key enhancement is the causal mask which ensures proper autoregressive generation,
    making each token only attend to previous positions in the sequence.
    """

    def __init__(self, config: MultimodalFuturePredictorConfig):
        # Initialize with parent class, but we'll replace the transformer blocks
        super().__init__(config)

        # Flag to indicate this is an autoregressive model
        self.autoregressive = True

        # Replace standard transformer blocks with causal ones
        # This ensures autoregressive behavior in the encoder as well as the decoder
        self.transformer_encoder = nn.ModuleList([
            CausalGPT2Block(
                dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

    def _create_causal_mask(self, seq_len: int, device: torch.device):
        """
        Create a causal attention mask for autoregressive generation.

        Args:
            seq_len: The sequence length
            device: The device to create the mask on

        Returns:
            A causal mask of shape (seq_len, seq_len) where future positions are masked
        """
        # Create an upper triangular mask with -inf above the diagonal
        # This ensures each position can only attend to itself and previous positions
        mask = torch.triu(torch.ones(seq_len, seq_len,
                          device=device) * float('-inf'), diagonal=1)
        return mask

    def predict_future_trajectory(self, current_trajectory: torch.Tensor, current_image: torch.Tensor = None, generate_noise: bool = False) -> tuple:
        """
        Autoregressive version of future trajectory prediction.
        Uses causal masking in the image decoder for autoregressive constraints.

        Args:
            current_trajectory: (B, H_current, D_state) tensor of current state trajectory
            current_image: (B, C, H, W) tensor of current image
            generate_noise: Whether to generate noise instead of image

        Returns:
            tuple containing prediction outputs:
            (future_state, future_output, state_uncertainty, output_uncertainty)
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

        # Pass through causal transformer blocks
        # This is different from the parent class as we explicitly enable causal masking
        transformer_output = sequence_for_transformer
        for block in self.transformer_encoder:
            # Apply causal masking to ensure autoregressive behavior in the encoder
            transformer_output = block(transformer_output, causal_mask=True)

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

            # Create causal mask for autoregressive generation
            # This is the key difference from the parent class
            seq_len = decoder_tokens.shape[1]
            tgt_mask = self._create_causal_mask(seq_len, decoder_tokens.device)
            memory_mask = None

            # Decode future image patches with causal attention
            decoded_patches = self.img_decoder(
                tgt=decoder_tokens,
                memory=patch_embeddings,
                tgt_mask=tgt_mask,  # Use causal mask
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
                patch_uncertainties = None

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
