#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/model/predictor/continuous_ar_transformer.py
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Tuple, Union


@dataclass
class ContinuousARTransformerConfig:
    """
    Configuration for the Continuous Bidirectional Autoregressive Transformer model with image decoder.
    """
    state_dim: int = 7                # Dimension of state vectors
    hidden_dim: int = 768             # Hidden dimension for transformer layers
    num_layers: int = 8               # Number of transformer layers
    num_heads: int = 12               # Number of attention heads
    dropout: float = 0.1              # Dropout rate
    max_position_value: int = 64      # Maximum position value
    pad_token_idx: int = -100         # Index for padding tokens
    layernorm_epsilon: float = 1e-5   # Epsilon for layer normalization
    bidirectional: bool = True        # Whether to support bidirectional generation
    image_channels: int = 3           # Number of image channels (RGB)
    image_size: int = 64              # Size of the output image (square)


class ContinuousARTransformer(nn.Module):
    """
    Continuous Bidirectional Autoregressive Transformer for trajectory generation.

    Features:
    - Generates continuous trajectories (not fixed-step)
    - Works in both forward and backward directions
    - Uses concatenated start/goal + standard causal mask approach
    - Uses nn.Embedding for standard absolute position encoding
    - Includes image decoder for generating goal images
    """

    def __init__(self, config: ContinuousARTransformerConfig):
        """
        Initialize the continuous transformer.

        Args:
            config: Configuration parameters
        """
        super().__init__()
        self.config = config

        # Input embedding layer for state vectors
        self.input_embedding = nn.Linear(config.state_dim, config.hidden_dim)

        # Direction embedding (0=forward, 1=backward)
        self.direction_embedding = nn.Embedding(2, config.hidden_dim)

        # Position embedding using standard Embedding
        # +1 for potential padding token position
        self.position_embedding = nn.Embedding(
            config.max_position_value + 2,
            config.hidden_dim
        )

        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        # Output projection layer for state prediction
        self.output_head = nn.Linear(config.hidden_dim, config.state_dim)

        # Image decoder for generating goal images
        self.image_decoder = self._build_image_decoder(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _build_image_decoder(self, config):
        """
        Build the decoder network for generating images from latent vectors.
        """
        # Define the smallest spatial dimension of the decoder
        initial_size = 4

        # Calculate the initial number of features
        initial_features = 64 * (2 ** 4)  # 1024

        # Layers to convert from hidden state to initial decoder features
        decoder_input = nn.Sequential(
            nn.Linear(config.hidden_dim, initial_features *
                      initial_size * initial_size),
            nn.ReLU(),
            nn.Unflatten(1, (initial_features, initial_size, initial_size))
        )

        # Upsampling layers to reach target image size
        decoder_layers = []
        current_size = initial_size
        current_features = initial_features

        # Calculate how many upsampling layers we need to reach target size
        while current_size < config.image_size:
            out_features = current_features // 2
            decoder_layers.extend([
                nn.ConvTranspose2d(
                    current_features, out_features,
                    kernel_size=4, stride=2, padding=1
                ),
                nn.BatchNorm2d(out_features),
                nn.ReLU()
            ])
            current_features = out_features
            current_size *= 2

        # Final layer to produce the image
        decoder_layers.append(
            nn.Conv2d(current_features, config.image_channels,
                      kernel_size=3, padding=1)
        )
        decoder_layers.append(nn.Tanh())  # Output in [-1, 1] range

        return nn.Sequential(*decoder_layers, decoder_input)

    def _init_weights(self, module):
        """
        Initialize the weights of the model.
        """
        if isinstance(module, nn.Linear):
            # Linear layers initialized normally
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # Embeddings initialized normally
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            # LayerNorm biases set to 0, weights to 1
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
            # Conv/ConvTranspose layers initialized with Kaiming initialization
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Create a causal attention mask for autoregressive generation.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            A causal attention mask for self-attention
        """
        # Lower triangular matrix (including diagonal) with 1s
        # Upper triangular with 0s (will be converted to -inf in attention)
        mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device),
            diagonal=1
        )
        return mask

    def forward(
        self,
        inputs: torch.Tensor,
        positions: torch.Tensor,
        direction: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        decode_image: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the transformer.
        Handles both 2D [B, D] and 3D [B, L, D] inputs.

        Args:
            inputs: Input tensor of shape [B, L, state_dim] or [B, state_dim]
            positions: Position IDs, shape [B, L] or [B] if inputs are 2D
            direction: Direction indicator (0=forward, 1=backward) [B]
            attention_mask: Optional attention mask (1=attend, 0=mask), shape [B, L] or [B] if inputs are 2D
            decode_image: Whether to decode the goal state into an image

        Returns:
            Output state predictions and optionally decoded images
        """
        # Check if inputs are 2D and reshape to 3D if necessary
        if inputs.dim() == 2:
            # We have [B, D] instead of [B, L, D]
            # Reshape to [B, 1, D] for the transformer
            B, D_inputs = inputs.shape
            inputs = inputs.reshape(B, 1, D_inputs)

            # Also reshape positions and attention_mask if needed
            # If inputs are 2D, positions and attention_mask are expected to be 1D [B] or 2D [B,1]
            if positions.dim() == 1:
                positions = positions.reshape(B, 1)

            if attention_mask is not None and attention_mask.dim() == 1:
                attention_mask = attention_mask.reshape(B, 1)

        batch_size, seq_len, _ = inputs.shape
        device = inputs.device

        # Embed inputs
        x = self.input_embedding(inputs)

        # Add position embeddings
        pos_emb = self.position_embedding(positions)
        x = x + pos_emb

        # Add direction embedding
        dir_emb = self.direction_embedding(direction).unsqueeze(1)
        x = x + dir_emb

        # Create causal mask (ensures autoregressive generation)
        causal_mask = self._create_causal_mask(seq_len, device)

        # Combine with padding mask if provided
        if attention_mask is not None:
            # Convert attention_mask (1=attend, 0=mask) to a boolean mask for the transformer
            padding_mask = (1.0 - attention_mask).bool()
        else:
            padding_mask = None

        # Pass through transformer encoder with causal mask
        x = self.transformer_encoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )

        # Project to output space for state prediction
        output = self.output_head(x)

        # Optionally decode the goal state (last state) into an image
        if decode_image:
            # Extract the last hidden state for each sequence
            last_token_idx = attention_mask.sum(dim=1).long(
            ) - 1 if attention_mask is not None else torch.full((batch_size,), seq_len - 1, device=device)
            batch_indices = torch.arange(batch_size, device=device)
            goal_states = x[batch_indices, last_token_idx]

            # Decode into images
            decoded_images = self.image_decoder(goal_states)
            return output, decoded_images

        return output

    def generate(
        self,
        start_z: torch.Tensor,
        goal_z: Optional[torch.Tensor] = None,
        direction: int = 0,  # 0=forward, 1=backward
        num_steps: int = 17,  # Number of steps in the trajectory
        decode_goal_image: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Generate a continuous trajectory autoregressively.

        Args:
            start_z: Initial state to start generation from [B, state_dim]
            goal_z: Goal state for conditioning [B, state_dim]
            direction: Generation direction (0=forward, 1=backward)
            num_steps: Number of steps to generate in the trajectory
            decode_goal_image: Whether to decode the goal state into an image

        Returns:
            Generated trajectory [B, num_steps, state_dim] and optionally a decoded goal image
        """
        batch_size, state_dim = start_z.shape
        device = start_z.device

        # Ensure direction is a tensor
        if isinstance(direction, int):
            direction = torch.full(
                (batch_size,), direction, dtype=torch.long, device=device)

        # Create evenly spaced position values
        max_pos = self.config.max_position_value
        position_values = torch.linspace(
            0, max_pos, num_steps).long().to(device)

        if direction[0] == 1:  # Reverse positions for backward generation
            position_values = torch.flip(position_values, dims=[0])

        # Initialize with start and goal states
        if goal_z is not None:
            if direction[0] == 0:  # Forward
                current_states = torch.cat([
                    start_z.unsqueeze(1),  # Initial state
                    goal_z.unsqueeze(1)    # Goal state
                ], dim=1)

                # Position IDs: [0, max_pos]
                current_positions = torch.tensor(
                    [[position_values[0], position_values[-1]]],
                    device=device
                ).expand(batch_size, 2)

            else:  # Backward
                current_states = torch.cat([
                    goal_z.unsqueeze(1),   # Goal state (in reverse)
                    start_z.unsqueeze(1)   # Initial state (in reverse)
                ], dim=1)

                # Position IDs: [max_pos, 0]
                current_positions = torch.tensor(
                    [[position_values[0], position_values[-1]]],
                    device=device
                ).expand(batch_size, 2)
        else:
            # Just start with the start state
            current_states = start_z.unsqueeze(1)
            current_positions = torch.tensor(
                [[position_values[0]]],
                device=device
            ).expand(batch_size, 1)

        # Generate all intermediate points
        for i in range(1, num_steps - 1):
            # Create attention mask (all 1s for valid tokens)
            attn_mask = torch.ones(
                batch_size, current_states.shape[1], device=device)

            # Forward pass to get next state prediction
            with torch.no_grad():
                outputs = self.forward(
                    inputs=current_states,
                    positions=current_positions,
                    direction=direction,
                    attention_mask=attn_mask
                )

            # Get the next state prediction (last position)
            next_z = outputs[:, -1:, :]

            # Get the next position
            next_position = torch.full(
                (batch_size, 1), position_values[i], dtype=torch.long, device=device)

            # Append to the current sequence
            current_states = torch.cat([current_states, next_z], dim=1)
            current_positions = torch.cat(
                [current_positions, next_position], dim=1)

        # Ensure we have the goal state at the end
        if goal_z is not None and direction[0] == 0:
            # For forward, replace the last state with the goal
            current_states[:, -1] = goal_z
        elif goal_z is not None and direction[0] == 1:
            # For backward, replace the last state with the start
            current_states[:, -1] = start_z

        # For backward generation, reverse the sequence back to forward order
        if direction[0] == 1:
            current_states = torch.flip(current_states, dims=[1])

        # Optionally decode the goal state into an image
        if decode_goal_image:
            # Extract the last hidden state
            with torch.no_grad():
                # Get the hidden states again
                attn_mask = torch.ones(
                    batch_size, current_states.shape[1], device=device)
                hidden_states = self.input_embedding(current_states)
                pos_emb = self.position_embedding(current_positions)
                hidden_states = hidden_states + pos_emb
                dir_emb = self.direction_embedding(direction).unsqueeze(1)
                hidden_states = hidden_states + dir_emb

                # Process through transformer without causal mask for final decoding
                hidden_states = self.transformer_encoder(
                    src=hidden_states,
                    mask=None,  # No causal mask needed here
                    src_key_padding_mask=None
                )

                # Get the goal state representation
                goal_state = hidden_states[:, -1]

                # Decode the goal image
                goal_image = self.image_decoder(goal_state)

                return current_states, goal_image

        return current_states

    def decode_image(self, state: torch.Tensor) -> torch.Tensor:
        """
        Decode a state vector into an image.

        Args:
            state: State vector [B, state_dim] or [B, seq_len, state_dim]

        Returns:
            Decoded image [B, C, H, W]
        """
        if state.dim() == 3:
            # Take the last state from each sequence
            state = state[:, -1]

        # Embed the state
        hidden = self.input_embedding(state)

        # Decode into an image
        return self.image_decoder(hidden)
