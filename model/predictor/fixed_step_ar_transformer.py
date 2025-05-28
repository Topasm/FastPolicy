#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/model/predictor/fixed_step_ar_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class FixedStepARTransformerConfig:
    """
    Configuration for the Fixed-Step Bidirectional Autoregressive Transformer model.
    """
    state_dim: int = 7                # Dimension of state vectors
    hidden_dim: int = 768             # Hidden dimension for transformer layers
    num_layers: int = 8               # Number of transformer layers
    num_heads: int = 12               # Number of attention heads
    dropout: float = 0.1              # Dropout rate
    # Maximum position value (for timestep 64)
    max_position_value: int = 64
    pad_token_idx: int = -100         # Index for padding tokens
    layernorm_epsilon: float = 1e-5   # Epsilon for layer normalization
    bidirectional: bool = True        # Whether to support bidirectional generation


class FixedStepARTransformer(nn.Module):
    """
    Fixed-Step Bidirectional Autoregressive Transformer for trajectory generation.

    Features:
    - Generates fixed-step trajectories (e.g., at steps [0, 4, 8, ..., 64])
    - Works in both forward and backward directions
    - Uses concatenated start/goal + standard causal mask approach
    - Uses nn.Embedding for standard absolute position encoding
    - Suitable for exact sampling at predefined intervals
    """

    def __init__(self, config: FixedStepARTransformerConfig):
        """
        Initialize the fixed-step transformer.

        Args:
            config: Configuration parameters
        """
        super().__init__()
        self.config = config

        # Define the fixed target steps for trajectory generation (17 evenly spaced points)
        self.TARGET_STEPS = list(range(0, config.max_position_value + 1, 4))

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

        # Output projection layer
        self.output_head = nn.Linear(config.hidden_dim, config.state_dim)

        # Initialize weights
        self.apply(self._init_weights)

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
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through the transformer.

        Args:
            inputs: Input tensor of shape [B, L, state_dim]
            positions: Position IDs, shape [B, L]
            direction: Direction indicator (0=forward, 1=backward) [B]
            attention_mask: Optional attention mask (1=attend, 0=mask), shape [B, L]

        Returns:
            Output state predictions
        """
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

        # Project to output space
        output = self.output_head(x)

        return output

    def generate(
        self,
        start_z: torch.Tensor,
        goal_z: Optional[torch.Tensor] = None,
        direction: int = 0,  # 0=forward, 1=backward
    ) -> torch.Tensor:
        """
        Generate a fixed-step trajectory autoregressively.

        Args:
            start_z: Initial state to start generation from [B, state_dim]
            goal_z: Goal state for conditioning [B, state_dim]
            direction: Generation direction (0=forward, 1=backward)

        Returns:
            Generated trajectory [B, 17, state_dim] at fixed steps
        """
        batch_size, state_dim = start_z.shape
        device = start_z.device

        # Determine steps list based on direction
        if direction == 0:  # forward
            steps = self.TARGET_STEPS.copy()
        else:  # backward
            steps = self.TARGET_STEPS.copy()
            steps.reverse()

        # Ensure direction is a tensor
        if isinstance(direction, int):
            direction = torch.full(
                (batch_size,), direction, dtype=torch.long, device=device)

        # Initialize with start and goal states
        if goal_z is not None:
            if direction[0] == 0:  # Forward
                current_states = torch.cat([
                    start_z.unsqueeze(1),  # z_0
                    goal_z.unsqueeze(1)    # z_64
                ], dim=1)

                # Position IDs: [0, 64]
                current_positions = torch.tensor(
                    [[steps[0], steps[-1]]],
                    device=device
                ).expand(batch_size, 2)

            else:  # Backward
                current_states = torch.cat([
                    goal_z.unsqueeze(1),   # z_64 (in reverse)
                    start_z.unsqueeze(1)   # z_0 (in reverse)
                ], dim=1)

                # Position IDs: [64, 0]
                current_positions = torch.tensor(
                    [[steps[0], steps[-1]]],
                    device=device
                ).expand(batch_size, 2)
        else:
            # Just start with the start state
            current_states = start_z.unsqueeze(1)
            current_positions = torch.tensor(
                [[steps[0]]],
                device=device
            ).expand(batch_size, 1)

        # Generate all intermediate points (15 more steps for a total of 17)
        for i in range(1, len(steps) - 1):
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
                (batch_size, 1), steps[i], dtype=torch.long, device=device)

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

        return current_states
