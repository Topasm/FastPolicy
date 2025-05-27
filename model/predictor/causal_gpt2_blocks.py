"""
GPT2-style blocks with causal masking for autoregressive models.
These enhanced blocks implement GPT2 architecture with explicit causal masking support.
"""
import torch
import torch.nn as nn


class CausalGPT2MLP(nn.Module):
    """
    GPT2-style MLP with GELU activation.
    This is a standard feed-forward network with GELU activation used in GPT2.
    """

    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.1):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = 4 * dim  # Standard 4x expansion factor in GPT2
        self.c_fc = nn.Linear(dim, hidden_dim)
        self.c_proj = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.c_fc(x)
        x = self.activation(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class CausalGPT2AttentionBlock(nn.Module):
    """
    GPT2-style self-attention block with explicit causal masking.
    This enforces autoregressive constraints for true autoregressive generation.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Pre-normalization for better training stability
        self.ln_1 = nn.LayerNorm(dim)

        # Self-attention with explicit support for causal masking
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = CausalGPT2MLP(dim, dropout=dropout)

        # Apply proper weight initialization
        self._init_weights()

    def _init_weights(self):
        # Initialize attention and projection weights according to GPT2 paper
        for name, p in self.named_parameters():
            if p.dim() > 1:  # weights, not biases
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'bias' in name:
                nn.init.zeros_(p)
            elif 'ln' in name and 'weight' in name:
                nn.init.ones_(p)

    def _create_causal_mask(self, seq_len: int, device: torch.device):
        """
        Create a causal attention mask for autoregressive generation.
        This mask ensures each position can only attend to previous positions.

        Args:
            seq_len: Length of the sequence
            device: Device to create the mask on

        Returns:
            A causal attention mask for self-attention
        """
        # Lower triangular matrix (incl. diagonal) are 0s, upper triangular are 1s
        mask = torch.triu(torch.ones(seq_len, seq_len,
                          device=device), diagonal=1).bool()
        return mask

    def forward(self, x, causal_mask=True):
        # First normalization and attention with causal masking (pre-norm architecture)
        normed_x = self.ln_1(x)

        # Create causal mask if requested
        attn_mask = None
        if causal_mask:
            seq_len = x.size(1)
            # Create causal mask for autoregressive generation
            attn_mask = self._create_causal_mask(seq_len, x.device)

        # Apply attention with optional causal mask
        attn_output, _ = self.attn(
            normed_x, normed_x, normed_x, attn_mask=attn_mask)
        x = x + attn_output  # Residual connection

        # Second normalization and MLP
        normed_x = self.ln_2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output  # Residual connection

        return x


class CausalGPT2Block(nn.Module):
    """
    Complete GPT2 transformer block with explicit causal masking support.
    This block enforces autoregressive constraints for true autoregressive generation.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_block = CausalGPT2AttentionBlock(dim, num_heads, dropout)

    def forward(self, x, causal_mask=True):
        return self.attn_block(x, causal_mask=causal_mask)
