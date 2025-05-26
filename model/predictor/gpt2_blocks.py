"""
GPT2-style blocks for transformer architectures.
These blocks implement the standard GPT2 architecture components.
"""
import torch.nn as nn


class GPT2MLP(nn.Module):
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


class GPT2AttentionBlock(nn.Module):
    """
    GPT2-style self-attention block with residual connection and layer normalization.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        # Pre-normalization for better training stability
        self.ln_1 = nn.LayerNorm(dim)

        # Self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Feed-forward network
        self.ln_2 = nn.LayerNorm(dim)
        self.mlp = GPT2MLP(dim, dropout=dropout)

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

    def forward(self, x):
        # First normalization and attention (pre-norm architecture)
        normed_x = self.ln_1(x)
        attn_output, _ = self.attn(normed_x, normed_x, normed_x)
        x = x + attn_output  # Residual connection

        # Second normalization and MLP
        normed_x = self.ln_2(x)
        mlp_output = self.mlp(normed_x)
        x = x + mlp_output  # Residual connection

        return x


class GPT2Block(nn.Module):
    """
    Complete GPT2 transformer block.
    """

    def __init__(self, dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn_block = GPT2AttentionBlock(dim, num_heads, dropout)

    def forward(self, x):
        return self.attn_block(x)
