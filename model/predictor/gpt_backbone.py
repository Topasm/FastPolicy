# 파일 경로: model/predictor/gpt_backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from torch import Tensor


class RotaryEmbedding(nn.Module):
    """
    Rotary Positional Embedding (RoPE).
    - 절대 위치 임베딩(nn.Embedding)을 대체하여 위치 정보를 어텐션 메커니즘에 직접 주입합니다.
    - 시퀀스 길이에 대한 일반화 성능이 더 좋다고 알려져 있습니다.
    """

    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._set_cos_sin_cache(max_seq_len)

    def _set_cos_sin_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device,
                         dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cache", emb.cos()[
                             None, :, None, :], persistent=False)
        self.register_buffer("sin_cache", emb.sin()[
                             None, :, None, :], persistent=False)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len)
        return (
            self.cos_cache[:, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cache[:, :seq_len, ...].to(dtype=x.dtype),
        )


def apply_rotary_emb(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) -> Tuple[Tensor, Tensor]:
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


def _rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class Attention(nn.Module):
    """Multi-Head Attention with Rotary Positional Embedding."""

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, rotary_emb: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len, _ = x.shape

        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # Reshape queries, keys, values for multi-head attention
        q = q.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head,
                   self.head_dim).transpose(1, 2)

        # Apply rotary embeddings - ensure dimensions match
        cos, sin = rotary_emb
        # Permute cos and sin to match q,k dimensions after transpose
        cos = cos.permute(0, 2, 1, 3)  # [1, 1, S, D] -> [1, S, 1, D]
        sin = sin.permute(0, 2, 1, 3)  # [1, 1, S, D] -> [1, S, 1, D]
        q, k = apply_rotary_emb(q, k, cos[:, :seq_len], sin[:, :seq_len])

        # Use efficient scaled dot-product attention
        # Handle mask properly for scaled_dot_product_attention
        if mask is not None:
            # PyTorch expects a 4D attention mask [B, H, T, S]
            if mask.dim() == 3:  # [B, T, S]
                mask = mask.unsqueeze(1)  # Add head dimension [B, 1, T, S]

        output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.attn_dropout.p if self.training else 0.0)

        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network.
    - ReLU 대신 SwiGLU 활성화를 사용하여 성능을 높이는 최신 FFN 구조입니다.
    """

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """
    A transformer block with Pre-Layer Normalization.
    - LayerNorm을 어텐션와 FFN '이후'가 아닌 '이전'에 적용하여 학습 안정성을 높입니다.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        self.n_head = n_head
        self.d_model = d_model

        self.attention = Attention(d_model, n_head, dropout)
        self.feed_forward = SwiGLUFeedForward(
            dim=d_model, hidden_dim=d_model * 4, dropout=dropout)
        self.attention_norm = nn.LayerNorm(d_model)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward(self, x: Tensor, rotary_emb: Tuple[Tensor, Tensor], mask: Optional[Tensor] = None) -> Tensor:
        # Pre-LN: x + SubLayer(LayerNorm(x))
        x = x + self.attention(self.attention_norm(x), rotary_emb, mask)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x


class GPTBackbone(nn.Module):
    """The main transformer backbone composed of transformer blocks."""

    def __init__(self, d_model: int, n_head: int, n_layers: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock(
            d_model, n_head, dropout) for _ in range(n_layers)])
        self.rotary_embeddings = RotaryEmbedding(dim=d_model // n_head)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # Get proper sequence length from input
        seq_len = x.shape[1]

        # Ensure rotary embeddings match the sequence length
        rotary_emb = self.rotary_embeddings(x)

        # Handle mask format - convert boolean mask to attention mask format
        # In PyTorch attention, True = masked position
        if mask is not None:
            # We'll invert our boolean mask if needed - in our case True means masked
            # For GPT models with causal attention or custom attention patterns

            # First ensure mask is proper shape for transformer layers
            if mask.dim() == 2:  # [S, S]
                # Add batch and head dimensions for broadcasting
                mask = mask.unsqueeze(0)  # [1, S, S]

        for layer in self.layers:
            x = layer(x, rotary_emb, mask)
        return x
