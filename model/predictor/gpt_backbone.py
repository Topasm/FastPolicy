# 파일 경로: model/predictor/gpt_backbone.py (또는 유사 파일)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class SwiGLU(nn.Module):
    """SwiGLU activation function"""

    def __init__(self, d_model: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GroupedQueryAttention(nn.Module):
    """Grouped Query Attention"""

    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int = None, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=True)
        self.wk = nn.Linear(d_model, self.n_kv_heads *
                            self.head_dim, bias=True)
        self.wv = nn.Linear(d_model, self.n_kv_heads *
                            self.head_dim, bias=True)
        self.wo = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout  # Changed to store probability directly

    def repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.n_rep == 1:
            return x
        bs, n_kv_heads, seq_len, head_dim = x.shape
        return x[:, :, None, :, :].expand(bs, n_kv_heads, self.n_rep, seq_len, head_dim).reshape(bs, n_kv_heads * self.n_rep, seq_len, head_dim)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.wq(x).view(batch_size, seq_len, self.n_heads,
                            self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads,
                            self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads,
                            self.head_dim).transpose(1, 2)

        k = self.repeat_kv(k)
        v = self.repeat_kv(v)

        attn_output = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, dropout_p=self.dropout_p if self.training else 0.0)

        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model)
        return self.wo(attn_output)


class ModernTransformerEncoder(nn.Module):
    """
    Modern transformer with RMSNorm, SwiGLU, GQA.
    It expects the input `src` to have position information already added.
    """

    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float,
                 activation: str = 'swiglu', layernorm_eps: float = 1e-6,
                 use_gqa: bool = True, n_kv_heads: int = None):
        super().__init__()

        # Store num_layers as an instance variable
        self.num_layers = num_layers

        # RoPE 관련 초기화 코드를 모두 제거했습니다.

        if use_gqa and n_kv_heads is not None:
            self.attentions = nn.ModuleList([GroupedQueryAttention(
                d_model, nhead, n_kv_heads, dropout) for _ in range(num_layers)])
        else:
            self.attentions = nn.ModuleList([GroupedQueryAttention(
                d_model, nhead, nhead, dropout) for _ in range(num_layers)])

        if activation.lower() == 'swiglu':
            swiglu_hidden_dim = int(2 * dim_feedforward / 3)
            self.ffns = nn.ModuleList(
                [SwiGLU(d_model, swiglu_hidden_dim) for _ in range(num_layers)])
        else:
            act_fn = nn.GELU() if activation.lower() == 'gelu' else nn.ReLU()
            self.ffns = nn.ModuleList([nn.Sequential(
                nn.Linear(d_model, dim_feedforward,
                          bias=False), act_fn, nn.Dropout(dropout),
                nn.Linear(dim_feedforward, d_model, bias=False)) for _ in range(num_layers)])

        self.norm1s = nn.ModuleList(
            [RMSNorm(d_model, eps=layernorm_eps) for _ in range(num_layers)])
        self.norm2s = nn.ModuleList(
            [RMSNorm(d_model, eps=layernorm_eps) for _ in range(num_layers)])
        self.dropout1s = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_layers)])
        self.dropout2s = nn.ModuleList(
            [nn.Dropout(dropout) for _ in range(num_layers)])
        self.final_norm = RMSNorm(d_model, eps=layernorm_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for i in range(self.num_layers):
            norm_output = self.norm1s[i](output)
            # 어텐션 호출 시 RoPE 인자 제거
            attn_output = self.attentions[i](norm_output, mask=mask)
            output = output + self.dropout1s[i](attn_output)

            norm_output = self.norm2s[i](output)
            ffn_output = self.ffns[i](norm_output)
            output = output + self.dropout2s[i](ffn_output)

        return self.final_norm(output)


class FinalTransformerEncoder(nn.Module):
    """
    PyTorch nn.TransformerEncoderLayer(norm_first=True)와 동일한 데이터 흐름을 가지는
    현대적인 트랜스포머 인코더 구현체입니다.
    """

    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float,
                 activation: str = 'swiglu', layernorm_eps: float = 1e-6,
                 use_gqa: bool = True, n_kv_heads: int = None):
        super().__init__()
        self.num_layers = num_layers

        self.attentions = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norm1s = nn.ModuleList()
        self.norm2s = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            if use_gqa and n_kv_heads is not None:
                self.attentions.append(GroupedQueryAttention(
                    d_model, nhead, n_kv_heads, dropout))
            else:
                self.attentions.append(GroupedQueryAttention(
                    d_model, nhead, nhead, dropout))

            if activation.lower() == 'swiglu':
                swiglu_hidden_dim = int(2 * dim_feedforward / 3)
                self.ffns.append(SwiGLU(d_model, swiglu_hidden_dim))
            else:
                act_fn = nn.GELU() if activation.lower() == 'gelu' else nn.ReLU()
                self.ffns.append(nn.Sequential(
                    nn.Linear(d_model, dim_feedforward, bias=False),
                    act_fn,
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model, bias=False)
                ))

            self.norm1s.append(RMSNorm(d_model, eps=layernorm_eps))
            self.norm2s.append(RMSNorm(d_model, eps=layernorm_eps))
            # Single dropout per layer for residual connections
            self.dropouts.append(nn.Dropout(dropout))

        self.final_norm = RMSNorm(d_model, eps=layernorm_eps)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Parameter):
            torch.nn.init.normal_(module, mean=0.0, std=0.02)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for i in range(self.num_layers):
            # Pre-LN structure: norm -> sublayer -> residual
            # Attention block
            attn_output = self.attentions[i](self.norm1s[i](output), mask=mask)
            output = output + self.dropouts[i](attn_output)

            # FFN block
            ffn_output = self.ffns[i](self.norm2s[i](output))
            output = output + self.dropouts[i](ffn_output)

        return self.final_norm(output)
