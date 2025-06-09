import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math
import copy

# --- 헬퍼(Helper) 모듈 ---


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

    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


# --- PyTorch 기본 라이브러리를 1:1로 복제한 Layer와 Encoder ---

class ReplicaTransformerEncoderLayer(nn.Module):
    """
    PyTorch의 nn.TransformerEncoderLayer(norm_first=True)와
    거의 동일하게 동작하도록 만든 '복제' 레이어입니다.

    가장 큰 특징은 내부적으로 nn.MultiheadAttention을 직접 사용하여,
    '보이지 않는' 가중치 초기화 방식까지 동일하게 맞춘 것입니다.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048,
                 dropout: float = 0.1, activation: str = "gelu",
                 layernorm_eps: float = 1e-5, batch_first: bool = True):
        super().__init__()

        # ❗️ 핵심: 커스텀 어텐션 대신 PyTorch의 MultiheadAttention을 사용
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=batch_first  # (batch, seq, feature) 순서
        )

        # FFN (Feed-Forward Network) - SwiGLU로 교체
        # SwiGLU의 hidden_dim은 보통 2/3 * dim_feedforward 규칙을 따름
        swiglu_hidden_dim = int(2 * dim_feedforward / 3)
        self.ffn = SwiGLU(d_model, swiglu_hidden_dim, dropout=dropout)

        # Pre-LN 구조를 위한 Layer Normalization
        self.norm1 = RMSNorm(d_model, eps=layernorm_eps)
        self.norm2 = RMSNorm(d_model, eps=layernorm_eps)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # PyTorch의 norm_first=True 로직과 완벽히 동일한 데이터 흐름

        # 1. Self-Attention 블록
        # MultiheadAttention은 Q, K, V를 모두 src로 받음
        norm_src = self.norm1(src)
        attn_output, _ = self.self_attn(
            norm_src, norm_src, norm_src, attn_mask=src_mask, need_weights=False)
        src = src + self.dropout1(attn_output)

        # 2. FFN 블록 - SwiGLU 사용
        norm_src = self.norm2(src)
        ffn_output = self.ffn(norm_src)
        src = src + self.dropout2(ffn_output)

        return src


class ReplicaTransformerEncoder(nn.Module):
    """
    ReplicaTransformerEncoderLayer를 N개 쌓아서 만든 전체 인코더.
    PyTorch의 nn.TransformerEncoder와 사용법 및 구조가 동일합니다.
    """

    def __init__(self, encoder_layer: nn.Module, num_layers: int, norm: Optional[nn.Module] = None):
        super().__init__()
        # encoder_layer를 num_layers 만큼 깊은 복사(deep copy)하여 사용
        self.layers = nn.ModuleList(
            [copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask)

        if self.norm is not None:
            output = self.norm(output)

        return output
