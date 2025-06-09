# 파일 경로: model/predictor/gpt_backbone.py (이 파일의 내용을 아래 코드로 완전히 교체하세요)

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import math

# --- 어텐션 모듈에 가중치 초기화 로직 추가 ---


class GroupedQueryAttention(nn.Module):
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
        self.dropout_p = dropout

        # ❗️❗️핵심 수정: PyTorch의 nn.MultiheadAttention과 동일한 초기화 수행
        self._reset_parameters()

    def _reset_parameters(self):
        # nn.Linear의 기본 초기화(Kaiming) 대신 Xavier Uniform으로 재설정
        nn.init.xavier_uniform_(self.wq.weight)
        nn.init.xavier_uniform_(self.wk.weight)
        nn.init.xavier_uniform_(self.wv.weight)
        if self.wq.bias is not None:
            nn.init.constant_(self.wq.bias, 0.)
            nn.init.constant_(self.wk.bias, 0.)
            nn.init.constant_(self.wv.bias, 0.)
        if self.wo.bias is not None:
            nn.init.constant_(self.wo.bias, 0.)

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

# --- PyTorch 기본 라이브러리를 1:1로 복제한 트랜스포머 ---


class PerfectReplicaEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int,
                 dim_feedforward: int, dropout: float,
                 activation: str = 'gelu', layernorm_eps: float = 1e-5):
        super().__init__()
        self.num_layers = num_layers

        self.attentions = nn.ModuleList()
        self.ffns = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        for _ in range(num_layers):
            self.attentions.append(GroupedQueryAttention(
                d_model, nhead, nhead, dropout))
            self.ffns.append(nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Linear(dim_feedforward, d_model)
            ))
            self.norms1.append(nn.LayerNorm(d_model, eps=layernorm_eps))
            self.norms2.append(nn.LayerNorm(d_model, eps=layernorm_eps))
            self.dropouts.append(nn.Dropout(dropout))

        self.final_norm = nn.LayerNorm(d_model, eps=layernorm_eps)

    def forward(self, src: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        output = src
        for i in range(self.num_layers):
            attn_output = self.attentions[i](self.norms1[i](output), mask=mask)
            output = output + self.dropouts[i](attn_output)

            ffn_output = self.ffns[i](self.norms2[i](output))
            output = output + self.dropouts[i](ffn_output)

        return self.final_norm(output)
