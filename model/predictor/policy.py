#!/usr/bin/env python3
"""
Bidirectional Autoregressive Transformer for image-conditioned trajectory generation.

This model implements the following pipeline with SOFT GOAL CONDITIONING and GLOBAL HISTORY CONDITIONING:
1. Input: sequence of initial images i_{t-k:t} and states st_{t-k:t} (n_obs_steps history)
2. Encode and flatten history into a single global_history_condition_embedding.
3. Using this global_history_condition_embedding:
    a. Generate goal image i_n (first prediction)
    b. Generate backward states st_n ... (conditioned on global history + goal)
    c. Generate forward states st_0 ... (conditioned on global history + goal + backward path)

The new prediction order (goal → backward → forward) enables soft conditioning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional

from model.modules.modules import SpatialSoftmax
from model.modules.custom_transformer import RMSNorm, ReplicaTransformerEncoderLayer, ReplicaTransformerEncoder
from model.predictor.config import BidirectionalARTransformerConfig
from model.modules.component_blocks import InputBlock, OutputHeadBlock
from model.modules.visual_modules import ImageEncoder, ImageDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, Any

# 필요한 다른 모듈들을 import 합니다.
# 이 클래스들은 별도의 파일(예: visual_modules.py)에 있어도 무방합니다.
from model.modules.visual_modules import ImageEncoder, ImageDecoder


def compute_loss(predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
    """모델의 예측값과 정답 값을 받아 전체 Loss를 계산합니다."""
    losses = {}
    weights = {
        'forward_state_loss': 1.0, 'backward_state_loss': 1.0,
        'goal_image_loss': 1.0, 'progress_loss': 0.5
    }

    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        losses['forward_state_loss'] = F.l1_loss(
            predictions['predicted_forward_states'], targets['forward_states'])
    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'], targets['goal_images'])
    if 'predicted_progress' in predictions and 'normalized_timestep' in targets:
        predicted = predictions['predicted_progress'].squeeze(-1)
        target = targets['normalized_timestep']
        losses['progress_loss'] = F.mse_loss(predicted, target)

    total_loss = torch.tensor(0.0, device=next(
        iter(predictions.values())).device)
    for name, loss in losses.items():
        if name in weights and loss is not None:
            total_loss += loss * weights.get(name, 1.0)

    return total_loss

# --- 2. 최종 정책 모델 ---


class HierarchicalAutoregressivePolicy(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, **kwargs):
        super().__init__()
        self.config = config

        # --- 모듈 초기화 ---
        self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.goal_latent_reprojection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)
        self.bwd_summary_projection = nn.Linear(
            config.hidden_dim, config.hidden_dim)

        # --- 인코더와 디코더 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads, dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.hidden_dim))

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads, dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True)
        self.prediction_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers, norm=nn.LayerNorm(config.hidden_dim))

        # --- 쿼리 토큰 및 임베딩 ---
        self.history_pos_embedding = nn.Embedding(
            config.n_obs_steps * 2, config.hidden_dim)
        self.query_embedding = nn.Embedding(
            3, config.hidden_dim)  # 0:Goal, 1:Bwd, 2:Fwd

        # --- 출력 헤드 ---
        self.goal_head = nn.Linear(config.hidden_dim, config.image_latent_dim)
        self.bwd_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)
        self.fwd_head = nn.Linear(
            config.hidden_dim, config.forward_steps * config.state_dim)
        self.progress_head = nn.Linear(config.hidden_dim, 1)

    def encode(self, initial_images, initial_states):
        """과거 이력을 인코딩하여 memory를 생성합니다."""
        device = initial_images.device
        batch_size, n_obs, _, _, _ = initial_images.shape
        img_embeds = self.image_encoder(
            initial_images.flatten(0, 1)).view(batch_size, n_obs, -1)
        state_embeds = self.state_projection(initial_states)
        history_sequence = torch.cat([img_embeds, state_embeds], dim=1)

        pos_ids = torch.arange(
            history_sequence.shape[1], device=device).unsqueeze(0)
        history_sequence += self.history_pos_embedding(pos_ids)

        return self.context_encoder(history_sequence)

    def forward(self, initial_images, initial_states, **kwargs) -> Dict[str, torch.Tensor]:
        """
        학습을 위한 forward 함수. 한 번의 pass로 모든 예측을 효율적으로 처리합니다.
        """
        device = initial_images.device
        batch_size = initial_images.shape[0]

        memory = self.encode(initial_images, initial_states)

        query_ids = torch.arange(3, device=device).unsqueeze(
            0).expand(batch_size, -1)
        query_embeds = self.query_embedding(query_ids)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            query_embeds.size(1)).to(device)

        decoder_output = self.prediction_decoder(
            tgt=query_embeds, memory=memory, tgt_mask=tgt_mask)

        goal_h, bwd_h, fwd_h = decoder_output[:,
                                              0], decoder_output[:, 1], decoder_output[:, 2]

        predictions = {
            "predicted_goal_latents": self.goal_head(goal_h),
            "predicted_backward_states": self.bwd_head(bwd_h).view(batch_size, self.config.backward_steps, -1),
            "predicted_forward_states": self.fwd_head(fwd_h).view(batch_size, self.config.forward_steps, -1),
            "predicted_progress": torch.sigmoid(self.progress_head(memory.mean(dim=1)))
        }
        predictions["predicted_goal_images"] = self.image_decoder(
            predictions["predicted_goal_latents"])

        return predictions

    @torch.no_grad()
    def generate(self, initial_images, initial_states) -> Dict[str, torch.Tensor]:
        """추론을 위한 generate 함수. Goal -> Bwd -> Fwd 순서로 순차 생성합니다."""
        self.eval()
        device = initial_images.device

        memory = self.encode(initial_images, initial_states)

        # 단계 1: Goal 예측
        goal_query = self.query_embedding.weight[0].unsqueeze(0).unsqueeze(0)
        goal_h = self.prediction_decoder(
            tgt=goal_query, memory=memory).squeeze(1)
        pred_goal_latent = self.goal_head(goal_h)
        pred_goal_image = self.image_decoder(pred_goal_latent)

        # 단계 2: Bwd 예측
        goal_result_embed = self.goal_latent_reprojection(
            pred_goal_latent).unsqueeze(1)
        bwd_query = self.query_embedding.weight[1].unsqueeze(0).unsqueeze(0)
        bwd_tgt = torch.cat([goal_result_embed, bwd_query], dim=1)
        bwd_mask = nn.Transformer.generate_square_subsequent_mask(
            bwd_tgt.size(1)).to(device)
        bwd_h = self.prediction_decoder(
            tgt=bwd_tgt, memory=memory, tgt_mask=bwd_mask)[:, -1, :]
        pred_bwd_states = self.bwd_head(bwd_h).view(
            1, self.config.backward_steps, -1)

        # 단계 3: Fwd 예측
        bwd_summary = self.state_projection(pred_bwd_states.mean(dim=1))
        bwd_result_embed = self.bwd_summary_projection(
            bwd_summary).unsqueeze(1)
        fwd_query = self.query_embedding.weight[2].unsqueeze(0).unsqueeze(0)
        fwd_tgt = torch.cat(
            [goal_result_embed, bwd_result_embed, fwd_query], dim=1)
        fwd_mask = nn.Transformer.generate_square_subsequent_mask(
            fwd_tgt.size(1)).to(device)
        fwd_h = self.prediction_decoder(
            tgt=fwd_tgt, memory=memory, tgt_mask=fwd_mask)[:, -1, :]
        pred_fwd_states = self.fwd_head(fwd_h).view(
            1, self.config.forward_steps, -1)

        return {
            "predicted_goal_images": pred_goal_image,
            "predicted_backward_states": pred_bwd_states,
            "predicted_forward_states": pred_fwd_states
        }
