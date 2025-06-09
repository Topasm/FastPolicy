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
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
import torchvision.models as models
import torchvision.transforms as transforms

from model.modules.modules import SpatialSoftmax
from model.modules.custom_transformer import RMSNorm, ReplicaTransformerEncoderLayer, ReplicaTransformerEncoder
from model.predictor.config import BidirectionalARTransformerConfig
from model.modules.component_blocks import InputBlock, OutputHeadBlock
from model.modules.visual_modules import ImageEncoder, ImageDecoder


class GoalConditionedAutoregressivePolicy(nn.Module):
    """
    Encoder-Decoder 구조를 사용하여 목표를 먼저 예측하고,
    이를 조건으로 궤적을 순차적으로 생성하는 최종 모델입니다.
    """

    def __init__(self, config: BidirectionalARTransformerConfig, **kwargs):
        super().__init__()
        self.config = config

        # --- 1. 입력 처리 모듈 ---
        self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # --- 2. 위치 및 타입 임베딩 ---
        # 이력, 목표, 생성될 궤적을 위한 임베딩
        self.history_pos_embedding = nn.Embedding(
            config.n_obs_steps * 2, config.hidden_dim)
        self.trajectory_pos_embedding = nn.Embedding(
            config.forward_steps + config.backward_steps, config.hidden_dim)
        # 0:ImgHist, 1:StateHist, 2:Goal, 3:StateTraj
        self.token_type_embedding = nn.Embedding(4, config.hidden_dim)

        # --- 3. 인코더와 디코더 (PyTorch 기본 모듈 사용) ---
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

        # --- 4. 출력 헤드 ---
        self.goal_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(
        ), nn.Linear(config.hidden_dim, config.image_latent_dim))
        self.next_state_head = nn.Linear(config.hidden_dim, config.state_dim)

        print("✅ Initialized Goal-Conditioned Encoder-Decoder Policy.")

    def encode(self, initial_images, initial_states):
        """과거 이력을 인코딩하여 memory를 생성합니다."""
        device = initial_images.device
        batch_size, n_obs, _, _, _ = initial_images.shape

        # 이력 임베딩
        img_embeds = self.image_encoder(
            initial_images.flatten(0, 1)).view(batch_size, n_obs, -1)
        img_embeds = self.image_latent_projection(img_embeds)
        state_embeds = self.state_projection(initial_states)

        # 이력 시퀀스 구성 및 임베딩
        history_sequence = torch.cat([img_embeds, state_embeds], dim=1)

        hist_pos_ids = torch.arange(n_obs * 2, device=device).unsqueeze(0)
        history_sequence += self.history_pos_embedding(hist_pos_ids)

        hist_type_ids = torch.cat([torch.full((n_obs,), 0, device=device), torch.full(
            (n_obs,), 1, device=device)], dim=0).unsqueeze(0)
        history_sequence += self.token_type_embedding(hist_type_ids)

        # 인코더 통과
        return self.context_encoder(history_sequence)

    def forward(self, initial_images, initial_states, goal_images=None, forward_states=None, backward_states=None, training=True, **kwargs):
        """학습(Training)을 위한 forward 함수"""
        # 1. 인코더로 이력 문맥(memory) 생성
        memory = self.encode(initial_images, initial_states)

        # 2. 목표(Goal) 예측 및 Loss 계산
        # memory의 모든 정보를 종합하여(mean) 목표 예측
        memory_summary = memory.mean(dim=1)
        predicted_goal_latent = self.goal_head(memory_summary)

        if goal_images is not None:
            with torch.no_grad():
                true_goal_latent = self.image_encoder(goal_images)
            goal_loss = F.mse_loss(predicted_goal_latent, true_goal_latent)
        else:
            goal_loss = torch.tensor(0.0, device=initial_images.device)

        # 3. 궤적(Trajectory) 예측 및 Loss 계산 (Teacher Forcing)
        if forward_states is not None and backward_states is not None:
            # 디코더 입력(tgt)으로 사용할 정답 궤적 준비
            target_states = torch.cat(
                [torch.flip(backward_states, [1]), forward_states], dim=1)

            # 디코더 입력의 시작을 알리는 [SOS] 토큰 역할로 예측된 목표를 사용
            goal_embed = self.image_latent_projection(
                predicted_goal_latent.detach()).unsqueeze(1)  # gradient 흐름 차단

            # 정답 궤적 state를 임베딩
            target_embeds = self.state_projection(target_states)

            # 최종 디코더 입력: [예측된 Goal, 정답 궤적의 첫 스텝 ~ 마지막-1 스텝]
            decoder_input = torch.cat(
                [goal_embed, target_embeds[:, :-1, :]], dim=1)

            # 위치 및 타입 임베딩 추가
            traj_pos_ids = torch.arange(
                decoder_input.shape[1], device=initial_images.device).unsqueeze(0)
            decoder_input += self.trajectory_pos_embedding(traj_pos_ids)

            traj_type_ids = torch.full(
                (decoder_input.shape[1],), 3, device=initial_images.device).unsqueeze(0)
            decoder_input += self.token_type_embedding(traj_type_ids)

            # Causal Mask와 함께 디코더 통과
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                decoder_input.shape[1]).to(initial_images.device)
            decoder_output = self.prediction_decoder(
                tgt=decoder_input, memory=memory, tgt_mask=tgt_mask)

            # 다음 스텝 상태 예측
            predicted_trajectory = self.next_state_head(decoder_output)

            # 궤적 Loss 계산
            trajectory_loss = F.l1_loss(predicted_trajectory, target_states)

            # 최종 Loss
            total_loss = goal_loss + trajectory_loss
            return total_loss
        else:
            return goal_loss

    @torch.no_grad()
    def generate(self, initial_images, initial_states, steps_to_generate):
        """추론(Inference)을 위한 Autoregressive 생성 함수"""
        self.eval()
        device = initial_images.device

        # 1. 인코더로 memory 생성
        memory = self.encode(initial_images, initial_states)

        # 2. 목표(Goal) 예측
        predicted_goal_latent = self.goal_head(memory.mean(dim=1))
        predicted_goal_image = self.image_decoder(predicted_goal_latent)

        # 3. Autoregressive 궤적 생성 시작
        # 디코더의 첫 입력([SOS] 토큰)으로 예측된 목표 사용
        current_token_embed = self.image_latent_projection(
            predicted_goal_latent).unsqueeze(1)
        generated_states = []

        for i in range(steps_to_generate):
            pos_id = torch.tensor([[i]], device=device)
            # Trajectory state type
            type_id = torch.tensor([[3]], device=device)

            decoder_input = current_token_embed + \
                self.trajectory_pos_embedding(
                    pos_id) + self.token_type_embedding(type_id)

            # 디코더는 매 스텝 전체 memory를 참고
            decoder_output = self.prediction_decoder(
                tgt=decoder_input, memory=memory)

            # 다음 state 예측
            next_state = self.next_state_head(decoder_output.squeeze(1))
            generated_states.append(next_state)

            # 다음 입력을 위해 예측된 state를 다시 임베딩
            current_token_embed = self.state_projection(
                next_state).unsqueeze(1)

        return torch.stack(generated_states, dim=1), predicted_goal_image
