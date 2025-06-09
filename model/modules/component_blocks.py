import torch
import torch.nn as nn
from typing import Dict

from model.modules.visual_modules import ImageEncoder, ImageDecoder
from model.predictor.config import HierarchicalPolicyConfig


class InputBlock(nn.Module):
    """이미지와 상태 입력을 받아 hidden_dim의 임베딩으로 변환합니다."""

    def __init__(self, config: HierarchicalPolicyConfig):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

    def forward(self, initial_images: torch.Tensor, initial_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, n_obs, _, H, W = initial_images.shape
        img_hist_flat = initial_images.view(
            batch_size * n_obs, self.config.image_channels, H, W)
        img_latents = self.image_encoder(img_hist_flat).view(
            batch_size, n_obs, self.config.image_latent_dim)
        img_embeds = self.image_latent_projection(img_latents)

        states_embeds = self.state_projection(initial_states)
        return img_embeds, states_embeds


class OutputHeadBlock(nn.Module):
    """트랜스포머의 출력을 받아 최종 예측값을 생성합니다."""

    def __init__(self, config: HierarchicalPolicyConfig):
        super().__init__()
        self.config = config
        self.image_decoder = ImageDecoder(config)
        self.progress_head = nn.Sequential(nn.Linear(
            config.hidden_dim, config.hidden_dim // 2), nn.ReLU(), nn.Linear(config.hidden_dim // 2, 1), nn.Sigmoid())
        self.forward_state_head = nn.Linear(
            config.hidden_dim, config.forward_steps * config.state_dim)
        self.goal_image_latent_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)
        self.backward_state_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, _, _ = hidden_states.shape
        num_hist_tokens = self.config.n_obs_steps * 2

        goal_query_output = hidden_states[:, num_hist_tokens]
        bwd_query_output = hidden_states[:, num_hist_tokens + 1]
        fwd_query_output = hidden_states[:, num_hist_tokens + 2]

        results = {}
        predicted_goal_latents = self.goal_image_latent_head(goal_query_output)
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)

        predicted_bwd_states = self.backward_state_head(bwd_query_output).view(
            batch_size, self.config.backward_steps, self.config.state_dim)
        results['predicted_backward_states'] = predicted_bwd_states

        predicted_fwd_states = self.forward_state_head(fwd_query_output).view(
            batch_size, self.config.forward_steps, self.config.state_dim)
        results['predicted_forward_states'] = predicted_fwd_states

        avg_history_embedding = torch.mean(
            hidden_states[:, :num_hist_tokens], dim=1)
        results['predicted_progress'] = self.progress_head(
            avg_history_embedding)
        return results
