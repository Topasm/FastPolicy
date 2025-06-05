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
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from pathlib import Path
from lerobot.configs.types import NormalizationMode
# from lerobot.configs.policies import PreTrainedConfig
# from lerobot.common.optim.optimizers import AdamConfig
# from lerobot.common.optim.schedulers import DiffuserSchedulerConfig


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-5
    image_latent_dim: int = 256  # Latent dimension for image features
    image_channels: int = 3
    image_size: int = 96
    forward_steps: int = 20
    backward_steps: int = 16
    n_obs_steps: int = 2  # Number of observation steps in history
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    # Number of pure query tokens (goal, backward, forward)
    num_query_tokens: int = 3

    # Number of action steps (not used in this model, but kept for compatibility)
    n_action_steps = 8

    image_features = 1
    # Token types: HistStep, QueryGoal, QueryBwd, QueryFwd
    token_type_count: int = 4

    def to_dict(self):
        def feature_to_dict(feat):
            if hasattr(feat, 'to_dict'):
                return feat.to_dict()
            if hasattr(feat, '__dataclass_fields__'):
                return asdict(feat)
            return str(feat)
        d = asdict(self)
        d["input_features"] = {k: feature_to_dict(
            v) for k, v in self.input_features.items()}
        d["output_features"] = {k: feature_to_dict(
            v) for k, v in self.output_features.items()}
        return d

    def save_pretrained(self, output_dir: Path):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def from_pretrained(cls, output_dir: Path):
        from lerobot.common.datasets.utils import PolicyFeature  # PolicyFeature 클래스 임포트
        from lerobot.configs.types import FeatureType           # FeatureType enum 임포트

        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        # input_features를 PolicyFeature 객체로 복원
        if "input_features" in config_dict and isinstance(config_dict["input_features"], dict):
            reconstructed_input_features = {}
            for k, v_dict in config_dict["input_features"].items():
                if isinstance(v_dict, dict) and "shape" in v_dict and "type" in v_dict:
                    try:
                        feature_type_str = v_dict["type"]
                        actual_type_name = feature_type_str.split(
                            '.')[-1] if '.' in feature_type_str else feature_type_str
                        feature_type_val = FeatureType[actual_type_name]

                        reconstructed_input_features[k] = PolicyFeature(
                            shape=tuple(v_dict["shape"]),
                            type=feature_type_val
                        )
                    except Exception as e:
                        print(
                            f"Warning: Could not reconstruct PolicyFeature for input_features key {k} from dict {v_dict}: {e}. Keeping as dict.")
                        reconstructed_input_features[k] = v_dict
                else:
                    reconstructed_input_features[k] = v_dict
            config_dict["input_features"] = reconstructed_input_features

        # output_features도 필요한 경우 동일한 방식으로 복원
        if "output_features" in config_dict and isinstance(config_dict["output_features"], dict):
            reconstructed_output_features = {}
            for k, v_dict in config_dict["output_features"].items():
                if isinstance(v_dict, dict) and "shape" in v_dict and "type" in v_dict:
                    try:
                        feature_type_str = v_dict["type"]
                        actual_type_name = feature_type_str.split(
                            '.')[-1] if '.' in feature_type_str else feature_type_str
                        feature_type_val = FeatureType[actual_type_name]
                        reconstructed_output_features[k] = PolicyFeature(
                            shape=tuple(v_dict["shape"]),
                            type=feature_type_val
                        )
                    except Exception as e:
                        print(
                            f"Warning: Could not reconstruct PolicyFeature for output_features key {k} from dict {v_dict}: {e}. Keeping as dict.")
                        reconstructed_output_features[k] = v_dict
                else:
                    reconstructed_output_features[k] = v_dict
            config_dict["output_features"] = reconstructed_output_features

        return cls(**config_dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )


class ImageEncoder(nn.Module):  # Original Simple CNN Encoder
    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.encoder = nn.Sequential(
            nn.Conv2d(config.image_channels, 64, kernel_size=4, stride=2,
                      padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                256), nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                512), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                512), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(512, config.image_latent_dim), nn.ReLU()
        )

    def forward(
        self, images: torch.Tensor) -> torch.Tensor: return self.encoder(images)


class ImageDecoder(nn.Module):  # Remains the same
    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.initial_linear = nn.Sequential(
            nn.Linear(config.image_latent_dim, 512 * 3 * 3), nn.ReLU())
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                512), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                256), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), nn.BatchNorm2d(
                64), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, config.image_channels,
                               kernel_size=4, stride=2, padding=1), nn.Tanh()
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.initial_linear(latents)
        x = x.view(-1, 512, 3, 3)
        return self.decoder(x)


class BidirectionalARTransformer(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, state_key: str = "observation.state", image_key: str = "observation.image"):
        super().__init__()
        self.config = config
        self.state_key = state_key
        self.image_key = image_key

        from lerobot.configs.types import FeatureType
        self.feature_type = FeatureType  # For potential use, not directly used now

        # Use simple ImageEncoder
        print("Using ImageEncoder in BidirectionalARTransformer.")
        self.image_encoder = ImageEncoder(config)

        self.image_decoder = ImageDecoder(config)

        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        # Add back the image_latent_projection that was missing
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # 각 이력 스텝의 (이미지 잠재값 + 상태 임베딩)을 hidden_dim으로 투영하는 레이어
        self.history_step_feature_dim = config.image_latent_dim + config.hidden_dim
        self.history_step_projector = nn.Linear(
            self.history_step_feature_dim, config.hidden_dim)

        # 토큰 타입 임베딩 (예: 0:HistStep, 1:QueryGoal, 2:QueryBwd, 3:QueryFwd)
        # config.token_type_count 사용 (현재는 4로 하드코딩된 것과 유사)
        self.NUM_QUERY_TYPES = 3  # Goal, Bwd, Fwd queries
        self.TYPE_HIST_STEP = 0
        self.TYPE_QUERY_GOAL = 1
        self.TYPE_QUERY_BWD = 2
        self.TYPE_QUERY_FWD = 3

        # Calculate total sequence length for transformer
        self.total_seq_len = config.n_obs_steps + self.NUM_QUERY_TYPES

        # 실제 Embedding 크기는 사용될 타입의 총 개수 (여기서는 4)
        self.token_type_embedding = nn.Embedding(4, config.hidden_dim)

        # 위치 임베딩: 전체 시퀀스 길이에 맞춰야 함 (n_obs_steps + 3개의 쿼리 토큰)
        # config.query_seq_len을 순수 쿼리 토큰 수 (3)으로 해석하고, 전체 길이는 동적 계산
        self.num_queries = 3  # goal, bwd, fwd
        self.position_embedding = nn.Embedding(
            self.total_seq_len, config.hidden_dim)

        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

        # Output heads remain unchanged
        self.forward_state_head = nn.Linear(
            config.hidden_dim, (config.forward_steps - 1) * config.state_dim)
        self.goal_image_latent_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)
        self.backward_state_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4, dropout=config.dropout,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        self.apply(self._init_weights)

    def _init_weights(self, module):  # Standard init
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)
        elif isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def _create_full_history_sequential_mask(self, device: torch.device) -> torch.Tensor:
        """
        Creates a custom attention mask for the new sequence structure.

        The sequence now consists of:
        [Hist_1, Hist_2, ..., Hist_n, Goal_Q, Bwd_Q, Fwd_Q]

        With the following attention pattern:
        - Each token can attend to itself
        - All tokens can attend to history tokens
        - Goal query can only attend to history
        - Backward query can attend to history and goal query
        - Forward query can attend to all previous tokens
        """
        n_obs = self.config.n_obs_steps
        num_queries = 3  # Goal, Bwd, Fwd
        seq_len = n_obs + num_queries

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool,
                          device=device)  # True = cannot attend
        mask.fill_diagonal_(False)  # Attend to self

        # 1. 이력 토큰들 간의 어텐션: 여기서는 간단히 인과적으로 설정 (또는 모두 False로 하여 전체 어텐션)
        for i in range(n_obs):
            mask[i, i+1:n_obs] = True  # H_i는 H_j (j>i)에 어텐션 불가 (인과적)
            # 또는 이 부분을 mask[i, :i+1] = False 로 하여 H_i가 H_0..H_i에 어텐션하도록 할 수 있음
            # Seer처럼 전체 이력을 한 번에 처리하려면 이력 내에서는 서로 다 볼 수 있게 할 수도:
            # mask[0:n_obs, 0:n_obs] = False (대각선 제외하고는 다 False)
            # 여기서는 각 이력 토큰이 이전 이력 토큰만 보도록 인과적으로 설정
            if i > 0:
                mask[i, 0:i] = False

        # 2. 모든 쿼리 토큰은 모든 이력 토큰에 어텐션 가능
        mask[n_obs:, :n_obs] = False

        # 3. 쿼리 토큰들 간의 순차적 의존성
        # Q_goal (idx n_obs)은 이력에만 의존 (이미 위에서 설정됨)

        # Q_bwd (idx n_obs + 1)은 이력 및 Q_goal(n_obs)에 어텐션 가능
        mask[n_obs + 1, n_obs] = False

        # Q_fwd (idx n_obs + 2)은 이력, Q_goal(n_obs), Q_bwd(n_obs + 1)에 어텐션 가능
        mask[n_obs + 2, n_obs: n_obs + 2] = False

        return mask

    def _forward_training_with_sequential_history(
        self,
        history_tokens: torch.Tensor,  # [B, n_obs_steps, hidden_dim]
        forward_states_gt: torch.Tensor,
        goal_images_gt: torch.Tensor,
        backward_states_gt: torch.Tensor,
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training using observation history tokens directly.
        """
        batch_size = history_tokens.shape[0]
        n_obs = self.config.n_obs_steps
        results = {}

        # Prepare query tokens
        goal_query = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_query = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_query = self.forward_seq_query_token.expand(batch_size, -1, -1)

        # Concatenate tokens to form the input sequence
        # [B, n_obs_steps + 3, hidden_dim]
        input_sequence = torch.cat(
            [history_tokens, goal_query, bwd_query, fwd_query], dim=1)

        # Add token type embeddings
        # First n_obs positions are history steps
        hist_types = torch.full((batch_size, n_obs), self.TYPE_HIST_STEP,
                                device=device, dtype=torch.long)

        # Last 3 positions are query tokens
        query_types = torch.tensor(
            [self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD],
            device=device
        ).unsqueeze(0).expand(batch_size, -1)

        all_token_types = torch.cat([hist_types, query_types], dim=1)
        input_sequence += self.token_type_embedding(all_token_types)

        # Add position embeddings
        positions = torch.arange(self.total_seq_len, device=device).unsqueeze(
            0).expand(batch_size, -1)
        input_sequence += self.position_embedding(positions)

        # Apply attention mask and run through transformer
        attn_mask = self._create_full_history_sequential_mask(device)
        hidden_states = self.transformer(src=input_sequence, mask=attn_mask)

        # Extract query outputs for predictions
        goal_hidden = hidden_states[:, n_obs]      # Goal query position
        bwd_hidden = hidden_states[:, n_obs+1]     # Backward query position
        fwd_hidden = hidden_states[:, n_obs+2]     # Forward query position

        # Generate predictions (unchanged)
        results['predicted_goal_latents'] = self.goal_image_latent_head(
            goal_hidden)
        results['predicted_goal_images'] = self.image_decoder(
            results['predicted_goal_latents'])

        bwd_flat = self.backward_state_head(bwd_hidden)
        results['predicted_backward_states'] = bwd_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim)

        fwd_flat = self.forward_state_head(fwd_hidden)
        results['predicted_forward_states'] = fwd_flat.view(
            batch_size, self.config.forward_steps - 1, self.config.state_dim)

        return results

    def _forward_inference_with_sequential_history(self,
                                                   # [B, n_obs, hidden_dim]
                                                   history_step_embeddings: torch.Tensor,
                                                   device: torch.device
                                                   ) -> Dict[str, torch.Tensor]:
        batch_size = history_step_embeddings.shape[0]
        n_obs = self.config.n_obs_steps
        num_queries = 3  # Goal, Bwd, Fwd
        results = {}

        # 1. 쿼리 토큰들 준비
        goal_query = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_query = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_query = self.forward_seq_query_token.expand(batch_size, -1, -1)

        # 2. 전체 시퀀스 구성: [이력스텝들, 목표쿼리, 역방향쿼리, 순방향쿼리]
        # history_step_embeddings: [B, n_obs, D]
        # goal_query, bwd_query, fwd_query: [B, 1, D]
        full_sequence = torch.cat(
            [history_step_embeddings, goal_query, bwd_query, fwd_query], dim=1
        )  # Shape: [B, n_obs + num_queries, hidden_dim]

        # 3. 토큰 타입 임베딩 적용
        hist_types = torch.full((batch_size, n_obs),
                                self.TYPE_HIST_STEP, device=device)
        query_types_list = [self.TYPE_QUERY_GOAL,
                            self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD]
        query_types = torch.tensor(query_types_list, device=device).unsqueeze(
            0).expand(batch_size, -1)

        all_token_types = torch.cat([hist_types, query_types], dim=1)
        full_sequence += self.token_type_embedding(all_token_types)

        # 4. 위치 임베딩 적용
        positions = torch.arange(full_sequence.shape[1], device=device).unsqueeze(
            0).expand(batch_size, -1)
        # self.position_embedding 크기 확인/조정 필요
        full_sequence += self.position_embedding(positions)

        # 5. 새로운 어텐션 마스크 적용
        attn_mask = self._create_full_history_sequential_mask(device)

        # 6. 트랜스포머 통과
        hidden_states = self.transformer(src=full_sequence, mask=attn_mask)

        # 7. 예측 헤드 사용 (쿼리 토큰들의 인덱스는 이제 n_obs 부터 시작)
        goal_query_output = hidden_states[:, n_obs]
        bwd_query_output = hidden_states[:, n_obs + 1]
        fwd_query_output = hidden_states[:, n_obs + 2]

        predicted_goal_latents = self.goal_image_latent_head(goal_query_output)
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)
        results['predicted_goal_latents'] = predicted_goal_latents

        predicted_bwd_states_flat = self.backward_state_head(bwd_query_output)
        results['predicted_backward_states'] = predicted_bwd_states_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim
        )

        predicted_fwd_states_flat = self.forward_state_head(fwd_query_output)
        results['predicted_forward_states'] = predicted_fwd_states_flat.view(
            batch_size, self.config.forward_steps - 1, self.config.state_dim
        )
        return results

    def forward(
        self,
        initial_images: torch.Tensor,  # Shape: [B, n_obs_steps, C, H, W]
        initial_states: torch.Tensor,  # Shape: [B, n_obs_steps, state_dim]
        forward_states: Optional[torch.Tensor] = None,  # GT for training
        goal_images: Optional[torch.Tensor] = None,     # GT for training
        backward_states: Optional[torch.Tensor] = None,  # GT for training
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        device = initial_images.device
        batch_size = initial_images.shape[0]
        n_obs = self.config.n_obs_steps

        # 1. 과거 이력의 각 스텝별 특징 추출
        img_hist_flat = initial_images.reshape(
            batch_size * n_obs, self.config.image_channels,
            self.config.image_size, self.config.image_size
        )
        img_latents_per_step_flat = self.image_encoder(img_hist_flat)
        img_latents_history = img_latents_per_step_flat.view(
            batch_size, n_obs, self.config.image_latent_dim
        )

        states_hist_flat = initial_states.reshape(
            batch_size * n_obs, self.config.state_dim)
        states_projected_per_step_flat = self.state_projection(
            states_hist_flat)
        states_projected_history = states_projected_per_step_flat.view(
            batch_size, n_obs, self.config.hidden_dim
        )

        # 2. 각 이력 스텝별 (이미지 잠재값 + 상태 임베딩) 결합 후 hidden_dim으로 투영
        combined_history_features_per_step = torch.cat(
            [img_latents_history, states_projected_history], dim=-1
        )  # Shape: [B, n_obs, image_latent_dim + hidden_dim]

        # history_step_embeddings: [B, n_obs, hidden_dim]
        history_step_embeddings = self.history_step_projector(
            combined_history_features_per_step)

        # 이제 history_step_embeddings를 사용하여
        # _forward_training_with_global_cond 또는 _forward_inference_with_sequential_history 호출
        if training:
            if forward_states is None or goal_images is None or backward_states is None:
                raise ValueError("Ground truth needed for training.")
            results = self._forward_training_with_sequential_history(
                history_step_embeddings,  # [B, n_obs, hidden_dim]
                forward_states,
                goal_images,
                backward_states,
                device
            )
        else:  # Inference
            results = self._forward_inference_with_sequential_history(history_step_embeddings,  # [B, n_obs, hidden_dim]
                                                                      device)

        return results


def compute_loss(
    # 여기서 'BidirectionalARTransformer'는 타입 힌트입니다.
    model: 'BidirectionalARTransformer',
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute losses for the query-based path outputs.


    It focuses on the outputs from the query-based mechanism.
    """
    losses = {}

    # Forward state prediction loss
    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
        # GT is st_0 to st_F-1, target for model is st_1 to st_F-1
        target_fwd = targets['forward_states'][:, 1:]
        losses['forward_state_loss'] = F.mse_loss(
            predictions['predicted_forward_states'], target_fwd)

    # Backward state prediction loss
    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.mse_loss(
            predictions['predicted_backward_states'], targets['backward_states'])

    # Goal image reconstruction loss
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'], targets['goal_images'])

    # Latent consistency for goal image
    if 'predicted_goal_latents' in predictions and 'goal_images' in targets:
        with torch.no_grad():
            # model.image_encoder is now always a single-frame encoder
            goal_image_latents_gt = model.image_encoder(targets['goal_images'])
        losses['goal_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_goal_latents'], goal_image_latents_gt)

    # Loss weighting - simplified without AR losses
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 2.0,          # 목표 이미지 예측에 더 큰 가중치
        'goal_latent_consistency_loss': 1.0,
    }
    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)

    for loss_name, loss_value in losses.items():
        if loss_name in weights and loss_value is not None:  # Check for None
            total_loss += weights.get(loss_name, 1.0) * loss_value
    losses['total_loss'] = total_loss
    return losses
