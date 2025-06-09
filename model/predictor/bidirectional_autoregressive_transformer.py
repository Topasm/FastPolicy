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
import torchvision.models as models
import torchvision.transforms as transforms

from model.modules.modules import SpatialSoftmax
from model.modules.custom_transformer import RMSNorm, ReplicaTransformerEncoderLayer, ReplicaTransformerEncoder


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8  # Changed from 12 to 8 to make 512 evenly divisible
    dropout: float = 0.1
    layernorm_epsilon: float = 1e-5
    image_latent_dim: int = 256  # Latent dimension for image features
    image_channels: int = 3
    image_size: int = 84
    output_image_size: int = 96  # Output image size after decoding
    forward_steps: int = 20
    backward_steps: int = 16
    n_obs_steps: int = 3  # Number of observation steps in history
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    # Image cropping parameter - only random vs center
    crop_is_random: bool = True

    # Number of pure query tokens (goal, backward, forward)
    num_query_tokens: int = 3

    # Number of action steps (not used in this model, but kept for compatibility)
    n_action_steps = 32

    image_features = 1
    # Token types: HistImg, HistState, QueryGoal, QueryBwd, QueryFwd
    # Changed from 6 to 5 (removed time conditioning token)
    token_type_count: int = 5

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

        config_path = Path(output_dir) / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)

        return cls(**config_dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )


class ImageEncoder(nn.Module):
    """
    Improved ResNet-18 based image encoder with SpatialSoftmax for better spatial feature extraction.

    This encoder uses a pretrained ResNet-18 backbone followed by SpatialSoftmax pooling
    which extracts spatial keypoint features from the convolutional feature maps.
    """

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config

        # Set up preprocessing for image cropping based on image_size
        self.do_crop = True  # Always enable cropping using image_size
        self.center_crop = transforms.CenterCrop(
            (config.image_size, config.image_size))
        if config.crop_is_random:
            self.maybe_random_crop = transforms.RandomCrop(
                (config.image_size, config.image_size))
        else:
            self.maybe_random_crop = self.center_crop

        # Load pre-trained ResNet-18
        resnet = models.resnet18(pretrained=True)

        # Remove the final fully connected layer and average pooling
        # Keep only the convolutional feature extraction parts
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])

        # Use a dry run to get the feature map shape
        dummy_shape = (1, config.image_channels,
                       config.image_size, config.image_size)
        self.register_buffer("dummy_input", torch.zeros(dummy_shape))
        with torch.no_grad():
            feature_map_shape = self.backbone(self.dummy_input).shape[1:]

        # Number of spatial keypoints to extract
        num_keypoints = 32  # Can be tuned based on needs

        # Set up spatial softmax pooling
        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=num_keypoints
        )

        # The output dim of SpatialSoftmax is num_kp * 2
        pool_out_dim = num_keypoints * 2

        # Project to latent dimension
        self.out = nn.Linear(pool_out_dim, config.image_latent_dim)
        self.layer_norm = nn.LayerNorm(config.image_latent_dim)

        # Optional: freeze early layers for transfer learning
        # for param in list(self.backbone.parameters())[:-4]:
        #     param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature, where D is config.image_latent_dim.
        """
        # Apply cropping if configured
        if self.do_crop:
            if self.training:
                images = self.maybe_random_crop(images)
            else:
                # Always use center crop for eval
                images = self.center_crop(images)

        # Extract backbone features
        features = self.backbone(images)  # (B, C, H, W)

        # Apply spatial softmax pooling
        keypoints = self.pool(features)  # (B, K, 2)

        # Flatten keypoints
        features_flat = torch.flatten(keypoints, start_dim=1)  # (B, K*2)

        # Apply final projection and layer norm
        # (B, image_latent_dim)
        output = self.layer_norm(self.out(features_flat))

        return output


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
                               kernel_size=4, stride=2, padding=1), nn.Sigmoid()
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        x = self.initial_linear(latents)
        x = x.view(-1, 512, 3, 3)
        return self.decoder(x)


# --- 1. 입력 처리 모듈 ---
class InputBlock(nn.Module):
    """이미지와 상태 입력을 받아 hidden_dim의 임베딩으로 변환합니다."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

    def forward(self, initial_images: torch.Tensor, initial_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = initial_images.shape[0]
        n_obs = self.config.n_obs_steps

        # 이미지 처리
        img_hist_flat = initial_images.view(
            batch_size * n_obs, self.config.image_channels,
            self.config.output_image_size, self.config.output_image_size
        )
        img_latents_per_step_flat = self.image_encoder(img_hist_flat)
        img_latents_history = img_latents_per_step_flat.view(
            batch_size, n_obs, self.config.image_latent_dim
        )
        img_history_embeddings = self.image_latent_projection(
            img_latents_history)

        # 상태 처리
        states_hist_flat = initial_states.view(
            batch_size * n_obs, self.config.state_dim)
        states_projected_per_step_flat = self.state_projection(
            states_hist_flat)
        state_history_embeddings = states_projected_per_step_flat.view(
            batch_size, n_obs, self.config.hidden_dim
        )

        return img_history_embeddings, state_history_embeddings


# --- 2. 출력 처리 모듈 ---
class OutputHeadBlock(nn.Module):
    """트랜스포머의 출력을 받아 최종 예측값을 생성합니다."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.image_decoder = ImageDecoder(config)
        self.progress_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1), nn.Sigmoid())
        self.forward_state_head = nn.Linear(
            config.hidden_dim, config.forward_steps * config.state_dim)
        self.goal_image_latent_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)
        self.backward_state_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        n_obs = self.config.n_obs_steps
        num_hist_tokens = n_obs * 2

        # 각 쿼리의 출력 추출
        goal_query_output = hidden_states[:, num_hist_tokens]
        bwd_query_output = hidden_states[:, num_hist_tokens + 1]
        fwd_query_output = hidden_states[:, num_hist_tokens + 2]

        # 예측 헤드 통과
        results = {}
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
            batch_size, self.config.forward_steps, self.config.state_dim
        )

        # 진행도(progress) 예측
        history_output_embeddings = hidden_states[:, :num_hist_tokens]
        avg_history_embedding = torch.mean(history_output_embeddings, dim=1)
        results['predicted_progress'] = self.progress_head(
            avg_history_embedding)

        return results


# --- 3. 메인 BidirectionalARTransformer 리팩토링 ---
class BidirectionalARTransformer(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, state_key: str = "observation.state", image_key: str = "observation.image"):
        super().__init__()
        self.config = config
        self.state_key = state_key
        self.image_key = image_key

        from lerobot.configs.types import FeatureType
        self.feature_type = FeatureType

        # --- 모듈화된 블록 초기화 ---
        self.input_block = InputBlock(config)
        # self.output_block = OutputHeadBlock(config)  # 제거

        # --- 새로운 단일 출력 헤드 추가 ---
        # 트랜스포머의 hidden_dim을 입력받아 다음 state_dim을 예측
        self.next_state_head = nn.Linear(config.hidden_dim, config.state_dim)

        # --- ❗️❗️ PyTorch 기본 라이브러리를 사용했을 때와 동일한 방식으로 조립 ❗️❗️ ---

        # 1. '복제' 레이어를 정의합니다.
        replica_encoder_layer = ReplicaTransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True  # 중요
        )

        # 2. '복제' 레이어를 N개 쌓아 최종 트랜스포머를 만듭니다.
        self.transformer = ReplicaTransformerEncoder(
            encoder_layer=replica_encoder_layer,
            num_layers=config.num_layers,
            norm=RMSNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        print("✅ Initialized with 'ReplicaTransformerEncoder' which is a 1:1 copy of PyTorch's implementation.")

        # --- 시퀀스 구성을 위한 임베딩 및 쿼리 토큰 (메인 클래스에서 관리) ---
        self.TYPE_HIST_IMG, self.TYPE_HIST_STATE, self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD = 0, 1, 2, 3, 4
        self.token_type_embedding = nn.Embedding(5, config.hidden_dim)
        self.num_queries = 3
        self.total_seq_len = (config.n_obs_steps * 2) + self.num_queries
        self.position_embedding = nn.Embedding(
            self.total_seq_len, config.hidden_dim)
        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

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
        Creates an attention mask for the sequence without time condition token.
        Seq: [ImgH_0, StateH_0, ..., ImgH_{n-1}, StateH_{n-1}, Q_goal, Q_bwd, Q_fwd]
        """
        n_obs = self.config.n_obs_steps
        num_hist_tokens = n_obs * 2  # Each step has image and state token
        num_queries = 3
        seq_len = num_hist_tokens + num_queries  # No time token

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)

        # History tokens can attend to each other
        mask[0:num_hist_tokens, 0:num_hist_tokens] = False

        # Query tokens can attend to all history tokens
        mask[num_hist_tokens:, :num_hist_tokens] = False

        # Q_bwd can attend to Q_goal
        mask[num_hist_tokens + 1, num_hist_tokens] = False

        # Q_fwd can attend to Q_goal and Q_bwd
        mask[num_hist_tokens + 2, num_hist_tokens:num_hist_tokens + 2] = False

        return mask

    # 추가: 미래 타임스텝을 보지 못하도록 하는 Causal Mask 생성 메서드
    def _create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """미래 타임스텝을 보지 못하도록 하는 Causal Mask를 생성합니다."""
        mask = torch.triu(torch.ones(seq_len, seq_len,
                          device=device), diagonal=1).bool()
        # True인 위치는 어텐션 계산에서 무시됩니다.
        return mask

    def forward(self,
                initial_images,
                initial_states,
                goal_images,
                backward_states,
                forward_states,
                **kwargs):
        device = initial_images.device
        batch_size = initial_images.shape[0]

        # --- 1. 입력 시퀀스 구성 ---
        # 가. 이력(History) 임베딩
        img_embed, state_embed = self.input_block(
            initial_images, initial_states)
        history_sequence = torch.stack(
            [img_embed, state_embed], dim=2
        ).flatten(start_dim=1, end_dim=2)

        # 나. 목표(Target) 시퀀스 구성 (state만 사용, 이미지는 생략)
        target_sequence_states = torch.cat([
            torch.flip(backward_states, [1]),  # 역방향 궤적을 시간순으로 뒤집음
            forward_states
        ], dim=1)

        # 다. 목표 시퀀스를 hidden_dim으로 임베딩
        target_sequence_embed = self.state_projection(target_sequence_states)

        # 라. 전체 시퀀스 결합: [이력, 목표 궤적]
        full_sequence = torch.cat(
            [history_sequence, target_sequence_embed], dim=1)
        seq_len = full_sequence.shape[1]

        # 위치 임베딩 적용
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        full_sequence += self.position_embedding(positions)

        # --- 2. Causal Mask를 사용하여 디코더 순전파 ---
        causal_mask = self._create_causal_mask(seq_len, device)
        hidden_states = self.transformer(src=full_sequence, mask=causal_mask)

        # --- 3. 다음 스텝 예측 및 Loss 계산 준비 ---
        predicted_next_states = self.next_state_head(hidden_states)

        # Loss 계산을 위해, 예측의 마지막 스텝은 제외
        predictions_for_loss = predicted_next_states[:, :-1, :]

        # 정답 시퀀스는 첫 스텝을 제외하고 예측 대상이 됨
        targets_for_loss = target_sequence_states

        loss = F.l1_loss(predictions_for_loss, targets_for_loss)

        return loss, predictions_for_loss

    @torch.no_grad()
    def generate(self, initial_images, initial_states, max_traj_len: int):
        """Autoregressively generate future states up to max_traj_len."""
        self.eval()  # switch to inference mode
        device = initial_images.device

        # 1. prepare history embeddings
        img_embed, state_embed = self.input_block(
            initial_images, initial_states)
        generated_sequence = torch.stack(
            [img_embed, state_embed], dim=2
        ).flatten(start_dim=1, end_dim=2)

        # 2. autoregressive loop
        for _ in range(max_traj_len):
            current_len = generated_sequence.shape[1]
            positions = torch.arange(current_len, device=device).unsqueeze(0)
            input_sequence = generated_sequence + \
                self.position_embedding(positions)

            causal_mask = self._create_causal_mask(current_len, device)
            hidden_states = self.transformer(
                src=input_sequence, mask=causal_mask)

            last_hidden = hidden_states[:, -1, :]
            next_state = self.next_state_head(last_hidden)

            next_embed = self.state_projection(next_state).unsqueeze(1)
            generated_sequence = torch.cat(
                [generated_sequence, next_embed], dim=1)

        # return embeddings for generated steps
        return generated_sequence[:, -max_traj_len:, :]


# --- 새로운 Encoder-Decoder 기반 트랜스포머 모델 ---
class EncoderDecoderTransformer(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, state_key: str = "observation.state", image_key: str = "observation.image"):
        super().__init__()
        self.config = config
        self.state_key = state_key
        self.image_key = image_key

        from lerobot.configs.types import FeatureType
        self.feature_type = FeatureType

        # --- 1. 입력 처리 모듈 ---
        self.input_block = InputBlock(config)

        # --- 2. 인코더 정의 ---
        # 이력 정보를 처리하여 '기억(memory)'을 생성
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads, dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.context_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        # --- 3. 디코더 정의 ---
        # '기억'을 바탕으로 쿼리에 대한 답을 생성
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim, nhead=config.num_heads, dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, activation='gelu', batch_first=True, norm_first=True
        )
        self.prediction_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_layers,
            norm=nn.LayerNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        # --- 4. 출력 처리 모듈 ---
        self.output_block = OutputHeadBlock(config)

        # --- 5. 임베딩 및 쿼리 토큰 ---
        self.TYPE_HIST_IMG, self.TYPE_HIST_STATE, self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD = 0, 1, 2, 3, 4
        self.token_type_embedding = nn.Embedding(5, config.hidden_dim)

        # 인코더와 디코더의 위치 임베딩을 별도로 가질 수 있음
        self.encoder_pos_embedding = nn.Embedding(
            config.n_obs_steps * 2, config.hidden_dim)
        self.decoder_pos_embedding = nn.Embedding(
            config.num_query_tokens, config.hidden_dim)

        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

        self.apply(self._init_weights)

        print("✅ Using Encoder-Decoder Transformer architecture")

    def _init_weights(self, module):
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

    def forward(
        self,
        initial_images: torch.Tensor,
        initial_states: torch.Tensor,
        forward_states: Optional[torch.Tensor] = None,
        goal_images: Optional[torch.Tensor] = None,
        backward_states: Optional[torch.Tensor] = None,
        training: bool = True
    ) -> Dict[str, torch.Tensor]:
        device = initial_images.device
        batch_size = initial_images.shape[0]

        # Training 체크 (그대로 유지)
        if training and (forward_states is None or goal_images is None or backward_states is None):
            raise ValueError("Ground truth needed for training.")

        # === 인코더 순전파 ===
        # 1. 이력 데이터를 임베딩으로 변환
        img_embed, state_embed = self.input_block(
            initial_images, initial_states)
        history_sequence = torch.stack(
            [img_embed, state_embed], dim=2).flatten(start_dim=1, end_dim=2)

        # 2. 이력 시퀀스에 위치 및 타입 정보 추가
        n_hist_tokens = history_sequence.shape[1]
        hist_types = torch.tensor(
            [self.TYPE_HIST_IMG, self.TYPE_HIST_STATE], device=device).repeat(self.config.n_obs_steps)
        hist_types = hist_types.unsqueeze(0).expand(batch_size, -1)
        history_sequence = history_sequence + \
            self.token_type_embedding(hist_types)

        pos_indices = torch.arange(n_hist_tokens, device=device).unsqueeze(
            0).expand(batch_size, -1)
        history_sequence = history_sequence + \
            self.encoder_pos_embedding(pos_indices)

        # 3. 인코더를 통과시켜 '기억' 생성
        memory = self.context_encoder(history_sequence)

        # === 디코더 순전파 ===
        # 4. 쿼리 토큰으로 디코더 입력(tgt) 생성
        goal_q = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_q = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_q = self.forward_seq_query_token.expand(batch_size, -1, -1)
        tgt_sequence = torch.cat([goal_q, bwd_q, fwd_q], dim=1)

        # 5. 디코더 입력에 위치 및 타입 정보 추가
        n_query_tokens = tgt_sequence.shape[1]
        query_types = torch.tensor(
            [self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD], device=device)
        query_types = query_types.unsqueeze(0).expand(batch_size, -1)
        tgt_sequence = tgt_sequence + self.token_type_embedding(query_types)

        query_pos_indices = torch.arange(
            n_query_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        tgt_sequence = tgt_sequence + \
            self.decoder_pos_embedding(query_pos_indices)

        # 6. 디코더를 통과시켜 최종 hidden_states 생성
        # 디코더는 타겟(tgt)과 메모리(memory)를 모두 입력으로 받음
        # 현재 구현에서는 마스크 없이 full attention을 사용
        decoder_output = self.prediction_decoder(
            tgt=tgt_sequence, memory=memory)

        # 7. 출력 헤드를 통해 최종 예측값 생성
        predictions = self.output_block(decoder_output)

        return predictions


def compute_loss(
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
        target_fwd = targets['forward_states']
        losses['forward_state_loss'] = F.l1_loss(
            predictions['predicted_forward_states'], target_fwd)

    # Backward state prediction loss
    if 'predicted_backward_states' in predictions and 'backward_states' in targets:
        losses['backward_state_loss'] = F.l1_loss(
            predictions['predicted_backward_states'], targets['backward_states'])

    # Goal image reconstruction loss
    if 'predicted_goal_images' in predictions and 'goal_images' in targets:
        losses['goal_image_loss'] = F.mse_loss(
            predictions['predicted_goal_images'], targets['goal_images'])

    # --- 핵심 수정 3: "진행 정도" 예측에 대한 손실 추가 ---
    if 'predicted_progress' in predictions and 'normalized_timestep' in targets:
        # predictions['predicted_progress'] shape: [B, 1]
        # targets['normalized_timestep'] shape: [B]
        # squeeze()를 사용하여 차원을 맞춰줌
        predicted = predictions['predicted_progress'].squeeze(-1)
        target = targets['normalized_timestep']
        losses['progress_loss'] = F.mse_loss(predicted, target)
    # --- 수정 완료 ---

    # Loss weighting - simplified without AR losses
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,
        'progress_loss': 0.5  # 새로운 손실 항에 대한 가중치 (하이퍼파라미터)
    }
    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)

    for loss_name, loss_value in losses.items():
        if loss_name in weights and loss_value is not None:  # Check for None
            total_loss += weights.get(loss_name, 1.0) * loss_value
    losses['total_loss'] = total_loss
    return losses
