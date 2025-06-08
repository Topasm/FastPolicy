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
from torch import Tensor

from model.modules.modules import SpatialSoftmax
from model.predictor.gpt_backbone import GPTBackbone


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 12
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


class BidirectionalARTransformer(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, state_key: str = "observation.state", image_key: str = "observation.image"):
        super().__init__()
        self.config = config
        self.state_key = state_key
        self.image_key = image_key

        from lerobot.configs.types import FeatureType
        self.feature_type = FeatureType

        print("Using ImageEncoder in BidirectionalARTransformer.")
        self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)

        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # Remove timestep projection layer

        # Keep progress_head for auxiliary loss
        self.progress_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1),
            nn.Sigmoid()
        )

        # Simplified token type definitions (5 types total)
        self.TYPE_HIST_IMG, self.TYPE_HIST_STATE, self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD = 0, 1, 2, 3, 4

        # Update token type embedding size to 5
        self.token_type_embedding = nn.Embedding(5, config.hidden_dim)

        # Update position embedding size (removed time token)
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

        # Output heads remain unchanged
        self.forward_state_head = nn.Linear(
            config.hidden_dim, (config.forward_steps) * config.state_dim)
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

    def _run_prediction(
        self,
        img_history_embeddings: torch.Tensor,
        state_history_embeddings: torch.Tensor,
        device: torch.device,
        forward_states: Optional[torch.Tensor] = None,
        goal_images: Optional[torch.Tensor] = None,
        backward_states: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        batch_size = img_history_embeddings.shape[0]
        n_obs = self.config.n_obs_steps

        # 1. Construct history token sequence (interleaving image-state)
        history_sequence = torch.stack(
            [img_history_embeddings, state_history_embeddings], dim=2
        ).flatten(start_dim=1, end_dim=2)

        # 2. Prepare query tokens
        goal_query = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_query = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_query = self.forward_seq_query_token.expand(batch_size, -1, -1)

        # 3. Construct full sequence (without time condition token)
        full_sequence = torch.cat(
            [history_sequence, goal_query, bwd_query, fwd_query], dim=1
        )

        # 4. Apply token type and position embeddings
        hist_types_per_step = torch.tensor(
            [self.TYPE_HIST_IMG, self.TYPE_HIST_STATE], device=device)
        hist_types = hist_types_per_step.repeat(n_obs)

        query_types = torch.tensor(
            [self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD], device=device)

        all_token_types = torch.cat([hist_types, query_types]).unsqueeze(
            0).expand(batch_size, -1)

        full_sequence += self.token_type_embedding(all_token_types)

        # Position embeddings
        positions = torch.arange(full_sequence.shape[1], device=device).unsqueeze(
            0).expand(batch_size, -1)
        full_sequence += self.position_embedding(positions)

        # 5. Apply attention mask and run transformer
        attn_mask = self._create_full_history_sequential_mask(device)
        hidden_states = self.transformer(src=full_sequence, mask=attn_mask)

        # 6. Predict progress (auxiliary prediction)
        num_hist_tokens = n_obs * 2
        history_output_embeddings = hidden_states[:, :num_hist_tokens]
        avg_history_embedding = torch.mean(history_output_embeddings, dim=1)
        predicted_progress = self.progress_head(avg_history_embedding)

        # 7. Use prediction heads (updated indices)
        # No time token to skip
        goal_query_output = hidden_states[:, num_hist_tokens]
        bwd_query_output = hidden_states[:, num_hist_tokens + 1]
        fwd_query_output = hidden_states[:, num_hist_tokens + 2]

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

        results['predicted_progress'] = predicted_progress

        return results

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
        n_obs = self.config.n_obs_steps

        # 1. 과거 이력의 각 스텝별 특징 추출 (기존과 동일)
        img_hist_flat = initial_images.view(
            batch_size * n_obs, self.config.image_channels,
            self.config.output_image_size, self.config.output_image_size
        )
        img_latents_per_step_flat = self.image_encoder(img_hist_flat)
        img_latents_history = img_latents_per_step_flat.view(
            batch_size, n_obs, self.config.image_latent_dim
        )

        states_hist_flat = initial_states.view(
            batch_size * n_obs, self.config.state_dim)
        states_projected_per_step_flat = self.state_projection(
            states_hist_flat)
        states_projected_history = states_projected_per_step_flat.view(
            batch_size, n_obs, self.config.hidden_dim
        )

        # 2. Seer와 유사하게, 이미지와 상태를 별도의 토큰으로 처리
        # 이미지 잠재 벡터를 hidden_dim으로 투영
        img_history_embeddings = self.image_latent_projection(
            img_latents_history)  # [B, n_obs, hidden_dim]
        # 상태 프로젝션은 이미 hidden_dim
        # [B, n_obs, hidden_dim]
        state_history_embeddings = states_projected_history

        # Training 체크는 하지만 내부 로직은 동일한 함수 사용
        if training and (forward_states is None or goal_images is None or backward_states is None):
            raise ValueError("Ground truth needed for training.")

        results = self._run_prediction(
            img_history_embeddings,
            state_history_embeddings,
            device,
            forward_states,
            goal_images,
            backward_states
        )

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
