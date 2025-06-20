#!/usr/bin/env python3
"""
Bidirectional Autoregressive Transformer for image-conditioned trajectory generation.

This model implements the following pipeline with SOFT GOAL CONDITIONING and GLOBAL HISTORY CONDITIONING:
1. Input: sequence of initial images i_{t-k:t} and states st_{t-k:t} (n_obs_steps history)
2. Encode and flatten history into a single global_history_condition_embedding.
3. Using this global_history_condition_embedding:
    a. Generate goal image i_n (first prediction) + progress
    b. Generate backward states st_n ... (conditioned on global history + goal) + progress
    c. Generate forward states st_0 ... (conditioned on global history + goal + backward path) + progress

The new prediction order (goal → backward → forward) enables soft conditioning.
Each query predicts both its main output and progress.
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
    forward_steps: int = 64
    backward_steps: int = 64
    n_obs_steps: int = 3  # Number of observation steps in history
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    # Image cropping parameter - only random vs center
    crop_is_random: bool = True

    # Number of pure query tokens (goal, backward, forward) - removed progress query
    num_query_tokens: int = 3

    # Number of action steps (not used in this model, but kept for compatibility)
    n_action_steps = 64

    image_features = 1
    # Token types: HistImg, HistState, QueryGoal, QueryBwd, QueryFwd
    # Back to 5 token types (removed progress query)
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
        output = self.layer_norm(self.out(features_flat))

        return output


class ImageDecoder(nn.Module):
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


class OutputHeadBlock(nn.Module):
    """트랜스포머의 출력을 받아 최종 예측값을 생성합니다. 각 쿼리가 메인 출력과 진행도를 함께 예측합니다."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        self.image_decoder = ImageDecoder(config)

        # Shared progress head for all queries
        self.progress_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2), nn.ReLU(),
            nn.Linear(config.hidden_dim // 2, 1), nn.Sigmoid())

        # Main output heads
        self.goal_image_latent_head = nn.Linear(
            config.hidden_dim, config.image_latent_dim)
        self.backward_state_head = nn.Linear(
            config.hidden_dim, config.backward_steps * config.state_dim)
        self.forward_state_head = nn.Linear(
            config.hidden_dim, config.forward_steps * config.state_dim)

    def forward(self, hidden_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = hidden_states.shape[0]
        n_obs = self.config.n_obs_steps
        num_hist_tokens = n_obs * 2

        # 각 쿼리의 출력 추출 (progress query 제거됨)
        goal_query_output = hidden_states[:, num_hist_tokens]
        bwd_query_output = hidden_states[:, num_hist_tokens + 1]
        fwd_query_output = hidden_states[:, num_hist_tokens + 2]

        results = {}

        # Goal prediction + progress
        predicted_goal_latents = self.goal_image_latent_head(goal_query_output)
        results['predicted_goal_images'] = self.image_decoder(
            predicted_goal_latents)
        results['predicted_goal_latents'] = predicted_goal_latents
        results['goal_predicted_progress'] = self.progress_head(
            goal_query_output)

        # Backward prediction + progress
        predicted_bwd_states_flat = self.backward_state_head(bwd_query_output)
        results['predicted_backward_states'] = predicted_bwd_states_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim
        )
        results['backward_predicted_progress'] = self.progress_head(
            bwd_query_output)

        # Forward prediction + progress
        predicted_fwd_states_flat = self.forward_state_head(fwd_query_output)
        results['predicted_forward_states'] = predicted_fwd_states_flat.view(
            batch_size, self.config.forward_steps, self.config.state_dim
        )
        results['forward_predicted_progress'] = self.progress_head(
            fwd_query_output)

        return results


class BidirectionalARTransformer(nn.Module):
    def __init__(self, config: BidirectionalARTransformerConfig, state_key: str = "observation.state", image_key: str = "observation.image"):
        super().__init__()
        self.config = config
        self.state_key = state_key
        self.image_key = image_key

        from lerobot.configs.types import FeatureType
        self.feature_type = FeatureType

        # Initialize modular blocks
        self.input_block = InputBlock(config)
        self.output_block = OutputHeadBlock(config)

        # --- Decoder-only transformer structure ---
        # Using ReplicaTransformerEncoder as decoder (with causal masking)
        replica_encoder_layer = ReplicaTransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )

        self.transformer = ReplicaTransformerEncoder(
            encoder_layer=replica_encoder_layer,
            num_layers=config.num_layers,
            norm=RMSNorm(config.hidden_dim, eps=config.layernorm_epsilon)
        )

        print("✅ Initialized with decoder-only structure using ReplicaTransformerEncoder.")

        # Sequence composition embeddings and query tokens
        # Token types: HistImg, HistState, QueryGoal, QueryBwd, QueryFwd (removed progress query)
        self.TYPE_HIST_IMG, self.TYPE_HIST_STATE, self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD = 0, 1, 2, 3, 4
        self.token_type_embedding = nn.Embedding(5, config.hidden_dim)
        self.num_queries = 3  # goal, backward, forward (removed progress)
        self.total_seq_len = (config.n_obs_steps * 2) + self.num_queries
        self.position_embedding = nn.Embedding(
            self.total_seq_len, config.hidden_dim)

        # Query tokens (removed progress query)
        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

        self.apply(self._init_weights)

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

    def _create_decoder_only_mask(self, device: torch.device) -> torch.Tensor:
        """
        Creates a decoder-only attention mask with query-based progression.
        Seq: [ImgH_0, StateH_0, ..., ImgH_{n-1}, StateH_{n-1}, Q_goal, Q_bwd, Q_fwd]

        Decoder-only structure with query progression:
        - History tokens can attend to previous history tokens (causal)
        - Goal query can attend to all history tokens
        - Backward query can attend to all history tokens + goal query
        - Forward query can attend to all history tokens + goal query + backward query
        """
        n_obs = self.config.n_obs_steps
        num_hist_tokens = n_obs * 2  # Each step has image and state token
        num_queries = 3  # goal, backward, forward
        seq_len = num_hist_tokens + num_queries

        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        mask.fill_diagonal_(False)

        # History tokens: causal masking (can only attend to previous tokens)
        for i in range(num_hist_tokens):
            # Can attend to tokens up to and including current position
            mask[i, :i+1] = False

        # Query tokens can attend to all history tokens
        mask[num_hist_tokens:, :num_hist_tokens] = False

        # Goal query (position num_hist_tokens) - already handled above

        # Backward query (position num_hist_tokens + 1) can attend to goal query
        mask[num_hist_tokens + 1, num_hist_tokens] = False

        # Forward query (position num_hist_tokens + 2) can attend to goal query + backward query
        mask[num_hist_tokens + 2, num_hist_tokens:num_hist_tokens + 2] = False

        return mask

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

        # 1. Input processing (module call)
        img_history_embeddings, state_history_embeddings = self.input_block(
            initial_images, initial_states)

        # Training validation
        if training and (forward_states is None or goal_images is None or backward_states is None):
            raise ValueError("Ground truth needed for training.")

        # 2. Sequence composition
        history_sequence = torch.stack(
            [img_history_embeddings, state_history_embeddings], dim=2
        ).flatten(start_dim=1, end_dim=2)

        # Query tokens (removed progress query)
        goal_query = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_query = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_query = self.forward_seq_query_token.expand(batch_size, -1, -1)

        full_sequence = torch.cat(
            [history_sequence, goal_query, bwd_query, fwd_query], dim=1
        )

        # 3. Apply token type and position embeddings
        hist_types_per_step = torch.tensor(
            [self.TYPE_HIST_IMG, self.TYPE_HIST_STATE], device=device)
        hist_types = hist_types_per_step.repeat(n_obs)

        query_types = torch.tensor(
            [self.TYPE_QUERY_GOAL, self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD], device=device)

        all_token_types = torch.cat([hist_types, query_types]).unsqueeze(
            0).expand(batch_size, -1)

        full_sequence += self.token_type_embedding(all_token_types)

        positions = torch.arange(full_sequence.shape[1], device=device).unsqueeze(
            0).expand(batch_size, -1)
        full_sequence += self.position_embedding(positions)

        # 4. Pass through transformer backbone with decoder-only masking
        attn_mask = self._create_decoder_only_mask(device)
        hidden_states = self.transformer(src=full_sequence, mask=attn_mask)

        # 5. Output processing (module call)
        results = self.output_block(hidden_states)

        return results


def compute_loss(
    model: 'BidirectionalARTransformer',
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute losses for the query-based path outputs with progress prediction at each step.
    """
    losses = {}

    # Forward state prediction loss
    if 'predicted_forward_states' in predictions and 'forward_states' in targets:
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

    # Progress prediction losses for each query
    if 'normalized_timestep' in targets:
        target_progress = targets['normalized_timestep']

        # Goal query progress loss
        if 'goal_predicted_progress' in predictions:
            predicted_goal_progress = predictions['goal_predicted_progress'].squeeze(
                -1)
            losses['goal_progress_loss'] = F.mse_loss(
                predicted_goal_progress, target_progress)

        # Backward query progress loss
        if 'backward_predicted_progress' in predictions:
            predicted_bwd_progress = predictions['backward_predicted_progress'].squeeze(
                -1)
            losses['backward_progress_loss'] = F.mse_loss(
                predicted_bwd_progress, target_progress)

        # Forward query progress loss
        if 'forward_predicted_progress' in predictions:
            predicted_fwd_progress = predictions['forward_predicted_progress'].squeeze(
                -1)
            losses['forward_progress_loss'] = F.mse_loss(
                predicted_fwd_progress, target_progress)

    # Loss weighting
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 1.0,
        'goal_progress_loss': 0.3,      # Weight for goal progress prediction
        'backward_progress_loss': 0.3,  # Weight for backward progress prediction
        'forward_progress_loss': 0.3,   # Weight for forward progress prediction
    }

    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)

    for loss_name, loss_value in losses.items():
        if loss_name in weights and loss_value is not None:
            total_loss += weights.get(loss_name, 1.0) * loss_value

    losses['total_loss'] = total_loss
    return losses
