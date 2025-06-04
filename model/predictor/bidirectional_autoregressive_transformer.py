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
The model is trained with both an autoregressive path and a query-based path.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any
from pathlib import Path
from lerobot.configs.types import NormalizationMode

from model.diffusion.diffusion_modules import DiffusionRgbEncoder
from model.diffusion.configuration_mymodel import DiffusionConfig


@dataclass
class BidirectionalARTransformerConfig:
    """Configuration for the Bidirectional Autoregressive Transformer."""
    state_dim: int = 7
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    max_position_value: int = 64  # Max length for AR path positional embeddings
    layernorm_epsilon: float = 1e-5
    image_channels: int = 3
    image_size: int = 96
    image_latent_dim: int = 256
    forward_steps: int = 20
    backward_steps: int = 16
    n_obs_steps: int = 2  # Number of observation steps in history
    input_features: Dict[str, Any] = field(default_factory=dict)
    output_features: Dict[str, Any] = field(default_factory=dict)

    use_diffusion_encoder: bool = True
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str = "IMAGENET1K_V1"
    spatial_softmax_num_keypoints: int = 32
    use_group_norm: bool = False
    crop_shape: Optional[tuple] = None
    crop_is_random: bool = False

    # Define number of query tokens for sequence structure
    # global_history_token (1) + goal_q (1) + bwd_q (1) + fwd_q (1) = 4
    query_seq_len: int = 4

    # For query-based path only (no autoregressive path)
    # HIST_COND, QUERY_GOAL, QUERY_BWD, QUERY_FWD
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
        with open(Path(output_dir) / "config.json", "r") as f:
            config_dict = json.load(f)
        return cls(**config_dict)

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX
        }
    )


class DiffusionImageEncoder(nn.Module):
    """Adapter for DiffusionRgbEncoder."""

    def __init__(self, config: BidirectionalARTransformerConfig):
        super().__init__()
        self.config = config
        diffusion_config_params = {
            "input_features": config.input_features,
            "output_features": config.output_features,
            "vision_backbone": config.vision_backbone,
            "pretrained_backbone_weights": config.pretrained_backbone_weights,
            "spatial_softmax_num_keypoints": config.spatial_softmax_num_keypoints,
            "use_group_norm": config.use_group_norm,
            "crop_shape": config.crop_shape,
            "crop_is_random": config.crop_is_random,
            # DiffusionRgbEncoder outputs transformer_dim
            "transformer_dim": config.hidden_dim,
        }
        # Ensure all required fields for DiffusionConfig are present or have defaults
        # Add other DiffusionConfig defaults if necessary
        required_diffusion_fields = {
            "n_obs_steps": config.n_obs_steps, "horizon": 1, "n_action_steps": 1,  # Dummy values
        }
        for k, v in required_diffusion_fields.items():
            if k not in diffusion_config_params:
                diffusion_config_params[k] = v

        try:
            diffusion_cfg = DiffusionConfig(**diffusion_config_params)
            self.diffusion_encoder = DiffusionRgbEncoder(diffusion_cfg)
            self.projection = nn.Linear(
                config.hidden_dim, config.image_latent_dim)
            self.use_valid_encoder = True
        except Exception as e:
            print(
                f"Warning: Failed to create DiffusionRgbEncoder due to missing FeatureSpec or other error: {e}. Falling back to simple CNN if available or erroring.")
            # Will rely on simple_encoder if that path is taken by parent
            self.use_valid_encoder = False
            # To make this class fully standalone even in error, we might init simple_encoder here too.
            # For now, assuming parent class (BidirectionalARTransformer) handles the fallback.

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not self.use_valid_encoder:
            raise RuntimeError(
                "DiffusionImageEncoder was not properly initialized.")
        # Heuristic for [-1, 1] range
        if images.min() < 0 and images.max() <= 1:
            images = (images + 1.0) / 2.0
        features = self.diffusion_encoder(images)
        return self.projection(features)


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

        if config.use_diffusion_encoder:
            self.image_encoder = DiffusionImageEncoder(config)
        else:
            self.image_encoder = ImageEncoder(config)
        self.image_decoder = ImageDecoder(config)

        self.state_projection = nn.Linear(config.state_dim, config.hidden_dim)
        # Add back the image_latent_projection that was missing
        self.image_latent_projection = nn.Linear(
            config.image_latent_dim, config.hidden_dim)

        # Projector for the flattened history vector
        flat_history_dim = config.n_obs_steps * \
            (config.image_latent_dim + config.hidden_dim)
        self.history_to_global_condition_projector = nn.Linear(
            flat_history_dim, config.hidden_dim)

        # Define token type indices for the query-based path
        # These token types are used to help the model distinguish between different
        # token roles, improving performance through role-aware attention

        # Token type constants for query path
        self.TYPE_HIST_COND = 0   # History condition token
        self.TYPE_QUERY_GOAL = 1  # Goal query token for generating goal image
        # Backward trajectory query token for generating backward states
        self.TYPE_QUERY_BWD = 2
        self.TYPE_QUERY_FWD = 3   # Forward trajectory query token for generating forward states

        # Token type embeddings - we only need 4 types now (simplified from previous design)
        self.token_type_embedding = nn.Embedding(
            4, config.hidden_dim)  # Simplified: only need 4 token types for query path

        # Position embeddings for the query sequence structure (length config.query_seq_len)
        self.position_embedding = nn.Embedding(
            config.query_seq_len, config.hidden_dim)

        self.goal_image_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.backward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)
        self.forward_seq_query_token = nn.Parameter(
            torch.randn(1, 1, config.hidden_dim) * 0.02)

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

        # No AR-specific output heads needed for query-only approach

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

    def _create_sequential_attention_mask(self, device: torch.device) -> torch.Tensor:
        """
        Creates a custom attention mask for the query-based sequence.

        The query path has a specific conditional information flow:
        - Each token can attend to itself
        - Goal query attends to history condition
        - Backward query attends to history condition and goal query
        - Forward query attends to all previous tokens

        This creates a directed sequence of information flow:
        History → Goal → Backward → Forward

        Token sequence structure: [HistCond, Q_goal, Q_bwd, Q_fwd] (length 4)
        Indices:                    0         1       2       3
        """
        seq_len = self.config.query_seq_len  # Should be 4
        # Initialize a mask where True = cannot attend (will be masked out)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool,
                          device=device)  # True = cannot attend
        mask.fill_diagonal_(False)  # Allow tokens to attend to themselves

        # Q_goal (idx 1) can attend to HistCond (idx 0)
        mask[1, 0] = False

        # Q_bwd (idx 2) can attend to HistCond (idx 0), Q_goal (idx 1)
        mask[2, 0:2] = False

        # Q_fwd (idx 3) can attend to HistCond (idx 0), Q_goal (idx 1), Q_bwd (idx 2)
        mask[3, 0:3] = False
        return mask

    def _forward_training_with_global_cond(
        self,
        global_history_condition_emb: torch.Tensor,  # [B, hidden_dim]
        forward_states_gt: torch.Tensor,    # [B, forward_steps, state_dim]
        # [B, C, H, W] (raw, for encoder to get target latent)
        goal_images_gt: torch.Tensor,
        backward_states_gt: torch.Tensor,   # [B, backward_steps, state_dim]
        device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training using only the query-based path.

        Uses global history condition embedding to generate predictions for:
        - Goal image
        - Backward trajectory states
        - Forward trajectory states
        """
        batch_size = global_history_condition_emb.shape[0]
        results = {}

        # Create the query-based sequence
        history_cond_token = global_history_condition_emb.unsqueeze(
            1)  # [B, 1, D]
        goal_query = self.goal_image_query_token.expand(batch_size, -1, -1)
        bwd_query = self.backward_seq_query_token.expand(batch_size, -1, -1)
        fwd_query = self.forward_seq_query_token.expand(batch_size, -1, -1)

        # Concatenate tokens to form the input sequence
        query_input_sequence = torch.cat(
            [history_cond_token, goal_query, bwd_query, fwd_query], dim=1)

        # Add token type embeddings
        query_token_types = torch.tensor(
            [self.TYPE_HIST_COND, self.TYPE_QUERY_GOAL,
             self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD],
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        query_input_sequence += self.token_type_embedding(query_token_types)

        # Add position embeddings
        query_positions = torch.arange(
            self.config.query_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        query_input_sequence += self.position_embedding(query_positions)

        # Apply attention mask and run through transformer
        query_attn_mask = self._create_sequential_attention_mask(device)
        hidden_states = self.transformer(
            src=query_input_sequence, mask=query_attn_mask)

        # Extract hidden states for each prediction head
        goal_hidden = hidden_states[:, 1]  # Goal query position
        bwd_hidden = hidden_states[:, 2]   # Backward query position
        fwd_hidden = hidden_states[:, 3]    # Forward query position

        # Generate predictions
        results['predicted_goal_latents'] = self.goal_image_latent_head(
            goal_hidden)
        results['predicted_goal_images'] = self.image_decoder(
            results['predicted_goal_latents'])

        # Predict backward states
        bwd_flat = self.backward_state_head(bwd_hidden)
        results['predicted_backward_states'] = bwd_flat.view(
            batch_size, self.config.backward_steps, self.config.state_dim)

        # Predict forward states
        fwd_flat = self.forward_state_head(fwd_hidden)
        results['predicted_forward_states'] = fwd_flat.view(
            batch_size, self.config.forward_steps - 1, self.config.state_dim)

        return results

    def _forward_inference_with_global_cond(self, global_history_condition_emb: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Implementation of the query-based inference path with global history conditioning.

        This method creates a sequence with the global history condition followed by query tokens
        for goal image, backward trajectory, and forward trajectory prediction.

        The sequence structure is:
        [HistoryCondition, GoalQuery, BackwardQuery, ForwardQuery]

        Each token is assigned a specific token type to help the model distinguish between roles.
        """
        batch_size = global_history_condition_emb.shape[0]
        results = {}

        # Prepare the input token sequence
        history_cond_token = global_history_condition_emb.unsqueeze(
            1)  # [B, 1, hidden_dim]
        goal_query = self.goal_image_query_token.expand(
            batch_size, -1, -1)  # [B, 1, hidden_dim]
        bwd_query = self.backward_seq_query_token.expand(
            batch_size, -1, -1)  # [B, 1, hidden_dim]
        fwd_query = self.forward_seq_query_token.expand(
            batch_size, -1, -1)   # [B, 1, hidden_dim]

        query_input_sequence = torch.cat(
            [history_cond_token, goal_query, bwd_query, fwd_query], dim=1)

        # Use the defined token type indices for inference
        query_token_types = torch.tensor(
            [self.TYPE_HIST_COND, self.TYPE_QUERY_GOAL,
                self.TYPE_QUERY_BWD, self.TYPE_QUERY_FWD],
            device=device
        ).unsqueeze(0).expand(batch_size, -1)
        query_input_sequence += self.token_type_embedding(query_token_types)

        query_positions = torch.arange(
            self.config.query_seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        query_input_sequence += self.position_embedding(query_positions)

        attn_mask = self._create_sequential_attention_mask(device)
        hidden_states = self.transformer(
            src=query_input_sequence, mask=attn_mask)

        goal_query_output = hidden_states[:, 1]
        bwd_query_output = hidden_states[:, 2]
        fwd_query_output = hidden_states[:, 3]

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

        img_hist_flat = initial_images.view(
            batch_size * n_obs, self.config.image_channels, self.config.image_size, self.config.image_size
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

        # Combine image latents and state projections for each timestep
        combined_history_per_step = torch.cat(
            [img_latents_history, states_projected_history], dim=-1)

        # Flatten the entire history into a single vector
        # This is the key step for global history conditioning - we compress
        # multiple observation steps into a single conditioning vector
        flat_history_vector = combined_history_per_step.flatten(start_dim=1)

        # Project the flattened history to the model's hidden dimension
        # This creates a single global condition embedding that captures all history information
        global_history_condition_embedding = self.history_to_global_condition_projector(
            flat_history_vector)

        if training:
            if forward_states is None or goal_images is None or backward_states is None:
                raise ValueError(
                    "Ground truth forward_states, goal_images, and backward_states must be provided for training.")
            results = self._forward_training_with_global_cond(
                global_history_condition_embedding,
                forward_states,  # GT for forward path
                # GT for goal image (will be encoded to latent for AR)
                goal_images,
                backward_states,  # GT for backward path
                device
            )
        else:  # Inference
            results = self._forward_inference_with_global_cond(
                global_history_condition_embedding, device)

        return results

    @classmethod
    def from_pretrained(cls, path, device=None, **kwargs):  # Standard from_pretrained
        path = Path(path)
        config_path = path / "config.json"
        with open(config_path, "r") as f:
            config_dict = json.load(f)
        config = BidirectionalARTransformerConfig(**config_dict)
        model = cls(config=config, **kwargs)
        model_path = path / "model_final.pth"
        if not model_path.exists():
            candidates = list(path.glob("*.pth"))
            if candidates:
                model_path = candidates[0]
                print(f"Using model checkpoint: {model_path}")
        state_dict = torch.load(model_path, map_location=torch.device(
            'cpu') if device is None else device)
        model.load_state_dict(state_dict)
        if device is not None:
            model = model.to(device)
        return model


def compute_loss(
    model: 'BidirectionalARTransformer',
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """
    Compute losses for the query-based path outputs.

    With the removal of the autoregressive path, we only compute losses
    for the query path predictions.
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
            goal_image_latents_gt = model.image_encoder(targets['goal_images'])
        losses['goal_latent_consistency_loss'] = F.mse_loss(
            predictions['predicted_goal_latents'], goal_image_latents_gt)

    # Loss weighting - simplified without AR losses
    weights = {
        'forward_state_loss': 1.0,
        'backward_state_loss': 1.0,
        'goal_image_loss': 2.0,
        'goal_latent_consistency_loss': 1.0,
    }
    total_loss = torch.tensor(
        0.0, device=predictions[next(iter(predictions))].device)
    for loss_name, loss_value in losses.items():
        if loss_name in weights and loss_value is not None:  # Check for None
            total_loss += weights.get(loss_name, 1.0) * loss_value
    losses['total_loss'] = total_loss
    return losses
