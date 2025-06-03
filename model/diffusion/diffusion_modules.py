import math
from typing import Callable, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE, OBS_ENV
import einops


def _make_noise_scheduler(name: str, **kwargs: dict) -> DDPMScheduler | DDIMScheduler:
    """Create a noise scheduler based on the specified name.

    Args:
        name: The scheduler type, either "DDPM" or "DDIM"
        **kwargs: Additional arguments to pass to the scheduler constructor

    Returns:
        An instance of the specified noise scheduler

    Raises:
        ValueError: If an unsupported noise scheduler type is specified
    """
    if name == "DDPM":
        return DDPMScheduler(**kwargs)
    elif name == "DDIM":
        return DDIMScheduler(**kwargs)
    else:
        raise ValueError(f"Unsupported noise scheduler type {name}")


class SpatialSoftmax(nn.Module):
    """
    Spatial Soft Argmax operation described in "Deep Spatial Autoencoders for Visuomotor Learning" by Finn et al.
    (https://arxiv.org/pdf/1509.06113). A minimal port of the robomimic implementation.

    At a high level, this takes 2D feature maps (from a convnet/ViT) and returns the "center of mass"
    of activations of each channel, i.e., keypoints in the image space for the policy to focus on.

    Example: take feature maps of size (512x10x12). We generate a grid of normalized coordinates (10x12x2):
    -----------------------------------------------------
    | (-1., -1.)   | (-0.82, -1.)   | ... | (1., -1.)   |
    | (-1., -0.78) | (-0.82, -0.78) | ... | (1., -0.78) |
    | ...          | ...            | ... | ...         |
    | (-1., 1.)    | (-0.82, 1.)    | ... | (1., 1.)    |
    -----------------------------------------------------
    This is achieved by applying channel-wise softmax over the activations (512x120) and computing the dot
    product with the coordinates (120x2) to get expected points of maximal activation (512x2).

    The example above results in 512 keypoints (corresponding to the 512 input channels). We can optionally
    provide num_kp != None to control the number of keypoints. This is achieved by a first applying a learnable
    linear mapping (in_channels, H, W) -> (num_kp, H, W).
    """

    def __init__(self, input_shape, num_kp=None):
        """
        Args:
            input_shape (list): (C, H, W) input feature map shape.
            num_kp (int): number of keypoints in output. If None, output will have the same number of channels as input.
        """
        super().__init__()

        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_kp is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_kp, kernel_size=1)
            self._out_c = num_kp
        else:
            self.nets = None
            self._out_c = self._in_c

        # we could use torch.linspace directly but that seems to behave slightly differently than numpy
        # and causes a small degradation in pc_success of pre-trained models.
        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self._in_w), np.linspace(-1.0, 1.0, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(
            self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(
            self._in_h * self._in_w, 1)).float()
        # register as buffer so it's moved to the correct device.
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: (B, C, H, W) input feature maps.
        Returns:
            (B, K, 2) image-space coordinates of keypoints.
        """
        if self.nets is not None:
            features = self.nets(features)

        # [B, K, H, W] -> [B * K, H * W] where K is number of keypoints
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2d softmax normalization
        attention = F.softmax(features, dim=-1)
        # [B * K, H * W] x [H * W, 2] -> [B * K, 2] for spatial coordinate mean in x and y dimensions
        expected_xy = attention @ self.pos_grid
        # reshape to [B, K, 2]
        feature_keypoints = expected_xy.view(-1, self._out_c, 2)

        return feature_keypoints


class DiffusionRgbEncoder(nn.Module):
    """Encodes an RGB image into a 1D feature vector.

    Includes the ability to normalize and crop the image first.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        # Set up optional preprocessing.
        if config.crop_shape is not None:
            self.do_crop = True
            # Always use center crop for eval
            self.center_crop = torchvision.transforms.CenterCrop(
                config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(
                    config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # Set up backbone.
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            weights=config.pretrained_backbone_weights
        )
        # Note: This assumes that the layer4 feature map is children()[-3]
        # TODO(alexander-soare): Use a safer alternative like `torchvision.models.feature_extraction`.
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(
                    # Ensure num_groups is at least 1 and divides num_channels if possible
                    num_groups=max(1, x.num_features //
                                   16) if x.num_features >= 16 else 1,
                    num_channels=x.num_features
                ),
            )

        # Set up pooling and final layers.
        # Use a dry run to get the feature map shape.
        images_shape = next(iter(config.image_features.values())).shape
        dummy_shape_h_w = config.crop_shape if config.crop_shape is not None else images_shape[
            1:]
        dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
        # Use a buffer for the dummy input to ensure it's on the correct device
        self.register_buffer("dummy_input", torch.zeros(dummy_shape))
        with torch.no_grad():
            feature_map_shape = self.backbone(self.dummy_input).shape[1:]

        self.pool = SpatialSoftmax(
            feature_map_shape, num_kp=config.spatial_softmax_num_keypoints)
        # The output dim of SpatialSoftmax is num_kp * 2
        pool_out_dim = config.spatial_softmax_num_keypoints * 2
        # Project to transformer dimension
        # Output dim should match transformer dim
        self.feature_dim = config.transformer_dim
        self.out = nn.Linear(pool_out_dim, self.feature_dim)
        self.layer_norm = nn.LayerNorm(self.feature_dim)  # Add layer norm

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor with pixel values in [0, 1].
        Returns:
            (B, D) image feature, where D is config.transformer_dim.
        """
        # Preprocess: maybe crop (if it was set up in the __init__).
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                # Always use center crop for eval.
                x = self.center_crop(x)
        # Extract backbone feature.
        x = self.backbone(x)
        # Pool features
        x = self.pool(x)  # (B, K, 2)
        # Flatten pooled features
        x = torch.flatten(x, start_dim=1)  # (B, K * 2)
        # Final linear layer and layer norm
        x = self.layer_norm(self.out(x))  # (B, D)
        return x


def _replace_submodules(
    root_module: nn.Module, predicate: Callable[[nn.Module], bool], func: Callable[[nn.Module], nn.Module]
) -> nn.Module:
    """
    Args:
        root_module: The module for which the submodules need to be replaced
        predicate: Takes a module as an argument and must return True if the that module is to be replaced.
        func: Takes a module as an argument and returns a new module to replace it with.
    Returns:
        The root module with its submodules replaced.
    """
    if predicate(root_module):
        return func(root_module)

    replace_list = [k.split(".") for k, m in root_module.named_modules(
        remove_duplicate=True) if predicate(m)]
    for *parents, k in replace_list:
        parent_module = root_module
        if len(parents) > 0:
            parent_module = root_module.get_submodule(".".join(parents))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all matching modules are replaced
    assert not any(predicate(m)
                   for _, m in root_module.named_modules(remove_duplicate=True))
    return root_module


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x: (B,) tensor of timesteps
        Output:
            (B, dim) tensor of embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)  # (B, dim)
        return emb


class DenoisingHead(nn.Module):
    """Simple MLP head to predict the noise or sample."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),  # Use GELU like in many transformers
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [B, T, D_transformer]
        Output: [B, T, D_action]
        """
        return self.net(x)


# --- Diffusion Transformer Components ---

class DitBlock(nn.Module):
    """
    A basic Diffusion Transformer block with adaptive layer norm (AdaLN-Zero).
    Uses self-attention. Cross-attention can be added if needed for conditioning.
    """

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(
            hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # AdaLN-Zero modulation (modulates scale and shift)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            # scale/shift for norm1, attn, norm2, mlp
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        """
        Args:
            x: (B, SeqLen, Dim) Input sequence
            c: (B, Dim) Conditioning vector (e.g., timestep embedding + global condition)
        Returns:
            (B, SeqLen, Dim) Output sequence
        """
        # AdaLN-Zero modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
            c).chunk(6, dim=1)

        # Apply modulation to norm1 and self-attention
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + \
            shift_msa.unsqueeze(1)
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # Apply modulation to norm2 and MLP
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + \
            shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x


class DiffusionTransformer(nn.Module):
    """
    A Diffusion Transformer model.
    Takes noisy data (actions or states), timestep, and global conditioning, predicts noise or clean data.
    """

    def __init__(self, config: DiffusionConfig, global_cond_dim: int, output_dim: int):
        super().__init__()
        self.config = config
        transformer_dim = config.transformer_dim

        # 1. Embeddings
        self.time_embed = nn.Sequential(
            DiffusionSinusoidalPosEmb(transformer_dim),
            nn.Linear(transformer_dim, transformer_dim),
            nn.GELU(),
            nn.Linear(transformer_dim, transformer_dim),
        )
        # Input embedding (for actions or states)
        self.input_embed = nn.Linear(output_dim, transformer_dim)
        # Conditioning embedding (time + global condition)
        self.cond_embed = nn.Linear(
            transformer_dim + global_cond_dim, transformer_dim)

        # Learnable positional embedding for the target sequence
        # Use horizon + 1 to match the saved model (includes extra position for conditioning)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.horizon + 1, transformer_dim))

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            DitBlock(transformer_dim, config.transformer_num_heads)
            for _ in range(config.transformer_num_layers)
        ])

        # 3. Final Layer Norm and Denoising Head
        self.norm_out = nn.LayerNorm(
            transformer_dim, elementwise_affine=False, eps=1e-6)
        # AdaLN-Zero modulation for the final layer norm
        self.adaLN_modulation_out = nn.Sequential(
            nn.SiLU(),
            # scale/shift for norm_out
            nn.Linear(transformer_dim, 2 * transformer_dim)
        )
        self.denoising_head = DenoisingHead(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,  # Can adjust hidden dim if needed
            output_dim=output_dim  # Use the provided output_dim
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding and input embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.input_embed.weight, std=0.02)
        nn.init.constant_(self.input_embed.bias, 0)

        # Initialize transformer blocks and MLP heads (like in Vision Transformer)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                if m.elementwise_affine:  # Only if affine params exist
                    nn.init.constant_(m.bias, 0)
                    nn.init.constant_(m.weight, 1.0)

        # Zero-out adaLN modulation final layer weights
        for m in self.blocks:
            nn.init.constant_(m.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(m.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.adaLN_modulation_out[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation_out[-1].bias, 0)
        # Zero-out output projection of DenoisingHead
        nn.init.constant_(self.denoising_head.net[-1].weight, 0)
        nn.init.constant_(self.denoising_head.net[-1].bias, 0)

    def forward(self, noisy_input: Tensor, timesteps: Tensor, global_cond: Tensor) -> Tensor:
        """
        Args:
            noisy_input: (B, T_horizon, output_dim) Noisy data (actions or states).
            timesteps: (B,) Diffusion timesteps.
            global_cond: (B, global_cond_dim) Global conditioning vector.
        Returns:
            (B, T_horizon, output_dim) Predicted noise or clean data.
        """
        B, T, D = noisy_input.shape

        # 1. Embeddings
        # (B, transformer_dim)
        time_emb = self.time_embed(timesteps)
        # (B, transformer_dim + global_cond_dim)
        cond = torch.cat([time_emb, global_cond], dim=1)
        # (B, transformer_dim) - Combined conditioning for AdaLN
        c = self.cond_embed(cond)

        # (B, T, transformer_dim)
        input_emb = self.input_embed(noisy_input)

        # Add positional embedding - handle variable sequence length
        # pos_embed is now [1, horizon+1, transformer_dim], take first T positions
        if T <= self.pos_embed.shape[1]:
            # Use the first T positions from the position embedding
            pos_embed = self.pos_embed[:, :T, :]
        else:
            # If T is larger than available positions, interpolate
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),  # [1, d_model, seq_len]
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)  # [1, seq_len, d_model]

        # Add positional embedding
        x = input_emb + pos_embed  # Uses broadcasting

        # 2. Apply Transformer blocks
        for block in self.blocks:
            x = block(x, c)

        # 3. Final Layer Norm and Head
        # Apply AdaLN-Zero modulation to the final layer norm
        shift_out, scale_out = self.adaLN_modulation_out(c).chunk(2, dim=1)
        x = self.norm_out(x)
        x = x * (1 + scale_out.unsqueeze(1)) + shift_out.unsqueeze(1)

        # (B, T, output_dim)
        pred = self.denoising_head(x)

        return pred


class DiffusionModel(nn.Module):
    """
    Main diffusion model for conditional generation of trajectories.
    Handles preprocessing of observations, conditioning, and diffusion sampling.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Calculate conditioning dimensions
        obs_only_cond_dim = 0
        if self.config.robot_state_feature:
            obs_only_cond_dim += self.config.robot_state_feature.shape[0]

        if self.config.image_features:
            num_images = len(self.config.image_features)
            if self.config.use_separate_rgb_encoder_per_camera:
                self.rgb_encoder = nn.ModuleList(
                    [DiffusionRgbEncoder(config) for _ in range(num_images)])
                if self.rgb_encoder:
                    obs_only_cond_dim += self.rgb_encoder[0].feature_dim * num_images
            else:
                self.rgb_encoder = DiffusionRgbEncoder(config)
                obs_only_cond_dim += self.rgb_encoder.feature_dim * num_images

        if self.config.env_state_feature:
            obs_only_cond_dim += self.config.env_state_feature.shape[0]

        global_cond_dim_per_step = obs_only_cond_dim
        global_cond_dim_total_for_transformer = global_cond_dim_per_step * config.n_obs_steps

        # Verify target configuration
        diffusion_target_key = config.diffusion_target_key
        if diffusion_target_key not in config.output_features:
            raise ValueError(
                f"Diffusion target key '{diffusion_target_key}' not found in config.output_features. "
                f"Ensure config.output_features includes the target. "
                f"Available output_features: {list(config.output_features.keys())}"
            )
        target_feature_spec = config.output_features[diffusion_target_key]
        if not hasattr(target_feature_spec, 'shape') or not target_feature_spec.shape:
            raise ValueError(
                f"Shape for diffusion target key '{diffusion_target_key}' is not properly defined in output_features."
            )
        diffusion_output_dim = target_feature_spec.shape[0]

        # Initialize models
        self.transformer = DiffusionTransformer(
            config,
            global_cond_dim=global_cond_dim_total_for_transformer,
            output_dim=diffusion_output_dim
        )

        # Initialize the async transformer only if needed
        self.async_transformer = None
        if hasattr(config, 'use_async_transformer') and config.use_async_transformer:
            from model.diffusion.async_modules import DenoisingTransformer
            self.async_transformer = DenoisingTransformer(
                config,
                global_cond_dim=global_cond_dim_total_for_transformer,
                output_dim=diffusion_output_dim
            )

        # Initialize noise scheduler
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        # Set inference steps
        if config.num_inference_steps is None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        """Encode image features and concatenate them all together along with the state vector.
           Expects batch to have normalized 'observation.state' and potentially 'observation.image'.
        """
        # Check required keys exist
        if OBS_ROBOT not in batch:
            raise KeyError(
                f"Missing '{OBS_ROBOT}' in batch for _prepare_global_conditioning")

        batch_size = batch[OBS_ROBOT].shape[0]
        n_obs_steps = self.config.n_obs_steps

        if batch[OBS_ROBOT].shape[1] < n_obs_steps:
            raise ValueError(
                f"{OBS_ROBOT} sequence length ({batch[OBS_ROBOT].shape[1]}) "
                f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
            )
        cond_state = batch[OBS_ROBOT][:, :n_obs_steps, :]
        global_cond_feats = [cond_state]

        # Check if images are configured AND present in the batch before processing
        if self.config.image_features and OBS_IMAGE in batch:
            images = batch[OBS_IMAGE]
            _B = images.shape[0]
            n_img_steps = images.shape[1]

            # Handle different image tensor shapes
            if len(images.shape) == 6:  # Shape: [b, t, n_cam, c, h, w]
                # Special case for 6D tensor
                if images.shape[2] == 1:  # If there's just one camera, squeeze that dimension
                    images = images.squeeze(2)  # Convert to [b, t, c, h, w]
                else:
                    # Use the _encode_images method which can handle multi-camera input
                    try:
                        img_features = self._encode_images(images)
                        global_cond_feats.append(img_features)
                        # Skip the rest of the processing for this case
                        goto_next_condition = True
                    except Exception as e:
                        print(f"Error processing 6D image tensor: {e}")
                        print(f"Image shape: {images.shape}")
                        # Create a fallback feature tensor
                        img_features = torch.zeros((batch_size, n_obs_steps, self.config.transformer_dim),
                                                   device=batch["observation.state"].device)
                        global_cond_feats.append(img_features)
                        goto_next_condition = True

            # If we need to skip to the next condition
            if 'goto_next_condition' in locals() and goto_next_condition:
                pass
            # Standard processing for 5D tensor [b, t, c, h, w]
            elif len(images.shape) == 5:
                if n_img_steps != n_obs_steps:
                    raise ValueError(
                        f"Image sequence length ({n_img_steps}) in batch does not match "
                        f"configured n_obs_steps ({n_obs_steps}). Check dataset delta_timestamps "
                        f"and policy config."
                    )
                assert _B == batch_size

                try:
                    images_reshaped = einops.rearrange(
                        images, "b t c h w -> (b t) c h w")
                    img_features = self.rgb_encoder(images_reshaped)
                    img_features = einops.rearrange(
                        img_features, "(b t) d -> b t d", b=batch_size, t=n_obs_steps
                    )
                    global_cond_feats.append(img_features)
                except Exception as e:
                    print(f"Error during standard image processing: {e}")
                    print(f"Image shape: {images.shape}")
                    # Create a fallback feature tensor
                    img_features = torch.zeros((batch_size, n_obs_steps, self.config.transformer_dim),
                                               device=batch["observation.state"].device)
                    global_cond_feats.append(img_features)
            else:
                # Unknown image tensor format
                raise ValueError(
                    f"Unsupported image tensor shape: {images.shape}")

        elif self.config.image_features and "observation.image" not in batch:
            # If images configured but not provided in this specific batch, print warning
            print("Warning: image_features configured but 'observation.image' not found in batch for _prepare_global_conditioning.")
            # Continue without image features for this batch

        # Check if env state is configured AND present
        if self.config.env_state_feature and OBS_ENV in batch:
            if batch[OBS_ENV].shape[1] < n_obs_steps:
                raise ValueError(
                    f"{OBS_ENV} sequence length ({batch[OBS_ENV].shape[1]}) "
                    f"is shorter than required n_obs_steps ({n_obs_steps}) for conditioning."
                )
            cond_env_state = batch[OBS_ENV][:, :n_obs_steps, :]
            global_cond_feats.append(cond_env_state)
        elif self.config.env_state_feature and OBS_ENV not in batch:
            print(
                f"Warning: env_state_feature configured but '{OBS_ENV}' not found in batch for _prepare_global_conditioning.")
            # Continue without env state features for this batch

        concatenated_features = torch.cat(global_cond_feats, dim=-1)
        global_cond = concatenated_features.flatten(start_dim=1)
        return global_cond

    def conditional_sample(self, batch_size: int, global_cond: Tensor, generator: Optional[torch.Generator] = None) -> Tensor:
        """
        Perform conditional sampling from the diffusion model.

        Args:
            batch_size: Number of samples to generate
            global_cond: Global conditioning tensor
            generator: Optional random generator for reproducibility

        Returns:
            Generated trajectory samples
        """
        device = global_cond.device
        dtype = global_cond.dtype

        sample_output_dim = self.transformer.denoising_head.net[-1].out_features

        # Start with random noise
        sample = torch.randn(
            size=(batch_size, self.config.horizon, sample_output_dim),
            dtype=dtype, device=device, generator=generator,
        )

        # Set up noise scheduler timesteps
        self.noise_scheduler.set_timesteps(
            self.num_inference_steps, device=device)

        # Iteratively denoise
        for t in self.noise_scheduler.timesteps:
            model_output = self.transformer(
                sample,
                torch.full((batch_size,), t, dtype=torch.long,
                           device=sample.device),
                global_cond=global_cond,
            )
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample

        return sample

    def async_conditional_sample(self, current_input_normalized: Tensor, global_cond: Tensor,
                                 generator: Optional[torch.Generator] = None) -> Tensor:
        """
        Perform asynchronous conditional sampling for iterative refinement.

        Args:
            current_input_normalized: Current normalized trajectory to refine
            global_cond: Global conditioning tensor
            generator: Optional random generator for reproducibility

        Returns:
            Refined trajectory samples
        """
        # Check if async transformer is available
        if self.async_transformer is None:
            raise ValueError(
                "async_transformer is not initialized. Set config.use_async_transformer=True to use this method.")

        device = current_input_normalized.device
        sample = current_input_normalized.clone()

        # Get number of refinement steps from config
        async_num_steps = getattr(
            self.config, 'async_refinement_steps', self.num_inference_steps)

        if async_num_steps == 0 and self.num_inference_steps > 0:
            async_num_steps = 1
        elif self.num_inference_steps == 0:
            async_num_steps = 0

        if async_num_steps == 0:
            return sample

        # Set up noise scheduler timesteps
        self.noise_scheduler.set_timesteps(async_num_steps, device=device)
        timesteps_to_refine = self.noise_scheduler.timesteps

        if not len(timesteps_to_refine):
            return sample

        # Iteratively refine
        for t in timesteps_to_refine:
            model_input_timesteps = t.repeat(sample.shape[0])
            use_async_mode_for_transformer = False

            model_output = self.async_transformer(
                sample,
                model_input_timesteps,
                global_cond=global_cond,
                async_mode=use_async_mode_for_transformer
            )
            sample = self.noise_scheduler.step(
                model_output, t, sample, generator=generator).prev_sample

        refined_plan = sample
        return refined_plan

    @torch.no_grad()
    def refine_state_path(self, initial_state_path: Tensor, observation_batch_for_cond: dict[str, Tensor],
                          num_refinement_steps: Optional[int] = None,
                          generator: Optional[torch.Generator] = None) -> Tensor:
        """
        Refine an existing state path using the diffusion model.

        Args:
            initial_state_path: Initial state path to refine
            observation_batch_for_cond: Observation batch for conditioning
            num_refinement_steps: Number of refinement steps (overrides config default)
            generator: Optional random generator

        Returns:
            Refined state path
        """
        device = initial_state_path.device
        dtype = initial_state_path.dtype
        batch_size = initial_state_path.shape[0]

        # Adjust horizon if necessary
        current_horizon = initial_state_path.shape[1]
        target_horizon = self.config.horizon

        if current_horizon != target_horizon:
            if current_horizon > target_horizon:
                initial_state_path_adjusted = initial_state_path[:,
                                                                 :target_horizon]
            else:
                padding_needed = target_horizon - current_horizon
                last_frame = initial_state_path[:, -1:, :]
                padding = last_frame.repeat(1, padding_needed, 1)
                initial_state_path_adjusted = torch.cat(
                    [initial_state_path, padding], dim=1)
        else:
            initial_state_path_adjusted = initial_state_path

        # Prepare conditioning
        global_cond = self._prepare_global_conditioning(
            observation_batch_for_cond)

        # Determine refinement steps
        effective_num_refinement_steps = num_refinement_steps
        if effective_num_refinement_steps is None:
            effective_num_refinement_steps = getattr(self.config, 'num_refinement_steps_default',
                                                     min(10, self.noise_scheduler.config.num_train_timesteps // 20) or 1)

        if effective_num_refinement_steps == 0:
            return initial_state_path_adjusted

        # Set up noise scheduler timesteps
        self.noise_scheduler.set_timesteps(
            effective_num_refinement_steps, device=device)

        if not len(self.noise_scheduler.timesteps):
            return initial_state_path_adjusted

        # Add noise to the initial state path
        noise = torch.randn_like(
            initial_state_path_adjusted, device=device, dtype=dtype)
        start_timestep_for_refinement = self.noise_scheduler.timesteps[0]

        sample = self.noise_scheduler.add_noise(
            initial_state_path_adjusted, noise,
            torch.full((batch_size,), start_timestep_for_refinement,
                       device=device, dtype=torch.long)
        )

        # Refine through denoising steps
        for t in self.noise_scheduler.timesteps:
            model_input_timesteps = torch.full(
                (batch_size,), t, dtype=torch.long, device=device)
            predicted_noise_or_sample = self.transformer(
                sample, model_input_timesteps, global_cond=global_cond
            )
            sample = self.noise_scheduler.step(
                predicted_noise_or_sample, t, sample, generator=generator).prev_sample

        return sample

    def compute_loss(self, trajectory_to_diffuse: Tensor, global_cond: Tensor,
                     batch_info_for_masking: Optional[dict[str, Tensor]] = None) -> Tensor:
        """
        Compute diffusion loss for training.

        Args:
            trajectory_to_diffuse: Target trajectory to diffuse
            global_cond: Global conditioning tensor
            batch_info_for_masking: Optional batch info for masking padding

        Returns:
            Loss tensor (scalar)
        """
        # Generate random noise
        eps = torch.randn(trajectory_to_diffuse.shape,
                          device=trajectory_to_diffuse.device)

        # Sample random timesteps
        timesteps = torch.randint(
            low=0, high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory_to_diffuse.shape[0],),
            device=trajectory_to_diffuse.device,
        ).long()

        # Add noise according to timesteps
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory_to_diffuse, eps, timesteps)

        # Get model prediction
        pred = self.transformer(
            noisy_trajectory, timesteps, global_cond=global_cond)

        # Determine target based on prediction type
        if self.config.prediction_type == "epsilon":
            target = eps
        elif self.config.prediction_type == "sample":
            target = trajectory_to_diffuse
        else:
            raise ValueError(
                f"Unsupported prediction type {self.config.prediction_type}")

        # Compute MSE loss
        loss = F.mse_loss(pred, target, reduction="none")

        # Apply masking for padding if configured
        if self.config.do_mask_loss_for_padding and batch_info_for_masking is not None:
            pad_mask_key = None
            specific_pad_mask_key = f"{self.config.diffusion_target_key}_is_pad"

            if specific_pad_mask_key in batch_info_for_masking:
                pad_mask_key = specific_pad_mask_key
            elif self.config.diffusion_target_key == "action" and "action_is_pad" in batch_info_for_masking:
                pad_mask_key = "action_is_pad"
            elif "is_pad" in batch_info_for_masking and batch_info_for_masking["is_pad"].shape[1] == trajectory_to_diffuse.shape[1]:
                pad_mask_key = "is_pad"

            if pad_mask_key and pad_mask_key in batch_info_for_masking:
                is_pad_mask = batch_info_for_masking[pad_mask_key]
                current_horizon = trajectory_to_diffuse.shape[1]

                if is_pad_mask.shape[1] < current_horizon:
                    padding_amount = current_horizon - is_pad_mask.shape[1]
                    mask_padding = torch.zeros(
                        is_pad_mask.shape[0], padding_amount, dtype=torch.bool, device=is_pad_mask.device)
                    is_pad_mask_adjusted = torch.cat(
                        [is_pad_mask, mask_padding], dim=1)
                else:
                    is_pad_mask_adjusted = is_pad_mask[:, :current_horizon]

                in_episode_bound = ~is_pad_mask_adjusted
                if loss.ndim == 3 and in_episode_bound.ndim == 2:
                    loss = loss * in_episode_bound.unsqueeze(-1)
                elif loss.ndim == in_episode_bound.ndim:
                    loss = loss * in_episode_bound
                else:
                    print(
                        f"Warning: Loss shape {loss.shape} and padding mask shape {in_episode_bound.shape} are not compatible for broadcasting in masking.")

            elif self.config.do_mask_loss_for_padding:
                print(
                    f"Warning: `do_mask_loss_for_padding` is True, but a suitable padding mask key ('{specific_pad_mask_key}', 'action_is_pad', or 'is_pad') was not found in `batch_info_for_masking` for target '{self.config.diffusion_target_key}'. Loss will not be masked for padding.")

        return loss.mean()

    @torch.no_grad()
    def sample_asynchronous_step(self, x_t: Tensor, timestep: int, global_cond: Tensor,
                                 generator: Optional[torch.Generator] = None,
                                 num_async_inference_steps: Optional[int] = None) -> Tensor:
        """
        Perform asynchronous denoising for the CL-DiffPhyCon algorithm.

        This implementation follows Algorithm 1 from the CL-DiffPhyCon paper, performing
        progressive denoising for each token in the sequence according to its physical timestep.
        Each token is denoised from time τ/H·T to 0 through multiple steps.

        Args:
            x_t (Tensor): The noisy input trajectory with varying noise levels per token, shape [B, H, D]
            timestep (int): Initial diffusion timestep (typically T/H)
            global_cond (Tensor): Global conditioning tensor for the transformer model, shape [B, global_cond_dim]
            generator (Optional[torch.Generator]): Random number generator for reproducible sampling
            num_async_inference_steps (Optional[int]): Number of denoising steps to perform (defaults to config value)

        Returns:
            Tensor: Fully denoised trajectory (at diffusion time 0), shape [B, H, D]
        """
        batch_size, horizon, feat_dim = x_t.shape
        device = x_t.device

        # Check if async transformer is available - required for this method
        if self.async_transformer is None:
            raise ValueError(
                "async_transformer is not initialized but required for asynchronous sampling. "
                "Set config.use_async_transformer=True to use this method."
            )

        # Convert scalar timestep to tensor if needed
        if isinstance(timestep, int):
            timestep_tensor = torch.tensor([timestep], device=device)
        else:
            timestep_tensor = timestep

        # Ensure timestep is within valid range
        max_timestep = self.noise_scheduler.config.num_train_timesteps - 1
        valid_timestep = torch.clamp(
            timestep_tensor, 0, max_timestep)[0].item()

        # Determine number of inference steps
        if num_async_inference_steps is None:
            # Default to either a config value or a reasonable fraction of train timesteps
            num_async_inference_steps = getattr(
                self.config, 'num_async_inference_steps',
                min(int(valid_timestep) + 1, max(10,
                    self.noise_scheduler.config.num_train_timesteps // 20))
            )

        # Set up noise scheduler timesteps for the denoising process
        # Starting from the provided timestep (typically T/H) down to 0
        self.noise_scheduler.set_timesteps(
            num_async_inference_steps, device=device
        )

        # Get the actual timestep values for the denoising process
        timesteps_for_denoising = self.noise_scheduler.timesteps

        # Filter timesteps to start from the provided timestep or lower
        valid_timestep_indices = timesteps_for_denoising <= valid_timestep
        if not valid_timestep_indices.any():
            # If no valid timesteps, log warning and use smallest available timestep
            print(f"Warning: No valid timesteps for denoising (requested {valid_timestep}). "
                  f"Using smallest available timestep: {timesteps_for_denoising[0]}")
            # Use at least one step
            filtered_timesteps = timesteps_for_denoising[:1]
        else:
            filtered_timesteps = timesteps_for_denoising[valid_timestep_indices]

        # Initialize the current noisy sample with the input
        sample = x_t.clone()

        # Perform progressive denoising through multiple steps
        for t in filtered_timesteps:
            # Create per-token timesteps tensor with shape [B, H]
            # For the asynchronous mode, each token in the sequence has its own timestep
            token_timesteps = torch.full(
                (batch_size, horizon), t, dtype=torch.long, device=device)

            # Predict noise or sample using the async transformer in async mode
            model_output = self.async_transformer(
                sample,               # Current noisy sample
                token_timesteps,      # Current timesteps for each token
                global_cond=global_cond,  # Global conditioning
                async_mode=True       # Critical: enable async mode for per-token conditioning
            )

            # Step the scheduler to denoise the sample
            # Note: t is already a scalar value that the scheduler can use directly
            try:
                sample = self.noise_scheduler.step(
                    model_output,  # Predicted noise or sample
                    t,             # Current timestep value
                    sample,        # Current noisy sample
                    generator=generator  # Optional RNG for reproducibility
                ).prev_sample
            except Exception as e:
                print(f"Error in noise scheduler step at timestep {t}: {e}")
                # Continue with the current sample if there's an error
                continue

        # Return the completely denoised trajectory
        return sample


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculate the output shape of a module given an input shape.

    Args:
        module: PyTorch module
        input_shape: Input tensor shape (batch_size, channels, height, width)

    Returns:
        Output shape as a tuple
    """
    with torch.no_grad():
        # Create a dummy tensor with the specified input shape
        dummy_input = torch.zeros(input_shape)
        # Forward pass through the module
        out = module(dummy_input)
        return out.shape
