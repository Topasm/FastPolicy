import math
from typing import Callable
import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import torchvision
from model.diffusion.configuration_mymodel import DiffusionConfig


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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.horizon, transformer_dim))

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
        B, T, _ = noisy_input.shape
        device = noisy_input.device

        # 1. Embeddings
        # (B, transformer_dim)
        time_emb = self.time_embed(timesteps)
        # (B, transformer_dim + global_cond_dim)
        cond = torch.cat([time_emb, global_cond], dim=1)
        # (B, transformer_dim) - Combined conditioning for AdaLN
        c = self.cond_embed(cond)

        # (B, T, transformer_dim)
        input_emb = self.input_embed(noisy_input)
        # Add positional embedding
        x = input_emb + self.pos_embed  # Uses broadcasting

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
