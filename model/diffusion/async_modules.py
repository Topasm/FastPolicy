#!/usr/bin/env python
"""
Async modules for RT-Diffusion that support asynchronous diffusion sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.diffusion_modules import DenoisingHead


# Extended version of DiffusionSinusoidalPosEmb that can handle 2D tensor inputs
class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need.
    Extended to support both 1D and 2D tensor inputs for async diffusion.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        """
        Input:
            x: (B,) or (B, T) tensor of timesteps
        Output:
            (B, dim) or (B, T, dim) tensor of embeddings
        """
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)

        # Handle both (B,) and (B, T) shapes
        if x.dim() == 1:
            # Original case: (B,) -> (B, dim)
            emb = x.unsqueeze(-1) * emb.unsqueeze(0)
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        else:
            # New case: (B, T) -> (B, T, dim)
            emb = x.unsqueeze(-1) * emb.unsqueeze(0).unsqueeze(0)
            emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        return emb


class AsyncDitBlock(nn.Module):
    """
    A Diffusion Transformer block with adaptive layer norm (AdaLN-Zero)
    that supports token-wise asynchronous conditioning.
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
            c: (B, SeqLen, Dim) or (B, Dim) Conditioning vector 
               (can be per-token in async mode or global in standard mode)
        Returns:
            (B, SeqLen, Dim) Output sequence
        """
        B, T, D = x.shape

        # Handle both standard and async conditioning
        if c.dim() == 2:  # (B, Dim) - standard global conditioning
            # AdaLN-Zero modulation with broadcast
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(
                c).chunk(6, dim=1)  # Each is (B, D)

            # Apply modulation to norm1 and self-attention with broadcasting
            x_norm1 = self.norm1(x)  # (B, T, D)
            x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)
                                 ) + shift_msa.unsqueeze(1)
            attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
            x = x + gate_msa.unsqueeze(1) * attn_output

            # Apply modulation to norm2 and MLP with broadcasting
            x_norm2 = self.norm2(x)
            x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)
                                 ) + shift_mlp.unsqueeze(1)
            mlp_output = self.mlp(x_norm2)
            x = x + gate_mlp.unsqueeze(1) * mlp_output

        else:  # (B, T, Dim) - async token-wise conditioning
            # AdaLN-Zero modulation per token
            modulations = self.adaLN_modulation(c)  # (B, T, 6*D)
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulations.chunk(
                6, dim=-1)
            # Each is (B, T, D)

            # Apply modulation to norm1 and self-attention
            x_norm1 = self.norm1(x)  # (B, T, D)
            x_norm1 = x_norm1 * (1 + scale_msa) + shift_msa
            attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
            x = x + gate_msa * attn_output

            # Apply modulation to norm2 and MLP
            x_norm2 = self.norm2(x)
            x_norm2 = x_norm2 * (1 + scale_mlp) + shift_mlp
            mlp_output = self.mlp(x_norm2)
            x = x + gate_mlp * mlp_output

        return x


class DenoisingTransformer(nn.Module):
    """
    A modified Diffusion Transformer that supports asynchronous denoising.
    Takes noisy data with per-token timesteps, global conditioning, and predicts noise or clean data.
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
        self.global_cond_embed = nn.Linear(
            transformer_dim + global_cond_dim, transformer_dim)

        # Token conditioning projection (for combining per-token time embeds with global)
        self.token_cond_proj = nn.Linear(transformer_dim * 2, transformer_dim)

        # Learnable positional embedding for the target sequence
        # Use horizon + 1 to match the saved model (includes extra position for conditioning)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.horizon + 1, transformer_dim))

        # 2. Transformer Blocks
        self.blocks = nn.ModuleList([
            AsyncDitBlock(transformer_dim, config.transformer_num_heads)
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

        # Token-wise AdaLN-Zero modulation for the final layer norm (async mode)
        self.token_adaLN_modulation_out = nn.Sequential(
            nn.SiLU(),
            # scale/shift for norm_out
            nn.Linear(transformer_dim, 2 * transformer_dim)
        )

        self.denoising_head = DenoisingHead(
            input_dim=transformer_dim,
            hidden_dim=transformer_dim,
            output_dim=output_dim
        )

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding and input embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.input_embed.weight, std=0.02)
        nn.init.constant_(self.input_embed.bias, 0)

        # Initialize transformer blocks and MLP heads
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
        nn.init.constant_(self.token_adaLN_modulation_out[-1].weight, 0)
        nn.init.constant_(self.token_adaLN_modulation_out[-1].bias, 0)
        # Zero-out output projection of DenoisingHead
        nn.init.constant_(self.denoising_head.net[-1].weight, 0)
        nn.init.constant_(self.denoising_head.net[-1].bias, 0)

    def forward(self, noisy_input: Tensor, timesteps: Tensor, global_cond: Tensor, async_mode: bool = False) -> Tensor:
        """
        Args:
            noisy_input: (B, T_horizon, output_dim) Noisy data (actions or states).
            timesteps: (B,) or (B, T_horizon) Diffusion timesteps (async_mode=True for second case).
            global_cond: (B, global_cond_dim) Global conditioning vector.
            async_mode: If True, use token-wise timesteps and conditioning.
        Returns:
            (B, T_horizon, output_dim) Predicted noise or clean data.
        """
        B, T, D = noisy_input.shape
        device = noisy_input.device

        # 1. Embeddings
        # Process the input
        input_emb = self.input_embed(noisy_input)  # (B, T, d_model)

        # Add positional embedding
        if T <= self.pos_embed.shape[1]:
            pos_embed = self.pos_embed[:, :T, :]
        else:
            pos_embed = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=T,
                mode='linear',
                align_corners=False
            ).transpose(1, 2)
        x = input_emb + pos_embed

        # Process time embeddings - differently depending on async mode
        if async_mode:
            # timesteps is (B, T) - get embeddings per token
            assert timesteps.shape == (
                B, T), "In async mode, timesteps should be (B, T)"
            time_emb = self.time_embed(timesteps)  # (B, T, d_model)

            # Process global conditioning separately
            # Use first timestep for global
            global_time_emb = self.time_embed(timesteps[:, 0])
            cond = torch.cat([global_time_emb, global_cond], dim=1)
            c_global = self.global_cond_embed(cond)  # (B, d_model)

            # Combine token-wise time emb with global conditioning
            c_global_expanded = c_global.unsqueeze(
                1).expand(-1, T, -1)  # (B, T, d_model)
            token_cond = torch.cat(
                [time_emb, c_global_expanded], dim=-1)  # (B, T, 2*d_model)
            c_token = self.token_cond_proj(token_cond)  # (B, T, d_model)

            # Apply transformer blocks with token-wise conditioning
            for block in self.blocks:
                x = block(x, c_token)

            # Apply token-wise AdaLN-Zero modulation to the final layer norm
            shift_out, scale_out = self.token_adaLN_modulation_out(
                c_token).chunk(2, dim=-1)
            x = self.norm_out(x)
            x = x * (1 + scale_out) + shift_out

        else:
            # Standard mode - timesteps is (B,)
            assert timesteps.dim() == 1, "In standard mode, timesteps should be (B,)"
            time_emb = self.time_embed(timesteps)  # (B, d_model)

            # Standard global conditioning
            # (B, d_model + global_cond_dim)
            cond = torch.cat([time_emb, global_cond], dim=1)
            c = self.global_cond_embed(cond)  # (B, d_model)

            # Apply transformer blocks with global conditioning
            for block in self.blocks:
                x = block(x, c)

            # Standard AdaLN-Zero modulation for the final layer norm
            shift_out, scale_out = self.adaLN_modulation_out(c).chunk(2, dim=1)
            x = self.norm_out(x)
            x = x * (1 + scale_out.unsqueeze(1)) + shift_out.unsqueeze(1)

        # Apply denoising head to get output
        pred = self.denoising_head(x)  # (B, T, D)

        return pred
