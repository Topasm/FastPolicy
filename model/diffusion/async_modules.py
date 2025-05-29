#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from torch import Tensor
import math
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.async_training import asyn_t_seq


class DiffusionSinusoidalPosEmb(nn.Module):
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

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


class DenoisingHead(nn.Module):
    """Simple MLP head to predict the noise or sample."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input: x [B, T, D_transformer]
        Output: [B, T, D_action]
        """
        return self.net(x)


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

    def async_conditional_sample(
        self,
        current_input_normalized: Tensor,  # This is your u_pred from previous step
        global_cond: Tensor
    ) -> Tensor:
        device = current_input_normalized.device
        batch_size = current_input_normalized.shape[0]
        horizon = self.config.horizon  # e.g., 16

        # Configurable parameters for feedback denoising
        # These should ideally come from self.config
        feedback_denoising_steps = getattr(
            self.config, "feedback_denoising_steps", 20)  # e.g., 20
        async_gap = getattr(self.config, "feedback_async_gap", 3)  # e.g., 3

        sample = current_input_normalized.clone()  # Start with the previous plan

        # This is a simplified loop. Proper integration with noise_scheduler.timesteps is better.
        # The timesteps here should go from feedback_denoising_steps-1 down to 0.
        # This range corresponds to how much "denoising work" the feedback model does.
        for i in range(feedback_denoising_steps):
            # Current effective "denoising step" for the feedback loop
            # This base_t decreases, signifying further denoising
            current_denoising_step = feedback_denoising_steps - 1 - i

            # Generate asynchronous timesteps for the DiT input
            # The base_t for asyn_t_seq should reflect the current_denoising_step
            # This makes all frames in the sequence "less noisy" by one conceptual step
            base_t_for_async = torch.full(
                (batch_size,),
                # This base_t is for the asyn_t_seq internal logic
                fill_value=current_denoising_step,
                dtype=torch.long,
                device=device
            )
            async_ts_for_transformer = asyn_t_seq(
                base_t_for_async, async_gap, horizon)

            # The 't' for the DDPMScheduler.step (if used) would be current_denoising_step
            # This 't' is what the overall sample is considered to be at.
            scheduler_t = torch.tensor(
                [current_denoising_step] * batch_size, device=device)

            model_output = self.async_transformer(
                noisy_input=sample,
                timesteps=async_ts_for_transformer,  # (B, 16)
                global_cond=global_cond,
                async_mode=True
            )

            # Use scheduler's step method (or its logic) to get prev_sample
            # This is the tricky part if prediction_type is epsilon, as scheduler.step expects a single t
            # If prediction_type is "sample", model_output is already the denoised sample
            if self.config.prediction_type == "sample":
                sample = model_output
            elif self.config.prediction_type == "epsilon":
                # This requires careful adaptation of the DDPMScheduler.step logic
                # to handle a sample that was denoised using per-token async_ts.
                # For simplicity, if your async_transformer is very powerful,
                # you might assume its output (after x0 reconstruction) is good enough.
                # Or, you approximate using the t of the first frame for the whole sequence.
                # Example of x0 reconstruction (needs to be refined for per-token t)
                # This is a rough approximation if you must use scheduler.step logic:
                # Consider scheduler_t (current denoising progress) for the step
                # scalar for the step function's t
                temp_scheduler_t_for_step = current_denoising_step

                # The DDPMScheduler.step function needs a (B,) timestep
                # but our model_output was conditioned on (B,T) async_ts.
                # This is where the direct application of the standard step function becomes problematic.
                # A proper implementation might involve predicting x0 per token using its async_t
                # and then averaging or taking the first token's x0.

                # <<< Placeholder: Simplification - Assume model_output (epsilon) can be used with a representative t >>>
                # This is a significant simplification and might not be robust.
                # A more robust way is to ensure async_transformer learns to predict x0 directly for this mode,
                # or implement a custom "async_step" function.

                # Option 1: predict x0 per frame and use that (if model predicts epsilon)
                # This part needs careful thought.
                # alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
                # sqrt_alpha_prod_t_async = torch.sqrt(alphas_cumprod[async_ts_for_transformer]).unsqueeze(-1)
                # sqrt_one_minus_alpha_prod_t_async = torch.sqrt(1.0 - alphas_cumprod[async_ts_for_transformer]).unsqueeze(-1)
                # pred_original_sample_async = (sample - sqrt_one_minus_alpha_prod_t_async * model_output) / sqrt_alpha_prod_t_async
                # sample = pred_original_sample_async # This is now the "denoised" sample, effectively x0 for each frame
                # For the next iteration of *this loop*, this 'sample' (which is x0_pred) needs to be
                # re-noised to the next (lower) effective t if the loop continues.
                # This gets very complex.

                # Option 2: If the feedback model is trained to denoise significantly in one go
                # (e.g. if base_t for training was always low, like 0 to gap_timesteps)
                # then one pass of async_transformer might be enough to give a good x0_pred.
                # In this case, the loop for `feedback_denoising_steps` might only run once, or not at all,
                # and async_transformer directly outputs the refined plan.
                # This is what your `async_conditional_sample` sketch previously suggested.
                # Let's stick to that simpler interpretation for now, assuming one pass is sufficient for feedback update:

                if i == feedback_denoising_steps - 1:  # Only use the result of the last step
                    if self.config.prediction_type == "epsilon":
                        # Reconstruct x0 using the t of the first frame as representative for the whole sequence
                        t_for_x0_est = async_ts_for_transformer[:, 0]  # (B,)
                        alpha_prod_t = self.noise_scheduler.alphas_cumprod[t_for_x0_est].to(
                            device)  # (B,)
                        beta_prod_t = 1.0 - alpha_prod_t  # (B,)

                        alpha_prod_t = alpha_prod_t.view(
                            batch_size, 1, 1)  # for broadcasting
                        beta_prod_t = beta_prod_t.view(batch_size, 1, 1)

                        pred_original_sample = (
                            sample - (beta_prod_t**0.5) * model_output) / (alpha_prod_t**0.5)
                        sample = pred_original_sample
                    # If 'sample', it's already handled above

            # If the loop runs only once (feedback_denoising_steps=1), this is simpler:
            # One pass of async_transformer, then reconstruct x0 if needed.
            # This interpretation aligns with training where base_t is low.
            # The model learns: given x_t (where t is low, effectively u_pred) and s_k (global_cond),
            # predict x_0 (the refined plan).

            # Let's assume the simpler "one-shot refinement" for now, which means the loop
            # for feedback_denoising_steps is not strictly needed if the model is trained for it.
            # The code you uncommented in `async_conditional_sample` before suggested one call.

            # Simplified one-shot refinement (assuming DenoisingTransformer is trained for this):
            # This means the `effective_base_t_for_feedback` should be a fixed low value,
            # representing the "noise level" of u_pred.
            fixed_low_base_t = torch.full(
                (batch_size,),
                # example, make this configurable
                fill_value=self.config.get("feedback_fixed_base_t", 5),
                dtype=torch.long, device=device
            )
            async_ts = asyn_t_seq(fixed_low_base_t, async_gap, horizon)

            model_output = self.async_transformer(  # Renamed from pred_noise_or_sample
                noisy_input=sample,  # This is u_pred_normalized
                timesteps=async_ts,
                global_cond=global_cond,
                async_mode=True
            )
            if self.config.prediction_type == "epsilon":
                t_for_x0_est = async_ts[:, 0]
                alpha_prod_t_sched = self.noise_scheduler.alphas_cumprod.to(
                    device)  # Ensure it's a tensor

                # Ensure t_for_x0_est is within bounds
                t_for_x0_est = torch.clamp(
                    t_for_x0_est, 0, alpha_prod_t_sched.shape[0] - 1)

                alpha_prod_t = alpha_prod_t_sched[t_for_x0_est]

                # Check if alpha_prod_t is a scalar and if so, expand
                # If it's a scalar (e.g. if batch_size was 1 and t_for_x0_est was scalar)
                if alpha_prod_t.ndim == 0:
                    alpha_prod_t = alpha_prod_t.unsqueeze(0)  # Make it (1,)

                beta_prod_t = 1.0 - alpha_prod_t

                alpha_prod_t_reshaped = alpha_prod_t.view(-1, 1, 1)
                beta_prod_t_reshaped = beta_prod_t.view(-1, 1, 1)

                pred_original_sample = (
                    sample - (beta_prod_t_reshaped**0.5) * model_output) / (alpha_prod_t_reshaped**0.5)
                final_plan_normalized = pred_original_sample
            elif self.config.prediction_type == "sample":
                final_plan_normalized = model_output
            else:
                raise ValueError(
                    f"Unsupported prediction type: {self.config.prediction_type}")

            return final_plan_normalized
