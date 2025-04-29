import torch
import torch.nn as nn
from typing import Optional

from .image_tokenizer import ImageTokenizer
from .denoising_head import DenoisingHead
from .scheduler_wrapper import DiffusionSchedulerWrapper
# Optional: from .transformer_core import LLaMACore # If using a custom core

# Placeholder for Sinusoidal Position Embedding


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        # Input x is expected to be timesteps [B,]
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class VisionConditionedDiffusionPolicy(nn.Module):
    def __init__(self,
                 image_tokenizer: ImageTokenizer,
                 denoising_head: DenoisingHead,
                 scheduler_wrapper: DiffusionSchedulerWrapper,
                 action_dim: int,
                 transformer_dim: int = 512,
                 num_transformer_layers: int = 6,
                 transformer_heads: int = 8,
                 task_embed_dim: Optional[int] = None,
                 use_task_embedding: bool = False,
                 prediction_type: str = 'epsilon',  # or 'sample'
                 ):
        super().__init__()
        self.image_tokenizer = image_tokenizer
        self.denoising_head = denoising_head
        self.scheduler = scheduler_wrapper
        self.action_dim = action_dim
        self.transformer_dim = transformer_dim
        self.use_task_embedding = use_task_embedding
        self.prediction_type = prediction_type

        # --- Embeddings ---
        self.time_embed = SinusoidalPosEmb(transformer_dim)
        self.action_embed = nn.Linear(action_dim, transformer_dim)
        if use_task_embedding:
            assert task_embed_dim is not None, "task_embed_dim must be provided if use_task_embedding is True"
            # Simple projection, could be more complex (e.g., another transformer)
            self.task_embed = nn.Linear(task_embed_dim, transformer_dim)
        else:
            self.task_embed = None

        # --- Transformer Decoder ---
        # Using standard PyTorch TransformerDecoder. Replace with LLaMACore if needed.
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_dim * 4,  # Common practice
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=num_transformer_layers)

        # --- Positional Encoding for Actions ---
        # Learnable positional embedding for the action sequence length
        # Max sequence length needs to be defined (e.g., action horizon)
        # Placeholder: Assume max action sequence length T_a
        # self.action_pos_embed = nn.Parameter(torch.zeros(1, MAX_ACTION_SEQ_LEN, transformer_dim))

    def get_condition_tokens(self, images: torch.Tensor, task_embedding: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Combine vision and optional task embeddings into context tokens."""
        vision_tokens = self.image_tokenizer(
            images)  # [B, N_vision, D_transformer]

        if self.use_task_embedding and task_embedding is not None:
            # Process task embedding
            # If task_embedding is [B, D_task], project and unsqueeze
            if task_embedding.ndim == 2:
                task_tokens = self.task_embed(task_embedding).unsqueeze(
                    1)  # [B, 1, D_transformer]
            # If task_embedding is [B, T_task, D_task], project each step
            elif task_embedding.ndim == 3:
                # [B, T_task, D_transformer]
                task_tokens = self.task_embed(task_embedding)
            else:
                raise ValueError("Invalid task embedding shape")
            # Concatenate vision and task tokens
            # [B, N_vision + N_task, D_transformer]
            context_tokens = torch.cat([vision_tokens, task_tokens], dim=1)
        else:
            context_tokens = vision_tokens  # [B, N_vision, D_transformer]

        return context_tokens

    def forward(self,
                images: torch.Tensor,
                actions: torch.Tensor,
                timesteps: torch.Tensor,
                task_embedding: Optional[torch.Tensor] = None,
                cond_mask: Optional[torch.Tensor] = None  # For CFG
                ) -> torch.Tensor:
        """
        Training forward pass.
        images: [B, T_img, C, H, W]
        actions: [B, T_act, A] (clean actions)
        timesteps: [B,]
        task_embedding: Optional [B, D_task] or [B, T_task, D_task]
        cond_mask: Optional [B,] mask for dropping conditioning (CFG)
        Returns: Predicted noise (epsilon) or predicted clean sample (x0) [B, T_act, A]
        """
        B, T_act, A = actions.shape
        device = actions.device

        # 1. Add noise to actions
        noisy_actions, noise = self.scheduler.add_noise(actions, timesteps)

        # 2. Get condition tokens (vision + task)
        context_tokens = self.get_condition_tokens(images, task_embedding)

        # --- Classifier-Free Guidance (CFG) during training ---
        if cond_mask is not None and self.training:
            # Create unconditional context (e.g., zero out vision/task tokens or use learned null embeddings)
            # Simple approach: zero out context for masked samples
            # A better approach might involve dedicated null embeddings
            uncond_context_tokens = torch.zeros_like(context_tokens)
            context_tokens = torch.where(cond_mask.view(B, 1, 1).expand_as(context_tokens),
                                         context_tokens,
                                         uncond_context_tokens)

        # 3. Embed inputs
        time_emb = self.time_embed(timesteps).unsqueeze(
            1)  # [B, 1, D_transformer]
        action_emb = self.action_embed(
            noisy_actions)  # [B, T_act, D_transformer]

        # Add positional encoding to actions if defined
        # if hasattr(self, 'action_pos_embed'):
        #     action_emb = action_emb + self.action_pos_embed[:, :T_act, :]

        # 4. Prepare Transformer inputs
        # Target sequence for decoder is noisy actions + time embedding
        # Memory for decoder is the context tokens
        # Prepend time embedding to the action sequence
        # [B, 1 + T_act, D_transformer]
        decoder_input = torch.cat([time_emb, action_emb], dim=1)

        # Create target mask (causal mask for decoder input)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(1), device=device)

        # 5. Run Transformer Decoder
        transformer_output = self.transformer_decoder(
            tgt=decoder_input,
            memory=context_tokens,
            tgt_mask=tgt_mask
            # Optional: memory_mask, tgt_key_padding_mask, memory_key_padding_mask
        )
        # Output is [B, 1 + T_act, D_transformer], take only action sequence part
        # [B, T_act, D_transformer]
        action_output_tokens = transformer_output[:, 1:, :]

        # 6. Denoising Head
        predicted_output = self.denoising_head(
            action_output_tokens)  # [B, T_act, A]

        # 7. Return based on prediction type
        if self.prediction_type == 'epsilon':
            return predicted_output  # Model predicts noise
        elif self.prediction_type == 'sample':
            return predicted_output  # Model predicts clean sample x0
        else:
            raise ValueError(
                f"Unknown prediction type: {self.prediction_type}")

    @torch.no_grad()
    def inference(self,
                  images: torch.Tensor,
                  num_inference_steps: int,
                  task_embedding: Optional[torch.Tensor] = None,
                  guidance_scale: float = 1.0,  # For CFG
                  # If different from training
                  action_horizon: Optional[int] = None,
                  initial_noise: Optional[torch.Tensor] = None
                  ) -> torch.Tensor:
        """
        Autoregressive inference loop.
        images: [B, T_img, C, H, W]
        num_inference_steps: Number of DDPM/DDIM steps
        task_embedding: Optional [B, D_task] or [B, T_task, D_task]
        guidance_scale: Strength of CFG. 1.0 means no guidance.
        action_horizon: Length of the action sequence to generate.
        initial_noise: Optional initial noise tensor [B, T_act, A]
        Returns: Denoised action sequence [B, T_act, A]
        """
        B = images.shape[0]
        device = images.device
        T_act = action_horizon if action_horizon is not None else self.scheduler.config.get(
            'train_timesteps', 1000)  # Placeholder, adjust as needed

        # 1. Prepare scheduler and initial noise
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        if initial_noise is None:
            noisy_actions = torch.randn(
                (B, T_act, self.action_dim), device=device)
        else:
            assert initial_noise.shape == (B, T_act, self.action_dim)
            noisy_actions = initial_noise

        # 2. Get condition tokens
        cond_tokens = self.get_condition_tokens(images, task_embedding)

        # Prepare unconditional tokens if using CFG
        uncond_tokens = None
        if guidance_scale > 1.0:
            # Simple approach: zero out context
            # Better: use dedicated null embeddings learned during training
            uncond_tokens = torch.zeros_like(cond_tokens)

        # 3. Denoising loop
        for t in self.scheduler.timesteps:
            # Expand timestep for batch
            timesteps_batch = torch.tensor(
                [t] * B, device=device, dtype=torch.long)

            # --- CFG Prediction ---
            if guidance_scale > 1.0 and uncond_tokens is not None:
                # Predict noise for both conditional and unconditional inputs
                # Combine inputs for a single forward pass
                combined_context = torch.cat(
                    [cond_tokens, uncond_tokens], dim=0)
                combined_noisy_actions = torch.cat(
                    [noisy_actions, noisy_actions], dim=0)
                combined_timesteps = torch.cat(
                    [timesteps_batch, timesteps_batch], dim=0)

                # Embed inputs for combined batch
                time_emb = self.time_embed(combined_timesteps).unsqueeze(1)
                action_emb = self.action_embed(combined_noisy_actions)
                # if hasattr(self, 'action_pos_embed'): action_emb += self.action_pos_embed[:, :T_act, :]
                decoder_input = torch.cat([time_emb, action_emb], dim=1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    decoder_input.size(1), device=device)

                # Run transformer
                transformer_output = self.transformer_decoder(
                    decoder_input, combined_context, tgt_mask=tgt_mask)
                action_output_tokens = transformer_output[:, 1:, :]

                # Run denoising head
                model_output_combined = self.denoising_head(
                    action_output_tokens)

                # Split predictions
                model_output_cond, model_output_uncond = torch.chunk(
                    model_output_combined, 2, dim=0)

                # Combine predictions using guidance scale
                model_output = model_output_uncond + guidance_scale * \
                    (model_output_cond - model_output_uncond)
            else:
                # Standard prediction without CFG
                time_emb = self.time_embed(timesteps_batch).unsqueeze(1)
                action_emb = self.action_embed(noisy_actions)
                # if hasattr(self, 'action_pos_embed'): action_emb += self.action_pos_embed[:, :T_act, :]
                decoder_input = torch.cat([time_emb, action_emb], dim=1)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    decoder_input.size(1), device=device)

                transformer_output = self.transformer_decoder(
                    decoder_input, cond_tokens, tgt_mask=tgt_mask)
                action_output_tokens = transformer_output[:, 1:, :]
                model_output = self.denoising_head(action_output_tokens)

            # 4. Scheduler step
            noisy_actions = self.scheduler.step(model_output, t, noisy_actions)

        # Return the final denoised actions
        return noisy_actions
