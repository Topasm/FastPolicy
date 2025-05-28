#!/usr/bin/env python
"""
Asynchronous Diffusion Training Implementation for RT-Diffusion

This module implements the specific asynchronous diffusion training process:
1. Sample a full 16-frame ground-truth sequence (x₀) from training dataset
2. Sample base time (t) from low range (0 to gap_timesteps)
3. Generate asynchronous times using asyn_t_seq function to create 16 time steps [t, t+gap, ..., t+15*gap]
4. Add noise using q_sample function with corresponding asynchronous time step for each frame
5. Predict noise using DiT with noisy 16-frame sequence and asynchronous time steps as embeddings
6. Calculate MSE loss between DiT's prediction and actual noise
7. Update weights via backpropagation
"""

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Optional
import numpy as np
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def asyn_t_seq(base_t: Tensor, gap: int, horizon: int) -> Tensor:
    """Generate asynchronous time sequence.

    Args:
        base_t: Base timestep (B,) tensor with values in range [0, gap_timesteps)
        gap: Gap between consecutive timesteps
        horizon: Number of frames in sequence (e.g., 16)

    Returns:
        Tensor of shape (B, horizon) with asynchronous timesteps [t, t+gap, ..., t+(horizon-1)*gap]
    """
    batch_size = base_t.shape[0]
    device = base_t.device

    # Create offset sequence: [0, gap, 2*gap, ..., (horizon-1)*gap]
    offsets = torch.arange(0, horizon * gap, gap,
                           device=device, dtype=base_t.dtype)
    offsets = offsets.unsqueeze(0).expand(batch_size, -1)  # (B, horizon)

    # Add base timestep to offsets
    async_timesteps = base_t.unsqueeze(1) + offsets  # (B, horizon)

    return async_timesteps


def q_sample(x_0: Tensor, timesteps: Tensor, noise_scheduler: DDPMScheduler, noise: Optional[Tensor] = None) -> Tensor:
    """Add noise to clean data according to diffusion forward process.

    Args:
        x_0: Clean data (B, T, D) - the ground truth sequence
        timesteps: Timesteps for each frame (B, T) - asynchronous timesteps
        noise_scheduler: DDPM scheduler for noise computation
        noise: Optional pre-generated noise (B, T, D)

    Returns:
        Noisy data (B, T, D) with different noise levels per frame
    """
    if noise is None:
        noise = torch.randn_like(x_0)

    B, T, D = x_0.shape
    device = x_0.device

    # Handle asynchronous timesteps - each frame has different noise level
    noisy_samples = torch.zeros_like(x_0)

    for t_idx in range(T):
        # Get timesteps for all samples at frame t_idx
        t_frame = timesteps[:, t_idx]  # (B,)

        # Add noise for this frame across all samples
        # Use the scheduler's add_noise method for each frame
        x_0_frame = x_0[:, t_idx, :]  # (B, D)
        noise_frame = noise[:, t_idx, :]  # (B, D)

        # Add noise using scheduler
        noisy_frame = noise_scheduler.add_noise(
            x_0_frame, noise_frame, t_frame)
        noisy_samples[:, t_idx, :] = noisy_frame

    return noisy_samples


class AsyncDiffusionTrainer:
    """Asynchronous Diffusion Training Handler for RT-Diffusion."""

    def __init__(self, gap_timesteps: int = 50, gap: int = 5, horizon: int = 16):
        """Initialize async diffusion trainer.

        Args:
            gap_timesteps: Maximum value for base timestep sampling (low range)
            gap: Gap between consecutive asynchronous timesteps  
            horizon: Number of frames in sequence
        """
        self.gap_timesteps = gap_timesteps
        self.gap = gap
        self.horizon = horizon

    def sample_async_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        """Sample base timesteps from low range and generate asynchronous sequence.

        Args:
            batch_size: Number of samples in batch
            device: Device for tensor creation

        Returns:
            Async timesteps (B, horizon) 
        """
        # Sample base time (t) from low range (0 to gap_timesteps)
        base_t = torch.randint(
            low=0,
            high=self.gap_timesteps,
            size=(batch_size,),
            device=device,
            dtype=torch.long
        )

        # Generate asynchronous time sequence [t, t+gap, ..., t+15*gap]
        async_timesteps = asyn_t_seq(base_t, self.gap, self.horizon)

        return async_timesteps

    def compute_async_loss(
        self,
        clean_sequence: Tensor,
        denoising_model,
        noise_scheduler: DDPMScheduler,
        global_cond: Optional[Tensor] = None
    ) -> Tensor:
        """Compute asynchronous diffusion loss.

        Args:
            clean_sequence: Ground truth 16-frame sequence (B, 16, D)
            denoising_model: DiT model for noise prediction
            noise_scheduler: DDPM scheduler
            global_cond: Optional global conditioning (B, cond_dim)

        Returns:
            MSE loss between predicted and actual noise
        """
        B, T, D = clean_sequence.shape
        device = clean_sequence.device

        # Step 1: Sample asynchronous timesteps
        async_timesteps = self.sample_async_timesteps(B, device)

        # Step 2: Sample noise
        noise = torch.randn_like(clean_sequence)

        # Step 3: Add noise using q_sample with asynchronous timesteps
        noisy_sequence = q_sample(
            clean_sequence, async_timesteps, noise_scheduler, noise)

        # Step 4: Predict noise using DiT with async mode
        pred_noise = denoising_model(
            noisy_input=noisy_sequence,
            timesteps=async_timesteps,  # (B, T) for async mode
            global_cond=global_cond,
            async_mode=True
        )

        # Step 5: Calculate MSE loss between prediction and actual noise
        loss = F.mse_loss(pred_noise, noise)

        return loss


def prepare_16_frame_sequence(batch: dict[str, Tensor], horizon: int = 16) -> Tensor:
    """Extract 16-frame ground-truth sequence from batch data.

    Args:
        batch: Batch containing action sequences
        horizon: Number of frames to extract

    Returns:
        16-frame sequence (B, 16, action_dim)
    """
    if "action" not in batch:
        raise KeyError("Batch must contain 'action' key")

    actions = batch["action"]  # (B, T, action_dim)

    if actions.shape[1] < horizon:
        raise ValueError(
            f"Action sequence too short. Need {horizon} frames, got {actions.shape[1]}")

    # Extract first 16 frames
    return actions[:, :horizon, :]


def create_async_training_batch(
    batch: dict[str, Tensor],
    trainer: AsyncDiffusionTrainer,
    denoising_model,
    noise_scheduler: DDPMScheduler,
    global_cond: Optional[Tensor] = None
) -> Tensor:
    """Create and process a batch for asynchronous diffusion training.

    Args:
        batch: Input batch from dataloader
        trainer: AsyncDiffusionTrainer instance  
        denoising_model: DiT model
        noise_scheduler: DDPM scheduler
        global_cond: Optional global conditioning

    Returns:
        Computed loss value
    """
    # Step 1: Sample a full 16-frame ground-truth sequence (x₀)
    clean_sequence = prepare_16_frame_sequence(batch, trainer.horizon)

    # Step 2-7: Compute async loss (handles all remaining steps)
    loss = trainer.compute_async_loss(
        clean_sequence=clean_sequence,
        denoising_model=denoising_model,
        noise_scheduler=noise_scheduler,
        global_cond=global_cond
    )

    return loss
