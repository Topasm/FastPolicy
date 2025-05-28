#!/usr/bin/env python

"""
Asynchronous RT-Diffusion Training Script

This script implements the asynchronous diffusion training process for RT-Diffusion model:
1. Sample a full 16-frame ground-truth sequence (x₀) from training dataset
2. Sample base time (t) from low range (0 to gap_timesteps)
3. Generate asynchronous times using asyn_t_seq function to create 16 time steps [t, t+gap, ..., t+15*gap]
4. Add noise using q_sample function with corresponding asynchronous time step for each frame
5. Predict noise using DiT with noisy 16-frame sequence and asynchronous time steps as embeddings
6. Calculate MSE loss between DiT's prediction and actual noise
7. Update weights via backpropagation
"""

import torch
from pathlib import Path
import safetensors.torch
from tqdm import tqdm
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize

from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.async_modules import DenoisingTransformer
from model.diffusion.async_training import AsyncDiffusionTrainer
from model.diffusion.modeling_clphycon import _make_noise_scheduler


class AsyncRTDiffusionModel(torch.nn.Module):
    """Asynchronous RT-Diffusion Model with DiT architecture."""

    def __init__(self, config: DiffusionConfig, dataset_stats: dict = None):
        super().__init__()
        self.config = config

        # Initialize normalization
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Calculate global conditioning dimension
        global_cond_dim = config.robot_state_feature.shape[0] * \
            config.n_obs_steps
        if config.image_features:
            # Add vision feature dimension (assuming ResNet18 features)
            global_cond_dim += 512 * \
                len(config.image_features) * config.n_obs_steps

        # Initialize the asynchronous denoising transformer (DiT)
        self.denoising_transformer = DenoisingTransformer(
            config=config,
            global_cond_dim=global_cond_dim,
            output_dim=config.action_feature.shape[0]
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

        # Initialize async trainer
        self.async_trainer = AsyncDiffusionTrainer(
            gap_timesteps=50,  # Low range for base timestep sampling
            gap=5,             # Gap between consecutive timesteps
            horizon=16         # 16-frame horizon
        )

    def _prepare_global_conditioning(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Prepare global conditioning vector from observations."""
        device = next(iter(batch.values())).device

        # Start with state conditioning
        obs_state = batch["observation.state"]  # (B, n_obs_steps, state_dim)
        B, n_obs, state_dim = obs_state.shape

        # Flatten state observations
        state_cond = obs_state.reshape(B, -1)  # (B, n_obs_steps * state_dim)

        # Add image conditioning if available
        if "observation.images" in batch:
            # Simple feature extraction (in practice, use a proper vision encoder)
            # (B, n_obs_steps, num_cameras, C, H, W)
            images = batch["observation.images"]
            B, n_obs, num_cameras, C, H, W = images.shape

            # Flatten and create dummy features (replace with actual vision encoder)
            image_features = torch.randn(
                B, 512 * num_cameras * n_obs, device=device)

            # Concatenate state and image features
            global_cond = torch.cat([state_cond, image_features], dim=1)
        else:
            global_cond = state_cond

        return global_cond

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass for asynchronous diffusion training."""
        # Normalize inputs and targets
        batch = self.normalize_inputs(batch)
        if self.config.image_features:
            batch = dict(batch)
            batch["observation.images"] = torch.stack(
                [batch[key] for key in self.config.image_features], dim=-4
            )
        batch = self.normalize_targets(batch)

        # Prepare global conditioning
        global_cond = self._prepare_global_conditioning(batch)

        # Extract 16-frame action sequence
        actions = batch["action"]  # (B, horizon, action_dim)
        if actions.shape[1] < 16:
            raise ValueError(
                f"Need at least 16 action frames, got {actions.shape[1]}")

        # Take first 16 frames as ground truth sequence
        clean_sequence = actions[:, :16, :]  # (B, 16, action_dim)

        # Compute asynchronous diffusion loss
        loss = self.async_trainer.compute_async_loss(
            clean_sequence=clean_sequence,
            denoising_model=self.denoising_transformer,
            noise_scheduler=self.noise_scheduler,
            global_cond=global_cond
        )

        return loss


def main():
    # Create output directory
    output_directory = Path("outputs/train/async_rtdiffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Training parameters
    training_steps = 10000
    log_freq = 100
    save_freq = 1000
    batch_size = 32

    # Dataset setup
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Define input and output features
    output_features = {"action": features["action"]}
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    # Configuration for asynchronous RT-Diffusion
    horizon = 16  # Full 16-frame horizon
    n_obs_steps = 2

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        n_action_steps=horizon,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        # Transformer parameters for DiT
        transformer_dim=512,
        transformer_num_layers=8,
        transformer_num_heads=8,
        # Vision parameters
        vision_backbone="resnet18",
        spatial_softmax_num_keypoints=32,
    )

    # Create async RT-Diffusion model
    model = AsyncRTDiffusionModel(cfg, dataset_stats=dataset_metadata.stats)
    model.train()
    model.to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Setup dataset with 16-frame action sequences
    state_range = [-1, 0]  # Observation history
    image_indices = [-1, 0]  # Image history
    action_range = list(range(16))  # 16 action frames

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
        "observation.state": [i / dataset_metadata.fps for i in state_range],
        "action": [i / dataset_metadata.fps for i in action_range],
    }

    # Create dataset and dataloader
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1e-4, weight_decay=1e-6)

    # Training loop
    print("Starting Asynchronous RT-Diffusion Training...")
    step = 0
    done = False

    while not done:
        for batch in tqdm(dataloader, desc=f"Training Step: {step}/{training_steps}"):
            # Move batch to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # Forward pass with asynchronous diffusion loss
            loss = model(batch)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            # Logging
            if step % log_freq == 0:
                print(
                    f"Step: {step}/{training_steps} | Async Loss: {loss.item():.4f}")

            # Checkpointing
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"async_model_step_{step}.pth"
                torch.save(model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final model
    final_path = output_directory / "async_model_final.pth"
    torch.save(model.state_dict(), final_path)

    # Save config and stats
    cfg.save_pretrained(output_directory)

    # Save dataset statistics
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")

    print(
        f"Asynchronous RT-Diffusion training complete! Model saved to: {output_directory}")


if __name__ == "__main__":
    main()
