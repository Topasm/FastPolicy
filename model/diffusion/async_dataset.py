#!/usr/bin/env python

from typing import Dict, List, Callable, Optional, Union, Any, Tuple
import torch
from torch.utils.data import Dataset
from torch import Tensor
import numpy as np
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


class AsynchronousTrajectoryDataset(Dataset):
    """Dataset wrapper that applies asynchronous noise levels to trajectories.
    This is a key component for the CL-DiffPhyCon system, allowing non-uniform noise 
    levels across tokens in a sequence to match the physical system's constraints.
    """

    def __init__(
        self,
        base_dataset: LeRobotDataset,
        noise_scheduler,
        state_key: str = "observation.state",
        action_key: str = "action",
        image_key: str = "observation.image",
        n_obs_steps: int = 2,
        horizon: int = 8,
        time_decay_factor: float = 0.9,
        min_noise_level: float = 0.05,
        max_noise_schedule: Optional[List[float]] = None,
    ):
        """Initialize the asynchronous trajectory dataset.

        Args:
            base_dataset: Original LeRobotDataset to wrap
            noise_scheduler: DDPM/DDIM scheduler to use for noise sampling
            state_key: Key for state observations in the dataset
            action_key: Key for actions in the dataset
            image_key: Key for image observations
            n_obs_steps: Number of observed steps used for context
            horizon: Number of future steps to predict
            time_decay_factor: Controls how quickly noise levels decay across the sequence
                               (higher means more difference between steps)
            min_noise_level: Minimum noise level for the nearest state
            max_noise_schedule: Optional custom noise schedule for each step in the horizon
        """
        self.base_dataset = base_dataset
        self.noise_scheduler = noise_scheduler
        self.state_key = state_key
        self.action_key = action_key
        self.image_key = image_key
        self.n_obs_steps = n_obs_steps
        self.horizon = horizon
        self.num_train_timesteps = noise_scheduler.config.num_train_timesteps

        # Create exponential decay schedule for noise levels if not provided
        if max_noise_schedule is None:
            # Generate exponentially decaying noise levels
            # First state has lowest noise (min_noise_level)
            # Last state has highest noise (close to 1.0)
            noise_levels = []
            for i in range(horizon):
                # Exponential decay from almost 1.0 to min_noise_level
                level = 1.0 - (1.0 - min_noise_level) * \
                    (time_decay_factor ** (horizon - i - 1))
                noise_levels.append(level)
            self.max_noise_schedule = torch.tensor(noise_levels)
        else:
            self.max_noise_schedule = torch.tensor(max_noise_schedule)

        assert len(
            self.max_noise_schedule) == horizon, "Noise schedule length must match horizon"

    def __len__(self):
        return len(self.base_dataset)

    def _get_async_timesteps(self, batch_size: int) -> Tensor:
        """Generate asynchronous timesteps for a batch based on the noise schedule.

        Returns:
            Tensor of shape (B, T) with different timestep for each position in sequence
        """
        # Generate a base random value for each sample in batch
        base_random = torch.rand(
            batch_size, 1, device=self.max_noise_schedule.device)

        # Scale the random value by the noise schedule for each position
        # This creates varying noise levels that follow the time_decay pattern
        noise_scale = self.max_noise_schedule.unsqueeze(
            0).expand(batch_size, -1)

        # Convert to actual timesteps (integers) for the scheduler
        # Scale from [0, 1) random values to integer timesteps
        async_noise_times = (base_random * noise_scale *
                             (self.num_train_timesteps - 1)).long()

        return async_noise_times

    def __getitem__(self, idx):
        """Get a data sample and apply asynchronous noise levels.

        Note: This doesn't actually add the noise - that happens in the training loop.
        This just prepares the timesteps for each position in the sequence.
        """
        # Get original sample from base dataset
        sample = self.base_dataset[idx]

        # Create asynchronous timesteps for this sample (B=1)
        async_timesteps = self._get_async_timesteps(batch_size=1).squeeze(0)

        # Add it to the sample dict
        sample["async_timesteps"] = async_timesteps

        return sample

    def collate_fn(self, batch):
        """Custom collate function that handles the asynchronous timesteps."""
        # Use default collate for everything except async_timesteps
        result = {}
        for key in batch[0].keys():
            if key != "async_timesteps":
                result[key] = torch.stack([b[key] for b in batch]) if torch.is_tensor(
                    batch[0][key]) else [b[key] for b in batch]

        # Stack async_timesteps
        result["async_timesteps"] = torch.stack(
            [b["async_timesteps"] for b in batch])

        return result
