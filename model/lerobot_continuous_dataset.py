#!/usr/bin/env python3
# filepath: /home/ahrilab/Desktop/FastPolicy/model/lerobot_continuous_dataset_fixed.py
import torch
import numpy as np
from typing import Dict, List, Optional
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.normalize import Normalize
from torch.utils.data import Dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LeRobotContinuousDataset(Dataset):
    """
    Dataset for training a Transformer for continuous trajectory generation.
    It can use any start and end points within a trajectory and samples
    intermediate points for continuous path generation.
    
    Fixed version that handles short trajectories and avoids negative strides.
    """

    def __init__(
        self,
        lerobot_dataset: LeRobotDataset,
        normalizer: Normalize,
        state_dim: int,
        bidirectional: bool = True,
        direction_weight: float = 0.5,
        min_traj_len: int = 32,
        pad_token_pos: int = -100,
        max_position_value: int = 64,
        seq_len: int = 17
    ):
        """
        Initialize the Continuous LeRobot dataset.

        Args:
            lerobot_dataset: LeRobotDataset instance.
            normalizer: Normalization utility for inputs.
            state_dim: Dimension of state vectors.
            bidirectional: Whether to train on both forward and backward trajectories.
            direction_weight: Weight for forward vs backward direction samples.
            min_traj_len: Minimum required trajectory length.
            pad_token_pos: Position ID for padding tokens.
            max_position_value: Maximum position value.
            seq_len: Number of points to sample per trajectory.
        """
        super().__init__()
        self.lerobot_dataset = lerobot_dataset
        self.normalizer = normalizer
        self.state_dim = state_dim
        self.bidirectional = bidirectional
        self.direction_weight = direction_weight
        self.min_traj_len = min_traj_len
        self.pad_token_pos = pad_token_pos
        self.max_position_value = max_position_value
        self.seq_len = min(seq_len, min_traj_len)  # Ensure seq_len <= min_traj_len

        # Filter for trajectories long enough
        self.valid_indices = []
        for i in range(len(self.lerobot_dataset)):
            try:
                traj_len = self.lerobot_dataset.episode_lengths[i]
            except Exception:
                traj_len = self.lerobot_dataset[i]["observation.state"].shape[0]
                
            if traj_len >= self.min_traj_len:
                self.valid_indices.append(i)

        if not self.valid_indices:
            raise ValueError(
                f"No trajectories long enough ({self.min_traj_len} steps) found.")

        logger.info(
            f"Initialized ContinuousDataset with {len(self.valid_indices)} valid trajectories.")

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample with continuous positions.
        """
        actual_idx = self.valid_indices[idx]
        raw_data = self.lerobot_dataset[actual_idx]
        norm_data = self.normalizer(raw_data)
        traj = norm_data["observation.state"].cpu().numpy()  # Shape [T, D]

        # Ensure trajectory has at least min_traj_len steps
        if traj.shape[0] < self.min_traj_len:
            # This should not happen due to filtering, but as a safeguard
            raise ValueError(
                f"Trajectory {actual_idx} is too short: {traj.shape[0]}")

        # Limit to the first min_traj_len steps if longer
        traj = traj[:self.min_traj_len].copy()  # Use .copy() to ensure contiguous

        # Decide whether to generate forward or backward
        use_forward = True if not self.bidirectional else np.random.random() > self.direction_weight

        # For the case of 2 states in a trajectory, we need to reshape things appropriately
        if self.seq_len == 2:
            # Just use the 2 states we have
            states = traj[:2].copy()
            
            # Calculate positions (evenly spaced in [0, max_position_value])
            positions = np.array([0, self.max_position_value], dtype=np.int64)
            
            # For backward direction, reverse everything manually to avoid negative strides
            if not use_forward:
                states = np.flip(states, axis=0).copy()  # Use .copy() after flip
                positions = np.flip(positions, axis=0).copy()
                direction = 1
            else:
                direction = 0
                
            # Use the first state to predict the second state
            input_states = np.expand_dims(states[0], axis=0)  # Shape: [1, D]
            target_state = states[1]  # Shape: [D]
            input_positions = np.array([positions[0]], dtype=np.int64)  # Shape: [1]
            target_position = positions[1]  # Shape: scalar
            
        else:
            # Sample a subset of points for training - same as original implementation
            # We always include the first and last point
            traj_length = traj.shape[0]

            # We need to sample self.seq_len - 2 intermediate points
            # and then add the first and last points
            if self.seq_len > 2:
                # Sample intermediate indices
                available_indices = np.arange(1, traj_length - 1)
                if len(available_indices) > self.seq_len - 2:
                    sampled_indices = np.sort(np.random.choice(
                        available_indices, self.seq_len - 2, replace=False))
                else:
                    # If not enough intermediate points, use all and possibly repeat
                    sampled_indices = np.sort(np.random.choice(
                        available_indices, self.seq_len - 2, replace=True))

                # Combine with first and last indices
                indices = np.concatenate(([0], sampled_indices, [traj_length - 1]))
            else:
                # Just use first and last if seq_len <= 2
                indices = np.array([0, traj_length - 1])

            # Calculate positions (evenly spaced in [0, max_position_value])
            positions = np.linspace(
                0, self.max_position_value, len(indices), dtype=np.int64)

            # Extract the states
            states = traj[indices].copy()  # Use .copy() to ensure contiguous

            # For backward direction, reverse everything manually to avoid negative strides
            if not use_forward:
                states = np.flip(states, axis=0).copy()  # Use .copy() after flip
                positions = np.flip(positions, axis=0).copy()
                direction = 1
            else:
                direction = 0

            # For training, we'll use all points except the last as inputs
            # and predict the next point
            input_states = states[:-1]
            target_state = states[-1]  # Last state is the prediction target
            input_positions = positions[:-1]
            target_position = positions[-1]

        # Create attention mask (all 1s for our sampled tokens)
        attention_mask = np.ones(len(input_states), dtype=np.float32)

        return {
            'inputs': torch.tensor(input_states, dtype=torch.float32),
            'targets': torch.tensor(target_state, dtype=torch.float32),
            'positions': torch.tensor(input_positions, dtype=torch.long),
            'target_position': torch.tensor(target_position, dtype=torch.long),
            'directions': torch.tensor(direction, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.float32),
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Collates samples into a batch.
        """
        # Stack all the tensors from the batch
        inputs = torch.stack([item['inputs'] for item in batch])
        targets = torch.stack([item['targets'] for item in batch])
        positions = torch.stack([item['positions'] for item in batch])
        directions = torch.stack([item['directions'] for item in batch])
        attention_mask = torch.stack(
            [item['attention_mask'] for item in batch])

        return {
            'inputs': inputs,
            'targets': targets,
            'positions': positions,
            'directions': directions,
            'attention_mask': attention_mask,
        }
