#!/usr/bin/env python3
"""
Dataset wrapper for the Bidirectional Autoregressive Transformer.

This dataset prepares training data in the format required by the bidirectional model:
- Initial image and state
- Forward trajectory states
- Goal image
- Backward trajectory states
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import Dataset


class BidirectionalTrajectoryDataset(Dataset):
    """
    Dataset wrapper for bidirectional autoregressive trajectory learning.

    This dataset creates training samples with:
    1. Initial image i_0 and initial state st_0
    2. Forward trajectory: st_0 → st_1 → ... → st_15 
    3. Goal image i_n (from the final timestep)
    4. Backward trajectory: st_n → st_n-1 → ... → st_n-15
    """

    def __init__(
        self,
        lerobot_dataset,
        normalizer=None,
        forward_steps: int = 16,
        backward_steps: int = 16,
        min_episode_length: int = 50,
        image_key: str = "observation.image",
        state_key: str = "observation.state"
    ):
        """
        Initialize the bidirectional dataset.

        Args:
            lerobot_dataset: Base LeRobot dataset
            normalizer: Optional normalizer for states
            forward_steps: Number of forward trajectory steps
            backward_steps: Number of backward trajectory steps  
            min_episode_length: Minimum episode length to consider
            image_key: Key for image observations
            state_key: Key for state observations
        """
        self.lerobot_dataset = lerobot_dataset
        self.normalizer = normalizer
        self.forward_steps = forward_steps
        self.backward_steps = backward_steps
        self.min_episode_length = min_episode_length
        self.image_key = image_key
        self.state_key = state_key

        # Create valid trajectory samples
        self.samples = self._create_samples()

    def _create_samples(self):
        """Create valid training samples from the dataset."""
        samples = []

        # Get episode information
        if hasattr(self.lerobot_dataset, 'episode_data_index'):
            # Use LeRobot's episode_data_index which has format:
            # {'from': tensor([start_indices]), 'to': tensor([end_indices])}
            episode_data_index = self.lerobot_dataset.episode_data_index

            # Process each episode
            for episode_idx in range(len(episode_data_index['from'])):
                from_idx = episode_data_index['from'][episode_idx].item()
                to_idx = episode_data_index['to'][episode_idx].item()
                episode_length = to_idx - from_idx

                # Skip episodes that are too short
                if episode_length < self.min_episode_length:
                    continue

                # For each episode, we can create multiple trajectory samples
                # by selecting different start and goal positions
                max_start_idx = episode_length - self.forward_steps - 1

                # Overlap samples
                for start_offset in range(0, max_start_idx, self.forward_steps // 2):
                    sample = {
                        'episode_idx': episode_idx,
                        'start_idx': from_idx + start_offset,
                        'goal_idx': from_idx + start_offset + self.forward_steps - 1
                    }
                    samples.append(sample)
        else:
            # Fallback: group by episode_index if available
            episode_data_index = {}
            for idx in range(len(self.lerobot_dataset)):
                try:
                    item = self.lerobot_dataset[idx]
                    episode_idx = item.get('episode_index', 0)
                    if episode_idx not in episode_data_index:
                        episode_data_index[episode_idx] = {
                            'from': idx, 'to': idx}
                    else:
                        episode_data_index[episode_idx]['to'] = idx
                except Exception:
                    continue

            # Process each episode
            for episode_idx, episode_info in episode_data_index.items():
                episode_length = episode_info['to'] - episode_info['from'] + 1

                # Skip episodes that are too short
                if episode_length < self.min_episode_length:
                    continue

                # For each episode, we can create multiple trajectory samples
                # by selecting different start and goal positions
                max_start_idx = episode_length - self.forward_steps - 1

                # Overlap samples
                for start_offset in range(0, max_start_idx, self.forward_steps // 2):
                    sample = {
                        'episode_idx': episode_idx,
                        'start_idx': episode_info['from'] + start_offset,
                        'goal_idx': episode_info['from'] + start_offset + self.forward_steps - 1
                    }
                    samples.append(sample)

        print(f"Created {len(samples)} bidirectional trajectory samples")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            Dictionary containing:
            - initial_images: [C, H, W] initial image
            - initial_states: [state_dim] initial state  
            - forward_states: [forward_steps, state_dim] forward trajectory
            - goal_images: [C, H, W] goal image
            - backward_states: [backward_steps, state_dim] backward trajectory
        """
        sample_info = self.samples[idx]

        # Get initial data
        initial_data = self.lerobot_dataset[sample_info['start_idx']]
        goal_data = self.lerobot_dataset[sample_info['goal_idx']]

        # Extract initial image and state
        initial_image = initial_data[self.image_key]  # Should be [C, H, W]
        initial_state = initial_data[self.state_key]   # Should be [state_dim]

        # Extract goal image
        goal_image = goal_data[self.image_key]

        # Collect forward trajectory states
        forward_states = []
        for i in range(self.forward_steps):
            step_idx = sample_info['start_idx'] + i
            step_data = self.lerobot_dataset[step_idx]
            state = step_data[self.state_key]
            forward_states.append(state)

        # Create backward trajectory (reverse of forward)
        backward_states = []
        for i in range(self.backward_steps):
            # Start from goal and go backwards
            step_idx = sample_info['goal_idx'] - i
            if step_idx >= sample_info['start_idx']:
                step_data = self.lerobot_dataset[step_idx]
                state = step_data[self.state_key]
                backward_states.append(state)
            else:
                # If we run out of data, repeat the first state
                backward_states.append(forward_states[0])

        # Convert to tensors
        result = {
            'initial_images': torch.as_tensor(initial_image, dtype=torch.float32),
            'initial_states': torch.as_tensor(initial_state, dtype=torch.float32),
            'forward_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in forward_states]),
            'goal_images': torch.as_tensor(goal_image, dtype=torch.float32),
            'backward_states': torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in backward_states])
        }

        return result

    @staticmethod
    def collate_fn(batch):
        """Custom collate function for batching."""
        # Stack all tensors
        result = {}
        for key in batch[0].keys():
            result[key] = torch.stack([item[key] for item in batch])
        return result
