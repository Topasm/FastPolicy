#!/usr/bin/env python3
"""
Dataset wrapper for the Bidirectional Autoregressive Transformer.

This dataset prepares training data in the format required by the bidirectional model:
- Initial image and state (start of a sampled forward segment)
- Forward trajectory states (from the start of the sampled forward segment)
- Goal image (from the TRUE END of the episode)
- Backward trajectory states (starting from the TRUE END of the episode)
"""

import torch
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import Dataset


class BidirectionalTrajectoryDataset(Dataset):
    """
    Dataset wrapper for bidirectional autoregressive trajectory learning.
    """

    def __init__(
        self,
        lerobot_dataset,
        normalizer=None,  # Kept for interface, but normalization handled externally
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
            normalizer: Optional normalizer (not used internally by this class anymore)
            forward_steps: Number of forward trajectory steps for the forward segment
            backward_steps: Number of backward trajectory steps from the episode's true end
            min_episode_length: Minimum episode length to consider
            image_key: Key for image observations
            state_key: Key for state observations
        """
        self.lerobot_dataset = lerobot_dataset
        self.normalizer = normalizer  # Not used internally for transformation
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
        episode_data_index = getattr(
            self.lerobot_dataset, 'episode_data_index', None)

        if episode_data_index and 'from' in episode_data_index and 'to' in episode_data_index:
            # Use LeRobot's episode_data_index
            num_episodes = len(episode_data_index['from'])
            for episode_idx in range(num_episodes):
                from_idx = episode_data_index['from'][episode_idx].item()
                # LeRobot's 'to' index is exclusive (one past the last valid index)
                to_idx_exclusive = episode_data_index['to'][episode_idx].item()
                # Convert to inclusive index for actual episode end
                to_idx = to_idx_exclusive - 1
                episode_length = to_idx - from_idx + 1

                if episode_length < self.min_episode_length:
                    continue

                # Max start index for the forward segment ensures the forward segment fits within the episode
                # A forward segment of length `forward_steps` requires indices from `start` to `start + forward_steps - 1`.
                # So, `start + forward_steps - 1` must be <= `to_idx`.
                # `start` must be <= `to_idx - forward_steps + 1`.
                # `start_offset` is relative to `from_idx`. So, `from_idx + start_offset <= to_idx - forward_steps + 1`.
                # `start_offset <= to_idx - from_idx - forward_steps + 1` which is `episode_length - forward_steps`.
                max_start_offset = episode_length - self.forward_steps
                if max_start_offset < 0:  # Should not happen if min_episode_length >= forward_steps
                    continue

                # Overlap samples for the forward part
                # The step for start_offset can be adjusted, e.g., self.forward_steps // 2 for 50% overlap
                for start_offset in range(0, max_start_offset + 1, self.forward_steps // 2 if self.forward_steps > 1 else 1):
                    current_start_idx = from_idx + start_offset
                    # forward_segment_end_idx is not strictly needed in sample_info anymore
                    # as forward_states are just taken for self.forward_steps from current_start_idx

                    sample = {
                        'episode_idx': episode_idx,  # Store for fetching episode bounds later
                        # Start of the forward state sequence
                        'start_idx_forward_segment': current_start_idx,
                        # Absolute end of the episode for goal_image and backward_states
                        'episode_true_end_idx': to_idx
                    }
                    samples.append(sample)
        else:
            # Fallback: This part might be less accurate if episode_index is not contiguous or well-defined
            print("Warning: `episode_data_index` not found or incomplete in `lerobot_dataset`. Falling back to `episode_index` grouping if available, which might be less robust.")
            # This fallback needs to be carefully reviewed or improved if used.
            # For simplicity, assuming the primary path above is used.
            # If you hit this warning, the logic below for grouping by 'episode_index' might need adjustments
            # to correctly identify `from_idx` and `to_idx` for each true episode.
            episode_map = {}
            for i in range(len(self.lerobot_dataset)):
                try:
                    item = self.lerobot_dataset[i]  # This can be slow
                    ep_idx = item.get('episode_index', 0)
                    if ep_idx not in episode_map:
                        episode_map[ep_idx] = {'indices': []}
                    episode_map[ep_idx]['indices'].append(i)
                except Exception:
                    continue  # Skip problematic items

            for ep_idx, data in episode_map.items():
                if not data['indices']:
                    continue
                from_idx = min(data['indices'])
                to_idx = max(data['indices'])
                episode_length = to_idx - from_idx + 1

                if episode_length < self.min_episode_length:
                    continue

                max_start_offset = episode_length - self.forward_steps
                if max_start_offset < 0:
                    continue

                for start_offset in range(0, max_start_offset + 1, self.forward_steps // 2 if self.forward_steps > 1 else 1):
                    current_start_idx = from_idx + start_offset
                    sample = {
                        'episode_idx': ep_idx,  # Using the grouped episode index
                        'start_idx_forward_segment': current_start_idx,
                        'episode_true_end_idx': to_idx
                    }
                    samples.append(sample)

        print(f"Created {len(samples)} bidirectional trajectory samples.")
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        sample_info = self.samples[idx]

        # --- Initial image and state (start of the forward segment) ---
        initial_data_idx = sample_info['start_idx_forward_segment']
        initial_data = self.lerobot_dataset[initial_data_idx]
        initial_image = initial_data[self.image_key]
        initial_state = initial_data[self.state_key]

        # --- Forward trajectory states (from start_idx_forward_segment for forward_steps) ---
        forward_states = []
        for i in range(self.forward_steps):
            step_idx = sample_info['start_idx_forward_segment'] + i
            # Bounds check to ensure we don't exceed episode or dataset bounds
            if step_idx > sample_info['episode_true_end_idx'] or step_idx >= len(self.lerobot_dataset):
                # This should not happen if _create_samples is correct, but safety check
                print(
                    f"Warning: Forward step index {step_idx} is out of bounds. Using last valid state.")
                state = forward_states[-1] if forward_states else initial_state
            else:
                step_data = self.lerobot_dataset[step_idx]
                state = step_data[self.state_key]
            forward_states.append(state)

        # --- Goal image (from THE TRUE END of the episode) ---
        episode_end_data_idx = sample_info['episode_true_end_idx']

        # Add bounds check to prevent index errors
        if episode_end_data_idx >= len(self.lerobot_dataset):
            print(
                f"Warning: Episode end index {episode_end_data_idx} is out of bounds for dataset of size {len(self.lerobot_dataset)}")
            episode_end_data_idx = len(self.lerobot_dataset) - 1

        episode_end_data = self.lerobot_dataset[episode_end_data_idx]
        goal_image = episode_end_data[self.image_key]
        # State at the true end
        true_episode_end_state = episode_end_data[self.state_key]

        # --- Backward trajectory states (starting from THE TRUE END of the episode) ---
        backward_states = []

        # Determine the absolute start index of the current episode for boundary checking
        episode_abs_start_idx = 0  # Default if not found
        episode_data_index = getattr(
            self.lerobot_dataset, 'episode_data_index', None)
        current_episode_id = sample_info['episode_idx']

        if episode_data_index and 'from' in episode_data_index:
            # Find the correct from_idx based on episode_id
            # This assumes episode_idx in sample_info corresponds to the iteration index of episodes
            # or is a direct key if the fallback path was used.
            # If it's an iteration index:
            if isinstance(current_episode_id, int) and current_episode_id < len(episode_data_index['from']):
                episode_abs_start_idx = episode_data_index['from'][current_episode_id].item(
                )
            # If it was a key from fallback, we might need a different way to get from_idx or store it in sample_info
            # For now, assuming primary path where episode_idx is 0 to num_episodes-1

        for i in range(self.backward_steps):
            current_bwd_idx = episode_end_data_idx - i
            # Check against episode start and dataset start
            if current_bwd_idx >= episode_abs_start_idx and current_bwd_idx >= 0:
                step_data = self.lerobot_dataset[current_bwd_idx]
                state = step_data[self.state_key]
                backward_states.append(state)
            else:
                # Pad with the earliest valid state collected in the backward sequence,
                # or if none collected yet (i.e., backward_steps=0 or first step is out of bounds),
                # pad with the true_episode_end_state.
                padding_state = backward_states[-1] if backward_states else true_episode_end_state
                backward_states.append(padding_state)

        # Ensure backward_states has length self.backward_steps (it should by the logic above)

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
        result = {}
        # Check if all items in the batch have the same keys
        if not batch:
            return result
        first_item_keys = batch[0].keys()

        for key in first_item_keys:
            # Collect all items for the current key
            key_items = [item[key] for item in batch if key in item]

            if not key_items:  # Should not happen if batch items are consistent
                continue

            if isinstance(key_items[0], torch.Tensor):
                result[key] = torch.stack(key_items)
            else:
                # Handle cases where items might not be tensors (e.g., metadata, though not typical here)
                # If they are lists of tensors (e.g. from a problematic padding), this might need adjustment
                try:
                    # Attempt to stack if they are lists of tensors that can be stacked
                    # This is a basic attempt; complex structures might need more specific handling
                    if all(isinstance(item, torch.Tensor) for item in key_items):
                        result[key] = torch.stack(key_items)
                    else:
                        # Store as list if not uniformly tensors
                        result[key] = key_items
                except Exception:
                    result[key] = key_items  # Fallback to list

        return result
