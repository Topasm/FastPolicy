import logging
from datetime import datetime
import random
import glob
import os
from PIL import Image
from collections import Counter
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
from torch.utils.data import Dataset as TorchDataset
import torch.nn.functional as F

import pickle
from huggingface_hub import HfApi

import lerobot
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from typing import Tuple
from torchvision import transforms


from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from diffusers import AutoencoderDC


import numpy as np
import torch
from typing import Dict, Any
from torch.utils.data import Dataset as TorchDataset


class MinMaxNormalizer:
    def __init__(self, min_values, max_values, target_range=(0, 1)):
        """
        Args:
            min_values: List of min values for each dimension (7D).
            max_values: List of max values for each dimension (7D).
            target_range: Tuple specifying the target range for normalization (default: (0, 1)).
        """
        self.min_values = torch.tensor(min_values, dtype=torch.float32)
        self.max_values = torch.tensor(max_values, dtype=torch.float32)
        self.target_min, self.target_max = target_range

    def normalize(self, data):
        """
        Normalize the input data using Min-Max normalization.

        Args:
            data: Tensor of shape (..., 7), where 7 corresponds to the 7-DOF features.

        Returns:
            Normalized data in the specified target range.
        """

        norm_data = (data - self.min_values) / \
            (self.max_values - self.min_values)
        norm_data = norm_data * \
            (self.target_max - self.target_min) + self.target_min
        return norm_data


class CustomSequenceDataset(TorchDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.dataset = LeRobotDataset(
            repo_id=config.repo_id, episodes=list(range(config.episodes)))
        self.ds_meta = LeRobotDatasetMetadata(repo_id=config.repo_id)
        self.camera_key = self.ds_meta.camera_keys[0]

        from_idx = self.dataset.episode_data_index["from"].numpy()
        to_idx = self.dataset.episode_data_index["to"].numpy()
        self.episode_indices = np.stack([from_idx, to_idx], axis=1)
        self.lengths = (to_idx - from_idx).astype(np.int64)
        self.num_steps = np.sum(self.lengths)

        self.max_len = 10  # Set the maximum sequence length
        self.mask_prob = 0.3  # Probability of masking each feature

        self.position_normalizer = MinMaxNormalizer(
            min_values=[-1.2867788076400757, -1.371172547340393, -0.3636113405227661,
                        -2.998310089111328, -0.5526790618896484, 0.0, -2.7811992168426514],
            max_values=[0.6249395608901978, 0.8843822479248047, 1.7471705675125122,
                        0.0, 2.4926300048828125, 3.4060142040252686, 2.872147798538208],
            target_range=(0, 1)  # Change to (-1, 1) if needed
        )

        self.velocity_normalizer = MinMaxNormalizer(
            min_values=[-0.5422850847244263, -0.8411960601806641, -0.21819494664669037,
                        -0.6476227045059204, -0.6400466561317444, -0.964364230632782, -0.3738694489002228],
            max_values=[0.22520515322685242, 0.5186036825180054, 0.8507077097892761,
                        0.5890041589736938, 0.3648236095905304, 0.5417290329933167, 0.8075176477432251],
            target_range=(0, 1)
        )

        self.torque_normalizer = MinMaxNormalizer(
            min_values=[-5.098331928253174, -57.641258239746094, -28.15553092956543,
                        0.0, -3.1010899543762207, -0.6073437333106995, -1.5022757053375244],
            max_values=[3.6654012203216553, 38.446388244628906, 8.76799201965332,
                        26.153221130371094, 1.6871180534362793, 3.5940113067626953, 0.8021065592765808],
            target_range=(0, 1)
        )

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        # Identify episode

        seq_len = random.randint(2, self.max_len)
        episode_id = np.searchsorted(
            np.cumsum(self.lengths), idx, side='right')
        from_idx, to_idx = self.episode_indices[episode_id]

        offset_in_episode = idx - \
            (0 if episode_id == 0 else np.cumsum(
                self.lengths[:episode_id])[-1])

        # Define sequence boundaries
        start = offset_in_episode - (seq_len - 1)
        end = min(start + seq_len, self.lengths[episode_id])

        padding_needed = max(0, -start)
        start = max(0, start)  # Ensure start is within valid range

        seq_indices = range(from_idx + start, from_idx + end)

        if padding_needed > 0:
            padded_indices = [None] * padding_needed + list(seq_indices)

        else:
            padded_indices = list(seq_indices)

        state_list, vel_list, torque_list = [], [], []
        obs = None
        for i in padded_indices:
            if i is None:
                state_list.append(torch.zeros(7))
                vel_list.append(torch.zeros(7))
                torque_list.append(torch.zeros(7))
            else:
                item = self.dataset[i]
                obs = item[self.camera_key]
                obs = F.interpolate(obs.unsqueeze(0), size=(224, 224),
                                    mode='bilinear', align_corners=False).squeeze(0)

                state_norm = self.position_normalizer.normalize(
                    item['observation.state'].float())
                vel_norm = self.velocity_normalizer.normalize(
                    item['observation.velocity'].float())
                torque_norm = self.torque_normalizer.normalize(
                    item['observation.torque'].float())

                state_list.append(state_norm)
                vel_list.append(vel_norm)
                torque_list.append(torque_norm)

        state = torch.stack(state_list)
        vel = torch.stack(vel_list)
        torque = torch.stack(torque_list)

        joint_seq = torch.cat([state, vel, torque], dim=1)

        mask = torch.rand(joint_seq.size()) < self.mask_prob
        joint_seq[mask] = 0.0

        obs2 = obs  # Last image of the sequence

        return obs2, joint_seq


class CustomSequenceDatasetEval(TorchDataset):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()

        # -------------------------------------------------
        #  FIX: Convert episodes to a normal Python list
        # -------------------------------------------------
        episodes = config['episodes']
        episodes = list(episodes)               # Convert ListConfig -> list
        episodes = [int(x) for x in episodes]   # Ensure each element is an int

        # Now pass the plain list[int] to LeRobotDataset
        self.dataset = LeRobotDataset(
            repo_id=config['repo_id'],
            episodes=episodes
        )
        self.ds_meta = LeRobotDatasetMetadata(repo_id=config['repo_id'])
        self.camera_key = self.ds_meta.camera_keys[0]

        from_idx = self.dataset.episode_data_index["from"].numpy()
        to_idx = self.dataset.episode_data_index["to"].numpy()
        self.episode_indices = np.stack([from_idx, to_idx], axis=1)
        self.lengths = (to_idx - from_idx).astype(np.int64)
        self.num_steps = np.sum(self.lengths)

        self.max_len = 10
        self.mask_prob = 0.0

        # --- Normalizers (same as training) ---
        self.position_normalizer = MinMaxNormalizer(
            min_values=[-1.2867788076400757, -1.371172547340393, -0.3636113405227661,
                        -2.998310089111328, -0.5526790618896484, 0.0, -2.7811992168426514],
            max_values=[0.6249395608901978, 0.8843822479248047, 1.7471705675125122,
                        0.0, 2.4926300048828125, 3.4060142040252686, 2.872147798538208],
            target_range=(0, 1)
        )

        self.velocity_normalizer = MinMaxNormalizer(
            min_values=[-0.5422850847244263, -0.8411960601806641, -0.21819494664669037,
                        -0.6476227045059204, -0.6400466561317444, -0.964364230632782, -0.3738694489002228],
            max_values=[0.22520515322685242, 0.5186036825180054, 0.8507077097892761,
                        0.5890041589736938, 0.3648236095905304, 0.5417290329933167, 0.8075176477432251],
            target_range=(0, 1)
        )

        self.torque_normalizer = MinMaxNormalizer(
            min_values=[-5.098331928253174, -57.641258239746094, -28.15553092956543,
                        0.0, -3.1010899543762207, -0.6073437333106995, -1.5022757053375244],
            max_values=[3.6654012203216553, 38.446388244628906, 8.76799201965332,
                        26.153221130371094, 1.6871180534362793, 3.5940113067626953, 0.8021065592765808],
            target_range=(0, 1)
        )

    def __len__(self):
        return self.num_steps

    def __getitem__(self, idx):
        episode_id = np.searchsorted(
            np.cumsum(self.lengths), idx, side='right')
        from_idx, to_idx = self.episode_indices[episode_id]

        if episode_id == 0:
            offset_in_episode = idx
        else:
            offset_in_episode = idx - np.cumsum(self.lengths[:episode_id])[-1]
        # seq_len = random.randint(2, self.max_len)
        seq_len = 10
        start = offset_in_episode - (seq_len - 1)
        end = min(start + seq_len, self.lengths[episode_id])

        padding_needed = max(0, -start)
        start = max(0, start)  # Ensure start is within valid range

        seq_indices = range(from_idx + start, from_idx + end)

        if padding_needed > 0:
            padded_indices = [None] * padding_needed + list(seq_indices)

        else:
            padded_indices = list(seq_indices)

        state_list, vel_list, torque_list = [], [], []
        obs = None
        for i in padded_indices:
            if i is None:
                state_list.append(torch.zeros(7))
                vel_list.append(torch.zeros(7))
                torque_list.append(torch.zeros(7))
            else:
                item = self.dataset[i]
                obs = item[self.camera_key]
                obs = F.interpolate(obs.unsqueeze(0), size=(224, 224),
                                    mode='bilinear', align_corners=False).squeeze(0)

                state_norm = self.position_normalizer.normalize(
                    item['observation.state'].float())
                vel_norm = self.velocity_normalizer.normalize(
                    item['observation.velocity'].float())
                torque_norm = self.torque_normalizer.normalize(
                    item['observation.torque'].float())

                state_list.append(state_norm)
                vel_list.append(vel_norm)
                torque_list.append(torque_norm)

        states = torch.stack(state_list, dim=0)
        vels = torch.stack(vel_list, dim=0)
        torques = torch.stack(torque_list, dim=0)
        joint_seq = torch.cat([states, vels, torques], dim=1)

        # Optional random mask
        mask = torch.rand(joint_seq.size()) < self.mask_prob
        joint_seq[mask] = 0.0

        return obs, joint_seq, episode_id
