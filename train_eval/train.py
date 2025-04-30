# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This scripts demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from model.diffusion.modeling_mymodel import MYDiffusionPolicy
from model.diffusion.configuration_mymodel import DiffusionConfig

from lerobot.configs.types import FeatureType


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/dit_plan_policy")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5000
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    features = dataset_to_policy_features(dataset_metadata.features)

    # Determine input and output features based on prediction mode
    predict_state_flag = True  # Set this based on your desired mode

    if predict_state_flag:
        # Diffusion model predicts future observation.state, policy outputs action
        target_state_key = "observation.state"  # Use observation.state as the target

        output_features = {
            # Target for diffusion
            target_state_key: features[target_state_key],
            "action": features["action"]  # Final output of the policy
        }
        # Input features will include observation.state (for past steps) and potentially images/env_state
        input_features = {key: ft for key,
                          ft in features.items() if key != "action"}
    else:
        # Diffusion model predicts action, policy outputs action
        output_features = {
            key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
        input_features = {key: ft for key,
                          ft in features.items() if key not in output_features}

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`. For this example,
    # we'll just use the defaults and so no arguments other than input/output features need to be passed.
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        predict_state=predict_state_flag,  # Use the flag here
    )

    # We can now instantiate our policy with this config and the dataset stats.
    policy = MYDiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Another policy-dataset interaction is with the delta_timestamps. Each policy expects a given number frames
    # which can differ for inputs, outputs and rewards (if there are some).
    # Use the config properties to define the required timestamps
    delta_timestamps = {
        # Input observations (past/present)
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices]
    }
    # If not predicting state, the action key might need different indices if they differ from target_delta_indices
    if not predict_state_flag:
        # Assuming action uses action_delta_indices if different from target_delta_indices
        # If action_delta_indices is the same as target_delta_indices, this line is redundant but harmless
        delta_timestamps["action"] = [
            i / dataset_metadata.fps for i in cfg.action_delta_indices]

    # We can then instantiate the dataset with these delta_timestamps configuration.
    dataset = LeRobotDataset(
        "lerobot/pusht", delta_timestamps=delta_timestamps)

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)


if __name__ == "__main__":
    main()
