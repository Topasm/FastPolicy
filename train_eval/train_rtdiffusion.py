#!/usr/bin/env python

"""This script demonstrates how to train Diffusion Policy on the PushT environment.

Once you have trained a model with this script, you can try to evaluate it on
examples/2_evaluate_pretrained_policy.py
"""

import torch
from pathlib import Path
import safetensors.torch
from tqdm import tqdm
import numpy as np

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features

from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_clphycon import CLDiffPhyConModel


def main():
    # Create a directory to store the training checkpoint
    output_directory = Path("outputs/train/rtdiffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Number of training steps and logging frequency
    training_steps = 5000
    log_freq = 100
    save_freq = 1000

    # Dataset and feature setup
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Define input and output features
    output_features = {"action": features["action"]}
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    # Configuration for the diffusion model
    horizon = 16
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
        # Transformer parameters
        transformer_dim=512,
        transformer_num_layers=6,
        transformer_num_heads=8,
        # Vision parameters
        vision_backbone="resnet18",
        spatial_softmax_num_keypoints=32,
        # Disable state interpolation for direct action prediction
        interpolate_state=False,
    )

    # Create and initialize the model
    policy = CLDiffPhyConModel(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # Setup delta timestamps for dataset
    # Use only the necessary states for conditioning (2 steps)
    # and the full 16-frame horizon for asynchronous action prediction
    # Just the previous (-1) and current (0) states
    state_range = [-1, 0]
    image_indices = [-1, 0]  # Previous and current image frames
    action_range = list(range(16))  # Full 16-frame action sequence

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
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Create optimizer
    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    # Training loop
    print("Starting Asynchronous RT-Diffusion Training...")
    step = 0
    done = False

    while not done:
        for batch in tqdm(dataloader, desc=f"Training Step: {step}/{training_steps}"):
            # Move batch to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # Use asynchronous diffusion training
            loss = policy.forward_async(batch)

            # Backprop and update
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)

            optimizer.step()

            # Logging
            if step % log_freq == 0:
                print(
                    f"Step: {step}/{training_steps} | Async Loss: {loss.item():.4f}")

            # Checkpointing
            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(policy.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # Save final model
    final_path = output_directory / "model_final.pth"
    torch.save(policy.state_dict(), final_path)

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

    print(f"Training finished. Model saved to: {output_directory}")


if __name__ == "__main__":
    main()
