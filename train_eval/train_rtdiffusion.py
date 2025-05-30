#!/usr/bin/env python

"""This script demonstrates how to train Diffusion Policy (CLDiffPhyConModel)
for STATE PREDICTION on the PushT environment.
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
    # Changed output dir name
    output_directory = Path("outputs/train/rtdiffusion_state_predictor")
    output_directory.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    training_steps = 5000
    log_freq = 100
    save_freq = 1000

    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # --- MODIFICATION: Define input and output features for STATE PREDICTION ---
    # The diffusion model will predict a sequence of "observation.state".
    output_features = {
        # The key here ("observation.state") will become cfg.diffusion_target_key
        # if interpolate_state=True
        "observation.state": features["observation.state"]
    }
    # Input features remain observations
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }

    horizon = 16  # Length of the predicted state sequence
    n_obs_steps = 2  # Number of past observation steps for conditioning

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,  # Output is now state
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        # n_action_steps is less relevant for diffusion target if not predicting actions,
        # but config requires it. It defines how LeRobotDataset loads "action" if "action"
        # were in output_features. Here, it's not the direct diffusion target.
        # For consistency with original script structure, let's set it.
        n_action_steps=horizon,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        transformer_dim=512,
        transformer_num_layers=6,
        transformer_num_heads=8,
        vision_backbone="resnet18",
        spatial_softmax_num_keypoints=32,
        # --- MODIFICATION: Set interpolate_state=True for state prediction ---
        interpolate_state=True,  # CRITICAL for state prediction
    )

    policy = CLDiffPhyConModel(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()
    policy.to(device)

    # --- MODIFICATION: Setup delta timestamps for dataset ---
    # `cfg.observation_delta_indices` (e.g., [-1,0]) for conditioning observations.
    # `cfg.target_delta_indices` (e.g., [1,...,H] for states) for target sequence.
    # `LeRobotDataset` uses these properties of `cfg` to determine which frames to load
    # for keys present in `input_features` and `output_features`.

    # Indices for conditioning images
    # e.g., [-1, 0] for n_obs_steps=2
    image_indices_obs = list(range(1 - cfg.n_obs_steps, 1))

    # For "observation.state", it's both an input (for conditioning) and an output (target).
    # `LeRobotDataset` will load data for all unique indices required by its role as input and output.
    # The actual indices used for input are cfg.observation_delta_indices.
    # The actual indices used for output are cfg.target_delta_indices.
    # We need to provide a `delta_timestamps["observation.state"]` that covers the union of these.

    # Example: n_obs_steps=2, horizon=16
    # cfg.observation_delta_indices = [-1, 0]
    # cfg.target_delta_indices (for state prediction) = [1, 2, ..., 16]
    # So, "observation.state" needs to be loaded for indices -1, 0, 1, ..., 16.

    all_state_indices_needed = sorted(
        list(set(cfg.observation_delta_indices + cfg.target_delta_indices)))

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in image_indices_obs],
        "observation.state": [i / dataset_metadata.fps for i in all_state_indices_needed],
        # "action" is no longer a direct output of this diffusion model, so it's not strictly needed
        # in delta_timestamps unless it's an input_feature or used by some other part of the config/dataset.
        # If "action" is still needed for some global conditioning (not typical for this setup) or
        # if LeRobotDataset structure breaks without it due to policy.action_feature access,
        # it might need to be included. For now, let's assume it's not needed for state predictor training target.
        # However, CLDiffPhyConModel's _queues still initializes an "action" queue.
        # If other parts of your code (e.g. evaluation) still expect 'action' to be loadable via this config
        # for the policy's final output (even if diffusion predicts states), keep it.
        # For this training script, focusing on state prediction, we primarily care about state targets.
        # Let's add a dummy action loading for now to avoid potential issues with dataset expecting it
        # if the config.output_features (for the *policy overall*, not diffusion target) still lists action.
        # The `cfg.output_features` passed to DiffusionConfig for *this training run* is `{"observation.state": ...}`.
        # So LeRobotDataset will *not* look for "action" as an output.
    }

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

    optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-4)

    print("Starting Asynchronous STATE PREDICTION RT-Diffusion Training...")
    step = 0
    done = False

    while not done:
        for batch in tqdm(dataloader, desc=f"Training Step: {step}/{training_steps}"):
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # policy.forward_async will now use "observation.state" as the clean_sequence
            # because cfg.diffusion_target_key will point to "observation.state"
            loss = policy.forward_async(batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            if step % log_freq == 0:
                # The loss is now for state prediction
                print(
                    f"Step: {step}/{training_steps} | Async State Prediction Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"model_step_{step}.pth"
                torch.save(policy.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    final_path = output_directory / "model_final.pth"
    torch.save(policy.state_dict(), final_path)
    # Saves the config used for this training
    cfg.save_pretrained(output_directory)

    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        # Keep np.ndarray for wider compatibility
        if isinstance(value, (torch.Tensor, np.ndarray)):
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")

    print(
        f"State prediction training finished. Model saved to: {output_directory}")


if __name__ == "__main__":
    main()
