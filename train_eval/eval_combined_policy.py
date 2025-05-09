# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for combining separately trained models.

"""
This script demonstrates how to evaluate a policy composed of separately trained
diffusion, inverse dynamics, and optional critic models.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path
from collections import deque
import os

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
import safetensors
import einops
import json

# Import necessary components
# Import PreTrainedConfig base class
from lerobot.configs.policies import PreTrainedConfig
# Keep this for type hints if needed
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic
from model.critic.multimodal_scorer import MultimodalTrajectoryScorer
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues


def main():
    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path("outputs/train/diffusion_only")
    invdyn_output_dir = Path("outputs/train/invdyn_only")
    # Assuming critic was trained here
    critic_output_dir = Path("outputs/train/critic_only")

    # Assume config.json and stats.safetensors are available, e.g., copied from dataset or saved during training.
    # Let's try loading them from the diffusion output directory first.
    # Alternatively, define a specific path if they are elsewhere.
    # ADJUST if config/stats are elsewhere
    config_stats_path = diffusion_output_dir

    output_directory = Path("outputs/eval/combined_policy")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_critic = False  # Set to False to disable critic-based sample selection
    # Number of state sequences to sample (if > 1 and use_critic=True)
    num_inference_samples = 8

    # --- Load Config and Stats ---
    config_path = config_stats_path

    cfg_path = Path(config_stats_path) / "config.json"
    data = json.loads(cfg_path.read_text())
    cfg = DiffusionConfig(**data)

    # Load config using the base class method
    # This relies on "policy_name": "mydiffusion" being present in config.json
    # and DiffusionConfig being registered correctly.
    cfg = DiffusionConfig.from_pretrained(config_path)

    cfg.device = device  # Override device if needed

    # Load from the same dir as config
    stats_path = config_stats_path / "stats.safetensors"

    metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
    dataset_stats: dict[str, dict[str, torch.Tensor]] = {}
    for key, stat in metadataset_stats.stats.items():
        dataset_stats[key] = {
            subkey: torch.as_tensor(subval, dtype=torch.float32, device=device)
            for subkey, subval in stat.items()
        }

    # --- Load Models ---
    # Diffusion Model
    diffusion_ckpt_path = diffusion_output_dir / \
        "diffusion_final.pth"  # Use final checkpoint name
    if not diffusion_ckpt_path.is_file():
        raise OSError(
            f"Diffusion checkpoint not found at {diffusion_ckpt_path}")
    diffusion_model = MyDiffusionModel(cfg)
    print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
    diff_state_dict = torch.load(diffusion_ckpt_path, map_location="cpu")
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.eval()
    diffusion_model.to(device)

    # Inverse Dynamics Model
    invdyn_ckpt_path = invdyn_output_dir / \
        "invdyn_final.pth"  # Use final checkpoint name
    if not invdyn_ckpt_path.is_file():
        raise OSError(
            f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path}")
    inv_dyn_model = MlpInvDynamic(
        o_dim=cfg.robot_state_feature.shape[0] * 2,
        a_dim=cfg.action_feature.shape[0],
        hidden_dim=cfg.inv_dyn_hidden_dim,
        # Add other params from MlpInvDynamic if needed (dropout, layernorm, activation)
    )
    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # Critic Model (Optional)
    critic_model = None
    critic_ckpt_path = critic_output_dir / \
        "critic_final.pth"  # Use final checkpoint name
    if use_critic and critic_ckpt_path.is_file():
        print(f"Loading critic state dict from: {critic_ckpt_path}")
        critic_model = MultimodalTrajectoryScorer(
            state_dim=cfg.robot_state_feature.shape[0],
            horizon=cfg.horizon,
            hidden_dim=cfg.critic_hidden_dim
        )
        crit_state_dict = torch.load(critic_ckpt_path, map_location="cpu")
        critic_model.load_state_dict(crit_state_dict)
        critic_model.eval()
        critic_model.to(device)
        print("Critic model loaded and will be used for sample selection.")
    elif use_critic:
        print(
            f"Info: Critic usage enabled but checkpoint not found at {critic_ckpt_path}. Critic will not be used.")
        use_critic = False  # Disable critic if file not found
    else:
        print("Critic usage disabled.")

    # --- Normalization ---
    normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_stats)
    unnormalize_action_output = Unnormalize(
        {"action": cfg.action_feature}, cfg.normalization_mapping, dataset_stats)

    # --- Environment Setup ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",  # Ensure this matches config expectations
        max_episode_steps=1000,
    )

    # --- Queues for History ---
    queues = {
        "observation.state": deque(maxlen=cfg.n_obs_steps),
        # Use "observation.image" consistent with config/training
        "observation.image": deque(maxlen=cfg.n_obs_steps),
        "action": deque(maxlen=cfg.n_action_steps),
    }
    # Add env state queue if needed by config
    if cfg.env_state_feature:
        queues["observation.environment_state"] = deque(maxlen=cfg.n_obs_steps)

    # --- Evaluation Loop ---
    numpy_observation, info = env.reset(seed=42)
    rewards = []
    frames = []
    frames.append(env.render())
    step = 0
    done = False

    print("Starting evaluation rollout...")
    while not done:
        # --- Prepare Observation ---
        state_np = numpy_observation["agent_pos"]
        image_np = numpy_observation["pixels"]

        state = torch.from_numpy(state_np).to(torch.float32)
        image = torch.from_numpy(image_np).to(torch.float32) / 255
        image = image.permute(2, 0, 1)  # HWC to CHW

        # Add batch dim and move to device
        state = state.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)

        # Create batch dict for normalization and queue population
        current_obs_batch = {
            "observation.state": state,
            "observation.image": image,
            # Add env state if needed: "observation.environment_state": env_state_tensor
        }

        # --- Normalize and Populate Queues ---
        norm_obs_batch = normalize_inputs(current_obs_batch)
        # Populates with normalized obs
        queues = populate_queues(queues, norm_obs_batch)

        # --- Action Generation ---
        if len(queues["action"]) == 0:
            if len(queues["observation.state"]) < cfg.n_obs_steps:
                print(
                    f"Waiting for observation queue to fill ({len(queues['observation.state'])}/{cfg.n_obs_steps})...")
                # Optionally take a default/random action while queue fills
                numpy_action = env.action_space.sample()
            else:
                print("Generating new action plan...")
                # Prepare batch for the model by stacking history from queues (already normalized)
                model_input_batch = {}
                for key, queue in queues.items():
                    if key.startswith("observation"):
                        # Items should already be tensors
                        queue_list = [item.to(device) for item in queue]
                        # Stack along time dimension (dim=1)
                        model_input_batch[key] = torch.stack(queue_list, dim=1)

                # Get the very last state (t=0 relative to window)
                # State queue has [s_{-1}, s_{0}] for n_obs=2. We need s_0.
                # Get the last state in the history
                current_state = model_input_batch["observation.state"][:, 0, :]
                num_samples = getattr(data, "num_inference_samples", 1)
                actions = diffusion_model.generate_actions_via_inverse_dynamics(
                    model_input_batch,  # Pass normalized batch
                    current_state,     # Pass normalized state
                    inv_dyn_model,
                    num_samples=num_samples,
                )

                actions_unnormalized = unnormalize_action_output(
                    {"action": actions})["action"]

        queues["action"].extend(actions_unnormalized.transpose(0, 1))
        action = queues["action"].popleft()

        numpy_action = action.squeeze(0).to("cpu").numpy()

        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)
        print(f"{step=} {reward=} {terminated=}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated or truncated or done
        step += 1
    if terminated:
        print("Success!")
    else:
        print("Failure!")

    # Get the speed of environment (i.e. its number of frames per second).
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video.
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
