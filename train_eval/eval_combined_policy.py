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

# Import necessary components
# Import PreTrainedConfig base class
from lerobot.configs.policies import PreTrainedConfig
# Keep this for type hints if needed
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic
from model.critic.critic_model import CriticScorer
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
    use_critic = True  # Set to False to disable critic-based sample selection
    # Number of state sequences to sample (if > 1 and use_critic=True)
    num_inference_samples = 8

    # --- Load Config and Stats ---
    config_path = config_stats_path / "config.json"  # Load from specified path
    if not config_path.is_file():
        # If not found, try loading from invdyn dir as fallback
        config_path = invdyn_output_dir / "config.json"
        if not config_path.is_file():
            # If still not found, try loading from critic dir
            config_path = critic_output_dir / "config.json"
            if not config_path.is_file():
                raise OSError(
                    f"config.json not found in specified paths: {config_stats_path}, {invdyn_output_dir}, {critic_output_dir}")
        # Update config_stats_path if found elsewhere
        config_stats_path = config_path.parent

    # Load config using the base class method
    # This relies on "policy_name": "mydiffusion" being present in config.json
    # and DiffusionConfig being registered correctly.
    cfg = PreTrainedConfig.from_json_file(config_path)
    # Check if the loaded config is actually the expected type (optional but good practice)
    if not isinstance(cfg, DiffusionConfig):
        print(
            f"Warning: Loaded config type is {type(cfg)}, expected DiffusionConfig. Check 'policy_name' in config.json.")

    cfg.device = device  # Override device if needed

    # Load from the same dir as config
    stats_path = config_stats_path / "stats.safetensors"
    if not stats_path.is_file():
        raise OSError(
            f"stats.safetensors not found in {config_stats_path}")
    with safetensors.safe_open(stats_path, framework="pt", device="cpu") as f:
        dataset_stats = {k: f.get_tensor(k) for k in f.keys()}

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
        critic_model = CriticScorer(
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
                norm_current_state = model_input_batch["observation.state"][:, -1, :]

                # Generate future states using Diffusion Model
                with torch.no_grad():
                    # Prepare conditioning
                    global_cond = diffusion_model._prepare_global_conditioning(
                        model_input_batch)

                    # Repeat global_cond if sampling multiple sequences
                    eff_num_samples = num_inference_samples if use_critic and critic_model else 1
                    if eff_num_samples > 1:
                        global_cond = global_cond.repeat_interleave(
                            eff_num_samples, dim=0)

                    # Sample future state sequences
                    predicted_states_flat = diffusion_model.conditional_sample(
                        batch_size=1 * eff_num_samples,  # B=1 for eval
                        global_cond=global_cond
                    )  # (B*N, H, D_state)

                    # Reshape and select best sample if using critic
                    if eff_num_samples > 1:
                        predicted_states = einops.rearrange(
                            predicted_states_flat, "(b n) h d -> b n h d", b=1, n=eff_num_samples
                        )  # (1, N, H, D_state)

                        # Score sequences with critic
                        scores = critic_model.score_sequences(
                            predicted_states.squeeze(0))  # Input (N, H, D), Output (N, 1)
                        best_sample_idx = torch.argmax(scores)
                        # (1, H, D_state)
                        selected_states = predicted_states[:, best_sample_idx]
                        print(
                            f"Critic selected sample {best_sample_idx.item()} with score {scores[best_sample_idx].item():.4f}")
                    else:
                        selected_states = predicted_states_flat.unsqueeze(
                            0)  # (1, H, D_state) if B=1, N=1

                    # Iteratively predict actions using Inverse Dynamics Model
                    predicted_actions = []
                    current_s = norm_current_state  # s_0 (normalized)
                    for i in range(cfg.n_action_steps):  # Predict n_action_steps actions
                        # Check if we need more states than predicted
                        if i >= selected_states.shape[1]:
                            print(
                                f"Warning: Requested {cfg.n_action_steps} actions, but only {selected_states.shape[1]} states predicted. Stopping action prediction.")
                            break
                        next_s = selected_states[:, i]  # s_{i+1} (normalized)

                        state_pair = torch.cat(
                            [current_s, next_s], dim=-1)  # (1, D_state * 2)
                        action_i = inv_dyn_model.predict(
                            state_pair)  # (1, D_action) (normalized)
                        predicted_actions.append(action_i)
                        current_s = next_s  # Update for next iteration

                    if not predicted_actions:
                        raise RuntimeError(
                            "No actions were predicted. Check state sequence length and n_action_steps.")

                    actions_normalized = torch.cat(
                        predicted_actions, dim=0)  # (n_actions, D_action)

                    # Unnormalize actions
                    actions_unnormalized = unnormalize_action_output(
                        # Add batch dim for unnorm
                        {"action": actions_normalized.unsqueeze(0)}
                    )["action"].squeeze(0)  # Remove batch dim -> (n_actions, D_action)

                    # Add predicted actions to the action queue
                    # Need to transpose? No, extend expects individual items.
                    for action_tensor in actions_unnormalized:
                        queues["action"].append(action_tensor)

            # Pop the next action from the queue if available
            if queues["action"]:
                action_tensor = queues["action"].popleft()
                numpy_action = action_tensor.to("cpu").numpy()
            # If queue was empty and still is (e.g., waiting for obs), numpy_action remains from previous step or sample
            elif 'numpy_action' not in locals():
                print("Action queue empty, sampling random action.")
                numpy_action = env.action_space.sample()

        # --- Step Environment ---
        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)
        print(
            f"Step: {step}, Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated | truncated
        step += 1

    # --- Log Results ---
    if terminated:
        print("Success!")
    else:
        print("Failure!")

    total_reward = sum(rewards)
    print(f"Total Steps: {step}")
    print(f"Total Reward: {total_reward:.4f}")

    fps = env.metadata["render_fps"]
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
