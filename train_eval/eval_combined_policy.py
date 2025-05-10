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
import argparse

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
import einops
import json
# Keep this for type hints if needed
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic, SeqInvDynamic
from model.critic.multimodal_scorer import MultimodalTrajectoryScorer
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.utils import populate_queues


# Custom helper function to handle both MLP and Sequential inverse dynamics models
def generate_actions_with_invdyn(
    diffusion_model,
    inv_dyn_model,
    norm_batch,
    norm_current_state,
    use_seq_model=False,
    seq_length=3,
    num_inference_samples=1
):
    """Generate actions using diffusion model for state prediction and inverse dynamics for action prediction.

    Args:
        diffusion_model: The diffusion model instance
        inv_dyn_model: The inverse dynamics model (either MLP or Sequential)
        norm_batch: Normalized batch dictionary with observation history
        norm_current_state: Current normalized state
        use_seq_model: Whether the inverse dynamics model is sequential
        seq_length: Length of state sequence for sequential model
        num_inference_samples: Number of diffusion samples to generate

    Returns:
        Tensor of shape [B, n_action_steps, action_dim] with normalized actions
    """
    batch_size = norm_current_state.shape[0]
    device = norm_current_state.device
    n_action_steps = diffusion_model.config.n_action_steps  # This is typically 4

    # Prepare conditioning from the normalized batch
    global_cond = diffusion_model._prepare_global_conditioning(norm_batch)

    # Repeat global_cond if num_inference_samples > 1
    if num_inference_samples > 1:
        global_cond = global_cond.repeat_interleave(
            num_inference_samples, dim=0)

    # Sample normalized future states
    predicted_states_flat = diffusion_model.conditional_sample(
        batch_size * num_inference_samples, global_cond=global_cond
    )  # Output: (B*N, horizon, D_state)

    predicted_states = einops.rearrange(
        predicted_states_flat, "(b n) h d -> b n h d", b=batch_size, n=num_inference_samples
    )

    # Always select the first sample if multiple are generated
    selected_states = predicted_states[:, 0]  # (B, horizon, D_state)

    # Iteratively predict actions
    predicted_actions = []

    if use_seq_model:
        # For sequential model, prepare sequence of states for each action prediction
        for i in range(n_action_steps):
            # For each action i, we need states from i-seq_length+1 to i+1
            # First gather history states that we need
            states_seq = []

            # Get the required states for the sequence
            # We need seq_length states including the next state to predict an action

            # Add states from history if needed (already available in norm_current_state and norm_batch)
            if i < seq_length - 1:
                history_states_needed = seq_length - 1 - i
                # Get states from history
                if "observation.state" in norm_batch and norm_batch["observation.state"].shape[1] >= history_states_needed:
                    history_states = norm_batch["observation.state"][:, -
                                                                     history_states_needed:, :]
                    states_seq.append(history_states)

                # Add current state
                states_seq.append(norm_current_state.unsqueeze(1))

            # Add already predicted future states
            future_states_needed = min(i + 1, seq_length - 1)
            if future_states_needed > 0:
                future_states = selected_states[:, :future_states_needed, :]
                states_seq.append(future_states)

            # Add the next state for which we want to predict action
            next_state = selected_states[:, i, :].unsqueeze(1)
            states_seq.append(next_state)

            # Concatenate all states in sequence
            # [B, seq_length, D_state]
            state_seq = torch.cat(states_seq, dim=1)

            # Ensure we have the right sequence length
            if state_seq.shape[1] != seq_length:
                # Pad or trim if necessary
                if state_seq.shape[1] < seq_length:
                    # Pad by repeating the first state
                    pad_length = seq_length - state_seq.shape[1]
                    padding = state_seq[:, :1, :].repeat(1, pad_length, 1)
                    state_seq = torch.cat([padding, state_seq], dim=1)
                else:
                    # Trim to keep the most recent states
                    state_seq = state_seq[:, -seq_length:, :]

            # Predict action using sequential model
            action_i = inv_dyn_model.predict(state_seq)
            # Get the final output for this position
            action_i = action_i[:, -1, :]  # Shape: [B, action_dim]
            predicted_actions.append(action_i)
    else:
        # Original MLP approach using state pairs
        current_s = norm_current_state  # s_0
        for i in range(n_action_steps):
            next_s = selected_states[:, i]  # s_1, s_2, s_3, s_4

            # Create state pair for MLP model
            state_pair = torch.cat([current_s, next_s], dim=-1)

            # Predict action
            action_i = inv_dyn_model.predict(state_pair)
            predicted_actions.append(action_i)

            # Update current state for next iteration
            current_s = next_s

    # Stack the predicted actions
    actions_to_execute = torch.stack(
        predicted_actions, dim=1)  # Shape: (B, n_action_steps, action_dim)

    return actions_to_execute


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Evaluate combined diffusion and inverse dynamics policy")
    parser.add_argument("--diffusion_dir", type=str, default="outputs/train/diffusion_only",
                        help="Directory containing the diffusion model checkpoint")
    parser.add_argument("--invdyn_dir", type=str, default="outputs/train/invdyn_only",
                        help="Directory containing the inverse dynamics model checkpoint")
    parser.add_argument("--critic_dir", type=str, default="outputs/train/critic_only",
                        help="Directory containing the critic model checkpoint")
    parser.add_argument("--output_dir", type=str, default="outputs/eval/combined_policy",
                        help="Directory to save evaluation results")
    parser.add_argument("--use_seq_model", action="store_true", default=False,
                        help="Use sequential inverse dynamics model")
    parser.add_argument("--seq_length", type=int, default=3,
                        help="State sequence length for sequential model")
    parser.add_argument("--gru_layers", type=int, default=2,
                        help="Number of GRU layers for sequential model")
    parser.add_argument("--use_critic", action="store_true", default=False,
                        help="Use critic model for sample selection")
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Number of diffusion samples to generate")

    args = parser.parse_args()

    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path(args.diffusion_dir)
    invdyn_output_dir = Path(args.invdyn_dir)
    # Assuming critic was trained here
    critic_output_dir = Path(args.critic_dir)

    # --- Inverse Dynamics Model Configuration ---
    # Get model configuration from command-line args
    use_seq_model = args.use_seq_model
    seq_length = args.seq_length
    gru_layers = args.gru_layers

    # Define model checkpoint naming pattern based on training script
    if use_seq_model:
        invdyn_ckpt_name = f"invdyn_seq_seq{seq_length}_final.pth"
    else:
        invdyn_ckpt_name = "invdyn_final.pth"  # Original MLP model checkpoint name

    # Assume config.json and stats.safetensors are available, e.g., copied from dataset or saved during training.
    # Let's try loading them from the diffusion output directory first.
    # Alternatively, define a specific path if they are elsewhere.
    # ADJUST if config/stats are elsewhere
    config_stats_path = diffusion_output_dir

    output_directory = Path(args.output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_critic = args.use_critic  # Use the command-line arg
    # Number of state sequences to sample (if > 1 and use_critic=True)
    num_inference_samples = args.num_samples

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
    invdyn_ckpt_path = invdyn_output_dir / invdyn_ckpt_name

    if not invdyn_ckpt_path.is_file():
        print(f"Warning: Checkpoint not found at {invdyn_ckpt_path}")

        # Try alternative naming schemes
        if use_seq_model:
            print("Trying alternative checkpoint names for sequential model...")
            alternatives = [
                # train_invdyn.py naming
                f"invdyn_seq_seq{seq_length}_final.pth",
                "invdyn_seq_final.pth",                  # without seq length
                "invdyn_seq_model_final.pth",             # generic seq name
                "invdyn_final.pth"                        # default name
            ]

            for alt_name in alternatives:
                alt_path = invdyn_output_dir / alt_name
                if alt_path.is_file():
                    invdyn_ckpt_path = alt_path
                    print(f"Found alternative checkpoint: {alt_path}")
                    break
        else:
            print("Falling back to default checkpoint name...")
            invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"

        # Final check if we found any checkpoint
        if not invdyn_ckpt_path.is_file():
            raise OSError(
                f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path} or any alternative locations in {invdyn_output_dir}")

    # Initialize the appropriate model type
    if use_seq_model:
        print(
            f"Using sequential inverse dynamics model with {seq_length} state context")
        # Use the number of layers from command line
        print(f"Using GRU with {gru_layers} layers")
        inv_dyn_model = SeqInvDynamic(
            state_dim=cfg.robot_state_feature.shape[0],
            action_dim=cfg.action_feature.shape[0],
            hidden_dim=cfg.inv_dyn_hidden_dim,
            n_layers=gru_layers,
            dropout=0.1
        )
    else:
        print("Using MLP inverse dynamics model")
        inv_dyn_model = MlpInvDynamic(
            o_dim=cfg.robot_state_feature.shape[0] * 2,
            a_dim=cfg.action_feature.shape[0],
            hidden_dim=cfg.inv_dyn_hidden_dim,
            dropout=0.1,
            use_layernorm=True,
            out_activation=torch.nn.Tanh(),
        )

    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")

    # Add debug option to print model and state dict structure
    debug_model_loading = True
    if debug_model_loading:
        print("\n--- Model Structure Debug ---")
        print("Model Parameter Keys:")
        for name, _ in inv_dyn_model.named_parameters():
            print(f"  {name}")
        print("\nState Dict Keys:")
        for key in inv_state_dict.keys():
            print(f"  {key}")
        print("---------------------------\n")

    try:
        inv_dyn_model.load_state_dict(inv_state_dict)
    except RuntimeError as e:
        print(f"Error loading state dict: {e}")
        # Attempt to load with strict=False which will ignore missing/unexpected keys
        print("Attempting to load with strict=False...")
        inv_dyn_model.load_state_dict(inv_state_dict, strict=False)
        print("Model loaded with non-strict matching. Some weights may not be used.")

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

    # Print evaluation configuration
    model_type = "Sequential" if use_seq_model else "MLP"
    print(
        f"Starting evaluation rollout with {model_type} inverse dynamics model...")
    if use_seq_model:
        print(f"Using sequence length of {seq_length} states for prediction")
    print(
        f"Sampling {num_inference_samples} trajectories from diffusion model")
    if use_critic:
        print("Using critic for trajectory selection")
    else:
        print("Not using critic (taking first sample)")
    print("Output will be saved to", output_directory)
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

                # Use our custom helper function that supports both model types
                actions = generate_actions_with_invdyn(
                    diffusion_model,
                    inv_dyn_model,
                    model_input_batch,
                    current_state,
                    use_seq_model=use_seq_model,
                    seq_length=seq_length,
                    num_inference_samples=num_inference_samples,
                )

                actions_unnormalized = unnormalize_action_output(
                    {"action": actions})["action"]

                # Convert actions_unnormalized to a list of individual action tensors
                # The tensor is of shape [B, n_action_steps, action_dim]
                # We need to extract each action and store it separately in the queue
                actions_list = []
                for i in range(actions_unnormalized.shape[1]):
                    # Create a new tensor for each action to ensure they're distinct objects
                    action_tensor = actions_unnormalized[0, i].clone()
                    actions_list.append(action_tensor)

                # Clear previous actions and add the new ones
                queues["action"].clear()  # Clear any old actions

                # Add each action tensor as a separate entry in the queue
                for i, action_tensor in enumerate(actions_list):
                    queues["action"].append(action_tensor)

                print(
                    f"Added {len(actions_list)} actions to queue. Queue now has {len(queues['action'])} actions.")

        # Get the next action from the queue, if empty, skip this step
        if len(queues["action"]) > 0:
            action = queues["action"].popleft()
            numpy_action = action.to("cpu").numpy()
        else:
            print("WARNING: Action queue is empty! Using random action.")
            numpy_action = env.action_space.sample()

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

    # Create a descriptive filename for the video
    model_type = "seq" if use_seq_model else "mlp"
    if use_seq_model:
        video_name = f"rollout_{model_type}_seq{seq_length}.mp4"
    else:
        video_name = "rollout_mlp.mp4"

    # Encode all frames into a mp4 video.
    video_path = output_directory / video_name
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)

    # Calculate and display the final performance metrics
    total_reward = sum(rewards)
    avg_reward = total_reward / len(rewards) if rewards else 0
    success = "Success!" if terminated else "Failure!"

    print(f"Evaluation complete: {success}")
    print(f"Total reward: {total_reward:.4f}")
    print(f"Average reward: {avg_reward:.4f}")
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
