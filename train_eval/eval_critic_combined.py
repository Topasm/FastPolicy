# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# Adapted for combining separately trained models with noise critic.

"""
This script demonstrates how to evaluate a policy composed of separately trained
diffusion, inverse dynamics, and noise critic models.

It requires the installation of the 'gym_pusht' simulation environment. Install it by running:
```bash
pip install -e ".[pusht]"
```
"""

from pathlib import Path
import torch
import numpy as np
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import json

# Import necessary components
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.critic.modeling_critic_combined import CombinedCriticPolicy
from model.invdynamics.invdyn import MlpInvDynamic
# Import both critic types for flexibility
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model.critic.modernbert_critic import ModernBertCritic, ModernBertCriticConfig
from model.critic.ciritic_modules import NoiseCriticConfig, TransformerCritic


def main():
    # --- Configuration ---
    # Define paths to the individual component outputs and config/stats
    diffusion_output_dir = Path("outputs/train/diffusion_only")
    invdyn_output_dir = Path("outputs/train/invdyn_only")
    # Using noise_critic for the critic model
    noise_critic_output_dir = Path("outputs/train/modernbert_critic")

    # Assume config.json is available in diffusion output directory
    config_stats_path = diffusion_output_dir

    # Output directory for evaluation results
    output_directory = Path("outputs/eval/noise_critic")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Noise levels to test
    noise_levels = [0.05, 0.1, 0.2]

    # --- Load Config and Stats ---
    cfg_path = Path(config_stats_path) / "config.json"
    data = json.loads(cfg_path.read_text())
    cfg = DiffusionConfig(**data)

    # Load config using the base class method
    cfg = DiffusionConfig.from_pretrained(config_stats_path)
    cfg.device = device  # Override device if needed

    # Load dataset stats
    metadataset_stats = LeRobotDatasetMetadata("lerobot/pusht")
    dataset_stats = {}
    for key, stat in metadataset_stats.stats.items():
        dataset_stats[key] = {
            subkey: torch.as_tensor(subval, dtype=torch.float32, device=device)
            for subkey, subval in stat.items()
        }

    # --- Load Models ---
    # Diffusion Model
    diffusion_ckpt_path = diffusion_output_dir / "diffusion_final.pth"
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
    invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"
    if not invdyn_ckpt_path.is_file():
        raise OSError(
            f"Inverse dynamics checkpoint not found at {invdyn_ckpt_path}")
    inv_dyn_model = MlpInvDynamic(
        # State dim * 2 (current + prev)
        o_dim=cfg.robot_state_feature.shape[0],
        a_dim=cfg.action_feature.shape[0],
        hidden_dim=cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh()
    )
    print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
    inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # Noise Critic Model
    noise_critic_ckpt_path = noise_critic_output_dir / "modernbert_critic_weights.pth"

    # Create ModernBert critic model
    print(
        f"Loading ModernBert critic state dict from: {noise_critic_ckpt_path}")

    # Load critic config from the saved file if it exists
    critic_config_path = noise_critic_output_dir / "config.json"
    if critic_config_path.is_file():
        critic_config_data = json.loads(critic_config_path.read_text())
        # Make sure we have required fields
        if "state_dim" not in critic_config_data:
            critic_config_data["state_dim"] = cfg.robot_state_feature.shape[0]
        if "horizon" not in critic_config_data:
            critic_config_data["horizon"] = cfg.horizon
        if "image_feature_dim" not in critic_config_data:
            critic_config_data["image_feature_dim"] = cfg.hidden_dim

        # Add ModernBertCritic-specific fields if needed
        if "num_heads" not in critic_config_data:
            critic_config_data["num_heads"] = 8
        if "swiglu_intermediate_factor" not in critic_config_data:
            critic_config_data["swiglu_intermediate_factor"] = 4

        # Create the config from file data and initialize ModernBertCritic
        critic_cfg = ModernBertCriticConfig(**critic_config_data)
        noise_critic_model = ModernBertCritic(critic_cfg)
    else:
        # If no config file, create directly with model parameters
        noise_critic_model = ModernBertCritic(
            ModernBertCriticConfig(
                state_dim=cfg.robot_state_feature.shape[0],
                horizon=cfg.horizon,
                hidden_dim=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                dropout=cfg.dropout,
                use_layernorm=cfg.use_layernorm,
                image_feature_dim=cfg.hidden_dim,  # Use hidden_dim as image feature dimension
                num_heads=8,  # Default number of attention heads
                swiglu_intermediate_factor=4  # Default SwiGLU factor
            )
        )

    # Load state dict and move to device
    critic_state_dict = torch.load(
        noise_critic_ckpt_path, map_location=device)

    # Handle compiled model state dict with _orig_mod. prefix
    if any(k.startswith('_orig_mod.') for k in critic_state_dict.keys()):
        print("Detected compiled model state dict with '_orig_mod.' prefix, removing prefix...")
        # Create a new state dict with modified keys
        fixed_state_dict = {}
        for key, value in critic_state_dict.items():
            if key.startswith('_orig_mod.'):
                fixed_state_dict[key[len('_orig_mod.'):]] = value
            else:
                fixed_state_dict[key] = value
        critic_state_dict = fixed_state_dict

    try:
        # Try loading ModernBert model
        noise_critic_model.load_state_dict(critic_state_dict)
        noise_critic_model.eval()
        noise_critic_model.to(device)
        print("ModernBert critic model loaded successfully.")
    except RuntimeError as e:
        print(f"Failed to load ModernBert model: {e}")
        print("Falling back to TransformerCritic model...")

        # Fallback to TransformerCritic
        transformer_critic_path = Path(
            "outputs/train/transformer_critic/transformer_critic_final.pth")
        if not transformer_critic_path.is_file():
            raise OSError(
                f"TransformerCritic checkpoint not found at {transformer_critic_path}")

        # Create TransformerCritic config
        critic_cfg = NoiseCriticConfig(
            state_dim=cfg.robot_state_feature.shape[0],
            horizon=cfg.horizon,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
            use_layernorm=cfg.use_layernorm,
            use_image_context=True,  # Always use image context
            image_feature_dim=cfg.hidden_dim,  # Use hidden_dim as image feature dimension
            transformer_dim=cfg.hidden_dim,
            image_features={"observation.image": (
                3, 84, 84)}  # Default image shape
        )

        # Create and load TransformerCritic model
        noise_critic_model = TransformerCritic(critic_cfg)
        transformer_state_dict = torch.load(
            transformer_critic_path, map_location=device)
        noise_critic_model.load_state_dict(transformer_state_dict)
        noise_critic_model.eval()
        noise_critic_model.to(device)
        print("TransformerCritic model loaded successfully.")

    # Create combined model with the appropriate critic
    # Important: Set use_modernbert to False for now, as the current implementation
    # in CombinedCriticPolicy has an issue with ModernBertCritic
    combined_model = CombinedCriticPolicy(
        diffusion_model=diffusion_model,
        inv_dyn_model=inv_dyn_model,
        critic_model=noise_critic_model,
        num_samples=4,  # Generate 4 trajectory samples
        use_modernbert=False  # Set to False to avoid the error with ModernBertCritic.score()
    )

    # --- Environment Setup ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",  # Ensure this matches config expectations
        max_episode_steps=500,
    )

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

        # Create observation dictionary for the model
        observation = {
            "observation.state": state,
            "observation.image": image,
        }

        # Reset combined model if starting a new episode
        if step == 0:
            combined_model.reset()

        # Get action from the combined policy
        # The select_action method may return just action or (action, trajectories)
        with torch.inference_mode():
            # Call our updated select_action method that uses the critic model
            result = combined_model.select_action(
                curr_state=observation["observation.state"],
                image=image
            )

            # Handle both possible return types
            if isinstance(result, tuple):
                action, _ = result  # Ignore trajectories
            else:
                action = result

        # Convert to numpy for environment step
        numpy_action = action.squeeze(0).cpu().numpy()

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

    # Get the speed of environment
    fps = env.metadata["render_fps"]

    # Encode all frames into a mp4 video
    video_path = output_directory / "rollout.mp4"
    imageio.mimsave(str(video_path), np.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")

    # Save metrics to a JSON file
    metrics = {
        "noise_levels": noise_levels,
        "returns": rewards,
        "mean_return": np.mean(rewards),
        "std_return": np.std(rewards),
        "auc": [],
        "accuracy": []
    }

    # Save metrics to JSON
    metrics_file = output_directory / "metrics.json"
    with open(metrics_file, "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        metrics_json = metrics.copy()
        for key, value in metrics_json.items():
            if isinstance(value, np.ndarray):
                metrics_json[key] = value.tolist()
        json.dump(metrics_json, f, indent=4)
    print(f"Metrics saved to {metrics_file}")

    # Close environment
    env.close()


if __name__ == "__main__":
    main()
