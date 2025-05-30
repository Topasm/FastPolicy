"""
This script demonstrates how to evaluate a policy that uses a bidirectional transformer 
for state prediction and an RT-Diffusion model for action generation.

The bidirectional transformer generates forward states (0-16) from images,
which are then passed to the RT-Diffusion model to generate actions.


"""
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)
from model.modeling_bidirectional_rtdiffusion import BidirectionalRTDiffusionPolicy
from pathlib import Path

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy
import torch
import json  # Added for loading rtdiff_config directly
from torch import nn
from torch.utils.data import DataLoader
import os
import numpy as np


def main():

    # --- Configuration ---
    # Define paths to the individual component outputs
    bidirectional_output_dir = Path("outputs/train/bidirectional_transformer")
    # RT-Diffusion model output dir
    rtdiff_output_dir = Path("outputs/train/rtdiffusion")

    output_directory = Path("outputs/eval/bidirectional_rtdiffusion")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Set up bidirectional transformer config ---
    bidir_cfg = BidirectionalARTransformerConfig(
        # Match the state_dim from the trained model (based on error message)
        state_dim=2,
        image_size=96,  # Match the image size used in training
        forward_steps=16,  # Number of forward steps to predict
        backward_steps=16,  # Not used in inference but required for model
    )

    # --- Load Dataset Metadata for normalization statistics ---
    print("Loading dataset metadata for normalization...")
    metadata = LeRobotDatasetMetadata("lerobot/pusht")

    # Print available keys to help debugging
    print(f"Available metadata stats keys: {metadata.stats.keys()}")
    for key_meta in metadata.stats.keys():  # Renamed key to key_meta to avoid conflict
        print(
            f"Keys in metadata.stats[{key_meta}]: {metadata.stats[key_meta].keys()}")

    # Process the metadata statistics for BidirectionalRTDiffusionPolicy (needs tensors on device)
    # This version will be passed to BidirectionalRTDiffusionPolicy
    processed_dataset_stats = {}
    for key, value in metadata.stats.items():
        processed_dataset_stats[key] = {}
        for stat_key, stat_value in value.items():
            if isinstance(stat_value, torch.Tensor):
                processed_dataset_stats[key][stat_key] = stat_value.to(device)
            else:
                processed_dataset_stats[key][stat_key] = torch.tensor(
                    stat_value, dtype=torch.float32, device=device)  # Ensure float32

    # --- Load Bidirectional Transformer Model ---
    bidirectional_ckpt_path = bidirectional_output_dir / "final_model.pt"
    if not bidirectional_ckpt_path.is_file():
        # Try model_weights.pt as fallback
        model_weights_path = bidirectional_output_dir / "model_weights.pt"
        if model_weights_path.is_file():
            bidirectional_ckpt_path = model_weights_path
        else:
            # Try model_final.pth as another fallback
            model_final_path = bidirectional_output_dir / "model_final.pth"
            if model_final_path.is_file():
                bidirectional_ckpt_path = model_final_path
            else:
                raise OSError(
                    f"Bidirectional transformer checkpoint not found at {bidirectional_output_dir} with common names.")

    transformer_model = BidirectionalARTransformer(bidir_cfg)
    print(f"Loading bidirectional transformer from: {bidirectional_ckpt_path}")

    # Load the checkpoint
    checkpoint = torch.load(bidirectional_ckpt_path, map_location="cpu")

    # Extract the model state dictionary - handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # This is a training checkpoint with metadata
        model_state_dict = checkpoint["model_state_dict"]
    else:
        # This might be a direct state dict
        model_state_dict = checkpoint

    # Load the state dictionary
    transformer_model.load_state_dict(model_state_dict)
    transformer_model.eval()
    transformer_model.to(device)

    # --- Load RT-Diffusion Model ---
    # Load configuration by directly reading the JSON and instantiating DiffusionConfig
    rtdiff_config_json_path = rtdiff_output_dir / \
        "config.json"  # config.json 경로 확인은 유지
    if not rtdiff_config_json_path.is_file():
        raise OSError(
            f"RT-Diffusion config JSON not found at {rtdiff_config_json_path}")
    print(
        f"Loading RT-Diffusion configuration from directory: {rtdiff_output_dir}")
    # from_pretrained 메서드를 사용하여 설정 로드
    rtdiff_config = DiffusionConfig.from_pretrained(rtdiff_output_dir)
    # rtdiff_config.device = device # DiffusionConfig might not have a device attribute, model is moved to device later

    # Create model instance
    # CLDiffPhyConModel (a PreTrainedPolicy) expects raw metadata.stats
    rt_diffusion_model = CLDiffPhyConModel(
        config=rtdiff_config,
        dataset_stats=metadata.stats  # Pass the original metadata.stats
    )

    # Load model weights - try various possible checkpoint file names
    possible_ckpt_names = ["model.pth", "model_weights.pt",
                           "model_final.pth", "final_model.pt"]
    rtdiff_ckpt_path = None

    for name in possible_ckpt_names:
        path = rtdiff_output_dir / name
        if path.is_file():
            rtdiff_ckpt_path = path
            break

    if rtdiff_ckpt_path is None:
        raise OSError(
            f"RT-Diffusion checkpoint not found in {rtdiff_output_dir} with common names.")

    print(f"Loading RT-Diffusion model from: {rtdiff_ckpt_path}")
    checkpoint_rtdiff = torch.load(
        rtdiff_ckpt_path, map_location="cpu")  # Renamed to avoid conflict

    # Extract the model state dictionary - handle different checkpoint formats
    if isinstance(checkpoint_rtdiff, dict) and "model_state_dict" in checkpoint_rtdiff:
        rtdiff_state_dict = checkpoint_rtdiff["model_state_dict"]
    else:
        rtdiff_state_dict = checkpoint_rtdiff

    rt_diffusion_model.load_state_dict(rtdiff_state_dict, strict=False)

    rt_diffusion_model.eval()
    rt_diffusion_model.to(device)

    # Create combined policy with bidirectional transformer and RT-Diffusion
    combined_policy = BidirectionalRTDiffusionPolicy(
        bidirectional_transformer=transformer_model,
        rt_diffusion_model=rt_diffusion_model,
        dataset_stats=processed_dataset_stats,  # Pass the device-processed stats here
        n_obs_steps=rtdiff_config.n_obs_steps
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
    frames.append(env.render())  # Render initial frame
    step = 0
    done = False

    print("Starting evaluation rollout with Bidirectional Transformer + RT-Diffusion...")
    while not done:
        # --- Prepare Observation ---
        state_np = numpy_observation["agent_pos"].astype(
            np.float32)  # Ensure float32
        image_np = numpy_observation["pixels"].astype(
            np.float32)  # Ensure float32

        # Normalize image to [0,1] and permute to CHW
        current_image_tensor = torch.from_numpy(
            image_np / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        current_state_tensor = torch.from_numpy(
            state_np).unsqueeze(0).to(device)

        # Create observation dictionary for the model
        observation = {
            "observation.state": current_state_tensor,
            "observation.image": current_image_tensor,
        }

        # Reset combined policy if starting a new episode
        if step == 0:
            combined_policy.reset()

        # Get action from the combined policy
        with torch.inference_mode():
            action = combined_policy.select_action(observation)

        # Convert to numpy for environment step
        numpy_action = action.squeeze(0).cpu().numpy()

        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)

        print(f"{step=} {reward=} {terminated=}")

        rewards.append(reward)
        frames.append(env.render())

        done = terminated or truncated  # Corrected done condition
        step += 1

    print(f"Episode ended after {step} steps.")
    total_reward = sum(rewards)
    print(f"Total reward: {total_reward}")
    if terminated and not truncated:  # Check for successful termination
        print("Success!")
    else:
        print("Failure or Timed Out!")

    # Get the speed of environment (i.e. its number of frames per second)
    fps = env.metadata.get("render_fps", 30)  # Added default for fps

    # Save the video
    video_path = output_directory / "rollout_bidir_rtdiff.mp4"  # Changed filename
    imageio.mimsave(str(video_path), numpy.stack(frames), fps=fps)
    print(f"Video of the evaluation is available in '{video_path}'.")


if __name__ == "__main__":
    main()
