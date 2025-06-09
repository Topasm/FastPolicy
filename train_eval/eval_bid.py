from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
# CLDiffPhyConModel will be used as the State Diffusion Model
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.predictor.policy import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.configs.types import NormalizationMode
from lerobot.common.datasets.utils import dataset_to_policy_features
# Import the modified BidirectionalRTDiffusionPolicy
from model.modeling_bidirectional_rtdiffusion import BidirectionalRTDiffusionPolicy
from model.invdyn.invdyn import MlpInvDynamic  # Import MlpInvDynamic

from pathlib import Path
import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio
import numpy  # numpy was imported as numpy, not np
import torch
import json


def main():
    # --- Configuration ---
    bidirectional_output_dir = Path(
        "outputs/train/bidirectional_transformer32")

    state_diffusion_output_dir = Path(
        "outputs/train/rtdiffusion_state_predictor")
    invdyn_output_dir = Path("outputs/train/invdyn_only")

    output_directory = Path("outputs/eval/bidirectional_rtdiffusion_3stage")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Dataset Metadata for normalization statistics ---
    print("Loading dataset metadata for normalization...")
    metadata = LeRobotDatasetMetadata("lerobot/pusht")

    predicted_image_dir = Path("outputs/eval/predicted_images")
    predicted_image_dir.mkdir(parents=True, exist_ok=True)

    # Get features for normalization
    # Convert raw feature dictionaries to PolicyFeature objects
    policy_features = dataset_to_policy_features(metadata.features)

    input_features = {
        "observation.state": policy_features["observation.state"]
    }

    output_features = {
        "action": policy_features["action"]
    }

    processed_dataset_stats = {}
    for key, value_dict in metadata.stats.items():
        processed_dataset_stats[key] = {}
        if isinstance(value_dict, dict):
            for stat_key, stat_value in value_dict.items():
                try:
                    # Ensure all stats are float32 for consistency with model parameters
                    processed_dataset_stats[key][stat_key] = torch.as_tensor(
                        stat_value, dtype=torch.float32, device=device)
                except Exception as e:
                    print(
                        f"Warning: Could not convert stat {stat_key} for {key} to tensor: {e}. Value: {stat_value}")
                    # Keep original if conversion fails
                    processed_dataset_stats[key][stat_key] = stat_value
        else:
            # Handle cases where a top-level stat might not be a dict (e.g. fps)
            processed_dataset_stats[key] = value_dict

    # --- Load BidirectionalARTransformer Config and Model ---
    bidir_config_path = bidirectional_output_dir / "config.json"
    if bidir_config_path.is_file():
        # Load the configuration from the pretrained model
        bidir_cfg = BidirectionalARTransformerConfig.from_pretrained(
            bidirectional_output_dir)
        print(
            f"Loaded BidirectionalARTransformerConfig from {bidir_config_path}")

        print("Note: Using modified BidirectionalARTransformer with query-based inference. "
              "This version should be significantly faster during inference.")
    else:
        print(
            f"BidirectionalARTransformerConfig json not found at {bidir_config_path}. Using manual config.")
        state_dim_from_meta = metadata.features.get(
            "observation.state", {}).get("shape", [2])[-1]
        # Ensure image_channels from metadata if available
        image_example_key = next(iter(metadata.camera_keys), None)
        image_channels_from_meta = 3  # default
        if image_example_key and image_example_key in metadata.features:
            image_channels_from_meta = metadata.features[image_example_key][
                "shape"][-1] if metadata.features[image_example_key]["shape"][-1] in [1, 3] else 3

        bidir_cfg = BidirectionalARTransformerConfig(
            state_dim=state_dim_from_meta,
            image_size=84,  # This should match training
            image_channels=image_channels_from_meta,  # This should match training
            forward_steps=32,  # This should match training
            backward_steps=32,
            input_features=input_features,  # Pass features for potential use in config
            # Bidir model defines its own outputs conceptually
            output_features=output_features,
        )
        print(
            f"Using state_dim={bidir_cfg.state_dim}, image_channels={bidir_cfg.image_channels} for BidirectionalARTransformer.")

    bidirectional_ckpt_path = bidirectional_output_dir / "model_final.pth"
    if not bidirectional_ckpt_path.is_file():
        raise OSError(
            f"BidirectionalARTransformer checkpoint not found at {bidirectional_ckpt_path}")

    # Create normalized transformer if possible
    print(f"Loading transformer model from: {bidirectional_ckpt_path}")

    # Create the BidirectionalARTransformer model (without normalizer and unnormalizer)
    print("Loading transformer model manually")
    print("Using default image encoder configuration")

    transformer_model = BidirectionalARTransformer(
        config=bidir_cfg,
        state_key="observation.state",
        image_key="observation.image" if metadata.camera_keys else None
    )

    checkpoint_bidir = torch.load(
        bidirectional_ckpt_path, map_location="cpu")
    model_state_dict_bidir = checkpoint_bidir.get(
        "model_state_dict", checkpoint_bidir)
    # Use non-strict loading to handle architecture differences in the image_encoder
    transformer_model.load_state_dict(model_state_dict_bidir)
    print("Loaded transformer model with strict=False to handle architectural changes")

    transformer_model.eval()
    transformer_model.to(device)

    # --- Load Inverse Dynamics Model (MlpInvDynamic) ---
    invdyn_o_dim = metadata.features["observation.state"]["shape"][-1]
    invdyn_a_dim = metadata.features["action"]["shape"][-1]
    # Use inv_dyn_hidden_dim from the state diffusion config if available, or a default
    invdyn_hidden_dim = 512

    inv_dyn_model = MlpInvDynamic(
        # MlpInvDynamic expects o_dim per state, so if input is s_t, s_{t+1}, it's 2*o_dim internally
        o_dim=invdyn_o_dim,
        a_dim=invdyn_a_dim,
        hidden_dim=invdyn_hidden_dim
    )
    # Try primary checkpoint first, fall back to alternative
    invdyn_ckpt_path = invdyn_output_dir / "invdyn_final.pth"
    if not invdyn_ckpt_path.is_file():
        invdyn_ckpt_path = invdyn_output_dir / "invdyn_model.pth"
        if not invdyn_ckpt_path.is_file():
            raise OSError(
                f"Inverse Dynamics checkpoint not found in {invdyn_output_dir}")

    print(f"Loading Inverse Dynamics model from: {invdyn_ckpt_path}")
    checkpoint_invdyn = torch.load(invdyn_ckpt_path, map_location="cpu")
    model_state_dict_invdyn = checkpoint_invdyn.get(
        "model_state_dict", checkpoint_invdyn)
    inv_dyn_model.load_state_dict(model_state_dict_invdyn)
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    combined_policy = BidirectionalRTDiffusionPolicy(
        bidirectional_transformer=transformer_model,
        inverse_dynamics_model=inv_dyn_model,
        all_dataset_features=metadata.features,  # MODIFICATION: Pass all feature specs
        n_obs_steps=3,
        dataset_stats=processed_dataset_stats,
        output_features=output_features,
    )

    # --- Environment Setup ---
    env = gym.make(
        "gym_pusht/PushT-v0",
        obs_type="pixels_agent_pos",
        max_episode_steps=500,
    )

    # --- Evaluation Loop ---
    numpy_observation, info = env.reset(seed=42)
    rewards = []
    frames = []

    initial_frame_render = env.render()
    # env.render() might return None or list
    if isinstance(initial_frame_render, numpy.ndarray):
        frames.append(initial_frame_render.astype(numpy.uint8))

    step = 0
    done = False

    print("Starting evaluation rollout with 3-stage pipeline...")
    while not done:
        state = torch.from_numpy(numpy_observation["agent_pos"])
        image = torch.from_numpy(numpy_observation["pixels"])

        state = state.to(torch.float32)
        image = image.to(torch.float32) / 255
        image = image.permute(2, 0, 1)

        state = state.to(device, non_blocking=True)
        image = image.to(device, non_blocking=True)

        state = state.unsqueeze(0)  # Add batch dimension
        image = image.unsqueeze(0)  # Add batch dimension

        observation_for_policy = {
            "observation.state": state,
            "observation.image": image,
        }

        with torch.inference_mode():
            action = combined_policy.select_action(observation_for_policy)

        predicted_img = combined_policy.latest_predicted_image
        if predicted_img is not None:
            print(
                f"Predicted image shape before processing: {predicted_img.shape}")
            print(f"Predicted image data type: {predicted_img.dtype}")

            # First make sure we have the right dimensions for an image
            predicted_img = predicted_img.squeeze(0).cpu().numpy()
            print(
                f"Predicted image shape after squeeze: {predicted_img.shape}")

            # Check if we need to transpose dimensions to get HWC format
            # Image should be in format (H, W, C) for saving
            if len(predicted_img.shape) == 3 and predicted_img.shape[0] in [1, 3]:
                # Image is likely in CHW format, convert to HWC
                predicted_img = numpy.transpose(predicted_img, (1, 2, 0))
                print(f"Transposed image shape: {predicted_img.shape}")

            # Convert from 0-1 range to 0-255 range for proper image saving
            predicted_img = (predicted_img * 255).astype(numpy.uint8)

            try:
                # Save the predicted image
                imageio.imwrite(
                    str(predicted_image_dir / f"predicted_image_{step}.png"), predicted_img)
                print(f"Successfully saved predicted image for step {step}")
            except Exception as e:
                print(
                    f"Error saving image: {e}, Image shape: {predicted_img.shape}")

        # Make sure action is on CPU before converting to numpy
        numpy_action = action.squeeze(0).cpu().numpy()
        numpy_observation, reward, terminated, truncated, info = env.step(
            numpy_action)

        print(
            f"Step: {step}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
        rewards.append(reward)

        rendered_frame = env.render()
        if isinstance(rendered_frame, numpy.ndarray):
            frames.append(rendered_frame.astype(numpy.uint8))

        done = terminated or truncated
        step += 1

    print(f"Episode ended after {step} steps.")
    total_reward = sum(rewards)
    print(f"Total reward: {total_reward}")
    if terminated and not truncated:
        print("Success!")
    else:
        print("Failure or Timed Out!")

    fps = env.metadata.get("render_fps", 30)
    video_path = output_directory / "rollout_3stage.mp4"
    if frames:  # Ensure frames list is not empty
        try:
            # Added macro_block_size for some codecs
            imageio.mimsave(str(video_path), frames,
                            fps=fps, macro_block_size=1)
            print(f"Video of the evaluation is available in '{video_path}'.")
        except Exception as e:
            print(
                f"Error saving video: {e}. Frames might be empty or have inconsistent shapes.")
    else:
        print("No frames recorded for video.")

    env.close()


if __name__ == "__main__":
    main()
