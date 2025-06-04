from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
# CLDiffPhyConModel will be used as the State Diffusion Model
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.predictor.bidirectional_autoregressive_transformer import (
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
    bidirectional_output_dir = Path("outputs/train/bidirectional_transformer")

    state_diffusion_output_dir = Path(
        "outputs/train/rtdiffusion_state_predictor")
    invdyn_output_dir = Path("outputs/train/invdyn_only")

    output_directory = Path("outputs/eval/bidirectional_rtdiffusion_3stage")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Dataset Metadata for normalization statistics ---
    print("Loading dataset metadata for normalization...")
    metadata = LeRobotDatasetMetadata("lerobot/pusht")
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
            image_size=96,  # This should match training
            image_channels=image_channels_from_meta,  # This should match training
            forward_steps=16,  # This should match training
            backward_steps=16,
            input_features=metadata.features,  # Pass features for potential use in config
            output_features={},  # Bidir model defines its own outputs conceptually
        )
        print(
            f"Using state_dim={bidir_cfg.state_dim}, image_channels={bidir_cfg.image_channels} for BidirectionalARTransformer.")

    bidirectional_ckpt_path = bidirectional_output_dir / "model_final.pth"
    if not bidirectional_ckpt_path.is_file():
        raise OSError(
            f"BidirectionalARTransformer checkpoint not found at {bidirectional_ckpt_path}")

    # Create normalized transformer if possible
    print(f"Loading transformer model from: {bidirectional_ckpt_path}")

    # Get features for normalization
    # Convert raw feature dictionaries to PolicyFeature objects
    policy_features = dataset_to_policy_features(metadata.features)

    input_features = {
        "observation.state": policy_features["observation.state"],
        "observation.image": policy_features["observation.image"]
    }

    output_features = {
        "action": policy_features["action"]
    }

    norm_mapping = {}
    for key, value in bidir_cfg.normalization_mapping.items():
        # Convert string values to NormalizationMode enum
        if isinstance(value, str):
            norm_mapping[key] = NormalizationMode(value)
        else:
            norm_mapping[key] = value

    unnormalize_action_output = Unnormalize(
        # Use all features for unnormalization
        output_features,
        norm_mapping,
        processed_dataset_stats
    )

    # Create the BidirectionalARTransformer model (without normalizer and unnormalizer)
    print("Loading transformer model manually")
    # Force disable diffusion encoder to ensure compatibility with trained weights
    bidir_cfg.use_diffusion_encoder = True
    print("Explicitly setting use_diffusion_encoder=False to match training configuration")

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

    # --- Load State Prediction Diffusion Model (CLDiffPhyConModel) ---
    state_diffusion_config_json_path = state_diffusion_output_dir / "config.json"
    if not state_diffusion_config_json_path.is_file():
        raise OSError(
            f"State Diffusion config JSON not found at {state_diffusion_config_json_path}")
    print(
        f"Loading State Diffusion configuration from directory: {state_diffusion_output_dir}")
    state_diff_cfg = DiffusionConfig.from_pretrained(
        state_diffusion_output_dir)

    # CRITICAL CHECK: Ensure the loaded config is for state prediction
    if not state_diff_cfg.interpolate_state:
        print(f"CRITICAL WARNING: State Diffusion model config at {state_diffusion_output_dir} "
              "has 'interpolate_state: False'. This model might be trained for ACTION prediction, "
              "not state prediction as required for this pipeline stage.")

    state_diffusion_model = CLDiffPhyConModel(
        config=state_diff_cfg,
        dataset_stats=metadata.stats
    )

    state_diff_ckpt_path = state_diffusion_output_dir / "model_final.pth"
    if not state_diff_ckpt_path.is_file():
        raise OSError(
            f"State Diffusion checkpoint not found at {state_diff_ckpt_path}")

    print(f"Loading State Diffusion model from: {state_diff_ckpt_path}")
    checkpoint_statediff = torch.load(state_diff_ckpt_path, map_location="cpu")
    model_state_dict_statediff = checkpoint_statediff.get(
        "model_state_dict", checkpoint_statediff)

    state_diffusion_model.load_state_dict(
        model_state_dict_statediff, strict=False)
    state_diffusion_model.eval()
    state_diffusion_model.to(device)

    # --- Load Inverse Dynamics Model (MlpInvDynamic) ---
    invdyn_o_dim = metadata.features["observation.state"]["shape"][-1]
    invdyn_a_dim = metadata.features["action"]["shape"][-1]
    # Use inv_dyn_hidden_dim from the state diffusion config if available, or a default
    invdyn_hidden_dim = getattr(state_diff_cfg, 'inv_dyn_hidden_dim', 512)

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
        state_diffusion_model=state_diffusion_model,
        inverse_dynamics_model=inv_dyn_model,
        all_dataset_features=metadata.features,  # MODIFICATION: Pass all feature specs
        n_obs_steps=state_diff_cfg.n_obs_steps,

        dataset_stats=processed_dataset_stats,
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
        state_np = numpy_observation["agent_pos"].astype(
            numpy.float32)  # [StateDim]
        image_np = numpy_observation["pixels"]  # [H,W,C] uint8

        # Policy expects BCHW float for image, Batch dim for state
        current_state_tensor = torch.from_numpy(
            state_np).unsqueeze(0)  # Add batch dim
        # image_np is HWC uint8. BidirectionalRTDiffusionPolicy._normalize_observation handles conversion
        current_image_tensor_for_policy = torch.from_numpy(
            image_np).unsqueeze(0)  # Add batch dim, still HWC uint8

        observation_for_policy = {
            "observation.state": current_state_tensor,
            "observation.image": current_image_tensor_for_policy,
        }

        if step == 0:
            combined_policy.reset()

        with torch.inference_mode():
            # The select_action method already returns unnormalized actions from _get_next_action
            action = combined_policy.select_action(observation_for_policy)

        act = {}
        act['action'] = action[0]  # Action is already unnormalized
        action = unnormalize_action_output(act)

        unnorm_action = action["action"]

        # Make sure action is on CPU before converting to numpy
        numpy_action = unnorm_action.squeeze(0).cpu().numpy()
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
