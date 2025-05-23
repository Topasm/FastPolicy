#!/usr/bin/env python
"""
Test script to verify the full interpolation model implementation.
This script tests the model's ability to generate interpolated states
between keyframes and use inverse dynamics to predict actions.
"""

import torch
from pathlib import Path
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.invdynamics.invdyn import MlpInvDynamic


def test_interpolation_model(horizon=16, mode="skip_even"):
    """Test the interpolation model with specified horizon and mode."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Testing interpolation with horizon={horizon}, mode={mode}")

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features needed for diffusion model conditioning and target
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }
    # Diffusion target is state, but config needs action in output_features too
    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]
    }

    # Create configuration - use simple keyframe indices for testing
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        interpolate_state=True,
        horizon=horizon,
        interpolation_mode=mode,
        keyframe_indices=[8],  # Use only one future keyframe for testing
    )

    print("\n=== Configuration ===")
    print(f"Keyframe indices: {cfg.keyframe_delta_indices}")
    print(f"Interpolation target indices: {cfg.interpolation_target_indices}")

    # --- Create models ---
    diffusion_model = MyDiffusionModel(cfg)
    diffusion_model.eval()
    diffusion_model.to(device)

    # Create inverse dynamics model
    state_dim = cfg.robot_state_feature.shape[0]
    action_dim = cfg.action_feature.shape[0]
    inv_dyn_model = MlpInvDynamic(
        o_dim=state_dim * 2,
        a_dim=action_dim,
        hidden_dim=cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh(),
    )
    inv_dyn_model.eval()
    inv_dyn_model.to(device)

    # --- Normalization ---
    normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)
    unnormalize_outputs = Unnormalize(
        {"observation.state": cfg.robot_state_feature},
        cfg.normalization_mapping, dataset_metadata.stats
    )

    # --- Dataset ---
    # For test purposes, get a small subset of data with more limited horizon
    delta_timestamps = {
        # Image history for conditioning: -1, 0
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        # Limited state sequence for testing - just enough for keyframe at 8
        # -0.1s to 0.9s
        "observation.state": [i / dataset_metadata.fps for i in range(-1, 10)],
        # Action sequence
        # 0.0s to 0.7s
        "action": [i / dataset_metadata.fps for i in range(0, 8)],
    }

    print("\n=== Dataset Setup ===")
    print(
        f"State delta timestamps: {[i / dataset_metadata.fps for i in range(-1, 10)]}")

    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True
    )

    # --- Test Model ---
    print("\n=== Testing Model ===")
    batch = next(iter(dataloader))
    print(f"Batch keys: {batch.keys()}")
    print(f"State sequence shape: {batch['observation.state'].shape}")
    print(f"Image sequence shape: {batch['observation.image'].shape}")

    # Normalize batch
    norm_batch = normalize_inputs(batch)

    # Move to device
    norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in norm_batch.items()}

    # Get current state (index corresponding to time 0)
    current_state_idx = cfg.n_obs_steps - 1
    current_state = norm_batch["observation.state"][:, current_state_idx, :]

    print("\n--- Computing Diffusion Loss ---")
    loss = diffusion_model.compute_diffusion_loss(norm_batch)
    print(f"Diffusion loss: {loss.item()}")

    print("\n--- Generating Actions via Inverse Dynamics ---")
    actions = diffusion_model.generate_actions_via_inverse_dynamics(
        norm_batch, current_state, inv_dyn_model
    )

    print(f"Generated actions shape: {actions.shape}")
    print(f"Expected output horizon: {cfg.output_horizon}")

    # Test conditional sampling directly
    print("\n--- Testing Conditional Sampling ---")
    global_cond = diffusion_model._prepare_global_conditioning(norm_batch)
    sampled_states = diffusion_model.conditional_sample(
        batch_size=norm_batch["observation.state"].shape[0],
        global_cond=global_cond
    )

    print(f"Sampled states shape: {sampled_states.shape}")
    print(f"Expected shape: [batch_size, {cfg.output_horizon}, {state_dim}]")

    # Unnormalize states for visualization
    # Make sure the tensors are on the same device
    unnorm_states = unnormalize_outputs(
        {"observation.state": sampled_states.cpu()}
    )["observation.state"]

    print("\n--- Sample States (unnormalized) ---")
    print(unnorm_states[0, :3, :])  # Print first 3 states of first batch item

    print("\n=== Test Complete ===")


def test_all_modes():
    """Test all interpolation modes."""
    horizons = [8, 16, 32]
    modes = ["dense", "skip_even", "sparse"]

    for horizon, mode in zip(horizons, modes):
        print(f"\n\n{'='*50}")
        print(f"TESTING HORIZON {horizon} WITH MODE {mode}")
        print(f"{'='*50}\n")
        test_interpolation_model(horizon, mode)


if __name__ == "__main__":
    test_all_modes()
