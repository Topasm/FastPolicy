#!/usr/bin/env python

import torch
from pathlib import Path
import numpy as np
import safetensors.torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize, Unnormalize

from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.invdynamics.invdyn import MlpInvDynamic


def evaluate_trajectory_continuity(model_path, num_samples=10):
    """
    Evaluate the continuity of generated trajectories.
    Continuity is a key aspect of CL-DiffPhyCon's performance.
    """
    model_path = Path(model_path)

    # Load config and stats
    config = DiffusionConfig.from_pretrained(model_path)
    with safetensors.safe_open(model_path / "stats.safetensors", framework="pt", device="cpu") as f:
        stats = {k: f.get_tensor(k) for k in f.keys()}

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load models
    diffusion_model = CLDiffPhyConModel(config)
    diff_state_dict = torch.load(
        model_path / "diffusion_final.pth", map_location="cpu")
    diffusion_model.load_state_dict(diff_state_dict)
    diffusion_model.to(device)
    diffusion_model.eval()

    inv_dyn_model = MlpInvDynamic(
        o_dim=config.robot_state_feature.shape[0] * 2,
        a_dim=config.action_feature.shape[0],
        hidden_dim=config.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh(),
    )
    inv_state_dict = torch.load(
        model_path / "invdyn_final.pth", map_location="cpu")
    inv_dyn_model.load_state_dict(inv_state_dict)
    inv_dyn_model.to(device)
    inv_dyn_model.eval()

    # Create dataset
    dataset_repo_id = "lerobot/pusht"  # Same dataset as training
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)

    # State indices from -1 to 8
    state_range = list(range(-1, 9))  # For n_obs_steps=2 and horizon=8
    image_indices = [-1, 0, 8]

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
        "observation.state": [i / dataset_metadata.fps for i in state_range],
        "action": [i / dataset_metadata.fps for i in range(config.horizon)],
    }

    eval_dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps, split="val")

    # Create normalizers
    normalize_inputs = Normalize(
        config.input_features, config.normalization_mapping, stats)
    unnormalize_states = Unnormalize(
        {"observation.state": config.robot_state_feature},
        config.normalization_mapping,
        stats
    )

    # Randomly sample episodes
    episode_indices = np.random.choice(
        len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)

    # Setup plots
    # Plot up to 4 dimensions
    n_dims_to_plot = min(4, config.robot_state_feature.shape[0])
    fig, axes = plt.subplots(num_samples, n_dims_to_plot,
                             figsize=(16, 3*num_samples))

    # Evaluate continuity
    continuity_errors = []

    for sample_idx, episode_idx in enumerate(episode_indices):
        print(f"Evaluating episode {episode_idx}")

        # Get episode data
        episode = eval_dataset[episode_idx]

        # Normalize and move to device
        norm_episode = normalize_inputs(
            {k: v.unsqueeze(0) for k, v in episode.items()})
        norm_episode = {k: v.to(device) for k, v in norm_episode.items()}

        # Extract history for conditioning
        n_obs = config.n_obs_steps
        input_batch = {
            "observation.state": norm_episode["observation.state"][:, :n_obs],
            "observation.image": norm_episode["observation.image"][:, :n_obs] if "observation.image" in norm_episode else None
        }
        input_batch = {k: v for k, v in input_batch.items() if v is not None}

        # State at t=0
        current_state = norm_episode["observation.state"][:, n_obs-1]

        # Generate trajectory with CL-DiffPhyCon
        with torch.no_grad():
            generated_states = diffusion_model.cl_phycon_inference(
                input_batch,
                current_state,
                inv_dyn_model=inv_dyn_model,
                num_samples=1,
                return_trajectory=True
            )

        # Extract ground truth trajectory for comparison
        gt_states = norm_episode["observation.state"][:,
                                                      n_obs:n_obs+config.horizon]

        # Unnormalize for visualization
        gen_states_unnorm = unnormalize_states(
            {"observation.state": generated_states.cpu()}
        )["observation.state"].squeeze(0).numpy()

        gt_states_unnorm = unnormalize_states(
            {"observation.state": gt_states.cpu()}
        )["observation.state"].squeeze(0).numpy()

        # Calculate continuity error (first derivative discontinuity)
        gen_diff = np.diff(gen_states_unnorm, axis=0)
        continuity_error = np.max(np.abs(np.diff(gen_diff, axis=0)))
        continuity_errors.append(continuity_error)

        # Plot state trajectories
        for dim in range(n_dims_to_plot):
            ax = axes[sample_idx, dim] if num_samples > 1 else axes[dim]
            ax.plot(gen_states_unnorm[:, dim], 'b-', label='Generated')
            ax.plot(gt_states_unnorm[:, dim], 'g--', label='Ground Truth')
            ax.set_title(f'State Dimension {dim}')
            ax.legend()

    plt.tight_layout()
    plt.savefig(model_path / "trajectory_evaluation.png")

    # Print continuity statistics
    avg_continuity_error = np.mean(continuity_errors)
    print(f"Average continuity error: {avg_continuity_error:.5f}")

    # Save statistics
    with open(model_path / "eval_metrics.txt", "w") as f:
        f.write(f"Average continuity error: {avg_continuity_error:.5f}\n")

    return avg_continuity_error


if __name__ == "__main__":
    model_path = "outputs/train/clphycon"
    evaluate_trajectory_continuity(model_path, num_samples=5)
