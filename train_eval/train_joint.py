import logging
from pathlib import Path

import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
# Import the new policy
from model.diffusion.modeling_jointdiffusion import JointDiffusionPolicy
from model.diffusion.configuration_mymodel import DiffusionConfig  # Reuse config for now

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')


def main():
    # --- Configuration ---
    repo_id = "lerobot/pusht"  # Dataset repository ID
    # Checkpoint directory
    output_directory = Path("outputs/train/joint_policy")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 10000  # Adjust as needed
    log_freq = 100
    save_freq = 1000  # Save checkpoints periodically
    batch_size = 64
    num_workers = 4
    learning_rate = 1e-4

    logging.info(f"Using device: {device}")
    output_directory.mkdir(parents=True, exist_ok=True)
    logging.info(f"Checkpoints will be saved to: {output_directory}")

    # --- Dataset Loading ---
    logging.info(f"Loading dataset metadata from {repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # --- Policy Configuration ---
    # Joint policy requires state prediction
    predict_state_flag = True

    # Define input/output features (must include state for diffusion target and action for inv dyn)
    target_state_key = "observation.state"  # Diffusion predicts future states
    output_features = {
        target_state_key: features[target_state_key],
        # Inv dyn predicts actions, policy outputs actions
        "action": features["action"]
    }
    # Input features include observations needed for conditioning
    input_features = {key: ft for key,
                      ft in features.items() if key != "action"}

    logging.info("Configuring JointDiffusionPolicy")
    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        predict_state=predict_state_flag,
        # Add any specific config overrides here if needed
        # e.g., horizon, n_obs_steps, transformer dims, inv_dyn_hidden_dim
        horizon=16,  # Example
        n_obs_steps=2,  # Example
        inv_dyn_hidden_dim=512,  # Example
        # TODO: Add teacher_forcing flag to config if implemented
    )

    # --- Instantiate Policy ---
    logging.info("Instantiating JointDiffusionPolicy")
    policy = JointDiffusionPolicy(cfg, dataset_stats=dataset_metadata.stats)
    policy.train()  # Set to training mode
    policy.to(device)
    logging.info(
        f"Policy parameters: {sum(p.numel() for p in policy.parameters())}")

    # --- Prepare Dataset and Dataloader ---
    # Define delta_timestamps based on config (ensure state and action are included)
    delta_timestamps = {
        # Past/present observations for conditioning
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        # Need past states for conditioning AND future states for diffusion target
        "observation.state": [i / dataset_metadata.fps for i in range(cfg.observation_delta_indices[0], cfg.target_delta_indices[-1] + 1)],
        # Need future actions for inverse dynamics loss
        # Assuming action indices match horizon
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices]
    }

    logging.info(
        f"Loading dataset {repo_id} with delta_timestamps: {delta_timestamps}")
    dataset = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,  # Important for consistent batch sizes
    )
    logging.info(f"Dataloader created with batch size {batch_size}")

    # --- Optimizer ---
    optimizer = torch.optim.Adam(policy.get_optim_params(), lr=learning_rate)
    logging.info(f"Optimizer: Adam with lr={learning_rate}")

    # --- Training Loop ---
    logging.info("Starting training loop...")
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            # Move batch to device
            batch = {k: (v.to(device) if isinstance(v, torch.Tensor) else v)
                     for k, v in batch.items()}

            # Forward pass
            total_loss, loss_dict = policy.forward(batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            # Optional: Gradient clipping
            # torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            optimizer.step()

            # Logging
            if step % log_freq == 0:
                log_msg = f"step: {step} total_loss: {total_loss.item():.4f}"
                if loss_dict:
                    log_msg += f" | diffusion_loss: {loss_dict.get('diffusion_loss', torch.tensor(0.0)).item():.4f}"
                    log_msg += f" | inv_dyn_loss: {loss_dict.get('inv_dyn_loss', torch.tensor(0.0)).item():.4f}"
                logging.info(log_msg)

            # Save checkpoint
            if step % save_freq == 0 and step > 0:
                checkpoint_dir = output_directory / f"step_{step}"
                logging.info(f"Saving checkpoint to {checkpoint_dir}")
                policy.save_pretrained(checkpoint_dir)

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Checkpoint ---
    logging.info(
        f"Training finished after {step} steps. Saving final checkpoint.")
    final_checkpoint_dir = output_directory / "final"
    policy.save_pretrained(final_checkpoint_dir)
    logging.info(f"Final checkpoint saved to {final_checkpoint_dir}")


if __name__ == "__main__":
    main()
