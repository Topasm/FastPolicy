import torch
from pathlib import Path
import torch.nn.functional as F
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
# Import the critic model directly
from model.critic.critic_model import CriticScorer
# For state/action dims, horizon etc.
from model.diffusion.configuration_mymodel import DiffusionConfig


def main():
    output_directory = Path("outputs/train/critic_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 5000  # Adjust as needed
    log_freq = 10
    save_freq = 500  # Frequency to save checkpoints

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Use DiffusionConfig just to get parameters easily
    cfg = DiffusionConfig(input_features={}, output_features={})

    # --- Model ---
    critic_model = CriticScorer(
        state_dim=features["observation.state"].shape[0],
        horizon=cfg.horizon,  # Critic scores the full state sequence
        hidden_dim=cfg.critic_hidden_dim,
    )
    critic_model.train()
    critic_model.to(device)

    # --- Normalization ---
    # Critic takes state sequence as input
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    # Need states s_0 to s_{H-1} for critic input
    # Also need action padding mask for dummy target
    critic_state_indices = list(range(0, cfg.horizon))  # 0 to 15
    delta_timestamps = {
        # Need state history for normalization stats consistency if MIN_MAX is used over sequence? Check Normalize logic.
        # Let's load the same state sequence as invdyn for simplicity, then slice.
        # -1 to 15
        "observation.state": [i / dataset_metadata.fps for i in cfg.state_delta_indices],
        # 0 to 15 (for padding mask)
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        critic_model.parameters(), lr=cfg.optimizer_lr)  # Use same LR for now
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Critic Model Training...")
    while not done:
        for batch in dataloader:
            # Keep batch on CPU for normalization

            # Prepare normalized batch for critic loss (on CPU)
            norm_batch = normalize_state(batch)
            # Add padding mask back (still on CPU)
            norm_batch['action_is_pad'] = batch['action_is_pad']

            # Now move the required normalized batch to GPU
            norm_batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in norm_batch.items()}

            # Extract state sequence s_0 to s_{H-1} (now on GPU)
            state_start_idx = cfg.n_obs_steps - 1  # Index for time 0
            state_end_idx = state_start_idx + cfg.horizon  # Index for time H-1 + 1
            state_sequence = norm_batch["observation.state"][:,
                                                             state_start_idx:state_end_idx, :]  # (B, H, D)

            # --- Dummy Target Calculation ---
            # Use the GPU version of the padding mask
            last_step_mask = ~norm_batch["action_is_pad"][:,
                                                          cfg.horizon - 1]  # (B,) True if not padded
            dummy_target = last_step_mask.float().unsqueeze(-1)  # (B, 1)

            # --- Compute Loss ---
            predicted_score = critic_model(
                state_sequence)  # Use forward for training
            # Use MSE loss against the dummy target
            loss = F.mse_loss(predicted_score, dummy_target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"critic_step_{step}.pth"
                torch.save(critic_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "critic_final.pth"
    torch.save(critic_model.state_dict(), final_path)
    print(f"Training finished. Final critic model saved to: {final_path}")


if __name__ == "__main__":
    main()
