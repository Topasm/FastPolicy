import torch
from pathlib import Path
import safetensors.torch  # Import safetensors for saving stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
# Import the diffusion model directly
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from lerobot.configs.types import FeatureType


def main():
    output_directory = Path("outputs/train/diffusion_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 500  # Adjust as needed
    log_freq = 10
    save_freq = 500  # Frequency to save checkpoints

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features needed for diffusion model conditioning and target
    input_features = {
        "observation.state": features["observation.state"],
        # Add image/env features if used by the config
        "observation.image": features["observation.image"],
    }
    # Diffusion target is state, but config needs action in output_features too
    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]  # Add action feature here
    }

    cfg = DiffusionConfig(
        input_features=input_features,
        # Pass the updated output_features
        output_features=output_features,
        predict_state=True,  # Must be true for state prediction
    )

    # --- Model ---
    diffusion_model = MyDiffusionModel(cfg)
    diffusion_model.train()
    diffusion_model.to(device)

    # --- Normalization ---
    # Normalize inputs (obs history) and targets (future states) using the same stats
    normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)
    normalize_targets = Normalize(
        {"observation.state": cfg.output_features["observation.state"]}, cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    # Need obs history (n_obs_steps) and future states (horizon)
    # state_delta_indices loads from 1-n_obs up to H-1. We need up to H for the target.
    # Let's define specific timestamps needed.
    # -1 to 16 for n_obs=2, H=16
    diffusion_state_indices = list(range(1 - cfg.n_obs_steps, cfg.horizon + 1))
    delta_timestamps = {
        # -1, 0
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        # -1 to 16
        "observation.state": [i / dataset_metadata.fps for i in diffusion_state_indices],
        # Action not strictly needed, but include for padding mask if used
        # 0 to 15
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Diffusion Model Training...")
    while not done:
        # batch contains state (-1..16), image (-1, 0), action (0..15)
        for batch in dataloader:

            # --- Normalize the batch (on CPU) ---
            # Use normalize_inputs, assuming it handles all relevant keys based on cfg.input_features
            norm_batch = normalize_inputs(batch)

            # Add padding mask back if it exists in the original batch
            if "action_is_pad" in batch:
                norm_batch["action_is_pad"] = batch["action_is_pad"]

            # --- Move normalized batch to GPU ---
            norm_batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in norm_batch.items()}

            # --- Compute Loss ---
            # Pass the entire normalized batch. compute_diffusion_loss will handle slicing.
            loss = diffusion_model.compute_diffusion_loss(norm_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"diffusion_step_{step}.pth"
                torch.save(diffusion_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "diffusion_final.pth"
    torch.save(diffusion_model.state_dict(), final_path)
    print(f"Training finished. Final diffusion model saved to: {final_path}")

    # --- Save Config and Stats ---
    cfg.save_pretrained(output_directory)
    # Filter stats to include only tensors
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items()}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
