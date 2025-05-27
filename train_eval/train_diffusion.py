import torch
from pathlib import Path
import safetensors.torch
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.configuration_mymodel import DiffusionConfig
import numpy


def main():
    output_directory = Path("outputs/train/diffusion_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 20000
    log_freq = 100
    save_freq = 500

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Features needed for diffusion model conditioning and target
    input_features = {
        "observation.state": features["observation.state"],
        "observation.image": features["observation.image"],
    }
    output_features = {
        "observation.state": features["observation.state"],
        "action": features["action"]
    }

    # Simplified configuration without interpolation
    # Fixed horizon of 8 frames for direct state prediction
    horizon = 8
    # We'll use the previous and current states (t-1, t=0) for conditioning
    n_obs_steps = 2

    cfg = DiffusionConfig(
        input_features=input_features,
        output_features=output_features,
        interpolate_state=False,  # Don't use interpolation
        horizon=horizon,
        n_obs_steps=n_obs_steps,
        # These parameters aren't used but set them to avoid defaults
        keyframe_indices=[],
        output_horizon=horizon
    )

    # --- Model ---
    diffusion_model = MyDiffusionModel(cfg)
    diffusion_model.train()
    diffusion_model.to(device)

    # --- Normalization ---
    normalize_inputs = Normalize(
        cfg.input_features, cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    # Define simple indices for the dataset

    # State indices from -1 to 8
    state_range = list(range(-1, 9))  # -1 for t=0, 0-8 for t=1-8

    image_indices = [-1, 0, 8]  # Using only available image frames

    # All state indices for validation
    all_state_indices = state_range.copy()

    print(f"Using state range: {state_range}")
    print(f"Using image indices: {image_indices}")

    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
        "observation.state": [i / dataset_metadata.fps for i in state_range],
    }
    # --- Optimizer & Dataloader ---
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    optimizer = torch.optim.AdamW(
        diffusion_model.parameters(), lr=cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    step = 0
    done = False
    print("Starting Diffusion Model Training...")

    while not done:
        for batch in dataloader:
            # Using fixed horizon with direct state prediction

            # Only print config at the start and when logging
            if step % log_freq == 0:
                print(
                    f"Using fixed horizon={horizon} with direct state prediction")

            # --- Normalize the batch and move to device ---
            norm_batch = normalize_inputs(batch)
            if "action_is_pad" in batch:
                norm_batch["action_is_pad"] = batch["action_is_pad"]
            norm_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                          for k, v in norm_batch.items()}

            # --- Compute Loss ---
            # This will use the _compute_original_diffusion_loss method
            # since interpolate_state=False in our config.
            # It directly trains the model to predict states 1-8 given state 0
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

    # Save dataset statistics
    stats_to_save = {}
    for key, value in dataset_metadata.stats.items():
        if isinstance(value, torch.Tensor) or isinstance(value, numpy.ndarray):
            if isinstance(value, numpy.ndarray):
                value = torch.from_numpy(value)
            stats_to_save[key] = value

    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
