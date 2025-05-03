import torch
from pathlib import Path
import safetensors.torch  # Import safetensors for saving stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
# Import the invdyn model directly
from model.invdynamics.invdyn import MlpInvDynamic
# For state/action dims, horizon etc.
from model.diffusion.configuration_mymodel import DiffusionConfig
# Need MyDiffusionModel only to reuse its compute_invdyn_loss method easily
from model.diffusion.modeling_mymodel import MyDiffusionModel


def main():
    output_directory = Path("outputs/train/invdyn_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 500  # Adjust as needed
    log_freq = 10
    save_freq = 100  # Frequency to save checkpoints

    # --- Dataset and Config Setup ---
    dataset_repo_id = "lerobot/pusht"
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Use DiffusionConfig just to get parameters easily
    # Provide dummy features to satisfy validation and property access
    cfg = DiffusionConfig(
        # Add dummy state feature
        input_features={"observation.state": features["observation.state"]},
        # Keep dummy action feature
        output_features={"action": features["action"]}
    )

    # --- Model ---
    invdyn_model = MlpInvDynamic(
        o_dim=features["observation.state"].shape[0] * 2,  # s_{t-1}, s_t
        a_dim=features["action"].shape[0],
        hidden_dim=cfg.inv_dyn_hidden_dim,
        dropout=0.1,
        use_layernorm=True,
        out_activation=torch.nn.Tanh(),
    )
    invdyn_model.train()
    invdyn_model.to(device)

    # Helper model instance to reuse loss computation logic
    loss_computer = MyDiffusionModel(cfg).to(
        device)  # Only used for loss method

    # --- Normalization ---
    # Normalize states and actions separately for invdyn
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, cfg.normalization_mapping, dataset_metadata.stats)
    normalize_action = Normalize(
        {"action": features["action"]}, cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    # Need states s_{-1} to s_{H-1} and actions a_0 to a_{H-1}
    delta_timestamps = {
        # -1 to 15
        "observation.state": [i / dataset_metadata.fps for i in cfg.state_delta_indices],
        # 0 to 15
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        invdyn_model.parameters(), lr=cfg.optimizer_lr)  # Use same LR for now
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Inverse Dynamics Model Training...")
    while not done:
        for batch in dataloader:
            # Prepare normalized batch for invdyn loss (on CPU)
            invdyn_loss_batch = normalize_state(batch)
            invdyn_loss_batch = normalize_action(invdyn_loss_batch)
            # Add padding mask back (still on CPU)
            invdyn_loss_batch['action_is_pad'] = batch['action_is_pad']

            # Now move the required normalized batch to GPU
            invdyn_loss_batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in invdyn_loss_batch.items()}

            # Compute loss using the helper method, passing the actual model instance
            loss = loss_computer.compute_invdyn_loss(
                invdyn_loss_batch, invdyn_model)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / f"invdyn_step_{step}.pth"
                torch.save(invdyn_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "invdyn_final.pth"
    torch.save(invdyn_model.state_dict(), final_path)
    print(
        f"Training finished. Final inverse dynamics model saved to: {final_path}")

    # --- Save Config and Stats ---
    # Save the config used (even if it's DiffusionConfig for parameters)
    cfg.save_pretrained(output_directory)
    # Filter stats to include only tensors
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items() if isinstance(v, torch.Tensor)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
