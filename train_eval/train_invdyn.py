import torch
import torch.nn.functional as F
from pathlib import Path
import safetensors.torch  # Import safetensors for saving stats
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
# Import the invdyn models directly
from model.invdynamics.invdyn import MlpInvDynamic, SeqInvDynamic
# For state/action dims, horizon etc.
from model.diffusion.configuration_mymodel import DiffusionConfig
# Need MyDiffusionModel only to reuse its compute_invdyn_loss method easily
from model.diffusion.modeling_mymodel import MyDiffusionModel


def main():
    output_directory = Path("outputs/train/invdyn_only")
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    training_steps = 5000  # Adjust as needed
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

    # --- Model Configuration ---
    # Set sequence length for inverse dynamics (number of historical states to use)
    seq_length = 3  # Use s_{t-2}, s_{t-1}, s_t for predicting action

    # Choose which model type to use
    use_seq_model = True  # Set to True for SeqInvDynamic, False for MlpInvDynamic

    if use_seq_model:
        # Use the GRU-based sequential model
        invdyn_model = SeqInvDynamic(
            state_dim=features["observation.state"].shape[0],
            action_dim=features["action"].shape[0],
            hidden_dim=cfg.inv_dyn_hidden_dim,
            n_layers=2,  # Can increase for more capacity
            dropout=0.1,
            out_activation=torch.nn.Tanh()  # Use same activation as MlpInvDynamic
        )
    else:
        # Original MLP model (for comparison)
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
    # Set sequence length for the dataset (how many historical states to use)
    seq_length = 3  # Using s_{t-2}, s_{t-1}, s_t for predicting action

    # Create custom indices for longer state history
    # For 3-step history: we need s_{t-2}, s_{t-1}, s_t for each action a_t
    custom_state_indices = []
    for i in range(-seq_length+1, len(cfg.action_delta_indices)+1):
        custom_state_indices.append(i)

    delta_timestamps = {
        # Extended state history (-seq_length+1 to H)
        "observation.state": [i / dataset_metadata.fps for i in custom_state_indices],
        # Original action sequence (0 to H-1)
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }

    print(f"Using sequence length {seq_length} for inverse dynamics")
    print(f"State indices: {custom_state_indices}")
    print(f"Action indices: {cfg.action_delta_indices}")

    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        invdyn_model.parameters(), lr=cfg.optimizer_lr)  # Use same LR for now
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=64, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Custom loss function for sequential model ---
    def compute_seq_invdyn_loss(batch, model, seq_length):
        """
        Custom loss function for sequential inverse dynamics model

        Args:
            batch: Dictionary containing normalized state and action sequences
            model: The inverse dynamics model
            seq_length: How many states to use for each prediction
        """
        # Get state and action tensors
        states = batch["observation.state"]  # Shape: [B, S, D]
        actions = batch["action"]  # Shape: [B, A, D_a]

        # Create sequences of states for each action
        total_actions = actions.shape[1]
        total_loss = 0.0

        # For each action at position i, we use states from i to i+seq_length
        for i in range(total_actions):
            # Get the sequence of states [s_{i}, s_{i+1}, ..., s_{i+seq_length-1}]
            if i + seq_length <= states.shape[1]:
                state_seq = states[:, i:i+seq_length, :]  # [B, seq_length, D]
                target_action = actions[:, i, :]  # [B, D_a]

                # Run the model to predict the action
                if isinstance(model, SeqInvDynamic):
                    # For sequential model, feed the whole sequence
                    pred_action = model(state_seq)
                    # Get the output at the last position
                    pred_action = pred_action[:, -1, :]
                else:
                    # For MLP model, concatenate last two states
                    last_two_states = torch.cat(
                        [state_seq[:, -2, :], state_seq[:, -1, :]], dim=1)
                    pred_action = model(last_two_states)

                # Compute MSE loss
                action_loss = F.mse_loss(pred_action, target_action)
                total_loss += action_loss

        return total_loss / total_actions if total_actions > 0 else torch.tensor(0.0, device=device)

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Inverse Dynamics Model Training...")
    print(
        f"Using {'Sequential' if use_seq_model else 'MLP'} model with {seq_length} state context length")

    while not done:
        for batch in dataloader:
            # Prepare normalized batch (on CPU)
            invdyn_loss_batch = normalize_state(batch)
            invdyn_loss_batch = normalize_action(invdyn_loss_batch)
            # Add padding mask back (still on CPU)
            invdyn_loss_batch['action_is_pad'] = batch['action_is_pad']

            # Move the required normalized batch to GPU
            invdyn_loss_batch = {k: v.to(device) if isinstance(
                v, torch.Tensor) else v for k, v in invdyn_loss_batch.items()}

            # Use custom loss for sequential model, or compute_invdyn_loss for MLP
            if use_seq_model:
                # Use our custom sequential loss function
                loss = compute_seq_invdyn_loss(
                    invdyn_loss_batch, invdyn_model, seq_length)
            else:
                # Original loss calculation for MLP model
                loss = loss_computer.compute_invdyn_loss(
                    invdyn_loss_batch, invdyn_model)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"Step: {step}/{training_steps} Loss: {loss.item():.4f}")

            if step % save_freq == 0 and step > 0:
                model_type = "seq" if use_seq_model else "mlp"
                ckpt_path = output_directory / \
                    f"invdyn_{model_type}_seq{seq_length}_step_{step}.pth"
                torch.save(invdyn_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    model_type = "seq" if use_seq_model else "mlp"
    final_path = output_directory / \
        f"invdyn_{model_type}_seq{seq_length}_final.pth"
    torch.save(invdyn_model.state_dict(), final_path)
    print(
        f"Training finished. Final inverse dynamics model ({model_type.upper()} with sequence length {seq_length}) saved to: {final_path}")

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
