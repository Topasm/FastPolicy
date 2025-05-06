import torch
from pathlib import Path
import torch.nn.functional as F
import safetensors.torch  # Import safetensors for saving stats
import einops
import random  # For negative sampling
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
# Import the new multimodal scorer model and config
from model.critic.multimodal_scorer import MultimodalTrajectoryScorer, MultimodalScorerConfig
# Import diffusion components to get image encoder and config details
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.diffusion_modules import DiffusionRgbEncoder


def main():
    output_directory = Path(
        "outputs/train/multimodal_critic_droid")  # New output dir
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 10000  # Adjust as needed
    log_freq = 10
    save_freq = 500  # Frequency to save checkpoints
    batch_size = 32  # Adjust batch size

    # --- Dataset and Config Setup ---
    # Use the DROID dataset
    dataset_repo_id = "cadene/droid_1.0.1"
    print(f"Loading dataset metadata for: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)
    print("Dataset metadata loaded.")

    # Define the image key to use (choose one camera)
    image_key = "observation.images.wrist_left"
    if image_key not in features:
        # Fallback or error if the chosen key isn't present
        available_img_keys = [
            k for k in features if k.startswith("observation.images")]
        if not available_img_keys:
            raise KeyError(
                "No 'observation.images.*' keys found in dataset features.")
        image_key = available_img_keys[0]
        print(
            f"Warning: '{image_key}' not found, using '{image_key}' instead.")

    # Use DiffusionConfig temporarily to get image encoder details and other params
    temp_diffusion_cfg = DiffusionConfig(
        # Use the chosen image key
        input_features={image_key: features[image_key]},
        # Use action shape from the new dataset
        output_features={"action": features["action"]}
    )

    # --- Instantiate Image Encoder ---
    image_encoder = DiffusionRgbEncoder(temp_diffusion_cfg).to(device)
    image_encoder.eval()
    image_feature_dim = temp_diffusion_cfg.transformer_dim

    # --- Multimodal Scorer Config ---
    # Use state and action shapes from the DROID dataset
    scorer_cfg = MultimodalScorerConfig(
        # Should be 8 for DROID
        state_dim=features["observation.state"].shape[0],
        image_feature_dim=image_feature_dim,
        max_state_len=temp_diffusion_cfg.horizon,
        max_image_len=temp_diffusion_cfg.n_obs_steps,
    )
    print(
        f"Scorer Config: State Dim={scorer_cfg.state_dim}, Image Feature Dim={scorer_cfg.image_feature_dim}")

    # --- Model ---
    critic_model = MultimodalTrajectoryScorer(scorer_cfg)
    critic_model.train()
    critic_model.to(device)

    # --- Compile Model (PyTorch 2.x+) ---
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            print("Compiling the critic model...")
            critic_model = torch.compile(critic_model, mode="reduce-overhead")
            print("Model compiled.")
        else:
            print("torch.compile not available. Skipping model compilation.")
    except Exception as e:
        print(f"Model compilation failed: {e}")

    # --- Normalization ---
    # Use stats from the DROID dataset
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, temp_diffusion_cfg.normalization_mapping, dataset_metadata.stats)

    # --- Dataset ---
    state_indices = list(range(temp_diffusion_cfg.n_obs_steps,
                         temp_diffusion_cfg.n_obs_steps + scorer_cfg.max_state_len))
    image_indices = list(range(0, scorer_cfg.max_image_len))
    lang_instruction_idx = 0

    delta_timestamps = {
        # Load image history using the chosen key
        image_key: [i / dataset_metadata.fps for i in image_indices],
        # Load future state sequence
        "observation.state": [i / dataset_metadata.fps for i in state_indices],
        # Load language instruction (assuming it's constant for the relevant window)
        "language_instruction": [lang_instruction_idx / dataset_metadata.fps],
        # Load actions corresponding to the state sequence for padding mask
        "action": [i / dataset_metadata.fps for i in range(scorer_cfg.max_state_len)],
    }
    print("Initializing dataset...")
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)
    print("Dataset initialized.")

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        critic_model.parameters(), lr=temp_diffusion_cfg.optimizer_lr)
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Loss Function ---
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Multimodal Critic Model Training...")
    while not done:
        for batch in dataloader:
            # --- Filter Batch for Valid Language Instructions ---
            valid_indices = [i for i, lang in enumerate(
                batch["language_instruction"]) if isinstance(lang, str) and lang.strip()]
            if not valid_indices:
                print("Warning: No valid language instructions in this batch, skipping.")
                continue  # Skip batch if no valid instructions

            # Create a filtered batch
            filtered_batch = {}
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    filtered_batch[key] = value[valid_indices]
                elif isinstance(value, list):
                    filtered_batch[key] = [value[i] for i in valid_indices]
                else:
                    filtered_batch[key] = value

            B_filtered = len(valid_indices)
            lang_instructions_filtered = filtered_batch["language_instruction"]

            # --- Prepare Positive Examples ---
            state_batch_pos = {
                "observation.state": filtered_batch["observation.state"]}
            # Use the chosen image key
            image_batch_pos = {image_key: filtered_batch[image_key]}

            # Normalize State (Positive)
            norm_state_batch_pos = normalize_state(state_batch_pos)
            state_sequence_pos = norm_state_batch_pos["observation.state"].to(
                device)  # (B_filtered, T_state, D_state)

            # Extract Image Features (Positive)
            with torch.no_grad():
                images_input_pos = image_batch_pos[image_key].to(
                    device)  # (B_filtered, T_img, C, H, W)
                Bf, T_img = images_input_pos.shape[:2]
                images_reshaped_pos = einops.rearrange(
                    images_input_pos, "b t c h w -> (b t) c h w")
                image_features_flat_pos = image_encoder(
                    images_reshaped_pos)  # (Bf*T_img, D_img_feat)
                image_features_pos = einops.rearrange(
                    image_features_flat_pos, "(b t) d -> b t d", b=Bf, t=T_img
                )  # (Bf, T_img, D_img_feat)

            # Prepare Padding Mask (Positive) - Assuming 'action_is_pad' exists
            state_padding_mask_pos = None
            if "action_is_pad" in filtered_batch:
                pad_mask_key = "action_is_pad"
                if filtered_batch[pad_mask_key].shape[1] >= scorer_cfg.max_state_len:
                    state_padding_mask_pos = filtered_batch[pad_mask_key][:, :scorer_cfg.max_state_len].to(
                        device)  # (Bf, T_state)

            # --- Create Negative Examples (Shuffle State Sequences) ---
            if B_filtered <= 1:
                print(
                    "Warning: Filtered batch size <= 1, cannot create negative samples. Skipping step.")
                continue

            shuffle_indices = list(range(B_filtered))
            # Ensure at least one element is shuffled
            while all(i == shuffle_indices[i] for i in range(B_filtered)):
                random.shuffle(shuffle_indices)

            # (B_filtered, T_state, D_state)
            state_sequence_neg = state_sequence_pos[shuffle_indices]
            state_padding_mask_neg = state_padding_mask_pos[
                shuffle_indices] if state_padding_mask_pos is not None else None

            # --- Combine Positive and Negative Data ---
            image_features_combined = torch.cat(
                [image_features_pos, image_features_pos], dim=0)  # (2*Bf, T_img, D_img_feat)
            lang_instructions_combined = lang_instructions_filtered + \
                lang_instructions_filtered  # List of length 2*Bf
            state_sequence_combined = torch.cat(
                [state_sequence_pos, state_sequence_neg], dim=0)  # (2*Bf, T_state, D_state)

            state_padding_mask_combined = None
            if state_padding_mask_pos is not None:
                state_padding_mask_combined = torch.cat(
                    [state_padding_mask_pos, state_padding_mask_neg], dim=0)  # (2*Bf, T_state)

            labels = torch.cat([
                torch.ones(B_filtered, device=device),
                torch.zeros(B_filtered, device=device)
            ], dim=0).float()  # (2*Bf,)

            # --- Compute Scores for Combined Batch ---
            state_sequence_combined_unsqueezed = state_sequence_combined.unsqueeze(
                1)  # (2*Bf, 1, T_state, D_state)
            state_padding_mask_combined_unsqueezed = state_padding_mask_combined.unsqueeze(
                1) if state_padding_mask_combined is not None else None

            image_padding_mask = None

            predicted_scores_single = critic_model(
                state_sequences=state_sequence_combined_unsqueezed,
                image_features=image_features_combined,
                lang_instruction=lang_instructions_combined,
                state_padding_mask=state_padding_mask_combined_unsqueezed,
                image_padding_mask=image_padding_mask,
            )  # Output shape (2*Bf, 1)

            predicted_scores = predicted_scores_single.squeeze(1)  # (2*Bf,)

            # --- Compute Loss ---
            loss = criterion(predicted_scores, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                preds = (torch.sigmoid(predicted_scores) > 0.5).float()
                acc = (preds == labels).float().mean()
                print(
                    f"Step: {step}/{training_steps} Loss: {loss.item():.4f} Acc: {acc.item():.3f} (Filtered Batch Size: {B_filtered})")

            if step % save_freq == 0 and step > 0:
                ckpt_path = output_directory / \
                    f"multimodal_critic_step_{step}.pth"
                torch.save(critic_model.state_dict(), ckpt_path)
                print(f"Saved checkpoint: {ckpt_path}")

            step += 1
            if step >= training_steps:
                done = True
                break

    # --- Save Final Model ---
    final_path = output_directory / "multimodal_critic_final.pth"
    torch.save(critic_model.state_dict(), final_path)
    print(f"Training finished. Final critic model saved to: {final_path}")

    # --- Save Config and Stats ---
    import json
    config_dict = scorer_cfg.__dict__
    with open(output_directory / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items() if isinstance(v, torch.Tensor)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
