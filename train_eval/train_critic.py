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
import json  # For loading task mappings


def main():

    output_directory = Path(
        "outputs/train/multimodal_critic_droid")  # New output dir
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 1000  # Adjust as needed
    log_freq = 10
    save_freq = 500  # Frequency to save checkpoints
    batch_size = 32  # Adjust batch size

    # --- Dataset and Config Setup ---
    # Use the DROID dataset
    dataset_repo_id = "cadene/droid_1.0.1"
    print(f"Loading dataset metadata for: {dataset_repo_id}")
    dataset_metadata = LeRobotDatasetMetadata(
        dataset_repo_id, root="/media/ahrilab/data/droid_1.0.1/")
    features = dataset_to_policy_features(dataset_metadata.features)
    print("Dataset metadata loaded.")

    # Define the image key to use (choose one camera)
    image_key = "observation.images.exterior_1_left"
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
        # Load task_index (assuming it's constant for the relevant window, using 0-th frame's index)
        "task_index": [lang_instruction_idx / dataset_metadata.fps],
        # Load actions corresponding to the state sequence for padding mask
        "action": [i / dataset_metadata.fps for i in range(scorer_cfg.max_state_len)],
    }
    print("Initializing dataset...")

    # Try a smaller range of episodes that's likely to be within bounds
    dataset = LeRobotDataset(
        dataset_repo_id, root="/media/ahrilab/data/droid_1.0.1/", delta_timestamps=delta_timestamps, episodes=list(range(0, 200)))

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
            # Handle task_index to get language instructions
            task_strings_batch = []
            valid_sample_indices_for_batch = []

            if "task_index" in batch:
                task_indices_tensor = batch["task_index"]
                task_indices_list = []  # Initialize to ensure it's defined

                if isinstance(task_indices_tensor, torch.Tensor):
                    task_indices_list = task_indices_tensor.cpu().flatten().tolist()
                elif isinstance(task_indices_tensor, list):
                    processed_indices = []
                    for ti_val in task_indices_tensor:
                        if isinstance(ti_val, (list, tuple)) and len(ti_val) > 0:
                            try:
                                processed_indices.append(int(ti_val[0]))
                            except ValueError:
                                continue
                        elif isinstance(ti_val, (float, int)):
                            processed_indices.append(int(ti_val))
                        else:
                            # Skip silently as code is working
                            continue
                    task_indices_list = processed_indices
                else:
                    # Skip batch silently
                    continue

                for i, task_idx_val in enumerate(task_indices_list):
                    try:
                        task_idx_int = int(task_idx_val)
                        if task_idx_int in dataset.meta.tasks:
                            task_strings_batch.append(
                                dataset.meta.tasks[task_idx_int])
                            valid_sample_indices_for_batch.append(i)
                    except (KeyError, IndexError, ValueError):
                        continue  # Skip this sample if task_idx is invalid or not found

                if not valid_sample_indices_for_batch:
                    # Skip batch silently
                    continue

                # Create a filtered batch
                filtered_batch = {}
                for key, value in batch.items():
                    if key == "task_index":
                        continue
                    if isinstance(value, torch.Tensor) and value.shape[0] == len(task_indices_list):
                        filtered_batch[key] = value[valid_sample_indices_for_batch]
                    elif isinstance(value, list) and len(value) == len(task_indices_list):
                        filtered_batch[key] = [
                            value[j] for j in valid_sample_indices_for_batch]
                    else:
                        filtered_batch[key] = value

                # This is the key the model expects
                filtered_batch["language_instruction"] = task_strings_batch
                batch = filtered_batch
                B_filtered = len(valid_sample_indices_for_batch)

            else:
                continue

            # --- Prepare Positive Examples ---
            state_batch_pos = {"observation.state": batch["observation.state"]}
            image_batch_pos = {image_key: batch[image_key]}

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
            if "action_is_pad" in batch:
                pad_mask_key = "action_is_pad"
                if batch[pad_mask_key].shape[1] >= scorer_cfg.max_state_len:
                    state_padding_mask_pos = batch[pad_mask_key][:, :scorer_cfg.max_state_len].to(
                        device)  # (Bf, T_state)

            # --- Create Negative Examples (Shuffle State Sequences) ---
            if B_filtered <= 1:
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

            # Combine language instructions
            if "language_instruction" in batch:
                lang_instructions_combined = batch["language_instruction"] + \
                    batch["language_instruction"]  # List of length 2*Bf
            else:
                # Fall back to empty strings if no language instruction is available
                empty_instructions = [""] * B_filtered
                lang_instructions_combined = empty_instructions + empty_instructions

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

            image_padding_mask = None  # Assuming no image padding mask for now

            try:
                predicted_scores_single = critic_model(
                    state_sequences=state_sequence_combined_unsqueezed,
                    image_features=image_features_combined,
                    lang_instruction=lang_instructions_combined,
                    state_padding_mask=state_padding_mask_combined_unsqueezed,
                    image_padding_mask=image_padding_mask,
                )  # Output shape (2*Bf, 1)

            except Exception as e:
                continue

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
