import torch
from pathlib import Path
import torch.nn.functional as F
import safetensors.torch  # Import safetensors for saving stats
import einops
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
        "outputs/train/multimodal_critic")  # New output dir
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    training_steps = 10000  # Adjust as needed
    log_freq = 10
    save_freq = 500  # Frequency to save checkpoints
    batch_size = 32  # Adjust batch size

    # --- Dataset and Config Setup ---
    # NOTE: Assumes a dataset providing language instructions and matching labels.
    # Replace "lerobot/pusht" with your actual dataset ID.
    # >>>>>>>> IMPORTANT: REPLACE THIS WITH YOUR DATASET REPO ID <<<<<<<<<<
    # NEEDS TO BE REPLACED with your language-conditioned dataset
    dataset_repo_id = "lerobot/pusht"
    # >>>>>>>> IMPORTANT: REPLACE THIS WITH YOUR DATASET REPO ID <<<<<<<<<<
    dataset_metadata = LeRobotDatasetMetadata(dataset_repo_id)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Use DiffusionConfig temporarily to get image encoder details and other params
    temp_diffusion_cfg = DiffusionConfig(
        # Need image for encoder
        input_features={"observation.image": features["observation.image"]},
        # Add dummy action feature to satisfy validation
        output_features={"action": features["action"]}
    )

    # --- Instantiate Image Encoder ---
    # Load a pre-trained image encoder or initialize a new one
    # Assuming DiffusionRgbEncoder is used
    image_encoder = DiffusionRgbEncoder(temp_diffusion_cfg).to(device)
    # Optionally load pre-trained weights for the image encoder here
    # image_encoder.load_state_dict(...)
    image_encoder.eval()  # Use encoder in eval mode for feature extraction
    image_feature_dim = temp_diffusion_cfg.transformer_dim  # Get output dim from config

    # --- Multimodal Scorer Config ---
    # Ensure state_dim and image_feature_dim are correctly set
    scorer_cfg = MultimodalScorerConfig(
        state_dim=features["observation.state"].shape[0],
        image_feature_dim=image_feature_dim,
        max_state_len=temp_diffusion_cfg.horizon,  # Use horizon from diffusion config
        # Use n_obs_steps from diffusion config
        max_image_len=temp_diffusion_cfg.n_obs_steps,
        # num_layers=8 # Can override default number of layers here if needed
    )

    # --- Model ---
    critic_model = MultimodalTrajectoryScorer(scorer_cfg)
    critic_model.train()
    critic_model.to(device)

    # --- Compile Model (PyTorch 2.x+) ---
    # Optional: Compile the model for potential speedup
    try:
        # Check if torch.compile is available (PyTorch 2.0+)
        if hasattr(torch, "compile"):
            print("Compiling the critic model...")
            # You can experiment with different modes like 'max-autotune'
            critic_model = torch.compile(critic_model, mode="reduce-overhead")
            print("Model compiled.")
        else:
            print("torch.compile not available. Skipping model compilation.")
    except Exception as e:
        print(f"Model compilation failed: {e}")

    # --- Normalization ---
    # Normalize states and potentially image features (if needed, often handled by encoder)
    normalize_state = Normalize(
        {"observation.state": features["observation.state"]}, temp_diffusion_cfg.normalization_mapping, dataset_metadata.stats)
    # Image normalization might be part of the encoder or done separately if needed

    # --- Dataset ---
    # Need image history (T_image), future states (T_state), language, and labels
    # Adjust indices based on MultimodalScorerConfig max lengths
    state_indices = list(range(temp_diffusion_cfg.n_obs_steps, temp_diffusion_cfg.n_obs_steps +
                         scorer_cfg.max_state_len))  # Future states s_1 to s_H
    # Image history t=0 to t=n_obs-1
    image_indices = list(range(0, scorer_cfg.max_image_len))

    delta_timestamps = {
        # Load image history
        "observation.image": [i / dataset_metadata.fps for i in image_indices],
        # Load future state sequence
        "observation.state": [i / dataset_metadata.fps for i in state_indices],
        # Add keys required by your specific dataset for language and labels
        # "instruction": ??? # Depends on how language is stored
        # "is_correct_sequence": ??? # Depends on how label is stored
        # Include action padding mask if used for state padding
        # For padding mask
        "action": [i / dataset_metadata.fps for i in range(scorer_cfg.max_state_len)],
    }
    # Ensure your dataset class loads 'instruction' (str) and 'is_correct_sequence' (float/long)
    # You might need a custom Dataset class if LeRobotDataset doesn't handle your specific format.
    dataset = LeRobotDataset(
        dataset_repo_id, delta_timestamps=delta_timestamps)

    # --- Optimizer & Dataloader ---
    optimizer = torch.optim.AdamW(
        critic_model.parameters(), lr=temp_diffusion_cfg.optimizer_lr)  # Use same LR for now
    dataloader = torch.utils.data.DataLoader(
        dataset, num_workers=4, batch_size=batch_size, shuffle=True, pin_memory=device.type != "cpu", drop_last=True
    )

    # --- Loss Function ---
    # Binary Cross Entropy with Logits for binary classification (correct/incorrect sequence)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --- Training Loop ---
    step = 0
    done = False
    print("Starting Multimodal Critic Model Training...")
    while not done:
        for batch in dataloader:
            # === VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV ===
            # === PLACEHOLDER: Load language and labels                     ===
            # === You MUST modify this section to load data from YOUR dataset ===
            # === The current dataset 'lerobot/pusht' does NOT have these keys ===
            # === VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV ===
            # Replace these lines with actual loading from your batch
            if "instruction" not in batch or "is_correct_sequence" not in batch:
                # This error means your dataset (or the batch loading) is missing the required keys.
                # 1. Make sure `dataset_repo_id` points to your correct dataset.
                # 2. Ensure your dataset loader provides 'instruction' and 'is_correct_sequence'.
                raise KeyError(
                    "Batch must contain 'instruction' (list[str]) and 'is_correct_sequence' (Tensor[float]). "
                    "Please update the dataset loading logic for your specific language-conditioned dataset."
                )
            lang_instructions = batch["instruction"]  # List of strings [B]
            # Target labels (B,), 1.0 for correct, 0.0 for incorrect
            labels = batch["is_correct_sequence"].float().to(device)
            # === ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ===
            # === END OF PLACEHOLDER SECTION                                 ===
            # === ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ ===

            # Keep state/image on CPU for normalization/feature extraction
            state_batch = {"observation.state": batch["observation.state"]}
            # (B, T_img, C, H, W)
            image_batch = {"observation.image": batch["observation.image"]}

            # --- Normalize State ---
            norm_state_batch = normalize_state(state_batch)
            state_sequence = norm_state_batch["observation.state"].to(
                device)  # (B, T_state, D_state)

            # --- Extract Image Features ---
            with torch.no_grad():
                images_input = image_batch["observation.image"].to(
                    device)  # (B, T_img, C, H, W)
                B, T_img = images_input.shape[:2]
                images_reshaped = einops.rearrange(
                    images_input, "b t c h w -> (b t) c h w")
                image_features_flat = image_encoder(
                    images_reshaped)  # (B*T_img, D_img_feat)
                image_features = einops.rearrange(
                    image_features_flat, "(b t) d -> b t d", b=B, t=T_img
                )  # (B, T_img, D_img_feat)

            # --- Prepare Padding Masks (Optional) ---
            # Assuming action_is_pad corresponds to state sequence padding
            state_padding_mask = None
            if "action_is_pad" in batch:
                # Ensure mask length matches state sequence length
                pad_mask_key = "action_is_pad"
                if batch[pad_mask_key].shape[1] >= scorer_cfg.max_state_len:
                    state_padding_mask = batch[pad_mask_key][:, :scorer_cfg.max_state_len].to(
                        device)  # (B, T_state)
                else:
                    print(
                        f"Warning: Padding mask length ({batch[pad_mask_key].shape[1]}) is shorter than state sequence ({scorer_cfg.max_state_len}).")
                    # Handle potential mismatch, e.g., by padding the mask or ignoring it

            # Image padding mask (if applicable, e.g., if T_img < max_image_len sometimes)
            image_padding_mask = None  # Assume no image padding for now

            # --- Compute Scores ---
            # Model expects state_sequences: (B, N_seq, T_state, D_state)
            # Here N_seq=1 as we process each sequence independently with its label
            state_sequence_unsqueezed = state_sequence.unsqueeze(
                1)  # (B, 1, T_state, D_state)
            state_padding_mask_unsqueezed = state_padding_mask.unsqueeze(
                1) if state_padding_mask is not None else None  # (B, 1, T_state)

            predicted_scores_single = critic_model(
                # Pass as (B, 1, T, D)
                state_sequences=state_sequence_unsqueezed,
                image_features=image_features,
                lang_instruction=lang_instructions,
                state_padding_mask=state_padding_mask_unsqueezed,
                image_padding_mask=image_padding_mask,
            )  # Output shape (B, 1)

            predicted_scores = predicted_scores_single.squeeze(1)  # (B,)

            # --- Compute Loss ---
            loss = criterion(predicted_scores, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                # Calculate accuracy (optional)
                preds = (torch.sigmoid(predicted_scores) > 0.5).float()
                acc = (preds == labels).float().mean()
                print(
                    f"Step: {step}/{training_steps} Loss: {loss.item():.4f} Acc: {acc.item():.3f}")

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
    # Save the MultimodalScorerConfig
    # Need a way to save dataclasses to json, or manually create dict
    import json
    config_dict = scorer_cfg.__dict__
    with open(output_directory / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)

    # Filter and save stats (only tensors)
    stats_to_save = {
        k: v for k, v in dataset_metadata.stats.items() if isinstance(v, torch.Tensor)}
    safetensors.torch.save_file(
        stats_to_save, output_directory / "stats.safetensors")
    print(f"Config and stats saved to: {output_directory}")


if __name__ == "__main__":
    main()
