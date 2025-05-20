"""
GAN-style finetuning for a diffusion model with a critic model as the discriminator.
Uses N generated samples (default: 3) and 1 ground truth trajectory for training.

Usage:
    python -m train_eval.finetune_ganloss  # With default paths
    python -m train_eval.finetune_ganloss --diffusion_model /path/to/model --critic_model /path/to/critic
"""

import os
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import safetensors if available
try:
    import safetensors.torch
    has_safetensors = True
except ImportError:
    has_safetensors = False
    print("safetensors not available, falling back to PyTorch format for stats saving.")

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.normalize import Normalize
from model.diffusion.modeling_mymodel import MyDiffusionModel
from model.diffusion.configuration_mymodel import DiffusionConfig
from model.critic.modernbert_critic import ModernBertCritic, ModernBertCriticConfig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a diffusion model with GAN-style loss using a critic model")
    parser.add_argument("--dataset", type=str, default="lerobot/pusht",
                        help="Name of the dataset to use")

    # Default paths based on eval_critic_combined.py
    default_diffusion_path = "outputs/train/diffusion_only"
    default_critic_path = "outputs/train/modernbert_critic"
    default_critic_weights = os.path.join(
        default_critic_path, "modernbert_critic_weights.pth")
    default_critic_config = os.path.join(default_critic_path, "config.json")

    parser.add_argument("--diffusion_model", type=str, default=default_diffusion_path,
                        help="Path to pretrained diffusion model directory")
    parser.add_argument("--critic_model", type=str, default=default_critic_weights,
                        help="Path to pretrained critic model weights")
    parser.add_argument("--critic_config", type=str, default=default_critic_config,
                        help="Path to critic model config file")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for generator (diffusion model)")
    parser.add_argument("--steps", type=int, default=5000,
                        help="Number of training steps")
    parser.add_argument("--gen_weight", type=float, default=1.0,
                        help="Weight for generator loss")
    parser.add_argument("--disc_weight", type=float, default=0.5,
                        help="Weight for discriminator loss")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--output_dir", type=str, default="outputs/train/gan_finetuned",
                        help="Directory to save model and logs")
    parser.add_argument("--log_freq", type=int, default=50,
                        help="How often to log training metrics")
    parser.add_argument("--save_freq", type=int, default=1000,
                        help="How often to save model checkpoints")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of diffusion samples to generate for training (recommended: 3)")
    return parser.parse_args()


def resolve_path(path):
    """Resolve both absolute and relative paths"""
    if os.path.isabs(path):
        return path
    # Get the project root directory (assuming script is in train_eval folder)
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), ".."))
    return os.path.join(project_root, path)


def adjust_sequence_for_critic(state_sequence, target_horizon):
    """
    Adjust a state sequence to match the expected horizon length for the critic model.
    Will either pad or truncate the sequence as needed.

    Args:
        state_sequence: Tensor of shape [B, seq_len, D]
        target_horizon: Target sequence length

    Returns:
        Adjusted tensor of shape [B, target_horizon, D]
    """
    B, seq_len, D = state_sequence.shape

    if seq_len == target_horizon:
        # Already correct size
        return state_sequence
    elif seq_len > target_horizon:
        # Truncate to target length
        return state_sequence[:, :target_horizon, :]
    else:
        # Need to pad the sequence by repeating the last state
        padding_needed = target_horizon - seq_len

        # Create padding by repeating the last state
        last_states = state_sequence[:, -1:, :]
        padding = last_states.repeat(1, padding_needed, 1)

        # Concatenate with the original sequence
        padded_sequence = torch.cat([state_sequence, padding], dim=1)
        return padded_sequence


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Resolve paths for model and config files
    diffusion_model_path = resolve_path(args.diffusion_model)
    critic_model_path = resolve_path(args.critic_model)
    critic_config_path = resolve_path(args.critic_config)

    print(f"Using diffusion model from: {diffusion_model_path}")
    print(f"Using critic model from: {critic_model_path}")
    print(f"Using critic config from: {critic_config_path}")

    # Create output directory if it doesn't exist
    output_dir = resolve_path(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading dataset metadata for {args.dataset}...")
    dataset_metadata = LeRobotDatasetMetadata(args.dataset)
    features = dataset_to_policy_features(dataset_metadata.features)

    # Get state dimension and set image feature dimension
    state_dim = features["observation.state"].shape[0]

    # -- Load Diffusion Model (Generator) --
    print(f"Loading diffusion model from {diffusion_model_path}...")
    diffusion_config_path = Path(diffusion_model_path) / "config.json"

    if not diffusion_config_path.exists():
        raise FileNotFoundError(
            f"Diffusion config not found at {diffusion_config_path}")

    # Use the PreTrainedConfig from_pretrained method instead of from_json_file
    diffusion_config = DiffusionConfig.from_pretrained(
        str(diffusion_config_path.parent))

    # Load stats for normalization
    stats_path = Path(diffusion_model_path) / "stats.safetensors"
    if not stats_path.exists():
        stats_path = Path(diffusion_model_path) / "stats.pt"
        if not stats_path.exists():
            print(
                f"Warning: Stats file not found at {diffusion_model_path}. Creating default stats.")
            # Create default stats
            dataset_stats = {}
        else:
            # Load dataset stats
            dataset_stats = torch.load(stats_path, map_location="cpu")
    else:
        # Load dataset stats from safetensors
        with safetensors.torch.safe_open(stats_path, framework="pt", device="cpu") as f:
            dataset_stats = {k: f.get_tensor(k) for k in f.keys()}

    # If dataset_stats is empty, create default stats based on data dimensions
    if not dataset_stats:
        print("Creating default normalization statistics")
        # Create default stats for state
        state_dim = features["observation.state"].shape[0]
        dataset_stats["observation.state"] = {
            "min": torch.ones(state_dim) * -1,
            "max": torch.ones(state_dim) * 1,
            "mean": torch.zeros(state_dim),
            "std": torch.ones(state_dim)
        }

        # Create default stats for action
        action_dim = features["action"].shape[0]
        dataset_stats["action"] = {
            "min": torch.ones(action_dim) * -1,
            "max": torch.ones(action_dim) * 1,
            "mean": torch.zeros(action_dim),
            "std": torch.ones(action_dim)
        }

        # For image normalization
        dataset_stats["observation.image"] = {
            "mean": torch.tensor([0.5, 0.5, 0.5]),
            "std": torch.tensor([0.5, 0.5, 0.5]),
            # Not used for image normalization but included for completeness
            "min": torch.zeros(3),
            # Not used for image normalization but included for completeness
            "max": torch.ones(3)
        }

    # Check for invalid stats (inf or nan values)
    for key, value_dict in list(dataset_stats.items()):
        if isinstance(value_dict, dict):
            for stat_type, value in list(value_dict.items()):
                if torch.isinf(value).any() or torch.isnan(value).any():
                    print(
                        f"Warning: Invalid values found in stats for {key}.{stat_type}")
                    # Replace inf/nan with reasonable defaults
                    if stat_type == "min":
                        dataset_stats[key][stat_type] = torch.where(torch.isinf(value) | torch.isnan(value),
                                                                    torch.tensor(-1.0), value)
                    elif stat_type == "max":
                        dataset_stats[key][stat_type] = torch.where(torch.isinf(value) | torch.isnan(value),
                                                                    torch.tensor(1.0), value)
                    elif stat_type == "mean":
                        dataset_stats[key][stat_type] = torch.where(torch.isinf(value) | torch.isnan(value),
                                                                    torch.tensor(0.0), value)
                    elif stat_type == "std":
                        dataset_stats[key][stat_type] = torch.where(torch.isinf(value) | torch.isnan(value),
                                                                    torch.tensor(1.0), value)

    print("Dataset stats loaded. Keys:", list(dataset_stats.keys()))

    # First check which checkpoint file exists
    diffusion_ckpt_path = Path(diffusion_model_path) / "diffusion.pth"
    if not diffusion_ckpt_path.exists():
        diffusion_ckpt_path = Path(
            diffusion_model_path) / "diffusion_final.pth"
        if not diffusion_ckpt_path.exists():
            raise FileNotFoundError(
                f"Diffusion checkpoint not found at {diffusion_model_path}")

    # Make sure all stats tensors are on the proper device before creating the model
    for key in dataset_stats:
        if isinstance(dataset_stats[key], dict):
            for stat_type in dataset_stats[key]:
                if isinstance(dataset_stats[key][stat_type], torch.Tensor):
                    # We'll create the model on CPU first, so keep stats on CPU
                    dataset_stats[key][stat_type] = dataset_stats[key][stat_type].to(
                        'cpu')

    # Debug info about stats
    print("Stats keys and shapes:")
    for key in dataset_stats:
        if isinstance(dataset_stats[key], dict):
            print(f"  {key}:")
            for stat_type, value in dataset_stats[key].items():
                if isinstance(value, torch.Tensor):
                    print(
                        f"    {stat_type}: shape {value.shape}, device {value.device}")

    # Instantiate diffusion model with dataset stats on CPU first
    diffusion_model = MyDiffusionModel(diffusion_config, dataset_stats)

    # Load weights
    print(f"Loading diffusion weights from {diffusion_ckpt_path}")
    diffusion_state_dict = torch.load(diffusion_ckpt_path, map_location="cpu")

    # Load the state dict
    diffusion_model.load_state_dict(diffusion_state_dict, strict=False)
    print("Diffusion model loaded successfully.")

    # After loading, move the model to the target device
    diffusion_model = diffusion_model.to(device)
    print(f"Moved diffusion model to device: {device}")

    # -- Load Critic Model (Discriminator) --
    print(f"Loading critic model from {critic_model_path}...")

    # Load critic config
    with open(critic_config_path, 'r') as f:
        critic_config_dict = json.load(f)

    critic_config = ModernBertCriticConfig(
        state_dim=critic_config_dict.get('state_dim', state_dim),
        horizon=critic_config_dict.get('horizon', 16),
        hidden_dim=critic_config_dict.get('hidden_dim', 768),
        num_layers=critic_config_dict.get('num_layers', 8),
        num_heads=critic_config_dict.get('num_heads', 12),
        dropout=critic_config_dict.get('dropout', 0.1),
        use_layernorm=critic_config_dict.get('use_layernorm', True),
        swiglu_intermediate_factor=critic_config_dict.get(
            'swiglu_intermediate_factor', 4),
        half_horizon=critic_config_dict.get('half_horizon', 8)
    )

    # Instantiate critic model
    critic_model = ModernBertCritic(critic_config).to(device)

    # Load weights
    critic_state_dict = torch.load(critic_model_path, map_location="cpu")

    # Handle different saving formats
    if isinstance(critic_state_dict, dict) and 'model_state_dict' in critic_state_dict:
        critic_state_dict = critic_state_dict['model_state_dict']

    # Process state dict to remove "_orig_mod" prefix if present
    if all(k.startswith('_orig_mod.') for k in critic_state_dict.keys()):
        print("Removing '_orig_mod.' prefix from critic state dict keys")
        new_critic_state_dict = {}
        for k, v in critic_state_dict.items():
            new_critic_state_dict[k.replace('_orig_mod.', '')] = v
        critic_state_dict = new_critic_state_dict

    # Load with strict=False to allow for slight mismatches
    critic_model.load_state_dict(critic_state_dict, strict=False)
    print("Critic model loaded successfully.")

    # Configure dataset based on features
    horizon = diffusion_config.horizon

    # First, let's create our LeRobotDataset with minimum delta_timestamps to see what it provides
    # This will help us understand the real structure
    temp_dataset = LeRobotDataset(
        args.dataset,
        delta_timestamps={
            "observation.state": [0],
            "observation.image": [0]
        }
    )
    sample_batch = next(iter(DataLoader(temp_dataset, batch_size=1)))

    # Check the dimensions of the loaded tensors
    print("Sample batch dimensions:")
    for k, v in sample_batch.items():
        if isinstance(v, torch.Tensor) and ("observation.state" in k or "observation.image" in k):
            print(f"  {k}: {v.shape}")

    # Now, we need to ensure our diffusion config has the correct n_obs_steps

    # The original diffusion model was trained with n_obs_steps=2
    # Even if the sample batch shows a different value, we should keep the original
    # configuration to match the pretrained model's expectations
    original_n_obs_steps = 2  # Hardcoded from the original model
    print(f"Setting n_obs_steps to original value: {original_n_obs_steps}")
    diffusion_config.n_obs_steps = original_n_obs_steps

    # Setup delta timestamps for trajectory features with n_obs_steps=2 (original model)
    # For state: Need history (n_obs_steps) + horizon, in this case: -1, 0, 1, 2, ..., horizon-1
    # For image: Need exactly n_obs_steps entries: -1, 0
    delta_timestamps = {
        "observation.state": [i / dataset_metadata.fps for i in range(-1, horizon)],
        "observation.image": [i / dataset_metadata.fps for i in range(-1, 0)]
    }

    # Debug info about delta timestamps and config
    print(f"n_obs_steps: {diffusion_config.n_obs_steps}, horizon: {horizon}")
    print("Delta timestamps:", delta_timestamps)

    # Initialize dataset and dataloader
    print("Initializing dataset...")
    dataset = LeRobotDataset(args.dataset, delta_timestamps=delta_timestamps)
    print(f"Dataset initialized with {len(dataset)} samples")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=device.type == "cuda",
        drop_last=True
    )

    # -- Setup Optimizers --
    # Only optimize the diffusion model, keeping the critic fixed
    optimizer_gen = torch.optim.AdamW(
        diffusion_model.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Use cosine annealing schedule for better convergence
    scheduler_gen = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_gen, T_max=args.steps)

    # Set up training metrics tracking
    metrics = {
        'gen_losses': [],
        'disc_losses': [],
        'critic_scores_real': [],
        'critic_scores_fake': [],
        'lr': []
    }

    print(f"Starting GAN-style finetuning for {args.steps} steps...")
    print(
        f"Generator weight: {args.gen_weight}, Discriminator weight: {args.disc_weight}")

    # Training loop
    step = 0
    pbar = tqdm(total=args.steps, desc="GAN Finetuning")

    diffusion_model.train()
    critic_model.eval()  # Keep critic in eval mode as a fixed discriminator

    while step < args.steps:
        for batch in dataloader:
            # Debug print batch info once
            if step == 0:
                print("\nBatch keys:", list(batch.keys()))
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape {v.shape}, dtype {v.dtype}")

            try:
                # First move all batch tensors to the device where the model is
                device_batch = {k: (v.to(device) if isinstance(
                    v, torch.Tensor) else v) for k, v in batch.items()}

                # Special handling for images to match the expected format
                if "observation.image" in device_batch:
                    # Debug current format
                    img_shape = device_batch["observation.image"].shape
                    if step == 0:
                        print(f"Original image shape: {img_shape}")

                    # Reshape the image tensor to the format expected by _prepare_global_conditioning
                    # It MUST be in format [B, T, C, H, W] where T is exactly 2 (for n_obs_steps=2)

                    # Case 1: [B, C, H, W] - Need to add time dimension
                    if len(img_shape) == 4:  # [B, C, H, W]
                        # Add time dimension to make it [B, 1, C, H, W]
                        device_batch["observation.image"] = device_batch["observation.image"].unsqueeze(
                            1)

                        # We need exactly 2 timesteps for the original model, so duplicate the frame
                        device_batch["observation.image"] = device_batch["observation.image"].repeat(
                            1, 2, 1, 1, 1)

                        if step == 0:
                            print(
                                f"Added time dimension and duplicated frame: {device_batch['observation.image'].shape}")

                    # Case 2: [B, T, C, H, W] - Need to ensure exactly 2 timesteps
                    elif len(img_shape) == 5:
                        if img_shape[1] < 2:
                            # Repeat image if we have fewer than 2 timesteps
                            device_batch["observation.image"] = device_batch["observation.image"].repeat(
                                1, 2, 1, 1, 1)[:, :2]
                            if step == 0:
                                print(
                                    f"Repeated images to get 2 timesteps: {device_batch['observation.image'].shape}")
                        elif img_shape[1] > 2:
                            # Take only the first 2 timesteps if we have more
                            device_batch["observation.image"] = device_batch["observation.image"][:, :2]
                            if step == 0:
                                print(
                                    f"Sliced to get exactly 2 timesteps: {device_batch['observation.image'].shape}")

                    # Case 3: [B, T, N, C, H, W] - Multi-camera format, take first camera
                    elif len(img_shape) == 6:  # [B, T, N, C, H, W]
                        # Take first camera only to get [B, T, C, H, W]
                        device_batch["observation.image"] = device_batch["observation.image"][:, :, 0]

                        # Ensure exactly 2 timesteps
                        if device_batch["observation.image"].shape[1] < 2:
                            # Repeat frames if needed
                            device_batch["observation.image"] = device_batch["observation.image"].repeat(
                                1, 2, 1, 1, 1)[:, :2]
                        elif device_batch["observation.image"].shape[1] > 2:
                            # Slice to first 2 frames
                            device_batch["observation.image"] = device_batch["observation.image"][:, :2]

                        if step == 0:
                            print(
                                f"Processed multi-camera tensor to 2 timesteps: {device_batch['observation.image'].shape}")

                # In case our normalization still fails, let's add proper image normalization code
                if "observation.image" in device_batch:
                    # Standard normalization for images to [-1, 1] range
                    if not torch.is_floating_point(device_batch["observation.image"]):
                        device_batch["observation.image"] = device_batch["observation.image"].float(
                        ) / 127.5 - 1.0

                # Make sure observation.image shape is correct
                if "observation.image" in device_batch and step == 0:
                    print(
                        f"Processed image shape: {device_batch['observation.image'].shape}")

                # Also ensure observation.state has the correct length
                # For diffusion model: need exactly n_obs_steps + horizon
                if "observation.state" in device_batch:
                    # The total state length is n_obs_steps (for conditioning) + horizon (for generation)
                    # In this case, 2 + 16 = 18
                    total_state_length = diffusion_config.n_obs_steps + diffusion_config.horizon

                    if step == 0:
                        print(
                            f"State has {device_batch['observation.state'].shape[1]} timesteps")
                        print(
                            f"Required length: {total_state_length} (n_obs_steps={diffusion_config.n_obs_steps} + horizon={diffusion_config.horizon})")

                    # Handle the case where the state sequence is shorter than needed
                    if device_batch["observation.state"].shape[1] < total_state_length:
                        print(
                            f"Padding state sequence from {device_batch['observation.state'].shape[1]} to {total_state_length}")

                        # Pad the state sequence by repeating the last state
                        current_length = device_batch["observation.state"].shape[1]
                        padding_needed = total_state_length - current_length

                        # Extract last state for padding
                        last_states = device_batch["observation.state"][:, -1:, :]
                        padding = last_states.repeat(1, padding_needed, 1)

                        # Concatenate original sequence with padding
                        padded_states = torch.cat(
                            [device_batch["observation.state"], padding], dim=1)
                        device_batch["observation.state"] = padded_states

                        if step == 0:
                            print(
                                f"State shape after padding: {device_batch['observation.state'].shape}")

                        # Also pad padding mask if present
                        if "observation.state_is_pad" in device_batch:
                            # For padding mask, use True values (these are padded)
                            pad_mask = torch.ones(
                                device_batch["observation.state_is_pad"].shape[0],
                                padding_needed,
                                dtype=torch.bool,
                                device=device_batch["observation.state_is_pad"].device
                            )
                            device_batch["observation.state_is_pad"] = torch.cat([
                                device_batch["observation.state_is_pad"], pad_mask
                            ], dim=1)

                    elif device_batch["observation.state"].shape[1] > total_state_length:
                        # Slice to match exactly what we need
                        original_shape = device_batch["observation.state"].shape
                        device_batch["observation.state"] = device_batch["observation.state"][:,
                                                                                              :total_state_length]

                        if step == 0:
                            print(
                                f"Truncated state sequence from {original_shape[1]} to {total_state_length}")
                            print(
                                f"New state shape: {device_batch['observation.state'].shape}")

                        # Also adjust padding mask if present
                        if "observation.state_is_pad" in device_batch:
                            device_batch["observation.state_is_pad"] = device_batch["observation.state_is_pad"][:,
                                                                                                                :total_state_length]

                # Normalize input batch using diffusion model's normalize_inputs
                norm_batch = diffusion_model.normalize_inputs(device_batch)

                # Print normalized batch info once
                if step == 0:
                    print("\nNormalized batch keys:", list(norm_batch.keys()))
                    for k, v in norm_batch.items():
                        if isinstance(v, torch.Tensor):
                            print(
                                f"  {k}: shape {v.shape}, dtype {v.dtype}, device {v.device}")
                            print(
                                f"  {k} range: min={v.min().item():.4f}, max={v.max().item():.4f}")
            except Exception as e:
                print(f"Error in normalization: {e}")
                print("Debugging batch shapes:")
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        print(f"  {k}: shape {v.shape}")

                # As a fallback, create a simple normalization with correct tensor shapes
                print("Using simple normalization as fallback")
                norm_batch = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        # Move tensors to device first
                        v = v.to(device)
                        if v.dtype == torch.bool:
                            norm_batch[k] = v  # Keep boolean tensors unchanged
                        elif k == "observation.image":
                            # For images, we need to ensure the correct shape: [B, T, C, H, W] with T=2
                            if len(v.shape) == 4:  # [B, C, H, W]
                                # Add time dimension and repeat to get exactly 2 frames
                                v = v.unsqueeze(1)  # Now [B, 1, C, H, W]
                                # Now [B, 2, C, H, W]
                                v = v.repeat(1, 2, 1, 1, 1)
                                print(
                                    "Fallback: Added time dimension with 2 timesteps")
                            elif len(v.shape) == 5:  # [B, T, C, H, W]
                                # Make sure we have exactly 2 timesteps
                                if v.shape[1] != 2:
                                    if v.shape[1] > 2:
                                        v = v[:, :2]  # Slice to first 2 frames
                                    else:
                                        # Repeat frames to get 2 timesteps
                                        v = v.repeat(1, 2, 1, 1, 1)[:, :2]
                                    print(
                                        f"Fallback: Adjusted to exactly 2 timesteps: {v.shape}")
                            elif len(v.shape) == 6:  # [B, T, N, C, H, W]
                                # Take the first camera and ensure exactly 2 timesteps
                                v = v[:, :, 0]  # Now [B, T, C, H, W]
                                if v.shape[1] != 2:
                                    if v.shape[1] > 2:
                                        v = v[:, :2]  # Slice to first 2 frames
                                    else:
                                        # Repeat frames to get 2 timesteps
                                        v = v.repeat(1, 2, 1, 1, 1)[:, :2]
                                print(
                                    f"Fallback: Processed 6D tensor to 2 timesteps: {v.shape}")

                            # Simple normalization for images: scale to [-1, 1]
                            norm_batch[k] = (v.float() / 127.5) - 1.0

                        elif k == "observation.state":
                            # Make sure state has correct sequence length
                            total_state_length = diffusion_config.n_obs_steps + diffusion_config.horizon
                            if v.shape[1] != total_state_length:
                                if v.shape[1] > total_state_length:
                                    v = v[:, :total_state_length]
                                    print(
                                        f"Truncated state sequence to {total_state_length} steps")

                            # Simple normalization for states: scale to [-1, 1] using approximate bounds
                            norm_batch[k] = torch.clamp(v / 2.0, -1.0, 1.0)
                        else:
                            norm_batch[k] = v  # Keep other tensors unchanged
                    else:
                        norm_batch[k] = v  # Keep non-tensors unchanged

            # -- Generate Trajectories with Diffusion Model --
            try:
                # Prepare global conditioning from normalized batch
                global_cond = diffusion_model._prepare_global_conditioning(
                    norm_batch)

                # Generate multiple trajectories (fake samples)
                generated_trajectories = []
                for _ in range(args.num_samples):
                    trajectory = diffusion_model.conditional_sample(
                        batch_size=args.batch_size,
                        global_cond=global_cond
                    )
                    generated_trajectories.append(trajectory)
            except RuntimeError as e:
                print(f"Error in trajectory generation: {e}")
                # Let's print some debug information to help diagnose the issue
                print("\nDebugging dimensions:")
                print(
                    f"diffusion_config.n_obs_steps: {diffusion_config.n_obs_steps}")
                print(
                    f"n_obs_steps from batch: {norm_batch['observation.state'].shape[1]}")

                if "observation.image" in norm_batch:
                    print(
                        f"Image tensor shape: {norm_batch['observation.image'].shape}")

                # Check the global conditioning dimension
                try:
                    cond_embed_dim = diffusion_model.transformer.cond_embed.weight.shape[1]
                    print(f"Expected conditioning dim: {cond_embed_dim}")

                    # Calculate what the dimension should be
                    state_dim = diffusion_config.robot_state_feature.shape[0]
                    img_dim = diffusion_model.transformer.cond_embed.weight.shape[0] - \
                        state_dim * diffusion_config.n_obs_steps
                    print(
                        f"Calculated dimensions - state: {state_dim}, image: {img_dim}")
                except Exception as dim_error:
                    print(f"Error getting dimensions: {dim_error}")

                raise

            # -- Extract Real Trajectory --
            # Get ground truth trajectory (real sample)
            real_trajectory = norm_batch["observation.state"][:,
                                                              diffusion_config.n_obs_steps:, :]

            # Ensure the trajectories match the expected horizon length for the critic model
            horizon_length = min(
                real_trajectory.shape[1], critic_config.horizon)
            real_trajectory = adjust_sequence_for_critic(
                real_trajectory, horizon_length)

            # Also ensure generated trajectories match the same length
            for i in range(len(generated_trajectories)):
                generated_trajectories[i] = adjust_sequence_for_critic(
                    generated_trajectories[i], horizon_length)

            # -- Discriminator (Critic) Evaluation --
            # Get images for critic (only need the first timestep for state trajectories)
            if "observation.image" in norm_batch:
                # The critic model expects images in shape [B, C, H, W]
                img_tensor = norm_batch["observation.image"]
                if img_tensor.ndim == 5:  # [B, T, C, H, W]
                    # Get first timestep: [B, C, H, W]
                    critic_images = img_tensor[:, 0]
                elif img_tensor.ndim == 4:  # [B, C, H, W]
                    critic_images = img_tensor  # Already in correct format
                # Handle extra dimensions (e.g., multi-view)
                elif img_tensor.ndim > 5:
                    # Simplify by taking first indices of extra dimensions
                    while img_tensor.ndim > 5:
                        img_tensor = img_tensor[:, 0]
                    critic_images = img_tensor[:, 0]  # Now get first timestep
                else:
                    print(
                        f"Warning: Unexpected image shape {img_tensor.shape}, setting critic_images to None")
                    critic_images = None
            else:
                critic_images = None

            # Evaluate real trajectory
            try:
                with torch.no_grad():
                    real_score = critic_model.score(
                        trajectory_sequence=real_trajectory,
                        raw_images=critic_images,
                        second_half=False
                    )

                    # Print debug info for the first few iterations
                    if step < 3:
                        print(
                            f"Real trajectory shape: {real_trajectory.shape}")
                        print(
                            f"Real score shape: {real_score.shape}, numel: {real_score.numel()}")

            except Exception as e:
                print(f"Error evaluating real trajectory: {e}")
                print(f"Real trajectory shape: {real_trajectory.shape}")
                if critic_images is not None:
                    print(f"Critic images shape: {critic_images.shape}")
                raise

            # Evaluate each generated trajectory
            gen_scores = []
            for i, trajectory in enumerate(generated_trajectories):
                try:
                    with torch.no_grad():
                        score = critic_model.score(
                            trajectory_sequence=trajectory,
                            raw_images=critic_images,
                            second_half=False
                        )

                        # Print debug info for the first trajectory in the first few iterations
                        if i == 0 and step < 3:
                            print(
                                f"Gen trajectory {i} shape: {trajectory.shape}")
                            print(
                                f"Gen score {i} shape: {score.shape}, numel: {score.numel()}")

                    gen_scores.append(score)
                except Exception as e:
                    print(f"Error evaluating generated trajectory {i}: {e}")
                    print(f"Generated trajectory shape: {trajectory.shape}")
                    if critic_images is not None:
                        print(f"Critic images shape: {critic_images.shape}")
                    raise

            # -- Compute Generator Loss --
            # Generator wants to maximize critic scores (minimize negative scores)
            gen_loss = 0
            for score in gen_scores:
                # Handle tensor with multiple elements by ensuring we get a single scalar
                # The issue is that score might be [batch_size, 1] instead of [1]
                if score.numel() > 1:
                    # Properly reduce to scalar by flattening first and then mean
                    gen_loss -= score.flatten().mean()  # Negative since we want to maximize scores
                else:
                    gen_loss -= score.mean()  # Negative since we want to maximize scores

            # Also add standard diffusion loss for better state prediction
            diffusion_loss = diffusion_model.compute_diffusion_loss(norm_batch)

            # Combined loss
            total_gen_loss = args.gen_weight * diffusion_loss + args.disc_weight * gen_loss

            # Update generator (diffusion model)
            optimizer_gen.zero_grad()
            total_gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                diffusion_model.parameters(), max_norm=1.0)
            optimizer_gen.step()
            scheduler_gen.step()

            # -- Tracking Metrics --
            metrics['gen_losses'].append(total_gen_loss.item())
            # Handle potential multi-element tensors by flattening first
            metrics['critic_scores_real'].append(
                real_score.flatten().mean().item())
            metrics['critic_scores_fake'].append(
                sum([s.flatten().mean().item() for s in gen_scores]) / len(gen_scores))
            metrics['lr'].append(scheduler_gen.get_last_lr()[0])

            # -- Logging --
            if step % args.log_freq == 0:
                # Calculate average metrics over the last log_freq steps
                avg_gen_loss = sum(metrics['gen_losses'][-args.log_freq:]) / \
                    min(args.log_freq, len(metrics['gen_losses']))
                avg_real_score = sum(metrics['critic_scores_real'][-args.log_freq:]) / min(
                    args.log_freq, len(metrics['critic_scores_real']))
                avg_fake_score = sum(metrics['critic_scores_fake'][-args.log_freq:]) / min(
                    args.log_freq, len(metrics['critic_scores_fake']))

                pbar.set_postfix({
                    'gen_loss': f"{avg_gen_loss:.4f}",
                    'real_score': f"{avg_real_score:.4f}",
                    'fake_score': f"{avg_fake_score:.4f}",
                    'lr': f"{scheduler_gen.get_last_lr()[0]:.6f}"
                })

            # -- Saving --
            if step % args.save_freq == 0 and step > 0:
                checkpoint_path = Path(output_dir) / \
                    f"diffusion_gan_step_{step}.pth"
                torch.save(diffusion_model.state_dict(), checkpoint_path)
                print(f"\nCheckpoint saved to {checkpoint_path}")

                # Also save metrics
                metrics_path = Path(output_dir) / f"metrics_step_{step}.json"
                with open(metrics_path, 'w') as f:
                    # Convert tensors to floats for JSON serialization
                    json.dump(metrics, f, indent=2)

            step += 1
            pbar.update(1)
            if step >= args.steps:
                break

    # -- Save Final Model --
    final_path = Path(output_dir) / "diffusion_gan_final.pth"
    torch.save(diffusion_model.state_dict(), final_path)
    print(f"Training complete! Final model saved to {final_path}")

    # Save config and stats for future use
    diffusion_config.save_pretrained(output_dir)

    # Save stats file
    stats_output_path = Path(output_dir) / \
        ("stats.safetensors" if has_safetensors else "stats.pt")

    # Process and flatten stats dictionary for saving
    stats_to_save = {}
    for key, value in dataset_stats.items():
        if isinstance(value, torch.Tensor):
            stats_to_save[key] = value

    # Save using preferred format
    try:
        if has_safetensors:
            safetensors.torch.save_file(stats_to_save, stats_output_path)
        else:
            torch.save(stats_to_save, stats_output_path)
        print(f"Dataset statistics saved to {stats_output_path}")
    except Exception as e:
        fallback_path = Path(output_dir) / "stats.pt"
        print(f"Error saving stats: {e}")
        torch.save(stats_to_save, fallback_path)
        print(f"Stats saved using fallback format to {fallback_path}")

    # Save final metrics
    final_metrics_path = Path(output_dir) / "metrics_final.json"
    with open(final_metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
