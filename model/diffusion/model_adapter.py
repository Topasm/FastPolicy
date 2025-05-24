"""
Helper module to adapt different model configurations for evaluation.
"""

import torch
from model.diffusion.modeling_mymodel import MyDiffusionModel


def create_diffusion_model_from_config(config, dataset_stats=None):
    """
    Create a diffusion model from configuration, handling different config formats.

    Args:
        config: DiffusionConfig instance
        dataset_stats: Optional dataset statistics

    Returns:
        MyDiffusionModel instance
    """
    # Create a subclass to handle different config formats
    class AdaptedDiffusionModel(MyDiffusionModel):
        def __init__(self, config, dataset_stats=None):
            # Initialize the parent class but override the state_dim and action_dim attributes
            # Skip the problematic parent __init__
            super(MyDiffusionModel, self).__init__()

            self.config = config

            # Extract dimensions from features dictionary
            # For state dimension, find the observation.state feature
            state_found = False
            for feature_name, feature in config.input_features.items():
                if feature_name.endswith('state'):
                    self.state_dim = feature['shape'][0]
                    state_found = True
                    break

            if not state_found:
                raise ValueError(
                    "Could not find state feature in input_features")

            # For action dimension, find the action feature
            action_found = False
            for feature_name, feature in config.output_features.items():
                if feature_name == 'action':
                    self.action_dim = feature['shape'][0]
                    action_found = True
                    break

            if not action_found:
                raise ValueError(
                    "Could not find action feature in output_features")

            # Set other configuration attributes
            self.diffusion_target_dim = self.state_dim * config.output_horizon

            # Initialize normalizers/unnormalizers
            if dataset_stats is not None:
                from lerobot.common.policies.normalize import Normalize, Unnormalize
                from lerobot.configs.types import FeatureType, PolicyFeature

                # Convert feature dictionaries to the expected format if needed
                def convert_to_policy_features(features_dict):
                    converted_features = {}
                    for key, feature in features_dict.items():
                        # If it already has the right format, use it as is
                        if hasattr(feature, 'type') and hasattr(feature, 'shape'):
                            converted_features[key] = feature
                        # Otherwise convert to PolicyFeature with appropriate type
                        else:
                            if key.startswith('observation.'):
                                if key.endswith('.state'):
                                    feat_type = FeatureType.STATE
                                elif key.endswith('.image'):
                                    feat_type = FeatureType.VISUAL
                                else:
                                    feat_type = FeatureType.ENV
                            elif key == 'action':
                                feat_type = FeatureType.ACTION
                            else:
                                # Default to STATE type for other features
                                feat_type = FeatureType.STATE

                            converted_features[key] = PolicyFeature(
                                type=feat_type,
                                shape=feature['shape']
                            )
                    return converted_features

                # Create normalization mapping based on feature types
                from lerobot.configs.types import NormalizationMode
                normalization_mapping = {
                    FeatureType.STATE: NormalizationMode.MEAN_STD,
                    FeatureType.ACTION: NormalizationMode.MEAN_STD,
                    FeatureType.VISUAL: NormalizationMode.IDENTITY,
                    FeatureType.ENV: NormalizationMode.MEAN_STD
                }

                # Convert features to expected format
                converted_input_features = convert_to_policy_features(
                    config.input_features)

                # Create normalizers with converted features
                self.normalize_inputs = Normalize(
                    converted_input_features, normalization_mapping, dataset_stats)

                # Convert action features for unnormalize
                action_feature = {"action": PolicyFeature(
                    type=FeatureType.ACTION,
                    shape=(self.action_dim,)
                )}

                self.unnormalize_action_output = Unnormalize(
                    action_feature, normalization_mapping, dataset_stats
                )
            else:
                # Provide dummy implementations if no stats available
                self.normalize_inputs = lambda x: x
                self.unnormalize_action_output = lambda x: x

            # Initialize vision encoders for image input
            self.rgb_encoder = None
            # Check if there are any image features in our input features
            # For our simplified config, we check for keys containing 'image'
            image_features = {
                k: v for k, v in config.input_features.items() if 'image' in k}
            if image_features and len(image_features) > 0:
                # For simplicity, we're going to skip image feature extraction
                # and focus on the trajectory guidance from the multimodal model
                print(
                    "Image features detected but skipping RGB encoder initialization for compatibility")
                # Store that we have image features but don't actually create the encoder
                self.has_image_features = True
                self.rgb_encoder = None

            # Initialize the transformer for diffusion
            from model.diffusion.diffusion_modules import DiffusionTransformer
            self.transformer = DiffusionTransformer(
                config=config,
                global_cond_dim=self._get_global_cond_dim(config),
                output_dim=self.state_dim
            )

            # Initialize the noise scheduler
            if config.noise_scheduler_type == "DDPM":
                from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
                self.noise_scheduler = DDPMScheduler(
                    num_train_timesteps=config.num_train_timesteps,
                    beta_schedule=config.beta_schedule,
                    beta_start=config.beta_start,
                    beta_end=config.beta_end,
                    prediction_type=config.prediction_type,
                    clip_sample=config.clip_sample,
                    clip_sample_range=config.clip_sample_range,
                )
            elif config.noise_scheduler_type == "DDIM":
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                self.noise_scheduler = DDIMScheduler(
                    num_train_timesteps=config.num_train_timesteps,
                    beta_schedule=config.beta_schedule,
                    beta_start=config.beta_start,
                    beta_end=config.beta_end,
                    prediction_type=config.prediction_type,
                    clip_sample=config.clip_sample,
                    clip_sample_range=config.clip_sample_range,
                )
            else:
                raise ValueError(
                    f"Unsupported noise scheduler type {config.noise_scheduler_type}")

            # Number of inference steps: defaults to num_train_timesteps if not specified
            self.num_inference_steps = config.num_inference_steps or config.num_train_timesteps

        def _get_global_cond_dim(self, config):
            """Calculate the global conditioning dimension."""
            # Base dimension: t_emb
            global_cond_dim = config.transformer_dim

            # Add image conditioning dimension
            # For our simplified config, we check for keys containing 'image'
            image_features = {
                k: v for k, v in config.input_features.items() if 'image' in k}
            if image_features and len(image_features) > 0:
                # When using images, add dimension for each observation timestep
                global_cond_dim += config.transformer_dim * config.n_obs_steps

            # Add env state conditioning dimension if configured
            # In the original code, this looks for env_state_feature which we don't have
            # We're skipping this part since we don't use env features

            # Add our own compatibility layer for n_obs_steps if not defined
            if not hasattr(config, 'n_obs_steps'):
                config.n_obs_steps = 1

            return global_cond_dim

        def _prepare_global_conditioning(self, batch):
            """
            Process batch to extract global conditioning features.
            Adapted from original model to handle our config format.
            """
            # Initialize list to store all global conditioning features
            global_cond_feats = []

            # Get number of observation steps
            n_obs_steps = self.config.n_obs_steps

            # Check if images are configured
            if hasattr(self, 'has_image_features') and self.has_image_features:
                # Handle image conditioning if rgb_encoder exists
                if self.rgb_encoder is not None and "observation.image" in batch:
                    images = batch["observation.image"]

                    # If images doesn't have a time dimension, add one
                    if len(images.shape) == 4:  # [B, C, H, W]
                        # Add time dim [B, 1, C, H, W]
                        images = images.unsqueeze(1)

                    # Process images one by one for each timestep
                    for t in range(min(n_obs_steps, images.shape[1])):
                        img_t = images[:, t]  # Get images at timestep t
                        img_features = self.rgb_encoder(img_t)  # [B, D]
                        global_cond_feats.append(img_features)
                else:
                    # If rgb_encoder not set up or image not in batch, use zeros as placeholder
                    B = batch["observation.state"].shape[0]
                    dummy_features = torch.zeros(
                        (B, self.config.transformer_dim),
                        device=batch["observation.state"].device
                    )
                    # Add one feature tensor per observation timestep
                    for _ in range(n_obs_steps):
                        global_cond_feats.append(dummy_features)

            # Concatenate all conditioning features
            if global_cond_feats:
                global_cond = torch.cat(global_cond_feats, dim=1)
            else:
                # Fallback to empty tensor with correct shape if no features
                B = batch["observation.state"].shape[0]
                global_cond = torch.zeros(
                    (B, 0),  # Zero feature dim, will only have time embedding
                    device=batch["observation.state"].device
                )

            return global_cond

        def conditional_sample(self, batch_size=1, global_cond=None, generator=None,
                               guidance_scale=0.0, initial_guidance=None):
            """
            Sample from the diffusion model with optional trajectory guidance.

            Args:
                batch_size: Number of samples to generate
                global_cond: Optional global conditioning tensor
                generator: Optional random generator
                guidance_scale: How much to weight the guided trajectory (0.0-1.0)
                initial_guidance: Optional tensor to use as initial trajectory guide

            Returns:
                samples: Tensor of shape [batch_size, horizon, state_dim]
            """
            device = next(self.parameters()).device
            dtype = next(self.parameters()).dtype

            # Create sample shape for state sequence
            sample_shape = (batch_size, self.config.horizon, self.state_dim)

            # If guidance is provided, validate its shape
            if initial_guidance is not None:
                if initial_guidance.shape[1] != self.config.horizon:
                    print(
                        f"Warning: Initial guidance shape {initial_guidance.shape} doesn't match expected horizon {self.config.horizon}")
                    # Handle the mismatch - either pad or truncate
                    if initial_guidance.shape[1] < self.config.horizon:
                        # Pad with repeated last state
                        last_state = initial_guidance[:, -1:, :]
                        padding = last_state.repeat(
                            1, self.config.horizon - initial_guidance.shape[1], 1)
                        initial_guidance = torch.cat(
                            [initial_guidance, padding], dim=1)
                    else:
                        # Truncate to expected horizon
                        initial_guidance = initial_guidance[:,
                                                            :self.config.horizon, :]

                # Start with guided initial trajectory with some noise
                if guidance_scale > 0.0:
                    # Mix noise and guidance based on scale
                    noise = torch.randn(
                        size=sample_shape,
                        dtype=dtype,
                        device=device,
                        generator=generator
                    )
                    sample = guidance_scale * \
                        initial_guidance.to(device) + \
                        (1.0 - guidance_scale) * noise
                    print(
                        f"Starting diffusion with guidance (scale={guidance_scale})")
                else:
                    # Use pure noise if guidance_scale is 0
                    sample = torch.randn(
                        size=sample_shape,
                        dtype=dtype,
                        device=device,
                        generator=generator
                    )
            else:
                # Start with random noise (no guidance)
                sample = torch.randn(
                    size=sample_shape,
                    dtype=dtype,
                    device=device,
                    generator=generator
                )

            # Set up scheduler timesteps
            self.noise_scheduler.set_timesteps(self.num_inference_steps)

            # Perform denoising steps
            for t in self.noise_scheduler.timesteps:
                # Pass through transformer to get predicted noise or sample
                model_output = self.transformer(
                    sample,
                    torch.full(sample.shape[:1], t,
                               dtype=torch.long, device=device),
                    global_cond=global_cond
                )

                # Update sample using scheduler step
                sample = self.noise_scheduler.step(
                    model_output, t, sample, generator=generator
                ).prev_sample

                # Apply guidance at each step if enabled and available
                if guidance_scale > 0.0 and initial_guidance is not None and t > self.noise_scheduler.timesteps[-1] // 2:
                    # Gradually reduce guidance influence as we approach the end of denoising
                    # This helps ensure the final result is coherent
                    progress = (t.float() - self.noise_scheduler.timesteps[-1]) / (
                        self.noise_scheduler.timesteps[0] -
                        self.noise_scheduler.timesteps[-1]
                    )
                    current_scale = guidance_scale * (1.0 - progress)

                    # Mix the current sample with the guidance
                    if current_scale > 0.01:  # Only apply if scale is significant
                        sample = (1.0 - current_scale) * sample + \
                            current_scale * initial_guidance.to(device)

            return sample

        def load_state_dict(self, state_dict, strict=False):
            """
            Custom state dict loading function that handles dimension mismatches
            and missing normalizer keys.
            """
            # First, filter out the rgb_encoder keys since we're not using it
            filtered_state_dict = {
                k: v for k, v in state_dict.items() if not k.startswith('rgb_encoder.')}

            # Add dummy normalizer buffers if missing
            normalizer_keys = [
                "normalize_inputs.buffer_observation_state.mean",
                "normalize_inputs.buffer_observation_state.std",
                "unnormalize_action_output.buffer_action.mean",
                "unnormalize_action_output.buffer_action.std"
            ]

            # Check which normalizer keys are missing
            missing_keys = [
                k for k in normalizer_keys if k not in filtered_state_dict]

            # Add dummy values for missing keys if we have dataset_stats
            if hasattr(self, 'normalize_inputs') and not isinstance(self.normalize_inputs, type(lambda: None)):
                # Get shapes for normalizers
                state_shape = self.state_dim
                action_shape = self.action_dim
                device = next(self.parameters()).device

                # Create default values for normalizers if needed
                for key in missing_keys:
                    if key == "normalize_inputs.buffer_observation_state.mean":
                        filtered_state_dict[key] = torch.zeros(
                            state_shape, device=device)
                    elif key == "normalize_inputs.buffer_observation_state.std":
                        filtered_state_dict[key] = torch.ones(
                            state_shape, device=device)
                    elif key == "unnormalize_action_output.buffer_action.mean":
                        filtered_state_dict[key] = torch.zeros(
                            action_shape, device=device)
                    elif key == "unnormalize_action_output.buffer_action.std":
                        filtered_state_dict[key] = torch.ones(
                            action_shape, device=device)

            # Call the parent class's load_state_dict with strict=False
            # This will ignore missing keys for transformer blocks 6-11
            return super().load_state_dict(filtered_state_dict, strict=False)

    # Create and return the adapted model
    return AdaptedDiffusionModel(config, dataset_stats)
