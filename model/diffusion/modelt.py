# Create configuration file
print(default_api.insert_edit_into_file(
    filePath="/home/ahrilab/Desktop/FastPolicy/model/vision_policy/configuration.py",
    from dataclasses import dataclass, field
    from typing import Optional, Tuple, Dict, List

    # Placeholder for potential base config if needed
    # from lerobot.configs.policies import PreTrainedConfig
    # @PreTrainedConfig.register_subclass("vision_diffusion")

    # Define normalization modes similar to LeRobot
    class NormalizationMode:
    MEAN_STD="mean_std"
    MIN_MAX="min_max"

    @ dataclass
    class VisionDiffusionConfig:
    \"\"\"Configuration class for VisionConditionedDiffusionPolicy.\"\"\"

    # --- Input/Output Shapes ---
    # These should be filled based on the dataset/environment
    # Example shapes, adjust as needed
    input_features: Dict[str, List[int]]=field(default_factory=lambda: {
        "observation.image": [3, 224, 224],  # Example: C, H, W
        # "observation.state": [7], # Example state dim
        # "task_embedding": [512], # Example task embedding dim
    })
    output_features: Dict[str, List[int]]=field(default_factory=lambda: {
        "action": [7],  # Example action dim
    })

    # --- Normalization ---
    normalization_mapping: Dict[str, str]=field(default_factory=lambda: {
        "IMAGE": NormalizationMode.MEAN_STD,  # Example, adjust based on actual keys
        "STATE": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
        "TASK": NormalizationMode.MIN_MAX,  # If task embedding is used
    })

    # --- Architecture ---
    # Image Tokenizer
    image_tokenizer_kwargs: Dict=field(default_factory=lambda: {
        'image_size': (224, 224),
        'patch_size': 16,
        'embed_dim': 512,
        'num_frames': 16  # Corresponds to T_img
    })

    # Denoising Head
    denoising_head_kwargs: Dict=field(default_factory=lambda: {
        'input_dim': 512,  # Should match transformer_dim
        'hidden_dim': 256,
        'output_dim': 7  # Should match action_dim
    })

    # Transformer Core
    transformer_dim: int=512
    num_transformer_layers: int=6  # Or 4 based on robot_policy/train.py example
    transformer_heads: int=8
    use_task_embedding: bool=False
    # e.g., 512 if use_task_embedding is True
    task_embed_dim: Optional[int]=None

    # --- Diffusion Process ---
    # Scheduler Wrapper
    scheduler_kwargs: Dict=field(default_factory=lambda: {
        'scheduler_type': 'ddpm',  # or 'ddim'
        'num_train_timesteps': 100,
        'prediction_type': 'epsilon',  # or 'sample'
        'beta_start': 0.0001,  # Example, add if needed by scheduler
        'beta_end': 0.02,     # Example, add if needed by scheduler
        'beta_schedule': 'squaredcos_cap_v2',  # Example, add if needed
        'clip_sample': True,  # Example, add if needed
        'clip_sample_range': 1.0  # Example, add if needed
    })
    prediction_type: str='epsilon'  # Ensure consistency, matches scheduler_kwargs

    # --- Inference ---
    num_inference_steps: int=20  # Example
    guidance_scale: float=1.5  # Example CFG scale
    action_horizon: int=16  # Example, length of action sequence to generate

    # --- Training ---
    # These could be moved to a separate training config if preferred
    optimizer_lr: float=1e-4
    # Add other optimizer/scheduler params if needed (betas, eps, weight_decay, warmup_steps)
    cfg_prob: float=0.1  # Probability for CFG during training

    def __post_init__(self):
        # --- Validation ---
        if self.use_task_embedding:
            assert self.task_embed_dim is not None,
    "task_embed_dim must be provided if use_task_embedding is True"
            assert "task_embedding" in self.input_features,
    "task_embedding key must be in input_features if use_task_embedding is True"

        # Infer action_dim from output_features
        self.action_dim=self.output_features.get("action", [None])[0]
        if self.action_dim is None:
            raise ValueError("Action dimension not found in output_features")

        # Ensure denoising head output matches action dim
        self.denoising_head_kwargs['output_dim']=self.action_dim
        # Ensure denoising head input matches transformer dim
        self.denoising_head_kwargs['input_dim']=self.transformer_dim

        # Ensure image tokenizer embed_dim matches transformer dim (common practice)
        self.image_tokenizer_kwargs['embed_dim']=self.transformer_dim

        # Ensure prediction type consistency
        self.scheduler_kwargs['prediction_type']=self.prediction_type

        # TODO: Add more validation as needed (e.g., check keys in normalization_mapping)

    # --- Helper Properties (similar to LeRobot) ---
    @ property
    def image_keys(self) -> List[str]:
        return [k for k in self.input_features if k.startswith("observation.image")]

    # Add properties for state_key, task_key, action_key if needed
    # @property
    # def state_key(self) -> Optional[str]:
    #     # Find key like "observation.state"
    #     pass

    # @property
    # def action_key(self) -> str:
    #     return "action" # Assuming it's always "action"
    """,
    explanation="Create a configuration dataclass `VisionDiffusionConfig` in a new file, mirroring LeRobot's structure."
))

# Refactor scheduler_wrapper.py
print(default_api.insert_edit_into_file(
    filePath="/home/ahrilab/Desktop/FastPolicy/model/vision_policy/scheduler_wrapper.py",
    code="""import torch
    from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
    from typing import Optional, Union, Tuple

    class DiffusionSchedulerWrapper:
    \"\"\"Wraps HuggingFace schedulers for noise/denoise steps, similar to LeRobot usage.\"\"\"
    def __init__(self,
                 scheduler_type: str='ddpm',
                 num_train_timesteps: int=100,
                 prediction_type: str='epsilon',
                 **kwargs):
        \"\"\"
        Initializes the wrapper with a specific scheduler type and configuration.
        kwargs are passed directly to the underlying scheduler.
        \"\"\"
        self.scheduler_type=scheduler_type
        self.prediction_type=prediction_type
        self.num_train_timesteps=num_train_timesteps  # Store for reference

        scheduler_class=None
        if scheduler_type == 'ddpm':
            scheduler_class=DDPMScheduler
        elif scheduler_type == 'ddim':
            scheduler_class=DDIMScheduler
        else:
            raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

        # Filter kwargs relevant to the chosen scheduler if necessary, or pass all
        # Example: DDPMScheduler(**kwargs)
        self.scheduler=scheduler_class(
            num_train_timesteps=num_train_timesteps,
            prediction_type=prediction_type,
            **kwargs
    )
        self.timesteps=torch.arange(
            0, num_train_timesteps).long()  # Default timesteps

    def set_timesteps(self, num_inference_steps: int, device: Union[str, torch.device]):
        \"\"\"Sets the discrete timesteps used for the diffusion chain(inference).\"\"\"
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        self.timesteps=self.scheduler.timesteps  # Update internal timesteps

    def add_noise(self,
                  original_samples: torch.Tensor,
                  timesteps: torch.Tensor
                  ) -> Tuple[torch.Tensor, torch.Tensor]:
        \"\"\"Adds noise to the original samples for specified timesteps.\"\"\"
        noise=torch.randn_like(original_samples)
        noisy_samples=self.scheduler.add_noise(
            original_samples, noise, timesteps)
        return noisy_samples, noise

    def step(self,
             model_output: torch.Tensor,
             timestep: int,
             sample: torch.Tensor,
             generator: Optional[torch.Generator]=None
             ) -> torch.Tensor:
        \"\"\"Performs one step of the reverse diffusion process.\"\"\"
        # Use scheduler's step method. Output is often a dataclass, get prev_sample.
        prev_sample=self.scheduler.step(
            model_output, timestep, sample, generator=generator).prev_sample
        return prev_sample

    @ property
    def config(self):
        \"\"\"Provides access to the underlying scheduler's config.\"\"\"
        return self.scheduler.config
    """,
    explanation="Complete `DiffusionSchedulerWrapper` with `add_noise`, `step`, `set_timesteps` methods and config property."
))

# Refactor diffusion_policy.py
print(default_api.insert_edit_into_file(
    filePath="/home/ahrilab/Desktop/FastPolicy/model/vision_policy/diffusion_policy.py",
    code="""import torch
    import torch.nn as nn
    from typing import Optional, Dict, List
    from collections import deque
    import math  # Keep for SinusoidalPosEmb if defined here

    # Local imports (adjust paths if needed)
    from .configuration import VisionDiffusionConfig
    from .image_tokenizer import ImageTokenizer
    from .denoising_head import DenoisingHead
    from .scheduler_wrapper import DiffusionSchedulerWrapper
    # Optional: from .transformer_core import LLaMACore

    # --- Normalization Helpers (Mimicking LeRobot) ---
    # These could be moved to a separate utils/normalization.py file

    class Normalize:
    def __init__(self, features: Dict[str, List[int]], mapping: Dict[str, str], stats: Optional[Dict[str, Dict[str, torch.Tensor]]]):
        self.features=features
        # Maps feature type (e.g., IMAGE, STATE) to mode (min_max, mean_std)
        self.mapping=mapping
        self.stats=stats

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.stats is None:
            # print("Warning: Normalization stats not provided. Skipping normalization.")
            return batch

        normalized_batch={}
        for key, data in batch.items():
            feature_type=self._get_feature_type(key)
            mode=self.mapping.get(feature_type)
            stat=self.stats.get(key)

            if mode and stat:
                if mode == "min_max":
                    # Formula: 2 * (x - min) / (max - min) - 1
                    min_val=stat['min'].to(data.device, data.dtype)
                    max_val=stat['max'].to(data.device, data.dtype)
                    # Add epsilon to prevent division by zero if min == max
                    normalized_batch[key]=2 *
    (data - min_val) / (max_val - min_val + 1e-8) - 1
                elif mode == "mean_std":
                    mean=stat['mean'].to(data.device, data.dtype)
                    std=stat['std'].to(data.device, data.dtype)
                    # Add epsilon to prevent division by zero
                    normalized_batch[key]=(data - mean) / (std + 1e-8)
                else:
                    normalized_batch[key]=data  # Unknown mode
            else:
                normalized_batch[key]=data  # No normalization needed/possible
        return normalized_batch

    def _get_feature_type(self, key: str) -> Optional[str]:
        # Simple heuristic, refine as needed
        if "image" in key: return "IMAGE"
        if "state" in key: return "STATE"
        if "action" in key: return "ACTION"
        if "task" in key: return "TASK"
        return None

    class Unnormalize:
    def __init__(self, features: Dict[str, List[int]], mapping: Dict[str, str], stats: Optional[Dict[str, Dict[str, torch.Tensor]]]):
        self.features=features
        self.mapping=mapping
        self.stats=stats

    def __call__(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.stats is None:
            # print("Warning: Normalization stats not provided. Skipping unnormalization.")
            return batch

        unnormalized_batch={}
        for key, data in batch.items():
            feature_type=self._get_feature_type(key)
            mode=self.mapping.get(feature_type)
            stat=self.stats.get(key)

            if mode and stat:
                if mode == "min_max":
                    # Formula: (x + 1) * (max - min) / 2 + min
                    min_val=stat['min'].to(data.device, data.dtype)
                    max_val=stat['max'].to(data.device, data.dtype)
                    unnormalized_batch[key]=(
                        data + 1) * (max_val - min_val) / 2 + min_val
                elif mode == "mean_std":
                    mean=stat['mean'].to(data.device, data.dtype)
                    std=stat['std'].to(data.device, data.dtype)
                    unnormalized_batch[key]=data * std + mean
                else:
                    unnormalized_batch[key]=data  # Unknown mode
            else:
                # No unnormalization needed/possible
                unnormalized_batch[key]=data
        return unnormalized_batch

    def _get_feature_type(self, key: str) -> Optional[str]:
        # Simple heuristic, refine as needed
        if "image" in key: return "IMAGE"
        if "state" in key: return "STATE"
        if "action" in key: return "ACTION"
        if "task" in key: return "TASK"
        return None

    # --- Helper Modules ---
    # Keep SinusoidalPosEmb here or move to utils
    class SinusoidalPosEmb(nn.Module):
    # ... existing code ...

    # --- Core Diffusion Model ---
    class VisionDiffusionModel(nn.Module):
    \"\"\"Core network components for the vision-conditioned diffusion policy.\"\"\"
    def __init__(self, config: VisionDiffusionConfig):
        super().__init__()
        self.config=config

        # --- Components ---
        self.image_tokenizer=ImageTokenizer(**config.image_tokenizer_kwargs)
        self.denoising_head=DenoisingHead(**config.denoising_head_kwargs)
        self.scheduler=DiffusionSchedulerWrapper(
            **config.scheduler_kwargs)  # Contains the scheduler

        # --- Embeddings ---
        self.time_embed=SinusoidalPosEmb(config.transformer_dim)
        self.action_embed=nn.Linear(config.action_dim, config.transformer_dim)
        if config.use_task_embedding:
            self.task_embed=nn.Linear(
                config.task_embed_dim, config.transformer_dim)
        else:
            self.task_embed=None

        # --- Transformer Decoder ---
        decoder_layer=nn.TransformerDecoderLayer(
            d_model=config.transformer_dim,
            nhead=config.transformer_heads,
            dim_feedforward=config.transformer_dim * 4,
            batch_first=True
    )
        self.transformer_decoder=nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_transformer_layers)

        # Optional: Action Positional Embedding
        # self.action_pos_embed = nn.Parameter(torch.zeros(1, config.action_horizon, config.transformer_dim))

    def _prepare_context_tokens(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        \"\"\"Combine vision and optional task embeddings into context tokens.\"\"\"
        # Assumes images are already stacked under a single key if multiple cameras
        # This key needs to be defined (e.g., "observation.images")
        image_keys=self.config.image_keys
        if not image_keys:
    raise ValueError("No image keys found in config.input_features")

        # Simple case: only one image key
        if len(image_keys) == 1:
            images=batch[image_keys[0]]  # Shape [B, T_img, C, H, W]
        else:
            # Stack images from multiple cameras if needed
            # Assumes they are provided separately in the batch
            # Shape [B, T_img, N_cam, C, H, W]
            images=torch.stack([batch[key] for key in image_keys], dim=2)
            # Reshape for tokenizer: [B * T_img, N_cam, C, H, W] or flatten N_cam?
            # Tokenizer needs to handle multiple cameras or expect flattened input
            # Current ImageTokenizer likely expects [B', C, H, W] or [B', T, C, H, W]
            # Let's assume tokenizer handles [B, T_img, C, H, W] for now
            # If multiple cameras, adapt tokenizer or preprocessing
            raise NotImplementedError(
                "Multi-camera handling needs verification with ImageTokenizer")


        # Expected: [B, N_vision, D_transformer]
        vision_tokens=self.image_tokenizer(images)

        context_tokens=vision_tokens

        if self.config.use_task_embedding:
            task_key="task_embedding"  # Assuming this key exists in batch
            if task_key not in batch:
                 raise ValueError(
                     f"'{task_key}' not found in batch but use_task_embedding is True")
            task_embedding=batch[task_key]
            # Process task embedding
            if task_embedding.ndim == 2:  # [B, D_task]
                task_tokens=self.task_embed(task_embedding).unsqueeze(
                    1)  # [B, 1, D_transformer]
            elif task_embedding.ndim == 3:  # [B, T_task, D_task]
                # [B, T_task, D_transformer]
                task_tokens=self.task_embed(task_embedding)
            else:
                raise ValueError("Invalid task embedding shape")
            # Concatenate vision and task tokens
            # [B, N_vision + N_task, D_transformer]
            context_tokens=torch.cat([vision_tokens, task_tokens], dim=1)

        return context_tokens

    def compute_loss(self,
                     batch: Dict[str, torch.Tensor],
                     cond_mask: Optional[torch.Tensor]=None
                     ) -> torch.Tensor:
        \"\"\"Calculates the diffusion loss for training.\"\"\"
        # Assumes batch contains normalized data
        actions=batch["action"]  # Key assumed based on config
        # Assumes image keys are present, potentially state/task keys too

        B, T_act, A=actions.shape
        device=actions.device

        # 1. Sample timesteps
        timesteps=torch.randint(
            0, self.scheduler.num_train_timesteps, (B,), device=device).long()

        # 2. Add noise to actions
        noisy_actions, noise_target=self.scheduler.add_noise(
            actions, timesteps)

        # 3. Get context tokens (vision + task)
        context_tokens=self._prepare_context_tokens(batch)

        # 4. Classifier-Free Guidance (CFG) during training
        if cond_mask is not None and self.training:
            uncond_context_tokens=torch.zeros_like(context_tokens)
            context_tokens=torch.where(cond_mask.view(B, 1, 1).expand_as(context_tokens),
                                       context_tokens,
                                       uncond_context_tokens)

        # 5. Embed inputs
        time_emb=self.time_embed(timesteps).unsqueeze(
            1)  # [B, 1, D_transformer]
        # [B, T_act, D_transformer]
        action_emb=self.action_embed(noisy_actions)

        # Optional: Add positional encoding to actions
        # if hasattr(self, 'action_pos_embed'):
        #     action_emb = action_emb + self.action_pos_embed[:, :T_act, :]

        # 6. Prepare Transformer inputs
        # [B, 1 + T_act, D_transformer]
        decoder_input=torch.cat([time_emb, action_emb], dim=1)
        tgt_mask=nn.Transformer.generate_square_subsequent_mask(
            decoder_input.size(1), device=device)

        # 7. Run Transformer Decoder
        transformer_output=self.transformer_decoder(
            tgt=decoder_input,
            memory=context_tokens,
            tgt_mask=tgt_mask
    )
        # [B, T_act, D_transformer]
        action_output_tokens=transformer_output[:, 1:, :]

        # 8. Denoising Head
        predicted_output=self.denoising_head(
            action_output_tokens)  # [B, T_act, A]

        # 9. Determine target based on prediction type
        if self.config.prediction_type == 'epsilon':
            target=noise_target
        elif self.config.prediction_type == 'sample':
            target=actions  # Predict clean sample
        else:
            raise ValueError(
                f"Unknown prediction type: {self.config.prediction_type}")

        # 10. Calculate Loss (MSE)
        loss=nn.functional.mse_loss(predicted_output, target, reduction="none")

        # Optional: Mask loss for padding (similar to LeRobot)
        # if self.config.do_mask_loss_for_padding:
        #     if "action_is_pad" not in batch:
        #         raise ValueError("...")
        #     in_episode_bound = ~batch["action_is_pad"]
        #     loss = loss * in_episode_bound.unsqueeze(-1)

        return loss.mean()


    @ torch.no_grad()
    def generate_actions(self,
                         # Contains observations
                         batch: Dict[str, torch.Tensor],
                         generator: Optional[torch.Generator]=None
                         ) -> torch.Tensor:
        \"\"\"Generates action sequence via diffusion inference.\"\"\"
        # Assumes batch contains normalized observation data
        B=list(batch.values())[0].shape[0]  # Get batch size from first element
        device=list(batch.values())[0].device
        T_act=self.config.action_horizon
        action_dim=self.config.action_dim
        guidance_scale=self.config.guidance_scale
        num_inference_steps=self.config.num_inference_steps

        # 1. Prepare scheduler and initial noise
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        noisy_actions=torch.randn(
            (B, T_act, action_dim), device=device, generator=generator)

        # 2. Get conditional context tokens
        cond_tokens=self._prepare_context_tokens(batch)

        # 3. Prepare unconditional tokens if using CFG
        uncond_tokens=None
        if guidance_scale > 1.0:
            # Simple approach: zero out context
            # TODO: Use dedicated null embeddings if trained with them
            uncond_tokens=torch.zeros_like(cond_tokens)

        # 4. Denoising loop
        for t in self.scheduler.timesteps:
            timesteps_batch=torch.tensor(
                [t] * B, device=device, dtype=torch.long)

            # --- CFG Prediction ---
            if guidance_scale > 1.0 and uncond_tokens is not None:
                # Predict for both conditional and unconditional
                model_input_actions=torch.cat([noisy_actions] * 2, dim=0)
                model_context=torch.cat([cond_tokens, uncond_tokens], dim=0)
                model_timesteps=torch.cat([timesteps_batch] * 2, dim=0)

                # Embed inputs
                time_emb=self.time_embed(model_timesteps).unsqueeze(1)
                action_emb=self.action_embed(model_input_actions)
                # if hasattr(self, 'action_pos_embed'): action_emb += self.action_pos_embed[:, :T_act, :]
                decoder_input=torch.cat([time_emb, action_emb], dim=1)
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    decoder_input.size(1), device=device)

                # Run transformer
                transformer_output=self.transformer_decoder(
                    decoder_input, model_context, tgt_mask=tgt_mask)
                action_output_tokens=transformer_output[:, 1:, :]

                # Run head
                model_output_combined=self.denoising_head(action_output_tokens)

                # Split and combine using guidance scale
                model_output_cond, model_output_uncond=torch.chunk(
                    model_output_combined, 2, dim=0)
                model_output=model_output_uncond + guidance_scale *
                    (model_output_cond - model_output_uncond)
            else:
                # Standard prediction without CFG
                time_emb=self.time_embed(timesteps_batch).unsqueeze(1)
                action_emb=self.action_embed(noisy_actions)
                # if hasattr(self, 'action_pos_embed'): action_emb += self.action_pos_embed[:, :T_act, :]
                decoder_input=torch.cat([time_emb, action_emb], dim=1)
                tgt_mask=nn.Transformer.generate_square_subsequent_mask(
                    decoder_input.size(1), device=device)

                transformer_output=self.transformer_decoder(
                    decoder_input, cond_tokens, tgt_mask=tgt_mask)
                action_output_tokens=transformer_output[:, 1:, :]
                model_output=self.denoising_head(action_output_tokens)

            # 5. Scheduler step
            noisy_actions=self.scheduler.step(
                model_output, t, noisy_actions, generator=generator)

        # Return the final denoised actions
        return noisy_actions


    # --- Main Policy Class (Interface) ---
    # Rename the original class or create new
    # Or inherit from a base class
    class VisionConditionedDiffusionPolicy(nn.Module):
    \"\"\"
    Main policy interface, similar to LeRobot's DiffusionPolicy.
    Handles normalization, observation queuing(optional), and calls the core model.
    \"\"\"
    def __init__(self,
                 config: VisionDiffusionConfig,
                 dataset_stats: Optional[Dict[str, Dict[str, torch.Tensor]]]=None):
        super().__init__()
        self.config=config

        # Normalization layers
        self.normalize_inputs=Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets=Normalize(
            config.output_features, config.normalization_mapping, dataset_stats)
        self.unnormalize_outputs=Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats)

        # Core diffusion model
        self.model=VisionDiffusionModel(config)

        # Observation/Action Queues (Optional - for rollout)
        self._queues=None
        # self.reset() # Call if using queues

    # def reset(self):
    #     \"\"\"Clear observation and action queues.\"\"\"
    #     # Initialize deques based on config.input_features and action_horizon/n_action_steps
    #     # Example:
    #     # self._queues = {key: deque(maxlen=...) for key in self.config.input_features}
    #     # self._queues["action"] = deque(maxlen=...)
    #     pass

    # @torch.no_grad()
    # def select_action(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    #     \"\"\"Select action during rollout, managing queues.\"\"\"
    #     # 1. Normalize inputs
    #     # 2. Update queues (populate_queues equivalent)
    #     # 3. If action queue is empty:
    #     #    a. Prepare batch from queues
    #     #    b. Call self.model.generate_actions(queued_batch)
    #     #    c. Unnormalize actions
    #     #    d. Fill action queue
    #     # 4. Pop and return action from queue
    #     raise NotImplementedError("Rollout logic with queues not fully implemented yet.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Optional[Dict]]:
        \"\"\"Training forward pass: normalize, compute loss, return loss.\"\"\"
        # 1. Normalize inputs and targets
        normalized_batch=self.normalize_inputs(batch)
        # Targets (actions) also need normalization for loss calculation if prediction_type='sample'
        # or if noise target depends on normalized actions
        normalized_batch=self.normalize_targets(
            normalized_batch)  # Overwrites action key

        # 2. Calculate CFG mask (optional)
        cond_mask=None
        if self.training and self.config.cfg_prob > 0:
    B=list(normalized_batch.values())[0].shape[0]
    device=list(normalized_batch.values())[0].device
    cond_mask=(torch.rand(B, device=device) > self.config.cfg_prob).long()

        # 3. Compute loss using the core model
        loss=self.model.compute_loss(normalized_batch, cond_mask=cond_mask)

        # 4. Return loss (and optionally other metrics in a dict)
        return loss, None  # No extra outputs for now

    @ torch.no_grad()
    def inference(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
         \"\"\"Inference pass: normalize inputs, generate actions, unnormalize.\"\"\"
    # 1. Normalize inputs
    normalized_batch=self.normalize_inputs(batch)

    # 2. Generate actions using the core model
    generated_actions_normalized=self.model.generate_actions(normalized_batch)

    # 3. Unnormalize outputs
    # Create a dict for unnormalizer
    output_batch_normalized={"action": generated_actions_normalized}
    output_batch=self.unnormalize_outputs(output_batch_normalized)

    return output_batch["action"]

    # Add methods for saving/loading state_dict if needed, potentially handling stats
    # def load_state_dict(self, state_dict, strict=True):
    #     # Handle loading stats if they are part of the state_dict
    #     # Update self.normalize_inputs.stats, etc.
    #     super().load_state_dict(state_dict, strict)

    # def state_dict(self, destination=None, prefix='', keep_vars=False):
    #     # Include stats in state_dict if needed
    #     sd = super().state_dict(destination, prefix, keep_vars)
    #     # Add stats: sd[prefix + 'stats'] = self.normalize_inputs.stats
    #     return sd

    # Remove the old VisionConditionedDiffusionPolicy class definition if it was in this file
    # del VisionConditionedDiffusionPolicy # Or comment out the old one
    """,
    explanation="Refactor `diffusion_policy.py` to include `Normalize`, `Unnormalize`, the main `VisionConditionedDiffusionPolicy` interface, and the core `VisionDiffusionModel`."
))

# Update robot_policy/train.py to use the new structure
print(default_api.insert_edit_into_file(
    filePath="/home/ahrilab/Desktop/FastPolicy/robot_policy/train.py",
    code="""  # Placeholder for training script
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
    import yaml  # Add yaml import
    import os  # Add os import

    # Updated import paths
    # Assuming the refactored policy is now the main interface
    from model.vision_policy.diffusion_policy import VisionConditionedDiffusionPolicy
    # Import the configuration class
    from model.vision_policy.configuration import VisionDiffusionConfig

    # --- Dummy Dataset (Keep for now or replace with actual dataset loading) ---
    class DummyRobotDataset(Dataset):
    # ... existing code ...

    # --- Training Function ---
    def train():
    # 1. Load Config from YAML (or define default)
    config_path="configs/vision_policy_config.yaml"  # Example path
    if os.path.exists(config_path):
    with open(config_path, "r") as f:
    raw_config_dict=yaml.safe_load(f)
    # TODO: Map raw_config_dict to VisionDiffusionConfig fields if needed
    # config = VisionDiffusionConfig(**raw_config_dict) # Direct mapping might work
    # Or manually create config = VisionDiffusionConfig(field1=raw_config_dict['sec1']['field1'], ...)
    print(f"Loading config from {config_path}")
    # Example manual mapping (adjust based on your YAML structure)
    # This assumes YAML structure matches VisionDiffusionConfig fields directly
    config=VisionDiffusionConfig(**raw_config_dict)

    else:
        print(f"Config file not found at {config_path}, using default config.")
        config=VisionDiffusionConfig()  # Use default values

    # --- Training Settings ---
    device=torch.device(config.training.get('device', 'cuda' if torch.cuda.is_available(
    ) else 'cpu'))  # Example: get device from config
    batch_size=config.training.get('batch_size', 4)  # Example
    epochs=config.training.get('epochs', 5)  # Example
    lr=config.optimizer_lr  # Get LR from main config section

    # 2. Setup Dataset and DataLoader
    # TODO: Replace DummyRobotDataset with your actual dataset loading
    # dataset = YourActualDataset(...)
    # dataset_stats = dataset.get_stats() # IMPORTANT: Get normalization stats
    dataset=DummyRobotDataset(
        seq_len=config.image_tokenizer_kwargs['num_frames'],
        action_dim=config.action_dim
    )
    # Placeholder for stats - replace with actual stats from your dataset
    dataset_stats={
        # Example stats structure (replace with real data)
        "action": {"min": torch.zeros(config.action_dim)-1, "max": torch.ones(config.action_dim)},
        "observation.image": {"mean": torch.zeros(3, 1, 1)+0.5, "std": torch.ones(3, 1, 1)*0.5},
        # Add stats for state, task_embedding if used
    }
    dataloader=DataLoader(dataset, batch_size=batch_size, shuffle=True)


    # 3. Initialize Policy (using config and stats)
    policy=VisionConditionedDiffusionPolicy(
        config=config, dataset_stats=dataset_stats).to(device)

    # 4. Optimizer
    # TODO: Add scheduler if needed (e.g., CosineAnnealingLR) based on config
    # Use parameters() of the main policy class
    optimizer=torch.optim.AdamW(policy.parameters(), lr=lr)

    # 5. Training Loop
    policy.train()
    for epoch in range(epochs):
        total_loss=0
        for batch in dataloader:
            # Move batch to device
            batch={k: v.to(device) for k, v in batch.items()}

            # Forward pass through the policy interface class
            loss, _=policy(batch)  # policy.forward(batch)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # scheduler.step() # If using a learning rate scheduler

            total_loss += loss.item()

        avg_loss=total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    print("Training finished.")
    # Add code to save the model checkpoint
    # Consider saving config and stats along with the model state_dict
    # torch.save({
    #     'config': config, # Save config dataclass
    #     'stats': dataset_stats,
    #     'model_state_dict': policy.state_dict()
    # }, 'vision_diffusion_policy_checkpoint.pth')


    if __name__ == "__main__":
    # Create a dummy config file if it doesn't exist for testing
    config_path="configs/vision_policy_config.yaml"
    if not os.path.exists(config_path):
        os.makedirs("configs", exist_ok=True)
        default_config=VisionDiffusionConfig()
        # Convert dataclass to dict for saving (simple version)
        import dataclasses
        config_dict=dataclasses.asdict(default_config)
        # Add a dummy training section if not part of main config
        config_dict['training']={'device': 'cpu', 'batch_size': 4, 'epochs': 1}
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)
        print(f"Created dummy config file at {config_path}")

    train()

    """,
    explanation="Update `robot_policy/train.py` to use `VisionDiffusionConfig`, initialize `VisionConditionedDiffusionPolicy` with config and stats, and adapt the training loop."
))

# Update robot_policy/inference.py similarly
print(default_api.insert_edit_into_file(
    filePath="/home/ahrilab/Desktop/FastPolicy/robot_policy/inference.py",
    code="""  # Placeholder for inference script
    import torch
    import yaml  # Add yaml import
    import os  # Add os import

    # Updated import paths
    from model.vision_policy.diffusion_policy import VisionConditionedDiffusionPolicy
    from model.vision_policy.configuration import VisionDiffusionConfig

    def inference():
    # 1. Load Checkpoint and Config
    checkpoint_path='vision_diffusion_policy_checkpoint.pth'  # Example path
    if not os.path.exists(checkpoint_path):
        print(
            f"Checkpoint not found at {checkpoint_path}. Cannot run inference.")
        # Optionally, initialize with default config for structure check
        config=VisionDiffusionConfig()
        dataset_stats=None  # No stats available without checkpoint
        policy=VisionConditionedDiffusionPolicy(
            config=config, dataset_stats=dataset_stats)
        print("Initialized policy with default config (no weights loaded).")
    else:
        # Load to CPU first
        checkpoint=torch.load(checkpoint_path, map_location='cpu')
        config=checkpoint['config']  # Load config from checkpoint
        dataset_stats=checkpoint['stats']  # Load stats from checkpoint
        print(f"Loaded config and stats from {checkpoint_path}")

        # 2. Initialize Policy
        policy=VisionConditionedDiffusionPolicy(
            config=config, dataset_stats=dataset_stats)
        policy.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model weights from {checkpoint_path}")

    # --- Inference Settings ---
    device=torch.device(config.inference.get(
        'device', 'cuda' if torch.cuda.is_available() else 'cpu'))  # Example
    policy.to(device)
    policy.eval()

    # 3. Prepare Dummy Input Data (or load real data)
    batch_size=1
    # Create dummy data based on config.input_features
    dummy_batch={}
    for key, shape in config.input_features.items():
        # Add time dimension if needed (e.g., for images)
        # This depends on how the model expects input during inference
        # Assuming T_img = config.image_tokenizer_kwargs['num_frames']
        if "image" in key:
            num_frames=config.image_tokenizer_kwargs['num_frames']
            dummy_batch[key]=torch.randn(
                batch_size, num_frames, *shape).to(device)
        # Add other inputs like state, task_embedding if required by config
        # elif "state" in key:
        #     num_frames = ... # Determine required history length for inference
        #     dummy_batch[key] = torch.randn(batch_size, num_frames, *shape).to(device)
        else:
    # Assuming non-sequential inputs don't need extra time dim for inference prep
    dummy_batch[key]=torch.randn(batch_size, *shape).to(device)


    # 4. Run Inference via policy interface
    print("Running inference...")
    with torch.no_grad():
        # The policy.inference method handles normalization internally
        actions=policy.inference(dummy_batch)

    print(f"Inference complete. Generated actions shape: {actions.shape}")
    # Example: print first action of the first batch
    # print(f"First action: {actions[0, 0]}")


    if __name__ == \"__main__\":
    # Ensure a dummy checkpoint exists for testing structure (if train wasn't run)
    checkpoint_path='vision_diffusion_policy_checkpoint.pth'
    if not os.path.exists(checkpoint_path):
    print(
        f"Creating dummy checkpoint at {checkpoint_path} for structure testing.")
    # Create dummy config and stats
    config=VisionDiffusionConfig()
    dataset_stats={
        "action": {"min": torch.zeros(config.action_dim)-1, "max": torch.ones(config.action_dim)},
        "observation.image": {"mean": torch.zeros(3, 1, 1)+0.5, "std": torch.ones(3, 1, 1)*0.5},
    }
    # Initialize policy to get state_dict structure
    policy=VisionConditionedDiffusionPolicy(
        config=config, dataset_stats=dataset_stats)
    torch.save({
        'config': config,
        'stats': dataset_stats,
        'model_state_dict': policy.state_dict()
    }, checkpoint_path)

    inference()

    explanation="Update `robot_policy/inference.py` to load the config and stats from a checkpoint, initialize the policy, and use the `policy.inference` method."
))
