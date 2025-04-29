import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


class DiffusionSchedulerWrapper:
    """Wraps HuggingFace schedulers for noise/denoise steps."""

    def __init__(self, scheduler_type='ddpm', num_train_timesteps=1000, beta_schedule='linear', prediction_type='epsilon', **kwargs):
        if scheduler_type == 'ddpm':
            self.scheduler = DDPMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                prediction_type=prediction_type,
                **kwargs
            )
        elif scheduler_type == 'ddim':
            self.scheduler = DDIMScheduler(
                num_train_timesteps=num_train_timesteps,
                beta_schedule=beta_schedule,
                prediction_type=prediction_type,
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

    def add_noise(self, original_samples: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Add noise to samples according to the scheduler's definition."""
        noise = torch.randn_like(original_samples)
        noisy_samples = self.scheduler.add_noise(
            original_samples, noise, timesteps)
        return noisy_samples, noise

    def step(self, model_output: torch.Tensor, timestep: int, sample: torch.Tensor, **kwargs) -> tuple:
        """Perform one denoising step."""
        # Pass generator for reproducibility if needed
        return self.scheduler.step(model_output, timestep, sample, **kwargs).prev_sample

    def set_timesteps(self, num_inference_steps: int, device: torch.device):
        """Set the timesteps for the inference loop."""
        self.scheduler.set_timesteps(num_inference_steps, device=device)

    @property
    def timesteps(self):
        return self.scheduler.timesteps
