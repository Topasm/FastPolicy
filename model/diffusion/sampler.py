import torch


class TrajectorySampler:
    """
    Handles sampling candidate trajectories via a simple diffusion denoising pass.
    """

    def __init__(self, diffusion_model, horizon: int = 16, stride: int = 1):
        self.diffusion_model = diffusion_model
        self.horizon = horizon
        self.stride = stride

    def sample(self, s0: torch.Tensor, n_samples: int, goal: torch.Tensor = None) -> torch.Tensor:
        """
        s0: [state_dim] initial state
        n_samples: number of trajectories
        goal: optional [state_dim] goal conditioning
        returns: [n_samples, horizon/stride, state_dim] trajectories
        """
        device = s0.device
        # prepare base trajectories: repeat s0 over horizon
        base = s0.unsqueeze(0).unsqueeze(1).repeat(n_samples, self.horizon, 1)
        # initialize with Gaussian noise
        noise = torch.randn_like(base, device=device)
        x_noisy = base + noise
        # simple one-step denoising
        timesteps = torch.zeros(n_samples, dtype=torch.long, device=device)
        cond_goal = goal.unsqueeze(0).repeat(
            n_samples, 1) if goal is not None else None
        denoised = self.diffusion_model(x_noisy, timesteps, cond_goal)
        traj = x_noisy - denoised
        # apply stride for jump-step
        if self.stride > 1:
            traj = traj[:, ::self.stride, :]
        return traj
