import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class AdaLN(nn.Module):
    def __init__(self, num_features, num_conds):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))
        self.cond_to_gamma = nn.Linear(num_conds, num_features)
        self.cond_to_beta = nn.Linear(num_conds, num_features)

    def forward(self, x, z):
        gamma = self.cond_to_gamma(z) + self.gamma
        beta = self.cond_to_beta(z) + self.beta
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / (std + 1e-5)
        return gamma * x_norm + beta


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_conds):
        super().__init__()
        self.norm = AdaLN(in_channels, num_conds)
        self.fc1 = nn.Linear(in_channels, out_channels)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(out_channels, out_channels)
        self.res = (in_channels == out_channels)

    def forward(self, x, z):
        res = x
        out = self.norm(x, z)
        out = self.act(self.fc1(out))
        out = self.fc2(out)
        if self.res:
            return out + res
        else:
            raise ValueError("Residual dims do not match.")


class DenoisingMLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=3, width=1024, num_conds=None):
        super().__init__()
        assert num_conds is not None, "num_conds must be set for AdaLN"
        layers = []
        dims = [in_channels] + [width] * (num_blocks - 1)
        self.blocks = nn.ModuleList([
            ResidualBlock(dims[i], dims[i+1], num_conds)
            for i in range(len(dims)-1)
        ])
        self.final = nn.Linear(
            width, out_channels) if out_channels != width else nn.Identity()

    def forward(self, x, z, t):
        # x: [batch2, in_channels], z: [batch2, num_conds], t: [batch2, num_conds]
        cond = z + t
        out = x
        for blk in self.blocks:
            out = blk(out, cond)
        return self.final(out)


class DiffusionModel(nn.Module):
    """
    MLP-based diffusion denoiser with time embeddings.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 3):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(hidden_dim)
        self.denoiser = DenoisingMLP(
            in_channels=state_dim,
            out_channels=state_dim,
            num_blocks=num_layers,
            width=hidden_dim,
            num_conds=hidden_dim,
        )

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        x_flat = x.view(B * T, D)
        t_emb = self.time_emb(timesteps).repeat_interleave(T, dim=0)
        # use goal as zero if not provided
        if goal is None:
            cond = torch.zeros_like(t_emb)
        else:
            cond = goal.unsqueeze(1).repeat(1, T, 1).view(B * T, -1)
        out_flat = self.denoiser(x_flat, cond, t_emb)
        return out_flat.view(B, T, D)
