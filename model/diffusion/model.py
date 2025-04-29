import torch
import torch.nn as nn


class DiffusionModel(nn.Module):
    """
    Simple Transformer-based denoising model for diffusion planning.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128, num_layers: int = 4, nhead: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(hidden_dim, state_dim)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, horizon, state_dim] noisy trajectories
        timesteps: [batch] current diffusion steps (unused in this simple model)
        goal: optional [batch, state_dim] conditioning
        returns: predicted noise of same shape as x
        """
        # project inputs
        h = self.input_proj(x)  # [batch, horizon, hidden_dim]
        # transformer expects [seq_len, batch, hidden_dim]
        h = h.permute(1, 0, 2)
        h = self.transformer(h)
        h = h.permute(1, 0, 2)
        out = self.output_proj(h)
        return out
