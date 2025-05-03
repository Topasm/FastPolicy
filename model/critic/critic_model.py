import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticScorer(nn.Module):
    """
    Scores a sequence of states using an MLP.
    Input: (B, H, D_state)
    Output: (B, 1) score per sequence
    """

    def __init__(self, state_dim: int, horizon: int, hidden_dim: int = 128):
        super().__init__()
        self.state_dim = state_dim
        self.horizon = horizon
        input_dim = state_dim * horizon

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)  # Output a single score
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            state_seq: (B, H, D_state) tensor of state sequences.
        Returns:
            (B, 1) tensor of scores.
        """
        B, H, D = state_seq.shape
        if H != self.horizon or D != self.state_dim:
            raise ValueError(
                f"Input shape mismatch. Expected (B, {self.horizon}, {self.state_dim}), got {(B, H, D)}")

        # Flatten sequence: (B, H * D_state)
        state_flat = state_seq.view(B, -1)
        score = self.net(state_flat.float())
        return score

    @torch.no_grad()
    def score(self, state_seq: torch.Tensor) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        out = self.forward(state_seq)
        self.train()
        return out
