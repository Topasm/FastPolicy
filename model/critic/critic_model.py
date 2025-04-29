import torch
import torch.nn as nn


class CriticMLP(nn.Module):
    """
    Simple MLP to score flattened trajectories.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticScorer:
    """
    Loads a trained critic model and scores candidate trajectories.
    """

    def __init__(
        self,
        model_path: str,
        state_dim: int,
        horizon: int,
        hidden_dim: int = 128,
        device: str = "cpu"
    ):
        self.device = device
        input_dim = state_dim * horizon
        self.model = CriticMLP(input_dim, hidden_dim).to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.eval()

    def score(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        trajectories: [N, horizon, state_dim]
        returns: [N] critic scores
        """
        N = trajectories.shape[0]
        flat = trajectories.view(N, -1).to(self.device)
        with torch.no_grad():
            vals = self.model(flat).squeeze(-1)
        return vals
