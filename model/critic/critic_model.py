import torch
import torch.nn as nn


class CriticScorer(nn.Module):  # <-- Make sure it inherits from nn.Module
    def __init__(self, state_dim: int, horizon: int, hidden_dim: int = 256):
        super().__init__()  # <-- Add super().__init__() call
        self.horizon = horizon
        self.state_dim = state_dim

        # Example MLP structure - adjust as needed
        self.net = nn.Sequential(
            nn.Linear(state_dim * horizon, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)  # Output a single score
        )

    def score(self, state_sequence: torch.Tensor) -> torch.Tensor:
        """
        Scores a sequence of predicted states.

        Args:
            state_sequence: Tensor of shape (num_samples, horizon, state_dim)

        Returns:
            Tensor of shape (num_samples,) containing scores for each sequence.
        """
        num_samples, horizon, state_dim = state_sequence.shape
        assert horizon == self.horizon
        assert state_dim == self.state_dim

        # Flatten the sequence for the MLP
        flat_sequence = state_sequence.view(num_samples, -1)
        # Remove the last dimension
        scores = self.net(flat_sequence).squeeze(-1)
        return scores

    # Add a forward method for consistency, although score is used directly
    def forward(self, state_sequence: torch.Tensor) -> torch.Tensor:
        return self.score(state_sequence)
