import torch

from ..diffusion.sampler import TrajectorySampler
from ..critic.critic_model import CriticScorer


class DiffusionPlanner:
    """
    Runs sampling of candidate trajectories via diffusion and selects the best via critic scoring.
    """

    def __init__(self,
                 sampler: TrajectorySampler,
                 scorer: CriticScorer,
                 n_samples: int = 8):
        self.sampler = sampler
        self.scorer = scorer
        self.n_samples = n_samples

    def sample_trajectories(self, s0: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """
        Generate N candidate trajectories using the sampler
        """
        return self.sampler.sample(s0, self.n_samples, goal)

    def select_best(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Score trajectories using critic and return best trajectory
        """
        scores = self.scorer.score(trajectories)
        best_idx = torch.argmax(scores)
        return trajectories[best_idx]

    def plan(self, s0: torch.Tensor, goal: torch.Tensor = None) -> torch.Tensor:
        """
        Full planning loop: sample candidates, score, and return best
        """
        candidates = self.sample_trajectories(s0, goal)
        best = self.select_best(candidates)
        return best
