import torch

from ..diffusion.sampler import TrajectorySampler
from ..critic.critic_model import CriticScorer
from ..invdynamics.invdyn import MlpInvDynamic


class DiffusionPlanner:
    """
    Runs sampling of candidate trajectories via diffusion and selects the best via critic scoring.
    """

    def __init__(self,
                 sampler: TrajectorySampler,
                 scorer: CriticScorer,
                 inv_dyn: MlpInvDynamic,
                 n_samples: int = 8):
        self.sampler = sampler
        self.scorer = scorer
        self.inv_dyn = inv_dyn
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

    def plan(self, s0: torch.Tensor, goal: torch.Tensor = None):
        """
        Full planning loop: sample candidates, score, and return best
        """
        candidates = self.sample_trajectories(s0, goal)
        best = self.select_best(candidates)  # [horizon/stride, state_dim]
        # predict actions between successive jumped states
        actions = []
        for i in range(best.shape[0] - 1):
            a = self.inv_dyn(best[i], best[i+1])
            actions.append(a)
        actions = torch.stack(actions, dim=0)  # [T-1, action_dim]
        return best, actions
