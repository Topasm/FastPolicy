import torch
import yaml

from model.diffusion.model import DiffusionModel
from model.diffusion.sampler import TrajectorySampler
from model.critic.critic_model import CriticScorer
from model.planner.diffusion_planner import DiffusionPlanner


def main():
    # load configuration
    with open("configs/configs.yaml", "r") as f:
        conf = yaml.safe_load(f)

    # build diffusion model
    dm_cfg = conf["diffusion_model"]
    diffusion_model = DiffusionModel(
        state_dim=dm_cfg["state_dim"],
        hidden_dim=dm_cfg["hidden_dim"],
        num_layers=dm_cfg["num_layers"],
        nhead=dm_cfg["nhead"],
    )

    # build sampler
    sp_cfg = conf["sampler"]
    sampler = TrajectorySampler(
        diffusion_model,
        horizon=sp_cfg["horizon"],
        stride=sp_cfg["stride"],
    )

    # build critic scorer
    sc_cfg = conf["critic"]
    scorer = CriticScorer(
        model_path=sc_cfg["model_path"],
        state_dim=sc_cfg["state_dim"],
        horizon=sc_cfg["horizon"],
        hidden_dim=sc_cfg["hidden_dim"],
        device=sc_cfg["device"],
    )

    # build planner
    pl_cfg = conf["planner"]
    planner = DiffusionPlanner(sampler, scorer, n_samples=pl_cfg["n_samples"])

    # example initial state and optional goal
    s0 = torch.zeros(dm_cfg["state_dim"])
    goal = None

    # run planning
    best_traj = planner.plan(s0, goal)
    print("Best trajectory shape:", best_traj.shape)
    print("First state:", best_traj[0])


if __name__ == "__main__":
    main()
