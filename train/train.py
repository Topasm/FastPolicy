import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.libero import LiberoDataset
from model.diffusion.model import DiffusionModel
from model.invdynamics.invdyn import MlpInvDynamic


def train():
    # load config
    with open("configs/configs.yaml", "r") as f:
        conf = yaml.safe_load(f)
    # training hyperparams
    tr_cfg = conf.get("training", {})
    batch_size = tr_cfg.get("batch_size", 64)
    diffusion_lr = tr_cfg.get("diffusion_lr", 1e-3)
    invdyn_lr = tr_cfg.get("invdyn_lr", 5e-4)
    epochs = tr_cfg.get("epochs", 10)

    device = torch.device(conf.get("device", "cpu"))

    # dataset and loader
    data_path = tr_cfg.get("data_path", "data/libero.zarr")
    dataset = LiberoDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # diffusion model and optimizer
    dm_cfg = conf["diffusion_model"]
    diffusion_model = DiffusionModel(
        state_dim=dm_cfg["state_dim"],
        hidden_dim=dm_cfg["hidden_dim"],
        num_layers=dm_cfg["num_layers"],
    ).to(device)
    diffusion_opt = torch.optim.Adam(
        diffusion_model.parameters(), lr=diffusion_lr)
    mse_loss = nn.MSELoss()

    # inverse dynamics model and optimizer
    inv_cfg = conf["invdynamics"]
    inv_dyn = MlpInvDynamic(
        o_dim=inv_cfg["o_dim"],
        a_dim=inv_cfg["a_dim"],
        hidden_dim=inv_cfg["hidden_dim"],
        device=device,
    ).to(device)
    inv_opt = torch.optim.Adam(inv_dyn.mlp.parameters(), lr=invdyn_lr)

    # training loop
    for ep in range(1, epochs+1):
        total_diff_loss = 0.0
        total_inv_loss = 0.0
        for batch in loader:
            # observations: dict of arrays; use 'states'
            states = torch.tensor(
                batch['observation']['states'], dtype=torch.float32, device=device)
            # shape [B, T, state_dim]
            B, T, D = states.shape
            # diffusion training
            noise = torch.randn_like(states)
            x_noisy = states + noise
            timesteps = torch.zeros(B, dtype=torch.long, device=device)
            pred_noise = diffusion_model(x_noisy, timesteps)
            loss_diff = mse_loss(pred_noise, noise)

            diffusion_opt.zero_grad()
            loss_diff.backward()
            diffusion_opt.step()

            # inverse dynamics training: predict actions between states
            # flatten pairs
            o = states[:, :-1, :].reshape(-1, D)
            o_next = states[:, 1:, :].reshape(-1, D)
            # ground truth actions: use dataset action sequence
            actions = torch.tensor(
                batch['action'], dtype=torch.float32, device=device)
            # assume actions shape [B, T, action_dim]
            A = actions.shape[-1]
            # align to T-1
            a_gt = actions[:, :T-1, :].reshape(-1, A)
            inv_pred = inv_dyn.forward(o, o_next)
            loss_inv = mse_loss(inv_pred, a_gt)

            inv_opt.zero_grad()
            loss_inv.backward()
            inv_opt.step()

            total_diff_loss += loss_diff.item() * B
            total_inv_loss += loss_inv.item() * B

        n = len(dataset)
        print(
            f"Epoch {ep}/{epochs}: diffusion_loss={total_diff_loss/n:.4f}, invdyn_loss={total_inv_loss/n:.4f}")

    # save models
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(diffusion_model.state_dict(), "checkpoints/diffusion_model.pt")
    inv_dyn.save("checkpoints/inv_dyn.pt")


if __name__ == "__main__":
    train()
