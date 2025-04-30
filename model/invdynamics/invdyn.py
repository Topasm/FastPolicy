import torch
import torch.nn as nn
import torch.nn.functional as F


class Mlp(nn.Module):
    """Simple MLP backbone."""

    def __init__(self, in_dim, hidden_dims, out_dim, hidden_act, out_act):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last, h))
            layers.append(hidden_act)
            last = h
        layers.append(nn.Linear(last, out_dim))
        layers.append(out_act)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class MlpInvDynamic(nn.Module):
    """ Simple MLP-based inverse dynamics model. Predicts action from state pair.

    Args:
        o_dim: int, Dimension of observation/state space.
        a_dim: int, Dimension of action space.
        hidden_dim: int, Dimension of hidden layers. Default: 512.
        out_activation: nn.Module, Activation function for output layer. Default: nn.Tanh().
    """

    def __init__(
            self,
            o_dim: int,
            a_dim: int,
            hidden_dim: int = 512,
            out_activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.o_dim, self.a_dim, self.hidden_dim = o_dim, a_dim, hidden_dim
        self.out_activation = out_activation

        self.mlp = Mlp(
            2 * o_dim, [hidden_dim, hidden_dim], a_dim,
            nn.ReLU(), out_activation)
        self._init_weights()

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, o: torch.Tensor, o_next: torch.Tensor) -> torch.Tensor:
        """Predicts action given current and next state.

        Args:
            o: Tensor (B, ..., o_dim), current state(s).
            o_next: Tensor (B, ..., o_dim), next state(s).

        Returns:
            Tensor (B, ..., a_dim), predicted action(s).
        """
        o = o.float()
        o_next = o_next.float()
        input_features = torch.cat([o, o_next], dim=-1)
        return self.mlp(input_features)

    @torch.no_grad()
    def predict(self, o: torch.Tensor, o_next: torch.Tensor) -> torch.Tensor:
        """Alias for forward, used for clarity during inference."""
        return self.forward(o, o_next)
