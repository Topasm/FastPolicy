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
    """MLP-based inverse dynamics model matching the workspace structure.

    Args:
        o_dim: int, dimension of each state vector
        a_dim: int, dimension of the action vector
        hidden_dim: int, hidden size (default: 512)
        dropout: float, dropout probability (default: 0.1)
        use_layernorm: bool, whether to apply LayerNorm after each hidden layer (default: True)
        out_activation: nn.Module, activation on final layer (default: nn.Tanh())
    """

    def __init__(
        self,
        o_dim: int,
        a_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        out_activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.o_dim = o_dim
        self.a_dim = a_dim

        def _norm():
            return nn.LayerNorm(hidden_dim) if use_layernorm else nn.Identity()

        # Now only take o_dim, not 2*o_dim
        self.net = nn.Sequential(
            nn.Linear(o_dim, hidden_dim),
            _norm(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            _norm(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, a_dim),
            out_activation,
        )
        self._init_weights()

    def _init_weights(self):
        # Initialize weights similar to the workspace version
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                # Use Kaiming Normal for layers followed by GELU (approximated with 'relu')
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, o: torch.Tensor) -> torch.Tensor:
        """Predict action from current state only."""
        return self.net(o.float())

    @torch.no_grad()
    def predict(self, o: torch.Tensor) -> torch.Tensor:
        """Inference entrypoint."""
        self.eval()
        out = self.forward(o)
        self.train()
        return out
