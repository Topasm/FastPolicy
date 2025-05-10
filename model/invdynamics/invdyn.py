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

        self.net = nn.Sequential(
            nn.Linear(o_dim, hidden_dim),
            _norm(),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            _norm(),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            _norm(),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            _norm(),
            nn.GELU(),

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


class SeqInvDynamic(nn.Module):
    """GRU-based sequential inverse dynamics model.

    Args:
        state_dim: int, dimension of each state vector
        action_dim: int, dimension of the action vector
        hidden_dim: int, hidden size (default: 128)
        n_layers: int, number of GRU layers (default: 1)
        dropout: float, dropout probability (default: 0.1)
        out_activation: nn.Module, activation on final layer (default: nn.Tanh())
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 hidden_dim: int = 128,
                 n_layers: int = 1,
                 dropout: float = 0.1,
                 out_activation: nn.Module = nn.Tanh(),
                 ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=state_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # break GRU’s flat‐weight buffer into independent tensors
        for name, p in list(self.gru.named_parameters()):
            # clone the data so each param has its own storage
            new_p = nn.Parameter(p.data.clone(), requires_grad=p.requires_grad)
            setattr(self.gru, name, new_p)

        # now head & weight init as before
        self.head = nn.Linear(hidden_dim, action_dim)

        # Store the output activation function
        self.out_activation = out_activation

        # Initialize head weights AFTER removing the GRU parameter manipulation
        # The self.modules() call might behave differently otherwise.
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(obs_seq)
        # Apply activation to constrain output range
        return self.out_activation(self.head(gru_out))

    @torch.no_grad()
    def predict(self, obs_seq: torch.Tensor) -> torch.Tensor:
        # same signature for inference
        return self.forward(obs_seq)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TemporalUNetInvDynamic(nn.Module):
    """
    A small UNet over (H, D) axis using Conv2d.
    Input:  (B, H, D)
    Output: (B, H, A)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # input (B, H, D) → view as (B, 1, H, D)
        self.enc1 = ConvBlock(1, hidden_dim)
        self.enc2 = ConvBlock(hidden_dim, hidden_dim * 2)
        # Use ceil_mode=True in MaxPool2d if you want to ensure output size is ceil(input_size / 2)
        # However, standard U-Net often handles mismatch by padding/cropping later.
        # downsample (H, D) → (floor(H/2), floor(D/2))
        self.pool = nn.MaxPool2d(2)

        self.up = nn.ConvTranspose2d(
            hidden_dim * 2, hidden_dim, kernel_size=2, stride=2)
        # Input channels = hid (from d1) + hid (from e1)
        self.dec1 = ConvBlock(hidden_dim * 2, hidden_dim)

        self.out_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # AdaptiveAvgPool2d might not be ideal if spatial info along D is important.
            # Consider alternative pooling or reshaping if action depends on specific state dims.
            # Let's keep it for now but be aware.
            # nn.AdaptiveAvgPool2d((None, 1)),  # (B, hid, H, D) -> (B, hid, H, 1)
            # nn.Conv2d(hidden_dim, action_dim, kernel_size=1), # (B, hid, H, 1) -> (B, A, H, 1)
            # Alternative: Use Conv2d to reduce D dimension then reshape
            # Kernel size (1, D) to collapse D dim
            nn.Conv2d(hidden_dim, action_dim, kernel_size=(1, state_dim)),
            # Output shape: (B, A, H, 1)
        )

        # Add Tanh activation layer
        self.final_activation = nn.Tanh()

        # Fallback for 2D input (B, D)
        self.step_head = nn.Sequential(
            nn.Linear(state_dim, action_dim),
            nn.Tanh()  # Also add Tanh here for consistency
        )
        # Initialize linear layer
        nn.init.kaiming_normal_(self.step_head[0].weight, nonlinearity="relu")
        nn.init.zeros_(self.step_head[0].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            # Handle single step input (B, D)
            return self.step_head(x)  # step_head now includes Tanh
        elif x.dim() == 3:
            # Handle sequence input (B, H, D)
            B, H, D = x.shape
            x = x.unsqueeze(1)  # Add channel dim: (B, 1, H, D)
        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        e1 = self.enc1(x)           # (B, hid, H, D)
        e2 = self.enc2(self.pool(e1))  # (B, hid*2, floor(H/2), floor(D/2))

        # (B, hid, H_up, D_up) where H_up/D_up might not equal H/D if H/D were odd
        d1 = self.up(e2)

        # Pad d1 to match the spatial dimensions of e1 before concatenation
        pad_h = e1.size(2) - d1.size(2)  # Difference in H dimension
        pad_d = e1.size(3) - d1.size(3)  # Difference in D dimension
        # Apply padding: (pad_D_left, pad_D_right, pad_H_left, pad_H_right)
        # We pad right and bottom to compensate for floor division in pooling
        d1_padded = F.pad(d1, (0, pad_d, 0, pad_h))

        # Concatenate along the channel dimension (dim=1)
        cat = torch.cat([d1_padded, e1], dim=1)  # (B, hid + hid, H, D)
        out = self.dec1(cat)        # (B, hid, H, D)

        # Adjust output head if AdaptiveAvgPool was removed
        # Assuming state_dim was passed correctly and matches D
        if not isinstance(self.out_head[2], nn.AdaptiveAvgPool2d):
            # If using Conv2d(kernel_size=(1, D))
            out = self.out_head(out)  # (B, A, H, 1)
        else:
            # Original logic with AdaptiveAvgPool
            out = self.out_head(out)  # (B, A, H, 1)

        out = out.squeeze(-1)       # (B, A, H)
        out = out.permute(0, 2, 1)  # (B, H, A) - Match desired output shape

        # Apply final activation
        out = self.final_activation(out)

        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()  # Ensure model is in eval mode for prediction
        out = self.forward(x)
        self.train()  # Set back to train mode if necessary elsewhere
        return out
