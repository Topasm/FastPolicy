import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, TransformedDistribution


def get_default_init_scale(weight_tensor, scale=1.0):
    """Custom weight initialization logic similar to OGB code"""
    fan_in = weight_tensor.shape[1]
    bound = 1.0 / (fan_in ** 0.5) * scale
    return -bound, bound


def ogb_get_default_init_unif_a(weight_tensor, scale=1.0):
    """Computes the bound for uniform initialization as in OGB code.

    Args:
        weight_tensor: The weight tensor to initialize
        scale: Scaling factor for the bound (default: 1.0)

    Returns:
        float: The bound value for uniform initialization
    """
    fan_in = weight_tensor.shape[1]  # Input dimension
    return 1.0 / (fan_in ** 0.5) * scale


class CustomMLP(nn.Module):
    """Customizable MLP backbone similar to OgB_MLP."""

    def __init__(
        self,
        input_dim,
        hidden_dims,
        output_dim=None,
        activate_final=True,
        activation=nn.ReLU(),
        use_layer_norm=False,
        dropout_prob=0.0,
        init_scale=1.0
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))

            if use_layer_norm:
                layers.append(nn.LayerNorm(h_dim))

            layers.append(activation)

            if dropout_prob > 0:
                layers.append(nn.Dropout(dropout_prob))

            prev_dim = h_dim

        if output_dim is not None:
            self.output_dim = output_dim
            self.layers = nn.Sequential(*layers)
            self.final_layer = nn.Linear(prev_dim, output_dim)
            self.has_final = True
            self.activate_final = activate_final
            if activate_final:
                self.final_activation = activation
        else:
            self.output_dim = prev_dim
            self.layers = nn.Sequential(*layers)
            self.has_final = False

        # Initialize weights with custom logic
        self._init_weights(init_scale)

    def _init_weights(self, scale=1.0):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                a, b = get_default_init_scale(m.weight, scale=scale)
                nn.init.uniform_(m.weight, a=a, b=b)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        features = self.layers(x)

        if self.has_final:
            output = self.final_layer(features)
            if self.activate_final:
                output = self.final_activation(output)
            return output
        else:
            return features


class EnhancedInvDynamic(nn.Module):
    """Enhanced inverse dynamics model with features from OGB.

    This model takes a current state and next state to predict the action.
    Features include:
    - Optional probabilistic output
    - Temperature scaling
    - Customizable architecture
    - Separate goal encoding (next state as goal)

    Args:
        state_dim: Dimension of the state vector
        action_dim: Dimension of the action vector
        hidden_dims: List of hidden dimensions
        use_state_encoding: Whether to encode the next state separately
        is_probabilistic: Whether to output a probability distribution
        temperature: Temperature parameter for sampling (lower = more deterministic)
        dropout: Dropout probability
        use_layernorm: Whether to use layer normalization
        final_init_scale: Scale factor for final layer initialization
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list = [512, 512, 512, 512],
        use_state_encoding: bool = False,
        is_probabilistic: bool = False,
        temperature: float = 0.1,
        train_temp: float = 1.0,
        eval_temp: float = 0.0,
        dropout: float = 0.1,
        use_layernorm: bool = True,
        final_init_scale: float = 1e-2,
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        out_activation: nn.Module = nn.Tanh(),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.is_probabilistic = is_probabilistic
        self.temperature = temperature
        self.train_temp = train_temp
        self.eval_temp = eval_temp
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.out_activation = out_activation

        # If we encode the next state separately
        if use_state_encoding:
            # Create a goal encoder for the next state
            self.goal_encoder = CustomMLP(
                input_dim=state_dim,
                hidden_dims=hidden_dims[:2],  # Use part of the hidden dims
                output_dim=hidden_dims[0],    # Output a hidden representation
                activate_final=True,
                activation=nn.GELU(),
                use_layer_norm=use_layernorm,
                dropout_prob=dropout
            )
            # Main network takes current state + encoded next state
            input_dim = state_dim + self.goal_encoder.output_dim
        else:
            self.goal_encoder = None
            # Main network takes concatenated current and next states
            input_dim = state_dim * 2

        # Main feature network
        self.feature_net = CustomMLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=None,  # Use the last hidden dim as output
            activate_final=True,
            activation=nn.GELU(),
            use_layer_norm=use_layernorm,
            dropout_prob=dropout
        )

        # Action mean network
        self.mean_net = nn.Linear(self.feature_net.output_dim, action_dim)
        # Custom initialization for final layer
        a, b = get_default_init_scale(
            self.mean_net.weight, scale=final_init_scale)
        nn.init.uniform_(self.mean_net.weight, a=a, b=b)
        nn.init.zeros_(self.mean_net.bias)

        # Log standard deviation network (only used if probabilistic)
        if is_probabilistic:
            self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, curr_state: torch.Tensor, next_state: torch.Tensor, temperature=None) -> torch.Tensor:
        """
        Forward pass to compute action from current and next states.

        Args:
            curr_state: Current state tensor (B, state_dim)
            next_state: Next state tensor (B, state_dim)
            temperature: Optional temperature override (default: self.temperature)

        Returns:
            If is_probabilistic=False: Action tensor (B, action_dim)
            If is_probabilistic=True: Distribution over actions
        """
        # Apply the goal encoder if it exists
        if self.goal_encoder is not None:
            # Encode the next state
            next_state_encoded = self.goal_encoder(next_state)
            # Concatenate current state and encoded next state
            features_input = torch.cat(
                [curr_state, next_state_encoded], dim=-1)
        else:
            # Simply concatenate current and next states
            features_input = torch.cat([curr_state, next_state], dim=-1)

        # Get features from the main network
        features = self.feature_net(features_input)

        # Compute action mean
        mean = self.mean_net(features)

        # Apply output activation if not probabilistic
        if not self.is_probabilistic:
            return self.out_activation(mean)

        # For probabilistic output, create a distribution
        temperature = temperature if temperature is not None else self.temperature
        if self.training:
            temperature = temperature * self.train_temp
        else:
            temperature = temperature * self.eval_temp

        # Clamp log standard deviations
        log_std = torch.clamp(self.log_std, self.log_std_min, self.log_std_max)

        # Create normal distribution
        base_dist = Normal(loc=mean, scale=torch.exp(log_std) * temperature)
        dist = Independent(base_dist, 1)

        return dist

    def loss(self, curr_state: torch.Tensor, next_state: torch.Tensor,
             target_action: torch.Tensor) -> tuple:
        """
        Compute loss for training.

        Args:
            curr_state: Current state tensor (B, state_dim)
            next_state: Next state tensor (B, state_dim)
            target_action: Target action tensor (B, action_dim)

        Returns:
            tuple: (loss, info_dict)
        """
        if self.is_probabilistic:
            # Get distribution over actions
            pred_dist = self.forward(curr_state, next_state)
            # Compute negative log likelihood
            log_prob = pred_dist.log_prob(target_action)
            loss = -log_prob.mean()
            # Save the scalar value, but keep 'loss' as a tensor
            info = {"neg_log_likelihood": loss.item()}
        else:
            # Get predicted action
            pred_action = self.forward(curr_state, next_state)
            # Compute MSE loss
            loss = F.mse_loss(pred_action, target_action)
            # Save the scalar value, but keep 'loss' as a tensor
            info = {"mse": loss.item()}

        return loss, info

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference entrypoint. Assumes x is a concatenation of [curr_state, next_state].

        Args:
            x: Input tensor with shape [B, state_dim*2] or [B, T, state_dim*2]

        Returns:
            torch.Tensor: Predicted action
        """
        self.eval()

        # Handle different input formats
        if x.dim() == 2:
            # Split concatenated states
            half = x.shape[-1] // 2
            curr_state = x[..., :half]
            next_state = x[..., half:]

            if self.is_probabilistic:
                # Use deterministic mode (temperature=0) for inference
                dist = self.forward(curr_state, next_state,
                                    temperature=self.eval_temp)
                action = dist.mean
            else:
                action = self.forward(curr_state, next_state)

        elif x.dim() == 3:
            # Handle sequence data (B, T, state_dim*2)
            B, T, _ = x.shape
            half = x.shape[-1] // 2

            # Split along the feature dimension and keep sequence dim
            curr_state = x[..., :half]
            next_state = x[..., half:]

            # Process the sequence
            actions = []
            for t in range(T):
                if self.is_probabilistic:
                    dist = self.forward(
                        curr_state[:, t], next_state[:, t], temperature=self.eval_temp)
                    action_t = dist.mean
                else:
                    action_t = self.forward(curr_state[:, t], next_state[:, t])
                actions.append(action_t)

            # Stack along sequence dimension
            action = torch.stack(actions, dim=1)

        else:
            raise ValueError(f"Unsupported input dimension: {x.dim()}")

        self.train()
        return action

    def sample(self, curr_state: torch.Tensor, next_state: torch.Tensor,
               temperature: float = None) -> torch.Tensor:
        """
        Sample actions from the model.

        Args:
            curr_state: Current state tensor
            next_state: Next state tensor
            temperature: Temperature for sampling (None to use default)

        Returns:
            torch.Tensor: Sampled action
        """
        if not self.is_probabilistic:
            # If not probabilistic, just return deterministic output
            return self.forward(curr_state, next_state)

        # Get distribution with appropriate temperature
        dist = self.forward(curr_state, next_state, temperature=temperature)

        # Sample from distribution
        return dist.sample()
