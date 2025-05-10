"""
Helper functions for using the enhanced inverse dynamics model in evaluation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union
import logging

# Import the enhanced model
from model.invdynamics.enhanced_invdyn import EnhancedInvDynamic


def generate_actions_with_enhanced_invdyn(
    diffusion_model,
    inv_dyn_model,
    norm_batch,
    norm_current_state,
    num_inference_samples=1,
    temperature=None,
    use_probabilistic_sampling=False
):
    """
    Generate actions using diffusion model for state prediction and enhanced 
    inverse dynamics for action prediction.

    Args:
        diffusion_model: The diffusion model instance
        inv_dyn_model: EnhancedInvDynamic model instance
        norm_batch: Normalized batch dictionary with observation history
        norm_current_state: Current normalized state
        num_inference_samples: Number of future trajectories to sample
        temperature: Optional temperature override for sampling
        use_probabilistic_sampling: Whether to use stochastic sampling if model is probabilistic

    Returns:
        Predicted actions
    """
    device = next(inv_dyn_model.parameters()).device

    # Prepare global conditioning from the normalized batch
    global_cond = diffusion_model._prepare_global_conditioning(norm_batch)

    # Sample normalized future states using conditional_sample
    batch_size = norm_current_state.shape[0]

    # Repeat global_cond if num_inference_samples > 1
    if num_inference_samples > 1:
        global_cond = global_cond.repeat_interleave(
            num_inference_samples, dim=0)

    # Sample normalized future states using conditional_sample
    predicted_states_flat = diffusion_model.conditional_sample(
        batch_size * num_inference_samples, global_cond=global_cond
    )  # Output: (B*N, horizon, D_state)

    # Reshape the states if multiple samples were generated
    if num_inference_samples > 1:
        import einops
        predicted_states = einops.rearrange(
            predicted_states_flat, "(b n) h d -> b n h d",
            b=batch_size, n=num_inference_samples
        )
        # Extract the first future state (t=1) for each sample
        # Shape: [batch, num_samples, state_dim]
        future_state = predicted_states[:, :, 0, :]
    else:
        # Just extract the first future state (t=1)
        future_state = predicted_states_flat[:, 0, :].unsqueeze(
            1)  # Shape: [batch, 1, state_dim]

    # If we have multiple samples, we need to handle them
    if num_inference_samples > 1:
        # Reshape to [batch*samples, state_dim]
        batch_size = norm_current_state.shape[0]
        future_state = future_state.reshape(-1, future_state.shape[-1])
        norm_current_state = norm_current_state.repeat_interleave(
            num_inference_samples, dim=0)

    # Use the enhanced model to predict actions
    if (use_probabilistic_sampling and
        hasattr(inv_dyn_model, 'is_probabilistic') and
            inv_dyn_model.is_probabilistic):
        # Use stochastic sampling if available
        actions = inv_dyn_model.sample(
            norm_current_state, future_state, temperature=temperature)
    else:
        # For deterministic prediction, pass concatenated states if not EnhancedInvDynamic
        if isinstance(inv_dyn_model, EnhancedInvDynamic):
            # Use the model's specific API
            if hasattr(inv_dyn_model, 'forward'):
                actions = inv_dyn_model(norm_current_state, future_state)
                if hasattr(actions, 'mean'):  # Handle distribution output
                    actions = actions.mean
            else:
                # Fall back to predict method
                actions = inv_dyn_model.predict(
                    torch.cat([norm_current_state, future_state], dim=-1))
        else:
            # For other models, use the predict method with concatenated states
            actions = inv_dyn_model.predict(
                torch.cat([norm_current_state, future_state], dim=-1))

    # Reshape actions back to [batch, num_samples, action_dim] if needed
    if num_inference_samples > 1:
        actions = actions.reshape(batch_size, num_inference_samples, -1)

    return actions


def load_enhanced_invdyn_model(
    model_path: str,
    state_dim: int,
    action_dim: int,
    device: torch.device = None,
    hidden_dims: List[int] = None,
    use_probabilistic: bool = False,
    debug: bool = False
) -> EnhancedInvDynamic:
    """
    Load an enhanced inverse dynamics model from a checkpoint.

    Args:
        model_path: Path to the model checkpoint
        state_dim: Dimension of the state
        action_dim: Dimension of the action
        device: Device to load the model on
        hidden_dims: Hidden dimensions for the model
        use_probabilistic: Whether to create a probabilistic model
        debug: Whether to print debug information

    Returns:
        Loaded enhanced inverse dynamics model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if hidden_dims is None:
        hidden_dims = [512, 512, 512, 512]

    # Create model instance
    model = EnhancedInvDynamic(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        is_probabilistic=use_probabilistic,
        use_state_encoding=True,  # Recommended setting
        temperature=0.1,  # Default temperature
        out_activation=nn.Tanh(),
    )

    # Load weights
    try:
        state_dict = torch.load(model_path, map_location="cpu")

        if debug:
            print("\n--- Model Loading Debug ---")
            print("Model parameter keys:")
            for name, _ in model.named_parameters():
                print(f"  {name}")
            print("\nState dict keys:")
            for key in state_dict.keys():
                print(f"  {key}")
            print("---------------------------\n")

        model.load_state_dict(state_dict)
        print(f"Successfully loaded model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Using initialized model")

    model.to(device)
    model.eval()  # Set to evaluation mode

    return model
