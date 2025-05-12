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
    use_probabilistic_sampling=False,
    generate_action_sequence=True
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
        generate_action_sequence: If True, generate actions for the entire trajectory,
                                 otherwise just for the first future state (original behavior)

    Returns:
        Predicted actions. The shape depends on the value of generate_action_sequence:
        - If True: [batch, num_samples, horizon-1, action_dim] or [batch, horizon-1, action_dim]
        - If False: [batch, num_samples, action_dim] or [batch, action_dim]
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

    # Debug information about predicted states
    print(
        f"Generated predicted states with shape: {predicted_states_flat.shape}")

    # Reshape the states if multiple samples were generated
    if num_inference_samples > 1:
        import einops
        predicted_states = einops.rearrange(
            predicted_states_flat, "(b n) h d -> b n h d",
            b=batch_size, n=num_inference_samples
        )

        # Store the full trajectory for action sequence generation
        # Shape: [batch, num_samples, horizon, state_dim]
        all_future_states = predicted_states

        # For backward compatibility, also extract just the first future state
        # Shape: [batch, num_samples, state_dim]
        first_future_state = predicted_states[:, :, 1, :]
    else:
        # Store the full trajectory for action sequence generation
        # Shape: [batch, horizon, state_dim]
        all_future_states = predicted_states_flat

        # Just extract the first future state (t=1)
        # Shape: [batch, state_dim]
        first_future_state = predicted_states_flat[:, 1, :]

    # If we have multiple samples, we need to handle them
    if num_inference_samples > 1:
        # Reshape to [batch*samples, state_dim] for initial action prediction
        future_state = first_future_state.reshape(
            -1, first_future_state.shape[-1])

    # Determine the action dimension
    action_dim = 2  # Default for PushT environment

    # Try to infer action dimension from the model
    if hasattr(inv_dyn_model, 'action_dim'):
        action_dim = inv_dyn_model.action_dim
    elif hasattr(inv_dyn_model, 'a_dim'):
        action_dim = inv_dyn_model.a_dim
    else:
        try:
            if hasattr(inv_dyn_model, 'out_layer'):
                action_dim = inv_dyn_model.out_layer.out_features
            elif hasattr(inv_dyn_model, 'output_layer'):
                action_dim = inv_dyn_model.output_layer.out_features
        except Exception as e:
            print(
                f"WARNING: Could not determine action dimension from model attributes: {e}")
            print("Using default action dimension of 2 for PushT environment")

    print(f"Using action dimension: {action_dim}")

    if generate_action_sequence:
        print("Generating action sequence for the full trajectory")

        if num_inference_samples > 1:
            # Multiple samples case
            # Create storage for action sequences: [batch, num_samples, horizon-1, action_dim]
            # Set to a fixed horizon length to handle unexpected dimensions
            # PushT typically uses horizon=16
            action_horizon = min(15, all_future_states.shape[2]-1)
            action_sequences = torch.zeros(
                batch_size, num_inference_samples, action_horizon, action_dim,
                device=device
            )

            # For each sample, generate actions between consecutive states
            for b in range(batch_size):
                for s in range(num_inference_samples):
                    # Get trajectory for this sample: [horizon, state_dim]
                    trajectory = all_future_states[b, s]

                    # Start with the current state
                    current = norm_current_state[b].unsqueeze(
                        0)  # [1, state_dim]

                    # Generate actions for each pair of states
                    for t in range(min(action_horizon, trajectory.shape[0]-1)):
                        try:
                            # Get the next state from the trajectory (t+1 for more logical flow)
                            # [1, state_dim]
                            next_state = trajectory[t+1].unsqueeze(0)

                            # Debug state shapes
                            print(
                                f"Multi-sample batch {b}, sample {s}, step {t}:")
                            print(f"  - Current state: {current.shape}")
                            print(f"  - Next state: {next_state.shape}")

                            # Predict action from current to next state
                            if use_probabilistic_sampling and hasattr(inv_dyn_model, 'is_probabilistic') and inv_dyn_model.is_probabilistic:
                                action = inv_dyn_model.sample(
                                    current, next_state, temperature=temperature)
                            else:
                                if isinstance(inv_dyn_model, EnhancedInvDynamic):
                                    if hasattr(inv_dyn_model, 'forward'):
                                        action = inv_dyn_model(
                                            current, next_state)
                                        if hasattr(action, 'mean'):
                                            action = action.mean
                                    else:
                                        # Combine states for traditional inverse dynamics
                                        combined = torch.cat(
                                            [current, next_state], dim=-1)
                                        action = inv_dyn_model.predict(
                                            combined)
                                else:
                                    # Standard MLP model
                                    combined = torch.cat(
                                        [current, next_state], dim=-1)
                                    action = inv_dyn_model.predict(combined)

                            # Debug action shape
                            print(f"  - Raw action shape: {action.shape}")

                            # Process action shape
                            if len(action.shape) > 1:
                                action = action.squeeze(0)
                                print(
                                    f"  - Squeezed action shape: {action.shape}")

                            # Handle dimension mismatch
                            target_shape = action_sequences[b, s, t].shape
                            if action.shape[0] != target_shape[0]:
                                print(
                                    f"  - WARNING: Action dimension mismatch - got {action.shape[0]}, expected {target_shape[0]}")
                                if action.shape[0] > target_shape[0]:
                                    # Truncate
                                    action = action[:target_shape[0]]
                                else:
                                    # Pad with zeros
                                    padded = torch.zeros_like(
                                        action_sequences[b, s, t])
                                    padded[:action.shape[0]] = action
                                    action = padded

                            # Store the action
                            action_sequences[b, s, t] = action

                            # Update current state for next iteration
                            current = next_state

                        except Exception as e:
                            print(
                                f"Error in multi-sample processing at b={b}, s={s}, t={t}: {e}")
                            # Use zeros as fallback
                            action_sequences[b, s, t] = torch.zeros_like(
                                action_sequences[b, s, t])

            # Return the action sequences
            actions = action_sequences

        else:
            # Single sample case
            # Create storage for action sequence: [batch, horizon-1, action_dim]
            # Set to a fixed horizon length to handle unexpected dimensions
            # PushT typically uses horizon=16
            action_horizon = min(15, all_future_states.shape[1]-1)
            action_sequence = torch.zeros(
                batch_size, action_horizon, action_dim,
                device=device
            )

            # For each batch element, generate actions between consecutive states
            for b in range(batch_size):
                # Get trajectory for this batch
                trajectory = all_future_states[b]

                # Start with the current state
                current = norm_current_state[b].unsqueeze(0)  # [1, state_dim]

                # Generate actions for each pair of states
                for t in range(min(action_horizon, trajectory.shape[0]-1)):
                    try:
                        # Get the next state from the trajectory (t+1 for more logical flow)
                        # [1, state_dim]
                        next_state = trajectory[t+1].unsqueeze(0)

                        # Debug state shapes
                        print(f"Single-sample batch {b}, step {t}:")
                        print(f"  - Current state: {current.shape}")
                        print(f"  - Next state: {next_state.shape}")

                        # Predict action from current to next state
                        if use_probabilistic_sampling and hasattr(inv_dyn_model, 'is_probabilistic') and inv_dyn_model.is_probabilistic:
                            action = inv_dyn_model.sample(
                                current, next_state, temperature=temperature)
                        else:
                            if isinstance(inv_dyn_model, EnhancedInvDynamic):
                                if hasattr(inv_dyn_model, 'forward'):
                                    action = inv_dyn_model(current, next_state)
                                    if hasattr(action, 'mean'):
                                        action = action.mean
                                else:
                                    # Combine states for traditional inverse dynamics
                                    combined = torch.cat(
                                        [current, next_state], dim=-1)
                                    action = inv_dyn_model.predict(combined)
                            else:
                                # Standard MLP model
                                combined = torch.cat(
                                    [current, next_state], dim=-1)
                                action = inv_dyn_model.predict(combined)

                        # Debug action shape
                        print(f"  - Raw action shape: {action.shape}")

                        # Process action shape
                        if len(action.shape) > 1:
                            action = action.squeeze(0)
                            print(f"  - Squeezed action shape: {action.shape}")

                        # Handle dimension mismatch
                        target_shape = action_sequence[b, t].shape
                        if action.shape[0] != target_shape[0]:
                            print(
                                f"  - WARNING: Action dimension mismatch - got {action.shape[0]}, expected {target_shape[0]}")
                            if action.shape[0] > target_shape[0]:
                                # Truncate
                                action = action[:target_shape[0]]
                            else:
                                # Pad with zeros
                                padded = torch.zeros_like(
                                    action_sequence[b, t])
                                padded[:action.shape[0]] = action
                                action = padded

                        # Store the action
                        action_sequence[b, t] = action

                        # Update current state for next iteration
                        current = next_state

                    except Exception as e:
                        print(
                            f"Error in single-sample processing at b={b}, t={t}: {e}")
                        # Use zeros as fallback
                        action_sequence[b, t] = torch.zeros_like(
                            action_sequence[b, t])

            # Return the action sequence
            actions = action_sequence

    else:
        # Original behavior: just predict the first action using the first future state
        if num_inference_samples > 1:
            # Use the expanded current state for multiple samples
            current_state_for_pred = norm_current_state.repeat_interleave(
                num_inference_samples, dim=0)
        else:
            current_state_for_pred = norm_current_state

        # Use the enhanced model to predict actions
        if (use_probabilistic_sampling and
            hasattr(inv_dyn_model, 'is_probabilistic') and
                inv_dyn_model.is_probabilistic):
            # Use stochastic sampling if available
            actions = inv_dyn_model.sample(
                current_state_for_pred, future_state, temperature=temperature)
        else:
            # For deterministic prediction, handle models differently
            if isinstance(inv_dyn_model, EnhancedInvDynamic):
                # Use the model's specific API that takes separate states
                if hasattr(inv_dyn_model, 'forward'):
                    actions = inv_dyn_model(
                        current_state_for_pred, future_state)
                    if hasattr(actions, 'mean'):  # Handle distribution output
                        actions = actions.mean
                else:
                    # Fall back to predict method
                    actions = inv_dyn_model.predict(
                        torch.cat([current_state_for_pred, future_state], dim=-1))
            else:
                # For standard MLP/Sequential models, concatenate along feature dimension
                # Ensure both tensors have the same batch dimension structure before concat
                print(
                    f"Input shapes - current_state_for_pred: {current_state_for_pred.shape}, future_state: {future_state.shape}")

                # Make sure dimensions match before concatenation
                if current_state_for_pred.dim() != future_state.dim():
                    if current_state_for_pred.dim() < future_state.dim():
                        # Add missing dimensions to current_state_for_pred
                        current_state_for_pred = current_state_for_pred.view(
                            current_state_for_pred.shape[0], 1, -1) if current_state_for_pred.dim() == 2 else current_state_for_pred
                    elif future_state.dim() < current_state_for_pred.dim():
                        # Remove extra dimension from future_state if needed
                        future_state = future_state.squeeze(
                            1) if future_state.dim() == 3 else future_state

                print(
                    f"Adjusted shapes - current_state_for_pred: {current_state_for_pred.shape}, future_state: {future_state.shape}")

                # Process the concatenation safely based on adjusted dimensions
                actions = inv_dyn_model.predict(
                    torch.cat([current_state_for_pred, future_state], dim=-1))

        # Reshape actions back to [batch, num_samples, action_dim] if needed
        if num_inference_samples > 1:
            actions = actions.reshape(batch_size, num_inference_samples, -1)

    # Add debugging information about the output actions
    print(f"Generated actions with shape: {actions.shape}")

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
