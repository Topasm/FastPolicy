"""
Evaluation utilities for interpolated states.
"""
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def compute_interpolation_smoothness(state_sequence: torch.Tensor) -> torch.Tensor:
    """
    Computes a smoothness metric for interpolated state sequences.
    Lower values indicate smoother transitions.

    Args:
        state_sequence: Tensor of shape (B, T, D) containing interpolated states

    Returns:
        smoothness: Tensor of shape (B,) with smoothness score per batch
    """
    # Calculate first-order differences (velocity)
    first_diff = state_sequence[:, 1:] - state_sequence[:, :-1]

    # Calculate second-order differences (acceleration)
    second_diff = first_diff[:, 1:] - first_diff[:, :-1]

    # Compute the average magnitude of acceleration (smoothness metric)
    # Lower acceleration magnitude means smoother interpolation
    smoothness = torch.mean(torch.norm(second_diff, dim=-1), dim=-1)

    return smoothness


def evaluate_interpolation_quality(
    keyframe_states: torch.Tensor,
    interpolated_states: torch.Tensor,
    keyframe_indices: List[int],
    interpolated_indices: List[int]
) -> Dict[str, torch.Tensor]:
    """
    Evaluates the quality of interpolated states based on multiple metrics.

    Args:
        keyframe_states: Tensor of shape (B, K, D) with the keyframe states
        interpolated_states: Tensor of shape (B, T, D) with the interpolated states
        keyframe_indices: List of indices for the keyframes
        interpolated_indices: List of indices for the interpolated states

    Returns:
        metrics: Dictionary with quality metrics:
            - 'smoothness': Lower is better
            - 'keyframe_error': Should be near zero if interpolation preserves keyframes
            - 'path_length_ratio': Ratio of interpolated path length to linear path
    """
    batch_size = keyframe_states.shape[0]
    metrics = {}

    # Compute smoothness
    smoothness = compute_interpolation_smoothness(interpolated_states)
    metrics['smoothness'] = smoothness

    # Compute keyframe preservation error
    keyframe_error = torch.zeros(batch_size, device=keyframe_states.device)

    # Map interpolated indices to their positions in the sequence
    interp_idx_to_pos = {idx: pos for pos,
                         idx in enumerate(interpolated_indices)}

    # For each keyframe, check if it's preserved in the interpolated sequence
    for k_idx, kf_index in enumerate(keyframe_indices):
        if kf_index in interp_idx_to_pos:
            # Get the corresponding interpolated state
            interp_pos = interp_idx_to_pos[kf_index]
            # Compute error (should be close to zero)
            error = torch.norm(
                interpolated_states[:, interp_pos] - keyframe_states[:, k_idx],
                dim=-1
            )
            keyframe_error += error

    # Average error across all keyframes
    keyframe_error = keyframe_error / len(keyframe_indices)
    metrics['keyframe_error'] = keyframe_error

    # Compute path length ratio (indicates if path is direct or circuitous)
    # Linear path length (direct connections between keyframes)
    linear_length = torch.zeros(batch_size, device=keyframe_states.device)
    for i in range(keyframe_states.shape[1] - 1):
        segment_length = torch.norm(
            keyframe_states[:, i+1] - keyframe_states[:, i],
            dim=-1
        )
        linear_length += segment_length

    # Interpolated path length
    interp_length = torch.zeros(batch_size, device=interpolated_states.device)
    for i in range(interpolated_states.shape[1] - 1):
        segment_length = torch.norm(
            interpolated_states[:, i+1] - interpolated_states[:, i],
            dim=-1
        )
        interp_length += segment_length

    # Ratio of interpolated to linear path length
    # Value close to 1 means the path is direct
    # Value much larger than 1 means the path is taking a circuitous route
    path_ratio = interp_length / (linear_length + 1e-8)  # avoid div by zero
    metrics['path_length_ratio'] = path_ratio

    return metrics


def select_best_interpolation(
    linear_states: torch.Tensor,
    cubic_states: torch.Tensor,
    keyframe_states: torch.Tensor,
    keyframe_indices: List[int],
    interpolated_indices: List[int]
) -> Tuple[torch.Tensor, List[str]]:
    """
    Selects the best interpolation method per batch element
    based on interpolation quality metrics.

    Args:
        linear_states: Linearly interpolated states (B, T, D)
        cubic_states: Cubic interpolated states (B, T, D)
        keyframe_states: Keyframe states (B, K, D)
        keyframe_indices: Indices of keyframes
        interpolated_indices: Indices of interpolated states

    Returns:
        best_states: The best interpolated states per batch (B, T, D)
        method_choices: List of strings indicating which method was chosen for each batch
    """
    batch_size = linear_states.shape[0]

    # Evaluate both methods
    linear_metrics = evaluate_interpolation_quality(
        keyframe_states, linear_states, keyframe_indices, interpolated_indices)
    cubic_metrics = evaluate_interpolation_quality(
        keyframe_states, cubic_states, keyframe_indices, interpolated_indices)

    # Create scoring function (lower is better)
    # Weight smoothness more for longer sequences
    smoothness_weight = 1.0 if len(interpolated_indices) <= 8 else 2.0

    linear_scores = (
        smoothness_weight * linear_metrics['smoothness'] +
        5.0 * linear_metrics['keyframe_error'] +
        0.5 * torch.abs(linear_metrics['path_length_ratio'] - 1.0)
    )

    cubic_scores = (
        smoothness_weight * cubic_metrics['smoothness'] +
        5.0 * cubic_metrics['keyframe_error'] +
        0.5 * torch.abs(cubic_metrics['path_length_ratio'] - 1.0)
    )

    # Select the best method for each batch element
    best_states = torch.zeros_like(linear_states)
    method_choices = []

    for b in range(batch_size):
        if cubic_scores[b] < linear_scores[b]:
            best_states[b] = cubic_states[b]
            method_choices.append("cubic")
        else:
            best_states[b] = linear_states[b]
            method_choices.append("linear")

    return best_states, method_choices
