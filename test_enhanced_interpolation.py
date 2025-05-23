"""
Test the enhanced interpolation with variable keyframes.
"""
from model.diffusion.interpolation_utils import (
    linear_interpolate_states,
    cubic_interpolate_states,
    adaptive_interpolate_states,
    smart_interpolate_states
)
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.dirname(__file__)))


def test_interpolation_comparison():
    """Compare different interpolation methods with quality metrics."""
    print("Starting enhanced interpolation test...")

    # Create a more complex set of keyframe states with challenging patterns
    batch_size = 2
    num_keyframes = 5
    state_dim = 4

    # Define keyframe times (variable spacing)
    keyframe_times = torch.tensor([0.0, 4.0, 8.0, 16.0, 32.0])
    print(f"Keyframe times: {keyframe_times}")

    # Create test data: two batches with different patterns
    keyframe_states = torch.zeros((batch_size, num_keyframes, state_dim))

    # Batch 0: Smooth curves and sharp transitions
    # Dimension 0: Smooth sine wave
    keyframe_states[0, :, 0] = torch.tensor([0.0, 0.7, 1.0, 0.7, 0.0])
    # Dimension 1: Sharp step function
    keyframe_states[0, :, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    # Dimension 2: Quadratic curve
    keyframe_states[0, :, 2] = torch.tensor([0.0, 0.4, 1.0, 0.4, 0.0])
    # Dimension 3: Linear ramp with discontinuity
    keyframe_states[0, :, 3] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5])

    # Batch 1: Different patterns
    # Dimension 0: Linear ramp up
    keyframe_states[1, :, 0] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    # Dimension 1: Oscillating function
    keyframe_states[1, :, 1] = torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5])
    # Dimension 2: Exponential growth
    keyframe_states[1, :, 2] = torch.tensor([0.1, 0.2, 0.4, 0.7, 1.0])
    # Dimension 3: Constant with spike
    keyframe_states[1, :, 3] = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5])

    print(f"Keyframe states batch 0:\n{keyframe_states[0]}")
    print(f"Keyframe states batch 1:\n{keyframe_states[1]}")

    # Define target times (finer granularity)
    target_times = torch.linspace(0.0, 32.0, 16)  # 16 interpolated points
    print(f"Target times: {target_times}")

    # Compute interpolations with different methods
    print("\nComputing linear interpolation...")
    linear_results = linear_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing cubic interpolation...")
    cubic_results = cubic_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing adaptive interpolation...")
    adaptive_results = adaptive_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing smart interpolation...")
    smart_results = smart_interpolate_states(
        keyframe_states, keyframe_times, target_times, method="auto")

    # Print sample interpolated states
    print("\nSample interpolated states (first batch, first 3 points):")
    print(f"Linear:\n{linear_results[0, :3]}")
    print(f"Cubic:\n{cubic_results[0, :3]}")
    print(f"Adaptive:\n{adaptive_results[0, :3]}")
    print(f"Smart:\n{smart_results[0, :3]}")

    # Calculate a simple smoothness metric manually
    def simple_smoothness(states):
        # Calculate first-order differences (velocity)
        first_diff = states[:, 1:] - states[:, :-1]
        # Calculate second-order differences (acceleration)
        second_diff = first_diff[:, 1:] - first_diff[:, :-1]
        # Compute the average magnitude of acceleration
        return torch.mean(torch.norm(second_diff, dim=-1), dim=-1)

    # Compare smoothness
    linear_smoothness = simple_smoothness(linear_results)
    cubic_smoothness = simple_smoothness(cubic_results)
    adaptive_smoothness = simple_smoothness(adaptive_results)
    smart_smoothness = simple_smoothness(smart_results)

    print("\nSmootness metrics (lower is better):")
    print(f"Linear: {linear_smoothness.mean().item():.5f}")
    print(f"Cubic: {cubic_smoothness.mean().item():.5f}")
    print(f"Adaptive: {adaptive_smoothness.mean().item():.5f}")
    print(f"Smart: {smart_smoothness.mean().item():.5f}")

    print("\nEnhanced interpolation test completed successfully!")


if __name__ == "__main__":
    test_interpolation_comparison()


def test_interpolation_comparison():
    """Compare different interpolation methods with quality metrics."""
    print("Starting enhanced interpolation test...")

    # Create a more complex set of keyframe states with challenging patterns
    batch_size = 2
    num_keyframes = 5
    state_dim = 4

    # Define keyframe times (variable spacing)
    keyframe_times = torch.tensor([0.0, 4.0, 8.0, 16.0, 32.0])
    print(f"Keyframe times: {keyframe_times}")

    # Create test data: two batches with different patterns
    keyframe_states = torch.zeros((batch_size, num_keyframes, state_dim))

    # Batch 0: Smooth curves and sharp transitions
    # Dimension 0: Smooth sine wave
    keyframe_states[0, :, 0] = torch.tensor([0.0, 0.7, 1.0, 0.7, 0.0])
    # Dimension 1: Sharp step function
    keyframe_states[0, :, 1] = torch.tensor([0.0, 0.0, 1.0, 1.0, 0.0])
    # Dimension 2: Quadratic curve
    keyframe_states[0, :, 2] = torch.tensor([0.0, 0.4, 1.0, 0.4, 0.0])
    # Dimension 3: Linear ramp with discontinuity
    keyframe_states[0, :, 3] = torch.tensor([0.0, 0.5, 1.0, 0.0, 0.5])

    # Batch 1: Different patterns
    # Dimension 0: Linear ramp up
    keyframe_states[1, :, 0] = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    # Dimension 1: Oscillating function
    keyframe_states[1, :, 1] = torch.tensor([0.5, -0.5, 0.5, -0.5, 0.5])
    # Dimension 2: Exponential growth
    keyframe_states[1, :, 2] = torch.tensor([0.1, 0.2, 0.4, 0.7, 1.0])
    # Dimension 3: Constant with spike
    keyframe_states[1, :, 3] = torch.tensor([0.5, 0.5, 1.0, 0.5, 0.5])

    print(f"Keyframe states batch 0:\n{keyframe_states[0]}")
    print(f"Keyframe states batch 1:\n{keyframe_states[1]}")

    # Define target times (finer granularity)
    target_times = torch.linspace(0.0, 32.0, 16)  # 16 interpolated points
    print(f"Target times: {target_times}")

    # Compute interpolations with different methods
    print("\nComputing linear interpolation...")
    linear_results = linear_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing cubic interpolation...")
    cubic_results = cubic_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing adaptive interpolation...")
    adaptive_results = adaptive_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    print("\nComputing smart interpolation...")
    smart_results = smart_interpolate_states(
        keyframe_states, keyframe_times, target_times, method="auto")

    # Evaluate quality metrics
    print("\nEvaluating interpolation quality...")

    linear_metrics = evaluate_interpolation_quality(
        keyframe_states, linear_results,
        keyframe_indices=keyframe_times.tolist(),
        interpolated_indices=target_times.tolist()
    )

    cubic_metrics = evaluate_interpolation_quality(
        keyframe_states, cubic_results,
        keyframe_indices=keyframe_times.tolist(),
        interpolated_indices=target_times.tolist()
    )

    adaptive_metrics = evaluate_interpolation_quality(
        keyframe_states, adaptive_results,
        keyframe_indices=keyframe_times.tolist(),
        interpolated_indices=target_times.tolist()
    )

    smart_metrics = evaluate_interpolation_quality(
        keyframe_states, smart_results,
        keyframe_indices=keyframe_times.tolist(),
        interpolated_indices=target_times.tolist()
    )

    # Print metrics
    print("\nInterpolation Quality Metrics:")
    print(f"Linear: Smoothness={linear_metrics['smoothness'].mean().item():.5f}, "
          f"Keyframe Error={linear_metrics['keyframe_error'].mean().item():.5f}, "
          f"Path Ratio={linear_metrics['path_length_ratio'].mean().item():.5f}")

    print(f"Cubic: Smoothness={cubic_metrics['smoothness'].mean().item():.5f}, "
          f"Keyframe Error={cubic_metrics['keyframe_error'].mean().item():.5f}, "
          f"Path Ratio={cubic_metrics['path_length_ratio'].mean().item():.5f}")

    print(f"Adaptive: Smoothness={adaptive_metrics['smoothness'].mean().item():.5f}, "
          f"Keyframe Error={adaptive_metrics['keyframe_error'].mean().item():.5f}, "
          f"Path Ratio={adaptive_metrics['path_length_ratio'].mean().item():.5f}")

    print(f"Smart: Smoothness={smart_metrics['smoothness'].mean().item():.5f}, "
          f"Keyframe Error={smart_metrics['keyframe_error'].mean().item():.5f}, "
          f"Path Ratio={smart_metrics['path_length_ratio'].mean().item():.5f}")

    # Select best interpolation
    best_results, method_choices = select_best_interpolation(
        linear_results, cubic_results, keyframe_states,
        keyframe_indices=keyframe_times.tolist(),
        interpolated_indices=target_times.tolist()
    )

    print(f"\nBest methods selected: {method_choices}")

    # Print some sample interpolated states
    print("\nSample interpolated states (first batch, first 3 points):")
    print(f"Linear:\n{linear_results[0, :3]}")
    print(f"Cubic:\n{cubic_results[0, :3]}")
    print(f"Adaptive:\n{adaptive_results[0, :3]}")
    print(f"Smart:\n{smart_results[0, :3]}")
    print(f"Best:\n{best_results[0, :3]}")

    print("\nEnhanced interpolation test completed successfully!")
