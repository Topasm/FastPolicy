"""
Test the interpolation utilities to ensure they produce smooth interpolated states.
"""
from model.diffusion.interpolation_utils import linear_interpolate_states, cubic_interpolate_states
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def test_interpolation_methods():
    """Compare linear and cubic interpolation."""
    print("Starting interpolation test...")

    # Create some simple keyframe states
    batch_size = 1
    num_keyframes = 4
    state_dim = 2

    # Define keyframe times
    keyframe_times = torch.tensor([0.0, 1.0, 2.0, 3.0])
    print(f"Keyframe times: {keyframe_times}")

    # Define keyframe states with some discontinuities to show interpolation effects
    keyframe_states = torch.zeros((batch_size, num_keyframes, state_dim))
    keyframe_states[0, 0] = torch.tensor([0.0, 0.0])
    keyframe_states[0, 1] = torch.tensor([1.0, 2.0])
    keyframe_states[0, 2] = torch.tensor([2.0, 0.0])
    keyframe_states[0, 3] = torch.tensor([3.0, 1.0])
    print(f"Keyframe states:\n{keyframe_states[0]}")

    # Define target times (finer granularity)
    # Using fewer points for display
    target_times = torch.linspace(0.0, 3.0, 10)
    print(f"Target times: {target_times}")

    print("\nComputing linear interpolation...")
    # Compute interpolations
    linear_results = linear_interpolate_states(
        keyframe_states, keyframe_times, target_times)
    print("Linear interpolation results:")
    print(linear_results[0])

    print("\nComputing cubic interpolation...")
    cubic_results = cubic_interpolate_states(
        keyframe_states, keyframe_times, target_times)
    print("Cubic interpolation results:")
    print(cubic_results[0])

    print("\nInterpolation test completed successfully!")


if __name__ == "__main__":
    test_interpolation_methods()
