#!/usr/bin/env python
"""
Test script to verify interpolation configuration logic without full dependencies
"""


def test_interpolation_indices():
    """Test the interpolation index calculation logic"""

    # Simulate configuration values
    n_obs_steps = 2
    output_horizon = 8
    keyframe_indices = [8, 16, 32]

    # Test keyframe_delta_indices
    history_indices = list(range(1 - n_obs_steps, 1))  # [-1, 0]
    keyframe_delta_indices = history_indices + \
        keyframe_indices  # [-1, 0, 8, 16, 32]

    print("Keyframe delta indices:", keyframe_delta_indices)

    # Test interpolation_target_indices for different modes
    horizons = [8, 16, 32]
    modes = ["dense", "skip_even", "sparse"]

    for horizon in horizons:
        print(f"\nHorizon {horizon}:")

        if horizon == 8:
            # Dense output: 0, 1, 2, 3, 4, 5, 6, 7
            target_indices = list(range(output_horizon))
            print(f"  Dense: {target_indices}")

        elif horizon == 16:
            # Skip even: 0, 2, 4, 6, 8, 10, 12, 14
            target_indices = list(range(0, 16, 2))
            print(f"  Skip even: {target_indices}")

        elif horizon == 32:
            # Sparse: 0, 4, 8, 12, 16, 20, 24, 28
            target_indices = list(range(0, 32, 4))
            print(f"  Sparse: {target_indices}")

        print(
            f"  Target length: {len(target_indices)} (should be {output_horizon})")


def test_state_indexing():
    """Test state sequence indexing logic"""

    n_obs_steps = 2
    keyframe_indices = [-1, 0, 8, 16, 32]
    target_indices = [0, 2, 4, 6, 8, 10, 12, 14]  # skip_even for horizon=16

    print("\nState indexing test:")
    print("Keyframe indices (relative):", keyframe_indices)
    print("Target indices (relative):", target_indices)

    # Convert to absolute indices (assuming time 0 is at index n_obs_steps-1)
    keyframe_abs = [idx + n_obs_steps - 1 for idx in keyframe_indices]
    target_abs = [idx + n_obs_steps - 1 for idx in target_indices]

    print("Keyframe indices (absolute):", keyframe_abs)
    print("Target indices (absolute):", target_abs)

    # Required sequence length
    max_abs_idx = max(max(keyframe_abs), max(target_abs))
    print(f"Required sequence length: {max_abs_idx + 1}")


if __name__ == "__main__":
    test_interpolation_indices()
    test_state_indexing()
