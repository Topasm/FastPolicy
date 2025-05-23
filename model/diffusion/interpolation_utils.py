"""
Interpolation utilities for state interpolation.
"""
import torch
import torch.nn.functional as F


def linear_interpolate_states(keyframe_states, keyframe_times, target_times):
    """
    Performs linear interpolation between keyframe states.

    Args:
        keyframe_states: Tensor of shape (B, K, D) where K is number of keyframes, D is state dimension
        keyframe_times: Tensor of shape (K,) with the time value for each keyframe
        target_times: Tensor of shape (T,) with the time values to interpolate at

    Returns:
        Interpolated states tensor of shape (B, T, D)
    """
    batch_size, num_keyframes, state_dim = keyframe_states.shape
    num_targets = len(target_times)
    device = keyframe_states.device

    # Initialize output tensor
    interpolated_states = torch.zeros(
        (batch_size, num_targets, state_dim), device=device)

    # Convert time tensors to lists for easier indexing
    keyframe_times_list = keyframe_times.tolist()

    # For each target time, find the two surrounding keyframes and interpolate
    for i, t in enumerate(target_times):
        # Find the two keyframes that surround this target time
        # If target is before first keyframe or after last, we'll clamp appropriately
        if t <= keyframe_times[0]:
            interpolated_states[:, i] = keyframe_states[:, 0]
            continue
        elif t >= keyframe_times[-1]:
            interpolated_states[:, i] = keyframe_states[:, -1]
            continue

        # Find the keyframe indices that bound the target time
        next_idx = torch.searchsorted(keyframe_times, t).item()
        prev_idx = next_idx - 1

        # Get the times and states at those indices
        t0 = keyframe_times_list[prev_idx]
        t1 = keyframe_times_list[next_idx]
        s0 = keyframe_states[:, prev_idx]
        s1 = keyframe_states[:, next_idx]

        # Calculate interpolation weight (0 to 1)
        alpha = (t - t0) / (t1 - t0)

        # Linear interpolation: s = (1-α)s0 + αs1
        interpolated_states[:, i] = (1 - alpha) * s0 + alpha * s1

    return interpolated_states


def cubic_interpolate_states(keyframe_states, keyframe_times, target_times):
    """
    Performs cubic spline interpolation between keyframe states for smoother transitions.
    Uses Catmull-Rom spline interpolation, which ensures that the curve passes through all keyframe points.

    Args:
        keyframe_states: Tensor of shape (B, K, D) where K is number of keyframes, D is state dimension
        keyframe_times: Tensor of shape (K,) with the time value for each keyframe
        target_times: Tensor of shape (T,) with the time values to interpolate at

    Returns:
        Interpolated states tensor of shape (B, T, D)
    """
    batch_size, num_keyframes, state_dim = keyframe_states.shape
    num_targets = len(target_times)
    device = keyframe_states.device

    # Create tensor for storing results
    interpolated_states = torch.zeros(
        (batch_size, num_targets, state_dim), device=device)

    # Convert time tensors to lists for easier indexing
    keyframe_times_list = keyframe_times.tolist()

    # Process all batches and dimensions
    for b in range(batch_size):
        for i, t in enumerate(target_times):
            # Handle edge cases - if target time is outside keyframe range
            if t <= keyframe_times[0]:
                interpolated_states[b, i] = keyframe_states[b, 0]
                continue
            elif t >= keyframe_times[-1]:
                interpolated_states[b, i] = keyframe_states[b, -1]
                continue

            # Find the keyframe segment that contains the target time
            next_idx = torch.searchsorted(keyframe_times, t).item()
            prev_idx = next_idx - 1

            # Get the normalized position within the segment (0 to 1)
            t0 = keyframe_times_list[prev_idx]
            t1 = keyframe_times_list[next_idx]
            alpha = (t - t0) / (t1 - t0)

            # For Catmull-Rom spline, we need 4 control points:
            # Two points on either side of our segment
            p_idx = [
                # Point before previous keyframe
                max(0, prev_idx - 1),
                prev_idx,                        # Previous keyframe
                next_idx,                        # Next keyframe
                # Point after next keyframe
                min(num_keyframes - 1, next_idx + 1)
            ]

            # Get the state values at those control points
            p0 = keyframe_states[b, p_idx[0]]
            p1 = keyframe_states[b, p_idx[1]]
            p2 = keyframe_states[b, p_idx[2]]
            p3 = keyframe_states[b, p_idx[3]]

            # Calculate Catmull-Rom basis functions
            # These ensure C1 continuity (continuous first derivative)
            alpha2 = alpha * alpha
            alpha3 = alpha2 * alpha

            # Hermite basis functions
            h00 = 2*alpha3 - 3*alpha2 + 1    # p1 coefficient
            h10 = alpha3 - 2*alpha2 + alpha  # m1 coefficient (tangent at p1)
            h01 = -2*alpha3 + 3*alpha2       # p2 coefficient
            h11 = alpha3 - alpha2            # m2 coefficient (tangent at p2)

            # Calculate tangents (Catmull-Rom uses neighboring points)
            m1 = 0.5 * (p2 - p0)  # tangent at p1
            m2 = 0.5 * (p3 - p1)  # tangent at p2

            # Compute the interpolated state using Hermite interpolation
            interpolated_state = h00*p1 + h10*m1 + h01*p2 + h11*m2
            interpolated_states[b, i] = interpolated_state

    return interpolated_states


def adaptive_interpolate_states(keyframe_states, keyframe_times, target_times):
    """
    Adaptively selects the best interpolation method for each state dimension.
    Uses cubic interpolation for smooth dimensions and linear interpolation for dimensions
    that should have sharp transitions.

    Args:
        keyframe_states: Tensor of shape (B, K, D) where K is number of keyframes, D is state dimension
        keyframe_times: Tensor of shape (K,) with the time value for each keyframe
        target_times: Tensor of shape (T,) with the time values to interpolate at

    Returns:
        Interpolated states tensor of shape (B, T, D)
    """
    batch_size, num_keyframes, state_dim = keyframe_states.shape
    num_targets = len(target_times)
    device = keyframe_states.device

    # Calculate both linear and cubic interpolations
    linear_states = linear_interpolate_states(
        keyframe_states, keyframe_times, target_times)
    cubic_states = cubic_interpolate_states(
        keyframe_states, keyframe_times, target_times)

    # Initialize output tensor
    interpolated_states = torch.zeros(
        (batch_size, num_targets, state_dim), device=device)

    # For each state dimension, determine which interpolation method is better
    for d in range(state_dim):
        # Extract this dimension from both interpolation methods
        linear_dim = linear_states[:, :, d]  # (B, T)
        cubic_dim = cubic_states[:, :, d]    # (B, T)

        # Calculate the "smoothness" for this dimension
        # Second derivative of keyframe states indicates how smooth this dimension is
        dim_keyframes = keyframe_states[:, :, d]  # (B, K)

        # Only perform adaptive selection if we have enough keyframes
        if num_keyframes >= 4:
            # Calculate first differences
            first_diff = dim_keyframes[:, 1:] - \
                dim_keyframes[:, :-1]  # (B, K-1)

            # Calculate second differences (if possible)
            if first_diff.shape[1] >= 2:
                second_diff = first_diff[:, 1:] - \
                    first_diff[:, :-1]  # (B, K-2)

                # Calculate average magnitude of second differences
                avg_second_diff = torch.mean(
                    torch.abs(second_diff), dim=1)  # (B,)

                # Threshold for determining if we should use cubic interpolation
                # Lower threshold for dimensions that should be smoother
                smooth_threshold = 0.1

                # Create a mask where True means "use cubic interpolation"
                use_cubic_mask = avg_second_diff <= smooth_threshold  # (B,)

                # Apply the mask to select the appropriate interpolation for each batch
                for b in range(batch_size):
                    if use_cubic_mask[b]:
                        interpolated_states[b, :, d] = cubic_dim[b]
                    else:
                        interpolated_states[b, :, d] = linear_dim[b]
            else:
                # Not enough keyframes for second differences, use cubic by default
                interpolated_states[:, :, d] = cubic_dim
        else:
            # Not enough keyframes, default to linear
            interpolated_states[:, :, d] = linear_dim

    return interpolated_states


def smart_interpolate_states(keyframe_states, keyframe_times, target_times, method="auto"):
    """
    Smart interpolation that selects the appropriate method based on context.

    Args:
        keyframe_states: Tensor of shape (B, K, D) where K is number of keyframes
        keyframe_times: Tensor of shape (K,) with the time value for each keyframe
        target_times: Tensor of shape (T,) with the time values to interpolate at
        method: Interpolation method - "linear", "cubic", "adaptive", or "auto"

    Returns:
        Interpolated states tensor of shape (B, T, D)
    """
    if method == "linear":
        return linear_interpolate_states(keyframe_states, keyframe_times, target_times)
    elif method == "cubic":
        return cubic_interpolate_states(keyframe_states, keyframe_times, target_times)
    elif method == "adaptive":
        return adaptive_interpolate_states(keyframe_states, keyframe_times, target_times)
    elif method == "auto":
        # Auto mode - select method based on number of keyframes and spacing
        num_keyframes = len(keyframe_times)

        if num_keyframes <= 2:
            # With only 2 keyframes, linear is sufficient and stable
            return linear_interpolate_states(keyframe_states, keyframe_times, target_times)
        elif num_keyframes >= 5:
            # With many keyframes, adaptive method provides best results
            return adaptive_interpolate_states(keyframe_states, keyframe_times, target_times)
        else:
            # With moderate number of keyframes, cubic provides good smoothness
            return cubic_interpolate_states(keyframe_states, keyframe_times, target_times)
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
