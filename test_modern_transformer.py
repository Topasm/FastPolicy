#!/usr/bin/env python3
"""
Test script for the modern transformer implementation.
"""

import torch
import torch.nn as nn
from model.predictor.bidirectional_autoregressive_transformer import (
    BidirectionalARTransformer,
    BidirectionalARTransformerConfig
)


def count_parameters(model):
    """Count the total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_modern_transformer():
    """Test the modern transformer implementation."""
    print("🧪 Testing Modern Transformer Implementation")
    print("=" * 50)

    # Create a test configuration
    config = BidirectionalARTransformerConfig(
        state_dim=7,
        hidden_dim=256,  # Smaller for testing
        num_layers=2,    # Fewer layers for testing
        num_heads=8,
        dropout=0.1,
        n_obs_steps=3,
        forward_steps=10,
        backward_steps=8
    )

    # Create the model
    print("Creating model...")
    model = BidirectionalARTransformer(config)

    # Count parameters
    param_count = count_parameters(model)
    print(f"📊 Total trainable parameters: {param_count:,}")

    # Create dummy input data
    batch_size = 2
    device = torch.device('cpu')  # Use CPU for testing

    # Input images: (batch, n_obs_steps, channels, height, width)
    input_images = torch.randn(batch_size, config.n_obs_steps, 3, 96, 96)

    # Input states: (batch, n_obs_steps, state_dim)
    input_states = torch.randn(
        batch_size, config.n_obs_steps, config.state_dim)

    print("🔄 Running forward pass...")

    try:
        # Run the model in inference mode
        model.eval()
        with torch.no_grad():
            outputs = model(input_images, input_states, training=False)

        print("✅ Forward pass successful!")
        print("📋 Output shapes:")
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                print(f"   {key}: {value.shape}")
            else:
                print(f"   {key}: {type(value)}")

        # Verify output shapes
        expected_shapes = {
            'predicted_goal_images': (batch_size, 3, config.output_image_size, config.output_image_size),
            'predicted_forward_states': (batch_size, config.forward_steps, config.state_dim),
            'predicted_backward_states': (batch_size, config.backward_steps, config.state_dim),
            'predicted_progress': (batch_size, 1)
        }

        print("\n🔍 Verifying output shapes...")
        all_correct = True
        for key, expected_shape in expected_shapes.items():
            if key in outputs:
                actual_shape = outputs[key].shape
                if actual_shape == expected_shape:
                    print(f"   ✅ {key}: {actual_shape} (correct)")
                else:
                    print(
                        f"   ❌ {key}: {actual_shape} (expected {expected_shape})")
                    all_correct = False
            else:
                print(f"   ❌ {key}: missing from outputs")
                all_correct = False

        if all_correct:
            print("\n🎉 All tests passed! Modern transformer is working correctly.")
        else:
            print("\n⚠️  Some tests failed. Please check the implementation.")

    except Exception as e:
        print(f"❌ Forward pass failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


def test_modern_features():
    """Test specific modern transformer features."""
    print("\n🔬 Testing Modern Transformer Components")
    print("=" * 50)

    from model.predictor.gpt_backbone import (
        RMSNorm, RotaryPositionalEmbedding, SwiGLU, GroupedQueryAttention
    )

    d_model = 128
    seq_len = 10
    batch_size = 2

    # Test RMSNorm
    print("Testing RMSNorm...")
    rms_norm = RMSNorm(d_model)
    x = torch.randn(batch_size, seq_len, d_model)
    normed_x = rms_norm(x)
    print(f"   Input shape: {x.shape}, Output shape: {normed_x.shape}")

    # Test RoPE
    print("Testing Rotary Position Embedding...")
    rope = RotaryPositionalEmbedding(d_model // 8)  # head_dim
    cos, sin = rope(x, seq_len)
    print(f"   Cos shape: {cos.shape}, Sin shape: {sin.shape}")

    # Test SwiGLU
    print("Testing SwiGLU...")
    swiglu = SwiGLU(d_model, d_model * 2)
    swiglu_out = swiglu(x)
    print(f"   Input shape: {x.shape}, Output shape: {swiglu_out.shape}")

    # Test GQA
    print("Testing Grouped Query Attention...")
    gqa = GroupedQueryAttention(d_model, n_heads=8, n_kv_heads=2)
    gqa_out = gqa(x, rope)
    print(f"   Input shape: {x.shape}, Output shape: {gqa_out.shape}")

    print("✅ All modern components tested successfully!")


if __name__ == "__main__":
    # Test the complete transformer
    success = test_modern_transformer()

    if success:
        # Test individual components
        test_modern_features()

    print("\n🏁 Testing complete!")
