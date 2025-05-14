# Trajectory Noise Critic Model

This module implements a critic model that distinguishes between original trajectories and trajectories with progressively added noise to future states. The model can be used to evaluate the quality of generated or predicted trajectories.

## Models

The implementation offers three different architectures for the noise critic model:

1. **MLP-based critic** - Flattens the trajectory and processes it through an MLP network.
2. **Transformer-based critic** - Uses a transformer encoder to process the sequence of states.
3. **GRU-based critic** - Uses a GRU to encode the sequence of states.

All models can optionally incorporate image features as context to improve their discrimination capability.

## Training

The training script adds noise to trajectories with increasing magnitude for each timestep:

```python
current_noise_scale = base_noise_scale
for t_step in range(1, H):
    noise = torch.randn_like(negative_state_trajectory[:, t_step]) * current_noise_scale
    negative_state_trajectory[:, t_step] += noise
    current_noise_scale *= noise_growth_factor
```

This creates a challenging discrimination task where the model must learn to identify trajectories with physically implausible state transitions.

### Usage

To train the model:

```bash
python train_eval/train_noise_critic.py \
    --architecture mlp \
    --dataset lerobot/pusht \
    --batch_size 64 \
    --use_images \
    --base_noise_scale 0.05 \
    --noise_growth_factor 1.2 \
    --steps 10000 \
    --device cuda
```

Parameters:
- `--architecture`: Model architecture choice (mlp, transformer, gru)
- `--use_images`: Flag to use image features as context
- `--base_noise_scale`: Initial noise magnitude
- `--noise_growth_factor`: Factor by which noise increases per timestep

## Evaluation

The evaluation script tests the model's ability to distinguish between original and noisy trajectories across different noise levels. It generates plots showing:

1. Score distributions
2. ROC curves
3. Performance vs. noise level
4. Confusion matrix

### Usage

To evaluate a trained model:

```bash
python train_eval/eval_noise_critic.py \
    --model_path outputs/train/noise_critic/noise_critic_final.pth \
    --config_path outputs/train/noise_critic/config.json \
    --dataset lerobot/pusht \
    --noise_levels 0.01,0.05,0.1,0.2,0.5,1.0 \
    --num_samples 100
```

## Integration

This critic model can be integrated with other generative models (like diffusion models) to:

1. Evaluate the quality of generated trajectories
2. Guide the generation process through rejection sampling
3. Provide additional loss signals during training

## Configuration

The model is configured through the `NoiseCriticConfig` class, which specifies:

- State dimensions
- Sequence horizon length
- Architecture choice
- Hidden dimensions
- Whether to use image context
- And other hyperparameters
