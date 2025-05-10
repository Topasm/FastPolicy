# Enhanced Inverse Dynamics Model

This module provides an enhanced inverse dynamics model for the FastPolicy framework, designed to produce smoother actions by using sequential inputs and incorporating features from reference models.

## Features

- **Sequential Input Processing**: Uses a GRU-based architecture to process sequences of states for better temporal coherence
- **Enhanced Architecture**: Incorporates features from the reference MLP_InvDyn_OgB_V3 model
- **Probabilistic Output**: Optional probabilistic output for stochastic sampling during inference
- **Temperature Scaling**: Adjust the exploration/exploitation trade-off during inference
- **Configurable Architecture**: Easily adjust network width, depth, and other parameters
- **Custom Weight Initialization**: Improves training stability and performance

## Files Overview

1. `enhanced_invdyn.py` - Core implementation of the enhanced inverse dynamics model
2. `integration.py` - Utility functions for integrating with the existing FastPolicy pipeline
3. `evaluation_utils.py` - Helper functions for evaluation and inference
4. `train_enhanced_invdyn.py` - Training script for the enhanced model

## Usage Guide

### Training a New Model

Use the dedicated training script with various options:

```bash
python train_eval/train_enhanced_invdyn.py \
  --output_dir outputs/train/enhanced_invdyn \
  --model_type enhanced \
  --batch_size 64 \
  --lr 1e-4 \
  --steps 5000 \
  --probabilistic \
  --temperature 0.1
```

Options:
- `--model_type`: Choose between `enhanced`, `seq`, or `mlp`
- `--probabilistic`: Enable probabilistic output for the enhanced model
- `--temperature`: Set sampling temperature for probabilistic model
- `--seq_length`: Set sequence length for sequential models (default: 3)
- `--hidden_dim`: Set hidden dimension size (default: 512)

### Using the Model for Inference

Import the necessary functions:

```python
from model.invdynamics.enhanced_invdyn import EnhancedInvDynamic
from model.invdynamics.evaluation_utils import (
    load_enhanced_invdyn_model,
    generate_actions_with_enhanced_invdyn
)
```

Load a trained model:

```python
model = load_enhanced_invdyn_model(
    model_path="outputs/train/enhanced_invdyn/invdyn_enhanced_final.pth",
    state_dim=state_dimension,
    action_dim=action_dimension,
    device=device,
    use_probabilistic=True  # Set to match training configuration
)
```

Generate actions:

```python
actions = generate_actions_with_enhanced_invdyn(
    diffusion_model=diffusion_model,
    inv_dyn_model=model,
    norm_batch=normalized_batch,
    norm_current_state=normalized_current_state,
    num_inference_samples=num_samples,
    temperature=0.1,  # Adjust for exploration/exploitation balance
    use_probabilistic_sampling=True  # Set to use stochastic sampling
)
```

### Integration with Existing Pipeline

You can directly replace the inverse dynamics model in the evaluation pipeline:

```python
# In eval_combined_policy.py

from model.invdynamics.evaluation_utils import (
    load_enhanced_invdyn_model,
    generate_actions_with_enhanced_invdyn
)

# Load standard diffusion model
diffusion_model = MyDiffusionModel(diffusion_config)
diffusion_model.load_state_dict(torch.load(diffusion_path))

# Load enhanced inverse dynamics model
inv_dyn_model = load_enhanced_invdyn_model(
    model_path=invdyn_path,
    state_dim=state_dim,
    action_dim=action_dim
)

# Use enhanced model for action generation
actions = generate_actions_with_enhanced_invdyn(
    diffusion_model,
    inv_dyn_model,
    norm_batch,
    norm_current_state,
    num_inference_samples=num_inference_samples
)
```

## Training Tips

1. **Temperature Tuning**: Start with a temperature of 0.1 and adjust based on performance
2. **Sequence Length**: For GRU models, a sequence length of 3-5 often provides good results
3. **Probabilistic vs. Deterministic**: Use probabilistic output for tasks requiring exploration
4. **Architecture**: The default 4-layer network with 512 hidden units works well for most tasks
5. **Normalization**: Ensure proper normalization of states and actions during training and inference

## Model Conversion

If you want to convert an existing MLP or GRU-based model to the enhanced architecture:

```python
from model.invdynamics.integration import convert_to_enhanced_invdyn

enhanced_model = convert_to_enhanced_invdyn(
    model_path="path/to/existing/model.pth",
    state_dim=state_dim,
    action_dim=action_dim,
    use_probabilistic=True,
    output_path="path/to/save/converted/model.pth"
)
```

## Performance Comparison

For best results, compare the performance of different inverse dynamics models:

1. Original MLP-based model
2. Sequential GRU-based model
3. Enhanced model (deterministic)
4. Enhanced model (probabilistic)

The enhanced model typically produces smoother actions and better handles temporal dependencies in sequential tasks.
