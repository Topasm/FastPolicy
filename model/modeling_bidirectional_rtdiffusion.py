from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
from typing import Dict

from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.policies.normalize import Normalize, Unnormalize

from lerobot.common.datasets.utils import PolicyFeature  # A plausible location

from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
from model.diffusion.modeling_clphycon import CLDiffPhyConModel  # state_diffusion_model
from model.invdyn.invdyn import MlpInvDynamic
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE
from lerobot.configs.types import FeatureType


def _safe_get_action_dim(config, inverse_dynamics_model=None, action_feature_data=None):
    """
    Safely get action dimension from multiple sources in order of preference:
    1. From config.action_feature if available
    2. From action_feature_data if it has shape
    3. From inverse_dynamics_model.a_dim
    4. Default to 2 as fallback (common for 2D tasks)
    """
    # Try to get from config
    if hasattr(config, 'action_feature') and hasattr(config.action_feature, 'shape'):
        return config.action_feature.shape[0]

    # Try to get from action_feature_data
    if action_feature_data and isinstance(action_feature_data, dict) and 'shape' in action_feature_data:
        shape = action_feature_data['shape']
        if isinstance(shape, (list, tuple)):
            return shape[0]
        return shape  # Assume scalar dimension

    # Try to get from inverse dynamics model
    if inverse_dynamics_model and hasattr(inverse_dynamics_model, 'a_dim'):
        return inverse_dynamics_model.a_dim

    # Default fallback
    return 2  # Common default for 2D tasks


class BidirectionalRTDiffusionPolicy(nn.Module):
    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        state_diffusion_model: CLDiffPhyConModel,
        inverse_dynamics_model: MlpInvDynamic,
        dataset_stats: dict,
        # Can be Dict[str, PolicyFeature] or Dict[str, dict]
        all_dataset_features: Dict[str, any],
        n_obs_steps: int
    ):
        super().__init__()
        self.bidirectional_transformer = bidirectional_transformer
        self.state_diffusion_model = state_diffusion_model
        self.inverse_dynamics_model = inverse_dynamics_model
        self.config = state_diffusion_model.config
        self.device = get_device_from_parameters(bidirectional_transformer)

        # CL-DiffPhyCon parameters
        self.use_cl_diffphycon = True  # Set to True to enable CL-DiffPhyCon
        self.cl_diffphycon_state = None
        self.cl_diffphycon_step_size = 10  # Number of diffusion steps per policy step

        processed_dataset_stats = {}
        if dataset_stats is not None:
            for key, stat_group in dataset_stats.items():
                processed_dataset_stats[key] = {}
                for subkey, subval in stat_group.items():
                    try:
                        processed_dataset_stats[key][subkey] = torch.as_tensor(
                            subval, dtype=torch.float32, device=self.device
                        )
                    except Exception:
                        processed_dataset_stats[key][subkey] = subval
        else:
            print("Warning: No dataset_stats provided to BidirectionalRTDiffusionPolicy.")

        if hasattr(self.config, 'input_features') and \
           hasattr(self.config, 'normalization_mapping'):

            # Prepare input_features for Normalize: ensure values are PolicyFeature instances
            # self.config.input_features should already contain PolicyFeature instances if loaded correctly
            valid_input_features_for_normalize = {}
            if self.config.input_features:
                for k, v_feat in self.config.input_features.items():
                    if isinstance(v_feat, dict) and 'type' in v_feat and 'shape' in v_feat:
                        try:
                            valid_input_features_for_normalize[k] = PolicyFeature(
                                type=FeatureType(v_feat['type']), shape=tuple(v_feat['shape']))
                        except Exception as e:
                            print(
                                f"Warning: Could not convert input_feature dict {v_feat} to PolicyFeature for key {k}: {e}")
                            # Optionally skip this feature or handle error
                    # Assuming it's already a PolicyFeature-like object
                    elif hasattr(v_feat, 'type') and hasattr(v_feat, 'shape'):
                        valid_input_features_for_normalize[k] = v_feat
                    # Else, if v_feat is None or not convertible, it might be skipped by Normalize/create_stats_buffers later or error there.

            self.normalize_inputs = Normalize(
                valid_input_features_for_normalize,  # Use potentially converted features
                self.config.normalization_mapping,
                processed_dataset_stats
            )

            # Get action_feature_spec from all_dataset_features and ensure it's a PolicyFeature instance
            action_feature_data = all_dataset_features.get("action")
            if action_feature_data is None:
                raise ValueError("Critical: 'action' feature specification not found in all_dataset_features. "
                                 "Cannot initialize Unnormalize for actions.")

            action_feature_spec = None

            # Check the structure of action_feature_data with detailed logging to help debug
            print(
                f"Processing action_feature_data: {type(action_feature_data)}")

            # Case 1: It's a dict with direct 'type' and 'shape' keys
            if isinstance(action_feature_data, dict) and 'type' in action_feature_data and 'shape' in action_feature_data:
                try:
                    action_feature_spec = PolicyFeature(
                        type=FeatureType(action_feature_data['type']),
                        shape=tuple(action_feature_data['shape'])
                    )
                    print(
                        f"Created PolicyFeature from dict with direct keys: {action_feature_spec}")
                except Exception as e:
                    print(
                        f"Error creating PolicyFeature from direct dict: {e}")

            # Case 2: It's a nested dict with metadata structure
            elif isinstance(action_feature_data, dict):
                # Try to extract nested information
                if 'shape' in action_feature_data:
                    shape_value = action_feature_data['shape']
                    # The shape might be a list or tuple already
                    if not isinstance(shape_value, (list, tuple)):
                        print(
                            f"Warning: shape is not a list/tuple: {shape_value}")
                        # Convert to list if it's a scalar
                        shape_value = [shape_value]

                    # Always use 'continuous' for actions when dealing with robot control
                    # This handles the case where there's no 'type' key or it's in another format
                    try:
                        action_feature_spec = PolicyFeature(
                            type=FeatureType.ACTION,  # Use FeatureType.ACTION for actions
                            shape=tuple(shape_value)
                        )
                        print(
                            f"Created PolicyFeature with FeatureType.ACTION and shape {shape_value}")
                    except Exception as e:
                        print(
                            f"Error creating PolicyFeature from nested dict: {e}, Data: {action_feature_data}")
                # Handle case with 'dtype' instead of 'type' key (from LeRobotDatasetMetadata)
                elif 'dtype' in action_feature_data and 'shape' in action_feature_data:
                    # Handle the specific format seen in your error message
                    shape_value = action_feature_data['shape']
                    try:
                        action_feature_spec = PolicyFeature(
                            type=FeatureType.ACTION,  # Use ACTION for actions
                            shape=tuple(shape_value)
                        )
                        print(
                            f"Created PolicyFeature with dtype format: {action_feature_spec}")
                    except Exception as e:
                        print(
                            f"Error creating PolicyFeature from dtype format: {e}, Data: {action_feature_data}")
                else:
                    print(
                        f"Dict missing expected keys. Available keys: {list(action_feature_data.keys())}")

            # Case 3: Object with type and shape attributes (duck typing for PolicyFeature)
            elif hasattr(action_feature_data, 'type') and hasattr(action_feature_data, 'shape'):
                action_feature_spec = action_feature_data  # Use as-is
                print(
                    f"Using object with type/shape attributes: {action_feature_spec}")

            # Final fallback - create from config if available
            if action_feature_spec is None and hasattr(self.config, 'action_feature'):
                print("Using action_feature from config as fallback")
                action_feature_spec = self.config.action_feature

            # Additional fallback - create from raw action dimension if we can infer it
            if action_feature_spec is None and isinstance(action_feature_data, dict) and 'shape' in action_feature_data:
                try:
                    # Create a simple PolicyFeature with just the shape
                    shape_value = action_feature_data['shape']
                    action_feature_spec = PolicyFeature(
                        type=FeatureType.ACTION,  # Use ACTION for robotic actions
                        shape=tuple(shape_value)
                    )
                    print(
                        f"Created emergency fallback PolicyFeature with shape {shape_value}")
                except Exception as e:
                    print(f"Error creating emergency fallback: {e}")

            # Last resort - manually create from inverse dynamics model dimensions
            if action_feature_spec is None:
                try:
                    # Get action dimension from the inverse dynamics model
                    action_dim = self.inverse_dynamics_model.a_dim
                    action_feature_spec = PolicyFeature(
                        type=FeatureType.ACTION,
                        shape=(action_dim,)
                    )
                    print(
                        f"Created last resort PolicyFeature from inverse dynamics a_dim={action_dim}")
                except Exception as e:
                    print(f"Failed to create last resort action feature: {e}")

            # Last chance: manually create a PolicyFeature based on dimensions from our helper function
            if action_feature_spec is None:
                try:
                    print("Using _safe_get_action_dim helper as last resort")
                    # Get action dimension using our helper function
                    action_dim = _safe_get_action_dim(
                        config=self.config,
                        inverse_dynamics_model=self.inverse_dynamics_model,
                        action_feature_data=action_feature_data
                    )
                    action_feature_spec = PolicyFeature(
                        type=FeatureType.ACTION,
                        shape=(action_dim,)
                    )
                    print(
                        f"Created final resort PolicyFeature with action_dim={action_dim}")
                except Exception as e:
                    print(f"Final attempt to create PolicyFeature failed: {e}")
                    raise TypeError(
                        f"Could not create PolicyFeature from action_feature_data: {type(action_feature_data)}. Data: {action_feature_data}")

            # At this point we should have a valid action_feature_spec or have raised an exception

            # Make sure we have a valid action feature spec before creating the Unnormalize instance
            if action_feature_spec is None:
                print(
                    "Warning: No valid action_feature_spec could be created. Using identity unnormalizer.")
                self.unnormalize_action_output = lambda x: x.get(
                    "action", x) if isinstance(x, dict) else x
            else:
                try:
                    self.unnormalize_action_output = Unnormalize(
                        {"action": action_feature_spec},
                        self.config.normalization_mapping,
                        processed_dataset_stats
                    )
                    print(
                        f"Successfully created unnormalizer with action_feature_spec: {action_feature_spec}")
                except Exception as e:
                    print(
                        f"Error creating unnormalizer: {e}. Using identity unnormalizer instead.")
                    self.unnormalize_action_output = lambda x: x.get(
                        "action", x) if isinstance(x, dict) else x
        else:
            print("Warning: Missing attributes 'input_features' or 'normalization_mapping' in config for normalizer creation. Using identity normalizers.")
            self.normalize_inputs = lambda x: x
            self.unnormalize_action_output = lambda x: x.get(
                "action", x) if isinstance(x, dict) else x

        self.n_obs_steps = n_obs_steps
        self.reset()

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()

        # Reset CL-DiffPhyCon state
        self.cl_diffphycon_state = None

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        raw_obs_on_device = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in current_raw_observation.items()
        }

        # Preprocess observation before normalization
        processed_obs = {}
        for k, v in raw_obs_on_device.items():
            if k == "observation.image" and v is not None:
                # Check if the image needs channel dimension adjustment for normalization
                if v.ndim == 4:  # [B, H, W, C] format
                    # Convert to [B, C, H, W] format for normalization
                    processed_obs[k] = v.permute(0, 3, 1, 2)
                    print(
                        f"Converted image from shape {v.shape} to {processed_obs[k].shape}")
                elif v.ndim == 3:  # [H, W, C] format
                    # Convert to [C, H, W] format for normalization
                    processed_obs[k] = v.permute(2, 0, 1).unsqueeze(0)
                    print(
                        f"Converted image from shape {v.shape} to {processed_obs[k].shape}")
                else:
                    # Use as is
                    processed_obs[k] = v
                    print(f"Using image with existing shape: {v.shape}")
            else:
                processed_obs[k] = v

        try:
            # 1. Normalize Current Observation
            normalized_obs_batch = self.normalize_inputs(processed_obs)
            print(
                f"Normalization succeeded with keys: {list(normalized_obs_batch.keys())}")
        except RuntimeError as e:
            print(f"Error during normalization: {e}")
            print(f"Raw observation keys: {list(raw_obs_on_device.keys())}")
            for k, v in raw_obs_on_device.items():
                if isinstance(v, torch.Tensor):
                    print(f"  {k} shape: {v.shape}, dtype: {v.dtype}")

            # Fallback to using raw observations if normalization fails
            print("Using raw observations as fallback due to normalization error")
            normalized_obs_batch = processed_obs

        main_state_key = OBS_ROBOT
        image_keys_in_config = list(
            self.config.image_features.keys()) if self.config.image_features else []
        # 瞰 Simplification
        main_image_key = image_keys_in_config[0] if image_keys_in_config else None

        if main_state_key not in normalized_obs_batch:
            raise KeyError(
                f"Normalized batch missing required state key: {main_state_key}")
        norm_state_current = normalized_obs_batch[main_state_key]

        norm_img_current = None
        # Check if BidirARTransformer actually uses images from its own config
        bidir_input_features = getattr(
            self.bidirectional_transformer.config, 'input_features', {})
        bidir_uses_image = any(k.startswith("observation.image")
                               for k in bidir_input_features)

        if bidir_uses_image:
            if main_image_key and main_image_key in normalized_obs_batch:
                norm_img_current = normalized_obs_batch[main_image_key]
            elif "observation.image" in normalized_obs_batch:
                norm_img_current = normalized_obs_batch["observation.image"]
            else:
                raise KeyError(
                    "Normalized batch missing required image key for BidirectionalARTransformer.")

        if norm_state_current.ndim == 1:
            norm_state_current = norm_state_current.unsqueeze(0)
        if norm_img_current is not None and norm_img_current.ndim == 3:
            norm_img_current = norm_img_current.unsqueeze(0)

        self._obs_state_queue.append(norm_state_current.unsqueeze(1))

        # Use norm_img_current for queue consistency
        if self.config.image_features and bidir_uses_image and norm_img_current is not None:
            # This part is for the state_diffusion_model's conditioning, which expects stacked cameras
            current_all_norm_images_stacked_for_queue = []
            for img_key in self.config.image_features:  # These are keys like "observation.image.cam1"
                if img_key in normalized_obs_batch:
                    img_tensor = normalized_obs_batch[img_key]
                    if img_tensor.ndim == 3:
                        img_tensor = img_tensor.unsqueeze(0)
                    current_all_norm_images_stacked_for_queue.append(
                        img_tensor)  # List of [B,C,H,W]

            if current_all_norm_images_stacked_for_queue:
                stacked_cams_for_step = torch.stack(
                    current_all_norm_images_stacked_for_queue, dim=1)  # [B, N_cam, C, H, W]
                self._obs_image_queue.append(
                    stacked_cams_for_step.unsqueeze(1))  # [B, 1, N_cam, C, H, W]

        if self._action_execution_queue:  # This queue should store normalized actions
            next_normalized_action = self._action_execution_queue.popleft()
            return self.unnormalize_action_output({"action": next_normalized_action})["action"]

        if len(self._obs_state_queue) < self.n_obs_steps:
            # Get action dimension safely from multiple sources
            action_dim = _safe_get_action_dim(
                config=self.config,
                inverse_dynamics_model=self.inverse_dynamics_model
            )
            print(f"Using action_dim={action_dim} for zero action")

            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]

        obs_history_state = torch.cat(list(self._obs_state_queue), dim=1)

        # Prepare conditioning information for the diffusion model
        observation_batch_for_cond = {OBS_ROBOT: obs_history_state}
        if self.config.image_features and self._obs_image_queue:
            obs_history_img_stacked = torch.cat(
                list(self._obs_image_queue), dim=1)
            image_key = OBS_IMAGE
            observation_batch_for_cond[image_key] = obs_history_img_stacked

        # CL-DiffPhyCon implementation
        if self.use_cl_diffphycon:
            print("Using CL-DiffPhyCon algorithm")

            # Initialize CL-DiffPhyCon state if needed
            if self.cl_diffphycon_state is None:
                print("Initializing CL-DiffPhyCon state")
                self.cl_diffphycon_state = self.state_diffusion_model.initialize_cl_diffphycon_state(
                    observation_batch_for_cond
                )

            # Perform a CL-DiffPhyCon step
            self.cl_diffphycon_state = self.state_diffusion_model.sample_cl_diffphycon_step(
                self.cl_diffphycon_state,
                step_size=self.cl_diffphycon_step_size
            )

            # Get the current plan from the CL-DiffPhyCon state
            refined_state_plan_normalized = self.cl_diffphycon_state["x_t"]
            print(
                f"CL-DiffPhyCon timestep: {self.cl_diffphycon_state['timestep']}, finished: {self.cl_diffphycon_state['finished']}")

        else:
            # Fall back to the original implementation
            print("Using original bidirectional + diffusion refinement")

            bidir_config_state_dim = self.bidirectional_transformer.config.state_dim
            if norm_state_current.shape[-1] != bidir_config_state_dim:
                norm_state_for_bidir = norm_state_current[:,
                                                          :bidir_config_state_dim]
            else:
                norm_state_for_bidir = norm_state_current

            # Ensure norm_img_current is not None if BidirARTransformer expects an image
            if bidir_uses_image and norm_img_current is None:
                raise ValueError(
                    "BidirectionalARTransformer requires an image but norm_img_current is None.")

            transformer_predictions = self.bidirectional_transformer(
                initial_images=norm_img_current,
                initial_states=norm_state_for_bidir,
                training=False
            )
            norm_predicted_future_states = transformer_predictions['predicted_forward_states']
            initial_state_plan_normalized = torch.cat(
                [norm_state_for_bidir.unsqueeze(1), norm_predicted_future_states], dim=1
            )

            diffusion_horizon = self.state_diffusion_model.config.horizon
            initial_state_plan_for_diffusion = initial_state_plan_normalized[:,
                                                                             :diffusion_horizon, :]

            try:
                # Try to run the diffusion refinement
                refined_state_plan_normalized = self.state_diffusion_model.diffusion.refine_state_path(
                    initial_state_path=initial_state_plan_for_diffusion,
                    observation_batch_for_cond=observation_batch_for_cond
                )
                print("✅ Diffusion refinement completed successfully!")
            except RuntimeError as e:
                print(f"Error during diffusion refinement: {e}")
                refined_state_plan_normalized = initial_state_plan_for_diffusion

        # Run inverse dynamics on the refined plan
        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim
        current_norm_state_for_invdyn = norm_state_current

        if current_norm_state_for_invdyn.shape[-1] != inv_dyn_state_dim:
            current_norm_state_for_invdyn = current_norm_state_for_invdyn[:,
                                                                          :inv_dyn_state_dim]

        refined_plan_for_invdyn = refined_state_plan_normalized
        if refined_plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            if refined_plan_for_invdyn.shape[-1] < inv_dyn_state_dim:
                padding_size = inv_dyn_state_dim - \
                    refined_plan_for_invdyn.shape[-1]
                padding = torch.zeros(
                    refined_plan_for_invdyn.shape[0],
                    refined_plan_for_invdyn.shape[1],
                    padding_size,
                    device=self.device
                )
                refined_plan_for_invdyn = torch.cat(
                    [refined_plan_for_invdyn, padding], dim=-1)
            else:
                refined_plan_for_invdyn = refined_plan_for_invdyn[:,
                                                                  :, :inv_dyn_state_dim]

        num_planned_actions = refined_plan_for_invdyn.shape[1]
        if num_planned_actions == 0:
            # Use safe helper to get action dimension
            action_dim = _safe_get_action_dim(
                config=self.config,
                inverse_dynamics_model=self.inverse_dynamics_model
            )
            print(
                f"No planned actions. Using action_dim={action_dim} for zero action")
            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]

        actions_normalized_list = []
        first_state_for_pair = current_norm_state_for_invdyn
        for i in range(num_planned_actions):
            second_state_for_pair = refined_plan_for_invdyn[:, i, :]
            state_pair = torch.cat(
                [first_state_for_pair, second_state_for_pair], dim=-1)
            action_i_normalized = self.inverse_dynamics_model(state_pair)
            actions_normalized_list.append(action_i_normalized)
            first_state_for_pair = second_state_for_pair

        actions_normalized_sequence = torch.stack(
            actions_normalized_list, dim=1)

        for i in range(actions_normalized_sequence.shape[1]):
            self._action_execution_queue.append(
                actions_normalized_sequence[:, i, :])

        if self._action_execution_queue:
            next_normalized_action = self._action_execution_queue.popleft()
            return self.unnormalize_action_output({"action": next_normalized_action})["action"]
        else:
            # Use safe helper to get action dimension
            action_dim = _safe_get_action_dim(
                config=self.config,
                inverse_dynamics_model=self.inverse_dynamics_model
            )
            print(
                f"Action execution queue empty. Using action_dim={action_dim} for zero action")
            zero_norm_action = torch.zeros(
                (norm_state_current.shape[0], action_dim), device=self.device)
            return self.unnormalize_action_output({"action": zero_norm_action})["action"]

    def _prepare_simplified_conditioning(self, obs_history_state):
        """
        Prepare simplified conditioning for the diffusion model - 
        only passes state (no images) to avoid dimension mismatch issues.
        """
        print("Using simplified conditioning with state only")

        # Get state dimension expected by the diffusion model
        # Either from config or from the actual state
        target_state_dim = None
        if hasattr(self.state_diffusion_model.config, 'robot_state_feature'):
            target_state_dim = getattr(
                self.state_diffusion_model.config.robot_state_feature, 'shape', None)
            if target_state_dim:
                if isinstance(target_state_dim, (list, tuple)):
                    target_state_dim = target_state_dim[0]
                print(f"Using state dim from config: {target_state_dim}")

        # If we couldn't get from config, use the input state's dimension
        if target_state_dim is None:
            target_state_dim = obs_history_state.shape[-1]
            print(f"Using state dim from input: {target_state_dim}")

        # Ensure the state has the right dimensions
        # If too small, pad with zeros
        # If too large, truncate
        current_state_dim = obs_history_state.shape[-1]

        processed_state = obs_history_state
        if current_state_dim != target_state_dim:
            if current_state_dim < target_state_dim:
                # Pad with zeros to match the expected dimension
                padding = torch.zeros(
                    *obs_history_state.shape[:-1],
                    target_state_dim - current_state_dim,
                    device=obs_history_state.device
                )
                processed_state = torch.cat(
                    [obs_history_state, padding], dim=-1)
                print(
                    f"Padded state from {current_state_dim} to {target_state_dim}")
            else:
                # Truncate to the expected dimension
                processed_state = obs_history_state[..., :target_state_dim]
                print(
                    f"Truncated state from {current_state_dim} to {target_state_dim}")

        # Check if the processed state has the right shape for conditioning
        # The diffusion model expects a specific shape for conditioning
        # Get the expected n_obs_steps from the diffusion model config
        expected_n_obs_steps = self.state_diffusion_model.config.n_obs_steps
        current_n_obs_steps = processed_state.shape[1]

        # Ensure the history has the right number of steps
        if current_n_obs_steps != expected_n_obs_steps:
            if current_n_obs_steps < expected_n_obs_steps:
                # If we have fewer steps than expected, duplicate the last step
                steps_to_add = expected_n_obs_steps - current_n_obs_steps
                last_step = processed_state[:, -1:, :]  # [B, 1, D]
                padding = last_step.repeat(
                    1, steps_to_add, 1)  # [B, steps_to_add, D]
                processed_state = torch.cat([processed_state, padding], dim=1)
                print(
                    f"Extended history from {current_n_obs_steps} to {expected_n_obs_steps} steps")
            else:
                # If we have more steps than expected, take the most recent ones
                processed_state = processed_state[:, -expected_n_obs_steps:, :]
                print(
                    f"Truncated history from {current_n_obs_steps} to {expected_n_obs_steps} steps")

        # Print shapes for debugging
        print(
            f"Final processed state shape for conditioning: {processed_state.shape}")

        # Get the expected global_cond_dim from the transformer
        transformer = self.state_diffusion_model.diffusion.transformer
        if hasattr(transformer, 'cond_embed'):
            global_cond_dim = transformer.cond_embed.in_features
            time_embed_dim = transformer.time_embed[-1].out_features
            expected_global_cond_dim = global_cond_dim - time_embed_dim
            print(
                f"Transformer expects global_cond_dim: {global_cond_dim}, time_embed_dim: {time_embed_dim}, expected_global_cond_dim: {expected_global_cond_dim}")

            # Flatten history for conditioning
            flattened_state_dim = processed_state.shape[-1] * \
                processed_state.shape[1]
            print(f"Flattened state dim: {flattened_state_dim}")

            # If the flattened state doesn't match the expected dimension, adjust it
            if flattened_state_dim != expected_global_cond_dim:
                print(
                    f"Warning: Flattened state dim ({flattened_state_dim}) doesn't match expected conditioning dim ({expected_global_cond_dim})")

                # Try a better approach: check what the _prepare_global_conditioning expects
                if hasattr(self.state_diffusion_model.diffusion, '_prepare_global_conditioning'):
                    print(
                        "Using the diffusion model's _prepare_global_conditioning logic")
                    # This creates a simpler conditioning dict that lets the diffusion model handle the preparation
                    return {OBS_ROBOT: processed_state}

        # Create a simple conditioning dict with just the state
        return {OBS_ROBOT: processed_state}

    def _skip_diffusion_refinement(self, initial_state_path):
        """
        Skip diffusion refinement step and return the initial path directly.
        This works as a fallback when the conditioning dimensions don't match.
        """
        print("⚠️ SKIPPING DIFFUSION REFINEMENT due to dimension mismatch issues!")
        print(f"Initial state path shape: {initial_state_path.shape}")

        # Log information about the expected transformer dimensions
        try:
            transformer = self.state_diffusion_model.diffusion.transformer
            time_embed_dim = transformer.time_embed[-1].out_features if hasattr(
                transformer, 'time_embed') else "unknown"
            cond_embed_in_features = transformer.cond_embed.in_features if hasattr(
                transformer, 'cond_embed') else "unknown"
            print(
                f"Transformer expects: time_embed_dim={time_embed_dim}, cond_embed.in_features={cond_embed_in_features}")
        except Exception as e:
            print(f"Could not inspect transformer dimensions: {e}")

        # Just return the initial path unmodified
        return initial_state_path

    def _prepare_correct_conditioning(self, obs_history_state):
        """
        Prepare global conditioning for the diffusion model that exactly matches the expected dimensions.
        This is a more direct approach than _prepare_simplified_conditioning, explicitly computing
        the expected dimensions and reshaping accordingly.
        """
        print("Preparing exact dimensional conditioning for diffusion model")

        try:
            # Get the expected dimensions from the transformer
            transformer = self.state_diffusion_model.diffusion.transformer

            # Get transformer input dimensions
            cond_in_features = transformer.cond_embed.in_features if hasattr(
                transformer, 'cond_embed') else None
            time_embed_dim = transformer.time_embed[-1].out_features if hasattr(
                transformer, 'time_embed') else None

            if cond_in_features is None or time_embed_dim is None:
                print(
                    "Could not determine transformer dimensions, falling back to simplified conditioning")
                return self._prepare_simplified_conditioning(obs_history_state)

            # Calculate expected global conditioning dimension (total minus time embedding)
            expected_global_cond_dim = cond_in_features - time_embed_dim
            print(
                f"Transformer expects global_cond_dim: {expected_global_cond_dim}")

            # Get state dimension from config
            if hasattr(self.state_diffusion_model.config, 'robot_state_feature'):
                state_dim = getattr(
                    self.state_diffusion_model.config.robot_state_feature, 'shape', [None])[0]
                if state_dim is None:
                    state_dim = obs_history_state.shape[-1]
            else:
                state_dim = obs_history_state.shape[-1]

            # Get expected observation steps
            expected_n_obs_steps = self.state_diffusion_model.config.n_obs_steps

            # Expected per-step feature dimension
            expected_per_step_dim = expected_global_cond_dim // expected_n_obs_steps
            print(f"Expected per-step feature dim: {expected_per_step_dim}")

            # Check if we need to adjust the state dimension
            if state_dim != expected_per_step_dim:
                print(
                    f"State dimension ({state_dim}) doesn't match expected per-step dimension ({expected_per_step_dim})")

                # Adjust state dimension
                if state_dim < expected_per_step_dim:
                    # Pad with zeros to match expected dimension
                    padding = torch.zeros(
                        *obs_history_state.shape[:-1],
                        expected_per_step_dim - state_dim,
                        device=obs_history_state.device
                    )
                    processed_state = torch.cat(
                        [obs_history_state, padding], dim=-1)
                    print(
                        f"Padded state from {state_dim} to {expected_per_step_dim}")
                else:
                    # Truncate to expected dimension
                    processed_state = obs_history_state[...,
                                                        :expected_per_step_dim]
                    print(
                        f"Truncated state from {state_dim} to {expected_per_step_dim}")
            else:
                processed_state = obs_history_state

            # Ensure we have the right number of observation steps
            current_n_obs_steps = processed_state.shape[1]
            if current_n_obs_steps != expected_n_obs_steps:
                if current_n_obs_steps < expected_n_obs_steps:
                    # Repeat the last state to fill
                    steps_to_add = expected_n_obs_steps - current_n_obs_steps
                    last_step = processed_state[:, -1:, :]
                    padding = last_step.repeat(1, steps_to_add, 1)
                    processed_state = torch.cat(
                        [processed_state, padding], dim=1)
                    print(
                        f"Extended from {current_n_obs_steps} to {expected_n_obs_steps} steps")
                else:
                    # Take most recent steps
                    processed_state = processed_state[:, -
                                                      expected_n_obs_steps:, :]
                    print(
                        f"Truncated from {current_n_obs_steps} to {expected_n_obs_steps} steps")

            # Final verification - ensure exact dimension match
            flattened_dim = processed_state.shape[1] * processed_state.shape[2]
            if flattened_dim != expected_global_cond_dim:
                print(
                    f"Warning: After adjustments, flattened dim ({flattened_dim}) still doesn't match expected ({expected_global_cond_dim})")

                # Force exact match by resizing the final dimension as needed
                # This is a more direct approach to ensure the dimensions match exactly
                batch_size = processed_state.shape[0]
                reshaped_state = processed_state.reshape(
                    batch_size, -1)  # Flatten to [B, flattened_dim]

                if flattened_dim < expected_global_cond_dim:
                    # Pad with zeros to reach the exact expected dimension
                    padding = torch.zeros(
                        batch_size,
                        expected_global_cond_dim - flattened_dim,
                        device=processed_state.device
                    )
                    reshaped_state = torch.cat(
                        [reshaped_state, padding], dim=1)
                    print(
                        f"Added final padding to get exact dimension: {expected_global_cond_dim}")
                else:
                    # Truncate to the exact expected dimension
                    reshaped_state = reshaped_state[:,
                                                    :expected_global_cond_dim]
                    print(
                        f"Truncated to get exact dimension: {expected_global_cond_dim}")

                # Try to maintain a sensible structure by reshaping back to [B, steps, dim_per_step]
                # Only if it divides evenly, otherwise keep it flat
                if expected_global_cond_dim % expected_n_obs_steps == 0:
                    dim_per_step = expected_global_cond_dim // expected_n_obs_steps
                    processed_state = reshaped_state.reshape(
                        batch_size, expected_n_obs_steps, dim_per_step)
                    print(
                        f"Reshaped to structure: [B, {expected_n_obs_steps}, {dim_per_step}]")
                else:
                    # Can't reshape nicely, let the _prepare_global_conditioning handle flattening
                    processed_state = reshaped_state.reshape(
                        batch_size, 1, expected_global_cond_dim)
                    print(
                        f"Reshaped to structure: [B, 1, {expected_global_cond_dim}]")

            print(
                f"Final processed state shape: {processed_state.shape}, will flatten to {processed_state.shape[1] * processed_state.shape[2]}")

            # Sanity check - must match exactly
            assert processed_state.shape[1] * processed_state.shape[2] == expected_global_cond_dim, \
                "Conditioning dimensions still don't match after processing!"

            # Create conditioning dict
            return {OBS_ROBOT: processed_state}

        except Exception as e:
            print(f"Error preparing exact dimensional conditioning: {e}")
            print("Falling back to simplified conditioning")
            return self._prepare_simplified_conditioning(obs_history_state)
