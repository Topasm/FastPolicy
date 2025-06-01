from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
import einops
from typing import Dict

from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE, OBS_ENV
from lerobot.configs.types import FeatureType
from lerobot.common.datasets.utils import PolicyFeature

from model.predictor.bidirectional_autoregressive_transformer import BidirectionalARTransformer
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.invdyn.invdyn import MlpInvDynamic


class BidirectionalRTDiffusionPolicy(nn.Module):
    """
    Combined policy class that integrates:
    1. Bidirectional Transformer for initial plan generation
    2. Diffusion model for plan refinement
    3. Inverse dynamics model for action prediction from states
    """

    def __init__(
        self,
        bidirectional_transformer: BidirectionalARTransformer,
        state_diffusion_model: CLDiffPhyConModel,
        inverse_dynamics_model: MlpInvDynamic,
        dataset_stats: dict,
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
        self.use_cl_diffphycon = True
        # Store previous step's noisy latent sequence
        self.cl_diffphycon_latent_z_prev = None

        # Process dataset stats for normalization
        processed_stats = self._process_dataset_stats(dataset_stats)

        # Create normalizers
        self.normalize_inputs, self.unnormalize_action_output = self._create_normalizers(
            all_dataset_features, processed_stats)

        # Initialize queues
        self.n_obs_steps = n_obs_steps

        # Initialize RGB encoder if images are used
        if self.config.image_features:
            from model.diffusion.diffusion_modules import DiffusionRgbEncoder
            self.rgb_encoder = DiffusionRgbEncoder(self.config)
            self.rgb_encoder.to(self.device)

        self.reset()
        # Initialize queues
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    def _process_dataset_stats(self, dataset_stats):
        """Process raw dataset statistics into tensor format for normalizers."""
        processed_stats = {}
        if dataset_stats is None:
            print("Warning: No dataset_stats provided to BidirectionalRTDiffusionPolicy.")
            return processed_stats

        for key, stat_group in dataset_stats.items():
            processed_stats[key] = {}
            if isinstance(stat_group, dict):
                for subkey, subval in stat_group.items():
                    try:
                        processed_stats[key][subkey] = torch.as_tensor(
                            subval, dtype=torch.float32, device=self.device)
                    except Exception:
                        processed_stats[key][subkey] = subval
            else:
                processed_stats[key] = stat_group

        return processed_stats

    def _create_normalizers(self, all_features, processed_stats):
        """Create input normalizer and action unnormalizer."""
        # Default no-op normalizers
        def normalize_fn(x): return x
        def unnormalize_fn(x): return x.get(
            "action", x) if isinstance(x, dict) else x

        if not hasattr(self.config, 'input_features') or not hasattr(self.config, 'normalization_mapping'):
            print("Warning: Missing required config attributes for normalization.")
            return normalize_fn, unnormalize_fn

        # Create normalizer for inputs
        try:
            # Prepare input features for normalizer
            valid_input_features = {}
            for k, v_feat in self.config.input_features.items():
                if isinstance(v_feat, dict) and 'type' in v_feat and 'shape' in v_feat:
                    try:
                        valid_input_features[k] = PolicyFeature(
                            type=FeatureType(v_feat['type']),
                            shape=tuple(v_feat['shape']))
                    except Exception as e:
                        print(f"Warning: Could not convert feature {k}: {e}")
                elif hasattr(v_feat, 'type') and hasattr(v_feat, 'shape'):
                    valid_input_features[k] = v_feat

            normalize_fn = Normalize(
                valid_input_features,
                self.config.normalization_mapping,
                processed_stats
            )
        except Exception as e:
            print(f"Error creating input normalizer: {e}")

        # Create unnormalizer for actions
        try:
            action_feature_data = all_features.get("action")
            if action_feature_data:
                if isinstance(action_feature_data, dict) and 'type' in action_feature_data and 'shape' in action_feature_data:
                    action_feature = PolicyFeature(
                        type=FeatureType(action_feature_data['type']),
                        shape=tuple(action_feature_data['shape'])
                    )
                    unnormalize_fn = Unnormalize(
                        {"action": action_feature},
                        self.config.normalization_mapping,
                        processed_stats
                    )
        except Exception as e:
            print(f"Error creating action unnormalizer: {e}")

        return normalize_fn, unnormalize_fn

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()
        self.cl_diffphycon_latent_z_prev = None

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """Select an action given the current observation."""
        # Move tensors to device
        raw_obs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in current_raw_observation.items()}

        # Process and normalize observation
        processed_obs = self._preprocess_observation(raw_obs)
        normalized_obs = self._normalize_observation(processed_obs)

        self._queues = populate_queues(
            self._queues, normalized_obs)
        # Generate new action plan only when the action queue is empty
        if len(self._queues["action"]) == 0:
            # Check if we have enough history to make a prediction
            required_obs = self.config.n_obs_steps
            if len(self._queues["observation.state"]) < required_obs:
                print(
                    f"Warning: Not enough history in queues. Have {len(self._queues['observation.state'])}, need {required_obs}")
                # Return zero action if we don't have enough history yet
                return torch.zeros((1, self.config.action_feature.shape[0]), device=self.device)

            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

        # If we already have actions queued, return the next one
        if self._action_execution_queue:
            return self._get_next_action()

        # # If we don't have enough history yet, return zero action
        # if len(self._obs_state_queue) < self.n_obs_steps:
        #     return self._get_zero_action(norm_state.shape[0])

        #        Args:
        # model_input_batch: Dict with observation history
        # batch_size: Batch size for diffusion model

        # Extract current state from the model input batch for inverse dynamics
        # Get the last state in the sequence
        norm_state = model_input_batch["observation.state"][:, -1, :]

        # Generate refined state plan
        refined_state_plan = self._generate_state_plan(model_input_batch)

        # Generate actions using inverse dynamics
        actions = self._generate_actions_from_states(
            norm_state, refined_state_plan)

        # Queue up actions for execution
        for i in range(actions.shape[1]):
            # Store each action as a tensor of shape [action_dim], not [batch_size, action_dim]
            # This way the queue contains simple action vectors
            self._action_execution_queue.append(
                actions[0, i, :])  # Assuming batch_size=1

        # Print queue size for debugging
        print(
            f"Queued {len(self._action_execution_queue)} actions for execution")

        # Return the first action
        if self._action_execution_queue:
            return self._get_next_action()
        else:
            return self._get_zero_action(norm_state.shape[0])

    def _preprocess_observation(self, raw_obs):
        """Preprocess raw observations, particularly adjusting image dimensions."""
        processed_obs = {}
        for k, v in raw_obs.items():
            if k == "observation.image" and v is not None:
                # Handle image format conversions
                if v.ndim == 4:  # [B, H, W, C]
                    processed_obs[k] = v.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                elif v.ndim == 3:  # [H, W, C]
                    processed_obs[k] = v.permute(
                        2, 0, 1).unsqueeze(0)  # -> [1, C, H, W]
                else:
                    processed_obs[k] = v
            else:
                processed_obs[k] = v

        return processed_obs

    def _normalize_observation(self, processed_obs):
        """Normalize preprocessed observations."""
        try:
            return self.normalize_inputs(processed_obs)
        except Exception as e:
            print(f"Error during normalization: {e}")
            return processed_obs  # Fallback to unnormalized

    # def _update_queues(self, norm_state, norm_img, normalized_obs):
    #     """Update observation history queues."""
    #     # Add state to queue
    #     self._obs_state_queue.append(norm_state.unsqueeze(1))

    #     # Add image to queue if available
    #     if self.config.image_features and norm_img is not None:
    #         # Stack multiple cameras if needed
    #         image_tensors = []
    #         for img_key in self.config.image_features:
    #             if img_key in normalized_obs:
    #                 img_tensor = normalized_obs[img_key]
    #                 if img_tensor.ndim == 3:
    #                     img_tensor = img_tensor.unsqueeze(0)
    #                 image_tensors.append(img_tensor)

    #         if image_tensors:
    #             stacked_cameras = torch.stack(
    #                 image_tensors, dim=1)  # [B, N_cam, C, H, W]
    #             self._obs_image_queue.append(
    #                 stacked_cameras.unsqueeze(1))  # [B, 1, N_cam, C, H, W]

    def _get_next_action(self):
        """Get the next normalized action from the queue and unnormalize it."""
        next_action = self._action_execution_queue.popleft()
        print(f"Action shape before unsqueeze: {next_action.shape}")
        # Handle different shapes - ensure it's a batch
        if len(next_action.shape) == 1:
            next_action = next_action.unsqueeze(0)
        print(f"Action shape after unsqueeze: {next_action.shape}")

        # Try to unnormalize the action
        try:
            # Make sure we're passing a properly formatted dictionary to unnormalize_action_output
            action_dict = {"action": next_action}
            unnorm_action = self.unnormalize_action_output(action_dict)

            # Check if the result is a dictionary or tensor
            if isinstance(unnorm_action, dict) and "action" in unnorm_action:
                print(
                    f"Unnormalized action successfully with keys: {unnorm_action.keys()}")
                return unnorm_action["action"]
            else:
                print(
                    f"Unnormalized action returned type: {type(unnorm_action)}")
                return unnorm_action
        except Exception as e:
            print(f"Error unnormalizing action: {e}")
            # Return the action as-is in case of error
            return next_action

    def _get_zero_action(self, batch_size):
        """Get a zero action tensor with the correct dimensions."""
        # Get action dimension from inverse dynamics model or config
        if hasattr(self.inverse_dynamics_model, 'a_dim'):
            action_dim = self.inverse_dynamics_model.a_dim
        elif hasattr(self.config, 'action_feature') and hasattr(self.config.action_feature, 'shape'):
            action_dim = self.config.action_feature.shape[0]
        else:
            # Last resort fallback
            print("Warning: Using fallback method to determine action_dim")
            action_dim = 2  # Default minimal value

        zero_action = torch.zeros((batch_size, action_dim), device=self.device)
        try:
            unnorm_result = self.unnormalize_action_output(
                {"action": zero_action})
            # Check if the result is a dictionary or tensor
            if isinstance(unnorm_result, dict) and "action" in unnorm_result:
                return unnorm_result["action"]
            else:
                return unnorm_result
        except Exception as e:
            print(f"Error in unnormalizing zero action: {e}")
            return zero_action

    def _generate_state_plan(self, model_input_batch):
        """Generate refined state plan using bidirectional transformer and diffusion.

        Args:
            model_input_batch: Dict containing observation history with keys like "observation.state", "observation.image"
        """
        # Extract current state and image
        # Get the last state
        norm_state_current = model_input_batch["observation.state"][:, -1, :]
        norm_img_current = None
        if "observation.image" in model_input_batch:
            # Extract last image from the sequence
            norm_img_current = model_input_batch["observation.image"][:, -1, :, :, :]

        if self.use_cl_diffphycon:
            return self._generate_cl_diffphycon_plan(
                norm_state_current,
                norm_img_current,
                model_input_batch
            )
        else:
            return self._generate_standard_plan(norm_state_current, norm_img_current, model_input_batch)

    def _generate_cl_diffphycon_plan(self, norm_state_current, norm_img_current, observation_batch_for_cond_history):
        """Generate state plan using CL-DiffPhyCon algorithm.

        Args:
            norm_state_current: Current normalized state (u_env,τ-1) from select_action
            norm_img_current: Current normalized image (for BidirectionalARTransformer) from select_action
            observation_batch_for_cond_history: Dict of historical observations for conditioning diffusion
        """
        print("Using CL-DiffPhyCon algorithm")

        # Initialize CL-DiffPhyCon latent state if needed (first call in an episode)
        if self.cl_diffphycon_latent_z_prev is None:
            print("Initializing CL-DiffPhyCon latent_z_prev")

            # Step 1: Generate Initial Clean Plan ("Sync Path") using Bidirectional Transformer
            # Prepare state for bidirectional transformer
            bidir_state_dim = self.bidirectional_transformer.config.state_dim
            current_state_for_bidir = norm_state_current
            if norm_state_current.shape[-1] != bidir_state_dim:
                current_state_for_bidir = norm_state_current[:,
                                                             :bidir_state_dim]

            # Generate initial plan with transformer using current observations
            tf_predictions = self.bidirectional_transformer(
                initial_images=norm_img_current,
                initial_states=current_state_for_bidir,
                training=False
            )

            # Extract predicted states and create initial plan
            norm_predicted_states = tf_predictions['predicted_forward_states']
            initial_clean_plan = torch.cat(
                [current_state_for_bidir.unsqueeze(1), norm_predicted_states], dim=1
            )

            # Adjust plan length to match diffusion horizon
            diffusion_horizon = self.state_diffusion_model.config.horizon
            if initial_clean_plan.shape[1] > diffusion_horizon:
                initial_clean_plan_for_diffusion = initial_clean_plan[:,
                                                                      :diffusion_horizon, :]
            elif initial_clean_plan.shape[1] < diffusion_horizon:
                # Handle cases where the plan is shorter than diffusion horizon
                padding_size = diffusion_horizon - initial_clean_plan.shape[1]
                # Pad with the last state
                last_state_in_plan = initial_clean_plan[:, -1:, :]
                padding = last_state_in_plan.repeat(1, padding_size, 1)
                initial_clean_plan_for_diffusion = torch.cat(
                    [initial_clean_plan, padding], dim=1)
            else:
                initial_clean_plan_for_diffusion = initial_clean_plan

            # Step 2: Initialize the Asynchronously Noised Latent State
            self.cl_diffphycon_latent_z_prev = self.state_diffusion_model.initialize_cl_diffphycon_state(
                initial_clean_plan_for_diffusion
            )

        # Step 3: Perform CL-DiffPhyCon Asynchronous Denoising Step
        # observation_batch_for_cond_history is used by _prepare_global_conditioning inside sample_cl_diffphycon_step
        current_step_predicted_state_at_t0, next_step_latent_z = \
            self.state_diffusion_model.sample_cl_diffphycon_step(
                self.cl_diffphycon_latent_z_prev,
                observation_batch_for_cond_history
            )

        # Step 4: Update the stored noisy latent for the next environment step
        self.cl_diffphycon_latent_z_prev = next_step_latent_z

        # Step 5: Return the fully denoised state prediction for the current step
        return current_step_predicted_state_at_t0

    def _generate_standard_plan(self, norm_state_current, norm_img_current, observation_batch_for_cond_history):
        """Generate state plan using bidirectional transformer and diffusion refinement.

        Args:
            norm_state_current: Current normalized state (latest state in the sequence) 
            norm_img_current: Current normalized image (latest image in the sequence)
            observation_batch_for_cond_history: Dict containing observation history
        """
        print("Using original bidirectional + diffusion refinement")

        # Prepare state for bidirectional transformer
        bidir_state_dim = self.bidirectional_transformer.config.state_dim
        if norm_state_current.shape[-1] != bidir_state_dim:
            norm_state_for_bidir = norm_state_current[:, :bidir_state_dim]
        else:
            norm_state_for_bidir = norm_state_current

        # Check if image is required but missing
        bidir_uses_image = any(k.startswith("observation.image")
                               for k in getattr(self.bidirectional_transformer.config, 'input_features', {}))
        if bidir_uses_image and norm_img_current is None:
            raise ValueError(
                "BidirectionalARTransformer requires an image but none is available")

        # Generate initial state plan with transformer
        transformer_predictions = self.bidirectional_transformer(
            initial_images=norm_img_current,
            initial_states=norm_state_for_bidir,
            training=False
        )

        # Extract predicted states and create initial plan
        norm_predicted_states = transformer_predictions['predicted_forward_states']
        initial_state_plan = torch.cat(
            [norm_state_for_bidir.unsqueeze(1), norm_predicted_states], dim=1
        )

        # Truncate to diffusion horizon if needed
        diffusion_horizon = self.state_diffusion_model.config.horizon
        initial_state_plan = initial_state_plan[:, :diffusion_horizon, :]

        # Refine plan with diffusion model
        try:
            refined_state_plan = self.state_diffusion_model.diffusion.refine_state_path(
                initial_state_path=initial_state_plan,
                observation_batch_for_cond=observation_batch_for_cond_history
            )
            print("✅ Diffusion refinement completed successfully!")
        except RuntimeError as e:
            print(f"Error during diffusion refinement: {e}")
            refined_state_plan = initial_state_plan

        return refined_state_plan

    def _generate_actions_from_states(self, norm_state, state_plan):
        """Generate actions from state plan using inverse dynamics."""
        # Prepare state dimensions for inverse dynamics
        inv_dyn_state_dim = self.inverse_dynamics_model.o_dim

        # Adjust current state if needed
        current_state = norm_state
        if current_state.shape[-1] != inv_dyn_state_dim:
            current_state = current_state[:, :inv_dyn_state_dim]

        # Handle different shapes of state_plan
        # For CL-DiffPhyCon, state_plan is typically [batch_size, state_dim]
        # For standard diffusion, state_plan is typically [batch_size, seq_len, state_dim]
        plan_for_invdyn = state_plan

        # Check if plan is a sequence or a single state
        if len(plan_for_invdyn.shape) == 2:  # [batch_size, state_dim]
            print(
                f"Plan shape is 2D: {plan_for_invdyn.shape}, expanding to sequence with len=1")
            # Make it a sequence of length 1: [batch_size, 1, state_dim]
            plan_for_invdyn = plan_for_invdyn.unsqueeze(1)

        # Adjust dimensions if needed
        if plan_for_invdyn.shape[-1] != inv_dyn_state_dim:
            if plan_for_invdyn.shape[-1] < inv_dyn_state_dim:
                # Pad if too small
                padding_size = inv_dyn_state_dim - plan_for_invdyn.shape[-1]
                padding = torch.zeros(
                    plan_for_invdyn.shape[0],
                    plan_for_invdyn.shape[1],
                    padding_size,
                    device=self.device
                )
                plan_for_invdyn = torch.cat([plan_for_invdyn, padding], dim=-1)
            else:
                # Truncate if too large
                plan_for_invdyn = plan_for_invdyn[:, :, :inv_dyn_state_dim]

        # Check if we have a valid plan
        num_planned_states = plan_for_invdyn.shape[1]
        if num_planned_states == 0:
            print("No planned states available")
            return torch.zeros((norm_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)

        # Generate actions from state pairs
        actions_list = []
        current_state_t = current_state

        for i in range(num_planned_states):
            next_state = plan_for_invdyn[:, i, :]
            state_pair = torch.cat([current_state_t, next_state], dim=-1)
            action_i = self.inverse_dynamics_model(state_pair)
            actions_list.append(action_i)
            current_state_t = next_state

        # Stack actions into sequence
        if actions_list:
            return torch.stack(actions_list, dim=1)
        else:
            return torch.zeros((norm_state.shape[0], 0, self.inverse_dynamics_model.a_dim), device=self.device)
