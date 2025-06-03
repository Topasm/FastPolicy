from collections import deque
import torch
import torch.nn as nn
from torch import Tensor
import einops
from typing import Dict
import threading
import queue
import time

from lerobot.common.policies.utils import get_device_from_parameters, populate_queues
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.constants import OBS_ROBOT, OBS_IMAGE, OBS_ENV
from lerobot.configs.types import FeatureType, NormalizationMode
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
        all_dataset_features: Dict[str, any],
        n_obs_steps: int,
        input_features: Dict[str, FeatureType] = None,
        norm_mapping: Dict[str, NormalizationMode] = None,
        dataset_stats: Dict[str, any] = None,
    ):
        super().__init__()
        # Store the transformer - now there's only one version with integrated normalization
        self.bidirectional_transformer = bidirectional_transformer
        self.base_transformer = bidirectional_transformer

        self.state_diffusion_model = state_diffusion_model
        self.inverse_dynamics_model = inverse_dynamics_model
        self.config = state_diffusion_model.config
        self.device = get_device_from_parameters(bidirectional_transformer)

        # CL-DiffPhyCon parameters
        self.use_cl_diffphycon = True
        # Store previous step's noisy latent sequence
        self.cl_diffphycon_latent_z_prev = None
        # Controls whether to add noise to transformer plans (Option A vs B)
        self.use_noisy_transformer_plans = False
        # Initial noise level for transformer plans if use_noisy_transformer_plans is True
        self.initial_noise_level = 0.8  # T/H ratio

        # Threading components for asynchronous transformer execution
        self.transformer_plan_queue = queue.Queue(maxsize=1)
        self.transformer_worker_thread = None
        self.stop_transformer_event = threading.Event()
        self.latest_obs_for_transformer = None
        self.transformer_lock = threading.Lock()  # Lock for thread-safe updates
        self.latest_obs_hash = None  # Track observation changes
        self.last_transformer_update_time = 0  # Track timing of updates

        self.normalize_inputs = Normalize(
            input_features, norm_mapping, dataset_stats)

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

    def reset(self):
        """Reset observation history queues. Should be called on env.reset()"""
        print("Resetting BidirectionalRTDiffusionPolicy queues")

        # First stop the worker thread to avoid race conditions
        self._stop_transformer_worker()

        # Reset all observation queues
        self._obs_image_queue = deque(maxlen=self.n_obs_steps)
        self._obs_state_queue = deque(maxlen=self.n_obs_steps)
        self._action_execution_queue = deque()

        # Clear the diffusion model's latent state
        self.cl_diffphycon_latent_z_prev = None

        # Safely clear the transformer plan queue
        try:
            while True:
                self.transformer_plan_queue.get_nowait()
        except queue.Empty:
            pass

        # Reset the latest observation and tracking variables for the transformer
        with self.transformer_lock:
            self.latest_obs_for_transformer = None
            self.latest_obs_hash = None
            self._last_processed_hash = None  # Also reset the worker's tracking hash
            self.last_transformer_update_time = 0

        # Start a fresh transformer worker thread
        self._start_transformer_worker()

        print("Reset complete, transformer worker restarted")

    @torch.no_grad()
    def select_action(self, current_raw_observation: Dict[str, Tensor]) -> Tensor:
        """Select an action given the current observation."""
        # Ensure the transformer worker thread is running
        if not hasattr(self, 'transformer_worker_thread') or self.transformer_worker_thread is None or not self.transformer_worker_thread.is_alive():
            self._start_transformer_worker()

        # Move tensors to device
        raw_obs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                   for k, v in current_raw_observation.items()}

        # Process and normalize observation
        processed_obs = self._preprocess_observation(raw_obs)
        normalized_obs = self.normalize_inputs(processed_obs)

        # Update observation queues
        self._queues = populate_queues(
            self._queues, normalized_obs)

        # Check if we have enough history to make a prediction
        required_obs = self.config.n_obs_steps
        if len(self._queues["observation.state"]) < required_obs:
            print(
                f"Warning: Not enough history in queues. Have {len(self._queues['observation.state'])}, need {required_obs}")
            # Return zero action if we don't have enough history yet
            return torch.zeros((1, self.config.action_feature.shape[0]), device=self.device)

        # If we already have actions queued, return the next one
        if self._action_execution_queue:
            return self._get_next_action()

        # Prepare batch for the model by stacking history from queues (already normalized)
        model_input_batch = {}
        for key, deque_obj in self._queues.items():
            if key.startswith("observation"):
                # Ensure tensors are on the correct device before stacking if needed
                queue_list = [item.to(self.device) if isinstance(
                    item, torch.Tensor) else item for item in deque_obj]
                model_input_batch[key] = torch.stack(queue_list, dim=1)

        # Update the observation data for the transformer worker thread
        # This will trigger a new plan computation in the background
        current_obs_hash = self._generate_obs_hash(model_input_batch)
        with self.transformer_lock:
            if current_obs_hash != self.latest_obs_hash:
                self.latest_obs_for_transformer = model_input_batch.copy()
                self.latest_obs_hash = current_obs_hash
                self.last_transformer_update_time = time.time()
                print("Updated transformer observation data")

        # Get the last state in the sequence
        norm_state = model_input_batch["observation.state"][:, -1, :]

        # Check if we have a new transformer plan available to use
        new_transformer_segment_plan = None
        try:
            # Try to get a new plan without blocking
            new_transformer_segment_plan = self.transformer_plan_queue.get_nowait()
            print("Integrating new transformer plan into policy")

            # Re-initialize the diffusion latent state with the new plan
            # Use the dedicated helper method to adjust the plan to match diffusion horizon
            diffusion_horizon = self.state_diffusion_model.config.horizon
            self.cl_diffphycon_latent_z_prev = self._adjust_plan_to_horizon(
                new_transformer_segment_plan, diffusion_horizon)

            # Option B: Add noise for principled start if configured
            if self.use_noisy_transformer_plans:
                self.cl_diffphycon_latent_z_prev = self._add_initial_noise_to_plan(
                    self.cl_diffphycon_latent_z_prev, self.initial_noise_level
                )

        except queue.Empty:
            # No new plan available
            if self.cl_diffphycon_latent_z_prev is None:
                print("Waiting for first transformer plan with timeout")
                try:
                    # Wait for the first plan with a short timeout
                    new_transformer_segment_plan = self.transformer_plan_queue.get(
                        timeout=0.1)
                    print("First transformer plan received")

                    # Initialize with the received plan using the helper method
                    diffusion_horizon = self.state_diffusion_model.config.horizon
                    self.cl_diffphycon_latent_z_prev = self._adjust_plan_to_horizon(
                        new_transformer_segment_plan, diffusion_horizon)

                    # Option B: Add noise for principled start if configured
                    if self.use_noisy_transformer_plans:
                        self.cl_diffphycon_latent_z_prev = self._add_initial_noise_to_plan(
                            self.cl_diffphycon_latent_z_prev, self.initial_noise_level
                        )

                except queue.Empty:
                    print(
                        "Timeout waiting for first transformer plan, generating synchronously")
                    # Fall through to the state plan generation, which will handle the None case

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
            # Create zero action directly
            batch_size = norm_state.shape[0] if 'norm_state' in locals() else 1
            action_dim = getattr(self.inverse_dynamics_model, 'a_dim',
                                 getattr(self.config.action_feature, 'shape', [2])[0])
            return torch.zeros((batch_size, action_dim), device=self.device)

    def _preprocess_observation(self, raw_obs):
        """Preprocess raw observations, particularly adjusting image dimensions."""
        processed_obs = {}
        for k, v in raw_obs.items():
            if k == "observation.image" and v is not None:
                # Handle image format conversions
                if v.ndim == 4:  # [B, H, W, C]
                    # Convert uint8 to float32 and normalize to [0,1] range first
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0
                    processed_obs[k] = v.permute(0, 3, 1, 2)  # -> [B, C, H, W]
                elif v.ndim == 3:  # [H, W, C]
                    # Convert uint8 to float32 and normalize to [0,1] range first
                    if v.dtype == torch.uint8:
                        v = v.float() / 255.0
                    processed_obs[k] = v.permute(
                        2, 0, 1).unsqueeze(0)  # -> [1, C, H, W]
                else:
                    processed_obs[k] = v
            else:
                processed_obs[k] = v

        return processed_obs

    # _normalize_observation method removed as it's no longer used

    def _get_next_action(self):
        """Get the next normalized action from the queue and unnormalize it."""
        next_action = self._action_execution_queue.popleft()

        # Handle different shapes - ensure it's a batch
        if len(next_action.shape) == 1:
            next_action = next_action.unsqueeze(0)

        # Log the normalized action values
        print(f"Normalized action values: {next_action}")

        # Return the action as-is in case of error
        return next_action

    def _generate_state_plan(self, model_input_batch):
        """Generate refined state plan using bidirectional transformer and diffusion.

        Args:
            model_input_batch: Dict containing observation history with keys like "observation.state", "observation.image"
        """
        # We now only support CL-DiffPhyCon, the standard plan generation has been removed
        return self._generate_cl_diffphycon_plan(model_input_batch)

    def _generate_cl_diffphycon_plan(self, model_input_batch):
        """Generate state plan using CL-DiffPhyCon algorithm with asynchronous transformer execution.

        This simplified implementation focuses on the core diffusion step and initializes 
        the latent state when needed.

        Args:
            model_input_batch: Dict containing observation history with keys like "observation.state", "observation.image"
                               This includes both current observation and history
        """
        print("Using CL-DiffPhyCon algorithm with asynchronous transformer")

        # Check if we need to initialize CL-DiffPhyCon latent state
        if self.cl_diffphycon_latent_z_prev is None:
            print("CL-DiffPhyCon latent_z_prev is None, initializing...")

            # We first try to get an initial plan from the transformer queue
            try:
                # Try to get a plan with a short timeout
                new_transformer_plan = self.transformer_plan_queue.get(
                    timeout=0.1)
                print("Using transformer plan from queue for initialization")

                # Initialize with the received plan, adjusted to diffusion horizon
                diffusion_horizon = self.state_diffusion_model.config.horizon
                initial_clean_plan = self._adjust_plan_to_horizon(
                    new_transformer_plan, diffusion_horizon)

                # Apply noise if configured (Option B)
                if self.use_noisy_transformer_plans:
                    self.cl_diffphycon_latent_z_prev = self._add_initial_noise_to_plan(
                        initial_clean_plan, self.initial_noise_level
                    )
                else:
                    # Option A: Use clean plan
                    self.cl_diffphycon_latent_z_prev = initial_clean_plan

            except queue.Empty:
                # No plan in queue, initialize using zeros or default values
                print("No transformer plan available for initialization, using default")
                batch_size = model_input_batch["observation.state"].shape[0]
                diffusion_horizon = self.state_diffusion_model.config.horizon
                state_dim = self.state_diffusion_model.config.output_features[
                    self.state_diffusion_model.config.diffusion_target_key].shape[0]

                # Initialize with random values as the model expects
                self.cl_diffphycon_latent_z_prev = torch.randn(
                    (batch_size, diffusion_horizon, state_dim),
                    device=self.device
                )

        # Perform CL-DiffPhyCon Asynchronous Denoising Step
        # model_input_batch is used by _prepare_global_conditioning inside sample_cl_diffphycon_step
        current_step_predicted_state_at_t0, next_step_latent_z = \
            self.state_diffusion_model.sample_cl_diffphycon_step(
                self.cl_diffphycon_latent_z_prev,
                model_input_batch
            )

        # Update the stored noisy latent for the next environment step
        self.cl_diffphycon_latent_z_prev = next_step_latent_z

        # Return the fully denoised state prediction for the current step
        return current_step_predicted_state_at_t0

    def _adjust_plan_to_horizon(self, plan, target_horizon, use_last_n_if_longer: bool = False):  # Add a flag
        if plan.shape[1] > target_horizon:
            if use_last_n_if_longer:
                # Takes the LAST target_horizon steps
                return plan[:, -target_horizon:, :]
            else:
                # Takes the FIRST target_horizon steps
                return plan[:, :target_horizon:, :]
        elif plan.shape[1] < target_horizon:
            # Pad with the last state if too short
            padding_size = target_horizon - plan.shape[1]
            # Slicing with -1: ensures it's still [B, 1, D]
            last_state = plan[:, -1:, :]
            padding = last_state.repeat(1, padding_size, 1)
            return torch.cat([plan, padding], dim=1)
        else:
            # Correct length, return as is
            return plan

    # _generate_standard_plan method removed as it's been replaced by the CL-DiffPhyCon implementation

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

    def _transformer_worker_loop(self):
        """Worker function that runs in a separate thread to generate plans asynchronously.

        This function continuously checks for new observations and generates transformer plans
        without blocking the main thread.
        """
        print("Starting transformer worker loop")

        # We'll track the last processed hash instead of the full observation

        while not self.stop_transformer_event.is_set():
            # Check if we have new observations to process
            current_hash = None
            with self.transformer_lock:
                model_input_batch = self.latest_obs_for_transformer
                current_hash = self.latest_obs_hash

            # Skip if no observations
            if model_input_batch is None:
                time.sleep(0.01)  # Brief sleep to avoid busy-waiting
                continue

            # Check if this is a new observation using the hash
            if current_hash is not None and current_hash == getattr(self, '_last_processed_hash', None):
                time.sleep(0.01)  # Brief sleep to avoid busy-waiting
                continue

            # Store the current hash for the next iteration
            self._last_processed_hash = current_hash

            # Process the new observation batch
            try:
                # Extract state and image from the observation batch
                norm_state_current = model_input_batch["observation.state"][:, -1, :]
                norm_img_current = None
                if "observation.image" in model_input_batch:
                    norm_img_current = model_input_batch["observation.image"][:, -1, :, :, :]

                # Prepare state for bidirectional transformer
                bidir_state_dim = self.base_transformer.config.state_dim
                current_state_for_bidir = norm_state_current
                if norm_state_current.shape[-1] != bidir_state_dim:
                    current_state_for_bidir = norm_state_current[:,
                                                                 :bidir_state_dim]

                # Generate new segment plan with transformer
                with torch.no_grad():
                    tf_predictions = self.bidirectional_transformer(
                        initial_images=norm_img_current,
                        initial_states=current_state_for_bidir,
                        training=False
                    )

                # Extract predicted states and create new segment plan
                norm_predicted_states = tf_predictions['predicted_forward_states']
                new_segment_plan = torch.cat(
                    [current_state_for_bidir.unsqueeze(1), norm_predicted_states], dim=1
                )

                # Adjust plan length to match diffusion horizon
                diffusion_horizon = self.state_diffusion_model.config.horizon
                # Use the dedicated helper method for consistent handling
                new_segment_plan = self._adjust_plan_to_horizon(
                    new_segment_plan, diffusion_horizon)

                # Clear the queue to ensure we only have the latest plan
                while not self.transformer_plan_queue.empty():
                    try:
                        self.transformer_plan_queue.get_nowait()
                    except queue.Empty:
                        break

                # Put the new segment plan in the queue
                try:
                    self.transformer_plan_queue.put_nowait(new_segment_plan)
                    print("Transformer worker generated new plan segment")
                except queue.Full:
                    print("Warning: Queue full, couldn't add new plan immediately")
                    # Try again after clearing
                    try:
                        self.transformer_plan_queue.get_nowait()  # Remove old plan
                        self.transformer_plan_queue.put_nowait(
                            new_segment_plan)
                        print("Successfully replaced old plan with new one")
                    except (queue.Empty, queue.Full):
                        print("Failed to insert new plan after clearing queue")

                # Reset the latest observation to avoid redundant processing
                with self.transformer_lock:
                    if self.latest_obs_for_transformer is model_input_batch:
                        self.latest_obs_for_transformer = None

            except Exception as e:
                print(f"Error in transformer worker loop: {e}")
                import traceback
                traceback.print_exc()

            # Small sleep to avoid hogging CPU even when processing
            time.sleep(0.01)

        print("Transformer worker loop stopped")

    # _transformer_worker method removed as it's now unused, replaced by _transformer_worker_loop

    def _start_transformer_worker(self):
        """Start the transformer worker thread if it's not already running."""
        if self.transformer_worker_thread is None or not self.transformer_worker_thread.is_alive():
            self.stop_transformer_event.clear()
            self.transformer_worker_thread = threading.Thread(
                target=self._transformer_worker_loop)
            self.transformer_worker_thread.daemon = True
            self.transformer_worker_thread.start()

    def _stop_transformer_worker(self):
        """Stop the transformer worker thread."""
        if self.transformer_worker_thread is not None:
            self.stop_transformer_event.set()
            if self.transformer_worker_thread.is_alive():
                self.transformer_worker_thread.join(timeout=1.0)
            self.transformer_worker_thread = None

    def cleanup(self):
        """Clean up resources when the policy is no longer needed.

        This should be called when the policy is being removed or when the program is shutting down.
        """
        print("Cleaning up BidirectionalRTDiffusionPolicy resources")
        self._stop_transformer_worker()

    def __del__(self):
        """Ensure resources are cleaned up when the object is garbage collected."""
        self.cleanup()

    def _generate_obs_hash(self, model_input_batch):
        """Generate a simple hash of the observation data to detect changes.

        This avoids redundant computation when the observation hasn't changed meaningfully.

        Args:
            model_input_batch: Dict containing observation data

        Returns:
            A hash value representing the key content of the observation
        """
        if model_input_batch is None:
            return None

        # Focus only on the last state and image (if present) for the hash
        hash_components = []
        if "observation.state" in model_input_batch:
            # Get last state and convert to a string representation
            last_state = model_input_batch["observation.state"][:, -1, :]
            hash_components.append(str(last_state.cpu().numpy().tobytes()))

        if "observation.image" in model_input_batch:
            # Get a sparse representation of the last image
            # We don't use the full image to avoid expensive hash computation
            img = model_input_batch["observation.image"][:, -1, :, :, :]
            # Sample a few pixels (e.g., corners) for the hash
            if len(img.shape) == 4:  # [B, C, H, W]
                sample = img[:, :, ::16, ::16]  # Sample every 16th pixel
                hash_components.append(str(sample.cpu().numpy().tobytes()))

        # Combine components into a single hash
        import hashlib
        combined = "".join(hash_components)
        return hashlib.md5(combined.encode()).hexdigest()

    def _add_initial_noise_to_plan(self, clean_plan, timestep_t_div_h=0.8):
        """Add initial noise to a clean transformer plan for principled start.

        This is Option B from the design document - adding a defined initial noise
        level to the clean plan to match the expected denoising process.

        Args:
            clean_plan: The clean plan from transformer [batch_size, horizon, state_dim]
            timestep_t_div_h: Initial noise level (T/H ratio in CL-DiffPhyCon)

        Returns:
            Tensor: Plan with added noise at specified level
        """
        try:
            # Get the noise scheduler from the diffusion model
            noise_scheduler = getattr(
                self.state_diffusion_model, 'noise_scheduler', None)

            if noise_scheduler is None:
                print("Warning: No noise_scheduler found, using clean plan as-is")
                return clean_plan

            # Calculate the actual timestep to use
            max_timesteps = getattr(
                noise_scheduler, 'num_train_timesteps', 1000)
            initial_timestep = int(max_timesteps * timestep_t_div_h)

            # Generate random noise of same shape as plan
            noise = torch.randn_like(clean_plan, device=self.device)

            # Add noise to the clean plan
            noisy_plan = noise_scheduler.add_noise(
                original_samples=clean_plan,
                noise=noise,
                timesteps=torch.tensor([initial_timestep], device=self.device)
            )

            print(
                f"Added noise at level {timestep_t_div_h} (timestep {initial_timestep}/{max_timesteps})")
            return noisy_plan

        except Exception as e:
            print(f"Error adding noise to plan: {e}, using clean plan")
            return clean_plan
