#!/usr/bin/env python

from collections import deque
import torch
from torch import Tensor, nn
import safetensors
from pathlib import Path

from lerobot.common.constants import OBS_ENV, OBS_ROBOT, OBS_IMAGE
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import (
    get_device_from_parameters,
    populate_queues,
)

from model.diffusion.configuration_mymodel import DiffusionConfig
from model.diffusion.modeling_clphycon import CLDiffPhyConModel
from model.invdynamics.invdyn import MlpInvDynamic


class CLDiffPhyConPolicy(PreTrainedPolicy):
    """
    CL-DiffPhyCon policy using a Diffusion Transformer with asynchronous denoising
    for closed-loop physical control.
    """

    config_class = DiffusionConfig
    name = "clphycon"

    def __init__(
        self,
        config: DiffusionConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration
            dataset_stats: Dataset statistics for normalization
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        # Normalizers/Unnormalizers
        self.normalize_inputs = Normalize(
            config.input_features, config.normalization_mapping, dataset_stats)

        # Normalize/unnormalize for inverse dynamics
        self.normalize_invdyn_state = Normalize(
            {"observation.state": config.robot_state_feature},
            config.normalization_mapping, dataset_stats
        )
        self.normalize_invdyn_action = Normalize(
            {"action": config.action_feature},
            config.normalization_mapping, dataset_stats
        )
        self.unnormalize_action_output = Unnormalize(
            {"action": config.action_feature},  # Only unnormalize action
            config.normalization_mapping, dataset_stats
        )

        # Queues for rollout
        self._queues = None
        self.state_dim = config.robot_state_feature.shape[0]
        self.action_dim = config.action_feature.shape[0]

        # Instantiate the CL-DiffPhyCon model
        self.diffusion = CLDiffPhyConModel(config)

        # Instantiate inverse dynamics model
        self.inv_dyn_model = MlpInvDynamic(
            o_dim=self.state_dim * 2,
            a_dim=self.action_dim,
            hidden_dim=self.config.inv_dyn_hidden_dim,
            dropout=0.1,
            use_layernorm=True,
            out_activation=nn.Tanh(),
        )

        # Move models to device
        self.diffusion.to(config.device)
        self.inv_dyn_model.to(config.device)
        self.device = get_device_from_parameters(self.diffusion)

        self.reset()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path, **kwargs):
        """
        Instantiate a policy from a pretrained checkpoint directory.
        Expects directory structure:
        - config.json
        - stats.safetensors
        - diffusion.pth (state_dict for CLDiffPhyConModel)
        - invdyn.pth (state_dict for MlpInvDynamic)
        """
        pretrained_model_name_or_path = Path(pretrained_model_name_or_path)

        # 1. Load config
        config_path = pretrained_model_name_or_path / "config.json"
        if not config_path.is_file():
            raise OSError(
                f"config.json not found in {pretrained_model_name_or_path}")
        config = cls.config_class.from_json_file(config_path)

        # 2. Load dataset stats
        stats_path = pretrained_model_name_or_path / "stats.safetensors"
        if not stats_path.is_file():
            raise OSError(
                f"stats.safetensors not found in {pretrained_model_name_or_path}")
        with safetensors.safe_open(stats_path, framework="pt", device="cpu") as f:
            dataset_stats = {k: f.get_tensor(k) for k in f.keys()}

        # 3. Instantiate the policy
        policy = cls(config, dataset_stats)
        policy.eval()  # Set to eval mode by default after loading

        # 4. Load individual component state dicts
        device = config.device  # Use device from config

        diffusion_ckpt_path = pretrained_model_name_or_path / "diffusion.pth"
        if diffusion_ckpt_path.is_file():
            print(f"Loading diffusion state dict from: {diffusion_ckpt_path}")
            diff_state_dict = torch.load(
                diffusion_ckpt_path, map_location="cpu")
            policy.diffusion.load_state_dict(diff_state_dict)
        else:
            print(
                f"Warning: diffusion.pth not found in {pretrained_model_name_or_path}. Diffusion model not loaded.")

        invdyn_ckpt_path = pretrained_model_name_or_path / "invdyn.pth"
        if invdyn_ckpt_path.is_file():
            print(f"Loading invdyn state dict from: {invdyn_ckpt_path}")
            inv_state_dict = torch.load(invdyn_ckpt_path, map_location="cpu")
            policy.inv_dyn_model.load_state_dict(inv_state_dict)
        else:
            print(
                f"Warning: invdyn.pth not found in {pretrained_model_name_or_path}. Inverse dynamics model not loaded.")

        # Move policy to the correct device AFTER loading state dicts
        policy.to(device)
        policy.device = device  # Update device attribute

        return policy

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if self.config.image_features:
            # Use a single key for stacked images in the queue
            self._queues["observation.image"] = deque(
                maxlen=self.config.n_obs_steps)
        if self.config.env_state_feature:
            self._queues["observation.environment_state"] = deque(
                maxlen=self.config.n_obs_steps)

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations using CL-DiffPhyCon."""
        # Ensure input batch tensors are on the correct device
        batch = {k: v.to(self.device) if isinstance(
            v, torch.Tensor) else v for k, v in batch.items()}

        # Normalize inputs
        norm_batch = self.normalize_inputs(batch)

        # Stack multiple camera views if necessary
        if self.config.image_features:
            # Create a temporary dict to avoid modifying the original input batch
            processed_batch = dict(norm_batch)
            processed_batch["observation.image"] = torch.stack(
                [norm_batch[key] for key in self.config.image_features], dim=-4
            )
        else:
            processed_batch = norm_batch  # Use the normalized batch

        # Populate queues with the latest *normalized* observation
        self._queues = populate_queues(self._queues, processed_batch)

        # Generate new action plan only when the action queue is empty
        if len(self._queues["action"]) == 0:
            # Prepare batch for the model by stacking history from queues (already normalized)
            model_input_batch = {}
            for key, queue in self._queues.items():
                if key.startswith("observation"):
                    # Ensure tensors are on the correct device before stacking if needed
                    queue_list = [item.to(self.device) if isinstance(
                        item, torch.Tensor) else item for item in queue]
                    model_input_batch[key] = torch.stack(queue_list, dim=1)

            # Get the very last state (already normalized)
            current_state = model_input_batch["observation.state"][:, 0, :]
            num_samples = getattr(self.config, "num_inference_samples", 1)

            # Generate actions using the CL-DiffPhyCon model
            actions = self.diffusion.cl_phycon_inference(
                model_input_batch,  # Pass normalized batch
                current_state,      # Pass normalized state
                self.inv_dyn_model,
                num_samples=num_samples,
            )  # Returns normalized actions

            # Unnormalize actions
            actions_unnormalized = self.unnormalize_action_output(
                {"action": actions})["action"]

            self._queues["action"].extend(actions_unnormalized.transpose(0, 1))

        # Pop the next action from the queue
        action = self._queues["action"].popleft()
        return action
