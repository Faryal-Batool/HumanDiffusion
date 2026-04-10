"""
High-level navigation model for trajectory generation.

This module provides the main HNav model class for diffusion-based
trajectory generation.

Diffusion uses a mask-based DDPM operating on pixel-space masks to generate
trajectories as 3-channel masks (start, goal, path pixels). It conditions on
traversability maps and supports optional start/goal inpainting.

The model acts as a wrapper for the diffusion generator, providing a
consistent interface for training and inference.
"""

import torch
from torch import nn

from src.models.diffusion import Diffusion

from src.utils.configs import DataDict, GeneratorType


class HNav(nn.Module):
    """
    High-level navigation model wrapper for trajectory generation.

    This class wraps the diffusion-based trajectory generation model.

    The diffusion generator produces 3-channel masks:
      * Channel 0: Start position pixels
      * Channel 1: Goal position pixels
      * Channel 2: Trajectory/path pixels

    Inputs include traversability maps and optional start/goal conditioning.
    The wrapper provides a single interface for training and inference.

    For the diffusion case, the core design is:

      Inputs (from dataset / dataloader):
        - trav    : (B,1,H,W)  traversability map in [0,1], 1 = free
        - mask_gt : (B,3,H,W)  ground-truth trajectory mask in pixel space
                              channel 0 -> start pixels
                              channel 1 -> goal  pixels
                              channel 2 -> trajectory/path pixels
        - start_px: (B,2) or (2,), optional conditioning, pixel (x_px, y_px)
        - end_px  : (B,2) or (2,), optional conditioning, pixel (x_px, y_px)

      Diffusion operates on:
        x0 = mask_gt   # in pixel space
        x_t ~ q(x_t | x0, t)
        eps_hat = UNet(x_t, t, trav, start_px, end_px, ...)

      Sampling / inpainting is handled inside Diffusion.sample().
    """

    def __init__(self, config, device):
        """
        Initialize the HNav model with specified generator type.

        Args:
            config: Configuration object containing model hyperparameters
                   and generator type specification
            device: Target device for model placement (CPU/GPU)

        The initialization creates a diffusion generator with internal
        conditioning.
        """
        super(HNav, self).__init__()
        self.config = config
        self.device = device

        self.generator_type = config.generator_type

        if self.generator_type != GeneratorType.diffusion:
            raise ValueError(f"Unsupported generator type: {self.generator_type}")

        # Diffusion approach: operates directly on traversability maps.
        # The diffusion model handles internal conditioning.
        self.generator = Diffusion(self.config.diffusion)

    # ----------------------------------------------------------------------
    # Helper: standardize optional pixel coordinate tensors
    # ----------------------------------------------------------------------
    @staticmethod
    def _ensure_batched_coords(tensor_or_none):
        """
        Ensure coordinate tensor is in batched shape (B,2) if present.

        This helper standardizes coordinate inputs that may come in different shapes:
        - None: No coordinates provided
        - (2,): Single coordinate pair, gets unsqueezed to (1,2)
        - (B,2): Already batched, returned as-is

        This ensures consistent tensor shapes for batch processing while being
        conservative about reshaping (only adds batch dim when clearly missing).

        Args:
            tensor_or_none: Coordinate tensor or None

        Returns:
            Standardized coordinate tensor or None
        """
        if tensor_or_none is None:
            return None
        if not torch.is_tensor(tensor_or_none):
            # Convert numpy arrays or other types to tensors
            tensor_or_none = torch.as_tensor(tensor_or_none)

        if tensor_or_none.dim() == 1 and tensor_or_none.numel() == 2:
            # Single coordinate (x,y) -> batched (1,2)
            return tensor_or_none.unsqueeze(0)
        # Assume already properly batched (B,2) or higher dimensional
        return tensor_or_none

    # ----------------------------------------------------------------------
    # Forward (TRAINING)
    # ----------------------------------------------------------------------
    def forward(self, input_dict, sample=False):
        """
        Forward pass for training or inference.

        Routes inputs to the appropriate generator based on generator_type.
        For diffusion models, handles the mask-based DDPM training pipeline.

        Args:
            input_dict: Dictionary containing input data from dataset/dataloader
            sample: If True, switches to inference mode (calls self.sample())

        Returns:
            Dictionary containing generator outputs plus pass-through fields
            for loss computation and logging
        """
        if sample:
            return self.sample(input_dict=input_dict)

        output = {}

        # Keep legacy fields for backward compatibility
        if DataDict.path in input_dict:
            output[DataDict.path] = input_dict[DataDict.path]
        if DataDict.local_map in input_dict:
            output[DataDict.local_map] = input_dict[DataDict.local_map]

        # Diffusion branch (mask-based DDPM training)
        # Extract and standardize inputs from dataset.
        rgb = input_dict.get("rgb", None)  # (B,3,H,W), optional RGB image
        mask_gt = input_dict.get("mask_gt", None)     # (B,3,H,W), float32 in {0,1}
        occ_map = input_dict.get("occ_map", None)     # (B,1,H,W), optional

        start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
        end_px   = self._ensure_batched_coords(input_dict.get("end_px", None))
        trav_step = input_dict.get(DataDict.traversable_step, None)

        assert mask_gt is not None, "HNav.forward: missing 'mask_gt' (B,3,H,W) for DDPM supervision"
        assert mask_gt.dim() == 4 and mask_gt.size(1) == 3, \
            f"HNav.forward: 'mask_gt' must be (B,3,H,W), got {tuple(mask_gt.shape)}"

        gen_out = self.generator(
            rgb=rgb,
            mask_gt=mask_gt,
            start_px=start_px,
            end_px=end_px,
            traversable_step=trav_step,
        )

        # Merge generator outputs into main output dictionary
        output.update(gen_out)

        # ------------------------------------------------------------------
        # Pass-through useful fields for loss computation and logging
        # ------------------------------------------------------------------
        # These fields are forwarded so that loss functions and logging code
        # can access the original inputs for computing losses or creating visualizations

        # RGB image (if available, useful for visualization overlays)
        if "rgb" in input_dict:
            output["rgb"] = input_dict["rgb"]  # (B,3,H,W)

        # Pixel-space coordinates (start/goal positions)
        if "start_px" in input_dict:
            output["start_px"] = input_dict["start_px"]  # (B,2) or (2,)
        if "end_px" in input_dict:
            output["end_px"] = input_dict["end_px"]    # (B,2) or (2,)

        # Ground-truth trajectory mask (for loss computation)
        if "mask_gt" in input_dict:
            output["mask_gt"] = input_dict["mask_gt"]  # (B,3,H,W)

        # Legacy normalized coordinates (if present in older pipelines)
        if "start_norm" in input_dict:
            output["start_norm"] = input_dict["start_norm"]
        if "end_norm" in input_dict:
            output["end_norm"] = input_dict["end_norm"]

        return output

    # ----------------------------------------------------------------------
    # Sampling / inference
    # ----------------------------------------------------------------------
    def sample(self, input_dict):
        """
        Inference sampling mode for trajectory generation.

        Routes to the appropriate generator's sampling method. For diffusion models,
        this performs reverse diffusion sampling with inpainting to ensure start/goal
        positions are correctly placed in the generated trajectory mask.

        Args:
            input_dict: Dictionary containing conditioning inputs for sampling

        Returns:
            Dictionary containing sampled trajectories and pass-through fields
            for visualization and evaluation
        """
        output = {}

        # Backward-compatibility pass-through fields
        if DataDict.path in input_dict:
            output[DataDict.path] = input_dict[DataDict.path]
        if DataDict.local_map in input_dict:
            output[DataDict.local_map] = input_dict[DataDict.local_map]

        # Diffusion sampling (mask-based trajectory generation)
        rgb = input_dict.get("rgb", None)  # (B,3,H,W), optional RGB image
        start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
        end_px   = self._ensure_batched_coords(input_dict.get("end_px", None))

        gen_out = self.generator.sample(
            rgb=rgb,
            start_px=start_px,
            end_px=end_px,
        )

        # Merge sampling outputs
        output.update(gen_out)

        # ------------------------------------------------------------------
        # Pass-through fields for visualization and evaluation
        # ------------------------------------------------------------------
        # Traversability map (useful for overlaying sampled trajectories)
        if "trav" in input_dict:
            output["trav"] = input_dict["trav"]
        # Start/goal coordinates (for evaluation metrics)
        if "start_px" in input_dict:
            output["start_px"] = input_dict["start_px"]
        if "end_px" in input_dict:
            output["end_px"] = input_dict["end_px"]
        # RGB image (for visualization overlays if available)
        if "rgb" in input_dict:
            output["rgb"] = input_dict["rgb"]

        return output


def get_model(config, device):
    """
    Factory function to create and initialize the HNav model.

    This is the main entry point for creating trajectory generation models.
    The function automatically configures the appropriate generator type based
    on the provided configuration.

    Args:
        config: Configuration object containing all model hyperparameters.
        device: Target device for model placement (CPU/GPU)

    Returns:
        HNav: Initialized model ready for training or inference

    Note:
        Device placement is stored but actual device transfer is typically
        handled by training code via model.to(device) for flexibility.
    """
    return HNav(config=config, device=device)
