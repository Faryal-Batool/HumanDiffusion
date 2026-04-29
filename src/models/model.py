# Module: High-level model wrapper around DDPM planning backends.

import torch
from torch import nn

from src.models.diffusion import Diffusion
from src.utils.configs import DataDict


# Class: Navigation model wrapper that connects inputs to the configured planner.
class HNav(nn.Module):
    """
    High-level navigation model wrapper for DDPM mask planning.

    Expected training inputs:
      - rgb:      (B,3,H,W)
      - mask_gt:  (B,3,H,W), start / goal / trajectory masks
      - start_px: (B,2) or (2,), pixel (x,y)
      - end_px:   (B,2) or (2,), pixel (x,y)
    """

    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, config, device):
        super(HNav, self).__init__()
        self.config = config
        self.device = device
        self.generator = Diffusion(self.config.diffusion)

    @staticmethod
    # Function: Ensure an optional coordinate tensor has a batch dimension.
    def _ensure_batched_coords(tensor_or_none):
        if tensor_or_none is None:
            return None
        if not torch.is_tensor(tensor_or_none):
            tensor_or_none = torch.as_tensor(tensor_or_none)
        if tensor_or_none.dim() == 1 and tensor_or_none.numel() == 2:
            return tensor_or_none.unsqueeze(0)
        return tensor_or_none

    @staticmethod
    # Function: Read RGB input from either the canonical key or camera alias.
    def _get_rgb(input_dict):
        return input_dict.get("rgb", input_dict.get(DataDict.camera, None))

    # Function: Run the module forward pass for training or encoding.
    def forward(self, input_dict, sample=False):
        if sample:
            return self.sample(input_dict=input_dict)

        rgb = self._get_rgb(input_dict)
        mask_gt = input_dict.get("mask_gt", None)
        start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
        end_px = self._ensure_batched_coords(input_dict.get("end_px", None))

        assert rgb is not None, "HNav.forward: missing 'rgb' (B,3,H,W) for DDPM conditioning"
        assert mask_gt is not None, "HNav.forward: missing 'mask_gt' (B,3,H,W) for DDPM supervision"
        assert mask_gt.dim() == 4 and mask_gt.size(1) == 3, \
            f"HNav.forward: 'mask_gt' must be (B,3,H,W), got {tuple(mask_gt.shape)}"

        output = self.generator(
            rgb=rgb,
            mask_gt=mask_gt,
            start_px=start_px,
            end_px=end_px,
        )

        output["rgb"] = rgb
        output["mask_gt"] = mask_gt
        if "start_px" in input_dict:
            output["start_px"] = input_dict["start_px"]
        if "end_px" in input_dict:
            output["end_px"] = input_dict["end_px"]
        return output

    # Function: Run DDPM reverse sampling to produce a trajectory mask.
    def sample(self, input_dict):
        rgb = self._get_rgb(input_dict)
        start_px = self._ensure_batched_coords(input_dict.get("start_px", None))
        end_px = self._ensure_batched_coords(input_dict.get("end_px", None))

        gen_out = self.generator.sample(
            rgb=rgb,
            start_px=start_px,
            end_px=end_px,
        )

        if rgb is not None:
            gen_out["rgb"] = rgb
        if "start_px" in input_dict:
            gen_out["start_px"] = input_dict["start_px"]
        if "end_px" in input_dict:
            gen_out["end_px"] = input_dict["end_px"]
        if "mask_gt" in input_dict:
            gen_out["mask_gt"] = input_dict["mask_gt"]
        return gen_out


# Function: Factory function for constructing the navigation model wrapper.
def get_model(config, device):
    return HNav(config=config, device=device)
