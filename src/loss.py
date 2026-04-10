"""
Loss module for diffusion-based trajectory generation.

This module implements mask-based loss computation for diffusion models that
predict 3-channel trajectory masks:
- channel 0: start location
- channel 1: goal location
- channel 2: trajectory path

The loss combines per-channel reconstruction errors and an optional
traversability penalty when occupancy maps are available.
"""

from torch import nn
import torch

from src.utils.configs import DataDict, LossNames


class Loss(nn.Module):
    """
    Compute diffusion mask losses for trajectory prediction.

    The Loss class is designed for diffusion models that operate on pixel masks
    instead of direct trajectory waypoints.
    """
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self.use_traversability = cfg.use_traversability

        # Loss for per-channel mask reconstruction: start / goal / trajectory
        self.mask_mse = nn.MSELoss(reduction="mean")

        self.mask_start_weight = getattr(cfg, "mask_start_weight", 1.0)
        self.mask_goal_weight = getattr(cfg, "mask_goal_weight", 2.0)
        self.mask_traj_weight = getattr(cfg, "mask_traj_weight", 2.0)

        # Weighted combination of per-channel losses
        self.distance_ratio = cfg.distance_ratio
        self.last_ratio = cfg.last_ratio
        self.traversability_ratio = cfg.traversability_ratio

    # ----------------------------------------------------------------------
    # DIFFUSION LOSS (MASK-BASED, NOT TRAJECTORY-BASED ANYMORE)
    # ----------------------------------------------------------------------
    def forward_diffusion(self, input_dict):
        """
        Compute diffusion loss from ground-truth and predicted masks.

        Expected input_dict keys:
            mask_gt: ground-truth mask tensor, shape (B,3,H,W)
            DataDict.prediction: predicted mask tensor, shape (B,3,H,W)
            occ_map: optional occupancy map tensor, shape (B,1,H,W)
        """
        mask_gt = input_dict["mask_gt"]
        mask_pred_full = input_dict[DataDict.prediction]

        B_gt = mask_gt.shape[0]
        B_pred = mask_pred_full.shape[0]
        device = mask_gt.device

        # If the diffusion model outputs a duplicated batch for traversability,
        # duplicate the ground truth so losses can be computed for both halves.
        if self.use_traversability and (B_pred == 2 * B_gt):
            mask_gt_expanded = torch.cat([mask_gt, mask_gt], dim=0)
            mask_pred = mask_pred_full
        else:
            mask_gt_expanded = mask_gt
            mask_pred = mask_pred_full

        assert mask_pred.shape == mask_gt_expanded.shape, \
            f"mask_pred {mask_pred.shape} and mask_gt {mask_gt_expanded.shape} must match."

        # Split mask channels into start, goal, and trajectory components.
        pred_start = mask_pred[:, 0:1, ...]
        pred_goal = mask_pred[:, 1:2, ...]
        pred_traj = mask_pred[:, 2:3, ...]

        gt_start = mask_gt_expanded[:, 0:1, ...]
        gt_goal = mask_gt_expanded[:, 1:2, ...]
        gt_traj = mask_gt_expanded[:, 2:3, ...]

        # Compute reconstruction losses for each channel.
        start_mse = self.mask_mse(pred_start, gt_start)
        goal_mse = self.mask_mse(pred_goal, gt_goal)
        traj_mse = self.mask_mse(pred_traj, gt_traj)

        # Apply optional per-channel weights.
        start_mse = self.mask_start_weight * start_mse
        goal_mse = self.mask_goal_weight * goal_mse
        traj_mse = self.mask_traj_weight * traj_mse

        endpoint_mse = 0.5 * (start_mse + goal_mse)

        # path_dis is defined by the trajectory channel loss.
        path_dis = traj_mse
        last_pose_dis = endpoint_mse

        output = {}
        output[LossNames.path_dis] = path_dis
        output[LossNames.last_dis] = last_pose_dis

        # Combine channel losses into a final scalar loss.
        all_loss = (
            self.distance_ratio * path_dis
            + self.last_ratio * last_pose_dis
        )

        # Optional traversability penalty: discourage predicted path activation on obstacles.
        occ_map = input_dict.get("occ_map", None)
        if self.use_traversability and occ_map is not None:
            occ = occ_map.to(device)
            if occ.dim() == 3:
                occ = occ.unsqueeze(1)
            elif occ.dim() == 4 and occ.size(1) != 1:
                occ = occ[:, :1, ...]

            if occ.shape[0] != mask_pred.shape[0]:
                factor = mask_pred.shape[0] // occ.shape[0]
                occ = occ.repeat(factor, 1, 1, 1)

            traj_pred_on_obstacles = pred_traj * occ
            trav_loss = traj_pred_on_obstacles.mean()

            all_loss = all_loss + self.traversability_ratio * trav_loss

            output[LossNames.traversability] = trav_loss

        output[LossNames.loss] = all_loss
        return output

    # ----------------------------------------------------------------------
    # FORWARD: diffusion-only loss dispatch
    # ----------------------------------------------------------------------
    def forward(self, input_dict):
        return self.forward_diffusion(input_dict=input_dict)

    @torch.no_grad()
    def evaluate(self, input_dict, indices=0):
        """
        Evaluate predicted masks without gradient computation.

        This mirrors the training reconstruction losses for logging and validation.
        """
        mask_gt = input_dict["mask_gt"]
        mask_pred = input_dict[DataDict.prediction]

        B_gt = mask_gt.shape[0]
        B_pred = mask_pred.shape[0]
        if B_pred != B_gt:
            mask_pred = mask_pred[:B_gt]

        pred_start = mask_pred[:, 0:1, ...]
        pred_goal = mask_pred[:, 1:2, ...]
        pred_traj = mask_pred[:, 2:3, ...]

        gt_start = mask_gt[:, 0:1, ...]
        gt_goal = mask_gt[:, 1:2, ...]
        gt_traj = mask_gt[:, 2:3, ...]

        start_mse = self.mask_mse(pred_start, gt_start)
        goal_mse = self.mask_mse(pred_goal, gt_goal)
        traj_mse = self.mask_mse(pred_traj, gt_traj)

        endpoint_mse = 0.5 * (start_mse + goal_mse)

        path_dis = traj_mse
        last_pose_dis = endpoint_mse

        output = {
            LossNames.evaluate_path_dis: path_dis,
            LossNames.evaluate_last_dis: last_pose_dis,
        }

        eval_loss = (
            self.distance_ratio * path_dis
            + self.last_ratio * last_pose_dis
        )
        output[LossNames.loss] = eval_loss

        return output
