# Module: Loss and evaluation metrics for DDPM mask reconstruction.

from torch import nn

from src.utils.configs import DataDict, LossNames


# Class: Loss module for mask-based DDPM training and evaluation.
class Loss(nn.Module):
    # Function: Initialize module layers, configuration fields, and runtime state.
    def __init__(self, cfg):
        super(Loss, self).__init__()

        self.mask_mse = nn.MSELoss(reduction="mean")

        self.mask_start_weight = getattr(cfg, "mask_start_weight", 1.0)
        self.mask_goal_weight = getattr(cfg, "mask_goal_weight", 2.0)
        self.mask_traj_weight = getattr(cfg, "mask_traj_weight", 2.0)

        self.distance_ratio = cfg.distance_ratio
        self.last_ratio = cfg.last_ratio

    # Function: Run the module forward pass for training or encoding.
    def forward(self, input_dict):
        """
        DDPM mask loss.

        mask_gt and prediction are (B,3,H,W):
          ch0 = start mask
          ch1 = goal mask
          ch2 = trajectory mask
        """
        mask_gt = input_dict["mask_gt"]
        mask_pred = input_dict[DataDict.prediction]

        assert mask_pred.shape == mask_gt.shape, \
            f"mask_pred {mask_pred.shape} and mask_gt {mask_gt.shape} must match."

        pred_start = mask_pred[:, 0:1, ...]
        pred_goal = mask_pred[:, 1:2, ...]
        pred_traj = mask_pred[:, 2:3, ...]

        gt_start = mask_gt[:, 0:1, ...]
        gt_goal = mask_gt[:, 1:2, ...]
        gt_traj = mask_gt[:, 2:3, ...]

        start_mse = self.mask_start_weight * self.mask_mse(pred_start, gt_start)
        goal_mse = self.mask_goal_weight * self.mask_mse(pred_goal, gt_goal)
        traj_mse = self.mask_traj_weight * self.mask_mse(pred_traj, gt_traj)

        endpoint_mse = 0.5 * (start_mse + goal_mse)
        loss = self.distance_ratio * traj_mse + self.last_ratio * endpoint_mse

        return {
            LossNames.path_dis: traj_mse,
            LossNames.last_dis: endpoint_mse,
            LossNames.loss: loss,
        }

    # Function: Compute evaluation metrics using the same mask loss surface.
    def evaluate(self, input_dict, indices=0):
        return self.forward(input_dict)
