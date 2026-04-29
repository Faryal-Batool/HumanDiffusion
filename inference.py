# Module: Standalone inference and evaluation utilities for the DDPM planner.

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from indoor_utils import ensure_dir
from src.data_loader.dataset_2 import TrajMaskDataset
from src.models.model import get_model
from src.utils.arguments import get_configuration
from src.utils.configs import DataDict


TEST_DATA_ROOT = Path("/home/isr-lab3/Faryal_Batool/Testing_samples_simulation")
CKPT_PATH = Path("/home/isr-lab3/Faryal_Batool/DTG-main_v6/results/models/hnav_29.pth")
OUTPUT_ROOT = Path("test_results_masks_64")
VIS_SIZE = 512


# Function: Compute mean-squared error between two mask tensors.
def compute_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a - b) ** 2).mean().item()


# Function: Compute thresholded intersection-over-union between two mask tensors.
def compute_iou(a: torch.Tensor, b: torch.Tensor, threshold: float = 0.5) -> float:
    if a.dim() == 3:
        a = a[0]
    if b.dim() == 3:
        b = b[0]

    a_bin = (a > threshold).float()
    b_bin = (b > threshold).float()
    intersection = (a_bin * b_bin).sum()
    union = (a_bin + b_bin).clamp(max=1.0).sum()
    if union.item() == 0.0:
        return 1.0 if intersection.item() == 0.0 else 0.0
    return (intersection / union).item()


# Function: Build the model, load checkpoint weights, and switch it to evaluation mode.
def load_model(cfgs, device):
    model = get_model(cfgs.model, device=device).to(device)
    print(f"[INFO] Loading checkpoint from: {CKPT_PATH}")
    state = torch.load(str(CKPT_PATH), map_location=device)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state, strict=False)
    model.eval()
    return model


# Function: Run DDPM sampling for one dataset sample and return predicted and ground-truth masks.
def predict_mask(model, sample, device):
    rgb = sample["rgb"].unsqueeze(0).to(device)
    mask_gt = sample["mask_gt"].unsqueeze(0).to(device)
    start_px = sample["start_px"].unsqueeze(0).to(device)
    end_px = sample["end_px"].unsqueeze(0).to(device)

    input_dict = {
        DataDict.camera: rgb,
        "rgb": rgb,
        "start_px": start_px,
        "end_px": end_px,
    }

    with torch.no_grad():
        out = model(input_dict, sample=True)

    mask_pred = out[DataDict.prediction][:1]
    if mask_pred.shape[-2:] != mask_gt.shape[-2:]:
        mask_pred = F.interpolate(mask_pred, size=mask_gt.shape[-2:], mode="bilinear", align_corners=False)
    return rgb, mask_gt, torch.clamp(mask_pred, 0.0, 1.0)


# Function: Compute per-channel start, goal, and trajectory mask metrics.
def channel_metrics(mask_pred, mask_gt):
    names = ("start", "goal", "traj")
    metrics = {}
    for idx, name in enumerate(names):
        pred = mask_pred[:, idx:idx + 1]
        gt = mask_gt[:, idx:idx + 1]
        metrics[f"mse_{name}"] = compute_mse(pred, gt)
        metrics[f"iou_{name}"] = compute_iou(pred, gt)
    return metrics


# Function: Prepare an RGB tensor as a fixed-size numpy image for overlays.
def rgb_for_visualization(rgb):
    rgb_np = rgb[0].permute(1, 2, 0).detach().cpu().numpy()
    if rgb_np.shape[:2] == (VIS_SIZE, VIS_SIZE):
        return rgb_np

    rgb_img = (np.clip(rgb_np, 0.0, 1.0) * 255).astype(np.uint8)
    rgb_img = Image.fromarray(rgb_img).resize((VIS_SIZE, VIS_SIZE), Image.BILINEAR)
    return np.asarray(rgb_img, dtype=np.float32) / 255.0


# Function: Save an RGB overlay comparing predicted and ground-truth trajectory masks.
def save_overlay(rgb, mask_pred, mask_gt, metrics, sample_id, overlay_dir):
    rgb_vis = rgb_for_visualization(rgb)
    mask_pred_vis = F.interpolate(mask_pred, size=(VIS_SIZE, VIS_SIZE), mode="bilinear", align_corners=False)
    mask_gt_vis = F.interpolate(mask_gt, size=(VIS_SIZE, VIS_SIZE), mode="nearest")

    traj_pred_vis = mask_pred_vis[0, 2].detach().cpu().numpy()
    traj_gt_vis = mask_gt_vis[0, 2].detach().cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(rgb_vis)
    plt.imshow(traj_gt_vis, cmap="Greens", alpha=0.5, vmin=0.0, vmax=1.0)
    plt.imshow(traj_pred_vis, cmap="Reds", alpha=0.5, vmin=0.0, vmax=1.0)
    plt.axis("off")
    plt.title(
        f"{sample_id} | Traj mask ({VIS_SIZE}x{VIS_SIZE})\n"
        f"MSE={metrics['mse_traj']:.4f}, IoU={metrics['iou_traj']:.4f}"
    )
    plt.tight_layout()
    plt.savefig(overlay_dir / f"{sample_id}_rgb.png", dpi=150)
    plt.close()


# Function: Write per-sample metric curves for quick inspection.
def save_summary_plots(output_root, metric_rows):
    idxs = np.arange(len(metric_rows))
    traj_mse = np.array([row["mse_traj"] for row in metric_rows], dtype=float)
    traj_iou = np.array([row["iou_traj"] for row in metric_rows], dtype=float)

    plt.figure(figsize=(8, 4))
    plt.plot(idxs, traj_mse, marker="o", linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("MSE (traj channel)")
    plt.title("Trajectory Mask MSE per Sample")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_root / "metric_traj_mse.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(idxs, traj_iou, marker="o", linewidth=1)
    plt.xlabel("Sample index")
    plt.ylabel("IoU (traj channel)")
    plt.title("Trajectory Mask IoU per Sample")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_root / "metric_traj_iou.png", dpi=150)
    plt.close()


# Function: Print aggregate metric means across all evaluated samples.
def print_summary(metric_rows):
    print("\n=== Overall test performance (mask-based, 64x64) ===")
    for metric in ("mse_start", "mse_goal", "mse_traj", "iou_start", "iou_goal", "iou_traj"):
        values = np.array([row[metric] for row in metric_rows], dtype=float)
        print(f"Mean {metric:10s}: {values.mean():.6f}")


# Function: Script entry point that assembles configuration and launches the requested workflow.
def main():
    cfgs = get_configuration()
    cfgs.data.root = str(TEST_DATA_ROOT)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    model = load_model(cfgs, device)
    dataset = TrajMaskDataset(root_dir=cfgs.data.root, n_points=getattr(cfgs.data, "n_points", 128), use_occ=False)
    print(f"[INFO] Test samples: {len(dataset)}")

    output_root = ensure_dir(OUTPUT_ROOT)
    overlay_dir = ensure_dir(output_root / "overlays_rgb")
    metrics_csv_path = output_root / "metrics_masks.csv"

    fieldnames = ["idx", "sample_id", "mse_start", "mse_goal", "mse_traj", "iou_start", "iou_goal", "iou_traj"]
    metric_rows = []
    with open(metrics_csv_path, mode="w", newline="") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=fieldnames)
        writer.writeheader()

        for idx in range(len(dataset)):
            sample = dataset[idx]
            sample_id = dataset.samples[idx].name
            rgb, mask_gt, mask_pred = predict_mask(model, sample, device)
            metrics = channel_metrics(mask_pred, mask_gt)
            row = {"idx": idx, "sample_id": sample_id, **metrics}
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
            metric_rows.append(row)
            save_overlay(rgb, mask_pred, mask_gt, metrics, sample_id, overlay_dir)

    print_summary(metric_rows)
    save_summary_plots(output_root, metric_rows)
    print(f"\n[DONE] Saved mask metrics and overlays to: {output_root}")


if __name__ == "__main__":
    main()
