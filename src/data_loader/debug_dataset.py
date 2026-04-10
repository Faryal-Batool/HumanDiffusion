"""
Debug script for visualizing trajectory dataset samples.

This script loads trajectory data from the dataset and creates overlay visualizations
showing RGB images with plotted trajectories, start points, and end points.
Useful for debugging data loading, verifying coordinate transformations,
and inspecting the quality of trajectory data.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import TrajDataset   # <-- import your existing dataset

# ----------------------------
# USER CONFIGURATION SETTINGS
# ----------------------------
DATASET_ROOT = r"/home/isr-lab3/Faryal_Batool/Training samples"  # Path to dataset root directory
BATCH_SIZE = 10  # Number of samples to process per batch
OUT_DIR = Path("debug_overlays")  # Output directory for visualization images
OUT_DIR.mkdir(exist_ok=True)  # Create output directory if it doesn't exist
# ----------------------------

def plot_overlay(rgb, traj, start, end, out_path):
    """
    Create and save an overlay visualization of trajectory data on RGB image.

    This function plots the RGB image as background and overlays the trajectory
    path (blue line), start point (green dot), and end point (red X) on top.

    Args:
        rgb: RGB image tensor with shape (3, H, W) in range [0,1]
        traj: Trajectory points with shape (N, 2) in pixel coordinates
        start: Start point coordinates (2,) in pixel coordinates
        end: End point coordinates (2,) in pixel coordinates
        out_path: Path where to save the output image
    """
    rgb = rgb.permute(1,2,0).cpu().numpy()   # Convert to (H,W,3) for matplotlib
    H, W = rgb.shape[:2]

    plt.figure(figsize=(6,6))
    plt.imshow(rgb)

    # Plot trajectory as blue line
    plt.plot(traj[:,0], traj[:,1], linewidth=2, c='blue')
    # Plot start point as green circle
    plt.scatter(start[0], start[1], s=60, c='green')
    # Plot end point as red X
    plt.scatter(end[0],   end[1],   s=60, c='red', marker='X')

    # Set correct image coordinate orientation (origin at top-left)
    plt.xlim(0, W)
    plt.ylim(H, 0)
    plt.axis("off")

    plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def main():
    """
    Main function to process the dataset and generate debug visualizations.

    This function:
    1. Creates a dataset and data loader
    2. Iterates through batches of data
    3. Converts normalized coordinates to pixel coordinates
    4. Generates overlay visualizations for each sample
    5. Saves images organized by batch folders
    """
    # Initialize dataset and data loader
    dataset = TrajDataset(DATASET_ROOT, n_points=128)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    batch_id = 0
    # Process each batch from the data loader
    for batch_idx, batch in enumerate(loader):
        # Create batch-specific output folder
        batch_folder = OUT_DIR / f"batch_{batch_id:03d}"
        batch_folder.mkdir(exist_ok=True)

        # Extract data from batch
        rgb = batch["rgb"]          # (B,3,H,W) RGB images
        traj = batch["traj"]        # (B,N,2) normalized trajectory points
        start = batch["start"]      # (B,2) normalized start coordinates
        end = batch["end"]          # (B,2) normalized end coordinates

        # Get image dimensions
        H = rgb.shape[-2]
        W = rgb.shape[-1]

        # Process each sample in the batch
        for i in range(rgb.shape[0]):
            sample_id = batch_idx * BATCH_SIZE + i
            out_file = batch_folder / f"sample_{sample_id:06d}.png"

            # Convert normalized coordinates to pixel coordinates
            traj_px = traj[i].cpu().numpy().copy()
            start_px = start[i].cpu().numpy().copy()
            end_px   = end[i].cpu().numpy().copy()

            # Scale normalized coordinates to pixel space
            traj_px[:,0] *= (W-1)  # X coordinates
            traj_px[:,1] *= (H-1)  # Y coordinates
            start_px *= np.array([W-1, H-1])  # Start point
            end_px   *= np.array([W-1, H-1])  # End point

            # Generate and save overlay visualization
            plot_overlay(rgb[i], traj_px, start_px, end_px, out_file)

        print(f"[Batch {batch_id}] Saved {len(rgb)} samples.")

        batch_id += 1

    print("\nAll batches processed! Check folder:", OUT_DIR.resolve())


if __name__ == "__main__":
    # Run the debug visualization script when executed directly
    main()