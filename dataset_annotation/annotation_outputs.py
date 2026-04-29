from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from annotation_pathing import resample_polyline_arc_length


def save_outputs_for_maps(
    maps_dir: Path,
    rgb_name: str,
    starts_xy: List[Tuple[int, int]],
    goal_xy: Optional[Tuple[int, int]],
    paths: List[List[Tuple[int, int]]],
    grid_bin: np.ndarray,
    dist_map: Optional[np.ndarray],
    clear_margin_px: float,
    also_norm11: bool,
    samples_per_px: float,
):
    div = 512.0

    for old_csv in maps_dir.glob("*.csv"):
        try:
            old_csv.unlink()
        except Exception as exc:
            print("[warn] Could not delete old csv {}: {}".format(old_csv, exc))

    selections_csv = maps_dir / "selections.csv"
    with open(selections_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["rgb_name", "type", "index", "x_px", "y_px", "x_norm01", "y_norm01"])
        for index, (sx, sy) in enumerate(starts_xy, start=1):
            writer.writerow([rgb_name, "start", index, sx, sy, sx / div, sy / div])
        if goal_xy is not None:
            gx, gy = goal_xy
            writer.writerow([rgb_name, "goal", 1, gx, gy, gx / div, gy / div])

    for index, path in enumerate(paths, start=1):
        sm = [(float(r), float(c)) for (r, c) in path]

        csv_orig_pix = maps_dir / "astar_original_waypoint_count_start{}.csv".format(index)
        with open(csv_orig_pix, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_px", "col_px"])
            for r, c in sm:
                writer.writerow([r, c])

        csv_orig_n01 = maps_dir / "astar_normalized_original_waypoint_count_start{}.csv".format(index)
        with open(csv_orig_n01, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_norm01", "col_norm01"])
            for r, c in sm:
                writer.writerow([r / div, c / div])

        n_points = 128
        sm_fixed = resample_polyline_arc_length(sm, n_points)

        csv_fixed_pix = maps_dir / "astar_fixed_waypoint_count_start{}.csv".format(index)
        with open(csv_fixed_pix, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_px", "col_px"])
            for r, c in sm_fixed:
                writer.writerow([r, c])

        csv_fixed_n01 = maps_dir / "astar_normalized_fixed_waypoint_count_start{}.csv".format(index)
        with open(csv_fixed_n01, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["row_norm01", "col_norm01"])
            for r, c in sm_fixed:
                writer.writerow([r / div, c / div])

        if also_norm11:
            csv_norm11_sm = maps_dir / "astar_path_norm11_start{}.csv".format(index)
            with open(csv_norm11_sm, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_norm11", "col_norm11"])
                for r, c in sm:
                    r01, c01 = float(r) / div, float(c) / div
                    writer.writerow([r01 * 2.0 - 1.0, c01 * 2.0 - 1.0])
