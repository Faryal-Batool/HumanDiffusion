from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from annotation_outputs import save_outputs_for_maps
from annotation_pathing import build_clearance_cost, inflate_obstacles
from annotation_settings import (
    AnnotationSettings,
    DEFAULT_ALSO_NORM11,
    DEFAULT_CLEAR_ALPHA,
    DEFAULT_CLEAR_MARGIN,
    DEFAULT_CLEAR_SIGMA,
    DEFAULT_EIGHT_CONNECTED,
    DEFAULT_END_FOLDER,
    DEFAULT_LINE_WIDTH,
    DEFAULT_MIN_CLEAR,
    DEFAULT_ROOT,
    DEFAULT_SPLINE_SAMPLES_PER_PX,
    DEFAULT_START_FOLDER,
    DEFAULT_THETA_STAR,
    DEFAULT_THRESHOLD,
    DEFAULT_TURN_PENALTY,
    DEFAULT_VIZ,
)
from annotation_ui import SessionController
from common import (
    first_rgb_image,
    list_maps_dirs,
    list_numeric_parent_folders,
    load_rgba_512_from_path,
    load_traversability_512,
    synthesize_rgba_from_scalar01,
    try_load_occupancy,
)


class AnnotationApp:
    def __init__(self, settings: AnnotationSettings):
        self.settings = settings

    def run(self):
        root = Path(self.settings.root)
        if not root.exists():
            raise FileNotFoundError("Root not found: {}".format(root))

        parents = list_numeric_parent_folders(root)
        if not parents:
            raise RuntimeError("No numeric folders (e.g., 1, 2, ...) found under root.")

        total_parents = len(parents)
        start_folder = max(1, self.settings.start_folder)
        end_folder = max(1, min(self.settings.end_folder, total_parents))
        if start_folder > end_folder:
            raise ValueError("start_folder ({}) cannot be after end_folder ({}).".format(start_folder, end_folder))

        actions_log = root / "actions_log.csv"
        if not actions_log.exists():
            with open(actions_log, "w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["folder", "img_subfolder", "action", "timestamp"])

        for parent in parents[start_folder - 1 : end_folder]:
            maps_dirs = list_maps_dirs(parent)
            if not maps_dirs:
                print("[warn] No 'maps' directories inside {}. Expected either '{}\\maps' or '{}\\img_XXXX\\maps'. Skipping.".format(parent, parent, parent))
                continue

            for maps_dir in maps_dirs:
                rgb_path = first_rgb_image(maps_dir)
                trav01 = load_traversability_512(maps_dir)
                occ = try_load_occupancy(maps_dir)

                if occ is not None:
                    grid_raw = occ.astype(np.uint8)
                else:
                    if trav01 is None:
                        print("[warn] Neither occupancy nor traversability in {}. Skipping.".format(maps_dir))
                        with open(actions_log, "a", newline="") as handle:
                            rel = str(maps_dir.relative_to(parent))
                            writer = csv.writer(handle)
                            writer.writerow([parent.name, rel, "skip(no_grid)", time.strftime("%Y-%m-%d %H:%M:%S")])
                        continue
                    grid_raw = (trav01 >= float(self.settings.threshold)).astype(np.uint8)

                grid_bin = inflate_obstacles(grid_raw, min_clear=int(self.settings.min_clear))
                move_cost, dist_map = build_clearance_cost(
                    grid_bin,
                    alpha=float(self.settings.clear_alpha),
                    sigma=float(self.settings.clear_sigma),
                )

                if rgb_path is not None:
                    image_rgba = load_rgba_512_from_path(rgb_path)
                    rgb_name = rgb_path.name
                else:
                    vis_scalar = trav01 if trav01 is not None else grid_bin.astype(np.float32)
                    image_rgba = synthesize_rgba_from_scalar01(vis_scalar)
                    rgb_name = "(synthesized).png"

                idx_label = "[{}\\{}]".format(parent.name, maps_dir.relative_to(parent))
                controller = SessionController(
                    image_rgba=image_rgba,
                    grid_binary=grid_bin,
                    move_cost=move_cost,
                    dist_map=dist_map,
                    trav01=trav01,
                    img_name=rgb_name,
                    maps_dir=maps_dir,
                    eight_connected=bool(self.settings.eight_connected),
                    line_width=int(self.settings.line_width),
                    viz=bool(self.settings.viz),
                    idx_label=idx_label,
                    use_theta_star=bool(self.settings.theta_star),
                    min_clear_px=int(self.settings.min_clear),
                    clear_margin_px=float(self.settings.clear_margin),
                    samples_per_px=float(self.settings.spline_samples_per_px),
                )
                controller.draw()

                subpath_for_log = str(maps_dir.relative_to(parent))
                if controller.choice == "skip":
                    with open(actions_log, "a", newline="") as handle:
                        writer = csv.writer(handle)
                        writer.writerow([parent.name, subpath_for_log, "skip", time.strftime("%Y-%m-%d %H:%M:%S")])
                    continue

                save_outputs_for_maps(
                    maps_dir=maps_dir,
                    rgb_name=rgb_name,
                    starts_xy=controller.starts_xy,
                    goal_xy=controller.goal_xy,
                    paths=controller.paths,
                    grid_bin=grid_bin,
                    dist_map=dist_map,
                    clear_margin_px=float(self.settings.clear_margin),
                    also_norm11=bool(self.settings.also_norm11),
                    samples_per_px=float(self.settings.spline_samples_per_px),
                )
                with open(actions_log, "a", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow([parent.name, subpath_for_log, "next", time.strftime("%Y-%m-%d %H:%M:%S")])

        print("Done.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Dataset root containing numeric folders 1,2,...")
    parser.add_argument("--start_folder", type=int, default=DEFAULT_START_FOLDER, help="Start numeric folder (1-based).")
    parser.add_argument("--end_folder", type=int, default=DEFAULT_END_FOLDER, help="End numeric folder (1-based, inclusive).")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="trav>=th -> free if no occupancy provided.")
    parser.add_argument("--eight_connected", action="store_true", default=DEFAULT_EIGHT_CONNECTED, help="Use 8-connectivity.")
    parser.add_argument("--line_width", type=int, default=DEFAULT_LINE_WIDTH, help="Overlay width.")
    parser.add_argument("--viz", action="store_true", default=DEFAULT_VIZ, help="Show overlays/panels.")
    parser.add_argument("--also_norm11", action="store_true", default=DEFAULT_ALSO_NORM11, help="Also save [-1,1] CSV.")
    parser.add_argument("--min_clear", type=int, default=DEFAULT_MIN_CLEAR, help="Obstacle inflation (pixels).")
    parser.add_argument("--clear_alpha", type=float, default=DEFAULT_CLEAR_ALPHA, help="Clearance penalty strength.")
    parser.add_argument("--clear_sigma", type=float, default=DEFAULT_CLEAR_SIGMA, help="Penalty decay length (px).")
    parser.add_argument("--turn_pen", type=float, default=DEFAULT_TURN_PENALTY, help="Turn penalty scale for A* (0 disables).")
    parser.add_argument("--clear_margin", type=float, default=DEFAULT_CLEAR_MARGIN, help="Required obstacle clearance (px) for smoothed path.")
    parser.add_argument("--theta_star", action="store_true", default=DEFAULT_THETA_STAR, help="Use Theta* (any-angle) instead of A*.")
    parser.add_argument(
        "--spline_samples_per_px",
        type=float,
        default=DEFAULT_SPLINE_SAMPLES_PER_PX,
        help="Sampling density along spline in samples per pixel of arc length (default 1.0).",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    settings = AnnotationSettings(
        root=args.root,
        start_folder=args.start_folder,
        end_folder=args.end_folder,
        threshold=args.threshold,
        eight_connected=args.eight_connected,
        line_width=args.line_width,
        viz=args.viz,
        also_norm11=args.also_norm11,
        min_clear=args.min_clear,
        clear_alpha=args.clear_alpha,
        clear_sigma=args.clear_sigma,
        turn_pen=args.turn_pen,
        clear_margin=args.clear_margin,
        theta_star=args.theta_star,
        spline_samples_per_px=args.spline_samples_per_px,
    )
    AnnotationApp(settings).run()
