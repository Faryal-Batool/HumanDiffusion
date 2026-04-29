from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from common import (
    first_rgb_image,
    list_maps_dirs,
    list_numeric_parent_folders,
    load_traversability_512,
    safe_mkdir,
    try_load_occupancy,
)
from csv_utils import find_cols, read_csv_rows

DEFAULT_ROOT = r"D:\Skoltech\Thesis\replica_dataset\Traversability_data_for_indoor_scenarios"
DEFAULT_START_FOLDER = 1
DEFAULT_END_FOLDER = 22
DEFAULT_ACTIONS_LOG = r"D:\Skoltech\Thesis\replica_dataset\Traversability_data_for_indoor_scenarios\actions_log.csv"
DEFAULT_START_SAMPLE_INDEX = 11799
OUT_DIR_NAME = "Training samples"


def copy_rgb_png_512_strict(dst_png: Path, maps_dir: Path):
    rgb_src = first_rgb_image(maps_dir)
    if rgb_src is None:
        raise FileNotFoundError("No RGB image found in {}".format(maps_dir))
    image = Image.open(rgb_src).convert("RGB")
    if image.size != (512, 512):
        image = image.resize((512, 512), resample=Image.BILINEAR)
    image.save(dst_png)


def save_trav_npy_if_available(dst_npy: Path, maps_dir: Path):
    trav = load_traversability_512(maps_dir)
    if trav is not None:
        np.save(dst_npy, trav.astype(np.float32))


def save_occ_png_if_available(dst_png: Path, maps_dir: Path):
    occ = try_load_occupancy(maps_dir)
    if occ is not None:
        Image.fromarray((occ * 255).astype(np.uint8)).save(dst_png)


def load_traj_xy_norm01(csv_path: Path) -> np.ndarray:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        cx = find_cols(headers, ["x_norm01", "x_norm", "x", "xn"])
        cy = find_cols(headers, ["y_norm01", "y_norm", "y", "yn"])
        if cx is None or cy is None:
            rows = list(reader)
            if not headers or len(headers) < 2:
                raise ValueError("Cannot infer columns in {}".format(csv_path))
            cx, cy = headers[0], headers[1]
            vals = [(float(row[cx]), float(row[cy])) for row in rows]
        else:
            vals = [(float(row[cx]), float(row[cy])) for row in reader]
    arr = np.array(vals, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("Trajectory must be (N,2); got {} from {}".format(arr.shape, csv_path))
    return np.clip(arr, 0.0, 1.0)


@dataclass
class SelectionBundle:
    starts: Dict[int, Tuple[float, float]]
    goal: Optional[Tuple[float, float]]


def parse_selections_csv(selections_csv: Path) -> SelectionBundle:
    rows = read_csv_rows(selections_csv)
    if not rows:
        return SelectionBundle(starts={}, goal=None)

    headers = list(rows[0].keys())
    c_type = find_cols(headers, ["type"])
    c_index = find_cols(headers, ["index", "idx"])
    c_x = find_cols(headers, ["x_norm01", "x_norm", "x"])
    c_y = find_cols(headers, ["y_norm01", "y_norm", "y"])
    if c_type is None or c_index is None or c_x is None or c_y is None:
        raise ValueError("selections.csv missing required columns in {}".format(selections_csv))

    starts: Dict[int, Tuple[float, float]] = {}
    goal: Optional[Tuple[float, float]] = None
    for row in rows:
        t = (row.get(c_type) or "").strip().lower()
        try:
            idx = int(float(row.get(c_index, "0")))
        except Exception:
            continue
        x = float(row.get(c_x))
        y = float(row.get(c_y))
        if t == "start":
            starts[idx] = (x, y)
        elif t == "goal":
            goal = (x, y)
    return SelectionBundle(starts=starts, goal=goal)


def read_actions_log(actions_csv: Optional[Path]) -> Dict[int, Dict[str, str]]:
    if actions_csv is None or not actions_csv.exists():
        return {}

    table: Dict[int, Dict[str, str]] = {}
    with actions_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                folder = int(row.get("folder", "").strip())
            except Exception:
                continue
            img_sub = (row.get("img_subfolder", "") or "").strip()
            action = (row.get("action", "") or "").strip().lower()
            table.setdefault(folder, {})[img_sub] = action
    return table


def should_process_img(folder_id: int, img_name: str, actions_map: Dict[int, Dict[str, str]]) -> bool:
    if folder_id not in actions_map:
        return True
    action = actions_map[folder_id].get(img_name, None)
    if action is None:
        return True
    return action.lower() != "skip"


@dataclass
class ConverterSettings:
    root: str = DEFAULT_ROOT
    start_folder: int = DEFAULT_START_FOLDER
    end_folder: int = DEFAULT_END_FOLDER
    start_sample_index: int = DEFAULT_START_SAMPLE_INDEX
    actions_log: str = DEFAULT_ACTIONS_LOG
    dry_run: bool = False


class SampleConverterApp:
    def __init__(self, settings: ConverterSettings):
        self.settings = settings

    def run(self) -> int:
        root = Path(self.settings.root)
        out_root = root / OUT_DIR_NAME
        if not self.settings.dry_run:
            out_root.mkdir(parents=True, exist_ok=True)

        actions_csv = Path(self.settings.actions_log) if self.settings.actions_log else None
        actions_map = read_actions_log(actions_csv)
        all_parents = list_numeric_parent_folders(root)
        parents = [path for path in all_parents if self.settings.start_folder <= int(path.name) <= self.settings.end_folder]

        next_idx = int(self.settings.start_sample_index)
        created_total = 0

        for parent in parents:
            folder_id = int(parent.name)
            maps_dirs = list_maps_dirs(parent)
            if not maps_dirs:
                print("[skip] No maps/ under {}".format(parent))
                continue

            for maps_dir in maps_dirs:
                if maps_dir.parent.name.lower().startswith("img_"):
                    img_name = maps_dir.parent.name
                    if not should_process_img(folder_id, img_name, actions_map):
                        print("[skip] {}/{} per actions_log".format(parent.name, img_name))
                        continue

                sel_csv = maps_dir / "selections.csv"
                if not sel_csv.exists():
                    print("[skip] No selections.csv in {}".format(maps_dir))
                    continue

                bundle = parse_selections_csv(sel_csv)
                if not bundle.starts or bundle.goal is None:
                    print("[warn] selections.csv missing starts/goal in {}".format(maps_dir))
                    continue

                for start_idx in sorted(bundle.starts.keys()):
                    traj_csv = None
                    for ext in (".csv", ".CSV"):
                        candidate = maps_dir / "astar_normalized_fixed_waypoint_count_start{}{}".format(start_idx, ext)
                        if candidate.exists():
                            traj_csv = candidate
                            break
                    if traj_csv is None:
                        print("[warn] Missing trajectory CSV for start{} in {}".format(start_idx, maps_dir))
                        continue

                    try:
                        traj = load_traj_xy_norm01(traj_csv)
                    except Exception as exc:
                        print("[warn] Bad traj in {}: {}".format(traj_csv, exc))
                        continue

                    sample_dir = out_root / "sample_{:06d}".format(next_idx)
                    if not self.settings.dry_run:
                        safe_mkdir(sample_dir)

                    try:
                        if not self.settings.dry_run:
                            copy_rgb_png_512_strict(sample_dir / "rgb.png", maps_dir)
                    except Exception as exc:
                        print("[warn] Skipping sample_{:06d} (RGB missing): {}".format(next_idx, exc))
                        if not self.settings.dry_run:
                            try:
                                sample_dir.rmdir()
                            except Exception:
                                pass
                        continue

                    if not self.settings.dry_run:
                        try:
                            save_trav_npy_if_available(sample_dir / "trav_map.npy", maps_dir)
                        except Exception as exc:
                            print("[warn] trav_map failed in {}: {}".format(sample_dir, exc))

                    if not self.settings.dry_run:
                        try:
                            save_occ_png_if_available(sample_dir / "occ_map.png", maps_dir)
                        except Exception as exc:
                            print("[warn] occ_map failed in {}: {}".format(sample_dir, exc))

                    if not self.settings.dry_run:
                        np.save(sample_dir / "traj_xy.npy", traj.astype(np.float32))

                    start_xy = bundle.starts[start_idx]
                    end_xy = bundle.goal
                    if not self.settings.dry_run:
                        (sample_dir / "start_xy.json").write_text(
                            json.dumps({"x": float(start_xy[0]), "y": float(start_xy[1])}, indent=2),
                            encoding="utf-8",
                        )
                        (sample_dir / "end_xy.json").write_text(
                            json.dumps({"x": float(end_xy[0]), "y": float(end_xy[1])}, indent=2),
                            encoding="utf-8",
                        )

                    source_name = maps_dir.parent.name if maps_dir.parent != parent else "maps"
                    print("[ok] {}  from {}/{}  (start={}, N={})".format(sample_dir.name, parent.name, source_name, start_idx, traj.shape[0]))
                    next_idx += 1
                    created_total += 1

        print("[done] Created {} samples into '{}'. Next available index: {}".format(created_total, OUT_DIR_NAME, next_idx))
        return created_total


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert CSV trajectories into global 'Training samples' folder with continuous sample numbering.")
    parser.add_argument("--root", type=str, default=DEFAULT_ROOT, help="Dataset root folder.")
    parser.add_argument("--actions-log", type=str, default=DEFAULT_ACTIONS_LOG, help="Path to actions_log.csv (folder,img_subfolder,action).")
    parser.add_argument("--start-folder", type=int, default=DEFAULT_START_FOLDER, help="First numeric parent folder to process (inclusive).")
    parser.add_argument("--end-folder", type=int, default=DEFAULT_END_FOLDER, help="Last numeric parent folder to process (inclusive).")
    parser.add_argument("--start-sample-index", type=int, default=DEFAULT_START_SAMPLE_INDEX, help="Global starting index for sample naming (e.g., 2001).")
    parser.add_argument("--dry-run", action="store_true", help="List actions without writing outputs.")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit("Root not found: {}".format(root))

    settings = ConverterSettings(
        root=args.root,
        start_folder=args.start_folder,
        end_folder=args.end_folder,
        start_sample_index=args.start_sample_index,
        actions_log=args.actions_log,
        dry_run=args.dry_run,
    )
    SampleConverterApp(settings).run()
