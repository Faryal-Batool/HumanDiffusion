from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def numeric_only_name(path: Path) -> Optional[int]:
    try:
        return int(path.name)
    except ValueError:
        return None


def numeric_tail(value: str) -> int:
    nums = re.findall(r"\d+", value)
    return int(nums[-1]) if nums else 10**12


def list_numeric_parent_folders(root: Path) -> List[Path]:
    candidates = [path for path in root.iterdir() if path.is_dir() and numeric_only_name(path) is not None]
    return sorted(candidates, key=lambda path: int(path.name))


def list_img_subfolders(parent: Path) -> List[Path]:
    candidates = [path for path in parent.iterdir() if path.is_dir() and path.name.lower().startswith("img_")]
    return sorted(candidates, key=lambda path: numeric_tail(path.name))


def find_maps_dir(img_folder: Path) -> Path:
    maps = img_folder / "maps"
    if not maps.exists():
        raise FileNotFoundError("'maps' not found in {}".format(img_folder))
    return maps


def list_maps_dirs(parent: Path) -> List[Path]:
    out: List[Path] = []

    direct = parent / "maps"
    if direct.is_dir():
        out.append(direct)

    for img_folder in list_img_subfolders(parent):
        maps = img_folder / "maps"
        if maps.is_dir():
            out.append(maps)

    return out


def normalize01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    mn, mx = float(arr.min()), float(arr.max())
    if mx > mn:
        return (arr - mn) / (mx - mn)
    return np.zeros_like(arr, dtype=np.float32)


def ensure_binary512(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        image = Image.fromarray(arr.astype(np.uint8)).convert("L")
        image = image.resize((512, 512), resample=Image.NEAREST)
        return (np.array(image) > 127).astype(np.uint8)

    if arr.shape != (512, 512):
        image = Image.fromarray(arr.astype(np.float32), mode="F").resize((512, 512), resample=Image.NEAREST)
        arr = np.array(image)

    if arr.dtype == np.bool_:
        return arr.astype(np.uint8)
    if np.issubdtype(arr.dtype, np.integer):
        if arr.max() <= 1:
            return arr.astype(np.uint8)
        return (arr > 0).astype(np.uint8)

    arr = arr.astype(np.float32)
    if arr.min() >= 0 and arr.max() <= 1:
        return (arr >= 0.5).astype(np.uint8)
    return (normalize01(arr) >= 0.5).astype(np.uint8)


def first_rgb_image(maps_dir: Path) -> Optional[Path]:
    images = sorted([path for path in maps_dir.iterdir() if path.suffix.lower() in IMG_EXTS])
    for path in images:
        if path.name.lower() != "occupancy_grid.png":
            return path
    return images[0] if images else None


def load_rgba_512_from_path(img_path: Path) -> np.ndarray:
    image = Image.open(img_path).convert("RGBA")
    if image.size != (512, 512):
        image = image.resize((512, 512), resample=Image.BILINEAR)
    return np.array(image)


def synthesize_rgba_from_scalar01(s01: np.ndarray) -> np.ndarray:
    arr = (np.clip(s01, 0.0, 1.0) * 255).astype(np.uint8)
    rgb = np.stack([arr, arr, arr], axis=-1)
    alpha = np.full_like(arr, 255)
    return np.concatenate([rgb, alpha[..., None]], axis=-1)


def load_traversability_512(maps_dir: Path) -> Optional[np.ndarray]:
    trav_path = maps_dir / "traversability_map.npy"
    if not trav_path.exists():
        return None

    arr = np.load(trav_path)
    if arr.ndim != 2:
        raise ValueError("traversability_map must be 2D, got shape {}".format(arr.shape))

    arr = arr.astype(np.float32)
    if not (arr.min() >= 0.0 and arr.max() <= 1.0):
        arr = normalize01(arr)

    image = Image.fromarray(arr, mode="F").resize((512, 512), resample=Image.BILINEAR)
    out = np.array(image, dtype=np.float32)
    return np.clip(out, 0.0, 1.0)


def try_load_occupancy(maps_dir: Path) -> Optional[np.ndarray]:
    npy_path = maps_dir / "occupancy_grid.npy"
    if npy_path.exists():
        return ensure_binary512(np.load(npy_path))

    png_path = maps_dir / "occupancy_grid.png"
    if png_path.exists():
        image = Image.open(png_path).convert("L").resize((512, 512), resample=Image.NEAREST)
        return (np.array(image) > 127).astype(np.uint8)

    return None


def overlay_path_on_image(image_rgba: np.ndarray, path_rc: List[Tuple[int, int]], line_width: int = 3) -> np.ndarray:
    if image_rgba.shape[2] == 3:
        image = Image.fromarray(image_rgba, mode="RGB").convert("RGBA")
    else:
        image = Image.fromarray(image_rgba, mode="RGBA")

    draw = ImageDraw.Draw(image)
    if len(path_rc) >= 2:
        xy = [(c, r) for (r, c) in path_rc]
        draw.line(xy, fill=(255, 0, 0, 255), width=line_width)
        r0, c0 = path_rc[0]
        r1, c1 = path_rc[-1]
        draw.ellipse((c0 - 3, r0 - 3, c0 + 3, r0 + 3), fill=(0, 255, 0, 255))
        draw.ellipse((c1 - 3, r1 - 3, c1 + 3, r1 + 3), fill=(0, 0, 255, 255))
    return np.array(image)


def clamp_xy_to_bounds(x: float, y: float, width: int, height: int) -> Tuple[int, int]:
    xi = int(round(x))
    yi = int(round(y))
    return max(0, min(width - 1, xi)), max(0, min(height - 1, yi))


def safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
