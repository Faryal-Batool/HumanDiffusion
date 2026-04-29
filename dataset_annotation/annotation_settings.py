from __future__ import annotations

from dataclasses import dataclass


DEFAULT_ROOT = r"D:\Skoltech\Thesis\Diffusion_with_imepdance\Occupancy_for_icuas"
DEFAULT_START_FOLDER = 1
DEFAULT_END_FOLDER = 8
DEFAULT_THRESHOLD = 0.3
DEFAULT_EIGHT_CONNECTED = True
DEFAULT_LINE_WIDTH = 3
DEFAULT_VIZ = True
DEFAULT_ALSO_NORM11 = False
DEFAULT_MIN_CLEAR = 10
DEFAULT_CLEAR_ALPHA = 20.0
DEFAULT_CLEAR_SIGMA = 14.0
DEFAULT_TURN_PENALTY = 0.8
DEFAULT_CLEAR_MARGIN = 6.0
DEFAULT_SPLINE_SAMPLES_PER_PX = 1.0
DEFAULT_THETA_STAR = False


@dataclass
class AnnotationSettings:
    root: str = DEFAULT_ROOT
    start_folder: int = DEFAULT_START_FOLDER
    end_folder: int = DEFAULT_END_FOLDER
    threshold: float = DEFAULT_THRESHOLD
    eight_connected: bool = DEFAULT_EIGHT_CONNECTED
    line_width: int = DEFAULT_LINE_WIDTH
    viz: bool = DEFAULT_VIZ
    also_norm11: bool = DEFAULT_ALSO_NORM11
    min_clear: int = DEFAULT_MIN_CLEAR
    clear_alpha: float = DEFAULT_CLEAR_ALPHA
    clear_sigma: float = DEFAULT_CLEAR_SIGMA
    turn_pen: float = DEFAULT_TURN_PENALTY
    clear_margin: float = DEFAULT_CLEAR_MARGIN
    theta_star: bool = DEFAULT_THETA_STAR
    spline_samples_per_px: float = DEFAULT_SPLINE_SAMPLES_PER_PX
