from __future__ import annotations

import math
from typing import List, Tuple, Optional

import cv2
import heapq
import heapq as _heapq
import numpy as np
from scipy.interpolate import CubicSpline

from annotation_settings import DEFAULT_SPLINE_SAMPLES_PER_PX


def resample_polyline_arc_length(path_rc_float, n_points: int):
    p = np.asarray(path_rc_float, dtype=np.float64)
    if len(p) == 0:
        return np.zeros((n_points, 2), dtype=np.float64)
    if len(p) == 1:
        return np.repeat(p, n_points, axis=0)
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] == 0:
        return np.repeat(p[:1], n_points, axis=0)
    t = np.linspace(0.0, s[-1], n_points)
    r = np.interp(t, s, p[:, 0])
    c = np.interp(t, s, p[:, 1])
    return np.stack([r, c], axis=1)


def _densify_two_point_path(p0: Tuple[int, int], p1: Tuple[int, int], samples_per_px: float):
    p0 = np.asarray(p0, dtype=np.float64)
    p1 = np.asarray(p1, dtype=np.float64)
    length = float(np.linalg.norm(p1 - p0))
    n_samples = max(int(math.ceil(length * max(1e-6, samples_per_px))), 20)
    tt = np.linspace(0.0, 1.0, n_samples)
    rr = p0[0] + (p1[0] - p0[0]) * tt
    cc = p0[1] + (p1[1] - p0[1]) * tt
    return [(float(r), float(c)) for r, c in zip(rr, cc)]


def _bresenham_line(r0, c0, r1, c1):
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _line_of_sight(grid_free_1: np.ndarray, a, b) -> bool:
    (r0, c0), (r1, c1) = a, b
    height, width = grid_free_1.shape
    for r, c in _bresenham_line(r0, c0, r1, c1):
        if r < 0 or r >= height or c < 0 or c >= width:
            return False
        if grid_free_1[r, c] == 0:
            return False
    return True


def prune_path_los(grid_free_1: np.ndarray, path_rc):
    if not path_rc or len(path_rc) <= 2:
        return path_rc
    pruned = [path_rc[0]]
    i = 0
    while i < len(path_rc) - 1:
        j = len(path_rc) - 1
        while j > i + 1 and not _line_of_sight(grid_free_1, path_rc[i], path_rc[j]):
            j -= 1
        pruned.append(path_rc[j])
        i = j
    return pruned


def _path_is_collision_free(grid_free_1: np.ndarray, path_float):
    if len(path_float) < 2:
        return True
    for k in range(len(path_float) - 1):
        r0, c0 = path_float[k]
        r1, c1 = path_float[k + 1]
        for rr, cc in _bresenham_line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))):
            if grid_free_1[rr, cc] == 0:
                return False
    return True


def path_has_clearance(dist_map: np.ndarray, path_float, clear_margin_px: float) -> bool:
    height, width = dist_map.shape
    if len(path_float) < 2:
        return True
    clear_margin = float(clear_margin_px)
    for k in range(len(path_float) - 1):
        r0, c0 = path_float[k]
        r1, c1 = path_float[k + 1]
        for rr, cc in _bresenham_line(int(round(r0)), int(round(c0)), int(round(r1)), int(round(c1))):
            if rr < 0 or rr >= height or cc < 0 or cc >= width:
                return False
            if dist_map[rr, cc] < clear_margin:
                return False
    return True


def _dedupe_consecutive(points_rc: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out = []
    last = None
    for point in points_rc:
        if last is None or point != last:
            out.append(point)
            last = point
    return out


def cubic_spline_path(
    points_rc: List[Tuple[int, int]],
    samples_per_px: float = DEFAULT_SPLINE_SAMPLES_PER_PX,
) -> List[Tuple[float, float]]:
    if len(points_rc) < 1:
        return []

    points = _dedupe_consecutive(points_rc)
    if len(points) == 1:
        r, c = points[0]
        return [(float(r), float(c))]
    if len(points) == 2:
        return _densify_two_point_path(points[0], points[1], samples_per_px)

    pts = np.asarray(points, dtype=np.float64)
    diffs = np.diff(pts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    t = np.zeros(len(pts), dtype=np.float64)
    t[1:] = np.cumsum(seglen)

    mask = np.diff(t) > 1e-9
    if not np.all(mask):
        keep = [0] + [i + 1 for i, value in enumerate(mask) if value]
        pts = pts[keep]
        t = t[keep]
        if len(pts) == 2:
            return _densify_two_point_path(tuple(pts[0]), tuple(pts[1]), samples_per_px)
        if len(pts) < 2:
            return [tuple(map(float, point)) for point in pts]

    r_spline = CubicSpline(t, pts[:, 0], bc_type="natural")
    c_spline = CubicSpline(t, pts[:, 1], bc_type="natural")

    total_len = float(t[-1]) if t[-1] > 0 else float(len(pts) - 1)
    n_samples = max(int(math.ceil(total_len * samples_per_px)), len(pts) * 10)
    tt = np.linspace(0.0, t[-1], n_samples)
    r_vals = r_spline(tt)
    c_vals = c_spline(tt)
    return [(float(r), float(c)) for r, c in zip(r_vals, c_vals)]


def smooth_path(
    grid_free_1: np.ndarray,
    path_rc: List[Tuple[int, int]],
    dist_map: Optional[np.ndarray] = None,
    clear_margin_px: float = 6.0,
    samples_per_px: float = DEFAULT_SPLINE_SAMPLES_PER_PX,
):
    if not path_rc:
        return path_rc

    pruned = prune_path_los(grid_free_1, path_rc)
    if len(pruned) == 2:
        dense = _densify_two_point_path(pruned[0], pruned[1], samples_per_px)
        if (dist_map is None) or path_has_clearance(dist_map, dense, clear_margin_px):
            return dense
        return [(float(r), float(c)) for (r, c) in pruned]

    spline_path = cubic_spline_path(pruned, samples_per_px=samples_per_px)

    if dist_map is None:
        if _path_is_collision_free(grid_free_1, spline_path):
            return spline_path
        if len(pruned) == 2:
            return _densify_two_point_path(pruned[0], pruned[1], samples_per_px)
        return [(float(r), float(c)) for (r, c) in pruned]

    if path_has_clearance(dist_map, spline_path, clear_margin_px):
        return spline_path

    if len(pruned) == 2:
        dense = _densify_two_point_path(pruned[0], pruned[1], samples_per_px)
        if path_has_clearance(dist_map, dense, clear_margin_px):
            return dense

    return [(float(r), float(c)) for (r, c) in pruned]


def inflate_obstacles(grid_free_1: np.ndarray, min_clear: int) -> np.ndarray:
    try:
        occ = (grid_free_1 == 0).astype(np.uint8) * 255
        kernel_size = 2 * min_clear + 1
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        inflated_occ = cv2.dilate(occ, kernel)
        return (inflated_occ == 0).astype(np.uint8)
    except Exception:
        height, width = grid_free_1.shape
        inflated = grid_free_1.copy()
        obstacles = np.argwhere(grid_free_1 == 0)
        for r, c in obstacles:
            r0, r1 = max(0, r - min_clear), min(height, r + min_clear + 1)
            c0, c1 = max(0, c - min_clear), min(width, c + min_clear + 1)
            inflated[r0:r1, c0:c1] = 0
        return inflated


def build_clearance_cost(grid_free_1: np.ndarray, alpha: float = 40.0, sigma: float = 18.0):
    try:
        free_u8 = (grid_free_1 > 0).astype(np.uint8)
        dist = cv2.distanceTransform(free_u8, cv2.DIST_L2, 3).astype(np.float32)
    except Exception:
        from scipy.ndimage import distance_transform_edt

        dist = distance_transform_edt(grid_free_1 > 0).astype(np.float32)

    inv_term = 1.0 / (dist + 1.0)
    exp_term = np.exp(-(dist / max(1e-6, sigma)))
    move_cost = 1.0 + alpha * (0.7 * inv_term + 0.3 * exp_term)
    move_cost[grid_free_1 == 0] = np.inf
    return move_cost.astype(np.float32), dist


def astar(
    grid_free_1: np.ndarray,
    start_rc: Tuple[int, int],
    goal_rc: Tuple[int, int],
    eight_connected: bool = False,
    move_cost: Optional[np.ndarray] = None,
    turn_penalty_scale: float = 0.0,
):
    height, width = grid_free_1.shape
    sr, sc = start_rc
    gr, gc = goal_rc
    if grid_free_1[sr, sc] == 0:
        raise ValueError("Start is in an occupied cell.")
    if grid_free_1[gr, gc] == 0:
        raise ValueError("Goal is in an occupied cell.")

    if move_cost is None:
        move_cost = np.where(grid_free_1 > 0, 1.0, np.inf).astype(np.float32)

    if eight_connected:
        neighbors = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, math.sqrt(2)),
            (-1, 1, math.sqrt(2)),
            (1, -1, math.sqrt(2)),
            (1, 1, math.sqrt(2)),
        ]
    else:
        neighbors = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    def heuristic(r, c):
        return math.hypot(r - gr, c - gc)

    open_heap = []
    came_from = {}
    g_cost = np.full((height, width), np.inf, dtype=np.float64)
    heapq.heappush(open_heap, (heuristic(sr, sc), 0.0, sr, sc, sr, sc))
    g_cost[sr, sc] = 0.0
    closed = np.zeros((height, width), dtype=np.uint8)

    while open_heap:
        f, g, r, c, pr, pc = heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        closed[r, c] = 1

        if (r, c) == (gr, gc):
            path = [(r, c)]
            while (r, c) in came_from:
                r, c = came_from[(r, c)]
                path.append((r, c))
            return list(reversed(path))

        for dr, dc, weight in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid_free_1[nr, nc] == 0:
                continue
            if dr != 0 and dc != 0:
                if grid_free_1[r, nc] == 0 or grid_free_1[nr, c] == 0:
                    continue

            step_cost = weight * 0.5 * (move_cost[r, c] + move_cost[nr, nc])
            turn_pen = 0.0
            if turn_penalty_scale > 0.0 and (r, c) != (pr, pc):
                vr0, vc0 = r - pr, c - pc
                vr1, vc1 = nr - r, nc - c
                dot = vr0 * vr1 + vc0 * vc1
                n0 = math.hypot(vr0, vc0) or 1.0
                n1 = math.hypot(vr1, vc1) or 1.0
                cos = max(-1.0, min(1.0, dot / (n0 * n1)))
                turn_pen = turn_penalty_scale * (1.0 - cos)

            ng = g + step_cost + turn_pen
            if ng < g_cost[nr, nc]:
                g_cost[nr, nc] = ng
                came_from[(nr, nc)] = (r, c)
                nf = ng + heuristic(nr, nc)
                heapq.heappush(open_heap, (nf, ng, nr, nc, r, c))
    return []


def theta_star(grid_free_1: np.ndarray, start_rc, goal_rc, move_cost: Optional[np.ndarray] = None):
    height, width = grid_free_1.shape
    if move_cost is None:
        move_cost = np.where(grid_free_1 > 0, 1.0, np.inf).astype(np.float32)

    sr, sc = start_rc
    gr, gc = goal_rc

    def heuristic(r, c):
        return math.hypot(r - gr, c - gc)

    neighbors = [
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2)),
        (-1, 1, math.sqrt(2)),
        (1, -1, math.sqrt(2)),
        (1, 1, math.sqrt(2)),
    ]

    g_cost = np.full((height, width), np.inf, dtype=np.float64)
    parent = {}
    open_heap = []
    _heapq.heappush(open_heap, (heuristic(sr, sc), (sr, sc)))
    g_cost[sr, sc] = 0.0
    parent[(sr, sc)] = (sr, sc)
    closed = np.zeros((height, width), dtype=np.uint8)

    while open_heap:
        _, (r, c) = _heapq.heappop(open_heap)
        if closed[r, c]:
            continue
        closed[r, c] = 1
        if (r, c) == (gr, gc):
            path = [(r, c)]
            while parent[(r, c)] != (r, c):
                r, c = parent[(r, c)]
                path.append((r, c))
            return list(reversed(path))

        for dr, dc, _weight in neighbors:
            nr, nc = r + dr, c + dc
            if nr < 0 or nr >= height or nc < 0 or nc >= width:
                continue
            if grid_free_1[nr, nc] == 0:
                continue
            if dr != 0 and dc != 0 and (grid_free_1[r, nc] == 0 or grid_free_1[nr, c] == 0):
                continue

            pr, pc = parent[(r, c)]
            if _line_of_sight(grid_free_1, (pr, pc), (nr, nc)):
                base_r, base_c = pr, pc
            else:
                base_r, base_c = r, c

            step = 0.5 * (move_cost[base_r, base_c] + move_cost[nr, nc]) * math.hypot(nr - base_r, nc - base_c)
            ng = g_cost[base_r, base_c] + step

            if ng < g_cost[nr, nc]:
                g_cost[nr, nc] = ng
                parent[(nr, nc)] = (base_r, base_c)
                f = ng + heuristic(nr, nc)
                _heapq.heappush(open_heap, (f, (nr, nc)))
    return []
