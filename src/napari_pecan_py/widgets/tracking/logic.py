"""Multi-object tracking for instance Labels volumes (conveyor L→R).

Takes per-frame instance IDs (e.g. YOLO-seg output that renumbers every frame)
and remaps them to stable track IDs so the same pecan keeps one label/color
while it is visible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Sequence

import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - scipy comes with scikit-image stack
    linear_sum_assignment = None


@dataclass
class TrackingConfig:
    """Parameters for frame-to-frame instance association."""

    max_match_distance: float = 80.0
    """Reject matches whose predicted-centroid distance exceeds this (px)."""

    iou_weight: float = 0.5
    """Blend of ``(1 - bbox IoU)`` into the match cost (0 = distance only)."""

    max_age: int = 5
    """Drop a track after this many consecutive unmatched frames."""

    min_area: float = 20.0
    """Ignore detections smaller than this (px²)."""

    entry_margin_frac: float = 0.25
    """Left fraction of the frame preferred for minting new track IDs."""

    exit_margin_frac: float = 0.15
    """Right fraction where lost tracks are retired immediately."""

    allow_birth_anywhere: bool = True
    """If False, new tracks only start inside the left entry margin."""

    velocity_smooth: float = 0.6
    """EMA factor for velocity updates in ``[0, 1]`` (higher = trust new Δ more)."""


@dataclass
class Detection:
    local_id: int
    cy: float
    cx: float
    area: float
    y0: int
    x0: int
    y1: int
    x1: int


@dataclass
class Track:
    track_id: int
    cy: float
    cx: float
    vy: float = 0.0
    vx: float = 0.0
    area: float = 0.0
    y0: int = 0
    x0: int = 0
    y1: int = 0
    x1: int = 0
    hits: int = 1
    age: int = 1
    time_since_update: int = 0


@dataclass
class TrackingResult:
    labels: np.ndarray
    n_tracks: int
    frames_with_objects: int
    total_frames: int
    id_maps: List[Dict[int, int]] = field(default_factory=list)
    """Per-frame map of source local_id → stable track_id."""


def _bbox_iou(a: Detection | Track, b: Detection | Track) -> float:
    ay0, ax0, ay1, ax1 = int(a.y0), int(a.x0), int(a.y1), int(a.x1)
    by0, bx0, by1, bx1 = int(b.y0), int(b.x0), int(b.y1), int(b.x1)
    inter_y0 = max(ay0, by0)
    inter_x0 = max(ax0, bx0)
    inter_y1 = min(ay1, by1)
    inter_x1 = min(ax1, bx1)
    iw = max(0, inter_x1 - inter_x0)
    ih = max(0, inter_y1 - inter_y0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter) / float(union)


def extract_detections(
    frame: np.ndarray,
    *,
    min_area: float = 20.0,
) -> List[Detection]:
    """Build detections from a 2D instance label map."""
    m = np.asarray(frame)
    if m.ndim != 2:
        raise ValueError(f"Expected 2D label frame, got shape {m.shape}")
    ids = [int(v) for v in np.unique(m) if int(v) > 0]
    dets: List[Detection] = []
    for lid in ids:
        ys, xs = np.where(m == lid)
        area = float(ys.size)
        if area < float(min_area):
            continue
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        dets.append(
            Detection(
                local_id=lid,
                cy=float(ys.mean()),
                cx=float(xs.mean()),
                area=area,
                y0=y0,
                x0=x0,
                y1=y1,
                x1=x1,
            )
        )
    return dets


def _predicted_centroid(track: Track) -> tuple[float, float]:
    return track.cy + track.vy, track.cx + track.vx


def _match_cost(
    track: Track,
    det: Detection,
    config: TrackingConfig,
) -> float:
    py, px = _predicted_centroid(track)
    dy = float(det.cy) - py
    dx = float(det.cx) - px
    dist = float(np.hypot(dy, dx))
    max_d = max(float(config.max_match_distance), 1e-6)
    if dist > max_d:
        return float("inf")
    # Soft penalty for strong leftward jumps on a rightward conveyor.
    direction_pen = 0.0
    if dx < -0.35 * max_d:
        direction_pen = 0.35
    iou = _bbox_iou(track, det)
    iou_term = (1.0 - iou) * float(config.iou_weight)
    return (dist / max_d) + iou_term + direction_pen


def _greedy_assignment(cost: np.ndarray) -> tuple[List[int], List[int]]:
    """Fallback when scipy is unavailable: repeatedly take the lowest cost pair."""
    cost = np.asarray(cost, dtype=np.float64).copy()
    rows: List[int] = []
    cols: List[int] = []
    if cost.size == 0:
        return rows, cols
    while True:
        if not np.isfinite(cost).any():
            break
        flat = int(np.nanargmin(np.where(np.isfinite(cost), cost, np.inf)))
        r, c = divmod(flat, cost.shape[1])
        if not np.isfinite(cost[r, c]):
            break
        rows.append(int(r))
        cols.append(int(c))
        cost[r, :] = np.inf
        cost[:, c] = np.inf
    return rows, cols


def associate_detections(
    tracks: Sequence[Track],
    detections: Sequence[Detection],
    config: TrackingConfig,
) -> tuple[List[tuple[int, int]], List[int], List[int]]:
    """Match tracks to detections.

    Returns ``(matches, unmatched_track_indices, unmatched_det_indices)``.
    """
    if not tracks:
        return [], [], list(range(len(detections)))
    if not detections:
        return [], list(range(len(tracks))), []

    cost = np.full((len(tracks), len(detections)), np.inf, dtype=np.float64)
    for ti, tr in enumerate(tracks):
        for di, det in enumerate(detections):
            cost[ti, di] = _match_cost(tr, det, config)

    finite = np.isfinite(cost)
    if not finite.any():
        return [], list(range(len(tracks))), list(range(len(detections)))

    # Replace inf with a large finite value for the solver, then filter.
    big = 1e6
    cost_solved = np.where(finite, cost, big)
    if linear_sum_assignment is not None:
        ri, ci = linear_sum_assignment(cost_solved)
        row_ind = [int(r) for r in ri]
        col_ind = [int(c) for c in ci]
    else:
        row_ind, col_ind = _greedy_assignment(cost)

    matches: List[tuple[int, int]] = []
    matched_t: set[int] = set()
    matched_d: set[int] = set()
    for r, c in zip(row_ind, col_ind):
        if not np.isfinite(cost[r, c]):
            continue
        matches.append((r, c))
        matched_t.add(r)
        matched_d.add(c)

    unmatched_t = [i for i in range(len(tracks)) if i not in matched_t]
    unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
    return matches, unmatched_t, unmatched_d


def _update_track(track: Track, det: Detection, config: TrackingConfig) -> None:
    dy = float(det.cy) - track.cy
    dx = float(det.cx) - track.cx
    a = float(np.clip(config.velocity_smooth, 0.0, 1.0))
    track.vy = (1.0 - a) * track.vy + a * dy
    track.vx = (1.0 - a) * track.vx + a * dx
    track.cy = float(det.cy)
    track.cx = float(det.cx)
    track.area = float(det.area)
    track.y0, track.x0, track.y1, track.x1 = det.y0, det.x0, det.y1, det.x1
    track.hits += 1
    track.age += 1
    track.time_since_update = 0


def _should_birth(
    det: Detection,
    width: int,
    config: TrackingConfig,
    *,
    first_frame: bool,
) -> bool:
    if first_frame or config.allow_birth_anywhere:
        return True
    margin = max(1, int(round(float(config.entry_margin_frac) * width)))
    return det.cx <= margin


def _should_retire(
    track: Track,
    width: int,
    config: TrackingConfig,
) -> bool:
    if track.time_since_update > int(config.max_age):
        return True
    margin = max(1, int(round(float(config.exit_margin_frac) * width)))
    # Lost near the right edge → assume it left the FOV.
    if track.time_since_update >= 1 and track.cx >= (width - margin):
        return True
    return False


def track_label_volume(
    labels: np.ndarray,
    config: TrackingConfig | None = None,
    *,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> TrackingResult:
    """Remap a ``(T, H, W)`` or ``(H, W)`` instance volume to stable track IDs."""
    cfg = config or TrackingConfig()
    arr = np.asarray(labels)
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Expected (T,H,W) or (H,W) labels, got shape {arr.shape}")

    t_count, height, width = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
    out = np.zeros((t_count, height, width), dtype=np.uint16)
    tracks: List[Track] = []
    next_id = 1
    id_maps: List[Dict[int, int]] = []
    frames_with_objects = 0

    for t in range(t_count):
        if cancel_callback is not None and bool(cancel_callback()):
            break

        dets = extract_detections(arr[t], min_area=cfg.min_area)
        id_map: Dict[int, int] = {}

        if t == 0 or not tracks:
            for det in dets:
                if not _should_birth(det, width, cfg, first_frame=True):
                    continue
                tr = Track(
                    track_id=next_id,
                    cy=det.cy,
                    cx=det.cx,
                    area=det.area,
                    y0=det.y0,
                    x0=det.x0,
                    y1=det.y1,
                    x1=det.x1,
                )
                next_id += 1
                tracks.append(tr)
                id_map[det.local_id] = tr.track_id
        else:
            matches, unmatched_t, unmatched_d = associate_detections(
                tracks, dets, cfg
            )
            for ti, di in matches:
                _update_track(tracks[ti], dets[di], cfg)
                id_map[dets[di].local_id] = tracks[ti].track_id

            surviving: List[Track] = []
            matched_set = {ti for ti, _ in matches}
            for ti, tr in enumerate(tracks):
                if ti in matched_set:
                    surviving.append(tr)
                    continue
                tr.time_since_update += 1
                tr.age += 1
                # Coast with last velocity while briefly unmatched.
                tr.cy = tr.cy + tr.vy
                tr.cx = tr.cx + tr.vx
                if not _should_retire(tr, width, cfg):
                    surviving.append(tr)
            tracks = surviving

            for di in unmatched_d:
                det = dets[di]
                if not _should_birth(det, width, cfg, first_frame=False):
                    continue
                tr = Track(
                    track_id=next_id,
                    cy=det.cy,
                    cx=det.cx,
                    area=det.area,
                    y0=det.y0,
                    x0=det.x0,
                    y1=det.y1,
                    x1=det.x1,
                )
                next_id += 1
                tracks.append(tr)
                id_map[det.local_id] = tr.track_id

        frame_out = np.zeros((height, width), dtype=np.uint16)
        src = arr[t]
        for local_id, track_id in id_map.items():
            frame_out[src == local_id] = np.uint16(track_id)
        out[t] = frame_out
        id_maps.append(id_map)
        if id_map:
            frames_with_objects += 1

        if progress_callback is not None:
            try:
                progress_callback(t + 1, t_count)
            except Exception:
                pass

    n_tracks = int(next_id - 1)
    if labels.ndim == 2 and out.shape[0] == 1:
        return TrackingResult(
            labels=out[0],
            n_tracks=n_tracks,
            frames_with_objects=frames_with_objects,
            total_frames=t_count,
            id_maps=id_maps,
        )
    return TrackingResult(
        labels=out,
        n_tracks=n_tracks,
        frames_with_objects=frames_with_objects,
        total_frames=t_count,
        id_maps=id_maps,
    )


def format_tracking_summary(result: TrackingResult) -> str:
    return (
        f"{result.n_tracks} track(s) across "
        f"{result.frames_with_objects}/{result.total_frames} frame(s)"
    )
