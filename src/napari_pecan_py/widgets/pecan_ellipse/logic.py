"""Fit an encompassing ellipse to a pecan (or other) region from a mask slice.

Uses OpenCV: binary mask → largest outer contour → ``cv2.fitEllipse``.
Coordinates follow OpenCV image convention internally (x = column, y = row);
outputs for napari use (row, col) = (y, x) for 2D or (t, y, x) when aligned
with a time-first label volume.
"""

from __future__ import annotations

import cv2
import numpy as np

__all__ = [
    "apply_ellipse_pipeline",
    "extract_mask_2d",
    "mask_to_binary_u8",
    "mask_volume_needs_time_coord",
    "resolve_time_index_for_volume",
    "fit_ellipse_from_binary",
    "fit_debug_summary",
    "opencv_ellipse_to_napari_vertices",
]


def extract_mask_2d(data: np.ndarray, frame_index: int) -> np.ndarray:
    """Return a single 2D mask array (H, W) from possibly volumetric data.

    * ``(H, W)`` — returned as-is.
    * ``(T, H, W)`` — ``data[frame_index]`` (typical label time series).
    * ``(H, W, C)`` with ``C in (3, 4)`` — treated as one RGB(A) frame; converted
      to a boolean mask (any channel > 0 after averaging to gray, then > 0).
    * ``(T, H, W, C)`` — one time slice, same RGB handling.
    """
    a = np.asarray(data)
    if a.ndim == 2:
        return a
    if a.ndim == 3:
        if a.shape[-1] in (3, 4):
            g = np.mean(a[..., :3], axis=-1)
            return (g > 0).astype(np.uint8)
        t = int(np.clip(frame_index, 0, a.shape[0] - 1))
        return a[t]
    if a.ndim == 4:
        t = int(np.clip(frame_index, 0, a.shape[0] - 1))
        sl = a[t]
        if sl.shape[-1] in (3, 4):
            g = np.mean(sl[..., :3], axis=-1)
            return (g > 0).astype(np.uint8)
        return sl
    raise ValueError(f"Unsupported mask array shape: {a.shape}")


def mask_to_binary_u8(slice_2d: np.ndarray, *, label_id: int | None) -> np.ndarray:
    """Build uint8 {0,255} binary image from a label or probability mask."""
    s = np.asarray(slice_2d)
    if label_id is not None and label_id > 0:
        lid = int(label_id)
        match = s == lid
        if not np.any(match) and np.any(s > 0):
            pos = s[s > 0]
            uniq = np.unique(pos)
            if uniq.size == 1:
                match = s == uniq[0]
            else:
                match = s > 0
        bin_mask = match.astype(np.uint8)
    else:
        if s.dtype in (np.float32, np.float64):
            bin_mask = (s > 0.5).astype(np.uint8) if float(s.max()) <= 1.0 else (s > 0).astype(np.uint8)
        else:
            bin_mask = (s > 0).astype(np.uint8)
    return (bin_mask * 255).astype(np.uint8)


def fit_ellipse_from_binary(
    binary_u8: np.ndarray,
    *,
    largest_only: bool = True,
) -> tuple[tuple[float, float], tuple[float, float], float] | None:
    """Return OpenCV ``fitEllipse`` result, or None if not enough foreground.

    Return is ``((cx, cy), (width, height), angle_degrees)`` in OpenCV coords.
    """
    contours, _ = cv2.findContours(
        np.asarray(binary_u8),
        cv2.RETR_EXTERNAL,
        # SIMPLE often leaves only 4 corners on rectangles; fitEllipse needs ≥5 points.
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return None

    def _fit_one(c) -> tuple[tuple[float, float], tuple[float, float], float] | None:
        if c is None or len(c) < 5:
            return None
        # OpenCV expects Nx1x2; float32 avoids some rare dtype errors.
        c = np.asarray(c, dtype=np.float32).reshape(-1, 1, 2)
        try:
            return cv2.fitEllipse(c)
        except Exception:
            return None

    if largest_only:
        cnt = max(contours, key=cv2.contourArea)
        return _fit_one(cnt)

    # Merge outer contours; fit can fail on fragmented / disjoint sets — fall back.
    cnt = np.concatenate(contours)
    ell = _fit_one(cnt)
    if ell is not None:
        return ell
    cnt_big = max(contours, key=cv2.contourArea)
    return _fit_one(cnt_big)


def opencv_ellipse_to_napari_vertices(
    fit: tuple[tuple[float, float], tuple[float, float], float],
    *,
    time_index: int | None = None,
) -> np.ndarray:
    """Convert ``cv2.fitEllipse`` output to napari ellipse vertices.

    * 2D: shape ``(4, 2)`` with rows ``(row, col)`` = ``(y, x)``.
    * With ``time_index``: shape ``(4, 3)`` as ``(t, y, x)`` for ``(T, Y, X)`` layers.
    """
    (cx, cy), (w, h), angle_deg = fit
    a = float(w) * 0.5
    b = float(h) * 0.5
    theta = np.deg2rad(float(angle_deg))
    # Semi-axes directions in CV (x, y) with x right, y down.
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
    corners_cv: list[np.ndarray] = []
    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corners_cv.append(sx * a * u + sy * b * v + np.array([cx, cy], dtype=np.float64))
    pts_cv = np.stack(corners_cv, axis=0)
    # napari 2D uses (y, x)
    yx = np.stack([pts_cv[:, 1], pts_cv[:, 0]], axis=1)
    if time_index is None:
        return yx.astype(np.float64)
    tcol = np.full((4, 1), float(time_index), dtype=np.float64)
    return np.concatenate([tcol, yx], axis=1)


def mask_volume_needs_time_coord(arr: np.ndarray) -> bool:
    """True if napari vertices should be ``(t, y, x)`` (time-first volume)."""
    a = np.asarray(arr)
    if a.ndim <= 2:
        return False
    if a.ndim == 3:
        return a.shape[-1] not in (3, 4)
    if a.ndim == 4:
        return True
    return False


def resolve_time_index_for_volume(data: np.ndarray, viewer) -> int:
    """Pick the viewer slider value that matches the volume length along ``data`` axis 0.

    Napari can reorder axes so ``current_step[0]`` is not always time; match
    ``dims.nsteps[i]`` to ``data.shape[0]`` when possible.
    """
    a = np.asarray(data)
    if not mask_volume_needs_time_coord(a):
        return 0
    t_size = int(a.shape[0])
    try:
        steps = tuple(int(x) for x in viewer.dims.nsteps)
        curr = tuple(int(x) for x in viewer.dims.current_step)
    except Exception:
        return 0
    match_axes = [i for i, n in enumerate(steps) if n == t_size]
    if len(match_axes) == 1:
        i = match_axes[0]
        return int(np.clip(curr[i], 0, t_size - 1))
    if len(steps) >= 1 and steps[0] == t_size:
        return int(np.clip(curr[0], 0, t_size - 1))
    return int(np.clip(curr[0], 0, t_size - 1))


def apply_ellipse_pipeline(
    data: np.ndarray,
    frame_index: int,
    *,
    label_id: int | None = 1,
    largest_only: bool = True,
) -> np.ndarray | None:
    """Return napari-style ellipse vertices for one frame, or None."""
    sl = extract_mask_2d(data, frame_index)
    bin_u8 = mask_to_binary_u8(sl, label_id=label_id)
    fit = fit_ellipse_from_binary(bin_u8, largest_only=largest_only)
    if fit is None:
        return None
    t_idx = int(frame_index) if mask_volume_needs_time_coord(data) else None
    return opencv_ellipse_to_napari_vertices(fit, time_index=t_idx)


def fit_debug_summary(
    data: np.ndarray,
    frame_index: int,
    *,
    label_id: int | None,
) -> str:
    """Human-readable hint when ellipse fitting returns None."""
    sl = extract_mask_2d(data, frame_index)
    u = np.unique(sl)
    if u.size > 8:
        u_show = f"{u[:4]} ... (+{u.size - 4} values)"
    else:
        u_show = str(u)
    bin_u8 = mask_to_binary_u8(sl, label_id=label_id)
    fg = int(np.sum(bin_u8 > 0))
    return (
        f"frame index {frame_index}, slice {sl.shape}, "
        f"unique pixel values {u_show}, "
        f"foreground after ID filter: {fg} px"
    )
