"""Fit an encompassing ellipse to a pecan (or other) region from a mask slice.

Uses OpenCV: binary mask → largest outer contour → ``cv2.fitEllipse``.
Coordinates follow OpenCV image convention internally (x = column, y = row);
outputs for napari use (row, col) = (y, x) for 2D or (t, y, x) when aligned
with a time-first label volume.
"""

from __future__ import annotations

import cv2
import numpy as np

OpenCVEllipse = tuple[tuple[float, float], tuple[float, float], float]

__all__ = [
    "apply_ellipse_pipeline",
    "extract_mask_2d",
    "fit_ellipses_volume",
    "mask_to_binary_u8",
    "mask_volume_needs_time_coord",
    "resolve_time_index_for_volume",
    "fit_ellipse_from_binary",
    "fit_debug_summary",
    "napari_vertices_to_opencv_fit",
    "normalize_smooth_window",
    "opencv_ellipse_to_napari_vertices",
    "smooth_opencv_ellipse_sequence",
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


def volume_shape_for_time(data) -> tuple[int, ...] | None:
    """Shape of a volume without materializing lazy adapters (e.g. ``LazyVideoArray``)."""
    shape = getattr(data, "shape", None)
    if shape is not None:
        return tuple(int(x) for x in shape)
    try:
        return tuple(int(x) for x in np.asarray(data).shape)
    except Exception:
        return None


def resolve_time_index_for_volume(data, viewer) -> int:
    """Pick the viewer slider value that matches the volume length along ``data`` axis 0.

    Napari can reorder axes so ``current_step[0]`` is not always time; match
    ``dims.nsteps[i]`` to ``data.shape[0]`` when possible.
    """
    shape = volume_shape_for_time(data)
    if shape is None:
        return 0
    if len(shape) == 4:
        t_size = int(shape[0])
    elif len(shape) == 3 and shape[-1] not in (3, 4):
        t_size = int(shape[0])
    else:
        return 0
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


def normalize_smooth_window(window: int) -> int:
    """Return an odd window size >= 3 for temporal smoothing."""
    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    return w


def _rolling_nan_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average that ignores NaNs."""
    n = int(values.size)
    out = np.full(n, np.nan, dtype=np.float64)
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        chunk = values[lo:hi]
        valid = chunk[~np.isnan(chunk)]
        if valid.size:
            out[i] = float(valid.mean())
    return out


def _smooth_angles_deg(angles_deg: np.ndarray, window: int) -> np.ndarray:
    """Smooth OpenCV ellipse angles (0–180°) with 180° ambiguity handled."""
    rad = np.deg2rad(angles_deg.astype(np.float64))
    sin2 = np.sin(2.0 * rad)
    cos2 = np.cos(2.0 * rad)
    sin2_s = _rolling_nan_mean(sin2, window)
    cos2_s = _rolling_nan_mean(cos2, window)
    out = np.rad2deg(0.5 * np.arctan2(sin2_s, cos2_s))
    return np.mod(out, 180.0)


def smooth_opencv_ellipse_sequence(
    fits: list[OpenCVEllipse | None],
    window: int,
) -> list[OpenCVEllipse | None]:
    """Smooth ellipse parameters over time; frames without a raw fit stay None."""
    n = len(fits)
    if n < 2:
        return list(fits)
    window = normalize_smooth_window(window)

    cx = np.full(n, np.nan)
    cy = np.full(n, np.nan)
    w = np.full(n, np.nan)
    h = np.full(n, np.nan)
    ang = np.full(n, np.nan)
    for i, fit in enumerate(fits):
        if fit is None:
            continue
        (center, size, angle) = fit
        cx[i] = float(center[0])
        cy[i] = float(center[1])
        w[i] = float(size[0])
        h[i] = float(size[1])
        ang[i] = float(angle)

    cx_s = _rolling_nan_mean(cx, window)
    cy_s = _rolling_nan_mean(cy, window)
    w_s = _rolling_nan_mean(w, window)
    h_s = _rolling_nan_mean(h, window)
    ang_s = _smooth_angles_deg(ang, window)

    out: list[OpenCVEllipse | None] = [None] * n
    for i in range(n):
        if fits[i] is None:
            continue
        if not (
            np.isfinite(cx_s[i])
            and np.isfinite(cy_s[i])
            and np.isfinite(w_s[i])
            and np.isfinite(h_s[i])
            and np.isfinite(ang_s[i])
        ):
            out[i] = fits[i]
            continue
        out[i] = (
            (float(cx_s[i]), float(cy_s[i])),
            (float(w_s[i]), float(h_s[i])),
            float(ang_s[i]),
        )
    return out


def napari_vertices_to_opencv_fit(vertices: np.ndarray) -> OpenCVEllipse | None:
    """Recover OpenCV-style ellipse parameters from napari ellipse box vertices."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape == (4, 3):
        yx = v[:, 1:3]
    elif v.shape == (4, 2):
        yx = v
    else:
        return None
    pts_cv = np.stack([yx[:, 1], yx[:, 0]], axis=1)
    center = pts_cv.mean(axis=0)
    du = pts_cv[1] - pts_cv[0]
    dv = pts_cv[3] - pts_cv[0]
    w = float(np.linalg.norm(du))
    h = float(np.linalg.norm(dv))
    if w < 1e-6 or h < 1e-6:
        return None
    u = du / w
    angle_deg = float(np.rad2deg(np.arctan2(u[1], u[0])) % 180.0)
    return ((float(center[0]), float(center[1])), (w, h), angle_deg)


def _frame_count_for_volume(data: np.ndarray) -> int:
    a = np.asarray(data)
    if a.ndim <= 2:
        return 1
    if a.ndim == 3 and a.shape[-1] in (3, 4):
        return 1
    return int(a.shape[0])


def fit_ellipse_opencv_for_frame(
    data: np.ndarray,
    frame_index: int,
    *,
    label_id: int | None = 1,
    largest_only: bool = True,
) -> OpenCVEllipse | None:
    """Return raw OpenCV ellipse fit for one frame, or None."""
    sl = extract_mask_2d(data, frame_index)
    bin_u8 = mask_to_binary_u8(sl, label_id=label_id)
    return fit_ellipse_from_binary(bin_u8, largest_only=largest_only)


def fit_ellipses_volume(
    data: np.ndarray,
    *,
    label_id: int | None = 1,
    largest_only: bool = True,
    temporal_smooth: bool = False,
    smooth_window: int = 5,
) -> list[np.ndarray]:
    """Fit ellipses for every frame; optional temporal smoothing of parameters."""
    n = _frame_count_for_volume(data)
    needs_time = mask_volume_needs_time_coord(data)
    fits: list[OpenCVEllipse | None] = []
    for t in range(n):
        fits.append(
            fit_ellipse_opencv_for_frame(
                data, t, label_id=label_id, largest_only=largest_only
            )
        )
    if temporal_smooth and n >= 2:
        fits = smooth_opencv_ellipse_sequence(fits, smooth_window)

    verts: list[np.ndarray] = []
    for t, fit in enumerate(fits):
        if fit is None:
            continue
        t_idx = int(t) if needs_time else None
        verts.append(opencv_ellipse_to_napari_vertices(fit, time_index=t_idx))
    return verts


def apply_ellipse_pipeline(
    data: np.ndarray,
    frame_index: int,
    *,
    label_id: int | None = 1,
    largest_only: bool = True,
) -> np.ndarray | None:
    """Return napari-style ellipse vertices for one frame, or None."""
    fit = fit_ellipse_opencv_for_frame(
        data, frame_index, label_id=label_id, largest_only=largest_only
    )
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
