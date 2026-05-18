"""Apply color thresholds to a frame (RGB from napari); uses BGR for OpenCV."""

from __future__ import annotations

import cv2
import numpy as np
from skimage import measure, morphology
from skimage.filters import threshold_otsu

from napari_pecan_py.widgets.color_thresholding.temporal_median_logic import (
    absdiff_scores,
    build_ellipse_roi_mask,
    compute_median_background,
    evenly_spaced_frame_indices,
)

from .defaults import COLOR_SPACE_PARAMS, MASK_COLORS, TARGETS
from .surface_blur import apply_surface_blur


def _ensure_uint8_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    """Ensure (H,W,3) uint8 RGB. Assumes input is either float [0,1] or uint8."""
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame (H,W,3); got shape={frame_rgb.shape}")
    if np.issubdtype(frame_rgb.dtype, np.floating):
        f = np.clip(frame_rgb, 0, 1)
        return (f * 255).astype(np.uint8)
    return np.asarray(frame_rgb, dtype=np.uint8)


def apply_brightness_contrast(frame_rgb: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    """Apply Photoshop-like brightness/contrast to an RGB image.

    Uses a standard Photoshop-like linear transform:
    - contrast affects slope around mid-gray (128)
    - brightness adds an offset scaled to [0..255]
    """
    img = _ensure_uint8_rgb(frame_rgb)
    c = float(contrast)
    b = float(brightness)
    # Photoshop-style contrast factor for contrast in [-100, 100]
    # alpha = (259*(c+255)) / (255*(259-c))
    denom = (255.0 * (259.0 - c))
    alpha = (259.0 * (c + 255.0)) / denom if denom != 0 else 1.0

    # Map brightness in [-100,100] to pixel offset.
    beta = b * (255.0 / 100.0)
    out = alpha * (img.astype(np.float32) - 128.0) + 128.0 + beta
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_levels(
    frame_rgb: np.ndarray,
    in_min: float,
    gamma: float,
    in_max: float,
    out_min: float = 0.0,
    out_max: float = 255.0,
) -> np.ndarray:
    """Apply Photoshop-like Levels (RGB) using a gamma curve between in_min..in_max."""
    img = _ensure_uint8_rgb(frame_rgb).astype(np.float32)
    denom = float(in_max) - float(in_min)
    if denom == 0:
        # Degenerate mapping: everything clamps.
        return np.clip(img * 0 + out_min, 0, 255).astype(np.uint8)

    x = (img - float(in_min)) / denom
    x = np.clip(x, 0.0, 1.0)
    # Photoshop's "Gamma/Midtones" behaves like exponent 1/gamma.
    g = max(float(gamma), 1e-6)
    y = np.power(x, 1.0 / g)
    out = float(out_min) + y * (float(out_max) - float(out_min))
    return np.clip(out, 0, 255).astype(np.uint8)


def _build_lut_from_points(x_points: list[int], y_points: list[int]) -> np.ndarray:
    xs = np.array(list(x_points), dtype=np.float32)
    ys = np.array(list(y_points), dtype=np.float32)
    if xs.shape[0] != ys.shape[0] or xs.shape[0] < 2:
        raise ValueError("Curves require at least 2 matching control points.")

    # Ensure endpoints exist for a full 0..255 LUT.
    if xs[0] != 0:
        xs = np.concatenate([np.array([0], dtype=np.float32), xs])
        ys = np.concatenate([np.array([ys[0]], dtype=np.float32), ys])
    if xs[-1] != 255:
        xs = np.concatenate([xs, np.array([255], dtype=np.float32)])
        ys = np.concatenate([ys, np.array([ys[-1]], dtype=np.float32)])

    # Sort by x for interpolation.
    order = np.argsort(xs)
    xs = xs[order]
    ys = ys[order]

    xgrid = np.arange(256, dtype=np.float32)
    lut = np.interp(xgrid, xs, ys)
    return np.clip(lut, 0, 255).astype(np.uint8)


def apply_curves(
    frame_rgb: np.ndarray,
    x_points: list[int],
    y_points: list[int],
) -> np.ndarray:
    """Apply RGB curves using control points (x,y) with x,y in 0..255."""
    img = _ensure_uint8_rgb(frame_rgb)
    lut = _build_lut_from_points(x_points=x_points, y_points=y_points)
    # LUT indexing works because uint8 values are 0..255.
    return lut[img]


def apply_normalization(
    frame_rgb: np.ndarray,
    method: str = "percentile",
    params: dict | None = None,
) -> np.ndarray:
    """Apply one of several normalization methods to an RGB frame."""
    img = _ensure_uint8_rgb(frame_rgb).astype(np.float32)
    p = dict(params or {})
    m = str(method or "percentile").lower()

    if m == "minmax":
        lo = np.min(img, axis=(0, 1), keepdims=True)
        hi = np.max(img, axis=(0, 1), keepdims=True)
        denom = np.maximum(hi - lo, 1.0)
        out = (img - lo) * (255.0 / denom)
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "percentile":
        lo_p = float(np.clip(p.get("low_percentile", 1.0), 0.0, 99.0))
        hi_p = float(np.clip(p.get("high_percentile", 99.0), lo_p + 1e-3, 100.0))
        lo = np.percentile(img, lo_p, axis=(0, 1), keepdims=True)
        hi = np.percentile(img, hi_p, axis=(0, 1), keepdims=True)
        denom = np.maximum(hi - lo, 1.0)
        out = (img - lo) * (255.0 / denom)
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "zscore":
        mean = np.mean(img, axis=(0, 1), keepdims=True)
        std = np.maximum(np.std(img, axis=(0, 1), keepdims=True), 1e-6)
        z_clip = float(np.clip(p.get("z_clip", 3.0), 0.5, 10.0))
        z = np.clip((img - mean) / std, -z_clip, z_clip)
        out = ((z + z_clip) / (2.0 * z_clip)) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "robust":
        q1 = np.percentile(img, 25.0, axis=(0, 1), keepdims=True)
        q3 = np.percentile(img, 75.0, axis=(0, 1), keepdims=True)
        med = np.median(img, axis=(0, 1), keepdims=True)
        iqr = np.maximum(q3 - q1, 1.0)
        iqr_mult = float(np.clip(p.get("iqr_multiplier", 1.5), 0.5, 5.0))
        lo = med - iqr_mult * iqr
        hi = med + iqr_mult * iqr
        out = (img - lo) * (255.0 / np.maximum(hi - lo, 1.0))
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "unit_l2":
        norm = np.linalg.norm(img, axis=-1, keepdims=True)
        out = (img / np.maximum(norm, 1e-6)) * 255.0
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "luminance_minmax":
        y = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        y_min = float(np.min(y))
        y_max = float(np.max(y))
        if y_max <= y_min:
            return np.clip(img, 0, 255).astype(np.uint8)
        scale = 255.0 / (y_max - y_min)
        out = (img - y_min) * scale
        return np.clip(out, 0, 255).astype(np.uint8)

    if m == "hist_eq":
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        ycrcb = cv2.cvtColor(u8, cv2.COLOR_RGB2YCrCb)
        ycrcb[..., 0] = cv2.equalizeHist(ycrcb[..., 0])
        return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

    if m == "clahe":
        u8 = np.clip(img, 0, 255).astype(np.uint8)
        clip_limit = float(np.clip(p.get("clip_limit", 2.0), 0.1, 40.0))
        tile = int(np.clip(p.get("tile_grid_size", 8), 2, 32))
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile, tile))
        lab = cv2.cvtColor(u8, cv2.COLOR_RGB2LAB)
        lab[..., 0] = clahe.apply(lab[..., 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    # Fallback to percentile for unknown method keys.
    lo_p = float(np.clip(p.get("low_percentile", 1.0), 0.0, 99.0))
    hi_p = float(np.clip(p.get("high_percentile", 99.0), lo_p + 1e-3, 100.0))
    lo = np.percentile(img, lo_p, axis=(0, 1), keepdims=True)
    hi = np.percentile(img, hi_p, axis=(0, 1), keepdims=True)
    denom = np.maximum(hi - lo, 1.0)
    out = (img - lo) * (255.0 / denom)
    return np.clip(out, 0, 255).astype(np.uint8)


def _ensure_temporal_median_cached(
    video_rgb: np.ndarray,
    adj: dict,
    cache: dict,
) -> np.ndarray:
    """Compute and cache per-pixel median (H,W,3) float32 from ``video_rgb`` (T,H,W,3)."""
    key = (
        "median_hwc",
        int(adj.get("n_sample_frames", 120)),
        video_rgb.shape,
    )
    if cache.get("_median_key") == key and "median_hwc" in cache:
        return cache["median_hwc"]

    arr = np.asarray(video_rgb)
    if arr.ndim != 4 or arr.shape[-1] < 3:
        raise ValueError("temporal_median_diff requires video_rgb shaped (T,H,W,3+)")
    T = int(arr.shape[0])
    n = max(1, min(int(adj.get("n_sample_frames", 120)), T))
    idx = evenly_spaced_frame_indices(T, n)
    stack = np.stack([arr[int(i), ..., :3].astype(np.float32, copy=False) for i in idx], axis=0)
    median_hwc = compute_median_background(stack)
    cache["_median_key"] = key
    cache["median_hwc"] = median_hwc
    return median_hwc


def _scores_to_preview_rgb(
    scores: np.ndarray,
    low_pct: float,
    high_pct: float,
) -> np.ndarray:
    lo = float(np.percentile(scores, np.clip(low_pct, 0.0, 100.0)))
    hi = float(np.percentile(scores, np.clip(high_pct, 0.0, 100.0)))
    if hi <= lo + 1e-9:
        g = np.zeros_like(scores, dtype=np.float32)
    else:
        g = np.clip((scores.astype(np.float32) - lo) / (hi - lo), 0.0, 1.0)
    u8 = (g * 255.0).astype(np.uint8)
    return np.stack([u8, u8, u8], axis=-1)


def apply_temporal_median_diff_rgb(
    frame_rgb: np.ndarray,
    adj: dict,
    *,
    video_rgb: np.ndarray,
    frame_index: int,
    cache: dict,
) -> np.ndarray:
    """|frame − median(video)| stretched to uint8 RGB for preview (``frame_index`` unused; median from ``video_rgb``)."""
    _ = int(frame_index)
    median_hwc = _ensure_temporal_median_cached(video_rgb, adj, cache)
    use_lum = bool(adj.get("use_luminance", False))
    scores = absdiff_scores(
        frame_rgb,
        median_hwc,
        use_luminance_only=use_lum,
        luminance_weights=(0.299, 0.587, 0.114),
    )
    lo_p = float(adj.get("preview_low_percentile", 2.0))
    hi_p = float(adj.get("preview_high_percentile", 98.0))
    return _scores_to_preview_rgb(scores, lo_p, hi_p)


def apply_motion_mask_threshold_rgb(frame_rgb: np.ndarray, adj: dict) -> np.ndarray:
    """Threshold a motion-score preview (typically gray×3) into a binary mask RGB."""
    img = _ensure_uint8_rgb(frame_rgb)
    g = img.astype(np.float32).mean(axis=-1)
    h, w = g.shape
    use_ellipse = bool(adj.get("use_ellipse", False))
    roi: np.ndarray | None = None
    if use_ellipse:
        roi = build_ellipse_roi_mask(
            (h, w),
            (float(adj.get("ellipse_center_row", 0.0)), float(adj.get("ellipse_center_col", 0.0))),
            (float(adj.get("ellipse_radius_row", 1.0)), float(adj.get("ellipse_radius_col", 1.0))),
            float(adj.get("ellipse_angle_deg", 0.0)),
        )
    eval_mask = roi if roi is not None else np.ones((h, w), dtype=bool)
    vals = g[eval_mask]

    mode = str(adj.get("threshold_mode", "otsu")).lower()
    if vals.size == 0:
        thr = 0.0
    elif mode == "fixed":
        thr = float(adj.get("fixed_threshold", 25.0))
    elif mode == "quantile":
        q = float(np.clip(float(adj.get("quantile", 0.88)), 0.0, 1.0))
        thr = float(np.quantile(vals.astype(np.float64, copy=False), q))
    else:
        v = vals.astype(np.float64, copy=False)
        thr = float(v.mean()) if np.ptp(v) < 1e-9 else float(threshold_otsu(v))

    binary = g > thr
    if roi is not None:
        binary &= roi
    u8 = (binary.astype(np.uint8) * 255)
    return np.stack([u8, u8, u8], axis=-1)


def apply_mask_morphology_rgb(frame_rgb: np.ndarray, adj: dict) -> np.ndarray:
    """Binary closing/opening on a mask encoded as RGB (uses max channel > 127)."""
    img = _ensure_uint8_rgb(frame_rgb)
    m = np.max(img, axis=-1) > 127
    cr = int(np.clip(int(adj.get("close_radius", 3)), 0, 50))
    orr = int(np.clip(int(adj.get("open_radius", 2)), 0, 50))
    if cr > 0:
        m = morphology.binary_closing(m, footprint=morphology.disk(cr))
    if orr > 0:
        m = morphology.binary_opening(m, footprint=morphology.disk(orr))
    u8 = m.astype(np.uint8) * 255
    return np.stack([u8, u8, u8], axis=-1)


def apply_mask_largest_component_rgb(frame_rgb: np.ndarray, adj: dict) -> np.ndarray:
    """Keep the largest connected component of a binary mask RGB."""
    img = _ensure_uint8_rgb(frame_rgb)
    m = np.max(img, axis=-1) > 127
    if not np.any(m):
        return np.zeros_like(img, dtype=np.uint8)
    lab = measure.label(m, connectivity=2)
    regions = measure.regionprops(lab)
    if not regions:
        return np.zeros_like(img, dtype=np.uint8)
    best = max(regions, key=lambda r: r.area)
    min_area = int(adj.get("min_area_px", 200))
    if best.area < min_area:
        return np.zeros_like(img, dtype=np.uint8)
    out = (lab == best.label).astype(np.uint8) * 255
    return np.stack([out, out, out], axis=-1)


def apply_adjustment_stack(
    frame_rgb: np.ndarray,
    adjustment_stack: list[dict],
    *,
    video_rgb: np.ndarray | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Apply an ordered list of RGB adjustments to an (H,W,3) frame.

    Parameters
    ----------
    video_rgb :
        Full source time series ``(T,H,W,3)`` uint8/float. Required when the stack
        contains ``temporal_median_diff`` (median is computed from this volume;
        the current ``frame_rgb`` is still the frame after prior steps in the stack).
    frame_index :
        Index of ``frame_rgb`` in ``video_rgb`` (for future use; median uses all T).
    """
    img = frame_rgb
    temporal_cache: dict = {}
    for adj in adjustment_stack or []:
        if not isinstance(adj, dict):
            continue
        if not adj.get("enabled", True):
            continue
        typ = adj.get("type")
        if typ == "brightness_contrast":
            img = apply_brightness_contrast(
                img,
                brightness=float(adj.get("brightness", 0.0)),
                contrast=float(adj.get("contrast", 0.0)),
            )
        elif typ == "levels":
            img = apply_levels(
                img,
                in_min=float(adj.get("in_min", 0.0)),
                gamma=float(adj.get("gamma", 1.0)),
                in_max=float(adj.get("in_max", 255.0)),
                out_min=float(adj.get("out_min", 0.0)),
                out_max=float(adj.get("out_max", 255.0)),
            )
        elif typ == "curves":
            img = apply_curves(
                img,
                x_points=list(adj.get("x_points", list(range(4)))),
                y_points=list(adj.get("y_points", [0, 64, 128, 255])),
            )
        elif typ == "surface_blur":
            img = apply_surface_blur(
                img,
                radius=int(adj.get("radius", 26)),
                threshold=int(adj.get("threshold", 20)),
            )
        elif typ == "normalization":
            method = str(adj.get("method", "percentile"))
            img = apply_normalization(
                img,
                method=method,
                params=adj,
            )
        elif typ == "temporal_median_diff":
            if video_rgb is None:
                raise ValueError(
                    "temporal_median_diff needs a (T,H,W,3) time series. "
                    "Select a video layer (not a single 2D image) and ensure the stack is applied from Adjustments."
                )
            img = apply_temporal_median_diff_rgb(
                img, adj, video_rgb=np.asarray(video_rgb), frame_index=int(frame_index), cache=temporal_cache
            )
        elif typ == "motion_mask_threshold":
            img = apply_motion_mask_threshold_rgb(img, adj)
        elif typ == "mask_morphology":
            img = apply_mask_morphology_rgb(img, adj)
        elif typ == "mask_largest_component":
            img = apply_mask_largest_component_rgb(img, adj)
        else:
            # Unknown adjustment types are ignored to keep tuning robust.
            continue
    return img


def _frame_rgb_to_bgr(frame_rgb: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) RGB to BGR for OpenCV."""
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def _frame_in_color_space(frame_bgr: np.ndarray, color_space: str) -> np.ndarray:
    """Convert BGR frame to the given color space; 'rgb' returns BGR (OpenCV channel order)."""
    if color_space == "rgb":
        return frame_bgr
    code = COLOR_SPACE_PARAMS[color_space]["conversion_code"]
    return cv2.cvtColor(frame_bgr, code)


def frame_rgb_to_color_space(frame_rgb: np.ndarray, color_space: str) -> np.ndarray:
    """Convert an (H, W, 3) RGB uint8 frame to the requested color space.

    Returns the converted frame with the same spatial shape.
    For 'rgb' the result is in BGR channel order (OpenCV convention used by thresholding).
    """
    frame_bgr = _frame_rgb_to_bgr(frame_rgb)
    return _frame_in_color_space(frame_bgr, color_space)


def apply_thresholds(
    frame_rgb: np.ndarray,
    color_space: str,
    target: str,
    thresholds: dict,
) -> np.ndarray:
    """
    Apply thresholds for one target to the current frame.

    frame_rgb: (H, W, 3) uint8 RGB (e.g. from napari Image layer).
    color_space: 'rgb', 'hsv', or 'lab'.
    target: one of pecan, kernel, damaged_kernel, crack, background.
    thresholds: dict[color_space][target]['lower'/'upper'] (length-3 uint8 arrays).

    Returns binary mask (H, W) uint8, 0 or 255.
    """
    frame_bgr = _frame_rgb_to_bgr(frame_rgb)
    frame_cs = _frame_in_color_space(frame_bgr, color_space)
    th = thresholds.get(color_space, {}).get(target, {})
    lower = np.array(th.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.array(th.get("upper", [255, 255, 255]), dtype=np.uint8)
    mask = cv2.inRange(frame_cs, lower, upper)
    return mask


def composite_masks(
    frame_rgb: np.ndarray,
    thresholds: dict,
    color_space: str,
) -> np.ndarray:
    """
    Build a 3-channel BGR overlay: each target's mask colored by MASK_COLORS.
    Result is (H, W, 3) uint8 in BGR (can be shown as RGB in napari by converting).
    """
    frame_bgr = _frame_rgb_to_bgr(frame_rgb)
    frame_cs = _frame_in_color_space(frame_bgr, color_space)
    h, w = frame_rgb.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for target in TARGETS:
        th = thresholds.get(color_space, {}).get(target, {})
        lower = np.array(th.get("lower", [0, 0, 0]), dtype=np.uint8)
        upper = np.array(th.get("upper", [255, 255, 255]), dtype=np.uint8)
        mask = cv2.inRange(frame_cs, lower, upper)
        color = MASK_COLORS.get(target, (128, 128, 128))
        out[mask > 0] = color
    return out
