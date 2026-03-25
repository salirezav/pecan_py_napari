"""Apply color thresholds to a frame (RGB from napari); uses BGR for OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .defaults import COLOR_SPACE_PARAMS, MASK_COLORS, TARGETS


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


def apply_adjustment_stack(
    frame_rgb: np.ndarray,
    adjustment_stack: list[dict],
) -> np.ndarray:
    """Apply an ordered list of RGB adjustments to an (H,W,3) frame."""
    img = frame_rgb
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
