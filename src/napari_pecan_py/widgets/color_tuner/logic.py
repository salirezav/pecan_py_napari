"""Apply color thresholds to a frame (RGB from napari); uses BGR for OpenCV."""

from __future__ import annotations

import cv2
import numpy as np

from .defaults import COLOR_SPACE_PARAMS, MASK_COLORS, TARGETS


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
