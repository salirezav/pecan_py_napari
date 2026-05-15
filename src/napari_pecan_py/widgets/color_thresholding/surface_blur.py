"""Surface Blur–style edge-preserving smoothing (Photoshop-like).

Photoshop's **Surface Blur** is not shipped in OpenCV. We approximate it with
``cv2.bilateralFilter`` on BGR:

- **Radius** → spatial extent via ``d = 2 * radius + 1`` (odd diameter) and
  ``sigmaSpace ≈ radius``.
- **Threshold** → color edge preservation via ``sigmaColor ≈ threshold`` (pixel
  value difference in 8-bit space).

This is a pragmatic match for labeling workflows, not a bit-exact Adobe clone.
Large **Radius** values can be slow on full-resolution video frames.

Kept under ``color_tuner`` (not ``color_adjustments``) so ``color_tuner.logic``
does not import the adjustments package and avoids circular imports.
"""

from __future__ import annotations

import cv2
import numpy as np


def _ensure_uint8_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame (H,W,3); got shape={frame_rgb.shape}")
    if np.issubdtype(frame_rgb.dtype, np.floating):
        f = np.clip(frame_rgb, 0, 1)
        return (f * 255).astype(np.uint8)
    return np.asarray(frame_rgb, dtype=np.uint8)


def apply_surface_blur(
    frame_rgb: np.ndarray,
    radius: int,
    threshold: int,
) -> np.ndarray:
    """Apply edge-preserving blur; ``radius`` and ``threshold`` match widget naming."""
    img = _ensure_uint8_rgb(frame_rgb)
    r = int(np.clip(radius, 1, 100))
    t = int(np.clip(threshold, 0, 255))
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    d = 2 * r + 1
    if d > 255:
        d = 255
    if d % 2 == 0:
        d -= 1
    if d < 3:
        d = 3
    # sigmaColor 0 makes bilateral a no-op in practice; use >= 1
    sigma_color = max(float(t), 1.0)
    sigma_space = float(max(r, 1))
    out = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
