"""Spatial denoise for RGB frames (adjustment-stack op).

Methods cover the usual noise regimes for industrial RGB video:

- **gaussian** — fast isotropic blur (mild sensor noise)
- **median** — impulse / salt-and-pepper speckles
- **bilateral** — edge-preserving smooth (same family as Surface Blur)
- **nlmeans** — OpenCV colored non-local means (strong Gaussian-like noise)
- **tv** — total-variation Chambolle (piecewise-smooth; good before seg)
- **wavelet** — multi-scale wavelet shrink (structured / mixed noise)

Kept under ``color_thresholding`` (not ``color_adjustments``) so
``color_thresholding.logic`` does not import the adjustments package.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.restoration import denoise_tv_chambolle


def _ensure_uint8_rgb(frame_rgb: np.ndarray) -> np.ndarray:
    if frame_rgb.ndim != 3 or frame_rgb.shape[-1] != 3:
        raise ValueError(f"Expected RGB frame (H,W,3); got shape={frame_rgb.shape}")
    if np.issubdtype(frame_rgb.dtype, np.floating):
        f = np.clip(frame_rgb, 0, 1)
        return (f * 255).astype(np.uint8)
    return np.asarray(frame_rgb, dtype=np.uint8)


def _odd_ksize(value: int, lo: int = 3, hi: int = 31) -> int:
    k = int(np.clip(value, lo, hi))
    if k % 2 == 0:
        k -= 1
    return max(lo if lo % 2 else lo + 1, k)


def apply_denoise(
    frame_rgb: np.ndarray,
    method: str = "gaussian",
    params: dict | None = None,
) -> np.ndarray:
    """Apply one of several spatial denoise methods to an RGB frame."""
    img = _ensure_uint8_rgb(frame_rgb)
    p = dict(params or {})
    m = str(method or "gaussian").lower()

    if m == "gaussian":
        k = _odd_ksize(int(p.get("ksize", 5)), lo=3, hi=31)
        sigma = float(np.clip(p.get("sigma", 0.0), 0.0, 20.0))
        return cv2.GaussianBlur(img, (k, k), sigmaX=sigma)

    if m == "median":
        k = _odd_ksize(int(p.get("ksize", 5)), lo=3, hi=15)
        return cv2.medianBlur(img, k)

    if m == "bilateral":
        d = int(np.clip(p.get("diameter", 9), 1, 25))
        if d % 2 == 0:
            d += 1
        sigma_color = float(np.clip(p.get("sigma_color", 75.0), 1.0, 250.0))
        sigma_space = float(np.clip(p.get("sigma_space", 75.0), 1.0, 250.0))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.bilateralFilter(bgr, d, sigma_color, sigma_space)
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if m == "nlmeans":
        h = float(np.clip(p.get("h", 10.0), 1.0, 30.0))
        h_color = float(np.clip(p.get("h_color", h), 1.0, 30.0))
        template = int(np.clip(p.get("template_window", 7), 3, 21))
        search = int(np.clip(p.get("search_window", 21), 7, 35))
        if template % 2 == 0:
            template += 1
        if search % 2 == 0:
            search += 1
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out = cv2.fastNlMeansDenoisingColored(
            bgr,
            None,
            h,
            h_color,
            template,
            search,
        )
        return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

    if m == "tv":
        weight = float(np.clip(p.get("weight", 0.1), 0.01, 2.0))
        f = img.astype(np.float32) / 255.0
        out = denoise_tv_chambolle(f, weight=weight, channel_axis=-1)
        return np.clip(np.asarray(out) * 255.0, 0, 255).astype(np.uint8)

    if m == "wavelet":
        try:
            from skimage.restoration import denoise_wavelet as _denoise_wavelet
        except Exception as exc:  # pragma: no cover - depends on optional pywt
            raise ImportError(
                "Wavelet denoise requires PyWavelets (pip install PyWavelets)."
            ) from exc
        try:
            import pywt  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "Wavelet denoise requires PyWavelets (pip install PyWavelets)."
            ) from exc
        sigma = float(np.clip(p.get("sigma_wavelet", 0.0), 0.0, 1.0))
        f = img.astype(np.float32) / 255.0
        kw: dict = {
            "image": f,
            "channel_axis": -1,
            "convert2ycbcr": True,
            "rescale_sigma": True,
        }
        if sigma > 0.0:
            kw["sigma"] = sigma
        out = _denoise_wavelet(**kw)
        return np.clip(np.asarray(out) * 255.0, 0, 255).astype(np.uint8)

    # Fallback: mild Gaussian.
    return cv2.GaussianBlur(img, (5, 5), sigmaX=0.0)
