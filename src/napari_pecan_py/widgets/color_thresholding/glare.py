"""Glare / specular highlight reduction for RGB frames (adjustment-stack op).

Methods for wet-surface specular bounce in industrial RGB video:

- **v_rolloff** — soft knee on HSV value for bright, desaturated pixels
- **specular_inpaint** — detect highlight mask and OpenCV inpaint
- **retinex** — single- or multi-scale Retinex (illumination vs reflectance)
- **homomorphic** — spatial homomorphic filter (compress illumination)
- **dichromatic** — specular-free chromaticity + local intensity rebuild

Kept under ``color_thresholding`` (not ``color_adjustments``) so
``color_thresholding.logic`` does not import the adjustments package.
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


def _odd_ksize(value: int, lo: int = 3, hi: int = 51) -> int:
    k = int(np.clip(value, lo, hi))
    if k % 2 == 0:
        k -= 1
    return max(lo if lo % 2 else lo + 1, k)


def _specular_weight(
    hsv: np.ndarray,
    *,
    v_threshold: float,
    sat_max: float,
    soft_width: float,
) -> np.ndarray:
    """Soft [0,1] weight for specular candidates (bright + low saturation)."""
    s = hsv[..., 1].astype(np.float32)
    v = hsv[..., 2].astype(np.float32)
    thr = float(np.clip(v_threshold, 0.0, 255.0))
    s_max = float(np.clip(sat_max, 0.0, 255.0))
    width = max(float(soft_width), 1.0)

    v_w = np.clip((v - (thr - width)) / width, 0.0, 1.0)
    # Full weight when S is well below sat_max; fall off across soft_width.
    s_w = np.clip(((s_max + width) - s) / width, 0.0, 1.0)
    return v_w * s_w


def _apply_v_rolloff(img: np.ndarray, p: dict) -> np.ndarray:
    v_threshold = float(np.clip(p.get("v_threshold", 200.0), 0.0, 255.0))
    sat_max = float(np.clip(p.get("sat_max", 60.0), 0.0, 255.0))
    strength = float(np.clip(p.get("strength", 0.7), 0.0, 1.0))
    soft_width = float(np.clip(p.get("soft_width", 25.0), 1.0, 128.0))
    target_v = float(np.clip(p.get("target_v", 160.0), 0.0, 255.0))

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    w = _specular_weight(hsv, v_threshold=v_threshold, sat_max=sat_max, soft_width=soft_width)
    w = w * strength

    v = hsv[..., 2]
    # Soft knee: pull V toward target_v in highlight regions.
    hsv[..., 2] = (1.0 - w) * v + w * np.minimum(v, target_v)
    hsv[..., 2] = np.clip(hsv[..., 2], 0.0, 255.0)
    out_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _apply_specular_inpaint(img: np.ndarray, p: dict) -> np.ndarray:
    v_threshold = float(np.clip(p.get("v_threshold", 220.0), 0.0, 255.0))
    sat_max = float(np.clip(p.get("sat_max", 40.0), 0.0, 255.0))
    soft_width = float(np.clip(p.get("soft_width", 15.0), 1.0, 128.0))
    mask_threshold = float(np.clip(p.get("mask_threshold", 0.5), 0.05, 1.0))
    dilate_radius = int(np.clip(p.get("dilate_radius", 2), 0, 15))
    inpaint_radius = float(np.clip(p.get("inpaint_radius", 3.0), 1.0, 20.0))
    algo = str(p.get("inpaint_algorithm", "telea")).lower()

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    w = _specular_weight(
        hsv.astype(np.float32),
        v_threshold=v_threshold,
        sat_max=sat_max,
        soft_width=soft_width,
    )
    mask = (w >= mask_threshold).astype(np.uint8) * 255
    if dilate_radius > 0:
        k = 2 * dilate_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.dilate(mask, kernel)

    if int(np.count_nonzero(mask)) == 0:
        return img.copy()

    flags = cv2.INPAINT_TELEA if algo != "ns" else cv2.INPAINT_NS
    out_bgr = cv2.inpaint(bgr, mask, inpaint_radius, flags)
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


def _rescale_to_uint8(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float32)
    lo = float(np.min(a))
    hi = float(np.max(a))
    if hi <= lo + 1e-8:
        return np.zeros(a.shape, dtype=np.uint8)
    out = (a - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def _apply_retinex(img: np.ndarray, p: dict) -> np.ndarray:
    mode = str(p.get("retinex_mode", "multi")).lower()
    sigma = float(np.clip(p.get("sigma", 80.0), 1.0, 300.0))
    sigma2 = float(np.clip(p.get("sigma2", 15.0), 1.0, 300.0))
    sigma3 = float(np.clip(p.get("sigma3", 200.0), 1.0, 300.0))
    gain = float(np.clip(p.get("gain", 1.0), 0.1, 5.0))
    restore = float(np.clip(p.get("restore", 0.0), 0.0, 1.0))

    f = img.astype(np.float32) + 1.0
    log_i = np.log(f)

    def _ssr(sig: float) -> np.ndarray:
        k = _odd_ksize(int(max(3, round(sig * 0.5))), lo=3, hi=151)
        blur = cv2.GaussianBlur(f, (k, k), sigmaX=sig)
        return log_i - np.log(blur + 1.0)

    if mode == "single":
        r = _ssr(sigma)
    else:
        r = (_ssr(sigma2) + _ssr(sigma) + _ssr(sigma3)) / 3.0

    r = r * gain
    out = _rescale_to_uint8(r)
    if restore > 0.0:
        out = np.clip(
            (1.0 - restore) * out.astype(np.float32) + restore * img.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
    return out


def _apply_homomorphic(img: np.ndarray, p: dict) -> np.ndarray:
    sigma = float(np.clip(p.get("sigma", 30.0), 1.0, 300.0))
    gamma_l = float(np.clip(p.get("gamma_l", 0.5), 0.05, 2.0))
    gamma_h = float(np.clip(p.get("gamma_h", 1.5), 0.05, 5.0))
    restore = float(np.clip(p.get("restore", 0.0), 0.0, 1.0))

    f = img.astype(np.float32) / 255.0
    log_i = np.log1p(f)
    k = _odd_ksize(int(max(3, round(sigma * 0.5))), lo=3, hi=151)
    low = cv2.GaussianBlur(log_i, (k, k), sigmaX=sigma)
    # Spatial homomorphic: compress illumination (low), boost reflectance (high).
    filtered = gamma_l * low + gamma_h * (log_i - low)
    out_f = np.expm1(filtered)
    out = _rescale_to_uint8(out_f)
    if restore > 0.0:
        out = np.clip(
            (1.0 - restore) * out.astype(np.float32) + restore * img.astype(np.float32),
            0,
            255,
        ).astype(np.uint8)
    return out


def _apply_dichromatic(img: np.ndarray, p: dict) -> np.ndarray:
    """Specular-free chromaticity with local intensity (dichromatic-inspired)."""
    v_threshold = float(np.clip(p.get("v_threshold", 200.0), 0.0, 255.0))
    sat_max = float(np.clip(p.get("sat_max", 55.0), 0.0, 255.0))
    soft_width = float(np.clip(p.get("soft_width", 20.0), 1.0, 128.0))
    strength = float(np.clip(p.get("strength", 0.85), 0.0, 1.0))
    blur_sigma = float(np.clip(p.get("blur_sigma", 5.0), 0.5, 50.0))

    f = img.astype(np.float32)
    cmin = np.min(f, axis=-1, keepdims=True)
    sf = np.maximum(f - cmin, 0.0)
    sf_sum = np.maximum(np.sum(sf, axis=-1, keepdims=True), 1e-3)
    chroma = sf / sf_sum

    gray = np.mean(f, axis=-1, keepdims=True)
    k = _odd_ksize(int(max(3, round(blur_sigma * 2))), lo=3, hi=51)
    local = cv2.GaussianBlur(gray, (k, k), sigmaX=blur_sigma)
    if local.ndim == 2:
        local = local[..., None]
    reconstructed = chroma * (local * 3.0)

    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    w = _specular_weight(hsv, v_threshold=v_threshold, sat_max=sat_max, soft_width=soft_width)
    w = (w * strength)[..., None]

    out = (1.0 - w) * f + w * reconstructed
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_glare_reduction(
    frame_rgb: np.ndarray,
    method: str = "v_rolloff",
    params: dict | None = None,
) -> np.ndarray:
    """Apply one of several glare / specular reduction methods to an RGB frame."""
    img = _ensure_uint8_rgb(frame_rgb)
    p = dict(params or {})
    m = str(method or "v_rolloff").lower()

    if m == "v_rolloff":
        return _apply_v_rolloff(img, p)
    if m == "specular_inpaint":
        return _apply_specular_inpaint(img, p)
    if m == "retinex":
        return _apply_retinex(img, p)
    if m == "homomorphic":
        return _apply_homomorphic(img, p)
    if m == "dichromatic":
        return _apply_dichromatic(img, p)

    # Fallback: mild value roll-off.
    return _apply_v_rolloff(img, p)
