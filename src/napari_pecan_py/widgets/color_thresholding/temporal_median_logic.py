"""Temporal median background helpers (used by Adjustments / ``apply_adjustment_stack``)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import tifffile
from skimage import measure, morphology
from skimage.draw import ellipse as sk_ellipse
from skimage.filters import threshold_otsu

__all__ = [
    "MedianBackgroundConfig",
    "absdiff_scores",
    "build_ellipse_roi_mask",
    "compute_median_background",
    "evenly_spaced_frame_indices",
    "foreground_mask_from_frame",
    "run_stage_a_median_pecan_masks",
]


@dataclass
class MedianBackgroundConfig:
    """Hyperparameters for median-background pecan masking."""

    n_sample_frames: int = 120
    diff_threshold: float | None = None
    diff_quantile: float | None = None
    morph_close_radius: int = 3
    morph_open_radius: int = 2
    min_component_area_px: int = 200
    ellipse_center_rc: tuple[float, float] | None = None
    ellipse_radii_rc: tuple[float, float] | None = None
    ellipse_angle_deg: float = 0.0
    use_luminance_only: bool = False
    luminance_weights: tuple[float, float, float] = (0.299, 0.587, 0.114)


def evenly_spaced_frame_indices(n_frames: int, n_sample: int) -> np.ndarray:
    """Return ``n_sample`` indices in ``[0, n_frames - 1]`` spread across the clip."""
    n_frames = int(n_frames)
    if n_frames <= 0:
        raise ValueError("n_frames must be positive")
    n_sample = max(1, min(int(n_sample), n_frames))
    if n_sample == 1:
        return np.array([0], dtype=np.int64)
    return np.linspace(0, n_frames - 1, num=n_sample, dtype=np.int64)


def _to_float_hwc(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim == 2:
        a = a[..., np.newaxis]
    if a.ndim != 3:
        raise ValueError(f"Expected 2D or 3D frame array, got shape {a.shape}")
    if a.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Last dimension must be 1, 3, or 4 channels, got {a.shape[-1]}")
    if a.shape[-1] == 4:
        a = a[..., :3].astype(np.float32, copy=False)
    else:
        a = a.astype(np.float32, copy=False)
    return a


def compute_median_background(stack_thwc: np.ndarray) -> np.ndarray:
    """Per-pixel median over time. ``stack_thwc`` is (T, H, W, C) float or uint."""
    if stack_thwc.ndim != 4:
        raise ValueError(f"Expected (T,H,W,C) stack, got {stack_thwc.shape}")
    x = stack_thwc.astype(np.float32, copy=False)
    return np.median(x, axis=0)


def _luminance(rgb: np.ndarray, w: Sequence[float]) -> np.ndarray:
    r, g, b = w[0], w[1], w[2]
    return rgb[..., 0] * r + rgb[..., 1] * g + rgb[..., 2] * b


def absdiff_scores(
    frame_hwc: np.ndarray,
    median_hwc: np.ndarray,
    *,
    use_luminance_only: bool,
    luminance_weights: Sequence[float],
) -> np.ndarray:
    """Pixel-wise dissimilarity to the median background (higher = less like BG)."""
    f = _to_float_hwc(frame_hwc)
    m = _to_float_hwc(median_hwc)
    if f.shape != m.shape:
        raise ValueError(f"Frame shape {f.shape} != median shape {m.shape}")
    if use_luminance_only and f.shape[-1] >= 3:
        sf = _luminance(f[..., :3], luminance_weights)
        sm = _luminance(m[..., :3], luminance_weights)
        return np.abs(sf - sm)
    if f.shape[-1] == 1:
        return np.abs(f[..., 0] - m[..., 0])
    return np.mean(np.abs(f - m), axis=-1)


def build_ellipse_roi_mask(
    shape_hw: tuple[int, int],
    center_rc: tuple[float, float],
    radii_rc: tuple[float, float],
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Boolean mask (H, W) True inside the filled ellipse."""
    h, w = int(shape_hw[0]), int(shape_hw[1])
    rr_c, cc_c = float(center_rc[0]), float(center_rc[1])
    r_rad, c_rad = float(radii_rc[0]), float(radii_rc[1])
    if r_rad <= 0 or c_rad <= 0:
        raise ValueError("Ellipse radii must be positive")
    rot = math.radians(float(angle_deg))
    rr, cc = sk_ellipse(rr_c, cc_c, r_rad, c_rad, rotation=rot, shape=(h, w))
    out = np.zeros((h, w), dtype=bool)
    out[rr, cc] = True
    return out


def _roi_from_config(cfg: MedianBackgroundConfig, shape_hw: tuple[int, int]) -> np.ndarray | None:
    if cfg.ellipse_center_rc is None or cfg.ellipse_radii_rc is None:
        return None
    return build_ellipse_roi_mask(
        shape_hw,
        cfg.ellipse_center_rc,
        cfg.ellipse_radii_rc,
        cfg.ellipse_angle_deg,
    )


def _apply_morphology(binary: np.ndarray, close_r: int, open_r: int) -> np.ndarray:
    m = binary.astype(bool, copy=False)
    if close_r > 0:
        m = morphology.binary_closing(m, footprint=morphology.disk(close_r))
    if open_r > 0:
        m = morphology.binary_opening(m, footprint=morphology.disk(open_r))
    return m


def _largest_component(mask_bool: np.ndarray, *, min_area: int) -> np.ndarray:
    if not np.any(mask_bool):
        return np.zeros_like(mask_bool, dtype=np.uint8)
    lab = measure.label(mask_bool, connectivity=2)
    regions = measure.regionprops(lab)
    if not regions:
        return np.zeros_like(mask_bool, dtype=np.uint8)
    best = max(regions, key=lambda r: r.area)
    if best.area < min_area:
        return np.zeros_like(mask_bool, dtype=np.uint8)
    return (lab == best.label).astype(np.uint8)


def foreground_mask_from_frame(
    frame_hwc: np.ndarray,
    median_hwc: np.ndarray,
    cfg: MedianBackgroundConfig,
) -> tuple[np.ndarray, dict]:
    """Return uint8 mask {0,1} foreground (largest motion blob) and debug scalars."""
    scores = absdiff_scores(
        frame_hwc,
        median_hwc,
        use_luminance_only=cfg.use_luminance_only,
        luminance_weights=cfg.luminance_weights,
    )
    roi = _roi_from_config(cfg, scores.shape)
    eval_region = roi if roi is not None else np.ones_like(scores, dtype=bool)
    vals = scores[eval_region]
    if vals.size == 0:
        thr = 0.0
    elif cfg.diff_threshold is not None:
        thr = float(cfg.diff_threshold)
    elif cfg.diff_quantile is not None:
        q = float(np.clip(cfg.diff_quantile, 0.0, 1.0))
        thr = float(np.quantile(vals, q))
    else:
        v = vals.astype(np.float64, copy=False)
        if np.ptp(v) < 1e-9:
            thr = float(v.mean())
        else:
            thr = float(threshold_otsu(v))

    binary = scores > thr
    if roi is not None:
        binary &= roi
    binary = _apply_morphology(binary, cfg.morph_close_radius, cfg.morph_open_radius)
    mask = _largest_component(binary, min_area=cfg.min_component_area_px)
    info = {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "threshold": thr,
        "threshold_mode": "fixed"
        if cfg.diff_threshold is not None
        else ("quantile" if cfg.diff_quantile is not None else "otsu"),
    }
    return mask, info


def _video_frame_count_cv2(path: Path) -> int:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)
    finally:
        cap.release()


def _read_video_frame_cv2(path: Path, index: int) -> np.ndarray:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, float(index))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            raise OSError(f"Failed to read frame {index} from {path}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def _read_tiff_frame(path: Path, index: int) -> np.ndarray:
    with tifffile.TiffFile(path) as tf:
        n = len(tf.pages)
        if index < 0 or index >= n:
            raise IndexError(f"Frame index {index} out of range for {n} pages")
        arr = tf.pages[index].asarray()
    if arr.ndim == 2:
        return arr[..., np.newaxis]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unsupported TIFF slice shape {arr.shape}")


def _tiff_frame_count(path: Path) -> int:
    with tifffile.TiffFile(path) as tf:
        return len(tf.pages)


def _is_tiff(path: Path) -> bool:
    return path.suffix.lower() in {".tif", ".tiff"}


def _gather_sample_stack(
    path: Path,
    indices: Sequence[int],
    *,
    log: Callable[[str], None] | None = None,
    progress_every: int | None = None,
) -> np.ndarray:
    """Return (T, H, W, C) float32."""
    path = Path(path)
    frames: list[np.ndarray] = []
    for k, idx in enumerate(indices):
        if log and progress_every and (k % progress_every == 0):
            log(f"Loading median sample {k + 1}/{len(indices)} (frame {idx})")
        if _is_tiff(path):
            fr = _read_tiff_frame(path, int(idx))
        else:
            fr = _read_video_frame_cv2(path, int(idx))
        frames.append(_to_float_hwc(fr))
    return np.stack(frames, axis=0)


def run_stage_a_median_pecan_masks(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: MedianBackgroundConfig | None = None,
    *,
    frame_indices_to_export: Sequence[int] | None = None,
    export_median_preview: bool = True,
    export_diff_preview: bool = True,
    preview_frame_index: int | None = None,
    progress_every: int | None = 20,
    log: Callable[[str], None] | None = None,
    on_export_progress: Callable[[int, int], None] | None = None,
    cancel_check: Callable[[], bool] | None = None,
) -> dict:
    """Run steps 1–3 and write masks (+ optional previews) under ``output_dir``."""
    cfg = cfg or MedianBackgroundConfig()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if _is_tiff(input_path):
        n_total = _tiff_frame_count(input_path)
    else:
        n_total = _video_frame_count_cv2(input_path)
    if n_total <= 0:
        raise ValueError(f"Could not determine frame count for {input_path}")

    sample_idx = evenly_spaced_frame_indices(n_total, cfg.n_sample_frames)
    if log:
        log(f"{input_path.name}: {n_total} frames, {len(sample_idx)} samples for median")

    stack = _gather_sample_stack(input_path, sample_idx, log=log, progress_every=progress_every)
    median_hwc = compute_median_background(stack)

    tifffile.imwrite(output_dir / "median_background.tif", median_hwc.astype(np.float32))

    if export_median_preview:
        med = median_hwc
        if med.shape[-1] == 1:
            prev = np.repeat(med[..., 0:1], 3, axis=-1)
        elif med.shape[-1] >= 3:
            prev = med[..., :3]
        else:
            prev = med
        prev_u8 = np.clip(prev, 0, 255).astype(np.uint8)
        tifffile.imwrite(output_dir / "median_preview_rgb.tif", prev_u8)

    if frame_indices_to_export is None:
        export_idx = np.arange(n_total, dtype=np.int64)
    else:
        export_idx = np.asarray(list(frame_indices_to_export), dtype=np.int64)

    masks_dir = output_dir / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)

    meta_frames: list[dict] = []
    mid = int(preview_frame_index) if preview_frame_index is not None else int(export_idx[len(export_idx) // 2])
    mid = int(np.clip(mid, 0, n_total - 1))

    n_export = len(export_idx)
    cancelled_run = False
    for k, fi in enumerate(export_idx):
        if cancel_check and cancel_check():
            cancelled_run = True
            if log:
                log("Cancelled by user.")
            break
        fi = int(fi)
        if fi < 0 or fi >= n_total:
            continue
        if log and progress_every and k % progress_every == 0:
            log(f"Mask {k + 1}/{n_export} (frame {fi})")
        if on_export_progress:
            on_export_progress(k + 1, n_export)
        if _is_tiff(input_path):
            frame = _read_tiff_frame(input_path, fi)
        else:
            frame = _read_video_frame_cv2(input_path, fi)
        mask, info = foreground_mask_from_frame(frame, median_hwc, cfg)
        tifffile.imwrite(masks_dir / f"frame_{fi:06d}.tif", (mask * 255).astype(np.uint8))
        meta_frames.append({"frame_index": fi, **info})

        if export_diff_preview and fi == mid:
            tifffile.imwrite(
                output_dir / f"diff_scores_frame_{fi:06d}.tif",
                absdiff_scores(
                    frame,
                    median_hwc,
                    use_luminance_only=cfg.use_luminance_only,
                    luminance_weights=cfg.luminance_weights,
                ).astype(np.float32),
            )

    payload = {
        "input_path": str(input_path.resolve()),
        "n_frames": n_total,
        "sample_frame_indices": sample_idx.tolist(),
        "config": asdict(cfg),
        "frames": meta_frames,
        "cancelled": cancelled_run,
    }
    (output_dir / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if log:
        log(f"Wrote {len(meta_frames)} masks to {masks_dir}")
    return payload
