"""Adjustment application logic for RGB video frames."""

from __future__ import annotations

import numpy as np

from ..color_thresholding.logic import apply_adjustment_stack


def _emit_progress(cb, current: int, total: int) -> None:
    if cb is None:
        return
    try:
        cb(int(current), int(total))
    except Exception:
        # Progress feedback must never break processing.
        pass


def apply_adjustments_to_single_frame(
    frame_rgb: np.ndarray,
    adjustment_stack: list[dict],
    *,
    video_rgb: np.ndarray | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Apply the adjustment stack to one (H, W, 3) or (H, W, 4) RGB frame.

    Pass ``video_rgb`` shaped ``(T,H,W,3)`` (same spatial size as ``frame_rgb``) when
    the stack includes ``temporal_median_diff``.
    """
    arr = np.asarray(frame_rgb)
    if arr.ndim != 3 or arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB frame (H,W,3+); got shape={arr.shape}")
    return apply_adjustment_stack(
        arr[..., :3],
        adjustment_stack,
        video_rgb=video_rgb,
        frame_index=int(frame_index),
    )


def apply_adjustments_to_video(
    video_rgb: np.ndarray,
    adjustment_stack: list[dict],
    progress_callback=None,
) -> np.ndarray:
    """Apply an ordered adjustment stack to each frame.

    Parameters
    ----------
    video_rgb:
        Shapes:
        - (T, H, W, 3) or (T, H, W, 4) or
        - (H, W, 3) (treated as a single frame)
    adjustment_stack:
        List of RGB adjustment dicts with keys understood by
        `napari_pecan_py.widgets.color_thresholding.logic.apply_adjustment_stack`.

    Returns
    -------
    np.ndarray:
        uint8 adjusted frames. Shape matches input (single-frame => (H,W,3)).
    """
    arr = np.asarray(video_rgb)

    squeeze_out = False
    if arr.ndim == 3:
        # (H,W,3) => (1,H,W,3)
        arr = arr[None, ...]
        squeeze_out = True
    elif arr.ndim != 4:
        raise ValueError(f"Unsupported input shape: {arr.shape}")

    if arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB(A) in last dimension; got {arr.shape[-1]}")

    arr = arr[..., :3]

    total = int(arr.shape[0])
    _emit_progress(progress_callback, 0, total)
    out_frames: list[np.ndarray] = []
    for t in range(total):
        out_frames.append(
            apply_adjustments_to_single_frame(arr[t], adjustment_stack, video_rgb=arr, frame_index=t)
        )
        _emit_progress(progress_callback, t + 1, total)

    out = np.stack(out_frames, axis=0)
    if squeeze_out:
        return out[0]
    return out

