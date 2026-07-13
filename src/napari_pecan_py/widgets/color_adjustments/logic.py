"""Adjustment application logic for RGB video frames."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..color_thresholding.logic import apply_adjustment_stack
from .parallelism import (
    segment_needs_video_context,
    split_stack_into_segments,
)


def _emit_progress(cb, current: int, total: int) -> None:
    if cb is None:
        return
    try:
        cb(int(current), int(total))
    except Exception:
        # Progress feedback must never break processing.
        pass


def materialize_rgb_volume(data: Any, *, progress_callback=None) -> np.ndarray:
    """Load a lazy or in-memory layer into uint8 RGB volume ``(T,H,W,3)`` or ``(H,W,3)``."""
    shape = getattr(data, "shape", None)
    if shape is None:
        raise ValueError("Source does not expose shape")
    if len(shape) == 3:
        return np.asarray(data, dtype=np.uint8)[..., :3]
    if len(shape) != 4:
        raise ValueError(f"Unsupported source shape: {shape}")

    total = int(shape[0])
    _emit_progress(progress_callback, 0, total)
    vol = np.empty((total, int(shape[1]), int(shape[2]), 3), dtype=np.uint8)
    for t in range(total):
        vol[t] = np.asarray(data[t], dtype=np.uint8)[..., :3]
        _emit_progress(progress_callback, t + 1, total)
    return vol


def _check_cancel(cancel_callback) -> None:
    if cancel_callback is None:
        return
    try:
        if bool(cancel_callback()):
            raise InterruptedError("Adjustment apply cancelled by user.")
    except InterruptedError:
        raise
    except Exception:
        return


def apply_adjustments_to_single_frame(
    frame_rgb: np.ndarray,
    adjustment_stack: list[dict],
    *,
    video_rgb: np.ndarray | None = None,
    frame_index: int = 0,
) -> np.ndarray:
    """Apply the adjustment stack to one (H, W, 3) or (H, W, 4) RGB frame.

    Pass ``video_rgb`` shaped ``(T,H,W,3)`` (same spatial size as ``frame_rgb``) when
    the stack includes ``temporal_median_diff`` or ``frame_diff``.
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


def _apply_segment_sequential(
    volume: np.ndarray,
    segment_stack: list[dict],
    *,
    progress_callback=None,
    cancel_callback=None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> np.ndarray:
    """Apply one segment to every frame sequentially."""
    total = int(volume.shape[0])
    report_total = int(progress_total) if progress_total is not None else total
    needs_video = segment_needs_video_context(segment_stack)
    out_frames: list[np.ndarray] = []
    for t in range(total):
        _check_cancel(cancel_callback)
        video_ctx = volume if needs_video else None
        out_frames.append(
            apply_adjustments_to_single_frame(
                volume[t],
                segment_stack,
                video_rgb=video_ctx,
                frame_index=t,
            )
        )
        _emit_progress(progress_callback, progress_offset + t + 1, report_total)
    return np.stack(out_frames, axis=0)


def _chunk_ranges(total: int, n_workers: int) -> list[tuple[int, int]]:
    """Contiguous frame index ranges for parallel workers."""
    n_workers = max(1, min(int(n_workers), int(total)))
    base, rem = divmod(total, n_workers)
    ranges: list[tuple[int, int]] = []
    start = 0
    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        if size <= 0:
            continue
        ranges.append((start, start + size))
        start += size
    return ranges


def _apply_segment_parallel(
    volume: np.ndarray,
    segment_stack: list[dict],
    *,
    max_workers: int,
    progress_callback=None,
    cancel_callback=None,
    progress_offset: int = 0,
    progress_total: int | None = None,
) -> np.ndarray:
    """Apply a frame-parallel segment using a thread pool."""
    total = int(volume.shape[0])
    report_total = int(progress_total) if progress_total is not None else total
    needs_video = segment_needs_video_context(segment_stack)
    video_ctx = volume if needs_video else None

    ranges = _chunk_ranges(total, max_workers)
    if len(ranges) <= 1:
        return _apply_segment_sequential(
            volume,
            segment_stack,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            progress_offset=progress_offset,
            progress_total=report_total,
        )

    out = np.empty_like(volume)
    completed = 0
    lock = threading.Lock()

    def _process_range(start: int, end: int) -> tuple[int, int, np.ndarray]:
        chunk = np.empty((end - start, *volume.shape[1:]), dtype=volume.dtype)
        for local_t, global_t in enumerate(range(start, end)):
            chunk[local_t] = apply_adjustments_to_single_frame(
                volume[global_t],
                segment_stack,
                video_rgb=video_ctx,
                frame_index=global_t,
            )
        return start, end, chunk

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [executor.submit(_process_range, start, end) for start, end in ranges]
        for fut in as_completed(futures):
            _check_cancel(cancel_callback)
            start, end, chunk = fut.result()
            out[start:end] = chunk
            with lock:
                completed += end - start
                _emit_progress(progress_callback, progress_offset + completed, report_total)

    return out


def apply_adjustments_to_video(
    video_rgb: np.ndarray,
    adjustment_stack: list[dict],
    progress_callback=None,
    *,
    max_workers: int | None = None,
    cancel_callback=None,
) -> np.ndarray:
    """Apply an ordered adjustment stack to each frame.

    Frame-independent segments run in parallel across worker threads. Temporal
    barrier ops (e.g. ``temporal_median_diff``) split the stack: parallel work
    runs before and after each barrier on the current working volume.

    Parameters
    ----------
    video_rgb:
        Shapes:
        - (T, H, W, 3) or (T, H, W, 4) or
        - (H, W, 3) (treated as a single frame)
    adjustment_stack:
        List of RGB adjustment dicts with keys understood by
        `napari_pecan_py.widgets.color_thresholding.logic.apply_adjustment_stack`.
    max_workers:
        Thread pool size for parallel segments. Defaults to
        ``min(cpu_count, T)``. Use ``1`` to force sequential processing.
    cancel_callback:
        Optional callable returning True when processing should stop.

    Returns
    -------
    np.ndarray:
        uint8 adjusted frames. Shape matches input (single-frame => (H,W,3)).
    """
    arr = np.asarray(video_rgb)

    squeeze_out = False
    if arr.ndim == 3:
        arr = arr[None, ...]
        squeeze_out = True
    elif arr.ndim != 4:
        raise ValueError(f"Unsupported input shape: {arr.shape}")

    if arr.shape[-1] < 3:
        raise ValueError(f"Expected RGB(A) in last dimension; got {arr.shape[-1]}")

    arr = arr[..., :3]
    total = int(arr.shape[0])

    segments = split_stack_into_segments(adjustment_stack)
    if not segments:
        out = arr.copy()
        _emit_progress(progress_callback, total, total)
        if squeeze_out:
            return out[0]
        return out

    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, total)
    max_workers = int(max_workers)

    progress_total = total * len(segments)
    _emit_progress(progress_callback, 0, progress_total)
    _check_cancel(cancel_callback)

    working = arr
    completed_frames = 0

    for kind, segment_stack in segments:
        _check_cancel(cancel_callback)
        if kind == "barrier":
            working = _apply_segment_sequential(
                working,
                segment_stack,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                progress_offset=completed_frames,
                progress_total=progress_total,
            )
        elif max_workers <= 1 or total <= 1:
            working = _apply_segment_sequential(
                working,
                segment_stack,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                progress_offset=completed_frames,
                progress_total=progress_total,
            )
        else:
            working = _apply_segment_parallel(
                working,
                segment_stack,
                max_workers=max_workers,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                progress_offset=completed_frames,
                progress_total=progress_total,
            )
        completed_frames += total

    if squeeze_out:
        return working[0]
    return working
