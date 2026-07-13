"""Mask retouching operations: morphological cleanup, area filtering, hole filling."""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


def _binary_u8(mask: np.ndarray) -> np.ndarray:
    """Foreground as uint8 {0,255} for OpenCV ops that reject int Labels dtypes."""
    return (np.asarray(mask) > 0).astype(np.uint8) * 255


def _apply_binary_result(mask: np.ndarray, binary_u8: np.ndarray) -> np.ndarray:
    """Map a binary morph result back onto *mask*, preserving label IDs where possible."""
    fg = binary_u8 > 0
    out = np.zeros_like(mask)
    keep = fg & (mask > 0)
    out[keep] = mask[keep]
    # Pixels added by dilate/close: mark as foreground label 1.
    added = fg & (mask == 0)
    if np.any(added):
        out[added] = np.array(1, dtype=mask.dtype)
    return out


def morphological_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Close small gaps between mask regions (dilate then erode)."""
    if kernel_size < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    closed = cv2.morphologyEx(_binary_u8(mask), cv2.MORPH_CLOSE, k)
    return _apply_binary_result(mask, closed)


def morphological_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Remove small noise from the mask (erode then dilate)."""
    if kernel_size < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(_binary_u8(mask), cv2.MORPH_OPEN, k)
    return _apply_binary_result(mask, opened)


def dilate(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size < 1 or iterations < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = cv2.dilate(_binary_u8(mask), k, iterations=iterations)
    return _apply_binary_result(mask, dilated)


def erode(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size < 1 or iterations < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = cv2.erode(_binary_u8(mask), k, iterations=iterations)
    return _apply_binary_result(mask, eroded)


def remove_small_regions(mask: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected components whose area is below *min_area*."""
    if min_area <= 0:
        return mask
    binary = (mask > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = mask[labels == i]
    return out


def fill_holes(
    mask: np.ndarray,
    *,
    min_area: int = 0,
    max_area: int = 0,
) -> np.ndarray:
    """Fill internal holes that do not touch the image border.

    Parameters
    ----------
    min_area:
        Only fill holes with area >= this (px). ``0`` = no lower bound.
    max_area:
        Only fill holes with area <= this (px). ``0`` = no upper bound.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    h, w = binary.shape[:2]

    # Pad with background so every border-connected region is reachable even
    # when the mask touches two adjacent frame edges (e.g. a corner pocket).
    padded = np.zeros((h + 2, w + 2), np.uint8)
    padded[1 : h + 1, 1 : w + 1] = binary
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(padded, flood_mask, (0, 0), 255)

    flood = padded[1 : h + 1, 1 : w + 1]
    holes = cv2.bitwise_not(flood)
    # Holes are background pixels that are not reachable from the border.
    # Intersect with the original background so we only consider true holes.
    holes = cv2.bitwise_and(holes, cv2.bitwise_not(binary))
    if not holes.any():
        return mask.copy()

    lo = max(0, int(min_area))
    hi = int(max_area)
    if lo <= 0 and hi <= 0:
        filled = cv2.bitwise_or(binary, holes)
        return np.where(filled > 0, np.maximum(mask, 1), 0).astype(mask.dtype)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    fill_mask = np.zeros_like(holes)
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < lo:
            continue
        if hi > 0 and area > hi:
            continue
        fill_mask[labels == i] = 255

    if not fill_mask.any():
        return mask.copy()

    filled = cv2.bitwise_or(binary, fill_mask)
    return np.where(filled > 0, np.maximum(mask, 1), 0).astype(mask.dtype)


def keep_largest_contour(mask: np.ndarray) -> np.ndarray:
    """Keep only the connected component with the largest area."""
    binary = (mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(binary)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return np.where(out > 0, mask, 0).astype(mask.dtype)


def smooth_boundary(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Gaussian-blur the mask edges for smoother contours, then re-threshold."""
    if kernel_size < 3:
        return mask
    ks = kernel_size | 1  # ensure odd
    binary = (mask > 0).astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(binary, (ks, ks), 0)
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    return np.where(smoothed > 0, np.maximum(mask, 1), 0).astype(mask.dtype)


def watershed_split(
    mask: np.ndarray,
    *,
    min_distance: int = 15,
    min_peak_fraction: float = 0.25,
) -> np.ndarray:
    """Split touching foreground blobs into distinct instance labels.

    Uses a distance-transform + local-maxima marker watershed. Intended for
    binary (or single-label) masks of touching elliptical objects.

    Parameters
    ----------
    mask:
        2-D label/binary mask. Non-zero pixels are foreground.
    min_distance:
        Minimum spacing between seed peaks (roughly object radius in px).
        Larger values merge more aggressively; smaller values over-segment.
    min_peak_fraction:
        Peaks must reach at least this fraction of the max distance value.
        Suppresses weak seeds from noise / shallow bumps.
    """
    if mask.ndim != 2:
        raise ValueError(f"watershed_split expects a 2-D mask, got shape {mask.shape}")

    binary = (mask > 0).astype(np.uint8)
    if not binary.any():
        return np.zeros_like(mask)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    peak_min = max(1, int(min_distance))
    threshold_abs = float(min_peak_fraction) * float(dist.max()) if dist.max() > 0 else 0.0

    coords = peak_local_max(
        dist,
        min_distance=peak_min,
        threshold_abs=threshold_abs if threshold_abs > 0 else None,
        labels=binary,
    )
    if coords.size == 0:
        # No reliable peaks — fall back to connected components.
        n_labels, labels = cv2.connectedComponents(binary, connectivity=8)
        return labels.astype(np.int32) if n_labels > 1 else mask.astype(np.int32)

    markers = np.zeros(dist.shape, dtype=np.int32)
    markers[tuple(coords.T)] = np.arange(1, len(coords) + 1, dtype=np.int32)

    labels = watershed(-dist, markers, mask=binary.astype(bool))
    return labels.astype(np.int32)


def apply_retouching_pipeline(
    mask: np.ndarray,
    *,
    close_size: int = 0,
    open_size: int = 0,
    dilate_size: int = 0,
    dilate_iter: int = 1,
    erode_size: int = 0,
    erode_iter: int = 1,
    min_area: int = 0,
    do_fill_holes: bool = False,
    fill_holes_min_area: int = 0,
    fill_holes_max_area: int = 0,
    do_watershed_split: bool = False,
    watershed_min_distance: int = 15,
    do_keep_largest: bool = False,
    smooth_size: int = 0,
) -> np.ndarray:
    """Run the full retouching pipeline on a single 2-D mask frame.

    The order follows pecan_py conventions:
      close -> open -> dilate -> erode -> remove small -> fill holes
      -> watershed split -> keep largest -> smooth

    Every step is frame-local (no temporal context), so volumes can be processed
    with :func:`apply_retouching_to_volume` in parallel across time.
    """
    out = mask.copy()
    if close_size >= 3:
        out = morphological_close(out, close_size)
    if open_size >= 3:
        out = morphological_open(out, open_size)
    if dilate_size >= 3:
        out = dilate(out, dilate_size, dilate_iter)
    if erode_size >= 3:
        out = erode(out, erode_size, erode_iter)
    if min_area > 0:
        out = remove_small_regions(out, min_area)
    if do_fill_holes:
        out = fill_holes(
            out,
            min_area=fill_holes_min_area,
            max_area=fill_holes_max_area,
        )
    if do_watershed_split:
        out = watershed_split(out, min_distance=watershed_min_distance)
    if do_keep_largest:
        out = keep_largest_contour(out)
    if smooth_size >= 3:
        out = smooth_boundary(out, smooth_size)
    return out


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


def _check_cancel(cancel_callback) -> None:
    if cancel_callback is None:
        return
    try:
        if bool(cancel_callback()):
            raise InterruptedError("Mask retouching cancelled by user.")
    except InterruptedError:
        raise
    except Exception:
        return


def _emit_progress(progress_callback, current: int, total: int) -> None:
    if progress_callback is None:
        return
    try:
        progress_callback(int(current), int(total))
    except Exception:
        pass


def apply_retouching_to_volume(
    volume: np.ndarray,
    *,
    close_size: int = 0,
    open_size: int = 0,
    dilate_size: int = 0,
    dilate_iter: int = 1,
    erode_size: int = 0,
    erode_iter: int = 1,
    min_area: int = 0,
    do_fill_holes: bool = False,
    fill_holes_min_area: int = 0,
    fill_holes_max_area: int = 0,
    do_watershed_split: bool = False,
    watershed_min_distance: int = 15,
    do_keep_largest: bool = False,
    smooth_size: int = 0,
    max_workers: int | None = None,
    progress_callback=None,
    cancel_callback=None,
) -> np.ndarray:
    """Apply retouching to a 2-D mask or (T,H,W) volume.

    All retouching ops are frame-independent, so multi-frame volumes are split
    across a thread pool (same pattern as Adjustments). Use ``max_workers=1``
    to force sequential processing.
    """
    arr = np.asarray(volume)
    params = dict(
        close_size=close_size,
        open_size=open_size,
        dilate_size=dilate_size,
        dilate_iter=dilate_iter,
        erode_size=erode_size,
        erode_iter=erode_iter,
        min_area=min_area,
        do_fill_holes=do_fill_holes,
        fill_holes_min_area=fill_holes_min_area,
        fill_holes_max_area=fill_holes_max_area,
        do_watershed_split=do_watershed_split,
        watershed_min_distance=watershed_min_distance,
        do_keep_largest=do_keep_largest,
        smooth_size=smooth_size,
    )

    if arr.ndim == 2:
        _emit_progress(progress_callback, 0, 1)
        _check_cancel(cancel_callback)
        out = apply_retouching_pipeline(arr, **params)
        _emit_progress(progress_callback, 1, 1)
        return out
    if arr.ndim != 3:
        raise ValueError(f"Expected mask shape (H,W) or (T,H,W); got {arr.shape}")

    total = int(arr.shape[0])
    if max_workers is None:
        max_workers = min(os.cpu_count() or 4, total)
    max_workers = max(1, int(max_workers))

    out_dtype = np.int32 if do_watershed_split else arr.dtype
    _emit_progress(progress_callback, 0, total)
    _check_cancel(cancel_callback)

    if max_workers <= 1 or total <= 1:
        frames = []
        for t in range(total):
            _check_cancel(cancel_callback)
            frames.append(apply_retouching_pipeline(arr[t], **params))
            _emit_progress(progress_callback, t + 1, total)
        return np.stack(frames, axis=0).astype(out_dtype, copy=False)

    ranges = _chunk_ranges(total, max_workers)
    out = np.empty(arr.shape, dtype=out_dtype)
    completed = 0
    lock = threading.Lock()

    def _process_range(start: int, end: int) -> tuple[int, int, np.ndarray]:
        chunk = np.empty((end - start, *arr.shape[1:]), dtype=out_dtype)
        for local_t, global_t in enumerate(range(start, end)):
            chunk[local_t] = apply_retouching_pipeline(arr[global_t], **params)
        return start, end, chunk

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        futures = [executor.submit(_process_range, start, end) for start, end in ranges]
        for fut in as_completed(futures):
            _check_cancel(cancel_callback)
            start, end, chunk = fut.result()
            out[start:end] = chunk
            with lock:
                completed += end - start
                _emit_progress(progress_callback, completed, total)

    return out
