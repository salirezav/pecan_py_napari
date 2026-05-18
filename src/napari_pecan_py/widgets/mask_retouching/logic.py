"""Mask retouching operations: morphological cleanup, area filtering, hole filling."""

from __future__ import annotations

import cv2
import numpy as np


def morphological_close(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Close small gaps between mask regions (dilate then erode)."""
    if kernel_size < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def morphological_open(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Remove small noise from the mask (erode then dilate)."""
    if kernel_size < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)


def dilate(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size < 1 or iterations < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.dilate(mask, k, iterations=iterations)


def erode(mask: np.ndarray, kernel_size: int, iterations: int) -> np.ndarray:
    if kernel_size < 1 or iterations < 1:
        return mask
    k = np.ones((kernel_size, kernel_size), np.uint8)
    return cv2.erode(mask, k, iterations=iterations)


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


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill internal holes that do not touch the image border."""
    binary = (mask > 0).astype(np.uint8) * 255

    # Flood-fill the background from the image border. Any remaining 0-valued
    # pixels after this step are enclosed holes.
    flood = binary.copy()
    h, w = flood.shape[:2]
    flood_mask = np.zeros((h + 2, w + 2), np.uint8)  # required by cv2.floodFill
    cv2.floodFill(flood, flood_mask, (0, 0), 255)

    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(binary, holes)
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
    do_keep_largest: bool = False,
    smooth_size: int = 0,
) -> np.ndarray:
    """Run the full retouching pipeline on a single 2-D mask frame.

    The order follows pecan_py conventions:
      close -> open -> dilate -> erode -> remove small -> fill holes
      -> keep largest -> smooth
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
        out = fill_holes(out)
    if do_keep_largest:
        out = keep_largest_contour(out)
    if smooth_size >= 3:
        out = smooth_boundary(out, smooth_size)
    return out
