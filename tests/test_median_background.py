"""Tests for median-background pecan masking (Stage A steps 1–3)."""

import numpy as np
import pytest

from napari_pecan_py.widgets.color_thresholding.temporal_median_logic import (
    MedianBackgroundConfig,
    compute_median_background,
    evenly_spaced_frame_indices,
    foreground_mask_from_frame,
)


def test_evenly_spaced_indices():
    x = evenly_spaced_frame_indices(100, 5)
    assert len(x) == 5
    assert x[0] == 0 and x[-1] == 99


def test_median_static_background_moving_blob():
    """Synthetic: black background; a bright 1-pixel dot visits unique pixels — median stays 0."""
    h, w = 64, 64
    t = 40
    frames = np.zeros((t, h, w, 3), dtype=np.float32)
    for i in range(t):
        frames[i, 5 + i, 10 + i, :] = 255.0  # diagonal crawl, no pixel repeated

    median = compute_median_background(frames)
    assert median.shape == (h, w, 3)
    assert float(np.max(median)) < 1.0

    cfg = MedianBackgroundConfig(
        n_sample_frames=t,
        diff_threshold=30.0,
        morph_close_radius=0,
        morph_open_radius=0,
        min_component_area_px=10,
    )
    test_frame = np.zeros((h, w, 3), dtype=np.float32)
    test_frame[30:38, 25:33, :] = 255.0
    mask, info = foreground_mask_from_frame(test_frame, median, cfg)
    assert mask.sum() > 0
    assert info["threshold"] == 30.0


def test_ellipse_roi_suppresses_outside():
    h, w = 80, 80
    median = np.zeros((h, w, 3), dtype=np.float32)
    frame = np.zeros((h, w, 3), dtype=np.float32)
    # Strong motion top-left and bottom-right; only top-left is inside ellipse centered upper.
    frame[:20, :20, :] = 200.0
    frame[60:78, 60:78, :] = 200.0

    cfg = MedianBackgroundConfig(
        diff_threshold=50.0,
        morph_close_radius=0,
        morph_open_radius=0,
        min_component_area_px=5,
        ellipse_center_rc=(15.0, 15.0),
        ellipse_radii_rc=(25.0, 25.0),
        ellipse_angle_deg=0.0,
    )
    mask, _ = foreground_mask_from_frame(frame, median, cfg)
    assert not np.any(mask[60:78, 60:78]), "bottom-right blob should be outside ellipse ROI"
