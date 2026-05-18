"""Temporal / mask steps in ``apply_adjustment_stack`` (Adjustments)."""

import numpy as np
import pytest

from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack


def test_temporal_median_diff_preview_rgb():
    t, h, w = 20, 32, 32
    video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        video[i, 5 + i, 8, :] = 200
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[15, 8, :] = 200
    stack = [
        {
            "type": "temporal_median_diff",
            "enabled": True,
            "n_sample_frames": 15,
            "use_luminance": False,
            "preview_low_percentile": 0.0,
            "preview_high_percentile": 100.0,
        }
    ]
    out = apply_adjustment_stack(frame, stack, video_rgb=video, frame_index=15)
    assert out.shape == (h, w, 3)
    assert out.dtype == np.uint8
    assert int(out[15, 8, 0]) > 0


def test_temporal_requires_video():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    stack = [{"type": "temporal_median_diff", "enabled": True, "n_sample_frames": 5}]
    with pytest.raises(ValueError, match="temporal_median_diff"):
        apply_adjustment_stack(frame, stack, video_rgb=None, frame_index=0)


def test_motion_mask_chain():
    h, w = 24, 24
    score = np.zeros((h, w, 3), dtype=np.uint8)
    score[8:16, 8:16, :] = 200
    stack = [
        {"type": "motion_mask_threshold", "enabled": True, "threshold_mode": "fixed", "fixed_threshold": 100.0, "use_ellipse": False},
        {"type": "mask_morphology", "enabled": True, "close_radius": 0, "open_radius": 0},
        {"type": "mask_largest_component", "enabled": True, "min_area_px": 10},
    ]
    out = apply_adjustment_stack(score, stack, video_rgb=None, frame_index=0)
    assert out.shape == (h, w, 3)
    assert int(np.max(out)) == 255
