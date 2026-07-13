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


def test_frame_diff_absolute_detects_motion():
    t, h, w = 5, 16, 16
    video = np.zeros((t, h, w, 3), dtype=np.uint8)
    video[2, 4, 5, :] = 200  # appears at t=2
    stack = [
        {
            "type": "frame_diff",
            "enabled": True,
            "lag": 1,
            "mode": "absolute",
            "use_luminance": False,
            "preview_low_percentile": 0.0,
            "preview_high_percentile": 100.0,
        }
    ]
    # Frame 0 vs itself → no motion
    out0 = apply_adjustment_stack(video[0], stack, video_rgb=video, frame_index=0)
    assert out0.shape == (h, w, 3)
    assert int(np.max(out0)) == 0

    # Frame 2 vs frame 1 → bright at the moved pixel
    out2 = apply_adjustment_stack(video[2], stack, video_rgb=video, frame_index=2)
    assert int(out2[4, 5, 0]) > 0
    assert int(out2[0, 0, 0]) == 0


def test_frame_diff_signed_midgray_when_still():
    t, h, w = 3, 8, 8
    video = np.full((t, h, w, 3), 40, dtype=np.uint8)
    video[1, 2, 3, :] = 90  # brighter at t=1
    stack = [
        {
            "type": "frame_diff",
            "enabled": True,
            "lag": 1,
            "mode": "signed",
            "use_luminance": False,
        }
    ]
    still = apply_adjustment_stack(video[2], stack, video_rgb=video, frame_index=2)
    # t=2 vs t=1: most pixels still → ~128; (2,3) went 90→40 so darker than mid-gray
    assert int(still[0, 0, 0]) == 128
    assert int(still[2, 3, 0]) < 128

    brighter = apply_adjustment_stack(video[1], stack, video_rgb=video, frame_index=1)
    assert int(brighter[2, 3, 0]) > 128


def test_frame_diff_requires_video():
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    stack = [{"type": "frame_diff", "enabled": True, "lag": 1}]
    with pytest.raises(ValueError, match="frame_diff"):
        apply_adjustment_stack(frame, stack, video_rgb=None, frame_index=0)


def test_frame_diff_via_apply_adjustments_to_video():
    from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video

    t, h, w = 4, 10, 10
    video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        video[i, i, 3, :] = 180
    stack = [
        {
            "type": "frame_diff",
            "enabled": True,
            "lag": 1,
            "mode": "absolute",
            "use_luminance": True,
            "preview_low_percentile": 0.0,
            "preview_high_percentile": 100.0,
        }
    ]
    out = apply_adjustments_to_video(video, stack)
    assert out.shape == video.shape
    assert out.dtype == np.uint8
    assert int(np.max(out[0])) == 0
    assert int(out[2, 2, 3, 0]) > 0
