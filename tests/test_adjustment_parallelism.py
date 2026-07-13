"""Tests for parallel adjustment stack segmentation and apply."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video
from napari_pecan_py.widgets.color_adjustments.parallelism import (
    FRAME_PARALLEL_TYPES,
    TEMPORAL_BARRIER_TYPES,
    is_parallelizable,
    split_stack_into_segments,
    stamp_parallelizable_flags,
)
from napari_pecan_py.widgets.color_thresholding.defaults import (
    DEFAULT_BRIGHTNESS_CONTRAST,
    DEFAULT_CURVES,
    DEFAULT_LEVELS,
    DEFAULT_NORMALIZATION,
    DEFAULT_SURFACE_BLUR,
    DEFAULT_TEMPORAL_MEDIAN_DIFF,
)


def _enabled(item: dict) -> dict:
    out = dict(item)
    out["enabled"] = True
    return out


def test_split_all_parallel():
    stack = [
        _enabled(DEFAULT_SURFACE_BLUR),
        _enabled(DEFAULT_NORMALIZATION),
        _enabled(DEFAULT_CURVES),
    ]
    segments = split_stack_into_segments(stack)
    assert segments == [("parallel", stack)]


def test_split_barrier_in_middle():
    blur = _enabled(DEFAULT_SURFACE_BLUR)
    norm = _enabled(DEFAULT_NORMALIZATION)
    temporal = _enabled(DEFAULT_TEMPORAL_MEDIAN_DIFF)
    curves = _enabled(DEFAULT_CURVES)
    stack = [blur, norm, temporal, blur, curves]
    segments = split_stack_into_segments(stack)
    assert segments == [
        ("parallel", [blur, norm]),
        ("barrier", [temporal]),
        ("parallel", [blur, curves]),
    ]


def test_split_adjacent_barriers():
    temporal_a = _enabled({**DEFAULT_TEMPORAL_MEDIAN_DIFF, "n_sample_frames": 5})
    temporal_b = _enabled({**DEFAULT_TEMPORAL_MEDIAN_DIFF, "n_sample_frames": 8})
    segments = split_stack_into_segments([temporal_a, temporal_b])
    assert segments == [
        ("barrier", [temporal_a]),
        ("barrier", [temporal_b]),
    ]


def test_split_ignores_disabled():
    blur = _enabled(DEFAULT_SURFACE_BLUR)
    disabled = {**DEFAULT_TEMPORAL_MEDIAN_DIFF, "enabled": False}
    curves = _enabled(DEFAULT_CURVES)
    segments = split_stack_into_segments([blur, disabled, curves])
    assert segments == [("parallel", [blur, curves])]


def test_stamp_parallelizable_flags():
    stack = [
        _enabled(DEFAULT_SURFACE_BLUR),
        _enabled(DEFAULT_LEVELS),
        _enabled(DEFAULT_CURVES),
        _enabled(DEFAULT_NORMALIZATION),
        _enabled(DEFAULT_TEMPORAL_MEDIAN_DIFF),
    ]
    stamped = stamp_parallelizable_flags(stack)
    assert stamped[0]["parallelizable"] is True
    assert stamped[1]["parallelizable"] is True
    assert stamped[2]["parallelizable"] is True
    assert stamped[3]["parallelizable"] is True
    assert stamped[4]["parallelizable"] is False
    assert "parallelizable" not in stack[0]


def test_frame_parallel_types_are_parallelizable():
    for typ in sorted(FRAME_PARALLEL_TYPES):
        assert is_parallelizable({"type": typ, "enabled": True}) is True


def test_temporal_barrier_types_are_not_parallelizable():
    for typ in sorted(TEMPORAL_BARRIER_TYPES):
        assert is_parallelizable({"type": typ, "enabled": True}) is False


def test_is_parallelizable_unknown_is_barrier():
    assert is_parallelizable({"type": "future_op", "enabled": True}) is False


def test_parallel_matches_sequential_surface_blur_curves():
    rng = np.random.default_rng(0)
    t, h, w = 8, 24, 24
    video = rng.integers(0, 256, size=(t, h, w, 3), dtype=np.uint8)
    stack = [
        {
            **_enabled(DEFAULT_SURFACE_BLUR),
            "radius": 3,
            "threshold": 20,
        },
        _enabled(DEFAULT_CURVES),
    ]
    seq = apply_adjustments_to_video(video, stack, max_workers=1)
    par = apply_adjustments_to_video(video, stack, max_workers=4)
    np.testing.assert_array_equal(seq, par)


def test_parallel_matches_sequential_levels_normalization_curves():
    rng = np.random.default_rng(1)
    t, h, w = 10, 20, 20
    video = rng.integers(20, 240, size=(t, h, w, 3), dtype=np.uint8)
    stack = [
        _enabled(DEFAULT_BRIGHTNESS_CONTRAST),
        _enabled(DEFAULT_LEVELS),
        _enabled(DEFAULT_NORMALIZATION),
        _enabled(DEFAULT_CURVES),
    ]
    seq = apply_adjustments_to_video(video, stack, max_workers=1)
    par = apply_adjustments_to_video(video, stack, max_workers=4)
    np.testing.assert_array_equal(seq, par)


def test_barrier_stack_parallel_matches_sequential():
    t, h, w = 12, 20, 20
    video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        video[i, 4 + (i % 3), 6, :] = 180
        video[i, 10, 10, :] = 60 + i * 5

    stack = [
        {
            **_enabled(DEFAULT_SURFACE_BLUR),
            "radius": 2,
            "threshold": 15,
        },
        _enabled(DEFAULT_NORMALIZATION),
        {
            **_enabled(DEFAULT_TEMPORAL_MEDIAN_DIFF),
            "n_sample_frames": 10,
            "use_luminance": False,
            "preview_low_percentile": 0.0,
            "preview_high_percentile": 100.0,
        },
        _enabled(DEFAULT_CURVES),
    ]
    seq = apply_adjustments_to_video(video, stack, max_workers=1)
    par = apply_adjustments_to_video(video, stack, max_workers=3)
    np.testing.assert_array_equal(seq, par)


def test_barrier_median_uses_post_blur_volume():
    """Temporal median after blur should differ from median on the raw video."""
    t, h, w = 10, 16, 16
    video = np.zeros((t, h, w, 3), dtype=np.uint8)
    for i in range(t):
        video[i, 5 + (i % 2), 7, :] = 200

    blur_then_median = [
        {
            **_enabled(DEFAULT_SURFACE_BLUR),
            "radius": 5,
            "threshold": 30,
        },
        {
            **_enabled(DEFAULT_TEMPORAL_MEDIAN_DIFF),
            "n_sample_frames": 8,
            "use_luminance": False,
            "preview_low_percentile": 0.0,
            "preview_high_percentile": 100.0,
        },
    ]
    median_only = [_enabled(DEFAULT_TEMPORAL_MEDIAN_DIFF)]

    out_blur_median = apply_adjustments_to_video(video, blur_then_median, max_workers=2)
    out_median_only = apply_adjustments_to_video(video, median_only, max_workers=1)
    assert not np.array_equal(out_blur_median, out_median_only)


def test_cancel_callback_raises():
    video = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    stack = [_enabled(DEFAULT_SURFACE_BLUR)]
    cancelled = False

    def cancel() -> bool:
        return cancelled

    import pytest

    cancelled = True
    with pytest.raises(InterruptedError, match="cancelled"):
        apply_adjustments_to_video(video, stack, max_workers=2, cancel_callback=cancel)
