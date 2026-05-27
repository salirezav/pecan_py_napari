"""Tests for temporal smoothing of pecan ellipse fits."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.pecan_ellipse.logic import (
    fit_ellipse_from_binary,
    fit_ellipses_volume,
    napari_vertices_to_opencv_fit,
    normalize_smooth_window,
    opencv_ellipse_to_napari_vertices,
    smooth_opencv_ellipse_sequence,
)


def _filled_ellipse_mask(
    hw: tuple[int, int],
    center: tuple[float, float],
    radii: tuple[float, float],
) -> np.ndarray:
    import cv2

    h, w = hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(
        mask,
        (int(center[0]), int(center[1])),
        (int(radii[0]), int(radii[1])),
        0.0,
        0,
        360,
        255,
        -1,
    )
    return mask


def test_normalize_smooth_window_forces_odd():
    assert normalize_smooth_window(4) == 5
    assert normalize_smooth_window(3) == 3
    assert normalize_smooth_window(2) == 3


def test_smooth_reduces_single_frame_size_spike():
    base = ((50.0, 50.0), (40.0, 30.0), 10.0)
    spike = ((50.0, 50.0), (80.0, 70.0), 10.0)
    fits = [base, base, spike, base, base]
    smoothed = smooth_opencv_ellipse_sequence(fits, window=5)
    assert smoothed[2] is not None
    _, size_spike_raw, _ = spike
    _, size_spike_smooth, _ = smoothed[2]  # type: ignore[misc]
    assert size_spike_smooth[0] < size_spike_raw[0]
    assert size_spike_smooth[1] < size_spike_raw[1]


def test_smooth_preserves_failed_frames():
    base = ((50.0, 50.0), (40.0, 30.0), 10.0)
    fits = [base, None, base]
    smoothed = smooth_opencv_ellipse_sequence(fits, window=3)
    assert smoothed[1] is None
    assert smoothed[0] is not None
    assert smoothed[2] is not None


def test_vertices_roundtrip():
    fit = ((100.0, 80.0), (60.0, 40.0), 25.0)
    verts = opencv_ellipse_to_napari_vertices(fit, time_index=None)
    recovered = napari_vertices_to_opencv_fit(verts)
    assert recovered is not None
    (c0, _), (s0, _), a0 = fit
    (c1, _), (s1, _), a1 = recovered
    np.testing.assert_allclose(c0, c1, rtol=0, atol=2.0)
    np.testing.assert_allclose(s0, s1, rtol=0, atol=2.0)
    assert abs((a0 - a1 + 180) % 180 - 90) < 91  # 180° ambiguity


def test_fit_ellipses_volume_smoothing_on_stack():
    h, w = 120, 120
    frames = []
    for t in range(5):
        r = 20 if t != 2 else 55
        frames.append(_filled_ellipse_mask((h, w), (60, 60), (r, 18)))
    volume = np.stack(frames, axis=0)

    raw_sizes = []
    for t in range(5):
        fit = fit_ellipse_from_binary(frames[t], largest_only=True)
        assert fit is not None
        raw_sizes.append(fit[1][0])

    smooth_verts = fit_ellipses_volume(
        volume, label_id=None, largest_only=True, temporal_smooth=True, smooth_window=5
    )
    assert len(smooth_verts) == 5
    smooth_mid = napari_vertices_to_opencv_fit(smooth_verts[2])
    assert smooth_mid is not None
    assert smooth_mid[1][0] < raw_sizes[2]
