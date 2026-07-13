"""Tests for layer frame trimming helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from napari_pecan_py.trim_frames import (
    apply_trim_to_layer,
    frame_axis_and_count,
    trim_array,
)


def test_trim_array_inclusive_range():
    data = np.arange(5 * 2 * 3).reshape(5, 2, 3)
    out = trim_array(data, 1, 3)
    assert out.shape == (3, 2, 3)
    np.testing.assert_array_equal(out, data[1:4])


def test_trim_array_rejects_invalid_range():
    data = np.zeros((4, 8, 8), dtype=np.uint8)
    with pytest.raises(ValueError, match="Invalid frame range"):
        trim_array(data, 2, 1)
    with pytest.raises(ValueError, match="Invalid frame range"):
        trim_array(data, 0, 4)


def test_apply_trim_to_layer_shrinks_viewer_dims():
    """Regression: dims slider must shrink so trimmed-away frames disappear."""
    import napari

    data = np.arange(20 * 4 * 4 * 3, dtype=np.uint8).reshape(20, 4, 4, 3)
    viewer = napari.Viewer(show=False)
    try:
        layer = viewer.add_image(data, rgb=True, name="video")
        assert viewer.dims.nsteps[0] == 20

        new_n = apply_trim_to_layer(layer, 5, 14)
        assert new_n == 10
        assert layer.data.shape[0] == 10
        assert viewer.dims.nsteps[0] == 10
        assert layer.extent.data[1, 0] == 9
    finally:
        viewer.close()


def test_frame_axis_and_count_for_video_rgb():
    layer = SimpleNamespace(
        data=np.zeros((10, 16, 16, 3), dtype=np.uint8),
        rgb=True,
        _type_string="image",
    )
    assert frame_axis_and_count(layer) == (0, 10)


def test_frame_axis_and_count_for_labels_stack():
    layer = SimpleNamespace(
        data=np.zeros((7, 16, 16), dtype=np.uint16),
        rgb=False,
        _type_string="labels",
    )
    assert frame_axis_and_count(layer) == (0, 7)


def test_frame_axis_and_count_rejects_2d_and_points():
    rgb2d = SimpleNamespace(
        data=np.zeros((16, 16, 3), dtype=np.uint8),
        rgb=True,
        _type_string="image",
    )
    gray2d = SimpleNamespace(
        data=np.zeros((16, 16), dtype=np.uint8),
        rgb=False,
        _type_string="image",
    )
    points = SimpleNamespace(
        data=np.zeros((5, 2), dtype=float),
        rgb=False,
        _type_string="points",
    )
    assert frame_axis_and_count(rgb2d) is None
    assert frame_axis_and_count(gray2d) is None
    assert frame_axis_and_count(points) is None


def test_register_trim_frames_action_is_idempotent():
    from napari._app_model import get_app_model

    from napari_pecan_py.trim_frames import (
        _ACTION_ID,
        register_trim_frames_action,
    )
    import napari_pecan_py.trim_frames as trim_frames

    trim_frames._registered = False
    register_trim_frames_action()
    assert _ACTION_ID in get_app_model().commands
    register_trim_frames_action()
    assert _ACTION_ID in get_app_model().commands
