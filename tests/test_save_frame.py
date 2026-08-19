"""Tests for native-resolution frame save / visible composite."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_pecan_py.save_frame import (
    FRAME_SAVE_SUFFIX,
    VISIBLE_FRAME_SAVE_SUFFIX,
    _LAYER_ACTION_ID,
    _LAYER_ACTION_ID_AS,
    _VISIBLE_ACTION_ID,
    _VISIBLE_ACTION_ID_AS,
    composite_visible_frame,
    default_frame_export_path,
    layer_rgb_at_frame,
    register_save_frame_actions,
    write_rgb_image,
)


def test_default_frame_export_path(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    assert (
        default_frame_export_path(src, 7)
        == tmp_path / f"clip{FRAME_SAVE_SUFFIX}000007.png"
    )
    assert (
        default_frame_export_path(src, 7, layer_name="mask")
        == tmp_path / f"clip{FRAME_SAVE_SUFFIX}000007 - mask.png"
    )
    assert (
        default_frame_export_path(src, 3, suffix=VISIBLE_FRAME_SAVE_SUFFIX)
        == tmp_path / f"clip{VISIBLE_FRAME_SAVE_SUFFIX}000003.png"
    )


def test_layer_rgb_at_frame_image_and_labels():
    video = np.zeros((4, 8, 10, 3), dtype=np.uint8)
    video[2, :, :] = (10, 20, 30)
    img = SimpleNamespace(
        data=video,
        rgb=True,
        _type_string="image",
        contrast_limits=None,
        name="vid",
    )
    rgb = layer_rgb_at_frame(img, 2)
    assert rgb.shape == (8, 10, 3)
    assert np.all(rgb == (10, 20, 30))

    labels = np.zeros((4, 8, 10), dtype=np.uint16)
    labels[1, 2:5, 2:5] = 3
    labs = SimpleNamespace(
        data=labels,
        _type_string="labels",
        name="labs",
        get_color=None,
    )
    lab_rgb = layer_rgb_at_frame(labs, 1)
    assert lab_rgb.shape == (8, 10, 3)
    assert np.all(lab_rgb[0, 0] == 0)
    assert not np.all(lab_rgb[3, 3] == 0)


def test_write_rgb_image_png(tmp_path: Path):
    rgb = np.zeros((16, 20, 3), dtype=np.uint8)
    rgb[:, :] = (255, 128, 0)
    out = write_rgb_image(rgb, tmp_path / "frame")
    assert out.suffix == ".png"
    assert out.exists()


def test_composite_visible_frame_respects_opacity_and_size():
    import napari

    viewer = napari.Viewer(show=False)
    try:
        video = np.zeros((3, 40, 50, 3), dtype=np.uint8)
        video[:, :, :] = (0, 0, 100)
        viewer.add_image(
            video,
            rgb=True,
            name="vid",
            metadata={"source_path": "/tmp/clip.mp4"},
        )
        labs = np.zeros((3, 40, 50), dtype=np.uint16)
        labs[:, 10:30, 10:30] = 1
        layer = viewer.add_labels(labs, name="mask")
        layer.opacity = 0.5
        viewer.dims.set_current_step(0, 1)

        out = composite_visible_frame(viewer, frame_index=1)
        assert out.shape == (40, 50, 3)
        # Outside mask: pure blue base.
        assert tuple(out[0, 0]) == (0, 0, 100)
        # Inside mask: blended (not pure blue, not pure label color).
        assert not np.array_equal(out[20, 20], (0, 0, 100))
        assert out.dtype == np.uint8
    finally:
        viewer.close()


def test_register_save_frame_actions_is_idempotent():
    from napari._app_model import get_app_model

    import napari_pecan_py.save_frame as mod

    mod._registered_layer = False
    mod._registered_visible = False
    register_save_frame_actions()
    app = get_app_model()
    for action_id in (
        _LAYER_ACTION_ID,
        _LAYER_ACTION_ID_AS,
        _VISIBLE_ACTION_ID,
        _VISIBLE_ACTION_ID_AS,
    ):
        assert action_id in app.commands
    register_save_frame_actions()
    assert _LAYER_ACTION_ID in app.commands
