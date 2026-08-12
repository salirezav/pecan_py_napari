"""Tests for layer → MP4 save helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_pecan_py.save_video import (
    _ACTION_ID,
    _ACTION_ID_AS,
    default_export_path,
    iter_layer_rgb_frames,
    labels_frame_to_rgb,
    register_save_video_actions,
    save_layer_as_video,
    write_rgb_frames_to_mp4,
)


def test_default_export_path_suffix(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    assert default_export_path(src) == tmp_path / "clip - saved.mp4"
    assert default_export_path(src, layer_name="clip") == tmp_path / "clip - saved.mp4"
    assert (
        default_export_path(src, layer_name="mask")
        == tmp_path / "clip - saved - mask.mp4"
    )


def test_labels_frame_to_rgb_colors_nonzero():
    frame = np.zeros((8, 8), dtype=np.uint16)
    frame[2:5, 2:5] = 1
    frame[5:7, 5:7] = 2
    rgb = labels_frame_to_rgb(frame)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    assert np.all(rgb[0, 0] == 0)
    assert not np.all(rgb[3, 3] == 0)
    assert not np.array_equal(rgb[3, 3], rgb[6, 6])


def test_iter_image_and_labels_rgb_frames():
    video = SimpleNamespace(
        data=np.arange(3 * 4 * 4 * 3, dtype=np.uint8).reshape(3, 4, 4, 3),
        rgb=True,
        _type_string="image",
        contrast_limits=None,
        name="vid",
    )
    frames = list(iter_layer_rgb_frames(video))
    assert len(frames) == 3
    assert frames[0].shape == (4, 4, 3)

    labels = SimpleNamespace(
        data=np.zeros((2, 5, 5), dtype=np.uint16),
        _type_string="labels",
        name="labs",
        get_color=None,
    )
    labels.data[0, 1:3, 1:3] = 1
    lab_frames = list(iter_layer_rgb_frames(labels))
    assert len(lab_frames) == 2
    assert lab_frames[0].shape == (5, 5, 3)


def test_write_and_save_layer_as_video(tmp_path: Path):
    frames = np.zeros((4, 16, 16, 3), dtype=np.uint8)
    frames[0, :, :] = (255, 0, 0)
    frames[1, :, :] = (0, 255, 0)
    out = write_rgb_frames_to_mp4(frames, tmp_path / "out.mp4", fps=5.0)
    assert out.exists()
    assert out.stat().st_size > 0

    layer = SimpleNamespace(
        data=frames,
        rgb=True,
        _type_string="image",
        contrast_limits=None,
        name="vid",
        metadata={"source_path": str(tmp_path / "src.mp4")},
    )
    written = save_layer_as_video(layer, tmp_path / "layer.mp4", fps=5.0)
    assert written.exists()


def test_save_rejects_unsupported_type():
    layer = SimpleNamespace(
        data=np.zeros((5, 2), dtype=float),
        _type_string="points",
        name="pts",
    )
    with pytest.raises(ValueError, match="Cannot export"):
        list(iter_layer_rgb_frames(layer))


def test_register_save_video_actions_is_idempotent():
    from napari._app_model import get_app_model

    import napari_pecan_py.save_video as save_video

    save_video._registered = False
    register_save_video_actions()
    app = get_app_model()
    assert _ACTION_ID_AS in app.commands
    assert _ACTION_ID in app.commands
    register_save_video_actions()
    assert _ACTION_ID_AS in app.commands
    assert _ACTION_ID in app.commands
