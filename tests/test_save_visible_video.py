"""Tests for composited viewer → MP4 save helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_pecan_py.save_visible_video import (
    VISIBLE_SAVE_SUFFIX,
    _ACTION_ID,
    _ACTION_ID_AS,
    _suggested_visible_path,
    _viewer_source_path,
    playback_axis_and_count,
    register_save_visible_video_actions,
    save_visible_as_video,
)


def test_viewer_source_path_prefers_visible_image():
    img = SimpleNamespace(
        visible=True,
        _type_string="image",
        metadata={"source_path": "/data/clip.mp4"},
        data=SimpleNamespace(path=None),
    )
    labels = SimpleNamespace(
        visible=True,
        _type_string="labels",
        metadata={},
        data=np.zeros((2, 2)),
    )
    viewer = SimpleNamespace(layers=[labels, img])
    assert _viewer_source_path(viewer) == "/data/clip.mp4"


def test_suggested_visible_path_uses_suffix(tmp_path: Path):
    src = tmp_path / "clip.mp4"
    img = SimpleNamespace(
        visible=True,
        _type_string="image",
        metadata={"source_path": str(src)},
        data=SimpleNamespace(path=None),
    )
    viewer = SimpleNamespace(layers=[img])
    out = _suggested_visible_path(viewer)
    assert out == tmp_path / f"clip{VISIBLE_SAVE_SUFFIX}.mp4"


def test_playback_axis_and_count():
    dims = SimpleNamespace(
        nsteps=(12, 64, 64),
        not_displayed=(0,),
        last_used=0,
    )
    viewer = SimpleNamespace(dims=dims)
    assert playback_axis_and_count(viewer) == (0, 12)

    still = SimpleNamespace(
        dims=SimpleNamespace(nsteps=(1, 64, 64), not_displayed=(0,), last_used=0)
    )
    assert playback_axis_and_count(still) is None


def test_save_visible_as_video_composites_each_frame(tmp_path: Path):
    import napari

    from napari_pecan_py.save_frame import iter_composited_visible_frames

    viewer = napari.Viewer(show=False)
    try:
        video = np.zeros((3, 24, 32, 3), dtype=np.uint8)
        video[0] = (30, 0, 0)
        video[1] = (0, 30, 0)
        video[2] = (0, 0, 30)
        viewer.add_image(video, rgb=True, name="vid")
        labs = np.zeros((3, 24, 32), dtype=np.uint16)
        labs[:, 5:15, 5:15] = 1
        layer = viewer.add_labels(labs, name="mask")
        layer.opacity = 0.4
        start = tuple(viewer.dims.current_step)

        frames = list(iter_composited_visible_frames(viewer, n_frames=3))
        assert len(frames) == 3
        assert frames[0].shape == (24, 32, 3)
        # Must not scrub the viewer slider (that was the old slow screenshot path).
        assert tuple(viewer.dims.current_step) == start
        # Frame 0 outside mask stays base red channel content.
        assert tuple(frames[0][0, 0]) == (30, 0, 0)
        assert not np.array_equal(frames[0][10, 10], (30, 0, 0))

        out = save_visible_as_video(viewer, tmp_path / "visible.mp4", fps=5.0)
        assert out.exists()
        assert out.stat().st_size > 0
        assert tuple(viewer.dims.current_step) == start
    finally:
        viewer.close()


def test_register_save_visible_video_actions_is_idempotent():
    from napari._app_model import get_app_model

    import napari_pecan_py.save_visible_video as mod

    mod._registered = False
    register_save_visible_video_actions()
    app = get_app_model()
    assert _ACTION_ID_AS in app.commands
    assert _ACTION_ID in app.commands
    register_save_visible_video_actions()
    assert _ACTION_ID_AS in app.commands
