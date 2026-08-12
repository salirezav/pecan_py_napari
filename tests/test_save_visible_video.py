"""Tests for composited viewer → MP4 save helpers."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_save_visible_as_video_screenshots_each_frame(tmp_path: Path):
    frames = [
        np.full((20, 30, 4), 10, dtype=np.uint8),
        np.full((20, 30, 4), 40, dtype=np.uint8),
        np.full((20, 30, 4), 80, dtype=np.uint8),
    ]
    calls = {"t": 0}

    def screenshot(*, canvas_only=True, flash=False):
        assert canvas_only is True
        assert flash is False
        idx = min(calls["t"], len(frames) - 1)
        return frames[idx]

    dims = MagicMock()
    dims.nsteps = (3, 20, 30)
    dims.not_displayed = (0,)
    dims.last_used = 0
    dims.current_step = (0, 0, 0)

    def set_current_step(axis, value):
        calls["t"] = int(value)
        dims.current_step = (int(value), 0, 0)

    dims.set_current_step.side_effect = set_current_step

    viewer = SimpleNamespace(
        layers=[],
        dims=dims,
        screenshot=screenshot,
    )

    out = save_visible_as_video(viewer, tmp_path / "visible.mp4", fps=5.0)
    assert out.exists()
    assert out.stat().st_size > 0
    assert dims.set_current_step.call_count == 3
    # Restored after export.
    assert dims.current_step == (0, 0, 0)


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
