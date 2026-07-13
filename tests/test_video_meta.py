"""Tests for persisted video frame-range metadata and windowed loading."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from napari_pecan_py.trim_frames import apply_trim_to_layer
from napari_pecan_py.video_meta import (
    clear_saved_frame_range,
    get_saved_frame_range,
    open_lazy_video,
    pecan_meta_path,
    set_saved_frame_range,
)


def _write_solid_video(path: Path, n_frames: int = 20, size: int = 16) -> Path:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 5.0, (size, size))
    assert writer.isOpened(), f"Could not open VideoWriter for {path}"
    try:
        for i in range(n_frames):
            frame = np.full((size, size, 3), i % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_set_and_get_saved_frame_range(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")  # path only; meta does not open the file
    out = set_saved_frame_range(video, 3, 9, native_frame_count=20)
    assert out == pecan_meta_path(video)
    assert out.is_file()
    assert get_saved_frame_range(video) == (3, 9)
    text = out.read_text(encoding="utf-8")
    assert '"start": 3' in text
    assert '"end": 9' in text


def test_clear_saved_frame_range_removes_sidecar(tmp_path: Path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    set_saved_frame_range(video, 1, 2)
    assert pecan_meta_path(video).is_file()
    assert clear_saved_frame_range(video) is True
    assert not pecan_meta_path(video).is_file()
    assert get_saved_frame_range(video) is None


def test_open_lazy_video_applies_saved_range(tmp_path: Path):
    video = _write_solid_video(tmp_path / "clip.mp4", n_frames=20)
    set_saved_frame_range(video, 5, 14, native_frame_count=20)

    lazy = open_lazy_video(video)
    assert lazy.shape[0] == 10
    assert lazy.frame_range == (5, 14)
    # Smoke-read every virtual frame (OpenCV seek is not always pixel-exact).
    for t in range(lazy.shape[0]):
        frame = np.asarray(lazy[t])
        assert frame.shape == (16, 16, 3)

    full = open_lazy_video(video, apply_saved_range=False)
    assert full.shape[0] == 20


def test_video_frame_count_honors_sidecar(tmp_path: Path):
    from napari_pecan_py.widgets.yolo_seg.model import video_frame_count

    video = _write_solid_video(tmp_path / "clip.mp4", n_frames=15)
    assert video_frame_count(video) == 15
    set_saved_frame_range(video, 2, 8)
    assert video_frame_count(video) == 7


def test_apply_trim_persists_sidecar_and_keeps_lazy_window(tmp_path: Path):
    import napari

    video = _write_solid_video(tmp_path / "clip.mp4", n_frames=20)
    viewer = napari.Viewer(show=False)
    try:
        from napari_pecan_py.video_meta import open_lazy_video

        frames = open_lazy_video(video, apply_saved_range=False)
        layer = viewer.add_image(
            frames,
            rgb=True,
            name="clip",
            metadata={"source_path": str(video)},
        )
        new_n = apply_trim_to_layer(layer, 4, 11, persist=True)
        assert new_n == 8
        assert layer.data.shape[0] == 8
        assert get_saved_frame_range(video) == (4, 11)
        assert layer.metadata["frame_range"] == {"start": 4, "end": 11}
        assert viewer.dims.nsteps[0] == 8
    finally:
        viewer.close()


def test_apply_trim_composes_with_existing_range(tmp_path: Path):
    import napari

    video = _write_solid_video(tmp_path / "clip.mp4", n_frames=30)
    set_saved_frame_range(video, 10, 29, native_frame_count=30)
    viewer = napari.Viewer(show=False)
    try:
        from napari_pecan_py.video_meta import open_lazy_video

        frames = open_lazy_video(video)
        assert frames.shape[0] == 20
        layer = viewer.add_image(
            frames,
            rgb=True,
            name="clip",
            metadata={
                "source_path": str(video),
                "frame_range": {"start": 10, "end": 29},
            },
        )
        # Trim relative indices 2..6 of the already-windowed stack → absolute 12..16
        new_n = apply_trim_to_layer(layer, 2, 6, persist=True)
        assert new_n == 5
        assert get_saved_frame_range(video) == (12, 16)
    finally:
        viewer.close()
