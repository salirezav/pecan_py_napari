"""Tests for YOLO segmentation model helpers."""

from pathlib import Path

from napari_pecan_py.widgets.yolo_seg.model import discover_videos_in_directory


def test_discover_videos_in_directory_recursive(tmp_path: Path):
    root = tmp_path / "batch"
    (root / "a").mkdir(parents=True)
    (root / "a" / "nested").mkdir()
    (root / "a" / "clip.mp4").write_bytes(b"")
    (root / "a" / "nested" / "other.MOV").write_bytes(b"")
    (root / "a" / "nested" / "notes.txt").write_text("skip")

    found = discover_videos_in_directory(root)
    names = {p.name for p in found}
    assert names == {"clip.mp4", "other.MOV"}
