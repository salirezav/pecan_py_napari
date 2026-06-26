"""Tests for YOLO segmentation model helpers."""

from pathlib import Path

import numpy as np

from napari_pecan_py.widgets.yolo_seg.model import (
    discover_mask_files,
    discover_videos_in_directory,
    is_multiclass_label_map,
    load_masks_by_class_from_paths,
    split_label_map_to_binary_masks,
)


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


def test_discover_combined_label_map_masks(tmp_path: Path):
    import tifffile

    video = tmp_path / "GH013154-cropped.MP4"
    video.write_bytes(b"")

    label_map = np.zeros((8, 8), dtype=np.uint8)
    label_map[1:3, 1:3] = 1  # Crack
    label_map[1:3, 4:6] = 2  # Kernel
    label_map[5:7, 1:3] = 3  # Pecan
    mask_path = tmp_path / "GH013154-cropped - Crack.tiff"
    tifffile.imwrite(mask_path, label_map)

    masks = discover_mask_files(video)
    assert set(masks.keys()) == {"Crack", "Kernel", "Pecan"}
    assert len(set(masks.values())) == 1
    assert masks["Crack"] == mask_path


def test_load_masks_by_class_from_combined_label_map(tmp_path: Path):
    import tifffile

    label_map = np.zeros((2, 6, 6), dtype=np.uint8)
    label_map[:, 1:3, 1:3] = 1
    label_map[:, 1:3, 4:6] = 2
    mask_path = tmp_path / "combined.tiff"
    tifffile.imwrite(mask_path, label_map)

    assert is_multiclass_label_map(label_map)
    split = split_label_map_to_binary_masks(label_map)
    assert split["Crack"].sum() > 0
    assert split["Kernel"].sum() > 0
    assert split["Pecan"].sum() == 0

    loaded = load_masks_by_class_from_paths(
        {
            "Crack": mask_path,
            "Kernel": mask_path,
            "Pecan": mask_path,
        }
    )
    assert set(loaded.keys()) == {"Crack", "Kernel", "Pecan"}
    assert loaded["Crack"][0, 1, 1] == 1
    assert loaded["Kernel"][0, 1, 4] == 1
    assert loaded["Pecan"].sum() == 0
