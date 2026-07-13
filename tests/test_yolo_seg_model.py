"""Tests for YOLO segmentation model helpers."""

from pathlib import Path

import numpy as np

import cv2

from napari_pecan_py.widgets.yolo_seg.model import (
    _polygon_xy_components,
    binary_mask_for_label_ids,
    discover_mask_files,
    discover_videos_in_directory,
    is_multiclass_label_map,
    load_masks_by_class_from_paths,
    parse_label_ids_text,
    format_label_ids_text,
    split_label_map_to_binary_masks,
    yolo_result_to_label_map,
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


def test_watershed_instance_map_not_treated_as_multiclass(tmp_path: Path):
    import tifffile

    video = tmp_path / "clip.MP4"
    video.write_bytes(b"")
    # Instance IDs 1..12 (includes semantic IDs but also higher ones).
    label_map = np.zeros((16, 16), dtype=np.uint8)
    for i in range(1, 13):
        label_map[i, i] = i
    mask_path = tmp_path / "clip - Pecan.tiff"
    tifffile.imwrite(mask_path, label_map)

    assert not is_multiclass_label_map(label_map)
    masks = discover_mask_files(video)
    assert set(masks.keys()) == {"Pecan"}

    loaded = load_masks_by_class_from_paths(
        {"Pecan": mask_path},
        label_ids_by_class={"Pecan": None},
    )
    assert loaded["Pecan"].sum() == 12

    loaded_one = load_masks_by_class_from_paths(
        {"Pecan": mask_path},
        label_ids_by_class={"Pecan": {3}},
    )
    assert int(loaded_one["Pecan"].sum()) == 1


def test_parse_and_format_label_ids_text():
    assert parse_label_ids_text("*") is None
    assert parse_label_ids_text("[*]") is None
    assert parse_label_ids_text("1, 2") == {1, 2}
    assert parse_label_ids_text("[3]") == {3}
    assert format_label_ids_text(None) == "*"
    assert format_label_ids_text({2, 1}) == "1, 2"
    assert binary_mask_for_label_ids(
        np.array([[0, 1], [2, 3]], dtype=np.uint8), {1, 3}
    ).tolist() == [[0, 1], [0, 1]]


def test_polygon_xy_components_splits_disconnected_islands():
    poly = np.array(
        [
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20],
            [100, 100],
            [110, 100],
            [110, 110],
            [100, 110],
        ],
        dtype=np.float64,
    )
    assert len(_polygon_xy_components(poly)) == 2


def test_yolo_result_to_label_map_avoids_polygon_connector_lines():
    class _FakeMasks:
        def __init__(self, poly):
            self.data = None
            self.xy = [poly]

    class _FakeResult:
        def __init__(self, poly):
            self.orig_shape = (120, 120)
            self.boxes = None
            self.masks = _FakeMasks(poly)

    poly = np.array(
        [
            [10, 10],
            [20, 10],
            [20, 20],
            [10, 20],
            [100, 100],
            [110, 100],
            [110, 110],
            [100, 110],
        ],
        dtype=np.float64,
    )
    label_map = yolo_result_to_label_map(_FakeResult(poly))
    assert label_map is not None

    bridged = np.zeros((120, 120), dtype=np.uint8)
    cv2.fillPoly(bridged, [np.round(poly).astype(np.int32)], 1)
    assert int(bridged[50:99, 10:110].any()) == 1
    assert int(label_map[50:99, 10:110].any()) == 0
    assert int(label_map[10:20, 10:20].any()) == 1
    assert int(label_map[100:110, 100:110].any()) == 1
