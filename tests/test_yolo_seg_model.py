"""Tests for YOLO segmentation model helpers."""

from pathlib import Path

import numpy as np

import cv2

from napari_pecan_py.widgets.yolo_seg.model import (
    _instance_polygons_from_mask,
    _polygon_xy_components,
    _write_yolo_label_lines,
    binary_mask_for_label_ids,
    discover_mask_files,
    discover_videos_in_directory,
    is_multiclass_label_map,
    label_mask_for_label_ids,
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


def test_preserve_instance_ids_keeps_touching_objects(tmp_path: Path):
    import tifffile

    # Two touching 8x8 squares with distinct IDs (share an edge).
    label_map = np.zeros((10, 17), dtype=np.uint8)
    label_map[1:9, 1:9] = 1
    label_map[1:9, 9:17] = 2
    mask_path = tmp_path / "touching - Pecan.tiff"
    tifffile.imwrite(mask_path, label_map)

    binary = load_masks_by_class_from_paths(
        {"Pecan": mask_path},
        label_ids_by_class={"Pecan": None},
        preserve_instance_ids=False,
    )["Pecan"]
    assert set(np.unique(binary).tolist()) == {0, 1}

    preserved = load_masks_by_class_from_paths(
        {"Pecan": mask_path},
        label_ids_by_class={"Pecan": None},
        preserve_instance_ids=True,
    )["Pecan"]
    assert set(int(v) for v in np.unique(preserved) if int(v) > 0) == {1, 2}

    lines = _write_yolo_label_lines({"Pecan": preserved}, ["Pecan"], 0, 10, 17)
    assert len(lines) == 2
    assert all(line.startswith("0 ") for line in lines)

    # Collapsed binary would yield a single merged polygon for touching blobs.
    merged_lines = _write_yolo_label_lines({"Pecan": binary}, ["Pecan"], 0, 10, 17)
    assert len(merged_lines) == 1


def test_binary_mask_still_splits_disconnected_components():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:12, 2:12] = 1
    mask[22:32, 22:32] = 1
    polys = _instance_polygons_from_mask(mask)
    assert len(polys) == 2


def test_label_mask_for_label_ids_preserves_values():
    m = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    assert label_mask_for_label_ids(m, None).tolist() == [[0, 1], [2, 3]]
    assert label_mask_for_label_ids(m, {1, 3}).tolist() == [[0, 1], [0, 3]]


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


def test_yolo_result_to_label_map_instance_ids():
    class _FakeBoxes:
        def __init__(self):
            import torch

            self.cls = torch.tensor([0.0, 0.0])
            self.conf = torch.tensor([0.9, 0.8])

    class _FakeMasks:
        def __init__(self):
            import torch

            # Two non-overlapping 4x4 masks in a 16x16 canvas (model scale).
            m0 = torch.zeros((16, 16))
            m0[2:6, 2:6] = 1
            m1 = torch.zeros((16, 16))
            m1[10:14, 10:14] = 1
            self.data = torch.stack([m0, m1], dim=0)
            self.xy = None

    class _FakeResult:
        def __init__(self):
            self.orig_shape = (16, 16)
            self.boxes = _FakeBoxes()
            self.masks = _FakeMasks()

    semantic = yolo_result_to_label_map(_FakeResult(), instance_labels=False)
    assert semantic is not None
    assert set(int(v) for v in np.unique(semantic) if int(v) > 0) == {1}

    instances = yolo_result_to_label_map(_FakeResult(), instance_labels=True)
    assert instances is not None
    assert instances.dtype == np.uint16
    assert set(int(v) for v in np.unique(instances) if int(v) > 0) == {1, 2}
