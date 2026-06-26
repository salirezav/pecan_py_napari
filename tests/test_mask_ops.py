"""Tests for mask_ops.logic."""

import numpy as np

from napari_pecan_py.widgets.mask_ops.logic import (
    apply_binary_operation,
    apply_binary_operation_bool,
    binarize_edge_raster,
    clip_mask_label_volume,
    close_edge_raster_gaps,
    detect_parallel_edge_bands_volume,
    expand_mask_to_layer_shape,
    label_only_volume,
    mask_volume_for_label,
    mask_volume_from_array,
    merge_label_mask_into_volume,
    new_labels_from_binary,
    positive_label_values,
)


def test_binarize_and_close_edge_raster():
    edges = np.zeros((10, 10), dtype=np.uint8)
    edges[2, 2:8] = 255
    edges[7, 2:8] = 255
    b = binarize_edge_raster(edges, threshold=1)
    assert b[2, 4] and b[7, 4]
    closed = close_edge_raster_gaps(b, kernel_size=0)
    np.testing.assert_array_equal(closed, b)


def test_detect_parallel_edge_bands_volume_simple():
    e = np.zeros((50, 60), dtype=np.uint8)
    e[10:40, 20] = 255
    e[10:40, 26] = 255
    bands = detect_parallel_edge_bands_volume(
        e,
        edge_threshold=1,
        pre_close_size=0,
        min_distance_px=2,
        max_distance_px=10,
        angle_tolerance_deg=30,
        min_component_px=10,
    )
    assert np.any(bands[15:35, 21:26])


def test_detect_parallel_edge_bands_prefers_closest_parallel_pair():
    """A and B are close parallel rails; C is far — pair A-B, not B-C."""
    e = np.zeros((50, 60), dtype=np.uint8)
    e[10:40, 10] = 255  # A
    e[10:40, 16] = 255  # B (6 px from A)
    e[10:40, 45] = 255  # C (far from B)

    bands = detect_parallel_edge_bands_volume(
        e,
        edge_threshold=1,
        pre_close_size=0,
        min_distance_px=2,
        max_distance_px=10,
        angle_tolerance_deg=30,
        min_component_px=10,
    )
    assert np.any(bands[15:35, 11:16])
    assert not np.any(bands[15:35, 17:44])


def test_detect_parallel_edge_bands_exclusive_when_decoy_is_closer():
    """A middle decoy line must not steal A from its outer parallel partner B."""
    e = np.zeros((50, 60), dtype=np.uint8)
    e[10:40, 10] = 255  # A
    e[10:40, 14] = 255  # decoy between A and B
    e[10:40, 20] = 255  # B

    bands = detect_parallel_edge_bands_volume(
        e,
        edge_threshold=1,
        pre_close_size=0,
        min_distance_px=2,
        max_distance_px=12,
        angle_tolerance_deg=30,
        min_component_px=10,
    )
    assert np.any(bands[15:35, 11:20])
    assert np.any(bands[15:35, 18:20])  # band reaches outer rail B, not just decoy


def test_mask_volume_from_array_rgb_image():
    rgb = np.zeros((4, 5, 3), dtype=np.uint8)
    rgb[1, 2] = [255, 0, 0]
    vol = mask_volume_from_array(rgb)
    assert vol.shape == (4, 5)
    assert vol[1, 2] == 1
    assert vol[0, 0] == 0


def test_expand_mask_to_layer_shape_rgb():
    template = np.zeros((4, 5, 3), dtype=np.float32)
    mask = np.zeros((4, 5), dtype=np.uint8)
    mask[1, 2] = 1
    out = expand_mask_to_layer_shape(mask, template)
    assert out.shape == template.shape
    assert np.all(out[1, 2] == 1.0)
    assert np.all(out[0, 0] == 0.0)


def test_apply_binary_operation_labels_and_image():
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1:4, 1:4] = 2
    image = np.zeros((6, 6), dtype=np.float32)
    image[2:5, 2:5] = 1.0
    res = apply_binary_operation(labels, image, op="and", template=labels)
    assert res.shape == (6, 6)
    assert res[2, 2] == 2
    assert res[1, 1] == 0
    assert res[4, 4] == 0


def test_apply_binary_operation_image_or():
    a = np.zeros((4, 4), dtype=np.float32)
    a[0, 0] = 1.0
    b = np.zeros((4, 4), dtype=np.float32)
    b[3, 3] = 1.0
    res = apply_binary_operation(a, b, op="or", template=a)
    expanded = expand_mask_to_layer_shape(res, a)
    assert expanded[0, 0] == 1.0
    assert expanded[3, 3] == 1.0
    assert expanded[1, 1] == 0.0


def test_positive_label_values_multi_roi():
    labels = np.zeros((8, 8), dtype=np.uint8)
    labels[1:3, 1:3] = 1
    labels[5:7, 5:7] = 3
    assert positive_label_values(labels) == [1, 3]


def test_mask_volume_for_label_isolates_one_roi():
    labels = np.zeros((6, 6), dtype=np.uint8)
    labels[1:3, 1:3] = 2
    labels[4:6, 4:6] = 5
    m2 = mask_volume_for_label(labels, 2)
    assert m2[2, 2] == 1
    assert m2[5, 5] == 0


def test_merge_label_mask_preserves_other_labels():
    original = np.zeros((6, 6), dtype=np.uint8)
    original[0:2, 0:2] = 1
    original[4:6, 4:6] = 3
    result = np.zeros((6, 6), dtype=bool)
    result[1:5, 1:5] = True
    merged = merge_label_mask_into_volume(original, result, 3)
    assert merged[0, 0] == 1
    assert merged[1, 1] == 1
    assert merged[2, 2] == 3
    assert merged[4, 4] == 3
    assert merged[0, 5] == 0


def test_clip_mask_label_volume_only_affects_selected_label():
    labels = np.zeros((10, 10), dtype=np.uint8)
    labels[:, :] = 0
    labels[2:8, 2:8] = 2
    labels[1:3, 1:3] = 5
    ellipse = np.zeros((10, 10), dtype=bool)
    ellipse[3:9, 3:9] = True
    clipped = clip_mask_label_volume(labels, ellipse, 2)
    assert clipped[2, 2] == 5
    assert clipped[4, 4] == 2
    assert clipped[2, 7] == 0


def test_apply_binary_operation_bool_with_labels():
    a = np.zeros((6, 6), dtype=np.uint8)
    a[1:4, 1:4] = 2
    a[0, 0] = 9
    b = np.zeros((6, 6), dtype=np.uint8)
    b[2:5, 2:5] = 1
    res = apply_binary_operation_bool(a, b, op="and", label_a=2, label_b=None)
    assert res[2, 2]
    assert not res[0, 0]
    assert not res[4, 4]


def test_new_labels_from_binary():
    mask = np.zeros((4, 4), dtype=bool)
    mask[1, 1] = True
    out = new_labels_from_binary(mask, 7, dtype=np.uint8)
    assert out[1, 1] == 7
    assert out[0, 0] == 0


def test_label_only_volume():
    vol = np.array([[0, 1], [2, 3]], dtype=np.uint8)
    out = label_only_volume(vol, 2)
    assert out.tolist() == [[0, 0], [2, 0]]
