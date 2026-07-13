"""Tests for mask retouching logic."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.mask_retouching.logic import (
    apply_retouching_pipeline,
    apply_retouching_to_volume,
    fill_holes,
    morphological_close,
    watershed_split,
)


def test_morphology_accepts_int32_labels():
    """Napari Labels / TIFF masks are often int32; OpenCV morph rejects that dtype."""
    mask = np.zeros((20, 20), dtype=np.int32)
    mask[5:15, 5:15] = 3
    mask[8:12, 8:12] = 0  # hole that close can fill

    closed = morphological_close(mask, 5)
    assert closed.dtype == np.int32
    assert closed[8:12, 8:12].all()
    assert (closed[5:15, 5:15] > 0).all()

    out = apply_retouching_pipeline(mask, close_size=5, open_size=3, dilate_size=3, erode_size=3)
    assert out.dtype == np.int32
    assert out.shape == mask.shape


def test_fill_holes_preserves_corner_pocket_when_mask_touches_two_edges():
    """Background in a frame corner must not be filled when the mask blocks both edges."""
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:9, 0:7] = 1
    mask[9, 1:9] = 1
    mask[4:9, 0] = 1
    mask[6:8, 3:5] = 0  # true internal hole

    filled = fill_holes(mask)

    assert filled[9, 0] == 0
    assert filled[6, 4] == 1
    assert filled[7, 4] == 1


def test_fill_holes_fills_fully_enclosed_hole():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    mask[4:6, 4:6] = 0

    filled = fill_holes(mask)

    assert filled[4:6, 4:6].all()
    assert not filled[0, 0]
    assert not filled[9, 9]


def test_fill_holes_respects_area_range():
    """Small hole filled, large hole left open when max_area is set."""
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[2:38, 2:38] = 1
    # Small 2x2 hole (area 4)
    mask[5:7, 5:7] = 0
    # Large 10x10 hole (area 100)
    mask[15:25, 15:25] = 0

    filled = fill_holes(mask, min_area=1, max_area=20)

    assert filled[5:7, 5:7].all()
    assert not filled[15:25, 15:25].any()
    # Foreground shell preserved
    assert filled[3, 3] == 1


def test_fill_holes_min_area_skips_tiny_holes():
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[2:28, 2:28] = 1
    mask[5:7, 5:7] = 0  # area 4
    mask[12:18, 12:18] = 0  # area 36

    filled = fill_holes(mask, min_area=10, max_area=0)

    assert not filled[5:7, 5:7].any()
    assert filled[12:18, 12:18].all()


def _two_touching_disks(h: int = 80, w: int = 120, r: int = 18) -> np.ndarray:
    """Binary mask with two disks that touch at a narrow neck."""
    yy, xx = np.ogrid[:h, :w]
    c1 = (h // 2, w // 2 - r + 2)
    c2 = (h // 2, w // 2 + r - 2)
    d1 = (yy - c1[0]) ** 2 + (xx - c1[1]) ** 2 <= r**2
    d2 = (yy - c2[0]) ** 2 + (xx - c2[1]) ** 2 <= r**2
    return (d1 | d2).astype(np.uint8)


def test_watershed_split_separates_two_touching_disks():
    mask = _two_touching_disks()
    # Without split: one connected component.
    n_before = len(np.unique(mask)) - (1 if 0 in mask else 0)
    assert n_before == 1

    split = watershed_split(mask, min_distance=10, min_peak_fraction=0.2)
    ids = [i for i in np.unique(split) if i != 0]
    assert len(ids) == 2
    # Both seeds should cover a meaningful area.
    assert all(int((split == i).sum()) > 100 for i in ids)


def test_watershed_split_empty_mask():
    mask = np.zeros((20, 20), dtype=np.uint8)
    out = watershed_split(mask, min_distance=5)
    assert out.shape == mask.shape
    assert not out.any()


def test_apply_retouching_pipeline_watershed_flag():
    mask = _two_touching_disks()
    out = apply_retouching_pipeline(
        mask,
        do_watershed_split=True,
        watershed_min_distance=10,
    )
    ids = [i for i in np.unique(out) if i != 0]
    assert len(ids) == 2


def test_apply_retouching_pipeline_watershed_off_preserves_binary():
    mask = _two_touching_disks()
    out = apply_retouching_pipeline(mask, do_watershed_split=False)
    assert set(np.unique(out)) <= {0, 1}


def test_apply_retouching_to_volume_parallel_matches_sequential():
    frame = _two_touching_disks(h=40, w=60, r=10)
    volume = np.stack([frame, np.roll(frame, 3, axis=1), np.roll(frame, 5, axis=0)], axis=0)
    params = dict(
        close_size=3,
        open_size=3,
        min_area=20,
        do_fill_holes=True,
        do_watershed_split=True,
        watershed_min_distance=8,
        smooth_size=3,
    )
    seq = apply_retouching_to_volume(volume, max_workers=1, **params)
    par = apply_retouching_to_volume(volume, max_workers=4, **params)
    np.testing.assert_array_equal(seq, par)


def test_apply_retouching_to_volume_2d_passthrough():
    mask = _two_touching_disks(h=40, w=60, r=10)
    out = apply_retouching_to_volume(mask, close_size=3, max_workers=4)
    assert out.ndim == 2
    assert out.shape == mask.shape
