"""Tests for mask retouching logic."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.mask_retouching.logic import fill_holes


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
