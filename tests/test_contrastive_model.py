"""Tests for contrastive similarity helpers."""

import numpy as np

from napari_pecan_py.widgets.contrastive_coding.model import (
    build_patch_highlight_mask,
    resolve_similarity_cutoff,
)


def test_resolve_similarity_cutoff_peak_fraction():
    grid = np.array([[0.5, 0.9], [0.7, 1.0]], dtype=np.float32)
    assert resolve_similarity_cutoff(grid, "peak_fraction", 0.9) == 0.9


def test_build_patch_highlight_mask_only_passing_patches():
    grid = np.array([[0.2, 0.95], [0.5, 0.4]], dtype=np.float32)
    ys = [4, 8]
    xs = [4, 8]
    mask = build_patch_highlight_mask(grid, ys, xs, patch_size=4, height=12, width=12, cutoff=0.9)
    assert mask.sum() > 0
    assert mask.sum() < mask.size
