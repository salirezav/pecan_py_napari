"""Tests for glare / specular reduction adjustment."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.color_adjustments.defaults import default_adjustment_item
from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video
from napari_pecan_py.widgets.color_adjustments.parallelism import FRAME_PARALLEL_TYPES, is_parallelizable
from napari_pecan_py.widgets.color_thresholding.defaults import DEFAULT_GLARE_REDUCTION
from napari_pecan_py.widgets.color_thresholding.glare import apply_glare_reduction
from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack


def _glare_rgb(size: int = 32) -> np.ndarray:
    """Matte surface with a bright desaturated specular hotspot."""
    img = np.full((size, size, 3), 80, dtype=np.uint8)
    img[8:24, 8:24] = (140, 90, 50)  # warm matte
    # Near-white specular blob
    img[12:18, 12:18] = 250
    return img


def test_default_glare_reduction_item():
    item = default_adjustment_item("glare_reduction")
    assert item["type"] == "glare_reduction"
    assert item["method"] == "v_rolloff"
    assert item["type"] == DEFAULT_GLARE_REDUCTION["type"]
    assert item["method"] == DEFAULT_GLARE_REDUCTION["method"]


def test_glare_reduction_is_frame_parallel():
    assert "glare_reduction" in FRAME_PARALLEL_TYPES
    assert is_parallelizable({"type": "glare_reduction", "enabled": True}) is True


def test_apply_glare_methods_preserve_shape_dtype():
    img = _glare_rgb()
    methods = {
        "v_rolloff": {"v_threshold": 200.0, "sat_max": 60.0, "strength": 0.8, "target_v": 150.0},
        "specular_inpaint": {
            "v_threshold": 220.0,
            "sat_max": 40.0,
            "mask_threshold": 0.4,
            "dilate_radius": 1,
            "inpaint_radius": 2.0,
            "inpaint_algorithm": "telea",
        },
        "retinex": {"retinex_mode": "single", "sigma": 20.0, "gain": 1.0},
        "homomorphic": {"sigma": 15.0, "gamma_l": 0.5, "gamma_h": 1.4},
        "dichromatic": {"v_threshold": 200.0, "sat_max": 55.0, "strength": 0.9, "blur_sigma": 3.0},
    }
    for method, params in methods.items():
        out = apply_glare_reduction(img, method=method, params=params)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_v_rolloff_darkens_specular_hotspot():
    img = _glare_rgb()
    out = apply_glare_reduction(
        img,
        method="v_rolloff",
        params={
            "v_threshold": 200.0,
            "sat_max": 80.0,
            "strength": 1.0,
            "target_v": 140.0,
            "soft_width": 10.0,
        },
    )
    # Hotspot center should get darker (or equal) after roll-off.
    assert float(out[15, 15].mean()) <= float(img[15, 15].mean())
    # Matte region outside hotspot should be unchanged-ish (allow tiny numeric drift).
    assert np.allclose(out[4, 4], img[4, 4], atol=2)


def test_stack_dispatch_glare_reduction():
    img = _glare_rgb()
    stack = [
        {
            "type": "glare_reduction",
            "enabled": True,
            "method": "dichromatic",
            "v_threshold": 200.0,
            "sat_max": 60.0,
            "strength": 0.9,
            "blur_sigma": 3.0,
        }
    ]
    out = apply_adjustment_stack(img, stack)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, img)


def test_glare_parallel_matches_sequential():
    rng = np.random.default_rng(0)
    t, h, w = 6, 24, 24
    video = rng.integers(40, 200, size=(t, h, w, 3), dtype=np.uint8)
    video[:, 8:14, 8:14] = 250
    stack = [
        {
            "type": "glare_reduction",
            "enabled": True,
            "method": "v_rolloff",
            "v_threshold": 210.0,
            "sat_max": 50.0,
            "strength": 0.75,
            "target_v": 150.0,
            "soft_width": 20.0,
        }
    ]
    seq = apply_adjustments_to_video(video, stack, max_workers=1)
    par = apply_adjustments_to_video(video, stack, max_workers=3)
    np.testing.assert_array_equal(seq, par)
