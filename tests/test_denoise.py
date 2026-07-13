"""Tests for spatial denoise adjustment."""

from __future__ import annotations

import numpy as np

from napari_pecan_py.widgets.color_adjustments.defaults import default_adjustment_item
from napari_pecan_py.widgets.color_adjustments.parallelism import FRAME_PARALLEL_TYPES, is_parallelizable
from napari_pecan_py.widgets.color_thresholding.defaults import DEFAULT_DENOISE
from napari_pecan_py.widgets.color_thresholding.denoise import apply_denoise
from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack


def _noisy_rgb(seed: int = 0, size: int = 32) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clean = np.full((size, size, 3), 128, dtype=np.uint8)
    clean[8:24, 8:24] = 200
    noise = rng.integers(-20, 21, size=clean.shape, dtype=np.int16)
    return np.clip(clean.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def test_default_denoise_item():
    item = default_adjustment_item("denoise")
    assert item["type"] == "denoise"
    assert item["method"] == "gaussian"
    assert item == DEFAULT_DENOISE or item["type"] == DEFAULT_DENOISE["type"]


def test_denoise_is_frame_parallel():
    assert "denoise" in FRAME_PARALLEL_TYPES
    assert is_parallelizable({"type": "denoise", "enabled": True}) is True


def test_apply_denoise_methods_preserve_shape_dtype():
    img = _noisy_rgb()
    methods = {
        "gaussian": {"ksize": 5, "sigma": 1.0},
        "median": {"ksize": 3},
        "bilateral": {"diameter": 5, "sigma_color": 50.0, "sigma_space": 50.0},
        "nlmeans": {"h": 6.0, "h_color": 6.0, "template_window": 5, "search_window": 11},
        "tv": {"weight": 0.05},
        "wavelet": {"sigma_wavelet": 0.0},
    }
    for method, params in methods.items():
        out = apply_denoise(img, method=method, params=params)
        assert out.shape == img.shape
        assert out.dtype == np.uint8


def test_gaussian_reduces_high_frequency_noise():
    img = _noisy_rgb(seed=2)
    out = apply_denoise(img, method="gaussian", params={"ksize": 5, "sigma": 1.5})
    # Local variance of differences should drop after smoothing.
    before = float(np.std(img.astype(np.float32)))
    after = float(np.std(out.astype(np.float32)))
    assert after <= before


def test_stack_dispatch_denoise():
    img = _noisy_rgb(seed=3)
    stack = [
        {
            "type": "denoise",
            "enabled": True,
            "method": "median",
            "ksize": 3,
        }
    ]
    out = apply_adjustment_stack(img, stack)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # Median on flat-ish noise should change some pixels.
    assert not np.array_equal(out, img)
