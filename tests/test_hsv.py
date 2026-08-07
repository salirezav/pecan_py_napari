"""Tests for HSV hue / saturation / value adjustment."""

from __future__ import annotations

import cv2
import numpy as np

from napari_pecan_py.widgets.color_adjustments.defaults import default_adjustment_item
from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video
from napari_pecan_py.widgets.color_adjustments.parallelism import FRAME_PARALLEL_TYPES, is_parallelizable
from napari_pecan_py.widgets.color_thresholding.defaults import DEFAULT_HSV
from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack, apply_hsv


def _solid_rgb(rgb=(180, 90, 40), size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def test_default_hsv_item():
    item = default_adjustment_item("hsv")
    assert item["type"] == "hsv"
    assert item["hue"] == 0
    assert item["saturation"] == 0
    assert item["value"] == 0
    assert item["type"] == DEFAULT_HSV["type"]


def test_hsv_is_frame_parallel():
    assert "hsv" in FRAME_PARALLEL_TYPES
    assert is_parallelizable({"type": "hsv", "enabled": True}) is True


def test_apply_hsv_identity_is_copy():
    img = _solid_rgb()
    out = apply_hsv(img, hue=0, saturation=0, value=0)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_apply_hsv_preserves_shape_dtype():
    img = _solid_rgb()
    out = apply_hsv(img, hue=40, saturation=30, value=-20)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_hue_shift_changes_color_but_keeps_rough_brightness():
    img = _solid_rgb((200, 80, 40))
    out = apply_hsv(img, hue=90, saturation=0, value=0)
    assert not np.array_equal(out, img)
    # Mean luminance should stay in a similar ballpark for a pure hue rotate.
    before = float(img.mean())
    after = float(out.mean())
    assert abs(after - before) < 40.0


def test_positive_saturation_increases_channel_spread():
    img = _solid_rgb((160, 100, 70))
    out = apply_hsv(img, hue=0, saturation=80, value=0)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    s0 = float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[0, 0, 1])
    s1 = float(cv2.cvtColor(out_bgr, cv2.COLOR_BGR2HSV)[0, 0, 1])
    assert s1 > s0


def test_negative_value_darkens():
    img = _solid_rgb((180, 120, 60))
    out = apply_hsv(img, hue=0, saturation=0, value=-50)
    assert float(out.mean()) < float(img.mean())


def test_stack_dispatch_hsv():
    img = _solid_rgb()
    stack = [
        {
            "type": "hsv",
            "enabled": True,
            "hue": 30,
            "saturation": 20,
            "value": -10,
        }
    ]
    out = apply_adjustment_stack(img, stack)
    expected = apply_hsv(img, hue=30, saturation=20, value=-10)
    np.testing.assert_array_equal(out, expected)


def test_apply_hsv_video_parallel_matches_sequential():
    rng = np.random.default_rng(0)
    video = rng.integers(0, 256, size=(6, 12, 12, 3), dtype=np.uint8)
    stack = [
        {
            "type": "hsv",
            "enabled": True,
            "hue": -45,
            "saturation": 25,
            "value": 10,
        }
    ]
    sequential = apply_adjustments_to_video(video, stack, max_workers=1)
    parallel = apply_adjustments_to_video(video, stack, max_workers=2)
    np.testing.assert_array_equal(sequential, parallel)
