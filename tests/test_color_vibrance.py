"""Tests for Photoshop-like Color / Vibrance adjustment."""

from __future__ import annotations

import cv2
import numpy as np

from napari_pecan_py.widgets.color_adjustments.defaults import default_adjustment_item
from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video
from napari_pecan_py.widgets.color_adjustments.parallelism import FRAME_PARALLEL_TYPES, is_parallelizable
from napari_pecan_py.widgets.color_thresholding.defaults import DEFAULT_COLOR_VIBRANCE
from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack, apply_color_vibrance


def _solid_rgb(rgb=(160, 110, 70), size: int = 16) -> np.ndarray:
    return np.full((size, size, 3), rgb, dtype=np.uint8)


def _mean_hsv_s(img: np.ndarray) -> float:
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[..., 1].mean())


def test_default_color_vibrance_item():
    item = default_adjustment_item("color_vibrance")
    assert item["type"] == "color_vibrance"
    assert item["temperature"] == 0
    assert item["tint"] == 0
    assert item["vibrance"] == 0
    assert item["saturation"] == 0
    assert item["use_legacy"] is False
    assert item["type"] == DEFAULT_COLOR_VIBRANCE["type"]


def test_color_vibrance_is_frame_parallel():
    assert "color_vibrance" in FRAME_PARALLEL_TYPES
    assert is_parallelizable({"type": "color_vibrance", "enabled": True}) is True


def test_apply_color_vibrance_identity_is_copy():
    img = _solid_rgb()
    out = apply_color_vibrance(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out, img)


def test_temperature_warms_and_cools():
    img = _solid_rgb((128, 128, 128))
    warm = apply_color_vibrance(img, temperature=60)
    cool = apply_color_vibrance(img, temperature=-60)
    # Warm: more red relative to blue; cool: opposite.
    warm_rb = int(warm[0, 0, 0]) - int(warm[0, 0, 2])
    cool_rb = int(cool[0, 0, 0]) - int(cool[0, 0, 2])
    base_rb = int(img[0, 0, 0]) - int(img[0, 0, 2])
    assert warm_rb > base_rb
    assert cool_rb < base_rb


def test_tint_magenta_vs_green():
    img = _solid_rgb((128, 128, 128))
    magenta = apply_color_vibrance(img, tint=60)
    green = apply_color_vibrance(img, tint=-60)
    assert float(magenta[0, 0, 1]) < float(img[0, 0, 1])
    assert float(green[0, 0, 1]) > float(img[0, 0, 1])


def test_positive_vibrance_increases_saturation():
    img = _solid_rgb((150, 120, 90))
    out = apply_color_vibrance(img, vibrance=70)
    assert _mean_hsv_s(out) > _mean_hsv_s(img)


def test_positive_saturation_increases_saturation():
    img = _solid_rgb((150, 120, 90))
    out = apply_color_vibrance(img, saturation=70)
    assert _mean_hsv_s(out) > _mean_hsv_s(img)


def test_legacy_flag_changes_skin_protection():
    # Warm mid-sat color near skin hue band.
    img = _solid_rgb((190, 140, 110))
    modern = apply_color_vibrance(img, vibrance=80, use_legacy=False)
    legacy = apply_color_vibrance(img, vibrance=80, use_legacy=True)
    # Legacy should allow a stronger saturation push on skin-ish hues.
    assert _mean_hsv_s(legacy) >= _mean_hsv_s(modern)


def test_stack_dispatch_color_vibrance():
    img = _solid_rgb()
    stack = [
        {
            "type": "color_vibrance",
            "enabled": True,
            "temperature": -40,
            "tint": 10,
            "vibrance": 25,
            "saturation": 5,
            "use_legacy": False,
        }
    ]
    out = apply_adjustment_stack(img, stack)
    expected = apply_color_vibrance(
        img,
        temperature=-40,
        tint=10,
        vibrance=25,
        saturation=5,
        use_legacy=False,
    )
    np.testing.assert_array_equal(out, expected)


def test_apply_color_vibrance_video_parallel_matches_sequential():
    rng = np.random.default_rng(1)
    video = rng.integers(0, 256, size=(6, 12, 12, 3), dtype=np.uint8)
    stack = [
        {
            "type": "color_vibrance",
            "enabled": True,
            "temperature": 20,
            "tint": -15,
            "vibrance": 30,
            "saturation": -10,
            "use_legacy": True,
        }
    ]
    sequential = apply_adjustments_to_video(video, stack, max_workers=1)
    parallel = apply_adjustments_to_video(video, stack, max_workers=2)
    np.testing.assert_array_equal(sequential, parallel)
