"""Tests for Photoshop-like Levels, including per-channel R/G/B."""

from __future__ import annotations

import copy

import numpy as np

from napari_pecan_py.widgets.color_adjustments.defaults import default_adjustment_item
from napari_pecan_py.widgets.color_adjustments.logic import apply_adjustments_to_video
from napari_pecan_py.widgets.color_adjustments.widget import ColorAdjustmentsWidget
from napari_pecan_py.widgets.color_thresholding.defaults import DEFAULT_LEVELS, IDENTITY_LEVELS_PARAMS
from napari_pecan_py.widgets.color_thresholding.logic import apply_adjustment_stack, apply_levels


def _rgb_frame() -> np.ndarray:
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    img[..., 0] = 80
    img[..., 1] = 140
    img[..., 2] = 200
    return img


def test_default_levels_has_channels():
    item = default_adjustment_item("levels")
    assert item["type"] == "levels"
    assert "channels" in item
    assert set(item["channels"]) == {"r", "g", "b"}
    for ch in item["channels"].values():
        assert ch == IDENTITY_LEVELS_PARAMS or (
            ch["in_min"] == 0 and ch["gamma"] == 1.0 and ch["in_max"] == 255
        )
    assert item["active_channel"] == "rgb"


def test_legacy_levels_without_channels_still_applies():
    img = _rgb_frame()
    legacy = {
        "type": "levels",
        "enabled": True,
        "in_min": 0,
        "gamma": 1.0,
        "in_max": 255,
        "out_min": 0,
        "out_max": 255,
    }
    out = apply_adjustment_stack(img, [legacy])
    np.testing.assert_array_equal(out, img)


def test_master_levels_affects_all_channels():
    img = _rgb_frame()
    # Crush whites: in_max=128 maps mid values up.
    out = apply_levels(img, in_min=0, gamma=1.0, in_max=128, out_min=0, out_max=255)
    assert out.dtype == np.uint8
    assert float(out.mean()) > float(img.mean())


def test_per_channel_levels_only_changes_that_channel():
    img = _rgb_frame()
    channels = {
        "r": {"in_min": 0, "gamma": 1.0, "in_max": 40, "out_min": 0, "out_max": 255},
        "g": dict(IDENTITY_LEVELS_PARAMS),
        "b": dict(IDENTITY_LEVELS_PARAMS),
    }
    out = apply_levels(
        img,
        in_min=0,
        gamma=1.0,
        in_max=255,
        out_min=0,
        out_max=255,
        channels=channels,
    )
    # Red should brighten (in_max crush); G/B unchanged under identity master+channel.
    assert int(out[0, 0, 0]) > int(img[0, 0, 0])
    np.testing.assert_array_equal(out[..., 1], img[..., 1])
    np.testing.assert_array_equal(out[..., 2], img[..., 2])


def test_stack_dispatch_per_channel_levels():
    img = _rgb_frame()
    stack = [
        {
            "type": "levels",
            "enabled": True,
            "in_min": 0,
            "gamma": 1.0,
            "in_max": 255,
            "out_min": 0,
            "out_max": 255,
            "channels": {
                "r": dict(IDENTITY_LEVELS_PARAMS),
                "g": {"in_min": 0, "gamma": 1.0, "in_max": 70, "out_min": 0, "out_max": 255},
                "b": dict(IDENTITY_LEVELS_PARAMS),
            },
        }
    ]
    out = apply_adjustment_stack(img, stack)
    assert int(out[0, 0, 1]) > int(img[0, 0, 1])
    np.testing.assert_array_equal(out[..., 0], img[..., 0])
    np.testing.assert_array_equal(out[..., 2], img[..., 2])


def test_ensure_levels_channels_migrates_legacy():
    adj = {"type": "levels", "in_min": 10, "gamma": 1.2, "in_max": 200, "out_min": 0, "out_max": 255}
    ColorAdjustmentsWidget._ensure_levels_channels(adj)
    assert set(adj["channels"]) == {"r", "g", "b"}
    assert adj["active_channel"] == "rgb"
    assert adj["channels"]["r"]["gamma"] == 1.0


def test_levels_params_for_channel_reads_nested():
    adj = copy.deepcopy(DEFAULT_LEVELS)
    adj["channels"]["b"] = {"in_min": 5, "gamma": 0.72, "in_max": 255, "out_min": 0, "out_max": 255}
    rgb = ColorAdjustmentsWidget._levels_params_for_channel(adj, "rgb")
    blue = ColorAdjustmentsWidget._levels_params_for_channel(adj, "b")
    assert rgb["gamma"] == adj["gamma"]
    assert blue["gamma"] == 0.72
    assert blue["in_min"] == 5


def test_apply_levels_video_parallel_matches_sequential():
    rng = np.random.default_rng(2)
    video = rng.integers(0, 256, size=(5, 10, 10, 3), dtype=np.uint8)
    stack = [
        {
            "type": "levels",
            "enabled": True,
            "in_min": 5,
            "gamma": 0.9,
            "in_max": 240,
            "out_min": 0,
            "out_max": 255,
            "channels": {
                "r": {"in_min": 0, "gamma": 1.1, "in_max": 250, "out_min": 0, "out_max": 255},
                "g": dict(IDENTITY_LEVELS_PARAMS),
                "b": {"in_min": 10, "gamma": 1.0, "in_max": 255, "out_min": 0, "out_max": 255},
            },
        }
    ]
    sequential = apply_adjustments_to_video(video, stack, max_workers=1)
    parallel = apply_adjustments_to_video(video, stack, max_workers=2)
    np.testing.assert_array_equal(sequential, parallel)
