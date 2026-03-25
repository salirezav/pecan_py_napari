"""Default thresholds and color space parameters (from pecan_py ColorTuner)."""

from __future__ import annotations

import copy

import cv2
import numpy as np

TARGETS = ("pecan", "kernel", "damaged_kernel", "crack", "background")
COLOR_SPACES = ("rgb", "hsv", "lab")

# ---- Color adjustment presets (Photoshop-like) -----------------------------
# These operate on the RGB image before converting into HSV/LAB for thresholding.
#
# The defaults are based on the values currently shown in your screenshots:
# - Brightness/Contrast: brightness=-32, contrast=77
# - Levels: input black=0, gamma=0.08, input white=214 (output levels fixed at 0..255)
# - Curves: RGB curves represented by 4 control points (x fixed, y editable).
#
# Note: Curves are inherently more flexible than Levels/Brightness/Contrast; if you
# want the exact curve to match, adjust the 4 "y" points in the widget once.
DEFAULT_BRIGHTNESS_CONTRAST = {
    "type": "brightness_contrast",
    "brightness": -32,
    "contrast": 77,
}

DEFAULT_LEVELS = {
    "type": "levels",
    "in_min": 0,
    "gamma": 0.08,
    "in_max": 214,
    "out_min": 0,
    "out_max": 255,
}

# Curves: fixed x points with editable y points.
_CURVES_X = [0, 64, 128, 255]
DEFAULT_CURVES = {
    "type": "curves",
    "x_points": _CURVES_X,
    # These y defaults are a reasonable crack-highlighting starting point.
    # Tune in-widget if needed.
    "y_points": [0, 70, 200, 255],
}

DEFAULT_ADJUSTMENT_STACKS = {
    # Kernel: brightness/contrast then levels
    "kernel": [DEFAULT_BRIGHTNESS_CONTRAST, DEFAULT_LEVELS],
    "damaged_kernel": [DEFAULT_BRIGHTNESS_CONTRAST, DEFAULT_LEVELS],
    # Crack: brightness/contrast then curves
    "crack": [DEFAULT_BRIGHTNESS_CONTRAST, DEFAULT_CURVES],
    # Pecan/background: no adjustments by default
    "pecan": [],
    "background": [],
}

# Default thresholds per color space and target (same structure as pecan_py ColorTuner)
DEFAULT_THRESHOLDS = {
    "rgb": {
        "pecan": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([50, 50, 50], dtype=np.uint8)},
        "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "damaged_kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
    },
    "hsv": {
        "pecan": {"lower": np.array([0, 84, 80], dtype=np.uint8), "upper": np.array([82, 255, 255], dtype=np.uint8)},
        "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([179, 255, 255], dtype=np.uint8)},
        "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([179, 255, 50], dtype=np.uint8)},
        "background": {"lower": np.array([0, 0, 60], dtype=np.uint8), "upper": np.array([179, 91, 255], dtype=np.uint8)},
        "damaged_kernel": {"lower": np.array([18, 63, 240], dtype=np.uint8), "upper": np.array([31, 138, 255], dtype=np.uint8)},
    },
    "lab": {
        "pecan": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "crack": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([100, 128, 128], dtype=np.uint8)},
        "background": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
        "damaged_kernel": {"lower": np.array([0, 0, 0], dtype=np.uint8), "upper": np.array([255, 255, 255], dtype=np.uint8)},
    },
}

# Channels, max values, and OpenCV conversion (BGR input)
COLOR_SPACE_PARAMS = {
    "rgb": {"channels": ["R", "G", "B"], "max_values": [255, 255, 255], "conversion_code": None},
    "hsv": {"channels": ["H", "S", "V"], "max_values": [179, 255, 255], "conversion_code": cv2.COLOR_BGR2HSV},
    "lab": {"channels": ["L", "A", "B"], "max_values": [255, 255, 255], "conversion_code": cv2.COLOR_BGR2LAB},
}

# BGR colors for composite overlay per target (pecan_py uses RGB tuple; we store BGR for cv2)
MASK_COLORS = {
    "pecan": (255, 0, 0),           # Blue
    "kernel": (0, 255, 0),          # Green
    "damaged_kernel": (0, 255, 255),  # Yellow
    "crack": (0, 0, 255),           # Red
    "background": (255, 255, 0),   # Cyan (RGB 255,255,0 -> BGR 0,255,255 would be yellow; cyan BGR = 255,255,0)
}


def copy_default_thresholds():
    """Return a deep copy of DEFAULT_THRESHOLDS for mutable state."""
    return copy.deepcopy(DEFAULT_THRESHOLDS)


def copy_default_adjustment_stacks():
    """Return a deep copy of DEFAULT_ADJUSTMENT_STACKS for mutable state."""
    return copy.deepcopy(DEFAULT_ADJUSTMENT_STACKS)
