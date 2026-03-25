"""Default parameters for the adjustment stack widget."""

from __future__ import annotations

import copy

from ..color_tuner.defaults import (
    DEFAULT_BRIGHTNESS_CONTRAST,
    DEFAULT_LEVELS,
    DEFAULT_CURVES,
    DEFAULT_ADJUSTMENT_STACKS,
)


def default_adjustment_stack() -> list[dict]:
    """Default stack used when the user adds the first adjustment."""
    # Start with the same ordering you'd use in Photoshop:
    # Brightness/Contrast -> Levels -> Curves.
    return [
        copy.deepcopy(DEFAULT_BRIGHTNESS_CONTRAST),
        copy.deepcopy(DEFAULT_LEVELS),
        copy.deepcopy(DEFAULT_CURVES),
    ]


def default_adjustment_item(typ: str) -> dict:
    if typ == "brightness_contrast":
        return copy.deepcopy(DEFAULT_BRIGHTNESS_CONTRAST)
    if typ == "levels":
        return copy.deepcopy(DEFAULT_LEVELS)
    if typ == "curves":
        return copy.deepcopy(DEFAULT_CURVES)
    raise ValueError(f"Unknown adjustment type: {typ}")

