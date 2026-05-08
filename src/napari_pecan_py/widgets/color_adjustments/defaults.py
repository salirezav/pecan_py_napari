"""Default parameters for the adjustment stack widget."""

from __future__ import annotations

import copy

from ..color_tuner.defaults import (
    DEFAULT_BRIGHTNESS_CONTRAST,
    DEFAULT_CURVES,
    DEFAULT_LEVELS,
    DEFAULT_NORMALIZATION,
    DEFAULT_SURFACE_BLUR,
)


def default_adjustment_stack() -> list[dict]:
    """Start with no adjustments; user builds stack explicitly."""
    return []


def default_adjustment_item(typ: str) -> dict:
    if typ == "brightness_contrast":
        return copy.deepcopy(DEFAULT_BRIGHTNESS_CONTRAST)
    if typ == "levels":
        return copy.deepcopy(DEFAULT_LEVELS)
    if typ == "curves":
        return copy.deepcopy(DEFAULT_CURVES)
    if typ == "surface_blur":
        return copy.deepcopy(DEFAULT_SURFACE_BLUR)
    if typ == "normalization":
        return copy.deepcopy(DEFAULT_NORMALIZATION)
    raise ValueError(f"Unknown adjustment type: {typ}")

