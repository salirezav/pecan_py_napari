"""Default parameters for the adjustment stack widget."""

from __future__ import annotations

import copy

from ..color_thresholding.defaults import (
    DEFAULT_BRIGHTNESS_CONTRAST,
    DEFAULT_CURVES,
    DEFAULT_LEVELS,
    DEFAULT_MASK_LARGEST_COMPONENT,
    DEFAULT_MASK_MORPHOLOGY,
    DEFAULT_MOTION_MASK_THRESHOLD,
    DEFAULT_NORMALIZATION,
    DEFAULT_SURFACE_BLUR,
    DEFAULT_TEMPORAL_MEDIAN_DIFF,
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
    if typ == "temporal_median_diff":
        return copy.deepcopy(DEFAULT_TEMPORAL_MEDIAN_DIFF)
    if typ == "motion_mask_threshold":
        return copy.deepcopy(DEFAULT_MOTION_MASK_THRESHOLD)
    if typ == "mask_morphology":
        return copy.deepcopy(DEFAULT_MASK_MORPHOLOGY)
    if typ == "mask_largest_component":
        return copy.deepcopy(DEFAULT_MASK_LARGEST_COMPONENT)
    raise ValueError(f"Unknown adjustment type: {typ}")

