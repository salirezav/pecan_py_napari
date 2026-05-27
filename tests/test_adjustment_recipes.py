"""Tests for adjustment recipe helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_pecan_py.widgets.color_adjustments.recipes import (
    METADATA_KEY,
    AdjustmentRecipe,
    default_output_basename,
    discover_recipes_for_source,
    infer_recipe_from_layer,
    merge_recipes,
    metadata_from_recipe,
    recipe_from_metadata,
    unique_output_name,
    write_recipe_metadata,
)


class _FakeViewer:
    def __init__(self, layers):
        self.layers = layers


def test_unique_output_name_increments():
    layers = [
        SimpleNamespace(name="video - adjusted"),
        SimpleNamespace(name="video - adjusted [2]"),
    ]
    viewer = _FakeViewer(layers)
    assert unique_output_name(viewer, "video") == "video - adjusted [3]"


def test_merge_recipes_prefers_richer_stack():
    a = AdjustmentRecipe.new("src", "src - adjusted", adjustment_stack=[])
    b = AdjustmentRecipe.new(
        "src",
        "src - adjusted",
        adjustment_stack=[{"type": "levels", "enabled": True}],
        recipe_id=a.recipe_id,
    )
    merged = merge_recipes([a], [b])
    assert len(merged) == 1
    assert merged[0].adjustment_stack[0]["type"] == "levels"


def test_metadata_roundtrip():
    recipe = AdjustmentRecipe.new(
        "video",
        "video - adjusted",
        adjustment_stack=[{"type": "brightness_contrast", "enabled": True, "brightness": 1, "contrast": 2}],
    )
    meta = metadata_from_recipe(recipe)
    restored = recipe_from_metadata(meta)
    assert restored is not None
    assert restored.source_layer == "video"
    assert restored.output_layer_name == "video - adjusted"
    assert restored.adjustment_stack[0]["brightness"] == 1


def test_infer_recipe_from_layer_name():
    layer = SimpleNamespace(name="clip - adjusted [2]", metadata={})
    recipe = infer_recipe_from_layer(layer)
    assert recipe is not None
    assert recipe.source_layer == "clip"
    assert recipe.output_layer_name == "clip - adjusted [2]"


def test_discover_legacy_adjusted_suffix():
    layer = SimpleNamespace(
        name="vid - Adjusted",
        metadata={},
    )
    viewer = _FakeViewer([layer])
    recipes = discover_recipes_for_source(viewer, "vid")
    assert len(recipes) == 1
    assert recipes[0].output_layer_name == "vid - Adjusted"


def test_write_recipe_metadata():
    layer = SimpleNamespace(metadata={})
    recipe = AdjustmentRecipe.new("s", "s - adjusted")
    write_recipe_metadata(layer, recipe)
    assert METADATA_KEY in layer.metadata
    assert layer.metadata[METADATA_KEY]["source_layer"] == "s"
