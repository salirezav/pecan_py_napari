"""Tests for adjustment recipe helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from napari_pecan_py.widgets.color_adjustments.recipes import (
    METADATA_KEY,
    AdjustmentRecipe,
    compose_output_name,
    default_output_basename,
    discover_recipes_for_source,
    infer_recipe_from_layer,
    merge_recipes,
    metadata_from_recipe,
    normalize_output_suffix,
    output_suffix_from_name,
    recipe_from_metadata,
    rename_recipe_output,
    unique_output_name,
    write_recipe_metadata,
)
from napari_pecan_py.widgets.pipeline_recorder.state import (
    PIPELINE_STORE,
    PipelineStep,
    rename_layer_references,
)


class _FakeViewer:
    def __init__(self, layers):
        self.layers = layers


class _Layers(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for item in self:
                if item.name == key:
                    return item
            raise KeyError(key)
        return list.__getitem__(self, key)


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


def test_output_suffix_helpers():
    assert output_suffix_from_name("cam", "cam - adjusted") == "adjusted"
    assert output_suffix_from_name("cam", "cam - Surface Blur applied") == "Surface Blur applied"
    assert normalize_output_suffix(" - glare") == "glare"
    assert compose_output_name("cam", "Surface Blur applied") == "cam - Surface Blur applied"
    with pytest.raises(ValueError):
        compose_output_name("cam", "   ")


def test_rename_recipe_output_updates_layer_and_metadata():
    layer = SimpleNamespace(name="video - adjusted", metadata={})
    viewer = _FakeViewer(_Layers([layer]))
    recipe = AdjustmentRecipe.new("video", "video - adjusted", adjustment_stack=[{"type": "levels"}])
    write_recipe_metadata(layer, recipe)

    new_name = rename_recipe_output(viewer, recipe, "video - Surface Blur applied")
    assert new_name == "video - Surface Blur applied"
    assert recipe.output_layer_name == new_name
    assert layer.name == new_name
    assert layer.metadata[METADATA_KEY]["output_layer_name"] == new_name


def test_rename_recipe_output_rejects_collision():
    other = SimpleNamespace(name="video - glare", metadata={})
    layer = SimpleNamespace(name="video - adjusted", metadata={})
    viewer = _FakeViewer(_Layers([layer, other]))
    recipe = AdjustmentRecipe.new("video", "video - adjusted")
    with pytest.raises(ValueError, match="already in use"):
        rename_recipe_output(viewer, recipe, "video - glare")


def test_rename_layer_references_updates_pipeline_and_spares_indexed():
    PIPELINE_STORE.clear()
    PIPELINE_STORE.set_steps(
        [
            PipelineStep(
                kind="color_adjustments.stack",
                description="Adjustments: video - adjusted",
                params={
                    "source_layer": "video",
                    "output_layer": "video - adjusted",
                    "adjustment_stack": [],
                },
            ),
            PipelineStep(
                kind="color_thresholding.threshold",
                description="Threshold on video - adjusted",
                params={"source_layer": "video - adjusted", "target": "new"},
            ),
            PipelineStep(
                kind="color_adjustments.stack",
                description="Adjustments: video - adjusted [2]",
                params={
                    "source_layer": "video",
                    "output_layer": "video - adjusted [2]",
                    "adjustment_stack": [],
                },
            ),
        ]
    )
    changed = rename_layer_references("video - adjusted", "video - Surface Blur applied")
    assert changed == 2
    steps = PIPELINE_STORE.steps
    assert steps[0].params["output_layer"] == "video - Surface Blur applied"
    assert steps[0].description == "Adjustments: video - Surface Blur applied"
    assert steps[1].params["source_layer"] == "video - Surface Blur applied"
    assert steps[2].params["output_layer"] == "video - adjusted [2]"
    assert steps[2].description == "Adjustments: video - adjusted [2]"
    PIPELINE_STORE.clear()


def test_default_output_basename():
    assert default_output_basename("x") == "x - adjusted"
