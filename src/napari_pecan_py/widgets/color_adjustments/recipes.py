"""Adjustment recipes: multiple stacks per source layer with named outputs."""

from __future__ import annotations

import copy
import re
import uuid
from dataclasses import dataclass
from typing import Any

METADATA_KEY = "pecan_adjustments"
OUTPUT_SUFFIX = "adjusted"
_LEGACY_SUFFIX = "Adjusted"

_INDEXED_SUFFIX_RE = re.compile(r"^(.+) \[(\d+)\]$")


@dataclass
class AdjustmentRecipe:
    """One adjustment pipeline from a source layer to a dedicated output layer."""

    recipe_id: str
    source_layer: str
    output_layer_name: str
    adjustment_stack: list[dict]

    @classmethod
    def new(
        cls,
        source_layer: str,
        output_layer_name: str,
        *,
        adjustment_stack: list[dict] | None = None,
        recipe_id: str | None = None,
    ) -> AdjustmentRecipe:
        return cls(
            recipe_id=recipe_id or str(uuid.uuid4()),
            source_layer=source_layer,
            output_layer_name=output_layer_name,
            adjustment_stack=copy.deepcopy(adjustment_stack) if adjustment_stack is not None else [],
        )


def default_output_basename(source_name: str) -> str:
    return f"{source_name} - {OUTPUT_SUFFIX}"


def _legacy_output_basename(source_name: str) -> str:
    return f"{source_name} - {_LEGACY_SUFFIX}"


def _layer_names(viewer: Any) -> set[str]:
    return {str(getattr(layer, "name", "")) for layer in viewer.layers}


def unique_output_name(viewer: Any, source_name: str) -> str:
    """Return ``{source} - adjusted`` or ``{source} - adjusted [N]`` if taken."""
    base = default_output_basename(source_name)
    existing = _layer_names(viewer)
    if base not in existing:
        return base
    n = 2
    while f"{base} [{n}]" in existing:
        n += 1
    return f"{base} [{n}]"


def _is_output_name_for_source(layer_name: str, source_name: str) -> bool:
    base = default_output_basename(source_name)
    legacy = _legacy_output_basename(source_name)
    if layer_name in (base, legacy):
        return True
    m = _INDEXED_SUFFIX_RE.match(layer_name)
    if m and m.group(1) in (base, legacy):
        return True
    return False


def metadata_from_recipe(recipe: AdjustmentRecipe) -> dict:
    return {
        "recipe_id": recipe.recipe_id,
        "source_layer": recipe.source_layer,
        "output_layer_name": recipe.output_layer_name,
        "adjustment_stack": copy.deepcopy(recipe.adjustment_stack),
    }


def recipe_from_metadata(meta: Any) -> AdjustmentRecipe | None:
    if not isinstance(meta, dict):
        return None
    src = meta.get("source_layer")
    out = meta.get("output_layer_name")
    if not src or not out:
        return None
    stack = meta.get("adjustment_stack")
    return AdjustmentRecipe(
        recipe_id=str(meta.get("recipe_id") or uuid.uuid4()),
        source_layer=str(src),
        output_layer_name=str(out),
        adjustment_stack=[dict(x) for x in (stack or [])],
    )


def read_recipe_from_layer(layer: Any) -> AdjustmentRecipe | None:
    md = getattr(layer, "metadata", None) or {}
    pecan = md.get(METADATA_KEY)
    if pecan is not None:
        recipe = recipe_from_metadata(pecan)
        if recipe is not None:
            return recipe
    name = str(getattr(layer, "name", ""))
    if not name:
        return None
    return None


def write_recipe_metadata(layer: Any, recipe: AdjustmentRecipe) -> None:
    md = dict(getattr(layer, "metadata", None) or {})
    md[METADATA_KEY] = metadata_from_recipe(recipe)
    layer.metadata = md


def _stable_recipe_id(output_layer_name: str) -> str:
    return f"pecan:output:{output_layer_name}"


def merge_recipes(*groups: list[AdjustmentRecipe]) -> list[AdjustmentRecipe]:
    """Merge recipe lists; prefer non-empty stacks when output names collide."""
    by_output: dict[str, AdjustmentRecipe] = {}
    for group in groups:
        for recipe in group:
            key = recipe.output_layer_name.lower()
            prev = by_output.get(key)
            if prev is None:
                by_output[key] = recipe
                continue
            if len(recipe.adjustment_stack) >= len(prev.adjustment_stack):
                by_output[key] = recipe
    return sorted(by_output.values(), key=lambda r: r.output_layer_name.lower())


def discover_recipes_for_source(viewer: Any, source_name: str) -> list[AdjustmentRecipe]:
    """Collect recipes from output-layer metadata and known output naming patterns."""
    found: dict[str, AdjustmentRecipe] = {}
    for layer in viewer.layers:
        if getattr(layer, "name", None) is None:
            continue
        recipe = read_recipe_from_layer(layer)
        if recipe is not None and recipe.source_layer == source_name:
            found[recipe.recipe_id] = recipe
            continue
        name = str(getattr(layer, "name", ""))
        if not _is_output_name_for_source(name, source_name):
            continue
        inferred = AdjustmentRecipe.new(source_name, name, recipe_id=_stable_recipe_id(name))
        found[inferred.recipe_id] = inferred

    return sorted(found.values(), key=lambda r: r.output_layer_name.lower())


def infer_recipe_from_layer(layer: Any) -> AdjustmentRecipe | None:
    """Return a recipe for *layer* if it is a pecan adjustment output (metadata or naming)."""
    recipe = read_recipe_from_layer(layer)
    if recipe is not None:
        return recipe
    name = str(getattr(layer, "name", ""))
    if " - " not in name:
        return None
    # Heuristic: "{source} - adjusted" or legacy "{source} - Adjusted" (+ optional [N])
    for suffix in (OUTPUT_SUFFIX, _LEGACY_SUFFIX):
        marker = f" - {suffix}"
        if marker not in name:
            continue
        source_name = name.split(marker, 1)[0]
        return AdjustmentRecipe.new(source_name, name, recipe_id=_stable_recipe_id(name))
    return None
