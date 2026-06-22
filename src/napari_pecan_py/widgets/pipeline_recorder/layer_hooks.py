"""Record napari layer-list actions (duplicate, convert to labels) into the pipeline."""

from __future__ import annotations

import weakref

from napari.layers import Image, Labels, Shapes

from .state import (
    is_pipeline_applying,
    is_recent_pipeline_record,
    record_pipeline_step,
)


def _source_parent(layer):
    source = getattr(layer, "source", None)
    if source is None:
        return None
    parent = getattr(source, "parent", None)
    if parent is None:
        return None
    try:
        return parent() if callable(parent) else parent
    except TypeError:
        return parent


class LayerPipelineHooks:
    def __init__(self, viewer) -> None:
        self._viewer = viewer
        self._selection_before: list = []
        self._selection_indices: dict[int, int] = {}
        self._connections: list = []

    def install(self) -> None:
        ll = self._viewer.layers
        self._connections.append(ll.events.inserting.connect(self._on_inserting))
        self._connections.append(ll.events.inserted.connect(self._on_inserted))

    def uninstall(self) -> None:
        for conn in self._connections:
            try:
                conn.disconnect()
            except Exception:
                pass
        self._connections.clear()

    def _on_inserting(self, _event) -> None:
        try:
            self._selection_before = list(self._viewer.layers.selection)
            self._selection_indices = {}
            for sel in self._selection_before:
                try:
                    self._selection_indices[id(sel)] = int(self._viewer.layers.index(sel))
                except Exception:
                    continue
        except Exception:
            self._selection_before = []
            self._selection_indices = {}

    def _on_inserted(self, event) -> None:
        if is_pipeline_applying() or is_recent_pipeline_record():
            return
        layer = event.value
        parent = _source_parent(layer)
        if parent is not None:
            record_pipeline_step(
                kind="layer.duplicate",
                description=f"Duplicate layer {parent.name}",
                params={
                    "source_layer": str(parent.name),
                    "output_layer": str(layer.name),
                },
            )
            return

        if not isinstance(layer, Labels):
            return

        insert_index = getattr(event, "index", None)
        new_name = str(layer.name)
        for sel in self._selection_before:
            if sel is layer or not isinstance(sel, (Image, Shapes)):
                continue
            src_name = str(sel.name)
            src_index = self._selection_indices.get(id(sel))
            if insert_index is not None and src_index is not None:
                if int(insert_index) not in (src_index, src_index + 1):
                    continue
            if new_name == src_name:
                record_pipeline_step(
                    kind="layer.convert_to_labels",
                    description=f"Convert {src_name} to Labels",
                    params={
                        "source_layer": src_name,
                        "output_layer": new_name,
                    },
                )
                return
            if isinstance(sel, Shapes) and new_name.startswith(src_name):
                record_pipeline_step(
                    kind="layer.convert_to_labels",
                    description=f"Convert {src_name} to Labels",
                    params={
                        "source_layer": src_name,
                        "output_layer": new_name,
                    },
                )
                return


_INSTALLED: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def ensure_layer_pipeline_hooks(viewer) -> LayerPipelineHooks:
    hooks = _INSTALLED.get(viewer)
    if hooks is not None:
        return hooks
    hooks = LayerPipelineHooks(viewer)
    hooks.install()
    _INSTALLED[viewer] = hooks
    return hooks
