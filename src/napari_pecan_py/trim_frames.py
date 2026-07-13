"""Trim time/frame axis of selected layers from the layer-list context menu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from napari.components import LayerList
    from napari.layers import Layer

_ACTION_ID = "napari-pecan-py.trim_frames"
_registered = False


def frame_axis_and_count(layer: Layer) -> tuple[int, int] | None:
    """Return ``(axis, n_frames)`` for a layer that can be trimmed, else ``None``.

    Videos and stacks in this plugin use axis 0 as time. RGB images need at
    least 4 dims ``(T, Y, X, C)``; non-RGB Image/Labels need ``ndim >= 3``.
    """
    data = getattr(layer, "data", None)
    if data is None:
        return None

    ndim = int(getattr(data, "ndim", 0))
    shape = tuple(getattr(data, "shape", ()))
    if not shape:
        return None

    is_rgb = bool(getattr(layer, "rgb", False))
    layer_type = getattr(layer, "_type_string", None)
    if layer_type not in {"image", "labels"}:
        return None

    if is_rgb:
        if ndim < 4:
            return None
    elif ndim < 3:
        return None

    return 0, int(shape[0])


def trim_array(data: Any, start: int, end: int, *, axis: int = 0) -> Any:
    """Return ``data`` sliced to inclusive ``[start, end]`` along ``axis``."""
    n = int(np.asarray(getattr(data, "shape"))[axis])
    if start < 0 or end < start or end >= n:
        raise ValueError(
            f"Invalid frame range [{start}, {end}] for axis length {n}"
        )

    slicer = [slice(None)] * int(getattr(data, "ndim"))
    slicer[axis] = slice(start, end + 1)
    return data[tuple(slicer)]


def _sync_viewer_dims_after_shape_change(layer: Layer) -> None:
    """Clear napari's stale extent cache and shrink the dims slider.

    Assigning ``layer.data`` to a shorter stack updates ``layer._extent_data``
    but leaves the cached ``layer.extent`` (and therefore ``viewer.dims``)
    at the old length, so scrubbing past the trimmed end shows blank frames.
    """
    clear = getattr(layer, "_clear_extent", None)
    if callable(clear):
        clear()

    viewer = None
    try:
        import napari

        viewer = napari.current_viewer()
    except Exception:
        viewer = None

    if viewer is None:
        return

    on_change = getattr(viewer, "_on_layers_change", None)
    if callable(on_change):
        on_change()

    # Keep the playhead inside the new frame range.
    dims = getattr(viewer, "dims", None)
    if dims is None:
        return
    try:
        current = list(dims.current_step)
        nsteps = list(dims.nsteps)
        for i, n in enumerate(nsteps):
            if n <= 0:
                continue
            current[i] = min(int(current[i]), int(n) - 1)
        dims.current_step = tuple(current)
    except Exception:
        pass


def apply_trim_to_layer(layer: Layer, start: int, end: int, *, axis: int = 0) -> int:
    """Trim ``layer`` in place to inclusive ``[start, end]``; return new frame count."""
    trimmed = trim_array(layer.data, start, end, axis=axis)
    # Materialize lazy video adapters so the layer owns a dense, shorter array.
    trimmed = np.asarray(trimmed)
    layer.data = trimmed
    _sync_viewer_dims_after_shape_change(layer)
    return int(np.asarray(trimmed.shape)[axis])


def _prompt_frame_range(
    n_frames: int,
    *,
    layer_name: str,
    parent=None,
) -> tuple[int, int] | None:
    """Show a dialog asking for inclusive start/end frames. Return None if cancelled."""
    from qtpy.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLabel,
        QSpinBox,
        QVBoxLayout,
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle("Trim frames")
    layout = QVBoxLayout(dlg)
    layout.addWidget(
        QLabel(
            f"Layer: {layer_name}\n"
            f"Frames: 0 … {n_frames - 1} ({n_frames} total)\n\n"
            "Keep an inclusive frame range (0-based, matching the napari slider):"
        )
    )

    form = QFormLayout()
    start_box = QSpinBox()
    start_box.setRange(0, max(0, n_frames - 1))
    start_box.setValue(0)
    end_box = QSpinBox()
    end_box.setRange(0, max(0, n_frames - 1))
    end_box.setValue(max(0, n_frames - 1))
    form.addRow("Start frame", start_box)
    form.addRow("End frame", end_box)
    layout.addLayout(form)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    def _sync_bounds() -> None:
        start = start_box.value()
        if end_box.value() < start:
            end_box.setValue(start)
        end_box.setMinimum(start)

    start_box.valueChanged.connect(_sync_bounds)

    if dlg.exec() != QDialog.Accepted:
        return None

    start = int(start_box.value())
    end = int(end_box.value())
    if end < start:
        return None
    return start, end


def trim_selected_layers(ll: LayerList) -> None:
    """Prompt for a frame range and trim selected Image/Labels layers in place."""
    from napari.utils.notifications import show_info, show_warning

    selected = list(ll.selection)
    if not selected:
        show_warning("Select a layer to trim.")
        return

    trimmable: list[tuple[Layer, int, int]] = []
    for layer in selected:
        info = frame_axis_and_count(layer)
        if info is not None and info[1] > 1:
            trimmable.append((layer, info[0], info[1]))

    if not trimmable:
        show_warning(
            "Selected layer(s) have no trimable time/frame axis "
            "(need Image/Labels with multiple frames)."
        )
        return

    # Use the active layer when possible, else the first trimmable selection.
    active = ll.selection.active
    primary = next(
        (t for t in trimmable if t[0] is active),
        trimmable[0],
    )
    primary_layer, _axis, n_frames = primary

    parent = None
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        parent = app.activeWindow() if app is not None else None
    except Exception:
        parent = None

    result = _prompt_frame_range(
        n_frames, layer_name=str(primary_layer.name), parent=parent
    )
    if result is None:
        return
    start, end = result

    if start == 0 and end == n_frames - 1 and len(trimmable) == 1:
        show_info("Nothing to trim — full frame range already selected.")
        return

    trimmed_names: list[str] = []
    for layer, axis, layer_n in trimmable:
        layer_end = min(end, layer_n - 1)
        if start > layer_end:
            show_warning(
                f"Skipped {layer.name}: start frame {start} is past last frame "
                f"{layer_n - 1}."
            )
            continue
        if start == 0 and layer_end == layer_n - 1:
            continue
        try:
            new_n = apply_trim_to_layer(layer, start, layer_end, axis=axis)
        except Exception as exc:
            show_warning(f"Failed to trim {layer.name}: {exc}")
            continue
        trimmed_names.append(
            f"{layer.name} [{start}:{layer_end}] → {new_n} frames"
        )

    if trimmed_names:
        show_info("Trimmed:\n" + "\n".join(trimmed_names))
    else:
        show_info("Nothing to trim.")


def register_trim_frames_action() -> None:
    """Register Trim frames… on the layer-list context menu (idempotent).

    Current napari versions exclude ``napari/layers/context`` from npe2's
    contributable-menu whitelist, so this registers the Action directly.
    """
    global _registered
    if _registered:
        return

    try:
        from app_model.types import Action, MenuRule
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
    except ImportError:
        return

    app = get_app_model()
    if _ACTION_ID in app.commands:
        _registered = True
        return

    app.register_action(
        Action(
            id=_ACTION_ID,
            title="Trim frames…",
            tooltip="Keep only a start–end frame range on the selected layer(s)",
            callback=trim_selected_layers,
            menus=[
                MenuRule(
                    id=MenuId.LAYERLIST_CONTEXT,
                    group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                    order=20,
                )
            ],
            enablement=LLSCK.num_selected_layers >= 1,
        )
    )
    _registered = True
