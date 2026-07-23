"""Trim time/frame axis of selected layers from the layer-list context menu."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np

if TYPE_CHECKING:
    from napari.components import LayerList
    from napari.layers import Layer

_ACTION_ID = "napari-pecan-py.trim_frames"
_registered = False

TrimMode = Literal["range", "sample"]


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


def sample_array(
    data: Any,
    start: int,
    step: int,
    count: int,
    *,
    axis: int = 0,
) -> Any:
    """Return every ``step``-th frame starting at ``start``, up to ``count`` frames."""
    from napari_pecan_py.video_meta import compute_sample_indices

    n = int(np.asarray(getattr(data, "shape"))[axis])
    indices = compute_sample_indices(n, start, step, count)
    return np.take(np.asarray(data), indices, axis=axis)


def _layer_source_path(layer: Layer) -> str | None:
    meta = getattr(layer, "metadata", None) or {}
    path = meta.get("source_path")
    if path:
        return str(path)
    data = getattr(layer, "data", None)
    path = getattr(data, "path", None)
    return str(path) if path else None


def _current_absolute_indices(layer: Layer, n_frames: int) -> list[int]:
    """Absolute native indices currently represented by ``layer``."""
    meta = getattr(layer, "metadata", None) or {}
    data = getattr(layer, "data", None)

    if hasattr(data, "frame_indices"):
        try:
            return list(data.frame_indices)
        except Exception:
            pass

    sample = meta.get("frame_sample")
    if isinstance(sample, dict) and all(
        k in sample for k in ("start", "step", "count")
    ):
        from napari_pecan_py.video_meta import compute_sample_indices

        # Reconstruct from saved sample against native count when known.
        native = meta.get("native_frame_count")
        if native is None and hasattr(data, "_native_frame_count"):
            native = int(data._native_frame_count)
        if native is not None:
            return compute_sample_indices(
                int(native),
                int(sample["start"]),
                int(sample["step"]),
                int(sample["count"]),
            )

    fr = meta.get("frame_range")
    if isinstance(fr, dict) and "start" in fr and "end" in fr:
        return list(range(int(fr["start"]), int(fr["end"]) + 1))

    if hasattr(data, "_frame_start") and hasattr(data, "_frame_end"):
        if getattr(data, "_frame_indices", None) is not None:
            return list(data._frame_indices)
        return list(range(int(data._frame_start), int(data._frame_end) + 1))

    return list(range(int(n_frames)))


def _current_absolute_frame_range(layer: Layer, n_frames: int) -> tuple[int, int]:
    """Absolute inclusive window covering the current layer frames."""
    indices = _current_absolute_indices(layer, n_frames)
    return int(indices[0]), int(indices[-1])


def _sync_viewer_dims_after_shape_change(layer: Layer) -> None:
    """Clear napari's stale extent cache and shrink the dims slider."""
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


def apply_trim_to_layer(
    layer: Layer,
    start: int,
    end: int,
    *,
    axis: int = 0,
    absolute_start: int | None = None,
    absolute_end: int | None = None,
    persist: bool = False,
) -> int:
    """Trim ``layer`` in place to inclusive ``[start, end]``; return new frame count."""
    from napari_pecan_py._reader import LazyVideoArray

    n_frames = int(np.asarray(layer.data.shape)[axis])
    base_start, _base_end = _current_absolute_frame_range(layer, n_frames)
    abs_start = (
        int(absolute_start)
        if absolute_start is not None
        else int(base_start) + int(start)
    )
    abs_end = (
        int(absolute_end) if absolute_end is not None else int(base_start) + int(end)
    )

    source_path = _layer_source_path(layer)
    used_lazy = False
    if (
        source_path
        and getattr(layer, "_type_string", None) == "image"
        and bool(getattr(layer, "rgb", False))
    ):
        try:
            trimmed = LazyVideoArray(
                source_path, frame_start=abs_start, frame_end=abs_end
            )
            layer.data = trimmed
            used_lazy = True
        except Exception:
            used_lazy = False

    if not used_lazy:
        trimmed = np.asarray(trim_array(layer.data, start, end, axis=axis))
        layer.data = trimmed

    meta = dict(getattr(layer, "metadata", None) or {})
    if source_path:
        meta["source_path"] = source_path
    meta.pop("frame_sample", None)
    meta["frame_range"] = {"start": abs_start, "end": abs_end}
    if used_lazy:
        meta["native_frame_count"] = int(layer.data._native_frame_count)
        meta["lazy_enabled"] = True
    layer.metadata = meta

    if persist and source_path:
        from napari_pecan_py.video_meta import set_saved_frame_range

        native = meta.get("native_frame_count")
        if native is None and used_lazy:
            native = int(layer.data._native_frame_count)
        set_saved_frame_range(
            source_path,
            abs_start,
            abs_end,
            native_frame_count=int(native) if native is not None else None,
        )

    _sync_viewer_dims_after_shape_change(layer)
    return int(np.asarray(layer.data.shape)[axis])


def apply_sample_trim_to_layer(
    layer: Layer,
    start: int,
    step: int,
    count: int,
    *,
    axis: int = 0,
    absolute_indices: list[int] | None = None,
    persist: bool = False,
) -> int:
    """Keep every ``step``-th frame from ``start``, up to ``count`` frames."""
    from napari_pecan_py._reader import LazyVideoArray
    from napari_pecan_py.video_meta import compute_sample_indices

    n_frames = int(np.asarray(layer.data.shape)[axis])
    if absolute_indices is None:
        current = _current_absolute_indices(layer, n_frames)
        rel = compute_sample_indices(len(current), start, step, count)
        abs_indices = [current[i] for i in rel]
    else:
        abs_indices = [int(i) for i in absolute_indices]
        if not abs_indices:
            raise ValueError("Sample produced no frames")

    source_path = _layer_source_path(layer)
    used_lazy = False
    if (
        source_path
        and getattr(layer, "_type_string", None) == "image"
        and bool(getattr(layer, "rgb", False))
    ):
        try:
            trimmed = LazyVideoArray(source_path, frame_indices=abs_indices)
            layer.data = trimmed
            used_lazy = True
        except Exception:
            used_lazy = False

    if not used_lazy:
        # Relative indices into the current layer stack.
        current = _current_absolute_indices(layer, n_frames)
        index_map = {abs_i: rel_i for rel_i, abs_i in enumerate(current)}
        rel_indices = [index_map[i] for i in abs_indices if i in index_map]
        if not rel_indices:
            raise ValueError("Sample indices do not intersect the current layer")
        layer.data = np.take(np.asarray(layer.data), rel_indices, axis=axis)

    # Recover sample params for metadata / sidecar (absolute into source).
    abs_start = int(abs_indices[0])
    abs_step = int(step)
    abs_count = len(abs_indices)
    if len(abs_indices) >= 2:
        # Prefer the requested step when indices are uniform.
        diffs = {abs_indices[i + 1] - abs_indices[i] for i in range(len(abs_indices) - 1)}
        if len(diffs) == 1:
            abs_step = int(next(iter(diffs)))

    meta = dict(getattr(layer, "metadata", None) or {})
    if source_path:
        meta["source_path"] = source_path
    meta.pop("frame_range", None)
    meta["frame_sample"] = {
        "start": abs_start,
        "step": abs_step,
        "count": abs_count,
    }
    if used_lazy:
        meta["native_frame_count"] = int(layer.data._native_frame_count)
        meta["lazy_enabled"] = True
    layer.metadata = meta

    if persist and source_path:
        from napari_pecan_py.video_meta import set_saved_frame_sample

        native = meta.get("native_frame_count")
        if native is None and used_lazy:
            native = int(layer.data._native_frame_count)
        set_saved_frame_sample(
            source_path,
            abs_start,
            abs_step,
            abs_count,
            native_frame_count=int(native) if native is not None else None,
        )

    _sync_viewer_dims_after_shape_change(layer)
    return int(np.asarray(layer.data.shape)[axis])


def _prompt_trim_params(
    n_frames: int,
    *,
    layer_name: str,
    can_persist: bool,
    parent=None,
) -> dict[str, Any] | None:
    """Dialog for contiguous range or strided sample trim. Return None if cancelled."""
    from qtpy.QtWidgets import (
        QButtonGroup,
        QCheckBox,
        QDialog,
        QDialogButtonBox,
        QFormLayout,
        QLabel,
        QRadioButton,
        QSpinBox,
        QVBoxLayout,
        QWidget,
    )

    dlg = QDialog(parent)
    dlg.setWindowTitle("Trim frames")
    layout = QVBoxLayout(dlg)
    layout.addWidget(
        QLabel(
            f"Layer: {layer_name}\n"
            f"Frames: 0 … {n_frames - 1} ({n_frames} total)"
        )
    )

    range_radio = QRadioButton("Contiguous range (start → end)")
    sample_radio = QRadioButton("Every n-th frame (first, step n, count l)")
    range_radio.setChecked(True)
    mode_group = QButtonGroup(dlg)
    mode_group.addButton(range_radio)
    mode_group.addButton(sample_radio)
    layout.addWidget(range_radio)
    layout.addWidget(sample_radio)

    range_panel = QWidget()
    range_form = QFormLayout(range_panel)
    start_box = QSpinBox()
    start_box.setRange(0, max(0, n_frames - 1))
    start_box.setValue(0)
    end_box = QSpinBox()
    end_box.setRange(0, max(0, n_frames - 1))
    end_box.setValue(max(0, n_frames - 1))
    range_form.addRow("Start frame", start_box)
    range_form.addRow("End frame", end_box)
    layout.addWidget(range_panel)

    sample_panel = QWidget()
    sample_form = QFormLayout(sample_panel)
    first_box = QSpinBox()
    first_box.setRange(0, max(0, n_frames - 1))
    first_box.setValue(0)
    first_box.setToolTip("First frame to keep (0-based)")
    step_box = QSpinBox()
    step_box.setRange(1, max(1, n_frames))
    step_box.setValue(1)
    step_box.setToolTip("Keep one frame every n frames")
    count_box = QSpinBox()
    count_box.setRange(1, max(1, n_frames))
    count_box.setValue(max(1, n_frames))
    count_box.setToolTip(
        "Maximum number of frames to keep (stops earlier if the video ends)"
    )
    sample_form.addRow("First frame", first_box)
    sample_form.addRow("Step n", step_box)
    sample_form.addRow("Count l", count_box)
    sample_panel.setEnabled(False)
    layout.addWidget(sample_panel)

    preview = QLabel("")
    preview.setStyleSheet("color: #888;")
    layout.addWidget(preview)

    persist_box = QCheckBox(
        "Remember selection on disk (.pecan.json next to the video)\n"
        "Training loaders will use only these frames; sidecar TIFF\n"
        "masks should match this trimmed length."
    )
    persist_box.setChecked(bool(can_persist))
    persist_box.setEnabled(bool(can_persist))
    layout.addWidget(persist_box)

    buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)

    def _sync_mode() -> None:
        is_range = range_radio.isChecked()
        range_panel.setEnabled(is_range)
        sample_panel.setEnabled(not is_range)
        _update_preview()

    def _sync_range_bounds() -> None:
        start = start_box.value()
        if end_box.value() < start:
            end_box.setValue(start)
        end_box.setMinimum(start)
        _update_preview()

    def _update_preview() -> None:
        from napari_pecan_py.video_meta import compute_sample_indices

        if range_radio.isChecked():
            s, e = start_box.value(), end_box.value()
            preview.setText(f"Keeps {e - s + 1} contiguous frame(s): [{s} … {e}]")
            return
        try:
            idxs = compute_sample_indices(
                n_frames, first_box.value(), step_box.value(), count_box.value()
            )
        except Exception as exc:
            preview.setText(str(exc))
            return
        if not idxs:
            preview.setText("Keeps 0 frames")
            return
        if len(idxs) <= 6:
            shown = ", ".join(str(i) for i in idxs)
        else:
            shown = (
                ", ".join(str(i) for i in idxs[:3])
                + ", …, "
                + ", ".join(str(i) for i in idxs[-2:])
            )
        preview.setText(f"Keeps {len(idxs)} frame(s): {shown}")

    range_radio.toggled.connect(_sync_mode)
    start_box.valueChanged.connect(_sync_range_bounds)
    end_box.valueChanged.connect(_update_preview)
    first_box.valueChanged.connect(_update_preview)
    step_box.valueChanged.connect(_update_preview)
    count_box.valueChanged.connect(_update_preview)
    _sync_mode()

    if dlg.exec() != QDialog.Accepted:
        return None

    persist = bool(persist_box.isChecked() and can_persist)
    if range_radio.isChecked():
        start = int(start_box.value())
        end = int(end_box.value())
        if end < start:
            return None
        return {"mode": "range", "start": start, "end": end, "persist": persist}

    first = int(first_box.value())
    step = int(step_box.value())
    count = int(count_box.value())
    return {
        "mode": "sample",
        "start": first,
        "step": step,
        "count": count,
        "persist": persist,
    }


def trim_selected_layers(ll: LayerList) -> None:
    """Prompt for a trim mode and apply it to selected Image/Labels layers."""
    from napari.utils.notifications import show_info, show_warning
    from napari_pecan_py.video_meta import compute_sample_indices

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

    active = ll.selection.active
    primary = next(
        (t for t in trimmable if t[0] is active),
        trimmable[0],
    )
    primary_layer, _axis, n_frames = primary
    can_persist = any(_layer_source_path(layer) for layer, _, _ in trimmable)

    parent = None
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        parent = app.activeWindow() if app is not None else None
    except Exception:
        parent = None

    result = _prompt_trim_params(
        n_frames,
        layer_name=str(primary_layer.name),
        can_persist=can_persist,
        parent=parent,
    )
    if result is None:
        return

    mode: TrimMode = result["mode"]
    persist = bool(result["persist"])
    trimmed_names: list[str] = []
    persisted_path = None

    if mode == "range":
        start = int(result["start"])
        end = int(result["end"])
        if start == 0 and end == n_frames - 1 and len(trimmable) == 1:
            show_info("Nothing to trim — full frame range already selected.")
            return

        base_indices = _current_absolute_indices(primary_layer, n_frames)
        abs_start = base_indices[start]
        abs_end = base_indices[min(end, len(base_indices) - 1)]

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
            layer_persist = bool(persist and _layer_source_path(layer))
            try:
                cur = _current_absolute_indices(layer, layer_n)
                layer_abs_start = cur[start]
                layer_abs_end = cur[min(layer_end, len(cur) - 1)]
                new_n = apply_trim_to_layer(
                    layer,
                    start,
                    layer_end,
                    axis=axis,
                    absolute_start=layer_abs_start,
                    absolute_end=layer_abs_end,
                    persist=layer_persist and persisted_path is None,
                )
                if layer_persist and persisted_path is None:
                    persisted_path = _layer_source_path(layer)
                    abs_start, abs_end = layer_abs_start, layer_abs_end
            except Exception as exc:
                show_warning(f"Failed to trim {layer.name}: {exc}")
                continue
            trimmed_names.append(
                f"{layer.name} [{start}:{layer_end}] → {new_n} frames"
            )

        if trimmed_names:
            msg = "Trimmed:\n" + "\n".join(trimmed_names)
            if persisted_path:
                from napari_pecan_py.video_meta import pecan_meta_path

                msg += (
                    f"\n\nSaved range [{abs_start}:{abs_end}] → "
                    f"{pecan_meta_path(persisted_path).name}"
                )
            show_info(msg)
        else:
            show_info("Nothing to trim.")
        return

    # Sample mode: first / step n / count l
    first = int(result["start"])
    step = int(result["step"])
    count = int(result["count"])
    primary_indices = _current_absolute_indices(primary_layer, n_frames)
    try:
        rel = compute_sample_indices(len(primary_indices), first, step, count)
    except Exception as exc:
        show_warning(str(exc))
        return
    if not rel:
        show_info("Nothing to trim — sample produced no frames.")
        return
    if len(rel) == n_frames and step == 1 and first == 0 and len(trimmable) == 1:
        show_info("Nothing to trim — sample keeps the full stack.")
        return

    abs_indices_primary = [primary_indices[i] for i in rel]
    saved_sample = None

    for layer, axis, layer_n in trimmable:
        layer_persist = bool(persist and _layer_source_path(layer))
        try:
            cur = _current_absolute_indices(layer, layer_n)
            layer_rel = [i for i in rel if i < len(cur)]
            if not layer_rel:
                show_warning(f"Skipped {layer.name}: sample start is past its length.")
                continue
            layer_abs = [cur[i] for i in layer_rel]
            new_n = apply_sample_trim_to_layer(
                layer,
                first,
                step,
                count,
                axis=axis,
                absolute_indices=layer_abs,
                persist=layer_persist and persisted_path is None,
            )
            if layer_persist and persisted_path is None:
                persisted_path = _layer_source_path(layer)
                meta = getattr(layer, "metadata", {}) or {}
                saved_sample = meta.get("frame_sample")
        except Exception as exc:
            show_warning(f"Failed to trim {layer.name}: {exc}")
            continue
        trimmed_names.append(
            f"{layer.name} first={first}, n={step}, l={count} → {new_n} frames"
        )

    if trimmed_names:
        msg = "Trimmed:\n" + "\n".join(trimmed_names)
        if persisted_path:
            from napari_pecan_py.video_meta import pecan_meta_path

            sample = saved_sample or {
                "start": abs_indices_primary[0],
                "step": step,
                "count": len(abs_indices_primary),
            }
            msg += (
                f"\n\nSaved sample start={sample['start']}, step={sample['step']}, "
                f"count={sample['count']} → {pecan_meta_path(persisted_path).name}"
            )
        show_info(msg)
    else:
        show_info("Nothing to trim.")


def register_trim_frames_action() -> None:
    """Register Trim frames… on the layer-list context menu (idempotent)."""
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
            tooltip=(
                "Keep a contiguous frame range or every n-th frame on the "
                "selected layer(s)"
            ),
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
