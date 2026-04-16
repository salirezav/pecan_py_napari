"""Color adjustments dock widget.

Creates a single adjusted Image layer above the selected input video layer by
applying an editable adjustment stack:
  - Brightness/Contrast
  - Levels
  - Curves
  - Surface Blur (edge-preserving blur, bilateral approximation)

Each adjustment has a checkbox to enable/disable it and supports add/remove/reorder.

Adjustments are applied **only to the currently displayed frame** (after a short
debounce). Other time points stay as plain copies of the source until you visit
them with the current stack. Cached frames are reused when the stack unchanged;
when the stack changes, previously adjusted slices are reset from the source so
old parameters are not left on-screen.

Use **Apply to all frames** to bake the current stack into every time slice. The
button turns **blue** when the stack no longer matches that last full export
(for example after you change parameters and only the visible frame updates).
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QTimer, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .defaults import default_adjustment_item, default_adjustment_stack
from .curves_histogram_editor import CurvesHistogramEditor
from .levels_histogram_editor import LevelsHistogramEditor
from .logic import apply_adjustments_to_single_frame, apply_adjustments_to_video
from ..pipeline_recorder.state import upsert_pipeline_step


_DEFAULT_TYPES = [
    ("brightness_contrast", "Brightness / Contrast"),
    ("levels", "Levels"),
    ("curves", "Curves (RGB)"),
    ("surface_blur", "Surface Blur"),
]


def _stack_fingerprint(stack: list[dict]) -> str:
    """Stable signature for the stack so we skip redundant per-frame work."""

    def _json_default(obj: Any):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Not JSON-serializable: {type(obj)!r}")

    return json.dumps(stack, sort_keys=True, default=_json_default)


_FRAME_DEBOUNCE_MS = 500

# "Apply to all" button: blue when the stack differs from the last full-volume bake.
_STYLE_APPLY_ALL_NEUTRAL = ""
_STYLE_APPLY_ALL_PENDING = (
    "QPushButton { background-color: #2a6ad8; color: #ffffff; font-weight: bold; "
    "padding: 6px 10px; border-radius: 4px; border: 1px solid #1f5abe; }"
    "QPushButton:hover { background-color: #3a7aee; }"
    "QPushButton:pressed { background-color: #1f5abe; }"
    "QPushButton:disabled { background-color: #555555; color: #aaaaaa; border: 1px solid #444; }"
)


class _AdjustAllWorker(QThread):
    finished = Signal(object)  # (job_id, fp, adjusted_volume)
    error = Signal(str)

    def __init__(self, job_id: int, fp: str, src: np.ndarray, stack: list[dict]):
        super().__init__()
        self._job_id = int(job_id)
        self._fp = fp
        self._src = np.asarray(src)
        self._stack = stack

    def run(self):
        try:
            adjusted = apply_adjustments_to_video(self._src, self._stack)
            adjusted = np.asarray(adjusted, copy=False)
            self.finished.emit((self._job_id, self._fp, adjusted))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _AdjustAllLazyWorker(QThread):
    finished = Signal(object)  # (job_id, fp, adjusted_volume)
    error = Signal(str)

    def __init__(self, job_id: int, fp: str, src_data: Any, stack: list[dict]):
        super().__init__()
        self._job_id = int(job_id)
        self._fp = fp
        self._src_data = src_data
        self._stack = stack

    def run(self):
        try:
            data = self._src_data
            shape = getattr(data, "shape", None)
            if shape is None:
                raise ValueError("Lazy source does not expose shape")
            if len(shape) == 3:
                frame = np.asarray(data)[..., :3]
                adjusted = np.array(
                    apply_adjustments_to_single_frame(frame, self._stack),
                    dtype=np.uint8,
                    copy=True,
                )
                self.finished.emit((self._job_id, self._fp, adjusted))
                return
            if len(shape) != 4:
                raise ValueError(f"Unsupported lazy source shape: {shape}")
            frames = []
            for t in range(int(shape[0])):
                frame = np.asarray(data[t])[..., :3]
                frames.append(
                    np.array(
                        apply_adjustments_to_single_frame(frame, self._stack),
                        dtype=np.uint8,
                        copy=True,
                    )
                )
            adjusted = np.stack(frames, axis=0)
            self.finished.emit((self._job_id, self._fp, adjusted))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _AdjustWorker(QThread):
    finished = Signal(object)  # (job_id, frame_index, fp, adjusted_frame)
    error = Signal(str)  # message

    def __init__(self, job_id: int, frame_index: int, fp: str, frame: np.ndarray, stack: list[dict]):
        super().__init__()
        self._job_id = int(job_id)
        self._frame_index = int(frame_index)
        self._fp = fp
        self._frame = np.asarray(frame)
        self._stack = stack

    def run(self):
        try:
            adjusted = np.array(
                apply_adjustments_to_single_frame(self._frame, self._stack),
                dtype=np.uint8,
                copy=True,
            )
            self.finished.emit((self._job_id, self._frame_index, self._fp, adjusted))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class ColorAdjustmentsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        self._original_layer: Image | None = None
        self._original_data: Any | None = None
        self._lazy_source = False
        self._output_layer_name: str | None = None
        self._current_stack: list[dict] = default_adjustment_stack()
        self._selected_stack_index = -1

        self._building_ui = False
        self._apply_job_id = 0
        self._worker: _AdjustWorker | None = None

        # Full-volume buffer (copy of source); slices updated as frames are adjusted.
        self._output_data: np.ndarray | None = None
        self._per_frame_fp: dict[int, str] = {}
        self._last_known_stack_fp: str | None = None
        self._all_frames_synced_fp: str | None = None
        self._apply_all_job_id = 0
        self._worker_all: _AdjustAllWorker | None = None
        # If user manually deletes "<name> - Adjusted", don't recreate on frame changes.
        # Re-enable recreation only on actual adjustment edits / layer changes.
        self._allow_output_recreate_next_apply = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---- Layer selector -----------------------------------------------
        layer_group = QWidget()
        layer_lay = QVBoxLayout(layer_group)
        layer_lay.setContentsMargins(0, 0, 0, 0)
        layer_lay.addWidget(QLabel("Input video layer (Image)"))

        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        layer_lay.addWidget(self._layer_combo)
        layout.addWidget(layer_group)

        # ---- Adjustment stack ---------------------------------------------
        stack_group = QWidget()
        stack_lay = QVBoxLayout(stack_group)
        stack_lay.setContentsMargins(0, 0, 0, 0)

        stack_lay.addWidget(QLabel("Adjustment stack (top to bottom order)"))

        self._stack_list = QListWidget()
        self._stack_list.itemSelectionChanged.connect(self._on_stack_selection_changed)
        self._stack_list.itemChanged.connect(self._on_stack_item_changed)
        stack_lay.addWidget(self._stack_list)

        # Add / remove / reorder
        add_row = QHBoxLayout()
        self._add_type_combo = QComboBox()
        for typ, label in _DEFAULT_TYPES:
            self._add_type_combo.addItem(label, typ)
        add_row.addWidget(self._add_type_combo, 1)

        self._btn_add = QPushButton("Add")
        self._btn_add.clicked.connect(self._add_adjustment)
        add_row.addWidget(self._btn_add)

        stack_lay.addLayout(add_row)

        ctrl_row = QHBoxLayout()
        self._btn_up = QPushButton("Up")
        self._btn_up.clicked.connect(self._move_selected_up)
        ctrl_row.addWidget(self._btn_up)

        self._btn_down = QPushButton("Down")
        self._btn_down.clicked.connect(self._move_selected_down)
        ctrl_row.addWidget(self._btn_down)

        self._btn_remove = QPushButton("Remove")
        self._btn_remove.clicked.connect(self._remove_selected)
        ctrl_row.addWidget(self._btn_remove)

        stack_lay.addLayout(ctrl_row)

        layout.addWidget(stack_group)

        # ---- Params editor ------------------------------------------------
        params_group = QWidget()
        params_lay = QVBoxLayout(params_group)
        params_lay.setContentsMargins(0, 0, 0, 0)
        self._params_layout = QVBoxLayout()
        params_lay.addLayout(self._params_layout)

        self._params_layout.addWidget(
            QLabel("Select an adjustment to edit its parameters.")
        )
        layout.addWidget(params_group)

        # ---- Apply full volume --------------------------------------------
        apply_row = QVBoxLayout()
        apply_row.setContentsMargins(0, 8, 0, 0)
        self._btn_apply_all = QPushButton("Apply to all frames")
        self._btn_apply_all.setToolTip(
            "Process every time slice with the current stack (can take a while). "
            "This button turns blue when the stack has changed since the last full export "
            "and only individual frames may be up to date."
        )
        self._btn_apply_all.clicked.connect(self._on_apply_all_clicked)
        apply_row.addWidget(self._btn_apply_all)
        self._apply_all_hint = QLabel(
            "Blue button = stack changed since last full export — only some frames may match."
        )
        self._apply_all_hint.setWordWrap(True)
        self._apply_all_hint.setStyleSheet("color: #888888; font-size: 11px;")
        apply_row.addWidget(self._apply_all_hint)
        layout.addLayout(apply_row)

        # ---- Debounce: visible frame + stack changes -----------------------
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(_FRAME_DEBOUNCE_MS)
        self._update_timer.timeout.connect(self._on_adjust_debounce)

        # ---- Events --------------------------------------------------------
        self._refresh_layer_list()
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
        self._viewer.dims.events.current_step.connect(self._on_dims_changed)

        # Build initial stack UI.
        self._build_stack_list()
        self._refresh_apply_all_button_appearance()

    def _current_stack_copy(self) -> list[dict]:
        return [dict(x) for x in (self._current_stack or [])]

    def _current_stack_fingerprint(self) -> str:
        return _stack_fingerprint(self._current_stack_copy())

    def _needs_apply_all_highlight(self) -> bool:
        if self._original_data is None:
            return False
        return self._all_frames_synced_fp != self._current_stack_fingerprint()

    def _refresh_apply_all_button_appearance(self) -> None:
        if not hasattr(self, "_btn_apply_all"):
            return
        busy = self._worker_all is not None and self._worker_all.isRunning()
        if self._original_data is None:
            self._btn_apply_all.setEnabled(False)
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_apply_all.setText("Apply to all frames")
            return
        self._btn_apply_all.setEnabled(not busy)
        if busy:
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_apply_all.setText("Applying to all frames…")
            return
        self._btn_apply_all.setText("Apply to all frames")
        if self._needs_apply_all_highlight():
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_PENDING)
        else:
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)

    def _refresh_layer_list(self):
        self._building_ui = True
        prev = self._layer_combo_current_layer()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                try:
                    data = layer.data
                    if data is None:
                        continue
                    ndim = getattr(data, "ndim", None)
                    if ndim is None:
                        shape = getattr(data, "shape", None)
                        ndim = len(shape) if shape is not None else None
                    if ndim is not None and int(ndim) >= 3:
                        self._layer_combo.addItem(layer.name, layer)
                except Exception:
                    # Some layers may not expose shape cleanly.
                    continue
        if prev is not None and prev in self._viewer.layers:
            idx = self._layer_combo.findData(prev)
            if idx >= 0:
                self._layer_combo.setCurrentIndex(idx)
        self._building_ui = False

    def _on_layer_removed(self, event=None):
        removed = getattr(event, "value", None)
        if removed is not None and self._output_layer_name is not None:
            try:
                if getattr(removed, "name", None) == self._output_layer_name:
                    self._allow_output_recreate_next_apply = False
            except Exception:
                pass
        self._refresh_layer_list()

    def _layer_combo_current_layer(self) -> Image | None:
        data = self._layer_combo.currentData()
        if data is None:
            return None
        if data in self._viewer.layers:
            return data
        return None

    def _on_layer_changed(self, _idx: int = 0):
        if self._building_ui:
            return
        self._original_layer = self._layer_combo_current_layer()
        self._original_data = None
        self._output_data = None
        self._per_frame_fp.clear()
        self._last_known_stack_fp = None
        self._all_frames_synced_fp = None
        if self._original_layer is None:
            self._refresh_apply_all_button_appearance()
            return
        self._original_data = np.asarray(self._original_layer.data).copy()
        self._lazy_source = False
        self._output_layer_name = f"{self._original_layer.name} - Adjusted"
        self._allow_output_recreate_next_apply = True
        self._ensure_output_layer_initialized(allow_create=True)
        self._schedule_update(allow_recreate=True)
        self._refresh_apply_all_button_appearance()

    def _schedule_update(self, *, allow_recreate: bool = True):
        if self._original_data is None:
            return
        if allow_recreate:
            self._allow_output_recreate_next_apply = True
        self._update_timer.start()
        self._refresh_apply_all_button_appearance()

    def _on_dims_changed(self, _event: Any = None) -> None:
        if self._original_data is None:
            return
        # Frame scrubbing should not resurrect a manually deleted output layer.
        self._schedule_update(allow_recreate=False)

    def _current_time_index(self) -> int:
        if self._original_data is None:
            return 0
        shape = getattr(self._original_data, "shape", None)
        if shape is None or len(shape) != 4:
            return 0
        t_size = int(shape[0])
        try:
            steps = tuple(int(x) for x in self._viewer.dims.nsteps)
            curr = tuple(int(x) for x in self._viewer.dims.current_step)
            match_axes = [i for i, n in enumerate(steps) if n == t_size]
            axis = match_axes[0] if len(match_axes) == 1 else 0
            t = int(curr[axis])
        except (IndexError, TypeError, ValueError):
            t = 0
        return int(np.clip(t, 0, t_size - 1))

    def _read_source_frame(self, t: int) -> np.ndarray:
        if self._original_data is None:
            raise ValueError("No source data")
        d = self._original_data
        shape = getattr(d, "shape", None)
        if shape is None:
            arr = np.asarray(d)
            return arr[..., :3] if arr.ndim == 3 else arr[int(t), ..., :3]
        if len(shape) == 3:
            return np.asarray(d)[..., :3]
        return np.asarray(d[int(t)])[..., :3]

    def _ensure_output_layer_initialized(self, *, allow_create: bool) -> bool:
        if self._original_data is None or self._output_layer_name is None:
            return False
        if self._lazy_source:
            if not allow_create:
                try:
                    _ = self._viewer.layers[self._output_layer_name]
                    return True
                except Exception:
                    return False
            try:
                layer = self._viewer.layers[self._output_layer_name]
                if getattr(layer.data, "shape", None) != self._read_source_frame(self._current_time_index()).shape:
                    layer.data = self._read_source_frame(self._current_time_index())
                    layer.refresh()
            except Exception:
                self._viewer.add_image(
                    self._read_source_frame(self._current_time_index()),
                    name=self._output_layer_name,
                )
            return True
        src = np.asarray(self._original_data)
        if self._output_data is not None and self._output_data.shape == src.shape:
            try:
                layer = self._viewer.layers[self._output_layer_name]
                if np.asarray(layer.data).shape != src.shape:
                    raise KeyError("shape")
            except Exception:
                if not allow_create:
                    return False
                self._output_data = None
            else:
                return True
        if not allow_create:
            return False
        self._output_data = src.copy()
        self._per_frame_fp.clear()
        self._last_known_stack_fp = None
        self._all_frames_synced_fp = None
        try:
            existing = self._viewer.layers[self._output_layer_name]
            if np.asarray(existing.data).shape == self._output_data.shape:
                existing.data = self._output_data
                existing.refresh()
            else:
                self._viewer.layers.remove(existing)
                raise KeyError
        except Exception:
            self._viewer.add_image(self._output_data, name=self._output_layer_name)
        return True

    def _invalidate_cached_slices(self) -> None:
        """After stack edits, undo previously adjusted slices so old params do not linger."""
        if self._output_data is None or self._original_data is None:
            self._per_frame_fp.clear()
            return
        orig = np.asarray(self._original_data)
        out = self._output_data
        for t in list(self._per_frame_fp.keys()):
            if orig.ndim == 3:
                np.copyto(out, orig)
            else:
                out[t] = orig[t]
        self._per_frame_fp.clear()

    def _write_adjusted_frame(self, t: int, adjusted: np.ndarray) -> None:
        if self._output_data is None:
            return
        out = self._output_data
        adj = np.asarray(adjusted, dtype=np.uint8)
        c = min(3, adj.shape[-1], out.shape[-1])
        if out.ndim == 3:
            out[..., :c] = adj[..., :c]
        else:
            out[int(t), ..., :c] = adj[..., :c]

    def _refresh_output_layer_data(self) -> None:
        if self._output_layer_name is None or self._output_data is None:
            return
        try:
            layer = self._viewer.layers[self._output_layer_name]
            layer.data = self._output_data
            layer.refresh()
        except Exception:
            pass

    def _on_adjust_debounce(self) -> None:
        if self._original_data is None or self._output_layer_name is None:
            return
        allow_create = bool(self._allow_output_recreate_next_apply)
        self._allow_output_recreate_next_apply = False
        ok = self._ensure_output_layer_initialized(allow_create=allow_create)
        if not ok or (not self._lazy_source and self._output_data is None):
            return

        stack_copy = self._current_stack_copy()
        fp = _stack_fingerprint(stack_copy)

        if fp != self._last_known_stack_fp:
            self._invalidate_cached_slices()
            self._last_known_stack_fp = fp

        t = self._current_time_index()
        if self._per_frame_fp.get(t) == fp:
            self._refresh_output_layer_data()
            return

        if self._lazy_source:
            try:
                frame = self._read_source_frame(t)
                adjusted = np.array(
                    apply_adjustments_to_single_frame(frame, stack_copy),
                    dtype=np.uint8,
                    copy=True,
                )
                layer = self._viewer.layers[self._output_layer_name]
                layer.data = adjusted
                layer.refresh()
                self._per_frame_fp[t] = fp
            except Exception as exc:
                from napari.utils.notifications import show_error

                show_error(f"Adjustment error:\n{exc}")
            self._refresh_apply_all_button_appearance()
            return

        self._apply_job_id += 1
        job_id = self._apply_job_id
        frame = self._read_source_frame(t)

        self._worker = _AdjustWorker(job_id, t, fp, frame, stack_copy)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, payload: Any):
        job_id, t, fp, adjusted = payload
        if job_id != self._apply_job_id:
            return
        if self._output_layer_name is None:
            return

        stack_now = [dict(x) for x in (self._current_stack or [])]
        if _stack_fingerprint(stack_now) != fp:
            return

        if self._lazy_source:
            try:
                layer = self._viewer.layers[self._output_layer_name]
                layer.data = np.asarray(adjusted, dtype=np.uint8)
                layer.refresh()
            except Exception:
                pass
        else:
            self._write_adjusted_frame(t, adjusted)
            self._per_frame_fp[t] = fp
            self._refresh_output_layer_data()
        self._refresh_apply_all_button_appearance()

    def _on_worker_error(self, msg: str):
        from napari.utils.notifications import show_error

        show_error(f"Adjustment error:\n{msg}")
        self._refresh_apply_all_button_appearance()

    def _on_apply_all_clicked(self) -> None:
        if self._original_data is None or self._output_layer_name is None:
            return
        self._allow_output_recreate_next_apply = True
        self._ensure_output_layer_initialized(allow_create=True)
        if self._output_data is None:
            return
        stack_copy = self._current_stack_copy()
        fp = _stack_fingerprint(stack_copy)
        self._apply_all_job_id += 1
        job_id = self._apply_all_job_id
        if self._lazy_source:
            self._worker_all = _AdjustAllLazyWorker(job_id, fp, self._original_data, stack_copy)
        else:
            src = np.asarray(self._original_data)
            self._worker_all = _AdjustAllWorker(job_id, fp, src, stack_copy)
        self._worker_all.finished.connect(self._on_apply_all_worker_finished)
        self._worker_all.error.connect(self._on_apply_all_worker_error)
        self._refresh_apply_all_button_appearance()
        self._worker_all.start()

    def _on_apply_all_worker_finished(self, payload: Any) -> None:
        job_id, fp, adjusted = payload
        if job_id != self._apply_all_job_id:
            return
        if self._output_layer_name is None:
            self._refresh_apply_all_button_appearance()
            return

        if _stack_fingerprint(self._current_stack_copy()) != fp:
            self._refresh_apply_all_button_appearance()
            return

        if self._lazy_source:
            try:
                layer = self._viewer.layers[self._output_layer_name]
                layer.data = np.asarray(adjusted)
                layer.refresh()
            except Exception:
                pass
            self._last_known_stack_fp = fp
            self._all_frames_synced_fp = fp
            self._refresh_apply_all_button_appearance()
            return

        if self._output_data is None:
            self._refresh_apply_all_button_appearance()
            return

        adj = np.asarray(adjusted)
        out = self._output_data
        c = min(3, adj.shape[-1], out.shape[-1])
        if adj.ndim == 3:
            out[..., :c] = adj[..., :c]
            self._per_frame_fp = {0: fp}
        else:
            out[..., :c] = adj[..., :c]
            n = adj.shape[0]
            self._per_frame_fp = {t: fp for t in range(n)}
        self._last_known_stack_fp = fp
        self._all_frames_synced_fp = fp
        self._refresh_output_layer_data()
        self._refresh_apply_all_button_appearance()

    def _on_apply_all_worker_error(self, msg: str) -> None:
        from napari.utils.notifications import show_error

        show_error(f"Apply-to-all error:\n{msg}")
        self._refresh_apply_all_button_appearance()

    # ---- Stack list -------------------------------------------------------

    def _build_stack_list(self):
        self._building_ui = True
        try:
            self._stack_list.clear()
            for i, adj in enumerate(self._current_stack):
                typ = adj.get("type", "unknown")
                enabled = bool(adj.get("enabled", True))
                # Actually create QListWidgetItem via addItem:
                label = f"{i+1}. {typ.replace('_', ' ').title()}"
                self._stack_list.addItem(label)
                item = self._stack_list.item(i)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked)
            if 0 <= self._selected_stack_index < len(self._current_stack):
                self._stack_list.setCurrentRow(self._selected_stack_index)
            else:
                self._selected_stack_index = 0 if self._current_stack else -1
                if self._selected_stack_index >= 0:
                    self._stack_list.setCurrentRow(self._selected_stack_index)
                else:
                    self._selected_stack_index = -1
        finally:
            self._building_ui = False

        self._on_stack_selection_changed()

    def _on_stack_selection_changed(self):
        if self._building_ui:
            return
        row = self._stack_list.currentRow()
        if row < 0 or row >= len(self._current_stack):
            self._selected_stack_index = -1
            self._clear_params_editor()
            self._params_layout.addWidget(QLabel("No adjustment selected."))
            return
        self._selected_stack_index = row
        self._rebuild_params_editor()

    def _clear_params_editor(self):
        if hasattr(self, "_levels_value_label"):
            delattr(self, "_levels_value_label")
        if hasattr(self, "_curves_value_label"):
            delattr(self, "_curves_value_label")
        while self._params_layout.count():
            item = self._params_layout.takeAt(0)
            w = item.widget() if hasattr(item, "widget") else None
            if w is not None:
                w.setParent(None)
                try:
                    w.deleteLater()
                except Exception:
                    pass
            l = item.layout() if hasattr(item, "layout") else None
            if l is not None:
                # recursively delete; simple enough for our case
                self._delete_layout(l)

    def _delete_layout(self, layout):
        while layout.count():
            it = layout.takeAt(0)
            w = it.widget() if hasattr(it, "widget") else None
            if w is not None:
                w.setParent(None)
                try:
                    w.deleteLater()
                except Exception:
                    pass
            sub = it.layout() if hasattr(it, "layout") else None
            if sub is not None:
                self._delete_layout(sub)

    def _rebuild_params_editor(self):
        self._clear_params_editor()
        if self._selected_stack_index < 0:
            return
        adj = self._current_stack[self._selected_stack_index]
        typ = adj.get("type")

        # Re-enable/disable editing based on typ
        if typ == "brightness_contrast":
            self._params_layout.addWidget(QLabel("Brightness / Contrast (RGB)"))
            b = int(adj.get("brightness", -32))
            c = int(adj.get("contrast", 77))

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Brightness:"))
            slider_b = QSlider(Qt.Orientation.Horizontal)
            slider_b.setRange(-200, 200)
            slider_b.setValue(b)
            spin_b = QSpinBox()
            spin_b.setRange(-200, 200)
            spin_b.setValue(b)

            def _on_brightness_slider(v: int) -> None:
                spin_b.blockSignals(True)
                spin_b.setValue(int(v))
                spin_b.blockSignals(False)
                self._set_adj_param("brightness", int(v))

            def _on_brightness_spin(v: int) -> None:
                slider_b.blockSignals(True)
                slider_b.setValue(int(v))
                slider_b.blockSignals(False)
                self._set_adj_param("brightness", int(v))

            slider_b.valueChanged.connect(_on_brightness_slider)
            spin_b.valueChanged.connect(_on_brightness_spin)
            row1.addWidget(slider_b, 1)
            row1.addWidget(spin_b)
            self._params_layout.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Contrast:"))
            slider_c = QSlider(Qt.Orientation.Horizontal)
            slider_c.setRange(-200, 200)
            slider_c.setValue(c)
            spin_c = QSpinBox()
            spin_c.setRange(-200, 200)
            spin_c.setValue(c)

            def _on_contrast_slider(v: int) -> None:
                spin_c.blockSignals(True)
                spin_c.setValue(int(v))
                spin_c.blockSignals(False)
                self._set_adj_param("contrast", int(v))

            def _on_contrast_spin(v: int) -> None:
                slider_c.blockSignals(True)
                slider_c.setValue(int(v))
                slider_c.blockSignals(False)
                self._set_adj_param("contrast", int(v))

            slider_c.valueChanged.connect(_on_contrast_slider)
            spin_c.valueChanged.connect(_on_contrast_spin)
            row2.addWidget(slider_c, 1)
            row2.addWidget(spin_c)
            self._params_layout.addLayout(row2)
            return

        if typ == "levels":
            self._params_layout.addWidget(
                QLabel("Levels — drag triangles (input: black / gamma / white; output: black / white).")
            )
            in_min = int(adj.get("in_min", 0))
            in_max = int(adj.get("in_max", 214))
            gamma = float(adj.get("gamma", 0.08))
            out_min = int(adj.get("out_min", 0))
            out_max = int(adj.get("out_max", 255))

            editor = LevelsHistogramEditor()
            hist = self._luma_histogram_for_source()
            if hist is not None:
                editor.set_histogram(hist)
            editor.set_levels(
                in_min, gamma, in_max, out_min, out_max, block_signals=True
            )
            editor.levels_changed.connect(self._on_levels_ui_changed)
            self._params_layout.addWidget(editor)

            self._levels_value_label = QLabel()
            self._levels_value_label.setWordWrap(True)
            self._update_levels_value_label(
                dict(
                    in_min=in_min,
                    gamma=gamma,
                    in_max=in_max,
                    out_min=out_min,
                    out_max=out_max,
                )
            )
            self._params_layout.addWidget(self._levels_value_label)
            return

        if typ == "curves":
            self._params_layout.addWidget(
                QLabel(
                    "Curves (RGB) — drag points; double-click the curve to add a point. "
                    "Endpoints fix input 0 and 255 (vertical drag only)."
                )
            )
            x_points = list(adj.get("x_points", [0, 64, 128, 255]))
            y_points = list(adj.get("y_points", [0, 70, 200, 255]))
            if len(x_points) != len(y_points) or len(x_points) < 2:
                x_points = [0, 64, 128, 255]
                y_points = [0, 70, 200, 255]

            ceditor = CurvesHistogramEditor()
            hist_c = self._luma_histogram_for_source()
            if hist_c is not None:
                ceditor.set_histogram(hist_c)
            ceditor.set_curve(x_points, y_points, block_signals=True)
            ceditor.curve_changed.connect(self._on_curves_ui_changed)
            self._params_layout.addWidget(ceditor)

            self._curves_value_label = QLabel()
            self._curves_value_label.setWordWrap(True)
            self._update_curves_value_label(x_points, y_points)
            self._params_layout.addWidget(self._curves_value_label)
            return

        if typ == "surface_blur":
            self._params_layout.addWidget(
                QLabel(
                    "Surface Blur — edge-preserving smooth (OpenCV bilateral approximation). "
                    "Large radius can be slow on big frames."
                )
            )
            rad = int(adj.get("radius", 26))
            thr = int(adj.get("threshold", 20))
            row_r, slid_r = self._surface_blur_radius_row(rad)
            row_t, slid_t = self._surface_blur_threshold_row(thr)
            self._params_layout.addLayout(row_r)
            self._params_layout.addWidget(slid_r)
            self._params_layout.addLayout(row_t)
            self._params_layout.addWidget(slid_t)
            return

        self._params_layout.addWidget(QLabel(f"Unknown adjustment type: {typ}"))

    def _surface_blur_radius_row(self, rad: int) -> tuple[QHBoxLayout, QSlider]:
        row = QHBoxLayout()
        row.addWidget(QLabel("Radius:"))
        spin = QSpinBox()
        spin.setRange(1, 100)
        spin.setValue(int(np.clip(rad, 1, 100)))
        row.addWidget(spin, 0)
        row.addWidget(QLabel("pixels"))

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(1, 100)
        slider.setValue(spin.value())

        def on_spin(v: int) -> None:
            if self._building_ui:
                return
            idx = self._selected_stack_index
            if idx < 0:
                return
            cur = self._current_stack[idx]
            if cur.get("type") != "surface_blur":
                return
            v = int(np.clip(v, 1, 100))
            cur["radius"] = v
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
            self._record_stack_step()
            self._schedule_update()

        def on_slide(v: int) -> None:
            if self._building_ui:
                return
            idx = self._selected_stack_index
            if idx < 0:
                return
            cur = self._current_stack[idx]
            if cur.get("type") != "surface_blur":
                return
            v = int(np.clip(v, 1, 100))
            cur["radius"] = v
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
            self._record_stack_step()
            self._schedule_update()

        spin.valueChanged.connect(on_spin)
        slider.valueChanged.connect(on_slide)
        return row, slider

    def _surface_blur_threshold_row(self, thr: int) -> tuple[QHBoxLayout, QSlider]:
        row = QHBoxLayout()
        row.addWidget(QLabel("Threshold:"))
        spin = QSpinBox()
        spin.setRange(0, 255)
        spin.setValue(int(np.clip(thr, 0, 255)))
        row.addWidget(spin, 0)
        row.addWidget(QLabel("levels"))

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 255)
        slider.setValue(spin.value())

        def on_spin(v: int) -> None:
            if self._building_ui:
                return
            idx = self._selected_stack_index
            if idx < 0:
                return
            cur = self._current_stack[idx]
            if cur.get("type") != "surface_blur":
                return
            v = int(np.clip(v, 0, 255))
            cur["threshold"] = v
            slider.blockSignals(True)
            slider.setValue(v)
            slider.blockSignals(False)
            self._record_stack_step()
            self._schedule_update()

        def on_slide(v: int) -> None:
            if self._building_ui:
                return
            idx = self._selected_stack_index
            if idx < 0:
                return
            cur = self._current_stack[idx]
            if cur.get("type") != "surface_blur":
                return
            v = int(np.clip(v, 0, 255))
            cur["threshold"] = v
            spin.blockSignals(True)
            spin.setValue(v)
            spin.blockSignals(False)
            self._record_stack_step()
            self._schedule_update()

        spin.valueChanged.connect(on_spin)
        slider.valueChanged.connect(on_slide)
        return row, slider

    def _set_adj_param(self, key: str, value):
        if self._selected_stack_index < 0:
            return
        self._current_stack[self._selected_stack_index][key] = value
        self._record_stack_step()
        self._schedule_update()

    def _luma_histogram_for_source(self) -> np.ndarray | None:
        """256-bin luma histogram over all frames of the selected source video (for display)."""
        if self._original_data is None:
            return None
        # Avoid forcing full materialization for lazy sources.
        if self._lazy_source:
            try:
                d = np.asarray(self._read_source_frame(self._current_time_index()))
            except Exception:
                return None
        else:
            d = np.asarray(self._original_data)
        if d.ndim == 4 and d.shape[-1] >= 3:
            r = d[..., 0].ravel().astype(np.float32)
            g = d[..., 1].ravel().astype(np.float32)
            b = d[..., 2].ravel().astype(np.float32)
        elif d.ndim == 3 and d.shape[-1] >= 3:
            r = d[..., 0].ravel().astype(np.float32)
            g = d[..., 1].ravel().astype(np.float32)
            b = d[..., 2].ravel().astype(np.float32)
        else:
            return None
        lum = 0.299 * r + 0.587 * g + 0.114 * b
        lum = np.clip(lum, 0.0, 255.0)
        hist, _ = np.histogram(lum, bins=256, range=(0.0, 255.0))
        return hist.astype(np.float64)

    def _update_levels_value_label(self, d: dict) -> None:
        if not hasattr(self, "_levels_value_label"):
            return
        self._levels_value_label.setText(
            f"In: black={d['in_min']}  γ={d['gamma']:.4f}  white={d['in_max']}   "
            f"Out: black→{d['out_min']}  white→{d['out_max']}"
        )

    def _on_levels_ui_changed(self, d: dict) -> None:
        if self._building_ui:
            return
        if self._selected_stack_index < 0:
            return
        adj = self._current_stack[self._selected_stack_index]
        if adj.get("type") != "levels":
            return
        adj["in_min"] = int(d["in_min"])
        adj["gamma"] = float(d["gamma"])
        adj["in_max"] = int(d["in_max"])
        adj["out_min"] = int(d["out_min"])
        adj["out_max"] = int(d["out_max"])
        self._update_levels_value_label(d)
        self._record_stack_step()
        self._schedule_update()

    def _update_curves_value_label(self, x_points: list[int], y_points: list[int]) -> None:
        if not hasattr(self, "_curves_value_label"):
            return
        parts = [f"({x},{y})" for x, y in zip(x_points, y_points)]
        self._curves_value_label.setText("Points: " + "  ".join(parts))

    def _on_curves_ui_changed(self, d: dict) -> None:
        if self._building_ui:
            return
        if self._selected_stack_index < 0:
            return
        adj = self._current_stack[self._selected_stack_index]
        if adj.get("type") != "curves":
            return
        x_points = [int(x) for x in d["x_points"]]
        y_points = [int(y) for y in d["y_points"]]
        adj["x_points"] = x_points
        adj["y_points"] = y_points
        self._update_curves_value_label(x_points, y_points)
        self._record_stack_step()
        self._schedule_update()

    def _on_stack_item_changed(self, item):
        if self._building_ui:
            return
        row = self._stack_list.row(item)
        if not (0 <= row < len(self._current_stack)):
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        self._current_stack[row]["enabled"] = bool(enabled)
        self._record_stack_step()
        self._schedule_update()

    # ---- Stack ops --------------------------------------------------------

    def _add_adjustment(self):
        typ = str(self._add_type_combo.currentData())
        self._current_stack.append(
            dict(default_adjustment_item(typ), enabled=True)
        )
        self._selected_stack_index = len(self._current_stack) - 1
        self._build_stack_list()
        self._record_stack_step()
        self._schedule_update()

    def _remove_selected(self):
        idx = self._selected_stack_index
        if idx < 0 or idx >= len(self._current_stack):
            return
        del self._current_stack[idx]
        self._selected_stack_index = min(idx, len(self._current_stack) - 1) if self._current_stack else -1
        self._build_stack_list()
        self._record_stack_step()
        self._schedule_update()

    def _move_selected_up(self):
        idx = self._selected_stack_index
        if idx <= 0 or idx >= len(self._current_stack):
            return
        self._current_stack[idx - 1], self._current_stack[idx] = (
            self._current_stack[idx],
            self._current_stack[idx - 1],
        )
        self._selected_stack_index = idx - 1
        self._build_stack_list()
        self._record_stack_step()
        self._schedule_update()

    def _move_selected_down(self):
        idx = self._selected_stack_index
        if idx < 0 or idx >= len(self._current_stack) - 1:
            return
        self._current_stack[idx + 1], self._current_stack[idx] = (
            self._current_stack[idx],
            self._current_stack[idx + 1],
        )
        self._selected_stack_index = idx + 1
        self._build_stack_list()
        self._record_stack_step()
        self._schedule_update()

    def _record_stack_step(self) -> None:
        if self._original_layer is None:
            return
        stack = self._current_stack_copy()
        src_name = self._original_layer.name
        out_name = self._output_layer_name or f"{src_name} - Adjusted"
        params = {
            "source_layer": src_name,
            "output_layer": out_name,
            "adjustment_stack": stack,
        }
        upsert_pipeline_step(
            kind="color_adjustments.stack",
            description=f"Color Adjustments stack on {src_name}",
            params=params,
            match=lambda st: (
                st.kind == "color_adjustments.stack"
                and str((st.params or {}).get("source_layer", "")) == src_name
                and str((st.params or {}).get("output_layer", "")) == out_name
            ),
        )

