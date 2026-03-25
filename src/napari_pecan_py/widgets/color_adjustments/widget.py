"""Color adjustments dock widget.

Creates a single adjusted Image layer above the selected input video layer by
applying an editable adjustment stack:
  - Brightness/Contrast
  - Levels
  - Curves

Each adjustment has a checkbox to enable/disable it and supports add/remove/reorder.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QTimer, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .defaults import default_adjustment_item, default_adjustment_stack
from .logic import apply_adjustments_to_video


_DEFAULT_TYPES = [
    ("brightness_contrast", "Brightness / Contrast"),
    ("levels", "Levels"),
    ("curves", "Curves (RGB)"),
]


class _AdjustWorker(QThread):
    finished = Signal(object)  # (job_id, adjusted_frames)
    error = Signal(str)  # message

    def __init__(self, job_id: int, src: np.ndarray, stack: list[dict]):
        super().__init__()
        self._job_id = int(job_id)
        self._src = np.asarray(src)
        self._stack = stack

    def run(self):
        try:
            adjusted = apply_adjustments_to_video(self._src, self._stack)
            self.finished.emit((self._job_id, adjusted))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class ColorAdjustmentsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer

        self._original_layer: Image | None = None
        self._original_data: np.ndarray | None = None
        self._output_layer_name: str | None = None
        self._current_stack: list[dict] = default_adjustment_stack()
        self._selected_stack_index = -1

        self._building_ui = False
        self._job_id = 0
        self._worker: _AdjustWorker | None = None

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

        # ---- Debounce update ----------------------------------------------
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)
        self._update_timer.timeout.connect(self._schedule_apply)

        # ---- Events --------------------------------------------------------
        self._refresh_layer_list()
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)

        # Build initial stack UI.
        self._build_stack_list()

    def _refresh_layer_list(self):
        self._building_ui = True
        prev = self._layer_combo_current_layer()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                try:
                    if layer.data is not None and np.asarray(layer.data).ndim >= 3:
                        self._layer_combo.addItem(layer.name, layer)
                except Exception:
                    # Some layers may not expose shape cleanly.
                    continue
        if prev is not None and prev in self._viewer.layers:
            idx = self._layer_combo.findData(prev)
            if idx >= 0:
                self._layer_combo.setCurrentIndex(idx)
        self._building_ui = False

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
        if self._original_layer is None:
            return
        # Keep a copy so adjusting doesn't accumulate on previous results.
        self._original_data = np.asarray(self._original_layer.data).copy()

        # Create / update output layer name.
        self._output_layer_name = f"{self._original_layer.name} - Adjusted"

        self._schedule_apply()

    def _schedule_update(self):
        if self._original_data is None:
            return
        self._update_timer.start()

    def _schedule_apply(self):
        # Apply adjustments to generate exactly one output layer.
        if self._original_data is None or self._output_layer_name is None:
            return
        self._job_id += 1
        job_id = self._job_id

        if self._worker is not None and self._worker.isRunning():
            # Don't block; newest job id will be applied and older results ignored.
            pass

        stack_copy = [dict(x) for x in (self._current_stack or [])]
        self._worker = _AdjustWorker(job_id=job_id, src=self._original_data, stack=stack_copy)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.error.connect(self._on_worker_error)
        self._worker.start()

    def _on_worker_finished(self, payload: Any):
        job_id, adjusted = payload
        if job_id != self._job_id:
            return

        if self._output_layer_name is None:
            return

        try:
            existing = self._viewer.layers[self._output_layer_name]
            if np.asarray(existing.data).shape == np.asarray(adjusted).shape:
                existing.data = adjusted
            else:
                # Replace layer if shape mismatches.
                self._viewer.layers.remove(existing)
                raise KeyError
            existing.refresh()
        except Exception:
            self._viewer.add_image(adjusted, name=self._output_layer_name)

    def _on_worker_error(self, msg: str):
        from napari.utils.notifications import show_error

        show_error(f"Adjustment error:\n{msg}")

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
            spin_b = QSpinBox()
            spin_b.setRange(-200, 200)
            spin_b.setValue(b)
            spin_b.valueChanged.connect(lambda v: self._set_adj_param("brightness", int(v)))
            row1.addWidget(spin_b, 1)
            self._params_layout.addLayout(row1)

            row2 = QHBoxLayout()
            row2.addWidget(QLabel("Contrast:"))
            spin_c = QSpinBox()
            spin_c.setRange(-200, 200)
            spin_c.setValue(c)
            spin_c.valueChanged.connect(lambda v: self._set_adj_param("contrast", int(v)))
            row2.addWidget(spin_c, 1)
            self._params_layout.addLayout(row2)
            return

        if typ == "levels":
            self._params_layout.addWidget(QLabel("Levels (RGB)"))
            in_min = int(adj.get("in_min", 0))
            in_max = int(adj.get("in_max", 214))
            gamma = float(adj.get("gamma", 0.08))
            out_min = int(adj.get("out_min", 0))
            out_max = int(adj.get("out_max", 255))

            def mk_spin(label: str, val: float, lo: float, hi: float, is_int: bool, step: float):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"{label}:"))
                if is_int:
                    sp = QSpinBox()
                    sp.setRange(int(lo), int(hi))
                    sp.setValue(int(val))
                    sp.setSingleStep(int(step) if step >= 1 else 1)
                else:
                    sp = QDoubleSpinBox()
                    sp.setRange(lo, hi)
                    sp.setDecimals(4)
                    sp.setSingleStep(step)
                    sp.setValue(float(val))
                row.addWidget(sp, 1)
                return row, sp

            row, sp = mk_spin("In min", in_min, 0, 255, True, 1)
            sp.valueChanged.connect(lambda v: self._set_adj_param("in_min", int(v)))
            self._params_layout.addLayout(row)

            row, sp = mk_spin("In max", in_max, 0, 255, True, 1)
            sp.valueChanged.connect(lambda v: self._set_adj_param("in_max", int(v)))
            self._params_layout.addLayout(row)

            row, sp = mk_spin("Gamma", gamma, 0.01, 10.0, False, 0.01)
            sp.valueChanged.connect(lambda v: self._set_adj_param("gamma", float(v)))
            self._params_layout.addLayout(row)

            row, sp = mk_spin("Out min", out_min, 0, 255, True, 1)
            sp.valueChanged.connect(lambda v: self._set_adj_param("out_min", int(v)))
            self._params_layout.addLayout(row)

            row, sp = mk_spin("Out max", out_max, 0, 255, True, 1)
            sp.valueChanged.connect(lambda v: self._set_adj_param("out_max", int(v)))
            self._params_layout.addLayout(row)
            return

        if typ == "curves":
            self._params_layout.addWidget(QLabel("Curves (RGB)"))
            x_points = list(adj.get("x_points", [0, 64, 128, 255]))
            y_points = list(adj.get("y_points", [0, 70, 200, 255]))
            if len(x_points) != 4 or len(y_points) != 4:
                x_points = [0, 64, 128, 255]
                y_points = [0, 70, 200, 255]

            for i in range(4):
                row = QHBoxLayout()
                row.addWidget(QLabel(f"y at x={int(x_points[i])}:"))
                sp = QSpinBox()
                sp.setRange(0, 255)
                sp.setSingleStep(1)
                sp.setValue(int(y_points[i]))
                sp.valueChanged.connect(lambda v, idx=i: self._set_y_point(idx, int(v)))
                row.addWidget(sp, 1)
                self._params_layout.addLayout(row)
            return

        self._params_layout.addWidget(QLabel(f"Unknown adjustment type: {typ}"))

    def _set_adj_param(self, key: str, value):
        if self._selected_stack_index < 0:
            return
        self._current_stack[self._selected_stack_index][key] = value
        self._schedule_update()

    def _set_y_point(self, idx_point: int, new_y: int):
        if self._selected_stack_index < 0:
            return
        adj = self._current_stack[self._selected_stack_index]
        y_points = list(adj.get("y_points", [0, 70, 200, 255]))
        if len(y_points) != 4:
            y_points = [0, 70, 200, 255]
        y_points[idx_point] = int(new_y)
        adj["y_points"] = y_points
        self._schedule_update()

    def _on_stack_item_changed(self, item):
        if self._building_ui:
            return
        row = self._stack_list.row(item)
        if not (0 <= row < len(self._current_stack)):
            return
        enabled = item.checkState() == Qt.CheckState.Checked
        self._current_stack[row]["enabled"] = bool(enabled)
        self._schedule_update()

    # ---- Stack ops --------------------------------------------------------

    def _add_adjustment(self):
        typ = str(self._add_type_combo.currentData())
        self._current_stack.append(
            dict(default_adjustment_item(typ), enabled=True)
        )
        self._selected_stack_index = len(self._current_stack) - 1
        self._build_stack_list()
        self._schedule_update()

    def _remove_selected(self):
        idx = self._selected_stack_index
        if idx < 0 or idx >= len(self._current_stack):
            return
        del self._current_stack[idx]
        self._selected_stack_index = min(idx, len(self._current_stack) - 1) if self._current_stack else -1
        self._build_stack_list()
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
        self._schedule_update()

