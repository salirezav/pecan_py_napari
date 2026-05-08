"""Edge Detection widget with method-specific parameters."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
)

from .logic import EDGE_METHODS, apply_edges_to_volume
from ..pipeline_recorder.state import upsert_pipeline_step


class EdgeDetectionWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building_ui = False
        self._structured_forest_state: dict[str, Any] = {}
        self._method_inputs: dict[str, dict[str, Any]] = {}
        self._output_layer_name: str | None = None
        self._output_data: np.ndarray | None = None
        self._per_frame_fp: dict[int, str] = {}
        self._last_stack_fp: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layer_group = QGroupBox("Input layer")
        layer_lay = QVBoxLayout(layer_group)
        layer_lay.addWidget(QLabel("Video/image layer (Image)"))
        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        layer_lay.addWidget(self._layer_combo)
        layout.addWidget(layer_group)

        method_group = QGroupBox("Edge detector")
        method_lay = QVBoxLayout(method_group)
        row = QHBoxLayout()
        row.addWidget(QLabel("Method:"))
        self._method_combo = QComboBox()
        for key, label in EDGE_METHODS.items():
            self._method_combo.addItem(label, key)
        row.addWidget(self._method_combo, 1)
        method_lay.addLayout(row)
        self._params_host = QWidget()
        self._params_layout = QFormLayout(self._params_host)
        method_lay.addWidget(self._params_host)
        layout.addWidget(method_group)

        self._btn_apply = QPushButton("Apply to all frames")
        self._btn_apply.clicked.connect(self._on_apply)
        layout.addWidget(self._btn_apply)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(300)
        self._debounce_timer.timeout.connect(self._on_live_preview_debounce)

        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._viewer.dims.events.current_step.connect(self._on_dims_changed)
        self._refresh_layer_list()
        self._on_method_changed()

    def _set_status(self, text: str) -> None:
        self._status.setText(text)

    def _selected_layer(self) -> Image | None:
        layer = self._layer_combo.currentData()
        if layer is None:
            return None
        if layer in self._viewer.layers and isinstance(layer, Image):
            return layer
        return None

    def _refresh_layer_list(self, _event: Any = None) -> None:
        self._building_ui = True
        try:
            prev = self._selected_layer()
            self._layer_combo.clear()
            self._layer_combo.addItem("(none)", None)
            for layer in self._viewer.layers:
                if not isinstance(layer, Image):
                    continue
                shape = getattr(layer.data, "shape", None)
                if shape is None:
                    continue
                if len(shape) >= 2:
                    self._layer_combo.addItem(layer.name, layer)
            if prev is not None and prev in self._viewer.layers:
                idx = self._layer_combo.findData(prev)
                if idx >= 0:
                    self._layer_combo.setCurrentIndex(idx)
        finally:
            self._building_ui = False

    def _clear_params_layout(self) -> None:
        while self._params_layout.rowCount() > 0:
            self._params_layout.removeRow(0)

    def _add_spin(self, method: str, key: str, lo: int, hi: int, value: int, step: int = 1) -> QSpinBox:
        w = QSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setValue(value)
        w.valueChanged.connect(self._on_param_changed)
        self._method_inputs.setdefault(method, {})[key] = w
        return w

    def _add_double(self, method: str, key: str, lo: float, hi: float, value: float, step: float = 0.1, decimals: int = 2) -> QDoubleSpinBox:
        w = QDoubleSpinBox()
        w.setRange(lo, hi)
        w.setSingleStep(step)
        w.setDecimals(decimals)
        w.setValue(value)
        w.valueChanged.connect(self._on_param_changed)
        self._method_inputs.setdefault(method, {})[key] = w
        return w

    def _add_check(self, method: str, key: str, value: bool) -> QCheckBox:
        w = QCheckBox()
        w.setChecked(bool(value))
        w.stateChanged.connect(self._on_param_changed)
        self._method_inputs.setdefault(method, {})[key] = w
        return w

    def _add_line(self, method: str, key: str, value: str = "") -> QLineEdit:
        w = QLineEdit()
        w.setText(str(value))
        w.textChanged.connect(self._on_param_changed)
        self._method_inputs.setdefault(method, {})[key] = w
        return w

    def _add_row(self, label: str, widget: QWidget) -> None:
        self._params_layout.addRow(QLabel(label), widget)

    def _read_widget_value(self, widget, default):
        try:
            if isinstance(widget, QSpinBox):
                return int(widget.value())
            if isinstance(widget, QDoubleSpinBox):
                return float(widget.value())
            if isinstance(widget, QCheckBox):
                return bool(widget.isChecked())
            if isinstance(widget, QLineEdit):
                return str(widget.text())
        except Exception:
            return default
        return default

    def _rebuild_method_params(self, _idx: int = 0) -> None:
        self._clear_params_layout()
        method = str(self._method_combo.currentData())
        prev_values = {
            k: self._read_widget_value(w, None) for k, w in self._method_inputs.get(method, {}).items()
        }
        self._method_inputs[method] = {}

        if method == "canny":
            self._add_row("Lower threshold", self._add_spin(method, "threshold1", 0, 255, int(prev_values.get("threshold1", 50) or 50)))
            self._add_row("Upper threshold", self._add_spin(method, "threshold2", 0, 255, int(prev_values.get("threshold2", 150) or 150)))
            self._add_row("Aperture size", self._add_spin(method, "aperture_size", 3, 7, int(prev_values.get("aperture_size", 3) or 3), step=2))
            self._add_row("Blur kernel", self._add_spin(method, "blur_ksize", 1, 31, int(prev_values.get("blur_ksize", 3) or 3), step=2))
            self._add_row("Blur sigma", self._add_double(method, "blur_sigma", 0.0, 20.0, float(prev_values.get("blur_sigma", 0.0) or 0.0), step=0.1))
            self._add_row("L2 gradient", self._add_check(method, "l2_gradient", bool(prev_values.get("l2_gradient", False))))
            return

        if method in ("sobel", "scharr"):
            if method == "sobel":
                self._add_row("Kernel size", self._add_spin(method, "ksize", 1, 31, int(prev_values.get("ksize", 3) or 3), step=2))
            self._add_row("Scale", self._add_double(method, "scale", 0.1, 10.0, float(prev_values.get("scale", 1.0) or 1.0), step=0.1))
            self._add_row("Delta", self._add_double(method, "delta", 0.0, 255.0, float(prev_values.get("delta", 0.0) or 0.0), step=1.0))
            self._add_row("Magnitude threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 40) or 40)))
            return

        if method == "laplacian":
            self._add_row("Kernel size", self._add_spin(method, "ksize", 1, 31, int(prev_values.get("ksize", 3) or 3), step=2))
            self._add_row("Scale", self._add_double(method, "scale", 0.1, 10.0, float(prev_values.get("scale", 1.0) or 1.0), step=0.1))
            self._add_row("Delta", self._add_double(method, "delta", 0.0, 255.0, float(prev_values.get("delta", 0.0) or 0.0), step=1.0))
            self._add_row("Response threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 25) or 25)))
            return

        if method in ("prewitt", "roberts"):
            default_thr = 40 if method == "prewitt" else 35
            self._add_row("Magnitude threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", default_thr) or default_thr)))
            return

        if method == "log":
            self._add_row("Gaussian kernel", self._add_spin(method, "blur_ksize", 1, 31, int(prev_values.get("blur_ksize", 5) or 5), step=2))
            self._add_row("Gaussian sigma", self._add_double(method, "sigma", 0.1, 20.0, float(prev_values.get("sigma", 1.2) or 1.2), step=0.1))
            self._add_row("Laplacian kernel", self._add_spin(method, "lap_ksize", 1, 31, int(prev_values.get("lap_ksize", 3) or 3), step=2))
            self._add_row("Response threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 20) or 20)))
            return

        if method == "dog":
            self._add_row("Kernel 1", self._add_spin(method, "ksize1", 1, 31, int(prev_values.get("ksize1", 3) or 3), step=2))
            self._add_row("Sigma 1", self._add_double(method, "sigma1", 0.1, 20.0, float(prev_values.get("sigma1", 1.0) or 1.0), step=0.1))
            self._add_row("Kernel 2", self._add_spin(method, "ksize2", 1, 31, int(prev_values.get("ksize2", 7) or 7), step=2))
            self._add_row("Sigma 2", self._add_double(method, "sigma2", 0.1, 20.0, float(prev_values.get("sigma2", 2.0) or 2.0), step=0.1))
            self._add_row("Difference threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 20) or 20)))
            return

        if method == "morph_gradient":
            self._add_row("Kernel size", self._add_spin(method, "kernel_size", 1, 51, int(prev_values.get("kernel_size", 3) or 3), step=2))
            self._add_row("Iterations", self._add_spin(method, "iterations", 1, 20, int(prev_values.get("iterations", 1) or 1)))
            self._add_row("Response threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 20) or 20)))
            return

        if method == "structured_forest":
            path_edit = self._add_line(method, "model_path", str(prev_values.get("model_path", "") or ""))
            row = QWidget()
            row_lay = QHBoxLayout(row)
            row_lay.setContentsMargins(0, 0, 0, 0)
            row_lay.addWidget(path_edit, 1)
            btn_browse = QPushButton("Browse")

            def _on_browse() -> None:
                path, _ = QFileDialog.getOpenFileName(self, "Select Structured Forest model", "", "Model (*.yml *.yml.gz);;All files (*.*)")
                if path:
                    path_edit.setText(path)

            btn_browse.clicked.connect(_on_browse)
            row_lay.addWidget(btn_browse)
            self._add_row("Model path", row)
            self._add_row("Use NMS", self._add_check(method, "use_nms", bool(prev_values.get("use_nms", True))))
            self._add_row("NMS radius", self._add_spin(method, "nms_radius", 1, 10, int(prev_values.get("nms_radius", 2) or 2)))
            self._add_row("NMS multiplier", self._add_double(method, "nms_mult", 0.1, 5.0, float(prev_values.get("nms_mult", 1.0) or 1.0), step=0.1))
            self._add_row("Response threshold", self._add_spin(method, "threshold", 0, 255, int(prev_values.get("threshold", 30) or 30)))

    def _on_method_changed(self, _idx: int = 0) -> None:
        self._rebuild_method_params()
        self._schedule_live_update()

    def _on_layer_changed(self, _idx: int = 0) -> None:
        self._output_data = None
        self._per_frame_fp.clear()
        self._last_stack_fp = None
        layer = self._selected_layer()
        self._output_layer_name = f"{layer.name} - Edges Preview" if layer is not None else None
        self._schedule_live_update()

    def _on_param_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._schedule_live_update()

    def _on_dims_changed(self, _event=None) -> None:
        self._schedule_live_update()

    def _schedule_live_update(self) -> None:
        if self._selected_layer() is None:
            return
        self._debounce_timer.start()

    def _stack_fingerprint(self, method: str, params: dict[str, Any]) -> str:
        return json.dumps({"method": method, "params": params}, sort_keys=True)

    def _ensure_output_initialized(self, layer: Image) -> bool:
        if self._output_layer_name is None:
            self._output_layer_name = f"{layer.name} - Edges Preview"
        data = layer.data
        shape = getattr(data, "shape", None)
        if shape is None:
            return False
        shp = tuple(int(x) for x in shape)
        if len(shp) == 2:
            wanted = (shp[0], shp[1])
        elif len(shp) == 3 and shp[-1] in (3, 4):
            wanted = (shp[0], shp[1])
        elif len(shp) == 3:
            wanted = (shp[0], shp[1], shp[2])
        elif len(shp) == 4:
            wanted = (shp[0], shp[1], shp[2])
        else:
            return False
        if self._output_data is None or tuple(self._output_data.shape) != wanted:
            self._output_data = np.zeros(wanted, dtype=np.uint8)
            self._per_frame_fp.clear()
            self._last_stack_fp = None
        try:
            out_layer = self._viewer.layers[self._output_layer_name]
            if tuple(np.asarray(out_layer.data).shape) != wanted:
                out_layer.data = self._output_data
                out_layer.refresh()
        except Exception:
            self._viewer.add_image(self._output_data, name=self._output_layer_name, colormap="gray")
        return True

    def _current_frame_index(self, layer: Image) -> int:
        shape = getattr(layer.data, "shape", None)
        if shape is None:
            return 0
        shp = tuple(int(x) for x in shape)
        if len(shp) not in (3, 4) or (len(shp) == 3 and shp[-1] in (3, 4)):
            return 0
        t_max = shp[0] - 1
        try:
            t = int(self._viewer.dims.current_step[0])
        except Exception:
            t = 0
        return int(np.clip(t, 0, t_max))

    def _read_source_frame(self, layer: Image, t: int) -> np.ndarray:
        data = layer.data
        shape = getattr(data, "shape", None)
        if shape is None:
            return np.asarray(data)
        shp = tuple(int(x) for x in shape)
        if len(shp) == 2:
            return np.asarray(data)
        if len(shp) == 3 and shp[-1] in (3, 4):
            return np.asarray(data)
        if len(shp) == 3:
            return np.asarray(data[int(t)])
        if len(shp) == 4:
            return np.asarray(data[int(t)])
        raise ValueError(f"Unsupported data shape: {shp}")

    def _write_preview_frame(self, t: int, edges: np.ndarray) -> None:
        if self._output_data is None:
            return
        out = self._output_data
        if out.ndim == 2:
            out[...] = edges
        else:
            out[int(t)] = edges

    def _push_output_layer(self) -> None:
        if self._output_layer_name is None or self._output_data is None:
            return
        try:
            out_layer = self._viewer.layers[self._output_layer_name]
            out_layer.data = self._output_data
            out_layer.refresh()
        except Exception:
            self._viewer.add_image(self._output_data, name=self._output_layer_name, colormap="gray")

    def _on_live_preview_debounce(self) -> None:
        from napari.utils.notifications import show_warning

        layer = self._selected_layer()
        if layer is None:
            return
        method = str(self._method_combo.currentData())
        params = self._collect_params(method)
        fp = self._stack_fingerprint(method, params)
        if fp != self._last_stack_fp:
            self._per_frame_fp.clear()
            self._last_stack_fp = fp
        if not self._ensure_output_initialized(layer):
            show_warning("Unsupported input shape for edge preview.")
            return
        t = self._current_frame_index(layer)
        if self._per_frame_fp.get(int(t)) == fp:
            self._push_output_layer()
            return
        try:
            frame = self._read_source_frame(layer, t)
            edges = apply_edges_to_volume(frame, method=method, params=params, state=self._structured_forest_state)
            edges_u8 = np.asarray(edges, dtype=np.uint8)
            self._write_preview_frame(t, edges_u8)
            self._per_frame_fp[int(t)] = fp
            self._push_output_layer()
            self._set_status(f"Preview updated for frame {t}.")
        except Exception as exc:
            show_warning(f"Edge preview failed: {exc}")
            self._set_status("")

    def _collect_params(self, method: str) -> dict:
        vals: dict[str, Any] = {}
        for key, widget in self._method_inputs.get(method, {}).items():
            if isinstance(widget, QSpinBox):
                vals[key] = int(widget.value())
            elif isinstance(widget, QDoubleSpinBox):
                vals[key] = float(widget.value())
            elif isinstance(widget, QCheckBox):
                vals[key] = bool(widget.isChecked())
            elif isinstance(widget, QLineEdit):
                vals[key] = str(widget.text())
        return vals

    def _on_apply(self) -> None:
        from napari.utils.notifications import show_info, show_warning

        layer = self._selected_layer()
        if layer is None:
            show_warning("Select an input Image layer.")
            return

        method = str(self._method_combo.currentData())
        method_label = str(self._method_combo.currentText())
        params = self._collect_params(method)

        try:
            data = layer.data
            out = apply_edges_to_volume(
                data,
                method=method,
                params=params,
                state=self._structured_forest_state,
                progress_callback=lambda c, t: self._set_status(f"Processing {c}/{t} frames..."),
            )
        except Exception as exc:
            show_warning(f"Edge detection failed: {exc}")
            self._set_status("")
            return

        out_name = f"{layer.name} - Edges ({method_label})"
        try:
            existing = self._viewer.layers[out_name]
        except Exception:
            existing = None
        if existing is not None and isinstance(existing, Image):
            existing.data = out.astype(np.uint8)
            existing.refresh()
        else:
            self._viewer.add_image(out.astype(np.uint8), name=out_name, colormap="gray")
        self._set_status(f"Created {out_name}.")
        show_info(f"Edge detection complete: {out_name}")

        rec_params = {
            "source_layer": layer.name,
            "method": method,
            "method_label": method_label,
            "params": dict(params),
            "output_layer": out_name,
        }
        upsert_pipeline_step(
            kind="edge_detection.apply",
            description=f"Edge Detection ({method_label}) on {layer.name}",
            params=rec_params,
            match=lambda st: (
                st.kind == "edge_detection.apply"
                and str((st.params or {}).get("source_layer", "")) == layer.name
                and str((st.params or {}).get("method", "")) == method
            ),
        )
