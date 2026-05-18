"""Pipeline Recorder widget for capture/save/load/apply workflows."""

from __future__ import annotations

import json
from pathlib import Path

from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QCheckBox,
    QSpinBox,
    QDoubleSpinBox,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from .logic import apply_pipeline_step_with_context, create_apply_context
from .state import PIPELINE_STORE, PipelineStep
from ..color_thresholding.defaults import COLOR_SPACE_PARAMS, COLOR_SPACES, TARGETS
from ..edge_detection.logic import EDGE_METHODS


def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except Exception:
        return False


class PipelineRecorderWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building = False
        self._unsubscribe = PIPELINE_STORE.subscribe(self._rebuild_list)
        self._is_processing = False
        self._processing_steps: list[dict] = []
        self._processing_descriptions: list[str] = []
        self._processing_indices: list[int] = []
        self._processing_i = 0
        self._apply_ctx = None
        self._disabled_widgets: list[QWidget] = []
        self._cancel_requested = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        layout.addWidget(QLabel("Recorded adjustments (execution order)"))
        self._list = QListWidget()
        self._list.itemChanged.connect(self._on_item_changed)
        layout.addWidget(self._list)

        row = QHBoxLayout()
        self._btn_apply_selected = QPushButton("Apply selected step(s)")
        self._btn_apply_selected.clicked.connect(self._apply_selected)
        row.addWidget(self._btn_apply_selected)
        self._btn_apply_all = QPushButton("Apply pipeline on video")
        self._btn_apply_all.clicked.connect(self._apply_all_checked)
        row.addWidget(self._btn_apply_all)
        self._btn_stop = QPushButton("Stop")
        self._btn_stop.clicked.connect(self._request_stop)
        self._btn_stop.setEnabled(False)
        row.addWidget(self._btn_stop)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        self._btn_save = QPushButton("Save selected as YAML/JSON")
        self._btn_save.clicked.connect(self._save_selected)
        row2.addWidget(self._btn_save)
        self._btn_load = QPushButton("Load pipeline")
        self._btn_load.clicked.connect(self._load_pipeline)
        row2.addWidget(self._btn_load)
        self._btn_edit = QPushButton("Edit selected step")
        self._btn_edit.clicked.connect(self._edit_selected_step)
        row2.addWidget(self._btn_edit)
        self._btn_clear = QPushButton("Clear")
        self._btn_clear.clicked.connect(lambda: PIPELINE_STORE.clear())
        row2.addWidget(self._btn_clear)
        layout.addLayout(row2)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(True)
        self._progress.hide()
        layout.addWidget(self._progress)
        layout.addStretch(1)
        self._rebuild_list()

    def closeEvent(self, event):  # noqa: N802
        try:
            self._unsubscribe()
        except Exception:
            pass
        super().closeEvent(event)

    def _rebuild_list(self) -> None:
        self._building = True
        try:
            self._list.clear()
            for i, step in enumerate(PIPELINE_STORE.steps):
                label = self._format_step_label(i, step.description, "idle")
                item = QListWidgetItem(label)
                item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Checked if step.enabled else Qt.CheckState.Unchecked)
                self._list.addItem(item)
        finally:
            self._building = False

    def _on_item_changed(self, item: QListWidgetItem) -> None:
        if self._building:
            return
        idx = self._list.row(item)
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            return
        steps[idx].enabled = item.checkState() == Qt.CheckState.Checked
        PIPELINE_STORE.set_steps(steps)

    def _format_step_label(self, idx: int, description: str, state: str) -> str:
        if state == "running":
            tag = "[RUN]"
        elif state == "done":
            tag = "[OK]"
        elif state == "error":
            tag = "[ERR]"
        else:
            tag = "[ ]"
        return f"{idx+1}. {tag} {description}"

    def _set_item_state(self, idx: int, state: str) -> None:
        if not (0 <= idx < self._list.count()):
            return
        item = self._list.item(idx)
        if item is None:
            return
        raw = PIPELINE_STORE.steps
        if 0 <= idx < len(raw):
            desc = raw[idx].description
        else:
            desc = item.text()
        item.setText(self._format_step_label(idx, desc, state))

    def _selected_steps(self, checked_only: bool = True) -> list[dict]:
        out: list[dict] = []
        for s in PIPELINE_STORE.steps:
            if checked_only and not s.enabled:
                continue
            out.append(s.to_dict())
        return out

    def _save_selected(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
        steps = self._selected_steps(checked_only=True)
        if not steps:
            self._status.setText("No checked steps to save.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save pipeline",
            "pipeline.yml",
            "YAML (*.yml *.yaml);;JSON (*.json)",
        )
        if not path:
            return
        payload = {"version": 1, "steps": steps}
        p = Path(path)
        try:
            if p.suffix.lower() == ".json":
                p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            else:
                if not _yaml_available():
                    raise RuntimeError("PyYAML is not installed. Save as .json or install pyyaml.")
                import yaml

                p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            self._status.setText(f"Could not save pipeline: {exc}")
            return
        self._status.setText(f"Saved {len(steps)} step(s) to {p.name}.")

    def _load_pipeline(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load pipeline",
            "",
            "YAML/JSON (*.yml *.yaml *.json)",
        )
        if not path:
            return
        p = Path(path)
        try:
            txt = p.read_text(encoding="utf-8")
            if p.suffix.lower() == ".json":
                raw = json.loads(txt)
            else:
                if not _yaml_available():
                    raise RuntimeError("PyYAML is not installed. Load a .json file or install pyyaml.")
                import yaml

                raw = yaml.safe_load(txt)
            steps_raw = list((raw or {}).get("steps", []))
            steps = [PipelineStep.from_dict(x) for x in steps_raw if isinstance(x, dict)]
        except Exception as exc:
            self._status.setText(f"Could not load pipeline: {exc}")
            return
        PIPELINE_STORE.set_steps(steps)
        self._status.setText(f"Loaded {len(steps)} step(s) from {p.name}.")

    def _edit_selected_step(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is running. Wait for completion.")
            return
        idx = self._list.currentRow()
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            self._status.setText("Select a step first.")
            return
        step = steps[idx]
        edited = self._edit_step_dialog(step)
        if edited is None:
            return
        steps[idx] = edited
        PIPELINE_STORE.set_steps(steps)
        self._list.setCurrentRow(idx)
        self._status.setText(f"Updated step {idx+1}: {edited.description}")

    def _edit_step_dialog(self, step: PipelineStep) -> PipelineStep | None:
        dlg = QDialog(self)
        dlg.setWindowTitle(f"Edit step ({step.kind})")
        dlg.setMinimumWidth(700)
        lay = QVBoxLayout(dlg)

        lay.addWidget(QLabel("Description"))
        desc = QLineEdit(step.description)
        lay.addWidget(desc)
        params = dict(step.params or {})
        out_params: dict = dict(params)

        def _add_fallback_json_editor() -> tuple[QPlainTextEdit, None]:
            lay.addWidget(QLabel("Params (JSON object)"))
            params_text = QPlainTextEdit()
            params_text.setPlainText(json.dumps(params, indent=2, sort_keys=True))
            params_text.setMinimumHeight(260)
            lay.addWidget(params_text)
            return params_text, None

        params_text: QPlainTextEdit | None = None
        kind = str(step.kind)
        if kind in ("color_thresholding.threshold", "color_tuner.threshold"):
            form = QFormLayout()
            source = QLineEdit(str(params.get("source_layer", "")))
            target = QComboBox()
            for t in TARGETS:
                target.addItem(t.replace("_", " ").title(), t)
            tgt_idx = target.findData(str(params.get("target", "pecan")))
            target.setCurrentIndex(tgt_idx if tgt_idx >= 0 else 0)
            cs = QComboBox()
            for c in COLOR_SPACES:
                cs.addItem(c.upper(), c)
            cs_idx = cs.findData(str(params.get("color_space", "hsv")))
            cs.setCurrentIndex(cs_idx if cs_idx >= 0 else 0)
            out_name = QLineEdit(str(params.get("output_mask_layer", "")))
            form.addRow("Source layer", source)
            form.addRow("Target", target)
            form.addRow("Color space", cs)
            form.addRow("Output mask", out_name)
            lay.addLayout(form)

            ch_host = QWidget()
            ch_form = QFormLayout(ch_host)
            lower_spins: list[QSpinBox] = []
            upper_spins: list[QSpinBox] = []

            def _rebuild_channels() -> None:
                while ch_form.rowCount() > 0:
                    ch_form.removeRow(0)
                lower_spins.clear()
                upper_spins.clear()
                cs_key = str(cs.currentData())
                params_cs = COLOR_SPACE_PARAMS.get(cs_key, {})
                channels = list(params_cs.get("channels", ["C0", "C1", "C2"]))
                max_vals = list(params_cs.get("max_values", [255, 255, 255]))
                lower_vals = [int(x) for x in list(params.get("lower", [0, 0, 0]))]
                upper_vals = [int(x) for x in list(params.get("upper", [255, 255, 255]))]
                for i in range(3):
                    lo = QSpinBox()
                    hi = QSpinBox()
                    hi.setRange(0, int(max_vals[i]))
                    lo.setRange(0, int(max_vals[i]))
                    lo.setValue(int(lower_vals[i] if i < len(lower_vals) else 0))
                    hi.setValue(int(upper_vals[i] if i < len(upper_vals) else max_vals[i]))
                    row = QWidget()
                    row_lay = QHBoxLayout(row)
                    row_lay.setContentsMargins(0, 0, 0, 0)
                    row_lay.addWidget(QLabel("min"))
                    row_lay.addWidget(lo)
                    row_lay.addWidget(QLabel("max"))
                    row_lay.addWidget(hi)
                    ch_form.addRow(f"{channels[i]}", row)
                    lower_spins.append(lo)
                    upper_spins.append(hi)

            cs.currentIndexChanged.connect(_rebuild_channels)
            _rebuild_channels()
            lay.addWidget(ch_host)

            def _collect_params() -> dict:
                return {
                    "source_layer": source.text().strip(),
                    "target": str(target.currentData()),
                    "color_space": str(cs.currentData()),
                    "lower": [int(w.value()) for w in lower_spins],
                    "upper": [int(w.value()) for w in upper_spins],
                    "output_mask_layer": out_name.text().strip(),
                }

        elif kind == "mask_ops.operation":
            form = QFormLayout()
            mode = QComboBox()
            mode.addItem("Binary", "binary")
            mode.addItem("Clip", "clip")
            mode_idx = mode.findData(str(params.get("mode", "binary")))
            mode.setCurrentIndex(mode_idx if mode_idx >= 0 else 0)
            form.addRow("Mode", mode)
            lay.addLayout(form)

            clip_widget = QWidget()
            clip_form = QFormLayout(clip_widget)
            clip_form.setContentsMargins(0, 0, 0, 0)
            ellipse_layer = QLineEdit(str(params.get("ellipse_layer", "")))
            mask_layer = QLineEdit(str(params.get("mask_layer", "")))
            output_mode = QComboBox()
            output_mode.addItem("New layer", "new")
            output_mode.addItem("Overwrite", "overwrite")
            output_mode.setCurrentIndex(output_mode.findData(str(params.get("output_mode", "new"))))
            clip_out = QLineEdit(str(params.get("output_layer", "")))
            clip_form.addRow("Ellipse layer", ellipse_layer)
            clip_form.addRow("Mask layer", mask_layer)
            clip_form.addRow("Output mode", output_mode)
            clip_form.addRow("Output layer", clip_out)
            lay.addWidget(clip_widget)

            bin_widget = QWidget()
            bin_form = QFormLayout(bin_widget)
            bin_form.setContentsMargins(0, 0, 0, 0)
            a_layer = QLineEdit(str(params.get("a_layer", "")))
            b_layer = QLineEdit(str(params.get("b_layer", "")))
            op = QComboBox()
            for label, value in (
                ("AND", "and"), ("OR", "or"), ("XOR", "xor"), ("NOT", "not"),
                ("A-B", "a-b"), ("B-A", "b-a"), ("A if B", "a-if-b")
            ):
                op.addItem(label, value)
            op.setCurrentIndex(max(0, op.findData(str(params.get("op", "and")))))
            target = QComboBox()
            target.addItem("New layer", "new")
            target.addItem("Overwrite A", "a")
            target.addItem("Overwrite B", "b")
            target.setCurrentIndex(max(0, target.findData(str(params.get("target", "new")))))
            bin_out = QLineEdit(str(params.get("output_layer", "")))
            bin_form.addRow("A layer", a_layer)
            bin_form.addRow("B layer", b_layer)
            bin_form.addRow("Operation", op)
            bin_form.addRow("Apply result to", target)
            bin_form.addRow("Output layer", bin_out)
            lay.addWidget(bin_widget)

            def _refresh_mode() -> None:
                is_clip = str(mode.currentData()) == "clip"
                clip_widget.setVisible(is_clip)
                bin_widget.setVisible(not is_clip)

            mode.currentIndexChanged.connect(_refresh_mode)
            _refresh_mode()

            def _collect_params() -> dict:
                if str(mode.currentData()) == "clip":
                    return {
                        "mode": "clip",
                        "ellipse_layer": ellipse_layer.text().strip(),
                        "mask_layer": mask_layer.text().strip(),
                        "output_mode": str(output_mode.currentData()),
                        "output_layer": clip_out.text().strip(),
                    }
                return {
                    "mode": "binary",
                    "a_layer": a_layer.text().strip(),
                    "b_layer": b_layer.text().strip(),
                    "op": str(op.currentData()),
                    "target": str(target.currentData()),
                    "output_layer": bin_out.text().strip(),
                }

        elif kind == "mask_retouching.apply":
            form = QFormLayout()
            mask_layer = QLineEdit(str(params.get("mask_layer", "")))
            form.addRow("Mask layer", mask_layer)
            spins: dict[str, QSpinBox] = {}
            for key, label, lo, hi in (
                ("close_size", "Close kernel", 0, 99),
                ("open_size", "Open kernel", 0, 99),
                ("dilate_size", "Dilate kernel", 0, 99),
                ("dilate_iter", "Dilate iterations", 1, 20),
                ("erode_size", "Erode kernel", 0, 99),
                ("erode_iter", "Erode iterations", 1, 20),
                ("min_area", "Min area", 0, 999999),
                ("smooth_size", "Smooth kernel", 0, 99),
            ):
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setValue(int(params.get(key, lo)))
                spins[key] = w
                form.addRow(label, w)
            fill_holes = QCheckBox()
            fill_holes.setChecked(bool(params.get("do_fill_holes", False)))
            keep_largest = QCheckBox()
            keep_largest.setChecked(bool(params.get("do_keep_largest", False)))
            form.addRow("Fill holes", fill_holes)
            form.addRow("Keep largest contour", keep_largest)
            lay.addLayout(form)

            def _collect_params() -> dict:
                out = {"mask_layer": mask_layer.text().strip()}
                out.update({k: int(w.value()) for k, w in spins.items()})
                out["do_fill_holes"] = bool(fill_holes.isChecked())
                out["do_keep_largest"] = bool(keep_largest.isChecked())
                return out

        elif kind == "mask_retouching.save_masks":
            form = QFormLayout()
            mask_layer = QLineEdit(str(params.get("mask_layer", "")))
            fmt = QComboBox()
            fmt.addItem("TIFF (.tiff)", "tiff")
            fmt.addItem("NumPy (.npy)", "npy")
            fmt.setCurrentIndex(max(0, fmt.findData(str(params.get("format", "tiff")))))
            output_dir = QLineEdit(str(params.get("output_dir", "")))
            output_dir.setPlaceholderText("(blank = save next to the source video)")
            form.addRow("Mask layer", mask_layer)
            form.addRow("Format", fmt)
            form.addRow("Output directory", output_dir)
            lay.addLayout(form)

            def _collect_params() -> dict:
                return {
                    "mask_layer": mask_layer.text().strip(),
                    "format": str(fmt.currentData()),
                    "output_dir": output_dir.text().strip(),
                }

        elif kind == "edge_detection.apply":
            form = QFormLayout()
            source_layer = QLineEdit(str(params.get("source_layer", "")))
            method = QComboBox()
            for key, label in EDGE_METHODS.items():
                method.addItem(label, key)
            method.setCurrentIndex(max(0, method.findData(str(params.get("method", "canny")))))
            out_layer = QLineEdit(str(params.get("output_layer", "")))
            form.addRow("Source layer", source_layer)
            form.addRow("Method", method)
            form.addRow("Output layer", out_layer)
            lay.addLayout(form)

            method_params = dict(params.get("params", {}) or {})
            params_host = QWidget()
            params_form = QFormLayout(params_host)
            params_form.setContentsMargins(0, 0, 0, 0)
            lay.addWidget(params_host)
            param_widgets: dict[str, QWidget] = {}

            def _add_spin(key: str, label: str, lo: int, hi: int, default: int, step: int = 1) -> None:
                w = QSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setValue(int(method_params.get(key, default)))
                params_form.addRow(label, w)
                param_widgets[key] = w

            def _add_double(key: str, label: str, lo: float, hi: float, default: float, step: float = 0.1) -> None:
                w = QDoubleSpinBox()
                w.setRange(lo, hi)
                w.setSingleStep(step)
                w.setDecimals(3)
                w.setValue(float(method_params.get(key, default)))
                params_form.addRow(label, w)
                param_widgets[key] = w

            def _add_check(key: str, label: str, default: bool) -> None:
                w = QCheckBox()
                w.setChecked(bool(method_params.get(key, default)))
                params_form.addRow(label, w)
                param_widgets[key] = w

            def _add_line(key: str, label: str, default: str = "") -> None:
                w = QLineEdit(str(method_params.get(key, default)))
                params_form.addRow(label, w)
                param_widgets[key] = w

            def _rebuild_method_params() -> None:
                while params_form.rowCount() > 0:
                    params_form.removeRow(0)
                param_widgets.clear()
                m = str(method.currentData())
                if m == "canny":
                    _add_spin("threshold1", "Lower threshold", 0, 255, 50)
                    _add_spin("threshold2", "Upper threshold", 0, 255, 150)
                    _add_spin("aperture_size", "Aperture size", 3, 7, 3, step=2)
                    _add_spin("blur_ksize", "Blur kernel", 1, 31, 3, step=2)
                    _add_double("blur_sigma", "Blur sigma", 0.0, 20.0, 0.0, step=0.1)
                    _add_check("l2_gradient", "L2 gradient", False)
                    return
                if m in ("sobel", "scharr"):
                    if m == "sobel":
                        _add_spin("ksize", "Kernel size", 1, 31, 3, step=2)
                    _add_double("scale", "Scale", 0.1, 10.0, 1.0, step=0.1)
                    _add_double("delta", "Delta", 0.0, 255.0, 0.0, step=1.0)
                    _add_spin("threshold", "Magnitude threshold", 0, 255, 40)
                    return
                if m == "laplacian":
                    _add_spin("ksize", "Kernel size", 1, 31, 3, step=2)
                    _add_double("scale", "Scale", 0.1, 10.0, 1.0, step=0.1)
                    _add_double("delta", "Delta", 0.0, 255.0, 0.0, step=1.0)
                    _add_spin("threshold", "Response threshold", 0, 255, 25)
                    return
                if m in ("prewitt", "roberts"):
                    _add_spin("threshold", "Magnitude threshold", 0, 255, 40 if m == "prewitt" else 35)
                    return
                if m == "log":
                    _add_spin("blur_ksize", "Gaussian kernel", 1, 31, 5, step=2)
                    _add_double("sigma", "Gaussian sigma", 0.1, 20.0, 1.2, step=0.1)
                    _add_spin("lap_ksize", "Laplacian kernel", 1, 31, 3, step=2)
                    _add_spin("threshold", "Response threshold", 0, 255, 20)
                    return
                if m == "dog":
                    _add_spin("ksize1", "Kernel 1", 1, 31, 3, step=2)
                    _add_double("sigma1", "Sigma 1", 0.1, 20.0, 1.0, step=0.1)
                    _add_spin("ksize2", "Kernel 2", 1, 31, 7, step=2)
                    _add_double("sigma2", "Sigma 2", 0.1, 20.0, 2.0, step=0.1)
                    _add_spin("threshold", "Difference threshold", 0, 255, 20)
                    return
                if m == "morph_gradient":
                    _add_spin("kernel_size", "Kernel size", 1, 51, 3, step=2)
                    _add_spin("iterations", "Iterations", 1, 20, 1, step=1)
                    _add_spin("threshold", "Response threshold", 0, 255, 20, step=1)
                    return
                if m == "structured_forest":
                    _add_line("model_path", "Model path", "")
                    _add_check("use_nms", "Use NMS", True)
                    _add_spin("nms_radius", "NMS radius", 1, 10, 2, step=1)
                    _add_double("nms_mult", "NMS multiplier", 0.1, 5.0, 1.0, step=0.1)
                    _add_spin("threshold", "Response threshold", 0, 255, 30, step=1)

            method.currentIndexChanged.connect(_rebuild_method_params)
            _rebuild_method_params()

            def _collect_params() -> dict:
                collected: dict[str, object] = {}
                for key, widget in param_widgets.items():
                    if isinstance(widget, QSpinBox):
                        collected[key] = int(widget.value())
                    elif isinstance(widget, QDoubleSpinBox):
                        collected[key] = float(widget.value())
                    elif isinstance(widget, QCheckBox):
                        collected[key] = bool(widget.isChecked())
                    elif isinstance(widget, QLineEdit):
                        collected[key] = widget.text().strip()
                method_key = str(method.currentData())
                return {
                    "source_layer": source_layer.text().strip(),
                    "method": method_key,
                    "method_label": str(EDGE_METHODS.get(method_key, method_key)),
                    "params": collected,
                    "output_layer": out_layer.text().strip(),
                }

        elif kind == "pecan_ellipse.fit":
            form = QFormLayout()
            mask_layer = QLineEdit(str(params.get("mask_layer", "")))
            out_shapes = QLineEdit(str(params.get("output_shapes_layer", "")))
            label_id = QSpinBox()
            label_id.setRange(0, 999999)
            label_id.setValue(int(params.get("label_id", 0) or 0))
            largest_only = QCheckBox()
            largest_only.setChecked(bool(params.get("largest_only", True)))
            mode = QComboBox()
            mode.addItem("Current frame", "current")
            mode.addItem("All frames", "all")
            mode.setCurrentIndex(max(0, mode.findData(str(params.get("mode", "all")))))
            time_idx = QSpinBox()
            time_idx.setRange(0, 999999)
            time_idx.setValue(int(params.get("time_index", 0)))
            form.addRow("Mask layer", mask_layer)
            form.addRow("Output Shapes layer", out_shapes)
            form.addRow("Label id (0=any)", label_id)
            form.addRow("Largest contour only", largest_only)
            form.addRow("Mode", mode)
            form.addRow("Time index", time_idx)
            lay.addLayout(form)

            def _collect_params() -> dict:
                lid = int(label_id.value())
                return {
                    "mask_layer": mask_layer.text().strip(),
                    "output_shapes_layer": out_shapes.text().strip(),
                    "label_id": None if lid <= 0 else lid,
                    "largest_only": bool(largest_only.isChecked()),
                    "mode": str(mode.currentData()),
                    "time_index": int(time_idx.value()),
                }

        elif kind == "color_adjustments.stack":
            form = QFormLayout()
            source_layer = QLineEdit(str(params.get("source_layer", "")))
            output_layer = QLineEdit(str(params.get("output_layer", "")))
            form.addRow("Source layer", source_layer)
            form.addRow("Output layer", output_layer)
            lay.addLayout(form)
            lay.addWidget(QLabel("Adjustment stack (JSON list of adjustment objects)"))
            params_text = QPlainTextEdit()
            params_text.setPlainText(json.dumps(list(params.get("adjustment_stack", [])), indent=2, sort_keys=True))
            params_text.setMinimumHeight(230)
            lay.addWidget(params_text)

            def _collect_params() -> dict:
                try:
                    stack = json.loads(params_text.toPlainText().strip() or "[]")
                except Exception as exc:
                    raise ValueError(f"Invalid adjustment stack JSON: {exc}") from exc
                if not isinstance(stack, list):
                    raise ValueError("Adjustment stack must be a JSON list.")
                return {
                    "source_layer": source_layer.text().strip(),
                    "output_layer": output_layer.text().strip(),
                    "adjustment_stack": stack,
                }

        else:
            params_text, _ = _add_fallback_json_editor()

            def _collect_params() -> dict:
                raw = params_text.toPlainText().strip() if params_text is not None else "{}"
                try:
                    parsed = json.loads(raw or "{}")
                except Exception as exc:
                    raise ValueError(f"Could not parse params JSON: {exc}") from exc
                if not isinstance(parsed, dict):
                    raise ValueError("Params must be a JSON object (dictionary).")
                return parsed

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return None

        new_desc = desc.text().strip() or step.description
        try:
            new_params = _collect_params()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid values", str(exc))
            return None

        return PipelineStep(
            kind=step.kind,
            description=new_desc,
            params=new_params,
            enabled=step.enabled,
        )

    def _apply_selected(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is already running.")
            return
        idx = self._list.currentRow()
        steps = PIPELINE_STORE.steps
        if not (0 <= idx < len(steps)):
            self._status.setText("Select a step first.")
            return
        self._start_processing([steps[idx].to_dict()], [steps[idx].description], [idx])

    def _apply_all_checked(self) -> None:
        if self._is_processing:
            self._status.setText("Pipeline is already running.")
            return
        steps = self._selected_steps(checked_only=True)
        if not steps:
            self._status.setText("No checked steps to apply.")
            return
        all_steps = PIPELINE_STORE.steps
        indices = [i for i, s in enumerate(all_steps) if s.enabled]
        descriptions = [all_steps[i].description for i in indices]
        for st in steps:
            if not isinstance(st, dict):
                self._status.setText("Apply failed: Invalid step format in pipeline.")
                return
        self._start_processing(steps, descriptions, indices)

    def _start_processing(self, steps: list[dict], descriptions: list[str], indices: list[int]) -> None:
        if not steps:
            return
        self._is_processing = True
        self._cancel_requested = False
        self._processing_steps = list(steps)
        self._processing_descriptions = list(descriptions)
        self._processing_indices = list(indices)
        self._processing_i = 0
        self._apply_ctx = create_apply_context(self._viewer)
        self._set_controls_enabled(False)
        self._progress.setValue(0)
        self._progress.show()
        for idx in self._processing_indices:
            self._set_item_state(idx, "idle")
        QTimer.singleShot(0, self._process_next_step)

    def _request_stop(self) -> None:
        if not self._is_processing:
            return
        self._cancel_requested = True
        self._status.setText("Stopping after current operation checkpoint...")

    def _is_cancel_requested(self) -> bool:
        return bool(self._cancel_requested)

    def _process_next_step(self) -> None:
        if not self._is_processing:
            return
        i = self._processing_i
        total = len(self._processing_steps)
        if i >= total:
            self._status.setText(f"Applied {total}/{total} step(s).")
            self._finish_processing()
            return
        step = self._processing_steps[i]
        desc = self._processing_descriptions[i] if i < len(self._processing_descriptions) else step.get("description", step.get("kind", "step"))
        idx = self._processing_indices[i] if i < len(self._processing_indices) else -1
        if idx >= 0:
            self._set_item_state(idx, "running")
        self._status.setText(f"Applying {desc} ({i+1}/{total})...")
        self._set_step_progress(i, total, 0, 1, "")
        QApplication.processEvents()
        try:
            msg = apply_pipeline_step_with_context(
                self._apply_ctx,
                step,
                progress_callback=self._on_step_progress,
                cancel_callback=self._is_cancel_requested,
            )
        except InterruptedError:
            if idx >= 0:
                self._set_item_state(idx, "idle")
            self._status.setText(f"Stopped at step {i+1}/{total}.")
            self._finish_processing()
            return
        except Exception as exc:
            if idx >= 0:
                self._set_item_state(idx, "error")
            self._status.setText(f"Apply failed at step {i+1}/{total}: {exc}")
            self._finish_processing()
            return
        if idx >= 0:
            self._set_item_state(idx, "done")
        self._status.setText(msg)
        self._set_step_progress(i, total, 1, 1, "")
        self._processing_i += 1
        QTimer.singleShot(0, self._process_next_step)

    def _on_step_progress(self, current: int, total: int, phase: str = "") -> None:
        if not self._is_processing:
            return
        if self._cancel_requested:
            raise InterruptedError("Pipeline apply cancelled by user.")
        step_idx = self._processing_i
        step_total = len(self._processing_steps)
        self._set_step_progress(step_idx, step_total, current, total, phase)
        QApplication.processEvents()

    def _set_step_progress(self, step_idx: int, step_total: int, current: int, total: int, phase: str) -> None:
        step_total_safe = max(1, int(step_total))
        total_safe = max(1, int(total))
        step_idx_clamped = max(0, min(int(step_idx), step_total_safe))
        current_clamped = max(0, min(int(current), total_safe))

        done = (step_idx_clamped + (current_clamped / total_safe)) / step_total_safe
        pct = max(0, min(100, int(round(done * 100.0))))
        self._progress.setValue(pct)
        self._progress.setFormat(f"{pct}%")

        if step_idx_clamped < step_total_safe:
            desc = self._processing_descriptions[step_idx_clamped]
            phase_txt = f" - {phase}" if phase else ""
            self._status.setText(
                f"Applying {desc} ({step_idx_clamped + 1}/{step_total_safe}){phase_txt}: "
                f"{current_clamped}/{total_safe}"
            )

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._btn_apply_selected.setEnabled(enabled)
        self._btn_apply_all.setEnabled(enabled)
        self._btn_stop.setEnabled(not enabled)
        self._btn_save.setEnabled(enabled)
        self._btn_load.setEnabled(enabled)
        self._btn_edit.setEnabled(enabled)
        self._btn_clear.setEnabled(enabled)
        if enabled:
            for w in self._disabled_widgets:
                try:
                    w.setEnabled(True)
                except Exception:
                    pass
            self._disabled_widgets.clear()
            return
        win = getattr(self._viewer, "window", None)
        dock_map = getattr(win, "_dock_widgets", {}) if win is not None else {}
        for dock_obj in dock_map.values():
            dock = dock_obj[0] if isinstance(dock_obj, tuple) else dock_obj
            dwidget = None
            if hasattr(dock, "widget"):
                try:
                    dwidget = dock.widget()
                except Exception:
                    dwidget = None
            if dwidget is None or dwidget is self:
                continue
            try:
                if dwidget.isEnabled():
                    dwidget.setEnabled(False)
                    self._disabled_widgets.append(dwidget)
            except Exception:
                continue

    def _finish_processing(self) -> None:
        self._is_processing = False
        self._cancel_requested = False
        self._processing_steps = []
        self._processing_descriptions = []
        self._processing_indices = []
        self._processing_i = 0
        self._apply_ctx = None
        self._progress.hide()
        self._set_controls_enabled(True)
