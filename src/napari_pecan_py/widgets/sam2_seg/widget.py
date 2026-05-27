"""SAM 2 interactive segmentation dock widget."""

from __future__ import annotations

import inspect
import sys
import traceback
from datetime import datetime
from typing import Any, Callable

import numpy as np
from napari.layers import Image, Labels, Points, Shapes
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_pecan_py.widgets.pecan_ellipse.logic import resolve_time_index_for_volume

from .logic import (
    CLASS_NAME_TO_ID,
    Sam2Model,
    conditioning_masks_from_labels,
    default_device,
    frame_rgb_uint8,
    gather_prompts,
    labels_2d_at_frame,
    load_sam2_backend,
    merge_class_into_labels,
    n_frames,
    prompts_ready,
    sam2_decord_available,
    summarize_prompts,
    video_path_for_sam2,
)

_POINTS_LAYER = "SAM2 prompts"
_BRUSH_LAYER = "SAM2 brush prompt"


class _WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(object)
    error = Signal(str)


class _Sam2Worker(QThread):
    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.signals = _WorkerSignals()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        def log(msg: str) -> None:
            self.signals.log.emit(str(msg))

        def progress_cb(current: int, total: int) -> None:
            self.signals.progress.emit(int(current), int(total))

        try:
            kwargs = dict(self._kwargs)
            params = inspect.signature(self._fn).parameters
            if "log" in params:
                kwargs["log"] = log
            if "progress_cb" in params:
                kwargs["progress_cb"] = progress_cb
            result = self._fn(*self._args, **kwargs)
            self.signals.finished.emit(result)
        except Exception as exc:
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _BackendLoadWorker(QThread):
    """Import torch + sam2 off the GUI thread (several seconds on first open)."""

    finished = Signal(object)

    def run(self):
        self.finished.emit(load_sam2_backend())


class Sam2SegWidget(QWidget):
    """Segment with SAM 2 using points, brush, polygon, and pipeline masks."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._model: Sam2Model | None = None
        self._worker: _Sam2Worker | None = None
        self._backend_worker: _BackendLoadWorker | None = None
        self._backend_ready = False
        self._click_mode: str | None = None  # "positive" | "negative"
        self._mouse_cb = None

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)

        root.addWidget(
            QLabel(
                "Prompt SAM 2 on the current frame, then propagate through the video. "
                "Use positive/negative clicks, a brush Labels layer, polygon Shapes, "
                "and/or a pipeline pseudo-mask Labels layer."
            )
        )

        # ---- Source video --------------------------------------------------
        src_g = QGroupBox("Source video (Image layer)")
        src_l = QVBoxLayout(src_g)
        self._image_combo = QComboBox()
        self._image_combo.currentIndexChanged.connect(self._on_layers_changed)
        src_l.addWidget(self._image_combo)
        root.addWidget(src_g)

        # ---- Model ---------------------------------------------------------
        m_g = QGroupBox("SAM 2 model (GPU)")
        self._model_form = QFormLayout(m_g)
        m_f = self._model_form
        self._device_combo = QComboBox()
        self._device_combo.addItem("cpu")
        m_f.addRow("Device:", self._device_combo)
        self._cpu_only_hint: QLabel | None = None

        self._ckpt_edit = QLineEdit()
        self._ckpt_edit.setPlaceholderText("(optional) path to .pt checkpoint")
        m_f.addRow("Checkpoint:", self._ckpt_edit)
        self._cfg_edit = QLineEdit()
        self._cfg_edit.setPlaceholderText("(optional) path to config .yaml")
        m_f.addRow("Config:", self._cfg_edit)

        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load SAM 2")
        self._load_btn.setEnabled(False)
        self._load_btn.clicked.connect(self._load_model)
        load_row.addWidget(self._load_btn)
        self._model_status = QLabel("Loading PyTorch / SAM 2…")
        self._model_status.setStyleSheet("color: #888;")
        load_row.addWidget(self._model_status, 1)
        m_f.addRow(load_row)
        root.addWidget(m_g)

        # ---- Target class --------------------------------------------------
        cls_g = QGroupBox("Target class (output Labels value)")
        cls_l = QVBoxLayout(cls_g)
        self._class_combo = QComboBox()
        for name in CLASS_NAME_TO_ID:
            self._class_combo.addItem(name, CLASS_NAME_TO_ID[name])
        cls_l.addWidget(self._class_combo)
        root.addWidget(cls_g)

        # ---- Prompts -------------------------------------------------------
        pr_g = QGroupBox("Prompts (current time slice)")
        pr_l = QVBoxLayout(pr_g)

        pt_row = QHBoxLayout()
        self._btn_pos = QPushButton("Add positive clicks")
        self._btn_pos.setCheckable(True)
        self._btn_pos.clicked.connect(lambda c: self._set_click_mode("positive", c))
        pt_row.addWidget(self._btn_pos)
        self._btn_neg = QPushButton("Add negative clicks")
        self._btn_neg.setCheckable(True)
        self._btn_neg.clicked.connect(lambda c: self._set_click_mode("negative", c))
        pt_row.addWidget(self._btn_neg)
        self._btn_clear_pts = QPushButton("Clear points")
        self._btn_clear_pts.clicked.connect(self._clear_points)
        pt_row.addWidget(self._btn_clear_pts)
        pr_l.addLayout(pt_row)

        self._btn_brush = QPushButton("Create / select brush prompt layer")
        self._btn_brush.clicked.connect(self._ensure_brush_layer)
        pr_l.addWidget(self._btn_brush)
        pr_l.addWidget(QLabel("Paint on the brush layer with napari's label tool."))

        sh_row = QHBoxLayout()
        sh_row.addWidget(QLabel("Polygon Shapes layer:"))
        self._shapes_combo = QComboBox()
        self._shapes_combo.addItem("(none)", None)
        sh_row.addWidget(self._shapes_combo, 1)
        pr_l.addLayout(sh_row)

        pm_row = QHBoxLayout()
        pm_row.addWidget(QLabel("Pipeline pseudo-mask (Labels):"))
        self._pipeline_combo = QComboBox()
        self._pipeline_combo.addItem("(none)", None)
        pm_row.addWidget(self._pipeline_combo, 1)
        pr_l.addLayout(pm_row)

        self._pipeline_label_spin = QComboBox()
        self._pipeline_label_spin.addItem("any foreground (>0)", None)
        for i in range(1, 16):
            self._pipeline_label_spin.addItem(f"label id {i}", i)
        pr_l.addWidget(QLabel("Pipeline mask label id:"))
        pr_l.addWidget(self._pipeline_label_spin)

        root.addWidget(pr_g)

        # ---- Run -----------------------------------------------------------
        run_g = QGroupBox("Run")
        run_l = QVBoxLayout(run_g)
        row = QHBoxLayout()
        self._btn_segment = QPushButton("Segment current frame")
        self._btn_segment.setEnabled(False)
        self._btn_segment.clicked.connect(self._segment_current_frame)
        row.addWidget(self._btn_segment)
        self._btn_propagate = QPushButton("Propagate in video")
        self._btn_propagate.setEnabled(False)
        self._btn_propagate.clicked.connect(self._propagate_video)
        row.addWidget(self._btn_propagate)
        run_l.addLayout(row)

        self._multimask = QCheckBox("Multimask output (pick best of 3)")
        self._multimask.setChecked(True)
        run_l.addWidget(self._multimask)

        self._preview_only = QCheckBox("Preview only (temporary layer, do not merge)")
        run_l.addWidget(self._preview_only)

        root.addWidget(run_g)

        root.addWidget(QLabel("Log"))
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMinimumHeight(140)
        self._log.setPlaceholderText("Actions and progress appear here…")
        root.addWidget(self._log)
        self._log_msg("SAM 2 widget opened; loading PyTorch / SAM 2 in background…")

        self._viewer.layers.events.inserted.connect(self._refresh_layer_combos)
        self._viewer.layers.events.removed.connect(self._refresh_layer_combos)
        self._refresh_layer_combos()

        self._backend_worker = _BackendLoadWorker()
        self._backend_worker.finished.connect(self._on_backend_loaded)
        self._backend_worker.start()

    def closeEvent(self, event):  # noqa: N802
        self._set_click_mode(None, False)
        if self._backend_worker is not None and self._backend_worker.isRunning():
            self._backend_worker.wait(3000)
        super().closeEvent(event)

    def _on_backend_loaded(self, state: dict) -> None:
        self._backend_worker = None
        if not state.get("torch_available"):
            self._model_status.setText("PyTorch not installed")
            self._model_status.setStyleSheet("color: #c44;")
            self._log_msg(
                "PyTorch is not installed in this Python environment.\n"
                "From the plugin repo: uv sync --all-extras --group dev && uv run napari"
            )
            return
        if not state.get("sam2_available"):
            detail = state.get("import_error") or "unknown import error"
            self._model_status.setText("SAM 2 import failed")
            self._model_status.setStyleSheet("color: #c44;")
            self._log_msg(
                f"SAM 2 could not be imported.\nPython: {sys.executable}\nError: {detail}\n"
                "Fix: uv sync --all-extras --group dev && uv run napari"
            )
            return

        torch_mod = state["torch"]
        self._device_combo.clear()
        self._device_combo.addItem("cpu")
        if torch_mod.cuda.is_available():
            for i in range(torch_mod.cuda.device_count()):
                self._device_combo.addItem(f"cuda:{i}")
        elif getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
            self._device_combo.addItem("mps")
        elif "+cpu" in torch_mod.__version__ and sys.platform in ("win32", "linux"):
            self._cpu_only_hint = QLabel(
                f"PyTorch is CPU-only ({torch_mod.__version__}). From the repo:\n"
                "  uv lock && uv sync --all-extras --group dev\n"
                "Then restart napari."
            )
            self._model_form.addRow("", self._cpu_only_hint)

        dev = default_device()
        idx = self._device_combo.findText(dev)
        if idx >= 0:
            self._device_combo.setCurrentIndex(idx)

        self._backend_ready = True
        self._load_btn.setEnabled(True)
        self._btn_segment.setEnabled(True)
        self._btn_propagate.setEnabled(True)
        self._model_status.setText("Model not loaded")
        self._model_status.setStyleSheet("color: #888;")
        self._log_msg("SAM 2 backend ready.")

    def _log_msg(self, msg: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S")
        self._log.append(f"[{stamp}] {msg}")
        self._log.verticalScrollBar().setValue(self._log.verticalScrollBar().maximum())
        QApplication.processEvents()

    def _worker_busy(self) -> bool:
        if self._worker is not None and self._worker.isRunning():
            self._log_msg("Busy — wait for the current job to finish.")
            return True
        return False

    def _refresh_layer_combos(self, *_args) -> None:
        cur_img = self._image_combo.currentData() if hasattr(self, "_image_combo") else None
        self._image_combo.blockSignals(True)
        self._image_combo.clear()
        self._image_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self._image_combo.addItem(layer.name, layer)
        if cur_img is not None:
            for i in range(self._image_combo.count()):
                if self._image_combo.itemData(i) is cur_img:
                    self._image_combo.setCurrentIndex(i)
                    break
        self._image_combo.blockSignals(False)

        cur_sh = self._shapes_combo.currentData() if hasattr(self, "_shapes_combo") else None
        self._shapes_combo.blockSignals(True)
        self._shapes_combo.clear()
        self._shapes_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Shapes):
                self._shapes_combo.addItem(layer.name, layer)
        self._shapes_combo.blockSignals(False)

        cur_pm = self._pipeline_combo.currentData() if hasattr(self, "_pipeline_combo") else None
        self._pipeline_combo.blockSignals(True)
        self._pipeline_combo.clear()
        self._pipeline_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                self._pipeline_combo.addItem(layer.name, layer)
        self._pipeline_combo.blockSignals(False)

    def _on_layers_changed(self) -> None:
        pass

    def _selected_image(self) -> Image | None:
        layer = self._image_combo.currentData()
        return layer if isinstance(layer, Image) else None

    def _current_frame_index(self) -> int:
        layer = self._selected_image()
        if layer is None:
            return 0
        return resolve_time_index_for_volume(layer.data, self._viewer)

    def _remove_layer_by_name(self, name: str) -> None:
        try:
            if name in self._viewer.layers:
                self._viewer.layers.remove(self._viewer.layers[name])
        except (KeyError, ValueError):
            pass

    def _go_to_frame(self, frame_index: int) -> None:
        """Move the viewer time slider to ``frame_index`` when the source is a video."""
        layer = self._selected_image()
        if layer is None:
            return
        shape = getattr(layer.data, "shape", None)
        if shape is None or len(shape) != 4:
            return
        try:
            steps = tuple(int(x) for x in self._viewer.dims.nsteps)
            curr = list(int(x) for x in self._viewer.dims.current_step)
            t_size = int(shape[0])
            match_axes = [i for i, n in enumerate(steps) if n == t_size]
            axis = match_axes[0] if len(match_axes) == 1 else 0
            if axis < len(curr):
                curr[axis] = int(np.clip(int(frame_index), 0, t_size - 1))
                self._viewer.dims.current_step = curr
        except Exception:
            pass

    def _find_points_layer(self) -> tuple[Points | None, str]:
        """Prefer widget prompt layer, then selected Points, then any Points layer."""
        try:
            return self._viewer.layers[_POINTS_LAYER], _POINTS_LAYER
        except KeyError:
            pass
        active = self._viewer.layers.selection.active
        if isinstance(active, Points):
            return active, active.name
        for layer in self._viewer.layers:
            if isinstance(layer, Points):
                return layer, layer.name
        return None, ""

    def _pos_to_t_yx(self, event, layer: Image) -> tuple[int | None, int | None, int | None]:
        try:
            if hasattr(layer, "world_to_data"):
                data_pos = layer.world_to_data(event.position)
            else:
                data_pos = event.position
            coords = np.asarray(data_pos, dtype=float).ravel()
        except Exception:
            return None, None, None
        data = layer.data
        shape = getattr(data, "shape", None)
        if shape is None:
            return None, None, None
        if len(shape) == 4:
            if coords.size < 2:
                return None, None, None
            # Use the viewer time slider (not event axis 0, which is often wrong in 2D view).
            t = self._current_frame_index()
            y = int(round(float(coords[-2])))
            x = int(round(float(coords[-1])))
        elif len(shape) == 3:
            t = 0
            y = int(round(coords[0]))
            x = int(round(coords[1]))
        else:
            return None, None, None
        h, w = int(shape[-3]), int(shape[-2])
        if not (0 <= y < h and 0 <= x < w):
            return None, None, None
        return t, y, x

    def _ensure_points_layer(self) -> Points:
        try:
            return self._viewer.layers[_POINTS_LAYER]
        except KeyError:
            pass
        layer = self._selected_image()
        ndim = 3
        if layer is not None:
            sh = getattr(layer.data, "shape", None)
            if sh is not None and len(sh) == 3:
                ndim = 2
        pts = self._viewer.add_points(
            name=_POINTS_LAYER,
            ndim=ndim,
            properties={"label": np.array([1], dtype=int)},
            face_color="green",
        )
        return pts

    def _ensure_brush_layer(self) -> None:
        layer = self._selected_image()
        if layer is None:
            self._log_msg("Select a source image layer first.")
            return
        data = layer.data
        shape = getattr(data, "shape", None)
        if shape is None:
            return
        try:
            lyr = self._viewer.layers[_BRUSH_LAYER]
            self._viewer.layers.selection.active = lyr
            self._log_msg(f"Selected brush layer '{_BRUSH_LAYER}'.")
            return
        except KeyError:
            pass
        if len(shape) == 4:
            empty = np.zeros(shape[:3], dtype=np.uint32)
        else:
            empty = np.zeros(shape[:2], dtype=np.uint32)
        lyr = self._viewer.add_labels(empty, name=_BRUSH_LAYER)
        self._viewer.layers.selection.active = lyr
        self._log_msg(f"Created '{_BRUSH_LAYER}' — paint with the label tool.")

    def _clear_points(self) -> None:
        try:
            self._viewer.layers.remove(_POINTS_LAYER)
        except KeyError:
            pass
        self._log_msg("Cleared prompt points.")

    def _set_click_mode(self, mode: str | None, active: bool) -> None:
        if self._mouse_cb is not None:
            try:
                self._viewer.mouse_drag_callbacks.remove(self._mouse_cb)
            except ValueError:
                pass
            self._mouse_cb = None
        self._click_mode = None
        self._btn_pos.blockSignals(True)
        self._btn_neg.blockSignals(True)
        self._btn_pos.setChecked(False)
        self._btn_neg.setChecked(False)
        self._btn_pos.blockSignals(False)
        self._btn_neg.blockSignals(False)

        if not active or mode is None:
            return
        self._click_mode = mode
        if mode == "positive":
            self._btn_pos.setChecked(True)
        else:
            self._btn_neg.setChecked(True)
        self._mouse_cb = self._on_image_click
        self._viewer.mouse_drag_callbacks.append(self._mouse_cb)
        self._log_msg(f"Click on the image to add {mode} points (left button).")

    def _on_image_click(self, viewer, event) -> None:
        if self._click_mode is None or event.button != 1:
            return
        layer = self._selected_image()
        if layer is None:
            return
        t, y, x = self._pos_to_t_yx(event, layer)
        if t is None:
            return
        label = 1 if self._click_mode == "positive" else 0
        pts_layer = self._ensure_points_layer()
        data = np.asarray(pts_layer.data)
        layer = self._selected_image()
        sh = getattr(layer.data, "shape", None) if layer else None
        if sh is not None and len(sh) == 3:
            new_row = np.array([[y, x]], dtype=float)
        else:
            new_row = np.array([[t, y, x]], dtype=float)
        if data.size == 0:
            combined = new_row
        else:
            combined = np.vstack([data, new_row])
        pts_layer.data = combined
        # Store labels in properties — napari Points properties per point
        n = len(combined)
        props = pts_layer.properties or {}
        labels = np.asarray(props.get("label", np.ones(n, dtype=int)))
        if labels.size < n:
            labels = np.concatenate([labels, np.array([label], dtype=int)])
        else:
            labels = labels.copy()
            labels[-1] = label
        pts_layer.properties = {"label": labels}
        pts_layer.face_color = ["green" if int(l) else "red" for l in labels]
        pts_layer.refresh()

    def _collect_prompts(
        self,
        frame_index: int,
        shape_hw: tuple[int, int],
        *,
        ignore_time_filter: bool = False,
    ) -> dict:
        points_data = None
        points_layer, points_name = self._find_points_layer()
        if points_layer is not None:
            pdata = np.asarray(points_layer.data)
            props = points_layer.properties or {}
            labs = np.asarray(props.get("label", []))
            if pdata.size > 0:
                rows = []
                for i, row in enumerate(pdata):
                    lab = int(labs[i]) if i < len(labs) else 1
                    if row.size >= 3:
                        rows.append([row[0], row[1], row[2], lab])
                    elif row.size >= 2:
                        rows.append([0, row[0], row[1], lab])
                if rows:
                    points_data = np.array(rows, dtype=float)
            self._last_points_layer_name = points_name
            self._last_raw_point_count = int(len(pdata)) if pdata.size else 0
        else:
            self._last_points_layer_name = ""
            self._last_raw_point_count = 0

        brush_2d = None
        try:
            bl = self._viewer.layers[_BRUSH_LAYER]
            brush_2d = labels_2d_at_frame(bl.data, frame_index)
        except KeyError:
            pass

        shapes_data = None
        sh = self._shapes_combo.currentData()
        if isinstance(sh, Shapes):
            shapes_data = list(sh.data)

        pipeline_2d = None
        pl_id = self._pipeline_label_spin.currentData()
        pm = self._pipeline_combo.currentData()
        if isinstance(pm, Labels):
            pipeline_2d = labels_2d_at_frame(pm.data, frame_index)

        return gather_prompts(
            frame_index=frame_index,
            shape_hw=shape_hw,
            points_layer_data=points_data,
            brush_labels_2d=brush_2d,
            shapes_data=shapes_data,
            pipeline_mask_2d=pipeline_2d,
            pipeline_label_id=pl_id,
            ignore_time_filter=ignore_time_filter,
        )

    def _output_layer_name(self) -> str:
        layer = self._selected_image()
        base = layer.name if layer else "image"
        return f"{base} - SAM2"

    def _get_or_create_output_labels(self) -> Labels:
        name = self._output_layer_name()
        try:
            return self._viewer.layers[name]
        except KeyError:
            pass
        src = self._selected_image()
        if src is None:
            raise ValueError("No source image")
        data = src.data
        shape = getattr(data, "shape", None)
        if shape is None:
            raise ValueError("Source has no shape")
        if len(shape) == 4:
            empty = np.zeros(shape[:3], dtype=np.uint32)
        else:
            empty = np.zeros(shape[:2], dtype=np.uint32)
        return self._viewer.add_labels(empty, name=name)

    def _load_model(self) -> None:
        if not self._backend_ready:
            self._log_msg("Cannot load model yet — backend still initializing.")
            return
        if self._worker_busy():
            return

        ckpt = self._ckpt_edit.text().strip() or None
        cfg = self._cfg_edit.text().strip() or None
        device = self._device_combo.currentText()
        self._log_msg(
            f"Load SAM 2: device={device}, "
            f"checkpoint={ckpt or '(auto download)'}, config={cfg or '(bundled)'}"
        )

        def _do(*, log: Callable[[str], None]):
            log("Resolving config and checkpoint paths…")
            model = Sam2Model(config_path=cfg, checkpoint_path=ckpt, device=device)
            log(f"Config: {model.config_path}")
            log(f"Checkpoint: {model.checkpoint_path}")
            log(f"Loading weights onto {model.device} (may take 1–2 minutes)…")
            model.warmup()
            log("Weights loaded.")
            return model

        self._load_btn.setEnabled(False)
        self._model_status.setText("Loading…")
        self._model_status.setStyleSheet("color: #888;")
        self._start_worker(_do, self._on_model_loaded, disable_load=True)

    def _start_worker(
        self,
        fn: Callable[..., Any],
        on_done: Callable[[Any], None],
        *,
        disable_load: bool = False,
    ) -> None:
        w = _Sam2Worker(fn)
        w.signals.log.connect(self._log_msg)
        w.signals.progress.connect(
            lambda c, t: self._log_msg(f"Progress: {c}/{t} frames")
        )
        w.signals.finished.connect(on_done)
        w.signals.error.connect(self._on_worker_error)
        self._worker = w
        if not disable_load:
            self._btn_segment.setEnabled(False)
            self._btn_propagate.setEnabled(False)
        w.start()

    def _on_model_loaded(self, model: Sam2Model) -> None:
        self._model = model
        self._load_btn.setEnabled(True)
        self._model_status.setText(f"Loaded on {model.device}")
        self._model_status.setStyleSheet("color: #6a6;")
        self._log_msg(f"SAM 2 ready ({model.device}).")

    def _ensure_model(self) -> Sam2Model | None:
        if not self._backend_ready:
            self._log_msg("Still loading PyTorch / SAM 2…")
            return None
        if self._model is None:
            self._log_msg("Load SAM 2 first.")
            return None
        return self._model

    def _segment_current_frame(self) -> None:
        try:
            self._run_segment_current_frame()
        except Exception as exc:
            self._log_msg(f"ERROR (before worker):\n{exc}\n{traceback.format_exc()}")

    def _run_segment_current_frame(self) -> None:
        self._log_msg("Segment current frame: clicked.")
        if self._worker_busy():
            return
        model = self._ensure_model()
        if model is None:
            return
        layer = self._selected_image()
        if layer is None:
            self._log_msg("Select a source image layer in the dropdown.")
            return
        t = self._current_frame_index()
        try:
            frame = frame_rgb_uint8(layer.data, t)
        except Exception as exc:
            self._log_msg(f"Could not read frame {t}: {exc}")
            return
        h, w = frame.shape[:2]
        prompts = self._collect_prompts(t, (h, w))
        pts_name = getattr(self, "_last_points_layer_name", "")
        raw_pts = getattr(self, "_last_raw_point_count", 0)
        if pts_name:
            self._log_msg(f"Points layer '{pts_name}': {raw_pts} point(s) in layer.")
        elif raw_pts == 0:
            self._log_msg(
                f"No Points layer found (create one with '{_POINTS_LAYER}' or Add positive clicks)."
            )
        self._log_msg(
            f"Source '{layer.name}', frame {t}/{max(0, n_frames(layer.data) - 1)}, "
            f"size {w}×{h}. {summarize_prompts(prompts)}"
        )
        if not prompts_ready(prompts) and raw_pts > 0:
            self._log_msg(
                f"Points on '{pts_name}' do not match frame {t} (time slider / axis order). "
                "Retrying with all points regardless of time…"
            )
            prompts = self._collect_prompts(t, (h, w), ignore_time_filter=True)
            self._log_msg(summarize_prompts(prompts))
        pm = self._pipeline_combo.currentData()
        if isinstance(pm, Labels):
            pl2d = labels_2d_at_frame(pm.data, t)
            if pl2d is not None:
                pl_id = self._pipeline_label_spin.currentData()
                fg = int(np.count_nonzero(pl2d if pl_id is None else pl2d == int(pl_id)))
                self._log_msg(
                    f"Pipeline mask '{pm.name}' on frame {t}: {fg} foreground pixels."
                )
            else:
                self._log_msg(f"Pipeline mask layer: '{pm.name}' (could not read frame {t}).")
        if not prompts_ready(prompts):
            self._log_msg(
                "No prompts on this frame. Add positive/negative clicks (use the widget's "
                "Add positive clicks button), paint a brush layer, pick a polygon Shapes layer, "
                "or select a pipeline pseudo-mask Labels layer."
            )
            return
        class_id = int(self._class_combo.currentData())
        class_name = self._class_combo.currentText()
        multimask = self._multimask.isChecked()
        preview = self._preview_only.isChecked()
        self._log_msg(
            f"Running SAM 2 (class={class_name}, id={class_id}, "
            f"multimask={multimask}, preview_only={preview})…"
        )

        def _run(*, log: Callable[[str], None]):
            log("Predicting mask for current frame…")
            mask, score = model.predict_frame(frame, prompts, multimask_output=multimask)
            log(f"Prediction done (score={score:.4f}).")
            return mask, score, t, class_id, preview

        self._start_worker(_run, self._on_frame_done)

    def _propagate_video(self) -> None:
        self._log_msg("Propagate in video: clicked.")
        if self._worker_busy():
            return
        model = self._ensure_model()
        if model is None:
            return
        layer = self._selected_image()
        if layer is None:
            self._log_msg("Select a source image layer in the dropdown.")
            return
        data = layer.data
        nf = n_frames(data)
        if nf < 2:
            self._log_msg(f"Video has {nf} frame(s); need at least 2 for propagation.")
            return
        t = self._current_frame_index()
        video_path = video_path_for_sam2(data)
        use_video_file = bool(video_path and sam2_decord_available())
        if use_video_file:
            self._log_msg(
                f"Propagating '{layer.name}' from file ({nf} frames, seed frame {t})…"
            )
            frame = frame_rgb_uint8(data, t)
            video_source: np.ndarray | str = video_path  # type: ignore[assignment]
        else:
            if video_path and not sam2_decord_available():
                self._log_msg(
                    "Package 'decord' is not installed; exporting frames for SAM2 "
                    "(run: uv sync). This may take a minute…"
                )
            else:
                self._log_msg(f"Preparing {nf} frames from '{layer.name}' (seed frame {t})…")
            frames = [frame_rgb_uint8(data, fi) for fi in range(nf)]
            video_source = np.stack(frames, axis=0)
            frame = video_source[t]
        h, w = int(frame.shape[0]), int(frame.shape[1])
        prompts = self._collect_prompts(t, (h, w))
        raw_pts = getattr(self, "_last_raw_point_count", 0)
        self._log_msg(summarize_prompts(prompts))
        if not prompts_ready(prompts) and raw_pts > 0:
            self._log_msg(
                f"Points do not match seed frame {t}; retrying with all points…"
            )
            prompts = self._collect_prompts(t, (h, w), ignore_time_filter=True)
            self._log_msg(summarize_prompts(prompts))
        if not prompts_ready(prompts):
            self._log_msg(
                "No prompts on the seed frame. Add clicks, brush, polygon, or pipeline mask."
            )
            return
        class_id = int(self._class_combo.currentData())
        preview = self._preview_only.isChecked()
        extra_cond: list[tuple[int, np.ndarray]] = []
        try:
            out_lyr = self._viewer.layers[self._output_layer_name()]
            extra_cond = conditioning_masks_from_labels(out_lyr.data, class_id)
            extra_cond = [(fi, m) for fi, m in extra_cond if fi != t]
        except KeyError:
            pass
        if extra_cond:
            frames_txt = ", ".join(str(fi) for fi, _ in extra_cond[:6])
            if len(extra_cond) > 6:
                frames_txt += ", …"
            self._log_msg(
                f"Also using {len(extra_cond)} existing label frame(s) as seeds: {frames_txt}"
            )
        self._log_msg(
            f"Propagating through {nf} frames (class id={class_id}, preview_only={preview})…"
        )

        def _run(*, log: Callable[[str], None], progress_cb: Callable[[int, int], None]):
            if isinstance(video_source, str):
                log("Starting video propagation (reading video file)…")
            else:
                log("Starting video propagation (writing temp frames)…")

            def _progress(done: int, total: int) -> None:
                progress_cb(done, total)
                if done == 1 or done == total or done % max(1, total // 20) == 0:
                    log(f"Propagating… {done}/{total} frames filled")

            masks = model.propagate_video(
                video_source,
                t,
                prompts,
                obj_id=1,
                progress_callback=_progress,
                extra_conditioning=extra_cond,
            )
            nz = [int(i) for i in range(masks.shape[0]) if np.any(masks[i])]
            if nz:
                log(
                    f"Propagation finished — frames {nz[0]}–{nz[-1]} have masks "
                    f"({len(nz)}/{masks.shape[0]} frames)."
                )
            else:
                log("Propagation finished — no foreground masks produced.")
            return masks, class_id, preview

        self._start_worker(_run, self._on_video_done)

    def _on_frame_done(self, payload) -> None:
        self._btn_segment.setEnabled(True)
        self._btn_propagate.setEnabled(True)
        mask, score, t, class_id, preview = payload
        n_px = int(mask.sum())
        self._log_msg(f"Done — frame {t}: score={score:.3f}, foreground pixels={n_px}")
        if n_px == 0:
            self._log_msg(
                "SAM 2 returned an empty mask. Try more points, a tighter pipeline mask, "
                "or a different target class."
            )
        if preview:
            name = f"SAM2 preview (t={t})"
            self._remove_layer_by_name(name)
            h, w = int(mask.shape[0]), int(mask.shape[1])
            src = self._selected_image()
            nf = n_frames(src.data) if src is not None else 1
            if nf > 1:
                preview_vol = np.zeros((nf, h, w), dtype=np.uint32)
                preview_vol[t][mask] = int(class_id)
                lyr = self._viewer.add_labels(preview_vol, name=name)
            else:
                preview_data = np.zeros((h, w), dtype=np.uint32)
                preview_data[mask] = int(class_id)
                lyr = self._viewer.add_labels(preview_data, name=name)
            lyr.visible = True
            lyr.opacity = 0.6
            self._viewer.layers.selection.active = lyr
            self._go_to_frame(t)
            self._log_msg(
                f"Added preview layer '{name}' ({n_px} px) on frame {t}."
            )
        else:
            try:
                out = self._get_or_create_output_labels()
                data = out.data
                if np.asarray(data).ndim == 3:
                    merged = merge_class_into_labels(data, mask, class_id, frame_index=t)
                else:
                    merged = merge_class_into_labels(data, mask, class_id)
                out.data = merged
                out.visible = True
                out.opacity = 0.6
                out.refresh()
                self._viewer.layers.selection.active = out
                self._go_to_frame(t)
                self._log_msg(
                    f"Merged class id {class_id} into '{out.name}' on frame {t} ({n_px} px)."
                )
            except Exception as exc:
                self._log_msg(f"ERROR writing labels:\n{exc}\n{traceback.format_exc()}")

    def _on_video_done(self, payload) -> None:
        self._btn_segment.setEnabled(True)
        self._btn_propagate.setEnabled(True)
        masks, class_id, preview = payload
        nz_frames = [int(i) for i in range(masks.shape[0]) if np.any(masks[i])]
        total_px = int(masks.sum())
        self._log_msg(
            f"Done — {masks.shape[0]} frames, {len(nz_frames)} with mask, "
            f"{total_px} total foreground pixels."
        )
        if nz_frames:
            sample = nz_frames[:8]
            extra = "…" if len(nz_frames) > 8 else ""
            self._log_msg(f"Frames with mask (sample): {sample}{extra}")
        if preview:
            name = "SAM2 preview (video)"
            self._remove_layer_by_name(name)
            preview_vol = np.zeros(masks.shape, dtype=np.uint32)
            preview_vol[masks] = int(class_id)
            lyr = self._viewer.add_labels(preview_vol, name=name)
            lyr.visible = True
            lyr.opacity = 0.6
            self._viewer.layers.selection.active = lyr
            self._log_msg(f"Added preview layer '{name}'.")
        else:
            try:
                out = self._get_or_create_output_labels()
                merged = merge_class_into_labels(out.data, masks, class_id)
                out.data = merged
                out.visible = True
                out.opacity = 0.6
                out.refresh()
                self._viewer.layers.selection.active = out
                self._log_msg(f"Merged class id {class_id} into '{out.name}' ({len(nz_frames)} frames).")
            except Exception as exc:
                self._log_msg(f"ERROR writing labels:\n{exc}\n{traceback.format_exc()}")

    def _on_worker_error(self, msg: str) -> None:
        self._btn_segment.setEnabled(True)
        self._btn_propagate.setEnabled(True)
        self._load_btn.setEnabled(True)
        if self._model is None:
            self._model_status.setText("Load failed")
            self._model_status.setStyleSheet("color: #c44;")
        self._log_msg(f"ERROR:\n{msg}")
