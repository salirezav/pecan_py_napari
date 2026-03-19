"""YOLO Segmentation training widget.

Uses `ultralytics` YOLO segmentation, with:
- Image layer (MP4 frames) as input
- Selected Labels layers (pecan, kernel, crack, etc.) as segmentation masks
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_ULTRA_AVAILABLE = False
try:
    import ultralytics  # type: ignore[import]

    _ULTRA_AVAILABLE = True
except Exception:
    pass


class _TrainSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)  # current_epoch, total_epochs
    finished = Signal(str)  # best weights path
    error = Signal(str)


class _TrainWorker(QThread):
    def __init__(
        self,
        image_data: np.ndarray,
        masks_by_class: Dict[str, np.ndarray],
        epochs: int,
        batch: int,
        lr: float,
        device: str,
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self._image_data = image_data
        self._masks_by_class = masks_by_class
        self._epochs = epochs
        self._batch = batch
        self._lr = lr
        self._device = device

        self._proc: subprocess.Popen | None = None
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass

    def run(self):
        try:
            from .model import export_napari_seg_dataset

            with tempfile.TemporaryDirectory(prefix="pecan_yolo_seg_") as tmpdir:
                spec = export_napari_seg_dataset(
                    self._image_data, self._masks_by_class, tmpdir
                )

                project_dir = Path(tmpdir) / "runs"
                run_name = "pecan-yolo-seg"
                project_dir.mkdir(parents=True, exist_ok=True)

                # Subprocess so we can stream stdout/stderr and kill it on stop.
                code = r"""
import sys
from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')
model.train(
    data=sys.argv[1],
    epochs=int(sys.argv[2]),
    batch=int(sys.argv[3]),
    lr0=float(sys.argv[4]),
    device=sys.argv[5],
    project=sys.argv[6],
    name=sys.argv[7],
    verbose=True,
)
"""
                cmd = [
                    sys.executable,
                    "-c",
                    code,
                    str(spec.data_yaml),
                    str(self._epochs),
                    str(self._batch),
                    str(self._lr),
                    str(self._device),
                    str(project_dir),
                    run_name,
                ]

                self.signals.log.emit("Starting YOLO training…")
                # Ensure UTF-8 stdout decoding in the subprocess.
                env = os.environ.copy()
                env["PYTHONIOENCODING"] = "utf-8"
                env["PYTHONUTF8"] = "1"

                self._proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    env=env,
                )

                assert self._proc.stdout is not None
                for line in self._proc.stdout:
                    if self._stop_requested:
                        break
                    line = line.rstrip("\n")
                    if not line:
                        continue
                    self.signals.log.emit(line)

                    m = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line)
                    if m:
                        cur = int(m.group(1))
                        tot = int(m.group(2))
                        self.signals.progress.emit(cur, tot)

                self._proc.wait()
                if self._stop_requested:
                    self.signals.error.emit("Training stopped by user.")
                    return
                if self._proc.returncode != 0:
                    self.signals.error.emit(f"Training failed (exit code {self._proc.returncode}).")
                    return

                # Find best weights anywhere under project_dir.
                best_pt = None
                best_mtime = -1.0
                for p in project_dir.rglob("best.pt"):
                    try:
                        mt = p.stat().st_mtime
                        if mt > best_mtime:
                            best_mtime = mt
                            best_pt = p
                    except Exception:
                        pass
                if best_pt is None:
                    self.signals.error.emit("Could not find best.pt after training.")
                    return

                self.signals.finished.emit(str(best_pt))

        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class YoloSegWidget(QWidget):
    """Napari dock widget for YOLO segmentation training and inference."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._best_weights: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not _ULTRA_AVAILABLE:
            layout.addWidget(
                QLabel(
                    "The 'ultralytics' package is not installed.\n"
                    "Install it with:\n"
                    "  pip install ultralytics\n"
                    "then restart napari."
                )
            )
            return

        # Image layer
        img_group = QGroupBox("Image layer")
        img_lay = QVBoxLayout(img_group)
        self._image_combo = QComboBox()
        self._image_combo.addItem("(none)", None)
        img_lay.addWidget(self._image_combo)
        layout.addWidget(img_group)

        # Class masks
        mask_group = QGroupBox("Class masks")
        mask_lay = QVBoxLayout(mask_group)
        mask_lay.addWidget(QLabel("Check Labels layers to use as classes:"))
        self._mask_container = QVBoxLayout()
        mask_lay.addLayout(self._mask_container)
        self._mask_checkboxes: List[tuple[QCheckBox, Labels]] = []
        layout.addWidget(mask_group)

        # Training params
        train_group = QGroupBox("Training")
        train_lay = QVBoxLayout(train_group)

        self._epochs_spin = self._spin_row(train_lay, "Epochs:", 50, 1, 500, 1)
        self._batch_spin = self._spin_row(train_lay, "Batch size:", 4, 1, 64, 1)
        self._lr_dspin = self._dspin_row(train_lay, "Learning rate:", 1e-3, 1e-5, 1.0, 1e-4)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItem("auto", "auto")
        self._device_combo.addItem("cpu", "cpu")
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    self._device_combo.addItem(f"cuda:{i}", str(i))
        except Exception:
            pass
        dev_row.addWidget(self._device_combo)
        train_lay.addLayout(dev_row)

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train YOLO")
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)

        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)

        self._load_btn = QPushButton("Load weights…")
        self._load_btn.clicked.connect(self._load_weights)
        btn_row.addWidget(self._load_btn)
        train_lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        train_lay.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(160)
        train_lay.addWidget(self._log)

        layout.addWidget(train_group)

        # Inference
        infer_group = QGroupBox("Inference")
        infer_lay = QVBoxLayout(infer_group)
        self._run_btn = QPushButton("Run on current frame")
        self._run_btn.clicked.connect(self._run_inference)
        infer_lay.addWidget(self._run_btn)
        layout.addWidget(infer_group)

        layout.addStretch(1)

        self._refresh_layers()
        self._viewer.layers.events.inserted.connect(self._refresh_layers)
        self._viewer.layers.events.removed.connect(self._refresh_layers)

    # helpers
    def _spin_row(self, parent, label, default, lo, hi, step):
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _dspin_row(self, parent, label, default, lo, hi, step):
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _refresh_layers(self, _evt=None):
        prev_img = self._image_combo.currentData()
        self._image_combo.clear()
        self._image_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                self._image_combo.addItem(layer.name, layer)
        if prev_img is not None and prev_img in self._viewer.layers:
            idx = self._image_combo.findData(prev_img)
            if idx >= 0:
                self._image_combo.setCurrentIndex(idx)

        for cb, _ in self._mask_checkboxes:
            cb.setParent(None)
        self._mask_checkboxes.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                cb = QCheckBox(layer.name)
                cb.setChecked(True)
                self._mask_container.addWidget(cb)
                self._mask_checkboxes.append((cb, layer))

    def _selected_image(self) -> Image | None:
        data = self._image_combo.currentData()
        if data is not None and data in self._viewer.layers:
            return data
        return None

    def _selected_masks(self) -> Dict[str, np.ndarray]:
        masks: Dict[str, np.ndarray] = {}
        for cb, layer in self._mask_checkboxes:
            if cb.isChecked() and layer in self._viewer.layers:
                masks[layer.name] = np.asarray(layer.data)
        return masks

    # training
    def _start_training(self):
        if not _ULTRA_AVAILABLE:
            from napari.utils.notifications import show_warning

            show_warning("ultralytics is not installed.")
            return

        img_layer = self._selected_image()
        if img_layer is None:
            from napari.utils.notifications import show_warning

            show_warning("Select an image layer first.")
            return
        masks = self._selected_masks()
        if not masks:
            from napari.utils.notifications import show_warning

            show_warning("Check at least one Labels layer as a class.")
            return

        data = np.asarray(img_layer.data)
        if data.ndim == 3 and data.shape[-1] in (3, 4):
            rgb = data[..., :3]
        elif data.ndim == 4 and data.shape[-1] in (3, 4):
            rgb = data[..., :3]
        else:
            from napari.utils.notifications import show_warning

            show_warning("Expected RGB frames (T, H, W, 3).")
            return

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255)
            if float(np.max(rgb)) <= 1.0:
                rgb = rgb * 255.0
            rgb = rgb.astype(np.uint8)

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._progress.setValue(0)
        self._log.clear()
        self._log.append("Starting training…")

        device_value = self._device_combo.currentData()
        if device_value is None:
            device_value = self._device_combo.currentText()

        self._worker = _TrainWorker(
            image_data=rgb,
            masks_by_class=masks,
            epochs=self._epochs_spin.value(),
            batch=self._batch_spin.value(),
            lr=self._lr_dspin.value(),
            device=str(device_value),
        )
        self._worker.signals.log.connect(self._log.append)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.finished.connect(self._on_training_finished)
        self._worker.signals.error.connect(self._on_training_error)
        self._worker.start()

    def _stop_training(self):
        if getattr(self, "_worker", None) is None:
            return
        self._log.append("Stop requested…")
        try:
            self._worker.stop()
        except Exception:
            pass

    def _on_progress(self, cur_epoch: int, total_epochs: int):
        if total_epochs <= 0:
            self._progress.setValue(0)
            return
        pct = int(100 * cur_epoch / total_epochs)
        self._progress.setValue(max(0, min(100, pct)))

    def _on_training_finished(self, path: str):
        self._best_weights = path
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        self._log.append(f"Training complete. Best weights: {path}")
        from napari.utils.notifications import show_info

        show_info(f"YOLO training complete. Best weights:\n{path}")

    def _on_training_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log.append(f"ERROR: {msg}")
        from napari.utils.notifications import show_error

        show_error(f"YOLO training error:\n{msg}")

    def _load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO weights", "", "YOLO weights (*.pt *.pth)"
        )
        if not path:
            return
        self._best_weights = path
        from napari.utils.notifications import show_info

        show_info(f"Using weights: {path}")

    def _run_inference(self):
        if self._best_weights is None:
            from napari.utils.notifications import show_warning

            show_warning("Train or load YOLO weights first.")
            return
        img_layer = self._selected_image()
        if img_layer is None:
            from napari.utils.notifications import show_warning

            show_warning("Select an image layer first.")
            return

        data = np.asarray(img_layer.data)
        if data.ndim == 4:
            t = int(self._viewer.dims.current_step[0]) if self._viewer.dims.ndim > 0 else 0
            t = max(0, min(t, data.shape[0] - 1))
            frame = data[t]
        elif data.ndim == 3:
            frame = data
        else:
            from napari.utils.notifications import show_warning

            show_warning("Unsupported image dimensionality.")
            return

        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255)
            if float(np.max(frame)) <= 1.0:
                frame = frame * 255.0
            frame = frame.astype(np.uint8)

        from ultralytics import YOLO

        model = YOLO(self._best_weights)
        res = model.predict(frame, imgsz=frame.shape[0], verbose=False)
        if not res or res[0].masks is None:
            from napari.utils.notifications import show_warning

            show_warning("No masks predicted.")
            return

        masks = res[0].masks.data.cpu().numpy().astype(bool)
        # convert [N, H, W] one-hot-ish to single label map
        H, W = masks.shape[1:]
        label_map = np.zeros((H, W), dtype=np.uint8)
        for i in range(masks.shape[0]):
            label_map[masks[i]] = i + 1

        name = f"{img_layer.name} - YOLO seg"
        from napari.utils.colormaps import label_colormap

        try:
            existing = self._viewer.layers[name]
            existing.data = label_map
            existing.refresh()
        except Exception:
            self._viewer.add_labels(label_map, name=name, opacity=0.5)


