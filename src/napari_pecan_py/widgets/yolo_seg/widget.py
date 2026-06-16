"""YOLO Segmentation widget with separate Inference and Training sections."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QObject, QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from napari_pecan_py._reader import VIDEO_EXTENSIONS

from .model import (
    IMAGE_EXTENSIONS,
    discover_mask_files,
    export_videos_seg_dataset,
    format_dataset_summary,
    inference_imgsz,
    load_image_rgb,
    load_video_rgb_frames,
    summarize_training_dataset,
    to_yolo_predict_source,
    yolo_result_to_label_map,
)

_ULTRA_AVAILABLE = False
try:
    import ultralytics  # type: ignore[import]

    _ULTRA_AVAILABLE = True
except Exception:
    pass

_MEDIA_FILTER = (
    "Media files ("
    + " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS | IMAGE_EXTENSIONS))
    + ")"
)


class _TrainSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(str)
    error = Signal(str)


class _InferSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)
    finished = Signal(list)
    error = Signal(str)


class _TrainWorker(QThread):
    def __init__(
        self,
        video_entries: Sequence[Tuple[str, Dict[str, str]]],
        epochs: int,
        batch: int,
        lr: float,
        device: str,
        output_dir: str,
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self._video_entries = list(video_entries)
        self._epochs = epochs
        self._batch = batch
        self._lr = lr
        self._device = device
        self._output_dir = Path(output_dir)
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
            self._output_dir.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="pecan_yolo_seg_") as tmpdir:
                spec = export_videos_seg_dataset(self._video_entries, tmpdir)
                if not spec.class_names:
                    self.signals.error.emit("No classes found in the training dataset.")
                    return

                project_dir = Path(tmpdir) / "runs"
                run_name = "pecan-yolo-seg"
                project_dir.mkdir(parents=True, exist_ok=True)

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

                self.signals.log.emit(
                    f"Dataset: {len(self._video_entries)} video(s), "
                    f"classes: {', '.join(spec.class_names)}"
                )
                self.signals.log.emit(
                    f"Training on {spec.total_frames} frame(s) "
                    f"(one YOLO sample per video frame)."
                )
                for cls in spec.class_names:
                    labeled = spec.frames_per_class.get(cls, 0)
                    self.signals.log.emit(
                        f"  {cls}: labeled on {labeled}/{spec.total_frames} frame(s)"
                    )
                self.signals.log.emit("Starting YOLO training…")

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
                        self.signals.progress.emit(int(m.group(1)), int(m.group(2)))

                self._proc.wait()
                if self._stop_requested:
                    self.signals.error.emit("Training stopped by user.")
                    return
                if self._proc.returncode != 0:
                    self.signals.error.emit(
                        f"Training failed (exit code {self._proc.returncode})."
                    )
                    return

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

                dest = self._output_dir / "best.pt"
                shutil.copy2(best_pt, dest)
                self.signals.finished.emit(str(dest))

        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _InferWorker(QThread):
    def __init__(
        self,
        weights_path: str,
        sources: Sequence[Tuple[str, np.ndarray]],
        device: str,
    ):
        super().__init__()
        self.signals = _InferSignals()
        self._weights_path = weights_path
        self._sources = list(sources)
        self._device = device

    def run(self):
        try:
            from ultralytics import YOLO

            model = YOLO(self._weights_path)
            results: List[Tuple[str, np.ndarray]] = []
            total = len(self._sources)
            for idx, (name, frames) in enumerate(self._sources):
                self.signals.log.emit(f"Inference on {name}…")
                if frames.ndim == 3:
                    frames = frames[None, ...]
                label_stack = []
                frames_with_preds = 0
                for t in range(frames.shape[0]):
                    frame = frames[t]
                    if frame.dtype != np.uint8:
                        frame = np.clip(frame, 0, 255)
                        if float(np.max(frame)) <= 1.0:
                            frame = frame * 255.0
                        frame = frame.astype(np.uint8)
                    h, w = frame.shape[:2]
                    frame_imgsz = inference_imgsz(h, w, model)
                    res = model.predict(
                        to_yolo_predict_source(frame),
                        imgsz=frame_imgsz,
                        conf=0.1,
                        device=self._device,
                        retina_masks=True,
                        verbose=False,
                    )
                    label_map = yolo_result_to_label_map(res[0] if res else None)
                    if label_map is None:
                        label_map = np.zeros((h, w), dtype=np.uint8)
                    else:
                        frames_with_preds += 1
                    label_stack.append(label_map)

                out = (
                    label_stack[0]
                    if len(label_stack) == 1
                    else np.stack(label_stack, axis=0)
                )
                results.append((name, out))
                self.signals.log.emit(
                    f"  {frames_with_preds}/{frames.shape[0]} frame(s) with predictions"
                )
                self.signals.progress.emit(idx + 1, total)

            self.signals.finished.emit(results)

        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _TrainingVideoRow(QWidget):
    def __init__(self, video_path: str, *, on_remove, parent=None):
        super().__init__(parent)
        self.video_path = str(Path(video_path).resolve())
        self._on_remove = on_remove
        self.masks = discover_mask_files(self.video_path)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        name = QLabel(Path(self.video_path).name)
        name.setToolTip(self.video_path)
        layout.addWidget(name, 1)

        if self.masks:
            mask_text = ", ".join(sorted(self.masks.keys()))
            mask_lbl = QLabel(f"masks: {mask_text}")
            mask_lbl.setStyleSheet("color: #2ecc71;")
        else:
            mask_lbl = QLabel("no masks found")
            mask_lbl.setStyleSheet("color: #e67e22;")
        mask_lbl.setToolTip(
            "\n".join(str(p) for p in self.masks.values()) or "No mask files detected"
        )
        layout.addWidget(mask_lbl, 2)

        remove_btn = QPushButton()
        remove_btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        remove_btn.setFixedSize(24, 24)
        remove_btn.clicked.connect(lambda: self._on_remove(self.video_path))
        layout.addWidget(remove_btn)


class YoloSegWidget(QWidget):
    """Napari dock widget for YOLO segmentation inference and training."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._weights_path: str | None = None
        self._training_rows: List[_TrainingVideoRow] = []
        self._infer_file_paths: List[str] = []
        self._infer_layer_checkboxes: List[tuple[QCheckBox, Image]] = []

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

        layout.addWidget(self._build_inference_section())
        layout.addWidget(self._build_training_section())
        layout.addStretch(1)

        self._refresh_infer_layers()
        self._viewer.layers.events.inserted.connect(self._refresh_infer_layers)
        self._viewer.layers.events.removed.connect(self._refresh_infer_layers)

    def _help_label(self, text: str, tooltip: str = "") -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        lbl.setMinimumWidth(0)
        if tooltip:
            lbl.setToolTip(tooltip)
        return lbl

    def _build_inference_section(self) -> QGroupBox:
        group = QGroupBox("1 — Inference")
        lay = QVBoxLayout(group)

        weights_row = QHBoxLayout()
        weights_row.addWidget(QLabel("Model weights:"))
        self._weights_label = QLabel("(none loaded)")
        self._weights_label.setStyleSheet("color: #888;")
        self._weights_label.setWordWrap(True)
        weights_row.addWidget(self._weights_label, 1)
        load_btn = QPushButton("Browse…")
        load_btn.clicked.connect(self._load_weights)
        weights_row.addWidget(load_btn)
        lay.addLayout(weights_row)

        lay.addWidget(QLabel("Napari image layers (check to include):"))
        self._infer_layer_container = QVBoxLayout()
        layer_wrap = QWidget()
        layer_wrap.setLayout(self._infer_layer_container)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(layer_wrap)
        scroll.setMaximumHeight(100)
        lay.addWidget(scroll)

        lay.addWidget(QLabel("Files on disk:"))
        self._infer_file_list = QListWidget()
        self._infer_file_list.setMaximumHeight(80)
        lay.addWidget(self._infer_file_list)

        infer_files_row = QHBoxLayout()
        browse_infer_btn = QPushButton("Browse files…")
        browse_infer_btn.clicked.connect(self._browse_infer_files)
        infer_files_row.addWidget(browse_infer_btn)
        clear_infer_btn = QPushButton("Clear files")
        clear_infer_btn.clicked.connect(self._clear_infer_files)
        infer_files_row.addWidget(clear_infer_btn)
        lay.addLayout(infer_files_row)

        self._infer_btn = QPushButton("Run inference")
        self._infer_btn.clicked.connect(self._run_inference)
        lay.addWidget(self._infer_btn)

        self._infer_progress = QProgressBar()
        self._infer_progress.setRange(0, 100)
        self._infer_progress.setValue(0)
        lay.addWidget(self._infer_progress)

        self._infer_log = QTextEdit()
        self._infer_log.setReadOnly(True)
        self._infer_log.setMaximumHeight(80)
        lay.addWidget(self._infer_log)

        return group

    def _build_training_section(self) -> QGroupBox:
        group = QGroupBox("2 — Training")
        lay = QVBoxLayout(group)

        lay.addWidget(
            self._help_label(
                "Add training videos (one YOLO sample per frame). "
                "Masks are auto-detected next to each video.",
                tooltip=(
                    "Each video frame is exported as a separate training image.\n\n"
                    "Mask volumes (TIFF/NPY) must have the same frame count as the video.\n\n"
                    "A class may be empty on some frames—for example, Crack is only "
                    "labeled when visible on camera, while Pecan is expected on every frame.\n\n"
                    "Mask files live in the same folder as the video and are named "
                    "'<video> - … - <Class>' (class = last word of the filename)."
                ),
            )
        )

        self._train_video_list = QListWidget()
        self._train_video_list.setMaximumHeight(140)
        lay.addWidget(self._train_video_list)

        train_videos_row = QHBoxLayout()
        add_train_btn = QPushButton("Browse videos…")
        add_train_btn.clicked.connect(self._browse_training_videos)
        train_videos_row.addWidget(add_train_btn)
        clear_train_btn = QPushButton("Clear list")
        clear_train_btn.clicked.connect(self._clear_training_videos)
        train_videos_row.addWidget(clear_train_btn)
        lay.addLayout(train_videos_row)

        self._dataset_summary = QLabel("Classes: (none)")
        self._dataset_summary.setWordWrap(True)
        self._dataset_summary.setStyleSheet("color: #888;")
        self._dataset_summary.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self._dataset_summary.setMinimumWidth(0)
        lay.addWidget(self._dataset_summary)

        self._epochs_spin = self._spin_row(lay, "Epochs:", 50, 1, 500, 1)
        self._batch_spin = self._spin_row(lay, "Batch size:", 4, 1, 64, 1)
        self._lr_dspin = self._dspin_row(lay, "Learning rate:", 1e-3, 1e-5, 1.0, 1e-4)

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
        lay.addLayout(dev_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save weights to:"))
        self._output_dir_label = QLabel(str(Path.cwd() / "yolo_seg_runs"))
        self._output_dir_label.setWordWrap(True)
        out_row.addWidget(self._output_dir_label, 1)
        out_dir_btn = QPushButton("Browse…")
        out_dir_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(out_dir_btn)
        lay.addLayout(out_row)
        self._output_dir = Path.cwd() / "yolo_seg_runs"

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train YOLO")
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)
        lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        lay.addWidget(self._progress)

        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(120)
        lay.addWidget(self._log)

        return group

    def _spin_row(self, parent, label, default, lo, hi, step):
        row = QHBoxLayout()
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        row.addWidget(QLabel(label))
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _dspin_row(self, parent, label, default, lo, hi, step):
        row = QHBoxLayout()
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        row.addWidget(QLabel(label))
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _refresh_infer_layers(self, _evt=None):
        for cb, _ in self._infer_layer_checkboxes:
            cb.setParent(None)
        self._infer_layer_checkboxes.clear()
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                cb = QCheckBox(layer.name)
                cb.setChecked(False)
                self._infer_layer_container.addWidget(cb)
                self._infer_layer_checkboxes.append((cb, layer))

    def _load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO weights", "", "YOLO weights (*.pt *.pth)"
        )
        if not path:
            return
        self._weights_path = path
        self._weights_label.setText(path)
        self._weights_label.setStyleSheet("")

    def _browse_infer_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select media for inference", "", _MEDIA_FILTER
        )
        for p in paths:
            resolved = str(Path(p).resolve())
            if resolved not in self._infer_file_paths:
                self._infer_file_paths.append(resolved)
                self._infer_file_list.addItem(Path(resolved).name)

    def _clear_infer_files(self):
        self._infer_file_paths.clear()
        self._infer_file_list.clear()

    def _browse_training_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Add training videos",
            "",
            "Videos ("
            + " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
            + ")",
        )
        for p in paths:
            self._add_training_video(p)

    def _add_training_video(self, path: str):
        resolved = str(Path(path).resolve())
        if any(r.video_path == resolved for r in self._training_rows):
            return
        row = _TrainingVideoRow(resolved, on_remove=self._remove_training_video)
        self._training_rows.append(row)
        item = QListWidgetItem()
        item.setSizeHint(row.sizeHint())
        self._train_video_list.addItem(item)
        self._train_video_list.setItemWidget(item, row)
        self._update_dataset_summary()

    def _remove_training_video(self, path: str):
        for i, row in enumerate(self._training_rows):
            if row.video_path == path:
                self._training_rows.pop(i)
                self._train_video_list.takeItem(i)
                break
        self._update_dataset_summary()

    def _clear_training_videos(self):
        self._training_rows.clear()
        self._train_video_list.clear()
        self._update_dataset_summary()

    def _update_dataset_summary(self):
        entries = [
            (
                row.video_path,
                {cls: str(path) for cls, path in row.masks.items()},
            )
            for row in self._training_rows
            if row.masks
        ]
        if not entries:
            if not self._training_rows:
                self._dataset_summary.setText("Classes: (none)")
            else:
                self._dataset_summary.setText(
                    f"{len(self._training_rows)} video(s) added, no masks detected yet."
                )
            return
        try:
            summary = summarize_training_dataset(entries)
            self._dataset_summary.setText(format_dataset_summary(summary))
        except Exception as exc:
            self._dataset_summary.setText(f"Dataset error: {exc}")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Training output directory")
        if not path:
            return
        self._output_dir = Path(path)
        self._output_dir_label.setText(str(self._output_dir))

    def _device_value(self) -> str:
        value = self._device_combo.currentData()
        return str(value if value is not None else self._device_combo.currentText())

    def _layer_rgb_frames(self, layer: Image) -> np.ndarray:
        data = layer.data
        if hasattr(data, "shape") and len(data.shape) == 4:
            frames = np.stack([np.asarray(data[t]) for t in range(data.shape[0])], axis=0)
        else:
            arr = np.asarray(data)
            if arr.ndim == 3 and arr.shape[-1] in (3, 4):
                frames = arr[None, ...]
            elif arr.ndim == 4:
                frames = arr
            else:
                raise ValueError(f"Unsupported layer shape {arr.shape}")
        if frames.shape[-1] == 4:
            frames = frames[..., :3]
        return frames

    def _load_infer_source_frames(self, path: str) -> Tuple[str, np.ndarray]:
        p = Path(path)
        suffix = p.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            return p.stem, load_video_rgb_frames(p)
        if suffix in IMAGE_EXTENSIONS:
            return p.stem, load_image_rgb(p)[None, ...]
        raise ValueError(f"Unsupported file type: {p}")

    def _collect_inference_sources(self) -> List[Tuple[str, np.ndarray]]:
        sources: List[Tuple[str, np.ndarray]] = []
        for cb, layer in self._infer_layer_checkboxes:
            if cb.isChecked() and layer in self._viewer.layers:
                sources.append((layer.name, self._layer_rgb_frames(layer)))
        for path in self._infer_file_paths:
            name, frames = self._load_infer_source_frames(path)
            sources.append((name, frames))
        return sources

    def _run_inference(self):
        from napari.utils.notifications import show_warning

        if self._weights_path is None:
            show_warning("Browse and load a trained model first.")
            return
        sources = self._collect_inference_sources()
        if not sources:
            show_warning("Select at least one napari layer or file for inference.")
            return

        self._infer_btn.setEnabled(False)
        self._infer_progress.setValue(0)
        self._infer_log.clear()

        self._infer_worker = _InferWorker(
            self._weights_path, sources, self._device_value()
        )
        self._infer_worker.signals.log.connect(self._infer_log.append)
        self._infer_worker.signals.progress.connect(self._on_infer_progress)
        self._infer_worker.signals.finished.connect(self._on_inference_finished)
        self._infer_worker.signals.error.connect(self._on_inference_error)
        self._infer_worker.start()

    def _on_infer_progress(self, cur: int, tot: int):
        if tot > 0:
            self._infer_progress.setValue(int(100 * cur / tot))

    def _on_inference_finished(self, results: list):
        from napari.utils.notifications import show_info, show_warning

        self._infer_btn.setEnabled(True)
        self._infer_progress.setValue(100)
        any_predictions = False
        for name, label_data in results:
            layer_name = f"{name} - YOLO seg"
            if np.any(label_data):
                any_predictions = True
            try:
                existing = self._viewer.layers[layer_name]
                if tuple(existing.data.shape) != tuple(label_data.shape):
                    self._viewer.layers.remove(existing)
                    self._viewer.add_labels(label_data, name=layer_name, opacity=0.5)
                else:
                    existing.data = label_data
                    existing.refresh()
            except Exception:
                self._viewer.add_labels(label_data, name=layer_name, opacity=0.5)
        if any_predictions:
            show_info(f"Inference complete on {len(results)} sample(s).")
        else:
            show_warning(
                "Inference finished but no masks were predicted. "
                "Check the inference log for per-frame counts."
            )

    def _on_inference_error(self, msg: str):
        from napari.utils.notifications import show_error

        self._infer_btn.setEnabled(True)
        self._infer_log.append(f"ERROR: {msg}")
        show_error(f"YOLO inference error:\n{msg}")

    def _start_training(self):
        from napari.utils.notifications import show_warning

        entries: List[Tuple[str, Dict[str, str]]] = []
        for row in self._training_rows:
            if not row.masks:
                continue
            entries.append(
                (
                    row.video_path,
                    {cls: str(path) for cls, path in row.masks.items()},
                )
            )
        if not entries:
            show_warning(
                "Add training videos with detected mask files in the same folder."
            )
            return

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)
        self._log.clear()

        self._worker = _TrainWorker(
            video_entries=entries,
            epochs=self._epochs_spin.value(),
            batch=self._batch_spin.value(),
            lr=self._lr_dspin.value(),
            device=self._device_value(),
            output_dir=str(self._output_dir),
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
        self._weights_path = path
        self._weights_label.setText(path)
        self._weights_label.setStyleSheet("")
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
