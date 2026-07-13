"""Segmentation widget: YOLO, flat U-Net, or cascaded U-Net inference and training."""

from __future__ import annotations

import json
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
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtGui import QColor, QPainter
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QScrollArea,
    QSizePolicy,
    QStyle,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from napari_pecan_py._reader import VIDEO_EXTENSIONS

from ..unet_seg.model import (
    BACKEND_UNET,
    UnetTrainConfig,
    format_unet_train_summary,
    guess_unet_save_suffix,
    run_unet_inference_on_frames,
    summarize_unet_frame_usage,
    train_unet_segmenter,
)
from ..cascade_seg.hierarchy import format_hierarchy_chain
from ..cascade_seg.model import (
    ARCH_UNET,
    ARCH_UNETPP,
    BACKEND_CASCADE,
    CascadeTrainConfig,
    detect_seg_checkpoint_backend,
    format_cascade_train_summary,
    guess_cascade_save_suffix,
    run_cascade_inference_on_frames,
    summarize_cascade_frame_usage,
    train_cascade_segmenter,
)
from ..pipeline_recorder.state import upsert_pipeline_step

from .model import (
    IMAGE_EXTENSIONS,
    ClassLabelMapping,
    binary_mask_for_label_ids,
    default_label_ids_for_discovered_class,
    discover_mask_files,
    discover_videos_in_directory,
    format_label_ids_text,
    load_mask_volume,
    parse_label_ids_text,
    video_path_for_saved_mask,
    export_videos_seg_dataset,
    format_dataset_summary,
    guess_save_suffix_from_weights,
    image_volume_to_rgb_frames,
    infer_labels_layer_name,
    infer_mask_output_path,
    load_image_rgb,
    load_video_rgb_frames,
    run_yolo_seg_inference_on_frames,
    save_mask_volume,
    summarize_training_dataset,
    resolve_yolo_device,
    YoloTrainAugmentConfig,
    format_augment_summary,
)

BACKEND_YOLO = "yolo"
_PREVIEW_IMAGE_LAYER = "_seg train preview image"
_PREVIEW_MASK_LAYER = "_seg train preview mask"

_ULTRA_AVAILABLE = False
try:
    import ultralytics  # type: ignore[import]

    _ULTRA_AVAILABLE = True
except Exception:
    pass

_SMP_AVAILABLE = False
try:
    import segmentation_models_pytorch  # type: ignore[import]  # noqa: F401

    _SMP_AVAILABLE = True
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
        val_fraction: float,
        split_by: str,
        augment: YoloTrainAugmentConfig,
        *,
        init_weights_path: str | None = None,
        apply_saved_range: bool = True,
        label_ids_by_class: Dict[str, set[int] | None] | None = None,
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self._video_entries = list(video_entries)
        self._epochs = epochs
        self._batch = batch
        self._lr = lr
        self._device = device
        self._output_dir = Path(output_dir)
        self._val_fraction = val_fraction
        self._split_by = split_by
        self._augment = augment
        self._init_weights_path = init_weights_path
        self._apply_saved_range = apply_saved_range
        self._label_ids_by_class = label_ids_by_class
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
                spec = export_videos_seg_dataset(
                    self._video_entries,
                    tmpdir,
                    val_fraction=self._val_fraction,
                    split_by=self._split_by,
                    apply_saved_range=self._apply_saved_range,
                    label_ids_by_class=self._label_ids_by_class,
                )
                if not spec.class_names:
                    self.signals.error.emit("No classes found in the training dataset.")
                    return

                project_dir = Path(tmpdir) / "runs"
                run_name = "pecan-yolo-seg"
                project_dir.mkdir(parents=True, exist_ok=True)

                code = r"""
import json
import sys
from ultralytics import YOLO

aug = json.loads(sys.argv[8])
weights = sys.argv[9] if len(sys.argv) > 9 and sys.argv[9] else 'yolov8n-seg.pt'
model = YOLO(weights)
model.train(
    data=sys.argv[1],
    epochs=int(sys.argv[2]),
    batch=int(sys.argv[3]),
    lr0=float(sys.argv[4]),
    device=sys.argv[5],
    project=sys.argv[6],
    name=sys.argv[7],
    verbose=True,
    **aug,
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
                    json.dumps(self._augment.to_train_kwargs()),
                    str(self._init_weights_path or ""),
                ]

                self.signals.log.emit(
                    f"Dataset: {len(self._video_entries)} video(s), "
                    f"classes: {', '.join(spec.class_names)}"
                )
                if spec.val_frames > 0:
                    self.signals.log.emit(
                        f"Split ({spec.split_by}): {spec.train_frames} train frame(s), "
                        f"{spec.val_frames} val frame(s)"
                    )
                    if self._split_by == "video" and spec.split_by == "frame":
                        self.signals.log.emit(
                            "Note: only one video available — used random frame split."
                        )
                else:
                    self.signals.log.emit(
                        "No validation split (all frames used for training)."
                    )
                self.signals.log.emit(
                    f"Training on {spec.total_frames} exported frame(s) "
                    f"(one YOLO sample per video frame)."
                )
                for cls in spec.class_names:
                    labeled = spec.frames_per_class.get(cls, 0)
                    self.signals.log.emit(
                        f"  {cls}: labeled on {labeled}/{spec.total_frames} frame(s)"
                    )
                self.signals.log.emit(format_augment_summary(self._augment))
                if self._init_weights_path:
                    self.signals.log.emit(
                        f"Fine-tuning YOLO from {self._init_weights_path}"
                    )
                else:
                    self.signals.log.emit("Starting YOLO training from yolov8n-seg.pt…")

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


class _UnetTrainWorker(QThread):
    def __init__(
        self,
        video_entries: Sequence[Tuple[str, Dict[str, str]]],
        selected_classes: Sequence[str],
        config: UnetTrainConfig,
        device: str,
        output_dir: str,
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self._video_entries = list(video_entries)
        self._selected_classes = list(selected_classes)
        self._config = config
        self._device = device
        self._output_dir = Path(output_dir)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self.signals.log.emit(format_unet_train_summary(self._config))
            self.signals.log.emit("Starting flat U-Net training…")

            def _log(msg: str) -> None:
                self.signals.log.emit(msg)

            def _progress(epoch: int, total: int) -> None:
                self.signals.progress.emit(epoch, total)

            path = train_unet_segmenter(
                self._video_entries,
                self._output_dir,
                self._device,
                self._config,
                selected_classes=self._selected_classes,
                log_callback=_log,
                progress_callback=_progress,
                cancel_callback=lambda: self._stop_requested,
            )
            if self._stop_requested:
                self.signals.error.emit("Training stopped by user.")
                return
            self.signals.finished.emit(path)
        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _CascadeTrainWorker(QThread):
    def __init__(
        self,
        video_entries: Sequence[Tuple[str, Dict[str, str]]],
        selected_classes: Sequence[str],
        config: CascadeTrainConfig,
        device: str,
        output_dir: str,
    ):
        super().__init__()
        self.signals = _TrainSignals()
        self._video_entries = list(video_entries)
        self._selected_classes = list(selected_classes)
        self._config = config
        self._device = device
        self._output_dir = Path(output_dir)
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            self._output_dir.mkdir(parents=True, exist_ok=True)
            self.signals.log.emit(format_cascade_train_summary(self._config))
            self.signals.log.emit(f"Hierarchy: {format_hierarchy_chain()}")
            self.signals.log.emit("Starting cascade training…")

            def _log(msg: str) -> None:
                self.signals.log.emit(msg)

            def _progress(epoch: int, total: int) -> None:
                self.signals.progress.emit(epoch, total)

            path = train_cascade_segmenter(
                self._video_entries,
                self._output_dir,
                self._device,
                self._config,
                selected_classes=self._selected_classes,
                log_callback=_log,
                progress_callback=_progress,
                cancel_callback=lambda: self._stop_requested,
            )
            if self._stop_requested:
                self.signals.error.emit("Training stopped by user.")
                return
            self.signals.finished.emit(path)
        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _InferWorker(QThread):
    def __init__(
        self,
        weights_path: str,
        sources: Sequence[Tuple[str, np.ndarray, str | None]],
        device: str,
        backend: str,
        *,
        save_masks: bool = False,
        save_suffix: str = "",
        save_fmt: str = "tiff",
    ):
        super().__init__()
        self.signals = _InferSignals()
        self._weights_path = weights_path
        self._sources = list(sources)
        self._device = device
        self._backend = backend
        self._save_masks = save_masks
        self._save_suffix = save_suffix
        self._save_fmt = save_fmt
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        try:
            results: List[Tuple[str, np.ndarray, str | None]] = []
            total = len(self._sources)
            for idx, (name, frames, source_path) in enumerate(self._sources):
                if self._stop_requested:
                    break
                self.signals.log.emit(f"Inference on {name}…")
                rgb = image_volume_to_rgb_frames(frames)
                infer_fn = {
                    BACKEND_CASCADE: run_cascade_inference_on_frames,
                    BACKEND_UNET: run_unet_inference_on_frames,
                }.get(self._backend, run_yolo_seg_inference_on_frames)
                out = infer_fn(
                    self._weights_path,
                    rgb,
                    self._device,
                    cancel_callback=lambda: self._stop_requested,
                )
                if out.ndim == 3:
                    frames_with_preds = sum(int(np.any(out[t])) for t in range(out.shape[0]))
                else:
                    frames_with_preds = int(np.any(out))

                saved_path: str | None = None
                if self._save_masks:
                    if not source_path:
                        self.signals.log.emit(
                            f"  Skipped save for {name}: no source file path."
                        )
                    else:
                        out_path = infer_mask_output_path(
                            source_path, self._save_suffix, self._save_fmt
                        )
                        save_mask_volume(out, out_path, self._save_fmt)
                        saved_path = str(out_path)
                        self.signals.log.emit(f"  Saved mask -> {saved_path}")

                results.append((name, out, saved_path))
                frame_count = rgb.shape[0]
                self.signals.log.emit(
                    f"  {frames_with_preds}/{frame_count} frame(s) with predictions"
                )
                self.signals.progress.emit(idx + 1, total)
                if self._stop_requested:
                    break

            if self._stop_requested:
                self.signals.log.emit("Inference stopped by user.")
            self.signals.finished.emit(results)

        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _ProgressButton(QPushButton):
    """Push button that can show an inline progress fill over its label."""

    def __init__(self, text: str = "", parent=None):
        super().__init__(text, parent)
        self._progress = 0
        self._base_text = text

    def setText(self, text: str) -> None:
        self._base_text = text
        if self._progress <= 0:
            super().setText(text)
        else:
            super().setText(self._label_with_progress())

    def set_progress(self, value: int) -> None:
        self._progress = int(max(0, min(100, value)))
        if self._progress > 0:
            super().setText(self._label_with_progress())
        self.update()

    def reset_progress(self) -> None:
        self._progress = 0
        super().setText(self._base_text)
        self.update()

    def _label_with_progress(self) -> str:
        return f"{self._base_text}  {self._progress}%"

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._progress <= 0:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        inset = 3
        inner_w = max(0, self.width() - 2 * inset)
        inner_h = max(0, self.height() - 2 * inset)
        fill_w = max(1, int(inner_w * self._progress / 100))
        painter.fillRect(inset, inset, fill_w, inner_h, QColor(56, 152, 255, 95))
        painter.fillRect(inset + fill_w - 2, inset, 2, inner_h, QColor(130, 210, 255, 200))
        painter.end()


class _TrainingVideoRow(QWidget):
    def __init__(self, video_path: str, *, on_remove, parent=None):
        super().__init__(parent)
        self.video_path = str(Path(video_path).resolve())
        self.masks = discover_mask_files(self.video_path)
        self._on_remove = on_remove

        lay = QHBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        self._name_label = QLabel(Path(self.video_path).name)
        self._name_label.setToolTip(self.video_path)
        lay.addWidget(self._name_label, stretch=1)
        self._mask_label = QLabel()
        self._mask_label.setStyleSheet("color: #888; font-size: 11px;")
        lay.addWidget(self._mask_label)
        remove_btn = QToolButton()
        remove_btn.setText("×")
        remove_btn.clicked.connect(lambda: self._on_remove(self.video_path))
        lay.addWidget(remove_btn)
        self._update_mask_label()

    def refresh_masks(self) -> bool:
        """Re-scan the video folder for mask files. Returns True if masks changed."""
        prev = set(self.masks.keys())
        self.masks = discover_mask_files(self.video_path)
        changed = set(self.masks.keys()) != prev
        self._update_mask_label()
        return changed

    def _update_mask_label(self) -> None:
        if self.masks:
            classes = sorted(self.masks.keys())
            combined = len(set(map(str, self.masks.values()))) < len(self.masks)
            mask_text = ", ".join(classes)
            if combined:
                mask_text += " (1 combined file)"
            self._mask_label.setText(mask_text)
            self._mask_label.setStyleSheet("color: #2a7; font-size: 11px;")
        else:
            self._mask_label.setText("no masks")
            self._mask_label.setStyleSheet("color: #c55; font-size: 11px;")


class _ClassMappingRow(QWidget):
    """One editable training class: name + pixel label IDs + mask source."""

    changed = Signal()
    preview_requested = Signal(object)
    remove_requested = Signal(object)

    def __init__(
        self,
        *,
        name: str,
        source_key: str,
        source_path: str,
        ids_text: str = "*",
        enabled: bool = True,
        source_options: Sequence[Tuple[str, str]] | None = None,
        parent=None,
    ):
        super().__init__(parent)
        self.source_key = source_key
        self.source_path = str(source_path)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 2, 0, 2)

        self.enable_cb = QCheckBox()
        self.enable_cb.setChecked(enabled)
        self.enable_cb.setToolTip("Include this class in training")
        self.enable_cb.toggled.connect(self.changed.emit)
        lay.addWidget(self.enable_cb)

        self.name_edit = QLineEdit(name)
        self.name_edit.setPlaceholderText("Class name")
        self.name_edit.setMinimumWidth(70)
        self.name_edit.editingFinished.connect(self._on_edited)
        self.name_edit.textEdited.connect(lambda _t: self.changed.emit())
        self.name_edit.focusInEvent = self._make_focus_handler(self.name_edit)  # type: ignore[method-assign]
        lay.addWidget(self.name_edit, stretch=1)

        lay.addWidget(QLabel(":"))

        self.ids_edit = QLineEdit(ids_text)
        self.ids_edit.setPlaceholderText("1 or *")
        self.ids_edit.setMaximumWidth(90)
        self.ids_edit.setToolTip(
            "Pixel label ID(s) for this class.\n"
            "Examples: 1   |   1, 2   |   * or [*] (any positive label)"
        )
        self.ids_edit.editingFinished.connect(self._on_edited)
        self.ids_edit.textEdited.connect(lambda _t: self.changed.emit())
        self.ids_edit.focusInEvent = self._make_focus_handler(self.ids_edit)  # type: ignore[method-assign]
        lay.addWidget(self.ids_edit)

        self.source_combo = QComboBox()
        self.source_combo.setMinimumWidth(120)
        self.source_combo.setToolTip("Mask file used as the pixel source for this class")
        self._set_source_options(source_options or [(source_key, source_path)])
        self.source_combo.currentIndexChanged.connect(self._on_source_changed)
        lay.addWidget(self.source_combo, stretch=1)

        preview_btn = QToolButton()
        preview_btn.setText("👁")
        preview_btn.setToolTip("Preview this label mapping on a sample frame in napari")
        preview_btn.clicked.connect(lambda: self.preview_requested.emit(self))
        lay.addWidget(preview_btn)

        remove_btn = QToolButton()
        remove_btn.setText("×")
        remove_btn.setToolTip("Remove this class mapping")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        lay.addWidget(remove_btn)

    def _make_focus_handler(self, edit: QLineEdit):
        def _handler(event) -> None:
            QLineEdit.focusInEvent(edit, event)
            self.preview_requested.emit(self)

        return _handler

    def _set_source_options(self, options: Sequence[Tuple[str, str]]) -> None:
        self.source_combo.blockSignals(True)
        self.source_combo.clear()
        selected = 0
        for i, (key, path) in enumerate(options):
            label = f"{key}  ({Path(path).name})"
            self.source_combo.addItem(label, (key, str(path)))
            if key == self.source_key or str(path) == self.source_path:
                selected = i
        if self.source_combo.count() == 0:
            self.source_combo.addItem(
                f"{self.source_key}  ({Path(self.source_path).name})",
                (self.source_key, self.source_path),
            )
        self.source_combo.setCurrentIndex(selected)
        self.source_combo.blockSignals(False)
        self._on_source_changed()

    def update_source_options(self, options: Sequence[Tuple[str, str]]) -> None:
        prev = self.source_combo.currentData()
        self._set_source_options(options)
        if prev is not None:
            for i in range(self.source_combo.count()):
                if self.source_combo.itemData(i) == prev:
                    self.source_combo.setCurrentIndex(i)
                    break

    def _on_source_changed(self) -> None:
        data = self.source_combo.currentData()
        if data:
            self.source_key, self.source_path = data
        self.changed.emit()

    def _on_edited(self) -> None:
        self.changed.emit()
        self.preview_requested.emit(self)

    def to_mapping(self) -> ClassLabelMapping | None:
        name = self.name_edit.text().strip()
        if not name:
            return None
        try:
            ids = parse_label_ids_text(self.ids_edit.text())
        except Exception:
            return None
        return ClassLabelMapping(
            name=name,
            source_key=self.source_key,
            source_path=self.source_path,
            label_ids=ids,
            enabled=self.enable_cb.isChecked(),
        )


class _CollapsibleSection(QWidget):
    """Accordion section: header toggles visibility of all content below it."""

    def __init__(self, title: str, *, expanded: bool = True, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 4)
        outer.setSpacing(2)

        self._toggle = QToolButton()
        self._toggle.setText(title)
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self._toggle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._toggle.setToolTip("Click to expand or collapse this section")
        self._toggle.setStyleSheet("QToolButton { text-align: left; font-weight: bold; }")
        self._toggle.toggled.connect(self._on_toggle)
        outer.addWidget(self._toggle)

        self._content = QWidget()
        self._content_layout = QVBoxLayout(self._content)
        self._content_layout.setContentsMargins(8, 0, 0, 0)
        self._content_layout.setSpacing(4)
        self._content.setVisible(expanded)
        outer.addWidget(self._content)

        self._set_arrow(expanded)

    def _set_arrow(self, expanded: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _on_toggle(self, expanded: bool) -> None:
        self._content.setVisible(expanded)
        self._set_arrow(expanded)

    def content_layout(self) -> QVBoxLayout:
        return self._content_layout


class SegmentationWidget(QWidget):
    """Napari dock widget for YOLO or cascaded U-Net segmentation."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._weights_path: str | None = None
        self._weights_backend: str | None = None
        self._training_rows: List[_TrainingVideoRow] = []
        self._class_mapping_rows: List[_ClassMappingRow] = []
        self._infer_file_paths: List[str] = []
        self._infer_dir_paths: List[str] = []
        self._infer_layer_checkboxes: List[tuple[QCheckBox, Image]] = []
        self._infer_save_masks = False
        self._weights_fallback_counter = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _ULTRA_AVAILABLE and not _SMP_AVAILABLE:
            err = QVBoxLayout()
            err.setContentsMargins(4, 4, 4, 4)
            err.addWidget(
                QLabel(
                    "No segmentation backend is available.\n"
                    "Install YOLO: pip install ultralytics\n"
                    "Install cascade: pip install segmentation-models-pytorch\n"
                    "then restart napari."
                )
            )
            layout.addLayout(err)
            return

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QScrollArea.NoFrame)

        body = QWidget()
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(4, 4, 4, 4)
        body_layout.setSpacing(6)
        body_layout.addWidget(self._build_backend_section())
        body_layout.addWidget(self._build_inference_section())
        body_layout.addWidget(self._build_training_section())
        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll)

        self._refresh_infer_layers()
        self._viewer.layers.events.inserted.connect(self._refresh_infer_layers)
        self._viewer.layers.events.removed.connect(self._refresh_infer_layers)
        self._on_backend_changed()

    def _help_label(self, text: str, tooltip: str = "") -> QLabel:
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        lbl.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum)
        lbl.setMinimumWidth(0)
        if tooltip:
            lbl.setToolTip(tooltip)
        return lbl

    def _build_backend_section(self) -> QWidget:
        wrap = QWidget()
        lay = QVBoxLayout(wrap)
        lay.setContentsMargins(0, 0, 0, 0)

        row = QHBoxLayout()
        row.addWidget(QLabel("Model backend:"))
        self._backend_combo = QComboBox()
        if _ULTRA_AVAILABLE:
            self._backend_combo.addItem("YOLO (Ultralytics)", BACKEND_YOLO)
        if _SMP_AVAILABLE:
            self._backend_combo.addItem("U-Net / U-Net++", BACKEND_UNET)
            self._backend_combo.addItem("Cascade U-Net (hierarchical)", BACKEND_CASCADE)
        self._backend_combo.currentIndexChanged.connect(self._on_backend_changed)
        row.addWidget(self._backend_combo, 1)
        lay.addLayout(row)

        self._backend_help = self._help_label(
            "U-Net: one model, one output channel per class. "
            "Cascade: Pecan first, then Crack and Kernel as siblings inside pecan.",
            tooltip=(
                "YOLO: instance segmentation via Ultralytics.\n\n"
                "U-Net / U-Net++: single multi-head model (simpler, often enough).\n\n"
                "Cascade: separate U-Net per class with parent-mask conditioning."
            ),
        )
        lay.addWidget(self._backend_help)
        return wrap

    def _selected_backend(self) -> str:
        if not hasattr(self, "_backend_combo"):
            if _ULTRA_AVAILABLE:
                return BACKEND_YOLO
            if _SMP_AVAILABLE:
                return BACKEND_UNET
            return BACKEND_YOLO
        value = self._backend_combo.currentData()
        return str(value if value is not None else BACKEND_YOLO)

    def _is_smp_backend(self, backend: str | None = None) -> bool:
        backend = backend or self._selected_backend()
        return backend in {BACKEND_CASCADE, BACKEND_UNET}

    def _default_output_dir_name(self, backend: str | None = None) -> str:
        backend = backend or self._selected_backend()
        if backend == BACKEND_YOLO:
            return "yolo_seg_runs"
        if backend == BACKEND_UNET:
            return "unet_seg_runs"
        return "cascade_seg_runs"

    def _on_backend_changed(self, _index: int | None = None) -> None:
        backend = self._selected_backend()
        is_yolo = backend == BACKEND_YOLO
        is_cascade = backend == BACKEND_CASCADE
        is_unet = backend == BACKEND_UNET
        is_smp = self._is_smp_backend(backend)

        if hasattr(self, "_yolo_augment_section"):
            self._yolo_augment_section.setVisible(is_yolo)
        if hasattr(self, "_cascade_options"):
            self._cascade_options.setVisible(is_smp)
        if hasattr(self, "_cascade_hierarchy_help"):
            self._cascade_hierarchy_help.setVisible(is_cascade)
        if hasattr(self, "_train_btn"):
            if is_yolo:
                train_label = "Train YOLO"
            elif is_unet:
                train_label = "Train U-Net"
            else:
                train_label = "Train cascade model"
            self._train_btn.setText(train_label)
        if hasattr(self, "_output_dir_label") and hasattr(self, "_output_dir"):
            default = self._default_output_dir_name(backend)
            if self._output_dir.name in {
                "yolo_seg_runs",
                "unet_seg_runs",
                "cascade_seg_runs",
            }:
                self._output_dir = Path.cwd() / default
                self._output_dir_label.setText(str(self._output_dir))

        self._update_fine_tune_enabled()

        if (
            self._weights_path
            and self._weights_backend
            and self._weights_backend != backend
        ):
            self._weights_path = None
            self._weights_backend = None
            if hasattr(self, "_weights_label"):
                self._weights_label.setText("(none loaded — backend changed)")
                self._weights_label.setStyleSheet("color: #888;")

    def _build_inference_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("1 — Inference", expanded=True)
        lay = section.content_layout()

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

        lay.addWidget(QLabel("Videos on disk (files or folders):"))
        self._infer_file_list = QListWidget()
        self._infer_file_list.setMaximumHeight(100)
        lay.addWidget(self._infer_file_list)

        infer_files_row = QHBoxLayout()
        browse_infer_btn = QPushButton("Browse files…")
        browse_infer_btn.clicked.connect(self._browse_infer_files)
        infer_files_row.addWidget(browse_infer_btn)
        browse_dirs_btn = QPushButton("Browse directories…")
        browse_dirs_btn.setToolTip(
            "Select one or more folders. All videos in each folder and its "
            "subfolders are included when running inference."
        )
        browse_dirs_btn.clicked.connect(self._browse_infer_directories)
        infer_files_row.addWidget(browse_dirs_btn)
        clear_infer_btn = QPushButton("Clear list")
        clear_infer_btn.clicked.connect(self._clear_infer_files)
        infer_files_row.addWidget(clear_infer_btn)
        lay.addLayout(infer_files_row)

        self._infer_skip_layers_checkbox = QCheckBox("Save only (don't add masks to napari)")
        self._infer_skip_layers_checkbox.setToolTip(
            "When saving masks to disk, write a mask file next to each source "
            "video without creating or updating napari Labels layers."
        )
        self._infer_skip_layers_checkbox.setEnabled(False)
        lay.addWidget(self._infer_skip_layers_checkbox)

        save_opts = QHBoxLayout()
        save_opts.addWidget(QLabel("Save suffix:"))
        self._save_suffix_edit = QLineEdit("")
        self._save_suffix_edit.setPlaceholderText(" e.g.  - Crack")
        self._save_suffix_edit.setToolTip(
            "Appended to source names for saved masks and napari Labels layers.\n"
            "Auto-filled when loading weights named like 'model - [Crack].pt' or "
            "'model - [Pecan, Kernel, Crack].pt'."
        )
        save_opts.addWidget(self._save_suffix_edit, 1)
        save_opts.addWidget(QLabel("Format:"))
        self._save_fmt_combo = QComboBox()
        self._save_fmt_combo.addItem("TIFF", "tiff")
        self._save_fmt_combo.addItem("PNG", "png")
        self._save_fmt_combo.addItem("NPY", "npy")
        save_opts.addWidget(self._save_fmt_combo)
        lay.addLayout(save_opts)

        infer_dev_row = QHBoxLayout()
        infer_dev_row.addWidget(QLabel("Device:"))
        self._infer_device_combo = QComboBox()
        self._populate_device_combo(self._infer_device_combo)
        infer_dev_row.addWidget(self._infer_device_combo)
        lay.addLayout(infer_dev_row)

        infer_run_row = QHBoxLayout()
        infer_run_row.setSpacing(0)

        self._infer_main_btn = _ProgressButton("Run inference")
        self._infer_main_btn.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._infer_main_btn.clicked.connect(self._run_selected_inference)
        self._infer_main_btn.setStyleSheet(
            "QPushButton { border-top-right-radius: 0; border-bottom-right-radius: 0; }"
        )
        infer_run_row.addWidget(self._infer_main_btn, 1)

        self._infer_menu_btn = QToolButton()
        self._infer_menu_btn.setPopupMode(QToolButton.InstantPopup)
        self._infer_menu_btn.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._infer_menu_btn.setFixedWidth(22)
        self._infer_menu_btn.setToolTip("Choose inference action")
        self._infer_menu_btn.setStyleSheet(
            "QToolButton { border-top-left-radius: 0; border-bottom-left-radius: 0; "
            "margin-left: -1px; }"
        )

        infer_menu = QMenu(self)
        infer_menu.addAction(
            "Run inference",
            lambda: self._set_infer_mode(save_masks=False),
        )
        infer_menu.addAction(
            "Run inference + Save Masks",
            lambda: self._set_infer_mode(save_masks=True),
        )
        self._infer_menu_btn.setMenu(infer_menu)
        infer_run_row.addWidget(self._infer_menu_btn)

        self._infer_stop_btn = QPushButton("Stop")
        self._infer_stop_btn.setEnabled(False)
        self._infer_stop_btn.clicked.connect(self._stop_inference)
        infer_run_row.addWidget(self._infer_stop_btn)
        lay.addLayout(infer_run_row)

        self._infer_log = QTextEdit()
        self._infer_log.setReadOnly(True)
        self._infer_log.setMaximumHeight(80)
        lay.addWidget(self._infer_log)

        return section

    def _build_training_section(self) -> _CollapsibleSection:
        section = _CollapsibleSection("2 — Training", expanded=False)
        lay = section.content_layout()

        lay.addWidget(
            self._help_label(
                "Add training videos (one sample per frame). "
                "Masks are auto-detected next to each video.",
                tooltip=(
                    "Each video frame is exported as a separate training image.\n\n"
                    "Mask volumes (TIFF/NPY) must have the same frame count as the video.\n\n"
                    "YOLO: polygon instance segmentation via Ultralytics.\n\n"
                    "U-Net / U-Net++: one model with one output channel per class.\n\n"
                    "Cascade: hierarchical coarse-to-fine masks. "
                    f"Tree: {format_hierarchy_chain()}. "
                    "Crack and Kernel are both inside Pecan (not nested in each other). "
                    "Include Pecan when training inner classes with cascade.\n\n"
                    "Check which mask classes to include. Not every video needs every class.\n\n"
                    "Fine-tuning: load a checkpoint under Inference, enable "
                    "'Fine-tune from loaded weights', add new videos, use ~10–20 epochs "
                    "and learning rate ~1e-4.\n\n"
                    "Pecan-only frames: leave the all-class filter OFF and keep "
                    "'Suppress crack/kernel on pecan-only frames' ON so intact shell "
                    "frames teach the model not to over-segment.\n\n"
                    "Per-class files: '<video> - <Class>' or combined label-map TIFFs "
                    "(1=Crack, 2=Kernel, 3=Pecan)."
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
        rescan_masks_btn = QPushButton("Rescan masks")
        rescan_masks_btn.setToolTip(
            "Re-detect mask files next to each listed video. "
            "Use after saving new masks (e.g. from inference)."
        )
        rescan_masks_btn.clicked.connect(self._refresh_training_masks)
        train_videos_row.addWidget(rescan_masks_btn)
        clear_train_btn = QPushButton("Clear list")
        clear_train_btn.clicked.connect(self._clear_training_videos)
        train_videos_row.addWidget(clear_train_btn)
        lay.addLayout(train_videos_row)

        lay.addWidget(QLabel("Training class mappings (name : label IDs):"))
        map_help = QLabel(
            "Auto-detected from masks; edit names/IDs, use * for any label, or add custom classes."
        )
        map_help.setStyleSheet("color: #888; font-size: 11px;")
        map_help.setWordWrap(True)
        lay.addWidget(map_help)

        self._class_container = QVBoxLayout()
        class_wrap = QWidget()
        class_wrap.setLayout(self._class_container)
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setWidget(class_wrap)
        class_scroll.setMaximumHeight(160)
        lay.addWidget(class_scroll)

        map_btns = QHBoxLayout()
        add_class_btn = QPushButton("Add class…")
        add_class_btn.setToolTip(
            "Add a custom training class and assign pixel label ID(s) from a mask file."
        )
        add_class_btn.clicked.connect(self._add_custom_class_mapping)
        map_btns.addWidget(add_class_btn)
        clear_preview_btn = QPushButton("Clear preview")
        clear_preview_btn.clicked.connect(self._clear_label_preview)
        map_btns.addWidget(clear_preview_btn)
        map_btns.addStretch(1)
        lay.addLayout(map_btns)

        self._no_classes_label = QLabel("Add videos to see available mask classes.")
        self._no_classes_label.setStyleSheet("color: #888; font-size: 11px;")
        self._no_classes_label.setWordWrap(True)
        self._no_classes_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self._no_classes_label.setMinimumWidth(0)
        lay.addWidget(self._no_classes_label)

        self._use_trimmed_videos_cb = QCheckBox(
            "Use saved trim ranges (.pecan.json)"
        )
        self._use_trimmed_videos_cb.setChecked(True)
        self._use_trimmed_videos_cb.setToolTip(
            "When checked, training loads only the frame range saved next to each "
            "video as '{stem}.pecan.json' (from Trim frames…). Sidecar TIFF masks "
            "must match that trimmed length.\n\n"
            "When unchecked, the full video is used and masks must match the "
            "full frame count."
        )
        self._use_trimmed_videos_cb.toggled.connect(self._update_dataset_summary)
        lay.addWidget(self._use_trimmed_videos_cb)

        self._dataset_summary = QLabel("Classes: (none)")
        self._dataset_summary.setWordWrap(True)
        self._dataset_summary.setStyleSheet("color: #888;")
        self._dataset_summary.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self._dataset_summary.setMinimumWidth(0)
        lay.addWidget(self._dataset_summary)

        split_row = QHBoxLayout()
        split_row.addWidget(QLabel("Val fraction:"))
        self._val_fraction_spin = QDoubleSpinBox()
        self._val_fraction_spin.setRange(0.0, 0.5)
        self._val_fraction_spin.setSingleStep(0.05)
        self._val_fraction_spin.setDecimals(2)
        self._val_fraction_spin.setValue(0.2)
        self._val_fraction_spin.setToolTip(
            "Fraction of videos or frames held out for validation (0 = no split)."
        )
        self._val_fraction_spin.valueChanged.connect(self._update_dataset_summary)
        split_row.addWidget(self._val_fraction_spin)
        split_row.addWidget(QLabel("Split by:"))
        self._split_mode_combo = QComboBox()
        self._split_mode_combo.addItem("Hold out whole videos", "video")
        self._split_mode_combo.addItem("Random frames", "frame")
        self._split_mode_combo.setToolTip(
            "Video split keeps all frames from a video in the same set. "
            "With one video, frame split is used automatically."
        )
        self._split_mode_combo.currentIndexChanged.connect(self._update_dataset_summary)
        split_row.addWidget(self._split_mode_combo)
        lay.addLayout(split_row)

        augment_section = _CollapsibleSection("Augmentations", expanded=False)
        augment_lay = augment_section.content_layout()
        augment_lay.addWidget(
            self._help_label(
                "On-the-fly training augmentations (masks are transformed with images).",
                tooltip=(
                    "These map to Ultralytics YOLO training augmentations applied each epoch.\n\n"
                    "Saturation / brightness / hue jitter help under different lighting and color.\n"
                    "Rotation and scale help when framing or distance varies.\n"
                    "RandAugment adds extra random color and contrast variation."
                ),
            )
        )
        self._aug_enabled_cb = QCheckBox("Enable augmentations")
        self._aug_enabled_cb.setChecked(True)
        self._aug_enabled_cb.toggled.connect(self._update_augment_controls_enabled)
        augment_lay.addWidget(self._aug_enabled_cb)

        self._aug_degrees_spin = self._spin_row(
            augment_lay,
            "Max rotation (±°):",
            45,
            0,
            90,
            5,
            tooltip="Random rotation up to this angle in either direction.",
        )
        self._aug_scale_spin = self._fraction_row(
            augment_lay,
            "Random scale:",
            0.5,
            tooltip="Random zoom in/out (Ultralytics scale gain, 0–1).",
        )
        self._aug_sat_spin = self._fraction_row(
            augment_lay,
            "Saturation / colorfulness:",
            0.7,
            tooltip="Random saturation change (hsv_s). Higher = more vivid color swings.",
        )
        self._aug_bright_spin = self._fraction_row(
            augment_lay,
            "Brightness:",
            0.5,
            tooltip="Random brightness change (hsv_v).",
        )
        self._aug_hue_spin = self._fraction_row(
            augment_lay,
            "Hue shift:",
            0.02,
            max_val=0.2,
            tooltip="Subtle random hue rotation (hsv_h).",
        )
        self._aug_fliplr_spin = self._fraction_row(
            augment_lay,
            "Horizontal flip prob:",
            0.5,
            tooltip="Probability of mirroring left↔right.",
        )
        self._aug_randaugment_cb = QCheckBox("RandAugment (extra color & contrast jitter)")
        self._aug_randaugment_cb.setChecked(True)
        self._aug_randaugment_cb.setToolTip(
            "Adds RandAugment photometric transforms on top of the HSV jitter above."
        )
        augment_lay.addWidget(self._aug_randaugment_cb)
        self._aug_controls = [
            self._aug_degrees_spin,
            self._aug_scale_spin,
            self._aug_sat_spin,
            self._aug_bright_spin,
            self._aug_hue_spin,
            self._aug_fliplr_spin,
            self._aug_randaugment_cb,
        ]
        self._yolo_augment_section = augment_section
        lay.addWidget(augment_section)

        self._cascade_options = QWidget()
        cascade_lay = QVBoxLayout(self._cascade_options)
        cascade_lay.setContentsMargins(0, 0, 0, 0)
        self._cascade_hierarchy_help = self._help_label(
            f"Cascade only: {format_hierarchy_chain()}. "
            "Crack and Kernel are siblings under Pecan.",
        )
        cascade_lay.addWidget(self._cascade_hierarchy_help)
        arch_row = QHBoxLayout()
        arch_row.addWidget(QLabel("Architecture:"))
        self._cascade_arch_combo = QComboBox()
        self._cascade_arch_combo.addItem("U-Net++ (recommended)", ARCH_UNETPP)
        self._cascade_arch_combo.addItem("U-Net", ARCH_UNET)
        arch_row.addWidget(self._cascade_arch_combo)
        cascade_lay.addLayout(arch_row)

        enc_row = QHBoxLayout()
        enc_row.addWidget(QLabel("Encoder:"))
        self._cascade_encoder_combo = QComboBox()
        for name in ("mobilenet_v2", "efficientnet-b0", "resnet34"):
            self._cascade_encoder_combo.addItem(name, name)
        self._cascade_encoder_combo.setToolTip(
            "mobilenet_v2 is fastest; efficientnet-b0 is slower but often more accurate."
        )
        enc_row.addWidget(self._cascade_encoder_combo)
        cascade_lay.addLayout(enc_row)

        self._cascade_image_size_spin = self._spin_row(
            cascade_lay,
            "Train image size:",
            384,
            256,
            1024,
            32,
            tooltip="Frames are resized square for training and inference. 384 is faster than 512.",
        )
        self._cascade_flip_spin = self._fraction_row(
            cascade_lay,
            "Horizontal flip prob:",
            0.5,
            tooltip="Random mirror augmentation during cascade training.",
        )
        self._cascade_require_all_classes_cb = QCheckBox(
            "Only use frames whose labels contain all selected classes"
        )
        self._cascade_require_all_classes_cb.setToolTip(
            "Keep only frames where every selected class has mask pixels.\n"
            "Useful for balanced crack/kernel learning, but the model never sees "
            "pecan-only frames and may over-segment intact shell. Leave OFF and use "
            "'Suppress crack/kernel on pecan-only frames' instead for most datasets."
        )
        self._cascade_require_all_classes_cb.toggled.connect(self._update_dataset_summary)
        cascade_lay.addWidget(self._cascade_require_all_classes_cb)
        self._train_absent_inner_cb = QCheckBox(
            "Suppress crack/kernel on intact-shell frames only"
        )
        self._train_absent_inner_cb.setChecked(True)
        self._train_absent_inner_cb.setToolTip(
            "Only when a frame has pecan labels but NO crack and NO kernel masks, "
            "train crack/kernel to stay off.\n\n"
            "Frames with crack but missing kernel labels are partial annotations — "
            "kernel loss is skipped (not treated as 'no kernel')."
        )
        self._train_absent_inner_cb.toggled.connect(self._update_dataset_summary)
        cascade_lay.addWidget(self._train_absent_inner_cb)
        lay.addWidget(self._cascade_options)

        self._fine_tune_cb = QCheckBox("Fine-tune from loaded weights")
        self._fine_tune_cb.setToolTip(
            "Continue training from the model loaded under Inference (same backend). "
            "Add new training videos, use fewer epochs (e.g. 10–20), and a lower learning "
            "rate (e.g. 1e-4). Image size and classes should match the checkpoint."
        )
        self._fine_tune_cb.toggled.connect(self._on_fine_tune_toggled)
        lay.addWidget(self._fine_tune_cb)

        self._epochs_spin = self._spin_row(lay, "Epochs:", 50, 1, 500, 1)
        self._batch_spin = self._spin_row(lay, "Batch size:", 4, 1, 64, 1)
        self._lr_dspin = self._dspin_row(lay, "Learning rate:", 1e-3, 1e-5, 1.0, 1e-4)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._populate_device_combo(self._device_combo)
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
        self._train_btn = QPushButton("Train model")
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

        return section

    def _spin_row(self, parent, label, default, lo, hi, step, *, tooltip: str = ""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        if tooltip:
            spin.setToolTip(tooltip)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _fraction_row(
        self,
        parent,
        label: str,
        default: float,
        *,
        max_val: float = 1.0,
        tooltip: str = "",
    ):
        row = QHBoxLayout()
        lbl = QLabel(label)
        if tooltip:
            lbl.setToolTip(tooltip)
        spin = QDoubleSpinBox()
        spin.setDecimals(2)
        spin.setRange(0.0, max_val)
        spin.setSingleStep(0.05)
        spin.setValue(default)
        if tooltip:
            spin.setToolTip(tooltip)
        row.addWidget(lbl)
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

    def _update_augment_controls_enabled(self, enabled: bool) -> None:
        for control in self._aug_controls:
            control.setEnabled(enabled)

    def _collect_augment_config(self) -> YoloTrainAugmentConfig:
        return YoloTrainAugmentConfig(
            enabled=self._aug_enabled_cb.isChecked(),
            degrees=float(self._aug_degrees_spin.value()),
            scale=float(self._aug_scale_spin.value()),
            hsv_h=float(self._aug_hue_spin.value()),
            hsv_s=float(self._aug_sat_spin.value()),
            hsv_v=float(self._aug_bright_spin.value()),
            fliplr=float(self._aug_fliplr_spin.value()),
            randaugment=self._aug_randaugment_cb.isChecked(),
        )

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

    def _apply_loaded_weights(self, path: str) -> None:
        backend = detect_seg_checkpoint_backend(path)
        if backend == BACKEND_CASCADE and _SMP_AVAILABLE:
            if hasattr(self, "_backend_combo"):
                idx = self._backend_combo.findData(BACKEND_CASCADE)
                if idx >= 0:
                    self._backend_combo.setCurrentIndex(idx)
        elif backend == BACKEND_UNET and _SMP_AVAILABLE:
            if hasattr(self, "_backend_combo"):
                idx = self._backend_combo.findData(BACKEND_UNET)
                if idx >= 0:
                    self._backend_combo.setCurrentIndex(idx)
        elif backend == BACKEND_YOLO and _ULTRA_AVAILABLE:
            if hasattr(self, "_backend_combo"):
                idx = self._backend_combo.findData(BACKEND_YOLO)
                if idx >= 0:
                    self._backend_combo.setCurrentIndex(idx)

        self._weights_path = path
        self._weights_backend = backend
        self._weights_label.setText(path)
        self._weights_label.setStyleSheet("")
        if backend == BACKEND_CASCADE:
            suffix, from_brackets = guess_cascade_save_suffix(
                path, fallback_index=self._weights_fallback_counter
            )
        elif backend == BACKEND_UNET:
            suffix, from_brackets = guess_unet_save_suffix(
                path, fallback_index=self._weights_fallback_counter
            )
        else:
            suffix, from_brackets = guess_save_suffix_from_weights(
                path, fallback_index=self._weights_fallback_counter
            )
        if not from_brackets:
            self._weights_fallback_counter += 1
        self._save_suffix_edit.setText(suffix)
        self._on_backend_changed()
        self._update_fine_tune_enabled()

    def _update_fine_tune_enabled(self) -> None:
        if not hasattr(self, "_fine_tune_cb"):
            return
        backend = self._selected_backend()
        can_ft = bool(
            self._weights_path
            and self._weights_backend == backend
            and Path(self._weights_path).is_file()
        )
        self._fine_tune_cb.setEnabled(can_ft)
        if not can_ft:
            self._fine_tune_cb.setChecked(False)

    def _on_fine_tune_toggled(self, checked: bool = False) -> None:
        if checked and hasattr(self, "_lr_dspin") and self._lr_dspin.value() >= 5e-4:
            self._lr_dspin.setValue(1e-4)
        if checked and hasattr(self, "_epochs_spin") and self._epochs_spin.value() > 30:
            self._epochs_spin.setValue(20)

    def _init_weights_for_training(self) -> str | None:
        if not getattr(self, "_fine_tune_cb", None) or not self._fine_tune_cb.isChecked():
            return None
        if not self._weights_path:
            return None
        if self._weights_backend != self._selected_backend():
            return None
        return str(self._weights_path)

    def _load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load segmentation weights", "", "Weights (*.pt *.pth)"
        )
        if not path:
            return
        self._apply_loaded_weights(path)

    def _browse_infer_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select media for inference", "", _MEDIA_FILTER
        )
        for p in paths:
            resolved = str(Path(p).resolve())
            if resolved not in self._infer_file_paths:
                self._infer_file_paths.append(resolved)
                item = QListWidgetItem(Path(resolved).name)
                item.setToolTip(resolved)
                self._infer_file_list.addItem(item)

    def _browse_infer_directories(self):
        dlg = QFileDialog(self, "Select directories for inference")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        tree = dlg.findChild(QTreeView)
        if tree is not None:
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QDialog.Accepted:
            return
        for p in dlg.selectedFiles():
            self._add_infer_directory(p)

    def _add_infer_directory(self, path: str) -> None:
        resolved = str(Path(path).resolve())
        if resolved in self._infer_dir_paths:
            return
        self._infer_dir_paths.append(resolved)
        video_count = len(discover_videos_in_directory(resolved))
        item = QListWidgetItem(f"[dir] {resolved} ({video_count} video(s))")
        item.setToolTip(resolved)
        self._infer_file_list.addItem(item)

    def _collect_infer_disk_paths(self) -> List[str]:
        paths: List[str] = []
        seen: set[str] = set()
        for raw in self._infer_file_paths:
            resolved = str(Path(raw).resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
        for directory in self._infer_dir_paths:
            for video in discover_videos_in_directory(directory):
                resolved = str(video.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)
        return paths

    def _clear_infer_files(self):
        self._infer_file_paths.clear()
        self._infer_dir_paths.clear()
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
        self._refresh_training_classes()
        self._update_dataset_summary()

    def _refresh_training_masks(self, video_path: str | None = None) -> None:
        for row in self._training_rows:
            if video_path is None or row.video_path == str(Path(video_path).resolve()):
                row.refresh_masks()
        self._refresh_training_classes()
        self._update_dataset_summary()

    def _remove_training_video(self, path: str):
        for i, row in enumerate(self._training_rows):
            if row.video_path == path:
                self._training_rows.pop(i)
                self._train_video_list.takeItem(i)
                break
        self._refresh_training_classes()
        self._update_dataset_summary()

    def _clear_training_videos(self):
        self._training_rows.clear()
        self._train_video_list.clear()
        self._refresh_training_classes()
        self._update_dataset_summary()

    def _discovered_class_sources(self) -> Dict[str, str]:
        """Map discovered class key → representative mask path."""
        sources: Dict[str, str] = {}
        for row in self._training_rows:
            for cls, path in row.masks.items():
                sources.setdefault(cls, str(Path(path).resolve()))
        return sources

    def _mask_source_options(self) -> List[Tuple[str, str]]:
        return sorted(self._discovered_class_sources().items(), key=lambda kv: kv[0].lower())

    def _clear_class_mapping_rows(self) -> None:
        for row in self._class_mapping_rows:
            row.setParent(None)
        self._class_mapping_rows.clear()

    def _add_mapping_row(
        self,
        *,
        name: str,
        source_key: str,
        source_path: str,
        ids_text: str = "*",
        enabled: bool = True,
    ) -> _ClassMappingRow:
        row = _ClassMappingRow(
            name=name,
            source_key=source_key,
            source_path=source_path,
            ids_text=ids_text,
            enabled=enabled,
            source_options=self._mask_source_options(),
        )
        row.changed.connect(self._update_dataset_summary)
        row.preview_requested.connect(self._preview_class_mapping)
        row.remove_requested.connect(self._remove_class_mapping_row)
        self._class_container.addWidget(row)
        self._class_mapping_rows.append(row)
        return row

    def _remove_class_mapping_row(self, row: _ClassMappingRow) -> None:
        if row in self._class_mapping_rows:
            self._class_mapping_rows.remove(row)
        row.setParent(None)
        self._update_dataset_summary()

    def _add_custom_class_mapping(self) -> None:
        from napari.utils.notifications import show_warning

        options = self._mask_source_options()
        if not options:
            show_warning("Add training videos with mask files before creating a class.")
            return
        source_key, source_path = options[0]
        existing = {
            (r.name_edit.text().strip() or "").lower() for r in self._class_mapping_rows
        }
        name = "NewClass"
        i = 2
        while name.lower() in existing:
            name = f"NewClass{i}"
            i += 1
        self._add_mapping_row(
            name=name,
            source_key=source_key,
            source_path=source_path,
            ids_text="1",
            enabled=True,
        )
        self._no_classes_label.setVisible(False)
        self._update_dataset_summary()

    def _refresh_training_classes(self):
        prev: Dict[str, tuple[bool, str, str, str]] = {}
        for row in self._class_mapping_rows:
            mapping = row.to_mapping()
            key = row.source_key
            prev[key] = (
                row.enable_cb.isChecked(),
                row.name_edit.text().strip() or key,
                row.ids_edit.text().strip() or "*",
                row.source_path,
            )
            if mapping is not None:
                prev[mapping.name] = prev[key]

        self._clear_class_mapping_rows()
        sources = self._discovered_class_sources()
        video_count = len(self._training_rows)
        has_classes = bool(sources) or bool(prev)

        self._no_classes_label.setVisible(not sources)
        if not sources:
            self._no_classes_label.setText(
                "Add videos to see available mask classes."
                if video_count == 0
                else "No mask files detected for the added videos."
            )
            # Keep purely custom rows that still have a valid source path.
            return

        for source_key, source_path in sorted(sources.items(), key=lambda kv: kv[0].lower()):
            enabled, name, ids_text = True, source_key, "*"
            if source_key in prev:
                enabled, name, ids_text, _ = prev[source_key]
            else:
                try:
                    data = load_mask_volume(source_path)
                    ids = default_label_ids_for_discovered_class(source_key, data)
                    ids_text = format_label_ids_text(ids)
                except Exception:
                    ids_text = "*"
            self._add_mapping_row(
                name=name,
                source_key=source_key,
                source_path=source_path,
                ids_text=ids_text,
                enabled=enabled,
            )

        # Re-attach custom mappings whose names/source_keys are not in discovery.
        for key, (enabled, name, ids_text, source_path) in prev.items():
            if key in sources:
                continue
            if any(r.source_key == key or r.name_edit.text().strip() == name for r in self._class_mapping_rows):
                continue
            # Custom: reuse path if it still exists among discovered files.
            matched = next(
                (
                    (sk, sp)
                    for sk, sp in sources.items()
                    if str(Path(sp).resolve()) == str(Path(source_path).resolve())
                ),
                None,
            )
            if matched is None:
                continue
            self._add_mapping_row(
                name=name,
                source_key=matched[0],
                source_path=matched[1],
                ids_text=ids_text,
                enabled=enabled,
            )

        for row in self._class_mapping_rows:
            row.update_source_options(self._mask_source_options())

    def _enabled_mappings(self) -> List[ClassLabelMapping]:
        out: List[ClassLabelMapping] = []
        for row in self._class_mapping_rows:
            mapping = row.to_mapping()
            if mapping is not None and mapping.enabled:
                out.append(mapping)
        return out

    def _selected_training_classes(self) -> set[str]:
        return {m.name for m in self._enabled_mappings()}

    def _label_ids_by_class(self) -> Dict[str, set[int] | None]:
        return {m.name: m.label_ids for m in self._enabled_mappings()}

    def _training_entries(self) -> List[Tuple[str, Dict[str, str]]]:
        mappings = self._enabled_mappings()
        entries: List[Tuple[str, Dict[str, str]]] = []
        for row in self._training_rows:
            masks: Dict[str, str] = {}
            for mapping in mappings:
                path = row.masks.get(mapping.source_key)
                if path is None:
                    want = str(Path(mapping.source_path).resolve())
                    for p in row.masks.values():
                        if str(Path(p).resolve()) == want:
                            path = p
                            break
                if path is not None:
                    masks[mapping.name] = str(path)
            if masks:
                entries.append((row.video_path, masks))
        return entries

    def _preview_class_mapping(self, row: _ClassMappingRow) -> None:
        from napari.utils.notifications import show_warning

        mapping = row.to_mapping()
        if mapping is None:
            show_warning("Fix the class name / label IDs before previewing.")
            return

        video_path = None
        mask_path = None
        for train_row in self._training_rows:
            path = train_row.masks.get(mapping.source_key)
            if path is None:
                want = str(Path(mapping.source_path).resolve())
                for p in train_row.masks.values():
                    if str(Path(p).resolve()) == want:
                        path = p
                        break
            if path is not None:
                video_path = train_row.video_path
                mask_path = path
                break
        if video_path is None or mask_path is None:
            show_warning("No training video provides the selected mask source.")
            return

        try:
            apply_saved = bool(self._use_trimmed_videos_cb.isChecked())
            frames = load_video_rgb_frames(
                video_path, apply_saved_range=apply_saved
            )
            mask_vol = load_mask_volume(mask_path)
            binary = binary_mask_for_label_ids(mask_vol, mapping.label_ids)
            t = 0
            if binary.ndim == 3:
                # Prefer a frame that actually contains the selected labels.
                for i in range(binary.shape[0]):
                    if np.any(binary[i]):
                        t = i
                        break
                frame = frames[min(t, frames.shape[0] - 1)]
                mask_frame = binary[min(t, binary.shape[0] - 1)]
            else:
                frame = frames[0]
                mask_frame = binary

            self._upsert_preview_layers(frame, mask_frame, mapping)
        except Exception as exc:
            show_warning(f"Preview failed: {exc}")

    def _upsert_preview_layers(
        self,
        frame: np.ndarray,
        mask_frame: np.ndarray,
        mapping: ClassLabelMapping,
    ) -> None:
        ids_txt = format_label_ids_text(mapping.label_ids)
        img_name = _PREVIEW_IMAGE_LAYER
        mask_name = f"{_PREVIEW_MASK_LAYER} [{mapping.name}:{ids_txt}]"

        def _find_layer(prefix: str):
            for layer in self._viewer.layers:
                if str(getattr(layer, "name", "")).startswith(prefix):
                    return layer
            return None

        img_layer = _find_layer(_PREVIEW_IMAGE_LAYER)
        if img_layer is not None:
            img_layer.data = frame
            img_layer.refresh()
        else:
            self._viewer.add_image(frame, name=img_name, rgb=True)

        mask_layer = _find_layer(_PREVIEW_MASK_LAYER)
        if mask_layer is not None:
            mask_layer.data = mask_frame.astype(np.uint8)
            mask_layer.name = mask_name
            mask_layer.refresh()
        else:
            self._viewer.add_labels(
                mask_frame.astype(np.uint8),
                name=mask_name,
                opacity=0.55,
            )

    def _clear_label_preview(self) -> None:
        for layer in list(self._viewer.layers):
            name = str(getattr(layer, "name", ""))
            if name.startswith(_PREVIEW_IMAGE_LAYER) or name.startswith(
                _PREVIEW_MASK_LAYER
            ):
                try:
                    self._viewer.layers.remove(layer)
                except Exception:
                    pass

    def _update_dataset_summary(self):
        selected = self._selected_training_classes()
        if self._training_rows and not selected:
            self._dataset_summary.setText("Select at least one training class.")
            return

        entries = self._training_entries()
        if not entries:
            if not self._training_rows:
                self._dataset_summary.setText("Classes: (none)")
            else:
                self._dataset_summary.setText(
                    f"{len(self._training_rows)} video(s) added, no masks detected yet."
                )
            return
        try:
            apply_saved_range = bool(self._use_trimmed_videos_cb.isChecked())
            label_ids_by_class = self._label_ids_by_class()
            if self._is_smp_backend():
                summarize_fn = (
                    summarize_cascade_frame_usage
                    if self._selected_backend() == BACKEND_CASCADE
                    else summarize_unet_frame_usage
                )
                summary = summarize_fn(
                    entries,
                    sorted(selected),
                    val_fraction=self._val_fraction_spin.value(),
                    split_by=str(self._split_mode_combo.currentData()),
                    require_all_classes_in_frame=bool(
                        getattr(self, "_cascade_require_all_classes_cb", None)
                        and self._cascade_require_all_classes_cb.isChecked()
                    ),
                    apply_saved_range=apply_saved_range,
                    label_ids_by_class=label_ids_by_class,
                )
            else:
                summary_obj = summarize_training_dataset(
                    entries,
                    val_fraction=self._val_fraction_spin.value(),
                    split_by=str(self._split_mode_combo.currentData()),
                    apply_saved_range=apply_saved_range,
                    label_ids_by_class=label_ids_by_class,
                )
                summary = format_dataset_summary(summary_obj)
            self._dataset_summary.setText(summary)
        except Exception as exc:
            self._dataset_summary.setText(f"Dataset error: {exc}")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Training output directory")
        if not path:
            return
        self._output_dir = Path(path)
        self._output_dir_label.setText(str(self._output_dir))

    def _populate_device_combo(self, combo: QComboBox) -> None:
        combo.addItem("auto", "auto")
        combo.addItem("cpu", "cpu")
        try:
            import torch

            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    combo.addItem(f"cuda:{i}", str(i))
        except Exception:
            pass

    def _combo_device_value(self, combo: QComboBox) -> str:
        value = combo.currentData()
        raw = str(value if value is not None else combo.currentText())
        return resolve_yolo_device(raw)

    def _device_value(self) -> str:
        return self._combo_device_value(self._device_combo)

    def _infer_device_raw(self) -> str:
        value = self._infer_device_combo.currentData()
        return str(value if value is not None else self._infer_device_combo.currentText())

    def _infer_device_value(self) -> str:
        return resolve_yolo_device(self._infer_device_raw())

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

    def _layer_source_path(self, layer: Image) -> str | None:
        meta = getattr(layer, "metadata", None) or {}
        if isinstance(meta, dict):
            src = meta.get("source_path")
            if src:
                return str(Path(src).resolve())
        return None

    def _collect_inference_sources(self) -> List[Tuple[str, np.ndarray, str | None]]:
        sources: List[Tuple[str, np.ndarray, str | None]] = []
        for cb, layer in self._infer_layer_checkboxes:
            if cb.isChecked() and layer in self._viewer.layers:
                sources.append(
                    (
                        layer.name,
                        self._layer_rgb_frames(layer),
                        self._layer_source_path(layer),
                    )
                )
        for path in self._collect_infer_disk_paths():
            name, frames = self._load_infer_source_frames(path)
            sources.append((name, frames, str(Path(path).resolve())))
        return sources

    def _set_infer_mode(self, *, save_masks: bool) -> None:
        self._infer_save_masks = save_masks
        self._infer_main_btn.setText(
            "Run inference + Save Masks" if save_masks else "Run inference"
        )
        self._infer_skip_layers_checkbox.setEnabled(save_masks)
        if not save_masks:
            self._infer_skip_layers_checkbox.setChecked(False)

    def _run_selected_inference(self) -> None:
        self._run_inference(save_masks=self._infer_save_masks)

    def _run_inference(self, *, save_masks: bool = False):
        from napari.utils.notifications import show_warning

        if self._weights_path is None:
            show_warning("Browse and load a trained model first.")
            return
        sources = self._collect_inference_sources()
        if not sources:
            show_warning("Select at least one napari layer or file for inference.")
            return
        if save_masks and not any(src for _, _, src in sources):
            show_warning(
                "Saving masks requires sources with a file path on disk. "
                "Use 'Browse files…', 'Browse directories…', or load videos from disk into napari."
            )
            return

        self._set_infer_running(True)
        self._infer_main_btn.reset_progress()
        self._infer_log.clear()
        self._last_infer_napari_layer_names = [
            layer.name
            for cb, layer in self._infer_layer_checkboxes
            if cb.isChecked() and layer in self._viewer.layers
        ]
        self._last_infer_save_masks = save_masks
        self._last_infer_skip_layers = self._infer_skip_layers_checkbox.isChecked()

        self._infer_worker = _InferWorker(
            self._weights_path,
            sources,
            self._infer_device_value(),
            self._weights_backend or self._selected_backend(),
            save_masks=save_masks,
            save_suffix=self._save_suffix_edit.text(),
            save_fmt=str(self._save_fmt_combo.currentData() or "tiff"),
        )
        self._infer_worker.signals.log.connect(self._infer_log.append)
        self._infer_worker.signals.progress.connect(self._on_infer_progress)
        self._infer_worker.signals.finished.connect(self._on_inference_finished)
        self._infer_worker.signals.error.connect(self._on_inference_error)
        self._infer_worker.start()

    def _set_infer_running(self, running: bool) -> None:
        self._infer_main_btn.setEnabled(not running)
        self._infer_menu_btn.setEnabled(not running)
        self._infer_stop_btn.setEnabled(running)

    def _stop_inference(self):
        if getattr(self, "_infer_worker", None) is None:
            return
        self._infer_log.append("Stop requested…")
        try:
            self._infer_worker.stop()
        except Exception:
            pass

    def _on_infer_progress(self, cur: int, tot: int):
        if tot > 0:
            self._infer_main_btn.set_progress(int(100 * cur / tot))

    def _on_inference_finished(self, results: list):
        from napari.utils.notifications import show_info, show_warning

        self._set_infer_running(False)
        self._infer_main_btn.set_progress(100)
        self._infer_main_btn.reset_progress()
        any_predictions = False
        saved_count = 0
        skip_layers = bool(getattr(self, "_last_infer_skip_layers", False))
        suffix = self._save_suffix_edit.text()
        for name, label_data, saved_path in results:
            if np.any(label_data):
                any_predictions = True
            if saved_path:
                saved_count += 1
            if skip_layers:
                continue
            layer_name = infer_labels_layer_name(name, suffix)
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
        if saved_count and skip_layers:
            show_info(
                f"Saved {saved_count} mask file(s) from {len(results)} sample(s) "
                "(not added to napari layers)."
            )
        elif any_predictions:
            msg = f"Inference complete on {len(results)} sample(s)."
            if saved_count:
                msg += f" Saved {saved_count} mask file(s)."
            show_info(msg)
        elif results:
            show_info(f"Inference stopped after {len(results)} sample(s).")
        else:
            show_warning(
                "Inference finished but no masks were predicted. "
                "Check the inference log for per-frame counts."
            )
        if saved_count:
            refreshed_videos: set[str] = set()
            for _, _, saved_path in results:
                if not saved_path:
                    continue
                video = video_path_for_saved_mask(saved_path)
                if video is not None:
                    refreshed_videos.add(str(video))
            for video_path in refreshed_videos:
                self._refresh_training_masks(video_path)
        self._record_inference_pipeline_steps()

    def _record_inference_pipeline_steps(self) -> None:
        if not self._weights_path:
            return
        layer_names = list(getattr(self, "_last_infer_napari_layer_names", []) or [])
        if not layer_names:
            return
        suffix = self._save_suffix_edit.text()
        save_masks = bool(getattr(self, "_last_infer_save_masks", False))
        save_fmt = str(self._save_fmt_combo.currentData() or "tiff")
        device = self._infer_device_raw()
        weights_path = str(self._weights_path)
        for layer_name in layer_names:
            backend = self._weights_backend or self._selected_backend()
            params = {
                "source_layer": layer_name,
                "weights_path": weights_path,
                "device": device,
                "backend": backend,
                "save_masks": save_masks,
                "save_suffix": suffix,
                "save_fmt": save_fmt,
                "output_mask_layer": infer_labels_layer_name(layer_name, suffix),
            }
            label = {
                BACKEND_CASCADE: "Cascade",
                BACKEND_UNET: "U-Net",
            }.get(backend, "YOLO")
            upsert_pipeline_step(
                kind="yolo_seg.inference",
                description=f"{label} Seg inference on {layer_name}",
                params=params,
                match=lambda st, ln=layer_name: (
                    st.kind == "yolo_seg.inference"
                    and str((st.params or {}).get("source_layer", "")) == ln
                ),
            )

    def _on_inference_error(self, msg: str):
        from napari.utils.notifications import show_error

        self._set_infer_running(False)
        self._infer_main_btn.reset_progress()
        self._infer_log.append(f"ERROR: {msg}")
        show_error(f"Segmentation inference error:\n{msg}")

    def _collect_unet_train_config(self) -> UnetTrainConfig:
        return UnetTrainConfig(
            encoder_name=str(self._cascade_encoder_combo.currentData()),
            architecture=str(self._cascade_arch_combo.currentData()),
            image_size=int(self._cascade_image_size_spin.value()),
            epochs=int(self._epochs_spin.value()),
            batch_size=int(self._batch_spin.value()),
            learning_rate=float(self._lr_dspin.value()),
            val_fraction=float(self._val_fraction_spin.value()),
            split_by=str(self._split_mode_combo.currentData()),
            horizontal_flip_prob=float(self._cascade_flip_spin.value()),
            require_all_classes_in_frame=self._cascade_require_all_classes_cb.isChecked(),
            init_weights_path=self._init_weights_for_training(),
            train_absent_inner_classes=self._train_absent_inner_cb.isChecked(),
            apply_saved_range=bool(self._use_trimmed_videos_cb.isChecked()),
            label_ids_by_class=self._label_ids_by_class(),
        )

    def _collect_cascade_train_config(self) -> CascadeTrainConfig:
        return CascadeTrainConfig(
            encoder_name=str(self._cascade_encoder_combo.currentData()),
            architecture=str(self._cascade_arch_combo.currentData()),
            image_size=int(self._cascade_image_size_spin.value()),
            epochs=int(self._epochs_spin.value()),
            batch_size=int(self._batch_spin.value()),
            learning_rate=float(self._lr_dspin.value()),
            val_fraction=float(self._val_fraction_spin.value()),
            split_by=str(self._split_mode_combo.currentData()),
            horizontal_flip_prob=float(self._cascade_flip_spin.value()),
            require_all_classes_in_frame=self._cascade_require_all_classes_cb.isChecked(),
            init_weights_path=self._init_weights_for_training(),
            train_absent_inner_classes=self._train_absent_inner_cb.isChecked(),
            apply_saved_range=bool(self._use_trimmed_videos_cb.isChecked()),
            label_ids_by_class=self._label_ids_by_class(),
        )

    def _start_training(self):
        from napari.utils.notifications import show_warning

        backend = self._selected_backend()
        if backend == BACKEND_YOLO and not _ULTRA_AVAILABLE:
            show_warning("YOLO backend requires ultralytics.")
            return
        if self._is_smp_backend(backend) and not _SMP_AVAILABLE:
            show_warning("U-Net backends require segmentation-models-pytorch.")
            return

        if self._training_rows and not self._selected_training_classes():
            show_warning("Select at least one mask class for training.")
            return

        entries = self._training_entries()
        if not entries:
            show_warning(
                "Add training videos with detected mask files, "
                "and select at least one class to train."
            )
            return

        selected = sorted(self._selected_training_classes())
        if backend == BACKEND_CASCADE and "Pecan" not in selected:
            inner = [c for c in selected if c != "Pecan"]
            if inner:
                show_warning(
                    "Cascade training needs Pecan masks when training "
                    f"inner classes ({', '.join(inner)})."
                )
                return

        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._progress.setValue(0)
        self._log.clear()

        if backend == BACKEND_CASCADE:
            self._worker = _CascadeTrainWorker(
                video_entries=entries,
                selected_classes=selected,
                config=self._collect_cascade_train_config(),
                device=self._device_value(),
                output_dir=str(self._output_dir),
            )
        elif backend == BACKEND_UNET:
            self._worker = _UnetTrainWorker(
                video_entries=entries,
                selected_classes=selected,
                config=self._collect_unet_train_config(),
                device=self._device_value(),
                output_dir=str(self._output_dir),
            )
        else:
            self._worker = _TrainWorker(
                video_entries=entries,
                epochs=self._epochs_spin.value(),
                batch=self._batch_spin.value(),
                lr=self._lr_dspin.value(),
                device=self._device_value(),
                output_dir=str(self._output_dir),
                val_fraction=self._val_fraction_spin.value(),
                split_by=str(self._split_mode_combo.currentData()),
                augment=self._collect_augment_config(),
                init_weights_path=self._init_weights_for_training(),
                apply_saved_range=bool(self._use_trimmed_videos_cb.isChecked()),
                label_ids_by_class=self._label_ids_by_class(),
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
        self._apply_loaded_weights(path)
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)
        self._log.append(f"Training complete. Best weights: {path}")
        from napari.utils.notifications import show_info

        show_info(f"Training complete. Best weights:\n{path}")

    def _on_training_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log.append(f"ERROR: {msg}")
        from napari.utils.notifications import show_error

        show_error(f"Segmentation training error:\n{msg}")


# Backwards-compatible alias
YoloSegWidget = SegmentationWidget
