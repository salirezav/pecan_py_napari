"""YOLO Segmentation widget with separate Inference and Training sections."""

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

from ..pipeline_recorder.state import upsert_pipeline_step

from .model import (
    IMAGE_EXTENSIONS,
    discover_mask_files,
    discover_videos_in_directory,
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
        val_fraction: float,
        split_by: str,
        augment: YoloTrainAugmentConfig,
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
        sources: Sequence[Tuple[str, np.ndarray, str | None]],
        device: str,
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
                out = run_yolo_seg_inference_on_frames(
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
        self._on_remove = on_remove
        self.masks = discover_mask_files(self.video_path)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)

        name = QLabel(Path(self.video_path).name)
        name.setToolTip(self.video_path)
        layout.addWidget(name, 1)

        self._mask_lbl = QLabel()
        layout.addWidget(self._mask_lbl, 2)
        self._update_mask_label()

        remove_btn = QPushButton()
        remove_btn.setIcon(self.style().standardIcon(QStyle.SP_TrashIcon))
        remove_btn.setFixedSize(24, 24)
        remove_btn.clicked.connect(lambda: self._on_remove(self.video_path))
        layout.addWidget(remove_btn)

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
            self._mask_lbl.setText(f"masks: {mask_text}")
            self._mask_lbl.setStyleSheet("color: #2ecc71;")
        else:
            self._mask_lbl.setText("no masks found")
            self._mask_lbl.setStyleSheet("color: #e67e22;")
        self._mask_lbl.setToolTip(
            "\n".join(str(p) for p in self.masks.values()) or "No mask files detected"
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


class YoloSegWidget(QWidget):
    """Napari dock widget for YOLO segmentation inference and training."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._weights_path: str | None = None
        self._training_rows: List[_TrainingVideoRow] = []
        self._class_checkboxes: Dict[str, QCheckBox] = {}
        self._infer_file_paths: List[str] = []
        self._infer_dir_paths: List[str] = []
        self._infer_layer_checkboxes: List[tuple[QCheckBox, Image]] = []
        self._infer_save_masks = False
        self._weights_fallback_counter = 1

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _ULTRA_AVAILABLE:
            err = QVBoxLayout()
            err.setContentsMargins(4, 4, 4, 4)
            err.addWidget(
                QLabel(
                    "The 'ultralytics' package is not installed.\n"
                    "Install it with:\n"
                    "  pip install ultralytics\n"
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
        body_layout.addWidget(self._build_inference_section())
        body_layout.addWidget(self._build_training_section())
        body_layout.addStretch(1)

        scroll.setWidget(body)
        layout.addWidget(scroll)

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
                "Add training videos (one YOLO sample per frame). "
                "Masks are auto-detected next to each video.",
                tooltip=(
                    "Each video frame is exported as a separate training image.\n\n"
                    "Mask volumes (TIFF/NPY) must have the same frame count as the video.\n\n"
                    "Check which mask classes to include in training. Not every video "
                    "needs every class—for example, some videos may only have Crack masks.\n\n"
                    "A class may be empty on some frames—for example, Crack is only "
                    "labeled when visible on camera, while Pecan is expected on every frame.\n\n"
                    "Mask files live in the same folder as the video.\n\n"
                    "Per-class files: '<video> - <Class>' or '<video> - … - <Class>'.\n\n"
                    "Combined label-map TIFFs (one file, multiple classes) are also "
                    "supported: pixel value 1 = Crack, 2 = Kernel, 3 = Pecan."
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

        lay.addWidget(QLabel("Training classes (check to include):"))
        self._class_container = QVBoxLayout()
        class_wrap = QWidget()
        class_wrap.setLayout(self._class_container)
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setWidget(class_wrap)
        class_scroll.setMaximumHeight(90)
        lay.addWidget(class_scroll)

        self._no_classes_label = QLabel("Add videos to see available mask classes.")
        self._no_classes_label.setStyleSheet("color: #888; font-size: 11px;")
        self._no_classes_label.setWordWrap(True)
        self._no_classes_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Minimum
        )
        self._no_classes_label.setMinimumWidth(0)
        lay.addWidget(self._no_classes_label)

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
        lay.addWidget(augment_section)

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
        self._weights_path = path
        self._weights_label.setText(path)
        self._weights_label.setStyleSheet("")
        suffix, from_brackets = guess_save_suffix_from_weights(
            path, fallback_index=self._weights_fallback_counter
        )
        if not from_brackets:
            self._weights_fallback_counter += 1
        self._save_suffix_edit.setText(suffix)

    def _load_weights(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO weights", "", "YOLO weights (*.pt *.pth)"
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

    def _discovered_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self._training_rows:
            for cls in row.masks:
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _refresh_training_classes(self):
        prev_checked = {
            cls: cb.isChecked() for cls, cb in self._class_checkboxes.items()
        }
        for cb in self._class_checkboxes.values():
            cb.setParent(None)
        self._class_checkboxes.clear()

        counts = self._discovered_class_counts()
        video_count = len(self._training_rows)
        has_classes = bool(counts)

        self._no_classes_label.setVisible(not has_classes)
        if not has_classes:
            self._no_classes_label.setText(
                "Add videos to see available mask classes."
                if video_count == 0
                else "No mask files detected for the added videos."
            )
            return

        for cls in sorted(counts):
            videos_with = counts[cls]
            cb = QCheckBox(f"{cls} ({videos_with}/{video_count} video(s))")
            cb.setChecked(prev_checked.get(cls, True))
            cb.setToolTip(
                f"Include '{cls}' masks in training.\n"
                f"Present in {videos_with} of {video_count} video(s); "
                "videos without this class are still used for other classes."
            )
            cb.toggled.connect(self._update_dataset_summary)
            self._class_container.addWidget(cb)
            self._class_checkboxes[cls] = cb

    def _selected_training_classes(self) -> set[str]:
        return {cls for cls, cb in self._class_checkboxes.items() if cb.isChecked()}

    def _training_entries(self) -> List[Tuple[str, Dict[str, str]]]:
        selected = self._selected_training_classes()
        entries: List[Tuple[str, Dict[str, str]]] = []
        for row in self._training_rows:
            masks = {
                cls: str(path)
                for cls, path in row.masks.items()
                if cls in selected
            }
            if masks:
                entries.append((row.video_path, masks))
        return entries

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
            summary = summarize_training_dataset(
                entries,
                val_fraction=self._val_fraction_spin.value(),
                split_by=str(self._split_mode_combo.currentData()),
            )
            self._dataset_summary.setText(format_dataset_summary(summary))
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
            params = {
                "source_layer": layer_name,
                "weights_path": weights_path,
                "device": device,
                "save_masks": save_masks,
                "save_suffix": suffix,
                "save_fmt": save_fmt,
                "output_mask_layer": infer_labels_layer_name(layer_name, suffix),
            }
            upsert_pipeline_step(
                kind="yolo_seg.inference",
                description=f"YOLO Seg inference on {layer_name}",
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
        show_error(f"YOLO inference error:\n{msg}")

    def _start_training(self):
        from napari.utils.notifications import show_warning

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
            val_fraction=self._val_fraction_spin.value(),
            split_by=str(self._split_mode_combo.currentData()),
            augment=self._collect_augment_config(),
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

        show_info(f"YOLO training complete. Best weights:\n{path}")

    def _on_training_error(self, msg: str):
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._log.append(f"ERROR: {msg}")
        from napari.utils.notifications import show_error

        show_error(f"YOLO training error:\n{msg}")
