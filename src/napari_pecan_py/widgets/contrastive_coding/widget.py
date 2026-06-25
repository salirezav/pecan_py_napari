"""Contrastive Coding dock widget – train on multi-label mask TIFFs and run
interactive similarity inference from napari layers or videos on disk.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from napari.layers import Image
from qtpy.QtCore import QObject, Qt, QThread, Signal
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QStyle,
    QTextEdit,
    QToolButton,
    QTreeView,
    QVBoxLayout,
    QWidget,
)

from napari_pecan_py._reader import VIDEO_EXTENSIONS
from napari_pecan_py.widgets.pecan_ellipse.logic import (
    resolve_time_index_for_volume,
    volume_shape_for_time,
)

from .data import (
    ContrastiveCheckpointMetadata,
    contrastive_checkpoint_filename,
    discover_label_values_in_mask,
    discover_multilabel_mask,
    discover_training_videos_in_directory,
    format_dataset_summary,
    label_values_to_names,
    load_contrastive_checkpoint,
    load_mask_volume,
    load_video_frame_rgb,
    multilabel_frame_to_class_masks,
    save_contrastive_checkpoint,
    summarize_contrastive_dataset,
    validate_training_pair,
)
_TORCH_AVAILABLE = False
try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    pass

SIMILARITY_LAYER_NAME = "Contrastive similarity"
PATCH_CURSOR_LAYER_NAME = "Contrastive patch cursor"

_VIDEO_FILTER = (
    "Videos ("
    + " ".join(f"*{ext}" for ext in sorted(VIDEO_EXTENSIONS))
    + ")"
)


class _TrainingSignals(QObject):
    progress = Signal(int, int, float, float)
    log = Signal(str)
    finished = Signal(object, object)
    error = Signal(str)


class _SweepSignals(QObject):
    progress = Signal(int, int)
    log = Signal(str)
    finished = Signal(object, object)
    error = Signal(str)


class _CollapsibleSection(QWidget):
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


class _TrainingVideoRow(QWidget):
    def __init__(self, video_path: str, *, on_remove, parent=None):
        super().__init__(parent)
        self.video_path = str(Path(video_path).resolve())
        self._on_remove = on_remove
        self.mask_path: str | None = None
        self.class_value_map: Dict[str, int] = {}
        self._scan_mask()

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

    def _scan_mask(self) -> None:
        found = discover_multilabel_mask(self.video_path)
        self.mask_path = str(found) if found else None
        self.class_value_map = {}
        if found is not None:
            try:
                mask_data = load_mask_volume(found)
                values = discover_label_values_in_mask(mask_data)
                names = label_values_to_names(values)
                self.class_value_map = {name: val for val, name in names.items()}
            except Exception:
                self.class_value_map = {}

    def refresh_mask(self) -> bool:
        prev = set(self.class_value_map.keys())
        self._scan_mask()
        changed = set(self.class_value_map.keys()) != prev
        self._update_mask_label()
        return changed

    def _update_mask_label(self) -> None:
        if self.mask_path and self.class_value_map:
            classes = ", ".join(sorted(self.class_value_map))
            self._mask_lbl.setText(f"labels: {classes}")
            self._mask_lbl.setStyleSheet("color: #2ecc71;")
            self._mask_lbl.setToolTip(self.mask_path)
        elif self.mask_path:
            self._mask_lbl.setText("mask found, no classes")
            self._mask_lbl.setStyleSheet("color: #e67e22;")
            self._mask_lbl.setToolTip(self.mask_path)
        else:
            self._mask_lbl.setText("no multi-label mask")
            self._mask_lbl.setStyleSheet("color: #e67e22;")
            self._mask_lbl.setToolTip("Expected a TIFF like '<video> - Pecan,Kernel,Crack.tiff'")


class _SweepWorker(QThread):
    finished = Signal(object, object)
    error = Signal(str)

    def __init__(
        self,
        state_dict,
        metadata: ContrastiveCheckpointMetadata,
        frame: np.ndarray,
        anchor_y: int,
        anchor_x: int,
        stride: int,
        device_str: str,
        threshold_mode: str,
        threshold_value: float,
    ):
        super().__init__()
        self._state_dict = state_dict
        self._metadata = metadata
        self._frame = frame
        self._ay = anchor_y
        self._ax = anchor_x
        self._stride = stride
        self._device_str = device_str
        self._threshold_mode = threshold_mode
        self._threshold_value = threshold_value

    def run(self):
        try:
            from .model import PatchEmbedder, similarity_mask_from_frame

            device = torch.device(self._device_str)
            model = PatchEmbedder(in_channels=self._metadata.in_channels).to(device)
            model.load_state_dict(self._state_dict)
            model.eval()
            mask, stats = similarity_mask_from_frame(
                model,
                self._frame,
                self._ay,
                self._ax,
                patch_size=self._metadata.patch_size,
                stride=self._stride,
                threshold_mode=self._threshold_mode,
                threshold_value=self._threshold_value,
                device=device,
            )
            self.finished.emit({"mask": mask, "stats": stats}, None)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _SweepAllWorker(QThread):
    def __init__(
        self,
        state_dict,
        metadata: ContrastiveCheckpointMetadata,
        frames: np.ndarray,
        anchor_y: int,
        anchor_x: int,
        stride: int,
        device_str: str,
        threshold_mode: str,
        threshold_value: float,
    ):
        super().__init__()
        self.signals = _SweepSignals()
        self._state_dict = state_dict
        self._metadata = metadata
        self._frames = frames
        self._ay = anchor_y
        self._ax = anchor_x
        self._stride = stride
        self._device_str = device_str
        self._threshold_mode = threshold_mode
        self._threshold_value = threshold_value
        self._stop = False

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .model import PatchEmbedder, similarity_mask_from_frame

            device = torch.device(self._device_str)
            model = PatchEmbedder(in_channels=self._metadata.in_channels).to(device)
            model.load_state_dict(self._state_dict)
            model.eval()

            n = int(self._frames.shape[0])
            h, w = int(self._frames.shape[1]), int(self._frames.shape[2])
            out = np.zeros((n, h, w), dtype=np.uint8)
            last_stats: dict = {}
            for t in range(n):
                if self._stop:
                    break
                self.signals.progress.emit(t + 1, n)
                mask, stats = similarity_mask_from_frame(
                    model,
                    self._frames[t],
                    self._ay,
                    self._ax,
                    patch_size=self._metadata.patch_size,
                    stride=self._stride,
                    threshold_mode=self._threshold_mode,
                    threshold_value=self._threshold_value,
                    device=device,
                )
                out[t] = mask
                last_stats = stats
            if self._stop:
                self.signals.log.emit("Sweep stopped by user.")
            self.signals.finished.emit(
                {"mask_volume": out, "stats": last_stats, "anchor_y": self._ay, "anchor_x": self._ax},
                None,
            )
        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class _TrainingWorker(QThread):
    def __init__(
        self,
        entries: Sequence[Tuple[str, str, Dict[str, int]]],
        in_channels: int,
        patch_size: int,
        patches_per_class: int,
        num_negatives: int,
        temperature: float,
        lr: float,
        epochs: int,
        steps_per_epoch: int,
        device_str: str,
    ):
        super().__init__()
        self.signals = _TrainingSignals()
        self._entries = list(entries)
        self._in_channels = in_channels
        self._patch_size = patch_size
        self._patches_per_class = patches_per_class
        self._num_negatives = num_negatives
        self._temperature = temperature
        self._lr = lr
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._device_str = device_str
        self._stop = False
        self._mask_volumes: List[Tuple[str, str, np.ndarray, Dict[str, int], int]] = []

    def stop(self):
        self._stop = True

    def _prepare_sources(self):
        self._mask_volumes.clear()
        for video_path, mask_path, class_value_map in self._entries:
            mask_data = load_mask_volume(mask_path)
            validate_training_pair(video_path, mask_path)
            n_frames = int(mask_data.shape[0] if mask_data.ndim == 3 else 1)
            self._mask_volumes.append(
                (video_path, mask_path, mask_data, class_value_map, n_frames)
            )

    def run(self):
        try:
            from .model import PatchEmbedder, contrastive_loss
            from .sampling import sample_triplets

            self._prepare_sources()
            device = torch.device(self._device_str)
            model = PatchEmbedder(in_channels=self._in_channels).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=self._lr)

            class_names = sorted(
                {name for _, _, _, cmap, _ in self._mask_volumes for name in cmap}
            )
            n_params = sum(p.numel() for p in model.parameters())
            self.signals.log.emit(
                f"Model: {n_params:,} parameters | device={device} | "
                f"videos={len(self._mask_volumes)} | classes={', '.join(class_names)}"
            )

            model.train()
            for epoch in range(1, self._epochs + 1):
                if self._stop:
                    self.signals.log.emit("Training stopped by user.")
                    break

                epoch_loss = 0.0
                for _step in range(1, self._steps_per_epoch + 1):
                    if self._stop:
                        break

                    video_path, _mask_path, mask_data, class_value_map, n_frames = random.choice(
                        self._mask_volumes
                    )
                    frame_index = random.randint(0, n_frames - 1)
                    frame = load_video_frame_rgb(video_path, frame_index)
                    if frame.ndim == 2:
                        frame = frame[..., np.newaxis]
                    if self._in_channels == 3 and frame.shape[-1] >= 3:
                        frame = frame[..., :3]
                    elif self._in_channels == 1:
                        frame = frame[..., :1]

                    if mask_data.ndim == 2:
                        labels = mask_data
                    else:
                        labels = mask_data[frame_index]
                    class_masks = multilabel_frame_to_class_masks(labels, class_value_map)

                    try:
                        anc, pos, neg, _labels = sample_triplets(
                            frame,
                            class_masks,
                            patch_size=self._patch_size,
                            patches_per_class=self._patches_per_class,
                            num_negatives=self._num_negatives,
                        )
                    except ValueError as exc:
                        self.signals.log.emit(f"Skip step: {exc}")
                        continue

                    anc_t = torch.from_numpy(anc).to(device)
                    pos_t = torch.from_numpy(pos).to(device)
                    neg_t = torch.from_numpy(neg).to(device)

                    anc_emb = model(anc_t)
                    pos_emb = model(pos_t)
                    n_batch, k, c, h, w = neg_t.shape
                    neg_emb = model(neg_t.reshape(n_batch * k, c, h, w))
                    neg_emb = neg_emb.reshape(n_batch, k, -1)

                    loss = contrastive_loss(anc_emb, pos_emb, neg_emb, self._temperature)
                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()
                    epoch_loss += loss.item()

                avg_loss = epoch_loss / max(1, self._steps_per_epoch)
                self.signals.progress.emit(epoch, self._epochs, avg_loss, self._lr)
                self.signals.log.emit(f"Epoch {epoch}/{self._epochs}  loss={avg_loss:.4f}")

            metadata = ContrastiveCheckpointMetadata(
                class_names=class_names,
                in_channels=self._in_channels,
                patch_size=self._patch_size,
                temperature=self._temperature,
            )
            self.signals.finished.emit(model.cpu().state_dict(), metadata)

        except Exception as exc:
            import traceback

            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class ContrastiveCodingWidget(QWidget):
    """Napari dock widget for contrastive learning and interactive inference."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._trained_state: dict | None = None
        self._checkpoint_meta: ContrastiveCheckpointMetadata | None = None
        self._worker: _TrainingWorker | None = None
        self._sweep_worker: _SweepWorker | None = None
        self._sweep_all_worker: _SweepAllWorker | None = None

        self._training_rows: List[_TrainingVideoRow] = []
        self._class_checkboxes: Dict[str, QCheckBox] = {}
        self._infer_file_paths: List[str] = []
        self._infer_dir_paths: List[str] = []
        self._infer_layer_checkboxes: List[tuple[QCheckBox, Image]] = []

        self._picker_active = False
        self._last_anchor: tuple[int, int, int | None] | None = None
        self._last_preview_yx: tuple[int, int] | None = None
        self._pending_infer_layer_name: str | None = None
        self._output_dir = Path.cwd() / "contrastive_runs"

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        if not _TORCH_AVAILABLE:
            err = QVBoxLayout()
            err.setContentsMargins(4, 4, 4, 4)
            err.addWidget(
                QLabel(
                    "PyTorch is not installed.\n"
                    "Install it with:  pip install torch\n"
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
        self._viewer.layers.events.inserted.connect(self._on_layers_changed)
        self._viewer.layers.events.removed.connect(self._on_layers_changed)

    def _on_layers_changed(self, _evt=None):
        self._refresh_infer_layers()
        self._refresh_active_source_combo()

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
        weights_row.addWidget(QLabel("Model:"))
        self._weights_label = QLabel("(none loaded)")
        self._weights_label.setStyleSheet("color: #888;")
        self._weights_label.setWordWrap(True)
        weights_row.addWidget(self._weights_label, 1)
        load_btn = QPushButton("Browse…")
        load_btn.clicked.connect(self._load_model)
        weights_row.addWidget(load_btn)
        lay.addLayout(weights_row)

        lay.addWidget(
            self._help_label(
                "Pick a patch on the active source to highlight similar regions. "
                "Use batch sweep to apply the same anchor across all frames.",
            )
        )

        lay.addWidget(QLabel("Active source (for picking):"))
        self._active_source_combo = QComboBox()
        self._active_source_combo.setToolTip(
            "Layer or file used for eyedropper clicks and similarity sweeps."
        )
        self._active_source_combo.currentIndexChanged.connect(self._on_active_source_changed)
        lay.addWidget(self._active_source_combo)

        load_active_btn = QPushButton("Load active file into napari")
        load_active_btn.clicked.connect(self._load_active_file_into_viewer)
        lay.addWidget(load_active_btn)

        lay.addWidget(QLabel("Napari image layers (check to include):"))
        self._infer_layer_container = QVBoxLayout()
        layer_wrap = QWidget()
        layer_wrap.setLayout(self._infer_layer_container)
        layer_scroll = QScrollArea()
        layer_scroll.setWidgetResizable(True)
        layer_scroll.setWidget(layer_wrap)
        layer_scroll.setMaximumHeight(90)
        lay.addWidget(layer_scroll)

        lay.addWidget(QLabel("Videos on disk (files or folders):"))
        self._infer_file_list = QListWidget()
        self._infer_file_list.setMaximumHeight(90)
        lay.addWidget(self._infer_file_list)

        infer_files_row = QHBoxLayout()
        browse_infer_btn = QPushButton("Browse files…")
        browse_infer_btn.clicked.connect(self._browse_infer_files)
        infer_files_row.addWidget(browse_infer_btn)
        browse_dirs_btn = QPushButton("Browse directories…")
        browse_dirs_btn.clicked.connect(self._browse_infer_directories)
        infer_files_row.addWidget(browse_dirs_btn)
        clear_infer_btn = QPushButton("Clear list")
        clear_infer_btn.clicked.connect(self._clear_infer_files)
        infer_files_row.addWidget(clear_infer_btn)
        lay.addLayout(infer_files_row)

        pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Pick patch")
        self._pick_btn.setCheckable(True)
        self._pick_btn.toggled.connect(self._on_pick_toggled)
        pick_row.addWidget(self._pick_btn)
        self._sweep_all_btn = QPushButton("Sweep all frames")
        self._sweep_all_btn.setEnabled(False)
        self._sweep_all_btn.setToolTip(
            "After picking an anchor on one frame, sweep the same (y, x) "
            "across every frame of the active source."
        )
        self._sweep_all_btn.clicked.connect(self._sweep_all_frames)
        pick_row.addWidget(self._sweep_all_btn)
        lay.addLayout(pick_row)
        lay.addWidget(
            self._help_label(
                "A yellow square shows the patch size under the cursor while picking.",
            )
        )

        self._stride_spin = self._spin_row(
            lay,
            "Sweep stride:",
            4,
            1,
            32,
            1,
            "Lower = denser/slower, higher = faster.",
        )
        thresh_mode_row = QHBoxLayout()
        thresh_mode_row.addWidget(QLabel("Match cutoff:"))
        self._thresh_mode_combo = QComboBox()
        self._thresh_mode_combo.addItem("Fraction of best match", "peak_fraction")
        self._thresh_mode_combo.addItem("Absolute cosine sim", "absolute")
        self._thresh_mode_combo.currentIndexChanged.connect(self._update_threshold_controls)
        thresh_mode_row.addWidget(self._thresh_mode_combo, 1)
        lay.addLayout(thresh_mode_row)

        self._thresh_dspin = self._dspin_row(
            lay,
            "Cutoff value:",
            0.92,
            0.5,
            1.0,
            0.01,
            "Fraction of best: 0.92 keeps patches within 92% of the best match.",
        )
        self._update_threshold_controls()

        infer_dev_row = QHBoxLayout()
        infer_dev_row.addWidget(QLabel("Device:"))
        self._infer_device_combo = QComboBox()
        self._populate_device_combo(self._infer_device_combo)
        infer_dev_row.addWidget(self._infer_device_combo)
        lay.addLayout(infer_dev_row)

        self._pick_info = QLabel("")
        lay.addWidget(self._pick_info)

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
                "Add training videos with a multi-label mask TIFF next to each file. "
                "Label values 1/2/3 = Crack / Kernel / Pecan; other values use generic names.",
                tooltip=(
                    "Each video needs a multi-class label volume with the same frame count.\n\n"
                    "Primary filename: '<video> - Pecan,Kernel,Crack.tiff'\n\n"
                    "Check which classes to include. Not every video needs every class."
                ),
            )
        )

        self._train_video_list = QListWidget()
        self._train_video_list.setMaximumHeight(140)
        lay.addWidget(self._train_video_list)

        train_row = QHBoxLayout()
        add_train_btn = QPushButton("Browse videos…")
        add_train_btn.clicked.connect(self._browse_training_videos)
        train_row.addWidget(add_train_btn)
        add_dir_btn = QPushButton("Browse directories…")
        add_dir_btn.clicked.connect(self._browse_training_directories)
        train_row.addWidget(add_dir_btn)
        rescan_btn = QPushButton("Rescan masks")
        rescan_btn.clicked.connect(self._refresh_training_masks)
        train_row.addWidget(rescan_btn)
        clear_btn = QPushButton("Clear list")
        clear_btn.clicked.connect(self._clear_training_videos)
        train_row.addWidget(clear_btn)
        lay.addLayout(train_row)

        lay.addWidget(QLabel("Training classes (check to include):"))
        self._class_container = QVBoxLayout()
        class_wrap = QWidget()
        class_wrap.setLayout(self._class_container)
        class_scroll = QScrollArea()
        class_scroll.setWidgetResizable(True)
        class_scroll.setWidget(class_wrap)
        class_scroll.setMaximumHeight(90)
        lay.addWidget(class_scroll)

        self._no_classes_label = QLabel("Add videos to see available label classes.")
        self._no_classes_label.setStyleSheet("color: #888; font-size: 11px;")
        self._no_classes_label.setWordWrap(True)
        lay.addWidget(self._no_classes_label)

        self._dataset_summary = QLabel("Classes: (none)")
        self._dataset_summary.setWordWrap(True)
        self._dataset_summary.setStyleSheet("color: #888;")
        lay.addWidget(self._dataset_summary)

        hp_group = _CollapsibleSection("Training parameters", expanded=True)
        hp_lay = hp_group.content_layout()
        self._patch_spin = self._spin_row(hp_lay, "Patch size:", 8, 4, 64, 2)
        self._patch_spin.valueChanged.connect(self._on_patch_size_changed)
        self._samples_spin = self._spin_row(hp_lay, "Patches / class:", 64, 8, 1024, 8)
        self._negatives_spin = self._spin_row(hp_lay, "Negatives / anchor:", 4, 1, 16, 1)
        self._temp_dspin = self._dspin_row(hp_lay, "Temperature:", 0.10, 0.01, 2.0, 0.01)
        self._lr_dspin = self._dspin_row(hp_lay, "Learning rate:", 1e-3, 1e-5, 1.0, 1e-4)
        self._epochs_spin = self._spin_row(hp_lay, "Epochs:", 10, 1, 500, 1)
        self._steps_spin = self._spin_row(hp_lay, "Steps / epoch:", 20, 1, 200, 5)
        lay.addWidget(hp_group)

        dev_row = QHBoxLayout()
        dev_row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._populate_device_combo(self._device_combo)
        dev_row.addWidget(self._device_combo)
        lay.addLayout(dev_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Save checkpoints to:"))
        self._output_dir_label = QLabel(str(self._output_dir))
        self._output_dir_label.setWordWrap(True)
        out_row.addWidget(self._output_dir_label, 1)
        out_dir_btn = QPushButton("Browse…")
        out_dir_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(out_dir_btn)
        lay.addLayout(out_row)

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train")
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

    def _spin_row(self, parent, label, default, lo, hi, step, tip=""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        if tip:
            spin.setToolTip(tip)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _dspin_row(self, parent, label, default, lo, hi, step, tip=""):
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QDoubleSpinBox()
        spin.setDecimals(5)
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        if tip:
            spin.setToolTip(tip)
        row.addWidget(lbl)
        row.addWidget(spin)
        parent.addLayout(row)
        return spin

    def _populate_device_combo(self, combo: QComboBox) -> None:
        combo.clear()
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                combo.addItem(f"cuda:{i}")
        combo.addItem("cpu")
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            combo.setCurrentIndex(0)
        else:
            combo.setCurrentIndex(combo.count() - 1)

    def _threshold_mode(self) -> str:
        value = self._thresh_mode_combo.currentData()
        return str(value if value is not None else "peak_fraction")

    def _threshold_value(self) -> float:
        return float(self._thresh_dspin.value())

    def _update_threshold_controls(self, _index: int | None = None) -> None:
        mode = self._threshold_mode()
        if mode == "absolute":
            self._thresh_dspin.setRange(0.0, 1.0)
            self._thresh_dspin.setSingleStep(0.02)
            if self._thresh_dspin.value() > 1.0 or self._thresh_dspin.value() < 0.5:
                self._thresh_dspin.setValue(0.88)
            self._thresh_dspin.setToolTip(
                "Minimum cosine similarity (0–1) for a patch to be highlighted."
            )
        else:
            self._thresh_dspin.setRange(0.5, 1.0)
            self._thresh_dspin.setSingleStep(0.01)
            if self._thresh_dspin.value() < 0.5:
                self._thresh_dspin.setValue(0.92)
            self._thresh_dspin.setToolTip(
                "Keep patches with similarity >= this fraction of the best match on the frame."
            )

    def _format_sweep_stats(self, stats: dict) -> str:
        if not stats:
            return "Sweep complete."
        return (
            f"Anchor ({stats.get('anchor_y', '?')}, {stats.get('anchor_x', '?')}) | "
            f"peak={stats.get('peak_sim', 0):.3f} cutoff={stats.get('cutoff', 0):.3f} | "
            f"{stats.get('patches_highlighted', 0)}/{stats.get('patches_total', 0)} patches"
        )

    def _register_pick_callbacks(self) -> None:
        if hasattr(self._viewer, "mouse_press_callbacks"):
            self._viewer.mouse_press_callbacks.append(self._inference_click)
        else:
            self._viewer.mouse_drag_callbacks.append(self._inference_click)

    def _unregister_pick_callbacks(self) -> None:
        for callbacks in (
            getattr(self._viewer, "mouse_press_callbacks", None),
            getattr(self._viewer, "mouse_drag_callbacks", None),
        ):
            if callbacks is None:
                continue
            try:
                callbacks.remove(self._inference_click)
            except ValueError:
                pass

    def _refresh_infer_layers(self, _evt=None):
        prev_checked = {
            layer.name
            for cb, layer in self._infer_layer_checkboxes
            if cb.isChecked()
        }
        if self._pending_infer_layer_name:
            prev_checked.add(self._pending_infer_layer_name)

        for cb, _ in self._infer_layer_checkboxes:
            cb.setParent(None)
        self._infer_layer_checkboxes.clear()

        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                cb = QCheckBox(layer.name)
                cb.setChecked(layer.name in prev_checked)
                cb.toggled.connect(self._on_infer_layer_toggled)
                self._infer_layer_container.addWidget(cb)
                self._infer_layer_checkboxes.append((cb, layer))

        self._refresh_active_source_combo()

        if self._pending_infer_layer_name:
            self._set_active_source_layer(self._pending_infer_layer_name)
            self._pending_infer_layer_name = None

    def _on_infer_layer_toggled(self, _checked: bool) -> None:
        self._refresh_active_source_combo()
        if self._picker_active and self._resolve_infer_image_layer() is None:
            self._remove_patch_cursor()
            self._last_preview_yx = None

    def _image_layer_for_path(self, path: str) -> Image | None:
        resolved = str(Path(path).resolve())
        for layer in self._viewer.layers:
            if not isinstance(layer, Image):
                continue
            meta = getattr(layer, "metadata", None) or {}
            if not isinstance(meta, dict):
                continue
            src = meta.get("source_path")
            if src and str(Path(src).resolve()) == resolved:
                return layer
        return None

    def _select_infer_layer(self, layer_name: str) -> None:
        for cb, layer in self._infer_layer_checkboxes:
            if layer.name == layer_name:
                cb.setChecked(True)
                break
        self._set_active_source_layer(layer_name)

    def _set_active_source_layer(self, layer_name: str) -> None:
        target = ("layer", layer_name)
        idx = self._active_source_combo.findData(target)
        if idx >= 0:
            self._active_source_combo.setCurrentIndex(idx)

    def _collect_infer_disk_paths(self) -> List[str]:
        paths: List[str] = []
        seen: set[str] = set()
        for raw in self._infer_file_paths:
            resolved = str(Path(raw).resolve())
            if resolved not in seen:
                seen.add(resolved)
                paths.append(resolved)
        for directory in self._infer_dir_paths:
            for video in discover_training_videos_in_directory(directory):
                resolved = str(video.resolve())
                if resolved not in seen:
                    seen.add(resolved)
                    paths.append(resolved)
        return paths

    def _refresh_active_source_combo(self, *_args):
        prev = self._active_source_combo.currentData()
        self._active_source_combo.blockSignals(True)
        self._active_source_combo.clear()
        for cb, layer in self._infer_layer_checkboxes:
            if cb.isChecked() and layer in self._viewer.layers:
                self._active_source_combo.addItem(f"layer: {layer.name}", ("layer", layer.name))
        for path in self._collect_infer_disk_paths():
            self._active_source_combo.addItem(f"file: {Path(path).name}", ("file", path))

        selected = False
        if prev is not None:
            idx = self._active_source_combo.findData(prev)
            if idx >= 0:
                self._active_source_combo.setCurrentIndex(idx)
                selected = True

        if not selected and prev is not None and prev[0] == "file":
            loaded = self._image_layer_for_path(prev[1])
            if loaded is not None:
                idx = self._active_source_combo.findData(("layer", loaded.name))
                if idx >= 0:
                    self._active_source_combo.setCurrentIndex(idx)
                    selected = True

        if not selected:
            for cb, layer in self._infer_layer_checkboxes:
                if cb.isChecked() and layer in self._viewer.layers:
                    idx = self._active_source_combo.findData(("layer", layer.name))
                    if idx >= 0:
                        self._active_source_combo.setCurrentIndex(idx)
                        break

        self._active_source_combo.blockSignals(False)

    def _active_image_layer(self) -> Image | None:
        data = self._active_source_combo.currentData()
        if not data:
            return None
        kind, value = data
        if kind == "layer":
            try:
                layer = self._viewer.layers[value]
                if isinstance(layer, Image):
                    return layer
            except KeyError:
                return None
        return None

    def _resolve_infer_image_layer(self) -> Image | None:
        layer = self._active_image_layer()
        if layer is not None:
            return layer

        active = self._viewer.layers.selection.active
        if isinstance(active, Image):
            for cb, lyr in self._infer_layer_checkboxes:
                if lyr is active and cb.isChecked() and lyr in self._viewer.layers:
                    self._set_active_source_layer(lyr.name)
                    return lyr

        for cb, lyr in self._infer_layer_checkboxes:
            if cb.isChecked() and lyr in self._viewer.layers:
                self._set_active_source_layer(lyr.name)
                return lyr
        return None

    def _ensure_active_layer_source(self) -> Image | None:
        layer = self._resolve_infer_image_layer()
        if layer is None:
            return None
        self._set_active_source_layer(layer.name)
        return layer

    def _on_active_source_changed(self, _index: int) -> None:
        layer = self._active_image_layer()
        if layer is not None:
            try:
                self._viewer.layers.selection.active = layer
            except Exception:
                pass
        if self._picker_active:
            self._remove_patch_cursor()
            self._last_preview_yx = None

    def _load_active_file_into_viewer(self):
        data = self._active_source_combo.currentData()
        if not data or data[0] != "file":
            self._infer_log.append("Select a disk file as the active source first.")
            return
        path = data[1]
        name = Path(path).stem
        try:
            existing = self._viewer.layers[name]
            if isinstance(existing, Image):
                self._viewer.layers.selection.active = existing
                self._select_infer_layer(name)
                return
        except KeyError:
            pass
        from napari_pecan_py.widgets.yolo_seg.model import load_video_rgb_frames

        self._pending_infer_layer_name = name
        frames = load_video_rgb_frames(path)
        layer = self._viewer.add_image(
            frames,
            name=name,
            metadata={"source_path": str(Path(path).resolve())},
        )
        self._viewer.layers.selection.active = layer
        self._select_infer_layer(name)
        self._infer_log.append(f"Loaded {name} into napari.")

    def _browse_infer_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select videos for inference", "", _VIDEO_FILTER)
        for p in paths:
            resolved = str(Path(p).resolve())
            if resolved not in self._infer_file_paths:
                self._infer_file_paths.append(resolved)
                item = QListWidgetItem(Path(resolved).name)
                item.setToolTip(resolved)
                self._infer_file_list.addItem(item)
        self._refresh_active_source_combo()

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
            resolved = str(Path(p).resolve())
            if resolved in self._infer_dir_paths:
                continue
            self._infer_dir_paths.append(resolved)
            n = len(discover_training_videos_in_directory(resolved))
            item = QListWidgetItem(f"[dir] {resolved} ({n} video(s))")
            item.setToolTip(resolved)
            self._infer_file_list.addItem(item)
        self._refresh_active_source_combo()

    def _clear_infer_files(self):
        self._infer_file_paths.clear()
        self._infer_dir_paths.clear()
        self._infer_file_list.clear()
        self._refresh_active_source_combo()

    def _browse_training_videos(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Add training videos", "", _VIDEO_FILTER)
        for p in paths:
            self._add_training_video(p)

    def _browse_training_directories(self):
        dlg = QFileDialog(self, "Select directories for training")
        dlg.setFileMode(QFileDialog.Directory)
        dlg.setOption(QFileDialog.DontUseNativeDialog, True)
        dlg.setOption(QFileDialog.ShowDirsOnly, True)
        tree = dlg.findChild(QTreeView)
        if tree is not None:
            tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        if dlg.exec() != QDialog.Accepted:
            return
        for directory in dlg.selectedFiles():
            for video in discover_training_videos_in_directory(directory):
                self._add_training_video(str(video))

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

    def _refresh_training_masks(self):
        for row in self._training_rows:
            row.refresh_mask()
        self._refresh_training_classes()
        self._update_dataset_summary()

    def _discovered_class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in self._training_rows:
            for cls in row.class_value_map:
                counts[cls] = counts.get(cls, 0) + 1
        return counts

    def _refresh_training_classes(self):
        prev_checked = {cls: cb.isChecked() for cls, cb in self._class_checkboxes.items()}
        for cb in self._class_checkboxes.values():
            cb.setParent(None)
        self._class_checkboxes.clear()

        counts = self._discovered_class_counts()
        has_classes = bool(counts)
        self._no_classes_label.setVisible(not has_classes)
        if not has_classes:
            self._no_classes_label.setText(
                "Add videos to see available label classes."
                if not self._training_rows
                else "No multi-label mask TIFF detected for the added videos."
            )
            return

        for cls in sorted(counts):
            videos_with = counts[cls]
            cb = QCheckBox(f"{cls} ({videos_with}/{len(self._training_rows)} video(s))")
            cb.setChecked(prev_checked.get(cls, True))
            cb.toggled.connect(self._update_dataset_summary)
            self._class_container.addWidget(cb)
            self._class_checkboxes[cls] = cb

    def _selected_training_classes(self) -> set[str]:
        return {cls for cls, cb in self._class_checkboxes.items() if cb.isChecked()}

    def _training_entries(self) -> List[Tuple[str, str, Dict[str, int]]]:
        selected = self._selected_training_classes()
        entries: List[Tuple[str, str, Dict[str, int]]] = []
        for row in self._training_rows:
            if not row.mask_path or not row.class_value_map:
                continue
            cmap = {
                name: value
                for name, value in row.class_value_map.items()
                if name in selected
            }
            if cmap:
                entries.append((row.video_path, row.mask_path, cmap))
        return entries

    def _update_dataset_summary(self):
        selected = self._selected_training_classes()
        if self._training_rows and not selected:
            self._dataset_summary.setText("Select at least one training class.")
            return
        entries = self._training_entries()
        if not entries:
            self._dataset_summary.setText(
                "Classes: (none)"
                if not self._training_rows
                else f"{len(self._training_rows)} video(s) added, no masks detected yet."
            )
            return
        try:
            summary = summarize_contrastive_dataset(entries)
            self._dataset_summary.setText(format_dataset_summary(summary))
        except Exception as exc:
            self._dataset_summary.setText(f"Dataset error: {exc}")

    def _browse_output_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Training output directory")
        if not path:
            return
        self._output_dir = Path(path)
        self._output_dir_label.setText(str(self._output_dir))

    def _apply_checkpoint(self, state_dict: dict, metadata: ContrastiveCheckpointMetadata | None):
        self._trained_state = state_dict
        self._checkpoint_meta = metadata
        if metadata is not None:
            self._patch_spin.setValue(metadata.patch_size)
            self._temp_dspin.setValue(metadata.temperature)
            classes = ", ".join(metadata.class_names) or "(unknown)"
            self._weights_label.setText(f"classes: {classes} | patch={metadata.patch_size}")
        else:
            self._weights_label.setText("(loaded — no metadata)")
        self._weights_label.setStyleSheet("")

    def _current_metadata(self) -> ContrastiveCheckpointMetadata:
        if self._checkpoint_meta is not None:
            return ContrastiveCheckpointMetadata(
                class_names=list(self._checkpoint_meta.class_names),
                in_channels=self._checkpoint_meta.in_channels,
                patch_size=self._patch_spin.value(),
                embed_dim=self._checkpoint_meta.embed_dim,
                temperature=self._temp_dspin.value(),
            )
        return ContrastiveCheckpointMetadata(
            class_names=[],
            in_channels=3,
            patch_size=self._patch_spin.value(),
            temperature=self._temp_dspin.value(),
        )

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load model weights", "", "PyTorch (*.pt *.pth)")
        if not path:
            return
        try:
            state, meta = load_contrastive_checkpoint(path)
            self._apply_checkpoint(state, meta)
            self._infer_log.append(f"Model loaded from {path}")
        except Exception as exc:
            self._infer_log.append(f"Failed to load: {exc}")

    def _start_training(self):
        from napari.utils.notifications import show_warning

        if self._training_rows and not self._selected_training_classes():
            show_warning("Select at least one label class for training.")
            return
        entries = self._training_entries()
        if not entries:
            show_warning(
                "Add training videos with multi-label mask TIFFs and select at least one class."
            )
            return

        self._log.clear()
        self._log.append("Starting training…")
        self._progress.setValue(0)
        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)

        self._worker = _TrainingWorker(
            entries=entries,
            in_channels=3,
            patch_size=self._patch_spin.value(),
            patches_per_class=self._samples_spin.value(),
            num_negatives=self._negatives_spin.value(),
            temperature=self._temp_dspin.value(),
            lr=self._lr_dspin.value(),
            epochs=self._epochs_spin.value(),
            steps_per_epoch=self._steps_spin.value(),
            device_str=self._device_combo.currentText(),
        )
        self._worker.signals.progress.connect(self._on_train_progress)
        self._worker.signals.log.connect(self._log.append)
        self._worker.signals.finished.connect(self._on_train_finished)
        self._worker.signals.error.connect(self._on_train_error)
        self._worker.start()

    def _stop_training(self):
        if self._worker is not None:
            self._worker.stop()
            self._log.append("Stopping…")

    def _on_train_progress(self, epoch: int, total: int, loss: float, _lr: float):
        self._progress.setValue(int(100 * epoch / total))

    def _on_train_finished(self, state_dict, metadata: ContrastiveCheckpointMetadata):
        self._apply_checkpoint(state_dict, metadata)
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._progress.setValue(100)

        self._output_dir.mkdir(parents=True, exist_ok=True)
        dest = self._output_dir / contrastive_checkpoint_filename(metadata.class_names)
        save_contrastive_checkpoint(dest, state_dict, metadata)
        self._log.append(f"Training complete. Saved: {dest}")

        from napari.utils.notifications import show_info

        show_info(f"Contrastive training complete.\nSaved:\n{dest}")

    def _on_train_error(self, msg: str):
        self._log.append(f"ERROR: {msg}")
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    def _patch_size_for_inference(self) -> int:
        if self._checkpoint_meta is not None:
            return int(self._checkpoint_meta.patch_size)
        return int(self._patch_spin.value())

    def _layer_data_shape(self, layer: Image) -> tuple[int, ...] | None:
        return volume_shape_for_time(layer.data)

    def _pos_to_yx(self, layer: Image, pos) -> tuple[int | None, int | None]:
        try:
            if hasattr(layer, "world_to_data"):
                data_pos = layer.world_to_data(pos)
            else:
                data_pos = pos
            coords = np.asarray(data_pos, dtype=float).ravel()
        except Exception:
            return None, None

        shape = self._layer_data_shape(layer)
        if shape is None or coords.size < 2:
            return None, None

        if len(shape) == 4 and shape[-1] in (3, 4):
            if coords.size >= 3:
                return int(round(float(coords[1]))), int(round(float(coords[2])))
        elif len(shape) == 3 and shape[-1] in (3, 4):
            return int(round(float(coords[0]))), int(round(float(coords[1])))
        elif len(shape) == 3:
            if coords.size >= 3:
                return int(round(float(coords[1]))), int(round(float(coords[2])))
        elif len(shape) == 2:
            return int(round(float(coords[0]))), int(round(float(coords[1])))

        return int(round(float(coords[-2]))), int(round(float(coords[-1])))

    def _time_index_for_layer(self, layer: Image) -> int | None:
        shape = self._layer_data_shape(layer)
        if shape is None:
            return None
        if len(shape) == 4:
            return resolve_time_index_for_volume(layer.data, self._viewer)
        if len(shape) == 3 and shape[-1] not in (3, 4):
            return resolve_time_index_for_volume(layer.data, self._viewer)
        return None

    def _spatial_shape(self, layer: Image) -> tuple[int, int] | None:
        shape = self._layer_data_shape(layer)
        if shape is None:
            return None
        if len(shape) == 4:
            return int(shape[1]), int(shape[2])
        if len(shape) == 3:
            if shape[-1] in (3, 4):
                return int(shape[0]), int(shape[1])
            return int(shape[1]), int(shape[2])
        if len(shape) == 2:
            return int(shape[0]), int(shape[1])
        return None

    def _extract_frame_from_layer(
        self, layer: Image, time_index: int | None
    ) -> np.ndarray | None:
        data = layer.data
        shape = self._layer_data_shape(layer)
        if shape is None:
            return None
        try:
            if len(shape) == 4:
                t = 0 if time_index is None else int(time_index)
                t = max(0, min(t, shape[0] - 1))
                frame = np.asarray(data[t])
            elif len(shape) == 3:
                if shape[-1] in (3, 4):
                    frame = np.asarray(data)
                else:
                    t = 0 if time_index is None else int(time_index)
                    t = max(0, min(t, shape[0] - 1))
                    frame = np.asarray(data[t])
            elif len(shape) == 2:
                frame = np.asarray(data)
            else:
                return None
        except Exception:
            return None

        if frame.ndim == 2:
            frame = frame[..., np.newaxis]
        if frame.shape[-1] >= 3:
            frame = frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            max_v = float(np.nanmax(frame)) if frame.size else 0.0
            if max_v <= 1.0:
                frame = np.clip(frame, 0.0, 1.0) * 255.0
            else:
                frame = np.clip(frame, 0.0, 255.0)
            frame = frame.astype(np.uint8)
        elif np.issubdtype(frame.dtype, np.integer):
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    def _resolve_click_target(
        self, layer: Image, pos
    ) -> tuple[np.ndarray, int | None, int, int]:
        shape = self._layer_data_shape(layer)
        if shape is None:
            raise ValueError("Could not read image layer shape.")
        if len(shape) not in (2, 3, 4):
            raise ValueError(f"Unsupported image shape: {shape}")

        y, x = self._pos_to_yx(layer, pos)
        if y is None or x is None:
            raise ValueError("Could not map click position to image coordinates.")

        spatial = self._spatial_shape(layer)
        if spatial is None:
            raise ValueError(f"Unsupported image shape: {shape}")
        h, w = spatial
        if not (0 <= y < h and 0 <= x < w):
            raise ValueError(f"Click outside image bounds ({y}, {x}).")

        t = self._time_index_for_layer(layer)
        frame = self._extract_frame_from_layer(layer, t)
        if frame is None:
            raise ValueError(f"Could not read frame from layer (shape {shape}).")
        return frame, t, y, x

    def _patch_rect_vertices(
        self, layer: Image, y: int, x: int, time_index: int | None
    ) -> np.ndarray | None:
        patch_size = self._patch_size_for_inference()
        half = patch_size / 2.0
        shape = self._layer_data_shape(layer)
        if shape is None:
            return None
        if len(shape) == 4 or (len(shape) == 3 and shape[-1] not in (3, 4)):
            t = 0.0 if time_index is None else float(time_index)
            return np.array(
                [
                    [t, y - half, x - half],
                    [t, y - half, x + half],
                    [t, y + half, x + half],
                    [t, y + half, x - half],
                ]
            )
        return np.array(
            [
                [y - half, x - half],
                [y - half, x + half],
                [y + half, x + half],
                [y + half, x - half],
            ]
        )

    def _sync_cursor_layer_to_image(self, cursor_layer, image_layer: Image) -> None:
        for attr in ("scale", "translate", "rotate", "shear"):
            if hasattr(image_layer, attr) and hasattr(cursor_layer, attr):
                try:
                    setattr(cursor_layer, attr, getattr(image_layer, attr))
                except Exception:
                    pass
        if hasattr(image_layer, "affine") and hasattr(cursor_layer, "affine"):
            try:
                cursor_layer.affine = image_layer.affine
            except Exception:
                pass

    def _update_patch_cursor(self, layer: Image, y: int, x: int) -> None:
        t = self._time_index_for_layer(layer)
        rect = self._patch_rect_vertices(layer, y, x, t)
        if rect is None:
            return
        try:
            cursor_layer = self._viewer.layers[PATCH_CURSOR_LAYER_NAME]
            cursor_layer.data = [rect]
            self._sync_cursor_layer_to_image(cursor_layer, layer)
            cursor_layer.refresh()
        except KeyError:
            cursor_layer = self._viewer.add_shapes(
                [rect],
                shape_type="polygon",
                edge_color="yellow",
                face_color="transparent",
                edge_width=2,
                name=PATCH_CURSOR_LAYER_NAME,
            )
            self._sync_cursor_layer_to_image(cursor_layer, layer)

    def _remove_patch_cursor(self) -> None:
        try:
            self._viewer.layers.remove(PATCH_CURSOR_LAYER_NAME)
        except ValueError:
            pass

    def _on_patch_size_changed(self, _value: int) -> None:
        if not self._picker_active or self._last_preview_yx is None:
            return
        layer = self._resolve_infer_image_layer()
        if layer is None:
            return
        y, x = self._last_preview_yx
        self._update_patch_cursor(layer, y, x)

    def _patch_preview_move(self, viewer, event) -> None:
        if not self._picker_active:
            return
        layer = self._resolve_infer_image_layer()
        if layer is None:
            return
        y, x = self._pos_to_yx(layer, event.position)
        if y is None or x is None:
            return
        spatial = self._spatial_shape(layer)
        if spatial is None:
            return
        h, w = spatial
        if not (0 <= y < h and 0 <= x < w):
            self._last_preview_yx = None
            self._remove_patch_cursor()
            return
        self._last_preview_yx = (y, x)
        self._update_patch_cursor(layer, y, x)

    def _on_pick_toggled(self, checked: bool):
        if checked:
            if self._trained_state is None:
                self._infer_log.append("Train or load a model first.")
                self._pick_btn.setChecked(False)
                return
            layer = self._ensure_active_layer_source()
            if layer is None:
                self._infer_log.append(
                    "Check an image layer for inference (or load a disk file first)."
                )
                self._pick_btn.setChecked(False)
                return
            self._viewer.layers.selection.active = layer
            self._picker_active = True
            self._pick_btn.setText("Click a patch on the image…")
            self._register_pick_callbacks()
            self._viewer.mouse_move_callbacks.append(self._patch_preview_move)
        else:
            self._picker_active = False
            self._pick_btn.setText("Pick patch")
            self._unregister_pick_callbacks()
            try:
                self._viewer.mouse_move_callbacks.remove(self._patch_preview_move)
            except ValueError:
                pass
            self._remove_patch_cursor()
            self._last_preview_yx = None

    def _inference_click(self, viewer, event):
        if not self._picker_active:
            return
        if event.button != 1:
            return
        if set(event.modifiers) - {"Shift", "Alt"}:
            return
        layer = self._resolve_infer_image_layer()
        if layer is None:
            return
        try:
            frame, t, y, x = self._resolve_click_target(layer, event.position)
        except ValueError as exc:
            self._infer_log.append(str(exc))
            return

        self._update_patch_cursor(layer, y, x)

        h, w = frame.shape[:2]
        if not (0 <= y < h and 0 <= x < w):
            return

        self._last_anchor = (y, x, t)
        self._sweep_all_btn.setEnabled(True)
        self._pick_info.setText(
            f"Computing similarity at ({y}, {x}), patch={self._patch_size_for_inference()}…"
        )
        self._pick_btn.setEnabled(False)

        meta = self._current_metadata()
        meta = ContrastiveCheckpointMetadata(
            class_names=list(meta.class_names),
            in_channels=meta.in_channels,
            patch_size=self._patch_size_for_inference(),
            embed_dim=meta.embed_dim,
            temperature=meta.temperature,
        )
        self._sweep_worker = _SweepWorker(
            state_dict=self._trained_state,
            metadata=meta,
            frame=frame,
            anchor_y=y,
            anchor_x=x,
            stride=self._stride_spin.value(),
            device_str=self._infer_device_combo.currentText(),
            threshold_mode=self._threshold_mode(),
            threshold_value=self._threshold_value(),
        )
        self._sweep_worker._frame_index = t
        self._sweep_worker._layer = layer
        self._sweep_worker.finished.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.start()

    def _on_sweep_done(self, result: dict, _extra):
        self._pick_btn.setEnabled(True)
        mask = result.get("mask")
        stats = result.get("stats", {})
        if mask is None:
            self._pick_info.setText("Sweep finished with no mask.")
            return
        layer = getattr(self._sweep_worker, "_layer", None)
        t = getattr(self._sweep_worker, "_frame_index", None)
        self._show_similarity_mask(mask, layer, t)
        summary = self._format_sweep_stats(stats)
        self._pick_info.setText(f"{summary} — click another patch or sweep all frames.")
        self._infer_log.append(summary)

    def _show_similarity_mask(self, mask: np.ndarray, layer: Image | None, t: int | None):
        mask = np.asarray(mask)
        if layer is not None and np.asarray(layer.data).ndim == 4 and t is not None and mask.ndim == 2:
            T = np.asarray(layer.data).shape[0]
            full = np.zeros((T,) + mask.shape, dtype=np.uint8)
            full[t] = mask
            mask_out = full
        else:
            mask_out = mask

        try:
            existing = self._viewer.layers[SIMILARITY_LAYER_NAME]
            if tuple(existing.data.shape) != tuple(mask_out.shape):
                self._viewer.layers.remove(SIMILARITY_LAYER_NAME)
                raise KeyError
            existing.data = mask_out
            existing.refresh()
        except KeyError:
            from napari.utils.colormaps import direct_colormap

            cmap = direct_colormap(
                {
                    0: np.array([0, 0, 0, 0], dtype=np.float32),
                    1: np.array([0.0, 1.0, 0.5, 0.6], dtype=np.float32),
                    None: np.array([0, 0, 0, 0], dtype=np.float32),
                }
            )
            labels = self._viewer.add_labels(
                mask_out,
                name=SIMILARITY_LAYER_NAME,
                opacity=0.55,
                colormap=cmap,
            )
            if layer is not None:
                self._sync_cursor_layer_to_image(labels, layer)

    def _on_sweep_error(self, msg: str):
        self._pick_btn.setEnabled(True)
        self._pick_info.setText("")
        self._infer_log.append(f"Sweep error: {msg}")

    def _sweep_all_frames(self):
        if self._trained_state is None or self._last_anchor is None:
            self._infer_log.append("Pick an anchor patch first.")
            return
        layer = self._resolve_infer_image_layer()
        if layer is None:
            self._infer_log.append("Select an active napari layer.")
            return

        shape = self._layer_data_shape(layer)
        if shape is None or len(shape) != 4:
            self._infer_log.append("Sweep all frames requires a multi-frame video layer.")
            return

        ay, ax, anchor_t = self._last_anchor
        n_frames = int(shape[0])
        first = self._extract_frame_from_layer(layer, 0)
        if first is None:
            self._infer_log.append("Could not read frame from active layer.")
            return
        frames = np.empty((n_frames, first.shape[0], first.shape[1], first.shape[2]), dtype=np.uint8)
        frames[0] = first
        for t in range(1, n_frames):
            f = self._extract_frame_from_layer(layer, t)
            if f is None:
                self._infer_log.append(f"Could not read frame {t}.")
                return
            frames[t] = f

        self._sweep_all_btn.setEnabled(False)
        self._pick_btn.setEnabled(False)
        self._infer_log.append(
            f"Sweeping {frames.shape[0]} frame(s) at anchor ({ay}, {ax}) "
            f"from frame {anchor_t if anchor_t is not None else resolve_time_index_for_volume(layer.data, self._viewer)}…"
        )

        meta = self._current_metadata()
        meta = ContrastiveCheckpointMetadata(
            class_names=list(meta.class_names),
            in_channels=meta.in_channels,
            patch_size=self._patch_size_for_inference(),
            embed_dim=meta.embed_dim,
            temperature=meta.temperature,
        )
        self._sweep_all_worker = _SweepAllWorker(
            state_dict=self._trained_state,
            metadata=meta,
            frames=frames,
            anchor_y=ay,
            anchor_x=ax,
            stride=self._stride_spin.value(),
            device_str=self._infer_device_combo.currentText(),
            threshold_mode=self._threshold_mode(),
            threshold_value=self._threshold_value(),
        )
        self._sweep_all_worker._layer = layer
        self._sweep_all_worker.signals.progress.connect(
            lambda c, t: self._pick_info.setText(f"Sweeping frame {c}/{t}…")
        )
        self._sweep_all_worker.signals.finished.connect(self._on_sweep_all_done)
        self._sweep_all_worker.signals.error.connect(self._on_sweep_all_error)
        self._sweep_all_worker.start()

    def _on_sweep_all_done(self, result: dict, _extra):
        self._sweep_all_btn.setEnabled(True)
        self._pick_btn.setEnabled(True)
        mask_volume = result.get("mask_volume")
        stats = result.get("stats", {})
        ay = result.get("anchor_y", stats.get("anchor_y"))
        ax = result.get("anchor_x", stats.get("anchor_x"))
        layer = getattr(self._sweep_all_worker, "_layer", None)
        if mask_volume is not None:
            self._show_similarity_mask(np.asarray(mask_volume), layer, t=None)
        summary = self._format_sweep_stats(stats)
        self._pick_info.setText(
            f"All-frame sweep at ({ay}, {ax}). {summary}"
        )
        self._infer_log.append(f"All-frame sweep at ({ay}, {ax}). {summary}")

    def _on_sweep_all_error(self, msg: str):
        self._sweep_all_btn.setEnabled(True)
        self._pick_btn.setEnabled(True)
        self._infer_log.append(f"Sweep all error: {msg}")
