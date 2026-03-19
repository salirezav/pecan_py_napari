"""Contrastive Coding dock widget – train a small contrastive embedding
on labelled pecan video frames directly from napari.

The widget reads the Image layer (video frames) and any Labels layers
created by the Color Tuner as class masks.  Each Labels layer is a
separate class; unlabelled pixels are treated as *background*.

Training runs in a background ``QThread`` so the GUI stays responsive.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import QObject, QThread, Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

_TORCH_AVAILABLE = False
try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    pass


SIMILARITY_LAYER_NAME = "Contrastive similarity"


class _TrainingSignals(QObject):
    """Signals emitted by the training worker."""
    progress = Signal(int, int, float, float)  # epoch, total_epochs, loss, lr
    log = Signal(str)
    finished = Signal(object)  # trained model state_dict
    error = Signal(str)


class _SweepWorker(QThread):
    """Compute the similarity heatmap off the main thread."""

    finished = Signal(object)  # np.ndarray similarity map
    error = Signal(str)

    def __init__(self, state_dict, in_channels, frame, anchor_y, anchor_x,
                 patch_size, stride, device_str):
        super().__init__()
        self._state_dict = state_dict
        self._in_channels = in_channels
        self._frame = frame
        self._ay = anchor_y
        self._ax = anchor_x
        self._patch_size = patch_size
        self._stride = stride
        self._device_str = device_str

    def run(self):
        try:
            from .model import PatchEmbedder, compute_similarity_map
            device = torch.device(self._device_str)
            model = PatchEmbedder(in_channels=self._in_channels).to(device)
            model.load_state_dict(self._state_dict)
            model.eval()
            sim = compute_similarity_map(
                model, self._frame, self._ay, self._ax,
                patch_size=self._patch_size, stride=self._stride, device=device,
            )
            self.finished.emit(sim)
        except Exception as exc:
            import traceback
            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _TrainingWorker(QThread):
    """Runs the contrastive training loop off the main thread."""

    def __init__(
        self,
        image_data: np.ndarray,
        class_masks: Dict[str, np.ndarray],
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
        self._stop = False

        self._image_data = image_data
        self._class_masks = class_masks
        self._in_channels = in_channels
        self._patch_size = patch_size
        self._patches_per_class = patches_per_class
        self._num_negatives = num_negatives
        self._temperature = temperature
        self._lr = lr
        self._epochs = epochs
        self._steps_per_epoch = steps_per_epoch
        self._device_str = device_str

    def stop(self):
        self._stop = True

    def run(self):
        try:
            from .model import PatchEmbedder, contrastive_loss
            from .sampling import sample_triplets

            device = torch.device(self._device_str)
            model = PatchEmbedder(in_channels=self._in_channels).to(device)
            optimiser = torch.optim.Adam(model.parameters(), lr=self._lr)

            n_params = sum(p.numel() for p in model.parameters())
            self.signals.log.emit(
                f"Model created: {n_params:,} parameters | device={device}"
            )
            self.signals.log.emit(
                f"Classes: {', '.join(self._class_masks.keys())}"
            )

            model.train()
            for epoch in range(1, self._epochs + 1):
                if self._stop:
                    self.signals.log.emit("Training stopped by user.")
                    break

                epoch_loss = 0.0
                for step in range(1, self._steps_per_epoch + 1):
                    if self._stop:
                        break

                    try:
                        anc, pos, neg, _labels = sample_triplets(
                            self._image_data,
                            self._class_masks,
                            patch_size=self._patch_size,
                            patches_per_class=self._patches_per_class,
                            num_negatives=self._num_negatives,
                        )
                    except ValueError as exc:
                        self.signals.error.emit(str(exc))
                        return

                    anc_t = torch.from_numpy(anc).to(device)
                    pos_t = torch.from_numpy(pos).to(device)
                    neg_t = torch.from_numpy(neg).to(device)

                    anc_emb = model(anc_t)
                    pos_emb = model(pos_t)
                    N, K, C, H, W = neg_t.shape
                    neg_emb = model(neg_t.reshape(N * K, C, H, W))
                    neg_emb = neg_emb.reshape(N, K, -1)

                    loss = contrastive_loss(anc_emb, pos_emb, neg_emb, self._temperature)

                    optimiser.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimiser.step()

                    epoch_loss += loss.item()

                avg_loss = epoch_loss / self._steps_per_epoch
                self.signals.progress.emit(epoch, self._epochs, avg_loss, self._lr)
                self.signals.log.emit(
                    f"Epoch {epoch}/{self._epochs}  loss={avg_loss:.4f}"
                )

            state = model.cpu().state_dict()
            self.signals.finished.emit(state)

        except Exception as exc:
            import traceback
            self.signals.error.emit(f"{exc}\n{traceback.format_exc()}")


class ContrastiveCodingWidget(QWidget):
    """Napari dock widget for contrastive-learning on labelled video frames."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._worker: _TrainingWorker | None = None
        self._trained_state: dict | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        if not _TORCH_AVAILABLE:
            layout.addWidget(QLabel(
                "PyTorch is not installed.\n"
                "Install it with:  pip install torch\n"
                "then restart napari."
            ))
            return

        # ---- Source layer ----------------------------------------------------
        src_group = QGroupBox("Source image layer")
        src_lay = QVBoxLayout(src_group)
        self._image_combo = QComboBox()
        self._image_combo.addItem("(none)", None)
        src_lay.addWidget(self._image_combo)
        layout.addWidget(src_group)

        # ---- Class masks (multi-select) --------------------------------------
        mask_group = QGroupBox("Class masks (Labels layers)")
        mask_lay = QVBoxLayout(mask_group)
        mask_lay.addWidget(QLabel("Check layers to use as classes:"))
        self._mask_container = QVBoxLayout()
        mask_lay.addLayout(self._mask_container)
        self._mask_checkboxes: list[tuple[QCheckBox, Labels]] = []
        layout.addWidget(mask_group)

        # ---- Hyperparameters -------------------------------------------------
        hp_group = QGroupBox("Training parameters")
        hp_lay = QVBoxLayout(hp_group)

        self._patch_spin = self._spin_row(hp_lay, "Patch size:", 8, 4, 64, 2,
                                           "Side length of square patches.")
        self._samples_spin = self._spin_row(hp_lay, "Patches / class:", 64, 8, 1024, 8,
                                             "Patches sampled per class per step.")
        self._negatives_spin = self._spin_row(hp_lay, "Negatives / anchor:", 4, 1, 16, 1)
        self._temp_dspin = self._dspin_row(hp_lay, "Temperature:", 0.10, 0.01, 2.0, 0.01)
        self._lr_dspin = self._dspin_row(hp_lay, "Learning rate:", 1e-3, 1e-5, 1.0, 1e-4)
        self._epochs_spin = self._spin_row(hp_lay, "Epochs:", 10, 1, 500, 1)
        self._steps_spin = self._spin_row(hp_lay, "Steps / epoch:", 20, 1, 200, 5,
                                           "Mini-batch steps per epoch.")

        row = QHBoxLayout()
        row.addWidget(QLabel("Device:"))
        self._device_combo = QComboBox()
        self._device_combo.addItem("cpu")
        if _TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                self._device_combo.addItem(f"cuda:{i}")
        row.addWidget(self._device_combo)
        hp_lay.addLayout(row)

        layout.addWidget(hp_group)

        # ---- Controls --------------------------------------------------------
        ctrl_group = QGroupBox("Training")
        ctrl_lay = QVBoxLayout(ctrl_group)

        btn_row = QHBoxLayout()
        self._train_btn = QPushButton("Train")
        self._train_btn.clicked.connect(self._start_training)
        btn_row.addWidget(self._train_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_training)
        btn_row.addWidget(self._stop_btn)
        ctrl_lay.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        ctrl_lay.addWidget(self._progress)

        self._save_btn = QPushButton("Save model…")
        self._save_btn.setEnabled(False)
        self._save_btn.clicked.connect(self._save_model)
        ctrl_lay.addWidget(self._save_btn)

        layout.addWidget(ctrl_group)

        # ---- Inference / eyedropper ------------------------------------------
        inf_group = QGroupBox("Inference (eyedropper)")
        inf_lay = QVBoxLayout(inf_group)

        pick_row = QHBoxLayout()
        self._pick_btn = QPushButton("Pick patch")
        self._pick_btn.setCheckable(True)
        self._pick_btn.setChecked(False)
        self._pick_btn.toggled.connect(self._on_pick_toggled)
        pick_row.addWidget(self._pick_btn)
        inf_lay.addLayout(pick_row)

        self._stride_spin = self._spin_row(inf_lay, "Sweep stride:", 4, 1, 32, 1,
                                            "Lower = denser/slower, higher = faster.")
        self._thresh_dspin = self._dspin_row(inf_lay, "Sim. threshold:", 0.5, -1.0, 1.0, 0.05,
                                              "Only highlight regions above this similarity.")

        load_row = QHBoxLayout()
        self._load_btn = QPushButton("Load model…")
        self._load_btn.clicked.connect(self._load_model)
        load_row.addWidget(self._load_btn)
        inf_lay.addLayout(load_row)

        self._pick_info = QLabel("")
        inf_lay.addWidget(self._pick_info)

        layout.addWidget(inf_group)

        self._picker_active = False
        self._sweep_worker: _SweepWorker | None = None

        # ---- Log -------------------------------------------------------------
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setMaximumHeight(140)
        layout.addWidget(self._log)

        layout.addStretch(1)

        # ---- Events ----------------------------------------------------------
        self._refresh_layers()
        self._viewer.layers.events.inserted.connect(self._refresh_layers)
        self._viewer.layers.events.removed.connect(self._refresh_layers)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # layer bookkeeping
    # ------------------------------------------------------------------

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
                masks[layer.name] = layer.data
        return masks

    # ------------------------------------------------------------------
    # training lifecycle
    # ------------------------------------------------------------------

    def _start_training(self):
        img_layer = self._selected_image()
        if img_layer is None:
            self._log.append("Select an image layer first.")
            return
        masks = self._selected_masks()
        if len(masks) < 1:
            self._log.append("Check at least one Labels layer as class mask.")
            return

        image_data = np.asarray(img_layer.data)
        if image_data.ndim == 3 and image_data.shape[-1] in (3, 4):
            in_ch = 3
            image_data = image_data[..., :3]
        elif image_data.ndim == 4 and image_data.shape[-1] in (3, 4):
            in_ch = 3
            image_data = image_data[..., :3]
        elif image_data.ndim == 2:
            in_ch = 1
            image_data = image_data[..., np.newaxis]
        elif image_data.ndim == 3:
            in_ch = 1
            image_data = image_data[..., np.newaxis]
        else:
            self._log.append(f"Unexpected image shape: {image_data.shape}")
            return

        self._log.clear()
        self._log.append("Starting training…")
        self._progress.setValue(0)
        self._train_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._save_btn.setEnabled(False)

        self._worker = _TrainingWorker(
            image_data=image_data,
            class_masks=masks,
            in_channels=in_ch,
            patch_size=self._patch_spin.value(),
            patches_per_class=self._samples_spin.value(),
            num_negatives=self._negatives_spin.value(),
            temperature=self._temp_dspin.value(),
            lr=self._lr_dspin.value(),
            epochs=self._epochs_spin.value(),
            steps_per_epoch=self._steps_spin.value(),
            device_str=self._device_combo.currentText(),
        )
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.log.connect(self._on_log)
        self._worker.signals.finished.connect(self._on_finished)
        self._worker.signals.error.connect(self._on_error)
        self._worker.start()

    def _stop_training(self):
        if self._worker is not None:
            self._worker.stop()
            self._log.append("Stopping…")

    def _on_progress(self, epoch: int, total: int, loss: float, lr: float):
        pct = int(100 * epoch / total)
        self._progress.setValue(pct)

    def _on_log(self, msg: str):
        self._log.append(msg)

    def _on_finished(self, state_dict):
        self._trained_state = state_dict
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._save_btn.setEnabled(True)
        self._progress.setValue(100)
        self._log.append("Training complete.")

    def _on_error(self, msg: str):
        self._log.append(f"ERROR: {msg}")
        self._train_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)

    # ------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------

    def _save_model(self):
        if self._trained_state is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save model weights", "", "PyTorch (*.pt *.pth)"
        )
        if not path:
            return
        torch.save(self._trained_state, path)
        self._log.append(f"Model saved to {path}")

    # ------------------------------------------------------------------
    # load
    # ------------------------------------------------------------------

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load model weights", "", "PyTorch (*.pt *.pth)"
        )
        if not path:
            return
        try:
            self._trained_state = torch.load(path, map_location="cpu", weights_only=True)
            self._save_btn.setEnabled(True)
            self._log.append(f"Model loaded from {path}")
        except Exception as exc:
            self._log.append(f"Failed to load: {exc}")

    # ------------------------------------------------------------------
    # inference eyedropper
    # ------------------------------------------------------------------

    def _on_pick_toggled(self, checked: bool):
        if checked:
            if self._trained_state is None:
                self._log.append("Train or load a model first.")
                self._pick_btn.setChecked(False)
                return
            self._picker_active = True
            self._pick_btn.setText("Click a patch on the image…")
            self._viewer.mouse_drag_callbacks.append(self._inference_click)
        else:
            self._picker_active = False
            self._pick_btn.setText("Pick patch")
            try:
                self._viewer.mouse_drag_callbacks.remove(self._inference_click)
            except ValueError:
                pass

    def _inference_click(self, viewer, event):
        if not self._picker_active:
            return
        if event.button != 1:
            return
        if set(event.modifiers) - {"Shift", "Alt"}:
            return
        img_layer = self._selected_image()
        if img_layer is None:
            self._log.append("Select an image layer first.")
            return

        data = np.asarray(img_layer.data)
        ndim = data.ndim
        pos = event.position

        if ndim == 4:
            t = int(round(pos[0]))
            y, x = int(round(pos[1])), int(round(pos[2]))
            t = max(0, min(t, data.shape[0] - 1))
            frame = data[t]
        elif ndim == 3:
            t = None
            y, x = int(round(pos[0])), int(round(pos[1]))
            frame = data
        else:
            return

        if frame.ndim == 3 and frame.shape[-1] in (3, 4):
            frame = frame[..., :3]
            in_ch = 3
        else:
            if frame.ndim == 2:
                frame = frame[..., np.newaxis]
            in_ch = 1

        H, W = frame.shape[:2]
        if not (0 <= y < H and 0 <= x < W):
            return

        self._pick_info.setText(f"Computing similarity at ({y}, {x})…")
        self._pick_btn.setEnabled(False)

        self._sweep_worker = _SweepWorker(
            state_dict=self._trained_state,
            in_channels=in_ch,
            frame=frame,
            anchor_y=y,
            anchor_x=x,
            patch_size=self._patch_spin.value(),
            stride=self._stride_spin.value(),
            device_str=self._device_combo.currentText(),
        )
        self._sweep_worker._frame_index = t
        self._sweep_worker.finished.connect(self._on_sweep_done)
        self._sweep_worker.error.connect(self._on_sweep_error)
        self._sweep_worker.start()

    def _on_sweep_done(self, sim_map: np.ndarray):
        self._pick_btn.setEnabled(True)
        self._pick_info.setText("Done. Click another patch or toggle off.")

        threshold = self._thresh_dspin.value()
        mask = (sim_map >= threshold).astype(np.uint8)

        img_layer = self._selected_image()
        t = getattr(self._sweep_worker, "_frame_index", None)

        if img_layer is not None and np.asarray(img_layer.data).ndim == 4 and t is not None:
            T = np.asarray(img_layer.data).shape[0]
            full = np.zeros((T,) + mask.shape, dtype=np.uint8)
            full[t] = mask
            mask_out = full
        else:
            mask_out = mask

        try:
            existing = self._viewer.layers[SIMILARITY_LAYER_NAME]
            if existing.data.shape == mask_out.shape:
                existing.data = mask_out
            else:
                self._viewer.layers.remove(SIMILARITY_LAYER_NAME)
                raise KeyError
            existing.refresh()
        except KeyError:
            from napari.utils.colormaps import direct_colormap
            cmap = direct_colormap({
                0: np.array([0, 0, 0, 0], dtype=np.float32),
                1: np.array([0.0, 1.0, 0.5, 0.6], dtype=np.float32),
                None: np.array([0, 0, 0, 0], dtype=np.float32),
            })
            self._viewer.add_labels(
                mask_out, name=SIMILARITY_LAYER_NAME, opacity=0.5,
                colormap=cmap,
            )

    def _on_sweep_error(self, msg: str):
        self._pick_btn.setEnabled(True)
        self._pick_info.setText("")
        self._log.append(f"Sweep error: {msg}")
