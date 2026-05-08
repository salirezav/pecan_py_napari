"""Mask Retouching dock widget: morphological cleanup for Labels layers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .logic import apply_retouching_pipeline
from ..pipeline_recorder.state import upsert_pipeline_step


class MaskRetouchingWidget(QWidget):
    """Controls for morphological mask cleanup applied live to all frames."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._original_data: np.ndarray | None = None
        self._observed_layer: Labels | None = None
        self._is_applying_pipeline = False
        self._building_ui = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # ---- Layer selector ------------------------------------------------
        layer_group = QGroupBox("Mask layer")
        layer_layout = QVBoxLayout(layer_group)
        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self._layer_combo)
        layout.addWidget(layer_group)

        # ---- Morphological operations --------------------------------------
        morph_group = QGroupBox("Morphological operations")
        morph_layout = QVBoxLayout(morph_group)

        self._close_spin = self._add_spin_row(morph_layout, "Close kernel:", 0, 0, 99, 2,
                                               "Fills small gaps (dilate then erode). 0 = off.")
        self._open_spin = self._add_spin_row(morph_layout, "Open kernel:", 0, 0, 99, 2,
                                              "Removes small noise (erode then dilate). 0 = off.")
        self._dilate_spin = self._add_spin_row(morph_layout, "Dilate kernel:", 0, 0, 99, 2,
                                                "Expand mask regions. 0 = off.")
        self._dilate_iter_spin = self._add_spin_row(morph_layout, "Dilate iters:", 1, 1, 20, 1)
        self._erode_spin = self._add_spin_row(morph_layout, "Erode kernel:", 0, 0, 99, 2,
                                               "Shrink mask regions. 0 = off.")
        self._erode_iter_spin = self._add_spin_row(morph_layout, "Erode iters:", 1, 1, 20, 1)
        layout.addWidget(morph_group)

        # ---- Filtering & cleanup -------------------------------------------
        filter_group = QGroupBox("Filtering && cleanup")
        filter_layout = QVBoxLayout(filter_group)

        self._min_area_spin = self._add_spin_row(filter_layout, "Min area (px):", 0, 0, 999999, 10,
                                                  "Remove regions smaller than this. 0 = off.")
        self._fill_holes_cb = QCheckBox("Fill internal holes")
        self._fill_holes_cb.stateChanged.connect(lambda _: self._schedule_update())
        filter_layout.addWidget(self._fill_holes_cb)

        self._keep_largest_cb = QCheckBox("Keep largest contour only")
        self._keep_largest_cb.stateChanged.connect(lambda _: self._schedule_update())
        filter_layout.addWidget(self._keep_largest_cb)

        self._smooth_spin = self._add_spin_row(filter_layout, "Smooth kernel:", 0, 0, 99, 2,
                                                "Gaussian-smooth mask edges. 0 = off.")
        layout.addWidget(filter_group)

        # ---- Buttons -------------------------------------------------------
        btn_layout = QHBoxLayout()
        self._btn_reset = QPushButton("Reset to original")
        self._btn_reset.clicked.connect(self._reset_to_original)
        btn_layout.addWidget(self._btn_reset)
        layout.addLayout(btn_layout)

        # ---- Save masks ----------------------------------------------------
        save_group = QGroupBox("Save masks")
        save_lay = QHBoxLayout(save_group)
        self._save_fmt_combo = QComboBox()
        self._save_fmt_combo.addItem("TIFF (.tiff)", "tiff")
        self._save_fmt_combo.addItem("NumPy (.npy)", "npy")
        save_lay.addWidget(self._save_fmt_combo)
        self._btn_save_masks = QPushButton("Save masks")
        self._btn_save_masks.clicked.connect(self._save_masks)
        save_lay.addWidget(self._btn_save_masks)
        layout.addWidget(save_group)

        layout.addStretch(1)

        # ---- Debounce timer ------------------------------------------------
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(120)
        self._update_timer.timeout.connect(self._apply_pipeline)

        # ---- Events --------------------------------------------------------
        self._refresh_layer_list()
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)

    # ---- Helpers -----------------------------------------------------------

    def _add_spin_row(self, parent_layout, label: str, default: int,
                      lo: int, hi: int, step: int, tooltip: str = "") -> QSpinBox:
        row = QHBoxLayout()
        lbl = QLabel(label)
        spin = QSpinBox()
        spin.setRange(lo, hi)
        spin.setSingleStep(step)
        spin.setValue(default)
        if tooltip:
            spin.setToolTip(tooltip)
        spin.valueChanged.connect(lambda _: self._schedule_update())
        row.addWidget(lbl)
        row.addWidget(spin)
        parent_layout.addLayout(row)
        return spin

    # ---- Layer management --------------------------------------------------

    def _refresh_layer_list(self, _event=None):
        self._building_ui = True
        self._disconnect_layer_events()
        prev = self._get_current_layer()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                self._layer_combo.addItem(layer.name, layer)
        if prev is not None and prev in self._viewer.layers:
            idx = self._layer_combo.findData(prev)
            if idx >= 0:
                self._layer_combo.setCurrentIndex(idx)
        self._building_ui = False
        self._on_layer_changed(self._layer_combo.currentIndex())

    def _get_current_layer(self):
        data = self._layer_combo.currentData()
        if data is None:
            return None
        if data in self._viewer.layers:
            return data
        return None

    def _layer_volume_data(self, layer: Labels) -> np.ndarray:
        """Return concrete array for full layer data (2D or time stack)."""
        d = layer.data
        if getattr(layer, "multiscale", False):
            d = d[0]
        arr = np.asarray(d)
        if arr.dtype == object:
            if isinstance(d, (list, tuple)) and len(d) > 0:
                try:
                    arr = np.stack([np.asarray(x) for x in d], axis=0)
                except Exception:
                    arr = np.asarray(d[0])
            else:
                arr = np.asarray(d)
        return np.asarray(arr)

    def _on_layer_changed(self, _idx: int = 0):
        if self._building_ui:
            return
        self._disconnect_layer_events()
        layer = self._get_current_layer()
        if layer is None:
            self._original_data = None
            return
        self._original_data = self._layer_volume_data(layer).copy()
        self._connect_layer_events(layer)

    def _connect_layer_events(self, layer: Labels) -> None:
        self._observed_layer = layer
        try:
            layer.events.data.connect(self._on_layer_data_changed)
        except Exception:
            self._observed_layer = None

    def _disconnect_layer_events(self) -> None:
        if self._observed_layer is None:
            return
        try:
            self._observed_layer.events.data.disconnect(self._on_layer_data_changed)
        except Exception:
            pass
        self._observed_layer = None

    def _on_layer_data_changed(self, _event=None) -> None:
        # Keep baseline in sync with user/manual edits so they persist.
        # Ignore writes produced by this widget's own pipeline pass.
        if self._is_applying_pipeline:
            return
        layer = self._get_current_layer()
        if layer is None:
            self._original_data = None
            return
        self._original_data = self._layer_volume_data(layer).copy()

    # ---- Pipeline ----------------------------------------------------------

    def _schedule_update(self):
        if self._get_current_layer() is not None:
            self._update_timer.start()

    def _apply_pipeline(self):
        layer = self._get_current_layer()
        if layer is None or self._original_data is None:
            return

        params = dict(
            close_size=self._close_spin.value(),
            open_size=self._open_spin.value(),
            dilate_size=self._dilate_spin.value(),
            dilate_iter=self._dilate_iter_spin.value(),
            erode_size=self._erode_spin.value(),
            erode_iter=self._erode_iter_spin.value(),
            min_area=self._min_area_spin.value(),
            do_fill_holes=self._fill_holes_cb.isChecked(),
            do_keep_largest=self._keep_largest_cb.isChecked(),
            smooth_size=self._smooth_spin.value(),
        )

        src = self._original_data
        if src.ndim == 2:
            result = apply_retouching_pipeline(src, **params)
        else:
            frames = []
            for t in range(src.shape[0]):
                frames.append(apply_retouching_pipeline(src[t], **params))
            result = np.stack(frames, axis=0)

        self._is_applying_pipeline = True
        try:
            layer.data = result
            layer.refresh()
        finally:
            self._is_applying_pipeline = False

        rec_params = {
            "mask_layer": layer.name,
            "close_size": int(params["close_size"]),
            "open_size": int(params["open_size"]),
            "dilate_size": int(params["dilate_size"]),
            "dilate_iter": int(params["dilate_iter"]),
            "erode_size": int(params["erode_size"]),
            "erode_iter": int(params["erode_iter"]),
            "min_area": int(params["min_area"]),
            "do_fill_holes": bool(params["do_fill_holes"]),
            "do_keep_largest": bool(params["do_keep_largest"]),
            "smooth_size": int(params["smooth_size"]),
        }
        upsert_pipeline_step(
            kind="mask_retouching.apply",
            description=f"Mask Retouching on {layer.name}",
            params=rec_params,
            match=lambda st: (
                st.kind == "mask_retouching.apply"
                and str((st.params or {}).get("mask_layer", "")) == layer.name
            ),
        )

    def _reset_to_original(self):
        layer = self._get_current_layer()
        if layer is None or self._original_data is None:
            return
        layer.data = self._original_data.copy()
        layer.refresh()
        self._close_spin.setValue(0)
        self._open_spin.setValue(0)
        self._dilate_spin.setValue(0)
        self._dilate_iter_spin.setValue(1)
        self._erode_spin.setValue(0)
        self._erode_iter_spin.setValue(1)
        self._min_area_spin.setValue(0)
        self._fill_holes_cb.setChecked(False)
        self._keep_largest_cb.setChecked(False)
        self._smooth_spin.setValue(0)

    # ---- Save masks --------------------------------------------------------

    def _find_source_path(self) -> str | None:
        """Look through Image layers for a source_path in metadata."""
        for layer in self._viewer.layers:
            if isinstance(layer, Image):
                src = layer.metadata.get("source_path")
                if src:
                    return src
        return None

    def _save_masks(self):
        fmt = self._save_fmt_combo.currentData()
        ext = ".tiff" if fmt == "tiff" else ".npy"

        src_path = self._find_source_path()
        src_dir = str(Path(src_path).parent) if src_path else None

        layer = self._get_current_layer()
        if layer is None:
            from napari.utils.notifications import show_warning
            show_warning("Select a mask layer first.")
            return

        if src_dir:
            out_path = str(Path(src_dir) / (layer.name + ext))
        else:
            out_path, _ = QFileDialog.getSaveFileName(
                self, f"Save {layer.name}", layer.name + ext,
                "TIFF (*.tiff)" if fmt == "tiff" else "NumPy (*.npy)",
            )
            if not out_path:
                return

        data = np.asarray(layer.data)
        if fmt == "tiff":
            import tifffile
            tifffile.imwrite(out_path, data)
        else:
            np.save(out_path, data)

        from napari.utils.notifications import show_info
        show_info(f"Saved {layer.name} to {out_path}")
