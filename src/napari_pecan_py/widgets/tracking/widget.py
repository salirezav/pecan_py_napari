"""Tracking dock widget: stable IDs across frames for instance Labels."""

from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Labels
from qtpy.QtCore import QThread, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .logic import TrackingConfig, format_tracking_summary, track_label_volume
from ..pipeline_recorder.state import upsert_pipeline_step


class _TrackWorker(QThread):
    finished = Signal(object)  # TrackingResult
    error = Signal(str)
    progress = Signal(int, int)
    log = Signal(str)

    def __init__(self, volume: np.ndarray, config: TrackingConfig):
        super().__init__()
        self._volume = np.asarray(volume)
        self._config = config
        self._stop = False

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:
        try:
            self.log.emit("Tracking instances across frames…")
            result = track_label_volume(
                self._volume,
                self._config,
                progress_callback=lambda c, t: self.progress.emit(int(c), int(t)),
                cancel_callback=lambda: self._stop,
            )
            if self._stop:
                self.log.emit("Tracking stopped by user.")
            self.finished.emit(result)
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class TrackingWidget(QWidget):
    """Associate per-frame instance labels into stable track IDs (L→R conveyor)."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building_ui = False
        self._worker: _TrackWorker | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        help_lbl = QLabel(
            "Input: instance <b>Labels</b> (unique ID per object per frame, "
            "e.g. YOLO with Unique instance IDs). "
            "Output: remapped Labels where each pecan keeps one ID/color "
            "while it stays in view (left → right)."
        )
        help_lbl.setWordWrap(True)
        help_lbl.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(help_lbl)

        src = QGroupBox("Source instance labels")
        src_lay = QVBoxLayout(src)
        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        src_lay.addWidget(self._layer_combo)
        layout.addWidget(src)

        out = QGroupBox("Output")
        out_lay = QVBoxLayout(out)
        suffix_row = QHBoxLayout()
        suffix_row.addWidget(QLabel("Layer suffix:"))
        self._suffix_edit = QLineEdit(" - tracked")
        self._suffix_edit.setToolTip(
            "Output Labels layer name = source name + this suffix."
        )
        suffix_row.addWidget(self._suffix_edit, 1)
        out_lay.addLayout(suffix_row)
        self._overwrite_cb = QCheckBox("Overwrite existing output layer")
        self._overwrite_cb.setChecked(True)
        out_lay.addWidget(self._overwrite_cb)
        layout.addWidget(out)

        params = QGroupBox("Association (conveyor L→R)")
        params_lay = QVBoxLayout(params)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max match distance (px):"))
        self._max_dist = QDoubleSpinBox()
        self._max_dist.setRange(5.0, 2000.0)
        self._max_dist.setValue(80.0)
        self._max_dist.setDecimals(1)
        self._max_dist.setToolTip(
            "Maximum distance between a track's predicted center and a detection."
        )
        row.addWidget(self._max_dist, 1)
        params_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("IoU weight:"))
        self._iou_weight = QDoubleSpinBox()
        self._iou_weight.setRange(0.0, 2.0)
        self._iou_weight.setSingleStep(0.1)
        self._iou_weight.setValue(0.5)
        self._iou_weight.setToolTip(
            "How much bbox overlap contributes to matching (0 = distance only)."
        )
        row.addWidget(self._iou_weight, 1)
        params_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Max age (frames):"))
        self._max_age = QSpinBox()
        self._max_age.setRange(1, 60)
        self._max_age.setValue(5)
        self._max_age.setToolTip(
            "Keep an unmatched track this many frames before retiring it."
        )
        row.addWidget(self._max_age, 1)
        params_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Min area (px²):"))
        self._min_area = QDoubleSpinBox()
        self._min_area.setRange(1.0, 1e6)
        self._min_area.setValue(20.0)
        self._min_area.setDecimals(0)
        row.addWidget(self._min_area, 1)
        params_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Entry margin (left frac):"))
        self._entry_frac = QDoubleSpinBox()
        self._entry_frac.setRange(0.05, 0.9)
        self._entry_frac.setSingleStep(0.05)
        self._entry_frac.setValue(0.25)
        self._entry_frac.setToolTip(
            "Left fraction of the frame preferred for new track births."
        )
        row.addWidget(self._entry_frac, 1)
        params_lay.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Exit margin (right frac):"))
        self._exit_frac = QDoubleSpinBox()
        self._exit_frac.setRange(0.05, 0.9)
        self._exit_frac.setSingleStep(0.05)
        self._exit_frac.setValue(0.15)
        self._exit_frac.setToolTip(
            "Lost tracks near the right edge are retired immediately."
        )
        row.addWidget(self._exit_frac, 1)
        params_lay.addLayout(row)

        self._birth_anywhere_cb = QCheckBox(
            "Allow new tracks anywhere (not only left edge)"
        )
        self._birth_anywhere_cb.setChecked(True)
        self._birth_anywhere_cb.setToolTip(
            "On for typical YOLO videos. Off: only mint IDs in the left entry margin."
        )
        params_lay.addWidget(self._birth_anywhere_cb)
        layout.addWidget(params)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Track instances")
        self._run_btn.clicked.connect(self._start_tracking)
        btn_row.addWidget(self._run_btn)
        self._stop_btn = QPushButton("Stop")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._stop_tracking)
        btn_row.addWidget(self._stop_btn)
        layout.addLayout(btn_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        layout.addWidget(self._progress)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        layout.addStretch(1)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._refresh_layer_list()

    def _refresh_layer_list(self, *_args) -> None:
        self._building_ui = True
        prev = self._layer_combo.currentData()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if not isinstance(layer, Labels):
                continue
            try:
                vol = self._layer_volume_data(layer)
                if vol.ndim >= 2:
                    self._layer_combo.addItem(layer.name, layer)
            except Exception:
                continue
        if prev is not None and prev in self._viewer.layers:
            idx = self._layer_combo.findData(prev)
            if idx >= 0:
                self._layer_combo.setCurrentIndex(idx)
        self._building_ui = False

    def _on_layer_changed(self) -> None:
        if self._building_ui:
            return
        self._status.clear()

    def _selected_layer(self) -> Labels | None:
        data = self._layer_combo.currentData()
        if data is None or data not in self._viewer.layers:
            return None
        if not isinstance(data, Labels):
            return None
        return data

    def _layer_volume_data(self, layer: Labels) -> np.ndarray:
        d = layer.data
        if getattr(layer, "multiscale", False):
            d = d[0]
        arr = np.asarray(d)
        if arr.dtype == object:
            if isinstance(d, (list, tuple)) and len(d) > 0:
                arr = np.stack([np.asarray(x) for x in d], axis=0)
            else:
                arr = np.asarray(layer.data[0])
        return arr

    def _collect_config(self) -> TrackingConfig:
        return TrackingConfig(
            max_match_distance=float(self._max_dist.value()),
            iou_weight=float(self._iou_weight.value()),
            max_age=int(self._max_age.value()),
            min_area=float(self._min_area.value()),
            entry_margin_frac=float(self._entry_frac.value()),
            exit_margin_frac=float(self._exit_frac.value()),
            allow_birth_anywhere=bool(self._birth_anywhere_cb.isChecked()),
        )

    def _output_name(self, source: Labels) -> str:
        suffix = self._suffix_edit.text()
        if not suffix:
            suffix = " - tracked"
        return f"{source.name}{suffix}"

    def _set_running(self, running: bool) -> None:
        self._run_btn.setEnabled(not running)
        self._stop_btn.setEnabled(running)
        self._layer_combo.setEnabled(not running)

    def _start_tracking(self) -> None:
        from napari.utils.notifications import show_warning

        layer = self._selected_layer()
        if layer is None:
            show_warning("Select an instance Labels layer to track.")
            return
        vol = self._layer_volume_data(layer)
        if vol.ndim == 2:
            # Still allow single-frame renumber; tracking is a no-op associator.
            pass
        elif vol.ndim != 3:
            show_warning(f"Unsupported Labels shape {vol.shape}; need (T,H,W) or (H,W).")
            return
        if not np.any(vol > 0):
            show_warning("Selected Labels layer has no foreground pixels.")
            return

        self._set_running(True)
        self._progress.setValue(0)
        self._status.setText("Running…")
        self._worker = _TrackWorker(vol, self._collect_config())
        self._worker.progress.connect(self._on_progress)
        self._worker.log.connect(self._status.setText)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()

    def _stop_tracking(self) -> None:
        if self._worker is not None:
            self._worker.stop()
            self._status.setText("Stop requested…")

    def _on_progress(self, cur: int, tot: int) -> None:
        if tot > 0:
            self._progress.setValue(int(100 * cur / tot))

    def _on_finished(self, result: Any) -> None:
        from napari.utils.notifications import show_info

        self._set_running(False)
        self._progress.setValue(100)
        layer = self._selected_layer()
        if layer is None:
            self._status.setText("Source layer disappeared; result discarded.")
            return

        out_name = self._output_name(layer)
        labels = np.asarray(result.labels)
        try:
            existing = self._viewer.layers[out_name]
        except Exception:
            existing = None

        if existing is not None and self._overwrite_cb.isChecked():
            if tuple(existing.data.shape) != tuple(labels.shape):
                self._viewer.layers.remove(existing)
                self._viewer.add_labels(labels, name=out_name, opacity=0.5)
            else:
                existing.data = labels
                existing.refresh()
        else:
            # Avoid clobbering an existing layer when overwrite is off.
            name = out_name
            if existing is not None and not self._overwrite_cb.isChecked():
                i = 2
                while name in self._viewer.layers:
                    name = f"{out_name} ({i})"
                    i += 1
            self._viewer.add_labels(labels, name=name, opacity=0.5)
            out_name = name

        summary = format_tracking_summary(result)
        self._status.setText(summary)
        show_info(f"Tracking complete: {summary}")

        upsert_pipeline_step(
            kind="tracking.apply",
            description=f"Track instances on {layer.name}",
            params={
                "source_layer": layer.name,
                "output_layer": out_name,
                "max_match_distance": float(self._max_dist.value()),
                "iou_weight": float(self._iou_weight.value()),
                "max_age": int(self._max_age.value()),
                "min_area": float(self._min_area.value()),
                "entry_margin_frac": float(self._entry_frac.value()),
                "exit_margin_frac": float(self._exit_frac.value()),
                "allow_birth_anywhere": bool(self._birth_anywhere_cb.isChecked()),
            },
            match=lambda st, ln=layer.name: (
                st.kind == "tracking.apply"
                and str((st.params or {}).get("source_layer", "")) == ln
            ),
        )

    def _on_error(self, msg: str) -> None:
        from napari.utils.notifications import show_error

        self._set_running(False)
        self._progress.setValue(0)
        self._status.setText(f"ERROR: {msg}")
        show_error(f"Tracking error:\n{msg}")
