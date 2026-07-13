"""Mask Retouching dock widget: morphological cleanup for Labels layers.

Live preview applies only to the currently displayed frame (debounced). Other
time points stay as copies of the original until visited or until you click
**Apply to all frames**. Full-volume apply runs on a background QThread with a
progress bar and temporarily disables this widget's controls.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from napari.layers import Image, Labels
from qtpy.QtCore import QThread, QTimer, Signal
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .logic import apply_retouching_pipeline, apply_retouching_to_volume
from ..pipeline_recorder.state import upsert_pipeline_step

_FRAME_DEBOUNCE_MS = 120

_STYLE_APPLY_ALL_NEUTRAL = ""
_STYLE_APPLY_ALL_PENDING = (
    "QPushButton { background-color: #2a6ad8; color: #ffffff; font-weight: bold; "
    "padding: 6px 10px; border-radius: 4px; border: 1px solid #1f5abe; }"
    "QPushButton:hover { background-color: #3a7aee; }"
    "QPushButton:pressed { background-color: #1f5abe; }"
    "QPushButton:disabled { background-color: #555555; color: #aaaaaa; border: 1px solid #444; }"
)


def _params_fingerprint(params: dict) -> str:
    def _json_default(obj: Any):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(f"Not JSON-serializable: {type(obj)!r}")

    return json.dumps(params, sort_keys=True, default=_json_default)


class _RetouchFrameWorker(QThread):
    finished = Signal(object)  # (job_id, frame_index, fp, adjusted_frame)
    error = Signal(str)

    def __init__(self, job_id: int, frame_index: int, fp: str, frame: np.ndarray, params: dict):
        super().__init__()
        self._job_id = int(job_id)
        self._frame_index = int(frame_index)
        self._fp = fp
        self._frame = np.asarray(frame)
        self._params = dict(params)

    def run(self):
        try:
            adjusted = apply_retouching_pipeline(self._frame, **self._params)
            self.finished.emit((self._job_id, self._frame_index, self._fp, np.asarray(adjusted)))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class _RetouchAllWorker(QThread):
    finished = Signal(object)  # (job_id, fp, adjusted_volume)
    error = Signal(str)
    progress = Signal(int, int)

    def __init__(self, job_id: int, fp: str, src: np.ndarray, params: dict):
        super().__init__()
        self._job_id = int(job_id)
        self._fp = fp
        self._src = np.asarray(src)
        self._params = dict(params)

    def run(self):
        try:
            adjusted = apply_retouching_to_volume(
                self._src,
                progress_callback=lambda c, t: self.progress.emit(int(c), int(t)),
                **self._params,
            )
            self.finished.emit((self._job_id, self._fp, np.asarray(adjusted)))
        except Exception as exc:
            import traceback

            self.error.emit(f"{exc}\n{traceback.format_exc()}")


class MaskRetouchingWidget(QWidget):
    """Controls for morphological mask cleanup with current-frame live preview."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._original_data: np.ndarray | None = None
        self._working_data: np.ndarray | None = None
        self._observed_layer: Labels | None = None
        self._is_applying_pipeline = False
        self._building_ui = False
        self._controls_enabled = True

        self._per_frame_fp: dict[int, str] = {}
        self._last_known_params_fp: str | None = None
        self._all_frames_synced_fp: str | None = None
        self._apply_job_id = 0
        self._apply_all_job_id = 0
        self._worker: _RetouchFrameWorker | None = None
        self._worker_all: _RetouchAllWorker | None = None

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

        self._close_spin = self._add_spin_row(
            morph_layout, "Close kernel:", 0, 0, 99, 2,
            "Fills small gaps (dilate then erode). 0 = off.",
        )
        self._open_spin = self._add_spin_row(
            morph_layout, "Open kernel:", 0, 0, 99, 2,
            "Removes small noise (erode then dilate). 0 = off.",
        )
        self._dilate_spin = self._add_spin_row(
            morph_layout, "Dilate kernel:", 0, 0, 99, 2,
            "Expand mask regions. 0 = off.",
        )
        self._dilate_iter_spin = self._add_spin_row(morph_layout, "Dilate iters:", 1, 1, 20, 1)
        self._erode_spin = self._add_spin_row(
            morph_layout, "Erode kernel:", 0, 0, 99, 2,
            "Shrink mask regions. 0 = off.",
        )
        self._erode_iter_spin = self._add_spin_row(morph_layout, "Erode iters:", 1, 1, 20, 1)
        layout.addWidget(morph_group)

        # ---- Filtering & cleanup -------------------------------------------
        filter_group = QGroupBox("Filtering && cleanup")
        filter_layout = QVBoxLayout(filter_group)

        self._min_area_spin = self._add_spin_row(
            filter_layout, "Min area (px):", 0, 0, 999999, 10,
            "Remove regions smaller than this. 0 = off.",
        )
        self._fill_holes_cb = QCheckBox("Fill internal holes")
        self._fill_holes_cb.setToolTip(
            "Fill enclosed background regions inside the mask. "
            "Use min/max area to limit which holes are filled."
        )
        self._fill_holes_cb.stateChanged.connect(self._on_fill_holes_toggled)
        filter_layout.addWidget(self._fill_holes_cb)

        self._fill_holes_min_spin = self._add_spin_row(
            filter_layout,
            "Hole min area (px):",
            0,
            0,
            999999,
            10,
            "Only fill holes at least this large. 0 = no lower bound.",
        )
        self._fill_holes_max_spin = self._add_spin_row(
            filter_layout,
            "Hole max area (px):",
            0,
            0,
            999999,
            10,
            "Only fill holes at most this large. 0 = no upper bound.",
        )
        self._on_fill_holes_toggled(self._fill_holes_cb.checkState())

        self._watershed_cb = QCheckBox("Split touching objects (watershed)")
        self._watershed_cb.setToolTip(
            "Distance-transform watershed: one label ID per object. "
            "Use after a binary pecan mask; tune min peak distance to object radius."
        )
        self._watershed_cb.stateChanged.connect(lambda _: self._schedule_update())
        filter_layout.addWidget(self._watershed_cb)

        self._watershed_dist_spin = self._add_spin_row(
            filter_layout,
            "Watershed min distance:",
            15,
            1,
            999,
            1,
            "Minimum spacing between object seeds (px). Larger = fewer splits.",
        )

        self._keep_largest_cb = QCheckBox("Keep largest contour only")
        self._keep_largest_cb.stateChanged.connect(lambda _: self._schedule_update())
        filter_layout.addWidget(self._keep_largest_cb)

        self._smooth_spin = self._add_spin_row(
            filter_layout, "Smooth kernel:", 0, 0, 99, 2,
            "Gaussian-smooth mask edges. 0 = off.",
        )
        layout.addWidget(filter_group)

        # ---- Apply / reset -------------------------------------------------
        btn_layout = QHBoxLayout()
        self._btn_reset = QPushButton("Reset to original")
        self._btn_reset.clicked.connect(self._reset_to_original)
        btn_layout.addWidget(self._btn_reset)
        layout.addLayout(btn_layout)

        apply_row = QVBoxLayout()
        apply_row.setContentsMargins(0, 4, 0, 0)
        self._btn_apply_all = QPushButton("Apply to all frames")
        self._btn_apply_all.setToolTip(
            "Bake the current settings into every time slice (runs in the background). "
            "The button turns blue when settings changed since the last full apply "
            "and only the current frame may be up to date."
        )
        self._btn_apply_all.clicked.connect(self._on_apply_all_clicked)
        apply_row.addWidget(self._btn_apply_all)
        self._apply_all_hint = QLabel(
            "Preview updates the current frame only. Blue button = settings changed "
            "since last full apply."
        )
        self._apply_all_hint.setWordWrap(True)
        self._apply_all_hint.setStyleSheet("color: #888888; font-size: 11px;")
        apply_row.addWidget(self._apply_all_hint)
        self._apply_all_progress = QProgressBar()
        self._apply_all_progress.setRange(0, 100)
        self._apply_all_progress.setValue(0)
        self._apply_all_progress.setTextVisible(True)
        self._apply_all_progress.hide()
        apply_row.addWidget(self._apply_all_progress)
        self._busy_label = QLabel("Processing…")
        self._busy_label.setStyleSheet("color: #2a6ad8; font-weight: bold;")
        self._busy_label.hide()
        apply_row.addWidget(self._busy_label)
        layout.addLayout(apply_row)

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

        self._interactive_widgets = [
            self._layer_combo,
            self._close_spin,
            self._open_spin,
            self._dilate_spin,
            self._dilate_iter_spin,
            self._erode_spin,
            self._erode_iter_spin,
            self._min_area_spin,
            self._fill_holes_cb,
            self._fill_holes_min_spin,
            self._fill_holes_max_spin,
            self._watershed_cb,
            self._watershed_dist_spin,
            self._keep_largest_cb,
            self._smooth_spin,
            self._btn_reset,
            self._btn_apply_all,
            self._save_fmt_combo,
            self._btn_save_masks,
        ]

        # ---- Debounce timer ------------------------------------------------
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(_FRAME_DEBOUNCE_MS)
        self._update_timer.timeout.connect(self._on_preview_debounce)

        # ---- Events --------------------------------------------------------
        self._refresh_layer_list()
        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._viewer.dims.events.current_step.connect(self._on_dims_changed)
        self._refresh_apply_all_button_appearance()

    # ---- Helpers -----------------------------------------------------------

    def _add_spin_row(
        self,
        parent_layout,
        label: str,
        default: int,
        lo: int,
        hi: int,
        step: int,
        tooltip: str = "",
    ) -> QSpinBox:
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

    def _on_fill_holes_toggled(self, _state=None) -> None:
        enabled = bool(self._fill_holes_cb.isChecked()) and self._controls_enabled
        self._fill_holes_min_spin.setEnabled(enabled)
        self._fill_holes_max_spin.setEnabled(enabled)
        self._schedule_update()

    def _current_params(self) -> dict:
        return dict(
            close_size=int(self._close_spin.value()),
            open_size=int(self._open_spin.value()),
            dilate_size=int(self._dilate_spin.value()),
            dilate_iter=int(self._dilate_iter_spin.value()),
            erode_size=int(self._erode_spin.value()),
            erode_iter=int(self._erode_iter_spin.value()),
            min_area=int(self._min_area_spin.value()),
            do_fill_holes=bool(self._fill_holes_cb.isChecked()),
            fill_holes_min_area=int(self._fill_holes_min_spin.value()),
            fill_holes_max_area=int(self._fill_holes_max_spin.value()),
            do_watershed_split=bool(self._watershed_cb.isChecked()),
            watershed_min_distance=int(self._watershed_dist_spin.value()),
            do_keep_largest=bool(self._keep_largest_cb.isChecked()),
            smooth_size=int(self._smooth_spin.value()),
        )

    def _current_params_fingerprint(self) -> str:
        return _params_fingerprint(self._current_params())

    def _is_apply_all_busy(self) -> bool:
        return self._worker_all is not None and self._worker_all.isRunning()

    def _set_controls_enabled(self, enabled: bool) -> None:
        self._controls_enabled = bool(enabled)
        for w in self._interactive_widgets:
            w.setEnabled(enabled)
        # Hole area spins stay gated by the fill-holes checkbox when idle.
        if enabled:
            hole_on = bool(self._fill_holes_cb.isChecked())
            self._fill_holes_min_spin.setEnabled(hole_on)
            self._fill_holes_max_spin.setEnabled(hole_on)
            self._refresh_apply_all_button_appearance()

    def _needs_apply_all_highlight(self) -> bool:
        if self._original_data is None:
            return False
        if self._original_data.ndim < 3:
            return False
        return self._all_frames_synced_fp != self._current_params_fingerprint()

    def _refresh_apply_all_button_appearance(self) -> None:
        if not hasattr(self, "_btn_apply_all"):
            return
        busy = self._is_apply_all_busy()
        has_volume = self._original_data is not None and self._original_data.ndim >= 3
        if self._original_data is None:
            self._btn_apply_all.setEnabled(False)
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_apply_all.setText("Apply to all frames")
            return
        if not has_volume:
            # 2-D mask: preview already is the full result.
            self._btn_apply_all.setEnabled(False)
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_apply_all.setText("Apply to all frames")
            return
        if not self._controls_enabled:
            self._btn_apply_all.setEnabled(False)
        else:
            self._btn_apply_all.setEnabled(not busy)
        if busy:
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_apply_all.setText("Applying to all frames…")
            return
        self._btn_apply_all.setText("Apply to all frames")
        if self._needs_apply_all_highlight():
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_PENDING)
        else:
            self._btn_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)

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
        self._cancel_workers()
        layer = self._get_current_layer()
        if layer is None:
            self._original_data = None
            self._working_data = None
            self._per_frame_fp.clear()
            self._last_known_params_fp = None
            self._all_frames_synced_fp = None
            self._refresh_apply_all_button_appearance()
            return
        self._original_data = self._layer_volume_data(layer).copy()
        self._working_data = self._original_data.copy()
        self._per_frame_fp.clear()
        self._last_known_params_fp = None
        self._all_frames_synced_fp = None
        self._connect_layer_events(layer)
        self._refresh_apply_all_button_appearance()
        self._schedule_update()

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
        if self._is_applying_pipeline or self._is_apply_all_busy():
            return
        layer = self._get_current_layer()
        if layer is None:
            self._original_data = None
            self._working_data = None
            return
        self._original_data = self._layer_volume_data(layer).copy()
        self._working_data = self._original_data.copy()
        self._per_frame_fp.clear()
        self._last_known_params_fp = None
        self._all_frames_synced_fp = None
        self._refresh_apply_all_button_appearance()

    def _cancel_workers(self) -> None:
        self._apply_job_id += 1
        self._apply_all_job_id += 1
        self._worker = None
        self._worker_all = None
        self._apply_all_progress.hide()
        self._busy_label.hide()
        self._set_controls_enabled(True)

    # ---- Time index / buffers ----------------------------------------------

    def _current_time_index(self) -> int:
        if self._original_data is None:
            return 0
        if self._original_data.ndim != 3:
            return 0
        try:
            t = int(self._viewer.dims.current_step[0])
        except (IndexError, TypeError, ValueError):
            t = 0
        return int(np.clip(t, 0, int(self._original_data.shape[0]) - 1))

    def _on_dims_changed(self, _event=None) -> None:
        if self._original_data is None or self._is_apply_all_busy():
            return
        self._seed_working_frame_from_source(self._current_time_index())
        self._schedule_update()

    def _seed_working_frame_from_source(self, t: int) -> None:
        if self._working_data is None or self._original_data is None:
            return
        if t in self._per_frame_fp:
            return
        if self._working_data.ndim == 2:
            return
        self._working_data[int(t)] = self._original_data[int(t)]

    def _invalidate_cached_slices(self) -> None:
        if self._working_data is None or self._original_data is None:
            self._per_frame_fp.clear()
            return
        for t in list(self._per_frame_fp.keys()):
            if self._working_data.ndim == 2:
                np.copyto(self._working_data, self._original_data)
            else:
                self._working_data[int(t)] = self._original_data[int(t)]
        self._per_frame_fp.clear()

    def _promote_working_dtype(self, sample: np.ndarray) -> None:
        if self._working_data is None:
            return
        needed = np.result_type(self._working_data.dtype, sample.dtype)
        if self._working_data.dtype != needed:
            self._working_data = self._working_data.astype(needed, copy=False)

    def _write_adjusted_frame(self, t: int, adjusted: np.ndarray) -> None:
        if self._working_data is None:
            return
        adj = np.asarray(adjusted)
        self._promote_working_dtype(adj)
        if self._working_data.ndim == 2:
            self._working_data[...] = adj
        else:
            self._working_data[int(t)] = adj

    def _commit_working_to_layer(self) -> None:
        layer = self._get_current_layer()
        if layer is None or self._working_data is None:
            return
        self._is_applying_pipeline = True
        try:
            layer.data = self._working_data
            layer.refresh()
        finally:
            self._is_applying_pipeline = False

    # ---- Preview (current frame) -------------------------------------------

    def _schedule_update(self):
        if self._building_ui:
            return
        if self._get_current_layer() is None or self._is_apply_all_busy():
            return
        self._update_timer.start()
        self._refresh_apply_all_button_appearance()

    def _on_preview_debounce(self) -> None:
        layer = self._get_current_layer()
        if layer is None or self._original_data is None or self._working_data is None:
            return
        if self._is_apply_all_busy():
            return

        params = self._current_params()
        fp = _params_fingerprint(params)

        if fp != self._last_known_params_fp:
            self._invalidate_cached_slices()
            self._last_known_params_fp = fp
            self._commit_working_to_layer()

        t = self._current_time_index()
        if self._per_frame_fp.get(t) == fp:
            self._commit_working_to_layer()
            self._refresh_apply_all_button_appearance()
            return

        if self._original_data.ndim == 2:
            frame = self._original_data
        else:
            frame = self._original_data[int(t)]

        self._apply_job_id += 1
        job_id = self._apply_job_id
        self._busy_label.setText("Updating current frame…")
        self._busy_label.show()
        self._worker = _RetouchFrameWorker(job_id, t, fp, frame, params)
        self._worker.finished.connect(self._on_frame_worker_finished)
        self._worker.error.connect(self._on_frame_worker_error)
        self._worker.start()
        self._record_pipeline_step(layer, params)
        self._refresh_apply_all_button_appearance()

    def _on_frame_worker_finished(self, payload: Any) -> None:
        job_id, t, fp, adjusted = payload
        if job_id != self._apply_job_id:
            return
        if self._is_apply_all_busy():
            return
        if _params_fingerprint(self._current_params()) != fp:
            return
        self._write_adjusted_frame(t, adjusted)
        self._per_frame_fp[int(t)] = fp
        self._commit_working_to_layer()
        if not self._is_apply_all_busy():
            self._busy_label.hide()
        self._refresh_apply_all_button_appearance()

    def _on_frame_worker_error(self, msg: str) -> None:
        from napari.utils.notifications import show_error

        show_error(f"Mask retouching error:\n{msg}")
        if not self._is_apply_all_busy():
            self._busy_label.hide()
        self._refresh_apply_all_button_appearance()

    # ---- Apply to all frames -----------------------------------------------

    def _on_apply_all_clicked(self) -> None:
        layer = self._get_current_layer()
        if layer is None or self._original_data is None:
            return
        if self._original_data.ndim < 3:
            return
        if self._is_apply_all_busy():
            return

        params = self._current_params()
        fp = _params_fingerprint(params)
        self._apply_all_job_id += 1
        job_id = self._apply_all_job_id
        # Invalidate in-flight single-frame jobs.
        self._apply_job_id += 1

        self._worker_all = _RetouchAllWorker(job_id, fp, self._original_data, params)
        self._worker_all.progress.connect(self._on_apply_all_progress)
        self._worker_all.finished.connect(self._on_apply_all_worker_finished)
        self._worker_all.error.connect(self._on_apply_all_worker_error)

        self._apply_all_progress.setValue(0)
        self._apply_all_progress.setFormat("0%")
        self._apply_all_progress.show()
        self._busy_label.setText("Applying to all frames…")
        self._busy_label.show()
        self._set_controls_enabled(False)
        self._refresh_apply_all_button_appearance()
        self._record_pipeline_step(layer, params)
        self._worker_all.start()

    def _on_apply_all_progress(self, current: int, total: int) -> None:
        total_safe = max(1, int(total))
        cur = max(0, min(int(current), total_safe))
        pct = int(round((cur / total_safe) * 100.0))
        self._apply_all_progress.setValue(pct)
        self._apply_all_progress.setFormat(f"{pct}% ({cur}/{total_safe})")

    def _on_apply_all_worker_finished(self, payload: Any) -> None:
        job_id, fp, adjusted = payload
        if job_id != self._apply_all_job_id:
            return
        if _params_fingerprint(self._current_params()) != fp:
            self._apply_all_progress.hide()
            self._busy_label.hide()
            self._worker_all = None
            self._set_controls_enabled(True)
            self._refresh_apply_all_button_appearance()
            return

        self._working_data = np.asarray(adjusted).copy()
        if self._working_data.ndim == 3:
            self._per_frame_fp = {t: fp for t in range(int(self._working_data.shape[0]))}
        else:
            self._per_frame_fp = {0: fp}
        self._last_known_params_fp = fp
        self._all_frames_synced_fp = fp

        self._apply_all_progress.hide()
        self._busy_label.hide()
        self._worker_all = None
        self._set_controls_enabled(True)
        self._refresh_apply_all_button_appearance()
        QTimer.singleShot(0, self._commit_working_to_layer)

    def _on_apply_all_worker_error(self, msg: str) -> None:
        from napari.utils.notifications import show_error

        show_error(f"Apply-to-all error:\n{msg}")
        self._apply_all_progress.hide()
        self._busy_label.hide()
        self._worker_all = None
        self._set_controls_enabled(True)
        self._refresh_apply_all_button_appearance()

    def _record_pipeline_step(self, layer: Labels, params: dict) -> None:
        rec_params = {"mask_layer": layer.name, **params}
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
        if self._is_apply_all_busy():
            return
        layer = self._get_current_layer()
        if layer is None or self._original_data is None:
            return
        self._apply_job_id += 1
        self._working_data = self._original_data.copy()
        self._per_frame_fp.clear()
        self._last_known_params_fp = None
        self._all_frames_synced_fp = None
        self._commit_working_to_layer()

        self._building_ui = True
        try:
            self._close_spin.setValue(0)
            self._open_spin.setValue(0)
            self._dilate_spin.setValue(0)
            self._dilate_iter_spin.setValue(1)
            self._erode_spin.setValue(0)
            self._erode_iter_spin.setValue(1)
            self._min_area_spin.setValue(0)
            self._fill_holes_cb.setChecked(False)
            self._fill_holes_min_spin.setValue(0)
            self._fill_holes_max_spin.setValue(0)
            self._watershed_cb.setChecked(False)
            self._watershed_dist_spin.setValue(15)
            self._keep_largest_cb.setChecked(False)
            self._smooth_spin.setValue(0)
        finally:
            self._building_ui = False
        self._refresh_apply_all_button_appearance()

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
        if self._is_apply_all_busy():
            return
        fmt = self._save_fmt_combo.currentData()
        ext = ".tiff" if fmt == "tiff" else ".npy"

        src_path = self._find_source_path()
        src_dir = str(Path(src_path).parent) if src_path else None

        layer = self._get_current_layer()
        if layer is None:
            from napari.utils.notifications import show_warning

            show_warning("Select a mask layer first.")
            return

        used_source_dir = False
        if src_dir:
            out_path = str(Path(src_dir) / (layer.name + ext))
            used_source_dir = True
        else:
            out_path, _ = QFileDialog.getSaveFileName(
                self,
                f"Save {layer.name}",
                layer.name + ext,
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

        # On manual save outside the source-video folder, pin the directory so
        # replays go to the same explicit location; otherwise default to the
        # current video's folder so pipelines stay portable across inputs.
        rec_output_dir = "" if used_source_dir else str(Path(out_path).parent)
        rec_params = {
            "mask_layer": layer.name,
            "format": str(fmt),
            "output_dir": rec_output_dir,
        }
        upsert_pipeline_step(
            kind="mask_retouching.save_masks",
            description=f"Save masks ({str(fmt).upper()}) for {layer.name}",
            params=rec_params,
            match=lambda st: (
                st.kind == "mask_retouching.save_masks"
                and str((st.params or {}).get("mask_layer", "")) == layer.name
            ),
        )

        from napari.utils.notifications import show_info

        show_info(f"Saved {layer.name} to {out_path}")
