"""Mask Ops widget: clip masks by ellipse and run binary mask operations."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from napari.layers import Image, Labels, Shapes
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .logic import (
    apply_binary_operation,
    apply_binary_operation_bool,
    build_roi_masks_for_volume,
    clip_mask_label_volume,
    clip_mask_outside_roi,
    detect_parallel_edge_bands_volume,
    expand_mask_to_layer_shape,
    label_only_volume,
    labels_from_bool_mask,
    mask_volume_for_label,
    merge_label_mask_into_volume,
    new_labels_from_binary,
    positive_label_values,
    raster_spatial_shape,
)
from ..pipeline_recorder.state import record_pipeline_step

_STYLE_APPLY_ALL_NEUTRAL = ""
_STYLE_APPLY_ALL_PENDING = (
    "QPushButton { background-color: #2a6ad8; color: #ffffff; font-weight: bold; "
    "padding: 6px 10px; border-radius: 4px; border: 1px solid #1f5abe; }"
    "QPushButton:hover { background-color: #3a7aee; }"
    "QPushButton:pressed { background-color: #1f5abe; }"
    "QPushButton:disabled { background-color: #555555; color: #aaaaaa; border: 1px solid #444; }"
)


def _section_label_with_help(title: str, tooltip: str) -> QWidget:
    """Section title with a (?) icon; full help text appears on hover."""
    container = QWidget()
    row = QHBoxLayout(container)
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(4)
    row.addWidget(QLabel(title))
    help_lbl = QLabel("?")
    help_lbl.setToolTip(tooltip)
    help_lbl.setFixedSize(16, 16)
    help_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
    help_lbl.setCursor(Qt.CursorShape.WhatsThisCursor)
    help_lbl.setStyleSheet(
        "QLabel { color: #aaaaaa; font-weight: bold; font-size: 10px; "
        "border: 1px solid #666666; border-radius: 8px; background: #2a2a2a; }"
    )
    row.addWidget(help_lbl)
    row.addStretch(1)
    return container


class CollapsibleGroupBox(QGroupBox):
    """QGroupBox frame and title stay visible; body below the toggle can collapse."""

    def __init__(self, title: str, *, expanded: bool = True, parent: QWidget | None = None):
        super().__init__(title, parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(4)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        self._toggle = QToolButton()
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonIconOnly)
        self._toggle.setFixedSize(18, 18)
        self._toggle.setToolTip("Expand or collapse this section")
        self._toggle.toggled.connect(self._on_toggle)
        header.addWidget(self._toggle)
        header.addStretch(1)
        outer.addLayout(header)

        self._body = QWidget()
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(0, 0, 0, 0)
        self._body_layout.setSpacing(4)
        self._body.setVisible(expanded)
        outer.addWidget(self._body)
        self._set_arrow(expanded)

    def _set_arrow(self, expanded: bool) -> None:
        self._toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)

    def _on_toggle(self, expanded: bool) -> None:
        self._body.setVisible(expanded)
        self._set_arrow(expanded)

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout


class MaskOpsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building_ui = False

        # Parallel-edge band preview state
        self._pband_output_name: str | None = None
        self._pband_output_data: np.ndarray | None = None
        self._pband_per_frame_fp: dict[int, str] = {}
        self._pband_last_params_fp: str | None = None
        self._pband_all_frames_synced_fp: str | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -------- Clip by shape ROI ---------------------------------------
        g_clip = CollapsibleGroupBox("Clip mask by shape (ROI)", expanded=True)
        l_clip = g_clip.body_layout()
        l_clip.addWidget(
            _section_label_with_help(
                "Shapes layer (rectangle / polygon / ellipse)",
                "Draw a rectangle (or polygon/ellipse) covering the region to keep. "
                "Pixels outside the shape are cleared on every frame. A 2D shape "
                "applies to the whole video. Works on Labels masks and Image/video "
                "rasters (RGB outside the ROI is set to black).",
            )
        )
        self._roi_shapes_combo = QComboBox()
        self._roi_shapes_combo.addItem("(none)", None)
        l_clip.addWidget(self._roi_shapes_combo)

        row_new_roi = QHBoxLayout()
        self._btn_new_roi = QPushButton("New rectangle layer")
        self._btn_new_roi.setToolTip(
            "Add a 2D Shapes layer (no time axis) and switch to rectangle mode. "
            "One rectangle is applied to every video frame when clipping."
        )
        self._btn_new_roi.clicked.connect(self._on_new_roi_shapes_layer)
        row_new_roi.addWidget(self._btn_new_roi)
        l_clip.addLayout(row_new_roi)

        self._clip_all_frames_cb = QCheckBox("Apply shape to all frames")
        self._clip_all_frames_cb.setChecked(True)
        self._clip_all_frames_cb.setToolTip(
            "On: use only the spatial (y, x) part of the shape and clip every "
            "frame the same way (typical for one keep-inside rectangle). "
            "Off: 3D shapes stay on their own frame only (per-frame ellipses)."
        )
        l_clip.addWidget(self._clip_all_frames_cb)

        l_clip.addWidget(QLabel("Source layer (Labels or Image)"))
        self._clip_mask_combo = QComboBox()
        self._clip_mask_combo.addItem("(none)", None)
        self._clip_mask_combo.currentIndexChanged.connect(self._on_clip_mask_changed)
        l_clip.addWidget(self._clip_mask_combo)

        row_clip_lbl = QHBoxLayout()
        self._clip_label_caption = QLabel("Target label:")
        row_clip_lbl.addWidget(self._clip_label_caption)
        self._clip_label_combo = QComboBox()
        row_clip_lbl.addWidget(self._clip_label_combo, 1)
        l_clip.addLayout(row_clip_lbl)
        self._clip_label_caption.setVisible(False)
        self._clip_label_combo.setVisible(False)

        row_clip_out = QHBoxLayout()
        row_clip_out.addWidget(QLabel("Output:"))
        self._clip_target_combo = QComboBox()
        self._clip_target_combo.addItem("New layer", "new")
        self._clip_target_combo.addItem("Overwrite source layer", "overwrite")
        row_clip_out.addWidget(self._clip_target_combo, 1)
        l_clip.addLayout(row_clip_out)

        self._btn_clip = QPushButton("Apply clip (keep inside shape)")
        self._btn_clip.setToolTip(
            "Zero pixels outside the selected shape(s) on every frame "
            "(Labels → 0, Image/video → black)."
        )
        self._btn_clip.clicked.connect(self._on_apply_clip)
        l_clip.addWidget(self._btn_clip)
        layout.addWidget(g_clip)

        # -------- Binary ops ----------------------------------------------
        g_bin = CollapsibleGroupBox("Binary operations", expanded=True)
        l_bin = g_bin.body_layout()
        l_bin.addWidget(QLabel("Mask A (Labels or Image)"))
        self._a_combo = QComboBox()
        self._a_combo.addItem("(none)", None)
        self._a_combo.currentIndexChanged.connect(self._on_a_mask_changed)
        l_bin.addWidget(self._a_combo)

        row_a_lbl = QHBoxLayout()
        self._a_label_caption = QLabel("A target label:")
        row_a_lbl.addWidget(self._a_label_caption)
        self._a_label_combo = QComboBox()
        row_a_lbl.addWidget(self._a_label_combo, 1)
        l_bin.addLayout(row_a_lbl)
        self._a_label_caption.setVisible(False)
        self._a_label_combo.setVisible(False)

        l_bin.addWidget(QLabel("Mask B (Labels or Image)"))
        self._b_combo = QComboBox()
        self._b_combo.addItem("(none)", None)
        self._b_combo.currentIndexChanged.connect(self._on_b_mask_changed)
        l_bin.addWidget(self._b_combo)

        row_b_lbl = QHBoxLayout()
        self._b_label_caption = QLabel("B target label:")
        row_b_lbl.addWidget(self._b_label_caption)
        self._b_label_combo = QComboBox()
        row_b_lbl.addWidget(self._b_label_combo, 1)
        l_bin.addLayout(row_b_lbl)
        self._b_label_caption.setVisible(False)
        self._b_label_combo.setVisible(False)

        row_op = QHBoxLayout()
        row_op.addWidget(QLabel("Operation:"))
        self._op_combo = QComboBox()
        for label, value in (
            ("AND", "and"),
            ("OR", "or"),
            ("XOR", "xor"),
            ("NOT", "not"),
            ("A-B", "a-b"),
            ("B-A", "b-a"),
            ("A if B", "a-if-b"),
        ):
            self._op_combo.addItem(label, value)
        row_op.addWidget(self._op_combo, 1)
        l_bin.addLayout(row_op)

        row_target = QHBoxLayout()
        row_target.addWidget(QLabel("Apply result to:"))
        self._bin_target_combo = QComboBox()
        self._bin_target_combo.addItem("New layer", "new")
        self._bin_target_combo.addItem("Overwrite A (same label)", "a")
        self._bin_target_combo.addItem("Overwrite B (same label)", "b")
        row_target.addWidget(self._bin_target_combo, 1)
        l_bin.addLayout(row_target)

        self._btn_bin = QPushButton("Apply binary operation")
        self._btn_bin.clicked.connect(self._on_apply_binary)
        l_bin.addWidget(self._btn_bin)
        layout.addWidget(g_bin)

        # -------- Parallel edge bands -------------------------------------
        g_pband = CollapsibleGroupBox("Crack thickness from parallel edges", expanded=False)
        l_pband = g_pband.body_layout()
        l_pband.addWidget(
            _section_label_with_help(
                "Edge layer (Image)",
                "Detect close approximately parallel edge rails and fill the strip "
                "between them. Useful for shell-thickness segmentation at cracks.",
            )
        )
        self._pband_edge_combo = QComboBox()
        self._pband_edge_combo.addItem("(none)", None)
        l_pband.addWidget(self._pband_edge_combo)

        l_pband.addWidget(QLabel("Optional limit mask (Labels)"))
        self._pband_limit_combo = QComboBox()
        self._pband_limit_combo.addItem("(none)", None)
        self._pband_limit_combo.currentIndexChanged.connect(self._on_pband_limit_changed)
        l_pband.addWidget(self._pband_limit_combo)

        row_pband_lbl = QHBoxLayout()
        self._pband_limit_label_caption = QLabel("Limit target label:")
        row_pband_lbl.addWidget(self._pband_limit_label_caption)
        self._pband_limit_label_combo = QComboBox()
        row_pband_lbl.addWidget(self._pband_limit_label_combo, 1)
        l_pband.addLayout(row_pband_lbl)
        self._pband_limit_label_caption.setVisible(False)
        self._pband_limit_label_combo.setVisible(False)

        row_et = QHBoxLayout()
        row_et.addWidget(QLabel("Edge threshold:"))
        self._pband_thr_spin = QSpinBox()
        self._pband_thr_spin.setRange(1, 255)
        self._pband_thr_spin.setValue(1)
        row_et.addWidget(self._pband_thr_spin, 1)
        l_pband.addLayout(row_et)

        row_pc = QHBoxLayout()
        row_pc.addWidget(QLabel("Pre-close gaps (px):"))
        self._pband_preclose_spin = QSpinBox()
        self._pband_preclose_spin.setRange(0, 31)
        self._pband_preclose_spin.setSingleStep(2)
        self._pband_preclose_spin.setValue(3)
        row_pc.addWidget(self._pband_preclose_spin, 1)
        l_pband.addLayout(row_pc)

        row_dist = QHBoxLayout()
        row_dist.addWidget(QLabel("Distance min/max (px):"))
        self._pband_min_dist_spin = QSpinBox()
        self._pband_min_dist_spin.setRange(1, 50)
        self._pband_min_dist_spin.setValue(2)
        self._pband_max_dist_spin = QSpinBox()
        self._pband_max_dist_spin.setRange(2, 80)
        self._pband_max_dist_spin.setValue(12)
        row_dist.addWidget(self._pband_min_dist_spin)
        row_dist.addWidget(self._pband_max_dist_spin, 1)
        l_pband.addLayout(row_dist)

        row_ang = QHBoxLayout()
        row_ang.addWidget(QLabel("Angle tolerance (deg):"))
        self._pband_ang_tol_spin = QSpinBox()
        self._pband_ang_tol_spin.setRange(1, 90)
        self._pband_ang_tol_spin.setValue(25)
        row_ang.addWidget(self._pband_ang_tol_spin, 1)
        l_pband.addLayout(row_ang)

        row_len = QHBoxLayout()
        row_len.addWidget(QLabel("Min edge component (px):"))
        self._pband_min_comp_spin = QSpinBox()
        self._pband_min_comp_spin.setRange(1, 1000)
        self._pband_min_comp_spin.setValue(20)
        row_len.addWidget(self._pband_min_comp_spin, 1)
        l_pband.addLayout(row_len)

        self._btn_pband_apply_all = QPushButton("Apply to all frames")
        self._btn_pband_apply_all.clicked.connect(self._on_apply_pband_all)
        l_pband.addWidget(self._btn_pband_apply_all)
        layout.addWidget(g_pband)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._pband_debounce = QTimer(self)
        self._pband_debounce.setSingleShot(True)
        self._pband_debounce.setInterval(300)
        self._pband_debounce.timeout.connect(self._on_pband_preview_debounce)

        self._op_combo.currentIndexChanged.connect(self._update_b_enabled)
        self._pband_edge_combo.currentIndexChanged.connect(self._on_pband_inputs_changed)
        self._pband_limit_combo.currentIndexChanged.connect(self._on_pband_inputs_changed)
        for w in (
            self._pband_thr_spin,
            self._pband_preclose_spin,
            self._pband_min_dist_spin,
            self._pband_max_dist_spin,
            self._pband_ang_tol_spin,
            self._pband_min_comp_spin,
        ):
            w.valueChanged.connect(self._on_pband_param_changed)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_lists)
        self._viewer.layers.events.removed.connect(self._refresh_layer_lists)
        self._viewer.dims.events.current_step.connect(self._on_dims_changed)
        self._refresh_layer_lists()
        self._update_b_enabled()
        self._refresh_pband_apply_all_button()

    # ------------------------------------------------------------------
    def _refresh_layer_lists(self, _event: Any = None) -> None:
        self._building_ui = True
        try:
            prev_roi = self._roi_shapes_combo.currentData()
            prev_clip_mask = self._clip_mask_combo.currentData()
            prev_a = self._a_combo.currentData()
            prev_b = self._b_combo.currentData()
            prev_pband_edge = self._pband_edge_combo.currentData()
            prev_pband_limit = self._pband_limit_combo.currentData()

            for cmb in (self._roi_shapes_combo, self._clip_mask_combo, self._a_combo, self._b_combo):
                cmb.clear()
                cmb.addItem("(none)", None)
            self._pband_edge_combo.clear()
            self._pband_edge_combo.addItem("(none)", None)
            self._pband_limit_combo.clear()
            self._pband_limit_combo.addItem("(none)", None)

            for layer in self._viewer.layers:
                if isinstance(layer, Shapes):
                    self._roi_shapes_combo.addItem(layer.name, layer)
                if isinstance(layer, Labels):
                    self._clip_mask_combo.addItem(layer.name, layer)
                    self._pband_limit_combo.addItem(layer.name, layer)
                if isinstance(layer, Image):
                    self._clip_mask_combo.addItem(layer.name, layer)
                    self._pband_edge_combo.addItem(layer.name, layer)
                if isinstance(layer, (Labels, Image)):
                    self._a_combo.addItem(layer.name, layer)
                    self._b_combo.addItem(layer.name, layer)

            self._restore_combo(self._roi_shapes_combo, prev_roi)
            self._restore_combo(self._clip_mask_combo, prev_clip_mask)
            self._restore_combo(self._a_combo, prev_a)
            self._restore_combo(self._b_combo, prev_b)
            self._restore_combo(self._pband_edge_combo, prev_pband_edge)
            self._restore_combo(self._pband_limit_combo, prev_pband_limit)
            self._refresh_label_combo_for_layer(self._clip_mask_combo, self._clip_label_combo, self._clip_label_caption)
            self._refresh_label_combo_for_layer(self._a_combo, self._a_label_combo, self._a_label_caption)
            self._refresh_label_combo_for_layer(self._b_combo, self._b_label_combo, self._b_label_caption)
            self._refresh_label_combo_for_layer(
                self._pband_limit_combo, self._pband_limit_label_combo, self._pband_limit_label_caption
            )
        finally:
            self._building_ui = False
        self._on_pband_inputs_changed()

    def _on_clip_mask_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._refresh_label_combo_for_layer(self._clip_mask_combo, self._clip_label_combo, self._clip_label_caption)

    def _on_a_mask_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._refresh_label_combo_for_layer(self._a_combo, self._a_label_combo, self._a_label_caption)

    def _on_b_mask_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._refresh_label_combo_for_layer(self._b_combo, self._b_label_combo, self._b_label_caption)

    def _on_pband_limit_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._refresh_label_combo_for_layer(
            self._pband_limit_combo, self._pband_limit_label_combo, self._pband_limit_label_caption
        )
        self._on_pband_inputs_changed()

    def _refresh_label_combo_for_layer(
        self,
        layer_combo: QComboBox,
        label_combo: QComboBox,
        caption: QLabel,
    ) -> None:
        layer = layer_combo.currentData()
        prev = label_combo.currentData()
        label_combo.blockSignals(True)
        label_combo.clear()
        if layer is None or not isinstance(layer, Labels):
            caption.setVisible(False)
            label_combo.setVisible(False)
            label_combo.blockSignals(False)
            return
        labels = positive_label_values(self._layer_data(layer))
        if len(labels) <= 1:
            caption.setVisible(False)
            label_combo.setVisible(False)
            if labels:
                label_combo.addItem(f"Label {labels[0]}", labels[0])
            else:
                label_combo.addItem("(no labels)", None)
            label_combo.blockSignals(False)
            return
        caption.setVisible(True)
        label_combo.setVisible(True)
        for lv in labels:
            label_combo.addItem(f"Label {lv}", lv)
        if prev in labels:
            idx = label_combo.findData(prev)
            if idx >= 0:
                label_combo.setCurrentIndex(idx)
        label_combo.blockSignals(False)

    def _resolved_label(self, layer, label_combo: QComboBox) -> int | None:
        if layer is None or not isinstance(layer, Labels):
            return None
        labels = positive_label_values(self._layer_data(layer))
        if not labels:
            return None
        if len(labels) == 1:
            return labels[0]
        return label_combo.currentData()

    def _restore_combo(self, combo: QComboBox, prev_layer) -> None:
        if prev_layer is None:
            return
        if prev_layer in self._viewer.layers:
            idx = combo.findData(prev_layer)
            if idx >= 0:
                combo.setCurrentIndex(idx)

    def _layer_data(self, layer) -> np.ndarray:
        d = layer.data
        if getattr(layer, "multiscale", False):
            d = d[0]
        return np.asarray(d)

    def _selected(self, combo: QComboBox):
        layer = combo.currentData()
        if layer is None:
            return None
        if layer in self._viewer.layers:
            return layer
        return None

    def _set_status(self, txt: str) -> None:
        self._status.setText(txt)

    def _update_b_enabled(self) -> None:
        op = str(self._op_combo.currentData())
        need_b = op != "not"
        self._b_combo.setEnabled(need_b)

    def _is_binary_mask_layer(self, layer) -> bool:
        return isinstance(layer, (Labels, Image))

    def _write_binary_result(self, layer, result: np.ndarray, template_raw: np.ndarray) -> None:
        layer.data = expand_mask_to_layer_shape(result, template_raw)
        layer.refresh()

    def _write_labels_result(
        self,
        layer: Labels,
        original_raw: np.ndarray,
        result_bool: np.ndarray,
        label_value: int,
    ) -> None:
        merged = merge_label_mask_into_volume(original_raw, result_bool, label_value)
        layer.data = merged
        layer.refresh()

    def _limit_mask_volume(self, limit_layer: Labels | None, label_combo: QComboBox) -> np.ndarray | None:
        if limit_layer is None:
            return None
        raw = self._layer_data(limit_layer)
        lv = self._resolved_label(limit_layer, label_combo)
        if lv is None:
            return mask_volume_for_label(raw, None)
        return mask_volume_for_label(raw, lv)

    def _add_binary_output(self, name: str, result: np.ndarray, ref_layer, template_raw: np.ndarray) -> None:
        data = expand_mask_to_layer_shape(result, template_raw)
        if isinstance(ref_layer, Image):
            self._viewer.add_image(data, name=name)
        else:
            self._viewer.add_labels(data, name=name)

    # ------------------------------------------------------------------
    # Parallel-edge bands — preview / apply all
    # ------------------------------------------------------------------
    def _on_dims_changed(self, _event: Any = None) -> None:
        self._schedule_pband_preview()

    def _volume_time_index(self, shape: tuple[int, ...]) -> int:
        if len(shape) == 2:
            return 0
        if len(shape) == 3 and shape[-1] in (1, 2, 3, 4) and shape[-1] < shape[-2]:
            return 0
        if len(shape) in (3, 4):
            t_max = int(shape[0]) - 1
            try:
                t = int(self._viewer.dims.current_step[0])
            except Exception:
                t = 0
            return int(np.clip(t, 0, t_max))
        raise ValueError(f"Unsupported volume shape: {shape}")

    def _read_volume_slice(self, data: np.ndarray, t: int) -> np.ndarray:
        arr = np.asarray(data)
        if arr.ndim == 2:
            return arr
        if arr.ndim == 3:
            return arr[int(t)]
        raise ValueError(f"Unsupported array shape: {arr.shape}")

    def _pband_layers_ready(self) -> tuple[Image, Labels | None] | None:
        edge = self._selected(self._pband_edge_combo)
        limit = self._selected(self._pband_limit_combo)
        if edge is None or not isinstance(edge, Image):
            return None
        if limit is not None and not isinstance(limit, Labels):
            limit = None
        return edge, limit

    def _pband_params(self) -> dict[str, Any]:
        mn = int(self._pband_min_dist_spin.value())
        mx = int(self._pband_max_dist_spin.value())
        if mx <= mn:
            mx = mn + 1
            self._pband_max_dist_spin.setValue(mx)
        return {
            "edge_threshold": int(self._pband_thr_spin.value()),
            "pre_close_size": int(self._pband_preclose_spin.value()),
            "min_distance_px": mn,
            "max_distance_px": mx,
            "angle_tolerance_deg": int(self._pband_ang_tol_spin.value()),
            "min_component_px": int(self._pband_min_comp_spin.value()),
        }

    def _pband_params_fingerprint(self) -> str:
        layers = self._pband_layers_ready()
        edge_name = layers[0].name if layers else ""
        lim_name = layers[1].name if (layers and layers[1] is not None) else ""
        return json.dumps({"edge": edge_name, "limit": lim_name, **self._pband_params()}, sort_keys=True)

    def _needs_pband_apply_all_highlight(self) -> bool:
        if self._pband_layers_ready() is None:
            return False
        return self._pband_all_frames_synced_fp != self._pband_params_fingerprint()

    def _refresh_pband_apply_all_button(self) -> None:
        if not hasattr(self, "_btn_pband_apply_all"):
            return
        if self._pband_layers_ready() is None:
            self._btn_pband_apply_all.setEnabled(False)
            self._btn_pband_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)
            self._btn_pband_apply_all.setText("Apply to all frames")
            return
        self._btn_pband_apply_all.setEnabled(True)
        self._btn_pband_apply_all.setText("Apply to all frames")
        if self._needs_pband_apply_all_highlight():
            self._btn_pband_apply_all.setStyleSheet(_STYLE_APPLY_ALL_PENDING)
        else:
            self._btn_pband_apply_all.setStyleSheet(_STYLE_APPLY_ALL_NEUTRAL)

    def _on_pband_inputs_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._pband_output_data = None
        self._pband_per_frame_fp.clear()
        self._pband_last_params_fp = None
        self._pband_all_frames_synced_fp = None
        layers = self._pband_layers_ready()
        if layers is not None:
            edge_layer = layers[0]
            self._pband_output_name = f"{edge_layer.name} - parallel bands"
        else:
            self._pband_output_name = None
        self._schedule_pband_preview()
        self._refresh_pband_apply_all_button()

    def _on_pband_param_changed(self, *_args) -> None:
        if self._building_ui:
            return
        self._schedule_pband_preview()

    def _schedule_pband_preview(self) -> None:
        if self._pband_layers_ready() is None:
            return
        self._pband_debounce.start()
        self._refresh_pband_apply_all_button()

    def _ensure_pband_output_initialized(self, template: np.ndarray) -> bool:
        if self._pband_output_name is None:
            return False
        wanted_shape = tuple(int(x) for x in template.shape)
        if self._pband_output_data is None or tuple(self._pband_output_data.shape) != wanted_shape:
            self._pband_output_data = np.zeros(wanted_shape, dtype=template.dtype)
            self._pband_per_frame_fp.clear()
            self._pband_last_params_fp = None
        try:
            out_layer = self._viewer.layers[self._pband_output_name]
            if not isinstance(out_layer, Labels):
                raise KeyError("wrong type")
            if tuple(np.asarray(out_layer.data).shape) != wanted_shape:
                out_layer.data = self._pband_output_data
                out_layer.refresh()
        except Exception:
            self._viewer.add_labels(self._pband_output_data, name=self._pband_output_name)
        return True

    def _write_pband_preview_slice(self, t: int, slice_labels: np.ndarray) -> None:
        if self._pband_output_data is None:
            return
        if self._pband_output_data.ndim == 2:
            self._pband_output_data[...] = slice_labels
        else:
            self._pband_output_data[int(t)] = slice_labels

    def _push_pband_output_layer(self) -> None:
        if self._pband_output_name is None or self._pband_output_data is None:
            return
        try:
            out_layer = self._viewer.layers[self._pband_output_name]
            out_layer.data = self._pband_output_data
            out_layer.refresh()
        except Exception:
            self._viewer.add_labels(self._pband_output_data, name=self._pband_output_name)

    def _on_pband_preview_debounce(self) -> None:
        from napari.utils.notifications import show_warning

        layers = self._pband_layers_ready()
        if layers is None:
            return
        edge_layer, limit_layer = layers
        edges = self._layer_data(edge_layer)
        limit = self._limit_mask_volume(limit_layer, self._pband_limit_label_combo)

        fp = self._pband_params_fingerprint()
        if fp != self._pband_last_params_fp:
            self._pband_per_frame_fp.clear()
            self._pband_last_params_fp = fp

        if not self._ensure_pband_output_initialized(edges):
            return
        t = self._volume_time_index(tuple(edges.shape))
        if self._pband_per_frame_fp.get(int(t)) == fp:
            self._push_pband_output_layer()
            return

        try:
            params = self._pband_params()
            edge_slice = self._read_volume_slice(edges, t)
            lim_slice = None
            if limit is not None:
                lim_slice = self._read_volume_slice(limit, t)
            bands_bool = detect_parallel_edge_bands_volume(
                edge_slice,
                limit_mask=lim_slice,
                **params,
            )
            band_slice = labels_from_bool_mask(bands_bool, edge_slice)
            self._write_pband_preview_slice(t, band_slice)
            self._pband_per_frame_fp[int(t)] = fp
            self._push_pband_output_layer()
            if edges.ndim == 3:
                self._set_status(f"Parallel-band preview updated for frame {t}.")
            else:
                self._set_status("Parallel-band preview updated.")
        except Exception as exc:
            show_warning(f"Parallel-band preview failed: {exc}")
            self._set_status("")
        self._refresh_pband_apply_all_button()

    def _on_apply_pband_all(self) -> None:
        from napari.utils.notifications import show_warning

        layers = self._pband_layers_ready()
        if layers is None:
            show_warning("Select an edge Image layer.")
            return
        edge_layer, limit_layer = layers
        edges = self._layer_data(edge_layer)
        limit = self._limit_mask_volume(limit_layer, self._pband_limit_label_combo)
        params = self._pband_params()
        try:
            bands_bool = detect_parallel_edge_bands_volume(
                edges,
                limit_mask=limit,
                **params,
            )
            bands = labels_from_bool_mask(bands_bool, edges)
        except Exception as exc:
            show_warning(f"Parallel-band detection failed: {exc}")
            return

        out_name = f"{edge_layer.name} - parallel bands"
        self._pband_output_name = out_name
        self._pband_output_data = np.asarray(bands)
        try:
            layer = self._viewer.layers[out_name]
            layer.data = bands
            layer.refresh()
        except Exception:
            self._viewer.add_labels(bands, name=out_name)

        fp = self._pband_params_fingerprint()
        self._pband_per_frame_fp.clear()
        if edges.ndim == 3:
            for ti in range(int(edges.shape[0])):
                self._pband_per_frame_fp[int(ti)] = fp
        else:
            self._pband_per_frame_fp[0] = fp
        self._pband_last_params_fp = fp
        self._pband_all_frames_synced_fp = fp
        self._set_status(f"Applied parallel-band detection -> {out_name}.")
        self._refresh_pband_apply_all_button()

        limit_label = (
            self._resolved_label(limit_layer, self._pband_limit_label_combo)
            if limit_layer is not None
            else None
        )
        record_pipeline_step(
            "mask_ops.operation",
            f"Mask Ops parallel bands from {edge_layer.name}",
            {
                "mode": "parallel_bands",
                "edge_layer": edge_layer.name,
                "limit_mask_layer": limit_layer.name if limit_layer is not None else "",
                "limit_mask_label": limit_label if limit_label is not None else "",
                "edge_threshold": params["edge_threshold"],
                "pre_close_size": params["pre_close_size"],
                "min_distance_px": params["min_distance_px"],
                "max_distance_px": params["max_distance_px"],
                "angle_tolerance_deg": params["angle_tolerance_deg"],
                "min_component_px": params["min_component_px"],
                "output_layer": out_name,
            },
        )

    # ------------------------------------------------------------------
    def _on_new_roi_shapes_layer(self) -> None:
        """Create a 2D Shapes layer and activate rectangle drawing."""
        base = "Keep-inside ROI"
        name = base
        existing = {getattr(lyr, "name", "") for lyr in self._viewer.layers}
        i = 2
        while name in existing:
            name = f"{base} {i}"
            i += 1
        # ndim=2 is important: drawing over a video must not attach a time
        # coordinate, so one rectangle can clip every frame identically.
        layer = self._viewer.add_shapes(
            [],
            ndim=2,
            name=name,
            shape_type="rectangle",
            edge_color="cyan",
            face_color=[0.0, 0.8, 1.0, 0.15],
            edge_width=2,
        )
        try:
            if hasattr(layer, "mode"):
                layer.mode = "add_rectangle"
        except Exception:
            pass
        self._viewer.layers.selection.active = layer
        self._clip_all_frames_cb.setChecked(True)
        idx = self._roi_shapes_combo.findData(layer)
        if idx < 0:
            self._refresh_layer_lists()
            idx = self._roi_shapes_combo.findData(layer)
        if idx >= 0:
            self._roi_shapes_combo.setCurrentIndex(idx)
        self._set_status(
            f"Created 2D '{name}'. Draw one rectangle — it will clip every frame."
        )

    def _on_apply_clip(self) -> None:
        from napari.utils.notifications import show_warning

        shapes_layer = self._selected(self._roi_shapes_combo)
        src_layer = self._selected(self._clip_mask_combo)
        if shapes_layer is None or not isinstance(shapes_layer, Shapes):
            show_warning("Select a Shapes layer (rectangle / polygon / ellipse).")
            return
        if src_layer is None or not isinstance(src_layer, (Labels, Image)):
            show_warning("Select a Labels or Image/video source layer.")
            return
        if len(shapes_layer.data) == 0:
            show_warning("Shapes layer is empty — draw a rectangle (or polygon/ellipse) first.")
            return

        src = self._layer_data(src_layer)
        try:
            spatial = raster_spatial_shape(src)
        except ValueError as exc:
            show_warning(str(exc))
            return

        is_labels = isinstance(src_layer, Labels)
        target_label = self._resolved_label(src_layer, self._clip_label_combo) if is_labels else None

        try:
            roi = build_roi_masks_for_volume(
                shapes_layer,
                spatial,
                apply_to_all_frames=bool(self._clip_all_frames_cb.isChecked()),
            )
            if not np.any(roi):
                show_warning(
                    "ROI is empty after rasterizing shapes. Use rectangle, polygon, or ellipse."
                )
                return
            if is_labels and target_label is not None:
                clipped = clip_mask_label_volume(src, roi, target_label)
            else:
                clipped = clip_mask_outside_roi(src, roi)
        except Exception as exc:
            show_warning(f"Clip failed: {exc}")
            return

        mode = str(self._clip_target_combo.currentData())
        apply_all = bool(self._clip_all_frames_cb.isChecked())
        if mode == "overwrite":
            src_layer.data = clipped
            src_layer.refresh()
            lbl_note = f" label {target_label}" if target_label is not None else ""
            self._set_status(f"Clipped outside shape and overwrote {src_layer.name}{lbl_note}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops clip {src_layer.name} by {shapes_layer.name} (overwrite)",
                {
                    "mode": "clip",
                    "ellipse_layer": shapes_layer.name,
                    "shapes_layer": shapes_layer.name,
                    "mask_layer": src_layer.name,
                    "mask_label": target_label if target_label is not None else "",
                    "apply_to_all_frames": apply_all,
                    "output_mode": "overwrite",
                    "output_layer": src_layer.name,
                },
            )
            return

        if is_labels and target_label is not None:
            out_data = label_only_volume(clipped, target_label)
        else:
            out_data = clipped
        name = f"{src_layer.name} - inside ROI"
        if target_label is not None:
            name = f"{src_layer.name} label {target_label} - inside ROI"
        if is_labels:
            self._viewer.add_labels(out_data, name=name)
        else:
            self._viewer.add_image(out_data, name=name)
        self._set_status(f"Created {name}.")
        record_pipeline_step(
            "mask_ops.operation",
            f"Mask Ops clip {src_layer.name} by {shapes_layer.name} (new)",
            {
                "mode": "clip",
                "ellipse_layer": shapes_layer.name,
                "shapes_layer": shapes_layer.name,
                "mask_layer": src_layer.name,
                "mask_label": target_label if target_label is not None else "",
                "apply_to_all_frames": apply_all,
                "output_mode": "new",
                "output_layer": name,
            },
        )

    # ------------------------------------------------------------------
    def _on_apply_binary(self) -> None:
        from napari.utils.notifications import show_warning

        a_layer = self._selected(self._a_combo)
        b_layer = self._selected(self._b_combo)
        if a_layer is None or not self._is_binary_mask_layer(a_layer):
            show_warning("Select mask A (Labels or Image).")
            return
        op = str(self._op_combo.currentData())
        if op != "not" and (b_layer is None or not self._is_binary_mask_layer(b_layer)):
            show_warning("Select mask B (Labels or Image).")
            return

        a_raw = self._layer_data(a_layer)
        label_a = self._resolved_label(a_layer, self._a_label_combo)
        label_b = self._resolved_label(b_layer, self._b_label_combo) if b_layer is not None else None
        if op == "not":
            b_raw = np.array(a_raw, copy=False)
            label_b = label_a
        else:
            b_raw = self._layer_data(b_layer)

        try:
            res_bool = apply_binary_operation_bool(
                a_raw, b_raw, op=op, label_a=label_a, label_b=label_b
            )
        except Exception as exc:
            show_warning(f"Binary operation failed: {exc}")
            return

        target = str(self._bin_target_combo.currentData())
        if target == "a":
            if isinstance(a_layer, Labels) and label_a is not None:
                self._write_labels_result(a_layer, a_raw, res_bool, label_a)
            else:
                res = apply_binary_operation(a_raw, b_raw, op=op, template=a_raw)
                self._write_binary_result(a_layer, res, a_raw)
            self._set_status(f"Applied {op.upper()} and overwrote {a_layer.name}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops {op.upper()} overwrite A ({a_layer.name})",
                {
                    "mode": "binary",
                    "a_layer": a_layer.name,
                    "b_layer": b_layer.name if b_layer is not None else "",
                    "a_label": label_a if label_a is not None else "",
                    "b_label": label_b if label_b is not None else "",
                    "op": op,
                    "target": "a",
                    "output_layer": a_layer.name,
                },
            )
            return
        if target == "b":
            if b_layer is None:
                show_warning("Cannot overwrite B: B not selected.")
                return
            if isinstance(b_layer, Labels) and label_b is not None:
                self._write_labels_result(b_layer, b_raw, res_bool, label_b)
            else:
                res = apply_binary_operation(a_raw, b_raw, op=op, template=b_raw)
                self._write_binary_result(b_layer, res, b_raw)
            self._set_status(f"Applied {op.upper()} and overwrote {b_layer.name}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops {op.upper()} overwrite B ({b_layer.name})",
                {
                    "mode": "binary",
                    "a_layer": a_layer.name,
                    "b_layer": b_layer.name,
                    "a_label": label_a if label_a is not None else "",
                    "b_label": label_b if label_b is not None else "",
                    "op": op,
                    "target": "b",
                    "output_layer": b_layer.name,
                },
            )
            return

        b_name = b_layer.name if b_layer is not None else "none"
        out_name = f"{a_layer.name} {op.upper()} {b_name}"
        if label_a is not None:
            out_name = f"{a_layer.name} L{label_a} {op.upper()} {b_name}"
        out_label = label_a if label_a is not None else 1
        if isinstance(a_layer, Labels):
            out_data = new_labels_from_binary(res_bool, out_label, dtype=a_raw.dtype)
            self._viewer.add_labels(out_data, name=out_name)
        else:
            res = apply_binary_operation(
                a_raw, b_raw, op=op, template=a_raw, label_a=label_a, label_b=label_b
            )
            self._add_binary_output(out_name, res, a_layer, a_raw)
        self._set_status(f"Created {out_name}.")
        record_pipeline_step(
            "mask_ops.operation",
            f"Mask Ops {op.upper()} new layer from {a_layer.name}",
            {
                "mode": "binary",
                "a_layer": a_layer.name,
                "b_layer": b_layer.name if b_layer is not None else "",
                "a_label": label_a if label_a is not None else "",
                "b_label": label_b if label_b is not None else "",
                "op": op,
                "target": "new",
                "output_layer": out_name,
            },
        )
