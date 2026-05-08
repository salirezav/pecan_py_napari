"""Mask Ops widget: clip masks by ellipse and run binary mask operations."""

from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Labels, Shapes
from qtpy.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .logic import (
    apply_binary_operation,
    build_ellipse_masks_for_volume,
    clip_mask_outside_ellipse,
)
from ..pipeline_recorder.state import record_pipeline_step


class MaskOpsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building_ui = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # -------- Clip by ellipse -----------------------------------------
        g_clip = QGroupBox("Clip mask by ellipse")
        l_clip = QVBoxLayout(g_clip)
        l_clip.addWidget(QLabel("Ellipse layer (Shapes)"))
        self._ellipse_combo = QComboBox()
        self._ellipse_combo.addItem("(none)", None)
        l_clip.addWidget(self._ellipse_combo)

        l_clip.addWidget(QLabel("Mask layer (Labels)"))
        self._clip_mask_combo = QComboBox()
        self._clip_mask_combo.addItem("(none)", None)
        l_clip.addWidget(self._clip_mask_combo)

        row_clip_out = QHBoxLayout()
        row_clip_out.addWidget(QLabel("Output:"))
        self._clip_target_combo = QComboBox()
        self._clip_target_combo.addItem("New layer", "new")
        self._clip_target_combo.addItem("Overwrite mask layer", "overwrite")
        row_clip_out.addWidget(self._clip_target_combo, 1)
        l_clip.addLayout(row_clip_out)

        self._btn_clip = QPushButton("Apply clip (remove outside ellipse)")
        self._btn_clip.clicked.connect(self._on_apply_clip)
        l_clip.addWidget(self._btn_clip)
        layout.addWidget(g_clip)

        # -------- Binary ops ----------------------------------------------
        g_bin = QGroupBox("Binary operations")
        l_bin = QVBoxLayout(g_bin)
        l_bin.addWidget(QLabel("Mask A (Labels)"))
        self._a_combo = QComboBox()
        self._a_combo.addItem("(none)", None)
        l_bin.addWidget(self._a_combo)

        l_bin.addWidget(QLabel("Mask B (Labels)"))
        self._b_combo = QComboBox()
        self._b_combo.addItem("(none)", None)
        l_bin.addWidget(self._b_combo)

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
        self._bin_target_combo.addItem("Overwrite A", "a")
        self._bin_target_combo.addItem("Overwrite B", "b")
        row_target.addWidget(self._bin_target_combo, 1)
        l_bin.addLayout(row_target)

        self._btn_bin = QPushButton("Apply binary operation")
        self._btn_bin.clicked.connect(self._on_apply_binary)
        l_bin.addWidget(self._btn_bin)
        layout.addWidget(g_bin)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)
        layout.addStretch(1)

        self._op_combo.currentIndexChanged.connect(self._update_b_enabled)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_lists)
        self._viewer.layers.events.removed.connect(self._refresh_layer_lists)
        self._refresh_layer_lists()
        self._update_b_enabled()

    # ------------------------------------------------------------------
    def _refresh_layer_lists(self, _event: Any = None) -> None:
        self._building_ui = True
        try:
            prev_ellipse = self._ellipse_combo.currentData()
            prev_clip_mask = self._clip_mask_combo.currentData()
            prev_a = self._a_combo.currentData()
            prev_b = self._b_combo.currentData()

            for cmb in (self._ellipse_combo, self._clip_mask_combo, self._a_combo, self._b_combo):
                cmb.clear()
                cmb.addItem("(none)", None)

            for layer in self._viewer.layers:
                if isinstance(layer, Shapes):
                    self._ellipse_combo.addItem(layer.name, layer)
                if isinstance(layer, Labels):
                    self._clip_mask_combo.addItem(layer.name, layer)
                    self._a_combo.addItem(layer.name, layer)
                    self._b_combo.addItem(layer.name, layer)

            self._restore_combo(self._ellipse_combo, prev_ellipse)
            self._restore_combo(self._clip_mask_combo, prev_clip_mask)
            self._restore_combo(self._a_combo, prev_a)
            self._restore_combo(self._b_combo, prev_b)
        finally:
            self._building_ui = False

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

    # ------------------------------------------------------------------
    def _on_apply_clip(self) -> None:
        from napari.utils.notifications import show_warning

        ellipse_layer = self._selected(self._ellipse_combo)
        mask_layer = self._selected(self._clip_mask_combo)
        if ellipse_layer is None or not isinstance(ellipse_layer, Shapes):
            show_warning("Select an ellipse Shapes layer.")
            return
        if mask_layer is None or not isinstance(mask_layer, Labels):
            show_warning("Select a mask Labels layer.")
            return

        mask = self._layer_data(mask_layer)
        if mask.ndim not in (2, 3):
            show_warning(f"Mask must be 2D or 3D (T,H,W), got {mask.shape}.")
            return

        try:
            ell = build_ellipse_masks_for_volume(ellipse_layer, tuple(mask.shape))
            clipped = clip_mask_outside_ellipse(mask, ell)
        except Exception as exc:
            show_warning(f"Clip failed: {exc}")
            return

        mode = str(self._clip_target_combo.currentData())
        if mode == "overwrite":
            mask_layer.data = clipped
            mask_layer.refresh()
            self._set_status(f"Clipped outside ellipse and overwrote {mask_layer.name}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops clip {mask_layer.name} by {ellipse_layer.name} (overwrite)",
                {
                    "mode": "clip",
                    "ellipse_layer": ellipse_layer.name,
                    "mask_layer": mask_layer.name,
                    "output_mode": "overwrite",
                    "output_layer": mask_layer.name,
                },
            )
            return

        name = f"{mask_layer.name} - inside ellipse"
        self._viewer.add_labels(clipped, name=name)
        self._set_status(f"Created {name}.")
        record_pipeline_step(
            "mask_ops.operation",
            f"Mask Ops clip {mask_layer.name} by {ellipse_layer.name} (new)",
            {
                "mode": "clip",
                "ellipse_layer": ellipse_layer.name,
                "mask_layer": mask_layer.name,
                "output_mode": "new",
                "output_layer": name,
            },
        )

    # ------------------------------------------------------------------
    def _on_apply_binary(self) -> None:
        from napari.utils.notifications import show_warning

        a_layer = self._selected(self._a_combo)
        b_layer = self._selected(self._b_combo)
        if a_layer is None or not isinstance(a_layer, Labels):
            show_warning("Select mask A (Labels).")
            return
        op = str(self._op_combo.currentData())
        if op != "not" and (b_layer is None or not isinstance(b_layer, Labels)):
            show_warning("Select mask B (Labels).")
            return

        a = self._layer_data(a_layer)
        if a.ndim not in (2, 3):
            show_warning(f"A must be 2D or 3D (T,H,W), got {a.shape}.")
            return
        if op == "not":
            b = np.array(a, copy=False)
            template = a
        else:
            b = self._layer_data(b_layer)
            if b.ndim not in (2, 3):
                show_warning(f"B must be 2D or 3D (T,H,W), got {b.shape}.")
                return
            if a.shape != b.shape:
                show_warning(f"A and B must have same shape, got {a.shape} vs {b.shape}.")
                return
            template = a

        try:
            res = apply_binary_operation(a, b, op=op, template=template)
        except Exception as exc:
            show_warning(f"Binary operation failed: {exc}")
            return

        target = str(self._bin_target_combo.currentData())
        if target == "a":
            a_layer.data = res
            a_layer.refresh()
            self._set_status(f"Applied {op.upper()} and overwrote {a_layer.name}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops {op.upper()} overwrite A ({a_layer.name})",
                {
                    "mode": "binary",
                    "a_layer": a_layer.name,
                    "b_layer": b_layer.name if b_layer is not None else "",
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
            b_layer.data = res
            b_layer.refresh()
            self._set_status(f"Applied {op.upper()} and overwrote {b_layer.name}.")
            record_pipeline_step(
                "mask_ops.operation",
                f"Mask Ops {op.upper()} overwrite B ({b_layer.name})",
                {
                    "mode": "binary",
                    "a_layer": a_layer.name,
                    "b_layer": b_layer.name,
                    "op": op,
                    "target": "b",
                    "output_layer": b_layer.name,
                },
            )
            return

        b_name = b_layer.name if b_layer is not None else "none"
        out_name = f"{a_layer.name} {op.upper()} {b_name}"
        self._viewer.add_labels(res, name=out_name)
        self._set_status(f"Created {out_name}.")
        record_pipeline_step(
            "mask_ops.operation",
            f"Mask Ops {op.upper()} new layer from {a_layer.name}",
            {
                "mode": "binary",
                "a_layer": a_layer.name,
                "b_layer": b_layer.name if b_layer is not None else "",
                "op": op,
                "target": "new",
                "output_layer": out_name,
            },
        )
