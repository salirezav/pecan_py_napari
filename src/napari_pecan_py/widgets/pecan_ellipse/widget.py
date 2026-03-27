"""Pecan ellipse: fit a napari **Shapes** layer (ellipse) from a mask (Labels / Image).

The ellipse always lives in its **own** layer (never merged into the mask). The layer
type is napari ``Shapes`` with ``shape_type='ellipse'``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Image, Labels, Layer, Shapes
from qtpy.QtCore import QTimer
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .logic import (
    apply_ellipse_pipeline,
    fit_debug_summary,
    mask_volume_needs_time_coord,
    resolve_time_index_for_volume,
)


class PecanEllipseWidget(QWidget):
    """Fit OpenCV ellipse to largest contour of a pecan (or other) mask slice."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._building_ui = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        src = QGroupBox("Mask layer")
        src_lay = QVBoxLayout(src)
        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        src_lay.addWidget(self._layer_combo)

        row = QHBoxLayout()
        row.addWidget(QLabel("Mask pixel value:", self))
        self._label_spin = QSpinBox(self)
        self._label_spin.setRange(0, 999999)
        self._label_spin.setValue(1)
        self._label_spin.setToolTip(
            "Integer stored in the mask array at pecan pixels (check the status bar / "
            "hover value). This is NOT the brush “selected label” in Layer controls—that "
            "only picks what you paint. Color Tuner pecan masks are usually 1. "
            "Use 0 = any foreground pixel."
        )
        row.addWidget(self._label_spin, 1)
        src_lay.addLayout(row)
        id_help = QLabel(
            "<b>Mask pixel value</b> = number in the image data (often <b>1</b> for "
            "Color Tuner masks). It is <b>not</b> the layer control “label” used for the "
            "paint brush. Use <b>0</b> to include every non‑zero pixel."
        )
        id_help.setWordWrap(True)
        id_help.setStyleSheet("color: #aaa; font-size: 11px;")
        src_lay.addWidget(id_help)
        layout.addWidget(src)

        out_info = QLabel(
            "Output: a separate <b>Shapes</b> layer named "
            "<i>«mask name» - ellipse</i> (editable in the layer list). "
            "It is not mixed into the mask layer."
        )
        out_info.setWordWrap(True)
        out_info.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(out_info)

        opt = QGroupBox("Contour")
        opt_lay = QVBoxLayout(opt)
        self._largest_cb = QCheckBox("Use largest outer contour only")
        self._largest_cb.setChecked(True)
        self._largest_cb.setToolTip("Typical for a single pecan blob; off merges all contours.")
        opt_lay.addWidget(self._largest_cb)
        self._auto_cb = QCheckBox("Auto-update on frame change (current slice)")
        self._auto_cb.setToolTip("Re-fit on the visible time index after a short delay.")
        self._auto_cb.stateChanged.connect(self._on_auto_changed)
        opt_lay.addWidget(self._auto_cb)
        layout.addWidget(opt)

        btn_row = QHBoxLayout()
        self._btn_current = QPushButton("Fit ellipse (current frame)")
        self._btn_current.clicked.connect(self._on_fit_current)
        btn_row.addWidget(self._btn_current)
        self._btn_all = QPushButton("Fit ellipse (all frames)")
        self._btn_all.clicked.connect(self._on_fit_all)
        btn_row.addWidget(self._btn_all)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        layout.addStretch(1)

        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(300)
        self._debounce.timeout.connect(self._on_fit_current)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)
        self._viewer.dims.events.current_step.connect(self._on_dims_changed)

        self._refresh_layer_list()

    def _refresh_layer_list(self) -> None:
        self._building_ui = True
        prev = self._layer_combo.currentData()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if isinstance(layer, (Image, Labels)):
                try:
                    if layer.data is None:
                        continue
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

    def _selected_layer(self) -> Layer | None:
        data = self._layer_combo.currentData()
        if data is None or data not in self._viewer.layers:
            return None
        return data

    def _layer_volume_data(self, layer: Layer) -> np.ndarray:
        """Full-resolution (T,H,W) or (H,W) array; multiscale Labels use level 0."""
        d = layer.data
        if getattr(layer, "multiscale", False):
            d = d[0]
        elif isinstance(d, (list, tuple)) and len(d) > 0 and hasattr(d[0], "shape"):
            if len(d[0].shape) >= 2:
                d = d[0]
        arr = np.asarray(d)
        if arr.dtype == object:
            arr = np.asarray(layer.data[0])
        return arr

    def _on_layer_changed(self) -> None:
        if self._building_ui:
            return
        self._status.clear()

    def _ellipse_layer_name(self) -> str:
        layer = self._selected_layer()
        if layer is None:
            return "Pecan ellipse"
        return f"{layer.name} - ellipse"

    def _current_t(self, data: np.ndarray) -> int:
        return resolve_time_index_for_volume(data, self._viewer)

    def _label_id_param(self) -> int | None:
        v = int(self._label_spin.value())
        if v <= 0:
            return None
        return v

    def _on_dims_changed(self, _event: Any = None) -> None:
        if not self._auto_cb.isChecked():
            return
        if self._selected_layer() is None:
            return
        self._debounce.start()

    def _on_auto_changed(self) -> None:
        if self._auto_cb.isChecked() and self._selected_layer() is not None:
            self._debounce.start()

    def _upsert_shapes_layer(self, ref: Layer, list_of_vertices: list[np.ndarray]) -> None:
        name = self._ellipse_layer_name()
        meta = {
            "pecan_ellipse": True,
            "source_mask_layer": ref.name,
            "layer_class": "Shapes",
        }
        kwargs: dict[str, Any] = dict(
            name=name,
            shape_type="ellipse",
            edge_width=2.5,
            edge_color="coral",
            face_color="transparent",
            metadata=meta,
        )
        try:
            kwargs["scale"] = ref.scale
            kwargs["translate"] = ref.translate
        except Exception:
            pass

        try:
            existing = self._viewer.layers[name]
        except KeyError:
            existing = None
        if existing is not None and isinstance(existing, Shapes):
            lyr: Shapes = existing
            m = dict(getattr(lyr, "metadata", None) or {})
            m.update(meta)
            lyr.metadata = m
            lyr.data = list_of_vertices
            lyr.refresh()
            return
        if existing is not None:
            self._viewer.layers.remove(name)
        try:
            self._viewer.add_shapes(list_of_vertices, **kwargs)
        except Exception as exc:
            kwargs.pop("face_color", None)
            try:
                self._viewer.add_shapes(list_of_vertices, **kwargs)
            except Exception as exc2:
                from napari.utils.notifications import show_error

                err_txt = f"{exc2}\n(original: {exc})"
                hint = ""
                if "bermuda" in err_txt.lower() and "triangulate" in err_txt.lower():
                    hint = (
                        "\n\nThis is an environment issue, not the Pecan Ellipse code. "
                        "Fix one of:\n"
                        "• Upgrade bermuda:  pip install -U \"bermuda>=0.1.7\"\n"
                        "• Or napari: Edit → Preferences → Experimental → Triangulation backend → "
                        "“Pure Python” (then restart napari).\n"
                        "• Ensure no file named bermuda.py is on PYTHONPATH (it can shadow the real package)."
                    )
                show_error(f"Could not create Shapes layer:\n{err_txt}{hint}")
                return

    def _on_fit_current(self) -> None:
        layer = self._selected_layer()
        if layer is None:
            self._status.setText("Select a mask layer.")
            return
        data = self._layer_volume_data(layer)
        t = self._current_t(data)
        lid = self._label_id_param()
        largest = self._largest_cb.isChecked()
        verts = apply_ellipse_pipeline(data, t, label_id=lid, largest_only=largest)
        if verts is None:
            detail = fit_debug_summary(data, t, label_id=lid)
            msg = (
                "Could not fit ellipse (no contour or too few boundary points). "
                f"{detail}"
            )
            self._status.setText(msg)
            from napari.utils.notifications import show_warning

            show_warning(
                "Pecan ellipse: fit failed.\n"
                + msg
                + "\n\nTry **Mask pixel value = 0** (any foreground), or match the "
                "integer under the cursor in the status bar (not the brush “label”)."
            )
            return
        try:
            self._upsert_shapes_layer(layer, [verts])
        except Exception:
            return
        self._status.setText(
            f"Shapes layer «{self._ellipse_layer_name()}» updated (frame {t})."
        )

    def _on_fit_all(self) -> None:
        layer = self._selected_layer()
        if layer is None:
            self._status.setText("Select a mask layer.")
            return
        data = self._layer_volume_data(layer)
        d = data
        if not mask_volume_needs_time_coord(d):
            self._on_fit_current()
            self._status.setText("Single frame — used current fit.")
            return
        T = int(d.shape[0])
        out: list[np.ndarray] = []
        label_id = self._label_id_param()
        largest = self._largest_cb.isChecked()
        for t in range(T):
            v = apply_ellipse_pipeline(data, t, label_id=label_id, largest_only=largest)
            if v is not None:
                out.append(v)
        if not out:
            hint = fit_debug_summary(data, 0, label_id=label_id)
            msg = f"Could not fit on any frame. Example (frame 0): {hint}"
            self._status.setText(msg)
            from napari.utils.notifications import show_warning

            show_warning(
                "Pecan ellipse: no ellipses created.\n"
                + msg
                + "\n\nTry **Mask pixel value = 0**, or verify the mask layer shows "
                "non‑zero data at frame 0."
            )
            return
        try:
            self._upsert_shapes_layer(layer, out)
        except Exception:
            return
        self._status.setText(
            f"Shapes «{self._ellipse_layer_name()}»: {len(out)} ellipse(s)."
        )
