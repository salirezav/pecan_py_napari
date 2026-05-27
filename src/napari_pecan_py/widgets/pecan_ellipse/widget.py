"""Pecan ellipse: fit a napari **Shapes** layer (ellipse) from a mask (Labels / Image).

The ellipse always lives in its **own** layer (never merged into the mask). The layer
type is napari ``Shapes`` with ``shape_type='ellipse'``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from napari.layers import Image, Labels, Layer, Shapes
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
    fit_debug_summary,
    fit_ellipses_volume,
    mask_volume_needs_time_coord,
    normalize_smooth_window,
)
from ..pipeline_recorder.state import upsert_pipeline_step


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
            "only picks what you paint. Color Thresholding pecan masks are usually 1. "
            "Use 0 = any foreground pixel."
        )
        row.addWidget(self._label_spin, 1)
        src_lay.addLayout(row)
        id_help = QLabel(
            "<b>Mask pixel value</b> = number in the image data (often <b>1</b> for "
            "Color Thresholding masks). It is <b>not</b> the layer control “label” used for the "
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
        auto_hint = QLabel("Click button to recompute ellipses from current mask data.")
        auto_hint.setWordWrap(True)
        auto_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        opt_lay.addWidget(auto_hint)
        layout.addWidget(opt)

        smooth = QGroupBox("Temporal smoothing")
        smooth_lay = QVBoxLayout(smooth)
        self._smooth_cb = QCheckBox("Smooth ellipse parameters across frames")
        self._smooth_cb.setToolTip(
            "Moving average of center, size, and angle over a sliding window. "
            "Reduces single-frame mask oversegmentation spikes; large windows "
            "can lag fast motion or tumbling."
        )
        self._smooth_cb.toggled.connect(self._on_smooth_toggled)
        smooth_lay.addWidget(self._smooth_cb)
        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Window (frames):", self))
        self._smooth_win = QSpinBox(self)
        self._smooth_win.setRange(3, 101)
        self._smooth_win.setSingleStep(2)
        self._smooth_win.setValue(5)
        self._smooth_win.setToolTip("Odd number of frames (centered average). Use 3–7 for light smoothing.")
        self._smooth_win.setEnabled(False)
        win_row.addWidget(self._smooth_win, 1)
        smooth_lay.addLayout(win_row)
        smooth_hint = QLabel(
            "Best for time-series masks with one pecan. Does nothing on a single 2D mask."
        )
        smooth_hint.setWordWrap(True)
        smooth_hint.setStyleSheet("color: #aaa; font-size: 11px;")
        smooth_lay.addWidget(smooth_hint)
        layout.addWidget(smooth)

        btn_row = QHBoxLayout()
        self._btn_all = QPushButton("Fit ellipses (all frames)")
        self._btn_all.clicked.connect(self._on_fit_all)
        btn_row.addWidget(self._btn_all)
        layout.addLayout(btn_row)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #888;")
        layout.addWidget(self._status)

        layout.addStretch(1)

        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._refresh_layer_list)

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
        arr = np.asarray(d)
        if arr.dtype == object:
            # Some backends expose frame stacks as list/tuple; keep all frames.
            if isinstance(d, (list, tuple)) and len(d) > 0:
                try:
                    arr = np.stack([np.asarray(x) for x in d], axis=0)
                except Exception:
                    arr = np.asarray(d[0])
            else:
                arr = np.asarray(layer.data[0])
        return arr

    def _on_layer_changed(self) -> None:
        if self._building_ui:
            return
        self._status.clear()

    def _on_smooth_toggled(self, checked: bool) -> None:
        self._smooth_win.setEnabled(bool(checked))

    def _ellipse_layer_name(self) -> str:
        layer = self._selected_layer()
        if layer is None:
            return "Pecan ellipse"
        return f"{layer.name} - ellipse"

    def _label_id_param(self) -> int | None:
        v = int(self._label_spin.value())
        if v <= 0:
            return None
        return v

    def _is_time_series_mask(self, layer: Layer, data: np.ndarray) -> bool:
        # Labels layers with ndim>=3 are treated as frame stacks.
        if isinstance(layer, Labels):
            return data.ndim >= 3
        return mask_volume_needs_time_coord(data)

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

    def _on_fit_all(self) -> None:
        layer = self._selected_layer()
        if layer is None:
            self._status.setText("Select a mask layer.")
            return
        data = self._layer_volume_data(layer)
        d = data
        if self._is_time_series_mask(layer, d):
            frame_indices = list(range(int(d.shape[0])))
        else:
            frame_indices = [0]
        label_id = self._label_id_param()
        largest = self._largest_cb.isChecked()
        temporal_smooth = self._smooth_cb.isChecked() and len(frame_indices) >= 2
        smooth_window = normalize_smooth_window(int(self._smooth_win.value()))
        out = fit_ellipses_volume(
            data,
            label_id=label_id,
            largest_only=largest,
            temporal_smooth=temporal_smooth,
            smooth_window=smooth_window,
        )
        if not out:
            hint_idx = int(frame_indices[0]) if frame_indices else 0
            hint = fit_debug_summary(data, hint_idx, label_id=label_id)
            msg = f"Could not fit on any frame. Example (frame {hint_idx}): {hint}"
            self._status.setText(msg)
            from napari.utils.notifications import show_warning

            show_warning(
                "Pecan ellipse: no ellipses created.\n"
                + msg
                + "\n\nTry **Mask pixel value = 0**, or verify the mask layer shows "
                f"non‑zero data at frame {hint_idx}."
            )
            return
        try:
            self._upsert_shapes_layer(layer, out)
        except Exception:
            return
        self._status.setText(
            f"Shapes «{self._ellipse_layer_name()}»: {len(out)} ellipse(s)."
        )
        params = {
            "mask_layer": layer.name,
            "output_shapes_layer": self._ellipse_layer_name(),
            "label_id": label_id,
            "largest_only": largest,
            "temporal_smooth": temporal_smooth,
            "smooth_window": smooth_window,
            "mode": "all",
            "time_index": int(frame_indices[0]) if frame_indices else 0,
        }
        upsert_pipeline_step(
            kind="pecan_ellipse.fit",
            description=f"Pecan Ellipse fit all frames on {layer.name}",
            params=params,
            match=lambda st: (
                st.kind == "pecan_ellipse.fit"
                and str((st.params or {}).get("mask_layer", "")) == layer.name
                and str((st.params or {}).get("output_shapes_layer", "")) == self._ellipse_layer_name()
            ),
        )
