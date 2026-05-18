"""Color Thresholding dock widget: layer, target, color space, channel sliders, frame, apply."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Image

# from napari.viewer import Viewer
from qtpy.QtCore import Qt, QTimer
from qtpy.QtWidgets import QComboBox, QFileDialog, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea, QSlider, QSpinBox, QVBoxLayout, QWidget

from .defaults import COLOR_SPACE_PARAMS, COLOR_SPACES, MASK_COLORS, TARGETS, DEFAULT_THRESHOLDS, copy_default_thresholds
from .logic import apply_thresholds, frame_rgb_to_color_space
from ..pipeline_recorder.state import upsert_pipeline_step

CURSOR_LAYER_NAME = "Color Thresholding cursor"


def _image_layer_data_shape(layer: Image) -> tuple[int, ...] | None:
    data = getattr(layer, "data", None)
    if data is None:
        return None
    shape = getattr(data, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(d) for d in shape)
    except (TypeError, ValueError):
        return None


def _image_layer_is_rgb_video(layer: Image) -> bool:
    """True for (T,Y,X,C) or (Y,X,C) with at least 3 channels; avoids materializing lazy arrays."""
    if not isinstance(layer, Image):
        return False
    shape = _image_layer_data_shape(layer)
    if not shape:
        return False
    ndim = len(shape)
    if ndim == 4 and shape[-1] >= 3:
        return True
    if ndim == 3 and shape[-1] >= 3:
        return True
    return False


class ColorThresholdingWidget(QWidget):
    """Vertical panel: layer, target, color space, channel sliders, frame, buttons."""

    def __init__(self, napari_viewer):
        super().__init__()
        self._viewer = napari_viewer
        self._thresholds = copy_default_thresholds()
        self._current_target = "pecan"
        self._current_color_space = "hsv"
        self._channel_widgets = []  # list of dicts: {name, min_slider, max_slider, min_spin, max_spin}
        self._frame_index = 0
        self._building_ui = False
        # Respect manual deletion of auto-generated mask layer until thresholds change.
        self._suppress_mask_autocreate: dict[str, bool] = {}
        self._mask_dirty: dict[str, bool] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # Layer
        layer_group = QGroupBox("Layer")
        layer_layout = QVBoxLayout(layer_group)
        self._layer_combo = QComboBox()
        self._layer_combo.addItem("(none)", None)
        self._layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        layer_layout.addWidget(self._layer_combo)
        layout.addWidget(layer_group)

        # Target
        target_group = QGroupBox("Target")
        target_layout = QVBoxLayout(target_group)
        self._target_combo = QComboBox()
        for t in TARGETS:
            self._target_combo.addItem(t.replace("_", " ").title(), t)
        self._target_combo.currentIndexChanged.connect(self._on_target_changed)
        target_layout.addWidget(self._target_combo)
        layout.addWidget(target_group)

        # Color space
        cs_group = QGroupBox("Color space")
        cs_layout = QVBoxLayout(cs_group)
        self._color_space_combo = QComboBox()
        preferred_order = ["hsv", "lab", "rgb"]
        ordered_spaces = [cs for cs in preferred_order if cs in COLOR_SPACES]
        ordered_spaces.extend([cs for cs in COLOR_SPACES if cs not in ordered_spaces])
        for cs in ordered_spaces:
            self._color_space_combo.addItem(cs.upper(), cs)
        self._color_space_combo.currentIndexChanged.connect(self._on_color_space_changed)
        cs_layout.addWidget(self._color_space_combo)
        layout.addWidget(cs_group)

        # Eyedropper
        picker_group = QGroupBox("Eyedropper")
        picker_layout = QVBoxLayout(picker_group)
        picker_row = QHBoxLayout()
        self._picker_btn = QPushButton("Pick color")
        self._picker_btn.setCheckable(True)
        self._picker_btn.setChecked(False)
        self._picker_btn.toggled.connect(self._on_picker_toggled)
        picker_row.addWidget(self._picker_btn)
        picker_row.addWidget(QLabel("Patch:"))
        self._patch_spin = QSpinBox()
        self._patch_spin.setRange(1, 51)
        self._patch_spin.setSingleStep(2)
        self._patch_spin.setValue(5)
        self._patch_spin.setToolTip("Patch size (n x n) for color sampling")
        picker_row.addWidget(self._patch_spin)
        picker_layout.addLayout(picker_row)
        self._picker_info = QLabel("")
        picker_layout.addWidget(self._picker_info)
        layout.addWidget(picker_group)
        self._picker_active = False
        self._sampled_pixels: list[np.ndarray] = []
        self._excluded_pixels: list[np.ndarray] = []

        # Channel sliders (rebuilt when color space or target changes)
        channels_group = QGroupBox("Color thresholds")
        self._channels_layout = QVBoxLayout(channels_group)
        layout.addWidget(channels_group)

        # Frame
        frame_group = QGroupBox("Frame")
        frame_layout = QVBoxLayout(frame_group)
        frame_row = QHBoxLayout()
        self._frame_slider = QSlider(Qt.Orientation.Horizontal)
        self._frame_slider.setMinimum(0)
        self._frame_slider.setMaximum(0)
        self._frame_slider.valueChanged.connect(self._on_frame_slider_changed)
        self._frame_spin = QSpinBox()
        self._frame_spin.setMinimum(0)
        self._frame_spin.setMaximum(0)
        self._frame_spin.valueChanged.connect(self._on_frame_spin_changed)
        frame_row.addWidget(self._frame_slider, 1)
        frame_row.addWidget(self._frame_spin)
        frame_layout.addLayout(frame_row)
        layout.addWidget(frame_group)

        # Debounce timer – applies thresholds to the visible frame only (lazy-friendly).
        self._update_timer = QTimer(self)
        self._update_timer.setSingleShot(True)
        self._update_timer.setInterval(80)
        self._update_timer.timeout.connect(self._apply_current_frame_mask)

        # Buttons
        btn_layout = QHBoxLayout()
        self._btn_reset_minmax = QPushButton("Reset to min/max")
        self._btn_reset_minmax.clicked.connect(self._reset_to_minmax)
        self._btn_reset_default = QPushButton("Reset to default")
        self._btn_reset_default.clicked.connect(self._reset_to_default)
        btn_layout.addWidget(self._btn_reset_minmax)
        btn_layout.addWidget(self._btn_reset_default)
        layout.addLayout(btn_layout)

        # Save masks
        save_group = QGroupBox("Save masks")
        save_lay = QHBoxLayout(save_group)
        self._save_fmt_combo = QComboBox()
        self._save_fmt_combo.addItem("TIFF (.tiff)", "tiff")
        self._save_fmt_combo.addItem("NumPy (.npy)", "npy")
        save_lay.addWidget(self._save_fmt_combo)
        self._btn_save_masks = QPushButton("Save masks")
        self._btn_save_masks.setToolTip("Exports label layers that belong to the current video. " "Per-frame masks (Color Thresholding) include the current frame index in the file name.")
        self._btn_save_masks.clicked.connect(self._save_masks)
        save_lay.addWidget(self._btn_save_masks)
        layout.addWidget(save_group)

        layout.addStretch(1)

        # Populate layer list and sync with viewer
        self._refresh_layer_list()
        self._build_channel_sliders()
        self._sync_frame_from_viewer()

        self._viewer.layers.events.inserted.connect(self._refresh_layer_list)
        self._viewer.layers.events.removed.connect(self._on_layer_removed)
        self._viewer.dims.events.current_step.connect(self._on_dims_current_step)

    def _on_layer_removed(self, event=None):
        removed = getattr(event, "value", None)
        if removed is not None:
            rname = getattr(removed, "name", None)
            if isinstance(rname, str):
                for tgt in TARGETS:
                    if rname == self._mask_layer_name(tgt):
                        self._suppress_mask_autocreate[tgt] = True
        self._refresh_layer_list()

    def _refresh_layer_list(self):
        self._building_ui = True
        prev_layer = self._get_current_layer()
        self._layer_combo.clear()
        self._layer_combo.addItem("(none)", None)
        for layer in self._viewer.layers:
            if not isinstance(layer, Image):
                continue
            if _image_layer_is_rgb_video(layer):
                self._layer_combo.addItem(layer.name, layer)
        # Restore selection if possible
        if prev_layer is not None and prev_layer in self._viewer.layers:
            idx = self._layer_combo.findData(prev_layer)
            if idx >= 0:
                self._layer_combo.setCurrentIndex(idx)
        self._building_ui = False
        if self._get_current_layer() != prev_layer:
            self._on_layer_changed()

    def _get_current_layer(self):
        data = self._layer_combo.currentData()
        if data is None:
            return None
        if data in self._viewer.layers:
            return data
        return None

    def _mask_layer_name(self, target: str | None = None) -> str:
        if target is None:
            target = self._current_target
        layer = self._get_current_layer()
        base = layer.name if layer is not None else "Video"
        return f"{base} - {target.replace('_', ' ').title()}"

    @staticmethod
    def _label_colormap(target: str):
        """Build a direct Labels colormap: 0 = transparent, 1 = target color."""
        bgr = MASK_COLORS.get(target, (128, 128, 128))
        r, g, b = bgr[2], bgr[1], bgr[0]
        from napari.utils.colormaps import direct_colormap

        return direct_colormap(color_dict={0: np.array([0, 0, 0, 0], dtype=np.float32), 1: np.array([r / 255, g / 255, b / 255, 1.0], dtype=np.float32), None: np.array([0, 0, 0, 0], dtype=np.float32)})

    def _on_layer_changed(self):
        if self._building_ui:
            return
        layer = self._get_current_layer()
        if layer is None:
            self._frame_slider.setMaximum(0)
            self._frame_spin.setMaximum(0)
            self._frame_index = 0
            self._remove_mask_layer()
            return
        shape = _image_layer_data_shape(layer)
        try:
            T = int(shape[0]) if shape is not None and len(shape) == 4 else 1
        except (IndexError, TypeError):
            T = 1
        T = max(1, T)
        self._frame_slider.setMaximum(T - 1)
        self._frame_spin.setMaximum(T - 1)
        synced = self._sync_frame_from_viewer(total_frames=T)
        if synced is None:
            self._frame_index = min(self._frame_index, T - 1)
        self._frame_slider.setValue(self._frame_index)
        self._frame_spin.setValue(self._frame_index)
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _on_target_changed(self):
        idx = self._target_combo.currentIndex()
        if idx < 0:
            return
        self._current_target = self._target_combo.currentData()
        self._update_sliders_from_thresholds()

        name = self._mask_layer_name()
        try:
            existing = self._viewer.layers[name]
            self._viewer.layers.selection.active = existing
            self._suppress_mask_autocreate[self._current_target] = False
        except KeyError:
            pass

        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _on_color_space_changed(self):
        idx = self._color_space_combo.currentIndex()
        if idx < 0:
            return
        self._current_color_space = self._color_space_combo.currentData()
        self._build_channel_sliders()
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    # ---- Eyedropper --------------------------------------------------------

    def _on_picker_toggled(self, checked: bool):
        if checked:
            self._picker_active = True
            self._sampled_pixels.clear()
            self._excluded_pixels.clear()
            self._picker_btn.setText("Pick color (click / Shift+add / Alt+remove)")
            self._viewer.mouse_drag_callbacks.append(self._eyedropper_callback)
            self._viewer.mouse_move_callbacks.append(self._eyedropper_move_callback)
        else:
            self._picker_active = False
            self._picker_btn.setText("Pick color")
            try:
                self._viewer.mouse_drag_callbacks.remove(self._eyedropper_callback)
            except ValueError:
                pass
            try:
                self._viewer.mouse_move_callbacks.remove(self._eyedropper_move_callback)
            except ValueError:
                pass
            self._remove_cursor_layer()

    def _pos_to_yx(self, pos, layer):
        """Convert event position to (y, x) in the frame coordinate space."""
        try:
            if hasattr(layer, "world_to_data"):
                data_pos = layer.world_to_data(pos)
            else:
                data_pos = pos
            coords = np.asarray(data_pos, dtype=float).ravel()
        except Exception:
            return None, None
        if coords.size < 2:
            return None, None
        return int(round(float(coords[-2]))), int(round(float(coords[-1])))

    def _eyedropper_move_callback(self, viewer, event):
        """Update the cursor rectangle overlay while the picker is active."""
        if not self._picker_active:
            return
        layer = self._get_current_layer()
        if layer is None:
            return
        y, x = self._pos_to_yx(event.position, layer)
        if y is None:
            return
        self._update_cursor_rect(y, x, layer)

    def _update_cursor_rect(self, y: int, x: int, layer):
        n = self._patch_spin.value()
        half = n / 2.0
        shape = _image_layer_data_shape(layer)
        ndim = len(shape) if shape else 0
        if ndim == 4:
            t = float(self._frame_index)
            rect = np.array([[t, y - half, x - half], [t, y - half, x + half], [t, y + half, x + half], [t, y + half, x - half]])
        elif ndim == 3:
            rect = np.array([[y - half, x - half], [y - half, x + half], [y + half, x + half], [y + half, x - half]])
        else:
            return
        try:
            cursor_layer = self._viewer.layers[CURSOR_LAYER_NAME]
            cursor_layer.data = [rect]
        except KeyError:
            self._viewer.add_shapes([rect], shape_type="polygon", edge_color="yellow", face_color="transparent", edge_width=1, name=CURSOR_LAYER_NAME)

    def _remove_cursor_layer(self):
        try:
            self._viewer.layers.remove(CURSOR_LAYER_NAME)
        except ValueError:
            pass

    def _eyedropper_callback(self, viewer, event):
        """Sample an n x n patch on click.

        Click        = fresh sample (replaces previous picks).
        Shift+click  = add patch to accumulated samples (widen range).
        Alt+click    = exclude patch colours from the range (narrow range).
        """
        if not self._picker_active:
            return
        if event.button != 1:
            return
        mods = set(event.modifiers)
        if mods - {"Shift", "Alt"}:
            return
        layer = self._get_current_layer()
        if layer is None:
            return
        frame_rgb = self._get_frame_rgb(layer, self._frame_index)
        if frame_rgb is None:
            return

        y, x = self._pos_to_yx(event.position, layer)
        if y is None:
            return
        h, w = frame_rgb.shape[:2]
        if not (0 <= y < h and 0 <= x < w):
            return

        n = self._patch_spin.value()
        half = n // 2
        y0, y1 = max(0, y - half), min(h, y + half + 1)
        x0, x1 = max(0, x - half), min(w, x + half + 1)

        patch_rgb = frame_rgb[y0:y1, x0:x1]
        patch_cs = frame_rgb_to_color_space(patch_rgb, self._current_color_space)
        pixels = patch_cs.reshape(-1, 3).astype(np.float64)

        shift_held = "Shift" in event.modifiers
        alt_held = "Alt" in event.modifiers

        if alt_held:
            self._excluded_pixels.append(pixels)
        elif shift_held:
            self._sampled_pixels.append(pixels)
        else:
            self._sampled_pixels = [pixels]
            self._excluded_pixels.clear()

        if not self._sampled_pixels:
            return
        all_included = np.concatenate(self._sampled_pixels, axis=0)
        all_excluded = np.concatenate(self._excluded_pixels, axis=0) if self._excluded_pixels else None
        lower, upper = self._compute_thresholds_from_samples(all_included, all_excluded)

        cs = self._current_color_space
        tgt = self._current_target
        if cs in self._thresholds and tgt in self._thresholds[cs]:
            self._thresholds[cs][tgt]["lower"] = lower
            self._thresholds[cs][tgt]["upper"] = upper
            self._update_sliders_from_thresholds()
            self._schedule_update()

        params = COLOR_SPACE_PARAMS.get(cs, {})
        ch_names = params.get("channels", ["C0", "C1", "C2"])
        n_inc = len(self._sampled_pixels)
        n_exc = len(self._excluded_pixels)
        info_parts = [f"{ch_names[i]}: {int(lower[i])}\u2013{int(upper[i])}" for i in range(3)]
        mode = f"+{n_inc}"
        if n_exc:
            mode += f" \u2212{n_exc}"
        self._picker_info.setText(f"Picks {mode}, {len(all_included)} px:  " + "  ".join(info_parts))

    def _compute_thresholds_from_samples(self, included: np.ndarray, excluded: np.ndarray | None = None):
        """Derive lower/upper from included pixels, narrowed by excluded pixels.

        1. Start with mean +/- 2*std of included pixels (min spread +/- 5).
        2. If excluded pixels exist, tighten bounds per channel so that the
           excluded region is pushed out:
           - For each channel where the excluded mean is *below* the included
             mean, raise the lower bound to the excluded max + 1.
           - Where it is *above*, lower the upper bound to the excluded min - 1.
           The included mean +/- 2*std range is kept as an outer limit so we
           never accidentally blow the range wider than the included data.
        """
        params = COLOR_SPACE_PARAMS.get(self._current_color_space, {})
        max_vals = np.array(params.get("max_values", [255, 255, 255]), dtype=np.float64)

        inc_mean = included.mean(axis=0)
        inc_std = np.maximum(included.std(axis=0), 5.0)

        lower = np.clip(inc_mean - 2.0 * inc_std, 0, max_vals)
        upper = np.clip(inc_mean + 2.0 * inc_std, 0, max_vals)

        if excluded is not None and len(excluded) > 0:
            exc_mean = excluded.mean(axis=0)
            exc_min = excluded.min(axis=0)
            exc_max = excluded.max(axis=0)
            for ch in range(3):
                if exc_mean[ch] < inc_mean[ch]:
                    lower[ch] = max(lower[ch], exc_max[ch] + 1)
                else:
                    upper[ch] = min(upper[ch], exc_min[ch] - 1)
            lower = np.clip(lower, 0, max_vals)
            upper = np.clip(upper, 0, max_vals)

        return lower.astype(np.uint8), upper.astype(np.uint8)

    # ---- Channel sliders ---------------------------------------------------

    def _build_channel_sliders(self):
        # Clear existing
        for w in self._channel_widgets:
            for k in ("min_slider", "max_slider", "min_spin", "max_spin", "row"):
                if k in w and w[k] is not None:
                    try:
                        w[k].setParent(None)
                    except Exception:
                        pass
        self._channel_widgets.clear()

        params = COLOR_SPACE_PARAMS.get(self._current_color_space, {})
        channels = params.get("channels", ["R", "G", "B"])
        max_vals = params.get("max_values", [255, 255, 255])

        for i, (ch_name, max_v) in enumerate(zip(channels, max_vals)):
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 2, 0, 2)
            row_layout.addWidget(QLabel(ch_name + ":"))
            min_slider = QSlider(Qt.Orientation.Horizontal)
            min_slider.setRange(0, max_v)
            min_slider.valueChanged.connect(lambda v, idx=i: self._on_channel_min_changed(idx, v))
            max_slider = QSlider(Qt.Orientation.Horizontal)
            max_slider.setRange(0, max_v)
            max_slider.valueChanged.connect(lambda v, idx=i: self._on_channel_max_changed(idx, v))
            min_spin = QSpinBox()
            min_spin.setRange(0, max_v)
            min_spin.valueChanged.connect(lambda v, idx=i: self._on_channel_min_changed(idx, v))
            max_spin = QSpinBox()
            max_spin.setRange(0, max_v)
            max_spin.valueChanged.connect(lambda v, idx=i: self._on_channel_max_changed(idx, v))
            row_layout.addWidget(min_slider, 1)
            row_layout.addWidget(min_spin)
            row_layout.addWidget(max_slider, 1)
            row_layout.addWidget(max_spin)
            self._channels_layout.addWidget(row)
            self._channel_widgets.append({"name": ch_name, "max_val": max_v, "min_slider": min_slider, "max_slider": max_slider, "min_spin": min_spin, "max_spin": max_spin, "row": row})
        self._update_sliders_from_thresholds()

    def _update_sliders_from_thresholds(self):
        cs = self._current_color_space
        tgt = self._current_target
        th = self._thresholds.get(cs, {}).get(tgt, {})
        lower = th.get("lower", np.array([0, 0, 0], dtype=np.uint8))
        upper = th.get("upper", np.array([255, 255, 255], dtype=np.uint8))
        if len(lower) < 3:
            lower = np.array([0, 0, 0], dtype=np.uint8)
        if len(upper) < 3:
            upper = np.array([255, 255, 255], dtype=np.uint8)
        for i, w in enumerate(self._channel_widgets):
            lo = int(lower[i]) if i < len(lower) else 0
            hi = int(upper[i]) if i < len(upper) else w["max_val"]
            w["min_slider"].blockSignals(True)
            w["max_slider"].blockSignals(True)
            w["min_spin"].blockSignals(True)
            w["max_spin"].blockSignals(True)
            w["min_slider"].setValue(lo)
            w["max_slider"].setValue(hi)
            w["min_spin"].setValue(lo)
            w["max_spin"].setValue(hi)
            w["min_slider"].blockSignals(False)
            w["max_slider"].blockSignals(False)
            w["min_spin"].blockSignals(False)
            w["max_spin"].blockSignals(False)

    def _on_channel_min_changed(self, channel_index: int, value: int):
        cs, tgt = self._current_color_space, self._current_target
        if cs not in self._thresholds or tgt not in self._thresholds[cs]:
            return
        self._thresholds[cs][tgt]["lower"][channel_index] = value
        if channel_index < len(self._channel_widgets):
            w = self._channel_widgets[channel_index]
            w["min_slider"].blockSignals(True)
            w["min_spin"].blockSignals(True)
            w["min_slider"].setValue(value)
            w["min_spin"].setValue(value)
            w["min_slider"].blockSignals(False)
            w["min_spin"].blockSignals(False)
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _on_channel_max_changed(self, channel_index: int, value: int):
        cs, tgt = self._current_color_space, self._current_target
        if cs not in self._thresholds or tgt not in self._thresholds[cs]:
            return
        self._thresholds[cs][tgt]["upper"][channel_index] = value
        if channel_index < len(self._channel_widgets):
            w = self._channel_widgets[channel_index]
            w["max_slider"].blockSignals(True)
            w["max_spin"].blockSignals(True)
            w["max_slider"].setValue(value)
            w["max_spin"].setValue(value)
            w["max_slider"].blockSignals(False)
            w["max_spin"].blockSignals(False)
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _on_frame_slider_changed(self, value: int):
        self._frame_index = value
        self._frame_spin.blockSignals(True)
        self._frame_spin.setValue(value)
        self._frame_spin.blockSignals(False)
        try:
            ax = self._viewer_time_axis(max_frames=self._frame_slider.maximum() + 1)
            self._viewer.dims.set_current_step(ax, value)
        except Exception:
            pass

    def _on_frame_spin_changed(self, value: int):
        self._frame_index = value
        self._frame_slider.blockSignals(True)
        self._frame_slider.setValue(value)
        self._frame_slider.blockSignals(False)
        try:
            ax = self._viewer_time_axis(max_frames=self._frame_slider.maximum() + 1)
            self._viewer.dims.set_current_step(ax, value)
        except Exception:
            pass

    def _viewer_time_axis(self, max_frames: int | None = None) -> int:
        """Return viewer dim axis corresponding to video time index."""
        if max_frames is None:
            max_frames = self._frame_slider.maximum() + 1
        try:
            steps = tuple(int(x) for x in self._viewer.dims.nsteps)
        except Exception:
            return 0
        if not steps:
            return 0
        match_axes = [i for i, n in enumerate(steps) if n == int(max_frames)]
        if len(match_axes) == 1:
            return int(match_axes[0])
        if int(steps[0]) == int(max_frames):
            return 0
        return int(match_axes[0]) if match_axes else 0

    def _sync_frame_from_viewer(self, total_frames: int | None = None) -> int | None:
        """Sync widget frame controls from the viewer; return new frame index if changed."""
        prev = self._frame_index
        if total_frames is None:
            total_frames = self._frame_slider.maximum() + 1
        step = self._viewer.dims.current_step
        axis = self._viewer_time_axis(max_frames=total_frames)
        if len(step) > axis:
            v = int(step[axis])
            if self._frame_slider.minimum() <= v <= self._frame_slider.maximum():
                self._frame_index = v
                self._frame_slider.blockSignals(True)
                self._frame_spin.blockSignals(True)
                self._frame_slider.setValue(v)
                self._frame_spin.setValue(v)
                self._frame_slider.blockSignals(False)
                self._frame_spin.blockSignals(False)
        return self._frame_index if self._frame_index != prev else None

    def _on_dims_current_step(self, event=None) -> None:
        """Keep frame controls in sync with viewer navigation."""
        self._sync_frame_from_viewer()

    def _schedule_update(self):
        """Debounce: restart the timer so the visible-frame mask updates after changes settle."""
        self._update_timer.start()

    def _get_frame_rgb(self, layer, frame_index: int):
        """Extract a single (H, W, 3) uint8 frame from the layer."""
        try:
            data = layer.data
            shape = getattr(data, "shape", None)
            if shape is None:
                return None
            if len(shape) == 3:
                frame = data
            elif len(shape) == 4:
                frame = data[int(frame_index)]
            else:
                return None
            frame = np.asarray(frame)
        except Exception:
            return None
        if frame.ndim != 3 or frame.shape[-1] < 3:
            return None
        frame = frame[..., :3]
        if np.issubdtype(frame.dtype, np.floating):
            # Support both normalized [0,1] and image-scale [0,255] float data.
            max_v = float(np.nanmax(frame)) if frame.size else 0.0
            if max_v <= 1.0:
                out = np.clip(frame, 0.0, 1.0) * 255.0
            else:
                out = np.clip(frame, 0.0, 255.0)
            return out.astype(np.uint8)
        if np.issubdtype(frame.dtype, np.integer):
            return np.clip(frame, 0, 255).astype(np.uint8)
        return np.asarray(np.clip(frame, 0, 255), dtype=np.uint8)

    def _remove_mask_layer(self):
        name = self._mask_layer_name()
        try:
            self._viewer.layers.remove(name)
        except ValueError:
            pass

    def _reset_to_minmax(self):
        params = COLOR_SPACE_PARAMS.get(self._current_color_space, {})
        max_vals = params.get("max_values", [255, 255, 255])
        cs, tgt = self._current_color_space, self._current_target
        if cs not in self._thresholds or tgt not in self._thresholds[cs]:
            return
        self._thresholds[cs][tgt]["lower"] = np.array([0, 0, 0], dtype=np.uint8)
        self._thresholds[cs][tgt]["upper"] = np.array(max_vals, dtype=np.uint8)
        self._update_sliders_from_thresholds()
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _reset_to_default(self):
        import copy

        cs, tgt = self._current_color_space, self._current_target
        if cs in DEFAULT_THRESHOLDS and tgt in DEFAULT_THRESHOLDS[cs]:
            self._thresholds[cs][tgt] = copy.deepcopy(DEFAULT_THRESHOLDS[cs][tgt])
        self._update_sliders_from_thresholds()
        self._mask_dirty[self._current_target] = True
        self._schedule_update()

    def _apply_current_frame_mask(self):
        """Apply thresholds and write a mask layer aligned to source dimensionality."""
        layer = self._get_current_layer()
        if layer is None:
            return
        tgt = self._current_target
        name = self._mask_layer_name(tgt)
        if self._suppress_mask_autocreate.get(tgt, False) and name not in self._viewer.layers and not self._mask_dirty.get(tgt, False):
            return
        shape = _image_layer_data_shape(layer)
        if not shape:
            return
        try:
            if len(shape) == 4:
                masks = []
                for t in range(int(shape[0])):
                    frame_rgb = self._get_frame_rgb(layer, t)
                    if frame_rgb is None:
                        raise ValueError(f"Could not read frame {t} for mask")
                    mask = apply_thresholds(frame_rgb, self._current_color_space, self._current_target, self._thresholds)
                    masks.append((mask > 0).astype(np.uint8))
                out_mask = np.stack(masks, axis=0).astype(np.uint8)
            elif len(shape) == 3:
                frame_rgb = self._get_frame_rgb(layer, 0)
                if frame_rgb is None:
                    raise ValueError("Could not read frame for mask")
                mask = apply_thresholds(frame_rgb, self._current_color_space, self._current_target, self._thresholds)
                out_mask = (mask > 0).astype(np.uint8)
            else:
                return
        except Exception as exc:
            try:
                from napari.utils.notifications import show_warning

                show_warning(f"Color Thresholding: could not build mask layer: {exc}")
            except Exception:
                pass
            return

        self._mask_dirty[tgt] = False
        try:
            existing = self._viewer.layers[name]
            try:
                existing.data = out_mask
                existing.refresh()
            except Exception:
                self._viewer.layers.remove(existing)
                raise KeyError
        except KeyError:
            self._viewer.add_labels(out_mask, name=name, opacity=0.4, colormap=self._label_colormap(self._current_target))
            self._suppress_mask_autocreate[tgt] = False
        self._record_threshold_step_if_needed()

    def _record_threshold_step_if_needed(self) -> None:
        layer = self._get_current_layer()
        if layer is None:
            return
        cs = self._current_color_space
        tgt = self._current_target
        th = self._thresholds.get(cs, {}).get(tgt, {})
        lower = [int(x) for x in np.asarray(th.get("lower", [0, 0, 0]), dtype=np.uint8).tolist()]
        upper = [int(x) for x in np.asarray(th.get("upper", [255, 255, 255]), dtype=np.uint8).tolist()]
        params = {"source_layer": layer.name, "target": tgt, "color_space": cs, "lower": lower, "upper": upper, "output_mask_layer": self._mask_layer_name(tgt)}
        upsert_pipeline_step(kind="color_thresholding.threshold", description=f"Color Thresholding [{tgt}] {cs.upper()} thresholds on {layer.name}", params=params, match=lambda st: (st.kind == "color_thresholding.threshold" and str((st.params or {}).get("source_layer", "")) == layer.name and str((st.params or {}).get("target", "")) == tgt))

    # ---- Save masks --------------------------------------------------------

    def _source_dir(self) -> str | None:
        """Return the directory of the source video from layer metadata."""
        layer = self._get_current_layer()
        if layer is None:
            return None
        src = layer.metadata.get("source_path")
        if src:
            return str(Path(src).parent)
        return None

    def _source_stem(self) -> str | None:
        """Return the stem (filename without extension) of the source video."""
        layer = self._get_current_layer()
        if layer is None:
            return None
        src = layer.metadata.get("source_path")
        if src:
            return Path(src).stem
        return layer.name

    def _save_masks(self):
        from napari.layers import Labels

        fmt = self._save_fmt_combo.currentData()
        ext = ".tiff" if fmt == "tiff" else ".npy"

        src_dir = self._source_dir()
        src_stem = self._source_stem()

        saved = 0
        for layer in list(self._viewer.layers):
            if not isinstance(layer, Labels):
                continue
            if src_stem and not layer.name.startswith(src_stem):
                continue

            data = np.asarray(layer.data)
            vid = self._get_current_layer()
            vshape = _image_layer_data_shape(vid) if vid is not None else None
            if data.ndim == 2 and vshape is not None and len(vshape) == 4:
                base_name = f"{layer.name} - frame{int(self._frame_index):06d}"
            else:
                base_name = layer.name

            if src_dir:
                out_path = str(Path(src_dir) / (base_name + ext))
            else:
                out_path, _ = QFileDialog.getSaveFileName(self, f"Save {base_name}", base_name + ext, "TIFF (*.tiff)" if fmt == "tiff" else "NumPy (*.npy)")
                if not out_path:
                    continue
            if fmt == "tiff":
                import tifffile

                tifffile.imwrite(out_path, data)
            else:
                np.save(out_path, data)
            saved += 1

        if saved:
            from napari.utils.notifications import show_info

            show_info(f"Saved {saved} mask(s) to {src_dir or 'selected location'}")
        else:
            from napari.utils.notifications import show_warning

            show_warning("No mask layers found to save.")
