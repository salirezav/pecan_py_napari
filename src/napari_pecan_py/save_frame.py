"""Save the current frame from a layer or the composited visible view.

Exports native-resolution RGB from layer data (no napari canvas screenshot,
so no black letterbox borders). Layer-list and File-menu actions are registered
separately.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from napari.components import LayerList
    from napari.layers import Layer
    from napari.viewer import Viewer

_LAYER_ACTION_ID = "napari-pecan-py.save_frame"
_LAYER_ACTION_ID_AS = "napari-pecan-py.save_frame_as"
_VISIBLE_ACTION_ID = "napari-pecan-py.save_visible_frame"
_VISIBLE_ACTION_ID_AS = "napari-pecan-py.save_visible_frame_as"
_registered_layer = False
_registered_visible = False

FRAME_SAVE_SUFFIX = " - frame"
VISIBLE_FRAME_SAVE_SUFFIX = " - visible_frame"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
_SUPPORTED_TYPES = frozenset({"image", "labels", "shapes"})


def _current_viewer() -> Viewer | None:
    try:
        import napari

        return napari.current_viewer()
    except Exception:
        return None


def current_frame_index(layer: Layer, viewer: Viewer | None = None) -> int:
    """Viewer time index for ``layer``, or ``0`` when unknown / 2D."""
    viewer = viewer or _current_viewer()
    data = getattr(layer, "data", None)
    if data is None:
        return 0
    if viewer is None:
        return 0
    from napari_pecan_py.widgets.pecan_ellipse.logic import (
        resolve_time_index_for_volume,
    )

    try:
        return int(resolve_time_index_for_volume(data, viewer))
    except Exception:
        return 0


def layer_rgb_at_frame(layer: Layer, frame_index: int) -> np.ndarray:
    """Return ``(H, W, 3)`` uint8 RGB for one layer at ``frame_index``."""
    from napari_pecan_py.save_video import (
        _image_frame_count_and_indexer,
        _shapes_to_labels_volume,
        _to_uint8_rgb_frame,
        labels_frame_to_rgb,
    )

    layer_type = getattr(layer, "_type_string", None)
    if layer_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Cannot export layer type {layer_type!r} "
            f"(supported: {', '.join(sorted(_SUPPORTED_TYPES))})"
        )

    t = int(frame_index)

    if layer_type == "shapes":
        labels = _shapes_to_labels_volume(layer)
        if labels.ndim == 2:
            return labels_frame_to_rgb(labels, layer=None)
        if labels.ndim >= 3:
            t = int(np.clip(t, 0, labels.shape[0] - 1))
            return labels_frame_to_rgb(labels[t], layer=None)
        raise ValueError(f"Unexpected shapes raster shape {labels.shape}")

    if layer_type == "labels":
        data = layer.data
        shape = tuple(data.shape)
        if len(shape) == 2:
            return labels_frame_to_rgb(np.asarray(data), layer=layer)
        if len(shape) >= 3:
            t = int(np.clip(t, 0, shape[0] - 1))
            return labels_frame_to_rgb(np.asarray(data[t]), layer=layer)
        raise ValueError(f"Unexpected labels shape {shape}")

    n_frames, time_axis = _image_frame_count_and_indexer(layer)
    contrast = getattr(layer, "contrast_limits", None)
    clim = tuple(contrast) if contrast is not None else None
    data = layer.data
    if time_axis is None:
        return _to_uint8_rgb_frame(np.asarray(data), contrast_limits=clim)
    t = int(np.clip(t, 0, max(0, n_frames - 1)))
    return _to_uint8_rgb_frame(np.asarray(data[t]), contrast_limits=clim)


def layer_rgba_at_frame(layer: Layer, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return RGB uint8 and alpha float32 in ``[0, 1]`` for compositing."""
    rgb = layer_rgb_at_frame(layer, frame_index)
    opacity = float(getattr(layer, "opacity", 1.0) or 0.0)
    opacity = float(np.clip(opacity, 0.0, 1.0))
    layer_type = getattr(layer, "_type_string", None)

    if layer_type in {"labels", "shapes"}:
        # Transparent background (label 0 / empty); opaque where content exists.
        alpha = (np.any(rgb > 0, axis=-1).astype(np.float32)) * opacity
    else:
        alpha = np.full(rgb.shape[:2], opacity, dtype=np.float32)
    return rgb, alpha


def _resize_rgb_alpha(
    rgb: np.ndarray,
    alpha: np.ndarray,
    height: int,
    width: int,
    *,
    nearest: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if rgb.shape[0] == height and rgb.shape[1] == width:
        return rgb, alpha
    import cv2

    interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
    rgb_out = cv2.resize(rgb, (width, height), interpolation=interp)
    alpha_out = cv2.resize(alpha, (width, height), interpolation=interp)
    return rgb_out, alpha_out.astype(np.float32)


def _infer_spatial_hw(layer: Layer) -> tuple[int, int] | None:
    """Return ``(H, W)`` from layer metadata/shape without rendering a frame."""
    layer_type = getattr(layer, "_type_string", None)
    if layer_type == "shapes":
        try:
            extent = layer._extent_data
            # extent rows: min/max; displayed spatial axes are typically last 2.
            mins = np.asarray(extent[0], dtype=float).reshape(-1)
            maxs = np.asarray(extent[1], dtype=float).reshape(-1)
            if mins.size >= 2 and maxs.size >= 2:
                h = max(1, int(round(maxs[-2] - mins[-2])))
                w = max(1, int(round(maxs[-1] - mins[-1])))
                return h, w
        except Exception:
            pass
        return None

    data = getattr(layer, "data", None)
    shape = tuple(getattr(data, "shape", ()) or ())
    if not shape:
        return None

    is_rgb = bool(getattr(layer, "rgb", False))
    ndim = len(shape)
    if layer_type == "image":
        if is_rgb:
            if ndim == 3 and shape[-1] in (3, 4):
                return int(shape[0]), int(shape[1])
            if ndim >= 4:
                return int(shape[1]), int(shape[2])
            return None
        if ndim == 2:
            return int(shape[0]), int(shape[1])
        if ndim == 3 and shape[-1] in (3, 4):
            return int(shape[0]), int(shape[1])
        if ndim >= 3:
            return int(shape[1]), int(shape[2])
        return None

    if layer_type == "labels":
        if ndim == 2:
            return int(shape[0]), int(shape[1])
        if ndim >= 3:
            return int(shape[1]), int(shape[2])
    return None


def _layer_spatial_hw(layer: Layer, frame_index: int) -> tuple[int, int] | None:
    hw = _infer_spatial_hw(layer)
    if hw is not None:
        return hw
    try:
        rgb = layer_rgb_at_frame(layer, frame_index)
    except Exception:
        return None
    return int(rgb.shape[0]), int(rgb.shape[1])


def _canvas_hw_for_viewer(viewer: Viewer, frame_index: int = 0) -> tuple[int, int]:
    """Prefer a visible Image layer size; else the largest visible layer."""
    best_img: tuple[int, int] | None = None
    best_any: tuple[int, int] | None = None
    for layer in viewer.layers:
        if not getattr(layer, "visible", True):
            continue
        if getattr(layer, "_type_string", None) not in _SUPPORTED_TYPES:
            continue
        hw = _layer_spatial_hw(layer, frame_index)
        if hw is None:
            continue
        if best_any is None or hw[0] * hw[1] > best_any[0] * best_any[1]:
            best_any = hw
        if getattr(layer, "_type_string", None) == "image" and best_img is None:
            best_img = hw
    if best_img is not None:
        return best_img
    if best_any is not None:
        return best_any
    raise ValueError("No visible Image/Labels/Shapes layers to composite")


def _labels_color_lut(layer: Layer | None, max_id: int) -> np.ndarray:
    """Build an ``(max_id + 1, 3)`` uint8 LUT for fast labels→RGB."""
    from napari_pecan_py.save_video import _label_color_rgb

    n = max(0, int(max_id)) + 1
    lut = np.zeros((n, 3), dtype=np.uint8)
    for lid in range(1, n):
        lut[lid] = _label_color_rgb(layer, lid)
    return lut


def _labels_to_rgb_fast(labels: np.ndarray, lut: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    ids = np.clip(labels.astype(np.int64, copy=False), 0, lut.shape[0] - 1)
    return lut[ids]


def composite_visible_frame(
    viewer: Viewer,
    *,
    frame_index: int | None = None,
    canvas_hw: tuple[int, int] | None = None,
) -> np.ndarray:
    """Alpha-composite visible layers at native resolution (no canvas borders).

    Layer order follows napari's list (index 0 = bottom). Label/shape background
    stays transparent so underlying video shows through at the layer opacity.
    """
    if frame_index is None:
        info_layer = next(
            (
                ly
                for ly in viewer.layers
                if getattr(ly, "visible", True)
                and getattr(ly, "_type_string", None) in _SUPPORTED_TYPES
            ),
            None,
        )
        frame_index = current_frame_index(info_layer, viewer) if info_layer else 0

    t = int(frame_index)
    if canvas_hw is None:
        height, width = _canvas_hw_for_viewer(viewer, t)
    else:
        height, width = int(canvas_hw[0]), int(canvas_hw[1])
    out = np.zeros((height, width, 3), dtype=np.float32)

    for layer in viewer.layers:
        if not getattr(layer, "visible", True):
            continue
        if getattr(layer, "_type_string", None) not in _SUPPORTED_TYPES:
            continue
        try:
            rgb, alpha = layer_rgba_at_frame(layer, t)
        except Exception:
            continue
        nearest = getattr(layer, "_type_string", None) in {"labels", "shapes"}
        rgb, alpha = _resize_rgb_alpha(rgb, alpha, height, width, nearest=nearest)
        a = alpha[..., None]
        out = rgb.astype(np.float32) * a + out * (1.0 - a)

    return np.clip(out, 0, 255).astype(np.uint8)


def iter_composited_visible_frames(
    viewer: Viewer,
    *,
    n_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield native-resolution composites for ``0 … n_frames-1`` from layer data.

    Does not scrub viewer dims or take canvas screenshots.
    """
    from napari_pecan_py.save_video import (
        _image_frame_count_and_indexer,
        _shapes_to_labels_volume,
        _to_uint8_rgb_frame,
    )

    visible = [
        ly
        for ly in viewer.layers
        if getattr(ly, "visible", True)
        and getattr(ly, "_type_string", None) in _SUPPORTED_TYPES
    ]
    if not visible:
        raise ValueError("No visible Image/Labels/Shapes layers to composite")

    if n_frames is None:
        # Prefer slider length; else longest visible stack.
        try:
            from napari_pecan_py.save_visible_video import playback_axis_and_count

            info = playback_axis_and_count(viewer)
        except Exception:
            info = None
        if info is not None:
            n_frames = int(info[1])
        else:
            lengths = []
            for ly in visible:
                lt = getattr(ly, "_type_string", None)
                if lt == "image":
                    n, _ax = _image_frame_count_and_indexer(ly)
                    lengths.append(int(n))
                elif lt == "labels":
                    shape = tuple(ly.data.shape)
                    lengths.append(1 if len(shape) == 2 else int(shape[0]))
                else:
                    lengths.append(1)
            n_frames = max(lengths) if lengths else 1

    n_frames = max(1, int(n_frames))
    height, width = _canvas_hw_for_viewer(viewer, 0)

    # Prepare per-layer sources once (shapes rasterized once; label LUTs once).
    prepared: list[dict] = []
    for layer in visible:
        lt = getattr(layer, "_type_string", None)
        opacity = float(np.clip(float(getattr(layer, "opacity", 1.0) or 0.0), 0.0, 1.0))
        entry: dict = {"layer": layer, "type": lt, "opacity": opacity, "nearest": lt in {"labels", "shapes"}}

        if lt == "shapes":
            labels_vol = _shapes_to_labels_volume(layer)
            max_id = int(np.max(labels_vol)) if labels_vol.size else 0
            entry["labels_vol"] = labels_vol
            entry["lut"] = _labels_color_lut(None, max_id)
        elif lt == "labels":
            data = layer.data
            shape = tuple(data.shape)
            entry["data"] = data
            entry["is_2d"] = len(shape) == 2
            try:
                sample = np.asarray(data if entry["is_2d"] else data[0])
                max_id = int(np.max(sample)) if sample.size else 0
                # Include a few more IDs if present later; expand lazily if needed.
                if not entry["is_2d"] and shape[0] > 1:
                    max_id = max(max_id, int(np.max(np.asarray(data[min(1, shape[0] - 1)]))))
            except Exception:
                max_id = 255
            entry["lut"] = _labels_color_lut(layer, max(max_id, 255))
        else:
            entry["data"] = layer.data
            entry["contrast"] = getattr(layer, "contrast_limits", None)
            entry["n_frames"], entry["time_axis"] = _image_frame_count_and_indexer(layer)

        prepared.append(entry)

    for t in range(n_frames):
        out = np.zeros((height, width, 3), dtype=np.float32)
        for entry in prepared:
            lt = entry["type"]
            opacity = entry["opacity"]
            try:
                if lt == "shapes":
                    vol = entry["labels_vol"]
                    if vol.ndim == 2:
                        labels = vol
                    else:
                        ti = int(np.clip(t, 0, vol.shape[0] - 1))
                        labels = vol[ti]
                    max_needed = int(np.max(labels)) if labels.size else 0
                    if max_needed >= entry["lut"].shape[0]:
                        entry["lut"] = _labels_color_lut(None, max_needed)
                    rgb = _labels_to_rgb_fast(labels, entry["lut"])
                    alpha = (labels > 0).astype(np.float32) * opacity
                elif lt == "labels":
                    if entry["is_2d"]:
                        labels = np.asarray(entry["data"])
                    else:
                        ti = int(np.clip(t, 0, entry["data"].shape[0] - 1))
                        labels = np.asarray(entry["data"][ti])
                    max_needed = int(np.max(labels)) if labels.size else 0
                    if max_needed >= entry["lut"].shape[0]:
                        entry["lut"] = _labels_color_lut(entry["layer"], max_needed)
                    rgb = _labels_to_rgb_fast(labels, entry["lut"])
                    alpha = (labels > 0).astype(np.float32) * opacity
                else:
                    clim = tuple(entry["contrast"]) if entry["contrast"] is not None else None
                    if entry["time_axis"] is None:
                        frame = np.asarray(entry["data"])
                    else:
                        ti = int(np.clip(t, 0, max(0, entry["n_frames"] - 1)))
                        frame = np.asarray(entry["data"][ti])
                    rgb = _to_uint8_rgb_frame(frame, contrast_limits=clim)
                    alpha = np.full(rgb.shape[:2], opacity, dtype=np.float32)
            except Exception:
                continue

            rgb, alpha = _resize_rgb_alpha(
                rgb, alpha, height, width, nearest=bool(entry["nearest"])
            )
            a = alpha[..., None]
            out = rgb.astype(np.float32) * a + out * (1.0 - a)

        yield np.clip(out, 0, 255).astype(np.uint8)


def write_rgb_image(rgb: np.ndarray, path: str | Path) -> Path:
    """Write an RGB uint8 image; format from the path suffix (default PNG)."""
    import cv2

    out = Path(path)
    if out.suffix.lower() not in _IMAGE_EXTENSIONS:
        out = out.with_suffix(".png")
    out.parent.mkdir(parents=True, exist_ok=True)
    arr = np.ascontiguousarray(rgb)
    if arr.ndim != 3 or arr.shape[-1] not in (3, 4):
        raise ValueError(f"Expected HxWx3 RGB, got shape {arr.shape}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(arr[..., :3], cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(out), bgr):
        raise RuntimeError(f"Could not write image to {out}")
    return out


def default_frame_export_path(
    source_path: str | Path,
    frame_index: int,
    *,
    layer_name: str | None = None,
    suffix: str = FRAME_SAVE_SUFFIX,
    ext: str = ".png",
) -> Path:
    """``{stem}{suffix}{frame:06d}[. - name].png`` next to the source file."""
    from napari_pecan_py.save_video import _sanitize_filename

    source = Path(source_path).resolve()
    ext = ext if ext.startswith(".") else f".{ext}"
    base = f"{source.stem}{suffix}{int(frame_index):06d}"
    if layer_name:
        safe = _sanitize_filename(layer_name)
        stem_lower = source.stem.casefold()
        if safe.casefold() != stem_lower and not safe.casefold().startswith(stem_lower):
            base = f"{base} - {safe}"
    return source.parent / f"{base}{ext}"


def _prompt_image_save_path(*, suggested: Path | None = None, parent=None) -> Path | None:
    from qtpy.QtWidgets import QFileDialog

    start = str(suggested) if suggested is not None else str(Path.home() / "frame.png")
    path, selected = QFileDialog.getSaveFileName(
        parent,
        "Save frame as",
        start,
        "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;TIFF image (*.tif *.tiff);;All files (*)",
    )
    if not path:
        return None
    out = Path(path)
    if out.suffix.lower() not in _IMAGE_EXTENSIONS:
        # Infer from filter when the user omitted an extension.
        sel = (selected or "").lower()
        if "jpeg" in sel or "jpg" in sel:
            out = out.with_suffix(".jpg")
        elif "tif" in sel:
            out = out.with_suffix(".tif")
        else:
            out = out.with_suffix(".png")
    return out


def _viewer_source_path(viewer: Viewer) -> str | None:
    from napari_pecan_py.save_visible_video import _viewer_source_path as _vsp

    return _vsp(viewer)


# --- Layer-list actions -----------------------------------------------------


def save_layer_frame(layer: Layer, path: str | Path, *, frame_index: int | None = None) -> Path:
    viewer = _current_viewer()
    t = int(frame_index) if frame_index is not None else current_frame_index(layer, viewer)
    rgb = layer_rgb_at_frame(layer, t)
    return write_rgb_image(rgb, path)


def save_selected_frames(ll: LayerList) -> None:
    """Layer context → Save frame (next to source, no dialog)."""
    from napari.utils.notifications import show_info, show_warning
    from napari_pecan_py.save_video import _layer_source_path

    selected = list(ll.selection)
    if not selected:
        show_warning("Select a layer to save a frame.")
        return

    layers = [ly for ly in selected if getattr(ly, "_type_string", None) in _SUPPORTED_TYPES]
    if not layers:
        show_warning("Selected layer(s) cannot be saved as a frame (need Image, Labels, or Shapes).")
        return

    viewer = _current_viewer()
    saved: list[str] = []
    skipped: list[str] = []
    for layer in layers:
        source = _layer_source_path(layer)
        if not source:
            skipped.append(layer.name)
            continue
        t = current_frame_index(layer, viewer)
        include_name = len(layers) > 1
        dest = default_frame_export_path(
            source,
            t,
            layer_name=str(layer.name) if include_name else None,
            suffix=FRAME_SAVE_SUFFIX,
        )
        try:
            written = save_layer_frame(layer, dest, frame_index=t)
            saved.append(f"{layer.name}[t={t}] → {written}")
        except Exception as exc:
            show_warning(f"Failed to save frame for {layer.name}: {exc}")

    if skipped:
        show_warning(
            "Skipped layer(s) with no on-disk source path: " + ", ".join(skipped)
        )
    if saved:
        show_info("Saved frame(s):\n" + "\n".join(saved))
    elif not skipped:
        show_warning("No frames were saved.")


def save_selected_frames_as(ll: LayerList) -> None:
    """Layer context → Save frame as…"""
    from napari.utils.notifications import show_info, show_warning
    from napari_pecan_py.save_video import (
        _layer_source_path,
        _prompt_save_directory,
        _qt_parent,
        _sanitize_filename,
    )

    selected = list(ll.selection)
    if not selected:
        show_warning("Select a layer to save a frame.")
        return

    layers = [ly for ly in selected if getattr(ly, "_type_string", None) in _SUPPORTED_TYPES]
    if not layers:
        show_warning("Selected layer(s) cannot be saved as a frame (need Image, Labels, or Shapes).")
        return

    viewer = _current_viewer()
    parent = _qt_parent()
    saved: list[str] = []

    if len(layers) == 1:
        layer = layers[0]
        t = current_frame_index(layer, viewer)
        source = _layer_source_path(layer)
        suggested = (
            default_frame_export_path(source, t, suffix=FRAME_SAVE_SUFFIX)
            if source
            else Path.home() / f"{_sanitize_filename(layer.name)}_frame{t:06d}.png"
        )
        dest = _prompt_image_save_path(suggested=suggested, parent=parent)
        if dest is None:
            return
        try:
            written = save_layer_frame(layer, dest, frame_index=t)
            saved.append(f"{layer.name}[t={t}] → {written}")
        except Exception as exc:
            show_warning(f"Failed to save frame for {layer.name}: {exc}")
            return
    else:
        first_source = next(
            (
                Path(_layer_source_path(ly)).parent
                for ly in layers
                if _layer_source_path(ly)
            ),
            Path.home(),
        )
        folder = _prompt_save_directory(suggested=first_source, parent=parent)
        if folder is None:
            return
        used: set[str] = set()
        for layer in layers:
            t = current_frame_index(layer, viewer)
            base = f"{_sanitize_filename(layer.name)}_frame{t:06d}"
            name = base
            n = 2
            while name.casefold() in used:
                name = f"{base} ({n})"
                n += 1
            used.add(name.casefold())
            dest = folder / f"{name}.png"
            try:
                written = save_layer_frame(layer, dest, frame_index=t)
                saved.append(f"{layer.name}[t={t}] → {written}")
            except Exception as exc:
                show_warning(f"Failed to save frame for {layer.name}: {exc}")

    if saved:
        show_info("Saved frame(s):\n" + "\n".join(saved))
    else:
        show_warning("No frames were saved.")


# --- File-menu visible composite actions ------------------------------------


def save_visible_frame(
    viewer: Viewer,
    path: str | Path,
    *,
    frame_index: int | None = None,
) -> Path:
    rgb = composite_visible_frame(viewer, frame_index=frame_index)
    return write_rgb_image(rgb, path)


def _visible_frame_index(viewer: Viewer) -> int:
    for layer in viewer.layers:
        if getattr(layer, "visible", True) and getattr(layer, "_type_string", None) in _SUPPORTED_TYPES:
            return current_frame_index(layer, viewer)
    return 0


def _suggested_visible_frame_path(viewer: Viewer) -> Path:
    t = _visible_frame_index(viewer)
    source = _viewer_source_path(viewer)
    if source:
        return default_frame_export_path(
            source, t, suffix=VISIBLE_FRAME_SAVE_SUFFIX
        )
    return Path.home() / f"visible_frame{t:06d}.png"


def save_visible_frame_action(viewer: Viewer) -> None:
    """File → Save frame (composited visible layers, no dialog)."""
    from napari.utils.notifications import show_info, show_warning

    if len(viewer.layers) == 0:
        show_warning("Add a layer before saving a frame.")
        return
    source = _viewer_source_path(viewer)
    if not source:
        show_warning(
            "Save frame requires a layer loaded from disk "
            "(metadata source_path / data.path). Use Save frame as… instead."
        )
        return
    t = _visible_frame_index(viewer)
    dest = _suggested_visible_frame_path(viewer)
    try:
        written = save_visible_frame(viewer, dest, frame_index=t)
    except Exception as exc:
        show_warning(f"Failed to save visible frame: {exc}")
        return
    show_info(f"Saved visible frame [t={t}] → {written}")


def save_visible_frame_as_action(viewer: Viewer) -> None:
    """File → Save frame as… (composited visible layers)."""
    from napari.utils.notifications import show_info, show_warning
    from napari_pecan_py.save_video import _qt_parent

    if len(viewer.layers) == 0:
        show_warning("Add a layer before saving a frame.")
        return
    t = _visible_frame_index(viewer)
    dest = _prompt_image_save_path(
        suggested=_suggested_visible_frame_path(viewer),
        parent=_qt_parent(),
    )
    if dest is None:
        return
    try:
        written = save_visible_frame(viewer, dest, frame_index=t)
    except Exception as exc:
        show_warning(f"Failed to save visible frame: {exc}")
        return
    show_info(f"Saved visible frame [t={t}] → {written}")


def register_save_frame_actions() -> None:
    """Register layer-list and File-menu save-frame actions (idempotent)."""
    global _registered_layer, _registered_visible

    try:
        from app_model.types import Action, MenuRule
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari._app_model.context import LayerListContextKeys as LLCK
        from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
    except ImportError:
        return

    app = get_app_model()

    if not _registered_layer:
        if _LAYER_ACTION_ID_AS not in app.commands:
            app.register_action(
                Action(
                    id=_LAYER_ACTION_ID_AS,
                    title="Save frame as…",
                    tooltip=(
                        "Save the current time point of the selected layer(s) "
                        "as an image (native resolution, no canvas borders)"
                    ),
                    callback=save_selected_frames_as,
                    menus=[
                        MenuRule(
                            id=MenuId.LAYERLIST_CONTEXT,
                            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                            order=23,
                        )
                    ],
                    enablement=LLSCK.num_selected_layers >= 1,
                )
            )
        if _LAYER_ACTION_ID not in app.commands:
            app.register_action(
                Action(
                    id=_LAYER_ACTION_ID,
                    title="Save frame",
                    tooltip=(
                        "Save the current time point next to the original file "
                        f"as '{{stem}}{FRAME_SAVE_SUFFIX}NNNNNN.png' "
                        "(requires an on-disk source path)"
                    ),
                    callback=save_selected_frames,
                    menus=[
                        MenuRule(
                            id=MenuId.LAYERLIST_CONTEXT,
                            group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                            order=24,
                        )
                    ],
                    enablement=LLSCK.num_selected_layers >= 1,
                )
            )
        _registered_layer = True

    if not _registered_visible:
        if _VISIBLE_ACTION_ID_AS not in app.commands:
            app.register_action(
                Action(
                    id=_VISIBLE_ACTION_ID_AS,
                    title="Save frame as…",
                    tooltip=(
                        "Save a native-resolution composite of all visible "
                        "layers at the current time (opacities included; "
                        "no canvas black borders)"
                    ),
                    callback=save_visible_frame_as_action,
                    menus=[
                        MenuRule(
                            id=MenuId.MENUBAR_FILE,
                            group=MenuGroup.SAVE,
                            order=42,
                        )
                    ],
                    enablement=LLCK.num_layers > 0,
                )
            )
        if _VISIBLE_ACTION_ID not in app.commands:
            app.register_action(
                Action(
                    id=_VISIBLE_ACTION_ID,
                    title="Save frame",
                    tooltip=(
                        "Save a composited visible frame next to the original "
                        f"file with '{VISIBLE_FRAME_SAVE_SUFFIX}' suffix "
                        "(requires an on-disk source path)"
                    ),
                    callback=save_visible_frame_action,
                    menus=[
                        MenuRule(
                            id=MenuId.MENUBAR_FILE,
                            group=MenuGroup.SAVE,
                            order=43,
                        )
                    ],
                    enablement=LLCK.num_layers > 0,
                )
            )
        _registered_visible = True
