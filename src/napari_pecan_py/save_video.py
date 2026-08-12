"""Save selected layers as MP4 from the layer-list context menu."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

import numpy as np

if TYPE_CHECKING:
    from napari.components import LayerList
    from napari.layers import Layer

_ACTION_ID_AS = "napari-pecan-py.save_video_as"
_ACTION_ID = "napari-pecan-py.save_video"
_registered = False

DEFAULT_FPS = 30.0
DEFAULT_SAVE_SUFFIX = " - saved"
_SUPPORTED_TYPES = frozenset({"image", "labels", "shapes"})


def _layer_source_path(layer: Layer) -> str | None:
    meta = getattr(layer, "metadata", None) or {}
    path = meta.get("source_path")
    if path:
        return str(path)
    data = getattr(layer, "data", None)
    path = getattr(data, "path", None)
    return str(path) if path else None


def _sanitize_filename(name: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(name)).strip(" .")
    return cleaned or "layer"


def default_export_path(
    source_path: str | Path,
    *,
    layer_name: str | None = None,
    suffix: str = DEFAULT_SAVE_SUFFIX,
) -> Path:
    """Build ``{stem}{suffix}.mp4`` (optionally including layer name) next to source."""
    source = Path(source_path).resolve()
    if layer_name:
        safe = _sanitize_filename(layer_name)
        stem_lower = source.stem.casefold()
        if safe.casefold() != stem_lower and not safe.casefold().startswith(
            stem_lower
        ):
            return source.parent / f"{source.stem}{suffix} - {safe}.mp4"
    return source.parent / f"{source.stem}{suffix}.mp4"


def probe_video_fps(path: str | Path, *, default: float = DEFAULT_FPS) -> float:
    """Read FPS from a video file, falling back to ``default``."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return float(default)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if not np.isfinite(fps) or fps <= 1e-3:
        return float(default)
    return float(fps)


def _layer_fps(layer: Layer) -> float:
    source = _layer_source_path(layer)
    if source and Path(source).is_file():
        return probe_video_fps(source)
    return float(DEFAULT_FPS)


def _to_uint8_rgb_frame(
    frame: np.ndarray,
    *,
    contrast_limits: tuple[float, float] | None = None,
) -> np.ndarray:
    """Normalize a single 2D/RGB(A) frame to ``(H, W, 3)`` uint8."""
    arr = np.asarray(frame)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        rgb = arr[..., :3]
        if rgb.dtype == np.uint8:
            return np.ascontiguousarray(rgb)
        lo, hi = float(np.min(rgb)), float(np.max(rgb))
        if contrast_limits is not None:
            lo, hi = float(contrast_limits[0]), float(contrast_limits[1])
        if hi <= lo:
            return np.zeros(rgb.shape[:2] + (3,), dtype=np.uint8)
        scaled = (np.clip(rgb.astype(np.float32), lo, hi) - lo) / (hi - lo)
        return (scaled * 255.0).astype(np.uint8)

    if arr.ndim != 2:
        raise ValueError(f"Unsupported frame shape {arr.shape}")

    if contrast_limits is not None:
        lo, hi = float(contrast_limits[0]), float(contrast_limits[1])
    else:
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if arr.dtype == np.uint8 and lo == 0 and hi == 255:
            gray = arr
            return np.stack([gray, gray, gray], axis=-1)

    if hi <= lo:
        gray = np.zeros(arr.shape, dtype=np.uint8)
    else:
        scaled = (np.clip(arr.astype(np.float32), lo, hi) - lo) / (hi - lo)
        gray = (scaled * 255.0).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=-1)


def _label_color_rgb(layer: Layer | None, label_id: int) -> tuple[int, int, int]:
    if layer is not None:
        get_color = getattr(layer, "get_color", None)
        if callable(get_color):
            try:
                color = get_color(int(label_id))
                if color is not None:
                    rgba = np.asarray(color, dtype=np.float64).reshape(-1)
                    if rgba.size >= 3:
                        return tuple(
                            int(np.clip(round(float(c) * 255.0), 0, 255))
                            for c in rgba[:3]
                        )
            except Exception:
                pass
    # Stable saturated fallback (golden-ratio hue).
    hue = (int(label_id) * 0.61803398875) % 1.0
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 1.0)
    return (
        int(round(r * 255.0)),
        int(round(g * 255.0)),
        int(round(b * 255.0)),
    )


def labels_frame_to_rgb(
    frame: np.ndarray,
    *,
    layer: Layer | None = None,
) -> np.ndarray:
    """Colorize a 2D labels frame to RGB uint8 (background stays black)."""
    labels = np.asarray(frame)
    if labels.ndim != 2:
        raise ValueError(f"Expected 2D labels frame, got shape {labels.shape}")
    rgb = np.zeros(labels.shape + (3,), dtype=np.uint8)
    for lid in np.unique(labels):
        lid_i = int(lid)
        if lid_i == 0:
            continue
        color = _label_color_rgb(layer, lid_i)
        rgb[labels == lid] = color
    return rgb


def _shapes_to_labels_volume(layer: Layer) -> np.ndarray:
    """Rasterize a Shapes layer to an integer label array."""
    try:
        return np.asarray(layer.to_labels())
    except TypeError:
        pass
    try:
        extent = layer._extent_data
        labels_shape = tuple(
            max(1, int(round(extent[1][i] - extent[0][i])))
            for i in range(len(extent[0]))
        )
        return np.asarray(layer.to_labels(labels_shape=labels_shape))
    except Exception:
        data = getattr(layer, "data", None)
        if data is not None and hasattr(data, "shape"):
            return np.asarray(
                layer.to_labels(labels_shape=tuple(int(x) for x in data.shape))
            )
        raise ValueError(f"Could not rasterize shapes layer {layer.name!r}")


def _image_frame_count_and_indexer(layer: Layer) -> tuple[int, Any]:
    data = layer.data
    shape = tuple(getattr(data, "shape", ()))
    if not shape:
        raise ValueError(f"Layer {layer.name!r} has no data")

    is_rgb = bool(getattr(layer, "rgb", False))
    ndim = len(shape)

    if is_rgb:
        if ndim == 3 and shape[-1] in (3, 4):
            return 1, None  # single frame
        if ndim >= 4:
            return int(shape[0]), 0
        raise ValueError(f"Unsupported RGB image shape {shape}")

    if ndim == 2:
        return 1, None
    if ndim >= 3:
        # (T, H, W) or (H, W, C) — treat trailing 3/4 as channels when not rgb flag.
        if shape[-1] in (3, 4) and ndim == 3:
            return 1, None
        return int(shape[0]), 0

    raise ValueError(f"Unsupported image shape {shape}")


def iter_layer_rgb_frames(layer: Layer) -> Iterator[np.ndarray]:
    """Yield ``(H, W, 3)`` uint8 RGB frames for ``layer``."""
    layer_type = getattr(layer, "_type_string", None)
    if layer_type not in _SUPPORTED_TYPES:
        raise ValueError(
            f"Cannot export layer type {layer_type!r} "
            f"(supported: {', '.join(sorted(_SUPPORTED_TYPES))})"
        )

    if layer_type == "shapes":
        labels = _shapes_to_labels_volume(layer)
        if labels.ndim == 2:
            yield labels_frame_to_rgb(labels, layer=None)
            return
        if labels.ndim >= 3:
            for t in range(int(labels.shape[0])):
                yield labels_frame_to_rgb(labels[t], layer=None)
            return
        raise ValueError(f"Unexpected shapes raster shape {labels.shape}")

    if layer_type == "labels":
        data = layer.data
        shape = tuple(data.shape)
        if len(shape) == 2:
            yield labels_frame_to_rgb(np.asarray(data), layer=layer)
            return
        if len(shape) >= 3:
            for t in range(int(shape[0])):
                yield labels_frame_to_rgb(np.asarray(data[t]), layer=layer)
            return
        raise ValueError(f"Unexpected labels shape {shape}")

    # Image
    n_frames, time_axis = _image_frame_count_and_indexer(layer)
    contrast = getattr(layer, "contrast_limits", None)
    clim = tuple(contrast) if contrast is not None else None
    data = layer.data
    if time_axis is None:
        yield _to_uint8_rgb_frame(np.asarray(data), contrast_limits=clim)
        return
    for t in range(n_frames):
        yield _to_uint8_rgb_frame(np.asarray(data[t]), contrast_limits=clim)


def write_rgb_frames_to_mp4(
    frames: Iterator[np.ndarray] | np.ndarray,
    path: str | Path,
    *,
    fps: float = DEFAULT_FPS,
) -> Path:
    """Encode RGB frames to an MP4 file via OpenCV. Returns the written path."""
    import cv2

    out_path = Path(path)
    if out_path.suffix.lower() != ".mp4":
        out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, np.ndarray):
        frame_iter: Iterator[np.ndarray] = (frames[t] for t in range(frames.shape[0]))
    else:
        frame_iter = iter(frames)

    writer = None
    size = None
    count = 0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    try:
        for frame in frame_iter:
            rgb = np.ascontiguousarray(_to_uint8_rgb_frame(frame))
            h, w = int(rgb.shape[0]), int(rgb.shape[1])
            if writer is None:
                size = (w, h)
                writer = cv2.VideoWriter(
                    str(out_path), fourcc, float(fps), size
                )
                if not writer.isOpened():
                    raise RuntimeError(f"Could not open VideoWriter for {out_path}")
            elif (w, h) != size:
                raise ValueError(
                    f"Frame size changed from {size} to {(w, h)} while writing {out_path}"
                )
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
            count += 1
    finally:
        if writer is not None:
            writer.release()

    if count == 0:
        if out_path.exists():
            out_path.unlink(missing_ok=True)
        raise ValueError("No frames to write")
    return out_path


def save_layer_as_video(
    layer: Layer,
    path: str | Path,
    *,
    fps: float | None = None,
) -> Path:
    """Rasterize ``layer`` if needed and write an MP4 to ``path``."""
    use_fps = float(fps) if fps is not None else _layer_fps(layer)
    return write_rgb_frames_to_mp4(
        iter_layer_rgb_frames(layer),
        path,
        fps=use_fps,
    )


def _qt_parent():
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        return app.activeWindow() if app is not None else None
    except Exception:
        return None


def _prompt_save_path(*, suggested: Path | None = None, parent=None) -> Path | None:
    from qtpy.QtWidgets import QFileDialog

    start = str(suggested) if suggested is not None else str(Path.home())
    path, _filter = QFileDialog.getSaveFileName(
        parent,
        "Save video as",
        start,
        "MP4 video (*.mp4);;All files (*)",
    )
    if not path:
        return None
    out = Path(path)
    if out.suffix.lower() != ".mp4":
        out = out.with_suffix(".mp4")
    return out


def _prompt_save_directory(*, suggested: Path | None = None, parent=None) -> Path | None:
    from qtpy.QtWidgets import QFileDialog

    start = str(suggested) if suggested is not None else str(Path.home())
    path = QFileDialog.getExistingDirectory(parent, "Save videos to folder", start)
    if not path:
        return None
    return Path(path)


def _exportable_layers(selected: list[Layer]) -> list[Layer]:
    out: list[Layer] = []
    for layer in selected:
        if getattr(layer, "_type_string", None) in _SUPPORTED_TYPES:
            out.append(layer)
    return out


def save_selected_layers_as(ll: LayerList) -> None:
    """Prompt for destination and save selected layer(s) as MP4."""
    from napari.utils.notifications import show_info, show_warning

    selected = list(ll.selection)
    if not selected:
        show_warning("Select a layer to save as video.")
        return

    layers = _exportable_layers(selected)
    if not layers:
        show_warning(
            "Selected layer(s) cannot be saved as video "
            "(need Image, Labels, or Shapes)."
        )
        return

    parent = _qt_parent()
    saved: list[str] = []

    if len(layers) == 1:
        layer = layers[0]
        source = _layer_source_path(layer)
        suggested = (
            default_export_path(source, layer_name=str(layer.name))
            if source
            else Path.home() / f"{_sanitize_filename(layer.name)}.mp4"
        )
        dest = _prompt_save_path(suggested=suggested, parent=parent)
        if dest is None:
            return
        try:
            written = save_layer_as_video(layer, dest)
            saved.append(f"{layer.name} → {written}")
        except Exception as exc:
            show_warning(f"Failed to save {layer.name}: {exc}")
            return
    else:
        first_source = next(
            (Path(_layer_source_path(layer)).parent for layer in layers if _layer_source_path(layer)),
            Path.home(),
        )
        folder = _prompt_save_directory(suggested=first_source, parent=parent)
        if folder is None:
            return
        used_names: set[str] = set()
        for layer in layers:
            base = _sanitize_filename(layer.name)
            name = base
            n = 2
            while name.casefold() in used_names:
                name = f"{base} ({n})"
                n += 1
            used_names.add(name.casefold())
            dest = folder / f"{name}.mp4"
            try:
                written = save_layer_as_video(layer, dest)
                saved.append(f"{layer.name} → {written}")
            except Exception as exc:
                show_warning(f"Failed to save {layer.name}: {exc}")

    if saved:
        show_info("Saved video(s):\n" + "\n".join(saved))
    else:
        show_warning("No videos were saved.")


def save_selected_layers(ll: LayerList) -> None:
    """Save selected layer(s) next to their on-disk source with a fixed suffix."""
    from napari.utils.notifications import show_info, show_warning

    selected = list(ll.selection)
    if not selected:
        show_warning("Select a layer to save as video.")
        return

    layers = _exportable_layers(selected)
    if not layers:
        show_warning(
            "Selected layer(s) cannot be saved as video "
            "(need Image, Labels, or Shapes)."
        )
        return

    with_path = [(layer, _layer_source_path(layer)) for layer in layers]
    savable = [(layer, path) for layer, path in with_path if path]
    skipped = [layer.name for layer, path in with_path if not path]

    if skipped:
        show_warning(
            "Skipped layer(s) with no on-disk source path: " + ", ".join(skipped)
        )
    if not savable:
        show_warning(
            "Save video requires a layer loaded from disk "
            "(metadata source_path / data.path)."
        )
        return

    # Disambiguate when several layers share the same default path.
    planned: list[tuple[Layer, Path]] = []
    claimed: dict[str, int] = {}
    for layer, source in savable:
        include_name = sum(1 for _, s in savable if s == source) > 1
        dest = default_export_path(
            source,
            layer_name=str(layer.name) if include_name else None,
        )
        key = str(dest.resolve())
        if key in claimed:
            claimed[key] += 1
            dest = dest.with_name(
                f"{dest.stem} ({claimed[key]}){dest.suffix}"
            )
            key = str(dest.resolve())
        claimed[key] = 1
        planned.append((layer, dest))

    saved: list[str] = []
    for layer, dest in planned:
        try:
            written = save_layer_as_video(layer, dest)
            saved.append(f"{layer.name} → {written}")
        except Exception as exc:
            show_warning(f"Failed to save {layer.name}: {exc}")

    if saved:
        show_info("Saved video(s):\n" + "\n".join(saved))
    else:
        show_warning("No videos were saved.")


def register_save_video_actions() -> None:
    """Register Save video / Save video as… on the layer-list context menu."""
    global _registered
    if _registered:
        return

    try:
        from app_model.types import Action, MenuRule
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari._app_model.context import LayerListSelectionContextKeys as LLSCK
    except ImportError:
        return

    app = get_app_model()
    if _ACTION_ID_AS in app.commands and _ACTION_ID in app.commands:
        _registered = True
        return

    if _ACTION_ID_AS not in app.commands:
        app.register_action(
            Action(
                id=_ACTION_ID_AS,
                title="Save video as…",
                tooltip=(
                    "Export the selected layer(s) as MP4 "
                    "(rasterizes Labels/Shapes to frames)"
                ),
                callback=save_selected_layers_as,
                menus=[
                    MenuRule(
                        id=MenuId.LAYERLIST_CONTEXT,
                        group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                        order=21,
                    )
                ],
                enablement=LLSCK.num_selected_layers >= 1,
            )
        )

    if _ACTION_ID not in app.commands:
        app.register_action(
            Action(
                id=_ACTION_ID,
                title="Save video",
                tooltip=(
                    "Save selected layer(s) as MP4 next to the original file "
                    f"with '{DEFAULT_SAVE_SUFFIX}' suffix "
                    "(requires an on-disk source path)"
                ),
                callback=save_selected_layers,
                menus=[
                    MenuRule(
                        id=MenuId.LAYERLIST_CONTEXT,
                        group=MenuGroup.LAYERLIST_CONTEXT.SPLIT_MERGE,
                        order=22,
                    )
                ],
                enablement=LLSCK.num_selected_layers >= 1,
            )
        )

    _registered = True
