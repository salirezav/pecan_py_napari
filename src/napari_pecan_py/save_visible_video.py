"""Save the composited viewer canvas (all visible layers) as an MP4.

Registered on the File menu. Unlike layer-list \"Save video\", this captures
whatever is on screen — opacities, blending, overlays — via napari screenshots.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Iterator

import numpy as np

if TYPE_CHECKING:
    from napari.viewer import Viewer

_ACTION_ID_AS = "napari-pecan-py.save_visible_video_as"
_ACTION_ID = "napari-pecan-py.save_visible_video"
_registered = False

VISIBLE_SAVE_SUFFIX = " - visible"


def _viewer_source_path(viewer: Viewer) -> str | None:
    """Prefer a visible Image layer's on-disk path; else any layer with a path."""
    from napari_pecan_py.save_video import _layer_source_path

    visible_paths: list[str] = []
    any_paths: list[str] = []
    for layer in viewer.layers:
        path = _layer_source_path(layer)
        if not path:
            continue
        any_paths.append(path)
        if getattr(layer, "visible", True):
            visible_paths.append(path)
            if getattr(layer, "_type_string", None) == "image":
                return path
    if visible_paths:
        return visible_paths[0]
    return any_paths[0] if any_paths else None


def playback_axis_and_count(viewer: Viewer) -> tuple[int, int] | None:
    """Return ``(axis, n_frames)`` for the slider axis to play, else ``None``.

    Only considers dims that are not currently displayed on the canvas
    (typically time), so spatial axes are never treated as a frame axis.
    """
    dims = viewer.dims
    nsteps = list(dims.nsteps)

    try:
        candidates = [int(a) for a in dims.not_displayed]
    except Exception:
        candidates = []

    if not candidates:
        try:
            last = int(dims.last_used)
            candidates = [last]
        except Exception:
            candidates = []

    for ax in candidates:
        if 0 <= ax < len(nsteps) and int(nsteps[ax]) > 1:
            return ax, int(nsteps[ax])
    return None


def _process_qt_events() -> None:
    try:
        from qtpy.QtWidgets import QApplication

        app = QApplication.instance()
        if app is not None:
            app.processEvents()
    except Exception:
        pass


def iter_visible_canvas_frames(
    viewer: Viewer,
    *,
    axis: int | None = None,
    n_frames: int | None = None,
) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames of the composited canvas across the time axis.

    Restores ``viewer.dims.current_step`` when finished (including on error).
    """
    info = playback_axis_and_count(viewer)
    if axis is None or n_frames is None:
        if info is None:
            # Single still frame.
            rgba = viewer.screenshot(canvas_only=True, flash=False)
            yield np.ascontiguousarray(np.asarray(rgba)[..., :3])
            return
        axis, n_frames = info

    axis = int(axis)
    n_frames = int(n_frames)
    start = tuple(viewer.dims.current_step)
    try:
        for t in range(n_frames):
            viewer.dims.set_current_step(axis, t)
            _process_qt_events()
            rgba = viewer.screenshot(canvas_only=True, flash=False)
            yield np.ascontiguousarray(np.asarray(rgba)[..., :3])
    finally:
        try:
            viewer.dims.current_step = start
        except Exception:
            pass
        _process_qt_events()


def save_visible_as_video(
    viewer: Viewer,
    path: str | Path,
    *,
    fps: float | None = None,
) -> Path:
    """Screenshot visible layers over time and write an MP4."""
    from napari_pecan_py.save_video import (
        DEFAULT_FPS,
        probe_video_fps,
        write_rgb_frames_to_mp4,
    )

    if fps is None:
        source = _viewer_source_path(viewer)
        fps = (
            probe_video_fps(source)
            if source and Path(source).is_file()
            else float(DEFAULT_FPS)
        )

    frames = iter_visible_canvas_frames(viewer)
    return write_rgb_frames_to_mp4(frames, path, fps=float(fps))


def _suggested_visible_path(viewer: Viewer) -> Path:
    from napari_pecan_py.save_video import default_export_path

    source = _viewer_source_path(viewer)
    if source:
        return default_export_path(source, suffix=VISIBLE_SAVE_SUFFIX)
    return Path.home() / "visible.mp4"


def save_visible_video_as(viewer: Viewer) -> None:
    """File → Save visible video as… (prompts for destination)."""
    from napari.utils.notifications import show_info, show_warning
    from napari_pecan_py.save_video import _prompt_save_path, _qt_parent

    if len(viewer.layers) == 0:
        show_warning("Add a layer before saving the visible video.")
        return

    dest = _prompt_save_path(
        suggested=_suggested_visible_path(viewer),
        parent=_qt_parent(),
    )
    if dest is None:
        return

    try:
        written = save_visible_as_video(viewer, dest)
    except Exception as exc:
        show_warning(f"Failed to save visible video: {exc}")
        return

    info = playback_axis_and_count(viewer)
    n = info[1] if info is not None else 1
    show_info(f"Saved visible video ({n} frame(s)) → {written}")


def save_visible_video(viewer: Viewer) -> None:
    """File → Save visible video (next to source, no dialog)."""
    from napari.utils.notifications import show_info, show_warning

    if len(viewer.layers) == 0:
        show_warning("Add a layer before saving the visible video.")
        return

    source = _viewer_source_path(viewer)
    if not source:
        show_warning(
            "Save visible video requires a layer loaded from disk "
            "(metadata source_path / data.path). Use Save visible video as… instead."
        )
        return

    dest = _suggested_visible_path(viewer)
    try:
        written = save_visible_as_video(viewer, dest)
    except Exception as exc:
        show_warning(f"Failed to save visible video: {exc}")
        return

    info = playback_axis_and_count(viewer)
    n = info[1] if info is not None else 1
    show_info(f"Saved visible video ({n} frame(s)) → {written}")


def register_save_visible_video_actions() -> None:
    """Register visible-canvas save actions on the File menu (idempotent)."""
    global _registered
    if _registered:
        return

    try:
        from app_model.types import Action, MenuRule
        from napari._app_model import get_app_model
        from napari._app_model.constants import MenuGroup, MenuId
        from napari._app_model.context import LayerListContextKeys as LLCK
    except ImportError:
        return

    app = get_app_model()
    if _ACTION_ID_AS in app.commands and _ACTION_ID in app.commands:
        _registered = True
        return

    # Place after napari's screenshot items in the Save group.
    if _ACTION_ID_AS not in app.commands:
        app.register_action(
            Action(
                id=_ACTION_ID_AS,
                title="Save visible video as…",
                tooltip=(
                    "Export the composited canvas (all visible layers, "
                    "opacities, and overlays) as an MP4 over the time axis"
                ),
                callback=save_visible_video_as,
                menus=[
                    MenuRule(
                        id=MenuId.MENUBAR_FILE,
                        group=MenuGroup.SAVE,
                        order=40,
                    )
                ],
                enablement=LLCK.num_layers > 0,
            )
        )

    if _ACTION_ID not in app.commands:
        app.register_action(
            Action(
                id=_ACTION_ID,
                title="Save visible video",
                tooltip=(
                    "Save the composited canvas as MP4 next to the original "
                    f"file with '{VISIBLE_SAVE_SUFFIX}' suffix "
                    "(requires an on-disk source path)"
                ),
                callback=save_visible_video,
                menus=[
                    MenuRule(
                        id=MenuId.MENUBAR_FILE,
                        group=MenuGroup.SAVE,
                        order=41,
                    )
                ],
                enablement=LLCK.num_layers > 0,
            )
        )

    _registered = True
