"""Batch pipeline application logic."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from napari.layers import Image, Labels, Shapes

from ..pipeline_recorder.logic import create_apply_context
from ..pipeline_recorder.state import PipelineStep


def _yaml_available() -> bool:
    try:
        import yaml  # noqa: F401

        return True
    except Exception:
        return False


def load_pipeline_file(path: str | Path) -> tuple[list[dict], str]:
    """Load pipeline steps from a YAML/JSON file.

    Returns enabled steps as dicts and the source file name.
    """
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".json":
        raw = json.loads(txt)
    else:
        if not _yaml_available():
            raise RuntimeError("PyYAML is not installed. Load a .json file or install pyyaml.")
        import yaml

        raw = yaml.safe_load(txt)
    steps_raw = list((raw or {}).get("steps", []))
    steps = [PipelineStep.from_dict(x) for x in steps_raw if isinstance(x, dict)]
    enabled = [step.to_dict() for step in steps if step.enabled]
    if not enabled:
        raise ValueError("Pipeline has no enabled steps.")
    return enabled, p.name


def clear_viewer_layers(viewer) -> None:
    names = [layer.name for layer in viewer.layers]
    for name in names:
        try:
            viewer.layers.remove(viewer.layers[name])
        except (KeyError, ValueError):
            continue


def _lazy_video_metadata(path: str, frames) -> dict:
    from napari_pecan_py._reader import _TARGET_CHUNK_BYTES

    return {
        "source_path": path,
        "lazy_enabled": True,
        "lazy_chunks_mb": int(_TARGET_CHUNK_BYTES / (1024 * 1024)),
        "frames_per_chunk": int(frames._frames_per_chunk),
    }


def _open_lazy_video(video_path: str | Path) -> tuple[str, str, object]:
    from napari_pecan_py._reader import LazyVideoArray

    path = str(Path(video_path).resolve())
    frames = LazyVideoArray(path)
    return path, Path(path).stem, frames


class _HeadlessLayerList:
    """Minimal napari LayerList stand-in for off-screen pipeline execution."""

    def __init__(self) -> None:
        self._by_name: dict[str, Image | Labels | Shapes] = {}
        self.selection = SimpleNamespace(active=None)

    def __iter__(self):
        return iter(self._by_name.values())

    def __len__(self) -> int:
        return len(self._by_name)

    def __getitem__(self, name: str):
        return self._by_name[str(name)]

    def __contains__(self, name: str) -> bool:
        return str(name) in self._by_name

    def _add(self, layer: Image | Labels | Shapes) -> Image | Labels | Shapes:
        self._by_name[str(layer.name)] = layer
        return layer


class HeadlessViewer:
    """In-memory viewer used to run pipelines without touching the napari UI."""

    def __init__(self) -> None:
        self.layers = _HeadlessLayerList()

    def add_image(self, data, *, name: str, metadata=None, colormap=None):
        layer = Image(data, name=name, metadata=dict(metadata or {}))
        if colormap is not None:
            layer.colormap = colormap
        return self.layers._add(layer)

    def add_labels(self, data, *, name: str):
        return self.layers._add(Labels(data, name=name))

    def add_shapes(
        self,
        data,
        *,
        name: str,
        shape_type="ellipse",
        face_color="transparent",
    ):
        return self.layers._add(
            Shapes(data, name=name, shape_type=shape_type, face_color=face_color)
        )


def load_video_into_headless_viewer(headless_viewer: HeadlessViewer, video_path: str | Path) -> str:
    """Register one lazy-loaded video in an off-screen viewer. Returns layer name."""
    path, name, frames = _open_lazy_video(video_path)
    headless_viewer.add_image(frames, name=name, metadata=_lazy_video_metadata(path, frames))
    return name


def create_headless_apply_context(video_path: str | Path):
    """Build a pipeline apply context that never touches the real napari viewer."""
    viewer = HeadlessViewer()
    load_video_into_headless_viewer(viewer, video_path)
    return create_apply_context(viewer)


def load_video_into_viewer(viewer, video_path: str | Path) -> str:
    """Replace viewer contents with a single lazy-loaded video layer."""
    clear_viewer_layers(viewer)
    path, name, frames = _open_lazy_video(video_path)
    viewer.add_image(
        frames,
        name=name,
        metadata=_lazy_video_metadata(path, frames),
    )
    return name
