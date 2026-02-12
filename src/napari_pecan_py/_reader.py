"""Napari reader for video files (.mp4, etc.) using pecan_py."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from napari.types import LayerData

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}


def get_reader(path: str | list[str]):
    """Return a reader function for supported video paths (npe2 reader protocol).

    Napari calls this with the path of a dropped file; we return a callable
    that napari then invokes with the path(s) to get layer data.
    """
    paths = [path] if isinstance(path, str) else path
    if not paths or Path(paths[0]).suffix not in VIDEO_EXTENSIONS:
        return None

    def _reader(path_or_paths: str | list[str]) -> list[LayerData]:
        from pecan_py import BaseVideo

        paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
        layer_data: list[LayerData] = []
        for p in paths:
            p = str(Path(p).resolve())
            if Path(p).suffix not in VIDEO_EXTENSIONS:
                continue
            try:
                video = BaseVideo(p)
                if video.frames.size == 0:
                    continue
                frames = video.frames
                # OpenCV loads as BGR; napari expects RGB.
                # Convert BGR->RGB for 3-channel data.
                if frames.ndim == 4 and frames.shape[-1] == 3:
                    frames = frames[..., ::-1]
                # (data, metadata, layer_type); napari Image expects (T, Y, X) or (T, Y, X, C)
                layer_data.append((frames, {"name": video.name}, "image"))
            except Exception:
                continue
        return layer_data

    return _reader
