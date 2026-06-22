"""Napari reader for video files (.mp4, etc.) with lazy frame chunks."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from napari.types import LayerData

try:
    from napari_pecan_py._menu_groups import _ensure_npe2_register_hook

    _ensure_npe2_register_hook()
except ImportError:
    pass

VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}
_TARGET_CHUNK_BYTES = 100 * 1024 * 1024  # ~100MiB in-memory frame chunks
_MAX_CHUNKS_IN_CACHE = 3


class LazyVideoArray:
    """Array-like video adapter that lazily loads/caches frame chunks."""

    def __init__(
        self,
        path: str,
        target_chunk_bytes: int = _TARGET_CHUNK_BYTES,
        max_chunks_in_cache: int = _MAX_CHUNKS_IN_CACHE,
    ) -> None:
        self.path = str(path)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {self.path}")
        try:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:
            cap.release()

        if frame_count <= 0 or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid video metadata for {Path(self.path).name}: "
                f"frames={frame_count}, width={width}, height={height}"
            )

        self._frame_count = frame_count
        self._height = height
        self._width = width
        self._channels = 3
        self._dtype = np.uint8

        bytes_per_frame = self._height * self._width * self._channels
        self._frames_per_chunk = max(1, int(target_chunk_bytes // max(1, bytes_per_frame)))
        self._max_chunks = max(1, int(max_chunks_in_cache))
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = RLock()

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return (self._frame_count, self._height, self._width, self._channels)

    @property
    def ndim(self) -> int:
        return 4

    @property
    def dtype(self):
        return np.dtype(self._dtype)

    def _chunk_bounds(self, chunk_index: int) -> tuple[int, int]:
        start = chunk_index * self._frames_per_chunk
        stop = min(start + self._frames_per_chunk, self._frame_count)
        return start, stop

    def _read_chunk(self, chunk_index: int) -> np.ndarray:
        start, stop = self._chunk_bounds(chunk_index)
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            raise ValueError(f"Could not reopen video for chunk read: {self.path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            frames: list[np.ndarray] = []
            needed = stop - start
            for _ in range(needed):
                ok, frm = cap.read()
                if not ok or frm is None:
                    break
                # OpenCV decodes BGR; napari expects RGB.
                frm = cv2.cvtColor(frm, cv2.COLOR_BGR2RGB)
                frames.append(np.asarray(frm, dtype=np.uint8))
        finally:
            cap.release()
        if not frames:
            raise ValueError(f"Failed to decode frames [{start}:{stop}) from {self.path}")
        return np.stack(frames, axis=0)

    def _get_chunk(self, chunk_index: int) -> np.ndarray:
        with self._lock:
            arr = self._cache.get(chunk_index)
            if arr is not None:
                self._cache.move_to_end(chunk_index)
                return arr
            arr = self._read_chunk(chunk_index)
            self._cache[chunk_index] = arr
            self._cache.move_to_end(chunk_index)
            while len(self._cache) > self._max_chunks:
                self._cache.popitem(last=False)
            return arr

    def _get_frame(self, frame_index: int) -> np.ndarray:
        if frame_index < 0:
            frame_index += self._frame_count
        if frame_index < 0 or frame_index >= self._frame_count:
            raise IndexError(f"frame index out of range: {frame_index}")
        chunk_index = frame_index // self._frames_per_chunk
        chunk = self._get_chunk(chunk_index)
        chunk_start, _ = self._chunk_bounds(chunk_index)
        local_index = frame_index - chunk_start
        if local_index >= chunk.shape[0]:
            raise IndexError(f"decoded chunk shorter than expected at frame {frame_index}")
        return chunk[local_index]

    def __getitem__(self, item):
        # Fast path for common napari access pattern layer.data[t]
        if isinstance(item, (int, np.integer)):
            return self._get_frame(int(item))

        if isinstance(item, tuple):
            if len(item) == 0:
                return self
            time_sel = item[0]
            rest = item[1:]
        else:
            time_sel = item
            rest = ()

        if isinstance(time_sel, (int, np.integer)):
            out = self._get_frame(int(time_sel))
            return out[rest] if rest else out

        if isinstance(time_sel, slice):
            indices = list(range(*time_sel.indices(self._frame_count)))
        elif time_sel is Ellipsis:
            indices = list(range(self._frame_count))
        else:
            # Handles numpy/list indexing (e.g. dims-generated integer arrays).
            indices = [int(i) for i in np.asarray(time_sel).tolist()]

        frames = [self._get_frame(i) for i in indices]
        stacked = (
            np.stack(frames, axis=0)
            if frames
            else np.empty((0, self._height, self._width, self._channels), dtype=np.uint8)
        )
        return stacked[(slice(None),) + rest] if rest else stacked


def get_reader(path: str | list[str]):
    """Return a reader function for supported video paths (npe2 reader protocol).

    Napari calls this with the path of a dropped file; we return a callable
    that napari then invokes with the path(s) to get layer data.
    """
    paths = [path] if isinstance(path, str) else path
    if not paths or Path(paths[0]).suffix not in VIDEO_EXTENSIONS:
        return None

    def _reader(path_or_paths: str | list[str]) -> list[LayerData]:
        import time
        import tifffile
        from napari.utils.notifications import show_info, show_warning

        paths = [path_or_paths] if isinstance(path_or_paths, str) else path_or_paths
        layer_data: list[LayerData] = []
        for p in paths:
            p = str(Path(p).resolve())
            if Path(p).suffix not in VIDEO_EXTENSIONS:
                continue
            try:
                show_info(f"Loading video: {Path(p).name}")
                t0 = time.perf_counter()
                frames = LazyVideoArray(p)
                # (data, metadata, layer_type); napari Image expects (T, Y, X) or (T, Y, X, C)
                meta = {
                    "name": Path(p).stem,
                    "metadata": {
                        "source_path": p,
                        "lazy_enabled": True,
                        "lazy_chunks_mb": int(_TARGET_CHUNK_BYTES / (1024 * 1024)),
                        "frames_per_chunk": int(frames._frames_per_chunk),
                    },
                }
                layer_data.append((frames, meta, "image"))
                dt = time.perf_counter() - t0
                show_info(
                    f"Loaded {Path(p).name}: {int(frames.shape[0])} frames in {dt:.1f}s"
                )

                # Optionally load any saved mask files next to the video
                mask_dir = Path(p).parent
                stem = Path(p).stem
                mask_paths = sorted(
                    q
                    for q in mask_dir.glob(f"{stem} - *")
                    if q.suffix.lower() in {".tiff", ".tif", ".npy"}
                )
                if not mask_paths:
                    continue

                load_masks = True
                try:
                    from qtpy.QtWidgets import QApplication, QMessageBox

                    app = QApplication.instance()
                    parent = app.activeWindow() if app is not None else None
                    msg = (
                        f"Found {len(mask_paths)} saved mask file(s) next to\n"
                        f"{Path(p).name}.\n\nLoad them as Labels layers?"
                    )
                    resp = QMessageBox.question(
                        parent,
                        "Load saved masks?",
                        msg,
                        QMessageBox.Yes | QMessageBox.No,
                        QMessageBox.Yes,
                    )
                    load_masks = resp == QMessageBox.Yes
                except Exception:
                    # If Qt is not available (headless), just auto-load.
                    load_masks = True

                if not load_masks:
                    continue

                for mp in mask_paths:
                    try:
                        if mp.suffix.lower() in {".tiff", ".tif"}:
                            data = tifffile.imread(mp)
                        else:
                            data = np.load(mp)
                        layer_data.append((data, {"name": mp.stem}, "labels"))
                    except Exception:
                        continue
            except Exception as exc:
                show_warning(f"Failed to read {Path(p).name}: {exc}")
                continue
        return layer_data

    return _reader
