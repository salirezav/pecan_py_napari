"""Per-video sidecar metadata (persisted trim / frame selection).

Stored next to the video as ``{stem}.pecan.json``. Indexes are inclusive
0-based into the *original* video file.

Modes:
- ``frame_range``: contiguous ``{start, end}``
- ``frame_sample``: strided ``{start, step, count}`` (every ``step`` frames,
  up to ``count`` frames or until the video ends)

Sidecar TIFF masks next to the video should match the effective trimmed length.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

META_VERSION = 1
META_SUFFIX = ".pecan.json"


def pecan_meta_path(video_path: str | Path) -> Path:
    """Return the sidecar path for ``video_path`` (``stem.pecan.json``)."""
    video_path = Path(video_path)
    return video_path.with_name(f"{video_path.stem}{META_SUFFIX}")


def load_pecan_meta(video_path: str | Path) -> dict[str, Any] | None:
    path = pecan_meta_path(video_path)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def save_pecan_meta(video_path: str | Path, meta: dict[str, Any]) -> Path:
    path = pecan_meta_path(video_path)
    payload = {"version": META_VERSION, **dict(meta)}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def compute_sample_indices(
    n_frames: int,
    start: int,
    step: int,
    count: int,
) -> list[int]:
    """Return absolute indices ``start, start+step, …`` capped by ``count`` / length."""
    start = int(start)
    step = int(step)
    count = int(count)
    if start < 0 or start >= n_frames:
        raise ValueError(f"Invalid sample start {start} for length {n_frames}")
    if step < 1:
        raise ValueError(f"Sample step must be >= 1, got {step}")
    if count < 1:
        raise ValueError(f"Sample count must be >= 1, got {count}")
    indices: list[int] = []
    i = start
    while i < n_frames and len(indices) < count:
        indices.append(i)
        i += step
    return indices


def get_saved_frame_range(video_path: str | Path) -> tuple[int, int] | None:
    """Return inclusive ``(start, end)`` from the sidecar, or ``None`` if absent."""
    meta = load_pecan_meta(video_path)
    if not meta:
        return None
    fr = meta.get("frame_range")
    if not isinstance(fr, dict):
        return None
    if "start" not in fr or "end" not in fr:
        return None
    start = int(fr["start"])
    end = int(fr["end"])
    if start < 0 or end < start:
        return None
    return start, end


def get_saved_frame_sample(
    video_path: str | Path,
) -> tuple[int, int, int] | None:
    """Return ``(start, step, count)`` from the sidecar, or ``None`` if absent."""
    meta = load_pecan_meta(video_path)
    if not meta:
        return None
    fs = meta.get("frame_sample")
    if not isinstance(fs, dict):
        return None
    if not all(k in fs for k in ("start", "step", "count")):
        return None
    start = int(fs["start"])
    step = int(fs["step"])
    count = int(fs["count"])
    if start < 0 or step < 1 or count < 1:
        return None
    return start, step, count


def set_saved_frame_range(
    video_path: str | Path,
    start: int,
    end: int,
    *,
    native_frame_count: int | None = None,
) -> Path:
    """Write / update the sidecar contiguous frame range."""
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame range [{start}, {end}]")
    if native_frame_count is not None and end >= int(native_frame_count):
        raise ValueError(
            f"Frame range end {end} exceeds native frame count {native_frame_count}"
        )

    meta = load_pecan_meta(video_path) or {}
    meta.pop("frame_sample", None)
    meta["frame_range"] = {"start": start, "end": end}
    if native_frame_count is not None:
        meta["native_frame_count"] = int(native_frame_count)
    video_path = Path(video_path).resolve()
    meta["source_path"] = str(video_path)
    return save_pecan_meta(video_path, meta)


def set_saved_frame_sample(
    video_path: str | Path,
    start: int,
    step: int,
    count: int,
    *,
    native_frame_count: int | None = None,
) -> Path:
    """Write / update the sidecar strided sample ``start`` / ``step`` / ``count``."""
    start = int(start)
    step = int(step)
    count = int(count)
    if start < 0 or step < 1 or count < 1:
        raise ValueError(
            f"Invalid frame sample start={start}, step={step}, count={count}"
        )
    if native_frame_count is not None and start >= int(native_frame_count):
        raise ValueError(
            f"Sample start {start} exceeds native frame count {native_frame_count}"
        )

    meta = load_pecan_meta(video_path) or {}
    meta.pop("frame_range", None)
    meta["frame_sample"] = {"start": start, "step": step, "count": count}
    if native_frame_count is not None:
        meta["native_frame_count"] = int(native_frame_count)
    video_path = Path(video_path).resolve()
    meta["source_path"] = str(video_path)
    return save_pecan_meta(video_path, meta)


def clear_saved_frame_range(video_path: str | Path) -> bool:
    """Remove trim keys from the sidecar (delete file if nothing else remains)."""
    path = pecan_meta_path(video_path)
    meta = load_pecan_meta(video_path)
    if not meta:
        return False
    meta.pop("frame_range", None)
    meta.pop("frame_sample", None)
    if "frame_range" not in meta and "frame_sample" not in meta:
        meta.pop("native_frame_count", None)
    leftover = {
        k: v
        for k, v in meta.items()
        if k not in {"version", "source_path"} and v is not None
    }
    if not leftover:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            return False
        return True
    save_pecan_meta(video_path, meta)
    return True


def open_lazy_video(video_path: str | Path, *, apply_saved_range: bool = True):
    """Open a ``LazyVideoArray``, applying a saved trim/sample when present."""
    from napari_pecan_py._reader import LazyVideoArray

    path = str(Path(video_path).resolve())
    if not apply_saved_range:
        return LazyVideoArray(path)

    sample = get_saved_frame_sample(path)
    if sample is not None:
        start, step, count = sample
        # Need native length to build indices; open full then wrap.
        full = LazyVideoArray(path)
        indices = compute_sample_indices(
            full._native_frame_count, start, step, count
        )
        return LazyVideoArray(path, frame_indices=indices)

    fr = get_saved_frame_range(path)
    if fr is None:
        return LazyVideoArray(path)
    start, end = fr
    return LazyVideoArray(path, frame_start=start, frame_end=end)
