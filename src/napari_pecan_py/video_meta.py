"""Per-video sidecar metadata (persisted trim / frame range).

Stored next to the video as ``{stem}.pecan.json``. Inclusive 0-based
``frame_range`` indexes into the *original* video file. Sidecar TIFF masks
next to the video are expected to already match the trimmed length
(``end - start + 1``), not the full native frame count.
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


def set_saved_frame_range(
    video_path: str | Path,
    start: int,
    end: int,
    *,
    native_frame_count: int | None = None,
) -> Path:
    """Write / update the sidecar frame range (inclusive, absolute into the file)."""
    start = int(start)
    end = int(end)
    if start < 0 or end < start:
        raise ValueError(f"Invalid frame range [{start}, {end}]")
    if native_frame_count is not None and end >= int(native_frame_count):
        raise ValueError(
            f"Frame range end {end} exceeds native frame count {native_frame_count}"
        )

    meta = load_pecan_meta(video_path) or {}
    meta["frame_range"] = {"start": start, "end": end}
    if native_frame_count is not None:
        meta["native_frame_count"] = int(native_frame_count)
    video_path = Path(video_path).resolve()
    meta["source_path"] = str(video_path)
    return save_pecan_meta(video_path, meta)


def clear_saved_frame_range(video_path: str | Path) -> bool:
    """Remove ``frame_range`` from the sidecar (delete file if nothing else remains)."""
    path = pecan_meta_path(video_path)
    meta = load_pecan_meta(video_path)
    if not meta:
        return False
    meta.pop("frame_range", None)
    # Drop bookkeeping keys that only exist for the range.
    if "frame_range" not in meta:
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
    """Open a ``LazyVideoArray``, applying a saved frame range when present."""
    from napari_pecan_py._reader import LazyVideoArray

    path = str(Path(video_path).resolve())
    if not apply_saved_range:
        return LazyVideoArray(path)

    fr = get_saved_frame_range(path)
    if fr is None:
        return LazyVideoArray(path)
    start, end = fr
    return LazyVideoArray(path, frame_start=start, frame_end=end)
