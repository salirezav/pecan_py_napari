"""Parallelism metadata and stack segmentation for adjustment stacks."""

from __future__ import annotations

import copy
from typing import Literal

SegmentKind = Literal["parallel", "barrier"]

# Ops that need the full (T,H,W,3) volume passed as video_rgb when applied.
TYPES_NEEDING_VIDEO = frozenset({"temporal_median_diff", "frame_diff"})

# Ops that must run on the full volume before frame-level parallelism can resume.
TEMPORAL_BARRIER_TYPES = frozenset({"temporal_median_diff"})

# Per-frame ops safe to run in parallel across disjoint frame index ranges.
FRAME_PARALLEL_TYPES = frozenset(
    {
        "brightness_contrast",
        "levels",
        "curves",
        "surface_blur",
        "normalization",
        "denoise",
        "motion_mask_threshold",
        "mask_morphology",
        "mask_largest_component",
        "frame_diff",
    }
)


def _enabled_adjustments(stack: list[dict] | None) -> list[dict]:
    if not stack:
        return []
    out: list[dict] = []
    for adj in stack:
        if not isinstance(adj, dict):
            continue
        if not adj.get("enabled", True):
            continue
        out.append(adj)
    return out


def is_parallelizable(adj: dict) -> bool:
    """Return whether an adjustment may run in a frame-parallel segment.

    Unknown types are treated as barriers for safety.
    """
    typ = adj.get("type")
    if typ in TEMPORAL_BARRIER_TYPES:
        return False
    return typ in FRAME_PARALLEL_TYPES


def stack_needs_video_context(stack: list[dict] | None) -> bool:
    """True when any enabled adjustment needs the full time series."""
    for adj in _enabled_adjustments(stack):
        if adj.get("type") in TYPES_NEEDING_VIDEO:
            return True
    return False


def segment_needs_video_context(segment_stack: list[dict]) -> bool:
    """True when a segment's substack needs video_rgb for any frame."""
    return stack_needs_video_context(segment_stack)


def stamp_parallelizable_flags(stack: list[dict]) -> list[dict]:
    """Deep-copy stack and set ``parallelizable`` from the registry (YAML visibility)."""
    out: list[dict] = []
    for adj in stack:
        if not isinstance(adj, dict):
            continue
        item = copy.deepcopy(adj)
        item["parallelizable"] = bool(is_parallelizable(item))
        out.append(item)
    return out


def split_stack_into_segments(stack: list[dict]) -> list[tuple[SegmentKind, list[dict]]]:
    """Split an adjustment stack into parallel and barrier segments.

  Parallel segments contain only frame-parallel ops and may be applied across
  worker threads on disjoint frame ranges. Barrier segments contain temporal
  barrier ops (e.g. temporal_median_diff) and run on the full volume.

  Disabled adjustments are omitted. Adjacent barriers are separate segments.
  """
    enabled = _enabled_adjustments(stack)
    if not enabled:
        return []

    segments: list[tuple[SegmentKind, list[dict]]] = []
    parallel_buf: list[dict] = []

    def flush_parallel() -> None:
        nonlocal parallel_buf
        if parallel_buf:
            segments.append(("parallel", parallel_buf))
            parallel_buf = []

    for adj in enabled:
        if is_parallelizable(adj):
            parallel_buf.append(adj)
        else:
            flush_parallel()
            segments.append(("barrier", [adj]))

    flush_parallel()
    return segments
