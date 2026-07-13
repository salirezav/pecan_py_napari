#!/usr/bin/env python3
"""Refine pecan / kernel / crack training masks (no napari UI).

Two input formats are supported:

**separate** — three mask TIFFs per video (`` - Pecan``, `` - Kernel``, `` - Crack``),
listed in a text file (one path per line).

**combined** — one multi-class label TIFF per video (`` - Pecan,Kernel,Crack.tiff``),
discovered recursively under a directory. Class IDs: 3=Pecan, 2=Kernel, 1=Crack.

For both formats:

1. Clip kernel and crack to the **nut region** (union of pecan, kernel, and crack).
2. Where kernel and crack overlap, keep kernel only.
3. Save in place (pecan pixels / files are left unchanged).

Examples::

    python refine_training_masks.py separate --file-list "files to use for training.txt"

    python refine_training_masks.py combined "D:\\VIDS & MASKS"

    python refine_training_masks.py combined ./masks --dry-run
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage

_REPO_SRC = Path(__file__).resolve().parent / "src"
_repo_src = str(_REPO_SRC)
while _repo_src in sys.path:
    sys.path.remove(_repo_src)
sys.path.insert(0, _repo_src)

from napari_pecan_py.widgets.yolo_seg.model import load_mask_volume, save_mask_volume  # noqa: E402

CLASS_SUFFIX_RE = re.compile(r" - (Pecan|Kernel|Crack)$", re.IGNORECASE)

COMBINED_LABEL_PECAN = 3
COMBINED_LABEL_KERNEL = 2
COMBINED_LABEL_CRACK = 1
COMBINED_MASK_STEM_SUFFIX = " - Pecan,Kernel,Crack"
_COMBINED_MASK_STEM_SUFFIXES = (
    COMBINED_MASK_STEM_SUFFIX,
    " - Pecan, Crack, Kernel",
    " - Pecan, Kernel, Crack",
)
_COMBINED_CLASS_TOKENS = frozenset({"pecan", "kernel", "crack"})


def parse_mask_path(path: Path) -> tuple[str, str] | None:
    """Return (video_key, class_name) for a mask file path."""
    match = CLASS_SUFFIX_RE.search(path.stem)
    if match is None:
        return None
    class_name = match.group(1).capitalize()
    video_key = str(path.parent / path.stem[: match.start()])
    return video_key, class_name


def group_mask_files(
    paths: list[Path],
) -> tuple[dict[str, dict[str, Path]], list[Path]]:
    groups: dict[str, dict[str, Path]] = defaultdict(dict)
    skipped: list[Path] = []
    for path in paths:
        parsed = parse_mask_path(path)
        if parsed is None:
            skipped.append(path)
            continue
        video_key, class_name = parsed
        groups[video_key][class_name] = path
    return dict(groups), skipped


def as_frame_stack(volume: np.ndarray) -> np.ndarray:
    arr = np.asarray(volume)
    if arr.ndim == 2:
        return arr[np.newaxis, ...]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected 2D or 3D mask volume, got shape {arr.shape}")


def _nut_foreground(
    pecan: np.ndarray,
    kernel: np.ndarray | None,
    crack: np.ndarray | None,
) -> np.ndarray:
    """Union mask for the whole nut (shell + crack + kernel), not shell-only."""
    nut_fg = pecan > 0
    if kernel is not None:
        nut_fg = nut_fg | (kernel > 0)
    if crack is not None:
        nut_fg = nut_fg | (crack > 0)
    return nut_fg


def refine_frame(
    pecan: np.ndarray,
    kernel: np.ndarray | None,
    crack: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Clip kernel/crack to the nut region and resolve kernel/crack overlap."""
    nut_fg = _nut_foreground(pecan, kernel, crack)

    if kernel is not None:
        kernel = np.where(nut_fg, kernel, 0).astype(kernel.dtype, copy=False)
    if crack is not None:
        crack = np.where(nut_fg, crack, 0).astype(crack.dtype, copy=False)
    if kernel is not None and crack is not None:
        crack = np.where(kernel > 0, 0, crack).astype(crack.dtype, copy=False)

    return kernel, crack


def refine_video_masks(
    pecan_path: Path,
    kernel_path: Path | None,
    crack_path: Path | None,
    *,
    save: bool = True,
) -> dict[str, int]:
    pecan_raw = load_mask_volume(pecan_path)
    pecan_vol = as_frame_stack(pecan_raw)

    kernel_raw = load_mask_volume(kernel_path) if kernel_path else None
    crack_raw = load_mask_volume(crack_path) if crack_path else None
    kernel_vol = as_frame_stack(kernel_raw) if kernel_raw is not None else None
    crack_vol = as_frame_stack(crack_raw) if crack_raw is not None else None

    n_frames = pecan_vol.shape[0]
    if kernel_vol is not None and kernel_vol.shape[0] != n_frames:
        raise ValueError(
            f"Frame count mismatch: pecan={n_frames}, kernel={kernel_vol.shape[0]} ({kernel_path})"
        )
    if crack_vol is not None and crack_vol.shape[0] != n_frames:
        raise ValueError(
            f"Frame count mismatch: pecan={n_frames}, crack={crack_vol.shape[0]} ({crack_path})"
        )

    stats = {
        "frames": n_frames,
        "kernel_clipped_px": 0,
        "crack_clipped_px": 0,
        "crack_removed_for_kernel_px": 0,
    }

    out_kernel_frames: list[np.ndarray] = []
    out_crack_frames: list[np.ndarray] = []

    for t in range(n_frames):
        pecan = pecan_vol[t]
        kernel = kernel_vol[t] if kernel_vol is not None else None
        crack = crack_vol[t] if crack_vol is not None else None

        if kernel is not None:
            nut_fg = _nut_foreground(pecan, kernel, crack)
            stats["kernel_clipped_px"] += int(np.count_nonzero((kernel > 0) & ~nut_fg))
        if crack is not None:
            nut_fg = _nut_foreground(pecan, kernel, crack)
            stats["crack_clipped_px"] += int(np.count_nonzero((crack > 0) & ~nut_fg))
            if kernel is not None:
                stats["crack_removed_for_kernel_px"] += int(
                    np.count_nonzero((crack > 0) & (kernel > 0))
                )

        kernel, crack = refine_frame(pecan, kernel, crack)

        if kernel is not None:
            out_kernel_frames.append(kernel)
        if crack is not None:
            out_crack_frames.append(crack)

    if save:
        if kernel_path is not None and out_kernel_frames:
            kernel_out = (
                out_kernel_frames[0]
                if np.ndim(kernel_raw) == 2
                else np.stack(out_kernel_frames, axis=0)
            )
            save_mask_volume(kernel_out, kernel_path, "tiff")
        if crack_path is not None and out_crack_frames:
            crack_out = (
                out_crack_frames[0]
                if np.ndim(crack_raw) == 2
                else np.stack(out_crack_frames, axis=0)
            )
            save_mask_volume(crack_out, crack_path, "tiff")

    return stats


def is_combined_mask_stem(stem: str) -> bool:
    """True when ``stem`` looks like a combined pecan/kernel/crack label TIFF."""
    if any(stem.endswith(suffix) for suffix in _COMBINED_MASK_STEM_SUFFIXES):
        return True
    if " - " not in stem:
        return False
    suffix_part = stem.rsplit(" - ", 1)[1]
    tokens = {
        token.strip().lower()
        for token in re.split(r"[,]+", suffix_part)
        if token.strip()
    }
    return _COMBINED_CLASS_TOKENS.issubset(tokens)


def discover_combined_mask_files(root: Path) -> list[Path]:
    """Find multi-class label TIFFs under ``root``."""
    paths: list[Path] = []
    for pattern in ("*.tiff", "*.tif", "*.TIFF", "*.TIF"):
        for path in root.rglob(pattern):
            if is_combined_mask_stem(path.stem):
                paths.append(path)
    return sorted(paths, key=lambda p: str(p).lower())


def combined_pecan_clip_mask(labels: np.ndarray) -> np.ndarray:
    """Foreground connected to pecan (label 3), dropping stray disconnected blobs."""
    labels = np.asarray(labels)
    fg = labels > 0
    if not np.any(fg):
        return np.zeros_like(labels, dtype=labels.dtype)

    labeled, _ = ndimage.label(fg)
    pecan_seed = labels == COMBINED_LABEL_PECAN
    if not np.any(pecan_seed):
        counts = np.bincount(labeled.ravel())
        counts[0] = 0
        largest = int(counts.argmax())
        return (labeled == largest).astype(labels.dtype)

    keep_ids = np.unique(labeled[pecan_seed])
    keep_ids = keep_ids[keep_ids != 0]
    return np.isin(labeled, keep_ids).astype(labels.dtype)


def refine_combined_frame(labels: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Refine one frame of a combined pecan/kernel/crack label image."""
    labels = np.asarray(labels)
    pecan_unchanged = labels == COMBINED_LABEL_PECAN
    pecan_clip = combined_pecan_clip_mask(labels)

    kernel = np.where(labels == COMBINED_LABEL_KERNEL, COMBINED_LABEL_KERNEL, 0)
    crack = np.where(labels == COMBINED_LABEL_CRACK, COMBINED_LABEL_CRACK, 0)

    stats = {
        "kernel_clipped_px": int(np.count_nonzero((kernel > 0) & ~(pecan_clip > 0))),
        "crack_clipped_px": int(np.count_nonzero((crack > 0) & ~(pecan_clip > 0))),
        "crack_removed_for_kernel_px": int(np.count_nonzero((crack > 0) & (kernel > 0))),
    }

    kernel, crack = refine_frame(pecan_clip, kernel, crack)

    out = np.zeros_like(labels)
    out[pecan_unchanged] = COMBINED_LABEL_PECAN
    out[kernel > 0] = COMBINED_LABEL_KERNEL
    out[crack > 0] = COMBINED_LABEL_CRACK
    return out, stats


def refine_combined_mask(path: Path, *, save: bool = True) -> dict[str, int]:
    """Load, refine, and optionally overwrite a combined multi-class mask TIFF."""
    raw = load_mask_volume(path)
    volume = as_frame_stack(raw)

    stats = {
        "frames": volume.shape[0],
        "kernel_clipped_px": 0,
        "crack_clipped_px": 0,
        "crack_removed_for_kernel_px": 0,
    }
    out_frames: list[np.ndarray] = []

    for frame in volume:
        refined, frame_stats = refine_combined_frame(frame)
        for key in ("kernel_clipped_px", "crack_clipped_px", "crack_removed_for_kernel_px"):
            stats[key] += frame_stats[key]
        out_frames.append(refined)

    if save:
        out = out_frames[0] if np.ndim(raw) == 2 else np.stack(out_frames, axis=0)
        save_mask_volume(out, path, "tiff")

    return stats


def _format_stats_line(label: str, stats: dict[str, int]) -> str:
    return (
        f"{label}: {stats['frames']} frames | "
        f"kernel clipped {stats['kernel_clipped_px']} px | "
        f"crack clipped {stats['crack_clipped_px']} px | "
        f"crack→kernel overlap removed {stats['crack_removed_for_kernel_px']} px"
    )


def _build_separate_jobs(
    file_list: Path,
) -> tuple[list[tuple[str, Path, Path | None, Path | None]], list[Path], int]:
    raw_lines = [
        line.strip()
        for line in file_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    all_paths = [Path(p) for p in raw_lines]
    groups, skipped_paths = group_mask_files(all_paths)

    jobs: list[tuple[str, Path, Path | None, Path | None]] = []
    pecan_only = 0

    for video_key, class_paths in sorted(groups.items()):
        pecan_path = class_paths.get("Pecan")
        kernel_path = class_paths.get("Kernel")
        crack_path = class_paths.get("Crack")

        if pecan_path is None:
            continue
        if kernel_path is None and crack_path is None:
            pecan_only += 1
            continue

        jobs.append((video_key, pecan_path, kernel_path, crack_path))

    return jobs, skipped_paths, pecan_only


def run_separate(file_list: Path, *, save: bool, verbose: bool) -> int:
    if not file_list.is_file():
        print(f"ERROR: file list not found: {file_list}", file=sys.stderr)
        return 1

    jobs, skipped_paths, pecan_only = _build_separate_jobs(file_list)
    print(f"File list: {file_list}")
    print(f"Videos to refine (pecan + kernel and/or crack): {len(jobs)}")
    print(f"Pecan-only videos (skipped): {pecan_only}")
    if skipped_paths:
        print(f"Unrecognized paths (skipped): {len(skipped_paths)}")
        for path in skipped_paths[:5]:
            print(f"  - {path}")
        if len(skipped_paths) > 5:
            print(f"  ... and {len(skipped_paths) - 5} more")
    if not jobs:
        print("Nothing to refine.")
        return 0

    print(f"Save in place: {save}\n")

    totals = {
        "frames": 0,
        "kernel_clipped_px": 0,
        "crack_clipped_px": 0,
        "crack_removed_for_kernel_px": 0,
    }
    errors: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for idx, (video_key, pecan_path, kernel_path, crack_path) in enumerate(jobs, start=1):
        label = Path(video_key).name
        prefix = f"[{idx}/{len(jobs)}] {label}"
        try:
            stats = refine_video_masks(
                pecan_path,
                kernel_path,
                crack_path,
                save=save,
            )
            for key in totals:
                totals[key] += stats[key]
            if verbose:
                print(_format_stats_line(prefix, stats))
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 130
        except Exception as exc:
            print(f"{prefix}  ERROR: {exc}", file=sys.stderr)
            errors.append((label, str(exc)))

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{len(jobs) - len(errors)} ok, {len(errors)} failed, {len(jobs)} total."
    )
    print(f"Total frames: {totals['frames']}")
    print(f"Kernel pixels clipped outside pecan: {totals['kernel_clipped_px']}")
    print(f"Crack pixels clipped outside pecan: {totals['crack_clipped_px']}")
    print(
        f"Crack pixels reassigned to kernel overlap: {totals['crack_removed_for_kernel_px']}"
    )
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for label, msg in errors:
            print(f"  - {label}: {msg}")
    return 1 if errors else 0


def run_combined(root: Path, *, save: bool, verbose: bool) -> int:
    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1

    jobs = discover_combined_mask_files(root)
    print(f"Scanning: {root}")
    print(f"Combined mask TIFFs found: {len(jobs)}")
    if jobs:
        print(f"Example: {jobs[0].name}")
    if not jobs:
        print("Nothing to refine.")
        return 0

    print(f"Save in place: {save}\n")

    totals = {
        "frames": 0,
        "kernel_clipped_px": 0,
        "crack_clipped_px": 0,
        "crack_removed_for_kernel_px": 0,
    }
    errors: list[tuple[str, str]] = []
    t0 = time.perf_counter()

    for idx, mask_path in enumerate(jobs, start=1):
        label = mask_path.stem
        prefix = f"[{idx}/{len(jobs)}] {label}"
        try:
            stats = refine_combined_mask(mask_path, save=save)
            for key in totals:
                totals[key] += stats[key]
            if verbose:
                print(_format_stats_line(prefix, stats))
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 130
        except Exception as exc:
            print(f"{prefix}  ERROR: {exc}", file=sys.stderr)
            errors.append((label, str(exc)))

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{len(jobs) - len(errors)} ok, {len(errors)} failed, {len(jobs)} total."
    )
    print(f"Total frames: {totals['frames']}")
    print(f"Kernel pixels clipped outside pecan: {totals['kernel_clipped_px']}")
    print(f"Crack pixels clipped outside pecan: {totals['crack_clipped_px']}")
    print(f"Crack pixels removed for kernel overlap: {totals['crack_removed_for_kernel_px']}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for label, msg in errors:
            print(f"  - {label}: {msg}")
    return 1 if errors else 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine pecan/kernel/crack training masks: clip kernel and crack to pecan, "
            "and resolve kernel/crack overlap (kernel wins)."
        )
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    separate = subparsers.add_parser(
        "separate",
        help="Refine separate Pecan / Kernel / Crack TIFFs listed in a text file",
    )
    separate.add_argument(
        "--file-list",
        "-f",
        type=Path,
        required=True,
        help="Text file with one mask path per line",
    )

    combined = subparsers.add_parser(
        "combined",
        help="Refine combined multi-class TIFFs under a directory",
    )
    combined.add_argument(
        "directory",
        type=Path,
        help=(
            "Folder to scan recursively for combined label TIFFs "
            "(e.g. '* - Pecan, Crack, Kernel.tiff')"
        ),
    )

    for sub in (separate, combined):
        sub.add_argument(
            "--dry-run",
            action="store_true",
            help="Compute stats without overwriting mask files",
        )
        sub.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            help="Print per-video stats lines",
        )

    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    save = not args.dry_run

    if args.mode == "separate":
        return run_separate(args.file_list.resolve(), save=save, verbose=args.verbose)
    if args.mode == "combined":
        return run_combined(args.directory.resolve(), save=save, verbose=args.verbose)
    print(f"ERROR: unknown mode: {args.mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
