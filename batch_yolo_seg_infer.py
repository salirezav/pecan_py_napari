#!/usr/bin/env python3
"""Batch YOLO segmentation inference on videos (no napari UI).

Uses the same inference path as the YOLO Segmentation widget: load each video,
run YOLO seg frame-by-frame, save a mask volume next to the source file.

Example:
    python batch_yolo_seg_infer.py "D:\\VIDS & MASKS" -w "yolo_seg_runs\\Crack Segmentor - [Crack].pt"

    python batch_yolo_seg_infer.py ./videos -w best.pt --device cuda:0 --suffix " - Crack" -f tiff
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_SRC = Path(__file__).resolve().parent / "src"
# Always prefer this repo's src tree (mask rasterization fixes live in model.py).
_repo_src = str(_REPO_SRC)
while _repo_src in sys.path:
    sys.path.remove(_repo_src)
sys.path.insert(0, _repo_src)

from napari_pecan_py.widgets.yolo_seg import model as _yolo_model  # noqa: E402
from napari_pecan_py.widgets.yolo_seg.model import (  # noqa: E402
    discover_videos_in_directory,
    guess_save_suffix_from_weights,
    infer_mask_output_path,
    load_video_rgb_frames,
    resolve_yolo_device,
    run_yolo_seg_inference_on_frames,
    save_mask_volume,
)


def _verify_mask_rasterization_fix() -> str:
    """Fail fast if an older installed package lacks the polygon-island fix."""
    import inspect

    if not hasattr(_yolo_model, "_polygon_xy_components"):
        raise RuntimeError(
            "The loaded napari_pecan_py is missing the YOLO mask rasterization fix "
            f"({_yolo_model.__file__}). Run this script from the repo or reinstall editable."
        )
    src = inspect.getsource(_yolo_model.yolo_result_to_label_map)
    if src.find("mask_data") > src.find("segments"):
        raise RuntimeError(
            "The loaded napari_pecan_py still rasterizes YOLO polygons before retina "
            f"bitmaps ({_yolo_model.__file__}). Reinstall from this repo."
        )
    return str(_yolo_model.__file__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO segmentation on every video under a directory and save "
            "mask files next to each video (same behavior as the napari widget)."
        )
    )
    parser.add_argument(
        "directory",
        type=Path,
        help="Folder to scan recursively for videos (.mp4, .avi, .mov, .mkv)",
    )
    parser.add_argument(
        "--weights",
        "-w",
        type=Path,
        required=True,
        help="Trained YOLO .pt weights file",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, 0, cuda:0, … (default: auto)",
    )
    parser.add_argument(
        "--suffix",
        default=None,
        help=(
            "Suffix appended to each video stem for the mask filename "
            "(default: parsed from weights, e.g. ' - Crack' from 'model - [Crack].pt')"
        ),
    )
    parser.add_argument(
        "--format",
        "-f",
        choices=("tiff", "png", "npy"),
        default="tiff",
        help="Mask file format (default: tiff)",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a video when its output mask file already exists",
    )
    parser.add_argument(
        "--instance-ids",
        action="store_true",
        help=(
            "Write a unique label ID per detection (1, 2, 3…) instead of "
            "class_index + 1. Use for instance-segmentation models."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.directory.resolve()
    weights = args.weights.resolve()

    if not root.is_dir():
        print(f"ERROR: not a directory: {root}", file=sys.stderr)
        return 1
    if not weights.is_file():
        print(f"ERROR: weights not found: {weights}", file=sys.stderr)
        return 1

    suffix = args.suffix
    if suffix is None:
        suffix, from_brackets = guess_save_suffix_from_weights(weights)
        if from_brackets:
            print(f"Save suffix from weights: {suffix!r}")
        else:
            print(f"Save suffix (default): {suffix!r}")

    device = resolve_yolo_device(args.device)
    model_path = _verify_mask_rasterization_fix()
    print(f"Device: {device}")
    print(f"Inference helpers: {model_path}")
    print(f"Scanning: {root}")

    videos = discover_videos_in_directory(root)
    if not videos:
        print("No videos found.")
        return 0

    print(f"Found {len(videos)} video(s).\n")

    ok = 0
    skipped = 0
    failed = 0
    t0 = time.perf_counter()

    for idx, video_path in enumerate(videos, start=1):
        out_path = infer_mask_output_path(video_path, suffix, args.format)
        prefix = f"[{idx}/{len(videos)}] {video_path.name}"

        if args.skip_existing and out_path.is_file():
            print(f"{prefix}  skip (exists: {out_path.name})")
            skipped += 1
            continue

        try:
            print(f"{prefix}  loading frames…", flush=True)
            frames = load_video_rgb_frames(video_path)
            frame_count = int(frames.shape[0])

            def _progress(cur: int, total: int, _pfx=prefix) -> None:
                if cur == 1 or cur == total or cur % max(1, total // 10) == 0:
                    print(f"  {_pfx}  infer {cur}/{total}", flush=True)

            print(f"{prefix}  inferring {frame_count} frame(s)…", flush=True)
            labels = run_yolo_seg_inference_on_frames(
                weights,
                frames,
                device,
                progress_callback=_progress,
                instance_labels=bool(args.instance_ids),
            )
            del frames

            if labels.ndim == 3:
                pred_frames = sum(int(labels[t].any()) for t in range(labels.shape[0]))
            else:
                pred_frames = int(labels.any())

            save_mask_volume(labels, out_path, args.format)
            del labels

            print(
                f"{prefix}  saved -> {out_path.name}  "
                f"({pred_frames}/{frame_count} frame(s) with predictions)"
            )
            ok += 1
        except KeyboardInterrupt:
            print("\nStopped by user.")
            return 130
        except Exception as exc:
            print(f"{prefix}  ERROR: {exc}", file=sys.stderr)
            failed += 1

    elapsed = time.perf_counter() - t0
    print(
        f"\nDone in {elapsed:.1f}s — "
        f"{ok} saved, {skipped} skipped, {failed} failed, {len(videos)} total."
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
