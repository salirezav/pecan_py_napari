"""Helpers for training and running YOLO segmentation in napari."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from napari_pecan_py._reader import VIDEO_EXTENSIONS

MASK_EXTENSIONS = {".tiff", ".tif", ".npy", ".png", ".jpg", ".jpeg"}
CLASS_NAME_RE = re.compile(r"\b(\w+)$")
MASK_CLASS_BRACKET_RE = re.compile(r"\[([^\]]+)\]\s*$")
WEIGHTS_CLASSES_RE = re.compile(r"\[([^\]]+)\]")
# Combined label-map TIFFs from YOLO inference (one file, multiple classes).
MULTICLASS_LABEL_IDS: Dict[int, str] = {
    1: "Crack",
    2: "Kernel",
    3: "Pecan",
    4: "Damaged Kernel",
}
MULTICLASS_NAME_TO_ID: Dict[str, int] = {
    name: label_id for label_id, name in MULTICLASS_LABEL_IDS.items()
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class DatasetSummary:
    """Per-frame training dataset statistics."""

    total_frames: int
    frames_per_class: Dict[str, int]
    class_names: List[str]
    video_count: int
    train_frames: int = 0
    val_frames: int = 0
    split_by: str = "none"


@dataclass
class ExportSpec:
    root: Path
    train_images: Path
    train_masks: Path
    class_names: List[str]
    data_yaml: Path
    total_frames: int = 0
    frames_per_class: Dict[str, int] = field(default_factory=dict)
    val_images: Path | None = None
    val_masks: Path | None = None
    train_frames: int = 0
    val_frames: int = 0
    split_by: str = "none"


@dataclass
class ClassLabelMapping:
    """How a training class name maps onto pixel values in a mask file."""

    name: str
    source_key: str
    source_path: str
    # ``None`` means wildcard: any positive label (``*`` / ``[*]``).
    label_ids: set[int] | None = None
    enabled: bool = True

    def ids_text(self) -> str:
        return format_label_ids_text(self.label_ids)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def _glob_escape_glob_chars(text: str) -> str:
    return "".join(f"[{c}]" if c in "*?[]" else c for c in text)


def parse_label_ids_text(text: str) -> set[int] | None:
    """Parse ``\"1, 2\"``, ``\"[1]\"``, or ``\"*\"`` / ``\"[*]\"`` (wildcard)."""
    raw = str(text).strip()
    if not raw:
        raise ValueError("Label IDs cannot be empty (use * for any label).")
    lowered = raw.lower()
    if lowered in {"*", "[*]", "any", "all"}:
        return None
    if raw.startswith("[") and raw.endswith("]"):
        raw = raw[1:-1].strip()
        if raw == "*" or raw.lower() in {"any", "all"}:
            return None
    ids: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if part == "*":
            return None
        ids.add(int(part))
    if not ids:
        raise ValueError(f"No label IDs found in {text!r}")
    return ids


def format_label_ids_text(label_ids: set[int] | None) -> str:
    if label_ids is None:
        return "*"
    return ", ".join(str(i) for i in sorted(int(v) for v in label_ids))


def binary_mask_for_label_ids(
    mask_data: np.ndarray,
    label_ids: set[int] | None,
) -> np.ndarray:
    """Binary uint8 mask: ``None`` keeps any positive label, else ``isin``."""
    m = np.asarray(mask_data)
    if label_ids is None:
        return (m > 0).astype(np.uint8)
    return np.isin(m, list(label_ids)).astype(np.uint8)


def class_name_from_mask_path(
    mask_path: str | Path,
    *,
    video_stem: str | None = None,
) -> str | None:
    """Return the class name encoded in a mask filename.

    Supports ``<video> - Crack``, ``<video> - Adjusted - Crack``, and
    ``<video> - YOLO Seg [Crack]``-style names.
    """
    stem = Path(mask_path).stem
    bracket = MASK_CLASS_BRACKET_RE.search(stem)
    if bracket:
        inner = bracket.group(1).strip()
        if re.fullmatch(r"[A-Za-z]\w*", inner):
            return inner
    match = CLASS_NAME_RE.search(stem)
    if match:
        return match.group(1)
    if video_stem and stem.startswith(f"{video_stem} - "):
        tail = stem[len(video_stem) + 3 :]
        if tail and " - " not in tail and re.fullmatch(r"\w+", tail):
            return tail
    return None


def label_ids_in_mask_volume(mask_data: np.ndarray) -> set[int]:
    """Return positive integer label values present in a mask volume."""
    m = np.asarray(mask_data)
    return {int(v) for v in np.unique(m) if int(v) > 0}


def label_ids_in_mask_file(mask_path: str | Path) -> set[int]:
    return label_ids_in_mask_volume(load_mask_volume(mask_path))


def is_multiclass_label_map(mask_data: np.ndarray) -> bool:
    """True when mask pixels use the shared 1/2/3 Crack/Kernel/Pecan label map.

    Instance / watershed maps (many IDs beyond the semantic set) return False so
    they are treated as a single class with ``[*]`` rather than Crack/Kernel/Pecan.
    """
    ids = label_ids_in_mask_volume(mask_data)
    if not ids:
        return False
    known_ids = set(MULTICLASS_LABEL_IDS.keys())
    known = ids & known_ids
    if not known:
        return False
    # Extra IDs outside the semantic map ⇒ instance labels, not a class map.
    if max(ids) > max(known_ids):
        return False
    return len(known) >= 2 or max(known) >= 2


def split_label_map_to_binary_masks(label_map: np.ndarray) -> Dict[str, np.ndarray]:
    """Split a combined label map into per-class binary mask volumes."""
    m = np.asarray(label_map)
    return {
        name: (m == label_id).astype(np.uint8)
        for label_id, name in MULTICLASS_LABEL_IDS.items()
    }


def _binary_mask_volume(mask_data: np.ndarray) -> np.ndarray:
    """Convert a single-class mask volume to uint8 with foreground as 1."""
    return (np.asarray(mask_data) > 0).astype(np.uint8)


def default_label_ids_for_discovered_class(
    class_name: str,
    mask_data: np.ndarray,
) -> set[int] | None:
    """Default ID filter for a discovered class (``None`` = wildcard)."""
    if is_multiclass_label_map(mask_data):
        label_id = MULTICLASS_NAME_TO_ID.get(class_name)
        if label_id is not None:
            return {int(label_id)}
    return None


def load_masks_by_class_from_paths(
    masks_by_path: Dict[str, str | Path],
    *,
    label_ids_by_class: Dict[str, set[int] | None] | None = None,
) -> Dict[str, np.ndarray]:
    """Load per-class mask volumes.

    When ``label_ids_by_class`` is provided, each class is extracted with that
    ID filter (``None`` = any positive label). Otherwise combined semantic
    label maps are split automatically and other files use any-positive.
    """
    classes_by_path: Dict[str, set[str]] = {}
    for cls, path in masks_by_path.items():
        key = str(Path(path).resolve())
        classes_by_path.setdefault(key, set()).add(cls)

    result: Dict[str, np.ndarray] = {}
    for path_key, classes in classes_by_path.items():
        arr = load_mask_volume(path_key)
        for cls in classes:
            if label_ids_by_class is not None and cls in label_ids_by_class:
                result[cls] = binary_mask_for_label_ids(
                    arr, label_ids_by_class[cls]
                )
                continue
            if is_multiclass_label_map(arr):
                split = split_label_map_to_binary_masks(arr)
                if cls in split:
                    result[cls] = split[cls]
            else:
                result[cls] = _binary_mask_volume(arr)
    return result


def _classes_from_mask_file(candidate: Path, video_stem: str) -> Dict[str, Path]:
    """Map class names to a mask path, including combined label-map TIFFs."""
    class_name = class_name_from_mask_path(candidate, video_stem=video_stem)
    if class_name is None:
        return {}

    try:
        data = load_mask_volume(candidate)
        ids = label_ids_in_mask_volume(data)
    except Exception:
        ids = set()
        data = None

    if data is not None and is_multiclass_label_map(data):
        known = {
            MULTICLASS_LABEL_IDS[label_id]
            for label_id in ids
            if label_id in MULTICLASS_LABEL_IDS
        }
        return {cls: candidate for cls in sorted(known)}
    if class_name:
        return {class_name: candidate}
    known = {
        MULTICLASS_LABEL_IDS[label_id]
        for label_id in ids
        if label_id in MULTICLASS_LABEL_IDS
    }
    if len(known) == 1:
        return {next(iter(known)): candidate}
    return {}


def _mask_file_priority(path: Path) -> int:
    """Higher priority wins when multiple files provide the same class."""
    try:
        data = load_mask_volume(path)
        if is_multiclass_label_map(data):
            return 2
    except Exception:
        pass
    return 1


def discover_mask_files(video_path: str | Path) -> Dict[str, Path]:
    """Find mask files next to a video; keys are class names from file stems."""
    video_path = Path(video_path).resolve()
    stem = video_path.stem
    mask_dir = video_path.parent
    masks: Dict[str, Path] = {}
    for candidate in sorted(mask_dir.glob(f"{_glob_escape_glob_chars(stem)} - *")):
        if candidate.suffix.lower() not in MASK_EXTENSIONS:
            continue
        for cls, path in _classes_from_mask_file(candidate, stem).items():
            if cls not in masks or _mask_file_priority(path) >= _mask_file_priority(
                masks[cls]
            ):
                masks[cls] = path
    return masks


_VIDEO_SUFFIXES = {ext.lower() for ext in VIDEO_EXTENSIONS}


def video_path_for_saved_mask(mask_path: str | Path) -> Path | None:
    """Guess the source video path for a mask saved next to it on disk."""
    mask_path = Path(mask_path).resolve()
    if " - " not in mask_path.stem:
        return None
    video_stem = mask_path.stem.split(" - ", 1)[0]
    parent = mask_path.parent
    for candidate in parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem == video_stem
            and candidate.suffix.lower() in _VIDEO_SUFFIXES
        ):
            return candidate.resolve()
    return None


def is_video_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in _VIDEO_SUFFIXES


def discover_videos_in_directory(directory: str | Path) -> List[Path]:
    """Recursively collect video files under a directory."""
    root = Path(directory).resolve()
    if not root.is_dir():
        return []
    videos = [
        p.resolve()
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in _VIDEO_SUFFIXES
    ]
    return sorted(videos, key=lambda p: str(p).lower())


def load_mask_volume(path: str | Path) -> np.ndarray:
    """Load a mask volume from TIFF or NPY."""
    path = Path(path)
    if path.suffix.lower() in {".tiff", ".tif"}:
        import tifffile

        return np.asarray(tifffile.imread(path))
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        import cv2

        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Could not read mask image: {path}")
        return gray
    raise ValueError(f"Unsupported mask format: {path.suffix}")


def mask_volume_frame_count(mask_data: np.ndarray) -> int:
    """Return the number of frames represented by a mask volume."""
    m = np.asarray(mask_data)
    if m.ndim == 2:
        return 1
    if m.ndim == 3:
        return int(m.shape[0])
    raise ValueError(f"Expected 2D or 3D mask, got shape {m.shape}")


def count_labeled_frames(mask_data: np.ndarray) -> int:
    """Count frames in a mask volume that contain at least one labeled pixel."""
    m = np.asarray(mask_data)
    if m.ndim == 2:
        return int(np.any(m > 0))
    if m.ndim == 3:
        return int(sum(1 for t in range(m.shape[0]) if np.any(m[t] > 0)))
    raise ValueError(f"Expected 2D or 3D mask, got shape {m.shape}")


def video_frame_count(
    video_path: str | Path,
    *,
    apply_saved_range: bool = True,
) -> int:
    """Return effective frame count (optionally honors ``.pecan.json`` trim)."""
    from napari_pecan_py.video_meta import open_lazy_video

    return int(
        open_lazy_video(
            Path(video_path).resolve(),
            apply_saved_range=apply_saved_range,
        ).shape[0]
    )


def _mask_frame(mask_data: np.ndarray, frame_index: int) -> np.ndarray:
    m = np.asarray(mask_data)
    if m.ndim == 2:
        return m
    if m.ndim == 3:
        if m.shape[0] == 1:
            return m[0]
        if frame_index >= m.shape[0]:
            raise IndexError(
                f"Mask frame index {frame_index} out of range for shape {m.shape}"
            )
        return m[frame_index]
    raise ValueError(f"Expected 2D or 3D mask, got shape {m.shape}")


def _validate_mask_volumes(
    frame_count: int,
    masks_by_class: Dict[str, np.ndarray],
    video_name: str,
) -> None:
    for cls, mask_data in masks_by_class.items():
        mask_frames = mask_volume_frame_count(mask_data)
        if mask_frames != frame_count:
            raise ValueError(
                f"Frame count mismatch for '{video_name}': video has {frame_count} "
                f"frames but mask '{cls}' has {mask_frames}."
            )


def load_video_rgb_frames(
    video_path: str | Path,
    *,
    apply_saved_range: bool = True,
) -> np.ndarray:
    """Load RGB frames as (T, H, W, 3) uint8.

    When ``apply_saved_range`` is True (default), a saved ``.pecan.json``
    frame range shortens the returned stack to the trimmed length.
    """
    from napari_pecan_py.video_meta import open_lazy_video

    lazy = open_lazy_video(
        Path(video_path).resolve(),
        apply_saved_range=apply_saved_range,
    )
    frames = [np.asarray(lazy[t], dtype=np.uint8) for t in range(lazy.shape[0])]
    if not frames:
        raise ValueError(f"No frames decoded from {video_path}")
    return np.stack(frames, axis=0)


def load_image_rgb(path: str | Path) -> np.ndarray:
    """Load a single RGB image as (H, W, 3) uint8."""
    from PIL import Image as PILImage

    img = np.asarray(PILImage.open(path).convert("RGB"), dtype=np.uint8)
    return img


def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255)
        if float(np.max(img)) <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.shape[-1] == 4:
        img = img[..., :3]
    return img


def _write_yolo_label_lines(
    masks_by_class: Dict[str, np.ndarray],
    class_names: List[str],
    frame_index: int,
    height: int,
    width: int,
) -> List[str]:
    """Write YOLO polygon lines for one frame.

    Classes with no visible pixels on this frame are omitted. That is the
    correct supervision signal when a class (e.g. Crack) is only visible on
    some frames while others (e.g. Pecan) appear on every frame.
    """
    import cv2

    frame_lines: List[str] = []
    for cls_idx, cls in enumerate(class_names):
        if cls not in masks_by_class:
            continue
        m2d = _mask_frame(masks_by_class[cls], frame_index)
        bin_mask = (m2d > 0).astype(np.uint8)
        if bin_mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        for cnt in contours:
            area = float(cv2.contourArea(cnt))
            if area < 10.0:
                continue
            peri = float(cv2.arcLength(cnt, True))
            if peri > 0:
                cnt = cv2.approxPolyDP(cnt, 0.002 * peri, True)
            pts = cnt.reshape(-1, 2)
            if pts.shape[0] < 3:
                continue
            xs = pts[:, 0] / float(width)
            ys = pts[:, 1] / float(height)
            coord_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in zip(xs.tolist(), ys.tolist()))
            frame_lines.append(f"{cls_idx} {coord_str}")
    return frame_lines


def _entry_frame_count(
    video_path: Path,
    *,
    apply_saved_range: bool = True,
) -> int:
    suffix = video_path.suffix.lower()
    if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
        return video_frame_count(video_path, apply_saved_range=apply_saved_range)
    if suffix in IMAGE_EXTENSIONS:
        return 1
    raise ValueError(f"Unsupported training input: {video_path}")


def plan_train_val_split(
    video_paths: Sequence[str | Path],
    frame_counts: Sequence[int],
    val_fraction: float,
    split_by: str = "video",
) -> tuple[str, set[int], set[tuple[int, int]]]:
    """Plan which videos or frames go to validation.

    Returns ``(effective_split_by, val_video_indices, val_frame_keys)``.
    Falls back to frame split when only one video is available.
    """
    import random

    n_videos = len(frame_counts)
    total_frames = int(sum(frame_counts))
    if val_fraction <= 0 or total_frames <= 1 or n_videos == 0:
        return "none", set(), set()

    if split_by == "video" and n_videos > 1:
        n_val = min(n_videos - 1, max(1, round(n_videos * val_fraction)))
        sorted_idx = sorted(range(n_videos), key=lambda i: str(video_paths[i]))
        val_video_indices = set(sorted_idx[-n_val:])
        return "video", val_video_indices, set()

    all_keys = [
        (video_idx, frame_idx)
        for video_idx, count in enumerate(frame_counts)
        for frame_idx in range(count)
    ]
    n_val = min(len(all_keys) - 1, max(1, round(len(all_keys) * val_fraction)))
    val_frame_keys = set(random.Random(42).sample(all_keys, n_val))
    return "frame", set(), val_frame_keys


def _split_name_for_frame(
    video_idx: int,
    frame_idx: int,
    effective_split_by: str,
    val_video_indices: set[int],
    val_frame_keys: set[tuple[int, int]],
) -> str:
    if effective_split_by == "video":
        return "val" if video_idx in val_video_indices else "train"
    if effective_split_by == "frame":
        return "val" if (video_idx, frame_idx) in val_frame_keys else "train"
    return "train"


def count_split_frames(
    frame_counts: Sequence[int],
    effective_split_by: str,
    val_video_indices: set[int],
    val_frame_keys: set[tuple[int, int]],
) -> tuple[int, int]:
    train_frames = 0
    val_frames = 0
    for video_idx, count in enumerate(frame_counts):
        for frame_idx in range(count):
            if (
                _split_name_for_frame(
                    video_idx,
                    frame_idx,
                    effective_split_by,
                    val_video_indices,
                    val_frame_keys,
                )
                == "val"
            ):
                val_frames += 1
            else:
                train_frames += 1
    return train_frames, val_frames


def summarize_training_dataset(
    video_entries: Sequence[Tuple[str | Path, Dict[str, str | Path]]],
    *,
    val_fraction: float = 0.2,
    split_by: str = "video",
    apply_saved_range: bool = True,
    label_ids_by_class: Dict[str, set[int] | None] | None = None,
) -> DatasetSummary:
    """Summarize how many frames each class is labeled on across all videos."""
    class_names_set: set[str] = set()
    frames_per_class: Dict[str, int] = {}
    total_frames = 0
    video_paths: List[str] = []
    frame_counts: List[int] = []

    for video_path, masks_by_path in video_entries:
        video_path = Path(video_path).resolve()
        t_count = _entry_frame_count(
            video_path, apply_saved_range=apply_saved_range
        )
        total_frames += t_count
        video_paths.append(str(video_path))
        frame_counts.append(t_count)
        class_names_set.update(masks_by_path.keys())

        loaded_masks = load_masks_by_class_from_paths(
            masks_by_path, label_ids_by_class=label_ids_by_class
        )
        _validate_mask_volumes(t_count, loaded_masks, video_path.name)

        for cls, mask_data in loaded_masks.items():
            frames_per_class[cls] = frames_per_class.get(cls, 0) + count_labeled_frames(
                mask_data
            )

    effective_split_by, val_video_indices, val_frame_keys = plan_train_val_split(
        video_paths, frame_counts, val_fraction, split_by
    )
    train_frames, val_frames = count_split_frames(
        frame_counts, effective_split_by, val_video_indices, val_frame_keys
    )

    return DatasetSummary(
        total_frames=total_frames,
        frames_per_class=frames_per_class,
        class_names=sorted(class_names_set),
        video_count=len(video_entries),
        train_frames=train_frames,
        val_frames=val_frames,
        split_by=effective_split_by,
    )


def format_dataset_summary(summary: DatasetSummary) -> str:
    if summary.total_frames == 0:
        return "No training frames."
    parts = []
    for cls in summary.class_names:
        labeled = summary.frames_per_class.get(cls, 0)
        parts.append(f"{cls}: {labeled}/{summary.total_frames} frames")
    split_text = ""
    if summary.val_frames > 0:
        split_text = (
            f" Split: {summary.train_frames} train / {summary.val_frames} val "
            f"({summary.split_by})."
        )
    return (
        f"{summary.total_frames} frame(s) from {summary.video_count} video(s)."
        + split_text
        + " "
        + "; ".join(parts)
    )


def export_videos_seg_dataset(
    video_entries: Sequence[Tuple[str | Path, Dict[str, str | Path]]],
    out_root: str | Path,
    *,
    val_fraction: float = 0.2,
    split_by: str = "video",
    apply_saved_range: bool = True,
    label_ids_by_class: Dict[str, set[int] | None] | None = None,
) -> ExportSpec:
    """Export multiple on-disk videos and their mask files to a YOLO-seg dataset."""
    from PIL import Image as PILImage
    import yaml

    root = Path(out_root).resolve()
    train_images_dir = _ensure_dir(root / "images" / "train")
    train_labels_dir = _ensure_dir(root / "labels" / "train")
    val_images_dir = _ensure_dir(root / "images" / "val")
    val_labels_dir = _ensure_dir(root / "labels" / "val")

    class_names_set: set[str] = set()
    for _, masks in video_entries:
        class_names_set.update(masks.keys())
    class_names = sorted(class_names_set)
    total_exported_frames = 0
    train_exported_frames = 0
    val_exported_frames = 0
    exported_frames_per_class: Dict[str, int] = {cls: 0 for cls in class_names}

    video_paths = [str(Path(v).resolve()) for v, _ in video_entries]
    frame_counts: List[int] = []
    for video_path, _ in video_entries:
        frame_counts.append(
            _entry_frame_count(
                Path(video_path), apply_saved_range=apply_saved_range
            )
        )

    effective_split_by, val_video_indices, val_frame_keys = plan_train_val_split(
        video_paths, frame_counts, val_fraction, split_by
    )

    if not class_names:
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/train",
                    "names": {},
                    "task": "segment",
                }
            )
        )
        return ExportSpec(root, train_images_dir, train_labels_dir, [], data_yaml)

    for video_idx, (video_path, masks_by_path) in enumerate(video_entries):
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            frames = load_video_rgb_frames(
                video_path, apply_saved_range=apply_saved_range
            )
        elif suffix in IMAGE_EXTENSIONS:
            frames = load_image_rgb(video_path)[None, ...]
        else:
            raise ValueError(f"Unsupported training input: {video_path}")

        masks_by_class = load_masks_by_class_from_paths(
            masks_by_path, label_ids_by_class=label_ids_by_class
        )
        stem = video_path.stem
        t_count = frames.shape[0]
        _validate_mask_volumes(t_count, masks_by_class, video_path.name)

        for t in range(t_count):
            split = _split_name_for_frame(
                video_idx, t, effective_split_by, val_video_indices, val_frame_keys
            )
            images_dir = val_images_dir if split == "val" else train_images_dir
            labels_dir = val_labels_dir if split == "val" else train_labels_dir

            img = _to_uint8_rgb(frames[t])
            img_name = f"{stem}_frame_{t:04d}.png"
            PILImage.fromarray(img).save(images_dir / img_name)

            lines = _write_yolo_label_lines(
                masks_by_class, class_names, t, img.shape[0], img.shape[1]
            )
            (labels_dir / f"{stem}_frame_{t:04d}.txt").write_text("\n".join(lines))
            total_exported_frames += 1
            if split == "val":
                val_exported_frames += 1
            else:
                train_exported_frames += 1
            for cls in class_names:
                if cls not in masks_by_class:
                    continue
                m2d = _mask_frame(masks_by_class[cls], t)
                if np.any(m2d > 0):
                    exported_frames_per_class[cls] += 1

    val_yaml_path = "images/val" if val_exported_frames > 0 else "images/train"
    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": val_yaml_path,
                "names": {i: n for i, n in enumerate(class_names)},
                "task": "segment",
            }
        )
    )
    return ExportSpec(
        root,
        train_images_dir,
        train_labels_dir,
        class_names,
        data_yaml,
        total_frames=total_exported_frames,
        frames_per_class=exported_frames_per_class,
        val_images=val_images_dir,
        val_masks=val_labels_dir,
        train_frames=train_exported_frames,
        val_frames=val_exported_frames,
        split_by=effective_split_by,
    )


def export_napari_seg_dataset(
    image_data: np.ndarray,
    masks_by_class: Dict[str, np.ndarray],
    out_root: str | Path,
) -> ExportSpec:
    """Export in-memory frames + Napari Labels masks to YOLO-seg polygon annotations."""
    from PIL import Image as PILImage
    import yaml

    root = Path(out_root).resolve()
    images_dir = _ensure_dir(root / "images" / "train")
    labels_dir = _ensure_dir(root / "labels" / "train")

    if image_data.ndim == 3:
        image_data = image_data[None, ...]
    t_count, height, width, _ = image_data.shape

    class_names = sorted(masks_by_class.keys())
    total_exported_frames = 0
    exported_frames_per_class: Dict[str, int] = {cls: 0 for cls in class_names}

    if not class_names:
        data_yaml = root / "data.yaml"
        data_yaml.write_text(
            yaml.safe_dump(
                {
                    "path": str(root),
                    "train": "images/train",
                    "val": "images/train",
                    "names": {},
                    "task": "segment",
                }
            )
        )
        return ExportSpec(root, images_dir, labels_dir, [], data_yaml)

    for t in range(t_count):
        img = _to_uint8_rgb(image_data[t])
        img_name = f"frame_{t:04d}.png"
        PILImage.fromarray(img).save(images_dir / img_name)
        lines = _write_yolo_label_lines(
            masks_by_class, class_names, t, height, width
        )
        (labels_dir / f"frame_{t:04d}.txt").write_text("\n".join(lines))
        total_exported_frames += 1
        for cls in class_names:
            m2d = _mask_frame(masks_by_class[cls], t)
            if np.any(m2d > 0):
                exported_frames_per_class[cls] += 1

    data_yaml = root / "data.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(root),
                "train": "images/train",
                "val": "images/train",
                "names": {i: n for i, n in enumerate(class_names)},
                "task": "segment",
            }
        )
    )
    return ExportSpec(
        root,
        images_dir,
        labels_dir,
        class_names,
        data_yaml,
        total_frames=total_exported_frames,
        frames_per_class=exported_frames_per_class,
    )


def to_yolo_predict_source(frame: np.ndarray):
    """Wrap an RGB frame for ``model.predict`` (PIL = RGB, unlike numpy = BGR)."""
    from PIL import Image as PILImage

    return PILImage.fromarray(_to_uint8_rgb(frame))


def inference_imgsz(height: int, width: int, model=None) -> int:
    """Pick an inference size aligned with YOLO stride (multiple of 32)."""
    if model is not None:
        for attr in ("overrides", "args"):
            args = getattr(model, attr, None)
            if isinstance(args, dict) and args.get("imgsz"):
                try:
                    saved = int(args["imgsz"])
                    if saved > 0:
                        return saved
                except (TypeError, ValueError):
                    pass
    size = max(int(height), int(width))
    return max(32, int(((size + 31) // 32) * 32))


def _polygon_xy_components(poly: np.ndarray, *, gap_px: float = 15.0) -> List[np.ndarray]:
    """Split ``masks.xy`` vertices into separate contours when points jump far apart.

    A single YOLO polygon can list disconnected islands in one array; filling it
    as one polygon draws connector edges between islands.
    """
    if poly is None or len(poly) < 3:
        return []
    poly = np.asarray(poly, dtype=np.float64)
    if len(poly) < 4:
        return [np.round(poly).astype(np.int32)]
    jumps = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    breaks = np.where(jumps > gap_px)[0] + 1
    if len(breaks) == 0:
        return [np.round(poly).astype(np.int32)]
    split_at = np.concatenate(([0], breaks, [len(poly)]))
    return [
        np.round(poly[a:b]).astype(np.int32)
        for a, b in zip(split_at[:-1], split_at[1:])
        if b - a >= 3
    ]


def yolo_result_to_label_map(result) -> np.ndarray | None:
    """Convert one ultralytics result to a 2D uint8 label map at native resolution."""
    import cv2

    if result is None or result.masks is None:
        return None

    height, width = int(result.orig_shape[0]), int(result.orig_shape[1])
    label_map = np.zeros((height, width), dtype=np.uint8)

    classes = None
    if result.boxes is not None and result.boxes.cls is not None:
        classes = result.boxes.cls.cpu().numpy().astype(int)

    mask_data = result.masks.data
    if mask_data is not None and mask_data.numel() > 0:
        masks_np = mask_data.cpu().numpy()
        if masks_np.ndim == 2:
            masks_np = masks_np[None]

        for i in range(masks_np.shape[0]):
            m = (masks_np[i] > 0.5).astype(np.uint8)
            if m.shape != (height, width):
                m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
                m = (m > 0.5).astype(np.uint8)
            label_val = (
                int(classes[i]) + 1
                if classes is not None and i < len(classes)
                else i + 1
            )
            label_map[m > 0] = label_val

        if np.any(label_map):
            return label_map

    segments = getattr(result.masks, "xy", None)
    if segments is not None and len(segments) > 0:
        for i, poly in enumerate(segments):
            label_val = (
                int(classes[i]) + 1
                if classes is not None and i < len(classes)
                else i + 1
            )
            for pts in _polygon_xy_components(poly):
                cv2.fillPoly(label_map, [pts], label_val)
        if np.any(label_map):
            return label_map

    return label_map if np.any(label_map) else None


def image_volume_to_rgb_frames(arr: np.ndarray) -> np.ndarray:
    """Normalize an image layer array to (T, H, W, 3) RGB."""
    if arr.ndim == 4:
        frames = np.stack([np.asarray(arr[t]) for t in range(arr.shape[0])], axis=0)
    else:
        data = np.asarray(arr)
        if data.ndim == 3 and data.shape[-1] in (3, 4):
            frames = data[None, ...]
        elif data.ndim == 4:
            frames = data
        else:
            raise ValueError(f"Unsupported image shape {data.shape}")
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    return frames


def run_yolo_seg_inference_on_frames(
    weights_path: str | Path,
    frames: np.ndarray,
    device: str,
    *,
    progress_callback=None,
    cancel_callback=None,
) -> np.ndarray:
    """Run YOLO segmentation on an RGB volume; returns a label volume."""
    from ultralytics import YOLO

    rgb = image_volume_to_rgb_frames(frames)
    model = YOLO(str(weights_path))
    label_stack: list[np.ndarray] = []
    total_frames = int(rgb.shape[0])

    for t in range(total_frames):
        if cancel_callback is not None:
            try:
                if bool(cancel_callback()):
                    break
            except Exception:
                pass
        frame = _to_uint8_rgb(rgb[t])
        h, w = frame.shape[:2]
        frame_imgsz = inference_imgsz(h, w, model)
        res = model.predict(
            to_yolo_predict_source(frame),
            imgsz=frame_imgsz,
            conf=0.1,
            device=resolve_yolo_device(device),
            retina_masks=True,
            verbose=False,
        )
        label_map = yolo_result_to_label_map(res[0] if res else None)
        if label_map is None:
            label_map = np.zeros((h, w), dtype=np.uint8)
        label_stack.append(label_map)
        if progress_callback is not None:
            try:
                progress_callback(t + 1, total_frames)
            except Exception:
                pass

    if not label_stack:
        raise ValueError("YOLO inference produced no frames.")

    if len(label_stack) == 1:
        return label_stack[0]
    return np.stack(label_stack, axis=0).astype(np.uint8)


def infer_mask_output_path(
    source_path: str | Path,
    name_suffix: str,
    fmt: str,
) -> Path:
    """Build an on-disk mask path next to the source media file."""
    source_path = Path(source_path).resolve()
    ext = {"tiff": ".tiff", "png": ".png", "npy": ".npy"}[str(fmt).lower()]
    return source_path.parent / f"{source_path.stem}{name_suffix}{ext}"


def save_mask_volume(data: np.ndarray, path: str | Path, fmt: str) -> None:
    """Save a label volume as TIFF, PNG, or NPY."""
    from PIL import Image as PILImage

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.asarray(data).astype(np.uint8)
    fmt = str(fmt).lower()

    if fmt == "tiff":
        import tifffile

        tifffile.imwrite(path, arr)
        return
    if fmt == "npy":
        np.save(path, arr)
        return
    if fmt == "png":
        if arr.ndim == 2:
            PILImage.fromarray(arr).save(path)
            return
        if arr.ndim == 3:
            frames = [PILImage.fromarray(arr[t]) for t in range(arr.shape[0])]
            frames[0].save(
                path,
                save_all=True,
                append_images=frames[1:],
                duration=0,
                loop=0,
            )
            return
    raise ValueError(f"Unsupported mask save format: {fmt}")


def default_auto_device() -> str:
    """Return the device used when ``auto`` is selected (first CUDA GPU, else CPU)."""
    try:
        import torch

        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            return "0"
    except Exception:
        pass
    return "cpu"


def auto_device_display_name() -> str:
    """Human-readable label for the device ``auto`` resolves to."""
    resolved = default_auto_device()
    if resolved == "cpu":
        return "CPU"
    if resolved.isdigit():
        return f"CUDA:{resolved}"
    if str(resolved).lower().startswith("cuda"):
        return str(resolved).upper()
    return str(resolved).upper()


def guess_save_suffix_from_weights(
    weights_path: str | Path,
    *,
    fallback_index: int = 1,
) -> tuple[str, bool]:
    """Guess save/layer suffix from weights filename ``… - [Class].pt`` convention.

    Returns ``(suffix, used_bracket_convention)``.
    """
    stem = Path(weights_path).stem
    match = WEIGHTS_CLASSES_RE.search(stem)
    if match:
        classes = match.group(1).strip()
        if classes:
            return f" - {classes}", True
    return f" - YOLO Seg [{fallback_index}]", False


def infer_labels_layer_name(source_name: str, suffix: str) -> str:
    """Build a napari Labels layer name from a source name and suffix."""
    suffix = suffix if suffix.startswith(" ") else f" {suffix}" if suffix else " - YOLO seg"
    return f"{source_name}{suffix}"


def resolve_yolo_device(device: str) -> str:
    """Resolve a widget/device-string for ultralytics train/predict."""
    if str(device).strip().lower() == "auto":
        return default_auto_device()
    return str(device)


@dataclass
class YoloTrainAugmentConfig:
    """Training-time augmentation settings passed to ``ultralytics`` ``model.train``."""

    enabled: bool = True
    degrees: float = 45.0
    scale: float = 0.5
    hsv_h: float = 0.02
    hsv_s: float = 0.7
    hsv_v: float = 0.5
    fliplr: float = 0.5
    translate: float = 0.1
    mosaic: float = 1.0
    randaugment: bool = True

    def to_train_kwargs(self) -> Dict[str, float | str | None]:
        """Build keyword arguments for ``YOLO.train`` augmentation hyperparameters."""
        if not self.enabled:
            return {
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "scale": 0.0,
                "translate": 0.0,
                "fliplr": 0.0,
                "flipud": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "auto_augment": None,
                "erasing": 0.0,
            }
        return {
            "hsv_h": float(self.hsv_h),
            "hsv_s": float(self.hsv_s),
            "hsv_v": float(self.hsv_v),
            "degrees": float(self.degrees),
            "scale": float(self.scale),
            "translate": float(self.translate),
            "fliplr": float(self.fliplr),
            "flipud": 0.0,
            "mosaic": float(self.mosaic),
            "mixup": 0.0,
            "copy_paste": 0.0,
            "auto_augment": "randaugment" if self.randaugment else None,
            "erasing": 0.0,
        }


def format_augment_summary(config: YoloTrainAugmentConfig) -> str:
    if not config.enabled:
        return "Augmentations: off"
    parts = [
        f"rotate ±{config.degrees:.0f}°",
        f"scale ±{config.scale:.2f}",
        f"sat ±{config.hsv_s:.2f}",
        f"bright ±{config.hsv_v:.2f}",
    ]
    if config.randaugment:
        parts.append("RandAugment")
    return "Augmentations: " + ", ".join(parts)


def train_yolo_seg(
    data_yaml: str | Path,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 50,
    batch: int = 4,
    lr: float = 1e-3,
    device: str = "auto",
    project: str | Path | None = None,
    name: str = "pecan-yolo-seg",
    augment: YoloTrainAugmentConfig | None = None,
) -> str:
    """Train a YOLO segmentation model using `ultralytics`."""
    from ultralytics import YOLO

    aug = augment or YoloTrainAugmentConfig()
    yolo = YOLO(model_name)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        lr0=lr,
        device=resolve_yolo_device(device),
        project=str(project) if project is not None else None,
        name=name,
        **aug.to_train_kwargs(),
    )
    return str(results.best)
