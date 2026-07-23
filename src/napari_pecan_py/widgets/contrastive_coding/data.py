"""Dataset discovery, multi-label mask helpers, and checkpoint I/O."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    TRAINING_MODE_HIERARCHICAL,
    TRAINING_MODE_SIMPLE,
    format_hierarchy_chain,
)

from napari_pecan_py.widgets.yolo_seg.model import (
    _glob_escape_glob_chars,
    discover_videos_in_directory,
    load_mask_volume,
    mask_volume_frame_count,
    video_frame_count,
)

MASK_EXTENSIONS = {".tiff", ".tif", ".npy"}

# Combined multi-class label TIFF convention (see refine_training_masks.ipynb).
COMBINED_MASK_STEM_SUFFIX = " - Pecan,Kernel,Crack"
KNOWN_LABEL_ID_TO_NAME: Dict[int, str] = {
    1: "Crack",
    2: "Kernel",
    3: "Pecan",
}


@dataclass
class ContrastiveDatasetSummary:
    video_count: int
    total_frames: int
    class_names: List[str]
    frames_per_class: Dict[str, int]


@dataclass
class ContrastiveCheckpointMetadata:
    class_names: List[str]
    in_channels: int
    patch_size: int
    embed_dim: int = 64
    temperature: float = 0.1
    training_mode: str = TRAINING_MODE_SIMPLE
    hierarchy_chain: List[str] | None = None
    soft_positive_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.hierarchy_chain is None:
            self.hierarchy_chain = list(DEFAULT_HIERARCHY_CHAIN)

    def to_dict(self) -> dict:
        return {
            "class_names": list(self.class_names),
            "in_channels": int(self.in_channels),
            "patch_size": int(self.patch_size),
            "embed_dim": int(self.embed_dim),
            "temperature": float(self.temperature),
            "training_mode": str(self.training_mode),
            "hierarchy_chain": list(self.hierarchy_chain or DEFAULT_HIERARCHY_CHAIN),
            "soft_positive_weight": float(self.soft_positive_weight),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContrastiveCheckpointMetadata":
        mode = str(data.get("training_mode", TRAINING_MODE_SIMPLE))
        chain = data.get("hierarchy_chain")
        if chain is None:
            chain = list(DEFAULT_HIERARCHY_CHAIN)
        else:
            chain = [str(c) for c in chain]
        return cls(
            class_names=[str(n) for n in data.get("class_names", [])],
            in_channels=int(data.get("in_channels", 3)),
            patch_size=int(data.get("patch_size", 8)),
            embed_dim=int(data.get("embed_dim", 64)),
            temperature=float(data.get("temperature", 0.1)),
            training_mode=mode,
            hierarchy_chain=chain,
            soft_positive_weight=float(data.get("soft_positive_weight", 0.5)),
        )

    def is_hierarchical(self) -> bool:
        return self.training_mode == TRAINING_MODE_HIERARCHICAL


def label_id_to_name(label_id: int) -> str:
    return KNOWN_LABEL_ID_TO_NAME.get(int(label_id), f"Label_{int(label_id)}")


def contrastive_checkpoint_filename(
    class_names: Sequence[str],
    *,
    training_mode: str = TRAINING_MODE_SIMPLE,
) -> str:
    prefix = (
        "contrastive-hier"
        if training_mode == TRAINING_MODE_HIERARCHICAL
        else "contrastive"
    )
    if not class_names:
        return f"{prefix}.pt"
    inner = ", ".join(sorted(class_names))
    return f"{prefix} - [{inner}].pt"


def save_contrastive_checkpoint(
    path: str | Path,
    state_dict: dict,
    metadata: ContrastiveCheckpointMetadata,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state_dict,
            "metadata": metadata.to_dict(),
        },
        path,
    )


def load_contrastive_checkpoint(
    path: str | Path,
) -> Tuple[dict, ContrastiveCheckpointMetadata | None]:
    import torch

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict) and "state_dict" in raw:
        meta = None
        if "metadata" in raw and isinstance(raw["metadata"], dict):
            meta = ContrastiveCheckpointMetadata.from_dict(raw["metadata"])
        return raw["state_dict"], meta
    return raw, None


def _mask_has_multiple_classes(mask_data: np.ndarray) -> bool:
    m = np.asarray(mask_data)
    if m.ndim == 2:
        sample = m
    else:
        sample = m[min(m.shape[0] // 2, m.shape[0] - 1)]
    values = {int(v) for v in np.unique(sample) if int(v) > 0}
    return len(values) >= 1


def discover_multilabel_mask(video_path: str | Path) -> Path | None:
    """Find the multi-class label TIFF next to a training video."""
    video_path = Path(video_path).resolve()
    stem = video_path.stem
    mask_dir = video_path.parent

    for ext in (".tiff", ".tif"):
        primary = mask_dir / f"{stem}{COMBINED_MASK_STEM_SUFFIX}{ext}"
        if primary.is_file():
            return primary

    for candidate in sorted(mask_dir.glob(f"{_glob_escape_glob_chars(stem)} - *")):
        if candidate.suffix.lower() not in MASK_EXTENSIONS:
            continue
        try:
            data = load_mask_volume(candidate)
        except Exception:
            continue
        if _mask_has_multiple_classes(data):
            return candidate.resolve()
    return None


def discover_label_values_in_mask(
    mask_data: np.ndarray,
    *,
    max_frames_sample: int = 24,
) -> List[int]:
    m = np.asarray(mask_data)
    if m.ndim == 2:
        frames = [m]
    else:
        n = int(m.shape[0])
        if n <= max_frames_sample:
            indices = range(n)
        else:
            step = max(1, n // max_frames_sample)
            indices = range(0, n, step)
        frames = [m[i] for i in indices]

    values: set[int] = set()
    for frame in frames:
        for v in np.unique(frame):
            iv = int(v)
            if iv > 0:
                values.add(iv)
    return sorted(values)


def label_values_to_names(label_values: Sequence[int]) -> Dict[int, str]:
    return {int(v): label_id_to_name(int(v)) for v in label_values}


def multilabel_frame_to_class_masks(
    labels: np.ndarray,
    class_value_map: Dict[str, int],
) -> Dict[str, np.ndarray]:
    labels = np.asarray(labels)
    return {name: labels == value for name, value in class_value_map.items()}


def count_labeled_frames_for_classes(
    mask_data: np.ndarray,
    class_value_map: Dict[str, int],
) -> Dict[str, int]:
    m = np.asarray(mask_data)
    counts = {name: 0 for name in class_value_map}
    if m.ndim == 2:
        frames = [m]
    else:
        frames = [m[t] for t in range(m.shape[0])]
    for frame in frames:
        for name, value in class_value_map.items():
            if np.any(frame == value):
                counts[name] += 1
    return counts


def load_video_frame_rgb(video_path: str | Path, frame_index: int) -> np.ndarray:
    from napari_pecan_py.video_meta import open_lazy_video

    lazy = open_lazy_video(Path(video_path).resolve())
    frame = np.asarray(lazy[int(frame_index)])
    if frame.ndim == 2:
        frame = frame[..., np.newaxis]
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def validate_training_pair(video_path: str | Path, mask_path: str | Path) -> None:
    """Require mask length to match the *effective* (possibly trimmed) video length."""
    video_frames = video_frame_count(video_path)
    mask_frames = mask_volume_frame_count(load_mask_volume(mask_path))
    if mask_frames != video_frames:
        from napari_pecan_py.video_meta import (
            get_saved_frame_range,
            get_saved_frame_sample,
            pecan_meta_path,
        )

        fr = get_saved_frame_range(video_path)
        sample = get_saved_frame_sample(video_path)
        trim_note = ""
        if sample is not None:
            trim_note = (
                f" Video uses saved sample start={sample[0]}, step={sample[1]}, "
                f"count≤{sample[2]} from {pecan_meta_path(video_path).name}; "
                f"the TIFF should have {video_frames} frames."
            )
        elif fr is not None:
            trim_note = (
                f" Video uses saved trim [{fr[0]}:{fr[1]}] from "
                f"{pecan_meta_path(video_path).name}; the TIFF should have "
                f"{video_frames} frames (same as the trimmed range)."
            )
        raise ValueError(
            f"Frame count mismatch for '{Path(video_path).name}': "
            f"video has {video_frames} frames but mask has {mask_frames}.{trim_note}"
        )


def summarize_contrastive_dataset(
    entries: Sequence[Tuple[str, str, Dict[str, int]]],
) -> ContrastiveDatasetSummary:
    """Summarize training videos with multi-label mask paths and class map."""
    if not entries:
        return ContrastiveDatasetSummary(0, 0, [], {})

    class_names_set: set[str] = set()
    total_frames = 0
    frames_per_class: Dict[str, int] = {}

    for video_path, mask_path, class_value_map in entries:
        class_names_set.update(class_value_map.keys())
        mask_data = load_mask_volume(mask_path)
        total_frames += mask_volume_frame_count(mask_data)
        per_video = count_labeled_frames_for_classes(mask_data, class_value_map)
        for cls, count in per_video.items():
            frames_per_class[cls] = frames_per_class.get(cls, 0) + count

    return ContrastiveDatasetSummary(
        video_count=len(entries),
        total_frames=total_frames,
        class_names=sorted(class_names_set),
        frames_per_class=frames_per_class,
    )


def format_dataset_summary(summary: ContrastiveDatasetSummary) -> str:
    if summary.video_count == 0:
        return "Classes: (none)"
    parts = [
        f"{summary.video_count} video(s), {summary.total_frames} frame(s)",
        f"classes: {', '.join(summary.class_names) or '(none)'}",
    ]
    if summary.frames_per_class:
        detail = ", ".join(
            f"{cls} on {summary.frames_per_class.get(cls, 0)} frame(s)"
            for cls in summary.class_names
        )
        parts.append(detail)
    return " | ".join(parts)


def discover_training_videos_in_directory(directory: str | Path) -> List[Path]:
    return discover_videos_in_directory(directory)


__all__ = [
    "COMBINED_MASK_STEM_SUFFIX",
    "KNOWN_LABEL_ID_TO_NAME",
    "ContrastiveCheckpointMetadata",
    "ContrastiveDatasetSummary",
    "contrastive_checkpoint_filename",
    "count_labeled_frames_for_classes",
    "discover_label_values_in_mask",
    "discover_multilabel_mask",
    "discover_training_videos_in_directory",
    "format_dataset_summary",
    "label_id_to_name",
    "label_values_to_names",
    "load_contrastive_checkpoint",
    "load_video_frame_rgb",
    "multilabel_frame_to_class_masks",
    "save_contrastive_checkpoint",
    "summarize_contrastive_dataset",
    "validate_training_pair",
]
