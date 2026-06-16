"""Helpers for training and running YOLO segmentation in napari."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

MASK_EXTENSIONS = {".tiff", ".tif", ".npy", ".jpg", ".jpeg"}
CLASS_NAME_RE = re.compile(r"\b(\w+)$")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class DatasetSummary:
    """Per-frame training dataset statistics."""

    total_frames: int
    frames_per_class: Dict[str, int]
    class_names: List[str]
    video_count: int


@dataclass
class ExportSpec:
    root: Path
    train_images: Path
    train_masks: Path
    class_names: List[str]
    data_yaml: Path
    total_frames: int = 0
    frames_per_class: Dict[str, int] = field(default_factory=dict)


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def class_name_from_mask_path(mask_path: str | Path) -> str | None:
    """Return the class name encoded as the last word of the mask file stem."""
    stem = Path(mask_path).stem
    match = CLASS_NAME_RE.search(stem)
    return match.group(1) if match else None


def discover_mask_files(video_path: str | Path) -> Dict[str, Path]:
    """Find mask files next to a video; keys are class names from file stems."""
    video_path = Path(video_path).resolve()
    stem = video_path.stem
    mask_dir = video_path.parent
    masks: Dict[str, Path] = {}
    for candidate in sorted(mask_dir.glob(f"{stem} - *")):
        if candidate.suffix.lower() not in MASK_EXTENSIONS:
            continue
        class_name = class_name_from_mask_path(candidate)
        if class_name is None:
            continue
        masks[class_name] = candidate
    return masks


def load_mask_volume(path: str | Path) -> np.ndarray:
    """Load a mask volume from TIFF or NPY."""
    path = Path(path)
    if path.suffix.lower() in {".tiff", ".tif"}:
        import tifffile

        return np.asarray(tifffile.imread(path))
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path))
    if path.suffix.lower() in {".jpg", ".jpeg"}:
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


def video_frame_count(video_path: str | Path) -> int:
    """Return frame count without decoding every pixel."""
    from napari_pecan_py._reader import LazyVideoArray

    return int(LazyVideoArray(str(Path(video_path).resolve())).shape[0])


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


def load_video_rgb_frames(video_path: str | Path) -> np.ndarray:
    """Load all RGB frames from a video as (T, H, W, 3) uint8."""
    from napari_pecan_py._reader import LazyVideoArray

    lazy = LazyVideoArray(str(Path(video_path).resolve()))
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


def summarize_training_dataset(
    video_entries: Sequence[Tuple[str | Path, Dict[str, str | Path]]],
) -> DatasetSummary:
    """Summarize how many frames each class is labeled on across all videos."""
    class_names_set: set[str] = set()
    frames_per_class: Dict[str, int] = {}
    total_frames = 0

    for video_path, masks_by_path in video_entries:
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            t_count = video_frame_count(video_path)
        elif suffix in IMAGE_EXTENSIONS:
            t_count = 1
        else:
            continue

        total_frames += t_count
        class_names_set.update(masks_by_path.keys())

        loaded_masks = {
            cls: load_mask_volume(path) for cls, path in masks_by_path.items()
        }
        _validate_mask_volumes(t_count, loaded_masks, video_path.name)

        for cls, mask_data in loaded_masks.items():
            frames_per_class[cls] = frames_per_class.get(cls, 0) + count_labeled_frames(
                mask_data
            )

    return DatasetSummary(
        total_frames=total_frames,
        frames_per_class=frames_per_class,
        class_names=sorted(class_names_set),
        video_count=len(video_entries),
    )


def format_dataset_summary(summary: DatasetSummary) -> str:
    if summary.total_frames == 0:
        return "No training frames."
    parts = []
    for cls in summary.class_names:
        labeled = summary.frames_per_class.get(cls, 0)
        parts.append(f"{cls}: {labeled}/{summary.total_frames} frames")
    return (
        f"{summary.total_frames} frame(s) from {summary.video_count} video(s). "
        + "; ".join(parts)
    )


def export_videos_seg_dataset(
    video_entries: Sequence[Tuple[str | Path, Dict[str, str | Path]]],
    out_root: str | Path,
) -> ExportSpec:
    """Export multiple on-disk videos and their mask files to a YOLO-seg dataset."""
    from PIL import Image as PILImage
    import yaml

    root = Path(out_root).resolve()
    images_dir = _ensure_dir(root / "images" / "train")
    labels_dir = _ensure_dir(root / "labels" / "train")

    class_names_set: set[str] = set()
    for _, masks in video_entries:
        class_names_set.update(masks.keys())
    class_names = sorted(class_names_set)
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

    for video_path, masks_by_path in video_entries:
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            frames = load_video_rgb_frames(video_path)
        elif suffix in IMAGE_EXTENSIONS:
            frames = load_image_rgb(video_path)[None, ...]
        else:
            raise ValueError(f"Unsupported training input: {video_path}")

        masks_by_class = {
            cls: load_mask_volume(path) for cls, path in masks_by_path.items()
        }
        stem = video_path.stem
        t_count = frames.shape[0]
        _validate_mask_volumes(t_count, masks_by_class, video_path.name)

        for t in range(t_count):
            img = _to_uint8_rgb(frames[t])
            img_name = f"{stem}_frame_{t:04d}.png"
            PILImage.fromarray(img).save(images_dir / img_name)

            lines = _write_yolo_label_lines(
                masks_by_class, class_names, t, img.shape[0], img.shape[1]
            )
            (labels_dir / f"{stem}_frame_{t:04d}.txt").write_text("\n".join(lines))
            total_exported_frames += 1
            for cls in class_names:
                if cls not in masks_by_class:
                    continue
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

    segments = getattr(result.masks, "xy", None)
    if segments is not None and len(segments) > 0:
        for i, poly in enumerate(segments):
            if poly is None or len(poly) < 3:
                continue
            label_val = (
                int(classes[i]) + 1
                if classes is not None and i < len(classes)
                else i + 1
            )
            pts = np.round(poly).astype(np.int32)
            cv2.fillPoly(label_map, [pts], label_val)
        if np.any(label_map):
            return label_map

    mask_data = result.masks.data
    if mask_data is None or mask_data.numel() == 0:
        return None

    masks_np = mask_data.cpu().numpy()
    if masks_np.ndim == 2:
        masks_np = masks_np[None]

    for i in range(masks_np.shape[0]):
        m = (masks_np[i] > 0.5).astype(np.uint8)
        if m.shape != (height, width):
            m = cv2.resize(m, (width, height), interpolation=cv2.INTER_LINEAR)
        label_val = (
            int(classes[i]) + 1
            if classes is not None and i < len(classes)
            else i + 1
        )
        label_map[m > 0] = label_val

    return label_map if np.any(label_map) else None


def train_yolo_seg(
    data_yaml: str | Path,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 50,
    batch: int = 4,
    lr: float = 1e-3,
    device: str = "auto",
    project: str | Path | None = None,
    name: str = "pecan-yolo-seg",
) -> str:
    """Train a YOLO segmentation model using `ultralytics`."""
    from ultralytics import YOLO

    yolo = YOLO(model_name)
    results = yolo.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        lr0=lr,
        device=device,
        project=str(project) if project is not None else None,
        name=name,
    )
    return str(results.best)
