"""Helpers for training YOLO segmentation from a napari session.

This uses the `ultralytics` package (YOLOv8/YOLO11). We export the
current Image layer and its selected mask Labels layers to a small
on-disk dataset and then call `YOLO.train(...)` on that dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class ExportSpec:
    root: Path
    train_images: Path
    train_masks: Path
    class_names: List[str]
    data_yaml: Path


def _ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def export_napari_seg_dataset(
    image_data: np.ndarray,
    masks_by_class: Dict[str, np.ndarray],
    out_root: str | Path,
) -> ExportSpec:
    """Export frames + Napari Labels masks to YOLO-seg polygon annotations.

    Ultralytics YOLO segmentation expects:
    - images in `images/train/*.png|jpg|...`
    - polygon labels in `labels/train/*.txt`
      Each line: `class_idx x1 y1 x2 y2 ...` with normalized coords.
    """
    from PIL import Image as PILImage
    import cv2
    import yaml

    root = Path(out_root).resolve()
    images_dir = _ensure_dir(root / "images" / "train")
    labels_dir = _ensure_dir(root / "labels" / "train")

    if image_data.ndim == 3:
        image_data = image_data[None, ...]
    T, H, W, C = image_data.shape

    # Keep only classes that contain any pixels in the provided masks.
    class_names: List[str] = []
    for cls in sorted(masks_by_class.keys()):
        m = np.asarray(masks_by_class[cls])
        if np.any(m > 0):
            class_names.append(cls)

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
        return ExportSpec(
            root=root,
            train_images=images_dir,
            train_masks=labels_dir,
            class_names=[],
            data_yaml=data_yaml,
        )

    for t in range(T):
        img = image_data[t]
        if img.dtype != np.uint8:
            img = np.clip(img, 0, 255)
            if float(img.max()) <= 1.0:
                img = img * 255.0
            img = img.astype(np.uint8)

        img_name = f"frame_{t:04d}.png"
        PILImage.fromarray(img).save(images_dir / img_name)

        frame_lines: List[str] = []

        for cls_idx, cls in enumerate(class_names):
            m = masks_by_class[cls]
            if np.asarray(m).ndim == 3:
                m2d = np.asarray(m)[t]
            else:
                m2d = np.asarray(m)

            bin_mask = (m2d > 0).astype(np.uint8)
            if bin_mask.sum() == 0:
                continue

            contours, _hier = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not contours:
                continue

            for cnt in contours:
                area = float(cv2.contourArea(cnt))
                if area < 10.0:
                    continue

                # Reduce the number of points for compact .txt labels.
                peri = float(cv2.arcLength(cnt, True))
                if peri > 0:
                    epsilon = 0.002 * peri
                    cnt = cv2.approxPolyDP(cnt, epsilon, True)

                pts = cnt.reshape(-1, 2)
                if pts.shape[0] < 3:
                    continue

                xs = pts[:, 0] / float(W)
                ys = pts[:, 1] / float(H)
                coord_str = " ".join(
                    f"{x:.6f} {y:.6f}" for x, y in zip(xs.tolist(), ys.tolist())
                )
                frame_lines.append(f"{cls_idx} {coord_str}")

        label_path = labels_dir / f"frame_{t:04d}.txt"
        label_path.write_text("\n".join(frame_lines))

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
        root=root,
        train_images=images_dir,
        train_masks=labels_dir,
        class_names=class_names,
        data_yaml=data_yaml,
    )


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
    return str(results.best)  # path to best weights


