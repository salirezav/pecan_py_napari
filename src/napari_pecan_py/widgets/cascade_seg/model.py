"""Cascaded U-Net segmentation: train, infer, and checkpoint I/O."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from .hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    LABEL_ID_BY_CLASS,
    LABEL_MERGE_PRIORITY,
    NUT_REGION_CLASSES,
    PARENT_REGION,
    STAGE_ABSENCE_LOSS_WEIGHT,
    STAGE_INFERENCE_THRESHOLD,
    STAGE_LOSS_WEIGHT,
    STAGE_POS_WEIGHT,
    format_hierarchy_chain,
    nut_region_mask_tensor,
    ordered_chain_classes,
    parent_mask_names,
    stage_input_channels,
)

from napari_pecan_py.widgets.yolo_seg.model import (
    IMAGE_EXTENSIONS,
    _mask_frame,
    _split_name_for_frame,
    _to_uint8_rgb,
    _validate_mask_volumes,
    image_volume_to_rgb_frames,
    load_image_rgb,
    load_masks_by_class_from_paths,
    load_video_rgb_frames,
    plan_train_val_split,
    resolve_yolo_device,
)

BACKEND_CASCADE = "cascade"
ARCH_UNET = "unet"
ARCH_UNETPP = "unetplusplus"
# Kept for checkpoint / API compatibility; maps to ``Unet`` (smp has no Unet3Plus).
ARCH_UNET3PLUS = ARCH_UNET
SUPPORTED_ARCHITECTURES = (ARCH_UNET, ARCH_UNETPP)


@dataclass
class CascadeTrainConfig:
    encoder_name: str = "mobilenet_v2"
    architecture: str = ARCH_UNETPP
    image_size: int = 384
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-3
    val_fraction: float = 0.2
    split_by: str = "video"
    horizontal_flip_prob: float = 0.5
    require_all_classes_in_frame: bool = False
    # Fraction of training steps that use GT parent masks (rest use predicted parents).
    teacher_forcing_mix: float = 0.5
    init_weights_path: str | None = None
    # On pecan-only frames, train crack/kernel heads to stay off (reduces over-segmentation).
    train_absent_inner_classes: bool = True


def count_frames_by_label_presence(
    samples: Sequence[Tuple],
    classes: Sequence[str],
) -> Dict[str, int]:
    """Count frames where each class has at least one foreground pixel."""
    counts = {cls: 0 for cls in classes}
    for item in samples:
        masks = item[2] if len(item) == 3 else item[1]
        for cls in classes:
            if cls in masks and np.any(masks[cls]):
                counts[cls] += 1
    return counts


def count_pecan_only_frames(
    samples: Sequence[Tuple],
    inner_classes: Sequence[str] = ("Crack", "Kernel"),
) -> int:
    """Frames with pecan labels but no crack or kernel pixels."""
    n = 0
    for item in samples:
        masks = item[2] if len(item) == 3 else item[1]
        if "Pecan" not in masks or not np.any(masks["Pecan"]):
            continue
        if all(
            cls not in masks or not np.any(masks[cls]) for cls in inner_classes
        ):
            n += 1
    return n


def count_partial_inner_label_frames(
    samples: Sequence[Tuple],
) -> Dict[str, int]:
    """Frames with pecan plus crack and/or kernel, but not every inner class."""
    counts = {"crack_no_kernel": 0, "kernel_no_crack": 0, "both_inner": 0}
    for item in samples:
        masks = item[2] if len(item) == 3 else item[1]
        if "Pecan" not in masks or not np.any(masks["Pecan"]):
            continue
        has_crack = "Crack" in masks and np.any(masks["Crack"])
        has_kernel = "Kernel" in masks and np.any(masks["Kernel"])
        if has_crack and has_kernel:
            counts["both_inner"] += 1
        elif has_crack:
            counts["crack_no_kernel"] += 1
        elif has_kernel:
            counts["kernel_no_crack"] += 1
    return counts


def _mask_has_foreground(masks, class_name: str) -> bool:
    import torch

    if class_name not in masks:
        return False
    value = masks[class_name]
    if isinstance(value, torch.Tensor):
        return float(value.sum()) > 0
    return bool(np.any(value))


def _is_intact_shell_frame(masks) -> bool:
    """True when labels show pecan shell but neither crack nor kernel."""
    if not _mask_has_foreground(masks, "Pecan"):
        return False
    return not _mask_has_foreground(masks, "Crack") and not _mask_has_foreground(
        masks, "Kernel"
    )


def _should_apply_absence_loss(
    stage_name: str,
    masks,
    *,
    train_absent_inner_classes: bool,
) -> bool:
    """Only teach crack/kernel=off on intact shell frames, not partial labels."""
    if not train_absent_inner_classes:
        return False
    if stage_name not in ("Crack", "Kernel"):
        return False
    return _is_intact_shell_frame(masks)


# Thin structures vanish on downscale; dilate once after resize to keep gradient signal.
_THIN_MASK_CLASSES = frozenset({"Crack"})


def frame_has_all_class_labels(
    masks: Dict[str, np.ndarray],
    required_classes: Sequence[str],
) -> bool:
    """True when every required class has at least one foreground pixel."""
    for cls in required_classes:
        if cls not in masks or not np.any(masks[cls]):
            return False
    return True


def filter_samples_require_all_classes(
    samples: Sequence[Tuple],
    required_classes: Sequence[str],
) -> list:
    """Keep samples whose masks contain foreground for every required class."""
    required = list(required_classes)
    kept: list = []
    for item in samples:
        masks = item[2] if len(item) == 3 else item[1]
        if frame_has_all_class_labels(masks, required):
            kept.append(item)
    return kept


def summarize_cascade_frame_usage(
    video_entries: Sequence[Tuple[str, Dict[str, str]]],
    selected_classes: Sequence[str],
    *,
    val_fraction: float = 0.2,
    split_by: str = "video",
    require_all_classes_in_frame: bool = False,
) -> str:
    """Human-readable frame counts for cascade training preview."""
    stages = ordered_chain_classes(selected_classes)
    if not stages or not video_entries:
        return "Classes: (none)"

    video_paths, frame_counts, raw = _collect_video_samples(video_entries, stages)
    total = len(raw)
    if require_all_classes_in_frame:
        filtered = filter_samples_require_all_classes(raw, stages)
        dropped = total - len(filtered)
        raw = filtered
        prefix = (
            f"{len(raw)} frame(s) with all selected classes "
            f"({dropped} frame(s) dropped, {total} total)"
        )
    else:
        prefix = f"{total} frame(s) with pecan labels"

    if not raw:
        return f"{prefix}; none available for training."

    train_samples, val_samples = _split_samples(
        raw,
        video_paths,
        frame_counts,
        val_fraction=val_fraction,
        split_by=split_by,
    )
    pecan_only_train = count_pecan_only_frames(train_samples)
    partial = count_partial_inner_label_frames(train_samples)
    extra_parts: list[str] = []
    if pecan_only_train > 0:
        extra_parts.append(f"{pecan_only_train} intact-shell (no crack/kernel)")
    if partial["crack_no_kernel"] > 0:
        extra_parts.append(
            f"{partial['crack_no_kernel']} with crack but no kernel label (kernel loss skipped)"
        )
    extra = f"; {', '.join(extra_parts)}" if extra_parts else ""
    return (
        f"{prefix}; split → {len(train_samples)} train, {len(val_samples)} val "
        f"({', '.join(stages)}){extra}"
    )


def detect_seg_checkpoint_backend(weights_path: str | Path) -> str:
    """Return ``cascade``, ``unet``, or ``yolo`` based on checkpoint contents."""
    path = Path(weights_path)
    if not path.is_file():
        return "yolo"
    try:
        import torch

        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict):
            backend = ckpt.get("backend")
            if backend == BACKEND_CASCADE:
                return BACKEND_CASCADE
            if backend == "unet":
                return "unet"
    except Exception:
        pass
    return "yolo"


def guess_cascade_save_suffix(
    weights_path: str | Path,
    *,
    fallback_index: int = 1,
) -> tuple[str, bool]:
    """Guess save suffix from ``cascade - [Classes].pt`` filenames."""
    from napari_pecan_py.widgets.yolo_seg.model import WEIGHTS_CLASSES_RE

    stem = Path(weights_path).stem
    match = WEIGHTS_CLASSES_RE.search(stem)
    if match:
        classes = match.group(1).strip()
        if classes:
            return f" - {classes}", True
    return f" - Cascade Seg [{fallback_index}]", False


def merge_stage_masks_to_label_map(
    stage_masks: Dict[str, np.ndarray],
    *,
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> np.ndarray:
    """Merge per-class binary masks into one exclusive uint8 label volume.

    Crack and Kernel are siblings under Pecan; where predictions overlap, later
    classes in LABEL_MERGE_PRIORITY win (display priority, not spatial nesting).
    """
    from .hierarchy import LABEL_MERGE_PRIORITY

    if not stage_masks:
        raise ValueError("No stage masks to merge.")

    sample = next(iter(stage_masks.values()))
    h, w = sample.shape[-2], sample.shape[-1]
    pecan = _bool_mask(stage_masks.get("Pecan"), h, w)
    crack = _bool_mask(stage_masks.get("Crack"), h, w)
    kernel = _bool_mask(stage_masks.get("Kernel"), h, w)
    crack = crack & ~kernel
    pecan = pecan & ~crack & ~kernel

    label_map = np.zeros((h, w), dtype=np.uint8)
    for class_name in LABEL_MERGE_PRIORITY:
        if class_name == "Pecan" and np.any(pecan):
            label_map[pecan] = LABEL_ID_BY_CLASS["Pecan"]
        elif class_name == "Crack" and np.any(crack):
            label_map[crack] = LABEL_ID_BY_CLASS["Crack"]
        elif class_name == "Kernel" and np.any(kernel):
            label_map[kernel] = LABEL_ID_BY_CLASS["Kernel"]
    return label_map


def filter_stage_masks_to_largest_nut(
    stage_masks: Dict[str, np.ndarray],
    *,
    close_kernel: int = 11,
    close_iters: int = 2,
) -> Dict[str, np.ndarray]:
    """Keep predictions inside the largest morphologically closed nut blob."""
    import cv2
    from scipy import ndimage

    if not stage_masks:
        return stage_masks
    sample = next(iter(stage_masks.values()))
    union = np.zeros(sample.shape[-2:], dtype=np.uint8)
    for mask in stage_masks.values():
        m = np.asarray(mask)
        if m.ndim == 3:
            m = m[0]
        union |= (m > 0).astype(np.uint8)
    if not np.any(union):
        return stage_masks

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_kernel, close_kernel))
    closed = cv2.morphologyEx(union, cv2.MORPH_CLOSE, k, iterations=close_iters)
    labeled, n = ndimage.label(closed)
    if n <= 0:
        return stage_masks
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    keep = labeled == int(counts.argmax())
    return {
        name: ((np.asarray(mask) > 0) & keep).astype(np.uint8)
        for name, mask in stage_masks.items()
    }


def _exclusive_part_masks(masks: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Make pecan / crack / kernel mutually exclusive (kernel > crack > pecan)."""
    pecan = (masks.get("Pecan", 0) > 0)
    crack = (masks.get("Crack", 0) > 0)
    kernel = (masks.get("Kernel", 0) > 0)
    crack = crack & np.logical_not(kernel)
    pecan = pecan & np.logical_not(crack) & np.logical_not(kernel)
    out = dict(masks)
    if "Pecan" in masks:
        out["Pecan"] = pecan.astype(masks["Pecan"].dtype)
    if "Crack" in masks:
        out["Crack"] = crack.astype(masks["Crack"].dtype)
    if "Kernel" in masks:
        out["Kernel"] = kernel.astype(masks["Kernel"].dtype)
    return out


def _bool_mask(mask: np.ndarray | None, height: int, width: int) -> np.ndarray:
    if mask is None:
        return np.zeros((height, width), dtype=bool)
    m = np.asarray(mask)
    if m.ndim == 3:
        m = m[0]
    if m.shape != (height, width):
        import cv2

        m = cv2.resize(m.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)
    return m > 0.5


def _build_segmentation_model(
    architecture: str,
    encoder_name: str,
    in_channels: int,
    *,
    classes: int = 1,
):
    import segmentation_models_pytorch as smp

    common = dict(
        encoder_name=encoder_name,
        encoder_weights="imagenet",
        in_channels=in_channels,
        classes=int(classes),
        activation=None,
    )
    arch = str(architecture).lower()
    if arch in (ARCH_UNETPP, "unetplusplus"):
        return smp.UnetPlusPlus(**common)
    if arch in (ARCH_UNET, ARCH_UNET3PLUS, "unet3plus"):
        return smp.Unet(**common)
    raise ValueError(f"Unsupported cascade architecture: {architecture}")


def _resolve_torch_device(device: str):
    import torch

    resolved = resolve_yolo_device(device)
    if resolved == "cpu":
        return torch.device("cpu")
    if resolved.isdigit():
        return torch.device(f"cuda:{resolved}")
    if str(resolved).lower().startswith("cuda"):
        return torch.device(resolved)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CascadedSegmenter:
    """Cascade of per-class U-Net stages with parent-mask conditioning."""

    def __init__(
        self,
        trained_stages: Sequence[str],
        *,
        encoder_name: str = "efficientnet-b0",
        architecture: str = ARCH_UNETPP,
        chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
    ) -> None:
        import torch.nn as nn

        self.chain = list(chain)
        self.trained_stages = ordered_chain_classes(trained_stages, self.chain)
        if not self.trained_stages:
            raise ValueError("At least one cascade stage must be trained.")
        if self.trained_stages[0] != self.chain[0]:
            raise ValueError(
                f"Cascade training must include '{self.chain[0]}' when training "
                f"inner classes ({format_hierarchy_chain(self.chain)})."
            )

        self.encoder_name = encoder_name
        self.architecture = architecture
        self.stages = nn.ModuleDict(
            {
                name: _build_segmentation_model(
                    architecture, encoder_name, stage_input_channels(name)
                )
                for name in self.trained_stages
            }
        )

    def forward_tensors(
        self,
        image,
        context_masks=None,
        *,
        teacher_forcing: bool = False,
        gt_parent_prob: float = 1.0,
    ):
        """Run the cascade; returns logits per trained stage."""
        import torch

        outputs: Dict[str, torch.Tensor] = {}
        predicted_probs: Dict[str, torch.Tensor] = {}

        for name in self.trained_stages:
            parents = parent_mask_names(name)
            extras = []
            for parent in parents:
                use_gt_parent = (
                    teacher_forcing
                    and context_masks is not None
                    and parent in context_masks
                    and (gt_parent_prob >= 1.0 or random.random() < gt_parent_prob)
                )
                mask = _cascade_conditioning_mask(
                    parent,
                    name,
                    context_masks=context_masks,
                    predicted_probs=predicted_probs,
                    use_gt=use_gt_parent,
                )
                extras.append(mask)
            stage_in = image if not extras else torch.cat([image, *extras], dim=1)
            logits = self.stages[name](stage_in)
            outputs[name] = logits
            predicted_probs[name] = torch.sigmoid(logits)
        return outputs

    def predict_probs(
        self,
        image,
        context_masks=None,
        *,
        teacher_forcing: bool = False,
        gt_parent_prob: float = 1.0,
    ):
        import torch

        self.stages.eval()
        with torch.no_grad():
            logits = self.forward_tensors(
                image,
                context_masks=context_masks,
                teacher_forcing=teacher_forcing,
                gt_parent_prob=gt_parent_prob,
            )
            return {name: torch.sigmoid(logit) for name, logit in logits.items()}

    def state_dict(self):
        return self.stages.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.stages.load_state_dict(state_dict)

    def to(self, device):
        self.stages.to(device)
        return self

    def train(self, mode: bool = True):
        self.stages.train(mode)
        return self

    def eval(self):
        return self.train(False)


def save_cascade_checkpoint(
    path: str | Path,
    model: CascadedSegmenter,
    *,
    image_size: int,
    label_id_by_class: Dict[str, int] | None = None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": BACKEND_CASCADE,
            "model_state": model.state_dict(),
            "architecture": model.architecture,
            "encoder_name": model.encoder_name,
            "hierarchy_chain": list(model.chain),
            "trained_stages": list(model.trained_stages),
            "image_size": int(image_size),
            "label_id_by_class": dict(label_id_by_class or LABEL_ID_BY_CLASS),
        },
        path,
    )


def load_cascade_checkpoint(weights_path: str | Path) -> tuple[CascadedSegmenter, dict]:
    import torch

    ckpt = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("backend") != BACKEND_CASCADE:
        raise ValueError(f"Not a cascade segmentation checkpoint: {weights_path}")

    model = CascadedSegmenter(
        ckpt.get("trained_stages") or ckpt.get("trained_classes") or [],
        encoder_name=str(ckpt.get("encoder_name", "efficientnet-b0")),
        architecture=str(ckpt.get("architecture", ARCH_UNET3PLUS)),
        chain=ckpt.get("hierarchy_chain") or DEFAULT_HIERARCHY_CHAIN,
    )
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def _dice_bce_loss(logits, target, region_mask=None, *, pos_weight: float = 1.0):
    import torch
    import torch.nn.functional as F

    target = target.float()
    pw = torch.tensor([float(pos_weight)], device=logits.device, dtype=logits.dtype)
    probs = torch.sigmoid(logits)
    smooth = 1.0

    if region_mask is not None:
        region = (region_mask > 0.5).float()
        # Siblings only exist inside the parent region; penalize the full frame so the
        # model cannot learn high logits outside pecan (which inference then discards).
        masked_target = target * region
        bce = F.binary_cross_entropy_with_logits(
            logits, masked_target, pos_weight=pw, reduction="mean"
        )

        intersection = (probs * masked_target).sum()
        fg_probs = (probs * region).sum()
        fg_target = masked_target.sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (fg_probs + fg_target + smooth)
    else:
        bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
        intersection = (probs * target).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (
            probs.sum() + target.sum() + smooth
        )
    return bce + dice


def _foreground_iou(
    logits,
    target,
    region_mask=None,
    *,
    threshold: float = 0.35,
) -> float:
    """IoU for foreground pixels (optionally restricted to a parent region)."""
    import torch

    probs = torch.sigmoid(logits)
    pred = (probs > threshold).float()
    target = target.float()
    if region_mask is not None:
        region = (region_mask > 0.5).float()
        pred = pred * region
        target = target * region
    intersection = (pred * target).sum()
    union = ((pred + target) > 0).float().sum()
    if float(union) <= 0:
        return float("nan")
    return float((intersection / union).detach().cpu())


def _stage_loss_weight(stage_name: str) -> float:
    return float(STAGE_LOSS_WEIGHT.get(stage_name, 1.0))


def _stage_pos_weight(stage_name: str) -> float:
    return float(STAGE_POS_WEIGHT.get(stage_name, 1.0))


def _stage_inference_threshold(stage_name: str) -> float:
    return float(STAGE_INFERENCE_THRESHOLD.get(stage_name, 0.5))


def _device_status_message(device) -> str:
    import torch

    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        try:
            name = torch.cuda.get_device_name(idx)
        except Exception:
            name = "CUDA"
        return f"GPU ({name})"
    return "CPU — expect several minutes per batch with 3 U-Net stages at 512px"


def _maybe_enable_cuda_fast_math(device) -> None:
    import torch

    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        if seconds < 10:
            return f"{seconds:.1f}s"
        return f"{seconds:.0f}s"
    minutes, rem = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m {rem}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _loss_region_for_stage(stage_name: str, masks) -> "torch.Tensor | None":
    """Spatial region for sibling-stage loss / IoU (whole nut, not shell-only pecan)."""
    import torch

    if stage_name == DEFAULT_HIERARCHY_CHAIN[0]:
        return None
    if stage_name not in ("Crack", "Kernel"):
        parent = PARENT_REGION.get(stage_name)
        if parent is not None and parent in masks:
            return (masks[parent] > 0.5).float()
        return None
    try:
        return nut_region_mask_tensor(masks, hard=True)
    except ValueError:
        return None


def _cascade_conditioning_mask(
    parent: str,
    stage_name: str,
    *,
    context_masks,
    predicted_probs,
    use_gt: bool,
):
    """Parent channel for inner cascade stages (nut region, not shell-only)."""
    import torch

    if parent == "Pecan" and stage_name in ("Crack", "Kernel"):
        if use_gt and context_masks is not None:
            return nut_region_mask_tensor(context_masks, hard=True)
        pred_source = dict(predicted_probs)
        dilate = 12 if stage_name == "Kernel" else 8
        if pred_source:
            return nut_region_mask_tensor(pred_source, hard=True, dilate=dilate)
        if context_masks is not None:
            return nut_region_mask_tensor(context_masks, hard=True, dilate=dilate)
        raise RuntimeError(f"Missing masks to build nut region for stage '{stage_name}'.")
    if use_gt and context_masks is not None and parent in context_masks:
        return context_masks[parent]
    if parent in predicted_probs:
        mask = predicted_probs[parent]
        return (mask > 0.5).float()
    raise RuntimeError(f"Missing parent mask '{parent}' for stage '{stage_name}'.")


def _compute_stage_losses(
    logits,
    target_tensors,
    context,
    device,
    *,
    train_absent_inner_classes: bool = True,
):
    """Return total loss and per-stage loss values for logging."""
    import torch

    stage_losses: Dict[str, float] = {}
    total = torch.tensor(0.0, device=device)
    region_source = {**target_tensors, **context}
    for stage_name, stage_logits in logits.items():
        if stage_name not in target_tensors:
            continue
        target = target_tensors[stage_name].to(device)
        target_sum = float(target.sum())
        is_inner = stage_name != DEFAULT_HIERARCHY_CHAIN[0]
        if is_inner and target_sum <= 0:
            if not _should_apply_absence_loss(
                stage_name,
                region_source,
                train_absent_inner_classes=train_absent_inner_classes,
            ):
                continue
            region = _loss_region_for_stage(stage_name, region_source)
            if region is None and "Pecan" in region_source:
                region = (region_source["Pecan"] > 0.5).float().to(device)
            elif region is not None:
                region = region.to(device)
            stage_loss = _dice_bce_loss(
                stage_logits,
                torch.zeros_like(target),
                region,
                pos_weight=1.0,
            )
            absence_w = float(STAGE_ABSENCE_LOSS_WEIGHT.get(stage_name, 0.35))
            weighted = stage_loss * _stage_loss_weight(stage_name) * absence_w
        else:
            region = _loss_region_for_stage(stage_name, region_source)
            if region is not None:
                region = region.to(device)
            stage_loss = _dice_bce_loss(
                stage_logits,
                target,
                region,
                pos_weight=_stage_pos_weight(stage_name),
            )
            weighted = stage_loss * _stage_loss_weight(stage_name)
        if torch.isfinite(stage_loss):
            stage_losses[stage_name] = float(stage_loss.detach().cpu())
            total = total + weighted
    return total, stage_losses


def _prepare_batch_tensors(batch, device, image_size: int):
    import torch

    images = []
    targets: List[Dict[str, np.ndarray]] = []
    class_names = sorted({cls for frame, masks in batch for cls in masks})
    for frame, masks in batch:
        img = _resize_rgb(frame, image_size)
        mask_dict = _exclusive_part_masks(
            {
                cls: masks.get(cls, np.zeros(frame.shape[:2], dtype=np.uint8))
                for cls in class_names
            }
        )
        mask_dict = {
            cls: _resize_mask(mask_dict[cls], image_size, class_name=cls)
            for cls in class_names
        }
        images.append(img)
        targets.append(mask_dict)

    image_t = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 255.0
    target_tensors: Dict[str, torch.Tensor] = {}
    for cls in class_names:
        stacked = np.stack([masks[cls] for masks in targets], axis=0)[:, None, ...]
        target_tensors[cls] = torch.from_numpy(stacked).float()
    return image_t.to(device), target_tensors, class_names


def _resize_rgb(frame: np.ndarray, size: int) -> np.ndarray:
    import cv2

    return cv2.resize(_to_uint8_rgb(frame), (size, size), interpolation=cv2.INTER_LINEAR)


def _resize_mask(mask: np.ndarray, size: int, *, class_name: str | None = None) -> np.ndarray:
    import cv2

    raw = np.asarray(mask)
    m = (raw > 0).astype(np.uint8)
    orig_max = max(m.shape[:2])
    out = cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST)
    if class_name in _THIN_MASK_CLASSES and orig_max > size and np.any(out):
        out = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
    return out


def _augment_sample(frame: np.ndarray, masks: Dict[str, np.ndarray], p_flip: float):
    if random.random() >= p_flip:
        return frame, masks
    frame = np.flip(frame, axis=1).copy()
    return frame, {cls: np.flip(m, axis=1).copy() for cls, m in masks.items()}


def _collate_frame_samples(batch):
    """Return batch as a list; default collate cannot stack variable-size frames."""
    return batch


def _collect_video_samples(
    video_entries: Sequence[Tuple[str, Dict[str, str]]],
    selected_classes: Sequence[str],
) -> Tuple[List[Path], List[int], List[Tuple[int, np.ndarray, Dict[str, np.ndarray]]]]:
    """Return paths, frame counts, and (video_idx, frame, masks) tuples."""
    video_paths: List[Path] = []
    frame_counts: List[int] = []
    samples: List[Tuple[int, np.ndarray, Dict[str, np.ndarray]]] = []

    for video_idx, (video_path, masks_by_path) in enumerate(video_entries):
        video_path = Path(video_path).resolve()
        suffix = video_path.suffix.lower()
        if suffix in {".mp4", ".avi", ".mov", ".mkv"}:
            frames = load_video_rgb_frames(video_path)
        elif suffix in IMAGE_EXTENSIONS:
            frames = load_image_rgb(video_path)[None, ...]
        else:
            raise ValueError(f"Unsupported training input: {video_path}")

        masks_by_class = load_masks_by_class_from_paths(masks_by_path)
        t_count = frames.shape[0]
        _validate_mask_volumes(t_count, masks_by_class, video_path.name)

        video_paths.append(video_path)
        frame_counts.append(t_count)

        for t in range(t_count):
            masks_t = {
                cls: (_mask_frame(masks_by_class[cls], t) > 0).astype(np.uint8)
                for cls in selected_classes
                if cls in masks_by_class
            }
            if DEFAULT_HIERARCHY_CHAIN[0] not in masks_t:
                continue
            samples.append((video_idx, _to_uint8_rgb(frames[t]), masks_t))

    return video_paths, frame_counts, samples


def _split_samples(
    samples: Sequence[Tuple[int, np.ndarray, Dict[str, np.ndarray]]],
    video_paths: Sequence[Path],
    frame_counts: Sequence[int],
    *,
    val_fraction: float,
    split_by: str,
) -> tuple[list, list]:
    effective_split_by, val_video_indices, val_frame_keys = plan_train_val_split(
        [str(p) for p in video_paths],
        list(frame_counts),
        val_fraction,
        split_by,
    )
    per_video_t: Dict[int, int] = {}
    train_samples = []
    val_samples = []
    for video_idx, frame, masks in samples:
        t = per_video_t.get(video_idx, 0)
        per_video_t[video_idx] = t + 1
        split = _split_name_for_frame(
            video_idx, t, effective_split_by, val_video_indices, val_frame_keys
        )
        if split == "val":
            val_samples.append((frame, masks))
        else:
            train_samples.append((frame, masks))
    return train_samples, val_samples


def train_cascade_segmenter(
    video_entries: Sequence[Tuple[str, Dict[str, str]]],
    output_dir: str | Path,
    device: str,
    config: CascadeTrainConfig,
    *,
    selected_classes: Sequence[str],
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> str:
    """Train a cascaded segmenter; returns path to best checkpoint."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    def log(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)

    trained_stages = ordered_chain_classes(selected_classes)
    if not trained_stages:
        raise ValueError("No training classes selected.")
    if trained_stages[0] != DEFAULT_HIERARCHY_CHAIN[0]:
        raise ValueError(
            f"Include '{DEFAULT_HIERARCHY_CHAIN[0]}' when training cascade stages "
            f"({format_hierarchy_chain()})."
        )

    video_paths, frame_counts, raw_samples = _collect_video_samples(
        video_entries, trained_stages
    )
    if not raw_samples:
        raise ValueError("No labeled frames found for cascade training.")

    if config.require_all_classes_in_frame:
        before = len(raw_samples)
        raw_samples = filter_samples_require_all_classes(raw_samples, trained_stages)
        log(
            f"All-class filter: kept {len(raw_samples)}/{before} frame(s) "
            f"with {', '.join(trained_stages)}."
        )
        if not raw_samples:
            raise ValueError(
                "No frames contain all selected classes. "
                "Disable the all-class filter or add more fully labeled frames."
            )

    train_samples, val_samples = _split_samples(
        raw_samples,
        video_paths,
        frame_counts,
        val_fraction=config.val_fraction,
        split_by=config.split_by,
    )
    if not train_samples:
        raise ValueError("Training split is empty.")

    def _label_coverage(samples: Sequence[Tuple[np.ndarray, Dict[str, np.ndarray]]]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for _, masks in samples:
            for cls, arr in masks.items():
                if np.any(arr):
                    counts[cls] = counts.get(cls, 0) + 1
        return counts

    train_cov = _label_coverage(train_samples)
    log(
        "Labeled frames (train): "
        + ", ".join(
            f"{cls}={train_cov.get(cls, 0)}/{len(train_samples)}"
            for cls in trained_stages
        )
    )
    if trained_stages[0] in train_cov:
        for inner in trained_stages[1:]:
            labeled = train_cov.get(inner, 0)
            if labeled == 0:
                log(f"WARNING: no training frames contain '{inner}' labels.")
            elif labeled < len(train_samples) * 0.1:
                log(
                    f"WARNING: '{inner}' labels are sparse "
                    f"({labeled}/{len(train_samples)} frames)."
                )
    pecan_only = count_pecan_only_frames(train_samples)
    partial = count_partial_inner_label_frames(train_samples)
    if partial["crack_no_kernel"] > 0:
        log(
            f"Partial labels: {partial['crack_no_kernel']} train frame(s) have crack "
            f"but no kernel mask — kernel loss skipped (not treated as 'no kernel')."
        )
    if pecan_only > 0 and config.train_absent_inner_classes:
        log(
            f"Intact-shell train frames: {pecan_only}/{len(train_samples)} "
            "(crack/kernel absence loss applied)."
        )

    class FrameDataset(Dataset):
        def __init__(self, items, flip_prob: float):
            self.items = list(items)
            self.flip_prob = flip_prob

        def __len__(self):
            return len(self.items)

        def __getitem__(self, idx):
            frame, masks = self.items[idx]
            return _augment_sample(frame, masks, self.flip_prob)

    train_loader = DataLoader(
        FrameDataset(train_samples, config.horizontal_flip_prob),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=_collate_frame_samples,
    )
    val_loader = (
        DataLoader(
            FrameDataset(val_samples, 0.0),
            batch_size=config.batch_size,
            shuffle=False,
            collate_fn=_collate_frame_samples,
        )
        if val_samples
        else None
    )

    torch_device = _resolve_torch_device(device)
    _maybe_enable_cuda_fast_math(torch_device)
    train_batches_total = max(1, (len(train_samples) + config.batch_size - 1) // config.batch_size)
    val_batches_total = (
        max(1, (len(val_samples) + config.batch_size - 1) // config.batch_size)
        if val_samples
        else 0
    )
    progress_units_per_epoch = train_batches_total + val_batches_total
    progress_units_total = max(1, progress_units_per_epoch * config.epochs)
    log(f"Device: {_device_status_message(torch_device)}")
    if torch_device.type == "cpu":
        log(
            "Training on CPU with 3 separate U-Net++ models is very slow. "
            "Use Device: cuda:0 if available, or try encoder mobilenet_v2 / image size 384."
        )
    log(
        f"Loading {config.encoder_name} weights and building {len(trained_stages)} "
        f"cascade stage(s)…"
    )
    log("First run may download ImageNet encoder weights.")
    if config.init_weights_path:
        log(f"Fine-tuning from checkpoint: {config.init_weights_path}")
        model, ckpt = load_cascade_checkpoint(config.init_weights_path)
        ckpt_stages = list(ckpt.get("trained_stages") or [])
        if ckpt_stages != trained_stages:
            log(
                f"WARNING: checkpoint stages {ckpt_stages} differ from selected "
                f"{trained_stages} — weights may not align."
            )
        ckpt_size = int(ckpt.get("image_size") or 0)
        if ckpt_size and ckpt_size != config.image_size:
            log(
                f"WARNING: checkpoint image size {ckpt_size} ≠ training "
                f"{config.image_size}px — use the same size for fine-tuning."
            )
        model = model.to(torch_device)
    else:
        model = CascadedSegmenter(
            trained_stages,
            encoder_name=config.encoder_name,
            architecture=config.architecture,
        ).to(torch_device)
    optimizer = torch.optim.AdamW(model.stages.parameters(), lr=config.learning_rate)
    log("Model ready. Starting epoch 1…")
    if config.init_weights_path:
        log(f"Fine-tune learning rate: {config.learning_rate:g}")
    if config.train_absent_inner_classes:
        log(
            "Intact-shell frames (no crack/kernel labels) teach crack/kernel heads to stay off. "
            "Frames with crack but no kernel label do NOT penalize kernel — partial labels."
        )
    log(
        f"Parent-mask mix: {config.teacher_forcing_mix:.0%} GT / "
        f"{1.0 - config.teacher_forcing_mix:.0%} predicted (matches inference)."
    )
    log(
        "Crack/Kernel loss uses the whole-nut region (pecan ∪ crack ∪ kernel), "
        "not shell-only pecan — required because kernel is interior to the shell."
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val = float("inf")

    log(
        f"Cascade stages: {' → '.join(trained_stages)} "
        f"({config.architecture}, {config.encoder_name})"
    )
    log(f"Train frames: {len(train_samples)}, val frames: {len(val_samples)}")
    log(f"Hierarchy: {format_hierarchy_chain()}")
    log(
        "Loss = sum of stage BCE+Dice (Crack/Kernel weighted 5×). "
        "Watch val IoU — it should rise if inner classes are learning."
    )

    import torch

    use_amp = torch_device.type == "cuda"
    for epoch in range(1, config.epochs + 1):
        if cancel_callback and cancel_callback():
            raise RuntimeError("Training stopped by user.")

        epoch_start = time.perf_counter()
        model.train(True)
        train_loss = 0.0
        train_batches = 0
        epoch_stage_totals: Dict[str, float] = {name: 0.0 for name in trained_stages}
        log(f"Epoch {epoch}/{config.epochs}: training ({train_batches_total} batch(es))…")
        for batch in train_loader:
            if cancel_callback and cancel_callback():
                raise RuntimeError("Training stopped by user.")
            batch_start = time.perf_counter()
            if train_batches == 0:
                log("  batch 1: preparing tensors…")
            image_t, target_tensors, _ = _prepare_batch_tensors(
                batch, torch_device, config.image_size
            )
            context = {cls: target_tensors[cls].to(torch_device) for cls in target_tensors}
            if train_batches == 0:
                log("  batch 1: forward + backward (3 stages)…")
            with torch.autocast(device_type=torch_device.type, enabled=use_amp):
                logits = model.forward_tensors(
                    image_t,
                    context_masks=context,
                    teacher_forcing=True,
                    gt_parent_prob=float(config.teacher_forcing_mix),
                )
                loss, stage_losses = _compute_stage_losses(
                    logits,
                    target_tensors,
                    context,
                    torch_device,
                    train_absent_inner_classes=config.train_absent_inner_classes,
                )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            batch_loss = float(loss.detach().cpu())
            train_loss += batch_loss
            for name, value in stage_losses.items():
                epoch_stage_totals[name] = epoch_stage_totals.get(name, 0.0) + value
            train_batches += 1
            running_avg = train_loss / train_batches
            batch_elapsed = time.perf_counter() - batch_start
            epoch_elapsed = time.perf_counter() - epoch_start
            avg_batch = epoch_elapsed / train_batches
            remaining_batches = train_batches_total - train_batches
            eta = avg_batch * remaining_batches
            stage_str = "  ".join(f"{k}={v:.3f}" for k, v in stage_losses.items())
            should_log = (
                train_batches == 1
                or train_batches == train_batches_total
                or train_batches % 10 == 0
            )
            if should_log:
                log(
                    f"  train {train_batches}/{train_batches_total}  "
                    f"loss={batch_loss:.4f}  avg={running_avg:.4f}  "
                    f"{stage_str}  "
                    f"({_format_duration(batch_elapsed)}, ETA {_format_duration(eta)})"
                )
            if progress_callback:
                done_units = (epoch - 1) * progress_units_per_epoch + train_batches
                progress_callback(done_units, progress_units_total)

        avg_train = train_loss / max(train_batches, 1)
        stage_avg_str = "  ".join(
            f"{name}={epoch_stage_totals[name] / max(train_batches, 1):.4f}"
            for name in trained_stages
        )
        avg_val = None
        val_stage_totals: Dict[str, float] = {name: 0.0 for name in trained_stages}
        val_iou_totals: Dict[str, float] = {name: 0.0 for name in trained_stages}
        val_iou_counts: Dict[str, int] = {name: 0 for name in trained_stages}
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_batches_total = max(
                1, (len(val_samples) + config.batch_size - 1) // config.batch_size
            )
            log(f"Epoch {epoch}/{config.epochs}: validating ({val_batches_total} batch(es))…")
            with torch.no_grad():
                for batch in val_loader:
                    image_t, target_tensors, _ = _prepare_batch_tensors(
                        batch, torch_device, config.image_size
                    )
                    with torch.autocast(device_type=torch_device.type, enabled=use_amp):
                        infer_logits = model.forward_tensors(
                            image_t, teacher_forcing=False
                        )
                        infer_loss, infer_stage_losses = _compute_stage_losses(
                            infer_logits,
                            target_tensors,
                            {cls: target_tensors[cls].to(torch_device) for cls in target_tensors},
                            torch_device,
                            train_absent_inner_classes=config.train_absent_inner_classes,
                        )
                    val_loss += float(infer_loss.detach().cpu())
                    for name, value in infer_stage_losses.items():
                        val_stage_totals[name] = val_stage_totals.get(name, 0.0) + value
                    gt_regions = {
                        cls: target_tensors[cls].to(torch_device) for cls in target_tensors
                    }
                    for stage_name, stage_logits in infer_logits.items():
                        if stage_name not in target_tensors:
                            continue
                        target = target_tensors[stage_name].to(torch_device)
                        region = _loss_region_for_stage(stage_name, gt_regions)
                        if region is not None:
                            region = region.to(torch_device)
                        iou = _foreground_iou(
                            stage_logits,
                            target,
                            region,
                            threshold=_stage_inference_threshold(stage_name),
                        )
                        if iou == iou:
                            val_iou_totals[stage_name] += iou
                            val_iou_counts[stage_name] += 1
                    val_batches += 1
                    if progress_callback:
                        done_units = (
                            (epoch - 1) * progress_units_per_epoch
                            + train_batches_total
                            + val_batches
                        )
                        progress_callback(done_units, progress_units_total)
            avg_val = val_loss / max(val_batches, 1)
            if avg_val < best_val:
                best_val = avg_val
                save_cascade_checkpoint(best_path, model, image_size=config.image_size)

        if progress_callback and val_loader is None:
            progress_callback(epoch * progress_units_per_epoch, progress_units_total)

        if avg_val is not None:
            val_stage_str = "  ".join(
                f"{name}={val_stage_totals[name] / max(val_batches, 1):.4f}"
                for name in trained_stages
            )
            val_iou_str = "  ".join(
                f"{name}={val_iou_totals[name] / max(val_iou_counts[name], 1):.3f}"
                if val_iou_counts[name] > 0
                else f"{name}=n/a"
                for name in trained_stages
            )
            log(
                f"Epoch {epoch}/{config.epochs}  train={avg_train:.4f}  val={avg_val:.4f}  "
                f"| train stages: {stage_avg_str} | val stages: {val_stage_str} "
                f"| val IoU: {val_iou_str}"
            )
        else:
            log(
                f"Epoch {epoch}/{config.epochs}  train={avg_train:.4f}  "
                f"| train stages: {stage_avg_str}"
            )
            if epoch == config.epochs or epoch % max(1, config.epochs // 5) == 0:
                save_cascade_checkpoint(best_path, model, image_size=config.image_size)

    if not best_path.is_file():
        save_cascade_checkpoint(best_path, model, image_size=config.image_size)

    classes_bracket = ", ".join(trained_stages)
    named = output_dir / f"cascade - [{classes_bracket}].pt"
    if best_path.is_file() and not named.is_file():
        import torch

        torch.save(torch.load(best_path, weights_only=False), named)

    return str(named if named.is_file() else best_path)


def _predict_frame_label_map(
    model: CascadedSegmenter,
    frame: np.ndarray,
    *,
    image_size: int,
    device,
) -> np.ndarray:
    import torch

    h, w = frame.shape[:2]
    resized = _resize_rgb(frame, image_size)
    image_t = (
        torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    probs = model.predict_probs(image_t, teacher_forcing=False)
    stage_masks: Dict[str, np.ndarray] = {}
    import cv2

    for name, prob in probs.items():
        thresh = _stage_inference_threshold(name)
        mask = (prob.squeeze().detach().cpu().numpy() > thresh).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        stage_masks[name] = mask

    stage_masks = filter_stage_masks_to_largest_nut(stage_masks)
    return merge_stage_masks_to_label_map(stage_masks, chain=model.chain)


def run_cascade_inference_on_frames(
    weights_path: str | Path,
    frames: np.ndarray,
    device: str,
    *,
    progress_callback=None,
    cancel_callback=None,
) -> np.ndarray:
    """Run cascaded segmentation on an RGB volume; returns a label volume."""
    import torch

    rgb = image_volume_to_rgb_frames(frames)
    model, ckpt = load_cascade_checkpoint(weights_path)
    torch_device = _resolve_torch_device(device)
    model.to(torch_device)
    model.eval()
    image_size = int(ckpt.get("image_size", 512))

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
        label_map = _predict_frame_label_map(
            model, frame, image_size=image_size, device=torch_device
        )
        label_stack.append(label_map)
        if progress_callback is not None:
            try:
                progress_callback(t + 1, total_frames)
            except Exception:
                pass

    if not label_stack:
        raise ValueError("Cascade inference produced no frames.")
    if len(label_stack) == 1:
        return label_stack[0]
    return np.stack(label_stack, axis=0).astype(np.uint8)


def format_cascade_train_summary(config: CascadeTrainConfig) -> str:
    return (
        f"Architecture: {config.architecture}, encoder: {config.encoder_name}, "
        f"image size: {config.image_size}px"
    )
