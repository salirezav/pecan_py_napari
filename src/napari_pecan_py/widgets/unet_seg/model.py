"""Single U-Net / U-Net++ with one output channel per class (no cascade)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from napari_pecan_py.widgets.cascade_seg.hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    LABEL_ID_BY_CLASS,
    STAGE_ABSENCE_LOSS_WEIGHT,
    STAGE_INFERENCE_THRESHOLD,
    STAGE_LOSS_WEIGHT,
    STAGE_POS_WEIGHT,
    ordered_chain_classes,
)
from napari_pecan_py.widgets.cascade_seg.model import (
    ARCH_UNET,
    ARCH_UNETPP,
    _augment_sample,
    _collate_frame_samples,
    _collect_video_samples,
    _device_status_message,
    _dice_bce_loss,
    _foreground_iou,
    _format_duration,
    _loss_region_for_stage,
    _maybe_enable_cuda_fast_math,
    _prepare_batch_tensors,
    _resolve_torch_device,
    _should_apply_absence_loss,
    _split_samples,
    _stage_inference_threshold,
    _stage_loss_weight,
    _stage_pos_weight,
    _to_uint8_rgb,
    count_partial_inner_label_frames,
    count_pecan_only_frames,
    filter_samples_require_all_classes,
    merge_stage_masks_to_label_map,
    summarize_cascade_frame_usage,
)
from napari_pecan_py.widgets.cascade_seg.model import _build_segmentation_model
from napari_pecan_py.widgets.yolo_seg.model import image_volume_to_rgb_frames

BACKEND_UNET = "unet"


@dataclass
class UnetTrainConfig:
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
    init_weights_path: str | None = None
    train_absent_inner_classes: bool = True


class FlatSegmenter:
    """One segmentation model; each output channel is an independent binary class."""

    def __init__(
        self,
        class_names: Sequence[str],
        *,
        encoder_name: str = "mobilenet_v2",
        architecture: str = ARCH_UNETPP,
    ) -> None:
        self.class_names = list(class_names)
        if not self.class_names:
            raise ValueError("At least one class is required.")
        self.encoder_name = encoder_name
        self.architecture = architecture
        self.net = _build_segmentation_model(
            architecture,
            encoder_name,
            in_channels=3,
            classes=len(self.class_names),
        )

    def forward_logits(self, image):
        return self.net(image)

    def predict_probs(self, image):
        import torch

        self.net.eval()
        with torch.no_grad():
            return torch.sigmoid(self.forward_logits(image))

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, state_dict) -> None:
        self.net.load_state_dict(state_dict)

    def to(self, device):
        self.net.to(device)
        return self

    def train(self, mode: bool = True):
        self.net.train(mode)
        return self

    def eval(self):
        return self.train(False)


def format_unet_train_summary(config: UnetTrainConfig) -> str:
    return (
        f"Flat U-Net: {config.architecture}, encoder: {config.encoder_name}, "
        f"image size: {config.image_size}px"
    )


def guess_unet_save_suffix(
    weights_path: str | Path,
    *,
    fallback_index: int = 1,
) -> tuple[str, bool]:
    from napari_pecan_py.widgets.yolo_seg.model import WEIGHTS_CLASSES_RE

    stem = Path(weights_path).stem
    match = WEIGHTS_CLASSES_RE.search(stem)
    if match:
        classes = match.group(1).strip()
        if classes:
            return f" - {classes}", True
    return f" - U-Net Seg [{fallback_index}]", False


def save_unet_checkpoint(
    path: str | Path,
    model: FlatSegmenter,
    *,
    image_size: int,
    label_id_by_class: Dict[str, int] | None = None,
) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "backend": BACKEND_UNET,
            "model_state": model.state_dict(),
            "architecture": model.architecture,
            "encoder_name": model.encoder_name,
            "class_names": list(model.class_names),
            "image_size": int(image_size),
            "label_id_by_class": dict(label_id_by_class or LABEL_ID_BY_CLASS),
        },
        path,
    )


def load_unet_checkpoint(weights_path: str | Path) -> tuple[FlatSegmenter, dict]:
    import torch

    ckpt = torch.load(Path(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or ckpt.get("backend") != BACKEND_UNET:
        raise ValueError(f"Not a flat U-Net checkpoint: {weights_path}")

    class_names = ckpt.get("class_names") or ckpt.get("trained_classes") or []
    model = FlatSegmenter(
        class_names,
        encoder_name=str(ckpt.get("encoder_name", "mobilenet_v2")),
        architecture=str(ckpt.get("architecture", ARCH_UNETPP)),
    )
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


def _compute_flat_losses(
    logits,
    target_tensors,
    class_names,
    device,
    *,
    train_absent_inner_classes: bool = True,
):
    import torch

    stage_losses: Dict[str, float] = {}
    total = torch.tensor(0.0, device=device)
    region_source = {cls: target_tensors[cls].to(device) for cls in target_tensors}
    for index, class_name in enumerate(class_names):
        if class_name not in target_tensors:
            continue
        target = target_tensors[class_name].to(device)
        target_sum = float(target.sum())
        is_inner = class_name in ("Crack", "Kernel")
        channel_logits = logits[:, index : index + 1]
        if is_inner and target_sum <= 0:
            if not _should_apply_absence_loss(
                class_name,
                region_source,
                train_absent_inner_classes=train_absent_inner_classes,
            ):
                continue
            region = _loss_region_for_stage(class_name, region_source)
            if region is None and "Pecan" in region_source:
                region = (region_source["Pecan"] > 0.5).float()
            stage_loss = _dice_bce_loss(
                channel_logits,
                torch.zeros_like(target),
                region,
                pos_weight=1.0,
            )
            absence_w = float(STAGE_ABSENCE_LOSS_WEIGHT.get(class_name, 0.35))
            weighted = stage_loss * _stage_loss_weight(class_name) * absence_w
        else:
            stage_loss = _dice_bce_loss(
                channel_logits,
                target,
                None,
                pos_weight=_stage_pos_weight(class_name),
            )
            weighted = stage_loss * _stage_loss_weight(class_name)
        if torch.isfinite(stage_loss):
            stage_losses[class_name] = float(stage_loss.detach().cpu())
            total = total + weighted
    return total, stage_losses


def train_unet_segmenter(
    video_entries: Sequence[Tuple[str, Dict[str, str]]],
    output_dir: str | Path,
    device: str,
    config: UnetTrainConfig,
    *,
    selected_classes: Sequence[str],
    log_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> str:
    """Train a flat multi-class U-Net; returns path to best checkpoint."""
    import torch
    from torch.utils.data import DataLoader, Dataset

    def log(msg: str) -> None:
        if log_callback is not None:
            log_callback(msg)

    class_names = ordered_chain_classes(selected_classes)
    if not class_names:
        raise ValueError("No training classes selected.")

    video_paths, frame_counts, raw_samples = _collect_video_samples(video_entries, class_names)
    if not raw_samples:
        raise ValueError("No labeled frames found for U-Net training.")

    if config.require_all_classes_in_frame:
        before = len(raw_samples)
        raw_samples = filter_samples_require_all_classes(raw_samples, class_names)
        log(
            f"All-class filter: kept {len(raw_samples)}/{before} frame(s) "
            f"with {', '.join(class_names)}."
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
    log(
        f"Loading {config.encoder_name} and building flat U-Net "
        f"({len(class_names)} classes)…"
    )
    log("First run may download ImageNet encoder weights.")
    if config.init_weights_path:
        log(f"Fine-tuning from checkpoint: {config.init_weights_path}")
        model, ckpt = load_unet_checkpoint(config.init_weights_path)
        ckpt_classes = list(ckpt.get("class_names") or [])
        if ckpt_classes != class_names:
            log(
                f"WARNING: checkpoint classes {ckpt_classes} differ from selected "
                f"{class_names}."
            )
        ckpt_size = int(ckpt.get("image_size") or 0)
        if ckpt_size and ckpt_size != config.image_size:
            log(
                f"WARNING: checkpoint image size {ckpt_size} ≠ training "
                f"{config.image_size}px."
            )
        model = model.to(torch_device)
    else:
        model = FlatSegmenter(
            class_names,
            encoder_name=config.encoder_name,
            architecture=config.architecture,
        ).to(torch_device)
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=config.learning_rate)
    log("Model ready. Starting epoch 1…")
    if config.init_weights_path:
        log(f"Fine-tune learning rate: {config.learning_rate:g}")
    partial = count_partial_inner_label_frames(train_samples)
    if partial["crack_no_kernel"] > 0:
        log(
            f"Partial labels: {partial['crack_no_kernel']} train frame(s) crack without "
            f"kernel mask — kernel not trained as background on those frames."
        )
    pecan_only = count_pecan_only_frames(train_samples)
    if pecan_only > 0 and config.train_absent_inner_classes:
        log(
            f"Intact-shell train frames: {pecan_only}/{len(train_samples)} "
            "(absence loss for crack/kernel)."
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "best.pt"
    best_val = float("inf")
    use_amp = torch_device.type == "cuda"

    for epoch in range(1, config.epochs + 1):
        if cancel_callback is not None and cancel_callback():
            break
        model.train()
        train_loss = 0.0
        train_batches = 0
        epoch_stage_totals: Dict[str, float] = {name: 0.0 for name in class_names}
        epoch_start = time.perf_counter()
        log(f"Epoch {epoch}/{config.epochs}: training ({train_batches_total} batch(es))…")

        for batch in train_loader:
            if cancel_callback is not None and cancel_callback():
                break
            batch_start = time.perf_counter()
            image_t, target_tensors, _ = _prepare_batch_tensors(
                batch, torch_device, config.image_size
            )
            with torch.autocast(device_type=torch_device.type, enabled=use_amp):
                logits = model.forward_logits(image_t)
                loss, stage_losses = _compute_flat_losses(
                    logits,
                    target_tensors,
                    class_names,
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
            eta = (epoch_elapsed / train_batches) * (train_batches_total - train_batches)
            stage_str = "  ".join(f"{k}={v:.3f}" for k, v in stage_losses.items())
            if train_batches == 1 or train_batches == train_batches_total or train_batches % 10 == 0:
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
            for name in class_names
        )
        avg_val = None
        val_stage_totals: Dict[str, float] = {name: 0.0 for name in class_names}
        val_iou_totals: Dict[str, float] = {name: 0.0 for name in class_names}
        val_iou_counts: Dict[str, int] = {name: 0 for name in class_names}
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
                        logits = model.forward_logits(image_t)
                        loss, stage_losses = _compute_flat_losses(
                            logits,
                            target_tensors,
                            class_names,
                            torch_device,
                            train_absent_inner_classes=config.train_absent_inner_classes,
                        )
                    val_loss += float(loss.detach().cpu())
                    for name, value in stage_losses.items():
                        val_stage_totals[name] = val_stage_totals.get(name, 0.0) + value
                    for index, class_name in enumerate(class_names):
                        if class_name not in target_tensors:
                            continue
                        channel_logits = logits[:, index : index + 1]
                        target = target_tensors[class_name].to(torch_device)
                        iou = _foreground_iou(
                            channel_logits,
                            target,
                            None,
                            threshold=_stage_inference_threshold(class_name),
                        )
                        if iou == iou:
                            val_iou_totals[class_name] += iou
                            val_iou_counts[class_name] += 1
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
                save_unet_checkpoint(best_path, model, image_size=config.image_size)

        if progress_callback and val_loader is None:
            progress_callback(epoch * progress_units_per_epoch, progress_units_total)

        if avg_val is not None:
            val_stage_str = "  ".join(
                f"{name}={val_stage_totals[name] / max(val_batches, 1):.4f}"
                for name in class_names
            )
            val_iou_str = "  ".join(
                f"{name}={val_iou_totals[name] / max(val_iou_counts[name], 1):.3f}"
                if val_iou_counts[name] > 0
                else f"{name}=n/a"
                for name in class_names
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
                save_unet_checkpoint(best_path, model, image_size=config.image_size)

    if not best_path.is_file():
        save_unet_checkpoint(best_path, model, image_size=config.image_size)

    classes_bracket = ", ".join(class_names)
    named = output_dir / f"unet - [{classes_bracket}].pt"
    if best_path.is_file() and not named.is_file():
        import torch

        torch.save(torch.load(best_path, weights_only=False), named)

    return str(named if named.is_file() else best_path)


def _predict_frame_label_map(
    model: FlatSegmenter,
    frame: np.ndarray,
    *,
    image_size: int,
    device,
) -> np.ndarray:
    import cv2
    import torch

    from napari_pecan_py.widgets.cascade_seg.model import (
        _resize_rgb,
        filter_stage_masks_to_largest_nut,
        merge_stage_masks_to_label_map,
    )

    h, w = frame.shape[:2]
    resized = _resize_rgb(frame, image_size)
    image_t = (
        torch.from_numpy(resized).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    ).to(device)

    probs = model.predict_probs(image_t).squeeze(0).detach().cpu().numpy()
    stage_masks: Dict[str, np.ndarray] = {}
    for index, class_name in enumerate(model.class_names):
        thresh = _stage_inference_threshold(class_name)
        mask = (probs[index] > thresh).astype(np.uint8)
        if mask.shape != (h, w):
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        stage_masks[class_name] = mask

    stage_masks = filter_stage_masks_to_largest_nut(stage_masks)
    return merge_stage_masks_to_label_map(stage_masks)


def run_unet_inference_on_frames(
    weights_path: str | Path,
    frames: np.ndarray,
    device: str,
    *,
    progress_callback=None,
    cancel_callback=None,
) -> np.ndarray:
    """Run flat U-Net segmentation on an RGB volume; returns a label volume."""
    rgb = image_volume_to_rgb_frames(frames)
    model, ckpt = load_unet_checkpoint(weights_path)
    torch_device = _resolve_torch_device(device)
    model.to(torch_device)
    model.eval()
    image_size = int(ckpt.get("image_size", 384))

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
        raise ValueError("U-Net inference produced no frames.")
    if len(label_stack) == 1:
        return label_stack[0]
    return np.stack(label_stack, axis=0).astype(np.uint8)


def summarize_unet_frame_usage(
    video_entries: Sequence[Tuple[str, Dict[str, str]]],
    selected_classes: Sequence[str],
    *,
    val_fraction: float = 0.2,
    split_by: str = "video",
    require_all_classes_in_frame: bool = False,
) -> str:
    return summarize_cascade_frame_usage(
        video_entries,
        selected_classes,
        val_fraction=val_fraction,
        split_by=split_by,
        require_all_classes_in_frame=require_all_classes_in_frame,
    )
