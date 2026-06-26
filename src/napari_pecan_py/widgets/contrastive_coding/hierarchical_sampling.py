"""Hierarchical patch sampling: soft positives along the class chain."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

from .hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    hard_negative_classes,
    soft_positive_classes,
)
from .sampling import _extract_patches, _extract_patches_at_centres, _valid_coords, sample_triplets


def _frame_at(image_data: np.ndarray, frame_index: int) -> np.ndarray:
    if image_data.ndim == 4:
        return image_data[frame_index]
    return image_data


def _mask_2d(mask: np.ndarray, frame_index: int) -> np.ndarray:
    if mask.ndim == 3:
        return mask[frame_index] > 0
    return mask > 0


def _build_per_class_coords(
    frame: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    frame_index: int,
    patch_size: int,
) -> Dict[str, np.ndarray]:
    per_class: Dict[str, np.ndarray] = {}
    for name, mask in class_masks.items():
        coords = _valid_coords(_mask_2d(mask, frame_index), patch_size)
        if len(coords) >= patch_size:
            per_class[name] = coords
    return per_class


def _ensure_background(
    frame: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    frame_index: int,
    patch_size: int,
    per_class_coords: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    if len(per_class_coords) >= 2:
        return per_class_coords
    combined = np.zeros(frame.shape[:2], dtype=bool)
    for mask in class_masks.values():
        combined |= _mask_2d(mask, frame_index)
    bg_coords = _valid_coords(~combined, patch_size)
    if len(bg_coords) >= patch_size:
        per_class_coords = dict(per_class_coords)
        per_class_coords["background"] = bg_coords
    return per_class_coords


def sample_hierarchical_triplets(
    image_data: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    patch_size: int = 8,
    patches_per_class: int = 64,
    num_negatives: int = 4,
    frame_index: int | None = None,
    hierarchy_chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
    max_soft_positives: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sample anchors, same-class positives, soft positives, and hard negatives.

    Soft positives are drawn from ancestor/descendant classes on the chain
    (Pecan ⊃ Crack ⊃ Kernel). Hard negatives use background and unrelated labels.

    Returns
    -------
    anchors, positives, soft_positives (N, S, C, ps, ps), negatives (N, K, C, ps, ps), labels
    """
    is_video = image_data.ndim == 4
    n_frames = image_data.shape[0] if is_video else 1
    if frame_index is None:
        frame_index = int(np.random.randint(n_frames))
    frame = _frame_at(image_data, frame_index)

    per_class_coords = _build_per_class_coords(frame, class_masks, frame_index, patch_size)
    per_class_coords = _ensure_background(
        frame, class_masks, frame_index, patch_size, per_class_coords
    )
    class_names = list(per_class_coords.keys())
    if len(class_names) < 2:
        raise ValueError(
            "Need at least 2 classes with enough labelled pixels on this frame."
        )

    foreground = [c for c in class_names if c != "background"]
    if not foreground:
        raise ValueError("Need at least one foreground class besides background.")

    all_anchors: List[np.ndarray] = []
    all_positives: List[np.ndarray] = []
    all_soft: List[np.ndarray] = []
    all_negatives: List[np.ndarray] = []
    all_labels: List[str] = []

    for cls in foreground:
        if cls not in per_class_coords:
            continue
        coords = per_class_coords[cls]
        n = min(patches_per_class, len(coords))
        idx = np.random.choice(len(coords), n, replace=len(coords) < n)
        centres = coords[idx]
        anc = _extract_patches_at_centres(frame, centres, patch_size)

        pos_frame_idx = int(np.random.randint(n_frames)) if is_video else 0
        pos_frame = _frame_at(image_data, pos_frame_idx)
        pos_mask = class_masks.get(cls)
        if pos_mask is not None:
            pos_coords = _valid_coords(_mask_2d(pos_mask, pos_frame_idx), patch_size)
        else:
            pos_coords = coords
        if is_video and n_frames > 1 and pos_frame_idx != frame_index:
            pos = _extract_patches_at_centres(pos_frame, centres, patch_size)
        else:
            if len(pos_coords) < 1:
                pos_coords = coords
            pos_idx = np.random.choice(len(pos_coords), n, replace=len(pos_coords) < n)
            pos = _extract_patches_at_centres(pos_frame, pos_coords[pos_idx], patch_size)

        soft_classes = soft_positive_classes(cls, class_names, hierarchy_chain)
        soft_list: List[np.ndarray] = []
        for soft_cls in soft_classes[:max_soft_positives]:
            soft_coords = per_class_coords.get(soft_cls)
            if soft_coords is None or len(soft_coords) < 1:
                continue
            soft_fi = int(np.random.randint(n_frames)) if is_video else 0
            soft_frame = _frame_at(image_data, soft_fi)
            soft_mask = class_masks.get(soft_cls)
            if soft_mask is not None:
                sc = _valid_coords(_mask_2d(soft_mask, soft_fi), patch_size)
                if len(sc) < 1:
                    sc = soft_coords
            else:
                sc = soft_coords
            soft_list.append(_extract_patches(soft_frame, sc, patch_size, n))

        if soft_list:
            soft_stack = np.stack(soft_list, axis=1)
        else:
            c = anc.shape[1]
            ps = anc.shape[2]
            soft_stack = np.zeros((n, 0, c, ps, ps), dtype=np.float32)

        hard_negs = hard_negative_classes(cls, class_names, hierarchy_chain)
        if not hard_negs:
            hard_negs = ["background"] if "background" in class_names else [
                c for c in class_names if c != cls
            ]
        neg_list: List[np.ndarray] = []
        for _ in range(num_negatives):
            neg_cls = hard_negs[int(np.random.randint(len(hard_negs)))]
            neg_coords = per_class_coords[neg_cls]
            neg_fi = int(np.random.randint(n_frames)) if is_video else 0
            neg_frame = _frame_at(image_data, neg_fi)
            neg_mask = class_masks.get(neg_cls)
            if neg_mask is not None:
                nc = _valid_coords(_mask_2d(neg_mask, neg_fi), patch_size)
                if len(nc) < 1:
                    nc = neg_coords
            else:
                nc = neg_coords
            neg_list.append(_extract_patches(neg_frame, nc, patch_size, n))
        negs = np.stack(neg_list, axis=1)

        all_anchors.append(anc)
        all_positives.append(pos)
        all_soft.append(soft_stack)
        all_negatives.append(negs)
        all_labels.extend([cls] * n)

    max_soft = max((s.shape[1] for s in all_soft), default=0)
    padded_soft: List[np.ndarray] = []
    for soft in all_soft:
        if soft.shape[1] == max_soft:
            padded_soft.append(soft)
            continue
        n, _, c, ps1, ps2 = soft.shape[0], max_soft, soft.shape[2], soft.shape[3], soft.shape[4]
        pad = np.zeros((n, max_soft, c, ps1, ps2), dtype=np.float32)
        if soft.shape[1] > 0:
            pad[:, : soft.shape[1]] = soft
        padded_soft.append(pad)

    return (
        np.concatenate(all_anchors),
        np.concatenate(all_positives),
        np.concatenate(padded_soft),
        np.concatenate(all_negatives),
        all_labels,
    )


def sample_training_batch(
    image_data: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    *,
    training_mode: str,
    patch_size: int,
    patches_per_class: int,
    num_negatives: int,
    frame_index: int | None = None,
    hierarchy_chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> Tuple:
    """Dispatch to simple or hierarchical triplet sampling."""
    from .hierarchy import TRAINING_MODE_HIERARCHICAL

    if training_mode == TRAINING_MODE_HIERARCHICAL:
        return sample_hierarchical_triplets(
            image_data,
            class_masks,
            patch_size=patch_size,
            patches_per_class=patches_per_class,
            num_negatives=num_negatives,
            frame_index=frame_index,
            hierarchy_chain=hierarchy_chain,
        )
    anc, pos, neg, labels = sample_triplets(
        image_data,
        class_masks,
        patch_size=patch_size,
        patches_per_class=patches_per_class,
        num_negatives=num_negatives,
        frame_index=frame_index,
    )
    n = anc.shape[0]
    c, ps = anc.shape[1], anc.shape[2]
    empty_soft = np.zeros((n, 0, c, ps, ps), dtype=np.float32)
    return anc, pos, empty_soft, neg, labels
