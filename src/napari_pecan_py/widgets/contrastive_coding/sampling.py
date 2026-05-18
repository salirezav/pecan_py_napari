"""Label-aware patch sampling for contrastive training.

Extracts anchor / positive / negative patch triplets from video frames
using the Labels layers produced by the Color Thresholding widget as class masks.

Classes
-------
- Each Labels layer is treated as one foreground class.
- Pixels not covered by any selected mask belong to "background".

Sampling strategy
-----------------
1. Pick a random frame.
2. For each class with enough labelled pixels, sample ``patches_per_class``
   random locations and extract square patches from the Image layer.
3. A *positive* for an anchor is another patch from the **same** class
   (may come from a different random location **or** a different frame).
4. *Negatives* are patches drawn from a **different** class.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np


def _valid_coords(
    mask_2d: np.ndarray,
    patch_size: int,
    max_coords: int = 5000,
) -> np.ndarray:
    """Return (row, col) coords where a full patch fits inside the mask.

    Parameters
    ----------
    mask_2d : 2-D boolean array (H, W)
    patch_size : side length of the square patch
    max_coords : subsample for speed if there are too many

    Returns
    -------
    coords : (M, 2) int array of (row, col)
    """
    half = patch_size // 2
    h, w = mask_2d.shape
    ys, xs = np.where(mask_2d)
    keep = (ys >= half) & (ys < h - half) & (xs >= half) & (xs < w - half)
    coords = np.stack([ys[keep], xs[keep]], axis=1)
    if len(coords) > max_coords:
        idx = np.random.choice(len(coords), max_coords, replace=False)
        coords = coords[idx]
    return coords


def _extract_patches(
    frame: np.ndarray,
    coords: np.ndarray,
    patch_size: int,
    n: int,
) -> np.ndarray:
    """Extract *n* random patches centred at random coords.

    Parameters
    ----------
    frame : (H, W, C) uint8 or float
    coords : (M, 2) candidate centres
    patch_size : side length
    n : number of patches to draw

    Returns
    -------
    patches : (n, C, patch_size, patch_size) float32 in [0, 1]
    """
    half = patch_size // 2
    idx = np.random.choice(len(coords), n, replace=len(coords) < n)
    centres = coords[idx]
    C = frame.shape[2] if frame.ndim == 3 else 1
    patches = np.empty((n, C, patch_size, patch_size), dtype=np.float32)
    for i, (r, c) in enumerate(centres):
        patch = frame[r - half : r - half + patch_size,
                      c - half : c - half + patch_size]
        if patch.ndim == 2:
            patch = patch[..., np.newaxis]
        patches[i] = patch.transpose(2, 0, 1).astype(np.float32) / 255.0
    return patches


def sample_triplets(
    image_data: np.ndarray,
    class_masks: Dict[str, np.ndarray],
    patch_size: int = 8,
    patches_per_class: int = 64,
    num_negatives: int = 4,
    frame_index: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Sample (anchor, positive, negatives) triplets from labelled frames.

    Parameters
    ----------
    image_data : (T, H, W, C) or (H, W, C) – full video or single frame.
    class_masks : ``{class_name: mask_array}`` where *mask_array* has the
        same leading dimensions as *image_data* and non-zero pixels indicate
        class membership.
    patch_size : side length of each square patch.
    patches_per_class : patches to draw per class per call.
    num_negatives : negative patches per anchor.
    frame_index : if *None*, a random frame is chosen.

    Returns
    -------
    anchors   : (N, C, ps, ps) float32
    positives : (N, C, ps, ps) float32
    negatives : (N, num_negatives, C, ps, ps) float32
    labels    : list[str] of length N – class name for each anchor
    """
    is_3d = image_data.ndim == 4
    n_frames = image_data.shape[0] if is_3d else 1

    if frame_index is None:
        frame_index = np.random.randint(n_frames)
    frame = image_data[frame_index] if is_3d else image_data

    per_class_coords: Dict[str, np.ndarray] = {}
    for name, mask in class_masks.items():
        m2d = mask[frame_index] if mask.ndim == 3 else mask
        m2d = m2d > 0
        coords = _valid_coords(m2d, patch_size)
        if len(coords) >= patch_size:
            per_class_coords[name] = coords

    class_names = list(per_class_coords.keys())
    if len(class_names) < 2:
        combined_mask = np.zeros(frame.shape[:2], dtype=bool)
        for m in class_masks.values():
            m2d = m[frame_index] if m.ndim == 3 else m
            combined_mask |= m2d > 0
        bg_mask = ~combined_mask
        bg_coords = _valid_coords(bg_mask, patch_size)
        if len(bg_coords) >= patch_size:
            per_class_coords["background"] = bg_coords
            class_names = list(per_class_coords.keys())

    if len(class_names) < 2:
        raise ValueError(
            "Need at least 2 classes with enough labelled pixels on this frame. "
            "Add more masks or increase labelled area."
        )

    all_anchors, all_positives, all_negatives, all_labels = [], [], [], []

    for cls in class_names:
        coords = per_class_coords[cls]
        n = min(patches_per_class, len(coords))
        anc = _extract_patches(frame, coords, patch_size, n)

        pos_frame_idx = np.random.randint(n_frames) if is_3d else 0
        pos_frame = image_data[pos_frame_idx] if is_3d else image_data
        pos_mask = class_masks.get(cls)
        if pos_mask is not None:
            pm2d = pos_mask[pos_frame_idx] if pos_mask.ndim == 3 else pos_mask
            pos_coords = _valid_coords(pm2d > 0, patch_size)
        else:
            pos_coords = coords
        if len(pos_coords) < 1:
            pos_coords = coords
        pos = _extract_patches(pos_frame, pos_coords, patch_size, n)

        other_classes = [c for c in class_names if c != cls]
        neg_list = []
        for _ in range(num_negatives):
            neg_cls = other_classes[np.random.randint(len(other_classes))]
            neg_coords = per_class_coords[neg_cls]
            neg_fi = np.random.randint(n_frames) if is_3d else 0
            neg_frame = image_data[neg_fi] if is_3d else image_data
            neg_mask = class_masks.get(neg_cls)
            if neg_mask is not None:
                nm2d = neg_mask[neg_fi] if neg_mask.ndim == 3 else neg_mask
                neg_c = _valid_coords(nm2d > 0, patch_size)
                if len(neg_c) < 1:
                    neg_c = neg_coords
            else:
                neg_c = neg_coords
            neg_list.append(_extract_patches(neg_frame, neg_c, patch_size, n))

        negs = np.stack(neg_list, axis=1)

        all_anchors.append(anc)
        all_positives.append(pos)
        all_negatives.append(negs)
        all_labels.extend([cls] * n)

    return (
        np.concatenate(all_anchors),
        np.concatenate(all_positives),
        np.concatenate(all_negatives),
        all_labels,
    )
