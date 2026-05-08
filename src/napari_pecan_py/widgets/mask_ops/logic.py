"""Mask clipping and binary operations."""

from __future__ import annotations

import math

import cv2
import numpy as np


def _as_mask_volume(arr: np.ndarray) -> np.ndarray:
    """Accept (H,W) or (T,H,W) and return same."""
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(f"Mask must be 2D or 3D (T,H,W); got shape={a.shape}")
    return a


def _ellipse_mask_from_vertices(vertices: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Rasterize one ellipse from 4 napari ellipse-box vertices in (y,x) order."""
    v = np.asarray(vertices, dtype=np.float64)
    if v.shape != (4, 2):
        raise ValueError(f"Expected (4,2) ellipse vertices; got {v.shape}")
    c = np.mean(v, axis=0)  # (y, x)
    a_vec = 0.5 * (v[1] - v[0])  # one semi-axis vector in (y, x)
    b_vec = 0.5 * (v[3] - v[0])  # the other semi-axis
    a = float(np.linalg.norm(a_vec))
    b = float(np.linalg.norm(b_vec))
    if a < 0.5 or b < 0.5:
        return np.zeros(hw, dtype=bool)

    # Convert to OpenCV params in (x, y) image coordinates.
    center_xy = (float(c[1]), float(c[0]))
    angle_deg = float(np.degrees(math.atan2(a_vec[0], a_vec[1])))
    axes = (max(1, int(round(a))), max(1, int(round(b))))

    out = np.zeros(hw, dtype=np.uint8)
    cv2.ellipse(
        out,
        (int(round(center_xy[0])), int(round(center_xy[1]))),
        axes,
        angle_deg,
        0,
        360,
        255,
        thickness=cv2.FILLED,
    )
    return out > 0


def build_ellipse_masks_for_volume(
    shapes_layer,
    out_shape: tuple[int, ...],
) -> np.ndarray:
    """Return bool ellipse mask with shape (H,W) or (T,H,W) matching out_shape."""
    if len(out_shape) == 2:
        h, w = out_shape
        out = np.zeros((h, w), dtype=bool)
    elif len(out_shape) == 3:
        t, h, w = out_shape
        out = np.zeros((t, h, w), dtype=bool)
    else:
        raise ValueError(f"Unsupported mask shape: {out_shape}")

    data = list(shapes_layer.data)
    stypes = list(shapes_layer.shape_type)
    for verts_raw, stype in zip(data, stypes):
        if str(stype).lower() != "ellipse":
            continue
        verts = np.asarray(verts_raw, dtype=np.float64)
        if verts.ndim != 2:
            continue

        if verts.shape[1] == 2:
            em = _ellipse_mask_from_vertices(verts, (h, w))
            if out.ndim == 2:
                out |= em
            else:
                out |= em[None, ...]
        elif verts.shape[1] == 3:
            t_idx = int(round(float(np.mean(verts[:, 0]))))
            if out.ndim != 3 or not (0 <= t_idx < out.shape[0]):
                continue
            em = _ellipse_mask_from_vertices(verts[:, 1:3], (h, w))
            out[t_idx] |= em
    return out


def clip_mask_outside_ellipse(mask_data: np.ndarray, ellipse_mask: np.ndarray) -> np.ndarray:
    """Keep original mask values only where ellipse_mask is True."""
    m = _as_mask_volume(mask_data)
    e = np.asarray(ellipse_mask, dtype=bool)
    if m.shape != e.shape:
        raise ValueError(f"Shape mismatch mask={m.shape} ellipse={e.shape}")
    out = np.array(m, copy=True)
    out[~e] = 0
    return out


def _bool_result(a: np.ndarray, b: np.ndarray, op: str) -> np.ndarray:
    op = op.lower()
    aa = np.asarray(a) > 0
    bb = np.asarray(b) > 0
    if op == "and":
        return aa & bb
    if op == "or":
        return aa | bb
    if op == "xor":
        return aa ^ bb
    if op == "not":
        return ~aa
    if op == "a-b":
        return aa & (~bb)
    if op == "b-a":
        return bb & (~aa)
    if op == "a-if-b":
        return _components_from_a_intersecting_b(aa, bb)
    raise ValueError(f"Unknown operation: {op}")


def _components_from_a_intersecting_b(aa: np.ndarray, bb: np.ndarray) -> np.ndarray:
    """
    Keep connected components from A only when they intersect B.

    For 3D masks (T,H,W), components are computed independently per time slice.
    """
    if aa.shape != bb.shape:
        raise ValueError(f"A and B must have same shape; got {aa.shape} vs {bb.shape}")

    def _slice_components(a2: np.ndarray, b2: np.ndarray) -> np.ndarray:
        a_u8 = (a2.astype(np.uint8)) * 255
        num_labels, labels = cv2.connectedComponents(a_u8, connectivity=8)
        if num_labels <= 1:
            return np.zeros_like(a2, dtype=bool)
        keep = np.zeros(num_labels, dtype=bool)
        overlap_labels = labels[b2]
        if overlap_labels.size > 0:
            keep[np.unique(overlap_labels)] = True
        keep[0] = False  # Never keep background.
        return keep[labels]

    if aa.ndim == 2:
        return _slice_components(aa, bb)
    if aa.ndim == 3:
        out = np.zeros_like(aa, dtype=bool)
        for t in range(aa.shape[0]):
            out[t] = _slice_components(aa[t], bb[t])
        return out
    raise ValueError(f"Mask must be 2D or 3D (T,H,W); got shape={aa.shape}")


def _fill_like(binary: np.ndarray, template: np.ndarray) -> np.ndarray:
    t = np.asarray(template)
    out = np.zeros_like(t)
    pos = t[t > 0]
    true_value = int(pos.max()) if pos.size else 1
    out[binary] = true_value
    return out


def apply_binary_operation(a: np.ndarray, b: np.ndarray, op: str, template: np.ndarray) -> np.ndarray:
    """Apply op on A/B as binary masks and return typed output like template."""
    aa = _as_mask_volume(a)
    bb = _as_mask_volume(b)
    if aa.shape != bb.shape:
        raise ValueError(f"A and B must have same shape; got {aa.shape} vs {bb.shape}")
    res_bool = _bool_result(aa, bb, op)
    return _fill_like(res_bool, template)
