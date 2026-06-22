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


def _scalar_mask_from_plane(plane: np.ndarray) -> np.ndarray:
    """Convert one 2D raster plane to a uint8 {0,1} foreground mask."""
    p = np.asarray(plane)
    if p.ndim != 2:
        raise ValueError(f"Expected 2D plane; got shape={p.shape}")
    if np.issubdtype(p.dtype, np.floating):
        if float(np.nanmax(p)) <= 1.0:
            return (p > 0.5).astype(np.uint8)
        return (p > 0).astype(np.uint8)
    return (p > 0).astype(np.uint8)


def mask_volume_from_array(arr: np.ndarray) -> np.ndarray:
    """Normalize Labels or Image data to (H,W) or (T,H,W) for binary ops."""
    a = np.asarray(arr)
    if a.ndim == 2:
        if np.issubdtype(a.dtype, np.floating) or a.max() <= 1:
            return _scalar_mask_from_plane(a)
        return a
    if a.ndim == 3:
        if a.shape[-1] in (1, 2, 3, 4) and a.shape[-1] < a.shape[-2]:
            if a.shape[-1] == 1:
                return _scalar_mask_from_plane(a[..., 0])
            return _scalar_mask_from_plane(np.max(a[..., :3], axis=-1))
        return _as_mask_volume(a)
    if a.ndim == 4:
        if a.shape[-1] not in (1, 2, 3, 4):
            raise ValueError(f"Unsupported mask array shape: {a.shape}")
        out = np.zeros(a.shape[:3], dtype=np.uint8)
        for t in range(a.shape[0]):
            sl = a[t]
            if sl.shape[-1] == 1:
                plane = _scalar_mask_from_plane(sl[..., 0])
            else:
                plane = _scalar_mask_from_plane(np.max(sl[..., :3], axis=-1))
            out[t] = plane
        return out
    raise ValueError(f"Unsupported mask array shape: {a.shape}")


def _foreground_fill_value(template: np.ndarray):
    pos = np.asarray(template)[np.asarray(template) > 0]
    if pos.size == 0:
        t = np.asarray(template)
        if np.issubdtype(t.dtype, np.floating):
            return 1.0 if float(np.nanmax(t)) <= 1.0 else float(np.nanmax(t))
        return np.uint8(255)
    return pos.max()


def expand_mask_to_layer_shape(result: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Map a (H,W) or (T,H,W) mask result back onto *template* shape/dtype."""
    res = np.asarray(result)
    t = np.asarray(template)
    if res.shape == t.shape:
        return res.astype(t.dtype, copy=False)
    mask = res > 0
    fill = _foreground_fill_value(t)
    if t.ndim == 2:
        out = np.zeros_like(t)
        out[mask] = fill
        return out
    if t.ndim == 3 and t.shape[-1] in (1, 2, 3, 4) and t.shape[-1] < t.shape[-2]:
        out = np.zeros_like(t)
        out[mask, ...] = fill
        return out
    if t.ndim == 3:
        out = np.zeros_like(t)
        out[mask] = fill
        return out
    if t.ndim == 4:
        out = np.zeros_like(t)
        out[mask, ...] = fill
        return out
    raise ValueError(f"Cannot expand mask shape {res.shape} to template {t.shape}")


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


def labels_from_bool_mask(binary: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Convert a boolean mask to a Labels array using dtype/label value from *template*."""
    return _fill_like(binary, template)


def apply_binary_operation(a: np.ndarray, b: np.ndarray, op: str, template: np.ndarray) -> np.ndarray:
    """Apply op on A/B as binary masks and return typed output like template."""
    aa = mask_volume_from_array(a)
    bb = mask_volume_from_array(b)
    if aa.shape != bb.shape:
        raise ValueError(f"A and B must have same shape; got {aa.shape} vs {bb.shape}")
    tmpl = mask_volume_from_array(template)
    if tmpl.shape != aa.shape:
        tmpl = aa
    res_bool = _bool_result(aa, bb, op)
    return _fill_like(res_bool, tmpl)


def _as_volume_2d_or_3d(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim not in (2, 3):
        raise ValueError(f"Expected 2D or 3D (T,H,W) array; got shape={a.shape}")
    return a


def binarize_edge_raster(edges: np.ndarray, threshold: int = 1) -> np.ndarray:
    """Threshold a uint8/float edge image to a boolean boundary mask."""
    return np.asarray(edges) >= int(threshold)


def close_edge_raster_gaps(edges_bool: np.ndarray, kernel_size: int) -> np.ndarray:
    """Morphologically close small breaks in a boolean edge raster."""
    if kernel_size < 3:
        return np.asarray(edges_bool, dtype=bool)
    k = int(kernel_size) | 1
    u8 = (np.asarray(edges_bool, dtype=bool).astype(np.uint8)) * 255
    kernel = np.ones((k, k), np.uint8)
    closed = cv2.morphologyEx(u8, cv2.MORPH_CLOSE, kernel)
    return closed > 0


def _component_angle_deg(mask: np.ndarray) -> float | None:
    ys, xs = np.where(mask)
    if ys.size < 3:
        return None
    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    vx, vy, *_ = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01)
    ang = np.degrees(np.arctan2(float(vy), float(vx)))
    return float((ang + 180.0) % 180.0)


def _angle_distance_deg(a: float, b: float) -> float:
    d = abs(float(a) - float(b))
    return float(min(d, 180.0 - d))


def _fill_holes_bool(mask: np.ndarray) -> np.ndarray:
    b = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    h, w = b.shape
    pad = np.zeros((h + 2, w + 2), np.uint8)
    pad[1 : h + 1, 1 : w + 1] = b
    flood_mask = np.zeros((h + 4, w + 4), np.uint8)
    cv2.floodFill(pad, flood_mask, (0, 0), 255)
    outside = pad[1 : h + 1, 1 : w + 1] > 0
    inside = (~outside) | (b > 0)
    return inside


def _distance_overlap_metrics(
    mask_from: np.ndarray,
    mask_to: np.ndarray,
    dist_from: np.ndarray,
    *,
    dmin: float,
    dmax: float,
) -> tuple[float, float] | None:
    """Fraction of `mask_to` pixels within [dmin, dmax] of `mask_from`, and their mean distance."""
    pts = np.where(mask_to)
    if pts[0].size == 0:
        return None
    dists = dist_from[pts].astype(np.float64)
    in_range = (dists >= dmin) & (dists <= dmax)
    overlap_frac = float(np.mean(in_range))
    if overlap_frac <= 0.0:
        return overlap_frac, float("inf")
    mean_d = float(np.mean(dists[in_range]))
    return overlap_frac, mean_d


def _band_mask_between_components(
    mask_i: np.ndarray,
    mask_j: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    pair = (mask_i | mask_j).astype(np.uint8) * 255
    closed = cv2.morphologyEx(pair, cv2.MORPH_CLOSE, kernel)
    filled = _fill_holes_bool(closed > 0)
    nf, lf = cv2.connectedComponents((filled.astype(np.uint8)) * 255, connectivity=8)
    keep = np.zeros_like(filled, dtype=bool)
    for fl in range(1, nf):
        region = lf == fl
        if np.any(region & mask_i) and np.any(region & mask_j):
            keep |= region
    return keep


def detect_parallel_edge_bands_slice(
    edges_slice: np.ndarray,
    *,
    edge_threshold: int = 1,
    pre_close_size: int = 0,
    min_distance_px: int = 1,
    max_distance_px: int = 12,
    angle_tolerance_deg: int = 25,
    min_component_px: int = 20,
    limit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Fill strips between nearby approximately parallel edge components.

    This targets crack-shell thickness where two close rails appear in edge maps.
    """
    e = binarize_edge_raster(edges_slice, threshold=edge_threshold)
    if pre_close_size >= 3:
        e = close_edge_raster_gaps(e, pre_close_size)
    if limit_mask is not None:
        e &= np.asarray(limit_mask, dtype=bool)

    e_u8 = (e.astype(np.uint8)) * 255
    n_labels, labels = cv2.connectedComponents(e_u8, connectivity=8)
    if n_labels <= 2:
        return np.zeros_like(e, dtype=bool)

    components: list[dict] = []
    for lbl in range(1, n_labels):
        comp = labels == lbl
        area = int(np.count_nonzero(comp))
        if area < max(3, int(min_component_px)):
            continue
        ang = _component_angle_deg(comp)
        if ang is None:
            continue
        components.append({"id": lbl, "mask": comp, "angle": ang, "area": area})
    if len(components) < 2:
        return np.zeros_like(e, dtype=bool)

    dmin = max(1, int(min_distance_px))
    dmax = max(dmin + 1, int(max_distance_px))
    close_k = max(3, (dmax * 2 + 1) | 1)
    kernel = np.ones((close_k, close_k), np.uint8)
    min_overlap_frac = 0.35

    dist_maps: list[np.ndarray] = []
    for comp in components:
        inv = (~comp["mask"]).astype(np.uint8)
        dist_maps.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))

    candidates: list[tuple[float, float, float, int, int]] = []
    for i in range(len(components)):
        ci = components[i]
        for j in range(i + 1, len(components)):
            cj = components[j]
            angle_diff = _angle_distance_deg(ci["angle"], cj["angle"])
            if angle_diff > float(angle_tolerance_deg):
                continue

            m_ij = _distance_overlap_metrics(
                ci["mask"], cj["mask"], dist_maps[i], dmin=float(dmin), dmax=float(dmax)
            )
            m_ji = _distance_overlap_metrics(
                cj["mask"], ci["mask"], dist_maps[j], dmin=float(dmin), dmax=float(dmax)
            )
            if m_ij is None or m_ji is None:
                continue
            frac_ij, mean_ij = m_ij
            frac_ji, mean_ji = m_ji
            overlap = min(frac_ij, frac_ji)
            if overlap < min_overlap_frac:
                continue

            mean_d = 0.5 * (mean_ij + mean_ji)
            # Prefer pairs that run alongside each other, then wider gaps (true outer rails).
            sort_key = (-overlap, -mean_d, angle_diff)
            candidates.append((sort_key[0], sort_key[1], sort_key[2], i, j))

    candidates.sort()
    used: set[int] = set()
    selected_pairs: list[tuple[int, int]] = []
    for _, _, _, i, j in candidates:
        if i in used or j in used:
            continue
        used.add(i)
        used.add(j)
        selected_pairs.append((i, j))

    out = np.zeros_like(e, dtype=bool)
    for i, j in selected_pairs:
        out |= _band_mask_between_components(components[i]["mask"], components[j]["mask"], kernel)

    out &= ~e  # emphasize thickness area between rails, not the rails themselves
    if limit_mask is not None:
        out &= np.asarray(limit_mask, dtype=bool)
    return out


def detect_parallel_edge_bands_volume(
    edges: np.ndarray,
    *,
    edge_threshold: int = 1,
    pre_close_size: int = 0,
    min_distance_px: int = 1,
    max_distance_px: int = 12,
    angle_tolerance_deg: int = 25,
    min_component_px: int = 20,
    limit_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Volume wrapper for `detect_parallel_edge_bands_slice`."""
    e = _as_volume_2d_or_3d(edges)
    lim = None if limit_mask is None else _as_mask_volume(limit_mask) > 0
    if lim is not None and lim.shape != e.shape:
        raise ValueError(f"Limit mask shape must match edges: {lim.shape} vs {e.shape}")
    if e.ndim == 2:
        return detect_parallel_edge_bands_slice(
            e,
            edge_threshold=edge_threshold,
            pre_close_size=pre_close_size,
            min_distance_px=min_distance_px,
            max_distance_px=max_distance_px,
            angle_tolerance_deg=angle_tolerance_deg,
            min_component_px=min_component_px,
            limit_mask=lim,
        )
    out = np.zeros(e.shape, dtype=bool)
    for t in range(e.shape[0]):
        out[t] = detect_parallel_edge_bands_slice(
            e[t],
            edge_threshold=edge_threshold,
            pre_close_size=pre_close_size,
            min_distance_px=min_distance_px,
            max_distance_px=max_distance_px,
            angle_tolerance_deg=angle_tolerance_deg,
            min_component_px=min_component_px,
            limit_mask=None if lim is None else lim[t],
        )
    return out
