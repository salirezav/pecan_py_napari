"""Pipeline execution logic."""

from __future__ import annotations

import numpy as np
from napari.layers import Image, Labels, Shapes

from ..color_adjustments.logic import apply_adjustments_to_video
from ..color_tuner.logic import apply_thresholds
from ..mask_ops.logic import (
    apply_binary_operation,
    build_ellipse_masks_for_volume,
    clip_mask_outside_ellipse,
)
from ..pecan_ellipse.logic import apply_ellipse_pipeline, mask_volume_needs_time_coord


def _layer_by_name(viewer, name: str):
    try:
        return viewer.layers[str(name)]
    except Exception:
        return None


def _image_data(layer: Image) -> np.ndarray:
    arr = np.asarray(layer.data)
    if arr.ndim not in (3, 4):
        raise ValueError(f"Expected image shape (H,W,C) or (T,H,W,C), got {arr.shape}")
    return arr


def _frame_rgb(arr: np.ndarray, t: int) -> np.ndarray:
    if arr.ndim == 3:
        frame = arr
    else:
        frame = arr[int(np.clip(t, 0, arr.shape[0] - 1))]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"Expected RGB frame, got {frame.shape}")
    return np.asarray(frame[..., :3], dtype=np.uint8)


def _apply_color_tuner_step(viewer, params: dict) -> str:
    src_name = str(params.get("source_layer", ""))
    target = str(params.get("target", "pecan"))
    color_space = str(params.get("color_space", "rgb"))
    lower = np.asarray(params.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.asarray(params.get("upper", [255, 255, 255]), dtype=np.uint8)
    out_name = str(params.get("output_mask_layer", f"{src_name} - {target.title()}"))

    src = _layer_by_name(viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")
    arr = _image_data(src)
    thresholds = {color_space: {target: {"lower": lower, "upper": upper}}}

    if arr.ndim == 3:
        mask = (apply_thresholds(_frame_rgb(arr, 0), color_space, target, thresholds) > 0).astype(np.uint8)
    else:
        masks = []
        for t in range(arr.shape[0]):
            masks.append((apply_thresholds(_frame_rgb(arr, t), color_space, target, thresholds) > 0).astype(np.uint8))
        mask = np.stack(masks, axis=0).astype(np.uint8)

    existing = _layer_by_name(viewer, out_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = mask
        existing.refresh()
    else:
        viewer.add_labels(mask, name=out_name)
    return f"Applied Color Tuner step -> {out_name}"


def _apply_color_adjustments_step(viewer, params: dict) -> str:
    src_name = str(params.get("source_layer", ""))
    out_name = str(params.get("output_layer", f"{src_name} - Adjusted"))
    stack = list(params.get("adjustment_stack", []) or [])

    src = _layer_by_name(viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")
    arr = _image_data(src)
    adjusted = np.asarray(apply_adjustments_to_video(arr, stack), dtype=np.uint8)

    existing = _layer_by_name(viewer, out_name)
    if existing is not None and isinstance(existing, Image):
        existing.data = adjusted
        existing.refresh()
    else:
        viewer.add_image(adjusted, name=out_name)
    return f"Applied Color Adjustments step -> {out_name}"


def _apply_pecan_ellipse_step(viewer, params: dict) -> str:
    src_name = str(params.get("mask_layer", ""))
    out_name = str(params.get("output_shapes_layer", f"{src_name} - ellipse"))
    label_id = params.get("label_id")
    if label_id is not None:
        label_id = int(label_id)
    largest_only = bool(params.get("largest_only", True))
    mode = str(params.get("mode", "current"))
    time_index = int(params.get("time_index", 0))

    src = _layer_by_name(viewer, src_name)
    if src is None:
        raise ValueError(f"Mask layer not found: {src_name}")
    data = np.asarray(src.data)

    verts_list = []
    if mode == "all" and mask_volume_needs_time_coord(data):
        for t in range(int(data.shape[0])):
            v = apply_ellipse_pipeline(data, t, label_id=label_id, largest_only=largest_only)
            if v is not None:
                verts_list.append(v)
    else:
        v = apply_ellipse_pipeline(data, time_index, label_id=label_id, largest_only=largest_only)
        if v is not None:
            verts_list = [v]

    if not verts_list:
        raise ValueError("No ellipse could be fit with current settings.")

    existing = _layer_by_name(viewer, out_name)
    if existing is not None and isinstance(existing, Shapes):
        existing.data = verts_list
        existing.refresh()
    else:
        viewer.add_shapes(verts_list, shape_type="ellipse", name=out_name, face_color="transparent")
    return f"Applied Pecan Ellipse step -> {out_name}"


def _apply_mask_ops_step(viewer, params: dict) -> str:
    mode = str(params.get("mode", "binary"))
    if mode == "clip":
        ellipse_name = str(params.get("ellipse_layer", ""))
        mask_name = str(params.get("mask_layer", ""))
        output_mode = str(params.get("output_mode", "new"))
        output_name = str(params.get("output_layer", f"{mask_name} - inside ellipse"))

        ellipse_layer = _layer_by_name(viewer, ellipse_name)
        mask_layer = _layer_by_name(viewer, mask_name)
        if ellipse_layer is None or not isinstance(ellipse_layer, Shapes):
            raise ValueError(f"Ellipse layer not found: {ellipse_name}")
        if mask_layer is None or not isinstance(mask_layer, Labels):
            raise ValueError(f"Mask layer not found: {mask_name}")
        mask = np.asarray(mask_layer.data)
        ell = build_ellipse_masks_for_volume(ellipse_layer, tuple(mask.shape))
        clipped = clip_mask_outside_ellipse(mask, ell)
        if output_mode == "overwrite":
            mask_layer.data = clipped
            mask_layer.refresh()
            return f"Applied clip and overwrote {mask_name}"
        existing = _layer_by_name(viewer, output_name)
        if existing is not None and isinstance(existing, Labels):
            existing.data = clipped
            existing.refresh()
        else:
            viewer.add_labels(clipped, name=output_name)
        return f"Applied clip -> {output_name}"

    a_name = str(params.get("a_layer", ""))
    b_name = str(params.get("b_layer", ""))
    op = str(params.get("op", "and"))
    target = str(params.get("target", "new"))
    output_name = str(params.get("output_layer", f"{a_name} {op.upper()} {b_name}"))
    a_layer = _layer_by_name(viewer, a_name)
    b_layer = _layer_by_name(viewer, b_name) if op != "not" else a_layer
    if a_layer is None or not isinstance(a_layer, Labels):
        raise ValueError(f"Mask A not found: {a_name}")
    if b_layer is None or not isinstance(b_layer, Labels):
        raise ValueError(f"Mask B not found: {b_name}")

    a = np.asarray(a_layer.data)
    b = np.asarray(b_layer.data)
    res = apply_binary_operation(a, b, op=op, template=a)
    if target == "a":
        a_layer.data = res
        a_layer.refresh()
        return f"Applied {op.upper()} and overwrote {a_name}"
    if target == "b":
        b_layer.data = res
        b_layer.refresh()
        return f"Applied {op.upper()} and overwrote {b_name}"

    existing = _layer_by_name(viewer, output_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = res
        existing.refresh()
    else:
        viewer.add_labels(res, name=output_name)
    return f"Applied {op.upper()} -> {output_name}"


def apply_pipeline_step(viewer, step: dict) -> str:
    kind = str(step.get("kind", ""))
    params = dict(step.get("params", {}) or {})
    if kind == "color_tuner.threshold":
        return _apply_color_tuner_step(viewer, params)
    if kind == "color_adjustments.stack":
        return _apply_color_adjustments_step(viewer, params)
    if kind == "pecan_ellipse.fit":
        return _apply_pecan_ellipse_step(viewer, params)
    if kind == "mask_ops.operation":
        return _apply_mask_ops_step(viewer, params)
    raise ValueError(f"Unknown pipeline step kind: {kind}")
