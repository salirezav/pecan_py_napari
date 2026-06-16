"""Pipeline execution logic."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Shapes

from ..color_adjustments.logic import apply_adjustments_to_video
from ..color_adjustments.recipes import AdjustmentRecipe, write_recipe_metadata
from ..color_thresholding.logic import apply_thresholds
from ..edge_detection.logic import apply_edges_to_volume
from ..mask_ops.logic import (
    apply_binary_operation,
    build_ellipse_masks_for_volume,
    clip_mask_outside_ellipse,
)
from ..mask_retouching.logic import apply_retouching_pipeline
from ..pecan_ellipse.logic import (
    apply_ellipse_pipeline,
    fit_ellipses_volume,
    mask_volume_needs_time_coord,
    normalize_smooth_window,
)


def _check_cancel(cancel_callback) -> None:
    if cancel_callback is None:
        return
    try:
        if bool(cancel_callback()):
            raise InterruptedError("Pipeline apply cancelled by user.")
    except InterruptedError:
        raise
    except Exception:
        return


def _emit_progress(progress_callback, current: int, total: int, phase: str = "", cancel_callback=None) -> None:
    _check_cancel(cancel_callback)
    if progress_callback is None:
        return
    try:
        progress_callback(int(current), int(total), str(phase))
        _check_cancel(cancel_callback)
    except InterruptedError:
        raise
    except Exception:
        # Progress callbacks are best-effort only.
        pass


class _ApplyContext:
    def __init__(self, viewer) -> None:
        self.viewer = viewer
        self.name_map: dict[str, str] = {}
        self.root_image_name = self._pick_root_image_name()

    def _pick_root_image_name(self) -> str | None:
        try:
            active = self.viewer.layers.selection.active
            if isinstance(active, Image):
                return str(active.name)
        except Exception:
            pass
        images = [layer for layer in self.viewer.layers if isinstance(layer, Image)]
        if len(images) == 1:
            return str(images[0].name)
        if images:
            return str(images[0].name)
        return None


def create_apply_context(viewer):
    return _ApplyContext(viewer)


def _layer_by_name(viewer, name: str):
    try:
        return viewer.layers[str(name)]
    except Exception:
        return None


def _resolve_input_name(ctx: _ApplyContext, recorded_name: str, expected_type=None) -> str:
    mapped = str(ctx.name_map.get(recorded_name, recorded_name))
    layer = _layer_by_name(ctx.viewer, mapped)
    if layer is not None and (expected_type is None or isinstance(layer, expected_type)):
        return mapped
    # If an image source from a different project name is missing, map to currently available input image.
    if expected_type is Image and ctx.root_image_name:
        root = str(ctx.root_image_name)
        root_layer = _layer_by_name(ctx.viewer, root)
        if root_layer is not None and isinstance(root_layer, Image):
            ctx.name_map[recorded_name] = root
            return root
    return mapped


def _derive_output_name(ctx: _ApplyContext, recorded_output: str, recorded_source: str, resolved_source: str) -> str:
    out = str(ctx.name_map.get(recorded_output, recorded_output))
    if _layer_by_name(ctx.viewer, out) is not None:
        return out
    # Preserve pipeline naming intent while rebasing onto current input video names.
    if recorded_output.startswith(recorded_source):
        suffix = recorded_output[len(recorded_source):]
        candidate = f"{resolved_source}{suffix}"
        return candidate
    return out


def _image_data(layer: Image) -> np.ndarray:
    data = layer.data
    shape = getattr(data, "shape", None)
    if shape is not None:
        try:
            shp = tuple(int(x) for x in shape)
        except Exception:
            shp = None
        if shp is not None and len(shp) in (3, 4):
            if isinstance(data, np.ndarray):
                arr = data
            else:
                # For lazy/video-backed sources, materialize safely by slice.
                if len(shp) == 3:
                    arr = np.asarray(data)
                else:
                    frames = [np.asarray(data[t]) for t in range(int(shp[0]))]
                    arr = np.stack(frames, axis=0)
            if arr.ndim in (3, 4):
                return arr
    arr = np.asarray(data)
    if arr.ndim not in (3, 4):
        raise ValueError(
            f"Expected image shape (H,W,C) or (T,H,W,C), got {arr.shape} "
            f"(layer={getattr(layer, 'name', '?')}, data_type={type(data).__name__})"
        )
    return arr


def _frame_rgb(arr: np.ndarray, t: int) -> np.ndarray:
    if arr.ndim == 3:
        frame = arr
    else:
        frame = arr[int(np.clip(t, 0, arr.shape[0] - 1))]
    if frame.ndim != 3 or frame.shape[-1] < 3:
        raise ValueError(f"Expected RGB frame, got {frame.shape}")
    return np.asarray(frame[..., :3], dtype=np.uint8)


def _apply_color_thresholding_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw, expected_type=Image)
    target = str(params.get("target", "pecan"))
    color_space = str(params.get("color_space", "rgb"))
    lower = np.asarray(params.get("lower", [0, 0, 0]), dtype=np.uint8)
    upper = np.asarray(params.get("upper", [255, 255, 255]), dtype=np.uint8)
    out_recorded = str(params.get("output_mask_layer", f"{src_name_raw} - {target.title()}"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)

    src = _layer_by_name(ctx.viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")
    arr = _image_data(src)
    thresholds = {color_space: {target: {"lower": lower, "upper": upper}}}

    total = 1 if arr.ndim == 3 else int(arr.shape[0])
    _emit_progress(progress_callback, 0, total, "thresholding", cancel_callback=cancel_callback)
    if arr.ndim == 3:
        mask = (apply_thresholds(_frame_rgb(arr, 0), color_space, target, thresholds) > 0).astype(np.uint8)
        _emit_progress(progress_callback, 1, total, "thresholding", cancel_callback=cancel_callback)
    else:
        masks = []
        for t in range(arr.shape[0]):
            _check_cancel(cancel_callback)
            masks.append((apply_thresholds(_frame_rgb(arr, t), color_space, target, thresholds) > 0).astype(np.uint8))
            _emit_progress(progress_callback, t + 1, total, "thresholding", cancel_callback=cancel_callback)
        mask = np.stack(masks, axis=0).astype(np.uint8)

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = mask
        existing.refresh()
    else:
        ctx.viewer.add_labels(mask, name=out_name)
    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    return f"Applied Color Thresholding step -> {out_name}"


def _apply_color_adjustments_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw, expected_type=Image)
    out_recorded = str(params.get("output_layer", f"{src_name_raw} - Adjusted"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)
    stack = list(params.get("adjustment_stack", []) or [])

    src = _layer_by_name(ctx.viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")
    arr = _image_data(src)
    adjusted = np.asarray(
        apply_adjustments_to_video(
            arr,
            stack,
            progress_callback=lambda c, t: _emit_progress(
                progress_callback, c, t, "adjusting", cancel_callback=cancel_callback
            ),
        ),
        dtype=np.uint8,
    )

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Image):
        existing.data = adjusted
        existing.refresh()
        out_layer = existing
    else:
        out_layer = ctx.viewer.add_image(adjusted, name=out_name)
    recipe = AdjustmentRecipe.new(src_name, out_name, adjustment_stack=stack)
    write_recipe_metadata(out_layer, recipe)
    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    return f"Applied Adjustments step -> {out_name}"


def _apply_pecan_ellipse_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    src_name_raw = str(params.get("mask_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw)
    out_recorded = str(params.get("output_shapes_layer", f"{src_name_raw} - ellipse"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)
    label_id = params.get("label_id")
    if label_id is not None:
        label_id = int(label_id)
    largest_only = bool(params.get("largest_only", True))
    temporal_smooth = bool(params.get("temporal_smooth", False))
    smooth_window = normalize_smooth_window(int(params.get("smooth_window", 5)))
    mode = str(params.get("mode", "current"))
    time_index = int(params.get("time_index", 0))

    src = _layer_by_name(ctx.viewer, src_name)
    if src is None:
        raise ValueError(f"Mask layer not found: {src_name}")
    raw = src.data
    data = np.asarray(raw)
    if data.dtype == object and isinstance(raw, (list, tuple)) and len(raw) > 0:
        try:
            data = np.stack([np.asarray(x) for x in raw], axis=0)
        except Exception:
            data = np.asarray(raw[0])
    is_time_series = (isinstance(src, Labels) and data.ndim >= 3) or mask_volume_needs_time_coord(data)

    verts_list = []
    if mode == "all" and is_time_series:
        total = int(data.shape[0])
        _emit_progress(progress_callback, 0, total, "fitting ellipse", cancel_callback=cancel_callback)
        _check_cancel(cancel_callback)
        verts_list = fit_ellipses_volume(
            data,
            label_id=label_id,
            largest_only=largest_only,
            temporal_smooth=temporal_smooth,
            smooth_window=smooth_window,
        )
        _emit_progress(progress_callback, total, total, "fitting ellipse", cancel_callback=cancel_callback)
    else:
        v = apply_ellipse_pipeline(data, time_index, label_id=label_id, largest_only=largest_only)
        if v is not None:
            verts_list = [v]
        _emit_progress(progress_callback, 1, 1, "fitting ellipse", cancel_callback=cancel_callback)

    if not verts_list:
        raise ValueError("No ellipse could be fit with current settings.")

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Shapes):
        existing.data = verts_list
        existing.refresh()
    else:
        ctx.viewer.add_shapes(verts_list, shape_type="ellipse", name=out_name, face_color="transparent")
    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    return f"Applied Pecan Ellipse step -> {out_name}"


def _apply_mask_ops_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    _emit_progress(progress_callback, 0, 1, "combining masks", cancel_callback=cancel_callback)
    mode = str(params.get("mode", "binary"))
    if mode == "clip":
        ellipse_name_raw = str(params.get("ellipse_layer", ""))
        mask_name_raw = str(params.get("mask_layer", ""))
        ellipse_name = _resolve_input_name(ctx, ellipse_name_raw, expected_type=Shapes)
        mask_name = _resolve_input_name(ctx, mask_name_raw, expected_type=Labels)
        output_mode = str(params.get("output_mode", "new"))
        out_recorded = str(params.get("output_layer", f"{mask_name_raw} - inside ellipse"))
        output_name = _derive_output_name(ctx, out_recorded, mask_name_raw, mask_name)

        ellipse_layer = _layer_by_name(ctx.viewer, ellipse_name)
        mask_layer = _layer_by_name(ctx.viewer, mask_name)
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
            _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
            return f"Applied clip and overwrote {mask_name}"
        existing = _layer_by_name(ctx.viewer, output_name)
        if existing is not None and isinstance(existing, Labels):
            existing.data = clipped
            existing.refresh()
        else:
            ctx.viewer.add_labels(clipped, name=output_name)
        ctx.name_map[ellipse_name_raw] = ellipse_name
        ctx.name_map[mask_name_raw] = mask_name
        ctx.name_map[out_recorded] = output_name
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied clip -> {output_name}"

    a_name_raw = str(params.get("a_layer", ""))
    b_name_raw = str(params.get("b_layer", ""))
    a_name = _resolve_input_name(ctx, a_name_raw, expected_type=Labels)
    b_name = _resolve_input_name(ctx, b_name_raw, expected_type=Labels)
    op = str(params.get("op", "and"))
    target = str(params.get("target", "new"))
    out_recorded = str(params.get("output_layer", f"{a_name_raw} {op.upper()} {b_name_raw}"))
    output_name = _derive_output_name(ctx, out_recorded, a_name_raw, a_name)
    a_layer = _layer_by_name(ctx.viewer, a_name)
    b_layer = _layer_by_name(ctx.viewer, b_name) if op != "not" else a_layer
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
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied {op.upper()} and overwrote {a_name}"
    if target == "b":
        b_layer.data = res
        b_layer.refresh()
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied {op.upper()} and overwrote {b_name}"

    existing = _layer_by_name(ctx.viewer, output_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = res
        existing.refresh()
    else:
        ctx.viewer.add_labels(res, name=output_name)
    ctx.name_map[a_name_raw] = a_name
    ctx.name_map[b_name_raw] = b_name
    ctx.name_map[out_recorded] = output_name
    _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
    return f"Applied {op.upper()} -> {output_name}"


def _write_mask_layer_to_disk(
    ctx: _ApplyContext,
    mask_layer: Labels,
    mask_name: str,
    mask_name_raw: str,
    fmt: str,
    output_dir: str = "",
    progress_callback=None,
    cancel_callback=None,
) -> str:
    fmt = str(fmt).lower()
    if fmt not in ("tiff", "npy"):
        raise ValueError(f"Unsupported save format: {fmt}")
    ext = ".tiff" if fmt == "tiff" else ".npy"

    recorded_dir = str(output_dir or "").strip()
    out_dir = recorded_dir or _find_source_video_dir(ctx)
    if not out_dir:
        raise ValueError(
            "Save masks step: no output directory available. Either set output_dir "
            "in the step or load a video so its folder can be used."
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = str(Path(out_dir) / (mask_name + ext))

    _emit_progress(progress_callback, 0, 1, "saving masks", cancel_callback=cancel_callback)
    data = np.asarray(mask_layer.data)
    if fmt == "tiff":
        import tifffile
        tifffile.imwrite(out_path, data)
    else:
        np.save(out_path, data)
    _emit_progress(progress_callback, 1, 1, "saving masks", cancel_callback=cancel_callback)

    ctx.name_map[mask_name_raw] = mask_name
    return f"Saved masks ({fmt.upper()}) -> {out_path}"


def _apply_mask_retouching_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    mask_name_raw = str(params.get("mask_layer", ""))
    mask_name = _resolve_input_name(ctx, mask_name_raw, expected_type=Labels)
    mask_layer = _layer_by_name(ctx.viewer, mask_name)
    if mask_layer is None or not isinstance(mask_layer, Labels):
        raise ValueError(f"Mask layer not found: {mask_name}")

    data = np.asarray(mask_layer.data)
    op_params = dict(
        close_size=int(params.get("close_size", 0)),
        open_size=int(params.get("open_size", 0)),
        dilate_size=int(params.get("dilate_size", 0)),
        dilate_iter=int(params.get("dilate_iter", 1)),
        erode_size=int(params.get("erode_size", 0)),
        erode_iter=int(params.get("erode_iter", 1)),
        min_area=int(params.get("min_area", 0)),
        do_fill_holes=bool(params.get("do_fill_holes", False)),
        do_keep_largest=bool(params.get("do_keep_largest", False)),
        smooth_size=int(params.get("smooth_size", 0)),
    )
    total = 1 if data.ndim == 2 else int(data.shape[0])
    _emit_progress(progress_callback, 0, total, "retouching", cancel_callback=cancel_callback)
    if data.ndim == 2:
        out = apply_retouching_pipeline(data, **op_params)
        _emit_progress(progress_callback, 1, total, "retouching", cancel_callback=cancel_callback)
    else:
        frames = []
        for t in range(total):
            _check_cancel(cancel_callback)
            frames.append(apply_retouching_pipeline(data[t], **op_params))
            _emit_progress(progress_callback, t + 1, total, "retouching", cancel_callback=cancel_callback)
        out = np.stack(frames, axis=0)
    mask_layer.data = out
    mask_layer.refresh()
    ctx.name_map[mask_name_raw] = mask_name
    msg = f"Applied Mask Retouching -> {mask_name}"
    if bool(params.get("save_mask", False)):
        save_msg = _write_mask_layer_to_disk(
            ctx,
            mask_layer,
            mask_name,
            mask_name_raw,
            str(params.get("format", "tiff")),
            str(params.get("output_dir", "") or ""),
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
        )
        msg = f"{msg}; {save_msg}"
    return msg


def _find_source_video_dir(ctx: _ApplyContext) -> str | None:
    """Return the directory of the first Image layer that has a source_path."""
    for layer in ctx.viewer.layers:
        if isinstance(layer, Image):
            src = getattr(layer, "metadata", {}).get("source_path")
            if src:
                return str(Path(src).parent)
    return None


def _apply_mask_retouching_save_masks_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    mask_name_raw = str(params.get("mask_layer", ""))
    mask_name = _resolve_input_name(ctx, mask_name_raw, expected_type=Labels)
    mask_layer = _layer_by_name(ctx.viewer, mask_name)
    if mask_layer is None or not isinstance(mask_layer, Labels):
        raise ValueError(f"Mask layer not found: {mask_name}")

    return _write_mask_layer_to_disk(
        ctx,
        mask_layer,
        mask_name,
        mask_name_raw,
        str(params.get("format", "tiff")),
        str(params.get("output_dir", "") or ""),
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
    )


def _apply_edge_detection_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw, expected_type=Image)
    method = str(params.get("method", "canny"))
    method_label = str(params.get("method_label", method))
    method_params = dict(params.get("params", {}) or {})
    out_recorded = str(params.get("output_layer", f"{src_name_raw} - Edges ({method_label})"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)

    src = _layer_by_name(ctx.viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")

    arr = _image_data(src)
    out = apply_edges_to_volume(
        arr,
        method=method,
        params=method_params,
        state={},
        progress_callback=lambda c, t: _emit_progress(
            progress_callback, c, t, "edge-detection", cancel_callback=cancel_callback
        ),
    ).astype(np.uint8)

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Image):
        existing.data = out
        existing.refresh()
    else:
        ctx.viewer.add_image(out, name=out_name, colormap="gray")
    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    return f"Applied Edge Detection ({method_label}) -> {out_name}"


def apply_pipeline_step(viewer, step: dict, progress_callback=None, cancel_callback=None) -> str:
    ctx = _ApplyContext(viewer)
    return _apply_pipeline_step_in_context(
        ctx, step, progress_callback=progress_callback, cancel_callback=cancel_callback
    )


def apply_pipeline_step_with_context(ctx, step: dict, progress_callback=None, cancel_callback=None) -> str:
    return _apply_pipeline_step_in_context(
        ctx, step, progress_callback=progress_callback, cancel_callback=cancel_callback
    )


def _apply_pipeline_step_in_context(ctx: _ApplyContext, step: dict, progress_callback=None, cancel_callback=None) -> str:
    kind = str(step.get("kind", ""))
    params = dict(step.get("params", {}) or {})
    if kind in ("color_thresholding.threshold", "color_tuner.threshold"):
        return _apply_color_thresholding_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "color_adjustments.stack":
        return _apply_color_adjustments_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "pecan_ellipse.fit":
        return _apply_pecan_ellipse_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "mask_ops.operation":
        return _apply_mask_ops_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "mask_retouching.apply":
        return _apply_mask_retouching_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "mask_retouching.save_masks":
        return _apply_mask_retouching_save_masks_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "edge_detection.apply":
        return _apply_edge_detection_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    raise ValueError(f"Unknown pipeline step kind: {kind}")


def apply_pipeline_steps(viewer, steps: list[dict]) -> list[str]:
    ctx = _ApplyContext(viewer)
    msgs: list[str] = []
    for step in steps:
        msgs.append(_apply_pipeline_step_in_context(ctx, step))
    return msgs
