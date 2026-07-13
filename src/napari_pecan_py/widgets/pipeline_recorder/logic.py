"""Pipeline execution logic."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
from napari.layers import Image, Labels, Layer, Shapes

from .state import ROOT_PLACEHOLDER, infer_recorded_root

from ..color_adjustments.logic import apply_adjustments_to_video
from ..color_adjustments.recipes import AdjustmentRecipe, write_recipe_metadata
from ..color_thresholding.logic import apply_thresholds
from ..edge_detection.logic import apply_edges_to_volume
from ..mask_ops.logic import (
    apply_binary_operation,
    apply_binary_operation_bool,
    build_ellipse_masks_for_volume,
    clip_mask_label_volume,
    clip_mask_outside_ellipse,
    detect_parallel_edge_bands_volume,
    expand_mask_to_layer_shape,
    label_only_volume,
    labels_from_bool_mask,
    mask_volume_for_label,
    merge_label_mask_into_volume,
    new_labels_from_binary,
)
from ..mask_retouching.logic import apply_retouching_to_volume
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
    def __init__(
        self,
        viewer,
        *,
        steps: list[dict] | None = None,
        recorded_root: str | None = None,
    ) -> None:
        self.viewer = viewer
        self.name_map: dict[str, str] = {}
        self.current_root = self._pick_current_root()
        self.recorded_root = infer_recorded_root(steps or [], recorded_root)
        if self.recorded_root == ROOT_PLACEHOLDER:
            self.recorded_root = self.current_root
        self.root_image_name = self.current_root

    def _pick_current_root(self) -> str | None:
        images_with_path: list[Image] = []
        images: list[Image] = []
        for layer in self.viewer.layers:
            if isinstance(layer, Image):
                images.append(layer)
                if getattr(layer, "metadata", {}).get("source_path"):
                    images_with_path.append(layer)
        try:
            active = self.viewer.layers.selection.active
            if isinstance(active, Image):
                return str(active.name)
        except Exception:
            pass
        if len(images_with_path) == 1:
            return str(images_with_path[0].name)
        if images_with_path:
            return str(images_with_path[0].name)
        if len(images) == 1:
            return str(images[0].name)
        if images:
            return str(images[0].name)
        return None


def create_apply_context(
    viewer,
    steps: list[dict] | None = None,
    recorded_root: str | None = None,
):
    return _ApplyContext(viewer, steps=steps, recorded_root=recorded_root)


def _layer_by_name(viewer, name: str):
    try:
        return viewer.layers[str(name)]
    except Exception:
        return None


def _rebase_name(ctx: _ApplyContext, name: str) -> str:
    text = str(name or "")
    if text == ROOT_PLACEHOLDER and ctx.current_root:
        return str(ctx.current_root)
    recorded = ctx.recorded_root
    current = ctx.current_root
    if not recorded or not current or recorded == current:
        return text
    return text.replace(recorded, current)


def _find_layer_by_suffix(ctx: _ApplyContext, recorded_name: str, expected_type=None) -> str | None:
    suffix = None
    recorded = str(recorded_name or "")
    if ctx.recorded_root and recorded.startswith(str(ctx.recorded_root)):
        suffix = recorded[len(str(ctx.recorded_root)) :]
    elif " - " in recorded:
        suffix = recorded[recorded.index(" - ") :]
    if not suffix:
        return None

    current = str(ctx.current_root or "")
    candidate = f"{current}{suffix}"
    layer = _layer_by_name(ctx.viewer, candidate)
    if layer is not None and (expected_type is None or isinstance(layer, expected_type)):
        return candidate

    for layer in ctx.viewer.layers:
        layer_name = str(layer.name)
        if layer_name.endswith(suffix) and (expected_type is None or isinstance(layer, expected_type)):
            return layer_name
    return None


def _resolve_input_name(ctx: _ApplyContext, recorded_name: str, expected_type=None) -> str:
    raw = str(recorded_name or "")
    mapped = str(ctx.name_map.get(raw, _rebase_name(ctx, raw)))
    layer = _layer_by_name(ctx.viewer, mapped)
    if layer is not None and (expected_type is None or isinstance(layer, expected_type)):
        return mapped

    suffix_match = _find_layer_by_suffix(ctx, raw, expected_type=expected_type)
    if suffix_match is not None:
        ctx.name_map[raw] = suffix_match
        return suffix_match

    rebased = _rebase_name(ctx, raw)
    if rebased != mapped:
        layer = _layer_by_name(ctx.viewer, rebased)
        if layer is not None and (expected_type is None or isinstance(layer, expected_type)):
            ctx.name_map[raw] = rebased
            return rebased

    if expected_type is Image and ctx.current_root:
        root = str(ctx.current_root)
        root_layer = _layer_by_name(ctx.viewer, root)
        if root_layer is not None and isinstance(root_layer, Image):
            ctx.name_map[raw] = root
            return root
    return mapped


def _derive_output_name(ctx: _ApplyContext, recorded_output: str, recorded_source: str, resolved_source: str) -> str:
    out = str(ctx.name_map.get(recorded_output, recorded_output))
    if _layer_by_name(ctx.viewer, out) is not None:
        return out

    rebased = _rebase_name(ctx, out)
    if rebased != out:
        return rebased

    if recorded_output.startswith(recorded_source):
        suffix = recorded_output[len(recorded_source) :]
        return f"{resolved_source}{suffix}"

    if recorded_source and recorded_source in recorded_output:
        return recorded_output.replace(recorded_source, resolved_source)

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
            cancel_callback=cancel_callback,
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


def _optional_int_param(params: dict, key: str) -> int | None:
    raw = params.get(key, "")
    if raw == "" or raw is None:
        return None
    return int(raw)


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
        mask_label = _optional_int_param(params, "mask_label")
        if mask_label is not None:
            clipped = clip_mask_label_volume(mask, ell, mask_label)
        else:
            clipped = clip_mask_outside_ellipse(mask, ell)
        if output_mode == "overwrite":
            mask_layer.data = clipped
            mask_layer.refresh()
            _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
            return f"Applied clip and overwrote {mask_name}"
        if mask_label is not None:
            clipped = label_only_volume(clipped, mask_label)
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

    if mode == "parallel_bands":
        edge_name_raw = str(params.get("edge_layer", ""))
        limit_name_raw = str(params.get("limit_mask_layer", ""))
        edge_name = _resolve_input_name(ctx, edge_name_raw, expected_type=Image)
        edge_layer = _layer_by_name(ctx.viewer, edge_name)
        if edge_layer is None or not isinstance(edge_layer, Image):
            raise ValueError(f"Edge layer not found: {edge_name}")
        edges = _image_data(edge_layer)

        limit = None
        if limit_name_raw.strip():
            lim_name = _resolve_input_name(ctx, limit_name_raw, expected_type=Labels)
            lim_layer = _layer_by_name(ctx.viewer, lim_name)
            if lim_layer is None or not isinstance(lim_layer, Labels):
                raise ValueError(f"Limit mask layer not found: {lim_name}")
            lim_raw = np.asarray(lim_layer.data)
            limit_label = _optional_int_param(params, "limit_mask_label")
            limit = mask_volume_for_label(lim_raw, limit_label)
            ctx.name_map[limit_name_raw] = lim_name

        bands_bool = detect_parallel_edge_bands_volume(
            edges,
            limit_mask=limit,
            edge_threshold=int(params.get("edge_threshold", 1)),
            pre_close_size=int(params.get("pre_close_size", 3)),
            min_distance_px=int(params.get("min_distance_px", 2)),
            max_distance_px=int(params.get("max_distance_px", 12)),
            angle_tolerance_deg=int(params.get("angle_tolerance_deg", 25)),
            min_component_px=int(params.get("min_component_px", 20)),
        )
        bands = labels_from_bool_mask(bands_bool, edges)

        out_recorded = str(params.get("output_layer", f"{edge_name_raw} - parallel bands"))
        out_name = _derive_output_name(ctx, out_recorded, edge_name_raw, edge_name)
        existing = _layer_by_name(ctx.viewer, out_name)
        if existing is not None and isinstance(existing, Labels):
            existing.data = bands
            existing.refresh()
        else:
            ctx.viewer.add_labels(bands, name=out_name)
        ctx.name_map[edge_name_raw] = edge_name
        ctx.name_map[out_recorded] = out_name
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied parallel bands -> {out_name}"

    a_name_raw = str(params.get("a_layer", ""))
    b_name_raw = str(params.get("b_layer", ""))
    mask_types = (Labels, Image)
    a_name = _resolve_input_name(ctx, a_name_raw, expected_type=mask_types)
    b_name = _resolve_input_name(ctx, b_name_raw, expected_type=mask_types)
    op = str(params.get("op", "and"))
    target = str(params.get("target", "new"))
    out_recorded = str(params.get("output_layer", f"{a_name_raw} {op.upper()} {b_name_raw}"))
    output_name = _derive_output_name(ctx, out_recorded, a_name_raw, a_name)
    a_layer = _layer_by_name(ctx.viewer, a_name)
    b_layer = _layer_by_name(ctx.viewer, b_name) if op != "not" else a_layer
    if a_layer is None or not isinstance(a_layer, mask_types):
        raise ValueError(f"Mask A not found: {a_name}")
    if b_layer is None or not isinstance(b_layer, mask_types):
        raise ValueError(f"Mask B not found: {b_name}")

    a_raw = np.asarray(a_layer.data)
    b_raw = np.asarray(b_layer.data) if op != "not" else np.array(a_raw, copy=False)
    label_a = _optional_int_param(params, "a_label")
    label_b = _optional_int_param(params, "b_label")
    if op == "not":
        label_b = label_a
    res_bool = apply_binary_operation_bool(a_raw, b_raw, op=op, label_a=label_a, label_b=label_b)

    def _write_binary(layer, template_raw: np.ndarray) -> None:
        layer.data = expand_mask_to_layer_shape(
            apply_binary_operation(
                a_raw, b_raw, op=op, template=template_raw, label_a=label_a, label_b=label_b
            ),
            template_raw,
        )
        layer.refresh()

    def _write_labels(layer: Labels, template_raw: np.ndarray, label_value: int) -> None:
        layer.data = merge_label_mask_into_volume(template_raw, res_bool, label_value)
        layer.refresh()

    if target == "a":
        if isinstance(a_layer, Labels) and label_a is not None:
            _write_labels(a_layer, a_raw, label_a)
        else:
            _write_binary(a_layer, a_raw)
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied {op.upper()} and overwrote {a_name}"
    if target == "b":
        if isinstance(b_layer, Labels) and label_b is not None:
            _write_labels(b_layer, b_raw, label_b)
        else:
            _write_binary(b_layer, b_raw)
        _emit_progress(progress_callback, 1, 1, "combining masks", cancel_callback=cancel_callback)
        return f"Applied {op.upper()} and overwrote {b_name}"

    if isinstance(a_layer, Labels):
        out_label = label_a if label_a is not None else 1
        out_data = new_labels_from_binary(res_bool, out_label, dtype=a_raw.dtype)
    else:
        out_data = expand_mask_to_layer_shape(
            apply_binary_operation(
                a_raw, b_raw, op=op, template=a_raw, label_a=label_a, label_b=label_b
            ),
            a_raw,
        )
    existing = _layer_by_name(ctx.viewer, output_name)
    if existing is not None and isinstance(existing, mask_types):
        existing.data = out_data
        existing.refresh()
    elif isinstance(a_layer, Image):
        ctx.viewer.add_image(out_data, name=output_name)
    else:
        ctx.viewer.add_labels(out_data, name=output_name)
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
        fill_holes_min_area=int(params.get("fill_holes_min_area", 0)),
        fill_holes_max_area=int(params.get("fill_holes_max_area", 0)),
        do_watershed_split=bool(params.get("do_watershed_split", False)),
        watershed_min_distance=int(params.get("watershed_min_distance", 15)),
        do_keep_largest=bool(params.get("do_keep_largest", False)),
        smooth_size=int(params.get("smooth_size", 0)),
    )
    out = apply_retouching_to_volume(
        data,
        **op_params,
        progress_callback=lambda c, t: _emit_progress(
            progress_callback, c, t, "retouching", cancel_callback=cancel_callback
        ),
        cancel_callback=cancel_callback,
    )
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


def _layer_source_path(layer: Image) -> str | None:
    meta = getattr(layer, "metadata", None) or {}
    if isinstance(meta, dict):
        src = meta.get("source_path")
        if src:
            return str(Path(src).resolve())
    return None


def _apply_yolo_seg_inference_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    from ..cascade_seg.model import (
        BACKEND_CASCADE,
        detect_seg_checkpoint_backend,
        run_cascade_inference_on_frames,
    )
    from ..unet_seg.model import BACKEND_UNET, run_unet_inference_on_frames
    from ..yolo_seg.model import (
        infer_labels_layer_name,
        infer_mask_output_path,
        resolve_yolo_device,
        run_yolo_seg_inference_on_frames,
        save_mask_volume,
    )

    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw, expected_type=Image)
    weights_path = Path(str(params.get("weights_path", "")))
    if not weights_path.is_file():
        raise ValueError(f"Segmentation weights not found: {weights_path}")

    backend = str(params.get("backend", "") or detect_seg_checkpoint_backend(weights_path))
    device = resolve_yolo_device(str(params.get("device", "auto")))
    save_masks = bool(params.get("save_masks", False))
    save_suffix = str(params.get("save_suffix", ""))
    save_fmt = str(params.get("save_fmt", "tiff"))
    out_recorded = str(
        params.get(
            "output_mask_layer",
            infer_labels_layer_name(src_name_raw, save_suffix),
        )
    )
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)

    src = _layer_by_name(ctx.viewer, src_name)
    if src is None or not isinstance(src, Image):
        raise ValueError(f"Source image layer not found: {src_name}")

    arr = _image_data(src)
    total = 1 if arr.ndim == 3 else int(arr.shape[0])
    step_tag = {
        BACKEND_CASCADE: "cascade-inference",
        BACKEND_UNET: "unet-inference",
    }.get(backend, "yolo-inference")
    _emit_progress(progress_callback, 0, total, step_tag, cancel_callback=cancel_callback)

    def _frame_progress(cur: int, tot: int) -> None:
        _emit_progress(
            progress_callback,
            cur,
            max(tot, total),
            step_tag,
            cancel_callback=cancel_callback,
        )

    infer_fn = {
        BACKEND_CASCADE: run_cascade_inference_on_frames,
        BACKEND_UNET: run_unet_inference_on_frames,
    }.get(backend, run_yolo_seg_inference_on_frames)
    labels = infer_fn(
        weights_path,
        arr,
        device,
        progress_callback=_frame_progress,
        cancel_callback=cancel_callback,
    )
    _emit_progress(progress_callback, total, total, step_tag, cancel_callback=cancel_callback)

    saved_msg = ""
    if save_masks:
        source_path = _layer_source_path(src)
        if not source_path:
            saved_msg = " (mask file not saved: source layer has no source_path metadata)"
        else:
            out_path = infer_mask_output_path(source_path, save_suffix, save_fmt)
            save_mask_volume(labels, out_path, save_fmt)
            saved_msg = f"; saved mask -> {out_path}"

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = labels
        existing.refresh()
    else:
        ctx.viewer.add_labels(labels, name=out_name)

    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    return f"YOLO Seg inference -> {out_name}{saved_msg}"


def _insert_created_layer(viewer, layer: Layer) -> Layer:
    layers = viewer.layers
    if hasattr(layers, "append"):
        try:
            layers.append(layer)
            return layer
        except Exception:
            pass
    if hasattr(layers, "insert"):
        try:
            layers.insert(len(layers), layer)
            return layer
        except Exception:
            pass
    data, state, type_str = layer.as_layer_data_tuple()
    state = dict(state)
    name = str(state.get("name") or layer.name)
    if type_str == "image":
        return viewer.add_image(deepcopy(data), name=name, metadata=dict(state.get("metadata") or {}))
    if type_str == "labels":
        return viewer.add_labels(deepcopy(data), name=name)
    if type_str == "shapes":
        return viewer.add_shapes(
            deepcopy(data),
            name=name,
            shape_type=str(state.get("shape_type", "ellipse")),
            face_color=str(state.get("face_color", "transparent")),
        )
    raise ValueError(f"Unsupported layer type for insertion: {type_str}")


def _duplicate_layer_in_viewer(viewer, src_layer, out_name: str) -> Layer:
    data, state, type_str = src_layer.as_layer_data_tuple()
    state = dict(state)
    state["name"] = out_name
    new = Layer.create(deepcopy(data), state, type_str)
    return _insert_created_layer(viewer, new)


def _image_to_labels_data(src_layer: Image) -> np.ndarray:
    data = np.asarray(src_layer.data)
    if np.issubdtype(data.dtype, np.integer):
        return data.astype(np.int32, copy=False)
    if data.dtype == bool:
        return data.astype(np.int32)
    return np.where(data != 0, 1, 0).astype(np.int32)


def _shapes_to_labels_data(src_layer: Shapes) -> np.ndarray:
    try:
        return np.asarray(src_layer.to_labels())
    except TypeError:
        pass
    try:
        extent = src_layer._extent_data
        labels_shape = tuple(int(extent[1][i] - extent[0][i]) for i in range(len(extent[0])))
        return np.asarray(src_layer.to_labels(labels_shape=labels_shape))
    except Exception:
        return np.asarray(src_layer.to_labels(labels_shape=tuple(int(x) for x in src_layer.data.shape)))


def _apply_layer_duplicate_step(ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None) -> str:
    _emit_progress(progress_callback, 0, 1, "duplicating layer", cancel_callback=cancel_callback)
    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw)
    out_recorded = str(params.get("output_layer", f"{src_name_raw} copy"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)

    src_layer = _layer_by_name(ctx.viewer, src_name)
    if src_layer is None:
        raise ValueError(f"Source layer not found: {src_name}")

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None:
        data, state, type_str = src_layer.as_layer_data_tuple()
        existing.data = deepcopy(data)
        if hasattr(existing, "refresh"):
            existing.refresh()
    else:
        _duplicate_layer_in_viewer(ctx.viewer, src_layer, out_name)

    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    _emit_progress(progress_callback, 1, 1, "duplicating layer", cancel_callback=cancel_callback)
    return f"Duplicated layer -> {out_name}"


def _apply_layer_convert_to_labels_step(
    ctx: _ApplyContext, params: dict, progress_callback=None, cancel_callback=None
) -> str:
    _emit_progress(progress_callback, 0, 1, "convert to labels", cancel_callback=cancel_callback)
    src_name_raw = str(params.get("source_layer", ""))
    src_name = _resolve_input_name(ctx, src_name_raw)
    out_recorded = str(params.get("output_layer", f"{src_name_raw} - Labels"))
    out_name = _derive_output_name(ctx, out_recorded, src_name_raw, src_name)

    src_layer = _layer_by_name(ctx.viewer, src_name)
    if src_layer is None:
        raise ValueError(f"Source layer not found: {src_name}")
    if isinstance(src_layer, Image):
        labels_data = _image_to_labels_data(src_layer)
    elif isinstance(src_layer, Shapes):
        labels_data = _shapes_to_labels_data(src_layer)
    else:
        raise ValueError(f"Convert to Labels requires Image or Shapes, got {type(src_layer).__name__}")

    existing = _layer_by_name(ctx.viewer, out_name)
    if existing is not None and isinstance(existing, Labels):
        existing.data = labels_data
        existing.refresh()
    else:
        ctx.viewer.add_labels(labels_data, name=out_name)

    ctx.name_map[src_name_raw] = src_name
    ctx.name_map[out_recorded] = out_name
    _emit_progress(progress_callback, 1, 1, "convert to labels", cancel_callback=cancel_callback)
    return f"Converted to Labels -> {out_name}"


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
    if kind == "yolo_seg.inference":
        return _apply_yolo_seg_inference_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "layer.duplicate":
        return _apply_layer_duplicate_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    if kind == "layer.convert_to_labels":
        return _apply_layer_convert_to_labels_step(
            ctx, params, progress_callback=progress_callback, cancel_callback=cancel_callback
        )
    raise ValueError(f"Unknown pipeline step kind: {kind}")


def apply_pipeline_steps(viewer, steps: list[dict]) -> list[str]:
    ctx = _ApplyContext(viewer)
    msgs: list[str] = []
    for step in steps:
        msgs.append(_apply_pipeline_step_in_context(ctx, step))
    return msgs
