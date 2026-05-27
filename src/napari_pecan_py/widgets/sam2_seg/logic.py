"""SAM 2 inference helpers (image + video propagation)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import cv2
import numpy as np
from skimage.draw import polygon as sk_polygon

if TYPE_CHECKING:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

# Label values written into the output Labels layer (match common pecan targets).
CLASS_NAME_TO_ID = {
    "pecan": 1,
    "crack": 4,
    "background": 5,
}

_BACKEND: dict[str, Any] | None = None


def load_sam2_backend() -> dict[str, Any]:
    """Import PyTorch and SAM2 once (slow; ~3–6 s). Safe to call from a worker thread."""
    global _BACKEND
    if _BACKEND is not None:
        return _BACKEND

    state: dict[str, Any] = {
        "torch_available": False,
        "sam2_available": False,
        "import_error": None,
        "torch": None,
        "build_sam2": None,
        "build_sam2_video_predictor": None,
        "SAM2ImagePredictor": None,
    }

    try:
        import torch

        state["torch"] = torch
        state["torch_available"] = True
    except ImportError as exc:
        state["import_error"] = f"ImportError: {exc}"
        _BACKEND = state
        return state

    try:
        from sam2.build_sam import build_sam2, build_sam2_video_predictor
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        state["build_sam2"] = build_sam2
        state["build_sam2_video_predictor"] = build_sam2_video_predictor
        state["SAM2ImagePredictor"] = SAM2ImagePredictor
        state["sam2_available"] = True
    except Exception as exc:
        state["import_error"] = f"{type(exc).__name__}: {exc}"

    _BACKEND = state
    return state


def torch_available() -> bool:
    return bool(load_sam2_backend()["torch_available"])


def sam2_available() -> bool:
    return bool(load_sam2_backend()["sam2_available"])


def backend_import_error() -> str | None:
    return load_sam2_backend().get("import_error")


def default_device() -> str:
    torch_mod = load_sam2_backend().get("torch")
    if torch_mod is not None and torch_mod.cuda.is_available():
        return "cuda"
    if torch_mod is not None and getattr(torch_mod.backends, "mps", None) and torch_mod.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_sam2_config_path(config_path: str | None) -> str:
    if config_path and Path(config_path).is_file():
        return str(Path(config_path).resolve())
    if not sam2_available():
        raise RuntimeError("sam2 is not installed")
    import sam2

    root = Path(sam2.__file__).resolve().parent
    candidates = [
        root / "configs/sam2.1/sam2.1_hiera_s.yaml",
        root / "configs/sam2/sam2_hiera_s.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return str(p)
    raise FileNotFoundError(
        "Could not find a bundled SAM2 config. Set a config .yaml path in the widget."
    )


def resolve_sam2_checkpoint(checkpoint_path: str | None) -> str:
    if checkpoint_path and Path(checkpoint_path).is_file():
        return str(Path(checkpoint_path).resolve())
    try:
        from huggingface_hub import hf_hub_download

        for repo, fname in (
            ("facebook/sam2.1-hiera-small", "sam2.1_hiera_small.pt"),
            ("facebook/sam2-hiera-small", "sam2_hiera_small.pt"),
        ):
            try:
                return hf_hub_download(repo_id=repo, filename=fname)
            except Exception:
                continue
    except ImportError:
        pass
    raise FileNotFoundError(
        "SAM2 checkpoint not found. Download a .pt checkpoint or install huggingface_hub "
        "so the widget can fetch facebook/sam2.1-hiera-small."
    )


def _volume_shape(data: Any) -> tuple[int, ...]:
    """Shape without forcing ``np.asarray`` on lazy video adapters."""
    shape = getattr(data, "shape", None)
    if shape is not None:
        return tuple(int(x) for x in shape)
    arr = np.asarray(data)
    return tuple(int(x) for x in arr.shape)


def frame_rgb_uint8(data: Any, frame_index: int) -> np.ndarray:
    """Extract one RGB uint8 frame from napari-like image data."""
    shape = _volume_shape(data)
    if len(shape) == 4:
        # LazyVideoArray and similar: index the time axis, do not materialize full video.
        sl = np.asarray(data[int(frame_index)])
    elif len(shape) == 3:
        sl = np.asarray(data)
    else:
        raise ValueError(f"Expected 3D or 4D image data, got shape {shape}")
    if sl.ndim == 2:
        sl = np.stack([sl, sl, sl], axis=-1)
    if sl.shape[-1] == 4:
        sl = sl[..., :3]
    if sl.shape[-1] == 1:
        sl = np.repeat(sl, 3, axis=-1)
    if np.issubdtype(sl.dtype, np.floating):
        mx = float(np.nanmax(sl)) if sl.size else 1.0
        if mx <= 1.0:
            sl = sl * 255.0
    return np.clip(sl, 0, 255).astype(np.uint8)


def labels_2d_at_frame(data: Any, frame_index: int) -> np.ndarray | None:
    """One 2D labels slice; supports lazy (T, Y, X) volumes without full load."""
    try:
        shape = _volume_shape(data)
    except Exception:
        return None
    if len(shape) == 4:
        t = int(np.clip(int(frame_index), 0, shape[0] - 1))
        return np.asarray(data[t])
    if len(shape) == 3:
        # (T, H, W) label stack — take one time slice (not a single RGB image).
        t = int(np.clip(int(frame_index), 0, shape[0] - 1))
        return np.asarray(data[t])
    if len(shape) == 2:
        return np.asarray(data)
    return None


def n_frames(data: Any) -> int:
    shape = _volume_shape(data)
    if len(shape) == 4:
        return int(shape[0])
    if len(shape) == 3:
        # (H, W, C) image vs (T, H, W) label / mask volume
        if shape[-1] in (3, 4):
            return 1
        return int(shape[0])
    raise ValueError(f"Unsupported image shape: {shape}")


_VIDEO_EXTENSIONS = {".mp4", ".MP4", ".avi", ".AVI", ".mov", ".MOV", ".mkv", ".MKV"}


def sam2_decord_available() -> bool:
    """True when SAM2 can load ``.mp4`` paths directly (requires ``decord``)."""
    try:
        import decord  # noqa: F401
    except ImportError:
        return False
    return True


def video_path_for_sam2(data: Any) -> str | None:
    """Return a video file path when ``data`` is a lazy file-backed reader."""
    path = getattr(data, "path", None)
    if not path:
        return None
    if Path(str(path)).suffix in _VIDEO_EXTENSIONS:
        return str(path)
    return None


def propagation_progress(filled: set[int], frame_idx: int, total: int) -> tuple[int, int]:
    """Track unique frames filled; returns ``(filled_count, total_frames)``."""
    filled.add(int(frame_idx))
    return len(filled), int(total)


def conditioning_masks_from_labels(
    labels_data: Any,
    class_id: int,
    *,
    min_pixels: int = 1,
) -> list[tuple[int, np.ndarray]]:
    """Frames that already contain ``class_id`` in a label volume (extra SAM2 seeds)."""
    try:
        nf = n_frames(labels_data)
    except ValueError:
        return []
    out: list[tuple[int, np.ndarray]] = []
    cid = int(class_id)
    for t in range(nf):
        sl = labels_2d_at_frame(labels_data, t)
        if sl is None:
            continue
        mask = np.asarray(sl) == cid
        if int(np.count_nonzero(mask)) >= int(min_pixels):
            out.append((t, mask))
    return out


def _mask_from_labels_slice(labels_2d: np.ndarray, *, label_id: int | None = None) -> np.ndarray:
    s = np.asarray(labels_2d)
    if label_id is not None and label_id > 0:
        return s == int(label_id)
    if s.dtype == np.bool_:
        return s
    return s > 0


def _rasterize_shapes_polygons(
    shapes_data: list,
    *,
    frame_index: int,
    shape_hw: tuple[int, int],
) -> np.ndarray:
    h, w = shape_hw
    out = np.zeros((h, w), dtype=bool)
    for item in shapes_data:
        verts = np.asarray(item, dtype=float)
        if verts.size == 0:
            continue
        if verts.ndim == 1:
            continue
        if verts.shape[-1] >= 3:
            # (t, row, col) or (row, col, t)
            if verts.shape[-1] == 3:
                t_coord = verts[:, 0]
                if np.ptp(t_coord) < 1e-6 or int(round(float(np.median(t_coord)))) == frame_index:
                    rr = verts[:, 1]
                    cc = verts[:, 2]
                else:
                    continue
            else:
                rr = verts[:, -2]
                cc = verts[:, -1]
        elif verts.shape[-1] == 2:
            rr = verts[:, 0]
            cc = verts[:, 1]
        else:
            continue
        rr_i = np.clip(np.round(rr).astype(int), 0, h - 1)
        cc_i = np.clip(np.round(cc).astype(int), 0, w - 1)
        if rr_i.size < 3:
            continue
        pr, pc = sk_polygon(rr_i, cc_i, shape=(h, w))
        out[pr, pc] = True
    return out


def _binary_to_mask_input(mask_bool: np.ndarray, *, logit_pos: float = 10.0, logit_neg: float = -10.0) -> np.ndarray:
    """Approximate SAM2 low-res mask input from a binary (H,W) prompt."""
    m = mask_bool.astype(np.float32)
    logits = np.where(m > 0, logit_pos, logit_neg).astype(np.float32)
    # SAM2 expects 256x256 low-res mask (1, 256, 256) after predictor internal handling.
    low = cv2.resize(logits, (256, 256), interpolation=cv2.INTER_LINEAR)
    return low[None, :, :]


def _mask_to_box(mask_bool: np.ndarray) -> np.ndarray | None:
    ys, xs = np.where(mask_bool)
    if ys.size == 0:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    return np.array([x0, y0, x1, y1], dtype=np.float32)


def gather_prompts(
    *,
    frame_index: int,
    shape_hw: tuple[int, int],
    points_layer_data: Any | None,
    brush_labels_2d: np.ndarray | None,
    shapes_data: list | None,
    pipeline_mask_2d: np.ndarray | None,
    pipeline_label_id: int | None,
    ignore_time_filter: bool = False,
) -> dict:
    """Collect point coords, labels, optional mask/box prompts for one frame."""
    h, w = shape_hw
    coords: list[list[float]] = []
    labels: list[int] = []

    if points_layer_data is not None:
        pts = np.asarray(points_layer_data)
        if pts.size > 0:
            if pts.ndim == 1:
                pts = pts.reshape(1, -1)
            for row in pts:
                if row.size < 2:
                    continue
                y, x = float(row[-2]), float(row[-1])
                if row.size >= 3 and not ignore_time_filter:
                    t = int(round(float(row[0])))
                    if t != int(frame_index):
                        continue
                lab = 1
                if row.size >= 4:
                    lab = int(row[3])
                if 0 <= y < h and 0 <= x < w:
                    coords.append([x, y])  # SAM uses x,y
                    labels.append(1 if lab != 0 else 0)
                elif 0 <= x < h and 0 <= y < w:
                    # Napari sometimes stores (x, y) in the first two columns.
                    coords.append([y, x])
                    labels.append(1 if lab != 0 else 0)

    mask_bool = np.zeros((h, w), dtype=bool)
    if brush_labels_2d is not None:
        mask_bool |= _mask_from_labels_slice(brush_labels_2d)
    if shapes_data:
        mask_bool |= _rasterize_shapes_polygons(shapes_data, frame_index=frame_index, shape_hw=shape_hw)
    if pipeline_mask_2d is not None:
        pm_bool = _mask_from_labels_slice(
            np.asarray(pipeline_mask_2d), label_id=pipeline_label_id
        )
        mask_bool |= resize_mask_to_hw(pm_bool, h, w)

    point_coords = np.array(coords, dtype=np.float32) if coords else None
    point_labels = np.array(labels, dtype=np.int32) if labels else None
    box = _mask_to_box(mask_bool) if np.any(mask_bool) else None
    mask_input = _binary_to_mask_input(mask_bool) if np.any(mask_bool) else None

    return {
        "point_coords": point_coords,
        "point_labels": point_labels,
        "box": box,
        "mask_input": mask_input,
        "prompt_mask": mask_bool,
    }


def prompts_ready(prompts: dict) -> bool:
    """True if SAM2 has at least one point, box, or mask prompt."""
    coords = prompts.get("point_coords")
    if coords is not None and len(coords) > 0:
        return True
    if prompts.get("box") is not None:
        return True
    if prompts.get("mask_input") is not None:
        return True
    mask = prompts.get("prompt_mask")
    return mask is not None and bool(np.any(mask))


def summarize_prompts(prompts: dict) -> str:
    """One-line summary for the widget log."""
    parts: list[str] = []
    coords = prompts.get("point_coords")
    if coords is not None and len(coords) > 0:
        labels = prompts.get("point_labels")
        if labels is None:
            labels = []
        else:
            labels = np.asarray(labels).tolist()
        pos = sum(1 for lb in labels if int(lb) == 1)
        neg = len(labels) - pos
        parts.append(f"{len(coords)} point(s) (+{pos}/-{neg})")
    if prompts.get("box") is not None:
        parts.append("box")
    mask = prompts.get("prompt_mask")
    if mask is not None and np.any(mask):
        parts.append(f"mask ({int(np.count_nonzero(mask))} px)")
    if not parts:
        return "prompts: none"
    return "prompts: " + ", ".join(parts)


class Sam2Model:
    """Lazy-loaded SAM2 image + video predictors."""

    def __init__(
        self,
        *,
        config_path: str | None = None,
        checkpoint_path: str | None = None,
        device: str | None = None,
    ):
        backend = load_sam2_backend()
        if not backend["sam2_available"]:
            err = backend.get("import_error") or "unknown"
            raise RuntimeError(
                f"sam2 is not installed ({err}). "
                "From the plugin repo, in the same env as napari: uv sync --all-extras --group dev"
            )
        if not backend["torch_available"]:
            raise RuntimeError("PyTorch is required for SAM2.")

        self._backend = backend
        self.config_path = resolve_sam2_config_path(config_path)
        self.checkpoint_path = resolve_sam2_checkpoint(checkpoint_path)
        self.device = device or default_device()

        self._image_predictor: SAM2ImagePredictor | None = None
        self._video_predictor = None

    def warmup(self) -> None:
        """Load checkpoint weights into the image predictor (slow; call after init)."""
        self._build_image_predictor()

    def _build_image_predictor(self) -> SAM2ImagePredictor:
        if self._image_predictor is None:
            build_sam2 = self._backend["build_sam2"]
            SAM2ImagePredictor = self._backend["SAM2ImagePredictor"]
            model = build_sam2(self.config_path, self.checkpoint_path, device=self.device)
            self._image_predictor = SAM2ImagePredictor(model)
        return self._image_predictor

    def _build_video_predictor(self):
        if self._video_predictor is None:
            self._video_predictor = self._backend["build_sam2_video_predictor"](
                self.config_path, self.checkpoint_path, device=self.device
            )
        return self._video_predictor

    def predict_frame(
        self,
        frame_rgb: np.ndarray,
        prompts: dict,
        *,
        multimask_output: bool = True,
    ) -> tuple[np.ndarray, float]:
        """Return best (H,W) bool mask and score for one frame."""
        predictor = self._build_image_predictor()
        image = np.asarray(frame_rgb, dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError("frame_rgb must be HxWx3")

        predictor.set_image(image)

        point_coords = prompts.get("point_coords")
        point_labels = prompts.get("point_labels")
        box = prompts.get("box")
        mask_input = prompts.get("mask_input")

        if (
            point_coords is None
            and box is None
            and mask_input is None
        ):
            raise ValueError(
                "No prompts: add positive/negative points, brush/polygon labels, or a pipeline mask."
            )

        h, w = int(image.shape[0]), int(image.shape[1])
        # Pipeline / brush prompts: box alone is more reliable than low-res mask_input.
        use_mask_input = mask_input
        if (
            mask_input is not None
            and point_coords is not None
            and len(point_coords) > 0
        ):
            use_mask_input = mask_input
        elif mask_input is not None and box is not None:
            use_mask_input = None

        masks, scores, _logits = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box[None, :] if box is not None else None,
            mask_input=use_mask_input,
            multimask_output=multimask_output,
        )
        # masks: C x H x W at original resolution
        if masks.ndim == 3 and masks.shape[0] > 1:
            best = int(np.argmax(scores))
            mask_out = masks[best]
            score_out = float(scores[best])
        else:
            mask_out = masks[0] if masks.ndim == 3 else masks
            score_out = float(scores[0])
        mask_bool = resize_mask_to_hw(mask_out, h, w)
        return mask_bool, score_out

    def _write_video_frames_jpg(self, vid: np.ndarray, frame_dir: Path) -> None:
        T = int(vid.shape[0])
        for t in range(T):
            cv2.imwrite(
                str(frame_dir / f"{t:06d}.jpg"),
                cv2.cvtColor(vid[t], cv2.COLOR_RGB2BGR),
            )

    def _register_video_prompts(
        self,
        predictor,
        state,
        seed_frame: int,
        prompts: dict,
        obj_id: int,
        *,
        height: int,
        width: int,
        extra_conditioning: Sequence[tuple[int, np.ndarray]] | None = None,
    ) -> None:
        """Register seed prompts and any extra per-frame masks on a fresh inference state."""
        point_coords = prompts.get("point_coords")
        point_labels = prompts.get("point_labels")
        box = prompts.get("box")
        mask = prompts.get("prompt_mask")
        seed = int(seed_frame)

        if point_coords is not None and len(point_coords) > 0:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seed,
                obj_id=int(obj_id),
                points=point_coords,
                labels=point_labels,
            )
        elif mask is not None and np.any(mask):
            predictor.add_new_mask(
                inference_state=state,
                frame_idx=seed,
                obj_id=int(obj_id),
                mask=np.asarray(mask, dtype=bool),
            )
        elif box is not None:
            predictor.add_new_points_or_box(
                inference_state=state,
                frame_idx=seed,
                obj_id=int(obj_id),
                box=box,
            )
        else:
            raise ValueError("No prompts for video propagation.")

        for t, cond_mask in extra_conditioning or ():
            ti = int(t)
            if ti == seed:
                continue
            m2d = resize_mask_to_hw(np.asarray(cond_mask, dtype=bool), height, width)
            if np.any(m2d):
                predictor.add_new_mask(
                    inference_state=state,
                    frame_idx=ti,
                    obj_id=int(obj_id),
                    mask=m2d,
                )

    @staticmethod
    def _mask_plane_from_video_output(
        video_res_masks,
        obj_ids,
        obj_id: int,
        height: int,
        width: int,
        *,
        torch_module,
    ) -> np.ndarray:
        masks_t = video_res_masks
        if not isinstance(masks_t, torch_module.Tensor):
            masks_t = torch_module.as_tensor(masks_t)
        oid_list = list(obj_ids)
        if int(obj_id) in oid_list:
            oi = oid_list.index(int(obj_id))
        else:
            oi = 0
        m = masks_t[oi]
        if m.ndim == 3:
            m = m[0]
        return resize_mask_to_hw((m.detach().cpu().numpy() > 0.0), height, width)

    def propagate_video(
        self,
        video_source: np.ndarray | str,
        frame_index: int,
        prompts: dict,
        *,
        obj_id: int = 1,
        progress_callback=None,
        extra_conditioning: Sequence[tuple[int, np.ndarray]] | None = None,
    ) -> np.ndarray:
        """Propagate prompts through a video. Returns bool (T,H,W).

        ``video_source`` is either a file path (``.mp4``, etc.) or a (T,H,W,3) uint8 stack.
        Forward and reverse passes each use a fresh SAM2 state so both temporal directions
        track reliably; progress reports unique frames filled out of ``T`` (not 2×T steps).
        """
        th = self._backend["torch"]
        predictor = self._build_video_predictor()
        seed = int(frame_index)
        filled: set[int] = set()

        with tempfile.TemporaryDirectory(prefix="pecan_sam2_") as tmp:
            if isinstance(video_source, str):
                video_path = str(video_source)
                if not os.path.isfile(video_path):
                    raise FileNotFoundError(f"Video not found: {video_path}")
            else:
                vid = np.asarray(video_source, dtype=np.uint8)
                if vid.ndim != 4 or vid.shape[-1] != 3:
                    raise ValueError("video_source array must be (T,H,W,3)")
                frame_dir = Path(tmp) / "frames"
                frame_dir.mkdir()
                self._write_video_frames_jpg(vid, frame_dir)
                video_path = str(frame_dir)

            out: np.ndarray | None = None
            for reverse in (False, True):
                state = predictor.init_state(video_path=video_path)
                total = int(state["num_frames"])
                height = int(state["video_height"])
                width = int(state["video_width"])
                if out is None:
                    out = np.zeros((total, height, width), dtype=bool)
                elif out.shape[0] != total:
                    raise ValueError(
                        f"Frame count changed between passes ({out.shape[0]} vs {total})"
                    )

                self._register_video_prompts(
                    predictor,
                    state,
                    seed,
                    prompts,
                    obj_id,
                    height=height,
                    width=width,
                    extra_conditioning=extra_conditioning,
                )
                max_track = seed + 1 if reverse else total - seed
                with th.inference_mode():
                    for out_frame_idx, out_obj_ids, video_res_masks in predictor.propagate_in_video(
                        state,
                        start_frame_idx=seed,
                        reverse=reverse,
                        max_frame_num_to_track=max_track,
                    ):
                        plane = self._mask_plane_from_video_output(
                            video_res_masks,
                            out_obj_ids,
                            obj_id,
                            height,
                            width,
                            torch_module=th,
                        )
                        out[int(out_frame_idx)] |= plane
                        if progress_callback:
                            done, tot = propagation_progress(filled, int(out_frame_idx), total)
                            progress_callback(done, tot)

        if out is None:
            raise RuntimeError("Video propagation produced no output.")
        return out


def resize_mask_to_hw(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize a boolean or numeric mask to ``(height, width)``."""
    m = np.asarray(mask).squeeze()
    if m.ndim != 2:
        raise ValueError(f"mask must be 2D after squeeze, got shape {np.asarray(mask).shape}")
    if m.shape == (height, width):
        return (m > 0).astype(bool)
    m_u8 = (m > 0).astype(np.uint8)
    return cv2.resize(m_u8, (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def merge_class_into_labels(
    labels_volume: np.ndarray | None,
    class_mask: np.ndarray,
    class_id: int,
    *,
    frame_index: int | None = None,
) -> np.ndarray:
    """Write ``class_id`` where ``class_mask`` is True (clears that class on the touched slice)."""
    if labels_volume is None:
        m0 = np.asarray(class_mask).squeeze()
        if m0.ndim != 2:
            raise ValueError(f"class_mask must be 2D, got {np.asarray(class_mask).shape}")
        out = np.zeros(m0.shape, dtype=np.uint32)
    else:
        out = np.asarray(labels_volume, dtype=np.uint32).copy()

    if class_mask.ndim == 3 and out.ndim == 3:
        h, w = int(out.shape[1]), int(out.shape[2])
        t_len = min(int(class_mask.shape[0]), int(out.shape[0]))
        for t in range(t_len):
            mask2d = resize_mask_to_hw(class_mask[t], h, w)
            plane = out[t]
            plane[plane == class_id] = 0
            plane[mask2d] = int(class_id)
            out[t] = plane
        return out

    if out.ndim == 3:
        if frame_index is None:
            raise ValueError("frame_index is required when merging into a (T, H, W) labels layer")
        t = int(np.clip(int(frame_index), 0, out.shape[0] - 1))
        h, w = int(out.shape[1]), int(out.shape[2])
        mask2d = resize_mask_to_hw(class_mask, h, w)
        plane = out[t]
        plane[plane == class_id] = 0
        plane[mask2d] = int(class_id)
        out[t] = plane
        return out

    if out.ndim != 2:
        raise ValueError(f"Unsupported labels volume shape: {out.shape}")
    h, w = int(out.shape[0]), int(out.shape[1])
    mask2d = resize_mask_to_hw(class_mask, h, w)
    out[out == class_id] = 0
    out[mask2d] = int(class_id)
    return out
