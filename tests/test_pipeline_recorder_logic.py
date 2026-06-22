"""Tests for pipeline recorder name rebasing and layer steps."""

import numpy as np
import pytest

from napari_pecan_py.widgets.batch_pipeline.logic import HeadlessViewer
from napari_pecan_py.widgets.pipeline_recorder.logic import (
    apply_pipeline_step_with_context,
    create_apply_context,
)
from napari_pecan_py.widgets.pipeline_recorder.state import infer_recorded_root


def test_infer_recorded_root_from_pipeline_steps():
    steps = [
        {
            "kind": "edge_detection.apply",
            "params": {
                "source_layer": "GH012976-cropped",
                "output_layer": "GH012976-cropped - Edges (Canny)",
            },
        },
        {
            "kind": "mask_ops.operation",
            "params": {
                "a_layer": "GH012976-cropped - Edges (Canny)",
                "b_layer": "GH012976-cropped - Pecan",
                "output_layer": "GH012976-cropped - Edges (Canny) A-B GH012976-cropped - Pecan",
            },
        },
    ]
    assert infer_recorded_root(steps) == "GH012976-cropped"


def test_pipeline_rebases_layer_names_for_different_video():
    viewer = HeadlessViewer()
    data = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    data[:, 2:6, 2:6, :] = 200
    viewer.add_image(data, name="new-video", metadata={"source_path": "/tmp/new-video.mp4"})
    viewer.add_labels(np.ones((2, 8, 8), dtype=np.uint8), name="new-video - Pecan")

    ctx = create_apply_context(
        viewer,
        steps=[],
        recorded_root="GH012976-cropped",
    )
    step = {
        "kind": "mask_ops.operation",
        "params": {
            "mode": "binary",
            "a_layer": "GH012976-cropped - Edges (Canny)",
            "b_layer": "GH012976-cropped - Pecan",
            "op": "and",
            "target": "new",
            "output_layer": "GH012976-cropped - Edges (Canny) AND GH012976-cropped - Pecan",
        },
    }
    viewer.add_image(data, name="new-video - Edges (Canny)")
    msg = apply_pipeline_step_with_context(ctx, step)
    assert "new-video" in msg
    assert any("new-video - Edges (Canny) AND new-video - Pecan" == layer.name for layer in viewer.layers)


def test_pipeline_duplicate_layer_step():
    viewer = HeadlessViewer()
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    viewer.add_image(data, name="clip", metadata={"source_path": "/tmp/clip.mp4"})

    ctx = create_apply_context(viewer, recorded_root="clip")
    step = {
        "kind": "layer.duplicate",
        "params": {
            "source_layer": "clip",
            "output_layer": "clip copy",
        },
    }
    msg = apply_pipeline_step_with_context(ctx, step)
    assert "clip copy" in msg
    assert "clip copy" in viewer.layers


def test_pipeline_duplicate_rebases_for_different_video():
    viewer = HeadlessViewer()
    data = np.zeros((4, 4, 3), dtype=np.uint8)
    viewer.add_image(data, name="other", metadata={"source_path": "/tmp/other.mp4"})

    ctx = create_apply_context(viewer, recorded_root="clip")
    step = {
        "kind": "layer.duplicate",
        "params": {
            "source_layer": "clip",
            "output_layer": "clip copy",
        },
    }
    msg = apply_pipeline_step_with_context(ctx, step)
    assert "other copy" in msg
    assert "other copy" in viewer.layers


def test_pipeline_yolo_seg_rebases_mask_layer_for_different_video(monkeypatch):
    viewer = HeadlessViewer()
    data = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    viewer.add_image(data, name="new-video", metadata={"source_path": "/tmp/new-video.mp4"})

    def _fake_inference(weights_path, frames, device, *, progress_callback=None, cancel_callback=None):
        del weights_path, device, progress_callback, cancel_callback
        if frames.ndim == 4:
            return np.zeros((frames.shape[0], frames.shape[1], frames.shape[2]), dtype=np.uint8)
        return np.zeros((frames.shape[0], frames.shape[1]), dtype=np.uint8)

    monkeypatch.setattr(
        "napari_pecan_py.widgets.yolo_seg.model.run_yolo_seg_inference_on_frames",
        _fake_inference,
    )

    ctx = create_apply_context(viewer, recorded_root="GH012976-cropped")
    step = {
        "kind": "yolo_seg.inference",
        "params": {
            "source_layer": "GH012976-cropped",
            "weights_path": __file__,
            "device": "cpu",
            "save_masks": False,
            "save_suffix": " - Crack",
            "save_fmt": "tiff",
            "output_mask_layer": "GH012976-cropped - Crack",
        },
    }
    msg = apply_pipeline_step_with_context(ctx, step)
    assert "new-video - Crack" in msg
    assert "new-video - Crack" in viewer.layers
