"""Tests for batch pipeline logic."""

from pathlib import Path

import numpy as np
import pytest

from napari_pecan_py.widgets.batch_pipeline.logic import (
    HeadlessViewer,
    load_pipeline_file,
)
from napari_pecan_py.widgets.pipeline_recorder.logic import (
    apply_pipeline_step_with_context,
    create_apply_context,
)


def test_load_pipeline_file_reads_enabled_steps():
    path = Path(__file__).resolve().parents[1] / "pipelines" / "01 - detects pecan adds ellipse.yml"
    if not path.exists():
        pytest.skip("example pipeline file not present")
    steps, name = load_pipeline_file(path)
    assert name == path.name
    assert len(steps) == 3
    assert steps[0]["kind"] == "color_adjustments.stack"
    assert all(step.get("enabled", True) for step in steps)


def test_load_pipeline_file_rejects_empty(tmp_path):
    p = tmp_path / "empty.yml"
    p.write_text("version: 1\nsteps: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no enabled steps"):
        load_pipeline_file(p)


def test_headless_viewer_runs_pipeline_without_napari_viewer():
    viewer = HeadlessViewer()
    data = np.zeros((2, 8, 8, 3), dtype=np.uint8)
    data[:, 2:6, 2:6, :] = 200
    viewer.add_image(data, name="clip", metadata={"source_path": "/tmp/clip.mp4"})
    ctx = create_apply_context(viewer)
    step = {
        "kind": "color_thresholding.threshold",
        "params": {
            "source_layer": "clip",
            "target": "pecan",
            "color_space": "rgb",
            "lower": [0, 0, 0],
            "upper": [255, 255, 255],
            "output_mask_layer": "clip - Pecan",
        },
    }
    msg = apply_pipeline_step_with_context(ctx, step)
    assert "clip - Pecan" in msg
    assert "clip - Pecan" in viewer.layers
    assert viewer.layers["clip - Pecan"].data.shape == (2, 8, 8)
