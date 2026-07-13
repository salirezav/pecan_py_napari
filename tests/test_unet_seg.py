"""Tests for flat U-Net segmentation."""

from pathlib import Path

import numpy as np
import pytest

from napari_pecan_py.widgets.cascade_seg.hierarchy import LABEL_ID_BY_CLASS
from napari_pecan_py.widgets.cascade_seg.model import (
    detect_seg_checkpoint_backend,
    merge_stage_masks_to_label_map,
)
from napari_pecan_py.widgets.unet_seg.model import (
    BACKEND_UNET,
    ARCH_UNETPP,
    FlatSegmenter,
)


def test_detect_unet_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")

    ckpt = tmp_path / "unet.pt"
    torch.save({"backend": BACKEND_UNET, "class_names": ["Pecan", "Crack"]}, ckpt)
    assert detect_seg_checkpoint_backend(ckpt) == BACKEND_UNET


@pytest.mark.skipif(
    pytest.importorskip("segmentation_models_pytorch", reason="smp not installed") is None,
    reason="segmentation_models_pytorch not installed",
)
def test_flat_segmenter_forward():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.unet_seg.model import FlatSegmenter

    model = FlatSegmenter(
        ["Pecan", "Crack", "Kernel"],
        encoder_name="mobilenet_v2",
        architecture=ARCH_UNETPP,
    )
    x = torch.rand(2, 3, 64, 64)
    logits = model.forward_logits(x)
    assert logits.shape == (2, 3, 64, 64)
    probs = model.predict_probs(x)
    assert probs.shape == (2, 3, 64, 64)


def test_flat_inference_merge_priority():
    h, w = 8, 8
    pecan = np.zeros((h, w), dtype=np.uint8)
    pecan[1:7, 1:7] = 1
    crack = np.zeros((h, w), dtype=np.uint8)
    crack[2, 2:6] = 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    kernel[4:6, 2:6] = 1

    label_map = merge_stage_masks_to_label_map(
        {"Pecan": pecan, "Crack": crack, "Kernel": kernel}
    )
    assert label_map[2, 3] == LABEL_ID_BY_CLASS["Crack"]
    assert label_map[5, 3] == LABEL_ID_BY_CLASS["Kernel"]
