"""Tests for cascaded segmentation helpers."""

from pathlib import Path

import numpy as np
import pytest

from napari_pecan_py.widgets.cascade_seg.hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    LABEL_ID_BY_CLASS,
    ordered_chain_classes,
    parent_mask_names,
    stage_input_channels,
)
from napari_pecan_py.widgets.cascade_seg.model import (
    ARCH_UNETPP,
    BACKEND_CASCADE,
    detect_seg_checkpoint_backend,
    merge_stage_masks_to_label_map,
)


def test_format_hierarchy_tree_shows_siblings():
    from napari_pecan_py.widgets.cascade_seg.hierarchy import format_hierarchy_tree

    text = format_hierarchy_tree(("Pecan", "Crack", "Kernel"))
    assert "Pecan ⊃ [Crack, Kernel]" in text
    assert "Crack ⊃ Kernel" not in text


def test_hierarchy_chain_is_pecan_crack_kernel():
    assert DEFAULT_HIERARCHY_CHAIN == ("Pecan", "Crack", "Kernel")


def test_parent_mask_names_kernel_uses_pecan_only():
    assert parent_mask_names("Kernel") == ["Pecan"]
    assert parent_mask_names("Crack") == ["Pecan"]


def test_stage_input_channels_follow_parents():
    assert stage_input_channels("Pecan") == 3
    assert stage_input_channels("Crack") == 4
    assert stage_input_channels("Kernel") == 4


def test_ordered_chain_classes():
    selected = ["Kernel", "Pecan", "Crack"]
    assert ordered_chain_classes(selected) == ["Pecan", "Crack", "Kernel"]


def test_nut_region_includes_kernel_outside_shell():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.hierarchy import nut_region_mask_tensor

    pecan = torch.zeros(1, 1, 10, 10)
    pecan[:, :, :, :2] = 1.0
    pecan[:, :, :, 8:] = 1.0
    kernel = torch.zeros(1, 1, 10, 10)
    kernel[:, :, 3:7, 3:7] = 1.0
    region = nut_region_mask_tensor({"Pecan": pecan, "Kernel": kernel}, hard=True)
    assert float(region[:, :, 5, 5]) == 1.0


def test_exclusive_part_masks():
    from napari_pecan_py.widgets.cascade_seg.model import _exclusive_part_masks

    masks = {
        "Pecan": np.ones((4, 4), dtype=np.uint8),
        "Crack": np.zeros((4, 4), dtype=np.uint8),
        "Kernel": np.zeros((4, 4), dtype=np.uint8),
    }
    masks["Kernel"][1:3, 1:3] = 1
    out = _exclusive_part_masks(masks)
    assert int(out["Kernel"].sum()) == 4
    assert int(out["Pecan"].sum()) == 12


def test_merge_kernel_survives_when_not_inside_shell_pecan():
    """Pecan = shell lobes only; kernel is interior and must not be gated by shell."""
    h, w = 100, 100
    pecan = np.zeros((h, w), dtype=np.uint8)
    pecan[:, :18] = 1
    pecan[:, 82:] = 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    kernel[30:70, 20:80] = 1
    crack = np.zeros((h, w), dtype=np.uint8)
    crack[28:34, 15:85] = 1

    label_map = merge_stage_masks_to_label_map(
        {"Pecan": pecan, "Crack": crack, "Kernel": kernel}
    )
    assert label_map[50, 50] == LABEL_ID_BY_CLASS["Kernel"]
    assert label_map[29, 50] == LABEL_ID_BY_CLASS["Crack"]
    assert label_map[50, 10] == LABEL_ID_BY_CLASS["Pecan"]


def test_merge_stage_masks_inner_classes_win():
    h, w = 8, 8
    pecan = np.zeros((h, w), dtype=np.uint8)
    pecan[1:7, 1:7] = 1
    crack = np.zeros((h, w), dtype=np.uint8)
    crack[1:3, 1:7] = 1
    kernel = np.zeros((h, w), dtype=np.uint8)
    kernel[4:6, 2:6] = 1

    label_map = merge_stage_masks_to_label_map(
        {"Pecan": pecan, "Crack": crack, "Kernel": kernel}
    )
    assert label_map[2, 2] == LABEL_ID_BY_CLASS["Crack"]
    assert label_map[5, 3] == LABEL_ID_BY_CLASS["Kernel"]
    assert label_map[6, 6] == LABEL_ID_BY_CLASS["Pecan"]


def test_frame_has_all_class_labels():
    from napari_pecan_py.widgets.cascade_seg.model import (
        filter_samples_require_all_classes,
        frame_has_all_class_labels,
    )

    masks = {
        "Pecan": np.array([[1, 1], [1, 1]], dtype=np.uint8),
        "Crack": np.array([[0, 1], [0, 0]], dtype=np.uint8),
        "Kernel": np.array([[0, 0], [1, 0]], dtype=np.uint8),
    }
    assert frame_has_all_class_labels(masks, ["Pecan", "Crack", "Kernel"])
    masks_no_crack = dict(masks)
    masks_no_crack["Crack"] = np.zeros((2, 2), dtype=np.uint8)
    assert not frame_has_all_class_labels(masks_no_crack, ["Pecan", "Crack", "Kernel"])

    samples = [
        (
            0,
            np.zeros((4, 4, 3), dtype=np.uint8),
            {
                "Pecan": np.ones((2, 2), dtype=np.uint8),
                "Crack": np.zeros((2, 2), dtype=np.uint8),
                "Kernel": np.zeros((2, 2), dtype=np.uint8),
            },
        ),
        (
            0,
            np.zeros((4, 4, 3), dtype=np.uint8),
            {
                "Pecan": np.ones((2, 2), dtype=np.uint8),
                "Crack": np.ones((2, 2), dtype=np.uint8),
                "Kernel": np.ones((2, 2), dtype=np.uint8),
            },
        ),
    ]
    kept = filter_samples_require_all_classes(samples, ["Pecan", "Crack", "Kernel"])
    assert len(kept) == 1


def test_absence_loss_on_empty_kernel_target():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.model import _compute_stage_losses

    logits = {"Kernel": torch.ones(1, 1, 8, 8) * 3.0}
    target_tensors = {
        "Pecan": torch.zeros(1, 1, 8, 8),
        "Kernel": torch.zeros(1, 1, 8, 8),
    }
    target_tensors["Pecan"][:, :, :, :3] = 1.0
    context = {k: v for k, v in target_tensors.items()}
    _, losses = _compute_stage_losses(
        logits,
        target_tensors,
        context,
        "cpu",
        train_absent_inner_classes=True,
    )
    assert "Kernel" in losses
    assert losses["Kernel"] > 0.1


def test_absence_loss_skipped_when_crack_present_without_kernel():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.model import _compute_stage_losses

    logits = {"Crack": torch.ones(1, 1, 8, 8), "Kernel": torch.ones(1, 1, 8, 8) * 3.0}
    target_tensors = {
        "Pecan": torch.zeros(1, 1, 8, 8),
        "Crack": torch.zeros(1, 1, 8, 8),
        "Kernel": torch.zeros(1, 1, 8, 8),
    }
    target_tensors["Pecan"][:, :, :, :4] = 1.0
    target_tensors["Crack"][:, :, 2:6, 2:6] = 1.0
    context = {k: v for k, v in target_tensors.items()}
    _, losses = _compute_stage_losses(
        logits,
        target_tensors,
        context,
        "cpu",
        train_absent_inner_classes=True,
    )
    assert "Crack" in losses
    assert "Kernel" not in losses


def test_absence_loss_skipped_when_disabled():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.model import _compute_stage_losses

    logits = {"Kernel": torch.ones(1, 1, 8, 8)}
    target_tensors = {"Kernel": torch.zeros(1, 1, 8, 8)}
    _, losses = _compute_stage_losses(
        logits,
        target_tensors,
        target_tensors,
        "cpu",
        train_absent_inner_classes=False,
    )
    assert "Kernel" not in losses


def test_stage_loss_weights_favor_inner_classes():
    from napari_pecan_py.widgets.cascade_seg.hierarchy import (
        STAGE_INFERENCE_THRESHOLD,
        STAGE_LOSS_WEIGHT,
        STAGE_POS_WEIGHT,
    )

    assert STAGE_LOSS_WEIGHT["Crack"] > STAGE_LOSS_WEIGHT["Pecan"]
    assert STAGE_INFERENCE_THRESHOLD["Crack"] < STAGE_INFERENCE_THRESHOLD["Pecan"]
    assert STAGE_POS_WEIGHT["Crack"] > STAGE_POS_WEIGHT["Pecan"]


def test_dice_bce_penalizes_logits_outside_parent_region():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.model import _dice_bce_loss

    # Target crack only inside a small pecan region; logits high everywhere.
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    region = torch.zeros(1, 1, 8, 8)
    region[:, :, 1:7, 1:7] = 1.0
    logits_high_outside = torch.ones(1, 1, 8, 8) * 4.0
    logits_low = torch.ones(1, 1, 8, 8) * -4.0

    loss_outside = float(_dice_bce_loss(logits_high_outside, target, region).detach())
    loss_low = float(_dice_bce_loss(logits_low, target, region).detach())
    assert loss_outside > loss_low


def test_resize_mask_dilates_thin_crack_on_downscale():
    from napari_pecan_py.widgets.cascade_seg.model import _resize_mask

    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 20:44] = 1
    small = _resize_mask(mask, 32, class_name="Crack")
    assert int(small.sum()) >= 1


def test_prepare_batch_tensors_fills_missing_classes():
    from napari_pecan_py.widgets.cascade_seg.model import _collate_frame_samples, _prepare_batch_tensors

    batch = _collate_frame_samples(
        [
            (
                np.zeros((64, 64, 3), dtype=np.uint8),
                {
                    "Pecan": np.ones((64, 64), dtype=np.uint8),
                    "Crack": np.zeros((64, 64), dtype=np.uint8),
                },
            ),
            (
                np.zeros((64, 64, 3), dtype=np.uint8),
                {"Pecan": np.ones((64, 64), dtype=np.uint8)},
            ),
        ]
    )
    _, target_tensors, class_names = _prepare_batch_tensors(batch, "cpu", 32)
    assert class_names == ["Crack", "Pecan"]
    assert int(target_tensors["Crack"][1].sum()) == 0


def test_prepare_batch_tensors_variable_frame_sizes():
    from napari_pecan_py.widgets.cascade_seg.model import _collate_frame_samples, _prepare_batch_tensors

    batch = _collate_frame_samples(
        [
            (
                np.zeros((244, 472, 3), dtype=np.uint8),
                {"Pecan": np.ones((244, 472), dtype=np.uint8)},
            ),
            (
                np.zeros((252, 484, 3), dtype=np.uint8),
                {"Pecan": np.ones((252, 484), dtype=np.uint8)},
            ),
        ]
    )
    image_t, target_tensors, class_names = _prepare_batch_tensors(batch, "cpu", 512)
    assert image_t.shape == (2, 3, 512, 512)
    assert target_tensors["Pecan"].shape == (2, 1, 512, 512)
    assert class_names == ["Pecan"]


def test_detect_cascade_checkpoint(tmp_path: Path):
    torch = pytest.importorskip("torch")

    ckpt = tmp_path / "model.pt"
    torch.save({"backend": BACKEND_CASCADE, "trained_stages": ["Pecan"]}, ckpt)
    assert detect_seg_checkpoint_backend(ckpt) == BACKEND_CASCADE

    yolo_like = tmp_path / "yolo.pt"
    torch.save({"model": "not-cascade"}, yolo_like)
    assert detect_seg_checkpoint_backend(yolo_like) == "yolo"


@pytest.mark.skipif(
    pytest.importorskip("segmentation_models_pytorch", reason="smp not installed") is None,
    reason="segmentation_models_pytorch not installed",
)
def test_cascaded_segmenter_forward():
    torch = pytest.importorskip("torch")
    from napari_pecan_py.widgets.cascade_seg.model import CascadedSegmenter

    model = CascadedSegmenter(
        ["Pecan", "Crack"],
        encoder_name="mobilenet_v2",
        architecture=ARCH_UNETPP,
    )
    x = torch.rand(1, 3, 64, 64)
    context = {"Pecan": (torch.rand(1, 1, 64, 64) > 0.5).float()}
    logits = model.forward_tensors(x, context_masks=context, teacher_forcing=True)
    assert set(logits.keys()) == {"Pecan", "Crack"}
    assert logits["Pecan"].shape == (1, 1, 64, 64)
