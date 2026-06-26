"""Tests for contrastive coding dataset helpers."""

from pathlib import Path

import numpy as np

from napari_pecan_py.widgets.contrastive_coding.data import (
    COMBINED_MASK_STEM_SUFFIX,
    contrastive_checkpoint_filename,
    discover_label_values_in_mask,
    discover_multilabel_mask,
    label_values_to_names,
    multilabel_frame_to_class_masks,
)


def test_discover_multilabel_mask_primary_suffix(tmp_path: Path):
    import tifffile

    video = tmp_path / "clip.mp4"
    video.write_bytes(b"")
    tiff_mask = tmp_path / f"clip{COMBINED_MASK_STEM_SUFFIX}.tiff"
    tifffile.imwrite(tiff_mask, np.zeros((2, 4, 4), dtype=np.uint8))

    found = discover_multilabel_mask(video)
    assert found == tiff_mask.resolve()


def test_label_values_to_names_and_masks():
    labels = np.array(
        [
            [0, 1, 0],
            [2, 3, 2],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    values = discover_label_values_in_mask(labels)
    assert values == [1, 2, 3]
    names = label_values_to_names(values)
    assert names[1] == "Crack"
    assert names[2] == "Kernel"
    assert names[3] == "Pecan"
    masks = multilabel_frame_to_class_masks(labels, {v: k for k, v in names.items()})
    assert masks["Crack"].sum() == 2
    assert masks["Kernel"].sum() == 2
    assert masks["Pecan"].sum() == 1


def test_contrastive_checkpoint_filename():
    assert contrastive_checkpoint_filename(["Pecan", "Crack", "Kernel"]) == (
        "contrastive - [Crack, Kernel, Pecan].pt"
    )
    assert contrastive_checkpoint_filename(
        ["Pecan"],
        training_mode="hierarchical",
    ).startswith("contrastive-hier")


def test_sync_layer_geometry_matches_video_rgb_to_labels():
    import napari

    from napari_pecan_py.widgets.contrastive_coding.widget import (
        _sync_layer_geometry_to_image,
    )

    viewer = napari.Viewer(show=False)
    try:
        img = viewer.add_image(np.zeros((5, 10, 12, 3), dtype=np.uint8))
        labels = viewer.add_labels(np.zeros((5, 10, 12), dtype=np.uint8))
        img.scale = (1, 2, 3, 4)
        img.translate = (10, 20, 30, 40)
        _sync_layer_geometry_to_image(labels, img)
        assert tuple(labels.scale) == (1, 2, 3)
        assert tuple(labels.translate) == (10, 20, 30)
        labels.get_status((0, 5, 5))
    finally:
        viewer.close()

