"""Tests for hierarchical contrastive helpers."""

import numpy as np

from napari_pecan_py.widgets.contrastive_coding.data import contrastive_checkpoint_filename
from napari_pecan_py.widgets.contrastive_coding.hierarchy import (
    DEFAULT_HIERARCHY_CHAIN,
    TRAINING_MODE_HIERARCHICAL,
    hard_negative_classes,
    hierarchy_relation,
    soft_positive_classes,
)
from napari_pecan_py.widgets.contrastive_coding.hierarchical_sampling import (
    sample_hierarchical_triplets,
)


def test_hierarchy_relation_chain():
    assert hierarchy_relation("Kernel", "Crack", DEFAULT_HIERARCHY_CHAIN) == "ancestor"
    assert hierarchy_relation("Crack", "Kernel", DEFAULT_HIERARCHY_CHAIN) == "descendant"
    assert hierarchy_relation("Pecan", "Kernel", DEFAULT_HIERARCHY_CHAIN) == "descendant"


def test_soft_and_hard_classes_for_kernel():
    available = ["Pecan", "Crack", "Kernel", "background"]
    soft = soft_positive_classes("Kernel", available)
    assert set(soft) == {"Crack", "Pecan"}
    hard = hard_negative_classes("Kernel", available)
    assert hard == ["background"]


def test_hierarchical_triplet_sampling():
    h, w = 32, 32
    frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    masks = {
        "Pecan": np.ones((h, w), dtype=bool),
        "Crack": np.zeros((h, w), dtype=bool),
        "Kernel": np.zeros((h, w), dtype=bool),
    }
    masks["Crack"][8:24, 8:24] = True
    masks["Kernel"][12:20, 12:20] = True
    anc, pos, soft, neg, labels = sample_hierarchical_triplets(
        frame,
        masks,
        patch_size=8,
        patches_per_class=4,
        num_negatives=2,
    )
    assert anc.shape[0] > 0
    assert pos.shape == anc.shape
    assert neg.shape[0] == anc.shape[0]
    assert "Kernel" in labels


def test_hierarchical_checkpoint_filename():
    name = contrastive_checkpoint_filename(
        ["Pecan", "Crack"],
        training_mode=TRAINING_MODE_HIERARCHICAL,
    )
    assert name.startswith("contrastive-hier")
