"""Class hierarchy helpers for hierarchical contrastive training.

Chain (coarse → fine): Pecan ⊃ Crack ⊃ Kernel
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

# Coarse (outer) to fine (inner).
DEFAULT_HIERARCHY_CHAIN: Tuple[str, ...] = ("Pecan", "Crack", "Kernel")

TRAINING_MODE_SIMPLE = "simple"
TRAINING_MODE_HIERARCHICAL = "hierarchical"


def hierarchy_index(class_name: str, chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN) -> int | None:
    try:
        return list(chain).index(class_name)
    except ValueError:
        return None


def hierarchy_distance(
    class_a: str,
    class_b: str,
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> int:
    """Return |level(a) - level(b)| for chain classes, else a large distance."""
    ia = hierarchy_index(class_a, chain)
    ib = hierarchy_index(class_b, chain)
    if ia is None or ib is None:
        return 10_000
    return abs(ia - ib)


def hierarchy_relation(
    anchor_class: str,
    other_class: str,
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> str:
    """One of: same, ancestor, descendant, unrelated."""
    if anchor_class == other_class:
        return "same"
    ia = hierarchy_index(anchor_class, chain)
    ib = hierarchy_index(other_class, chain)
    if ia is None or ib is None:
        return "unrelated"
    if ib < ia:
        return "ancestor"
    if ib > ia:
        return "descendant"
    return "unrelated"


def soft_positive_classes(
    anchor_class: str,
    available: Sequence[str],
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> List[str]:
    """Ancestor and descendant classes present on this frame."""
    rels = ("ancestor", "descendant")
    return [
        c
        for c in available
        if c != anchor_class and hierarchy_relation(anchor_class, c, chain) in rels
    ]


def hard_negative_classes(
    anchor_class: str,
    available: Sequence[str],
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> List[str]:
    """Classes to use as hard negatives (background + unrelated labels)."""
    out: List[str] = []
    for c in available:
        if c == anchor_class:
            continue
        rel = hierarchy_relation(anchor_class, c, chain)
        if rel in ("unrelated",) or c == "background":
            out.append(c)
    return out


def negative_loss_weight(
    anchor_class: str,
    negative_class: str,
    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,
) -> float:
    """Higher weight → stronger repulsion in weighted contrastive loss."""
    rel = hierarchy_relation(anchor_class, negative_class, chain)
    if rel == "unrelated" or negative_class == "background":
        return 1.0
    # Chain neighbours should not appear as negatives; guard with low weight.
    dist = hierarchy_distance(anchor_class, negative_class, chain)
    if dist <= 1:
        return 0.15
    if dist == 2:
        return 0.35
    return 1.0


def format_hierarchy_chain(chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN) -> str:
    return " ⊃ ".join(chain)
