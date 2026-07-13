"""Spatial class hierarchy for cascaded segmentation.



Label semantics (important):

  - **Pecan** = shell surface (not the whole nut bounding box).

  - **Crack** and **Kernel** = sibling interior parts (fracture zone, nut meat).

  - They are mutually exclusive pixel classes, not nested inside the shell mask.



Cascade tree (conditioning order):

  Pecan → Crack, Kernel (siblings)

"""



from __future__ import annotations



from typing import Dict, Sequence, Tuple



# Stage run order (coarse first, then siblings).

DEFAULT_HIERARCHY_CHAIN: Tuple[str, ...] = (

    "Pecan",

    "Crack",

    "Kernel",

)



# Classes offered for training / cascade stages in the Segmentation widget.

TRAINABLE_SEG_CLASSES: Tuple[str, ...] = DEFAULT_HIERARCHY_CHAIN



# Used for loss masking and (dilated) cascade conditioning — not the shell class alone.

NUT_REGION_CLASSES: Tuple[str, ...] = ("Pecan", "Crack", "Kernel")



# Legacy parent map (Damaged Kernel kept only for old label-map TIFF id 4).

PARENT_REGION: Dict[str, str | None] = {

    "Pecan": None,

    "Crack": "Pecan",

    "Kernel": "Pecan",

}



# Combined label-map pixel values (compatible with YOLO / contrastive conventions).

LABEL_ID_BY_CLASS: Dict[str, int] = {

    "Crack": 1,

    "Kernel": 2,

    "Pecan": 3,

    "Damaged Kernel": 4,  # legacy id only — not trained in this widget

}



CLASS_BY_LABEL_ID: Dict[int, str] = {v: k for k, v in LABEL_ID_BY_CLASS.items()}



STAGE_LOSS_WEIGHT: Dict[str, float] = {

    "Pecan": 1.0,

    "Crack": 5.0,

    "Kernel": 8.0,

}



STAGE_INFERENCE_THRESHOLD: Dict[str, float] = {

    "Pecan": 0.55,

    "Crack": 0.25,

    "Kernel": 0.35,

}



STAGE_POS_WEIGHT: Dict[str, float] = {

    "Pecan": 1.0,

    "Crack": 15.0,

    "Kernel": 20.0,

}



STAGE_ABSENCE_LOSS_WEIGHT: Dict[str, float] = {

    "Crack": 0.4,

    "Kernel": 0.4,

}



LABEL_MERGE_PRIORITY: Tuple[str, ...] = (

    "Pecan",

    "Crack",

    "Kernel",

)





def ordered_chain_classes(

    selected: Sequence[str],

    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,

) -> list[str]:

    """Return hierarchy classes present in ``selected``, in stage run order."""

    selected_set = set(selected)

    return [name for name in chain if name in selected_set]





def parent_mask_names(

    class_name: str,

    chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN,

) -> list[str]:

    """Mask channels fed into a stage (ancestors used as spatial context)."""

    del chain

    names: list[str] = []

    current = PARENT_REGION.get(class_name)

    while current is not None:

        names.insert(0, current)

        current = PARENT_REGION.get(current)

    return names





def stage_input_channels(class_name: str) -> int:

    """RGB + one channel per parent mask."""

    return 3 + len(parent_mask_names(class_name))





def format_hierarchy_chain(chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN) -> str:

    """Human-readable spatial tree (not a linear Pecan⊃Crack⊃Kernel chain)."""

    return format_hierarchy_tree(chain)





def format_hierarchy_tree(chain: Sequence[str] = DEFAULT_HIERARCHY_CHAIN) -> str:

    present = set(chain)

    parts: list[str] = []

    if "Pecan" in present:

        siblings = [c for c in ("Crack", "Kernel") if c in present]

        if siblings:

            parts.append(f"Pecan ⊃ [{', '.join(siblings)}]")

        else:

            parts.append("Pecan")

    return "; ".join(parts) if parts else " ⊃ ".join(chain)





def nut_region_mask_tensor(masks, *, hard: bool = False, dilate: int = 0):

    """Union of pecan / crack / kernel masks (the whole nut), batched (B,1,H,W)."""

    import torch

    import torch.nn.functional as F



    region = None

    for cls in NUT_REGION_CLASSES:

        if cls not in masks:

            continue

        m = masks[cls]

        if hard:

            m = (m > 0.5).float()

        else:

            m = m.float()

        region = m if region is None else torch.maximum(region, m)

    if region is None:

        raise ValueError("Cannot build nut region: no pecan/crack/kernel masks provided.")

    if dilate > 0:

        k = dilate * 2 + 1

        region = F.max_pool2d(region, kernel_size=k, stride=1, padding=dilate)

        if hard:

            region = (region > 0.5).float()

    return region


