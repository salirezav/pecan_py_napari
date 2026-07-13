"""Segmentation package (YOLO + cascaded U-Net in ``.widget``)."""

__all__ = ["SegmentationWidget", "YoloSegWidget"]


def __getattr__(name: str):
    if name in ("SegmentationWidget", "YoloSegWidget"):
        from .widget import SegmentationWidget

        return SegmentationWidget
    raise AttributeError(name)
