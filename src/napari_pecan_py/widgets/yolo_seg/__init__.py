"""YOLO Segmentation package (widget in ``.widget`` for lighter imports of ``.model``)."""

__all__ = ["YoloSegWidget"]


def __getattr__(name: str):
    if name == "YoloSegWidget":
        from .widget import YoloSegWidget

        return YoloSegWidget
    raise AttributeError(name)
