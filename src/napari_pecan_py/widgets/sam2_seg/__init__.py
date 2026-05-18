"""SAM 2 interactive segmentation (widget in ``.widget``)."""

__all__ = ["Sam2SegWidget"]


def __getattr__(name: str):
    if name == "Sam2SegWidget":
        from .widget import Sam2SegWidget

        return Sam2SegWidget
    raise AttributeError(name)
