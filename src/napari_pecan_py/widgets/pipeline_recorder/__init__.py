"""Pipeline recorder package (widget in ``.widget`` for lighter imports of ``.logic``)."""

__all__ = ["PipelineRecorderWidget"]


def __getattr__(name: str):
    if name == "PipelineRecorderWidget":
        from .widget import PipelineRecorderWidget

        return PipelineRecorderWidget
    raise AttributeError(name)
