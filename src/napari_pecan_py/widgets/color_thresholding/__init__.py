"""Color Thresholding package (widget lives in ``.widget`` for headless imports of ``.logic``)."""

__all__ = ["ColorThresholdingWidget", "color_thresholding_widget"]


def __getattr__(name: str):
    if name == "ColorThresholdingWidget":
        from .widget import ColorThresholdingWidget

        return ColorThresholdingWidget
    if name == "color_thresholding_widget":
        from .widget import ColorThresholdingWidget

        def color_thresholding_widget(viewer):
            return ColorThresholdingWidget(viewer)

        return color_thresholding_widget
    raise AttributeError(name)
