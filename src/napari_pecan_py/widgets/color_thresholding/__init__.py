"""Color Tuner: tune color thresholds (RGB/HSV/LAB) for pecan targets as a napari dock widget."""

from .widget import ColorTunerWidget

__all__ = ["ColorTunerWidget", "color_tuner_widget"]


def color_tuner_widget(viewer):
    """Return the Color Tuner Qt widget for the given napari viewer."""
    return ColorTunerWidget(viewer)
