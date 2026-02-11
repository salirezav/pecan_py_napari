"""Minimal napari widget for napari-pecan-py."""

from napari.utils.notifications import show_info


def hello_widget() -> None:
    """Show a hello message in napari."""
    show_info("Hello from napari-pecan-py!")
