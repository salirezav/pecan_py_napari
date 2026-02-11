"""Napari plugin for Pecan Py."""

try:
    from napari_pecan_py._version import __version__
except ImportError:
    __version__ = "0.0.1"

__all__ = ["__version__"]
