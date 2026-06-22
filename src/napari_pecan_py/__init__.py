"""Napari plugin for Pecan Py."""

try:
    from napari_pecan_py._version import __version__
except ImportError:
    __version__ = "0.0.1"

try:
    from napari_pecan_py._menu_groups import _ensure_npe2_register_hook
except ImportError:
    pass
else:
    _ensure_npe2_register_hook()

__all__ = ["__version__"]
