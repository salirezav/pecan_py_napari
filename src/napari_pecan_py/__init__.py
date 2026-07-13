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

try:
    from napari_pecan_py._notifications import install_notification_auto_dismiss
except ImportError:
    pass
else:
    install_notification_auto_dismiss()

try:
    from napari_pecan_py.trim_frames import register_trim_frames_action
except ImportError:
    pass
else:
    register_trim_frames_action()

__all__ = ["__version__"]
