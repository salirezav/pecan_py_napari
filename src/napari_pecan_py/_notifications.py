"""Configure napari toast notifications for this plugin.

Napari only starts the auto-dismiss timer when the viewer window is active.
When driving napari from a notebook (focus in the IDE), toasts otherwise stay
until closed manually. We always start the timer and use a 10s dismiss delay.
"""

from __future__ import annotations

_DISMISS_AFTER_MS = 10_000
_installed = False


def install_notification_auto_dismiss(dismiss_after_ms: int = _DISMISS_AFTER_MS) -> None:
    """Make napari GUI notifications auto-dismiss after ``dismiss_after_ms``."""
    global _installed
    if _installed:
        return
    try:
        from napari._qt.dialogs.qt_notification import NapariQtNotification
        from qtpy.QtWidgets import QDialog
    except ImportError:
        return

    NapariQtNotification.DISMISS_AFTER = int(dismiss_after_ms)

    def show(self) -> None:
        # Same as NapariQtNotification.show, but always start the dismiss timer
        # (upstream skips it when the parent window is not active).
        QDialog.show(self)
        self._instances.append(self)
        self.slide_in()
        if self.parent() is not None:
            for notification in self._instances:
                notification.timer_stop()
        if self.DISMISS_AFTER > 0:
            self.timer.setInterval(self.DISMISS_AFTER)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.close_with_fade)
            self.timer.start()

    NapariQtNotification.show = show  # type: ignore[method-assign]
    _installed = True
