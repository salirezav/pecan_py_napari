"""Tests for napari notification auto-dismiss configuration."""

from __future__ import annotations


def test_install_notification_auto_dismiss_sets_duration_and_is_idempotent():
    from napari._qt.dialogs.qt_notification import NapariQtNotification

    from napari_pecan_py._notifications import (
        _DISMISS_AFTER_MS,
        install_notification_auto_dismiss,
    )
    import napari_pecan_py._notifications as notifications

    # Reset so the test can re-run the installer even if package import already did.
    notifications._installed = False
    install_notification_auto_dismiss()
    assert NapariQtNotification.DISMISS_AFTER == _DISMISS_AFTER_MS
    patched = NapariQtNotification.show
    install_notification_auto_dismiss()
    assert NapariQtNotification.show is patched
