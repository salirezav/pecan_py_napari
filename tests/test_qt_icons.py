"""Qt icon helpers work across PyQt6 / PySide6 (missing legacy SP_* enums)."""

from qtpy.QtWidgets import QApplication, QStyle, QWidget

from napari_pecan_py._qt_icons import _resolve_standard_pixmap, theme_or_standard_icon


def test_resolve_standard_pixmap_falls_back_when_primary_missing():
    assert _resolve_standard_pixmap(
        "SP_ToolBarAddExtension",
        "SP_FileDialogNewFolder",
    ) == QStyle.SP_FileDialogNewFolder


def test_theme_or_standard_icon_does_not_require_missing_enum():
    app = QApplication.instance() or QApplication([])
    widget = QWidget()
    icon = theme_or_standard_icon(
        widget.style(),
        ("__napari_pecan_py_missing_theme_icon__",),
        "SP_ToolBarAddExtension",
        "SP_FileDialogNewFolder",
    )
    assert not icon.isNull()
