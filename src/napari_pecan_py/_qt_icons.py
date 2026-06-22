"""Shared Qt icon helpers for napari-pecan-py widgets."""

from __future__ import annotations

from qtpy.QtCore import Qt
from qtpy.QtGui import QColor, QIcon, QPainter, QPixmap
from qtpy.QtWidgets import QPushButton, QStyle, QWidget


def _resolve_standard_pixmap(*candidates) -> QStyle.StandardPixmap | None:
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, str):
            value = getattr(QStyle, candidate, None)
        else:
            value = candidate
        if value is not None:
            return value
    return None


def theme_or_standard_icon(
    style: QStyle,
    theme_names: tuple[str, ...],
    standard: QStyle.StandardPixmap | str,
    *fallback_standards: QStyle.StandardPixmap | str,
) -> QIcon:
    for name in theme_names:
        icon = QIcon.fromTheme(name)
        if not icon.isNull():
            return icon
    sp = _resolve_standard_pixmap(standard, *fallback_standards)
    if sp is not None:
        return style.standardIcon(sp)
    return QIcon()


def tinted_icon(icon: QIcon, color: QColor, size: int = 24) -> QIcon:
    pixmap = icon.pixmap(size, size)
    tinted = QPixmap(pixmap.size())
    tinted.fill(Qt.transparent)
    painter = QPainter(tinted)
    painter.setCompositionMode(QPainter.CompositionMode_Source)
    painter.drawPixmap(0, 0, pixmap)
    painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
    painter.fillRect(tinted.rect(), color)
    painter.end()
    return QIcon(tinted)


def icon_button(
    parent: QWidget,
    *,
    icon: QIcon,
    tooltip: str,
    on_click=None,
    fixed_size: int = 28,
) -> QPushButton:
    btn = QPushButton(parent)
    btn.setIcon(icon)
    btn.setToolTip(tooltip)
    btn.setFixedSize(fixed_size, fixed_size)
    if on_click is not None:
        btn.clicked.connect(on_click)
    return btn
