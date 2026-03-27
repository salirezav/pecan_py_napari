"""Photoshop-style Levels UI: histogram + draggable input/output handles.

Self-contained QWidget with no dependency on napari beyond qtpy.
Emits ``levels_changed`` when the user drags handles (or when set programmatically
with ``block_signals=True`` for silent updates).

Parameters match ``color_tuner.logic.apply_levels``:
  in_min, gamma, in_max, out_min, out_max
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QCursor, QLinearGradient, QPainter, QPen, QPolygonF
from qtpy.QtWidgets import QSizePolicy, QWidget

_Handle = Literal["none", "in_black", "in_gamma", "in_white", "out_black", "out_white"]


class LevelsHistogramEditor(QWidget):
    """Histogram + gradient bars with draggable triangles (input + output levels)."""

    levels_changed = Signal(dict)
    """Emitted with ``{"in_min", "gamma", "in_max", "out_min", "out_max"}``."""

    _HIST_TOP = 4
    _HIST_HEIGHT = 72
    _GAP = 6
    _GRAD_HEIGHT = 14
    _OUT_GAP = 8
    _OUT_GRAD_HEIGHT = 12
    _BOTTOM_PAD = 4
    _TRI_HW = 9  # half-width of triangle base
    _TRI_H = 7  # triangle height (pointing up for input, down for output)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(
            self._HIST_TOP
            + self._HIST_HEIGHT
            + self._GAP
            + self._GRAD_HEIGHT
            + self._OUT_GAP
            + self._OUT_GRAD_HEIGHT
            + self._BOTTOM_PAD
            + self._TRI_H
        )
        self.setMinimumWidth(200)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self._hist = np.ones(256, dtype=np.float64)  # display-only
        self._block_emit = False

        self._in_min = 0
        self._gamma = 1.0
        self._in_max = 255
        self._out_min = 0
        self._out_max = 255

        self._drag: _Handle = "none"
        self._last_mouse_x: float | None = None

        self.setMouseTracking(True)

    # ---- Public API -----------------------------------------------------

    def set_histogram(self, hist: np.ndarray | None) -> None:
        """Set 256-bin histogram (counts); normalized internally for drawing."""
        if hist is None:
            self._hist = np.ones(256, dtype=np.float64)
        else:
            h = np.asarray(hist, dtype=np.float64).reshape(-1)
            if h.size != 256:
                raise ValueError(f"Expected 256 histogram bins, got {h.size}")
            self._hist = np.maximum(h, 1e-6)
        self.update()

    def set_levels(
        self,
        in_min: int,
        gamma: float,
        in_max: int,
        out_min: int,
        out_max: int,
        *,
        block_signals: bool = False,
    ) -> None:
        self._in_min = int(np.clip(in_min, 0, 253))
        self._in_max = int(np.clip(in_max, self._in_min + 2, 255))
        self._gamma = float(np.clip(gamma, 0.01, 9.99))
        self._out_min = int(np.clip(out_min, 0, 254))
        self._out_max = int(np.clip(out_max, self._out_min + 1, 255))
        prev = self._block_emit
        self._block_emit = block_signals
        self.update()
        if not block_signals:
            self._emit()
        self._block_emit = prev

    def levels(self) -> dict:
        return {
            "in_min": self._in_min,
            "gamma": self._gamma,
            "in_max": self._in_max,
            "out_min": self._out_min,
            "out_max": self._out_max,
        }

    # ---- Gamma <-> mid handle position between in_min and in_max -------

    @staticmethod
    def _gamma_to_ratio(gamma: float) -> float:
        """Map gamma in [0.01, 9.99] to ratio in (0,1) for mid handle; low gamma -> right."""
        g = float(np.clip(gamma, 0.01, 9.99))
        lo = math.log(0.01)
        hi = math.log(9.99)
        t = (math.log(g) - lo) / (hi - lo)
        return float(np.clip(1.0 - t, 0.02, 0.98))

    @staticmethod
    def _ratio_to_gamma(ratio: float) -> float:
        r = float(np.clip(ratio, 0.02, 0.98))
        lo = math.log(0.01)
        hi = math.log(9.99)
        t = 1.0 - r
        return float(np.clip(math.exp(lo + t * (hi - lo)), 0.01, 9.99))

    # ---- Geometry -------------------------------------------------------

    def _content_rect(self) -> QRectF:
        m = 6.0
        return QRectF(m, float(self._HIST_TOP), self.width() - 2 * m, float(self.height() - self._HIST_TOP))

    def _hist_rect(self) -> QRectF:
        r = self._content_rect()
        return QRectF(r.left(), r.top(), r.width(), float(self._HIST_HEIGHT))

    def _input_grad_rect(self) -> QRectF:
        r = self._content_rect()
        y = r.top() + self._HIST_HEIGHT + self._GAP
        return QRectF(r.left(), y, r.width(), float(self._GRAD_HEIGHT))

    def _output_grad_rect(self) -> QRectF:
        r = self._content_rect()
        y = r.top() + self._HIST_HEIGHT + self._GAP + self._GRAD_HEIGHT + self._OUT_GAP
        return QRectF(r.left(), y, r.width(), float(self._OUT_GRAD_HEIGHT))

    def _x_for_level(self, level: float, rect: QRectF) -> float:
        return rect.left() + (level / 255.0) * rect.width()

    def _level_for_x(self, x: float, rect: QRectF) -> float:
        t = (x - rect.left()) / max(rect.width(), 1e-6)
        return float(np.clip(round(t * 255.0), 0, 255))

    # ---- Painting -------------------------------------------------------

    def paintEvent(self, event):
        del event
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        hist_r = self._hist_rect()
        in_r = self._input_grad_rect()
        out_r = self._output_grad_rect()

        # Histogram background
        p.fillRect(hist_r, QColor(40, 40, 40))

        # Draw histogram
        hmax = float(np.max(self._hist))
        pen_hist = QPen(QColor(160, 160, 160))
        p.setPen(pen_hist)
        for i in range(256):
            x0 = hist_r.left() + (i / 255.0) * hist_r.width()
            x1 = hist_r.left() + ((i + 1) / 255.0) * hist_r.width()
            nh = self._hist[i] / hmax
            bar_h = nh * hist_r.height()
            p.fillRect(
                QRectF(x0, hist_r.bottom() - bar_h, max(x1 - x0, 1.0), bar_h),
                QColor(200, 200, 200, 200),
            )

        # Input gradient (black -> white)
        gin = QLinearGradient(in_r.left(), in_r.top(), in_r.right(), in_r.top())
        gin.setColorAt(0, QColor(0, 0, 0))
        gin.setColorAt(1, QColor(255, 255, 255))
        p.fillRect(in_r, QBrush(gin))
        p.setPen(QPen(QColor(120, 120, 120)))
        p.drawRect(in_r)

        # Output gradient
        gout = QLinearGradient(out_r.left(), out_r.top(), out_r.right(), out_r.top())
        gout.setColorAt(0, QColor(self._out_min, self._out_min, self._out_min))
        gout.setColorAt(1, QColor(self._out_max, self._out_max, self._out_max))
        p.fillRect(out_r, QBrush(gout))
        p.setPen(QPen(QColor(120, 120, 120)))
        p.drawRect(out_r)

        # Input handles: triangles below input bar (pointing up)
        xb = self._x_for_level(self._in_min, in_r)
        xw = self._x_for_level(self._in_max, in_r)
        ratio = self._gamma_to_ratio(self._gamma)
        xm = self._in_min + ratio * (self._in_max - self._in_min)
        xg = self._x_for_level(xm, in_r)

        self._draw_input_triangle(p, QPointF(xb, in_r.bottom()), QColor(220, 220, 220))
        self._draw_input_triangle(p, QPointF(xg, in_r.bottom()), QColor(180, 220, 255))
        self._draw_input_triangle(p, QPointF(xw, in_r.bottom()), QColor(220, 220, 220))

        # Output handles: triangles above output bar (pointing down)
        xob = self._x_for_level(self._out_min, out_r)
        xow = self._x_for_level(self._out_max, out_r)
        self._draw_output_triangle(p, QPointF(xob, out_r.top()), QColor(220, 220, 220))
        self._draw_output_triangle(p, QPointF(xow, out_r.top()), QColor(220, 220, 220))

    def _draw_input_triangle(self, p: QPainter, tip_bottom_center: QPointF, fill: QColor):
        # Tip at bottom center, pointing up
        h = float(self._TRI_H)
        w = float(self._TRI_HW)
        tip = tip_bottom_center
        poly = QPolygonF(
            [
                tip,
                QPointF(tip.x() - w, tip.y() - h),
                QPointF(tip.x() + w, tip.y() - h),
            ]
        )
        p.setPen(QPen(QColor(30, 30, 30)))
        p.setBrush(QBrush(fill))
        p.drawPolygon(poly)

    def _draw_output_triangle(self, p: QPainter, tip_top_center: QPointF, fill: QColor):
        h = float(self._TRI_H)
        w = float(self._TRI_HW)
        tip = tip_top_center
        poly = QPolygonF(
            [
                tip,
                QPointF(tip.x() - w, tip.y() + h),
                QPointF(tip.x() + w, tip.y() + h),
            ]
        )
        p.setPen(QPen(QColor(30, 30, 30)))
        p.setBrush(QBrush(fill))
        p.drawPolygon(poly)

    # ---- Hit test / drag ------------------------------------------------

    def _handle_at(self, pos: QPointF) -> _Handle:
        in_r = self._input_grad_rect()
        out_r = self._output_grad_rect()
        tol = 12.0

        xb = self._x_for_level(self._in_min, in_r)
        xw = self._x_for_level(self._in_max, in_r)
        ratio = self._gamma_to_ratio(self._gamma)
        xm = self._in_min + ratio * (self._in_max - self._in_min)
        xg = self._x_for_level(xm, in_r)

        # Input triangles: tip on bottom edge of in_r
        for name, x in (
            ("in_black", xb),
            ("in_gamma", xg),
            ("in_white", xw),
        ):
            tip = QPointF(x, in_r.bottom())
            if abs(pos.x() - tip.x()) <= tol and abs(pos.y() - tip.y()) <= tol + self._TRI_H:
                return name  # type: ignore[return-value]

        xob = self._x_for_level(self._out_min, out_r)
        xow = self._x_for_level(self._out_max, out_r)
        for name, x in (("out_black", xob), ("out_white", xow)):
            tip = QPointF(x, out_r.top())
            if abs(pos.x() - tip.x()) <= tol and abs(pos.y() - tip.y()) <= tol + self._TRI_H:
                return name  # type: ignore[return-value]

        return "none"

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._drag = self._handle_at(event.position())
        self._last_mouse_x = event.position().x()
        if self._drag != "none":
            self.setCursor(QCursor(Qt.CursorShape.ClosedHandCursor))
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag == "none":
            h = self._handle_at(event.position())
            self.setCursor(
                QCursor(Qt.CursorShape.OpenHandCursor)
                if h != "none"
                else QCursor(Qt.CursorShape.ArrowCursor)
            )
            super().mouseMoveEvent(event)
            return

        in_r = self._input_grad_rect()
        out_r = self._output_grad_rect()
        x = event.position().x()

        if self._drag == "in_black":
            lvl = int(self._level_for_x(x, in_r))
            self._in_min = int(np.clip(lvl, 0, self._in_max - 2))
        elif self._drag == "in_white":
            lvl = int(self._level_for_x(x, in_r))
            self._in_max = int(np.clip(lvl, self._in_min + 2, 255))
        elif self._drag == "in_gamma":
            lvl = float(self._level_for_x(x, in_r))
            span = float(self._in_max - self._in_min)
            if span < 2:
                span = 2.0
            ratio = (lvl - self._in_min) / span
            self._gamma = self._ratio_to_gamma(ratio)
        elif self._drag == "out_black":
            lvl = int(self._level_for_x(x, out_r))
            self._out_min = int(np.clip(lvl, 0, self._out_max - 1))
        elif self._drag == "out_white":
            lvl = int(self._level_for_x(x, out_r))
            self._out_max = int(np.clip(lvl, self._out_min + 1, 255))

        self._last_mouse_x = x
        self.update()
        self._emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = "none"
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().mouseReleaseEvent(event)

    def _emit(self) -> None:
        if self._block_emit:
            return
        self.levels_changed.emit(self.levels())

    def leaveEvent(self, event):
        self.setCursor(QCursor(Qt.CursorShape.ArrowCursor))
        super().leaveEvent(event)
