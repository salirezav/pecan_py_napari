"""Photoshop-style Curves UI: histogram, grid, identity line, draggable control points.

Standalone QWidget (qtpy only). Output matches ``color_tuner.logic.apply_curves``:
  ``x_points``, ``y_points`` (0–255), piecewise linear LUT with endpoints at 0 and 255.
"""

from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPointF, QRectF, Qt, Signal
from qtpy.QtGui import QColor, QPainter, QPalette, QPen
from qtpy.QtWidgets import QSizePolicy, QWidget


class CurvesHistogramEditor(QWidget):
    """Input (horizontal) → output (vertical) curve over a square plot."""

    curve_changed = Signal(dict)
    """``{"x_points": [...], "y_points": [...]}`` with int values."""

    _MARGIN = 8
    _PLOT_MIN = 200
    _GRID_DIVS = 4
    _HIST_FRACTION = 0.32  # fraction of plot height for histogram bars (from bottom)
    _POINT_RADIUS = 6
    _HIT_PX = 10.0

    # Drawing colors (napari / dark docks often apply light QSS; we force a dark plot).
    _COLOR_FRAME = QColor(52, 52, 52)
    _COLOR_PLOT_BG = QColor(30, 30, 32)
    _COLOR_HIST = QColor(100, 120, 160, 110)
    _COLOR_GRID = QColor(110, 110, 118)
    _COLOR_IDENTITY = QColor(85, 85, 95)
    _COLOR_CURVE = QColor(255, 190, 105)
    _COLOR_BORDER = QColor(210, 210, 215)
    _COLOR_HANDLE_END = QColor(245, 245, 248)
    _COLOR_HANDLE_INTERIOR = QColor(64, 156, 255)
    _COLOR_HANDLE_RING = QColor(18, 18, 20)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(self._PLOT_MIN + 2 * self._MARGIN, self._PLOT_MIN + 2 * self._MARGIN)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)

        # Avoid parent stylesheets / palette filling this widget light before paintEvent.
        self.setAutoFillBackground(False)
        if hasattr(Qt, "WidgetAttribute"):
            self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
            self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        else:
            self.setAttribute(Qt.WA_OpaquePaintEvent, True)
            self.setAttribute(Qt.WA_StyledBackground, False)

        pal = self.palette()
        pal.setColor(QPalette.ColorRole.Window, self._COLOR_FRAME)
        pal.setColor(QPalette.ColorRole.Base, self._COLOR_PLOT_BG)
        self.setPalette(pal)

        self._hist = np.ones(256, dtype=np.float64)
        self._block_emit = False

        # Control points sorted by x; first x==0, last x==255
        self._xs: list[int] = [0, 64, 128, 255]
        self._ys: list[int] = [0, 70, 200, 255]

        self._drag_idx: int | None = None
        self.setMouseTracking(True)

    # ---- Public API -----------------------------------------------------

    def set_histogram(self, hist: np.ndarray | None) -> None:
        if hist is None:
            self._hist = np.ones(256, dtype=np.float64)
        else:
            h = np.asarray(hist, dtype=np.float64).reshape(-1)
            if h.size != 256:
                raise ValueError(f"Expected 256 histogram bins, got {h.size}")
            self._hist = np.maximum(h, 1e-6)
        self.update()

    def set_curve(
        self,
        x_points: list[int],
        y_points: list[int],
        *,
        block_signals: bool = False,
    ) -> None:
        xs = [int(np.clip(x, 0, 255)) for x in x_points]
        ys = [int(np.clip(y, 0, 255)) for y in y_points]
        if len(xs) != len(ys) or len(xs) < 2:
            xs = [0, 255]
            ys = [0, 255]
        order = np.argsort(xs)
        xs = [xs[i] for i in order]
        ys = [ys[i] for i in order]
        xs[0] = 0
        xs[-1] = 255
        self._xs = xs
        self._ys = ys
        prev = self._block_emit
        self._block_emit = block_signals
        self.update()
        if not block_signals:
            self._emit()
        self._block_emit = prev

    def curve(self) -> dict:
        return {"x_points": list(self._xs), "y_points": list(self._ys)}

    # ---- Plot geometry (data: x,y in 0..255; screen y flipped) --------

    def _plot_rect(self) -> QRectF:
        w = float(self.width()) - 2 * self._MARGIN
        h = float(self.height()) - 2 * self._MARGIN
        side = min(w, h)
        x0 = self._MARGIN + (w - side) / 2.0
        y0 = self._MARGIN + (h - side) / 2.0
        return QRectF(x0, y0, side, side)

    def _data_to_px(self, x_in: float, y_out: float, plot: QRectF) -> QPointF:
        px = plot.left() + (x_in / 255.0) * plot.width()
        py = plot.bottom() - (y_out / 255.0) * plot.height()
        return QPointF(px, py)

    def _px_to_data(self, p: QPointF, plot: QRectF) -> tuple[float, float]:
        t = (p.x() - plot.left()) / max(plot.width(), 1e-6)
        u = (plot.bottom() - p.y()) / max(plot.height(), 1e-6)
        x_in = float(np.clip(t * 255.0, 0.0, 255.0))
        y_out = float(np.clip(u * 255.0, 0.0, 255.0))
        return x_in, y_out

    def paintEvent(self, event):
        del event
        plot = self._plot_rect()
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        p.fillRect(self.rect(), self._COLOR_FRAME)
        p.fillRect(plot.toRect(), self._COLOR_PLOT_BG)

        self._draw_histogram(p, plot)
        self._draw_grid(p, plot)
        self._draw_identity(p, plot)
        self._draw_curve(p, plot)
        self._draw_points(p, plot)

        # drawRect uses the current brush; points left a solid brush — clear it or the
        # whole plot repaints as a flat slab (looked like a "white-out" in dark UI).
        _nb = getattr(Qt, "NoBrush", None)
        if _nb is None:
            _nb = Qt.BrushStyle.NoBrush
        p.setPen(QPen(self._COLOR_BORDER))
        p.setBrush(_nb)
        p.drawRect(plot)

    def _draw_histogram(self, p: QPainter, plot: QRectF):
        hmax = float(np.max(self._hist))
        max_bar_h = plot.height() * self._HIST_FRACTION
        for i in range(256):
            x0 = plot.left() + (i / 255.0) * plot.width()
            x1 = plot.left() + ((i + 1) / 255.0) * plot.width()
            nh = self._hist[i] / hmax
            bar_h = nh * max_bar_h
            p.fillRect(
                QRectF(x0, plot.bottom() - bar_h, max(x1 - x0, 1.0), bar_h),
                self._COLOR_HIST,
            )

    def _draw_grid(self, p: QPainter, plot: QRectF):
        pen = QPen(self._COLOR_GRID)
        pen.setStyle(Qt.PenStyle.DotLine)
        p.setPen(pen)
        for i in range(1, self._GRID_DIVS):
            t = i / float(self._GRID_DIVS)
            xv = plot.left() + t * plot.width()
            yv = plot.top() + t * plot.height()
            p.drawLine(QPointF(xv, plot.top()), QPointF(xv, plot.bottom()))
            p.drawLine(QPointF(plot.left(), yv), QPointF(plot.right(), yv))

    def _draw_identity(self, p: QPainter, plot: QRectF):
        pen = QPen(self._COLOR_IDENTITY)
        pen.setStyle(Qt.PenStyle.DashLine)
        p.setPen(pen)
        a = self._data_to_px(0, 0, plot)
        b = self._data_to_px(255, 255, plot)
        p.drawLine(a, b)

    def _draw_curve(self, p: QPainter, plot: QRectF):
        pen = QPen(self._COLOR_CURVE)
        pen.setWidthF(2.5)
        p.setPen(pen)
        for i in range(len(self._xs) - 1):
            a = self._data_to_px(float(self._xs[i]), float(self._ys[i]), plot)
            b = self._data_to_px(float(self._xs[i + 1]), float(self._ys[i + 1]), plot)
            p.drawLine(a, b)

    def _draw_points(self, p: QPainter, plot: QRectF):
        for i in range(len(self._xs)):
            c = self._data_to_px(float(self._xs[i]), float(self._ys[i]), plot)
            interior = (
                self._COLOR_HANDLE_INTERIOR if 0 < i < len(self._xs) - 1 else self._COLOR_HANDLE_END
            )
            p.setPen(QPen(self._COLOR_HANDLE_RING, 2.0))
            p.setBrush(interior)
            p.drawEllipse(c, float(self._POINT_RADIUS), float(self._POINT_RADIUS))

    def _nearest_point_index(self, pos: QPointF, plot: QRectF) -> int:
        best_i = -1
        best_d = 1e9
        for i in range(len(self._xs)):
            c = self._data_to_px(float(self._xs[i]), float(self._ys[i]), plot)
            d = (pos.x() - c.x()) ** 2 + (pos.y() - c.y()) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d <= self._HIT_PX**2:
            return best_i
        return -1

    def _interp_y_at_x(self, x: float) -> float:
        xs = np.array(self._xs, dtype=np.float32)
        ys = np.array(self._ys, dtype=np.float32)
        return float(np.interp(x, xs, ys))

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            return
        plot = self._plot_rect()
        idx = self._nearest_point_index(event.position(), plot)
        if idx >= 0:
            self._drag_idx = idx
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        plot = self._plot_rect()
        if self._drag_idx is None:
            idx = self._nearest_point_index(event.position(), plot)
            self.setCursor(
                Qt.CursorShape.OpenHandCursor if idx >= 0 else Qt.CursorShape.ArrowCursor
            )
            super().mouseMoveEvent(event)
            return

        idx = self._drag_idx
        x_in, y_out = self._px_to_data(event.position(), plot)
        y_out_i = int(round(y_out))
        y_out_i = int(np.clip(y_out_i, 0, 255))

        if idx == 0:
            self._ys[0] = y_out_i
        elif idx == len(self._xs) - 1:
            self._ys[-1] = y_out_i
        else:
            x_lo = self._xs[idx - 1] + 2
            x_hi = self._xs[idx + 1] - 2
            if x_hi < x_lo:
                x_hi = x_lo
            x_new = int(round(np.clip(x_in, x_lo, x_hi)))
            self._xs[idx] = x_new
            self._ys[idx] = y_out_i

        self.update()
        self._emit()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag_idx = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """Insert a point on the segment under the click (interior only)."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        plot = self._plot_rect()
        if not plot.contains(event.position()):
            return
        x_in, _ = self._px_to_data(event.position(), plot)
        x_in = int(round(np.clip(x_in, 1, 254)))

        for i in range(len(self._xs) - 1):
            if self._xs[i] < x_in < self._xs[i + 1]:
                if self._xs[i + 1] - self._xs[i] < 5:
                    return
                y_new = int(round(self._interp_y_at_x(float(x_in))))
                y_new = int(np.clip(y_new, 0, 255))
                self._xs.insert(i + 1, x_in)
                self._ys.insert(i + 1, y_new)
                self.update()
                self._emit()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        # Delete selected interior point: need focus + selection — keep simple: Del removes last added middle if any
        super().keyPressEvent(event)

    def _emit(self) -> None:
        if self._block_emit:
            return
        self.curve_changed.emit(self.curve())

    def leaveEvent(self, event):
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().leaveEvent(event)
