"""Custom UI widgets: LR-style sliders, sections, histogram, tone curve, color wheels."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QColor, QConicalGradient, QCursor, QImage, QPainter,
                           QPainterPath, QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QScrollArea, QSizePolicy, QSlider, QVBoxLayout,
                               QWidget)

from . import theme


# ------------------------------------------------------------------ slider row

class ValueBox(QLabel):
    """Editable value display; click to type, arrows to nudge."""
    valueEdited = Signal(float)
    dragDelta = Signal(float)   # accumulated while scrubbing

    def __init__(self, value: float, decimals: int = 2, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self._v = value
        self.setText(self._fmt())
        self.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setMinimumWidth(44)
        self.setCursor(Qt.SizeHorCursor)
        self.setStyleSheet(
            f"background:#262626; border:1px solid {theme.BORDER};"
            "border-radius:3px; padding:1px 4px; color:%s;" % theme.TEXT_DIM)
        self._press_x = None

    def _fmt(self) -> str:
        return f"{self._v:+.{self.decimals}f}" if self._v != 0 else f"{0:.{self.decimals}f}"

    def set_value(self, v: float):
        self._v = v
        self.setText(self._fmt())

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._press_x = e.position().x()
            self._last_x = e.position().x()
            e.accept()

    def mouseMoveEvent(self, e):
        if self._press_x is not None:
            dx = e.position().x() - self._last_x
            self._last_x = e.position().x()
            step = 0.05 if e.modifiers() & Qt.ShiftModifier else 0.01
            if abs(dx) >= 2:
                self.dragDelta.emit(dx * step * 2.0)
            e.accept()

    def mouseReleaseEvent(self, e):
        self._press_x = None

    def mouseDoubleClickEvent(self, e):
        self.valueEdited.emit(0.0)


class SliderRow(QWidget):
    """Lightroom-style labeled slider with scrubbable value box."""
    valueChanged = Signal(float)          # live
    editingFinished = Signal(float)       # on release (for history/undo)

    def __init__(self, label: str, minimum: float, maximum: float, value: float,
                 decimals: int = 0, default: float | None = None, parent=None):
        super().__init__(parent)
        self.minimum, self.maximum = minimum, maximum
        self.default = default if default is not None else 0.0
        self._decimals = decimals
        self._block = False

        lay = QHBoxLayout(self)
        lay.setContentsMargins(6, 1, 6, 1)
        lay.setSpacing(8)

        self.label = QLabel(label)
        self.label.setStyleSheet("background:transparent; color:%s;" % theme.TEXT_DIM)
        self.label.setFixedWidth(78)
        self.label.setCursor(Qt.PointingHandCursor)
        self.label.mouseDoubleClickEvent = lambda e: self.reset()
        lay.addWidget(self.label)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(int(minimum * 10), int(maximum * 10))
        self.slider.setValue(int(value * 10))
        self.slider.setFixedHeight(16)
        lay.addWidget(self.slider, 1)

        self.valuebox = ValueBox(value, decimals)
        lay.addWidget(self.valuebox)

        self.slider.valueChanged.connect(self._on_slider)
        self.valuebox.dragDelta.connect(self._on_scrub)
        self.valuebox.valueEdited.connect(self.reset)

    def _emit(self, v: float, final: bool):
        v = round(max(self.minimum, min(self.maximum, v)), self._decimals + 1)
        self._block = True
        self.slider.setValue(int(v * 10))
        self.valuebox.set_value(v)
        self._block = False
        self.valueChanged.emit(v)
        if final:
            self.editingFinished.emit(v)

    def _on_slider(self, iv: int):
        if self._block:
            return
        self._emit(iv / 10.0, False)

    def _on_scrub(self, d: float):
        cur = self.value()
        nv = cur + (d * (self.maximum - self.minimum) / 200.0)
        self._emit(nv, False)

    def mouseReleaseEvent(self, e):  # finalize from slider too
        super().mouseReleaseEvent(e)

    def value(self) -> float:
        return self.slider.value() / 10.0

    def set_value_silent(self, v: float):
        self._block = True
        self.slider.setValue(int(v * 10))
        self.valuebox.set_value(v)
        self._block = False

    def reset(self):
        self._emit(self.default, True)


# ------------------------------------------------------------------ collapsible section

class CollapsibleSection(QFrame):
    expandedChanged = Signal(bool)

    def __init__(self, title: str, expanded: bool = True, parent=None):
        super().__init__(parent)
        self.setObjectName("Section")
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 0, 8, 4)
        v.setSpacing(0)

        self.header = QPushButton()
        self.header.setObjectName("SectionHeader")
        self.header.setCursor(Qt.PointingHandCursor)
        self.header.setStyleSheet(
            "text-align:left; border:none; background:transparent; padding:7px 2px;")
        self.header.clicked.connect(self.toggle)
        v.addWidget(self.header)

        self.body = QWidget()
        self.body_lay = QVBoxLayout(self.body)
        self.body_lay.setContentsMargins(0, 2, 0, 6)
        self.body_lay.setSpacing(2)
        v.addWidget(self.body)

        self._title = title
        self.setExpanded(expanded)

    def setExpanded(self, ex: bool):
        arrow = "\u25be  " if ex else "\u25b8  "
        self.header.setText(f"{arrow}{self._title.upper()}")
        self.body.setVisible(ex)
        self.expandedChanged.emit(ex)

    def toggle(self):
        self.setExpanded(not self.body.isVisible())

    def add(self, w: QWidget):
        self.body_lay.addWidget(w)


# ------------------------------------------------------------------ histogram

class HistogramWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("HistogramBox")
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._hist = None

    def update_image(self, u8: np.ndarray | None):
        if u8 is None or u8.size == 0:
            self._hist = None
        else:
            hists = []
            for c in range(3):
                hh, _ = np.histogram(u8[..., c], bins=256, range=(0, 256))
                hists.append(hh.astype(np.float64))
            # log scale for display
            self._hist = [np.log1p(h) for h in hists]
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(1, 1, -2, -2)
        p.fillRect(r, QColor("#161616"))
        if not self._hist:
            p.end()
            return
        peak = max(h.max() for h in self._hist) or 1.0
        colors = [(QColor(200, 60, 60, 150)), (QColor(70, 190, 90, 150)),
                  (QColor(70, 120, 220, 150))]
        p.setClipRect(r)
        w, hgt = r.width(), r.height()
        for hist, col in zip(self._hist, colors):
            path = QPainterPath()
            path.moveTo(r.left(), r.bottom())
            n = len(hist)
            for i in range(n):
                x = r.left() + i / (n - 1) * w
                y = r.bottom() - (hist[i] / peak) * (hgt - 6)
                path.lineTo(x, y)
            path.lineTo(r.right(), r.bottom())
            p.fillPath(path, col)
        # frame gridlines
        pen = QPen(QColor("#ffffff14"), 1)
        p.setPen(pen)
        for fx in (0.25, 0.5, 0.75):
            x = r.left() + fx * w
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
        p.end()


# ------------------------------------------------------------------ tone curve

class ToneCurveWidget(QWidget):
    curveChanged = Signal(list)      # list[(x,y)] normalized, y up

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CurveBox")
        self.setMinimumHeight(210)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.points: list[tuple[float, float]] = []
        self.setMouseTracking(True)
        self._drag_i = None
        self._hist = None

    def set_histogram(self, hist_norm: np.ndarray | None):
        self._hist = hist_norm
        self.update()

    def set_points(self, pts):
        self.points = list(pts) or []
        self.update()

    def _curve_rect(self) -> QRectF:
        return QRectF(9, 9, self.width() - 18, self.height() - 18)

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        r = self._curve_rect()
        p.fillRect(self.rect(), QColor("#232323"))

        # backdrop histogram
        if self._hist is not None:
            hp = QPainterPath()
            hp.moveTo(r.left(), r.bottom())
            n = len(self._hist)
            for i in range(n):
                x = r.left() + i / max(1, n - 1) * r.width()
                y = r.bottom() - self._hist[i] * (r.height() - 4)
                hp.lineTo(x, y)
            hp.lineTo(r.right(), r.bottom())
            p.fillPath(hp, QColor("#ffffff12"))

        # grid
        pen = QPen(QColor("#ffffff18"), 1)
        p.setPen(pen)
        for f in (0.25, 0.5, 0.75):
            x = r.left() + f * r.width()
            y = r.bottom() - f * r.height()
            p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
            p.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))

        # diagonal reference
        p.setPen(QPen(QColor("#ffffff22"), 1, Qt.DashLine))
        p.drawLine(QPointF(r.left(), r.bottom()), QPointF(r.right(), r.top()))

        # curve
        if len(self.points) < 2:
            pts_sorted = sorted(self.points)
        else:
            pts_sorted = sorted(self.points)
        full = ([QPointF(r.left(), r.bottom())] +
                [QPointF(r.left() + px * r.width(), r.bottom() - py * r.height())
                 for px, py in pts_sorted] +
                [QPointF(r.right(), r.top())])
        if len(full) >= 3:
            from lumina.core.imaging import monotonic_spline
            xs, ys = monotonic_spline([(q[0], q[1]) for q in pts_sorted]) \
                if len(pts_sorted) >= 2 else (None, None)
            if xs is not None:
                path = QPainterPath()
                for i in range(len(xs)):
                    x = r.left() + xs[i] * r.width()
                    y = r.bottom() - ys[i] * r.height()
                    if i == 0:
                        path.moveTo(x, y)
                    else:
                        path.lineTo(x, y)
                p.setPen(QPen(QColor("#e8e8e8"), 1.6))
                p.drawPath(path)
            else:
                pass

        # points
        for i, (px, py) in enumerate(sorted(self.points)):
            x = r.left() + px * r.width()
            y = r.bottom() - py * r.height()
            p.setBrush(QColor("#e8e8e8") if i != self._drag_i else QColor(theme.ACCENT_HOVER))
            p.setPen(QPen(QColor("#111"), 1))
            p.drawEllipse(QPointF(x, y), 4.5, 4.5)
        p.end()

    def _hit_point(self, pos):
        r = self._curve_rect()
        best, bi = 12.0, None
        for i, (px, py) in enumerate(self.points):
            x = r.left() + px * r.width()
            y = r.bottom() - py * r.height()
            d = math.hypot(pos.x() - x, pos.y() - y)
            if d < best:
                best, bi = d, i
        return bi

    def mousePressEvent(self, e):
        r = self._curve_rect()
        pos = e.position()
        if e.button() == Qt.RightButton:
            i = self._hit_point(pos)
            if i is not None and 0 <= i < len(self.points):
                del self.points[i]
                self.curveChanged.emit(list(self.points))
                self.update()
            return
        i = self._hit_point(pos)
        if i is None:
            if r.contains(pos):
                nx = (pos.x() - r.left()) / r.width()
                ny = 1.0 - (pos.y() - r.top()) / r.height()
                self.points.append((float(nx), float(ny)))
                i = len(self.points) - 1
                self.curveChanged.emit(list(self.points))
        self._drag_i = i
        self.update()

    def mouseMoveEvent(self, e):
        r = self._curve_rect()
        pos = e.position()
        if self._drag_i is not None and 0 <= self._drag_i < len(self.points):
            nx = min(1.0, max(0.0, (pos.x() - r.left()) / r.width()))
            ny = min(1.0, max(0.0, 1.0 - (pos.y() - r.top()) / r.height()))
            x, y = self.points[self._drag_i]
            self.points[self._drag_i] = (float(nx), float(ny))
            self.curveChanged.emit(list(self.points))
            self.update()
        else:
            self.setCursor(Qt.CrossCursor if r.contains(pos) else Qt.ArrowCursor)

    def mouseReleaseEvent(self, e):
        self._drag_i = None
        self.update()


# ------------------------------------------------------------------ color wheel

_wheel_cache: dict[int, QImage] = {}


def _build_wheel(size: int) -> QImage:
    hit = _wheel_cache.get(size)
    if hit is not None:
        return hit
    img = QImage(size, size, QImage.Format_ARGB32_Premultiplied)
    img.fill(0)
    half = size / 2
    for y in range(size):
        for x in range(size):
            dx, dy = (x - half) / half, (y - half) / half
            d = math.hypot(dx, dy)
            if d > 1.0:
                continue
            hue = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
            sat = min(1.0, d)
            r, g, b = _hsv_to_rgb(hue, sat, 1.0)
            a = 255 if d < 0.985 else int(255 * (1.0 - d) / 0.015)
            img.setPixelColor(x, y, QColor(int(r*255), int(g*255), int(b*255), a))
    if len(_wheel_cache) > 4:
        _wheel_cache.clear()
    _wheel_cache[size] = img
    return img


def _hsv_to_rgb(h, s, v):
    i = int(h / 60) % 6
    f = h / 60 - int(h / 60)
    p, q, t = v*(1-s), v*(1-s*f), v*(1-s*(1-f))
    return [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)][i]


class ColorWheel(QWidget):
    changed = Signal(float, float)     # hue deg, sat 0-100

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.hue, self.sat = 0.0, 0.0
        self.title = title
        self.setMinimumSize(118, 118)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._pix = None

    def set_hs(self, hue: float, sat: float):
        self.hue, self.sat = hue, sat
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = min(self.width(), self.height())
        cx, cy = self.width()/2, self.height()/2
        radius = s/2 - 5
        key = int(s)
        pix = _wheel_cache.get(key) or _build_wheel(max(96, key))
        target = QRectF(cx-radius, cy-radius, radius*2, radius*2)
        p.drawImage(target, pix)
        # marker
        mr = radius * self.sat
        a = math.radians(self.hue)
        mx, my = cx + mr*math.cos(a), cy - mr*math.sin(a)
        p.setPen(QPen(QColor("#fff"), 1.6))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QPointF(mx, my), 5, 5)
        p.setPen(QPen(QColor("#000"), 1))
        p.drawEllipse(QPointF(mx, my), 6.5, 6.5)
        if self.title:
            p.setPen(QColor(theme.TEXT_FAINT))
            f = p.font(); f.setPointSize(10); p.setFont(f)
            p.drawText(self.rect().adjusted(0, 2, 0, 2), Qt.AlignHCenter | Qt.AlignBottom,
                       self.title.upper())
        p.end()

    def _update_from_pos(self, pos):
        cx, cy = self.width()/2, self.height()/2
        radius = min(self.width(), self.height())/2 - 5
        dx, dy = (pos.x()-cx)/radius, -(pos.y()-cy)/radius
        d = math.hypot(dx, dy)
        self.sat = min(1.0, d)
        if d > 0.001:
            self.hue = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0
        self.changed.emit(self.hue, self.sat*100.0)
        self.update()

    def mousePressEvent(self, e):
        self._update_from_pos(e.position())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._update_from_pos(e.position())

    def mouseDoubleClickEvent(self, e):
        self.hue, self.sat = 0.0, 0.0
        self.changed.emit(0.0, 0.0)
        self.update()


# ------------------------------------------------------------------ rating stars etc.

class RatingStars(QWidget):
    changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rating = 0
        self.setFixedSize(130, 24)
        self.setCursor(Qt.PointingHandCursor)

    def set_rating(self, r: int):
        self.rating = r
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        f = p.font(); f.setPixelSize(19); p.setFont(f)
        for i in range(5):
            x = 4 + i * 25
            filled = i < self.rating
            p.setPen(QColor("#dcdcdc" if filled else "#555555"))
            p.drawText(QRectF(x, 0, 22, self.height()), Qt.AlignCenter, "\u2605")
        p.end()

    def _star_at(self, x):
        return int((x - 4) // 25) + 1 if x >= 2 else 0

    def mousePressEvent(self, e):
        r = self._star_at(e.position().x())
        self.rating = 0 if r == self.rating else r
        self.changed.emit(self.rating)
        self.update()


class FlagPicker(QWidget):
    changed = Signal(int)     # -1 reject, 0 none, 1 pick

    def __init__(self, parent=None):
        super().__init__(parent)
        self.flag = 0
        self.setFixedSize(84, 24)
        self.setCursor(Qt.PointingHandCursor)

    def set_flag(self, f: int):
        self.flag = f
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        f = p.font(); f.setPixelSize(15); p.setFont(f)
        p.setPen(QColor("#cc4444" if self.flag == -1 else "#666666"))
        p.drawText(QRectF(2, 0, 26, self.height()), Qt.AlignCenter, "\u2717")
        p.setPen(QColor("#59b356" if self.flag == 1 else "#666666"))
        p.drawText(QRectF(56, 0, 26, self.height()), Qt.AlignCenter, "\u2713")
        p.end()

    def mousePressEvent(self, e):
        x = e.position().x()
        if x < 42:
            self.flag = -1 if self.flag != -1 else 0
        else:
            self.flag = 1 if self.flag != 1 else 0
        self.changed.emit(self.flag)
        self.update()


COLOR_LABELS = ["", "#c94c4c", "#d8a13a", "#57ad4e", "#4477cf", "#8d52c9"]


class ColorLabelPicker(QWidget):
    changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.color_idx = 0
        self.setFixedSize(120, 20)
        self.setCursor(Qt.PointingHandCursor)

    def set_color(self, idx: int):
        self.color_idx = idx
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        for i in range(1, 6):
            r = QRectF(4 + (i-1)*23, 2, 17, 16)
            c = QColor(COLOR_LABELS[i])
            if self.color_idx == i:
                p.setPen(QPen(QColor("#fff"), 1.6))
            else:
                p.setPen(QPen(QColor("#00000000")))
            p.setBrush(c)
            p.drawEllipse(r)
        p.end()

    def mousePressEvent(self, e):
        idx = int((e.position().x() - 4) // 23) + 1
        self.color_idx = idx if idx != self.color_idx else 0
        self.changed.emit(self.color_idx)
        self.update()


# ------------------------------------------------------------------ thumbnail badge painter

def make_badge_pixmap(base: QPixmap, rating: int = 0, flag: int = 0,
                      color: int = 0, edited: bool = False) -> QPixmap:
    pm = base.copy()
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    w, h = pm.width(), pm.height()
    if flag == 1:
        p.setPen(QColor("#59b356")); f = p.font(); f.setPixelSize(11); p.setFont(f)
        p.drawText(QRectF(3, h-16, 14, 14), Qt.AlignCenter, "\u2713")
    elif flag == -1:
        p.setPen(QColor("#cc4444"))
        p.drawText(QRectF(3, h-16, 14, 14), Qt.AlignCenter, "\u2717")
    if color:
        p.setPen(QPen(QColor(theme.COLOR_LABELS[color]), 2.5))
        p.drawRect(1, 1, w-2, h-2)
    if rating:
        p.setPen(QColor("#e8e8e8")); f = p.font(); f.setPixelSize(10); p.setFont(f)
        stars = "\u2605" * rating
        p.drawText(QRectF(w - 14*rating - 4, h - 16, 14*rating + 2, 14),
                   Qt.AlignRight | Qt.AlignVCenter, stars)
    if edited:
        p.setPen(QColor("#7fb2e5")); f2 = p.font(); f2.setPixelSize(10); p.setFont(f2)
        p.drawText(QRectF(w - 16, 2, 14, 14), Qt.AlignCenter, "\u270e")
    p.end()
    return pm
