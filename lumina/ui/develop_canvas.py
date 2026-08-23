"""Develop canvas - image display with zoom/pan, crop tool, mask & spot drawing."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import QWidget


def np_to_pixmap(u8: np.ndarray) -> QPixmap:
    h, w = u8.shape[:2]
    img = QImage(u8.data, w, h, w * 3, QImage.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class DevelopCanvas(QWidget):
    cropChanged = Signal(list)
    cropCommitted = Signal()
    maskDrawn = Signal(str, object)
    brushStrokeFinished = Signal(object)
    spotsChanged = Signal()
    subjectPicked = Signal(float, float)
    statusMessage = Signal(str)

    HANDLE_R = 9.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("CanvasFrame")
        self.setMouseTracking(True)

        self._image = None            # QPixmap of processed straightened frame
        self._before = None           # QPixmap of original oriented frame
        self.show_before = False
        self.mode = "view"   # view|crop|linear_new|radial_new|brush|spot|subject_pick
        self.crop = None              # [x0,y0,x1,y1] normalized
        self.aspect = "free"
        self.mask_overlay = None      # QPixmap aligned to _image or None
        self.spots = []
        self.spot_radius_norm = 0.03
        self.spot_mode = "heal"
        self.brush_radius_norm = 0.06
        self.brush_flow = 1.0
        self.zoom = 1.0               # 1.0 = fit
        self.pan = QPointF(0.0, 0.0)
        self._dragging_pan = False
        self._press_pos = QPointF(0, 0)
        self._crop_drag = None
        self._spot_drag = None
        self._selected_spot = None
        self._draw_start = None
        self._draw_cur = None
        self._stroke_pts = []
        self._cursor_img = QPointF(-1e6, -1e6)
        self._img_size = (0, 0)
        self.split_mode = False
        self.split_x = 0.5          # normalized divider position
        self._dragging_split = False

    # ---------------------------------------------------------------- data
    def set_image(self, u8):
        if u8 is None:
            self._image = None
            self._img_size = (0, 0)
        else:
            self._image = np_to_pixmap(u8)
            self._img_size = (u8.shape[1], u8.shape[0])
        self.update()

    def set_before_image(self, u8):
        self._before = np_to_pixmap(u8) if u8 is not None else None
        self.update()

    def set_mask_overlay(self, pm):
        self.mask_overlay = pm
        self.update()

    def set_crop(self, crop):
        self.crop = list(crop) if crop else None
        self.update()

    def set_zoom_fit(self):
        self.zoom = 1.0
        self.pan = QPointF(0, 0)
        self.update()

    def set_zoom_100(self):
        fit = self._fit_scale()
        if fit > 0:
            self.zoom = max(1.01, 1.0 / fit)
        self.pan = QPointF(0, 0)
        self.update()

    def wheelEvent_zoom(self, factor: float):
        self.zoom = min(16.0, max(1.0, self.zoom * factor))
        if self.zoom == 1.0:
            self.pan = QPointF(0, 0)
        self.update()

    def toggle_split(self):
        self.split_mode = not self.split_mode
        self.update()

    def _split_line_x(self) -> float:
        r = self._display_rect()
        return r.left() + self.split_x * r.width()

    def _near_split(self, x: float) -> bool:
        return abs(x - self._split_line_x()) < 8
        self.show_before = on
        self.update()

    # ------------------------------------------------------------ transforms
    def _fit_scale(self) -> float:
        sw, sh = self._source_size()
        if not sw:
            return 1.0
        return min((self.width() - 24) / sw, (self.height() - 24) / sh, 4.0)

    def _source_size(self):
        return self._img_size

    def _scale(self) -> float:
        return self._fit_scale() * self.zoom

    def _display_rect(self) -> QRectF:
        sw, sh = self._source_size()
        s = self._scale()
        w, h = sw * s, sh * s
        x = (self.width() - w) / 2 + self.pan.x()
        y = (self.height() - h) / 2 + self.pan.y()
        return QRectF(x, y, w, h)

    def img_from_widget(self, p: QPointF) -> QPointF:
        r = self._display_rect()
        if r.width() < 1:
            return QPointF(0, 0)
        return QPointF((p.x() - r.left()) / r.width(),
                       (p.y() - r.top()) / r.height())

    def widget_from_img(self, nx: float, ny: float) -> QPointF:
        r = self._display_rect()
        return QPointF(r.left() + nx * r.width(), r.top() + ny * r.height())

    # ------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        p.fillRect(self.rect(), QColor("#191919"))

        disp = self._display_rect()
        split_active = (self.split_mode and not self.show_before
                        and self._image is not None and self._before is not None)

        if split_active:
            # draw processed full frame
            p.drawPixmap(disp, self._image, QRectF(self._image.rect()))
            # overlay original on LEFT of divider
            sx = self.split_x * self._image.width()
            sy = 0; sw = sx; sh = self._image.height()
            left_w = self.split_x * disp.width()
            if left_w > 0.5:
                p.drawPixmap(QRectF(disp.left(), disp.top(),
                                    left_w, disp.height()),
                             self._before,
                             QRectF(0, 0,
                                    self.split_x * self._before.width(),
                                    self._before.height()))
            # divider line + grip
            line_x = self._split_line_x()
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawLine(QPointF(line_x, disp.top()),
                       QPointF(line_x, disp.bottom()))
            p.setBrush(QColor("#ffffff"))
            p.setPen(QPen(QColor("#333"), 1))
            mid_y = disp.top() + disp.height()/2
            p.drawEllipse(QPointF(line_x, mid_y), 7, 7)
            p.setPen(QColor("#333"))
            f = p.font(); f.setPixelSize(8); p.setFont(f)
            p.drawText(QPointF(line_x - 3, mid_y + 3), "↔")
            p.end(); return

        src = self._before if (self.show_before and self._before) else self._image
        if src is None:
            p.setPen(QColor("#555555"))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No photo selected\n\nImport a folder in Library (G), "
                       "then double-click a photo or press D")
            p.end()
            return

        crop_mode = self.mode == "crop" and not self.show_before

        if crop_mode and self.crop is not None:
            p.drawPixmap(disp, src, QRectF(src.rect()))
            c = self.crop
            cr = QRectF(self.widget_from_img(c[0], c[1]),
                        self.widget_from_img(c[2], c[3]))
            outside = QPainterPath()
            outside.addRect(disp)
            hole = QPainterPath()
            hole.addRect(cr)
            p.setRenderHint(QPainter.Antialiasing)
            p.fillPath(outside.subtracted(hole), QColor(10, 10, 10, 150))
            p.setPen(QPen(QColor(255, 255, 255, 70), 1))
            for f in (1/3, 2/3):
                p.drawLine(QPointF(cr.left() + cr.width()*f, cr.top()),
                           QPointF(cr.left() + cr.width()*f, cr.bottom()))
                p.drawLine(QPointF(cr.left(), cr.top() + cr.height()*f),
                           QPointF(cr.right(), cr.top() + cr.height()*f))
            p.setPen(QPen(QColor("#ffffff"), 1.5))
            p.drawRect(cr)
            p.setBrush(QColor("#ffffff"))
            p.setPen(QPen(QColor("#333333"), 1))
            for hx, hy in self._handle_positions(c):
                pt = self.widget_from_img(hx, hy)
                p.drawEllipse(pt, self.HANDLE_R, self.HANDLE_R)
        elif self.mode == "view" and not self.show_before and self.crop is not None:
            c = self.crop
            sx = c[0] * src.width()
            sy = c[1] * src.height()
            sw = (c[2] - c[0]) * src.width()
            sh = (c[3] - c[1]) * src.height()
            p.drawPixmap(QRectF(disp.left(), disp.top(), disp.width(), disp.height()),
                         src, QRectF(sx, sy, sw, sh))
        else:
            p.drawPixmap(disp, src, QRectF(src.rect()))

        # active-mask red tint overlay (CPU path)
        if (self.mask_overlay is not None and not self.show_before
                and not crop_mode and self.mode != "view"):
            p.setOpacity(0.85)
            p.drawPixmap(disp, self.mask_overlay,
                         QRectF(self.mask_overlay.rect()))
            p.setOpacity(1.0)

        # spot markers
        if self.mode == "spot" and self.spots and not self.show_before:
            p.setRenderHint(QPainter.Antialiasing)
            for sp in self.spots:
                pt = self.widget_from_img(sp["cx"], sp["cy"])
                rr = max(4.0, sp.get("r", 0.03) *
                         min(disp.width(), disp.height()) * self.zoom)
                active = sp is self._selected_spot
                col = QColor("#ffd75e") if active else QColor("#57b3ff")
                p.setPen(QPen(col, 1.6))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(pt, rr, rr)
                if sp.get("mode") == "clone":
                    st = self.widget_from_img(sp.get("sx", sp["cx"]-0.08),
                                              sp.get("sy", sp["cy"]))
                    p.setPen(QPen(QColor(87, 179, 255, 140), 1, Qt.DashLine))
                    p.drawLine(st, pt)
                    p.setPen(QPen(col, 1.2))
                    p.drawEllipse(st, rr * 0.8, rr * 0.8)

        # gradient/radial draw preview
        if self._draw_start is not None and self._draw_cur is not None:
            p.setRenderHint(QPainter.Antialiasing)
            a = self.widget_from_img(self._draw_start.x(), self._draw_start.y())
            b = self.widget_from_img(self._draw_cur.x(), self._draw_cur.y())
            p.setPen(QPen(QColor("#ffffff"), 1.5, Qt.DashLine))
            if self.mode == "linear_new":
                p.drawLine(a, b)
                for pt in (a, b):
                    p.drawEllipse(pt, 5, 5)
            elif self.mode == "radial_new":
                rect = QRectF(a, b).normalized()
                p.drawEllipse(rect)
                cx, cy = rect.center().x(), rect.center().y()
                p.drawLine(QPointF(cx - 6, cy), QPointF(cx + 6, cy))
                p.drawLine(QPointF(cx, cy - 6), QPointF(cx, cy + 6))

        # brush stroke in progress
        if self._stroke_pts:
            p.setRenderHint(QPainter.Antialiasing)
            p.setPen(QPen(QColor(255, 80, 80, 160), 3))
            pts = [self.widget_from_img(x, y) for x, y in self._stroke_pts]
            path = QPainterPath(pts[0])
            for q in pts[1:]:
                path.lineTo(q)
            p.drawPath(path)

        # brush cursor ring
        if self.mode == "brush" and not self.show_before:
            p.setRenderHint(QPainter.Antialiasing)
            brad = self.brush_radius_norm * min(disp.width(), disp.height()) * self.zoom
            cp = self.widget_from_img(self._cursor_img.x(), self._cursor_img.y())
            p.setPen(QPen(QColor("#ffffff"), 1.2))
            p.setBrush(Qt.NoBrush)
            p.drawEllipse(cp, brad, brad)
            p.setPen(QPen(QColor(0, 0, 0, 130), 1))
            p.drawEllipse(cp, brad + 1, brad + 1)

        if self.zoom > 1.001:
            p.setPen(QColor("#777777"))
            f = p.font()
            f.setPixelSize(11)
            p.setFont(f)
            p.drawText(self.rect().adjusted(8, 6, -8, 0),
                       Qt.AlignRight | Qt.AlignTop, f"{self.zoom*100:.0f}%")
        p.end()

    # ================================================================ crop helpers
    HANDLE_MAP = {0: (0, 1), 1: (None, 1), 2: (2, 1), 3: (2, None),
                  4: (2, 3), 5: (None, 3), 6: (0, 3), 7: (0, None)}
    OPPOSITE = {0: (2, 3), 1: (None, 3), 2: (0, 3), 3: (0, None),
                4: (0, 1), 5: (None, 1), 6: (2, 1), 7: (2, None)}

    def _handle_positions(self, c):
        x0, y0, x1, y1 = c
        xm, ym = (x0+x1)/2, (y0+y1)/2
        return [(x0,y0),(xm,y0),(x1,y0),(x1,ym),(x1,y1),(xm,y1),(x0,y1),(x0,ym)]

    def _hit_crop_handle(self, wp: QPointF):
        if self.crop is None:
            return None
        c = self.crop
        for i, (hx, hy) in enumerate(self._handle_positions(c)):
            pt = self.widget_from_img(hx, hy)
            if abs(pt.x()-wp.x()) <= self.HANDLE_R+3 and \
                    abs(pt.y()-wp.y()) <= self.HANDLE_R+3:
                return (("c" if i % 2 == 0 else "e"), i)
        cr = QRectF(self.widget_from_img(c[0], c[1]),
                    self.widget_from_img(c[2], c[3]))
        if cr.contains(wp):
            return ("m", 0)
        return None

    def _aspect_ratio(self):
        rw, rh = self._img_size
        if self.aspect == "orig":
            return rw / rh if rh else None
        if self.aspect == "free":
            return None
        aw, ah = self.aspect.split(":")
        return float(aw) / float(ah)

    def _drag_crop_to(self, idx: int, n: QPointF):
        c = self.crop if self.crop else [0.05, 0.05, 0.95, 0.95]
        xi, yi = self.HANDLE_MAP[idx]
        ax_i, ay_i = self.OPPOSITE[idx]
        ratio = self._aspect_ratio()
        rw, rh = self._img_size
        if xi is not None:
            c[xi] = min(max(n.x(), 0.0), 1.0)
        if yi is not None:
            c[yi] = min(max(n.y(), 0.0), 1.0)
        if ratio and xi is not None and yi is not None:
            ax = c[ax_i] if ax_i is not None else 0.5
            ay = c[ay_i] if ay_i is not None else 0.5
            w_px = abs(c[xi] - ax) * rw
            h_sign = 1.0 if (c[yi] - ay) >= 0 else -1.0
            new_y = ay + h_sign * ((w_px / ratio) / rh)
            if 0.0 <= new_y <= 1.0:
                c[yi] = new_y
        lo_x, hi_x = min(c[0], c[2]), max(c[0], c[2])
        lo_y, hi_y = min(c[1], c[3]), max(c[1], c[3])
        m = 0.002
        hi_x = max(hi_x, lo_x + m)
        hi_y = max(hi_y, lo_y + m)
        self.crop = [lo_x, lo_y, hi_x, hi_y]

    @staticmethod
    def _clamp_crop_full(c):
        m = 0.002
        w = c[2]-c[0]
        h = c[3]-c[1]
        c[0] = min(max(c[0], 0.0), 1.0-w)
        c[1] = min(max(c[1], 0.0), 1.0-h)
        c[2] = c[0]+w
        c[3] = c[1]+h

    def _hit_spot(self, wp: QPointF):
        best, bi = 1e9, None
        r = self._display_rect()
        for i, sp in enumerate(self.spots):
            pt = self.widget_from_img(sp["cx"], sp["cy"])
            d = math.hypot(pt.x()-wp.x(), pt.y()-wp.y())
            rr = max(self.HANDLE_R, sp.get("r", .03) *
                     min(r.width(), r.height()) * self.zoom)
            if d < rr + 4 and d < best:
                best, bi = d, i
        return bi

    # ================================================================ mouse
    def mousePressEvent(self, e):
        wp = e.position()
        right = e.button() == Qt.RightButton
        if e.button() != Qt.LeftButton and not right:
            return

        if self.mode == "view" and self.split_mode and self._near_split(wp.x()):
            self._dragging_split = True
            e.accept(); return

        if self.mode == "crop":
            hit = self._hit_crop_handle(wp)
            if hit:
                self._crop_drag = hit
            elif not right and self.crop is None:
                n = self.img_from_widget(wp)
                self.crop = [n.x(), n.y(), n.x(), n.y()]
                self._crop_drag = ("c", 4)
                self.cropChanged.emit(list(self.crop))
            e.accept()
            return

        if self.mode == "spot":
            idx = self._hit_spot(wp)
            if right or bool(e.modifiers() & Qt.AltModifier):
                if idx is not None:
                    del self.spots[idx]
                    self.spotsChanged.emit()
                    self.update()
                e.accept()
                return
            n = self.img_from_widget(wp)
            sp = {"cx": float(min(max(n.x(), 0.0), 1.0)),
                  "cy": float(min(max(n.y(), 0.0), 1.0)),
                  "r": float(self.spot_radius_norm), "mode": self.spot_mode}
            if self.spot_mode == "clone":
                sp["sx"] = float(sp["cx"] - 0.08)
                sp["sy"] = sp["cy"]
            self.spots.append(sp)
            self._selected_spot = sp
            self._spot_drag = len(self.spots)-1
            self.spotsChanged.emit()
            self.update()
            e.accept()
            return

        if self.mode == "subject_pick":
            n = self.img_from_widget(wp)
            self.subjectPicked.emit(float(min(max(n.x(), 0), 1)),
                                    float(min(max(n.y(), 0), 1)))
            e.accept()
            return

        if self.mode in ("linear_new", "radial_new"):
            self._draw_start = self.img_from_widget(wp)
            self._draw_cur = self._draw_start
            e.accept()
            return

        if self.mode == "brush":
            n = self.img_from_widget(wp)
            self._stroke_pts = [(n.x(), n.y())]
            e.accept()
            return

        self._dragging_pan = True
        self._press_pos = wp
        self.setCursor(Qt.ClosedHandCursor)
        e.accept()

    def mouseMoveEvent(self, e):
        wp = e.position()
        self._cursor_img = self.img_from_widget(wp)

        if self.mode == "crop" and self._crop_drag:
            kind, idx = self._crop_drag
            n = self.img_from_widget(wp)
            if kind in ("c", "e"):
                self._drag_crop_to(idx, n)
            else:
                prev = getattr(self, "_last_move_img", None)
                if prev is not None and self.crop:
                    dx = n.x() - prev.x()
                    dy = n.y() - prev.y()
                    c = self.crop
                    nc = [c[0]+dx, c[1]+dy, c[2]+dx, c[3]+dy]
                    self._clamp_crop_full(nc)
                    self.crop = nc
            self._last_move_img = n
            self.cropChanged.emit(list(self.crop))
            self.update()
            e.accept()
            return

        if self.mode == "spot" and self._spot_drag is not None:
            n = self.img_from_widget(wp)
            if 0 <= self._spot_drag < len(self.spots):
                sp = self.spots[self._spot_drag]
                sp["cx"] = float(min(max(n.x(), 0.0), 1.0))
                sp["cy"] = float(min(max(n.y(), 0.0), 1.0))
                self.spotsChanged.emit()
            self.update()
            e.accept()
            return

        if self.mode in ("linear_new", "radial_new") and self._draw_start:
            self._draw_cur = self.img_from_widget(wp)
            self.update()
            e.accept()
            return

        if self.mode == "brush" and (e.buttons() & Qt.LeftButton) and self._stroke_pts:
            last = self._stroke_pts[-1]
            d = math.hypot(self._cursor_img.x()-last[0], self._cursor_img.y()-last[1])
            step = self.brush_radius_norm * 0.35
            while d > step:
                t = step / d
                nx = last[0] + (self._cursor_img.x()-last[0])*t
                ny = last[1] + (self._cursor_img.y()-last[1])*t
                self._stroke_pts.append((nx, ny))
                last = (nx, ny)
                d = math.hypot(self._cursor_img.x()-nx, self._cursor_img.y()-ny)
            self.update()
            e.accept()
            return

        if self._dragging_pan:
            self.pan += (wp - self._press_pos)
            self._press_pos = wp
            self.update()
            e.accept()
            return

        if self.mode == "crop" and self.crop is not None:
            hit = self._hit_crop_handle(wp)
            if hit and hit[0] == "c":
                self.setCursor(Qt.SizeAllCursor)
            elif hit:
                self.setCursor(Qt.SplitHCursor if hit[1] in (3, 7)
                               else Qt.SplitVCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        elif self.mode == "brush":
            self.setCursor(Qt.BlankCursor)
        elif self.mode == "spot":
            self.setCursor(Qt.PointingHandCursor)
        elif self.mode == "subject_pick":
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.OpenHandCursor if self.zoom > 1.001 else Qt.ArrowCursor)
        self.update()

    def mouseReleaseEvent(self, e):
        if self._dragging_split:
            self._dragging_split = False
            e.accept(); return
        if self.mode == "crop" and self._crop_drag:
            self._crop_drag = None
            self.cropCommitted.emit()
            e.accept()
            return
        if self._spot_drag is not None:
            self._spot_drag = None
            e.accept()
            return
        if self.mode in ("linear_new", "radial_new") and self._draw_start:
            a, b = self._draw_start, self._draw_cur
            d = math.hypot(b.x()-a.x(), b.y()-a.y())
            if self.mode == "linear_new" and d > 0.02:
                self.maskDrawn.emit("linear", {"x1": a.x(), "y1": a.y(),
                                               "x2": b.x(), "y2": b.y()})
            elif self.mode == "radial_new" and d > 0.02:
                self.maskDrawn.emit("radial", {
                    "cx": (a.x()+b.x())/2, "cy": (a.y()+b.y())/2,
                    "rx": abs(b.x()-a.x())/2 + 0.02,
                    "ry": abs(b.y()-a.y())/2 + 0.02,
                    "feather": 0.35})
            self._draw_start = self._draw_cur = None
            self.update()
            e.accept()
            return
        if self.mode == "brush" and self._stroke_pts:
            stroke = (self._stroke_pts, self.brush_radius_norm, self.brush_flow)
            self._stroke_pts = []
            self.brushStrokeFinished.emit(stroke)
            e.accept()
            return
        self._dragging_pan = False
        self.setCursor(Qt.OpenHandCursor if self.zoom > 1.001 else Qt.ArrowCursor)

    def mouseDoubleClickEvent(self, e):
        if self.mode == "crop":
            self.cropCommitted.emit()

    def wheelEvent(self, e):
        if self.mode == "crop":
            return
        delta = e.angleDelta().y()
        factor = 1.15 if delta > 0 else 1/1.15
        newzoom = min(16.0, max(1.0, self.zoom * factor))
        if newzoom == self.zoom:
            return
        before = self.img_from_widget(e.position())
        self.zoom = newzoom
        after = self.img_from_widget(e.position())
        r = self._display_rect()
        self.pan += QPointF((after.x() - before.x()) * r.width(),
                            (after.y() - before.y()) * r.height())
        self.update()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self.update()
