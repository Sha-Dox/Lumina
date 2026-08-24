"""Professional scopes: RGB histogram, luma, waveform, vectorscope."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

SCOPE_NAMES = ["RGB", "Luma", "Waveform", "Vectorscope"]


class ScopeView(QWidget):
    """Click to cycle between RGB / Luma / Waveform / Vectorscope."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode_idx = 0
        self._hist = None           # list of 3 arrays (RGB hist)
        self._waveform = None       # 2D float (h×w) luminance
        self._uv = None             # (u_vals, v_vals) for vectorscope
        self.setMinimumHeight(110)
        from PySide6.QtWidgets import QSizePolicy
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to cycle: RGB → Luma → Waveform → Vectorscope")

    # ------------------------------------------------------------ data in
    def update_image(self, u8: np.ndarray | None):
        if u8 is None or u8.size == 0:
            return
        small = u8[::max(1, u8.shape[0]//200), ::max(1, u8.shape[1]//200)]
        self._build_hist(small)
        if self.mode_idx == 2:
            self._build_waveform(small)
        elif self.mode_idx == 3:
            self._build_vectorscope(small)
        self.update()

    def _build_hist(self, u8):
        f = u8.astype(np.float32) / 255.0
        luma = f[...,0]*0.2126 + f[...,1]*0.7152 + f[...,2]*0.0722
        hists = []
        for c in range(3):
            h, _ = np.histogram(f[...,c], bins=128, range=(0,1))
            hists.append(np.log1p(h).astype(np.float32))
        hists.append(np.log1p(np.histogram(luma, bins=128, range=(0,1))[0]).astype(np.float32))
        self._hist = hists  # [r,g,b,luma]

    def _build_waveform(self, u8):
        f = u8.astype(np.float32) / 255.0
        luma = f[...,0]*0.2126 + f[...,1]*0.7152 + f[...,2]*0.0722
        h, w = luma.shape
        cols = min(w, 256)
        sample = np.asarray(Image.fromarray(
            (luma*255).astype(np.uint8)).resize((cols, h//2),
            Image.Resampling.NEAREST), dtype=np.float32) / 255.0
        self._waveform = sample

    def _build_vectorscope(self, u8):
        from lumina.core.imaging import rgb_to_hsv
        f = u8.astype(np.float32)/255.0
        h, w = f.shape[:2]
        px = f.reshape(-1,3)
        mx = px.max(axis=-1); mn = px.min(axis=-1)
        sat = (mx-mn) / np.maximum(mx, 1e-4)
        r,g,b = px[:,0], px[:,1], px[:,2]
        # UV-like coordinates
        u = (b - (r+g)/2).clip(-.5,.5)
        v = (r - g).clip(-.5,.5)
        keep = sat > 0.04
        self._uv = (u[keep], v[keep])

    # ------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect().adjusted(1,1,-1,-1)
        p.fillRect(r, QColor("#101010"))

        name = SCOPE_NAMES[self.mode_idx] if self.mode_idx < len(SCOPE_NAMES) else "?"
        p.setPen(QColor("#888888"))
        f = p.font(); f.setPixelSize(9); p.setFont(f)
        p.drawText(r.adjusted(4,2,-4,0), Qt.AlignmentFlag.AlignRight|Qt.AlignmentFlag.AlignTop,
                   name.upper())

        if self.mode_idx == 0 and self._hist:
            self._paint_rgb(p, r)
        elif self.mode_idx == 1 and self._hist:
            self._paint_luma(p, r)
        elif self.mode_idx == 2 and self._waveform is not None:
            self._paint_waveform(p, r)
        elif self.mode_idx == 3 and self._uv is not None:
            self._paint_vectorscope(p, r)
        else:
            # fallback: draw RGB if we have it
            if self._hist:
                self._paint_rgb(p, r)

        p.end()

    def _paint_rgb(self, p: QPainter, r: QRectF):
        if not self._hist or len(self._hist) < 3:
            return
        peak = max(h.max() for h in self._hist[:3]) or 1
        colors = [QColor(220,60,60,140), QColor(70,190,70,140), QColor(70,120,220,140)]
        w = r.width(); bot = r.bottom()
        for hi, col in zip(self._hist[:3], colors):
            path = QPainterPath()
            path.moveTo(r.left(), bot)
            n = len(hi)
            for i in range(n):
                x = r.left() + i/(n-1)*w
                y = bot - float(hi[i]/peak)*(r.height()-14)
                path.lineTo(x, y)
            path.lineTo(r.right(), bot)
            path.closeSubpath()
            p.fillPath(path, col)

    def _paint_luma(self, p: QPainter, r: QRectF):
        if not self._hist or len(self._hist) < 4:
            return
        lum = self._hist[3]
        peak = lum.max() or 1
        path = QPainterPath()
        path.moveTo(r.left(), r.bottom())
        n = len(lum)
        for i in range(n):
            x = r.left() + i/(n-1)*r.width()
            y = r.bottom() - float(lum[i]/peak)*(r.height()-14)
            path.lineTo(x, y)
        path.lineTo(r.right(), r.bottom())
        path.closeSubpath()
        p.fillPath(path, QColor(200,200,200,160))

    def _paint_waveform(self, p: QPainter, r: QRectF):
        wf = self._waveform
        if wf is None:
            return
        h_px = int(min(r.height()-14, wf.shape[0]))
        w_px = int(min(r.width()-4, wf.shape[1]))
        data = wf[:h_px, :w_px]
        img = np.zeros((h_px, w_px, 4), dtype=np.uint8)

        # accumulate brightness per bin
        bins_y = np.linspace(0, 1, 64)
        vis = np.zeros((64, w_px), dtype=np.float32)
        for row in range(h_px):
            yi = min(63, int(data[row] * 63))
            vis[yi] += 1
        peak = vis.max() or 1
        vis_n = np.log1p(vis) / np.log1p(peak)
        for yi in range(64):
            y_img = int((1-yi/63) * (r.height()-16)) + 8
            brightness = int(vis_n[yi].mean() * 255) if vis_n[yi].mean() > 0 else 0
            brightness = max(brightness, int(vis_n[yi].max()*255))
            if brightness > 5:
                yy = r.top() + 10 + y_img
                p.setPen(QColor(80+brightness//2, 180, 80, min(255,brightness+40)))
                p.drawLine(QPointF(r.left()+2, yy), QPointF(r.right()-2, yy))

    def _paint_vectorscope(self, p: QPainter, r: QRectF):
        cx = r.center().x()
        cy = r.center().y() + 5
        radius = min(r.width(), r.height()) / 2 - 12

        # graticule circle
        p.setPen(QColor(255,255,255,25))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPointF(cx,cy), radius, radius)
        p.drawEllipse(QPointF(cx,cy), radius/2, radius/2)
        p.drawLine(QPointF(cx-radius, cy), QPointF(cx+radius, cy))
        p.drawLine(QPointF(cx, cy-radius), QPointF(cx, cy+radius))

        uv = self._uv
        if uv is None or len(uv[0]) == 0:
            return
        us, vs = uv
        step = max(1, len(us)//800)
        for i in range(0, len(us), step):
            dx = us[i] * radius * 2
            dy = -vs[i] * radius * 2
            mag = math.hypot(dx,dy)
            if mag > radius: dx *= radius/mag; dy *= radius/mag
            alpha = min(200, 40 + int(mag/radius * 180))
            p.setPen(QColor(100,255,150,alpha))
            p.drawPoint(QPointF(cx+dx, cy+dy))

    # ------------------------------------------------------------ mouse
    def mousePressEvent(self, e):
        self.mode_idx = (self.mode_idx + 1) % len(SCOPE_NAMES)
        if self.mode_idx == 2 and self._waveform is None:
            pass  # will be built on next update_image call
        elif self.mode_idx == 3 and self._uv is None:
            pass
        self.update()
