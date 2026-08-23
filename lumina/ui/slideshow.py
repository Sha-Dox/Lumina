"""Slideshow module - fullscreen playback with crossfades + Ken Burns."""
from __future__ import annotations

from PySide6.QtCore import (QEasingCurve, QPropertyAnimation, Qt, QTimer)
from PySide6.QtWidgets import (QComboBox, QGraphicsOpacityEffect, QHBoxLayout,
                               QLabel, QPushButton, QWidget)

from ..core import catalog, export as excore, imaging
from .develop_canvas import np_to_pixmap


class SlideshowWindow(QWidget):

    def __init__(self, photo_rows, screen_size, parent=None):
        super().__init__(None,
                         Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Lumina Slideshow")
        self._zoom_anim = None
        self._fade_anim = None
        self.setWindowState(Qt.WindowState.WindowFullScreen)
        self.setStyleSheet("background:#000;")
        from PySide6.QtCore import Signal
        if not hasattr(type(self), "closed_sig"):
            pass

        self.rows = list(photo_rows)
        self.idx = 0
        self.screen_w, self.screen_h = screen_size
        self.duration_s = 4.0
        self.ken_burns = True
        self._playing = True
        self._fading = False
        self._prefetched = {}

        self.a = QLabel(self)
        self.b = QLabel(self)
        for lab in (self.a, self.b):
            lab.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lab.setGeometry(0, 0, self.screen_w, self.screen_h)
            lab.setStyleSheet("background:#000;")
            eff = QGraphicsOpacityEffect(lab)
            eff.setOpacity(1.0)
            lab.setGraphicsEffect(eff)
        self._front = self.a
        self._back = self.b

        bar = QWidget(self)
        bar.setStyleSheet("background:rgba(18,18,18,215); border-radius:8px;")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(12, 6, 12, 6)
        b_prev = QPushButton("\u23ee")
        self.b_play = QPushButton("\u23f8")
        b_next = QPushButton("\u23ed")
        for b in (b_prev, self.b_play, b_next):
            b.setFixedSize(40, 28)
            bl.addWidget(b)
        bl.addWidget(QLabel("Hold"))
        self.dur_combo = QComboBox()
        self.dur_combo.addItems(["2 s", "4 s", "7 s", "12 s"])
        self.dur_combo.setCurrentIndex(1)
        bl.addWidget(self.dur_combo)
        self.kb_toggle = QPushButton("Ken Burns")
        self.kb_toggle.setCheckable(True)
        self.kb_toggle.setChecked(True)
        bl.addWidget(self.kb_toggle)
        hint = QLabel("Esc exits · \u2190 \u2192 · Space pause")
        hint.setStyleSheet("color:#999;")
        bl.addWidget(hint)
        bar.adjustSize()
        bar.move((self.screen_w - bar.width()) // 2, self.screen_h - 58)
        self._bar = bar

        b_prev.clicked.connect(lambda: self.step(-1))
        b_next.clicked.connect(lambda: self.step(1))
        self.b_play.clicked.connect(self.toggle_play)
        self.dur_combo.currentTextChanged.connect(
            lambda t: setattr(self, "duration_s", float(t.split()[0])))
        self.kb_toggle.toggled.connect(lambda on: setattr(self, "ken_burns", on))

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self._auto_next)

        if self.rows:
            pm = self._render_photo(self.rows[0])
            if pm is not None:
                fitted = self._fit(pm)
                self.a.setPixmap(fitted)
                self._apply_kenburns(self.a, first=True)
            QTimer.singleShot(120, lambda: self._prefetch(1))
            self.timer.start(int(self.duration_s * 1000))

    # ------------------------------------------------------------ rendering
    @staticmethod
    def _as_id(row_or_id):
        if isinstance(row_or_id, int):
            return row_or_id
        try:
            return int(row_or_id["id"])
        except Exception:
            try:
                return int(getattr(row_or_id, "id"))
            except Exception:
                return None

    def _render_photo(self, row_or_id):
        key = self._as_id(row_or_id)
        if key in self._prefetched:
            return self._prefetched.pop(key)
        try:
            settings = imaging.sanitize_settings(catalog.load_settings(key) or {})
            u8 = excore.render_for_export(
                catalog.get_photo(key)["path"], settings,
                max_edge=max(self.screen_w, self.screen_h),
                output_sharpen=False)
            return np_to_pixmap(u8)
        except Exception as e:
            print("[slideshow]", e)
            return None

    def _fit(self, pm):
        return pm.scaled(self.screen_w, self.screen_h,
                         Qt.AspectRatioMode.KeepAspectRatio,
                         Qt.TransformationMode.SmoothTransformation)

    def _kb_geometry(self, pm, t):
        """Ken Burns: zoom 1.02 -> 1.14 with slight drift."""
        f = 1.02 + 0.12 * max(0.0, min(1.0, t))
        w = int(pm.width() * f)
        h = int(pm.height() * f)
        x = -(w - self.screen_w) // 2 + int(-self.screen_w * 0.04 * t)
        y = -(h - self.screen_h) // 2 + int(-self.screen_h * 0.03 * t)
        return x, y, w, h

    def _apply_kenburns(self, label, first=False):
        pm = label.pixmap()
        if not self.ken_burns or pm is None or pm.isNull():
            label.setGeometry(0, 0, self.screen_w, self.screen_h)
            return
        x, y, w, h = self._kb_geometry(pm, 0.0 if first else 0.0)
        ex, ey, ew, eh = self._kb_geometry(pm, 1.0)
        anim = QPropertyAnimation(label, b"geometry", self)
        anim.setDuration(int(self.duration_s * 1000) + 500)
        anim.setStartValue(__import__("PySide6.QtCore",
                                      fromlist=["QRect"]).QRect(x, y, w, h))
        anim.setEndValue(__import__("PySide6.QtCore",
                                    fromlist=["QRect"]).QRect(ex, ey, ew, eh))
        anim.setEasingCurve(QEasingCurve.Type.Linear)
        anim.start()
        self._zoom_anim = anim

    def _prefetch(self, offset=1):
        if not self.rows:
            return
        rid = self.rows[(self.idx + offset) % len(self.rows)]["id"]
        if rid in self._prefetched:
            return
        row = catalog.get_photo(rid)
        if row is None:
            return
        pm = self._render_photo(rid)
        if pm is not None:
            self._prefetched[rid] = pm

    # ------------------------------------------------------------ playback
    def _auto_next(self):
        self.step(1)

    def step(self, direction: int):
        if len(self.rows) < 2:
            return
        if self._fading:
            return
        self._fading = True
        self.timer.stop()
        self.idx = (self.idx + direction) % len(self.rows)
        pm = self._render_photo(self.rows[self.idx])
        if pm is None:
            self._fading = False
            self.timer.start(600)
            return

        front, back = self._front, self._back
        back.setPixmap(self._fit(pm))
        back.setGeometry(0, 0, self.screen_w, self.screen_h)
        back.show()
        back.raise_()
        self._bar.raise_()
        self._front, self._back = back, front
        self._apply_kenburns(back)

        from PySide6.QtWidgets import QGraphicsOpacityEffect
        eff_f = front.graphicsEffect()
        eff_b = back.graphicsEffect()
        if not isinstance(eff_b, QGraphicsOpacityEffect):
            return
        eff_b.setOpacity(1.0)

        if not isinstance(eff_f, QGraphicsOpacityEffect):
            self._fading = False
            return
        fade_out = QPropertyAnimation(eff_f, b"opacity", self)
        fade_out.setDuration(650)
        fade_out.setStartValue(1.0)
        fade_out.setEndValue(0.0)
        fade_out.setEasingCurve(QEasingCurve.Type.InOutQuad)
        try:
            fade_out.finished.disconnect()
        except Exception:
            pass

        def done():
            front.hide()
            if isinstance(eff_f, QGraphicsOpacityEffect):
                pass
            eff_f.setOpacity(1.0)
            self._fading = False
            if self._playing:
                self.timer.start(int(self.duration_s * 1000))
            QTimer.singleShot(80, lambda: self._prefetch(1))
        fade_out.finished.connect(done)
        fade_out.start()
        self._fade_anim = fade_out

    def toggle_play(self):
        self._playing = not self._playing
        self.b_play.setText("\u25b6" if not self._playing else "\u23f8")
        if self._playing:
            self.timer.start(int(self.duration_s * 1000))
        else:
            self.timer.stop()

    # ------------------------------------------------------------ events
    def keyPressEvent(self, e):
        k = e.key()
        if k == Qt.Key.Key_Escape:
            self.close()
        elif k == Qt.Key.Key_Space:
            self.toggle_play()
        elif k == Qt.Key.Key_Right:
            self.step(1)
        elif k == Qt.Key.Key_Left:
            self.step(-1)

    def mouseDoubleClickEvent(self, e):
        self.close()

    def closeEvent(self, e):
        self.timer.stop()
        super().closeEvent(e)


