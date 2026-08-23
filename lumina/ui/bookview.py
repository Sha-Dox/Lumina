"""Book module - themed photo books exported as PDF."""
from __future__ import annotations

import os

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPdfWriter, QPageSize
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QLineEdit, QPushButton, QVBoxLayout, QWidget)

from ..core import catalog, rawio


class BookPreview(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 320)
        self.render_fn = None
        self.info = ""

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#191919"))
        pw, ph = 210.0, 210.0     # square spread preview in mm units
        scale = min((self.width()-40)/pw, (self.height()-40)/ph)
        x = (self.width()-pw*scale)/2
        y = (self.height()-ph*scale)/2
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(QRectF(x, y, pw*scale, ph*scale), QColor("#ffffff"))
        if self.render_fn:
            p.save(); p.translate(x, y); p.scale(scale, scale)
            self.render_fn(p, pw, ph)
            p.restore()
        p.setPen(QColor("#777777"))
        p.drawText(self.rect().adjusted(0, 4, 0, 0),
                   Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop,
                   self.info)
        p.end()


THEMES = {
    "Classic": {"bg": (255, 255, 255), "text": "#222222", "font_px": 26},
    "Magazine": {"bg": (18, 18, 18), "text": "#eeeeee", "font_px": 30},
    "Minimal": {"bg": (248, 246, 242), "text": "#333333", "font_px": 22},
}


class BookView(QWidget):
    statusMessage = None   # set by app via set_status_fn

    def __init__(self, parent=None):
        super().__init__(parent)
        self.filmstrip_ids = []
        self.page_index = 0
        self._thumbs = {}

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        left = QWidget()
        left.setObjectName("SidePanel")
        left.setFixedWidth(250)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(8)

        t1 = QLabel("BOOK THEME")
        t1.setObjectName("SectionTitle")
        lv.addWidget(t1)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(THEMES.keys())
        self.theme_combo.currentTextChanged.connect(lambda _t: self.refresh())
        lv.addWidget(self.theme_combo)

        t2 = QLabel("TITLE PAGE")
        t2.setObjectName("SectionTitle")
        lv.addWidget(t2)
        self.title_edit = QLineEdit("My Photo Book")
        lv.addWidget(self.title_edit)
        self.sub_edit = QLineEdit("Made with Lumina")
        lv.addWidget(self.sub_edit)

        per_row = QHBoxLayout()
        per_row.addWidget(QLabel("Photos/page"))
        self.per_combo = QComboBox()
        self.per_combo.addItems(["1", "2", "4"])
        self.per_combo.setCurrentIndex(0)
        self.per_combo.currentIndexChanged.connect(lambda _i: self.refresh())
        per_row.addWidget(self.per_combo)
        lv.addLayout(per_row)

        nav = QHBoxLayout()
        b_prev = QPushButton("< Prev")
        b_next = QPushButton("Next >")
        b_prev.clicked.connect(lambda: self.change_page(-1))
        b_next.clicked.connect(lambda: self.change_page(1))
        nav.addWidget(b_prev); nav.addWidget(b_next)
        lv.addLayout(nav)
        lv.addStretch(1)

        b_pdf = QPushButton("Export Book PDF…")
        b_pdf.setObjectName("Primary")
        b_pdf.clicked.connect(self.export_pdf)
        lv.addWidget(b_pdf)
        h.addWidget(left)

        self.preview = BookPreview()
        self.preview.render_fn = self.render_page
        h.addWidget(self.preview, 1)
        self.refresh()

    def set_status_fn(self, fn):
        self.statusMessage = fn

    def set_photos(self, ids):
        self.filmstrip_ids = list(ids or [])
        self.page_index = 0
        self._thumbs.clear()
        self.refresh()

    def _pages(self):
        per = int(self.per_combo.currentText())
        return max(1, -(-len(self.filmstrip_ids) // per) + 1)   # +cover

    def change_page(self, d):
        self.page_index = max(0, min(self._pages()-1, self.page_index + d))
        self.refresh()

    def _theme(self):
        return THEMES[self.theme_combo.currentText()]

    def _thumb(self, pid):
        pm = self._thumbs.get(pid)
        if pm is None:
            row = catalog.get_photo(pid)
            tp = rawio.make_thumbnail(row["path"], 700) if row else None
            from PySide6.QtGui import QPixmap
            pm = QPixmap(tp) if tp else QPixmap()
            self._thumbs[pid] = pm
        return pm

    def refresh(self):
        self.preview.info = (f"{self.theme_combo.currentText()} · "
                             f"page {self.page_index+1}/{self._pages()}")
        self.preview.update()

    def render_page(self, p: QPainter, w_mm: float, h_mm: float):
        """p is scaled so units are mm."""
        theme = self._theme()
        bg = theme["bg"]
        p.fillRect(QRectF(0, 0, w_mm, h_mm),
                   QColor(bg[0], bg[1], bg[2]))
        title_f = QFont(); title_f.setPixelSize(int(theme["font_px"]*3))
        cap_f = QFont(); cap_f.setPixelSize(int(theme["font_px"]))

        # cover page
        if self.page_index == 0:
            p.setPen(QColor(theme["text"]))
            p.setFont(title_f)
            p.drawText(QRectF(0, h_mm*0.38, w_mm, 20),
                       Qt.AlignmentFlag.AlignCenter, self.title_edit.text())
            p.setFont(cap_f)
            p.drawText(QRectF(0, h_mm*0.38+22, w_mm, 10),
                       Qt.AlignmentFlag.AlignCenter, self.sub_edit.text())
            if self.filmstrip_ids:
                pm = self._thumb(self.filmstrip_ids[0])
                if pm and not pm.isNull():
                    side = min(w_mm*0.5, h_mm*0.28)
                    src = pm.rect()
                    tgt_ar = 1.0
                    nw = min(src.width(), int(src.height()*tgt_ar))
                    sx = (src.width()-nw)//2
                    tx = (w_mm-side)/2
                    p.drawPixmap(QRectF(tx, h_mm*0.08, side, side),
                                 pm, QRectF(sx, 0, nw, src.height()))
            return

        per = int(self.per_combo.currentText())
        start = (self.page_index-1) * per
        margin = 12.0
        gap = 6.0
        if per == 1:
            cells = [QRectF(margin, margin, w_mm-2*margin, h_mm-2*margin)]
        elif per == 2:
            cw = (w_mm-2*margin-gap)/2
            cells = [QRectF(margin, margin, cw, h_mm-2*margin),
                     QRectF(margin+cw+gap, margin, cw, h_mm-2*margin)]
        else:
            cw = (w_mm-2*margin-gap)/2
            chh = (h_mm-2*margin-gap)/2
            cells = [QRectF(margin, margin, cw, chh),
                     QRectF(margin+cw+gap, margin, cw, chh),
                     QRectF(margin, margin+chh+gap, cw, chh),
                     QRectF(margin+cw+gap, margin+chh+gap, cw, chh)]
        for i, cell in enumerate(cells):
            idx = start + i
            if idx >= len(self.filmstrip_ids):
                break
            pm = self._thumb(self.filmstrip_ids[idx])
            if pm.isNull():
                continue
            src = pm.rect()
            tgt_ar = cell.width()/cell.height()
            src_ar = src.width()/src.height()
            if src_ar > tgt_ar:
                nw = int(src.height()*tgt_ar)
                srcrect = __import__("PySide6.QtCore",
                                     fromlist=["QRect"]).QRect(
                    (src.width()-nw)//2, 0, nw, src.height())
            else:
                nh = int(src.width()/tgt_ar)
                srcrect = __import__("PySide6.QtCore",
                                     fromlist=["QRect"]).QRect(
                    0, (src.height()-nh)//2, src.width(), nh)
            p.drawPixmap(cell, pm, srcrect)
            row = catalog.get_photo(self.filmstrip_ids[idx])
            if row:
                p.setFont(cap_f)
                p.setPen(QColor(theme["text"]))
                p.drawText(QRectF(cell.left(), cell.bottom()+1.5,
                                  cell.width(), 5),
                           Qt.AlignmentFlag.AlignCenter, row["filename"])

    def export_pdf(self):
        if not self.filmstrip_ids:
            return
        dest_dir = os.path.expanduser("~/Pictures/Lumina Books")
        os.makedirs(dest_dir, exist_ok=True)
        path, _ = __import__("PySide6.QtWidgets", fromlist=["QFileDialog"])\
            .QFileDialog.getSaveFileName(
                self, "Export book PDF",
                os.path.join(dest_dir,
                             f"{self.title_edit.text() or 'book'}.pdf"),
                "PDF (*.pdf)")
        if not path:
            return
        writer = QPdfWriter(path)
        from PySide6.QtCore import QSizeF
        writer.setPageSize(QPageSize(QSizeF(210.0, 210.0),
                                     QPageSize.Unit.Millimeter))
        writer.setResolution(150)
        save_idx = self.page_index
        p = QPainter(writer)
        for pg in range(self._pages()):
            self.page_index = pg
            if pg > 0:
                writer.newPage()
            self.render_page(p, float(writer.width())/writer.logicalDpiX()*25.4,
                             float(writer.height())/writer.logicalDpiY()*25.4)
        p.end()
        self.page_index = save_idx
        if callable(getattr(self, "statusMessage", None)):
            self.statusMessage(f"Book exported: {path}")
