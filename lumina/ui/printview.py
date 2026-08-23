"""Print module — page layouts, preview, printer + PDF output."""
from __future__ import annotations

import os

from PySide6.QtCore import Signal, QPointF, QRect, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPixmap
from PySide6.QtWidgets import (QComboBox, QFileDialog, QHBoxLayout, QLabel,
                               QPushButton, QRadioButton, QSpinBox, QWidget,
                               QVBoxLayout)

from ..core import catalog, rawio

PAGE_SIZES_MM = {
    "A4 (210×297)": (210.0, 297.0),
    "Letter (216×279)": (215.9, 279.4),
    "A3 (297×420)": (297.0, 420.0),
    '5×7" (127×178)': (127.0, 177.8),
    '8×10" (203×254)': (203.2, 254.0),
    '4×6" (102×152)': (101.6, 152.4),
}
TEMPLATES = ["Full Page", "2-up", "4-up", "2×3", "3×3"]


def template_grid(name: str):
    if name == "2-up":
        return 2, 1
    if name == "4-up":
        return 2, 2
    if name == "2×3":
        return 2, 3      # cols, rows
    if name == "3×3":
        return 3, 3
    return 1, 1


class PagePreview(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        self.render_fn = None       # callable(painter, rect_px)
        self.page_info = ""

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#191919"))
        pw, ph = self._page_mm()
        scale = min((self.width() - 40) / pw, (self.height() - 40) / ph)
        w_px, h_px = pw * scale, ph * scale
        x = (self.width() - w_px) / 2
        y = (self.height() - h_px) / 2
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(QRectF(x, y, w_px, h_px), QColor("#ffffff"))
        p.setPen(QColor("#333333"))
        p.drawRect(QRectF(x, y, w_px, h_px))
        if self.render_fn:
            p.save()
            p.translate(x, y)
            p.scale(scale, scale)
            self.render_fn(p, pw, ph)     # painter in mm units
            p.restore()
        p.setPen(QColor("#777777"))
        p.drawText(self.rect().adjusted(0, 4, 0, 0),
                   Qt.AlignHCenter | Qt.AlignTop, self.page_info)
        p.end()

    def _page_mm(self):
        return getattr(self, "_pw", 210.0), getattr(self, "_ph", 297.0)


class PrintView(QWidget):
    statusMessage = Signal(str)

    def __init__(self, parent=None, filmstrip_ids=None):
        super().__init__(parent)
        self.filmstrip_ids = filmstrip_ids or []
        self.page_index = 0

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # ---------------- left controls
        left = QWidget()
        left.setObjectName("SidePanel")
        left.setFixedWidth(240)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(12, 12, 12, 12)
        lv.setSpacing(8)

        t1 = QLabel("LAYOUT")
        t1.setObjectName("SectionTitle")
        lv.addWidget(t1)
        self.template_combo = QComboBox()
        self.template_combo.addItems(TEMPLATES)
        self.template_combo.currentTextChanged.connect(self._changed)
        lv.addWidget(self.template_combo)

        t2 = QLabel("PAGE SIZE")
        t2.setObjectName("SectionTitle")
        lv.addWidget(t2)
        self.size_combo = QComboBox()
        self.size_combo.addItems(PAGE_SIZES_MM.keys())
        self.size_combo.setCurrentIndex(0)
        self.size_combo.currentTextChanged.connect(self._changed)
        lv.addWidget(self.size_combo)

        orow = QHBoxLayout()
        self.rb_portrait = QRadioButton("Portrait")
        self.rb_portrait.setChecked(True)
        self.rb_landscape = QRadioButton("Landscape")
        for rb in (self.rb_portrait, self.rb_landscape):
            rb.toggled.connect(self._changed)
            orow.addWidget(rb)
        lv.addLayout(orow)

        mrow = QHBoxLayout()
        mrow.addWidget(QLabel("Margin"))
        self.margin_spin = QSpinBox()
        self.margin_spin.setRange(0, 40)
        self.margin_spin.setValue(10)
        self.margin_spin.setSuffix(" mm")
        self.margin_spin.valueChanged.connect(self._changed)
        mrow.addWidget(self.margin_spin)
        mrow.addWidget(QLabel("Gutter"))
        self.gutter_spin = QSpinBox()
        self.gutter_spin.setRange(0, 30)
        self.gutter_spin.setValue(5)
        self.gutter_spin.setSuffix(" mm")
        self.gutter_spin.valueChanged.connect(self._changed)
        mrow.addWidget(self.gutter_spin)
        lv.addLayout(mrow)

        nav = QHBoxLayout()
        b_prev = QPushButton("< Prev")
        b_next = QPushButton("Next >")
        b_prev.clicked.connect(lambda: self.change_page(-1))
        b_next.clicked.connect(lambda: self.change_page(1))
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        lv.addLayout(nav)
        lv.addStretch(1)

        b_print = QPushButton("Print…")
        b_print.setObjectName("Primary")
        b_print.clicked.connect(self.do_print)
        lv.addWidget(b_print)
        b_pdf = QPushButton("Export PDF…")
        b_pdf.clicked.connect(self.export_pdf)
        lv.addWidget(b_pdf)
        h.addWidget(left)

        # ---------------- preview
        self.preview = PagePreview()
        self.preview.render_fn = self.render_page
        h.addWidget(self.preview, 1)
        self._thumb_cache: dict[int, QPixmap] = {}
        self._changed()

    # ------------------------------------------------------------ helpers
    def set_photos(self, ids):
        self.filmstrip_ids = list(ids or [])
        self.page_index = 0
        self._changed()

    def _page_size_mm(self):
        pw, ph = PAGE_SIZES_MM[self.size_combo.currentText()]
        if self.rb_landscape.isChecked():
            pw, ph = ph, pw
        self.preview._pw, self.preview._ph = pw, ph
        return pw, ph

    def _cells_per_page(self):
        cols, rows_ = template_grid(self.template_combo.currentText())
        return cols * rows_

    def _pages(self):
        n = max(1, len(self.filmstrip_ids))
        per = self._cells_per_page()
        return max(1, (n + per - 1) // per)

    def change_page(self, d: int):
        self.page_index = max(0, min(self._pages() - 1, self.page_index + d))
        self._changed()

    def _changed(self):
        pw, ph = self._page_size_mm()
        self.preview.page_info = (
            f"{self.template_combo.currentText()} · {self.size_combo.currentText()} · "
            f"page {self.page_index+1}/{self._pages()} · "
            f"{len(self.filmstrip_ids)} photos queued")
        self.preview.update()

    def _photo_for_cell(self, idx: int):
        start = self.page_index * self._cells_per_page()
        pid_at = start + idx
        if pid_at >= len(self.filmstrip_ids):
            return None
        pid = self.filmstrip_ids[pid_at]
        row = catalog.get_photo(pid)
        if row is None:
            return None
        pm = self._thumb_cache.get(pid)
        if pm is None:
            tp = rawio.make_thumbnail(row["path"], 600)
            pm = QPixmap(tp) if tp else QPixmap()
            self._thumb_cache[pid] = pm
        return row, pm

    # ------------------------------------------------------------ rendering
    def render_page(self, p: QPainter, pw_mm: float, ph_mm: float):
        """Draw into a painter scaled so 1 unit == 1 mm."""
        margin = float(self.margin_spin.value())
        gutter = float(self.gutter_spin.value())
        cols, rows_ = template_grid(self.template_combo.currentText())
        cw = (pw_mm - margin * 2 - gutter * (cols - 1)) / cols
        chh = (ph_mm - margin * 2 - gutter * (rows_ - 1)) / rows_
        for r in range(rows_):
            for c in range(cols):
                idx = r * cols + c
                cell = QRectF(margin + c * (cw + gutter),
                              margin + r * (chh + gutter), cw, chh)
                item = self._photo_for_cell(idx)
                if item is None:
                    continue
                row, pm = item
                if pm.isNull():
                    continue
                src = pm.rect()
                tgt_ar = cell.width() / cell.height()
                src_ar = src.width() / src.height()
                if src_ar > tgt_ar:      # crop sides
                    nw = int(src.height() * tgt_ar)
                    sx = (src.width() - nw) // 2
                    srcrect = QRect(sx, 0, nw, src.height())
                else:                    # crop top/bottom
                    nh = int(src.width() / tgt_ar)
                    sy = (src.height() - nh) // 2
                    srcrect = QRect(0, sy, src.width(), nh)
                p.drawPixmap(cell, pm, srcrect)

    def _paint_target(self, printer_like):
        p = QPainter(printer_like)
        pw_mm = printer_like.widthMM()
        ph_mm = printer_like.heightMM()
        self.render_page(p, float(pw_mm), float(ph_mm))
        p.end()

    def do_print(self):
        from PySide6.QtPrintSupport import QPrintDialog, QPrinter
        printer = QPrinter(QPrinter.HighResolution)
        dlg = QPrintDialog(printer, self)
        if dlg.exec() != QPrintDialog.Accepted:
            return
        pages = self._pages()
        save_idx = self.page_index
        for pg in range(pages):
            self.page_index = pg
            if pg > 0:
                printer.newPage()
            self._paint_target(printer)
        self.page_index = save_idx
        self.statusMessage.emit(f"Sent {pages} page(s) to printer")

    def export_pdf(self):
        from PySide6.QtGui import QPdfWriter, QPageSize
        dest_dir = os.path.expanduser("~/Pictures/Lumina Prints")
        os.makedirs(dest_dir, exist_ok=True)
        path, _ = QFileDialog.getSaveFileName(
            self, "Export PDF", os.path.join(dest_dir, "lumina_print.pdf"),
            "PDF (*.pdf)")
        if not path:
            return
        writer = QPdfWriter(path)
        pw, ph = self._page_size_mm()
        from PySide6.QtCore import QSizeF
        writer.setPageSize(QPageSize(QSizeF(float(pw), float(ph)),
                                     QPageSize.Unit.Millimeter))
        writer.setResolution(300)
        pages = self._pages()
        save_idx = self.page_index
        p = QPainter(writer)
        for pg in range(pages):
            self.page_index = pg
            if pg > 0:
                writer.newPage()
            self.render_page(p, float(writer.width()) / writer.logicalDpiX() * 25.4,
                             float(writer.height()) / writer.logicalDpiY() * 25.4)
        p.end()
        self.page_index = save_idx
        self.statusMessage.emit(f"PDF exported: {path}")
