"""Lumina main window — modules, filmstrip, shortcuts, wiring."""
from __future__ import annotations

import os

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtGui import QAction, QGuiApplication, QIcon, QKeySequence, QPainter, QPixmap, QColor
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QMainWindow, QPushButton,
                               QStackedWidget, QVBoxLayout, QWidget)

from ..core import catalog, rawio, export as excore
from . import theme
from ..core import imaging as im_mod
import copy as _copymod
from .brand import AboutDialog, draw_logo
from .develop import DevelopView
from .dialogs import ExportDialog, ImportDialog
from .filmstrip import Filmstrip
from .library import LibraryView
from .merge_dialogs import MergeDialog
from .slideshow import SlideshowWindow
from .sync_dialog import SyncDialog
from .cull_dialog import CullDialog
from .tether import TetherDialog


def app_icon() -> QIcon:
    try:
        from .brand import app_icon as brand_icon
        return brand_icon()
    except Exception:
        return QIcon()


class ModuleTabs(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setObjectName("ModuleTab")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)


class LuminaWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        catalog.init_db()
        self.setWindowTitle("Lumina")
        self.setWindowIcon(app_icon())
        QTimer.singleShot(50, self._ensure_dock_icon)
        self.resize(1680, 1020)
        self._current_pid = None

        central = QWidget()
        central.setObjectName("RootWindow")
        v = QVBoxLayout(central)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.setCentralWidget(central)

        # ---------------- top bar
        top = QWidget()
        top.setObjectName("TopBar")
        top.setFixedHeight(46)
        tl = QHBoxLayout(top)
        tl.setContentsMargins(14, 0, 14, 0)
        logo_lbl = QLabel()
        logo_pm = draw_logo(44)
        if not logo_pm.isNull():
            logo_lbl.setPixmap(logo_pm)
        tl.addWidget(logo_lbl)
        brand = QLabel("LUMINA")
        brand.setObjectName("BrandLabel")
        tl.addWidget(brand)
        ver = QLabel("RAW photo editor")
        ver.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:10px;")
        tl.addWidget(ver)
        tl.addStretch(1)

        self.tabs = [ModuleTabs("Library"), ModuleTabs("Develop"),
                     ModuleTabs("Print"), ModuleTabs("Book")]
        for i, tb in enumerate(self.tabs):
            tl.addWidget(tb, 0, Qt.AlignVCenter)
            tb.clicked.connect(lambda _=False, ix=i: self.switch_module(ix))
        self.tabs[0].setChecked(True)
        tl.addStretch(1)

        b_about = QPushButton("ⓘ")
        b_about.setToolTip("About Lumina")
        b_about.setStyleSheet("border:none; background:transparent; font-size:17px;")
        b_about.setCursor(Qt.PointingHandCursor)
        b_about.clicked.connect(lambda: AboutDialog(self).exec())
        tl.addWidget(b_about)
        b_slideshow = QPushButton("▶ Slideshow")
        b_slideshow.setToolTip("Play the current filmstrip fullscreen")
        b_slideshow.clicked.connect(self.do_slideshow)
        tl.addWidget(b_slideshow)
        b_gallery = QPushButton("Web")
        b_gallery.setToolTip("Export a self-contained HTML web gallery")
        b_gallery.clicked.connect(self.do_web_gallery)
        tl.addWidget(b_gallery)
        b_import = QPushButton("Import…")
        b_export = QPushButton("Export…")
        b_export.setObjectName("Primary")
        b_import.clicked.connect(self.do_import)
        b_export.clicked.connect(self.do_export)
        tl.addWidget(b_import)
        tl.addWidget(b_export)
        v.addWidget(top)

        # ---------------- stacked modules + filmstrip
        # Heavy views are constructed lazily on first visit (fast launch).
        self.stack = QStackedWidget()
        self.library = LibraryView()
        self._views = {0: self.library}
        self.develop = None
        self.print_view = None
        self.book_view = None
        self.stack.addWidget(self.library)      # 0
        v.addWidget(self.stack, 1)

        self.filmstrip = Filmstrip()
        v.addWidget(self.filmstrip)

        self.statusBar().showMessage(
            "Import a folder to begin · G grid · D develop · double-click a photo to edit")

        self._wire()
        self._shortcuts()

        QTimer.singleShot(120, self._startup_refresh)
        QTimer.singleShot(250, self._prewarm_engine)

    def _prewarm_engine(self):
        """Compile the numba kernel + warm cv2 in the background so the
        first slider drag is instant."""
        def work():
            try:
                import numpy as np
                from lumina.core.imaging import render_global
                tiny = np.full((48, 64, 3), 0.5, dtype=np.float32)
                s = {"temp": 5, "tint": -2, "exposure": 0.3, "contrast": 10,
                     "highlights": -10, "shadows": 12, "whites": 5,
                     "blacks": -5, "vibrance": 8, "saturation": 0,
                     "clarity": 6, "dehaze": 0, "sharp_amount": 20,
                     "sharp_radius": 1.2, "nr_lum": 8, "nr_color": 8,
                     "bw": False, "curve_rgb": [[0.3, 0.28], [0.7, 0.72]],
                     "curve_r": [], "curve_g": [], "curve_b": [],
                     "hsl": {b: [0, 0, 0] for b in
                             ["red","orange","yellow","green","aqua",
                              "blue","purple","magenta"]},
                     "grade_shadows": [210, 15, -3], "grade_midtones": [0, 0, 0],
                     "grade_highlights": [35, 10, 2], "blender": 50,
                     "balance": 0, "vignette_amount": -12,
                     "vignette_midpoint": 50, "vignette_feather": 60,
                     "grain_amount": 5, "grain_size": 25}
                for _ in range(2):
                    render_global(tiny, s, scale=0.3, seed_key="warm")
            except Exception as e:
                print("[prewarm]", e)
        import threading
        threading.Thread(target=work, daemon=True).start()

    def _ensure_dock_icon(self):
        try:
            import os
            png = os.path.expanduser("~/.lumina/brand/logo512.png")
            if not os.path.exists(png):
                draw_logo(512).save(png, "PNG")
            from AppKit import NSApplication, NSImage
            from Foundation import NSURL
            nsimg = NSImage.alloc().initWithContentsOfURL_(
                NSURL.fileURLWithPath_(png))
            if nsimg:
                NSApplication.sharedApplication().setApplicationIconImage_(nsimg)
        except Exception:
            pass

    def _startup_refresh(self):
        removed = catalog.prune_missing()
        self.library.rebuild_collections()
        self.library.rebuild_folders()
        rows = self.library.refresh()
        self.filmstrip.load_photos(rows)
        if removed:
            self.statusBar().showMessage(f"Removed {removed} missing photos from catalog")

    # ------------------------------------------------------------ wiring
    def _wire(self):
        lib, dev, strip = self.library, self.develop, self.filmstrip

        lib.openPhoto.connect(self.open_in_develop)
        lib.selectionChangedId.connect(self._library_selected)
        lib.photoEdited.connect(strip.refresh_thumb)
        lib.importRequested.connect(self.do_import)
        lib.hdrRequested.connect(self.do_hdr_merge)
        lib.panoRequested.connect(self.do_pano)
        lib.tetherRequested.connect(self.do_tether)
        lib.syncRequested.connect(self.do_sync_settings)
        lib.cullRequested.connect(self.do_auto_cull)
        lib.dupsRequested.connect(self.do_find_dups)
        lib.statusMessage.connect(self.statusBar().showMessage)

        strip.selectionChangedId.connect(self._strip_selected)
        strip.openRequested.connect(self.open_in_develop)

    def _library_selected(self, pid: int):
        if pid == self._current_pid:
            return
        self._current_pid = pid
        self.filmstrip.select_id(pid)

    def _strip_selected(self, pid: int):
        self._current_pid = pid
        if self.stack.currentIndex() == 0:
            self.library.select_id(pid)

    def _view(self, idx: int):
        """Lazy construction of module views."""
        if idx in self._views:
            return self._views[idx]
        if idx == 1:
            v = DevelopView()
            v.statusMessage.connect(self.statusBar().showMessage)
            v.photoEdited.connect(self.filmstrip.refresh_thumb)
        elif idx == 2:
            from .printview import PrintView
            v = PrintView()
            v.statusMessage = lambda m: self.statusBar().showMessage(m)
        elif idx == 3:
            from .bookview import BookView
            v = BookView()
            v.set_status_fn(lambda m: self.statusBar().showMessage(m))
        else:
            return None
        self._views[idx] = v
        self.stack.addWidget(v)
        return v

    def switch_module(self, idx: int):
        if idx == self.stack.currentIndex() and idx in getattr(self, "_views", {}):
            return
        for i, tb in enumerate(self.tabs):
            tb.setChecked(i == idx)

        view = self._view(idx)
        if idx == 1:
            self.develop = view
        elif idx == 2:
            self.print_view = view
        elif idx == 3:
            self.book_view = view

        if idx == 1:
            pid = self._current_pid
            if pid is None:
                ids = self.filmstrip.ids_in_order()
                if ids:
                    pid = ids[0]
                    self._current_pid = pid
                    self.filmstrip.select_id(pid)
            if pid is not None:
                QTimer.singleShot(
                    30, lambda: getattr(self, "develop", None)
                    and self.develop.load_photo(pid))
        elif idx == 2:
            self.print_view.set_photos(self.filmstrip.ids_in_order())
        elif idx == 3:
            self.book_view.set_photos(self.filmstrip.ids_in_order())

        self.stack.setCurrentWidget(view)

    def open_in_develop(self, pid: int):
        self._current_pid = pid
        self.filmstrip.select_id(pid)
        self.develop = self._view(1)
        for i, tb in enumerate(self.tabs):
            tb.setChecked(i == 1)
        self.stack.setCurrentIndex(1)
        QTimer.singleShot(30, lambda: self.develop.load_photo(pid))

    def current_pid(self):
        dev = getattr(self, "develop", None)
        if self.stack.currentIndex() == 2 and dev and dev.photo_id:
            return dev.photo_id
        return self._current_pid or self.library.current_id()

    # ------------------------------------------------------------ actions
    def do_import(self):
        dlg = ImportDialog(self)
        def done(n):
            self.library.rebuild_folders()
            rows = self.library.refresh()
            self.filmstrip.load_photos(rows)
            self.statusBar().showMessage(f"Imported {n} photos")
        dlg.imported.connect(done)

        dlg.exec()

    def _selected_rows_for_merge(self, minimum: int) -> list:
        rows = self.library.selected_rows_ordered() or []
        if len(rows) < minimum and self.current_pid():
            row = catalog.get_photo(self.current_pid())
            rows = [row] if row else []
        return rows

    def do_hdr_merge(self):
        rows = self._selected_rows_for_merge(2)
        if len(rows) < 2:
            self.statusBar().showMessage("Select 2+ bracketed photos for HDR merge")
            return
        dlg = MergeDialog("hdr", rows, self)

        def done(pid):
            self.library.rebuild_folders()
            self.library.refresh()
            rows2 = self.library.refresh()
            self.filmstrip.load_photos(rows2)
            self.filmstrip.select_id(pid)
            self._current_pid = pid
            self.statusBar().showMessage("HDR merge complete — imported result")
        dlg.finished_merge.connect(done)
        dlg.exec()

    def do_pano(self):
        rows = self._selected_rows_for_merge(2)
        if len(rows) < 2:
            self.statusBar().showMessage("Select 2+ overlapping photos for panorama")
            return
        dlg = MergeDialog("pano", rows, self)

        def done(pid):
            self.library.rebuild_collections()
            self.library.rebuild_folders()
            rows2 = self.library.refresh()
            self.filmstrip.load_photos(rows2)
            self.filmstrip.select_id(pid)
            self._current_pid = pid
            self.statusBar().showMessage("Panorama stitched — imported result")
        dlg.finished_merge.connect(done)
        dlg.exec()

    def do_tether(self):
        dlg = TetherDialog(self)
        dlg.imported.connect(lambda n: None)

        def refresh_after():
            rows = self.library.refresh()
            self.filmstrip.load_photos(rows)
            self.statusBar().showMessage("Tether capture imported a photo")
        dlg.imported.connect(lambda _pid: refresh_after())
        dlg.exec()

    def do_auto_cull(self):
        rows = self.library.selected_rows_ordered()
        if len(rows) < 2:
            rows = self.library.refresh() or []
        if not rows:
            return
        dlg = CullDialog(rows, self)

        def done(applied, rejects):
            self.library.refresh()
            self.filmstrip.load_photos(catalog.query())
            self.statusBar().showMessage(
                f"Auto-cull: rated {applied}, flagged {rejects} blurry")
        dlg.finishedCull.connect(done)
        dlg.start()
        dlg.exec()

    def do_sync_settings(self):
        rows = self.library.selected_rows_ordered()
        if len(rows) < 2:
            self.statusBar().showMessage(
                "Select 2+ photos - settings copy from the first to the rest")
            return
        src_row, targets = rows[0], rows[1:]
        dlg = SyncDialog(src_row["filename"], len(targets), self)

        def apply():
            groups = dlg.chosen_groups()
            keys = [k for keys in groups.values() for k in keys]
            s_src = im_mod.sanitize_settings(
                catalog.load_settings(src_row["id"]) or {})
            for r in targets:
                s_t = im_mod.sanitize_settings(catalog.load_settings(r["id"]) or {})
                for k in keys:
                    s_t[k] = _copymod.deepcopy(s_src.get(k))
                catalog.save_settings(r["id"], s_t)
                self.filmstrip.refresh_thumb(r["id"])
            self.library.refresh()
            self.statusBar().showMessage(
                f"Synced {len(keys)} settings to {len(targets)} photos")
        dlg.accepted.connect(apply)
        dlg.exec()

    def do_find_dups(self):
        rows = catalog.query()
        if not rows:
            return
        from PySide6.QtWidgets import QProgressDialog
        prog = QProgressDialog("Hashing photos…", None, 0, len(rows), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        def work():
            from ..core.dupfind import dhash, find_similar
            hashes = {}
            for i, r in enumerate(rows):
                try:
                    prev = rawio.decode_preview(r["path"], 64)
                    hashes[r["id"]] = dhash(prev)
                except Exception:
                    pass
                QTimer.singleShot(0, lambda i=i: prog.setValue(i+1))
            groups = find_similar(hashes, threshold=10)
            QTimer.singleShot(0, lambda: (
                prog.close(),
                self._show_dup_results(groups, rows)))
        import threading
        threading.Thread(target=work, daemon=True).start()

    def _show_dup_results(self, groups, all_rows):
        by_id = {r["id"]: r for r in all_rows}
        lines = []
        for g in groups[:20]:
            names = [by_id[pid]["filename"] for pid in g if pid in by_id]
            lines.append("  · ".join(names))
        msg = f"{len(groups)} similar group(s):\n" + "\n".join(lines[:15])
        self.statusBar().showMessage(msg.replace("\n", " | ")[:200])
        from PySide6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Similar Photos", msg)

    def do_slideshow(self):
        ids = self.filmstrip.ids_in_order()
        rows = [r for r in (catalog.get_photo(i) for i in ids) if r]
        if not rows:
            self.statusBar().showMessage("Nothing to play")
            return
        screen = QGuiApplication.primaryScreen()
        geo = screen.availableGeometry()
        self._slideshow = SlideshowWindow(rows, (geo.width(), geo.height()))
        self._slideshow.showFullScreen()

    def do_web_gallery(self):
        rows = [r for r in (catalog.get_photo(i)
                            for i in self.filmstrip.ids_in_order()) if r]
        if not rows:
            self.statusBar().showMessage("No photos for gallery")
            return
        out_dir = os.path.expanduser("~/Pictures/Lumina Web Gallery")
        from PySide6.QtWidgets import QProgressDialog
        prog = QProgressDialog("Rendering gallery…", None, 0, len(rows), self)
        prog.setWindowModality(Qt.WindowModality.WindowModal)
        prog.setMinimumDuration(0)

        def step(i, n):
            prog.setValue(i)

        def worker():
            try:
                from ..core.webgallery import export_gallery
                path = export_gallery(rows, out_dir, progress=step)
                QTimer.singleShot(0, lambda: (
                    prog.close(),
                    self.statusBar().showMessage(
                        f"Gallery exported: {path} — open index.html")))
            except Exception as e:
                QTimer.singleShot(0, lambda: (prog.close(),
                                              self.statusBar().showMessage(
                                                  f"Gallery failed: {e}")))
        import threading
        threading.Thread(target=worker, daemon=True).start()

    def do_export(self):
        pid = self.current_pid()
        if pid is None:
            self.statusBar().showMessage("Nothing selected to export")
            return
        row = catalog.get_photo(pid)
        if row is None:
            return
        settings = catalog.load_settings(pid) or {}
        from ..core.imaging import sanitize_settings
        settings = sanitize_settings(settings)
        jobs = [(row["path"], settings)]
        dlg = ExportDialog(jobs, self)
        dlg.exported.connect(
            lambda n: self.statusBar().showMessage(
                f"Exported {n} photo{'s' if n != 1 else ''}"))
        dlg.exec()

    # ------------------------------------------------------------ shortcuts
    def _shortcuts(self):
        def sc(seq, cb):
            a = QAction(self)
            a.setShortcut(QKeySequence(seq))
            a.triggered.connect(cb)
            self.addAction(a)
            return a

        sc("G", lambda: self.switch_module(0))
        sc("D", lambda: self.switch_module(1))
        sc("E", lambda: self.switch_module(1))
        sc("R", self._toggle_crop_shortcut)
        sc("F", lambda: self._dev_canvas_call("set_zoom_fit"))
        sc("+", lambda: self._dev_canvas_call("wheelEvent_zoom", 1.25))
        sc("-", lambda: self._dev_canvas_call("wheelEvent_zoom", 0.8))
        sc("\\", lambda: None)   # handled via button press/release; keep binding harmless
        sc("P", lambda: self._set_flag_current(1))
        sc("X", lambda: self._set_flag_current(-1))
        sc("U", lambda: self._set_flag_current(0))
        for n in range(6):
            sc(str(n), lambda checked=False, r=n: self._set_rating_current(r))
        sc(QKeySequence.Undo, lambda: self._dev_call("undo"))
        sc(QKeySequence.Redo, lambda: self._dev_call("redo"))
        sc(QKeySequence.Copy, lambda: self._dev_call("copy_settings"))
        sc(QKeySequence.Paste, lambda: self._dev_call("paste_settings"))
        sc(QKeySequence("Ctrl+E"), self.do_export)
        sc(QKeySequence("Ctrl+I"), self.do_import)
        sc(QKeySequence("Right"), lambda: self._navigate(1))
        sc(QKeySequence("Left"), lambda: self._navigate(-1))

    def _dev_canvas_call(self, method: str, *args):
        dev = getattr(self, "develop", None)
        if dev and hasattr(dev, "canvas"):
            getattr(dev.canvas, method)(*args)

    def _dev_call(self, method: str):
        dev = getattr(self, "develop", None)
        if dev and hasattr(dev, method):
            getattr(dev, method)()

    def _toggle_crop_shortcut(self):
        if self.stack.currentIndex() == 1:
            dev = getattr(self, "develop", None)
            if dev:
                dev.b_crop.toggle()
                dev.toggle_crop()

    def _set_rating_current(self, r: int):
        pid = self.current_pid()
        if pid is None:
            return
        cur = catalog.get_photo(pid)
        if cur is not None and cur["rating"] == r and r > 0:
            r = 0
        catalog.update_fields(pid, rating=r)
        row = catalog.get_photo(pid)
        if row:
            from .core_xmp_write import write_sidecar
            write_sidecar(row["path"], rating=r,
                          flag=row["flag"], color=row["color"],
                          keywords=row["keywords"] or "",
                          settings_json=row["settings_json"])
        self.library.refresh_one(pid)
        self.filmstrip.refresh_thumb(pid)
        self.statusBar().showMessage(f"Rating: {'★'*r or 'none'}")

    def _set_flag_current(self, f: int):
        pid = self.current_pid()
        if pid is None:
            return
        cur = catalog.get_photo(pid)
        if cur is not None and cur["flag"] == f and f != 0:
            f = 0
        catalog.update_fields(pid, flag=f)
        row = catalog.get_photo(pid)
        if row:
            from .core_xmp_write import write_sidecar
            write_sidecar(row["path"], rating=row["rating"],
                          flag=f, color=row["color"],
                          keywords=row["keywords"] or "",
                          settings_json=row["settings_json"])
        self.library.refresh_one(pid)
        self.filmstrip.refresh_thumb(pid)
        self.statusBar().showMessage({-1: "Rejected", 0: "Flag cleared",
                                      1: "Picked"}.get(f, ""))

    def _navigate(self, step: int):
        pid = self.filmstrip.next_id(step)
        if pid is not None:
            self._current_pid = pid
            if self.stack.currentIndex() == 0:
                self.library.select_id(pid)
            elif self.stack.currentIndex() == 1:
                self.develop.load_photo(pid)
            self.filmstrip.select_id(pid)

    def closeEvent(self, e):
        dev = getattr(self, "develop", None)
        if dev:
            dev._persist_now()
        super().closeEvent(e)
