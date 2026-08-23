"""Library module — catalog browsing, filtering, rating, metadata."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QObject, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (QComboBox, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QListWidget, QListWidgetItem,
                               QPushButton, QSlider, QTreeWidget,
                               QTreeWidgetItem, QVBoxLayout, QWidget)

from ..core import catalog, rawio
from . import theme
from .widgets import (COLOR_LABELS, ColorLabelPicker, FlagPicker, RatingStars,
                      SliderRow, make_badge_pixmap)


class _ThumbBridge(QObject):
    done = Signal(int, str)


from PySide6.QtCore import QRunnable


class _ThumbTask(QRunnable):
    def __init__(self, bridge, photo_id, path):
        super().__init__()
        self.bridge, self.photo_id, self.path = bridge, photo_id, path

    def run(self):
        tpath = rawio.make_thumbnail(self.path)
        if tpath:
            self.bridge.done.emit(self.photo_id, tpath)


class MetadataPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(3)
        title = QLabel("METADATA")
        title.setObjectName("SectionTitle")
        v.addWidget(title)
        self.rows: dict[str, QLabel] = {}
        for key, label in [("filename", "File"), ("dims", "Dimensions"),
                           ("camera", "Camera"), ("lens", "Lens"),
                           ("exposure", "Exposure"), ("date", "Captured"),
                           ("size", "Size")]:
            hl = QHBoxLayout()
            k = QLabel(label)
            k.setStyleSheet(f"color:{theme.TEXT_FAINT}; font-size:11px;")
            k.setFixedWidth(72)
            val = QLabel("—")
            val.setStyleSheet(f"color:{theme.TEXT_DIM};")
            val.setWordWrap(True)
            hl.addWidget(k)
            hl.addWidget(val, 1)
            v.addLayout(hl)
            self.rows[key] = val
        v.addStretch(1)

    def show_row(self, r):
        if r is None:
            for lab in self.rows.values():
                lab.setText("—")
            return
        self.rows["filename"].setText(r["filename"])
        self.rows["filename"].setToolTip(r["path"])
        self.rows["dims"].setText(f"{r['width']} × {r['height']} px"
                                  if r["width"] else "—")
        self.rows["camera"].setText(r["camera"] or "—")
        self.rows["lens"].setText(r["lens"] or "—")
        exp = []
        if r["focal"]:
            exp.append(f"{r['focal']:.0f}mm")
        if r["aperture"]:
            exp.append(f"ƒ/{r['aperture']:.1f}")
        if r["shutter"]:
            exp.append(r["shutter"])
        if r["iso"]:
            exp.append(f"ISO {r['iso']}")
        self.rows["exposure"].setText(" ".join(exp) or "—")
        self.rows["date"].setText(r["date_taken"] or "—")
        self.rows["size"].setText(rawio.human_size(r["size"] or 0))


class QuickDevelop(QFrame):
    quickEdit = Signal(str, float)
    autoRequested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(10, 8, 10, 8)
        v.setSpacing(2)
        t = QLabel("QUICK DEVELOP")
        t.setObjectName("SectionTitle")
        v.addWidget(t)
        self.sl_ex = SliderRow("Exposure", -5, 5, 0, 2)
        self.sl_ct = SliderRow("Contrast", -100, 100, 0)
        self.sl_vb = SliderRow("Vibrance", -100, 100, 0)
        for s in (self.sl_ex, self.sl_ct, self.sl_vb):
            s.label.setFixedWidth(64)
            v.addWidget(s)
        self.sl_ex.editingFinished.connect(lambda v: self.quickEdit.emit("exposure", v))
        self.sl_ct.editingFinished.connect(lambda v: self.quickEdit.emit("contrast", v))
        self.sl_vb.editingFinished.connect(lambda v: self.quickEdit.emit("vibrance", v))
        auto = QPushButton("Auto Tone")
        auto.clicked.connect(self.autoRequested)
        v.addWidget(auto)
        self.setEnabled(False)

    def set_silent(self, ex=0.0, ct=0.0, vb=0.0):
        for s, val in ((self.sl_ex, ex), (self.sl_ct, ct), (self.sl_vb, vb)):
            s.set_value_silent(val)


class LibraryView(QWidget):
    openPhoto = Signal(int)
    selectionChangedId = Signal(int)
    photoEdited = Signal(int)
    importRequested = Signal()
    collectionSelectedId = Signal(object)     # cid or None
    hdrRequested = Signal()
    panoRequested = Signal()
    tetherRequested = Signal()
    syncRequested = Signal()
    cullRequested = Signal()
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._bridge = _ThumbBridge()
        self._bridge.done.connect(self._thumb_done)
        self.current_folder = None
        self._id_item: dict[int, QListWidgetItem] = {}
        self.thumb_px = 190

        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # ---------------- left panel
        left = QWidget()
        left.setObjectName("SidePanel")
        left.setFixedWidth(210)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 6, 10)
        lv.setSpacing(8)
        ct = QLabel("COLLECTIONS")
        ct.setObjectName("SectionTitle")
        lv.addWidget(ct)
        self.coll_list = QListWidget()
        self.coll_list.setFixedHeight(110)
        self.coll_list.itemClicked.connect(self._collection_selected)
        lv.addWidget(self.coll_list)
        crow = QHBoxLayout()
        b_add_c = QPushButton("+ New")
        b_add_c.setToolTip("New collection from selected photos")
        b_add_c.clicked.connect(self._new_collection)
        b_del_c = QPushButton("− Del")
        b_del_c.clicked.connect(self._delete_collection)
        crow.addWidget(b_add_c); crow.addWidget(b_del_c); crow.addStretch(1)
        lv.addLayout(crow)

        lt = QLabel("FOLDERS")
        lt.setObjectName("SectionTitle")
        lv.addWidget(lt)
        self.folder_tree = QTreeWidget()
        self.folder_tree.setHeaderHidden(True)
        self.folder_tree.itemSelectionChanged.connect(self._folder_selected)
        lv.addWidget(self.folder_tree, 1)
        btn_import = QPushButton("Import Folder…")
        btn_import.setObjectName("Primary")
        btn_import.clicked.connect(self.importRequested)
        lv.addWidget(btn_import)
        h.addWidget(left)

        # ---------------- center
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        bar = QWidget()
        bar.setObjectName("TopBar")
        bl = QHBoxLayout(bar)
        bl.setContentsMargins(10, 5, 10, 5)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search filename, camera, lens…")
        self.search.setClearButtonEnabled(True)
        self.search.setFixedWidth(230)
        self.search.textChanged.connect(self.refresh)
        bl.addWidget(self.search)
        bl.addSpacing(12)
        bl.addWidget(QLabel("Rating ≥"))
        self.f_rating = QComboBox()
        self.f_rating.addItems(["Any", "★+", "★★+", "★★★+", "★★★★+", "★★★★★"])
        self.f_rating.currentIndexChanged.connect(self.refresh)
        bl.addWidget(self.f_rating)
        bl.addWidget(QLabel("Flags"))
        self.f_flag = QComboBox()
        self.f_flag.addItems(["All", "Picked", "Unflagged", "Rejected"])
        self.f_flag.currentIndexChanged.connect(self.refresh)
        bl.addWidget(self.f_flag)
        bl.addWidget(QLabel("Label"))
        self.f_color = QComboBox()
        self.f_color.addItem("Any", None)
        for i in range(1, 6):
            self.f_color.addItem("●", i)
            self.f_color.setItemData(i, QColor_safe(COLOR_LABELS[i]), Qt.DecorationRole)
        self.f_color.currentIndexChanged.connect(self.refresh)
        bl.addWidget(self.f_color)
        bl.addStretch(1)
        b_hdr = QPushButton("HDR")
        b_hdr.setToolTip("Merge selected bracketed shots (2+)")
        b_hdr.clicked.connect(self.hdrRequested)
        bl.addWidget(b_hdr)
        b_pano = QPushButton("Pano")
        b_pano.setToolTip("Stitch selected overlapping shots into a panorama")
        b_pano.clicked.connect(self.panoRequested)
        bl.addWidget(b_pano)
        b_cull = QPushButton("Auto-Cull")
        b_cull.setToolTip("Score burst/folder and auto-assign ratings")
        b_cull.clicked.connect(self.cullRequested)
        bl.addWidget(b_cull)
        b_sync = QPushButton("Sync…")
        b_sync.setToolTip("Copy develop settings from first selected photo to the rest")
        b_sync.clicked.connect(self.syncRequested)
        bl.addWidget(b_sync)
        b_tether = QPushButton("Tether…")
        b_tether.setToolTip("Capture from a camera or watch a folder")
        b_tether.clicked.connect(self.tetherRequested)
        bl.addWidget(b_tether)
        bl.addWidget(QLabel("Size"))
        self.size_slider = QSlider(Qt.Horizontal)
        self.size_slider.setRange(110, 340)
        self.size_slider.setValue(self.thumb_px)
        self.size_slider.setFixedWidth(110)
        self.size_slider.valueChanged.connect(self._thumb_size_changed)
        bl.addWidget(self.size_slider)
        cv.addWidget(bar)

        self.grid = QListWidget()
        self.grid.setObjectName("GridList")
        self.grid.setViewMode(QListWidget.IconMode)
        self.grid.setIconSize(QSize(self.thumb_px, self.thumb_px))
        self.grid.setResizeMode(QListWidget.Adjust)
        self.grid.setMovement(QListWidget.Static)
        self.grid.setSpacing(10)
        self.grid.setSelectionMode(QListWidget.ExtendedSelection)
        self.grid.setWordWrap(True)
        self.grid.itemClicked.connect(self._item_clicked)
        self.grid.itemDoubleClicked.connect(
            lambda it: self.openPhoto.emit(it.data(Qt.UserRole)))
        cv.addWidget(self.grid, 1)
        h.addWidget(center, 1)

        # ---------------- right panel
        right = QWidget()
        right.setObjectName("SidePanel")
        right.setFixedWidth(250)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 10, 0, 10)
        rv.setSpacing(10)

        self.meta = MetadataPanel()
        rv.addWidget(self.meta)

        kw_frame = QFrame()
        kv = QVBoxLayout(kw_frame)
        kv.setContentsMargins(10, 4, 10, 4)
        kt = QLabel("KEYWORDS")
        kt.setObjectName("SectionTitle")
        kv.addWidget(kt)
        self.keywords_edit = QLineEdit()
        self.keywords_edit.setPlaceholderText("comma, separated, tags")
        self.keywords_edit.setClearButtonEnabled(True)
        self.keywords_edit.editingFinished.connect(self._keywords_saved)
        kv.addWidget(self.keywords_edit)
        rv.addWidget(kw_frame)

        rat = QFrame()
        ratv = QVBoxLayout(rat)
        ratv.setContentsMargins(10, 4, 10, 4)
        ratv.setSpacing(6)
        rt = QLabel("RATING & LABELS")
        rt.setObjectName("SectionTitle")
        ratv.addWidget(rt)
        self.stars = RatingStars()
        self.stars.changed.connect(self._set_rating)
        ratv.addWidget(self.stars)
        self.flags = FlagPicker()
        self.flags.changed.connect(self._set_flag)
        ratv.addWidget(self.flags)
        self.colors = ColorLabelPicker()
        self.colors.changed.connect(self._set_color)
        ratv.addWidget(self.colors)
        rv.addWidget(rat)

        self.quick = QuickDevelop()
        self.quick.quickEdit.connect(self._quick_edit)
        self.quick.autoRequested.connect(self._quick_auto)
        rv.addWidget(self.quick)
        rv.addStretch(1)
        h.addWidget(right)

    # ------------------------------------------------------------ folders
    def rebuild_folders(self):
        self.folder_tree.blockSignals(True)
        self.folder_tree.clear()
        total = len(catalog.query())
        root = QTreeWidgetItem([f"All Photographs  ({total})"])
        root.setData(0, Qt.UserRole, None)
        self.folder_tree.addTopLevelItem(root)
        for f in catalog.list_folders():
            n = len(catalog.query(folder=f))
            it = QTreeWidgetItem([f"{os.path.basename(f)}  ({n})"])
            it.setData(0, Qt.UserRole, f)
            it.setToolTip(0, f)
            self.folder_tree.addTopLevelItem(it)
        self.folder_tree.setCurrentItem(root)
        self.folder_tree.expandAll()
        self.folder_tree.blockSignals(False)

    def _folder_selected(self):
        items = self.folder_tree.selectedItems()
        if not items:
            return
        self.current_folder = items[0].data(0, Qt.UserRole)
        self.refresh()

    # ------------------------------------------------------------ grid
    def refresh(self):
        min_rating = max(0, self.f_rating.currentIndex())
        flag = ["all", "picked", "unflagged", "rejected"][self.f_flag.currentIndex()]
        color = self.f_color.currentData()
        if getattr(self, "active_collection", None):
            ids = set(catalog.collection_member_ids(self.active_collection))
            rows = [r for r in catalog.query(min_rating=min_rating, flag=flag,
                                             color=color,
                                             search=self.search.text().strip())
                    if r["id"] in ids]
        else:
            rows = catalog.query(folder=self.current_folder,
                                 min_rating=min_rating, flag=flag, color=color,
                                 search=self.search.text().strip())
        self.grid.clear()
        self._id_item.clear()
        for r in rows:
            it = QListWidgetItem(r["filename"])
            it.setData(Qt.UserRole, r["id"])
            it.setSizeHint(QSize(self.thumb_px, self.thumb_px + 22))
            it.setTextAlignment(Qt.AlignHCenter | Qt.AlignBottom)
            self.grid.addItem(it)
            self._id_item[r["id"]] = it
            pm = self._cached_pixmap(r["id"])
            if pm is not None:
                self._apply_thumb(r["id"], pm)
            else:
                self._pool.start(_ThumbTask(self._bridge, r["id"], r["path"]), 1)
        return rows

    def _cached_pixmap(self, photo_id) -> QPixmap | None:
        row = catalog.get_photo(photo_id)
        if row is None:
            return None
        tpath = rawio.thumb_cache_path(row["path"])
        if os.path.exists(tpath):
            pm = QPixmap(tpath)
            return pm if not pm.isNull() else None
        return None

    def _apply_thumb(self, photo_id: int, pm: QPixmap):
        it = self._id_item.get(photo_id)
        if it is None:
            return
        row = catalog.get_photo(photo_id)
        if row is None:
            return
        pm2 = pm.scaled(self.thumb_px - 8, self.thumb_px - 8,
                        Qt.KeepAspectRatio, Qt.SmoothTransformation)
        pm2 = make_badge_pixmap(pm2, row["rating"], row["flag"], row["color"],
                                bool(row["settings_json"]))
        it.setIcon(QIcon(pm2))

    def _thumb_done(self, photo_id: int, tpath: str):
        pm = QPixmap(tpath)
        if not pm.isNull():
            self._apply_thumb(photo_id, pm)

    def _thumb_size_changed(self, v: int):
        self.thumb_px = v
        self.grid.setIconSize(QSize(v, v))
        for pid, it in self._id_item.items():
            it.setSizeHint(QSize(v, v + 22))
            pm = self._cached_pixmap(pid)
            if pm is not None:
                self._apply_thumb(pid, pm)

    def refresh_one(self, photo_id: int):
        pm = self._cached_pixmap(photo_id)
        if pm is not None:
            self._apply_thumb(photo_id, pm)

    def select_id(self, photo_id: int):
        it = self._id_item.get(photo_id)
        if it:
            self.grid.blockSignals(True)
            self.grid.setCurrentItem(it)
            self.grid.blockSignals(False)
            self._show_meta(photo_id)

    def current_id(self) -> int | None:
        it = self.grid.currentItem()
        return it.data(Qt.UserRole) if it else None

    def selected_rows_ordered(self):
        """All selected photos in grid order (for HDR/pano merges)."""
        sel = set(it.data(Qt.UserRole) for it in self.grid.selectedItems())
        out = []
        for i in range(self.grid.count()):
            it = self.grid.item(i)
            if it.data(Qt.UserRole) in sel:
                row = catalog.get_photo(it.data(Qt.UserRole))
                if row is not None:
                    out.append(row)
        return out or (self.current_id() and [catalog.get_photo(self.current_id())]
                       or [])

    def _item_clicked(self, it):
        pid = it.data(Qt.UserRole)
        self.selectionChangedId.emit(pid)
        self._show_meta(pid)

    def _show_meta(self, photo_id: int):
        from ..core.imaging import sanitize_settings
        row = catalog.get_photo(photo_id)
        self.meta.show_row(row)
        self.quick.setEnabled(row is not None)
        if row:
            s = sanitize_settings(catalog.load_settings(photo_id) or {})
            self.quick.set_silent(s["exposure"], s["contrast"], s["vibrance"])
            self.keywords_edit.setText(row["keywords"] or "")
            for w, val in ((self.stars, row["rating"]), (self.flags, row["flag"]),
                           (self.colors, row["color"])):
                w.blockSignals(True)
                if isinstance(w, RatingStars):
                    w.set_rating(val)
                elif isinstance(w, FlagPicker):
                    w.set_flag(val)
                else:
                    w.set_color(val)
                w.blockSignals(False)

    # ------------------------------------------------------------ collections
    def rebuild_collections(self):
        self.coll_list.blockSignals(True)
        self.coll_list.clear()
        it = QListWidgetItem("All Photographs")
        it.setData(Qt.UserRole, None)
        self.coll_list.addItem(it)
        for c in catalog.list_collections():
            n = len(catalog.collection_member_ids(c["id"]))
            item = QListWidgetItem(f"▦ {c['name']}  ({n})")
            item.setData(Qt.UserRole, c["id"])
            self.coll_list.addItem(item)
        self.coll_list.setCurrentRow(0)
        self.coll_list.blockSignals(False)
        self.active_collection = None

    def _collection_selected(self, it):
        cid = it.data(Qt.UserRole)
        self.active_collection = cid
        self.current_folder = None
        self.folder_tree.blockSignals(True)
        self.folder_tree.clearSelection()
        self.folder_tree.blockSignals(False)
        self.collectionSelectedId.emit(cid)
        self.refresh()

    def _new_collection(self):
        from PySide6.QtWidgets import QInputDialog
        sel = self.selected_rows_ordered()
        name, ok = QInputDialog.getText(self, "New Collection",
                                        f"Name ({len(sel)} selected photos will be added):")
        if not ok or not name.strip():
            return
        cid = catalog.create_collection(name.strip())
        for r in sel:
            catalog.collection_add(cid, r["id"])
        self.rebuild_collections()
        self.statusMessage.emit(f"Collection created: {name}")

    def _delete_collection(self):
        it = self.coll_list.currentItem()
        if not it or it.data(Qt.UserRole) is None:
            return
        catalog.delete_collection(it.data(Qt.UserRole))
        self.rebuild_collections()
        self.refresh()

    # ------------------------------------------------------------ keywords
    def _keywords_saved(self):
        pid = self.current_id()
        if pid:
            catalog.set_keywords(pid, self.keywords_edit.text())
            self.statusMessage.emit("Keywords saved")

    # ------------------------------------------------------------ rating ops
    def _set_rating(self, v):
        pid = self.current_id()
        if pid:
            catalog.update_fields(pid, rating=v)
            self.refresh_one(pid)
            self.statusMessage.emit(f"Rating set to {v}★")

    def _set_flag(self, v):
        pid = self.current_id()
        if pid:
            catalog.update_fields(pid, flag=v)
            self.refresh_one(pid)
            self.statusMessage.emit({-1: "Rejected", 0: "Flag cleared",
                                     1: "Picked"}.get(v, ""))

    def _set_color(self, v):
        pid = self.current_id()
        if pid:
            catalog.update_fields(pid, color=v)
            self.refresh_one(pid)

    # ------------------------------------------------------------ quick develop
    def _quick_edit(self, key: str, value: float):
        pid = self.current_id()
        if not pid:
            return
        from ..core.imaging import sanitize_settings
        s = sanitize_settings(catalog.load_settings(pid) or {})
        s[key] = value
        catalog.save_settings(pid, s)
        self.photoEdited.emit(pid)
        self.statusMessage.emit("Quick develop applied")

    def _quick_auto(self):
        from ..core.imaging import compute_auto_tone, sanitize_settings
        pid = self.current_id()
        if not pid:
            return
        try:
            prev = rawio.decode_preview(catalog.get_photo(pid)["path"], 1200)
            ev, blacks, whites = compute_auto_tone(prev.astype(np.float32) / 255.0)
        except Exception:
            ev, blacks, whites = 0.0, 0.0, 0.0
        s = sanitize_settings(catalog.load_settings(pid) or {})
        s["exposure"], s["blacks"], s["whites"] = round(ev, 2), round(blacks, 1), round(whites, 1)
        catalog.save_settings(pid, s)
        self.quick.set_silent(s["exposure"], s["contrast"], s["vibrance"])
        self.photoEdited.emit(pid)
        self.statusMessage.emit("Auto tone applied")


def QColor_safe(hexstr: str):
    from PySide6.QtGui import QColor
    return QColor(hexstr)
