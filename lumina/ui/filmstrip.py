"""Bottom filmstrip shared by Library and Develop modules."""
from __future__ import annotations

import os

from PySide6.QtCore import (QObject, QRunnable, QThreadPool, QSize, Qt,
                            Signal)
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QListWidget, QListWidgetItem

from ..core import catalog, rawio
from .widgets import make_badge_pixmap


class _StripBridge(QObject):
    done = Signal(int)


class _ThumbJob(QRunnable):
    def __init__(self, bridge, photo_id):
        super().__init__()
        self.bridge = bridge
        self.photo_id = photo_id

    def run(self):
        row = catalog.get_photo(self.photo_id)
        if row is None:
            return
        tpath = rawio.make_thumbnail(row["path"])
        if tpath:
            self.bridge.done.emit(self.photo_id)


class Filmstrip(QListWidget):
    selectionChangedId = Signal(int)          # photo id
    openRequested = Signal(int)               # double-click

    THUMB = 74

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Filmstrip")
        self._pool = QThreadPool.globalInstance()
        self._bridge = _StripBridge(self)
        self._bridge.done.connect(self._thumb_ready)
        self._id_item = {}
        self._suppress = False

        self.setViewMode(QListWidget.ViewMode.IconMode)
        self.setFlow(QListWidget.Flow.LeftToRight)
        self.setWrapping(False)
        self.setIconSize(QSize(self.THUMB - 12, self.THUMB - 24))
        self.setFixedHeight(self.THUMB + 14)
        self.setSpacing(4)
        self.setUniformItemSizes(True)
        self.setHorizontalScrollMode(QListWidget.ScrollMode.ScrollPerPixel)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.itemClicked.connect(self._clicked)
        self.itemDoubleClicked.connect(
            lambda it: self.openRequested.emit(it.data(Qt.ItemDataRole.UserRole)))

    # ------------------------------------------------------------ helpers
    def _clicked(self, item):
        self.selectionChangedId.emit(item.data(Qt.ItemDataRole.UserRole))

    def _cached_pixmap(self, photo_id):
        row = catalog.get_photo(photo_id)
        if row is None:
            return None
        tp = rawio.thumb_cache_path(row["path"])
        if os.path.exists(tp):
            pm = QPixmap(tp)
            return pm if not pm.isNull() else None
        return None

    def _styled(self, pm, photo_id):
        row = catalog.get_photo(photo_id)
        if row is None:
            return pm
        pm2 = pm.scaled(self.THUMB - 12, self.THUMB - 24,
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
        return make_badge_pixmap(pm2, row["rating"], row["flag"],
                                 row["color"], bool(row["settings_json"]))

    def _thumb_ready(self, photo_id):
        pm = self._cached_pixmap(photo_id)
        it = self._id_item.get(photo_id)
        if pm is not None and it is not None:
            it.setIcon(QIcon(self._styled(pm, photo_id)))

    # ------------------------------------------------------------ API
    def load_photos(self, rows):
        self._suppress = True
        self.clear()
        self._id_item.clear()
        pending = []
        for r in rows:
            it = QListWidgetItem(r["filename"])
            it.setData(Qt.ItemDataRole.UserRole, int(r["id"]))
            it.setSizeHint(QSize(self.THUMB, self.THUMB - 8))
            it.setToolTip(r["filename"])
            self.addItem(it)
            self._id_item[int(r["id"])] = it
            pm = self._cached_pixmap(int(r["id"]))
            if pm is not None:
                it.setIcon(QIcon(self._styled(pm, int(r["id"]))))
            else:
                pending.append(int(r["id"]))
        self._suppress = False
        for pid in pending:
            self._pool.start(_ThumbJob(self._bridge, pid))

    def refresh_thumb(self, photo_id: int):
        pid = int(photo_id)
        pm = self._cached_pixmap(pid)
        if pm is not None:
            self._thumb_ready(pid)
        else:
            self._pool.start(_ThumbJob(self._bridge, pid))

    def select_id(self, photo_id: int):
        it = self._id_item.get(int(photo_id))
        if it is None:
            return
        self._suppress = True
        self.setCurrentItem(it)
        self.scrollToItem(it, QListWidget.ScrollHint.PositionAtCenter)
        self._suppress = False

    def current_id(self):
        it = self.currentItem()
        return int(it.data(Qt.ItemDataRole.UserRole)) if it else None

    def ids_in_order(self):
        return [int(self.item(i).data(Qt.ItemDataRole.UserRole))
                for i in range(self.count())]

    def next_id(self, step=1):
        ids = self.ids_in_order()
        cur = self.current_id()
        if not ids or cur is None:
            return ids[0] if ids else None
        try:
            i = ids.index(cur)
        except ValueError:
            return ids[0]
        return ids[min(len(ids) - 1, max(0, i + step))]
