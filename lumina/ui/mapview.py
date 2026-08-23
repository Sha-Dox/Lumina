"""Map module — OpenStreetMap slippy map, photo pins, geotagging."""
from __future__ import annotations

import math
import os
from io import BytesIO

from PySide6.QtCore import QByteArray, QPointF, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
                               QPushButton, QVBoxLayout, QWidget)

from ..core import catalog
from . import theme

TILE_DIR = os.path.expanduser("~/.lumina/cache/tiles")
TILE_SIZE = 256


def _lonlat_to_px(lon: float, lat: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z * TILE_SIZE
    x = (lon + 180.0) / 360.0 * n
    lat_r = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_r) + 1 / math.cos(lat_r)) / math.pi) / 2.0 * n
    return x, y


def _px_to_lonlat(x: float, y: float, z: int) -> tuple[float, float]:
    n = 2.0 ** z * TILE_SIZE
    lon = x / n * 360.0 - 180.0
    nn = math.pi - 2.0 * math.pi * y / n
    lat = math.degrees(math.atan(0.5 * (math.exp(nn) - math.exp(-nn))))
    return lon, lat


class SlippyMap(QWidget):
    photoPinned = Signal(int)                 # clicked a pin
    mapClicked = Signal(float, float)         # lon, lat (with modifier)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.zoom = 3
        self.center_lon, self.center_lat = 29.0, 41.0   # start over Istanbul
        self._tiles = {}                                # (z,x,y) -> QPixmap
        self._pending = set()
        self._nam = QNetworkAccessManager(self)
        self._nam.finished.connect(self._tile_done)
        self._drag = None
        self.setMouseTracking(True)

    # ------------------------------------------------------------ tiles
    def _request_tile(self, z, x, y):
        key = (z, x, y)
        if key in self._tiles or key in self._pending:
            return
        path = os.path.join(TILE_DIR, str(z), str(x), f"{y}.png")
        if os.path.exists(path):
            pm = QPixmap(path)
            if not pm.isNull():
                self._tiles[key] = pm
                self.update()
                return
        self._pending.add(key)
        url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        req = QNetworkRequest(QUrl(url))
        req.setRawHeader(b"User-Agent", b"Lumina/1.0 (personal photo manager)")
        reply = self._nam.get(req)
        reply.setProperty("tilekey", list(key))

    def _tile_done(self, reply: QNetworkReply):
        try:
            key = tuple(reply.property("tilekey") or [])
            self._pending.discard(key)
            from PySide6.QtNetwork import QNetworkReply as _QNR
            if reply.error() != _QNR.NetworkError.NoError:
                return
            data = bytes(reply.readAll())
            pm = QPixmap()
            if not pm.loadFromData(QByteArray(data)):
                return
            z, x, y = key
            path = os.path.join(TILE_DIR, str(z), str(x))
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{y}.png"), "wb") as f:
                f.write(data)
            if len(self._tiles) > 900:
                self._tiles.clear()
            self._tiles[key] = pm
            self.update()
        finally:
            reply.deleteLater()

    # ------------------------------------------------------------ geometry
    def _center_px(self) -> tuple[float, float]:
        return _lonlat_to_px(self.center_lon, self.center_lat, self.zoom)

    def _lonlat_at(self, pos: QPointF) -> tuple[float, float]:
        cx, cy = self._center_px()
        wx = cx + pos.x() - self.width() / 2
        wy = cy + pos.y() - self.height() / 2
        return _px_to_lonlat(wx, wy, self.zoom)

    def set_center(self, lon: float, lat: float, zoom: int | None = None):
        self.center_lon, self.center_lat = lon, lat
        if zoom:
            self.zoom = max(1, min(19, zoom))
        self.update()

    # ------------------------------------------------------------ painting
    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#101418"))
        cx, cy = self._center_px()
        ox = self.width() / 2 - cx
        oy = self.height() / 2 - cy

        tx0 = int((cx - self.width() / 2) // TILE_SIZE)
        ty0 = int((cy - self.height() / 2) // TILE_SIZE)
        tx1 = int((cx + self.width() / 2) // TILE_SIZE) + 1
        ty1 = int((cy + self.height() / 2) // TILE_SIZE) + 1
        n = 2 ** self.zoom
        for tx in range(max(0, tx0), min(n, tx1 + 1)):
            for ty in range(max(0, ty0), min(n, ty1 + 1)):
                px = tx * TILE_SIZE + ox
                py = ty * TILE_SIZE + oy
                self._request_tile(self.zoom, tx, ty)
                pm = self._tiles.get((self.zoom, tx, ty))
                if pm:
                    p.drawPixmap(QPointF(px, py), pm)
                else:
                    p.fillRect(QRectF(px, py, TILE_SIZE, TILE_SIZE),
                               QColor("#1a2026"))

        # pins
        p.setRenderHint(QPainter.Antialiasing)
        rows = catalog.photos_with_gps()
        font = p.font()
        font.setPixelSize(10)
        p.setFont(font)
        from .widgets import COLOR_LABELS
        for r in rows:
            px, py = _lonlat_to_px(r["gps_lon"], r["gps_lat"], self.zoom)
            sx = px + ox
            sy = py + oy
            if -20 <= sx <= self.width() + 20 and -20 <= sy <= self.height() + 20:
                col = QColor(COLOR_LABELS[r["color"]] or "#e05555")
                p.setBrush(col)
                p.setPen(QPen(QColor("#ffffff"), 1.5))
                p.drawEllipse(QPointF(sx, sy), 7, 7)
                p.setPen(QColor("#dddddd"))
                p.drawText(QPointF(sx + 10, sy + 3), r["filename"][:22])

        # attribution
        p.setPen(QColor("#88888888"))
        p.drawText(self.rect().adjusted(0, -4, -6, 0),
                   Qt.AlignRight | Qt.AlignBottom,
                   "© OpenStreetMap contributors")
        p.end()

    # ------------------------------------------------------------ interaction
    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self._drag = e.position()

    def mouseMoveEvent(self, e):
        if self._drag is not None:
            d = e.position() - self._drag
            self._drag = e.position()
            cx, cy = self._center_px()
            lon, lat = _px_to_lonlat(cx - d.x(), cy - d.y(), self.zoom)
            self.center_lon, self.center_lat = lon, lat
            self.update()

    def mouseReleaseEvent(self, e):
        moved = getattr(self, "_moved", False)
        if e.button() == Qt.LeftButton and not moved:
            lon, lat = self._lonlat_at(e.position())
            hit = None
            best = 14.0
            for r in catalog.photos_with_gps():
                px, py = _lonlat_to_px(r["gps_lon"], r["gps_lat"], self.zoom)
                sx = px + self.width()/2 - self._center_px()[0]
                sy = py + self.height()/2 - self._center_px()[1]
                dd = math.hypot(sx - e.position().x(), sy - e.position().y())
                if dd < best:
                    best, hit = dd, r["id"]
            if hit is not None:
                self.photoPinned.emit(hit)
            else:
                self.mapClicked.emit(lon, lat)
        self._drag = None
        self._moved = False

    def wheelEvent(self, e):
        old = self._lonlat_at(e.position())
        self.zoom = max(1, min(19, self.zoom + (1 if e.angleDelta().y() > 0 else -1)))
        new = self._lonlat_at(e.position())
        dl = old[0] - new[0]
        dla = old[1] - new[1]
        self.center_lon += dl
        self.center_lat += dla
        self.update()


class MapView(QWidget):
    openPhoto = Signal(int)
    statusMessage = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        left = QWidget()
        left.setObjectName("SidePanel")
        left.setFixedWidth(230)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 8, 10)
        lt = QLabel("GEOTAGGED PHOTOS")
        lt.setObjectName("SectionTitle")
        lv.addWidget(lt)
        self.list = QListWidget()
        self.list.itemClicked.connect(
            lambda it: self.openPhoto.emit(it.data(Qt.UserRole)))
        lv.addWidget(self.list, 1)
        self.lbl_hint = QLabel("Right-click the map to place the\n"
                               "current Library selection here.")
        self.lbl_hint.setObjectName("PanelHint")
        self.lbl_hint.setWordWrap(True)
        lv.addWidget(self.lbl_hint)
        h.addWidget(left)

        self.map = SlippyMap()
        self.map.photoPinned.connect(lambda pid: self.openPhoto.emit(pid))
        h.addWidget(self.map, 1)

        self.geotagRequested = None  # set by app: Signal bridge
        from PySide6.QtCore import Signal as _Sig
        if not hasattr(type(self), "geotagHere"):
            type(self).geotagHere = _Sig(float, float)
        b_here = QPushButton("Place selection at map center")
        b_here.clicked.connect(
            lambda: self.geotagHere.emit(self.map.center_lon, self.map.center_lat))
        lv.addWidget(b_here)

    def refresh(self, select_pid: int | None = None):
        self.list.clear()
        rows = catalog.photos_with_gps()
        for r in rows:
            it = QListWidgetItem(f"📍 {r['filename']}")
            it.setData(Qt.UserRole, r["id"])
            it.setToolTip(r["path"])
            self.list.addItem(it)
        if rows:
            last = rows[-1]
            self.map.set_center(last["gps_lon"], last["gps_lat"])
        return rows
