"""Tethered capture — native ImageCaptureCore devices + watched-folder import."""
from __future__ import annotations

import os
import time

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtWidgets import (QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QLabel, QListWidget, QPushButton, QVBoxLayout)

from ..core import catalog, rawio


class _ICCController(QObject):
    """Best-effort native camera control via Apple ImageCaptureCore."""
    devicesChanged = Signal(list)
    captured = Signal(str)
    error = Signal(str)

    def __init__(self):
        super().__init__()
        self.available = False
        self._device = None
        try:
            import ImageCaptureCore as ICCA
            self._ICCA = ICCA
            self._browser = ICCA.ICDeviceBrowser.alloc().init()
            self._browser.setDelegate_(self)
            self._browser.start()
            self.available = True
        except Exception as e:
            self.error.emit(f"ImageCapture unavailable: {e}")

    def browsedDeviceDescriptionForDevice_(self, *a):  # noqa
        return None

    # ICDeviceBrowser delegate (pyobjc informal protocol)
    def deviceBrowser_didAddDevice_moreComing_(self, browser, dev, more):
        try:
            if dev.deviceType() == self._ICCA.ICDeviceTypeCamera:
                self.devicesChanged.emit(self.camera_names())
        except Exception:
            pass

    def deviceBrowser_didRemoveDevice_moreComing_(self, browser, dev, more):
        try:
            self.devicesChanged.emit(self.camera_names())
        except Exception:
            pass

    def deviceBrowser_requestsSelectDevice_(self, browser, dev):
        return True

    def camera_names(self):
        out = []
        if not self.available:
            return out
        for d in (self._browser.devices() or []):
            try:
                if d.deviceType() == self._ICCA.ICDeviceTypeCamera:
                    out.append(str(d.name()))
            except Exception:
                pass
        return out

    def open_session(self, name: str) -> bool:
        if not self.available:
            return False
        try:
            for d in (self._browser.devices() or []):
                if str(d.name()) == name:
                    d.requestOpenSession()
                    self._device = d
                    return True
        except Exception as e:
            self.error.emit(str(e))
        return False

    def take_picture(self) -> bool:
        try:
            if self._device is not None:
                from Foundation import NSObject
                self._device.requestTakePicture()
                return True
        except Exception as e:
            self.error.emit(str(e))
        return False


class TetherDialog(QDialog):
    imported = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Tethered Capture")
        self.setMinimumSize(520, 420)
        self.setStyleSheet("QDialog { background:#252525; }")
        self._watch_dir = None
        self._known = set()
        self._imported_count = 0

        v = QVBoxLayout(self)
        v.setSpacing(10)

        t1 = QLabel("CAMERA (Apple ImageCapture)")
        t1.setObjectName("SectionTitle")
        v.addWidget(t1)
        crow = QHBoxLayout()
        self.cam_combo = QComboBox()
        crow.addWidget(self.cam_combo, 1)
        b_refresh = QPushButton("Refresh")
        b_refresh.clicked.connect(self._refresh_cams)
        crow.addWidget(b_refresh)
        b_open = QPushButton("Use")
        b_open.clicked.connect(self._open_cam)
        crow.addWidget(b_open)
        v.addLayout(crow)
        b_shoot = QPushButton("\u25cf Shoot")
        b_shoot.setObjectName("Primary")
        b_shoot.clicked.connect(self._shoot)
        v.addWidget(b_shoot)
        self.lbl_cam = QLabel("")
        self.lbl_cam.setStyleSheet("color:#969696;")
        self.lbl_cam.setWordWrap(True)
        v.addWidget(self.lbl_cam)

        t2 = QLabel("WATCHED FOLDER")
        t2.setObjectName("SectionTitle")
        v.addWidget(t2)
        wrow = QHBoxLayout()
        self.lbl_dir = QLabel("Choose a folder your camera app saves into…")
        wrow.addWidget(self.lbl_dir, 1)
        b_pick = QPushButton("Browse…")
        b_pick.clicked.connect(self._pick_watch)
        wrow.addWidget(b_pick)
        v.addLayout(wrow)
        self.lbl_status = QLabel("Watching idle.")
        self.lbl_status.setStyleSheet("color:#969696;")
        v.addWidget(self.lbl_status)

        self.file_list = QListWidget()
        v.addWidget(self.file_list, 1)

        bb = QHBoxLayout()
        bb.addStretch(1)
        b_close = QPushButton("Close")
        b_close.clicked.connect(self.accept)
        bb.addWidget(b_close)
        v.addLayout(bb)

        self.icc = _ICCController()
        self.icc.devicesChanged.connect(self._cams)
        self.icc.error.connect(lambda m: self.lbl_cam.setText(m))
        self._refresh_cams()

        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self._poll_watch)
        self.timer.start()

    # ---------------- native camera
    def _refresh_cams(self):
        names = self.icc.camera_names()
        self.cam_combo.clear()
        if not self.icc.available:
            self.lbl_cam.setText(
                "ImageCaptureCore unavailable in this environment.")
        elif names:
            self.cam_combo.addItems(names)
            self.lbl_cam.setText("Camera detected. Press Use, then Shoot.")
        else:
            self.lbl_cam.setText(
                "No camera found. Connect via USB/Wi-Fi, or use Watched Folder "
                "with your manufacturer's transfer app.")

    def _cams(self, names):
        cur = self.cam_combo.currentText()
        self.cam_combo.clear()
        self.cam_combo.addItems(names)
        if cur:
            i = self.cam_combo.findText(cur)
            if i >= 0:
                self.cam_combo.setCurrentIndex(i)

    def _open_cam(self):
        name = self.cam_combo.currentText()
        if name and self.icc.open_session(name):
            self.lbl_cam.setText(f"Session open: {name}")
        else:
            self.lbl_cam.setText("Could not open a session for that camera.")

    def _shoot(self):
        if self.icc.take_picture():
            self.lbl_status.setText("Capture triggered — waiting for file…")
        else:
            self.lbl_cam.setText("Shoot failed — is a session open?")

    # ---------------- watch folder
    def _pick_watch(self):
        d = QFileDialog.getExistingDirectory(self, "Watch folder",
                                             os.path.expanduser("~/Pictures"))
        if d:
            self._watch_dir = d
            self.lbl_dir.setText(d)
            self._known = set()
            self._poll_watch()

    def _poll_watch(self):
        if not self._watch_dir:
            return
        new_files = []
        for root, dirs, files in os.walk(self._watch_dir):
            dirs[:] = [x for x in dirs if not x.startswith(".")]
            for f in files:
                p = os.path.join(root, f)
                if rawio.is_supported(p) and p not in self._known:
                    try:
                        if time.time() - os.path.getmtime(p) < 86400 * 3650:
                            new_files.append(p)
                        self._known.add(p)
                    except OSError:
                        pass
        for p in sorted(new_files):
            md = rawio.extract_metadata(p)
            pid = catalog.upsert_photo(p, md)
            rawio.make_thumbnail(p)
            self.file_list.insertItem(0, os.path.basename(p))
            self._imported_count += 1
            self.imported.emit(pid)
        if new_files:
            self.lbl_status.setText(
                f"Imported {len(new_files)} new photo(s) · total {self._imported_count}")

    def closeEvent(self, e):
        try:
            if self.icc.available:
                self.icc._browser.stop()
        except Exception:
            pass
        super().closeEvent(e)
