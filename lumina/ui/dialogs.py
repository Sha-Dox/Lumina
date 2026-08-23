"""Import & export dialogs."""
from __future__ import annotations

import os

import numpy as np
from PySide6.QtCore import QObject, Qt, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFileDialog, QHBoxLayout, QLabel, QLineEdit,
                               QProgressBar, QPushButton, QRadioButton,
                               QSlider, QVBoxLayout)

from ..core import catalog, export as excore, rawio


class _ScanBridge(QObject):
    done = Signal(list)


class _ScanTask(QRunnable):
    def __init__(self, bridge, folder):
        super().__init__()
        self.bridge, self.folder = bridge, folder

    def run(self):
        found = []
        for root, dirs, files in os.walk(self.folder):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for f in files:
                p = os.path.join(root, f)
                if rawio.is_supported(p):
                    found.append(p)
        self.bridge.done.emit(found)


class _ImportBridge(QObject):
    progress = Signal(int, int)
    done = Signal(int)


class _ImportTask(QRunnable):
    def __init__(self, bridge, paths):
        super().__init__()
        self.bridge, self.paths = bridge, paths

    def run(self):
        n = len(self.paths)
        for i, p in enumerate(self.paths):
            md = rawio.extract_metadata(p)
            catalog.upsert_photo(p, md)
            self.bridge.progress.emit(i + 1, n)
        self.bridge.done.emit(n)


class ImportDialog(QDialog):
    imported = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Import Photos")
        self.setMinimumWidth(460)
        self.setStyleSheet("QDialog { background:#252525; }")

        v = QVBoxLayout(self)
        v.setSpacing(12)
        row = QHBoxLayout()
        self.lbl_folder = QLabel("Choose a folder to import…")
        row.addWidget(self.lbl_folder, 1)
        b_pick = QPushButton("Browse…")
        b_pick.clicked.connect(self._pick)
        row.addWidget(b_pick)
        v.addLayout(row)

        self.info = QLabel("")
        self.info.setStyleSheet("color:#969696;")
        v.addWidget(self.info)
        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Import")
        bb.button(QDialogButtonBox.Ok).setEnabled(False)
        bb.button(QDialogButtonBox.Ok).setObjectName("Primary")
        bb.accepted.connect(self._run_import)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._paths = []
        self._bridge = _ScanBridge()
        self._bridge.done.connect(self._scan_done)
        self._imp_bridge = _ImportBridge()
        self._imp_bridge.progress.connect(
            lambda i, n: self.bar.setValue(int(i * 100 / max(1, n))))
        self._imp_bridge.done.connect(self._import_done)
        self._pool = QThreadPool.globalInstance()

    def _pick(self):
        d = QFileDialog.getExistingDirectory(self, "Choose folder", os.path.expanduser("~"))
        if not d:
            return
        self.lbl_folder.setText(d)
        self.info.setText("Scanning…")
        self._pool.start(_ScanTask(self._bridge, d))

    def _scan_done(self, paths):
        self._paths = paths
        raws = sum(1 for p in paths if rawio.is_raw(p))
        exts = sorted({os.path.splitext(p)[1].lower() for p in paths})
        self.info.setText(f"{len(paths)} photos found ({raws} RAW)"
                          f"   ·   {', '.join(exts[:8])}")
        bb = self.findChild(QDialogButtonBox)
        bb.button(QDialogButtonBox.Ok).setEnabled(len(paths) > 0)

    def _run_import(self):
        self.bar.setVisible(True)
        self.bar.setRange(0, 100)
        self._pool.start(_ImportTask(self._imp_bridge, self._paths))

    def _import_done(self, n):
        self.imported.emit(n)
        self.accept()


class _ExportBridge(QObject):
    progress = Signal(int, int)
    done = Signal(int)


class _ExportTask(QRunnable):
    def __init__(self, bridge, jobs):
        super().__init__()
        self.bridge, self.jobs = bridge, jobs

    def run(self):
        done = 0
        for path, settings, dest_fmt, quality, max_edge, out_dir, wm in self.jobs:
            try:
                u8 = excore.render_for_export(path, settings,
                                              max_edge=max_edge or None)
                if wm and wm.get("text"):
                    u8 = excore.apply_watermark(u8, wm["text"],
                                                wm.get("opacity", 0.55))
                stem = os.path.splitext(os.path.basename(path))[0] + "_edit"
                dest = excore.unique_path(out_dir, stem, "." + dest_fmt.lower())
                excore.save_image(u8, dest, dest_fmt, quality)
            except Exception as e:
                print("[export]", path, e)
            done += 1
            self.bridge.progress.emit(done, len(self.jobs))
        self.bridge.done.emit(done)


class ExportDialog(QDialog):
    exported = Signal(int)

    def __init__(self, photo_jobs, parent=None):
        """photo_jobs: list of (path, settings_dict) to export."""
        super().__init__(parent)
        self.setWindowTitle("Export Photos")
        self.setMinimumWidth(420)
        self.setStyleSheet("QDialog { background:#252525; }")
        self._jobs_base = photo_jobs

        v = QVBoxLayout(self)
        v.setSpacing(10)
        v.addWidget(QLabel(f"Exporting {len(photo_jobs)} photo"
                           f"{'s' if len(photo_jobs)!=1 else ''}"))

        f_row = QHBoxLayout()
        f_row.addWidget(QLabel("Format"))
        self.fmt = QComboBox()
        self.fmt.addItems(["JPEG", "PNG", "TIFF"])
        f_row.addWidget(self.fmt, 1)
        v.addLayout(f_row)

        q_row = QHBoxLayout()
        q_row.addWidget(QLabel("JPEG quality"))
        self.quality = QComboBox()
        self.quality.addItems(["100", "95", "90", "85", "75"])
        self.quality.setCurrentText("90")
        q_row.addWidget(self.quality, 1)
        v.addLayout(q_row)

        r_row = QHBoxLayout()
        r_row.addWidget(QLabel("Long edge (px)"))
        self.resize_combo = QComboBox()
        self.resize_combo.addItems(["Original", "4096", "2560", "1920", "1080"])
        r_row.addWidget(self.resize_combo, 1)
        v.addLayout(r_row)

        wm_row = QHBoxLayout()
        self.chk_wm = QCheckBox("Watermark")
        wm_row.addWidget(self.chk_wm)
        self.wm_text = QLineEdit("© Lumina")
        self.wm_text.setEnabled(False)
        wm_row.addWidget(self.wm_text, 1)
        self.chk_wm.toggled.connect(lambda on: self.wm_text.setEnabled(on))
        v.addLayout(wm_row)

        d_row = QHBoxLayout()
        self.lbl_dir = QLabel(os.path.expanduser("~/Pictures/Lumina Exports"))
        d_row.addWidget(self.lbl_dir, 1)
        b_dir = QPushButton("Choose…")
        b_dir.clicked.connect(self._pick_dir)
        d_row.addWidget(b_dir)
        v.addLayout(d_row)

        self.bar = QProgressBar()
        self.bar.setVisible(False)
        v.addWidget(self.bar)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Export")
        bb.button(QDialogButtonBox.Ok).setObjectName("Primary")
        bb.accepted.connect(self._run)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._out_dir = self.lbl_dir.text()
        self._bridge = _ExportBridge()
        self._bridge.progress.connect(lambda i, n: self.bar.setValue(int(i*100/max(1,n))))
        self._bridge.done.connect(self._done)
        self._pool = QThreadPool.globalInstance()

    def _pick_dir(self):
        d = QFileDialog.getExistingDirectory(self, "Export to",
                                             os.path.expanduser("~/Pictures"))
        if d:
            self._out_dir = d
            self.lbl_dir.setText(d)

    def _run(self):
        fmt = self.fmt.currentText()
        quality = int(self.quality.currentText())
        re_txt = self.resize_combo.currentText()
        max_edge = None if re_txt == "Original" else int(re_txt)
        os.makedirs(self._out_dir, exist_ok=True)
        wm = None
        if self.chk_wm.isChecked() and self.wm_text.text().strip():
            wm = {"text": self.wm_text.text().strip(), "opacity": 0.55}
        jobs = [(p, s, fmt, quality, max_edge, self._out_dir, wm)
                for p, s in self._jobs_base]
        self.bar.setVisible(True)
        self.bar.setRange(0, 100)
        self._pool.start(_ExportTask(self._bridge, jobs))

    def _done(self, n):
        self.exported.emit(n)
        self.accept()
