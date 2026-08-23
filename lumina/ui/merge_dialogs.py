"""HDR / Panorama merge dialogs."""
from __future__ import annotations

import os

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal
from PySide6.QtWidgets import (QDialog, QDialogButtonBox, QLabel,
                               QProgressBar, QVBoxLayout)

from ..core import catalog
from ..core import merge as mergemod


class _MergeBridge(QObject):
    progress = Signal(int, int)
    done = Signal(object)      # dict result


class _MergeTask(QRunnable):
    def __init__(self, bridge, kind, paths):
        super().__init__()
        self.bridge, self.kind, self.paths = bridge, kind, paths

    def run(self):
        try:
            if self.kind == "hdr":
                u8 = mergemod.merge_hdr(self.paths,
                                        progress=lambda i, n: self.bridge.progress.emit(i, n))
            else:
                u8 = mergemod.stitch_panorama(
                    self.paths, progress=lambda i, n: self.bridge.progress.emit(i, n))
            out_path = mergemod.save_merged(u8, "HDR" if self.kind == "hdr" else "Pano")
            from ..core import rawio
            pid = catalog.upsert_photo(out_path, rawio.extract_metadata(out_path))
            self.bridge.done.emit({"ok": True, "path": out_path, "id": pid})
        except Exception as e:
            self.bridge.done.emit({"ok": False, "error": str(e)})


class MergeDialog(QDialog):
    """Runs HDR or Pano merge over given photo ids; imports the result."""

    finished_merge = Signal(int)      # new photo id

    def __init__(self, kind: str, photo_rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HDR Merge" if kind == "hdr" else "Panorama")
        self.setMinimumWidth(420)
        self.setStyleSheet("QDialog { background:#252525; }")
        self._kind = kind

        v = QVBoxLayout(self)
        v.setSpacing(10)
        names = ", ".join(os.path.basename(r["filename"]) for r in photo_rows[:6])
        more = f" …+{len(photo_rows)-6}" if len(photo_rows) > 6 else ""
        info = ("Exposure-fusing brackets into one tone-mapped photo."
                if kind == "hdr" else
                "Stitching overlapping shots into a panorama.")
        v.addWidget(QLabel(f"<b>{info}</b>"))
        lbl = QLabel(f"{len(photo_rows)} photos:\n{names}{more}")
        lbl.setStyleSheet("color:#969696;")
        lbl.setWordWrap(True)
        v.addWidget(lbl)

        self.status = QLabel("")
        self.status.setStyleSheet("color:#969696;")
        v.addWidget(self.status)
        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        v.addWidget(self.bar)

        bb = QDialogButtonBox(QDialogButtonBox.Cancel)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

        self._bridge = _MergeBridge()
        self._bridge.progress.connect(self._progress)
        self._bridge.done.connect(self._done)
        self._pool = QThreadPool.globalInstance()
        paths = [r["path"] for r in photo_rows]
        self._pool.start(_MergeTask(self._bridge, kind, paths))

    def _progress(self, i, n):
        self.status.setText(f"Processing {i}/{n}…")
        self.bar.setValue(int(i * 100 / max(1, n)))

    def _done(self, res: dict):
        if res.get("ok"):
            self.status.setText(f"Saved: {res['path']}")
            self.finished_merge.emit(res["id"])
            QTimer.singleShot(600, self.accept)
        else:
            self.status.setText(f"Failed: {res['error']}")
