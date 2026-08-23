"""Auto-cull batch dialog."""
from __future__ import annotations

import threading
import time

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QHBoxLayout, QLabel,
                               QProgressBar, QPushButton, QVBoxLayout)

from ..core.cull import analyze, assign_ratings
from ..core import rawio


class CullDialog(QDialog):
    finishedCull = Signal(int, int)      # n_scored, n_rejected

    def __init__(self, rows, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Auto-Cull")
        self.setMinimumWidth(400)
        self.setStyleSheet("QDialog { background:#252525; }")
        self.rows = rows
        self.results_out = []

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            f"<b>Score {len(rows)} photos</b><br>"
            "<span style='color:#969696'>Sharpness + exposure + people "
            "→ suggested ratings (★3 best / ★2 keep / ★1 weak).</span>"))
        self.chk_respect = QCheckBox("Don't touch photos already rated")
        self.chk_respect.setChecked(True)
        v.addWidget(self.chk_respect)
        self.chk_reject = QCheckBox("Flag blurry shots as Rejected")
        v.addWidget(self.chk_reject)

        self.bar = QProgressBar()
        self.bar.setRange(0, len(rows))
        v.addWidget(self.bar)
        self.status = QLabel("")
        self.status.setStyleSheet("color:#969696;")
        v.addWidget(self.status)

        bb = QHBoxLayout()
        bb.addStretch(1)
        self.b_close = QPushButton("Close")
        self.b_close.clicked.connect(self.reject)
        bb.addWidget(self.b_close)
        v.addLayout(bb)

    def start(self):
        import threading
        threading.Thread(target=self._work, daemon=True).start()

    def _work(self):
        results, ids = [], []
        use_faces = True
        for i, r in enumerate(self.rows):
            try:
                prev = rawio.decode_preview(r["path"], 480)
                res = analyze(prev, use_faces=use_faces)
            except Exception as e:
                print("[cull]", e)
                res = {"score": -99, "blurry": False}
            results.append(res)
            ids.append(r["id"])
            self.bar.setValue(i + 1)
            self.status.setText(f"Analyzed {i+1}/{len(self.rows)}")
        rated = assign_ratings(results, ids,
                               respect_existing=self.chk_respect.isChecked(),
                               mark_rejects=self.chk_reject.isChecked())
        rejects = 0
        applied = 0
        from ..core import catalog
        for pid, rating, flag, _res in rated:
            row = catalog.get_photo(pid)
            if not row:
                continue
            if self.chk_respect.isChecked() and (row["rating"] or 0) > 0:
                continue
            catalog.update_fields(pid, rating=rating)
            if flag is not None:
                catalog.update_fields(pid, flag=flag)
                rejects += 1
            applied += 1
        self.results_out = (applied, rejects)
        self.finishedCull.emit(applied, rejects)
