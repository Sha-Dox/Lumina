"""Sync develop settings across multiple photos."""
from __future__ import annotations

from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QVBoxLayout)

GROUPS = {
    "White Balance": ["temp", "tint"],
    "Tone": ["exposure", "contrast", "highlights", "shadows",
             "whites", "blacks"],
    "Presence": ["clarity", "dehaze", "vibrance", "saturation", "bw"],
    "Tone Curve": ["curve_rgb", "curve_r", "curve_g", "curve_b"],
    "Color Mixer": ["hsl"],
    "Color Grading": ["grade_shadows", "grade_midtones", "grade_highlights",
                      "blender", "balance"],
    "Detail": ["sharp_amount", "sharp_radius", "nr_lum", "nr_color"],
    "Effects": ["vignette_amount", "vignette_midpoint", "vignette_feather",
                "grain_amount", "grain_size", "glow_amount"],
    "AI Tools": ["sky_enabled", "sky_preset", "sky_strength",
                 "sky_softness", "sky_offset",
                 "relight_angle", "relight_strength", "lut_file",
                 "lut_enabled"],
}
DEFAULT_CHECKED = {"White Balance", "Tone", "Presence", "Color Mixer",
                   "Color Grading"}


class SyncDialog(QDialog):
    """Returns set of chosen group names via chosen_groups()."""

    def __init__(self, source_name: str, n_targets: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Synchronize Settings")
        self.setMinimumWidth(360)
        self.setStyleSheet("QDialog { background:#252525; }")

        v = QVBoxLayout(self)
        v.addWidget(QLabel(
            f"<b>Copy develop settings from</b><br>{source_name}<br>"
            f"<b>to {n_targets} other photo{'s' if n_targets != 1 else ''}?</b>"))

        self._boxes = {}
        for group, keys in GROUPS.items():
            cb = QCheckBox(group)
            cb.setChecked(group in DEFAULT_CHECKED)
            cb.stateChanged.connect(lambda _, k=keys, b=cb: None)
            v.addWidget(cb)
            self._boxes[group] = (cb, keys)

        note = QLabel("Geometry, crop and masks are never synced.")
        note.setStyleSheet("color:#888;")
        v.addWidget(note)

        bb = QDialogButtonBox(QDialogButtonBox.Apply |
                              QDialogButtonBox.Cancel)
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        v.addWidget(bb)

    def chosen_groups(self) -> dict[str, list[str]]:
        return {g: keys for g, (cb, keys) in self._boxes.items()
                if cb.isChecked()}
