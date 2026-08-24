"""Develop module - adjustment panels, render orchestration, presets, history."""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QSlider
from PySide6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QFrame,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..core import aimask, catalog, heal as healmod, imaging, rawio
from . import theme
from .develop_canvas import np_to_pixmap, DevelopCanvas
from .scopes import ScopeView
from .widgets import (CollapsibleSection, ColorWheel,
                      SliderRow, ToneCurveWidget)

PRESET_DIR = os.path.expanduser("~/.lumina/presets")
os.makedirs(PRESET_DIR, exist_ok=True)

BANDS = imaging.HSL_BANDS

BUILTIN_PRESETS = {
    "Punch": {"contrast": 18, "vibrance": 24, "clarity": 12, "blacks": -12, "whites": 8},
    "Warm Portrait": {"temp": 7, "tint": 2, "exposure": 0.15, "shadows": 20,
                      "highlights": -14, "clarity": -6, "saturation": -5},
    "Teal & Orange": {"contrast": 10, "grade_shadows": [198, 38, -6],
                      "grade_midtones": [30, 12, 0], "grade_highlights": [38, 26, 4]},
    "Faded Film": {"contrast": -14, "shadows": 22, "blacks": 14, "saturation": -18,
                   "grain_amount": 32, "curve_rgb": [(0.12, 0.16), (0.85, 0.82)]},
    "B&W Classic": {"bw": True, "contrast": 20, "clarity": 12, "vignette_amount": -14},
    "Crisp Landscape": {"vibrance": 26, "clarity": 20, "dehaze": 12, "whites": 12,
                        "blacks": -8, "sharp_amount": 55},
    "Moody": {"temp": -12, "exposure": -0.25, "contrast": 14, "shadows": -12,
              "vignette_amount": -28, "saturation": -10},
    "Golden Hour": {"temp": 18, "tint": 4, "exposure": 0.1, "highlights": -18,
                    "grade_midtones": [42, 20, 2], "vibrance": 12},
    # --- film emulations
    "Portra Warm": {"temp": 8, "tint": 3, "exposure": 0.15, "contrast": -6,
                    "shadows": 14, "blacks": 6, "saturation": -8,
                    "vibrance": 16, "glow_amount": 10,
                    "hsl": {"red": [3, 8, 5], "orange": [-2, 12, 8],
                            "yellow": [0, -8, 5], "green": [-5, -20, -3],
                            "aqua": [0, 0, 0], "blue": [0, -5, -3],
                            "purple": [0, 0, 0], "magenta": [0, 0, 0]}},
    "Velvia Punch": {"contrast": 28, "saturation": 22, "vibrance": 18,
                     "blacks": -18, "whites": 10, "clarity": 15,
                     "sharp_amount": 50,
                     "hsl": {"red": [0, 25, 0], "green": [5, 30, -5],
                             "blue": [0, 20, -5]}},
    "Tri-X Grain": {"bw": True, "contrast": 24, "clarity": 18,
                    "blacks": -14, "grain_amount": 42, "grain_size": 32,
                    "sharp_amount": 40, "vignette_amount": -18},
    "Cinematic Teal": {"temp": -6, "contrast": 18, "shadows": -8,
                       "grade_shadows": [192, 35, -8],
                       "grade_midtones": [180, 10, 0],
                       "grade_highlights": [38, 22, 6],
                       "saturation": -6, "curve_rgb": [[0.1, 0.13], [0.85, 0.88]]},
    "Kodachrome": {"contrast": 22, "saturation": 12, "vibrance": 8,
                   "blacks": -12, "clarity": 10,
                   "hsl": {"red": [2, 18, 3], "yellow": [-3, 15, 0],
                           "green": [-8, 22, -8], "blue": [0, 15, -4]},
                   "sharp_amount": 45},
}

WB_PRESETS = {
    "As Shot": None,
    "Auto": "auto",
    "Daylight": (5, -2),
    "Cloudy": (13, 4),
    "Shade": (21, 8),
    "Tungsten": (-30, 7),
    "Fluorescent": (-13, 11),
    "Flash": (8, -3),
}

ASPECTS = ["free", "orig", "1:1", "4:3", "3:2", "16:9", "9:16"]


def cv2_resize_small(f32: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = f32.shape[:2]
    sc = long_edge / max(h, w)
    if sc >= 1.0:
        return f32
    import cv2
    return cv2.resize(f32, (int(w * sc), int(h * sc)),
                      interpolation=cv2.INTER_AREA)


def settings_for_save(s: dict) -> dict:
    out = copy.deepcopy(s)
    for m in out.get("masks", []):
        m.get("params", {}).pop("_subject_array", None)
    return out


class _RenderBridge(QObject):
    rendered = Signal(int, int, object)


class _RenderTask(QRunnable):
    def __init__(self, bridge, gen, tier, img_f32, settings, seed):
        super().__init__()
        self.bridge, self.gen, self.tier = bridge, gen, tier
        self.img, self.settings, self.seed = img_f32, settings, seed

    def run(self):
        try:
            if not self.bridge or not self.bridge.parent():
                return
            s = self.settings
            out = imaging.render_global(self.img, s, scale=max(0.5, self.tier),
                                        seed_key=self.seed)
            ang = float(s.get("straighten", 0.0))
            if abs(ang) > 0.01:
                out = imaging.apply_straighten(out, ang)
            out = imaging.apply_transform(out, s.get("transform_v", 0.0),
                                          s.get("transform_h", 0.0),
                                          s.get("transform_scale", 0.0))
            shape = out.shape
            for m in s.get("masks", []):
                try:
                    marr = imaging.rasterize_mask(shape, m)
                    out = imaging.apply_mask_to_image(
                        out, marr, m["adjustments"], bool(m.get("invert")),
                        max(0.5, self.tier))
                except Exception:
                    continue
            self.bridge.rendered.emit(self.gen, self.tier, out)
        except RuntimeError:
            pass
        except Exception as e:
            print("[render] error:", e)


class _AiBridge(QObject):
    done = Signal(object, object)


class _AiTask(QRunnable):
    def __init__(self, bridge, u8, mode, tag):
        super().__init__()
        self.bridge, self.u8, self.mode, self.tag = bridge, u8, mode, tag

    def run(self):
        try:
            mask = aimask.compute_subject_mask(self.u8, mode=self.mode)
            if mask is None:
                self.bridge.done.emit("__nosubject__", {"mode": self.mode})
            else:
                self.bridge.done.emit(mask, {"mode": self.mode})
        except aimask.NoSubjectFound:
            self.bridge.done.emit("__nosubject__", {"mode": self.mode})
        except Exception as e:
            print("[ai]", e)
            self.bridge.done.emit("__error__", {"error": str(e)})


class PresetStore:
    @staticmethod
    def all_presets() -> dict:
        out = {k: copy.deepcopy(v) for k, v in BUILTIN_PRESETS.items()}
        for fn in sorted(os.listdir(PRESET_DIR)):
            if fn.endswith(".json"):
                try:
                    with open(os.path.join(PRESET_DIR, fn)) as f:
                        out[fn[:-5]] = json.load(f)
                except Exception:
                    pass
        return out

    @staticmethod
    def save_user(name: str, partial: dict) -> bool:
        try:
            with open(os.path.join(PRESET_DIR, f"{name}.json"), "w") as f:
                json.dump(partial, f)
            return True
        except Exception:
            return False

    @staticmethod
    def delete(name: str) -> None:
        p = os.path.join(PRESET_DIR, f"{name}.json")
        if os.path.exists(p):
            os.remove(p)


class DevelopView(QWidget):
    statusMessage = Signal(str)
    photoEdited = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pool = QThreadPool.globalInstance()
        self._bridge = _RenderBridge(self)
        self._bridge.rendered.connect(self._on_rendered)
        self._ai_bridge = _AiBridge(self)
        self._ai_bridge.done.connect(self._on_ai_done)

        self.photo_id = None
        self.photo_path = None
        self.settings = imaging.default_settings()
        self._clipboard = None
        self._gen = 0
        self._busy_tiers = set()
        self._work_f32 = None
        self._work_u8 = None
        self._full_f32 = None
        self._result_quality = None
        self._history = []
        self._hist_index = -1
        self._active_mask = -1
        self.uw_unit_ft = False
        self._last_label = ''
        self._suppress_panel_sync = False
        # progressive-resolution buffers (all oriented, healed)
        self._oriented_u8 = None      # raw after flip/rot90
        self._heal_hash = None
        self._drag_f32 = None         # ~640px during slider drags
        self._heal_cache = (None, None)

        self._drag_timer = QTimer(self)
        self._drag_timer.setSingleShot(True)
        self._drag_timer.setInterval(25)
        self._drag_timer.timeout.connect(lambda: self._start_render(-1))
        self._hist_timer2 = QTimer(self)
        self._hist_timer2.setSingleShot(True)
        self._hist_timer2.setInterval(500)
        self._hist_timer2.timeout.connect(lambda: self._start_render(0))

        self._fast_timer = QTimer(self)
        self._fast_timer.setSingleShot(True)
        self._fast_timer.setInterval(30)
        self._fast_timer.timeout.connect(lambda: self._start_render(0))
        self._quality_timer = QTimer(self)
        self._quality_timer.setSingleShot(True)
        self._quality_timer.setInterval(420)
        self._quality_timer.timeout.connect(self._load_full_and_render)
        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(900)
        self._save_timer.timeout.connect(self._persist)
        self._hist_timer = QTimer(self)
        self._hist_timer.setSingleShot(True)
        self._hist_timer.setInterval(650)
        self._hist_timer.timeout.connect(self._push_history)

        self._build_ui()

    # ================================================================ UI
    def _tool_btn(self, tl, text, tip, cb, checkable=False):
        b = QPushButton(text)
        b.setObjectName("FlatTool")
        b.setToolTip(tip)
        b.setCheckable(checkable)
        b.setCursor(Qt.PointingHandCursor)
        b.clicked.connect(cb)
        tl.addWidget(b)
        return b

    def _build_ui(self):
        h = QHBoxLayout(self)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)

        # ---------------- left: presets & history
        left = QWidget()
        left.setObjectName("SidePanel")
        left.setFixedWidth(212)
        lv = QVBoxLayout(left)
        lv.setContentsMargins(10, 10, 8, 10)
        lv.setSpacing(8)

        pt = QLabel("PRESETS")
        pt.setObjectName("SectionTitle")
        lv.addWidget(pt)
        self.preset_list = QListWidget()
        self.preset_list.itemClicked.connect(self._apply_preset_item)
        lv.addWidget(self.preset_list, 3)
        ph = QHBoxLayout()
        b_save_p = QPushButton("Save…")
        b_save_p.clicked.connect(self._save_preset_dialog)
        b_del_p = QPushButton("Delete")
        b_del_p.clicked.connect(self._delete_preset)
        ph.addWidget(b_save_p)
        ph.addWidget(b_del_p)
        lv.addLayout(ph)

        ht = QLabel("HISTORY")
        ht.setObjectName("SectionTitle")
        lv.addWidget(ht)
        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self._goto_history)
        lv.addWidget(self.history_list, 2)

        cp = QHBoxLayout()
        b_copy = QPushButton("Copy")
        b_copy.setToolTip("Copy settings (Cmd+C)")
        b_copy.clicked.connect(self.copy_settings)
        b_paste = QPushButton("Paste")
        b_paste.setToolTip("Paste settings (Cmd+V)")
        b_paste.clicked.connect(self.paste_settings)
        cp.addWidget(b_copy)
        cp.addWidget(b_paste)
        lv.addLayout(cp)

        nav = QHBoxLayout()
        b_prev = QPushButton("< Prev")
        b_next = QPushButton("Next >")
        b_prev.clicked.connect(lambda: self.navigate(-1))
        b_next.clicked.connect(lambda: self.navigate(1))
        nav.addWidget(b_prev)
        nav.addWidget(b_next)
        lv.addLayout(nav)

        b_reset = QPushButton("Reset All Edits")
        b_reset.clicked.connect(self.reset_all)
        lv.addWidget(b_reset)
        h.addWidget(left)

        # ---------------- center: toolbar + canvas
        center = QWidget()
        cv = QVBoxLayout(center)
        cv.setContentsMargins(0, 0, 0, 0)
        cv.setSpacing(0)

        tools = QWidget()
        tools.setObjectName("TopBar")
        tl = QHBoxLayout(tools)
        tl.setContentsMargins(10, 4, 10, 4)
        tl.setSpacing(6)

        self.b_crop = self._tool_btn(tl, "Crop", "Crop & straighten (R)",
                                     self.toggle_crop, True)
        self.b_heal = self._tool_btn(tl, "Heal", "Spot removal (Q) — click blemishes",
                                     self.toggle_spot_mode, True)
        self.b_before = self._tool_btn(tl, "Before", "Hold to view original (\\)",
                                       lambda: False)
        self.b_split = self._tool_btn(tl, "Split", "Before/After split view",
                                      self.toggle_split, True)
        self.b_before.pressed.connect(lambda: self.canvas.toggle_before(True))
        self.b_before.released.connect(lambda: self.canvas.toggle_before(False))
        self.canvas = DevelopCanvas()

        self.b_fit = self._tool_btn(tl, "Fit", "Zoom to fit (F)",
                                    self.canvas.set_zoom_fit)
        self.b_100 = self._tool_btn(tl, "100%", "Zoom to 100% (+/-)",
                                    self.canvas.set_zoom_100)
        tl.addWidget(QLabel("  "))  # separator
        self.b_rotl = self._tool_btn(tl, "\u21ba", "Rotate left",
                                     lambda: self._rotate90(-1))
        self.b_rotr = self._tool_btn(tl, "\u21bb", "Rotate right",
                                     lambda: self._rotate90(1))
        self.b_fliph = self._tool_btn(tl, "\u2194", "Flip horizontal",
                                      lambda: self._flip_h())
        self.b_undo = self._tool_btn(tl, "Undo", "Undo (Cmd+Z)", self.undo)
        self.b_redo = self._tool_btn(tl, "Redo", "Redo (Shift+Cmd+Z)", self.redo)
        tl.addStretch(2)
        cv.addWidget(tools)

        self.canvas.cropChanged.connect(self._crop_live)
        self.canvas.cropCommitted.connect(self._crop_commit)
        self.canvas.maskDrawn.connect(self._mask_drawn)
        self.canvas.brushStrokeFinished.connect(self._brush_stroke)
        self.canvas.spotsChanged.connect(self._spots_from_canvas)
        self.canvas.subjectPicked.connect(self._subject_picked)
        cv.addWidget(self.canvas, 1)
        h.addWidget(center, 1)

        # ---------------- right: scroll of sections
        right = QWidget()
        right.setObjectName("SidePanel")
        right.setFixedWidth(312)
        rscroll = QScrollArea()
        rscroll.setWidgetResizable(True)
        rscroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        rscroll.setFrameShape(QScrollArea.NoFrame)
        rv_container = QWidget()
        rv_container.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        rv = QVBoxLayout(rv_container)
        rv.setContentsMargins(10, 8, 12, 20)
        rv.setSpacing(4)

        self.histogram = ScopeView()
        self.histogram.setToolTip("Click to cycle scopes")
        rv.addWidget(self.histogram)

        rv.addWidget(self._build_aitools_section())
        rv.addWidget(self._build_underwater_section())
        rv.addWidget(self._build_basic_section())
        rv.addWidget(self._build_curve_section())
        rv.addWidget(self._build_hsl_section())
        rv.addWidget(self._build_grading_section())
        rv.addWidget(self._build_detail_section())
        rv.addWidget(self._build_lens_section())
        rv.addWidget(self._build_effects_section())
        rv.addWidget(self._build_geometry_section())
        rv.addWidget(self._build_transform_section())
        rv.addWidget(self._build_calibration_section())
        rv.addWidget(self._build_masking_section())
        rv.addStretch(1)

        rscroll.setWidget(rv_container)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(rscroll)
        h.addWidget(right)

        self.rebuild_preset_list()

    def _bind_slider(self, s: SliderRow, key: str, label: str):
        def live(v):
            if not self._suppress_panel_sync:
                self.settings[key] = v
                self._live_change()
        def final(v):
            if not self._suppress_panel_sync:
                self.settings[key] = v
                self._changed(label, immediate_history=True)
        s.valueChanged.connect(live)
        s.editingFinished.connect(final)

    def _build_aitools_section(self):
        sec = CollapsibleSection("AI Tools", False)

        row1 = QHBoxLayout()
        b_enh = QPushButton("\u2728 AI Enhance")
        b_enh.setObjectName("Primary")
        b_enh.setToolTip("One-click smart enhancement: exposure, WB, tone, color")
        b_enh.clicked.connect(self._ai_enhance)
        b_denoise = QPushButton("Noiseless")
        b_denoise.setToolTip("Strong noise reduction preset")
        b_denoise.clicked.connect(self._ai_noiseless)
        row1.addWidget(b_enh); row1.addWidget(b_denoise)
        sec.body_lay.addLayout(row1)

        sky_lbl = QLabel("SKY REPLACEMENT")
        sky_lbl.setObjectName("SectionTitle")
        sec.add(sky_lbl)
        self.chk_sky = QCheckBox("Replace sky")
        self.chk_sky.toggled.connect(self._sky_toggled)
        sec.add(self.chk_sky)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Preset"))
        self.sky_preset_combo = QComboBox()
        from ..core.sky import PRESETS
        self.sky_preset_combo.addItems(PRESETS)
        self.sky_preset_combo.currentTextChanged.connect(self._sky_changed)
        srow.addWidget(self.sky_preset_combo, 1)
        sec.body_lay.addLayout(srow)

        self.s_sky_str = SliderRow("Strength", 0, 100, 75)
        self.s_sky_soft = SliderRow("Edge Softness", 0, 100, 45)
        self.s_sky_off = SliderRow("Horizon Shift", -100, 100, 0)
        for r, key in ((self.s_sky_str, "sky_strength"),
                       (self.s_sky_soft, "sky_softness"),
                       (self.s_sky_off, "sky_offset")):
            sec.add(r)
            r.valueChanged.connect(lambda v, k=key: self._live_key(k, float(v)))
            r.editingFinished.connect(
                lambda v, k=key: self._commit_key(k, float(v), "Sky Replace"))

        rel_lbl = QLabel("RELIGHT AI")
        rel_lbl.setObjectName("SectionTitle")
        sec.add(rel_lbl)
        self.s_rel_dir = SliderRow("Direction", 0, 360, 300)
        self.s_rel_str = SliderRow("Strength", -100, 100, 0)
        for r, key in ((self.s_rel_dir, "relight_angle"),
                       (self.s_rel_str, "relight_strength")):
            sec.add(r)
            r.valueChanged.connect(lambda v, k=key: self._live_key(k, float(v)))
            r.editingFinished.connect(
                lambda v, k=key: self._commit_key(k, float(v), "Relight AI"))

        lut_lbl = QLabel("LOOK LUT")
        lut_lbl.setObjectName("SectionTitle")
        sec.add(lut_lbl)
        lut_row = QHBoxLayout()
        b_imp = QPushButton("Import .cube…")
        b_imp.clicked.connect(self._import_lut)
        b_exp = QPushButton("Export Look…")
        b_exp.clicked.connect(self._export_look_lut)
        b_clr = QPushButton("None")
        b_clr.clicked.connect(self._clear_lut)
        lut_row.addWidget(b_imp); lut_row.addWidget(b_exp); lut_row.addWidget(b_clr)
        sec.body_lay.addLayout(lut_row)
        self.lbl_lut = QLabel("")
        self.lbl_lut.setStyleSheet("color:#969696; font-size:10px;")
        sec.add(self.lbl_lut)
        return sec

    def _import_lut(self):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getOpenFileName(self, "Import LUT", "",
                                           "Cube files (*.cube)")
        if not p:
            return
        try:
            from ..core.lutio import parse_cube
            parse_cube(p)          # validate
        except Exception as e:
            self.statusMessage.emit(f"LUT load failed: {e}")
            return
        self.settings["lut_path"] = p
        self.settings["lut_enabled"] = True
        self.lbl_lut.setText(os.path.basename(p))
        Pipeline.invalidateLUTCache if False else None
        self._changed("Import LUT", immediate_history=True)
        self.statusMessage.emit(f"LUT loaded: {os.path.basename(p)}")

    def _export_look_lut(self):
        from PySide6.QtWidgets import QFileDialog
        p, _ = QFileDialog.getSaveFileName(self, "Export Look (.cube)",
                                           "lumina_look.cube", "*.cube")
        if not p:
            return
        from ..core.lutio import export_cube
        export_cube(p, self.settings, dim=64)
        self.statusMessage.emit(f"Look exported: {p}")

    def _clear_lut(self):
        self.settings["lut_path"] = ""
        self.settings["lut_enabled"] = False
        self.lbl_lut.setText("")
        self._changed("Clear LUT", immediate_history=True)

    def _ai_enhance(self):
        if self._work_f32 is None:
            return
        f32 = self._work_f32
        L = imaging.luma(f32)
        mean_l = float(L.mean())

        # only fix exposure if it's actually off
        ev, _bl, _wh = imaging.compute_auto_tone(f32)
        exposure = max(-0.8, min(0.8, ev)) if abs(ev) > 0.15 else 0.0

        # WB: only correct significant color casts
        t0, ti = imaging.compute_auto_wb(f32)
        temp = round(max(-20, min(20, t0)), 1) if abs(t0) > 4 else 0.0
        tint = round(max(-12, min(12, ti)), 1) if abs(ti) > 3 else 0.0

        upd = {"exposure": exposure, "temp": temp, "tint": tint}

        # contrast only if flat
        p5, p95 = float(np.percentile(L, 5)), float(np.percentile(L, 95))
        if p95 - p5 < 0.55:
            upd["contrast"] = 12

        # recover blown highlights / crushed shadows
        hi_frac = float((L > 0.95).mean())
        lo_frac = float((L < 0.03).mean())
        if hi_frac > 0.02:
            upd["highlights"] = -25
        if lo_frac > 0.05:
            upd["shadows"] = 18

        # gentle vibrance boost (skip if already saturated)
        from lumina.core.fastpath import pack_params, build_curves  # noqa
        mx = f32.max(axis=-1); mn = f32.min(axis=-1)
        mean_sat = float(((mx-mn)/np.maximum(mx, 1e-4)).mean())
        if mean_sat < 0.15:
            upd["vibrance"] = 14
        elif mean_sat > 0.45:
            upd["saturation"] = -6

        for k, v in upd.items():
            self.settings[k] = v
        self._sync_panels()
        self._changed("AI Enhance", immediate_history=True)
        n_applied = len(upd)
        self.statusMessage.emit(f"AI Enhance: {n_applied} adjustments")

    def _ai_noiseless(self):
        self.settings["nr_lum"] = 38
        self.settings["nr_color"] = 48
        self.settings["sharp_amount"] = 22
        self._sync_panels()
        self._changed("Noiseless", immediate_history=True)
        self.statusMessage.emit("Noiseless applied")

    def _sky_toggled(self, on):
        self.settings["sky_enabled"] = bool(on)
        self._changed("Sky Replace", immediate_history=True)

    def _sky_changed(self, name):
        self.settings["sky_preset"] = name
        self._live_key("sky_preset", name)
        self._changed("Sky Preset", immediate_history=True)

    def _build_underwater_section(self):
        sec = CollapsibleSection("Underwater", False)
        info = QLabel("Restores colours absorbed by water. "
                      "Set your approximate shooting depth.")
        info.setWordWrap(True)
        info.setStyleSheet("color:#888; font-size:10px;")
        sec.add(info)

        # depth + unit toggle row
        dep_row = QHBoxLayout()
        dep_row.addWidget(QLabel("Depth"))
        self.uw_depth_slider = QSlider(Qt.Horizontal)
        self.uw_depth_slider.setRange(0, 300)   # internal: 0.1m steps
        self.uw_depth_slider.setValue(60)       # default 6m
        self.uw_depth_slider.valueChanged.connect(self._uw_depth_changed)
        dep_row.addWidget(self.uw_depth_slider, 1)
        self.b_unit = QPushButton("m")
        self.b_unit.setFixedWidth(28)
        self.b_unit.setToolTip("Toggle metres / feet")
        self.b_unit.clicked.connect(self._toggle_uw_unit)
        dep_row.addWidget(self.b_unit)
        sec.body_lay.addLayout(dep_row)
        self.lbl_depth = QLabel("")
        self.lbl_depth.setStyleSheet("color:#aaa; font-size:11px;")
        sec.add(self.lbl_depth)

        # strength
        self.s_uw_str = SliderRow("Correction", 0, 100, 0)
        self.s_uw_str.valueChanged.connect(
            lambda v: self._live_key("uw_strength", float(v)))
        self.s_uw_str.editingFinished.connect(
            lambda v: self._commit_key("uw_strength", float(v), "Underwater"))
        sec.add(self.s_uw_str)
        return sec

    def _uw_m(self) -> float:
        """Slider ticks → metres."""
        return self.uw_depth_slider.value() / 10.0

    def _uw_depth_changed(self):
        m = self._uw_m()
        if self.uw_unit_ft:
            ft = m * 3.28084
            self.lbl_depth.setText(f"{ft:.0f} ft")
        else:
            self.lbl_depth.setText(f"{m:.1f} m")
        # map to 0..100 for pipeline
        self.settings["uw_depth"] = min(100.0, (m / 30.0) * 100.0)
        self._live_key("uw_depth", self.settings["uw_depth"])

    def _toggle_uw_unit(self):
        self.uw_unit_ft = not getattr(self, 'uw_unit_ft', False)
        self.b_unit.setText("ft" if self.uw_unit_ft else "m")
        self._uw_depth_changed()

    def _uw_sync_display(self):
        if hasattr(self, 'uw_depth_slider'):
            m = self.settings.get("uw_depth", 30) / 100.0 * 30.0   # back to metres
            self.uw_depth_slider.blockSignals(True)
            self.uw_depth_slider.setValue(int(m * 10))
            self.uw_depth_slider.blockSignals(False)
            self._uw_depth_changed()

    def _build_basic_section(self):
        sec = CollapsibleSection("Basic", True)
        wb_row = QHBoxLayout()
        wb_row.addWidget(QLabel("WB"))
        self.wb_combo = QComboBox()
        self.wb_combo.addItems(list(WB_PRESETS.keys()))
        self.wb_combo.currentTextChanged.connect(self._wb_preset)
        wb_row.addWidget(self.wb_combo, 1)
        b_autowb = QPushButton("Auto WB")
        b_autowb.clicked.connect(self._auto_wb)
        wb_row.addWidget(b_autowb)
        sec.body_lay.addLayout(wb_row)

        self.s_temp = SliderRow("Temp", -100, 100, 0)
        self.s_tint = SliderRow("Tint", -100, 100, 0)
        self.sl_exposure = SliderRow("Exposure", -5, 5, 0.0, 2)
        self.s_contrast = SliderRow("Contrast", -100, 100, 0)
        self.s_highlights = SliderRow("Highlights", -100, 100, 0)
        self.s_shadows = SliderRow("Shadows", -100, 100, 0)
        self.s_whites = SliderRow("Whites", -100, 100, 0)
        self.s_blacks = SliderRow("Blacks", -100, 100, 0)
        for w_, key in ((self.s_temp, "temp"), (self.s_tint, "tint"),
                        (self.sl_exposure, "exposure"), (self.s_contrast, "contrast"),
                        (self.s_highlights, "highlights"), (self.s_shadows, "shadows"),
                        (self.s_whites, "whites"), (self.s_blacks, "blacks")):
            sec.add(w_)
            self._bind_slider(w_, key, "White Balance" if key in ("temp", "tint")
                              else ("Exposure" if key == "exposure" else "Tone"))

        ar = QHBoxLayout()
        b_autotone = QPushButton("Auto Tone")
        b_autotone.clicked.connect(self._auto_tone)
        b_zero = QPushButton("Zero Basic")
        b_zero.clicked.connect(self._zero_basic)
        ar.addWidget(b_autotone)
        ar.addWidget(b_zero)
        sec.body_lay.addLayout(ar)

        plbl = QLabel("PRESENCE")
        plbl.setObjectName("SectionTitle")
        sec.add(plbl)
        self.s_clarity = SliderRow("Clarity", -100, 100, 0)
        self.s_dehaze = SliderRow("Dehaze", -100, 100, 0)
        self.s_vibrance = SliderRow("Vibrance", -100, 100, 0)
        self.s_saturation = SliderRow("Saturation", -100, 100, 0)
        for w_, key in ((self.s_clarity, "clarity"), (self.s_dehaze, "dehaze"),
                        (self.s_vibrance, "vibrance"), (self.s_saturation, "saturation")):
            sec.add(w_)
            self._bind_slider(w_, key, "Presence")

        bw_row = QHBoxLayout()
        bw_row.addWidget(QLabel("B&W"))
        self.chk_bw = QCheckBox("Black && white")
        self.chk_bw.toggled.connect(self._set_bw)
        bw_row.addWidget(self.chk_bw)
        bw_row.addStretch(1)
        sec.body_lay.addLayout(bw_row)
        return sec

    def _build_curve_section(self):
        sec = CollapsibleSection("Tone Curve", False)
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("Channel"))
        self.curve_channel = QComboBox()
        self.curve_channel.addItems(["RGB", "R", "G", "B"])
        self.curve_channel.currentTextChanged.connect(self._curve_channel_changed)
        ch_row.addWidget(self.curve_channel, 1)
        b_reset_c = QPushButton("Reset")
        b_reset_c.clicked.connect(self._reset_curve)
        ch_row.addWidget(b_reset_c)
        sec.body_lay.addLayout(ch_row)

        self.curve_widget = ToneCurveWidget()
        self.curve_widget.curveChanged.connect(self._curve_changed)
        sec.add(self.curve_widget)
        hint = QLabel("Double-click: add point · drag out / right-click: remove")
        hint.setObjectName("PanelHint")
        sec.add(hint)
        return sec

    def _build_hsl_section(self):
        sec = CollapsibleSection("Color Mixer", False)
        band_row = QHBoxLayout()
        band_row.setSpacing(3)
        self.band_buttons = []
        for i, band in enumerate(BANDS):
            b = QPushButton(band[:3].title())
            b.setObjectName("FlatTool")
            b.setCheckable(True)
            b.setFixedWidth(32)
            b.setToolTip(band.title())
            b.clicked.connect(lambda _, ix=i: self._select_band(ix))
            band_row.addWidget(b)
            self.band_buttons.append(b)
        band_row.addStretch(1)
        sec.body_lay.addLayout(band_row)

        self.hsl_sliders = []
        for j, name in enumerate(("Hue", "Sat", "Lum")):
            s = SliderRow(name, -100, 100, 0)
            s.valueChanged.connect(lambda v, jj=j: self._hsl_live(jj, v))
            s.editingFinished.connect(
                lambda v, jj=j: self._hsl_final(jj, v))
            sec.add(s)
            self.hsl_sliders.append(s)
        self._band_idx = 0
        return sec

    def _build_grading_section(self):
        sec = CollapsibleSection("Color Grading", False)
        self.wheel_sh = ColorWheel("Shadows")
        self.wheel_mt = ColorWheel("Midtones")
        self.wheel_hi = ColorWheel("Highlights")
        row = QHBoxLayout()
        for w in (self.wheel_sh, self.wheel_mt, self.wheel_hi):
            row.addWidget(w, 1)
        sec.body_lay.addLayout(row)
        self.wheel_sh.changed.connect(lambda hu, sa: self._grade_wheel("shadows", hu, sa))
        self.wheel_mt.changed.connect(lambda hu, sa: self._grade_wheel("midtones", hu, sa))
        self.wheel_hi.changed.connect(lambda hu, sa: self._grade_wheel("highlights", hu, sa))

        self.gl_sh = SliderRow("Shd Lum", -100, 100, 0)
        self.gl_mt = SliderRow("Mid Lum", -100, 100, 0)
        self.gl_hi = SliderRow("Hi Lum", -100, 100, 0)
        for w_, key in ((self.gl_sh, "shadows"), (self.gl_mt, "midtones"),
                        (self.gl_hi, "highlights")):
            sec.add(w_)
            w_.valueChanged.connect(
                lambda v, k=key: self._grade_lum(k, v))
        self.s_blender = SliderRow("Blending", 0, 100, 50)
        self.s_balance = SliderRow("Balance", -100, 100, 0)
        sec.add(self.s_blender)
        sec.add(self.s_balance)
        self.s_blender.valueChanged.connect(lambda v: self._simple_key("grade_blender", v))
        self.s_balance.valueChanged.connect(lambda v: self._simple_key("grade_balance", v))
        return sec

    def _build_detail_section(self):
        sec = CollapsibleSection("Detail", False)
        self.s_sharp = SliderRow("Sharpening", 0, 150, 0)
        self.s_sharp_r = SliderRow("Radius", 0.5, 3.0, 1.2, 1)
        self.s_nrl = SliderRow("Noise Luminance", 0, 100, 0)
        self.s_nrc = SliderRow("Noise Color", 0, 100, 0)
        for w_, key in ((self.s_sharp, "sharp_amount"), (self.s_sharp_r, "sharp_radius"),
                        (self.s_nrl, "nr_lum"), (self.s_nrc, "nr_color")):
            sec.add(w_)
            self._bind_slider(w_, key, "Detail")
        return sec

    def _build_lens_section(self):
        sec = CollapsibleSection("Lens Corrections", False)
        self.s_dist = SliderRow("Distortion", -30, 30, 0)
        self.s_ca = SliderRow("Chrom. Aberr.", -100, 100, 0)
        for w_, key in ((self.s_dist, "lens_distortion"),
                        (self.s_ca, "ca_shift")):
            sec.add(w_)
            w_.valueChanged.connect(lambda v, k=key: self._live_key(k, float(v)))
            w_.editingFinished.connect(
                lambda v, k=key: self._commit_key(k, float(v), "Lens"))
        return sec

    def _build_effects_section(self):
        sec = CollapsibleSection("Effects", False)
        vl = QLabel("VIGNETTE")
        vl.setObjectName("SectionTitle")
        sec.add(vl)
        self.s_vig_amt = SliderRow("Amount", -100, 100, 0)
        self.s_vig_mid = SliderRow("Midpoint", 0, 100, 50)
        self.s_vig_fea = SliderRow("Feather", 0, 100, 60)
        for w_, key in ((self.s_vig_amt, "vignette_amount"),
                        (self.s_vig_mid, "vignette_midpoint"),
                        (self.s_vig_fea, "vignette_feather")):
            sec.add(w_)
            self._bind_slider(w_, key, "Vignette")
        gl = QLabel("GRAIN")
        gl.setObjectName("SectionTitle")
        sec.add(gl)
        gl2 = QLabel("GLOW")
        gl2.setObjectName("SectionTitle")
        sec.add(gl2)
        self.s_glow = SliderRow("Orton Glow", 0, 100, 0)
        sec.add(self.s_glow)
        self._bind_slider(self.s_glow, "glow_amount", "Glow")
        self.s_grain_a = SliderRow("Amount", 0, 100, 0)
        self.s_grain_s = SliderRow("Size", 0, 100, 25)
        for w_, key in ((self.s_grain_a, "grain_amount"), (self.s_grain_s, "grain_size")):
            sec.add(w_)
            self._bind_slider(w_, key, "Grain")
        return sec

    def _build_geometry_section(self):
        sec = CollapsibleSection("Crop & Rotate", True)

        # aspect ratio
        ar_row = QHBoxLayout()
        ar_row.addWidget(QLabel("Aspect"))
        self.aspect_combo = QComboBox()
        for a in ASPECTS:
            label = {"free": "Free", "orig": "Original"}.get(a, a)
            self.aspect_combo.addItem(label, a)
        self.aspect_combo.setCurrentIndex(0)
        self.aspect_combo.currentIndexChanged.connect(self._aspect_changed)
        ar_row.addWidget(self.aspect_combo, 1)
        sec.body_lay.addLayout(ar_row)

        # straighten slider
        str_row = QHBoxLayout()
        str_lbl = QLabel("Straighten")
        str_lbl.setMinimumWidth(60)
        str_row.addWidget(str_lbl)
        self.straighten_slider = SliderRow("", -45, 45, 0.0, 1)
        self.straighten_slider.label.hide()
        self.straighten_slider.valueChanged.connect(self._straighten_live)
        self.straighten_slider.editingFinished.connect(
            lambda v: self._changed("Straighten", immediate_history=True))
        str_row.addWidget(self.straighten_slider, 1)
        sec.body_lay.addLayout(str_row)

        # rotate + flip buttons
        rot_row = QHBoxLayout()
        rot_row.setSpacing(4)
        from PySide6.QtWidgets import QPushButton as _PB
        for text, handler in [("\u21ba L", lambda: self._rotate90(-1)),
                              ("\u21bb R", lambda: self._rotate90(1)),
                              ("Flip H", self._flip_h),
                              ("Flip V", self._flip_v)]:
            btn = _PB(text)
            btn.setFlat(False)
            btn.clicked.connect(handler)
            rot_row.addWidget(btn)
        sec.body_lay.addLayout(rot_row)

        # reset
        reset_r = QHBoxLayout()
        b_rc = QPushButton("Reset Crop & Straighten")
        b_rc.clicked.connect(self._reset_crop)
        reset_r.addWidget(b_rc)
        reset_r.addStretch(1)
        sec.body_lay.addLayout(reset_r)
        return sec

    def _build_transform_section(self):
        sec = CollapsibleSection("Transform", False)
        self.s_tf_v = SliderRow("Vertical", -45, 45, 0.0, 1)
        self.s_tf_h = SliderRow("Horizontal", -45, 45, 0.0, 1)
        self.s_tf_s = SliderRow("Scale", -50, 50, 0)
        for w_, key in ((self.s_tf_v, "transform_v"), (self.s_tf_h, "transform_h"),
                        (self.s_tf_s, "transform_scale")):
            sec.add(w_)
            w_.valueChanged.connect(lambda v, k=key: self._live_key(k, v))
            w_.editingFinished.connect(
                lambda v, k=key: self._commit_key(k, v, "Transform"))
        b_reset_tf = QPushButton("Reset Transform")
        b_reset_tf.clicked.connect(self._reset_transform)
        sec.add(b_reset_tf)
        return sec

    def _build_calibration_section(self):
        sec = CollapsibleSection("Calibration", False)
        self.s_cal_hue = SliderRow("Shadow Hue", 0, 360, 30)
        self.s_cal_amt = SliderRow("Shadows Tint", -100, 100, 0)
        self.s_cal_r = SliderRow("Red Primary", -100, 100, 0)
        self.s_cal_g = SliderRow("Green Primary", -100, 100, 0)
        self.s_cal_b = SliderRow("Blue Primary", -100, 100, 0)
        for w_, key in ((self.s_cal_hue, "cal_shadow_hue"),
                        (self.s_cal_amt, "cal_shadow_amt"),
                        (self.s_cal_r, "cal_r"), (self.s_cal_g, "cal_g"),
                        (self.s_cal_b, "cal_b")):
            sec.add(w_)
            if key == "cal_shadow_hue":
                w_.valueChanged.connect(lambda v: None)  # hue only matters with amt
            w_.valueChanged.connect(
                lambda v, k=key: self._live_key(k, float(v)))
            w_.editingFinished.connect(
                lambda v, k=key: self._commit_key(k, float(v), "Calibration"))
        return sec

    def _live_key(self, key: str, v):
        self.settings[key] = v
        self._live_change()

    def _commit_key(self, key: str, v, label: str):
        self.settings[key] = v
        self._changed(label, immediate_history=True)

    def _reset_transform(self):
        for k in ("transform_v", "transform_h", "transform_scale"):
            self.settings[k] = 0.0
            {"transform_v": self.s_tf_v, "transform_h": self.s_tf_h,
             "transform_scale": self.s_tf_s}[k].set_value_silent(0.0)
        self._changed("Transform Reset", immediate_history=True)

    def _build_masking_section(self):
        sec = CollapsibleSection("Masking", True)
        tools_row = QHBoxLayout()
        tools_row.setSpacing(4)

        def mb(text, tip, mode_or_ai):
            b = QPushButton(text)
            b.setObjectName("CheckableTool" if not isinstance(mode_or_ai, tuple) else "FlatTool")
            b.setToolTip(tip)
            b.clicked.connect(lambda: self._mask_tool_clicked(mode_or_ai, b))
            tools_row.addWidget(b)
            return b

        self.mb_linear = mb("Gradient", "Drag on image to draw a linear gradient mask",
                            "linear_new")
        self.mb_radial = mb("Radial", "Drag on image to draw a radial mask", "radial_new")
        self.mb_brush = mb("Brush", "Paint directly on the image", "brush")
        ai_row = QHBoxLayout()
        self.mb_subject = QPushButton("\u2726 Select Subject")
        self.mb_subject.setObjectName("Primary")
        self.mb_subject.setToolTip("AI subject detection (Apple Vision)")
        self.mb_subject.clicked.connect(lambda: self._run_ai("subject"))
        self.mb_person = QPushButton("\u270e Select Person")
        self.mb_person.setObjectName("Primary")
        self.mb_person.setToolTip("AI person segmentation (Apple Vision)")
        self.mb_person.clicked.connect(lambda: self._run_ai("person"))
        self.mb_click_subj = QPushButton("\u270e Click Subject")
        self.mb_click_subj.setObjectName("Primary")
        self.mb_click_subj.setToolTip(
            "Segment subjects, then click one on the photo to select it")
        self.mb_click_subj.clicked.connect(self._arm_subject_pick)
        ai_row.addWidget(self.mb_click_subj)
        ai_row.addWidget(self.mb_subject)
        ai_row.addWidget(self.mb_person)
        sec.body_lay.addLayout(ai_row)
        sec.body_lay.addLayout(tools_row)

        self.mask_chips_widget = QWidget()
        self.mask_chips_lay = QHBoxLayout(self.mask_chips_widget)
        self.mask_chips_lay.setContentsMargins(0, 2, 0, 2)
        self.mask_chips_lay.setSpacing(4)
        sec.body_lay.addLayout(self.mask_chips_lay) if False else None
        wrap = QVBoxLayout()
        wrap.setContentsMargins(0, 0, 0, 0)
        wrap.addWidget(self.mask_chips_widget)
        sec.body_lay.addLayout(wrap)

        opt_row = QHBoxLayout()
        self.chk_invert = QCheckBox("Invert")
        self.chk_invert.toggled.connect(self._mask_invert_toggled)
        self.chk_show_overlay = QCheckBox("Show overlay (O)")
        self.chk_show_overlay.setChecked(True)
        self.chk_show_overlay.toggled.connect(lambda _: self._update_mask_overlay())
        b_del_mask = QPushButton("Delete Mask")
        b_del_mask.clicked.connect(self._delete_active_mask)
        opt_row.addWidget(self.chk_invert)
        opt_row.addWidget(self.chk_show_overlay)
        opt_row.addWidget(b_del_mask)
        sec.body_lay.addLayout(opt_row)

        ml = QLabel("MASK ADJUSTMENTS")
        ml.setObjectName("SectionTitle")
        sec.add(ml)
        self.mask_adj_sliders = {}
        for key in imaging.MASK_ADJ_KEYS:
            lab = key.replace("_", " ").title() if key != "temp" else "Temp"
            if key == "tint":
                lab = "Tint"
            s = SliderRow(lab, -5 if key == "exposure" else -100,
                          5 if key == "exposure" else 100, 0,
                          2 if key == "exposure" else 0)
            s.valueChanged.connect(
                lambda v, k=key: self._mask_adj_live(k, v))
            s.label.setFixedWidth(84)
            sec.add(s)
            self.mask_adj_sliders[key] = s
        self.mask_hint = QLabel("Create a mask above, then adjust it here.")
        self.mask_hint.setObjectName("PanelHint")
        sec.add(self.mask_hint)

        sl = QLabel("SPOT REMOVAL")
        sl.setObjectName("SectionTitle")
        sec.add(sl)
        srow = QHBoxLayout()
        srow.addWidget(QLabel("Mode"))
        self.spot_mode_combo = QComboBox()
        self.spot_mode_combo.addItems(["Heal", "Clone", "Red-Eye"])
        self.spot_mode_combo.currentTextChanged.connect(self._spot_mode_changed)
        srow.addWidget(self.spot_mode_combo, 1)
        sec.body_lay.addLayout(srow)
        self.spot_size = SliderRow("Size", 1, 12, 3)
        self.spot_size.valueChanged.connect(self._spot_size_changed)
        sec.add(self.spot_size)
        b_clear_spots = QPushButton("Clear All Spots")
        b_clear_spots.clicked.connect(self._clear_spots)
        sec.add(b_clear_spots)
        spot_hint = QLabel("Press Q or click Heal, then click blemishes.")
        spot_hint.setObjectName("PanelHint")
        sec.add(spot_hint)
        return sec

    # ================================================================ rendering
    def load_photo(self, photo_id: int):
        if self.photo_id == photo_id:
            return
        self._persist_now()
        row = catalog.get_photo(photo_id)
        if row is None:
            return
        self._gen += 1                      # cancel in-flight renders
        self.photo_id = photo_id
        self.photo_path = row["path"]
        saved = catalog.load_settings(photo_id)
        self.settings = imaging.sanitize_settings(saved) if saved else imaging.default_settings()
        self._history = []
        self._hist_index = -1
        self._active_mask = -1
        self.uw_unit_ft = False
        self._full_f32 = None
        try:
            prev = rawio.decode_preview(self.photo_path, 1200)
            self._oriented_u8 = imaging.apply_geometry_flip_rot(
                prev, self.settings.get("rotate90", 0),
                self.settings.get("flip_h", False), self.settings.get("flip_v", False))
            self._heal_cache = (None, None)
            self._rebuild_buffers()
        except Exception as e:
            print("[develop] decode failed:", e)
            self.statusMessage.emit(f"Could not decode {row['filename']}")
            return
        self.canvas.set_zoom_fit()
        self.canvas.spots = [dict(s) for s in (self.settings.get("spots") or [])]
        self._sync_panels()
        self._start_render(0)
        self._quality_timer.start()
        self._recompute_subject_masks_async()
        self._push_history(initial=True)
        self.statusMessage.emit(row["filename"])

    # ---------------- healing (cached)
    def _healed_u8(self):
        """Oriented image with spot removal applied; cached by spots hash."""
        shash = healmod.spots_hash(self.settings.get("spots") or [])
        if self._heal_cache[0] == shash and self._heal_cache[1] is not None:
            return self._heal_cache[1]
        if not (self.settings.get("spots") or []):
            out = self._oriented_u8
        else:
            out = healmod.apply_spots(self._oriented_u8,
                                      self.settings.get("spots") or [])
        self._heal_cache = (shash, out)
        return out

    def _rebuild_buffers(self, keep_full=False):
        u8 = self._healed_u8()
        if u8 is None:
            return
        self._work_u8 = u8
        self._work_f32 = u8.astype(np.float32) / 255.0
        if not keep_full:
            self._full_f32 = None
            self._drag_f32 = None

    def _load_full_and_render(self):
        if not self.photo_path:
            return
        if self._full_f32 is None:
            try:
                prev = rawio.decode_preview(self.photo_path, 2400)
                u8 = imaging.apply_geometry_flip_rot(
                    prev, self.settings.get("rotate90", 0),
                    self.settings.get("flip_h", False),
                    self.settings.get("flip_v", False))
                self._full_f32 = u8.astype(np.float32) / 255.0
            except Exception as e:
                print("[develop] full decode failed:", e)
                return
        self._start_render(1)

    def _live_change(self):
        """During slider drags: ultra-fast low-res tier."""
        self._drag_timer.start()
        self._save_timer.start()
        self._last_label = self._last_label or "Edit"
        self._hist_timer2.start()

    def _changed(self, label: str, immediate_history: bool = False):
        """Called whenever settings change from the UI (commit point)."""
        self._fast_timer.start()
        self._quality_timer.start()
        self._save_timer.start()
        self._last_label = label
        if immediate_history:
            self._push_history()
        else:
            self._hist_timer.start()

    def _simple_key(self, key: str, v):
        self.settings[key] = float(v)
        self._changed("Color Grading")

    def _start_render(self, tier: int):
        if self.photo_path is None or (tier == 0 and self._work_f32 is None):
            return
        if tier == -1:
            if self._drag_f32 is None:
                try:
                    prev = rawio.decode_preview(self.photo_path, 640)
                    u8 = imaging.apply_geometry_flip_rot(
                        prev, self.settings.get("rotate90", 0),
                        self.settings.get("flip_h", False),
                        self.settings.get("flip_v", False))
                    self._drag_raw_u8 = u8
                except Exception:
                    self._drag_f32 = self._work_f32
                    tier = 0
            if getattr(self, "_drag_raw_u8", None) is not None:
                healed = healmod.apply_spots(self._drag_raw_u8,
                                             self.settings.get("spots")
                                             or []) \
                    if self.settings.get("spots") else self._drag_raw_u8
                self._drag_f32 = healed.astype(np.float32) / 255.0
            img = self._drag_f32 if self._drag_f32 is not None else self._work_f32
        else:
            img = self._work_f32 if tier == 0 else (
                self._full_f32 if self._full_f32 is not None else self._work_f32)
        self._gen += 1
        gen = self._gen
        seed = os.path.basename(self.photo_path or "x")
        task = _RenderTask(self._bridge, gen, max(tier, 0),
                           img, copy.deepcopy(self.settings), seed)
        self._pool.start(task, max(tier, 0) + 2)

    def _on_rendered(self, gen: int, tier: int, u8):
        if gen != self._gen or u8 is None:
            return
        if tier == 1:
            self._result_quality = u8
        self.canvas.set_image(u8)
        self.canvas.set_crop(self.settings.get("crop"))
        self._update_histogram(u8)
        self._update_mask_overlay(shape=u8.shape)

    def _update_histogram(self, u8: np.ndarray):
        small = u8[:: max(1, u8.shape[0] // 300), :: max(1, u8.shape[1] // 300)]
        self.histogram.update_image(small)
        luma = imaging.luma(small.astype(np.float32) / 255.0)
        hist, _ = np.histogram(luma, bins=64, range=(0, 1))
        peak = hist.max() or 1
        self.curve_widget.set_histogram((hist / peak).astype(np.float32))

    # ================================================================ panels sync
    def _sync_panels(self):
        self._suppress_panel_sync = True
        s = self.settings
        m = {
            self.s_temp: s["temp"], self.s_tint: s["tint"],
            self.sl_exposure: s["exposure"], self.s_contrast: s["contrast"],
            self.s_highlights: s["highlights"], self.s_shadows: s["shadows"],
            self.s_whites: s["whites"], self.s_blacks: s["blacks"],
            self.s_clarity: s["clarity"], self.s_dehaze: s["dehaze"],
            self.s_vibrance: s["vibrance"], self.s_saturation: s["saturation"],
            self.s_sharp: s["sharp_amount"], self.s_sharp_r: s["sharp_radius"],
            self.s_nrl: s["nr_lum"], self.s_nrc: s["nr_color"],
            self.s_vig_amt: s["vignette_amount"], self.s_vig_mid: s["vignette_midpoint"],
            self.s_vig_fea: s["vignette_feather"], self.s_grain_a: s["grain_amount"],
            self.s_grain_s: s["grain_size"], self.s_blender: s["grade_blender"],
            self.s_balance: s["grade_balance"],
        }
        for w_, v in m.items():
            w_.set_value_silent(v)
        self.chk_bw.blockSignals(True); self.chk_bw.setChecked(s.get("bw", False))
        self.chk_bw.blockSignals(False)
        for i, band in enumerate(BANDS):
            b = self.band_buttons[i]
            hs = s["hsl"][band]
            active = any(abs(x) > 0.01 for x in hs)
            b.setStyleSheet("color:%s;" % ("#eaf2fa" if active else theme.TEXT_DIM))
        self._select_band(self._band_idx if hasattr(self, "_band_idx") else 0)
        ch = self.curve_channel.currentText().lower().replace("rgb", "rgb")
        keymap = {"RGB": "curve_rgb", "R": "curve_r", "G": "curve_g", "B": "curve_b"}
        self.curve_widget.set_points(s.get(keymap[self.curve_channel.currentText()]) or [])
        for wname, gkey in ((self.wheel_sh, "grade_shadows"),
                            (self.wheel_mt, "grade_midtones"),
                            (self.wheel_hi, "grade_highlights")):
            g = s[gkey]
            wname.set_hs(g[0], g[1])
        self.gl_sh.set_value_silent(s["grade_shadows"][2])
        self.gl_mt.set_value_silent(s["grade_midtones"][2])
        self.gl_hi.set_value_silent(s["grade_highlights"][2])
        idx = ASPECTS.index(s.get("crop_aspect") or "free")
        self.aspect_combo.blockSignals(True)
        self.aspect_combo.setCurrentIndex(idx)
        self.aspect_combo.blockSignals(False)
        self.straighten_slider.set_value_silent(s.get("straighten", 0.0))
        for w_, k in ((getattr(self, 's_uw_depth', None), s.get("uw_depth", 30)),
                      (getattr(self, 's_uw_str', None), s.get("uw_strength", 0))):
            if w_ is not None:
                w_.set_value_silent(float(k))

        if hasattr(self, 'uw_depth_slider'):
            m = s.get("uw_depth", 30) / 100.0 * 30.0
            self.uw_depth_slider.blockSignals(True)
            self.uw_depth_slider.setValue(int(m * 10))
            self.uw_depth_slider.blockSignals(False)
            self._uw_depth_changed()
        if hasattr(self, 'chk_sky'):
            self.chk_sky.blockSignals(True)
            self.chk_sky.setChecked(bool(s.get("sky_enabled")))
            self.chk_sky.blockSignals(False)
        if hasattr(self, 'sky_preset_combo'):
            i = self.sky_preset_combo.findText(s.get("sky_preset", "Golden Sunset"))
            self.sky_preset_combo.blockSignals(True)
            self.sky_preset_combo.setCurrentIndex(max(0, i))
            self.sky_preset_combo.blockSignals(False)
        for w_, k in ((getattr(self, 's_dist', None), s.get("lens_distortion", 0)),
                      (getattr(self, 's_ca', None), s.get("ca_shift", 0)),
                      (getattr(self, 's_glow', None), s.get("glow_amount", 0)),
                      (getattr(self, 's_rel_dir', None), s.get("relight_angle", 300)),
                      (getattr(self, 's_rel_str', None), s.get("relight_strength", 0)),
                      (getattr(self, 's_sky_str', None), s.get("sky_strength", 75)),
                      (getattr(self, 's_sky_soft', None), s.get("sky_softness", 45)),
                      (getattr(self, 's_sky_off', None), s.get("sky_offset", 0))):
            if w_ is not None:
                w_.set_value_silent(float(k))
        lp = s.get("lut_path") or ""
        if hasattr(self, 'lbl_lut'):
            self.lbl_lut.setText(os.path.basename(lp) if lp else "")
        for w_, k in ((self.s_sky_str, "sky_strength"),
                      (self.s_sky_soft, "sky_softness"),
                      (self.s_sky_off, "sky_offset"),
                      (self.s_rel_dir, "relight_angle"),
                      (self.s_rel_str, "relight_strength")):
            w_.set_value_silent(float(s.get(k, 0)))
        for w_, k in ((self.s_tf_v, s.get("transform_v", 0.0)),
                      (self.s_tf_h, s.get("transform_h", 0.0)),
                      (self.s_tf_s, s.get("transform_scale", 0.0)),
                      (self.s_cal_hue, s.get("cal_shadow_hue", 30.0)),
                      (self.s_cal_amt, s.get("cal_shadow_amt", 0.0)),
                      (self.s_cal_r, s.get("cal_r", 0.0)),
                      (self.s_cal_g, s.get("cal_g", 0.0)),
                      (self.s_cal_b, s.get("cal_b", 0.0))):
            w_.set_value_silent(float(k))
        self.canvas.set_crop(s.get("crop"))
        self._rebuild_mask_chips()
        self.chk_invert.blockSignals(True)
        am = self._active_mask_def()
        self.chk_invert.setChecked(bool(am.get("invert")) if am else False)
        self.chk_invert.blockSignals(False)
        self._sync_mask_adj_sliders()
        self._suppress_panel_sync = False

    def _set_bw(self, on: bool):
        self.settings["bw"] = bool(on)
        self._changed("B&W", immediate_history=True)

    # ================================================================ curves
    def _curve_key(self) -> str:
        t = self.curve_channel.currentText()
        return {"RGB": "curve_rgb", "R": "curve_r", "G": "curve_g", "B": "curve_b"}[t]

    def _curve_changed(self, pts):
        self.settings[self._curve_key()] = [list(p) for p in pts]
        self._changed("Tone Curve")

    def _reset_curve(self):
        self.settings[self._curve_key()] = []
        self.curve_widget.set_points([])
        self._changed("Tone Curve", immediate_history=True)

    def _curve_channel_changed(self):
        pts = self.settings.get(self._curve_key()) or []
        self.curve_widget.set_points(pts)

    # ================================================================ HSL
    def _select_band(self, idx: int):
        self._band_idx = idx
        for i, b in enumerate(self.band_buttons):
            b.setChecked(i == idx)
        vals = self.settings["hsl"][BANDS[idx]]
        for j, s in enumerate(self.hsl_sliders):
            s.set_value_silent(vals[j])

    def _hsl_live(self, j: int, v: float):
        self.settings["hsl"][BANDS[self._band_idx]][j] = v
        self._changed("Color Mixer")

    def _hsl_final(self, j: int, v: float):
        self.settings["hsl"][BANDS[self._band_idx]][j] = v
        self._changed("Color Mixer", immediate_history=True)

    # ================================================================ grading
    def _grade_wheel(self, which: str, hue: float, sat: float):
        key = f"grade_{which}"
        g = self.settings[key]
        g[0], g[1] = round(float(hue), 1), round(float(sat), 1)
        self._changed("Color Grading")

    def _grade_lum(self, which: str, v: float):
        self.settings[f"grade_{which}"][2] = float(v)
        self._changed("Color Grading")

    # ================================================================ geometry
    def _rotate90(self, k: int):
        self.settings["rotate90"] = self.settings.get("rotate90", 0) + k
        self._reload_oriented()
        self._changed("Rotate", immediate_history=True)

    def _flip_h(self):
        self.settings["flip_h"] = not self.settings.get("flip_h", False)
        self._reload_oriented()
        self._changed("Flip", immediate_history=True)

    def _flip_v(self):
        self.settings["flip_v"] = not self.settings.get("flip_v", False)
        self._reload_oriented()
        self._changed("Flip", immediate_history=True)

    def _reload_oriented(self):
        rot = self.settings.get("rotate90", 0)
        fh = self.settings.get("flip_h", False)
        fv = self.settings.get("flip_v", False)
        if self.photo_path:
            prev = rawio.decode_preview(self.photo_path, 1200)
            self._oriented_u8 = imaging.apply_geometry_flip_rot(prev, rot, fh, fv)
            self._heal_cache = (None, None)
            self._rebuild_buffers()
            self._drag_f32 = None
        self.canvas.set_zoom_fit()

    # ---------------- spot removal handlers
    def toggle_spot_mode(self):
        on = getattr(self.b_heal, "isChecked", lambda: False)()
        if on:
            self.canvas.mode = "spot"
            for b in (self.mb_linear, self.mb_radial, self.mb_brush):
                b.setChecked(False)
            self.canvas.spots = [dict(s)
                                 for s in (self.settings.get("spots") or [])]
            self.statusMessage.emit(
                "Click blemishes to heal · Alt/right-click removes a spot")
        else:
            self.canvas.mode = "view"

    def _spots_from_canvas(self):
        self.settings["spots"] = [dict(s)
                                  for s in getattr(self.canvas, "spots", [])]
        self._heal_cache = (None, None)
        self._rebuild_buffers()
        self._drag_f32 = None
        self._changed("Spot Removal", immediate_history=True)

    def toggle_split(self):
        self.canvas.toggle_split()
        self.statusMessage.emit(
            "Split view " + ("on — drag divider" if self.canvas.split_mode else "off"))

    def _spot_mode_changed(self, mode: str):
        self.canvas.spot_mode = mode.lower()

    def _spot_size_changed(self, v: float):
        self.canvas.spot_radius_norm = v / 100.0

    def _clear_spots(self):
        self.settings["spots"] = []
        self.canvas.spots = []
        self._heal_cache = (None, None)
        self._rebuild_buffers()
        self._drag_f32 = None
        self._fast_timer.start()
        self._quality_timer.start()
        self._changed("Clear Spots", immediate_history=True)

    def _aspect_changed(self):
        a = self.aspect_combo.currentData()
        self.settings["crop_aspect"] = a
        self.canvas.aspect = a
        c = self.settings.get("crop")
        rw, rh = self.canvas._img_size
        if c and a != "free" and rw:
            ratio = rw / rh if a == "orig" else \
                float(a.split(":")[0]) / float(a.split(":")[1])
            cx, cy = (c[0]+c[2])/2, (c[1]+c[3])/2
            w = min(c[2]-c[0], (c[3]-c[1])*ratio)
            h_ = w / ratio
            nc = [cx-w/2, cy-h_/2, cx+w/2, cy+h_/2]
            DevelopCanvas._clamp_crop_full(nc)
            self.settings["crop"] = nc
            self.canvas.set_crop(nc)
        self._changed("Crop Aspect", immediate_history=True)

    def _straighten_live(self, v: float):
        self.settings["straighten"] = v
        self._changed("Straighten")

    def toggle_crop(self):
        on = self.b_crop.isChecked()
        if on:
            self.canvas.mode = "crop"
            if self.settings.get("crop") is None:
                self.settings["crop"] = [0.02, 0.02, 0.98, 0.98]
                self.settings["crop_aspect"] = \
                    self.aspect_combo.currentData() or "free"
            self.canvas.set_crop(self.settings["crop"])
            self.canvas.set_zoom_fit()
        else:
            self.canvas.mode = "view"
            self.canvas.set_zoom_fit()
            c = self.settings.get("crop")
            if c and (c[2]-c[0] > 0.995 and c[3]-c[1] > 0.995):
                self.settings["crop"] = None      # full-frame crop = no crop
            self._persist_soon()

    def _crop_live(self, rect):
        self.settings["crop"] = list(rect)
        self._save_timer.start()

    def _crop_commit(self):
        c = self.settings.get("crop")
        if c and (c[2]-c[0] < 0.03 or c[3]-c[1] < 0.03):
            return
        self._changed("Crop", immediate_history=True)

    def _reset_crop(self):
        self.settings["crop"] = None
        self.settings["straighten"] = 0.0
        self.straighten_slider.set_value_silent(0.0)
        self.canvas.set_crop(None)
        self._changed("Crop Reset", immediate_history=True)

    # ================================================================ masking
    def _mask_tool_clicked(self, mode, btn=None):
        if isinstance(mode, tuple):
            mode = mode[0]
        for b in (self.mb_linear, self.mb_radial, self.mb_brush):
            if b is not btn:
                b.setChecked(False)
        if mode == "linear_new":
            self.canvas.mode = "linear_new" if btn.isChecked() else "view"
            if self.canvas.mode != "view":
                self.statusMessage.emit("Drag across the image to draw the gradient")
        elif mode == "radial_new":
            self.canvas.mode = "radial_new" if btn.isChecked() else "view"
            if self.canvas.mode != "view":
                self.statusMessage.emit("Drag to size the radial mask")
        elif mode == "brush":
            self.canvas.mode = "brush" if btn.isChecked() else "view"
            if self.canvas.mode == "brush":
                if self._active_mask_def() is None or \
                        self._active_mask_def()["type"] != "brush":
                    self._add_mask("brush", {"strokes": []}, select=True)
                self.statusMessage.emit("Paint on the image (Alt-drag erases)")
        else:
            self.canvas.mode = "view"

    def _active_mask_def(self):
        ms = self.settings.get("masks", [])
        if 0 <= self._active_mask < len(ms):
            return ms[self._active_mask]
        return None

    def _add_mask(self, type_: str, params: dict, select=True) -> dict:
        import uuid
        names = {"linear": "Linear Gradient", "radial": "Radial Gradient",
                 "brush": "Brush", "subject": "Subject", "person": "Person"}
        md = {
            "id": uuid.uuid4().hex[:8],
            "name": f"{names.get(type_, 'Mask')} {len(self.settings['masks'])+1}",
            "type": type_,
            "params": params,
            "invert": False,
            "adjustments": copy.deepcopy(imaging.DEFAULT_MASK_ADJUSTMENTS),
        }
        self.settings.setdefault("masks", []).append(md)
        if select:
            self._active_mask = len(self.settings["masks"]) - 1
        self._rebuild_mask_chips()
        self._sync_mask_adj_sliders()
        return md

    def _mask_drawn(self, type_: str, params: dict):
        self._add_mask(type_, params)
        self.canvas.mode = "view"
        for b in (self.mb_linear, self.mb_radial, self.mb_brush):
            b.setChecked(False)
        self._changed("Add Mask", immediate_history=True)
        self.statusMessage.emit("Mask created — adjust it below")

    def _brush_stroke(self, stroke):
        md = self._active_mask_def()
        if md is None or md["type"] != "brush":
            return
        from PySide6.QtWidgets import QApplication
        erase = bool(QApplication.keyboardModifiers() & Qt.AltModifier)
        pts, radius, flow = stroke
        md["params"].setdefault("strokes", []).append((pts, radius, 0.0 if erase else flow))
        self._changed("Brush Mask")
        self._update_mask_overlay()

    def _run_ai(self, mode: str):
        if self._work_u8 is None:
            return
        if not aimask.vision_available():
            self.statusMessage.emit("AI masking unavailable (requires macOS 14+)")
            return
        tag = (self.photo_id, mode)
        self.statusMessage.emit("Analyzing photo with Apple Vision…")
        self.mb_subject.setEnabled(False)
        self.mb_person.setEnabled(False)
        self._pool.start(_AiTask(self._ai_bridge, self._work_u8.copy(), mode, tag), 9)

    def _on_ai_done(self, mask, meta):
        self.mb_subject.setEnabled(True)
        self.mb_person.setEnabled(True)
        mode = meta.get("mode", "subject")
        if mask == "__nosubject__":
            self.statusMessage.emit(
                "No subject detected — try another photo or draw a manual mask")
            return
        if mask == "__error__":
            self.statusMessage.emit(f"AI mask error: {meta.get('error','?')}")
            return
        existing = next((m for m in self.settings.get("masks", [])
                         if m["type"] == mode), None)
        if existing:
            existing["params"]["_subject_array"] = mask
            self._active_mask = self.settings["masks"].index(existing)
        else:
            self._add_mask(mode, {"_subject_array": mask})
        self._rebuild_mask_chips()
        self._sync_mask_adj_sliders()
        self._changed(f"AI {mode.title()} Mask", immediate_history=True)
        self.statusMessage.emit("AI mask ready — adjust exposure/color inside it")

    def _recompute_subject_masks_async(self):
        """Re-derive AI arrays after photo load (arrays are runtime-only)."""
        for i, m in enumerate(self.settings.get("masks", [])):
            if m["type"] in ("subject", "person") and \
                    "_subject_array" not in m["params"]:
                self._pool.start(_AiTask(self._ai_bridge, self._work_u8.copy(),
                                         m["type"], (self.photo_id, m["type"])), 9)

    def _arm_subject_pick(self):
        self.canvas.mode = "subject_pick"
        for b in (self.mb_linear, self.mb_radial, self.mb_brush):
            b.setChecked(False)
        self.statusMessage.emit(
            "Click directly on a subject to select it · Esc cancels")

    def _cancel_subject_pick(self):
        if getattr(self.canvas, "mode", "") == "subject_pick":
            self.canvas.mode = "view"

    def _subject_picked(self, nx: float, ny: float):
        u8 = self._oriented_u8
        if u8 is None:
            return
        self.statusMessage.emit("Selecting clicked subject…")

        def work():
            try:
                from ..core import aimask
                mask = aimask.compute_subject_at_point(u8, nx, ny)
            except aimask.NoSubjectFound:
                mask = "__empty__"
            except Exception as e:
                print("[pick]", e)
                mask = None
            QTimer.singleShot(0, lambda: self._subject_pick_done(mask))

        import threading
        threading.Thread(target=work, daemon=True).start()

    def _subject_pick_done(self, mask):
        if mask == "__empty__":
            self.statusMessage.emit("Nothing under that click — try again")
            return
        if mask is None:
            self.statusMessage.emit("Subject selection failed")
            return
        import uuid
        md = {"id": uuid.uuid4().hex[:8],
              "name": f"Clicked Subject {len(self.settings['masks'])+1}",
              "type": "subject", "invert": False,
              "params": {"_subject_array": mask},
              "adjustments": dict(imaging.DEFAULT_MASK_ADJUSTMENTS)}
        self.settings.setdefault("masks", []).append(md)
        self._active_mask = len(self.settings["masks"]) - 1
        self.canvas.mode = "view"
        self.rebuildMaskChipsSafe()
        self._changed("Click Subject", immediate_history=True)
        self.statusMessage.emit("Subject selected — adjust below")

    def rebuildMaskChipsSafe(self):
        try:
            self._rebuild_mask_chips()
            self._sync_mask_adj_sliders()
        except Exception as e:
            print("[chips]", e)

    def _delete_active_mask(self):
        ms = self.settings.get("masks", [])
        if 0 <= self._active_mask < len(ms):
            del ms[self._active_mask]
            self._active_mask = -1
        self._rebuild_mask_chips()
        self._sync_mask_adj_sliders()
        self._update_mask_overlay()
        self._changed("Delete Mask", immediate_history=True)

    def _mask_invert_toggled(self, on):
        md = self._active_mask_def()
        if md:
            md["invert"] = bool(on)
            self._changed("Mask Invert")

    def _select_mask_chip(self, idx):
        self._active_mask = idx
        self._rebuild_mask_chips()
        self._sync_mask_adj_sliders()
        self._update_mask_overlay()

    def _rebuild_mask_chips(self):
        lay = self.mask_chips_lay
        while lay.count():
            it = lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        ms = self.settings.get("masks", [])
        icons = {"linear": "\u25f9", "radial": "\u25cb", "brush": "\u271b",
                 "subject": "\u2726", "person": "\u270e"}
        for i, m in enumerate(ms):
            chip = QPushButton(f"{icons.get(m['type'],'')} {m['name']}")
            chip.setObjectName("MaskChip")
            chip.setCheckable(True)
            chip.setChecked(i == self._active_mask)
            chip.setCursor(Qt.PointingHandCursor)
            chip.clicked.connect(lambda _, ix=i: self._select_mask_chip(ix))
            lay.addWidget(chip)
        lay.addStretch(1)
        has = len(ms) > 0
        self.mask_chips_widget.setVisible(has)
        for s in self.mask_adj_sliders.values():
            s.setVisible(has)
        self.mask_hint.setVisible(not has)

    def _sync_mask_adj_sliders(self):
        md = self._active_mask_def()
        for key, s in self.mask_adj_sliders.items():
            if md:
                s.setEnabled(True)
                s.set_value_silent(md["adjustments"].get(key, 0.0))
            else:
                s.setEnabled(False)
                s.set_value_silent(0.0)

    def _mask_adj_live(self, key: str, v: float):
        md = self._active_mask_def()
        if md:
            md["adjustments"][key] = v
            self._changed("Mask Adjustment")

    def _update_mask_overlay(self, shape=None):
        if shape is None:
            shape = getattr(self.canvas, "_img_size", (0, 0))
            if not shape[0]:
                self.canvas.set_mask_overlay(None)
                return
            shape = (shape[1], shape[0])
        md = self._active_mask_def()
        if md is None or not self.chk_show_overlay.isChecked() or \
                self.canvas.mode == "view":
            self.canvas.set_mask_overlay(None)
            return
        arr = imaging.rasterize_mask(tuple(shape), md)
        if md.get("invert"):
            arr = 1.0 - arr
        h, w = arr.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 230
        rgba[..., 1] = 70
        rgba[..., 2] = 60
        rgba[..., 3] = (np.clip(arr, 0, 1) * 110).astype(np.uint8)
        img = QImage(rgba.data, w, h, w * 4, QImage.Format_RGBA8888)
        self.canvas.set_mask_overlay(QPixmap.fromImage(img.copy()))

    # ================================================================ WB / auto
    def _wb_preset(self, name):
        spec = WB_PRESETS.get(name)
        if spec is None:
            return
        if spec == "auto":
            self._auto_wb()
            return
        self.settings["temp"], self.settings["tint"] = float(spec[0]), float(spec[1])
        self.s_temp.set_value_silent(spec[0])
        self.s_tint.set_value_silent(spec[1])
        self._changed("White Balance", immediate_history=True)

    def _auto_wb(self):
        if self._work_f32 is None:
            return
        t, ti = imaging.compute_auto_wb(self._work_f32)
        self.settings["temp"], self.settings["tint"] = round(t, 1), round(ti, 1)
        self.s_temp.set_value_silent(t)
        self.s_tint.set_value_silent(ti)
        self._changed("Auto WB", immediate_history=True)
        self.statusMessage.emit("Auto white balance applied")

    def _auto_tone(self):
        if self._work_f32 is None:
            return
        ev, blacks, whites = imaging.compute_auto_tone(self._work_f32)
        self.sl_exposure.set_value_silent(ev)
        self.s_blacks.set_value_silent(blacks)
        self.s_whites.set_value_silent(whites)
        self.settings["exposure"], self.settings["blacks"] = ev, blacks
        self.settings["whites"] = whites
        self._changed("Auto Tone", immediate_history=True)
        self.statusMessage.emit("Auto tone applied")

    def _zero_basic(self):
        d = imaging.default_settings()
        keys = ["temp", "tint", "exposure", "contrast", "highlights", "shadows",
                "whites", "blacks", "clarity", "dehaze", "vibrance", "saturation"]
        for k in keys:
            self.settings[k] = 0.0
        self.settings["bw"] = False
        self._suppress_panel_sync = True
        for k in keys:
            pass
        self._sync_panels()
        self._suppress_panel_sync = False
        self._changed("Zero Basic", immediate_history=True)

    # ================================================================ history
    def _push_history(self, initial=False):
        snap = copy.deepcopy(self.settings)
        label = self._last_label or ("Import" if initial else "Edit")
        if initial:
            label = "Original"
        # drop redo tail
        self._history = self._history[:self._hist_index + 1]
        if self._history and not initial:
            prev = self._history[-1]["settings"]
            if json.dumps(settings_for_save(prev), sort_keys=True, default=str) == \
                    json.dumps(settings_for_save(snap), sort_keys=True, default=str):
                return
        self._history.append({"label": label,
                              "time": __import__("datetime").datetime.now(),
                              "settings": snap})
        if len(self._history) > 60:
            self._history.pop(0)
        self._hist_index = len(self._history) - 1
        self._refresh_history_list()

    def _refresh_history_list(self):
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for i, h in enumerate(self._history):
            it = QListWidgetItem(f"{h['label']}  ·  {h['time']:%H:%M:%S}")
            it.setData(Qt.UserRole, i)
            self.history_list.addItem(it)
        if 0 <= self._hist_index < self.history_list.count():
            self.history_list.setCurrentRow(self._hist_index)
        self.history_list.blockSignals(False)

    def _goto_history(self, item):
        idx = item.data(Qt.UserRole)
        self._apply_snapshot(idx)

    def _apply_snapshot(self, idx: int):
        if not (0 <= idx < len(self._history)):
            return
        self._hist_index = idx
        self.settings = copy.deepcopy(self._history[idx]["settings"])

        # restore canvas state to match snapshot
        self.canvas.spots = [dict(s) for s in (self.settings.get("spots") or [])]
        self._heal_cache = (None, None)
        self._rebuild_buffers()
        self._drag_f32 = None

        # re-derive AI mask arrays that are still referenced
        for m in self.settings.get("masks", []):
            if m["type"] in ("subject", "person") and                     "_subject_array" not in m.get("params", {}):
                pass  # will be recomputed below

        # restore canvas state from snapshot
        self.canvas.spots = [dict(s)
                             for s in (self.settings.get("spots") or [])]

        self._recompute_subject_masks_async()
        self._sync_panels()

        self._start_render(0)
        self._quality_timer.start()
        self._save_timer.start()

    def undo(self):
        if self._hist_index > 0:
            self.statusMessage.emit(f"Undo: {self._history[self._hist_index]['label']}")
            self._apply_snapshot(self._hist_index - 1)

    def redo(self):
        if self._hist_index < len(self._history) - 1:
            self._apply_snapshot(self._hist_index + 1)
            self.statusMessage.emit(f"Redo: {self._history[self._hist_index]['label']}")

    # ================================================================ presets
    def rebuild_preset_list(self):
        self.preset_list.clear()
        for name in PresetStore.all_presets():
            it = QListWidgetItem(name)
            self.preset_list.addItem(it)

    def _apply_preset_item(self, item):
        name = item.text()
        preset = PresetStore.all_presets().get(name)
        if not preset:
            return
        # reset pointwise settings to defaults so presets don't stack
        fresh = imaging.default_settings()
        for k in ("temp","tint","exposure","contrast","highlights","shadows",
                  "whites","blacks","clarity","dehaze","vibrance","saturation",
                  "bw","curve_rgb","curve_r","curve_g","curve_b",
                  "grade_shadows","grade_midtones","grade_highlights",
                  "grade_blender","grade_balance","sharp_amount","sharp_radius",
                  "nr_lum","nr_color","vignette_amount","vignette_midpoint",
                  "vignette_feather","grain_amount","grain_size","glow_amount"):
            self.settings[k] = fresh[k]
        for k, v in preset.items():
            if k == "hsl":
                for band, vals in v.items():
                    if band in BANDS:
                        self.settings["hsl"][band] = list(vals)
            else:
                self.settings[k] = copy.deepcopy(v)
        self._sync_panels()
        self._changed(f"Preset: {name}", immediate_history=True)
        self.statusMessage.emit(f"Preset applied: {name}")

    def _save_preset_dialog(self):
        from PySide6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Save Preset",
                                        "Preset name (current look):")
        if not ok or not name.strip():
            return
        partial = {k: copy.deepcopy(v) for k, v in self.settings.items()
                   if k not in ("rotate90", "flip_h", "flip_v", "straighten",
                                "crop", "crop_aspect", "masks")}
        PresetStore.save_user(name.strip(), partial)
        self.rebuild_preset_list()
        self.statusMessage.emit(f"Preset saved: {name}")

    def _delete_preset(self):
        it = self.preset_list.currentItem()
        if it is None:
            return
        name = it.text()
        if name in BUILTIN_PRESETS:
            self.statusMessage.emit("Built-in presets cannot be deleted")
            return
        PresetStore.delete(name)
        self.rebuild_preset_list()

    # ================================================================ copy/paste
    def copy_settings(self):
        self._clipboard = {k: copy.deepcopy(v) for k, v in self.settings.items()
                           if k not in ("rotate90", "flip_h", "flip_v",
                                        "straighten", "crop", "crop_aspect")}
        self.statusMessage.emit("Settings copied")

    def paste_settings(self):
        if self._clipboard is None:
            self.statusMessage.emit("Nothing copied yet")
            return
        for k, v in self._clipboard.items():
            self.settings[k] = copy.deepcopy(v)
        self._recompute_subject_masks_async()
        self._sync_panels()
        self._changed("Paste Settings", immediate_history=True)

    # ================================================================ nav/persist
    def navigate(self, step: int):
        pid = self.next_photo_id(step)
        if pid is not None:
            self.load_photo(pid)

    def next_photo_id(self, step: int):
        rows = catalog.query()
        ids = [r["id"] for r in rows]
        if not ids or self.photo_id is None:
            return ids[0] if ids else None
        try:
            i = ids.index(self.photo_id)
        except ValueError:
            return ids[0]
        j = min(len(ids)-1, max(0, i+step))
        return ids[j]

    def _persist_soon(self):
        self._save_timer.start()

    def _persist(self):
        if self.photo_id is None:
            return
        catalog.save_settings(self.photo_id, settings_for_save(self.settings))
        self.photoEdited.emit(self.photo_id)

    def _persist_now(self):
        if self.photo_id is not None and self._save_timer.isActive():
            self._save_timer.stop()
            self._persist()

    def reset_all(self):
        keep = {"rotate90": self.settings.get("rotate90", 0)}
        self.settings = imaging.default_settings()
        self.settings.update(keep)
        self._active_mask = -1
        self.uw_unit_ft = False
        self._sync_panels()
        self._changed("Reset All", immediate_history=True)
        self.statusMessage.emit("All edits reset")
