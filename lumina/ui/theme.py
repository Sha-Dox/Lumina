"""Lumina dark theme — Lightroom-inspired charcoal UI."""
from __future__ import annotations

BG          = "#1e1e1e"   # app background
BG_PANEL    = "#252525"   # side panels
BG_SUNKEN   = "#191919"   # canvas area / filmstrip well
BG_CONTROL  = "#2f2f2f"   # inputs, buttons
BG_HOVER    = "#3a3a3a"
BORDER      = "#101010"
BORDER_SOFT = "#383838"
TEXT        = "#cfcfcf"
TEXT_DIM    = "#969696"
TEXT_FAINT  = "#6b6b6b"
ACCENT      = "#4f8fcc"
ACCENT_HOVER= "#69a7de"
ACCENT_DIM  = "#2d5578"
SELECT_BG   = "#394b5c"
STAR        = "#d8d8d8"
RED         = "#cc4c4c"

QSS = f"""
* {{
    outline: none;
}}
QWidget {{
    background: {BG_PANEL};
    color: {TEXT};
    font-family: 'Helvetica Neue', Arial, sans-serif;
    font-size: 12px;
}}
QMainWindow, #RootWindow {{ background: {BG}; }}

/* ---------- top bar ---------- */
#TopBar {{
    background: {BG};
    border-bottom: 1px solid {BORDER};
}}
#BrandLabel {{
    background: transparent;
    color: {TEXT};
    font-size: 15px;
    font-weight: 600;
    letter-spacing: 1px;
}}
#ModuleTabs {{ background: transparent; }}
QPushButton#ModuleTab {{
    background: transparent;
    color: {TEXT_FAINT};
    border: none;
    padding: 6px 18px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton#ModuleTab:hover {{ color: {TEXT_DIM}; }}
QPushButton#ModuleTab:checked {{
    color: {TEXT};
    border-bottom: 2px solid {ACCENT};
}}

/* ---------- panels & scroll areas ---------- */
#SidePanel {{
    background: {BG_PANEL};
    border-right: 1px solid {BORDER};
}}
#RightPanel {{ border-left: 1px solid {BORDER}; }}
QScrollArea {{
    background: {BG_PANEL};
    border: none;
}}
QScrollArea > QWidget > QWidget {{ background: {BG_PANEL}; }}

QScrollBar:vertical {{
    background: transparent; width: 10px; margin: 2px;
}}
QScrollBar::handle:vertical {{
    background: #454545; min-height: 24px; border-radius: 4px;
}}
QScrollBar::handle:vertical:hover {{ background: #555; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar:horizontal {{
    background: transparent; height: 10px; margin: 2px;
}}
QScrollBar::handle:horizontal {{
    background: #454545; min-width: 24px; border-radius: 4px;
}}
QScrollBar::handle:horizontal:hover {{ background: #555; }}

/* ---------- section headers ---------- */
SectionHeader {{
    background: transparent;
    border: none;
    border-bottom: 1px solid #303030;
    padding: 7px 4px 7px 2px;
    text-align: left;
}}
SectionHeader:hover {{ background: #2a2a2a; }}
QLabel#SectionTitle {{
    background: transparent;
    color: {TEXT_DIM};
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.4px;
}}

/* ---------- sliders ---------- */
QSlider::groove:horizontal {{
    border: none;
    height: 3px;
    background: qlineargradient(x1:0 y1:0, x2:1 y2:0,
        stop:0 #484848, stop:1 #484848);
    border-radius: 1px;
}}
SliderRow QSlider::groove:horizontal {{
    background: {BG_CONTROL};
}}
QSlider::sub-page:horizontal {{
    background: {ACCENT_DIM};
    border-radius: 1px;
}}
QSlider::handle:horizontal {{
    background: #e8e8e8;
    width: 13px; height: 13px;
    margin: -5px 0;
    border-radius: 6px;
    border: 1px solid #111;
}}
QSlider::handle:horizontal:hover {{ background: #ffffff; }}

/* ---------- inputs ---------- */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 3px 6px;
    selection-background-color: {ACCENT_DIM};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border: 1px solid {ACCENT};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER};
    selection-background-color: {SELECT_BG};
    color: {TEXT};
}}
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    width: 0; height: 0;
}}

/* ---------- buttons ---------- */
QPushButton {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 5px 14px;
    color: {TEXT};
}}
QPushButton:hover {{ background: {BG_HOVER}; }}
QPushButton:pressed {{ background: #444; }}
QPushButton:disabled {{ color: {TEXT_FAINT}; background: #282828; }}
QPushButton#Primary {{
    background: {ACCENT_DIM};
    border: 1px solid #1d3a54;
    color: #eaf2fa;
    font-weight: 600;
}}
QPushButton#Primary:hover {{ background: #35608a; }}
QPushButton#FlatTool {{
    background: transparent;
    border: none;
    padding: 4px 8px;
    color: {TEXT_DIM};
}}
QPushButton#FlatTool:hover {{ color: {TEXT}; background: {BG_HOVER}; border-radius: 4px; }}
QPushButton#FlatTool:checked {{ color: #fff; background: {ACCENT_DIM}; border-radius: 4px; }}
QPushButton#CheckableTool:checked {{
    background: {ACCENT_DIM};
    color: #fff;
    border: 1px solid #35608a;
}}

/* ---------- lists / trees ---------- */
QTreeWidget, QListWidget {{
    background: {BG_PANEL};
    border: none;
    alternate-background-color: {BG_PANEL};
}}
QTreeWidget::item, QListWidget::item {{
    color: {TEXT_DIM};
    padding: 3px 2px;
    border-radius: 3px;
}}
QTreeWidget::item:hover, QListWidget::item:hover {{
    background: {BG_HOVER};
    color: {TEXT};
}}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {SELECT_BG};
    color: #fff;
}}
QListWidget#GridList {{
    background: {BG_SUNKEN};
}}
QListWidget#GridList::item:selected {{ border: 2px solid {ACCENT}; background: #222; }}
QListWidget#Filmstrip {{ background: {BG_SUNKEN}; border-top: 1px solid {BORDER}; }}

/* ---------- misc ---------- */
QToolTip {{
    background: #111;
    color: {TEXT};
    border: 1px solid {BORDER_SOFT};
    padding: 4px 8px;
    font-size: 11px;
}}
QMenu {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER_SOFT};
    padding: 4px;
}}
QMenu::item {{ padding: 4px 22px; border-radius: 3px; }}
QMenu::item:selected {{ background: {SELECT_BG}; }}
QProgressBar {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 4px;
    text-align: center;
    color: {TEXT_DIM};
    height: 14px;
}}
QProgressBar::chunk {{ background: {ACCENT_DIM}; border-radius: 3px; }}
QSplitter::handle {{ background: {BORDER}; width: 1px; height: 1px; }}
QStatusBar {{
    background: {BG};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
    font-size: 11px;
}}
#CanvasFrame {{ background: {BG_SUNKEN}; }}
#HistogramBox {{ background: {BG_PANEL}; border: 1px solid #303030; border-radius: 4px; }}
#CurveBox {{ background: #232323; border: 1px solid #343434; border-radius: 3px; }}
#PanelHint {{ color: {TEXT_FAINT}; background: transparent; font-size: 11px; }}
#MaskChip {{
    background: {BG_CONTROL};
    border: 1px solid {BORDER};
    border-radius: 10px;
    padding: 2px 10px;
}}
#MaskChip:checked {{
    background: {ACCENT_DIM};
    border-color: #35608a;
    color: #fff;
}}
"""
