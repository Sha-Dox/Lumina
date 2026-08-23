"""Lumina brand assets — programmatic logo, icons, About dialog."""
from __future__ import annotations

import os
import subprocess

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QLinearGradient, QPainter, QPainterPath,
                           QPen, QPixmap, QRadialGradient)

BRAND_DIR = os.path.expanduser("~/.lumina/brand")


def draw_logo(size: int = 1024, flat: bool = False) -> QPixmap:
    """Aperture-iris mark: dark tile, gradient ring, six blades, bright core."""
    pm = QPixmap(size, size)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    s = size

    # tile
    if not flat:
        grad = QRadialGradient(s*0.38, s*0.32, s*1.05)
        grad.setColorAt(0.0, QColor("#2c3440"))
        grad.setColorAt(1.0, QColor("#12161c"))
        p.setBrush(grad)
        p.setPen(Qt.NoPen)
        p.drawRoundedRect(QRectF(s*0.04, s*0.04, s*0.92, s*0.92), s*0.22, s*0.22)

    # outer gradient ring
    ring = QRectF(s*0.14, s*0.14, s*0.72, s*0.72)
    g2 = QLinearGradient(ring.topLeft(), ring.bottomRight())
    g2.setColorAt(0.00, QColor("#57b3ff"))
    g2.setColorAt(0.55, QColor("#7a6bff"))
    g2.setColorAt(1.00, QColor("#ff9d6b"))
    pen_w = s * 0.055
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(g2, pen_w))
    p.drawEllipse(ring.adjusted(pen_w/2, pen_w/2, -pen_w/2, -pen_w/2))

    # six aperture blades (rotated chords)
    cx, cy = s/2, s/2
    r_out = s * 0.315
    blade_col = QColor(230, 238, 248, 215)
    p.setPen(Qt.NoPen)
    p.setBrush(blade_col)
    for i in range(6):
        ang = i * 60.0
        p.save()
        p.translate(cx, cy)
        p.rotate(ang)
        path = QPainterPath()
        path.moveTo(r_out*0.10, -r_out*0.98)
        path.arcTo(QRectF(-r_out, -r_out, r_out*2, r_out*2),
                   -90 + 14, 44)
        path.lineTo(r_out*0.16, r_out*0.42)
        path.closeSubpath()
        p.drawPath(path)
        p.restore()

    # bright core
    core_r = s * 0.115
    cg = QRadialGradient(cx - core_r*0.25, cy - core_r*0.3, core_r*1.7)
    cg.setColorAt(0.0, QColor("#ffffff"))
    cg.setColorAt(0.55, QColor("#bcd9f7"))
    cg.setColorAt(1.0, QColor("#6ea8dd"))
    p.setBrush(cg)
    p.drawEllipse(QPointF(cx, cy), core_r, core_r)
    p.end()
    return pm


def app_icon() -> QIcon:
    os.makedirs(BRAND_DIR, exist_ok=True)
    png = os.path.join(BRAND_DIR, "logo512.png")
    if not os.path.exists(png):
        draw_logo(512).save(png, "PNG")
    return QIcon(png)


def write_iconset() -> str | None:
    """Write macOS .icns; returns path or None."""
    try:
        os.makedirs(BRAND_DIR, exist_ok=True)
        src = os.path.join(BRAND_DIR, "icon.iconset")
        subprocess.run(["rm", "-rf", src], check=False)
        os.makedirs(src, exist_ok=True)
        sizes = [(16, "icon_16x16"), (32, "icon_16x16@2x"), (32, "icon_32x32"),
                 (64, "icon_32x32@2x"), (128, "icon_128x128"),
                 (256, "icon_128x128@2x"), (256, "icon_256x256"),
                 (512, "icon_256x256@2x"), (512, "icon_512x512"),
                 (1024, "icon_512x512@2x")]
        base = draw_logo(1024)
        for px, name in sizes:
            base.scaled(px, px, Qt.IgnoreAspectRatio,
                        Qt.SmoothTransformation).save(
                os.path.join(src, f"{name}.png"), "PNG")
        icns = os.path.join(BRAND_DIR, "lumina.icns")
        subprocess.run(["iconutil", "-c", "icns", src, "-o", icns],
                       check=True, capture_output=True)
        return icns
    except Exception as e:
        print("[brand] icns failed:", e)
        return None


def build_app_bundle() -> str | None:
    """Create ~/Applications/Lumina.app wrapping the launcher."""
    home = os.path.expanduser("~")
    root = os.path.join(home, "Applications", "Lumina.app")
    macos = os.path.join(root, "Contents", "MacOS")
    res = os.path.join(root, "Contents", "Resources")
    try:
        os.makedirs(macos, exist_ok=True)
        os.makedirs(res, exist_ok=True)
        icns = write_iconset() or ""
        if icns:
            subprocess.run(["cp", icns, os.path.join(res, "lumina.icns")],
                           check=True, capture_output=True)
        import sys as _sys, subprocess as _sp
        proj = os.path.expanduser("~/Lumina")
        interp = _sys.executable          # absolute path to OUR python

        # Compile a real arm64 launcher binary — script-based executables get
        # architecture ambiguity in LaunchServices (stale Rosetta registrations
        # caused hard crashes with arm64 numpy).
        launcher_c = os.path.join(res, "launcher.c")
        with open(launcher_c, "w") as f:
            f.write(f"""#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc, char **argv) {{
    (void)argc; (void)argv;
    if (chdir("{proj}") != 0) return 2;
    char py[1024];
    snprintf(py, sizeof(py), "{interp}");
    char *args[] = {{ "python3", "-u", "main.py", NULL }};
    execv(py, args);
    return 1;
}}
""")
        binpath = os.path.join(macos, "Lumina")
        cc = _sp.run(["clang", "-arch", "arm64", "-o", binpath, launcher_c],
                     capture_output=True)
        if cc.returncode != 0:
            print("[brand] clang failed:", cc.stderr.decode(errors="replace")[:400])
            # Fallback: universal script (still better than stale Rosetta)
            with open(binpath, "w") as f:
                f.write(f"""#!/bin/bash
cd "{proj}"
LOG="$HOME/.lumina/launch.log"
echo "=== $(date) fallback-script launch ===" >> "$LOG"
exec /usr/bin/arch -arm64 "{interp}" -u main.py "$@" >> "$LOG" 2>&1
""")
            os.chmod(binpath, 0o755)

        plist = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
 <key>CFBundleName</key><string>Lumina</string>
 <key>CFBundleDisplayName</key><string>Lumina</string>
 <key>CFBundleIdentifier</key><string>app.lumina.photo.v2</string>
 <key>CFBundleVersion</key><string>1.0</string>
 <key>CFBundleShortVersionString</key><string>1.0</string>
 <key>CFBundleExecutable</key><string>Lumina</string>
 <key>CFBundleIconFile</key><string>lumina</string>
 <key>CFBundlePackageType</key><string>APPL</string>
 <key>NSHighResolutionCapable</key><true/>
 <key>NSSupportsAutomaticGraphicsSwitching</key><true/>
</dict></plist>"""
        with open(os.path.join(root, "Contents", "Info.plist"), "w") as f:
            f.write(plist)
        return root
    except Exception as e:
        print("[brand] bundle failed:", e)
        return None


class AboutDialog(__import__("PySide6.QtWidgets", fromlist=["QDialog"]).QDialog):
    def __init__(self, parent=None):
        from PySide6.QtWidgets import QDialogButtonBox, QLabel, QVBoxLayout
        super().__init__(parent)
        self.setWindowTitle("About Lumina")
        self.setStyleSheet("QDialog{background:#252525;}")
        v = QVBoxLayout(self)
        v.setSpacing(10)
        icon_lbl = QLabel()
        icon_lbl.setPixmap(draw_logo(148))
        icon_lbl.setAlignment(Qt.AlignCenter)
        v.addWidget(icon_lbl)
        t = QLabel("<div align='center'><b style='font-size:22px'>Lumina</b><br>"
                   "<span style='color:#969696'>RAW photo editor · v1.0</span>"
                   "<br><span style='color:#777'>LibRaw · Apple Vision · OpenCV · Qt</span></div>")
        t.setTextFormat(Qt.RichText)
        v.addWidget(t)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(self.reject)
        bb.clicked.connect(lambda *_: self.accept())
        v.addWidget(bb)


if __name__ == "__main__":
    import sys
    from PySide6.QtWidgets import QApplication
    app = QApplication(sys.argv)
    pm = draw_logo(512)
    pm.save("/tmp/lumina_logo.png")
    print("logo drawn:", os.path.getsize("/tmp/lumina_logo.png"), "bytes")
    print("icns:", write_iconset())
    print("bundle:", build_app_bundle())
