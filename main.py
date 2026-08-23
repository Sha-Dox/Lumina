#!/usr/bin/env python3
"""Lumina — a Lightroom-class RAW photo editor. Entry point."""
import os
import sys
import traceback

LOG = os.path.expanduser("~/.lumina/launch.log")


def _log(msg: str):
    try:
        os.makedirs(os.path.dirname(LOG), exist_ok=True)
        with open(LOG, "a") as f:
            f.write(msg + "\n")
    except Exception:
        pass


def main():
    # Route all output into the launch log even when launched by Finder
    # (no shell redirection involved).
    try:
        log = open(LOG, "a", buffering=1)
        sys.stdout = log
        sys.stderr = log
        import datetime as _dt
        print(f"=== launch {_dt.datetime.now():%Y-%m-%d %H:%M:%S} ===")
    except Exception:
        pass
    try:
        _run()
    except Exception:
        tb = traceback.format_exc()
        _log("=== CRASH ===\n" + tb)
        print(tb, file=sys.stderr)
        raise


def _run():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv)
    app.setApplicationName("Lumina")
    app.setStyle("Fusion")

    from lumina.ui.theme import QSS
    app.setStyleSheet(QSS)

    # --- real Dock icon on macOS (setWindowIcon alone shows the Python rocket)
    def _apply_dock_icon():
        try:
            import os
            from PySide6.QtGui import QIcon
            png = os.path.expanduser("~/.lumina/brand/logo512.png")
            if not os.path.exists(png):
                from lumina.ui.brand import draw_logo
                d = os.path.dirname(png)
                os.makedirs(d, exist_ok=True)
                draw_logo(512).save(png, "PNG")
            ic = QIcon(png)
            app.setWindowIcon(ic)
            from AppKit import NSApplication, NSImage
            from Foundation import NSURL
            nsimg = NSImage.alloc().initWithContentsOfURL_(
                NSURL.fileURLWithPath_(png))
            if nsimg:
                NSApplication.sharedApplication().setApplicationIconImage_(nsimg)
        except Exception as e:
            print("[brand] dock icon:", e)

    _apply_dock_icon()

    from lumina.ui.app import LuminaWindow
    win = LuminaWindow()

    # macOS: full-height dark titlebar look
    try:
        from PySide6.QtCore import QOperatingSystemVersion
        if QOperatingSystemVersion.currentType() == QOperatingSystemVersion.MacOS:
            win.setWindowFlag(Qt.WindowFullscreenButtonHint, True)
    except Exception:
        pass

    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
