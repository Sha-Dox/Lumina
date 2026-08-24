<div align="center">

<img src="logo.png" width="128" height="128" alt="Lumina">

# Lumina

**A free, open-source RAW photo editor for macOS**

Built with Python · PySide6 · LibRaw · Apple Vision · OpenCV

[![macOS 13+](https://img.shields.io/badge/macOS-13%2B-black?logo=apple&logoColor=white)](https://www.apple.com/macos)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Features](#features) · [Quick Start](#quick-start) · [Modules](#modules) · [Keyboard Shortcuts](#keyboard-shortcuts) · [Performance](#performance)

</div>

---

## Features

### Develop
Full non-destructive RAW editing with progressive-resolution rendering
(14 ms drag / 54 ms release / 125 ms quality at 2016px).

| Panel | Controls |
|-------|----------|
| **Basic** | Temp · Tint · Exposure · Contrast · Highlights · Shadows · Whites · Blacks · Clarity · Dehaze · Vibrance · Saturation · B&W |
| **Tone Curve** | Interactive point editor (RGB + per-channel R/G/B) with histogram backdrop and monotonic spline interpolation |
| **HSL Color Mixer** | 8 bands × Hue / Sat / Lum |
| **Color Grading** | Shadows / Midtones / Highlights wheels + Blender + Balance |
| **Detail** | Sharpening (amount/radius) · Luminance & Chroma NR |
| **Lens Corrections** | Distortion (barrel/pincushion) · Chromatic Aberration removal |
| **Effects** | Vignette (amount/midpoint/feather) · Film Grain (amount/size) · Orton Glow |
| **Geometry** | Crop with aspect presets · Straighten ±45° · Rotate 90° · Flip H/V |
| **Transform** | Vertical/horizontal keystone · Scale |
| **Calibration** | Shadow tint · RGB primary gains |

### AI Tools
- **✨ AI Enhance** — one-click conservative enhancement (analyses exposure/WB/tone/saturation)
- **Sky Replacement** — heuristic horizon detection + 5 procedural skies with strength/softness/offset controls
- **Relight AI** — directional light: angle 0–360°, strength ±100
- **Noiseless** — one-click noise reduction preset
- **Click Subject** — segment the scene with Vision, then click any object to select it as a mask

### Masking
Linear gradient · Radial gradient · Brush paint (Alt = erase) · AI Subject · Select Person · Invert · per-mask adjustment stack

### Spot Removal
Heal (inpainting) · Clone stamp with source handle · Red-eye correction

### Underwater Mode
Depth slider in metres or feet. Restores red/green colours absorbed by water using adaptive channel rebalancing calibrated against inverse Beer-Lambert absorption.

### Professional Scopes
Click to cycle: **RGB Histogram** → **Luma Histogram** → **Waveform** → **Vectorscope**

### Library
Folder tree · Thumbnail grid with async decoding · Star ratings (0–5) · Pick/reject flags · Colour labels · Keywords with search · Collections · Metadata panel (EXIF) · Quick Develop · Auto-Cull (sharpness/exposure/people scoring) · Duplicate finder · Face detection · XMP sidecar write

### Other Modules
| Module | Description |
|--------|-------------|
| **Print** | Page templates (full/2-up/4-up/2×3/3×3), page sizes, PDF export, system printer |
| **Book** | Themed photo books (Classic/Magazine/Minimal) with title page → PDF |
| **Slideshow** | Fullscreen playback with crossfades, Ken Burns effect, 2–12s holds |

### Workflow
Presets (13 built-in film looks + save your own) · **Lightroom .xmp preset import/export** · **Versions** (virtual copies) · History panel with undo/redo · Copy/paste settings between photos · **Sync settings** across selected photos · Before/After hold + Split view · Zoom fit/100%/scroll-wheel · Non-destructive (originals never modified) · XMP sidecar writing

### Export
JPEG (quality control) · PNG · TIFF · Long-edge resize · Text watermark with opacity · Batch export of all queued photos

### Import / Merge
HDR Merge (Mertens exposure fusion + ECC alignment) · Panorama stitching (OpenCV) · Tethered capture via Apple ImageCaptureCore + watched-folder importer · Folder import with recursive scan

---

## Quick Start

```bash
git clone https://github.com/Sha-Dox/Lumina.git
cd Lumina
pip3 install -r requirements.txt
python3 main.py
```

Or build the native macOS app bundle:

```bash
chmod +x lumina.command
./lumina.command        # launches from Terminal
open ~/Applications/Lumina.app  # if you built the .app bundle
```

### Requirements

```
PySide6 ≥ 6.6          # Qt GUI framework
numpy ≥ 1.26           # array processing
scipy ≥ 1.11           # gaussian filtering
Pillow ≥ 10.0          # image I/O
rawpy ≥ 0.18           # LibRaw RAW decoder (CR2/CR3/ARW/RAF/NEF/DNG…)
opencv-python-headless ≥ 4.8  # inpaint, stitcher, merge, remap
exifread ≥ 3.0         # EXIF metadata parsing
pyobjc-framework-Vision ≥ 9.0   # AI subject/person segmentation
pyobjc-framework-Quartz ≥ 9.0   # Core Graphics helpers
pyobjc-framework-ImageCaptureCore ≥ 9.0  # tethered capture
pyobjc-framework-CoreMedia ≥ 9.0  # pixel buffer access
```

---

## Supported Formats

| Type | Extensions |
|------|-----------|
| **RAW** | CR2 · CR3 · ARW · RAF · NEF · NRW · ORF · RW2 · DNG · PEF · SRW |
| **Raster** | JPEG · PNG · TIFF · BMP · WebP |

*RAW decoding powered by LibRaw 0.22 via [rawpy](https://github.com/letmaik/rawpy).*

---

## Performance

Progressive-resolution rendering keeps the UI responsive:

| Tier | Resolution | Latency (typical edit) |
|------|-----------|----------------------|
| Drag | 640px | **~14 ms** (70 fps) |
| Release | 1200px | ~54 ms |
| Quality | 2016px | ~125 ms |
| Export | Full sensor | 12MP DNG in ~1.1 s |

The entire pointwise pipeline is fused into a single numba-parallel kernel,
verified pixel-identical to the reference implementation.
Exposure follows the industry-standard 2^EV linear-light model.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `G` / `D` | Library / Develop module |
| `R` | Crop mode |
| `Q` | Spot removal mode |
| `0` – `5` | Star rating |
| `P` / `X` / `U` | Pick / Reject / Unflag |
| `\` (hold) | Before / after |
| `O` | Overlay toggle |
| `F` | Fit to window |
| `+` / `-` | Zoom in/out |
| `⌘C` / `⌘V` | Copy / paste develop settings |
| `⌘Z` / `⇧⌘Z` | Undo / redo |
| `⌘I` / `⌘E` | Import / Export |
| `←` / `→` | Previous / next photo |

---

## Architecture

```
lumina/
├── core/
│   ├── imaging.py      # numpy pipeline (pointwise + spatial ops)
│   ├── fastpath.py     # numba-fused kernel (14ms drag tier)
│   ├── rawio.py        # ImageIO/LibRaw decode + thumb cache
│   ├── catalog.py      # SQLite catalog (thread-safe)
│   ├── aimask.py       # Vision framework AI masks
│   ├── sky.py          # sky replacement engine
│   ├── cull.py         # auto-cull scoring
│   ├── heal.py         # spot heal/clone/red-eye
│   ├── merge.py        # HDR + panorama
│   ├── lutio.py        # .cube import/export + trilinear
│   ├── export.py       # full-res render + watermark
│   ├── xmp.py          # Adobe-compatible sidecars
│   └── gpu.py          # experimental OpenGL renderer
├── ui/
│   ├── app.py          # main window + module switching
│   ├── develop.py      # develop panels + orchestration
│   ├── library.py      # library grid + filters
│   ├── develop_canvas.py # canvas with crop/mask/spot tools
│   ├── scopes.py       # RGB/luma/waveform/vectorscope
│   └── …               # print, book, slideshow, map, tether
└── main.py             # entry point
```

All edits are stored as JSON in a SQLite catalog (`~/.lumina/catalog.db`).
Original files are never modified — verified byte-for-byte.

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing`)
3. Run the test suite (`QT_QPA_PLATFORM=offscreen python3 /tmp/lumina_full_smoke.py`)
4. Commit your changes
5. Push and open a Pull Request

## License

MIT — see [LICENSE](LICENSE)
