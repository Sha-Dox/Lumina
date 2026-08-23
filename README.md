# Lumina

A Lightroom-class RAW photo editor and library manager for macOS, built with
Python + PySide6 + LibRaw + Apple Vision.

![stack](https://img.shields.io/badge/RAW-LibRaw%200.22-blue) — CR2 · CR3 · ARW · RAF · NEF · NRW · ORF · RW2 · DNG · PEF · SRW + JPEG/PNG/TIFF/WebP

## Launch

```bash
./lumina.command          # double-click in Finder, or:
python3 main.py
```

First run installs nothing else; dependencies are already pip-installed (see
`requirements.txt`).

## What it does

### Library module (`G`)
- Import folders of RAW/JPEG photos into a fast SQLite catalog
- Folder tree, thumbnail grid with async decoding + cached thumbs
- Star ratings `0–5`, pick/reject flags `P` / `X` / `U`, color labels
- Filter by rating / flag / color label, search across filename·camera·lens
- EXIF metadata panel (camera, lens, exposure triangle, date)
- Quick Develop for instant edits without leaving the grid
- Filmstrip shared with Develop, `←`/`→` to navigate

### Develop module (`D`)
**Basic** — Temp/Tint (+ WB presets & Auto WB), Exposure, Contrast,
Highlights/Shadows/Whites/Blacks, Clarity, Dehaze, Vibrance, Saturation, B&W,
Auto Tone.

**Tone Curve** — interactive point curve (RGB composite + per-channel R/G/B),
monotonic spline like LR, histogram backdrop. Double-click adds points,
right-click removes.

**Color Mixer (HSL)** — 8 bands × Hue/Sat/Lum with smooth band falloffs;
B&W conversion follows the mixer's luminance values.

**Color Grading** — three-way wheels (Shadows/Midtones/Highlights) with hue/sat
pickers + luminance sliders, blending & balance.

**Detail** — Sharpening (amount/radius), luminance & chroma noise reduction.

**Effects** — vignette (amount/midpoint/feather), film grain (amount/size).

**Geometry** — rotate 90°, flip H/V, straighten ±45°, crop tool (`R`) with
aspect presets (Free/Original/1:1/4:3/3:2/16:9/9:16), rule-of-thirds overlay.

**Masking** — local adjustments with their own full adjustment stack:
- Linear gradient — drag on image
- Radial gradient — drag to size
- Brush — paint, `Alt`-drag to erase
- ✦ Select Subject — AI subject detection via Apple Vision
- ✎ Select Person — AI person segmentation via Apple Vision
- Invert any mask, red overlay preview (`O`), stack unlimited masks

**Workflow** — presets (8 built-in + save your own), full history panel with
undo/redo (`⌘Z` / `⇧⌘Z`), copy/paste settings between photos (`⌘C`/`⌘V`),
before/after hold (`\`), zoom fit/100% (`F` / scroll).

### Export (`⌘E`)
Full-resolution render through the identical pipeline: JPEG (quality) /
PNG / TIFF, optional long-edge resize with output sharpening.

## Speed
The entire pointwise adjustment chain (WB → exposure → contrast → tone
regions → curves → vibrance/saturation → HSL mixer → B&W → grading) is fused
into a single **numba-parallel kernel** — verified pixel-identical to the
legacy numpy path (meanΔ < 1/255 across every control). Spatial filters use
OpenCV. Progressive tiers: drags render at 640px in **~14 ms typical**
(33 ms with every slider active), release at 1200px (~54 ms), idle sharpens
to ~2000px (~125 ms). Exports: **12MP DNG in ~1.1 s**. The JIT compiles once
at launch in a background thread so first drags are instant.

## Branding
Programmatic aperture-iris logo (see `lumina/ui/brand.py`) rendered to app
icon, `.icns`, and a native **`~/Applications/Lumina.app`** bundle — launch
from Finder/Dock like any Mac app (re-run `python3 -c "from lumina.ui.brand
import build_app_bundle; build_app_bundle()"` from ~/Lumina to refresh).

## Modules (all)
Library · Develop · **Map** · **Print** · **Book** — plus fullscreen
**Slideshow** (▶ button: crossfades, Ken Burns, 2–12 s holds) and one-click
**Web gallery** export (self-contained HTML + lightbox, edited renders,
ratings/keyword captions).

## AI Tools (Luminar-style)
- **✨ AI Enhance** — one-click: analyzes exposure/WB/tone/color stats and
  applies a balanced edit (capped, history-tracked).
- **Sky Replacement** — heuristic horizon detection + five procedural skies
  (Golden Sunset, Dramatic Storm, Clear Blue, Twilight Stars, Pastel Dream)
  with strength / edge softness / horizon shift; light-wrap tinting.
- **Relight AI** — directional relight: angle + strength.
- **Noiseless** — one-click noise reduction preset.
- **Click Subject** — segment the scene, then click any subject on the photo
  to select exactly that object as a mask (component-matched, soft-feathered).

## AI Tools
- **✨ AI Enhance** — one-click smart edit: WB, exposure, tone curve,
  vibrance/clarity tuned from image statistics (capped, history-tracked).
- **Sky Replacement** — heuristic horizon detection; five procedural skies
  (Golden Sunset / Dramatic Storm / Clear Blue / Twilight Stars / Pastel
  Dream); strength, edge softness, horizon shift, light-wrap.
- **Relight AI** — directional light: angle 0–360° ± strength.
- **Noiseless** — one-click noise-reduction preset.
- **Click Subject** — Vision segments the scene; click any object on the
  canvas to select exactly that subject as a soft mask.

## New modules
- **Spot Removal** (`Q`, Heal toolbar button) — click blemishes; AI-free
  Telea inpainting, clone-stamp mode with source handle, Alt/right-click erase.
- **HDR Merge** (Library → HDR) — Mertens exposure fusion of 2+ selected
  brackets with ECC alignment; result auto-imported.
- **Panorama** (Library → Pano) — OpenCV feature stitching; auto-imported.
- **Map** tab — OpenStreetMap slippy map with pins for geotagged photos,
  click-to-open, "Place selection at map center" writes GPS into the catalog
  plus an XMP sidecar next to the RAW.
- **Print** tab — page templates (full/2-up/4-up/2×3/3×3), A4–4×6 sizes,
  portrait/landscape, margins/gutters, multi-page, system printer + PDF export.
- **Tether…** — native Apple ImageCaptureCore camera detection & shutter
  release, plus a watched-folder importer for manufacturer apps.
- **Transform panel** — vertical/horizontal keystone + scale (perspective).
- **Calibration panel** — shadow tint (hue/amount) + RGB primary gains.
- **Red-eye** — third Spot Removal mode; recolors red pupils automatically.
- **Watermarking** — text watermark with opacity in the export dialog.
- **Collections** — group photos into named sets from the Library left panel;
  click a collection to filter the grid.
- **Keywords** — comma-separated tags per photo, searchable from the toolbar.

## Architecture notes
- All processing is numpy-vectorized float32 with LUT-accelerated curves.
- Edits persist per-photo inside the catalog (`~/.lumina/catalog.db`) and are
  re-applied automatically on reopen. Thumbnails cache under `~/.lumina/cache`.
- `gpu.py` contains an experimental OpenGL pipeline (disabled pending driver
  hardening); the app does not depend on it.

## Keyboard map
| Keys | Action |
|---|---|
| `G` / `D` | Library / Develop module |
| `R` | Crop mode |
| `0–5` | Rating |
| `P` / `X` / `U` | Pick / Reject / Unflag |
| `\` (hold) | Before / after |
| `O` | Mask overlay toggle |
| `⌘C` / `⌘V` | Copy / paste develop settings |
| `⌘Z` / `⇧⌘Z` | Undo / redo |
| `⌘I` / `⌘E` | Import / Export |
| `←` / `→` | Prev / next photo |
| Scroll / `+` / `-` | Zoom |

## Roadmap (not yet implemented)
Spot healing, tethered capture, HDR merge, panorama stitching, print module,
map module.
