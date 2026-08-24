"""Lumina command-line interface — programmatic photo editing for AI agents and scripts.

Usage:
    python3 cli.py edit   <photo> [options]     Apply adjustments
    python3 cli.py enhance <photo> [--output f]  AI one-click enhancement
    python3 cli.py sky    <photo> --preset NAME  Sky replacement
    python3 cli.py underwater <photo> --depth N  Underwater colour restore
    python3 cli.py batch  <dir> [options]        Batch process directory
    python3 cli.py export <photo> --format FMT   Export in specific format
    python3 cli.py info   <photo>                Print metadata as JSON
    python3 cli.py cull   <dir> [--threshold N]  Score + auto-rate photos
    python3 cli.py hdr    <photos...> -o out.tif HDR merge
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from lumina.core import imaging, rawio


def _load(path: str, max_edge: int = 0) -> np.ndarray:
    from .core.rawio import decode_preview, decode_full, is_raw
    if max_edge > 0:
        arr = decode_preview(path, max_edge)
    else:
        arr = decode_full(path)
    return arr.astype(np.float32) / 255.0


def _save(arr_u8, path, quality=92):
    from PIL import Image
    ext = os.path.splitext(path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        Image.fromarray(arr_u8).save(path, "JPEG", quality=quality)
    elif ext == ".png":
        Image.fromarray(arr_u8).save(path, "PNG")
    elif ext == ".tif" or ext == ".tiff":
        Image.fromarray(arr_u8).save(path, "TIFF", compression="tiff_lzw")
    else:
        Image.fromarray(arr_u8).save(path, "JPEG", quality=quality)


def _build_settings(args) -> dict:
    """Build a settings dict from CLI arguments."""
    s = {
        # tone
        "temp": args.temp or 0.0,
        "tint": args.tint or 0.0,
        "exposure": args.exposure or 0.0,
        "contrast": args.contrast or 0.0,
        "highlights": args.highlights if args.highlights is not None else 0.0,
        "shadows": args.shadows if args.shadows is not None else 0.0,
        "whites": args.whites if args.whites is not None else 0.0,
        "blacks": args.blacks if args.blacks is not None else 0.0,
        # presence
        "clarity": args.clarity or 0.0,
        "dehaze": args.dehaze or 0.0,
        "vibrance": args.vibrance or 0.0,
        "saturation": args.saturation or 0.0,
        "bw": args.bw,
        # detail
        "sharp_amount": args.sharpness or 0.0,
        "sharp_radius": args.radius or 1.2,
        "nr_lum": args.nr_lum or 0.0,
        "nr_color": args.nr_color or 0.0,
        # effects
        "vignette_amount": args.vignette if args.vignette is not None else 0.0,
        "grain_amount": args.grain or 0.0,
        "glow_amount": args.glow or 0.0,
        # geometry
        "rotate90": args.rotate or 0,
        "straighten": args.straighten or 0.0,
        # lens
        "lens_distortion": (args.distortion / 100.0 * 30.0) if args.distortion else 0.0,
        "ca_shift": args.ca or 0.0,
        # underwater
        "uw_depth": args.uw_depth or 0.0,
        "uw_strength": args.uw_strength or 0.0,
        # sky
        "sky_enabled": False,
        "sky_preset": "",
        "sky_strength": 75.0,
        "sky_offset": 0.0,
        # LUT
        "lut_path": args.lut or "",
        "lut_enabled": bool(args.lut),
        # defaults for everything else
        "hsl": {b: [0, 0, 0] for b in ["red","orange","yellow","green",
                                        "aqua","blue","purple","magenta"]},
        "grade_shadows": [0,0,0], "grade_midtones": [0,0,0],
        "grade_highlights": [0,0,0], "blender": 50, "balance": 0,
        "curve_rgb": [], "curve_r": [], "curve_g": [], "curve_b": [],
        "cal_shadow_hue": 30, "cal_shadow_amt": 0,
        "cal_r": 0, "cal_g": 0, "cal_b": 0,
        "transform_v": 0, "transform_h": 0, "transform_scale": 0,
        "flip_h": False, "flip_v": False,
        "crop": None, "crop_aspect": "free", "masks": [],
        "vignette_midpoint": 50, "vignette_feather": 60,
    }
    return s


def cmd_edit(args):
    """Apply adjustments to a single photo."""
    from lumina.core.imaging import render_global
    img = _load(args.input, args.max_edge)
    s = _build_settings(args)
    u8 = render_global(img, s, scale=args.max_edge/2400 if args.max_edge else 1.0)
    _save(u8, args.output, args.quality)
    print(json.dumps({"status": "ok", "input": args.input,
                       "output": args.output,
                       "size": list(u8.shape[:2])}))


def cmd_enhance(args):
    """AI one-click enhancement."""
    from lumina.core.imaging import (render_global, compute_auto_wb,
                                    compute_auto_tone, default_settings)
    img = _load(args.input, args.max_edge)

    s = default_settings()
    t0, ti = compute_auto_wb(img)
    ev, bl, wh = compute_auto_tone(img)

    L = imaging.luma(img)
    mean_l = float(L.mean())
    p5, p95 = float(np.percentile(L, 5)), float(np.percentile(L, 95))

    if abs(ev) > 0.15:
        s["exposure"] = round(max(-0.8, min(0.8, ev)), 2)
    if abs(t0) > 4:
        s["temp"] = round(max(-20, min(20, t0)), 1)
    if abs(ti) > 3:
        s["tint"] = round(max(-12, min(12, ti)), 1)
    if p95 - p5 < 0.55:
        s["contrast"] = 14
    hi_frac = float((L > 0.95).mean())
    lo_frac = float((L < 0.03).mean())
    if hi_frac > 0.02: s["highlights"] = -22
    if lo_frac > 0.05: s["shadows"] = 18
    mx = img.max(axis=-1); mn = img.min(axis=-1)
    mean_sat = float(((mx-mn)/np.maximum(mx, 1e-4)).mean())
    if mean_sat < 0.15: s["vibrance"] = 14

    u8 = render_global(img, s, scale=args.max_edge/2400 if args.max_edge else 1.0)
    _save(u8, args.output, args.quality)
    applied = {k: v for k, v in s.items() if k in ("exposure","contrast","highlights",
               "shadows","temp","tint","vibrance") and v != 0}
    print(json.dumps({"status": "ok", "output": args.output,
                       "applied": applied}))


def cmd_sky(args):
    """Sky replacement."""
    from lumina.core.imaging import render_global
    from lumina.core.sky import detect_sky_mask, generate_sky, replace_sky
    img = _load(args.input, args.max_edge)
    mask = detect_sky_mask(img, offset=args.offset, softness=args.softness)
    sky_rgb = generate_sky(args.preset, img.shape[1], img.shape[0])
    strength = min(1.0, args.strength / 100.0)
    comp = replace_sky(img, mask, args.preset, strength=strength)
    u8 = np.clip(comp * 255, 0, 255).astype(np.uint8)
    _save(u8, args.output, args.quality)
    print(json.dumps({"status": "ok", "output": args.output,
                       "preset": args.preset, "strength": args.strength,
                       "mask_coverage": round(float(mask.mean()), 3)}))


def cmd_underwater(args):
    """Underwater colour restoration."""
    from lumina.core.imaging import apply_underwater, estimate_underwater_depth
    img = _load(args.input, args.max_edge)
    depth = args.depth if args.depth is not None else estimate_underwater_depth(img)
    strength = args.strength or 70
    corrected = apply_underwater(img, depth, strength)
    u8 = np.clip(corrected * 255, 0, 255).astype(np.uint8)
    _save(u8, args.output, args.quality)
    print(json.dumps({"status": "ok", "output": args.output,
                       "depth": depth, "strength": strength}))


def cmd_batch(args):
    """Batch process a directory."""
    from lumina.core.imaging import render_global
    import glob
    patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(args.dir, pat)))
    files.sort()
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    s = _build_settings(args)
    for i, path in enumerate(files):
        try:
            img = _load(path, args.max_edge)
            u8 = render_global(img, s, scale=1.0)
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(args.output_dir, f"{stem}_edit{args.ext}")
            _save(u8, out_path, args.quality)
            results.append({"input": path, "output": out_path})
        except Exception as e:
            results.append({"input": path, "error": str(e)})
    print(json.dumps({"status": "ok", "processed": len(results), "results": results}))


def cmd_export(args):
    """Export in specific format with resize."""
    from lumina.core.export import render_for_export
    from lumina.core.imaging import sanitize_settings
    sv = sanitize_settings(_build_settings(args))
    u8 = render_for_export(args.input, sv, max_edge=args.resize or None)
    _save(u8, args.output, args.quality)
    print(json.dumps({"status": "ok", "output": args.output,
                       "dimensions": [u8.shape[1], u8.shape[0]]}))


def cmd_info(args):
    """Print metadata as JSON."""
    meta = rawio.extract_metadata(args.input)
    meta["file_size"] = os.path.getsize(args.input)
    meta["is_raw"] = rawio.is_raw(args.input)
    print(json.dumps(meta, indent=2, default=str))


def cmd_cull(args):
    """Score photos and optionally auto-rate."""
    from lumina.core.cull import analyze, assign_ratings
    from lumina.core.rawio import decode_preview, extract_metadata
    import glob

    patterns = ["*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff"]
    files = []
    for pat in patterns:
        files.extend(glob.glob(os.path.join(args.dir, pat)))
    files.sort()

    results = []
    for path in files:
        try:
            prev = decode_preview(path, 480)
            res = analyze(prev)
            res["path"] = path
            results.append(res)
        except Exception as e:
            results.append({"path": path, "score": -99, "blurry": True})

    rated = assign_ratings(results, list(range(len(results))),
                           respect_existing=False,
                           mark_rejects=args.reject_blurry)
    out = []
    for (_pid, rating, flag, res) in rated:
        entry = {"path": res.get("path", ""), "score": res.get("score", 0),
                 "rating": rating}
        if flag is not None:
            entry["flag"] = flag
        out.append(entry)
    print(json.dumps({"status": "ok", "photos": len(out), "results": out}, indent=2))


def cmd_hdr(args):
    """HDR merge of bracketed exposures."""
    from lumina.core.merge import merge_hdr
    result = merge_hdr(args.inputs)
    _save(result, args.output, args.quality)
    print(json.dumps({"status": "ok", "output": args.output,
                       "inputs": len(args.inputs)}))


def cmd_pano(args):
    """Panorama stitch."""
    from lumina.core.merge import stitch_panorama
    result = stitch_panorama(args.inputs)
    _save(result, args.output, args.quality)
    print(json.dumps({"status": "ok", "output": args.output,
                       "size": [result.shape[1], result.shape[0]]}))


def main():
    parser = argparse.ArgumentParser(
        prog="lumina",
        description="Lumina RAW photo editor — CLI for AI agents and scripts")
    sub = parser.add_subparsers(dest="command")

    # edit
    p_edit = sub.add_parser("edit", help="Apply adjustments to a photo")
    p_edit.add_argument("input")
    p_edit.add_argument("-o", "--output", default=None)
    p_edit.add_argument("-q", "--quality", type=int, default=92)
    p_edit.add_argument("--max-edge", type=int, default=0)
    _add_tone_args(p_edit)
    _add_fx_args(p_edit)

    # enhance
    p_enh = sub.add_parser("enhance", help="AI one-click enhancement")
    p_enh.add_argument("input")
    p_enh.add_argument("-o", "--output", default=None)
    p_enh.add_argument("--max-edge", type=int, default=0)
    p_enh.add_argument("-q", "--quality", type=int, default=92)

    # sky
    p_sky = sub.add_parser("sky", help="Sky replacement")
    p_sky.add_argument("input")
    p_sky.add_argument("-o", "--output", default=None)
    p_sky.add_argument("--preset", default="Golden Sunset",
                       choices=["Golden Sunset","Dramatic Storm","Clear Blue",
                                "Twilight Stars","Pastel Dream"])
    p_sky.add_argument("--strength", type=float, default=85)
    p_sky.add_argument("--offset", type=float, default=0)
    p_sky.add_argument("--softness", type=float, default=45)
    p_sky.add_argument("--max-edge", type=int, default=0)
    p_sky.add_argument("-q", "--quality", type=int, default=92)

    # underwater
    p_uw = sub.add_parser("underwater", help="Underwater colour restoration")
    p_uw.add_argument("input")
    p_uw.add_argument("-o", "--output", default=None)
    p_uw.add_argument("--depth", type=float, default=None,
                      help="Depth 0-100 (auto-detected if not given)")
    p_uw.add_argument("--strength", type=float, default=70)
    p_uw.add_argument("--max-edge", type=int, default=0)
    p_uw.add_argument("-q", "--quality", type=int, default=92)

    # batch
    p_batch = sub.add_parser("batch", help="Batch process directory")
    p_batch.add_argument("dir")
    p_batch.add_argument("--output-dir", required=True)
    p_batch.add_argument("--ext", default=".jpg")
    p_batch.add_argument("--max-edge", type=int, default=0)
    p_batch.add_argument("-q", "--quality", type=int, default=92)
    _add_tone_args(p_batch)
    _add_fx_args(p_batch)

    # export
    p_exp = sub.add_parser("export", help="Export in specific format")
    p_exp.add_argument("input")
    p_exp.add_argument("-o", "--output", required=True)
    p_exp.add_argument("--resize", type=int, default=0)
    p_exp.add_argument("-q", "--quality", type=int, default=92)
    _add_tone_args(p_exp)

    # info
    p_info = sub.add_parser("info", help="Print metadata as JSON")
    p_info.add_argument("input")

    # cull
    p_cull = sub.add_parser("cull", help="Score photos for culling")
    p_cull.add_argument("dir")
    p_cull.add_argument("--reject-blurry", action="store_true")

    # HDR
    p_hdr = sub.add_parser("hdr", help="HDR merge bracketed exposures")
    p_hdr.add_argument("inputs", nargs="+")
    p_hdr.add_argument("-o", "--output", required=True)
    p_hdr.add_argument("-q", "--quality", type=int, default=92)

    # panorama
    p_pano = sub.add_parser("pano", help="Stitch panorama")
    p_pano.add_argument("inputs", nargs="+")
    p_pano.add_argument("-o", "--output", required=True)
    p_pano.add_argument("-q", "--quality", type=int, default=92)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    # set default output
    if hasattr(args, 'output') and args.output is None:
        stem = os.path.splitext(os.path.basename(args.input))[0]
        d = os.path.dirname(args.input) or "."
        args.output = os.path.join(d, f"{stem}_edited.jpg")
    if hasattr(args, 'quality') and not hasattr(args, '_quality_set'):
        pass

    cmds = {"edit": cmd_edit, "enhance": cmd_enhance, "sky": cmd_sky,
            "underwater": cmd_underwater, "batch": cmd_batch,
            "export": cmd_export, "info": cmd_info, "cull": cmd_cull,
            "hdr": cmd_hdr, "pano": cmd_pano}
    cmds[args.command](args)


def _add_tone_args(p):
    p.add_argument("--exposure", type=float, default=0.0)
    p.add_argument("--contrast", type=float, default=0.0)
    p.add_argument("--highlights", type=float, default=None)
    p.add_argument("--shadows", type=float, default=None)
    p.add_argument("--whites", type=float, default=None)
    p.add_argument("--blacks", type=float, default=None)
    p.add_argument("--temp", type=float, default=None)
    p.add_argument("--tint", type=float, default=None)
    p.add_argument("--clarity", type=float, default=0.0)
    p.add_argument("--dehaze", type=float, default=0.0)
    p.add_argument("--vibrance", type=float, default=0.0)
    p.add_argument("--saturation", type=float, default=0.0)
    p.add_argument("--bw", action="store_true")


def _add_fx_args(p):
    p.add_argument("--sharpness", type=float, default=0.0)
    p.add_argument("--radius", type=float, default=1.2)
    p.add_argument("--nr-lum", type=float, default=0.0)
    p.add_argument("--nr-color", type=float, default=0.0)
    p.add_argument("--vignette", type=float, default=None)
    p.add_argument("--grain", type=float, default=0.0)
    p.add_argument("--glow", type=float, default=0.0)
    p.add_argument("--rotate", type=int, default=0, choices=[-1, 0, 1])
    p.add_argument("--straighten", type=float, default=0.0)
    p.add_argument("--distortion", type=float, default=0.0)
    p.add_argument("--ca", type=float, default=0.0)
    p.add_argument("--uw-depth", type=float, default=0.0)
    p.add_argument("--uw-strength", type=float, default=0.0)
    p.add_argument("--lut", type=str, default="")


if __name__ == "__main__":
    main()
