"""Lumina MCP Server — lets Claude Code edit photos directly."""
import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmcp import FastMCP

mcp = FastMCP("lumina",
    instructions="Lumina RAW photo editor. Use these tools to programmatically adjust, enhance, analyse and export photos. All edits are non-destructive.")


def _load(path, max_edge=0):
    from lumina.core.rawio import decode_preview, decode_full
    arr = decode_full(path) if max_edge <= 0 else decode_preview(path, max_edge)
    return arr.astype(np.float32)/255.0


def _save(arr_u8, path):
    ext = os.path.splitext(path)[1].lower()
    from PIL import Image as PILImg
    if ext in (".jpg",".jpeg"):
        PILImg.fromarray(arr_u8).save(path, "JPEG", quality=92)
    elif ext == ".png":
        PILImg.fromarray(arr_u8).save(path, "PNG")
    else:
        PILImg.fromarray(arr_u8).save(path, "TIFF", compression="tiff_lzw")


def _resolve_output(path, suffix):
    base, ext = os.path.splitext(path)
    return f"{base}_{suffix}{ext or '.jpg'}"


@mcp.tool
def lumina_edit(
    path: str,
    exposure: float = 0.0,
    contrast: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    whites: float = 0.0,
    blacks: float = 0.0,
    temp: float = 0.0,
    tint: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    vibrance: float = 0.0,
    saturation: float = 0.0,
    bw: bool = False,
    sharpness: float = 0.0,
    nr_lum: float = 0.0,
    nr_color: float = 0.0,
    vignette: float = 0.0,
    grain: float = 0.0,
    glow: float = 0.0,
    rotate: int = 0,
    straighten: float = 0.0,
    distortion: float = 0.0,
    ca_shift: float = 0.0,
    uw_depth: float = 0.0,
    uw_strength: float = 0.0,
    lut_path: str = "",
    output: str = "",
) -> str:
    """Apply adjustments to a photo using Lumina's develop pipeline.
    All parameters are optional — only non-zero values are applied.
    Original file is never modified.

    Args:
        path: Path to the photo file (RAW or raster).
        exposure: Exposure in stops (-5..5). +1 doubles brightness.
        contrast: Contrast (-100..100).
        highlights: Highlight recovery (-100..100). Negative recovers detail.
        shadows: Shadow lift/deepen (-100..100). Positive brightens.
        whites: White point (-100..100).
        blacks: Black point (-100..100).
        temp: White balance temperature (-100..100). Positive warms.
        tint: White balance tint (-100..100). Positive shifts magenta.
        clarity: Midtone local contrast (-100..100).
        dehaze: Haze removal (-100..100).
        vibrance: Smart saturation (-100..100). Protects skin tones.
        saturation: Global saturation (-100..100).
        bw: Convert to black & white.
        sharpness: Sharpening amount (0..150).
        nr_lum: Luminance noise reduction (0..100).
        nr_color: Chroma noise reduction (0..100).
        vignette: Vignette amount (-100..100). Negative darkens corners.
        grain: Film grain amount (0..100).
        glow: Orton glow/bloom amount (0..100).
        rotate: Rotate 90°: -1 counter-clockwise, 1 clockwise.
        straighten: Straighten angle in degrees (-45..45).
        distortion: Lens distortion correction (-30..30).
        ca_shift: Chromatic aberration removal (0..100).
        uw_depth: Underwater depth estimate (0..100).
        uw_strength: Underwater correction strength (0..100).
        lut_path: Path to a .cube LUT file to apply.
        output: Output file path. Defaults to input_edited.jpg.
    """
    from lumina.core.imaging import render_global
    img = _load(path)
    s = {
        "temp": temp, "tint": tint, "exposure": exposure,
        "contrast": contrast, "highlights": highlights,
        "shadows": shadows, "whites": whites, "blacks": blacks,
        "clarity": clarity, "dehaze": dehaze, "vibrance": vibrance,
        "saturation": saturation, "bw": bw,
        "sharp_amount": sharpness, "sharp_radius": 1.2,
        "nr_lum": nr_lum, "nr_color": nr_color,
        "vignette_amount": vignette, "vignette_midpoint": 50.0,
        "vignette_feather": 60.0, "grain_amount": grain,
        "grain_size": 25.0, "glow_amount": glow,
        "rotate90": rotate, "straighten": straighten,
        "lens_distortion": distortion, "ca_shift": ca_shift,
        "uw_depth": uw_depth, "uw_strength": uw_strength,
        "lut_path": lut_path, "lut_enabled": bool(lut_path),
        "curve_rgb": [], "curve_r": [], "curve_g": [], "curve_b": [],
        "hsl": {b: [0,0,0] for b in ["red","orange","yellow","green",
                "aqua","blue","purple","magenta"]},
        "grade_shadows":[0,0,0], "grade_midtones":[0,0,0],
        "grade_highlights":[0,0,0], "blender":50.0, "balance":0.0,
        "cal_shadow_hue":30.0, "cal_shadow_amt":0.0,
        "cal_r":0.0, "cal_g":0.0, "cal_b":0.0,
        "transform_v":0.0, "transform_h":0.0, "transform_scale":0.0,
        "flip_h":False, "flip_v":False, "crop":None,
        "crop_aspect":"free", "masks":[],
        "sky_enabled":False, "sky_preset":"Golden Sunset",
        "sky_strength":75.0, "sky_softness":45.0, "sky_offset":0.0,
    }
    u8 = render_global(img, s, scale=1.0)
    out = output or _resolve_output(path, "edited")
    _save(u8, out)
    return json.dumps({"status":"ok", "output":out,
                       "dimensions":[u8.shape[1], u8.shape[0]]})


@mcp.tool
def lumina_enhance(path: str, output: str = "") -> str:
    """AI one-click enhancement. Analyses the photo and applies balanced
    corrections for exposure, white balance, contrast, highlight recovery,
    shadow lift, and vibrance.

    Args:
        path: Path to the photo.
        output: Output path. Defaults to input_enhanced.jpg.
    """
    from lumina.core.imaging import (render_global, compute_auto_wb,
                                      compute_auto_tone, default_settings)
    img = _load(path)
    s = default_settings()
    t0, ti = compute_auto_wb(img)
    ev, _, _ = compute_auto_tone(img)
    L = imaging.luma(img)
    p5, p95 = np.percentile(L, 5), np.percentile(L, 95)
    if abs(ev) > 0.15: s["exposure"] = round(max(-0.8,min(0.8,ev)),2)
    if abs(t0)>4: s["temp"]=round(max(-20,min(20,t0)),1)
    if abs(ti)>3: s["tint"]=round(max(-12,min(12,ti)),1)
    if p95-p5<0.55: s["contrast"]=14
    hi_f=float((L>0.95).mean()); lo_f=float((L<0.03).mean())
    if hi_f>0.02: s["highlights"]=-22
    if lo_f>0.05: s["shadows"]=18
    mx=img.max(axis=-1); mn=img.min(axis=-1)
    ms=float(((mx-mn)/np.maximum(mx,1e-4)).mean())
    if ms<0.15: s["vibrance"]=14
    u8 = render_global(img, s, scale=1.0)
    out = output or _resolve_output(path,"enhanced")
    _save(u8,out)
    applied = {k:v for k,v in s.items() if k in ("exposure","contrast",
               "highlights","shadows","temp","tint","vibrance") and v!=0}
    return json.dumps({"status":"ok","output":out,"applied":applied})


@mcp.tool
def lumina_sky_replace(
    path: str,
    preset: str = "Golden Sunset",
    strength: float = 85.0,
    offset: float = 0.0,
    softness: float = 45.0,
    output: str = "",
) -> str:
    """Replace the sky in a landscape photo with a procedural sky.

    Args:
        path: Photo file path.
        preset: One of Golden Sunset, Dramatic Storm, Clear Blue, Twilight Stars, Pastel Dream.
        strength: Blend strength 0-100.
        offset: Horizon shift -100 to 100.
        softness: Edge feathering 0-100.
        output: Output path.
    """
    from lumina.core.sky import detect_sky_mask, generate_sky, replace_sky
    img = _load(path)
    mask = detect_sky_mask(img, offset=offset, softness=softness)
    comp = replace_sky(img, mask, preset, strength=min(1.0,strength/100))
    u8 = np.clip(comp*255,0,255).astype(np.uint8)
    out = output or _resolve_output(path,"sky")
    _save(u8,out)
    return json.dumps({"status":"ok","output":out,"preset":preset,
                       "mask_coverage":round(float(mask.mean()),3)})


@mcp.tool
def lumina_underwater(path: str, depth: float = 0.0,
                      strength: float = 70.0, output: str = "") -> str:
    """Underwater colour restoration. Restores red/green absorbed by water.

    Args:
        path: Underwater photo path.
        depth: Depth estimate 0-100 (0 = auto-detect from colour cast).
        strength: Correction intensity 0-100.
        output: Output path.
    """
    from lumina.core.imaging import apply_underwater, estimate_underwater_depth
    img = _load(path)
    d = depth if depth > 0 else estimate_underwater_depth(img)
    corrected = apply_underwater(img, d, strength)
    u8 = np.clip(corrected*255,0,255).astype(np.uint8)
    out = output or _resolve_output(path,"uw")
    _save(u8,out)
    return json.dumps({"status":"ok","output":out,"depth_used":d,"strength":strength})


@mcp.tool
def lumina_info(path: str) -> str:
    """Get photo metadata: camera, lens, EXIF, dimensions, format.

    Args:
        path: Photo file path.
    """
    from lumina.core.rawio import extract_metadata, is_raw
    meta = extract_metadata(path)
    meta["file_size_bytes"] = os.path.getsize(path)
    meta["is_raw"] = is_raw(path)
    return json.dumps(meta, indent=2, default=str)


@mcp.tool
def lumina_batch_edit(
    directory: str,
    exposure: float = 0.0, contrast: float = 0.0,
    shadows: float = 0.0, highlights: float = 0.0,
    vibrance: float = 0.0, temperature: float = 0.0,
    output_dir: str = "",
) -> str:
    """Batch apply the same adjustments to all photos in a directory.

    Args:
        directory: Directory containing photos.
        exposure: Exposure in stops.
        contrast: Contrast adjustment.
        shadows: Shadow lift.
        highlights: Highlight recovery.
        vibrance: Vibrance boost.
        temperature: White balance shift.
        output_dir: Output directory for edited photos.
    """
    from lumina.core.imaging import render_global
    import glob
    files = []
    for pat in ["*.jpg","*.jpeg","*.png","*.tif","*.tiff"]:
        files.extend(glob.glob(os.path.join(directory,pat)))
    files.sort()
    od = output_dir or os.path.join(directory,"edited")
    os.makedirs(od, exist_ok=True)
    s = {"exposure":exposure,"contrast":contrast,"shadows":shadows,
         "highlights":highlights,"vibrance":vibrance,"temp":temperature}
    results = []
    for path in files:
        try:
            img = _load(path)
            u8 = render_global(img,s,scale=1.0)
            stem = os.path.splitext(os.path.basename(path))[0]
            op = os.path.join(od,f"{stem}_edit.jpg")
            _save(u8,op); results.append({"input":path,"output":op})
        except Exception as e:
            results.append({"input":path,"error":str(e)})
    return json.dumps({"status":"ok","processed":len(results),"results":results})


@mcp.tool
def lumina_export(
    path: str, output: str, format: str = "jpeg",
    quality: int = 92, resize_long_edge: int = 0,
    watermark_text: str = "", watermark_opacity: float = 0.6,
) -> str:
    """Export a photo with optional resize and watermark.

    Args:
        path: Source photo path.
        output: Output file path.
        format: jpeg, png, or tiff.
        quality: JPEG quality 1-100.
        resize_long_edge: Resize longest edge to N pixels (0=original).
        watermark_text: Text watermark overlay (empty=none).
        watermark_opacity: Watermark opacity 0-1.
    """
    from lumina.core.export import render_for_export, apply_watermark
    from lumina.core.imaging import default_settings
    sv = default_settings()
    u8 = render_for_export(path, sv, max_edge=resize_long_edge or None)
    if watermark_text:
        u8 = excore_apply_watermark(u8, watermark_text, watermark_opacity)
    _save(u8,output,quality)
    return json.dumps({"status":"ok","output":output})


def excore_apply_watermark(u8,text,opacity):
    from lumina.core.export import apply_watermark
    return apply_watermark(u8,text,opacity)


@mcp.tool
def lumina_cull(directory: str, reject_blurry: bool = True) -> str:
    """Score photos in a directory for culling. Rates best ★3, acceptable ★2,
    weak ★1, optionally flags blurry as rejected.

    Args:
        directory: Directory of photos to score.
        reject_blurry: Flag blurry shots as rejected.
    """
    from lumina.core.cull import analyze, assign_ratings
    from lumina.core.rawio import decode_preview
    import glob
    files = []
    for pat in ["*.jpg","*.jpeg","*.png","*.tif","*.tiff"]:
        files.extend(glob.glob(os.path.join(directory,pat)))
    files.sort()
    results = []; ids = list(range(len(files)))
    for path in files:
        try:
            prev = decode_preview(path,480); res = analyze(prev)
        except Exception: res = {"score":-99,"blurry":False}
        results.append(res)
    rated = assign_ratings(results, ids, respect_existing=False,
                           mark_rejects=reject_blurry)
    out = []
    for i,(pid,rating,flag,res) in enumerate(rated):
        entry = {"path":files[i],"score":res.get("score",0),"rating":rating}
        if flag is not None: entry["flag"] = flag
        out.append(entry)
    return json.dumps({"status":"ok","photos":len(out),
                       "results":out}, indent=2)


@mcp.tool
def lumina_hdr_merge(paths: list[str], output: str, align: bool = True) -> str:
    """Merge bracketed exposures into a single HDR image using Mertens fusion.

    Args:
        paths: 2+ bracketed photo paths (dark to bright).
        output: Output file path.
        align: Align before fusion (recommended for handheld).
    """
    from lumina.core.merge import merge_hdr
    result = merge_hdr(paths, align=align)
    _save(result,output)
    return json.dumps({"status":"ok","output":output,"inputs":len(paths)})


@mcp.tool
def lumina_panorama(paths: list[str], output: str) -> str:
    """Stitch overlapping photos into a panorama.

    Args:
        paths: Overlapping photo paths (left to right).
        output: Output file path.
    """
    from lumina.core.merge import stitch_panorama
    result = stitch_panorama(paths)
    _save(result,output)
    return json.dumps({"status":"ok","output":output,
                       "size":[result.shape[1],result.shape[0]]})


if __name__ == "__main__":
    mcp.run()
