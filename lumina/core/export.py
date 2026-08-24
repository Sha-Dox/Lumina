"""Full-resolution export rendering.

GUARANTEE: This module NEVER writes to or modifies the original source file.
All exports create new files at the user-specified destination.
Originals are preserved byte-for-byte.
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image

from . import imaging


def render_for_export(path: str, settings: dict, max_edge: int | None = None,
                      output_sharpen: bool = True) -> np.ndarray:
    """Decode full-res, run identical pipeline as previews, return uint8 RGB."""
    from . import rawio
    from . import heal as healmod
    arr = rawio.decode_full(path)
    oriented = imaging.apply_geometry_flip_rot(arr, settings.get("rotate90", 0),
                                               settings.get("flip_h", False),
                                               settings.get("flip_v", False))
    # spot removal on the oriented image (matches interactive pipeline)
    oriented = healmod.apply_spots(oriented, settings.get("spots") or [])

    base = oriented.astype(np.float32) / 255.0
    seed = path
    out = imaging.render_global(base, settings, scale=2.5, seed_key=seed)

    # straighten first so masks live in the same space as the editor view
    ang = float(settings.get("straighten", 0.0))
    if abs(ang) > 0.01:
        out = imaging.apply_straighten(out, ang)
    out = imaging.apply_transform(out,
                                  settings.get("transform_v", 0.0),
                                  settings.get("transform_h", 0.0),
                                  settings.get("transform_scale", 0.0))

    # local adjustments (rasterized at straightened resolution)
    for m in settings.get("masks", []):
        try:
            marr = imaging.rasterize_mask(out.shape, m)
            out = imaging.apply_mask_to_image(out, marr, m["adjustments"],
                                              bool(m.get("invert")), 2.5)
        except Exception:
            continue

    out = imaging.crop_u8(out, settings.get("crop"))

    img = Image.fromarray(out)
    if max_edge and max(img.size) > max_edge:
        s = max_edge / max(img.size)
        img = img.resize((max(1, int(img.width * s)), max(1, int(img.height * s))),
                         Image.Resampling.LANCZOS)
        if output_sharpen:
            img = img.filter(__import__("PIL.ImageFilter", fromlist=["UnsharpMask"])
                             .UnsharpMask(radius=1.1, percent=60, threshold=2))
    elif output_sharpen:
        pass  # keep native sharpness; pipeline sharpening already applied
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def apply_watermark(u8: np.ndarray, text: str, opacity: float = 0.55,
                    scale: float = 3.0) -> np.ndarray:
    """Bottom-right text watermark. opacity 0..1, scale ~ font px per 100px."""
    if not text:
        return u8
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(u8).convert("RGBA")
    w_px, h_px = img.size
    font_px = max(14, int(h_px / 100.0 * max(1.5, scale)))
    font = None
    for cand in ("/System/Library/Fonts/Helvetica.ttc",
                 "/System/Library/Fonts/HelveticaNeue.ttc",
                 "/Library/Fonts/Arial.ttf"):
        try:
            font = ImageFont.truetype(cand, font_px)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, thh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    pad = int(font_px * 0.6)
    x = w_px - tw - pad * 2
    y = h_px - thh - pad * 2
    alpha = int(max(0.05, min(1.0, opacity)) * 255)
    draw.text((x + 2, y + 2), text, font=font,
              fill=(0, 0, 0, alpha))                       # shadow
    draw.text((x, y), text, font=font, fill=(255, 255, 255, alpha))
    out = Image.alpha_composite(img, overlay).convert("RGB")
    return np.asarray(out, dtype=np.uint8)


def save_image(u8: np.ndarray, dest_path: str, fmt: str, quality: int = 90) -> str:
    img = Image.fromarray(u8)
    fmt = fmt.upper()
    if fmt == "JPEG":
        if not dest_path.lower().endswith((".jpg", ".jpeg")):
            dest_path += ".jpg"
        img.save(dest_path, "JPEG", quality=int(quality), subsampling=1, optimize=True)
    elif fmt == "PNG":
        if not dest_path.lower().endswith(".png"):
            dest_path += ".png"
        img.save(dest_path, "PNG", compress_level=6)
    elif fmt == "TIFF":
        if not dest_path.lower().endswith((".tif", ".tiff")):
            dest_path += ".tif"
        img.save(dest_path, "TIFF", compression="tiff_lzw")
    else:
        raise ValueError(fmt)
    return dest_path


def unique_path(folder: str, stem: str, ext: str) -> str:
    cand = os.path.join(folder, f"{stem}{ext}")
    i = 1
    while os.path.exists(cand):
        cand = os.path.join(folder, f"{stem} ({i}){ext}")
        i += 1
    return cand
