"""Import Lightroom .xmp develop settings."""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET

# LR attribute → (Lumina key, scale_factor, is_bool)
_LR_MAP = {
    "Exposure2012":         ("exposure", 1.0, False),
    "Contrast2012":          ("contrast", 1.0, False),
    "Highlights2012":        ("highlights", 1.0, False),
    "Shadows2012":           ("shadows", 1.0, False),
    "Whites2012":            ("whites", 1.0, False),
    "Blacks2012":            ("blacks", 1.0, False),
    "Temperature":           ("temp", 1.0, False),
    "Tint":                  ("tint", 1.0, False),
    "Clarity2012":           ("clarity", 1.0, False),
    "Dehaze":                ("dehaze", 1.0, False),
    "Vibrance":              ("vibrance", 1.0, False),
    "Saturation":            ("saturation", 1.0, False),
    "Sharpness":             ("sharp_amount", 1.0, False),
    "LuminanceSmoothing":    ("nr_lum", 1.0, False),
    "ColorNoiseReduction":   ("nr_color", 1.0, False),
    "PostCropVignetteAmount":("vignette_amount", 1.0, False),
    "GrainAmount":           ("grain_amount", 1.0, False),
}

_HSL_MAP = {
    "HueAdjustmentRed": ("red", 0), "SaturationAdjustmentRed": ("red", 1),
    "LuminanceAdjustmentRed": ("red", 2),
    "HueAdjustmentOrange": ("orange", 0), "SaturationAdjustmentOrange": ("orange", 1),
    "LuminanceAdjustmentOrange": ("orange", 2),
    "HueAdjustmentYellow": ("yellow", 0), "SaturationAdjustmentYellow": ("yellow", 1),
    "LuminanceAdjustmentYellow": ("yellow", 2),
    "HueAdjustmentGreen": ("green", 0), "SaturationAdjustmentGreen": ("green", 1),
    "LuminanceAdjustmentGreen": ("green", 2),
    "HueAdjustmentAqua": ("aqua", 0), "SaturationAdjustmentAqua": ("aqua", 1),
    "LuminanceAdjustmentAqua": ("aqua", 2),
    "HueAdjustmentBlue": ("blue", 0), "SaturationAdjustmentBlue": ("blue", 1),
    "LuminanceAdjustmentBlue": ("blue", 2),
    "HueAdjustmentPurple": ("purple", 0), "SaturationAdjustmentPurple": ("purple", 1),
    "LuminanceAdjustmentPurple": ("purple", 2),
    "HueAdjustmentMagenta": ("magenta", 0), "SaturationAdjustmentMagenta": ("magenta", 1),
    "LuminanceAdjustmentMagenta": ("magenta", 2),
}


def parse_lr_preset(path_or_text: str) -> dict | None:
    """Parse a Lightroom .xmp develop settings file.
    Returns dict compatible with Lumina's EditSettings."""
    if os.path.isfile(path_or_text):
        text = open(path_or_text, "r", errors="replace").read()
    else:
        text = path_or_text

    # find crs: prefixed attributes in the RDF description
    out: dict = {}
    hsl = {b: [0.0, 0.0, 0.0] for b in
           ["red","orange","yellow","green","aqua","blue","purple","magenta"]}

    for key, (lum_key, scale, is_bool) in _LR_MAP.items():
        pattern = rf"<crs:{key}>([^<]+)</crs:{key}>"
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1)) * scale
                out[lum_key] = val
            except ValueError:
                pass

    # HSL adjustments
    for lr_name, (band, idx) in _HSL_MAP.items():
        pattern = rf"<crs:{lr_name}>([^<]+)</crs:{lr_name}>"
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                hsl[band][idx] = val
            except ValueError:
                pass

    if any(any(abs(x) > 0.01 for x in v) for v in hsl.values()):
        out["hsl"] = {b: list(hsl[b]) for b in hsl}

    # B&W
    if re.search(r"<crs:ConvertToGrayscale>\s*True\s*</crs:ConvertToGrayscale>", text):
        out["bw"] = True

    # Tone curve PV2012: points like "0, 0 r, g, b" — we take RGB composite
    tc_match = re.search(
        r'<crs:ToneCurvePV2012>\s*<rdf:Seq>(.*?)</rdf:Seq>',
        text, re.S)
    if tc_match:
        points = re.findall(r"<rdf:li>\s*(\d+)\.\s+(\d+)", tc_match.group(1))
        if len(points) >= 2:
            out["curve_rgb"] = [[int(a)/255.0, int(b)/255.0] for a, b in points]

    if not out:
        return None
    return out


def export_look_as_xmp(settings: dict, path: str) -> str:
    """Export current look as LR-compatible .xmp."""
    def fmt(v):
        return f"{v:+.4g}" if isinstance(v, (int, float)) else str(v)

    lines = ['''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Lumina">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/">''']

    crs_map = {v[0]: k for k, v in _LR_MAP.items()}
    for lum_key, (lr_key, _, _) in _LR_MAP.items():
        if lum_key in settings and abs(float(settings[lum_key])) > 0.001:
            val = settings[lum_key]
            lines.append(f"   <crs:{lr_key}>{fmt(val)}</crs:{lr_key}>")

    # B&W
    if settings.get("bw"):
        lines.append("   <crs:ConvertToGrayscale>True</crs:ConvertToGrayscale>")

    # HSL
    hsl = settings.get("hsl", {})
    band_names = {"red":"Red","orange":"Orange","yellow":"Yellow",
                  "green":"Green","aqua":"Aqua","blue":"Blue",
                  "purple":"Purple","magenta":"Magenta"}
    prefixes = [("HueAdjustment", 0), ("SaturationAdjustment", 1),
                ("LuminanceAdjustment", 2)]
    for prefix, idx in prefixes:
        for band, bname in band_names.items():
            v = hsl.get(band, [0,0,0])[idx]
            if abs(v) > 0.01:
                lines.append(f"   <crs:{prefix}{bname}>{fmt(v)}</crs:{prefix}{bname}>")

    # Tone curve
    curve = settings.get("curve_rgb", [])
    if len(curve) >= 2:
        pts = " ".join(f"<rdf:li>{int(p[0]*255)}, {int(p[1]*255)}</rdf:li>"
                       for p in curve)
        lines.append(f'''   <crs:ToneCurvePV2012>
    <rdf:Seq>{pts}</rdf:Seq>
   </crs:ToneCurvePV2012>''')

    lines += ["  </rdf:Description>", " </rdf:RDF>", "</x:xmpmeta>",
              '<?xpacket end="w"?>']
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


import os
