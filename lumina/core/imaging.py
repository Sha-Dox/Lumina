"""Lumina imaging pipeline — pure numpy/PIL processing engine.

All functions operate on float32 images, HxWx3, RGB, range [0,1] (sRGB gamma space).
The same code path is used for interactive previews and full-resolution exports.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

_gaussian_filter = None


def _gauss():
    global _gaussian_filter
    if _gaussian_filter is None:
        from scipy.ndimage import gaussian_filter as _gf
        _gaussian_filter = _gf
    return _gaussian_filter

# ----------------------------------------------------------------------------- constants

HSL_BANDS = ["red", "orange", "yellow", "green", "aqua", "blue", "purple", "magenta"]
# hue ranges in degrees (start, end, falloff width)
HSL_RANGES = {
    "red":     (345.0, 15.0,  22.0),
    "orange":  (12.0,  46.0,  18.0),
    "yellow":  (38.0,  72.0,  18.0),
    "green":   (58.0, 165.0,  30.0),
    "aqua":    (150.0, 205.0, 22.0),
    "blue":    (192.0, 265.0, 24.0),
    "purple":  (252.0, 305.0, 22.0),
    "magenta": (292.0, 350.0, 22.0),
}

MASK_ADJ_KEYS = [
    "exposure", "contrast", "highlights", "shadows", "whites", "blacks",
    "temp", "tint", "clarity", "dehaze", "vibrance", "saturation",
]

DEFAULT_MASK_ADJUSTMENTS = {k: 0.0 for k in MASK_ADJ_KEYS}


def default_settings() -> dict:
    return {
        # White balance
        "temp": 0.0, "tint": 0.0,
        # Tone
        "exposure": 0.0, "contrast": 0.0,
        "highlights": 0.0, "shadows": 0.0, "whites": 0.0, "blacks": 0.0,
        # Presence
        "clarity": 0.0, "dehaze": 0.0, "vibrance": 0.0, "saturation": 0.0,
        "bw": False,
        # Tone curve (normalized points, y up = brighter); empty = linear
        "curve_rgb": [], "curve_r": [], "curve_g": [], "curve_b": [],
        # HSL / color mixer {band: [hue, sat, lum]}
        "hsl": {b: [0.0, 0.0, 0.0] for b in HSL_BANDS},
        # Color grading [hue_deg, sat 0-100, lum -100..100]
        "grade_shadows": [0.0, 0.0, 0.0],
        "grade_midtones": [0.0, 0.0, 0.0],
        "grade_highlights": [0.0, 0.0, 0.0],
        "grade_blender": 50.0, "grade_balance": 0.0,
        # Transform / perspective
        "transform_v": 0.0,        # vertical keystone -45..45
        "transform_h": 0.0,        # horizontal keystone
        "transform_scale": 0.0,    # -50..+50 percent
        # Calibration
        "cal_shadow_hue": 30.0, "cal_shadow_amt": 0.0,
        "cal_r": 0.0, "cal_g": 0.0, "cal_b": 0.0,
        # Detail
        "sharp_amount": 0.0, "sharp_radius": 1.2, "nr_lum": 0.0, "nr_color": 0.0,
        # Effects
        "vignette_amount": 0.0, "vignette_midpoint": 50.0, "vignette_feather": 60.0,
        "grain_amount": 0.0, "grain_size": 25.0,
        # Geometry
        "rotate90": 0,            # -1 | 0 | 1
        "flip_h": False, "flip_v": False,
        "straighten": 0.0,        # -45..45 degrees
        "crop": None,             # [x0,y0,x1,y1] normalized on oriented image
        "crop_aspect": "free",
        # AI tools
        "sky_enabled": False, "sky_preset": "Golden Sunset",
        "sky_strength": 75.0, "sky_softness": 45.0, "sky_offset": 0.0,
        "relight_angle": 300.0, "relight_strength": 0.0,
        "lut_path": "", "lut_enabled": False,
        "lens_distortion": 0.0, "ca_shift": 0.0,
        "glow_amount": 0.0,
        # Local adjustments
        "masks": [],
    }


def sanitize_settings(s: dict) -> dict:
    """Merge stored settings over defaults so old catalogs stay compatible."""
    d = default_settings()
    if not isinstance(s, dict):
        return d
    for k, v in s.items():
        if k == "hsl" and isinstance(v, dict):
            for b in HSL_BANDS:
                if b in v and isinstance(v[b], (list, tuple)) and len(v[b]) == 3:
                    d["hsl"][b] = [float(x) for x in v[b]]
        elif k == "masks" and isinstance(v, list):
            out = []
            for m in v:
                if isinstance(m, dict):
                    mm = dict(m)
                    adj = dict(DEFAULT_MASK_ADJUSTMENTS)
                    adj.update({k2: float(x) for k2, x in (mm.get("adjustments") or {}).items()
                                if k2 in MASK_ADJ_KEYS})
                    mm["adjustments"] = adj
                    out.append(mm)
            d["masks"] = out
        else:
            d[k] = v
    return d


# ----------------------------------------------------------------------------- helpers

def luma(img: np.ndarray) -> np.ndarray:
    """Rec.709 luma, HxW."""
    return img[..., 0] * 0.2126 + img[..., 1] * 0.7152 + img[..., 2] * 0.0722


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    if e1 <= e0:
        e0, e1 = e1, e0
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0).astype(np.float32)
    return t * t * (3.0 - 2.0 * t)


def gauss_blur_f(arr: np.ndarray, radius: float) -> np.ndarray:
    """Gaussian blur a float32 array (HxW or HxWx3), sigma in px."""
    if radius <= 0.02:
        return arr
    r = max(0.1, float(radius))
    gf = _gauss()
    if arr.ndim == 2:
        return gf(arr, sigma=r, mode="nearest").astype(np.float32)
    return gf(arr, sigma=(r, r, 0.0), mode="nearest").astype(np.float32)


# ------------------------------------------------------------------- vectorized HSV

def rgb_to_hsv(rgb: np.ndarray):
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    diff = mx - mn
    h = np.zeros_like(mx)
    m = diff > 1e-6
    idx = m & (mx == r)
    h[idx] = (60.0 * ((g[idx] - b[idx]) / diff[idx])) % 360.0
    idx = m & (mx == g) & (mx != r)
    h[idx] = (60.0 * ((b[idx] - r[idx]) / diff[idx])) + 120.0
    idx = m & (mx == b) & (mx != r) & (mx != g)
    h[idx] = (60.0 * ((r[idx] - g[idx]) / diff[idx])) + 240.0
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return h.astype(np.float32), s.astype(np.float32), mx.astype(np.float32)


def hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    h = np.mod(h, 360.0) / 60.0
    i = np.floor(h).astype(np.int32) % 6
    f = h - np.floor(h)
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return np.stack([r, g, b], axis=-1).astype(np.float32)


# ------------------------------------------------------------------- tone curve

def _lut_lookup(img: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fast LUT: assumes xs is uniform 0..1 grid. Integer-index gather."""
    n = len(xs)
    idx = (img * (n - 1)).astype(np.int32)
    np.clip(idx, 0, n - 1, out=idx)
    return ys[idx]


def monotonic_spline(pts, n=1024):
    """Fritsch-Carlson monotonic cubic through pts [(x,y)...] normalized.
    Returns (xs, ys) grids of length n."""
    pts = sorted(((float(p[0]), float(p[1])) for p in pts))
    xs = np.array([p[0] for p in pts], dtype=np.float64)
    ys = np.array([p[1] for p in pts], dtype=np.float64)
    # ensure endpoints span
    if xs[0] > 0.0:
        xs = np.concatenate([[0.0], xs]); ys = np.concatenate([[ys[0]], ys])
    if xs[-1] < 1.0:
        xs = np.concatenate([xs, [1.0]]); ys = np.concatenate([ys, [ys[-1]]])
    dxs = np.diff(xs)
    dys = np.diff(ys)
    dys = np.where(dxs == 0, 0.0, dys)
    dxs = np.where(dxs == 0, 1e-9, dxs)
    ms = dys / dxs
    c1 = np.zeros_like(ms)
    if len(ms) > 1:
        for i in range(1, len(ms)):
            if ms[i - 1] * ms[i] <= 0:
                c1[i] = 0.0
            else:
                w1, w2 = 2 * dxs[i] + dxs[i - 1], dxs[i] + 2 * dxs[i - 1]
                c1[i] = (w1 + w2) / (w1 / ms[i - 1] + w2 / ms[i])
    c1 = np.concatenate([[ms[0]], c1, [ms[-1]]])  # length len(xs)

    xq = np.linspace(0.0, 1.0, n)
    seg = np.clip(np.searchsorted(xs, xq, side="right") - 1, 0, len(dxs) - 1)
    sdx = xq - xs[seg]
    h = dxs[seg]
    yq = (ys[seg]
          + c1[seg] * sdx
          + (3 * ms[seg] - 2 * c1[seg] - c1[seg + 1]) * sdx**2 / h
          + (ms[seg] - c1[seg] - c1[seg + 1]) * sdx**3 / (h * h))
    return xq.astype(np.float32), np.clip(yq, -0.05, 1.05).astype(np.float32)


def apply_curve_lut(img: np.ndarray, pts) -> np.ndarray:
    if not pts or len(pts) < 2:
        return img
    xs, ys = monotonic_spline(pts)
    out = img.copy()
    for c in range(3):
        out[..., c] = _lut_lookup(out[..., c], xs, ys)
    return np.clip(out, 0.0, 1.0)


def apply_per_channel_curves(img: np.ndarray, chan_pts) -> np.ndarray:
    """chan_pts: list of 3 point-lists for R,G,B."""
    xs = np.linspace(0.0, 1.0, 1024).astype(np.float32)
    out = img.copy()
    for c in range(3):
        pts = chan_pts[c]
        if pts and len(pts) >= 2:
            _, ys = monotonic_spline(pts)
            out[..., c] = _lut_lookup(out[..., c], xs, ys)
    return np.clip(out, 0.0, 1.0)


# ------------------------------------------------------------------- global ops

def apply_wb(img: np.ndarray, temp: float, tint: float) -> np.ndarray:
    if abs(temp) < 0.01 and abs(tint) < 0.01:
        return img
    t = temp / 100.0
    ti = tint / 100.0
    gains = np.array([1.0 + 0.28 * t,
                      1.0 - 0.10 * abs(ti) * 0.0 - 0.06 * ti * 0.0 + 1.0 - 1.0,
                      1.0 - 0.28 * t], dtype=np.float32)
    gains[1] = 1.0 - 0.14 * ti
    out = img.copy()
    for c in range(3):
        out[..., c] *= gains[c]
    return np.clip(out, 0.0, 1.0)


_exposure_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def _exposure_lut(ev: float):
    key = round(float(ev), 3)
    hit = _exposure_cache.get(key)
    if hit is not None:
        return hit
    x = np.linspace(0.0, 1.0, 2048, dtype=np.float64)
    lin = x ** 2.2
    fac = 2.0 ** ev
    out_lin = np.clip(lin * fac, 0.0, 1.0)
    y = (out_lin ** (1.0 / 2.2)).astype(np.float32)
    if len(_exposure_cache) > 48:
        _exposure_cache.clear()
    _exposure_cache[key] = (x.astype(np.float32), y)
    return x.astype(np.float32), y


def apply_exposure(img: np.ndarray, ev: float) -> np.ndarray:
    if abs(ev) < 0.001:
        return img
    xs, ys = _exposure_lut(ev)
    out = _lut_lookup(img, xs, ys)
    return np.clip(out, 0.0, 1.0)


def apply_contrast(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    c = amount / 100.0
    L = luma(img)[..., None]
    if c >= 0:
        k = 1.0 + 6.0 * c
        sig = 1.0 / (1.0 + np.exp(np.clip(-(L - 0.5) * k, -30, 30)))
        out = img * (1.0 - c) + sig * c
    else:
        f = 1.0 + c * 0.85          # c negative → flatten
        out = 0.5 + (img - 0.5) * f
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_highlights(img: np.ndarray, amount: float) -> np.ndarray:
    """Negative recovers (darkens) highlights, positive brightens."""
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)
    w = smoothstep(0.35, 0.95, L) ** 1.2
    if a < 0:
        # recovery: pull down proportional to how bright, protect mids
        out = img + (a * 0.75) * w[..., None] * (img - L[..., None] * 0.6)
    else:
        out = img + (a * 0.55) * w[..., None] * (1.0 - img)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_shadows(img: np.ndarray, amount: float) -> np.ndarray:
    """Positive lifts shadows, negative deepens."""
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)
    w = (1.0 - smoothstep(0.0, 0.55, L)) ** 1.4
    if a > 0:
        out = img + (a * 0.62) * w[..., None] * (1.0 - img)
    else:
        out = img + (a * 0.55) * w[..., None] * img
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_whites(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)
    w = smoothstep(0.45, 1.0, L) ** 2.0
    if a > 0:
        out = img + (a * 0.45) * w[..., None] * (1.0 - img)
    else:
        out = img + (a * 0.40) * w[..., None] * img
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_blacks(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)
    w = (1.0 - smoothstep(0.02, 0.42, L)) ** 2.0
    if a < 0:
        out = img + (a * 0.40) * w[..., None] * img
    else:
        out = img + (a * 0.42) * w[..., None] * (0.35 - img).clip(-0.35, 0.35) * 1.4
    return np.clip(out.astype(np.float32), 0.0, 1.0)


# ------------------------------------------------------------------- presence

def apply_clarity(img: np.ndarray, amount: float, scale: float = 1.0) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)
    radius = max(2.0, 14.0 * scale)
    Lb = gauss_blur_f(L, radius)
    detail = L - Lb
    midw = 1.0 - np.abs(2.0 * smoothstep(0.0, 1.0, L) - 1.0)
    midw = midw ** 0.7
    delta = (a * 0.9) * detail * midw
    out = img.copy()
    if a > 0:
        out += delta[..., None] * (out / (L[..., None] + 1e-4))
    else:
        out += delta[..., None]
    return np.clip(out, 0.0, 1.0)


def apply_dehaze(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    dark = np.min(gauss_blur_f(img.copy(), max(3.0, img.shape[1] / 180.0)), axis=-1)
    if a > 0:
        omega = min(0.92, 0.65 * a + 0.10)
        flat_dark = dark.reshape(-1)
        n_pick = max(64, int(flat_dark.size * 0.001))
        idx = np.argpartition(flat_dark, -n_pick)[-n_pick:]
        A = np.percentile(img.reshape(-1, 3)[idx], 90, axis=0).astype(np.float32)
        A = np.clip(A, 0.08, 0.98)
        t = np.clip(1.0 - omega * (dark / float(np.mean(A))), 0.18, 1.0)[..., None]
        J = (img - (1.0 - t) * A[None, None, :]) / t
        J = np.nan_to_num(J, nan=0.0, posinf=1.0, neginf=0.0)
        out = np.clip(J, 0.0, 1.0)
    else:
        # add haze
        k = -a
        veil = np.array([0.62, 0.66, 0.72], dtype=np.float32)
        out = img * (1.0 - k * 0.55) + veil[None, None, :] * (k * 0.55)
    return np.clip(out.astype(np.float32), 0.0, 1.0)


def apply_vibrance(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    h, s, v = rgb_to_hsv(img)
    boost = 1.0 + a * (1.0 - s) * 1.1
    s2 = np.clip(s * boost, 0.0, 1.0)
    return hsv_to_rgb(h, s2, v)


def apply_saturation(img: np.ndarray, amount: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    a = amount / 100.0
    L = luma(img)[..., None]
    return np.clip(L + (img - L) * (1.0 + a), 0.0, 1.0)


# ------------------------------------------------------------------- HSL / B&W

def _band_weight(hue: np.ndarray, band: str) -> np.ndarray:
    lo, hi, fall = HSL_RANGES[band]
    d = np.mod(hue - lo, 360.0)
    span = (hi - lo) % 360.0
    inside = smoothstep(-fall * 0.5, fall * 0.5 + 0.001, d) * \
        (1.0 - smoothstep(span - fall * 0.5, span + fall * 0.5 + 0.001, d))
    core = smoothstep(-fall * 0.15, fall * 0.15, d) * \
        (1.0 - smoothstep(span - fall * 0.15, span + fall * 0.15, d))
    return np.maximum(inside * 0.55, core).astype(np.float32)


def apply_hsl(img: np.ndarray, hsl: dict) -> np.ndarray:
    hs = [abs(float(v[0])) + abs(float(v[1])) + abs(float(v[2])) for v in hsl.values()]
    if max(hs) < 0.01:
        return img
    h, s, v = rgb_to_hsv(img)
    active = s > 0.004
    hue_shift = np.zeros_like(h)
    sat_mul = np.ones_like(s)
    lum_add = np.zeros_like(s)
    for band, (dh, ds, dl) in hsl.items():
        if abs(dh) + abs(ds) + abs(dl) < 0.01:
            continue
        w = _band_weight(h, band) * active
        hue_shift += (dh / 100.0) * 32.0 * w
        sat_mul *= (1.0 + (ds / 100.0) * 1.15 * w)
        lum_add += (dl / 100.0) * 0.16 * w
    s2 = np.clip(s * sat_mul, 0.0, 1.0)
    h2 = np.mod(h + hue_shift, 360.0)
    out = hsv_to_rgb(h2, s2, v)
    L = luma(out)[..., None]
    out = out + (lum_add[..., None])
    return np.clip(out, 0.0, 1.0)


def apply_bw(img: np.ndarray, hsl: dict) -> np.ndarray:
    """Black & white conversion using mixer luminance as channel weights."""
    wr = 0.2126 * (1.0 + hsl["red"][2] / 130.0)
    wg = 0.7152 * (1.0 + hsl["green"][2] / 130.0 + hsl["yellow"][2] / 220.0)
    wb_ = 0.0722 * (1.0 + hsl["blue"][2] / 110.0 + hsl["aqua"][2] / 190.0)
    total = wr + wg + wb_
    gray = (img[..., 0] * wr + img[..., 1] * wg + img[..., 2] * wb_) / total
    return np.clip(gray[..., None].repeat(3, axis=-1), 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------- color grading

def apply_color_grade(img: np.ndarray, sh: list, mt: list, hi: list,
                      blender: float, balance: float) -> np.ndarray:
    def wheel_active(w):
        return abs(w[1]) > 0.01 or abs(w[2]) > 0.01
    if not (wheel_active(sh) or wheel_active(mt) or wheel_active(hi)):
        return img
    L = luma(img)
    bal = balance / 100.0
    center = 0.5 + bal * 0.25
    spread = 0.28 + (blender / 100.0) * 0.34

    w_shadow = (1.0 - smoothstep(center - spread * 0.9, center + spread, L)) ** 1.3
    w_high = smoothstep(center - spread, center + spread * 0.9, L) ** 1.3
    sigma = spread * 0.62
    w_mid = np.exp(-((L - center) ** 2) / (2.0 * sigma * sigma))
    w_mid = w_mid ** 1.2

    out = img.copy()
    for wmap, spec, mode in (
        (w_shadow, sh, "shadow"), (w_mid, mt, "mid"), (w_high, hi, "high"),
    ):
        hue, sat, lum = float(spec[0]), float(spec[1]) / 100.0, float(spec[2]) / 100.0
        if sat < 0.005 and abs(lum) < 0.005:
            continue
        tcolor = hsv_to_rgb(np.full_like(L, hue % 360.0),
                            np.full_like(L, min(1.0, sat * 1.6)), np.ones_like(L))
        strength = sat * 0.42
        if mode == "shadow":
            tinted = out * (1.0 - strength * 0.85) + tcolor * strength * 0.85 * out * 1.7
            out = out * (1.0 - strength) + np.clip(tinted, 0, 1) * strength
        elif mode == "high":
            tinted = out + (tcolor - 0.5) * strength * out * 1.6
            out = out * (1.0 - strength) + np.clip(tinted, 0, 1) * strength
        else:
            bell = out * (1.0 - out) * 4.0
            tinted = out + (tcolor - 0.5) * strength * bell
            out = out * (1.0 - strength) + np.clip(tinted, 0, 1) * strength
        if abs(lum) > 0.005:
            out = out + lum * 0.28 * wmap[..., None]
    return np.clip(out.astype(np.float32), 0.0, 1.0)


# ------------------------------------------------------------------- detail

def apply_sharpen(img: np.ndarray, amount: float, radius: float, scale: float = 1.0) -> np.ndarray:
    if amount < 0.5:
        return img
    a = amount / 100.0
    L = luma(img)
    r = max(0.3, radius * scale)
    Lb = gauss_blur_f(L, r)
    detail = L - Lb
    out = img.copy()
    out += (a * 1.35 * detail)[..., None]
    return np.clip(out, 0.0, 1.0)


def apply_nr(img: np.ndarray, nr_lum: float, nr_color: float) -> np.ndarray:
    out = img
    if nr_color > 0.5:
        a = min(nr_color, 100.0) / 100.0
        L = luma(out)[..., None]
        C = out - L
        Cb = gauss_blur_f(C, 1.5 + 4.0 * a)
        out = np.clip(L + Cb, 0.0, 1.0)
    if nr_lum > 0.5:
        a = min(nr_lum, 100.0) / 100.0
        L = luma(out)
        Lb = gauss_blur_f(L, 0.8 + 2.2 * a)
        diff = np.abs(L - Lb)
        keep = smoothstep(0.015, 0.09, diff)
        Lnew = Lb + (L - Lb) * keep
        out = out + (Lnew - L)[..., None]
    return np.clip(out.astype(np.float32), 0.0, 1.0)


# ------------------------------------------------------------------- effects

def apply_vignette(img: np.ndarray, amount: float, midpoint: float, feather: float) -> np.ndarray:
    if abs(amount) < 0.01:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = w / 2.0, h / 2.0
    d = np.sqrt(((xx - cx) / (w / 2.0)) ** 2 + ((yy - cy) / (h / 2.0)) ** 2) / math.sqrt(2)
    m = max(0.02, midpoint / 100.0)
    f = max(0.02, feather / 100.0)
    end = m + (1.0 - m) * f
    fall = smoothstep(m, end, d) ** 2
    a = amount / 100.0
    out = img * (1.0 + a * 1.15 * fall[..., None])
    return np.clip(out, 0.0, 1.0)


_grain_cache: dict[tuple, np.ndarray] = {}


def apply_grain(img: np.ndarray, amount: float, size: float, seed_key: str = "") -> np.ndarray:
    if amount < 0.5:
        return img
    h, w = img.shape[:2]
    cell = max(1.0, (size / 100.0) * 3.2)
    gh, gw = max(2, int(h / cell)), max(2, int(w / cell))
    key = (gh, gw, seed_key)
    n = _grain_cache.get(key)
    if n is None:
        rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))
        small = rng.standard_normal((gh, gw)).astype(np.float32)
        n = gauss_blur_f(small, 0.6)
        if len(_grain_cache) > 8:
            _grain_cache.clear()
        _grain_cache[key] = n
    big = np.asarray(
        Image.fromarray(n, mode="F").resize((w, h), Image.Resampling.BILINEAR), dtype=np.float32)
    a = amount / 100.0 * 0.16
    return np.clip(img + (big * a)[..., None], 0.0, 1.0)


# ------------------------------------------------------------------- geometry

def apply_lens_distortion(img_f32: np.ndarray, k1: float) -> np.ndarray:
    """k1 > 0 = barrel reduction, k1 < 0 = pincushion. Range ~ -0.3..0.3."""
    if abs(k1) < 0.002:
        return img_f32
    import cv2
    h, w = img_f32.shape[:2]
    f = max(w, h)
    cam = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]], dtype=np.float64)
    dist = np.array([k1, 0, 0, 0, 0], dtype=np.float64)
    map1, map2 = cv2.initUndistortRectifyMap(cam, dist, None, cam,
                                              (w, h), cv2.CV_32FC1)
    return cv2.remap(img_f32, map1, map2, cv2.INTER_LINEAR)


def remove_chromatic_aberration(img_f32: np.ndarray,
                                amount: float) -> np.ndarray:
    """amount -100..100; shifts R/B radially toward G alignment."""
    if abs(amount) < 0.5:
        return img_f32
    import cv2
    h, w = img_f32.shape[:2]
    s = (abs(amount)/100.0) * 0.004 * (1 if amount > 0 else -1)
    cx, cy = w/2.0, h/2.0
    xs = (np.arange(w, dtype=np.float32) - cx) / cx
    ys = (np.arange(h, dtype=np.float32) - cy) / cy
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    r2 = xx*xx + yy*yy
    scale_r = 1.0 - s * r2
    scale_b = 1.0 + s * r2
    out = img_f32.copy()
    for ch, sc in ((0, scale_b), (2, scale_r)):
        mx = (xx * sc * cx + cx).astype(np.float32)
        my = (yy * sc * cy + cy).astype(np.float32)
        out[..., ch] = cv2.remap(img_f32[..., ch].astype(np.float32),
                                 mx, my, cv2.INTER_LINEAR)
    return out


def apply_glow(out_f: np.ndarray, amount: float) -> np.ndarray:
    """Orton-style bloom: screen-blur blend."""
    if amount < 0.5:
        return out_f
    from .fastpath import blur_f
    a = min(amount, 100.0) / 100.0 * 0.55
    blurred = blur_f(out_f, max(6.0, out_f.shape[1]*0.02))
    screen = 1.0 - (1.0 - out_f) * (1.0 - blurred * 0.85)
    return np.clip(out_f + (screen - out_f) * a, 0.0, 1.0)


def apply_lut_if_enabled(out_u8: np.ndarray, s: dict) -> np.ndarray:
    if not s.get("lut_enabled") or not s.get("lut_path"):
        return out_u8
    from .lutio import load_cube, apply_cube
    try:
        cube, dim = load_cube(s["lut_path"])
        f = out_u8.astype(np.float32) / 255.0
        return (apply_cube(f, cube, dim) * 255).astype(np.uint8)
    except Exception as e:
        print("[lut]", e)
        return out_u8


def apply_geometry_flip_rot(out_f: np.ndarray, angle_deg: float,
                  strength: float) -> np.ndarray:
    """Directional relight - brightens one side, shades the other."""
    if abs(strength) < 0.01:
        return out_f
    h, w = out_f.shape[:2]
    rad = math.radians(angle_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    xn = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    yn = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    t = (xn * dx + yn * dy)
    t = (t + 1.0) * 0.5
    t = t ** 1.15
    factor = 2.0 ** ((t - 0.5) * (strength / 100.0) * 1.7)
    return np.clip(out_f * factor[..., None], 0.0, 1.0)


def apply_relight(out_f: np.ndarray, angle_deg: float,
                  strength: float) -> np.ndarray:
    """Directional relight - brightens one side, shades the other."""
    if abs(strength) < 0.01:
        return out_f
    import math as _m
    h, w = out_f.shape[:2]
    rad = _m.radians(angle_deg)
    dx, dy = _m.cos(rad), _m.sin(rad)
    xn = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]
    yn = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    t = (xn * dx + yn * dy)
    t = (t + 1.0) * 0.5
    t = t ** 1.15
    factor = 2.0 ** ((t - 0.5) * (strength / 100.0) * 1.7)
    return np.clip(out_f * factor[..., None], 0.0, 1.0)


def apply_sky_if_enabled(arr: np.ndarray, s: dict):
    """Returns composited float32 [0,1] when sky replacement is enabled."""
    if not s.get("sky_enabled"):
        return None
    from .sky import detect_sky_mask, replace_sky
    f = arr.astype(np.float32) / 255.0 if arr.dtype == np.uint8 else arr
    m = detect_sky_mask(f, offset=s.get("sky_offset", 0.0),
                        softness=s.get("sky_softness", 45.0))
    out = replace_sky(f, m, s.get("sky_preset", "Golden Sunset"),
                      strength=min(1.0, s.get("sky_strength", 75.0)/100.0))
    return np.clip(out, 0.0, 1.0)


def apply_geometry_flip_rot(img: np.ndarray, rotate90: int, flip_h: bool, flip_v: bool) -> np.ndarray:
    out = img
    if flip_h:
        out = out[:, ::-1]
    if flip_v:
        out = out[::-1]
    if rotate90 == 1:
        out = np.rot90(out, k=-1)
    elif rotate90 == -1:
        out = np.rot90(out, k=1)
    return np.ascontiguousarray(out)


def apply_transform(img_u8: np.ndarray, vert: float, horiz: float,
                    scale_pct: float) -> np.ndarray:
    """Perspective keystone + scale about center. All params in LR-like units."""
    import cv2
    if abs(vert) < 0.01 and abs(horiz) < 0.01 and abs(scale_pct) < 0.01:
        return img_u8
    h, w = img_u8.shape[:2]
    kx = max(-45.0, min(45.0, float(vert))) / 100.0     # fraction of width
    ky = max(-45.0, min(45.0, float(horiz))) / 100.0
    z = max(0.5, min(2.0, 1.0 + float(scale_pct) / 100.0))
    cx, cy = w / 2.0, h / 2.0

    # source corners mapped from a straightened output rect with keystoning
    off_x = w * 0.5 * kx
    off_y = h * 0.5 * ky
    src = np.float32([[cx - w*0.35 - off_x, cy - h*0.35],
                      [cx + w*0.35 - off_x, cy - h*0.35],
                      [cx + w*0.35 + off_x, cy + h*0.35],
                      [cx - w*0.35 + off_x, cy + h*0.35]])
    dst = np.float32([[cx - w*0.35*z, cy - h*0.35*z],
                      [cx + w*0.35*z, cy - h*0.35*z],
                      [cx + w*0.35*z, cy + h*0.35*z],
                      [cx - w*0.35*z, cy + h*0.35*z]])
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img_u8, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def apply_calibration(img: np.ndarray, sh_hue: float, sh_amt: float,
                      cr: float, cg: float, cb: float) -> np.ndarray:
    """Shadow tint + RGB primary gains (approximate camera-calibration feel)."""
    out = img
    if abs(sh_amt) > 0.005:
        L = luma(out)
        wsh = ((1.0 - smoothstep(0.0, 0.4, L)) ** 1.6)[..., None]
        tc = hsv_to_rgb(np.full_like(L, sh_hue % 360.0),
                        np.full_like(L, 1.0), np.ones_like(L))
        out = out + tc * wsh * (sh_amt / 100.0) * 0.22
        out = np.clip(out, 0.0, 1.0)
    gains = np.array([1.0 + cr / 320.0, 1.0 + cg / 320.0, 1.0 + cb / 320.0],
                     dtype=np.float32)
    if not np.allclose(gains, 1.0):
        L = luma(out)[..., None]
        out = np.clip(L + (out - L) * 0.25 + out * gains[None, None, :] * 0.75,
                      0.0, 1.0)
    return out.astype(np.float32)


def apply_straighten(img_u8: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate with expanded canvas (bicubic). Input/output uint8 RGB."""
    if abs(angle_deg) < 0.01:
        return img_u8
    im = Image.fromarray(img_u8, mode="RGB")
    return np.asarray(im.rotate(angle_deg, resample=Image.Resampling.BICUBIC, expand=True,
                                fillcolor=(0, 0, 0)), dtype=np.uint8)


def crop_u8(img_u8: np.ndarray, crop) -> np.ndarray:
    """crop = [x0,y0,x1,y1] normalized on this image. None = no crop."""
    if not crop:
        return img_u8
    h, w = img_u8.shape[:2]
    x0 = int(round(crop[0] * w)); y0 = int(round(crop[1] * h))
    x1 = int(round(crop[2] * w)); y1 = int(round(crop[3] * h))
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return img_u8
    return np.ascontiguousarray(img_u8[y0:y1, x0:x1])


# ------------------------------------------------------------------- auto tones

def compute_auto_wb(img: np.ndarray):
    """Gray-world auto WB → suggested (temp, tint) offsets."""
    L = luma(img)
    sel = (L > 0.08) & (L < 0.92)
    if not np.any(sel):
        return 0.0, 0.0
    mr, mg, mb = [float(np.mean(img[..., c][sel])) for c in range(3)]
    dr = (mr - mg) / max(mg, 1e-4)   # + → too warm already → cool down
    db = (mb - mg) / max(mg, 1e-4)   # + → too blue already → warm up
    temp = -(dr * 140.0 - db * 140.0)
    dm = mg - (mr + mb) / 2.0
    tint = dm / max(mg, 1e-4) * 160.0
    return float(np.clip(temp, -100, 100)), float(np.clip(tint, -100, 100))


def compute_auto_tone(img: np.ndarray):
    """Suggest exposure/blacks/whites like LR Auto (conservative)."""
    L = luma(img)
    p2 = float(np.percentile(L, 2))
    p98 = float(np.percentile(L, 98))
    pm = float(np.percentile(L, 50))
    ev = 0.0
    if pm > 0.03:
        ev = math.log2(max(0.18, 0.44) / max(pm, 0.02)) * 0.85
    ev = float(np.clip(ev, -2.5, 2.5))
    blacks = float(np.clip((-0.04 - p2) * 420, -70, 40))
    whites = float(np.clip((0.985 - p98) * 380, -30, 80))
    return ev, blacks, whites


# ------------------------------------------------------------------- dispatch

def apply_adjustment_subset(img: np.ndarray, adj: dict, scale: float = 1.0) -> np.ndarray:
    """Apply a (possibly partial) adjustment dict — used for masks & quick edits."""
    out = img
    out = apply_wb(out, adj.get("temp", 0.0), adj.get("tint", 0.0))
    out = apply_exposure(out, adj.get("exposure", 0.0))
    out = apply_contrast(out, adj.get("contrast", 0.0))
    out = apply_highlights(out, adj.get("highlights", 0.0))
    out = apply_shadows(out, adj.get("shadows", 0.0))
    out = apply_whites(out, adj.get("whites", 0.0))
    out = apply_blacks(out, adj.get("blacks", 0.0))
    out = apply_clarity(out, adj.get("clarity", 0.0), scale)
    out = apply_dehaze(out, adj.get("dehaze", 0.0))
    out = apply_vibrance(out, adj.get("vibrance", 0.0))
    out = apply_saturation(out, adj.get("saturation", 0.0))
    return out


_render_impl = None


def render_global(img: np.ndarray, s: dict, scale: float = 1.0,
                  seed_key: str = "") -> np.ndarray:
    """Full global pipeline on float32 [0,1] image. Returns uint8 RGB.
    Automatically uses the fused numba fast path when available."""
    global _render_impl
    if _render_impl is None:
        try:
            from .fastpath import render_global_fast, available
            _render_impl = render_global_fast if available() else _legacy_render
        except Exception:
            _render_impl = _legacy_render
    return _render_impl(img, s, scale, seed_key)


def _legacy_render(img: np.ndarray, s: dict, scale: float = 1.0,
                   seed_key: str = "") -> np.ndarray:
    out = apply_lens_distortion(img, s.get("lens_distortion", 0.0)/100.0)
    out = remove_chromatic_aberration(out, s.get("ca_shift", 0.0))
    out = apply_wb(out, s["temp"], s["tint"])
    out = apply_calibration(out, s.get("cal_shadow_hue", 30.0),
                            s.get("cal_shadow_amt", 0.0),
                            s.get("cal_r", 0.0), s.get("cal_g", 0.0),
                            s.get("cal_b", 0.0))
    out = apply_exposure(out, s["exposure"])
    out = apply_contrast(out, s["contrast"])
    out = apply_highlights(out, s["highlights"])
    out = apply_shadows(out, s["shadows"])
    out = apply_whites(out, s["whites"])
    out = apply_blacks(out, s["blacks"])

    has_curve = any(len(s.get(k) or []) >= 2
                    for k in ("curve_rgb", "curve_r", "curve_g", "curve_b"))
    if has_curve:
        out = apply_curve_lut(out, s.get("curve_rgb"))
        out = apply_per_channel_curves(out, [s.get("curve_r"), s.get("curve_g"),
                                             s.get("curve_b")])

    out = apply_dehaze(out, s["dehaze"])
    out = apply_clarity(out, s["clarity"], scale)
    out = apply_vibrance(out, s["vibrance"])
    out = apply_saturation(out, s["saturation"])

    out = apply_hsl(out, s["hsl"])
    if s.get("bw"):
        out = apply_bw(out, s["hsl"])

    out = apply_color_grade(out, s["grade_shadows"], s["grade_midtones"],
                            s["grade_highlights"], s["grade_blender"],
                            s["grade_balance"])

    out = apply_relight(out, s.get("relight_angle", 300.0),
                        s.get("relight_strength", 0.0))
    sky_out = apply_sky_if_enabled(out, s)
    if sky_out is not None:
        out = sky_out

    out = apply_nr(out, s["nr_lum"], s["nr_color"])
    out = apply_sharpen(out, s["sharp_amount"], s["sharp_radius"], scale)
    out = apply_vignette(out, s["vignette_amount"], s["vignette_midpoint"],
                         s["vignette_feather"])
    out = apply_grain(out, s["grain_amount"], s["grain_size"], seed_key)

    # out is still float32 [0,1] here (vignette/grain preserve that)
    if hasattr(out, 'dtype') and out.dtype == np.uint8:
        out_f = out.astype(np.float32) / 255.0
    else:
        out_f = out.astype(np.float32)
    out_f = apply_glow(out_f, s.get("glow_amount", 0.0))
    out_u8 = np.clip(out_f * 255.0, 0, 255).astype(np.uint8)
    return apply_lut_if_enabled(out_u8, s)


def apply_mask_to_image(base_u8: np.ndarray, mask: np.ndarray,
                        adj: dict, invert: bool, scale: float) -> np.ndarray:
    """Blend locally-adjusted version into base according to mask weight."""
    w = mask
    if invert:
        w = 1.0 - w
    if float(w.max()) <= 0.002:
        return base_u8
    base = base_u8.astype(np.float32) / 255.0
    w = np.clip(w, 0.0, 1.0)
    # feather slightly for smoothness
    wf = gauss_blur_f(w, 1.0)
    w = np.maximum(w, wf * 0.85)
    edited = apply_adjustment_subset(base, adj, scale)
    w3 = w[..., None]
    out = base * (1.0 - w3) + edited * w3
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


# ------------------------------------------------------------------- masks raster

def rasterize_linear_mask(shape, params) -> np.ndarray:
    h, w = shape[:2]
    x1, y1, x2, y2 = params["x1"], params["y1"], params["x2"], params["y2"]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w, yy / h
    dx, dy = x2 - x1, y2 - y1
    len2 = dx * dx + dy * dy
    if len2 < 1e-9:
        return np.ones((h, w), dtype=np.float32)
    t = ((xn - x1) * dx + (yn - y1) * dy) / len2
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def rasterize_radial_mask(shape, params) -> np.ndarray:
    h, w = shape[:2]
    cx, cy = params["cx"], params["cy"]
    rx, ry = max(params["rx"], 0.02), max(params["ry"], 0.02)
    feather = params.get("feather", 0.35)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.sqrt(((xx / w - cx) / rx) ** 2 + ((yy / h - cy) / ry) ** 2)
    inner = 1.0 - feather
    return np.clip((inner + feather - d) / max(feather, 0.02), 0.0, 1.0).astype(np.float32)


def rasterize_brush_mask(shape, strokes) -> np.ndarray:
    """strokes: list of (points[(x,y,norm)], radius_norm, flow). Rasterize via PIL draw."""
    h, w = shape[:2]
    total = np.zeros((h, w), dtype=np.float32)
    from PIL import ImageDraw
    for pts, radius_norm, flow in strokes:
        layer = Image.new("F", (w, h), 0.0)
        draw = ImageDraw.Draw(layer)
        rpx = max(2, int(radius_norm * min(w, h)))
        px = [(p[0] * w, p[1] * h) for p in pts]
        if len(px) == 1:
            px.append((px[0][0] + 0.01, px[0][1]))
        draw.line(px, fill=float(flow), width=rpx * 2, joint="curve")
        for p in (px[0], px[-1]):
            draw.ellipse([p[0] - rpx, p[1] - rpx, p[0] + rpx, p[1] + rpx],
                         fill=float(flow))
        total = np.minimum(1.0, total + np.asarray(layer, dtype=np.float32))
    return total


def rasterize_mask(shape, mask_def: dict) -> np.ndarray:
    t = mask_def["type"]
    if t == "linear":
        return rasterize_linear_mask(shape, mask_def["params"])
    if t == "radial":
        return rasterize_radial_mask(shape, mask_def["params"])
    if t == "brush":
        return rasterize_brush_mask(shape, mask_def["params"].get("strokes", []))
    if t == "subject":
        arr = mask_def["params"].get("_subject_array")
        if arr is None:
            return np.zeros(shape[:2], dtype=np.float32)
        ih, iw = shape[:2]
        if arr.shape != (ih, iw):
            arr = np.asarray(Image.fromarray((arr * 255).astype(np.uint8), "L").resize(
                (iw, ih), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
        return arr
    return np.zeros(shape[:2], dtype=np.float32)


def settings_hash(s: dict) -> str:
    blob = repr(sorted((k, str(v)) for k, v in s.items() if k != "masks"))
    return hashlib.md5(blob.encode()).hexdigest()[:10]
