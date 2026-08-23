"""Sky replacement: heuristic sky detection + procedural skies."""
from __future__ import annotations

import numpy as np
from PIL import Image as PILImage

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

from .imaging import gauss_blur_f

PRESETS = ["Golden Sunset", "Dramatic Storm", "Clear Blue",
           "Twilight Stars", "Pastel Dream"]

# (top rgb, bottom rgb, accent color, feature kind)
_PRESET_DEF = {
    "Golden Sunset": ((0.07, 0.09, 0.24), (1.00, 0.72, 0.38), (1.0, 0.80, 0.45), "sun"),
    "Dramatic Storm": ((0.13, 0.15, 0.19), (0.52, 0.56, 0.62), (0.9, 0.9, 0.95), "clouds"),
    "Clear Blue": ((0.16, 0.32, 0.60), (0.66, 0.82, 0.94), (0.9, 0.95, 1.0), "haze"),
    "Twilight Stars": ((0.02, 0.03, 0.08), (0.10, 0.14, 0.26), (1.0, 1.0, 1.0), "stars"),
    "Pastel Dream": ((0.72, 0.58, 0.86), (1.00, 0.82, 0.78), (1.0, 0.9, 0.9), "clouds"),
}


# ------------------------------------------------------------------ detection

def detect_sky_mask(img_f32: np.ndarray, offset: float = 0.0,
                    softness: float = 0.45,
                    window_frac: float = 0.055) -> np.ndarray:
    """Heuristic sky mask (1 = sky) for typical landscape photos.

    offset:   -1..1 shifts the detected horizon up/down (fraction of height)
    softness: 0..1 controls edge feathering
    """
    h, w = img_f32.shape[:2]
    r, g, b = img_f32[..., 0], img_f32[..., 1], img_f32[..., 2]
    mx = np.max(img_f32, axis=-1)
    mn = np.min(img_f32, axis=-1)
    sat = (mx - mn) / np.maximum(mx, 1e-4)

    bluish = (b - np.maximum(r, g)) > 0.04
    pale = (mx > 0.70) & (sat < 0.22)
    cand = bluish | pale

    # restrict search to upper region (+offset slack)
    limit = int(h * min(0.95, max(0.15, 0.68 + offset)))
    cand[limit:, :] = False
    cand[: max(1, int(h * 0.01)), :] = True   # very top is always sky-ish

    # gap-tolerant column scan via cumulative counts
    nonsky = (~cand).astype(np.float32)
    win = max(3, int(h * window_frac))
    csum = np.cumsum(nonsky, axis=0)
    csum_shift = np.vstack([np.zeros((win, w), np.float32), csum[:-win]])
    ratio = csum_shift / win                       # nonsky ratio in window ABOVE+AT row

    # a row belongs to sky while the running nonsky ratio stays low
    boundary = (ratio > 0.55).astype(np.float32)
    # first ground row per column
    first = np.argmax(boundary > 0, axis=0)
    any_ground = boundary.max(axis=0) > 0
    first[~any_ground] = h

    col_limit = np.clip(first + int(offset * h * 0.25), 1, h)
    rows = np.arange(h, dtype=np.float32)[:, None]
    mask = (rows < col_limit[None, :]).astype(np.float32)

    # feather
    sigma = max(1.0, softness * h * 0.012)
    if _HAS_CV2:
        k = int(sigma * 4) | 1
        mask = cv2.GaussianBlur(mask, (k, k), sigmaX=sigma)
    else:
        mask = gauss_blur_f(mask, sigma)
    return np.clip(mask, 0.0, 1.0)


# ------------------------------------------------------------------ skies

def _fbm(w: int, h: int, seed: int, octaves=(6, 14, 30)) -> np.ndarray:
    """Cheap fractal noise via repeated random blur (values ~0..1)."""
    rng = np.random.default_rng(seed)
    acc = np.zeros((h, w), np.float32)
    amp_total = 0.0
    amp = 1.0
    for o in octaves:
        gw = max(2, w // o)
        gh = max(2, h // o)
        layer = rng.standard_normal((gh, gw)).astype(np.float32)
        if _HAS_CV2:
            layer = cv2.resize(layer, (w, h), interpolation=cv2.INTER_CUBIC)
            layer = cv2.GaussianBlur(layer, (0, 0), o * 0.35)
        else:
            layer = np.asarray(
                PILImage.fromarray(layer.astype(np.float32), mode="F")
                .resize((w, h), PILImage.Resampling.BILINEAR),
                dtype=np.float32)
            layer = gauss_blur_f(layer, o * 0.35)
        mn, mx_ = float(layer.min()), float(layer.max())
        if mx_ > mn:
            layer = (layer - mn) / (mx_ - mn)
        acc += layer * amp
        amp_total += amp
        amp *= 0.55
    acc /= max(amp_total, 1e-6)
    return acc


def generate_sky(preset: str, w: int, h: int, seed: int = 7) -> np.ndarray:
    top, bot, accent, feat = _PRESET_DEF.get(preset, _PRESET_DEF["Clear Blue"])
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]

    tt = np.clip(yy * 1.15, 0, 1)[..., None]
    grad = (np.array(top, np.float32)[None, None, :] * (1 - tt) +
            np.array(bot, np.float32)[None, None, :] * tt)
    sky = np.repeat(grad, w, axis=1)

    clouds = _fbm(w, h, seed + 11)
    cloud_amt = 0.18
    if preset == "Dramatic Storm":
        cloud_amt = 0.65
    elif preset == "Golden Sunset":
        cloud_amt = 0.38
    elif preset == "Pastel Dream":
        cloud_amt = 0.30
    elif preset == "Clear Blue":
        cloud_amt = 0.10
    # fade clouds toward horizon
    fade1 = np.clip(1.0 - yy[:, 0] * 1.4, 0, 1)           # (h,)
    lum = (clouds - 0.5) * cloud_amt * fade1[:, None]     # (h,w)
    sky *= (1.0 + lum[..., None] * 1.6)

    if feat == "sun":
        sx, sy = 0.68, 0.42
        d = np.sqrt(((np.arange(w)[None,:] - w*sx)/w)**2 +
                    ((np.arange(h)[:,None] - h*sy)/h)**2)
        glow = np.exp(-(d/0.22)**2).astype(np.float32)
        sky += glow[..., None] * np.array([1.0, 0.72, 0.35], np.float32) * 0.9
    elif feat == "stars":
        rng = np.random.default_rng(seed + 99)
        stars = (rng.random((h//3, w//3)) > 0.9985).astype(np.float32)
        st_big = np.asarray(
            PILImage.fromarray((stars*255).astype(np.uint8))
            .resize((w, h), PILImage.Resampling.BILINEAR),
            dtype=np.float32) / 255.0
        st_big *= (1.0 - yy * 2.2).clip(0, 1)      # only high sky
        sky += st_big[:, :, None] * 0.9

    return np.clip(sky, 0.0, 1.0).astype(np.float32)


_sky_cache: dict[tuple, np.ndarray] = {}


def cached_sky(preset: str, w: int, h: int) -> np.ndarray:
    key = (preset, w, h)
    m = _sky_cache.get(key)
    if m is None:
        m = generate_sky(preset, w, h)
        if len(_sky_cache) > 6:
            _sky_cache.clear()
        _sky_cache[key] = m
    return m


def replace_sky(img_f32: np.ndarray, mask01: np.ndarray, preset: str,
                strength: float = 0.7, light_wrap: bool = True) -> np.ndarray:
    """img/mask float32; strength 0..1. Returns composited float image."""
    h, w = img_f32.shape[:2]
    sky = cached_sky(preset, w, h)
    m = (mask01 * min(1.0, max(0.0, strength)))[..., None]
    out = img_f32 * (1.0 - m) + sky * m
    if light_wrap and strength > 0.05:
        wrap_m = gauss_blur_f(mask01, max(4.0, h * 0.02))
        avg = float(np.mean(sky)) 
        tint = np.array([avg*1.05, avg*1.0, avg*0.95], np.float32)
        edge = (wrap_m * (1.0 - mask01))[..., None]
        out = out + (tint[None,None,:] - out) * (edge * 0.30 * strength)
    return np.clip(out, 0.0, 1.0)
