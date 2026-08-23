"""Fast-path renderer: the entire pointwise adjustment chain fused into a
single numba-parallel kernel, plus OpenCV-accelerated spatial filters.

Falls back transparently to the legacy numpy path if numba/cv2 are absent.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    cv2 = None
    _HAS_CV2 = False

try:
    import threading as _threading
    from numba import njit, prange
    from math import exp
    # Workqueue layer is not thread-safe; serialize kernel entry app-side.
    _KERNEL_LOCK = _threading.Lock()
    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# ------------------------------------------------------------------ packing

NPARAMS = 64

BANDS = ["red", "orange", "yellow", "green",
         "aqua", "blue", "purple", "magenta"]
# (lo, span, falloff)
BAND_GEOM = np.array([
    [345.0, 30.0, 22.0], [12.0, 34.0, 18.0], [38.0, 34.0, 18.0],
    [58.0, 107.0, 30.0], [150.0, 55.0, 22.0], [192.0, 73.0, 24.0],
    [252.0, 53.0, 22.0], [292.0, 58.0, 22.0]], dtype=np.float64)


def pack_params(s: dict, w: int, h: int, seed_off=(0.31, 0.77)) -> np.ndarray:
    p = np.zeros(NPARAMS, dtype=np.float64)
    p[0] = s.get("temp", 0.0) / 100.0
    p[1] = s.get("tint", 0.0) / 100.0
    p[2] = s.get("exposure", 0.0)
    p[3] = s.get("contrast", 0.0) / 100.0
    p[4] = s.get("highlights", 0.0) / 100.0
    p[5] = s.get("shadows", 0.0) / 100.0
    p[6] = s.get("whites", 0.0) / 100.0
    p[7] = s.get("blacks", 0.0) / 100.0
    p[8] = s.get("vibrance", 0.0) / 100.0
    p[9] = s.get("saturation", 0.0) / 100.0
    p[10] = 1.0 if s.get("bw") else 0.0
    hsl = s.get("hsl") or {}
    wr, wg, wb_ = 0.2126, 0.7152, 0.0722
    for bi, name in enumerate(BANDS):
        v = hsl.get(name, (0.0, 0.0, 0.0))
        p[11 + bi*3 + 0] = v[0] / 100.0
        p[11 + bi*3 + 1] = v[1] / 100.0
        p[11 + bi*3 + 2] = v[2] / 100.0
        dl = v[2]
        if name == "red":
            wr *= 1 + dl/130.0
        elif name in ("green", "yellow"):
            wg *= 1 + dl/(130.0 if name == "green" else 220.0)
        elif name in ("blue", "aqua"):
            wb_ *= 1 + dl/(110.0 if name == "blue" else 190.0)
    tot = wr + wg + wb_
    p[58] = wr/tot; p[59] = wg/tot; p[60] = wb_/tot
    gs = s.get("grade_shadows", [0, 0, 0])
    gm = s.get("grade_midtones", [0, 0, 0])
    gh = s.get("grade_highlights", [0, 0, 0])
    p[37] = gs[0]; p[38] = gs[1]/100; p[39] = gs[2]/100
    p[40] = gm[0]; p[41] = gm[1]/100; p[42] = gm[2]/100
    p[43] = gh[0]; p[44] = gh[1]/100; p[45] = gh[2]/100
    p[46] = s.get("blender", 50)/100.0
    p[47] = s.get("balance", 0)/100.0
    p[48] = s.get("vignette_amount", 0)/100.0
    p[49] = max(0.02, s.get("vignette_midpoint", 50)/100.0)
    p[50] = max(0.02, s.get("vignette_feather", 60)/100.0)
    p[51] = s.get("grain_amount", 0)/100.0
    p[52], p[53] = seed_off
    p[54] = w; p[55] = h
    return p


def build_curves(s: dict, n: int = 1024) -> np.ndarray:
    """(4, n) float32: composite + R + G + B response LUTs."""
    from .imaging import monotonic_spline
    out = np.zeros((4, n), dtype=np.float32)
    xs = np.linspace(0.0, 1.0, n)

    def ev(pts):
        pts = pts or []
        if len(pts) >= 2:
            _, ys = monotonic_spline([tuple(p) for p in pts], n=n)
            return ys.astype(np.float32)
        return xs.astype(np.float32)

    comp = ev(s.get("curve_rgb"))
    rc, gc, bc = ev(s.get("curve_r")), ev(s.get("curve_g")), ev(s.get("curve_b"))
    # channel curves applied after composite (same order as CPU path)
    comp2 = comp.copy()
    comp2 = np.interp(comp2, xs, gc * 0 + bc * 0 + rc)  # placeholder no-op
    # apply each channel curve to its own axis of the composite output
    lut = np.stack([comp, rc, gc, bc]).astype(np.float32)
    return lut


# ------------------------------------------------------------------ kernel

if _HAS_NUMBA:
    @njit(parallel=True, cache=True)
    def _pointwise_kernel(img, P, curves, noise, grain):
        H, W = img.shape[0], img.shape[1]
        out = np.empty_like(img)
        temp = P[0]; tint = P[1]; expo = P[2]; con = P[3]
        hi = P[4]; sh = P[5]; wh = P[6]; bl = P[7]
        vib = P[8]; sat = P[9]
        is_bw = P[10] > 0.5
        bwr = P[58]; bwg = P[59]; bwb = P[60]
        blender = P[46]; balance = P[47]
        vigA = P[48]; vigM = P[49]; vigF = P[50]
        grainA = P[51]; sx = P[52]; sy = P[53]

        exp_fac = 2.0 ** expo
        inv22 = 1.0 / 2.2
        gH = grain.shape[0]
        gW = grain.shape[1]

        for i in prange(H):
            yn = i / H
            for j in range(W):
                xn = j / W
                r = img[i, j, 0]
                g = img[i, j, 1]
                b = img[i, j, 2]

                # ---- white balance
                r = min(1.0, max(0.0, r * (1.0 + 0.28 * temp)))
                g = min(1.0, max(0.0, g * (1.0 - 0.14 * tint)))
                b = min(1.0, max(0.0, b * (1.0 - 0.28 * temp)))

                # ---- exposure (linear light roundtrip)
                r = min(1.0, r) ** 2.2 * exp_fac
                g = min(1.0, g) ** 2.2 * exp_fac
                b = min(1.0, b) ** 2.2 * exp_fac
                r = min(r, 1.0) ** inv22
                g = min(g, 1.0) ** inv22
                b = min(b, 1.0) ** inv22

                L = 0.2126*r + 0.7152*g + 0.0722*b

                # ---- contrast
                if con >= 0.0:
                    c = con
                    k = 1.0 + 6.0*c
                    sig = 1.0 / (1.0 + exp(-(L-0.5)*k))
                    r = r*(1-c) + sig*c
                    g = g*(1-c) + sig*c
                    b = b*(1-c) + sig*c
                else:
                    f = 1.0 + con*0.85
                    r = 0.5 + (r-0.5)*f
                    g = 0.5 + (g-0.5)*f
                    b = 0.5 + (b-0.5)*f

                L = 0.2126*r + 0.7152*g + 0.0722*b

                # ---- smoothstep helper inlined per region weight
                # highlights
                t = (L-0.35)/0.60
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                wH = (t*t*(3-2*t)) ** 1.2
                if hi < 0.0:
                    d = hi * 0.75 * wH
                    r += d*(r - L*0.6); g += d*(g - L*0.6); b += d*(b - L*0.6)
                else:
                    d = hi * 0.55 * wH
                    r += d*(1.0-r); g += d*(1.0-g); b += d*(1.0-b)

                L = 0.2126*r + 0.7152*g + 0.0722*b
                # shadows
                t = (L-0.0)/0.55
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                wS = (1 - (t*t*(3-2*t))) ** 1.4
                if sh > 0.0:
                    d = sh * 0.62 * wS
                    r += d*(1.0-r); g += d*(1.0-g); b += d*(1.0-b)
                else:
                    d = sh * 0.55 * wS
                    r += d*r; g += d*g; b += d*b

                L = 0.2126*r + 0.7152*g + 0.0722*b
                # whites
                t = (L-0.45)/0.55
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                wWt = (t*t*(3-2*t)) ** 2.0
                if wh > 0.0:
                    d = wh * 0.45 * wWt
                    r += d*(1.0-r); g += d*(1.0-g); b += d*(1.0-b)
                else:
                    d = wh * 0.40 * wWt
                    r += d*r; g += d*g; b += d*b

                L = 0.2126*r + 0.7152*g + 0.0722*b
                # blacks
                t = (L-0.02)/0.40
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                wB = (1 - (t*t*(3-2*t))) ** 2.0
                if bl < 0.0:
                    d = bl * 0.40 * wB
                    r += d*r; g += d*g; b += d*b
                else:
                    d = bl * 0.42 * wB
                    dd0 = 0.35 - r
                    if dd0 > 0.35: dd0 = 0.35
                    elif dd0 < -0.35: dd0 = -0.35
                    dd1 = 0.35 - g
                    if dd1 > 0.35: dd1 = 0.35
                    elif dd1 < -0.35: dd1 = -0.35
                    dd2 = 0.35 - b
                    if dd2 > 0.35: dd2 = 0.35
                    elif dd2 < -0.35: dd2 = -0.35
                    r += d*dd0*1.4; g += d*dd1*1.4; b += d*dd2*1.4

                # ---- tone curves: composite LUT per channel, then each
                #      channel's own LUT sampled at its own composited value
                ci = r * 1023.0
                if ci < 0.0: ci = 0.0
                elif ci > 1023.0: ci = 1023.0
                i0 = int(ci); fr = ci - i0
                i1 = i0 + 1 if i0 < 1023 else 1023

                gj = g * 1023.0
                if gj < 0.0: gj = 0.0
                elif gj > 1023.0: gj = 1023.0
                k0 = int(gj); fg = gj - k0
                k1 = k0 + 1 if k0 < 1023 else 1023

                bj = b * 1023.0
                if bj < 0.0: bj = 0.0
                elif bj > 1023.0: bj = 1023.0
                m0 = int(bj); fb = bj - m0
                m1 = m0 + 1 if m0 < 1023 else 1023

                cr = curves[0, i0]*(1-fr) + curves[0, i1]*fr
                cg_ = curves[0, k0]*(1-fg) + curves[0, k1]*fg
                cb = curves[0, m0]*(1-fb) + curves[0, m1]*fb

                ri = cr * 1023.0
                if ri < 0.0: ri = 0.0
                elif ri > 1023.0: ri = 1023.0
                n0 = int(ri); fnr = ri - n0
                n1 = n0 + 1 if n0 < 1023 else 1023

                gi2 = cg_ * 1023.0
                if gi2 < 0.0: gi2 = 0.0
                elif gi2 > 1023.0: gi2 = 1023.0
                p0 = int(gi2); fgn = gi2 - p0
                p1 = p0 + 1 if p0 < 1023 else 1023

                bi2 = cb * 1023.0
                if bi2 < 0.0: bi2 = 0.0
                elif bi2 > 1023.0: bi2 = 1023.0
                q0 = int(bi2); fbn = bi2 - q0
                q1 = q0 + 1 if q0 < 1023 else 1023

                r = curves[1, n0]*(1-fnr) + curves[1, n1]*fnr
                g = curves[2, p0]*(1-fgn) + curves[2, p1]*fgn
                b = curves[3, q0]*(1-fbn) + curves[3, q1]*fbn

                # ---- vibrance (hsv s scale) + saturation
                mx = max(r, g, b)
                mn = min(r, g, b)
                d0 = mx - mn
                if mx > 1e-6 and d0 > 1e-6:
                    sv = d0 / mx
                    k = min(1.0, sv * (1.0 + vib*(1.0-sv)*1.1)) / sv
                    r = mx - (mx-r)*k
                    g = mx - (mx-g)*k
                    b = mx - (mx-b)*k
                Lc = 0.2126*r + 0.7152*g + 0.0722*b
                sf2 = 1.0 + sat
                r = min(1.0, max(0.0, Lc + (r-Lc)*sf2))
                g = min(1.0, max(0.0, Lc + (g-Lc)*sf2))
                b = min(1.0, max(0.0, Lc + (b-Lc)*sf2))

                # ---- HSL mixer (runs before B&W, matching legacy order)
                if True:
                    mx = max(r, g, b); mn = min(r, g, b); df = mx-mn
                    hue = 0.0; sv = 0.0
                    if df > 0.004 and mx > 0.004:
                        sv = df/mx
                        if mx == r:
                            hue = (60.0*((g-b)/df)) % 360.0
                        elif mx == g:
                            hue = 60.0*((b-r)/df) + 120.0
                        else:
                            hue = 60.0*((r-g)/df) + 240.0
                        hueShift = 0.0; satMul = 1.0; lumAdd = 0.0
                        for bi in range(8):
                            dh = P[11+bi*3]; ds = P[12+bi*3]; dl = P[13+bi*3]
                            if dh == 0.0 and ds == 0.0 and dl == 0.0:
                                continue
                            lo = 345.0; span = 30.0; fall = 22.0
                            if bi == 1: lo = 12.0; span = 34.0; fall = 18.0
                            elif bi == 2: lo = 38.0; span = 34.0; fall = 18.0
                            elif bi == 3: lo = 58.0; span = 107.0; fall = 30.0
                            elif bi == 4: lo = 150.0; span = 55.0; fall = 22.0
                            elif bi == 5: lo = 192.0; span = 73.0; fall = 24.0
                            elif bi == 6: lo = 252.0; span = 53.0; fall = 22.0
                            elif bi == 7: lo = 292.0; span = 58.0; fall = 22.0
                            dd = (hue - lo) % 360.0
                            tt = (dd + fall*0.5)/(fall + 0.001)
                            if tt < 0.0: tt = 0.0
                            elif tt > 1.0: tt = 1.0
                            rise = tt*tt*(3-2*tt)
                            t1b = span - fall*0.5
                            t2b = span + fall*0.5 + 0.001
                            tt = (dd - t1b)/(t2b - t1b)
                            if tt < 0.0: tt = 0.0
                            elif tt > 1.0: tt = 1.0
                            decay = 1.0 - tt*tt*(3-2*tt)
                            inside = rise*decay
                            ct0 = -fall*0.15
                            ct1 = fall*0.15
                            ct = (dd - ct0)/(ct1 - ct0)
                            if ct < 0.0: ct = 0.0
                            elif ct > 1.0: ct = 1.0
                            core_rise = ct*ct*(3-2*ct)
                            ct2 = span + fall*0.15
                            ct = (dd - (span - fall*0.15))/(ct2 - (span - fall*0.15))
                            if ct < 0.0: ct = 0.0
                            elif ct > 1.0: ct = 1.0
                            core_decay = 1.0 - ct*ct*(3-2*ct)
                            core = core_rise*core_decay
                            wgt = inside*0.55
                            if core > wgt: wgt = core
                            hueShift += dh*32.0*wgt
                            satMul *= 1.0 + ds*1.15*wgt
                            lumAdd += dl*0.16*wgt
                        if hueShift != 0.0 or satMul != 1.0 or lumAdd != 0.0:
                            sv = min(1.0, sv * satMul)
                            # rotate rgb by hue shift via hsv->rgb on shifted h
                            hh = (hue + hueShift) % 360.0 / 60.0
                            ii = int(hh) % 6
                            ff = hh - int(hh)
                            vv = mx
                            pp = vv*(1-sv); qq = vv*(1-sv*ff); tt2 = vv*(1-sv*(1-ff))
                            if ii == 0: rn, gn, bn = vv, tt2, pp
                            elif ii == 1: rn, gn, bn = qq, vv, pp
                            elif ii == 2: rn, gn, bn = pp, vv, tt2
                            elif ii == 3: rn, gn, bn = pp, qq, vv
                            elif ii == 4: rn, gn, bn = tt2, pp, vv
                            else: rn, gn, bn = vv, pp, qq
                            r = min(1.0, max(0.0, rn + lumAdd))
                            g = min(1.0, max(0.0, gn + lumAdd))
                            b = min(1.0, max(0.0, bn + lumAdd))
                            if sv <= 0.004:
                                r += lumAdd; g += lumAdd; b += lumAdd

                # ---- B&W
                if is_bw:
                    gray = r*bwr + g*bwg + b*bwb
                    r = gray; g = gray; b = gray
                else:
                    if r < 0.0: r = 0.0
                    elif r > 1.0: r = 1.0
                    if g < 0.0: g = 0.0
                    elif g > 1.0: g = 1.0
                    if b < 0.0: b = 0.0
                    elif b > 1.0: b = 1.0

                L = 0.2126*r + 0.7152*g + 0.0722*b

                # ---- color grading wheels
                bal = balance
                center = 0.5 + bal*0.25
                spread = 0.28 + blender*0.34
                t = (L-(center-spread*0.9))/(spread*1.9)
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                wsh = (1 - (t*t*(3-2*t))) ** 1.3
                t = (L-(center-spread))/(spread*1.9)
                if t < 0.0: t = 0.0
                elif t > 1.0: t = 1.0
                whi = (t*t*(3-2*t)) ** 1.3
                sg = spread*0.62
                wm = exp(-((L-center)*(L-center))/(2*sg*sg)) ** 1.2

                for wi in range(3):
                    if wi == 0:
                        hueW = P[37]; satW = P[38]; lumW = P[39]; wmap = wsh
                        mode = 0
                    elif wi == 1:
                        hueW = P[40]; satW = P[41]; lumW = P[42]; wmap = wm
                        mode = 1
                    else:
                        hueW = P[43]; satW = P[44]; lumW = P[45]; wmap = whi
                        mode = 2
                    if satW < 0.005 and abs(lumW) < 0.005:
                        continue
                    st = satW * 0.42
                    # tint color from hue/sat
                    th = (hueW % 360.0) / 60.0
                    ti6 = int(th) % 6
                    tf = th - int(th)
                    tv = 1.0
                    tsat = min(1.0, satW*1.6)
                    tp = tv*(1-tsat); tq = tv*(1-tsat*tf); tw = tv*(1-tsat*(1-tf))
                    if ti6 == 0: tr, tg, tb = tv, tw, tp
                    elif ti6 == 1: tr, tg, tb = tq, tv, tp
                    elif ti6 == 2: tr, tg, tb = pp, tv, tw
                    elif ti6 == 3: tr, tg, tb = tp, tq, tv
                    elif ti6 == 4: tr, tg, tb = tw, tp, tv
                    else: tr, tg, tb = tv, tp, tq
                    if mode == 0:
                        k0 = st*0.85
                        rr = r*k0*0.0 + (r*(1-k0) + tr*k0*r*1.7)
                        gg = g*(1-k0) + tg*k0*g*1.7
                        bb = b*(1-k0) + tb*k0*b*1.7
                        if rr > 1.0: rr = 1.0
                        if gg > 1.0: gg = 1.0
                        if bb > 1.0: bb = 1.0
                        r = r*(1-st) + rr*st
                        g = g*(1-st) + gg*st
                        b = b*(1-st) + bb*st
                    elif mode == 1:
                        bell = (r*(1-r)*4 + g*(1-g)*4 + b*(1-b)*4)/3
                        rr = r + (tr-0.5)*st*bell
                        gg = g + (tg-0.5)*st*bell
                        bb = b + (tb-0.5)*st*bell
                        if rr > 1.0: rr = 1.0
                        elif rr < 0.0: rr = 0.0
                        if gg > 1.0: gg = 1.0
                        elif gg < 0.0: gg = 0.0
                        if bb > 1.0: bb = 1.0
                        elif bb < 0.0: bb = 0.0
                        r = r*(1-st) + rr*st
                        g = g*(1-st) + gg*st
                        b = b*(1-st) + bb*st
                    else:
                        rr = r + (tr-0.5)*st*r*1.6
                        gg = g + (tg-0.5)*st*g*1.6
                        bb = b + (tb-0.5)*st*b*1.6
                        if rr > 1.0: rr = 1.0
                        elif rr < 0.0: rr = 0.0
                        if gg > 1.0: gg = 1.0
                        elif gg < 0.0: gg = 0.0
                        if bb > 1.0: bb = 1.0
                        elif bb < 0.0: bb = 0.0
                        r = r*(1-st) + rr*st
                        g = g*(1-st) + gg*st
                        b = b*(1-st) + bb*st
                    if abs(lumW) > 0.005:
                        add = lumW*0.28*wmap
                        r += add; g += add; b += add

                # ---- vignette
                if abs(vigA) > 0.001:
                    dxn = (xn - 0.5) * 2.0
                    dyn = (yn - 0.5) * 2.0
                    dist = (dxn*dxn + dyn*dyn) ** 0.5 / 1.41421356
                    end = vigM + (1.0 - vigM)*vigF
                    t = (dist - vigM)/(end - vigM)
                    if t < 0.0: t = 0.0
                    elif t > 1.0: t = 1.0
                    fall = t*t*(3-2*t)
                    fall = fall*fall
                    mul = 1.0 + vigA*1.15*fall
                    r *= mul; g *= mul; b *= mul

                # ---- grain
                if grainA > 0.001:
                    nz = grain[i, j]
                    add = nz * grainA * 0.16
                    r += add; g += add; b += add

                if r < 0.0: r = 0.0
                elif r > 1.0: r = 1.0
                if g < 0.0: g = 0.0
                elif g > 1.0: g = 1.0
                if b < 0.0: b = 0.0
                elif b > 1.0: b = 1.0

                out[i, j, 0] = r
                out[i, j, 1] = g
                out[i, j, 2] = b

        return out
else:
    _pointwise_kernel = None


# ------------------------------------------------------------------ noise

_noise_cache: dict[tuple, np.ndarray] = {}


def _noise_grid(seed_key: str) -> np.ndarray:
    key = hash(seed_key) & 0xFFFF
    n = _noise_cache.get(key)
    if n is None:
        rng = np.random.default_rng(key or 12345)
        small = rng.standard_normal((256, 256)).astype(np.float32)
        if _HAS_CV2:
            small = cv2.GaussianBlur(small, (0, 0), 0.7)
        n = small.astype(np.float32)
        if len(_noise_cache) > 8:
            _noise_cache.clear()
        _noise_cache[key] = n
    return n


def seed_offsets(seed_key: str):
    h = abs(hash(seed_key)) % 9973
    return ((h % 331) / 331.0 * 2.7, (h % 617) / 617.0 * 2.9)


# ------------------------------------------------------------------ blurs

def blur_f(arr: np.ndarray, radius: float) -> np.ndarray:
    """Fast gaussian via cv2; sigma≈radius to match legacy visually."""
    if not _HAS_CV2 or radius <= 0.02:
        from .imaging import gauss_blur_f
        return gauss_blur_f(arr, radius)
    return cv2.GaussianBlur(arr, (0, 0), radius)


# ------------------------------------------------------------------ render

def available() -> bool:
    return _HAS_NUMBA and _pointwise_kernel is not None


_grain_map_cache: dict[tuple, np.ndarray] = {}


def grain_map(w: int, h: int, size: float, seed_key: str) -> np.ndarray:
    """Legacy-exact grain field: coarse randn -> blur -> bilinear upscale."""
    import math
    cell = max(1.0, (size/100.0)*3.2)
    gh, gw = max(2, int(h/cell)), max(2, int(w/cell))
    key = (gh, gw, seed_key)
    m = _grain_map_cache.get(key)
    if m is None:
        rng = np.random.default_rng(abs(hash(seed_key)) % (2**32))
        small = rng.standard_normal((gh, gw)).astype(np.float32)
        from .imaging import gauss_blur_f
        small = gauss_blur_f(small, 0.6)
        from PIL import Image as PILImage
        big = np.asarray(
            PILImage.fromarray(small, mode="F").resize(
                (w, h), PILImage.Resampling.BILINEAR), dtype=np.float32)
        if len(_grain_map_cache) > 4:
            _grain_map_cache.clear()
        _grain_map_cache[key] = big
        m = big
    return m


_vig_cache: dict[tuple, np.ndarray] = {}


def _vignette_map(w: int, h: int, midpoint: float, feather: float) -> np.ndarray:
    key = (w, h, round(midpoint, 2), round(feather, 2))
    m = _vig_cache.get(key)
    if m is None:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.sqrt(((xx - w/2)/(w/2))**2 + ((yy - h/2)/(h/2))**2) / 1.41421356
        m = max(0.02, midpoint / 100.0)
        f = max(0.02, feather / 100.0)
        end = m + (1.0 - m)*f
        t = np.clip((d - m)/max(end - m, 1e-6), 0.0, 1.0)
        m = (t*t*(3-2*t))**2
        if len(_vig_cache) > 12:
            _vig_cache.clear()
        _vig_cache[key] = m
    return m


def render_global_fast(img: np.ndarray, s: dict, scale: float = 1.0,
                       seed_key: str = "") -> np.ndarray:
    """Drop-in replacement for imaging.render_global using the fused kernel."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    P = pack_params(s, w, h, seed_offsets(seed_key))
    curves = build_curves(s)
    noise = np.zeros((4, 4), dtype=np.float32)
    gm = grain_map(w, h, s.get("grain_size", 25.0), seed_key)

    # vignette & grain are applied AFTER spatial ops (legacy order), so the
    # kernel gets them zeroed here.
    P[48] = 0.0
    P[51] = 0.0
    with _KERNEL_LOCK:
        out_f = _pointwise_kernel(img, P, curves, noise, gm)

    from .imaging import apply_relight
    out_f = apply_relight(out_f, s.get("relight_angle", 300.0),
                          s.get("relight_strength", 0.0))
    if s.get("sky_enabled"):
        from . import sky as _skymod
        m = _skymod.detect_sky_mask(out_f, offset=s.get("sky_offset", 0.0),
                                    softness=s.get("sky_softness", 45.0))
        out_f = _skymod.replace_sky(
            out_f, m, s.get("sky_preset", "Golden Sunset"),
            strength=min(1.0, s.get("sky_strength", 75.0)/100.0))

    # ---- spatial ops (cv2 accelerated where possible)
    dh = s.get("dehaze", 0.0)
    if abs(dh) > 0.01:
        import math as _m
        a = dh / 100.0
        radius_dh = max(3.0, w/180.0)
        if _HAS_CV2:
            blurred3 = cv2.GaussianBlur(out_f, (0, 0), radius_dh)
            dark = np.min(blurred3, axis=-1)
        else:
            from .imaging import gauss_blur_f
            dark = np.min(gauss_blur_f(out_f, radius_dh), axis=-1)
        if a > 0:
            omega = min(0.92, 0.65*a + 0.10)
            flat_dark = dark.reshape(-1)
            n_pick = max(64, int(flat_dark.size * 0.001))
            idx = np.argpartition(flat_dark, -n_pick)[-n_pick:]
            A = np.percentile(out_f.reshape(-1, 3)[idx], 90,
                              axis=0).astype(np.float32)
            A = np.clip(A, 0.08, 0.98)
            t = np.clip(1.0 - omega * (dark / float(np.mean(A))), 0.18, 1.0)
            out_f = np.clip((out_f - (1.0 - t[..., None]) *
                             A[None, None, :]) / t[..., None], 0.0, 1.0)
        else:
            k = -a
            veil = np.array([0.62, 0.66, 0.72], dtype=np.float32)
            out_f = np.clip(out_f*(1.0 - k*0.55) +
                            veil[None, None, :]*(k*0.55), 0.0, 1.0)

    cl = s.get("clarity", 0.0)
    if abs(cl) > 0.01:
        a = cl / 100.0
        L = out_f[..., 0]*0.2126 + out_f[..., 1]*0.7152 + out_f[..., 2]*0.0722
        radius = max(2.0, 14.0*scale)
        Lb = blur_f(L, radius)
        detail = L - Lb
        midw = 1.0 - np.abs(2.0*smoothstep_arr(L) - 1.0)
        midw = midw ** 0.7
        delta = (a*0.9) * detail * midw
        if a > 0:
            out_f = out_f + delta[..., None] * (out_f/(L[..., None]+1e-4))
        else:
            out_f = out_f + delta[..., None]

    sa = s.get("sharp_amount", 0.0)
    if sa >= 0.5:
        a = min(sa, 150.0)/100.0
        L = out_f[..., 0]*0.2126 + out_f[..., 1]*0.7152 + out_f[..., 2]*0.0722
        r = max(0.3, s.get("sharp_radius", 1.2)*scale)
        Lb = blur_f(L, r)
        out_f = out_f + ((a*1.35)*(L-Lb))[..., None]

    nrl = s.get("nr_lum", 0.0)
    if nrl > 0.5:
        a = min(nrl, 100.0)/100.0
        L = out_f[..., 0]*0.2126 + out_f[..., 1]*0.7152 + out_f[..., 2]*0.0722
        Lb = blur_f(L, 0.8 + 2.2*a)
        diff = np.abs(L - Lb)
        tt = np.clip((diff - 0.015)/0.075, 0.0, 1.0)
        keep = tt*tt*(3-2*tt)
        out_f = out_f + ((Lb + (L-Lb)*keep) - L)[..., None]

    nrc = s.get("nr_color", 0.0)
    if nrc > 0.5:
        a = min(nrc, 100.0)/100.0
        L = out_f[..., 0]*0.2126 + out_f[..., 1]*0.7152 + out_f[..., 2]*0.0722
        C = out_f - L[..., None]
        Cb = blur_f(C, 1.5 + 4.0*a)
        out_f = np.clip(L[..., None] + Cb, 0.0, 1.0)

    # ---- effects last (legacy ordering)
    vigA = s.get("vignette_amount", 0.0)
    if abs(vigA) > 0.01:
        vm = _vignette_map(w, h, s.get("vignette_midpoint", 50.0),
                           s.get("vignette_feather", 60.0))
        out_f = out_f * (1.0 + (vigA/100.0)*1.15*vm[..., None])

    ga = s.get("grain_amount", 0.0)
    if ga > 0.5:
        out_f = out_f + (gm * (min(ga, 100.0)/100.0)*0.16)[..., None]

    out_f = np.clip(out_f, 0.0, 1.0)
    return (out_f * 255.0).astype(np.uint8)


def smoothstep_arr(x):
    t = np.clip(x, 0.0, 1.0)
    return t*t*(3-2*t)
