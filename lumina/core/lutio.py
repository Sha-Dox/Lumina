"""3D LUT (.cube) export of the current look + third-party LUT import."""
from __future__ import annotations

import os

import numpy as np


# ------------------------------------------------------------------ export

def export_cube(path: str, settings: dict, dim: int = 64) -> str:
    """Write current pointwise look as a .cube file."""
    from .fastpath import (pack_params, build_curves, _pointwise_kernel,
                           _KERNEL_LOCK)
    from .imaging import default_settings

    base = default_settings()
    s = dict(base)
    s.update(settings or {})
    # strip spatial/geometry so the cube is pure color
    for k in ("vignette_amount", "grain_amount"):
        s[k] = 0.0

    P = pack_params(s, dim*dim*dim, 1, (0.31, 0.77))
    curves = build_curves(s)
    noise = np.zeros((4, 4), dtype=np.float32)

    # lattice: r fastest, then g, then b (standard .cube order)
    lin = np.linspace(0.0, 1.0, dim, dtype=np.float32)
    bb, gg, rr = np.meshgrid(lin, lin, lin, indexing="ij")
    lattice = np.stack([rr.ravel(), gg.ravel(), bb.ravel()], axis=-1)
    img = lattice.reshape(-1, 1, 3).astype(np.float32)

    with _KERNEL_LOCK:
        out = _pointwise_kernel(img, P, curves, noise, np.zeros((4, 4), np.float32))
    vals = out.reshape(-1, 3).astype(np.float32)

    lines = [f"TITLE \"Lumina Look\"",
             f"LUT_3D_SIZE {dim}",
             "DOMAIN_MIN 0.0 0.0 0.0",
             "DOMAIN_MAX 1.0 1.0 1.0", ""]
    for v in vals:
        lines.append(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return path


# ------------------------------------------------------------------ import

_cache: dict[tuple, tuple] = {}


def parse_cube(path: str):
    """Returns (data float32 (dim,dim,dim,3), dim) for 3D cubes.
    1D cubes are expanded into a diagonal-ish 3D equivalent."""
    size_3d = None
    size_1d = None
    vals = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            t = line.strip()
            if not t or t.startswith("#") or t.upper().startswith("TITLE") \
                    or t.upper().startswith("DOMAIN"):
                continue
            u = t.upper()
            if u.startswith("LUT_3D_SIZE"):
                size_3d = int(float(t.split()[1]))
                continue
            if u.startswith("LUT_1D_SIZE"):
                size_1d = int(float(t.split()[1]))
                continue
            parts = t.split()
            if len(parts) == 3:
                try:
                    vals.append([float(p) for p in parts])
                except ValueError:
                    pass
    if size_3d and len(vals) >= size_3d ** 3:
        arr = np.asarray(vals[:size_3d**3], dtype=np.float32)
        # file order: r fastest
        cube = arr.reshape(size_3d, size_3d, size_3d, 3)   # [b][g][r]
        return np.ascontiguousarray(cube), size_3d
    if size_1d and len(vals) >= size_1d:
        row = np.asarray(vals[:size_1d], dtype=np.float32)   # (n,3)
        n = max(8, min(64, size_1d))
        xs = np.linspace(0, 1, n)
        interp = np.stack([np.interp(xs, np.linspace(0, 1, size_1d), row[:, c])
                           for c in range(3)], axis=-1).astype(np.float32)
        cube = np.empty((n, n, n, 3), dtype=np.float32)
        cube[:] = interp[::-1, None, None, :]                # b varies slowest
        return cube, n
    raise ValueError("unsupported or corrupt .cube file")


def load_cube(path: str):
    st = os.stat(path)
    key = (path, st.st_mtime_ns, st.st_size)
    hit = _cache.get(key)
    if hit is None:
        hit = parse_cube(path)
        if len(_cache) > 6:
            _cache.clear()
        _cache[key] = hit
    return hit


def apply_cube(img_f32: np.ndarray, cube: np.ndarray, dim: int) -> np.ndarray:
    """Trilinear-sampled 3D LUT, vectorized."""
    h, w = img_f32.shape[:2]
    px = np.clip(img_f32.reshape(-1, 3), 0.0, 1.0) * (dim - 1)
    i0 = np.floor(px).astype(np.int32)
    frac = px - i0
    i1 = np.minimum(i0 + 1, dim - 1)

    def idx(b, g, r):
        return (b * dim + g) * dim + r

    b_c, g_c, r_c = i0[:, 2], i0[:, 1], i0[:, 0]
    r_n, g_n, b_n = i1[:, 0], i1[:, 1], i1[:, 2]
    fr, fg, fb = frac[:, 0][:, None], frac[:, 1][:, None], frac[:, 2][:, None]

    c000 = cube[b_c, g_c, r_c]; c100 = cube[b_c, g_c, r_n]
    c010 = cube[b_c, g_n, r_c]; c110 = cube[b_c, g_n, r_n]
    c001 = cube[b_n, g_c, r_c]; c101 = cube[b_n, g_c, r_n]
    c011 = cube[b_n, g_n, r_c]; c111 = cube[b_n, g_n, r_n]

    c00 = c000*(1-fr) + c100*fr
    c10 = c010*(1-fr) + c110*fr
    c01 = c001*(1-fr) + c101*fr
    c11 = c011*(1-fr) + c111*fr
    c0 = c00*(1-fg) + c10*fg
    c1 = c01*(1-fg) + c11*fg
    out = c0*(1-fb) + c1*fb
    return np.clip(out.reshape(h, w, 3), 0.0, 1.0)
