"""HDR merge (Mertens exposure fusion) and panorama stitching via OpenCV."""
from __future__ import annotations

import os

import numpy as np


def _decode_rgb(path: str, long_edge: int = 2400) -> np.ndarray:
    from . import rawio
    arr = rawio.decode_preview(path, long_edge)
    return arr


def _align_to_ref(bgr_f32: np.ndarray, ref_gray: np.ndarray) -> np.ndarray:
    """ECC affine alignment onto reference (grayscale float)."""
    import cv2
    gray = cv2.cvtColor((bgr_f32 * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.float32) / 255.0
    try:
        warp = np.eye(2, 3, dtype=np.float32)
        crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
        _, warp = cv2.findTransformECC(ref_gray, gray, warp,
                                       cv2.MOTION_AFFINE, crit,
                                       None, 5)
        h, w = bgr_f32.shape[:2]
        return cv2.warpAffine(bgr_f32, warp, (w, h),
                              flags=cv2.INTER_LINEAR +
                              cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT)
    except Exception:
        return bgr_f32


def merge_hdr(paths: list, progress=None, align=True) -> np.ndarray:
    """Mertens exposure fusion of bracketed shots. Returns uint8 RGB."""
    import cv2
    if len(paths) < 2:
        raise ValueError("HDR needs at least 2 photos")
    imgs = []
    grays = []
    for i, p in enumerate(paths):
        rgb = _decode_rgb(p, 2600)
        if progress:
            progress(i + 1, len(paths) + 1)
        # MergeMertens operates on 0..255 floats and preserves that scale
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        imgs.append(bgr)
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        g = cv2.resize(g, (g.shape[1] // 4, g.shape[0] // 4))
        grays.append(g)

    ref_idx = len(paths) // 2
    if align:
        aligned = [imgs[ref_idx]]
        for i, im in enumerate(imgs):
            if i == ref_idx:
                continue
            aligned.append(_align_to_ref(im, grays[ref_idx]))
            if progress:
                progress(len(aligned), len(paths) + 1)
        imgs = aligned

    merge = cv2.createMergeMertens()
    fused = merge.process(imgs)          # returns float BGR normalized to ~0..1
    fused = np.clip(fused * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(fused, cv2.COLOR_BGR2RGB)


def stitch_panorama(paths: list, progress=None) -> np.ndarray:
    """cv2.Stitcher panorama. Returns uint8 RGB."""
    import cv2
    if len(paths) < 2:
        raise ValueError("Panorama needs at least 2 photos")
    imgs = []
    for i, p in enumerate(paths):
        rgb = _decode_rgb(p, 2200)
        imgs.append(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if progress:
            progress(i + 1, len(paths) + 1)
    stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(imgs)
    if status != cv2.Stitcher_OK or pano is None:
        msgs = {1: "need more images", 2: "not enough image memory",
                3: "homography estimation failed",
                4: "camera parameters adjustment failed"}
        raise RuntimeError(f"Stitching failed ({msgs.get(status, 'error')})")
    return cv2.cvtColor(pano, cv2.COLOR_BGR2RGB)


def save_merged(u8: np.ndarray, kind: str = "HDR") -> str:
    """Save merged result into the catalog dir and return path."""
    out_dir = os.path.expanduser(f"~/Pictures/Lumina {kind} Merges")
    os.makedirs(out_dir, exist_ok=True)
    from PIL import Image
    i = 1
    while True:
        cand = os.path.join(out_dir, f"{kind}_Merge_{i}.tif")
        if not os.path.exists(cand):
            break
        i += 1
    Image.fromarray(u8).save(cand, "TIFF", compression="tiff_lzw")
    return cand
