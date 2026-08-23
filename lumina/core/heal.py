"""Spot removal: heal (inpaint) + clone stamp."""
from __future__ import annotations

import hashlib

import numpy as np


def spots_hash(spots: list) -> str:
    blob = repr(sorted(repr(s) for s in spots))
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def apply_spots(base_u8: np.ndarray, spots: list) -> np.ndarray:
    """spots: [{cx,cy,r (normalized), mode:'heal'|'clone', sx,sy (norm clone src)}]"""
    if not spots:
        return base_u8
    try:
        import cv2
    except ImportError:
        return base_u8
    h, w = base_u8.shape[:2]
    diag = (w * w + h * h) ** 0.5

    out = base_u8.copy()
    bgr = out[:, :, ::-1]

    # --- clone patches first
    for s in spots:
        if s.get("mode") != "clone":
            continue
        r_px = max(3, int(float(s.get("r", 0.03)) * diag))
        cx, cy = int(float(s["cx"]) * w), int(float(s["cy"]) * h)
        sx = int(float(s.get("sx", s["cx"] - 0.08)) * w)
        sy = int(float(s.get("sy", s["cy"])) * h)
        y0d, y1d = max(0, cy - r_px), min(h, cy + r_px)
        x0d, x1d = max(0, cx - r_px), min(w, cx + r_px)
        if y1d - y0d < 2 or x1d - x0d < 2:
            continue
        ph, pw = y1d - y0d, x1d - x0d
        sy0 = max(0, min(h - ph, sy - ph // 2))
        sx0 = max(0, min(w - pw, sx - pw // 2))
        patch = bgr[sy0:sy0 + ph, sx0:sx0 + pw].copy()
        if patch.shape[:2] != (ph, pw):
            continue
        yy, xx = np.mgrid[0:ph, 0:pw].astype(np.float32)
        rr = np.sqrt((yy - ph / 2) ** 2 + (xx - pw / 2) ** 2) / max(r_px, 1)
        alpha = np.clip((1.05 - rr) / 0.25, 0, 1)[..., None].astype(np.float32)
        region = bgr[y0d:y1d, x0d:x1d].astype(np.float32)
        blended = region * (1 - alpha) + patch.astype(np.float32) * alpha
        bgr[y0d:y1d, x0d:x1d] = blended.astype(np.uint8)

    # --- red-eye correction (no inpaint; recolor red pupils)
    for s in spots:
        if s.get("mode") != "redeye":
            continue
        r_px = max(2, int(float(s.get("r", 0.015)) * diag))
        cx, cy = int(float(s["cx"]) * w), int(float(s["cy"]) * h)
        y0d, y1d = max(0, cy - r_px), min(h, cy + r_px)
        x0d, x1d = max(0, cx - r_px), min(w, cx + r_px)
        if y1d - y0d < 1 or x1d - x0d < 1:
            continue
        reg = out[y0d:y1d, x0d:x1d].astype(np.float32)
        R, Gc, Bc = reg[..., 2], reg[..., 1], reg[..., 0]
        redness = R - np.maximum(Gc, Bc) * 1.25 - 18
        m = (redness > 0).astype(np.float32)[..., None]
        gray = ((Gc + Bc) * 0.5 * 0.55 + 14)[..., None]
        reg = reg * (1 - m) + np.stack([gray[..., 0] * 1.02,
                                        gray[..., 0], gray[..., 0]], axis=-1) * m
        out[y0d:y1d, x0d:x1d] = np.clip(reg, 0, 255).astype(np.uint8)

    # --- heal spots: single inpaint pass
    mask = np.zeros((h, w), dtype=np.uint8)
    has_heal = False
    for s in spots:
        if s.get("mode") == "clone":
            continue
        has_heal = True
        r_px = max(3, int(float(s.get("r", 0.03)) * diag))
        cv2.circle(mask, (int(float(s["cx"]) * w), int(float(s["cy"]) * h)),
                   r_px, 255, -1)
    if has_heal:
        fixed = cv2.inpaint(bgr, mask, 5, cv2.INPAINT_TELEA)
        m3 = (mask > 0)[..., None]
        bgr = np.where(m3, fixed, bgr)
    return bgr[:, :, ::-1].copy()


if __name__ == "__main__":
    # smoke test
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    yy, xx = np.mgrid[0:300, 0:400]
    img[..., 0] = (xx * 255 // 400)
    img[..., 1] = (yy * 255 // 300)
    dusty = img.copy()
    cv2_circle = None
    try:
        import cv2
        cv2.circle(dusty, (200, 150), 12, (240, 240, 240), -1)
        healed = apply_spots(dusty, [{"cx": 0.5, "cy": 0.5, "r": 0.04,
                                      "mode": "heal"}])
        diff_before = np.abs(dusty.astype(int) - img.astype(int)).sum()
        diff_after = np.abs(healed.astype(int) - img.astype(int)).sum()
        print(f"heal smoke: dust delta {diff_before} -> healed delta {diff_after}")
        assert diff_after < diff_before * 0.4, "heal did not remove spot"
        cloned = apply_spots(dusty, [{"cx": 0.7, "cy": 0.5, "r": 0.04,
                                      "mode": "clone", "sx": 0.3, "sy": 0.5}])
        print(f"clone smoke ok, changed px: "
              f"{int((np.abs(cloned.astype(int)-img.astype(int)).sum(-1)>40).sum())}")
        print("HEAL MODULE PASSED")
    except ImportError:
        print("cv2 missing")
