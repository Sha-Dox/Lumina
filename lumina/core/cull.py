"""Auto-cull: objective photo scoring for burst selection.

score = weighted mix of sharpness (Laplacian variance), exposure quality
(clipping penalties), and a small bonus for containing people.
"""
from __future__ import annotations

import numpy as np

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

SHARP_FLOOR = 12.0        # below this = almost certainly motion/defocus blur


def sharpness_score(gray_f32: np.ndarray) -> float:
    if _HAS_CV2:
        lap = cv2.Laplacian(gray_f32, cv2.CV_32F)
    else:
        lap = np.diff(gray_f32, axis=0)[:, :-1] ** 2 + \
              np.diff(gray_f32, axis=-1)[:-1, :] ** 2
    return float(np.var(lap))


def exposure_penalty(img_f32: np.ndarray) -> float:
    """0..1 — fraction of pixels clipped to pure black/white."""
    L = img_f32[..., 0]*0.2126 + img_f32[..., 1]*0.7152 + img_f32[..., 2]*0.0722
    return float(((L < 0.02) | (L > 0.98)).mean())


def detect_faces(rgb_u8: np.ndarray):
    """Normalized (x,y,w,h) face rects via Apple Vision."""
    try:
        import io as _io
        import Vision
        from PIL import Image as PILImage
        buf = _io.BytesIO()
        PILImage.fromarray(rgb_u8).save(buf, "JPEG", quality=90)
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            buf.getvalue(), None)
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        ok, _e = handler.performRequests_error_([req], None)
        out = []
        if ok:
            for r in (req.results() or []):
                bb = r.boundingBox()
                out.append((float(bb.origin.x), float(bb.origin.y),
                            float(bb.size.width), float(bb.size.height)))
        return out
    except Exception as e:
        print("[cull faces]", e)
        return []


def analyze(preview_u8: np.ndarray, use_faces: bool = True,
            face_detect_fn=None) -> dict:
    """preview_u8: RGB uint8 preview (any size ≥ ~400px recommended)."""
    from lumina.core.imaging import gauss_blur_f  # noqa: F401 (warm import)
    from .aimask import detect_faces as _vision_faces
    face_detect_fn = face_detect_fn or _vision_faces
    f32 = preview_u8.astype(np.float32) / 255.0
    gray = f32[..., 0]*0.2126 + f32[..., 1]*0.7152 + f32[..., 2]*0.0722

    sharp = sharpness_score(gray)
    # perceptual scaling: variance grows fast with texture; compress
    sharp_c = float(np.log1p(sharp))

    clip = exposure_penalty(f32)

    faces = []
    if use_faces and face_detect_fn:
        faces = face_detect_fn(preview_u8) or []

    score = sharp_c * 2.2 - clip * 6.0
    if faces:
        score += 0.55 + min(len(faces), 4) * 0.15

    return {"sharp": sharp, "sharp_c": sharp_c, "clip": clip,
            "faces": len(faces), "score": round(float(score), 3),
            "blurry": sharp < SHARP_FLOOR}


def assign_ratings(results: list[dict], ids: list[int],
                   respect_existing: bool = True,
                   mark_rejects: bool = True):
    """results parallel to ids. Returns list of (id, rating, flag)."""
    scored = sorted(zip(results, ids), key=lambda t: t[0]["score"],
                    reverse=True)
    n = len(scored)
    out = []
    for rank, (res, pid) in enumerate(scored):
        frac = rank / max(n - 1, 1)
        if res["blurry"]:
            rating = 1
        elif frac <= 0.33:
            rating = 3
        elif frac <= 0.66:
            rating = 2
        else:
            rating = 1
        flag = -1 if (mark_rejects and res["blurry"]) else None
        out.append((pid, rating, flag, res))
    return out
