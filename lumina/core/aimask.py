"""AI masking via Apple Vision framework.

- compute_subject_mask : foreground lift or person segmentation
- compute_subject_at_point : click-to-select a specific connected subject
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image


class NoSubjectFound(Exception):
    pass


def vision_available() -> bool:
    try:
        import Vision  # noqa: F401
        return hasattr(Vision, "VNGenerateForegroundInstanceMaskRequest")
    except Exception:
        return False


def _run_foreground_request(rgb_u8, work_size):
    """Returns (observation, handler, scaled_pil_image). Raises on failure."""
    import Vision

    h0, w0 = rgb_u8.shape[:2]
    scale = work_size / max(h0, w0)
    if scale < 1.0:
        im = Image.fromarray(rgb_u8).resize(
            (int(w0 * scale), int(h0 * scale)), Image.Resampling.LANCZOS)
    else:
        im = Image.fromarray(rgb_u8)

    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=94)
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        buf.getvalue(), None)
    request = Vision.VNGenerateForegroundInstanceMaskRequest.alloc().init()
    ok, _err = handler.performRequests_error_([request], None)
    results = list(request.results() or [])
    if not ok or not results:
        raise NoSubjectFound("no subject detected")
    return results[0], handler, im


def _mask_pb_to_array(mask_pb) -> np.ndarray:
    import Quartz
    Quartz.CVPixelBufferLockBaseAddress(
        mask_pb, Quartz.kCVPixelBufferLock_ReadOnly)
    try:
        pw = Quartz.CVPixelBufferGetWidth(mask_pb)
        ph = Quartz.CVPixelBufferGetHeight(mask_pb)
        bpr = Quartz.CVPixelBufferGetBytesPerRow(mask_pb)
        base = Quartz.CVPixelBufferGetBaseAddress(mask_pb)
        arr = np.frombuffer(bytes(base.as_buffer(bpr * ph)),
                            dtype=np.uint8).reshape(ph, bpr)[:, :pw]
        return arr.astype(np.float32) / 255.0
    finally:
        Quartz.CVPixelBufferUnlockBaseAddress(
            mask_pb, Quartz.kCVPixelBufferLock_ReadOnly)


def _resize_to(img_arr: np.ndarray, w0: int, h0: int) -> np.ndarray:
    mh, mw = img_arr.shape
    if (mh, mw) != (h0, w0):
        img_arr = np.asarray(Image.fromarray(
            (img_arr * 255).astype(np.uint8), "L").resize(
            (w0, h0), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    return img_arr


def _soften(mask: np.ndarray, h0: int, w0: int,
            feather_scale: float = 900.0) -> np.ndarray:
    from scipy.ndimage import gaussian_filter
    mask = gaussian_filter(mask, sigma=max(1.5, max(h0, w0) / feather_scale))
    return np.clip(mask, 0.0, 1.0).astype(np.float32)


# ------------------------------------------------------------------ public

def compute_subject_mask(rgb_u8: np.ndarray, work_size: int = 1024,
                         mode: str = "subject") -> np.ndarray | None:
    """Full-scene AI mask ('subject' = all foreground, 'person' = people).
    Returns HxW float32 [0,1] or None."""
    try:
        import Vision

        h0, w0 = rgb_u8.shape[:2]

        if mode == "person":
            im = Image.fromarray(rgb_u8)
            w_pt, h_pt = im.size
            if max(w_pt, h_pt) > work_size:
                sc = work_size / max(w_pt, h_pt)
                im = im.resize((int(w_pt*sc), int(h_pt*sc)),
                               Image.Resampling.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=92)
            handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
                buf.getvalue(), None)
            req = Vision.VNGeneratePersonSegmentationRequest.alloc()\
                .initWithCompletionHandler_(None)
            req.setQualityLevel_(3)
            ok, _e = handler.performRequests_error_([req], None)
            res = list(req.results() or [])
            if not ok or not res:
                return None
            mask = _mask_pb_to_array(res[0].pixelBuffer())
            mask = _resize_to(mask, w0, h0)
            from scipy.ndimage import gaussian_filter
            return np.clip(gaussian_filter(
                mask, sigma=max(1.5, max(h0, w0)/900.0)), 0, 1).astype(np.float32)

        obs, _handler, _im = _run_foreground_request(rgb_u8, work_size)
        instances = obs.allInstances()
        mask_pb, err = obs.generateMaskForInstances_error_(instances, None)
        if err is not None or mask_pb is None:
            return None
        mask = _mask_pb_to_array(mask_pb)
        mask = _resize_to(mask, w0, h0)
        from scipy.ndimage import gaussian_filter
        return np.clip(gaussian_filter(
            mask, sigma=max(1.0, max(h0, w0)/900.0)), 0, 1).astype(np.float32)
    except NoSubjectFound:
        raise
    except Exception as e:
        print("[aimask]", e)
        return None


def compute_subject_at_point(rgb_u8: np.ndarray, x_norm: float, y_norm: float,
                             work_size: int = 1024,
                             feather_scale: float = 700.0) -> np.ndarray:
    """Click-to-select: segment everything, keep the connected component
    under the clicked point. Raises NoSubjectFound when the click lands on
    empty space or nothing is detected."""
    from scipy.ndimage import label as nd_label

    base_mask = compute_subject_mask(rgb_u8, work_size=work_size)
    if base_mask is None:
        raise NoSubjectFound("no subject detected")

    h0, w0 = base_mask.shape
    cx = min(w0 - 1, max(0, int(x_norm * w0)))
    cy = min(h0 - 1, max(0, int(y_norm * h0)))

    # normalize by peak so faint-confidence subjects still segment cleanly
    peak = float(base_mask.max())
    solid = (base_mask / max(peak, 1e-6)) > 0.35
    base_mask = base_mask / max(peak, 1e-6) if peak > 0 else base_mask
    r = max(6, int(min(h0, w0) * 0.03))
    y0, y1 = max(0, cy - r), min(h0, cy + r)
    x0, x1 = max(0, cx - r), min(w0, cx + r)

    if not solid[cy, cx] and not solid[y0:y1, x0:x1].any():
        raise NoSubjectFound("clicked empty area")

    lab, n = nd_label(solid)
    pick = lab[cy, cx]
    if pick == 0:
        window = lab[y0:y1, x0:x1]
        vals, counts = np.unique(window[window > 0], return_counts=True)
        if len(vals) == 0:
            raise NoSubjectFound("clicked empty area")
        pick = vals[np.argmax(counts)]     # biggest component near click

    comp = (lab == pick).astype(np.float32)
    out = base_mask * comp
    return _soften(out, h0, w0, feather_scale)


def compute_background_mask(rgb_u8: np.ndarray,
                            work_size: int = 1024) -> np.ndarray | None:
    m = compute_subject_mask(rgb_u8, work_size=work_size)
    if m is None:
        return None
    return (1.0 - m).astype(np.float32)


def detect_faces(rgb_u8):
    """Face rectangles normalized (x,y,w,h) via Vision. [] on failure."""
    try:
        import Vision
        buf = io.BytesIO()
        Image.fromarray(rgb_u8).save(buf, "JPEG", quality=90)
        handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
            buf.getvalue(), None)
        req = Vision.VNDetectFaceRectanglesRequest.alloc().init()
        ok, _e = handler.performRequests_error_([req], None)
        out = []
        if ok:
            for r in (req.results() or []):
                bb = r.boundingBox()
                out.append((bb.origin.x, bb.origin.y,
                            bb.size.width, bb.size.height))
        return out
    except Exception as e:
        print("[faces]", e)
        return []
