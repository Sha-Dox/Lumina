"""RAW & raster file decoding, metadata extraction, thumbnail cache."""
from __future__ import annotations

import hashlib
import io
import os
import threading
import time

import numpy as np
import rawpy
from PIL import Image

RAW_EXTS = {".cr2", ".cr3", ".arw", ".raf", ".nef", ".nrw", ".orf", ".rw2",
            ".dng", ".pef", ".ptx", ".srw", ".x3f", ".3fr", ".erf", ".mef",
            ".mos", ".iiq", ".kdc", ".dcr", ".srf", ".sr2", ".rwl"}
RASTER_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".webp"}
SUPPORTED_EXTS = RAW_EXTS | RASTER_EXTS

CACHE_DIR = os.path.expanduser("~/.lumina/cache")
THUMB_DIR = os.path.join(CACHE_DIR, "thumbs")
os.makedirs(THUMB_DIR, exist_ok=True)

# in-process decoded preview cache (path+mtime -> uint8 array)
_preview_cache: dict[str, tuple[float, np.ndarray]] = {}
_cache_lock = threading.Lock()
_PREVIEW_CACHE_MAX = 4


def is_raw(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in RAW_EXTS


def is_supported(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTS


def _decode_raw(path: str, half: bool) -> np.ndarray:
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            half_size=half,
            no_auto_bright=False,
            output_bps=8,
            user_flip=0,
        )
    return np.ascontiguousarray(rgb[:, :, :3])


def _decode_raster(path: str) -> np.ndarray:
    im = Image.open(path)
    im = im.convert("RGB")
    return np.asarray(im, dtype=np.uint8)


def decode_full(path: str) -> np.ndarray:
    """Full-resolution 8-bit RGB decode (used for export)."""
    if is_raw(path):
        return _decode_raw(path, half=False)
    return _decode_raster(path)


def decode_preview(path: str, max_long_edge: int = 2560) -> np.ndarray:
    """Cached decode sized for interactive editing (per-size cache keys)."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        mtime = 0.0
    key = f"{path}|{mtime}|{int(max_long_edge)}"
    with _cache_lock:
        hit = _preview_cache.get(key)
        if hit is not None and hit[0] == mtime:
            return hit[1]
    if is_raw(path):
        arr = _decode_raw(path, half=True)
    else:
        arr = _decode_raster(path)
        scale = max_long_edge / max(arr.shape[:2])
        if scale < 1.0:
            im = Image.fromarray(arr).resize(
                (max(1, int(arr.shape[1] * scale)), max(1, int(arr.shape[0] * scale))),
                Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.uint8)
    # cap very large half-size RAW decodes as well
    h, w = arr.shape[:2]
    scale = max_long_edge / max(h, w)
    if scale < 1.0:
        im = Image.fromarray(arr).resize((int(w * scale), int(h * scale)),
                                         Image.Resampling.LANCZOS)
        arr = np.asarray(im, dtype=np.uint8)
    with _cache_lock:
        if len(_preview_cache) >= _PREVIEW_CACHE_MAX:
            _preview_cache.pop(next(iter(_preview_cache)))
        _preview_cache[key] = (mtime, arr)
    return arr


# ------------------------------------------------------------------ thumbnails

def thumb_cache_path(path: str) -> str:
    try:
        st = os.stat(path)
    except OSError:
        return os.path.join(THUMB_DIR, "missing.jpg")
    h = hashlib.md5(f"{path}|{st.st_mtime_ns}|{st.st_size}".encode()).hexdigest()
    return os.path.join(THUMB_DIR, f"{h}.jpg")


def make_thumbnail(path: str, size: int = 420) -> str | None:
    """Generate (cached) JPEG thumbnail; returns cache path or None."""
    if not os.path.exists(path):
        return None
    out = thumb_cache_path(path)
    if os.path.exists(out):
        return out
    try:
        img = None
        if is_raw(path):
            with rawpy.imread(path) as raw:
                try:
                    thumb = raw.extract_thumb()
                    if thumb.format == rawpy.ThumbFormat.JPEG:
                        img = Image.open(io.BytesIO(thumb.data)).convert("RGB")
                    elif thumb.format == rawpy.ThumbFormat.BITMAP:
                        img = Image.fromarray(thumb.data)
                except Exception:
                    img = None
            if img is None:
                arr = _decode_raw(path, half=True)
                img = Image.fromarray(arr)
        else:
            img = Image.open(path).convert("RGB")
        img.thumbnail((size, size), Image.Resampling.LANCZOS)
        img.save(out, "JPEG", quality=88)
        return out
    except Exception:
        return None


# ------------------------------------------------------------------ metadata

def extract_metadata(path: str) -> dict:
    md = {
        "camera": "", "lens": "", "iso": None, "aperture": None,
        "shutter": "", "focal": None, "date_taken": "",
        "width": 0, "height": 0,
    }
    try:
        import exifread
        with open(path, "rb") as f:
            tags = exifread.process_file(f, details=False, strict=True)
        def g(name):
            t = tags.get(name)
            if t is None:
                return ""
            s = str(t)
            return s.strip()
        make, model = g("Image Make"), g("Image Model")
        if model and model.startswith(make):
            md["camera"] = model
        else:
            md["camera"] = f"{make} {model}".strip()
        md["lens"] = g("EXIF LensModel") or g("Image LensModel")
        iso = tags.get("EXIF ISOSpeedRatings") or tags.get("EXIF PhotographicSensitivity")
        if iso:
            try:
                md["iso"] = int(str(iso))
            except ValueError:
                pass
        fn = g("EXIF FNumber")
        if fn:
            try:
                md["aperture"] = float(eval(fn)) if "/" in fn else float(fn)
            except Exception:
                pass
        et = g("EXIF ExposureTime")
        if et:
            md["shutter"] = et
            try:
                v = eval(et)
                if isinstance(v, (int, float)):
                    md["shutter"] = f"1/{round(1/v)}" if v < 1 else f"{v:g}\""
            except Exception:
                pass
        fl = g("EXIF FocalLength")
        if fl:
            try:
                md["focal"] = float(eval(fl)) if "/" in fl else float(fl)
            except Exception:
                pass
        dt = g("EXIF DateTimeOriginal") or g("Image DateTime")
        if dt:
            md["date_taken"] = dt.replace(",", "")
        # GPS
        def gps_coord(tag_name, ref_name):
            t = tags.get(tag_name)
            if not t:
                return None
            try:
                vals = [float(str(v).split("/")[0]) /
                        (float(str(v).split("/")[1]) if "/" in str(v) else 1.0)
                        for v in str(t).replace("[", "").replace("]", "").split(",")]
                while len(vals) < 3:
                    vals.append(0.0)
                dec = vals[0] + vals[1] / 60.0 + vals[2] / 3600.0
                ref = g(ref_name).upper()
                return -dec if ref in ("S", "W") else dec
            except Exception:
                return None
        lat = gps_coord("GPS GPSLatitude", "GPS GPSLatitudeRef")
        lon = gps_coord("GPS GPSLongitude", "GPS GPSLongitudeRef")
        if lat is not None and lon is not None:
            md["gps_lat"], md["gps_lon"] = lat, lon
    except Exception:
        pass
    try:
        with Image.open(path) as im:
            md["width"], md["height"] = im.size
    except Exception:
        pass
    if is_raw(path) and not md["width"]:
        try:
            with rawpy.imread(path) as raw:
                md["width"], md["height"] = raw.sizes.width, raw.sizes.height
        except Exception:
            pass
    return md


def human_size(n: int) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"
