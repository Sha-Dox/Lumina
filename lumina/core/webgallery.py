"""Self-contained static HTML web gallery export."""
from __future__ import annotations

import base64
import html
import os
import shutil

import numpy as np
from PIL import Image

from . import catalog, export as excore, imaging


def export_gallery(photo_rows, out_dir: str, title: str = "Lumina Gallery",
                   thumb_px: int = 480, large_px: int = 1800,
                   progress=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    img_dir = os.path.join(out_dir, "images")
    shutil.rmtree(img_dir, ignore_errors=True)
    os.makedirs(img_dir, exist_ok=True)

    items = []
    n = len(photo_rows)
    for i, row in enumerate(photo_rows):
        try:
            settings = imaging.sanitize_settings(
                catalog.load_settings(row["id"]) or {})
            u8 = excore.render_for_export(row["path"], settings,
                                          max_edge=large_px,
                                          output_sharpen=True)
            stem = f"photo{i+1:03d}"
            Image.fromarray(u8).save(os.path.join(img_dir, stem + ".jpg"),
                                     "JPEG", quality=87)
            th = np.asarray(Image.fromarray(u8).resize(
                (thumb_px, int(u8.shape[0] * thumb_px / u8.shape[1])),
                Image.Resampling.LANCZOS))
            Image.fromarray(th).save(os.path.join(img_dir, stem + "_t.jpg"),
                                     "JPEG", quality=82)
            stars = "\u2605" * int(row["rating"] or 0)
            kw = (row["keywords"] or "").split(",")[0].strip()
            cap = html.escape(row["filename"])
            if kw:
                cap += f" <span class='tag'>{html.escape(kw)}</span>"
            if stars:
                cap += f" <span class='stars'>{'★'*min(5,int(row['rating'] or 0))}</span>"
            items.append((stem, cap))
        except Exception as e:
            print("[gallery] skip", row["filename"], e)
        if progress:
            progress(i + 1, n)

    cards = "\n".join(
        f"""<figure><img loading="lazy" src="images/{stem}_t.jpg"
 data-full="images/{stem}.jpg" alt=""><figcaption>{cap}</figcaption></figure>"""
        for stem, cap in items)

    page = """<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>""" + html.escape(title) + """</title>
<style>
 :root { color-scheme: dark; }
 body { margin:0; background:#141414; color:#ddd;
        font-family:-apple-system,'Helvetica Neue',Arial,sans-serif; }
 header { padding:34px 20px 10px; text-align:center; }
 h1 { font-weight:600; letter-spacing:.06em; margin:0 0 4px; }
 header p { color:#888; margin:0; font-size:13px; }
 .grid { columns: 4 240px; column-gap:14px; padding:24px; max-width:1600px;
         margin:0 auto; }
 figure { margin:0 0 14px; break-inside:avoid; cursor:zoom-in;
          border-radius:10px; overflow:hidden; background:#1d1d1d;
          transition:transform .15s ease; }
 figure:hover { transform:translateY(-2px); }
 img { width:100%; display:block; }
 figcaption { font-size:12px; color:#9a9a9a; padding:8px 10px; }
 .tag { border:1px solid #3a6ea5; color:#7fb2e5; border-radius:20px;
        padding:1px 8px; font-size:11px; }
 .stars { color:#e2c14c; letter-spacing:2px; }
 #lb { position:fixed; inset:0; background:rgba(0,0,0,.94); display:none;
       align-items:center; justify-content:center; z-index:10; cursor:zoom-out; }
 #lb img { max-width:96vw; max-height:96vh; border-radius:4px; }
 footer { text-align:center; color:#555; font-size:12px; padding:30px; }
</style></head><body>
<header><h1>""" + html.escape(title) + """</h1>
<p>""" + str(len(items)) + """ photographs · Lumina</p></header>
<div class='grid'>
""" + cards + """
</div>
<footer>Rendered by Lumina</footer>
<div id='lb'><img alt=''></div>
<script>
const lb=document.getElementById('lb'), im=lb.querySelector('img');
document.querySelectorAll('.grid figure').forEach(f=>{
  f.onclick=()=>{ im.src=f.querySelector('img').dataset.full; lb.style.display='flex'; };
});
lb.onclick=()=>lb.style.display='none';
document.addEventListener('keydown',e=>{if(e.key==='Escape')lb.style.display='none';});
</script></body></html>"""

    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    logo = os.path.expanduser("~/.lumina/brand/logo512.png")
    return out_dir
