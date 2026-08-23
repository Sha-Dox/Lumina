"""Minimal XMP sidecar read/write for ratings, flags, labels, keywords."""
from __future__ import annotations

import os
import re

COLOR_NAMES = {0: "", 1: "Red", 2: "Yellow", 3: "Green", 4: "Blue", 5: "Purple"}


def write_xmp(photo_path: str, rating: int = 0, flag: int = 0,
              color: int = 0, keywords: str = "") -> str | None:
    base = os.path.splitext(photo_path)[0]
    sidecar = base + ".xmp"
    label = COLOR_NAMES.get(color, "")
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    kw_xml = "".join(f"<li>{k}</li>" for k in kw_list) or "<li/>"
    try:
        with open(sidecar, "w", encoding="utf-8") as f:
            f.write(f'''<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Lumina">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:xmp="http://ns.adobe.com/xap/1.0/"
    xmlns:xmpDM="http://ns.adobe.com/xmp/1.0/DynamicMedia/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/"
    xmlns:lumina="https://lumina.app/ns/">
   <xmp:Rating>{rating}</xmp:Rating>
   <xmp:Label>{label}</xmp:Label>
   <lumina:Flag>{flag}</lumina:Flag>
   <photoshop:SupplementalCategories>
    <rdf:Bag>{kw_xml}</rdf:Bag>
   </photoshop:SupplementalCategories>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>')''')
        return sidecar
    except Exception:
        return None


def read_xmp(photo_path: str) -> dict:
    base = os.path.splitext(photo_path)[0]
    sidecar = base + ".xmp"
    if not os.path.exists(sidecar):
        return {}
    try:
        txt = open(sidecar, "r", encoding="utf-8").read()
        out = {}
        m = re.search(r"<xmp:Rating>(\d+)</xmp:Rating>", txt)
        if m:
            out["rating"] = int(m.group(1))
        m = re.search(r"<lumina:Flag>(-?\d+)</lumina:Flag>", txt)
        if m:
            out["flag"] = int(m.group(1))
        for ckey, cname in COLOR_NAMES.items():
            if cname and f"<xmp:Label>{cname}</xmp:Label>" in txt:
                out["color"] = ckey
                break
        kws = re.findall(r"<li>([^<]+)</li>",
                         re.search(r"SupplementalCategories.*?</rdf:Bag>", txt,
                                   re.S).group(0) if "SupplementalCategories" in txt else "")
        if kws:
            out["keywords"] = ", ".join(kws)
        return out
    except Exception:
        return {}
