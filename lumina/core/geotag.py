"""Geotag helpers: write LR-style XMP sidecars with GPS coordinates."""
from __future__ import annotations

import os


def _dms(dec: float) -> tuple[int, int, float]:
    dec = abs(dec)
    d = int(dec)
    m_full = (dec - d) * 60
    m = int(m_full)
    s = (m_full - m) * 60
    return d, m, round(s, 3)


def xmp_for(lat: float, lon: float) -> str:
    ld, lm, ls = _dms(lat)
    od, om, os_ = _dms(lon)
    lref = "N" if lat >= 0 else "S"
    oref = "E" if lon >= 0 else "W"
    return f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Lumina">
 <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
  <rdf:Description rdf:about=""
    xmlns:exif="http://ns.adobe.com/exif/1.0/"
    xmlns:photoshop="http://ns.adobe.com/photoshop/1.0/"
    xmlns:tiff="http://ns.adobe.com/tiff/1.0/">
   <exif:GPSVersionID>2.3.0.0</exif:GPSVersionID>
   <exif:GPSLatitude>{ld},{lm},{ls}{"N" if lat>=0 else "S"}</exif:GPSLatitude>
   <exif:GPSLongitude>{od},{om},{os_}{"E" if lon>=0 else "W"}</exif:GPSLongitude>
   <exif:GPSLatitudeRef>{lref}</exif:GPSLatitudeRef>
   <exif:GPSLongitudeRef>{oref}</exif:GPSLongitudeRef>
   <photoshop:Headline>Geotagged in Lumina</photoshop:Headline>
  </rdf:Description>
 </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''


def write_gps_sidecar(photo_path: str, lat: float, lon: float) -> str:
    base = os.path.splitext(photo_path)[0]
    sidecar = base + ".xmp"
    with open(sidecar, "w", encoding="utf-8") as f:
        f.write(xmp_for(lat, lon))
    return sidecar
