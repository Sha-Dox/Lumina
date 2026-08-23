from ..core.xmp import write_xmp


def write_sidecar(path, **kw):
    write_xmp(path, rating=kw.get("rating", 0),
              flag=kw.get("flag", 0), color=kw.get("color", 0),
              keywords=kw.get("keywords", ""))
