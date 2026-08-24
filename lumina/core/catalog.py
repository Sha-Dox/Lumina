"""SQLite catalog — the library database (paths, ratings, flags, edits, metadata)."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

DB_PATH = os.path.expanduser("~/.lumina/catalog.db")
_lock = threading.Lock()


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH, timeout=10)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_db() -> None:
    with _lock, _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS photos (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                path TEXT UNIQUE NOT NULL,
                folder TEXT NOT NULL,
                filename TEXT NOT NULL,
                mtime REAL, size INTEGER,
                rating INTEGER DEFAULT 0,
                flag INTEGER DEFAULT 0,
                color INTEGER DEFAULT 0,
                camera TEXT DEFAULT '', lens TEXT DEFAULT '',
                iso INTEGER, aperture REAL, shutter TEXT DEFAULT '',
                focal REAL, date_taken TEXT DEFAULT '',
                width INTEGER DEFAULT 0, height INTEGER DEFAULT 0,
                settings_json TEXT DEFAULT '',
                added_at REAL
            )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_photos_folder ON photos(folder)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_photos_path ON photos(path)")
        cols = {r["name"] for r in c.execute("PRAGMA table_info(photos)").fetchall()}
        if "gps_lat" not in cols:
            c.execute("ALTER TABLE photos ADD COLUMN gps_lat REAL")
        if "gps_lon" not in cols:
            c.execute("ALTER TABLE photos ADD COLUMN gps_lon REAL")
        if "keywords" not in cols:
            c.execute("ALTER TABLE photos ADD COLUMN keywords TEXT DEFAULT ''")
        c.execute("""CREATE TABLE IF NOT EXISTS collections (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL)""")
        c.execute("""CREATE TABLE IF NOT EXISTS collection_photos (
                        collection_id INTEGER NOT NULL,
                        photo_id INTEGER NOT NULL,
                        UNIQUE(collection_id, photo_id))""")
        c.execute("""CREATE TABLE IF NOT EXISTS versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        photo_id INTEGER NOT NULL,
                        name TEXT DEFAULT 'Edit',
                        settings_json TEXT DEFAULT '{}',
                        created_at REAL,
                        FOREIGN KEY (photo_id) REFERENCES photos(id))""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_versions_photo ON versions(photo_id)")


def upsert_photo(path: str, md: dict) -> int:
    folder = os.path.dirname(path)
    filename = os.path.basename(path)
    try:
        st = os.stat(path)
        mtime, size = st.st_mtime, st.st_size
    except OSError:
        mtime, size = 0.0, 0
    glat = md.get("gps_lat")
    glon = md.get("gps_lon")
    with _lock, _conn() as c:
        cur = c.execute("SELECT id FROM photos WHERE path=?", (path,))
        row = cur.fetchone()
        if row:
            if glat is None:
                old = c.execute("SELECT gps_lat FROM photos WHERE id=?",
                                (row["id"],)).fetchone()
                glat = old["gps_lat"] if old else None
                glon = (c.execute("SELECT gps_lon FROM photos WHERE id=?",
                                  (row["id"],)).fetchone() or {"gps_lon": None})["gps_lon"]
            c.execute("""UPDATE photos SET folder=?, filename=?, mtime=?, size=?,
                         camera=?, lens=?, iso=?, aperture=?, shutter=?, focal=?,
                         date_taken=?, width=?, height=?, gps_lat=?, gps_lon=? WHERE id=?""",
                      (folder, filename, mtime, size, md.get("camera", ""),
                       md.get("lens", ""), md.get("iso"), md.get("aperture"),
                       md.get("shutter", ""), md.get("focal"), md.get("date_taken", ""),
                       md.get("width", 0), md.get("height", 0), glat, glon, row["id"]))
            return row["id"]
        cur = c.execute("""INSERT INTO photos
            (path, folder, filename, mtime, size, camera, lens, iso, aperture,
             shutter, focal, date_taken, width, height, added_at, gps_lat, gps_lon)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (path, folder, filename, mtime, size, md.get("camera", ""),
             md.get("lens", ""), md.get("iso"), md.get("aperture"),
             md.get("shutter", ""), md.get("focal"), md.get("date_taken", ""),
             md.get("width", 0), md.get("height", 0), time.time(), glat, glon))
        return cur.lastrowid


def get_photo_by_path(path: str):
    with _lock, _conn() as c:
        return c.execute("SELECT * FROM photos WHERE path=?", (path,)).fetchone()


def get_photo(photo_id: int):
    with _lock, _conn() as c:
        return c.execute("SELECT * FROM photos WHERE id=?", (photo_id,)).fetchone()


def update_fields(photo_id: int, **fields) -> None:
    if not fields:
        return
    cols = ", ".join(f"{k}=?" for k in fields)
    with _lock, _conn() as c:
        c.execute(f"UPDATE photos SET {cols} WHERE id=?", (*fields.values(), photo_id))


def remove_photo(photo_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM photos WHERE id=?", (photo_id,))


def prune_missing() -> int:
    """Drop catalog entries whose files vanished. Returns count removed."""
    removed = 0
    with _lock, _conn() as c:
        for row in c.execute("SELECT id, path FROM photos").fetchall():
            if not os.path.exists(row["path"]):
                c.execute("DELETE FROM photos WHERE id=?", (row["id"],))
                removed += 1
    return removed


def photos_with_gps() -> list:
    with _lock, _conn() as c:
        return c.execute(
            "SELECT * FROM photos WHERE gps_lat IS NOT NULL "
            "AND gps_lon IS NOT NULL ORDER BY filename").fetchall()


def set_gps(photo_id: int, lat: float, lon: float) -> None:
    update_fields(photo_id, gps_lat=float(lat), gps_lon=float(lon))


def set_keywords(photo_id: int, kw: str) -> None:
    update_fields(photo_id, keywords=kw.strip())


# ------------------------------------------------------------------ collections

def list_collections() -> list:
    with _lock, _conn() as c:
        return c.execute("SELECT * FROM collections ORDER BY name").fetchall()


def create_collection(name: str) -> int:
    with _lock, _conn() as c:
        cur = c.execute("INSERT OR IGNORE INTO collections(name) VALUES (?)",
                        (name,))
        row = c.execute("SELECT id FROM collections WHERE name=?", (name,)).fetchone()
        return row["id"] if row else cur.lastrowid


def delete_collection(cid: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM collections WHERE id=?", (cid,))
        c.execute("DELETE FROM collection_photos WHERE collection_id=?", (cid,))


def collection_add(cid: int, photo_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("""INSERT OR IGNORE INTO collection_photos(collection_id, photo_id)
                     VALUES (?,?)""", (cid, photo_id))


def collection_remove(cid: int, photo_id: int) -> None:
    with _lock, _conn() as c:
        c.execute("""DELETE FROM collection_photos
                     WHERE collection_id=? AND photo_id=?""", (cid, photo_id))


def collection_member_ids(cid: int) -> list[int]:
    with _lock, _conn() as c:
        rows = c.execute("""SELECT photo_id FROM collection_photos
                            WHERE collection_id=?""", (cid,)).fetchall()
    return [r["photo_id"] for r in rows]


def list_folders() -> list[str]:
    with _lock, _conn() as c:
        rows = c.execute("SELECT DISTINCT folder FROM photos ORDER BY folder").fetchall()
    return [r["folder"] for r in rows]


def query(folder: str | None = None, min_rating: int = 0, flag: str = "all",
          color: int | None = None, search: str = "") -> list[sqlite3.Row]:
    sql = "SELECT * FROM photos WHERE 1=1"
    args: list = []
    if folder:
        sql += " AND folder=?"
        args.append(folder)
    if min_rating > 0:
        sql += " AND rating>=?"
        args.append(min_rating)
    if flag == "picked":
        sql += " AND flag=1"
    elif flag == "unflagged":
        sql += " AND flag>=0"
    elif flag == "rejected":
        sql += " AND flag=-1"
    if color is not None:
        sql += " AND color=?"
        args.append(color)
    if search:
        sql += " AND (filename LIKE ? OR camera LIKE ? OR lens LIKE ? OR keywords LIKE ?)"
        like = f"%{search}%"
        args += [like, like, like, like]
    sql += " ORDER BY filename"
    with _lock, _conn() as c:
        return c.execute(sql, args).fetchall()


# ------------------------------------------------------------------ edit settings

def load_settings(photo_id: int) -> dict | None:
    with _lock, _conn() as c:
        row = c.execute("SELECT settings_json FROM photos WHERE id=?",
                        (photo_id,)).fetchone()
    if row and row["settings_json"]:
        try:
            return json.loads(row["settings_json"])
        except Exception:
            return None
    return None


def save_settings(photo_id: int, settings: dict) -> None:
    blob = json.dumps(settings)
    with _lock, _conn() as c:
        c.execute("UPDATE photos SET settings_json=? WHERE id=?", (blob, photo_id))


# ------------------------------------------------------------------ versions

def list_versions(photo_id: int) -> list[dict]:
    with _lock, _conn() as c:
        c.row_factory = sqlite3.Row
        rows = c.execute(
            "SELECT id, name, settings_json, created_at FROM versions "
            "WHERE photo_id=? ORDER BY created_at", (photo_id,)).fetchall()
    return [{"id": r["id"], "name": r["name"],
             "settings": r["settings_json"],
             "created": r["created_at"]} for r in rows]


def create_version(photo_id: int, name: str, settings_json: str) -> int:
    with _lock, _conn() as c:
        cur = c.execute(
            "INSERT INTO versions (photo_id, name, settings_json, created_at) "
            "VALUES (?,?,?,?)",
            (photo_id, name, settings_json, time.time()))
        return cur.lastrowid


def load_version(vid: int) -> dict | None:
    with _lock, _conn() as c:
        c.row_factory = sqlite3.Row
        row = c.execute("SELECT * FROM versions WHERE id=?", (vid,)).fetchone()
    if row:
        return {"id": row["id"], "photo_id": row["photo_id"],
                "name": row["name"], "settings": row["settings_json"]}
    return None


def save_version(vid: int, settings_json: str) -> None:
    with _lock, _conn() as c:
        c.execute("UPDATE versions SET settings_json=? WHERE id=?",
                  (settings_json, vid))


def delete_version(vid: int) -> None:
    with _lock, _conn() as c:
        c.execute("DELETE FROM versions WHERE id=?", (vid,))
