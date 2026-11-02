"""The walls of a chunk: ``chunk.meta["masonry"]["walls"]``, records ``{id, name, kind, faces, made}`` - ``kind`` wall,
building or tower, ``faces`` the ids of the program's elevations (``chunk.meta["elevations"]``) that are the faces of
the wall - and the plugin's files under ``<chunk data>/masonry/<wall id>/`` (``<face id>/`` for each face: the bulge
raster, the sections, the comparison with another survey; ``wall.json`` the summary of the shape of every face).

The elevations stay the program's: a face removed here is an elevation the program still lists unless the user asks to
remove it too; an elevation the user removes in the program is no longer a face (:func:`prune`)."""
from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path

from .texts import kind_title, t

KINDS = ("wall", "building", "tower")


class NotReady(RuntimeError):
    """Something a step needs is missing: the message says what."""


def store(chunk) -> dict:
    return chunk.meta.setdefault("masonry", {})


def walls(chunk) -> list[dict]:
    return list(((chunk.meta.get("masonry") or {}).get("walls")) or [])


def folder(chunk) -> Path:
    return Path(chunk.data_dir) / "masonry"


def wall_folder(chunk, wid: str) -> Path:
    return folder(chunk) / wid


def face_folder(chunk, wid: str, eid: str) -> Path:
    return wall_folder(chunk, wid) / eid


def _new_name(chunk, kind: str) -> str:
    base = kind_title(kind)
    taken = {w["name"] for w in walls(chunk)}
    i = 1
    while f"{base} {i}" in taken:
        i += 1
    return f"{base} {i}"


def new_wall(chunk, name: str | None = None, kind: str = "wall") -> dict:
    """A new wall of ``kind`` (wall, building, tower) without faces."""
    if kind not in KINDS:
        raise ValueError(f"kind: one of {', '.join(KINDS)}")
    rec = {"id": uuid.uuid4().hex[:8], "name": name or _new_name(chunk, kind), "kind": kind, "faces": [],
           "made": time.strftime("%Y-%m-%d %H:%M:%S")}
    store(chunk).setdefault("walls", []).append(rec)
    return rec


def find(chunk, key: str) -> dict | None:
    """The wall whose id or name is ``key``."""
    for w in walls(chunk):
        if w["id"] == key:
            return w
    for w in walls(chunk):
        if w["name"] == key:
            return w
    return None


def need(chunk, key: str) -> dict:
    w = find(chunk, key)
    if w is None:
        raise NotReady(t(f"Nessun muro «{key}» nel chunk.", f"No wall «{key}» in the chunk."))
    return w


def _live(chunk, wid: str) -> dict:
    """The record itself (not a copy) of the wall ``wid``."""
    for w in store(chunk).get("walls") or []:
        if w["id"] == wid:
            return w
    raise NotReady(t(f"Nessun muro «{wid}» nel chunk.", f"No wall «{wid}» in the chunk."))


def elevation(chunk, eid: str) -> dict | None:
    for e in chunk.meta.get("elevations") or []:
        if e["id"] == eid:
            return e
    return None


def need_face(chunk, wall_key: str, face_key: str | None = None) -> tuple[dict, dict]:
    """``(wall, elevation entry)``: the face ``face_key`` (an elevation id or name) of the wall; None: its only face."""
    w = need(chunk, wall_key)
    entries = [e for e in (elevation(chunk, f) for f in w["faces"]) if e is not None]
    if face_key is None:
        if len(entries) == 1:
            return w, entries[0]
        raise NotReady(t(f"Il muro «{w['name']}» ha {len(entries)} facce: scegline una.",
                         f"The wall «{w['name']}» has {len(entries)} faces: choose one."))
    for e in entries:
        if face_key in (e["id"], e.get("name")):
            return w, e
    raise NotReady(t(f"Nessuna faccia «{face_key}» nel muro «{w['name']}».",
                     f"No face «{face_key}» in the wall «{w['name']}»."))


def add_face(chunk, wid: str, eid: str) -> dict:
    rec = _live(chunk, wid)
    if eid not in rec["faces"]:
        rec["faces"].append(eid)
    return rec


def wall_of_face(chunk, eid: str) -> dict | None:
    return next((w for w in walls(chunk) if eid in w["faces"]), None)


def remove_face(chunk, wid: str, eid: str, elevation_too: bool = False) -> bool:
    """The face out of the wall, its files with it; ``elevation_too``: the program's elevation removed as well."""
    rec = _live(chunk, wid)
    if eid not in rec["faces"]:
        return False
    rec["faces"].remove(eid)
    rec.get("results", {}).pop(eid, None)
    shutil.rmtree(face_folder(chunk, wid, eid), ignore_errors=True)
    if elevation_too:
        from pyarchinit_photogrammetry.engine.elevation import remove_elevation

        remove_elevation(chunk, eid)
    write_summary(chunk, wid)
    return True


def remove_wall(chunk, wid: str, elevations_too: bool = False) -> bool:
    rec = find(chunk, wid)
    if rec is None:
        return False
    if elevations_too:
        from pyarchinit_photogrammetry.engine.elevation import remove_elevation

        for eid in rec["faces"]:
            remove_elevation(chunk, eid)
    store(chunk)["walls"] = [w for w in walls(chunk) if w["id"] != rec["id"]]
    shutil.rmtree(wall_folder(chunk, rec["id"]), ignore_errors=True)
    return True


def prune(chunk) -> list[str]:
    """The faces whose elevation the program no longer has taken out of their walls (with their files). Returns the
    ids taken out."""
    known = {e["id"] for e in chunk.meta.get("elevations") or []}
    gone = []
    for w in store(chunk).get("walls") or []:
        for eid in [f for f in w["faces"] if f not in known]:
            w["faces"].remove(eid)
            w.get("results", {}).pop(eid, None)
            try:
                shutil.rmtree(face_folder(chunk, w["id"], eid), ignore_errors=True)
            except RuntimeError:  # a chunk without a folder
                pass
            gone.append(eid)
    return gone


def drawings_of(chunk, eid: str) -> dict:
    """How many drawings the elevation ``eid`` holds (the program's drawings kept with an elevation): ``units``,
    ``quote``, ``points``, ``stones``."""
    try:
        from pyarchinit_photogrammetry.engine.elevation import frame_drawings

        data = frame_drawings(chunk, eid)
    except ImportError:  # a program without drawings on elevations
        data = {}
    stones = (data.get("stones") or {}).get(eid) or []
    return {"units": len(data.get("units") or []), "quote": len(data.get("quote") or []),
            "points": len(data.get("points") or []), "stones": len(stones)}


def face_info(chunk, eid: str) -> dict:
    """What the pane lists of a face: its name, surface (plane or cylinder), width and height (metres, from its
    bounds), resolution, size in pixels, whether ``v`` is the height, the drawings it holds."""
    e = elevation(chunk, eid)
    if e is None:
        raise NotReady(t("La faccia non ha più il suo prospetto.", "The face has no elevation any more."))
    u0, v0, u1, v1 = e.get("bounds") or (0, 0, 0, 0)
    return {"id": eid, "name": e.get("name", eid), "surface": (e.get("frame") or {}).get("kind", "plane"),
            "width": float(u1 - u0), "height": float(v1 - v0), "bounds": [float(x) for x in (u0, v0, u1, v1)],
            "resolution": float(e.get("resolution") or 0.0), "size": list(e.get("size") or []),
            "v_is_height": bool(e.get("v_is_height")), "drawings": drawings_of(chunk, eid)}


def set_result(chunk, wid: str, eid: str | None, key: str, value) -> None:
    """A result kept with the wall (and in ``wall.json``): of the face ``eid`` (``shape``, ``sections``, ``surveys``)
    or, ``eid`` None, of the wall (``axis`` of a tower)."""
    rec = _live(chunk, wid)
    if eid is None:
        rec[key] = value
    else:
        rec.setdefault("results", {}).setdefault(eid, {})[key] = value
    write_summary(chunk, wid)


def results(chunk, wid: str, eid: str) -> dict:
    rec = find(chunk, wid)
    return dict(((rec or {}).get("results") or {}).get(eid) or {})


def write_summary(chunk, wid: str) -> Path | None:
    """``<chunk data>/masonry/<wall>/wall.json``: the wall and, per face, its elevation and the measures of its shape
    (``shape``, ``sections``, ``surveys``); the axis of a tower."""
    rec = find(chunk, wid)
    if rec is None:
        return None
    faces = []
    for eid in rec["faces"]:
        try:
            info = face_info(chunk, eid)
        except NotReady:
            continue
        info.update((rec.get("results") or {}).get(eid) or {})
        faces.append(info)
    out = {"id": rec["id"], "name": rec["name"], "kind": rec["kind"], "made": rec.get("made"), "crs": chunk.crs,
           "faces": faces}
    if rec.get("axis"):
        out["axis"] = rec["axis"]
    path = wall_folder(chunk, rec["id"]) / "wall.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    return path
