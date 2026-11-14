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


def raster_extent(chunk, entry: dict) -> tuple[float, float, float, float] | None:
    """``(u0, v0, u1, v1)`` of the elevation's relief (or orthophoto) raster, None when it has neither."""
    import rasterio

    for key in ("relief", "orthophoto"):
        if entry.get(key) and (Path(chunk.data_dir) / entry[key]).exists():
            with rasterio.open(Path(chunk.data_dir) / entry[key]) as src:
                b = src.bounds
            return float(b.left), float(b.bottom), float(b.right), float(b.top)
    return None


def with_bounds(chunk, entry: dict) -> dict:
    """``entry`` itself when it has its ``bounds``; else a copy whose ``bounds`` are its raster's extent (an elevation
    made by an older program, or by hand), marked ``_bounds_from_raster``. For reading only: the program's entry is
    never changed."""
    if entry.get("bounds"):
        return entry
    extent = raster_extent(chunk, entry)
    if extent is None:
        return entry
    return {**entry, "bounds": list(extent), "_bounds_from_raster": True}


def need_face(chunk, wall_key: str, face_key: str | None = None) -> tuple[dict, dict]:
    """``(wall, elevation entry)``: the face ``face_key`` (an elevation id or name) of the wall; None: its only face.
    The entry has its ``bounds`` (:func:`with_bounds`): read it, do not change it."""
    w = need(chunk, wall_key)
    entries = [e for e in (elevation(chunk, f) for f in w["faces"]) if e is not None]
    if face_key is None:
        if len(entries) == 1:
            return w, with_bounds(chunk, entries[0])
        raise NotReady(t(f"Il muro «{w['name']}» ha {len(entries)} facce: scegline una.",
                         f"The wall «{w['name']}» has {len(entries)} faces: choose one."))
    for e in entries:
        if face_key in (e["id"], e.get("name")):
            return w, with_bounds(chunk, e)
    raise NotReady(t(f"Nessuna faccia «{face_key}» nel muro «{w['name']}».",
                     f"No face «{face_key}» in the wall «{w['name']}»."))


def add_face(chunk, wid: str, eid: str) -> dict:
    rec = _live(chunk, wid)
    if eid not in rec["faces"]:
        rec["faces"].append(eid)
    return rec


def wall_of_face(chunk, eid: str) -> dict | None:
    return next((w for w in walls(chunk) if eid in w["faces"]), None)


def _usable(chunk, entry: dict) -> str | None:
    """Why the program's elevation ``entry`` cannot be a face (None when it can): the plugin reads the orthophoto,
    the relief and the frame of a plane or a cylinder, as it does for the faces it makes itself."""
    if (entry.get("frame") or {}).get("kind") not in ("plane", "cylinder"):
        return t("Il prospetto non è su un piano né su un cilindro: il plugin lavora su quelli.",
                 "The elevation is neither on a plane nor on a cylinder: the plugin works on those.")
    if not entry.get("orthophoto") or not (Path(chunk.data_dir) / entry["orthophoto"]).exists():
        return t("Il prospetto non ha la sua ortofoto sul disco.", "The elevation has no orthophoto on disk.")
    return None


def free_elevations(chunk) -> list[dict]:
    """The chunk's elevations that can become a face and are not one yet: made by the program's own tools (the
    *Prospetto* button, ``ppg elevation``), with their orthophoto on disk, in the chunk's order."""
    taken = {f for w in walls(chunk) for f in w["faces"]}
    return [e for e in chunk.meta.get("elevations") or []
            if isinstance(e, dict) and e.get("id") not in taken and _usable(chunk, e) is None]


def adopt_face(chunk, wid: str, key: str) -> dict:
    """An elevation the program already has (``key``: its id or its name) made a face of the wall ``wid`` as it is:
    nothing is rebuilt. Refused when it is a face of another wall, has no orthophoto, or is neither on a plane nor on
    a cylinder (a section). Returns the elevation entry."""
    rec = _live(chunk, wid)
    entries = [e for e in chunk.meta.get("elevations") or [] if isinstance(e, dict)]
    entry = next((e for e in entries if key in (e.get("id"), e.get("name"))), None)
    if entry is None:
        raise NotReady(t(f"Nessun prospetto «{key}» nel chunk.", f"No elevation «{key}» in the chunk."))
    other = wall_of_face(chunk, entry["id"])
    if other is not None and other["id"] != rec["id"]:
        raise NotReady(t(f"Il prospetto «{entry.get('name', key)}» è già una faccia del muro «{other['name']}».",
                         f"The elevation «{entry.get('name', key)}» is already a face of the wall «{other['name']}»."))
    why = _usable(chunk, entry)
    if why:
        raise NotReady(why)
    add_face(chunk, rec["id"], entry["id"])
    write_summary(chunk, rec["id"])
    return entry


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


def drops_drawings() -> bool:
    """Whether the program removes the drawings kept with an elevation (its stones, units, spot heights, points,
    lines) together with the elevation (1.10.0); before, they stayed in the project, unseen."""
    try:
        from pyarchinit_photogrammetry.engine.elevation_drawings import take_frame  # noqa: F401
    except ImportError:
        return False
    return True


def drawings_lost_text(chunk, eids, removed: bool = False) -> str:
    """The sentence that tells which drawings go with the elevations ``eids`` when they are removed (the stones of a
    face are kept with its elevation) - ``removed``: said of elevations being removed, not as a question; empty when
    they hold none or the program keeps them (before 1.10.0)."""
    if not drops_drawings():
        return ""
    from .texts import plural

    total = {"stones": 0, "units": 0, "quote": 0, "points": 0}
    for eid in eids:
        n = drawings_of(chunk, eid)
        for k in total:
            total[k] += n.get(k, 0)
    words = {"stones": (("pietra", "pietre"), ("stone", "stones")), "units": (("US", "US"), ("unit", "units")),
             "quote": (("quota", "quote"), ("spot height", "spot heights")),
             "points": (("punto", "punti"), ("point", "points"))}
    parts = [plural(n, *words[k]) for k, n in total.items() if n]
    if not parts:
        return ""
    if removed:
        return t(f" I disegni fatti sui prospetti rimossi sono andati via con loro: {', '.join(parts)}.",
                 f" The drawings made on the elevations removed went with them: {', '.join(parts)}.")
    return t(f" Se rimuovi i prospetti, i disegni fatti su di loro vanno via con loro: {', '.join(parts)}.",
             f" If you remove the elevations, the drawings made on them go with them: {', '.join(parts)}.")


def face_info(chunk, eid: str) -> dict:
    """What the pane lists of a face: its name, surface (plane or cylinder), width and height (metres, from its
    bounds), resolution, size in pixels, whether ``v`` is the height, the drawings it holds."""
    e = elevation(chunk, eid)
    if e is None:
        raise NotReady(t("La faccia non ha più il suo prospetto.", "The face has no elevation any more."))
    u0, v0, u1, v1 = with_bounds(chunk, e).get("bounds") or (0, 0, 0, 0)
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
