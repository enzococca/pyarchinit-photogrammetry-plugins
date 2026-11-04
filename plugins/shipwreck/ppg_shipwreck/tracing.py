"""The tracing (``ricalco``): the timbers of the hull as records of the catalogue (:mod:`timbers`) drawn on the relief
of the hull from above (:mod:`relief`). A proposal (:mod:`proposals`) is accepted into a record or rejected; a timber
is traced by hand on the relief, or traced again (corrected); the frames take the site's names from a table; the
strakes, which run under the frames, get their visible parts; a timber found out of place is flagged as loose.

Every record written here is ``measured`` (:data:`timbers.STATUS`), with its ``geometry`` in the chunk's coordinates
(through the ship frame) and ``ship_xy`` in the ship plan, the key of the ``chunk``, the relief's ``elevation`` entry
and, for an accepted proposal, its ``source``. ``geometry`` is the record; ``ship_xy`` (and ``visible_xy``) a cache in
the plan of the frame whose key (:func:`relief.frame_key`) is ``frame_key``: read it through :func:`ship_xy_of`, which
recomputes it when the frame changed. Nothing is read or traced in the plan of a frame that is no longer the chunk's:
the proposals carry the frame they were read in, the relief its own. Only ``chunk.meta["shipwreck"]`` changes: the chunk's transform, model
and products never; a record is never touched unless asked (a correction keeps its id; a new reading of the proposals
never proposes over it)."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from . import proposals, relief, ship, timbers
from .texts import t

GEOMETRY_TYPES = ("line", "polygon", "point")
CUTTERS = ("frame", "keelson", "stringer")  # the timbers lying over the planking
CUT = ("plank",)  # the timbers whose visible parts they cut (seen from inside the ceiling lies over the frames)
FALLBACK_SPACE = 0.5  # metres: the frame space taken with fewer than two frames on a side


def _proposals(chunk) -> dict:
    prop = ship.state(chunk).get("proposals")
    if not prop:
        raise ship.NotReady(t("Nessuna proposta: leggi prima il rilievo dello scafo.",
                              "No proposals: read the relief of the hull first."))
    return prop


def _items(chunk, ids) -> tuple[dict, list[dict]]:
    """The proposals and the items of ``ids`` in that order; an id not (or no longer) among them is refused."""
    prop = _proposals(chunk)
    by_id = {i["id"]: i for i in prop["items"]}
    ids = list(ids)
    twice = sorted({i for i in ids if ids.count(i) > 1})
    if twice:
        raise ship.NotReady(t(f"Proposte ripetute: {', '.join(twice)}.", f"Proposals given twice: {', '.join(twice)}."))
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise ship.NotReady(t(f"Proposte non trovate (già accettate o scartate?): {', '.join(missing)}.",
                              f"Proposals not found (already accepted or rejected?): {', '.join(missing)}."))
    return prop, [by_id[i] for i in ids]


def _shape(xy, geometry_type: str):
    import shapely

    xy = np.asarray(xy, float).reshape(-1, 2)
    if geometry_type == "point":
        return shapely.Point(xy[0])
    if geometry_type == "line" or len(xy) < 3:
        return shapely.LineString(xy)
    return shapely.Polygon(xy)


def _geometry_type(item: dict) -> str:
    return "line" if item["kind"] == "seam" or len(item["ship_xy"]) < 3 else "polygon"


def _world(frame: ship.Frame, ship_xy) -> np.ndarray:
    xy = np.asarray(ship_xy, float).reshape(-1, 2)
    return frame.to_world(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]


def _elevation(chunk) -> str | None:
    e = relief.entry(chunk)
    return None if e is None else e.get("id")


def need_fresh_relief(chunk) -> None:
    """A relief built for another ship frame or another model has another (u, v): refused."""
    if relief.entry(chunk) is not None and relief.is_stale(chunk):
        raise ship.NotReady(t("Il rilievo dello scafo è di un altro sistema della nave o di un altro modello: "
                              "costruiscilo di nuovo.",
                              "The relief of the hull is of another ship frame or another model: build it again."))


def _need_current_proposals(chunk, prop: dict, frame: ship.Frame) -> None:
    """Proposals read in another ship frame (the bow moved, the keel traced again) are in another plan: accepting
    them would put each timber at its image in the new frame (the mirror, with the bow moved): refused, as on a stale
    relief."""
    need_fresh_relief(chunk)
    read_in = prop.get("frame")
    if read_in is not None and relief.frame_key(ship.Frame.from_dict(read_in)) != relief.frame_key(frame):
        raise ship.NotReady(t("Le proposte sono state lette in un altro sistema della nave: ricostruisci il rilievo "
                              "dello scafo e leggi di nuovo le proposte.",
                              "The proposals were read in another ship frame: build the relief of the hull again and "
                              "read the proposals again."))


def _name_of(item: dict, names, k: int) -> str:
    if names is None:
        return item["name"]
    if isinstance(names, Mapping):
        return str(names.get(item["id"], item["name"]))
    names = list(names)
    return str(names[k]) if k < len(names) and names[k] not in (None, "") else item["name"]


def accept(chunk, ids, names=None) -> list[dict]:
    """The proposals ``ids`` into the catalogue, one record each, in that order: the kind, side and name of the
    proposal (``names``: a mapping id -> name, or names in the order of ``ids``), ``geometry`` in the chunk's
    coordinates (a seam a line, the others polygons), ``ship_xy`` as proposed, ``status`` measured, ``chunk``,
    ``source`` (the proposal's id), ``elevation`` (the relief's), ``thickness_mm`` (a seam's or strake's step). The
    items leave the proposals. Returns the records."""
    frame = ship.need_frame(chunk)
    prop, items = _items(chunk, ids)
    _need_current_proposals(chunk, prop, frame)
    elevation, key = _elevation(chunk), relief.frame_key(frame)
    out = []
    for k, item in enumerate(items):
        gtype = _geometry_type(item)
        extra = {"ship_xy": [[float(x), float(y)] for x, y in item["ship_xy"]], "status": "measured",
                 "chunk": chunk.key, "source": item["id"], "elevation": elevation, "frame_key": key}
        if item.get("step_mm") is not None and item["kind"] in ("seam", "plank"):
            extra["thickness_mm"] = float(item["step_mm"])
        out.append(timbers.add(chunk, item["kind"], item["side"], _name_of(item, names, k),
                               timbers.geometry(gtype, _world(frame, item["ship_xy"])), **extra))
    taken = {i["id"] for i in items}
    prop["items"] = [i for i in prop["items"] if i["id"] not in taken]
    return out


def reject(chunk, ids) -> int:
    """The proposals ``ids`` rejected: their geometries (ship plan, WKT) go to ``rejected``, which a new reading never
    proposes again, and they leave the proposals. Returns how many."""
    prop, items = _items(chunk, ids)
    _need_current_proposals(chunk, prop, ship.need_frame(chunk))
    rejected = prop.setdefault("rejected", [])
    for item in items:
        rejected.append(_shape(item["ship_xy"], _geometry_type(item)).wkt)
    gone = {i["id"] for i in items}
    prop["items"] = [i for i in prop["items"] if i["id"] not in gone]
    return len(items)


def _ship_of(uv, frame: ship.Frame, geometry_type: str) -> np.ndarray:
    if geometry_type not in GEOMETRY_TYPES:
        raise ValueError(f"geometry_type: one of {', '.join(GEOMETRY_TYPES)}")
    xy = relief.ship_of_uv(uv, frame.sign)
    timbers.geometry(geometry_type, xy)  # enough vertices
    return xy


def _side(side: str, xy: np.ndarray) -> str:
    if side != "auto":
        return side
    y = float(xy[:, 1].mean())
    if abs(y) <= timbers.SIDE_TOLERANCE:
        return "centre"
    return "port" if y > 0 else "starboard"


def add_traced(chunk, kind: str, side: str, name: str, uv, geometry_type: str) -> dict:
    """A timber traced by hand on the relief: ``uv`` the vertices in the relief's (u, v) (as the view gives them),
    ``geometry_type`` "line", "polygon" or "point"; ``side`` "auto" from the mean y in the ship plan (on the keel line
    within :data:`timbers.SIDE_TOLERANCE`: centre). A measured record with ``ship_xy``."""
    frame = ship.need_frame(chunk)
    need_fresh_relief(chunk)
    xy = _ship_of(uv, frame, geometry_type)
    return timbers.add(chunk, kind, _side(side, xy), name, timbers.geometry(geometry_type, _world(frame, xy)),
                       ship_xy=xy.tolist(), status="measured", chunk=chunk.key, elevation=_elevation(chunk),
                       frame_key=relief.frame_key(frame))


def record(chunk, tid: str) -> dict:
    """The record ``tid`` of the catalogue; :class:`ship.NotReady` when there is none."""
    found = next((r for r in ship.state(chunk).get("timbers", []) if r["id"] == tid), None)
    if found is None:
        raise ship.NotReady(t(f"Legno {tid} non trovato.", f"Timber {tid} not found."))
    return found


def correct(chunk, tid: str, uv) -> dict:
    """The record ``tid`` traced again on the relief (``uv`` as in :func:`add_traced`, its geometry's type kept): same
    id, name, kind and side, new ``geometry`` and ``ship_xy``, ``corrected`` True, ``status`` measured (a hypothesis
    traced again on the model becomes a measure: its hypothesis or import note goes). The visible parts of the
    strakes follow when they were computed."""
    frame = ship.need_frame(chunk)
    need_fresh_relief(chunk)
    rec = record(chunk, tid)
    gtype = rec["geometry"]["type"]
    xy = _ship_of(uv, frame, gtype)
    if not timbers.is_measured(rec):
        rec["notes"] = ""
    rec.update(geometry=timbers.geometry(gtype, _world(frame, xy)), ship_xy=xy.tolist(), corrected=True,
               status="measured", elevation=_elevation(chunk), frame_key=relief.frame_key(frame))
    rec.setdefault("chunk", chunk.key)
    rec.pop("visible_xy", None)
    if any("visible_xy" in r for r in ship.state(chunk).get("timbers", [])):
        update_visible(chunk)
    return rec


def ship_xy_of(chunk, record: dict, frame: ship.Frame | None = None) -> np.ndarray:
    """A record's geometry in the plan of the chunk's ship frame (``frame``: that frame, when the caller has it):
    its ``ship_xy`` when cached in this frame (``frame_key``), else its ``geometry`` (the chunk's coordinates, the
    record proper) through the frame. A cache of another frame is refreshed in the record (``ship_xy``,
    ``frame_key``; its ``visible_xy``, in the old plan, dropped); a record without a cache (0.2.0, a hand record) is
    left as it is."""
    frame = frame or ship.need_frame(chunk)
    key = relief.frame_key(frame)
    if record.get("ship_xy") is not None and record.get("frame_key") == key:
        return np.asarray(record["ship_xy"], float).reshape(-1, 2)
    w = np.asarray(record["geometry"]["xy"], float).reshape(-1, 2)
    xy = frame.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2]
    if record.get("ship_xy") is not None:
        record["ship_xy"], record["frame_key"] = xy.tolist(), key
        record.pop("visible_xy", None)
    return xy


def _names_table(path) -> list[dict]:
    header, rows = ship.rows_of(path)
    name_col = next((header.index(k) for k in ("nome", "name", "ordinata", "frame") if k in header), 0)
    x_col = next((header.index(k) for k in ("x", "dist", "distanza") if k in header), 1)
    out = []
    for i, row in rows:
        try:
            out.append({"name": " ".join(row[name_col].split()), "x": ship.number(row[x_col])})
        except (ValueError, IndexError):
            raise ship.NotReady(t(f"Riga {i} della tabella delle ordinate: serve nome, x.",
                                  f"Line {i} of the table of frames: name, x needed.")) from None
    if not out:
        raise ship.NotReady(t("La tabella delle ordinate è vuota.", "The table of frames is empty."))
    return out


def rename_frames(chunk, path) -> tuple[int, int]:
    """The frames of the catalogue named from a table ``nome, x`` (``name, x``; ``;``, tab or comma, a decimal comma
    with ``;`` or tabs, as :func:`ship.rows_of`): rows and frames paired one to one on each side, the closest pairs
    first (the mean x of a frame's geometry in the ship plan, :func:`ship_xy_of`), within half the median frame space
    (:func:`proposals.frame_space`; :data:`FALLBACK_SPACE` with fewer than two frames on a side). A row names one frame
    per side (the port and starboard pieces of one frame share it; a frame on the centre takes both sides). Returns
    (frames renamed, rows matching no frame)."""
    frame = ship.need_frame(chunk)
    rows = _names_table(path)
    frames = [r for r in ship.state(chunk).get("timbers", []) if r["kind"] == "frame"]
    xs = [float(ship_xy_of(chunk, r, frame)[:, 0].mean()) for r in frames]
    space = proposals.frame_space([{"x": x, "side": r["side"]} for r, x in zip(frames, xs)]) or FALLBACK_SPACE
    pairs = sorted((abs(row["x"] - x), k, j) for k, row in enumerate(rows) for j, x in enumerate(xs)
                   if abs(row["x"] - x) <= space / 2.0)
    named, sides_of_row = set(), {}
    for _d, k, j in pairs:
        sides = {"port", "starboard"} if frames[j]["side"] == "centre" else {frames[j]["side"]}
        if j in named or sides & sides_of_row.get(k, set()):
            continue
        frames[j]["name"] = rows[k]["name"]
        named.add(j)
        sides_of_row.setdefault(k, set()).update(sides)
    return len(named), len(rows) - len(sides_of_row)


def _rings(geom) -> list:
    """The rings of a (multi)polygon, outer counter-clockwise and holes clockwise: the signed areas add up."""
    import shapely

    out = []
    for poly in getattr(geom, "geoms", [geom]):
        if poly.geom_type != "Polygon" or poly.is_empty:
            continue
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        for ring in [poly.exterior, *poly.interiors]:
            out.append([[float(x), float(y)] for x, y in np.asarray(ring.coords)[:-1]])
    return out


def update_visible(chunk) -> int:
    """``visible_xy`` of every plank with a polygon: its polygon in the ship plan minus the polygons of the frames,
    keelsons and stringers (the strakes are traced under them, their footprints included; the ceiling lies over the
    frames and is not cut): a list of rings in the ship plan (:func:`_rings`). Returns how many records were given
    their visible parts."""
    import shapely

    frame = ship.need_frame(chunk)
    records = ship.state(chunk).get("timbers", [])
    cutters = []
    for r in records:
        if r["kind"] in CUTTERS and r["geometry"]["type"] == "polygon":
            poly = shapely.Polygon(ship_xy_of(chunk, r, frame))
            cutters.append(poly if poly.is_valid else poly.buffer(0))
    over = shapely.union_all(cutters) if cutters else None
    count = 0
    for r in records:
        if r["kind"] not in CUT or r["geometry"]["type"] != "polygon":
            continue
        poly = shapely.Polygon(ship_xy_of(chunk, r, frame))
        if not poly.is_valid:
            poly = poly.buffer(0)
        r["visible_xy"] = _rings(poly.difference(over) if over is not None else poly)
        count += 1
    return count


def loose(chunk, ids, flag: bool = True) -> int:
    """The records ``ids`` flagged as loose timbers found out of place (``kind`` other, ``loose`` True; the kind they
    had kept in ``kind_before``), or the flag taken off (``flag`` False: the kind they had back). Returns how many."""
    wanted = set(ids)
    missing = wanted - {r["id"] for r in ship.state(chunk).get("timbers", [])}
    if missing:
        raise ship.NotReady(t(f"Legni non trovati: {', '.join(sorted(missing))}.",
                              f"Timbers not found: {', '.join(sorted(missing))}."))
    count = 0
    for r in ship.state(chunk).get("timbers", []):
        if r["id"] in wanted:
            if flag and r["kind"] != "other":
                r["kind_before"] = r["kind"]
                r["kind"] = "other"
                r.pop("visible_xy", None)
            elif not flag and r.get("kind_before"):
                r["kind"] = r.pop("kind_before")
            r["loose"] = bool(flag)
            count += 1
    return count
