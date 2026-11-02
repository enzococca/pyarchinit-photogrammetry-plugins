"""The USM of a face: the units drawn on the face's elevation (the program's drawings kept with the elevation,
``unita_tipo`` / ``tipo`` USM or US), the relationships the geometry on the face proposes, and their saving in
PyArchInit through the program's bridge.

Rules of the proposals (on the face, ``v`` upwards):

* **si appoggia a**: a unit above another and touching it along at least 20 % of its own width;
* **taglia**: a unit whose outline interrupts a course of the stones of another (the line through the course's
  centres, within the other unit, enters the first);
* **copre**: a unit above another (its lowest point not below the other's top) that runs along the other's top and
  spans its whole width;
* **riempie**: a unit enclosed by another (within it, the other running along at least 90 % of its outline): an
  opening or a gap blocked inside a wall. The wall around it gets no other relationship to it.

For one pair the strongest is proposed (riempie, taglia, copre, then si appoggia a), each with its evidence. The rules
read shapes: they cannot tell a cut from a straight joint where one unit abuts another. Nothing is
written without the user's check: the relationships are proposals until accepted."""
from __future__ import annotations

import numpy as np

from . import walls
from .texts import metres, num, t
from .walls import NotReady

TOUCH_SHARE = 0.20
INVERSE = {"Si appoggia a": "Gli si appoggia", "Taglia": "Tagliato da", "Copre": "Coperto da",
           "Riempie": "Riempito da"}
RELATION_EN = {"Si appoggia a": "abuts", "Taglia": "cuts", "Copre": "covers", "Riempie": "fills"}
ENCLOSED_SHARE = 0.9  # a unit whose outline runs along another for this share of its length, inside it: enclosed


def face_units(chunk, eid: str) -> list[tuple[object, dict]]:
    """The units drawn on the elevation ``eid``: ``(geometry in (u, v), record)``."""
    from shapely import wkt

    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    out = []
    for rec in frame_drawings(chunk, eid).get("units") or []:
        try:
            g = wkt.loads(rec["wkt"]).buffer(0)
        except Exception:  # noqa: BLE001 - a broken record is left out
            continue
        if not g.is_empty and g.area > 0:
            out.append((g, rec))
    return out


def unit_label(rec: dict) -> str:
    return f"{rec.get('tipo') or 'US'} {rec.get('us') or '?'}"


def _tolerance(entry: dict) -> float:
    return max(0.01, 3.0 * float(entry.get("resolution") or 0.0))


def _touching(a, b, tol: float):
    """The part of ``a``'s outline along ``b`` (within ``tol``)."""
    return a.boundary.intersection(b.buffer(tol))


def _u_cover(g) -> float:
    """How much of the ``u`` axis the lines of ``g`` cover (overlaps counted once)."""
    spans = []
    for part in getattr(g, "geoms", [g]):
        if part.is_empty:
            continue
        if part.geom_type in ("LineString", "LinearRing"):
            xy = np.asarray(part.coords)
            for (a, _b), (c, _d) in zip(xy[:-1], xy[1:]):
                spans.append((min(a, c), max(a, c)))
        elif hasattr(part, "geoms"):
            spans.append((0.0, _u_cover(part)))  # a collection inside: its own cover, as a span
    spans.sort()
    total, end = 0.0, -np.inf
    for lo, hi in spans:
        if hi <= end:
            continue
        total += hi - max(lo, end)
        end = hi
    return float(total)


def _rests_on(a, b, tol: float):
    """Whether ``a`` stands above ``b`` touching it along >= 20 % of ``a``'s width: ``(share, contact height)`` or
    None. The contact is the part of ``a``'s outline along ``b`` and below ``a``'s centre."""
    va, vb = a.centroid.y, b.centroid.y
    if va <= vb:
        return None
    from shapely.geometry import box

    bu0, _bv0, bu1, _bv1 = b.bounds  # the contact within b's own width: the tolerance does not run past its corners
    contact = _touching(a, b, tol).intersection(_below(a, va)).intersection(box(bu0, -1e9, bu1, 1e9))
    if contact.is_empty:
        return None
    u0, _v0, u1, _v1 = a.bounds
    share = _u_cover(contact) / max(u1 - u0, 1e-9)
    if share < TOUCH_SHARE:
        return None
    _cu0, c0, _cu1, c1 = contact.bounds
    return share, float((c0 + c1) / 2)


def _below(a, v):
    from shapely.geometry import box

    u0, v0, u1, _v1 = a.bounds
    return box(u0 - 1, v0 - 1, u1 + 1, v)


def _cuts(a, b, courses: list[dict], stones_uv: list) -> dict | None:
    """Whether ``a``'s outline interrupts a course of ``b``: a course (3 stones or more, most of them inside ``b``)
    whose line through its centres, prolonged across ``a``, enters ``a`` - ``a`` stands where the course would go on."""
    from shapely.geometry import LineString, Point

    for c in courses:
        if not c.get("counts"):
            continue
        inside = [p for p in stones_uv if p[2] == c["n"]]
        in_b = [p for p in inside if b.contains(Point(p[0], p[1]))]
        if len(in_b) < max(3, 0.5 * len(inside)):
            continue
        u = np.array([p[0] for p in inside])
        v = np.array([p[1] for p in inside])
        k, q = np.polyfit(u, v, 1) if np.ptp(u) > 1e-6 else (0.0, float(v.mean()))
        ua0, _va0, ua1, _va1 = a.bounds
        lo, hi = min(u.min(), ua0) - 0.01, max(u.max(), ua1) + 0.01
        line = LineString([(lo, k * lo + q), (hi, k * hi + q)])
        if not line.intersects(a.buffer(-1e-6)):
            continue
        # the course is on both sides of a, or ends against it
        near = a.buffer(max(0.05, 0.5 * c["thickness"]))
        if any(near.contains(Point(p[0], p[1])) for p in in_b):
            return {"course": c["n"], "height": c["height"]}
    return None


def _outer(g):
    """``g`` with its holes filled (a wall drawn around the opening it holds)."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    return unary_union([Polygon(p.exterior) for p in getattr(g, "geoms", [g])])


def _enclosed(outer, inner, tol: float) -> float | None:
    """Whether ``inner`` lies within ``outer`` (holes filled) and ``outer`` runs along at least ``ENCLOSED_SHARE`` of
    its outline - an opening blocked inside a wall: the share, or None."""
    if not _outer(outer).buffer(tol).contains(inner):
        return None
    along = inner.boundary.intersection(outer.buffer(tol)).length / max(inner.boundary.length, 1e-12)
    return float(along) if along >= ENCLOSED_SHARE else None


def _covers(a, b, tol: float) -> dict | None:
    """Whether ``a`` covers ``b``: ``a`` lies above ``b`` (its lowest point not below ``b``'s top, within ``tol``), runs
    along at least 90 % of ``b``'s top line (its highest point at each ``u``) and spans ``b``'s whole width."""
    from shapely.geometry import Point, box

    u0, v0, u1, v1 = b.bounds
    au0, av0, au1, _av1 = a.bounds
    if av0 < v1 - tol or au0 > u0 + tol or au1 < u1 - tol:
        return None
    us = np.linspace(u0 + 0.05 * (u1 - u0), u1 - 0.05 * (u1 - u0), 21)
    tops = []
    for u in us:
        cut = b.intersection(box(u - 1e-4, v0 - 1, u + 1e-4, v1 + 1))
        if not cut.is_empty:
            tops.append((u, cut.bounds[3]))
    if len(tops) < 5:
        return None
    grown = a.buffer(tol)
    along = sum(grown.contains(Point(u, v)) for u, v in tops) / len(tops)
    return {"along": float(along)} if along >= 0.9 else None


def propose(chunk, wid: str, eid: str | None = None) -> list[dict]:
    """The relationships the geometry of the units on the face proposes: ``{"unit", "relation", "other", "evidence",
    "unit_rec", "other_rec"}`` (records as drawn: ``us``, ``tipo``, ``area``, ``sito``)."""
    w, entry = walls.need_face(chunk, wid, eid)
    units = face_units(chunk, entry["id"])
    if len(units) < 2:
        raise NotReady(t("Disegna sulla faccia almeno due USM (strumento Disegna US del programma sul prospetto).",
                         "Draw at least two USM on the face (the program's Draw US tool on the elevation)."))
    tol = _tolerance(entry)
    stones_res = walls.results(chunk, w["id"], entry["id"]).get("stones") or {}
    courses = stones_res.get("course_rows") or []
    stones_uv = []
    if courses:
        import json

        path = walls.face_folder(chunk, w["id"], entry["id"]) / "pietre.json"
        if path.exists():
            stones_uv = [(r["u"], r["v"], r["course"]) for r in json.loads(path.read_text())["stones"]]
    out = []
    for i, (a, ra) in enumerate(units):
        for j, (b, rb) in enumerate(units):
            if i == j:
                continue
            found = None
            if _enclosed(a, b, tol) is not None:
                continue  # a wall around an opening blocked in it: the fill speaks for itself (below)
            fill = _enclosed(b, a, tol)
            if fill is not None:
                found = ("Riempie", t(f"{unit_label(ra)} sta dentro {unit_label(rb)}, che la circonda per il "
                                      f"{num(100 * fill, 0)} % del contorno: un'apertura o una lacuna tamponata (il "
                                      "taglio nel muro, se c'è, è un'altra unità)",
                                      f"{unit_label(ra)} lies within {unit_label(rb)}, which surrounds it along "
                                      f"{num(100 * fill, 0)} % of its outline: a blocked opening or gap (the cut in "
                                      "the wall, if any, is another unit)"))
            cut = _cuts(a, b, courses, stones_uv) if stones_uv and found is None else None
            if cut is not None:
                found = ("Taglia", t(f"il filare {cut['course']} (quota {num(cut['height'], 2)}) di {unit_label(rb)} "
                                     f"si interrompe contro {unit_label(ra)}",
                                     f"course {cut['course']} (height {num(cut['height'], 2)}) of {unit_label(rb)} "
                                     f"stops against {unit_label(ra)}"))
            if found is None:
                cov = _covers(a, b, tol)
                if cov is not None:
                    found = ("Copre", t(f"{unit_label(ra)} sta sopra {unit_label(rb)}, ne segue la sommità per il "
                                        f"{num(100 * cov['along'], 0)} % e lo copre per tutta la larghezza",
                                        f"{unit_label(ra)} lies above {unit_label(rb)}, runs along "
                                        f"{num(100 * cov['along'], 0)} % of its top and spans its whole width"))
            if found is None:
                rest = _rests_on(a, b, tol)
                if rest is not None:
                    found = ("Si appoggia a", t(f"{unit_label(ra)} sta sopra {unit_label(rb)} e lo tocca per il "
                                                f"{num(100 * rest[0], 0)} % della sua larghezza (quota "
                                                f"{num(rest[1], 2)})",
                                                f"{unit_label(ra)} stands above {unit_label(rb)} touching it along "
                                                f"{num(100 * rest[0], 0)} % of its width (height {num(rest[1], 2)})"))
            if found is not None:
                out.append({"unit": unit_label(ra), "relation": found[0], "other": unit_label(rb),
                            "evidence": found[1], "unit_rec": _rec(ra), "other_rec": _rec(rb)})
    if not stones_uv:
        for p in out:
            p.setdefault("note", t("senza le pietre misurate la regola «taglia» non si applica",
                                   "without the stones measured the rule «cuts» does not apply"))
    return out


def _rec(rec: dict) -> dict:
    return {k: rec.get(k) for k in ("us", "tipo", "area", "sito")}


def relation_title(rel: str) -> str:
    return t(rel, RELATION_EN.get(rel, rel))


def proposal_text(p: dict) -> str:
    return f"{p['unit']} {relation_title(p['relation']).lower()} {p['other']}: {p['evidence']}"


def save(pa, chunk, wid: str, eid: str | None, relations: list[dict], sito: str | None = None,
         area: str | None = None, media: bool = True, export_folder=None) -> dict:
    """The units of the face as US forms of PyArchInit (made when missing, with the fields drawn), the ``relations``
    accepted by the user added both ways in the program's format (``[['Si appoggia a', '2', '1', 'Sito']]``), and the
    face's drawings (the program's DXF and SVG) registered as media of the site linked to these forms. ``sito``: the
    units' own, else the project's; ``area``: the units' own, else this (default "1"). Returns ``{"created",
    "relations", "media", "skipped"}``."""
    from pyarchinit_photogrammetry.archaeo.stratigraphy import UNIT_FIELDS

    w, entry = walls.need_face(chunk, wid, eid)
    doc = getattr(chunk, "document", None)
    project_site = (doc.meta.get("sito") if doc is not None else None) or None
    units = face_units(chunk, entry["id"])
    created, skipped, links = [], [], []

    def key(rec):
        s = rec.get("sito") or sito or project_site
        a = str(rec.get("area") or area or "1")
        return s, a, str(rec.get("us") or ""), str(rec.get("tipo") or "US")

    for _g, rec in units:
        s, a, us, tipo = key(rec)
        if not us or not s:
            skipped.append(unit_label(rec))
            continue
        if pa.us_record(s, a, us, tipo) is None:
            attrs = rec.get("attributes") or {}
            fields = {k: attrs[k] for k in UNIT_FIELDS if attrs.get(k) not in (None, "")}
            for k in ("d_stratigrafica", "descrizione", "interpretazione", "periodo_iniziale", "fase_iniziale"):
                if attrs.get(k) not in (None, ""):
                    fields[k] = attrs[k]
            pa.create_us(s, a, us, tipo, **fields)
            created.append(f"{tipo} {us}")
        links.append((a, us, tipo))
    written = 0
    for r in relations:
        ra, rb = r["unit_rec"], r["other_rec"]
        sa, aa, ua, ta = key(ra)
        sb, ab, ub, tb = key(rb)
        if not ua or not ub or not sa or not sb:
            continue
        for (s, a, u, tp), rel, (s2, a2, u2, _t2) in (((sa, aa, ua, ta), r["relation"], (sb, ab, ub, tb)),
                                                     ((sb, ab, ub, tb), INVERSE[r["relation"]], (sa, aa, ua, ta))):
            n = pa.add_relationships(s, a, u, [[rel, u2, a2, s2]], tp)
            written += n or 0
    media_ids = []
    if media and links:
        from pathlib import Path

        from pyarchinit_photogrammetry.engine.elevation import export_drawings

        live = walls.elevation(chunk, entry["id"])  # the program's entry: the media ids are kept there
        folder = Path(export_folder) if export_folder else walls.face_folder(chunk, w["id"], entry["id"]) / "pyarchinit"
        paths = export_drawings(chunk, walls.with_bounds(chunk, live), folder)
        media_ids = pa.register_elevation_media(chunk, live, paths, sito=sito or project_site, link_us=links)
    result = {"created": created, "relations": written, "media": media_ids, "skipped": skipped}
    done = walls.results(chunk, w["id"], entry["id"]).get("usm") or {}
    done.update(saved=result, accepted=[{k: r[k] for k in ("unit", "relation", "other")} for r in relations])
    walls.set_result(chunk, w["id"], entry["id"], "usm", done)
    return result


def units_text(chunk, eid: str) -> str:
    units = face_units(chunk, eid)
    if not units:
        return t("Nessuna USM disegnata sulla faccia.", "No USM drawn on the face.")
    parts = []
    for g, rec in units:
        u0, v0, u1, v1 = g.bounds
        parts.append(f"{unit_label(rec)}: {metres(u1 - u0)} × {metres(v1 - v0)}, {num(g.area, 2)} m²")
    return "\n".join(parts)
