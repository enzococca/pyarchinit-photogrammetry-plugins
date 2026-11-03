"""The records of the plugin, all in ``chunk.meta["conservation"]`` (the program's undo and saving cover them):

* ``campaigns`` - ``{id, elevation, date (YYYY-MM-DD), name, surveyor, method, weather, notes, classes: {cc, uc,
  reason}, made}``: one survey of a face (an elevation of the program) on a date; the EN 16096 condition and urgency
  classes are the surveyor's, never filled by the plugin;
* ``features`` - what is mapped on a face in a campaign: ``{id, campaign, kind (decay|crack|treatment|note), code,
  code_uni, code_iscs (the ICOMOS-ISCS codes of the term), geometry (WKT in (u, v): POLYGON for areas, LINESTRING for
  cracks, POINT for notes), intensity, extent (m2 or m, computed), width_mm, depth_mm, cause, condition, urgency,
  notes, photos, source, status (accepted|proposed), treatment, links: {us}, made}``;
* ``assessments`` - ``{date: {rc, reason, surveyor}}``: the EN 16096 recommendation class of the whole object
  surveyed on that date, one per chunk and date;
* ``legend``, ``local_terms`` - the project's colours and hatches and its own terms (:mod:`.vocabulary`).

Files of a campaign live in ``<chunk data>/conservation/<eid>/<campaign id>/`` and can be made again from the
records. These functions only change the meta: the window marks the project modified."""
from __future__ import annotations

import shutil
import uuid
from datetime import date as _date, datetime
from pathlib import Path

import numpy as np

from .texts import t

FEATURE_KINDS = ("decay", "crack", "treatment", "note")
GEOMETRY_OF = {"decay": ("Polygon", "MultiPolygon"), "crack": ("LineString", "MultiLineString"),
               "treatment": ("Polygon", "MultiPolygon", "LineString", "MultiLineString"), "note": ("Point",)}
WKT_OF = {"decay": "POLYGON", "crack": "LINESTRING", "treatment": "POLYGON or LINESTRING", "note": "POINT"}
INTENSITIES = ("low", "medium", "high")
STATUSES = ("accepted", "proposed")
CLASSES = (0, 1, 2, 3)  # EN 16096: CC, UC and RC 0-3
CAMPAIGN_FIELDS = ("date", "name", "surveyor", "method", "weather", "notes", "classes")
FEATURE_FIELDS = ("kind", "code", "code_uni", "geometry", "intensity", "width_mm", "depth_mm", "cause", "condition",
                  "urgency", "notes", "photos", "source", "status", "treatment", "links")
LINK_SHARE = 0.10  # a unit is linked to a feature it covers for at least this share of the feature's area


class NotReady(RuntimeError):
    """Something a step needs is missing: the message says what."""


def store(chunk) -> dict:
    """``chunk.meta["conservation"]``, made with its empty lists when missing."""
    rec = chunk.meta.setdefault("conservation", {})
    for key, empty in (("campaigns", list), ("features", list), ("assessments", dict), ("legend", dict),
                       ("local_terms", list)):
        if not isinstance(rec.get(key), empty):
            rec[key] = empty()
    return rec


def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _day(value) -> str:
    """``YYYY-MM-DD`` of a date, a datetime or an ISO string."""
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, _date):
        return value.isoformat()
    try:
        return datetime.fromisoformat(str(value).strip()).date().isoformat()
    except ValueError:
        raise ValueError(t(f"Data non valida: {value!r} (AAAA-MM-GG).", f"Not a date: {value!r} (YYYY-MM-DD).")) \
            from None


def _class(value, what: str):
    if value is None or value == "":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = -1
    if n not in CLASSES or float(value) != n:
        raise ValueError(t(f"La classe {what} va da 0 a 3, non {value!r}.",
                           f"The class {what} is 0 to 3, not {value!r}."))
    return n


# ---------------------------------------------------------------- campaigns
def campaigns(chunk, eid: str | None = None) -> list[dict]:
    """The campaigns of the chunk, of the elevation ``eid`` only when given."""
    found = ((chunk.meta.get("conservation") or {}).get("campaigns")) or []
    return [c for c in found if eid is None or c.get("elevation") == eid]


def campaign(chunk, cid: str) -> dict:
    for c in campaigns(chunk):
        if c["id"] == cid:
            return c
    raise NotReady(t(f"Nessuna campagna {cid}.", f"No campaign {cid}."))


def campaign_folder(chunk, eid: str, cid: str) -> Path:
    """``<chunk data>/conservation/<eid>/<cid>/``: the files of a campaign (not made here)."""
    return Path(chunk.data_dir) / "conservation" / eid / cid


def _elevation(chunk, eid: str) -> dict:
    for entry in chunk.meta.get("elevations") or []:
        if entry.get("id") == eid:
            return entry
    raise NotReady(t(f"Il prospetto {eid} non esiste in questo chunk.", f"The elevation {eid} is not in this chunk."))


def new_campaign(chunk, eid: str, date=None, name: str | None = None, surveyor: str = "", method: str = "",
                 weather: str = "", notes: str = "") -> dict:
    """A campaign on the elevation ``eid``; ``date`` default: the date of the chunk (its own, else of its photos),
    else today."""
    from pyarchinit_photogrammetry.archaeo.timeline import phase_date

    _elevation(chunk, eid)
    if date is None:
        when = phase_date(chunk)
        day = when.date().isoformat() if when is not None else _date.today().isoformat()
    else:
        day = _day(date)
    rec = {"id": uuid.uuid4().hex[:8], "elevation": eid, "date": day,
           "name": name or t(f"Campagna {day}", f"Campaign {day}"), "surveyor": surveyor, "method": method,
           "weather": weather, "notes": notes, "classes": {"cc": None, "uc": None, "reason": ""}, "made": _now()}
    store(chunk)["campaigns"].append(rec)
    return rec


def set_campaign(chunk, cid: str, **fields) -> dict:
    """Change the fields of a campaign (``classes`` merged: ``{"cc": 2}`` keeps ``uc`` and ``reason``)."""
    rec = campaign(chunk, cid)
    wrong = set(fields) - set(CAMPAIGN_FIELDS)
    if wrong:
        raise ValueError(t(f"campi di una campagna: {', '.join(CAMPAIGN_FIELDS)}; non {', '.join(sorted(wrong))}",
                           f"fields of a campaign: {', '.join(CAMPAIGN_FIELDS)}; not {', '.join(sorted(wrong))}"))
    if "classes" in fields:
        classes = dict(rec.get("classes") or {"cc": None, "uc": None, "reason": ""})
        given = dict(fields.pop("classes") or {})
        if set(given) - {"cc", "uc", "reason"}:
            raise ValueError(t("le classi di una campagna sono cc, uc e reason",
                               "the classes of a campaign are cc, uc and reason"))
        for key in ("cc", "uc"):
            if key in given:
                classes[key] = _class(given[key], key.upper())
        if "reason" in given:
            classes["reason"] = str(given["reason"] or "")
        rec["classes"] = classes
    if "date" in fields:
        fields["date"] = _day(fields["date"])
    rec.update(fields)
    return rec


def remove_campaign(chunk, cid: str, files: bool = True) -> None:
    """A campaign taken away with its features and, with ``files``, its folder."""
    rec = campaign(chunk, cid)
    st = store(chunk)
    st["campaigns"] = [c for c in st["campaigns"] if c["id"] != cid]
    st["features"] = [f for f in st["features"] if f.get("campaign") != cid]
    if files:
        shutil.rmtree(campaign_folder(chunk, rec["elevation"], cid), ignore_errors=True)


def assessment(chunk, date) -> dict | None:
    """The EN 16096 recommendation class of the object surveyed on ``date`` (``{rc, reason, surveyor}``)."""
    return ((chunk.meta.get("conservation") or {}).get("assessments") or {}).get(_day(date))


def set_assessment(chunk, date, rc, reason: str = "", surveyor: str = "") -> dict:
    rec = {"rc": _class(rc, "RC"), "reason": reason, "surveyor": surveyor}
    store(chunk)["assessments"][_day(date)] = rec
    return rec


# ---------------------------------------------------------------- features
def _checked(chunk, kind: str, code: str, geometry):
    """``(term, geometry, wkt, extent)`` of a feature, or ValueError naming what was expected."""
    import shapely

    from .geometry import as_geometry
    from .vocabulary import load

    if kind not in FEATURE_KINDS:
        raise ValueError(t(f"il tipo di un elemento è uno fra {', '.join(FEATURE_KINDS)}, non {kind!r}",
                           f"the kind of a feature is one of {', '.join(FEATURE_KINDS)}, not {kind!r}"))
    term = load(chunk).term(code)
    if kind == "note":
        if term.kind == "treatment":
            raise ValueError(t(f"{term.code} è un intervento: serve il tipo 'treatment'",
                               f"{term.code} is a treatment: kind 'treatment' expected"))
    elif term.kind != kind:
        raise ValueError(t(f"{term.code} è di tipo {term.kind!r}, non {kind!r}",
                           f"{term.code} is a {term.kind}: kind {term.kind!r} expected, not {kind!r}"))
    geom = as_geometry(geometry)
    if geom is None or geom.is_empty:
        raise ValueError(t("la geometria dell'elemento è vuota", "the geometry of a feature is empty"))
    if geom.geom_type not in GEOMETRY_OF[kind]:
        raise ValueError(t(f"un elemento di tipo {kind!r} è un {WKT_OF[kind]}, non un {geom.geom_type.upper()}",
                           f"a feature of kind {kind!r} is a {WKT_OF[kind]}, not a {geom.geom_type.upper()}"))
    text = shapely.to_wkt(geom, rounding_precision=6, trim=True)
    geom = shapely.from_wkt(text)
    if geom.geom_type in ("Polygon", "MultiPolygon"):
        extent = float(geom.area)
    elif geom.geom_type in ("LineString", "MultiLineString"):
        extent = float(geom.length)
    else:
        extent = None
    return term, geom, text, extent


def _optional(value, name: str):
    if value is None or value == "":
        return None
    v = float(value)
    if not np.isfinite(v) or v < 0:
        raise ValueError(t(f"{name} è un numero positivo di millimetri, non {value!r}",
                           f"{name} is a positive number of millimetres, not {value!r}"))
    return v


def _check_fields(rec: dict) -> None:
    from .vocabulary import UNNAMED

    if rec.get("intensity") not in (None, *INTENSITIES):
        raise ValueError(t(f"l'intensità è una fra {', '.join(INTENSITIES)}, non {rec['intensity']!r}",
                           f"the intensity is one of {', '.join(INTENSITIES)}, not {rec['intensity']!r}"))
    if rec.get("status") not in STATUSES:
        raise ValueError(t(f"lo stato è uno fra {', '.join(STATUSES)}, non {rec.get('status')!r}",
                           f"the status is one of {', '.join(STATUSES)}, not {rec.get('status')!r}"))
    if rec.get("status") == "accepted" and rec.get("code") == UNNAMED:
        raise ValueError(t("un elemento ancora da nominare non può essere accettato: dagli prima un termine",
                           "a feature still to be named cannot be accepted: give it a term first"))
    rec["condition"] = _class(rec.get("condition"), "CC")
    rec["urgency"] = _class(rec.get("urgency"), "UC")
    rec["width_mm"] = _optional(rec.get("width_mm"), "width_mm")
    rec["depth_mm"] = _optional(rec.get("depth_mm"), "depth_mm")


def iscs_codes(term) -> list[str] | None:
    """The ICOMOS-ISCS codes of a term: its mapping for a UNI term, itself for an ISCS term, None for local terms and
    interventions."""
    if term.iscs:
        return list(term.iscs)
    return [term.code] if term.code.startswith("ISCS.") else None


def add_feature(chunk, cid: str, kind: str, code: str, geometry, status: str = "accepted", source: str = "drawn",
                code_uni: str | None = None, intensity: str | None = None, width_mm=None, depth_mm=None,
                cause: str = "", condition=None, urgency=None, notes: str = "", photos=None,
                treatment: dict | None = None) -> dict:
    """A feature of the campaign ``cid``: ``geometry`` a shapely geometry or its WKT in ``(u, v)`` of the campaign's
    elevation; the term's code (a sub-type by its short form too) stored in full, ``code_uni`` from the vocabulary's
    mapping when not given; ``extent`` the area (m2) of a polygon, the length (m) of a line. ValueError when the kind
    does not match the term or the geometry, or when an accepted feature would carry the working term
    ``PPG.UNNAMED`` (a proposal may)."""
    campaign(chunk, cid)
    term, _geom, text, extent = _checked(chunk, kind, code, geometry)
    rec = {"id": uuid.uuid4().hex[:8], "campaign": cid, "kind": kind, "code": term.code,
           "code_uni": code_uni if code_uni is not None else term.uni, "code_iscs": iscs_codes(term),
           "geometry": text, "intensity": intensity,
           "extent": extent, "width_mm": width_mm, "depth_mm": depth_mm, "cause": cause, "condition": condition,
           "urgency": urgency, "notes": notes, "photos": list(photos or []), "source": source, "status": status,
           "treatment": dict(treatment) if treatment is not None else None, "links": {"us": []}, "made": _now()}
    _check_fields(rec)
    store(chunk)["features"].append(rec)
    return rec


def features(chunk, cid: str | None = None, eid: str | None = None, status: str | None = None) -> list[dict]:
    """The features of the chunk; of a campaign, of an elevation, of a status when given."""
    found = ((chunk.meta.get("conservation") or {}).get("features")) or []
    of_face = {c["id"] for c in campaigns(chunk, eid)} if eid is not None else None
    return [f for f in found if (cid is None or f.get("campaign") == cid)
            and (of_face is None or f.get("campaign") in of_face) and (status is None or f.get("status") == status)]


def feature(chunk, fid: str) -> dict:
    for f in features(chunk):
        if f["id"] == fid:
            return f
    raise NotReady(t(f"Nessun elemento {fid}.", f"No feature {fid}."))


def update_feature(chunk, fid: str, **fields) -> dict:
    """Change the fields of a feature; a new geometry, term or kind is checked again and gives the extent again (a
    new term without ``code_uni`` takes its own)."""
    rec = feature(chunk, fid)
    wrong = set(fields) - set(FEATURE_FIELDS)
    if wrong:
        raise ValueError(t(f"campi di un elemento: {', '.join(FEATURE_FIELDS)}; non {', '.join(sorted(wrong))}",
                           f"fields of a feature: {', '.join(FEATURE_FIELDS)}; not {', '.join(sorted(wrong))}"))
    new = dict(rec)
    new.update(fields)
    if {"geometry", "code", "kind"} & set(fields):
        term, _geom, text, extent = _checked(chunk, new["kind"], new["code"], new["geometry"])
        new.update(code=term.code, geometry=text, extent=extent)
        if "code" in fields and "code_uni" not in fields:
            new["code_uni"] = term.uni
        new["code_iscs"] = iscs_codes(term)
    _check_fields(new)
    rec.update(new)
    return rec


def remove_feature(chunk, fid: str) -> None:
    feature(chunk, fid)
    st = store(chunk)
    st["features"] = [f for f in st["features"] if f["id"] != fid]


def accept(chunk, fids) -> int:
    """Proposed features accepted into their campaign (their ``source`` still says what proposed them). NotReady,
    with nothing accepted, when one of them still carries the working term ``PPG.UNNAMED``."""
    from .texts import t
    from .vocabulary import UNNAMED

    fids = set(fids)
    unnamed = [f for f in features(chunk) if f["id"] in fids and f["code"] == UNNAMED]
    if unnamed:
        raise NotReady(t(f"{len(unnamed)} elementi sono ancora da nominare: dai un termine prima di accettare "
                         "(Modifica..., o Come questo).", f"{len(unnamed)} features are still to be named: give "
                         "them a term before accepting (Edit..., or Like this)."))
    n = 0
    for f in features(chunk):
        if f["id"] in fids:
            f["status"] = "accepted"
            n += 1
    return n


def discard(chunk, fids) -> int:
    """Features (proposed ones, usually) taken away."""
    fids = set(fids)
    st = store(chunk)
    before = len(st["features"])
    st["features"] = [f for f in st["features"] if f["id"] not in fids]
    return before - len(st["features"])


def link_units(chunk, entry: dict, fid: str) -> list[tuple]:
    """The units drawn on the elevation (the program's ``frame_drawings(chunk, eid)["units"]``) that cover at least a
    tenth of the feature's area: ``[(area, us, unita_tipo)]``, kept in ``feature["links"]["us"]``."""
    from shapely import wkt

    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    rec = feature(chunk, fid)
    geom = wkt.loads(rec["geometry"])
    found = []
    if geom.area > 0:
        for unit in frame_drawings(chunk, entry["id"]).get("units") or []:
            try:
                g = wkt.loads(unit["wkt"]).buffer(0)
            except Exception:  # noqa: BLE001 - a broken drawing is left out
                continue
            if g.is_empty or g.intersection(geom).area < LINK_SHARE * geom.area:
                continue
            tipo = unit.get("tipo") or unit.get("unita_tipo") or "US"
            found.append((str(unit.get("area") or ""), str(unit.get("us") or ""), str(tipo)))
    rec.setdefault("links", {})["us"] = [list(x) for x in found]
    return found


def face_area(chunk, entry: dict) -> float | None:
    """The surface of the face: the cells of the elevation's relief with a value times the cell area (m2)."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    path = elevation_paths(chunk, entry)[1]
    if not path.exists():
        return None
    n = 0
    with rasterio.open(path) as src:
        cell = abs(src.transform.a * src.transform.e)
        for _ij, window in src.block_windows(1):
            block = src.read(1, window=window)
            ok = np.isfinite(block)
            if src.nodata is not None and np.isfinite(src.nodata):
                ok &= block != src.nodata
            n += int(ok.sum())
    return n * cell


def totals(chunk, cid: str) -> dict:
    """What a campaign maps (accepted features only): ``area_by_code`` and ``area_by_family`` of the decay areas
    (m2); ``area_by_iscs`` the same areas on the first ICOMOS-ISCS code of each feature (a UNI and an ISCS drawing of
    one phenomenon meet there; local terms keep their own code); ``length_by_code`` of the cracks (m),
    ``treatment_area_by_code`` / ``treatment_length_by_code``, ``face_area`` (m2) and ``share_of_face`` = the union of
    the areas of deterioration (not the general terms nor a patina "without visible deterioration"), within the
    face's bounds, / the face's surface (at most 1); the widest crack and the deepest loss (mm)."""
    from shapely import wkt
    from shapely.geometry import box
    from shapely.ops import unary_union

    from pyarchinit_photogrammetry.engine.elevation import bounds_of

    from .vocabulary import UNNAMED, deteriorates, load

    rec = campaign(chunk, cid)
    vocab = load(chunk)
    out = {"area_by_code": {}, "area_by_iscs": {}, "area_by_family": {}, "length_by_code": {},
           "treatment_area_by_code": {}, "treatment_length_by_code": {}, "face_area": None, "share_of_face": None,
           "crack_max_width_mm": None, "loss_max_depth_mm": None, "features": 0}
    areas = []

    def add(table, key, value):
        out[table][key] = out[table].get(key, 0.0) + float(value)

    for f in features(chunk, cid, status="accepted"):
        if f["code"] == UNNAMED:  # a proposal never named: not part of the survey
            continue
        out["features"] += 1
        g = wkt.loads(f["geometry"])
        polygon = g.geom_type in ("Polygon", "MultiPolygon")
        if f["kind"] == "decay" and polygon:
            add("area_by_code", f["code"], g.area)
            add("area_by_iscs", (f.get("code_iscs") or [f["code"]])[0], g.area)
            known = f["code"] in vocab
            add("area_by_family", vocab.term(f["code"]).family if known else "masonry", g.area)
            if not known or deteriorates(vocab, vocab.term(f["code"])):
                areas.append(g)
        elif f["kind"] == "crack":
            add("length_by_code", f["code"], g.length)
        elif f["kind"] == "treatment":
            add("treatment_area_by_code" if polygon else "treatment_length_by_code", f["code"],
                g.area if polygon else g.length)
        if f["kind"] == "crack" and f.get("width_mm") is not None:
            out["crack_max_width_mm"] = max(out["crack_max_width_mm"] or 0.0, float(f["width_mm"]))
        if f["kind"] == "decay" and f.get("depth_mm") is not None:
            out["loss_max_depth_mm"] = max(out["loss_max_depth_mm"] or 0.0, float(f["depth_mm"]))
    entry = next((e for e in chunk.meta.get("elevations") or [] if e.get("id") == rec["elevation"]), None)
    if entry is not None:
        out["face_area"] = face_area(chunk, entry)
        if out["face_area"]:
            mapped = unary_union(areas).intersection(box(*bounds_of(chunk, entry))).area if areas else 0.0
            out["share_of_face"] = min(1.0, mapped / out["face_area"])
    return out


def photos_of(chunk, entry: dict, feature: dict, limit: int | None = 2) -> list[dict]:
    """The aligned photos that see the feature's centre (on the surface through the relief, else on the plane):
    ``[{"key", "label", "px", "py", "distance" (m), "angle" (degrees from the face's normal)}]``, those where the
    point falls nearest the photo's centre first; empty when the chunk has no aligned cameras."""
    from shapely import wkt

    from pyarchinit_photogrammetry.engine.elevation import world_of
    from pyarchinit_photogrammetry.markers.gcp import photos_seeing

    from .geometry import frame_of

    if not chunk.aligned_cameras:
        return []
    g = wkt.loads(feature["geometry"])
    if g.geom_type == "Point":
        centre = g
    elif g.geom_type in ("LineString", "MultiLineString"):
        centre = g.interpolate(0.5, normalized=True)
    else:
        centre = g.centroid if g.contains(g.centroid) else g.representative_point()
    u, v = float(centre.x), float(centre.y)
    world = world_of(entry, u, v, None, chunk)
    if world is None:
        world = world_of(entry, u, v, 0.0)
    normal = frame_of(chunk, entry).view(np.array([u]), np.array([v]))[0]
    out = []
    for cam, px, py, _depth in photos_seeing(chunk, chunk.to_internal(world[None])[0]):
        towards = chunk.to_world(cam.center[None])[0] - world
        distance = float(np.linalg.norm(towards))
        cos = float(towards @ normal) / max(distance * float(np.linalg.norm(normal)), 1e-12)
        out.append({"key": cam.key, "label": cam.label, "px": float(px), "py": float(py), "distance": distance,
                    "angle": float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))})
        if limit is not None and len(out) >= limit:
            break
    return out
