"""The decay mapping on the web site of a survey (program 1.10.0 or later: ``context.add_site_contributions``).

For every chunk that goes on a site, :func:`contributions` gives the program what the campaigns of its elevations
hold (only the accepted features with a term, as every output of the plugin):

* a layer of the plan, ``decay``: the areas of decay of every campaign, coloured by their term with the project's
  legend (a tick per term in the list of layers), their attributes on a click. The areas are drawn on the elevation in
  its ``(u, v)``; on the plan they go where the elevation lies, as the program draws the trace of an elevation: on a
  face that lies down (a floor, a pavement, a sloping face) the area itself, placed on the plane of the elevation; on
  an upright face or a cylinder (a wall, a tower) the stretch of the wall it covers, a line along the trace of the
  face at the middle height of the area;
* a tab of cards, ``summary``: per campaign, every term mapped with its features, extent (m² of the areas, m of the
  cracks and of the linear treatments) and share of the face;
* a tab of cards, ``campaigns``: every campaign with its date, method, the share of the face with decay, the widest
  crack and the EN 16096 classes chosen by the surveyor (the name of the surveyor is not published).

The program moves the coordinates into the frame of the site (shifted in the local one, the heights too) and cleans
every text. An elevation made in another system than the chunk's (the chunk georeferenced again afterwards) is left
out of the layer: its coordinates are the old ones; its campaigns stay in the tabs.

With a program whose hook tells the frame of the site (``context.site_hook`` 2, :func:`contributions_with_site`), the
areas of decay go on the drawing of their elevation in the tab *Sezioni e prospetti*, in its ``(u, v)`` (one layer
``decay`` per elevation on the site, with the lowest and the highest height of each area in the heights of the site
when the elevation keeps them); only the campaigns of an elevation the site does not publish stay on the plan. The
terms, their families, the kinds and the intensities are given in both languages, the page shows those of its own.
"""
from __future__ import annotations

import logging
import re

import numpy as np

from . import records

LOG = logging.getLogger("ppg_conservation")

LAYER_ID = "decay"
SUMMARY_ID = "summary"
CAMPAIGNS_ID = "campaigns"
TITLES = {LAYER_ID: {"it": "Degrado", "en": "Decay"},
          SUMMARY_ID: {"it": "Degrado: riepilogo", "en": "Decay: summary"},
          CAMPAIGNS_ID: {"it": "Campagne di conservazione", "en": "Conservation campaigns"}}
GREY = "#808080"
LYING = 0.7  # a plane whose v axis rises less than this lies down (the program's test for the trace of an elevation)
ARC_STEP = 0.05  # metres between the points of a stretch on a cylinder
DECIMALS = 3
LAYER_FIELDS = {"code": {"it": "Codice", "en": "Code"},
                "term_it": {"it": "Termine", "en": "Term (Italian)"},
                "term_en": {"it": "Termine (inglese)", "en": "Term"},
                "family": {"it": "Famiglia", "en": "Family"},
                "extent": {"it": "Area (m²)", "en": "Area (m²)"},
                "intensity": {"it": "Intensità", "en": "Intensity"},
                "elevation": {"it": "Prospetto", "en": "Elevation"},
                "campaign": {"it": "Campagna", "en": "Campaign"},
                "date": {"it": "Data", "en": "Date"}}
LAYER_FIELDS_BOTH = {"code": {"it": "Codice", "en": "Code"},
                     "term": {"it": "Termine", "en": "Term"},
                     "family": {"it": "Famiglia", "en": "Family"},
                     "extent": {"it": "Area (m²)", "en": "Area (m²)"},
                     "intensity": {"it": "Intensità", "en": "Intensity"},
                     "z_min": {"it": "Quota minima (m)", "en": "Lowest height (m)"},
                     "z_max": {"it": "Quota massima (m)", "en": "Highest height (m)"},
                     "elevation": {"it": "Prospetto", "en": "Elevation"},
                     "campaign": {"it": "Campagna", "en": "Campaign"},
                     "date": {"it": "Data", "en": "Date"}}
SUMMARY_COLUMNS = (("elevation", {"it": "Prospetto", "en": "Elevation"}),
                   ("campaign", {"it": "Campagna", "en": "Campaign"}),
                   ("date", {"it": "Data", "en": "Date"}),
                   ("kind", {"it": "Tipo", "en": "Kind"}),
                   ("code", {"it": "Codice", "en": "Code"}),
                   ("term_it", {"it": "Termine", "en": "Term (Italian)"}),
                   ("term_en", {"it": "Termine (inglese)", "en": "Term"}),
                   ("features", {"it": "Elementi", "en": "Features"}),
                   ("extent", {"it": "Estensione", "en": "Extent"}),
                   ("unit", {"it": "Unità", "en": "Unit"}),
                   ("share", {"it": "Quota della faccia (%)", "en": "Share of the face (%)"}))
SUMMARY_COLUMNS_BOTH = tuple(("term", {"it": "Termine", "en": "Term"}) if k == "term_it" else (k, v)
                             for k, v in SUMMARY_COLUMNS if k != "term_en")  # the term as one value in two languages
CAMPAIGN_COLUMNS = (("elevation", {"it": "Prospetto", "en": "Elevation"}),
                    ("campaign", {"it": "Campagna", "en": "Campaign"}),
                    ("date", {"it": "Data", "en": "Date"}),
                    ("method", {"it": "Metodo", "en": "Method"}),
                    ("features", {"it": "Elementi", "en": "Features"}),
                    ("share", {"it": "Faccia con degrado (%)", "en": "Face with decay (%)"}),
                    ("crack_mm", {"it": "Lesione più aperta (mm)", "en": "Widest crack (mm)"}),
                    ("cc", {"it": "Classe di condizione (EN 16096)", "en": "Condition class (EN 16096)"}),
                    ("uc", {"it": "Classe di urgenza (EN 16096)", "en": "Urgency class (EN 16096)"}),
                    ("reason", {"it": "Motivo", "en": "Reason"}),
                    ("rc", {"it": "Classe di raccomandazione (EN 16096)",
                            "en": "Recommendation class (EN 16096)"}),
                    ("notes", {"it": "Note", "en": "Notes"}))


def _both(pair) -> dict:
    """A pair (Italian, English) of the plugin's words as a value of the site in both languages."""
    return {"it": pair[0], "en": pair[1]}


def _r(value, d: int = DECIMALS):
    return None if value is None else round(float(value), d)


def _colour(style: dict | None) -> str:
    colour = (style or {}).get("colour")
    return colour if isinstance(colour, str) and re.fullmatch(r"#[0-9a-fA-F]{6}", colour) else GREY


def _same_system(chunk, entry) -> bool:
    given = entry.get("crs")
    if not given:
        return True
    from pyarchinit_photogrammetry.engine.elevation import same_crs

    return same_crs(given, chunk.crs)


def _lies_down(frame) -> bool:
    return frame.kind == "plane" and abs(float(frame.v_axis[2])) < LYING


def place(frame, polygon) -> list[dict]:
    """GeoJSON geometries on the plan (3D, in the coordinates of the chunk) of a polygon in ``(u, v)`` of the elevation
    ``frame`` (see the module): the polygons placed on a plane that lies down, else a line along the face."""
    parts = [polygon] if polygon.geom_type == "Polygon" else list(getattr(polygon, "geoms", []))
    out = []
    for p in parts:
        if p.is_empty:
            continue
        if _lies_down(frame):
            rings = [np.asarray(p.exterior.coords)] + [np.asarray(r.coords) for r in p.interiors]
            out.append({"type": "Polygon", "coordinates": [
                frame.point(r[:, 0], r[:, 1], np.zeros(len(r))).tolist() for r in rings]})
            continue
        u0, _v0, u1, _v1 = p.bounds
        v = float(p.centroid.y)
        n = 2 if frame.kind == "plane" else int(max(2, min(256, np.ceil((u1 - u0) / ARC_STEP) + 1)))
        us = np.linspace(u0, u1, n)
        out.append({"type": "LineString", "coordinates": frame.point(us, np.full(n, v), np.zeros(n)).tolist()})
    return out


def _named(chunk, cid: str) -> list[dict]:
    from .vocabulary import UNNAMED

    return [f for f in records.features(chunk, cid, status="accepted") if f["code"] != UNNAMED]


def _props(vocab, f, entry, c, both: bool) -> dict:
    """The attributes of an area of decay on the site: in the interface's language, or (``both``) the words in both
    languages and the term as one value."""
    from .outputs import INTENSITY_WORDS, _family_name, _word, glossary_names
    from .vocabulary import FAMILIES

    names = glossary_names(vocab, f["code"], f.get("code_iscs"))
    if both:
        family = names["family"]
        props = {"code": f["code"], "term": {"it": names["it"], "en": names["en"]},
                 "family": _both(FAMILIES[family]) if family in FAMILIES else family,
                 "intensity": _both(INTENSITY_WORDS[f["intensity"]]) if f.get("intensity") in INTENSITY_WORDS
                 else f.get("intensity")}
    else:
        props = {"code": f["code"], "term_it": names["it"], "term_en": names["en"],
                 "family": _family_name(vocab, names["family"]),
                 "intensity": _word(INTENSITY_WORDS, f.get("intensity"))}
    props.update(extent=_r(f.get("extent")), elevation=entry.get("name") or entry.get("id"), campaign=c["name"],
                 date=c["date"])
    return props


def _legend(vocab, legend, codes) -> dict:
    from .outputs import glossary_names

    items = []
    for code in codes:
        names = glossary_names(vocab, code)
        items.append({"value": code, "title": {"it": names["it"], "en": names["en"]},
                      "colour": _colour(legend.get(code))})
    return {"field": "code", "items": items}


def _areas(chunk):
    """``(entry of the elevation, campaign, feature, polygon in (u, v))`` of every accepted area of decay."""
    from shapely import wkt

    for c in records.campaigns(chunk):
        try:
            entry = records._elevation(chunk, c["elevation"])
        except records.NotReady:
            continue
        for f in _named(chunk, c["id"]):
            if f["kind"] != "decay":
                continue
            g = wkt.loads(f["geometry"])
            if g.geom_type in ("Polygon", "MultiPolygon"):
                yield entry, c, f, g


def _layer(chunk, vocab, legend, both: bool = False, skip=()) -> dict | None:
    """The layer of the plan; ``skip``: the ids of the elevations left out (on the site's elevations instead)."""
    from .geometry import frame_of

    feats, codes, other = [], [], set()
    for entry, c, f, g in _areas(chunk):
        if entry.get("id") in skip:
            continue
        if not _same_system(chunk, entry):
            if entry.get("id") not in other:
                LOG.info("conservation: the elevation %s is in another system than the chunk: not on the site's "
                         "plan", entry.get("name") or entry.get("id"))
                other.add(entry.get("id"))
            continue
        props = {k: v for k, v in _props(vocab, f, entry, c, both).items() if v not in (None, "")}
        for geom in place(frame_of(chunk, entry), g):
            feats.append({"type": "Feature", "geometry": geom, "properties": props})
        if f["code"] not in codes:
            codes.append(f["code"])
    if not feats:
        return None
    return {"kind": "layer", "id": LAYER_ID, "title": TITLES[LAYER_ID], "features": feats, "colour": GREY,
            "legend": _legend(vocab, legend, codes), "fields": LAYER_FIELDS_BOTH if both else LAYER_FIELDS}


def _elevation_layers(chunk, vocab, legend, site) -> list[dict]:
    """A layer ``decay`` on every elevation of the site with areas of decay, in its ``(u, v)``; the lowest and the
    highest height of an area in the heights of the site when the elevation keeps them (``site.uv``)."""
    from shapely.geometry import mapping

    published = {e["elevation"]: e for e in site.elevations}
    by: dict = {}
    for entry, c, f, g in _areas(chunk):
        e = published.get(entry.get("id"))
        if e is None:
            continue
        props = _props(vocab, f, entry, c, True)
        if e["heights"]:
            _u0, v0, _u1, v1 = g.bounds
            props.update(z_min=site.uv(e["elevation"], 0.0, v0)[1], z_max=site.uv(e["elevation"], 0.0, v1)[1])
        feats, codes = by.setdefault(e["elevation"], ([], []))
        feats.append({"type": "Feature", "geometry": mapping(g),
                      "properties": {k: v for k, v in props.items() if v not in (None, "")}})
        if f["code"] not in codes:
            codes.append(f["code"])
    return [{"kind": "layer", "id": LAYER_ID, "elevation": eid, "title": TITLES[LAYER_ID], "features": feats,
             "colour": GREY, "legend": _legend(vocab, legend, codes), "fields": LAYER_FIELDS_BOTH}
            for eid, (feats, codes) in by.items()]


def _tables(chunk, vocab, both: bool = False) -> tuple[list, list]:
    from shapely import wkt

    from .outputs import KIND_WORDS, _word, glossary_names

    summary, campaigns = [], []
    for c in records.campaigns(chunk):
        try:
            entry = records._elevation(chunk, c["elevation"])
        except records.NotReady:
            continue
        name = entry.get("name") or entry.get("id")
        tot = records.totals(chunk, c["id"])
        face = tot.get("face_area") or 0.0
        groups: dict = {}
        for f in _named(chunk, c["id"]):
            if f["kind"] == "note":
                continue
            area = wkt.loads(f["geometry"]).geom_type in ("Polygon", "MultiPolygon")
            key = (f["kind"], f["code"], area)
            g = groups.setdefault(key, {"n": 0, "extent": 0.0, "iscs": f.get("code_iscs")})
            g["n"] += 1
            g["extent"] += float(f.get("extent") or 0.0)
        rank = {"decay": 0, "crack": 1, "treatment": 2}
        for (kind, code, area), g in sorted(groups.items(), key=lambda kv: (rank.get(kv[0][0], 3), -kv[1]["extent"])):
            names = glossary_names(vocab, code, g["iscs"])
            words = ({"kind": _both(KIND_WORDS[kind]) if kind in KIND_WORDS else kind,
                      "term": {"it": names["it"], "en": names["en"]}} if both else
                     {"kind": _word(KIND_WORDS, kind), "term_it": names["it"], "term_en": names["en"]})
            summary.append({"elevation": name, "campaign": c["name"], "date": c["date"], **words,
                            "code": code, "features": g["n"], "extent": _r(g["extent"]),
                            "unit": "m²" if area else "m",
                            "share": _r(100.0 * g["extent"] / face, 2) if kind == "decay" and area and face else None})
        classes = c.get("classes") or {}
        rc = (records.assessment(chunk, c["date"]) or {}).get("rc")
        campaigns.append({"elevation": name, "campaign": c["name"], "date": c["date"], "method": c.get("method"),
                          "features": tot.get("features"),
                          "share": _r(100.0 * tot["share_of_face"], 2) if tot.get("share_of_face") is not None
                          else None,
                          "crack_mm": _r(tot.get("crack_max_width_mm"), 2), "cc": classes.get("cc"),
                          "uc": classes.get("uc"), "reason": classes.get("reason"), "rc": rc,
                          "notes": c.get("notes")})
    return _filled(summary), _filled(campaigns)


def _filled(rows: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if v not in (None, "")} for r in rows]


def _columns(spec, rows) -> list[dict]:
    present = {k for r in rows for k in r}
    return [{"key": key, "title": title} for key, title in spec if key in present]


def contributions(chunk) -> list[dict]:
    """What the plugin adds to the web site for ``chunk`` (see the module): the layer ``decay`` and the tabs
    ``summary`` and ``campaigns``; [] without a campaign."""
    return _contributions(chunk, None)


def contributions_with_site(chunk, site) -> list[dict]:
    """The same for a program that tells the frame of the site (``context.site_hook`` 2): the areas on the drawings
    of the elevations the site publishes, in their ``(u, v)``, the plan only for the others; words in both
    languages."""
    return _contributions(chunk, site)


def _contributions(chunk, site) -> list[dict]:
    from .vocabulary import legend, load

    if not records.campaigns(chunk):
        return []
    vocab = load(chunk)
    colours = legend(chunk)
    both = site is not None
    out = _elevation_layers(chunk, vocab, colours, site) if both else []
    layer = _layer(chunk, vocab, colours, both, skip={x["elevation"] for x in out})
    if layer is not None:
        out.append(layer)
    summary, campaigns = _tables(chunk, vocab, both)
    if summary:
        spec = SUMMARY_COLUMNS_BOTH if both else SUMMARY_COLUMNS
        out.append({"kind": "table", "id": SUMMARY_ID, "title": TITLES[SUMMARY_ID],
                    "columns": _columns(spec, summary), "rows": summary})
    if campaigns:
        out.append({"kind": "table", "id": CAMPAIGNS_ID, "title": TITLES[CAMPAIGNS_ID],
                    "columns": _columns(CAMPAIGN_COLUMNS, campaigns), "rows": campaigns})
    return out


__all__ = ["contributions", "contributions_with_site", "place", "LAYER_ID", "SUMMARY_ID", "CAMPAIGNS_ID"]
