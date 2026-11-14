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
out of the layer: its coordinates are the old ones; its campaigns stay in the tabs."""
from __future__ import annotations

import logging
import re

import numpy as np

from . import records
from .texts import t

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


def _layer(chunk, vocab, legend) -> dict | None:
    from shapely import wkt

    from .geometry import frame_of
    from .outputs import INTENSITY_WORDS, _family_name, _word, glossary_names

    feats, codes = [], []
    for c in records.campaigns(chunk):
        try:
            entry = records._elevation(chunk, c["elevation"])
        except records.NotReady:
            continue
        if not _same_system(chunk, entry):
            LOG.info("conservation: the elevation %s is in another system than the chunk: not on the site's plan",
                     entry.get("name") or entry.get("id"))
            continue
        frame = frame_of(chunk, entry)
        for f in _named(chunk, c["id"]):
            if f["kind"] != "decay":
                continue
            g = wkt.loads(f["geometry"])
            if g.geom_type not in ("Polygon", "MultiPolygon"):
                continue
            names = glossary_names(vocab, f["code"], f.get("code_iscs"))
            props = {"code": f["code"], "term_it": names["it"], "term_en": names["en"],
                     "family": _family_name(vocab, names["family"]), "extent": _r(f.get("extent")),
                     "intensity": _word(INTENSITY_WORDS, f.get("intensity")),
                     "elevation": entry.get("name") or entry.get("id"), "campaign": c["name"], "date": c["date"]}
            props = {k: v for k, v in props.items() if v not in (None, "")}
            for geom in place(frame, g):
                feats.append({"type": "Feature", "geometry": geom, "properties": props})
            if f["code"] not in codes:
                codes.append(f["code"])
    if not feats:
        return None
    items = []
    for code in codes:
        names = glossary_names(vocab, code)
        items.append({"value": code, "title": {"it": names["it"], "en": names["en"]},
                      "colour": _colour(legend.get(code))})
    return {"kind": "layer", "id": LAYER_ID, "title": TITLES[LAYER_ID], "features": feats, "colour": GREY,
            "legend": {"field": "code", "items": items}, "fields": LAYER_FIELDS}


def _tables(chunk, vocab) -> tuple[list, list]:
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
            summary.append({"elevation": name, "campaign": c["name"], "date": c["date"],
                            "kind": _word(KIND_WORDS, kind), "code": code, "term_it": names["it"],
                            "term_en": names["en"], "features": g["n"], "extent": _r(g["extent"]),
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
    from .vocabulary import legend, load

    if not records.campaigns(chunk):
        return []
    vocab = load(chunk)
    out = []
    layer = _layer(chunk, vocab, legend(chunk))
    if layer is not None:
        out.append(layer)
    summary, campaigns = _tables(chunk, vocab)
    if summary:
        out.append({"kind": "table", "id": SUMMARY_ID, "title": TITLES[SUMMARY_ID],
                    "columns": _columns(SUMMARY_COLUMNS, summary), "rows": summary})
    if campaigns:
        out.append({"kind": "table", "id": CAMPAIGNS_ID, "title": TITLES[CAMPAIGNS_ID],
                    "columns": _columns(CAMPAIGN_COLUMNS, campaigns), "rows": campaigns})
    return out


__all__ = ["contributions", "place", "LAYER_ID", "SUMMARY_ID", "CAMPAIGNS_ID"]
