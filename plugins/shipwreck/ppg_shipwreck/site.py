"""The timber catalogue on the web site of a survey (program 1.10.0 or later: ``context.add_site_contributions``).

For every chunk that goes on a site, :func:`contributions` gives the program:

* a layer of the plan, ``timbers``: every timber of the catalogue with its outline as it was drawn or traced (a line,
  a polygon or a point, in the coordinates of the chunk), coloured by its kind (keel, frame, plank...: the colours of
  the plugin's layer on the plan) with a tick per kind in the list of layers, its attributes on a click;
* a tab of cards, ``catalogue``: a card per timber with its attributes and its measures - the position in the ship
  frame, the length, the width and the heights over the model.

The program moves the outlines into the frame of the site (shifted in the local one) and cleans every text; the
measures are in the ship frame, which tells nothing of the place: ``x`` along the keel from its start, ``y`` across
it, the heights from the start of the keel line (also when the ship frame has a datum of its own: an absolute height
never goes out). A chunk without a catalogue gives nothing; a ship frame set before the chunk's system changed gives
the catalogue without its layer and its measures (the outlines are in the old coordinates)."""
from __future__ import annotations

import logging

from . import ship, timbers
from .texts import t

LOG = logging.getLogger("ppg_shipwreck")

LAYER_ID = "timbers"
TABLE_ID = "catalogue"
TITLES = {LAYER_ID: {"it": "Legni del relitto", "en": "Timbers of the wreck"},
          TABLE_ID: {"it": "Catalogo dei legni", "en": "Timber catalogue"}}
OTHER_COLOUR = "#b0b0b0"
# the attributes of a timber: key, title on the site, also on the plan (the measures only in the catalogue)
FIELDS = (("name", {"it": "Nome", "en": "Name"}, True),
          ("kind", {"it": "Tipo", "en": "Kind"}, True),
          ("side", {"it": "Lato", "en": "Side"}, True),
          ("station", {"it": "Stazione", "en": "Station"}, True),
          ("status", {"it": "Stato", "en": "Status"}, True),
          ("loose", {"it": "Fuori posto", "en": "Out of place"}, True),
          ("x", {"it": "x lungo la chiglia (m)", "en": "x along the keel (m)"}, False),
          ("y", {"it": "y di traverso (m)", "en": "y across (m)"}, False),
          ("length", {"it": "Lunghezza (m)", "en": "Length (m)"}, True),
          ("width", {"it": "Larghezza (m)", "en": "Width (m)"}, True),
          ("z_min", {"it": "Quota minima (m)", "en": "Lowest height (m)"}, False),
          ("z_max", {"it": "Quota massima (m)", "en": "Highest height (m)"}, False),
          ("z_min_water", {"it": "Quota minima corretta dall'acqua (m)",
                           "en": "Lowest height corrected for the water (m)"}, False),
          ("z_max_water", {"it": "Quota massima corretta dall'acqua (m)",
                           "en": "Highest height corrected for the water (m)"}, False),
          ("notes", {"it": "Note", "en": "Notes"}, False),
          ("warning", {"it": "Avviso", "en": "Warning"}, False))
DECIMALS = 3


def _stale(chunk) -> bool:
    """The ship frame (and the outlines drawn with it) is in another system than the chunk's."""
    d = (chunk.meta.get(ship.KEY) or {}).get("frame")
    return d is not None and d.get("crs", chunk.crs) != chunk.crs


def _measures(chunk) -> dict:
    """``{id: row of timbers.measure}``, the heights from the start of the keel line; {} without a ship frame (or
    one in another system)."""
    frame = ship.frame_of(chunk)
    if frame is None or _stale(chunk):
        return {}
    try:
        rows = timbers.measure(chunk)
    except ship.NotReady:
        return {}
    lift = frame.z0 - frame.start[2]  # 0 with the default datum: the heights stay those of the catalogue
    for row in rows:
        for key in ("z_min", "z_max", "z_min_water", "z_max_water"):
            if row.get(key) is not None:
                row[key] = row[key] + lift
    return {r["id"]: r for r in rows}


def _value(value):
    if isinstance(value, float):
        return round(value, DECIMALS)
    return "" if value is None else value


def _attributes(record: dict, row: dict | None) -> dict:
    out = {"name": record.get("name") or record["id"], "kind": timbers.kind_title(record["kind"]),
           "side": timbers.side_title(record["side"]), "station": record.get("station") or "",
           "status": timbers.status_title(timbers.status_of(record)),
           "loose": t("sì", "yes") if record.get("loose") else "", "notes": record.get("notes") or ""}
    for key in ("x", "y", "length", "width", "z_min", "z_max", "z_min_water", "z_max_water", "warning"):
        out[key] = _value((row or {}).get(key))
    return {k: v for k, v in out.items() if v != ""}


def _geometry(geom: dict) -> dict | None:
    xy = [[float(x), float(y)] for x, y in geom.get("xy") or []]
    kind = geom.get("type")
    if kind == "point" and xy:
        return {"type": "Point", "coordinates": xy[0]}
    if kind == "line" and len(xy) >= 2:
        return {"type": "LineString", "coordinates": xy}
    if kind == "polygon" and len(xy) >= 3:
        ring = xy if xy[0] == xy[-1] else xy + [xy[0]]
        return {"type": "Polygon", "coordinates": [ring]}
    return None


def contributions(chunk) -> list[dict]:
    """What the plugin adds to the web site for ``chunk`` (see the module): the layer ``timbers`` and the tab
    ``catalogue``; [] without a catalogue."""
    records = timbers.timbers_of(chunk)
    if not records:
        return []
    rows = _measures(chunk)
    out = []
    if not _stale(chunk):
        features, kinds = [], []
        for r in records:
            g = _geometry(r.get("geometry") or {})
            if g is None:
                continue
            props = _attributes(r, rows.get(r["id"]))
            features.append({"type": "Feature", "geometry": g,
                             "properties": {k: v for k, v in props.items()
                                            if any(k == f[0] and f[2] for f in FIELDS)}})
            if r["kind"] not in kinds:
                kinds.append(r["kind"])
        if features:
            from . import KIND_COLOURS

            order = [k for k in timbers.KINDS if k in kinds] + [k for k in kinds if k not in timbers.KINDS]
            legend = [{"value": timbers.kind_title(k),
                       "title": dict(zip(("it", "en"), timbers.KIND_TITLES.get(k, (k, k)))),
                       "colour": KIND_COLOURS.get(k, OTHER_COLOUR)} for k in order]
            out.append({"kind": "layer", "id": LAYER_ID, "title": TITLES[LAYER_ID], "features": features,
                        "colour": OTHER_COLOUR, "legend": {"field": "kind", "items": legend},
                        "fields": {key: title for key, title, on_plan in FIELDS if on_plan}})
    else:
        LOG.info("shipwreck: the ship frame of %s is in another system than the chunk: no layer on the site",
                 chunk.label)
    table = [{"id": r["id"], **_attributes(r, rows.get(r["id"]))} for r in records]
    present = {k for row in table for k in row}
    columns = [{"key": "id", "title": {"it": "N.", "en": "No."}}] + \
        [{"key": key, "title": title} for key, title, _ in FIELDS if key in present]
    out.append({"kind": "table", "id": TABLE_ID, "title": TITLES[TABLE_ID], "columns": columns, "rows": table})
    return out


__all__ = ["contributions", "LAYER_ID", "TABLE_ID"]
