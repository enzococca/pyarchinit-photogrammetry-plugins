"""The anomalies read on a geophysical grid: polygons drawn on the raster view of the program, in
its layer *Anomalie geofisiche*, each with the grid and the band it was read on and a note.

They belong to the project: ``chunk.meta["geophysics"]["anomalies"]`` (``id``, ``etichetta``,
``griglia``, ``grid_id``, ``banda``, ``note``, ``coords`` in the coordinate system of the chunk,
``area_m2``, ``made``), exported by :func:`.export.export_gpkg` (table ``anomalie``). The view
shows them as a layer of its own (:class:`.anomaly_view.ViewSync` draws it again when the view shows another
raster; a polygon deleted there with the program's *Select* tool leaves the project).

The program's polygon tool always makes stratigraphic units, and its raster view has no drawing
mode for a plugin: :class:`.anomaly_view.PolygonTool` is the plugin's own, an event filter on the
view that takes the clicks while it is on. This module holds the data only (no Qt): the command
line uses it too.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

LAYER = "Anomalie geofisiche"  # the name of the layer of the view (an identifier, as the program's own)
COLOUR = "#ff7f00"


# ------------------------------------------------------------------ in the project
def entries(chunk) -> list[dict]:
    return list((chunk.meta.get("geophysics") or {}).get("anomalies") or []) if chunk is not None else []


def area(coords) -> float:
    """The area of a polygon (shoelace), in the units of the coordinates squared."""
    pts = np.asarray(coords, float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def next_label(chunk) -> str:
    return f"A{max((int(a.get('id', 0)) for a in entries(chunk)), default=0) + 1}"


def add(chunk, coords, grid: dict | None = None, band: str = "", note: str = "", label: str = "") -> dict:
    """A new anomaly of the chunk: ``coords`` the polygon (x, y) in the coordinate system of the
    chunk, ``grid`` the registry entry of the grid it was read on, ``band`` the label of its band
    (the slice of a GPR survey), ``label`` (default ``A<n>``) and ``note`` the interpretation."""
    meta = chunk.meta.setdefault("geophysics", {"grids": []})
    items = meta.setdefault("anomalies", [])
    n = max((int(a.get("id", 0)) for a in items), default=0) + 1
    pts = [[float(x), float(y)] for x, y in np.asarray(coords, float)[:, :2]]
    if len(pts) < 3:
        raise ValueError("an anomaly needs at least three vertices")
    a = {"id": n, "etichetta": label or f"A{n}", "griglia": (grid or {}).get("name", ""),
         "grid_id": (grid or {}).get("id", ""), "banda": str(band), "note": str(note), "coords": pts,
         "area_m2": area(pts), "made": datetime.now().isoformat(timespec="seconds")}
    items.append(a)
    return a


def remove(chunk, ids) -> int:
    """The anomalies ``ids`` out of the chunk; how many went."""
    ids = set(ids)
    meta = chunk.meta.get("geophysics") or {}
    items = list(meta.get("anomalies") or [])
    keep = [a for a in items if a.get("id") not in ids]
    if items:
        meta["anomalies"] = keep
    return len(items) - len(keep)


def geometry(a: dict):
    from shapely.geometry import Polygon

    return Polygon([tuple(p) for p in a["coords"]])


def attributes(a: dict) -> dict:
    """What a click on the polygon shows (and what the program keeps when it draws its layers again)."""
    return {"id": a["id"], "griglia": a.get("griglia", ""), "banda": a.get("banda", ""), "note": a.get("note", ""),
            "area_m2": round(float(a.get("area_m2", 0.0)), 3)}
