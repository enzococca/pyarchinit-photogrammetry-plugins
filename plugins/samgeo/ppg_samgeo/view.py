"""The runs on the 2D view of the program: each a layer ``samgeo: <name>`` drawn from its GeoPackage -
on every raster the view shows, and after every change of the run - its objects with their number,
text, score and area as attributes (shown when clicked, in the live QGIS link); those already made
units are outlines. Qt only: the window imports this module, the command line does not."""
from __future__ import annotations

import numpy as np

from .runs import features, layer, records


def draw(view, chunk, run: dict) -> int:
    """The layer of ``run`` drawn again (taken away when the run is hidden); returns the objects
    drawn (none when the view shows nothing)."""
    if view is None:
        return 0
    name = layer(run)
    if name in view.layers:
        view.remove_layer(name)
    if view.tr is None or not run.get("visible", True):
        return 0
    rows = features(chunk, run)
    made = set(run.get("units_made") or [])
    with view.batch():
        for geom, a in rows:
            n = int(a.get("n") or 0)
            attrs = {"samgeo": run["id"], "n": n, "prompt": a.get("prompt", ""), "label": a.get("label", ""),
                     "score": round(float(a.get("score") or 0.0), 3), "area_m2": round(float(a.get("area_m2") or 0.0), 4),
                     "us": n in made}
            for part in getattr(geom, "geoms", [geom]):  # one feature per part, as the units
                view.add_polygon(np.asarray(part.exterior.coords)[:, :2], label="", color=run.get("colour", "#ff8c00"),
                                 fill_alpha=0 if n in made else 60, layer=name, attrs=attrs)
    return len(rows)


def draw_all(view, chunk) -> None:
    """The layers of every run of the chunk."""
    for run in records(chunk):
        draw(view, chunk, run)


def selected(view, run: dict) -> set[int]:
    """The numbers of the objects of ``run`` selected on the view (the program's *Seleziona*)."""
    if view is None:
        return set()
    name = layer(run)
    return {int(attrs.get("n") or 0) for lay, _i, attrs in view.selected_features() if lay == name} - {0}
