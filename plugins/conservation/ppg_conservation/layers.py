"""The features of a campaign drawn on the window's raster view while it shows their elevation.

The program does not keep a plugin's layers: they are drawn again whenever the view loads the elevation, on the
window's refresh and after every change, and taken away when the view leaves the elevation. Each feature's attributes
carry ``fid`` and ``code``, so a deletion with the window's *Seleziona* + Canc maps back to the records."""
from __future__ import annotations

import numpy as np

LAYERS = {"decay": "Degrado", "crack": "Lesioni", "treatment": "Interventi", "note": "Note", "proposed": "Proposte"}
FILL_ALPHA = round(0.35 * 255)
PROPOSED_ALPHA = round(0.15 * 255)
# matplotlib hatches of the legend -> the Qt brush drawn over the fill on the view
QT_HATCH = (("x", "DiagCrossPattern"), ("+", "CrossPattern"), ("/", "BDiagPattern"), ("\\", "FDiagPattern"),
            ("|", "VerPattern"), ("-", "HorPattern"), (".", "Dense6Pattern"), ("o", "Dense6Pattern"),
            ("O", "Dense5Pattern"), ("*", "Dense5Pattern"))


def hatch_style(hatch: str):
    from PySide6.QtCore import Qt

    for sign, style in QT_HATCH:
        if sign in (hatch or ""):
            return getattr(Qt.BrushStyle, style)
    return None


def shown_eid(window) -> str | None:
    """The elevation the window's view shows, None on the plan (or without a window)."""
    if window is None:
        return None
    eid = getattr(window, "drawings_frame", None)
    version = getattr(window, "_raster_version", None)
    if eid is None or (version is not None and version != ("elevation", eid)):
        return None
    return eid


def drop(view) -> None:
    """The plugin's layers taken away from ``view``."""
    if view is not None:
        for layer in LAYERS.values():
            view.remove_layer(layer)


def dress(item, style: dict, kind: str, proposed: bool, done: bool) -> None:
    """The hatch of the term over the fill (a child of the polygon: it goes with it), a dashed outline for a
    proposal and for a treatment still to do."""
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QBrush, QColor, QPen
    from PySide6.QtWidgets import QGraphicsPathItem

    hatch = hatch_style((style.get("hatch") or {}).get("mpl", ""))
    if hatch is not None and not proposed:
        child = QGraphicsPathItem(item.path(), item)
        child.setPen(QPen(Qt.PenStyle.NoPen))
        child.setBrush(QBrush(QColor(style["colour"]), hatch))
    if proposed or (kind == "treatment" and not done):
        pen = QPen(item.pen())
        pen.setStyle(Qt.PenStyle.DashLine)
        item.setPen(pen)


def draw(view, chunk, cid: str) -> bool:
    """The layers of the campaign ``cid`` on ``view`` (which shows its elevation): polygons filled with the legend's
    colour and its hatch, lines in the term's colour, notes as discs; the proposals apart, pale and marked "?".
    Returns whether anything could be drawn (the view has a raster)."""
    from shapely import get_parts, wkt

    from . import records
    from .geometry import disc
    from .vocabulary import legend

    drop(view)
    if view is None or view.tr is None:
        return False
    colours = legend(chunk)
    with view.batch():
        for f in records.features(chunk, cid):
            style = colours.get(f["code"], {"colour": "#808080", "hatch": {"mpl": ""}})
            colour = style["colour"]
            proposed = f["status"] == "proposed"
            layer = LAYERS["proposed" if proposed else f["kind"]]
            attrs = {"fid": f["id"], "code": f["code"]}
            label = "?" if proposed else ""
            geom = wkt.loads(f["geometry"])
            if f["kind"] == "note":
                geom = disc(geom)
            done = bool((f.get("treatment") or {}).get("done"))
            for part in get_parts(geom):
                if part.geom_type == "Polygon":
                    xy = np.asarray(part.exterior.coords)[:-1]
                    item = view.add_polygon(xy, label, colour, PROPOSED_ALPHA if proposed else FILL_ALPHA,
                                            layer=layer, attrs=attrs)
                    dress(item, style, f["kind"], proposed, done)
                elif part.geom_type == "LineString":
                    view.add_line(np.asarray(part.coords), colour, label, layer=layer, attrs=attrs, width=3.0,
                                  dash=proposed or (f["kind"] == "treatment" and not done))
    return True
