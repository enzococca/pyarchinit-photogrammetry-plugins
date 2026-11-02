"""The section or the profile of an element as DXF (ezdxf, in the program), in metres at 1:1 for the drawings: a
column's half-profile with its mirror and the axis, a moulding's section, an arch's ring with the springing line; the
measured points beside the ideal line, on layers of their own."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import engine

LAYERS = {"SOLIDO": 7, "MISURATO": 1, "ASSE": 5, "IMPOSTA": 3}


def lines_of(e: dict) -> dict:
    """The polylines of the drawing of an element, by layer: ``{layer: [(N, 2) arrays, closed?]}``."""
    out: dict = {k: [] for k in LAYERS}
    g, p, prof = e["geometry"], e.get("params") or {}, e.get("profile") or {}
    if e["kind"] == "column":
        c = np.asarray(g["profile"], float)
        half = np.vstack([[0.0, c[0, 1]], c, [0.0, c[-1, 1]]])
        out["SOLIDO"].append((half, False))
        out["SOLIDO"].append((np.column_stack([-half[:, 0], half[:, 1]]), False))
        tail = 0.05 * p["height"]
        out["ASSE"].append((np.array([[0.0, c[0, 1] - tail], [0.0, c[-1, 1] + tail]]), False))
        m = np.column_stack([prof["radii"], prof["heights"]])
        out["MISURATO"].append((m, False))
    elif e["kind"] == "moulding":
        out["SOLIDO"].append((np.asarray(g["section"], float), True))
        for ln in prof.get("cut") or []:
            out["MISURATO"].append((np.asarray(ln, float), False))
    else:
        out["SOLIDO"].append((engine.outline_polygon(g["outline"]), True))
        intr = np.asarray(prof.get("intrados") or [], float).reshape(-1, 2)
        if len(intr):
            order = np.argsort(np.arctan2(intr[:, 1] - p["springing"], intr[:, 0] - p["centre_u"]))
            out["MISURATO"].append((intr[order], False))
        span = p["span"]
        out["IMPOSTA"].append((np.array([[p["centre_u"] - span / 2 - p["thickness"], p["springing"]],
                                         [p["centre_u"] + span / 2 + p["thickness"], p["springing"]]]), False))
        out["ASSE"].append((np.array([[p["centre_u"], p["springing"]],
                                      [p["centre_u"], p["springing"] + p["rise"] + p["thickness"]]]), False))
    return out


def write_dxf(e: dict, path) -> Path:
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    for name, colour in LAYERS.items():
        doc.layers.add(name, color=colour)
    msp = doc.modelspace()
    for layer, items in lines_of(e).items():
        for pts, closed in items:
            if len(pts) < 2:
                continue
            msp.add_lwpolyline([tuple(map(float, q)) for q in pts], close=closed, dxfattribs={"layer": layer})
    msp.add_text(e.get("name") or "", dxfattribs={"layer": "SOLIDO", "height": 0.05})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path
