"""The section or the profile of an element as DXF (ezdxf, in the program), in metres at 1:1 for the drawings: a
column's half-profile with its mirror and the axis, a moulding's section, an arch's ring with the springing line, an
opening's plan (reveals and wall faces, measured beside ideal) and its elevation with the sill and the lintel; the
measured points beside the ideal line, on layers of their own."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import engine

LAYERS = {"SOLIDO": 7, "MISURATO": 1, "ASSE": 5, "IMPOSTA": 3, "FACCE": 4, "SOGLIA": 2, "ARCHITRAVE": 6}


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
    elif e["kind"] == "opening":
        _opening_lines(e, out)
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


def opening_frame(e: dict):
    """The opening's own plan frame: ``(centre, along, across)`` - the middle of the outer face, along the wall (the
    axis turned left), across it towards the outer face."""
    lines = e["geometry"]["lines"]
    a = np.asarray(lines["axis"], float)
    left = engine.intersect_lines(*map(np.asarray, lines["left"]), *map(np.asarray, lines["outer"]))
    right = engine.intersect_lines(*map(np.asarray, lines["right"]), *map(np.asarray, lines["outer"]))
    return 0.5 * (left + right), np.array([-a[1], a[0]]), a


def _opening_lines(e: dict, out: dict) -> None:
    """The plan (x along the wall, y across it towards the outside, from the middle of the outer face) and, to its
    right, the elevation seen from outside (x along the wall, y the world height)."""
    g, p, prof = e["geometry"], e["params"], e.get("profile") or {}
    c, along, across = opening_frame(e)

    def local(pts):
        q = np.asarray(pts, float).reshape(-1, 2) - c
        return np.column_stack([q @ along, q @ across])

    plan = local(g["plan"])
    out["SOLIDO"].append((plan, True))
    span = float(np.ptp(plan[:, 0])) + 0.6
    lines = g["lines"]
    for name in ("outer", "inner"):
        pt, d = np.asarray(lines[name][0], float), np.asarray(lines[name][1], float)
        mid = engine.intersect_lines(pt, d, c, across)
        out["FACCE"].append((local([mid - span * d, mid + span * d]), False))
    for key in ("left", "right", "face_outer", "face_inner"):
        pts = np.asarray(prof.get(key) or [], float).reshape(-1, 2)
        if len(pts) > 1:
            q = local(pts)
            order = np.argsort(q[:, 1] if key in ("left", "right") else q[:, 0])
            out["MISURATO"].append((q[order], False))
    out["ASSE"].append((np.array([[0.0, -p["reveal_depth"] - 0.3], [0.0, 0.3]]), False))
    # the elevation, from outside: the outer face's width, the sill and the lintel (the arch's intrados above)
    dx = float(np.ptp(plan[:, 0])) + 1.0 + float(plan[:, 0].max())
    half = p["width"] / 2
    sill, lintel = p["sill"], p["lintel"]
    top = p["springing"] if p.get("top") == "arch" and p.get("springing") is not None else lintel
    rect = np.array([[-half, sill], [half, sill], [half, top]])
    if p.get("top") == "arch" and g.get("void"):
        outline = g["void"]["outline"]
        vs = float(outline["points"][0][1])  # the right springing point comes first
        curve = engine.outline_polygon(outline, arc_points=64)
        curve = curve[curve[:, 1] >= vs - 1e-9]
        mid = 0.5 * (curve[:, 0].min() + curve[:, 0].max())
        arc = np.column_stack([curve[:, 0] - mid, curve[:, 1] - vs + top])
        arc = arc[np.argsort(-arc[:, 0])]
        rect = np.vstack([rect, arc, [[-half, top]]])
    else:
        rect = np.vstack([rect, [[-half, top]]])
    out["SOLIDO"].append((rect + [dx, 0.0], True))
    reach = half + 0.4
    out["SOGLIA"].append((np.array([[dx - reach, sill], [dx + reach, sill]]), False))
    out["ARCHITRAVE"].append((np.array([[dx - reach, lintel], [dx + reach, lintel]]), False))
    if p.get("springing") is not None:
        out["IMPOSTA"].append((np.array([[dx - reach, p["springing"]], [dx + reach, p["springing"]]]), False))


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
