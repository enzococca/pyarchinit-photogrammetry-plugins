"""Geoarchaeological sections: along a trace (a polyline drawn on the view, typed, or through
boreholes) the profile of the ground and of the top of every unit, the units filled between them,
and the boreholes within a distance of the trace projected onto it with their logs.

A section is drawn at a scale as the program's own sections are (the most detailed standard
scale fitting about an A0 sheet when none is given) with a vertical exaggeration (chosen so that
the drawing is not a thin strip when none is given): an SVG and a PNG with matplotlib, a DXF in
section coordinates (distance along the trace, elevation; true elevations), a CSV of the
profile. Files in ``<chunk data>/geoarchaeology/sections/``, registry in
``chunk.meta["geoarchaeology"]["sections"]``.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np

from . import boreholes as bh
from . import implicit
from . import surfaces as sf
from .figures import save
from .texts import signed, t

EXAGGERATIONS = (1, 2, 5, 10, 20, 50, 100)
MAX_MM = 5000.0  # widest drawing (a scale given that makes it wider is reduced)
SCALES = (20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000)


def folder(chunk) -> Path:
    return bh.folder(chunk) / "sections"


def drawing_scale(width_m: float, height_m: float = 0.0) -> int:
    """The program's choice of scale (1:20 for a trench, 1:200 across a tell...)."""
    try:
        from pyarchinit_photogrammetry.archaeo.sections import drawing_scale as program_scale

        return int(program_scale(width_m, height_m))
    except Exception:  # noqa: BLE001 - an older program
        size = max(width_m, height_m)
        return next((s for s in SCALES if size * 1000.0 / s <= 1000.0), SCALES[-1])


# ------------------------------------------------------------------ geometry of the trace
def polyline(coords) -> np.ndarray:
    p = np.asarray(coords, float).reshape(-1, 2)
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-9]
    p = p[keep]
    if len(p) < 2:
        raise ValueError(t("La traccia di una sezione vuole almeno due punti diversi",
                           "The trace of a section needs at least two different points"))
    return p


def stations(p: np.ndarray, step: float) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step`` metres along the polyline, its vertices included: ``(distance, xy)``."""
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    cum = np.r_[0.0, np.cumsum(seg)]
    d = np.unique(np.r_[np.arange(0.0, cum[-1], max(step, 1e-6)), cum])
    k = np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(seg) - 1)
    f = (d - cum[k]) / np.where(seg[k] > 0, seg[k], 1.0)
    return d, p[k] + f[:, None] * (p[k + 1] - p[k])


def project(p: np.ndarray, xy) -> tuple[np.ndarray, np.ndarray]:
    """Distance along the polyline of the nearest point to each of ``xy`` and the signed offset
    from it (positive on the left of the direction of the trace)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    seg = np.diff(p, axis=0)
    length = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(length)]
    best_d = np.full(len(xy), np.inf)
    station = np.zeros(len(xy))
    offset = np.zeros(len(xy))
    for k in range(len(seg)):
        u = seg[k] / max(length[k], 1e-12)
        rel = xy - p[k]
        along = np.clip(rel @ u, 0.0, length[k])
        foot = p[k] + along[:, None] * u
        dist = np.linalg.norm(xy - foot, axis=1)
        better = dist < best_d - 1e-12
        best_d[better] = dist[better]
        station[better] = cum[k] + along[better]
        offset[better] = np.sign(u[0] * rel[better, 1] - u[1] * rel[better, 0]) * dist[better]
    return station, offset


# ------------------------------------------------------------------ the profile
def profile(chunk, coords, step: float | None = None) -> dict:
    """The ground and the top of every unit along the trace: ``{"distance", "x", "y", "ground",
    "tops": [(unit, values)], "length"}`` (NaN where a surface has no value). When the model of the
    chunk is the implicit one the tops are its surfaces (a section across a pit shows the pit) and
    ``"method"`` says so."""
    p = polyline(coords)
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    grid = sf.grid_of(chunk)
    if step is None:
        step = min(grid.cell if grid else length / 500.0, length / 500.0)
        step = max(step, length / 5000.0)
    d, xy = stations(p, step)
    dem = bh.dem_path(chunk)
    ground = bh.raster_values(dem, xy[:, 0], xy[:, 1]) if dem is not None else np.full(len(d), np.nan)
    tops = []
    implicit_tops = implicit.active_surfaces(chunk)
    for e in implicit_tops or sf.registry(chunk).get("contacts", []):
        path = Path(chunk.data_dir) / e["file"]
        values = bh.raster_values(path, xy[:, 0], xy[:, 1]) if path.exists() else np.full(len(d), np.nan)
        tops.append((e["unit"], values))
    if dem is None and tops:  # no DEM: the top of the first unit stands for the ground
        ground = tops[0][1].copy()
    out = {"distance": d, "x": xy[:, 0], "y": xy[:, 1], "ground": ground, "tops": tops, "length": length,
           "vertices": p}
    if implicit_tops:
        out["method"] = "implicit"
    return out


def near(chunk, coords, distance: float) -> list[dict]:
    """The boreholes within ``distance`` of the trace: ``{"borehole", "station", "offset"}`` along it."""
    p = polyline(coords)
    holes = bh.load(chunk)
    if not holes:
        return []
    station, offset = project(p, [(b.x, b.y) for b in holes])
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    out = [{"borehole": b, "station": float(s), "offset": float(o)} for b, s, o in zip(holes, station, offset)
           if abs(o) <= distance and -1e-9 <= s <= length + 1e-9]
    return sorted(out, key=lambda r: r["station"])


def default_distance(chunk) -> float:
    holes = bh.load(chunk)
    xy = np.array([[b.x, b.y] for b in holes], float) if holes else np.zeros((0, 2))
    return 0.5 * sf.default_buffer(xy) if len(xy) >= 2 else 5.0


def through(chunk, ids) -> np.ndarray:
    """The trace through the boreholes ``ids``, in that order."""
    by = {b.id: b for b in bh.load(chunk)}
    missing = [i for i in ids if str(i) not in by]
    if missing:
        raise ValueError(t("Sondaggi sconosciuti: {0}", "Unknown boreholes: {0}").format(", ".join(map(str, missing))))
    return np.array([[by[str(i)].x, by[str(i)].y] for i in ids], float)


# ------------------------------------------------------------------ building
def build(chunk, coords, name: str | None = None, distance: float | None = None, scale: float | None = None,
          exaggeration: float | None = None, step: float | None = None, progress=None) -> dict:
    """The section along ``coords``: profile, boreholes within ``distance`` (half the usual spacing
    of the boreholes by default), figure (SVG, PNG), DXF and CSV; returns its registry entry, with
    ``warnings`` when a format could not be written (the SVG in an installation without its
    matplotlib backend: the PNG is there)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    m = bh.meta(chunk)
    sections = m.setdefault("sections", [])
    if not name:
        n = len(sections) + 1
        while any(s["name"] == t("Sezione {0}", "Section {0}").format(n) for s in sections):
            n += 1
        name = t("Sezione {0}", "Section {0}").format(n)
    distance = float(distance) if distance is not None else default_distance(chunk)
    progress(0.1, "Profile")
    prof = profile(chunk, coords, step)
    holes = near(chunk, coords, distance)
    progress(0.4, "Drawing")
    out = folder(chunk)
    out.mkdir(parents=True, exist_ok=True)
    stem = sf.slug(name).lower()
    files = {}
    info = {}
    try:
        info = figure(chunk, prof, holes, name, out / f"{stem}.svg", out / f"{stem}.png", scale, exaggeration)
        files.update({p.suffix.lstrip(".").lower(): p.name for p in info["files"]})
    except ImportError:  # no matplotlib: the DXF and the CSV still
        info = {"scale": scale or drawing_scale(prof["length"]), "exaggeration": exaggeration or 1}
    progress(0.75, "DXF and CSV")
    write_dxf(chunk, prof, holes, name, out / f"{stem}.dxf")
    write_csv(prof, out / f"{stem}.csv")
    files.update(dxf=f"{stem}.dxf", csv=f"{stem}.csv")
    entry = {"id": stem, "name": name, "coords": [[float(x), float(y)] for x, y in prof["vertices"]],
             "length": prof["length"], "distance": distance, "scale": info.get("scale"),
             "exaggeration": info.get("exaggeration"), "files": {k: str((out / v).relative_to(chunk.data_dir))
                                                                 for k, v in files.items()},
             "boreholes": [{"id": h["borehole"].id, "station": round(h["station"], 3),
                            "offset": round(h["offset"], 3) + 0.0} for h in holes],
             "made": time.strftime("%Y-%m-%d %H:%M")}
    if prof.get("method"):
        entry["method"] = prof["method"]
    m["sections"] = [s for s in sections if s["id"] != stem] + [entry]
    try:
        from .export import write_layers

        write_layers(chunk)
    except Exception:  # noqa: BLE001 - the GeoPackage of the layers is a convenience
        pass
    progress(1.0, f"{name}: {prof['length']:.1f} m, {len(holes)} boreholes")
    if info.get("notes"):  # a format this installation cannot write: said to the user, not kept
        return {**entry, "warnings": list(info["notes"])}
    return entry


def entries(chunk) -> list[dict]:
    return list((chunk.meta.get("geoarchaeology") or {}).get("sections", []))


def remove(chunk, sid: str | None = None) -> int:
    m = bh.meta(chunk)
    keep, gone = [], []
    for s in m.get("sections", []):
        (gone if sid is None or s["id"] == sid or s["name"] == sid else keep).append(s)
    for s in gone:
        for rel in s.get("files", {}).values():
            p = Path(chunk.data_dir) / rel
            if p.exists():
                p.unlink()
    m["sections"] = keep
    return len(gone)


# ------------------------------------------------------------------ outputs
def _lower(prof: dict, holes: list[dict]) -> tuple[float, float]:
    vals = [prof["ground"]] + [v for _u, v in prof["tops"]]
    z = np.concatenate([np.asarray(v, float).ravel() for v in vals])
    for h in holes:
        b = h["borehole"]
        if b.z is not None:
            z = np.r_[z, b.z, b.z - b.bottom]
    z = z[np.isfinite(z)]
    if z.size == 0:
        return 0.0, 1.0
    lo, hi = float(z.min()), float(z.max())
    pad = max(0.05 * (hi - lo), 0.05)
    return lo - pad, hi + pad


def figure(chunk, prof: dict, holes: list[dict], name: str, svg: Path | None, png: Path | None,
           scale: float | None = None, exaggeration: float | None = None) -> dict:
    """The section drawn at 1:``scale`` with ``exaggeration`` (None: chosen), as SVG and PNG
    (``files``: those written; ``notes``: the formats this installation cannot write)."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.patches import Patch, Rectangle

    length = prof["length"]
    zlo, zhi = _lower(prof, holes)
    height = zhi - zlo
    if not exaggeration:
        exaggeration = next((e for e in EXAGGERATIONS if height * e >= length / 6.0), EXAGGERATIONS[-1])
    exaggeration = float(exaggeration)
    if not scale:
        scale = drawing_scale(length, height * exaggeration)
    scale = float(scale)
    while max(length, height * exaggeration) * 1000.0 / scale > MAX_MM:
        scale = next((s for s in SCALES if s > scale), scale * 2)
    k = 1000.0 / scale  # metres -> millimetres of paper
    aw, ah = max(length * k, 40.0), max(height * exaggeration * k, 25.0)
    zlo = zhi - ah / (exaggeration * k)  # a drawing at its least size shows more ground, at the same scale
    m_left, m_right, m_bottom, m_top = 22.0, 62.0, 16.0, 16.0
    W, H = aw + m_left + m_right, ah + m_bottom + m_top
    fig = Figure(figsize=(W / 25.4, H / 25.4))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([m_left / W, m_bottom / H, aw / W, ah / H])
    ax.set_xlim(0, aw / k)
    ax.set_ylim(zlo, zhi)
    d = prof["distance"]
    colours = bh.colours(chunk)
    units = {u["code"]: u for u in bh.units(chunk)}
    tops = prof["tops"]
    handles = []
    for i, (unit, top) in enumerate(tops):
        base = tops[i + 1][1] if i + 1 < len(tops) else np.full_like(top, zlo)
        where = np.isfinite(top) & np.isfinite(base) & (top > base + 1e-6)
        colour = colours.get(unit, "#bbbbbb")
        info = units.get(unit, {})
        hatch = bh.hatch_of(info.get("lithology", ""), info.get("description", ""))
        if where.any():
            ax.fill_between(d, np.where(where, base, zlo), np.where(where, top, zlo), where=where, facecolor=colour,
                            edgecolor=_darker(colour), hatch=hatch or None, linewidth=0.0, alpha=0.9, zorder=1)
        ax.plot(d, top, color="#3a3a3a", linewidth=0.5, zorder=2)
        label = unit + (f" - {info.get('description')[:34]}" if info.get("description") else "")
        handles.append(Patch(facecolor=colour, edgecolor=_darker(colour), hatch=hatch or None, label=label))
    ax.plot(d, prof["ground"], color="black", linewidth=1.3, zorder=4, label=t("terreno", "ground"))
    width_m = 2.5 / k  # a column 2.5 mm wide on the paper
    for h in holes:
        b, s = h["borehole"], h["station"]
        if b.z is None:
            continue
        for iv in b.intervals:
            colour = colours.get(iv.unit, "#bbbbbb")
            ax.add_patch(Rectangle((s - width_m / 2, b.z - iv.base), width_m, iv.base - iv.top, facecolor=colour,
                                   edgecolor="black", linewidth=0.4, hatch=bh.hatch_of(iv.lithology, iv.description) or None,
                                   zorder=5))
        ax.plot([s - width_m, s + width_m], [b.z - b.bottom] * 2, color="black", linewidth=0.8, zorder=6)
        ax.annotate(f"{b.id}\n{signed(h['offset'], 1)} m", (s, b.z), xytext=(0, 4), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6, zorder=7)
    ax.set_xlabel(t("Distanza lungo la traccia (m)", "Distance along the trace (m)"), fontsize=7)
    ax.set_ylabel(t("Quota (m)", "Elevation (m)"), fontsize=7)
    ax.tick_params(labelsize=6)
    ax.grid(alpha=0.25, linewidth=0.4)
    title = f"{name} - 1:{scale:g}"
    if exaggeration != 1:
        title += t(", esagerazione verticale x{0:g}", ", vertical exaggeration x{0:g}").format(exaggeration)
    if prof.get("method") == "implicit":
        title += " - " + t("modello implicito: un'interpretazione dei dati", "implicit model: an interpretation of the data")
    fig.text(m_left / W, 1 - 6 / H, title, fontsize=8, va="top")
    if handles:
        fig.legend(handles=handles, loc="upper left", bbox_to_anchor=((m_left + aw + 4) / W, 1 - m_top / H),
                   fontsize=6, frameon=False, title=t("Unità", "Units"), title_fontsize=7)
    written = save(fig, [p for p in (svg, png) if p is not None], png_dpi=min(150.0, 12000.0 / (W / 25.4)))
    return {"scale": scale, "exaggeration": exaggeration, "width_mm": W, "height_mm": H, "files": list(written),
            "notes": written.notes}


def _darker(colour: str, f: float = 0.55) -> str:
    c = colour.lstrip("#")
    r, g, b = (int(int(c[i:i + 2], 16) * f) for i in (0, 2, 4))
    return f"#{r:02x}{g:02x}{b:02x}"


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``(start, stop)`` of the runs of True of a 1-D mask."""
    m = np.r_[False, np.asarray(mask, bool), False]
    edges = np.flatnonzero(np.diff(m.astype(np.int8)))
    return list(zip(edges[::2], edges[1::2]))


def write_dxf(chunk, prof: dict, holes: list[dict], name: str, path: Path) -> Path:
    """DXF in metres, section coordinates (x = distance, y = elevation): the ground, the top of
    every unit, the units as solid hatches of their colour, the boreholes as columns, a grid."""
    import ezdxf

    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    doc.header["$LWDISPLAY"] = 1
    msp = doc.modelspace()
    for layer, colour, weight in (("GEO_TERRENO", 7, 50), ("GEO_CONTATTI", 8, 25), ("GEO_SONDAGGI", 7, 18),
                                  ("GEO_TESTI", 1, 18), ("GEO_GRIGLIA", 9, 13)):
        doc.layers.add(layer, color=colour, lineweight=weight)
    d = prof["distance"]
    colours = bh.colours(chunk)
    tops = prof["tops"]
    zlo, zhi = _lower(prof, holes)
    for i, (unit, top) in enumerate(tops):
        layer = f"GEO_UNITA_{sf.slug(unit)}"
        if layer not in doc.layers:
            doc.layers.add(layer, color=7)
        rgb = tuple(int(colours.get(unit, "#bbbbbb")[j:j + 2], 16) for j in (1, 3, 5))
        base = tops[i + 1][1] if i + 1 < len(tops) else np.full_like(top, zlo)
        ok = np.isfinite(top) & np.isfinite(base) & (top > base + 1e-6)
        for a, b in _runs(ok):
            if b - a < 2:
                continue
            ring = [(float(x), float(z)) for x, z in zip(d[a:b], top[a:b])]
            ring += [(float(x), float(z)) for x, z in zip(d[a:b][::-1], base[a:b][::-1])]
            hatch = msp.add_hatch(dxfattribs={"layer": layer})
            hatch.rgb = rgb
            hatch.paths.add_polyline_path(ring, is_closed=True)
        for a, b in _runs(np.isfinite(top)):
            if b - a >= 2:
                msp.add_lwpolyline(np.column_stack([d[a:b], top[a:b]]).tolist(), dxfattribs={"layer": "GEO_CONTATTI"})
    g = prof["ground"]
    for a, b in _runs(np.isfinite(g)):
        if b - a >= 2:
            msp.add_lwpolyline(np.column_stack([d[a:b], g[a:b]]).tolist(), dxfattribs={"layer": "GEO_TERRENO"})
    w = max(prof["length"] / 150.0, 0.05)
    for h in holes:
        b, s = h["borehole"], h["station"]
        if b.z is None:
            continue
        for iv in b.intervals:
            box = [(s - w / 2, b.z - iv.top), (s + w / 2, b.z - iv.top), (s + w / 2, b.z - iv.base),
                   (s - w / 2, b.z - iv.base)]
            msp.add_lwpolyline(box, close=True, dxfattribs={"layer": "GEO_SONDAGGI"})
        msp.add_text(f"{b.id} ({h['offset']:+.1f} m)", height=w * 0.8,
                     dxfattribs={"layer": "GEO_TESTI"}).set_placement((s - w / 2, b.z + w * 0.5))
    from pyarchinit_photogrammetry.archaeo.sections import grid_step

    step = grid_step(zhi - zlo)
    for z in np.arange(np.floor(zlo / step) * step, zhi + step, step):
        msp.add_line((0.0, z), (prof["length"], z), dxfattribs={"layer": "GEO_GRIGLIA"})
        msp.add_text(f"{z:.2f}", height=step * 0.15, dxfattribs={"layer": "GEO_GRIGLIA"}).set_placement((-step, z))
    msp.add_text(name, height=step * 0.3, dxfattribs={"layer": "GEO_TESTI"}).set_placement((0.0, zhi + step * 0.6))
    doc.saveas(path)
    return path


def write_csv(prof: dict, path: Path) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["distance", "x", "y", "ground"] + [f"top_{u}" for u, _v in prof["tops"]])
        for i in range(len(prof["distance"])):
            row = [prof["distance"][i], prof["x"][i], prof["y"][i], prof["ground"][i]] + [v[i] for _u, v in prof["tops"]]
            w.writerow(["" if not np.isfinite(v) else round(float(v), 4) for v in row])
    return path
