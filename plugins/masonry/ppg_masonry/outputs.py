"""The outputs of a face: the plate at scale and the DXF of its drawing, with everything the plugin and the program
know of it, and the whole export (plate, DXF, the program's export of the face's drawings with the stones' measures,
the tables, the bulge).

* the plate (SVG and PDF, matplotlib): the face's orthophoto at a standard scale (the largest that fits an A3 sheet),
  the stones, the courses, the USM with their numbers, the lines of the horizontal sections, the bulge's contours
  (optional), a scale bar and a title block with the face's name, the wall's, the scale, the date and the measures;
* the DXF in the face's ``(u, v)`` metres: layers ``PIETRE`` (outlines), ``CORSI`` (the line of each course),
  ``USM`` (outlines and numbers), ``SEZIONI`` (the lines of the sections), ``BORDO`` (the face's outline, or the
  wall's area drawn on it), ``QUOTE`` (spot heights drawn on the face)."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

import numpy as np

from . import walls
from .texts import degrees, num, t

SCALES = (10, 20, 25, 50, 100, 200, 250, 500, 1000)
SHEET = (384.0, 199.0)  # the drawing on an A3 sheet (420 x 297 mm, landscape) beside its margins and title block
TITLE_MM = 62.0
MARGIN_MM = 18.0


def choose_scale(width_m: float, height_m: float, sheet=SHEET) -> int:
    """The most detailed standard scale at which the face fits the drawing area of the sheet."""
    for n in SCALES:
        if width_m * 1000.0 / n <= sheet[0] and height_m * 1000.0 / n <= sheet[1]:
            return n
    return int(np.ceil(max(width_m * 1000.0 / sheet[0], height_m * 1000.0 / sheet[1]) / 1000.0) * 1000)


def _bar_length(width_m: float) -> float:
    """A round length for the scale bar, about a fifth of the drawing."""
    target = width_m / 5.0
    e = 10 ** np.floor(np.log10(max(target, 1e-3)))
    return float(max(m * e for m in (1, 2, 5) if m * e <= target) if target >= e else e)


MAX_CONTOURS = 12


def contour_levels(lo: float, hi: float, max_levels: int = MAX_CONTOURS) -> tuple[list[float], float]:
    """The bulge's contour levels (cm) over ``lo``-``hi`` (its 0.5th and 99.5th percentiles): the step the smallest
    of 0.5, 1, 2, 5 x 10^n cm that gives at most ``max_levels`` levels, zero left out. ``(levels, step)``."""
    steps = sorted(m * 10.0 ** e for e in range(-1, 7) for m in (1.0, 2.0, 5.0) if m * 10.0 ** e >= 0.5)
    for st in steps:
        levels = [float(x) for x in np.arange(np.ceil(lo / st) * st, hi + 1e-9, st) if abs(x) > 1e-9]
        if len(levels) <= max_levels:
            return levels, st
    return [], steps[-1]


def _rings_of(geom) -> list[np.ndarray]:
    """The outlines (exteriors and holes) of a polygon or of every part of a multipolygon; none for None."""
    if geom is None or geom.is_empty:
        return []
    out = []
    for part in getattr(geom, "geoms", [geom]):
        if part.geom_type != "Polygon":
            continue
        out.append(np.asarray(part.exterior.coords))
        out += [np.asarray(h.coords) for h in part.interiors]
    return out


def course_lines(courses: list[dict]):
    """``[(n, [(u0, v0), (u1, v1)])]``: the line of each course through its centres, from its first stone to the
    last."""
    out = []
    for c in courses or []:
        if not c.get("counts"):
            continue
        k = np.tan(np.radians(c.get("slope_deg") or 0.0))
        um = c.get("u_mean", 0.5 * (c["u_from"] + c["u_to"]))
        out.append((c["n"], [(c["u_from"], c["height"] + k * (c["u_from"] - um)),
                             (c["u_to"], c["height"] + k * (c["u_to"] - um))]))
    return out


def section_lines(entry: dict, sections: dict | None):
    """``[(height, [(u0, v), (u1, v)])]`` of a plane face: where each horizontal section crosses the face."""
    if not sections:
        return []
    from .shape import _frame, _v_at_height

    frame = _frame(entry)
    u0, _v0, u1, _v1 = entry["bounds"]
    out = []
    for h in sections.get("heights") or []:
        try:
            out.append((h, [(u0, _v_at_height(frame, u0, h)), (u1, _v_at_height(frame, u1, h))]))
        except walls.NotReady:
            continue
    return out


def _context(chunk, wid: str, eid: str | None):
    from . import stones, usm

    w, entry = walls.need_face(chunk, wid, eid)
    res = walls.results(chunk, w["id"], entry["id"])
    st = res.get("stones") or {}
    region = stones.drawn_region(chunk, w["id"], entry["id"])
    return {"wall": w, "entry": entry, "results": res, "stones": stones.face_stones(chunk, entry["id"]),
            "courses": st.get("course_rows") or [], "units": usm.face_units(chunk, entry["id"]), "region": region,
            "quote": _quote(chunk, entry["id"])}


def _quote(chunk, eid: str) -> list[tuple[float, float, str]]:
    from pyarchinit_photogrammetry.engine.elevation import frame_drawings

    out = []
    for q in frame_drawings(chunk, eid).get("quote") or []:
        if "u" in q and "v" in q:
            z = q.get("quota_q")
            out.append((float(q["u"]), float(q["v"]), "?" if z in (None, "") else f"{float(z):.2f}"))
    return out


def measures_lines(ctx: dict) -> list[str]:
    """The measures of the face in a few lines, for the title block."""
    from .usm import unit_label

    res = ctx["results"]
    sh = res.get("shape") or {}
    lines = []
    if sh.get("surface") == "plane":
        deg, cmm = sh.get("lean_deg") or 0.0, num(sh.get("lean_cm_per_m", 0), 1)
        if deg < 0.05:
            lean = t("A piombo (entro 0,05°)", "Plumb (within 0.05°)")
        elif sh.get("lean_toward") == "out":
            lean = t(f"Strapiombo {degrees(deg)} ({cmm} cm/m, verso chi guarda)",
                     f"Overhang {degrees(deg)} ({cmm} cm/m, towards the viewer)")
        else:
            lean = t(f"Fuori piombo all'indietro {degrees(deg)} ({cmm} cm/m)",
                     f"Leaning back {degrees(deg)} ({cmm} cm/m)")
        if not sh.get("lean_reliable", True):
            lean += t(" - non affidabile: la superficie non è piana", " - not reliable: the surface is not a plane")
        lines.append(lean + t(f"; spanciamento p90 {num(sh.get('bulge_p90_cm', 0), 1)} cm, da "
                              f"{num(sh.get('bulge_min_cm', 0), 1)} a {num(sh.get('bulge_max_cm', 0), 1)} cm",
                              f"; bulge p90 {num(sh.get('bulge_p90_cm', 0), 1)} cm, from "
                              f"{num(sh.get('bulge_min_cm', 0), 1)} to {num(sh.get('bulge_max_cm', 0), 1)} cm"))
    elif sh:
        bad = not sh.get("lean_reliable", True)
        lines.append(t(f"Asse inclinato {degrees(sh.get('axis_deg'))}, raggio {num(sh.get('radius', 0), 3)} m"
                       + (" - non affidabile: la superficie non è un cilindro" if bad else ""),
                       f"Axis inclined {degrees(sh.get('axis_deg'))}, radius {num(sh.get('radius', 0), 3)} m"
                       + (" - not reliable: the surface is not a cylinder" if bad else "")))
    st = res.get("stones") or {}
    if st.get("typology"):
        lines.append(t(f"{st['stones']} pietre, {st['courses']} filari, malta {num(100 * (st['mortar'] or 0), 1)} %",
                       f"{st['stones']} stones, {st['courses']} courses, mortar {num(100 * (st['mortar'] or 0), 1)} %"))
        lines.append(st["typology"]["line"])
    if ctx["units"]:
        lines.append("USM: " + ", ".join(unit_label(r) for _g, r in ctx["units"]))
    return lines


def _wrap(text: str, width: int = 150) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text


def plate(chunk, wid: str, eid: str | None, folder=None, scale: int | None = None, contours: bool = True,
          stem: str = "tavola", formats=("svg", "pdf")) -> dict:
    """The plate of the face (SVG and PDF, or ``formats``) in ``folder`` (default the face's folder). Returns ``{"svg",
    "pdf", "scale"}`` (``"png"`` instead when matplotlib cannot write vector files here)."""
    import matplotlib.patches
    import rasterio
    from matplotlib.figure import Figure
    from rasterio.enums import Resampling

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    ctx = _context(chunk, wid, eid)
    entry, w = ctx["entry"], ctx["wall"]
    u0, v0, u1, v1 = entry["bounds"]
    width_m, height_m = u1 - u0, v1 - v0
    n = int(scale or choose_scale(width_m, height_m))
    draw_w, draw_h = width_m * 1000.0 / n, height_m * 1000.0 / n
    fig_w = max(draw_w, 200.0) + 2 * MARGIN_MM
    fig_h = draw_h + TITLE_MM + 2 * MARGIN_MM
    fig = Figure(figsize=(fig_w / 25.4, fig_h / 25.4))
    ax = fig.add_axes([MARGIN_MM / fig_w, (MARGIN_MM + TITLE_MM) / fig_h, draw_w / fig_w, draw_h / fig_h])
    ortho, _relief = elevation_paths(chunk, entry)
    if ortho.exists():
        with rasterio.open(ortho) as src:
            k = max(1, int(np.ceil(max(src.width, src.height) / 3000)))
            img = src.read(out_shape=(src.count, max(1, src.height // k), max(1, src.width // k)),
                           resampling=Resampling.average)
            b = src.bounds
        rgba = np.moveaxis(img[:4], 0, -1) if img.shape[0] >= 3 else np.repeat(np.moveaxis(img, 0, -1), 3, -1)
        ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), interpolation="bilinear")
    for p in ctx["stones"]:
        xy = np.asarray(p.exterior.coords)
        ax.plot(xy[:, 0], xy[:, 1], color="black", lw=0.35)
    for nc, line in course_lines(ctx["courses"]):
        xy = np.asarray(line)
        ax.plot(xy[:, 0], xy[:, 1], color="#1f77b4", lw=0.8, ls="--")
        ax.text(xy[0, 0] - 0.01 * width_m, xy[0, 1], f"{nc}", color="#1f77b4", fontsize=5, ha="right", va="center")
    from .usm import unit_label

    for g, rec in ctx["units"]:
        for part in getattr(g, "geoms", [g]):
            xy = np.asarray(part.exterior.coords)
            ax.plot(xy[:, 0], xy[:, 1], color="#d62728", lw=1.1)
        c = g.representative_point()
        ax.text(c.x, c.y, unit_label(rec), color="#d62728", fontsize=7, ha="center", va="center",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.7})
    for h, line in section_lines(entry, ctx["results"].get("sections")):
        xy = np.asarray(line)
        ax.plot(xy[:, 0], xy[:, 1], color="#2ca02c", lw=0.8, ls="-.")
        ax.text(xy[1, 0], xy[1, 1], f" {num(h, 2)}", color="#2ca02c", fontsize=5, va="center")
    bulge_path = chunk.data_dir / ((ctx["results"].get("shape") or {}).get("bulge") or "nessuno")
    contour_step = None
    if contours and bulge_path.is_file():
        with rasterio.open(bulge_path) as src:
            k = max(1, int(np.ceil(max(src.width, src.height) / 1500)))
            z = src.read(1, out_shape=(max(1, src.height // k), max(1, src.width // k)), resampling=Resampling.average)
            b = src.bounds
        if np.isfinite(z).sum() > 10:
            from scipy.ndimage import gaussian_filter

            # smoothed over about 5 cm: the contours of the wall's shape, not of each joint and stone
            sigma = max(1.0, 0.05 / max(abs(b.right - b.left) / z.shape[1], 1e-6))
            ok = np.isfinite(z)
            num_ = gaussian_filter(np.where(ok, z, 0.0), sigma)
            den = gaussian_filter(ok.astype(float), sigma)
            zz = np.where(ok & (den > 0.3), num_ / np.maximum(den, 1e-9), np.nan)
            lo, hi = (float(x) for x in np.nanpercentile(zz, [0.5, 99.5]))
            levels, step = contour_levels(lo, hi)
            contour_step = step
            if levels:
                xs = np.linspace(b.left, b.right, zz.shape[1])
                ys = np.linspace(b.top, b.bottom, zz.shape[0])
                cs = ax.contour(xs, ys, zz, levels=levels, cmap="coolwarm", linewidths=0.5)
                ax.clabel(cs, fmt=lambda x: f"{x:+g}", fontsize=4)
    for ring in _rings_of(ctx["region"]):  # the wall's area drawn, every part and hole of it
        ax.plot(ring[:, 0], ring[:, 1], color="#555555", lw=0.6)
    for qu, qv, label in ctx["quote"]:
        ax.plot([qu], [qv], marker="+", color="#ff7f0e", ms=4)
        ax.text(qu, qv, f" {label}", color="#ff7f0e", fontsize=5)
    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)
    ax.set_xlabel(t("u lungo la faccia (m)", "u along the face (m)"), fontsize=6)
    ax.set_ylabel(t("quota (m)" if entry.get("v_is_height") else "v (m)", "height (m)" if entry.get("v_is_height")
                    else "v (m)"), fontsize=6)
    # the scale bar, in the drawing's metres, below it
    bar = _bar_length(width_m)
    by = (MARGIN_MM + TITLE_MM - 10.0) / fig_h
    bx = MARGIN_MM / fig_w
    bw = bar * 1000.0 / n / fig_w
    halves = fig.add_axes([bx, by, bw, 2.0 / fig_h])
    halves.set_axis_off()
    halves.add_patch(matplotlib.patches.Rectangle((0, 0), 0.5, 1, color="black"))
    halves.add_patch(matplotlib.patches.Rectangle((0.5, 0), 0.5, 1, fc="white", ec="black", lw=0.5))
    halves.set_xlim(0, 1)
    halves.set_ylim(0, 1)
    fig.text(bx, by - 1.0 / fig_h, "0", fontsize=6, va="top")
    fig.text(bx + bw, by - 1.0 / fig_h, f"{num(bar, 2 if bar < 1 else 0)} m", fontsize=6, va="top", ha="center")
    # the title block
    tb = fig.add_axes([MARGIN_MM / fig_w, MARGIN_MM / fig_h, (fig_w - 2 * MARGIN_MM) / fig_w,
                       (TITLE_MM - 16.0) / fig_h])
    tb.set_xticks([])
    tb.set_yticks([])
    for side in tb.spines.values():
        side.set_linewidth(0.8)
    tb.text(0.01, 0.88, f"{entry.get('name', '')}", fontsize=11, weight="bold", va="top", transform=tb.transAxes)
    tb.text(0.01, 0.66, t(f"Muro: {w['name']}  ·  Scala 1:{n}  ·  {time.strftime('%d/%m/%Y')}",
                          f"Wall: {w['name']}  ·  Scale 1:{n}  ·  {time.strftime('%Y-%m-%d')}"), fontsize=7, va="top",
            transform=tb.transAxes)
    text = "\n".join(_wrap(x, int(fig_w / 2.1)) for x in measures_lines(ctx))
    tb.text(0.01, 0.48, text, fontsize=6, va="top", transform=tb.transAxes, linespacing=1.3)
    every = (t(f" ogni {num(contour_step, 1)} cm", f" every {num(contour_step, 1)} cm") if contour_step else "")
    legend = t(f"pietre (nero) · filari (blu tratteggio) · USM (rosso) · sezioni (verde) · spanciamento (curve{every})",
               f"stones (black) · courses (blue dashes) · USM (red) · sections (green) · bulge (contours{every})")
    tb.text(0.99, 0.88, legend, fontsize=5.5, va="top", ha="right", transform=tb.transAxes)
    tb.text(0.99, 0.06, "pyarchinit-photogrammetry · Murature", fontsize=5, ha="right", transform=tb.transAxes,
            color="#666666")
    folder = Path(folder) if folder else walls.face_folder(chunk, w["id"], entry["id"])
    folder.mkdir(parents=True, exist_ok=True)
    out = {"scale": n}
    try:
        for ext in formats:
            path = folder / f"{stem}.{ext}"
            fig.savefig(path, format=ext, **({"dpi": 200} if ext == "png" else {}))
            out[ext] = path
    except (ValueError, ImportError):  # a matplotlib without its vector backends (an old packaged program)
        path = folder / f"{stem}.png"
        fig.savefig(path, format="png", dpi=200)
        out["png"] = path
    return out


def dxf(chunk, wid: str, eid: str | None, path) -> Path:
    """The face's drawing in its ``(u, v)`` metres, layers PIETRE, CORSI, USM, SEZIONI, BORDO, QUOTE."""
    import ezdxf
    from shapely.geometry import box

    from .usm import unit_label

    ctx = _context(chunk, wid, eid)
    entry = ctx["entry"]
    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    for name, colour in (("PIETRE", 7), ("CORSI", 5), ("USM", 1), ("SEZIONI", 3), ("BORDO", 8), ("QUOTE", 2)):
        doc.layers.add(name, color=colour)
    height = max(0.02, 4 * float(entry.get("resolution") or 0.005))
    for p in ctx["stones"]:
        msp.add_lwpolyline([tuple(map(float, q)) for q in np.asarray(p.exterior.coords)[:-1]], close=True,
                           dxfattribs={"layer": "PIETRE"})
    for nc, line in course_lines(ctx["courses"]):
        msp.add_lwpolyline(line, dxfattribs={"layer": "CORSI"})
        msp.add_text(str(nc), height=height, dxfattribs={"layer": "CORSI"}).set_placement(line[0])
    for g, rec in ctx["units"]:
        for part in getattr(g, "geoms", [g]):
            msp.add_lwpolyline([tuple(map(float, q)) for q in np.asarray(part.exterior.coords)[:-1]], close=True,
                               dxfattribs={"layer": "USM"})
        c = g.representative_point()
        msp.add_text(unit_label(rec), height=height, dxfattribs={"layer": "USM"}).set_placement((c.x, c.y))
    for h, line in section_lines(entry, ctx["results"].get("sections")):
        msp.add_lwpolyline(line, dxfattribs={"layer": "SEZIONI"})
        msp.add_text(f"{h:.3f}", height=height, dxfattribs={"layer": "SEZIONI"}).set_placement(line[1])
    border = ctx["region"] if ctx["region"] is not None else box(*entry["bounds"])
    for ring in _rings_of(border):
        msp.add_lwpolyline([tuple(map(float, q)) for q in ring[:-1]], close=True, dxfattribs={"layer": "BORDO"})
    for qu, qv, label in ctx["quote"]:
        msp.add_point((qu, qv), dxfattribs={"layer": "QUOTE"})
        msp.add_text(label, height=height, dxfattribs={"layer": "QUOTE"}).set_placement((qu, qv))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


def export_face(chunk, wid: str, eid: str | None, folder, scale: int | None = None) -> dict:
    """Everything of a face in ``folder``: the plate (SVG, PDF), the DXF of its drawing, the program's export of its
    drawings with the stones' measures (DXF, SVG, PNG, the two GeoPackages), the tables, the sections, the bulge."""
    from pyarchinit_photogrammetry.engine.elevation import _safe_stem

    from . import stones

    w, entry = walls.need_face(chunk, wid, eid)
    folder = Path(folder)
    stem = _safe_stem(entry.get("name") or entry["id"])
    out = {f"plate_{k}": v for k, v in plate(chunk, w["id"], entry["id"], folder, scale, stem=f"{stem}_tavola").items()}
    out["dxf"] = dxf(chunk, w["id"], entry["id"], folder / f"{stem}_muratura.dxf")
    out.update({f"drawings_{k}": v for k, v in stones.export_face(chunk, w["id"], entry["id"], folder,
                                                                  stem=stem).items()})
    res = walls.results(chunk, w["id"], entry["id"])
    bulge = (res.get("shape") or {}).get("bulge")
    if bulge and (chunk.data_dir / bulge).exists():
        out["bulge"] = folder / f"{stem}_spanciamento.tif"
        shutil.copy2(chunk.data_dir / bulge, out["bulge"])
    for name, path in (res.get("surveys") or {}).items():
        if (chunk.data_dir / path["raster"]).exists():
            target = folder / f"{stem}_{Path(path['raster']).name}"
            shutil.copy2(chunk.data_dir / path["raster"], target)
            out[f"survey_{name}"] = target
    summary = walls.write_summary(chunk, w["id"])
    if summary is not None:
        out["json"] = folder / f"{stem}_muro.json"
        out["json"].write_text(summary.read_text(encoding="utf-8"), encoding="utf-8")
    (folder / f"{stem}_file.json").write_text(json.dumps({k: str(v) for k, v in out.items()}, indent=1))
    return out
