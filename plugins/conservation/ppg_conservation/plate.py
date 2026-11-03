"""The plate of a campaign (A3 landscape, as the masonry plugin's plates; matplotlib's Figure API, no pyplot): the
elevation's orthophoto, the features with the legend's colours and hatches, cracks with their opening, treatments
with their own hatches (solid outline when done, dashed when to do), the legend of the terms present with both the
ICOMOS-ISCS and the UNI 11182 names (colours and hatches said to be the plugin's), the title block with the campaign,
its EN 16096 classes, the scale and - on a vertical plane in a map system - the direction the face looks to. SVG and
PDF; PNG when the program's matplotlib has no vector backends."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from . import records
from .texts import lang, num, t

SCALES = (10, 20, 25, 50, 100, 200, 250, 500, 1000)
SHEET = (384.0, 199.0)  # the drawing on an A3 sheet (420 x 297 mm, landscape) beside its margins and title block
TITLE_MM = 62.0
MARGIN_MM = 18.0
MIN_WIDTH_MM = 320.0  # the sheet is at least this wide inside its margins: the legend needs it


def _o():
    """The shared helpers of :mod:`.outputs` (imported late: :mod:`.outputs` exports this module's functions)."""
    from . import outputs

    return outputs


def _context(chunk, cid):
    return _o()._context(chunk, cid)


def _order(ctx):
    return _o()._order(ctx)


def _codes_present(ctx):
    return _o()._codes_present(ctx)


def _geom(f):
    return _o()._geom(f)


def _parts(geom, kind):
    return _o()._parts(geom, kind)


def _anchor(geom):
    return _o()._anchor(geom)


def _opening(f):
    return _o()._opening(f)


def glossary_names(vocab, code, codes_iscs=None):
    return _o().glossary_names(vocab, code, codes_iscs)



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


def _path(geom):
    """A matplotlib path of a polygon or multipolygon, holes included (rings oriented as matplotlib fills them)."""
    from matplotlib.path import Path as MPath
    from shapely.geometry.polygon import orient

    verts, codes = [], []
    for part in _parts(geom, "Polygon"):
        part = orient(part, 1.0)
        for ring in [part.exterior, *part.interiors]:
            xy = np.asarray(ring.coords)
            verts += [tuple(p) for p in xy]
            codes += [MPath.MOVETO] + [MPath.LINETO] * (len(xy) - 2) + [MPath.CLOSEPOLY]
    return MPath(verts, codes) if verts else None


def _facing(chunk, entry) -> float | None:
    """The azimuth (degrees from grid north, clockwise) the face looks to: only for a vertical plane in a map system
    (on a cylinder, a section or a local system there is no single direction to give)."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    from .geometry import frame_of

    if entry.get("kind") == "section" or not entry.get("v_is_height"):
        return None
    if crs_mod.is_local(entry.get("crs") or chunk.crs) or (entry.get("frame") or {}).get("kind") != "plane":
        return None
    n = np.asarray(frame_of(chunk, entry).normal, float)
    if np.hypot(n[0], n[1]) < 1e-6:
        return None
    return float(np.degrees(np.arctan2(n[0], n[1])) % 360.0)


def _cardinal(az: float) -> str:
    it = ("N", "NE", "E", "SE", "S", "SO", "O", "NO")
    en = ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
    return t(it[int((az + 22.5) // 45) % 8], en[int((az + 22.5) // 45) % 8])


def legend_line(vocab, code: str) -> str:
    """The line of a term in the plate's legend: its name in the interface, the ICOMOS-ISCS and UNI 11182 names, the
    code (a local term and an intervention said as such)."""
    g = glossary_names(vocab, code)
    own = g["it"] if lang() == "it" else g["en"]
    if g["treatment"]:
        return f"{own} ({vocab.term(code).names['en'] if code in vocab else code}) - " + \
            t("intervento (famiglie di EN 15898)", "intervention (families of EN 15898)") + f"  [{code}]"
    if g["local"]:
        names = (vocab.term(code).names if code in vocab else {"it": code, "en": code})
        return f"{names['it']} / {names['en']} - " + t("termine locale del plugin, non normativo",
                                                       "local term of the plugin, not from a standard") + f"  [{code}]"
    iscs = ", ".join(g["iscs_names"]) or "-"
    uni = g["uni_name"] or "-"
    return f"{own} - ICOMOS-ISCS: {iscs} · UNI 11182: {uni}  [{code}]"


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)) or text


def plate(chunk, cid: str, folder=None, scale: int | None = None, formats=("svg", "pdf"), stem: str = "tavola") -> dict:
    """The plate of the campaign ``cid`` (A3 landscape, as the masonry plugin's) in ``folder`` (default the
    campaign's folder). Returns ``{"svg", "pdf", "scale"}`` (``"png"`` instead when matplotlib cannot write vector
    files here)."""
    import matplotlib
    import matplotlib.patches as mpatches
    import rasterio
    from matplotlib.colors import to_rgba
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D
    from rasterio.enums import Resampling

    from pyarchinit_photogrammetry.engine.elevation import bounds_of, elevation_paths

    from .assess import class_text

    ctx = _context(chunk, cid)
    c, entry, vocab, leg = ctx["campaign"], ctx["entry"], ctx["vocab"], ctx["legend"]
    language = lang()
    u0, v0, u1, v1 = bounds_of(chunk, entry)
    width_m, height_m = u1 - u0, v1 - v0
    n = int(scale or choose_scale(width_m, height_m))
    draw_w, draw_h = width_m * 1000.0 / n, height_m * 1000.0 / n
    fig_w = max(draw_w, MIN_WIDTH_MM) + 2 * MARGIN_MM
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
        rgba = np.moveaxis(img[:4], 0, -1) if img.shape[0] >= 3 else np.repeat(np.moveaxis(img[:1], 0, -1), 3, -1)
        ax.imshow(rgba, extent=(b.left, b.right, b.bottom, b.top), interpolation="bilinear", alpha=0.85)

    def style(code):
        s = leg.get(code) or {"colour": "#808080", "hatch": {"mpl": "//"}}
        return s["colour"], (s.get("hatch") or {}).get("mpl") or None

    notes = 0
    for f in _order(ctx):
        g = _geom(f)
        colour, hatch = style(f["code"])
        if f["kind"] in ("decay", "treatment") and g.geom_type in ("Polygon", "MultiPolygon"):
            path = _path(g)
            if path is None:
                continue
            if f["kind"] == "decay":
                patch = mpatches.PathPatch(path, fc=to_rgba(colour, 0.30), ec=to_rgba(colour, 1.0), lw=0.6,
                                           hatch=hatch)
            else:
                done = bool((f.get("treatment") or {}).get("done"))
                patch = mpatches.PathPatch(path, fc="none", ec=colour, lw=1.0, hatch=hatch,
                                           ls="-" if done else "--")
            ax.add_patch(patch)
        elif f["kind"] in ("crack", "treatment"):
            done = bool((f.get("treatment") or {}).get("done")) if f["kind"] == "treatment" else True
            for part in _parts(g, "LineString"):
                xy = np.asarray(part.coords)
                ax.plot(xy[:, 0], xy[:, 1], color=colour, lw=1.2 if f["kind"] == "crack" else 2.4,
                        ls="-" if done else "--", alpha=1.0 if f["kind"] == "crack" else 0.7,
                        solid_capstyle="round")
            if f["kind"] == "crack" and _opening(f):
                a = _anchor(g)
                ax.text(a.x, a.y, f" {_opening(f)}", color=colour, fontsize=5, va="bottom",
                        bbox={"boxstyle": "round,pad=0.1", "fc": "white", "ec": "none", "alpha": 0.7})
        elif f["kind"] == "note":
            notes += 1
            a = _anchor(g)
            ax.plot([a.x], [a.y], marker="o", ms=4, mfc="white", mec=colour, mew=1.0)
            ax.text(a.x, a.y, f"  {notes}", color=colour, fontsize=5, va="center")
    ax.set_xlim(u0, u1)
    ax.set_ylim(v0, v1)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=5)
    ax.set_xlabel(t("u lungo la faccia (m)", "u along the face (m)"), fontsize=6)
    ax.set_ylabel(t("quota (m)", "height (m)") if entry.get("v_is_height") else "v (m)", fontsize=6)
    # the scale bar, in the drawing's metres, below it
    bar = _bar_length(width_m)
    by = (MARGIN_MM + TITLE_MM - 10.0) / fig_h
    bx = MARGIN_MM / fig_w
    bw = bar * 1000.0 / n / fig_w
    halves = fig.add_axes([bx, by, bw, 2.0 / fig_h])
    halves.set_axis_off()
    halves.add_patch(mpatches.Rectangle((0, 0), 0.5, 1, color="black"))
    halves.add_patch(mpatches.Rectangle((0.5, 0), 0.5, 1, fc="white", ec="black", lw=0.5))
    halves.set_xlim(0, 1)
    halves.set_ylim(0, 1)
    fig.text(bx, by - 1.0 / fig_h, "0", fontsize=6, va="top")
    fig.text(bx + bw, by - 1.0 / fig_h, f"{num(bar, 2 if bar < 1 else 0)} m", fontsize=6, va="top", ha="center")
    # the direction the face looks to
    az = _facing(chunk, entry)
    if az is not None:
        size = 12.0
        na = fig.add_axes([1 - (MARGIN_MM + size) / fig_w, by - 4.0 / fig_h, size / fig_w, size / fig_h])
        na.set_axis_off()
        na.set_xlim(-1.3, 1.3)
        na.set_ylim(-1.3, 1.3)
        na.set_aspect("equal")
        na.add_patch(mpatches.Circle((0, 0), 1.0, fc="none", ec="black", lw=0.5))
        na.annotate("", xy=(0, 1.0), xytext=(0, 0), arrowprops={"arrowstyle": "-|>", "lw": 0.6, "color": "black"})
        na.text(0, 1.12, "N", ha="center", va="bottom", fontsize=5)
        r = np.radians(az)
        na.annotate("", xy=(np.sin(r), np.cos(r)), xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": "#d62728"})
        fig.text(1 - MARGIN_MM / fig_w, by - 5.0 / fig_h,
                 t(f"faccia esposta a {num(az, 0)}° ({_cardinal(az)})", f"face looking {num(az, 0)}° "
                   f"({_cardinal(az)})"), fontsize=5, va="top", ha="right", color="#d62728")
    # the title block: the campaign on the left, the legend on the right
    tb_w = fig_w - 2 * MARGIN_MM
    tb_h = TITLE_MM - 16.0
    tb = fig.add_axes([MARGIN_MM / fig_w, MARGIN_MM / fig_h, tb_w / fig_w, tb_h / fig_h])
    tb.set_xticks([])
    tb.set_yticks([])
    for side in tb.spines.values():
        side.set_linewidth(0.8)
    left = 0.40  # the share of the block for the campaign
    tb.axvline(left, color="black", lw=0.5)
    tb.text(0.01, 0.92, entry.get("name", ""), fontsize=10, weight="bold", va="top",
            transform=tb.transAxes)
    rc = records.assessment(chunk, c["date"]) or {}
    classes = c.get("classes") or {}
    lines = [t(f"Campagna: {c['name']} del {c['date']}", f"Campaign: {c['name']} of {c['date']}")]
    who = [x for x in (c.get("surveyor"), c.get("method")) if x]
    if who:
        lines.append(t("Rilevatore e metodo: ", "Surveyor and method: ") + " · ".join(who))
    lines.append(t(f"Scala 1:{n}  ·  stampata il {time.strftime('%d/%m/%Y')}",
                   f"Scale 1:{n}  ·  printed on {time.strftime('%Y-%m-%d')}"))
    lines.append(class_text("cc", classes.get("cc"), language) if classes.get("cc") is not None
                 else t("CC non scelta", "CC not chosen"))
    lines.append(class_text("uc", classes.get("uc"), language) if classes.get("uc") is not None
                 else t("UC non scelta", "UC not chosen"))
    lines.append((class_text("rc", rc.get("rc"), language) + t(" (dell'oggetto, alla data)", " (of the object, on "
                                                                                              "the date)"))
                 if rc.get("rc") is not None else t("RC non scelta", "RC not chosen"))
    if classes.get("reason"):
        lines.append(t("Motivo: ", "Reason: ") + str(classes["reason"]))
    lines.append(t("Classi EN 16096:2012 scelte dal rilevatore; lessico ICOMOS-ISCS 2008 e UNI 11182:2006",
                   "EN 16096:2012 classes chosen by the surveyor; vocabulary ICOMOS-ISCS 2008 and UNI 11182:2006"))
    chars = int(left * tb_w / 1.25)
    tb.text(0.01, 0.72, "\n".join(_wrap(x, chars) for x in lines), fontsize=5.5, va="top", transform=tb.transAxes,
            linespacing=1.35)
    tb.text(left - 0.01, 0.04, "pyarchinit-photogrammetry · " + t("Conservazione", "Conservation"), fontsize=4.5,
            ha="right", transform=tb.transAxes, color="#666666")
    # the legend of the terms present
    codes = _codes_present(ctx)
    tb.text(left + 0.01, 0.94, t("Legenda (colori e retini del plugin: nessuna norma li prescrive)",
                                 "Legend (the plugin's colours and hatches: no standard prescribes them)"),
            fontsize=5.5, weight="bold", va="top", transform=tb.transAxes)
    row_mm = 3.4
    rows = max(1, int((tb_h - 8.0) // row_mm))
    cols = max(1, int(np.ceil(len(codes) / rows)))
    col_w = (1.0 - left - 0.02) / cols
    kinds = {}
    for f in ctx["features"]:
        kinds.setdefault(f["code"], f["kind"])
    sw, sh = 5.0 / tb_w, 2.2 / tb_h  # a swatch, in the block's fractions
    for i, code in enumerate(codes):
        col, row = divmod(i, rows)
        x = left + 0.01 + col * col_w
        y = 1.0 - (6.5 + (row + 1) * row_mm) / tb_h
        colour, hatch = style(code)
        kind = kinds.get(code)
        if kind == "crack":
            tb.add_line(Line2D([x, x + sw], [y + sh / 2, y + sh / 2], color=colour, lw=1.2, transform=tb.transAxes))
        elif kind == "note":
            tb.add_line(Line2D([x + sw / 2], [y + sh / 2], marker="o", ms=3, mfc="white", mec=colour,
                               transform=tb.transAxes))
        else:
            fc = "none" if kind == "treatment" else to_rgba(colour, 0.30)
            tb.add_patch(mpatches.Rectangle((x, y), sw, sh, fc=fc, ec=colour, lw=0.5, hatch=hatch,
                                            transform=tb.transAxes))
        text = legend_line(vocab, code)
        tb.text(x + sw + 0.005, y + sh / 2, _wrap(text, int(col_w * tb_w / 0.95)) if cols > 1 else text,
                fontsize=4.5, va="center", transform=tb.transAxes)
    if any(f["kind"] == "treatment" for f in ctx["features"]):
        tb.text(left + 0.01, 0.03, t("Interventi: contorno continuo = eseguito, tratteggiato = da eseguire",
                                     "Treatments: solid outline = done, dashed = to be done"),
                fontsize=4.5, transform=tb.transAxes, color="#444444")
    folder = Path(folder) if folder else records.campaign_folder(chunk, entry["id"], c["id"])
    folder.mkdir(parents=True, exist_ok=True)
    out = {"scale": n}
    with matplotlib.rc_context({"svg.fonttype": "none", "pdf.fonttype": 42, "hatch.linewidth": 0.4}):
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
