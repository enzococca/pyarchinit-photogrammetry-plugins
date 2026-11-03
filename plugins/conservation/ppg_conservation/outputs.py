"""The outputs of a campaign: what a restorer hands in, made again from the records whenever asked (the files are
derived: the records in ``chunk.meta["conservation"]`` are the primary data). Only the accepted features are a
campaign's: the proposals not accepted are left out of every output.

* :func:`plate` - the plate at scale on an A3 sheet (SVG and PDF, matplotlib; PNG when the program's matplotlib has
  no vector backends): the elevation's orthophoto, the features with the legend's colours and hatches, cracks with
  their opening, treatments with their own hatches (solid outline when done, dashed when to do), the legend of the
  terms present with both the ICOMOS-ISCS and the UNI 11182 names (colours and hatches said to be the plugin's), the
  title block with the campaign, its EN 16096 classes, the scale and - on a vertical plane in a map system - the
  direction the face looks to;
* :func:`tables` - Excel (``io.tables.write_xlsx``) and CSV: features, areas per term (and per intensity), per family,
  cracks, treatments, the lines of the EN 16096 report;
* :func:`dxf` - in the elevation's ``(u, v)`` metres: a layer per family ``DEGRADO_<FAMILY>``, ``LESIONI``,
  ``INTERVENTI``, ``NOTE``, ``BORDO``; a block per term with the attributes ``ID``, ``CODE``, ``NAME_IT``, ``NAME_EN``
  inserted on every feature, hatches with the legend's DXF patterns;
* :func:`geopackages` - the program's pair: the unfolded GeoPackage in ``(u, v)`` (no system, srs -1, a ``geom_world``
  WKT column) and the 3D one in the chunk's system (``engine.elevation.write_world_tables``, on the plane of the
  elevation, ``d`` = 0, as the program's drawings), tables ``degrado``, ``lesioni``, ``interventi``, ``note``;
* :func:`qgis_styles` / :func:`qgis_project` - a ``.qml`` per table (categories on ``code``), the elevation's QGIS
  project of the program with the plugin's tables and those styles;
* :func:`export_campaign` - all of it in the campaign's folder ``<chunk data>/conservation/<eid>/<cid>/``.
"""
from __future__ import annotations

import csv
import logging
import re
import time
from pathlib import Path

import numpy as np

from . import records
from .texts import lang, num, t

SCALES = (10, 20, 25, 50, 100, 200, 250, 500, 1000)
SHEET = (384.0, 199.0)  # the drawing on an A3 sheet (420 x 297 mm, landscape) beside its margins and title block
TITLE_MM = 62.0
MARGIN_MM = 18.0
LOG = logging.getLogger("ppg_conservation")
MIN_WIDTH_MM = 320.0  # the sheet is at least this wide inside its margins: the legend needs it

# the tables of the GeoPackages: (geometry type, geometry of the QGIS style, the feature kinds in it)
TABLES = {"degrado": ("POLYGON", "area", ("decay",)), "lesioni": ("LINESTRING", "line", ("crack",)),
          "interventi": ("POLYGON", "area", ("treatment",)), "note": ("POINT", "point", ("note",))}
TABLE_TITLES = {"degrado": ("Degrado", "Decay"), "lesioni": ("Lesioni", "Cracks"),
                "interventi": ("Interventi", "Treatments"), "note": ("Note", "Notes")}
COLUMNS = {"id": "TEXT", "campaign": "TEXT", "date": "TEXT", "kind": "TEXT", "code": "TEXT", "code_iscs": "TEXT",
           "code_uni": "TEXT", "name_it": "TEXT", "name_en": "TEXT", "family": "TEXT", "extent": "REAL",
           "intensity": "TEXT", "width_mm": "REAL", "depth_mm": "REAL", "cause": "TEXT", "condition": "INTEGER",
           "urgency": "INTEGER", "status": "TEXT", "source": "TEXT", "photos": "TEXT", "us": "TEXT", "notes": "TEXT"}
TREATMENT_COLUMNS = {"technique": "TEXT", "materials": "TEXT", "treatment_date": "TEXT", "done": "INTEGER",
                     "done_by": "TEXT", "shape": "TEXT"}
KIND_WORDS = {"decay": ("degrado", "decay"), "crack": ("lesione", "crack"), "treatment": ("intervento", "treatment"),
              "note": ("nota", "note")}
INTENSITY_WORDS = {"low": ("bassa", "low"), "medium": ("media", "medium"), "high": ("alta", "high")}
STATUS_WORDS = {"accepted": ("accettato", "accepted"), "proposed": ("proposto", "proposed")}
FAMILY_ORDER = ("general", "crack_deformation", "detachment", "material_loss", "discoloration_deposit", "biological",
                "masonry", "intervention")


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


def _tr(pair) -> str:
    return t(*pair)


def _word(table: dict, key) -> str:
    return _tr(table[key]) if key in table else ("" if key is None else str(key))


def _source_word(source: str | None) -> str:
    if not source or source == "drawn":
        return t("disegnato", "drawn")
    if source.startswith("proposed:"):
        return t(f"proposto ({source.split(':', 1)[1]})", f"proposed ({source.split(':', 1)[1]})")
    return t("modificato", "edited") if source == "edited" else source


# ---------------------------------------------------------------- what a campaign holds
def _context(chunk, cid: str) -> dict:
    from .vocabulary import UNNAMED, legend, load

    c = records.campaign(chunk, cid)
    entry = records._elevation(chunk, c["elevation"])
    accepted = records.features(chunk, cid, status="accepted")
    feats = [f for f in accepted if f["code"] != UNNAMED]
    unnamed = len(accepted) - len(feats)
    if unnamed:
        LOG.info(t(f"{unnamed} elementi da nominare esclusi", f"{unnamed} features to be named left out"))
    return {"campaign": c, "entry": entry, "features": feats, "unnamed": unnamed, "vocab": load(chunk),
            "legend": legend(chunk)}


def unnamed_count(chunk, cid: str) -> int:
    """The features of the campaign still to be named (``PPG.UNNAMED``), left out of every output."""
    from .vocabulary import UNNAMED

    return sum(1 for f in records.features(chunk, cid) if f["code"] == UNNAMED)


def _stem(ctx: dict) -> str:
    from pyarchinit_photogrammetry.engine.elevation import _safe_stem

    return _safe_stem(f"{ctx['entry'].get('name') or ctx['entry']['id']}_{ctx['campaign']['name']}")


def glossary_names(vocab, code: str, codes_iscs=None) -> dict:
    """The names of a term in both glossaries: ``{"iscs": [codes], "iscs_names": [English names], "uni": code,
    "uni_name": Italian name, "it", "en" (the interface's names), "family", "local", "treatment"}``. A UNI term: the
    ICOMOS-ISCS terms it is mapped onto; an ICOMOS-ISCS term: the UNI term it corresponds to; a local term or an
    intervention: neither."""
    if code not in vocab:
        return {"iscs": list(codes_iscs or []), "iscs_names": [], "uni": "", "uni_name": "", "it": code, "en": code,
                "family": "masonry", "local": True, "treatment": False}
    term = vocab.term(code)
    iscs = list(codes_iscs) if codes_iscs else records.iscs_codes(term) or []
    uni = term.uni or ""
    return {"iscs": iscs, "iscs_names": [vocab.term(x).names["en"] for x in iscs if x in vocab], "uni": uni,
            "uni_name": vocab.term(uni).names["it"] if uni and uni in vocab else "",
            "it": vocab.name(code, "it"), "en": vocab.name(code, "en"), "family": term.family,
            "local": bool(term.local or code.startswith("LOCAL.")), "treatment": term.kind == "treatment"}


def _order(ctx: dict) -> list[dict]:
    """The features in the order they are drawn and listed: areas of decay (larger first, so the small ones stay on
    top), treatments, cracks, notes."""
    rank = {"decay": 0, "treatment": 1, "crack": 2, "note": 3}
    return sorted(ctx["features"], key=lambda f: (rank.get(f["kind"], 4), -(float(f.get("extent") or 0.0))))


def _codes_present(ctx: dict) -> list[str]:
    """The codes of the campaign in the legend's order: decay by family, cracks, treatments, notes."""
    vocab = ctx["vocab"]

    def key(f):
        fam = vocab.term(f["code"]).family if f["code"] in vocab else "masonry"
        kind = {"decay": 0, "crack": 1, "treatment": 2, "note": 3}.get(f["kind"], 4)
        return kind, FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else 99, f["code"]

    return list(dict.fromkeys(f["code"] for f in sorted(ctx["features"], key=key)))


def _geom(f):
    from .geometry import as_geometry

    return as_geometry(f["geometry"])


def _parts(geom, kind: str) -> list:
    import shapely

    want = {"Polygon": ("Polygon",), "LineString": ("LineString",), "Point": ("Point",)}[kind]
    return [g for g in shapely.get_parts(geom) if g.geom_type in want and not g.is_empty]


def _anchor(geom):
    """Where a feature's label or block goes: a point inside an area, the middle of a line, the point."""
    if geom.geom_type in ("LineString", "MultiLineString"):
        return geom.interpolate(0.5, normalized=True)
    if geom.geom_type == "Point":
        return geom
    c = geom.centroid
    return c if geom.contains(c) else geom.representative_point()


def _links_text(f: dict) -> str:
    out = []
    for link in (f.get("links") or {}).get("us") or []:
        area, us = link[0], link[1]
        tipo = link[2] if len(link) > 2 and link[2] else "US"
        out.append(f"{tipo} {us}" + (f" ({t('area', 'area')} {area})" if area else ""))
    return "; ".join(out)


def _photos_text(f: dict) -> str:
    return "; ".join(str(p.get("label") or p.get("key") or "") for p in f.get("photos") or [])


def _opening(f: dict) -> str:
    w = f.get("width_mm")
    if w is None:
        return ""
    w = float(w)
    return f"{num(w, 0 if abs(w - round(w)) < 1e-9 else 1)} mm"


# ---------------------------------------------------------------- the plate
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
        rgba = np.moveaxis(img[:4], 0, -1) if img.shape[0] >= 3 else np.repeat(np.moveaxis(img, 0, -1), 3, -1)
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
    tb.text(0.01 / left * left, 0.92, entry.get("name", ""), fontsize=10, weight="bold", va="top",
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


# ---------------------------------------------------------------- the tables
def _yes(value) -> str:
    return t("sì", "yes") if value else t("no", "no")


def _r(value, d: int = 5):
    return None if value is None else round(float(value), d)


def _family_name(vocab, family: str) -> str:
    from .vocabulary import FAMILIES

    return _tr(FAMILIES[family]) if family in FAMILIES else family


def table_data(chunk, cid: str) -> dict:
    """The sheets of :func:`tables`: ``{sheet name: {"headers", "rows"}}`` in the interface's language."""
    from .assess import report_lines

    ctx = _context(chunk, cid)
    vocab = ctx["vocab"]
    tot = records.totals(chunk, cid)
    face = tot.get("face_area") or 0.0
    feats = _order(ctx)
    head = ["id", t("Tipo", "Kind"), t("Codice", "Code"), t("Termine", "Term"), t("Codice ISCS", "ISCS code"),
            t("Termine ISCS", "ISCS term"), t("Codice UNI", "UNI code"), t("Termine UNI", "UNI term"),
            t("Famiglia", "Family"), t("Estensione", "Extent"), t("Unità", "Unit"), t("Intensità", "Intensity"),
            t("Apertura (mm)", "Opening (mm)"), t("Profondità (mm)", "Depth (mm)"), t("Causa", "Cause"),
            t("Condizione", "Condition"), t("Urgenza", "Urgency"), t("Stato", "Status"), t("Fonte", "Source"),
            t("Foto", "Photos"), "US", t("Note", "Notes")]
    rows = []
    for f in feats:
        g = glossary_names(vocab, f["code"], f.get("code_iscs"))
        polygon = f["kind"] in ("decay", "treatment") and f["geometry"].startswith(("POLYGON", "MULTIPOLYGON"))
        unit = "m²" if polygon else ("m" if f["kind"] in ("crack", "treatment") else "")
        rows.append([f["id"], _word(KIND_WORDS, f["kind"]), f["code"], g["it"] if lang() == "it" else g["en"],
                     "; ".join(g["iscs"]), "; ".join(g["iscs_names"]), f.get("code_uni") or g["uni"] or "",
                     g["uni_name"], _family_name(vocab, g["family"]), _r(f.get("extent")), unit,
                     _word(INTENSITY_WORDS, f.get("intensity")), _r(f.get("width_mm"), 2), _r(f.get("depth_mm"), 2),
                     f.get("cause") or "", f.get("condition"), f.get("urgency"), _word(STATUS_WORDS, f["status"]),
                     _source_word(f.get("source")), _photos_text(f), _links_text(f), f.get("notes") or ""])
    # areas per term (and per intensity)
    by_code: dict = {}
    for f in feats:
        if f["kind"] == "decay" and f["geometry"].startswith(("POLYGON", "MULTIPOLYGON")):
            rec = by_code.setdefault(f["code"], {"n": 0, "area": 0.0, "low": 0.0, "medium": 0.0, "high": 0.0,
                                                 None: 0.0, "iscs": f.get("code_iscs"), "uni": f.get("code_uni")})
            rec["n"] += 1
            rec["area"] += float(f.get("extent") or 0.0)
            rec[f.get("intensity") if f.get("intensity") in INTENSITY_WORDS else None] += float(f.get("extent") or 0)
    area_head = [t("Codice", "Code"), t("Termine", "Term"), t("Codice ISCS", "ISCS code"),
                 t("Termine ISCS", "ISCS term"), t("Codice UNI", "UNI code"), t("Termine UNI", "UNI term"),
                 t("Elementi", "Features"), t("Area (m²)", "Area (m²)"), t("Quota della faccia (%)",
                                                                           "Share of the face (%)"),
                 t("Intensità bassa (m²)", "Low intensity (m²)"), t("Intensità media (m²)", "Medium intensity (m²)"),
                 t("Intensità alta (m²)", "High intensity (m²)"), t("Intensità non indicata (m²)",
                                                                   "Intensity not given (m²)")]
    area_rows = []
    for code, rec in by_code.items():
        g = glossary_names(vocab, code, rec["iscs"])
        area_rows.append([code, g["it"] if lang() == "it" else g["en"], "; ".join(g["iscs"]),
                          "; ".join(g["iscs_names"]), rec["uni"] or g["uni"] or "", g["uni_name"], rec["n"],
                          _r(rec["area"]), _r(100.0 * rec["area"] / face, 3) if face else None, _r(rec["low"]),
                          _r(rec["medium"]), _r(rec["high"]), _r(rec[None])])
    # per family
    fam: dict = {}
    for code, rec in by_code.items():
        g = glossary_names(vocab, code)
        x = fam.setdefault(g["family"], [0, 0.0])
        x[0] += rec["n"]
        x[1] += rec["area"]
    fam_head = [t("Famiglia", "Family"), t("Nome", "Name"), t("Elementi", "Features"), t("Area (m²)", "Area (m²)"),
                t("Quota della faccia (%)", "Share of the face (%)")]
    fam_rows = [[k, _family_name(vocab, k), v[0], _r(v[1]), _r(100.0 * v[1] / face, 3) if face else None]
                for k, v in sorted(fam.items(), key=lambda kv: FAMILY_ORDER.index(kv[0])
                                   if kv[0] in FAMILY_ORDER else 99)]
    # cracks
    crack_head = ["id", t("Codice", "Code"), t("Termine", "Term"), t("Lunghezza (m)", "Length (m)"),
                  t("Apertura (mm)", "Opening (mm)"), t("Intensità", "Intensity"), t("Causa", "Cause"),
                  t("Urgenza", "Urgency"), "US", t("Note", "Notes")]
    crack_rows = []
    for f in feats:
        if f["kind"] == "crack":
            g = glossary_names(vocab, f["code"], f.get("code_iscs"))
            crack_rows.append([f["id"], f["code"], g["it"] if lang() == "it" else g["en"], _r(f.get("extent")),
                               _r(f.get("width_mm"), 2), _word(INTENSITY_WORDS, f.get("intensity")),
                               f.get("cause") or "", f.get("urgency"), _links_text(f), f.get("notes") or ""])
    # treatments
    treat_head = ["id", t("Codice", "Code"), t("Tecnica", "Technique"), t("Forma", "Shape"),
                  t("Estensione", "Extent"), t("Unità", "Unit"), t("Materiali", "Materials"), t("Data", "Date"),
                  t("Eseguito", "Done"), t("Eseguito da", "Done by"), "US", t("Note", "Notes")]
    treat_rows = []
    for f in feats:
        if f["kind"] == "treatment":
            tr = f.get("treatment") or {}
            technique = tr.get("technique") or f["code"]
            g = glossary_names(vocab, technique)
            polygon = f["geometry"].startswith(("POLYGON", "MULTIPOLYGON"))
            treat_rows.append([f["id"], f["code"], g["it"] if lang() == "it" else g["en"],
                               t("area", "area") if polygon else t("linea", "line"), _r(f.get("extent")),
                               "m²" if polygon else "m", tr.get("materials") or "", tr.get("date") or "",
                               _yes(tr.get("done")), tr.get("by") or "", _links_text(f), f.get("notes") or ""])
    report = [[line] for line in report_lines(chunk, cid)]
    return {t("Elementi", "Features"): {"headers": head, "rows": rows},
            t("Aree per termine", "Areas per term"): {"headers": area_head, "rows": area_rows},
            t("Aree per famiglia", "Areas per family"): {"headers": fam_head, "rows": fam_rows},
            t("Lesioni", "Cracks"): {"headers": crack_head, "rows": crack_rows},
            t("Interventi", "Treatments"): {"headers": treat_head, "rows": treat_rows},
            t("Rapporto EN 16096", "EN 16096 report"): {"headers": [t("Rapporto sullo stato di conservazione "
                                                                      "(EN 16096)", "Condition report (EN 16096)")],
                                                        "rows": report}}


def _write_csv(path: Path, sheet: dict) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(sheet["headers"])
        for row in sheet["rows"]:
            out.writerow(["" if v is None else v for v in row])
    return path


def tables(chunk, cid: str, folder, stem: str = "tabelle") -> dict:
    """The tables of the campaign in ``folder``: ``<stem>.xlsx`` (sheets *Elementi*, *Aree per termine*, *Aree per
    famiglia*, *Lesioni*, *Interventi*, *Rapporto EN 16096*), ``<stem>_elementi.csv`` and ``<stem>_aree.csv``.
    Returns ``{"xlsx", "csv_features", "csv_areas"}``."""
    from pyarchinit_photogrammetry.io.tables import write_xlsx

    data = table_data(chunk, cid)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    c = records.campaign(chunk, cid)
    sheets = list(data.values())
    return {"xlsx": write_xlsx(folder / f"{stem}.xlsx", data, title=c["name"]),
            "csv_features": _write_csv(folder / f"{stem}_{t('elementi', 'features')}.csv", sheets[0]),
            "csv_areas": _write_csv(folder / f"{stem}_{t('aree', 'areas')}.csv", sheets[1])}


# ---------------------------------------------------------------- the DXF
def _rgb(colour: str) -> tuple[int, int, int]:
    return tuple(int(colour[i:i + 2], 16) for i in (1, 3, 5))


def dxf_layer(vocab, f: dict) -> str:
    """The DXF layer of a feature: ``DEGRADO_<FAMILY>`` for decay, ``LESIONI``, ``INTERVENTI``, ``NOTE``."""
    if f["kind"] == "crack":
        return "LESIONI"
    if f["kind"] == "treatment":
        return "INTERVENTI"
    if f["kind"] == "note":
        return "NOTE"
    family = vocab.term(f["code"]).family if f["code"] in vocab else "masonry"
    return "DEGRADO_" + re.sub(r"[^A-Z0-9_]", "_", family.upper())


def block_name(code: str) -> str:
    return "T_" + re.sub(r"[^A-Za-z0-9_]", "_", code)


def dxf(chunk, cid: str, path) -> Path:
    """The campaign's drawing in the elevation's ``(u, v)`` metres (see the module): outlines, hatches with the
    legend's DXF patterns, a block per term inserted on every feature with its attributes."""
    import ezdxf
    from ezdxf import colors
    from shapely.geometry import box

    from pyarchinit_photogrammetry.engine.elevation import bounds_of

    ctx = _context(chunk, cid)
    entry, vocab, leg = ctx["entry"], ctx["vocab"], ctx["legend"]
    doc = ezdxf.new("R2010", setup=True)
    doc.units = ezdxf.units.M
    msp = doc.modelspace()
    height = max(0.02, 4 * float(entry.get("resolution") or 0.005))
    layers: dict[str, str] = {}
    for f in _order(ctx):
        layers.setdefault(dxf_layer(vocab, f), (leg.get(f["code"]) or {}).get("colour") or "#808080")
    for name, colour in layers.items():
        doc.layers.add(name).rgb = _rgb(colour)
    doc.layers.add("BORDO", color=8)
    for code in dict.fromkeys(f["code"] for f in ctx["features"]):
        blk = doc.blocks.new(block_name(code))
        blk.add_circle((0, 0), height / 4)
        for i, tag in enumerate(("ID", "CODE", "NAME_IT", "NAME_EN")):
            blk.add_attdef(tag, insert=(height / 2, -i * height * 1.2), text="",
                           dxfattribs={"height": height, "flags": 0 if tag in ("CODE",) else 1})
    for f in _order(ctx):
        g = _geom(f)
        layer = dxf_layer(vocab, f)
        style = leg.get(f["code"]) or {"colour": "#808080", "hatch": {"dxf": "ANSI31"}}
        attribs = {"layer": layer, "true_color": colors.rgb2int(_rgb(style["colour"]))}
        dashed = f["kind"] == "treatment" and not (f.get("treatment") or {}).get("done")
        line_attribs = dict(attribs, **({"linetype": "DASHED"} if dashed else {}))
        for poly in _parts(g, "Polygon"):
            rings = [np.asarray(poly.exterior.coords)[:-1]] + [np.asarray(r.coords)[:-1] for r in poly.interiors]
            for ring in rings:
                msp.add_lwpolyline([tuple(map(float, q)) for q in ring], close=True, dxfattribs=line_attribs)
            hatch = msp.add_hatch(dxfattribs=attribs)
            for ring in rings:
                hatch.paths.add_polyline_path([tuple(map(float, q)) for q in ring], is_closed=True)
            pattern = (style.get("hatch") or {}).get("dxf") or "ANSI31"
            try:
                if pattern.upper() == "SOLID":
                    hatch.set_solid_fill(rgb=_rgb(style["colour"]))
                else:
                    hatch.set_pattern_fill(pattern, scale=0.5)
            except Exception:  # noqa: BLE001 - a pattern this ezdxf does not know: lines at 45 degrees
                hatch.set_pattern_fill("ANSI31", scale=0.5)
        for line in _parts(g, "LineString"):
            msp.add_lwpolyline([tuple(map(float, q)) for q in line.coords], dxfattribs=line_attribs)
        for p in _parts(g, "Point"):
            msp.add_point((p.x, p.y), dxfattribs=attribs)
        a = _anchor(g)
        names = glossary_names(vocab, f["code"], f.get("code_iscs"))
        ref = msp.add_blockref(block_name(f["code"]), (a.x, a.y), dxfattribs={"layer": layer})
        ref.add_auto_attribs({"ID": f["id"], "CODE": f["code"], "NAME_IT": names["it"], "NAME_EN": names["en"]})
    border = box(*bounds_of(chunk, entry))
    msp.add_lwpolyline([tuple(map(float, q)) for q in np.asarray(border.exterior.coords)[:-1]], close=True,
                       dxfattribs={"layer": "BORDO"})
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


# ---------------------------------------------------------------- the GeoPackages
def _on_plane(chunk, entry):
    """The function placing a ``(u, v)`` geometry on the elevation's plane (``d`` = 0), 3D in the chunk's system - as
    the program places its drawings in the 3D GeoPackage."""
    import shapely

    from .geometry import frame_of

    frame = frame_of(chunk, entry)
    return lambda geom: shapely.transform(shapely.force_3d(geom), lambda xy: frame.point(
        xy[:, 0], xy[:, 1], np.zeros(len(xy))), include_z=True)


def feature_rows(chunk, cid: str) -> dict:
    """``{table: [(geometry in (u, v), values)]}`` of the accepted features: a part of a multi-geometry is a row, a
    treated crack (a line) the band of its opening in ``interventi``."""
    from .geometry import band

    ctx = _context(chunk, cid)
    vocab, c = ctx["vocab"], ctx["campaign"]
    out = {name: [] for name in TABLES}
    for f in _order(ctx):
        g = _geom(f)
        names = glossary_names(vocab, f["code"], f.get("code_iscs"))
        values = {"id": f["id"], "campaign": c["id"], "date": c["date"], "kind": f["kind"], "code": f["code"],
                  "code_iscs": "; ".join(names["iscs"]), "code_uni": f.get("code_uni") or names["uni"] or "",
                  "name_it": names["it"], "name_en": names["en"], "family": names["family"],
                  "extent": f.get("extent"), "intensity": f.get("intensity"), "width_mm": f.get("width_mm"),
                  "depth_mm": f.get("depth_mm"), "cause": f.get("cause") or "", "condition": f.get("condition"),
                  "urgency": f.get("urgency"), "status": f["status"], "source": f.get("source") or "",
                  "photos": _photos_text(f), "us": _links_text(f), "notes": f.get("notes") or ""}
        if f["kind"] == "decay":
            out["degrado"] += [(p, values) for p in _parts(g, "Polygon")]
        elif f["kind"] == "crack":
            out["lesioni"] += [(p, values) for p in _parts(g, "LineString")]
        elif f["kind"] == "note":
            out["note"] += [(p, values) for p in _parts(g, "Point")]
        elif f["kind"] == "treatment":
            tr = f.get("treatment") or {}
            extra = {"technique": tr.get("technique") or f["code"], "materials": tr.get("materials") or "",
                     "treatment_date": tr.get("date") or "", "done": 1 if tr.get("done") else 0,
                     "done_by": tr.get("by") or ""}
            for p in _parts(g, "Polygon"):
                out["interventi"].append((p, dict(values, shape="area", **extra)))
            for line in _parts(g, "LineString"):
                out["interventi"].append((band(line, (f.get("width_mm") or 0.0) / 1000.0),
                                          dict(values, shape="line", **extra)))
    return out


def _columns(table: str) -> dict:
    return dict(COLUMNS, **(TREATMENT_COLUMNS if table == "interventi" else {}))


def _fresh(path: Path) -> None:
    from pyarchinit_photogrammetry.io import gpkg

    if path.exists():
        gpkg.refuse_foreign(path)
        path.unlink()


def geopackages(chunk, cid: str, folder, stem: str) -> dict:
    """``<stem>.gpkg`` (unfolded: ``(u, v)`` metres, srs -1, with ``geom_world``, the WKT of the geometry on the
    plane in the chunk's system) and ``<stem>_mondo.gpkg`` (3D in the chunk's system, through the program's
    ``write_world_tables``), tables ``degrado`` (POLYGON), ``lesioni`` (LINESTRING), ``interventi`` (POLYGON),
    ``note`` (POINT). Returns ``{"gpkg", "gpkg_world"}``."""
    import shapely

    from pyarchinit_photogrammetry.engine.elevation import write_world_tables
    from pyarchinit_photogrammetry.io import gpkg

    ctx = _context(chunk, cid)
    entry = ctx["entry"]
    rows = feature_rows(chunk, cid)
    place = _on_plane(chunk, entry)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    flat, world = folder / f"{stem}.gpkg", folder / f"{stem}_mondo.gpkg"
    for path in (flat, world):
        _fresh(path)
    con = gpkg.connect(flat)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, gpkg.srs_of(None))
        for table, (gtype, _style, _kinds) in TABLES.items():
            columns = dict(_columns(table), geom_world="TEXT")
            cols = gpkg.ensure_table(con, table, gtype, columns, -1, recreate=True)
            feats = [(g, dict(v, geom_world=shapely.to_wkt(place(g), rounding_precision=4))) for g, v in rows[table]]
            gpkg.replace_rows(con, table, cols, feats, -1)
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    srs = gpkg.srs_of(entry.get("crs") or chunk.crs)
    con = gpkg.connect(world)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, srs)
        write_world_tables(con, {table: (gtype, _columns(table), [(place(g), v) for g, v in rows[table]])
                                 for table, (gtype, _s, _k) in TABLES.items()}, srs[0])
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return {"gpkg": flat, "gpkg_world": world}


# ---------------------------------------------------------------- QGIS
def qgis_styles(folder, codes, legend: dict, chunk=None) -> dict:
    """A QGIS style per table (``<table>.qml`` in ``folder``, through :func:`.vocabulary.write_qml`: categories on
    the column ``code`` with the legend's colours and hatches). ``codes``: ``{table: codes}``, or the same codes for
    the four tables. Returns ``{table: path}``.

    QGIS does not load a ``.qml`` by itself when the name differs from the GeoPackage's: :func:`qgis_project` puts the
    styles into the project it writes; otherwise *Proprietà del layer > Simbologia > Stile > Carica stile* reads
    them."""
    from .vocabulary import write_qml

    folder = Path(folder)
    per_table = codes if isinstance(codes, dict) else {table: list(codes) for table in TABLES}
    out = {}
    for table, found in per_table.items():
        if table not in TABLES:
            raise ValueError(f"the tables are {', '.join(TABLES)}, not {table!r}")
        out[table] = write_qml(folder / f"{table}.qml", list(found), legend, geometry=TABLES[table][1], chunk=chunk)
    return out


def _renderer(qml: Path) -> str | None:
    m = re.search(r"<renderer-v2\b.*?</renderer-v2>", qml.read_text(encoding="utf-8"), re.S)
    return m.group(0) if m else None


def apply_styles(project: Path, gpkg_path: Path, styles: dict) -> int:
    """The renderer of each layer of ``project`` reading a table of ``gpkg_path`` replaced by its style's (the
    program's project builder gives every vector a single colour). Returns the layers restyled."""
    text = Path(project).read_text(encoding="utf-8")
    done = 0

    def restyle(m):
        nonlocal done
        block = m.group(0)
        source = re.search(r"<datasource>(.*?)</datasource>", block, re.S)
        if source is None or Path(gpkg_path).name not in source.group(1):
            return block
        layer = re.search(r"\|layername=([^|<]+)", source.group(1))
        if layer is None or layer.group(1) not in styles:
            return block
        renderer = _renderer(Path(styles[layer.group(1)]))
        if renderer is None or not re.search(r"<renderer-v2\b.*?</renderer-v2>", block, re.S):
            return block
        done += 1
        return re.sub(r"<renderer-v2\b.*?</renderer-v2>", lambda _m: renderer, block, count=1, flags=re.S)

    text = re.sub(r"<maplayer\b.*?</maplayer>", restyle, text, flags=re.S)
    Path(project).write_text(text, encoding="utf-8")
    return done


def qgis_project(chunk, cid: str, folder, open_it: bool = True) -> Path:
    """The elevation's QGIS project as the program makes it (its orthophoto and the program's drawings of the
    elevation, ``export_drawings(..., raster=True)``) with the campaign's tables and their styles, in ``folder``;
    opened in QGIS when ``open_it``. NotReady without QGIS."""
    from pyarchinit_photogrammetry.engine.elevation import _safe_stem, elevation_paths, export_drawings
    from pyarchinit_photogrammetry.io import qgis_link

    from .vocabulary import legend

    ctx = _context(chunk, cid)
    entry, c, leg = ctx["entry"], ctx["campaign"], ctx["legend"]
    q = qgis_link.find_qgis()
    if q is None:
        raise records.NotReady(t("QGIS non trovato: installalo da qgis.org (o indica il percorso con PPG_QGIS).",
                                 "QGIS not found: install it from qgis.org (or give its path with PPG_QGIS)."))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    base = f"{_safe_stem(entry.get('name') or entry['id'])}_{entry['id']}"
    files = export_drawings(chunk, entry, folder, stem=base, raster=True)
    ortho, _relief = elevation_paths(chunk, entry)
    rasters = [{"name": entry.get("name") or entry["id"], "path": str(ortho)}] if ortho.exists() else []
    vectors = [{"name": t(title, title), "path": str(files["gpkg"]), "table": table, "geometry": geometry,
                "color": colour, "label": "us" if table == "unita" else ""}
               for table, (title, geometry, colour) in qgis_link.ELEVATION_TABLES.items()
               if table != "profilo" or entry.get("kind") == "section"]
    stem = _stem(ctx)
    ours = geopackages(chunk, cid, folder, f"{stem}_conservazione")
    rows = feature_rows(chunk, cid)
    codes = {table: list(dict.fromkeys(v["code"] for _g, v in rows[table])) for table in TABLES}
    styles = qgis_styles(folder, codes, leg or legend(chunk), chunk=chunk)
    for table, (gtype, _style, _kinds) in TABLES.items():
        first = codes[table][0] if codes[table] else None
        colour = (leg.get(first) or {}).get("colour") if first else None
        vectors.append({"name": f"{_tr(TABLE_TITLES[table])} - {c['name']}", "path": str(ours["gpkg"]),
                        "table": table, "geometry": gtype, "color": colour or "#808080", "label": ""})
    project = folder / f"{stem}_conservazione.qgs"
    qgis_link.build_project(project, "", {}, rasters, title=f"{entry.get('name') or entry['id']} - {c['name']}",
                            qgis=q, extra_vectors=vectors)
    apply_styles(project, ours["gpkg"], styles)
    if open_it:
        qgis_link.open_in_qgis(project, q)
    return project


# ---------------------------------------------------------------- everything
def export_campaign(chunk, cid: str, folder=None, scale: int | None = None) -> dict:
    """The plate, the tables, the DXF, the two GeoPackages and the QGIS styles of the campaign in ``folder`` (default
    ``<chunk data>/conservation/<eid>/<cid>/``). Returns the paths: ``plate_svg``, ``plate_pdf`` (or ``plate_png``),
    ``scale``, ``xlsx``, ``csv_features``, ``csv_areas``, ``dxf``, ``gpkg``, ``gpkg_world``, ``styles``."""
    ctx = _context(chunk, cid)
    folder = Path(folder) if folder else records.campaign_folder(chunk, ctx["entry"]["id"], cid)
    folder.mkdir(parents=True, exist_ok=True)
    stem = _stem(ctx)
    out = {"unnamed": ctx["unnamed"]}
    made = plate(chunk, cid, folder, scale, stem=f"{stem}_{t('tavola', 'plate')}")
    out["scale"] = made.pop("scale")
    out.update({f"plate_{k}": v for k, v in made.items()})
    out.update(tables(chunk, cid, folder, stem=f"{stem}_{t('tabelle', 'tables')}"))
    out["dxf"] = dxf(chunk, cid, folder / f"{stem}.dxf")
    out.update(geopackages(chunk, cid, folder, f"{stem}_conservazione"))
    rows = feature_rows(chunk, cid)
    out["styles"] = qgis_styles(folder, {table: list(dict.fromkeys(v["code"] for _g, v in rows[table]))
                                         for table in TABLES}, ctx["legend"], chunk=chunk)
    return out


__all__ = ["plate", "tables", "table_data", "dxf", "geopackages", "feature_rows", "qgis_styles", "qgis_project",
           "apply_styles", "export_campaign", "choose_scale", "legend_line", "glossary_names", "SHEET", "SCALES",
           "TABLES"]
