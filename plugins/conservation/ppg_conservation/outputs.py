"""The outputs of a campaign: what a restorer hands in, made again from the records whenever asked (the files are
derived: the records in ``chunk.meta["conservation"]`` are the primary data). Only the accepted features are a
campaign's: the proposals not accepted are left out of every output.

* :func:`plate` (:mod:`.plate`) - the plate at scale on an A3 sheet, with the legend in both glossaries and the
  campaign's EN 16096 classes;
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
from pathlib import Path

import numpy as np

from . import records
from .texts import lang, num, t

LOG = logging.getLogger("ppg_conservation")

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
ELEVATION_TITLES_EN = {"unita": "Units", "quote": "Spot heights", "punti": "Measuring points", "pietre": "Stones",
                       "profilo": "Profile", "linee": "Lines"}  # the program's layers of an elevation, in English
FAMILY_ORDER = ("general", "crack_deformation", "detachment", "material_loss", "discoloration_deposit", "biological",
                "masonry", "intervention")


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


def unnamed_count(chunk, cid: str, status: str | None = "accepted") -> int:
    """The features of the campaign still to be named (``PPG.UNNAMED``): the accepted ones (left out of every output)
    by default, the proposals with ``status="proposed"``, all with None."""
    from .vocabulary import UNNAMED

    return sum(1 for f in records.features(chunk, cid, status=status) if f["code"] == UNNAMED)


def unnamed_text(chunk, cid: str) -> str:
    """The line the pane and the command line print about the features still to be named ("" when none)."""
    out = []
    left = unnamed_count(chunk, cid)
    proposed = unnamed_count(chunk, cid, "proposed")
    if left:
        out.append(t(f"{left} elementi da nominare esclusi", f"{left} features to be named left out"))
    if proposed:
        out.append(t(f"{proposed} proposte ancora da nominare (non sono nelle uscite)",
                     f"{proposed} proposals still to be named (not in the outputs)"))
    return "; ".join(out)


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


def crack_numbers(ctx: dict) -> dict:
    """The number of each crack of the campaign (``{feature id: n}``, from 1 in the order of :func:`_order`): the
    column *N.* of the *Lesioni* table and the label of a crack on a plate too crowded for its opening."""
    return {f["id"]: i for i, f in enumerate((f for f in _order(ctx) if f["kind"] == "crack"), 1)}


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
    crack_head = [t("N.", "No."), "id", t("Codice", "Code"), t("Termine", "Term"), t("Lunghezza (m)", "Length (m)"),
                  t("Apertura (mm)", "Opening (mm)"), t("Intensità", "Intensity"), t("Causa", "Cause"),
                  t("Urgenza", "Urgency"), "US", t("Note", "Notes")]
    crack_rows = []
    numbers = crack_numbers(ctx)
    for f in feats:
        if f["kind"] == "crack":
            g = glossary_names(vocab, f["code"], f.get("code_iscs"))
            crack_rows.append([numbers[f["id"]], f["id"], f["code"], g["it"] if lang() == "it" else g["en"], _r(f.get("extent")),
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
    vectors = [{"name": t(title, ELEVATION_TITLES_EN.get(table, title)), "path": str(files["gpkg"]), "table": table,
                "geometry": geometry, "color": colour, "label": "us" if table == "unita" else ""}
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
    filled = [table for table in TABLES if rows[table]]
    restyled = apply_styles(project, ours["gpkg"], {k: v for k, v in styles.items() if k in filled})
    if restyled < len(filled):
        LOG.warning("QGIS project %s: %d of the %d layers with features took the legend's style (the others keep "
                    "one colour)", project, restyled, len(filled))
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
    out = {"unnamed": ctx["unnamed"], "unnamed_proposed": unnamed_count(chunk, cid, "proposed")}
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


from .plate import SCALES, SHEET, choose_scale, legend_line, plate  # noqa: E402 - the plate in its own module
