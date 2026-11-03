"""``ppg plugin conservation list|campaign|map|detect|plate|export|pyarchinit|legend PROJECT ...``: the plugin's steps
from the command line, the same functions as the pane's. Common options: ``--chunk LABEL``, ``--elevation ID|NAME``,
``--campaign ID|NAME`` (default: the only elevation, the latest campaign of the elevation). Geometry in ``(u, v)``
metres of the elevation, from WKT (text or a file holding it) or a GeoPackage table. What is missing is printed as one
line and the exit code is 1 (2 for an unknown chunk)."""
from __future__ import annotations

import argparse
from pathlib import Path

from . import records
from .records import NotReady
from .texts import num, t

COMMANDS = ("list", "campaign", "map", "detect", "plate", "export", "pyarchinit", "legend")


def parser() -> argparse.ArgumentParser:
    options = argparse.ArgumentParser(add_help=False)
    options.add_argument("--chunk", help="the label of the chunk (default: the active one)")
    options.add_argument("--elevation", help="the elevation: its id or name (default: the only one)")
    options.add_argument("--campaign", help="the campaign: its id or name (default: the latest of the elevation)")
    common = argparse.ArgumentParser(add_help=False, parents=[options])
    common.add_argument("project", help="the .ppg file")
    p = argparse.ArgumentParser(prog="ppg plugin conservation", description="decay mapping on elevations")
    sub = p.add_subparsers(dest="what", required=True)
    sub.add_parser("list", parents=[common], help="the campaigns and their features")
    s = sub.add_parser("campaign", parents=[common], help="the campaigns of an elevation; --new makes one")
    s.add_argument("--new", action="store_true")
    s.add_argument("--date", help="YYYY-MM-DD (default: the date of the chunk)")
    s.add_argument("--name")
    s.add_argument("--surveyor", default="")
    s.add_argument("--method", default="")
    s.add_argument("--notes", default="")
    s = sub.add_parser("map", parents=[common], help="a feature with its term, from WKT or a GeoPackage table")
    s.add_argument("--term", required=True, help="the code of the term (ISCS..., UNI..., LOCAL..., INT...)")
    s.add_argument("--wkt", help="the geometry in WKT, or a file holding it")
    s.add_argument("--gpkg", help="a GeoPackage in (u, v): every row of --table is a feature")
    s.add_argument("--table")
    s.add_argument("--kind", choices=records.FEATURE_KINDS, help="default: from the term and the geometry")
    s.add_argument("--intensity", choices=records.INTENSITIES)
    s.add_argument("--width-mm", type=float)
    s.add_argument("--depth-mm", type=float)
    s.add_argument("--notes", default="")
    s.add_argument("--done", action="store_true", help="a treatment already done")
    s = sub.add_parser("detect", parents=[options], help="proposals of a detector (accepted with --accept): "
                       "detect cracks|loss|colours|sam PROJECT")
    s.add_argument("detector", choices=("cracks", "loss", "colours", "sam"))
    s.add_argument("project", help="the .ppg file")
    s.add_argument("--bounds", help="u0,v0,u1,v1 (default: the whole face)")
    s.add_argument("--accept", action="store_true", help="accept the proposals that have a term")
    s.add_argument("--name", action="append", default=[], help="colours: CLASS=CODE (a class named, 1-based)")
    s.add_argument("--k", type=int, default=5)
    s.add_argument("--min-length-cm", type=float, default=10.0)
    s.add_argument("--min-depth-cm", type=float, default=2.0)
    s.add_argument("--min-area-m2", type=float, default=0.01)
    s = sub.add_parser("plate", parents=[common], help="the plate (SVG, PDF)")
    s.add_argument("--scale", type=int)
    s.add_argument("--folder")
    s = sub.add_parser("export", parents=[common], help="plate, tables, DXF, GeoPackages, QGIS styles")
    s.add_argument("--scale", type=int)
    s.add_argument("--folder")
    s = sub.add_parser("pyarchinit", parents=[common], help="media and US forms into PyArchInit")
    s.add_argument("--sito", help="the site (default: the project's)")
    s.add_argument("--words", help="the four words of CC 0-3, separated by commas")
    s.add_argument("--config", help="PyArchInit's config.cfg (default: PyArchInit's own)")
    s = sub.add_parser("legend", parents=[common], help="the legend of the project; --set changes a term")
    s.add_argument("--set", action="append", default=[], help="CODE=#rrggbb[:hatch]")
    s.add_argument("--qml", help="a folder: the QGIS styles of the four tables")
    return p


# ---------------------------------------------------------------- choosing
def _elevation(chunk, key: str | None) -> dict:
    found = chunk.meta.get("elevations") or []
    if key:
        for e in found:
            if key in (e.get("id"), e.get("name")):
                return e
        raise NotReady(t(f"Nessun prospetto {key!r}.", f"No elevation {key!r}."))
    if len(found) == 1:
        return found[0]
    with_campaigns = {c["elevation"] for c in records.campaigns(chunk)}
    if len(with_campaigns) == 1:
        return next(e for e in found if e.get("id") in with_campaigns)
    raise NotReady(t("Indica il prospetto con --elevation (id o nome).", "Give the elevation with --elevation (id "
                                                                         "or name)."))


def _campaign(chunk, a) -> dict:
    if a.campaign:
        for c in records.campaigns(chunk):
            if a.campaign in (c["id"], c["name"]) and (not a.elevation or
                                                        _elevation(chunk, a.elevation)["id"] == c["elevation"]):
                return c
        raise NotReady(t(f"Nessuna campagna {a.campaign!r}.", f"No campaign {a.campaign!r}."))
    entry = _elevation(chunk, a.elevation)
    found = records.campaigns(chunk, entry["id"])
    if not found:
        raise NotReady(t("Nessuna campagna su questo prospetto: campaign --new.",
                         "No campaign on this elevation: campaign --new."))
    return found[-1]


def _bounds(text: str | None):
    if not text:
        return None
    values = [float(x) for x in text.replace(";", ",").split(",")]
    if len(values) != 4:
        raise ValueError(t("--bounds vuole u0,v0,u1,v1.", "--bounds wants u0,v0,u1,v1."))
    return tuple(values)


def _kind(term, geom, given: str | None) -> str:
    if given:
        return given
    if term.kind == "treatment":
        return "treatment"
    if geom.geom_type == "Point":
        return "note"
    return term.kind


# ---------------------------------------------------------------- the commands
def _list(doc, chunk, a) -> int:
    from . import texts
    from .vocabulary import load

    vocab = load(chunk)
    lang = texts.lang()
    names = {e["id"]: e.get("name", e["id"]) for e in chunk.meta.get("elevations") or []}
    found = records.campaigns(chunk)
    if not found:
        print(t("Nessuna campagna.", "No campaign."))
    for c in found:
        feats = records.features(chunk, c["id"])
        print(f"{c['id']}  {c['date']}  {c['name']}  [{names.get(c['elevation'], c['elevation'])}]  {len(feats)}")
        for f in feats:
            label = vocab.name(f["code"], lang) if f["code"] in vocab else f["code"]
            extent = "" if f["extent"] is None else f"  {f['extent']:.3f}"
            print(f"    {f['id']}  {f['kind']:<9} {f['status']:<8} {label}{extent}")
    return 0


def _campaign_cmd(doc, chunk, a) -> int:
    entry = _elevation(chunk, a.elevation)
    if a.new:
        c = records.new_campaign(chunk, entry["id"], date=a.date, name=a.name, surveyor=a.surveyor, method=a.method,
                                 notes=a.notes)
        doc.save()
        print(t(f"Campagna {c['id']} «{c['name']}» del {c['date']} su {entry.get('name')}",
                f"Campaign {c['id']} “{c['name']}” of {c['date']} on {entry.get('name')}"))
        return 0
    found = records.campaigns(chunk, entry["id"])
    if not found:
        print(t("Nessuna campagna.", "No campaign."))
    for c in found:
        cl = c.get("classes") or {}
        print(f"{c['id']}  {c['date']}  {c['name']}  CC {cl.get('cc', '-')}  UC {cl.get('uc', '-')}  "
              f"{len(records.features(chunk, c['id']))}")
    return 0


def _map(doc, chunk, a) -> int:
    from .geometry import as_geometry
    from .vocabulary import load

    c = _campaign(chunk, a)
    term = load(chunk).term(a.term)
    geoms = []
    if a.wkt:
        text = a.wkt
        if Path(text).is_file():
            text = Path(text).read_text(encoding="utf-8").strip()
        geoms.append(as_geometry(text))
    elif a.gpkg:
        from pyarchinit_photogrammetry.io import gpkg

        if not a.table:
            raise ValueError(t("--gpkg vuole anche --table.", "--gpkg needs --table too."))
        geoms += [g for g, _attrs in gpkg.read_table(a.gpkg, a.table) if g is not None and not g.is_empty]
    else:
        raise ValueError(t("La geometria: --wkt oppure --gpkg con --table.", "The geometry: --wkt or --gpkg with "
                                                                              "--table."))
    made = []
    for g in geoms:
        kind = _kind(term, g, a.kind)
        treatment = {"technique": term.code, "materials": "", "date": c["date"], "done": bool(a.done), "by": ""} \
            if kind == "treatment" else None
        made.append(records.add_feature(chunk, c["id"], kind, term.code, g, intensity=a.intensity,
                                        width_mm=a.width_mm, depth_mm=a.depth_mm, notes=a.notes,
                                        treatment=treatment))
    entry = records._elevation(chunk, c["elevation"])
    for f in made:
        records.link_units(chunk, entry, f["id"])
    doc.save()
    for f in made:
        extent = "" if f["extent"] is None else f"  {num(f['extent'], 3)}"
        print(f"{f['id']}  {f['kind']}  {f['code']}{extent}")
    return 0


def _detect(doc, chunk, a) -> int:
    from pyarchinit_photogrammetry.core.progress import print_progress

    from . import detect
    from .vocabulary import UNNAMED

    c = _campaign(chunk, a)
    entry = records._elevation(chunk, c["elevation"])
    bounds = _bounds(a.bounds)
    name = a.detector
    if name == "cracks":
        found = detect.cracks(chunk, entry, bounds, min_length=a.min_length_cm / 100.0, progress=print_progress)
    elif name == "loss":
        found = detect.material_loss(chunk, entry, bounds, min_depth=a.min_depth_cm / 100.0,
                                     min_area=a.min_area_m2, progress=print_progress)
    elif name == "colours":
        named = {}
        for item in a.name:
            cls, _, code = item.partition("=")
            named[int(cls) - 1] = code.strip()
        found = [{"kind": "decay", "code": named.get(i), "geometry": g, "width_mm": None, "depth_mm": None,
                  "notes": cl["notes"], "coverage": cl["coverage"]}
                 for i, cl in enumerate(detect.colour_classes(chunk, entry, bounds, k=a.k, progress=print_progress))
                 for g in cl["polygons"]]
    else:
        from pyarchinit_photogrammetry.ai.sam_env import make_segmenter

        found = detect.sam_areas(chunk, entry, bounds, progress=print_progress, segmenter=make_segmenter())
    source = "proposed:" + {"loss": "material_loss"}.get(name, name)
    made = []
    for p in found:
        cover = float(p.get("coverage") or 0.0)
        made.append(records.add_feature(
            chunk, c["id"], p["kind"], p["code"] or UNNAMED, p["geometry"], status="proposed", source=source,
            width_mm=None if p.get("width_mm") is None else round(float(p["width_mm"]), 1),
            depth_mm=p.get("depth_mm"), notes=p["notes"] + t(f"; rilievo sul {num(cover * 100, 0)} % della finestra",
                                                             f"; relief on {num(cover * 100, 0)} % of the window")))
    accepted = 0
    if a.accept:
        accepted = records.accept(chunk, [f["id"] for f in made if f["code"] != UNNAMED])
    doc.save()
    unnamed = sum(1 for f in made if f["code"] == UNNAMED)
    print(t(f"{len(made)} proposte ({accepted} accettate, {unnamed} da nominare)",
            f"{len(made)} proposals ({accepted} accepted, {unnamed} to be named)"))
    for f in made:
        extent = "" if f["extent"] is None else f"  {num(f['extent'], 3)}"
        print(f"    {f['id']}  {f['kind']}  {f['status']}  {f['code']}{extent}")
    return 0


def _plate(doc, chunk, a) -> int:
    from . import outputs

    c = _campaign(chunk, a)
    out = outputs.plate(chunk, c["id"], a.folder, a.scale)
    print(t(f"Tavola in scala 1:{out['scale']}", f"Plate at 1:{out['scale']}"))
    for key in ("svg", "pdf", "png"):
        if key in out:
            print(f"  {out[key]}")
    return 0


def _export(doc, chunk, a) -> int:
    from . import outputs

    c = _campaign(chunk, a)
    out = outputs.export_campaign(chunk, c["id"], a.folder, a.scale)
    for key, value in out.items():
        if key == "styles":
            for path in value.values():
                print(f"  {path}")
        elif isinstance(value, Path):
            print(f"  {value}")
    if out.get("unnamed"):
        print(t(f"{out['unnamed']} elementi da nominare esclusi", f"{out['unnamed']} features to be named left out"))
    return 0


def _pyarchinit(doc, chunk, a) -> int:
    from pyarchinit_photogrammetry.pyarchinit.bridge import PyArchInit

    from . import pyarchinit

    c = _campaign(chunk, a)
    words = [w.strip() for w in a.words.split(",")] if a.words else None
    pa = PyArchInit.connect(config_path=a.config) if a.config else PyArchInit.connect()
    out = pyarchinit.save(pa, chunk, c["id"], sito=a.sito, words=words)
    doc.save()
    print(t(f"Media registrati: {', '.join(map(str, out['media'])) or '-'}",
            f"Media registered: {', '.join(map(str, out['media'])) or '-'}"))
    for area, us, tipo, id_us in out["us"]:
        print(t(f"  {tipo} {us} (area {area}): aggiornata", f"  {tipo} {us} (area {area}): updated")
              + (f" - {out['word']}" if out["word"] else ""))
    for area, us, tipo in out["missing"]:
        print(t(f"  {tipo} {us} (area {area}): scheda non trovata", f"  {tipo} {us} (area {area}): no form found"))
    return 0


def _legend(doc, chunk, a) -> int:
    from . import outputs, texts
    from .vocabulary import legend, load, set_legend

    for item in a.set:
        code, _, rest = item.partition("=")
        colour, _, hatch = rest.partition(":")
        set_legend(chunk, code.strip(), colour=colour.strip() or None,
                   hatch={"mpl": hatch} if hatch else None)
    if a.set:
        doc.save()
    vocab = load(chunk)
    leg = legend(chunk)
    used = list(dict.fromkeys(f["code"] for f in records.features(chunk)))
    for code in used or [tm.code for tm in vocab.terms()]:
        s = leg.get(code) or {}
        name = vocab.name(code, texts.lang()) if code in vocab else code
        print(f"{code:<34} {s.get('colour', ''):<8} {(s.get('hatch') or {}).get('mpl', ''):<5} {name}")
    if a.qml:
        rows = {}
        for f in records.features(chunk, status="accepted"):
            table = next(k for k, v in outputs.TABLES.items() if f["kind"] in v[2])
            rows.setdefault(table, []).append(f["code"])
        codes = {table: list(dict.fromkeys(rows.get(table) or [])) for table in outputs.TABLES}
        for path in outputs.qgis_styles(a.qml, codes, leg, chunk=chunk).values():
            print(f"  {path}")
    return 0


HANDLERS = {"list": _list, "campaign": _campaign_cmd, "map": _map, "detect": _detect, "plate": _plate,
            "export": _export, "pyarchinit": _pyarchinit, "legend": _legend}


def run(args) -> int:
    """The command line: parse ``args`` and run the step; NotReady and the errors of the data as one line."""
    import pyarchinit_photogrammetry as ppg

    a = parser().parse_args(list(args))
    doc = ppg.Document.open(a.project)
    chunk = next((c for c in doc.chunks if c.label == a.chunk), None) if a.chunk else doc.chunk
    if chunk is None:
        print(f"no chunk {a.chunk!r}")
        return 2
    try:
        return int(HANDLERS[a.what](doc, chunk, a) or 0)
    except (NotReady, ValueError, KeyError, FileExistsError) as exc:
        print(str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))
        return 1


__all__ = ["run", "parser", "COMMANDS"]
