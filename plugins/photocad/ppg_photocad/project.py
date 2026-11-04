"""A conversion of the project: the image rendered, the confirmation, the upload, the wait, the drawing brought
back and kept. No Qt here.

What is made lives in ``<chunk data>/photocad/<id>/``: ``image.jpg`` (what was sent), ``photocad.dxf`` (as
PhotoCAD gave it, metres from the lower-left corner of the image), ``drawing.dxf`` (georeferenced),
``drawing.gpkg`` (table ``photocad``; ``photocad_3d`` for an elevation in 3D), ``drawing.svg`` with
``drawing.svg.json`` (where it lies), ``drawing_3d.dxf``, ``drawing.pdf`` when asked, ``conversion.json`` (the
source, the bounds, the pixel, the date, the credits used - never the key). The list of the conversions is in
``chunk.meta["photocad"]["conversions"]``; whether to ask before each upload in ``doc.meta["photocad"]["confirm"]``.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import client as client_mod
from . import drawing, source
from .texts import t


class Declined(RuntimeError):
    """The user did not confirm the upload: nothing was sent."""


# ------------------------------------------------------------------ the project's settings
def confirmation_wanted(doc) -> bool:
    """True unless the user chose not to be asked again for this project."""
    return bool(((getattr(doc, "meta", None) or {}).get("photocad") or {}).get("confirm", True))


def set_confirmation_wanted(doc, wanted: bool) -> None:
    doc.meta.setdefault("photocad", {})["confirm"] = bool(wanted)


# ------------------------------------------------------------------ records
def _state(chunk, create: bool = False) -> dict:
    if create:
        return chunk.meta.setdefault("photocad", {})
    return chunk.meta.get("photocad") or {}


def records(chunk) -> list[dict]:
    return list(_state(chunk).get("conversions") or [])


def record(chunk, rid: str) -> dict | None:
    return next((r for r in records(chunk) if r.get("id") == rid), None)


def folder_of(chunk, rec: dict) -> Path:
    return chunk.data_dir / rec["folder"]


def path_of(chunk, rec: dict, what: str) -> Path | None:
    rel = (rec.get("files") or {}).get(what)
    return None if not rel else folder_of(chunk, rec) / rel


def remove(chunk, rid: str) -> bool:
    """The conversion out of the project with its folder."""
    state = _state(chunk)
    rec = record(chunk, rid)
    if rec is None:
        return False
    state["conversions"] = [r for r in state.get("conversions") or [] if r.get("id") != rid]
    folder = folder_of(chunk, rec)
    if folder.is_dir() and folder.parent == chunk.data_dir / "photocad":
        shutil.rmtree(folder, ignore_errors=True)
    return True


def features(chunk, rec: dict):
    """``(shapely geometry, attributes)`` of the drawing in the frame it was made on: the coordinates of the chunk
    (plan) or the ``(u, v)`` of the elevation ``rec["frame"]``."""
    from pyarchinit_photogrammetry.io import gpkg

    path = path_of(chunk, rec, "gpkg")
    if path is None or not path.exists():
        return []
    return [(g, a) for g, a in gpkg.read_table(path, drawing.TABLE) if g is not None]


def summary(rec: dict) -> str:
    c = rec.get("counts") or {}
    where = t("pianta", "plan") if rec.get("frame") is None else t("prospetto {0}", "elevation {0}").format(rec["frame"])
    return t("{0}: {1} poligoni, {2} linee ({3}, {4} cm/px)", "{0}: {1} polygons, {2} lines ({3}, {4} cm/px)").format(
        rec.get("name"), c.get("polygons", 0), c.get("lines", 0), where,
        f"{float((rec.get('image') or {}).get('pixel', 0)) * 100:.2f}")


# ------------------------------------------------------------------ the conversion
def _client(key):
    return client_mod.Client(key)


def prepare(chunk, src: str, key: str, bbox=None, fmt: str = "jpeg") -> dict:
    """The image of ``src`` (and ``bbox``) and the credits on the account (the key is checked here, before
    anything is sent): ``{"rendered", "credits", "source", "bbox"}``."""
    rendered = source.render(chunk, src, bbox, fmt=fmt)
    credits = client_mod.credits(_client(key).account())
    return {"rendered": rendered, "credits": credits, "source": src, "bbox": None if bbox is None else list(bbox)}


def confirm_info(prep: dict) -> dict:
    """What the user is told before the upload: the text and its numbers."""
    r = prep["rendered"]
    credits = prep.get("credits")
    text = t("L'immagine ({0} x {1} px, {2} kB, {3} cm/px) sarà inviata a photocad.ai, un servizio esterno, per "
             "vettorializzarla. Una conversione usa un credito (circa 2 $). Crediti rimasti: {4}.",
             "The image ({0} x {1} px, {2} kB, {3} cm/px) will be sent to photocad.ai, an external service, to "
             "vectorise it. One conversion uses one credit (about $2). Credits left: {4}.").format(
        r.width, r.height, max(1, round(len(r.data) / 1024)), f"{r.pixel * 100:.2f}",
        "?" if credits is None else credits)
    return {"text": text, "credits": credits, "width": r.width, "height": r.height, "bytes": len(r.data),
            "pixel": r.pixel, "source": prep["source"]}


def _name(chunk, src: str) -> str:
    if src == "ortho":
        base = t("Ortofoto", "Orthophoto")
    elif src == "dem":
        base = "DEM"
    elif src.startswith("elevation:"):
        base = str(source.elevation_entry(chunk, src.split(":", 1)[1]).get("name") or src)
    else:
        base = t("Rilievo", "Relief")
    taken = {r.get("name") for r in records(chunk)}
    n = 1
    while f"{base} {n}" in taken:
        n += 1
    return f"{base} {n}"


def _new_id(chunk) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    taken = {r.get("id") for r in records(chunk)}
    rid, n = stamp, 1
    while rid in taken or (chunk.data_dir / "photocad" / rid).exists():
        n += 1
        rid = f"{stamp}_{n}"
    return rid


def upload(chunk, prep: dict, key: str, detailed: bool = True, remove_background: bool = False,
           three_d: bool = False, pdf: bool = False, progress=None, name: str | None = None) -> dict:
    """Send the prepared image, wait for PhotoCAD, bring the drawing back and keep it: the record (also in
    ``chunk.meta["photocad"]["conversions"]``). ``progress(fraction, message)`` raises ``Cancelled`` when the
    user cancels: the conversion is deleted on PhotoCAD then."""
    progress = progress or (lambda *a: None)
    r = prep["rendered"]
    c = _client(key)
    progress(0.02, t("Invio dell'immagine a PhotoCAD", "Sending the image to PhotoCAD"))
    cid = c.convert(r.data, r.filename, r.content_type, detailed=detailed, remove_background=remove_background)
    done = client_mod.wait(c, cid, progress)
    progress(0.92, t("Scarico il disegno", "Downloading the drawing"))
    params = {"units": "m", "longest": float(r.longest_m), "margin": 0}
    rid = _new_id(chunk)
    folder = chunk.data_dir / "photocad" / rid
    folder.mkdir(parents=True, exist_ok=True)
    try:
        (folder / r.filename).write_bytes(r.data)
        raw = folder / "photocad.dxf"
        raw.write_bytes(c.file(cid, "dxf", **params))
        files = {"image": r.filename, "dxf_photocad": raw.name}
        left = drawing.georeference_dxf(raw, folder / "drawing.dxf", r.x0, r.y0)
        files["dxf"] = "drawing.dxf"
        items = drawing.paths(folder / "drawing.dxf", tolerance=max(r.pixel / 4, 1e-4))
        feats = drawing.geometries(items)
        drawing.write_gpkg(folder / "drawing.gpkg", feats, r.crs)
        files["gpkg"] = "drawing.gpkg"
        try:
            (folder / "drawing.svg").write_bytes(c.file(cid, "svg", **params))
            drawing.svg_note(folder / "drawing.svg.json", r, r.crs, r.frame)
            files["svg"], files["svg_georef"] = "drawing.svg", "drawing.svg.json"
        except client_mod.PhotoCADError as exc:  # the DXF is what matters: the SVG is a courtesy
            progress(0.95, t("SVG non scaricato: {0}", "SVG not downloaded: {0}").format(exc))
        if pdf:
            (folder / "drawing.pdf").write_bytes(c.file(cid, "pdf"))
            files["pdf"] = "drawing.pdf"
        if three_d and r.frame is not None:
            entry = source.elevation_entry(chunk, r.frame)
            items3d = drawing.world_items(chunk, entry, [it for it in items if len(it["xy"]) >= 2])
            drawing.write_dxf_3d(folder / "drawing_3d.dxf", items3d)
            drawing.write_gpkg_3d(folder / "drawing.gpkg", items3d, chunk.crs)
            files["dxf_3d"] = "drawing_3d.dxf"
    except BaseException:
        shutil.rmtree(folder, ignore_errors=True)
        raise
    polygons = sum(1 for g, _a in feats if g.geom_type == "Polygon")
    rec = {"id": rid, "name": name or _name(chunk, prep["source"]), "photocad_id": cid, "source": prep["source"],
           "frame": r.frame, "crs": r.crs, "bbox": prep.get("bbox"), "folder": f"photocad/{rid}", "files": files,
           "image": r.describe(), "options": {"detailed": bool(detailed), "remove_background": bool(remove_background),
                                              "three_d": bool(three_d and r.frame is not None), "pdf": bool(pdf)},
           "page_mm": [done.get("width_mm"), done.get("height_mm")], "dxf_request": params,
           "credits_used": 1, "credits_remaining": client_mod.credits(done),
           "counts": {"polygons": polygons, "lines": len(feats) - polygons, "left_out": left},
           "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "visible": True}
    (folder / "conversion.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    _state(chunk, True).setdefault("conversions", []).append(rec)
    progress(1.0, summary(rec))
    return dict(rec)


def convert(chunk, src: str, key: str, bbox=None, detailed: bool = True, remove_background: bool = False,
            three_d: bool = False, pdf: bool = False, confirm=None, progress=None, name: str | None = None) -> dict:
    """:func:`prepare`, ``confirm(info)`` (:func:`confirm_info`; False: :class:`Declined`, nothing sent; None:
    no question), :func:`upload`."""
    prep = prepare(chunk, src, key, bbox)
    if confirm is not None and not confirm(confirm_info(prep)):
        raise Declined(t("Conversione annullata: nulla è stato inviato.", "Conversion cancelled: nothing was sent."))
    return upload(chunk, prep, key, detailed, remove_background, three_d, pdf, progress, name)


# ------------------------------------------------------------------ out of the project
def export(chunk, rid: str, out) -> Path:
    """The drawing as ``out`` (.dxf georeferenced, .gpkg, .svg with its note, .pdf when downloaded)."""
    rec = record(chunk, rid)
    if rec is None:
        raise ValueError(t("Nessuna conversione «{0}».", "No conversion '{0}'.").format(rid))
    out = Path(out)
    what = {".dxf": "dxf", ".gpkg": "gpkg", ".svg": "svg", ".pdf": "pdf"}.get(out.suffix.lower())
    if what is None:
        raise ValueError(t("Formato non previsto: {0} (.dxf, .gpkg, .svg, .pdf)",
                           "Format not provided for: {0} (.dxf, .gpkg, .svg, .pdf)").format(out.suffix))
    src = path_of(chunk, rec, what)
    if src is None or not src.exists():
        raise ValueError(t("La conversione non ha il file {0}.", "The conversion has no {0} file.").format(what))
    out.parent.mkdir(parents=True, exist_ok=True)
    if what == "gpkg":
        from pyarchinit_photogrammetry.io import gpkg

        if out.exists() and out.stat().st_size and not gpkg.is_geopackage(out):
            gpkg.refuse_foreign(out)
    shutil.copyfile(src, out)
    if what == "svg":
        note = path_of(chunk, rec, "svg_georef")
        if note is not None and note.exists():
            shutil.copyfile(note, out.with_name(out.name + ".json"))
    return out


def unit_polygons(feats, tipo: str = "US", start: int | None = None, tolerance: float = 1e-6):
    """``(polygons, attributes, skipped)`` for ``context.add_units``: the closed outlines (polygons, and lines whose
    ends meet) with ``tipo``, the PhotoCAD layer and - from ``start`` - their numbers; open lines skipped and
    counted."""
    from shapely.geometry import Polygon

    polygons, attrs, skipped = [], [], 0
    for g, a in feats:
        geom = None
        if g.geom_type == "Polygon":
            geom = g
        elif g.geom_type == "LineString":
            xy = list(g.coords)
            if len(xy) >= 4 and Polygon(xy).area > 0 and \
                    abs(xy[0][0] - xy[-1][0]) <= tolerance and abs(xy[0][1] - xy[-1][1]) <= tolerance:
                geom = Polygon(xy)
        if geom is None or geom.is_empty or geom.area <= 0:
            skipped += 1
            continue
        row = {"tipo": tipo, "photocad_layer": str(a.get("layer") or "")}
        if start:
            row["us"] = str(int(start) + len(polygons))
        polygons.append(geom)
        attrs.append(row)
    return polygons, attrs, skipped
