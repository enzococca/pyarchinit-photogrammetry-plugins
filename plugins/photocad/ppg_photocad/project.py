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

try:
    from pyarchinit_photogrammetry.core.progress import Cancelled
except ImportError:  # pragma: no cover - every program with plugins has it
    class Cancelled(RuntimeError):  # noqa: N818
        pass


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
    if rec.get("status", "done") != "done":
        return []
    return [(g, a) for g, a in gpkg.read_table(path, drawing.TABLE) if g is not None]


STATUS = {"pending": ("inviata, non ancora finita", "sent, not finished yet"),
          "complete": ("finita, non scaricata", "finished, not downloaded"),
          "mismatch": ("disegno non corrispondente all'immagine", "drawing not matching the image")}


def summary(rec: dict) -> str:
    c = rec.get("counts") or {}
    if rec.get("status", "done") in STATUS:
        return f"{rec.get('name')}: {t(*STATUS[rec['status']])} (PhotoCAD {rec.get('photocad_id')})"
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


class FrameMismatch(client_mod.PhotoCADError):
    """PhotoCAD's drawing is not in the image's frame: nothing placed, the files kept for a diagnosis."""


def _hint(rec: dict) -> str:
    return t("La conversione PhotoCAD {0} è pagata e conservata nel progetto: recuperala con Plugin > PhotoCAD > "
             "Recupera una conversione, o `ppg plugin photocad fetch PROGETTO --id {0}`.",
             "The PhotoCAD conversion {0} is paid for and kept in the project: get it with Plugin > PhotoCAD > "
             "Fetch a conversion, or `ppg plugin photocad fetch PROJECT --id {0}`.").format(rec["photocad_id"])


def _write_json(chunk, rec: dict) -> None:
    folder = folder_of(chunk, rec)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "conversion.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")


def _forget(chunk, rec: dict) -> None:
    """A conversion that will never give a drawing (failed and refunded, deleted on PhotoCAD) out of the project."""
    remove(chunk, rec["id"])


def upload(chunk, prep: dict, key: str, detailed: bool = True, remove_background: bool = False,
           three_d: bool = False, pdf: bool = False, progress=None, name: str | None = None) -> dict:
    """Send the prepared image, wait for PhotoCAD, bring the drawing back and keep it: the record (also in
    ``chunk.meta["photocad"]["conversions"]``).

    The conversion is paid from the 202: its PhotoCAD id is written (``conversion.json``, status ``pending``, and the
    record) before the first poll. A failure after that keeps it - status ``pending`` (not yet finished) or
    ``complete`` (finished, not downloaded) or ``mismatch`` (the frame check refused it) - with the folder, and
    :func:`fetch` gets it later without a new upload. ``progress(fraction, message)`` raises ``Cancelled`` when the
    user cancels: before the conversion is complete it is deleted on PhotoCAD and forgotten, after it it is kept."""
    progress = progress or (lambda *a: None)
    r = prep["rendered"]
    c = _client(key)
    progress(0.02, t("Invio dell'immagine a PhotoCAD", "Sending the image to PhotoCAD"))
    cid = c.convert(r.data, r.filename, r.content_type, detailed=detailed, remove_background=remove_background)
    rid = _new_id(chunk)
    rec = {"id": rid, "name": name or _name(chunk, prep["source"]), "photocad_id": cid, "status": "pending",
           "source": prep["source"], "frame": r.frame, "crs": r.crs, "bbox": prep.get("bbox"),
           "folder": f"photocad/{rid}", "files": {"image": r.filename}, "image": r.describe(),
           "options": {"detailed": bool(detailed), "remove_background": bool(remove_background),
                       "three_d": bool(three_d and r.frame is not None), "pdf": bool(pdf)},
           "credits_used": 1, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "visible": True}
    folder = folder_of(chunk, rec)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / r.filename).write_bytes(r.data)
    _write_json(chunk, rec)
    _state(chunk, True).setdefault("conversions", []).append(rec)
    return dict(_complete(chunk, rec, c, progress))


def _complete(chunk, rec: dict, c, progress) -> dict:
    """Wait for the conversion of ``rec`` (a record of the chunk, changed in place), then :func:`_download`."""
    cid = rec["photocad_id"]
    try:
        done = client_mod.wait(c, cid, progress)
    except client_mod.PhotoCADError as exc:
        if exc.code in ("failed", "timeout"):  # refunded, or deleted on PhotoCAD: nothing to fetch
            _forget(chunk, rec)
            raise
        rec["error"] = str(exc)
        _write_json(chunk, rec)
        raise client_mod.PhotoCADError(f"{exc}\n{_hint(rec)}", exc.status, exc.code) from None
    except BaseException:  # cancelled while PhotoCAD worked: wait() deleted the conversion
        _forget(chunk, rec)
        raise
    rec.update(status="complete", page_mm=[done.get("width_mm"), done.get("height_mm")],
               credits_remaining=client_mod.credits(done))
    rec.pop("error", None)
    _write_json(chunk, rec)
    try:
        _download(chunk, rec, c, progress)
    except FrameMismatch:
        raise
    except Cancelled:  # cancelled after PhotoCAD finished: kept (status complete), fetched later
        raise
    except client_mod.PhotoCADError as exc:
        rec["error"] = str(exc)
        _write_json(chunk, rec)
        raise client_mod.PhotoCADError(f"{exc}\n{_hint(rec)}", exc.status, exc.code) from None
    except Exception as exc:  # noqa: BLE001 - a DXF ezdxf cannot read, a disk error: the conversion stays
        rec["error"] = f"{type(exc).__name__}: {exc}"
        _write_json(chunk, rec)
        raise client_mod.PhotoCADError(t("Il disegno di PhotoCAD non è stato elaborato ({0}).",
                                         "PhotoCAD's drawing could not be processed ({0}).").format(rec["error"])
                                       + "\n" + _hint(rec)) from exc
    return rec


def _download(chunk, rec: dict, c, progress) -> None:
    """Download the drawing of a complete conversion, check its frame, georeference it, write the files."""
    from types import SimpleNamespace

    cid, img = rec["photocad_id"], rec["image"]
    r = SimpleNamespace(x0=img["x0"], y0=img["y0"], pixel=img["pixel"], width=img["width"], height=img["height"],
                        width_m=img["width"] * img["pixel"], height_m=img["height"] * img["pixel"])
    longest = float(max(r.width, r.height) * r.pixel)
    folder = folder_of(chunk, rec)
    progress(0.92, t("Scarico il disegno", "Downloading the drawing"))
    params = {"units": "m", "longest": longest, "margin": 0}
    files = {"image": rec["files"].get("image", "image.jpg")}
    raw = folder / "photocad.dxf"
    raw.write_bytes(c.file(cid, "dxf", **params))
    files["dxf_photocad"] = raw.name
    raw_mm = folder / "photocad_mm.dxf"  # the same drawing at its natural size: for the frame check
    try:
        raw_mm.write_bytes(c.file(cid, "dxf", units="mm", margin=0))
        files["dxf_photocad_mm"] = raw_mm.name
    except client_mod.PhotoCADError:
        raw_mm = None
    rec["dxf_request"] = params
    rec["files"] = files
    check = drawing.check_frame(raw, raw_mm, img, rec.get("page_mm"), longest)
    rec["check"] = check
    if not check["ok"]:
        rec["status"] = "mismatch"
        _write_json(chunk, rec)
        raise FrameMismatch(t("Il disegno di PhotoCAD (conversione {0}) non corrisponde all'immagine inviata: {1}. "
                              "Non è stato messo nel rilievo; i file sono in {2} per capire cosa è successo.",
                              "PhotoCAD's drawing (conversion {0}) does not match the image sent: {1}. It was not "
                              "placed in the survey; the files are in {2} to see what happened.").format(
            cid, "; ".join(check["problems"]), folder), 0, "frame")
    progress(0.95, t("Georeferenzio il disegno", "Georeferencing the drawing"))
    left = drawing.georeference_dxf(raw, folder / "drawing.dxf", r.x0, r.y0)
    files["dxf"] = "drawing.dxf"
    items = drawing.paths(folder / "drawing.dxf", tolerance=max(r.pixel / 4, 1e-4))
    feats = drawing.geometries(items)
    gpkg_path = folder / "drawing.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    drawing.write_gpkg(gpkg_path, feats, rec.get("crs"))
    files["gpkg"] = "drawing.gpkg"
    try:
        (folder / "drawing.svg").write_bytes(c.file(cid, "svg", **params))
        drawing.svg_note(folder / "drawing.svg.json", r, rec.get("crs"), rec.get("frame"))
        files["svg"], files["svg_georef"] = "drawing.svg", "drawing.svg.json"
    except client_mod.PhotoCADError as exc:  # the DXF is what matters: the SVG is a courtesy
        progress(0.96, t("SVG non scaricato: {0}", "SVG not downloaded: {0}").format(exc))
    opts = rec.get("options") or {}
    if opts.get("pdf"):
        (folder / "drawing.pdf").write_bytes(c.file(cid, "pdf"))
        files["pdf"] = "drawing.pdf"
    if opts.get("three_d") and rec.get("frame") is not None:
        entry = source.elevation_entry(chunk, rec["frame"])
        items3d = drawing.world_items(chunk, entry, [it for it in items if len(it["xy"]) >= 2])
        drawing.write_dxf_3d(folder / "drawing_3d.dxf", items3d)
        drawing.write_gpkg_3d(gpkg_path, items3d, chunk.crs)
        files["dxf_3d"] = "drawing_3d.dxf"
    polygons = sum(1 for g, _a in feats if g.geom_type == "Polygon")
    rec.update(status="done", files=files, counts={"polygons": polygons, "lines": len(feats) - polygons,
                                                   "left_out": left})
    rec.pop("error", None)
    _write_json(chunk, rec)
    progress(1.0, summary(rec))


def find(chunk, ident: str) -> dict | None:
    """A conversion by its id or its PhotoCAD id; also one only on disk (``<chunk data>/photocad/*/conversion.json``,
    a project not saved after the upload), which is put back in the project."""
    rec = next((r for r in records(chunk) if ident in (r.get("id"), r.get("photocad_id"))), None)
    if rec is not None:
        return rec  # the record of chunk.meta itself: changed in place
    base = chunk.data_dir / "photocad"
    for path in sorted(base.glob("*/conversion.json")) if base.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if ident in (data.get("id"), data.get("photocad_id")) and data.get("folder") == f"photocad/{path.parent.name}":
            _state(chunk, True).setdefault("conversions", []).append(data)
            return data
    return None


def fetch(chunk, ident: str, key: str, progress=None) -> dict:
    """Download and georeference again a conversion already paid for (by its id or its PhotoCAD id): waits for it
    when still running; no new upload. The record, status ``done``."""
    rec = find(chunk, ident)
    if rec is None:
        raise ValueError(t("Nessuna conversione «{0}» in questo chunk.", "No conversion '{0}' in this chunk.")
                         .format(ident))
    return dict(_complete(chunk, rec, _client(key), progress or (lambda *a: None)))


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
