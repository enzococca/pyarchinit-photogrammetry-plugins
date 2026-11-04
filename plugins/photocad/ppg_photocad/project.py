"""A conversion of the project: the images rendered (one per tile), the confirmation, the uploads, the waits, the
drawings brought back, placed by registration, merged and kept. No Qt here.

What is made lives in ``<chunk data>/photocad/<id>/``: one folder per tile, ``t0/``, ``t1/``... with ``image.jpg``
(what was sent), ``photocad_mm.dxf`` (PhotoCAD's drawing at its natural size in mm), ``geometry.json``, ``drawing.svg``
(+ ``drawing.svg.json``: where it lies), ``drawing.dxf`` (the tile placed), ``drawing.pdf`` when asked; and for the
whole: ``drawing.dxf`` (georeferenced, the tiles merged), ``drawing.gpkg`` (table ``photocad``; ``photocad_3d`` for an
elevation in 3D), ``drawing_3d.dxf``, ``conversion.json`` (the source, the tiles with their PhotoCAD ids, idempotency
keys, images and registrations, the date, the credits used - never the key). The list of the conversions is in
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


STATUS = {"uploading": ("non ancora inviata del tutto", "not all sent yet"),
          "unknown": ("esito di un invio sconosciuto: controlla su photocad.ai", "outcome of an upload unknown: check "
                      "on photocad.ai"),
          "pending": ("inviata, non ancora finita", "sent, not finished yet"),
          "complete": ("finita, non scaricata", "finished, not downloaded"),
          "mismatch": ("disegno non corrispondente all'immagine", "drawing not matching the image")}


def _tiles(rec: dict) -> list[dict]:
    """The tiles of a record; a record made before the tiles (0.1.0 in development: one PhotoCAD id, its files in the
    folder itself) becomes one tile, in place."""
    if "tiles" not in rec:
        status = {"done": "complete", "mismatch": "complete", "complete": "complete"}.get(rec.get("status"), "pending")
        rec["tiles"] = [{"n": 0, "folder": "", "photocad_id": rec.get("photocad_id"), "idempotency_key": None,
                         "status": status if rec.get("photocad_id") else "new", "image": dict(rec.get("image") or {}),
                         "page_mm": rec.get("page_mm"), "files": {"image": (rec.get("files") or {}).get("image",
                                                                                                       "image.jpg")}}]
    return rec["tiles"]


def _pixel(rec: dict) -> float:
    tiles = rec.get("tiles") or [{"image": rec.get("image") or {}}]
    return float((tiles[0].get("image") or {}).get("pixel", 0))


def summary(rec: dict) -> str:
    c = rec.get("counts") or {}
    ids = ", ".join(str(tl.get("photocad_id")) for tl in rec.get("tiles") or [] if tl.get("photocad_id")) \
        or str(rec.get("photocad_id"))
    if rec.get("status", "done") in STATUS:
        return f"{rec.get('name')}: {t(*STATUS[rec['status']])} (PhotoCAD {ids})"
    where = t("pianta", "plan") if rec.get("frame") is None else t("prospetto {0}", "elevation {0}").format(rec["frame"])
    n = len(rec.get("tiles") or [1])
    tiles = "" if n == 1 else t(", {0} tessere", ", {0} tiles").format(n)
    stones = "" if c.get("stones") is None else t(", {0} pietre", ", {0} stones").format(c["stones"])
    return t("{0}: {1} poligoni, {2} linee{6} ({3}, {4} cm/px{5})",
             "{0}: {1} polygons, {2} lines{6} ({3}, {4} cm/px{5})").format(
        rec.get("name"), c.get("polygons", 0), c.get("lines", 0), where, f"{_pixel(rec) * 100:.2f}", tiles, stones)


# ------------------------------------------------------------------ the conversion
def _client(key):
    return client_mod.Client(key)


class NotEnough(client_mod.PhotoCADError):
    """Fewer credits and free conversions than the tiles need."""


def prepare(chunk, src: str, key: str, bbox=None, fmt: str = "jpeg", max_tiles: int = source.MAX_TILES) -> dict:
    """The tiles of ``src`` (and ``bbox``) rendered, and the account (the key is checked here, before anything is
    sent): ``{"tiles" [Rendered], "credits", "free_conversions", "account_text", "source", "bbox"}``. ``ValueError``
    for more tiles than ``max_tiles`` or a strip too thin; :class:`NotEnough` when the account cannot pay them."""
    plan = source.tiles(chunk, src, bbox)
    if len(plan) > max_tiles:
        raise ValueError(t("Servirebbero {0} tessere ({0} conversioni), più del limite di {1}: scegli una parte "
                           "(un'area, --bbox) o alza il limite (--max-tiles).",
                           "{0} tiles ({0} conversions) would be needed, more than the limit of {1}: choose a part "
                           "(an area, --bbox) or raise the limit (--max-tiles).").format(len(plan), max_tiles))
    rendered = [source.render(chunk, src, p["window"], p["zoom"], fmt=fmt) for p in plan]
    account = _client(key).account()
    credits, free = client_mod.credits(account), client_mod.free_conversions(account)
    text = client_mod.account_text(account)
    if credits is not None or free is not None:
        available = (credits or 0) + (free or 0)
        if available < len(rendered):
            raise NotEnough(t("Servono {0} conversioni, ne restano {1} ({2}).",
                              "{0} conversions are needed, {1} are left ({2}).").format(len(rendered), available, text),
                            402, "not_enough")
    return {"tiles": rendered, "rendered": rendered[0], "credits": credits, "free_conversions": free,
            "account_text": text, "source": src, "bbox": None if bbox is None else list(bbox)}


def confirm_info(prep: dict) -> dict:
    """What the user is told before the upload: the text and its numbers."""
    tiles = prep["tiles"]
    r = tiles[0]
    n = len(tiles)
    size = sum(len(x.data) for x in tiles)
    if n == 1:
        what = t("L'immagine ({0} x {1} px, {2} kB, {3} cm/px) sarà inviata a photocad.ai, un servizio esterno, per "
                 "vettorializzarla. Serve 1 conversione: un credito (circa 2 $) o una conversione gratuita.",
                 "The image ({0} x {1} px, {2} kB, {3} cm/px) will be sent to photocad.ai, an external service, to "
                 "vectorise it. 1 conversion is needed: a credit (about $2) or a free conversion.")
    else:
        what = t("L'immagine va in {4} tessere (fino a {0} x {1} px, {2} kB in tutto, {3} cm/px) inviate a photocad.ai, "
                 "un servizio esterno, per vettorializzarle. Servono {4} conversioni: un credito (circa 2 $) o una "
                 "conversione gratuita ciascuna.",
                 "The image goes in {4} tiles (up to {0} x {1} px, {2} kB in all, {3} cm/px) sent to photocad.ai, an "
                 "external service, to vectorise them. {4} conversions are needed: a credit (about $2) or a free "
                 "conversion each.")
    text = what.format(max(x.width for x in tiles), max(x.height for x in tiles), max(1, round(size / 1024)),
                       f"{r.pixel * 100:.2f}", n) + " " + (prep.get("account_text") or "") + "."
    return {"text": text, "credits": prep.get("credits"), "free_conversions": prep.get("free_conversions"),
            "tiles": n, "width": r.width, "height": r.height, "bytes": size, "pixel": r.pixel, "source": prep["source"]}


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
    """PhotoCAD's drawing does not lie on the image: nothing placed, the files kept for a diagnosis."""


def _hint(rec: dict) -> str:
    ids = ", ".join(str(tl.get("photocad_id")) for tl in rec.get("tiles") or [] if tl.get("photocad_id"))
    return t("La conversione {0} (PhotoCAD {1}) è conservata nel progetto: recuperala con Plugin > PhotoCAD > Recupera "
             "una conversione, o `ppg plugin photocad fetch PROGETTO --id {0}` (una tessera già inviata si rimanda con la sua "
             "chiave di idempotenza: PhotoCAD non la addebita di nuovo).",
             "The conversion {0} (PhotoCAD {1}) is kept in the project: get it with Plugin > PhotoCAD > Fetch a "
             "conversion, or `ppg plugin photocad fetch PROJECT --id {0}` (a tile already sent goes again with its "
             "idempotency key: PhotoCAD does not charge it again).").format(rec["id"], ids or "-")


def _write_json(chunk, rec: dict) -> None:
    folder = folder_of(chunk, rec)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "conversion.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")


def _forget(chunk, rec: dict) -> None:
    """A conversion that will never give a drawing (failed and refunded, deleted on PhotoCAD) out of the project."""
    remove(chunk, rec["id"])


def _tile_dir(chunk, rec: dict, tile: dict) -> Path:
    d = folder_of(chunk, rec) / (tile.get("folder") or "")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _status(rec: dict) -> str:
    states = {tl.get("status") for tl in rec["tiles"]}
    if "unknown" in states:
        return "unknown"
    for s in ("new", "pending"):
        if s in states:
            return "uploading" if s == "new" else "pending"
    if "complete" in states:
        return "complete"
    if "mismatch" in states:
        return "mismatch"
    return "done"


def upload(chunk, prep: dict, key: str, detailed: bool = True, remove_background: bool = False,
           three_d: bool = False, pdf: bool = False, progress=None, name: str | None = None) -> dict:
    """Send the prepared tiles, wait for PhotoCAD, bring the drawings back, place and merge them, keep them: the
    record (also in ``chunk.meta["photocad"]["conversions"]``).

    Each tile has an ``Idempotency-Key`` written in ``conversion.json`` BEFORE its POST, so an upload asked again (a
    429, a lost connection, :func:`fetch` after a crash) never charges twice; its PhotoCAD id is written as soon as the
    202 comes. A failure afterwards keeps the record - ``uploading``, ``pending``, ``complete`` or ``mismatch`` - and
    :func:`fetch` goes on from there. ``progress(fraction, message)`` raises ``Cancelled`` when the user cancels:
    before a conversion is complete it is deleted on PhotoCAD (the record forgotten when nothing is left), after it
    it is kept."""
    import uuid

    progress = progress or (lambda *a: None)
    rid = _new_id(chunk)
    tiles = []
    for i, r in enumerate(prep["tiles"]):
        tiles.append({"n": i, "folder": f"t{i}", "idempotency_key": uuid.uuid4().hex, "status": "new",
                      "photocad_id": None, "image": r.describe(), "files": {"image": r.filename}})
    rec = {"id": rid, "name": name or _name(chunk, prep["source"]), "status": "uploading", "source": prep["source"],
           "frame": prep["tiles"][0].frame, "crs": prep["tiles"][0].crs, "bbox": prep.get("bbox"),
           "folder": f"photocad/{rid}", "tiles": tiles, "files": {},
           "options": {"detailed": bool(detailed), "remove_background": bool(remove_background),
                       "three_d": bool(three_d and prep["tiles"][0].frame is not None), "pdf": bool(pdf)},
           "credits_used": len(tiles), "account_before": {"credits": prep.get("credits"),
                                                          "free_conversions": prep.get("free_conversions")},
           "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "visible": True}
    for tile, r in zip(tiles, prep["tiles"]):
        (_tile_dir(chunk, rec, tile) / r.filename).write_bytes(r.data)
    _write_json(chunk, rec)
    _state(chunk, True).setdefault("conversions", []).append(rec)
    return dict(_complete(chunk, rec, _client(key), progress))


MAX_POSTS = 2  # POSTs of unknown outcome per tile: the first and one resend with the same Idempotency-Key


def _unknown_text(rec: dict, tiles: list) -> str:
    return t("Non so se PhotoCAD ha ricevuto {0} tessera/e della conversione {1} (la connessione si è interrotta due "
             "volte): controlla su photocad.ai (Conversions) prima di riprovare. Il plugin non le rimanda da solo: "
             "una conversione rimandata senza l'idempotenza di PhotoCAD sarebbe addebitata due volte.",
             "I do not know whether PhotoCAD received {0} tile(s) of the conversion {1} (the connection broke twice): "
             "check on photocad.ai (Conversions) before trying again. The plugin does not send them again by itself: "
             "a conversion sent again without PhotoCAD's idempotency would be charged twice.").format(
        len(tiles), rec["id"])


def pending_uploads(rec: dict) -> int:
    """How many tiles of a record are still to be sent (a fetch would send them: one conversion each)."""
    return sum(1 for tl in (rec.get("tiles") or []) if tl.get("status") == "new")


def _fail(chunk, rec, exc, status=None, code=None):
    rec["status"] = _status(rec)
    rec["error"] = str(exc)
    _write_json(chunk, rec)
    return client_mod.PhotoCADError(f"{exc}\n{_hint(rec)}", status if status is not None else getattr(exc, "status", 0),
                                    code if code is not None else getattr(exc, "code", ""))


def _complete(chunk, rec: dict, c, progress, search_px: float | None = None) -> dict:
    """Upload, wait, download and place the tiles of ``rec`` still to do (a record of the chunk, changed in place)."""
    tiles = _tiles(rec)
    opts = rec.get("options") or {}
    n = len(tiles)
    # 1. the uploads (the key on disk already). A tile is POSTed at most MAX_POSTS times whose outcome is not known
    # (a 429 is known: not processed); after that it is "unknown" and never sent again by the plugin
    for tile in tiles:
        if tile.get("status") != "new":
            continue
        img = tile["files"].get("image", "image.jpg")
        data = (_tile_dir(chunk, rec, tile) / img).read_bytes()
        progress(0.02 + 0.06 * tile["n"] / n, t("Invio a PhotoCAD ({0}/{1})", "Sending to PhotoCAD ({0}/{1})")
                 .format(tile["n"] + 1, n))
        ctype = "image/png" if img.endswith(".png") else "image/jpeg"

        def before_post(event, tile=tile):
            if event == "rate_limited":
                tile["posts"] = max(0, int(tile.get("posts", 0)) - 1)
                _write_json(chunk, rec)
                return None
            if int(tile.get("posts", 0)) >= MAX_POSTS:
                return False
            tile["posts"] = int(tile.get("posts", 0)) + 1
            _write_json(chunk, rec)  # on disk before the POST leaves
            return True

        try:
            cid = c.convert(data, img, ctype, detailed=opts.get("detailed", True),
                            remove_background=opts.get("remove_background", False),
                            idempotency_key=tile.get("idempotency_key"), before_post=before_post)
        except client_mod.PhotoCADError as exc:
            if exc.code == "unknown" or (exc.code == "network" and int(tile.get("posts", 0)) >= MAX_POSTS):
                tile["status"] = "unknown"
                tile["error"] = str(exc)
                rec["status"] = _status(rec)
                _write_json(chunk, rec)
                raise client_mod.PhotoCADError(_unknown_text(rec, [tile]), 0, "unknown") from None
            raise _fail(chunk, rec, exc) from None
        tile.update(photocad_id=cid, status="pending")
        rec["status"] = _status(rec)
        _write_json(chunk, rec)
    unknown = [tl for tl in tiles if tl.get("status") == "unknown"]
    if unknown:
        rec["status"] = _status(rec)
        _write_json(chunk, rec)
        raise client_mod.PhotoCADError(_unknown_text(rec, unknown), 0, "unknown")
    # 2. the waits
    for tile in tiles:
        if tile.get("status") != "pending":
            continue
        try:
            done = client_mod.wait(c, tile["photocad_id"], progress)
        except client_mod.PhotoCADError as exc:
            if exc.code in ("failed", "timeout"):  # refunded, or deleted on PhotoCAD: nothing to fetch from it
                tile.update(status="failed", error=str(exc))
                _write_json(chunk, rec)
                continue
            raise _fail(chunk, rec, exc) from None
        except BaseException as exc:  # cancelled while PhotoCAD worked: wait() deleted this one, the others go too
            tile["status"] = "pending" if getattr(exc, "photocad_not_deleted", None) else "deleted"
            for other in tiles:
                if other is not tile and other.get("status") == "pending" and other.get("photocad_id"):
                    try:
                        c.delete(other["photocad_id"])
                        other["status"] = "deleted"
                    except client_mod.PhotoCADError:  # it may still be made and charged: kept, to check or fetch
                        pass
            if not any(tl.get("status") in ("complete", "placed", "mismatch", "pending", "unknown") for tl in tiles):
                _forget(chunk, rec)
            else:
                rec["status"] = _status(rec)
                _write_json(chunk, rec)
            raise
        tile.update(status="complete", page_mm=[done.get("width_mm"), done.get("height_mm")],
                    title=done.get("title"), credits_remaining=client_mod.credits(done))
        tile.pop("error", None)
        rec["credits_remaining"] = client_mod.credits(done)
        rec["status"] = _status(rec)
        _write_json(chunk, rec)
    if all(tl.get("status") in ("failed", "deleted") for tl in tiles):
        errors = "; ".join(str(tl.get("error") or "") for tl in tiles if tl.get("error"))
        _forget(chunk, rec)
        raise client_mod.PhotoCADError(errors or t("PhotoCAD non ha dato nessun disegno.", "PhotoCAD gave no drawing."),
                                       0, "failed")
    # 3. the downloads and the registrations
    for tile in tiles:
        if tile.get("status") not in ("complete", "mismatch"):
            continue
        try:
            _download_tile(chunk, rec, tile, c, progress, search_px)
        except Cancelled:
            rec["status"] = _status(rec)
            _write_json(chunk, rec)
            raise
        except client_mod.PhotoCADError as exc:
            raise _fail(chunk, rec, exc) from None
        except Exception as exc:  # noqa: BLE001 - a DXF ezdxf cannot read, a disk error: the conversion stays
            raise _fail(chunk, rec, t("Il disegno di PhotoCAD non è stato elaborato ({0}).",
                                      "PhotoCAD's drawing could not be processed ({0}).").format(
                f"{type(exc).__name__}: {exc}")) from exc
        _write_json(chunk, rec)
    bad = [tl for tl in tiles if tl.get("status") == "mismatch"]
    if bad:
        rec["status"] = "mismatch"
        _write_json(chunk, rec)
        problems = "; ".join(f"PhotoCAD {tl['photocad_id']}: " + ", ".join(tl["registration"]["problems"]) for tl in bad)
        raise FrameMismatch(t("Il disegno di PhotoCAD non combacia con l'immagine inviata ({0}). Non è stato messo nel "
                              "rilievo; i file sono in {1}.", "PhotoCAD's drawing does not match the image sent ({0}). "
                              "It was not placed in the survey; the files are in {1}.").format(
            problems, folder_of(chunk, rec)), 0, "frame")
    _merge(chunk, rec, progress)
    return rec


def _download_tile(chunk, rec: dict, tile: dict, c, progress, search_px: float | None = None) -> None:
    """The drawing of a complete tile at its natural size (mm, no ``longest``: PhotoCAD writes mm whatever ``units``
    says), its registration on the tile's image, the tile placed (``placed``) or refused (``mismatch``)."""
    from . import register

    cid = tile["photocad_id"]
    d = _tile_dir(chunk, rec, tile)
    progress(0.92, t("Scarico il disegno ({0})", "Downloading the drawing ({0})").format(cid))
    raw = d / "photocad_mm.dxf"
    raw.write_bytes(c.file(cid, "dxf", units="mm", margin=0))
    files = dict(tile.get("files") or {})
    files["dxf_photocad_mm"] = raw.name
    try:
        (d / "geometry.json").write_bytes(c.file(cid, "geometry", units="mm"))
        files["geometry"] = "geometry.json"
    except client_mod.PhotoCADError:
        pass
    if (rec.get("options") or {}).get("pdf"):
        (d / "drawing.pdf").write_bytes(c.file(cid, "pdf"))
        files["pdf"] = "drawing.pdf"
    items, factor = register.read_mm(raw)
    image = (d / files.get("image", "image.jpg")).read_bytes()
    reg = register.register(items, tile.get("page_mm"), image, search_px)
    tile["registration"] = reg
    tile["files"] = files
    if not reg["ok"]:
        tile["status"] = "mismatch"
        return
    m, tx, ty = register.similarity(reg, tile["image"])
    tile["similarity"] = [m, tx, ty]
    drawing.georeference_dxf(raw, d / "drawing.dxf", tx, ty, scale=m * factor)
    files["dxf"] = "drawing.dxf"
    try:
        (d / "drawing.svg").write_bytes(c.file(cid, "svg", units="mm", margin=0))
        drawing.svg_note(d / "drawing.svg.json", reg["page_mm"], (m, tx, ty), rec.get("crs"), rec.get("frame"))
        files["svg"], files["svg_georef"] = "drawing.svg", "drawing.svg.json"
    except client_mod.PhotoCADError as exc:  # the DXF is what matters: the SVG is a courtesy
        progress(0.96, t("SVG non scaricato: {0}", "SVG not downloaded: {0}").format(exc))
    tile["status"] = "placed"


def _centre(tile: dict) -> tuple[float, float]:
    im = tile["image"]
    return im["x0"] + im["width"] * im["pixel"] / 2, im["y0"] + im["height"] * im["pixel"] / 2


def _merge(chunk, rec: dict, progress) -> None:
    """The placed tiles into one drawing: from each tile what is nearer its own centre than any other tile's (the
    copies in the overlaps once), DXF, GeoPackage, 3D."""
    import numpy as np

    folder = folder_of(chunk, rec)
    placed = [tl for tl in rec["tiles"] if tl.get("status") == "placed"]
    progress(0.97, t("Unisco e georeferenzio il disegno", "Merging and georeferencing the drawing"))
    centres = np.array([_centre(tl) for tl in placed])
    items, left = [], 0
    for i, tl in enumerate(placed):
        d = _tile_dir(chunk, rec, tl)
        pixel = float(tl["image"]["pixel"])
        for it in drawing.paths(d / "drawing.dxf", tolerance=max(pixel / 4, 1e-4)):
            if len(placed) > 1:
                c = it["xy"].mean(axis=0)
                if int(np.argmin(np.hypot(*(centres - c).T))) != i:
                    left += 1
                    continue
            items.append(it)
    files = {}
    if len(placed) == 1:
        one = _tile_dir(chunk, rec, placed[0]) / "drawing.dxf"
        if one.resolve() != (folder / "drawing.dxf").resolve():  # a record made before the tiles: the same file
            shutil.copyfile(one, folder / "drawing.dxf")
        for what in ("svg", "svg_georef", "pdf"):
            if what in placed[0]["files"]:
                files[what] = f"{placed[0].get('folder') or '.'}/{placed[0]['files'][what]}"
    else:
        drawing.write_dxf(folder / "drawing.dxf", items)
    files["dxf"] = "drawing.dxf"
    feats = drawing.geometries(items)
    gpkg_path = folder / "drawing.gpkg"
    if gpkg_path.exists():
        gpkg_path.unlink()
    drawing.write_gpkg(gpkg_path, feats, rec.get("crs"))
    files["gpkg"] = "drawing.gpkg"
    if (rec.get("options") or {}).get("three_d") and rec.get("frame") is not None:
        entry = source.elevation_entry(chunk, rec["frame"])
        items3d = drawing.world_items(chunk, entry, [it for it in items if len(it["xy"]) >= 2])
        drawing.write_dxf_3d(folder / "drawing_3d.dxf", items3d)
        drawing.write_gpkg_3d(gpkg_path, items3d, chunk.crs)
        files["dxf_3d"] = "drawing_3d.dxf"
    polygons = sum(1 for g, _a in feats if g.geom_type == "Polygon")
    rec.update(status="done", files=files, counts={
        "polygons": polygons, "lines": len(feats) - polygons, "overlap_copies_removed": left,
        "tiles_placed": len(placed), "tiles_failed": sum(1 for tl in rec["tiles"] if tl.get("status") == "failed"),
        "residual_px": [round(float(tl["registration"]["residual_px"]), 3) for tl in placed]})
    rec.pop("error", None)
    try:
        progress(0.99, t("Ricavo le pietre dalle linee", "Making the stones from the lines"))
        polygonize(chunk, rec)
    except Exception as exc:  # noqa: BLE001 - the drawing is there: the stones can be made again (polygonize)
        rec["polygonize"] = {"error": f"{type(exc).__name__}: {exc}"}
    _write_json(chunk, rec)
    progress(1.0, summary(rec))


# ------------------------------------------------------------------ stones from the lines
SNAP_PX = 2.0  # ends of lines this close (in pixels of the image) are joined
MIN_AREA_PX = 50.0  # smaller faces are left out (pixels of the image)
BACKGROUND_SHARE = 0.5  # faces with more of their pixels white (no data) are background
SLIVER = 0.02  # area / perimeter^2 under this: a thin face (a joint between two stones), not a stone (square 0.0625)


def is_sliver(face) -> bool:
    """A thin face: its area over its perimeter squared under :data:`SLIVER` (a rectangle about 8 times longer than
    wide or thinner)."""
    p = float(face.length)
    return p > 0 and float(face.area) / (p * p) < SLIVER
STONES_TABLE = "polygons"


def _tile_of(rec: dict, point) -> dict:
    placed = [tl for tl in rec["tiles"] if tl.get("status") == "placed"]
    return min(placed, key=lambda tl: (_centre(tl)[0] - point[0]) ** 2 + (_centre(tl)[1] - point[1]) ** 2)


def _background_share(grey, image: dict, face) -> float:
    """The share of white (no data) pixels of the tile image under ``face``."""
    import numpy as np
    from rasterio.features import geometry_mask
    from rasterio.transform import Affine

    p, h, w = float(image["pixel"]), grey.shape[0], grey.shape[1]
    top = image["y0"] + h * p
    minx, miny, maxx, maxy = face.bounds
    c0, c1 = max(0, int((minx - image["x0"]) / p)), min(w, int(np.ceil((maxx - image["x0"]) / p)))
    r0, r1 = max(0, int((top - maxy) / p)), min(h, int(np.ceil((top - miny) / p)))
    if c1 <= c0 or r1 <= r0:
        return 1.0
    tr = Affine(p, 0, image["x0"] + c0 * p, 0, -p, top - r0 * p)
    inside = geometry_mask([face], out_shape=(r1 - r0, c1 - c0), transform=tr, invert=True, all_touched=False)
    if not inside.any():
        return 0.0
    return float((grey[r0:r1, c0:c1][inside] >= 250).mean())


def polygonize(chunk, rec: dict, snap_px: float = SNAP_PX, min_area_px: float = MIN_AREA_PX) -> int:
    """The stones of a conversion: its lines (and the outlines of its polygons) noded - each end within ``snap_px``
    pixels of the image of another line, or of its own other end, joined to it - and polygonized; faces smaller than
    ``min_area_px`` pixels and faces mostly on white (no data) left out. Written in ``polygons.gpkg`` (table
    ``polygons``); the record says how many and what was left out. Returns the number of stones."""
    import cv2
    import shapely
    from shapely.geometry import LineString
    from shapely.ops import nearest_points

    feats = features(chunk, rec)
    placed = [tl for tl in rec["tiles"] if tl.get("status") == "placed"]
    pixel = min(float(tl["image"]["pixel"]) for tl in placed)
    tol, min_area = snap_px * pixel, min_area_px * pixel * pixel
    lines = []
    for g, _a in feats:
        if g.geom_type == "Polygon":
            lines.append(LineString(g.exterior.coords))
        elif g.geom_type == "LineString" and g.length > 0:
            lines.append(g)
    # on a grid of a fiftieth of a pixel: the ends PhotoCAD made equal come back from the placement 1e-16 apart,
    # and unary_union would not node them (Khutm: 3 faces out of 1993 chained lines without this)
    grid = pixel / 50
    lines = [g for g in shapely.set_precision(lines, grid) if not g.is_empty and g.length > 0]
    tree = shapely.STRtree(lines)
    connectors = []
    for i, line in enumerate(lines):
        coords = list(line.coords)
        if line.is_closed:
            continue
        for end, other in ((coords[0], coords[-1]), (coords[-1], coords[0])):
            p = shapely.Point(end)
            best = None
            if grid < p.distance(shapely.Point(other)) <= tol:
                best = (p.distance(shapely.Point(other)), other)
            for j in tree.query(p.buffer(tol)):
                if j == i:
                    continue
                d = lines[j].distance(p)
                if d <= tol and (best is None or d < best[0]):
                    q = nearest_points(lines[j], p)[0]
                    best = (d, (q.x, q.y))
            if best is not None and best[0] > grid:
                connectors.append(LineString([end, best[1]]))
    noded = shapely.unary_union(lines + connectors, grid_size=grid)
    faces = list(shapely.get_parts(shapely.polygonize(list(shapely.get_parts(noded)))))
    greys: dict = {}
    kept, small, background, slivers = [], 0, 0, 0
    for face in faces:
        if face.area < min_area:
            small += 1
            continue
        if is_sliver(face):
            slivers += 1
            continue
        c = face.representative_point()
        tile = _tile_of(rec, (c.x, c.y))
        if tile["n"] not in greys:
            path = _tile_dir(chunk, rec, tile) / tile["files"].get("image", "image.jpg")
            greys[tile["n"]] = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if _background_share(greys[tile["n"]], tile["image"], face) > BACKGROUND_SHARE:
            background += 1
            continue
        kept.append(face)
    from pyarchinit_photogrammetry.io import gpkg

    path = folder_of(chunk, rec) / "polygons.gpkg"
    if path.exists():
        path.unlink()
    gpkg.write_table(path, STONES_TABLE, "POLYGON", {"n": "INTEGER", "area_m2": "REAL", "perimeter_m": "REAL",
                                                      "source": "TEXT"},
                     [(f, {"n": i + 1, "area_m2": float(f.area), "perimeter_m": float(f.exterior.length),
                           "source": "polygonize"}) for i, f in enumerate(kept)], rec.get("crs"), identifier="pietre")
    rec.setdefault("files", {})["polygons"] = "polygons.gpkg"
    rec.setdefault("counts", {})["stones"] = len(kept)
    rec["polygonize"] = {"snap_m": tol, "snap_px": snap_px, "min_area_m2": min_area, "min_area_px": min_area_px,
                         "lines": len(lines), "connectors": len(connectors), "faces": len(faces), "small": small,
                         "slivers": slivers, "background": background, "stones": len(kept)}
    _write_json(chunk, rec)
    return len(kept)


def stone_features(chunk, rec: dict):
    """``(polygon, attributes)`` of the stones of a conversion (:func:`polygonize`), in its frame."""
    from pyarchinit_photogrammetry.io import gpkg

    path = path_of(chunk, rec, "polygons")
    if rec.get("status", "done") != "done" or path is None or not path.exists():
        return []
    return [(g, a) for g, a in gpkg.read_table(path, STONES_TABLE) if g is not None]


def unit_features(chunk, rec: dict, source: str | None = None):
    """What becomes units: ``polygons`` (the stones), ``lines`` (the drawing as PhotoCAD gave it), None: the stones
    when there are some, else the drawing."""
    if source == "lines":
        return features(chunk, rec)
    stones = stone_features(chunk, rec)
    if source == "polygons" or stones:
        return stones
    return features(chunk, rec)


def find(chunk, ident: str) -> dict | None:
    """A conversion by its id, the PhotoCAD id of one of its tiles (or of a record made before the tiles); also one
    only on disk (``<chunk data>/photocad/*/conversion.json``, a project not saved after the upload), which is put
    back in the project."""
    def ids(r):
        return {r.get("id"), r.get("photocad_id")} | {tl.get("photocad_id") for tl in r.get("tiles") or []}

    rec = next((r for r in records(chunk) if ident in ids(r)), None)
    if rec is not None:
        return rec  # the record of chunk.meta itself: changed in place
    base = chunk.data_dir / "photocad"
    for path in sorted(base.glob("*/conversion.json")) if base.is_dir() else []:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if ident in ids(data) and data.get("folder") == f"photocad/{path.parent.name}":
            _state(chunk, True).setdefault("conversions", []).append(data)
            return data
    return None


def fetch(chunk, ident: str, key: str, progress=None, search_px: float | None = None) -> dict:
    """Go on with a conversion already made (by its id or a PhotoCAD id): tiles not sent are sent with their
    idempotency key (never charged twice), running ones waited for, finished ones downloaded and placed again.
    The record, status ``done``."""
    rec = find(chunk, ident)
    if rec is None:
        raise ValueError(t("Nessuna conversione «{0}» in questo chunk.", "No conversion '{0}' in this chunk.")
                         .format(ident))
    _tiles(rec)
    return dict(_complete(chunk, rec, _client(key), progress or (lambda *a: None), search_px))


def convert(chunk, src: str, key: str, bbox=None, detailed: bool = True, remove_background: bool = False,
            three_d: bool = False, pdf: bool = False, confirm=None, progress=None, name: str | None = None,
            max_tiles: int = source.MAX_TILES) -> dict:
    """:func:`prepare`, ``confirm(info)`` (:func:`confirm_info`; False: :class:`Declined`, nothing sent; None:
    no question), :func:`upload`."""
    prep = prepare(chunk, src, key, bbox, max_tiles=max_tiles)
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
