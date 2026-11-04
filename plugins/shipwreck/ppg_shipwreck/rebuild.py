"""The whole reconstruction in one build, and the table of what is measured and what is reconstructed.

:func:`build` runs, in order, the mirror (:func:`reconstruct.mirror`), the extension of the strakes
(:func:`reconstruct.extend`), the frames at the measured spacing (:func:`reconstruct.frames_by_spacing`), the faired
hull (:func:`reconstruct.loft`) and the solids of the extended strakes and of the spaced frames, on the hull's
surface where the wreck shows nothing (:func:`solids.surface_for`):

- an extended strake is a shell (:func:`solids.strake_shell`) of its source strake's thickness; off the model its
  inner face is read first across the band between its two edges' heights (linear between the lower and the upper
  edge at the same x, where both have one), then on the mirrored solids, then on the faired hull;
- a spaced frame is a prism standing on the surface under its outline, as high above it as its two neighbours are
  (their volume over their outline's area; else the median of the measured frames').

A piece that cannot be built is refused with the reason and listed; the build goes on. The reconstruction is
replaced as a whole (``state["reconstruction"]``: ``params``, ``built``, ``timbers``, ``mirror``, ``extension``,
``spacing``, ``loft``, ``table``, ``open_ends``; the user's ``ends`` kept), ids kept for the same source and
position, the files no record holds removed. Whatever stops the build - a step that is not ready, a cancel - leaves
the earlier reconstruction as it was, records and files. The measured catalogue is never touched.

With other chunks of the same wreck in the project (:func:`chunks.members`: aligned to this one, or sharing its ship
frame through markers of the same names, with a current relief) the build reads them all: the merged height raster
(:func:`chunks.prepare`, in this chunk's folder) and the merged records (:func:`chunks.merged`); the members are
read only. ``state["reconstruction"]["members"]`` lists them (and the chunks refused, with the reason).

:func:`table` lists every record of the catalogue and of the reconstruction with its length and area, measured or
reconstructed; :func:`write_table` writes it as CSV and Excel."""
from __future__ import annotations

import copy
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import ship, solids, timbers, tracing
from .ship import NotReady
from .texts import t

STEP = 0.10
SAMPLES = 41
STATUSES = ("reconstructed:mirror", "reconstructed:extension", "reconstructed:spacing")
STEPS = ("mirror", "extension", "frames", "loft")  # the steps of a build, in their order
TABLE_COLUMNS = ("id", "name", "kind", "side", "status", "measured_length", "measured_area", "reconstructed_length",
                 "reconstructed_area", "length_kind")
LENGTH_KINDS = {"3D": ("3D sul modello", "3D on the model"), "pianta": ("in pianta", "in plan")}
"""How a row's length is measured (``length_kind``): along the record over the model, or in the ship plan."""
TABLE_FILE = "tabella"


def _label(r: dict) -> str:
    return r.get("name") or r["id"]


def _reason(exc: Exception) -> str:
    return str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"


# ----------------------------------------------------------------------------- the solids of the new parts
def _edge_array(edge) -> np.ndarray:
    e = np.asarray([[p[0], abs(p[1]), np.nan if p[2] is None else p[2]] for p in edge or []], float).reshape(-1, 3)
    return e[np.argsort(e[:, 0], kind="stable")] if len(e) else e


def _along(e: np.ndarray, x: np.ndarray, column: int) -> np.ndarray:
    """An edge's ``column`` (1: |y|, 2: z) at ``x``, NaN outside the x range of its points with a value."""
    ok = np.isfinite(e[:, column]) if len(e) else np.zeros(0, bool)
    out = np.full(len(x), np.nan)
    if ok.sum() < 2:
        return out
    ex, ev = e[ok, 0], e[ok, column]
    inside = (x >= ex.min()) & (x <= ex.max())
    out[inside] = np.interp(x[inside], ex, ev)
    return out


def _band_surface(record: dict, base):
    """``base`` (:func:`solids.surface_for`) with the band's own edges first: z linear across the band between the
    lower and the upper edge's heights at the same x, where both have one."""
    lower, upper = _edge_array(record["edges"]["lower"]), _edge_array(record["edges"]["upper"])

    def surface(x, y):
        x, y = np.asarray(x, float).ravel(), np.asarray(y, float).ravel()
        ay = np.abs(y)
        yl, yu = _along(lower, x, 1), _along(upper, x, 1)
        zl, zu = _along(lower, x, 2), _along(upper, x, 2)
        run = yu - yl
        f = np.where(np.abs(run) > 1e-6, (ay - yl) / np.where(np.abs(run) > 1e-6, run, 1.0), 0.5)
        z = zl + np.clip(f, 0.0, 1.0) * (zu - zl)
        rest = ~np.isfinite(z)
        if rest.any():
            z[rest] = base(x[rest], y[rest])
        return z

    return surface


def _extension_solid(chunk, record: dict, by_id: dict, cache: dict, records=None) -> solids.Solid:
    source = by_id.get(record["source"])
    if source is None:
        raise NotReady(t(f"{_label(record)}: la fascia da cui viene non c'è più.",
                         f"{_label(record)}: the strake it comes from is gone."))
    mm, _how = solids.thickness_of(chunk, source)
    record["thickness_mm"] = float(mm)
    poly = solids._polygon(record["ship_xy"])
    surface = _band_surface(record, solids.surface_for(chunk, poly, cache))
    return solids.strake_shell(chunk, record, mm, surface=surface, path=_height(chunk), records=records)


def _height(chunk):
    from . import reconstruct

    return reconstruct.height_path(chunk)


def _frame_height(r: dict) -> float | None:
    """A frame's mean height above the hull: its volume over its outline's area."""
    if not r.get("volume") or not r.get("ship_xy"):
        return None
    area = solids._polygon(r["ship_xy"]).area
    return float(r["volume"]) / area if area > 0 else None


def _spacing_solid(chunk, record: dict, by_id: dict, frames_median: float | None, cache: dict) -> solids.Solid:
    heights = [h for h in (_frame_height(by_id[s]) for s in record.get("sources", []) if s in by_id) if h]
    height = float(np.median(heights)) if heights else frames_median
    if not height:
        raise NotReady(t(f"{_label(record)}: nessuna ordinata costruita da cui prendere l'altezza.",
                         f"{_label(record)}: no frame built to take the height from."))
    poly = solids._polygon(record["ship_xy"])
    grid = solids._Grid(chunk, poly.bounds, margin=2, surface=solids.surface_for(chunk, poly, cache),
                        path=_height(chunk))
    mask = solids._no_bowties(grid.inside(poly) & np.isfinite(grid.z))
    if not mask.any():
        raise NotReady(t(f"{_label(record)}: nessuna superficie dello scafo sotto il contorno.",
                         f"{_label(record)}: no surface of the hull under the outline."))
    bottom = grid.z
    top = bottom + height
    v, f, hid, vol = solids._closed(mask, solids._plane(grid, solids._corners(top, mask)),
                                    solids._plane(grid, solids._corners(bottom, mask)))
    return solids.Solid(v, f, hid, record["status"], record["name"], record["kind"], vol, None, record["id"])


def _new_solids(chunk, frame, progress, records=None) -> tuple[list[dict], list[dict]]:
    """The solids of the extension's pieces and of the spaced frames: (built, refused)."""
    from . import reconstruct

    given = records
    records = timbers.timbers_of(chunk) if records is None else records
    recon = reconstruct.records_of(chunk)
    by_id = {r["id"]: r for r in records} | {r["id"]: r for r in recon}
    measured = [h for h in (_frame_height(r) for r in records
                            if r["kind"] == "frame" and timbers.is_measured(r) and not r.get("loose")) if h]
    frames_median = float(np.median(measured)) if measured else None
    todo = [r for r in ship.state(chunk)["reconstruction"]["timbers"]
            if r.get("status") in (reconstruct.EXTENSION, reconstruct.SPACING)]
    folder = reconstruct.folder(chunk)
    built, refused = [], []
    cache: dict = {}  # the 3D files read for the surface, once per build
    for k, r in enumerate(todo):
        progress(k / max(1, len(todo)), t(f"Legni ricostruiti in 3D: {_label(r)}", f"Reconstructed timbers in 3D: "
                                                                                  f"{_label(r)}"))
        r.pop("solid", None)
        r.pop("volume", None)
        try:
            if r["status"] == reconstruct.EXTENSION:
                solid = _extension_solid(chunk, r, by_id, cache, given)
            else:
                solid = _spacing_solid(chunk, r, by_id, frames_median, cache)
            path = solids.write_ply(folder / f"{r['id']}.ply", solid, frame)
        except Exception as exc:  # noqa: BLE001 - one piece refused, never the whole build
            (folder / f"{r['id']}.ply").unlink(missing_ok=True)
            refused.append({"id": r["id"], "name": _label(r), "kind": r["kind"], "status": r["status"],
                            "reason": _reason(exc)})
            continue
        r["solid"] = str(path.relative_to(chunk.data_dir))
        r["volume"] = float(solid.volume)
        built.append({"id": r["id"], "name": _label(r), "kind": r["kind"], "status": r["status"], "file": r["solid"],
                      "volume": float(solid.volume)})
    return built, refused


# ----------------------------------------------------------------------------- the build
def _backup(folder: Path) -> Path | None:
    """A copy of the reconstruction's folder by hard links (the files are replaced, never written over in place);
    a plain copy on a file system without hard links (FAT, exFAT, some network shares)."""
    bak = folder.with_name(folder.name + ".bak")
    shutil.rmtree(bak, ignore_errors=True)
    if not folder.exists():
        return None
    try:
        shutil.copytree(folder, bak, copy_function=os.link)
    except OSError:
        shutil.rmtree(bak, ignore_errors=True)
        shutil.copytree(folder, bak, copy_function=shutil.copy2)
    return bak


def _restore(folder: Path, bak: Path | None) -> None:
    shutil.rmtree(folder, ignore_errors=True)
    if bak is not None:
        bak.replace(folder)
    else:
        folder.mkdir(parents=True, exist_ok=True)


def _clean(chunk, folder: Path) -> None:
    from . import reconstruct

    keep = {f"{r['id']}.ply" for r in reconstruct.records_of(chunk) if r.get("solid")}
    if (reconstruct.state(chunk).get("loft") or {}).get("file"):
        keep.add(reconstruct.LOFT_FILE)
    reconstruct.remove_orphans(folder, keep)


def _members(chunk, doc, progress) -> tuple[list[dict] | None, list[dict] | None, dict | None]:
    """The other chunks of the wreck for a build: (the merged records or None, the merged station sections or None,
    the ``members`` entry to keep: ``chunks``, ``refused``, ``raster``, ``warnings`` - the members of the previous build
    gone from the project, and the merge's). The merged raster is written here."""
    from . import chunks, reconstruct, relief

    doc = doc if doc is not None else getattr(chunk, "document", None)
    previous = (reconstruct.state(chunk).get("members") or {}).get("chunks") or []
    if doc is None or relief.entry(chunk) is None:
        return None, None, None
    present = {c.key for c in doc.chunks}
    warnings = [t(f"Il chunk «{m.get('label', m['chunk'])}» della ricostruzione precedente non è più nel progetto: la "
                  "ricostruzione è fatta senza.",
                  f"The chunk «{m.get('label', m['chunk'])}» of the previous reconstruction is no longer in the project: "
                  "the reconstruction is made without it.")
                for m in previous if m["chunk"] != chunk.key and m["chunk"] not in present]
    members, refused = chunks.members(doc, chunk)
    entry = {"chunks": [], "refused": [], "raster": None, "warnings": warnings}
    records = sections = None
    if len(members) < 2:
        entry["chunks"] = [{"chunk": chunk.key, "label": chunk.label, "cell": members[0]["cell"] if members else None,
                            "coverage": 1.0}]
        shutil.rmtree(reconstruct.folder(chunk) / chunks.FOLDER, ignore_errors=True)
    else:
        entry.update(chunks.prepare(doc, chunk, progress, members))
        records = chunks.merged(doc, chunk, members, warnings)
        try:
            sections = chunks.station_sections(members, refused)
        except NotReady:
            sections = None  # the main's own out of date: the loft says so
    entry["refused"] = [{"chunk": r["chunk"].key, "label": r["chunk"].label, "reason": r["reason"]} for r in refused]
    return records, sections, entry


def _skip(chunk, status: str | None, key: str) -> None:
    """A step left out of a build: nothing of its kind is kept (its records and its entry; the files go with
    :func:`_clean`)."""
    from . import reconstruct

    if status is not None:
        reconstruct._replace(chunk, status, [])
    reconstruct.state(chunk).pop(key, None)


def build(chunk, progress=None, doc=None, steps=STEPS) -> dict:
    """The whole reconstruction (see the module): mirror, extension, frames by spacing, loft, the new parts' solids.
    ``steps``: those of :data:`STEPS` to run; a step left out keeps nothing of its kind (no mirrored record, no
    extension, no spaced frame, no faired hull), recorded in ``params["steps"]``.
    Returns ``{"mirrored", "extended", "spaced", "loft", "built", "refused", "open_ends", "warnings", "notes",
    "table", "members"}`` (``warnings``: the extension's, the loft's and those of the members - a member of the
    previous build no longer in the project, the merge's - kept in ``members["warnings"]`` too; ``notes``: the
    loft's). The stored disagreement of the two sides (``disagreement``) is
    kept with the user's ends. ``doc``: the project whose other chunks of the wreck are read (default the chunk's)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import reconstruct

    progress = as_progress(progress)
    wanted = set(steps)
    if wanted - set(STEPS):
        raise ValueError(f"steps: {sorted(wanted - set(STEPS))} unknown (of {', '.join(STEPS)})")
    steps = [s for s in STEPS if s in wanted]
    frame = ship.need_frame(chunk)
    root = ship.state(chunk)
    before = copy.deepcopy(root.get("reconstruction"))
    folder = reconstruct.folder(chunk)
    bak = _backup(folder)
    try:
        progress(0.0, t("Ricostruzione: i chunk del relitto", "Reconstruction: the chunks of the wreck"))
        records, sections, members = _members(chunk, doc, progress.sub(0.0, 0.05))
        if members is not None:
            reconstruct.state(chunk)["members"] = members
        else:
            reconstruct.state(chunk).pop("members", None)
        mirrored = extended = spaced = []
        solid = None
        if "mirror" in steps:
            progress(0.05, t("Ricostruzione: simmetria", "Reconstruction: mirror"))
            mirrored = reconstruct.mirror(chunk, progress.sub(0.05, 0.20), records=records)
        else:
            _skip(chunk, reconstruct.STATUS, "mirror")
        if "extension" in steps:
            progress(0.25, t("Ricostruzione: prolungamento delle fasce", "Reconstruction: extension of the strakes"))
            extended = reconstruct.extend(chunk, progress.sub(0.25, 0.20), records=records)
        else:
            _skip(chunk, reconstruct.EXTENSION, "extension")
        if "frames" in steps:
            progress(0.45, t("Ricostruzione: ordinate al passo misurato",
                             "Reconstruction: frames at the measured spacing"))
            spaced = reconstruct.frames_by_spacing(chunk, records=records)
        else:
            _skip(chunk, reconstruct.SPACING, "spacing")
        if "loft" in steps:
            progress(0.55, t("Ricostruzione: carenatura", "Reconstruction: fairing"))
            solid = reconstruct.loft(chunk, STEP, SAMPLES, records=records, sections=sections or None)
        else:
            _skip(chunk, None, "loft")
        progress(0.7, t("Ricostruzione: legni in 3D", "Reconstruction: timbers in 3D"))
        built, refused = _new_solids(chunk, frame, progress.sub(0.7, 0.3), records)
        st = reconstruct.state(chunk)
        new = {"params": {"step": STEP, "samples": SAMPLES, "steps": steps},
               "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "timbers": [r for r in st.get("timbers", []) if r.get("status") in STATUSES]}
        for key in ("mirror", "extension", "spacing", "loft", "ends", "disagreement", "members"):
            if key in st:
                new[key] = st[key]
        new["open_ends"] = list((st.get("loft") or {}).get("open_ends") or [])
        root["reconstruction"] = new
        _clean(chunk, folder)
        new["table"] = table(chunk, records)
    except BaseException:
        if before is None:
            root.pop("reconstruction", None)
        else:
            root["reconstruction"] = before
        _restore(folder, bak)
        raise
    finally:
        if bak is not None and bak.exists():
            shutil.rmtree(bak, ignore_errors=True)
    progress(1.0, t(f"Ricostruzione: {len(mirrored)} per simmetria, {len(extended)} prolungamenti, {len(spaced)} "
                    f"ordinate, {len(refused)} rifiutati",
                    f"Reconstruction: {len(mirrored)} mirrored, {len(extended)} extended, {len(spaced)} frames, "
                    f"{len(refused)} refused"))
    return {"mirrored": len(mirrored), "extended": len(extended), "spaced": len(spaced), "loft": solid is not None,
            "built": built, "refused": refused, "open_ends": new["open_ends"],
            "warnings": list((new.get("extension") or {}).get("warnings") or [])
            + list((new.get("loft") or {}).get("warnings") or []) + list((members or {}).get("warnings") or []),
            "notes": list((new.get("loft") or {}).get("notes") or []),
            "table": new["table"], "members": copy.deepcopy(new.get("members"))}


# ----------------------------------------------------------------------------- the table
def _area(chunk, r: dict, frame) -> float | None:
    if r["geometry"]["type"] != "polygon":
        return None
    return float(solids._polygon(tracing.ship_xy_of(chunk, r, frame)).area)


def _round(v, digits):
    return None if v is None else round(float(v), digits)


def _plan_length(chunk, r: dict, frame) -> float | None:
    """The length in plan of a record merged from several chunks (a polygon's long side, a line's run)."""
    xy = tracing.ship_xy_of(chunk, r, frame)
    if r["geometry"]["type"] == "polygon":
        return timbers._polygon_sides(r["geometry"], frame)[0]
    if r["geometry"]["type"] == "line":
        return float(np.hypot(*np.diff(xy, axis=0).T).sum())
    return None


def _merged(chunk, r: dict) -> bool:
    sources = [s for s in r.get("sources") or [] if isinstance(s, dict)]
    return bool(sources) and (len(sources) > 1 or sources[0].get("chunk") != chunk.key)


def table(chunk, records=None) -> dict:
    """What is measured and what is reconstructed: ``rows`` (one per record of the catalogue, then of the
    reconstruction, then the faired hull: ``id``, ``name``, ``kind``, ``side``, ``status``, ``measured_length``,
    ``measured_area``, ``reconstructed_length``, ``reconstructed_area`` - a measured record fills the first two, any
    other the last two; length in m, as :func:`timbers.measure` gives it for the catalogue, area in m2 in the ship
    plan - and ``length_kind``, how the length was measured: "3D" along a line over the model, "pianta" in the ship
    plan (a polygon's long side, a record merged from several chunks, the reconstruction), "" without a length), ``totals`` (``measured`` and ``reconstructed`` sums, per ``kind`` and per ``status``), ``counts`` per
    status, ``mirror`` (its pieces, refusals and the disagreements of the two sides). ``records``: the measured
    records (default the catalogue; the merged records of several chunks: a record merged from more than one chunk
    has its length in plan, the catalogue's are measured on the model)."""
    from . import reconstruct

    frame = ship.need_frame(chunk)
    rows = []
    lengths = {m["id"]: m["length"] for m in timbers.measure(chunk)}

    model = chunk.product_path("model")
    on_model = model is not None and Path(model).exists()

    def row(r, status, length, area, how="pianta"):
        measured = status == "measured"
        return {"id": r["id"], "name": r.get("name") or "", "kind": r["kind"], "side": r.get("side") or "",
                "status": status,
                "measured_length": _round(length, 4) if measured else None,
                "measured_area": _round(area, 6) if measured else None,
                "reconstructed_length": None if measured else _round(length, 4),
                "reconstructed_area": None if measured else _round(area, 6),
                "length_kind": how if length is not None else ""}

    for r in timbers.timbers_of(chunk) if records is None else records:
        if _merged(chunk, r):  # several chunks' parts: its length in plan
            length, how = _plan_length(chunk, r, frame), "pianta"
        else:  # as the catalogue measures it: a line along the model, a polygon's long side in plan
            length = lengths.get(r["id"])
            how = "3D" if r["geometry"]["type"] == "line" and on_model else "pianta"
        rows.append(row(r, timbers.status_of(r), length, _area(chunk, r, frame), how))
    for r in reconstruct.records_of(chunk):
        length = (timbers._polygon_sides(r["geometry"], frame)[0] if r["geometry"]["type"] == "polygon" else None)
        rows.append(row(r, r.get("status") or "reconstructed", length, _area(chunk, r, frame)))
    loft = reconstruct.state(chunk).get("loft") or {}
    if loft.get("file"):
        rows.append(row({"id": "carenatura", "name": t("carenatura", "fairing"), "kind": "hull", "side": "centre"},
                        "reconstructed:fairing", loft.get("length"), loft.get("area")))
    totals = {"measured": {"length": 0.0, "area": 0.0}, "reconstructed": {"length": 0.0, "area": 0.0}, "kind": {},
              "status": {}}
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        k = totals["kind"].setdefault(r["kind"], {"measured_length": 0.0, "measured_area": 0.0,
                                                  "reconstructed_length": 0.0, "reconstructed_area": 0.0})
        s = totals["status"].setdefault(r["status"], {"length": 0.0, "area": 0.0})
        for what in ("measured", "reconstructed"):
            for q in ("length", "area"):
                v = r[f"{what}_{q}"]
                if v is None:
                    continue
                totals[what][q] += v
                k[f"{what}_{q}"] += v
                s[q] += v
    for group in (totals["measured"], totals["reconstructed"], *totals["kind"].values(), *totals["status"].values()):
        for q in group:
            group[q] = round(group[q], 6)
    try:
        disagreements = reconstruct.disagreement_rows(chunk, records)
    except NotReady:
        disagreements = []
    mirror = reconstruct.state(chunk).get("mirror") or {}
    summary = {"pieces": counts.get(reconstruct.STATUS, 0), "refused": len(mirror.get("refused") or []),
               "disagreements": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
                                 for d in disagreements]}
    return {"rows": rows, "totals": totals, "counts": counts, "mirror": summary}


def _merged_records(chunk, doc=None) -> list[dict] | None:
    """The records of the reconstruction's chunks merged (:func:`chunks.merged`) when it was made over several, else
    None (the catalogue)."""
    from . import chunks, reconstruct

    members = (reconstruct.state(chunk).get("members") or {}).get("chunks") or []
    doc = doc if doc is not None else getattr(chunk, "document", None)
    if len(members) < 2 or doc is None:
        return None
    found, _refused = chunks.members(doc, chunk)
    return chunks.merged(doc, chunk, found) if len(found) > 1 else None


def write_table(chunk, folder=None, records=None, doc=None) -> dict:
    """:func:`table` as ``tabella.csv`` (codes, as the plugin's other tables) and ``tabella.xlsx`` (titles, through the
    program's :func:`pyarchinit_photogrammetry.io.tables.write_xlsx`; the sheet "misurato e ricostruito", the
    workbook's title "misurato e ricostruito (ipotesi)") in ``folder`` (default: the reconstruction's). ``records``:
    the measured records (default: the records of the chunks merged when the reconstruction was made over several,
    so that the table and the files agree; else the catalogue). Returns ``{"csv", "xlsx", "rows"}``."""
    import csv

    from pyarchinit_photogrammetry.io.tables import Sheet, write_xlsx

    from . import reconstruct

    rows = table(chunk, records if records is not None else _merged_records(chunk, doc))["rows"]
    folder = Path(folder) if folder is not None else reconstruct.folder(chunk)
    folder.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = folder / f"{TABLE_FILE}.csv", folder / f"{TABLE_FILE}.xlsx"
    tmp = csv_path.with_name(csv_path.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(TABLE_COLUMNS)
        for r in rows:
            w.writerow(["" if r[c] is None else r[c] for c in TABLE_COLUMNS])
    tmp.replace(csv_path)
    headers = [t("id", "id"), t("nome", "name"), t("tipo", "kind"), t("lato", "side"), t("stato", "status"),
               t("lunghezza misurata (m)", "measured length (m)"), t("area misurata (m²)", "measured area (m²)"),
               t("lunghezza ricostruita (m)", "reconstructed length (m)"),
               t("area ricostruita (m²)", "reconstructed area (m²)"), t("lunghezza misurata", "length measured")]
    titled = [[r["id"], r["name"], timbers.kind_title(r["kind"]) if r["kind"] in timbers.KIND_TITLES
               else t("scafo", "hull"), timbers.side_title(r["side"]) if r["side"] in timbers.SIDE_TITLES else r["side"],
               timbers.status_title(r["status"]), r["measured_length"], r["measured_area"], r["reconstructed_length"],
               r["reconstructed_area"], t(*LENGTH_KINDS[r["length_kind"]]) if r["length_kind"] else ""] for r in rows]
    tmp = xlsx_path.with_name(f"{TABLE_FILE}.tmp.xlsx")
    try:
        write_xlsx(tmp, {t("misurato e ricostruito", "measured and reconstructed"):
                         Sheet(headers, titled, [None, None, None, None, None, 3, 4, 3, 4, None])},
                   title=t("misurato e ricostruito (ipotesi)", "measured and reconstructed (hypothesis)"))
        tmp.replace(xlsx_path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"csv": csv_path, "xlsx": xlsx_path, "rows": len(rows)}
