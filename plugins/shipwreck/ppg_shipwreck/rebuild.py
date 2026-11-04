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
TABLE_COLUMNS = ("id", "name", "kind", "side", "status", "measured_length", "measured_area", "reconstructed_length",
                 "reconstructed_area")
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


def _extension_solid(chunk, record: dict, by_id: dict, cache: dict) -> solids.Solid:
    source = by_id.get(record["source"])
    if source is None:
        raise NotReady(t(f"{_label(record)}: la fascia da cui viene non c'è più.",
                         f"{_label(record)}: the strake it comes from is gone."))
    mm, _how = solids.thickness_of(chunk, source)
    record["thickness_mm"] = float(mm)
    poly = solids._polygon(record["ship_xy"])
    surface = _band_surface(record, solids.surface_for(chunk, poly, cache))
    return solids.strake_shell(chunk, record, mm, surface=surface)


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
    grid = solids._Grid(chunk, poly.bounds, margin=2, surface=solids.surface_for(chunk, poly, cache))
    mask = solids._no_bowties(grid.inside(poly) & np.isfinite(grid.z))
    if not mask.any():
        raise NotReady(t(f"{_label(record)}: nessuna superficie dello scafo sotto il contorno.",
                         f"{_label(record)}: no surface of the hull under the outline."))
    bottom = grid.z
    top = bottom + height
    v, f, hid, vol = solids._closed(mask, solids._plane(grid, solids._corners(top, mask)),
                                    solids._plane(grid, solids._corners(bottom, mask)))
    return solids.Solid(v, f, hid, record["status"], record["name"], record["kind"], vol, None, record["id"])


def _new_solids(chunk, frame, progress) -> tuple[list[dict], list[dict]]:
    """The solids of the extension's pieces and of the spaced frames: (built, refused)."""
    from . import reconstruct

    recon = reconstruct.records_of(chunk)
    by_id = {r["id"]: r for r in timbers.timbers_of(chunk)} | {r["id"]: r for r in recon}
    measured = [h for h in (_frame_height(r) for r in timbers.timbers_of(chunk)
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
                solid = _extension_solid(chunk, r, by_id, cache)
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


def build(chunk, progress=None) -> dict:
    """The whole reconstruction (see the module): mirror, extension, frames by spacing, loft, the new parts' solids.
    Returns ``{"mirrored", "extended", "spaced", "loft", "built", "refused", "open_ends", "warnings", "notes",
    "table"}`` (``warnings``: the extension's and the loft's; ``notes``: the loft's). The stored disagreement of
    the two sides (``disagreement``) is kept with the user's ends."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import reconstruct

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    root = ship.state(chunk)
    before = copy.deepcopy(root.get("reconstruction"))
    folder = reconstruct.folder(chunk)
    bak = _backup(folder)
    try:
        progress(0.0, t("Ricostruzione: simmetria", "Reconstruction: mirror"))
        mirrored = reconstruct.mirror(chunk, progress.sub(0.0, 0.25))
        progress(0.25, t("Ricostruzione: prolungamento delle fasce", "Reconstruction: extension of the strakes"))
        extended = reconstruct.extend(chunk, progress.sub(0.25, 0.20))
        progress(0.45, t("Ricostruzione: ordinate al passo misurato", "Reconstruction: frames at the measured spacing"))
        spaced = reconstruct.frames_by_spacing(chunk)
        progress(0.55, t("Ricostruzione: carenatura", "Reconstruction: fairing"))
        solid = reconstruct.loft(chunk, STEP, SAMPLES)
        progress(0.7, t("Ricostruzione: legni in 3D", "Reconstruction: timbers in 3D"))
        built, refused = _new_solids(chunk, frame, progress.sub(0.7, 0.3))
        st = reconstruct.state(chunk)
        new = {"params": {"step": STEP, "samples": SAMPLES},
               "built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "timbers": [r for r in st.get("timbers", []) if r.get("status") in STATUSES]}
        for key in ("mirror", "extension", "spacing", "loft", "ends", "disagreement"):
            if key in st:
                new[key] = st[key]
        new["open_ends"] = list((st.get("loft") or {}).get("open_ends") or [])
        root["reconstruction"] = new
        _clean(chunk, folder)
        new["table"] = table(chunk)
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
            + list((new.get("loft") or {}).get("warnings") or []),
            "notes": list((new.get("loft") or {}).get("notes") or []),
            "table": new["table"]}


# ----------------------------------------------------------------------------- the table
def _area(chunk, r: dict, frame) -> float | None:
    if r["geometry"]["type"] != "polygon":
        return None
    return float(solids._polygon(tracing.ship_xy_of(chunk, r, frame)).area)


def _round(v, digits):
    return None if v is None else round(float(v), digits)


def table(chunk) -> dict:
    """What is measured and what is reconstructed: ``rows`` (one per record of the catalogue, then of the
    reconstruction, then the faired hull: ``id``, ``name``, ``kind``, ``side``, ``status``, ``measured_length``,
    ``measured_area``, ``reconstructed_length``, ``reconstructed_area`` - a measured record fills the first two, any
    other the last two; length in m, as :func:`timbers.measure` gives it for the catalogue, area in m2 in the ship
    plan), ``totals`` (``measured`` and ``reconstructed`` sums, per ``kind`` and per ``status``), ``counts`` per
    status, ``mirror`` (its pieces, refusals and the disagreements of the two sides)."""
    from . import reconstruct

    frame = ship.need_frame(chunk)
    rows = []
    lengths = {m["id"]: m["length"] for m in timbers.measure(chunk)}

    def row(r, status, length, area):
        measured = status == "measured"
        return {"id": r["id"], "name": r.get("name") or "", "kind": r["kind"], "side": r.get("side") or "",
                "status": status,
                "measured_length": _round(length, 4) if measured else None,
                "measured_area": _round(area, 6) if measured else None,
                "reconstructed_length": None if measured else _round(length, 4),
                "reconstructed_area": None if measured else _round(area, 6)}

    for r in timbers.timbers_of(chunk):
        rows.append(row(r, timbers.status_of(r), lengths.get(r["id"]), _area(chunk, r, frame)))
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
        disagreements = reconstruct.disagreement_rows(chunk)
    except NotReady:
        disagreements = []
    mirror = reconstruct.state(chunk).get("mirror") or {}
    summary = {"pieces": counts.get(reconstruct.STATUS, 0), "refused": len(mirror.get("refused") or []),
               "disagreements": [{k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}
                                 for d in disagreements]}
    return {"rows": rows, "totals": totals, "counts": counts, "mirror": summary}


def write_table(chunk, folder=None) -> dict:
    """:func:`table` as ``tabella.csv`` (codes, as the plugin's other tables) and ``tabella.xlsx`` (titles, through the
    program's :func:`pyarchinit_photogrammetry.io.tables.write_xlsx`; the sheet "misurato e ricostruito", the
    workbook's title "misurato e ricostruito (ipotesi)") in ``folder`` (default: the reconstruction's). Returns
    ``{"csv", "xlsx", "rows"}``."""
    import csv

    from pyarchinit_photogrammetry.io.tables import Sheet, write_xlsx

    from . import reconstruct

    rows = table(chunk)["rows"]
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
               t("area ricostruita (m²)", "reconstructed area (m²)")]
    titled = [[r["id"], r["name"], timbers.kind_title(r["kind"]) if r["kind"] in timbers.KIND_TITLES
               else t("scafo", "hull"), timbers.side_title(r["side"]) if r["side"] in timbers.SIDE_TITLES else r["side"],
               timbers.status_title(r["status"]), r["measured_length"], r["measured_area"], r["reconstructed_length"],
               r["reconstructed_area"]] for r in rows]
    tmp = xlsx_path.with_name(f"{TABLE_FILE}.tmp.xlsx")
    try:
        write_xlsx(tmp, {t("misurato e ricostruito", "measured and reconstructed"):
                         Sheet(headers, titled, [None, None, None, None, None, 3, 4, 3, 4])},
                   title=t("misurato e ricostruito (ipotesi)", "measured and reconstructed (hypothesis)"))
        tmp.replace(xlsx_path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"csv": csv_path, "xlsx": xlsx_path, "rows": len(rows)}
