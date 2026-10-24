"""The Sentinel-2 analysis of a chunk: what was searched, read and made, in
``chunk.meta["sentinel2"]`` and in ``<chunk data>/sentinel2/``:

* ``scenes.json``: the acquisitions of the last search, with the hrefs of their bands;
* ``scenes/<acquisition>.tif``: the window of the area of each acquisition read (the cache);
* ``indices/<NDVI|NDRE|NDMI>.tif``: one band per clear acquisition, in time order (UTM);
* ``series.csv``, ``series_monthly.csv``, ``series.png``: the time series over the area;
* ``products/<id>.tif``: composites and crop-mark maps on the grid of the scenes (UTM), and
  ``products/<id>_chunk.tif`` the same in the coordinate system of the chunk on a grid aligned
  with its DEM or orthomosaic (what QGIS shows beside the survey).

``chunk.meta["sentinel2"]`` holds the area, the grid, the search, a row per acquisition (date,
cloud cover of the tiles, clear part of the area, status), the settings of the masking, the
stacks of the indices with their dates, the series and the products.
"""
from __future__ import annotations

import csv
import json
import math
import os
import shutil
import time
from datetime import date
from pathlib import Path

import numpy as np

from . import analysis, reader, stac
from .area import Area, Grid, NotPlaced, area_mask, chunk_crs, grid_of
from .texts import lang, product_name, t

FOLDER = "sentinel2"
CHUNK_CELL_M = 2.5  # the cell of the copies on the chunk's grid: a quarter of a 10 m pixel
CHUNK_MAX_SIDE = 4000
Z_LIMIT = 3.0  # the ends of the colours of a crop-mark map (z-scores)


class NoData(RuntimeError):
    """Nothing to work on yet: no search, no acquisition read, no clear date in a period."""


def default_dates() -> tuple[str, str]:
    """The dates of a search when none is given: from the first of January two years ago to today."""
    today = date.today()
    return date(today.year - 2, 1, 1).isoformat(), today.isoformat()


# ------------------------------------------------------------------ where things are
def folder(chunk) -> Path:
    return Path(chunk.data_dir) / FOLDER


def state(chunk) -> dict:
    """The plugin's part of ``chunk.meta`` (created when missing)."""
    return chunk.meta.setdefault("sentinel2", {})


def peek(chunk) -> dict:
    """The plugin's part of ``chunk.meta`` as it is (empty when missing), not created."""
    return (chunk.meta.get("sentinel2") or {}) if chunk is not None else {}


def rel(chunk, path) -> str:
    return Path(os.path.relpath(Path(path), Path(chunk.data_dir))).as_posix()


def absolute(chunk, relpath: str | None) -> Path | None:
    return None if not relpath else Path(chunk.data_dir) / relpath


def settings(chunk) -> dict:
    """The masking: ``min_clear`` (the least clear part of the area), ``clear`` (the SCL classes
    kept), ``buffer_m`` (around clouds and shadows)."""
    s = dict(peek(chunk).get("settings") or {})
    s.setdefault("min_clear", reader.MIN_CLEAR)
    s.setdefault("clear", list(reader.CLEAR))
    s.setdefault("buffer_m", reader.BUFFER_M)
    return s


def area_of(chunk) -> Area | None:
    d = peek(chunk).get("area")
    return Area.from_dict(d) if d else None


def grid_of_chunk(chunk) -> Grid | None:
    d = peek(chunk).get("grid")
    return Grid.from_dict(d) if d else None


def set_area(chunk, area: Area) -> Grid:
    """The area of the analysis (and its grid); the scenes found before stay, read again on the
    new grid when it changed."""
    grid = grid_of(area)
    st = state(chunk)
    old = st.get("grid")
    st["area"] = area.to_dict()
    st["grid"] = grid.to_dict()
    if old and Grid.from_dict(old) != grid:
        for row in st.get("scenes") or []:  # what was read belongs to the old grid
            if row.get("status") in ("cached", "cloudy"):
                row["status"], row["clear"], row["file"] = "found", None, None
    return grid


def years_of(chunk) -> list[int]:
    return sorted({int(d["date"][:4]) for d in dates(chunk)})


def attribution(years) -> str:
    """The notice the licence of the Sentinel data asks for modified data."""
    ys = sorted({int(y) for y in years})
    if not ys:
        ys = [date.today().year]
    span = str(ys[0]) if ys[0] == ys[-1] else f"{ys[0]}-{ys[-1]}"
    return f"Contains modified Copernicus Sentinel data {span}"


def attribution_of(chunk) -> str:
    years = years_of(chunk) or sorted({int(r["date"][:4]) for r in peek(chunk).get("scenes") or [] if r.get("date")})
    return attribution(years)


# ------------------------------------------------------------------ search
def acquisitions(chunk) -> list[dict]:
    """The acquisitions of the last search with their bands (``scenes.json``)."""
    path = folder(chunk) / "scenes.json"
    if not path.exists():
        return []
    try:
        return list(json.loads(path.read_text(encoding="utf-8")).get("acquisitions") or [])
    except (OSError, ValueError):
        return []


def scene_rows(chunk) -> list[dict]:
    return list(peek(chunk).get("scenes") or [])


def search(chunk, area: Area, start, end, max_cloud: float = 30.0, catalog: str = stac.EARTH_SEARCH,
           collection: str = stac.COLLECTION, opener=None, client: str | None = None, progress=None) -> dict:
    """The acquisitions over ``area`` (with its margin) from ``start`` to ``end`` with at most
    ``max_cloud`` % of clouds on their tiles; they replace those of the previous search (what
    was read of them stays in the cache)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    bbox = area.lonlat_bbox(margin=True)
    progress(0.0, t("Ricerca nel catalogo {0}", "Searching the catalog {0}").format(catalog))
    items = stac.search(bbox, start, end, max_cloud, catalog, collection, opener=opener, client=client,
                        progress=progress.sub(0.0, 0.9))
    acqs = stac.acquisitions(items)
    grid = set_area(chunk, area)
    st = state(chunk)
    old = {r["id"]: r for r in st.get("scenes") or []}
    cache = folder(chunk) / "scenes"
    rows = []
    for a in acqs:
        row = {"id": a["id"], "date": a["date"], "datetime": a["datetime"], "platform": a["platform"],
               "tiles": a["tiles"], "cloud": a["cloud"], "status": "found", "clear": None, "file": None}
        path = cache / f"{a['id']}.tif"
        if reader.cache_valid(path, grid):
            row.update(status="cached", clear=reader.cache_clear_fraction(path), file=rel(chunk, path))
        elif old.get(a["id"], {}).get("status") == "cloudy":
            row.update(status="cloudy", clear=old[a["id"]].get("clear"))
        rows.append(row)
    folder(chunk).mkdir(parents=True, exist_ok=True)
    payload = {"catalog": catalog, "collection": collection, "bbox": list(bbox), "acquisitions": acqs}
    (folder(chunk) / "scenes.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    st["scenes"] = rows
    st["search"] = {"catalog": catalog, "collection": collection, "start": str(start), "end": str(end),
                    "max_cloud": float(max_cloud), "bbox": [float(v) for v in bbox], "items": len(items),
                    "acquisitions": len(acqs), "when": time.strftime("%Y-%m-%d %H:%M")}
    progress(1.0, t("{0} acquisizioni ({1} scene del catalogo)", "{0} acquisitions ({1} catalog scenes)")
             .format(len(acqs), len(items)))
    return {"items": len(items), "acquisitions": len(acqs), "cached": sum(r["status"] == "cached" for r in rows),
            "bbox": list(bbox), "grid": grid.to_dict()}


# ------------------------------------------------------------------ reading
def fetch(chunk, ids=None, min_clear: float | None = None, workers: int = reader.WORKERS, clear=None,
          buffer_m: float | None = None, progress=None, compute_after: bool = True) -> dict:
    """The acquisitions of the search (or ``ids``) read on the grid of the area and cached; the
    known cloudy ones and those in the cache are not read again. Then the indices and the series
    (:func:`compute`) unless ``compute_after`` is False."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    area, grid = area_of(chunk), grid_of_chunk(chunk)
    acqs = acquisitions(chunk)
    if area is None or grid is None or not acqs:
        raise NoData(t("Nessuna scena: cerca prima nel catalogo.", "No scene: search the catalog first."))
    st = state(chunk)
    s = settings(chunk)
    if min_clear is not None:
        s["min_clear"] = float(min_clear)
    if clear is not None:
        s["clear"] = sorted({int(c) for c in clear})
    if buffer_m is not None:
        s["buffer_m"] = float(buffer_m)
    st["settings"] = s
    rows = {r["id"]: r for r in st.get("scenes") or []}
    wanted = None if not ids else {str(i) for i in ids}
    chosen, skipped = [], 0
    for a in acqs:
        if wanted is not None and a["id"] not in wanted:
            continue
        row = rows.get(a["id"]) or {}
        if row.get("status") == "cloudy" and (row.get("clear") or 0.0) < s["min_clear"]:
            skipped += 1
            continue
        chosen.append(a)
    if wanted is not None and not chosen and not skipped:
        raise NoData(t("Nessuna delle acquisizioni indicate è nella ricerca.", "None of the acquisitions given is in the search."))
    mask = area_mask(area, grid)
    span = 0.75 if compute_after else 1.0
    results = reader.fetch(chosen, grid, mask, folder(chunk) / "scenes", s["min_clear"], s["clear"], s["buffer_m"],
                           workers, attribution=lambda a: attribution([int(a["date"][:4])]),
                           progress=progress.sub(0.0, span))
    for acq_id, r in results.items():
        row = rows.get(acq_id)
        if row is None:
            continue
        row["status"], row["clear"] = r["status"], r.get("clear")
        row["file"] = rel(chunk, r["file"]) if r.get("file") else None
        if r.get("error"):
            row["error"] = r["error"]
        else:
            row.pop("error", None)
    counts = reader.describe(results)
    counts["skipped_cloudy"] = skipped
    if compute_after:
        try:
            counts["computed"] = compute(chunk, progress.sub(span, 1.0 - span))
        except NoData as exc:
            counts["computed"] = {"dates": 0, "note": str(exc)}
    mark_missing(chunk)  # the products whose acquisitions were read again lose their mark
    return counts


# ------------------------------------------------------------------ the cache of the acquisitions read
def cache_dir(chunk) -> Path:
    return folder(chunk) / "scenes"


def cached_ids(chunk) -> list[str]:
    """The acquisitions of the search read into the cache (status ``cached``), in time order."""
    rows = sorted(scene_rows(chunk), key=lambda r: (r.get("datetime") or "", r.get("id")))
    return [r["id"] for r in rows if r.get("status") == "cached"]


def _acquisition_of(file: Path) -> str:
    name = file.name
    return name[:-len(".part.tif")] if name.endswith(".part.tif") else name[:-len(".tif")]


def cache_size(chunk) -> tuple[int, int]:
    """How many files the cache of the chunk has, and their bytes."""
    d = cache_dir(chunk)
    files = list(d.glob("*.tif")) if d.exists() else []
    return len(files), sum(p.stat().st_size for p in files)


def products_using(chunk, ids) -> list[dict]:
    """The products computed (also) from the acquisitions ``ids``."""
    ids = {str(i) for i in ids}
    return [e for e in products(chunk) if ids & set(e.get("dates") or [])]


def mark_missing(chunk) -> list[str]:
    """Every product told which of its acquisitions are no longer in the cache (``missing``: removed,
    or read on another grid since): the product stays as it was computed, the mark says it cannot be
    made again from the cache as it is. Products whose acquisitions are all back lose the mark.
    Returns the ids of the products marked."""
    grid = grid_of_chunk(chunk)
    status = {r["id"]: r.get("status") for r in scene_rows(chunk)}
    marked = []
    for e in state(chunk).get("products") or []:
        gone = [d for d in e.get("dates") or []
                if status.get(d) != "cached" and (status.get(d) is not None or grid is None
                                                  or not reader.cache_valid(cache_dir(chunk) / f"{d}.tif", grid))]
        if gone:
            e["missing"] = gone
            marked.append(e["id"])
        else:
            e.pop("missing", None)
    return marked


def clear_series(chunk) -> None:
    """No clear acquisition left in the cache: the stacks of the indices, the series and its chart
    deleted, with their entries (nothing stale stays behind)."""
    st = state(chunk)
    for key in ("indices", "series", "dates"):
        st.pop(key, None)
    base = folder(chunk)
    stale = list((base / "indices").glob("*.tif")) if (base / "indices").exists() else []
    for p in stale + [base / "series.csv", base / "series_monthly.csv", base / "series.png"]:
        try:
            p.unlink()
        except FileNotFoundError:
            pass


def remove_scenes(chunk, ids=None, with_products: bool = False, progress=None) -> dict:
    """Acquisitions out of the cache: their files deleted, their rows back to ``found`` (``fetch``
    reads them again from the catalog). ``ids`` None empties the cache: every file in it, those of
    an older search too. The stacks of the indices and the series are computed again from what is
    left, without network. The products computed from the acquisitions removed are removed too with
    ``with_products``; otherwise they stay as they are and are marked (``missing``: made from
    acquisitions no longer in the cache). Returns what was done."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    st = state(chunk)
    rows = {r["id"]: r for r in st.get("scenes") or []}
    d = cache_dir(chunk)
    if ids is None:
        files_now = sorted(d.glob("*.tif")) if d.exists() else []
        targets = sorted({_acquisition_of(p) for p in files_now} | set(cached_ids(chunk)))
    else:
        targets = list(dict.fromkeys(str(i) for i in ids))
    files, freed = 0, 0
    for acq in targets:
        for p in (d / f"{acq}.tif", d / f"{acq}.part.tif"):
            if p.exists():
                freed += p.stat().st_size
                p.unlink()
                files += 1
        row = rows.get(acq)
        if row is not None and row.get("status") == "cached":
            row.update(status="found", clear=None, file=None)
            row.pop("error", None)
    affected = [e["id"] for e in products_using(chunk, targets)]
    gone = []
    if with_products:
        for pid in affected:
            remove(chunk, pid)
            gone.append(pid)
    progress(0.2, t("{0} acquisizioni tolte dalla cache ({1:.1f} MB)", "{0} acquisitions out of the cache ({1:.1f} MB)")
             .format(len(targets), freed / 1e6))
    try:
        computed = compute(chunk, progress.sub(0.2, 0.8))
    except NoData:
        clear_series(chunk)
        computed = {"dates": 0}
    marked = mark_missing(chunk)
    progress(1.0, t("Cache: {0} file cancellati, {1} date serene restano", "Cache: {0} files deleted, {1} clear "
                    "dates left").format(files, computed.get("dates", 0)))
    return {"removed": targets, "files": files, "bytes": freed, "products_removed": gone,
            "products_marked": [pid for pid in marked if pid in affected], "computed": computed}


# ------------------------------------------------------------------ indices and series
def _read_scl(path: Path, grid: Grid) -> np.ndarray | None:
    import rasterio

    if not reader.cache_valid(path, grid):
        return None
    with rasterio.open(path) as src:
        k = list(src.descriptions).index("SCL") + 1
        return src.read(k).astype(np.uint8)


def compute(chunk, progress=None, min_clear: float | None = None) -> dict:
    """The indices of every clear acquisition in the cache (``indices/<name>.tif``, a band per
    date), the series over the area per date and per month (CSV) and their chart (PNG).
    ``min_clear``: the least clear part of the area an acquisition needs (saved in the settings);
    the one of the last reading when not given."""
    import rasterio
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    area, grid = area_of(chunk), grid_of_chunk(chunk)
    if area is None or grid is None:
        raise NoData(t("Nessuna scena: cerca prima nel catalogo.", "No scene: search the catalog first."))
    st = state(chunk)
    s = settings(chunk)
    if min_clear is not None:
        s["min_clear"] = float(min_clear)
        st["settings"] = s
    mask_area = area_mask(area, grid)
    n_area = max(int(mask_area.sum()), 1)
    usable = []
    rows = sorted([r for r in st.get("scenes") or [] if r.get("status") == "cached"], key=lambda r: r["datetime"])
    for r in rows:
        path = folder(chunk) / "scenes" / f"{r['id']}.tif"
        scl = _read_scl(path, grid)
        if scl is None:  # the cache file is gone: read it again next time
            r["status"], r["file"], r["clear"] = "found", None, None
            continue
        ok = reader.clear_mask(scl, s["clear"], s["buffer_m"], grid.cell)
        r["clear"] = float((ok & mask_area).sum()) / n_area
        if r["clear"] >= s["min_clear"]:
            usable.append((r, path))
    out_dir = folder(chunk) / "indices"
    out_dir.mkdir(parents=True, exist_ok=True)
    if not usable:
        clear_series(chunk)
        raise NoData(t("Nessuna acquisizione serena nella cache: scarica prima le scene.",
                       "No clear acquisition in the cache: read the scenes first."))
    years = sorted({int(r["date"][:4]) for r, _p in usable})
    notice = attribution(years)
    profile = {"driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(usable), "dtype": "float32",
               "crs": grid.crs, "transform": grid.transform, "nodata": float("nan"), "compress": "deflate",
               "predictor": 3, "BIGTIFF": "IF_SAFER"}
    if grid.width >= 256 and grid.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    tmp = {name: out_dir / f"{name}.part.tif" for name in analysis.INDICES}
    series_rows, pooled = [], {}
    dsts = {name: rasterio.open(tmp[name], "w", **profile) for name in analysis.INDICES}
    try:
        for k, (r, path) in enumerate(usable, start=1):
            cache = reader.read_cache(path, grid)
            ok = reader.clear_mask(cache["scl"], s["clear"], s["buffer_m"], grid.cell)
            month = r["date"][:7]
            for name in analysis.INDICES:
                v = analysis.index_of(cache["reflectance"], name, ok)
                dsts[name].write(v, k)
                dsts[name].set_band_description(k, r["id"])
                inside = v[mask_area]
                series_rows.append({"date": r["date"], "acquisition": r["id"], "platform": r["platform"],
                                    "index": name, "clear_fraction": round(r["clear"], 4),
                                    **analysis.stats(inside)})
                ids, values = pooled.setdefault((month, name), ([], []))
                ids.append(r["id"])
                values.append(analysis.sample(inside, seed=k))
            progress(k / len(usable) * 0.85, t("Indici del {0}", "Indices of {0}").format(r["date"]))
        for name, dst in dsts.items():
            dst.update_tags(INDEX=name, FORMULA=analysis.FORMULAS[name],
                            ACQUISITIONS=json.dumps([r["id"] for r, _p in usable]),
                            DATES=json.dumps([r["date"] for r, _p in usable]), ATTRIBUTION=notice,
                            TIFFTAG_COPYRIGHT=notice, TIFFTAG_SOFTWARE="pyarchinit-photogrammetry sentinel2")
    except BaseException:  # cancelled or failed: the stacks of before stay, no half file
        for dst in dsts.values():
            dst.close()
        for p in tmp.values():
            p.unlink(missing_ok=True)
        raise
    for dst in dsts.values():
        dst.close()
    for name in analysis.INDICES:
        os.replace(tmp[name], out_dir / f"{name}.tif")
    months = analysis.monthly(pooled)
    base = folder(chunk)
    write_series_csv(base / "series.csv", series_rows, notice)
    write_monthly_csv(base / "series_monthly.csv", months, notice)
    progress(0.9, t("Grafico della serie", "Chart of the series"))
    title = (t("Sentinel-2 sull'area: {0} date serene, {1}", "Sentinel-2 over the area: {0} clear dates, {1}")
             .format(len(usable), notice.rsplit(" ", 1)[-1]))
    analysis.chart(series_rows, months, base / "series.png", title, notice, lang())
    st["indices"] = {name: {"file": rel(chunk, out_dir / f"{name}.tif"), "formula": analysis.FORMULAS[name]}
                     for name in analysis.INDICES}
    st["dates"] = [{"id": r["id"], "date": r["date"], "datetime": r["datetime"], "platform": r["platform"],
                    "clear": round(r["clear"], 4)} for r, _p in usable]
    st["series"] = {"csv": rel(chunk, base / "series.csv"), "monthly": rel(chunk, base / "series_monthly.csv"),
                    "chart": rel(chunk, base / "series.png"), "attribution": notice,
                    "when": time.strftime("%Y-%m-%d %H:%M")}
    progress(1.0, t("{0} date serene, {1} righe della serie", "{0} clear dates, {1} rows of the series")
             .format(len(usable), len(series_rows)))
    return {"dates": len(usable), "rows": len(series_rows), "months": len(months)}


SERIES_FIELDS = ("date", "acquisition", "platform", "index", "clear_fraction", "cells", "median", "p10", "p25", "p75",
                 "p90", "mean")
MONTHLY_FIELDS = ("month", "index", "dates", "cells", "median", "p10", "p25", "p75", "p90", "mean")


def _csv_header(f, notice: str) -> None:
    f.write(f"# {notice}\n")
    f.write("# " + "; ".join(f"{n} = {analysis.FORMULAS[n]}" for n in analysis.INDICES)
            + "; over the area, clear cells only (scene classification)\n")


def write_series_csv(path: Path, rows: list[dict], notice: str) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        _csv_header(f, notice)
        w = csv.DictWriter(f, fieldnames=SERIES_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    return path


def write_monthly_csv(path: Path, rows: list[dict], notice: str) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        _csv_header(f, notice)
        w = csv.DictWriter(f, fieldnames=MONTHLY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()})
    return path


def read_csv(path: Path) -> list[dict]:
    """The rows of a series CSV (the ``#`` lines skipped), numbers as floats."""
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        lines = [line for line in f if not line.startswith("#")]
    out = []
    for row in csv.DictReader(lines):
        for k, v in list(row.items()):
            if k not in ("date", "acquisition", "platform", "index", "month") and v not in (None, ""):
                try:
                    row[k] = float(v)
                except ValueError:
                    pass
        out.append(row)
    return out


def series(chunk, monthly: bool = False) -> list[dict]:
    s = peek(chunk).get("series") or {}
    path = absolute(chunk, s.get("monthly" if monthly else "csv"))
    return read_csv(path) if path is not None else []


# ------------------------------------------------------------------ the stacks
def dates(chunk) -> list[dict]:
    """The clear acquisitions of the stacks, in time order (``id``, ``date``, ``datetime``)."""
    return list(peek(chunk).get("dates") or [])


def stack_path(chunk, index: str) -> Path:
    info = (peek(chunk).get("indices") or {}).get(index)
    if not info:
        raise NoData(t("Indici non ancora calcolati: scarica e calcola.", "Indices not computed yet: read and compute."))
    return absolute(chunk, info["file"])


def read_stack(chunk, index: str, ids=None):
    """``(values (dates, rows, cols), ids, transform, crs)`` of an index, all dates or ``ids``."""
    import rasterio

    path = stack_path(chunk, index)
    with rasterio.open(path) as src:
        names = list(src.descriptions)
        pick = [i for i, n in enumerate(names) if ids is None or n in set(ids)]
        if not pick:
            return np.zeros((0, src.height, src.width), np.float32), [], src.transform, src.crs
        values = src.read([i + 1 for i in pick]).astype(np.float32)
        return values, [names[i] for i in pick], src.transform, src.crs


def period_ids(chunk, period: str) -> list[str]:
    """The clear acquisitions of a period (see :func:`analysis.check_period`)."""
    period = analysis.check_period(period)
    return [d["id"] for d in dates(chunk) if analysis.in_period(date.fromisoformat(d["date"]), d["id"], period)]


def periods(chunk) -> list[str]:
    return analysis.periods_of([date.fromisoformat(d["date"]) for d in dates(chunk)])


def composite_values(chunk, index: str, period: str):
    """``(median, count, transform, crs, ids)`` of an index over a period (a single acquisition:
    its values)."""
    ids = period_ids(chunk, period)
    if not ids:
        raise NoData(t("Nessuna data serena nel periodo {0}.", "No clear date in the period {0}.").format(period))
    values, ids, tr, crs = read_stack(chunk, index, ids)
    med, count = analysis.median_composite(values)
    return med, count, tr, crs, ids


# ------------------------------------------------------------------ the chunk's grid
def chunk_grid(chunk, grid: Grid) -> dict | None:
    """The grid in the chunk's coordinate system that covers ``grid``: cells of about
    :data:`CHUNK_CELL_M` (a multiple of the cell of the chunk's DEM or orthomosaic, with its
    corners on their lattice); None for a chunk in a local system."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import Affine
    from rasterio.warp import transform_bounds

    try:
        crs = chunk_crs(chunk)
    except NotPlaced:
        return None
    target = CRS.from_user_input(crs)
    b = transform_bounds(grid.crs, target, *grid.bounds, densify_pts=21)
    metres = max(grid.bounds[2] - grid.bounds[0], grid.bounds[3] - grid.bounds[1])
    units = max(b[2] - b[0], b[3] - b[1])
    cell = max(CHUNK_CELL_M * units / metres, units / CHUNK_MAX_SIDE)
    origin = None
    for path in (getattr(chunk, "dem_path", None), getattr(chunk, "orthomosaic_path", None)):
        if path is not None and path.exists():
            with rasterio.open(path) as src:
                if src.crs is None or src.crs == target:
                    origin, base = (src.transform.c, src.transform.f), abs(src.transform.a)
                    break
    if origin is not None:
        if base < cell:
            cell = base * max(1, round(cell / base))
        x0 = origin[0] + math.floor((b[0] - origin[0]) / cell) * cell
        y1 = origin[1] - math.floor((origin[1] - b[3]) / cell) * cell
    else:
        x0, y1 = math.floor(b[0] / cell) * cell, math.ceil(b[3] / cell) * cell
    width, height = int(math.ceil((b[2] - x0) / cell)), int(math.ceil((y1 - b[1]) / cell))
    return {"crs": target, "transform": Affine(cell, 0.0, x0, 0.0, -cell, y1), "width": width, "height": height}


def onto(values: np.ndarray, transform, crs, dst_transform, dst_crs, shape, nearest: bool = True) -> np.ndarray:
    """A raster (NaN where nothing) on another grid: the nearest cell (the pixels of Sentinel-2
    stay pixels), NaN outside."""
    from rasterio.warp import Resampling, reproject

    dst = np.full(shape, np.nan, np.float32)
    with reader.WARP_LOCK:  # one reprojection at a time (see reader.WARP_LOCK)
        reproject(source=np.asarray(values, np.float32), destination=dst, src_transform=transform, src_crs=crs,
                  src_nodata=np.nan, dst_transform=dst_transform, dst_crs=dst_crs, dst_nodata=np.nan,
                  resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return dst


def _write(path: Path, bands: list[np.ndarray], transform, crs, descriptions: list[str], tags: dict) -> Path:
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = bands[0].shape
    profile = {"driver": "GTiff", "width": w, "height": h, "count": len(bands), "dtype": "float32", "crs": crs,
               "transform": transform, "nodata": float("nan"), "compress": "deflate", "predictor": 3,
               "BIGTIFF": "IF_SAFER"}
    if w >= 256 and h >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    tmp = path.with_suffix(".part.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        for k, (band, text) in enumerate(zip(bands, descriptions), start=1):
            dst.write(band.astype(np.float32), k)
            dst.set_band_description(k, text)
        dst.update_tags(**{k: str(v) for k, v in tags.items()})
    os.replace(tmp, path)
    return path


def to_chunk(chunk, values: np.ndarray, grid: Grid, path: Path, description: str, tags: dict) -> Path | None:
    """``values`` on ``grid`` copied onto the chunk's grid (:func:`chunk_grid`) as ``path``."""
    cg = chunk_grid(chunk, grid)
    if cg is None:
        return None
    out = onto(values, grid.transform, grid.crs, cg["transform"], cg["crs"], (cg["height"], cg["width"]))
    return _write(path, [out], cg["transform"], cg["crs"], [description], tags)


# ------------------------------------------------------------------ products
def products(chunk) -> list[dict]:
    return list(peek(chunk).get("products") or [])


def product(chunk, pid: str) -> dict | None:
    return next((p for p in products(chunk) if p.get("id") == pid), None)


def _slug(text: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in text)
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_")


def _register(chunk, entry: dict) -> dict:
    st = state(chunk)
    st["products"] = [p for p in st.get("products") or [] if p.get("id") != entry["id"]] + [entry]
    return entry


def _product(chunk, kind: str, index: str, period: str, ids: list[str], bands: list[np.ndarray], names: list[str],
             grid: Grid, params: dict, stats: dict, pid: str) -> dict:
    notice = attribution({int(i[:4]) for i in ids})
    tags = {"KIND": kind, "INDEX": index, "PERIOD": period, "ACQUISITIONS": json.dumps(ids), "ATTRIBUTION": notice,
            "TIFFTAG_COPYRIGHT": notice, "TIFFTAG_SOFTWARE": "pyarchinit-photogrammetry sentinel2",
            "PARAMS": json.dumps(params)}
    out = folder(chunk) / "products"
    utm = _write(out / f"{pid}.tif", bands, grid.transform, grid.crs, names, tags)
    on_chunk = to_chunk(chunk, bands[0], grid, out / f"{pid}_chunk.tif", names[0], tags)
    entry = {"id": pid, "kind": kind, "index": index, "period": period, "dates": ids, "file": rel(chunk, utm),
             "chunk_file": rel(chunk, on_chunk) if on_chunk else None, "params": params, "stats": stats,
             "attribution": notice, "made": time.strftime("%Y-%m-%d %H:%M")}
    entry["name"] = product_name(entry, "it")
    return _register(chunk, entry)


def composite(chunk, index: str, period: str, progress=None) -> dict:
    """The median of an index over a period, cell by cell (and how many dates gave a value), as
    a product."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    if index not in analysis.INDICES:
        raise ValueError(f"index: {', '.join(analysis.INDICES)}")
    progress(0.1, t("Composito {0} {1}", "Composite {0} {1}").format(index, period))
    med, count, _tr, _crs, ids = composite_values(chunk, index, period)
    grid = grid_of_chunk(chunk)
    inside = med[area_mask(area_of(chunk), grid)]
    pid = _slug(f"composite_{index}_{period}")
    entry = _product(chunk, "composite", index, period, ids, [med, count.astype(np.float32)],
                     [f"{index} median", "dates with a value"], grid, {}, analysis.stats(inside), pid)
    progress(1.0, t("{0}: {1} date", "{0}: {1} dates").format(entry["name"], len(ids)))
    return entry


def cropmarks(chunk, index: str = "NDVI", period: str = "all", background_m: float = analysis.BACKGROUND_M,
              method: str = "gaussian", progress=None) -> dict:
    """The crop-mark map of an index over a period: the median of the period minus its smooth
    background, divided by the robust spread (a z-score), as a product."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    if index not in analysis.INDICES:
        raise ValueError(f"index: {', '.join(analysis.INDICES)}")
    progress(0.1, t("Crop mark {0} {1}", "Crop marks {0} {1}").format(index, period))
    med, _count, _tr, _crs, ids = composite_values(chunk, index, period)
    grid = grid_of_chunk(chunk)
    progress(0.4, t("Sfondo e scarti", "Background and residuals"))
    z, fit = analysis.anomaly(med, grid.cell, background_m, method)
    inside = z[area_mask(area_of(chunk), grid)]
    st = analysis.stats(inside)
    ok = inside[np.isfinite(inside)]
    if ok.size:
        st["low"] = float((ok < -2).mean())  # stressed: walls, roads, compacted ground
        st["high"] = float((ok > 2).mean())  # greener: ditches, pits
    params = {"background_m": float(background_m), "method": method, "median": fit["median"],
              "spread": fit["spread"]}
    pid = _slug(f"cropmarks_{index}_{period}_{float(background_m):g}m" + ("" if method == "gaussian" else f"_{method}"))
    entry = _product(chunk, "cropmarks", index, period, ids, [z], [f"{index} crop-mark z-score"], grid, params, st, pid)
    progress(1.0, t("{0}: pronta", "{0}: ready").format(entry["name"]))
    return entry


def remove(chunk, pid: str) -> list[str]:
    """A product out of the chunk, its files deleted; the files deleted (relative)."""
    e = product(chunk, pid)
    if e is None:
        return []
    st = state(chunk)
    st["products"] = [p for p in st.get("products") or [] if p.get("id") != pid]
    gone = []
    for key in ("file", "chunk_file"):
        p = absolute(chunk, e.get(key))
        if p is not None and p.exists():
            try:
                p.unlink()
                gone.append(e[key])
            except OSError:
                pass
    return gone


def read_product(chunk, pid: str):
    """``(values, transform, crs)`` of the first band of a product, on the grid of the scenes."""
    import rasterio

    e = product(chunk, pid)
    if e is None:
        raise KeyError(pid)
    with rasterio.open(absolute(chunk, e["file"])) as src:
        return src.read(1).astype(np.float32), src.transform, src.crs


def rgb(chunk, acq_id: str):
    """``(red, green, blue reflectance (3, rows, cols), transform, crs)`` of a cached acquisition."""
    grid = grid_of_chunk(chunk)
    cache = reader.read_cache(folder(chunk) / "scenes" / f"{acq_id}.tif", grid)
    if cache is None:
        raise NoData(t("L'acquisizione {0} non è nella cache.", "The acquisition {0} is not in the cache.").format(acq_id))
    r = cache["reflectance"]
    return np.stack([r["B04"], r["B03"], r["B02"]]), grid.transform, grid.crs


# ------------------------------------------------------------------ exports
def export_date(chunk, index: str, acq_id: str, path) -> list[Path]:
    """One index of one date as a GeoTIFF on the grid of the scenes (UTM), and its copy on the
    chunk's grid beside it (``<name>_chunk.tif``)."""
    values, ids, tr, crs = read_stack(chunk, index, [acq_id])
    if not ids:
        raise NoData(t("Nessun {0} del {1}: la data non è fra quelle serene.", "No {0} of {1}: the date is not among "
                       "the clear ones.").format(index, acq_id))
    notice = attribution([int(acq_id[:4])])
    tags = {"INDEX": index, "FORMULA": analysis.FORMULAS[index], "ACQUISITION": acq_id, "ATTRIBUTION": notice,
            "TIFFTAG_COPYRIGHT": notice, "TIFFTAG_SOFTWARE": "pyarchinit-photogrammetry sentinel2"}
    path = Path(path)
    out = [_write(path, [values[0]], tr, crs, [f"{index} {acq_id}"], tags)]
    on_chunk = to_chunk(chunk, values[0], grid_of_chunk(chunk), path.with_name(path.stem + "_chunk.tif"),
                        f"{index} {acq_id}", tags)
    if on_chunk is not None:
        out.append(on_chunk)
    return out


def export_acquisition(chunk, acq_id: str, path) -> Path:
    """The reflectance of the seven bands of a cached acquisition (float32, NaN where no data) and
    its scene classification as the eighth band, on the grid of the scenes (UTM)."""
    grid = grid_of_chunk(chunk)
    cache = reader.read_cache(cache_dir(chunk) / f"{acq_id}.tif", grid) if grid is not None else None
    if cache is None:
        raise NoData(t("L'acquisizione {0} non è nella cache.", "The acquisition {0} is not in the cache.").format(acq_id))
    notice = attribution([int(acq_id[:4])])
    bands = [cache["reflectance"][b] for b in stac.BANDS] + [cache["scl"].astype(np.float32)]
    names = [f"{b} reflectance" for b in stac.BANDS] + ["SCL scene classification (0 = no data)"]
    tags = {"ACQUISITION": acq_id, "ATTRIBUTION": notice, "TIFFTAG_COPYRIGHT": notice,
            "TIFFTAG_SOFTWARE": "pyarchinit-photogrammetry sentinel2", **{k: v for k, v in cache["tags"].items()
                                                                        if k in ("DATETIME", "PLATFORM", "TILES", "ITEMS")}}
    return _write(Path(path), bands, grid.transform, grid.crs, names, tags)


def export_stack(chunk, path) -> Path:
    """Every index of every clear date in one GeoTIFF (UTM of the area): bands named
    ``NDVI 2023-06-14 20230614_S2B``."""
    import rasterio

    path = Path(path)
    ds = dates(chunk)
    if not ds:
        raise NoData(t("Indici non ancora calcolati: scarica e calcola.", "Indices not computed yet: read and compute."))
    grid = grid_of_chunk(chunk)
    by_id = {d["id"]: d for d in ds}
    notice = attribution_of(chunk)
    bands = []
    for name in analysis.INDICES:
        values, ids, _tr, _crs = read_stack(chunk, name)
        bands += [(f"{name} {by_id[i]['date']} {i}", values[k]) for k, i in enumerate(ids)]
    profile = {"driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(bands), "dtype": "float32",
               "crs": grid.crs, "transform": grid.transform, "nodata": float("nan"), "compress": "deflate",
               "predictor": 3, "BIGTIFF": "IF_SAFER"}
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for k, (text, values) in enumerate(bands, start=1):
            dst.write(values, k)
            dst.set_band_description(k, text)
        dst.update_tags(ATTRIBUTION=notice, TIFFTAG_COPYRIGHT=notice,
                        FORMULAS=json.dumps(analysis.FORMULAS), TIFFTAG_SOFTWARE="pyarchinit-photogrammetry sentinel2")
    return path


def export_series(chunk, path) -> list[Path]:
    """The series per date as ``path`` (CSV), the monthly one beside it (``<name>_monthly.csv``)
    and the chart (``<name>.png``)."""
    path = Path(path)
    s = peek(chunk).get("series") or {}
    if not s:
        raise NoData(t("Serie non ancora calcolata: scarica e calcola.", "Series not computed yet: read and compute."))
    path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for key, target in (("csv", path), ("monthly", path.with_name(path.stem + "_monthly.csv")),
                        ("chart", path.with_suffix(".png"))):
        src = absolute(chunk, s.get(key))
        if src is not None and src.exists() and src.resolve() != target.resolve():
            shutil.copyfile(src, target)
            out.append(target)
    return out


def export_all(chunk, target, progress=None) -> list[Path]:
    """Everything made into a folder: the stack of the indices, one GeoTIFF per index, the series
    (CSV per date and per month, chart), the products (on the grid of the scenes and on the
    chunk's), a note with the attribution."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    out = [export_stack(chunk, target / "sentinel2_stack.tif")]
    progress(0.4, "stack")
    for name in analysis.INDICES:
        out.append(Path(shutil.copyfile(stack_path(chunk, name), target / f"sentinel2_{name}.tif")))
    out += export_series(chunk, target / "sentinel2_series.csv")
    progress(0.7, "series")
    for e in products(chunk):
        for key in ("file", "chunk_file"):
            src = absolute(chunk, e.get(key))
            if src is not None and src.exists():
                out.append(Path(shutil.copyfile(src, target / src.name)))
    notice = attribution_of(chunk)
    lines = [notice, "", "Sentinel-2 L2A, " + ", ".join(f"{n} = {f}" for n, f in analysis.FORMULAS.items()), ""]
    lines += [f"{e['id']}: {product_name(e, 'en')}" for e in products(chunk)]
    (target / "sentinel2_README.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.append(target / "sentinel2_README.txt")
    progress(1.0, f"{len(out)} files")
    return out


def qgis_entries(chunk) -> tuple[list[dict], list[dict]]:
    """The products on the chunk's grid for the QGIS project of the program, with the ramp of the
    vegetation indices (red-yellow-green) as in the pane: the composites between their 10th and 90th
    percentiles, the crop-mark maps between -3 (red: vegetation weaker than around it, over walls and
    roads) and 3 (green: greener, over ditches and pits)."""
    from .texts import product_name as name_of

    rasters = []
    for e in products(chunk):
        p = absolute(chunk, e.get("chunk_file"))
        if p is None or not p.exists():
            continue
        st = e.get("stats") or {}
        if e.get("kind") == "cropmarks":  # the same colours as the pane's default: red low, green high
            entry = {"ramp": "index", "limits": [-Z_LIMIT, Z_LIMIT]}
        else:
            entry = {"ramp": "index", "limits": [float(st.get("p10", -1.0)), float(st.get("p90", 1.0))]}
        rasters.append({"name": f"Sentinel-2: {name_of(e)}", "path": str(p), "band": 1, "opacity": 0.75,
                        "hidden": True, **entry})
    return rasters, []


def summary(chunk) -> dict:
    st = peek(chunk)
    files, size = cache_size(chunk)
    return {"area": st.get("area"), "grid": st.get("grid"), "search": st.get("search"), "settings": settings(chunk),
            "cache": {"files": files, "megabytes": round(size / 1e6, 2), "acquisitions": cached_ids(chunk)},
            "scenes": [{k: r.get(k) for k in ("id", "date", "platform", "tiles", "cloud", "clear", "status")}
                       for r in st.get("scenes") or []],
            "dates": [d["id"] for d in st.get("dates") or []], "products": st.get("products") or [],
            "attribution": attribution_of(chunk) if st.get("dates") else None}
