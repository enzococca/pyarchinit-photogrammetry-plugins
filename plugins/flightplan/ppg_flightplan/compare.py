"""After the flight: the plan compared with the photos that were taken, and the coverage they give compared with
the coverage the plan predicted.

* **Where the photos are** (:func:`photo_positions`): the camera references the program keeps (the GPS of the EXIF
  as the photos were added, a positions file, the PPK), read in their own system (``meta["reference_crs"]``, else
  the chunk's) and brought to WGS84 by the program's own conversion (:func:`~pyarchinit_photogrammetry.markers.gcp.
  camera_references`); a photo without a reference falls back on the GPS of its EXIF. The height above the takeoff
  is the ``drone-dji:RelativeAltitude`` of the photo's XMP, when the drone writes it. The photos are only read,
  never written.
* **Shots and photos paired** (:func:`compare`): every planned shot (:func:`ppg_flightplan.route.shot_points`, leg
  by leg) with the nearest photo within ``max_distance`` (half the step between the shots of its leg by default),
  one photo for one shot, the closest pairs first. A shot without a photo is *missing* (with its leg); a photo far
  from every shot is *extra*; a photo near a shot already taken by another is a *repeat*. Distances are horizontal,
  in metres on a local azimuthal projection centred on the plan (pyproj).
* **The coverage obtained**, when the chunk is aligned and has a projected system: the program's field check
  (:func:`pyarchinit_photogrammetry.engine.fieldcheck.coverage`, aligned photos framing every cell) at the cell of
  the predicted coverage, resampled on the predicted grid. Cells the reconstruction does not reach are not
  measured: counted apart (``outside_reconstruction``), left out of ``cells_worse`` (cells measured, predicted at or
  above the threshold and obtained below it) and of the obtained fractions. The comparison with its coverage
  becomes old when the alignment, the georeference or the DEM change (:func:`coverage_basis`).

The result of :func:`compare` is JSON-ready except the keys beginning with ``_`` (the obtained grid), which
:func:`write` turns into rasters. :func:`compare_for` does the whole step for a plan of the project and keeps the
record in ``chunk.meta["flightplan"]["comparison"][id]``.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .texts import t

STEM = "confronto"
MISSING_TABLE = "scatti_mancanti"
_RELATIVE = re.compile(rb"drone-dji:RelativeAltitude\s*(?:=\s*\"|>)\s*([+-]?[0-9]+(?:\.[0-9]*)?)")
XMP_BYTES = 256 * 1024          # the XMP packet of a drone's JPEG is in its first segments


class CompareError(ValueError):
    """Nothing to compare: no shot in the plan, no photo with a position."""


# ------------------------------------------------------------------ the photos
def relative_height(path) -> Optional[float]:
    """``drone-dji:RelativeAltitude`` (metres above the takeoff) from the XMP of the photo at ``path``: the
    attribute or the element form, read from the head of the file only; None when the file or the value is not
    there."""
    try:
        with open(path, "rb") as f:
            head = f.read(XMP_BYTES)
    except OSError:
        return None
    m = _RELATIVE.search(head)
    if m is None:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def photo_positions(chunk) -> list[dict]:
    """Every enabled photo of the chunk: ``label``, ``key``, ``latlon`` ((lat, lon) or None), ``source``
    (``reference``, ``exif`` or None), ``relative_height`` (from the XMP, or None) and ``aligned``. The references
    are converted one system at a time: a system that cannot be converted sends only its photos to the EXIF GPS
    (``source`` "exif", counted in the statistics as ``from_exif``)."""
    from pyarchinit_photogrammetry.markers import gcp

    cams = [c for c in chunk.cameras if getattr(c, "enabled", True)]
    groups: dict = {}
    for c in cams:
        if c.reference.location:
            groups.setdefault(str(c.meta.get("reference_crs", chunk.crs)), []).append(c)
    refs: dict = {}
    for group in groups.values():   # one system at a time: one that cannot be read leaves the others as they are
        try:
            got, _crs = gcp.camera_references(chunk, group, crs="EPSG:4326")
            refs.update(got)
        except Exception:  # noqa: BLE001 - a reference system that cannot be read: the EXIF below for those
            pass
    out = []
    for cam in cams:
        latlon, source = None, None
        loc = refs.get(cam.key)
        if loc is not None and np.all(np.isfinite(np.asarray(loc[:2], float))):
            latlon, source = (float(loc[1]), float(loc[0])), "reference"
        else:
            gps = _exif_gps(cam.path)
            if gps is not None:
                latlon, source = gps, "exif"
        out.append({"label": cam.label, "key": cam.key, "latlon": latlon, "source": source,
                    "relative_height": relative_height(cam.path), "aligned": bool(cam.aligned)})
    return out


def _exif_gps(path) -> Optional[tuple[float, float]]:
    try:
        from pyarchinit_photogrammetry.io.exif import read_exif

        gps = read_exif(path).get("gps")
    except Exception:  # noqa: BLE001 - missing, unreadable, not an image
        return None
    if not gps:
        return None
    lon, lat = float(gps[0]), float(gps[1])
    return (lat, lon) if (lat, lon) != (0.0, 0.0) else None


def photos_signature(chunk) -> str:
    """What the comparison depends on among the photos: their keys, references and alignment (a photo added,
    moved, aligned or switched off makes the comparison old)."""
    parts = []
    for c in chunk.cameras:
        loc = c.reference.location
        parts.append([c.key, bool(getattr(c, "enabled", True)), bool(c.aligned),
                      None if loc is None else [float(v).hex() for v in loc],     # full floats: degrees too
                      c.meta.get("reference_crs")])
    return hashlib.sha1(json.dumps(parts, separators=(",", ":"), default=str).encode()).hexdigest()


def coverage_basis(chunk, id_: str) -> str:
    """What the coverage obtained depends on besides the photos' references: the chunk's transform, the poses of
    the aligned photos, the tie points (their file), the chunk's DEM and the predicted coverage of the plan ``id_``
    (its stamp, DEM, cell and threshold). A re-alignment, an optimisation, a new georeference or a new DEM changes
    it, and the comparison with its coverage becomes old."""
    from . import project

    def floats(a) -> list:
        return [float(v).hex() for v in np.asarray(a, float).ravel()]

    tp = chunk.product_path("tie_points") if hasattr(chunk, "product_path") else None
    tp_id = None
    if tp is not None and Path(tp).is_file():
        st = Path(tp).stat()
        tp_id = [Path(tp).name, st.st_size, st.st_mtime_ns]
    transform = getattr(chunk, "transform", None)
    pred = project.coverage_record(chunk, id_) or {}
    parts = {"transform": None if transform is None else floats(transform),
             "poses": [[c.key, floats(c.transform)] for c in chunk.cameras if c.transform is not None],
             "tie_points": tp_id, "dem": project._dem_signature(chunk),
             "predicted": [pred.get("stamp"), pred.get("dem"), pred.get("cell"), pred.get("low")]}
    return hashlib.sha1(json.dumps(parts, separators=(",", ":"), default=str).encode()).hexdigest()


# ------------------------------------------------------------------ the shots
def planned_shots(piano: dict) -> list[dict]:
    """The shots of the active blocks, leg by leg in the order they are flown: ``latlon``, ``height`` (above the
    takeoff of the leg), ``leg`` (its number, from 1, as the flight sheet counts them), ``tratta`` (the leg's file
    name), ``block`` (its name), ``n`` (the shot's number in the leg, from 1) and ``step`` (the step between the
    shots of the leg, metres)."""
    from . import plan as pl
    from .route import shot_points

    blocks = piano.get("blocchi") or []
    out = []
    for k, leg in enumerate(pl.plan_legs(piano), 1):
        b = blocks[leg.block_index] if leg.block_index < len(blocks) else {}
        step = float(max(leg.route.effective_step, 0.5))
        for n, (p, h) in enumerate(shot_points(leg.route), 1):
            out.append({"latlon": (float(p[0]), float(p[1])), "height": float(h), "leg": k,
                        "tratta": leg.file_name, "block": b.get("nome") or str(leg.block_index + 1), "n": n,
                        "step": step})
    return out


def _local_xy(latlons, centre) -> np.ndarray:
    """(lat, lon) on an azimuthal equidistant projection centred on ``centre``: metres, (N, 2)."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", f"+proj=aeqd +lat_0={centre[0]} +lon_0={centre[1]} +datum=WGS84 +units=m",
                              always_xy=True)
    a = np.asarray(latlons, float).reshape(-1, 2)
    if not len(a):
        return np.zeros((0, 2))
    x, y = tr.transform(a[:, 1], a[:, 0])
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)


def _pairs(shots_xy: np.ndarray, photos_xy: np.ndarray, reach: np.ndarray) -> list[tuple[int, int, float]]:
    """(shot, photo, distance) one to one, the closest pairs first, each pair within the shot's ``reach``."""
    from scipy.spatial import cKDTree

    if not len(shots_xy) or not len(photos_xy):
        return []
    tree = cKDTree(photos_xy)
    candidates = []
    for i, near in enumerate(tree.query_ball_point(shots_xy, r=reach)):
        for j in near:
            candidates.append((float(np.hypot(*(shots_xy[i] - photos_xy[j]))), i, j))
    candidates.sort()
    used_s, used_p, out = set(), set(), []
    for d, i, j in candidates:
        if i in used_s or j in used_p:
            continue
        used_s.add(i)
        used_p.add(j)
        out.append((i, j, d))
    return out


def _round(v, digits=2):
    return None if v is None else round(float(v), digits)


# ------------------------------------------------------------------ the comparison
def compare(piano: dict, chunk, max_distance: Optional[float] = None, progress: Optional[Callable] = None,
            predicted=None, photos: Optional[list[dict]] = None) -> dict:
    """The plan ``piano`` against the photos of ``chunk`` (see the module docstring).

    ``max_distance`` metres (default: half the step of each shot's leg); ``predicted`` the predicted coverage
    (:class:`ppg_flightplan.coverage.Coverage`) whose grid the obtained coverage is put on (without it, or for a
    chunk not aligned or without a projected system, no coverage); ``photos`` the photos already read
    (:func:`photo_positions`). :class:`CompareError` when the plan has no shot or no photo has a position."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    progress(0.02, t("Le foto del volo", "The photos of the flight"))
    shots = planned_shots(piano)
    if not shots:
        raise CompareError(t("il piano non ha scatti: nessun blocco acceso", "the plan has no shots: no active block"))
    photos = photo_positions(chunk) if photos is None else photos
    placed = [p for p in photos if p["latlon"] is not None]
    if not placed:
        raise CompareError(t("nessuna foto del chunk ha una posizione (riferimento o GPS dell'EXIF): aggiungi le foto "
                             "del volo", "no photo of the chunk has a position (reference or EXIF GPS): add the "
                             "photos of the flight"))
    progress.check()
    centre = tuple(np.mean(np.asarray([s["latlon"] for s in shots], float), axis=0))
    sxy = _local_xy([s["latlon"] for s in shots], centre)
    pxy = _local_xy([p["latlon"] for p in placed], centre)
    if max_distance is not None and not float(max_distance) > 0:
        raise CompareError(t("la distanza massima deve essere positiva", "the maximum distance must be positive"))
    reach = np.array([float(max_distance) if max_distance else s["step"] / 2.0 for s in shots])
    pairs = _pairs(sxy, pxy, reach)
    progress(0.2, t("Scatti e foto appaiati", "Shots and photos paired"))

    matched, offsets, dhs = [], [], []
    taken_shot = set()
    taken_photo = set()
    for i, j, d in sorted(pairs):
        s, p = shots[i], placed[j]
        dh = None if p["relative_height"] is None else p["relative_height"] - s["height"]
        matched.append({"leg": s["leg"], "tratta": s["tratta"], "n": s["n"], "foto": p["label"],
                        "offset_m": round(d, 2), "dh_m": _round(dh), "quota_m": round(s["height"], 1)})
        offsets.append(d)
        if dh is not None:
            dhs.append(dh)
        taken_shot.add(i)
        taken_photo.add(j)
    missing = [{"leg": s["leg"], "tratta": s["tratta"], "block": s["block"], "n": s["n"],
                "lat": round(s["latlon"][0], 8), "lon": round(s["latlon"][1], 8), "quota_m": round(s["height"], 1)}
               for i, s in enumerate(shots) if i not in taken_shot]
    extra, repeats = [], []
    if len(placed) > len(taken_photo):
        from scipy.spatial import cKDTree

        tree = cKDTree(sxy)
        widest = float(reach.max())
        for j, p in enumerate(placed):
            if j in taken_photo:
                continue
            d, i = tree.query(pxy[j])
            row = {"foto": p["label"], "lat": round(p["latlon"][0], 8), "lon": round(p["latlon"][1], 8),
                   "distanza_m": round(float(d), 1), "leg": shots[int(i)]["leg"]}
            # within the reach of any shot (each with its own leg's step), not only of the nearest one: a repeat
            near = tree.query_ball_point(pxy[j], r=widest)
            within = any(np.hypot(*(sxy[k] - pxy[j])) <= reach[k] for k in near)
            (repeats if within else extra).append(row)
    by_leg: dict[int, dict] = {}
    for s in shots:
        r = by_leg.setdefault(s["leg"], {"leg": s["leg"], "tratta": s["tratta"], "planned": 0, "matched": 0,
                                         "missing": 0})
        r["planned"] += 1
    for m in matched:
        by_leg[m["leg"]]["matched"] += 1
    for m in missing:
        by_leg[m["leg"]]["missing"] += 1
    stats = {"planned": len(shots), "taken": len(placed), "matched": len(matched), "missing": len(missing),
             "extra": len(extra), "repeats": len(repeats),
             "without_position": len(photos) - len(placed),
             "from_exif": sum(1 for p in placed if p.get("source") == "exif"),
             "median_offset_m": _round(np.median(offsets)) if offsets else None,
             "p90_offset_m": _round(np.percentile(offsets, 90)) if offsets else None,
             "median_dh_m": _round(np.median(dhs), 3) if dhs else None,
             "with_height": len(dhs)}
    result = {"stats": stats, "matched": matched, "missing": missing, "extra": extra, "repeats": repeats,
              "legs": [by_leg[k] for k in sorted(by_leg)],
              "max_distance_m": None if max_distance is None else float(max_distance)}
    if predicted is not None:
        cov = obtained_coverage(chunk, predicted, progress=progress.sub(0.25, 0.7))
        if cov is not None:
            result["coverage"], result["_raster"] = cov
    progress(1.0, t("Confronto fatto", "Comparison done"))
    return result


def obtained_coverage(chunk, predicted, progress=None):
    """``(stats, raster)`` of the coverage the aligned photos give, on the grid of ``predicted``; None when the
    chunk is not aligned or has no projected system."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    from pyarchinit_photogrammetry.engine import fieldcheck

    from . import project

    if not project.has_crs(chunk) or not getattr(chunk, "aligned_cameras", None):
        return None
    tp = getattr(chunk, "tie_points", None)
    if tp is None or len(tp) < 10:
        return None
    counts, transform, _z = fieldcheck.coverage(chunk, resolution=float(predicted.cell), progress=progress)
    h, w = predicted.counts.shape
    out = np.full((h, w), -1, dtype=np.int32)
    crs = rasterio.crs.CRS.from_user_input(predicted.crs)
    reproject(np.asarray(counts, np.int32), out, src_transform=transform, src_crs=crs, src_nodata=-1,
              dst_transform=predicted.transform, dst_crs=crs, dst_nodata=-1, resampling=Resampling.nearest)
    obtained = np.maximum(out, 0).astype(np.int16)        # out of the reconstruction: no photo there
    inside = predicted.inside
    low = int(predicted.low)
    pred = predicted.counts
    reached = inside & (out >= 0)            # cells the field check measured (inside its reconstruction)
    worse = reached & (pred >= low) & (obtained < low)
    n = int(inside.sum())
    m = int(reached.sum())
    stats = {"cell": round(float(predicted.cell), 4), "low": low, "cells": n, "cells_measured": m,
             "obtained_low_fraction": round(float((obtained[reached] < low).mean()), 4) if m else None,
             "predicted_low_fraction": round(float((pred[inside] < low).mean()), 4) if n else None,
             "obtained_median": round(float(np.median(obtained[reached])), 1) if m else None,
             "predicted_median": round(float(np.median(pred[inside])), 1) if n else None,
             "cells_worse": int(worse.sum()),
             "area_worse_m2": round(float(worse.sum()) * float(predicted.cell) ** 2, 1),
             "outside_reconstruction": int((inside & (out < 0)).sum())}
    raster = {"counts": obtained, "inside": inside, "worse": worse, "transform": predicted.transform,
              "crs": predicted.crs, "low": low}
    return stats, raster


# ------------------------------------------------------------------ files
def public(result: dict) -> dict:
    """The result without its private keys (the grids)."""
    return {k: v for k, v in result.items() if not k.startswith("_")}


def write(result: dict, folder, stem: str, crs: Optional[str] = None) -> dict:
    """``<stem>_confronto.json`` (the whole result), ``<stem>_mancanti.gpkg`` (the missing shots as points, table
    ``scatti_mancanti``, in ``crs`` - the chunk's system - or in WGS84) and, when the coverage was obtained,
    ``<stem>_copertura_ottenuta.tif`` (photos per cell, int16, -1 outside the areas) with its colours
    ``<stem>_copertura_ottenuta_rgba.tif``: their paths by kind (``json``, ``gpkg``, ``count``, ``rgba``)."""
    import shapely.geometry

    from pyarchinit_photogrammetry.io import gpkg

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {"json": folder / f"{stem}_confronto.json", "gpkg": folder / f"{stem}_mancanti.gpkg"}
    paths["json"].write_text(json.dumps(public(result), indent=1, ensure_ascii=False), encoding="utf-8")
    missing = result.get("missing") or []
    target = crs or "EPSG:4326"
    if missing:
        if crs:
            from pyproj import CRS, Transformer

            system = CRS.from_user_input(crs)
            system = system.sub_crs_list[0] if system.is_compound else system
            tr = Transformer.from_crs("EPSG:4326", system, always_xy=True)
            xy = [tr.transform(m["lon"], m["lat"]) for m in missing]
        else:
            xy = [(m["lon"], m["lat"]) for m in missing]
    else:
        xy = []
    features = [(shapely.geometry.Point(float(x), float(y)),
                 {"tratta": m["leg"], "file_tratta": m["tratta"], "blocco": m["block"], "scatto": m["n"],
                  "quota_decollo_m": m["quota_m"]}) for (x, y), m in zip(xy, missing)]
    if paths["gpkg"].exists():
        paths["gpkg"].unlink()              # our own file of a comparison before: the table is written again
    gpkg.write_table(paths["gpkg"], MISSING_TABLE, "POINT",
                     {"tratta": "INTEGER", "file_tratta": "TEXT", "blocco": "TEXT", "scatto": "INTEGER",
                      "quota_decollo_m": "REAL"}, features, target)
    raster = result.get("_raster")
    for kind in ("count", "rgba"):
        old = folder / f"{stem}_copertura_ottenuta{'' if kind == 'count' else '_rgba'}.tif"
        if old.exists():
            old.unlink()
    if raster is not None:
        paths.update(_write_raster(raster, folder, stem))
    return paths


def _write_raster(raster: dict, folder: Path, stem: str) -> dict:
    import rasterio
    import rasterio.enums

    from .coverage import rgba

    counts, inside = raster["counts"], raster["inside"]
    h, w = counts.shape
    base = dict(driver="GTiff", width=w, height=h, crs=raster["crs"], transform=raster["transform"],
                compress="deflate", BIGTIFF="IF_SAFER")
    out = {"count": folder / f"{stem}_copertura_ottenuta.tif", "rgba": folder / f"{stem}_copertura_ottenuta_rgba.tif"}
    with rasterio.open(out["count"], "w", count=1, dtype="int16", nodata=-1, **base) as ds:
        ds.write(np.where(inside, counts, -1).astype(np.int16), 1)
        ds.set_band_description(1, t("foto per cella (ottenute)", "photos per cell (obtained)"))
    with rasterio.open(out["rgba"], "w", count=4, dtype="uint8", photometric="RGB", **base) as ds:
        ds.write(rgba(counts, inside, raster["low"]))
        ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
    return out


# ------------------------------------------------------------------ in the project
def compare_for(chunk, id_: str, max_distance: Optional[float] = None, progress: Optional[Callable] = None) -> dict:
    """The comparison of the plan ``id_`` with the photos of ``chunk``: files in ``flightplan/<id>/`` and the record
    kept in ``chunk.meta["flightplan"]["comparison"][id]`` (only while the plan is still in the meta): ``stamp`` (the
    plan's), ``photos`` (:func:`photos_signature`), ``stats``, ``legs``, ``missing``, ``extra``, ``coverage`` (or
    None), ``files`` (relative to the chunk's folder). The predicted coverage is computed first when the chunk is
    aligned and in a projected system and it is missing or old."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import coverage as cov_mod
    from . import project

    progress = as_progress(progress)
    e = project.entry(chunk, id_)
    if e is None:
        raise KeyError(t("nessun piano «{0}» nel chunk", "no plan '{0}' in the chunk").format(id_))
    stamp = e["stamp"]
    photos = photo_positions(chunk)
    if not planned_shots(e["piano"]):
        raise CompareError(t("il piano non ha scatti: nessun blocco acceso", "the plan has no shots: no active block"))
    if not any(p["latlon"] is not None for p in photos):     # before any coverage is computed for nothing
        raise CompareError(t("nessuna foto del chunk ha una posizione (riferimento o GPS dell'EXIF): aggiungi le foto "
                             "del volo", "no photo of the chunk has a position (reference or EXIF GPS): add the "
                             "photos of the flight"))
    predicted = None
    if project.has_crs(chunk) and getattr(chunk, "aligned_cameras", None):
        try:
            project.coverage_for(chunk, id_, progress=progress.sub(0.0, 0.25))
            predicted = project.read_coverage(chunk, id_)
        except cov_mod.CoverageError:
            predicted = None
    result = compare(e["piano"], chunk, max_distance=max_distance, progress=progress.sub(0.25, 0.7),
                     predicted=predicted, photos=photos)
    crs = None
    if project.has_crs(chunk):
        from pyarchinit_photogrammetry.core import crs as crs_mod

        crs = str(chunk.crs) if crs_mod.epsg_code(chunk.crs) is not None else None
    paths = write(result, project.plan_folder(chunk, id_), STEM, crs=crs)
    rec = {"stamp": stamp, "photos": photos_signature(chunk), "stats": result["stats"], "legs": result["legs"],
           "missing": result["missing"], "extra": result["extra"], "coverage": result.get("coverage"),
           "max_distance_m": result["max_distance_m"],
           "basis": coverage_basis(chunk, id_) if result.get("coverage") else None,
           "files": {k: Path(p).relative_to(chunk.data_dir).as_posix() for k, p in paths.items()}}
    progress(1.0, t("Confronto fatto", "Comparison done"))
    if project.entry(chunk, id_) is None:
        project.restore_files(chunk)        # the plan left the meta meanwhile: no record, its folder aside
        return rec
    project.state(chunk)["comparison"][id_] = rec
    return rec


def comparison_record(chunk, id_: str) -> Optional[dict]:
    """The comparison kept for the plan ``id_`` when it is still current: the plan (stamp) and the photos
    (:func:`photos_signature`) as they were, with a coverage obtained also what it depended on
    (:func:`coverage_basis`), its files there; else None."""
    from . import project

    e = project.entry(chunk, id_)
    rec = (project.peek(chunk).get("comparison") or {}).get(id_)
    if e is None or not rec or rec.get("stamp") != e.get("stamp"):
        return None
    if rec.get("photos") != photos_signature(chunk):
        return None
    if rec.get("coverage") and rec.get("basis") != coverage_basis(chunk, id_):
        return None                         # re-aligned, optimised, georeferenced again, another DEM
    files = rec.get("files") or {}
    if not files or not all((Path(chunk.data_dir) / f).is_file() for f in files.values()):
        return None
    return rec


# ------------------------------------------------------------------ words
def worst_leg(rec: dict) -> Optional[dict]:
    """The leg with the most missing shots (the first one on equal counts), or None when none is missing."""
    legs = [g for g in rec.get("legs") or [] if g.get("missing")]
    return max(legs, key=lambda g: (g["missing"], -g["leg"])) if legs else None


def summary_line(rec: dict) -> str:
    """"612 di 640 scatti fatti; scarto mediano 1,8 m; 28 mancanti nella tratta 3" and what else there is to say
    (heights, photos far from the plan, the coverage obtained)."""
    from .texts import num, plural

    s = rec.get("stats") or {}
    parts = [t("{0} di {1} scatti fatti", "{0} of {1} shots taken").format(s.get("matched", 0), s.get("planned", 0))]
    if s.get("median_offset_m") is not None:
        parts.append(t("scarto mediano {0} m", "median offset {0} m").format(num(s["median_offset_m"], 1)))
    n = int(s.get("missing") or 0)
    worst = worst_leg(rec)
    if not n:
        parts.append(t("nessuno mancante", "none missing"))
    elif worst is not None and worst["missing"] == n:
        parts.append(plural(n, t("{0} mancante", "{0} missing"), t("{0} mancanti", "{0} missing"))
                     + " " + t("nella tratta {0}", "in leg {0}").format(worst["leg"]))
    elif worst is not None:
        parts.append(plural(n, t("{0} mancante", "{0} missing"), t("{0} mancanti", "{0} missing"))
                     + " " + t("({0} nella tratta {1})", "({0} in leg {1})").format(worst["missing"], worst["leg"]))
    if s.get("median_dh_m") is not None:
        parts.append(t("quota mediana {0} m rispetto al piano", "median height {0} m from the plan")
                     .format(_signed(s["median_dh_m"])))
    if s.get("extra"):
        parts.append(plural(s["extra"], t("{0} foto lontana dal piano", "{0} photo far from the plan"),
                            t("{0} foto lontane dal piano", "{0} photos far from the plan")))
    if s.get("repeats"):
        parts.append(plural(s["repeats"], t("{0} foto ripetuta", "{0} repeated photo"),
                            t("{0} foto ripetute", "{0} repeated photos")))
    if s.get("from_exif"):
        parts.append(plural(s["from_exif"], t("{0} foto col GPS dell'EXIF", "{0} photo with the EXIF GPS"),
                            t("{0} foto col GPS dell'EXIF", "{0} photos with the EXIF GPS")))
    if s.get("without_position"):
        parts.append(plural(s["without_position"], t("{0} foto senza posizione", "{0} photo without a position"),
                            t("{0} foto senza posizione", "{0} photos without a position")))
    text = "; ".join(parts) + "."
    c = rec.get("coverage")
    if c and c.get("obtained_low_fraction") is not None:
        text += " " + t("Copertura ottenuta: sotto {0} foto il {1}% dell'area (previsto {2}%); {3} peggio del "
                        "previsto.", "Coverage obtained: under {0} photos on {1}% of the area (predicted {2}%); {3} "
                        "worse than predicted.").format(
            c["low"], num(100 * c["obtained_low_fraction"], 0), num(100 * (c["predicted_low_fraction"] or 0), 0),
            plural(c["cells_worse"], t("{0} cella", "{0} cell"), t("{0} celle", "{0} cells")))
        if c.get("outside_reconstruction"):
            text += " " + plural(c["outside_reconstruction"],
                                 t("{0} cella dell'area è fuori dalla ricostruzione: non misurata.",
                                   "{0} cell of the area is outside the reconstruction: not measured."),
                                 t("{0} celle dell'area sono fuori dalla ricostruzione: non misurate.",
                                   "{0} cells of the area are outside the reconstruction: not measured."))
    return text


def _signed(v: float) -> str:
    from .texts import num

    text = num(abs(v), 2)
    return ("+" if v > 0 else "−" if v < 0 else "") + text


def for_sheet(rec: dict) -> dict:
    """The comparison for the flight sheet's page "Confronto": single values with their names in the language of
    the interface, and the tables of the legs, the missing shots and the photos far from the plan."""
    s = rec.get("stats") or {}
    out: dict = {}
    names = (("planned", ("Scatti previsti", "Shots planned")), ("taken", ("Foto con posizione", "Photos with a "
             "position")), ("matched", ("Scatti fatti", "Shots taken")), ("missing", ("Scatti mancanti", "Shots "
             "missing")), ("extra", ("Foto lontane dal piano", "Photos far from the plan")),
             ("repeats", ("Foto ripetute", "Repeated photos")),
             ("without_position", ("Foto senza posizione", "Photos without a position")),
             ("from_exif", ("Foto col GPS dell'EXIF (nessun riferimento usabile)",
                            "Photos with the EXIF GPS (no usable reference)")),
             ("median_offset_m", ("Scarto mediano (m)", "Median offset (m)")),
             ("p90_offset_m", ("Scarto al 90° percentile (m)", "Offset at the 90th percentile (m)")),
             ("median_dh_m", ("Quota mediana rispetto al piano (m)", "Median height from the plan (m)")))
    for key, pair in names:
        if s.get(key) is not None:
            out[t(*pair)] = s[key]
    if rec.get("max_distance_m"):
        out[t("Distanza massima per appaiare (m)", "Maximum distance to pair (m)")] = rec["max_distance_m"]
    else:
        out[t("Distanza massima per appaiare", "Maximum distance to pair")] = t("metà del passo tra gli scatti",
                                                                                 "half the step between the shots")
    c = rec.get("coverage")
    if c:
        cover = (("low", ("Soglia di copertura (foto)", "Coverage threshold (photos)")),
                 ("predicted_low_fraction", ("Area sotto soglia prevista", "Area under the threshold, predicted")),
                 ("obtained_low_fraction", ("Area sotto soglia ottenuta", "Area under the threshold, obtained")),
                 ("predicted_median", ("Foto per cella previste (mediana)", "Photos per cell predicted (median)")),
                 ("obtained_median", ("Foto per cella ottenute (mediana)", "Photos per cell obtained (median)")),
                 ("cells_worse", ("Celle peggio del previsto", "Cells worse than predicted")),
                 ("outside_reconstruction", ("Celle fuori dalla ricostruzione (non misurate)",
                                             "Cells outside the reconstruction (not measured)")),
                 ("area_worse_m2", ("Area peggio del previsto (m²)", "Area worse than predicted (m²)")),
                 ("cell", ("Cella (m)", "Cell (m)")))
        for key, pair in cover:
            if c.get(key) is not None:
                out[t(*pair)] = c[key]
    leg = t("Tratta", "Leg")
    out[t("Tratte", "Legs")] = [{leg: g["leg"], t("File", "File"): g["tratta"], t("Previsti", "Planned"): g["planned"],
                                 t("Fatti", "Taken"): g["matched"], t("Mancanti", "Missing"): g["missing"]}
                                for g in rec.get("legs") or []]
    if rec.get("missing"):
        out[t("Scatti mancanti", "Missing shots")] = [
            {leg: m["leg"], t("Scatto", "Shot"): m["n"], t("Blocco", "Block"): m["block"],
             t("Quota sul decollo (m)", "Height above the takeoff (m)"): m["quota_m"], "Lat": m["lat"],
             "Lon": m["lon"]} for m in rec["missing"]]
    if rec.get("extra"):
        out[t("Foto lontane dal piano", "Photos far from the plan")] = [
            {t("Foto", "Photo"): x["foto"], t("Distanza dallo scatto più vicino (m)", "Distance from the nearest "
             "shot (m)"): x["distanza_m"], "Lat": x["lat"], "Lon": x["lon"]} for x in rec["extra"]]
    return out


def missing_xy(chunk, rec: dict) -> np.ndarray:
    """The missing shots in the chunk's system, (N, 2)."""
    from . import project

    pts = [(m["lat"], m["lon"]) for m in rec.get("missing") or []]
    return project.to_chunk(chunk, pts) if pts else np.zeros((0, 2))


__all__ = ["CompareError", "compare", "compare_for", "comparison_record", "for_sheet", "obtained_coverage",
           "photo_positions", "planned_shots", "public", "relative_height", "summary_line", "write"]
