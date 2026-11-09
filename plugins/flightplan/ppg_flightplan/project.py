"""The plans of a chunk: kept in ``chunk.meta["flightplan"]``, their files in ``<chunk data>/flightplan/<id>/``.

``chunk.meta["flightplan"]`` = ``{"versione": 1, "plans": [{"id", "name", "piano", "stamp"}], "active": id,
"coverage": {id: {"stamp", "files", "stats", "cell", "low", "surface", "dem"}}, "comparison": {id: {...}}}``. The
program's undo takes snapshots of the plugins' data (1.7.3+): every change made here is followed by
``context.mark_modified()`` in the interface thread, by the caller. A removed plan's files are moved into
``flightplan/.removed/<id>/``; when the undo puts its id back in the meta, :func:`restore_files` brings them back.

Files derived from a plan (the coverage rasters) carry its stamp (:func:`stamp`, the sha1 of its canonical JSON)
and are made again when the plan changed. Exports refresh the terrain heights first
(:func:`ppg_flightplan.design.refresh_terrain`) and refuse a plan whose heights are stale
(:class:`ppg_flightplan.design.StaleHeights`).

The plan's geometry is WGS84 latitude and longitude; the chunk's system appears only at the boundary
(:func:`to_wgs84`, :func:`to_chunk`), with pyproj.
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import re
import shutil
import uuid
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from . import coverage as cov_mod
from . import design
from .geometry import LatLon
from .texts import t

KEY = "flightplan"
FOLDER = "flightplan"
REMOVED = ".removed"
COVERAGE_STEM = "copertura"
PLAN_GPKG = "piano.gpkg"
_ID = re.compile(r"^p[0-9a-f]{8}$")

FORMATS = ("kmz", "kmz-zip", "djifly-zip", "plan", "waypoints", "mavlink", "litchi", "kml", "geojson", "gpkg",
           "area-kml", "sheet-xlsx", "sheet-html", "maps-zip")
SUFFIXES = {"kmz": ".kmz", "kmz-zip": ".zip", "djifly-zip": ".zip", "plan": ".plan", "waypoints": ".waypoints",
            "mavlink": ".mavlink", "litchi": ".csv", "kml": ".kml", "geojson": ".geojson", "gpkg": ".gpkg",
            "area-kml": ".kml", "sheet-xlsx": ".xlsx", "sheet-html": ".html", "maps-zip": ".zip"}
PER_LEG = ("kmz", "plan", "waypoints", "mavlink", "litchi", "area-kml")


# ------------------------------------------------------------------ the meta and the folder
def state(chunk) -> dict:
    """The plugin's part of ``chunk.meta`` (created, and completed, when missing)."""
    st = chunk.meta.setdefault(KEY, {})
    st.setdefault("versione", 1)
    st.setdefault("plans", [])
    st.setdefault("active", None)
    st.setdefault("coverage", {})
    st.setdefault("comparison", {})
    return st


def peek(chunk) -> dict:
    """The plugin's part of ``chunk.meta`` as it is, not created (empty when missing)."""
    return (chunk.meta.get(KEY) or {}) if chunk is not None else {}


def folder(chunk) -> Path:
    return Path(chunk.data_dir) / FOLDER


def plan_folder(chunk, id_: str) -> Path:
    return folder(chunk) / id_


def stamp(piano: dict) -> str:
    """The sha1 of the plan's canonical JSON: what the files made from it carry."""
    text = json.dumps(piano, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def plans(chunk) -> list[dict]:
    return list(peek(chunk).get("plans") or [])


def entry(chunk, id_: str) -> Optional[dict]:
    return next((p for p in peek(chunk).get("plans") or [] if p.get("id") == id_), None)


def _require(chunk, id_: str) -> dict:
    e = entry(chunk, id_)
    if e is None:
        raise KeyError(t("nessun piano «{0}» nel chunk", "no plan '{0}' in the chunk").format(id_))
    return e


def add_plan(chunk, piano: dict, name: Optional[str] = None) -> str:
    """The plan into the chunk, as the active one; its id. A plan that comes in without the height signature (a
    plan of the planner, a file saved before) is signed now (:func:`ppg_flightplan.design.adopt_heights`)."""
    st = state(chunk)
    restore_files(chunk)
    piano = copy.deepcopy(piano)
    design.adopt_heights(piano)
    if name:
        piano["nome"] = name
    taken = {p.get("id") for p in st["plans"]}
    id_ = "p" + uuid.uuid4().hex[:8]
    while id_ in taken:
        id_ = "p" + uuid.uuid4().hex[:8]
    st["plans"].append({"id": id_, "name": piano.get("nome") or name or id_, "piano": piano, "stamp": stamp(piano)})
    st["active"] = id_
    return id_


def active_plan(chunk) -> Optional[dict]:
    """The active plan's entry (``id``, ``name``, ``piano``, ``stamp``), or None."""
    st = peek(chunk)
    found = entry(chunk, st.get("active")) if st.get("active") else None
    return found


def set_active(chunk, id_: str) -> None:
    _require(chunk, id_)
    state(chunk)["active"] = id_


def update_plan(chunk, id_: str, piano: dict) -> str:
    """The plan ``id_`` replaced by ``piano`` (its name follows ``nome``); its new stamp."""
    e = _require(chunk, id_)
    e["piano"] = piano
    e["name"] = piano.get("nome") or e.get("name")
    e["stamp"] = stamp(piano)
    return e["stamp"]


def remove_plan(chunk, id_: str) -> None:
    """The plan out of the meta and its files into ``flightplan/.removed/<id>/`` (the undo can put both back)."""
    st = state(chunk)
    _require(chunk, id_)
    st["plans"] = [p for p in st["plans"] if p.get("id") != id_]
    st["coverage"].pop(id_, None)
    st["comparison"].pop(id_, None)
    if st.get("active") == id_:
        st["active"] = st["plans"][-1]["id"] if st["plans"] else None
    _move_away(chunk, id_)


def _move_away(chunk, id_: str) -> None:
    src = plan_folder(chunk, id_)
    if not src.is_dir():
        return
    dest = folder(chunk) / REMOVED / id_
    if dest.exists():
        shutil.rmtree(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dest)


def restore_files(chunk) -> dict:
    """The plans' folders as the meta says: the files of a plan back in the meta (an undo of its removal) come
    back from ``.removed``; the folder of a plan no longer in the meta (an undo of its addition) goes there."""
    base = folder(chunk)
    known = {p.get("id") for p in peek(chunk).get("plans") or []}
    back, away = [], []
    removed = base / REMOVED
    if removed.is_dir():
        for d in removed.iterdir():
            if d.is_dir() and d.name in known and not (base / d.name).exists():
                d.rename(base / d.name)
                back.append(d.name)
    if base.is_dir():
        for d in base.iterdir():
            if d.is_dir() and _ID.match(d.name) and d.name not in known:
                _move_away(chunk, d.name)
                away.append(d.name)
    return {"restored": back, "removed": away}


def purge_removed(chunk) -> list[str]:
    """The folders in ``flightplan/.removed/`` of plans that are not in the meta, deleted: their ids. To call only
    when no undo can bring those plans back - a project just opened (its meta is the project as saved), or the
    command line after saving. A folder of a plan that is in the meta (an undo put it back and the files have
    not followed yet) is left there."""
    removed = folder(chunk) / REMOVED
    if not removed.is_dir():
        return []
    known = {p.get("id") for p in peek(chunk).get("plans") or []}
    gone = []
    for d in removed.iterdir():
        if d.is_dir() and d.name not in known:
            shutil.rmtree(d, ignore_errors=True)
            gone.append(d.name)
    try:
        removed.rmdir()                     # empty: away too
    except OSError:
        pass
    return gone


# ------------------------------------------------------------------ the chunk's system
def _horizontal(chunk):
    from pyproj import CRS

    crs = getattr(chunk, "crs", None)
    if not crs or str(crs).strip().upper() == "LOCAL":
        raise ValueError(t("serve un sistema di riferimento: il chunk è in un sistema locale",
                           "a reference system is needed: the chunk is in a local frame"))
    system = CRS.from_user_input(crs)
    return system.sub_crs_list[0] if system.is_compound else system


def has_crs(chunk) -> bool:
    try:
        _horizontal(chunk)
        return True
    except Exception:  # noqa: BLE001
        return False


def to_wgs84(chunk, xy) -> list[LatLon]:
    """Points in the chunk's system (x, y) to (lat, lon); ValueError for a local chunk."""
    from pyproj import Transformer

    tr = Transformer.from_crs(_horizontal(chunk), "EPSG:4326", always_xy=True)
    a = np.asarray(xy, dtype=float).reshape(-1, 2)
    lon, lat = tr.transform(a[:, 0], a[:, 1])
    return [(float(la), float(lo)) for la, lo in zip(np.atleast_1d(lat), np.atleast_1d(lon))]


def to_chunk(chunk, latlons) -> np.ndarray:
    """(lat, lon) to the chunk's system, (N, 2); ValueError for a local chunk."""
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", _horizontal(chunk), always_xy=True)
    a = np.asarray(latlons, dtype=float).reshape(-1, 2)
    x, y = tr.transform(a[:, 1], a[:, 0])
    return np.column_stack([np.atleast_1d(x), np.atleast_1d(y)]).astype(float)


# ------------------------------------------------------------------ the ground
def chunk_dem(chunk):
    """The chunk's DEM as a :class:`~ppg_flightplan.dem.RasterDem` (to close), or None."""
    from .dem import chunk_dem as _chunk_dem

    return _chunk_dem(chunk) if chunk is not None else None


def reference_point(piano: dict) -> Optional[LatLon]:
    """The point the heights of the plan refer to: its takeoff, else the first waypoint of the first active block."""
    from . import plan as pl

    point = pl.plan_takeoff(piano)
    if point is not None and pl.takeoff_far(piano) is None:
        return point
    for b in pl.active_blocks(piano):
        p, _ = pl.route_of_block(b, piano)
        if p.points:
            return p.points[0]
    return point


def takeoff_ground(chunk, piano: dict) -> Optional[float]:
    """The ground height at the takeoff: the plan's ``quotaDecollo`` when the terrain was applied (in the system of
    its ``fonteQuote``), else the chunk's DEM at the takeoff (or at the first waypoint without one); None unknown."""
    q = piano.get("quotaDecollo")
    if q is not None:
        return float(q)
    point = reference_point(piano)
    dem = chunk_dem(chunk)
    if dem is None or point is None:
        return None
    try:
        return dem.height(point[0], point[1])
    finally:
        dem.close()


def dem_name(chunk) -> str:
    """The file name of the chunk's DEM (never its folder), or ""."""
    path = getattr(chunk, "dem_path", None) if chunk is not None else None
    return Path(path).name if path else ""


# ------------------------------------------------------------------ the coverage
def _coverage_ground(chunk, piano: dict, dem) -> tuple[Optional[float], Optional[str]]:
    """The takeoff ground in the DEM's height system, for the coverage: the DEM at the reference point; else the
    plan's ``quotaDecollo`` when it is the project DEM's; else None (flat ground)."""
    point = reference_point(piano)
    if dem is not None and point is not None:
        q = dem.height(point[0], point[1])
        if q is not None:
            return q, "dem"
    if dem is not None and piano.get("quotaDecollo") is not None \
            and (piano.get("fonteQuote") or {}).get("sorgente") == "dem":
        return float(piano["quotaDecollo"]), "dem"
    return None, None


def _leg_ground_for(piano: dict, general: Optional[float], dem):
    """The ground under a leg's own takeoff in the same system as ``general``: measured by the terrain, else the
    DEM there; None (the plan's) for a leg flying from the plan's takeoff."""
    from .gis import _leg_ground
    from .plan import plan_takeoff

    def ground(leg):
        if leg.takeoff is None or leg.takeoff == plan_takeoff(piano) or general is None:
            return None
        g = _leg_ground(piano, leg, general)
        if g is None and dem is not None:
            g = dem.height(leg.takeoff[0], leg.takeoff[1])
        return g
    return ground


def _dem_signature(chunk) -> Optional[str]:
    path = getattr(chunk, "dem_path", None)
    if not path or not Path(path).is_file():
        return None
    st = Path(path).stat()
    return f"{Path(path).name}:{st.st_size}:{int(st.st_mtime)}"


def coverage_record(chunk, id_: str) -> Optional[dict]:
    """The coverage stored for the plan ``id_`` when it is still the plan's (same stamp, files there); else None."""
    e = entry(chunk, id_)
    rec = (peek(chunk).get("coverage") or {}).get(id_)
    if e is None or not rec or rec.get("stamp") != e.get("stamp"):
        return None
    if rec.get("dem") != _dem_signature(chunk):
        return None
    files = rec.get("files") or {}
    if not files or not all((Path(chunk.data_dir) / f).is_file() for f in files.values()):
        return None
    return rec


def coverage_for(chunk, id_: str, progress: Optional[Callable] = None, cell: Optional[float] = None,
                 low: int = cov_mod.LOW, force: bool = False) -> dict:
    """The coverage of the plan ``id_`` on the chunk's DEM (``flightplan/<id>/copertura_*.tif``), made again when
    the plan, the DEM, the cell or the threshold changed: the record kept in the meta (``stamp``, ``files``
    relative to the chunk's folder, ``stats``, ``cell``, ``low``, ``surface``). :class:`CoverageError` for a chunk
    without a projected system."""
    e = _require(chunk, id_)
    try:
        crs = _horizontal(chunk)
    except ValueError:
        raise cov_mod.CoverageError(t("serve un sistema di riferimento proiettato: il chunk è in un sistema locale",
                                      "a projected reference system is needed: the chunk is in a local frame")) from None
    rec = coverage_record(chunk, id_)
    if rec is not None and not force and int(rec.get("low", cov_mod.LOW)) == int(low) \
            and (cell is None or abs(float(rec.get("cell", 0)) - float(cell)) < 1e-9):
        return rec
    restore_files(chunk)
    piano = e["piano"]
    dem = chunk_dem(chunk)
    try:
        ground, _ = _coverage_ground(chunk, piano, dem)
        cov = cov_mod.predict(piano, crs.to_wkt(), dem=dem if ground is not None else None, ground_takeoff=ground,
                              cell=cell, progress=progress, low=low,
                              leg_ground=_leg_ground_for(piano, ground, dem))
    finally:
        if dem is not None:
            dem.close()
    out = plan_folder(chunk, id_)
    paths = cov_mod.write(cov, out, COVERAGE_STEM)
    rec = {"stamp": e["stamp"], "files": {k: Path(p).relative_to(chunk.data_dir).as_posix() for k, p in paths.items()},
           "stats": cov.stats, "cell": cov.cell, "low": cov.low, "surface": cov.surface, "shots": cov.shots,
           "dem": _dem_signature(chunk)}
    if entry(chunk, id_) is None:
        # the plan left the meta while the coverage was being computed (an undo of its addition): no record for
        # a plan that is not there; its folder goes aside with the others
        restore_files(chunk)
        return rec
    state(chunk)["coverage"][id_] = rec
    return rec


def read_coverage(chunk, id_: str) -> Optional[cov_mod.Coverage]:
    """The stored coverage of the plan read back from its rasters (for the low-coverage polygons), or None."""
    import rasterio

    rec = coverage_record(chunk, id_)
    if rec is None:
        return None
    base = Path(chunk.data_dir)
    with rasterio.open(base / rec["files"]["count"]) as ds:
        counts = ds.read(1)
        transform, crs = ds.transform, ds.crs.to_wkt()
    with rasterio.open(base / rec["files"]["gsd"]) as ds:
        gsd = ds.read(1)
    inside = counts >= 0
    return cov_mod.Coverage(np.maximum(counts, 0).astype(np.int16), gsd, transform, crs, inside,
                            rec.get("surface", "flat"), dict(rec.get("stats") or {}), float(rec.get("cell", 0)),
                            int(rec.get("low", cov_mod.LOW)), int(rec.get("shots", 0)))


# ------------------------------------------------------------------ export
def prepare(piano: dict, dem=None) -> tuple[dict, dict]:
    """A copy of the plan with its terrain heights refreshed (:func:`ppg_flightplan.design.refresh_terrain`) and the
    outcome; :class:`ppg_flightplan.design.StaleHeights` (with ``reasons``) when heights stay stale, or when the
    ground of a moved takeoff could not be measured again (no network, the takeoff off the DEM): never a file with
    heights, or descents, counted from the old takeoff."""
    piano = copy.deepcopy(piano)
    outcome = design.refresh_terrain(piano, dem)
    unsourced = next((f["motivo"] for f in outcome.get("falliti") or [] if f.get("senzaFonte")), None)
    stale = [i for i in design.blocks_with_old_heights(piano) if 0 <= i < len(piano.get("blocchi", []))]
    if unsourced and not stale:
        # nowhere to read the ground without asking the network unasked: say what to do, nothing else
        e = design.StaleHeights(piano, [], text=unsourced[:1].upper() + unsourced[1:])
        e.reasons = [unsourced]
        raise e
    try:
        design.require_exportable(piano)
        # a block that flies and could not be redone is refused also when its signature happens to match
        from . import plan as pl

        blocks = piano.get("blocchi", [])
        failed = [f["blocco"] for f in outcome.get("falliti") or []
                  if f.get("blocco") is not None and 0 <= f["blocco"] < len(blocks) and pl.active(blocks[f["blocco"]])]
        if failed:
            raise design.StaleHeights(piano, sorted(set(failed)))
        if any(f.get("blocco") is None for f in outcome.get("falliti") or []):
            raise design.StaleHeights(piano, [], takeoff=t("il suolo dei punti speciali non si è potuto misurare",
                                                           "the ground of the special points could not be measured"))
    except design.StaleHeights as e:
        e.reasons = [f.get("motivo") for f in outcome.get("falliti") or [] if f.get("motivo")]
        if e.reasons:
            e.args = (f"{e} ({'; '.join(e.reasons)})",)
        raise
    return piano, outcome


class Written(list):
    """The files written; ``replaced`` the per-leg files next to the chosen path that were already there (an
    earlier export with the same names) and were replaced: the caller says so."""

    def __init__(self, files=(), replaced=()):
        super().__init__(files)
        self.replaced = list(replaced)


def _write_many(files: list[tuple[str, bytes]], path: Path, readme: Optional[tuple[str, str]] = None) -> list[Path]:
    """Files next to ``path`` (one alone: as ``path``), or all in the zip ``path`` when it ends in ``.zip``. The
    per-leg files that replaced files already there are listed in the result's ``replaced``
    (:class:`Written`): the user chose only ``path``, the others are said."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".zip":
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
            for name, data in files:
                z.writestr(name, data)
            if readme:
                z.writestr(*readme)
        path.write_bytes(buffer.getvalue())
        return [path]
    if len(files) == 1:
        path.write_bytes(files[0][1])
        return [path]
    out, replaced = [], []
    for name, data in files:
        target = path.parent / name
        if target.exists():
            replaced.append(target)
        target.write_bytes(data)
        out.append(target)
    return Written(out, replaced)


def export_plan(piano: dict, fmt: str, path, *, crs: Optional[str] = None, ground_takeoff: Optional[float] = None,
                dem_file: str = "", background=None, coverage: Optional[dict] = None,
                comparison: Optional[dict] = None, missions: Optional[list[str]] = None) -> list[Path]:
    """``piano`` (already refreshed: :func:`prepare`) in the format ``fmt`` (:data:`FORMATS`) at ``path``: the files
    written. ``ground_takeoff`` and ``dem_file`` (its name only) are the project DEM's height at the takeoff, for a
    plan without terrain following; ``crs`` the chunk's system (the GeoPackage); ``background`` the maps' background
    (the chunk's orthophoto); ``coverage``/``comparison`` the sheet's sections; ``missions`` the DJI Fly
    containers (``djifly-zip``)."""
    from . import djifly, export_dji, export_formats, gis, legmap, sheet
    from . import plan as pl

    if fmt not in FORMATS:
        raise ValueError(t("formato sconosciuto: {0} (sono {1})", "unknown format: {0} (they are {1})")
                         .format(fmt, ", ".join(FORMATS)))
    design.require_exportable(piano)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = piano.get("quotaDecollo") is None
    ground = float(ground_takeoff) if flat and ground_takeoff is not None else None
    source = ({"sorgente": "dem", "file": gis.file_name(dem_file)} if dem_file else {"sorgente": "dem"}) \
        if ground is not None else None
    if fmt == "kmz":
        return _write_many(export_dji.kmz_per_leg(piano), path)
    if fmt == "kmz-zip":
        path.write_bytes(export_dji.zip_kmz(piano, ground_takeoff=ground, height_source=source))
        return [path]
    if fmt == "djifly-zip":
        if not missions:
            raise ValueError(t("servono le missioni di DJI Fly (una per tratta) in cui mettere le tratte",
                               "the DJI Fly missions (one per leg) to put the legs into are needed"))
        legs = export_dji.kmz_per_leg(piano)
        previews = [legmap.dji_preview(piano, k, background=background) for k in range(len(legs))]
        path.write_bytes(djifly.zip_containers(legs, list(missions), previews))
        return [path]
    if fmt in ("plan", "waypoints", "mavlink", "litchi"):
        if path.suffix.lower() == ".zip":         # with the README of the format
            path.write_bytes(export_formats.zip_format(piano, fmt))
            return [path]
        return _write_many(export_formats.files_per_leg(piano, fmt), path)
    if fmt == "kml":
        path.write_text(gis.plan_kml(piano, ground_takeoff=ground, height_source=source), encoding="utf-8")
        return [path]
    if fmt == "geojson":
        cam = pl.plan_camera(piano)
        path.write_text(gis.geojson(pl.plan_routes(piano), cam, bool(piano.get("alta", False)),
                                    pl.active_areas(piano)), encoding="utf-8")
        return [path]
    if fmt == "gpkg":
        if not crs:
            raise ValueError(t("serve un sistema di riferimento: un chunk locale non ha coordinate per il GeoPackage",
                               "a reference system is needed: a local chunk has no coordinates for the GeoPackage"))
        gis.write_gpkg(piano, path, crs, ground_takeoff=ground, dem_name=dem_file if ground is not None else "")
        return [path]
    if fmt == "area-kml":
        return _write_many([(n, s.encode("utf-8")) for n, s in gis.areas_kml(piano)], path)
    if fmt == "sheet-xlsx":
        path.write_bytes(sheet.sheet_xlsx(piano, background=background, coverage=coverage, comparison=comparison))
        return [path]
    if fmt == "sheet-html":
        path.write_text(sheet.sheet_html(piano, background=background, coverage=coverage, comparison=comparison),
                        encoding="utf-8")
        return [path]
    # maps-zip
    if path.suffix.lower() == ".zip":
        path.write_bytes(legmap.zip_images(piano, background=background))
        return [path]
    return _write_many(legmap.leg_images(piano, background=background), path)


def export(chunk, id_: str, fmt: str, path, progress: Optional[Callable] = None,
           missions: Optional[list[str]] = None) -> list[Path]:
    """The plan ``id_`` in the format ``fmt`` (:data:`FORMATS`) at ``path`` (per-leg formats: one file per leg next
    to ``path``, or a zip when ``path`` ends in ``.zip``): the files written.

    The terrain heights are refreshed first (on the chunk's DEM): when they were redone the plan in the meta is
    updated (the caller marks the project modified); :class:`ppg_flightplan.design.StaleHeights` when they cannot
    be. A plan without terrain following gets the DEM's height at the takeoff and the DEM's file name."""
    from .legmap import orthophoto_background

    from .compare import comparison_record, for_sheet

    e = _require(chunk, id_)
    # the comparison of this plan with these photos, taken before the terrain refresh may change the plan's stamp
    current = comparison_record(chunk, id_)
    comparison = for_sheet(current) if current is not None else None
    if progress is not None:
        progress(0.05, t("Quote del terreno", "Terrain heights"))
    dem = chunk_dem(chunk)
    try:
        piano, outcome = prepare(e["piano"], dem)
    finally:
        if dem is not None:
            dem.close()
    if outcome.get("rifatti") or outcome.get("suolo"):
        update_plan(chunk, id_, piano)
    crs = None
    if has_crs(chunk):
        from pyarchinit_photogrammetry.core import crs as crs_mod

        crs = str(chunk.crs) if crs_mod.epsg_code(chunk.crs) is not None else None
    ground = takeoff_ground(chunk, piano) if piano.get("quotaDecollo") is None else None
    rec = coverage_record(chunk, id_)
    if progress is not None:
        progress(0.3, t("Scrittura", "Writing"))
    background = orthophoto_background(chunk) if fmt in ("djifly-zip", "sheet-xlsx", "sheet-html", "maps-zip") \
        else None
    files = export_plan(piano, fmt, path, crs=crs, ground_takeoff=ground, dem_file=dem_name(chunk) if ground is not None
                        else "", background=background, coverage=rec, comparison=comparison, missions=missions)
    if progress is not None:
        progress(1.0, t("Esportato", "Exported"))
    return files


def plan_gpkg(chunk, id_: str) -> Optional[Path]:
    """``flightplan/<id>/piano.gpkg`` written again when the plan changed (its stamp in ``piano.gpkg.stamp``);
    None when it cannot be (a local chunk, stale heights)."""
    e = entry(chunk, id_)
    if e is None or not has_crs(chunk):
        return None
    out = plan_folder(chunk, id_) / PLAN_GPKG
    mark = out.with_name(PLAN_GPKG + ".stamp")
    if out.is_file() and mark.is_file() and mark.read_text().strip() == e["stamp"]:
        return out
    try:
        piano = copy.deepcopy(e["piano"])
        design.require_current_heights(piano)
        from pyarchinit_photogrammetry.core import crs as crs_mod

        if crs_mod.epsg_code(chunk.crs) is None:
            return None
        ground = takeoff_ground(chunk, piano) if piano.get("quotaDecollo") is None else None
        out.parent.mkdir(parents=True, exist_ok=True)
        from .gis import write_gpkg

        write_gpkg(piano, out, str(chunk.crs), ground_takeoff=ground,
                   dem_name=dem_name(chunk) if ground is not None else "")
    except (design.StaleHeights, ValueError, OSError):
        return None
    mark.write_text(e["stamp"])
    return out


def _gpkg_styles() -> tuple:
    return (("ppg_area", "POLYGON", "#1e88e5", t("area", "area")),
            ("ppg_strisciate", "LINESTRING", "#e53935", t("strisciate", "strips")),
            ("ppg_scatti", "POINT", "#fb8c00", t("scatti", "shots")),
            ("ppg_decolli", "POINT", "#43a047", t("decollo", "takeoff")),
            ("ppg_target", "POINT", "#8e24aa", t("target", "targets")))


def qgis_entries(chunk) -> tuple[list[dict], list[dict]]:
    """For the program's QGIS project: the coverage of every plan (RGBA, current only) and the layers of the active
    plan from its GeoPackage; for the active plan compared with its flight, the coverage obtained and the missing
    shots."""
    rasters, vectors = [], []
    if chunk is None:
        return rasters, vectors
    active = peek(chunk).get("active")
    for e in plans(chunk):
        rec = coverage_record(chunk, e["id"])
        if rec is not None:
            rasters.append({"name": t("Piano di volo: copertura {0}", "Flight plan: coverage {0}").format(e["name"]),
                            "path": str(Path(chunk.data_dir) / rec["files"]["rgba"]), "opacity": 0.7,
                            "hidden": e["id"] != active})
    if active:
        from .compare import MISSING_TABLE, comparison_record

        done = comparison_record(chunk, active)
        files = (done or {}).get("files") or {}
        if files.get("rgba"):
            rasters.append({"name": t("Piano di volo: copertura ottenuta", "Flight plan: coverage obtained"),
                            "path": str(Path(chunk.data_dir) / files["rgba"]), "opacity": 0.7, "hidden": True})
        if files.get("gpkg") and done.get("missing"):
            vectors.append({"name": t("Piano di volo: scatti mancanti", "Flight plan: missing shots"),
                            "path": str(Path(chunk.data_dir) / files["gpkg"]), "table": MISSING_TABLE,
                            "geometry": "POINT", "color": "#d32f2f"})
        gp = plan_gpkg(chunk, active)
        if gp is not None:
            for table, geometry, colour, label in _gpkg_styles():
                vectors.append({"name": t("Piano di volo: {0}", "Flight plan: {0}").format(label), "path": str(gp),
                                "table": table, "geometry": geometry, "color": colour})
    return rasters, vectors


def raster_entries(chunk) -> list[dict]:
    """The coverage rasters for the raster calculator (``add_rasters``): photos per cell and GSD of every plan."""
    out = []
    if chunk is None:
        return out
    for e in plans(chunk):
        rec = coverage_record(chunk, e["id"])
        if rec is None:
            continue
        for kind, label in (("count", t("foto per cella", "photos per cell")), ("gsd", "GSD cm/px")):
            out.append({"name": f"flightplan_{e['id']}_{kind}", "label": f"{e['name']}: {label}",
                        "path": str(Path(chunk.data_dir) / rec["files"][kind])})
    return out


# ------------------------------------------------------------------ a plan without a project
def standalone_load(path) -> dict:
    """A plan from its JSON file (the plan itself, or ``{"piano": ...}``), signed when it comes in without the
    height signature."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    piano = data.get("piano") if isinstance(data, dict) and isinstance(data.get("piano"), dict) else data
    if not isinstance(piano, dict) or not isinstance(piano.get("blocchi"), list):
        raise ValueError(t("{0}: non è un piano di volo", "{0}: not a flight plan").format(Path(path).name))
    design.adopt_heights(piano)
    return piano


def standalone_save(path, piano: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(piano, ensure_ascii=False, indent=1), encoding="utf-8")
    return path


def summary(chunk) -> dict:
    """What the chunk holds: the plans with their numbers, the active one, the coverage made."""
    from . import plan as pl

    st = peek(chunk)
    out = []
    for e in st.get("plans") or []:
        tot = pl.compute(e["piano"])["totale"]
        rec = coverage_record(chunk, e["id"])
        out.append({"id": e["id"], "nome": e["name"], "attivo": e["id"] == st.get("active"), "scatti": tot["scatti"],
                    "minuti": tot["minuti"], "batterie": tot["batterie"], "tratte": tot["tratte"],
                    "copertura": rec.get("stats") if rec else None})
    return {"piani": out, "attivo": st.get("active")}
