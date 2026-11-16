"""Ellipsoidal and orthometric heights: a DEM or a point cloud moved from one height system to
another, as a new version, through terrio's geoid models (EGM2008, EGM96, national geoids).

A drone with GNSS gives heights above the ellipsoid; national and global DEMs (Copernicus,
ALOS, NASADEM) give heights above a geoid - "sea level". The difference is the undulation of the
geoid, tens of metres in many places (on Tell Sultan the Copernicus DEM lay 37.6 m below the
drone DEM). A height system is written as:

* ``ellipsoid``: heights above the ellipsoid of the chunk's datum - WGS 84 for UTM, ETRS89 for
  ETRS89 systems, RDN2008 for RDN2008 - or of WGS 84 when the chunk's system is on an older
  datum (Monte Mario, OSGB36: GNSS heights are never on those);
* a vertical CRS: ``EPSG:3855`` (EGM2008), ``EPSG:5773`` (EGM96), a national one
  (``EPSG:5701`` ODN, ``EPSG:5703`` NAVD88...): terrio uses the geoid model and the
  transformations the EPSG dataset gives for it, as PROJ does;
* a geoid grid file (``.gtx`` or a PROJ GeoTIFF grid ``.tif``), for a national geoid that the
  EPSG dataset or the PROJ CDN do not carry: heights above it.

The grids come from the PROJ CDN (https://cdn.proj.org) when they are missing and the network is
allowed: kept whole in the plugin folder (``<plugin data>/grids``, :func:`environment.grids_dir`)
for later use without the network, or read in place by range requests (``keep=False``). Grids
already on the computer - a folder given, PROJ's own (``PROJ_DATA``, ``projsync``'s user
folder) - are used first. A height that no grid moved (grid missing, area outside the model) is
an error, never a silent copy: :class:`HeightsUnavailable`.

One point no grid covers refuses them all. A cloud across the edge of a grid is the one case where the
others can go on without it: with a terrio that tells the points it could not move (0.9, ``errcheck``)
:func:`convert_cloud` leaves those points out of the new version and says how many - in the log, in
its result and in the parameters of the version; the cloud of before keeps them. A DEM, whose cells
cannot be left out, is refused whole as before, and so is everything with an older terrio and a cloud
of which no point was moved.

The shift is computed on a grid of samples over the area (every 25 m at most, 129 a side) and
interpolated bilinearly: the geoid models are grids of 1-15 minutes of arc, so the
interpolation adds nothing measurable.
"""
from __future__ import annotations

import logging
import os
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.progress import as_progress

from . import cloudfile, environment, transforms
from .texts import t

ELLIPSOID = "ellipsoid"
PRESETS = [(ELLIPSOID, ("Ellissoide (quote GNSS)", "Ellipsoid (GNSS heights)")),
           ("EPSG:3855", ("Geoide EGM2008 (EPSG:3855)", "EGM2008 geoid (EPSG:3855)")),
           ("EPSG:5773", ("Geoide EGM96 (EPSG:5773)", "EGM96 geoid (EPSG:5773)"))]
SHORT = {ELLIPSOID: ("ellissoidiche", "ellipsoidal"), "EPSG:3855": ("EGM2008", "EGM2008"),
         "EPSG:5773": ("EGM96", "EGM96"), "EPSG:5703": ("NAVD88", "NAVD88"), "EPSG:5701": ("ODN", "ODN")}
GRID_SUFFIXES = (".gtx", ".tif", ".tiff")
STEP, MAX_SAMPLES = 25.0, 129  # metres between samples of the shift, samples a side at most
BLOCK = 1_000_000  # points of a cloud asked of terrio at a time, where the samples do not give their shift
_ELLIPSOID_WORDS = ("ellipsoid", "ellipsoidal", "ellissoide", "ellissoidale", "ellissoidiche")


log = logging.getLogger("pyarchinit_photogrammetry.plugins.terrio")  # the plugin's own (its context's ``log``)


class HeightsUnavailable(RuntimeError):
    """No geoid model moved the heights: its grid is missing, or it does not cover the area.
    ``reason``: what terrio said of the grid in its warnings, "" when it said nothing."""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason


def normalise(spec) -> str:
    """``ellipsoid``, ``EPSG:nnnn`` (the vertical part of a compound ``EPSG:4326+3855``), the
    absolute path of a grid file, or "" for no system."""
    text = str(spec or "").strip()
    if not text:
        return ""
    if text.lower() in _ELLIPSOID_WORDS:
        return ELLIPSOID
    path = Path(text).expanduser()
    if path.suffix.lower() in GRID_SUFFIXES:
        if not path.exists():
            raise FileNotFoundError(f"no geoid grid {path}")
        return str(path.resolve())
    if text.isdigit():
        return f"EPSG:{text}"
    if "+" in text:  # a compound CRS: its vertical part
        text = text.split("+", 1)[1]
        if text.isdigit():
            return f"EPSG:{text}"
    if text.upper().startswith("EPSG:"):
        return "EPSG:" + text.split(":", 1)[1].strip()
    raise ValueError(f"height system {spec!r}: ellipsoid, a vertical CRS (EPSG:3855) or a geoid grid file")


def is_grid_file(spec: str) -> bool:
    return bool(spec) and Path(spec).suffix.lower() in GRID_SUFFIXES


def short_label(spec: str) -> str:
    """``ellissoidiche``/``ellipsoidal``, ``EGM2008``, ``EPSG:5214``, ``geoide.gtx``."""
    spec = normalise(spec)
    if spec in SHORT:
        return t(*SHORT[spec])
    return Path(spec).name if is_grid_file(spec) else spec


def label(spec: str) -> str:
    """The name of a height system in the language of the program."""
    spec = normalise(spec)
    for key, names in PRESETS:
        if key == spec:
            return t(*names)
    if is_grid_file(spec):
        return t("Geoide da file ({0})", "Geoid from a file ({0})").format(Path(spec).name)
    return spec


def _geodetic(crs: str) -> dict:
    """The geographic systems of the heights for the chunk system ``crs``: 2D and 3D codes and
    the PROJ string of the datum - the chunk's datum when the EPSG dataset has its 3D CRS, else
    WGS 84 (see the module)."""
    from pyproj import CRS

    found = None
    try:
        c = CRS.from_user_input(crs)
        if c.is_compound:
            c = c.sub_crs_list[0]
        g = c.geodetic_crs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            two, three = g.to_2d().to_epsg(min_confidence=70), g.to_3d().to_epsg(min_confidence=70)
            if two and three:
                found = {"2d": f"EPSG:{two}", "3d": f"EPSG:{three}",
                         "proj4": g.to_2d().to_proj4().replace("+type=crs", "").strip()}
    except Exception:  # an unusual system: WGS 84 below
        found = None
    return found or {"2d": "EPSG:4326", "3d": "EPSG:4979", "proj4": "+proj=longlat +datum=WGS84 +no_defs"}


def _crs3d(spec: str, geo: dict) -> str:
    if spec == ELLIPSOID:
        return geo["3d"]
    if is_grid_file(spec):  # the grid by name: its folder is among the search paths (paths may have spaces)
        return f"{geo['proj4']} +geoidgrids={Path(spec).name} +vunits=m"
    return f"{geo['2d']}+{spec.split(':', 1)[1]}"


def _environment_grid_paths() -> list[Path]:
    """The folders terrio searches by itself (TERRIO_GRID_DIR, PROJ_DATA, PROJ_LIB, PROJ's user
    folder, terrio's download cache), which :func:`grid_access` keeps after its own."""
    out = []
    for var in ("TERRIO_GRID_DIR", "PROJ_DATA", "PROJ_LIB"):
        out += [Path(p) for p in os.environ.get(var, "").split(os.pathsep) if p]
    home = Path.home()
    if sys.platform == "darwin":
        out.append(home / "Library" / "Application Support" / "proj")
    elif sys.platform == "win32":
        out.append(Path(os.environ.get("LOCALAPPDATA", home)) / "proj")
    else:
        out.append(Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share")) / "proj")
    out.append(Path(os.environ.get("TERRIO_GRID_CACHE") or Path(os.environ.get("XDG_CACHE_HOME", home / ".cache"))
                    / "terrio" / "grids"))
    return [p for p in out if p.is_dir()]


@contextmanager
def grid_access(network: bool = True, keep: bool = True, folders=(), cache=None, environment_paths: bool = True):
    """terrio set up for the grids: the ``folders`` first, then the plugin's grid folder
    (``cache``, by default :func:`environment.grids_dir`), then terrio's usual places; missing
    grids from the PROJ CDN when ``network``, downloaded whole into the plugin's folder when
    ``keep`` (else read in place). Everything back as it was afterwards."""
    terrio = environment.load_features(t("La conversione delle quote", "The conversion of heights"))
    cache = Path(cache) if cache is not None else environment.grids_dir()
    paths = [Path(f) for f in folders if f]
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        paths.append(cache)
    if environment_paths:
        paths += _environment_grid_paths()
    seen, ordered = set(), []
    for p in paths:
        if str(p) not in seen:
            seen.add(str(p))
            ordered.append(str(p))
    saved = {k: os.environ.get(k) for k in ("TERRIO_GRID_CACHE", "TERRIO_GRID_DOWNLOAD")}
    if cache is not None:
        os.environ["TERRIO_GRID_CACHE"] = str(cache)
    os.environ["TERRIO_GRID_DOWNLOAD"] = "YES" if keep else "NO"
    terrio.set_grid_search_paths(ordered)
    terrio.set_network_enabled(bool(network))
    try:
        yield terrio
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        terrio.set_grid_search_paths(None)
        terrio.set_network_enabled(None)


def vertical_shift(crs: str, xs, ys, source: str, target: str, network: bool = True, keep: bool = True,
                   folders=(), cache=None, environment_paths: bool = True, partial: bool = False,
                   said: list | None = None) -> np.ndarray:
    """What to add to heights in the system ``source`` at the points ``(xs, ys)`` of ``crs`` to
    have them in ``target`` (for ellipsoid -> geoid, minus the undulation of the geoid).

    A point no model moved refuses them all (:class:`HeightsUnavailable`). ``partial``, with a terrio
    that tells the points it could not move (0.9): NaN at those points and the shift at the others,
    for a caller that leaves them out and counts them; no point moved is still the refusal. ``said``,
    a list: takes the warnings of terrio of a call that left points out, for what the caller says of
    them (they are issued again else, as those of every call that moved all its points)."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    source, target = normalise(source), normalise(target)
    xs, ys = np.asarray(xs, np.float64).ravel(), np.asarray(ys, np.float64).ravel()
    if not source or not target:
        raise ValueError("give the height system of the data and the one wanted")
    if source == target:
        return np.zeros(len(xs))
    if crs_mod.is_local(crs):
        raise RuntimeError("the chunk needs a reference system (align it with GPS or markers first): heights "
                           "above a geoid depend on where the survey is")
    geo = _geodetic(crs)
    folders = list(folders) + [Path(s).parent for s in (source, target) if is_grid_file(s)]
    what = t("Le quote", "The heights")
    z0 = np.zeros(len(xs))
    refused, heard = None, []  # what terrio warned of while it moved the heights: for the refusal, if there is one
    with grid_access(network, keep, folders, cache, environment_paths) as terrio:
        partial = partial and transforms.has_errcheck(terrio)
        lon, lat = transforms.transform_points(terrio, xs, ys, crs, geo["2d"], what=what)[:2]
        try:  # never a ballpark that leaves the heights as they are (terrio 0.6: only_best)
            out = transforms.transform_points(terrio, lon, lat, _crs3d(source, geo), _crs3d(target, geo), zs=z0,
                                              what=what, said=heard, partial=partial)
            # a point terrio did not move has no number in some coordinate: its height is none, whatever came back
            z1 = np.where(transforms.finite(out), np.asarray(out[2], np.float64).ravel(), np.nan)
        except transforms.TransformUnavailable as exc:
            z1, refused = np.full(len(xs), np.nan), exc
    dz = z1 - z0
    missing = ~np.isfinite(dz) | (dz == 0.0)  # and with an older terrio: a height no grid moved
    if missing.any() and (missing.all() or not partial):
        reason, rest = (refused.reason, []) if refused is not None else transforms.words(heard)
        transforms.again(rest)
        where = t("in {0} punti su {1}", "at {0} points of {1}").format(int(missing.sum()), len(dz))
        raise HeightsUnavailable(t(
            "Nessun modello ha spostato le quote da {0} a {1} ({2}): la griglia del geoide manca - attiva la rete "
            "per scaricarla dal CDN di PROJ, o mettila in una cartella delle griglie - oppure non copre il rilievo.",
            "No model moved the heights from {0} to {1} ({2}): the grid of the geoid is missing - allow the network "
            "to download it from the PROJ CDN, or put it in a folder of grids - or it does not cover the survey.")
            .format(label(source), label(target), where) + (f" terrio: {reason}" if reason else ""),
            reason) from refused
    if missing.any() and said is not None:  # points left out: what terrio said meanwhile is for who counts them
        said.extend(heard)
    else:
        transforms.again(heard)
    dz[missing] = np.nan  # with ``partial``: the points to leave out
    return dz


class ShiftField:
    """The shift between two height systems sampled over ``bounds`` (``min_x, min_y, max_x,
    max_y`` in ``crs``) and interpolated bilinearly: :meth:`at` gives it at any point. With
    ``partial`` and ``said`` (see :func:`vertical_shift`) the samples no model moved are NaN, and so
    is the shift :meth:`at` gives beside them: those points are to be asked one by one."""

    def __init__(self, crs, bounds, source, target, step: float = STEP, max_samples: int = MAX_SAMPLES,
                 partial: bool = False, said: list | None = None, **grids):
        x0, y0, x1, y1 = (float(v) for v in bounds)
        if x1 <= x0:
            x1 = x0 + 1.0
        if y1 <= y0:
            y1 = y0 + 1.0
        nx = int(np.clip(np.ceil((x1 - x0) / step) + 1, 2, max_samples))
        ny = int(np.clip(np.ceil((y1 - y0) / step) + 1, 2, max_samples))
        self.x, self.y = np.linspace(x0, x1, nx), np.linspace(y0, y1, ny)
        gx, gy = np.meshgrid(self.x, self.y)
        self.grid = vertical_shift(crs, gx.ravel(), gy.ravel(), source, target, partial=partial, said=said,
                                   **grids).reshape(ny, nx)

    def at(self, xs, ys) -> np.ndarray:
        xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
        fx = np.clip((xs - self.x[0]) / (self.x[-1] - self.x[0]) * (len(self.x) - 1), 0, len(self.x) - 1)
        fy = np.clip((ys - self.y[0]) / (self.y[-1] - self.y[0]) * (len(self.y) - 1), 0, len(self.y) - 1)
        i, j = np.minimum(fx.astype(np.int64), len(self.x) - 2), np.minimum(fy.astype(np.int64), len(self.y) - 2)
        u, v = fx - i, fy - j
        g = self.grid
        return ((1 - u) * (1 - v) * g[j, i] + u * (1 - v) * g[j, i + 1] + (1 - u) * v * g[j + 1, i]
                + u * v * g[j + 1, i + 1])

    def stats(self) -> tuple[float, float, float]:
        return float(self.grid.min()), float(self.grid.mean()), float(self.grid.max())


def _grid_options(network, keep, folders, cache, environment_paths=True) -> dict:
    return {"network": network, "keep": keep, "folders": tuple(folders or ()), "cache": cache,
            "environment_paths": environment_paths}


def _version_name(base: str, target: str) -> str:
    return t("{0} (quote {1})", "{0} ({1} heights)").format(base, short_label(target))[:100]


def _new_version_folder(chunk, product: str) -> tuple[str, Path]:
    vid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = chunk.data_dir / "versions" / product / vid
    folder.mkdir(parents=True)
    return vid, folder


def convert_dem(chunk, source: str, target: str, version: str | None = None, name: str | None = None,
                activate: bool = False, network: bool = True, keep: bool = True, folders=(), cache=None,
                environment_paths: bool = True, progress=None) -> dict:
    """The heights of the active DEM (or of the kept version ``version``) moved from ``source``
    to ``target`` (see the module), written as a new version of the DEM - made the active one
    with ``activate``. Returns the file, the version, its name and the shift (min, mean, max)."""
    import rasterio
    from pyarchinit_photogrammetry.core import versions
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    progress = as_progress(progress)
    source, target = normalise(source), normalise(target)
    if version:
        src = versions.version_file(chunk, "dem", version)
        entry = next(e for e in versions.versions(chunk, "dem") if e["id"] == version)
        base, params = entry["name"], dict(entry.get("params") or {})
    else:
        src = chunk.dem_path
        if src is None or not src.exists():
            raise RuntimeError("build the DEM first")
        base, params = versions.active_info(chunk, "dem")["name"], dict(chunk.meta.get("dem") or {})
    with rasterio.open(src) as s:
        crs = s.crs.to_string() if s.crs is not None else chunk.crs
        bounds, profile, nodata = s.bounds, s.profile.copy(), s.nodata
    progress(0.05, t("Spostamento da {0} a {1}", "Shift from {0} to {1}").format(label(source), label(target)))
    field = ShiftField(crs, (bounds.left, bounds.bottom, bounds.right, bounds.top), source, target,
                       **_grid_options(network, keep, folders, cache, environment_paths))
    vid, folder = _new_version_folder(chunk, "dem")
    out = folder / "dem.tif"
    profile.update(driver="GTiff", dtype="float32", nodata=nodata if nodata is not None else -32767.0,
                   compress="deflate", tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    fill = profile["nodata"]
    try:
        with rasterio.open(src) as s, rasterio.open(out, "w", **profile) as d:
            rows = 512
            for r0 in range(0, s.height, rows):
                h = min(rows, s.height - r0)
                w = Window(0, r0, s.width, h)
                z = s.read(1, window=w).astype(np.float64)
                valid = np.isfinite(z) & ((z != nodata) if nodata is not None and np.isfinite(nodata) else True)
                tr = s.transform
                cols, rws = np.meshgrid(np.arange(s.width) + 0.5, np.arange(r0, r0 + h) + 0.5)
                xs, ys = tr.c + cols * tr.a + rws * tr.b, tr.f + cols * tr.d + rws * tr.e
                dz = field.at(xs, ys)
                outz = np.where(valid, z + dz, fill).astype(np.float32)
                d.write(outz, 1, window=w)
                progress(0.1 + 0.8 * (r0 + h) / s.height, t("Righe {0} di {1}", "Rows {0} of {1}").format(r0 + h, s.height))
            d.build_overviews([2, 4, 8, 16], Resampling.average)
    except BaseException:
        import shutil

        shutil.rmtree(folder, ignore_errors=True)
        raise
    shift = field.stats()
    label_name = name or _version_name(base, target)
    terrio = environment.load_features()
    params.update({"heights": target, "heights_from": source, "from_version": version or "active",
                   "height_shift": [round(v, 4) for v in shift], "terrio": str(terrio.__version__)})
    entry = {"id": vid, "name": label_name, "created": datetime.now().isoformat(timespec="seconds"),
             "params": params, "files": {"dem": str(out.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault("dem", []).append(entry)
    if activate:
        versions.activate(chunk, "dem", vid)
    progress(1.0, label_name)
    return {"path": chunk.dem_path if activate else out, "version": vid, "name": label_name, "shift": shift,
            "source": source, "target": target, "active": bool(activate), "grids": _grids_folder(cache)}


def convert_cloud(chunk, source: str, target: str, name: str | None = None, activate: bool = False,
                  network: bool = True, keep: bool = True, folders=(), cache=None, environment_paths: bool = True,
                  progress=None) -> dict:
    """The heights of the dense cloud moved from ``source`` to ``target`` (see the module), written
    as a new version of the cloud with its classes, colours and the plugin's dimensions - made the
    active one with ``activate``. Returns the file, the version, its name and the shift.

    The points no model moved (the cloud crosses the edge of a grid; terrio 0.9 tells them) are not
    in the new version, with everything that runs along the points cut as they are: ``points`` is
    how many were written, ``not_transformed`` how many were left out (also in the parameters of
    the version when above zero, and in the log) and ``reason`` what terrio said of a grid
    meanwhile. No point moved, or an older terrio: :class:`HeightsUnavailable`, nothing written."""
    from pyarchinit_photogrammetry.core import versions

    from .texts import num

    progress = as_progress(progress)
    source, target = normalise(source), normalise(target)
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError("build the dense cloud first")
    extras = cloudfile.read_extras(chunk.product_path("point_cloud"))
    progress(0.1, t("Spostamento da {0} a {1}", "Shift from {0} to {1}").format(label(source), label(target)))
    world = chunk.to_world(pc.xyz)
    lo, hi = world.min(0), world.max(0)
    bounds, grids = (lo[0], lo[1], hi[0], hi[1]), _grid_options(network, keep, folders, cache, environment_paths)
    total = len(pc)
    heard, why = [], ""  # what terrio said while it left points out: said with their count, issued again else
    try:
        field = ShiftField(chunk.crs, bounds, source, target, partial=True, said=heard, **grids)  # none moved: refused
        progress(0.5, t("{0} punti", "{0} points").format(total))
        dz = field.at(world[:, 0], world[:, 1])
        beside = np.nonzero(~np.isfinite(dz))[0]  # beside a sample no model moved: each asked of terrio
        for start in range(0, len(beside), BLOCK):
            part = beside[start:start + BLOCK]
            try:
                dz[part] = vertical_shift(chunk.crs, world[part, 0], world[part, 1], source, target, partial=True,
                                          said=heard, **grids)
            except HeightsUnavailable as exc:  # none of these was moved: they stay without a shift
                why = why or exc.reason
            progress(0.5 + 0.3 * (start + len(part)) / len(beside),
                     t("Punti al bordo della griglia: {0} di {1}", "Points at the edge of the grid: {0} of {1}")
                     .format(num(start + len(part)), num(len(beside))))
        if not np.isfinite(dz).any():  # no point of the cloud was moved: the refusal of always, as for a DEM here
            field = ShiftField(chunk.crs, bounds, source, target, **grids)
            dz = field.at(world[:, 0], world[:, 1])
        moved = np.isfinite(dz)
        left_out = total - int(moved.sum())
        if left_out:
            said, rest = transforms.words(heard)
            heard[:], why = rest, why or said
    finally:
        transforms.again(heard)
    # the shift of the samples, as before, when they all have one; else that of the points written
    whole = bool(np.isfinite(field.grid).all())
    shift = field.stats() if whole else (float(dz[moved].min()), float(dz[moved].mean()), float(dz[moved].max()))
    if left_out:  # never written without a height in the new system: left out and counted
        log.warning("heights of the dense cloud from %s to %s: %d points of %d not transformed (no grid of the "
                    "geoid covers them), left out of the new version%s", source, target, left_out, total,
                    f" - terrio: {why}" if why else "")
        pc, extras, world, dz = pc.subset(moved), cloudfile.subset(extras, moved), world[moved], dz[moved]
    world[:, 2] += dz
    pc.xyz = chunk.to_internal(world)
    vid, folder = _new_version_folder(chunk, "point_cloud")
    out = cloudfile.save(pc, folder / "point_cloud.ply", extras)
    base = versions.active_info(chunk, "point_cloud")["name"]
    label_name = name or _version_name(base, target)
    params = dict(chunk.meta.get("dense") or {})
    params.pop("not_transformed", None)  # of the conversion the active cloud came from, not of this one
    params.update({"heights": target, "heights_from": source, "height_shift": [round(v, 4) for v in shift],
                   "terrio": str(environment.load_features().__version__)})
    if left_out:
        params["not_transformed"] = left_out
    entry = {"id": vid, "name": label_name, "created": datetime.now().isoformat(timespec="seconds"),
             "params": params, "files": {"point_cloud": str(out.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault("point_cloud", []).append(entry)
    if activate:
        versions.activate(chunk, "point_cloud", vid)
    progress(1.0, label_name)
    return {"path": chunk.product_path("point_cloud") if activate else out, "version": vid, "name": label_name,
            "shift": shift, "source": source, "target": target, "active": bool(activate), "points": len(pc),
            "not_transformed": left_out, "reason": why, "grids": _grids_folder(cache)}


def _grids_folder(cache) -> str:
    folder = Path(cache) if cache is not None else environment.grids_dir()
    return "" if folder is None else str(folder)


def shift_raster(raster, crs: str, source: str, target: str, **grids):
    """A terrio raster (a DEM in ``crs``) with its heights moved from ``source`` to ``target``:
    a new raster in memory, and the shift (min, mean, max)."""
    terrio = environment.load_features()
    z = np.asarray(raster.read(1), np.float64)
    nodata = raster.nodata
    valid = np.isfinite(z) & ((z != nodata) if nodata is not None and np.isfinite(nodata) else True)
    x0, a, b, y0, d, e = raster.transform
    field = ShiftField(crs, raster.bounds, source, target, **grids)
    cols, rows = np.meshgrid(np.arange(raster.width) + 0.5, np.arange(raster.height) + 0.5)
    xs, ys = x0 + cols * a + rows * b, y0 + cols * d + rows * e
    dz = field.at(xs.ravel(), ys.ravel()).reshape(z.shape)
    fill = nodata if nodata is not None else np.nan
    out = np.where(valid, z + dz, fill).astype(np.float32)
    return terrio.Raster.from_array(out, tuple(raster.transform), crs=crs, nodata=fill), field.stats()
