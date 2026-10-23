"""The terrain products of a chunk: rasters made by WhiteboxTools from the DEM, kept in
``<chunk data>/terrain/<id>.tif`` on the grid and in the frame of the DEM, registered in
``chunk.meta["terrain"]["products"]`` with their kind, parameters and statistics; the vectors
(stream network, basins, depressions) in GeoPackages beside them.

WhiteboxTools reads GeoTIFFs with GeoKeys: the DEM is copied once into ``terrain/_work/`` as
float32 with the CRS of the chunk, or a stand-in CRS (EPSG:3857) when the project is local, and
every result comes back onto the DEM's own frame (no CRS again when local).

The names of the products are stored in Italian (they are data, as the program's); the pane and
the QGIS project show :func:`display_name`, in the language of the interface.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import wbt
from .texts import PRODUCTS, lang, t

NODATA = -32767.0
STANDIN_CRS = "EPSG:3857"  # any projected CRS does: the tools only want GeoKeys

METRICS = {
    # key: (tool, the parameters from the size in cells, name it, name en, kind of the values)
    "tpi": ("relative_topographic_position", lambda c: {"filterx": c, "filtery": c},
            "Posizione topografica relativa (TPI)", "Relative topographic position (TPI)", "diverging"),
    "dev": ("dev_from_mean_elev", lambda c: {"filterx": c, "filtery": c},
            "Deviazione dalla quota media", "Deviation from the mean height", "diverging"),
    "geomorphons": ("geomorphons", lambda c: {"search": c, "threshold": 1.0, "fdist": 0, "skip": 0, "forms": True},
                    "Geomorphons (forme del terreno)", "Geomorphons (landforms)", "forms"),
    "ruggedness": ("ruggedness_index", lambda c: {}, "Indice di rugosità (TRI)", "Ruggedness index (TRI)", "grey"),
    "profile_curvature": ("profile_curvature", lambda c: {}, "Curvatura di profilo", "Profile curvature", "diverging"),
    "plan_curvature": ("plan_curvature", lambda c: {}, "Curvatura planare", "Plan curvature", "diverging"),
    # the horizon angles of WhiteboxTools, the course of the sun computed here: see daylight()
    "daylight": ("horizon_angle", None, "Ore di luce (frazione dell'anno)", "Time in daylight (fraction of the year)", "grey"),
}
SIZED = ("tpi", "dev", "geomorphons")  # the metrics with a window or a search distance
GEOMORPHON_FORMS = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur", 6: "slope", 7: "hollow",
                    8: "footslope", 9: "valley", 10: "pit"}
DAYLIGHT = {"az_fraction": 15.0, "max_dist": 100.0, "minutes": 10.0}  # azimuth step (deg), search (m), time step (min)


# ------------------------------------------------------------------ the DEM for the tools
def folder(chunk) -> Path:
    return Path(chunk.data_dir) / "terrain"


def dem_info(chunk) -> dict:
    """Transform, CRS, size and cell of the DEM of the chunk (``RuntimeError`` without one)."""
    import rasterio

    path = chunk.dem_path
    if path is None or not Path(path).exists():
        raise RuntimeError("the chunk has no DEM: build it first")
    with rasterio.open(path) as src:
        return {"path": Path(path), "transform": src.transform, "crs": src.crs, "width": src.width,
                "height": src.height, "cell": float(abs(src.transform.a)), "nodata": src.nodata}


def dem_for_tools(chunk, inverted: bool = False) -> tuple[Path, dict]:
    """The DEM as WhiteboxTools wants it (float32, a CRS, a numeric nodata), written once in
    ``terrain/_work`` (again when the DEM is newer); ``inverted``: ``-z``, for the mounds."""
    import rasterio

    info = dem_info(chunk)
    work = folder(chunk) / "_work"
    work.mkdir(parents=True, exist_ok=True)
    out = work / ("dem_inverted.tif" if inverted else "dem.tif")
    if out.exists() and out.stat().st_mtime >= info["path"].stat().st_mtime:
        return out, info
    with rasterio.open(info["path"]) as src:
        z = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        profile = src.profile.copy()
    if inverted:
        z = -z
    z = np.where(np.isfinite(z), z, NODATA).astype(np.float32)
    crs = info["crs"] if info["crs"] is not None else STANDIN_CRS
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=NODATA, crs=crs, compress="deflate", tiled=False)
    for k in ("blockxsize", "blockysize", "interleave"):
        profile.pop(k, None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(z, 1)
    return out, info


def work_path(chunk, name: str) -> Path:
    work = folder(chunk) / "_work"
    work.mkdir(parents=True, exist_ok=True)
    return work / name


# ------------------------------------------------------------------ the names of the products
def product_id(metric: str, size_m: float = 0.0, cell_m: float | None = None) -> str:
    """The id of a metric: ``tpi_10`` (a window of 10 m), ``ruggedness``; with ``cell_m`` (computed
    on a coarser copy of the DEM) also the cell: ``geomorphons_30_c1``, ``plan_curvature_c0_5``."""
    pid = f"{metric}_{size_m:g}" if metric in SIZED else metric
    if cell_m:
        pid += f"_c{cell_m:g}"
    return pid.replace(".", "_").replace("-", "m")


def name_of(pid: str, params: dict | None = None, language: str = "it") -> str | None:
    """The name of a product in ``language`` (``it``: the name stored in the project), from its id
    and parameters; None for an id the plugin does not know."""
    params = params or {}
    en = language == "en"
    if pid in PRODUCTS:
        return PRODUCTS[pid][1 if en else 0]
    metric = params.get("metric")
    if metric in METRICS:
        _tool, _make, name_it, name_en, _kind = METRICS[metric]
        name = name_en if en else name_it
        if metric in SIZED:
            name += f" {float(params.get('size_m', 0.0)):g} m"
        if params.get("coarse"):
            name += (", cell " if en else ", cella ") + f"{float(params['cell_m']):g} m"
        return name
    if pid.startswith("viewshed") and {"x", "y", "height_m"} <= set(params):
        x, y, h = float(params["x"]), float(params["y"]), float(params["height_m"])
        return (f"Viewshed from ({x:.1f}, {y:.1f}), eye at {h:g} m" if en
                else f"Visibilità da ({x:.1f}, {y:.1f}), occhio a {h:g} m")
    return None


def display_name(entry: dict) -> str:
    """The name of a product as the interface shows it: the stored (Italian) name in Italian,
    the English one in English."""
    stored = entry.get("name") or entry.get("id", "")
    if lang() != "en":
        return stored
    return name_of(str(entry.get("id", "")), entry.get("params"), "en") or stored


# ------------------------------------------------------------------ the products
def read_result(path: Path) -> np.ndarray:
    """A raster written by WhiteboxTools as float64 with NaN where nodata."""
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1, masked=True).astype(np.float64).filled(np.nan)


def slug(name: str) -> str:
    keep = "".join(c if c.isalnum() else "_" for c in name.lower())
    while "__" in keep:
        keep = keep.replace("__", "_")
    return keep.strip("_") or "prodotto"


def store(chunk, values: np.ndarray, pid: str, name: str, kind: str, params: dict | None = None,
          integer: bool = False, vector: str | None = None, classes: dict | None = None,
          description: str = "") -> dict:
    """``values`` (NaN where nothing) as ``terrain/<pid>.tif`` on the DEM's frame, registered in
    the chunk; ``integer``: int32 with 0 as nodata (ids, classes). Returns the entry."""
    import rasterio

    info = dem_info(chunk)
    out = folder(chunk) / f"{pid}.tif"
    out.parent.mkdir(parents=True, exist_ok=True)
    ok = np.isfinite(values)
    if integer:
        arr = np.where(ok, np.nan_to_num(values), 0).astype(np.int32)
        profile = {"dtype": "int32", "nodata": 0}
    else:
        arr = np.where(ok, values, np.nan).astype(np.float32)
        profile = {"dtype": "float32", "nodata": float("nan")}
    profile.update(driver="GTiff", count=1, width=info["width"], height=info["height"], transform=info["transform"],
                   crs=info["crs"], compress="deflate", tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(arr, 1)
        if description:
            dst.set_band_description(1, description)
    v = values[ok]
    stats = {"cells": int(ok.sum())}
    if v.size:
        stats.update({"min": float(v.min()), "max": float(v.max()), "mean": float(v.mean()),
                      "p2": float(np.percentile(v, 2)), "p98": float(np.percentile(v, 98)), "median": float(np.median(v))})
    entry = {"id": pid, "name": name, "kind": kind, "file": str(out.relative_to(chunk.data_dir)), "integer": integer,
             "params": params or {}, "stats": stats, "made": time.strftime("%Y-%m-%d %H:%M")}
    if vector:
        entry["vector"] = vector
    if classes:
        entry["classes"] = {str(k): v for k, v in classes.items()}
    meta = chunk.meta.setdefault("terrain", {})
    products = [e for e in meta.get("products", []) if e.get("id") != pid]
    products.append(entry)
    meta["products"] = products
    return entry


def entries(chunk) -> list[dict]:
    return list((chunk.meta.get("terrain") or {}).get("products", []))


def entry(chunk, pid: str) -> dict | None:
    return next((e for e in entries(chunk) if e.get("id") == pid), None)


def load(chunk, pid: str):
    """``(values with NaN, transform, crs)`` of a product."""
    import rasterio

    e = entry(chunk, pid)
    if e is None:
        raise KeyError(pid)
    with rasterio.open(Path(chunk.data_dir) / e["file"]) as src:
        a = src.read(1, masked=True).astype(np.float64).filled(np.nan)
        return a, src.transform, src.crs


def sharing(chunk, pid: str) -> list[dict]:
    """The other products that use the GeoPackage of ``pid`` (the stream links and the basins share
    ``hydrology.gpkg``, the depth and the levels of the depressions ``depressioni.gpkg``)."""
    e = entry(chunk, pid)
    if e is None or not e.get("vector"):
        return []
    return [x for x in entries(chunk) if x.get("id") != pid and x.get("vector") == e["vector"]]


def remove(chunk, pid: str) -> list[str]:
    """The product out of the chunk: its raster deleted, and the GeoPackage of its lines and
    polygons too when no other product still uses it. Returns the files deleted (relative to the
    chunk's folder)."""
    e = entry(chunk, pid)
    if e is None:
        return []
    meta = chunk.meta.setdefault("terrain", {})
    rest = [x for x in meta.get("products", []) if x.get("id") != pid]
    meta["products"] = rest
    used = {x.get(k) for x in rest for k in ("file", "vector") if x.get(k)}
    gone = []
    for key in ("file", "vector"):
        rel = e.get(key)
        if not rel or rel in used:
            continue
        p = Path(chunk.data_dir) / rel
        if p.exists():
            try:
                p.unlink()
                gone.append(rel)
            except OSError:
                pass
    return gone


def _cells(size_m: float, cell: float, odd: bool = True, least: int = 3) -> int:
    n = max(least, int(round(size_m / max(cell, 1e-9))))
    if odd and n % 2 == 0:
        n += 1
    return n


# ------------------------------------------------------------------ hydrology
def hydrology(chunk, threshold_m2: float = 50.0, breach_m: float = 10.0, progress=None) -> dict:
    """Depressions breached (least cost, up to ``breach_m``), D8 pointer and flow accumulation,
    the stream network of the cells draining at least ``threshold_m2`` (as links) and the basins.
    Products ``flow_accumulation`` (log10 of the cells), ``streams`` (link ids), ``basins`` (ids)
    and the GeoPackage ``terrain/hydrology.gpkg`` with the lines of the network and the polygons
    of the basins."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import vectors

    progress = as_progress(progress)
    dem, info = dem_for_tools(chunk)
    cell = info["cell"]
    w = lambda n: str(work_path(chunk, n))  # noqa: E731
    progress(0.0, t("Apertura delle depressioni", "Breaching the depressions"))
    wbt.run("breach_depressions_least_cost", progress=progress.sub(0.0, 0.2), dem=str(dem), output=w("breached.tif"),
            dist=max(1, int(round(breach_m / cell))), fill=True)
    progress(0.2, t("Direzioni e accumulo di flusso", "Flow directions and accumulation"))
    wbt.run("d8_pointer", progress=progress.sub(0.2, 0.05), dem=w("breached.tif"), output=w("d8.tif"))
    wbt.run("d8_flow_accumulation", progress=progress.sub(0.25, 0.2), input=w("breached.tif"), output=w("facc.tif"),
            out_type="cells")
    threshold = max(1, int(round(threshold_m2 / (cell * cell))))
    progress(0.45, t("Reticolo: celle che drenano almeno {0:g} m²", "Stream network: cells draining {0:g} m² or more")
             .format(threshold_m2))
    wbt.run("extract_streams", progress=progress.sub(0.45, 0.05), flow_accum=w("facc.tif"), output=w("streams.tif"),
            threshold=threshold, zero_background=True)
    wbt.run("stream_link_identifier", progress=progress.sub(0.5, 0.1), d8_pntr=w("d8.tif"), streams=w("streams.tif"),
            output=w("links.tif"), zero_background=True)
    progress(0.6, t("Bacini", "Basins"))
    wbt.run("basins", progress=progress.sub(0.6, 0.15), d8_pntr=w("d8.tif"), output=w("basins.tif"))
    facc = read_result(work_path(chunk, "facc.tif"))
    links = read_result(work_path(chunk, "links.tif"))
    basins = read_result(work_path(chunk, "basins.tif"))
    d8 = read_result(work_path(chunk, "d8.tif"))
    links = np.where(np.isfinite(links) & (links > 0), links, np.nan)
    basins = np.where(np.isfinite(basins) & (basins > 0), basins, np.nan)
    progress(0.75, t("Linee del reticolo e poligoni dei bacini", "Lines of the network and polygons of the basins"))
    lines = vectors.stream_lines(links, d8, facc, info["transform"])
    polys = vectors.basin_polygons(basins, info["transform"])
    gpkg = folder(chunk) / "hydrology.gpkg"
    vectors.write_hydrology(gpkg, lines, polys, info["crs"], cell)
    rel = str(gpkg.relative_to(chunk.data_dir))
    params = {"threshold_m2": float(threshold_m2), "breach_m": float(breach_m), "threshold_cells": threshold}
    store(chunk, np.log10(np.where(facc > 0, facc, np.nan)), "flow_accumulation", name_of("flow_accumulation"),
          "log", params, description="log10 of the cells draining into the cell")
    store(chunk, links, "streams", name_of("streams"), "classes", params, integer=True, vector=rel,
          description="stream link id")
    store(chunk, basins, "basins", name_of("basins"), "classes", params, integer=True, vector=rel,
          description="basin id")
    progress(1.0, t("Idrologia: {0} tratti, {1} bacini", "Hydrology: {0} stream links, {1} basins").format(len(lines), len(polys)))
    return {"links": len(lines), "basins": len(polys), "gpkg": str(gpkg), "threshold_cells": threshold}


# ------------------------------------------------------------------ geomorphometry
def geomorphometry(chunk, metric: str = "tpi", size_m: float = 10.0, progress=None, flatness: float = 1.0,
                   cell_m: float = 0.0) -> dict:
    """One of :data:`METRICS` from the DEM, ``size_m`` the side of the window (TPI, deviation)
    or the search distance (geomorphons; ``flatness`` the angle in degrees under which the ground
    counts as flat). ``cell_m``: the metric computed on the DEM resampled to this cell and brought
    back onto the DEM's grid - the landforms of a field read at a metre, not the furrows read at
    five centimetres; any cell larger than the DEM's is used, a smaller one (or 0) leaves the DEM's
    own cell and the entry says so (``note``). Product ``<metric>_<size>`` (``_c<cell>`` added when
    computed on another cell, so the results of two cells stay side by side)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: " + ", ".join(METRICS))
    tool, make, name_it, _name_en, kind = METRICS[metric]
    dem, info = dem_for_tools(chunk)
    dem_cell = info["cell"]
    asked = float(cell_m or 0.0)
    coarse = asked > dem_cell * (1.0 + 1e-6)
    cell = asked if coarse else dem_cell
    note = ""
    if asked > 0 and not coarse:
        note = t("cella di calcolo di {0:g} m non più grande di quella del DEM ({1:.3g} m): calcolato sulle celle del DEM",
                 "a cell of the computation of {0:g} m is not larger than the DEM's ({1:.3g} m): computed on the "
                 "cells of the DEM").format(asked, dem_cell)
        progress.message(note)
    if coarse:
        dem = coarse_dem(chunk, dem, cell, progress.sub(0.0, 0.05))
    pid = product_id(metric, size_m, cell if coarse else None)
    out = work_path(chunk, f"{pid}.tif")
    progress(0.05, name_it if lang() != "en" else _name_en)
    if metric == "daylight":
        lat, lon = _latlon(info)
        params = {"lat": round(lat, 5), "long": round(lon, 5), **DAYLIGHT}
        daylight(dem, out, lat, progress=progress.sub(0.05, 0.8), az_fraction=DAYLIGHT["az_fraction"],
                 max_dist=DAYLIGHT["max_dist"], minutes=DAYLIGHT["minutes"])
    else:
        n = _cells(size_m, cell)
        params = make(n)
        if metric == "geomorphons":
            params["threshold"] = float(flatness)
        wbt.run(tool, progress=progress.sub(0.05, 0.8), dem=str(dem), output=str(out), **params)
    integer = metric == "geomorphons"
    values = read_result(out) if not coarse else onto_dem(out, info, nearest=integer)
    if integer:
        values = np.where(np.isfinite(values) & (values >= 1) & (values <= 10), values, np.nan)
    params = {"metric": metric, "size_m": float(size_m), "cell_m": float(cell), "dem_cell_m": float(dem_cell),
              "coarse": bool(coarse), **({"cell_asked_m": asked} if asked and not coarse else {}), **params}
    entry_ = store(chunk, values, pid, name_of(pid, params, "it"), kind, params, integer=integer,
                   classes=GEOMORPHON_FORMS if integer else None, description=name_it)
    if note:
        entry_["note"] = note
    progress(1.0, t("{0}: fatto", "{0}: done").format(display_name(entry_)))
    return entry_


def coarse_dem(chunk, dem: Path, cell_m: float, progress=None) -> Path:
    """The DEM for the tools resampled (bilinear) to ``cell_m``: ``terrain/_work/dem_<cell>.tif``."""
    out = work_path(chunk, f"dem_{cell_m:g}m.tif".replace(".", "_").replace("_tif", ".tif"))
    if out.exists() and out.stat().st_mtime >= Path(dem).stat().st_mtime:
        return out
    wbt.run("resample", progress=progress, inputs=str(dem), output=str(out), cell_size=float(cell_m), method="bilinear")
    return out


def onto_dem(path: Path, info: dict, nearest: bool = False) -> np.ndarray:
    """A raster of the tools on another grid brought onto the DEM's grid (the same frame)."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    with rasterio.open(path) as src:
        a = src.read(1, masked=True).astype(np.float32).filled(np.nan)
        crs = src.crs
        tr = src.transform
    dst = np.full((info["height"], info["width"]), np.nan, np.float32)
    reproject(source=a, destination=dst, src_transform=tr, src_crs=crs, dst_transform=info["transform"], dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return dst.astype(np.float64)


def _latlon(info: dict) -> tuple[float, float]:
    """Latitude and longitude of the DEM's centre (a georeferenced chunk only)."""
    if info["crs"] is None:
        raise RuntimeError("the hours of daylight need the latitude: the project has no coordinate system")
    from pyproj import Transformer

    from rasterio.transform import xy

    cx, cy = xy(info["transform"], info["height"] / 2.0, info["width"] / 2.0, offset="center")
    lon, lat = Transformer.from_crs(info["crs"], "EPSG:4326", always_xy=True).transform(cx, cy)
    return float(lat), float(lon)


# ------------------------------------------------------------------ time in daylight
def solar_track(lat: float, minutes: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """Altitude and azimuth of the sun (degrees; azimuth clockwise from north) every ``minutes``
    of solar time through a year (365 days), only while it is above the horizon: the declination
    from NOAA's series of the fractional year, the hour angle from the solar time. Over whole days
    neither the longitude nor the time zone matters; no refraction (the sun is up when its centre
    is above the horizon)."""
    day = np.arange(365, dtype=float)[:, None]
    hours = (np.arange(0.0, 24.0 * 60.0, float(minutes)) / 60.0)[None, :]
    g = 2.0 * np.pi / 365.0 * (day + (hours - 12.0) / 24.0)
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
           + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    h = np.radians(15.0 * (hours - 12.0))
    phi = np.radians(float(lat))
    sin_alt = np.sin(phi) * np.sin(dec) + np.cos(phi) * np.cos(dec) * np.cos(h)
    alt = np.degrees(np.arcsin(np.clip(sin_alt, -1.0, 1.0)))
    az = (np.degrees(np.arctan2(np.sin(h), np.cos(h) * np.sin(phi) - np.tan(dec) * np.cos(phi))) + 180.0) % 360.0
    up = alt > 0
    return alt[up], az[up]


def daylight(dem: Path, out: Path, lat: float, progress=None, az_fraction: float = 15.0, max_dist: float = 100.0,
             minutes: float = 10.0, block_rows: int = 512) -> Path:
    """The share of the daylight time of the year in which each cell of ``dem`` is not in the
    shadow of the relief within ``max_dist``: open, level ground is 1.

    The positions of the sun (:func:`solar_track`, every ``minutes``) are grouped by azimuth in bins
    of ``az_fraction`` degrees; for each bin WhiteboxTools' ``HorizonAngle`` gives the angle of the
    horizon of every cell in that direction, and the sun is hidden when it stands lower. This is
    what WhiteboxTools' own ``TimeInDaylight`` means to do, with the course of the sun computed here:
    version 2.4 of that tool passes the time to its solar almanac in microseconds where milliseconds
    are expected, so its sun positions do not follow the day and open level ground stays well under
    1 (about 0.79 at 40 degrees north)."""
    import rasterio
    from pyarchinit_photogrammetry.core.progress import as_progress
    from rasterio.windows import Window

    progress = as_progress(progress)
    alt, az = solar_track(lat, minutes)
    nbins = max(1, int(round(360.0 / float(az_fraction))))
    step = 360.0 / nbins
    bins = np.rint(az / step).astype(int) % nbins
    used = np.unique(bins)
    total = len(alt)
    with rasterio.open(dem) as src:
        profile = src.profile.copy()
        height, width = src.height, src.width
        valid = src.read_masks(1) > 0
    count = np.uint16 if total < np.iinfo(np.uint16).max else np.uint32  # samples in shadow, per cell
    shadow = np.zeros((height, width), count)
    ha_path = Path(out).with_name(Path(out).stem + "_horizon.tif")
    for k, b in enumerate(used.tolist()):
        p = progress.sub(k / len(used), 1.0 / len(used))
        azimuth = b * step
        # HorizonAngle 2.4 looks south at an azimuth of exactly 0 (or 360): a hair east of north
        wbt.run("horizon_angle", progress=p.sub(0.0, 0.9), dem=str(dem), output=str(ha_path),
                azimuth=f"{azimuth if azimuth % 360 else 0.01:g}", max_dist=float(max_dist))
        sorted_alt = np.sort(alt[bins == b])
        with rasterio.open(ha_path) as src:
            for r0 in range(0, height, block_rows):
                n = min(block_rows, height - r0)
                ha = src.read(1, window=Window(0, r0, width, n), masked=True).astype(np.float64)
                valid[r0:r0 + n] &= ~np.ma.getmaskarray(ha)
                # the positions of this bin lower than the horizon of the cell: in shadow
                shadow[r0:r0 + n] += np.searchsorted(sorted_alt, ha.filled(-np.inf), side="left").astype(count)
        p(1.0, t("Ore di luce: azimut {0:g}°", "Time in daylight: azimuth {0:g}°").format(azimuth))
    try:
        ha_path.unlink()
    except OSError:
        pass
    values = np.where(valid, 1.0 - shadow / max(total, 1), np.nan).astype(np.float32)
    profile.update(driver="GTiff", dtype="float32", count=1, nodata=float("nan"), compress="deflate")
    for key in ("blockxsize", "blockysize", "interleave", "tiled"):
        profile.pop(key, None)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(values, 1)
    return Path(out)


# ------------------------------------------------------------------ viewshed
def viewshed(chunk, x: float, y: float, height_m: float = 1.7, progress=None) -> dict:
    """The cells visible from the point ``(x, y)`` of the DEM's frame with the eye ``height_m``
    above the ground: product ``viewshed_<x>_<y>`` (1 visible, 0 hidden)."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    from . import vectors

    progress = as_progress(progress)
    dem, info = dem_for_tools(chunk)
    station = work_path(chunk, "station.shp")
    vectors.write_point_shapefile(station, [(float(x), float(y))], info["crs"] if info["crs"] is not None else STANDIN_CRS)
    out = work_path(chunk, "viewshed.tif")
    progress(0.0, t("Visibilità", "Viewshed"))
    wbt.run("viewshed", progress=progress.sub(0.05, 0.85), dem=str(dem), stations=str(station), output=str(out),
            height=float(height_m))
    values = read_result(out)
    pid = slug(f"viewshed_{x:.1f}_{y:.1f}")
    visible = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    params = {"x": float(x), "y": float(y), "height_m": float(height_m), "visible_fraction": visible}
    e = store(chunk, np.where(np.isfinite(values), values, np.nan), pid, name_of(pid, params, "it"),
              "binary", params, description="1 visible from the point, 0 hidden")
    progress(1.0, t("Visibilità: visibile il {0:.0f} % del terreno", "Viewshed: {0:.0f} % of the ground visible")
             .format(visible * 100))
    return e


def summary(chunk) -> str:
    return json.dumps({"products": entries(chunk)}, indent=1, ensure_ascii=False)
