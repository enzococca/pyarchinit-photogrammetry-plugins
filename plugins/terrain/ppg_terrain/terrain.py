"""The terrain products of a chunk: rasters made by WhiteboxTools from the DEM, kept in
``<chunk data>/terrain/<id>.tif`` on the grid and in the frame of the DEM, registered in
``chunk.meta["terrain"]["products"]`` with their kind, parameters and statistics; the vectors
(stream network, basins, depressions) in GeoPackages beside them.

WhiteboxTools reads GeoTIFFs with GeoKeys: the DEM is copied once into ``terrain/_work/`` as
float32 with the CRS of the chunk, or a stand-in CRS (EPSG:3857) when the project is local, and
every result comes back onto the DEM's own frame (no CRS again when local).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import wbt

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
    "daylight": ("time_in_daylight", None, "Ore di luce (frazione dell'anno)", "Time in daylight (fraction of the year)", "grey"),
}
SIZED = ("tpi", "dev", "geomorphons")  # the metrics with a window or a search distance
GEOMORPHON_FORMS = {1: "flat", 2: "peak", 3: "ridge", 4: "shoulder", 5: "spur", 6: "slope", 7: "hollow",
                    8: "footslope", 9: "valley", 10: "pit"}


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


def remove(chunk, pid: str) -> None:
    e = entry(chunk, pid)
    if e is None:
        return
    for key in ("file", "vector"):
        p = Path(chunk.data_dir) / e[key] if e.get(key) else None
        if p is not None and p.exists():
            try:
                p.unlink()
            except OSError:
                pass
    meta = chunk.meta.setdefault("terrain", {})
    meta["products"] = [x for x in meta.get("products", []) if x.get("id") != pid]


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
    progress(0.0, "Breaching the depressions")
    wbt.run("breach_depressions_least_cost", dem=str(dem), output=w("breached.tif"),
            dist=max(1, int(round(breach_m / cell))), fill=True)
    progress(0.2, "Flow directions and accumulation")
    wbt.run("d8_pointer", dem=w("breached.tif"), output=w("d8.tif"))
    wbt.run("d8_flow_accumulation", input=w("breached.tif"), output=w("facc.tif"), out_type="cells")
    threshold = max(1, int(round(threshold_m2 / (cell * cell))))
    progress(0.45, f"Stream network: cells draining {threshold_m2:g} m² or more")
    wbt.run("extract_streams", flow_accum=w("facc.tif"), output=w("streams.tif"), threshold=threshold, zero_background=True)
    wbt.run("stream_link_identifier", d8_pntr=w("d8.tif"), streams=w("streams.tif"), output=w("links.tif"), zero_background=True)
    progress(0.6, "Basins")
    wbt.run("basins", d8_pntr=w("d8.tif"), output=w("basins.tif"))
    facc = read_result(work_path(chunk, "facc.tif"))
    links = read_result(work_path(chunk, "links.tif"))
    basins = read_result(work_path(chunk, "basins.tif"))
    d8 = read_result(work_path(chunk, "d8.tif"))
    links = np.where(np.isfinite(links) & (links > 0), links, np.nan)
    basins = np.where(np.isfinite(basins) & (basins > 0), basins, np.nan)
    progress(0.75, "Lines of the network and polygons of the basins")
    lines = vectors.stream_lines(links, d8, facc, info["transform"])
    polys = vectors.basin_polygons(basins, info["transform"])
    gpkg = folder(chunk) / "hydrology.gpkg"
    vectors.write_hydrology(gpkg, lines, polys, info["crs"], cell)
    rel = str(gpkg.relative_to(chunk.data_dir))
    params = {"threshold_m2": float(threshold_m2), "breach_m": float(breach_m), "threshold_cells": threshold}
    store(chunk, np.log10(np.where(facc > 0, facc, np.nan)), "flow_accumulation", "Accumulo di flusso (log10 celle)",
          "log", params, description="log10 of the cells draining into the cell")
    store(chunk, links, "streams", "Reticolo (tratti)", "classes", params, integer=True, vector=rel,
          description="stream link id")
    store(chunk, basins, "basins", "Bacini", "classes", params, integer=True, vector=rel, description="basin id")
    progress(1.0, f"Hydrology: {len(lines)} stream links, {len(polys)} basins")
    return {"links": len(lines), "basins": len(polys), "gpkg": str(gpkg), "threshold_cells": threshold}


# ------------------------------------------------------------------ geomorphometry
def geomorphometry(chunk, metric: str = "tpi", size_m: float = 10.0, progress=None, flatness: float = 1.0,
                   cell_m: float = 0.0) -> dict:
    """One of :data:`METRICS` from the DEM, ``size_m`` the side of the window (TPI, deviation)
    or the search distance (geomorphons; ``flatness`` the angle in degrees under which the ground
    counts as flat). ``cell_m``: the metric computed on the DEM resampled to this cell (0: the
    DEM's own) and brought back onto the DEM's grid - the landforms of a field read at a metre, not
    the furrows read at five centimetres. Product ``<metric>_<size>``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    if metric not in METRICS:
        raise ValueError(f"unknown metric {metric!r}: " + ", ".join(METRICS))
    tool, make, name_it, _name_en, kind = METRICS[metric]
    dem, info = dem_for_tools(chunk)
    cell = info["cell"]
    coarse = cell_m and cell_m > 1.5 * cell
    if coarse:
        dem = coarse_dem(chunk, dem, float(cell_m), progress.sub(0.0, 0.05))
        cell = float(cell_m)
    uses_size = metric in SIZED
    pid = f"{metric}_{size_m:g}".replace(".", "_") if uses_size else metric
    out = work_path(chunk, f"{pid}.tif")
    progress(0.05, name_it)
    if metric == "daylight":
        lat, lon = _latlon(info)
        params = {"lat": round(lat, 5), "long": round(lon, 5), "az_fraction": 15.0, "max_dist": 100.0, "utc_offset": "+00:00"}
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
    entry_ = store(chunk, values, pid, f"{name_it} {size_m:g} m" if uses_size else name_it, kind,
                   {"metric": metric, "size_m": float(size_m), "cell_m": float(cell_m or 0.0), **params}, integer=integer,
                   classes=GEOMORPHON_FORMS if integer else None, description=name_it)
    progress(1.0, f"{name_it}: done")
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
    progress(0.0, "Viewshed")
    wbt.run("viewshed", progress=progress.sub(0.05, 0.85), dem=str(dem), stations=str(station), output=str(out),
            height=float(height_m))
    values = read_result(out)
    pid = slug(f"viewshed_{x:.1f}_{y:.1f}")
    visible = float(np.nanmean(values)) if np.isfinite(values).any() else 0.0
    e = store(chunk, np.where(np.isfinite(values), values, np.nan), pid, f"Visibilità da ({x:.1f}, {y:.1f}), occhio a {height_m:g} m",
              "binary", {"x": float(x), "y": float(y), "height_m": float(height_m), "visible_fraction": visible},
              description="1 visible from the point, 0 hidden")
    progress(1.0, f"Viewshed: {visible * 100:.0f} % of the ground visible")
    return e


def summary(chunk) -> str:
    return json.dumps({"products": entries(chunk)}, indent=1, ensure_ascii=False)
