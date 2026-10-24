"""The window of the area read from the Cloud-Optimized GeoTIFFs of the scenes, and its cache.

Each band is read with rasterio from its COG over HTTPS (GDAL's ``/vsicurl/``: only the bytes of
the window and of the header travel, with retries) or from a local file; only the window that
covers the grid of the area, a few pixels more, is read, then put on the grid: the 10 m bands
as they are, the 20 m bands (B05, B8A, B11) bilinear to 10 m, the scene classification (SCL)
nearest. A scene whose tile is in another UTM zone is warped onto the grid the same way. The
tiles of one acquisition fill each other's empty cells.

The scene classification is read first: clouds (8, 9), thin cirrus (10) and cloud shadows (3),
with a buffer, and every class not kept as clear (by default vegetation 4, bare soil 5 and water
6) are masked; when the clear part of the area is less than ``min_clear`` the acquisition is
left out (``cloudy``) and its other bands are never downloaded. The numbers become reflectance
with the scale and offset of the catalog, unless the offset cannot be right (the blue
reflectance of the land would be below zero: the numbers were already harmonised, the metadata
say otherwise). What was read is cached in
``<chunk data>/sentinel2/scenes/<acquisition>.tif`` (the numbers of the seven bands and SCL,
with their scale and offset): a second look needs no network.
"""
from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .stac import BANDS

log = logging.getLogger("pyarchinit_photogrammetry.plugins.sentinel2")

SCL_CLASSES = {0: "no data", 1: "saturated or defective", 2: "dark area pixels", 3: "cloud shadows",
               4: "vegetation", 5: "not vegetated", 6: "water", 7: "unclassified", 8: "cloud medium probability",
               9: "cloud high probability", 10: "thin cirrus", 11: "snow or ice"}
CLEAR = (4, 5, 6)  # the classes kept: vegetation, not vegetated (bare soil), water
SPREAD = (3, 8, 9, 10)  # clouds, cirrus and their shadows: grown by the buffer
BUFFER_M = 20.0  # metres around clouds and shadows also masked
MIN_CLEAR = 0.5  # the least part of the area that must be clear
WORKERS = 4  # acquisitions read at the same time
BAND_WORKERS = 4  # bands of one acquisition read at the same time
ALL_BANDS = BANDS + ("SCL",)
GDAL_OPTIONS = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",  # a COG is one file: no listing of its folder
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.tiff,.TIF,.TIFF",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_CONNECTTIMEOUT": "20",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_USE_HEAD": "NO",  # the size of a file from its first range: a round trip less per band
    "AWS_NO_SIGN_REQUEST": "YES",  # s3:// hrefs of public buckets
}


# rasterio's reproject of arrays silences a warning with warnings.catch_warnings(), which is not
# thread-safe: reprojecting in several threads at once lets it through. The warping of a window
# takes milliseconds, the downloads around it run in parallel.
WARP_LOCK = threading.Lock()


class Stopped(RuntimeError):
    """The reading was cancelled."""


def gdal_path(href: str) -> str:
    """The path GDAL opens for an href: ``/vsicurl/`` for HTTP(S), ``/vsis3/`` for ``s3://``,
    a local path as it is."""
    h = str(href)
    if h.startswith(("http://", "https://")):
        return "/vsicurl/" + h
    if h.startswith("s3://"):
        return "/vsis3/" + h[5:]
    if h.startswith("file://"):
        return h[7:]
    return h


def gdal_env():
    """The options of GDAL for reading COGs over the network (the certificates of certifi when
    no other bundle is set: a packaged application may not find the system ones)."""
    import rasterio

    options = dict(GDAL_OPTIONS)
    if not os.environ.get("CURL_CA_BUNDLE") and not os.environ.get("SSL_CERT_FILE"):
        try:
            import certifi

            options["CURL_CA_BUNDLE"] = certifi.where()
        except Exception:  # noqa: BLE001
            pass
    return rasterio.Env(**options)


def read_band(href: str, grid, classes: bool = False) -> np.ndarray:
    """One band of a scene on ``grid`` (``area.Grid``): uint16 or uint8 numbers as in the file,
    0 where the scene has no data or does not reach. A band of the grid's zone and cell is copied
    as it is; a coarser one, or one of another UTM zone, is interpolated (bilinear), the classes
    (``classes``: SCL) taken from the nearest pixel."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.warp import Resampling, reproject, transform_bounds
    from rasterio.windows import Window, from_bounds

    with gdal_env(), rasterio.open(gdal_path(href)) as src:
        nodata = src.nodata if src.nodata is not None else 0
        crs = src.crs
        same = crs is not None and crs == CRS.from_user_input(grid.crs)
        t = src.transform
        exact = (same and abs(abs(t.a) - grid.cell) < 1e-9 and abs(abs(t.e) - grid.cell) < 1e-9 and t.b == 0 and t.d == 0
                 and abs(((t.c - grid.x0) / grid.cell + 0.5) % 1 - 0.5) < 1e-6
                 and abs(((t.f - grid.y1) / grid.cell + 0.5) % 1 - 0.5) < 1e-6)  # the same lattice: a copy
        resampling = "nearest" if classes or exact else "bilinear"
        bounds = grid.bounds if same else transform_bounds(grid.crs, crs, *grid.bounds, densify_pts=21)
        win = from_bounds(*bounds, transform=src.transform)
        c0, r0 = int(np.floor(win.col_off)) - 2, int(np.floor(win.row_off)) - 2
        c1, r1 = int(np.ceil(win.col_off + win.width)) + 2, int(np.ceil(win.row_off + win.height)) + 2
        c0, r0, c1, r1 = max(c0, 0), max(r0, 0), min(c1, src.width), min(r1, src.height)
        dtype = np.dtype(src.dtypes[0])
        out = np.zeros(grid.shape, dtype)
        if c1 <= c0 or r1 <= r0:  # the tile does not reach the area
            return out
        window = Window(c0, r0, c1 - c0, r1 - r0)
        arr = src.read(1, window=window)
        win_transform = src.window_transform(window)
    with WARP_LOCK:
        reproject(source=arr, destination=out, src_transform=win_transform, src_crs=crs, src_nodata=nodata,
                  dst_transform=grid.transform, dst_crs=grid.crs, dst_nodata=0, resampling=getattr(Resampling, resampling))
    return out


def clear_mask(scl: np.ndarray, clear=CLEAR, buffer_m: float = BUFFER_M, cell: float = 10.0) -> np.ndarray:
    """The cells of a scene classification that are clear: of the ``clear`` classes, and not
    within ``buffer_m`` of a cloud, cirrus or cloud shadow."""
    ok = np.isin(scl, np.asarray(list(clear), scl.dtype))
    bad = np.isin(scl, np.asarray(SPREAD, scl.dtype))
    radius = int(round(float(buffer_m) / float(cell)))
    if radius > 0 and bad.any():
        from scipy import ndimage

        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        bad = ndimage.binary_dilation(bad, structure=(xx * xx + yy * yy) <= radius * radius)
    return ok & ~bad


def _mosaic(arrays: list[np.ndarray]) -> np.ndarray:
    """The first array, its empty (0) cells filled from the next ones."""
    out = arrays[0].copy()
    for a in arrays[1:]:
        empty = out == 0
        if not empty.any():
            break
        out[empty] = a[empty]
    return out


def _dn_to(values: np.ndarray, scale: float, offset: float, to_scale: float, to_offset: float) -> np.ndarray:
    """Numbers of one scaling as numbers of another (two tiles of an acquisition with a different
    offset), 0 kept as no data."""
    if abs(scale - to_scale) < 1e-12 and abs(offset - to_offset) < 1e-12:
        return values
    ref = values.astype(np.float64) * scale + offset
    out = np.clip(np.rint((ref - to_offset) / to_scale), 1, 65535).astype(values.dtype)
    return np.where(values == 0, 0, out)


def read_acquisition(acq: dict, grid, mask_area: np.ndarray, min_clear: float = MIN_CLEAR, clear=CLEAR,
                     buffer_m: float = BUFFER_M, stop: threading.Event | None = None) -> dict:
    """An acquisition (:func:`stac.acquisitions`) on the grid. Its scene classification first:
    ``{"status": "cloudy", "clear": f}`` when less than ``min_clear`` of the area is clear;
    else ``{"status": "read", "clear", "bands": {band: numbers}, "scl", "scale", "offset"}``."""
    def check():
        if stop is not None and stop.is_set():
            raise Stopped("cancelled")

    items = acq["items"]
    check()
    scl = _mosaic([read_band(it["assets"]["SCL"]["href"], grid, classes=True) for it in items])
    ok = clear_mask(scl, clear, buffer_m, grid.cell)
    n_area = int(mask_area.sum())
    frac = float((ok & mask_area).sum()) / n_area if n_area else 0.0
    if frac < float(min_clear):
        return {"status": "cloudy", "clear": frac}

    def one(band):
        check()
        first = items[0]["assets"][band]
        s, o = float(first["scale"]), float(first["offset"])
        parts = []
        for it in items:
            a = it["assets"][band]
            parts.append(_dn_to(read_band(a["href"], grid), float(a["scale"]), float(a["offset"]), s, o))
        return band, _mosaic(parts), s, o

    bands, scale, offset = {}, {}, {}
    with ThreadPoolExecutor(min(BAND_WORKERS, len(BANDS)), thread_name_prefix="sentinel2-band") as pool:
        for band, values, s, o in pool.map(one, BANDS):
            bands[band], scale[band], offset[band] = values, s, o
    unshifted = _offset_guard(bands, scl, scale, offset)
    if unshifted:
        log.warning("%s: the numbers carry no offset (blue reflectance of the land below 0 with it): offset 0",
                    acq["id"])
        offset = {b: 0.0 for b in BANDS}
    return {"status": "read", "clear": frac, "bands": bands, "scl": scl, "scale": scale, "offset": offset,
            "offset_checked": unshifted}


def _offset_guard(bands: dict, scl: np.ndarray, scale: dict, offset: dict) -> bool:
    """True when the offset declared for the numbers cannot be right: with it the blue reflectance
    of most of the land (vegetation, bare soil) would be below zero - the numbers were already
    harmonised, the catalog's metadata say otherwise."""
    if offset.get("B02", 0.0) > -0.05:
        return False
    land = np.isin(scl, (4, 5)) & (bands["B02"] > 0)
    if land.sum() < 20:
        return False
    return float(np.median(bands["B02"][land])) * scale["B02"] + offset["B02"] < -0.005


# ------------------------------------------------------------------ the cache
def write_cache(path: Path, grid, acq: dict, result: dict, attribution: str = "") -> Path:
    """The bands and SCL of an acquisition as one GeoTIFF of eight uint16 bands (0 = no data),
    the scale and offset of each band in its metadata."""
    import rasterio

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part.tif")
    profile = {"driver": "GTiff", "width": grid.width, "height": grid.height, "count": len(ALL_BANDS),
               "dtype": "uint16", "crs": grid.crs, "transform": grid.transform, "nodata": 0, "compress": "deflate",
               "predictor": 2, "BIGTIFF": "IF_SAFER"}
    if grid.width >= 256 and grid.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(tmp, "w", **profile) as dst:
        for k, band in enumerate(ALL_BANDS, start=1):
            values = result["scl"] if band == "SCL" else result["bands"][band]
            dst.write(values.astype(np.uint16), k)
            dst.set_band_description(k, band)
        dst.scales = tuple(1.0 if b == "SCL" else float(result["scale"][b]) for b in ALL_BANDS)
        dst.offsets = tuple(0.0 if b == "SCL" else float(result["offset"][b]) for b in ALL_BANDS)
        dst.update_tags(ACQUISITION=acq["id"], DATETIME=acq.get("datetime", ""), PLATFORM=acq.get("platform", ""),
                        TILES=",".join(acq.get("tiles") or []), ITEMS=",".join(it["id"] for it in acq["items"]),
                        CLEAR_FRACTION=f"{result['clear']:.4f}", ATTRIBUTION=attribution,
                        OFFSET_CORRECTED="YES" if result.get("offset_checked") else "NO",
                        TIFFTAG_COPYRIGHT=attribution, TIFFTAG_SOFTWARE="pyarchinit-photogrammetry sentinel2")
    os.replace(tmp, path)
    return path


def read_cache(path: Path, grid=None) -> dict | None:
    """``{"reflectance": {band: float32 with NaN}, "scl": uint8, "tags"}`` of a cached
    acquisition; None when the file is missing, unreadable, or on another grid than ``grid``."""
    import rasterio

    path = Path(path)
    if not path.exists():
        return None
    try:
        with rasterio.open(path) as src:
            if grid is not None and not grid.same(src.transform, src.crs, (src.height, src.width)):
                return None
            names = list(src.descriptions)
            if any(b not in names for b in ALL_BANDS):
                return None
            data = src.read()
            scales, offsets = src.scales, src.offsets
            tags = src.tags()
    except Exception as exc:  # noqa: BLE001 - a broken cache file is read again from the catalog
        log.warning("%s: %s", path, exc)
        return None
    out = {}
    for band in BANDS:
        k = names.index(band)
        dn = data[k]
        out[band] = np.where(dn > 0, dn.astype(np.float32) * np.float32(scales[k]) + np.float32(offsets[k]),
                             np.nan).astype(np.float32)
    return {"reflectance": out, "scl": data[names.index("SCL")].astype(np.uint8), "tags": tags}


def cache_valid(path: Path, grid) -> bool:
    """A cached acquisition is there, readable, with every band, on ``grid``."""
    import rasterio

    path = Path(path)
    if not path.exists():
        return False
    try:
        with rasterio.open(path) as src:
            return (grid.same(src.transform, src.crs, (src.height, src.width))
                    and all(b in list(src.descriptions) for b in ALL_BANDS))
    except Exception:  # noqa: BLE001
        return False


def cache_clear_fraction(path: Path) -> float | None:
    import rasterio

    try:
        with rasterio.open(path) as src:
            return float(src.tags().get("CLEAR_FRACTION"))
    except Exception:  # noqa: BLE001
        return None


# ------------------------------------------------------------------ many acquisitions
def fetch(acqs: list[dict], grid, mask_area: np.ndarray, cache_dir: Path, min_clear: float = MIN_CLEAR,
          clear=CLEAR, buffer_m: float = BUFFER_M, workers: int = WORKERS, attribution=None,
          progress=None) -> dict:
    """The acquisitions read on the grid, ``workers`` at a time, and cached; those already in
    the cache are not read again. Returns ``{acquisition id: {"status": "cached"|"cloudy"|"failed",
    "clear", "file", "error", "read" (True when read from the catalog now)}}``. ``progress`` raises
    when the user cancels: the readings under way stop at their next band."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    cache_dir = Path(cache_dir)
    results: dict[str, dict] = {}
    todo = []
    for acq in acqs:
        path = cache_dir / f"{acq['id']}.tif"
        if cache_valid(path, grid):
            results[acq["id"]] = {"status": "cached", "clear": cache_clear_fraction(path), "file": path, "read": False}
        else:
            todo.append(acq)
    total = max(1, len(todo))
    if not todo:
        progress(1.0, f"{len(results)} acquisitions in the cache")
        return results
    stop = threading.Event()
    done = 0
    progress(0.0, f"reading {len(todo)} acquisitions ({len(results)} in the cache)")

    def job(acq):
        r = read_acquisition(acq, grid, mask_area, min_clear, clear, buffer_m, stop)
        if r["status"] == "read":
            text = attribution(acq) if callable(attribution) else (attribution or "")
            path = write_cache(cache_dir / f"{acq['id']}.tif", grid, acq, r, text)
            return {"status": "cached", "clear": r["clear"], "file": path, "read": True}
        return {"status": "cloudy", "clear": r["clear"], "file": None, "read": True}

    pool = ThreadPoolExecutor(max(1, int(workers)), thread_name_prefix="sentinel2")
    futures = {pool.submit(job, acq): acq for acq in todo}
    try:
        for fut in as_completed(futures):
            acq = futures[fut]
            try:
                results[acq["id"]] = fut.result()
            except Stopped:
                continue
            except Exception as exc:  # noqa: BLE001 - one scene that cannot be read does not stop the others
                log.warning("%s: %s", acq["id"], exc)
                results[acq["id"]] = {"status": "failed", "clear": None, "file": None, "read": True,
                                      "error": str(exc)[:300]}
            done += 1
            r = results[acq["id"]]
            clear_pct = "" if r.get("clear") is None else f", {r['clear'] * 100:.0f}% clear"
            progress(done / total, f"{acq['date']} {acq['platform']}: {r['status']}{clear_pct}")
    except BaseException:
        stop.set()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    pool.shutdown(wait=True)
    return results


def describe(results: dict) -> dict:
    """How many acquisitions were read, found in the cache, cloudy, failed."""
    out = {"read": 0, "cached": 0, "cloudy": 0, "failed": 0}
    for r in results.values():
        if r["status"] == "cached":
            out["read" if r.get("read") else "cached"] += 1
        else:
            out[r["status"]] = out.get(r["status"], 0) + 1
    return out
