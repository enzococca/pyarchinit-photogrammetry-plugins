"""Rasters through terrio: web tiles of the orthomosaic (PMTiles, MBTiles) and a raster of
reference read from the web (a Cloud Optimized GeoTIFF, a STAC search, a collection of Microsoft
Planetary Computer) or from a file, warped onto the survey and kept as a version of the DEM or of
the orthomosaic - a DEM with its heights moved into the survey's height system when asked
(:mod:`.heights`).

Planetary Computer (https://planetarycomputer.microsoft.com): its STAC API is searched over the
survey (a ``POST`` search, every page followed) and the items found are read as one mosaic, by
window; terrio signs the addresses of its files with the SAS tokens of the Planetary Computer by
itself, as its ``planetary-computer`` package does. Without an account the access is anonymous
and rate limited; a subscription key in ``PC_SDK_SUBSCRIPTION_KEY`` is sent when set.
"""
from __future__ import annotations

import json
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.progress import as_progress

from . import environment
from .texts import t

PLANETARY = "https://planetarycomputer.microsoft.com/api/stac/v1"
# collections of the Planetary Computer offered by name: the asset read, the product it becomes,
# the height system of a DEM (heights.normalise), the bands kept, the search for imagery
CATALOGUE = {
    "cop-dem-glo-30": {"asset": "data", "product": "dem", "heights": "EPSG:3855",
                       "label": ("Copernicus DEM GLO-30 (30 m, quote EGM2008)", "Copernicus DEM GLO-30 (30 m, EGM2008 heights)")},
    "cop-dem-glo-90": {"asset": "data", "product": "dem", "heights": "EPSG:3855",
                       "label": ("Copernicus DEM GLO-90 (90 m, quote EGM2008)", "Copernicus DEM GLO-90 (90 m, EGM2008 heights)")},
    "alos-dem": {"asset": "data", "product": "dem", "heights": "EPSG:5773",
                 "label": ("ALOS World 3D (30 m, quote EGM96)", "ALOS World 3D (30 m, EGM96 heights)")},
    "nasadem": {"asset": "elevation", "product": "dem", "heights": "EPSG:5773",
                "label": ("NASADEM (30 m, quote EGM96)", "NASADEM (30 m, EGM96 heights)")},
    "3dep-seamless": {"asset": "data", "product": "dem", "heights": "EPSG:5703",
                      "label": ("USGS 3DEP (solo Stati Uniti, 1-10 m, quote NAVD88)",
                                "USGS 3DEP (United States only, 1-10 m, NAVD88 heights)")},
    "sentinel-2-l2a": {"asset": "visual", "product": "orthomosaic", "cloud": 10, "max_items": 1,
                       "label": ("Sentinel-2 L2A, colori naturali (10 m)", "Sentinel-2 L2A, natural colour (10 m)")},
    "naip": {"asset": "image", "product": "orthomosaic", "bands": [1, 2, 3], "max_items": 1,
             "label": ("NAIP (solo Stati Uniti, 0,3-1 m)", "NAIP (United States only, 0.3-1 m)")},
}

FILES = {"dem": "dem.tif", "orthomosaic": "orthomosaic.tif"}  # as core.versions names the files
TILE_SUFFIXES = (".pmtiles", ".mbtiles")


def export_tiles(chunk, path, progress=None) -> Path:
    """The orthomosaic as web map tiles (PMTiles or MBTiles, Web Mercator): terrio reads the
    GeoTIFF window by window and reprojects it."""
    terrio = environment.load()
    progress = as_progress(progress)
    src = chunk.orthomosaic_path
    if src is None or not src.exists():
        raise RuntimeError("build the orthomosaic first")
    path = Path(path)
    if path.suffix.lower() not in TILE_SUFFIXES:
        raise ValueError(f"{path.name}: .pmtiles or .mbtiles")
    progress(0.1, f"Reading {src.name}")
    raster = terrio.open_raster(str(src), lazy=True)
    progress(0.3, f"Writing the tiles of {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    raster.write(str(path))
    progress(1.0, str(path))
    return path


def survey_bounds(chunk) -> tuple[float, float, float, float]:
    """``(min_x, min_y, max_x, max_y)`` of the survey in the chunk CRS: of the DEM, else of the
    orthomosaic, else of the dense cloud, else of the aligned photos."""
    for p in (chunk.dem_path, chunk.orthomosaic_path):
        if p is not None and p.exists():
            import rasterio

            with rasterio.open(p) as s:
                b = s.bounds
            return float(b.left), float(b.bottom), float(b.right), float(b.top)
    pc = chunk.point_cloud
    if pc is not None and len(pc):
        w = chunk.to_world(pc.xyz)
        return float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 0].max()), float(w[:, 1].max())
    cams = chunk.aligned_cameras
    if cams:
        w = chunk.to_world(np.array([c.center for c in cams], float))
        return float(w[:, 0].min()), float(w[:, 1].min()), float(w[:, 0].max()), float(w[:, 1].max())
    raise RuntimeError("the chunk has no DEM, orthomosaic, dense cloud or aligned photos to take its extent from")


def _reference_prefix() -> str:
    """The start of the name of a reference version, in the language of the program: the name is
    what the *Versioni* menu shows (the version is found by its id)."""
    return t("Riferimento: ", "Reference: ")


def catalogue_label(key: str) -> str:
    entry = CATALOGUE.get(key)
    return t(*entry["label"]) if entry else key


def lonlat_bounds(chunk, margin: float = 0.0) -> tuple[float, float, float, float]:
    """The survey (:func:`survey_bounds`) plus ``margin`` metres, in longitude and latitude."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    terrio = environment.load()
    if crs_mod.is_local(chunk.crs):
        raise RuntimeError("the chunk needs a reference system (align it with GPS or markers first)")
    x0, y0, x1, y1 = survey_bounds(chunk)
    m = float(margin)
    xs = np.array([x0 - m, x1 + m, x0 - m, x1 + m, (x0 + x1) / 2, (x0 + x1) / 2, x0 - m, x1 + m])
    ys = np.array([y0 - m, y0 - m, y1 + m, y1 + m, y0 - m, y1 + m, (y0 + y1) / 2, (y0 + y1) / 2])
    lon, lat = terrio.transform_points(xs, ys, chunk.crs, "EPSG:4326")[:2]
    return float(np.min(lon)), float(np.min(lat)), float(np.max(lon)), float(np.max(lat))


def planetary_source(bbox, collection: str, asset: str | None = None, datetime_range: str | None = None,
                     max_items: int | None = None, cloud: float | None = None) -> str:
    """The terrio name of a mosaic of a Planetary Computer collection over ``bbox`` (longitude and
    latitude): ``STACIT:"<search>":asset=...,method=POST``. ``asset``, ``max_items`` and ``cloud``
    (the most cloud cover, per cent) default to those of :data:`CATALOGUE`; ``datetime_range``:
    ``2024-06-01/2024-09-30`` or a date. With a date or a cloud limit the newest items come first."""
    preset = CATALOGUE.get(collection, {})
    asset = asset or preset.get("asset")
    if not asset:
        raise ValueError(f"collection {collection!r}: name the asset to read")
    params = {"collections": collection, "bbox": ",".join(f"{v:.6f}" for v in bbox)}
    if datetime_range:
        params["datetime"] = datetime_range
    cloud = preset.get("cloud") if cloud is None else cloud
    if cloud is not None and float(cloud) < 100:
        params["query"] = json.dumps({"eo:cloud_cover": {"lt": float(cloud)}})
    if datetime_range or "max_items" in preset or cloud is not None:
        params["sortby"] = "-datetime"
    options = [f"asset={asset}", "method=POST"]
    limit = max_items if max_items is not None else preset.get("max_items")
    if limit:
        options.append(f"max_items={int(limit)}")
    return f'STACIT:"{PLANETARY}/search?{urllib.parse.urlencode(params)}":' + ",".join(options)


def reference_raster(chunk, url: str | None = None, product: str | None = None, margin: float = 20.0,
                     resolution: float | None = None, name: str | None = None, heights_from: str | None = None,
                     heights_to: str | None = None, collection: str | None = None, asset: str | None = None,
                     datetime_range: str | None = None, max_items: int | None = None, cloud: float | None = None,
                     grids: dict | None = None, progress=None) -> dict:
    """A raster of the web (a COG by URL, a STAC search as ``STACIT:...``) or a file - or, with
    ``collection``, a collection of the Planetary Computer (:data:`CATALOGUE` or any other with its
    ``asset``) searched over the survey - warped into the CRS of the chunk over the survey plus
    ``margin`` metres (``resolution`` in metres, else the source's), kept as a version of the DEM or
    of the orthomosaic (*Versioni* in the window; the active one is not touched), named ``name`` or
    ``Riferimento: <file>`` (``Reference: <file>`` when the program speaks English).
    ``heights_from``/``heights_to``: a DEM's heights moved from the system of the source (by default
    the one :data:`CATALOGUE` gives) to the survey's (:mod:`.heights`; ``grids``: the options of the
    grids). Returns the file, the version and its size."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    terrio = environment.load()
    progress = as_progress(progress)
    preset = CATALOGUE.get(collection or "", {})
    product = product or preset.get("product") or "dem"
    if product not in FILES:
        raise ValueError(f"product {product!r}: dem or orthomosaic")
    if crs_mod.is_local(chunk.crs):
        raise RuntimeError("the chunk needs a reference system (align it with GPS or markers first)")
    if collection:
        terrio = environment.load_features(t("Il Planetary Computer", "The Planetary Computer"))
        url = planetary_source(lonlat_bounds(chunk, float(margin) + 100.0), collection, asset, datetime_range,
                               max_items, cloud)
        heights_from = heights_from or preset.get("heights")
    if not url:
        raise ValueError("give a URL, a file or a collection")
    if heights_to and product != "dem":
        raise ValueError("heights are moved for a DEM only")
    x0, y0, x1, y1 = survey_bounds(chunk)
    m = float(margin)
    extent = (x0 - m, y0 - m, x1 + m, y1 + m)
    progress(0.1, f"Reading {collection or url}")
    kw = {"t_srs": chunk.crs, "te": extent, "resampling": "bilinear" if product == "dem" else "cubic"}
    if resolution:
        kw["tr"] = (float(resolution), float(resolution))
    raster = terrio.warp(str(url), **kw)
    bands = preset.get("bands")
    if product == "orthomosaic" and bands and raster.count > len(bands):
        raster = terrio.translate(raster, bands=list(bands))
    shift = None
    if product == "dem" and heights_from and heights_to:
        from . import heights

        if heights.normalise(heights_from) != heights.normalise(heights_to):
            progress(0.6, t("Quote da {0} a {1}", "Heights from {0} to {1}").format(
                heights.label(heights_from), heights.label(heights_to)))
            raster, shift = heights.shift_raster(raster, chunk.crs, heights_from, heights_to, **(grids or {}))
    vid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = chunk.data_dir / "versions" / product / vid
    folder.mkdir(parents=True)
    out = folder / FILES[product]
    progress(0.8, f"Writing {out.name}")
    raster.write(str(out), compress="DEFLATE", tile_size=512, overviews=[2, 4, 8, 16],
                 overview_resampling="average")
    if name:
        label = name
    elif collection:
        label = _reference_prefix() + collection
    else:
        label = _reference_prefix() + (Path(str(url)).name or str(url))
    if shift is not None and not name:
        from . import heights

        label = t("{0} (quote {1})", "{0} ({1} heights)").format(label, heights.short_label(heights_to))
    label = label[:100]
    params = {"source": str(url), "margin": m, "resolution": float(resolution or 0), "terrio": str(terrio.__version__)}
    if collection:
        params.update({"catalogue": "planetary-computer", "collection": collection})
    if shift is not None:
        from . import heights

        params.update({"heights": heights.normalise(heights_to), "heights_from": heights.normalise(heights_from),
                       "height_shift": [round(v, 4) for v in shift]})
    entry = {"id": vid, "name": label, "created": datetime.now().isoformat(timespec="seconds"),
             "params": params, "files": {product: str(out.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault(product, []).append(entry)
    progress(1.0, f"{raster.width} x {raster.height} pixels")
    return {"path": out, "version": vid, "name": label, "width": int(raster.width), "height": int(raster.height),
            "resolution": tuple(float(v) for v in raster.res), "bounds": tuple(float(v) for v in raster.bounds),
            "source": str(url), "product": product, "shift": shift}
