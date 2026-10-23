"""Rasters through terrio: web tiles of the orthomosaic (PMTiles, MBTiles) and a raster of
reference read from the web (a Cloud Optimized GeoTIFF, a STAC search) or from a file, warped
onto the survey and kept as a version of the DEM or of the orthomosaic."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from pyarchinit_photogrammetry.core.progress import as_progress

from . import environment

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
    from . import t

    return t("Riferimento: ", "Reference: ")


def reference_raster(chunk, url: str, product: str = "dem", margin: float = 20.0, resolution: float | None = None,
                     name: str | None = None, progress=None) -> dict:
    """A raster of the web (a COG by URL, a STAC search as ``STACIT:...``) or a file, warped into
    the CRS of the chunk over the survey plus ``margin`` metres (``resolution`` in metres, else the
    source's), kept as a version of the DEM or of the orthomosaic (*Versioni* in the window; the
    active one is not touched), named ``name`` or ``Riferimento: <file>`` (``Reference: <file>`` when
    the program speaks English). Returns the file, the version and its size."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    terrio = environment.load()
    progress = as_progress(progress)
    if product not in FILES:
        raise ValueError(f"product {product!r}: dem or orthomosaic")
    if crs_mod.is_local(chunk.crs):
        raise RuntimeError("the chunk needs a reference system (align it with GPS or markers first)")
    x0, y0, x1, y1 = survey_bounds(chunk)
    m = float(margin)
    extent = (x0 - m, y0 - m, x1 + m, y1 + m)
    progress(0.1, f"Reading {url}")
    kw = {"t_srs": chunk.crs, "te": extent, "resampling": "bilinear" if product == "dem" else "cubic"}
    if resolution:
        kw["tr"] = (float(resolution), float(resolution))
    raster = terrio.warp(str(url), **kw)
    vid = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    folder = chunk.data_dir / "versions" / product / vid
    folder.mkdir(parents=True)
    out = folder / FILES[product]
    progress(0.8, f"Writing {out.name}")
    raster.write(str(out), compress="DEFLATE", tile_size=512, overviews=[2, 4, 8, 16],
                 overview_resampling="average")
    label = name or (_reference_prefix() + (Path(str(url)).name or str(url)))[:80]
    entry = {"id": vid, "name": label, "created": datetime.now().isoformat(timespec="seconds"),
             "params": {"source": str(url), "margin": m, "resolution": float(resolution or 0),
                        "terrio": str(terrio.__version__)},
             "files": {product: str(out.relative_to(chunk.data_dir))}}
    chunk.meta.setdefault("versions", {}).setdefault(product, []).append(entry)
    progress(1.0, f"{raster.width} x {raster.height} pixels")
    return {"path": out, "version": vid, "name": label, "width": int(raster.width), "height": int(raster.height),
            "resolution": tuple(float(v) for v in raster.res), "bounds": tuple(float(v) for v in raster.bounds)}
