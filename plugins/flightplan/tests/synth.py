"""Synthetic projects for the tests: a document with one chunk in a projected system and, when asked, a DEM over
a neutral place (``x0, y0 = 500000, 4600000`` in the chunk's system: no site)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

X0, Y0 = 500000.0, 4600000.0
DEM_HALF = 1200.0          # the DEM spans 2.4 km around (X0, Y0)
DEM_CELL = 4.0


def write_dem(path: Path, heights, crs: str = "EPSG:32632", half: float = DEM_HALF, cell: float = DEM_CELL) -> Path:
    """A float32 GeoTIFF around (X0, Y0): ``heights`` a number (a plane) or ``f(x, y) -> z`` on the cell centres."""
    import rasterio
    from rasterio.transform import from_origin

    n = int(round(2 * half / cell))
    xs = X0 - half + (np.arange(n) + 0.5) * cell
    ys = Y0 + half - (np.arange(n) + 0.5) * cell
    gx, gy = np.meshgrid(xs, ys)
    z = np.full(gx.shape, float(heights), "float32") if not callable(heights) else \
        np.asarray(heights(gx, gy), "float32")
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=1, dtype="float32", crs=crs,
                       transform=from_origin(X0 - half, Y0 + half, cell, cell), nodata=float("nan")) as ds:
        ds.write(z, 1)
    return path


def chunk(tmp_path: Path, crs: str | None = "EPSG:32632", dem=None, name: str = "campo"):
    """A saved project with one chunk in ``crs`` (None: a local chunk) and, with ``dem`` (a number or
    ``f(x, y)``), a DEM product. Returns ``(doc, chunk)``."""
    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    ch = doc.add_chunk("Campo")
    doc.chunk = ch
    doc.save(Path(tmp_path) / f"{name}.ppg")
    if crs:
        ch.crs = crs
    if dem is not None:
        path = write_dem(ch.data_dir / "dem.tif", dem, crs or "EPSG:32632")
        ch._set_product("dem", path)
    doc.save()
    return doc, ch


def rectangle_xy(w: float, h: float, cx: float = X0, cy: float = Y0) -> list[tuple[float, float]]:
    """A rectangle of ``w`` x ``h`` metres (east x north) around (cx, cy), in the chunk's system."""
    return [(cx - w / 2, cy - h / 2), (cx + w / 2, cy - h / 2), (cx + w / 2, cy + h / 2), (cx - w / 2, cy + h / 2)]


def area_latlon(w: float, h: float, crs: str = "EPSG:32632", cx: float = X0, cy: float = Y0) -> list[tuple]:
    """The same rectangle in latitude and longitude (pyproj)."""
    from pyproj import Transformer

    tr = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    out = []
    for x, y in rectangle_xy(w, h, cx, cy):
        lon, lat = tr.transform(x, y)
        out.append((lat, lon))
    return out
