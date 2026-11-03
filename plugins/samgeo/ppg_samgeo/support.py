"""Objects held to the relief: what stands above the ground stays, the rest goes.

Grounding DINO sees a wall from above poorly (a nadir wall scores 0.05-0.2, the stones around it
as much), so its proposals cover the wall and its surroundings alike. The DEM knows where the wall
is: :func:`hold` keeps of every object the part standing at least ``min_height`` above the local
ground - the ground being the DEM *opened* (a grey opening: the lowest surface the window of
:data:`GROUND_WINDOW` metres can slide under, which a wall thinner than the window cannot lift) and
the height the DEM minus it - and drops the objects whose standing part is less than
:data:`MIN_SHARE` of them. Each object is worked on its own window of the DEM, at the coarser of the
DEM's resolution, the run's and :data:`GRID` metres: a site at 1 cm/px never has to be in memory.
"""
from __future__ import annotations

import math

import numpy as np

GROUND_WINDOW = 3.0  # m: the window of the opening that gives the ground; walls are thinner
MIN_SHARE = 0.2  # share of an object standing above the ground for the object to stay
GRID = 0.02  # m/px: the heights are never computed finer than this
MAX_GRID = 2000  # px: an object's window is never computed on a finer grid than this many pixels a side


def heights(dem_path, bounds, resolution: float, window: float = GROUND_WINDOW):
    """The height above the local ground on the grid ``bounds`` at ``resolution`` m/px: ``(heights
    float32 with NaN where the DEM has no value, transform)``. The ground is the DEM opened with a
    square window of ``window`` metres a side; the DEM is read with a margin of half the window
    around the bounds so that the opening at the edge sees the ground beyond them. Where the DEM
    has no value the nearest value stands in for the opening (a hole never lifts or lowers the
    ground around it); the heights there stay NaN."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds
    from scipy import ndimage

    x0, y0, x1, y1 = bounds
    w = max(1, int(math.ceil((x1 - x0) / resolution)))
    h = max(1, int(math.ceil((y1 - y0) / resolution)))
    k = max(1, int(round(float(window) / (2 * resolution))))  # half the window, in pixels
    tr = Affine(resolution, 0.0, x0, 0.0, -resolution, y1)
    with rasterio.open(dem_path) as src:
        win = from_bounds(x0 - k * resolution, y0 - k * resolution, x1 + k * resolution, y1 + k * resolution,
                          transform=src.transform)
        data = src.read(1, window=win, out_shape=(h + 2 * k, w + 2 * k), resampling=Resampling.bilinear,
                        boundless=True, fill_value=src.nodata if src.nodata is not None else np.nan, masked=True)
    z = np.ma.filled(data.astype(np.float32), np.nan)
    valid = np.isfinite(z)
    if not valid.any():
        return np.full((h, w), np.nan, np.float32), tr
    if valid.all():
        filled = z
    else:
        rows, cols = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
        filled = z[rows, cols]
    size = 2 * k + 1
    ground = ndimage.grey_opening(filled, size=(size, size), mode="nearest")
    above = np.where(valid, z - ground, np.nan).astype(np.float32)
    return above[k:k + h, k:k + w], tr


def hold(objects: list[dict], dem_path, min_height: float, resolution: float | None = None,
         window: float = GROUND_WINDOW, min_share: float = MIN_SHARE, progress=None) -> list[dict]:
    """The ``objects`` (dicts with a shapely ``geometry``) held to the relief of the DEM ``dem_path``:
    every object cut to the part standing ``min_height`` metres or more above the local ground
    (``relief_m``: the median height of that part, ``relief_share``: how much of the object stood),
    those standing on less than ``min_share`` of their area left out. ``resolution``: the run's
    (the heights are computed at the coarser of it, the DEM's and :data:`GRID`, and for a big object
    coarser still, so that its window is at most :data:`MAX_GRID` pixels a side)."""
    import rasterio
    from rasterio.features import rasterize
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.stratigraphy import mask_to_polygons
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    with rasterio.open(dem_path) as src:
        native = float(min(abs(src.res[0]), abs(src.res[1])))
        extent = box(*src.bounds)
    res = max(native, float(resolution or 0.0), GRID)
    out = []
    n = len(objects)
    for i, o in enumerate(objects):
        if i % 10 == 0:
            progress(i / max(n, 1), f"samgeo: relief {i + 1} / {n}")
        g = o["geometry"]
        if g.is_empty or not g.intersects(extent):
            continue
        b = g.bounds
        grid = max(res, max(b[2] - b[0], b[3] - b[1]) / MAX_GRID)
        bounds = (b[0] - grid, b[1] - grid, b[2] + grid, b[3] + grid)
        above, tr = heights(dem_path, bounds, grid, window)
        shape = above.shape
        mask = rasterize([(g, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8").astype(bool)
        tiny = not mask.any()
        if tiny:  # smaller than a cell: judged by the cells it touches, kept whole
            mask = rasterize([(g, 1)], out_shape=shape, transform=tr, fill=0, dtype="uint8", all_touched=True).astype(bool)
        total = int(mask.sum())
        if not total:
            continue
        standing = mask & np.isfinite(above) & (above >= float(min_height))
        share = standing.sum() / total
        if share < min_share:
            continue
        if tiny:
            out.append({**o, "relief_m": round(float(np.nanmedian(above[standing])), 3), "relief_share": round(float(share), 3)})
            continue
        kept = mask_to_polygons(standing, tr, simplify=grid, min_area=grid * grid)
        if kept is None or kept.is_empty:
            continue
        cut = kept.intersection(g)
        if not cut.is_valid:
            cut = cut.buffer(0)
        parts = [p for p in getattr(cut, "geoms", [cut]) if p.geom_type == "Polygon" and p.area > 0]
        if not parts:
            continue
        from shapely.geometry import MultiPolygon

        geom = parts[0] if len(parts) == 1 else MultiPolygon(parts)
        out.append({**o, "geometry": geom, "relief_m": round(float(np.nanmedian(above[standing])), 3),
                    "relief_share": round(float(share), 3)})
    progress(1.0, f"samgeo: relief {len(out)}")
    return out
