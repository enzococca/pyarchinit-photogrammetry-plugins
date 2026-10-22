"""Nested depressions (and mounds) of the DEM by the level-set method of Wu, Lane, Wang et al.
(2019): the DEM is filled (WhiteboxTools) and every sink - a connected region where the filled
surface stands above the ground - is sliced from its bottom to its spill level; at each level the
connected parts of the ground under it are the depressions of that level, nested in the ones of
the level above. Each node carries the level, its depth, area, volume and elongation and the id of
the node it lies in. The mounds come from the same on the inverted DEM.

Written here in NumPy and SciPy (the ``lidar`` package needs the GDAL bindings, which the program
does not carry); the method is the published one, the code is ours.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import terrain, wbt


def find(chunk, min_area_m2: float = 1.0, min_depth: float = 0.05, interval: float = 0.0, mounds: bool = False,
         max_levels: int = 20, max_sinks: int = 2000, progress=None) -> dict:
    """The depressions (``mounds``: the hills) of the DEM deeper than ``min_depth`` and larger
    than ``min_area_m2``, sliced every ``interval`` metres (0: a tenth of each sink's depth, at
    most ``max_levels`` slices). Products: ``depth`` (how deep each cell lies under the spill
    level), ``levels`` (how many nested depressions hold the cell) and the GeoPackage
    ``terrain/depressioni.gpkg`` (``rilievi.gpkg``) with the polygons and their properties."""
    from pyarchinit_photogrammetry.core.progress import as_progress
    from scipy import ndimage

    from . import vectors

    progress = as_progress(progress)
    dem, info = terrain.dem_for_tools(chunk, inverted=mounds)
    cell = info["cell"]
    filled_path = terrain.work_path(chunk, "filled_inverted.tif" if mounds else "filled.tif")
    what = "Rilievi" if mounds else "Depressioni"
    progress(0.0, f"{what}: filling the DEM")
    wbt.run("fill_depressions", progress=progress.sub(0.02, 0.2), dem=str(dem), output=str(filled_path), fix_flats=True)
    z = terrain.read_result(dem)
    f = terrain.read_result(filled_path)
    depth = f - z
    ok = np.isfinite(depth) & np.isfinite(z)
    depth = np.where(ok, depth, np.nan)
    structure = np.ones((3, 3), bool)
    labels, n = ndimage.label(ok & (depth > 1e-4), structure=structure)
    progress(0.25, f"{what}: {n} sinks found, sizing them")
    min_cells = max(1, int(round(min_area_m2 / (cell * cell))))
    kept = []
    if n:
        idx = np.arange(1, n + 1)
        areas = ndimage.sum(np.ones_like(labels, dtype=np.int32), labels, idx)
        maxd = ndimage.maximum(np.nan_to_num(depth), labels, idx)
        order = np.argsort(-areas)
        for k in order:
            if areas[k] >= min_cells and maxd[k] >= min_depth:
                kept.append(int(idx[k]))
            if len(kept) >= max_sinks:
                break
    nodes: list[dict] = []
    level_count = np.zeros(z.shape, np.int32)
    next_id = 1
    boxes = ndimage.find_objects(labels) if n else []  # the bounding box of every sink: no pass over the whole DEM per sink
    for si, sink in enumerate(kept):
        progress(0.3 + 0.55 * si / max(len(kept), 1), f"{what}: slicing sink {si + 1} of {len(kept)}")
        box = boxes[sink - 1]
        if box is None:
            continue
        r0, r1, c0, c1 = box[0].start, box[0].stop, box[1].start, box[1].stop
        m = labels[r0:r1, c0:c1] == sink
        zz = z[r0:r1, c0:c1]
        ff = f[r0:r1, c0:c1]
        z_min = float(np.nanmin(np.where(m, zz, np.nan)))
        z_spill = float(np.nanmax(np.where(m, ff, np.nan)))
        total = z_spill - z_min
        if total < min_depth:
            continue
        step = float(interval) if interval and interval > 0 else max(min_depth, total / 10.0)
        n_levels = int(np.ceil(total / step))
        if n_levels > max_levels:
            n_levels = max_levels
            step = total / n_levels
        levels = [z_spill - i * step for i in range(n_levels)]  # from the spill level down
        parent_of_cell = np.zeros(m.shape, np.int64)  # the node id holding each cell at the level above
        for rank, level in enumerate(levels, 1):
            under = m & (zz <= level + 1e-9)
            lab, nn = ndimage.label(under, structure=structure)
            if not nn:
                break
            ids = np.arange(1, nn + 1)
            counts = ndimage.sum(np.ones_like(lab, dtype=np.int32), lab, ids)
            new_parent = np.zeros(m.shape, np.int64)
            from rasterio.windows import Window, transform as window_transform

            sub_tr = window_transform(Window(c0, r0, c1 - c0, r1 - r0), info["transform"])
            polys = vectors._polygons(np.where(lab > 0, lab, np.nan).astype(float), sub_tr)
            for comp, count in zip(ids.tolist(), counts.tolist()):
                if count < min_cells:
                    continue
                cells = lab == comp
                dz = level - zz[cells]
                dmax = float(np.nanmax(dz))
                if dmax < min_depth:
                    continue
                parent = int(np.bincount(parent_of_cell[cells]).argmax()) if rank > 1 else 0
                rr, cc = np.nonzero(cells)
                elong = _elongation(rr, cc)
                from rasterio.transform import xy

                x, y = xy(info["transform"], r0 + rr.mean(), c0 + cc.mean(), offset="center")
                geom = polys.get(comp)
                if geom is None:
                    continue
                sign = -1.0 if mounds else 1.0
                nodes.append({"id": next_id, "sink": sink, "parent": parent, "rank": rank, "level": sign * level,
                              "depth_max": dmax, "depth_mean": float(np.nanmean(dz)), "area_m2": float(count * cell * cell),
                              "volume_m3": float(np.nansum(dz) * cell * cell), "elongation": elong,
                              "x": float(x), "y": float(y), "geometry": geom})
                new_parent[cells] = next_id
                level_count[r0:r1, c0:c1][cells] += 1
                next_id += 1
            parent_of_cell = new_parent
    gpkg = terrain.folder(chunk) / ("rilievi.gpkg" if mounds else "depressioni.gpkg")
    table = "rilievi" if mounds else "depressioni"
    progress(0.9, f"{what}: writing {len(nodes)} polygons")
    vectors.write_depressions(gpkg, table, nodes, info["crs"], what)
    rel = str(gpkg.relative_to(chunk.data_dir))
    params = {"min_area_m2": float(min_area_m2), "min_depth": float(min_depth), "interval": float(interval), "mounds": bool(mounds)}
    depth = np.where(depth > 1e-4, depth, np.nan)  # transparent where the ground is not in a sink
    terrain.store(chunk, depth, "mound_height" if mounds else "depth", "Altezza dei rilievi" if mounds else "Profondità delle depressioni",
                  "grey", params, vector=rel, description="height above the ground around" if mounds else "depth below the spill level")
    terrain.store(chunk, np.where(level_count > 0, level_count, np.nan).astype(float), "mound_levels" if mounds else "depression_levels",
                  "Livelli dei rilievi" if mounds else "Livelli delle depressioni", "classes", params, integer=True, vector=rel,
                  description="nested levels holding the cell")
    largest = max(nodes, key=lambda nd: nd["area_m2"]) if nodes else None
    progress(1.0, f"{what}: {len(kept)} sinks, {len(nodes)} nested polygons")
    return {"sinks": len(kept), "nodes": len(nodes), "gpkg": str(gpkg), "table": table,
            "largest": {k: v for k, v in largest.items() if k != "geometry"} if largest else None}


def _elongation(rows: np.ndarray, cols: np.ndarray) -> float:
    """Major over minor axis of the cells (the second central moments), 1 for a disc."""
    if len(rows) < 3:
        return 1.0
    xy = np.stack([cols - cols.mean(), rows - rows.mean()], axis=1)
    cov = xy.T @ xy / len(rows)
    w = np.linalg.eigvalsh(cov)
    if w[0] <= 1e-9:
        return float("inf") if w[1] > 1e-9 else 1.0
    return float(np.sqrt(w[1] / w[0]))


def read_nodes(chunk, mounds: bool = False) -> list[tuple[object, dict]]:
    """The polygons and properties of the last search, from the GeoPackage."""
    from pyarchinit_photogrammetry.io import gpkg

    path = terrain.folder(chunk) / ("rilievi.gpkg" if mounds else "depressioni.gpkg")
    if not Path(path).exists():
        return []
    return gpkg.read_table(path, "rilievi" if mounds else "depressioni")
