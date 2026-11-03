"""The geometry of a feature: in ``(u, v)`` metres of its elevation (the records), and in the world for the IFC and
the 3D GeoPackage - the polygon with a grid of inner points, triangulated on ``(u, v)`` (constrained Delaunay) and
lifted on the surface through the elevation's relief (the program's ``world_of``); where the relief has no value a
vertex stays on the plane of the elevation (``d = 0``) and is counted, for the export log to say so. A crack becomes a
band of its opening (at least 5 mm), a note a disc of 5 cm, before being lifted."""
from __future__ import annotations

import numpy as np

MIN_BAND = 0.005  # metres: the narrowest band of a crack
NOTE_RADIUS = 0.05  # metres
KEY = 1e-7  # metres: vertices nearer than this on (u, v) are one vertex


def as_geometry(geometry):
    """A shapely geometry from a geometry or its WKT; an invalid polygon repaired (``buffer(0)``). ValueError for a
    text that is not WKT, or a polygon that is empty once repaired."""
    if isinstance(geometry, str):
        import shapely

        try:
            geometry = shapely.from_wkt(geometry)
        except Exception:  # noqa: BLE001 - GEOS says why in its own words
            raise ValueError(f"not a geometry in WKT: {geometry[:80]!r}") from None
    if geometry is None or not hasattr(geometry, "geom_type"):
        raise ValueError(f"not a geometry: {geometry!r}")
    if geometry.geom_type in ("Polygon", "MultiPolygon") and not geometry.is_valid:
        geometry = geometry.buffer(0)
        if geometry.is_empty:
            raise ValueError("the polygon is not valid and nothing is left of it once repaired")
    return geometry


def frame_of(chunk, entry):
    """The frame (Plane or Cylinder) of the elevation ``entry``."""
    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict

    return frame_from_dict(entry["frame"])


def band(line, width_m: float):
    """The band of a crack: the line widened to ``width_m`` (at least ``MIN_BAND``), flat at its ends."""
    return as_geometry(line).buffer(max(float(width_m or 0.0), MIN_BAND) / 2.0, cap_style="flat")


def disc(point, radius: float = NOTE_RADIUS):
    """The disc of a note around its point."""
    return as_geometry(point).buffer(float(radius), quad_segs=16)


def _polygons(geom) -> list:
    import shapely

    return [g for g in shapely.get_parts(geom) if g.geom_type == "Polygon" and not g.is_empty and g.area > 0]


def triangulate(geometry, step: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """``(uv (N, 2), triangles (M, 3))`` of a polygon (or multipolygon) in ``(u, v)``: cut by a grid every ``step``
    metres (the inner points), each piece triangulated with ``shapely.constrained_delaunay_triangles``; triangles
    anticlockwise in ``(u, v)`` (facing the viewer), vertices shared between pieces."""
    import shapely

    geom = as_geometry(geometry)
    if geom.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(f"a polygon is lifted, not a {geom.geom_type}: make a band of a line or a disc of a point")
    step = float(step)
    if step <= 0:
        raise ValueError("the step of the inner points is a positive length")
    corners = []
    for part in _polygons(geom):
        u0, v0, u1, v1 = part.bounds
        i0, i1 = int(np.floor(u0 / step)), int(np.ceil(u1 / step))
        j0, j1 = int(np.floor(v0 / step)), int(np.ceil(v1 / step))
        ii, jj = np.meshgrid(np.arange(i0, max(i1, i0 + 1)), np.arange(j0, max(j1, j0 + 1)))
        cells = shapely.box(ii.ravel() * step, jj.ravel() * step, (ii.ravel() + 1) * step, (jj.ravel() + 1) * step)
        cells = cells[shapely.intersects(cells, part)]
        pieces = [p for g in shapely.intersection(cells, part) for p in _polygons(g)]
        for tris in shapely.constrained_delaunay_triangles(np.array(pieces, dtype=object)):
            for tri in shapely.get_parts(tris):
                corners.append(np.asarray(tri.exterior.coords)[:3])
    if not corners:
        return np.zeros((0, 2)), np.zeros((0, 3), np.int64)
    c = np.asarray(corners, np.float64)  # (M, 3, 2)
    a, b = c[:, 1] - c[:, 0], c[:, 2] - c[:, 0]
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    keep = np.abs(cross) > 1e-14
    c, cross = c[keep], cross[keep]
    c[cross < 0] = c[cross < 0][:, ::-1]
    flat = c.reshape(-1, 2)
    keys = np.round(flat / KEY).astype(np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return flat[first], inverse.reshape(-1, 3).astype(np.int64)


def depth_at(chunk, entry, u, v) -> np.ndarray:
    """The depth ``d`` of the relief of the elevation at the points ``(u, v)`` (as the program's ``relief_at``: the
    cell that holds the point); NaN where the relief has no value or the point is outside it."""
    import rasterio
    from rasterio.windows import Window

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths

    u, v = np.asarray(u, np.float64), np.asarray(v, np.float64)
    out = np.full(u.shape, np.nan)
    if not len(u):
        return out
    path = elevation_paths(chunk, entry)[1]
    if not path.exists():
        return out
    with rasterio.open(path) as src:
        t = src.transform
        cols = np.floor((u - t.c) / t.a).astype(np.int64)
        rows = np.floor((v - t.f) / t.e).astype(np.int64)
        inside = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
        if not inside.any():
            return out
        r0, r1 = int(rows[inside].min()), int(rows[inside].max()) + 1
        c0, c1 = int(cols[inside].min()), int(cols[inside].max()) + 1
        block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
        vals = block[rows[inside] - r0, cols[inside] - c0]
        if src.nodata is not None and np.isfinite(src.nodata):
            vals[vals == src.nodata] = np.nan
        out[inside] = vals
    return out


def lift(chunk, entry, geometry, step: float = 0.10) -> tuple[np.ndarray, np.ndarray, int]:
    """The polygon ``geometry`` (``(u, v)`` of the elevation ``entry``) in the world: ``(vertices (N, 3) in the
    chunk's CRS, triangles (M, 3), the number of vertices put on the plane)``. Each vertex is the program's
    ``world_of(entry, u, v, None, chunk)`` - the surface through the relief - or, where the relief has no value,
    ``world_of(entry, u, v, 0.0)`` on the plane of the elevation (the relief is read once for all the vertices)."""
    uv, tris = triangulate(geometry, step)
    d = depth_at(chunk, entry, uv[:, 0], uv[:, 1])
    missing = ~np.isfinite(d)
    d[missing] = 0.0
    if not len(uv):
        return np.zeros((0, 3)), tris, 0
    verts = frame_of(chunk, entry).point(uv[:, 0], uv[:, 1], d)
    return np.asarray(verts, np.float64), tris, int(missing.sum())
