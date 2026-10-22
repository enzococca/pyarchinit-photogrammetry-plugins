"""Vectors from the rasters of WhiteboxTools: the stream network as lines traced along the D8
pointers, the basins and the depressions as polygons, written into GeoPackages with the program's
own writer; a minimal point Shapefile for the tools that want one (the viewshed's station)."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

# WhiteboxTools' D8 pointer: the value of a cell says where it flows (row step, column step)
D8 = {1: (-1, 1), 2: (0, 1), 4: (1, 1), 8: (1, 0), 16: (1, -1), 32: (0, -1), 64: (-1, -1), 128: (-1, 0)}


def _xy(transform, row: int, col: int) -> tuple[float, float]:
    from rasterio.transform import xy

    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def stream_lines(links: np.ndarray, d8: np.ndarray, facc: np.ndarray, transform) -> list[dict]:
    """One line per stream link: from the cell of the link no other cell of it flows into, along
    the pointers to the cell past its end (so the links join); ``cells`` the flow accumulation at
    its end, ``length_m`` its length."""
    out = []
    ok = np.isfinite(links) & (links > 0)
    if not ok.any():
        return out
    ids = np.where(ok, links, 0).astype(np.int64)
    h, w = links.shape
    rows, cols = np.nonzero(ok)
    # the downstream cell of every stream cell
    down = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        p = d8[r, c]
        if not np.isfinite(p) or int(p) not in D8:
            continue
        dr, dc = D8[int(p)]
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            down[(r, c)] = (rr, cc)
    fed = set(down.values())
    by_link: dict[int, list] = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        by_link.setdefault(int(ids[r, c]), []).append((r, c))
    for link, cells in by_link.items():
        heads = [rc for rc in cells if rc not in fed or all(ids[s] != link for s in _sources(rc, down, cells))]
        start = heads[0] if heads else cells[0]
        if not heads:  # a loop or a link fed only by itself: start at its highest accumulation inverse
            start = min(cells, key=lambda rc: facc[rc] if np.isfinite(facc[rc]) else 0.0)
        path = [start]
        seen = {start}
        cur = start
        while cur in down:
            nxt = down[cur]
            path.append(nxt)
            if nxt in seen or ids[nxt] != link:
                break
            seen.add(nxt)
            cur = nxt
        coords = [_xy(transform, r, c) for r, c in path]
        if len(coords) < 2:
            continue
        a = np.asarray(coords)
        length = float(np.sum(np.hypot(np.diff(a[:, 0]), np.diff(a[:, 1]))))
        end = path[-2] if len(path) > 1 else path[-1]
        cells_at_end = float(facc[end]) if np.isfinite(facc[end]) else 0.0
        out.append({"link": link, "coords": coords, "cells": cells_at_end, "length_m": length})
    return out


def _sources(rc, down: dict, cells: list):
    return [s for s in cells if down.get(s) == rc]


def _polygons(values: np.ndarray, transform) -> dict[int, object]:
    """``{id: shapely (Multi)Polygon}`` of the cells of every id of an integer raster."""
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    ok = np.isfinite(values) & (values > 0)
    if not ok.any():
        return {}
    ids = np.where(ok, values, 0).astype(np.int32)
    parts: dict[int, list] = {}
    for geom, v in features.shapes(ids, mask=ok, transform=transform, connectivity=8):
        parts.setdefault(int(v), []).append(shape(geom))
    return {k: unary_union(g) if len(g) > 1 else g[0] for k, g in parts.items()}


def basin_polygons(basins: np.ndarray, transform) -> list[dict]:
    polys = _polygons(basins, transform)
    out = []
    for bid, geom in sorted(polys.items(), key=lambda kv: -kv[1].area):
        out.append({"basin": bid, "geometry": geom, "area_m2": float(geom.area)})
    return out


def _multi(geom):
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    return geom


def write_hydrology(path: Path, lines: list[dict], basins: list[dict], crs, cell: float) -> Path:
    from pyarchinit_photogrammetry.io import gpkg
    from shapely.geometry import LineString

    crs_text = crs.to_string() if crs is not None else None
    path = Path(path)
    if path.exists():
        path.unlink()
    gpkg.write_table(path, "reticolo", "LINESTRING",
                     {"link": "INTEGER", "cells": "REAL", "area_m2": "REAL", "length_m": "REAL"},
                     [(LineString(ln["coords"]), {"link": ln["link"], "cells": ln["cells"],
                                                  "area_m2": ln["cells"] * cell * cell, "length_m": ln["length_m"]})
                      for ln in lines], crs_text, identifier="Reticolo idrografico")
    gpkg.write_table(path, "bacini", "MULTIPOLYGON", {"basin": "INTEGER", "area_m2": "REAL"},
                     [(_multi(b["geometry"]), {"basin": b["basin"], "area_m2": b["area_m2"]}) for b in basins],
                     crs_text, identifier="Bacini")
    return path


def write_depressions(path: Path, table: str, nodes: list[dict], crs, identifier: str) -> Path:
    from pyarchinit_photogrammetry.io import gpkg

    crs_text = crs.to_string() if crs is not None else None
    path = Path(path)
    if path.exists():
        path.unlink()
    columns = {"id": "INTEGER", "sink": "INTEGER", "parent": "INTEGER", "rank": "INTEGER", "level": "REAL",
               "depth_max": "REAL", "depth_mean": "REAL", "area_m2": "REAL", "volume_m3": "REAL", "elongation": "REAL",
               "x": "REAL", "y": "REAL"}
    rows = [(_multi(n["geometry"]), {k: n[k] for k in columns}) for n in nodes]
    gpkg.write_table(path, table, "MULTIPOLYGON", columns, rows, crs_text, identifier=identifier)
    return path


# ------------------------------------------------------------------ a point Shapefile
def write_point_shapefile(path: Path, points: list[tuple[float, float]], crs=None) -> Path:
    """``.shp``, ``.shx``, ``.dbf`` (a field ``id``) and ``.prj`` of ``points``: what a tool of
    WhiteboxTools reads as its stations or pour points."""
    path = Path(path)
    base = path.with_suffix("")
    n = len(points)
    xs = [p[0] for p in points] or [0.0]
    ys = [p[1] for p in points] or [0.0]
    content_words = 10  # shape type + x + y = 20 bytes
    shp_len = 50 + n * (4 + content_words)  # 16-bit words
    shx_len = 50 + n * 4

    def header(length_words: int) -> bytes:
        return (struct.pack(">i", 9994) + b"\0" * 20 + struct.pack(">i", length_words) + struct.pack("<ii", 1000, 1)
                + struct.pack("<4d", min(xs), min(ys), max(xs), max(ys)) + struct.pack("<4d", 0, 0, 0, 0))

    with open(base.with_suffix(".shp"), "wb") as shp, open(base.with_suffix(".shx"), "wb") as shx:
        shp.write(header(shp_len))
        shx.write(header(shx_len))
        offset = 50
        for i, (x, y) in enumerate(points, 1):
            shp.write(struct.pack(">ii", i, content_words) + struct.pack("<i", 1) + struct.pack("<2d", x, y))
            shx.write(struct.pack(">ii", offset, content_words))
            offset += 4 + content_words
    with open(base.with_suffix(".dbf"), "wb") as dbf:
        field = b"id".ljust(11, b"\0") + b"N" + b"\0" * 4 + bytes([10, 0]) + b"\0" * 14
        dbf.write(bytes([3, 24, 1, 1]) + struct.pack("<I", n) + struct.pack("<HH", 32 + 32 + 1, 11) + b"\0" * 20)
        dbf.write(field + b"\r")
        for i in range(1, n + 1):
            dbf.write(b" " + str(i).rjust(10).encode())
        dbf.write(b"\x1a")
    if crs is not None:
        try:
            import rasterio

            wkt = rasterio.crs.CRS.from_user_input(crs).to_wkt()
            base.with_suffix(".prj").write_text(wkt)
        except Exception:  # noqa: BLE001
            pass
    return base.with_suffix(".shp")
