"""Exports of the terrain products: a GeoTIFF of one, a GeoPackage with every vector table
(stream network, basins, depressions, mounds), the entries for the QGIS project of the program
(named in the language of the interface; the tables keep their Italian names, they are data)."""
from __future__ import annotations

import shutil
from pathlib import Path

from . import overlay, terrain
from .texts import t, table_title

RAMPS = {"diverging": "diverging", "grey": "grey", "log": "grey", "binary": "grey", "classes": "grey", "forms": "grey"}
VECTOR_TABLES = {"hydrology.gpkg": [("reticolo", "LINESTRING", "#1f78ff", "Reticolo idrografico"),
                                    ("bacini", "MULTIPOLYGON", "#b15928", "Bacini")],
                 "depressioni.gpkg": [("depressioni", "MULTIPOLYGON", "#e31a1c", "Depressioni")],
                 "rilievi.gpkg": [("rilievi", "MULTIPOLYGON", "#33a02c", "Rilievi")]}


def export_raster(chunk, pid: str, path: str | Path) -> Path:
    e = terrain.entry(chunk, pid)
    if e is None:
        raise KeyError(pid)
    src = Path(chunk.data_dir) / e["file"]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, path)
    return path


def vector_files(chunk) -> list[tuple[Path, str, str, str, str]]:
    """``(file, table, geometry type, colour, name)`` of the vector tables that exist."""
    out = []
    for name, tables in VECTOR_TABLES.items():
        p = terrain.folder(chunk) / name
        if p.exists():
            for table, geometry, colour, title in tables:
                out.append((p, table, geometry, colour, title))
    return out


def export_gpkg(chunk, path: str | Path, window=None, progress=None) -> Path:
    """Every vector table of the chunk's terrain analyses into one GeoPackage."""
    from pyarchinit_photogrammetry.io import gpkg

    path = Path(path)
    if path.exists():
        path.unlink()
    info = terrain.dem_info(chunk)
    crs = info["crs"].to_string() if info["crs"] is not None else None
    n = 0
    for src, table, geometry, _colour, title in vector_files(chunk):
        rows = [(g, {k: v for k, v in attrs.items() if k != "fid"}) for g, attrs in gpkg.read_table(src, table)]
        if not rows:
            continue
        columns = {}
        for _g, attrs in rows:
            for k, v in attrs.items():
                columns.setdefault(k, "REAL" if isinstance(v, float) else ("INTEGER" if isinstance(v, int) else "TEXT"))
        gpkg.write_table(path, table, geometry, columns, rows, crs, identifier=title)
        n += len(rows)
        if progress:
            progress(0.5, f"{table}: {len(rows)} features")
    if n == 0:
        raise RuntimeError("no vector of the terrain analyses to export: run the hydrology or the depressions first")
    if progress:
        progress(1.0, f"{n} features in {path.name}")
    return path


def qgis_entries(chunk) -> tuple[list[dict], list[dict]]:
    """The rasters (with their ramp and limits) and the vector tables for ``window.qgis_providers``."""
    rasters, vectors = [], []
    try:
        info = terrain.dem_info(chunk)
    except RuntimeError:
        return rasters, vectors
    if info["crs"] is None:
        return rasters, vectors
    for e in terrain.entries(chunk):
        p = Path(chunk.data_dir) / e["file"]
        if not p.exists():
            continue
        st = e.get("stats") or {}
        kind = e.get("kind", "other")
        if kind == "diverging":
            m = max(abs(st.get("p2", 0.0)), abs(st.get("p98", 0.0)), 1e-9)
            lims = [-m, m]
        elif kind == "binary":
            lims = [0.0, 1.0]
        else:
            lims = [st.get("p2", st.get("min", 0.0)), st.get("p98", st.get("max", 1.0))]
        rasters.append({"name": f"{t('Terreno', 'Terrain')}: {terrain.display_name(e)}", "path": str(p),
                        "ramp": RAMPS.get(kind, "grey"),
                        "limits": [float(lims[0]), float(lims[1])], "band": 1, "opacity": 0.8, "hidden": True})
    for src, table, geometry, colour, _title in vector_files(chunk):
        vectors.append({"name": f"{t('Terreno', 'Terrain')}: {table_title(table)}", "path": str(src), "table": table,
                        "geometry": geometry, "color": colour})
    return rasters, vectors


def limits_text(e: dict, lo: float, hi: float) -> str:
    kind = e.get("kind", "other")
    if kind == "classes":
        return f"{terrain.display_name(e)}: {int(lo)} - {int(hi)}"
    return f"{terrain.display_name(e)}: {lo:.3g} - {hi:.3g}"


__all__ = ["export_raster", "export_gpkg", "qgis_entries", "vector_files", "limits_text", "overlay"]
