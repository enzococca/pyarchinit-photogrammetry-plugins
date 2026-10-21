"""Exports of the geophysical grids: GeoTIFF and Surfer GRD of a grid, a GeoPackage with the
outlines of the grids and the anomalies drawn (the polygons of the layer *Anomalie geofisiche*
of the raster view), the entries for the QGIS project of the program."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .grids import Grid, entries, load_grid, write_geotiff, write_grd
from .overlay import limits

ANOMALIES_LAYER = "Anomalie geofisiche"


def export_grid(chunk, grid_id: str, path: str | Path, band: int | None = None) -> Path:
    """A grid of the chunk as GeoTIFF (all its bands) or Surfer GRD (one band) by the extension."""
    g = load_grid(chunk, grid_id)
    path = Path(path)
    if path.suffix.lower() == ".grd":
        return write_grd(g, path, band or 0)
    return write_geotiff(g, path, None if g.crs == "LOCAL" else g.crs)


def outline(grid: Grid):
    from shapely.geometry import Polygon

    return Polygon([tuple(p) for p in grid.corners()])


def export_gpkg(chunk, path: str | Path, window=None, progress=None) -> Path:
    """GeoPackage: the outlines of the georeferenced grids (name, kind, units, cell, file) and
    the anomalies drawn in the window (layer *Anomalie geofisiche*, with their attributes); the
    grids themselves as GeoTIFFs beside it."""
    from pyarchinit_photogrammetry.io import gpkg

    path = Path(path)
    crs = chunk.crs if chunk.crs != "LOCAL" else None
    rows = []
    for k, e in enumerate(entries(chunk)):
        if progress is not None:
            progress(0.1 + 0.6 * k / max(len(entries(chunk)), 1), f"Griglia {e['name']}")
        g = load_grid(chunk, e["id"])
        if not g.georeferenced():
            continue
        tif = path.with_name(f"{path.stem}_{_safe(g.name)}.tif")
        write_geotiff(g, tif, g.crs)
        rows.append((outline(g), {"nome": g.name, "tipo": g.kind, "unita": g.units, "cella_m": g.cell,
                                  "righe": g.shape[0], "colonne": g.shape[1], "bande": g.bands,
                                  "filtri": ", ".join(f["name"] for f in g.filters), "file": tif.name}))
    gpkg.write_table(path, "griglie", "POLYGON", {"nome": "TEXT", "tipo": "TEXT", "unita": "TEXT", "cella_m": "REAL",
                                                   "righe": "INTEGER", "colonne": "INTEGER", "bande": "INTEGER",
                                                   "filtri": "TEXT", "file": "TEXT"}, rows, crs, "griglie geofisiche")
    anomalies = []
    if window is not None:  # the polygons drawn in the layer of the anomalies (attributes with the hidden geometry)
        view = getattr(window, "ortho_view", None)
        feats = view.features(ANOMALIES_LAYER) if view is not None and ANOMALIES_LAYER in getattr(view, "layers", {}) else []
        for f in feats:
            geom = f.get("_geom")
            if geom is None or getattr(geom, "is_empty", True):
                continue
            anomalies.append((geom, {"etichetta": str(f.get("etichetta", "")), "tipo": str(f.get("tipo", "")),
                                     "note": str(f.get("note", ""))}))
    if anomalies:
        gpkg.write_table(path, "anomalie", "POLYGON", {"etichetta": "TEXT", "tipo": "TEXT", "note": "TEXT"}, anomalies, crs,
                         "anomalie geofisiche")
    if progress is not None:
        progress(1.0, str(path))
    return path


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)[:60] or "griglia"


def qgis_entries(chunk) -> tuple[list[dict], list[dict]]:
    """The georeferenced grids as rasters of the QGIS project (the ramp of their kind, the
    limits of the readings), for ``window.qgis_providers``."""
    rasters = []
    for e in entries(chunk):
        if e.get("crs", "LOCAL") == "LOCAL":
            continue
        p = chunk.data_dir / e["file"]
        if not p.exists():
            continue
        st = e.get("stats") or {}
        try:
            g = load_grid(chunk, e["id"])
            lo, hi = limits(g.values[0], g.kind)
        except Exception:  # noqa: BLE001
            lo, hi = st.get("p2", 0.0), st.get("p98", 1.0)
        ramp = "diverging" if e.get("kind") in ("magnetometry", "em") else ("grey" if e.get("kind") == "gpr" else "grey")
        rasters.append({"name": f"Geofisica: {e['name']}", "path": str(p), "ramp": ramp, "limits": [float(lo), float(hi)],
                        "band": 1, "opacity": 0.8, "hidden": True})
    return rasters, []


def summary(chunk) -> str:
    lines = []
    for e in entries(chunk):
        st = e.get("stats") or {}
        where = e.get("crs", "LOCAL")
        lines.append(f"{e['name']}: {e.get('kind')} {e.get('units', '')}, {e['cols']} x {e['rows']} celle di "
                     f"{e['cell']:g} m, {e.get('bands', 1)} banda/e, {where}, mediana {st.get('median', float('nan')):.3g}"
                     + (f", filtri {', '.join(f['name'] for f in e.get('filters', []))}" if e.get("filters") else ""))
    return "\n".join(lines) if lines else "nessuna griglia"


def anomaly_limits_text(grid: Grid, band: int = 0) -> str:
    lo, hi = limits(grid.values[band], grid.kind)
    return f"{lo:.3g} .. {hi:.3g} {grid.units}".strip()


__all__ = ["ANOMALIES_LAYER", "export_grid", "export_gpkg", "qgis_entries", "summary", "anomaly_limits_text", "np"]
