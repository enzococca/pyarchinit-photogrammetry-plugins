"""Exports of the geophysical grids: GeoTIFF (all the bands) and Surfer GRD (one band, north up)
of a grid, a GeoPackage with the outlines of the grids and the anomalies drawn (table
``anomalie``), the entries for the QGIS project of the program, the summary of the grids in the
language of the interface."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import anomalies
from .grids import Grid, entries, load_grid, write_geotiff, write_grd
from .overlay import limits
from .texts import t

ANOMALIES_LAYER = anomalies.LAYER
KIND_NAMES = {"magnetometry": ("magnetometria", "magnetometry"), "resistivity": ("resistività", "resistivity"),
              "gpr": ("GPR", "GPR"), "em": ("elettromagnetico", "electromagnetic"),
              "susceptibility": ("suscettività", "susceptibility"), "other": ("altro", "other")}


def export_grid(chunk, grid_id: str, path: str | Path, band: int | None = None) -> Path:
    """A grid of the chunk by the extension of ``path``: GeoTIFF with all its bands (and the rotation
    of its placement), or Surfer GRD of the band ``band`` (counted from 0), a turned grid resampled
    north up because a Surfer grid has no rotation (:func:`.grids.write_grd`)."""
    g = load_grid(chunk, grid_id)
    path = Path(path)
    if path.suffix.lower() == ".grd":
        b = int(band or 0)
        if not 0 <= b < g.bands:
            raise ValueError(f"the grid {g.name} has {g.bands} band(s): no band {b}")
        return write_grd(g, path, b)
    return write_geotiff(g, path, None if g.crs == "LOCAL" else g.crs)


def outline(grid: Grid):
    from shapely.geometry import Polygon

    return Polygon([tuple(p) for p in grid.corners()])


def export_gpkg(chunk, path: str | Path, window=None, progress=None) -> Path:
    """GeoPackage: the outlines of the georeferenced grids (table ``griglie``: name, kind, units,
    cell, rows, columns, bands, filters, file; the grids as GeoTIFFs beside it) and the anomalies
    drawn (table ``anomalie``: label, grid, band, note, area in m²), from the project - the window
    is not needed. The other tables of an existing GeoPackage are kept."""
    from pyarchinit_photogrammetry.io import gpkg

    path = Path(path)
    crs = chunk.crs if chunk.crs != "LOCAL" else None
    rows = []
    for k, e in enumerate(entries(chunk)):
        if progress is not None:
            progress(0.1 + 0.6 * k / max(len(entries(chunk)), 1), t("Griglia {0}", "Grid {0}").format(e["name"]))
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
    found = [(anomalies.geometry(a), {"etichetta": a.get("etichetta", ""), "griglia": a.get("griglia", ""),
                                      "banda": a.get("banda", ""), "note": a.get("note", ""),
                                      "area_m2": float(a.get("area_m2", 0.0))})
             for a in anomalies.entries(chunk) if len(a.get("coords") or []) >= 3]
    # always written (empty when none): an older table of the same file never survives the export
    gpkg.write_table(path, "anomalie", "POLYGON", {"etichetta": "TEXT", "griglia": "TEXT", "banda": "TEXT",
                                                    "note": "TEXT", "area_m2": "REAL"}, found, crs, "anomalie geofisiche")
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
        ramp = "diverging" if e.get("kind") in ("magnetometry", "em") else "grey"
        rasters.append({"name": f"{t('Geofisica', 'Geophysics')}: {e['name']}", "path": str(p), "ramp": ramp,
                        "limits": [float(lo), float(hi)], "band": 1, "opacity": 0.8, "hidden": True})
    return rasters, []


def notes_text(e: dict) -> str:
    """How the readings of a grid became its cells (an XYZ table), in the language of the interface."""
    n = e.get("notes") or {}
    parts = []
    if n.get("interpolated"):
        lines = t("righe", "rows") if n["interpolated"] == "rows" else t("colonne", "columns")
        fine = min(float(n.get("spacing_x", 0.0)), float(n.get("spacing_y", 0.0)))
        parts.append(t("{0} interpolate fra traverse a {1:g} m (letture ogni {2:g} m)",
                       "{0} interpolated between traverses {1:g} m apart (readings every {2:g} m)")
                     .format(lines, float(n.get("traverse_spacing", 0.0)), fine))
    if n.get("averaged"):
        parts.append(t("letture mediate nelle celle", "readings averaged in the cells"))
    if n.get("dummies"):
        parts.append(t("{0} letture fittizie tolte", "{0} dummy readings left out").format(int(n["dummies"])))
    if n.get("decimal_comma"):
        parts.append(t("virgola decimale", "decimal comma"))
    return "; ".join(parts)


def summary_line(e: dict) -> str:
    """One line on a grid of the registry, in the language of the interface."""
    st = e.get("stats") or {}
    bands = int(e.get("bands", 1))
    where = e.get("crs", "LOCAL")
    text = t("{0}: {1} {2}, {3} x {4} celle di {5:g} m, {6} {7}, {8}, mediana {9:.3g}",
             "{0}: {1} {2}, {3} x {4} cells of {5:g} m, {6} {7}, {8}, median {9:.3g}").format(
        e["name"], t(*KIND_NAMES.get(e.get("kind", "other"), (e.get("kind", ""), e.get("kind", "")))),
        e.get("units", ""), e["cols"], e["rows"], float(e["cell"]), bands,
        t("banda", "band") if bands == 1 else t("bande", "bands"), t("locale", "local") if where == "LOCAL" else where,
        st.get("median", float("nan")))
    if e.get("filters"):
        text += t(", filtri {0}", ", filters {0}").format(", ".join(f["name"] for f in e["filters"]))
    note = notes_text(e)
    return text + (f" ({note})" if note else "")


def summary(chunk) -> str:
    """The grids of the chunk, a line each, and the anomalies drawn: in the language of the interface."""
    lines = [summary_line(e) for e in entries(chunk)]
    n = len(anomalies.entries(chunk))
    if n:
        lines.append(t("Anomalie disegnate: {0}", "Anomalies drawn: {0}").format(n))
    return "\n".join(lines) if lines else t("nessuna griglia", "no grid")


def anomaly_limits_text(grid: Grid, band: int = 0) -> str:
    lo, hi = limits(grid.values[band], grid.kind)
    return f"{lo:.3g} .. {hi:.3g} {grid.units}".strip()


__all__ = ["ANOMALIES_LAYER", "export_grid", "export_gpkg", "qgis_entries", "summary", "summary_line", "notes_text",
           "anomaly_limits_text", "np"]
