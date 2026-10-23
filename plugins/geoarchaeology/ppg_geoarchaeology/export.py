"""Exports of the geoarchaeology of a chunk.

* :func:`write_layers` - ``<chunk data>/geoarchaeology/sondaggi.gpkg``, kept up to date: the
  boreholes as points with their attributes (table ``sondaggi``), the intervals of the logs as a
  table without geometry (``intervalli``: GeoPackage *attributes*; ``unita`` the unit as shown,
  ``unita_scritta`` its code as written in the log), the traces of the sections (``sezioni``); the
  QGIS project of the program reads it.
* :func:`export_gpkg` - the same into a GeoPackage chosen by the user, with the surfaces and
  thicknesses as GeoTIFFs and the volumes of the model as CSV beside it.
* :func:`export_raster` - one surface or thickness as GeoTIFF.
* :func:`qgis_entries` - the rasters and layers for ``window.qgis_providers`` (also of a chunk in a
  local system).

The GeoPackages are written with the program's ``io.gpkg`` (sqlite3, no GDAL).
"""
from __future__ import annotations

import shutil
from pathlib import Path

from . import boreholes as bh
from . import model as model_mod
from . import section as section_mod
from . import surfaces as sf

LAYERS = "sondaggi.gpkg"
BOREHOLE_COLUMNS = {"id": "TEXT", "x": "REAL", "y": "REAL", "z": "REAL", "z_source": "TEXT", "dem_z": "REAL",
                    "depth": "REAL", "base_z": "REAL", "kind": "TEXT", "units": "TEXT", "intervals": "INTEGER",
                    "source": "TEXT"}
INTERVAL_COLUMNS = {"sondaggio": "TEXT", "da_m": "REAL", "a_m": "REAL", "quota_tetto": "REAL", "quota_base": "REAL",
                    "unita": "TEXT", "unita_scritta": "TEXT", "descrizione": "TEXT", "colore": "TEXT",
                    "litologia": "TEXT"}
SECTION_COLUMNS = {"nome": "TEXT", "lunghezza_m": "REAL", "distanza_m": "REAL", "scala": "INTEGER",
                   "sondaggi": "TEXT", "svg": "TEXT"}


def layers_path(chunk) -> Path:
    return bh.folder(chunk) / LAYERS


def _crs(chunk) -> str | None:
    return sf.chunk_crs(chunk)


def write_attributes(path: Path, table: str, columns: dict[str, str], rows: list[dict], crs: str | None = None,
                     identifier: str | None = None) -> Path:
    """A table without geometry in a GeoPackage (``data_type`` *attributes*), rewritten whole."""
    from pyarchinit_photogrammetry.io import gpkg

    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, gpkg.srs_of(crs))
        con.execute(f"DROP TABLE IF EXISTS {gpkg.quote(table)}")
        con.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (table,))
        cols = ", ".join(f"{gpkg.quote(c)} {kind}" for c, kind in columns.items())
        con.execute(f"CREATE TABLE {gpkg.quote(table)} (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, {cols})")
        con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier) VALUES (?, 'attributes', ?)",
                    (table, identifier or table))
        names = list(columns)
        con.executemany(f"INSERT INTO {gpkg.quote(table)} ({', '.join(gpkg.quote(c) for c in names)}) "
                        f"VALUES ({', '.join('?' * len(names))})", [[r.get(c) for c in names] for r in rows])
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return path


def read_attributes(path: Path, table: str) -> list[dict]:
    import sqlite3

    from pyarchinit_photogrammetry.io import gpkg

    con = sqlite3.connect(str(path))
    try:
        cur = con.execute(f"SELECT * FROM {gpkg.quote(table)}")
        names = [c[0] for c in cur.description]
        return [dict(zip(names, row)) for row in cur]
    finally:
        con.close()


def write_layers(chunk, path: Path | None = None) -> Path:
    """The boreholes, the intervals and the section traces of the chunk into a GeoPackage
    (``sondaggi.gpkg`` of the chunk by default)."""
    from shapely.geometry import LineString, Point

    from pyarchinit_photogrammetry.io import gpkg

    path = Path(path) if path is not None else layers_path(chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    crs = _crs(chunk)
    holes = bh.load(chunk)
    rows = []
    for b in holes:
        rows.append((Point(b.x, b.y), {"id": b.id, "x": b.x, "y": b.y, "z": b.z, "z_source": b.z_source, "dem_z": b.dem_z,
                                       "depth": b.bottom, "base_z": None if b.z is None else b.z - b.bottom,
                                       "kind": b.kind, "units": " / ".join(b.units), "intervals": len(b.intervals),
                                       "source": b.source}))
    gpkg.write_table(path, "sondaggi", "POINT", BOREHOLE_COLUMNS, rows, crs, identifier="sondaggi")
    ivs = []
    for b in holes:
        for iv in b.intervals:
            ivs.append({"sondaggio": b.id, "da_m": iv.top, "a_m": iv.base,
                        "quota_tetto": None if b.z is None else b.z - iv.top,
                        "quota_base": None if b.z is None else b.z - iv.base, "unita": iv.unit,
                        "unita_scritta": iv.unit_raw or iv.unit,
                        "descrizione": iv.description, "colore": iv.colour, "litologia": iv.lithology})
    write_attributes(path, "intervalli", INTERVAL_COLUMNS, ivs, crs, identifier="intervalli")
    secs = [(LineString(s["coords"]), {"nome": s["name"], "lunghezza_m": s["length"], "distanza_m": s["distance"],
                                       "scala": int(s["scale"] or 0), "sondaggi": ", ".join(h["id"] for h in s["boreholes"]),
                                       "svg": s.get("files", {}).get("svg", "")})
            for s in section_mod.entries(chunk) if len(s.get("coords", [])) >= 2]
    gpkg.write_table(path, "sezioni", "LINESTRING", SECTION_COLUMNS, secs, crs, identifier="sezioni geoarcheologiche")
    return path


def export_raster(chunk, pid: str, path: str | Path) -> Path:
    e = sf.entry(chunk, pid)
    if e is None:
        raise KeyError(pid)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(Path(chunk.data_dir) / e["file"], path)
    return path


def export_gpkg(chunk, path: str | Path, window=None, progress=None) -> Path:
    """Boreholes, intervals and sections into ``path``; beside it ``<name>_<surface>.tif`` for every
    surface and thickness and ``<name>_volumi.csv`` when the model was built."""
    path = Path(path)
    if path.exists():
        path.unlink()
    if not bh.load(chunk):
        raise RuntimeError("no borehole to export")
    if progress:
        progress(0.1, "Boreholes, intervals, sections")
    write_layers(chunk, path)
    written = [path]
    for e in sf.entries(chunk):
        src = Path(chunk.data_dir) / e["file"]
        if src.exists():
            written.append(export_raster(chunk, e["id"], path.with_name(f"{path.stem}_{e['id']}.tif")))
    reg = model_mod.registry(chunk)
    csv = reg.get("files", {}).get("csv")
    if csv and (Path(chunk.data_dir) / csv).exists():
        dst = path.with_name(f"{path.stem}_volumi.csv")
        shutil.copy(Path(chunk.data_dir) / csv, dst)
        written.append(dst)
    if progress:
        progress(1.0, f"{len(written)} files")
    return path


def qgis_entries(chunk) -> tuple[list[dict], list[dict]]:
    """The surfaces (grey, the thicknesses in the iron ramp) and the layers of the boreholes and
    of the sections, for the QGIS project of the program. A chunk in a local system has them too,
    as the program does with its own layers: GeoTIFFs and GeoPackage layers without a reference
    system (an undefined cartesian one), in a project without one."""
    rasters, vectors = [], []
    for e in sf.entries(chunk):
        p = Path(chunk.data_dir) / e["file"]
        if not p.exists():
            continue
        st = e.get("stats") or {}
        lims = [st.get("p2", st.get("min", 0.0)), st.get("p98", st.get("max", 1.0))]
        if lims[1] <= lims[0]:
            lims[1] = lims[0] + 1e-3
        thick = e.get("kind") == "thickness"
        rasters.append({"name": f"Geoarcheologia: {e['name']}", "path": str(p), "ramp": "thermal" if thick else "grey",
                        "limits": [float(lims[0]), float(lims[1])], "band": 1, "opacity": 0.8, "hidden": True})
    layers = layers_path(chunk)
    if bh.load(chunk):
        if not layers.exists():
            write_layers(chunk)
        vectors.append({"name": "Geoarcheologia: sondaggi", "path": str(layers), "table": "sondaggi",
                        "geometry": "POINT", "color": "#d62728", "label": "id"})
        if section_mod.entries(chunk):
            vectors.append({"name": "Geoarcheologia: sezioni", "path": str(layers), "table": "sezioni",
                            "geometry": "LINESTRING", "color": "#ff7f0e", "label": "nome"})
    return rasters, vectors


def summary(chunk) -> str:
    lines = [bh.summary(chunk)]
    reg = sf.registry(chunk)
    if reg:
        g = reg["grid"]
        modes = reg["params"].get("modes") or {}
        lines.append(f"surfaces ({reg['made']}{', stale: build them again' if reg.get('stale') else ''}): "
                     f"{g['width']} x {g['height']} cells of {g['cell']:g} m, mode {reg['params']['mode']}"
                     + (" (" + ", ".join(f"{u}={m}" for u, m in modes.items()) + ")" if modes else ""))
        if reg.get("without_dem"):
            lines.append(f"  {reg['without_dem']['area_m2']:.1f} m2 without a DEM: the ground there is the surface of "
                         "the collars")
        for e in reg.get("contacts", []):
            st = e.get("stats") or {}
            how = e.get("method", "")
            if how == "kriging":
                how += f" {e['variogram']['model']}, trend {e.get('trend')}, LOO {e.get('loocv_rmse', 0):.3f} m"
            if e.get("mode") in ("elevation", "depth"):
                why = (e.get("mode_reason") or {}).get("rule", "")
                how += f", by {e['mode']}" + (f" ({why})" if why else "")
            cl = e.get("clamped") or {}
            lines.append(f"  {e['id']}: {how}, {e.get('points', 0)} boreholes, {st.get('min', float('nan')):.3f}.."
                         f"{st.get('max', float('nan')):.3f} m, moved by the stack on {cl.get('cells', 0)} cells "
                         f"(lowered up to {cl.get('lowered_max_m', cl.get('max_m', 0)):.3f} m, raised up to "
                         f"{cl.get('raised_max_m', 0):.3f} m)")
        for e in reg.get("thickness", []):
            st = e.get("stats") or {}
            line = (f"  {e['id']}: mean {st.get('mean', 0):.3f} m, max {st.get('max', 0):.3f} m, "
                    f"{e.get('volume_m3', 0):.2f} m3")
            if e.get("outside_dem"):
                line += f" (+ {e['outside_dem']['volume_m3']:.2f} m3 where there is no DEM)"
            if e.get("absent"):
                line += f", missing on {e['absent']['area_m2']:.1f} m2"
            lines.append(line)
    for s in section_mod.entries(chunk):
        lines.append(f"section {s['id']} {s['name']!r}: {s['length']:.2f} m, 1:{s['scale']:g}, "
                     f"{len(s['boreholes'])} boreholes")
    reg = model_mod.registry(chunk)
    if reg:
        lines.append(f"model ({reg['made']}): cell {reg['cell']:g} m, floor {reg.get('floor') or 0:.3f} m")
        for r in reg.get("units", []):
            lines.append(f"  {r['unit']}: {r['volume_m3']:.3f} m3 on {r.get('area_m2', 0):.1f} m2"
                         + (f" (+ {r['outside_dem']['volume_m3']:.3f} m3 where there is no DEM)"
                            if r.get("outside_dem") else "")
                         + (" (clipped to the US drawn)" if r.get("clipped") else "")
                         + (" (down to the floor)" if r.get("to_floor") else ""))
    return "\n".join(lines)
