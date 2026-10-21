"""The powerlines found (``chunk.meta["powerlines"]``) as files and as layers of the window:
a GeoPackage with the conductors as 3D lines and the poles as 3D points, a DXF with 3D
polylines, a CSV table; the layers *Linee elettriche* and *Pali* on the orthomosaic and DEM
views; a table in words."""
from __future__ import annotations

import csv
import struct
from pathlib import Path

import numpy as np

from .texts import t

WIRES_LAYER = "Linee elettriche"
POLES_LAYER = "Pali"
WIRE_COLOUR, POLE_COLOUR = "#ff2d9b", "#ff9f1a"
WIRE_COLUMNS = {"id": "INTEGER", "linea": "INTEGER", "lunghezza_m": "REAL", "freccia_m": "REAL", "franco_m": "REAL",
                "ostacolo_m": "REAL", "z_min_m": "REAL", "z_max_m": "REAL", "azimut_gradi": "REAL",
                "punti": "INTEGER", "rms_m": "REAL"}
POLE_COLUMNS = {"id": "INTEGER", "altezza_m": "REAL", "z_terreno_m": "REAL", "conduttori": "TEXT", "linee": "TEXT",
                "rilevato": "INTEGER", "punti": "INTEGER"}


def wire_row(w: dict) -> dict:
    """The attributes of a wire, with the column names of the exports."""
    return {"id": w["id"], "linea": w["line"], "lunghezza_m": w["length_m"], "freccia_m": w["sag_m"],
            "franco_m": w["clearance_m"], "ostacolo_m": w["obstacle_m"], "z_min_m": w["z_min_m"], "z_max_m": w["z_max_m"],
            "azimut_gradi": w["azimuth_deg"], "punti": w["points"], "rms_m": w["rms_m"]}


def pole_row(p: dict) -> dict:
    return {"id": p["id"], "altezza_m": p["height_m"], "z_terreno_m": p["z_ground"],
            "conduttori": ",".join(str(i) for i in p["wires"]), "linee": ",".join(str(i) for i in p["lines"]),
            "rilevato": int(bool(p["detected"])), "punti": p["points"]}


def _report(chunk) -> dict:
    report = chunk.meta.get("powerlines")
    if not report:
        raise RuntimeError(t("Nessuna linea elettrica in questo chunk: prima Plugin > Linee elettriche > Rileva.",
                             "No powerlines in this chunk: first Plugin > Powerlines > Detect."))
    return report


# ---------------------------------------------------------------------------------- GeoPackage
def _blob_z(geom, srs_id: int) -> bytes:
    """GeoPackage binary geometry with Z: header (little endian, xyz envelope) + ISO WKB."""
    import shapely

    coords = shapely.get_coordinates(geom, include_z=True)
    lo, hi = coords.min(0), coords.max(0)
    head = b"GP" + bytes([0, 0b00000101]) + struct.pack("<i", srs_id) + struct.pack("<6d", lo[0], hi[0], lo[1], hi[1],
                                                                                      lo[2], hi[2])
    return head + shapely.to_wkb(geom, byte_order=1, output_dimension=3, flavor="iso")


def _table_z(con, table: str, geometry_type: str, columns: dict, srs_id: int, features) -> int:
    """A feature table with 3D geometries, rewritten; returns the rows written."""
    from pyarchinit_photogrammetry.io.gpkg import quote

    have = [r[1] for r in con.execute(f"PRAGMA table_info({quote(table)})")]
    if not have:
        cols = "".join(f", {quote(c)} {typ}" for c, typ in columns.items())
        con.execute(f"CREATE TABLE {quote(table)} (fid INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL, "
                    f"geom {geometry_type}{cols})")
        con.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, srs_id) VALUES (?, 'features', ?, ?)",
                    (table, table, srs_id))
        con.execute("INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', ?, ?, 1, 0)", (table, geometry_type, srs_id))
    rows, bounds = [], []
    for geom, values in features:
        rows.append([_blob_z(geom, srs_id)] + [values.get(c) for c in columns])
        bounds.append(geom.bounds)
    con.execute(f"DELETE FROM {quote(table)}")
    names = "".join(f", {quote(c)}" for c in columns)
    con.executemany(f"INSERT INTO {quote(table)} (geom{names}) VALUES ({', '.join('?' * (len(columns) + 1))})", rows)
    ext = (min(b[0] for b in bounds), min(b[1] for b in bounds), max(b[2] for b in bounds),
           max(b[3] for b in bounds)) if bounds else (None, None, None, None)
    con.execute("UPDATE gpkg_contents SET last_change = strftime('%Y-%m-%dT%H:%M:%fZ','now'), "
                "min_x = ?, min_y = ?, max_x = ?, max_y = ? WHERE table_name = ?", (*ext, table))
    return len(rows)


def write_geopackage(report: dict, path, crs: str | None) -> Path:
    """Tables ``linee_elettriche`` (LINESTRING Z, one per conductor, vertices every metre) and
    ``pali`` (POINT Z at the ground) in a GeoPackage (created or, if it is one, kept with its
    other tables), in the CRS given (a local frame: undefined cartesian)."""
    from shapely.geometry import LineString, Point

    from pyarchinit_photogrammetry.io import gpkg

    path = Path(path)
    srs = gpkg.srs_of(crs)
    if not gpkg.is_geopackage(path) and path.exists():
        path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, srs)
        _table_z(con, "linee_elettriche", "LINESTRING", WIRE_COLUMNS, srs[0],
                 [(LineString(w["vertices"]), wire_row(w)) for w in report["wires"]])
        _table_z(con, "pali", "POINT", POLE_COLUMNS, srs[0],
                 [(Point(p["x"], p["y"], p["z_ground"]), pole_row(p)) for p in report["poles"]])
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return path


# ----------------------------------------------------------------------------------------- DXF
def write_dxf(report: dict, path) -> Path:
    """3D polylines of the conductors on the layer ``LINEE_ELETTRICHE`` (metres, world
    coordinates), the poles as vertical lines on ``PALI``, labels ``L1``, ``P1``..."""
    import ezdxf

    doc = ezdxf.new("R2010")
    doc.units = ezdxf.units.M
    doc.layers.add("LINEE_ELETTRICHE", color=6)
    doc.layers.add("PALI", color=2)
    doc.layers.add("ETICHETTE", color=7)
    msp = doc.modelspace()
    for w in report["wires"]:
        msp.add_polyline3d(w["vertices"], dxfattribs={"layer": "LINEE_ELETTRICHE"})
        x, y, z = w["vertices"][len(w["vertices"]) // 2]
        msp.add_text(f"L{w['id']}", dxfattribs={"layer": "ETICHETTE", "height": 1.0}).set_placement((x, y, z))
    for p in report["poles"]:
        msp.add_line((p["x"], p["y"], p["z_ground"]), (p["x"], p["y"], p["z_ground"] + p["height_m"]),
                     dxfattribs={"layer": "PALI"})
        msp.add_text(f"P{p['id']}", dxfattribs={"layer": "ETICHETTE", "height": 1.0}).set_placement(
            (p["x"] + 0.5, p["y"] + 0.5, p["z_ground"]))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.saveas(path)
    return path


# ----------------------------------------------------------------------------------------- CSV
def write_csv(report: dict, path) -> Path:
    """One row per conductor in ``path``; the poles in ``<name>_pali.csv`` beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(WIRE_COLUMNS) + ["x_inizio", "y_inizio", "z_inizio", "x_fine",
                                                                      "y_fine", "z_fine"])
        writer.writeheader()
        for w in report["wires"]:
            (x0, y0, z0), (x1, y1, z1) = w["start"], w["end"]
            writer.writerow({**wire_row(w), "x_inizio": x0, "y_inizio": y0, "z_inizio": z0, "x_fine": x1, "y_fine": y1,
                             "z_fine": z1})
    poles = path.with_name(path.stem + "_pali.csv")
    with open(poles, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "x", "y", "z_terreno_m", "altezza_m", "conduttori", "linee",
                                               "rilevato", "punti"])
        writer.writeheader()
        for p in report["poles"]:
            writer.writerow({**pole_row(p), "x": p["x"], "y": p["y"]})
    return path


def export(chunk, path, progress=None) -> Path:
    """The powerlines of ``chunk`` as ``.gpkg``, ``.dxf`` or ``.csv`` by the suffix of ``path``."""
    report = _report(chunk)
    path = Path(path)
    suffix = path.suffix.lower()
    if progress is not None:
        progress(0.1, t("Scrittura di {0}", "Writing {0}").format(path.name))
    if suffix == ".gpkg":
        out = write_geopackage(report, path, chunk.crs)
    elif suffix == ".dxf":
        out = write_dxf(report, path)
    elif suffix == ".csv":
        out = write_csv(report, path)
    else:
        raise ValueError(t("Formato {0} non previsto: .gpkg, .dxf o .csv", "Format {0} not known: .gpkg, .dxf or .csv")
                         .format(suffix or path.name))
    if progress is not None:
        progress(1.0, str(out))
    return out


# --------------------------------------------------------------------------------------- words
def table(report: dict) -> str:
    """The conductors and the poles in words, one line each."""
    lines = [t("{0} conduttori in {1} linee, {2} pali ({3} con punti); terreno dai {4}; {5:,} punti di conduttore",
               "{0} wires in {1} lines, {2} poles ({3} with points); ground from {4}; {5:,} wire points")
             .format(len(report["wires"]), report["lines"], len(report["poles"]),
                     sum(1 for p in report["poles"] if p["detected"]),
                     t("punti di classe 2", "class 2 points") if report["ground"] == "classes"
                     else t("filtro morfologico", "the morphological filter"), report["wire_points"])]
    head = t("  id  linea  lunghezza  freccia  franco  ostacolo  azimut  punti", "  id  line  length  sag  clearance  obstacle  azimuth  points")
    lines.append(head)
    for w in report["wires"]:
        obstacle = "-" if w["obstacle_m"] is None else f"{w['obstacle_m']:.2f}"
        lines.append(f"  L{w['id']:<3} {w['line']:>4}  {w['length_m']:8.1f} m  {w['sag_m']:5.2f} m  {w['clearance_m']:5.2f} m"
                     f"  {obstacle:>6} m  {w['azimuth_deg']:5.1f}  {w['points']:6d}")
    for p in report["poles"]:
        lines.append(t("  P{0}: altezza {1:.1f} m, conduttori {2}{3}", "  P{0}: height {1:.1f} m, wires {2}{3}")
                     .format(p["id"], p["height_m"], ", ".join(f"L{i}" for i in p["wires"]),
                             "" if p["detected"] else t(" (posizione dagli estremi dei conduttori)",
                                                        " (placed from the wire ends)")))
    return "\n".join(lines)


def summary(report: dict) -> str:
    """A few lines for the message of the window."""
    wires_ = report["wires"]
    if not wires_:
        return t("Nessun conduttore trovato. Serve una nuvola densa in cui i cavi siano ricostruiti: voli vicini "
                 "(ispezione) e mappe di profondita' di qualita' alta; altrimenti abbassa l'altezza minima o la linearita'.",
                 "No wire found. The dense cloud must hold the cables: close (inspection) flights and high quality "
                 "depth maps; else lower the minimum height or the linearity.")
    total = sum(w["length_m"] for w in wires_)
    lowest = min(wires_, key=lambda w: w["clearance_m"])
    near = [w for w in wires_ if w["obstacle_m"] is not None]
    text = t("{0} conduttori in {1} linee, {2:.0f} m in tutto, {3} pali.\nFranco minimo dal suolo {4:.2f} m (L{5}).",
             "{0} wires in {1} lines, {2:.0f} m in all, {3} poles.\nLowest clearance {4:.2f} m (L{5}).") \
        .format(len(wires_), report["lines"], total, len(report["poles"]), lowest["clearance_m"], lowest["id"])
    if near:
        closest = min(near, key=lambda w: w["obstacle_m"])
        text += t("\nOstacolo piu' vicino a {0:.2f} m (L{1}).", "\nNearest obstacle at {0:.2f} m (L{1}).") \
            .format(closest["obstacle_m"], closest["id"])
    if report.get("classified"):
        text += t("\nI punti dei conduttori hanno la classe 14, quelli dei pali la 15 (vista 3D: colori per classe).",
                  "\nWire points have class 14, pole points class 15 (3D view: colours by class).")
    return text


# ------------------------------------------------------------------------------------- window
def draw(window, report: dict | None) -> None:
    """The conductors and the poles as line layers of the raster views of ``window``
    (kept across DEM and orthomosaic reloads); ``None`` removes them."""
    view = window.ortho_view
    for name in (WIRES_LAYER, POLES_LAYER):
        window.line_layers.pop(name, None)
        view.remove_layer(name)
    if not report:
        return
    add = window._add_line_feature
    with view.batch():
        for w in report["wires"]:
            attrs = {k: ("" if v is None else v) for k, v in wire_row(w).items()}
            add(WIRES_LAYER, np.asarray(w["vertices"], float)[:, :2], WIRE_COLOUR, label=f"L{w['id']}", attrs=attrs,
                width=2.0)
        for p in report["poles"]:
            x, y, r = p["x"], p["y"], 1.0
            attrs = pole_row(p)
            add(POLES_LAYER, [[x - r, y - r], [x + r, y + r]], POLE_COLOUR, label=f"P{p['id']}", attrs=attrs, width=2.5)
            add(POLES_LAYER, [[x - r, y + r], [x + r, y - r]], POLE_COLOUR, attrs=attrs, width=2.5)
