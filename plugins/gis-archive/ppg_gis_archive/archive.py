"""Archiving the survey of a project into a GIS database outside PyArchInit: a GeoPackage, a
PostgreSQL/PostGIS database or a SpatiaLite file.

What goes in, each in its table ``<prefix><name>`` (``ppg_`` by default):

* ``quote`` - the spot heights drawn (points: height, area, US and the other fields of
  ``pyarchinit_quote``);
* ``curve`` - contour lines of the DEM at the interval asked for (lines: height, index lines);
* ``us`` - the stratigraphic units drawn (multipolygons: site, area, US, type, the fields of
  ``pyunitastratigrafiche``, area in square metres);
* ``sezioni`` - the traces of the sections (lines: name, length, drawing). The sections belong to
  the project (``sezioni/sezioni.json`` beside it), not to a chunk: they are written once per
  project, with no chunk;
* ``pietre`` - the stones drawn with SAM on the orthomosaic (polygons: group, area). Stones drawn
  on the elevation of a wall are in the coordinates of the elevation (metres along the wall and
  up), not of the map: they are left out, and the report says so;
* ``marker`` - markers and control points (points: height, reference, error);
* ``raster`` - the DEM and the orthomosaic: footprint, file, the resolution and size of what is
  archived (the copy, when it is reduced), and the table that holds the raster itself when it is
  copied (tiles of a GeoPackage, PostGIS raster tiles).

Every row carries ``progetto`` (the file name of the project), ``chunk``, ``sito`` and
``archiviato`` (date and time), and what says whose it is: ``progetto_id`` - an id the project
keeps in ``doc.meta["gis_archive"]``, made at its first archive (a project without it takes the
id of the rows written from the same project file; a copy of a project, made with *Save as* or by
copying its folder, gets an id of its own) - ``progetto_file`` (the path of the project file) and
``chunk_id`` (the key of the chunk, which a new name of the chunk does not change). Archiving the
same project and chunk again replaces its rows and leaves those of every other project, also of
another project with the same file name. The rows written by version 0.1.0 of the plugin, which
have no id, are taken as the rows of the project of the same file name, unless the rasters they
record lie in the folder of another project file.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import struct
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from . import t

LAYERS = ("quote", "curve", "us", "sezioni", "pietre", "marker")
RASTERS = ("dem", "ortofoto")
EVERYTHING = LAYERS + RASTERS
TILE = 256  # side of the raster tiles written into PostGIS
ID_KEY = "gis_archive"  # doc.meta[ID_KEY] = {"id": ..., "file": ...}: who the project is in the archives
MAX_NAME = 63  # PostgreSQL cuts longer table names
ELEVATION_FILES = ("elevation.tif", "relief.tif")  # the rasters of a wall elevation (engine.elevation)


class ArchiveError(Exception):
    """The archive could not be written; the message says why."""


COMMON = {"progetto": "TEXT", "chunk": "TEXT", "sito": "TEXT", "archiviato": "TEXT", "progetto_id": "TEXT",
          "progetto_file": "TEXT", "chunk_id": "INTEGER"}


@dataclass
class Key:
    """Which rows of an archive are those of the project and chunk archived now: the rows with
    ``progetto_id`` = ``pid`` (and ``chunk_id``), and, when ``legacy``, the rows of version 0.1.0
    of the plugin (no id) with the file name ``name`` of the project (and the label ``chunk``)."""

    pid: str
    name: str
    chunk: str
    chunk_id: int
    legacy: bool = True

    def where(self, scope: str = "chunk", kinds=None, ids: bool = True) -> tuple[str, dict]:
        """``(SQL condition, parameters)`` of these rows, with named parameters as both sqlite3
        and SQLAlchemy take them. ``scope``: ``"chunk"`` (the rows of this chunk) or
        ``"project"`` (of the whole project: the sections); ``kinds``: only these values of
        ``tipo``; ``ids``: the table has the columns ``progetto_id`` and ``chunk_id`` (a table
        written by version 0.1.0 has not, until the next archive adds them)."""
        per_chunk = scope == "chunk"
        terms, params = [], {}
        if ids:
            terms.append("(COALESCE(progetto_id, '') = :pid" + (" AND COALESCE(chunk_id, -1) = :cid)" if per_chunk else ")"))
            params["pid"] = self.pid
            if per_chunk:
                params["cid"] = int(self.chunk_id)
        if self.legacy:
            old = "COALESCE(progetto, '') = :name" + (" AND COALESCE(chunk, '') = :label" if per_chunk else "")
            terms.append(f"(COALESCE(progetto_id, '') = '' AND {old})" if ids else f"({old})")
            params["name"] = self.name
            if per_chunk:
                params["label"] = self.chunk
        cond = " OR ".join(terms) if terms else "1 = 0"
        if kinds:
            params.update({f"k{i}": k for i, k in enumerate(kinds)})
            cond = f"({cond}) AND tipo IN ({', '.join(f':k{i}' for i in range(len(kinds)))})"
        return cond, params


@dataclass
class Layer:
    """A table to write: ``features`` are ``(shapely geometry, {column: value})``. Archiving again
    replaces the rows of ``key`` in ``scope`` (``"chunk"`` or ``"project"``), only those whose
    ``tipo`` is among ``kinds`` when given."""

    name: str
    geometry: str  # POINT, LINESTRING, POLYGON, MULTIPOLYGON
    columns: dict  # column -> TEXT, REAL or INTEGER
    features: list = field(default_factory=list)
    key: Key | None = None
    scope: str = "chunk"
    kinds: tuple | None = None

    def where(self, ids: bool = True) -> tuple[str, dict]:
        return self.key.where(self.scope, self.kinds, ids)


def slug(text: str) -> str:
    """A table name made of lowercase letters, digits and underscores."""
    s = re.sub(r"[^a-z0-9]+", "_", str(text).lower()).strip("_")
    return s or "x"


def ident(name: str) -> str:
    """``name`` as a quoted SQL identifier."""
    return '"' + str(name).replace('"', '""') + '"'


def _number(value):
    try:
        return None if value in (None, "") else float(value)
    except (TypeError, ValueError):
        return None


def _multi(geom):
    from shapely.geometry import MultiPolygon, Polygon

    if isinstance(geom, Polygon):
        return MultiPolygon([geom])
    if isinstance(geom, MultiPolygon):
        return geom
    polys = [g for g in getattr(geom, "geoms", []) if isinstance(g, Polygon)]
    return MultiPolygon(polys) if polys else None


def _resolved(path) -> Path:
    try:
        return Path(path).expanduser().resolve()
    except (OSError, RuntimeError):
        return Path(os.path.abspath(str(path)))


# ---------------------------------------------------------------------------- who the project is
def project_name(doc) -> str:
    """The file name of the project without ``.ppg`` (``senza_nome`` when never saved)."""
    return Path(doc.path).stem if getattr(doc, "path", None) else "senza_nome"


def project_file(doc) -> str:
    """The path of the project file (empty when never saved)."""
    return str(doc.path) if getattr(doc, "path", None) else ""


def project_key(doc, chunk) -> tuple:
    """``(progetto, chunk)``: the file name of the project and the label of the chunk, as the rows
    show them (the rows are told apart by :class:`Key`)."""
    return project_name(doc), chunk.label


def _holds_id(path: str, pid: str) -> bool:
    """Whether the project file ``path`` exists and keeps ``pid`` as its archive id."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return ((data.get("meta") or {}).get(ID_KEY) or {}).get("id") == pid
    except (OSError, ValueError, AttributeError, TypeError):
        return False


def stored_id(doc) -> str | None:
    """The archive id the project keeps (``doc.meta["gis_archive"]["id"]``), or None: never
    archived, or a copy - made with *Save as* or by copying the folder - of a project file that
    is still there and keeps the same id: a copy is another project."""
    info = (getattr(doc, "meta", None) or {}).get(ID_KEY)
    pid = info.get("id") if isinstance(info, dict) else None
    if not pid:
        return None
    here, there = project_file(doc), str(info.get("file") or "")
    if here and there and _resolved(here) != _resolved(there) and _holds_id(there, str(pid)):
        return None
    return str(pid)


def store_identity(path, identity: dict) -> None:
    """Write the archive id into the project file ``path`` and nothing else (the program may have
    saved the project meanwhile: its other contents are left as they are)."""
    p = Path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    data.setdefault("meta", {})[ID_KEY] = dict(identity)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    tmp.replace(p)


def _file_of_project(path: str, doc) -> bool:
    """Whether a raster recorded in an archive is a file of the project ``doc``: inside its
    ``.ppg.files`` folder, or the DEM or orthomosaic of one of its chunks."""
    if not path or not getattr(doc, "path", None):
        return False
    p = _resolved(path)
    try:
        p.relative_to(_resolved(str(doc.path) + ".files"))
        return True
    except ValueError:
        pass
    for c in getattr(doc, "chunks", []):
        for q in (c.dem_path, c.orthomosaic_path):
            if q is not None and _resolved(q) == p:
                return True
    return False


def _id_from_rows(db, tables, file: str) -> str | None:
    """The id of the rows written from the project file ``file`` (a project whose id was not
    saved with it finds its own rows again)."""
    for table in tables:
        if {"progetto_id", "progetto_file"} <= db.columns(table):
            rows = db.select(table, "progetto_id", "progetto_file = :f AND COALESCE(progetto_id, '') <> ''",
                             {"f": file})
            if rows:
                return str(rows[0][0])
    return None


def _legacy_rows_are_ours(db, raster_table: str, doc, name: str) -> tuple[bool, str]:
    """Whether the rows of version 0.1.0 (no id) with the file name of this project are its own:
    yes, unless the rasters they record lie in the folder of another project file (then that
    file is returned too)."""
    cols = db.columns(raster_table)
    if not {"progetto", "file"} <= cols:
        return True, ""
    cond = "COALESCE(progetto, '') = :name" + (" AND COALESCE(progetto_id, '') = ''" if "progetto_id" in cols else "")
    files = [str(r[0]) for r in db.select(raster_table, "file", cond, {"name": name}) if r[0]]
    if not files or any(_file_of_project(f, doc) for f in files):
        return True, ""
    return False, files[0]


def identify(db, doc, chunk, prefix: str = "ppg_") -> tuple[Key, dict, list]:
    """The :class:`Key` of the rows of ``chunk`` of ``doc`` in the database ``db``, the identity to
    keep in the project (``{"id", "file"}``) and notes for the report."""
    name, here = project_name(doc), project_file(doc)
    tables = [f"{prefix}{n}" for n in (*LAYERS, "raster")]
    pid = stored_id(doc)
    if pid is None and here:
        pid = _id_from_rows(db, tables, here)
    pid = pid or uuid.uuid4().hex
    legacy, other = _legacy_rows_are_ours(db, f"{prefix}raster", doc, name)
    notes = []
    if not legacy:
        notes.append(t(f"le righe archiviate dalla versione 0.1.0 con il nome '{name}' sono di un altro progetto "
                       f"({other}): restano accanto a quelle di questo",
                       f"the rows archived by version 0.1.0 under the name '{name}' belong to another project "
                       f"({other}): they stay beside the rows of this one"))
    return Key(pid, name, chunk.label, int(chunk.key), legacy), {"id": pid, "file": here}, notes


# ---------------------------------------------------------------------------- what the project holds
def _elevation_images(chunk) -> set:
    """Names of the rasters of the wall elevations of ``chunk``: rasters in metres along the wall
    and up, without a reference system. The stones are kept by the name of the image they were
    drawn on (``elevation.tif`` for every elevation)."""
    names = set(ELEVATION_FILES)
    for e in chunk.meta.get("elevations") or []:
        for k in ("orthophoto", "relief"):
            if isinstance(e, dict) and e.get(k):
                names.add(Path(str(e[k])).name)
    maps = {p.name for p in (chunk.orthomosaic_path, chunk.dem_path) if p is not None}
    return names - maps


def collect(doc, chunk, what=EVERYTHING, interval: float | None = None, progress=None, key: Key | None = None,
            notes: list | None = None) -> tuple[list, list]:
    """The tables (:class:`Layer`) and the rasters (``{"kind", "path"}``) of ``chunk``; ``key``:
    whose rows they are (by default the id kept in the project, or a new one); ``notes`` gets
    what is left out and why."""
    from shapely import wkt as shapely_wkt
    from shapely.geometry import LineString, Point

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UNIT_FIELDS, UnitPolygon

    progress = progress or (lambda *a: None)
    what = set(what)
    key = key or Key(stored_id(doc) or uuid.uuid4().hex, project_name(doc), chunk.label, int(chunk.key))
    notes = notes if notes is not None else []
    base = {"progetto": key.name, "chunk": chunk.label, "sito": (doc.meta or {}).get("sito") or "",
            "archiviato": datetime.now().isoformat(timespec="seconds"), "progetto_id": key.pid,
            "progetto_file": project_file(doc), "chunk_id": int(chunk.key)}
    drawings = chunk.meta.get("drawings") or {}
    layers: list[Layer] = []
    if "quote" in what:
        layer = Layer("quote", "POINT", {**COMMON, "area": "TEXT", "us": "TEXT", "tipo": "TEXT", "quota": "REAL",
                                         "unita_misura": "TEXT", "precisione": "REAL", "rilievo": "TEXT",
                                         "data": "TEXT", "disegnatore": "TEXT"})
        for q in drawings.get("quote", []):
            if q.get("x") is None or q.get("y") is None:
                continue
            layer.features.append((Point(float(q["x"]), float(q["y"])), {
                **base, "sito": q.get("sito_q") or base["sito"], "area": q.get("area_q"), "us": q.get("us_q"),
                "tipo": q.get("unita_tipo_q"), "quota": _number(q.get("quota_q")), "unita_misura": q.get("unita_misu_q"),
                "precisione": _number(q.get("precisione_q")), "rilievo": q.get("rilievo_originale"),
                "data": q.get("data"), "disegnatore": q.get("disegnatore")}))
        layers.append(layer)
    if "us" in what:
        extra = {k: ("INTEGER" if k == "stratigraph_index_us" else "TEXT") for k in UNIT_FIELDS}
        layer = Layer("us", "MULTIPOLYGON", {**COMMON, "area": "TEXT", "us": "TEXT", "tipo": "TEXT", **extra,
                                             "area_m2": "REAL", "layer": "TEXT"})
        for d in drawings.get("units", []):
            unit = UnitPolygon.from_dict(d)
            geom = _multi(unit.geometry)
            if geom is None or geom.is_empty:
                continue
            attrs = unit.attributes or {}
            layer.features.append((geom, {
                **base, "sito": unit.sito or base["sito"], "area": unit.area, "us": unit.us, "tipo": unit.tipo,
                **{k: attrs.get(k) for k in UNIT_FIELDS}, "stratigraph_index_us": unit.stratigraphic_index,
                "area_m2": round(float(geom.area), 4), "layer": d.get("layer")}))
        layers.append(layer)
    if "pietre" in what:
        layer = Layer("pietre", "POLYGON", {**COMMON, "gruppo": "TEXT", "area_m2": "REAL"})
        elevations = _elevation_images(chunk)
        left: dict[str, int] = {}
        for group, items in (chunk.meta.get("stones") or {}).items():
            if group in elevations:  # in the frame of a wall elevation, not of the map
                left[group] = left.get(group, 0) + len(items)
                continue
            for text in items:
                geom = shapely_wkt.loads(text)
                for poly in getattr(geom, "geoms", [geom]):
                    if poly.geom_type == "Polygon" and not poly.is_empty:
                        layer.features.append((poly, {**base, "gruppo": group, "area_m2": round(float(poly.area), 5)}))
        if left:
            n = sum(left.values())
            notes.append(t(f"{n} pietre disegnate sul prospetto di un muro ({', '.join(sorted(left))}) non archiviate: "
                           "sono nelle coordinate del prospetto (metri lungo il muro e in altezza), non della mappa",
                           f"{n} stones drawn on the elevation of a wall ({', '.join(sorted(left))}) not archived: "
                           "they are in the coordinates of the elevation (metres along the wall and up), not of the map"))
        layers.append(layer)
    if "sezioni" in what:
        # the sections belong to the project, not to a chunk: one set of rows per project
        layer = Layer("sezioni", "LINESTRING", {**COMMON, "nome": "TEXT", "lunghezza": "REAL", "disegno": "TEXT"},
                      scope="project")
        sections = Path(doc.path).parent / "sezioni" / "sezioni.json" if getattr(doc, "path", None) else None
        if sections is not None and sections.exists():
            try:
                for s in json.loads(sections.read_text(encoding="utf-8")):
                    layer.features.append((LineString([s["p1"][:2], s["p2"][:2]]), {
                        **base, "chunk": None, "chunk_id": None, "nome": s.get("label"),
                        "lunghezza": _number(s.get("length")), "disegno": s.get("svg")}))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        layers.append(layer)
    if "marker" in what:
        layer = Layer("marker", "POINT", {**COMMON, "nome": "TEXT", "z": "REAL", "ref_x": "REAL", "ref_y": "REAL",
                                          "ref_z": "REAL", "errore_m": "REAL", "controllo": "INTEGER"})
        for m in chunk.markers:
            ref = m.reference.location if m.reference is not None else None
            pos = chunk.to_world(np.asarray(m.position, float)[None])[0] if m.position is not None else None
            xyz = pos if pos is not None else (np.asarray(ref, float) if ref is not None else None)
            if xyz is None:
                continue
            error = float(np.linalg.norm(pos - np.asarray(ref, float))) if pos is not None and ref is not None else None
            layer.features.append((Point(float(xyz[0]), float(xyz[1])), {
                **base, "nome": m.label, "z": float(xyz[2]),
                "ref_x": None if ref is None else float(ref[0]), "ref_y": None if ref is None else float(ref[1]),
                "ref_z": None if ref is None else float(ref[2]), "errore_m": error,
                "controllo": int(bool(ref is not None and m.reference.enabled and not m.checkpoint))}))
        layers.append(layer)
    dem = chunk.dem_path
    if "curve" in what:
        layer = Layer("curve", "LINESTRING", {**COMMON, "quota": "REAL", "principale": "INTEGER",
                                              "equidistanza": "REAL"})
        if dem is not None and Path(dem).exists():
            from pyarchinit_photogrammetry.archaeo import terrain

            progress(0.1, t("curve di livello", "contour lines"))
            step = float(interval) if interval else terrain.suggest_interval(dem)
            for c in terrain.contours(dem, step):
                coords = np.asarray(c["coords"], float)
                if len(coords) >= 2:
                    layer.features.append((LineString(coords[:, :2]), {
                        **base, "quota": float(c["elevation"]), "principale": int(bool(c.get("index"))),
                        "equidistanza": step}))
        layers.append(layer)
    rasters = []
    if "dem" in what and dem is not None and Path(dem).exists():
        rasters.append({"kind": "dem", "path": Path(dem), **base})
    ortho = chunk.orthomosaic_path
    if "ortofoto" in what and ortho is not None and Path(ortho).exists():
        rasters.append({"kind": "ortofoto", "path": Path(ortho), **base})
    for layer in layers:
        layer.key = key
    return layers, rasters


def raster_layer(rasters: list, key: Key, kinds=None) -> Layer:
    """The rows of the ``raster`` table: footprint, file, resolution and size of what is archived
    (of the copy, when one was made: it may be reduced), table of the copy. Archiving again
    replaces only the rows of the ``kinds`` given (by default those of ``rasters``)."""
    import rasterio
    from shapely.geometry import box

    kinds = tuple(kinds) if kinds is not None else tuple(dict.fromkeys(r["kind"] for r in rasters))
    layer = Layer("raster", "POLYGON", {**COMMON, "tipo": "TEXT", "file": "TEXT", "risoluzione": "REAL",
                                        "larghezza": "INTEGER", "altezza": "INTEGER", "crs": "TEXT",
                                        "tabella": "TEXT"}, key=key, kinds=kinds)
    for r in rasters:
        copy = r.get("copy") or {}
        with rasterio.open(r["path"]) as src:
            b = src.bounds
            layer.features.append((box(b.left, b.bottom, b.right, b.top), {
                **{k: r.get(k) for k in COMMON}, "tipo": r["kind"], "file": str(r["path"]),
                "risoluzione": float(copy.get("resolution", src.res[0])),
                "larghezza": int(copy.get("width", src.width)), "altezza": int(copy.get("height", src.height)),
                "crs": src.crs.to_string() if src.crs else "", "tabella": r.get("table") or ""}))
    return layer


def _table_name(prefix: str, kind: str, name: str, chunk: str, suffix: str = "") -> str:
    """``<prefix><kind>_<project>_<chunk>[_<suffix>]``, cut to the length PostgreSQL keeps."""
    tail = f"_{suffix}" if suffix else ""
    return f"{prefix}{kind}_{slug(name)}_{slug(chunk)}"[: MAX_NAME - len(tail)].rstrip("_") + tail


def raster_table(db, table: str, key: Key, kind: str, prefix: str = "ppg_") -> str:
    """The name of the copy of the raster ``kind`` of ``key``: the one its rows in ``table`` (the
    ``raster`` table) already name, so that archiving again replaces the same copy; otherwise
    ``<prefix><kind>_<project>_<chunk>``, with a part of the project id after it when that name
    is the copy of another project (two projects with the same file name)."""
    readable = _table_name(prefix, kind, key.name, key.chunk)
    cols = db.columns(table)
    if not {"tipo", "tabella", "progetto", "chunk"} <= cols:
        return readable
    ids = "progetto_id" in cols
    cond, params = key.where("chunk", (kind,), ids)
    mine = [str(r[0]) for r in db.select(table, "tabella", f"({cond}) AND COALESCE(tabella, '') <> ''", params)]
    if mine:
        return mine[0]
    cond, params = key.where("chunk", None, ids)
    taken = db.select(table, "tabella", f"UPPER(COALESCE(tabella, '')) = UPPER(:table_name) "
                                        f"AND CASE WHEN ({cond}) THEN 1 ELSE 0 END = 0", {**params, "table_name": readable})
    return _table_name(prefix, kind, key.name, key.chunk, key.pid[:8]) if taken else readable


@contextmanager
def _reduced(src, max_side: int):
    """``src`` itself, or a copy in memory at most ``max_side`` pixels on its longer side
    (averaged; GDAL leaves the nodata cells out of the averages)."""
    if not max_side or max(src.width, src.height) <= max_side:
        yield src
        return
    from rasterio.enums import Resampling
    from rasterio.io import MemoryFile
    from rasterio.transform import Affine

    f = max(src.width, src.height) / max_side
    w, h = max(1, round(src.width / f)), max(1, round(src.height / f))
    data = src.read(out_shape=(src.count, h, w), resampling=Resampling.average)
    with MemoryFile() as mem:
        with mem.open(driver="GTiff", width=w, height=h, count=src.count, dtype=src.dtypes[0], crs=src.crs,
                      transform=src.transform @ Affine.scale(src.width / w, src.height / h),
                      nodata=src.nodata) as dst:
            dst.write(data)
            dst.colorinterp = src.colorinterp
        with mem.open() as view:
            yield view


def _copy_info(table: str, view) -> dict:
    """What a raster copy holds: its table, resolution and size."""
    return {"table": table, "resolution": float(view.res[0]), "width": int(view.width), "height": int(view.height)}


# ---------------------------------------------------------------------------- PostGIS raster tiles
PIXTYPES = {"uint8": (4, "B"), "int16": (5, "h"), "uint16": (6, "H"), "int32": (7, "i"), "uint32": (8, "I"),
            "float32": (10, "f"), "float64": (11, "d")}


def raster_wkb(data: np.ndarray, transform, srid: int, nodata=None) -> bytes:
    """A PostGIS raster in its well-known binary form (little endian, in-db bands): ``data``
    ``(bands, height, width)`` placed by the affine ``transform``. Nothing is asked of the GDAL
    of the server, which PostGIS keeps disabled by default."""
    bands, height, width = data.shape
    if height > 65535 or width > 65535:
        raise ArchiveError(t("un tassello raster non puo' superare 65535 pixel",
                             "a raster tile cannot be larger than 65535 pixels"))
    code, fmt = PIXTYPES[data.dtype.name]
    out = [struct.pack("<BHHddddddiHH", 1, 0, bands, transform.a, transform.e, transform.c, transform.f,
                       transform.b, transform.d, srid, width, height)]
    for b in range(bands):
        has_nodata = nodata is not None
        out.append(struct.pack("<B", code | (0x40 if has_nodata else 0)))
        out.append(struct.pack("<" + fmt, nodata if has_nodata else 0))
        out.append(np.ascontiguousarray(data[b]).astype(data.dtype.newbyteorder("<"), copy=False).tobytes())
    return b"".join(out)


# ---------------------------------------------------------------------------- the targets
class GeoPackageTarget:
    """A GeoPackage file (created when missing; a file that is not a GeoPackage is refused)."""

    kind = "GeoPackage"

    def __init__(self, path):
        from pyarchinit_photogrammetry.io import gpkg

        self.path = Path(path)
        if self.path.exists() and not gpkg.is_geopackage(self.path):
            raise ArchiveError(t(f"{self.path} esiste e non e' un GeoPackage",
                                 f"{self.path} exists and is not a GeoPackage"))

    def _reader(self) -> sqlite3.Connection | None:
        if not self.path.exists():
            return None  # nothing archived yet; sqlite3 would leave an empty file
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)

    def columns(self, table: str) -> set:
        """The columns of ``table`` (empty when it does not exist)."""
        con = self._reader()
        if con is None:
            return set()
        try:
            return {r[1] for r in con.execute(f"PRAGMA table_info({ident(table)})")}
        finally:
            con.close()

    def select(self, table: str, column: str, where: str, params: dict) -> list:
        """``SELECT column FROM table WHERE where`` (no rows when the table cannot be read)."""
        con = self._reader()
        if con is None:
            return []
        try:
            return con.execute(f"SELECT {ident(column)} FROM {ident(table)} WHERE {where}", params).fetchall()
        except sqlite3.Error:
            return []
        finally:
            con.close()

    def write(self, layer: Layer, crs: str | None, replace: bool) -> int:
        from pyarchinit_photogrammetry.io import gpkg

        srs = gpkg.srs_of(crs)
        con = gpkg.connect(self.path)
        try:
            con.execute("BEGIN IMMEDIATE")
            gpkg.create(con, srs)
            table = layer.name
            cols = gpkg.ensure_table(con, table, layer.geometry, layer.columns, srs[0])
            if replace:
                where, params = layer.where()
                con.execute(f"DELETE FROM {gpkg.quote(table)} WHERE {where}", params)
            rows, bounds = [], []
            for geom, values in layer.features:
                if geom is None or geom.is_empty:
                    continue
                rows.append([gpkg.blob(geom, srs[0])] + [values.get(c) for c in cols])
                bounds.append(geom.bounds)
            names = "".join(f", {gpkg.quote(c)}" for c in cols)
            con.executemany(f"INSERT INTO {gpkg.quote(table)} (geom{names}) VALUES ({', '.join('?' * (len(cols) + 1))})",
                            rows)
            old = con.execute("SELECT min_x, min_y, max_x, max_y FROM gpkg_contents WHERE table_name = ?",
                              (table,)).fetchone()
            ext = [b for b in bounds] + ([old] if old and old[0] is not None else [])
            if ext:
                con.execute("UPDATE gpkg_contents SET last_change = strftime('%Y-%m-%dT%H:%M:%fZ','now'), min_x = ?, "
                            "min_y = ?, max_x = ?, max_y = ? WHERE table_name = ?",
                            (min(e[0] for e in ext), min(e[1] for e in ext), max(e[2] for e in ext),
                             max(e[3] for e in ext), table))
            con.execute("COMMIT")
        except BaseException:
            if con.in_transaction:
                con.execute("ROLLBACK")
            raise
        finally:
            con.close()
        return len(rows)

    def _drop_raster(self, table: str) -> None:
        """A raster table written before (GDAL cannot replace one), with its metadata; a table of
        that name holding something else is left alone."""
        if not self.path.exists():
            return  # sqlite3 would leave an empty file, which GDAL cannot add a raster to
        con = sqlite3.connect(self.path)
        try:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
            if table not in tables:
                return
            row = con.execute("SELECT data_type FROM gpkg_contents WHERE table_name = ?", (table,)).fetchone() \
                if "gpkg_contents" in tables else None
            if row is None or row[0] not in ("tiles", "2d-gridded-coverage"):
                raise ArchiveError(t(f"la tabella {table} esiste gia' e non e' un raster: non viene sostituita",
                                     f"the table {table} exists and is not a raster: it is not replaced"))
            con.execute(f"DROP TABLE {ident(table)}")
            for meta, column in (("gpkg_contents", "table_name"), ("gpkg_tile_matrix", "table_name"),
                                 ("gpkg_tile_matrix_set", "table_name"), ("gpkg_extensions", "table_name"),
                                 ("gpkg_2d_gridded_coverage_ancillary", "tile_matrix_set_name"),
                                 ("gpkg_2d_gridded_tile_ancillary", "tpudt_name")):
                if meta in tables:
                    con.execute(f'DELETE FROM "{meta}" WHERE "{column}" = ?', (table,))
            con.commit()
        finally:
            con.close()

    def write_raster(self, raster: dict, table: str, srid, max_side: int, replace: bool) -> dict:
        """The raster as a raster table of the GeoPackage; returns its table, resolution and size."""
        import rasterio
        import rasterio.shutil

        table = table.upper()  # rasterio passes the creation options upper-cased: the name GDAL writes
        self._drop_raster(table)
        with rasterio.open(raster["path"]) as src, _reduced(src, max_side) as view:
            options = {"driver": "GPKG", "RASTER_TABLE": table, "APPEND_SUBDATASET": "YES"}
            if src.dtypes[0] not in ("uint8",):
                options["TILE_FORMAT"] = "TIFF"  # elevations: tiled gridded coverage of floats
            rasterio.shutil.copy(view, str(self.path), **options)
            return _copy_info(table, view)

    def close(self) -> None:
        pass


class SpatialiteTarget:
    """A SpatiaLite database (created when missing; mod_spatialite as the program finds it)."""

    kind = "SpatiaLite"
    TYPES = {"TEXT": "TEXT", "REAL": "DOUBLE", "INTEGER": "INTEGER"}

    def __init__(self, path):
        from pyarchinit_photogrammetry.pyarchinit import database

        database._warm_up_proj()  # PROJ and pycolmap before mod_spatialite (see the program's notes)
        lib = database.spatialite_path()
        if not lib:
            raise ArchiveError(t("SpatiaLite (mod_spatialite) non trovato: installa QGIS o libspatialite, "
                                 "oppure imposta SPATIALITE_LIBRARY_PATH",
                                 "SpatiaLite (mod_spatialite) not found: install QGIS or libspatialite, "
                                 "or set SPATIALITE_LIBRARY_PATH"))
        from pyarchinit_photogrammetry.io import gpkg

        self.path = Path(path)
        if self.path.exists() and gpkg.is_geopackage(self.path):
            raise ArchiveError(t(f"{self.path} e' un GeoPackage: scegli il tipo GeoPackage",
                                 f"{self.path} is a GeoPackage: choose the GeoPackage type"))
        self.con = sqlite3.connect(self.path)
        try:
            self.con.enable_load_extension(True)
            self.con.load_extension(lib)
            database.pin_spatialite(lib)
            if not self.con.execute("SELECT 1 FROM sqlite_master WHERE name = 'spatial_ref_sys'").fetchone():
                self.con.execute("SELECT InitSpatialMetaData(1)")
                self.con.commit()
        except sqlite3.DatabaseError as exc:  # not a database at all
            self.con.close()
            raise ArchiveError(t(f"{self.path} non e' un database SQLite: {exc}",
                                 f"{self.path} is not an SQLite database: {exc}")) from exc

    def columns(self, table: str) -> set:
        return {r[1] for r in self.con.execute(f"PRAGMA table_info({ident(table)})")}

    def select(self, table: str, column: str, where: str, params: dict) -> list:
        try:
            return self.con.execute(f"SELECT {ident(column)} FROM {ident(table)} WHERE {where}", params).fetchall()
        except sqlite3.Error:
            return []

    def write(self, layer: Layer, crs: str | None, replace: bool) -> int:
        from pyarchinit_photogrammetry.core.crs import epsg_code

        srid = epsg_code(crs) or -1
        con, table = self.con, layer.name
        have = [r[1] for r in con.execute(f"PRAGMA table_info({ident(table)})")]
        if not have:
            cols = "".join(f", {ident(c)} {self.TYPES[kind]}" for c, kind in layer.columns.items())
            con.execute(f"CREATE TABLE {ident(table)} (id INTEGER PRIMARY KEY AUTOINCREMENT{cols})")
            con.execute("SELECT AddGeometryColumn(?, 'geom', ?, ?, 'XY')", (table, srid, layer.geometry))
            con.execute("SELECT CreateSpatialIndex(?, 'geom')", (table,))
            have = ["id", *layer.columns, "geom"]
        for c, kind in layer.columns.items():
            if c not in have:
                con.execute(f"ALTER TABLE {ident(table)} ADD COLUMN {ident(c)} {self.TYPES[kind]}")
        if replace:
            where, params = layer.where()
            con.execute(f"DELETE FROM {ident(table)} WHERE {where}", params)
        cols = list(layer.columns)
        rows = [[geom.wkb] + [values.get(c) for c in cols] for geom, values in layer.features
                if geom is not None and not geom.is_empty]
        names = "".join(f", {ident(c)}" for c in cols)
        con.executemany(f"INSERT INTO {ident(table)} (geom{names}) VALUES (GeomFromWKB(?, {int(srid)})"
                        f'{", ?" * len(cols)})', rows)
        con.commit()
        return len(rows)

    def write_raster(self, raster, table, srid, max_side, replace):
        raise ArchiveError(t("SpatiaLite: i raster si registrano solo come riferimento (file e impronta)",
                             "SpatiaLite: the rasters are recorded as references only (file and footprint)"))

    def close(self) -> None:
        self.con.close()


def engine_url(url: str, password: str | None = None):
    """``(SQLAlchemy URL, schema)`` of a ``postgresql://`` address: the driver the program uses
    (``database.postgres_driver()``: psycopg2, or psycopg 3 where psycopg2 is missing, as in the
    Windows ARM64 application) unless the address names one; the password given, else the one of
    the address, else ``PGPASSWORD``; the schema from ``?schema=`` (``public`` by default)."""
    from sqlalchemy.engine import make_url

    from pyarchinit_photogrammetry.pyarchinit import database

    u = make_url(str(url).replace("postgres://", "postgresql://", 1))
    schema = u.query.get("schema") or "public"
    if isinstance(schema, (tuple, list)):
        schema = schema[0]
    u = u.difference_update_query(["schema"])
    if u.drivername in ("postgresql", "postgres"):
        u = u.set(drivername="postgresql+" + database.postgres_driver())
    if password:
        u = u.set(password=password)
    elif u.password is None and os.environ.get("PGPASSWORD"):
        u = u.set(password=os.environ["PGPASSWORD"])
    return u, str(schema)


class PostgisTarget:
    """A PostgreSQL database with PostGIS (``postgresql://user[:password]@host[:port]/database
    [?schema=name]``; the password can also come from ``password`` or ``PGPASSWORD``)."""

    kind = "PostGIS"
    TYPES = {"TEXT": "text", "REAL": "double precision", "INTEGER": "integer"}
    GEOMETRY = {"POINT": "Point", "LINESTRING": "LineString", "POLYGON": "Polygon", "MULTIPOLYGON": "MultiPolygon"}

    def __init__(self, url: str, password: str | None = None):
        from sqlalchemy import create_engine, text

        u, self.schema = engine_url(url, password)
        self.text = text
        try:
            self.engine = create_engine(u, pool_pre_ping=True)
            with self.engine.begin() as c:
                if not c.execute(text("SELECT 1 FROM pg_extension WHERE extname = 'postgis'")).first():
                    c.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
                if self.schema != "public":
                    c.execute(text(f"CREATE SCHEMA IF NOT EXISTS {ident(self.schema)}"))
        except Exception as exc:  # no driver, connection refused, authentication, no PostGIS
            raise ArchiveError(f"PostgreSQL: {str(exc).splitlines()[0]}") from exc

    def _raster_table_is_ours(self, c, table: str) -> bool | None:
        """None when ``table`` does not exist, else whether it is a raster table of this plugin."""
        cols = {r[0] for r in c.execute(self.text(
            "SELECT column_name FROM information_schema.columns WHERE table_schema = :s AND table_name = :t"),
            {"s": self.schema, "t": table})}
        return None if not cols else {"rid", "progetto", "chunk", "rast"} <= cols

    def _q(self, table: str) -> str:
        return f"{ident(self.schema)}.{ident(table)}"

    def columns(self, table: str) -> set:
        with self.engine.connect() as c:
            return {r[0] for r in c.execute(self.text(
                "SELECT column_name FROM information_schema.columns WHERE table_schema = :s AND table_name = :t"),
                {"s": self.schema, "t": table})}

    def select(self, table: str, column: str, where: str, params: dict) -> list:
        with self.engine.connect() as c:
            return [tuple(r) for r in c.execute(self.text(
                f"SELECT {ident(column)} FROM {self._q(table)} WHERE {where}"), params)]

    def write(self, layer: Layer, crs: str | None, replace: bool) -> int:
        from pyarchinit_photogrammetry.core.crs import epsg_code

        text = self.text
        srid = epsg_code(crs) or 0
        q = self._q(layer.name)
        cols = list(layer.columns)
        with self.engine.begin() as c:
            defs = "".join(f", {ident(k)} {self.TYPES[kind]}" for k, kind in layer.columns.items())
            c.execute(text(f"CREATE TABLE IF NOT EXISTS {q} (id serial PRIMARY KEY, "
                           f"geom geometry({self.GEOMETRY[layer.geometry]}, {int(srid)}){defs})"))
            for k, kind in layer.columns.items():
                c.execute(text(f"ALTER TABLE {q} ADD COLUMN IF NOT EXISTS {ident(k)} {self.TYPES[kind]}"))
            c.execute(text(f"CREATE INDEX IF NOT EXISTS {ident(layer.name + '_geom_idx')} ON {q} USING GIST (geom)"))
            if replace:
                where, params = layer.where()
                c.execute(text(f"DELETE FROM {q} WHERE {where}"), params)
            rows = [{"wkb": geom.wkb, **{f"v{i}": values.get(k) for i, k in enumerate(cols)}}
                    for geom, values in layer.features if geom is not None and not geom.is_empty]
            if rows:
                names = "".join(f", {ident(k)}" for k in cols)
                params = "".join(f", :v{i}" for i in range(len(cols)))
                c.execute(text(f"INSERT INTO {q} (geom{names}) VALUES (ST_GeomFromWKB(:wkb, {int(srid)}){params})"),
                          rows)
        return len(rows)

    def write_raster(self, raster: dict, table: str, srid, max_side: int, replace: bool) -> dict:
        """The raster in tiles of ``TILE`` pixels, in a table of its own made again each time (like
        the GeoPackage copy), with the spatial index and the constraints QGIS reads; returns its
        table, resolution and size."""
        import rasterio
        from rasterio.windows import Window

        text = self.text
        q = self._q(table)
        with self.engine.begin() as c:
            try:
                c.execute(text("SELECT postgis_raster_lib_version()"))
            except Exception as exc:
                raise ArchiveError(t("PostGIS raster non disponibile (CREATE EXTENSION postgis_raster): "
                                     "i raster restano come riferimento",
                                     "PostGIS raster not available (CREATE EXTENSION postgis_raster): "
                                     "the rasters stay as references")) from exc
        with rasterio.open(raster["path"]) as src, self.engine.begin() as c:
            ours = self._raster_table_is_ours(c, table)
            if ours is False:
                raise ArchiveError(t(f"la tabella {self.schema}.{table} esiste gia' e non e' un raster di questo "
                                     "plugin: non viene sostituita",
                                     f"the table {self.schema}.{table} exists and is not a raster of this plugin: "
                                     "it is not replaced"))
            if ours:
                c.execute(text(f"DROP TABLE {q}"))
            c.execute(text(f"CREATE TABLE {q} (rid serial PRIMARY KEY, progetto text, chunk text, rast raster)"))
            with _reduced(src, max_side) as view:
                info = _copy_info(table, view)
                nodata = view.nodata if view.dtypes[0] != "uint8" else None
                for row in range(0, view.height, TILE):
                    batch = []
                    for col in range(0, view.width, TILE):
                        win = Window(col, row, min(TILE, view.width - col), min(TILE, view.height - row))
                        data = view.read(window=win)
                        batch.append({"p": raster["progetto"], "c": raster["chunk"],
                                      "wkb": raster_wkb(data, view.window_transform(win), int(srid or 0), nodata)})
                    c.execute(text(f"INSERT INTO {q} (progetto, chunk, rast) VALUES (:p, :c, ST_RastFromWKB(:wkb))"),
                              batch)
            c.execute(text(f"CREATE INDEX {ident(table + '_rast_idx')} ON {q} USING GIST (ST_ConvexHull(rast))"))
            try:
                with c.begin_nested():  # the constraints fill raster_columns: QGIS reads the table faster
                    c.execute(text("SELECT AddRasterConstraints(CAST(:s AS name), CAST(:t AS name), CAST('rast' AS name), "
                                   "VARIADIC ARRAY['srid', 'scale_x', 'scale_y', 'num_bands', 'pixel_types', "
                                   "'nodata_values', 'alignment', 'extent'])"), {"s": self.schema, "t": table})
            except Exception:  # an older PostGIS: the table works without them
                pass
        return info

    def close(self) -> None:
        self.engine.dispose()


def open_target(target: str, password: str | None = None):
    """The database of ``target``: a ``.gpkg`` file, a SpatiaLite file (``.sqlite``, ``.db``,
    ``spatialite:PATH``) or ``postgresql://...`` (also ``postgresql+psycopg2://`` or
    ``postgresql+psycopg://`` to choose the driver)."""
    target = str(target)
    if target.startswith(("postgresql://", "postgres://", "postgresql+psycopg2://", "postgresql+psycopg://")):
        return PostgisTarget(target, password)
    if target.startswith("spatialite:"):
        return SpatialiteTarget(target[len("spatialite:"):])
    if target.startswith("gpkg:"):
        return GeoPackageTarget(target[len("gpkg:"):])
    suffix = Path(target).suffix.lower()
    if suffix == ".gpkg":
        return GeoPackageTarget(target)
    if suffix in (".sqlite", ".db", ".spatialite", ".sqlite3"):
        return SpatialiteTarget(target)
    raise ArchiveError(t(f"database non riconosciuto: {target} (un .gpkg, un .sqlite o postgresql://...)",
                         f"unknown database: {target} (a .gpkg, a .sqlite or postgresql://...)"))


# ---------------------------------------------------------------------------- the whole archive
def archive(doc, target: str, what=EVERYTHING, chunk=None, interval: float | None = None, prefix: str = "ppg_",
            replace: bool = True, rasters: str = "copy", max_side: int = 0, password: str | None = None,
            progress=None, remember: bool = True) -> dict:
    """Write the survey of ``chunk`` (the active one by default) into ``target``. ``rasters``:
    "copy" (into the database when it can hold them, else a reference) or "reference" (file and
    footprint only); ``max_side`` reduces the copies (0: full resolution). The archive id of the
    project goes into ``doc.meta["gis_archive"]`` unless ``remember`` is False (it is in the result
    anyway: ``identity``, and ``project_changed`` says whether the project has to keep a new one).
    Returns ``{"target", "kind", "tables": {table: rows}, "rasters": [...], "warnings": [...],
    "identity", "project_changed"}``."""
    from pyarchinit_photogrammetry.core.crs import epsg_code

    progress = progress or (lambda *a: None)
    chunk = chunk or doc.chunk
    if chunk is None:
        raise ArchiveError(t("il progetto non ha chunk", "the project has no chunk"))
    crs = chunk.crs
    what = list(what)
    warnings = []
    if epsg_code(crs) is None:
        warnings.append(t("il chunk e' in coordinate locali: le tabelle non hanno un sistema di riferimento (EPSG)",
                          "the chunk is in local coordinates: the tables have no reference system (EPSG)"))
    db = open_target(target, password)
    report: dict = {}
    try:
        progress(0.02, t("raccolta dei dati del progetto", "reading the project"))
        key, identity, notes = identify(db, doc, chunk, prefix)
        warnings += notes
        changed = (getattr(doc, "meta", None) or {}).get(ID_KEY) != identity
        if remember and changed:
            doc.meta[ID_KEY] = dict(identity)
        layers, raster_list = collect(doc, chunk, what, interval, progress, key=key, notes=warnings)
        for layer in layers:
            layer.name = f"{prefix}{layer.name}"
        n = max(1, len(layers) + len(raster_list))
        for i, layer in enumerate(layers):
            progress(0.1 + 0.6 * i / n, t(f"tabella {layer.name}", f"table {layer.name}"))
            report[layer.name] = db.write(layer, crs, replace)
        done = []
        rows_table = f"{prefix}raster"
        for j, r in enumerate(raster_list):
            progress(0.1 + 0.6 * (len(layers) + j) / n, f"raster {r['kind']}")
            if rasters == "copy":
                table = raster_table(db, rows_table, key, r["kind"], prefix)
                try:
                    r["copy"] = db.write_raster(r, table, epsg_code(crs), max_side, replace)
                    r["table"] = r["copy"]["table"]
                except ArchiveError as exc:
                    warnings.append(str(exc))
            done.append(r)
        kinds = tuple(k for k in RASTERS if k in what)
        if kinds:  # also when there are none: the rows of an earlier archive of these products go
            layer = raster_layer(done, key, kinds)
            layer.name = rows_table
            report[layer.name] = db.write(layer, crs, replace)
    finally:
        db.close()
    progress(1.0, t("archivio scritto", "archive written"))
    return {"target": str(target), "kind": db.kind, "tables": report,
            "rasters": [{"kind": r["kind"], "file": str(r["path"]), "table": r.get("table") or "",
                         "resolution": (r.get("copy") or {}).get("resolution")} for r in raster_list],
            "warnings": warnings, "identity": identity, "project_changed": changed}
