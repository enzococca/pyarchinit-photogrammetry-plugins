"""The drawing of PhotoCAD brought into the survey.

PhotoCAD's DXF is in millimetres from the lower-left corner of its page (the subject's box); :mod:`.register` finds
the similarity (one scale, an offset) that lays it on the image, and these functions apply and write it.

* :func:`georeference_dxf` - the DXF scaled and moved (every entity, the PhotoCAD layers kept), units metres;
* :func:`paths` - every entity as vertices (curves flattened), closed or open, with its layer;
* :func:`write_dxf` - polylines on their layers (the tiles merged);
* :func:`write_gpkg` - closed outlines as polygons, the rest as lines, with the PhotoCAD layer;
* :func:`write_dxf_3d`, :func:`write_gpkg_3d` - the drawing of an elevation on its surface, in the world
  (``engine.elevation.world_of``, the depth read from its relief);
* :func:`svg_note` - where the SVG lies (it keeps PhotoCAD's own frame: Y down from the page's upper-left corner).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("pyarchinit_photogrammetry.plugins.photocad")

TABLE = "photocad"
TABLE_3D = "photocad_3d"
COLUMNS = {"layer": "TEXT", "entity": "TEXT", "closed": "INTEGER", "length_m": "REAL", "area_m2": "REAL"}


def georeference_dxf(src: Path, dst: Path, dx: float, dy: float, scale: float = 1.0) -> int:
    """``src`` scaled by ``scale`` (from its drawing units to metres) and moved by ``(dx, dy)`` into ``dst``; how many
    entities could not be moved (left out)."""
    import ezdxf
    from ezdxf.math import Matrix44

    doc = ezdxf.readfile(str(src))
    msp = doc.modelspace()
    m = Matrix44.scale(scale, scale, scale) @ Matrix44.translate(dx, dy, 0.0)
    left = 0
    for e in list(msp):
        try:
            e.transform(m)
        except Exception as exc:  # noqa: BLE001 - an entity ezdxf cannot transform: out of the drawing
            log.warning("PhotoCAD DXF: %s left out (%s)", e.dxftype(), type(exc).__name__)
            msp.delete_entity(e)
            left += 1
    doc.header["$INSUNITS"] = 6  # metres
    from ezdxf import bbox

    ext = bbox.extents(msp)
    if ext.has_data:  # the extents moved with the drawing: "zoom extents" lands on it
        doc.header["$EXTMIN"] = tuple(ext.extmin)
        doc.header["$EXTMAX"] = tuple(ext.extmax)
        msp.dxf.extmin, msp.dxf.extmax = tuple(ext.extmin), tuple(ext.extmax)  # ezdxf writes the header from these
    doc.saveas(str(dst))
    return left


def paths(dxf: Path, tolerance: float = 0.001) -> list[dict]:
    """``{"layer", "entity", "closed", "xy" (N, 2)}`` of every entity of the model space; curves flattened to
    ``tolerance`` (drawing units); texts, points and blocks left out (blocks exploded)."""
    import ezdxf
    from ezdxf import path as ezpath

    doc = ezdxf.readfile(str(dxf))
    out: list[dict] = []

    def take(e):
        kind = e.dxftype()
        if kind == "INSERT":
            try:
                for sub in e.virtual_entities():
                    take(sub)
            except Exception:  # noqa: BLE001
                pass
            return
        if kind in ("TEXT", "MTEXT", "POINT", "DIMENSION", "ATTRIB", "ATTDEF", "IMAGE", "VIEWPORT"):
            return
        try:
            if kind == "HATCH":
                ps = list(ezpath.from_hatch(e))
            else:
                ps = [ezpath.make_path(e)]
        except Exception:  # noqa: BLE001 - an entity without a path
            return
        for p in ps:
            for sub in (p.sub_paths() if p.has_sub_paths else [p]):
                xy = np.array([(v.x, v.y) for v in sub.flattening(tolerance)], float)
                if len(xy) < 2:
                    continue
                closed = bool(sub.is_closed) or (len(xy) > 3 and np.allclose(xy[0], xy[-1], atol=tolerance))
                if closed and np.allclose(xy[0], xy[-1], atol=tolerance):
                    xy = xy[:-1]
                out.append({"layer": e.dxf.layer, "entity": kind, "closed": closed and len(xy) >= 3, "xy": xy})

    for e in doc.modelspace():
        take(e)
    return out


def write_dxf(path: Path, items: list[dict]) -> Path:
    """Polylines (closed or open) on their PhotoCAD layers, metres: the drawing merged from several tiles."""
    import ezdxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for it in items:
        if it["layer"] not in doc.layers:
            doc.layers.add(it["layer"])
        msp.add_lwpolyline([tuple(p) for p in it["xy"]], close=bool(it["closed"]), dxfattribs={"layer": it["layer"]})
    from ezdxf import bbox

    ext = bbox.extents(msp)
    if ext.has_data:
        doc.header["$EXTMIN"], doc.header["$EXTMAX"] = tuple(ext.extmin), tuple(ext.extmax)
        msp.dxf.extmin, msp.dxf.extmax = tuple(ext.extmin), tuple(ext.extmax)
    doc.saveas(str(path))
    return path


def geometries(items: list[dict]):
    """``(shapely geometry, attributes)`` of :func:`paths`: closed -> Polygon (made valid), else LineString."""
    from shapely.geometry import LineString, Polygon
    from shapely.validation import make_valid

    out = []
    for it in items:
        xy = it["xy"]
        attrs = {"layer": it["layer"], "entity": it["entity"], "closed": int(bool(it["closed"]))}
        if it["closed"]:
            g = Polygon(xy)
            if not g.is_valid:
                g = make_valid(g)
                g = max((p for p in getattr(g, "geoms", [g]) if p.geom_type == "Polygon"), key=lambda p: p.area,
                        default=None)
            if g is not None and g.area > 0:
                out.append((g, {**attrs, "length_m": g.exterior.length, "area_m2": g.area}))
                continue
        g = LineString(xy)
        if g.length > 0:
            out.append((g, {**attrs, "closed": 0, "length_m": g.length, "area_m2": None}))
    return out


def write_gpkg(path: Path, features, crs: str | None, table: str = TABLE) -> Path:
    """The polygons and lines in one table (geometry type GEOMETRY) of ``path``; ``crs`` None: undefined
    cartesian (the (u, v) of an elevation)."""
    from pyarchinit_photogrammetry.io import gpkg

    return gpkg.write_table(path, table, "GEOMETRY", COLUMNS, features, crs, identifier=table)


def world_items(chunk, entry: dict, items: list[dict]) -> list[dict]:
    """The vertices of the elevation's drawing (``(u, v)``) on its surface in the world: ``"xyz"`` (N, 3), the depth
    read from the relief of the elevation (``engine.elevation.world_of`` for every vertex); a vertex where the relief
    has no value takes the median depth of the drawing (0, the plane, when none has one)."""
    import rasterio

    from pyarchinit_photogrammetry.engine.elevation import elevation_paths, world_of

    _ortho, relief = elevation_paths(chunk, entry)
    depths = []
    if relief.exists():
        with rasterio.open(relief) as src:
            for it in items:
                d = np.array([v[0] for v in src.sample([(float(u), float(v)) for u, v in it["xy"]])], float)
                if src.nodata is not None:
                    d[d == src.nodata] = np.nan
                depths.append(d)
    else:
        depths = [np.full(len(it["xy"]), np.nan) for it in items]
    known = np.concatenate([d[np.isfinite(d)] for d in depths]) if depths else np.empty(0)
    fill = float(np.median(known)) if known.size else 0.0
    out = []
    for it, d in zip(items, depths):
        d = np.where(np.isfinite(d), d, fill)
        xyz = np.array([world_of(entry, u, v, float(dd)) for (u, v), dd in zip(it["xy"], d)], float)
        out.append({**it, "xyz": xyz})
    return out


def write_dxf_3d(path: Path, items3d: list[dict]) -> Path:
    """3D polylines on the PhotoCAD layers, world metres."""
    import ezdxf

    doc = ezdxf.new("R2000")
    doc.header["$INSUNITS"] = 6
    msp = doc.modelspace()
    for it in items3d:
        if it["layer"] not in doc.layers:
            doc.layers.add(it["layer"])
        msp.add_polyline3d([tuple(p) for p in it["xyz"]], close=bool(it["closed"]),
                           dxfattribs={"layer": it["layer"]})
    doc.saveas(str(path))
    return path


def write_gpkg_3d(path: Path, items3d: list[dict], crs: str | None, table: str = TABLE_3D) -> Path:
    """The drawing of an elevation in the world, with Z (lines; closed outlines closed), in ``path``."""
    from shapely.geometry import LineString

    from pyarchinit_photogrammetry.io import gpkg

    srs = gpkg.srs_of(crs)
    features = []
    for it in items3d:
        xyz = it["xyz"]
        if it["closed"]:
            xyz = np.vstack([xyz, xyz[:1]])
        g = LineString(xyz)
        features.append((g, {"layer": it["layer"], "entity": it["entity"], "closed": int(bool(it["closed"])),
                             "length_m": g.length, "area_m2": None}))
    con = gpkg.connect(path)
    try:
        con.execute("BEGIN IMMEDIATE")
        gpkg.create(con, srs)
        if gpkg.table_srs(con, table) is not None:
            gpkg.drop_table(con, table)
        cols = gpkg.ensure_table(con, table, "LINESTRINGZ", COLUMNS, srs[0], table, z=True)
        gpkg.replace_rows(con, table, cols, features, srs[0], z=True)
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()
    return path


def svg_note(path: Path, page_mm, similarity, crs: str | None, frame: str | None) -> Path:
    """Beside the SVG: where it lies. The SVG is PhotoCAD's, in mm at its natural size, with its origin at the
    upper-left corner of the page (the subject's box) and Y down: a point (sx, sy) of the SVG is at
    ``x = tx + m sx``, ``y = ty + m (page_height - sy)``."""
    m, tx, ty = similarity
    note = {"metres_per_mm": m, "page_mm": list(page_mm), "page_lower_left": [tx, ty],
            "page_upper_left": [tx, ty + m * float(page_mm[1])], "y_axis": "down (SVG)", "crs": crs,
            "frame": ("plan" if frame is None else f"elevation {frame} (u, v)"),
            "formula": "x = page_upper_left_x + metres_per_mm * svg_x; y = page_upper_left_y - metres_per_mm * svg_y"}
    path.write_text(json.dumps(note, indent=1), encoding="utf-8")
    return path
