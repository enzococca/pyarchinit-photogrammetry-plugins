"""The drawing of PhotoCAD brought into the survey.

PhotoCAD's DXF, asked with ``units=m``, ``margin=0`` and ``longest`` = the long side of the image in metres, is
in metres from the lower-left corner of the image, Y up. Moving it by the corner of the image (:class:`Rendered`
``x0, y0``) puts it in the coordinates of the chunk (plan) or in the ``(u, v)`` of the elevation; for an
elevation each vertex can also go to the world on the elevation's surface (``engine.elevation.world_of``, the
depth read from its relief).

* :func:`georeference_dxf` - the DXF moved (every entity, the PhotoCAD layers kept), units metres;
* :func:`paths` - every entity as vertices (curves flattened), closed or open, with its layer;
* :func:`write_gpkg` - closed outlines as polygons, the rest as lines, with the PhotoCAD layer;
* :func:`write_dxf_3d`, :func:`write_gpkg_3d` - the drawing of an elevation on its surface, in the world;
* :func:`svg_note` - where the SVG lies (it keeps PhotoCAD's own frame: Y down from the upper-left corner).
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


def georeference_dxf(src: Path, dst: Path, dx: float, dy: float) -> int:
    """``src`` moved by ``(dx, dy)`` into ``dst``; how many entities could not be moved (left out)."""
    import ezdxf
    from ezdxf.math import Matrix44

    doc = ezdxf.readfile(str(src))
    msp = doc.modelspace()
    m = Matrix44.translate(dx, dy, 0.0)
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


def check_frame(raw_m: Path, raw_mm: Path | None, image: dict, page_mm=None, longest_m: float | None = None) -> dict:
    """Is PhotoCAD's drawing in the image's frame - origin at its lower-left corner, the long side of the *page*
    (the whole image) = ``longest_m``? Three checks, an image pixel of tolerance:

    * the extents of the drawing (asked in metres) inside the image, ``[0, width_m] x [0, height_m]``;
    * the page (``width_mm`` / ``height_mm`` of the conversion) with the image's proportions (1 %);
    * the drawing asked in metres against the same drawing asked at its natural size in mm (``raw_mm``, no
      ``longest``): the corners of every entity fitted as ``m = a mm + b`` per axis must give ``b = 0``, the same
      ``a`` in x and y, and ``a`` = ``longest_m`` / the long side of the page. A server that scales the drawing's
      content instead of the page shows an offset and another scale here, even when the content stays inside.

    ``{"ok", "problems" [text], "extents_m", "page_aspect_ratio", "scale_ratio", "offset_m", "compared"}``."""
    import numpy as np

    from .texts import t

    w_m, h_m = image["width"] * image["pixel"], image["height"] * image["pixel"]
    tol = float(image["pixel"])
    longest_m = float(longest_m or max(w_m, h_m))
    problems, out = [], {"compared": False}
    items = paths(raw_m, tolerance=max(tol / 4, 1e-4))
    if items:
        allp = np.vstack([it["xy"] for it in items])
        lo, hi = allp.min(axis=0), allp.max(axis=0)
        out["extents_m"] = [float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])]
        if lo[0] < -tol or lo[1] < -tol or hi[0] > w_m + tol or hi[1] > h_m + tol:
            problems.append(t("il disegno esce dall'immagine ({0:.3f}, {1:.3f} - {2:.3f}, {3:.3f} m contro 0, 0 - "
                              "{4:.3f}, {5:.3f} m)", "the drawing goes beyond the image ({0:.3f}, {1:.3f} - {2:.3f}, "
                              "{3:.3f} m against 0, 0 - {4:.3f}, {5:.3f} m)").format(*lo, *hi, w_m, h_m))
    pw, ph = (list(page_mm or []) + [None, None])[:2]
    if pw and ph and float(pw) > 0 and float(ph) > 0:
        ratio = (float(pw) / float(ph)) / (image["width"] / image["height"])
        out["page_aspect_ratio"] = ratio
        if abs(ratio - 1) > 0.01:
            problems.append(t("la pagina di PhotoCAD ({0:g} x {1:g} mm) non ha le proporzioni dell'immagine "
                              "({2} x {3} px)", "PhotoCAD's page ({0:g} x {1:g} mm) does not have the proportions of "
                              "the image ({2} x {3} px)").format(float(pw), float(ph), image["width"], image["height"]))
    if raw_mm is not None and items:
        try:
            other = paths(raw_mm, tolerance=1.0)
        except Exception:  # noqa: BLE001 - a file that cannot be read: the other checks stay
            other = []
        if len(other) == len(items):
            a = np.vstack([np.vstack([it["xy"].min(axis=0), it["xy"].max(axis=0)]) for it in items])
            b = np.vstack([np.vstack([it["xy"].min(axis=0), it["xy"].max(axis=0)]) for it in other])
            scale, offset = [], []
            for axis in (0, 1):
                if np.ptp(b[:, axis]) <= 0:
                    continue
                k, c = np.polyfit(b[:, axis], a[:, axis], 1)
                scale.append(float(k))
                offset.append(float(c))
            if scale:
                out["compared"] = True
                out["offset_m"] = offset
                if max(abs(c) for c in offset) > tol:
                    problems.append(t("il disegno in metri è spostato di {0:.3f} m rispetto all'angolo dell'immagine",
                                      "the drawing in metres is shifted by {0:.3f} m from the corner of the image")
                                    .format(max(abs(c) for c in offset)))
                if len(scale) == 2 and abs(scale[0] / scale[1] - 1) > 0.01:
                    problems.append(t("scale diverse in x e in y", "different scales in x and y"))
                if pw and ph:
                    expected = longest_m / max(float(pw), float(ph))
                    out["scale_ratio"] = float(np.mean(scale)) / expected
                    if abs(out["scale_ratio"] - 1) > 0.01:
                        problems.append(t("scala {0:.4f} invece di {1:.4f} (longest applicato al disegno, non alla "
                                          "pagina?)", "scale {0:.4f} instead of {1:.4f} (longest applied to the "
                                          "drawing, not to the page?)").format(float(np.mean(scale)), expected))
    out["problems"] = problems
    out["ok"] = not problems
    return out


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


def svg_note(path: Path, rendered, crs: str | None, frame: str | None) -> Path:
    """Beside the SVG: where it lies. The SVG is PhotoCAD's, in metres (``longest`` asked), with its origin at the
    upper-left corner of the image and Y down: a point (sx, sy) of the SVG is x = x0 + sx, y = y0 + height - sy."""
    note = {"lower_left": [rendered.x0, rendered.y0], "upper_left": [rendered.x0, rendered.y0 + rendered.height_m],
            "width_m": rendered.width_m, "height_m": rendered.height_m, "units": "m", "y_axis": "down (SVG)",
            "crs": crs, "frame": ("plan" if frame is None else f"elevation {frame} (u, v)"),
            "formula": "x = upper_left_x + svg_x; y = upper_left_y - svg_y"}
    path.write_text(json.dumps(note, indent=1), encoding="utf-8")
    return path
