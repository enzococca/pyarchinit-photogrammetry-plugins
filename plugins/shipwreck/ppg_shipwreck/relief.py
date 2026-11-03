"""The relief of the hull seen from above, in the ship frame.

An elevation of the program (``engine.elevation.build_elevation``) on a horizontal plane: ``u`` along the keel line
(u = ship x), ``v`` to the left of who walks along x (v = ``sign`` * ship y: the same as y when port is on the left, its
opposite on a mirror frame), the depth ``d`` up (d = ship z, the plane at the datum ``z0``). The raster view of the
program shows it in ship plan metres, its drawing tools and snap work on it and the drawings are kept with it. Its
``relief.tif`` is the height raster (the model's upper surface per cell: the inside of a hull lying on its bottom, NaN off
the model); the plugin adds a hillshade from four azimuths and a simple local relief model on the same grid
(``<chunk data>/shipwreck/relief/``), as ``archaeo.relief`` does for DEMs. For GIS a copy north-up in the chunk's
system is made on request (:func:`export_world`); the elevation itself is in ship metres, the frame in the tags.

The plugin writes ``chunk.meta["shipwreck"]["relief"]``, its folder and the elevation entry the program makes; the
chunk's transform, model and products are never changed.
"""
from __future__ import annotations

import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.core import crs as crs_mod
from pyarchinit_photogrammetry.core import geometry
from pyarchinit_photogrammetry.core.progress import as_progress

from . import lines, ship
from .ship import NotReady
from .texts import num, t

NAME = ("Scafo dall'alto (sistema della nave)", "Hull from above (ship frame)")
DEFAULT_CELL = 0.005
MAX_CELLS = 200_000_000
KINDS = ("height", "hillshade", "slrm")
DEPTH_MARGIN = 0.05  # the depth band: the model's z range and this much on either side
TILE = 1024
_DERIVED = {"hillshade": ("multi_hillshade", {"directions": 4}), "slrm": ("slrm", {})}


# ----------------------------------------------------------------------------- the plane and its coordinates
def plane_of(frame: ship.Frame):
    """The horizontal plane of the relief: origin on the keel line at ship x = 0 and the height ``z0``, ``u`` the plan
    direction of x, ``v`` its left, the normal up: u = x, v = sign * y, d = ship z."""
    from pyarchinit_photogrammetry.engine.elevation import Plane

    dx, dy = frame.direction
    origin = frame.to_world(np.zeros((1, 3)))[0]
    return Plane(origin, [dx, dy, 0.0], [-dy, dx, 0.0])


def uv_of_ship(xy, sign: int) -> np.ndarray:
    """``(N, 2)`` ship plan (x, y) to the relief's (u, v)."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    return np.column_stack([xy[:, 0], sign * xy[:, 1]])


def ship_of_uv(uv, sign: int) -> np.ndarray:
    """``(N, 2)`` relief (u, v) to the ship plan (x, y): the exact inverse of :func:`uv_of_ship`."""
    uv = np.asarray(uv, float).reshape(-1, 2)
    return np.column_stack([uv[:, 0], sign * uv[:, 1]])


# ----------------------------------------------------------------------------- the grid
def default_cell(chunk) -> float:
    """The cell from the model's density: its median edge in world metres, never below :data:`DEFAULT_CELL`."""
    mesh = ship._mesh(chunk)
    faces = np.asarray(mesh.faces)
    faces = faces[:: max(1, len(faces) // 20000)]
    p = geometry.apply(chunk.transform, np.asarray(mesh.vertices, float)[faces.ravel()]).reshape(-1, 3, 3)
    edges = np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2).ravel()
    edge = float(np.median(edges)) if len(edges) else 0.0
    return max(DEFAULT_CELL, edge)


def _extent(chunk, frame: ship.Frame):
    """``(u0, v0, u1, v1)`` and ``(zmin, zmax)`` of the model in the relief's coordinates."""
    v = np.asarray(lines.ship_mesh(chunk, frame).vertices)
    uv = uv_of_ship(v[:, :2], frame.sign)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    return (float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])), (float(v[:, 2].min()), float(v[:, 2].max()))


def _shape(bounds, cell: float) -> tuple[int, int]:
    """Rows and columns of the grid the program makes on ``bounds`` widened by one cell (as ``FrameSurface``)."""
    u0, v0, u1, v1 = bounds[0] - cell, bounds[1] - cell, bounds[2] + cell, bounds[3] + cell
    ou, ov = math.floor(u0 / cell) * cell, math.ceil(v1 / cell) * cell
    return int(math.ceil((ov - v0) / cell)) + 1, int(math.ceil((u1 - ou) / cell)) + 1


def check_size(chunk, cell: float) -> tuple[int, int]:
    """Rows and columns of the relief at ``cell``; :class:`NotReady` over :data:`MAX_CELLS`, naming the smallest cell
    that fits (rounded up to the millimetre)."""
    frame = ship.need_frame(chunk)
    bounds, _z = _extent(chunk, frame)
    h, w = _shape(bounds, cell)
    if h * w <= MAX_CELLS:
        return h, w
    fit = math.ceil(math.sqrt((bounds[2] - bounds[0]) * (bounds[3] - bounds[1]) / MAX_CELLS) * 1000.0) / 1000.0
    while np.prod(_shape(bounds, fit)) > MAX_CELLS:
        fit += 0.001
    mm = num(fit * 1000.0, 0)
    most = num(MAX_CELLS, 0)
    raise NotReady(t(f"Una cella di {num(cell * 1000.0, 2)} mm fa {h} x {w} celle, troppe (al massimo {most}). "
                     f"La cella più piccola che ci sta è di {mm} mm.",
                     f"A cell of {num(cell * 1000.0, 2)} mm makes {h} x {w} cells, too many (at most {most}). "
                     f"The smallest cell that fits is {mm} mm."))


# ----------------------------------------------------------------------------- what is kept
def _relief_state(chunk) -> dict | None:
    return chunk.meta.get(ship.KEY, {}).get("relief")


def entry(chunk) -> dict | None:
    """The elevation entry of the relief, or None (not built, or removed in the program)."""
    st = _relief_state(chunk)
    if not st:
        return None
    from pyarchinit_photogrammetry.engine.elevation import elevations

    return next((e for e in elevations(chunk) if e.get("id") == st.get("elevation")), None)


def _frame_key(d: dict | None):
    if d is None:
        return None
    return [list(map(float, d["start"])), list(map(float, d["end"])), d.get("bow", "start"), float(d["z0"]),
            float(d.get("x0", 0.0))]


def _model_key(chunk) -> dict | None:
    path = chunk.product_path("model")
    if path is None or not Path(path).exists():
        return None
    st = Path(path).stat()
    return {"file": Path(path).name, "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def is_stale(chunk) -> bool:
    """Whether the relief was built on another ship frame or another model than the chunk's now (False when there is no
    relief)."""
    st = _relief_state(chunk)
    if not st:
        return False
    current = ship.frame_of(chunk)
    return _frame_key(st.get("frame")) != _frame_key(None if current is None else current.to_dict()) or \
        st.get("model") != _model_key(chunk)


def _path(chunk, kind: str) -> Path:
    if kind not in KINDS:
        raise ValueError(f"unknown relief raster {kind!r}: one of {', '.join(KINDS)}")
    e = entry(chunk)
    if e is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    if kind == "height":
        from pyarchinit_photogrammetry.engine.elevation import elevation_paths

        return elevation_paths(chunk, e)[1]
    return chunk.data_dir / _relief_state(chunk)["files"][kind]


def load(chunk, kind: str):
    """The raster ``kind`` (float, NaN off the model) and its affine in (u, v)."""
    import rasterio

    with rasterio.open(_path(chunk, kind)) as src:
        arr = src.read(1)
        if src.nodata is not None and np.isfinite(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr).astype(arr.dtype)
        return arr, src.transform


# ----------------------------------------------------------------------------- the derived rasters
def _tags(frame: ship.Frame, kind: str) -> dict:
    return {"ppg_ship_frame": json.dumps(frame.to_dict()), "ppg_v": "sign*y", "ppg_sign": str(frame.sign),
            "ppg_kind": kind, "ppg_units": "ship metres (u = x, v = sign*y)"}


def _derive(height: Path, out: Path, kind: str, cell: float, frame: ship.Frame, progress) -> None:
    """The hillshade or local relief model of the height raster, by tiles with the margin the method reads (the result
    is the one on the whole raster), float32 with NaN off the model."""
    import rasterio
    from rasterio.windows import Window

    from pyarchinit_photogrammetry.archaeo.relief import compute, default_radius, margin_cells

    method, extra = _DERIVED[kind]
    radius = default_radius(cell)
    margin = margin_cells(method, cell, radius)
    with rasterio.open(height) as src:
        profile = {"driver": "GTiff", "height": src.height, "width": src.width, "count": 1, "dtype": "float32",
                   "transform": src.transform, "nodata": float("nan"), "compress": "deflate", "BIGTIFF": "IF_SAFER"}
        if src.width >= 256 and src.height >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        tiles = [(c, r, min(TILE, src.width - c), min(TILE, src.height - r))
                 for r in range(0, src.height, TILE) for c in range(0, src.width, TILE)]
        with rasterio.open(out, "w", **profile) as dst:
            dst.update_tags(**_tags(frame, kind), ppg_method=method, ppg_radius_m=str(radius))
            for i, (c, r, w, h) in enumerate(tiles):
                c0, r0 = max(0, c - margin), max(0, r - margin)
                c1, r1 = min(src.width, c + w + margin), min(src.height, r + h + margin)
                z = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
                if np.isfinite(z).any():
                    block = compute(z, cell, method, radius, **extra)[r - r0:r - r0 + h, c - c0:c - c0 + w]
                else:
                    block = np.full((h, w), np.nan)
                dst.write(block.astype(np.float32), 1, window=Window(c, r, w, h))
                progress((i + 1) / len(tiles), t(f"Rilievo ({kind}): {i + 1}/{len(tiles)}",
                                                  f"Relief ({kind}): {i + 1}/{len(tiles)}"))


def remove(chunk) -> bool:
    """Take the relief away: its elevation entry, its rasters and its record (the drawings made on it go with the
    elevation, as the program does for any elevation removed)."""
    from pyarchinit_photogrammetry.engine.elevation import remove_elevation

    st = ship.state(chunk).pop("relief", None)
    shutil.rmtree(chunk.data_dir / ship.KEY / "relief", ignore_errors=True)
    return bool(st) and remove_elevation(chunk, st.get("elevation", ""))


# ----------------------------------------------------------------------------- the drawings made on the relief
DRAWING_KEYS = ("units", "quote", "points", "lines", "stones")


def _move_drawings(data: dict, old_id: str, new_id: str, mapping=None) -> tuple[dict, int]:
    """The drawings of an elevation (``chunk.meta["drawings"]["frames"][old_id]``, records in ``(u, v)``) for the
    elevation ``new_id``: the stones keyed by the new id, the coordinates through ``mapping`` (``(N, 2) -> (N, 2)``)
    when given. Returns them and how many records they hold. Keys the plugin does not know are kept as they are."""
    import shapely

    def geom(text: str) -> str:
        g = shapely.from_wkt(text)
        if mapping is not None:
            g = shapely.transform(g, mapping)
        return shapely.to_wkt(g, rounding_precision=-1)

    def pair(a, b):
        return mapping(np.array([[float(a), float(b)]]))[0] if mapping is not None else (a, b)

    out, count = dict(data), 0
    if data.get("units"):
        out["units"] = []
        for rec in data["units"]:
            rec = dict(rec)
            if rec.get("wkt"):
                try:
                    rec["wkt"] = geom(rec["wkt"])
                except Exception:  # noqa: BLE001 - a broken record goes on as it was
                    pass
            out["units"].append(rec)
        count += len(out["units"])
    for key in ("quote", "points"):
        if data.get(key):
            out[key] = []
            for rec in data[key]:
                rec = dict(rec)
                for a, b in (("u", "v"), ("x", "y")):
                    if a in rec and b in rec:
                        u, v = pair(rec[a], rec[b])
                        rec[a], rec[b] = float(u), float(v)
                        break
                out[key].append(rec)
            count += len(out[key])
    if data.get("lines"):
        out["lines"] = []
        for rec in data["lines"]:
            rec = dict(rec)
            if rec.get("coords") and mapping is not None:
                c = [list(q) for q in rec["coords"]]  # the window writes [u, v, None]: only u, v are moved
                uv = mapping(np.array([[float(q[0]), float(q[1])] for q in c]))
                rec["coords"] = [[float(a), float(b), *q[2:]] for (a, b), q in zip(uv, c)]
            out["lines"].append(rec)
        count += len(out["lines"])
    stones = dict(data.get("stones") or {})
    if old_id in stones:
        polys = []
        for text in stones.pop(old_id) or []:
            try:
                polys.append(geom(text))
            except Exception:  # noqa: BLE001
                polys.append(text)
        stones[new_id] = polys
        count += len(polys)
    if "stones" in data:
        out["stones"] = stones
    return out, count


def _carried_drawings(chunk, old_entry: dict | None, new_entry: dict):
    """The drawings made on the previous relief, ready for the new one (nothing in the chunk is changed): as they are
    when the plane is the same (u, v are ship metres, whatever the cell), else through the old plane to the world and
    into the new one. None when there are none; else ``(data, {"records", "moved"})`` for :func:`_apply_drawings`."""
    from pyarchinit_photogrammetry.engine.elevation import frame_from_dict

    frames = (chunk.meta.get("drawings") or {}).get("frames") or {}
    if old_entry is None or old_entry["id"] not in frames:
        return None
    old_plane, new_plane = frame_from_dict(old_entry["frame"]), frame_from_dict(new_entry["frame"])
    same = all(np.allclose(getattr(old_plane, k), getattr(new_plane, k), rtol=0, atol=1e-9)
               for k in ("origin", "u_axis", "v_axis"))

    def mapping(uv):
        uv = np.asarray(uv, float).reshape(-1, 2)
        world = old_plane.point(uv[:, 0], uv[:, 1], np.zeros(len(uv)))
        return new_plane.coords(world)[:, :2]

    data, count = _move_drawings(frames[old_entry["id"]], old_entry["id"], new_entry["id"], None if same else mapping)
    return data, {"records": count, "moved": not same}


def _apply_drawings(chunk, old_id: str, new_id: str, carried) -> dict:
    """The drawings of :func:`_carried_drawings` under the new id, the old id's taken away."""
    if carried is None:
        return {"records": 0, "moved": False}
    frames = chunk.meta["drawings"]["frames"]
    frames.pop(old_id, None)
    frames[new_id] = carried[0]
    return carried[1]


# ----------------------------------------------------------------------------- building
def build(chunk, cell: float | None = None, progress=None) -> dict:
    """Build (again) the relief of the hull: the elevation on :func:`plane_of`, then its hillshade and local relief
    model. The previous relief is replaced only once the new one is complete (a failure or a cancel keeps it), and the
    drawings made on it go over to the new one (``state["relief"]["drawings"]``: how many, and whether they were moved
    because the frame changed). Returns ``state["relief"]``."""
    from pyarchinit_photogrammetry.engine.elevation import build_elevation, elevation_paths, remove_elevation

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    ship.check_metric(chunk)
    cell = float(cell or default_cell(chunk))
    if cell <= 0:
        raise ValueError("the cell must be positive")
    check_size(chunk, cell)
    bounds, (zmin, zmax) = _extent(chunk, frame)
    old_entry = entry(chunk)
    progress(0.0, t("Rilievo dello scafo dall'alto", "Relief of the hull from above"))
    e = build_elevation(chunk, plane_of(frame), cell, t(*NAME), depth=(zmin - DEPTH_MARGIN, zmax + DEPTH_MARGIN),
                        bounds=(bounds[0] - cell, bounds[1] - cell, bounds[2] + cell, bounds[3] + cell),
                        progress=progress.sub(0.0, 0.8))
    height = elevation_paths(chunk, e)[1]
    final = ship.folder(chunk) / "relief"
    work = ship.folder(chunk) / "relief.new"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    backup = ship.folder(chunk) / "relief.old"
    try:  # everything that can fail before the old relief is touched; then nothing can
        for k, kind in enumerate(("hillshade", "slrm")):
            _derive(height, work / f"{kind}.tif", kind, cell, frame, progress.sub(0.8 + 0.1 * k, 0.1))
        carried = _carried_drawings(chunk, old_entry, e)
        shutil.rmtree(backup, ignore_errors=True)
        if final.exists():
            final.rename(backup)
        try:
            work.rename(final)
        except BaseException:
            if backup.exists():
                backup.rename(final)
            raise
    except BaseException:
        remove_elevation(chunk, e["id"])
        shutil.rmtree(work, ignore_errors=True)
        raise
    old_id = (_relief_state(chunk) or {}).get("elevation")
    drawings = _apply_drawings(chunk, old_entry["id"] if old_entry else "", e["id"], carried)
    if old_id:
        remove_elevation(chunk, old_id)
    shutil.rmtree(backup, ignore_errors=True)
    files = {kind: str((final / f"{kind}.tif").relative_to(chunk.data_dir)) for kind in ("hillshade", "slrm")}
    st = {"elevation": e["id"], "cell": cell, "files": files,
          "built": datetime.now(timezone.utc).isoformat(timespec="seconds"), "frame": frame.to_dict(),
          "model": _model_key(chunk), "drawings": drawings}
    ship.state(chunk)["relief"] = st
    progress(1.0, t(f"Rilievo dello scafo: {e['size'][0]} x {e['size'][1]} celle di {num(cell * 1000.0, 1)} mm",
                    f"Relief of the hull: {e['size'][0]} x {e['size'][1]} cells of {num(cell * 1000.0, 1)} mm"))
    return st


# ----------------------------------------------------------------------------- exports
def _frame_of_relief(chunk) -> ship.Frame:
    return ship.Frame.from_dict(_relief_state(chunk)["frame"])


def export_ship(chunk, kind: str, path) -> Path:
    """The raster ``kind`` as a GeoTIFF in ship metres (u = x, v = sign*y; no system), the frame in the tags."""
    import rasterio

    src_path, path = _path(chunk, kind), Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src_path, path)
    with rasterio.open(path, "r+") as dst:
        dst.update_tags(**_tags(_frame_of_relief(chunk), kind))
    return path


def _rasterio_crs(chunk):
    from rasterio.crs import CRS

    if not chunk.crs or crs_mod.is_local(chunk.crs):
        return None
    try:
        return CRS.from_user_input(chunk.crs)
    except Exception:  # a system rasterio does not know: the copy without one
        return None


def export_world(chunk, kind: str, path) -> Path:
    """The raster ``kind`` north-up in the chunk's system (same cell, NaN off the model), warped from the rotated
    (u, v) -> world affine. Heights are the chunk's (ship z + ``z0``, tag ``ppg_heights = "world"``); the hillshade and
    the local relief model are relative and stay as they are."""
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.transform import Affine, from_origin
    from rasterio.warp import reproject
    from rasterio.windows import Window

    frame = _frame_of_relief(chunk)
    plane = plane_of(frame)
    (ux, uy), (vx, vy) = plane.u_axis[:2], plane.v_axis[:2]
    ox, oy = plane.origin[:2]
    crs = _rasterio_crs(chunk)
    work_crs = crs or CRS.from_epsg(3857)  # the same on both sides: only the affine changes
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(_path(chunk, kind)) as src:
        a = src.transform
        world = Affine(ux * a.a + vx * a.d, ux * a.b + vx * a.e, ox + ux * a.c + vx * a.f,
                       uy * a.a + vy * a.d, uy * a.b + vy * a.e, oy + uy * a.c + vy * a.f)
        corners = np.array([world @ (c, r) for c, r in ((0, 0), (src.width, 0), (0, src.height),
                                                        (src.width, src.height))])
        cell = abs(a.a)
        x0, y1 = math.floor(corners[:, 0].min() / cell) * cell, math.ceil(corners[:, 1].max() / cell) * cell
        w = int(math.ceil((corners[:, 0].max() - x0) / cell))
        h = int(math.ceil((y1 - corners[:, 1].min()) / cell))
        dst_t = from_origin(x0, y1, cell, cell)
        profile = {"driver": "GTiff", "height": h, "width": w, "count": 1, "dtype": "float32", "transform": dst_t,
                   "nodata": float("nan"), "compress": "deflate", "BIGTIFF": "IF_SAFER"}
        if crs is not None:
            profile["crs"] = crs
        if w >= 256 and h >= 256:
            profile.update(tiled=True, blockxsize=256, blockysize=256)
        nodata = src.nodata if src.nodata is not None else float("nan")
        back = ~world
        with rasterio.open(path, "w", **profile) as dst:
            dst.update_tags(**_tags(frame, kind), ppg_z0=str(frame.z0), ppg_chunk_crs=str(chunk.crs),
                            ppg_heights="world" if kind == "height" else "relative")
            for r in range(0, h, TILE):
                for c in range(0, w, TILE):
                    th, tw = min(TILE, h - r), min(TILE, w - c)
                    tile_t = dst_t @ Affine.translation(c, r)
                    cols, rows = zip(*(back @ (tile_t @ (x, y)) for x, y in ((0, 0), (tw, 0), (0, th), (tw, th))))
                    c0, r0 = max(0, int(math.floor(min(cols))) - 2), max(0, int(math.floor(min(rows))) - 2)
                    c1 = min(src.width, int(math.ceil(max(cols))) + 2)
                    r1 = min(src.height, int(math.ceil(max(rows))) + 2)
                    out = np.full((th, tw), np.nan, np.float32)
                    if c1 > c0 and r1 > r0:
                        block = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float32)
                        reproject(block, out, src_transform=world @ Affine.translation(c0, r0), src_crs=work_crs,
                                  src_nodata=nodata, dst_transform=tile_t, dst_crs=work_crs,
                                  dst_nodata=float("nan"), resampling=Resampling.bilinear)
                    if kind == "height":  # the chunk's heights: ship z + the datum
                        out = (out.astype(np.float64) + frame.z0).astype(np.float32)
                    dst.write(out, 1, window=Window(c, r, tw, th))
    return path
