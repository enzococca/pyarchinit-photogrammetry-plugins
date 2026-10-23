"""The 3D model of the units: one closed solid per unit between its top and the top of the unit
below (the last unit down to a floor), built on the grid of the surfaces as the program builds the
proxies of the stratigraphic units (``archaeo/proxies.py``): the cells where the unit has a
thickness, their corners shared and set to the mean of the cells around them, a top, a bottom and
vertical walls along the edge - watertight by construction. The volume of a unit is the sum of
its thickness over its cells (the mesh encloses the same volume up to the corners of its edge).

A unit whose code matches stratigraphic units drawn in the chunk (``US 12``, ``12`` and ``US12``
match the unit 12 drawn as US; ``USM3`` only a USM) is clipped to their polygons.

Files in ``<chunk data>/geoarchaeology/model/``: a PLY per unit (double precision coordinates,
the colour of the unit), one OBJ with an object per unit and its material (``unita.obj``,
``unita.mtl``), the volumes in ``volumi.csv``; the registry in
``chunk.meta["geoarchaeology"]["model"]``.
"""
from __future__ import annotations

import csv
import re
import time
from pathlib import Path

import numpy as np

from . import boreholes as bh
from . import surfaces as sf
from .texts import t

MODEL_CELLS = 60_000  # cells of the model grid when no cell is given (a coarser copy of the surfaces' grid)
_US = re.compile(r"^\s*(US[A-Za-z]{0,2}|SU)[\s\-_.]*(?=\d)(.*)$", re.IGNORECASE)


def folder(chunk) -> Path:
    return bh.folder(chunk) / "model"


# ------------------------------------------------------------------ the mesh
def prism_mesh(top: np.ndarray, bottom: np.ndarray, mask: np.ndarray, transform) -> tuple[np.ndarray, np.ndarray]:
    """Closed mesh of the cells of ``mask`` between ``top`` and ``bottom`` (rasters of the same
    grid, world ``transform``): ``(vertices (n, 3) float64, faces (m, 3) int64)``, outward faces.
    The corners take the mean of the cells of the mask around them, as the proxies do."""
    mask = np.asarray(mask, bool) & np.isfinite(top) & np.isfinite(bottom)
    h, w = mask.shape
    if not mask.any():
        return np.zeros((0, 3)), np.zeros((0, 3), np.int64)
    weight = mask.astype(np.float64)

    def corners(vals):
        acc = np.zeros((h + 1, w + 1))
        cnt = np.zeros((h + 1, w + 1))
        v = np.where(mask, vals, 0.0)
        for dr in (0, 1):
            for dc in (0, 1):
                acc[dr:dr + h, dc:dc + w] += v
                cnt[dr:dr + h, dc:dc + w] += weight
        with np.errstate(invalid="ignore", divide="ignore"):
            return acc / np.where(cnt > 0, cnt, np.nan), cnt > 0

    hi = np.maximum(top, bottom)
    lo = np.minimum(top, bottom)
    ztop, used = corners(hi)
    zbot, _ = corners(lo)
    n = int(used.sum())
    idx_t = -np.ones((h + 1, w + 1), np.int64)
    idx_t[used] = np.arange(n)
    idx_b = np.where(used, idx_t + n, -1)
    rr, cc = np.nonzero(used)
    xs = transform.c + cc * transform.a + rr * transform.b
    ys = transform.f + cc * transform.d + rr * transform.e
    verts = np.vstack([np.column_stack([xs, ys, ztop[rr, cc]]), np.column_stack([xs, ys, zbot[rr, cc]])])
    r, c = np.nonzero(mask)
    ta, tb, tc, td = idx_t[r, c], idx_t[r, c + 1], idx_t[r + 1, c + 1], idx_t[r + 1, c]
    ba, bb, bc, bd = idx_b[r, c], idx_b[r, c + 1], idx_b[r + 1, c + 1], idx_b[r + 1, c]
    faces = [np.column_stack(f) for f in ((ta, td, tc), (ta, tc, tb), (ba, bb, bc), (ba, bc, bd))]
    pm = np.pad(mask, 1)
    # the walls where the neighbour is not of the unit: north (a, b), east (b, c), south (c, d), west (d, a)
    for open_, (tp, tq, bp, bq) in ((~pm[r, c + 1], (ta, tb, ba, bb)), (~pm[r + 1, c + 2], (tb, tc, bb, bc)),
                                    (~pm[r + 2, c + 1], (tc, td, bc, bd)), (~pm[r + 1, c], (td, ta, bd, ba))):
        if open_.any():
            k = open_
            faces.append(np.column_stack([tp[k], bq[k], bp[k]]))
            faces.append(np.column_stack([tp[k], tq[k], bq[k]]))
    faces = np.vstack(faces).astype(np.int64)
    if signed_volume(verts, faces) < 0:  # a south-up grid turns everything inside out
        faces = faces[:, ::-1]
    return verts, faces


def signed_volume(verts: np.ndarray, faces: np.ndarray) -> float:
    """Volume enclosed by a closed mesh (positive when the faces look outwards)."""
    if len(faces) == 0:
        return 0.0
    v = np.asarray(verts, float) - np.asarray(verts, float).mean(0)
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


# ------------------------------------------------------------------ stratigraphic units drawn
def unit_key(code) -> tuple[str | None, str]:
    """``(kind, number)`` of a unit code: ``"US 12"`` -> ("US", "12"), ``"USM3"`` -> ("USM", "3"),
    ``"12"`` -> (None, "12"), ``"Sabbia"`` -> (None, "sabbia")."""
    s = str(code).strip()
    m = _US.match(s)
    if m:
        kind = m.group(1).upper()
        return ("US" if kind == "SU" else kind), m.group(2).strip().lower()
    return None, s.lower()


def drawn_units(chunk) -> list:
    """The stratigraphic units drawn in the chunk (``UnitPolygon``), as the program keeps them."""
    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    out = []
    for d in (chunk.meta.get("drawings") or {}).get("units", []):
        try:
            out.append(UnitPolygon.from_dict(d))
        except Exception:  # noqa: BLE001 - a broken record of the project is left out
            continue
    return out


def polygons_for(code, units) -> list:
    """The drawn units whose number (and kind, when the code gives one) is that of ``code``."""
    kind, number = unit_key(code)
    out = []
    for u in units:
        if not u.us or u.geometry is None or u.geometry.is_empty:
            continue
        if str(u.us).strip().lower() == number and (kind is None or (u.tipo or "US").upper() == kind):
            out.append(u)
    return out


def _clip_mask(grid: sf.Grid, polygons) -> np.ndarray:
    from rasterio.features import geometry_mask
    from shapely.ops import unary_union

    shape = unary_union([u.geometry for u in polygons])
    return geometry_mask([shape], out_shape=(grid.height, grid.width), transform=grid.transform, invert=True)


# ------------------------------------------------------------------ the model
def coarsen(a: np.ndarray, f: int) -> np.ndarray:
    """Mean of the ``f`` x ``f`` blocks (NaN where fewer than half the cells have a value)."""
    if f <= 1:
        return a
    h, w = a.shape
    H, W = -(-h // f), -(-w // f)
    p = np.full((H * f, W * f), np.nan)
    p[:h, :w] = a
    b = p.reshape(H, f, W, f)
    ok = np.isfinite(b)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, b, 0.0).sum(axis=(1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n >= max(1, (f * f) // 2), s / n, np.nan)


def model_grid(grid: sf.Grid, cell: float | None) -> tuple[sf.Grid, int]:
    """The grid of the model: the surfaces' one, or coarser by an integer factor (``cell``, or the
    factor keeping it within :data:`MODEL_CELLS`)."""
    if cell:
        f = max(1, int(round(float(cell) / grid.cell)))
    else:
        f = max(1, int(np.ceil(np.sqrt(grid.width * grid.height / MODEL_CELLS))))
    return sf.Grid(grid.x0, grid.y0, grid.cell * f, -(-grid.width // f), -(-grid.height // f), grid.crs), f


def build(chunk, cell: float | None = None, clip: bool = True, floor: float | None = None, last: bool = True,
          min_thickness: float = 0.001, offset=None, units: list[str] | None = None, progress=None) -> dict:
    """A closed solid per unit (``units``: these only) from the surfaces of the chunk; ``clip``:
    the units drawn in the chunk cut the solids of their unit; ``last``: the last unit down to
    ``floor`` (the deepest bottom of the boreholes by default); ``offset``: subtracted from the
    coordinates of the OBJ (written in its header and in ``unita.obj.offset.txt``). Returns the
    registry of the model."""
    from pyarchinit_photogrammetry.core.progress import as_progress
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    progress = as_progress(progress)
    reg = sf.registry(chunk)
    grid0 = sf.grid_of(chunk)
    if grid0 is None or not reg.get("contacts"):
        raise RuntimeError(t("Costruisci prima le superfici di contatto", "Build the contact surfaces first"))
    grid, f = model_grid(grid0, cell)
    tops = [(u, coarsen(v, f)) for u, v in sf.tops(chunk)]
    holes = bh.load(chunk)
    if floor is None:
        bottoms = [b.z - b.bottom for b in holes if b.z is not None]
        floor = float(min(bottoms)) if bottoms else None
    drawn = drawn_units(chunk) if clip else []
    colours = bh.colours(chunk)
    descriptions = {u["code"]: u.get("description", "") for u in bh.units(chunk)}
    out_dir = folder(chunk)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.iterdir():
        if old.is_file():
            old.unlink()
    area = grid.cell * grid.cell
    rows, meshes = [], []
    for k, (unit, top) in enumerate(tops):
        if units and unit not in units:
            continue
        progress(0.05 + 0.8 * k / max(len(tops), 1), f"Solid of {unit}")
        if k + 1 < len(tops):
            base = tops[k + 1][1]
        elif last and floor is not None:
            base = np.where(np.isfinite(top), np.minimum(np.full_like(top, floor), top), np.nan)
        else:
            continue
        thick = top - base
        mask = np.isfinite(thick) & (thick > min_thickness)
        clipped = polygons_for(unit, drawn) if drawn else []
        if clipped:
            mask &= _clip_mask(grid, clipped)
        if not mask.any():
            rows.append({"unit": unit, "cells": 0, "volume_m3": 0.0, "area_m2": 0.0, "clipped": bool(clipped)})
            continue
        verts, faces = prism_mesh(top, base, mask, grid.transform)
        rgb = np.array([int(colours.get(unit, "#a0a0a0")[i:i + 2], 16) for i in (1, 3, 5)], np.uint8)
        name = sf.slug(unit)
        ply = write_mesh_ply(out_dir / f"{name}.ply", verts, faces, np.tile(rgb, (len(verts), 1)), double=True)
        th = thick[mask]
        rows.append({"unit": unit, "description": descriptions.get(unit, ""), "cells": int(mask.sum()),
                     "area_m2": float(mask.sum() * area), "volume_m3": float(th.sum() * area),
                     "mesh_volume_m3": signed_volume(verts, faces), "mean_thickness_m": float(th.mean()),
                     "max_thickness_m": float(th.max()), "top_min": float(np.nanmin(top[mask])),
                     "top_max": float(np.nanmax(top[mask])), "base_min": float(np.nanmin(base[mask])),
                     "base_max": float(np.nanmax(base[mask])), "clipped": bool(clipped),
                     "clipped_to": [f"{u.tipo or 'US'} {u.us}" for u in clipped], "faces": int(len(faces)),
                     "ply": str(ply.relative_to(chunk.data_dir)), "to_floor": k + 1 == len(tops)})
        meshes.append((unit, verts, faces, rgb))
    progress(0.88, "OBJ and volumes")
    files = {}
    if meshes:
        obj = write_obj(out_dir / "unita.obj", meshes, offset)
        files["obj"] = str(obj.relative_to(chunk.data_dir))
    csv_path = write_volumes(out_dir / "volumi.csv", rows)
    files["csv"] = str(csv_path.relative_to(chunk.data_dir))
    registry = {"made": time.strftime("%Y-%m-%d %H:%M"), "cell": grid.cell, "factor": f, "floor": floor,
                "clip": bool(clip), "units": rows, "files": files}
    bh.meta(chunk)["model"] = registry
    progress(1.0, f"{len(meshes)} solids, {sum(r['volume_m3'] for r in rows):.3f} m3")
    return registry


def write_obj(path: Path, meshes, offset=None) -> Path:
    """One OBJ with an object per unit (``o <unit>``) and a material of its colour."""
    off = np.zeros(3) if offset is None else np.asarray(offset, float).reshape(3)
    mtl = path.with_suffix(".mtl")
    with open(mtl, "w", encoding="utf-8") as f:
        for unit, _v, _f, rgb in meshes:
            r, g, b = (float(c) / 255.0 for c in rgb)
            f.write(f"newmtl {sf.slug(unit)}\nKd {r:.4f} {g:.4f} {b:.4f}\nKa 0 0 0\nd 1\nillum 1\n\n")
    base = 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("# pyarchinit-photogrammetry geoarchaeology: units between their contact surfaces\n")
        f.write(f"# offset {off[0]:.3f} {off[1]:.3f} {off[2]:.3f}\nmtllib {mtl.name}\n")
        for unit, verts, faces, _rgb in meshes:
            v = np.asarray(verts) - off
            f.write(f"o {sf.slug(unit)}\nusemtl {sf.slug(unit)}\n")
            np.savetxt(f, v, fmt="v %.3f %.3f %.3f")
            np.savetxt(f, np.asarray(faces) + base, fmt="f %d %d %d")
            base += len(v)
    if offset is not None:
        path.with_suffix(path.suffix + ".offset.txt").write_text(f"{off[0]:.3f} {off[1]:.3f} {off[2]:.3f}\n")
    return path


VOLUME_COLUMNS = ("unit", "description", "volume_m3", "area_m2", "mean_thickness_m", "max_thickness_m", "top_min",
                  "top_max", "base_min", "base_max", "cells", "clipped", "to_floor")


def write_volumes(path: Path, rows: list[dict]) -> Path:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(VOLUME_COLUMNS)
        for r in rows:
            w.writerow([round(r[c], 4) if isinstance(r.get(c), float) else r.get(c, "") for c in VOLUME_COLUMNS])
    return path


def registry(chunk) -> dict:
    return (chunk.meta.get("geoarchaeology") or {}).get("model") or {}


def remove(chunk) -> None:
    out_dir = folder(chunk)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
    bh.meta(chunk).pop("model", None)
