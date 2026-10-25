"""The runs of a chunk: what each one looked for and how, kept in the project
(``chunk.meta["samgeo"]["runs"]``), with its files in ``<chunk folder>/samgeo/``:

* ``<id>.gpkg``: the objects found (table ``samgeo``: ``n``, the number of the object in the mask;
  ``prompt``, the text; ``label``, the phrase that found it; ``score``; ``quality``; ``area_m2``),
  multipolygons in the coordinates of the chunk;
* ``<id>_mask.tif``: the objects numbered on the grid of the run (0 where there is none), named
  ``samgeo: <name>`` (its ``ppg_name`` tag): the raster calculator finds it by itself.

The objects become units (:func:`unit_rows` for ``context.add_units``), classes of the dense cloud
(:func:`classify`) or a GeoPackage of their own (:func:`export`).
"""
from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import numpy as np

from .texts import slug, t

FOLDER = "samgeo"
TABLE = "samgeo"
COLUMNS = {"n": "INTEGER", "prompt": "TEXT", "label": "TEXT", "score": "REAL", "quality": "REAL", "area_m2": "REAL"}
COLOURS = ("#ff8c00", "#00c8ff", "#ff3cdc", "#7cfc00", "#ffd700", "#ff4040", "#40e0d0", "#b070ff")
FIRST_CLASS = 150  # the classes of the runs, above those the program gives to the units (100 up)
BLOCK = 2048  # the mask is written in windows of this size: a whole site at 1 cm/px does not fit in memory


def records(chunk) -> list[dict]:
    """The runs of the chunk, the oldest first."""
    if chunk is None:
        return []
    return [r for r in ((chunk.meta.get("samgeo") or {}).get("runs") or []) if isinstance(r, dict)]


def record(chunk, run_id: str) -> dict | None:
    return next((r for r in records(chunk) if r.get("id") == run_id), None)


def find_record(chunk, name: str) -> dict | None:
    """A run by its id or its name (the command line)."""
    return next((r for r in records(chunk) if name in (r.get("id"), r.get("name"))), None)


def folder(chunk) -> Path:
    return Path(chunk.data_dir) / FOLDER


def layer(run: dict) -> str:
    """The layer of the run on the 2D view."""
    return f"samgeo: {run.get('name', '?')}"


def path_of(chunk, run: dict, kind: str) -> Path | None:
    """The file of the run, ``gpkg`` or ``mask`` (None when the run has none)."""
    rel = (run.get("files") or {}).get(kind)
    return Path(chunk.data_dir) / rel if rel else None


def identity(chunk, text: str) -> tuple[str, str]:
    """A name for a new run of ``text`` (``pietre``, ``pietre 2``...) and its id for the files
    (``pietre``, ``pietre_2``), neither used by the runs of the chunk nor by a file there."""
    base = " ".join(text.split())[:60] or "samgeo"
    names = {r.get("name") for r in records(chunk)}
    ids = {r.get("id") for r in records(chunk)}
    k = 1
    while True:
        name = base if k == 1 else f"{base} {k}"
        rid = slug(name)
        taken = rid in ids or (folder(chunk) / f"{rid}.gpkg").exists() or (folder(chunk) / f"{rid}_mask.tif").exists()
        if name not in names and not taken:
            return name, rid
        k += 1


def write(chunk, the_plan, objects: list[dict], info: dict) -> dict:
    """The files of a new run (its objects, the best score first) and its record, which the caller
    adds to the project with :func:`add` (in the interface thread, in the window). ``info``: what the
    run looked for and how (``text``, ``prompt``, ``engine``, ``raster``...)."""
    from pyarchinit_photogrammetry.io import gpkg

    name, rid = identity(chunk, info.get("text") or info.get("prompt") or "samgeo")
    out = folder(chunk)
    out.mkdir(parents=True, exist_ok=True)
    base = Path(chunk.data_dir)
    rows = [(_multi(o["geometry"]), {"n": k, "prompt": info.get("text", ""), "label": o.get("label", ""),
                                      "score": round(float(o["score"]), 4), "quality": round(float(o.get("quality", 0.0)), 4),
                                      "area_m2": round(float(o["geometry"].area), 4)})
            for k, o in enumerate(objects, 1)]
    g = gpkg.write_table(out / f"{rid}.gpkg", TABLE, "MULTIPOLYGON", COLUMNS, rows, getattr(chunk, "crs", None),
                         f"samgeo: {name}")
    m = write_mask(out / f"{rid}_mask.tif", [o["geometry"] for o in objects], the_plan, f"samgeo: {name}")
    used = {r.get("colour") for r in records(chunk)}
    colour = next((c for c in COLOURS if c not in used), COLOURS[len(records(chunk)) % len(COLOURS)])
    polygon = the_plan.polygon
    return {"id": rid, "name": name, **info, "bounds": [round(v, 4) for v in the_plan.bounds],
            "polygon": polygon.wkt if polygon is not None else None,
            "resolution": the_plan.resolution, "tiles": len(the_plan.tiles), "count": len(rows),
            "files": {"gpkg": g.relative_to(base).as_posix(), "mask": m.relative_to(base).as_posix()},
            "colour": colour, "visible": True, "time": dt.datetime.now().isoformat(timespec="seconds")}


def add(chunk, run: dict) -> None:
    """The run into the project (``chunk.meta``)."""
    meta = chunk.meta.setdefault("samgeo", {})
    meta["runs"] = [r for r in records(chunk) if r.get("id") != run["id"]] + [run]


def _multi(geom):
    from shapely.geometry import MultiPolygon

    return geom if geom.geom_type == "MultiPolygon" else MultiPolygon([geom])


def write_mask(path: Path, geoms: list, the_plan, name: str) -> Path:
    """The objects numbered (1 for the first of ``geoms``) on the grid of the run, the smaller ones
    over the larger; written in windows, tagged ``ppg_name``."""
    import rasterio
    from affine import Affine
    from rasterio.features import rasterize
    from rasterio.windows import Window
    from shapely.geometry import box
    from shapely.strtree import STRtree

    x0, y0, x1, y1 = the_plan.bounds
    res = the_plan.resolution
    w, h = the_plan.size
    tr = Affine(res, 0.0, x0, 0.0, -res, y1)
    dtype = "uint16" if len(geoms) < 65535 else "uint32"
    profile = {"driver": "GTiff", "width": w, "height": h, "count": 1, "dtype": dtype, "transform": tr, "nodata": 0,
               "compress": "deflate", "BIGTIFF": "IF_SAFER"}
    if w >= 512 and h >= 512:
        profile.update(tiled=True, blockxsize=512, blockysize=512)
    if the_plan.crs:
        profile["crs"] = the_plan.crs
    tree = STRtree(geoms) if geoms else None
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for r in range(0, h, BLOCK):
            for c in range(0, w, BLOCK):
                win = Window(c, r, min(BLOCK, w - c), min(BLOCK, h - r))
                wtr = dst.window_transform(win)
                wb = rasterio.windows.bounds(win, tr)
                found = [] if tree is None else [int(i) for i in tree.query(box(*wb))]
                shapes = [(geoms[i], i + 1) for i in sorted(found, key=lambda i: -geoms[i].area)]
                arr = rasterize(shapes, out_shape=(int(win.height), int(win.width)), transform=wtr, fill=0,
                                dtype=dtype) if shapes else np.zeros((int(win.height), int(win.width)), dtype)
                dst.write(arr, 1, window=win)
        dst.update_tags(ppg_name=name)
        dst.set_band_description(1, name)
    return path


def features(chunk, run: dict) -> list[tuple]:
    """The objects of the run: ``(shapely geometry, {n, prompt, label, score, quality, area_m2})``."""
    from pyarchinit_photogrammetry.io import gpkg

    path = path_of(chunk, run, "gpkg")
    if path is None or not path.exists():
        return []
    return [(g, a) for g, a in gpkg.read_table(path, TABLE) if g is not None and not g.is_empty]


def remove_objects(chunk, run: dict, numbers) -> int:
    """The objects ``numbers`` (their ``n``) out of the run: out of its GeoPackage, zero in its mask.
    Returns how many went."""
    from pyarchinit_photogrammetry.io import gpkg

    numbers = {int(n) for n in numbers}
    rows = features(chunk, run)
    gone = [(g, a) for g, a in rows if int(a.get("n") or 0) in numbers]
    if not gone:
        return 0
    kept = [(g, a) for g, a in rows if int(a.get("n") or 0) not in numbers]
    gpkg.write_table(path_of(chunk, run, "gpkg"), TABLE, "MULTIPOLYGON", COLUMNS, kept, getattr(chunk, "crs", None),
                     f"samgeo: {run.get('name')}")
    mask = path_of(chunk, run, "mask")
    if mask is not None and mask.exists():
        import rasterio
        from rasterio.windows import Window, from_bounds

        with rasterio.open(mask, "r+") as dst:
            for g, a in gone:
                w = from_bounds(*g.bounds, transform=dst.transform)  # its pixels, one more all around
                c0, r0 = max(int(math.floor(w.col_off)) - 1, 0), max(int(math.floor(w.row_off)) - 1, 0)
                c1 = min(int(math.ceil(w.col_off + w.width)) + 1, dst.width)
                r1 = min(int(math.ceil(w.row_off + w.height)) + 1, dst.height)
                if c1 <= c0 or r1 <= r0:
                    continue
                win = Window(c0, r0, c1 - c0, r1 - r0)
                arr = dst.read(1, window=win)
                arr[arr == int(a["n"])] = 0
                dst.write(arr, 1, window=win)
    run["count"] = len(kept)
    if run.get("units_made"):
        run["units_made"] = [n for n in run["units_made"] if n not in numbers]
    return len(gone)


def remove(chunk, run_id: str) -> dict | None:
    """The run out of the project, its files deleted. Returns its record (None when unknown)."""
    run = record(chunk, run_id)
    if run is None:
        return None
    for kind in ("gpkg", "mask"):
        p = path_of(chunk, run, kind)
        if p is not None:
            for f in (p, p.with_name(p.name + ".aux.xml")):
                try:
                    f.unlink(missing_ok=True)
                except OSError:
                    pass
    meta = chunk.meta.setdefault("samgeo", {})
    meta["runs"] = [r for r in records(chunk) if r.get("id") != run_id]
    if not meta["runs"]:
        chunk.meta.pop("samgeo", None)
    return run


def export(chunk, path, run_ids=None) -> Path:
    """The runs (all, or ``run_ids``) into one GeoPackage, a table each named by its id (its layer in
    QGIS named as on the view)."""
    from pyarchinit_photogrammetry.io import gpkg

    chosen = [r for r in records(chunk) if run_ids is None or r.get("id") in run_ids]
    if not chosen:
        raise ValueError(t("Nessuna ricerca da esportare", "No run to export"))
    for run in chosen:
        gpkg.write_table(path, run["id"], "MULTIPOLYGON", COLUMNS, features(chunk, run), getattr(chunk, "crs", None),
                         layer(run))
    return Path(path)


def unit_rows(run: dict, rows) -> tuple[list, list[dict]]:
    """The polygons and the attributes of the units made of ``rows`` (objects of the run): the text and
    the score of each kept with the unit."""
    geoms = [g for g, _a in rows]
    attrs = [{"samgeo_prompt": run.get("text", ""), "samgeo_score": round(float(a.get("score") or 0.0), 3)}
             for _g, a in rows]
    return geoms, attrs


def class_name(run: dict) -> str:
    return f"samgeo: {run.get('name', '?')}"


def classify(chunk, run: dict, numbers=None, keep_manual: bool = True, progress=None) -> dict:
    """The points of the dense cloud inside the objects of the run (``numbers``: only those; in plan
    view) get the class of the run, ``samgeo: <name>``: the code it has already, else the first free
    one from :data:`FIRST_CLASS`; the points of its earlier extent no longer inside go back to 1
    (unclassified), as for the classes of the units. With ``keep_manual`` the points labelled by hand
    keep their class. Returns ``{"name", "code", "points"}``."""
    import shapely
    from shapely.ops import unary_union

    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    rows = features(chunk, run)
    if numbers is not None:
        wanted = {int(n) for n in numbers}
        rows = [(g, a) for g, a in rows if int(a.get("n") or 0) in wanted]
    if not rows:
        raise ValueError(t("Nessun oggetto da classificare", "No object to classify"))
    progress(0.05, t("Lettura della nuvola densa", "Reading the dense cloud"))
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError(t("Il chunk non ha la nuvola densa: costruiscila prima.",
                             "The chunk has no dense cloud: build it first."))
    name = class_name(run)
    names = {int(k): str(v) for k, v in (chunk.meta.get("classes") or {}).items()}
    code = next((c for c, n in names.items() if n == name), None)
    if code is None:
        from pyarchinit_photogrammetry.ai.classify import ARCHAEO_CLASSES, ASPRS_CLASSES

        used = set(ASPRS_CLASSES) | set(ARCHAEO_CLASSES) | set(names) | set(np.unique(pc.classification).tolist())
        code = next((c for c in range(FIRST_CLASS, 256) if c not in used), None)
        if code is None:
            raise ValueError(t("Non ci sono più codici di classe liberi (arrivano a 255)",
                               "No class codes are left (they go up to 255)"))
    progress(0.3, t("Punti dentro gli oggetti", "Points inside the objects"))
    shape = unary_union([g for g, _a in rows])
    shapely.prepare(shape)
    world = chunk.to_world(pc.xyz)
    x0, y0, x1, y1 = shape.bounds
    near = np.flatnonzero((world[:, 0] >= x0) & (world[:, 0] <= x1) & (world[:, 1] >= y0) & (world[:, 1] <= y1))
    inside = np.zeros(len(world), bool)
    if len(near):
        inside[near] = shapely.intersects_xy(shape, world[near, 0], world[near, 1])
    free = np.ones(len(world), bool) if not keep_manual or pc.manual is None else ~pc.manual
    pc.classification[(pc.classification == code) & free & ~inside] = 1  # its earlier extent
    pc.classification[inside & free] = code
    progress(0.8, t("Scrittura della nuvola densa", "Writing the dense cloud"))
    chunk.point_cloud = pc
    chunk.meta.setdefault("classes", {})[str(code)] = name
    progress(1.0, name)
    return {"name": name, "code": int(code), "points": int((inside & free).sum())}


def summary(run: dict) -> str:
    """One line on a run: the text, how many objects, the raster, the resolution."""
    from .texts import number, plural

    cm = float(run.get("resolution") or 0.0) * 100
    n = int(run.get("count", 0))
    return t("«{0}»: {1} {2} su {3}, {4} cm/px", "'{0}': {1} {2} on {3}, {4} cm/px").format(
        run.get("text", "?"), n, plural(n, t("oggetto", "object"), t("oggetti", "objects")),
        run.get("raster_label") or run.get("raster", "?"), number(cm, 0 if math.isclose(cm, round(cm)) else 1))
