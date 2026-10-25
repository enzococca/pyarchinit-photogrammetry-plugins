"""From a raster and a text to polygons, in the program's Python (the models run in the worker).

1. :func:`plan`: the area asked (the whole raster, the bounds of a polygon or of the part shown) at
   the resolution chosen - the raster's, or coarser: 1 cm/px is plenty for stones - in tiles of
   1024 px overlapping by 128.
2. :func:`read_tile`: the window of a tile, read at that resolution (from the overviews): three
   bands of 8 bits as they are, any other raster (16 bits, reflectances, one band: a DEM or its relief
   visualisation) stretched between the 2nd and the 98th percentile of the area, one band as grey.
3. :func:`find`: each tile to the worker; the masks into polygons in the coordinates of the chunk;
   the same object seen by two neighbouring tiles (overlapping by more than half of the smaller in
   their common strip) joined into one, the best score kept; the duplicates of a tile (two boxes on
   one object) dropped; then the score, the area and, with a polygon, the place filters. When the area
   has more than one tile, a box as large as its tile is left out: Grounding DINO often answers "the
   whole image" beside the objects, and those boxes, joined from tile to tile, would cover the area.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TILE = 1024
OVERLAP = 128
DEFAULT_RESOLUTION = 0.01  # m/px: the default never reads a finer raster finer than this
SAME_OBJECT = 0.5  # share of the smaller part, in the strip two tiles have in common
DUPLICATE = 0.7  # intersection over union of two objects of one tile that are the same
WHOLE_TILE = 0.95  # a box this wide and this high in a tile is the tile, not an object (with more than one tile)


@dataclass
class Plan:
    """Where a run looks: ``bounds`` (xmin, ymin, xmax, ymax) at ``resolution`` (m/px), in ``tiles``
    (their bounds), on the raster ``path`` (``native``: its own pixel; ``crs``: its CRS as WKT)."""

    path: str
    bounds: tuple
    resolution: float
    native: float
    tiles: list = field(default_factory=list)
    crs: str | None = None
    polygon: object = None  # the shapely polygon drawn, when one was

    @property
    def size(self) -> tuple[int, int]:
        """Columns and rows of the area at the resolution of the run."""
        x0, y0, x1, y1 = self.bounds
        return max(1, int(math.ceil((x1 - x0) / self.resolution))), max(1, int(math.ceil((y1 - y0) / self.resolution)))


def tiles_of(bounds, resolution: float, tile: int = TILE, overlap: int = OVERLAP) -> list[tuple]:
    """World bounds of square tiles of ``tile`` pixels at ``resolution`` overlapping by ``overlap``
    pixels; the last ones of a row or a column are pushed back inside the area."""
    x0, y0, x1, y1 = bounds
    size, step = tile * resolution, (tile - overlap) * resolution
    nx = max(1, int(math.ceil(max(x1 - x0 - size, 0) / step - 1e-9)) + 1)
    ny = max(1, int(math.ceil(max(y1 - y0 - size, 0) / step - 1e-9)) + 1)
    out = []
    for j in range(ny):
        for i in range(nx):
            tx0 = min(x0 + i * step, max(x1 - size, x0))
            ty1 = max(y1 - j * step, min(y0 + size, y1))
            out.append((tx0, max(ty1 - size, y0), min(tx0 + size, x1), ty1))
    return out


def plan(path, bounds=None, resolution: float | None = None, polygon=None, tile: int = TILE,
         overlap: int = OVERLAP) -> Plan:
    """The tiles of a run on the raster ``path``: ``bounds`` (the whole raster when None) or the
    bounds of ``polygon`` (``(N, 2)`` vertices or a shapely polygon), clipped to the raster;
    ``resolution`` m/px (None: the raster's, at least :data:`DEFAULT_RESOLUTION`; never finer than
    the raster). Tiles that do not touch the polygon are left out. ``ValueError`` when the area is
    outside the raster."""
    import rasterio
    from shapely.geometry import Polygon, box
    from shapely.geometry.base import BaseGeometry

    with rasterio.open(path) as src:
        b = src.bounds
        native = float(min(abs(src.res[0]), abs(src.res[1])))
        crs = src.crs.to_wkt() if src.crs else None
    full = (b.left, b.bottom, b.right, b.top)
    shape = None
    if polygon is not None:
        shape = polygon if isinstance(polygon, BaseGeometry) else Polygon(np.asarray(polygon, float)[:, :2])
        if not shape.is_valid:  # an outline crossing itself: its parts
            from shapely.validation import make_valid

            shape = make_valid(shape)
            shape = shape.buffer(0) if shape.geom_type == "GeometryCollection" else shape
        if shape.is_empty or shape.area <= 0:
            raise ValueError("the polygon encloses no area")
        bounds = shape.bounds
    area = full if bounds is None else (max(bounds[0], full[0]), max(bounds[1], full[1]),
                                        min(bounds[2], full[2]), min(bounds[3], full[3]))
    if area[2] <= area[0] or area[3] <= area[1]:
        raise ValueError("the area is outside the raster")
    res = max(float(resolution) if resolution else max(native, DEFAULT_RESOLUTION), native)
    tiles = tiles_of(area, res, tile, overlap)
    if shape is not None:
        tiles = [tb for tb in tiles if shape.intersects(box(*tb))]
    return Plan(str(path), tuple(float(v) for v in area), res, native, tiles, crs, shape)


def _bands(src) -> list[int]:
    """The three bands shown as red, green and blue: those named so, else the first three; one band
    for a raster of one or two (the second being its alpha)."""
    if src.count < 3:
        return [1]
    names = [str(d or "").lower() for d in src.descriptions]
    found = [next((k + 1 for k, n in enumerate(names) if n.startswith(c) or n == c[0]), None)
             for c in ("red", "green", "blue")]
    if all(found):
        return found
    return [1, 2, 3]


def stretch(path, bounds, side: int = 1024) -> list[tuple[float, float]] | None:
    """The 2nd and the 98th percentile of each band shown, in ``bounds``, read at most ``side``
    pixels on the long side; None for three bands of 8 bits, shown as they are."""
    import rasterio
    from rasterio.enums import Resampling

    with rasterio.open(path) as src:
        bands = _bands(src)
        if len(bands) == 3 and all(src.dtypes[b - 1] == "uint8" for b in bands):
            return None
        win = _window(src, bounds)
        f = max(1.0, max(win.width, win.height) / side)
        shape = (len(bands), max(1, int(win.height / f)), max(1, int(win.width / f)))
        data = src.read(bands, window=win, out_shape=shape, resampling=Resampling.average, masked=True)
    out = []
    for k in range(len(bands)):
        v = np.ma.compressed(data[k]).astype(float)
        v = v[np.isfinite(v)]
        if not len(v):
            out.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(v, [2, 98])
        out.append((float(lo), float(hi) if hi > lo else float(lo) + 1.0))
    return out


def _window(src, bounds):
    from rasterio.errors import WindowError
    from rasterio.windows import Window, from_bounds

    try:
        return from_bounds(*bounds, transform=src.transform).intersection(Window(0, 0, src.width, src.height))
    except WindowError as exc:
        raise ValueError("the area is outside the raster") from exc


def read_tile(path, bounds, resolution: float, levels=None):
    """The tile ``bounds`` of the raster at ``resolution`` m/px: an RGB image of 8 bits ``(H, W, 3)``,
    the mask of its valid pixels and its transform. ``levels``: the stretch of :func:`stretch`."""
    import rasterio
    from affine import Affine
    from rasterio.enums import Resampling

    x0, y0, x1, y1 = bounds
    w = max(1, int(round((x1 - x0) / resolution)))
    h = max(1, int(round((y1 - y0) / resolution)))
    tr = Affine(resolution, 0.0, x0, 0.0, -resolution, y1)
    with rasterio.open(path) as src:
        bands = _bands(src)
        from rasterio.windows import from_bounds

        win = from_bounds(x0, y0, x1, y1, transform=src.transform)
        coarser = resolution > float(min(abs(src.res[0]), abs(src.res[1]))) * 1.01
        rs = Resampling.average if coarser else Resampling.bilinear
        data = src.read(bands, window=win, out_shape=(len(bands), h, w), resampling=rs, boundless=True, fill_value=0,
                        masked=False).astype(np.float32)
        valid = src.read_masks(1, window=win, out_shape=(h, w), resampling=Resampling.nearest, boundless=True) > 0
    if levels is None:
        img = np.clip(data, 0, 255)
    else:
        img = np.stack([(data[k] - lo) / (hi - lo) * 255.0 for k, (lo, hi) in enumerate(levels)])
    img = np.clip(np.nan_to_num(img), 0, 255).astype(np.uint8)
    valid &= np.all(np.isfinite(data), axis=0)
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.moveaxis(img, 0, -1)
    img[~valid] = 0
    return np.ascontiguousarray(img), valid, tr


def _polygons(mask, transform, resolution: float):
    """A mask cut to its box into a shapely (Multi)Polygon: parts smaller than a few pixels left
    out, the outline simplified by one pixel."""
    from pyarchinit_photogrammetry.archaeo.stratigraphy import mask_to_polygons

    geom = mask_to_polygons(mask, transform, simplify=resolution, min_area=9 * resolution * resolution)
    parts = [g for g in getattr(geom, "geoms", [geom]) if not g.is_empty and g.is_valid and g.area > 0]
    if not parts:
        return None
    from shapely.geometry import MultiPolygon

    return parts[0] if len(parts) == 1 else MultiPolygon(parts)


def find(segmenter, the_plan: Plan, prompt: str, threshold: float = 0.3, text_threshold: float = 0.25,
         min_area: float = 0.0, max_area: float = 0.0, progress=None) -> list[dict]:
    """The objects of ``prompt`` in the area of ``the_plan``: dicts ``geometry`` (shapely, in the
    coordinates of the raster), ``score``, ``quality``, ``label``, the best score first.
    ``min_area``, ``max_area``: m2 (0: no limit)."""
    from affine import Affine

    from pyarchinit_photogrammetry.core.progress import as_progress

    from .client import unpack

    progress = as_progress(progress)
    levels = stretch(the_plan.path, the_plan.bounds)
    res = the_plan.resolution
    found = []  # (tile index, object)
    n = len(the_plan.tiles)
    for k, tb in enumerate(the_plan.tiles):
        progress(0.02 + 0.9 * k / max(n, 1), f"samgeo: {k + 1} / {n}")
        img, valid, tr = read_tile(the_plan.path, tb, res, levels)
        if valid.mean() < 0.02:
            continue
        h, w = valid.shape
        for obj in segmenter.segment(img, prompt, threshold, text_threshold, progress=progress):
            bx = obj.get("box") or (0, 0, 0, 0)
            if n > 1 and bx[2] - bx[0] >= WHOLE_TILE * w and bx[3] - bx[1] >= WHOLE_TILE * h:
                continue
            r0, c0, mask = unpack(obj)
            mask = mask & valid[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
            if mask.sum() < 9:
                continue
            geom = _polygons(mask, tr @ Affine.translation(c0, r0), res)
            if geom is not None:
                found.append((k, {"geometry": geom, "score": float(obj.get("score", 0.0)),
                                  "quality": float(obj["quality"]) if obj.get("quality") is not None else None, "label": str(obj.get("label", ""))}))
    progress(0.93, f"samgeo: {len(found)}")
    objects = merge(found, the_plan.tiles)
    kept = []
    for o in objects:
        g = o["geometry"]
        if o["score"] < threshold or g.area < min_area or (max_area and g.area > max_area):
            continue
        if the_plan.polygon is not None and not the_plan.polygon.intersects(g.representative_point()):
            continue
        kept.append(o)
    progress(1.0, f"samgeo: {len(kept)}")
    return sorted(kept, key=lambda o: (-o["score"], -o["geometry"].area))


def merge(found, tiles) -> list[dict]:
    """The objects of the tiles (``(tile index, object)``) as objects of the area: an object cut by
    the border of a tile is joined with its other part from the neighbouring tile (they agree in the
    strip the two tiles share); two objects of one tile that are the same (intersection over union
    above :data:`DUPLICATE`), the one with the lower score dropped."""
    from shapely.geometry import box
    from shapely.ops import unary_union
    from shapely.strtree import STRtree

    if not found:
        return []
    owner = [k for k, _ in found]
    objs = [o for _, o in found]
    geoms = [o["geometry"] for o in objs]
    parent = list(range(len(objs)))

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    tree = STRtree(geoms)
    for i, g in enumerate(geoms):
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i or owner[i] == owner[j]:
                continue
            ta, tb = tiles[owner[i]], tiles[owner[j]]
            common = box(max(ta[0], tb[0]), max(ta[1], tb[1]), min(ta[2], tb[2]), min(ta[3], tb[3]))
            if common.is_empty or common.area <= 0:
                continue
            a, c = g.intersection(common), geoms[j].intersection(common)
            if a.is_empty or c.is_empty:
                continue
            if a.intersection(c).area > SAME_OBJECT * min(a.area, c.area):
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(objs)):
        groups.setdefault(root(i), []).append(i)
    merged = []
    for members in groups.values():
        best = max(members, key=lambda i: objs[i]["score"])
        geom = unary_union([geoms[i] for i in members]) if len(members) > 1 else geoms[best]
        if not geom.is_valid:
            geom = geom.buffer(0)
        if geom.is_empty:
            continue
        quality = max((objs[i]["quality"] for i in members if objs[i]["quality"] is not None), default=None)
        merged.append({**objs[best], "geometry": geom, "quality": quality})
    merged.sort(key=lambda o: -o["score"])
    kept: list[dict] = []
    tree = STRtree([o["geometry"] for o in merged])
    dropped = set()
    for i, o in enumerate(merged):  # the best score first: a later one that is the same object goes
        if i in dropped:
            continue
        kept.append(o)
        g = o["geometry"]
        for j in tree.query(g, predicate="intersects"):
            j = int(j)
            if j <= i or j in dropped:
                continue
            h = merged[j]["geometry"]
            inter = g.intersection(h).area
            union = g.area + h.area - inter
            if union > 0 and inter / union > DUPLICATE:
                dropped.add(j)
    return kept
