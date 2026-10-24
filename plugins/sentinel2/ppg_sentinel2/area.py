"""The area of the analysis and the grid its scenes are read on.

The area is the footprint of the chunk's DEM or orthomosaic (the cells with data, as a convex
polygon), a polygon drawn on the view, or a box of longitudes and latitudes, in the coordinate
system of the chunk; a margin goes around it, because the background of a crop-mark map needs
the ground around the site. The scenes are read on one grid: 10 m cells in the WGS84/UTM zone
of the area's centre, the corners on multiples of 20 m as those of the Sentinel-2 tiles, so a
20 m pixel is always four 10 m cells. A chunk in a local system (LOCAL) has no place on the
Earth: it cannot be searched, and :class:`NotPlaced` says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .texts import t

CELL = 10.0  # metres: the grid of the scenes
SNAP = 20.0  # its corners on multiples of 20 m (the 20 m pixels of the tiles never split)
MARGIN = 300.0  # metres around the area by default
MAX_SIDE = 2500  # cells: 25 km a side, more than the surroundings of any site need


class NotPlaced(RuntimeError):
    """The chunk has no coordinate system on the Earth (a local frame)."""


def not_placed_text() -> str:
    return t("Il chunk è in un sistema locale (LOCAL): non ha un posto sulla Terra e Sentinel-2 non si può cercare. "
             "Georeferenzia il chunk (marker con coordinate, GPS delle foto) oppure indica un riquadro in longitudine "
             "e latitudine.",
             "The chunk is in a local system (LOCAL): it has no place on the Earth and Sentinel-2 cannot be searched. "
             "Georeference the chunk (markers with coordinates, GPS of the photos) or give a box of longitudes and "
             "latitudes.")


def _is_local(crs) -> bool:
    return crs is None or str(crs).strip().upper() in ("", "LOCAL", "NONE")


def chunk_crs(chunk) -> str:
    """The coordinate system of the chunk as text (``EPSG:32633``, a WKT); that of its DEM or
    orthomosaic when the chunk says nothing; :class:`NotPlaced` for a local chunk."""
    crs = getattr(chunk, "crs", None)
    if not _is_local(crs):
        return str(crs)
    import rasterio

    for path in (getattr(chunk, "dem_path", None), getattr(chunk, "orthomosaic_path", None)):
        if path is not None and path.exists():
            with rasterio.open(path) as src:
                if src.crs is not None:
                    return src.crs.to_string()
    raise NotPlaced(not_placed_text())


def _transformer(src: str, dst: str):
    from pyproj import CRS, Transformer

    return Transformer.from_crs(CRS.from_user_input(src), CRS.from_user_input(dst), always_xy=True)


def reproject_geometry(geom, src: str, dst: str, step: float | None = None):
    """A shapely geometry from ``src`` into ``dst``, its edges cut every ``step`` units first so
    that they bend as they should."""
    import shapely

    if step:
        geom = shapely.segmentize(geom, step)
    tr = _transformer(src, dst)

    def move(xy):
        x, y = tr.transform(xy[:, 0], xy[:, 1])
        return np.column_stack([x, y])

    return shapely.transform(geom, move)


def utm_epsg(lon: float, lat: float) -> str:
    zone = min(60, max(1, int((lon + 180.0) // 6.0) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


@dataclass
class Area:
    """A polygon on the ground: ``ring`` (``[[x, y], ...]``) in ``crs``, where it came from
    (``dem``, ``orthomosaic``, ``polygon``, ``bbox``) and the margin around it in metres."""

    crs: str
    ring: list
    source: str = "dem"
    margin_m: float = MARGIN

    def polygon(self):
        from shapely.geometry import Polygon

        return Polygon([(float(x), float(y)) for x, y in self.ring])

    def _step(self) -> float:
        x0, y0, x1, y1 = self.polygon().bounds
        return max(x1 - x0, y1 - y0) / 32.0 or 1.0

    def lonlat(self):
        """The polygon in longitude and latitude (WGS84)."""
        return reproject_geometry(self.polygon(), self.crs, "EPSG:4326", self._step())

    def utm_crs(self) -> str:
        c = self.lonlat().centroid
        return utm_epsg(c.x, c.y)

    def in_crs(self, crs: str, margin: bool = False):
        """The polygon in ``crs`` (with the margin, made in metres in the UTM zone)."""
        utm = self.utm_crs()
        poly = reproject_geometry(self.polygon(), self.crs, utm, self._step())
        if margin and self.margin_m > 0:
            poly = poly.buffer(float(self.margin_m), join_style=2)
        if crs == utm:
            return poly
        x0, y0, x1, y1 = poly.bounds
        return reproject_geometry(poly, utm, crs, max(x1 - x0, y1 - y0) / 32.0 or 1.0)

    def lonlat_bbox(self, margin: bool = True) -> tuple[float, float, float, float]:
        """West, south, east, north in degrees (with the margin): the box of the search."""
        return tuple(float(v) for v in self.in_crs("EPSG:4326", margin=margin).bounds)

    def size_m(self) -> tuple[float, float]:
        x0, y0, x1, y1 = self.in_crs(self.utm_crs()).bounds
        return x1 - x0, y1 - y0

    def to_dict(self) -> dict:
        return {"crs": self.crs, "ring": [[float(x), float(y)] for x, y in self.ring], "source": self.source,
                "margin_m": float(self.margin_m)}

    @classmethod
    def from_dict(cls, d: dict) -> "Area":
        return cls(str(d["crs"]), [list(map(float, p)) for p in d["ring"]], str(d.get("source", "polygon")),
                   float(d.get("margin_m", MARGIN)))


def footprint(path) -> tuple[list, str | None]:
    """The cells with data of a raster as a convex polygon (read at most 512 cells a side) in
    the raster's coordinates, and its coordinate system."""
    import rasterio
    from rasterio import features
    from rasterio.transform import Affine
    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    with rasterio.open(path) as src:
        f = max(1.0, max(src.width, src.height) / 512.0)
        h, w = max(1, int(round(src.height / f))), max(1, int(round(src.width / f)))
        mask = src.dataset_mask(out_shape=(h, w)) > 0
        tr = src.transform @ Affine.scale(src.width / w, src.height / h)
        crs = src.crs.to_string() if src.crs is not None else None
        bounds = src.bounds
    shapes = [shape(g) for g, v in features.shapes(mask.astype(np.uint8), mask=mask, transform=tr) if v == 1]
    geom = unary_union(shapes).convex_hull if shapes else box(*bounds)
    if geom.is_empty or geom.area <= 0:
        geom = box(*bounds)
    return [[float(x), float(y)] for x, y in list(geom.exterior.coords)[:-1]], crs


def area_of_chunk(chunk, source: str = "auto", margin_m: float = MARGIN) -> Area:
    """The footprint of the chunk's DEM (``dem``) or orthomosaic (``orthomosaic``; ``auto``: the
    DEM, else the orthomosaic) in the chunk's coordinate system."""
    crs = chunk_crs(chunk)
    paths = {"dem": getattr(chunk, "dem_path", None), "orthomosaic": getattr(chunk, "orthomosaic_path", None)}
    order = ["dem", "orthomosaic"] if source == "auto" else [source]
    for name in order:
        path = paths.get(name)
        if path is not None and path.exists():
            ring, raster_crs = footprint(path)
            if raster_crs is not None and not _same_crs(raster_crs, crs):
                from shapely.geometry import Polygon

                poly = reproject_geometry(Polygon(ring), raster_crs, crs)
                ring = [list(p) for p in list(poly.exterior.coords)[:-1]]
            return Area(crs, ring, name, margin_m)
    raise RuntimeError(t("Il chunk non ha né DEM né ortofoto: disegna l'area sulla vista o indica un riquadro.",
                         "The chunk has neither a DEM nor an orthomosaic: draw the area on the view or give a box."))


def area_of_polygon(chunk, xy, margin_m: float = MARGIN) -> Area:
    """A polygon drawn on the view (coordinates of the chunk)."""
    xy = np.asarray(xy, float)[:, :2]
    if len(xy) < 3:
        x0, y0 = xy.min(axis=0)
        x1, y1 = xy.max(axis=0)
        xy = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return Area(chunk_crs(chunk), xy.tolist(), "polygon", margin_m)


def area_of_bbox(chunk, west: float, south: float, east: float, north: float, margin_m: float = MARGIN) -> Area:
    """A box of longitudes and latitudes, in the chunk's coordinate system (in WGS84 for a local
    chunk: then nothing goes on the chunk's grid)."""
    if not (east > west and north > south):
        raise ValueError("the box needs west < east and south < north")
    from shapely.geometry import box

    try:
        crs = chunk_crs(chunk)
    except NotPlaced:
        crs = "EPSG:4326"
    poly = box(west, south, east, north)
    if crs != "EPSG:4326":
        poly = reproject_geometry(poly, "EPSG:4326", crs, max(east - west, north - south) / 32.0)
    return Area(crs, [list(p) for p in list(poly.exterior.coords)[:-1]], "bbox", margin_m)


def _same_crs(a: str, b: str) -> bool:
    from pyproj import CRS

    try:
        return CRS.from_user_input(a) == CRS.from_user_input(b)
    except Exception:  # noqa: BLE001
        return str(a) == str(b)


@dataclass(frozen=True)
class Grid:
    """A north-up grid: ``crs``, the corner ``(x0, y1)`` (west, north), the ``cell`` and the size."""

    crs: str
    x0: float
    y1: float
    cell: float
    width: int
    height: int

    @property
    def transform(self):
        from rasterio.transform import Affine

        return Affine(self.cell, 0.0, self.x0, 0.0, -self.cell, self.y1)

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y1 - self.height * self.cell, self.x0 + self.width * self.cell, self.y1

    @property
    def shape(self) -> tuple[int, int]:
        return self.height, self.width

    def to_dict(self) -> dict:
        return {"crs": self.crs, "x0": self.x0, "y1": self.y1, "cell": self.cell, "width": self.width,
                "height": self.height}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(str(d["crs"]), float(d["x0"]), float(d["y1"]), float(d["cell"]), int(d["width"]), int(d["height"]))

    def same(self, transform, crs, shape) -> bool:
        """The grid of a raster (its transform, CRS and (rows, columns)) is this one."""
        a = self.transform
        return (tuple(shape) == self.shape and all(abs(p - q) < 1e-6 for p, q in zip(tuple(transform)[:6], tuple(a)[:6]))
                and crs is not None and _same_crs(str(crs), self.crs))


def grid_of(area: Area, cell: float = CELL, snap: float = SNAP) -> Grid:
    """The grid of the scenes over the area and its margin (UTM zone of the area's centre)."""
    crs = area.utm_crs()
    x0, y0, x1, y1 = area.in_crs(crs, margin=True).bounds
    x0, y0 = math.floor(x0 / snap) * snap, math.floor(y0 / snap) * snap
    x1, y1 = math.ceil(x1 / snap) * snap, math.ceil(y1 / snap) * snap
    width, height = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
    if max(width, height) > MAX_SIDE:
        raise RuntimeError(t("L'area è troppo grande per Sentinel-2 in questo plugin ({0:.0f} x {1:.0f} km): al massimo "
                             "{2:.0f} km di lato.", "The area is too large for Sentinel-2 in this plugin ({0:.0f} x "
                             "{1:.0f} km): {2:.0f} km a side at most.").format(width * cell / 1000, height * cell / 1000,
                                                                              MAX_SIDE * cell / 1000))
    return Grid(crs, float(x0), float(y1), float(cell), width, height)


def area_mask(area: Area, grid: Grid, margin: bool = False) -> np.ndarray:
    """The cells of ``grid`` the area touches (``margin``: with its margin)."""
    from rasterio import features

    poly = area.in_crs(grid.crs, margin=margin)
    mask = features.rasterize([(poly, 1)], out_shape=grid.shape, transform=grid.transform, fill=0,
                              all_touched=True, dtype="uint8")
    return mask.astype(bool)
