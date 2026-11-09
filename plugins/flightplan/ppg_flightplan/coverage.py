"""The coverage a plan is expected to give, before flying: how many photos see every cell of the ground and the
finest ground resolution a photo gives there.

Not in the planner the plugin comes from: its strips and steps assume flat ground at the takeoff's height,
in its flat-earth metres. Here every shot is put back in the chunk's coordinate system with pyproj, as a
pinhole camera with its heading and pitch, over the ground of the project DEM (or a flat ground at the
takeoff when there is none), and the footprints are counted on a grid of cells. This is the truth the user
sees: where the terrain rises or the strips are short, it shows the overlap the plan really gives.

The camera model: the optical axis ``d = (sin psi cos th, cos psi cos th, sin th)`` (``psi`` the heading from the
grid north clockwise, ``th`` the pitch, -90 straight down); the image's x axis is the sensor's WIDTH (its long
side), across the track as in the planner's strip spacing, ``r = (cos psi, -sin psi, 0)``; its y axis
``u = r x d``. A cell ``P`` is seen by a shot at ``C`` when ``z_c = (P-C).d > 0``,
``|(P-C).r| / z_c <= width / (2 f)`` and ``|(P-C).u| / z_c <= height / (2 f)``. Nothing hides anything (no
occlusion: said in the statistics).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from .geometry import LatLon, distance, interpolate
from .route import Route
from .texts import t

LOW = 5            # fewer photos than this on a cell: the coverage is low (the design says "meno di 5 foto")
CELLS = 400        # cells on the long side of the grid by default, as the program's field check
SHOTS_PER_BLOCK = 64
CELLS_PER_CHUNK = 32768
MAX_LOOK_DEGREES = 75.0   # how far from the vertical the grid reaches for an oblique view (tan 75 = 3.7 x height)

# The colours of the RGBA raster: no photo, fewer than ``low``, up to twice ``low``, more.
RAMP = ((215, 48, 39), (244, 146, 52), (190, 210, 70), (40, 150, 70))
ALPHA = 200


class CoverageError(ValueError):
    """The coverage cannot be computed (a geographic or local system, no shot)."""


@dataclass
class Coverage:
    counts: np.ndarray            # int16 (h, w): photos seeing the cell
    gsd_cm: np.ndarray            # float32 (h, w): finest GSD over the shots seeing it, NaN where none does
    transform: object             # rasterio Affine of the grid
    crs: str                      # the chunk's system the grid is in
    inside: np.ndarray            # bool (h, w): cells inside the active areas (or seen at all, without areas)
    surface: str                  # "dem" or "flat"
    stats: dict = field(default_factory=dict)
    cell: float = 0.0
    low: int = LOW
    shots: int = 0


@dataclass(frozen=True)
class Shot:
    """A photo of the plan: where (lat, lon) and how high above the ground of its takeoff, where it points."""
    position: LatLon
    height: float                 # above the takeoff of its leg
    pitch: float                  # degrees, -90 straight down
    toward: Optional[LatLon]      # a point it looks at (horizontally), or None
    along: Optional[tuple[LatLon, LatLon]]   # the segment it flies on (its heading), or None
    direction: Optional[float]    # a nose direction given by the plan (degrees from north), or None
    high: bool
    ground: Optional[float] = None   # the ground of its takeoff in the height system of the surface


def _pitch_of(route: Route, i: int) -> float:
    s = route.stop(i)
    if s is not None:
        if s.pitch is not None:
            return float(s.pitch)
        for a in reversed(s.actions):
            if a.startswith("camera:"):
                try:
                    return float(a.split(":", 1)[1])
                except ValueError:
                    pass
    return float(route.pitch)


def _segment(route: Route, i: int) -> Optional[tuple[LatLon, LatLon]]:
    """The segment the drone flies at waypoint ``i``: towards the next one, or from the previous at the end."""
    pts = route.points
    for j in range(i + 1, len(pts)):
        if distance(pts[i], pts[j]) > 0.01:
            return pts[i], pts[j]
    for j in range(i - 1, -1, -1):
        if distance(pts[j], pts[i]) > 0.01:
            return pts[j], pts[i]
    return None


def route_shots(route: Route, high: bool, ground: Optional[float] = None) -> list[Shot]:
    """The photos of a route, as :func:`ppg_flightplan.route.shot_points` places them (the same shots in the
    same order), each with where it points: the route's pitch (a special point its own), the heading along the
    track (towards the aim, ``poi``, when the route has one; the nose direction of a special point when given)."""
    pts = route.points
    out: list[Shot] = []

    def shot(p: LatLon, h: float, i: int, along) -> Shot:
        s = route.stop(i)
        direction = s.direction if s is not None and s.direction is not None else None
        return Shot(p, float(h), _pitch_of(route, i), route.poi, along, direction, high, ground)

    if route.mode == "per_punto":
        video = route.in_video
        for i, p in enumerate(pts):
            along = _segment(route, i)
            if route.shoots_by_itself(i, video):
                out.append(shot(p, route.height_at(i), i, along))
            if (s := route.stop(i)) is not None:
                out.extend(shot(p, route.height_at(i), i, along) for _ in range(s.actions.count("foto")))
        return out
    if not pts:
        return out
    if len(pts) == 1:
        out.append(shot(pts[0], route.height_at(0), 0, None))
    step = max(route.effective_step, 0.5)
    travelled = 0.0
    next_shot = 0.0
    for i in range(len(pts)):
        if i > 0:
            a, b = pts[i - 1], pts[i]
            seg = distance(a, b)
            ha, hb = route.height_at(i - 1), route.height_at(i)
            while seg > 0 and next_shot <= travelled + seg + 1e-9:
                f = min(1.0, (next_shot - travelled) / seg)
                out.append(Shot(interpolate(a, b, f), ha + (hb - ha) * f, float(route.pitch), route.poi, (a, b),
                                None, high, ground))
                next_shot += step
            travelled += seg
        if (s := route.stop(i)) is not None:
            out.extend(shot(pts[i], route.height_at(i), i, _segment(route, i)) for _ in range(s.actions.count("foto")))
    return out


def plan_shots(piano: dict, ground_takeoff: Optional[float] = None, leg_ground=None) -> list[Shot]:
    """Every photo of the active blocks, leg by leg (the heights above the takeoff each leg flies from), with
    the ground of that takeoff: ``ground_takeoff`` for the plan's, ``leg_ground(leg)`` for a leg with its own
    (None: the plan's)."""
    from . import plan as pl

    blocks = piano.get("blocchi", [])
    out: list[Shot] = []
    for leg in pl.plan_legs(piano):
        b = blocks[leg.block_index]
        high = bool(b.get("alta", piano.get("alta", False)))
        ground = leg_ground(leg) if leg_ground is not None else None
        out += route_shots(leg.route, high, ground if ground is not None else ground_takeoff)
    return out


# ------------------------------------------------------------------ the grid

def _projected(crs: str):
    """The horizontal part of ``crs`` as pyproj reads it, refused unless projected in metres."""
    from pyproj import CRS

    try:
        system = CRS.from_user_input(crs)
    except Exception as e:  # noqa: BLE001 - "LOCAL", an empty string, garbage
        raise CoverageError(t("serve un sistema di riferimento proiettato: «{0}» non lo è ({1})",
                              "a projected reference system is needed: '{0}' is not one ({1})").format(crs, e)) from None
    if system.is_compound:
        system = system.sub_crs_list[0]
    if not system.is_projected:
        raise CoverageError(t("serve un sistema di riferimento proiettato, in metri: «{0}» è in gradi",
                              "a projected reference system in metres is needed: '{0}' is in degrees")
                            .format(system.name))
    return system


def _look_reach(cam, shot: Shot) -> float:
    """How far from below the shot its footprint can reach on flat ground, metres (for the frame of the grid)."""
    half = math.degrees(math.atan(math.hypot(cam.width_mm, cam.height_mm) / (2 * cam.focal_mm)))
    off = 90.0 - abs(shot.pitch) if shot.pitch < 0 else 90.0
    angle = min(MAX_LOOK_DEGREES, off + half)
    return max(abs(shot.height), 1.0) * math.tan(math.radians(angle))


def _heights_on_grid(dem, crs, transform, width: int, height: int) -> Optional[np.ndarray]:
    """The DEM on the grid (bilinear), NaN where it has no data; None when it covers nothing of it."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT

    path = getattr(dem, "path", dem)
    with rasterio.open(path) as src:
        if src.crs is None:
            return None
        kw = dict(crs=crs, transform=transform, width=width, height=height, resampling=Resampling.bilinear,
                  dtype="float32", nodata=float("nan"))
        if src.nodata is not None:
            kw["src_nodata"] = src.nodata
        with WarpedVRT(src, **kw) as vrt:
            z = vrt.read(1).astype("float32")
    z[~np.isfinite(z) | (z < -1000.0)] = np.nan
    return z if np.isfinite(z).any() else None


def predict(piano: dict, crs: str, dem=None, ground_takeoff: Optional[float] = None, cell: Optional[float] = None,
            bounds=None, progress: Optional[Callable] = None, low: int = LOW, leg_ground=None,
            shots: Optional[list[Shot]] = None) -> Coverage:
    """The coverage of ``piano`` on a grid in ``crs`` (projected, else :class:`CoverageError`).

    The grid spans the active areas and the shots, grown by how far a photo reaches; ``cell`` metres (else the
    long side over :data:`CELLS`); ``bounds`` (xmin, ymin, xmax, ymax in ``crs``) replaces it. The ground of
    every cell comes from ``dem`` (a GeoTIFF path or a :class:`~ppg_flightplan.dem.RasterDem`, read on the grid)
    with the shots at ``ground_takeoff`` + their height above the takeoff, in the DEM's height system; without a
    DEM (or without ``ground_takeoff``) the ground is flat at the takeoff (``surface`` "flat"). ``progress(fraction,
    text)`` is called for every block of shots; an exception it raises (cancel) stops the work."""
    import rasterio.features
    from pyproj import Transformer
    from rasterio.transform import from_origin

    from . import plan as pl

    system = _projected(crs)
    cam = pl.plan_camera(piano)
    if shots is None:
        shots = plan_shots(piano, ground_takeoff, leg_ground)
    if not shots:
        raise CoverageError(t("il piano non ha scatti: nessun blocco acceso", "the plan has no shots: no active block"))
    to_xy = Transformer.from_crs("EPSG:4326", system, always_xy=True)

    def xy(points: list[LatLon]) -> np.ndarray:
        if not points:
            return np.zeros((0, 2))
        lat = np.array([p[0] for p in points], dtype=float)
        lon = np.array([p[1] for p in points], dtype=float)
        x, y = to_xy.transform(lon, lat)
        return np.column_stack([np.asarray(x, float), np.asarray(y, float)])

    centres = xy([s.position for s in shots])
    areas = [xy(a) for a in pl.active_areas(piano) if len(a) >= 3]
    if bounds is None:
        reach = max(max(_look_reach(cam, s) for s in shots), cam.footprint(max(abs(s.height) for s in shots)))
        every = np.vstack([centres] + areas)
        xmin, ymin = every.min(axis=0) - reach
        xmax, ymax = every.max(axis=0) + reach
    else:
        xmin, ymin, xmax, ymax = (float(v) for v in bounds)
    long_side = max(xmax - xmin, ymax - ymin)
    cell = float(cell) if cell else long_side / CELLS
    if not cell > 0:
        raise CoverageError(t("la cella della griglia deve essere positiva", "the cell of the grid must be positive"))
    width = max(1, int(math.ceil((xmax - xmin) / cell - 1e-6)))
    height = max(1, int(math.ceil((ymax - ymin) / cell - 1e-6)))
    transform = from_origin(xmin, ymax, cell, cell)
    crs_text = system.to_wkt()

    # the ground of the cells
    flat_at = 0.0 if ground_takeoff is None else float(ground_takeoff)
    ground = None
    if dem is not None and ground_takeoff is not None:
        ground = _heights_on_grid(dem, system, transform, width, height)
    surface = "dem" if ground is not None else "flat"
    without_dem = 0.0
    if ground is None:
        ground = np.full((height, width), flat_at, dtype="float32")
    else:
        missing = ~np.isfinite(ground)
        without_dem = float(missing.mean())
        ground[missing] = flat_at

    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    px = (xmin + (cols.ravel() + 0.5) * cell).astype("float64")
    py = (ymax - (rows.ravel() + 0.5) * cell).astype("float64")
    pz = ground.ravel().astype("float64")

    # the poses of the shots
    n = len(shots)
    cz = np.array([(s.ground if s.ground is not None else flat_at) + s.height for s in shots], dtype=float)
    if surface == "flat":      # flat ground: the shots stand at their height above the takeoff
        cz = np.array([flat_at + s.height for s in shots], dtype=float)
    psi = np.zeros(n)
    for k, s in enumerate(shots):
        if s.direction is not None:
            psi[k] = math.radians(s.direction)      # a nose given by the plan: from the true north, near enough
            continue
        if s.toward is not None and distance(s.position, s.toward) > 0.01:
            a, b = centres[k], xy([s.toward])[0]
        elif s.along is not None:
            a, b = xy(list(s.along))
        else:
            continue
        psi[k] = math.atan2(b[0] - a[0], b[1] - a[1])
    th = np.radians(np.array([s.pitch for s in shots], dtype=float))
    d = np.column_stack([np.sin(psi) * np.cos(th), np.cos(psi) * np.cos(th), np.sin(th)])
    r = np.column_stack([np.cos(psi), -np.sin(psi), np.zeros(n)])
    u = np.cross(r, d)
    tan_w = cam.width_mm / (2 * cam.focal_mm)
    tan_h = cam.height_mm / (2 * cam.focal_mm)
    gsd_factor = np.array([cam.width_mm / cam.pixels_for(s.high) / cam.focal_mm * 100.0 for s in shots])

    counts = np.zeros(px.size, dtype=np.int32)
    best = np.full(px.size, np.inf, dtype=np.float64)
    blocks = range(0, n, SHOTS_PER_BLOCK)
    for k, start in enumerate(blocks):
        if progress is not None:
            progress(k / max(len(blocks), 1), t("Copertura: scatti {0}-{1} di {2}", "Coverage: shots {0}-{1} of {2}")
                     .format(start + 1, min(n, start + SHOTS_PER_BLOCK), n))
        sl = slice(start, start + SHOTS_PER_BLOCK)
        c0, c1, c2 = centres[sl, 0][:, None], centres[sl, 1][:, None], cz[sl][:, None]
        d0, d1, d2 = d[sl, 0][:, None], d[sl, 1][:, None], d[sl, 2][:, None]
        r0, r1 = r[sl, 0][:, None], r[sl, 1][:, None]
        u0, u1, u2 = u[sl, 0][:, None], u[sl, 1][:, None], u[sl, 2][:, None]
        factor = gsd_factor[sl][:, None]
        for a in range(0, px.size, CELLS_PER_CHUNK):     # cells in chunks: a few tens of MB at most
            b = a + CELLS_PER_CHUNK
            dx = px[None, a:b] - c0
            dy = py[None, a:b] - c1
            dz = pz[None, a:b] - c2
            zc = dx * d0 + dy * d1 + dz * d2
            xr = dx * r0 + dy * r1
            yu = dx * u0 + dy * u1 + dz * u2
            seen = (zc > 0) & (np.abs(xr) <= tan_w * zc) & (np.abs(yu) <= tan_h * zc)
            counts[a:b] += seen.sum(axis=0)
            g = np.where(seen, zc * factor, np.inf)
            np.minimum(best[a:b], g.min(axis=0), out=best[a:b])
    if progress is not None:
        progress(1.0, t("Copertura calcolata", "Coverage computed"))

    counts = np.minimum(counts, np.iinfo(np.int16).max).astype(np.int16).reshape(height, width)
    gsd = np.where(np.isfinite(best), best, np.nan).astype(np.float32).reshape(height, width)
    if areas:
        shapes = [({"type": "Polygon", "coordinates": [[(float(x), float(y)) for x, y in a]
                                                       + [(float(a[0][0]), float(a[0][1]))]]}, 1) for a in areas]
        inside = rasterio.features.rasterize(shapes, out_shape=(height, width), transform=transform, fill=0,
                                             dtype="uint8").astype(bool)
    else:
        inside = counts > 0
    cov = Coverage(counts, gsd, transform, crs_text, inside, surface, cell=cell, low=int(low), shots=n)
    cov.stats = statistics(cov)
    if surface == "dem" and without_dem > 0:
        cov.stats["cells_without_dem"] = round(without_dem, 4)
    return cov


def statistics(cov: Coverage) -> dict:
    """``cells`` (inside), ``median`` and ``p10`` of the photos per cell, ``low_fraction`` (cells with fewer than
    ``low``), ``gsd_median_cm`` and ``gsd_max_cm`` (where seen), ``occlusion`` False (nothing hides the ground)."""
    c = cov.counts[cov.inside].astype(float)
    g = cov.gsd_cm[cov.inside]
    g = g[np.isfinite(g)]
    if not c.size:
        return {"cells": 0, "median": 0.0, "p10": 0.0, "low_fraction": 1.0, "gsd_median_cm": None,
                "gsd_max_cm": None, "occlusion": False}
    return {"cells": int(c.size), "median": round(float(np.median(c)), 1),
            "p10": round(float(np.percentile(c, 10)), 1),
            "low_fraction": round(float((c < cov.low).mean()), 4),
            "gsd_median_cm": round(float(np.median(g)), 3) if g.size else None,
            "gsd_max_cm": round(float(g.max()), 3) if g.size else None,
            "occlusion": False}


def low_polygons(cov: Coverage, threshold: Optional[int] = None, min_area_m2: Optional[float] = None) -> list:
    """The parts of the areas seen by fewer than ``threshold`` photos (the coverage's ``low``), as shapely
    polygons in the grid's system; pieces smaller than ``min_area_m2`` (four cells) left out."""
    import rasterio.features
    import shapely.geometry

    threshold = cov.low if threshold is None else int(threshold)
    min_area_m2 = 4 * cov.cell ** 2 if min_area_m2 is None else float(min_area_m2)
    mask = (cov.inside & (cov.counts < threshold)).astype("uint8")
    out = []
    for geom, value in rasterio.features.shapes(mask, mask=mask.astype(bool), transform=cov.transform):
        if value != 1:
            continue
        poly = shapely.geometry.shape(geom)
        if poly.area >= min_area_m2:
            out.append(poly)
    return out


def rgba(counts: np.ndarray, inside: np.ndarray, low: int = LOW) -> np.ndarray:
    """The colours of the coverage (4, h, w) uint8: red with no photo, orange under ``low``, yellow-green up to
    twice ``low``, green above; transparent outside ``inside``."""
    h, w = counts.shape
    out = np.zeros((4, h, w), dtype=np.uint8)
    classes = np.select([counts <= 0, counts < low, counts < 2 * low], [0, 1, 2], 3)
    for k, colour in enumerate(RAMP):
        sel = classes == k
        for band in range(3):
            out[band][sel] = colour[band]
    out[3][inside] = ALPHA
    return out


def write(cov: Coverage, folder, stem: str) -> dict:
    """``<stem>_count.tif`` (int16, -1 outside), ``<stem>_gsd.tif`` (float32 cm/px, NaN where unseen or outside)
    and ``<stem>_rgba.tif`` (the colours, transparent outside) in ``folder``, on the same grid: their paths."""
    import rasterio
    import rasterio.enums

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    h, w = cov.counts.shape
    base = dict(driver="GTiff", width=w, height=h, crs=cov.crs, transform=cov.transform, compress="deflate",
                BIGTIFF="IF_SAFER")
    paths = {"count": folder / f"{stem}_count.tif", "gsd": folder / f"{stem}_gsd.tif",
             "rgba": folder / f"{stem}_rgba.tif"}
    counts = np.where(cov.inside, cov.counts, -1).astype(np.int16)
    with rasterio.open(paths["count"], "w", count=1, dtype="int16", nodata=-1, **base) as ds:
        ds.write(counts, 1)
        ds.set_band_description(1, t("foto per cella", "photos per cell"))
    gsd = np.where(cov.inside, cov.gsd_cm, np.nan).astype(np.float32)
    with rasterio.open(paths["gsd"], "w", count=1, dtype="float32", nodata=float("nan"), **base) as ds:
        ds.write(gsd, 1)
        ds.set_band_description(1, "GSD cm/px")
    with rasterio.open(paths["rgba"], "w", count=4, dtype="uint8", photometric="RGB", **base) as ds:
        ds.write(rgba(cov.counts, cov.inside, cov.low))
        ds.colorinterp = [rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue, rasterio.enums.ColorInterp.alpha]
    return paths
