"""The heights of the ground under the drawn area, from the public terrain tiles.

They come from the "terrarium" tiles of the open data on AWS (Copernicus/SRTM, about 30 m): free, no key,
fetched like any map tile, so the mechanism is the same on Mac, Windows and Linux. Every pixel carries the
height coded in its colours. Without network nothing is made up: the heights are said to be missing.

The tiles are fetched only when a plan asks for the terrain, kept in memory and, when the plugin gives a
folder (:func:`set_cache_dir`), on disk: a second plan of the same site needs no network.

Ported from the author's Photo3D planner (heights, sampling, statistics): algorithms, constants and rounding
unchanged; the relief image and the contour lines of its web page are left out.
"""
from __future__ import annotations

import io
import math
import os
import threading
import urllib.request
from collections import OrderedDict
from pathlib import Path
from typing import Optional

from PIL import Image

from .texts import t

LatLon = tuple[float, float]

TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ATTRIBUTION = "Quote: Terrain Tiles (AWS Open Data) — Copernicus DEM / SRTM"
ZOOM = 13                      # ~16 m per pixel at mid latitudes: finer than the data (30 m)
# The data has holes: at some places the tile of zoom 14 is empty (270 bytes, all black) while the 13 and the
# 15 are there. When the height of a point is missing, these zooms are tried before giving up.
FALLBACK_ZOOMS = (13, 15, 12, 11, 10)
# How finely to sample. The data below has its own resolution (30 m in the Middle East, 10 m in the United
# States, 10-25 m in much of Europe): going finer shows no new terrain, but the heights pass from one point
# to the next without steps, and in flight it shows.
DETAILS = {"normale": 13, "fine": 14, "finissimo": 15}
TIMEOUT_S = 10.0
USER_AGENT = "pyarchinit-photogrammetry flightplan"


def zoom_for_detail(name: Optional[str]) -> int:
    return DETAILS.get((name or "").strip().lower(), ZOOM)


def metres_per_pixel(z: int, lat: float) -> float:
    """How many metres a tile pixel covers at that zoom and latitude."""
    return 156543.033928 * math.cos(math.radians(lat)) / (2 ** z)


# Below this height there is no land: it is the value the format uses for "no data" (black pixel = -32768)
# and it is never taken for a real height.
MIN_VALID_HEIGHT = -500.0
_cache: "OrderedDict[tuple[int, int, int], Optional[Image.Image]]" = OrderedDict()
_lock = threading.RLock()
_cache_dir: Optional[Path] = None


class TerrainError(RuntimeError):
    pass


def set_cache_dir(path) -> None:
    """Where to keep the tiles on disk (one PNG per ``z/x/y``); None keeps them only in memory."""
    global _cache_dir
    with _lock:
        _cache_dir = None if path is None else Path(path)


def _tile_from_network(z: int, x: int, y: int) -> Optional[Image.Image]:
    request = urllib.request.Request(TILES.format(z=z, x=x, y=y), headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as r:
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    except Exception:  # noqa: BLE001 - without network the answer is "heights not available"
        return None


def _empty(img: Image.Image) -> bool:
    """An all-black tile is a hole in the data, not a terrain at -32768 m."""
    return all(lo == 0 and hi == 0 for lo, hi in img.getextrema()[:3])


def _disk_path(z: int, x: int, y: int) -> Optional[Path]:
    return None if _cache_dir is None else _cache_dir / str(z) / str(x) / f"{y}.png"


def _from_disk(z: int, x: int, y: int) -> Optional[Image.Image]:
    path = _disk_path(z, x, y)
    if path is None or not path.is_file():
        return None
    try:
        with Image.open(path) as im:
            return im.convert("RGB")
    except Exception:  # noqa: BLE001 - a broken file in the cache is fetched again
        return None


def _to_disk(z: int, x: int, y: int, img: Image.Image) -> None:
    path = _disk_path(z, x, y)
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.{threading.get_ident()}.tmp")
        img.save(tmp, format="PNG")
        os.replace(tmp, path)
    except OSError:
        pass            # a cache that cannot be written is no reason to stop planning


def _tile(z: int, x: int, y: int) -> Optional[Image.Image]:
    key = (z, x, y)
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    img = _from_disk(z, x, y)
    if img is None:
        img = _tile_from_network(z, x, y)
        if img is not None:
            _to_disk(z, x, y, img)
    if img is not None and _empty(img):
        img = None
    with _lock:
        _cache[key] = img
        while len(_cache) > 400:
            _cache.popitem(last=False)
    return img


def _world_pixel(lat: float, lon: float, z: int) -> tuple[float, float]:
    n = 2 ** z * 256
    x = (lon + 180.0) / 360.0 * n
    rad = math.radians(max(-85.0, min(85.0, lat)))
    y = (1.0 - math.log(math.tan(rad) + 1 / math.cos(rad)) / math.pi) / 2.0 * n
    return x, y


def _height_at_zoom(lat: float, lon: float, z: int) -> Optional[float]:
    px, py = _world_pixel(lat, lon, z)
    img = _tile(z, int(px // 256) % (2 ** z), int(py // 256))
    if img is None:
        return None
    r, g, b = img.getpixel((int(px) % 256, int(py) % 256))[:3]
    q = (r * 256 + g + b / 256.0) - 32768.0
    return q if q > MIN_VALID_HEIGHT else None


def height(lat: float, lon: float, z: int = ZOOM) -> Optional[float]:
    """The height above sea level at that point, or None if the data is missing.

    If it is missing at that zoom (empty tile or pixel without data) the other zooms are tried: the data is
    the same, only the cut of the tiles changes, and the holes are not all in the same place.
    """
    for zz in (z, *(x for x in FALLBACK_ZOOMS if x != z)):
        q = _height_at_zoom(lat, lon, zz)
        if q is not None:
            return q
    return None


def _inside(p: LatLon, polygon: list[LatLon]) -> bool:
    x, y, inside = p[1], p[0], False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        if (a[0] > y) != (b[0] > y) and x < (b[1] - a[1]) * (y - a[0]) / ((b[0] - a[0]) or 1e-12) + a[1]:
            inside = not inside
    return inside


def _metres_per_degree_lon(lat: float) -> float:
    return 111_320.0 * math.cos(math.radians(lat))


def _grid(polygon: list[LatLon], step_m: float, z: int) -> dict[tuple[int, int], tuple[float, float, float]]:
    """The heights inside the polygon, on the nodes of a grid of constant step, with their row and column: they
    tell who is next to whom, and so how steep the terrain is."""
    if len(polygon) < 3:
        raise TerrainError(t("servono almeno tre vertici", "at least three vertices are needed"))
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    lat0 = sum(lats) / len(lats)
    d_lat = step_m / 111_320.0
    d_lon = step_m / max(_metres_per_degree_lon(lat0), 1.0)
    nodes: dict[tuple[int, int], tuple[float, float, float]] = {}
    i = 0
    lat = min(lats)
    while lat <= max(lats) + 1e-12:
        j = 0
        lon = min(lons)
        while lon <= max(lons) + 1e-12:
            if _inside((lat, lon), polygon):
                q = height(lat, lon, z)
                if q is not None:
                    nodes[(i, j)] = (lat, lon, q)
            lon += d_lon
            j += 1
        lat += d_lat
        i += 1
    if not nodes:
        # very small areas: at least the centre
        centre = (lat0, sum(lons) / len(lons))
        q = height(centre[0], centre[1], z)
        if q is None:
            raise TerrainError(t("quote non disponibili: serve la rete",
                                 "heights not available: the network is needed"))
        nodes[(0, 0)] = (centre[0], centre[1], q)
    return nodes


def sample(polygon: list[LatLon], step_m: float = 20.0, z: int = ZOOM) -> list[tuple[float, float, float]]:
    """The heights inside the polygon, on a regular grid: (lat, lon, height)."""
    return list(_grid(polygon, step_m, z).values())


def mean_slope(nodes: dict[tuple[int, int], tuple[float, float, float]], step_m: float) -> float:
    """How steep the terrain is, in degrees: the true slope between nodes that touch.

    For every node the drop towards the neighbour to the east and the one to the south is taken and the slope
    is composed in the two directions; the mean is that of the area. Comparing sorted heights would only say
    how scattered they are, which is another thing.
    """
    step = max(step_m, 0.1)
    full, partial = [], []
    for (i, j), (_, _, q) in nodes.items():
        south, east = nodes.get((i + 1, j)), nodes.get((i, j + 1))
        if south is None and east is None:
            continue
        d_south = (south[2] - q) if south else 0.0
        d_east = (east[2] - q) if east else 0.0
        (full if (south and east) else partial).append(math.hypot(d_south, d_east) / step)
    used = full or partial          # at the edges, and in narrow areas, with what there is
    if not used:
        return 0.0
    return math.degrees(math.atan(sum(used) / len(used)))


def statistics(polygon: list[LatLon], step_m: float = 20.0, z: int = ZOOM) -> dict:
    """What the terrain under the area is like: heights, drop, slope and area in hectares (the planner's keys)."""
    nodes = _grid(polygon, step_m, z)
    heights = sorted(q for _, _, q in nodes.values())
    n = len(heights)
    lowest, highest = heights[0], heights[-1]
    mean = sum(heights) / n
    median = heights[n // 2]
    slope = mean_slope(nodes, step_m)
    lats = [p[0] for p in polygon]
    lons = [p[1] for p in polygon]
    side_ns = (max(lats) - min(lats)) * 111_320.0
    side_ew = (max(lons) - min(lons)) * _metres_per_degree_lon(sum(lats) / len(lats))
    return {
        "campioni": n,
        "minima": round(lowest, 1),
        "massima": round(highest, 1),
        "media": round(mean, 1),
        "mediana": round(median, 1),
        "dislivello": round(highest - lowest, 1),
        "pendenzaGradi": round(slope, 1),
        "riquadroM": [round(side_ns), round(side_ew)],
        "ettari": round(side_ns * side_ew / 10_000.0, 2),
        "passoM": step_m,
        "fonte": ATTRIBUTION,
    }


def point_heights(points: list[LatLon], z: int = ZOOM) -> list[Optional[float]]:
    """The height under every waypoint: it serves the flight that follows the terrain."""
    return [height(lat, lon, z) for lat, lon in points]
