"""A synthetic Sentinel-2 world for the tests: fields in UTM zone 32N (round, made-up
coordinates), six acquisitions of a season as small GeoTIFFs named like the bands of the COGs,
the STAC items that point to them, a fake catalog that answers in pages, and a project whose
chunk (in EPSG:3035, so everything is reprojected) has a DEM over the site.

The fields have an NDVI that rises in spring and falls in autumn with a gentle slope from west
to east; a buried wall crosses the site as a strip 20 m wide whose NDVI is 0.08 lower in the
summer dates only (a negative crop mark). 20 May is covered by a cloud, 9 July has a cloud and
its shadow over part of the site, 14 June comes in two tiles that overlap, 10 April from an old
processing baseline (no offset, no raster:bands), 3 August harmonised as Earth Search does it
(numbers without the offset, raster:bands still listing it, ``earthsearch:boa_offset_applied``),
12 September harmonised without saying so (the reader must notice), 1 October is 95 % cloudy and
the fake server forgets to filter it out."""
from __future__ import annotations

import io
import json
import urllib.error
from pathlib import Path

import numpy as np

import pyarchinit_photogrammetry  # noqa: F401 - first: it points rasterio at its own PROJ data

UTM = "EPSG:32632"
CHUNK_CRS = "EPSG:3035"
TILE_X0, TILE_Y1, TILE_SIDE = 499980.0, 5005020.0, 3000.0  # corners on multiples of 60 m, like the tiles
SITE = (501000.0, 5003500.0, 501400.0, 5003800.0)  # x0, y0, x1, y1 (UTM)
WALL = ((500900.0, 5003450.0), (501500.0, 5003850.0))  # the buried wall, end to end (UTM)
WALL_HALF = 10.0
CATALOG = "https://stac.example.org/v1"
DATES = {  # date: (field NDVI, wall effect, kind)
    "2023-04-10": (0.45, 0.0, "old"),
    "2023-05-20": (0.60, 0.0, "cloudy"),
    "2023-06-14": (0.72, -0.08, "two tiles"),
    "2023-07-09": (0.65, -0.08, "cloud spot"),
    "2023-08-03": (0.55, -0.08, "harmonised"),
    "2023-09-12": (0.40, 0.0, "mislabelled"),
}
CLOUD_SPOT = (501080.0, 5003740.0, 60.0)  # x, y, radius of the cloud of 9 July
SHADOW_SPOT = (501250.0, 5003620.0, 40.0)


def wall_distance(x, y):
    (ax, ay), (bx, by) = WALL
    dx, dy = bx - ax, by - ay
    return np.abs(dy * (x - ax) - dx * (y - ay)) / np.hypot(dx, dy)


def centres(cell: float):
    n = int(TILE_SIDE / cell)
    xs = TILE_X0 + (np.arange(n) + 0.5) * cell
    ys = TILE_Y1 - (np.arange(n) + 0.5) * cell
    return np.meshgrid(xs, ys)


def ndvi_truth(day: str, x, y, noise: bool = True):
    field, wall, _ = DATES[day]
    v = field + 0.05 * (x - 501200.0) / 1000.0
    v = v + np.where(wall_distance(x, y) <= WALL_HALF, wall, 0.0)
    if noise:
        rng = np.random.default_rng(int(day.replace("-", "")))
        v = v + rng.normal(0.0, 0.015, np.shape(x))
    return v


def _write(path: Path, values: np.ndarray, cell: float, dtype: str):
    import rasterio
    from rasterio.transform import from_origin

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=values.shape[1], height=values.shape[0], count=1, dtype=dtype,
                       crs=UTM, transform=from_origin(TILE_X0, TILE_Y1, cell, cell), nodata=0, tiled=True,
                       blockxsize=64, blockysize=64, compress="deflate") as dst:
        dst.write(values.astype(dtype), 1)


def _disc(x, y, spot):
    cx, cy, r = spot
    return np.hypot(x - cx, y - cy) <= r


def scene_files(folder: Path, day: str, tile: str = "32TMR", keep=None) -> dict:
    """The eight band files of one acquisition (``keep(x, y)``: where this tile has data)."""
    field, _wall, kind = DATES[day]
    # numbers = reflectance x 10000 (+ 1000 from baseline 04.00, unless harmonised)
    offset = 0.0 if kind in ("old", "harmonised", "mislabelled") else 0.1
    x10, y10 = centres(10.0)
    x20, y20 = centres(20.0)
    ndvi10 = ndvi_truth(day, x10, y10)
    nir10 = np.full(x10.shape, 0.32)
    red10 = nir10 * (1 - ndvi10) / (1 + ndvi10)
    ndvi20 = ndvi10.reshape(150, 2, 150, 2).mean(axis=(1, 3))
    nir20 = np.full(x20.shape, 0.32)
    re20 = nir20 * (1 - 0.6 * ndvi20) / (1 + 0.6 * ndvi20)
    sw20 = nir20 * (1 - (0.5 * ndvi20 - 0.05)) / (1 + (0.5 * ndvi20 - 0.05))
    refl = {"B02": (np.full(x10.shape, 0.04), 10.0), "B03": (np.full(x10.shape, 0.07), 10.0), "B04": (red10, 10.0),
            "B08": (nir10, 10.0), "B05": (re20, 20.0), "B8A": (nir20, 20.0), "B11": (sw20, 20.0)}
    scl = np.full(x20.shape, 4, np.uint8)
    if kind == "cloudy":
        scl[:] = 9
        scl[:, :20] = 4
    elif kind == "cloud spot":
        scl[_disc(x20, y20, CLOUD_SPOT)] = 9
        scl[_disc(x20, y20, SHADOW_SPOT)] = 3
    files = {}
    stem = folder / f"{day}_{tile}"
    for band, (values, cell) in refl.items():
        dn = np.clip(np.rint((values + offset) * 10000), 1, 65535)
        if keep is not None:
            xx, yy = (x10, y10) if cell == 10.0 else (x20, y20)
            dn = np.where(keep(xx, yy), dn, 0)
        _write(Path(f"{stem}_{band}.tif"), dn, cell, "uint16")
        files[band] = Path(f"{stem}_{band}.tif")
    if keep is not None:
        scl = np.where(keep(x20, y20), scl, 0)
    _write(Path(f"{stem}_SCL.tif"), scl, 20.0, "uint8")
    files["SCL"] = Path(f"{stem}_SCL.tif")
    return files


KEYS = {"B02": "blue", "B03": "green", "B04": "red", "B05": "rededge1", "B08": "nir", "B8A": "nir08",
        "B11": "swir16", "SCL": "scl"}


def item(day: str, files: dict, tile: str = "32TMR", platform: str = "S2B", cloud: float = 5.0, old: bool = False,
         seq: int = 0, harmonised: bool = False) -> dict:
    baseline = "03.00" if old else "05.09"
    stamp = day.replace("-", "")
    assets = {}
    for band, path in files.items():
        a = {"href": str(path), "type": "image/tiff; application=geotiff; profile=cloud-optimized"}
        if not old and band != "SCL":
            a["raster:bands"] = [{"nodata": 0, "data_type": "uint16", "scale": 0.0001, "offset": -0.1}]
        assets[KEYS[band]] = a
        assets[KEYS[band] + "-jp2"] = {"href": str(path) + ".jp2", "type": "image/jp2"}
    it = {"type": "Feature", "stac_version": "1.0.0", "id": f"{platform}_{tile}_{stamp}_{seq}_L2A",
          "bbox": [8.99, 45.17, 9.04, 45.20],
          "geometry": {"type": "Polygon", "coordinates": [[[8.99, 45.17], [9.04, 45.17], [9.04, 45.20],
                                                          [8.99, 45.20], [8.99, 45.17]]]},
          "properties": {"datetime": f"{day}T10:20:31.024000Z", "platform": f"sentinel-2{platform[-1].lower()}",
                         "eo:cloud_cover": cloud, "proj:epsg": 32632, "grid:code": f"MGRS-{tile}",
                         "s2:processing_baseline": baseline, "s2:sequence": str(seq),
                         "s2:datatake_id": f"G{platform}_{stamp}T102031_000000_N{baseline}",
                         "s2:product_uri": f"{platform}_MSIL2A_{stamp}T102031_N{baseline.replace('.', '')}_R065_T{tile}_x.SAFE"},
          "assets": assets, "links": []}
    if harmonised:  # as Earth Search v1: the offset taken away from the numbers, still listed in raster:bands
        it["properties"]["earthsearch:boa_offset_applied"] = True
    return it


def world(folder: Path) -> list[dict]:
    """The items of the synthetic season (their files written in ``folder``)."""
    items = []
    for day, (_f, _w, kind) in DATES.items():
        if kind == "two tiles":
            west = scene_files(folder, day, "32TMR", keep=lambda x, y: x < 501200.0)
            east = scene_files(folder, day, "32TNR", keep=lambda x, y: x >= 501150.0)
            items += [item(day, west, "32TMR", cloud=3.0), item(day, east, "32TNR", cloud=4.0)]
        else:
            files = scene_files(folder, day)
            cloud = {"cloudy": 28.0, "cloud spot": 12.0}.get(kind, 2.0)
            items.append(item(day, files, cloud=cloud, old=kind == "old", harmonised=kind == "harmonised"))
    # the same tile of 3 August processed once more: the later copy must win (identical files here)
    items.append(item("2023-08-03", scene_files(folder, "2023-08-03"), seq=1, harmonised=True))
    # a 95 % cloudy scene the server forgets to filter out
    stormy = item("2023-10-01", {b: Path(str(p)) for b, p in scene_files(folder, "2023-09-12").items()}, cloud=95.0)
    stormy["properties"]["datetime"] = "2023-10-01T10:20:31.024000Z"
    stormy["id"] = "S2B_32TMR_20231001_0_L2A"
    items.append(stormy)
    return items


class Response(io.BytesIO):
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


class FakeCatalog:
    """A STAC API in a function, for ``urlopen``: POST /search filtered by date (and by cloud
    cover, except the scenes it forgets), answered ``page`` items at a time with ``next`` links
    whose body carries a token; ``fail`` first answers HTTP 503 that many times; ``refuse_query``
    answers 400 to a body with the query extension (then CQL2 is expected)."""

    def __init__(self, items, page: int = 3, fail: int = 0, refuse_query: bool = False, forget=("S2B_32TMR_20231001_0_L2A",)):
        self.items, self.page, self.fail, self.refuse_query, self.forget = items, page, fail, refuse_query, set(forget)
        self.requests = []

    def __call__(self, request, timeout):
        body = json.loads(request.data.decode()) if request.data else {}
        self.requests.append((request.full_url, request.get_method(), body))
        if self.fail > 0:
            self.fail -= 1
            raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b"busy"))
        if self.refuse_query and "query" in body:
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {},
                                         io.BytesIO(json.dumps({"description": "query not supported"}).encode()))
        start, end = body["datetime"].split("/")
        cloud = None
        if "query" in body:
            cloud = body["query"]["eo:cloud_cover"]["lte"]
        elif "filter" in body:
            cloud = body["filter"]["args"][1]
        found = []
        for it in self.items:
            when = it["properties"]["datetime"]
            if not (start <= when <= end):
                continue
            if cloud is not None and it["properties"]["eo:cloud_cover"] > cloud and it["id"] not in self.forget:
                continue
            found.append(it)
        first = int(body.get("next") or 0)
        limit = min(int(body.get("limit", 10)), self.page)
        page = found[first:first + limit]
        links = [{"rel": "root", "href": CATALOG}]
        if first + limit < len(found):
            links.append({"rel": "next", "href": request.full_url, "method": "POST", "merge": False,
                          "body": {**{k: v for k, v in body.items() if k != "next"}, "next": str(first + limit)}})
        return Response(json.dumps({"type": "FeatureCollection", "features": page, "links": links,
                                    "numberMatched": len(found), "numberReturned": len(page)}).encode())


def project(tmp_path: Path, crs: str = CHUNK_CRS):
    """A saved project whose chunk has a DEM over the site (1 m cells) in ``crs``."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    ch = doc.add_chunk("Campo")
    doc.chunk = ch
    doc.save(tmp_path / "campo.ppg")
    ch.crs = crs
    b = transform_bounds(UTM, crs, *SITE)
    x0, y1 = float(np.floor(b[0])), float(np.ceil(b[3]))
    w, h = int(np.ceil(b[2] - x0)), int(np.ceil(y1 - b[1]))
    z = np.full((h, w), 100.0, np.float32)
    path = ch.data_dir / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=w, height=h, count=1, dtype="float32", crs=crs,
                       transform=from_origin(x0, y1, 1.0, 1.0), nodata=float("nan")) as dst:
        dst.write(z, 1)
    ch._set_product("dem", path)
    doc.save()
    return doc, ch
