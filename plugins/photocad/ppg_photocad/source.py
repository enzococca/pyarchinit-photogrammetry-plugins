"""The images sent to PhotoCAD, rendered from a raster of the project, and where they lie.

Sources: ``ortho`` (the orthophoto of the chunk), ``dem`` (its DEM as a hillshade), ``relief:<path>`` (a
greyscale raster of the plan, such as a relief map the window shows), ``elevation:<id>`` (the orthophoto of an
elevation or a section, a GeoTIFF in its ``(u, v)`` metres, v upwards). All are north-up rasters.

:func:`tiles` cuts the raster (or the part in a box) into tiles at its NATIVE resolution - at most 2400 px on the
long side (what PhotoCAD reads), about 10 % of overlap - one conversion each. A raster whose short side is under
600 px is enlarged by a whole factor up to 4 (a DEM of coarse cells); thinner than that it is not sent: PhotoCAD
fails on thin strips (the whole tower of Khutm sent as 2274 x 200 px). :func:`render` makes the image of one tile:
every pixel of it is the cell divided by the zoom, and its lower-left corner is a known point.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .texts import t

MAX_SIDE = 2400
MAX_BYTES = 4_000_000
WHITE = 255


class NoRaster(RuntimeError):
    """The chunk has no raster of that kind."""


@dataclass
class Rendered:
    """The image and its place: ``x0, y0`` = lower-left corner in the coordinates of the chunk (plan) or the
    ``(u, v)`` of the elevation ``frame``; ``pixel`` metres per pixel of the image."""

    data: bytes
    fmt: str  # jpeg | png
    width: int
    height: int
    x0: float
    y0: float
    pixel: float
    kind: str  # ortho | dem | relief | elevation
    source: str
    frame: str | None = None
    crs: str | None = None
    path: str = ""
    factor: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def width_m(self) -> float:
        return self.width * self.pixel

    @property
    def height_m(self) -> float:
        return self.height * self.pixel

    @property
    def longest_m(self) -> float:
        return max(self.width, self.height) * self.pixel

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0, self.x0 + self.width_m, self.y0 + self.height_m

    @property
    def content_type(self) -> str:
        return "image/jpeg" if self.fmt == "jpeg" else "image/png"

    @property
    def filename(self) -> str:
        return "image.jpg" if self.fmt == "jpeg" else "image.png"

    def describe(self) -> dict:
        """What the record and ``conversion.json`` keep of it (no pixels)."""
        return {"width": self.width, "height": self.height, "pixel": self.pixel, "x0": self.x0, "y0": self.y0,
                "zoom": int(self.extra.get("zoom", 1)), "window": self.extra.get("window"), "format": self.fmt,
                "bytes": len(self.data), "kind": self.kind, "raster": self.path}


def elevation_entry(chunk, eid: str) -> dict:
    entry = next((e for e in chunk.meta.get("elevations") or [] if e.get("id") == eid), None)
    if entry is None:
        names = ", ".join(f"{e.get('id')} ({e.get('name')})" for e in chunk.meta.get("elevations") or []) or "-"
        raise NoRaster(t("Nessun prospetto «{0}» nel chunk (ci sono: {1}).", "No elevation '{0}' in the chunk "
                         "(there are: {1}).").format(eid, names))
    return entry


def raster_of(chunk, source: str) -> tuple[Path, str, str | None, str | None]:
    """``(path, kind, crs, frame)`` of a source."""
    if source == "ortho":
        path = chunk.orthomosaic_path
        if path is None or not Path(path).exists():
            raise NoRaster(t("Il chunk non ha un'ortofoto.", "The chunk has no orthophoto."))
        return Path(path), "ortho", chunk.crs, None
    if source == "dem":
        path = chunk.dem_path
        if path is None or not Path(path).exists():
            raise NoRaster(t("Il chunk non ha un DEM.", "The chunk has no DEM."))
        return Path(path), "dem", chunk.crs, None
    if source.startswith("relief:"):
        path = Path(source.split(":", 1)[1])
        if not path.exists():
            raise NoRaster(str(path))
        return path, "relief", chunk.crs, None
    if source.startswith("elevation:"):
        from pyarchinit_photogrammetry.engine.elevation import elevation_paths

        entry = elevation_entry(chunk, source.split(":", 1)[1])
        ortho, _relief = elevation_paths(chunk, entry)
        if not ortho.exists():
            raise NoRaster(t("Il prospetto «{0}» non ha più la sua ortofoto.", "The elevation '{0}' has lost its "
                             "orthophoto.").format(entry.get("name")))
        return ortho, "elevation", None, entry["id"]
    raise ValueError(t("Sorgente sconosciuta: {0} (ortho, dem, elevation:<id>)",
                       "Unknown source: {0} (ortho, dem, elevation:<id>)").format(source))


MIN_SHORT = 600
MAX_ZOOM = 4
OVERLAP = 0.10
MAX_TILES = 6


def _block(src, bbox):
    """``(c0, r0, ncols, nrows)``: the whole cells of the raster in ``bbox`` (all of it when None)."""
    tr = src.transform
    if abs(tr.b) > 1e-12 or abs(tr.d) > 1e-12 or tr.e >= 0:
        raise ValueError(t("Il raster non è orientato a nord.", "The raster is not north-up."))
    if abs(abs(tr.a) - abs(tr.e)) > 1e-9 * max(1.0, abs(tr.a)):
        raise ValueError(t("Le celle del raster non sono quadrate.", "The cells of the raster are not square."))
    if bbox is None:
        return 0, 0, src.width, src.height
    x0, y0, x1, y1 = (float(v) for v in bbox)
    x0, x1 = min(x0, x1), max(x0, x1)
    y0, y1 = min(y0, y1), max(y0, y1)
    c0 = max(0, math.floor((x0 - tr.c) / tr.a))
    c1 = min(src.width, math.ceil((x1 - tr.c) / tr.a))
    r0 = max(0, math.floor((y1 - tr.f) / tr.e))
    r1 = min(src.height, math.ceil((y0 - tr.f) / tr.e))
    if c1 - c0 < 2 or r1 - r0 < 2:
        raise ValueError(t("L'area scelta è fuori dal raster.", "The area chosen is outside the raster."))
    return c0, r0, c1 - c0, r1 - r0


def _starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    n = math.ceil((length - overlap) / (tile - overlap))
    return [int(round(v)) for v in np.linspace(0, length - tile, n)]


def tiles(chunk, source: str, bbox=None, max_side: int = MAX_SIDE, min_short: int = MIN_SHORT,
          max_zoom: int = MAX_ZOOM, overlap: float = OVERLAP) -> list[dict]:
    """The tiles of ``source`` (in ``bbox``): ``{"window": (c0, r0, ncols, nrows) in cells, "zoom": u}``; each image
    is ``ncols * u`` x ``nrows * u`` px, at most ``max_side`` on the long side, at least ``min_short`` on the short
    one. ``ValueError`` outside the raster, or for a strip too thin even enlarged ``max_zoom`` times."""
    import rasterio

    path, _kind, _crs, _frame = raster_of(chunk, source)
    with rasterio.open(path) as src:
        c0, r0, nc, nr = _block(src, bbox)
        res = abs(src.transform.a)
    short = min(nc, nr)
    zoom = max(1, math.ceil(min_short / short))
    if zoom > max_zoom:
        raise ValueError(t("Troppo sottile per PhotoCAD: {0} x {1} celle ({2:.2f} x {3:.2f} m); servono almeno {4} px "
                           "sul lato corto ({5} celle ingrandite al massimo {6} volte). Scegli un'area più alta "
                           "o una risoluzione più fine.",
                           "Too thin for PhotoCAD: {0} x {1} cells ({2:.2f} x {3:.2f} m); at least {4} px are needed "
                           "on the short side ({5} cells enlarged at most {6} times). Choose a taller area or a finer "
                           "resolution.").format(nc, nr, nc * res, nr * res, min_short,
                                                 math.ceil(min_short / max_zoom), max_zoom))
    tile = max(1, max_side // zoom)
    ov = int(round(overlap * tile))
    out = []
    for rs in _starts(nr, tile, ov):
        for cs in _starts(nc, tile, ov):
            out.append({"window": (c0 + cs, r0 + rs, min(tile, nc), min(tile, nr)), "zoom": zoom})
    return out


def _read(src, bands, window, fill, nodata=None) -> np.ndarray:
    """The cells of ``window`` as float, nodata as NaN (float rasters) or ``fill``."""
    from rasterio.windows import Window

    c0, r0, nc, nr = window
    a = src.read(bands, window=Window(c0, r0, nc, nr)).astype(np.float64)
    if nodata is not None and np.isfinite(nodata):
        a[a == nodata] = fill
    return a


def _mean(a, axis):
    """Mean without the NaN cells (NaN where all are)."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def _hillshade(z: np.ndarray, cell: float, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Hillshade 0-255 (NaN = white), central-difference gradients, light from the north-west."""
    valid = np.isfinite(z)
    filled = np.where(valid, z, np.nanmean(z) if valid.any() else 0.0)
    gy, gx = np.gradient(filled, cell, cell)  # gy along the rows (southwards)
    slope = np.arctan(np.hypot(gx, gy))
    aspect = np.arctan2(gy, -gx)
    az, alt = np.radians(360.0 - azimuth + 90.0), np.radians(altitude)
    shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    out = np.clip(shade, 0, 1) * 255.0
    out[~valid] = WHITE
    return out


def _stretch(a: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a)
    if not valid.any():
        return np.full(a.shape, WHITE, np.float64)
    lo, hi = np.percentile(a[valid], [2, 98])
    out = np.clip((a - lo) / max(hi - lo, 1e-12), 0, 1) * 255.0
    out[~valid] = WHITE
    return out


def _encode(rgb: np.ndarray, fmt: str, max_bytes: int) -> tuple[bytes, str]:
    """JPEG (quality lowered until it fits) or PNG (JPEG when the PNG does not fit)."""
    from PIL import Image

    img = Image.fromarray(np.ascontiguousarray(rgb.transpose(1, 2, 0).astype(np.uint8)))
    if fmt == "png":
        buf = io.BytesIO()
        img.save(buf, "PNG", optimize=True)
        if buf.tell() <= max_bytes:
            return buf.getvalue(), "png"
    for quality in (92, 85, 78, 70, 60, 50, 40, 30, 20):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality)
        if buf.tell() <= max_bytes:
            return buf.getvalue(), "jpeg"
    raise ValueError(t("Immagine troppo grande anche alla qualità minima.", "The image is too large even at the "
                       "lowest quality."))


def render(chunk, source: str, window, zoom: int = 1, max_bytes: int = MAX_BYTES, fmt: str = "jpeg") -> Rendered:
    """The image of the cells ``window`` (``(c0, r0, ncols, nrows)``) of ``source``, enlarged ``zoom`` times."""
    import cv2
    import rasterio

    path, kind, crs, frame = raster_of(chunk, source)
    c0, r0, nc, nr = (int(v) for v in window)
    with rasterio.open(path) as src:
        tr = src.transform
        res = abs(tr.a)
        if kind in ("dem", "relief"):
            a = _read(src, [1], window, np.nan, src.nodata)[0]
            grey = _hillshade(a, res) if kind == "dem" else _stretch(a)
            rgb = np.repeat(grey[None], 3, axis=0)
        else:
            bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            if src.count >= 4:  # transparent cells white, the edges blended on white
                a = _read(src, bands + [4], window, 0)
                alpha = np.clip(a[3] / 255.0, 0, 1)
                rgb = a[:3] * alpha + WHITE * (1 - alpha)
            else:
                from rasterio.windows import Window

                rgb = _read(src, bands, window, WHITE)
                mask = src.dataset_mask(window=Window(c0, r0, nc, nr)).astype(np.float64) / 255.0
                rgb = rgb * mask + WHITE * (1 - mask)
        x0 = tr.c + c0 * tr.a
        y0 = tr.f + (r0 + nr) * tr.e  # the lower edge of the block
    rgb = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    if zoom > 1:
        img = cv2.resize(np.ascontiguousarray(rgb.transpose(1, 2, 0)), (nc * zoom, nr * zoom),
                         interpolation=cv2.INTER_LINEAR)
        rgb = img.transpose(2, 0, 1)
    data, fmt = _encode(rgb, fmt, max_bytes)
    h, w = rgb.shape[1:]
    out = Rendered(data, fmt, int(w), int(h), float(x0), float(y0), float(res / zoom), kind, source, frame, crs,
                   str(path), 1)
    out.extra = {"zoom": int(zoom), "window": [c0, r0, nc, nr]}
    return out
