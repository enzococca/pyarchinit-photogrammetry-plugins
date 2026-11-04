"""The image sent to PhotoCAD, rendered from a raster of the project, and where it lies.

Sources: ``ortho`` (the orthophoto of the chunk), ``dem`` (its DEM as a hillshade), ``relief:<path>`` (a
greyscale raster of the plan, such as a relief map the window shows), ``elevation:<id>`` (the orthophoto of an
elevation or a section, a GeoTIFF in its ``(u, v)`` metres, v upwards). All are north-up rasters (``u`` to the
right, ``v`` up for an elevation): the image covers exactly a block of whole cells of the raster, reduced by an
integer factor ``k`` so that its long side is at most 2400 px (PhotoCAD reduces to that anyway): every pixel of
the image is ``k`` x the cell, square, and the lower-left corner of the image is a known point. That is what
brings the drawing back: PhotoCAD draws in the image's frame (origin at the lower-left corner, Y up), asked in
metres with the long side of the image in metres and no margin.
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
                "factor": self.factor, "format": self.fmt, "bytes": len(self.data), "kind": self.kind,
                "raster": self.path}


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


def _window(src, bbox, max_side: int):
    """``(col0, row0, ncols, nrows, k)`` of the block read: whole cells, a multiple of ``k`` in each direction."""
    tr = src.transform
    if abs(tr.b) > 1e-12 or abs(tr.d) > 1e-12 or tr.e >= 0:
        raise ValueError(t("Il raster non è orientato a nord.", "The raster is not north-up."))
    if bbox is None:
        c0, r0, c1, r1 = 0, 0, src.width, src.height
    else:
        x0, y0, x1, y1 = (float(v) for v in bbox)
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        c0 = max(0, math.floor((x0 - tr.c) / tr.a))
        c1 = min(src.width, math.ceil((x1 - tr.c) / tr.a))
        r0 = max(0, math.floor((y1 - tr.f) / tr.e))
        r1 = min(src.height, math.ceil((y0 - tr.f) / tr.e))
        if c1 - c0 < 2 or r1 - r0 < 2:
            raise ValueError(t("L'area scelta è fuori dal raster.", "The area chosen is outside the raster."))
    k = max(1, math.ceil(max(c1 - c0, r1 - r0) / max_side))
    ncols, nrows = math.ceil((c1 - c0) / k) * k, math.ceil((r1 - r0) / k) * k
    return c0, r0, ncols, nrows, k


def _read(src, bands, c0, r0, ncols, nrows, k, fill, nodata=None):
    """The block (possibly beyond the right and bottom edges: filled) averaged ``k`` x ``k``."""
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    w, h = ncols // k, nrows // k
    inside_c, inside_r = min(ncols, src.width - c0), min(nrows, src.height - r0)
    full_c, full_r = inside_c // k, inside_r // k  # whole k-blocks inside the raster
    out = np.full((len(bands), h, w), fill, np.float64)
    if full_c and full_r:
        out[:, :full_r, :full_c] = src.read(bands, window=Window(c0, r0, full_c * k, full_r * k),
                                            out_shape=(len(bands), full_r, full_c),
                                            resampling=Resampling.average).astype(np.float64)
    # the last partial blocks (at the right and bottom edges, narrower than k): averaged over the cells there
    def strip(cs, rs, nc, nr):
        a = src.read(bands, window=Window(c0 + cs, r0 + rs, nc, nr)).astype(np.float64)
        if nodata is not None and np.isfinite(nodata):
            a[a == nodata] = np.nan  # averaged over the valid cells only (rasterio does so for the whole blocks)
        return a

    if full_c < w and inside_c > full_c * k:  # the right strip, rows of whole blocks
        a = strip(full_c * k, 0, inside_c - full_c * k, full_r * k)
        if a.size:
            out[:, :full_r, full_c] = _mean(a.reshape(len(bands), full_r, k, -1), (2, 3))
    if full_r < h and inside_r > full_r * k:  # the bottom strip, columns of whole blocks
        a = strip(0, full_r * k, full_c * k, inside_r - full_r * k)
        if a.size:
            out[:, full_r, :full_c] = _mean(a.reshape(len(bands), -1, full_c, k), (1, 3))
    if full_c < w and full_r < h and inside_c > full_c * k and inside_r > full_r * k:  # the corner
        a = strip(full_c * k, full_r * k, inside_c - full_c * k, inside_r - full_r * k)
        out[:, full_r, full_c] = _mean(a.reshape(len(bands), -1), 1)
    return out


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


def render(chunk, source: str, bbox=None, max_side: int = MAX_SIDE, max_bytes: int = MAX_BYTES,
           fmt: str = "jpeg") -> Rendered:
    """The image of ``source`` (all of it, or the part in ``bbox`` = xmin, ymin, xmax, ymax in the coordinates of
    the plan or the (u, v) of the elevation) for PhotoCAD."""
    import rasterio

    path, kind, crs, frame = raster_of(chunk, source)
    with rasterio.open(path) as src:
        c0, r0, ncols, nrows, k = _window(src, bbox, max_side)
        tr = src.transform
        pixel = abs(tr.a) * k
        if abs(abs(tr.a) - abs(tr.e)) > 1e-9 * max(1.0, abs(tr.a)):
            raise ValueError(t("Le celle del raster non sono quadrate.", "The cells of the raster are not square."))
        if kind in ("dem", "relief"):
            a = _read(src, [1], c0, r0, ncols, nrows, k, np.nan, src.nodata)[0]
            if src.nodata is not None and np.isfinite(src.nodata):
                a[a == src.nodata] = np.nan
            grey = _hillshade(a, pixel) if kind == "dem" else _stretch(a)
            rgb = np.repeat(grey[None], 3, axis=0)
        else:
            bands = [1, 2, 3] if src.count >= 3 else [1, 1, 1]
            has_alpha = src.count >= 4
            a = _read(src, bands + ([4] if has_alpha else []), c0, r0, ncols, nrows, k, WHITE if not has_alpha else 0)
            rgb = a[:3]
            if has_alpha:  # transparent cells white, the edges blended on white
                alpha = np.clip(a[3] / 255.0, 0, 1)
                rgb = rgb * alpha + WHITE * (1 - alpha)
            else:
                mask = _read_mask(src, c0, r0, ncols, nrows, k)
                if mask is not None:
                    rgb = rgb * mask + WHITE * (1 - mask)
        rgb = np.clip(np.rint(rgb), 0, 255)
        data, fmt = _encode(rgb, fmt, max_bytes)
        x0 = tr.c + c0 * tr.a
        y0 = tr.f + (r0 + nrows) * tr.e  # the lower edge of the block
    h, w = rgb.shape[1:]
    return Rendered(data, fmt, int(w), int(h), float(x0), float(y0), float(pixel), kind, source, frame, crs, str(path),
                    int(k))


def _read_mask(src, c0, r0, ncols, nrows, k):
    """The dataset mask (nodata) as 0-1 coverage per pixel of the image, or None when all is valid."""
    from rasterio.enums import MaskFlags

    if all(MaskFlags.all_valid in flags for flags in src.mask_flag_enums):
        return None
    from rasterio.enums import Resampling
    from rasterio.windows import Window

    w, h = ncols // k, nrows // k
    m = src.dataset_mask(window=Window(c0, r0, ncols, nrows), out_shape=(h, w), resampling=Resampling.average,
                         boundless=True)
    return np.clip(m.astype(np.float64) / 255.0, 0, 1)
