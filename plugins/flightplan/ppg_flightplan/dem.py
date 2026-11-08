"""The project's own DEM: the heights measured by the survey, instead of the free ones at thirty metres.

The chunk's DEM (or any GeoTIFF of heights) is read and queried point by point by latitude and longitude,
so the next flight can follow the terrain as it really is, not as it is in the world model. It plays the
role of the planner's "DEM of the flights". The duck type the design asks of a DEM is
``height(lat, lon) -> float | None`` and ``to_json() -> dict`` (``file``, ``epsg``: the outcome of the terrain
and the plan's ``fonteQuote`` name it); ``heights(latlons)`` is used when present, for speed.

Adapted from the author's Photo3D planner: rasterio and pyproj read any coordinate system (Photo3D read only
degrees and WGS84 UTM, with its own projection). The cell is the one that contains the point (GeoTIFF
pixels are areas: the corner of the raster is the corner of its first cell).
"""
from __future__ import annotations

import math
import threading
import warnings
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .texts import t

LatLon = tuple[float, float]

# Below this height it is "no data": ODM writes -9999, Metashape -32767.
MIN_VALID_HEIGHT = -1000.0
# Points read together are grouped in windows of at most this side, so a scattered set never reads the
# whole raster at once.
BLOCK = 512


class DemError(RuntimeError):
    """The file cannot be used: the message is already in the language of the interface."""


class DemNotGeoreferenced(DemError):
    """The file does not say where it is in the world (no coordinate system or no transform)."""


class RasterDem:
    """A GeoTIFF of heights, read point by point. Outside its frame it answers "don't know" (None)."""

    def __init__(self, path):
        import rasterio
        import rasterio.errors
        from pyproj import CRS, Transformer

        self.path = Path(path)
        if not self.path.is_file():
            raise DemError(t("DEM non trovato: {0}", "DEM not found: {0}").format(self.path.name))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", rasterio.errors.NotGeoreferencedWarning)
                self._ds = rasterio.open(self.path)
        except Exception as e:  # noqa: BLE001 - a broken file must not stop the planner
            raise DemError(t("DEM non leggibile: {0}", "DEM not readable: {0}").format(e)) from e
        ds = self._ds
        if ds.crs is None or ds.transform.is_identity:
            ds.close()
            raise DemNotGeoreferenced(t("Il file non dice dove sta nel mondo (manca la georeferenziazione)",
                                        "The file does not say where it is in the world (no georeference)"))
        self.crs = CRS.from_user_input(ds.crs.to_wkt())
        self.epsg = self.crs.to_epsg()
        self.in_degrees = self.crs.is_geographic
        self.width, self.height_px = ds.width, ds.height
        self.transform = ds.transform
        self._inverse = ~ds.transform
        self.nodata = ds.nodata
        self._to_raster = Transformer.from_crs("EPSG:4326", self.crs, always_xy=True)
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ reading
    def close(self) -> None:
        self._ds.close()

    def __enter__(self) -> "RasterDem":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def _cells(self, latlons: list[LatLon]) -> tuple[np.ndarray, np.ndarray]:
        lat = np.asarray([p[0] for p in latlons], dtype=float)
        lon = np.asarray([p[1] for p in latlons], dtype=float)
        x, y = self._to_raster.transform(lon, lat)
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        i = self._inverse
        col, row = i.a * x + i.b * y + i.c, i.d * x + i.e * y + i.f
        col = np.where(np.isfinite(col), col, -1.0)
        row = np.where(np.isfinite(row), row, -1.0)
        return np.floor(row).astype(np.int64), np.floor(col).astype(np.int64)

    def _valid(self, v: np.ndarray) -> np.ndarray:
        ok = np.isfinite(v) & (v > MIN_VALID_HEIGHT)
        if self.nodata is not None and not (isinstance(self.nodata, float) and math.isnan(self.nodata)):
            ok &= v != self.nodata
        return ok

    def heights(self, latlons: Iterable[LatLon]) -> list[Optional[float]]:
        """The heights at many points at once (read by windows of the raster), None where unknown."""
        from rasterio.windows import Window

        latlons = [(float(p[0]), float(p[1])) for p in latlons]
        out: list[Optional[float]] = [None] * len(latlons)
        if not latlons:
            return out
        rows, cols = self._cells(latlons)
        inside = (rows >= 0) & (rows < self.height_px) & (cols >= 0) & (cols < self.width)
        idx = np.nonzero(inside)[0]
        if not len(idx):
            return out
        blocks: dict[tuple[int, int], list[int]] = {}
        for k in idx:
            blocks.setdefault((int(rows[k]) // BLOCK, int(cols[k]) // BLOCK), []).append(int(k))
        with self._lock:
            for members in blocks.values():
                r = rows[members]
                c = cols[members]
                r0, c0 = int(r.min()), int(c.min())
                win = Window(c0, r0, int(c.max()) - c0 + 1, int(r.max()) - r0 + 1)
                data = self._ds.read(1, window=win).astype(float)
                values = data[r - r0, c - c0]
                ok = self._valid(values)
                for k, v, good in zip(members, values, ok):
                    if good:
                        out[k] = float(v)
        return out

    def height(self, lat: float, lon: float) -> Optional[float]:
        """The height at that point, or None if it falls outside the raster or the data is missing there."""
        return self.heights([(lat, lon)])[0]

    def cell_m(self) -> float:
        """The side of a cell in metres (degrees taken at the planner's 111 320 m)."""
        side = abs(self.transform.a)
        return side * (111_320.0 if self.in_degrees else self.crs.axis_info[0].unit_conversion_factor)

    def to_json(self) -> dict:
        return {"file": str(self.path), "larghezza": self.width, "altezza": self.height_px,
                "magliaM": round(self.cell_m(), 3), "epsg": self.epsg, "inGradi": self.in_degrees}


def chunk_dem(chunk) -> Optional[RasterDem]:
    """The DEM of the chunk, when it has one that says where it is in the world; else None (a chunk in a local
    system has no latitude and longitude to plan with: the public tiles are used)."""
    path = getattr(chunk, "dem_path", None)
    if path is None or not Path(path).is_file():
        return None
    try:
        return RasterDem(path)
    except DemNotGeoreferenced:
        return None
