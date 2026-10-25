"""Windowed, aligned physical raster bands and model-owned channel normalisation."""
from __future__ import annotations

from contextlib import ExitStack
import math
from pathlib import Path

import numpy as np

KINDS = {"rgb", "dem", "multispectral", "thermal", "raster"}
RESAMPLING = {"nearest", "bilinear", "cubic", "average"}
MAX_CHANNELS = 64


def check_grid(src) -> float:
    """Reference pixels are square, north-up and expressed in metres."""
    if src.crs is not None:
        if src.crs.is_geographic:
            raise ValueError("Use a projected reference raster in metres, or a local metre grid")
        try:
            if not math.isclose(src.crs.linear_units_factor[1], 1.0):
                raise ValueError("The reference raster CRS must use metres")
        except AttributeError:
            pass
    tr = src.transform
    if (not all(math.isfinite(v) for v in tr) or tr.b or tr.d or tr.a <= 0 or tr.e >= 0
            or not math.isclose(tr.a, -tr.e, rel_tol=1e-5)):
        raise ValueError("Use a north-up reference raster with square pixels")
    return float(tr.a)


def validate_schema(schema) -> int:
    """Check semantic channel descriptions independently of source paths."""
    if not isinstance(schema, list) or not schema:
        raise ValueError("Choose at least one raster input")
    names, channels = set(), 0
    for item in schema:
        if not isinstance(item, dict) or set(item) != {"name", "kind", "bands", "resampling", "units"}:
            raise ValueError("Invalid raster input schema")
        name, bands, units = item.get("name"), item.get("bands"), item.get("units")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Raster input names must be non-empty and unique")
        if item.get("kind") not in KINDS or item.get("resampling") not in RESAMPLING:
            raise ValueError("Invalid raster input kind or resampling")
        if (not isinstance(bands, list) or not bands or any(type(b) is not int or b < 1 for b in bands)
                or len(set(bands)) != len(bands)):
            raise ValueError("Raster bands must be distinct positive integers")
        if not isinstance(units, list) or len(units) != len(bands) or any(not isinstance(u, str) for u in units):
            raise ValueError("Each selected band needs a unit string (empty when unknown)")
        names.add(name)
        channels += len(bands)
    if not 1 <= channels <= MAX_CHANNELS:
        raise ValueError(f"Choose between 1 and {MAX_CHANNELS} input channels")
    return channels


def channel_count(meta: dict) -> int:
    if meta.get("format") == 1:
        return 3
    if meta.get("format") == 2:
        count = validate_schema(meta.get("inputs"))
        if type(meta.get("channels")) is not int or meta["channels"] != count:
            raise ValueError("Model channel count does not match its raster schema")
        return count
    raise ValueError("Unsupported GeoAI input format")


def normalise(image: np.ndarray, metadata: dict) -> np.ndarray:
    """Apply training statistics; callers zero invalid pixels using the saved valid mask."""
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 3 or image.shape[0] != channel_count(metadata):
        raise ValueError("Raster channel count does not match the trained model")
    if metadata.get("format") == 1:
        out = image / 255.0
    else:
        norm = metadata["normalisation"]
        mean = np.asarray(norm["mean"], dtype=np.float32)[:, None, None]
        std = np.asarray(norm["std"], dtype=np.float32)[:, None, None]
        out = (image - mean) / std
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


class RasterStack:
    """Read selected bands on a reference grid without creating a full stacked raster.

    Undefined CRS on both sides means the caller's common local metre coordinates; a known CRS
    is never silently assigned to an unknown one. Band scales/offsets are applied once, after
    reprojection, and the intersection of all band masks defines valid input pixels.
    """

    def __init__(self, reference, sources, schema=None):
        self.reference = reference
        self.sources = sources
        self.expected_schema = schema
        self._opened = None

    def __enter__(self):
        import rasterio
        from rasterio.enums import Resampling
        from rasterio.vrt import WarpedVRT

        opened = self._opened = ExitStack()
        try:
            ref = opened.enter_context(rasterio.open(self.reference))
            self.resolution = check_grid(ref)
            self.profile, self.width, self.height = ref.profile.copy(), ref.width, ref.height
            self.transform, self.crs, self.shape = ref.transform, ref.crs, (ref.height, ref.width)
            if not isinstance(self.sources, list) or not self.sources:
                raise ValueError("Choose at least one raster input")
            self.schema, self._readers = [], []
            for source in self.sources:
                if (not isinstance(source, dict) or not isinstance(source.get("path"), (str, Path))
                        or not str(source["path"])):
                    raise ValueError("Every raster input needs a source path")
                src = opened.enter_context(rasterio.open(Path(source["path"])))
                bands = source.get("bands")
                if (not isinstance(bands, list) or not bands
                        or any(type(b) is not int or b < 1 or b > src.count for b in bands)):
                    raise ValueError(f"Invalid selected bands for {source.get('name', 'raster')}")
                if any(np.dtype(src.dtypes[b - 1]).kind not in "iuf" for b in bands):
                    raise ValueError("GeoAI inputs must contain real numeric bands, not complex values")
                if (src.crs is None) != (ref.crs is None):
                    raise ValueError("Reference and source rasters must both have a CRS, or share local coordinates")
                unit = source.get("unit")
                units = ([unit] * len(bands) if unit is not None else source.get("units"))
                if units is None:
                    units = [src.units[b - 1] or "" for b in bands]
                name = source.get("name", "")
                item = {"name": name.strip() if isinstance(name, str) else name, "kind": source.get("kind", "raster"),
                        "bands": list(bands), "resampling": source.get("resampling", "bilinear"), "units": units}
                # Validate before accessing the resampling enum or allocating data.
                validate_schema([item])
                if any(src.units[b - 1] and src.units[b - 1] != u for b, u in zip(bands, units)):
                    raise ValueError("Declared raster units disagree with the source band's units")
                scales = np.asarray([src.scales[b - 1] for b in bands], dtype=np.float32)
                offsets = np.asarray([src.offsets[b - 1] for b in bands], dtype=np.float32)
                if not np.isfinite(scales).all() or not np.isfinite(offsets).all():
                    raise ValueError("Raster band scales and offsets must be finite")
                crs = ref.crs or rasterio.crs.CRS.from_epsg(3857)
                vrt = opened.enter_context(WarpedVRT(src, src_crs=src.crs or crs, crs=crs,
                    transform=ref.transform, width=ref.width, height=ref.height, dtype="float32",
                    nodata=float("nan"), resampling=Resampling[item["resampling"]]))
                self.schema.append(item)
                self._readers.append((vrt, bands, scales[:, None, None], offsets[:, None, None]))
            self.channels = validate_schema(self.schema)
            if self.expected_schema is not None:
                validate_schema(self.expected_schema)
                if self.schema != self.expected_schema:
                    raise ValueError("Raster inputs do not match the model's ordered names, kinds, bands, units and resampling")
            return self
        except BaseException:
            opened.close()
            raise

    def __exit__(self, *args):
        return self._opened.__exit__(*args)

    def read(self, window):
        from rasterio.windows import Window

        values = (window.col_off, window.row_off, window.width, window.height)
        if any(not math.isfinite(v) or int(v) != v for v in values) or min(values[2:]) < 1:
            raise ValueError("RasterStack windows must have integer offsets and positive integer sizes")
        x, y, width, height = map(int, values)
        image = np.zeros((self.channels, height, width), dtype=np.float32)
        valid = np.zeros((height, width), dtype=bool)
        left, top, right, bottom = max(0, x), max(0, y), min(self.width, x + width), min(self.height, y + height)
        if left >= right or top >= bottom:
            return image, valid
        target = (slice(top - y, bottom - y), slice(left - x, right - x))
        valid[target] = True
        offset = 0
        for vrt, bands, scales, shifts in self._readers:
            data = vrt.read(bands, window=Window(left, top, right - left, bottom - top), masked=True)
            physical = np.asarray(data.data, dtype=np.float32) * scales + shifts
            known = (~np.ma.getmaskarray(data) & np.isfinite(physical)).all(axis=0)
            valid[target] &= known
            image[(slice(offset, offset + len(bands)), *target)] = np.where(np.isfinite(physical), physical, 0)
            offset += len(bands)
        image[:, ~valid] = 0
        return image, valid
