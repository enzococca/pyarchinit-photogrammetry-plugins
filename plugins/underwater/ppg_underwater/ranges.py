"""The range of every pixel of a photo: how far, along its ray, the surface it shows is.

Under water the colour of a pixel depends on that distance: the light it sends is absorbed along
the way, and the suspended particles add their own light (backscatter) growing with it. The program
knows the distance from its depth maps (*Flusso di lavoro > Mappe di profondità*, also built by the
dense cloud): COLMAP depth maps of the *undistorted* photos, written in
``<chunk data>/dense/stereo/depth_maps/<image name>.geometric.bin``, one per aligned photo, with
the pinhole camera of the undistorted photo in ``dense/sparse``. A depth map holds ``z`` (along the
optical axis) in the units of the internal frame, its pixel ``(col, row)`` seeing the ray
``K^-1 (col, row, 1)``.

The photos are corrected in their own geometry (distorted, full size): for a grid over the photo,
every pixel is taken to its normalised ray with the camera model of the photo (``Sensor.unproject``,
COLMAP's convention: the centre of the first pixel at 0.5), that ray to the depth map with its
pinhole ``K``, and ``range = z * sqrt(1 + x^2 + y^2)``; metres when the chunk is in metres (a
georeferenced, scaled or levelled chunk: ``chunk.scale``), else units of the alignment. Pixels the
depth map does not reach (holes, the water above the horizon, the margins the undistortion cut) take
the range of the nearest pixel that has one.

Depth maps made elsewhere - in metres, one image per photo with the same name, as the Sea-thru
dataset gives them (made with Metashape) - are read by :class:`FileRanges`.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np


def is_metric(chunk) -> bool:
    """The internal units of ``chunk`` are taken to metres by its transform."""
    from pyarchinit_photogrammetry.core import crs as crs_mod

    return (not crs_mod.is_local(chunk.crs)) or abs(chunk.scale - 1.0) > 1e-9 or bool(chunk.meta.get("metric"))


def fill_nearest(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``values`` with the invalid cells taken from the nearest valid one, smoothed away from the valid
    ones (the nearest value alone draws the edges of a Voronoi diagram in a wide hole: the water above
    the horizon would get seams of colour)."""
    if valid.all() or not valid.any():
        return values
    from scipy.ndimage import distance_transform_edt

    dist, (ri, ci) = distance_transform_edt(~valid, return_indices=True)
    near = values[ri, ci].astype(np.float32)
    sigma = 0.03 * max(values.shape)
    smooth = cv2.GaussianBlur(near, (0, 0), sigma)
    w = np.clip(dist / (2.0 * sigma), 0.0, 1.0).astype(np.float32)
    return np.where(valid, values, (1.0 - w) * near + w * smooth).astype(values.dtype)


class RangeMaps:
    """Ranges of the photos of ``chunk`` from the program's depth maps."""

    def __init__(self, chunk):
        from pyarchinit_photogrammetry.engine.confidence import depth_maps

        self.chunk = chunk
        self.entries = {e["name"]: e for e in depth_maps(chunk)}
        self.names = {int(k): v for k, v in ((chunk.meta.get("sfm") or {}).get("names") or {}).items()}
        self.metric = is_metric(chunk)
        self.factor = float(chunk.scale) if self.metric else 1.0

    def entry(self, cam):
        name = self.names.get(cam.key)
        return self.entries.get(name) if name is not None else None

    def available(self, cam) -> bool:
        return self.entry(cam) is not None

    def __call__(self, cam, shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
        """``(ranges, valid)``: a float32 ``(h, w)`` grid over the photo (the whole photo resampled
        to ``shape``), holes filled from the nearest pixel that has a range, and where the depth map
        reached. ``(None, None)`` without a depth map."""
        from pyarchinit_photogrammetry.engine.dense import read_colmap_array

        e = self.entry(cam)
        if e is None:
            return None, None
        depth = read_colmap_array(Path(e["path"]))
        if depth.ndim != 2:
            depth = depth[..., 0]
        depth = depth.astype(np.float32)
        hd, wd = depth.shape
        fx, fy, cx, cy = e["K"]
        sx, sy = wd / e["size"][0], hd / e["size"][1]
        fx, cx, fy, cy = fx * sx, cx * sx, fy * sy, cy * sy
        sensor = self.chunk.sensor(cam)
        h, w = shape
        # COLMAP pixel coordinates of the centres of the grid cells, in the photo's own pixels
        us = (np.arange(w, dtype=np.float64) + 0.5) * sensor.width / w
        vs = (np.arange(h, dtype=np.float64) + 0.5) * sensor.height / h
        uu, vv = np.meshgrid(us, vs)
        norm = sensor.unproject(np.column_stack([uu.ravel(), vv.ravel()]))
        x = norm[:, 0].reshape(h, w)
        y = norm[:, 1].reshape(h, w)
        col = (fx * x + cx).astype(np.float32)
        row = (fy * y + cy).astype(np.float32)
        z = cv2.remap(depth, col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = np.isfinite(z) & (z > 0)
        rng = (z * np.sqrt(1.0 + x * x + y * y)).astype(np.float32) * self.factor
        if not valid.any():
            return None, None
        return fill_nearest(np.where(valid, rng, 0).astype(np.float32), valid), valid


class FileRanges:
    """Ranges from depth images made elsewhere: in ``folder``, one per photo with the same stem
    (``depth<stem>.tif`` or ``<stem>.tif``/``.exr``/``.png``...), values in metres; ``kind`` "depth"
    (along the optical axis, as Metashape exports them: turned into ranges with ``fov`` degrees of
    horizontal field of view) or "range"."""

    def __init__(self, folder, kind: str = "depth", fov: float | None = None, scale: float = 1.0):
        self.folder = Path(folder)
        self.kind = kind
        self.fov = fov
        self.factor = float(scale)
        self.metric = True
        self.index = {}
        for p in self.folder.rglob("*"):
            if p.suffix.lower() in (".tif", ".tiff", ".exr", ".png", ".pfm", ".npy"):
                self.index[p.stem.lower()] = p

    def path_for(self, cam_or_path) -> Path | None:
        path = Path(getattr(cam_or_path, "path", cam_or_path))
        stem = path.stem.lower()
        for cand in (stem, "depth" + stem, "depth_" + stem, stem + "_depth", "deptht_" + stem, "deptht" + stem):
            if cand in self.index:
                return self.index[cand]
        return None

    def available(self, cam) -> bool:
        return self.path_for(cam) is not None

    @staticmethod
    def read(path: Path) -> np.ndarray:
        if path.suffix.lower() == ".npy":
            return np.load(path).astype(np.float32)
        arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED | cv2.IMREAD_ANYDEPTH)
        if arr is None:
            from PIL import Image

            with Image.open(path) as im:
                arr = np.asarray(im)
        arr = np.asarray(arr, np.float32)
        return arr[..., 0] if arr.ndim == 3 else arr

    def __call__(self, cam, shape: tuple[int, int]) -> tuple[np.ndarray | None, np.ndarray | None]:
        path = self.path_for(cam)
        if path is None:
            return None, None
        z = self.read(path) * self.factor
        h, w = shape
        valid = np.isfinite(z) & (z > 0)
        zr = cv2.resize(np.where(valid, z, 0).astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        vr = cv2.resize(valid.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        if self.kind == "depth":
            fov = math.radians(self.fov or 60.0)
            f = 0.5 * w / math.tan(fov / 2.0)
            xs = (np.arange(w) + 0.5 - w / 2.0) / f
            ys = (np.arange(h) + 0.5 - h / 2.0) / f
            xx, yy = np.meshgrid(xs, ys)
            zr = zr * np.sqrt(1.0 + xx * xx + yy * yy).astype(np.float32)
        if not vr.any():
            return None, None
        return fill_nearest(zr, vr), vr
