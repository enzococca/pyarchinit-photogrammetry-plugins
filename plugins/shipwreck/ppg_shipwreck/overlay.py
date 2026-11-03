"""A raster of the relief of the hull (its heights, hillshade or local relief model) in grey over the relief shown in
the program's raster view, as the masonry plugin shows a bulge over a face. It is not one of the view's drawings: it lies
over the raster (z -2 and its detail -1) and under the drawings (z 0) and goes with the raster when another one is
shown (the plugin shows it again when the relief comes back)."""
from __future__ import annotations

import numpy as np

from .texts import t

STRETCH = (2.0, 98.0)  # the percentiles at the two ends of the grey ramp


def grey(values: np.ndarray, stretch=STRETCH) -> np.ndarray:
    """RGBA (h, w, 4) of ``values``: black to white between the ``stretch`` percentiles of the finite values,
    transparent where there is no value."""
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if not ok.any():
        return rgba
    lo, hi = np.percentile(values[ok], stretch)
    span = float(hi - lo) or 1.0
    g = np.clip((np.where(ok, values, lo) - lo) / span, 0.0, 1.0)
    level = np.round(g * 255.0).astype(np.uint8)
    for k in range(3):
        rgba[..., k] = level
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def resample_to(path, transform, shape) -> np.ndarray:
    """The raster ``path`` on the grid of the view (``transform``, (rows, cols)), both in the relief's (u, v): bilinear,
    NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the (u, v) of the relief on both sides: any projected CRS does
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        if src.nodata is not None and np.isfinite(src.nodata):
            values[values == src.nodata] = np.nan
        src_tr = src.transform
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=values, destination=dst, src_transform=src_tr, src_crs=crs, dst_transform=transform,
              dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return dst


class Overlay:
    """One raster of the relief over the view at a time."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.key = None
        self.title = ""

    def show(self, path, key: str, title: str, opacity: float = 0.6) -> None:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        rect = view.sceneRect()
        w, h = int(round(rect.width())), int(round(rect.height()))
        if view.tr is None or w < 1 or h < 1:
            raise RuntimeError(t("Niente nella vista raster: mostra prima il rilievo dello scafo.",
                                 "Nothing in the raster view: show the relief of the hull first."))
        self.remove()
        values = resample_to(path, view.tr, (h, w))
        if not np.isfinite(values).any():
            raise RuntimeError(t("Il raster sta fuori da quello mostrato: mostra il rilievo dello scafo.",
                                 "The raster lies outside the one shown: show the relief of the hull."))
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(grey(values))))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings
        item.setOpacity(float(opacity))
        self.item, self.key, self.title = item, key, title

    def remove(self) -> None:
        if self.item is not None:
            try:
                if self.item.scene() is self.view.scene():
                    self.view.scene().removeItem(self.item)
            except RuntimeError:  # the scene was cleared: the item is gone already
                pass
        self.item, self.key, self.title = None, None, ""

    def shown(self) -> bool:
        try:
            return self.item is not None and self.item.scene() is self.view.scene()
        except RuntimeError:
            return False
