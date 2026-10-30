"""The layers of a run over the raster view of the program, as the terrain plugin shows its products:
a diverging palette around zero for erosion and deposition (and the differences with another DEM),
symmetric robust limits (the larger of the 2nd and 98th percentiles in absolute value); a
hypsometric ramp for the surfaces (2nd to 98th percentile); a blue ramp on the logarithm of the
drainage area (the channels stand out of the slopes)."""
from __future__ import annotations

import numpy as np

PALETTES = {
    "diverging": [(0.0, (33, 102, 172)), (0.25, (103, 169, 207)), (0.5, (247, 247, 247)), (0.75, (239, 138, 98)),
                  (1.0, (178, 24, 43))],
    "terrain": [(0.0, (0, 104, 55)), (0.2, (102, 189, 99)), (0.4, (217, 239, 139)), (0.6, (254, 224, 139)),
                (0.8, (215, 140, 70)), (1.0, (250, 250, 250))],
    "water": [(0.0, (247, 251, 255)), (0.35, (158, 202, 225)), (0.7, (33, 113, 181)), (1.0, (8, 29, 88))],
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
}
DEFAULT_PALETTE = {"diverging": "diverging", "elevation": "terrain", "log": "water"}


def prepared(values: np.ndarray, kind: str) -> np.ndarray:
    """The values the colours follow: log10 of the drainage area (cells without area: none)."""
    if kind == "log":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(values > 0, np.log10(np.where(values > 0, values, 1.0)), np.nan)
    return values


def limits(values: np.ndarray, kind: str = "elevation") -> tuple[float, float]:
    """The ends of the ramp: symmetric around zero for ``diverging``, else the 2nd and 98th percentiles
    (of the logarithm for ``log``: give the :func:`prepared` values)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (-1.0, 1.0) if kind == "diverging" else (0.0, 1.0)
    lo, hi = (float(x) for x in np.percentile(v, [2, 98]))
    if kind == "diverging":
        m = max(abs(lo), abs(hi), 1e-6)
        return -m, m
    return (lo, hi) if hi > lo else (lo, lo + 1e-6)


def render(values: np.ndarray, lo: float, hi: float, palette: str = "terrain") -> np.ndarray:
    """RGBA ``(h, w, 4)`` of a raster: the palette between ``lo`` and ``hi``, transparent where NaN."""
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if not ok.any():
        return rgba
    stops = PALETTES.get(palette, PALETTES["grey"])
    f = np.clip((np.where(ok, values, lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    pos = np.array([s[0] for s in stops])
    for k in range(3):
        rgba[..., k] = np.interp(f, pos, [s[1][k] for s in stops]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def resample_to(values: np.ndarray, transform, dst_transform, shape, nearest: bool = False) -> np.ndarray:
    """The raster on another grid of the same frame (the view's), bilinear or nearest, NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the same frame on both sides: any projected CRS does
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=values.astype(np.float32), destination=dst, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return dst


class Overlay:
    """A layer over the raster view (``window.ortho_view``), a pixmap item with an opacity; one at a time."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.key = None  # (run id, layer)

    def show(self, values: np.ndarray, transform, kind: str = "elevation", palette: str | None = None,
             opacity: float = 0.7, key=None) -> tuple[float, float]:
        """Show ``values``; the limits of the ramp (of log10 for ``log``)."""
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        if view.tr is None or getattr(view, "_base_size", None) is None:
            raise RuntimeError("nothing in the raster view to overlay on: show the DEM first")
        self.remove()
        w, h = view._base_size
        shown = resample_to(prepared(values, kind), transform, view.tr, (h, w), nearest=kind == "log")
        lo, hi = limits(shown, kind)
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(render(shown, lo, hi,
                                                                         palette or DEFAULT_PALETTE.get(kind, "grey")))))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings (0 and above)
        item.setOpacity(float(opacity))
        self.item, self.key = item, key
        if hasattr(view, "overlays"):
            view.overlays.append(item)
        return lo, hi

    def set_opacity(self, opacity: float) -> None:
        if self.item is not None:
            try:
                self.item.setOpacity(float(opacity))
            except RuntimeError:  # the scene was cleared
                self.item = None

    def remove(self) -> None:
        if self.item is not None:
            try:
                if hasattr(self.view, "overlays") and self.item in self.view.overlays:
                    self.view.overlays.remove(self.item)
                self.view.scene().removeItem(self.item)
            except RuntimeError:
                pass
        self.item, self.key = None, None
