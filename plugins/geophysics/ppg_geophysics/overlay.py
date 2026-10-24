"""The grids shown: colours for the values (grey for magnetometry, a diverging ramp for the
anomalies around zero, a colour ramp for resistivity), the limits of the ramp from the readings,
and the overlay of a grid on the orthomosaic or DEM of the raster view of the program, with its
opacity, in the frame of what the view shows - or, when the chunk has neither, on an empty frame
the size of the grid (:func:`frame_of`, shown by the program: ``context.show_frame``)."""
from __future__ import annotations

import numpy as np

from .grids import Grid
from .texts import t

PALETTES = {
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "grey_inverted": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
    "diverging": [(0.0, (33, 102, 172)), (0.5, (247, 247, 247)), (1.0, (178, 24, 43))],
    "earth": [(0.0, (0, 60, 48)), (0.25, (1, 133, 113)), (0.5, (245, 245, 245)), (0.75, (191, 129, 45)),
              (1.0, (84, 48, 5))],
    "heat": [(0.0, (0, 0, 0)), (0.3, (128, 0, 80)), (0.6, (230, 90, 0)), (0.85, (255, 210, 60)), (1.0, (255, 255, 255))],
}
DEFAULT_PALETTE = {"magnetometry": "grey", "resistivity": "earth", "gpr": "grey", "em": "earth", "susceptibility": "heat",
                   "other": "grey"}


def limits(values: np.ndarray, kind: str = "other", k: float = 2.5) -> tuple[float, float]:
    """The ends of the colour ramp: for magnetometry and EM (readings around zero) the median
    plus and minus ``k`` times the robust deviation, symmetric; otherwise the 2nd and 98th
    percentiles."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return -1.0, 1.0
    if kind in ("magnetometry", "em", "gpr"):
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med))) * 1.4826
        if mad <= 0:
            mad = float(v.std()) or 1.0
        return med - k * mad, med + k * mad
    lo, hi = np.percentile(v, [2, 98])
    return float(lo), float(hi) if hi > lo else float(lo) + 1e-6


def render(values: np.ndarray, lo: float, hi: float, palette: str = "grey") -> np.ndarray:
    """RGBA (h, w, 4) of a band: the palette between ``lo`` and ``hi``, transparent where NaN."""
    stops = PALETTES.get(palette, PALETTES["grey"])
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if not ok.any():
        return rgba
    t = np.clip((np.where(ok, values, lo) - lo) / max(hi - lo, 1e-12), 0, 1)
    pos = np.array([s[0] for s in stops])
    for k in range(3):
        rgba[..., k] = np.interp(t, pos, [s[1][k] for s in stops]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def resample_to(grid: Grid, band: int, transform, shape: tuple[int, int]) -> np.ndarray:
    """The band of the grid on another grid (``transform`` and (rows, cols)) of the same frame,
    bilinear, NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the same frame on both sides: any projected CRS does
    src = grid.values[band].astype(np.float32)
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=src, destination=dst, src_transform=grid.transform(), src_crs=crs, dst_transform=transform,
              dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return dst


def frame_of(grid: Grid, max_side: int = 2000) -> tuple[tuple[float, float, float, float], float]:
    """The empty frame of the raster view for a grid shown alone (the chunk has neither orthomosaic
    nor DEM): ``(bounds, resolution)`` for ``context.show_frame`` - the extent of ``grid``, north up,
    with a margin of 5 %, and the metres per pixel that give about ``max_side`` pixels on its longer
    side. The grid shows on it, the drawing of the program works on it, and the next orthomosaic or
    DEM shown replaces it."""
    x0, y0, x1, y1 = grid.bounds()
    m = 0.05 * max(x1 - x0, y1 - y0, grid.cell)
    x0, y0, x1, y1 = x0 - m, y0 - m, x1 + m, y1 + m
    return (x0, y0, x1, y1), max(x1 - x0, y1 - y0) / float(max_side)


class Overlay:
    """A grid over the raster view of the program (``window.ortho_view``), as a pixmap item in
    the scene with an opacity; one at a time. The item is not one of the view's drawings
    (``view.overlays``): it stays when the program draws its layers again (a feature deleted, an
    undo), and goes with the raster when another one is shown (the plugin shows it again on the
    new one). The pixmap has the size of the scene of the view, which is the raster shown."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.grid_id = None

    def show(self, grid: Grid, band: int = 0, palette: str | None = None, opacity: float = 0.6,
             lims: tuple[float, float] | None = None) -> tuple[float, float]:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        rect = view.sceneRect()
        w, h = int(round(rect.width())), int(round(rect.height()))
        if view.tr is None or w < 1 or h < 1:
            raise RuntimeError(t("Niente nella vista raster su cui sovrapporre la griglia: mostra l'ortofoto o il DEM.",
                                 "Nothing in the raster view to overlay the grid on: show the orthomosaic or the DEM."))
        self.remove()
        values = resample_to(grid, band, view.tr, (h, w))
        if not np.isfinite(values).any():
            raise RuntimeError(t("La griglia sta fuori dal raster mostrato nella vista: controlla la sua posizione "
                                 "(Georeferenzia...) o mostra un altro raster.",
                                 "The grid lies outside the raster shown in the view: check its position "
                                 "(Georeference...) or show another raster."))
        lo, hi = lims or limits(values, grid.kind)
        rgba = render(values, lo, hi, palette or DEFAULT_PALETTE.get(grid.kind, "grey"))
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(rgba)))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings (0 and above)
        item.setOpacity(float(opacity))
        self.item, self.grid_id = item, grid.id
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
                if self.item.scene() is self.view.scene():
                    self.view.scene().removeItem(self.item)
            except RuntimeError:  # the scene was cleared (another raster): the item is gone already
                pass
        self.item, self.grid_id = None, None

    def shown(self) -> bool:
        """Whether the grid is still in the view (another raster shown clears it)."""
        try:
            return self.item is not None and self.item.scene() is self.view.scene()
        except RuntimeError:
            return False
