"""A raster of a face (the bulge, the difference from another survey: centimetres, signed) over the elevation shown in
the program's raster view, with a diverging palette symmetric around zero and a legend in the corner of the view. The
overlay is not one of the view's drawings: it stays while the program draws its layers again and goes with the raster
when another one is shown (the plugin shows it again when the face comes back)."""
from __future__ import annotations

import numpy as np

from .texts import num, t

DIVERGING = [(0.0, (33, 102, 172)), (0.25, (103, 169, 207)), (0.5, (247, 247, 247)), (0.75, (239, 138, 98)),
             (1.0, (178, 24, 43))]


def limit(values: np.ndarray) -> float:
    """The end of the symmetric ramp (cm): the 99.5th percentile of the size of the values, rounded up to a millimetre,
    at least half a centimetre."""
    v = np.abs(values[np.isfinite(values)])
    if not v.size:
        return 1.0
    return max(0.5, float(np.ceil(np.percentile(v, 99.5) * 10.0) / 10.0))


def render(values: np.ndarray, lim: float) -> np.ndarray:
    """RGBA (h, w, 4): blue (towards the wall) - white (on the fitted surface) - red (towards the viewer)."""
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    x = np.clip((np.where(ok, values, 0.0) / max(lim, 1e-9) + 1.0) / 2.0, 0, 1)
    pos = np.array([s[0] for s in DIVERGING])
    for k in range(3):
        rgba[..., k] = np.interp(x, pos, [s[1][k] for s in DIVERGING]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def legend_image(lim: float, width: int = 220, height: int = 14) -> np.ndarray:
    x = np.linspace(-lim, lim, width)[None, :].repeat(height, axis=0)
    return render(x, lim)


def resample_to(path, transform, shape) -> np.ndarray:
    """The raster ``path`` on the grid of the view (``transform``, (rows, cols)), same frame, bilinear, NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the frame of the elevation on both sides: any projected CRS does
    with rasterio.open(path) as src:
        values = src.read(1).astype(np.float32)
        src_tr = src.transform
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=values, destination=dst, src_transform=src_tr, src_crs=crs, dst_transform=transform,
              dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.bilinear)
    return dst


class Overlay:
    """One raster of a face over the view at a time, with its legend."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.legend = None
        self.key = None
        self.limit = None

    def show(self, path, key: str, title: str, opacity: float = 0.45) -> float:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        rect = view.sceneRect()
        w, h = int(round(rect.width())), int(round(rect.height()))
        if view.tr is None or w < 1 or h < 1:
            raise RuntimeError(t("Niente nella vista raster: mostra prima il prospetto della faccia.",
                                 "Nothing in the raster view: show the face's elevation first."))
        self.remove()
        values = resample_to(path, view.tr, (h, w))
        if not np.isfinite(values).any():
            raise RuntimeError(t("La carta sta fuori dal raster mostrato: mostra il prospetto della faccia.",
                                 "The map lies outside the raster shown: show the face's elevation."))
        with np.errstate(invalid="ignore"):
            lim = limit(values)
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(render(values, lim))))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings
        item.setOpacity(float(opacity))
        self.item, self.key, self.limit = item, key, lim
        self._legend(title, lim)
        return lim

    def _legend(self, title: str, lim: float) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        box = QWidget(self.view)
        box.setObjectName("masonry_legend")
        box.setStyleSheet("#masonry_legend { background: rgba(255, 255, 255, 220); border: 1px solid #888; }"
                          "QLabel { color: black; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        head = QLabel(title)
        lay.addWidget(head)
        bar = QLabel()
        bar.setPixmap(QPixmap.fromImage(to_qimage(legend_image(lim))))
        lay.addWidget(bar)
        ends = QLabel(f"-{num(lim, 1)} cm  ·  0  ·  +{num(lim, 1)} cm")
        ends.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ends)
        note = QLabel(t("rosso: verso chi guarda; blu: verso il muro", "red: towards the viewer; blue: into the wall"))
        lay.addWidget(note)
        box.adjustSize()
        box.move(8, 8)
        box.show()
        self.legend = box

    def set_opacity(self, opacity: float) -> None:
        if self.item is not None:
            try:
                self.item.setOpacity(float(opacity))
            except RuntimeError:
                self.item = None

    def remove(self) -> None:
        if self.item is not None:
            try:
                if self.item.scene() is self.view.scene():
                    self.view.scene().removeItem(self.item)
            except RuntimeError:  # the scene was cleared: the item is gone already
                pass
        if self.legend is not None:
            try:
                self.legend.deleteLater()
            except RuntimeError:
                pass
        self.item = self.legend = self.key = self.limit = None

    def shown(self) -> bool:
        try:
            return self.item is not None and self.item.scene() is self.view.scene()
        except RuntimeError:
            return False
