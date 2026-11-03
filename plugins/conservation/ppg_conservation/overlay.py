"""A raster of a comparison (the colour change of the orthophoto, the change of the relief) over the elevation shown
in the program's raster view: a diverging palette symmetric around zero, the opacity of the pane's slider and a legend
in the corner of the view - the masonry plugin's overlay, for the conservation plugin's two rasters. It is not one of
the view's drawings: it goes with the raster when another one is shown, and the plugin shows it again when the face
comes back."""
from __future__ import annotations

import numpy as np

from .texts import num, t

DIVERGING = [(0.0, (33, 102, 172)), (0.25, (103, 169, 207)), (0.5, (247, 247, 247)), (0.75, (239, 138, 98)),
             (1.0, (178, 24, 43))]
KINDS = {
    "colour": {"title": ("Cambiamento di colore (ΔE, Lab)", "Colour change (ΔE, Lab)"), "unit": "ΔE",
               "note": ("rosso: più chiaro in B; blu: più scuro in B", "red: lighter in B; blue: darker in B"),
               "least": 1.0},
    "relief": {"title": ("Cambiamento del rilievo (cm)", "Change of the relief (cm)"), "unit": "cm",
               "note": ("rosso: B verso chi guarda; blu: B verso il muro",
                        "red: B towards the viewer; blue: B into the wall"), "least": 0.5},
}


def limit(values: np.ndarray, least: float = 0.5) -> float:
    """The end of the symmetric ramp: the 99.5th percentile of the size of the values, rounded up to a tenth, at least
    ``least``."""
    v = np.abs(values[np.isfinite(values)])
    if not v.size:
        return float(least)
    return max(float(least), float(np.ceil(np.percentile(v, 99.5) * 10.0) / 10.0))


def render(values: np.ndarray, lim: float) -> np.ndarray:
    """RGBA (h, w, 4) of the diverging palette; transparent where there is no value."""
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
    """Band 1 of the raster ``path`` on the grid of the view (``transform``, (rows, cols)): the same frame, bilinear,
    NaN outside."""
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
    """One raster of a comparison over the view at a time, with its legend."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.legend = None
        self.key = None
        self.limit = None

    def show(self, path, key: str, kind: str, opacity: float = 0.6) -> float:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        rect = view.sceneRect()
        w, h = int(round(rect.width())), int(round(rect.height()))
        if view.tr is None or w < 1 or h < 1:
            raise RuntimeError(t("Niente nella vista raster: mostra prima il prospetto.",
                                 "Nothing in the raster view: show the elevation first."))
        self.remove()
        values = resample_to(path, view.tr, (h, w))
        if not np.isfinite(values).any():
            raise RuntimeError(t("Il confronto sta fuori dal raster mostrato: mostra il prospetto della campagna A.",
                                 "The comparison lies outside the raster shown: show the elevation of campaign A."))
        spec = KINDS[kind]
        with np.errstate(invalid="ignore"):
            lim = limit(values, spec["least"])
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(render(values, lim))))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings
        item.setOpacity(float(opacity))
        self.item, self.key, self.limit = item, key, lim
        self._legend(spec, lim)
        return lim

    def _legend(self, spec: dict, lim: float) -> None:
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QPixmap
        from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        box = QWidget(self.view)
        box.setObjectName("conservation_legend")
        box.setStyleSheet("#conservation_legend { background: rgba(255, 255, 255, 220); border: 1px solid #888; }"
                          "QLabel { color: black; }")
        lay = QVBoxLayout(box)
        lay.setContentsMargins(6, 4, 6, 4)
        lay.setSpacing(2)
        lay.addWidget(QLabel(t(*spec["title"])))
        bar = QLabel()
        bar.setPixmap(QPixmap.fromImage(to_qimage(legend_image(lim))))
        lay.addWidget(bar)
        unit = spec["unit"]
        ends = QLabel(f"-{num(lim, 1)} {unit}  ·  0  ·  +{num(lim, 1)} {unit}")
        ends.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(ends)
        lay.addWidget(QLabel(t(*spec["note"])))
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

    def opacity(self) -> float | None:
        try:
            return None if self.item is None else float(self.item.opacity())
        except RuntimeError:
            return None

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
