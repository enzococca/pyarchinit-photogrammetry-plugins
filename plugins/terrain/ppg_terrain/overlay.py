"""The products shown over the raster view of the program: colours for the values (greys,
diverging around zero for the metrics with a sign, categorical for the classes and ids), the
limits of the ramp from the values, the overlay of a product on the orthomosaic or DEM of the
view in the frame of what the view shows, with its opacity."""
from __future__ import annotations

import numpy as np

PALETTES = {
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "grey_inverted": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
    "diverging": [(0.0, (33, 102, 172)), (0.5, (247, 247, 247)), (1.0, (178, 24, 43))],
    "earth": [(0.0, (0, 60, 48)), (0.25, (1, 133, 113)), (0.5, (245, 245, 245)), (0.75, (191, 129, 45)),
              (1.0, (84, 48, 5))],
    "heat": [(0.0, (0, 0, 0)), (0.3, (128, 0, 80)), (0.6, (230, 90, 0)), (0.85, (255, 210, 60)), (1.0, (255, 255, 255))],
    "water": [(0.0, (240, 248, 255)), (0.5, (70, 130, 180)), (1.0, (8, 29, 88))],
}
CATEGORICAL = "classes"
FORMS = "forms"  # the ten landforms of the geomorphons, a fixed colour each (1 flat ... 10 pit)
FORM_COLOURS = {1: (200, 200, 200), 2: (160, 0, 0), 3: (240, 90, 0), 4: (250, 190, 0), 5: (220, 240, 60),
                6: (90, 180, 60), 7: (90, 220, 220), 8: (90, 150, 240), 9: (40, 70, 200), 10: (120, 30, 160)}
DEFAULT_PALETTE = {"diverging": "diverging", "grey": "grey", "log": "water", "classes": CATEGORICAL, "binary": "heat",
                   "forms": FORMS, "other": "grey"}


def limits(values: np.ndarray, kind: str = "other") -> tuple[float, float]:
    """The ends of the colour ramp: symmetric around zero for the diverging metrics (the larger
    of the 2nd and 98th percentiles in absolute value), 0-1 for a binary raster, the ids for the
    classes, the 2nd and 98th percentiles otherwise."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return 0.0, 1.0
    if kind == "binary":
        return 0.0, 1.0
    if kind in ("classes", "forms"):
        return float(v.min()), float(max(v.max(), v.min() + 1))
    lo, hi = np.percentile(v, [2, 98])
    if kind == "diverging":
        m = max(abs(float(lo)), abs(float(hi)), 1e-9)
        return -m, m
    return float(lo), float(hi) if hi > lo else float(lo) + 1e-6


def class_colours(ids: np.ndarray) -> np.ndarray:
    """A colour per id (golden-angle hues: neighbouring ids far apart), ``(n, 3)`` uint8."""
    import colorsys

    out = np.zeros((len(ids), 3), np.uint8)
    for i, k in enumerate(ids):
        h = (float(k) * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(h, 0.65, 0.95)
        out[i] = (int(r * 255), int(g * 255), int(b * 255))
    return out


def render(values: np.ndarray, lo: float, hi: float, palette: str = "grey") -> np.ndarray:
    """RGBA (h, w, 4) of a raster: the palette between ``lo`` and ``hi`` (``classes``: a colour
    per integer value), transparent where NaN."""
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if not ok.any():
        return rgba
    if palette in (CATEGORICAL, FORMS):
        ids = np.where(ok, np.rint(values), 0).astype(np.int64)
        uniq = np.unique(ids[ok])
        cols = class_colours(uniq)
        lut = dict(zip(uniq.tolist(), cols))
        if palette == FORMS:
            lut = {k: np.array(FORM_COLOURS.get(int(k), (0, 0, 0)), np.uint8) for k in uniq.tolist()}
        flat = ids.reshape(-1)
        out = np.zeros((flat.size, 3), np.uint8)
        for k, c in lut.items():
            out[flat == k] = c
        rgba[..., :3] = out.reshape(values.shape + (3,))
    else:
        stops = PALETTES.get(palette, PALETTES["grey"])
        t = np.clip((np.where(ok, values, lo) - lo) / max(hi - lo, 1e-12), 0, 1)
        pos = np.array([s[0] for s in stops])
        for k in range(3):
            rgba[..., k] = np.interp(t, pos, [s[1][k] for s in stops]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def resample_to(values: np.ndarray, transform, dst_transform, shape: tuple[int, int], nearest: bool = False) -> np.ndarray:
    """The raster on another grid of the same frame (``dst_transform``, (rows, cols)), bilinear
    (``nearest`` for ids and classes), NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the same frame on both sides: any projected CRS does
    src = values.astype(np.float32)
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=src, destination=dst, src_transform=transform, src_crs=crs, dst_transform=dst_transform,
              dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return dst


class Overlay:
    """A product over the raster view of the program (``window.ortho_view``), as a pixmap item
    in the scene with an opacity; one at a time."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.product_id = None

    def show(self, values: np.ndarray, transform, kind: str = "other", palette: str | None = None,
             opacity: float = 0.7, lims: tuple[float, float] | None = None, product_id: str | None = None) -> tuple[float, float]:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        if view.tr is None or getattr(view, "_base_size", None) is None:
            raise RuntimeError("nothing in the raster view to overlay on: show the orthomosaic or the DEM first")
        self.remove()
        w, h = view._base_size
        palette = palette or DEFAULT_PALETTE.get(kind, "grey")
        values = resample_to(values, transform, view.tr, (h, w),
                             nearest=palette in (CATEGORICAL, FORMS) or kind in ("classes", "forms", "binary"))
        lo, hi = lims or limits(values, kind)
        rgba = render(values, lo, hi, palette)
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(rgba)))
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings (0 and above)
        item.setOpacity(float(opacity))
        self.item, self.product_id = item, product_id
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
        self.item, self.product_id = None, None
