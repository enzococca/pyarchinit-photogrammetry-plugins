"""The surfaces shown over the raster view of the program: a band of a surface (its elevation,
the kriging variance or the correction of the stack) or a thickness, with a palette between the
2nd and 98th percentiles of its values - from zero for variances and thicknesses, around zero for
the correction, whose cells lowered and raised take the two ends of a diverging ramp -, as a
pixmap in the frame of what the view shows, with an opacity; the boreholes as a layer of points
of the view, the traces of the sections as a layer of lines (the plugin draws both again on every
raster the view shows: ``context.on_raster_loaded``)."""
from __future__ import annotations

import numpy as np

PALETTES = {
    "terrain": [(0.0, (40, 90, 60)), (0.3, (150, 180, 100)), (0.55, (230, 215, 150)), (0.8, (170, 110, 60)),
                (1.0, (245, 245, 245))],
    "earth": [(0.0, (0, 60, 48)), (0.25, (1, 133, 113)), (0.5, (245, 245, 245)), (0.75, (191, 129, 45)),
              (1.0, (84, 48, 5))],
    "thickness": [(0.0, (255, 250, 220)), (0.35, (240, 190, 90)), (0.7, (180, 90, 30)), (1.0, (90, 30, 10))],
    "grey": [(0.0, (0, 0, 0)), (1.0, (255, 255, 255))],
    "grey_inverted": [(0.0, (255, 255, 255)), (1.0, (0, 0, 0))],
    "heat": [(0.0, (0, 0, 0)), (0.3, (128, 0, 80)), (0.6, (230, 90, 0)), (0.85, (255, 210, 60)), (1.0, (255, 255, 255))],
    "diverging": [(0.0, (33, 102, 172)), (0.5, (247, 247, 247)), (1.0, (178, 24, 43))],
}
# the palette of what is shown: band 1 of a surface, its variance (2), its correction (3), a thickness
DEFAULT = {("surface", 1): "terrain", ("surface", 2): "heat", ("surface", 3): "diverging", ("thickness", 1): "thickness"}
BOREHOLES_LAYER = "Geoarcheologia: sondaggi"
SECTIONS_LAYER = "Geoarcheologia: sezioni"


def style(kind: str, band: int) -> tuple[str, bool, bool]:
    """``(palette, from zero, around zero)`` of a product shown: the elevation of a surface on the
    hypsometric ramp, its kriging variance from zero, the correction of the stack around zero on a
    diverging ramp (lowered one way, raised the other), a thickness from zero."""
    palette = DEFAULT.get((kind, band), "terrain")
    return palette, kind == "thickness" or (kind == "surface" and band == 2), kind == "surface" and band == 3


def limits(values: np.ndarray, zero: bool = False, symmetric: bool = False) -> tuple[float, float]:
    """The ends of the ramp: the 2nd and 98th percentiles (from 0 when ``zero``: thicknesses,
    variances; from -m to m, the larger of the two in size, when ``symmetric``: the corrections,
    so that zero is the middle of a diverging ramp)."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    lo, hi = (float(q) for q in np.percentile(v, [2, 98]))
    if symmetric:
        m = max(abs(lo), abs(hi)) or float(np.abs(v).max()) or 1e-6
        return -m, m
    if zero:
        lo = 0.0
        if hi <= 0:
            hi = float(v.max())
    return lo, hi if hi > lo else lo + 1e-6


def render(values: np.ndarray, lo: float, hi: float, palette: str = "terrain") -> np.ndarray:
    """RGBA (h, w, 4) of a raster: the palette between ``lo`` and ``hi``, transparent where NaN."""
    stops = PALETTES.get(palette, PALETTES["terrain"])
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


def resample_to(values: np.ndarray, transform, dst_transform, shape: tuple[int, int]) -> np.ndarray:
    """The raster on another grid of the same frame, bilinear, NaN outside."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    crs = rasterio.crs.CRS.from_epsg(3857)  # the same frame on both sides: any projected CRS does
    dst = np.full(shape, np.nan, np.float32)
    reproject(source=values.astype(np.float32), destination=dst, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs, src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    return dst


class Overlay:
    """A product over the raster view of the program (``window.ortho_view``), one at a time."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.product_id = None

    def show(self, values: np.ndarray, transform, palette: str = "terrain", opacity: float = 0.7,
             zero: bool = False, product_id: str | None = None, symmetric: bool = False) -> tuple[float, float]:
        from PySide6.QtGui import QPixmap

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        view = self.view
        rect = view.sceneRect()  # the raster shown, in its pixels
        w, h = int(round(rect.width())), int(round(rect.height()))
        if view.tr is None or w < 1 or h < 1:
            raise RuntimeError("nothing in the raster view to overlay on: show the orthomosaic or the DEM first")
        self.remove()
        values = resample_to(values, transform, view.tr, (h, w))
        lo, hi = limits(values, zero, symmetric)
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


def draw_boreholes(view, holes, colour: str = "#d62728") -> int:
    """The boreholes as points of the layer :data:`BOREHOLES_LAYER` of the view (attributes: id,
    z, depth, units); the layer drawn again from scratch. Returns the points drawn."""
    if view is None or view.tr is None:
        return 0
    if BOREHOLES_LAYER in getattr(view, "layers", {}):
        view.remove_layer(BOREHOLES_LAYER)
    batch = getattr(view, "batch", None)
    ctx = batch() if batch is not None else _Nothing()
    with ctx:
        for b in holes:
            view.add_point(b.x, b.y, colour, b.id, layer=BOREHOLES_LAYER,
                           attrs={"id": b.id, "z": None if b.z is None else round(b.z, 3), "depth": round(b.bottom, 3),
                                  "units": " / ".join(b.units), "kind": b.kind})
    return len(holes)


def draw_traces(view, sections, colour: str = "#ff7f0e") -> int:
    """The traces of ``sections`` (the entries of :func:`.section.entries`) as lines of the layer
    :data:`SECTIONS_LAYER` of the view (attributes: id, length, boreholes); the layer drawn again
    from scratch, or taken away when there is no section. Returns the lines drawn."""
    if view is None or view.tr is None:
        return 0
    if SECTIONS_LAYER in getattr(view, "layers", {}):
        view.remove_layer(SECTIONS_LAYER)
    if not sections:
        return 0
    batch = getattr(view, "batch", None)
    ctx = batch() if batch is not None else _Nothing()
    with ctx:
        for s in sections:
            view.add_line(np.asarray(s["coords"], float)[:, :2], colour, s["name"], layer=SECTIONS_LAYER,
                          attrs={"id": s["id"], "lunghezza_m": round(float(s["length"]), 2),
                                 "sondaggi": ", ".join(h["id"] for h in s.get("boreholes", []))}, width=2.0)
    return len(sections)


class _Nothing:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
