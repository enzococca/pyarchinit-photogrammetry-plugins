"""Colours of the products and their overlay on the 2D view of the program.

The palettes are those of the program's vegetation indices (red-yellow-green, greys, viridis,
turbo) and a diverging blue-white-red; an index is stretched between its 2nd and 98th
percentiles (within -1 and 1), a crop-mark map between -3 and 3 (its z-scores: the colours keep
their meaning from one map to the next), natural colour band by band. The overlay is computed on
the grid of the view (in the chunk's coordinate system) from the product on the grid of the
scenes (UTM), each 10 m pixel kept as a square: on a coarser copy of the view's grid, drawn
scaled, since a pixel of Sentinel-2 is hundreds of pixels of an orthomosaic.
"""
from __future__ import annotations

import numpy as np

PALETTES = ("rdylgn", "grey", "viridis", "turbo", "diverging")
PALETTE_NAMES = {"rdylgn": ("Rosso-giallo-verde", "Red-yellow-green"), "grey": ("Scala di grigi", "Greyscale"),
                 "viridis": ("Viridis", "Viridis"), "turbo": ("Turbo", "Turbo"),
                 "diverging": ("Divergente (blu-bianco-rosso)", "Diverging (blue-white-red)")}
RDYLGN = [(0.0, (165, 0, 38)), (0.25, (244, 109, 67)), (0.5, (254, 224, 139)), (0.75, (102, 189, 99)),
          (1.0, (0, 104, 55))]  # the ramp of the program's vegetation indices
DIVERGING = [(0.0, (33, 102, 172)), (0.5, (247, 247, 247)), (1.0, (178, 24, 43))]
Z_LIMIT = 3.0


def index_limits(values: np.ndarray) -> tuple[float, float]:
    """The 2nd and 98th percentiles, within -1 and 1."""
    v = values[np.isfinite(values)]
    if v.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(v, [2, 98])
    lo, hi = max(float(lo), -1.0), min(float(hi), 1.0)
    return (lo, hi) if hi > lo else (lo, lo + 1e-6)


def limits_for(values: np.ndarray, kind: str) -> tuple[float, float]:
    return (-Z_LIMIT, Z_LIMIT) if kind == "z" else index_limits(values)


def render(values: np.ndarray, lo: float, hi: float, palette: str = "rdylgn") -> np.ndarray:
    """RGBA (rows, cols, 4) of ``values`` on a palette between ``lo`` and ``hi``, transparent
    where there is no value."""
    ok = np.isfinite(values)
    rgba = np.zeros(values.shape + (4,), np.uint8)
    if not ok.any():
        return rgba
    t = np.clip((np.where(ok, values, lo) - lo) / max(hi - lo, 1e-12), 0.0, 1.0)
    if palette == "grey":
        rgba[..., :3] = (t * 255).astype(np.uint8)[..., None]
    elif palette in ("viridis", "turbo"):
        import cv2

        cmap = cv2.COLORMAP_VIRIDIS if palette == "viridis" else cv2.COLORMAP_TURBO
        rgba[..., :3] = cv2.applyColorMap(np.ascontiguousarray((t * 255).astype(np.uint8)), cmap)[..., ::-1]
    else:
        stops = DIVERGING if palette == "diverging" else RDYLGN
        pos = np.array([s[0] for s in stops])
        for k in range(3):
            rgba[..., k] = np.interp(t, pos, [s[1][k] for s in stops]).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def rgb_limits(rgb: np.ndarray) -> list[tuple[float, float]]:
    """The 2nd-98th percentile stretch of each of the three bands."""
    out = []
    for band in rgb:
        v = band[np.isfinite(band)]
        if v.size == 0:
            out.append((0.0, 1.0))
            continue
        lo, hi = np.percentile(v, [2, 98])
        out.append((float(lo), float(hi) if hi > lo else float(lo) + 1e-6))
    return out


def render_rgb(rgb: np.ndarray, limits=None) -> np.ndarray:
    """RGBA of three reflectance bands (3, rows, cols), each stretched, transparent where empty."""
    limits = limits or rgb_limits(rgb)
    ok = np.all(np.isfinite(rgb), axis=0)
    rgba = np.zeros(rgb.shape[1:] + (4,), np.uint8)
    for k, (lo, hi) in enumerate(limits):
        rgba[..., k] = (np.clip((np.where(ok, rgb[k], lo) - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba


def legend_text(kind: str, lo: float, hi: float, palette: str) -> str:
    from .texts import t

    if kind == "rgb":
        return t("Colori naturali (B04, B03, B02), contrasto 2-98 %", "Natural colour (B04, B03, B02), 2-98 % stretch")
    if kind == "z":
        return t("z da {0:g} a {1:g}: sotto -2 vegetazione più debole dello sfondo (muri, strade), sopra 2 più "
                 "vigorosa (fossati, fosse)",
                 "z from {0:g} to {1:g}: under -2 vegetation weaker than the background (walls, roads), over 2 "
                 "greener (ditches, pits)").format(lo, hi)
    return t("da {0:.3f} a {1:.3f}", "from {0:.3f} to {1:.3f}").format(lo, hi)


class ViewOverlay:
    """A product over the raster view of the program (``window.ortho_view``) with an opacity;
    one at a time. ``spec`` says what it shows, so it can be drawn again on another raster."""

    def __init__(self, view):
        self.view = view
        self.item = None
        self.spec = None
        self.scale = 1

    def _alive(self) -> bool:
        if self.item is None:
            return False
        try:
            import shiboken6

            return shiboken6.isValid(self.item) and self.item.scene() is not None
        except Exception:  # noqa: BLE001
            return False

    def show(self, values: np.ndarray, transform, crs, view_crs, kind: str = "index", palette: str = "rdylgn",
             opacity: float = 0.7, limits=None, spec=None) -> tuple:
        """``values`` (rows, cols; or (3, rows, cols) for natural colour) of ``transform`` and
        ``crs`` over the view, whose coordinates are in ``view_crs``; the limits of the colours."""
        from PySide6.QtGui import QPixmap
        from rasterio.transform import Affine, array_bounds
        from rasterio.warp import transform_bounds

        from pyarchinit_photogrammetry.gui.image_views import to_qimage

        from .project import onto

        view = self.view
        if view.tr is None or not getattr(view, "_base_size", (0, 0))[0]:
            raise RuntimeError("nothing in the raster view to overlay on: show the orthomosaic, the DEM or the area")
        self.remove()
        bw, bh = view._base_size
        rows, cols = values.shape[-2:]
        sb = array_bounds(rows, cols, transform)
        vb = transform_bounds(crs, view_crs, sb[0], sb[1], sb[2], sb[3], densify_pts=21)
        src_cell = (vb[2] - vb[0]) / max(cols, 1)  # a source pixel in the units of the view
        k = max(1, int(src_cell / 4.0 / abs(view.tr.a)), -(-max(bw, bh) // 3000))  # at most 3000 cells a side
        ow, oh = -(-bw // k), -(-bh // k)
        dst_tr = view.tr @ Affine.scale(k)
        if kind == "rgb":
            out = np.stack([onto(b, transform, crs, dst_tr, view_crs, (oh, ow)) for b in values])
            lims = limits or rgb_limits(values)
            rgba = render_rgb(out, lims)
        else:
            out = onto(values, transform, crs, dst_tr, view_crs, (oh, ow))
            lims = limits or limits_for(values, kind)
            rgba = render(out, lims[0], lims[1], palette)
        item = view.scene().addPixmap(QPixmap.fromImage(to_qimage(rgba)))
        item.setScale(k)
        item.setZValue(-0.5)  # over the raster (-2) and its detail (-1), under the drawings
        item.setOpacity(float(opacity))
        self.item, self.spec, self.scale = item, spec, k
        return lims

    def set_opacity(self, opacity: float) -> None:
        if self._alive():
            self.item.setOpacity(float(opacity))

    def remove(self, forget: bool = False) -> None:
        if self._alive():
            try:
                self.view.scene().removeItem(self.item)
            except RuntimeError:
                pass
        self.item = None
        if forget:
            self.spec = None
