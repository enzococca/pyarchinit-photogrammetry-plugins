"""The contact surfaces of the units and their thicknesses.

The top of every unit is interpolated from its elevations at the boreholes (or from its depths
below the ground, in ``mode="depth"``, then taken from the ground surface) on a grid aligned
with the DEM of the chunk - the DEM's own cells, or coarser ones for a large DEM, or a cell size
given - over the boreholes and a margin. Kriging (:mod:`.kriging`), inverse distance weighting
below ``min_points`` boreholes. Where a borehole ends above a contact, the contact is kept below
its bottom (the bottom becomes a datum when the first estimate passes above it).

Then the stack is made consistent: no contact above the ground (the DEM, filled with the
surface of the collars where the DEM has no cells) and none above the contact of the unit over
it; every correction is counted and kept as the third band of the surface. The top of a unit
crossed at the collar of every borehole is the ground itself.

Every surface is ``<chunk data>/geoarchaeology/surfaces/top_<k>_<unit>.tif``: float32, band 1
the elevation, band 2 the kriging variance (m², NaN for inverse distance), band 3 how much the
estimate was lowered by the clamping (m), NaN outside the hull of its boreholes plus a buffer.
The thickness of a unit (isopach) is the difference of its top and the top of the unit below:
``thickness_<unit>.tif``. Both are registered in ``chunk.meta["geoarchaeology"]["surfaces"]``.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import boreholes as bh
from . import kriging
from .texts import t

MAX_CELLS = 1_000_000  # cells of a grid made from the DEM's when no cell size is given
NICE = (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
STANDIN_CRS = "EPSG:3857"  # resampling between grids of one frame: any projected CRS does


@dataclass
class Grid:
    """A north-up grid: the corner (``x0``, ``y0`` top-left), the cell, the size."""

    x0: float
    y0: float
    cell: float
    width: int
    height: int
    crs: str | None = None

    @property
    def transform(self):
        from rasterio.transform import from_origin

        return from_origin(self.x0, self.y0, self.cell, self.cell)

    def centres(self, mask: np.ndarray | None = None) -> np.ndarray:
        """World coordinates of the cell centres (of the cells of ``mask``), ``(N, 2)``."""
        if mask is None:
            r, c = np.indices((self.height, self.width))
            r, c = r.ravel(), c.ravel()
        else:
            r, c = np.nonzero(mask)
        return np.column_stack([self.x0 + (c + 0.5) * self.cell, self.y0 - (r + 0.5) * self.cell])

    def to_dict(self) -> dict:
        return {"x0": self.x0, "y0": self.y0, "cell": self.cell, "width": self.width, "height": self.height,
                "crs": self.crs}

    @classmethod
    def from_dict(cls, d: dict) -> "Grid":
        return cls(float(d["x0"]), float(d["y0"]), float(d["cell"]), int(d["width"]), int(d["height"]), d.get("crs"))


def folder(chunk) -> Path:
    return bh.folder(chunk) / "surfaces"


def _nice(value: float) -> float:
    return next((v for v in NICE if v >= value), float(10 ** np.ceil(np.log10(value))))


def chunk_crs(chunk) -> str | None:
    from pyarchinit_photogrammetry.core import crs as crs_mod

    return None if crs_mod.is_local(getattr(chunk, "crs", None)) else str(chunk.crs)


def make_grid(chunk, xy: np.ndarray, buffer: float, cell: float | None = None) -> Grid:
    """The grid over the points ``xy`` and ``buffer`` around them: aligned with the DEM of the
    chunk (its cell, or an integer multiple of it to stay within :data:`MAX_CELLS`, or ``cell``),
    or with round coordinates when there is no DEM."""
    import rasterio

    xmin, ymin = xy.min(0) - buffer
    xmax, ymax = xy.max(0) + buffer
    crs = chunk_crs(chunk)
    dem = bh.dem_path(chunk)
    if dem is not None:
        with rasterio.open(dem) as src:
            tr = src.transform
        base = abs(tr.a)
        if cell is None:
            n = (xmax - xmin) * (ymax - ymin) / (base * base)
            cell = base * max(1, int(np.ceil(np.sqrt(n / MAX_CELLS))))
        ox, oy = tr.c, tr.f
    else:
        if cell is None:
            cell = _nice(max(xmax - xmin, ymax - ymin) / 300.0)
        ox, oy = 0.0, 0.0
    cell = float(cell)
    x0 = ox + np.floor((xmin - ox) / cell) * cell
    y0 = oy + np.ceil((ymax - oy) / cell) * cell
    width = max(1, int(np.ceil((xmax - x0) / cell)))
    height = max(1, int(np.ceil((y0 - ymin) / cell)))
    return Grid(float(x0), float(y0), cell, width, height, crs)


def dem_on_grid(chunk, grid: Grid) -> np.ndarray | None:
    """The DEM of the chunk on ``grid`` (the mean of the DEM cells in each cell), NaN where it
    has none; None without a DEM."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine
    from rasterio.warp import reproject
    from rasterio.windows import Window

    dem = bh.dem_path(chunk)
    if dem is None:
        return None
    out = np.full((grid.height, grid.width), np.nan, np.float32)
    with rasterio.open(dem) as src:
        inv = ~src.transform
        c0, r0 = inv @ (grid.x0, grid.y0)
        c1, r1 = inv @ (grid.x0 + grid.width * grid.cell, grid.y0 - grid.height * grid.cell)
        ca, ra = max(int(np.floor(min(c0, c1))), 0), max(int(np.floor(min(r0, r1))), 0)
        cb, rb = min(int(np.ceil(max(c0, c1))), src.width), min(int(np.ceil(max(r0, r1))), src.height)
        if cb <= ca or rb <= ra:
            return out
        win = Window(ca, ra, cb - ca, rb - ra)
        dec = max(1, int(np.ceil(np.sqrt((cb - ca) * (rb - ra) / (4.0 * MAX_CELLS)))))  # a large DEM read reduced
        shape = (int(np.ceil((rb - ra) / dec)), int(np.ceil((cb - ca) / dec)))
        a = src.read(1, window=win, out_shape=shape, resampling=Resampling.average, masked=True)
        a = np.ma.filled(a.astype(np.float32), np.nan)
        wtr = src.window_transform(win) @ Affine.scale((cb - ca) / shape[1], (rb - ra) / shape[0])
        crs = src.crs or STANDIN_CRS
    same = dec == 1 and abs(abs(wtr.a) - grid.cell) < 1e-9 * grid.cell
    reproject(source=a, destination=out, src_transform=wtr, src_crs=crs, dst_transform=grid.transform, dst_crs=crs,
              src_nodata=np.nan, dst_nodata=np.nan, resampling=Resampling.nearest if same else Resampling.average)
    return out


def hull_mask(grid: Grid, xy: np.ndarray, buffer: float) -> np.ndarray:
    """The cells whose centre lies in the convex hull of ``xy`` grown by ``buffer``."""
    from rasterio.features import geometry_mask
    from shapely.geometry import MultiPoint

    shape = MultiPoint([tuple(p) for p in xy]).convex_hull.buffer(max(buffer, 1e-6))
    return geometry_mask([shape], out_shape=(grid.height, grid.width), transform=grid.transform, invert=True)


def default_buffer(xy: np.ndarray, cell: float = 0.0) -> float:
    """The median distance of a borehole from its nearest one (at least two cells)."""
    from scipy.spatial import cKDTree

    if len(xy) < 2:
        return max(5.0, 2 * cell)
    d, _ = cKDTree(xy).query(xy, 2)
    return float(max(np.median(d[:, 1]), 2 * cell, 1e-3))


def clamp_stack(ground: np.ndarray | None, raws: list[np.ndarray], is_ground: list[bool] | None = None
                ) -> tuple[list, list, list]:
    """Every contact brought down to the ground and to the contacts above it, from the top: returns
    the contacts, the corrections (raw minus kept) and, per contact, the counts of the cells
    lowered by the ground and by an upper contact (a contact that is the ground itself,
    ``is_ground``, counts as the ground)."""
    is_ground = list(is_ground or [False] * len(raws))
    ceiling = None if ground is None else ground.astype(np.float64).copy()
    source = None if ground is None else np.where(np.isfinite(ground), 0, -1)
    out, corr, counts = [], [], []
    for k, raw in enumerate(raws):
        c = raw.astype(np.float64).copy()
        n_ground = n_upper = 0
        if ceiling is not None:
            over = np.isfinite(c) & np.isfinite(ceiling) & (c > ceiling)
            n_ground = int((over & (source == 0)).sum())
            n_upper = int((over & (source > 0)).sum())
            c[over] = ceiling[over]
        out.append(c)
        corr.append(np.where(np.isfinite(raw), raw - c, np.nan))
        counts.append((n_ground, n_upper))
        tag = 0 if is_ground[k] else k + 1
        if ceiling is None:
            ceiling = c.copy()
            source = np.where(np.isfinite(c), tag, -1)
        else:
            upd = np.isfinite(c)
            ceiling = np.where(upd, c, ceiling)
            source = np.where(upd, tag, source)
    return out, corr, counts


def write_tif(path: Path, bands: list[np.ndarray], grid: Grid, descriptions: list[str]) -> Path:
    import rasterio

    from pyarchinit_photogrammetry.core import crs as crs_mod

    path.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width, "count": len(bands), "dtype": "float32",
               "transform": grid.transform, "nodata": float("nan"), "compress": "deflate"}
    if grid.width >= 256 and grid.height >= 256:
        profile.update(tiled=True, blockxsize=256, blockysize=256)
    if grid.crs:
        profile["crs"] = crs_mod.to_pyproj(grid.crs).to_wkt()
    tmp = path.with_suffix(".tmp.tif")
    with rasterio.open(tmp, "w", **profile) as dst:
        for k, (b, d) in enumerate(zip(bands, descriptions), start=1):
            dst.write(np.asarray(b, np.float32), k)
            dst.set_band_description(k, d)
    tmp.replace(path)
    return path


def read_band(path: Path, band: int = 1) -> tuple[np.ndarray, object]:
    import rasterio

    with rasterio.open(path) as src:
        return src.read(band, masked=True).astype(np.float64).filled(np.nan), src.transform


def _stats(v: np.ndarray) -> dict:
    ok = np.isfinite(v)
    out = {"cells": int(ok.sum())}
    if ok.any():
        w = v[ok]
        out.update(min=float(w.min()), max=float(w.max()), mean=float(w.mean()), median=float(np.median(w)),
                   p2=float(np.percentile(w, 2)), p98=float(np.percentile(w, 98)))
    return out


def slug(text: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in str(text))
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("_")[:40] or "u"


# ------------------------------------------------------------------ building
def build(chunk, model: str = "auto", cell: float | None = None, nugget: float | None = None,
          mode: str = "elevation", trend: str = "auto", buffer: float | None = None, min_points: int = 5,
          codes: list[str] | None = None, progress=None) -> dict:
    """The tops of the units (in their order, or ``codes``) and their thicknesses on a grid: see
    the module. ``model``: "auto" (the variogram model with the least leave-one-out error),
    "spherical", "exponential" or "gaussian"; ``trend``: "auto", "none" or "linear" (universal
    kriging with a plane); ``mode``: "elevation" (the elevations of the contacts) or "depth" (their
    depths below the ground, then hung from it). Returns the registry of the products."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    if mode not in ("elevation", "depth"):
        raise ValueError(f"mode {mode!r}: elevation or depth")
    holes = bh.load(chunk)
    if not holes:
        raise RuntimeError(t("Nessun sondaggio nel chunk: importali prima", "No borehole in the chunk: import them first"))
    if mode == "elevation" and all(b.z is None for b in holes):
        raise RuntimeError(t("Nessun sondaggio ha una quota (né il chunk un DEM sotto di essi): le superfici vogliono "
                             "le quote, oppure il modo per profondità con un DEM",
                             "No borehole has an elevation (nor a DEM under it): the surfaces need elevations, or the "
                             "depth mode with a DEM"))
    if codes:
        bh.set_order(chunk, codes)
    codes = bh.order(chunk)
    data, notes = bh.contact_data(holes, codes)
    positioned = np.array([[b.x, b.y] for b in holes], float)
    progress(0.02, "Grid")
    base_cell = 0.0
    dem = bh.dem_path(chunk)
    if dem is not None:
        import rasterio

        with rasterio.open(dem) as src:
            base_cell = abs(src.transform.a)
    buffer = float(buffer) if buffer is not None else default_buffer(positioned, cell or base_cell)
    grid = make_grid(chunk, positioned, buffer, cell)
    ground_dem = dem_on_grid(chunk, grid)
    opts = dict(model=model, nugget=nugget, trend=trend, min_points=min_points)
    # the surface of the collars: the ground where the DEM has no cells, and the ground without a DEM
    collars = [b for b in holes if b.z is not None]
    collar_surface, collar_report = None, None
    if collars:
        cxy = np.array([[b.x, b.y] for b in collars])
        cz = np.array([b.z for b in collars])
        cmask = hull_mask(grid, cxy, buffer)
        if ground_dem is None or not np.isfinite(ground_dem[cmask]).all():
            progress(0.05, "Surface of the collars")
            collar_surface = np.full((grid.height, grid.width), np.nan)
            est, _var, collar_report = kriging.interpolate(cxy, cz, grid.centres(cmask), variance=False, **opts)
            collar_surface[cmask] = est
    if ground_dem is not None:
        ground = np.where(np.isfinite(ground_dem), ground_dem, collar_surface if collar_surface is not None else np.nan)
    else:
        ground = collar_surface
    raws, variances, reports = [], [], []
    for k, cd in enumerate(data):
        p = progress.sub(0.08 + 0.75 * k / max(len(data), 1), 0.75 / max(len(data), 1))
        p(0.0, f"Top of {cd.unit}: {len(cd.ids)} boreholes")
        raw, var, rep = _contact(grid, cd, ground, mode, buffer, opts, p)
        raws.append(raw)
        variances.append(var)
        reports.append(rep)
    progress(0.85, "Clamping the stack")
    tops, corrections, counts = clamp_stack(ground, raws, [r.get("method") == "ground" for r in reports])
    out_dir = folder(chunk)
    if out_dir.exists():
        for old in [*out_dir.glob("*.tif"), *out_dir.glob("variogram_*.png")]:
            old.unlink()
    colours = bh.colours(chunk)
    cell_area = grid.cell * grid.cell
    entries, thick = [], []
    for k, cd in enumerate(data):
        rep = reports[k]
        name = f"top_{k}_{slug(cd.unit)}"
        path = write_tif(out_dir / f"{name}.tif", [tops[k], variances[k], corrections[k]], grid,
                         [f"top of {cd.unit} (m)", "kriging variance (m2)", "lowered by the clamping (m)"])
        lowered = np.isfinite(corrections[k]) & (corrections[k] > 1e-6)
        rep["clamped"] = {"cells": int(lowered.sum()), "area_m2": float(lowered.sum() * cell_area),
                          "max_m": float(np.nanmax(corrections[k][lowered])) if lowered.any() else 0.0,
                          "by_ground": counts[k][0], "by_upper": counts[k][1]}
        entries.append({"id": name, "kind": "surface", "unit": cd.unit, "k": k, "colour": colours.get(cd.unit),
                        "name": t("Tetto di {0}", "Top of {0}").format(cd.unit),
                        "file": str(path.relative_to(chunk.data_dir)), "stats": _stats(tops[k]),
                        "variance": _stats(variances[k]), **rep})
    progress(0.92, "Thicknesses")
    for k in range(len(data) - 1):
        top, base = tops[k], tops[k + 1]
        th = np.where(np.isfinite(top) & np.isfinite(base), np.maximum(top - base, 0.0), np.nan)
        unit = data[k].unit
        name = f"thickness_{slug(unit)}"
        path = write_tif(out_dir / f"{name}.tif", [th], grid, [f"thickness of {unit} (m)"])
        st = _stats(th)
        thick.append({"id": name, "kind": "thickness", "unit": unit, "k": k, "colour": colours.get(unit),
                      "name": t("Spessore di {0}", "Thickness of {0}").format(unit),
                      "file": str(path.relative_to(chunk.data_dir)), "stats": st,
                      "volume_m3": float(np.nansum(th) * cell_area), "area_m2": float((th > 1e-6).sum() * cell_area)})
    registry = {"grid": grid.to_dict(), "made": time.strftime("%Y-%m-%d %H:%M"), "stale": False,
                "params": {"model": model, "cell": cell, "nugget": nugget, "mode": mode, "trend": trend,
                           "buffer": buffer, "min_points": min_points},
                "order": codes, "notes": notes, "contacts": entries, "thickness": thick,
                "ground": "dem" if ground_dem is not None else ("collars" if collar_surface is not None else "none")}
    if collar_report is not None:
        registry["collars"] = {k: v for k, v in collar_report.items() if k != "candidates"}
    bh.meta(chunk)["surfaces"] = registry
    try:
        variogram_figures(chunk)
    except Exception:  # noqa: BLE001 - the figures are a convenience (no matplotlib, for one)
        pass
    progress(1.0, f"{len(entries)} contacts, {len(thick)} thicknesses on {grid.width} x {grid.height} cells of {grid.cell:g} m")
    return registry


def _contact(grid: Grid, cd, ground, mode: str, buffer: float, opts: dict, progress) -> tuple:
    """One contact on the grid: ``(elevation, variance, report)`` before the clamping."""
    shape = (grid.height, grid.width)
    raw = np.full(shape, np.nan)
    var = np.full(shape, np.nan)
    if cd.at_surface and ground is not None:  # the top of the logs: the ground itself
        mask = hull_mask(grid, cd.xy, buffer)
        raw[mask] = ground[mask]
        return raw, var, {"method": "ground", "points": len(cd.ids)}
    if mode == "depth":
        if ground is None:
            raise RuntimeError(t("Le profondità si appendono al terreno: serve il DEM o la quota dei sondaggi",
                                 "Depths hang from the ground: the DEM or the elevations of the boreholes are needed"))
        ok = np.isfinite(cd.depth)
        xy, values = cd.xy[ok], cd.depth[ok]
        bxy, bvalues = cd.below_xy, cd.below_depth
    else:
        ok = np.isfinite(cd.z)
        xy, values = cd.xy[ok], cd.z[ok]
        good = np.isfinite(cd.below_z)
        bxy, bvalues = cd.below_xy[good], cd.below_z[good]
    if len(values) == 0:
        return raw, var, {"method": "none", "points": 0}
    mask = hull_mask(grid, xy, buffer)
    targets = grid.centres(mask)
    m = len(targets)
    added = 0
    for _round in range(5):  # boreholes ending above the contact: it must lie below their bottoms
        pts = np.vstack([targets, bxy]) if len(bvalues) else targets
        est_all, v_all, rep = kriging.interpolate(xy, values, pts, progress=progress, **opts)
        est, v = est_all[:m], v_all[:m]
        if len(bvalues) == 0:
            break
        at = est_all[m:]
        # a depth must reach at least the bottom, an elevation stay at most at it
        wrong = (at < bvalues - 1e-6) if mode == "depth" else (at > bvalues + 1e-6)
        if not wrong.any():
            break
        xy = np.vstack([xy, bxy[wrong]])
        values = np.concatenate([values, bvalues[wrong]])
        added += int(wrong.sum())
        bxy, bvalues = bxy[~wrong], bvalues[~wrong]
    raw[mask] = ground[mask] - est if mode == "depth" else est
    var[mask] = v
    rep = dict(rep)
    rep.update(points=len(cd.ids), below=len(cd.below_ids), bottoms_used=added, mode=mode)
    return raw, var, rep


# ------------------------------------------------------------------ the registry
def registry(chunk) -> dict:
    return (chunk.meta.get("geoarchaeology") or {}).get("surfaces") or {}


def entries(chunk) -> list[dict]:
    """The contacts and the thicknesses of the chunk, for the pane and the exports."""
    reg = registry(chunk)
    return list(reg.get("contacts", [])) + list(reg.get("thickness", []))


def entry(chunk, pid: str) -> dict | None:
    return next((e for e in entries(chunk) if e.get("id") == pid), None)


def grid_of(chunk) -> Grid | None:
    reg = registry(chunk)
    return Grid.from_dict(reg["grid"]) if reg.get("grid") else None


def load(chunk, pid: str, band: int = 1) -> tuple[np.ndarray, object]:
    """``(values with NaN, transform)`` of a band of a product."""
    e = entry(chunk, pid)
    if e is None:
        raise KeyError(pid)
    return read_band(Path(chunk.data_dir) / e["file"], band)


def tops(chunk) -> list[tuple[str, np.ndarray]]:
    """``(unit, top elevation)`` of every contact, from the top down."""
    out = []
    for e in registry(chunk).get("contacts", []):
        values, _tr = read_band(Path(chunk.data_dir) / e["file"], 1)
        out.append((e["unit"], values))
    return out


def remove(chunk) -> None:
    out_dir = folder(chunk)
    if out_dir.exists():
        for p in out_dir.iterdir():
            if p.is_file():
                p.unlink()
    bh.meta(chunk).pop("surfaces", None)


def variogram_figures(chunk) -> list[Path]:
    """A figure per kriged contact: the experimental variogram of its points and the model."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    out = []
    holes = bh.load(chunk)
    reg = registry(chunk)
    data, _ = bh.contact_data(holes, reg.get("order", []))
    for e, cd in zip(reg.get("contacts", []), data):
        if e.get("method") != "kriging":
            continue
        mode = e.get("mode", "elevation")
        ok = np.isfinite(cd.depth if mode == "depth" else cd.z)
        xy, z = cd.xy[ok], (cd.depth if mode == "depth" else cd.z)[ok]
        if e.get("trend") == "linear":
            c = xy.mean(0)
            s = max(float(np.ptp(xy, axis=0).max()), 1.0)
            coef = kriging.plane((xy - c) / s, z)
            z = z - kriging._drift((xy - c) / s, 1) @ coef
        lags, gam, pairs = kriging.empirical(xy, z)
        v = kriging.Variogram(**{k: e["variogram"][k] for k in ("model", "nugget", "sill", "range")})
        fig = Figure(figsize=(4.5, 3.2), dpi=110)
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        ax.scatter(lags, gam, s=12 + 40 * pairs / max(pairs.max(), 1), color="#7d5a3c", zorder=3,
                   label=t("sperimentale", "experimental"))
        h = np.linspace(0, max(lags.max(), v.range) * 1.1, 200)
        ax.plot(h[1:], v(h[1:]), color="#1f4e79", label=f"{v.model}, {t('portata', 'range')} {v.range:.3g} m")
        ax.set_xlabel(t("distanza (m)", "distance (m)"))
        ax.set_ylabel(t("semivarianza (m²)", "semivariance (m²)"))
        ax.set_title(f"{e['name']} - LOO {e.get('loocv_rmse', float('nan')):.3f} m", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = folder(chunk) / f"variogram_{e['id']}.png"
        fig.savefig(path)
        out.append(path)
    return out
