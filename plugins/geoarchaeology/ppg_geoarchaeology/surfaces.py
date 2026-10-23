"""The contact surfaces of the units and their thicknesses.

The top of every unit is interpolated on a grid aligned with the DEM of the chunk - the DEM's
own cells, or coarser ones for a large DEM, or a cell size given - over the boreholes and a
margin: kriging (:mod:`.kriging`), inverse distance weighting below ``min_points`` boreholes.
Each contact is interpolated in one of two ways:

* by elevation: its elevations at the boreholes - a contact with a shape of its own (the
  natural soil under a tell, a buried floor, a channel);
* by depth: its depths below the ground, then hung from the ground - a contact that follows
  the ground (the base of the topsoil, the layers draped over a mound).

``mode="auto"`` chooses for every contact (:func:`choose_mode`): by depth when its depths vary
across the boreholes much less than its elevations (the scatter of each about a plane, less
than :data:`FOLLOWS_GROUND` times), by elevation otherwise; ``modes`` (or ``"US2=depth"`` in
``mode``) sets it for the units named. The registry tells the mode of every contact and why.
Where a borehole ends above a contact, the contact is kept below its bottom (the bottom becomes
a datum when the first estimate passes above it).

Then the stack is made consistent (:func:`clamp_stack`): where two contacts cross, the one with
the larger kriging variance yields - in every column the variance-weighted regression that
keeps the contacts in order, so a well determined natural is not pushed down by an uncertain
contact above it - and none rises above the ground (the DEM, filled with the surface of the
collars where the DEM has no cells). A unit missing from a log is missing around it too: the
boreholes vote for the presence of every unit (inverse distance), and where those without it
win its top and its base are one surface. Every correction is counted and kept as the third
band. The top of a unit crossed at the collar of every borehole is the ground itself.

Every surface is ``<chunk data>/geoarchaeology/surfaces/top_<k>_<unit>.tif``: float32, band 1
the elevation, band 2 the kriging variance (m², NaN for inverse distance), band 3 the
correction made by the stack (m: positive where the estimate was lowered, negative where it was
raised), NaN outside the hull of its boreholes plus a buffer. The thickness of a unit (isopach)
is the difference of its top and the top of the unit below: ``thickness_<unit>.tif``; its volume
is summed over the cells with a DEM, the cells whose ground is the surface of the collars are
reported apart. Both are registered in ``chunk.meta["geoarchaeology"]["surfaces"]``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import boreholes as bh
from . import kriging
from .texts import num, t

MAX_CELLS = 1_000_000  # cells of a grid made from the DEM's when no cell size is given
NICE = (0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0, 25.0, 50.0, 100.0)
STANDIN_CRS = "EPSG:3857"  # resampling between grids of one frame: any projected CRS does
MODES = ("elevation", "depth")
FOLLOWS_GROUND = 0.5  # by depth when the scatter of the depths is under this share of that of the elevations
PRESENT = 0.5  # a unit is there where the boreholes that have it outweigh (inverse distance) those that do not
SNAP_SIGMAS = 2.0  # where it is not, a thickness under this many standard deviations of its contacts is none
SNAP_FLOOR = 0.01  # m: and always one under a centimetre
EXACT_VARIANCE = 1e-6  # m²: the variance given to an exact contact (the ground, a constant one) in the stack
_MODE_WORDS = {"auto": "auto", "elevation": "elevation", "elevations": "elevation", "elev": "elevation",
               "quota": "elevation", "quote": "elevation", "q": "elevation", "z": "elevation", "depth": "depth",
               "depths": "depth", "profondita": "depth", "prof": "depth", "p": "depth", "d": "depth"}


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


def clamp_stack(ground: np.ndarray | None, raws: list[np.ndarray], is_ground: list[bool] | None = None,
                variances: list | None = None, ties: list | None = None) -> tuple[list, list, list]:
    """The contacts (from the top down) made a consistent stack: none above the ground, none
    above the contact of the unit over it. Returns the contacts, the corrections (raw minus kept:
    positive lowered, negative raised) and, per contact, the cells ``(lowered to the ground,
    lowered by a contact above, raised by a contact below)``. A contact that is the ground itself
    (``is_ground``) is exact.

    With ``variances`` (the kriging variance of each contact, NaN where unknown) the contacts that
    cross meet at their variance-weighted mean, so the one with the larger variance yields: in
    every column the weighted regression that keeps the contacts in order (pool adjacent
    violators), then capped by the ground. ``ties`` (for unit k, the cells where it is missing,
    or None) make the top of unit k and the top of unit k + 1 one surface there. Without either:
    every contact brought down to the ground and to the contacts above it, from the top."""
    if variances is None and ties is None:
        return _top_down(ground, raws, is_ground)
    return _weighted_stack(ground, raws, is_ground, variances, ties)


def _weighted_stack(ground, raws, is_ground, variances, ties) -> tuple[list, list, list]:
    n = len(raws)
    if n == 0:
        return [], [], []
    is_ground = list(is_ground or [False] * n)
    C = np.stack([np.asarray(r, np.float64) for r in raws])
    shape = C.shape[1:]
    ok = np.isfinite(C)
    if variances is None:
        V = np.ones_like(C)
    else:
        V = np.stack([np.full(shape, np.nan) if v is None else np.asarray(v, np.float64) for v in variances])
    known = ok & np.isfinite(V)
    fallback = max(float(np.median(V[known])), EXACT_VARIANCE) if known.any() else 1.0  # inverse distance: unknown
    V = np.where(np.isfinite(V), np.maximum(V, EXACT_VARIANCE), fallback)
    for k in range(n):
        if is_ground[k]:
            V[k] = EXACT_VARIANCE
    W = np.where(ok, 1.0 / V, 0.0).reshape(n, -1)
    X = np.where(ok, C, 0.0).reshape(n, -1)
    tie = np.zeros((max(n - 1, 0), W.shape[1]), bool)
    for k, tk in enumerate((ties or [])[:n - 1]):
        if tk is not None:
            tie[k] = np.asarray(tk, bool).ravel()
    out = np.full(W.shape, np.nan)
    step = 1 << 18
    for s in range(0, W.shape[1], step):
        w, x, tt = W[:, s:s + step], X[:, s:s + step], tie[:, s:s + step]
        m = w.shape[1]
        start = np.ones((n, m), bool)  # a contact tied to the one above (its unit missing) is not a start
        end = np.ones((n, m), bool)  # nor, tied to the one below, an end
        if n > 1:
            start[1:] = ~tt
            end[:-1] = ~tt
        mean = {}
        for i in range(n):
            sw, sx = np.zeros(m), np.zeros(m)
            for j in range(i, n):
                sw += w[j]
                sx += w[j] * x[j]
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean[i, j] = np.where(sw > 0, sx / sw, np.nan)
        # the antitonic regression: t_k = min over the starts i <= k of max over the ends j >= k of mean(i..j)
        for k in range(n):
            low = np.full(m, np.inf)
            for i in range(k + 1):
                high = np.full(m, -np.inf)
                for j in range(k, n):
                    high = np.fmax(high, np.where(end[j], mean[i, j], -np.inf))
                low = np.fmin(low, np.where(start[i], high, np.inf))
            out[k, s:s + step] = low
    out = out.reshape(C.shape)
    out[~ok] = np.nan
    G = None if ground is None else np.asarray(ground, np.float64)
    if G is not None:
        cap = np.isfinite(G)
        out = np.where(cap & ok, np.minimum(out, np.where(cap, G, np.inf)), out)
    corr = np.where(ok, C - out, np.nan)
    counts = []
    for k in range(n):
        d = corr[k]
        lowered = np.isfinite(d) & (d > 1e-6)
        raised = np.isfinite(d) & (d < -1e-6)
        if G is not None:
            at_ground = lowered & np.isfinite(G) & (np.abs(out[k] - np.where(np.isfinite(G), G, np.nan)) < 1e-6)
        else:
            at_ground = np.zeros_like(lowered)
        counts.append((int(at_ground.sum()), int((lowered & ~at_ground).sum()), int(raised.sum())))
    return [out[k] for k in range(n)], [corr[k] for k in range(n)], counts


def _top_down(ground, raws, is_ground) -> tuple[list, list, list]:
    """The rule without variances: every contact brought down to the ground and to the contacts
    above it, from the top (a contact that is the ground itself counts as the ground)."""
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
        counts.append((n_ground, n_upper, 0))
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


# ------------------------------------------------------------------ the mode of every contact
def mode_word(text: str) -> str:
    """"auto", "elevation" or "depth" for a word of the user (English or Italian: quota, profondità)."""
    word = bh.normalise(text)
    if word not in _MODE_WORDS:
        raise ValueError(t("Modo sconosciuto: {0} (auto, quota/elevation, profondità/depth)",
                           "Unknown mode: {0} (auto, elevation/quota, depth/profondità)").format(text))
    return _MODE_WORDS[word]


def parse_modes(text) -> tuple[str, dict]:
    """``(mode of the contacts not named, {unit: mode})`` of a text such as ``"auto"``,
    ``"depth"``, ``"US2=depth, US5=elevation"`` or ``"auto; US 2=profondità"``."""
    default, given = "auto", {}
    for part in re.split(r"[;,]", str(text or "")):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            unit, word = part.split("=", 1)
            if not unit.strip():
                raise ValueError(t("Modo senza unità: {0}", "A mode without its unit: {0}").format(part))
            given[unit.strip()] = mode_word(word)
        else:
            default = mode_word(part)
    return default, given


def unit_of(name: str, codes: list[str]) -> str:
    """The unit of ``codes`` that ``name`` means, in any spelling ("US2", "us 2" and, when one
    unit only has that number, "2" are "US 2"): :func:`boreholes.resolve_unit`."""
    unit = bh.resolve_unit(name, codes)
    if unit is None:
        raise ValueError(t("Unità sconosciuta: {0} (le unità: {1})", "Unknown unit: {0} (the units: {1})")
                         .format(name, ", ".join(codes)))
    return unit


def scatter(xy, values) -> float:
    """How much ``values`` vary across the boreholes at ``xy``: the root mean square about their
    plane (four boreholes or more, not in a line), their standard deviation otherwise."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    values = np.asarray(values, float)
    if len(values) < 2:
        return 0.0
    if len(values) >= 4 and kriging.spans_a_plane(xy):
        c = xy.mean(0)
        s = max(float(np.ptp(xy, axis=0).max()), 1.0)
        coef = kriging.plane((xy - c) / s, values)
        res = values - kriging._drift((xy - c) / s, 1) @ coef
        return float(np.sqrt(np.mean(res ** 2)))
    return float(np.std(values))


def choose_mode(cd, has_ground: bool) -> tuple[str, dict]:
    """``(mode, why)`` of a contact under ``mode="auto"``: by depth when the scatter of its depths
    at the boreholes is less than :data:`FOLLOWS_GROUND` times that of its elevations (it follows
    the ground), by elevation otherwise (a shape of its own: the depths vary as much as the ground).
    ``why["rule"]``: ``follows_ground``, ``own_shape``, ``no_ground``, ``no_elevations`` or
    ``few_boreholes``, with the two scatters."""
    if not has_ground:
        return "elevation", {"rule": "no_ground"}
    if not np.isfinite(cd.z).any():
        return "depth", {"rule": "no_elevations"}
    ok = np.isfinite(cd.z) & np.isfinite(cd.depth)
    if ok.sum() < 3:
        return "elevation", {"rule": "few_boreholes", "boreholes": int(ok.sum())}
    sz = scatter(cd.xy[ok], cd.z[ok])
    sd = scatter(cd.xy[ok], cd.depth[ok])
    why = {"scatter_depth_m": sd, "scatter_elevation_m": sz, "threshold": FOLLOWS_GROUND, "boreholes": int(ok.sum())}
    if sz > 1e-6 and sd < FOLLOWS_GROUND * sz:
        return "depth", {"rule": "follows_ground", **why}
    return "elevation", {"rule": "own_shape", **why}


# ------------------------------------------------------------------ building
def build(chunk, model: str = "auto", cell: float | None = None, nugget: float | None = None,
          mode: str = "auto", trend: str = "auto", buffer: float | None = None, min_points: int = 5,
          codes: list[str] | None = None, progress=None, modes: dict | None = None) -> dict:
    """The tops of the units (in their order, or ``codes``) and their thicknesses on a grid: see
    the module. ``model``: "auto" (the steadiest variogram model among those within a few per cent
    of the least leave-one-out error), "spherical", "exponential" or "gaussian"; ``trend``: "auto",
    "none" or "linear" (universal kriging with a plane); ``mode``: "auto" (by depth the contacts
    that follow the ground, by elevation the others: :func:`choose_mode`), "elevation", "depth",
    or a text with the units named (``"auto, US2=depth"``); ``modes``: ``{unit: mode}`` for some
    units. Returns the registry of the products."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    default, given = parse_modes(mode)
    for unit, m in (modes or {}).items():
        given[unit] = mode_word(m)
    holes = bh.load(chunk)
    if not holes:
        raise RuntimeError(t("Nessun sondaggio nel chunk: importali prima", "No borehole in the chunk: import them first"))
    if all(b.z is None for b in holes) and (default == "elevation" or bh.dem_path(chunk) is None):
        raise RuntimeError(t("Nessun sondaggio ha una quota (né il chunk un DEM sotto di essi): le superfici vogliono "
                             "le quote, oppure il modo per profondità con un DEM",
                             "No borehole has an elevation (nor a DEM under it): the surfaces need elevations, or the "
                             "depth mode with a DEM"))
    if codes:
        bh.set_order(chunk, codes)
    codes = bh.order(chunk)
    given = {unit_of(u, codes): m for u, m in given.items()}
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
        p = progress.sub(0.08 + 0.72 * k / max(len(data), 1), 0.72 / max(len(data), 1))
        p(0.0, f"Top of {cd.unit}: {len(cd.ids)} boreholes")
        if cd.at_surface and ground is not None:
            m, why = "ground", {"rule": "ground"}
        else:
            m = given.get(cd.unit, default)
            why = {"rule": "given"}
            if m == "auto":
                m, why = choose_mode(cd, ground is not None)
        raw, var, rep = _contact(grid, cd, ground, "elevation" if m == "ground" else m, buffer, opts, p)
        rep["mode"] = "ground" if rep.get("method") == "ground" else m
        rep["mode_reason"] = why
        raws.append(raw)
        variances.append(var)
        reports.append(rep)
    progress(0.82, "Units missing from the logs")
    ties, absent = _missing_units(bh.presence(holes, codes), raws, variances, grid)
    progress(0.86, "The stack")
    tops, corrections, counts = clamp_stack(ground, raws, [r.get("method") == "ground" for r in reports],
                                            variances=variances, ties=ties)
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
                         [f"top of {cd.unit} (m)", "kriging variance (m2)",
                          "correction by the stack (m, + lowered, - raised)"])
        d = corrections[k]
        lowered = np.isfinite(d) & (d > 1e-6)
        raised = np.isfinite(d) & (d < -1e-6)
        moved = lowered | raised
        rep["clamped"] = {"cells": int(moved.sum()), "area_m2": float(moved.sum() * cell_area),
                          "max_m": float(np.nanmax(np.abs(d[moved]))) if moved.any() else 0.0,
                          "lowered_max_m": float(d[lowered].max()) if lowered.any() else 0.0,
                          "raised_max_m": float(-d[raised].min()) if raised.any() else 0.0,
                          "by_ground": counts[k][0], "by_upper": counts[k][1], "by_lower": counts[k][2]}
        entries.append({"id": name, "kind": "surface", "unit": cd.unit, "k": k, "colour": colours.get(cd.unit),
                        "name": t("Tetto di {0}", "Top of {0}").format(cd.unit),
                        "file": str(path.relative_to(chunk.data_dir)), "stats": _stats(tops[k]),
                        "variance": _stats(variances[k]), **rep})
    progress(0.92, "Thicknesses")
    has_dem = np.isfinite(ground_dem) if ground_dem is not None else None
    for k in range(len(data) - 1):
        top, base = tops[k], tops[k + 1]
        th = np.where(np.isfinite(top) & np.isfinite(base), np.maximum(top - base, 0.0), np.nan)
        unit = data[k].unit
        name = f"thickness_{slug(unit)}"
        path = write_tif(out_dir / f"{name}.tif", [th], grid, [f"thickness of {unit} (m)"])
        e = {"id": name, "kind": "thickness", "unit": unit, "k": k, "colour": colours.get(unit),
             "name": t("Spessore di {0}", "Thickness of {0}").format(unit),
             "file": str(path.relative_to(chunk.data_dir)), "stats": _stats(th)}
        e.update(volumes(th, has_dem, cell_area))
        if absent[k] is not None:
            e["absent"] = absent[k] | {"area_m2": absent[k]["cells"] * cell_area}
        thick.append(e)
    registry = {"grid": grid.to_dict(), "made": time.strftime("%Y-%m-%d %H:%M"), "stale": False,
                "params": {"model": model, "cell": cell, "nugget": nugget, "mode": default, "modes": given,
                           "trend": trend, "buffer": buffer, "min_points": min_points},
                "modes": {e["unit"]: e["mode"] for e in entries},
                "order": codes, "notes": notes, "contacts": entries, "thickness": thick,
                "ground": "dem" if ground_dem is not None else ("collars" if collar_surface is not None else "none")}
    if has_dem is not None and ground is not None:
        filled = np.isfinite(ground) & ~has_dem
        if filled.any():  # the cells whose ground is the surface of the collars
            registry["without_dem"] = {"cells": int(filled.sum()), "area_m2": float(filled.sum() * cell_area)}
    if collar_report is not None:
        registry["collars"] = {k: v for k, v in collar_report.items() if k != "candidates"}
    bh.meta(chunk)["surfaces"] = registry
    try:
        variogram_figures(chunk)
    except Exception:  # noqa: BLE001 - the figures are a convenience (no matplotlib, for one)
        pass
    progress(1.0, f"{len(entries)} contacts, {len(thick)} thicknesses on {grid.width} x {grid.height} cells of {grid.cell:g} m")
    return registry


def volumes(th: np.ndarray, has_dem: np.ndarray | None, cell_area: float) -> dict:
    """The volume and the area of a thickness raster over the cells with a DEM (``has_dem``; all
    the cells without a DEM at all), the cells without one apart (``outside_dem``)."""
    finite = np.isfinite(th)
    on = finite if has_dem is None else finite & has_dem
    out = {"volume_m3": float(th[on].sum() * cell_area), "area_m2": float((on & (th > 1e-6)).sum() * cell_area)}
    if has_dem is not None:
        off = finite & ~has_dem
        if off.any():
            out["outside_dem"] = {"cells": int(off.sum()), "area_m2": float(off.sum() * cell_area),
                                  "volume_m3": float(th[off].sum() * cell_area)}
    return out


def _missing_units(pres: list, raws: list[np.ndarray], variances: list[np.ndarray], grid: Grid) -> tuple[list, list]:
    """For every unit but the last, the cells where it is missing (its top and the top of the unit
    below tied in the stack) and a report. The boreholes vote for its presence by inverse distance
    (:data:`PRESENT`; those that end above its place in the sequence do not vote); where those
    without it win, the unit is missing unless the two contacts tell a thickness the kriging can
    tell from none (more than :data:`SNAP_SIGMAS` times their combined standard deviation, and
    than :data:`SNAP_FLOOR`): a thin unit well known stays, a spurious one goes."""
    ties, absent = [], []
    for k in range(len(raws) - 1):
        pr = pres[k] if k < len(pres) else None
        both = np.isfinite(raws[k]) & np.isfinite(raws[k + 1])
        if pr is None or len(pr.ids) == 0 or pr.everywhere or not both.any():
            ties.append(None)
            absent.append(None)
            continue
        cells = np.zeros(both.shape, bool)
        if pr.present.any():
            vote = kriging.idw(pr.xy, pr.present.astype(float), grid.centres(both))
            cells[both] = vote < PRESENT
        else:
            cells = both.copy()
        spread = np.sqrt(_variance_or_typical(variances[k]) + _variance_or_typical(variances[k + 1]))
        thick = raws[k] - raws[k + 1]
        cells &= np.where(both, thick <= np.maximum(SNAP_FLOOR, SNAP_SIGMAS * spread), False)
        ties.append(cells)
        absent.append({"cells": int(cells.sum()),
                       "boreholes_without": [i for i, p in zip(pr.ids, pr.present) if not p]})
    return ties, absent


def _variance_or_typical(v: np.ndarray) -> np.ndarray:
    """The kriging variance, its median where it is unknown (inverse distance), 0 without any."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    return np.where(ok, np.maximum(v, 0.0), float(np.median(v[ok])) if ok.any() else 0.0)


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
        model_name = {"spherical": t("sferico", "spherical"), "exponential": t("esponenziale", "exponential"),
                      "gaussian": t("gaussiano", "Gaussian")}.get(v.model, v.model)
        ax.plot(h[1:], v(h[1:]), color="#1f4e79", label=f"{model_name}, {t('portata', 'range')} {num(v.range, 1)} m")
        ax.set_xlabel(t("distanza (m)", "distance (m)"))
        ax.set_ylabel(t("semivarianza (m²)", "semivariance (m²)"))
        what = t("profondità", "depths") if mode == "depth" else t("quote", "elevations")
        ax.set_title(f"{e['name']} ({what}) - LOO {num(e.get('loocv_rmse', float('nan')), 3)} m", fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = folder(chunk) / f"variogram_{e['id']}.png"
        fig.savefig(path)
        out.append(path)
    return out
