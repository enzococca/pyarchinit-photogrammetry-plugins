"""The usual processing of geophysical grids, NaN-aware: spikes removed, the stripes of the
traverses taken off (zero-mean or zero-median traverse), values clipped, a low-pass and a high-pass
(the residual after a Gaussian smoothing, which brings out the small anomalies over the geology),
the gaps filled, the grid resampled. Every filter returns a new array; :func:`apply` runs one by
its name with its parameters and records it in the grid.

The defaults of :data:`FILTERS` are the same in the window, on the command line and here: a
low-pass of sigma 1 cell (the noise of the readings, the anomalies untouched) and a high-pass of
sigma 10 cells (5 m on cells of 0.5 m: the geology goes, walls and pits stay whole)."""
from __future__ import annotations

import numpy as np

from .grids import Grid, like

FILTERS = {  # name: (description, parameters with defaults)
    "despike": ("Spikes replaced by the local median (readings farther than k times the local noise)",
                {"size": 3, "k": 4.0}),
    "destripe": ("The median (or mean) of every traverse taken off: zero-mean traverse", {"axis": "rows", "stat": "median"}),
    "clip": ("Values clipped to the percentiles (or to absolute limits)", {"low": 1.0, "high": 99.0, "absolute": False}),
    "lowpass": ("Gaussian smoothing", {"sigma": 1.0}),
    "highpass": ("The residual after a Gaussian smoothing: the small anomalies", {"sigma": 10.0}),
    "fill": ("Gaps filled from the readings around (up to a distance in cells)", {"distance": 3}),
    "resample": ("Cells of another size", {"cell": 0.25}),
}


def _nan_gaussian(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that ignores NaN (normalised by the smoothed mask)."""
    from scipy.ndimage import gaussian_filter

    ok = np.isfinite(a)
    filled = np.where(ok, a, 0.0)
    num = gaussian_filter(filled, sigma, mode="nearest")
    den = gaussian_filter(ok.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 1e-6, num / den, np.nan)
    out[~ok] = np.nan
    return out


def despike(a: np.ndarray, size: int = 3, k: float = 4.0) -> np.ndarray:
    """A reading farther than ``k`` times the noise (1.4826 x the median absolute difference from
    the local median) from the median of its ``size`` x ``size`` neighbourhood is replaced by
    that median."""
    from scipy.ndimage import generic_filter

    ok = np.isfinite(a)
    filled = np.where(ok, a, np.nan)
    med = generic_filter(filled, np.nanmedian, size=int(size), mode="nearest")
    diff = filled - med
    noise = 1.4826 * np.nanmedian(np.abs(diff[np.isfinite(diff)])) if np.isfinite(diff).any() else 0.0
    out = filled.copy()
    if noise > 0:
        spikes = ok & (np.abs(diff) > k * noise)
        out[spikes] = med[spikes]
    return out


def destripe(a: np.ndarray, axis: str = "rows", stat: str = "median") -> np.ndarray:
    """The median (or the mean) of every row (``axis="rows"``: the traverses run along x) or
    every column taken off, so that all the traverses share the zero: the stripes of a
    magnetometer walked up and down, or of a resistivity array, go."""
    ax = 1 if axis == "rows" else 0
    f = np.nanmedian if stat == "median" else np.nanmean
    with np.errstate(all="ignore"):
        level = f(a, axis=ax, keepdims=True)
    level = np.where(np.isfinite(level), level, 0.0)
    return a - level


def clip(a: np.ndarray, low: float = 1.0, high: float = 99.0, absolute: bool = False) -> np.ndarray:
    """Values clipped to the ``low`` and ``high`` percentiles, or to those values when ``absolute``."""
    ok = np.isfinite(a)
    if not ok.any():
        return a.copy()
    lo, hi = (low, high) if absolute else np.percentile(a[ok], [low, high])
    out = a.copy()
    out[ok] = np.clip(a[ok], lo, hi)
    return out


def lowpass(a: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """A Gaussian smoothing of ``sigma`` cells that skips the empty cells."""
    return _nan_gaussian(a, float(sigma))


def highpass(a: np.ndarray, sigma: float = 10.0) -> np.ndarray:
    """The residual after a Gaussian smoothing of ``sigma`` cells: the regional trend goes."""
    return a - _nan_gaussian(a, float(sigma))


def fill(a: np.ndarray, distance: int = 3) -> np.ndarray:
    """The gaps (NaN) filled by inverse-distance weighting of the readings within ``distance`` cells."""
    from rasterio.fill import fillnodata

    ok = np.isfinite(a)
    if ok.all() or not ok.any():
        return a.copy()
    out = fillnodata(a.astype(np.float32), mask=ok.astype(np.uint8), max_search_distance=int(distance),
                     smoothing_iterations=0)
    out = np.asarray(out, np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def resample_grid(grid: Grid, cell: float) -> Grid:
    """The grid with cells of ``cell`` metres (area average when coarser, bilinear when finer)."""
    import cv2

    h, w = grid.shape
    new_w = max(1, int(round(w * grid.cell / cell)))
    new_h = max(1, int(round(h * grid.cell / cell)))
    interp = cv2.INTER_AREA if cell > grid.cell else cv2.INTER_LINEAR
    bands = []
    for b in grid.values:
        ok = np.isfinite(b).astype(np.float32)
        v = cv2.resize(np.where(np.isfinite(b), b, 0.0).astype(np.float32), (new_w, new_h), interpolation=interp)
        m = cv2.resize(ok, (new_w, new_h), interpolation=interp)
        with np.errstate(invalid="ignore", divide="ignore"):
            bands.append(np.where(m > 0.5, v / m, np.nan).astype(np.float32))
    # the survey cell follows: a placed grid stays where it is, and placed again it keeps its new cell
    return like(grid, np.stack(bands), cell=float(cell), local_cell=grid.survey_cell() * float(cell) / grid.cell)


def apply(grid: Grid, name: str, **params) -> Grid:
    """``name`` applied to every band of a copy of the grid, recorded in ``filters``."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}: {', '.join(FILTERS)}")
    p = {**FILTERS[name][1], **params}
    if name == "resample":
        out = resample_grid(grid, float(p["cell"]))
    else:
        fn = {"despike": despike, "destripe": destripe, "clip": clip, "lowpass": lowpass, "highpass": highpass,
              "fill": fill}[name]
        bands = [fn(b.astype(np.float64), **p).astype(np.float32) for b in grid.values]
        out = like(grid, np.stack(bands))
    out.filters.append({"name": name, **{k: (float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else v)
                                         for k, v in p.items()}})
    return out


def from_window(name: str, values: dict) -> dict:
    """The parameters of the filter ``name`` from the values of the *Filters* window: the fields of
    that filter only (the window has one sigma for the low-pass and one for the high-pass; its clip
    is on percentiles - absolute limits are given on the command line, ``absolute=1``)."""
    if name not in FILTERS:
        raise ValueError(f"unknown filter {name!r}: {', '.join(FILTERS)}")
    if name == "lowpass":
        return {"sigma": float(values.get("sigma_low", FILTERS["lowpass"][1]["sigma"]))}
    if name == "highpass":
        return {"sigma": float(values.get("sigma_high", FILTERS["highpass"][1]["sigma"]))}
    params = {k: values[k] for k in FILTERS[name][1] if k in values}
    if name == "clip":
        params["absolute"] = False
    return params
