"""The numbers: vegetation indices of an acquisition, statistics over the area, periods of the
year, composites, the crop-mark anomaly, the chart of the series.

Crop marks are buried walls and ditches seen through the crop above them: over a wall the roots
find less water and the crop is stressed (a lower index, a *negative* mark), over a ditch they
find more and the crop is greener (a *positive* mark), in some weeks of some years. A crop-mark
map takes the index of a date or the median of a period, removes the smooth variation of the
fields (a large Gaussian, or median, background: 50-100 m) and divides what is left by its
robust spread (1.4826 x the median absolute deviation): a z-score, near 0 over even ground,
below -2 or above 2 along the marks, which are lines a few pixels wide.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np

INDICES = {  # name: (band a, band b) of (a - b) / (a + b)
    "NDVI": ("B08", "B04"),  # NIR and red, 10 m: vigour of the vegetation
    "NDRE": ("B8A", "B05"),  # narrow NIR and red edge, 20 m: chlorophyll, less saturated on dense crops
    "NDMI": ("B8A", "B11"),  # narrow NIR and short-wave infrared, 20 m: water in the leaves
}
FORMULAS = {name: f"({a} - {b}) / ({a} + {b})" for name, (a, b) in INDICES.items()}
SEASON_MONTHS = {"spring": (3, 4, 5), "summer": (6, 7, 8), "autumn": (9, 10, 11), "winter": (12, 1, 2)}
PERCENTILES = (10, 25, 75, 90)
SAMPLE = 20000  # cells of each date pooled for the monthly percentiles
BACKGROUND_M = 50.0  # scale of the background of a crop-mark map (Gaussian sigma; the median window is twice)
METHODS = ("gaussian", "median")


def normalized_difference(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """``(a - b) / (a + b)`` in float32, NaN where a value is missing, the cell is not clear
    (``mask`` False) or the sum is too small to mean anything; within -1 and 1."""
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    s = a + b
    ok = np.isfinite(a) & np.isfinite(b) & (s > 0.01) & (a > -0.02) & (b > -0.02)
    if mask is not None:
        ok &= mask
    out = np.full(a.shape, np.nan, np.float32)
    out[ok] = np.clip((a[ok] - b[ok]) / s[ok], -1.0, 1.0)
    return out


def index_of(reflectance: dict, name: str, mask: np.ndarray | None = None) -> np.ndarray:
    a, b = INDICES[name]
    return normalized_difference(reflectance[a], reflectance[b], mask)


def stats(values: np.ndarray) -> dict:
    """``cells``, ``median``, ``mean`` and the :data:`PERCENTILES` of the finite values."""
    v = values[np.isfinite(values)]
    out = {"cells": int(v.size)}
    if v.size:
        q = np.percentile(v, (50,) + PERCENTILES)
        out["median"] = float(q[0])
        out["mean"] = float(v.mean())
        for p, value in zip(PERCENTILES, q[1:]):
            out[f"p{p}"] = float(value)
    return out


def sample(values: np.ndarray, n: int = SAMPLE, seed: int = 0) -> np.ndarray:
    """At most ``n`` of the finite values, picked evenly at random (the same every time)."""
    v = values[np.isfinite(values)]
    if v.size <= n:
        return v
    return np.random.default_rng(seed).choice(v, n, replace=False)


# ------------------------------------------------------------------ periods
def check_period(period: str) -> str:
    """The period as stored (``all``, a season, ``2023-summer``, ``2023``, ``2023-06``,
    ``2023-06-14``, ``2023-04-01/2023-06-30`` or an acquisition id); ValueError otherwise."""
    p = str(period or "all").strip()
    if p in ("all",) or p in SEASON_MONTHS:
        return p
    patterns = (r"\d{4}-(spring|summer|autumn|winter)", r"\d{4}", r"\d{4}-\d{2}", r"\d{4}-\d{2}-\d{2}",
                r"\d{4}-\d{2}-\d{2}/\d{4}-\d{2}-\d{2}", r"\d{8}_S2[A-Z](_R\d{3})?")
    if any(re.fullmatch(pt, p) for pt in patterns):
        if re.fullmatch(r"\d{4}-\d{2}", p) and not 1 <= int(p[5:]) <= 12:
            raise ValueError(f"no month {p!r}")
        for part in re.findall(r"\d{4}-\d{2}-\d{2}", p):
            date.fromisoformat(part)
        return p
    raise ValueError(f"unknown period {p!r}: all, spring, 2023-summer, 2023, 2023-06, 2023-06-14, "
                     "2023-04-01/2023-06-30 or an acquisition (20230614_S2B)")


def in_period(day: date, acq_id: str, period: str) -> bool:
    """The acquisition ``acq_id`` of ``day`` belongs to ``period`` (see :func:`check_period`)."""
    p = check_period(period)
    if p == "all":
        return True
    if p == acq_id:
        return True
    if re.fullmatch(r"\d{8}_S2[A-Z](_R\d{3})?", p):
        return False
    if p in SEASON_MONTHS:
        return day.month in SEASON_MONTHS[p]
    m = re.fullmatch(r"(\d{4})-(spring|summer|autumn|winter)", p)
    if m:
        year, season = int(m.group(1)), m.group(2)
        if season == "winter":  # the winter that ends in that year: December before, January, February
            return (day.month == 12 and day.year == year - 1) or (day.month in (1, 2) and day.year == year)
        return day.year == year and day.month in SEASON_MONTHS[season]
    if re.fullmatch(r"\d{4}", p):
        return day.year == int(p)
    if re.fullmatch(r"\d{4}-\d{2}", p):
        return (day.year, day.month) == (int(p[:4]), int(p[5:]))
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", p):
        return day == date.fromisoformat(p)
    a, b = p.split("/")
    return date.fromisoformat(a) <= day <= date.fromisoformat(b)


def periods_of(days: list[date]) -> list[str]:
    """The periods worth offering for these dates: the seasons of each year with dates, every
    season over all years, every year, the whole period."""
    out = []
    years = sorted({d.year for d in days})
    for y in years + ([years[-1] + 1] if years else []):  # a December belongs to the winter of the next year
        for s in ("winter", "spring", "summer", "autumn"):
            if any(in_period(d, "", f"{y}-{s}") for d in days):
                out.append(f"{y}-{s}")
    if len(years) > 1:
        out += [s for s in SEASON_MONTHS if any(d.month in SEASON_MONTHS[s] for d in days)]
        out += [str(y) for y in years]
    out.append("all")
    return out


# ------------------------------------------------------------------ composites and anomalies
def median_composite(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The median of the dates of ``stack`` (dates, rows, cols) cell by cell, and how many dates
    had a value there."""
    import warnings

    count = np.isfinite(stack).sum(axis=0).astype(np.int16)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # cells with no date: NaN, as they should
        med = np.nanmedian(stack, axis=0).astype(np.float32) if len(stack) else np.full(stack.shape[1:], np.nan, np.float32)
    return med, count


def background(values: np.ndarray, cell: float, background_m: float = BACKGROUND_M, method: str = "gaussian") -> np.ndarray:
    """The smooth variation of the fields under ``values`` (NaN where nothing): a Gaussian of
    sigma ``background_m`` that ignores the empty cells (normalized convolution), or the median
    of a window of side twice ``background_m``."""
    from scipy import ndimage

    if method not in METHODS:
        raise ValueError(f"method: {', '.join(METHODS)}")
    valid = np.isfinite(values)
    sigma = max(float(background_m) / float(cell), 0.5)
    weight = ndimage.gaussian_filter(valid.astype(np.float64), sigma, mode="constant", cval=0.0)
    total = ndimage.gaussian_filter(np.where(valid, values, 0.0).astype(np.float64), sigma, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        bg = np.where(weight > 0.05, total / np.maximum(weight, 1e-12), np.nan)
    if method == "median":
        seed = np.where(valid, values, bg)
        fill = np.nanmedian(values) if valid.any() else 0.0
        seed = np.where(np.isfinite(seed), seed, fill)
        size = 2 * int(round(sigma)) + 1  # a window of side twice the background scale
        bg = np.where(np.isfinite(bg), ndimage.median_filter(seed, size=size, mode="nearest"), np.nan)
    return bg.astype(np.float32)


def anomaly(values: np.ndarray, cell: float, background_m: float = BACKGROUND_M, method: str = "gaussian") -> tuple[np.ndarray, dict]:
    """The crop-mark z-score of ``values``: ``(values - background - m) / s`` with ``m`` the
    median and ``s`` the robust spread (1.4826 MAD) of the residuals; NaN where no value."""
    bg = background(values, cell, background_m, method)
    resid = values.astype(np.float64) - bg
    r = resid[np.isfinite(resid)]
    if r.size == 0:
        return np.full(values.shape, np.nan, np.float32), {"median": 0.0, "spread": 0.0}
    med = float(np.median(r))
    spread = 1.4826 * float(np.median(np.abs(r - med)))
    if not spread > 1e-9:
        spread = float(np.std(r)) or 1e-9
    z = ((resid - med) / spread).astype(np.float32)
    return z, {"median": med, "spread": spread}


# ------------------------------------------------------------------ the series
def monthly(rows_by_month: dict) -> list[dict]:
    """The monthly rows: ``rows_by_month[(month, index)] = (dates, pooled values)``."""
    out = []
    for (month, name), (dates, values) in sorted(rows_by_month.items()):
        v = np.concatenate(values) if values else np.zeros(0, np.float32)
        s = stats(v)
        out.append({"month": month, "index": name, "dates": len(dates), **s})
    return out


def chart(rows: list[dict], months: list[dict], path, title: str, attribution: str, language: str = "it",
          names=tuple(INDICES)) -> None:
    """The series as a PNG: for each index the median of the area at each date (dots), the
    25-75 (thick) and 10-90 (thin) percentiles as bars, the monthly median as a line; the
    attribution of the Copernicus data at the foot (and in the PNG's metadata)."""
    from datetime import datetime

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    en = language == "en"
    colours = {"NDVI": "#1a9850", "NDRE": "#b8860b", "NDMI": "#2c7fb8"}
    fig = Figure(figsize=(9.0, 1.9 * len(names) + 0.9), dpi=110)
    FigureCanvasAgg(fig)
    axes = fig.subplots(len(names), 1, sharex=True, squeeze=False)[:, 0]
    for ax, name in zip(axes, names):
        r = [row for row in rows if row["index"] == name and row.get("cells")]
        colour = colours.get(name, "#444444")
        if r:
            days = [datetime.fromisoformat(row["date"]) for row in r]
            ax.vlines(days, [row["p10"] for row in r], [row["p90"] for row in r], color=colour, lw=0.8, alpha=0.35)
            ax.vlines(days, [row["p25"] for row in r], [row["p75"] for row in r], color=colour, lw=3.0, alpha=0.55)
            ax.plot(days, [row["median"] for row in r], "o", color=colour, ms=3.5)
        m = [row for row in months if row["index"] == name and row.get("cells")]
        if m:
            mdays = [datetime.fromisoformat(row["month"] + "-15") for row in m]
            ax.plot(mdays, [row["median"] for row in m], "-", color=colour, lw=1.0, alpha=0.8)
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("date" if en else "data")
    fig.suptitle(title, fontsize=11)
    fig.text(0.01, 0.005, attribution, fontsize=7, color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    fig.savefig(str(path), format="png", metadata={"Copyright": attribution, "Title": title,
                                                    "Software": "pyarchinit-photogrammetry sentinel2"})
