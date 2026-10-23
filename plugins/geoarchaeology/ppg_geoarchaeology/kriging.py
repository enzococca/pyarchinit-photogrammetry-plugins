"""Kriging written in NumPy and SciPy: the surfaces between the boreholes.

* :func:`empirical` - the experimental semivariogram: half the mean squared difference of the
  pairs of points, by classes of distance up to half the largest distance.
* :func:`fit` - a spherical, exponential or Gaussian model with its nugget, fitted to it by
  weighted least squares with Cressie's weights (the pairs of a class over the model value
  squared: the classes with many pairs and the short distances count most).
* :class:`Kriging` - ordinary kriging (an unknown constant mean) or universal kriging with a
  linear drift (an unknown plane: a contact that dips across the site), solved together for up to
  :data:`MAX_GLOBAL` points; beyond, the nearest :data:`NEIGHBOURS` points for each cell, the
  plane removed first. The kriging variance comes with every estimate; the leave-one-out errors at
  the data points come from one inverse of the kriging matrix (Dubrule 1983).
* :func:`idw` - inverse distance weighting, for fewer points than a variogram needs.
* :func:`interpolate` - all of it: the model and the drift chosen by the leave-one-out error
  when asked (``"auto"``), the estimate and the variance at the targets, a report.

When asked to choose (``"auto"``), the least leave-one-out error does not win by itself: a
handful of boreholes cannot tell the models apart by a few per cent, and the Gaussian model
that wins such a close race overshoots between and around the boreholes. Among the models
within :data:`CLEAR_MARGIN` of the least error the steadiest is taken (:data:`PREFERENCE`:
spherical, exponential, without a drift before with one, Gaussian last); every model tried is
reported with its error.

The models use the practical range: the spherical model reaches its sill at ``range``, the
exponential and Gaussian ones 95 % of it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MODELS = ("spherical", "exponential", "gaussian")
MAX_GLOBAL = 400  # data points of one kriging system; more: a moving neighbourhood
NEIGHBOURS = 32
BLOCK = 4_000_000  # numbers of the right-hand sides solved at once (memory bound)
CLEAR_MARGIN = 0.05  # another model or a drift is taken only when its leave-one-out error is smaller by this much
# the order of preference among the candidates within the margin: (model, drift)
PREFERENCE = (("spherical", 0), ("exponential", 0), ("spherical", 1), ("exponential", 1), ("gaussian", 0),
              ("gaussian", 1))


def semivariance(model: str, h, nugget: float, sill: float, rng: float) -> np.ndarray:
    """The model at the distances ``h``: 0 at 0, the nugget plus the structured part beyond."""
    h = np.asarray(h, float)
    r = h / max(float(rng), 1e-12)
    if model == "spherical":
        s = np.where(r < 1.0, 1.5 * r - 0.5 * r ** 3, 1.0)
    elif model == "exponential":
        s = 1.0 - np.exp(-3.0 * r)
    elif model == "gaussian":
        s = 1.0 - np.exp(-3.0 * r * r)
    else:
        raise ValueError(f"unknown variogram model {model!r}: " + ", ".join(MODELS))
    return np.where(h > 0, nugget + sill * s, 0.0)


@dataclass
class Variogram:
    model: str
    nugget: float
    sill: float  # the partial sill: what the structured part adds to the nugget
    range: float

    def __call__(self, h) -> np.ndarray:
        return semivariance(self.model, h, self.nugget, self.sill, self.range)

    def to_dict(self) -> dict:
        return {"model": self.model, "nugget": float(self.nugget), "sill": float(self.sill), "range": float(self.range)}


def empirical(xy: np.ndarray, z: np.ndarray, n_lags: int | None = None, max_lag: float | None = None):
    """``(lags, gamma, pairs)`` of the classes with pairs: the mean distance, half the mean
    squared difference and the number of pairs of each class, up to ``max_lag`` (half the
    largest distance by default)."""
    from scipy.spatial.distance import pdist

    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    d = pdist(xy)
    g = 0.5 * pdist(z.reshape(-1, 1), "sqeuclidean")
    if d.size == 0:
        raise ValueError("a variogram needs two points at least")
    max_lag = float(max_lag) if max_lag else float(d.max()) / 2.0
    sel = d <= max_lag
    if sel.sum() < 3:
        sel, max_lag = np.ones_like(d, bool), float(d.max())
    n_lags = int(n_lags or np.clip(np.sqrt(sel.sum()), 4, 15))
    k = np.clip((d[sel] / max(max_lag, 1e-12) * n_lags).astype(np.int64), 0, n_lags - 1)
    pairs = np.bincount(k, minlength=n_lags)
    lag = np.bincount(k, weights=d[sel], minlength=n_lags)
    gam = np.bincount(k, weights=g[sel], minlength=n_lags)
    ok = pairs > 0
    return lag[ok] / pairs[ok], gam[ok] / pairs[ok], pairs[ok]


def fit(lags, gamma, pairs, model: str = "spherical", nugget: float | None = None) -> Variogram:
    """The model fitted to the experimental variogram by weighted least squares (Cressie 1985:
    the pairs of a class times the relative misfit squared); ``nugget`` fixed when given."""
    from scipy.optimize import least_squares

    lags, gamma, pairs = (np.asarray(a, float) for a in (lags, gamma, pairs))
    gmax = max(float(gamma.max()), 1e-12)
    hmax = max(float(lags.max()), 1e-9)
    n0 = 0.0
    if nugget is None and len(gamma) >= 2 and lags[1] > lags[0]:
        n0 = float(np.clip(gamma[0] - (gamma[1] - gamma[0]) * lags[0] / (lags[1] - lags[0]), 0.0, 0.5 * gmax))
    elif nugget is not None:
        n0 = float(nugget)
    s0 = max(gmax - n0, 1e-3 * gmax)
    reach = np.nonzero(gamma >= 0.9 * gmax)[0]
    r0 = float(np.clip(lags[reach[0]] if reach.size else hmax, 0.05 * hmax, 2.0 * hmax))
    w = np.sqrt(pairs)

    def unpack(p):
        return (p[0], p[1], p[2]) if nugget is None else (float(nugget), p[0], p[1])

    def relative(p):
        nug, sill, rng = unpack(p)
        m = np.maximum(semivariance(model, lags, nug, sill, rng), 1e-9 * gmax)
        return w * (gamma / m - 1.0)

    def absolute(p):
        nug, sill, rng = unpack(p)
        return w * (semivariance(model, lags, nug, sill, rng) - gamma) / gmax

    lo = [1e-6 * gmax, 1e-3 * hmax]
    hi = [10.0 * gmax, 20.0 * hmax]
    x0 = [s0, r0]
    if nugget is None:
        lo, hi, x0 = [0.0] + lo, [gmax] + hi, [n0] + x0
    x0 = np.clip(x0, np.asarray(lo) + 1e-12, np.asarray(hi) - 1e-12)
    best = None
    for fun in (relative, absolute):
        try:
            r = least_squares(fun, x0, bounds=(lo, hi), method="trf")
        except (ValueError, FloatingPointError):
            continue
        if np.all(np.isfinite(r.x)):
            best = r.x
            break
    if best is None:
        raise ValueError("the variogram model could not be fitted")
    nug, sill, rng = unpack(best)
    if model == "gaussian":  # without a nugget its kriging matrix is close to singular
        nug = max(nug, 1e-6 * (nug + sill))
    return Variogram(model, float(nug), float(sill), float(rng))


def _drift(xy: np.ndarray, order: int) -> np.ndarray:
    return np.ones((len(xy), 1)) if order == 0 else np.column_stack([np.ones(len(xy)), xy[:, 0], xy[:, 1]])


def plane(xy: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Least-squares plane ``(a, b, c)``: z = a + b x + c y."""
    A = _drift(np.asarray(xy, float), 1)
    return np.linalg.lstsq(A, np.asarray(z, float), rcond=None)[0]


def spans_a_plane(xy: np.ndarray) -> bool:
    """Whether the points are not all on a line (a linear drift can be estimated)."""
    xy = np.asarray(xy, float)
    if len(xy) < 3:
        return False
    c = xy - xy.mean(0)
    s = np.linalg.svd(c, compute_uv=False)
    return bool(s[-1] > 1e-6 * max(s[0], 1e-12))


class Kriging:
    """Kriging of ``z`` at ``xy`` with ``variogram``; ``drift`` 0 (ordinary) or 1 (a linear
    drift, universal kriging). Coordinates are taken relative to their centre."""

    def __init__(self, xy, z, variogram: Variogram, drift: int = 0, neighbours: int = NEIGHBOURS):
        from scipy.linalg import lu_factor, lu_solve
        from scipy.spatial.distance import cdist

        self.v = variogram
        self.drift = int(drift)
        xy = np.asarray(xy, float)
        self.centre = xy.mean(0)
        self.scale = max(float(np.ptp(xy, axis=0).max()), 1.0)
        self.xy = xy - self.centre
        self.z = np.asarray(z, float)
        self.n = len(self.z)
        self.p = 1 if self.drift == 0 else 3
        self.neighbours = int(neighbours)
        self.local = self.n > MAX_GLOBAL
        if self.local:  # the plane removed first, the residuals kriged in a moving neighbourhood
            from scipy.spatial import cKDTree

            self.coef = plane(self.xy / self.scale, self.z) if self.drift else np.array([self.z.mean(), 0, 0])
            self.res = self.z - _drift(self.xy / self.scale, 1) @ self.coef
            self.tree = cKDTree(self.xy)
            return
        n, p = self.n, self.p
        M = np.zeros((n + p, n + p))
        M[:n, :n] = self.v(cdist(self.xy, self.xy))
        F = _drift(self.xy / self.scale, self.drift)
        M[:n, n:] = F
        M[n:, :n] = F.T
        self.M = M
        self.lu = lu_factor(M, check_finite=False)
        self.alpha = lu_solve(self.lu, np.concatenate([self.z, np.zeros(p)]), check_finite=False)
        self._lu_solve = lu_solve

    def loocv(self) -> np.ndarray:
        """The leave-one-out errors (datum minus the estimate from the others) from the inverse of
        the kriging matrix: e_i = (M⁻¹b)_i / (M⁻¹)_ii (Dubrule 1983)."""
        if self.local:
            raise RuntimeError("leave-one-out on a subset: build a Kriging of at most MAX_GLOBAL points")
        inv = self._lu_solve(self.lu, np.eye(self.n + self.p), check_finite=False)
        return self.alpha[:self.n] / np.diag(inv)[:self.n]

    def predict(self, targets, variance: bool = True, progress=None) -> tuple[np.ndarray, np.ndarray | None]:
        """Estimate (and kriging variance, m²) at the ``targets`` points (world coordinates)."""
        from scipy.spatial.distance import cdist

        t = np.asarray(targets, float).reshape(-1, 2) - self.centre
        m = len(t)
        est = np.empty(m)
        var = np.empty(m) if variance else None
        if self.local:
            return self._predict_local(t, variance, progress)
        step = max(256, BLOCK // (self.n + self.p))
        for s in range(0, m, step):
            tt = t[s:s + step]
            g0 = self.v(cdist(self.xy, tt))
            f0 = _drift(tt / self.scale, self.drift).T
            rhs = np.vstack([g0, f0])
            if not variance:
                est[s:s + step] = rhs.T @ self.alpha
            else:
                sol = self._lu_solve(self.lu, rhs, check_finite=False)
                lam, mu = sol[:self.n], sol[self.n:]
                est[s:s + step] = self.z @ lam
                var[s:s + step] = np.maximum((lam * g0).sum(0) + (mu * f0).sum(0), 0.0)
            if progress is not None:
                progress(min(1.0, (s + step) / max(m, 1)), "")
        return est, var

    def _predict_local(self, t, variance, progress):
        k = min(self.neighbours, self.n)
        m = len(t)
        est = np.empty(m)
        var = np.empty(m) if variance else None
        step = max(64, BLOCK // ((k + 1) ** 2))
        for s in range(0, m, step):
            tt = t[s:s + step]
            d, idx = self.tree.query(tt, k)
            d, idx = d.reshape(len(tt), k), idx.reshape(len(tt), k)
            P = self.xy[idx]
            D = np.linalg.norm(P[:, :, None, :] - P[:, None, :, :], axis=-1)
            M = np.ones((len(tt), k + 1, k + 1))
            M[:, :k, :k] = self.v(D)
            M[:, k, k] = 0.0
            rhs = np.ones((len(tt), k + 1))
            rhs[:, :k] = self.v(d)
            try:
                sol = np.linalg.solve(M, rhs[..., None])[..., 0]
            except np.linalg.LinAlgError:
                sol = np.stack([np.linalg.lstsq(M[i], rhs[i], rcond=None)[0] for i in range(len(tt))])
            lam = sol[:, :k]
            trend = _drift(tt / self.scale, 1) @ self.coef
            est[s:s + step] = trend + (lam * self.res[idx]).sum(1)
            if variance:
                var[s:s + step] = np.maximum((lam * rhs[:, :k]).sum(1) + sol[:, k], 0.0)
            if progress is not None:
                progress(min(1.0, (s + step) / max(m, 1)), "")
        return est, var


def idw(xy, z, targets, power: float = 2.0) -> np.ndarray:
    """Inverse distance weighting of ``z`` at ``targets`` (the datum itself on a data point)."""
    from scipy.spatial.distance import cdist

    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    t = np.asarray(targets, float).reshape(-1, 2)
    out = np.empty(len(t))
    step = max(256, BLOCK // max(len(z), 1))
    for s in range(0, len(t), step):
        d = cdist(t[s:s + step], xy)
        exact = d < 1e-9
        with np.errstate(divide="ignore"):
            w = 1.0 / np.maximum(d, 1e-12) ** power
        w[exact.any(1)] = exact[exact.any(1)].astype(float)
        out[s:s + step] = (w @ z) / w.sum(1)
    return out


def _merge_duplicates(xy: np.ndarray, z: np.ndarray, tol: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Points closer than ``tol`` merged (their mean value): a kriging matrix with two equal rows
    is singular."""
    from scipy.spatial import cKDTree

    if len(z) < 2:
        return xy, z, 0
    pairs = cKDTree(xy).query_pairs(tol, output_type="ndarray")
    if len(pairs) == 0:
        return xy, z, 0
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    n = len(z)
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    _k, label = connected_components(g, directed=False)
    counts = np.bincount(label)
    mx = np.bincount(label, weights=xy[:, 0]) / counts
    my = np.bincount(label, weights=xy[:, 1]) / counts
    mz = np.bincount(label, weights=z) / counts
    return np.column_stack([mx, my]), mz, int(n - len(counts))


def preferred(candidates: list[dict], margin: float = CLEAR_MARGIN) -> dict | None:
    """The candidate to use among those tried (``{"model", "trend", "loocv_rmse"}``): the ones
    within ``margin`` of the least leave-one-out error are as good as it, and among them the
    first in :data:`PREFERENCE` is taken."""
    if not candidates:
        return None
    best = min(float(c["loocv_rmse"]) for c in candidates)
    eligible = [c for c in candidates if float(c["loocv_rmse"]) <= best * (1.0 + margin) + 1e-9]

    def rank(c):
        key = (c["model"], 1 if c["trend"] == "linear" else 0)
        return PREFERENCE.index(key) if key in PREFERENCE else len(PREFERENCE)

    return min(eligible, key=lambda c: (rank(c), float(c["loocv_rmse"])))


def choose(xy, z, model: str = "auto", nugget: float | None = None, trend: str = "auto",
           rng: np.random.Generator | None = None) -> tuple[Variogram, int, list[dict], np.ndarray]:
    """The variogram model and the drift (0 none, 1 linear) among those asked (``model`` one of
    :data:`MODELS` or "auto", ``trend`` "none", "linear" or "auto"): the steadiest of those within
    :data:`CLEAR_MARGIN` of the least leave-one-out error (:func:`preferred`); with more than
    :data:`MAX_GLOBAL` points on a random subset of them. Returns the variogram, the drift, the
    candidates tried (each with its error, the one taken marked ``chosen``) and the
    leave-one-out errors of the one taken."""
    xy = np.asarray(xy, float)
    z = np.asarray(z, float)
    if len(z) > MAX_GLOBAL:
        rng = rng or np.random.default_rng(0)
        pick = np.sort(rng.choice(len(z), MAX_GLOBAL, replace=False))
        xy, z = xy[pick], z[pick]
    models = MODELS if model == "auto" else (model,)
    if trend == "linear":
        drifts = (1,) if spans_a_plane(xy) else (0,)
    elif trend == "none":
        drifts = (0,)
    else:
        drifts = (0, 1) if len(z) >= 6 and spans_a_plane(xy) else (0,)
    centre = xy.mean(0)
    scale = max(float(np.ptp(xy, axis=0).max()), 1.0)
    tried, fitted = [], {}
    for drift in drifts:
        if drift:
            coef = plane((xy - centre) / scale, z)
            resid = z - _drift((xy - centre) / scale, 1) @ coef
        else:
            resid = z - z.mean()
        exact = drift and np.ptp(resid) <= 1e-9 * max(1.0, float(np.ptp(z)))
        try:
            lags, gam, pairs = empirical(xy, resid)
        except ValueError:
            continue
        for m in models:
            try:
                if exact:  # the plane is the whole surface: any variogram gives it back
                    v = Variogram(m, 0.0, 1.0, float(lags.max()) * 2.0 or 1.0)
                else:
                    v = fit(lags, gam, pairs, m, nugget)
                errors = Kriging(xy, z, v, drift).loocv()
            except (ValueError, np.linalg.LinAlgError, RuntimeError):
                continue
            if not np.all(np.isfinite(errors)):
                continue
            rmse = float(np.sqrt(np.mean(errors ** 2)))
            tried.append({"model": m, "trend": "linear" if drift else "none", "loocv_rmse": rmse, **v.to_dict(),
                          "exact": bool(exact)})
            fitted[(m, drift)] = (v, errors)
    if not tried:
        raise ValueError("no variogram model could be fitted to these points")
    pick = preferred(tried)
    for c in tried:
        c["chosen"] = c is pick
    drift = 1 if pick["trend"] == "linear" else 0
    v, errors = fitted[(pick["model"], drift)]
    return v, drift, tried, errors


def interpolate(xy, z, targets, model: str = "auto", nugget: float | None = None, trend: str = "auto",
                min_points: int = 5, variance: bool = True, progress=None) -> tuple[np.ndarray, np.ndarray, dict]:
    """``z`` known at ``xy`` estimated at ``targets``: ``(estimate, kriging variance, report)``.
    Kriging with the model and drift asked (or the best by leave-one-out); with fewer than
    ``min_points`` points (or when no model fits) inverse distance weighting, the variance NaN."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    z = np.asarray(z, float)
    keep = np.isfinite(z) & np.isfinite(xy).all(1)
    xy, z = xy[keep], z[keep]
    if len(z) == 0:
        raise ValueError("no point to interpolate")
    span = float(np.ptp(xy, axis=0).max()) if len(z) > 1 else 0.0
    xy, z, merged = _merge_duplicates(xy, z, max(1e-6 * span, 1e-6))
    targets = np.asarray(targets, float).reshape(-1, 2)
    report = {"points": int(len(z)), "merged": merged}
    if np.ptp(z) < 1e-9:
        report.update(method="constant")
        return np.full(len(targets), float(z[0])), np.zeros(len(targets)), report
    if len(z) < max(int(min_points), 3):
        report.update(method="idw", power=2.0)
        return idw(xy, z, targets), np.full(len(targets), np.nan), report
    try:
        v, drift, tried, errors = choose(xy, z, model, nugget, trend)
    except ValueError as exc:
        report.update(method="idw", power=2.0, note=str(exc))
        return idw(xy, z, targets), np.full(len(targets), np.nan), report
    est, var = Kriging(xy, z, v, drift).predict(targets, variance=variance, progress=progress)
    if var is not None and any(c.get("chosen") and c.get("exact") for c in tried):
        var = np.zeros(len(targets))  # the plane is the whole surface: known exactly (the sill was a stand-in)
    least = min(tried, key=lambda c: c["loocv_rmse"])
    report.update(method="kriging", variogram=v.to_dict(), trend="linear" if drift else "none",
                  loocv_rmse=float(np.sqrt(np.mean(errors ** 2))), loocv_mae=float(np.mean(np.abs(errors))),
                  loocv_max=float(np.max(np.abs(errors))), candidates=tried,
                  least_error={"model": least["model"], "trend": least["trend"], "loocv_rmse": least["loocv_rmse"]},
                  margin=CLEAR_MARGIN, neighbourhood="moving" if len(z) > MAX_GLOBAL else "global")
    return est, (var if var is not None else np.full(len(targets), np.nan)), report
