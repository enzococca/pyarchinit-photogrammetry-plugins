"""The chunk levelled and scaled by the depths of its cameras.

Under water there is no GPS: the depth of every photo - from the pressure sensor of an ROV or a dive
computer, matched by time (:mod:`.depthlog`) - is the only absolute measure. The depths give three
things at once, as the program's camera references give them with GPS:

* the vertical (gravity): the direction along which the depth of the cameras changes;
* the scale: how many metres of depth a unit of the alignment is;
* the height of everything: ``Z = -depth`` (the surface of the water at 0, up positive).

The program's camera references are positions in three coordinates, each with one accuracy for
x, y and z together: a reference with only Z (and X, Y unknown) is not one they can hold. So the
levelling is done here, in a frame of its own: with ``c`` the centre of a camera in the internal
frame of the chunk and ``d`` its depth, the depths ask ``v . c + b = -d`` for every camera, linear in
``v`` (3) and ``b``; ``|v|`` is the scale and ``v / |v|`` the vertical. Solved by least squares with
Huber weights (a few wrong matches of time do not tilt it), the residual of every camera kept.
X, Y and the heading stay arbitrary - the heading is kept from the chunk as it was - unless known
positions (USBL, a few cameras placed by hand) give them: then a horizontal similarity (rotation
about the vertical, translation, optionally the scale) puts the levelled chunk on them, in their CRS.

When the depth changes too little for the scale (a survey at a constant depth: the cameras on a
horizontal plane), the scale must come from elsewhere - the control scale bars of the chunk, or a
number - and the vertical is then the normal of the plane of the cameras, the side that the cameras
look away from being up. A single straight line of cameras leaves the vertical turned about the line
undetermined: the mean direction the cameras look in (down, in a survey) decides it.
"""
from __future__ import annotations

import math

import numpy as np

HUBER = 1.345
WEAK_SCALE = 0.05  # relative standard error of the scale above which the depths do not give it


def _huber(r: np.ndarray, sigma: float) -> np.ndarray:
    a = np.abs(r) / max(sigma, 1e-12)
    return np.where(a <= HUBER, 1.0, HUBER / np.maximum(a, 1e-12))


def _robust_sigma(r: np.ndarray, w: np.ndarray | None = None) -> float:
    if not len(r):
        return 0.0
    med = float(np.median(r))
    return float(1.4826 * np.median(np.abs(r - med))) or float(np.std(r)) or 1e-9


def solve(centres, depths, weights=None, scale: float | None = None, down=None, iterations: int = 10) -> dict:
    """Vertical, scale and offset from camera ``centres`` (N, 3, internal frame) and their ``depths``
    (metres, positive down). ``scale``: metres per internal unit when known (scale bars); else from
    the depths. ``down``: a guess of the downward direction in the internal frame (the mean viewing
    direction of the cameras) for the parts of the vertical the cameras do not decide.

    Returns ``{"up", "scale", "offset", "residuals", "rms", "sigma", "scale_std", "tilt_std",
    "inliers", "observable", "scale_from"}``: ``up`` the unit vertical in the internal frame, the
    height of a point ``p`` is ``scale * up . p + offset``."""
    c = np.asarray(centres, float).reshape(-1, 3)
    d = np.asarray(depths, float).reshape(-1)
    n = len(c)
    if n < 3:
        raise ValueError("at least three cameras with a depth are needed")
    w0 = np.ones(n) if weights is None else np.asarray(weights, float)
    y = -d
    w = w0.copy()
    mean = w @ c / w.sum()
    x = c - mean
    # the directions the cameras move along (their spread), to know what the depths can decide
    _, sv, vt = np.linalg.svd(x * np.sqrt(w0)[:, None], full_matrices=False)
    observable = sv > max(sv[0], 1e-12) * 1e-3
    rank = int(observable.sum())
    v = np.zeros(3)
    b = float(w @ y / w.sum())
    scale_from = "depths"
    for _ in range(iterations):
        sw = np.sqrt(w)
        a = np.column_stack([x, np.ones(n)]) * sw[:, None]
        sol, *_ = np.linalg.lstsq(a, y * sw, rcond=1e-6)
        v, b = sol[:3], float(sol[3])
        r = x @ v + b - y
        sigma = _robust_sigma(r)
        w = w0 * _huber(r, sigma)
    # observable part of v: the projection on the directions the cameras span
    basis = vt[:rank]
    v_obs = basis.T @ (basis @ v)
    a = np.column_stack([x, np.ones(n)]) * np.sqrt(w)[:, None]
    r = x @ v + b - y
    sigma = _robust_sigma(r)
    try:
        cov = sigma ** 2 * np.linalg.pinv(a.T @ a, rcond=1e-9)[:3, :3]
    except np.linalg.LinAlgError:
        cov = np.full((3, 3), np.inf)
    s_est = float(np.linalg.norm(v_obs))
    full = rank == 3
    scale_std = math.inf
    if full and s_est > 0:
        u_est = v_obs / s_est
        scale_std = float(math.sqrt(max(u_est @ cov @ u_est, 0.0)))
    weak = not full or s_est <= 0 or scale_std / max(s_est, 1e-12) > WEAK_SCALE
    if scale is None and weak:
        raise ValueError(t_("La profondità delle camere cambia troppo poco per dare la scala (errore "
                            "relativo {0}): serve una barra di scala, o una scala data.",
                            "The depth of the cameras changes too little to give the scale (relative error {0}): "
                            "a scale bar, or a given scale, is needed.").format(
            "inf" if not math.isfinite(scale_std) or s_est <= 0 else f"{scale_std / s_est:.0%}"))
    if scale is None:
        s = s_est
        up = v_obs / s
    else:
        s = float(scale)
        scale_from = "given"
        u_obs = v_obs / s
        m = float(np.linalg.norm(u_obs))
        if m >= 1.0:
            up = u_obs / m
        else:
            # what the cameras do not decide: from the downward guess, else from the null space alone
            null = vt[rank:] if rank < 3 else np.zeros((0, 3))
            if len(null) == 0:
                up = u_obs / max(m, 1e-12)
            else:
                guess = -np.asarray(down, float) if down is not None else null[0]
                g = null.T @ (null @ guess)
                if np.linalg.norm(g) < 1e-9:
                    g = null[0]
                g = g / np.linalg.norm(g)
                up = u_obs + math.sqrt(max(0.0, 1.0 - m * m)) * g
                up = up / np.linalg.norm(up)
    if down is not None and rank < 3 and float(np.dot(up, down)) > 0 and scale is not None:
        up = -up  # the cameras look down
    # offset again with the vertical chosen (the scale fixed): weighted median-free mean of the inliers
    h = s * (x @ up)
    r = h + b - y
    sigma = _robust_sigma(r)
    w = w0 * _huber(r, sigma)
    b = float(np.sum(w * (y - h)) / np.sum(w))
    r = h + b - y
    sigma = _robust_sigma(r)
    inliers = np.abs(r) <= max(4.0 * sigma, 1e-9)
    perp = np.eye(3) - np.outer(up, up)
    tilt_cov = perp @ cov @ perp
    tilt_std = float(math.sqrt(max(np.linalg.eigvalsh(tilt_cov).max(), 0.0)) / s) if np.all(np.isfinite(tilt_cov)) \
        else math.inf
    if rank < 3:
        tilt_std = math.inf if scale is None else tilt_std
    return {"up": up, "scale": float(s), "offset": float(b - s * float(up @ mean)), "residuals": r,
            "rms": float(np.sqrt(np.mean(r[inliers] ** 2))) if inliers.any() else float("nan"),
            "sigma": float(sigma), "scale_std": float(scale_std) if scale is None else 0.0,
            "tilt_std": tilt_std, "inliers": inliers, "observable": rank, "scale_from": scale_from,
            "depth_range": float(np.ptp(d))}


def t_(it: str, en: str) -> str:
    from .texts import t

    return t(it, en)


def _rotation(up: np.ndarray, heading_hint: np.ndarray) -> np.ndarray:
    """Rows: world X, Y, Z axes in the internal frame; Z = ``up``, X the hint made horizontal."""
    z = up / np.linalg.norm(up)
    xh = heading_hint - np.dot(heading_hint, z) * z
    if np.linalg.norm(xh) < 1e-6:
        alt = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        xh = alt - np.dot(alt, z) * z
    xh = xh / np.linalg.norm(xh)
    yh = np.cross(z, xh)
    return np.vstack([xh, yh, z])


def similarity_2d(src, dst, weights=None, with_scale: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """``(scale, R (2x2), t)`` with ``dst ~ scale * R @ src + t`` (weighted Umeyama in the plane)."""
    src, dst = np.asarray(src, float), np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    ms, md = w @ src, w @ dst
    xs, xd = src - ms, dst - md
    cov = (xd * w[:, None]).T @ xs
    u, sv, vt = np.linalg.svd(cov)
    sign = np.diag([1.0, np.sign(np.linalg.det(u @ vt)) or 1.0])
    rot = u @ sign @ vt
    var = float(np.sum(w * np.sum(xs ** 2, axis=1)))
    s = float(np.trace(np.diag(sv) @ sign) / var) if with_scale and var > 0 else 1.0
    return s, rot, md - s * rot @ ms


def camera_depths(chunk) -> dict[int, float]:
    """The depths of the cameras of ``chunk`` (``camera.meta["underwater"]["depth"]``)."""
    out = {}
    for cam in chunk.cameras:
        d = (cam.meta.get("underwater") or {}).get("depth")
        if d is not None and np.isfinite(d):
            out[cam.key] = float(d)
    return out


def bar_scale(chunk) -> float | None:
    """Metres per internal unit given by the control scale bars of the chunk (None without them)."""
    try:
        from pyarchinit_photogrammetry.markers import gcp

        chunk.triangulate_markers()
        return gcp._bar_scale(chunk, gcp.control_scalebars(chunk))
    except Exception:  # noqa: BLE001 - no bars, an older program
        return None


def mean_view_direction(cams) -> np.ndarray | None:
    dirs = [np.asarray(c.transform)[:3, 2] for c in cams if c.transform is not None]
    if not dirs:
        return None
    m = np.mean(dirs, axis=0)
    return m / np.linalg.norm(m) if np.linalg.norm(m) > 1e-9 else None


def level(chunk, scale: float | str | None = None, positions: dict | None = None, positions_crs: str | None = None,
          position_scale: bool = False, apply: bool = True) -> dict:
    """Level, scale and place ``chunk`` by the depths of its aligned cameras (see the module).

    ``scale``: ``None`` from the depths (the control scale bars when the depths cannot give it),
    ``"bars"`` from the scale bars, a number (metres per internal unit) as given. ``positions``:
    ``{camera key: (x, y)}`` known positions (USBL) in ``positions_crs`` (``None``: local); with
    two or more the heading and X, Y come from them (and the scale too with ``position_scale``).
    ``apply`` False only computes. Returns the report (also kept in
    ``chunk.meta["underwater"]["levelling"]``)."""
    from pyarchinit_photogrammetry.core import crs as crs_mod
    from pyarchinit_photogrammetry.core import geometry

    depths = camera_depths(chunk)
    cams = [c for c in chunk.aligned_cameras if c.key in depths]
    if len(cams) < 3:
        raise ValueError(t_("Servono almeno tre foto allineate con la profondità ({0}).",
                            "At least three aligned photos with a depth are needed ({0}).").format(len(cams)))
    centres = np.array([c.center for c in cams])
    d = np.array([depths[c.key] for c in cams])
    fixed = None
    if isinstance(scale, str) and scale == "bars":
        fixed = bar_scale(chunk)
        if fixed is None:
            raise ValueError(t_("Il chunk non ha barre di scala di controllo.", "The chunk has no control scale bars."))
    elif scale is not None:
        fixed = float(scale)
    down = mean_view_direction(cams)
    try:
        sol = solve(centres, d, scale=fixed, down=down)
    except ValueError:
        if fixed is not None:
            raise
        bars = bar_scale(chunk)
        if bars is None:
            raise
        sol = solve(centres, d, scale=bars, down=down)
        sol["scale_from"] = "bars"
    if fixed is not None and scale == "bars":
        sol["scale_from"] = "bars"
    s, up = sol["scale"], sol["up"]
    s0, r0, t0 = geometry.decompose_similarity(np.asarray(chunk.transform, float))
    rot = _rotation(up, r0[0])
    mean = centres.mean(axis=0)
    t = np.zeros(3)
    t[2] = sol["offset"]
    old_xy = geometry.apply(np.asarray(chunk.transform, float), mean[None])[0][:2] if crs_mod.is_local(chunk.crs) \
        else np.zeros(2)
    t[:2] = old_xy - (s * rot @ mean)[:2]
    new = geometry.compose_similarity(s, rot, t)
    crs = crs_mod.LOCAL
    placed = None
    if positions:
        keys = [c.key for c in cams if c.key in positions]
        if len(keys) >= 2:
            by_key = {c.key: c for c in cams}
            src = geometry.apply(new, np.array([by_key[k].center for k in keys]))[:, :2]
            dst = np.array([positions[k][:2] for k in keys], float)
            ps, prot, pt = similarity_2d(src, dst, with_scale=position_scale)
            lift = np.eye(4)
            lift[:2, :2] = ps * prot
            lift[:2, 3] = pt
            # a uniform similarity: Z scaled too, and moved back so that it stays -depth on average
            z_before = geometry.apply(new, centres)[:, 2]
            lift[2, 2] = ps
            lift[2, 3] = float(np.mean(z_before) - ps * np.mean(z_before))
            new = lift @ new
            res = geometry.apply(new, np.array([by_key[k].center for k in keys]))[:, :2] - dst
            placed = {"cameras": len(keys), "rms": float(np.sqrt(np.mean(np.sum(res ** 2, axis=1)))),
                      "scale": ps, "crs": positions_crs or crs_mod.LOCAL}
            if positions_crs and not crs_mod.is_local(positions_crs):
                crs = positions_crs
    heights = geometry.apply(new, centres)[:, 2]
    resid = heights + d
    report = {"cameras": len(cams), "scale": float(geometry.decompose_similarity(new)[0]), "scale_from": sol["scale_from"],
              "scale_std": sol["scale_std"], "rms": float(np.sqrt(np.mean(resid[sol["inliers"]] ** 2))),
              "max": float(np.max(np.abs(resid))), "outliers": int((~sol["inliers"]).sum()),
              "tilt_std_deg": math.degrees(sol["tilt_std"]) if math.isfinite(sol["tilt_std"]) else None,
              "depth_range": sol["depth_range"], "observable": sol["observable"],
              "tilt_deg": float(math.degrees(math.acos(np.clip(abs(float(r0[2] @ up)), -1.0, 1.0)))),
              "residuals": {str(c.key): float(r) for c, r in zip(cams, resid)},
              "transform": np.asarray(new).tolist(), "crs": crs, "positions": placed}
    if apply:
        chunk.transform = new
        chunk.crs = crs
        chunk.meta["metric"] = True
        uw = chunk.meta.setdefault("underwater", {})
        uw["levelling"] = {k: v for k, v in report.items() if k != "residuals"}
        for c, r in zip(cams, resid):
            c.meta.setdefault("underwater", {})["depth_residual"] = float(r)
    return report


def levelled(chunk) -> bool:
    """The chunk is still as the last levelling left it (an alignment done again resets it)."""
    rec = (chunk.meta.get("underwater") or {}).get("levelling")
    return bool(rec) and np.allclose(np.asarray(rec.get("transform"), float), np.asarray(chunk.transform, float),
                                     rtol=1e-9, atol=1e-9)


def offset_search(chunk, log, around: float = 0.0, span: float = 600.0, step: float | None = None,
                  tolerance: float = 2.0, progress=None) -> dict:
    """The time offset (seconds added to the camera clock) that fits the depths of the log best to
    the aligned cameras: for every offset tried, the cameras' depths are matched and the levelling
    solved; its residual RMS is the measure. Coarse then fine. Returns ``{"offset", "rms",
    "curve"}`` (``curve``: offset, rms, cameras matched)."""
    from . import depthlog

    cams = chunk.aligned_cameras
    times = depthlog.photo_times(cams)
    if len(times) < 5:
        raise ValueError(t_("Servono almeno cinque foto allineate con l'ora.", "At least five aligned photos with a time are needed."))
    by_key = {c.key: c for c in cams}
    stamps = np.array([v[0] for v in times.values()])
    step = step or max(0.5, float(np.ptp(stamps)) / max(len(stamps), 1))
    curve = []

    def rms_at(off: float):
        m = depthlog.match(cams, log, off, tolerance=tolerance, times=times)
        keys = [k for k in m if k in by_key]
        if len(keys) < max(5, len(times) // 2):
            return math.inf, len(keys)
        try:
            sol = solve(np.array([by_key[k].center for k in keys]), np.array([m[k] for k in keys]))
        except ValueError:
            return math.inf, len(keys)
        return sol["rms"], len(keys)

    grid = np.arange(around - span, around + span + step / 2, step)
    best = (math.inf, around)
    for i, off in enumerate(grid):
        if progress is not None and i % 10 == 0:
            progress(i / len(grid) * 0.8, t_("Scarto di tempo: {0} s", "Time offset: {0} s").format(round(off, 1)))
        r, k = rms_at(float(off))
        curve.append((float(off), r, k))
        if r < best[0]:
            best = (r, float(off))
    # golden refinement around the best
    lo, hi = best[1] - step, best[1] + step
    for _ in range(25):
        a, b2 = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if rms_at(a)[0] <= rms_at(b2)[0]:
            hi = b2
        else:
            lo = a
    off = 0.5 * (lo + hi)
    r, k = rms_at(off)
    if r > best[0]:
        off, r = best[1], best[0]
    if progress is not None:
        progress(1.0, "")
    return {"offset": off, "rms": r, "cameras": k, "curve": curve}
