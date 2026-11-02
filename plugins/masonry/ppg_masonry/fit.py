"""Least squares with a robust weight (Huber's): a plane, a circle, a cylinder, a line through points. Points on a wall
are not all wall: a lamp, a scaffolding pole, a patch of vegetation stand off its surface; Huber's weight lets them
count less than the wall without throwing them away. The scale of the residuals is the median absolute deviation
(x 1.4826), the threshold ``k`` = 1.345 of it (95 % efficiency on Gaussian residuals)."""
from __future__ import annotations

import numpy as np

HUBER_K = 1.345


def robust_scale(r: np.ndarray, floor: float = 1e-6) -> float:
    r = np.asarray(r, np.float64)
    med = float(np.median(r))
    return max(1.4826 * float(np.median(np.abs(r - med))), floor)


def huber_weights(r: np.ndarray, k: float = HUBER_K) -> np.ndarray:
    s = robust_scale(r)
    a = np.abs(r) / s
    return np.where(a <= k, 1.0, k / np.maximum(a, 1e-12))


def robust_plane(points: np.ndarray, iterations: int = 30) -> dict:
    """The plane fitted to ``points`` (N, 3): ``centre``, unit ``normal`` (sign arbitrary), ``rms`` of the residuals,
    ``scale`` (robust), ``weights``. Iteratively reweighted total least squares."""
    p = np.asarray(points, np.float64)
    if len(p) < 3:
        raise ValueError("three points at least")
    origin = p.mean(axis=0)
    q = p - origin
    w = np.ones(len(q))
    normal = None
    for _ in range(iterations):
        c = (w[:, None] * q).sum(axis=0) / w.sum()
        d = q - c
        cov = (w[:, None] * d).T @ d
        n = np.linalg.eigh(cov)[1][:, 0]
        if normal is not None and n @ normal < 0:
            n = -n
        r = d @ n
        w_new = huber_weights(r)
        done = normal is not None and abs(1.0 - abs(n @ normal)) < 1e-14
        normal, w = n, w_new
        if done:
            break
    c = (w[:, None] * q).sum(axis=0) / w.sum()
    r = (q - c) @ normal
    return {"centre": origin + c, "normal": normal, "rms": float(np.sqrt(np.mean(r ** 2))),
            "scale": robust_scale(r), "weights": w, "residuals": r}


def circle_kasa(xy: np.ndarray) -> tuple[float, float, float]:
    q = np.asarray(xy, np.float64)
    a = np.column_stack([2 * q, np.ones(len(q))])
    sol = np.linalg.lstsq(a, (q ** 2).sum(axis=1), rcond=None)[0]
    return float(sol[0]), float(sol[1]), float(np.sqrt(max(sol[2] + sol[0] ** 2 + sol[1] ** 2, 0.0)))


def robust_circle(xy: np.ndarray, robust: bool = True) -> dict:
    """The circle fitted to plan points: ``centre`` (x, y), ``radius``, ``residuals`` (distance from the centre minus
    the radius, metres), ``rms``. Geometric least squares from the algebraic circle, Huber loss (plain least squares
    without ``robust``)."""
    from scipy.optimize import least_squares

    q = np.asarray(xy, np.float64)[:, :2]
    if len(q) < 3:
        raise ValueError("three points at least")
    origin = q.mean(axis=0)
    q = q - origin
    cx, cy, r0 = circle_kasa(q)

    def res(x):
        return np.hypot(q[:, 0] - x[0], q[:, 1] - x[1]) - x[2]

    scale = robust_scale(res([cx, cy, r0]), 1e-5)
    if robust:
        sol = least_squares(res, [cx, cy, r0], loss="huber", f_scale=HUBER_K * scale, x_scale="jac")
    else:
        sol = least_squares(res, [cx, cy, r0], x_scale="jac")
    r = res(sol.x)
    return {"centre": origin + sol.x[:2], "radius": float(sol.x[2]), "residuals": r,
            "rms": float(np.sqrt(np.mean(r ** 2)))}


def circle_through(p1, p2, p3) -> tuple[np.ndarray, float] | None:
    """The circle through three plan points ``(centre, radius)``; None when they are on a line."""
    ax, ay = p1
    bx, by = p2
    cx, cy = p3
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    centre = np.array([ux, uy])
    return centre, float(np.linalg.norm(np.asarray(p1) - centre))


def arc_coverage(xy: np.ndarray, centre) -> float:
    """The angular span (degrees) the points cover around ``centre``: 360 minus the largest empty gap."""
    a = np.sort(np.degrees(np.arctan2(xy[:, 1] - centre[1], xy[:, 0] - centre[0])) % 360.0)
    if len(a) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([a, [a[0] + 360.0]]))
    return float(360.0 - gaps.max())


REFINE_SHARES = (0.02, 0.04, 0.07, 0.10, 0.13, 0.16, 0.20, 0.25)  # the ring's band, grown pass after pass
GROWTH_DENSITY = 0.2  # a pass whose new points are sparser than this share of the tight band's stops the growth
PIECE_SHARE = 0.25  # beyond the tight band a piece of the slice joins the ring only with this share of it in the band


def _binned_coverage(p: np.ndarray, centre, bins: int = 72) -> float:
    """:func:`arc_coverage` on ``bins`` sectors (fast, for scoring many candidates): 360 minus the longest run of empty
    sectors."""
    if len(p) == 0:
        return 0.0
    k = (np.floor((np.degrees(np.arctan2(p[:, 1] - centre[1], p[:, 0] - centre[0])) % 360.0) / (360.0 / bins))
         .astype(int) % bins)
    occupied = np.zeros(bins, bool)
    occupied[k] = True
    if occupied.all():
        return 360.0
    empty = np.concatenate([~occupied, ~occupied])  # the longest run, across 0 degrees too
    run = longest = 0
    for e in empty:
        run = run + 1 if e else 0
        longest = max(longest, run)
    return 360.0 - min(longest, bins) * 360.0 / bins


def ransac_circle(xy: np.ndarray, iterations: int = 600, seed: int = 0, band_share: float = 0.02,
                  band_min: float = 0.05, refine_shares=REFINE_SHARES, expect_radius: float | None = None,
                  weights=None, labels=None) -> dict:
    """A circle through the plan points of a slice, all its pieces together: the points centred on their mean (map
    coordinates lose their digits in the products of a fit), circles through three random points, the one with most
    points within ``max(band_share x radius, band_min)`` weighted by the arc they cover (points x coverage / 360: a
    ring broken in pieces round the whole tower beats a longer arc of a wall beside it); a candidate whose radius
    differs from ``expect_radius`` by more than 25 % scores half; ``weights`` (per point, 0-1) count the points of
    the seed's score (the slice gives less to the points of short pieces: the bits of a rough ground), and the points
    weighing less than 1 never join the ring beyond the tight band; ``labels`` (the piece of the slice of each point):
    beyond the tight band a piece joins the ring only when a quarter of it lies in the band - a ring's piece runs
    along it, a contour of the sliced ground crosses it. Then the ring is grown around it: passes with the band
    widening from that tight one (``refine_shares`` of the radius, up to 25 %), each refitting the circle with Huber's
    loss, until a wider band brings no more points or brings them sparser than a fifth of the ring's density so far (the
    points per metre of band width: an oval ring keeps it, the ground or a collapse at the foot does not; with
    ``labels`` only the points of pieces not yet in the ring count, the ring's own pieces grow freely); the last fit
    is plain least squares on that ring - an oval ring gives its mean circle, not the curve of one end, and another
    structure beyond the band stays out.

    ``centre``, ``radius``, ``inliers`` (mask: the ring), ``residuals`` (of the ring's points), ``rms``, ``coverage``
    (degrees of arc the ring covers)."""
    q = np.asarray(xy, np.float64)[:, :2]
    if len(q) < 3:
        raise ValueError("three points at least")
    origin = q.mean(axis=0)
    p = q - origin
    rng = np.random.default_rng(seed)
    wts = np.ones(len(p)) if weights is None else np.asarray(weights, np.float64)
    labels = None if labels is None else np.asarray(labels, np.int64)
    grow = wts >= 1.0  # the points of short pieces join the ring only within the tight band
    pick = np.arange(len(p)) if len(p) <= 4000 else rng.choice(len(p), 4000, replace=False)
    sample, sample_w = p[pick], wts[pick]
    extent = float(np.ptp(p, axis=0).max())
    best, best_score = None, -1.0
    for _ in range(iterations):
        c = circle_through(*sample[rng.choice(len(sample), 3, replace=False)])
        if c is None or c[1] < 0.05 or c[1] > 5 * max(extent, 1.0):
            continue
        band = max(band_share * c[1], band_min)
        on = np.abs(np.hypot(sample[:, 0] - c[0][0], sample[:, 1] - c[0][1]) - c[1]) <= band
        if int(on.sum()) < 3:
            continue
        n = float(sample_w[on].sum())
        if n <= best_score:  # it cannot beat the best: coverage is at most 1
            continue
        score = n * _binned_coverage(sample[on], c[0]) / 360.0
        if expect_radius and abs(c[1] - expect_radius) > 0.25 * expect_radius:
            score *= 0.5
        if score > best_score:
            best, best_score = c, score
    if best is None:
        raise ValueError("no circle through the points")
    centre, radius = best

    def within(share, floor):
        dist = np.abs(np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1]) - radius)
        near = dist <= max(share * radius, floor)
        if share > band_share:  # beyond the tight band: only pieces lying along the ring, not short bits
            tight = dist <= max(band_share * radius, band_min)
            if labels is not None:  # a piece joins when a fair share of it lies in the band: not a contour across it
                inside = np.bincount(labels, weights=near.astype(float), minlength=labels.max() + 1)
                total = np.bincount(labels, minlength=labels.max() + 1)
                along = (inside / np.maximum(total, 1))[labels] >= PIECE_SHARE
            else:
                along = np.ones(len(p), bool)
            near &= (grow & along) | tight
        return near

    inl = within(band_share, band_min)
    prev_share = None
    for share in refine_shares:  # the ring grown while a wider band brings more points, as dense as the ring's
        grown = within(share, band_min)
        n = int(grown.sum())
        if n < 3:
            break
        if prev_share is not None:  # what this band adds to the last one, both around the circle as it is now
            base = within(prev_share, band_min)
            nb = int(base.sum())
            w0, w1 = max(prev_share * radius, band_min), max(share * radius, band_min)
            if n <= nb:
                break
            added = grown & ~base
            if labels is not None:  # the ring's own pieces grow freely (an oval ring is one piece): the density
                added &= ~np.isin(labels, np.unique(labels[base]))  # rule weighs the pieces it would take in
            if w1 > w0 and nb > 0 and 0 < int(added.sum()) and \
                    added.sum() / (2.0 * (w1 - w0)) < GROWTH_DENSITY * nb / (2.0 * w0):
                break  # what the wider band adds is scattered (the ground, a collapse at the foot), not the ring
        inl, prev_share = grown, share
        f = robust_circle(p[inl])
        centre, radius = np.asarray(f["centre"]), f["radius"]
    f = robust_circle(p[inl], robust=False)  # plain least squares on the ring: its mean circle
    centre, radius = np.asarray(f["centre"]), f["radius"]
    r = np.hypot(p[inl, 0] - centre[0], p[inl, 1] - centre[1]) - radius
    return {"centre": origin + centre, "radius": float(radius), "inliers": inl, "residuals": r,
            "rms": float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0, "coverage": arc_coverage(p[inl], centre)}


def _basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = axis / np.linalg.norm(axis)
    helper = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(a, helper)
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(a, e1)


def robust_cylinder(points: np.ndarray, axis0, centre0, radius0: float) -> dict:
    """The cylinder fitted to ``points`` (N, 3) from a first guess: ``axis`` (unit, upwards), ``centre`` (a point of
    the axis, the closest to ``centre0``), ``radius``, ``residuals`` (distance from the axis minus the radius), ``rms``.
    Huber loss on the radial residuals."""
    from scipy.optimize import least_squares

    p = np.asarray(points, np.float64)
    c0 = np.asarray(centre0, np.float64)
    a0 = np.asarray(axis0, np.float64)
    a0 = a0 / np.linalg.norm(a0)
    q = p - c0
    e1, e2 = _basis(a0)

    def unpack(x):
        a = a0 + x[0] * e1 + x[1] * e2
        a = a / np.linalg.norm(a)
        c = x[2] * e1 + x[3] * e2
        return a, c, x[4]

    def res(x):
        a, c, r = unpack(x)
        d = q - c
        radial = d - (d @ a)[:, None] * a
        return np.linalg.norm(radial, axis=1) - r

    x0 = np.array([0.0, 0.0, 0.0, 0.0, float(radius0)])
    scale = robust_scale(res(x0), 1e-5)
    sol = least_squares(res, x0, loss="huber", f_scale=HUBER_K * scale, x_scale="jac")
    scale = robust_scale(res(sol.x), 1e-5)
    sol = least_squares(res, sol.x, loss="huber", f_scale=HUBER_K * scale, x_scale="jac")
    a, c, r = unpack(sol.x)
    if a[2] < 0:
        a = -a
    rr = res(sol.x)
    return {"axis": a, "centre": c0 + c, "radius": float(r), "residuals": rr, "rms": float(np.sqrt(np.mean(rr ** 2)))}


def tilt(axis_up: np.ndarray) -> tuple[float, float | None]:
    """``(degrees from the vertical, azimuth of the top)`` of an axis pointing upwards; azimuth clockwise from the
    north of the grid (+y), None for a vertical axis."""
    a = np.asarray(axis_up, np.float64)
    a = a / np.linalg.norm(a)
    if a[2] < 0:
        a = -a
    h = float(np.hypot(a[0], a[1]))
    deg = float(np.degrees(np.arctan2(h, a[2])))
    return deg, (float(np.degrees(np.arctan2(a[0], a[1])) % 360.0) if h > 1e-12 else None)


def line_through(centres: np.ndarray, weights=None) -> dict:
    """The axis through points at several heights (``x = x0 + a z``, ``y = y0 + b z``, least squares, Huber weights
    when there are four points or more): ``axis`` (unit, upwards), ``point`` (at the mean height), ``residuals``."""
    c = np.asarray(centres, np.float64)
    z = c[:, 2] - c[:, 2].mean()
    w = np.ones(len(c)) if weights is None else np.asarray(weights, np.float64)
    a_mat = np.column_stack([np.ones(len(c)), z])
    for _ in range(20 if len(c) >= 4 else 1):
        sw = np.sqrt(w)
        sx = np.linalg.lstsq(a_mat * sw[:, None], c[:, 0] * sw, rcond=None)[0]
        sy = np.linalg.lstsq(a_mat * sw[:, None], c[:, 1] * sw, rcond=None)[0]
        r = np.hypot(c[:, 0] - a_mat @ sx, c[:, 1] - a_mat @ sy)
        if len(c) >= 4:
            w = huber_weights(r)
    axis = np.array([sx[1], sy[1], 1.0])
    return {"axis": axis / np.linalg.norm(axis), "point": np.array([sx[0], sy[0], c[:, 2].mean()]), "residuals": r}
