"""Powerlines in the dense cloud: the conductors as catenary curves, the poles, the clearance
of every conductor from the ground and its distance from the nearest obstacle.

The detection works on the points of the dense cloud, in the world frame of the chunk (metres,
z up: a georeferenced or levelled survey):

1. the height of every point above the ground - from the ground points when the cloud is
   classified (class 2), otherwise from the progressive morphological filter of the program;
2. the candidates: points high above the ground whose nearest points lie along a line (the
   direction of the point), with nothing beside the line (a roof has points beside every point,
   a wire has none nearer than the next conductor) and nothing in the air under it (a roof edge
   has the wall under it, a branch the trunk, a wire hangs free);
3. the wires: candidates linked when each lies along the direction of the other (parallel
   conductors half a metre apart are never linked), groups with enough points and length;
4. a catenary ``z = z0 + a (cosh((s - s0) / a) - 1)`` fitted to every wire along its direction in
   plan by robust least squares; pieces of the same wire separated by a gap of the
   reconstruction are merged when the catenary of one predicts the other;
5. the poles where several wires end together, confirmed by a vertical structure of points when
   the cloud has one (thin poles often leave no points in photogrammetry: the pole is then
   placed from the wire ends);
6. for every wire the clearance from the ground along its length and the distance from the
   nearest point that is not a wire (vegetation, buildings).

The result is a plain dictionary (``chunk.meta["powerlines"]``): the wires with their
catenary, vertices every metre, length, sag, clearance and obstacle distance, and the poles.
"""
from __future__ import annotations

import numpy as np

from pyarchinit_photogrammetry.core.progress import as_progress

from .texts import t

WIRE, TOWER = 14, 15  # ASPRS classes: wire - conductor, transmission tower
GROUND = 2
CLASS_LABELS = {WIRE: "Conduttore (linea elettrica) / Wire conductor", TOWER: "Palo o traliccio / Pole or tower"}

DEFAULTS = {
    "min_height": 4.0,   # m above the ground: lower points are not looked at
    "radius": 1.0,       # m: the neighbourhood of a point gives its direction
    "linearity": 0.85,   # (l1 - l2) / l1 of the points along the direction: 1 = a perfect line
    "max_slope": 0.6,    # sine of the steepest wire (0.6 = 37 degrees)
    "gap": 2.0,          # m: points of one wire are linked up to this distance
    "min_points": 10,    # points of the smallest wire
    "min_length": 5.0,   # m: length of the shortest wire
    "merge_gap": 15.0,   # m: pieces of a wire separated by a gap of the reconstruction are joined
    "cell": 1.0,         # m: cell of the ground grid
}
NEIGHBOURS = 24          # nearest points that give the direction of a point
TUBE = 0.25              # m: lateral distance from the direction within which a point is on the wire
RING = 0.5               # m: up to here nothing else may lie beside the wire
MAX_RING = 2             # points tolerated beside it (lateral noise)
BELOW = (0.3, 3.0)       # m under a point: the layer of air that must be empty under a wire
BELOW_RADIUS = 0.75      # m: horizontal radius of that layer
MAX_BELOW = 2            # points tolerated in it
PERP_LINK = 0.25         # m: a point is linked to another when it lies this close to its line
LATERAL_MAX = 0.4        # m: a wire straighter than this in plan (std of the offsets)
RMS_MAX = 0.75           # m: the catenary must fit this well
KINK_RATIO = 0.5         # a run of points is cut in two where two parabolas fit twice as well as one
LATERAL_MERGE = 0.5      # m: pieces of one wire lie on the same line in plan
SLOPE_MERGE = 0.08       # the slopes of two pieces of one wire agree where they face each other
POLE_RADIUS = 3.0        # m: wire ends this close end on the same pole
STRUCTURE_RADIUS = 1.5   # m: points this close to a pole make its structure
STRUCTURE_POINTS = 20
STRUCTURE_SPREAD = 1.0   # m: a pole is thin (std of x and y of its points)
OBSTACLE_HEIGHT = 0.5    # m above the ground: lower points are not obstacles
OCCUPANCY_CELL = 5.0     # m: cells of the map that says where the survey has points
BORDER_REACH = (5.0, 10.0, 15.0)  # m beyond a wire end: no points there = the survey ends, not the wire
STEP = 1.0               # m between the vertices of a wire


# ---------------------------------------------------------------------------------------- ground
class Ground:
    """The ground as a grid of heights: :meth:`height_at` and :meth:`above`."""

    def __init__(self, z: np.ndarray, x0: float, y0: float, cell: float, south_first: bool, source: str):
        self.z, self.x0, self.y0, self.cell, self.south_first, self.source = z, x0, y0, cell, south_first, source

    @classmethod
    def of(cls, xyz: np.ndarray, classes: np.ndarray | None, cell: float) -> "Ground":
        """From the ground points when at least 2 % of the cloud (or 1000 points) are class 2,
        else from the progressive morphological filter (windows up to 40 m)."""
        if classes is not None and (classes == GROUND).sum() >= min(max(1000, 0.02 * len(xyz)), len(xyz)):
            from pyarchinit_photogrammetry.archaeo.measures import ground_surface

            z, x0, y0, cell = ground_surface(xyz, classes, cell)
            return cls(z, x0, y0, cell, True, "classes")
        from pyarchinit_photogrammetry.ai.classify import ground_surface

        z, (x0, y1), cell = ground_surface(xyz, cell=cell, max_window=40.0, slope=0.3, dh0=0.1, dh_max=3.0)
        return cls(z, x0, y1, cell, False, "morphology")

    def height_at(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        h, w = self.z.shape
        col = np.clip(((xy[:, 0] - self.x0) / self.cell).astype(int), 0, w - 1)
        if self.south_first:
            row = np.clip(((xy[:, 1] - self.y0) / self.cell).astype(int), 0, h - 1)
        else:
            row = np.clip(((self.y0 - xy[:, 1]) / self.cell).astype(int), 0, h - 1)
        return self.z[row, col]

    def above(self, xyz: np.ndarray) -> np.ndarray:
        return xyz[:, 2] - self.height_at(xyz[:, :2])


# ------------------------------------------------------------------------------------ candidates
def _pca(Q: np.ndarray, w: np.ndarray):
    """Eigenvalues (ascending) and eigenvectors of the covariance of the points ``Q`` (m, k, 3)
    weighted by ``w`` (m, k, 1): one small problem per row."""
    n = w.sum((1, 2))
    mean = (Q * w).sum(1) / np.maximum(n, 1)[:, None]
    D = np.where(w, Q - mean[:, None, :], 0.0)
    C = np.einsum("mki,mkj->mij", D, D) / np.maximum(n - 1, 1)[:, None, None]
    evals, evecs = np.linalg.eigh(C)
    return evals, evecs, n


def candidates(xyz: np.ndarray, hag: np.ndarray, min_height: float, radius: float, linearity: float,
               max_slope: float, progress=None):
    """Indices (into ``xyz``) of the points that look like a wire, their directions (unit
    vectors), and the indices of the points at least 1 m above the ground, used later for
    the poles and the obstacles."""
    from scipy.spatial import cKDTree

    progress = as_progress(progress)
    # wires, what could lie in the air under them, obstacles and poles: never the mass of the ground
    high = np.nonzero(hag >= max(min(min_height - BELOW[1], OBSTACLE_HEIGHT), 0.3))[0]
    P = xyz[high]
    top = np.nonzero(hag[high] >= min_height)[0]  # indices into P
    empty = (high[np.zeros(0, int)], np.zeros((0, 3)), high)
    if len(top) == 0 or len(P) < 4:
        return empty
    tree = cKDTree(P)
    direction = np.zeros((len(top), 3))
    keep = np.zeros(len(top), bool)
    block = 100_000
    for start in range(0, len(top), block):
        sel = top[start:start + block]
        d, nb = tree.query(P[sel], k=NEIGHBOURS, distance_upper_bound=radius)
        valid = np.isfinite(d)
        nb = np.where(valid, nb, 0)
        Q = P[nb]
        w = valid[..., None]
        evals, evecs, n = _pca(Q, w)
        d0 = evecs[:, :, 2]
        # lateral distance of every neighbour from the line through the point along d0
        V = Q - P[sel][:, None, :]
        along = (V * d0[:, None, :]).sum(2)
        lateral = np.linalg.norm(V - along[..., None] * d0[:, None, :], axis=2)
        tube = valid & (lateral <= TUBE)
        ring = valid & (lateral > TUBE) & (lateral <= RING)
        evals2, evecs2, n2 = _pca(Q, tube[..., None])
        l1, l2 = evals2[:, 2], evals2[:, 1]
        lin = np.where(l1 > 1e-12, (l1 - l2) / np.maximum(l1, 1e-12), 0.0)
        d2 = evecs2[:, :, 2]
        ok = (n >= 4) & (n2 >= 4) & (lin >= linearity) & (ring.sum(1) <= MAX_RING) & (np.abs(d2[:, 2]) <= max_slope)
        keep[start:start + len(sel)] = ok
        direction[start:start + len(sel)] = d2
        progress(min(0.9, (start + len(sel)) / len(top)), t("Punti lineari in aria: {0:,} su {1:,}",
                                                               "Linear points in the air: {0:,} of {1:,}")
                 .format(int(keep.sum()), len(top)))
    sel = top[keep]
    dirs = direction[keep]
    if len(sel) == 0:
        return empty
    # free air under the point: nothing between BELOW[0] and BELOW[1] metres lower, within BELOW_RADIUS
    centre = P[sel] - np.array([0.0, 0.0, (BELOW[0] + BELOW[1]) / 2])
    reach = float(np.hypot(BELOW_RADIUS, (BELOW[1] - BELOW[0]) / 2)) + 1e-6
    lists = tree.query_ball_point(centre, reach, return_sorted=False)
    lengths = np.fromiter((len(lst) for lst in lists), int, len(lists))
    below = np.zeros(len(sel), int)
    if lengths.sum():
        flat = np.fromiter((i for lst in lists for i in lst), int, int(lengths.sum()))
        owner = np.repeat(np.arange(len(sel)), lengths)
        q = P[flat]
        c = P[sel][owner]
        dz = c[:, 2] - q[:, 2]
        inside = (dz >= BELOW[0]) & (dz <= BELOW[1]) & (np.hypot(q[:, 0] - c[:, 0], q[:, 1] - c[:, 1]) <= BELOW_RADIUS)
        below = np.bincount(owner[inside], minlength=len(sel))
    free = below <= MAX_BELOW
    progress(1.0, t("Candidati: {0:,}", "Candidates: {0:,}").format(int(free.sum())))
    return high[sel[free]], dirs[free], high


# ----------------------------------------------------------------------------------------- wires
def link(P: np.ndarray, dirs: np.ndarray, gap: float) -> np.ndarray:
    """Label of the wire of every candidate: points linked when each lies within ``gap`` and
    within :data:`PERP_LINK` of the line of the other."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    n = len(P)
    if n == 0:
        return np.zeros(0, int)
    pairs = cKDTree(P).query_pairs(gap, output_type="ndarray")
    if len(pairs):
        v = P[pairs[:, 1]] - P[pairs[:, 0]]
        d0, d1 = dirs[pairs[:, 0]], dirs[pairs[:, 1]]
        perp0 = np.linalg.norm(v - (v * d0).sum(1)[:, None] * d0, axis=1)
        perp1 = np.linalg.norm(v - (v * d1).sum(1)[:, None] * d1, axis=1)
        pairs = pairs[(perp0 <= PERP_LINK) & (perp1 <= PERP_LINK)]
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
    return connected_components(g, directed=False)[1]


class Catenary:
    """``z = z0 + a (cosh((s - s0) / a) - 1)`` along the direction ``u`` (unit, in plan) from
    ``origin``; ``s`` runs from ``s_min`` to ``s_max``."""

    def __init__(self, origin, u, a, s0, z0, s_min, s_max):
        self.origin = np.asarray(origin, float)
        self.u = np.asarray(u, float)
        self.a, self.s0, self.z0, self.s_min, self.s_max = float(a), float(s0), float(z0), float(s_min), float(s_max)

    @property
    def normal(self) -> np.ndarray:
        return np.array([-self.u[1], self.u[0]])

    def z_at(self, s) -> np.ndarray:
        x = np.clip((np.asarray(s, float) - self.s0) / self.a, -300.0, 300.0)
        return self.z0 + self.a * (np.cosh(x) - 1.0)

    def points(self, s) -> np.ndarray:
        s = np.asarray(s, float)
        return np.column_stack([self.origin + s[:, None] * self.u, self.z_at(s)])

    def s_of(self, xy) -> np.ndarray:
        return (np.asarray(xy, float).reshape(-1, 2) - self.origin) @ self.u

    @property
    def length(self) -> float:
        """The arc length between the ends."""
        return self.a * float(np.sinh((self.s_max - self.s0) / self.a) - np.sinh((self.s_min - self.s0) / self.a))

    def samples(self, step: float = STEP) -> np.ndarray:
        n = max(int(np.ceil((self.s_max - self.s_min) / step)), 1)
        return self.points(np.linspace(self.s_min, self.s_max, n + 1))

    @property
    def sag(self) -> float:
        """The largest distance below the chord between the ends."""
        s = np.linspace(self.s_min, self.s_max, 101)
        z = self.z_at(s)
        chord = z[0] + (z[-1] - z[0]) * (s - s[0]) / max(s[-1] - s[0], 1e-9)
        return float(max((chord - z).max(), 0.0))

    def as_dict(self) -> dict:
        return {"origin": np.round(self.origin, 3).tolist(), "u": np.round(self.u, 6).tolist(), "a": round(self.a, 3),
                "s0": round(self.s0, 3), "z0": round(self.z0, 3), "s_min": round(self.s_min, 3),
                "s_max": round(self.s_max, 3)}

    @classmethod
    def from_dict(cls, d: dict) -> "Catenary":
        return cls(d["origin"], d["u"], d["a"], d["s0"], d["z0"], d["s_min"], d["s_max"])


def _plan_axis(Q: np.ndarray):
    """``(centre, u, s, lateral)``: the axis of the points in plan (2D principal direction,
    oriented eastward), the coordinate of every point along it and its offset beside it."""
    c = Q[:, :2].mean(0)
    X = Q[:, :2] - c
    _, _, vt = np.linalg.svd(X, full_matrices=False)
    u = vt[0]
    if u[0] < 0 or (u[0] == 0 and u[1] < 0):
        u = -u
    return c, u, X @ u, X @ np.array([-u[1], u[0]])


def _parabola_sse(s: np.ndarray, z: np.ndarray) -> float:
    sm = s.mean()
    A = np.column_stack([np.ones_like(s), s - sm, (s - sm) ** 2])
    coef, *_ = np.linalg.lstsq(A, z, rcond=None)
    r = A @ coef - z
    return float(r @ r)


def split_at_kinks(s: np.ndarray, z: np.ndarray, min_points: int, min_length: float) -> list[np.ndarray]:
    """Groups of indices (into ``s``, ``z``) that are separate catenaries along one line in
    plan: a cable linked across a pole with the next span sags on both sides of it, and two
    parabolas fit the run of points far better than one; the cut goes where they fit best."""
    order = np.argsort(s)
    s, z = s[order], z[order]
    reach = max(10.0, 2.0 * min_length)  # m on each side of a cut: the fits are compared there
    out = []

    def rec(lo: int, hi: int) -> None:
        ss, zz = s[lo:hi], z[lo:hi]
        n = hi - lo
        if n < 2 * min_points or ss[-1] - ss[0] < 2 * min_length:
            out.append(order[lo:hi])
            return

        def score(k: int) -> float | None:
            """The two-parabola SSE around the cut ``k`` when it beats one parabola, else None."""
            if ss[k] - ss[0] < min_length or ss[-1] - ss[k] < min_length:
                return None
            a, b = int(np.searchsorted(ss, ss[k] - reach)), int(np.searchsorted(ss, ss[k] + reach))
            if k - a < 5 or b - k < 5:
                return None
            one = _parabola_sse(ss[a:b], zz[a:b])
            two = _parabola_sse(ss[a:k], zz[a:k]) + _parabola_sse(ss[k:b], zz[k:b])
            return two if one > 1e-6 * (b - a) and two <= KINK_RATIO * one else None

        step = max(1, int(n / max(ss[-1] - ss[0], 1.0)))  # about one cut tried per metre, then every point
        best, best_k = np.inf, None
        for k in range(min_points, n - min_points, step):
            two = score(k)
            if two is not None and two < best:
                best, best_k = two, k
        if best_k is None:
            out.append(order[lo:hi])
            return
        for k in range(max(min_points, best_k - 3 * step), min(n - min_points, best_k + 3 * step)):
            two = score(k)
            if two is not None and two < best:
                best, best_k = two, k
        rec(lo, lo + best_k)
        rec(lo + best_k, hi)

    rec(0, len(s))
    return out


def fit(Q: np.ndarray) -> tuple[Catenary, float, float]:
    """The catenary through the points ``Q`` of one wire: its direction in plan from the 2D
    principal axis, then ``(a, slope and height at the middle)`` by robust least squares (the
    lowest point may lie outside the piece). Returns ``(catenary, rms of the heights, std of the
    lateral offsets)``."""
    from scipy.optimize import least_squares

    c, u, s, lateral = _plan_axis(Q)
    z = Q[:, 2]
    sm = float(s.mean())
    A = np.column_stack([np.ones_like(s), s - sm, (s - sm) ** 2])
    (c0, c1, c2), *_ = np.linalg.lstsq(A, z, rcond=None)
    a0 = min(max(1.0 / (2.0 * c2), 1.0), 1e6) if c2 > 1e-7 else 1e5

    def model(p, s_):
        a, m0, zm = p  # a, the slope and the height at the middle of the piece
        x = np.clip((s_ - sm) / a + np.arcsinh(m0), -300.0, 300.0)
        return zm + a * (np.cosh(x) - np.sqrt(1.0 + m0 * m0))

    res = least_squares(lambda p: model(p, s) - z, [a0, float(c1), float(c0)], loss="soft_l1", f_scale=0.1,
                        bounds=([1.0, -5.0, -np.inf], [1e6, 5.0, np.inf]), x_scale="jac", max_nfev=200)
    a, m0, zm = res.x
    s0 = sm - a * np.arcsinh(m0)
    z0 = zm - a * (np.sqrt(1.0 + m0 * m0) - 1.0)
    cat = Catenary(c, u, a, s0, z0, s.min(), s.max())
    rms = float(np.sqrt(np.mean((cat.z_at(s) - z) ** 2)))
    return cat, rms, float(lateral.std())


def _merge_pieces(pieces: list[dict], merge_gap: float) -> list[list[int]]:
    """Groups of pieces that are one wire: same direction, same line in plan, a gap between
    them of at most ``merge_gap`` and the catenary of one reaching the other."""
    n = len(pieces)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ci = pieces[i]["cat"]
        for j in range(i + 1, n):
            cj = pieces[j]["cat"]
            cos = float(ci.u @ cj.u)
            if abs(cos) < 0.98:
                continue
            if abs((cj.origin - ci.origin) @ ci.normal) > LATERAL_MERGE or \
                    abs((ci.origin - cj.origin) @ cj.normal) > LATERAL_MERGE:
                continue
            ends_j = sorted(ci.s_of(cj.points(np.array([cj.s_min, cj.s_max]))[:, :2]).tolist())
            gap = max(ends_j[0] - ci.s_max, ci.s_min - ends_j[1])
            if gap > merge_gap or gap < -1.0:
                continue
            near_i = ci.s_max if ends_j[0] >= ci.s_max else ci.s_min  # the end of i that faces j
            near_j = cj.s_min if (cos > 0) == (ends_j[0] >= ci.s_max) else cj.s_max
            # one wire keeps its slope across the gap; two spans meeting on a pole both sag away from it
            slope_i = float(np.sinh((near_i - ci.s0) / ci.a))
            slope_j = float(np.sinh((near_j - cj.s0) / cj.a)) * (1.0 if cos > 0 else -1.0)
            if abs(slope_i - slope_j) > SLOPE_MERGE + 1.5 * max(gap, 0.0) / min(ci.a, cj.a):
                continue
            z_true = float(cj.z_at(near_j))
            z_pred = float(ci.z_at(ci.s_of(cj.points(np.array([near_j]))[:, :2])[0]))
            z_back = float(cj.z_at(cj.s_of(ci.points(np.array([near_i]))[:, :2])[0]))
            z_true_i = float(ci.z_at(near_i))
            tol = 0.5 + 0.03 * max(gap, 0.0)
            if abs(z_pred - z_true) > tol and abs(z_back - z_true_i) > tol:
                continue
            parent[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


def wires(xyz: np.ndarray, cand: np.ndarray, dirs: np.ndarray, gap: float, min_points: int, min_length: float,
          merge_gap: float, progress=None) -> list[dict]:
    """The wires among the candidate points: ``[{"index": indices into xyz, "cat": Catenary,
    "rms": m, "lateral": m}, ...]``."""
    progress = as_progress(progress)
    P = xyz[cand]
    labels = link(P, dirs, gap)
    pieces = []
    if len(labels):
        counts = np.bincount(labels)
        for lab in np.nonzero(counts >= min_points)[0]:
            idx = cand[labels == lab]
            Q = xyz[idx]
            if np.ptp(Q[:, :2], axis=0).max() < min_length / 2:
                continue
            _c, _u, s, _lateral = _plan_axis(Q)
            for sub in split_at_kinks(s, Q[:, 2], min_points, min_length):
                if len(sub) < min_points:
                    continue
                cat, rms, lateral = fit(Q[sub])
                if rms > RMS_MAX or lateral > LATERAL_MAX:
                    continue
                pieces.append({"index": idx[sub], "cat": cat, "rms": rms, "lateral": lateral})
    progress(0.5, t("Tratti: {0}", "Pieces: {0}").format(len(pieces)))
    out = []
    for group in _merge_pieces(pieces, merge_gap):
        idx = np.concatenate([pieces[i]["index"] for i in group])
        cat, rms, lateral = fit(xyz[idx]) if len(group) > 1 else (pieces[group[0]]["cat"], pieces[group[0]]["rms"],
                                                                    pieces[group[0]]["lateral"])
        if cat.length < min_length:
            continue
        out.append({"index": idx, "cat": cat, "rms": rms, "lateral": lateral})
    out.sort(key=lambda w: -w["cat"].length)
    progress(1.0, t("Conduttori: {0}", "Wires: {0}").format(len(out)))
    return out


def _group_lines(found: list[dict]) -> list[int]:
    """The line (a group of parallel conductors side by side) of every wire, numbered from 1."""
    n = len(found)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        ci = found[i]["cat"]
        for j in range(i + 1, n):
            cj = found[j]["cat"]
            if abs(float(ci.u @ cj.u)) < 0.98 or abs((cj.origin - ci.origin) @ ci.normal) > 5.0:
                continue
            ends = sorted(ci.s_of(cj.points(np.array([cj.s_min, cj.s_max]))[:, :2]).tolist())
            overlap = min(ends[1], ci.s_max) - max(ends[0], ci.s_min)
            if overlap >= 0.5 * min(ends[1] - ends[0], ci.s_max - ci.s_min):
                parent[find(i)] = find(j)
    numbers: dict[int, int] = {}
    return [numbers.setdefault(find(i), len(numbers) + 1) for i in range(n)]


# ----------------------------------------------------------------------------------------- poles
def _occupancy(xyz: np.ndarray):
    """Cells of :data:`OCCUPANCY_CELL` metres that hold points of the survey."""
    lo = np.floor(xyz[:, :2].min(0) / OCCUPANCY_CELL).astype(int)
    hi = np.floor(xyz[:, :2].max(0) / OCCUPANCY_CELL).astype(int)
    w, h = hi[0] - lo[0] + 1, hi[1] - lo[1] + 1
    col = (np.floor(xyz[:, 0] / OCCUPANCY_CELL).astype(int) - lo[0])
    row = (np.floor(xyz[:, 1] / OCCUPANCY_CELL).astype(int) - lo[1])
    grid = np.zeros((h, w), bool)
    grid[row, col] = True

    def occupied(xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        c = np.floor(xy[:, 0] / OCCUPANCY_CELL).astype(int) - lo[0]
        r = np.floor(xy[:, 1] / OCCUPANCY_CELL).astype(int) - lo[1]
        inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.zeros(len(xy), bool)
        out[inside] = grid[r[inside], c[inside]]
        return out

    return occupied


def poles(found: list[dict], xyz: np.ndarray, high: np.ndarray, wire_index: np.ndarray, ground: Ground,
          occupied) -> tuple[list[dict], np.ndarray]:
    """The poles: wire ends that meet (within :data:`POLE_RADIUS`), away from the border of the
    survey, with a vertical structure of points when the cloud has one. Returns the poles and
    the indices (into ``xyz``) of the points of their structures."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from scipy.spatial import cKDTree

    ends, owner = [], []
    for k, w in enumerate(found):
        cat = w["cat"]
        for s_end, outward in ((cat.s_min, -1.0), (cat.s_max, 1.0)):
            p = cat.points(np.array([s_end]))[0]
            beyond = np.array([p[:2] + outward * r * cat.u for r in BORDER_REACH])
            if not occupied(beyond).any():
                continue  # the survey ends here, not the wire
            ends.append(p)
            owner.append(k)
    if not ends:
        return [], np.zeros(0, int)
    E = np.array(ends)
    owner = np.array(owner)
    pairs = cKDTree(E[:, :2]).query_pairs(POLE_RADIUS, output_type="ndarray")
    g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(E), len(E)))
    labels = connected_components(g, directed=False)[1]
    is_wire = np.zeros(len(xyz), bool)
    is_wire[wire_index] = True
    structure_pool = high[~is_wire[high]]
    tree = cKDTree(xyz[structure_pool, :2]) if len(structure_pool) else None
    out, structure = [], []
    for lab in np.unique(labels):
        sel = np.nonzero(labels == lab)[0]
        wires_here = sorted(set(owner[sel].tolist()))
        xy = E[sel, :2].mean(0)
        top = float(E[sel, 2].max())
        detected, points = False, np.zeros(0, int)
        if tree is not None:
            near = structure_pool[tree.query_ball_point(xy, STRUCTURE_RADIUS)]
            if len(near) >= STRUCTURE_POINTS and xyz[near, :2].std(0).max() <= STRUCTURE_SPREAD:
                detected, points = True, near
                xy = xyz[near, :2].mean(0)
                top = max(top, float(xyz[near, 2].max()))  # the pole may rise above the wires
        if not detected and len(wires_here) < 2:
            continue
        z_ground = float(ground.height_at(xy)[0])
        out.append({"x": float(xy[0]), "y": float(xy[1]), "z_ground": z_ground, "height_m": top - z_ground,
                    "wires": wires_here, "detected": detected, "points": int(len(points))})
        structure.append(points)
    out.sort(key=lambda p: (p["x"], p["y"]))
    return out, (np.concatenate(structure) if structure else np.zeros(0, int))


# ------------------------------------------------------------------------------------- the whole
def detect(xyz: np.ndarray, classes: np.ndarray | None = None, progress=None, **params) -> dict:
    """The powerlines of a cloud in the world frame (metres, z up). Returns the report (a plain
    dictionary) with ``"wire_index"`` and ``"pole_index"`` (arrays of point indices, not kept in
    the project) added."""
    from scipy.spatial import cKDTree

    p = {**DEFAULTS, **params}
    unknown = set(params) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown parameters: {', '.join(sorted(unknown))}")
    progress = as_progress(progress)
    xyz = np.asarray(xyz, float).reshape(-1, 3)
    progress(0.02, t("Terreno e altezze dal suolo di {0:,} punti", "Ground and heights above it of {0:,} points")
             .format(len(xyz)))
    ground = Ground.of(xyz, None if classes is None else np.asarray(classes), float(p["cell"]))
    hag = ground.above(xyz)
    cand, dirs, high = candidates(xyz, hag, float(p["min_height"]), float(p["radius"]), float(p["linearity"]),
                                  float(p["max_slope"]), progress.sub(0.1, 0.45))
    found = wires(xyz, cand, dirs, float(p["gap"]), int(p["min_points"]), float(p["min_length"]),
                  float(p["merge_gap"]), progress.sub(0.55, 0.2))
    wire_index = np.concatenate([w["index"] for w in found]) if found else np.zeros(0, int)
    progress(0.78, t("Pali", "Poles"))
    pole_list, pole_index = poles(found, xyz, high, wire_index, ground, _occupancy(xyz)) if found else ([], np.zeros(0, int))
    progress(0.85, t("Franco dal suolo e ostacoli", "Clearance and obstacles"))
    is_wire = np.zeros(len(xyz), bool)
    is_wire[wire_index] = True
    is_pole = np.zeros(len(xyz), bool)
    is_pole[pole_index] = True
    obstacles = high[~is_wire[high] & ~is_pole[high] & (hag[high] >= OBSTACLE_HEIGHT)]
    tree = cKDTree(xyz[obstacles]) if len(obstacles) else None
    lines = _group_lines(found)
    report_wires = []
    for k, w in enumerate(found):
        cat = w["cat"]
        pts = cat.samples(STEP)
        clearance = pts[:, 2] - ground.height_at(pts[:, :2])
        obstacle = float(tree.query(pts)[0].min()) if tree is not None else None
        start, end = pts[0], pts[-1]
        azimuth = float(np.degrees(np.arctan2(cat.u[0], cat.u[1])) % 180.0)
        report_wires.append({
            "id": k + 1, "line": lines[k], "points": int(len(w["index"])), "length_m": round(cat.length, 3),
            "sag_m": round(cat.sag, 3), "rms_m": round(w["rms"], 3), "lateral_m": round(w["lateral"], 3),
            "clearance_m": round(float(clearance.min()), 3), "obstacle_m": None if obstacle is None else round(obstacle, 3),
            "z_min_m": round(float(pts[:, 2].min()), 3), "z_max_m": round(float(pts[:, 2].max()), 3),
            "azimuth_deg": round(azimuth, 1), "start": np.round(start, 3).tolist(), "end": np.round(end, 3).tolist(),
            "catenary": cat.as_dict(), "vertices": np.round(pts, 3).tolist(),
        })
    for i, pole in enumerate(pole_list):
        ids = [w + 1 for w in pole["wires"]]
        pole_list[i] = {"id": i + 1, "x": round(pole["x"], 3), "y": round(pole["y"], 3),
                        "z_ground": round(pole["z_ground"], 3), "height_m": round(pole["height_m"], 3),
                        "wires": ids, "lines": sorted({lines[w - 1] for w in ids}), "detected": pole["detected"],
                        "points": pole["points"]}
    report = {
        "version": 1, "params": {k: p[k] for k in DEFAULTS}, "ground": ground.source, "points": int(len(xyz)),
        "candidates": int(len(cand)), "wire_points": int(len(wire_index)), "lines": int(max(lines, default=0)),
        "wires": report_wires, "poles": pole_list,
    }
    progress(1.0, t("{0} conduttori, {1} pali", "{0} wires, {1} poles").format(len(report_wires), len(pole_list)))
    return {**report, "wire_index": wire_index, "pole_index": pole_index}


def detect_chunk(chunk, classify: bool = True, progress=None, **params) -> dict:
    """The powerlines of the dense cloud of ``chunk``, kept in ``chunk.meta["powerlines"]``;
    with ``classify`` the points of the wires take class 14 and those of the detected poles
    class 15 (points labelled by hand keep their class; the classes 14 and 15 of a previous run
    are reset first). Returns the report."""
    pc = chunk.point_cloud
    if pc is None:
        raise RuntimeError(t("Costruisci prima la nuvola densa", "Build the dense cloud first"))
    progress = as_progress(progress)
    xyz = chunk.to_world(pc.xyz)
    result = detect(xyz, pc.classification, progress.sub(0.0, 0.85 if classify else 1.0), **params)
    wire_index, pole_index = result.pop("wire_index"), result.pop("pole_index")
    if classify:
        progress(0.86, t("Classi 14 e 15 nella nuvola", "Classes 14 and 15 in the cloud"))
        codes = pc.classification
        manual = pc.manual if pc.manual is not None else np.zeros(len(pc), bool)
        codes[np.isin(codes, (WIRE, TOWER)) & ~manual] = 1
        for idx, code in ((wire_index, WIRE), (pole_index, TOWER)):
            if len(idx):
                codes[idx[~manual[idx]]] = code
        pc.classification = codes
        chunk.point_cloud = pc
        labels = chunk.meta.setdefault("classes", {})
        for code, name in CLASS_LABELS.items():
            labels.setdefault(str(code), name)
        result["classified"] = True
    else:
        result["classified"] = False
    chunk.meta["powerlines"] = result
    progress(1.0, t("{0} conduttori, {1} pali", "{0} wires, {1} poles").format(len(result["wires"]), len(result["poles"])))
    return result


def clear_chunk(chunk) -> bool:
    """Forget the powerlines of ``chunk``: the report and the classes 14 and 15 given by the
    plugin (hand labels stay). Returns whether there was something to forget."""
    had = chunk.meta.pop("powerlines", None) is not None
    pc = chunk.point_cloud
    if pc is not None and had:
        manual = pc.manual if pc.manual is not None else np.zeros(len(pc), bool)
        ours = np.isin(pc.classification, (WIRE, TOWER)) & ~manual
        if ours.any():
            pc.classification[ours] = 1
            chunk.point_cloud = pc
    labels = chunk.meta.get("classes") or {}
    for code, name in CLASS_LABELS.items():
        if labels.get(str(code)) == name:
            labels.pop(str(code))
    return had
