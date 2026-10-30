"""The implicit 3D model of the ground: scalar fields fitted by LoopStructural, painted by us.

This file runs only in the worker (:mod:`implicit_worker`), never in the program's process: it
works on plain arrays and knows nothing of the chunk. :func:`run` takes the job (described in
:mod:`implicit_worker`) and the DEM and returns a :class:`Result`.

The fields
    Every series is one scalar field of LoopStructural. A series in ``depth`` space is fitted on
    ``z - DEM(x, y)`` (contacts that follow the ground), one in ``elevation`` space on ``z``. In both
    the vertical is exaggerated by ``k`` (:func:`vertical_scale`), or a layer thinner than an element
    disappears. The value of a contact is the mean height of its points in those coordinates, so the
    field grows upwards at about one unit per scaled metre and ``(value - field) / k`` is a distance
    in metres, positive under the contact. A series of one contact has no gradient of its own: a
    normal is added at the mean of its points (:func:`plane_normal`).

    A cut has a field of its own, in a box of its own around it (:func:`cut_constraints`), positive
    inside. Several fills of one cut are ordered by one more field in the same box. Fields and data
    stay independent of each other.

The painter (:func:`paint`)
    LoopStructural's stratigraphic column is not used. The column is walked from the bottom up
    keeping, for every entry so far, the signed function of the *cumulative* set "this unit and
    those below it":

    * a layer takes what is under its top and nobody has taken yet: ``max(phi, s_unit)``;
    * a cut clears its inside in every set painted so far: ``min(phi_j, -s_cut)``;
    * a fill takes what is inside its cut and under its top: ``max(phi, min(s_fill, s_cut))``;
    * the highest layer takes everything left under the DEM (its top is the ground, whatever its
      contacts say; a fill over it stays inside its cut), and so does the unit at the top of the column.

    So no two units overlap by construction, and every voxel under the DEM belongs to one unit.

Solids (:func:`solids`)
    The function of a unit is ``min(phi_j, -phi_below, ground)``: the function of its top, the
    opposite of the function of the unit under it, and the ground. Two neighbouring units therefore
    change sign on the very same function, with the very same values at the voxels, and marching
    cubes puts their two surfaces at the same position: no gap and no smoothing of labels, which
    would shrink thin units and round the corners of a cut. Outside the box the function is the mirror
    of the last voxel, so a solid closes exactly on the face of the box.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

NELEMENTS = 20000
FOLDS = 12
SEED = 20261030
K_MIN, K_MAX = 1.0, 40.0
ELEMENTS_PER_UNIT = 3  # the thinnest unit spans at least these many elements
MAX_VOXELS = 60_000_000  # float32 arrays of it, one per unit
BIG = 1000.0  # a signed function far from its zero (metres)
BATCH = 2_000_000  # points evaluated at a time
JACKKNIFE_CELLS = 40_000  # columns of the rasters of the variance
COLUMN_SAMPLES = 256  # heights looked at along a column for a crossing
CUT_RING = 24  # points of the ring around a cut known to be outside
EDGE_RAYS = 72  # directions of an estimated edge


def versions() -> dict:
    """The versions of the libraries of the worker, for the report of a run."""
    import importlib.metadata as md
    import platform

    out = {"python": platform.python_version()}
    for key, package in (("loopstructural", "LoopStructural"), ("numpy", "numpy"), ("pandas", "pandas"),
                         ("scikit-image", "scikit-image")):
        try:
            out[key] = md.version(package)
        except Exception:  # noqa: BLE001 - a version is not worth a failure
            out[key] = ""
    return out


# ---------------------------------------------------------------------- the frame
class Grid:
    """The raster of the job: north up, ``(x0, y0)`` the top-left corner, as ``surfaces.Grid``."""

    def __init__(self, spec: dict):
        self.x0, self.y0, self.cell = float(spec["x0"]), float(spec["y0"]), float(spec["cell"])
        self.width, self.height = int(spec["width"]), int(spec["height"])

    @property
    def xs(self) -> np.ndarray:
        return self.x0 + (np.arange(self.width) + 0.5) * self.cell

    @property
    def ys(self) -> np.ndarray:
        return self.y0 - (np.arange(self.height) + 0.5) * self.cell

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.x0, self.y0 - self.height * self.cell, self.x0 + self.width * self.cell, self.y0


class Dem:
    """The DEM read at any point: bilinear on the cell centres, the cells without a value taking
    the nearest one (a borehole just outside the mask keeps its depth, nothing is drawn there)."""

    def __init__(self, grid: Grid, values: np.ndarray):
        from scipy import ndimage

        self.grid = grid
        self.values = np.asarray(values, dtype=np.float64)
        if self.values.shape != (grid.height, grid.width):
            raise ValueError(f"the DEM is {self.values.shape}, the grid {(grid.height, grid.width)}")
        self.mask = np.isfinite(self.values)
        if not self.mask.any():
            raise ValueError("the DEM has no cell with a value")
        index = ndimage.distance_transform_edt(~self.mask, return_distances=False, return_indices=True)
        self.filled = self.values[tuple(index)]

    def at(self, x, y) -> np.ndarray:
        g = self.grid
        col = np.clip((np.asarray(x, dtype=np.float64) - g.x0) / g.cell - 0.5, 0, g.width - 1)
        row = np.clip((g.y0 - np.asarray(y, dtype=np.float64)) / g.cell - 0.5, 0, g.height - 1)
        c0, r0 = np.minimum(col.astype(int), max(g.width - 2, 0)), np.minimum(row.astype(int), max(g.height - 2, 0))
        c1, r1 = np.minimum(c0 + 1, g.width - 1), np.minimum(r0 + 1, g.height - 1)
        fc, fr = col - c0, row - r0
        top = self.filled[r0, c0] * (1 - fc) + self.filled[r0, c1] * fc
        low = self.filled[r1, c0] * (1 - fc) + self.filled[r1, c1] * fc
        return top * (1 - fr) + low * fr

    def slope(self, x, y) -> tuple[np.ndarray, np.ndarray]:
        """dz/dx and dz/dy (east, north) of the bilinear surface around a point."""
        h = self.grid.cell
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        return (self.at(x + h, y) - self.at(x - h, y)) / (2 * h), (self.at(x, y + h) - self.at(x, y - h)) / (2 * h)


def vertical_scale(thinnest: float | None, extent, nelements: int = NELEMENTS) -> float:
    """The exaggeration ``k`` of the vertical that gives the thinnest unit (mean thickness, m)
    :data:`ELEMENTS_PER_UNIT` elements, in a box of ``extent`` (dx, dy, dz in metres).

    Elements are cubes of side ``h = (dx dy dz k / n) ** (1/3)`` in the scaled box, and the unit
    spans ``t k``: from ``t k = 3 h`` follows ``k = sqrt(27 dx dy dz / (n t^3))``."""
    if not thinnest or thinnest <= 0:
        return K_MIN
    dx, dy, dz = (max(float(v), 1e-6) for v in extent)
    k = math.sqrt(ELEMENTS_PER_UNIT ** 3 * dx * dy * dz / (nelements * thinnest ** 3))
    return float(min(max(k, K_MIN), K_MAX))


def scaled_normal(normal, k: float, slope=(0.0, 0.0)) -> np.ndarray:
    """A world normal in the fitted coordinates ``(x, y, (z - D(x, y)) k)``.

    Normals change with the inverse transpose of the Jacobian of the map, not with the map: a plane
    dipping d degrees has a scaled dip of ``atan(k tan d)``, steeper, while multiplying nz by k would
    make it shallower. ``slope`` is (dD/dx, dD/dy), zero in elevation space. The result is a unit vector
    on the same side of the surface as ``normal``."""
    n = np.asarray(normal, dtype=np.float64).reshape(-1, 3)
    out = np.column_stack([n[:, 0] + slope[0] * n[:, 2], n[:, 1] + slope[1] * n[:, 2], n[:, 2] / k])
    length = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.where(length > 0, length, 1.0)


def plane_normal(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The upward normal of the plane through ``points`` (N, 3), the vertical when they do not
    make a plane (fewer than three, or along a line)."""
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 3:
        return np.array([0.0, 0.0, 1.0])
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=np.float64)
    centred = (pts - np.average(pts, axis=0, weights=w)) * np.sqrt(w)[:, None]
    _u, sv, vt = np.linalg.svd(centred, full_matrices=False)
    if sv[1] <= 1e-6 * max(sv[0], 1e-12) or abs(vt[2, 2]) < 1e-6:
        return np.array([0.0, 0.0, 1.0])
    return vt[2] * np.sign(vt[2, 2])


@dataclass
class Result:
    """What :func:`run` returns: ``labels`` (rows, cols, nz; -1 air, else an index of ``units``; ``k``
    = 0 is the lowest voxel); per unit its top (raster), the jackknife variance when the unit is a
    contact of a series, its solid (vertices in world metres, faces) and its volume."""

    units: list
    labels: np.ndarray
    tops: dict = field(default_factory=dict)
    variance: dict = field(default_factory=dict)
    solids: dict = field(default_factory=dict)
    volumes: dict = field(default_factory=dict)
    loo: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)


# ---------------------------------------------------------------------- the data of a job
def contact_table(job: dict) -> list:
    """The contacts as ``(unit, x, y, z, weight, hole)``; a row without a hole has one by its position."""
    out = []
    for row in job.get("contacts") or ():
        unit, x, y, z = row[0], float(row[1]), float(row[2]), float(row[3])
        weight = float(row[4]) if len(row) > 4 and row[4] is not None else 1.0
        hole = row[5] if len(row) > 5 and row[5] is not None else f"{x:.3f},{y:.3f}"
        out.append((str(unit), x, y, z, weight, str(hole)))
    return out


def check_column(job: dict) -> list:
    """The column as ``(unit, kind)`` top-down, after the checks the painter relies on."""
    column = [(str(e["unit"]), str(e.get("kind", "layer"))) for e in job["column"]]
    names = [u for u, _k in column]
    if len(set(names)) != len(names):
        raise ValueError(f"a unit is twice in the column: {names}")
    for unit, kind in column:
        if kind not in ("layer", "cut", "fill"):
            raise ValueError(f"unknown kind of unit {unit}: {kind}")
    if not column or column[0][1] == "cut":
        raise ValueError("the top of the column must be a unit with material, not a cut")
    if not any(kind == "layer" for _u, kind in column):
        raise ValueError("the column has no layer")
    fills = {str(a): str(b) for a, b in (job.get("fills") or {}).items()}
    for unit, kind in column:
        if kind == "fill":
            cut = fills.get(unit)
            if cut is None or cut not in names or dict(column)[cut] != "cut":
                raise ValueError(f"the fill {unit} has no cut in the column")
            if names.index(cut) < names.index(unit):
                raise ValueError(f"the fill {unit} is under its cut {cut} in the column")
    return column


def thinnest_unit(contacts: list, column: list) -> float | None:
    """The smallest mean thickness of a unit, from the holes that see its top and what comes next."""
    order = {u: i for i, (u, _k) in enumerate(column)}
    holes: dict = {}
    for unit, _x, _y, z, _w, hole in contacts:
        if unit in order:
            holes.setdefault(hole, []).append((order[unit], z, unit))
    seen: dict = {}
    for rows in holes.values():
        rows.sort()
        for (_i, z_a, unit), (_j, z_b, _next) in zip(rows, rows[1:]):
            if z_a > z_b:
                seen.setdefault(unit, []).append(z_a - z_b)
    means = [float(np.mean(v)) for v in seen.values()]
    return min(means) if means else None


class Frame:
    """The voxels of the job: ``nz`` layers of ``dz`` from ``zmin``, on the DEM's grid. ``under`` are
    those below the DEM, ``evaluated`` these and the one above each column (for the crossing there)."""

    def __init__(self, job: dict, dem: np.ndarray):
        self.grid = Grid(job["grid"])
        self.dem = Dem(self.grid, dem)
        self.zmin, self.zmax, self.dz = float(job["zmin"]), float(job["zmax"]), float(job["dz"])
        if self.zmax <= self.zmin or self.dz <= 0:
            raise ValueError(f"bad vertical range: {self.zmin} to {self.zmax} by {self.dz}")
        self.nz = max(int(math.ceil((self.zmax - self.zmin) / self.dz - 1e-9)), 1)
        self.shape = (self.grid.height, self.grid.width, self.nz)
        total = self.shape[0] * self.shape[1] * self.shape[2]
        if total > MAX_VOXELS:
            raise MemoryError(f"Unable to allocate {total} voxels: at most {MAX_VOXELS}")
        self.zc = self.zmin + (np.arange(self.nz) + 0.5) * self.dz
        self.under = self.zc[None, None, :] < np.where(self.dem.mask, self.dem.values, -np.inf)[:, :, None]
        self.evaluated = self.under.copy()
        self.evaluated[:, :, 1:] |= self.under[:, :, :-1]

    def points(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """World coordinates of the voxels of ``evaluated``, in the order of a boolean index."""
        r, c, k = np.nonzero(self.evaluated)
        return self.grid.xs[c], self.grid.ys[r], self.zc[k]

    def volume(self, values: np.ndarray, fill: float = -BIG) -> np.ndarray:
        """A 3-D array from the values at the voxels of ``evaluated``."""
        out = np.full(self.shape, fill, dtype=np.float32)
        out[self.evaluated] = values
        return out

    def ground(self) -> np.ndarray:
        """Height under the DEM (m) of every voxel, negative above it and outside the mask."""
        g = self.dem.values[:, :, None] - self.zc[None, None, :]
        return np.where(self.dem.mask[:, :, None], g, -self.grid.cell).astype(np.float32)


# ---------------------------------------------------------------------- the fields
def _geological_model():
    """LoopStructural's class, imported at the first use: the stub of pyvista is the worker's job."""
    from LoopStructural import GeologicalModel

    return GeologicalModel


@dataclass
class Fitted:
    """A solved field: its coordinates, and the value and the gain of its contacts."""

    feature: object
    space: str
    k: float
    dem: Dem
    zlo: float  # the scaled limits of its box: heights are kept inside them
    zhi: float
    values: dict = field(default_factory=dict)
    gain: float = 1.0

    def coordinate(self, x, y, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return (z - self.dem.at(x, y) if self.space == "depth" else z) * self.k

    def evaluate(self, x, y, scaled) -> np.ndarray:
        """The field at scaled heights, those beyond its box at the edge of the box."""
        x, y, scaled = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(scaled))
        pts = np.column_stack([x.ravel(), y.ravel(), np.clip(scaled.ravel(), self.zlo, self.zhi)])
        out = np.empty(len(pts), dtype=np.float64)
        for start in range(0, len(pts), BATCH):
            out[start:start + BATCH] = self.feature.evaluate_value(pts[start:start + BATCH])
        return out.reshape(x.shape)

    def at(self, x, y, z) -> np.ndarray:
        return self.evaluate(x, y, self.coordinate(x, y, z))


def _solve(name: str, rows: list, lo, hi, nelements: int):
    """A foliation of LoopStructural from rows of the data frame (dictionaries), solved."""
    import pandas as pd

    columns = ["X", "Y", "Z", "val", "interface", "nx", "ny", "nz", "gx", "gy", "gz", "w"]
    data = pd.DataFrame(rows, columns=columns, dtype=float)
    data["feature_name"] = name
    model = _geological_model()(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float))
    model.data = data
    feature = model.create_and_add_foliation(name, nelements=int(nelements))
    feature.evaluate_value(np.array([0.5 * (np.asarray(lo, float) + np.asarray(hi, float))]))  # solves it now
    return feature


def _row(x, y, z, val=None, normal=None, gradient=None, weight=1.0, interface=None) -> dict:
    row = {"X": x, "Y": y, "Z": z, "w": weight}
    if val is not None:
        row["val"] = val
    if interface is not None:
        row["interface"] = interface
    if normal is not None:
        row["nx"], row["ny"], row["nz"] = (float(v) for v in normal)
    if gradient is not None:
        row["gx"], row["gy"], row["gz"] = (float(v) for v in gradient)
    return row


def _monotonic(values: dict, order: list, step: float) -> dict:
    """The contact values strictly decreasing down the column: the means of two contacts seen in
    different boreholes can come out inverted, and a field cannot honour that."""
    out, last = {}, None
    for unit in order:
        if unit not in values:
            continue
        out[unit] = values[unit] if last is None else min(values[unit], last - step)
        last = out[unit]
    return out


def series_box(frame: Frame, xy: np.ndarray, scaled: np.ndarray, space: str, k: float):
    """The box of a field, in scaled coordinates: the grid and the data; vertically the data of a
    depth series with a margin (what is below or above belongs to the first and the last contact and
    resolution is not spent there), the range of the job for an elevation series."""
    x0, y0, x1, y1 = frame.grid.bounds
    margin = 2 * frame.grid.cell
    lo = [min(x0, xy[:, 0].min()) - margin, min(y0, xy[:, 1].min()) - margin]
    hi = [max(x1, xy[:, 0].max()) + margin, max(y1, xy[:, 1].max()) + margin]
    if space == "depth":
        span = max(float(scaled.max() - scaled.min()), 4 * frame.dz * k)
        zlo, zhi = min(float(scaled.min()), 0.0) - 0.25 * span, max(float(scaled.max()), 0.0) + 0.1 * span
    else:
        pad = 3 * frame.dz * k
        zlo, zhi = min(frame.zmin * k, float(scaled.min())) - pad, max(frame.zmax * k, float(scaled.max())) + pad
    return lo + [zlo], hi + [zhi]


def fit_series(frame: Frame, spec: dict, contacts: list, orientations: list, order: list, k: float,
               nelements: int) -> Fitted | None:
    """The field of a series from ``contacts`` and ``orientations`` (of any series): the units of
    ``order`` (top-down) in ``spec``; None when none of them has a point.

    Several contacts: every point is asked for the mean height of its contact. Interface constraints
    (one unknown value per contact, a normal for the gradient) were tried on the 18 boreholes of the
    spike and lose: leave-one-out of the rare contact 0.59 m against 0.29 m, of the thick one 1.86 m
    against 1.56 m (x8). One contact: it has nothing to set the size of the gradient but a normal
    and it is asked only to stay one surface: 0.32 m against 0.79 m with a value at x40."""
    space = str(spec.get("space", "elevation"))
    if space not in ("depth", "elevation"):
        raise ValueError(f"unknown space of series {spec.get('name')}: {space}")
    names = {str(u) for u in spec["units"]}
    rows = [c for c in contacts if c[0] in names]
    if not rows:
        return None
    xy = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    z = np.array([r[3] for r in rows], dtype=np.float64)
    weight = np.array([r[4] for r in rows], dtype=np.float64)
    unit_of = np.array([r[0] for r in rows])
    probe = Fitted(None, space, k, frame.dem, 0.0, 0.0)
    scaled = probe.coordinate(xy[:, 0], xy[:, 1], z)
    lo, hi = series_box(frame, xy, scaled, space, k)
    seen = [u for u in order if u in names and (unit_of == u).any()]
    means = {u: float(np.average(scaled[unit_of == u], weights=weight[unit_of == u])) for u in seen}
    values = _monotonic(means, seen, 0.1 * frame.dz * k)
    interface = len(seen) < 2
    data = []
    for i in range(len(rows)):
        if interface:
            data.append(_row(xy[i, 0], xy[i, 1], scaled[i], interface=seen.index(str(unit_of[i])), weight=weight[i]))
        else:
            data.append(_row(xy[i, 0], xy[i, 1], scaled[i], val=values[str(unit_of[i])], weight=weight[i]))
    measured = [o for o in orientations if str(o[0]) in names]
    for o in measured:
        x, y, zo = float(o[1]), float(o[2]), float(o[3])
        slope = frame.dem.slope(np.array([x]), np.array([y])) if space == "depth" else (0.0, 0.0)
        n = scaled_normal(np.array(o[4:7], dtype=np.float64), k, tuple(float(np.ravel(s)[0]) for s in slope))[0]
        at = float(probe.coordinate(np.array([x]), np.array([y]), np.array([zo]))[0])
        # the direction only: the values of two contacts already fix the gradient's size
        data.append(_row(x, y, at, normal=n) if interface else _row(x, y, at, gradient=n))
    if interface and not measured:
        # nothing says which way the field grows: the plane of the points, at their mean
        pts = np.column_stack([xy, scaled])
        data.append(_row(*(float(v) for v in np.average(pts, axis=0, weights=weight)),
                         normal=plane_normal(pts, weight)))
    fitted = Fitted(_solve(str(spec.get("name", "series")), data, lo, hi, nelements), space, k, frame.dem, lo[2], hi[2])
    h = 0.5 * (hi[2] - lo[2]) / max(COLUMN_SAMPLES, 1)
    upper = fitted.evaluate(xy[:, 0], xy[:, 1], scaled + h)
    lower = fitted.evaluate(xy[:, 0], xy[:, 1], scaled - h)
    gain = float(np.nanmedian((upper - lower) / (2 * h)))
    fitted.gain = gain if np.isfinite(gain) and gain > 1e-3 else 1.0
    if interface:
        reached = fitted.evaluate(xy[:, 0], xy[:, 1], scaled)
        values = {u: float(np.nanmean(reached[unit_of == u])) for u in seen}
    fitted.values = values
    return fitted


def crossing(fitted: Fitted, value: float, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """The height (m) where the field takes ``value`` above each point, the highest crossing of the
    column; NaN when it does not cross inside the box."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    heights = np.linspace(fitted.zlo, fitted.zhi, COLUMN_SAMPLES)
    grid = fitted.evaluate(x[:, None], y[:, None], heights[None, :]) - value
    change = np.sign(grid[:, :-1]) * np.sign(grid[:, 1:]) <= 0
    any_change = change.any(axis=1)
    last = COLUMN_SAMPLES - 2 - np.argmax(change[:, ::-1], axis=1)
    rows = np.arange(len(x))
    a, b = grid[rows, last], grid[rows, last + 1]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(a != b, a / (a - b), 0.0)
    scaled = heights[last] + t * (heights[1] - heights[0])
    out = scaled / fitted.k + (fitted.dem.at(x, y) if fitted.space == "depth" else 0.0)
    return np.where(any_change, out, np.nan)


# ---------------------------------------------------------------------- cuts and fills
def densify(points: np.ndarray, step: float, dem: Dem) -> np.ndarray:
    """A closed outline (N, 3) with a point at least every ``step``, the new ones on the ground."""
    if len(points) < 2:
        return points
    out = []
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        n = max(int(math.ceil(float(np.hypot(*(b - a)[:2])) / step)), 1)
        out.extend(a[:2] + (b[:2] - a[:2]) * i / n for i in range(n))
    xy = np.array(out)
    return np.column_stack([xy, dem.at(xy[:, 0], xy[:, 1])])


def estimated_edge(inside: np.ndarray, outside: np.ndarray, dem: Dem, step: float) -> np.ndarray:
    """The edge of a cut nobody drew: from the middle of the holes inside, along each direction, the
    first point nearer to a hole outside than to one inside - halfway between the two, the
    boundary of the nearest hole. A direction with no hole outside stops as far as the holes inside go."""
    from scipy.spatial import cKDTree

    centre = inside.mean(axis=0)
    reach = float(max(np.hypot(*(outside - centre).T).max(), np.hypot(*(inside - centre).T).max(), step))
    own = max(float(np.hypot(*(inside - centre).T).max()), 2 * step)
    angles = 2 * np.pi * np.arange(EDGE_RAYS) / EDGE_RAYS
    directions = np.column_stack([np.cos(angles), np.sin(angles)])
    radii = np.arange(0.0, reach + step, step)
    samples = centre + radii[None, :, None] * directions[:, None, :]
    near_in = cKDTree(inside).query(samples.reshape(-1, 2))[0].reshape(samples.shape[:2])
    near_out = cKDTree(outside).query(samples.reshape(-1, 2))[0].reshape(samples.shape[:2])
    past = near_out < near_in
    first = np.argmax(past, axis=1)
    radius = np.where(past.any(axis=1), radii[first], own)
    xy = centre + radius[:, None] * directions
    return np.column_stack([xy, dem.at(xy[:, 0], xy[:, 1])])


@dataclass
class CutData:
    """The constraints of a cut: ``points`` (x, y, z in metres) with their signed distance from its
    surface, the box in scaled coordinates, its vertical scale, the positions of the holes inside."""

    points: np.ndarray
    values: np.ndarray
    lo: list
    hi: list
    k: float
    seeds: np.ndarray
    edge: str


def _rows3(rows, columns: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(-1, columns) if len(rows or ()) else np.zeros((0, columns))


def cut_constraints(spec: dict, dem: Dem, dz: float, step: float) -> CutData | None:
    """The data of the field of a cut, or None when there is not enough to place it (no base, or no
    edge and no hole outside to estimate one).

    On the surface of the cut the value is zero: the base where the boreholes crossed it, the edge
    on the ground. The rest says on which side a point is, with its distance from those points as the
    value: the column of a hole inside is positive above the base and negative just below; the
    column of a hole outside, and of a ring outside everything else, is negative all the way down.
    The field lives in a box of its own around the cut, close to a cube (its own vertical scale), so
    the elements of a pit are small whatever the size of the site. ``step`` is the spacing of the edge."""
    from scipy.spatial import cKDTree

    base, edge = _rows3(spec.get("base"), 3), _rows3(spec.get("edge"), 3)
    inside, outside = _rows3(spec.get("inside"), 2), _rows3(spec.get("outside"), 2)
    if not len(base):
        return None
    if not len(inside):
        inside = np.unique(base[:, :2], axis=0)
    kind = "drawn"
    if len(edge):
        edge = densify(edge, step, dem)
    elif len(outside):
        edge, kind = estimated_edge(inside, outside, dem, step), "estimated"
    else:
        return None
    zero = np.vstack([base, edge])
    centre = edge[:, :2].mean(axis=0)
    radius = max(float(np.hypot(*(zero[:, :2] - centre).T).max()), 2 * step)
    ground_of_base = dem.at(base[:, 0], base[:, 1])
    depth = max(float((ground_of_base - base[:, 2]).max()), 2 * dz)
    ring_r = 1.2 * radius
    half = 1.4 * radius
    lo_xy, hi_xy = centre - half, centre + half
    near = dem.at(*np.meshgrid(np.linspace(lo_xy[0], hi_xy[0], 15), np.linspace(lo_xy[1], hi_xy[1], 15)))
    zlo, zhi = float(zero[:, 2].min()) - 0.5 * depth, max(float(near.max()), float(zero[:, 2].max())) + 0.4 * depth
    k = float(np.clip(2 * half / (zhi - zlo), K_MIN, K_MAX))

    def scaled(points):
        return np.asarray(points, dtype=np.float64) * [1.0, 1.0, k]

    positive, negative = [], []
    for x, y in inside:
        ground = float(dem.at(np.array([x]), np.array([y]))[0])
        here = base[np.hypot(base[:, 0] - x, base[:, 1] - y) < 1e-6]
        bottom = float(here[:, 2].max()) if len(here) else ground - 0.5 * depth
        positive.extend((x, y, bottom + f * (ground - bottom)) for f in (0.35, 0.65, 1.0, 1.3))
        if len(here):
            negative.append((x, y, bottom - max(2 * dz, 0.15 * depth)))
    angles = 2 * np.pi * np.arange(CUT_RING) / CUT_RING
    ring = centre + ring_r * np.column_stack([np.cos(angles), np.sin(angles)])
    keep = outside[np.all(np.abs(outside - centre) < half, axis=1)] if len(outside) else outside
    for x, y in np.vstack([ring, keep]):
        ground = float(dem.at(np.array([x]), np.array([y]))[0])
        negative.extend((x, y, ground + f * depth) for f in (0.3, 0.0, -0.4, -0.8, -1.2))
    # the outline parts the ground: just out of it is out of the cut, whatever the cut does further down
    for offset, side in ((0.12, negative), (0.3, negative), (0.55, negative), (-0.12, positive)):
        moved = edge[:, :2] + (edge[:, :2] - edge[:, :2].mean(axis=0)) * offset
        side.extend(zip(moved[:, 0], moved[:, 1], dem.at(moved[:, 0], moved[:, 1])))
    positive, negative = np.array(positive), np.array(negative).reshape(-1, 3)
    tree = cKDTree(scaled(zero))
    points = np.vstack([zero, positive, negative])
    values = np.concatenate([np.zeros(len(zero)), tree.query(scaled(positive))[0], -tree.query(scaled(negative))[0]])
    lo, hi = [float(lo_xy[0]), float(lo_xy[1]), zlo * k], [float(hi_xy[0]), float(hi_xy[1]), zhi * k]
    seeds = positive[:, :2]
    return CutData(points, values, lo, hi, k, seeds, kind)


def fit_cut(name: str, data: CutData, dem: Dem, orientations: list, nelements: int) -> Fitted:
    """The field of a cut, positive inside; a measured wall (an orientation of the cut, its normal
    pointing into the cut) adds to the data."""
    rows = [_row(p[0], p[1], p[2] * data.k, val=float(v)) for p, v in zip(data.points, data.values)]
    for o in orientations:
        if str(o[0]) == name:
            n = scaled_normal(np.array(o[4:7], dtype=np.float64), data.k)[0]
            rows.append(_row(float(o[1]), float(o[2]), float(o[3]) * data.k, gradient=n))
    return Fitted(_solve(f"cut {name}", rows, data.lo, data.hi, nelements), "elevation", data.k, dem, data.lo[2],
                  data.hi[2])


def cut_function(frame: Frame, fitted: Fitted, data: CutData, xyz) -> np.ndarray:
    """The signed function of a cut at the evaluated voxels: its field inside the box of the cut,
    negative beyond, without the bits of the field that are not joined to a hole inside."""
    from scipy import ndimage

    x, y, z = xyz
    inside = (x >= data.lo[0]) & (x <= data.hi[0]) & (y >= data.lo[1]) & (y <= data.hi[1])
    out = np.full(len(x), -BIG, dtype=np.float32)
    out[inside] = fitted.at(x[inside], y[inside], z[inside]).astype(np.float32)
    cube = frame.volume(out)
    positive = (cube > 0) & frame.under
    if positive.any():
        labels, count = ndimage.label(positive)
        g = frame.grid
        cols = np.clip(((data.seeds[:, 0] - g.x0) / g.cell).astype(int), 0, g.width - 1)
        rows = np.clip(((g.y0 - data.seeds[:, 1]) / g.cell).astype(int), 0, g.height - 1)
        joined = np.unique(labels[rows, cols, :])
        joined = joined[joined > 0]
        if len(joined):
            drop = np.ones(count + 1, dtype=bool)
            drop[joined] = False
            drop[0] = False
            cube[drop[labels]] = -g.cell
    return cube[frame.evaluated]


def fit_fills(frame: Frame, name: str, fills: list, contacts: list, data: CutData, nelements: int,
              covered: bool = False) -> Fitted | None:
    """The field that orders the ``fills`` of a cut (top-down), None when they have no contact; one
    fill needs it only when a layer lies on it (``covered``), else it has all of the cut."""
    if len(fills) < 2 and not covered:
        return None
    rows = [c for c in contacts if c[0] in fills]
    if not rows:
        return None
    xy = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    scaled = np.array([r[3] for r in rows], dtype=np.float64) * data.k
    weight = np.array([r[4] for r in rows], dtype=np.float64)
    unit_of = np.array([r[0] for r in rows])
    seen = [u for u in fills if (unit_of == u).any()]
    means = {u: float(np.average(scaled[unit_of == u], weights=weight[unit_of == u])) for u in seen}
    values = _monotonic(means, seen, 0.1 * frame.dz * data.k)
    table = [_row(xy[i, 0], xy[i, 1], scaled[i], val=values[str(unit_of[i])], weight=weight[i])
             for i in range(len(rows))]
    if len(seen) < 2:
        table.append(_row(float(xy[:, 0].mean()), float(xy[:, 1].mean()), float(scaled.mean()),
                          normal=plane_normal(np.column_stack([xy, scaled]), weight)))
    feature = _solve(f"fills {name}", table, data.lo, data.hi, nelements)
    return Fitted(feature, "elevation", data.k, frame.dem, data.lo[2], data.hi[2], values=values)


# ---------------------------------------------------------------------- the painter
@dataclass
class Painted:
    """The units from the top of the column down, with the cumulative function of each one:
    ``phi[i]`` > 0 on the voxels of unit i and of those after it in the list."""

    units: list
    phi: list
    labels: np.ndarray


def paint(frame: Frame, column: list, fills: dict, series: dict, cuts: dict, fill_fields: dict, xyz) -> Painted:
    """The units on the voxels (see the head of this file). ``series``: unit -> :class:`Fitted` of its
    series; ``cuts``: cut -> signed function at the evaluated voxels; ``fill_fields``: cut ->
    :class:`Fitted` of its fills. Units without a function take no voxels."""
    x, y, z = xyz
    raw: dict = {}

    def signed(unit: str, fitted: Fitted) -> np.ndarray:
        key = id(fitted)
        if key not in raw:
            raw[key] = fitted.at(x, y, z).astype(np.float32)
        out = (fitted.values[unit] - raw[key]) / (fitted.gain * fitted.k)
        return np.nan_to_num(out, nan=-BIG).astype(np.float32)

    phi = np.full(len(x), -BIG, dtype=np.float32)
    stages: list = []  # (unit, cumulative function), from the bottom of the column
    highest_layer = next(u for u, kind in column if kind == "layer")
    for unit, kind in reversed(column):
        if kind == "cut":
            if unit in cuts:
                against = -cuts[unit]
                phi = np.minimum(phi, against)
                stages = [(u, np.minimum(p, against)) for u, p in stages]
            continue
        if kind == "fill":
            cut = fills[unit]
            if cut not in cuts:
                own = None
            else:
                fitted = fill_fields.get(cut)
                if fitted is None or unit not in fitted.values:
                    own = cuts[cut]  # one fill, or the last: all that is left of the cut
                else:
                    own = np.minimum(signed(unit, fitted), cuts[cut])
        elif unit == highest_layer:
            own = np.full(len(x), BIG, dtype=np.float32)  # its top is the ground: it has all that is left
        else:
            fitted = series.get(unit)
            own = signed(unit, fitted) if fitted is not None and unit in fitted.values else None
        if own is not None:
            phi = np.maximum(phi, own)
        stages.append((unit, phi.copy()))
    stages[-1] = (stages[-1][0], np.full(len(x), BIG, dtype=np.float32))  # the top: all that is left
    stages.reverse()
    units = [u for u, _p in stages]
    index = np.zeros(len(x), dtype=np.int16)
    for i in range(1, len(stages)):
        index[stages[i][1] > 0] = i
    labels = np.full(frame.shape, -1, dtype=np.int16)
    labels[frame.evaluated] = index
    labels[~frame.under] = -1
    return Painted(units, [p for _u, p in stages], labels)


def top_of(frame: Frame, phi: np.ndarray) -> np.ndarray:
    """The top of a cumulative set on the raster: the highest voxel of the set under the DEM, moved
    to the zero of its function between it and the voxel above (at most the DEM). An overhang keeps
    the highest crossing; NaN outside the DEM and where the set stays below the box."""
    cube = frame.volume(phi)
    mask = frame.under & (cube > 0)
    found = mask.any(axis=2)
    last = frame.nz - 1 - np.argmax(mask[:, :, ::-1], axis=2)
    rows, cols = np.indices(found.shape)
    here = cube[rows, cols, last]
    above = cube[rows, cols, np.minimum(last + 1, frame.nz - 1)]
    z = frame.zc[last].copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        crossed = (above <= 0) & (last + 1 < frame.nz)
        z = np.where(crossed, z + frame.dz * here / (here - above), np.inf)
    z = np.minimum(z, frame.dem.filled)
    return np.where(found & frame.dem.mask, z, np.nan).astype(np.float32)


def _mesh(function: np.ndarray, spacing: tuple, origin: tuple) -> tuple[np.ndarray, np.ndarray]:
    """Marching cubes of a 3-D function positive inside; vertices in world metres (``origin`` the
    centre of its first voxel, rows going south), faces outward. Beyond the array the function is
    the mirror of its border, so the surface closes on the faces of the outer voxels."""
    from skimage import measure

    a = np.pad(function, 1, mode="edge")
    shell = np.ones(a.shape, dtype=bool)
    shell[1:-1, 1:-1, 1:-1] = False
    a[shell] = -np.abs(a[shell])
    a[a == 0] = -1e-6  # a voxel exactly on the level makes triangles without area
    verts, faces, _n, _v = measure.marching_cubes(a, level=0.0, allow_degenerate=True, method="lewiner")
    verts = verts - 1.0
    cell, dz = spacing
    world = np.column_stack([origin[0] + verts[:, 1] * cell, origin[1] - verts[:, 0] * cell,
                             origin[2] + verts[:, 2] * dz])
    faces = faces.astype(np.int32)
    if mesh_volume(world, faces) < 0:
        faces = faces[:, ::-1]
    return world, faces


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """The signed volume of a closed mesh (positive with the faces outward)."""
    origin = vertices.mean(axis=0)
    a, b, c = (vertices[faces[:, i]] - origin for i in range(3))
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def solids(frame: Frame, painted: Painted, progress=None) -> dict:
    """One closed mesh per unit that has voxels: ``{unit: (vertices, faces)}``."""
    out = {}
    ground = frame.ground()
    grid = frame.grid
    for i, unit in enumerate(painted.units):
        mine = painted.labels == i
        if not mine.any():
            continue
        spans = []
        for axis in range(3):
            hit = np.nonzero(mine.any(axis=tuple(a for a in range(3) if a != axis)))[0]
            spans.append(slice(max(int(hit[0]) - 1, 0), min(int(hit[-1]) + 2, frame.shape[axis])))
        spans = tuple(spans)
        s = np.minimum(frame.volume(painted.phi[i], 0.0), ground)
        if i + 1 < len(painted.phi):
            s = np.minimum(s, -frame.volume(painted.phi[i + 1], 0.0))
        s = s[spans]
        origin = (grid.x0 + (spans[1].start + 0.5) * grid.cell, grid.y0 - (spans[0].start + 0.5) * grid.cell,
                  float(frame.zc[spans[2].start]))
        out[unit] = _mesh(s, (grid.cell, frame.dz), origin)
        if progress is not None:
            progress((i + 1) / len(painted.units), unit)
    return out


# ---------------------------------------------------------------------- leave-one-out
def fold_groups(holes: list, folds: int, seed: int = SEED) -> list:
    """The holes in at most ``folds`` groups of about the same size, by a seeded shuffle."""
    order = list(sorted(set(holes)))
    if len(order) < 2:
        return []
    np.random.default_rng(seed).shuffle(order)
    return [list(g) for g in np.array_split(np.array(order, dtype=object), min(int(folds), len(order)))]


def _coarse_cells(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Rows and columns of the coarse raster the jackknife is computed on."""
    step = max(int(math.ceil(math.sqrt(grid.width * grid.height / JACKKNIFE_CELLS))), 1)
    rows = np.arange(step // 2, grid.height, step)
    cols = np.arange(step // 2, grid.width, step)
    return rows, cols


def jackknife(surfaces: list) -> np.ndarray:
    """The jackknife variance of the estimates of a raster, from one surface per group."""
    stack = np.array(surfaces, dtype=np.float64)
    good = np.isfinite(stack).sum(axis=0)
    with np.errstate(all="ignore"):
        spread = np.nansum((stack - np.nanmean(stack, axis=0)) ** 2, axis=0)
        out = (good - 1) / np.maximum(good, 1) * spread
    return np.where(good >= 2, out, np.nan)


def _to_full(coarse: np.ndarray, rows: np.ndarray, cols: np.ndarray, grid: Grid, mask: np.ndarray) -> np.ndarray:
    """A coarse raster on the full grid (bilinear), NaN outside the mask."""
    from scipy import ndimage

    known = np.isfinite(coarse)
    if not known.any():
        return np.full(mask.shape, np.nan, dtype=np.float32)
    index = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    filled = coarse[tuple(index)]
    r = np.clip(np.interp(np.arange(grid.height), rows, np.arange(len(rows))), 0, len(rows) - 1)
    c = np.clip(np.interp(np.arange(grid.width), cols, np.arange(len(cols))), 0, len(cols) - 1)
    rr, cc = np.meshgrid(r, c, indexing="ij")
    out = ndimage.map_coordinates(filled, [rr, cc], order=1, mode="nearest")
    return np.where(mask, out, np.nan).astype(np.float32)


def leave_one_out(frame: Frame, series: list, contacts: list, orientations: list, order: list, k: float,
                  nelements: int, folds: int, seed: int, progress=None) -> tuple[dict, dict]:
    """Each group of boreholes left out in turn, the fields fitted again without them: the errors at
    the contacts left out (``{unit: {"rmse", "mae", "n", "nan"}}``), and the jackknife variance of
    every contact on the raster. Cuts and fills stay out of it: they have too few points to lose some."""
    names = {str(u) for s in series for u in s["units"]}
    used = [c for c in contacts if c[0] in names]
    groups = fold_groups([c[5] for c in used], folds, seed)
    if len(groups) < 2:
        return {}, {}
    grid = frame.grid
    rows, cols = _coarse_cells(grid)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    wide_x, wide_y = grid.xs[cc].ravel(), grid.ys[rr].ravel()
    errors: dict = {}
    surfaces: dict = {}
    total = len(groups) * len(series)
    done = 0
    for group in groups:
        out = set(group)
        kept = [c for c in contacts if c[5] not in out]
        held = [c for c in used if c[5] in out]
        for spec in series:
            members = {str(u) for u in spec["units"]}
            fitted = None
            if any(c[0] in members for c in kept):
                try:
                    fitted = fit_series(frame, spec, kept, orientations, order, k, nelements)
                except Exception:  # noqa: BLE001 - a fold that cannot be solved counts as errors missing
                    fitted = None
            for unit in sorted(members):
                mine = [c for c in held if c[0] == unit]
                if fitted is None or unit not in fitted.values:
                    if mine:
                        errors.setdefault(unit, []).extend([np.nan] * len(mine))
                    continue
                value = fitted.values[unit]
                if mine:
                    x = np.array([c[1] for c in mine])
                    y = np.array([c[2] for c in mine])
                    z = np.array([c[3] for c in mine])
                    errors.setdefault(unit, []).extend((crossing(fitted, value, x, y) - z).tolist())
                surfaces.setdefault(unit, []).append(crossing(fitted, value, wide_x, wide_y).reshape(rr.shape))
            done += 1
            if progress is not None:
                progress(done / total)
    loo = {}
    for unit, values in errors.items():
        e = np.array(values, dtype=np.float64)
        good = e[np.isfinite(e)]
        loo[unit] = {"rmse": float(np.sqrt(np.mean(good ** 2))) if len(good) else None,
                     "mae": float(np.mean(np.abs(good))) if len(good) else None, "n": int(len(good)),
                     "nan": int(len(e) - len(good))}
    variance = {u: _to_full(jackknife(s), rows, cols, grid, frame.dem.mask) for u, s in surfaces.items() if len(s) >= 2}
    return loo, variance


# ---------------------------------------------------------------------- the run
def run(job: dict, dem: np.ndarray, progress=None) -> Result:
    """The model of ``job`` on ``dem``; ``progress(fraction, phase, **info)`` says where it is."""
    say = progress or (lambda *_a, **_k: None)
    say(0.0, "fit")
    frame = Frame(job, dem)
    column = check_column(job)
    fills = {str(a): str(b) for a, b in (job.get("fills") or {}).items()}
    contacts = contact_table(job)
    orientations = list(job.get("orientations") or ())
    options = job.get("options") or {}
    nelements = int(options.get("nelements") or NELEMENTS)
    order = [u for u, _k in column]
    series_specs = list(job.get("series") or ())
    if not series_specs:
        raise ValueError("the job has no series")
    for spec in series_specs:
        wrong = [u for u in spec["units"] if u in fills or dict(column).get(u) == "cut" or u not in order]
        if wrong:
            raise ValueError(f"series {spec.get('name')}: {wrong} are not layers of the column")
    warnings: list = []
    x0, y0, x1, y1 = frame.grid.bounds
    k = options.get("vertical")
    if k is None:
        thin = thinnest_unit([c for c in contacts if c[0] not in fills], column)
        k = vertical_scale(thin, (x1 - x0, y1 - y0, frame.zmax - frame.zmin), nelements)
    k = float(k)
    fields: dict = {}
    for i, spec in enumerate(series_specs):
        fitted = fit_series(frame, spec, contacts, orientations, order, k, nelements)
        if fitted is None:
            warnings.append({"code": "series_without_data", "series": str(spec.get("name"))})
            continue
        for unit in spec["units"]:
            fields[str(unit)] = fitted
        say(0.05 + 0.25 * (i + 1) / len(series_specs), "fit")
    layers = [u for u, kind in column if kind == "layer"]
    missing = [u for u in layers[1:] if u not in fields or u not in fields[u].values]
    warnings.extend({"code": "unit_without_contact", "unit": u} for u in missing)
    xyz = frame.points()
    cuts: dict = {}
    fill_fields: dict = {}
    cut_info: dict = {}
    step = frame.grid.cell
    for cut, spec in (job.get("cuts") or {}).items():
        cut = str(cut)
        data = cut_constraints(spec, frame.dem, frame.dz, step)
        if data is None:
            warnings.append({"code": "cut_without_data", "unit": cut})
            cut_info[cut] = {"edge": "none"}
            continue
        fitted = fit_cut(cut, data, frame.dem, orientations, nelements)
        cuts[cut] = cut_function(frame, fitted, data, xyz)
        own = [u for u, _kind in column if fills.get(u) == cut]
        covered = bool(own) and any(kind == "layer" for u, kind in column[:order.index(own[0])])
        ordered = fit_fills(frame, cut, own, contacts, data, nelements, covered)
        if ordered is not None:
            fill_fields[cut] = ordered
        cut_info[cut] = {"edge": data.edge, "vertical": data.k}
        if data.edge == "estimated":
            warnings.append({"code": "edge_estimated", "unit": cut})
    say(0.35, "paint")
    painted = paint(frame, column, fills, fields, cuts, fill_fields, xyz)
    say(0.5, "tops")
    tops = {u: top_of(frame, p) for u, p in zip(painted.units, painted.phi)}
    say(0.55, "solids")
    meshes = solids(frame, painted, lambda f, _unit: say(0.55 + 0.15 * f, "solids"))
    volume_of_voxel = frame.grid.cell ** 2 * frame.dz
    volumes = {u: float((painted.labels == i).sum() * volume_of_voxel) for i, u in enumerate(painted.units)}
    loo: dict = {}
    variance: dict = {}
    folds = options.get("folds")
    if folds:
        say(0.7, "loo")
        loo, variance = leave_one_out(frame, series_specs, contacts, orientations, order, k, nelements,
                                      FOLDS if folds is True else int(folds), int(options.get("seed", SEED)),
                                      lambda f: say(0.7 + 0.28 * f, "loo"))
    info = {"vertical": k, "voxel": [frame.grid.cell, frame.grid.cell, frame.dz], "shape": list(frame.shape),
            "voxels_under_dem": int(frame.under.sum()), "cuts": cut_info, "warnings": warnings,
            "contact_values": {u: f.values[u] for u, f in fields.items() if u in f.values}}
    return Result(painted.units, painted.labels, tops, variance, meshes, volumes, loo, info)
