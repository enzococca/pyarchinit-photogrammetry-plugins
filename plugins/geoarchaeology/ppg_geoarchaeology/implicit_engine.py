"""The implicit 3D model of the ground: scalar fields fitted by LoopStructural, painted by us.

This file runs only in the worker (:mod:`implicit_worker`), never in the program's process: it
works on plain arrays and knows nothing of the chunk. :func:`run` takes the job (described in
:mod:`implicit_worker`) and the DEM and returns a :class:`Result`.

The fields
    Every series is one scalar field of LoopStructural. A series in ``depth`` space is fitted on
    ``z - DEM(x, y)`` (contacts that follow the ground), one in ``elevation`` space on ``z``. In both
    the vertical is exaggerated by ``k`` (:func:`vertical_scale`), or a layer thinner than an element
    disappears. The values of the contacts are as far apart as the boreholes say they are
    (:func:`contact_values`), so the field grows upwards at about one unit per scaled metre and
    ``(value - field) / (gain k)`` is a distance in metres, positive under the contact. A series of one
    contact has no gradient of its own: a normal is added at the mean of its points
    (:func:`plane_normal`), and so for a series whose positions do not make a plane. The box of a field
    holds each contact over the whole grid (:func:`series_box`); beyond it the field goes on along
    its gradient.

    Several fills of one cut are ordered by one more field in a box around the cut. Fields and data stay
    independent of each other.

Cuts
    A cut is not a field: a smooth field cannot draw a thin shape from one to three measures. It is a
    plan, a boolean raster (:func:`cut_plan`), and an explicit floor on it, ``z = ground - s(x, y) P(t)``
    (:func:`floors`): ``t`` the distance from the edge of the plan over the local half width, ``P`` the
    profile of the cut, ``s`` the measured depths over ``P``, spread by inverse distance, so the floor
    passes through its measures. Few holes cannot tell the shape of a floor: the volume of a cut comes
    with the three profiles (``volume_range``). Only a cut with rows marked ``wall``, which bends back
    under the ground, is a field of LoopStructural in a box of its own (:func:`wall_field`).

The painter (:func:`paint`)
    LoopStructural's stratigraphic column is not used. The column is walked from the bottom up
    keeping, for every entry so far, the signed function of the *cumulative* set "this unit and
    those below it":

    * a layer takes what is under its top and nobody has taken yet: ``max(phi, s_unit)``;
    * a cut clears its inside in every set painted so far: ``min(phi_j, -s_cut)``;
    * a fill takes what is inside its cut and under its top: ``max(phi, min(s_fill, s_cut))``; the
      youngest fill of a cut with no layer over it takes all that is left of the cut;
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
COLLINEAR = 0.05  # positions spread across their line less than this share of its length do not make a plane
CUT_ELEMENTS = 120_000  # at most these many elements for the field of a cut with walls
PROFILES = {"flat": lambda t: 1.0 - (1.0 - np.minimum(t / 0.25, 1.0)) ** 2,  # steep walls, flat floor
            "bowl": lambda t: 1.0 - (1.0 - t) ** 3,
            "v": lambda t: t}
RIM_T = 0.25  # holes all nearer the edge than this share of the half width: the floor's depth was never measured
P_MIN = 0.25  # the profile at a hole next to the edge counts as this much: its depth at most x4 in the middle
IDW_BATCH = 100_000  # cells of a plan at a time


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

    def cell_of(self, x, y) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rows and columns of the cells of points, the nearest cell for a point off the raster, and
        whether the point is on the raster."""
        row = np.floor((self.y0 - np.asarray(y, dtype=np.float64)) / self.cell).astype(int)
        col = np.floor((np.asarray(x, dtype=np.float64) - self.x0) / self.cell).astype(int)
        within = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return np.clip(row, 0, self.height - 1), np.clip(col, 0, self.width - 1), within

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


def fit_plane(points: np.ndarray, weights: np.ndarray | None = None) -> tuple[np.ndarray, bool]:
    """``(a, b, c)`` of ``z = a x + b y + c`` through ``points`` (N, 3), and whether their positions
    make a plane. Positions that do not, along a line or at one place, give the horizontal: a small
    difference between two close holes must not tilt a whole site."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    w = np.ones(len(pts)) if weights is None else np.asarray(weights, dtype=np.float64)
    mean = np.average(pts, axis=0, weights=w)
    centred = pts - mean
    level = np.array([0.0, 0.0, mean[2]])
    if len(pts) < 3:
        return level, False
    spread = np.linalg.svd(centred[:, :2] * np.sqrt(w)[:, None], compute_uv=False) / math.sqrt(float(w.sum()))
    if spread[0] < 1e-6 or spread[1] < COLLINEAR * spread[0]:
        return level, False
    root = np.sqrt(w)[:, None]
    (a, b), *_rest = np.linalg.lstsq(centred[:, :2] * root, centred[:, 2:] * root, rcond=None)
    a, b = float(a[0]), float(b[0])
    return np.array([a, b, mean[2] - a * mean[0] - b * mean[1]]), True


def plane_normal(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """The upward normal of :func:`fit_plane`: the vertical when the positions make no plane."""
    (a, b, _c), _plane = fit_plane(points, weights)
    n = np.array([-a, -b, 1.0])
    return n / np.linalg.norm(n)


def _upward(normal) -> tuple[np.ndarray, bool]:
    """A normal pointing up, and whether it had to be turned round."""
    n = np.asarray(normal, dtype=np.float64)
    return (-n, True) if n[2] < 0 else (n, False)


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


def thinnest_unit(contacts: list, column: list, dem: "Dem | None" = None) -> float | None:
    """The smallest mean thickness of a unit, from the holes that see its top and what comes next. A
    thin unit between the ground and the first contact of a hole counts too (``dem``)."""
    order = {u: i for i, (u, _k) in enumerate(column)}
    names = [u for u, _k in column]
    holes: dict = {}
    for unit, x, y, z, _w, hole in contacts:
        if unit in order:
            holes.setdefault(hole, []).append((order[unit], z, unit, x, y))
    seen: dict = {}
    for rows in holes.values():
        rows.sort()
        for (_i, z_a, unit, _x, _y), (_j, z_b, *_rest) in zip(rows, rows[1:]):
            if z_a > z_b:
                seen.setdefault(unit, []).append(z_a - z_b)
        first, z_first, _unit, x, y = rows[0]
        if dem is not None and first > 0:
            depth = float(dem.at(np.array([x]), np.array([y]))[0]) - z_first
            if depth > 0:
                seen.setdefault(names[first - 1], []).append(depth)
    means = [float(np.mean(v)) for v in seen.values()]
    return min(means) if means else None


def column_contradictions(contacts: list, column: list) -> int:
    """How many boreholes show two units in the opposite order of the column."""
    order = {u: i for i, (u, _k) in enumerate(column)}
    holes: dict = {}
    for unit, _x, _y, z, _w, hole in contacts:
        if unit in order:
            holes.setdefault(hole, []).append((order[unit], z))
    count = 0
    for rows in holes.values():
        rows.sort()
        heights = [z for _i, z in rows]
        if any(a < b - 1e-9 for a, b in zip(heights, heights[1:])):
            count += 1
    return count


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
    zlo: float  # the scaled limits of its box
    zhi: float
    values: dict = field(default_factory=dict)
    gain: float = 1.0  # growth per scaled metre; 0 for a cut, which does not grow upwards

    def coordinate(self, x, y, z) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        return (z - self.dem.at(x, y) if self.space == "depth" else z) * self.k

    def evaluate(self, x, y, scaled) -> np.ndarray:
        """The field at scaled heights; beyond its box it goes on from the edge of the box along its
        gradient, so a voxel out of the box stays on the side of the contacts it is on."""
        x, y, scaled = np.broadcast_arrays(np.asarray(x, np.float64), np.asarray(y, np.float64), np.asarray(scaled))
        kept = np.clip(scaled.ravel(), self.zlo, self.zhi)
        pts = np.column_stack([x.ravel(), y.ravel(), kept])
        out = np.empty(len(pts), dtype=np.float64)
        for start in range(0, len(pts), BATCH):
            out[start:start + BATCH] = self.feature.evaluate_value(pts[start:start + BATCH])
        return (out + self.gain * (scaled.ravel() - kept)).reshape(x.shape)

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


def contact_values(seen: list, unit_of: np.ndarray, scaled: np.ndarray, holes: np.ndarray, planes: dict,
                   centres: dict) -> tuple[dict, list]:
    """The value of each contact: how far apart the contacts are, measured in the boreholes that see two
    of them, by least squares, so that a contact seen in few holes keeps its distance from its
    neighbours wherever those holes are (means over different boreholes mix the distance with the
    dip of the site). Returns the values and the contacts that share no hole with the main group: those
    are placed by the plane of the best-seen contact, at their own position."""
    index = {u: i for i, u in enumerate(seen)}
    measured: dict = {}
    for hole in np.unique(holes):
        here = holes == hole
        heights = sorted((index[u], float(np.mean(scaled[here & (unit_of == u)]))) for u in set(unit_of[here].tolist()))
        for (a, z_a), (b, z_b) in zip(heights, heights[1:]):
            measured.setdefault((a, b), []).append(z_a - z_b)
    # a distance that stays the same from hole to hole counts more than one that wanders: else a thick,
    # erratic unit moves the contact next to it (Sultan: 0.63 m of leave-one-out on US 3 against 0.33 m)
    spread = {pair: float(np.var(d)) / len(d) for pair, d in measured.items() if len(d) > 1}
    usual = float(np.median(list(spread.values()))) if spread else 1.0
    least = max(1e-6 * (float(np.ptp(scaled)) ** 2 + 1.0), 1e-9)
    pairs = [(a, b, float(np.mean(d)), 1.0 / math.sqrt(max(spread.get((a, b), usual), least)))
             for (a, b), d in sorted(measured.items())]
    group = list(range(len(seen)))

    def find(i: int) -> int:
        while group[i] != i:
            i = group[i]
        return i

    for a, b, _d, _w in pairs:
        group[find(a)] = find(b)
    roots = [find(i) for i in range(len(seen))]
    counts = [int((unit_of == u).sum()) for u in seen]
    size = {r: sum(c for c, g in zip(counts, roots) if g == r) for r in set(roots)}
    main = max(size, key=lambda r: (size[r], -min(i for i, g in enumerate(roots) if g == r)))
    members = [i for i, g in enumerate(roots) if g == main]
    anchor = max(members, key=lambda i: counts[i])
    rows, right = [], []
    for a, b, d, w in pairs:
        if roots[a] == main:
            row = np.zeros(len(seen))
            row[a], row[b] = w, -w
            rows.append(row)
            right.append(w * d)
    row = np.zeros(len(seen))
    row[anchor] = 1.0
    rows.append(row)
    fixed = float(np.mean(scaled[unit_of == seen[anchor]]))
    right.append(fixed)
    solved = np.linalg.lstsq(np.array(rows), np.array(right), rcond=None)[0]
    values = {seen[i]: float(solved[i]) for i in members}
    apart = [seen[i] for i in range(len(seen)) if i not in members]
    a, b, c = planes[seen[anchor]]
    for unit in apart:
        cx, cy = centres[unit]
        values[unit] = fixed + float(np.mean(scaled[unit_of == unit])) - (a * cx + b * cy + c)
    return values, apart


def series_box(frame: Frame, xy: np.ndarray, scaled: np.ndarray, planes: dict, space: str, k: float):
    """The box of a field, in scaled coordinates: the grid and the data, and vertically each contact
    (its plane, ``planes``) over the whole grid with a margin, within the range of the job. Returns the box
    and the contacts that leave the range of the job somewhere on the grid."""
    x0, y0, x1, y1 = frame.grid.bounds
    margin = 2 * frame.grid.cell
    lo = [min(x0, xy[:, 0].min()) - margin, min(y0, xy[:, 1].min()) - margin]
    hi = [max(x1, xy[:, 0].max()) + margin, max(y1, xy[:, 1].max()) + margin]
    gx, gy = np.meshgrid(np.linspace(x0, x1, 21), np.linspace(y0, y1, 21))
    ground = frame.dem.at(gx, gy) if space == "depth" else np.zeros(gx.shape)
    bottom, top = (frame.zmin - ground) * k, (frame.zmax - ground) * k  # the job's range, in these coordinates
    low, high = float(scaled.min()), float(scaled.max())
    leaving = []
    for unit, (a, b, c) in planes.items():
        height = a * gx + b * gy + c
        if (height < bottom).any() or (height > top).any():
            leaving.append(unit)
        low = min(low, float(np.maximum(height, bottom).min()))
        high = max(high, float(np.minimum(height, top).max()))
    pad = max(3 * frame.dz * k, 0.1 * (high - low))
    zlo = min(max(low - pad, float(bottom.min())), float(scaled.min()) - frame.dz * k)
    zhi = max(min(high + pad, float(top.max())), float(scaled.max()) + frame.dz * k)
    return lo + [zlo], hi + [zhi], leaving


def fit_series(frame: Frame, spec: dict, contacts: list, orientations: list, order: list, k: float,
               nelements: int, notes: list | None = None) -> Fitted | None:
    """The field of a series from ``contacts`` and ``orientations`` (of any series): the units of
    ``order`` (top-down) in ``spec``; None when none of them has a point. ``notes`` collects the warnings.

    Several contacts: every point is asked for the value of its contact (:func:`contact_values`).
    Interface constraints (one unknown value per contact, a normal for the gradient) were tried on the
    18 boreholes of the spike and lose: leave-one-out of the rare contact 0.59 m against 0.29 m, of the
    thick one 1.86 m against 1.56 m (x8). One contact: it has nothing to set the size of the gradient
    but a normal and it is asked only to stay one surface: 0.32 m against 0.79 m with a value at x40.
    Positions that do not make a plane say nothing about the tilt across them: a normal says it,
    as for one contact."""
    space = str(spec.get("space", "elevation"))
    if space not in ("depth", "elevation"):
        raise ValueError(f"unknown space of series {spec.get('name')}: {space}")
    notes = notes if notes is not None else []
    names = {str(u) for u in spec["units"]}
    rows = [c for c in contacts if c[0] in names]
    if not rows:
        return None
    xy = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    z = np.array([r[3] for r in rows], dtype=np.float64)
    weight = np.array([r[4] for r in rows], dtype=np.float64)
    unit_of = np.array([r[0] for r in rows])
    holes = np.array([r[5] for r in rows])
    probe = Fitted(None, space, k, frame.dem, 0.0, 0.0)
    scaled = probe.coordinate(xy[:, 0], xy[:, 1], z)
    seen = [u for u in order if u in names and (unit_of == u).any()]
    measured = []
    for o in orientations:
        if str(o[0]) not in names:
            continue
        x, y, zo = float(o[1]), float(o[2]), float(o[3])
        normal, turned = _upward(np.array(o[4:7], dtype=np.float64))
        if turned:
            notes.append({"code": "normal_turned_up", "unit": str(o[0])})
        slope = frame.dem.slope(np.array([x]), np.array([y])) if space == "depth" else (0.0, 0.0)
        n = scaled_normal(normal, k, tuple(float(np.ravel(s)[0]) for s in slope))[0]
        measured.append((str(o[0]), x, y, float(probe.coordinate(np.array([x]), np.array([y]), np.array([zo]))[0]), n))
    pts = np.column_stack([xy, scaled])
    _whole, positions_plane = fit_plane(np.column_stack([xy, np.zeros(len(xy))]), weight)
    planes, centres = {}, {}
    for unit in seen:
        mine = unit_of == unit
        plane, is_plane = fit_plane(pts[mine], weight[mine])
        centres[unit] = np.average(xy[mine], axis=0, weights=weight[mine])
        given = [m for m in measured if m[0] == unit] or ([] if is_plane else measured)
        if given and abs(float(np.mean([m[4][2] for m in given]))) > 1e-6:
            n = np.mean([m[4] for m in given], axis=0)
            a, b = -n[0] / n[2], -n[1] / n[2]
            plane = np.array([a, b, float(np.average(pts[mine, 2], weights=weight[mine]))
                              - a * centres[unit][0] - b * centres[unit][1]])
        planes[unit] = plane
    lo, hi, leaving = series_box(frame, xy, scaled, planes, space, k)
    notes.extend({"code": "contact_leaves_box", "unit": u} for u in leaving)
    interface = len(seen) < 2
    values: dict = {}
    if not interface:
        values, apart = contact_values(seen, unit_of, scaled, holes, planes, centres)
        notes.extend({"code": "contact_shares_no_hole", "unit": u} for u in apart)
        values = _monotonic(values, seen, 0.1 * frame.dz * k)
    data = []
    for i in range(len(rows)):
        if interface:
            data.append(_row(xy[i, 0], xy[i, 1], scaled[i], interface=0, weight=weight[i]))
        else:
            data.append(_row(xy[i, 0], xy[i, 1], scaled[i], val=values[str(unit_of[i])], weight=weight[i]))
    for _unit, x, y, at, n in measured:
        # the direction only: the values of two contacts already fix the gradient's size
        data.append(_row(x, y, at, normal=n) if interface else _row(x, y, at, gradient=n))
    if not positions_plane and not measured:
        notes.append({"code": "few_holes", "series": str(spec.get("name", "series"))})
    if not measured and (interface or not positions_plane):
        # nothing says which way the field grows or leans: the plane of the points, at their mean,
        # and across the grid when they are not a plane themselves, only away from the holes: on a
        # transect a pin on the line would have the weight of the data
        x0, y0, x1, y1 = frame.grid.bounds
        clear = 0.25 * min(x1 - x0, y1 - y0)
        across = [(a, b) for a in (x0, 0.5 * (x0 + x1), x1) for b in (y0, 0.5 * (y0 + y1), y1)
                  if np.hypot(xy[:, 0] - a, xy[:, 1] - b).min() >= clear]
        for unit in seen:
            mine = unit_of == unit
            normal = plane_normal(pts[mine], weight[mine])
            data.append(_row(*(float(v) for v in np.average(pts[mine], axis=0, weights=weight[mine])), normal=normal))
            if not positions_plane:
                a, b, c = planes[unit]
                for px, py in across:
                    height = float(np.clip(a * px + b * py + c, lo[2], hi[2]))
                    data.append(_row(px, py, height, normal=normal))
                    data.append(_row(px, py, height, interface=0) if interface else
                                _row(px, py, height, val=values[unit]))
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
def _rows(rows, columns: int) -> np.ndarray:
    return np.asarray(rows, dtype=np.float64).reshape(-1, columns) if len(rows or ()) else np.zeros((0, columns))


def split_base(rows) -> tuple[np.ndarray, np.ndarray]:
    """The rows of a cut's ``base`` as (N, 3) heights and a boolean array of those marked ``wall``."""
    rows = list(rows or ())
    points = _rows([r[:3] for r in rows], 3)
    return points, np.array([len(r) > 3 and r[3] == "wall" for r in rows], dtype=bool)


def half_cell_lattice(mask: np.ndarray) -> np.ndarray:
    """The plan on a lattice of half cells (``2 rows + 1`` x ``2 cols + 1``): cell centres at odd
    indices, and between them the points that have only cells of the plan around them. The border of the
    raster is not an edge: a plan that leaves the raster goes on past it."""
    m = np.pad(mask, 1, mode="edge")
    out = np.zeros((2 * m.shape[0] + 1, 2 * m.shape[1] + 1), dtype=bool)
    out[1::2, 1::2] = m
    out[1::2, 2:-1:2] = m[:, :-1] & m[:, 1:]
    out[2:-1:2, 1::2] = m[:-1, :] & m[1:, :]
    out[2:-1:2, 2:-1:2] = m[:-1, :-1] & m[:-1, 1:] & m[1:, :-1] & m[1:, 1:]
    return out[2:-2, 2:-2]


def edge_and_width(mask: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """On the lattice of :func:`half_cell_lattice`: the distance (m) from the edge of the plan, and the
    local half width: the radius of the widest circle on a ridge of the distance (its local maxima) that
    holds the point, else the width at the nearest point that has one. So an elbow sets the width of the
    elbow, not of the whole ditch, and a hand-drawn outline keeps its width: its small bumps are ridges
    too, and the nearest ridge would be one of them for most of the rim. The half cells keep both right
    on a ditch with an even number of cells: its axis is between two rows of cells."""
    from scipy import ndimage

    lattice = half_cell_lattice(mask)
    if lattice.all():
        return np.full(lattice.shape, BIG), np.full(lattice.shape, BIG)
    reach = ndimage.distance_transform_edt(lattice)
    rows, cols = np.nonzero(lattice & (reach >= ndimage.maximum_filter(reach, size=3)))
    width = np.zeros(lattice.shape)
    for i in np.argsort(-reach[rows, cols], kind="stable"):
        row, col = rows[i], cols[i]
        if width[row, col] > 0:
            continue
        radius = reach[row, col]
        n = int(radius)
        top, left = max(row - n, 0), max(col - n, 0)
        ys, xs = np.ogrid[top - row:min(row + n + 1, lattice.shape[0]) - row,
                          left - col:min(col + n + 1, lattice.shape[1]) - col]
        window = width[top:top + ys.size, left:left + xs.size]
        window[(ys ** 2 + xs ** 2 <= radius ** 2) & (window == 0)] = radius
    index = ndimage.distance_transform_edt(width == 0, return_distances=False, return_indices=True)
    return reach * 0.5 * cell, width[tuple(index)] * 0.5 * cell


@dataclass
class Plan:
    """A cut in plan: ``mask`` its cells on the grid, ``t`` (lattice) the distance from its edge over
    the local half width, up to 1, ``signed`` the distance (m) from its edge at the cells, positive
    inside; ``edge`` ``drawn`` or ``estimated``."""

    mask: np.ndarray
    t: np.ndarray
    signed: np.ndarray
    edge: str

    @classmethod
    def of(cls, mask: np.ndarray, cell: float, edge: str) -> "Plan":
        from scipy import ndimage

        distance, width = edge_and_width(mask, cell)
        t = np.minimum(distance / np.maximum(width, 1e-9), 1.0)
        out = ndimage.distance_transform_edt(~mask) * cell - 0.5 * cell
        return cls(mask, t, np.where(mask, distance[1::2, 1::2], -out).astype(np.float32), edge)

    @property
    def cell_t(self) -> np.ndarray:
        return self.t[1::2, 1::2]


def estimated_plan(inside: np.ndarray, outside: np.ndarray, grid: Grid, deepest: float) -> tuple[np.ndarray, float]:
    """The cells of a cut nobody drew, and the limit used: those nearer to the holes inside than to those
    outside, within the median spacing of the holes inside. A single hole has no spacing: the limit is
    how far the nearest hole outside is, or with no hole at all the depth (nothing else says it)."""
    from scipy.spatial import cKDTree

    holes = np.unique(inside, axis=0)
    tree = cKDTree(holes)
    if len(holes) > 1:
        limit = float(np.median(tree.query(holes, k=2)[0][:, 1]))
    elif len(outside):
        limit = float(cKDTree(outside).query(holes)[0].min())
    else:
        limit = max(deepest, 2 * grid.cell)
    cx, cy = np.meshgrid(grid.xs, grid.ys)
    cells = np.column_stack([cx.ravel(), cy.ravel()])
    near_in = tree.query(cells, distance_upper_bound=limit * (1 + 1e-9))[0]
    keep = np.isfinite(near_in)
    if len(outside):
        keep &= near_in < cKDTree(outside).query(cells)[0]
    mask = keep.reshape(cx.shape)
    rows, cols, within = grid.cell_of(holes[:, 0], holes[:, 1])
    mask[rows[within], cols[within]] = True
    return mask, limit


def cut_plan(spec: dict, grid: Grid, inside: np.ndarray, outside: np.ndarray, deepest: float) -> tuple[Plan, float | None]:
    """The plan of a cut: the raster of ``outline`` (a ``.npy`` on the grid, or the array), else the
    estimate; with the limit of the estimate. ``None`` as limit: it was drawn."""
    outline = spec.get("outline")
    if outline is None:
        mask, limit = estimated_plan(inside, outside, grid, deepest)
        return Plan.of(mask, grid.cell, "estimated"), limit
    mask = np.load(outline, allow_pickle=False) if isinstance(outline, str) else np.asarray(outline)
    if mask.shape != (grid.height, grid.width):
        raise ValueError(f"the outline is {mask.shape}, the grid {(grid.height, grid.width)}")
    return Plan.of(mask.astype(bool), grid.cell, "drawn"), None


@dataclass
class BaseRows:
    """The rows of a floor on a plan: where they count (``x``, ``y``: the row, or the nearest plan cell
    for a row past the edge), their cell, their ``t`` and depth, the distance past the edge."""

    x: np.ndarray
    y: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    t: np.ndarray
    depth: np.ndarray
    past: np.ndarray


def base_rows(base: np.ndarray, plan: Plan, grid: Grid, dem: Dem) -> BaseRows:
    from scipy import ndimage

    x, y = base[:, 0].copy(), base[:, 1].copy()
    rows, cols, within = grid.cell_of(x, y)
    nearest = ndimage.distance_transform_edt(~plan.mask, return_distances=False, return_indices=True)
    in_plan = plan.mask[rows, cols] & within
    near_r, near_c = nearest[0][rows, cols], nearest[1][rows, cols]
    lat_r = np.clip(np.rint(2 * (grid.y0 - y) / grid.cell).astype(int), 0, plan.t.shape[0] - 1)
    lat_c = np.clip(np.rint(2 * (x - grid.x0) / grid.cell).astype(int), 0, plan.t.shape[1] - 1)
    t = np.where(in_plan, plan.t[lat_r, lat_c], plan.cell_t[near_r, near_c])
    depth = np.maximum(dem.at(x, y) - base[:, 2], 0.0)
    cx, cy = grid.xs[near_c], grid.ys[near_r]
    past = np.where(in_plan, 0.0, np.maximum(np.hypot(x - cx, y - cy) - 0.5 * grid.cell, 0.0))
    return BaseRows(np.where(in_plan, x, cx), np.where(in_plan, y, cy), near_r, near_c, t, depth, past)


def floors(plan: Plan, table: BaseRows, grid: Grid, dem: Dem, base: np.ndarray) -> tuple[dict, dict]:
    """The depth (m, raster) of the floor for every profile, and which rows it does not reach.

    ``depth = s(x, y) P(t)``: the scale of a row is its depth over the profile at its place, with the
    profile counted as at least :data:`P_MIN` (a hole beside the edge says little of the middle: the
    floor there is a guess and ``base_near_edge`` says so); between the rows it is spread by inverse
    distance. A row not held by that limit is met exactly at its cell."""
    shares = {name: profile(table.t) for name, profile in PROFILES.items()}
    wanted = {name: table.depth / np.maximum(share, P_MIN) for name, share in shares.items()}
    held = {name: shares[name] < P_MIN for name in PROFILES}
    scales = np.column_stack([wanted[name] for name in PROFILES])
    rows, cols = np.nonzero(plan.mask)
    s = np.empty((len(rows), len(PROFILES)))
    for start in range(0, len(rows), IDW_BATCH):
        r, c = rows[start:start + IDW_BATCH], cols[start:start + IDW_BATCH]
        d2 = (grid.xs[c][:, None] - table.x) ** 2 + (grid.ys[r][:, None] - table.y) ** 2
        weight = 1.0 / np.maximum(d2, 1e-9)
        s[start:start + IDW_BATCH] = (weight @ scales) / weight.sum(axis=1, keepdims=True)
    t = plan.cell_t[rows, cols]
    out = {}
    met = table.past == 0.0
    for i, (name, profile) in enumerate(PROFILES.items()):
        depth = np.zeros(plan.mask.shape)
        depth[rows, cols] = s[:, i] * profile(t)
        exact = met & ~held[name]
        depth[table.rows[exact], table.cols[exact]] = (dem.filled[table.rows, table.cols] - base[:, 2])[exact]
        out[name] = np.maximum(depth, 0.0)
    return out, held


def thinned(xy: np.ndarray, spacing: float) -> np.ndarray:
    """One point (N, 2) per square of side ``spacing``."""
    if not len(xy):
        return xy
    _keys, first = np.unique(np.floor(xy / spacing).astype(np.int64), axis=0, return_index=True)
    return xy[np.sort(first)]


@dataclass
class Box:
    """The box of a field in its coordinates (``z`` scaled by ``k``) and its elements."""

    lo: list
    hi: list
    k: float
    elements: int


def plan_box(plan: Plan, depth: np.ndarray, frame: Frame, rows: np.ndarray, nelements: int) -> Box:
    """The box of the field that orders the fills of a cut: around the plan, from the floor to the
    ground, and around ``rows`` (x, y, z), its depth spanning six elements."""
    grid = frame.grid
    r, c = np.nonzero(plan.mask)
    ground = frame.dem.filled[r, c]
    points = np.vstack([np.column_stack([grid.xs[c], grid.ys[r], ground - depth[r, c]]),
                        np.column_stack([grid.xs[c], grid.ys[r], ground]), rows])
    lo = points.min(axis=0) - [2 * grid.cell, 2 * grid.cell, 2 * frame.dz]
    hi = points.max(axis=0) + [2 * grid.cell, 2 * grid.cell, 2 * frame.dz]
    size = hi - lo
    k = float(np.clip(math.sqrt(216 * size[0] * size[1] / (nelements * size[2] ** 2)), K_MIN, K_MAX))
    return Box([float(lo[0]), float(lo[1]), float(lo[2]) * k], [float(hi[0]), float(hi[1]), float(hi[2]) * k], k, nelements)


def plan_function(frame: Frame, plan: Plan, depth: np.ndarray) -> np.ndarray:
    """The signed function of a cut at the evaluated voxels: positive inside the plan above the floor."""
    floor = np.where(plan.mask, frame.dem.filled - depth, -BIG)  # past the edge only the edge counts: a wall stands on it
    cube = (frame.zc[None, None, :] - floor[:, :, None]).astype(np.float32)
    cube[cube == 0] = -1e-6  # a voxel exactly on the floor is under it: zero would belong to nobody
    return np.minimum(cube, plan.signed[:, :, None])[frame.evaluated]


def wall_field(frame: Frame, name: str, plan: Plan, rows: np.ndarray, inside: np.ndarray, orientations: list,
               nelements: int) -> tuple[Fitted, Box, np.ndarray]:
    """The field of a cut with wall rows, positive inside, in a box of its own, and the positions known
    inside: only a field can bend back under the ground. On its surface the value is zero: the rows,
    and the ground along the edge of the plan. The rest says on which side a point is, with its distance
    from those points as the value: on the ground rings at fixed distances on both sides of the edge,
    following its shape; the columns of the holes inside, positive above their base and negative just
    below it; the columns of rings past the reach of the rows, negative all the way down. A measured wall (an orientation of the cut, its normal pointing into the cut) adds to the data.
    Elements a third of the width, the depth spanning six of them, at most :data:`CUT_ELEMENTS`."""
    from scipy.spatial import cKDTree

    grid, dem, dz = frame.grid, frame.dem, frame.dz
    cell = grid.cell
    inner, outer = np.where(plan.mask, plan.signed, -1.0), np.where(plan.mask, -1.0, -plan.signed)

    def ring(distances, distance, spacing):
        r, c = np.nonzero(np.abs(distances - distance) <= 0.75 * cell)
        return thinned(np.column_stack([grid.xs[c], grid.ys[r]]), spacing)

    def on_ground(xy, lift=0.0):
        return np.column_stack([xy, dem.at(xy[:, 0], xy[:, 1]) + lift])

    r, c = np.nonzero((inner > 0) & (inner <= cell))
    edge = on_ground(thinned(np.column_stack([grid.xs[c], grid.ys[r]]), cell))
    zero = np.vstack([rows, edge])
    row, col, _within = grid.cell_of(rows[:, 0], rows[:, 1])
    passes = float(outer[row, col].max())
    reach = passes if passes > cell else 0.0
    widest = float(inner.max())
    margin = max(4 * cell, min(2 * widest, 0.25 * float(np.ptp(zero[:, :2], axis=0).max())))
    depth = max(float((dem.at(rows[:, 0], rows[:, 1]) - rows[:, 2]).max()), 2 * dz)
    low, high = zero[:, :2].min(axis=0) - margin, zero[:, :2].max(axis=0) + margin
    gx, gy = np.meshgrid(np.linspace(low[0], high[0], 15), np.linspace(low[1], high[1], 15))
    around = dem.at(gx, gy)
    lo = np.array([low[0], low[1], min(float(zero[:, 2].min()) - 0.5 * depth, float(around.min()) - 1.25 * depth)])
    hi = np.array([high[0], high[1], float(around.max()) + 0.4 * depth])
    size = hi - lo
    area = float(size[0] * size[1])
    wanted = min(2 * widest / 3.0, (area * size[2] / nelements) ** (1 / 3))
    k = float(np.clip(6 * wanted / depth, K_MIN, K_MAX))
    count = int(np.clip(area * size[2] * k / wanted ** 3, nelements, max(nelements, CUT_ELEMENTS)))
    spacing = max((area * size[2] * k / count) ** (1 / 3), cell)
    near_in = ring(inner, min(0.2 * margin, 0.5 * widest), spacing)
    positive = [on_ground(near_in), on_ground(ring(inner, 0.5 * widest, spacing))]
    negative = [np.zeros((0, 3))]
    for share in (0.25, 0.5, 0.9):
        negative.append(on_ground(ring(outer, share * margin, spacing)))
        down = ring(outer, reach + share * margin, spacing)
        negative.extend(on_ground(down, f * depth) for f in (0.3, -0.4, -0.8, -1.2))
    for x, y in inside:
        top = float(dem.at(np.array([x]), np.array([y]))[0])
        here = rows[np.hypot(rows[:, 0] - x, rows[:, 1] - y) < 1e-6]
        bottom = float(here[:, 2].max()) if len(here) else top - 0.5 * depth
        positive.append(np.array([(x, y, bottom + f * (top - bottom)) for f in (0.35, 0.65, 1.0, 1.3)]))
        if len(here):
            negative.append(np.array([(x, y, bottom - max(2 * dz, 0.15 * depth))]))
    positive, negative = np.vstack(positive), np.vstack(negative)
    stretch = np.array([1.0, 1.0, k])
    tree = cKDTree(zero * stretch)
    points = np.vstack([zero, positive, negative])
    values = np.concatenate([np.zeros(len(zero)), tree.query(positive * stretch)[0], -tree.query(negative * stretch)[0]])
    data = [_row(p[0], p[1], p[2] * k, val=float(v)) for p, v in zip(points, values)]
    for o in orientations:
        if str(o[0]) == name:
            n = scaled_normal(np.array(o[4:7], dtype=np.float64), k)[0]
            data.append(_row(float(o[1]), float(o[2]), float(o[3]) * k, gradient=n))
    box = Box([float(lo[0]), float(lo[1]), float(lo[2]) * k], [float(hi[0]), float(hi[1]), float(hi[2]) * k], k, count)
    fitted = Fitted(_solve(f"cut {name}", data, box.lo, box.hi, count), "elevation", k, dem, box.lo[2], box.hi[2], gain=0.0)
    return fitted, box, positive[:, :2]


def wall_misses(fitted: Fitted, rows: np.ndarray, reach: float) -> list:
    """How far the surface of a field passes from each row: the zero nearest to it along its column,
    within ``reach`` metres; None without one."""
    heights = np.linspace(-reach, reach, 121)
    grid = fitted.at(rows[:, 0:1], rows[:, 1:2], rows[:, 2:3] + heights[None, :])
    misses: list = []
    for row in grid:
        where = np.nonzero(np.sign(row[:-1]) * np.sign(row[1:]) <= 0)[0]
        t = row[where] / (row[where] - row[where + 1] + 1e-30)
        misses.append(float(np.min(np.abs(heights[where] + t * (heights[1] - heights[0])))) if len(where) else None)
    return misses


def wall_function(frame: Frame, fitted: Fitted, box: Box, seeds: np.ndarray, xyz) -> np.ndarray:
    """The signed function of a field at the evaluated voxels: the field inside its box, negative beyond
    it (under the box too: a field says nothing there), without the bits not joined to a place known
    inside."""
    from scipy import ndimage

    x, y, z = xyz
    inside = (x >= box.lo[0]) & (x <= box.hi[0]) & (y >= box.lo[1]) & (y <= box.hi[1]) & (z >= box.lo[2] / box.k)
    out = np.full(len(x), -BIG, dtype=np.float32)
    out[inside] = fitted.at(x[inside], y[inside], z[inside]).astype(np.float32)
    cube = frame.volume(out)
    positive = (cube > 0) & frame.under
    if positive.any():
        labels, count = ndimage.label(positive)
        rows, cols, _within = frame.grid.cell_of(seeds[:, 0], seeds[:, 1])
        joined = np.unique(labels[rows, cols, :])
        joined = joined[joined > 0]
        if len(joined):
            drop = np.ones(count + 1, dtype=bool)
            drop[joined] = False
            drop[0] = False
            cube[drop[labels]] = -frame.grid.cell
    return cube[frame.evaluated]


@dataclass
class Cut:
    """What a cut gives the painter: its signed function, the box of its fills, its report."""

    function: np.ndarray
    box: Box
    info: dict


def build_cut(frame: Frame, name: str, spec: dict, fill_rows: np.ndarray, orientations: list, nelements: int,
              xyz, notes: list) -> Cut | None:
    """A cut from its entry in the job, None when there is not enough to place it. The floor is explicit
    on the plan (:func:`floors`) unless rows marked ``wall`` ask for a field (:func:`wall_field`). ``notes``
    collects the warnings."""
    grid, dem = frame.grid, frame.dem
    points, walls = split_base(spec.get("base"))
    if not len(points):
        return None
    profile = str(spec.get("profile") or "bowl")
    if profile not in PROFILES:
        raise ValueError(f"unknown profile of cut {name}: {profile}")
    base = points[~walls] if (~walls).any() else points
    inside = _rows(spec.get("inside"), 2)
    if not len(inside):
        inside = np.unique(base[:, :2], axis=0)
    outside = _rows(spec.get("outside"), 2)
    deepest = max(float((dem.at(base[:, 0], base[:, 1]) - base[:, 2]).max()), frame.dz)
    plan, limit = cut_plan(spec, grid, inside, outside, deepest)
    if not plan.mask.any():
        return None
    if plan.edge == "estimated":
        notes.append({"code": "edge_estimated", "unit": name, "limit": limit, "effect": "the volume may be overstated"})
    table = base_rows(base, plan, grid, dem)
    depths, held = floors(plan, table, grid, dem, base)
    for x, y, distance in zip(base[:, 0], base[:, 1], table.past):
        if distance > 0:
            notes.append({"code": "base_outside_outline", "unit": name, "x": float(x), "y": float(y),
                          "distance": float(distance)})
    if held[profile].any():
        notes.append({"code": "base_near_edge", "unit": name, "rows": int(held[profile].sum())})
    if float(table.t.max()) < RIM_T:
        notes.append({"code": "floor_unmeasured", "unit": name})
    room = np.maximum(dem.filled - frame.zmin, 0.0)
    if np.any((depths[profile] > room + 1e-9) & plan.mask & dem.mask):
        notes.append({"code": "cut_reaches_zmin", "unit": name})
    within = plan.mask & dem.mask
    depths = {p: np.minimum(depth, room) for p, depth in depths.items()}
    info = {"edge": plan.edge, "profile": profile, "plan_area": float(plan.mask.sum() * grid.cell ** 2),
            "volume_range": {p: float(depth[within].sum() * grid.cell ** 2) for p, depth in depths.items()}}
    if walls.any():
        fitted, box, seeds = wall_field(frame, name, plan, points, inside, orientations, nelements)
        span = max(float(np.ptp(points[:, 2])), 4 * frame.dz, 1.0)
        info.update(field=True, vertical=box.k, elements=box.elements, base_miss=wall_misses(fitted, points, span),
                    base_capped=[False] * len(points))
        return Cut(wall_function(frame, fitted, box, seeds, xyz), box, info)
    floor = dem.filled - depths[profile]
    miss = np.abs(floor[table.rows, table.cols] - base[:, 2])
    info.update(field=False, base_miss=[float(v) for v in miss], base_capped=[bool(v) for v in held[profile]])
    box = plan_box(plan, depths[profile], frame, fill_rows, nelements)
    return Cut(plan_function(frame, plan, depths[profile]), box, info)


def fit_fills(frame: Frame, name: str, fills: list, contacts: list, box: Box, nelements: int,
              covered: bool = False) -> Fitted | None:
    """The field that orders the ``fills`` of a cut (top-down), None when they have no contact; one
    fill needs it only when a layer lies on it (``covered``), else it has all of the cut."""
    if len(fills) < 2 and not covered:
        return None
    rows = [c for c in contacts if c[0] in fills]
    if not rows:
        return None
    xy = np.array([(r[1], r[2]) for r in rows], dtype=np.float64)
    scaled = np.array([r[3] for r in rows], dtype=np.float64) * box.k
    weight = np.array([r[4] for r in rows], dtype=np.float64)
    unit_of = np.array([r[0] for r in rows])
    seen = [u for u in fills if (unit_of == u).any()]
    means = {u: float(np.average(scaled[unit_of == u], weights=weight[unit_of == u])) for u in seen}
    values = _monotonic(means, seen, 0.1 * frame.dz * box.k)
    table = [_row(xy[i, 0], xy[i, 1], scaled[i], val=values[str(unit_of[i])], weight=weight[i])
             for i in range(len(rows))]
    if len(seen) < 2:
        table.append(_row(float(xy[:, 0].mean()), float(xy[:, 1].mean()), float(scaled.mean()),
                          normal=plane_normal(np.column_stack([xy, scaled]), weight)))
    feature = _solve(f"fills {name}", table, box.lo, box.hi, nelements)
    return Fitted(feature, "elevation", box.k, frame.dem, box.lo[2], box.hi[2], values=values)


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
    youngest = {}  # the first fill of each cut, going down the column
    for unit, kind in column:
        if kind == "fill":
            youngest.setdefault(fills[unit], unit)
    names = [u for u, _kind in column]
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
                covered = any(k == "layer" for _u, k in column[:names.index(unit)])
                if fitted is None or unit not in fitted.values or (unit == youngest[cut] and not covered):
                    own = cuts[cut]  # all that is left of the cut: nothing younger can have it
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
    contradicted = column_contradictions([c for c in contacts if c[0] not in fills], column)
    if contradicted:
        warnings.append({"code": "column_contradicted", "holes": contradicted})
    x0, y0, x1, y1 = frame.grid.bounds
    k = options.get("vertical")
    if k is None:
        thin = thinnest_unit([c for c in contacts if c[0] not in fills], column, frame.dem)
        k = vertical_scale(thin, (x1 - x0, y1 - y0, frame.zmax - frame.zmin), nelements)
    k = float(k)
    fields: dict = {}
    for i, spec in enumerate(series_specs):
        fitted = fit_series(frame, spec, contacts, orientations, order, k, nelements, warnings)
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
    listed = {str(c) for c in (job.get("cuts") or {})}
    for cut in (u for u, kind in column if kind == "cut" and u not in listed):
        warnings.append({"code": "cut_without_data", "unit": cut})
        cut_info[cut] = {"edge": "none"}
    for cut, spec in (job.get("cuts") or {}).items():
        cut = str(cut)
        own = [u for u, _kind in column if fills.get(u) == cut]
        fill_rows = np.array([c[1:4] for c in contacts if c[0] in own], dtype=np.float64).reshape(-1, 3)
        built = build_cut(frame, cut, spec, fill_rows, orientations, nelements, xyz, warnings)
        if built is None:
            warnings.append({"code": "cut_without_data", "unit": cut})
            cut_info[cut] = {"edge": "none"}
            continue
        cuts[cut] = built.function
        cut_info[cut] = built.info
        covered = bool(own) and any(kind == "layer" for u, kind in column[:order.index(own[0])])
        ordered = fit_fills(frame, cut, own, contacts, built.box, nelements, covered)
        if ordered is not None:
            fill_fields[cut] = ordered
        if len(own) > 1 or covered:
            told = {c[0] for c in contacts}
            warnings.extend({"code": "fill_without_contact", "unit": u, "cut": cut} for u in own if u not in told)
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
