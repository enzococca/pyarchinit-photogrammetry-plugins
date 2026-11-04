"""Every traced timber a closed 3D object, in the ship frame (x along the keel line, y to port, z up from the datum).

All of it is read on the relief of the hull from above (:mod:`relief`): a height field on a regular grid, the model's
upper surface per cell (the inside of a hull lying on its bottom).

- A strake (``plank``) is a SHELL: its inner face is the relief at the cells whose centres lie inside its polygon;
  where a frame, the keelson or a stringer lies over it (their polygons buffered by one cell: the cell on their side
  is half wall) the relief is that timber's top, so the strake's face there is interpolated along x from the strake's
  own cells on both sides in the same row (a row with none on either side: from all its visible cells) and the faces
  are flagged ``hidden``. The outer face is the inner one moved by the thickness along the field's unit normal pointing
  down and outward (the gradient smoothed over the thickness, so a land's step does not fold it). The thickness comes
  from :func:`thickness_of`.
- A frame, the keelson, a stringer, the ceiling, any other polygon is a SOLID: its top is the relief in its footprint,
  its bottom the hull under it, interpolated (linear, nearest at the rim) from a ring of cells 2 to 4 cells outside
  the footprint that lie in no other timber; a long timber is read across its short axis from the ring along its
  long sides (beyond its ends the timber may go on past what was traced: a frame stopped short of the sheer). What lies over a frame (the keelson, a stringer, the ceiling) owns the
  volume over its footprint: the frame's solid stops at it and the upper timber's bottom is the hull.
- The keel, unless a measured keel record is traced, is a hypothesis from typed siding and moulding
  (:func:`set_keel`): a box along the keel line whose top follows the keelson's bottom (else the hull's surface at
  y = 0), status ``reconstructed:given``.

Each object is a closed 2-manifold (every edge in exactly two faces, faces turned outward): the surfaces are built
on the corners of the cells (a corner's height is the mean of the cells around it) and closed by vertical walls along
the boundary of the cells. :func:`build` writes ``<chunk data>/shipwreck/timbers/<id>.ply`` per polygon record (face
property ``hidden``) and sets the record's ``solid`` (the file, relative to the chunk's data folder) and ``volume``
(m3); a thickness typed for the build goes into the ``thickness_mm`` of the strakes that had none. Nothing else of a
record is touched; the chunk's transform, model and products never.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import relief, ship, timbers, tracing
from .ship import NotReady
from .texts import num, t

SHELLS = ("plank",)
SOLIDS = ("frame", "keelson", "stringer", "ceiling", "other", "keel")
OVER_FRAMES = ("keelson", "stringer", "ceiling")  # timbers lying over the frames: a frame's solid stops at them
HIDING = tracing.CUTTERS  # the timbers whose footprint hides a strake (frame, keelson, stringer)
RING = (2.0, 4.0)  # cells: the ring around a footprint where the hull under it is read
ELONGATED = 2.0  # a footprint this many times as long as wide reads its bottom from its long sides only
SEAM_REACH = 0.03  # metres: a seam this close to a strake's outline bounds it
KEEL_FILE = "keel.ply"
FOLDER = "timbers"


@dataclass
class Solid:
    """A closed triangle mesh in the ship frame: ``vertices`` (N, 3), ``faces`` (M, 3) turned outward, ``hidden`` (M,)
    the faces of a strake's span under another timber (interpolated, not seen), the record's ``status``, ``name``,
    ``kind``, ``volume`` (m3) and, for a shell, ``thickness_mm``."""

    vertices: np.ndarray
    faces: np.ndarray
    hidden: np.ndarray
    status: str
    name: str
    kind: str
    volume: float
    thickness_mm: float | None = None
    id: str | None = None


# ----------------------------------------------------------------------------- the relief, read by windows
class _Grid:
    """A window of the relief's height raster around a part of the ship plan: ``z`` (rows, cols) with NaN off the
    model, the ship ``x`` of each column and ``y`` of each row (cell centres), the corners' ``xe``, ``ye``."""

    def __init__(self, chunk, bounds, margin: int = 6):
        import rasterio
        from rasterio.windows import Window

        tracing.need_fresh_relief(chunk)
        frame = ship.need_frame(chunk)
        sign = frame.sign
        path = relief.path_of(chunk, "height")
        x0, y0, x1, y1 = bounds
        with rasterio.open(path) as src:
            a = src.transform
            self.cell = float(abs(a.a))
            vs = sorted((sign * y0, sign * y1))
            c0 = int(math.floor((x0 - a.c) / a.a)) - margin
            c1 = int(math.ceil((x1 - a.c) / a.a)) + margin
            r0 = int(math.floor((vs[1] - a.f) / a.e)) - margin
            r1 = int(math.ceil((vs[0] - a.f) / a.e)) + margin
            c0, r0 = max(0, c0), max(0, r0)
            c1, r1 = min(src.width, max(c0, c1)), min(src.height, max(r0, r1))
            z = src.read(1, window=Window(c0, r0, c1 - c0, r1 - r0)).astype(np.float64)
            if src.nodata is not None and np.isfinite(src.nodata):
                z[z == src.nodata] = np.nan
        self.z = z
        self.xe = a.c + (c0 + np.arange(c1 - c0 + 1)) * a.a
        self.ye = sign * (a.f + (r0 + np.arange(r1 - r0 + 1)) * a.e)
        self.x = 0.5 * (self.xe[:-1] + self.xe[1:])
        self.y = 0.5 * (self.ye[:-1] + self.ye[1:])
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def inside(self, geom) -> np.ndarray:
        """The cells whose centres lie inside ``geom`` (a shapely geometry in the ship plan)."""
        import shapely

        if geom is None or geom.is_empty:
            return np.zeros(self.z.shape, bool)
        shapely.prepare(geom)
        return shapely.contains_xy(geom, self.X, self.Y)


def _polygon(xy):
    import shapely

    poly = shapely.Polygon(np.asarray(xy, float).reshape(-1, 2))
    return poly if poly.is_valid else poly.buffer(0)


def _polygons(chunk, kinds, frame, exclude=None) -> list:
    """The polygons in the ship plan of the records of ``kinds`` (all but ``exclude``)."""
    out = []
    for r in ship.state(chunk).get("timbers", []):
        if r["kind"] in kinds and r["geometry"]["type"] == "polygon" and r["id"] != exclude:
            out.append(_polygon(tracing.ship_xy_of(chunk, r, frame)))
    return out


def _union(polys):
    import shapely

    return shapely.union_all(polys) if polys else None


# ----------------------------------------------------------------------------- closed meshes on the cells
def _no_bowties(mask: np.ndarray) -> np.ndarray:
    """``mask`` without cells that touch the rest only at a corner (two cells on a diagonal, the other two off): the
    corner would be shared by two separate pieces of surface and its vertical edge by four walls."""
    m = mask.copy()
    while True:
        a, b, c, d = m[:-1, :-1], m[:-1, 1:], m[1:, :-1], m[1:, 1:]
        one, two = a & d & ~b & ~c, b & c & ~a & ~d
        if not (one.any() or two.any()):
            return m
        m[1:, 1:][one] = False
        m[1:, :-1][two] = False


def _corners(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The mean of ``values`` over the cells of ``mask`` around each corner: (rows + 1, cols + 1) (+ trailing dims)."""
    v = np.where(mask.reshape(mask.shape + (1,) * (values.ndim - 2)), values, 0.0)
    w = mask.astype(float)
    pad_v = np.zeros((v.shape[0] + 2, v.shape[1] + 2) + v.shape[2:])
    pad_w = np.zeros((w.shape[0] + 2, w.shape[1] + 2))
    pad_v[1:-1, 1:-1], pad_w[1:-1, 1:-1] = v, w
    s = pad_v[:-1, :-1] + pad_v[1:, :-1] + pad_v[:-1, 1:] + pad_v[1:, 1:]
    n = pad_w[:-1, :-1] + pad_w[1:, :-1] + pad_w[:-1, 1:] + pad_w[1:, 1:]
    n = np.maximum(n, 1.0).reshape(n.shape + (1,) * (values.ndim - 2))
    return s / n


def _signed_volume(v: np.ndarray, f: np.ndarray) -> float:
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def _closed(mask: np.ndarray, top: np.ndarray, bottom: np.ndarray, hidden_cells: np.ndarray | None = None):
    """The closed mesh over the cells of ``mask``: the surfaces through the corners ``top`` and ``bottom`` ((rows + 1,
    cols + 1, 3) positions), vertical walls along the cells' boundary. Returns vertices, faces (outward), hidden flags
    (the top and bottom faces of ``hidden_cells``) and the volume."""
    rows, cols = mask.shape
    width = cols + 1
    count = (rows + 1) * width
    rr, cc = np.nonzero(mask)
    v00 = rr * width + cc
    v01, v10 = v00 + 1, v00 + width
    v11 = v10 + 1
    top_f = np.vstack([np.column_stack([v00, v01, v11]), np.column_stack([v00, v11, v10])])
    bot_f = top_f[:, ::-1] + count
    hid = np.zeros(len(rr), bool) if hidden_cells is None else hidden_cells[rr, cc]
    hid = np.r_[hid, hid]
    padded = np.zeros((rows + 2, cols + 2), bool)
    padded[1:-1, 1:-1] = mask
    sides = []  # the directed edges of the top's boundary, as the top's faces run them
    for (dr, dc), a, b in (((-1, 0), v00, v01), ((0, 1), v01, v11), ((1, 0), v11, v10), ((0, -1), v10, v00)):
        out = ~padded[rr + 1 + dr, cc + 1 + dc]
        sides.append(np.column_stack([a[out], b[out]]))
    edges = np.vstack(sides)
    a, b = edges[:, 0], edges[:, 1]
    walls = np.vstack([np.column_stack([b, a, a + count]), np.column_stack([b, a + count, b + count])])
    faces = np.vstack([top_f, bot_f, walls])
    hidden = np.r_[hid, hid, np.zeros(len(walls), bool)]
    positions = np.vstack([top.reshape(-1, 3), bottom.reshape(-1, 3)])
    used = np.zeros(len(positions), bool)
    used[faces.ravel()] = True
    remap = np.cumsum(used) - 1
    vertices, faces = positions[used], remap[faces]
    volume = _signed_volume(vertices, faces)
    if volume < 0:
        faces, volume = faces[:, ::-1], -volume
    return vertices, faces.astype(np.int64), hidden, volume


def _plane(grid: _Grid, values: np.ndarray) -> np.ndarray:
    """Corner positions (x, y, ``values``)."""
    xe, ye = np.meshgrid(grid.xe, grid.ye)
    return np.stack([xe, ye, values], axis=-1)


def _require(mask: np.ndarray, record: dict) -> None:
    if not mask.any():
        raise NotReady(t(f"{record['name'] or record['id']}: nessuna cella del rilievo dentro il contorno.",
                         f"{record['name'] or record['id']}: no cell of the relief inside the outline."))


def _scattered(points: np.ndarray, values: np.ndarray, at: np.ndarray) -> np.ndarray:
    """``values`` at the points ``at``: ``griddata`` linear where the ``points`` span an area (nearest outside their
    hull); points on one line (a ring of one row at the model's edge, a strake one cell wide) or fewer than three:
    linear along that line, held at its ends; no points: NaN."""
    from scipy.interpolate import griddata

    out = np.full(len(at), np.nan)
    if not len(points):
        return out
    centred = points - points.mean(axis=0)
    rank = np.linalg.matrix_rank(centred, tol=1e-9) if len(points) >= 3 else 0
    if rank == 2:
        try:
            out = griddata(points, values, at, method="linear")
        except Exception:  # noqa: BLE001 - a degenerate set qhull refuses: nearest below
            out = np.full(len(at), np.nan)
    elif len(points) >= 2 and np.ptp(centred, axis=0).max() > 0:
        axis = np.linalg.svd(centred, full_matrices=False)[2][0]
        s = centred @ axis
        order = np.argsort(s)
        out = np.interp((at - points.mean(axis=0)) @ axis, s[order], values[order])
    bad = ~np.isfinite(out)
    if bad.any():
        out[bad] = griddata(points, values, at[bad], method="nearest") if len(points) > 1 else values[0]
    return out


# ----------------------------------------------------------------------------- the thickness of a strake
def _clinker_thickness(chunk, record) -> float | None:
    from . import clinker

    try:
        rows = clinker.strake_rows(chunk)
    except NotReady:
        return None
    code = str(record.get("name") or "").split()[-1:] or [""]
    for row in rows:
        if row.get("strake") == code[0] and row.get("side") == record.get("side") and row.get("thickness_mm"):
            return float(row["thickness_mm"])
    return None


def _seam_steps(chunk, record) -> list[float]:
    """The steps (mm) of the seams along the strake's outline: accepted seam records with a thickness, and the seams
    still among the proposals; a seam counts when most of its points within the strake's x range lie within
    :data:`SEAM_REACH` of the outline."""
    import shapely

    frame = ship.need_frame(chunk)
    outline = _polygon(tracing.ship_xy_of(chunk, record, frame)).boundary
    x0, _y0, x1, _y1 = outline.bounds
    lines = [(tracing.ship_xy_of(chunk, r, frame), r.get("thickness_mm"))
             for r in ship.state(chunk).get("timbers", []) if r["kind"] == "seam" and r["geometry"]["type"] == "line"]
    prop = ship.state(chunk).get("proposals") or {}
    read_in = prop.get("frame")
    if read_in is None or relief.frame_key(ship.Frame.from_dict(read_in)) == relief.frame_key(frame):
        lines += [(np.asarray(i["ship_xy"], float), i.get("step_mm")) for i in prop.get("items", [])
                  if i["kind"] == "seam"]
    steps = []
    for xy, step in lines:
        if step is None or len(xy) < 2:
            continue
        line = shapely.LineString(xy)
        n = max(2, int(line.length / 0.05))
        pts = np.asarray([line.interpolate(d).coords[0] for d in np.linspace(0.0, line.length, n)])
        pts = pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x1)]
        if len(pts) < 2:
            continue
        near = shapely.distance(outline, shapely.points(pts)) <= SEAM_REACH
        if near.mean() >= 0.5:
            steps.append(float(step))
    return steps


def _positive(thickness) -> float:
    value = float(thickness)
    if not value > 0:
        raise NotReady(t(f"Spessore non valido: {num(value, 1)} mm (deve essere maggiore di zero).",
                         f"Invalid thickness: {num(value, 1)} mm (it must be greater than zero)."))
    return value


def thickness_of(chunk, record: dict, given: float | None = None) -> tuple[float, str]:
    """The thickness (mm) of a strake and where it comes from: ``given`` ("given") > the record's ``thickness_mm``
    ("record") > the strake of the same name and side in the strake model of the sections ("clinker") > the median
    step of the seams along its outline ("seams"). :class:`NotReady` "spessore sconosciuto: scrivilo" otherwise."""
    if given is not None:
        return _positive(given), "given"
    if record.get("thickness_mm"):
        return float(record["thickness_mm"]), "record"
    found = _clinker_thickness(chunk, record)
    if found:
        return found, "clinker"
    steps = _seam_steps(chunk, record)
    if steps:
        return float(np.median(steps)), "seams"
    raise NotReady(t("spessore sconosciuto: scrivilo", "unknown thickness: type it"))


# ----------------------------------------------------------------------------- strakes
def _fill_along_x(z: np.ndarray, known: np.ndarray, wanted: np.ndarray, grid: _Grid) -> np.ndarray:
    """``z`` at the ``wanted`` cells interpolated along x from the ``known`` cells of the same row; a row with none
    known: from all the known cells (linear, nearest outside)."""
    out = z.copy()
    left = np.zeros(z.shape, bool)
    for r in np.flatnonzero(wanted.any(axis=1)):
        k = known[r]
        w = wanted[r]
        if k.sum() >= 1:
            out[r, w] = np.interp(grid.x[w], grid.x[k], z[r, k])
        else:
            left[r] = w
    if left.any():
        pts = np.column_stack([grid.X[known], grid.Y[known]])
        out[left] = _scattered(pts, z[known], np.column_stack([grid.X[left], grid.Y[left]]))
    return out


def _normals(field: np.ndarray, mask: np.ndarray, grid: _Grid, sigma: float) -> np.ndarray:
    """Unit normals pointing down/outward of the height ``field`` (known on ``mask``), the gradient smoothed over
    ``sigma`` cells: (rows, cols, 3)."""
    from scipy import ndimage

    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    filled = field[tuple(idx)]
    if sigma > 0:
        filled = ndimage.gaussian_filter(filled, sigma, mode="nearest")
    gy, gx = np.gradient(filled, grid.y, grid.x) if min(filled.shape) > 1 else (
        np.zeros_like(filled), np.zeros_like(filled))
    n = np.stack([gx, gy, -np.ones_like(gx)], axis=-1)
    return n / np.linalg.norm(n, axis=-1, keepdims=True)


def strake_shell(chunk, record: dict, thickness: float) -> Solid:
    """The shell of a strake (``thickness`` in mm): see the module. :class:`NotReady` when the relief is missing or
    stale, or no cell of it lies inside the strake."""
    thickness = _positive(thickness)
    frame = ship.need_frame(chunk)
    poly = _polygon(tracing.ship_xy_of(chunk, record, frame))
    grid = _Grid(chunk, poly.bounds)
    inside = grid.inside(poly)
    over = _union(_polygons(chunk, HIDING, frame, exclude=record["id"]))
    hiding = grid.inside(over.buffer(grid.cell)) if over is not None else np.zeros(inside.shape, bool)
    finite = np.isfinite(grid.z)
    known = inside & ~hiding & finite
    hidden = inside & hiding
    mask = _no_bowties(known | hidden)
    _require(mask & known, record)
    z = _fill_along_x(grid.z, mask & known, mask & hidden, grid)
    hidden &= mask
    th = float(thickness) / 1000.0
    normals = _normals(z, mask, grid, max(1.0, th / grid.cell))
    inner = _plane(grid, _corners(z, mask))
    nc = _corners(normals, mask)
    nc = nc / np.maximum(np.linalg.norm(nc, axis=-1, keepdims=True), 1e-12)
    v, f, hid, vol = _closed(mask, inner, inner + th * nc, hidden)
    return Solid(v, f, hid, timbers.status_of(record), record["name"], record["kind"], vol, float(thickness),
                 record["id"])


# ----------------------------------------------------------------------------- solids
def _footprint(chunk, record, frame):
    """A timber's footprint in the ship plan: its polygon, a frame's less what lies over it (:data:`OVER_FRAMES`)."""
    poly = _polygon(tracing.ship_xy_of(chunk, record, frame))
    if record["kind"] == "frame":
        over = _union(_polygons(chunk, OVER_FRAMES, frame, exclude=record["id"]))
        if over is not None:
            poly = poly.difference(over)
    return poly


def _axes(cells: np.ndarray):
    """The centre and the unit long and short axes of a footprint's cells; whether it is long (:data:`ELONGATED`)."""
    centre = cells.mean(axis=0)
    vectors = np.linalg.eigh(np.cov((cells - centre).T))[1] if len(cells) > 2 else np.eye(2)
    along = vectors[:, -1]
    across = np.array([-along[1], along[0]])
    u, w = (cells - centre) @ along, (cells - centre) @ across
    return centre, along, across, bool(np.ptp(u) >= ELONGATED * max(np.ptp(w), 1e-9))


MAX_SLOPE = 6.0  # the steepest floor read in a ring (80 degrees): steeper is the side of a timber
SAG = 0.5  # the bottom across a timber leaves the chord by at most this share of the run between the ring's edges


def _across(w: np.ndarray, z: np.ndarray, near: np.ndarray, far: np.ndarray, at: np.ndarray, cell: float) -> np.ndarray:
    """The hull under a strip of a long footprint at ``at`` (positions across) from the ring's points of the strip
    (``w``, ``z``; ``near`` and ``far``: those on either side). The outer edge of the ring on each side carries the
    height (an outline traced a cell or two inside the timber leaves its top in the inner edge of the ring) and the
    slope there, read on the outer cells of that side (from :data:`RING` [0] + 1 cells out): a cubic through both ends
    with both slopes, so that a floor curving under a timber (the rocker at stem and stern) is followed and not cut by
    a chord. A side with one cell only, or slopes that disagree with the chord by more than its own run, gives the
    chord."""
    i, j = np.flatnonzero(near)[0], np.flatnonzero(far)[-1]
    w0, z0, w1, z1 = w[i], z[i], w[j], z[j]
    run = w1 - w0
    chord = z0 + (at - w0) * (z1 - z0) / run
    fits = []
    for side, outer in ((near, w0), (far, w1)):
        k = side & (np.abs(w - outer) <= (RING[1] - RING[0]) * cell + 1e-9)
        if np.ptp(w[k]) < cell * 0.99 if k.sum() >= 2 else True:
            return chord
        fits.append((np.polyfit(w[k], z[k], 1)[0], float(w[k].mean())))
    # a line's slope is the floor's at the middle of its cells: carried to the outer edge with the curvature that the
    # two sides' slopes give (exact on a parabola)
    curvature = (fits[1][0] - fits[0][0]) / (fits[1][1] - fits[0][1])
    slopes = [fits[0][0] + curvature * (w0 - fits[0][1]), fits[1][0] + curvature * (w1 - fits[1][1])]
    if any(abs(sl) > MAX_SLOPE for sl in slopes):  # a wall of another timber in the ring, not the floor
        return chord
    u = (at - w0) / run  # Hermite cubic on [0, 1]
    h00, h10, h01, h11 = 2 * u**3 - 3 * u**2 + 1, u**3 - 2 * u**2 + u, -2 * u**3 + 3 * u**2, u**3 - u**2
    curve = h00 * z0 + h10 * run * slopes[0] + h01 * z1 + h11 * run * slopes[1]
    if np.abs(curve - chord).max(initial=0.0) > SAG * abs(run):
        return chord
    return curve


def _hull_under(cells: np.ndarray, ring: np.ndarray, zs: np.ndarray, cell: float) -> np.ndarray:
    """The hull's height at ``cells`` from the ``ring`` points (heights ``zs``). A long footprint is read across its
    short axis, in strips one cell wide along the long one: linear between the outer edges of the ring on its two
    long sides (an outline traced a cell or two inside the timber leaves its top in the ring's inner edge; beyond its
    ends the timber may go on past what was traced - a frame stopped short of the sheer - and the ring there would be
    its top; a triangulation of the lattice links cells rows apart, wrong where the hull is steep). What is left, and
    any other footprint: ``griddata`` linear on the ring, nearest at the rim (:func:`_scattered`)."""
    out = np.full(len(cells), np.nan)
    centre, along, across, long = _axes(cells)
    if long:
        cu, cw = (cells - centre) @ along, (cells - centre) @ across
        ru, rw = (ring - centre) @ along, (ring - centre) @ across
        cb, rb = np.round(cu / cell).astype(np.int64), np.round(ru / cell).astype(np.int64)
        order = np.lexsort((rw, rb))
        rb, rw, rz = rb[order], rw[order], zs[order]
        starts = np.searchsorted(rb, np.unique(cb))
        ends = np.searchsorted(rb, np.unique(cb), side="right")
        for b, lo, hi in zip(np.unique(cb), starts, ends):
            here = np.flatnonzero(cb == b)
            w, z = rw[lo:hi], rz[lo:hi]
            near, far = w < cw[here].min(), w > cw[here].max()
            if not (near.any() and far.any()):
                continue
            out[here] = _across(w, z, near, far, cw[here], cell)
        keep = (ru >= cu.min()) & (ru <= cu.max())
        if keep.sum() >= 3:
            ring, zs = ring[keep], zs[keep]
    left = ~np.isfinite(out)
    if left.any():
        out[left] = _scattered(ring, zs, cells[left])
    return out


def _solid_grid(chunk, record):
    """The grid of a timber's solid: (grid, mask, top, bottom) on its cells."""
    from scipy import ndimage

    frame = ship.need_frame(chunk)
    poly = _footprint(chunk, record, frame)
    if poly.is_empty:
        raise NotReady(t(f"{record['name'] or record['id']}: l'impronta è vuota.",
                         f"{record['name'] or record['id']}: the footprint is empty."))
    grid = _Grid(chunk, poly.bounds, margin=int(RING[1]) + 3)
    finite = np.isfinite(grid.z)
    mask = _no_bowties(grid.inside(poly) & finite)
    _require(mask, record)
    others = _union([p for p in _polygons(chunk, SOLIDS, frame, exclude=record["id"])])
    taken = grid.inside(others.buffer(grid.cell)) if others is not None else np.zeros(mask.shape, bool)
    if record["kind"] == "frame":  # the part of the frame cut away is not hull either
        taken |= grid.inside(_polygon(tracing.ship_xy_of(chunk, record, frame)).buffer(grid.cell))
    distance = ndimage.distance_transform_edt(~mask)
    ring = (distance >= RING[0]) & (distance <= RING[1]) & finite & ~taken & ~mask
    if not ring.any():
        raise NotReady(t(f"{record['name'] or record['id']}: nessuna parte dello scafo attorno da cui leggere il "
                         f"fondo.",
                         f"{record['name'] or record['id']}: no hull around it to read its bottom from."))
    pts = np.column_stack([grid.X[ring], grid.Y[ring]])
    at = np.column_stack([grid.X[mask], grid.Y[mask]])
    vals = _hull_under(at, pts, grid.z[ring], grid.cell)
    top = np.where(mask, grid.z, np.nan)
    bottom = np.full(mask.shape, np.nan)
    bottom[mask] = np.minimum(vals, grid.z[mask])
    return grid, mask, top, bottom


def timber_solid(chunk, record: dict) -> Solid:
    """The solid of a frame, the keelson, a stringer, the ceiling or another polygon: see the module."""
    grid, mask, top, bottom = _solid_grid(chunk, record)
    v, f, hid, vol = _closed(mask, _plane(grid, _corners(top, mask)), _plane(grid, _corners(bottom, mask)))
    return Solid(v, f, hid, timbers.status_of(record), record["name"], record["kind"], vol, None, record["id"])


# ----------------------------------------------------------------------------- the keel as given
def _keelsons(chunk) -> list[dict]:
    return [r for r in ship.state(chunk).get("timbers", []) if r["kind"] == "keelson"
            and r["geometry"]["type"] == "polygon"]


def _relief_x_range(chunk) -> tuple[float, float]:
    """The x range of the relief's cells on the model (read at a reduced size)."""
    import rasterio

    tracing.need_fresh_relief(chunk)
    with rasterio.open(relief.path_of(chunk, "height")) as src:
        step = max(1, int(math.ceil(max(src.width, src.height) / 2000)))
        shape = (max(1, src.height // step), max(1, src.width // step))
        z = src.read(1, out_shape=shape).astype(np.float64)
        if src.nodata is not None and np.isfinite(src.nodata):
            z[z == src.nodata] = np.nan
        a = src.transform
        cols = np.flatnonzero(np.isfinite(z).any(axis=0))
        if not len(cols):
            raise NotReady(t("Il rilievo dello scafo è vuoto.", "The relief of the hull is empty."))
        cw = a.a * src.width / shape[1]
        return float(a.c + cols[0] * cw), float(a.c + (cols[-1] + 1) * cw)


def set_keel(chunk, siding, moulding, x0=None, x1=None) -> dict | None:
    """The keel from typed dimensions (m): ``siding`` across, ``moulding`` up, from ``x0`` to ``x1`` along the keel line
    (default: the keelson's x range, else the relief's), kept in ``state["keel"]`` (its file and volume come with the
    next :func:`build`). ``siding`` None removes it."""
    st = ship.state(chunk)
    if siding is None or moulding is None:
        st.pop("keel", None)
        return None
    siding, moulding = float(siding), float(moulding)
    if siding <= 0 or moulding <= 0:
        raise NotReady(t("Larghezza e altezza della chiglia devono essere positive.",
                         "The keel's siding and moulding must be positive."))
    if x0 is None or x1 is None:
        frame = ship.need_frame(chunk)
        keelsons = _keelsons(chunk)
        if keelsons:
            xs = np.concatenate([tracing.ship_xy_of(chunk, r, frame)[:, 0] for r in keelsons])
            lo, hi = float(xs.min()), float(xs.max())
        else:
            lo, hi = _relief_x_range(chunk)
        x0 = lo if x0 is None else x0
        x1 = hi if x1 is None else x1
    x0, x1 = sorted((float(x0), float(x1)))
    if x1 - x0 <= 0:
        raise NotReady(t("La chiglia ha lunghezza zero.", "The keel has no length."))
    st["keel"] = {"siding": siding, "moulding": moulding, "x0": x0, "x1": x1, "status": "reconstructed:given"}
    return st["keel"]


def _measured_keel(chunk) -> bool:
    return any(r["kind"] == "keel" and r["geometry"]["type"] == "polygon" and timbers.is_measured(r)
               for r in ship.state(chunk).get("timbers", []))


def _keel_line(chunk, xs: np.ndarray, siding: float) -> np.ndarray:
    """The height of the keel's top at ``xs``: the keelson's bottom on the keel line where a keelson lies (its ends
    held to the stations within its outline), else the lowest of the relief across the keel's siding."""
    import warnings

    top = np.full(len(xs), np.nan)
    covered = np.zeros(len(xs), bool)
    frame = ship.need_frame(chunk)
    for r in _keelsons(chunk):
        x0, _y0, x1, _y1 = _polygon(tracing.ship_xy_of(chunk, r, frame)).bounds
        here = (xs >= x0) & (xs <= x1)
        try:
            grid, mask, _top, bottom = _solid_grid(chunk, r)
        except NotReady:
            continue  # not covered: the keel's top there is read on the hull
        row = int(np.argmin(np.abs(grid.y)))
        ok = mask[row] & np.isfinite(bottom[row])
        if ok.sum() < 2:
            continue
        covered |= here
        top[here] = np.fmin(top[here], np.interp(xs[here], grid.x[ok], bottom[row, ok]))
    missing = ~np.isfinite(top) & ~covered
    if missing.any():
        grid = _Grid(chunk, (float(xs.min()), -siding / 2.0, float(xs.max()), siding / 2.0), margin=1)
        band = np.abs(grid.y) <= max(siding / 2.0, grid.cell)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # columns off the model
            low = np.nanmin(np.where(band[:, None], grid.z, np.nan), axis=0)
        ok = np.isfinite(low)
        if ok.any():
            top[missing] = np.interp(xs[missing], grid.x[ok], low[ok])
    if not np.isfinite(top).any():
        raise NotReady(t("Nessuna superficie dello scafo lungo la linea di chiglia.",
                         "No surface of the hull along the keel line."))
    ok = np.isfinite(top)
    return np.interp(xs, xs[ok], top[ok])


def keel_solid(chunk) -> Solid | None:
    """The keel from the typed dimensions (:func:`set_keel`): a box along the keel line under the keelson's bottom
    (else under the hull's surface at y = 0), the sides vertical, status ``reconstructed:given``, name "chiglia".
    None when nothing was typed or a measured keel record is traced."""
    keel = ship.state(chunk).get("keel")
    if not keel or _measured_keel(chunk):
        return None
    s, m, x0, x1 = keel["siding"], keel["moulding"], keel["x0"], keel["x1"]
    cell = float(relief.entry(chunk) and ship.state(chunk).get("relief", {}).get("cell") or relief.DEFAULT_CELL)
    n = int(min(2000, max(2, math.ceil((x1 - x0) / max(cell, (x1 - x0) / 2000)) + 1)))
    xs = np.linspace(x0, x1, n)
    ztop = _keel_line(chunk, xs, s)
    ye = np.array([-s / 2.0, s / 2.0])
    X, Y = np.meshgrid(xs, ye)
    Z = np.vstack([ztop, ztop])
    top = np.stack([X, Y, Z], axis=-1)
    bottom = np.stack([X, Y, Z - m], axis=-1)
    v, f, hid, vol = _closed(np.ones((1, n - 1), bool), top, bottom)
    return Solid(v, f, hid, "reconstructed:given", t("chiglia", "keel"), "keel", vol, None, None)


# ----------------------------------------------------------------------------- files and the build
def _one_line(text) -> str:
    """A header comment's text: no line breaks (they would end the comment and break the header for other readers)."""
    return " ".join(str(text).split())


def write_ply(path, solid: Solid, frame: ship.Frame | None = None) -> Path:
    """A binary little-endian PLY of ``solid`` in the ship frame: double vertices, triangle faces with the uchar
    property ``hidden``, comments with the record (as :func:`pyarchinit_photogrammetry.io.mesh_ply.write_mesh` writes
    a mesh, which has no place for a face property; the program's :class:`~pyarchinit_photogrammetry.io.mesh_ply.
    PlyMesh` and ``read_mesh`` read it). Written to a temporary file first."""
    path = Path(path)
    comments = ["ppg_shipwreck solid", "system ship (x along the keel line, y to port, z up from the datum)",
                f"kind {solid.kind}", f"status {solid.status}", f"name {_one_line(solid.name)}"]
    if solid.id:
        comments.append(f"id {solid.id}")
    if solid.thickness_mm is not None:
        comments.append(f"thickness_mm {solid.thickness_mm:.3f}")
    if frame is not None:
        import json

        comments.append(f"ship_frame {json.dumps(frame.to_dict())}")
    header = ["ply", "format binary_little_endian 1.0", *(f"comment {c}" for c in comments),
              f"element vertex {len(solid.vertices)}", "property double x", "property double y", "property double z",
              f"element face {len(solid.faces)}", "property list uchar int vertex_indices", "property uchar hidden",
              "end_header"]
    vout = np.asarray(solid.vertices, "<f8")
    fout = np.empty(len(solid.faces), dtype=[("n", "u1"), ("v", "<i4", (3,)), ("hidden", "u1")])
    fout["n"], fout["v"], fout["hidden"] = 3, solid.faces, solid.hidden.astype(np.uint8)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as out:
        out.write(("\n".join(header) + "\n").encode("utf-8"))
        out.write(vout.tobytes())
        out.write(fout.tobytes())
    tmp.replace(path)
    return path


def _label(record: dict) -> str:
    return record.get("name") or record["id"]


def _drop_solid(record: dict, folder: Path) -> None:
    record.pop("solid", None)
    record.pop("volume", None)
    (folder / f"{record['id']}.ply").unlink(missing_ok=True)


def build(chunk, thickness: float | None = None, progress=None) -> dict:
    """The 3D object of every polygon record: ``<chunk data>/shipwreck/timbers/<id>.ply`` (:func:`write_ply`), the
    record's ``solid`` (the file, relative to the chunk's data folder) and ``volume`` (m3). A strake's thickness is
    its own (:func:`thickness_of`: the record's, the strake model's, the seams'); ``thickness`` (mm) is the DEFAULT
    for the strakes that have none, written into their ``thickness_mm`` (a strake with its own is never changed).
    A record that cannot be built - for whatever reason - is refused with the reason (its old file and fields go)
    and the build goes on; lines and points are listed without a solid; the typed keel (:func:`keel_solid`) goes to
    ``keel.ply`` and ``state["keel"]``. At the end, whatever happened, the files of records no longer in the
    catalogue are removed and ``state["solids"]`` is written. Returns ``{"built": [...], "refused": [...],
    "without_solid": [...], "keel": {...} | None}``."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    frame = ship.need_frame(chunk)
    if relief.entry(chunk) is None:
        raise NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
    tracing.need_fresh_relief(chunk)
    if thickness is not None:
        thickness = _positive(thickness)
    folder = ship.folder(chunk) / FOLDER
    folder.mkdir(parents=True, exist_ok=True)
    records = list(ship.state(chunk).get("timbers", []))
    polygons = [r for r in records if r["geometry"]["type"] == "polygon"]
    out = {"built": [], "refused": [], "without_solid": [], "keel": None}
    for r in records:
        if r["geometry"]["type"] != "polygon":
            out["without_solid"].append({"id": r["id"], "name": _label(r), "kind": r["kind"],
                                         "type": r["geometry"]["type"]})
    total = len(polygons) + 1
    complete = False
    try:
        for k, r in enumerate(polygons):
            progress(k / total, t(f"Legni in 3D: {_label(r)} ({k + 1}/{len(polygons)})",
                                  f"Timbers in 3D: {_label(r)} ({k + 1}/{len(polygons)})"))
            typed = False
            try:
                if r["kind"] in SHELLS:
                    try:
                        mm, _source = thickness_of(chunk, r)
                    except NotReady:
                        if thickness is None:
                            raise
                        mm, typed = thickness, True
                    solid = strake_shell(chunk, r, mm)
                else:
                    solid = timber_solid(chunk, r)
                path = write_ply(folder / f"{r['id']}.ply", solid, frame)
            except Exception as exc:  # noqa: BLE001 - one record refused, never the whole build
                _drop_solid(r, folder)
                reason = str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"
                out["refused"].append({"id": r["id"], "name": _label(r), "kind": r["kind"], "reason": reason})
                continue
            r["solid"] = str(path.relative_to(chunk.data_dir))
            r["volume"] = float(solid.volume)
            if typed:
                r["thickness_mm"] = float(thickness)
            out["built"].append({"id": r["id"], "name": _label(r), "kind": r["kind"], "file": r["solid"],
                                 "volume": solid.volume, "status": solid.status, "hidden": int(solid.hidden.sum()),
                                 "thickness_mm": solid.thickness_mm})
        progress(len(polygons) / total, t("Chiglia", "Keel"))
        keel_state = ship.state(chunk).get("keel")
        if keel_state:
            keel_state.pop("solid", None)
            keel_state.pop("volume", None)
            try:
                keel = keel_solid(chunk)
                if keel is not None:
                    path = write_ply(folder / KEEL_FILE, keel, frame)
                    keel_state["solid"] = str(path.relative_to(chunk.data_dir))
                    keel_state["volume"] = float(keel.volume)
                    out["keel"] = {"file": keel_state["solid"], "volume": keel.volume, "status": keel.status,
                                   "name": keel.name}
            except Exception as exc:  # noqa: BLE001
                reason = str(exc) if isinstance(exc, NotReady) else f"{type(exc).__name__}: {exc}"
                out["refused"].append({"id": "keel", "name": t("chiglia", "keel"), "kind": "keel", "reason": reason})
        complete = True
    finally:
        _clean(chunk, folder)
        ship.state(chunk)["solids"] = {"built": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                                       "count": len(out["built"]), "refused": [r["id"] for r in out["refused"]],
                                       "frame_key": relief.frame_key(frame), "inputs": inputs_key(chunk)}
        if not complete:  # cancelled (or failed): what was built so far, said as partial
            ship.state(chunk)["solids"]["cancelled"] = True
    volume = sum(b["volume"] for b in out["built"]) + (out["keel"]["volume"] if out["keel"] else 0.0)
    progress(1.0, t(f"Legni in 3D: {len(out['built'])} costruiti, {len(out['refused'])} rifiutati, "
                    f"{num(volume, 3)} m³",
                    f"Timbers in 3D: {len(out['built'])} built, {len(out['refused'])} refused, {num(volume, 3)} m³"))
    return out


CACHES = ("ship_xy", "visible_xy", "frame_key", "solid", "volume")  # fields of a record no solid depends on


def inputs_key(chunk) -> str:
    """What the solids are built from: the records (without their plan caches and their own solids), the typed keel
    and the relief (its elevation and when it was built). A change of any makes the solids older than the records."""
    import hashlib
    import json

    st = ship.state(chunk)
    value = {"timbers": [{k: v for k, v in r.items() if k not in CACHES} for r in st.get("timbers", [])],
             "keel": {k: v for k, v in (st.get("keel") or {}).items() if k not in ("solid", "volume")},
             "relief": {k: (st.get("relief") or {}).get(k) for k in ("elevation", "built", "cell")}}
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def is_stale(chunk) -> bool:
    """Whether the solids were built (also partly: :func:`build` cancelled) from records, a keel, a relief or a ship
    frame other than the present ones: the 3D files no longer say what the catalogue says. False when none was built."""
    done = ship.state(chunk).get("solids")
    if not done:
        return False
    frame = ship.frame_of(chunk)
    return (done.get("inputs") != inputs_key(chunk)
            or (frame is not None and done.get("frame_key") != relief.frame_key(frame)))


def _clean(chunk, folder: Path) -> None:
    """The files no record holds any more: of records gone from the catalogue (or turned into lines and points), the
    keel's when no keel is typed (or a measured one replaces it), left-over temporary files."""
    keep = {f"{r['id']}.ply" for r in ship.state(chunk).get("timbers", []) if r["geometry"]["type"] == "polygon"}
    keel = ship.state(chunk).get("keel")
    if keel and keel.get("solid"):
        keep.add(KEEL_FILE)
    for old in list(folder.glob("*.ply")) + list(folder.glob("*.ply.tmp")):
        if old.name not in keep:
            old.unlink(missing_ok=True)
