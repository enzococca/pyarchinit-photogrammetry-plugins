"""The measuring half of the plugin, in the program's process: numpy, scipy, shapely and trimesh only.

It reads the photogrammetric model and turns a part of it into the numbers of an idealised element:

* a **column**: the axis of a surface of revolution (from the normals, as the pottery of the program finds the
  axis of a vessel, then refined on the centres of circles fitted to slices), the measured profile (the median
  radius per height and its spread), a spline through K knots fitted to it by least squares, the entasis;
* a **moulding**: the frames of a path (``u`` = up x tangent, ``v`` up), the section of the model by the plane
  normal to the path at a station, kept in a window around the path and closed against the wall, simplified;
  the polygonal sweep of a section along a path with mitred corners (the volume to expect from the solid);
* an **arch**: the intrados seen from a point inside the opening (and the extrados beyond it) on a section of the
  model, a circle (semicircular or segmental), a pointed arch (two circles centred on the springing line), an
  axis-aligned ellipse or the curve as measured, by least squares with the jambs below the springing line, the
  type chosen by the RMS when asked for "auto"; the outline of the ring as lines and three-point arcs;
* the **fit**: the distance of the model's points to the tessellated solid (median, 90th percentile, maximum).

Every coordinate here is in metres in a frame with a small origin (the plugin subtracts a round origin from the
chunk's world coordinates first): nothing here knows the chunk, the CRS or CadQuery.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

UP = np.array([0.0, 0.0, 1.0])
ARCH_TYPES = ("semicircular", "segmental", "pointed", "elliptical", "measured")
FIT_FAMILIES = ("circle", "pointed", "ellipse")


def unit(v) -> np.ndarray:
    v = np.asarray(v, np.float64)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    if np.any(n < 1e-12):
        raise ValueError("a direction of zero length")
    return v / n


def perpendicular(d) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to ``d`` and to each other (``d``, ``e1``, ``e2`` right-handed)."""
    d = unit(d)
    e1 = np.cross(d, [1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.cross(d, [0.0, 1.0, 0.0])
    e1 = unit(e1)
    return e1, np.cross(d, e1)


def frame_matrix(origin, x_axis, y_axis, z_axis) -> np.ndarray:
    """The 4 x 4 matrix of a local frame (columns: axes and origin)."""
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = x_axis, y_axis, z_axis, origin
    return m


def apply(matrix, points) -> np.ndarray:
    p = np.asarray(points, np.float64)
    return p @ np.asarray(matrix)[:3, :3].T + np.asarray(matrix)[:3, 3]


# ====================================================================== selecting a part of the model
def select_faces(vertices, faces, polygon=None, zmin=None, zmax=None, box=None):
    """The faces whose three vertices are inside the region: a plan ``polygon`` (x, y) between the heights
    ``zmin`` and ``zmax``, or an axis-aligned ``box`` ``(xmin, ymin, zmin, xmax, ymax, zmax)``. Returns
    ``(vertices, faces)`` of the part, the vertices renumbered."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    inside = np.ones(len(v), bool)
    if box is not None:
        lo, hi = np.asarray(box[:3], float), np.asarray(box[3:], float)
        inside &= np.all((v >= lo) & (v <= hi), axis=1)
    if polygon is not None:
        inside &= points_in_polygon(v[:, :2], polygon)
    if zmin is not None:
        inside &= v[:, 2] >= zmin
    if zmax is not None:
        inside &= v[:, 2] <= zmax
    keep = inside[f].all(axis=1)
    used = np.unique(f[keep])
    renum = -np.ones(len(v), np.int64)
    renum[used] = np.arange(len(used))
    return v[used], renum[f[keep]]


def points_in_polygon(xy, polygon) -> np.ndarray:
    """Which plan points are inside ``polygon`` ((N, 2) vertices or a shapely polygon)."""
    import shapely
    from shapely.geometry import Polygon

    poly = polygon if hasattr(polygon, "exterior") else Polygon(np.asarray(polygon, float)[:, :2])
    xy = np.asarray(xy, np.float64)
    return np.asarray(shapely.contains_xy(poly, xy[:, 0], xy[:, 1]), bool)


def mesh_resolution(vertices, faces) -> float:
    """The median edge length of a mesh."""
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if not len(f):
        return 0.0
    e = np.linalg.norm(v[f[:, [1, 2, 0]]] - v[f], axis=2)
    return float(np.median(e))


# ====================================================================== column
@dataclass
class ColumnMeasure:
    """What the model says of a column: the axis (``point`` at the base, ``direction`` up), the height, the
    measured profile (median radius per height ``radii`` at ``heights`` from the base, ``spread`` = the
    interquartile range of the radius at that height), the step of the profile and how well the slices are
    circles around the axis (``axis_rms``, metres)."""

    point: np.ndarray
    direction: np.ndarray
    height: float
    heights: np.ndarray
    radii: np.ndarray
    spread: np.ndarray
    step: float
    axis_rms: float = 0.0
    method: str = "normals+circles"

    def to_dict(self) -> dict:
        return {"point": self.point.tolist(), "direction": self.direction.tolist(), "height": self.height,
                "heights": self.heights.tolist(), "radii": self.radii.tolist(), "spread": self.spread.tolist(),
                "step": self.step, "axis_rms": self.axis_rms, "method": self.method}


def _circle_fit(xy: np.ndarray):
    """Circle by algebraic least squares then geometric refinement: ``(cx, cy, r, rms)`` or None."""
    from scipy.optimize import least_squares

    if len(xy) < 5:
        return None
    m = xy.mean(axis=0)
    p = xy - m
    a = np.column_stack([2 * p, np.ones(len(p))])
    b = (p ** 2).sum(axis=1)
    try:
        sol = np.linalg.lstsq(a, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    c0 = sol[:2]
    r0 = np.sqrt(max(sol[2] + c0 @ c0, 1e-12))

    def res(x):
        return np.linalg.norm(p - x[:2], axis=1) - x[2]

    x = least_squares(res, np.r_[c0, r0], method="lm").x
    rms = float(np.sqrt(np.mean(res(x) ** 2)))
    return x[0] + m[0], x[1] + m[1], abs(x[2]), rms


def refine_axis(vertices, point, direction, slices: int = 24, iterations: int = 3):
    """The axis of a surface of revolution from the centres of circles fitted to slices of its vertices
    across the axis (a column, a drum): returns ``(point, direction, rms)``, ``rms`` = the mean RMS of the
    circles (how round the slices are). Slices that are not circles (a capital's corner, a broken part)
    count less: the line through the centres is a weighted, trimmed fit."""
    v = np.asarray(vertices, np.float64)
    p, d = np.asarray(point, float), unit(direction)
    rms_all = 0.0
    for _ in range(iterations):
        e1, e2 = perpendicular(d)
        h = (v - p) @ d
        lo, hi = np.percentile(h, [1, 99])
        edges = np.linspace(lo, hi, slices + 1)
        thick = (hi - lo) / slices * 0.25
        centres, weights, rmss = [], [], []
        for a in 0.5 * (edges[:-1] + edges[1:]):
            m = np.abs(h - a) < thick
            if m.sum() < 12:
                continue
            q = v[m] - p
            xy = np.column_stack([q @ e1, q @ e2])
            fit = _circle_fit(xy)
            if fit is None or fit[2] <= 0:
                continue
            cx, cy, r, rms = fit
            centres.append(p + cx * e1 + cy * e2 + a * d)
            weights.append(1.0 / (rms + 1e-5) ** 2)
            rmss.append(rms)
        if len(centres) < 3:
            break
        c = np.asarray(centres)
        w = np.asarray(weights)
        rmss = np.asarray(rmss)
        keep = rmss <= max(np.median(rmss) * 3.0, 1e-5)
        c, w = c[keep], w[keep]
        if len(c) < 3:
            break
        mean = (c * w[:, None]).sum(0) / w.sum()
        cov = ((c - mean) * w[:, None]).T @ (c - mean)
        nd = np.linalg.eigh(cov)[1][:, -1]
        if nd @ d < 0:
            nd = -nd
        p, d = mean, unit(nd)
        rms_all = float(np.median(rmss[keep]))
    return p, d, rms_all


def column_axis(vertices, faces, initial=None):
    """The axis of a column part: ``(point, direction, rms)``. ``initial`` ``(point, direction)`` skips the
    search from the normals (the axis of another element: "same axis as element X"). The direction points
    up (positive z) unless the shaft lies flat, when it keeps the orientation found."""
    v = np.asarray(vertices, np.float64)
    if initial is not None:
        p0, d0 = np.asarray(initial[0], float), unit(initial[1])
    else:
        try:
            import trimesh

            from pyarchinit_photogrammetry.archaeo.pottery import estimate_axis

            ax = estimate_axis(trimesh.Trimesh(v, faces, process=False))
            p0, d0 = ax.point, ax.direction
        except ImportError:  # pragma: no cover - the program without its pottery: principal direction
            c = v.mean(axis=0)
            d0 = np.linalg.eigh((v - c).T @ (v - c))[1][:, -1]
            p0 = c
    if abs(d0[2]) > 0.5 and d0[2] < 0:
        d0 = -d0
    p, d, rms = refine_axis(v, p0, d0)
    if abs(d[2]) > 0.5 and d[2] < 0:
        d = -d
    return p, d, rms


def _profile_points(vertices, faces, point, direction, angles: int):
    """(r, h) points of the model on ``angles`` half-planes through the axis (the pottery's cuts), h from the
    point on the axis; the vertices themselves when the cut cannot be made."""
    v = np.asarray(vertices, np.float64)
    d = unit(direction)
    try:
        import trimesh

        from pyarchinit_photogrammetry.archaeo.pottery import Axis, profile_at

        mesh = trimesh.Trimesh(v, faces, process=False)
        axis = Axis(point, d)
        h0 = ((v - point) @ d).min()
        parts = []
        for k in range(angles):
            prof = profile_at(mesh, axis, 360.0 * k / angles)
            if prof.curves:
                pts = prof.points.copy()
                pts[:, 1] += h0  # profile_at measures from the lowest vertex: back to the axis point
                parts.append(pts)
        if parts:
            return np.vstack(parts)
    except ImportError:  # pragma: no cover - a program without the pottery's cut: the vertices
        pass
    q = v - point
    h = q @ d
    r = np.linalg.norm(q - h[:, None] * d, axis=1)
    return np.column_stack([r, h])


def column_profile(vertices, faces, point, direction, step: float | None = None, angles: int = 36,
                   rms: float = 0.0) -> ColumnMeasure:
    """The measured profile of a column part around its axis: the median radius per height (``step``: from the
    mesh's resolution when None) and its spread. The base is the lowest point of the part on the axis."""
    v = np.asarray(vertices, np.float64)
    d = unit(direction)
    p = np.asarray(point, np.float64)
    h_all = (v - p) @ d
    base = p + h_all.min() * d
    height = float(np.ptp(h_all))
    if step is None:
        step = max(1.5 * mesh_resolution(v, faces), height / 400.0, 0.002)
    rh = _profile_points(v, faces, base, d, angles)
    r, h = rh[:, 0], rh[:, 1]
    n = max(int(np.ceil(height / step)), 2)
    edges = np.linspace(0.0, height, n + 1)
    idx = np.clip(np.searchsorted(edges, h, side="right") - 1, 0, n - 1)
    heights, radii, spread = [], [], []
    for k in range(n):
        m = idx == k
        if m.sum() < 3:
            continue
        rr = r[m]
        q1, med, q3 = np.percentile(rr, [25, 50, 75])
        heights.append(0.5 * (edges[k] + edges[k + 1]))
        radii.append(med)
        spread.append(q3 - q1)
    if len(heights) < 4:
        raise ValueError("too few points on the part of the model to measure a profile")
    heights = np.asarray(heights)
    # the ends of the profile at the ends of the part: the first and last bins moved to 0 and the height
    heights[0], heights[-1] = 0.0, height
    return ColumnMeasure(base, d, height, heights, np.asarray(radii), np.asarray(spread), float(step), float(rms))


def _spline(zk, rk):
    from scipy.interpolate import CubicSpline

    return CubicSpline(np.asarray(zk, float), np.asarray(rk, float), bc_type="not-a-knot" if len(zk) > 3 else "natural")


def fit_knots(heights, radii, k: int = 12) -> tuple[np.ndarray, np.ndarray]:
    """K knots evenly spaced from the base to the top and their radii fitted by least squares so that the cubic
    spline through them follows the measured profile (the spline is linear in the radii of its knots)."""
    z = np.asarray(heights, float)
    r = np.asarray(radii, float)
    k = int(max(2, min(k, len(z))))
    zk = np.linspace(z.min(), z.max(), k)
    if k == 2:
        a = np.column_stack([(zk[1] - z) / (zk[1] - zk[0]), (z - zk[0]) / (zk[1] - zk[0])])
    else:
        a = np.column_stack([_spline(zk, np.eye(k)[j])(z) for j in range(k)])
    rk = np.linalg.lstsq(a, r, rcond=None)[0]
    return zk, rk


def profile_curve(zk, rk, step: float = 0.005, max_points: int = 400) -> np.ndarray:
    """The profile of the solid, ``(r, z)`` from base to top: the spline through the knots, sampled densely
    (the same points go to CadQuery, as a smooth spline, and to the IFC, as a polyline)."""
    zk, rk = np.asarray(zk, float), np.asarray(rk, float)
    if len(zk) == 2:
        return np.column_stack([rk, zk])
    n = int(np.clip(np.ceil((zk[-1] - zk[0]) / step) + 1, 16, max_points))
    z = np.linspace(zk[0], zk[-1], n)
    r = _spline(zk, rk)(z)
    if np.any(r <= 0):
        raise ValueError("the profile through the knots goes through the axis: a radius is zero or negative")
    return np.column_stack([r, z])


def entasis(profile) -> float:
    """The largest radius over the chord from the base radius to the top radius (metres; 0 for a straight or
    concave shaft)."""
    p = np.asarray(profile, float)
    r, z = p[:, 0], p[:, 1]
    chord = r[0] + (r[-1] - r[0]) * (z - z[0]) / max(z[-1] - z[0], 1e-12)
    return float(max((r - chord).max(), 0.0))


def revolved_volume(profile) -> float:
    """The volume of the solid of revolution of a polyline profile ``(r, z)`` (frusta)."""
    p = np.asarray(profile, float)
    r1, r2 = p[:-1, 0], p[1:, 0]
    h = np.diff(p[:, 1])
    return float(np.sum(np.pi / 3.0 * h * (r1 * r1 + r1 * r2 + r2 * r2)))


def revolved_mesh(profile, segments: int = 180):
    """The solid of revolution of ``(r, z)`` around the z axis as a closed triangle mesh (for checks)."""
    import trimesh

    p = np.asarray(profile, float)
    a = np.linspace(0.0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([np.cos(a), np.sin(a)])
    n = len(p)
    v = np.concatenate([np.column_stack([p[i, 0] * ring, np.full(segments, p[i, 1])]) for i in range(n)]
                       + [[[0.0, 0.0, p[0, 1]], [0.0, 0.0, p[-1, 1]]]])
    f = []
    for i in range(n - 1):
        for j in range(segments):
            a0, a1 = i * segments + j, i * segments + (j + 1) % segments
            b0, b1 = a0 + segments, a1 + segments
            f += [[a0, a1, b1], [a0, b1, b0]]
    bottom, top = n * segments, n * segments + 1
    for j in range(segments):
        f.append([bottom, (j + 1) % segments, j])
        f.append([top, (n - 1) * segments + j, (n - 1) * segments + (j + 1) % segments])
    return trimesh.Trimesh(v, np.asarray(f), process=False)


def column_frame(point, direction) -> np.ndarray:
    """Local frame of a column: z along the axis from the base point, x and y across it."""
    d = unit(direction)
    e1, e2 = perpendicular(d)
    return frame_matrix(point, e1, e2, d)


# ====================================================================== paths, sections, moulding
def path_frames(path) -> dict:
    """The frames of a polyline path: unit tangent of each segment, at each vertex the section's axes ``u`` (up x
    tangent, horizontal for a level path) and ``v`` (tangent x u, up for a level path) of the plane normal to the
    path, and the mitre at the vertices (the plane bisecting two segments, its scale ``1 / cos(half angle)``)."""
    p = np.asarray(path, np.float64)
    if len(p) < 2:
        raise ValueError("a path of two points at least")
    seg = np.diff(p, axis=0)
    keep = np.linalg.norm(seg, axis=1) > 1e-9
    if not keep.all():
        p = np.vstack([p[0], p[1:][keep]])
        seg = np.diff(p, axis=0)
    t = unit(seg)
    tv = np.vstack([t[0], unit(t[:-1] + t[1:]) if len(t) > 1 else np.zeros((0, 3)), t[-1]])  # vertex tangents
    upv = np.tile(UP, (len(p), 1))
    if np.any(np.abs((tv * UP).sum(1)) > 0.999):  # a vertical path: "up" is y
        upv[np.abs((tv * UP).sum(1)) > 0.999] = [0.0, 1.0, 0.0]
    u = unit(np.cross(upv, tv))
    v = np.cross(tv, u)
    scale = np.ones(len(p))
    if len(t) > 1:
        cos_half = np.clip((t[:-1] * tv[1:-1]).sum(1), 1e-3, 1.0)
        scale[1:-1] = 1.0 / cos_half
    return {"points": p, "tangents": t, "vertex_tangents": tv, "u": u, "v": v, "mitre": scale}


def path_length(path) -> float:
    return float(np.linalg.norm(np.diff(np.asarray(path, float), axis=0), axis=1).sum())


def point_at(path, station: float):
    """The point at ``station`` metres along the path and the tangent of its segment."""
    p = np.asarray(path, np.float64)
    seg = np.diff(p, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(lengths)]
    s = float(np.clip(station, 0.0, cum[-1]))
    i = int(np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1))
    f = (s - cum[i]) / max(lengths[i], 1e-12)
    return p[i] + f * seg[i], unit(seg[i])


def section_frame(tangent, origin):
    """The plane normal to the path at ``origin``: ``(origin, u, v, normal)``; normal = tangent."""
    t = unit(tangent)
    up = UP if abs(t @ UP) < 0.999 else np.array([0.0, 1.0, 0.0])
    u = unit(np.cross(up, t))
    v = np.cross(t, u)
    return np.asarray(origin, float), u, v, t


def mesh_section(vertices, faces, origin, u_axis, v_axis) -> list[np.ndarray]:
    """The polylines where the plane through ``origin`` spanned by ``u_axis`` and ``v_axis`` cuts the mesh, in
    ``(u, v)`` metres (the program's ``engine.elevation.cut_profile`` on a mesh of the plugin)."""
    from shapely.geometry import MultiLineString
    from shapely.ops import linemerge

    xyz = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    o, u, v = np.asarray(origin, float), unit(u_axis), unit(v_axis)
    n = np.cross(u, v)
    s = (xyz - o) @ n
    above = s >= 0
    fa = above[f]
    cross = fa.any(axis=1) & ~fa.all(axis=1)
    ff = f[cross]
    if not len(ff):
        return []
    pts, eds = [], []
    for i, j in ((0, 1), (1, 2), (2, 0)):
        # the edge in a fixed order (lower index first): the two faces of an edge give the same point, bit for bit,
        # and the segments join
        vi, vj = np.minimum(ff[:, i], ff[:, j]), np.maximum(ff[:, i], ff[:, j])
        edge = above[vi] != above[vj]
        tt = s[vi] / np.where(edge, s[vi] - s[vj], 1.0)
        pts.append(xyz[vi] + tt[:, None] * (xyz[vj] - xyz[vi]))
        eds.append(edge)
    points = np.stack(pts, axis=1)
    edges = np.stack(eds, axis=1)
    order = np.argsort(~edges, axis=1, kind="stable")[:, :2]
    ends = np.take_along_axis(points, order[:, :, None], axis=1) - o
    uv = np.stack([ends @ u, ends @ v], axis=-1)
    segs = [tuple(map(tuple, s2)) for s2 in uv if np.linalg.norm(s2[1] - s2[0]) > 1e-12]
    if not segs:
        return []
    merged = linemerge(MultiLineString(segs))
    parts = getattr(merged, "geoms", [merged])
    return [np.asarray(g.coords) for g in parts if not g.is_empty and len(g.coords) > 1]


def moulding_section(polylines, window, out: float = 1.0, depth: float = 0.1, simplify: float = 0.0) -> np.ndarray:
    """The section of a moulding from the cut of the model: the parts inside ``window`` ``(u0, v0, u1, v1)``, the
    polyline nearest the path point (0, 0) kept, closed against the wall: its two ends joined by a back line
    ``depth`` metres inside the wall (``out`` = +1 when the moulding projects towards +u, -1 towards -u).
    Returns the closed outline (N, 2) counter-clockwise, without the repeated first point."""
    from shapely.geometry import LineString, Point, Polygon, box

    win = box(*window)
    best, dist = None, np.inf
    for pl in polylines:
        if len(pl) < 2:
            continue
        g = LineString(pl).intersection(win)
        for part in getattr(g, "geoms", [g]):
            if part.is_empty or part.geom_type != "LineString":
                continue
            d = part.distance(Point(0.0, 0.0))
            if d < dist or (d == dist and part.length > best.length):
                best, dist = part, d
    if best is None:
        raise ValueError("the model has no section inside the window around the path")
    if simplify > 0:
        best = best.simplify(simplify, preserve_topology=False)
    c = np.asarray(best.coords)
    if np.linalg.norm(c[0] - c[-1]) < 1e-9:  # already a closed outline (a free-standing member)
        ring = c[:-1]
    else:
        if out >= 0:
            back = c[:, 0].min() - depth
        else:
            back = c[:, 0].max() + depth
        ring = np.vstack([c, [[back, c[-1, 1]], [back, c[0, 1]]]])
    poly = Polygon(ring)
    if not poly.is_valid:
        poly = poly.buffer(0)
        if poly.geom_type != "Polygon":
            poly = max(poly.geoms, key=lambda g: g.area)
        ring = np.asarray(poly.exterior.coords)[:-1]
    if Polygon(ring).exterior.is_ccw is False:
        ring = ring[::-1]
    return ring


def polygon_area(ring) -> float:
    r = np.asarray(ring, float)
    x, y = r[:, 0], r[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def sweep_mesh(path, section):
    """The section ``(u, v)`` swept along ``path`` with mitred corners, as a closed triangle mesh: the solid to
    expect (its volume) and the tessellation for checks. Each vertex of the path carries the section in its mitre
    plane, stretched across the bisector by the mitre scale."""
    import trimesh

    fr = path_frames(path)
    sec = np.asarray(section, float)
    if polygon_area(sec) < 0:
        sec = sec[::-1]
    p, u, v, sc = fr["points"], fr["u"], fr["v"], fr["mitre"]
    m = len(sec)
    rings = []
    t = fr["tangents"]
    for i in range(len(p)):
        uu, vv = u[i] * sc[i], v[i]  # in the mitre plane: the bisector's u, stretched (1 at the ends)
        rings.append(p[i] + sec[:, :1] * uu + sec[:, 1:2] * vv)
    vert = np.vstack(rings)
    faces = []
    for i in range(len(p) - 1):
        for j in range(m):
            a0, a1 = i * m + j, i * m + (j + 1) % m
            b0, b1 = a0 + m, a1 + m
            faces += [[a0, b1, b0], [a0, a1, b1]]  # outwards: the section is counter-clockwise in (u, v)
    cap = _triangulate(sec)
    cap = np.asarray(cap)
    # orient the caps outwards: the start cap faces -t, the end cap +t
    n0 = len(p) - 1
    for k, (tri_, base, sign) in enumerate(((cap, 0, -1.0), (cap, n0 * m, 1.0))):
        tri_ = tri_ + base
        a, b, c = vert[tri_[:, 0]], vert[tri_[:, 1]], vert[tri_[:, 2]]
        nrm = np.cross(b - a, c - a)
        tt = t[0] if sign < 0 else t[-1]
        flip = (nrm @ tt) * sign < 0
        tri_[flip] = tri_[flip][:, ::-1]
        faces += tri_.tolist()
    mesh = trimesh.Trimesh(vert, np.asarray(faces), process=False)
    if mesh.volume < 0:
        mesh.invert()
    return mesh


def prism_mesh(polygon, depth: float):
    """A planar polygon (N, 2), holes not allowed, extruded from z = 0 to z = ``depth`` as a closed mesh."""
    import trimesh

    poly = np.asarray(polygon, float)
    if len(poly) > 1 and np.allclose(poly[0], poly[-1]):
        poly = poly[:-1]
    if polygon_area(poly) < 0:
        poly = poly[::-1]
    n = len(poly)
    v = np.vstack([np.column_stack([poly, np.zeros(n)]), np.column_stack([poly, np.full(n, float(depth))])])
    tri = np.asarray(_triangulate(poly))
    faces = [tri[:, ::-1], tri + n]
    j = np.arange(n)
    k = (j + 1) % n
    faces.append(np.column_stack([j, k, k + n]))
    faces.append(np.column_stack([j, k + n, j + n]))
    return trimesh.Trimesh(v, np.vstack(faces), process=False)


def _triangulate(poly) -> list[list[int]]:
    """Triangles (indices into ``poly``) of a simple polygon: shapely's constrained Delaunay triangulation, ear
    clipping when that is not there."""
    import shapely
    from shapely.geometry import Polygon

    if hasattr(shapely, "constrained_delaunay_triangles"):
        lookup = {tuple(np.round(q, 12)): k for k, q in enumerate(poly)}
        out = []
        for g in getattr(shapely.constrained_delaunay_triangles(Polygon(poly)), "geoms", []):
            try:
                out.append([lookup[tuple(np.round(q, 12))] for q in np.asarray(g.exterior.coords)[:3]])
            except KeyError:
                out = []
                break
        if out:
            tri = np.asarray(out)
            a, b, c = poly[tri[:, 0]], poly[tri[:, 1]], poly[tri[:, 2]]
            cw = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])) < 0
            tri[cw] = tri[cw][:, ::-1]  # every triangle counter-clockwise, as the polygon
            return tri.tolist()
    return _ear_clip(poly)


def _ear_clip(poly) -> list[list[int]]:
    """Triangles of a simple polygon by ear clipping (counter-clockwise input)."""
    idx = list(range(len(poly)))
    out = []

    def area(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])

    guard = 0
    while len(idx) > 3 and guard < 10000:
        guard += 1
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            a, b, c = poly[i0], poly[i1], poly[i2]
            if area(a, b, c) <= 0:
                continue
            ok = True
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                q = poly[j]
                if area(a, b, q) >= 0 and area(b, c, q) >= 0 and area(c, a, q) >= 0:
                    ok = False
                    break
            if ok:
                out.append([i0, i1, i2])
                idx.pop(k)
                break
        else:
            break
    if len(idx) == 3:
        out.append(idx)
    return out


def sweep_volume(path, section) -> float:
    """The volume of the section swept along the path with mitred corners: area x length of the path of the
    section's centroid (Pappus for a polyline with mitres)."""
    return float(abs(sweep_mesh(path, section).volume))


def is_straight(path, tolerance: float = 1e-6) -> bool:
    p = np.asarray(path, float)
    if len(p) == 2:
        return True
    d = unit(p[-1] - p[0])
    q = p - p[0]
    return bool(np.max(np.linalg.norm(q - (q @ d)[:, None] * d, axis=1)) < tolerance)


def model_height_at(vertices, faces, xy, radius: float = 0.05) -> float:
    """The height of the model's top surface at a plan point (the highest vertex within ``radius``), NaN when no
    vertex is there."""
    v = np.asarray(vertices, float)
    m = np.hypot(v[:, 0] - xy[0], v[:, 1] - xy[1]) < radius
    return float(v[m, 2].max()) if m.any() else float("nan")


# ====================================================================== arch
def visible_points(polylines, pick, rays: int = 1440, down: float = 60.0):
    """What a point inside an opening sees of a section: the first crossing of each ray (the intrados and the
    jambs) and the last one (beyond the ring: the extrados when the arch stands free). The rays go all round but
    for a cone of ``down`` degrees under the point (the floor, the sill). Returns ``(near, far)`` (N, 2)."""
    from shapely.geometry import LineString, MultiLineString

    lines = MultiLineString([LineString(p) for p in polylines if len(p) > 1])
    if lines.is_empty:
        return np.zeros((0, 2)), np.zeros((0, 2))
    pick = np.asarray(pick, float)
    b = lines.bounds
    reach = 2.0 * np.hypot(b[2] - b[0], b[3] - b[1]) + np.hypot(*(pick - [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2]))
    half = np.radians(down) / 2
    start = -np.pi / 2 + half
    angles = np.linspace(start, start + 2 * np.pi - 2 * half, rays)
    near, far = [], []
    for a in angles:
        end = pick + reach * np.array([np.cos(a), np.sin(a)])
        hit = lines.intersection(LineString([pick, end]))
        if hit.is_empty:
            continue
        pts = []
        for g in getattr(hit, "geoms", [hit]):
            if g.geom_type == "Point":
                pts.append((g.x, g.y))
            else:
                pts.extend(np.asarray(g.coords).tolist())
        if not pts:
            continue
        pts = np.asarray(pts)
        d = np.hypot(*(pts - pick).T)
        near.append(pts[d.argmin()])
        if len(pts) > 1:
            far.append(pts[d.argmax()])
    return np.asarray(near).reshape(-1, 2), np.asarray(far).reshape(-1, 2)


def guess_pick(polylines, grid: int = 40) -> tuple[float, float]:
    """A point inside an opening of a section, for the dialog to start from: among the points of a grid that see a
    line above, to the left and to the right of them (and are not inside closed material: an even number of
    crossings upwards), the one farthest from every line (the middle of the largest opening)."""
    from shapely.geometry import LineString, MultiLineString, Point

    lines = MultiLineString([LineString(p) for p in polylines if len(p) > 1])
    if lines.is_empty:
        return 0.0, 0.0
    x0, y0, x1, y1 = lines.bounds
    reach = 2 * max(x1 - x0, y1 - y0)
    best, score = ((x0 + x1) / 2, (y0 + y1) / 2), -1.0
    for x in np.linspace(x0, x1, grid + 2)[1:-1]:
        for y in np.linspace(y0, y1, grid + 2)[1:-1]:
            up = lines.intersection(LineString([(x, y), (x, y + reach)]))
            if up.is_empty or len(getattr(up, "geoms", [up])) % 2:
                continue
            if lines.intersection(LineString([(x, y), (x - reach, y)])).is_empty or \
                    lines.intersection(LineString([(x, y), (x + reach, y)])).is_empty:
                continue
            d = lines.distance(Point(x, y))
            if d > score:
                best, score = (float(x), float(y)), d
    return best


@dataclass
class ArchFit:
    """A fitted arch in the section plane (u, v metres): ``type`` (one of :data:`ARCH_TYPES`), ``family``
    (circle, pointed, ellipse, measured), ``params`` (all numbers), ``rms`` of the intrados points, ``springing``
    (the height of the springing line), ``span``, ``rise``, ``centre_u`` (the axis of the arch)."""

    type: str
    family: str
    params: dict
    rms: float
    springing: float
    span: float
    rise: float
    centre_u: float
    points: int
    candidates: dict = field(default_factory=dict)
    measured: list | None = None  # the intrados as measured (u, v), left to right

    def to_dict(self) -> dict:
        return {"type": self.type, "family": self.family, "params": self.params, "rms": self.rms,
                "springing": self.springing, "span": self.span, "rise": self.rise, "centre_u": self.centre_u,
                "points": self.points, "candidates": self.candidates, "measured": self.measured}


def _jamb_res(u, uL, uR):
    return np.minimum(np.abs(u - uL), np.abs(u - uR))


def _fit_circle_family(P, scale):
    from scipy.optimize import least_squares

    u, v = P[:, 0], P[:, 1]
    top = v > np.percentile(v, 40)
    c = _circle_fit(P[top]) or _circle_fit(P)
    if c is None:
        return None
    cx, cy, r, _ = c

    def model(x):
        uc, vc, R, s = x
        vs = vc + s
        half = np.sqrt(max(R * R - s * s, 0.0))
        arc = v >= vs
        res = np.where(arc, np.hypot(u - uc, v - vc) - R, _jamb_res(u, uc - half, uc + half))
        return res

    best = None
    for s0 in (0.0, 0.2 * r, 0.5 * r, 0.8 * r):
        x0 = np.array([cx, cy, r, s0])
        try:
            sol = least_squares(model, x0, bounds=([-np.inf, -np.inf, 1e-4, 0.0], [np.inf, np.inf, np.inf, np.inf]),
                                loss="soft_l1", f_scale=scale)
        except ValueError:
            continue
        rms = float(np.sqrt(np.mean(model(sol.x) ** 2)))
        if best is None or rms < best[1]:
            best = (sol.x, rms)
    if best is None:
        return None
    (uc, vc, R, s), rms = best
    s = min(s, R)
    return {"uc": float(uc), "vc": float(vc), "R": float(R), "vs": float(vc + s)}, rms


def _fit_pointed(P, scale):
    from scipy.optimize import least_squares

    u, v = P[:, 0], P[:, 1]
    um0 = float(u[v >= np.percentile(v, 95)].mean())
    s0 = 0.5 * float(np.ptp(u))

    def model(x):
        um, vs, s, k = x
        R = s * k
        cL, cR = um - s + R, um + s - R
        arc = v >= vs
        left = u < um
        d = np.where(left, np.hypot(u - cL, v - vs), np.hypot(u - cR, v - vs)) - R
        return np.where(arc, d, _jamb_res(u, um - s, um + s))

    best = None
    vlo, vhi = float(v.min()), float(v.max())
    for f in (0.2, 0.4, 0.6):
        for k0 in (1.2, 1.7, 2.5):
            x0 = np.array([um0, vlo + f * (vhi - vlo) * 0.6, s0, k0])
            try:
                sol = least_squares(model, x0, bounds=([-np.inf, -np.inf, 1e-4, 1.0], [np.inf, np.inf, np.inf, 50.0]),
                                    loss="soft_l1", f_scale=scale)
            except ValueError:
                continue
            rms = float(np.sqrt(np.mean(model(sol.x) ** 2)))
            if best is None or rms < best[1]:
                best = (sol.x, rms)
    if best is None:
        return None
    (um, vs, s, k), rms = best
    return {"um": float(um), "vs": float(vs), "s": float(s), "R": float(s * k)}, rms


def _fit_ellipse(P, scale):
    from scipy.optimize import least_squares

    u, v = P[:, 0], P[:, 1]
    c = _circle_fit(P[v > np.percentile(v, 40)]) or _circle_fit(P)
    if c is None:
        return None
    cx, cy, r, _ = c

    def model(x):
        uc, vc, a, b, s = x
        vs = vc + s
        du, dv = u - uc, v - vc
        F = (du / a) ** 2 + (dv / b) ** 2 - 1.0
        g = 2.0 * np.hypot(du / a ** 2, dv / b ** 2)
        samp = F / np.maximum(g, 1e-12)
        half = a * np.sqrt(max(1.0 - (s / b) ** 2, 0.0))
        return np.where(v >= vs, samp, _jamb_res(u, uc - half, uc + half))

    best = None
    for s0 in (0.0, 0.3 * r):
        for ab in ((1.0, 1.0), (1.0, 0.6), (0.7, 1.0)):
            x0 = np.array([cx, cy, r * ab[0], r * ab[1], s0])
            try:
                sol = least_squares(model, x0, bounds=([-np.inf, -np.inf, 1e-4, 1e-4, 0.0], [np.inf] * 5),
                                    loss="soft_l1", f_scale=scale)
            except ValueError:
                continue
            rms = float(np.sqrt(np.mean(model(sol.x) ** 2)))
            if best is None or rms < best[1]:
                best = (sol.x, rms)
    if best is None:
        return None
    (uc, vc, a, b, s), rms = best
    s = min(s, b)
    return {"uc": float(uc), "vc": float(vc), "a": float(a), "b": float(b), "vs": float(vc + s)}, rms


def _arch_from(family: str, params: dict, rms: float, n: int, lowest: float | None = None) -> ArchFit:
    if family == "circle":
        uc, vc, R, vs = params["uc"], params["vc"], params["R"], params["vs"]
        if lowest is not None and vs < lowest and vs - vc < 0.02 * R:
            pass
        s = vs - vc
        half = float(np.sqrt(max(R * R - s * s, 0.0)))
        kind = "semicircular" if s < 0.02 * R else "segmental"
        return ArchFit(kind, family, params, rms, vs, 2 * half, vc + R - vs, uc, n)
    if family == "pointed":
        um, vs, s, R = params["um"], params["vs"], params["s"], params["R"]
        rise = float(np.sqrt(max(R * R - (R - s) ** 2, 0.0)))
        return ArchFit("pointed", family, params, rms, vs, 2 * s, rise, um, n)
    if family == "ellipse":
        uc, vc, a, b, vs = params["uc"], params["vc"], params["a"], params["b"], params["vs"]
        s = vs - vc
        half = a * float(np.sqrt(max(1 - (s / b) ** 2, 0.0)))
        return ArchFit("elliptical", family, params, rms, vs, 2 * half, vc + b - vs, uc, n)
    raise ValueError(family)


def _arc_only(P, fit: ArchFit) -> np.ndarray:
    return P[P[:, 1] >= fit.springing - 1e-9]


def fit_arch(points, kind: str = "auto", noise: float = 0.002) -> ArchFit:
    """The arch that best explains the intrados ``points`` (u, v): ``kind`` one of "auto", "semicircular" /
    "segmental" (a circle), "pointed", "elliptical", "measured". Points below the springing line are the jambs
    (vertical lines from the springing points). "auto": the simplest family whose RMS is within 25 % (+ 0.3 mm)
    of the best one (a circle before a pointed arch before an ellipse)."""
    P = np.asarray(points, float)
    if len(P) < 8:
        raise ValueError("too few points of the intrados to fit an arch")
    families = {"auto": FIT_FAMILIES, "semicircular": ("circle",), "segmental": ("circle",), "circle": ("circle",),
                "pointed": ("pointed",), "elliptical": ("ellipse",), "ellipse": ("ellipse",),
                "measured": FIT_FAMILIES}[kind]
    fits = {}
    for fam in families:
        f = {"circle": _fit_circle_family, "pointed": _fit_pointed, "ellipse": _fit_ellipse}[fam](P, noise)
        if f is not None:
            fits[fam] = f
    if not fits:
        raise ValueError("no arch fits the intrados")
    best_rms = min(r for _p, r in fits.values())
    chosen = None
    for fam in FIT_FAMILIES:
        if fam in fits and fits[fam][1] <= best_rms * 1.25 + 0.0003:
            chosen = fam
            break
    params, rms = fits[chosen]
    fit = _arch_from(chosen, params, rms, len(P))
    # no jamb under the springing line: the springing line is the chord through the lowest points of the arc
    if not (P[:, 1] < fit.springing - 3 * max(rms, 1e-4)).any():
        left = P[P[:, 0] < fit.centre_u]
        right = P[P[:, 0] >= fit.centre_u]
        if len(left) and len(right):
            vs = 0.5 * (left[:, 1].min() + right[:, 1].min())
            params = dict(params, vs=float(max(vs, params.get("vc", -np.inf))))
            fit = _arch_from(chosen, params, rms, len(P))
    fit.candidates = {fam: {"rms": r, "params": p} for fam, (p, r) in fits.items()}
    arc = _arc_only(P, fit)
    order = np.argsort(arc[:, 0])
    fit.measured = arc[order].tolist()
    if kind == "measured":
        fit.type, fit.family = "measured", "measured"
        fit.rms = 0.0
    return fit


def arch_curve(fit: ArchFit, offset: float = 0.0, n: int = 96) -> np.ndarray:
    """The intrados (``offset`` 0) or a curve ``offset`` metres outside it, from the left springing point to the
    right one (u, v)."""
    p = fit.params
    vs = fit.springing
    if fit.family == "circle":
        uc, vc, R = p["uc"], p["vc"], p["R"] + offset
        a0 = np.arcsin(np.clip((vs - vc) / R, -1, 1))
        a = np.linspace(np.pi - a0, a0, n)
        return np.column_stack([uc + R * np.cos(a), vc + R * np.sin(a)])
    if fit.family == "pointed":
        um, s, R = p["um"], p["s"], p["R"]
        Ro = R + offset
        cL, cR = um - s + R, um + s - R
        top = np.arccos(np.clip((um - cL) / Ro, -1, 1))  # angle of the apex seen from the left centre
        aL = np.linspace(np.pi, top, n // 2)
        left = np.column_stack([cL + Ro * np.cos(aL), vs + Ro * np.sin(aL)])
        aR = np.linspace(np.pi - top, 0.0, n // 2)
        right = np.column_stack([cR + Ro * np.cos(aR), vs + Ro * np.sin(aR)])
        return np.vstack([left, right[1:]])
    if fit.family == "ellipse":
        uc, vc, a, b = p["uc"], p["vc"], p["a"] + offset, p["b"] + offset
        a0 = np.arcsin(np.clip((vs - vc) / b, -1, 1))
        t = np.linspace(np.pi - a0, a0, n)
        return np.column_stack([uc + a * np.cos(t), vc + b * np.sin(t)])
    if fit.family == "measured":
        from shapely.geometry import LineString

        m = np.asarray(fit.measured, float)
        line = LineString(m)
        if offset:
            line = line.offset_curve(-offset if _left_to_right(m) else offset, join_style="round")
        c = np.asarray(line.coords)
        if c[0, 0] > c[-1, 0]:
            c = c[::-1]
        return c
    raise ValueError(fit.family)


def _left_to_right(m) -> bool:
    return bool(m[-1, 0] >= m[0, 0])


def arch_thickness(fit: ArchFit, extrados) -> float | None:
    """The thickness of the ring: the median distance of the extrados points (above the springing line) to the
    intrados; None without extrados points."""
    from scipy.spatial import cKDTree

    e = np.asarray(extrados, float).reshape(-1, 2)
    e = e[e[:, 1] > fit.springing + 0.02 * max(fit.rise, 1e-3)]
    if len(e) < 5:
        return None
    curve = arch_curve(fit, 0.0, n=2000)
    d, _ = cKDTree(curve).query(e)
    # only the points beyond the intrados, near it (the far side of a wall around an opening is not an extrados)
    d = d[d > 1e-4]
    if len(d) < 5:
        return None
    return float(np.median(d))


def arch_outline(fit: ArchFit, thickness: float) -> dict:
    """The section of the arch ring above the springing line, as a closed curve of points and segments (lines
    ``["L", i, j]`` and three-point arcs ``["A", i, j, k]``, indices from 0), counter-clockwise: the same curve
    goes to CadQuery (lines and ``threePointArc``) and to the IFC (``IfcIndexedPolyCurve`` with
    ``IfcLineIndex``/``IfcArcIndex``)."""
    t = float(thickness)
    if t <= 0:
        raise ValueError("the thickness of the arch must be positive")
    p, vs = fit.params, fit.springing
    if fit.family == "circle":
        uc, vc, R = p["uc"], p["vc"], p["R"]
        s = vs - vc
        hi = np.sqrt(max(R * R - s * s, 0.0))
        he = np.sqrt(max((R + t) ** 2 - s * s, 0.0))
        pts = [[uc + hi, vs], [uc + he, vs], [uc, vc + R + t], [uc - he, vs], [uc - hi, vs], [uc, vc + R]]
        segs = [["L", 0, 1], ["A", 1, 2, 3], ["L", 3, 4], ["A", 4, 5, 0]]
        return {"points": pts, "segments": segs}
    if fit.family == "pointed":
        um, s, R = p["um"], p["s"], p["R"]
        cL, cR = um - s + R, um + s - R
        apex_i = vs + np.sqrt(max(R * R - (R - s) ** 2, 0.0))
        apex_e = vs + np.sqrt(max((R + t) ** 2 - (R - s) ** 2, 0.0))

        def mid(c, Rr, a_from, a_to):
            a = 0.5 * (a_from + a_to)
            return [c + Rr * np.cos(a), vs + Rr * np.sin(a)]

        top_i = np.arctan2(apex_i - vs, um - cL)
        top_e = np.arctan2(apex_e - vs, um - cL)
        pts = [[um + s, vs], [um + s + t, vs], mid(cR, R + t, 0.0, np.pi - top_e), [um, apex_e],
               mid(cL, R + t, top_e, np.pi), [um - s - t, vs], [um - s, vs], mid(cL, R, np.pi, top_i), [um, apex_i],
               mid(cR, R, np.pi - top_i, 0.0)]
        segs = [["L", 0, 1], ["A", 1, 2, 3], ["A", 3, 4, 5], ["L", 5, 6], ["A", 6, 7, 8], ["A", 8, 9, 0]]
        return {"points": pts, "segments": segs}
    inner = arch_curve(fit, 0.0)
    outer = arch_curve(fit, t)
    if fit.family == "measured":
        inner = np.vstack([inner[inner[:, 1] >= vs - 1e-9]])
        outer = outer[outer[:, 1] >= vs - 1e-9]
        outer = np.vstack([[outer[0, 0], vs], outer, [outer[-1, 0], vs]]) if len(outer) else outer
    ring = np.vstack([inner[::-1], outer])  # right to left inside, then left to right outside: clockwise
    ring = ring[::-1]
    n = len(ring)
    return {"points": ring.tolist(), "segments": [["L", i, (i + 1) % n] for i in range(n)]}


def outline_polygon(outline: dict, arc_points: int = 48) -> np.ndarray:
    """The closed curve of :func:`arch_outline` as a dense polygon (N, 2) (lines kept, arcs sampled)."""
    pts = np.asarray(outline["points"], float)
    out = []
    for seg in outline["segments"]:
        if seg[0] == "L":
            out.append(pts[seg[1]])
        else:
            out.extend(_arc_points(pts[seg[1]], pts[seg[2]], pts[seg[3]], arc_points)[:-1])
    return np.asarray(out)


def _arc_points(a, b, c, n):
    """Points along the circle through a, b, c from a to c through b."""
    ax, ay = a
    bx, by = b
    cx, cy = c
    d = 2 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-15:
        return np.linspace(a, c, n)
    ux = ((ax ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay) + (cx ** 2 + cy ** 2) * (ay - by)) / d
    uy = ((ax ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax - cx) + (cx ** 2 + cy ** 2) * (bx - ax)) / d
    r = np.hypot(ax - ux, ay - uy)
    t0, t1, t2 = (np.arctan2(q[1] - uy, q[0] - ux) for q in (a, b, c))

    def ccw(x, y):
        return (y - x) % (2 * np.pi)

    if ccw(t0, t1) < ccw(t0, t2):
        ts = t0 + np.linspace(0, ccw(t0, t2), n)
    else:
        ts = t0 - np.linspace(0, ccw(t2, t0), n)
    return np.column_stack([ux + r * np.cos(ts), uy + r * np.sin(ts)])


def footprint_depth(points_uvd, outline_poly) -> tuple[float, float] | None:
    """The extent along the plane's normal (d) of the model's points whose (u, v) is inside the arch's
    section (the 1st to 99th percentile): the depth of a vault. None when too few points are there."""
    q = np.asarray(points_uvd, float)
    inside = points_in_polygon(q[:, :2], outline_poly)
    if inside.sum() < 20:
        return None
    d = q[inside, 2]
    lo, hi = np.percentile(d, [1, 99])
    return float(lo), float(hi)


def arch_frame(origin, u_axis, v_axis) -> np.ndarray:
    """Local frame of an arch: x = u, y = v (the section plane), z = u x v (the depth of the vault)."""
    u, v = unit(u_axis), unit(v_axis)
    return frame_matrix(origin, u, v, np.cross(u, v))


# ====================================================================== the fit of the model to the solid
def distances_to_mesh(points, vertices, faces, max_edge: float | None = None, k: int = 16) -> np.ndarray:
    """The distance of each point to a triangle mesh (exact point-to-triangle distance for the ``k`` triangles
    nearest by their centres; the mesh is subdivided first so that no triangle is longer than ``max_edge``)."""
    import trimesh
    from scipy.spatial import cKDTree

    P = np.asarray(points, np.float64)
    v = np.asarray(vertices, np.float64)
    f = np.asarray(faces, np.int64)
    if max_edge is None:
        size = float(np.ptp(v, axis=0).max())
        max_edge = max(size / 60.0, 1e-3)
    try:
        v, f = trimesh.remesh.subdivide_to_size(v, f, max_edge=max_edge, max_iter=20)
    except ValueError:
        pass
    tri = v[f]
    centres = tri.mean(axis=1)
    k = int(min(k, len(f)))
    _d, idx = cKDTree(centres).query(P, k=k)
    idx = idx.reshape(len(P), k)
    best = np.full(len(P), np.inf)
    for j in range(k):
        t = tri[idx[:, j]]
        d = _point_triangle_distance(P, t[:, 0], t[:, 1], t[:, 2])
        best = np.minimum(best, d)
    return best


def _point_triangle_distance(p, a, b, c) -> np.ndarray:
    """Distance from points to triangles (row by row), Ericson's closest point on a triangle."""
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(1)
    d2 = (ac * ap).sum(1)
    bp = p - b
    d3 = (ab * bp).sum(1)
    d4 = (ac * bp).sum(1)
    cp = p - c
    d5 = (ab * cp).sum(1)
    d6 = (ac * cp).sum(1)
    vc = d1 * d4 - d3 * d2
    vb = d5 * d2 - d1 * d6
    va = d3 * d6 - d5 * d4
    denom = va + vb + vc
    denom = np.where(np.abs(denom) < 1e-30, 1e-30, denom)
    v = vb / denom
    w = vc / denom
    closest = a + ab * v[:, None] + ac * w[:, None]  # inside the face
    t3 = d1 / np.where(np.abs(d1 - d3) < 1e-30, 1e-30, d1 - d3)
    t5 = d2 / np.where(np.abs(d2 - d6) < 1e-30, 1e-30, d2 - d6)
    den6 = (d4 - d3) + (d5 - d6)
    t6 = (d4 - d3) / np.where(np.abs(den6) < 1e-30, 1e-30, den6)
    # the regions in reverse order of Ericson's tests: the first test that holds is written last and wins
    regions = (((va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0), b + (c - b) * t6[:, None]),
               ((vb <= 0) & (d2 >= 0) & (d6 <= 0), a + ac * t5[:, None]),
               ((d6 >= 0) & (d5 <= d6), c),
               ((vc <= 0) & (d1 >= 0) & (d3 <= 0), a + ab * t3[:, None]),
               ((d3 >= 0) & (d4 <= d3), b),
               ((d1 <= 0) & (d2 <= 0), a))
    for mask, point in regions:
        closest[mask] = point[mask]
    return np.linalg.norm(p - closest, axis=1)


def fit_report(distances) -> dict:
    """The numbers of the fit: median, 90th percentile and maximum distance (metres) and the number of points."""
    d = np.asarray(distances, float)
    d = d[np.isfinite(d)]
    if not len(d):
        return {"median": None, "p90": None, "max": None, "points": 0}
    return {"median": float(np.median(d)), "p90": float(np.percentile(d, 90)), "max": float(d.max()),
            "points": int(len(d))}
