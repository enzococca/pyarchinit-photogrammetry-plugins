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


def _circle_fit(xy: np.ndarray, trim: bool = True):
    """Circle by algebraic least squares then geometric refinement: ``(cx, cy, r, rms)`` or None. ``trim``: twice
    more on the points within 3 robust deviations of the circle (the fins and the flying points of a photogrammetric
    mesh at a silhouette, a chip of the shaft), the rms of those."""
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
    if trim:  # a robust start: the median radius around the median centre of the points
        c0 = np.median(p, axis=0)
        r0 = float(np.median(np.linalg.norm(p - c0, axis=1)))
    keep = np.ones(len(p), bool)
    x = np.r_[c0, r0]
    for _ in range(3 if trim else 1):
        q = p[keep]
        if len(q) < 5:
            break

        def res(x, q=q):
            return np.linalg.norm(q - x[:2], axis=1) - x[2]

        x = least_squares(res, x, loss="soft_l1" if trim else "linear",
                          f_scale=max(0.002, 0.01 * abs(x[2]))).x
        if not trim:
            break
        r_all = np.linalg.norm(p - x[:2], axis=1) - x[2]
        mad = 1.4826 * np.median(np.abs(r_all[keep] - np.median(r_all[keep]))) + 1e-5
        keep = np.abs(r_all) < 3 * mad
    r_all = np.linalg.norm(p - x[:2], axis=1) - x[2]
    rms = float(np.sqrt(np.mean(r_all[keep] ** 2))) if keep.any() else float("inf")
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
    search (the axis of another element: "same axis as element X"). Two starts, each refined on the circles of
    its slices: the axis the pottery finds from the normals (any direction: a fallen shaft) and the vertical through
    the middle of the part (a standing column, whose normals the fins of a photogrammetric mesh may spoil); the one
    whose slices are rounder wins. The direction points up (positive z) unless the shaft lies flat."""
    v = np.asarray(vertices, np.float64)
    starts = []
    if initial is not None:
        starts.append((np.asarray(initial[0], float), unit(initial[1])))
    else:
        try:
            import trimesh

            from pyarchinit_photogrammetry.archaeo.pottery import estimate_axis

            ax = estimate_axis(trimesh.Trimesh(v, faces, process=False))
            starts.append((ax.point, ax.direction))
        except (ImportError, ValueError):  # the program without its pottery, or too flat for the normals
            pass
        c = np.median(v, axis=0)
        starts.append((c, UP.copy()))
    best = None
    for p0, d0 in starts:
        if abs(d0[2]) > 0.5 and d0[2] < 0:
            d0 = -d0
        p, d, rms = refine_axis(v, p0, d0)
        if best is None or rms < best[2]:
            best = (p, d, rms)
    p, d, rms = best
    if abs(d[2]) > 0.5 and d[2] < 0:
        d = -d
    return p, d, rms


def _profile_points(vertices, faces, point, direction, angles: int):
    """(r, h) points of the model on ``angles`` half-planes through the axis (the pottery's cuts), h from the
    point on the axis, only where the cut line is steeper than 17 degrees (a bed, a top, a cap cut through the axis
    would give every radius at one height); the vertices themselves when the cut cannot be made."""
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
            for c in prof.curves:
                c = np.asarray(c, float)
                if len(c) < 2:
                    continue
                seg = np.diff(c, axis=0)
                steep = np.abs(seg[:, 1]) > 0.3 * np.abs(seg[:, 0])  # the shaft's surface, not a bed or a top
                keep = np.r_[steep, False] | np.r_[False, steep]
                pts = c[keep].copy()
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
        rr = _shaft_radii(r[m])
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


def _shaft_radii(rr: np.ndarray, band: float | None = None) -> np.ndarray:
    """The radii of one height that belong to the shaft: around the densest radius (a smoothed histogram's peak at
    a millimetre), not the fins and the flying points of a photogrammetric mesh at the silhouette, which scatter
    outwards and pull a median with them. (The beds and tops are left out before: :func:`_profile_points`.)"""
    rr = np.asarray(rr, float)
    if band is None:  # 3 mm on a radius of 30 cm
        band = max(0.003, 0.01 * float(np.median(rr))) if len(rr) else 0.003
    if len(rr) < 20 or np.ptp(rr) < 2 * band:
        return rr
    bins = int(np.clip(np.ptp(rr) / 0.001, 10, 600))
    hist, edges = np.histogram(rr, bins=bins)
    smooth = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
    k = int(np.argmax(smooth))
    peak = 0.5 * (edges[k] + edges[k + 1])
    near = rr[np.abs(rr - peak) < band]
    if len(near) < 3:
        return rr
    centre = np.median(near)
    return rr[np.abs(rr - centre) < band]


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


SEARCH = (-1.0, -1.0, 1.0, 1.0)  # where the cut is looked at around the path's point when no window is given
MARGIN = 0.1  # the window reported: the moulding's extent and this much around it


def wall_section(polylines, out: float, depth: float = 0.1, simplify: float = 0.0, resolution: float = 0.005,
                 window=None) -> dict:
    """The section of a moulding standing out of a wall, from the cut of the model: the line of the cut nearest the
    path's point inside ``window`` (default :data:`SEARCH`); the **wall face** on it (the u where its near-vertical
    stretches are, weighted by their length); the stretch of the line that stands out of the face by more than the
    mesh's ``resolution`` towards ``out`` (+1: towards +u), the one nearest the path's point; its two ends brought
    onto the face, the stretch simplified (Douglas-Peucker, ``simplify``) and closed by a back line ``depth`` behind
    the face. A closed line (a free-standing member) is kept as it is.

    Returns ``{"section", "wall_u", "v_range", "projection", "height", "window", "line"}``: the projection is
    measured from the wall face, the height is the moulding's own (``v_range``), ``window`` is the moulding's extent
    plus :data:`MARGIN` (u0, v0, u1, v1), ``line`` the line of the cut that was used (for an edit)."""
    from shapely.geometry import LineString, Point, box

    win = box(*(window if window is not None else SEARCH))
    best, dist = None, np.inf
    for pl in polylines:
        if len(pl) < 2:
            continue
        g = LineString(pl).intersection(win)
        for part in getattr(g, "geoms", [g]):
            if part.is_empty or part.geom_type != "LineString":
                continue
            d = part.distance(Point(0.0, 0.0))
            if d < dist:
                best, dist = part, d
    if best is None:
        raise ValueError("the model has no section around the path's point")
    c = np.asarray(best.coords)
    if np.linalg.norm(c[0] - c[-1]) < 1e-9:  # a free-standing member: its whole outline
        ring = c[:-1]
        if simplify > 0:
            from shapely.geometry import Polygon

            simple = Polygon(ring).simplify(simplify, preserve_topology=True)
            if simple.geom_type == "Polygon" and not simple.is_empty:
                ring = np.asarray(simple.exterior.coords)[:-1]
        ring = drop_repeated(ring, 1e-6)
        if polygon_area(ring) < 0:
            ring = ring[::-1]
        lo, hi = ring.min(axis=0), ring.max(axis=0)
        return {"section": ring, "wall_u": None, "v_range": [float(lo[1]), float(hi[1])], "projection": None,
                "width": float(hi[0] - lo[0]), "height": float(hi[1] - lo[1]), "front_area": abs(polygon_area(ring)),
                "warning": "", "closed": True,
                "window": [float(lo[0] - MARGIN), float(lo[1] - MARGIN), float(hi[0] + MARGIN), float(hi[1] + MARGIN)],
                "line": c.tolist()}
    sign = -1.0 if out < 0 else 1.0  # the moulding stands towards sign * u
    tol = max(2.0 * resolution, 0.003)
    warning = ""

    def stretch_at(face_u):
        """The run of points standing out of a face at ``face_u`` nearest the path's point (indices), or None."""
        dep = sign * (c[:, 0] - face_u)
        stands = dep > tol
        found, start = [], None
        for i, flag in enumerate(stands):
            if flag and start is None:
                start = i
            if not flag and start is not None:
                found.append((start, i - 1))
                start = None
        if start is not None:
            found.append((start, len(c) - 1))
        if not found:
            return None
        return min(found, key=lambda r: np.hypot(*c[r[0]:r[1] + 1].T).min())

    # a vertical run with both ends of the line deeper than it is part of the moulding (a fillet, a fascia), not the
    # wall: the wall is where the line comes from or goes to
    ends = -sign * c[[0, -1], 0]
    faces = [f for f in _vertical_faces(c, tol) if not np.all(ends > -sign * f[0] + tol)]
    if faces:
        # the deepest face (farthest behind the moulding's front), preferring one seen both below and above the
        # moulding: a cornice's own fascia is a vertical face too, in front of the wall, and a parapet behind is seen
        # above only
        best = None
        for face_u, spans in sorted(faces, key=lambda f: -sign * f[0]):  # deepest first
            run = stretch_at(face_u)
            if run is None:
                continue
            v_lo, v_hi = c[run[0]:run[1] + 1, 1].min(), c[run[0]:run[1] + 1, 1].max()
            both = any(b <= v_lo + tol for a, b in spans) and any(a >= v_hi - tol for a, b in spans)
            if best is None:
                best = (face_u, run)
            if both:
                best = (face_u, run)
                break
        if best is None:
            raise ValueError("no part of the cut stands out of the wall face near the path")
        wall_u, (i0, i1) = best
    else:  # no wall face in the window: the deepest point of the line, as a face
        wall_u = float(c[:, 0].min() if sign > 0 else c[:, 0].max())
        warning = "no-wall-face"
        run = stretch_at(wall_u)
        if run is None:
            raise ValueError("no part of the cut stands out of its deepest point near the path")
        i0, i1 = run
    dep = sign * (c[:, 0] - wall_u)  # how far each point stands out of the face

    def on_face(i, j):  # the point of the segment i-j where it meets the face (dep = 0), j outside the run
        if j < 0 or j >= len(c):
            return np.array([wall_u, c[i, 1]])
        a, b = dep[i], dep[j]
        f = a / (a - b) if a != b else 0.0
        p = c[i] + f * (c[j] - c[i])
        return np.array([wall_u, p[1]])

    stretch = np.vstack([on_face(i0, i0 - 1), c[i0:i1 + 1], on_face(i1, i1 + 1)])
    if simplify > 0 and len(stretch) > 2:
        stretch = np.asarray(LineString(stretch).simplify(simplify, preserve_topology=False).coords)
    back = wall_u - sign * float(depth)
    if depth > 0:
        ring = np.vstack([stretch, [[back, stretch[-1, 1]], [back, stretch[0, 1]]]])
    else:
        ring = stretch
    ring = drop_repeated(ring, 1e-6)
    if polygon_area(ring) < 0:
        ring = ring[::-1]
    v_lo, v_hi = float(min(stretch[0, 1], stretch[-1, 1], stretch[:, 1].min())), float(stretch[:, 1].max())
    projection = float((sign * (stretch[:, 0] - wall_u)).max())
    u_out = wall_u + sign * projection
    u0, u1 = sorted((wall_u, u_out))
    front = polygon_area(np.vstack([stretch, stretch[:1]])[:-1]) if len(stretch) > 2 else 0.0
    return {"section": ring, "wall_u": wall_u, "v_range": [v_lo, v_hi], "projection": projection,
            "height": v_hi - v_lo, "window": [u0 - MARGIN, v_lo - MARGIN, u1 + MARGIN, v_hi + MARGIN],
            "line": c.tolist(), "front_area": abs(float(front)), "warning": warning, "closed": False}


def _vertical_faces(c, tol: float, shortest: float = 0.03) -> list[tuple[float, list[tuple[float, float]]]]:
    """The vertical faces of an open line of a cut: runs of consecutive near-vertical segments at least ``shortest``
    long (3 cm), grouped by their u (within ``tol``). Each face: ``(u, [(v from, v to), ...])``."""
    seg = np.diff(c, axis=0)
    vertical = np.abs(seg[:, 1]) > 3 * np.abs(seg[:, 0])
    runs, k = [], 0
    while k < len(seg):
        if not vertical[k]:
            k += 1
            continue
        j = k
        while j + 1 < len(seg) and vertical[j + 1] and abs(c[j + 2, 0] - c[k, 0]) <= tol:
            j += 1
        pts = c[k:j + 2]
        if np.ptp(pts[:, 1]) >= shortest - 1e-9:
            runs.append((float(np.median(pts[:, 0])), (float(pts[:, 1].min()), float(pts[:, 1].max()))))
        k = j + 1
    faces: list[tuple[float, list]] = []
    for u, span in sorted(runs):
        if faces and abs(u - faces[-1][0]) <= tol:
            faces[-1][1].append(span)
        else:
            faces.append((u, [span]))
    return faces


def drop_repeated(ring, tolerance: float = 1e-7) -> np.ndarray:
    """A closed outline without points repeated one after the other (or the last on the first): OpenCascade refuses
    an edge of zero length."""
    r = np.asarray(ring, np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(r, axis=0), axis=1) > tolerance]
    r = r[keep]
    while len(r) > 3 and np.linalg.norm(r[-1] - r[0]) <= tolerance:
        r = r[:-1]
    return r


def drop_repeated_path(path, tolerance: float = 1e-7) -> np.ndarray:
    """A path without a point repeated right after itself (a segment of zero length has no direction)."""
    p = np.asarray(path, np.float64)
    if len(p) < 2:
        return p
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > tolerance]
    return p[keep]


def polygon_area(ring) -> float:
    r = np.asarray(ring, float)
    x, y = r[:, 0], r[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def sweep_mesh(path, section):
    """The section ``(u, v)`` swept along ``path`` with mitred corners, as a closed triangle mesh: the solid to
    expect (its volume) and the tessellation for checks (the fake worker of the tests). The mitre is exact for turns
    about the vertical (a level path); the real solids are CadQuery's. Each vertex of the path carries the section in
    its mitre plane, stretched across the bisector by the mitre scale."""
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


def path_coordinates(points, path) -> np.ndarray:
    """``(station, u, v)`` of points in the frame of the path's nearest segment (u = up x tangent, v = tangent x u):
    where a point of the model lies with respect to a moulding's section."""
    p = np.asarray(points, np.float64)
    pth = np.asarray(path, np.float64)
    seg = np.diff(pth, axis=0)
    lengths = np.linalg.norm(seg, axis=1)
    cum = np.r_[0.0, np.cumsum(lengths)]
    best = np.full(len(p), np.inf)
    out = np.zeros((len(p), 3))
    for i in range(len(seg)):
        o, u, v, t = section_frame(seg[i], pth[i])
        q = p - o
        s = np.clip(q @ t, 0.0, lengths[i])
        d = np.linalg.norm(q - s[:, None] * t, axis=1)
        m = d < best
        best[m] = d[m]
        out[m] = np.column_stack([cum[i] + s, q @ u, q @ v])[m]
    return out


def behind_back(points, path, back: float, out: float, margin: float) -> np.ndarray:
    """Which points lie behind the back line of a moulding's section (inside the wall: ``u`` beyond ``back`` on the
    side away from where the moulding projects, ``out``), or within ``margin`` of it."""
    u = path_coordinates(points, path)[:, 1]
    return u > back - margin if out < 0 else u < back + margin


def surface_depth(uvd, faces, uv) -> np.ndarray:
    """The depth ``d`` of a mesh seen along a frame's normal at frame points ``uv``: ``uvd`` are the mesh's vertices
    in the frame (u, v, d towards the viewer); for each point the triangles whose (u, v) contain it, the largest d (the
    surface nearest the viewer); NaN where no triangle is."""
    q = np.asarray(uvd, np.float64)
    f = np.asarray(faces, np.int64)
    a, b, c = q[f[:, 0]], q[f[:, 1]], q[f[:, 2]]
    det = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1])
    ok = np.abs(det) > 1e-18
    lo = np.minimum(np.minimum(a[:, :2], b[:, :2]), c[:, :2])
    hi = np.maximum(np.maximum(a[:, :2], b[:, :2]), c[:, :2])
    out = np.full(len(uv), np.nan)
    for i, (x, y) in enumerate(np.asarray(uv, np.float64)[:, :2]):
        m = ok & (lo[:, 0] <= x) & (hi[:, 0] >= x) & (lo[:, 1] <= y) & (hi[:, 1] >= y)
        if not m.any():
            continue
        aa, bb, cc, dd = a[m], b[m], c[m], det[m]
        w1 = ((x - aa[:, 0]) * (cc[:, 1] - aa[:, 1]) - (cc[:, 0] - aa[:, 0]) * (y - aa[:, 1])) / dd
        w2 = ((bb[:, 0] - aa[:, 0]) * (y - aa[:, 1]) - (x - aa[:, 0]) * (bb[:, 1] - aa[:, 1])) / dd
        inside = (w1 >= -1e-12) & (w2 >= -1e-12) & (w1 + w2 <= 1 + 1e-12)
        if inside.any():
            d = aa[:, 2] + w1 * (bb[:, 2] - aa[:, 2]) + w2 * (cc[:, 2] - aa[:, 2])
            out[i] = float(d[inside].max())
    return out


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
    jambs) and the second one (where the ray leaves the ring: the extrados of a ring or an archivolt that stands
    out of the cut). The rays go all round but
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
        order = np.argsort(np.hypot(*(pts - pick).T))
        near.append(pts[order[0]])
        if len(pts) > 1:
            far.append(pts[order[1]])  # where the ray leaves the ring: not what lies beyond it (a cornice above)
    return np.asarray(near).reshape(-1, 2), np.asarray(far).reshape(-1, 2)


def guess_pick(polylines, grid: int = 40) -> tuple[float, float]:
    """A point inside an opening of a section, for the dialog to start from: among the points of a grid that see a
    line above, to the left and to the right of them, the one farthest from every line (the middle of the largest
    opening); a point with an odd number of crossings upwards may be inside material and counts half."""
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
            if up.is_empty:
                continue
            if lines.intersection(LineString([(x, y), (x - reach, y)])).is_empty or \
                    lines.intersection(LineString([(x, y), (x + reach, y)])).is_empty:
                continue
            d = lines.distance(Point(x, y))
            if len(getattr(up, "geoms", [up])) % 2:  # maybe inside material (open curves make the parity unsure)
                d *= 0.5
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


def _fit_circle_family(P, scale, mode: str | None = None):
    """A circle with the jambs below its springing line. ``mode``: None (the centre at or below the springing line,
    the data decide), "semicircular" (the centre on the springing line), "segmental" (the centre below it by at least
    2 % of the radius)."""
    from scipy.optimize import least_squares

    u, v = P[:, 0], P[:, 1]
    top = v > np.percentile(v, 40)
    c = _circle_fit(P[top], trim=False) or _circle_fit(P, trim=False)
    if c is None:
        return None
    cx, cy, r, _ = c
    k_lo, k_hi = {"semicircular": (0.0, 1e-9), "segmental": (0.02, 0.999)}.get(mode, (0.0, 1.0))

    def model(x):
        uc, vc, R, k = x
        s = k * R  # the springing line k radii above the centre
        vs = vc + s
        half = np.sqrt(max(R * R - s * s, 0.0))
        arc = v >= vs
        return np.where(arc, np.hypot(u - uc, v - vc) - R, _jamb_res(u, uc - half, uc + half))

    best = None
    for k0 in (0.0, 0.2, 0.5, 0.8):
        k0 = float(np.clip(k0, k_lo, k_hi))
        x0 = np.array([cx, cy, r, k0])
        try:
            sol = least_squares(model, x0, bounds=([-np.inf, -np.inf, 1e-4, k_lo], [np.inf, np.inf, np.inf, k_hi]),
                                loss="soft_l1", f_scale=scale)
        except ValueError:
            continue
        rms = float(np.sqrt(np.mean(model(sol.x) ** 2)))
        if best is None or rms < best[1]:
            best = (sol.x, rms)
    if best is None:
        return None
    (uc, vc, R, k), rms = best
    return {"uc": float(uc), "vc": float(vc), "R": float(R), "vs": float(vc + k * R)}, rms


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
    c = _circle_fit(P[v > np.percentile(v, 40)], trim=False) or _circle_fit(P, trim=False)
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


def _arch_from(family: str, params: dict, rms: float, n: int, kind: str | None = None) -> ArchFit:
    if family == "circle":
        uc, vc, R, vs = params["uc"], params["vc"], params["R"], params["vs"]
        s = vs - vc
        half = float(np.sqrt(max(R * R - s * s, 0.0)))
        if kind not in ("semicircular", "segmental"):
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
        if fam == "circle":
            f = _fit_circle_family(P, noise, kind if kind in ("semicircular", "segmental") else None)
        else:
            f = {"pointed": _fit_pointed, "ellipse": _fit_ellipse}[fam](P, noise)
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
    forced = kind if kind in ("semicircular", "segmental") else None  # the type asked for, enforced by the fit
    fit = _arch_from(chosen, params, rms, len(P), forced)
    # no jamb under the springing line: the springing line is the chord through the lowest points of the arc
    if not (P[:, 1] < fit.springing - 3 * max(rms, 1e-4)).any():
        left = P[P[:, 0] < fit.centre_u]
        right = P[P[:, 0] >= fit.centre_u]
        if len(left) and len(right):
            vs = 0.5 * (left[:, 1].min() + right[:, 1].min())
            if forced != "semicircular":
                params = dict(params, vs=float(max(vs, params.get("vc", -np.inf))))
                fit = _arch_from(chosen, params, rms, len(P), forced)
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


def arch_thickness(fit: ArchFit, extrados) -> tuple[float | None, str]:
    """The thickness of the ring from the extrados points (where the rays leave the ring, above the springing
    line): their median distance to the intrados, ``(thickness, "")`` when they agree (interquartile range under 20 %
    of the median, the median under half the span), else ``(None, reason)``: a ring flush with the wall around it
    gives the wall's sides and top, not an extrados, and the thickness must then be given."""
    from scipy.spatial import cKDTree

    e = np.asarray(extrados, float).reshape(-1, 2)
    e = e[e[:, 1] > fit.springing + 0.02 * max(fit.rise, 1e-3)]
    if len(e) < 5:
        return None, "no extrados is seen from the point"
    curve = arch_curve(fit, 0.0, n=2000)
    d, _ = cKDTree(curve).query(e)
    d = d[d > 1e-4]
    if len(d) < 5:
        return None, "no extrados is seen from the point"
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    if q3 - q1 > 0.2 * med:
        return None, f"the far side is not a ring: its distance from the intrados varies from {q1:.3f} to {q3:.3f} m"
    if med > 0.5 * fit.span:
        return None, f"the far side is {med:.3f} m from the intrados, more than half the span: a wall, not a ring"
    return float(med), ""


def near_ring(points_uv, fit: ArchFit, thickness: float, tolerance: float) -> np.ndarray:
    """Which (u, v) points lie within ``tolerance`` of the intrados or of the extrados curve (the ring's curved faces,
    where the solid says something about the model)."""
    from scipy.spatial import cKDTree

    q = np.asarray(points_uv, float).reshape(-1, 2)
    curves = np.vstack([arch_curve(fit, 0.0, n=4000), arch_curve(fit, float(thickness), n=4000)])
    d, _ = cKDTree(curves).query(q)
    return d <= tolerance


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


# ====================================================================== opening (door, window)
FACE_SEARCH = 4.0  # how far from the point the wall faces around an opening are looked for (metres): any width up to ~6 m
MAX_THICKNESS = 3.0  # the farthest a second face may be from the first and still be the same wall
SPLAY_MAX = 35.0  # degrees: a reveal turning more than this from the face's normal is not looked for


def _clear_of_vertices(values, at: float, gap: float = 1e-7) -> float:
    """``at`` moved by a micrometre or two when vertices lie on it (a plane through vertices cuts the mesh into
    pieces of zero length that do not join)."""
    v = np.asarray(values, float)
    at = float(at)
    for _ in range(8):
        if not np.any(np.abs(v - at) < gap):
            break
        at += 1.3e-6
    return at


def plan_cut(vertices, faces, height: float) -> list[np.ndarray]:
    """The polylines of the horizontal cut of a mesh at ``height``, in plan (x, y)."""
    h = _clear_of_vertices(np.asarray(vertices, float)[:, 2], height)
    return mesh_section(vertices, faces, (0.0, 0.0, h), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0))


def fit_line(points, trim: float = 3.0, iterations: int = 3):
    """A plan line through points by orthogonal least squares, trimmed of the points farther than ``trim`` robust
    deviations: ``(point, unit direction, rms, kept mask)``."""
    p = np.asarray(points, float)[:, :2]
    keep = np.ones(len(p), bool)
    c, d = p.mean(axis=0), np.array([1.0, 0.0])
    for _ in range(iterations):
        q = p[keep]
        if len(q) < 2:
            break
        c = q.mean(axis=0)
        _w, vec = np.linalg.eigh(np.cov((q - c).T))
        d = vec[:, -1]
        r = np.abs((p - c) @ np.array([-d[1], d[0]]))
        mad = 1.4826 * np.median(r[keep]) + 1e-5
        keep = r <= max(trim * mad, 1e-4)
    r = (p[keep] - c) @ np.array([-d[1], d[0]])
    return c, d / np.linalg.norm(d), float(np.sqrt(np.mean(r ** 2))) if len(r) else 0.0, keep


def intersect_lines(p1, d1, p2, d2) -> np.ndarray:
    """The point where two plan lines meet."""
    a = np.column_stack([d1, -np.asarray(d2, float)])
    t = np.linalg.solve(a, np.asarray(p2, float) - np.asarray(p1, float))
    return np.asarray(p1, float) + t[0] * np.asarray(d1, float)


def _tolerances(resolution: float) -> tuple[float, float]:
    """``(on a line, a gap)``: how far a point of the cut may be from a face's line and still be on it; how long a
    stretch with no point must be to be an opening and not a hole of the mesh."""
    return max(3.0 * resolution, 0.015), max(4.0 * resolution, 0.05)


def face_lines(polylines, centre, resolution: float = 0.005, search: float = FACE_SEARCH) -> list[dict]:
    """The wall faces around a point on a horizontal cut: straight runs of the cut (lines through its points within a
    tolerance of the mesh's resolution) on which the point's projection falls in a **gap** with the run present on
    both sides of it - the near and far faces of the wall the opening goes through. A wall of a room behind the
    opening, or a facing building, is either across the point's projection (no gap) or ends before it. Each:
    ``{"point", "dir", "gap": (t0, t1), "offset", "points"}``, ``t`` along ``dir`` from ``point``, ``offset`` the
    signed distance of the line from the point along the line's left normal."""
    from shapely.geometry import LineString

    c = np.asarray(centre, float)[:2]
    on_tol, gap_min = _tolerances(resolution)
    parts = [np.asarray(p, float)[:, :2] for p in polylines if len(p) > 1]
    if not parts:
        return []
    allpts = np.vstack(parts)
    near = np.linalg.norm(allpts - c, axis=1) <= search
    q = allpts[near]
    if len(q) < 4:
        return []
    seeds = []
    for p in parts:
        simple = np.asarray(LineString(p).simplify(max(2.0 * resolution, 0.01)).coords)
        for a, b in zip(simple[:-1], simple[1:]):
            if np.linalg.norm(b - a) >= max(0.1, 5 * on_tol) and \
                    min(np.linalg.norm(a - c), np.linalg.norm(b - c)) <= search:
                seeds.append((a, b))
    seeds.sort(key=lambda s: -np.linalg.norm(s[1] - s[0]))
    found: list[dict] = []
    for a, b in seeds[:400]:
        d = (b - a) / np.linalg.norm(b - a)
        n = np.array([-d[1], d[0]])
        if any(abs(f["dir"] @ n) < np.sin(np.radians(3))
               and abs((a - f["point"]) @ np.array([-f["dir"][1], f["dir"][0]])) < on_tol for f in found):
            continue
        on = np.abs((q - a) @ n) < on_tol
        if on.sum() < 8:
            continue
        p0, d0, _r, keep = fit_line(q[on])
        n0 = np.array([-d0[1], d0[0]])
        on = np.abs((q - p0) @ n0) < on_tol
        t = np.sort((q[on] - p0) @ d0)
        tc = float((c - p0) @ d0)
        below, above = t[t < tc], t[t > tc]
        if len(below) < 4 or len(above) < 4:
            continue
        t0, t1 = float(below.max()), float(above.min())
        # a face goes on from each end of the gap without a break: a run, not points of things crossing the line
        if t1 - t0 < gap_min or _run_length(below[::-1], gap_min) < RUN_MIN or _run_length(above, gap_min) < RUN_MIN:
            continue
        # the line itself, not a crossing of other things: its points are spread along it on both sides
        found.append({"point": p0, "dir": d0, "gap": (t0, t1), "offset": float((p0 - c) @ n0),
                      "points": q[on & (np.abs(((q - p0) @ d0) - 0.5 * (t0 + t1)) < 0.5 * (t1 - t0) + 0.6)]})
    return found


RUN_MIN = 0.15  # metres of face, without a break, on each side of an opening


def _run_length(t, gap: float) -> float:
    """How far the sorted positions ``t`` (from the gap outwards) go on without a break longer than ``gap``."""
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0
    breaks = np.flatnonzero(np.abs(np.diff(t)) >= gap)
    end = breaks[0] if len(breaks) else len(t) - 1
    return float(abs(t[end] - t[0]))


def _pick_faces(lines: list[dict], centre, resolution: float):
    """The near face (the gap-line nearest the point) and the far face (the nearest line parallel to it, more than a
    tolerance and less than :data:`MAX_THICKNESS` away, on the point's side of the near face if there is one, else on
    the other): ``(near, far or None)``."""
    if not lines:
        return None, None
    on_tol, _gap = _tolerances(resolution)
    c = np.asarray(centre, float)[:2]
    near = min(lines, key=lambda f: abs(f["offset"]))
    n = np.array([-near["dir"][1], near["dir"][0]])
    side_c = np.sign((c - near["point"]) @ n) or 1.0
    best = {1.0: None, -1.0: None}
    for f in lines:
        if f is near or abs(f["dir"] @ near["dir"]) < np.cos(np.radians(10)):
            continue
        sep = float((f["point"] - near["point"]) @ n)
        if not (2 * on_tol < abs(sep) < MAX_THICKNESS):
            continue
        # another face, not the near one again at a slant: its points are all off the near face's line
        if np.percentile(np.abs((f["points"] - near["point"]) @ n), 10) < 1.5 * on_tol:
            continue
        s = np.sign(sep) * side_c
        if best[s] is None or abs(sep) < abs(best[s][1]):
            best[s] = (f, sep)
    far = best[1.0] or best[-1.0]
    return near, None if far is None else far[0]


def left_of(axis) -> np.ndarray:
    """The left of an opening seen from outside (looking along ``-axis``, the axis out of the wall)."""
    a = np.asarray(axis, float)
    return np.array([a[1], -a[0]])


@dataclass
class OpeningPlan:
    """The plan of an opening at the height of a cut (plan metres): ``axis`` the outer face's normal, out of the wall;
    the left and the right as seen from outside (:func:`left_of`); the two reveals and the two wall faces as lines
    ``(point, direction)``; ``polygon`` the void (counter-clockwise); ``thickness`` between the faces, ``measured``
    whether both faces were on the model; the splays in degrees against the outer face's normal (positive: the
    opening widens towards the inner face, an embrasure). ``near``: whether the outer face is the face nearer the
    point."""

    centre: np.ndarray
    axis: np.ndarray
    left: tuple
    right: tuple
    outer: tuple
    inner: tuple
    polygon: np.ndarray
    thickness: float
    measured: bool
    width_outer: float
    width_inner: float
    splay_left: float
    splay_right: float
    reveal_rms: float
    points: dict = field(default_factory=dict)
    near: bool = True

    def to_dict(self) -> dict:
        def line(x):
            return [np.asarray(x[0], float).tolist(), np.asarray(x[1], float).tolist()]

        return {"centre": self.centre.tolist(), "axis": self.axis.tolist(), "left": line(self.left),
                "right": line(self.right), "outer": line(self.outer), "inner": line(self.inner),
                "polygon": self.polygon.tolist(), "thickness": self.thickness, "measured": self.measured,
                "width_outer": self.width_outer, "width_inner": self.width_inner, "splay_left": self.splay_left,
                "splay_right": self.splay_right, "reveal_rms": self.reveal_rms, "near": self.near,
                "points": {k: np.asarray(v).tolist() for k, v in self.points.items()}}

    def flipped(self) -> "OpeningPlan":
        """The same plan with the other face as the outer one."""
        pts = dict(self.points)
        pts["left"], pts["right"] = self.points.get("right", []), self.points.get("left", [])
        return opening_plan_from_lines(self.centre, -self.axis, self.right, self.left, self.inner, self.outer,
                                       self.measured, self.reveal_rms, pts, near=not self.near)


def opening_plan_from_lines(centre, axis, left, right, outer, inner, measured: bool, rms: float = 0.0,
                            points=None, near: bool = True) -> OpeningPlan:
    """The plan of an opening from its four lines (reveals and faces): the void's corners, widths, thickness and the
    splays against the outer face's normal (``axis`` gives only its sense: out of the wall)."""
    lines = {}
    for name, (p, d) in (("left", left), ("right", right), ("outer", outer), ("inner", inner)):
        d = np.asarray(d, float)
        lines[name] = (np.asarray(p, float), d / np.linalg.norm(d))
    do = lines["outer"][1]
    a = np.array([-do[1], do[0]])
    if a @ np.asarray(axis, float) < 0:
        a = -a
    w = left_of(a)
    corners = np.array([intersect_lines(*lines[s], *lines[f]) for s, f in
                        (("left", "outer"), ("right", "outer"), ("right", "inner"), ("left", "inner"))])
    poly = corners if polygon_area(corners) > 0 else corners[::-1]
    width_outer = float(np.linalg.norm(corners[1] - corners[0]))
    width_inner = float(np.linalg.norm(corners[2] - corners[3]))
    mid_out = corners[:2].mean(axis=0)
    thickness = float(abs((mid_out - lines["inner"][0]) @ np.array([-lines["inner"][1][1], lines["inner"][1][0]])))

    def splay(c_out, c_in, side):
        r = c_in - c_out
        return float(np.degrees(np.arctan2(r @ side, r @ (-a))))

    return OpeningPlan(np.asarray(centre, float)[:2], a, lines["left"], lines["right"], lines["outer"], lines["inner"],
                       poly, thickness, measured, width_outer, width_inner, splay(corners[0], corners[3], w),
                       splay(corners[1], corners[2], -w), float(rms), dict(points or {}), near)


def opening_plan(polylines, centre, thickness: float | None = None, resolution: float = 0.005,
                 search: float = FACE_SEARCH) -> OpeningPlan:
    """The plan of the void of a door or a window from a horizontal cut (plan polylines) and a point inside it. First
    the wall faces (:func:`face_lines`): the near face, whose gap around the point is the opening, and the far face
    parallel to it (or, when only one face is on the model, the given ``thickness`` behind the near one, towards the
    point). The reveals are only the cut's points between the two faces that start at the gap's ends: each fitted by a
    line (a splayed reveal stays splayed). Nothing beyond the far face (a room, a street) and nothing away from the
    gap is a reveal. The outer face here is the near one (the caller decides which is outer:
    :meth:`OpeningPlan.flipped`).
    Raises ValueError with the reason."""
    c = np.asarray(centre, float)[:2]
    on_tol, _gap = _tolerances(resolution)
    near, far = _pick_faces(face_lines(polylines, c, resolution, search), c, resolution)
    if near is None:
        raise ValueError("no gap in a wall face around the point: it is not inside an opening")
    p0, d0 = near["point"], near["dir"]
    n0 = np.array([-d0[1], d0[0]])
    into = -1.0 if (c - p0) @ n0 < 0 else 1.0  # from the near face into the wall
    if far is None and (c - p0) @ n0 == 0:
        into = 1.0
    if far is not None:
        sep = float((far["point"] - p0) @ n0)
        into = np.sign(sep)
        T = abs(sep)
        far_line = (far["point"], far["dir"] if far["dir"] @ d0 > 0 else -far["dir"])
    elif thickness:
        T = float(thickness)
        far_line = (p0 + into * T * n0, d0)
    else:
        q = np.vstack([np.asarray(p, float)[:, :2] for p in polylines if len(p)])
        off = (q - p0) @ n0 * into
        t = (q - p0) @ d0
        g0, g1 = near["gap"]
        clear = 4.0 * on_tol  # the reveals' points near the gap's ends are not a second face
        beside = ((t < g0 - clear) & (t > g0 - 0.6)) | ((t > g1 + clear) & (t < g1 + 0.6))
        if (beside & (off > 0.4 * on_tol) & (off < 2.5 * on_tol)).sum() >= 8:
            raise ValueError(f"the faces of a thin wall cannot be told apart at this resolution (edges of "
                             f"{resolution * 1000:.0f} mm)")
        raise ValueError("only one face of the wall is on the model: the thickness of the wall must be given")
    inward = into * n0
    g0, g1 = near["gap"]
    if far is not None:
        h = sorted(((far["point"] + far["dir"] * t) - p0) @ d0 for t in far["gap"])
    else:
        h = [g0, g1]
    allpts = np.vstack([np.asarray(p, float)[:, :2] for p in polylines if len(p)])
    t_all = (allpts - p0) @ d0
    u_all = (allpts - p0) @ inward  # 0 on the near face, T on the far one
    edge = min(on_tol, T / 4)
    band = (u_all > edge) & (u_all < T - edge)
    spread = T * np.tan(np.radians(SPLAY_MAX))
    margin = max(0.03, 3 * on_tol)
    mid = 0.25 * (g0 + g1 + h[0] + h[1])
    reveals, rms = {}, []
    for name, a_end, b_end in (("first", g0, h[0]), ("second", g1, h[1])):
        lo = min(a_end, b_end) - margin - (spread if far is None else 0.0)
        hi = max(a_end, b_end) + margin + (spread if far is None else 0.0)
        m = band & (t_all >= lo) & (t_all <= hi) & ((t_all < mid) if name == "first" else (t_all > mid))
        pts = allpts[m]
        if len(pts) < 4:
            raise ValueError("no reveal at one end of the gap (an opening at the end of a wall, or a reveal not on "
                             "the model)")
        p, d, r, keep = fit_line(pts)
        if abs(d @ inward) < np.cos(np.radians(SPLAY_MAX + 10)):
            raise ValueError("a reveal is not across the wall")
        if d @ inward < 0:
            d = -d
        reveals[name] = (p, d, pts[keep])
        rms.append(r)
    a = -inward  # out of the wall through the near face: the near face is the outer one here
    w = left_of(a)
    first_left = (reveals["first"][0] - c) @ w > (reveals["second"][0] - c) @ w
    left, right = (reveals["first"], reveals["second"]) if first_left else (reveals["second"], reveals["first"])
    points = {"left": left[2], "right": right[2], "face_near": near["points"]}
    if far is not None:
        points["face_far"] = far["points"]
    return opening_plan_from_lines(c, a, left[:2], right[:2], (p0, d0), far_line, far is not None,
                                   float(np.sqrt(np.mean(np.square(rms)))), points, near=True)


def with_thickness(plan: OpeningPlan, thickness: float) -> OpeningPlan:
    """The plan with the inner face ``thickness`` behind the outer one (the outer face and the reveals kept)."""
    p, d = plan.outer
    n = np.array([-d[1], d[0]])
    n = n if n @ plan.axis < 0 else -n
    return opening_plan_from_lines(plan.centre, plan.axis, plan.left, plan.right, plan.outer,
                                   (np.asarray(p) + float(thickness) * n, d), False, plan.reveal_rms, plan.points,
                                   near=plan.near)


def vertical_hits(vertices, faces, centre, axis, height: float, u_range, samples: int = 15):
    """Sill and lintel from the vertical section through ``centre`` across the wall (plane spanned by ``axis`` and
    up): vertical rays from the cut height at ``samples`` places across the middle 60 % of ``u_range`` (the faces'
    positions along the axis from the centre), the first line met upwards and downwards. Returns ``(down, up,
    polylines, us)``: the heights met (world-local z, NaN where nothing), the section's polylines in (u, z) and the
    places of the rays (u)."""
    c = np.asarray(centre, float)[:2]
    a = np.r_[np.asarray(axis, float)[:2], 0.0]
    across = np.array([-a[1], a[0]])
    v = np.asarray(vertices, float)
    off = _clear_of_vertices((v[:, :2] - c) @ across, 0.0)
    c = c + off * across
    lines = mesh_section(v, faces, (c[0], c[1], float(height)), a, UP)
    lo, hi = sorted(map(float, u_range))
    us = np.linspace(lo + 0.2 * (hi - lo), hi - 0.2 * (hi - lo), samples)
    down = np.full(samples, np.nan)
    up = np.full(samples, np.nan)
    segs = [np.stack([p[:-1], p[1:]], axis=1) for p in lines if len(p) > 1]
    if segs:
        s = np.concatenate(segs)
        u0, v0, u1, v1 = s[:, 0, 0], s[:, 0, 1], s[:, 1, 0], s[:, 1, 1]
        for k, u in enumerate(us):
            m = (np.minimum(u0, u1) <= u) & (np.maximum(u0, u1) >= u) & (np.abs(u1 - u0) > 1e-12)
            if not m.any():
                continue
            f = (u - u0[m]) / (u1[m] - u0[m])
            vv = v0[m] + f * (v1[m] - v0[m])
            if (vv > 1e-9).any():
                up[k] = vv[vv > 1e-9].min()
            if (vv < -1e-9).any():
                down[k] = vv[vv < -1e-9].max()
    return down + float(height), up + float(height), [np.column_stack([p[:, 0], p[:, 1] + float(height)])
                                                       for p in lines], us


def robust_level(values) -> tuple[float | None, float]:
    """The height of a run of hits: their median (None when fewer than three) and their spread (interquartile)."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return None, float("nan")
    q1, med, q3 = np.percentile(v, [25, 50, 75])
    return float(med), float(q3 - q1)


def level_slope(us, values) -> float | None:
    """The slope of a run of hits across the wall (rise over run, a line by least squares), None with fewer than
    three."""
    u = np.asarray(us, float)
    v = np.asarray(values, float)
    ok = np.isfinite(v)
    if ok.sum() < 3 or np.ptp(u[ok]) < 1e-6:
        return None
    return float(np.polyfit(u[ok], v[ok], 1)[0])


def _along_axis(plan: OpeningPlan, line) -> float:
    """Where a face line crosses the axis through the plan's centre: its distance along the axis."""
    q = intersect_lines(plan.centre, plan.axis, *line)
    return float((q - plan.centre) @ plan.axis)


def opening_heights(vertices, faces, centre, zmin: float, zmax: float, step: float = 0.05,
                    thickness: float | None = None, resolution: float = 0.005) -> tuple[float, float] | None:
    """The heights between ``zmin`` and ``zmax`` at which the plan of an opening around ``centre`` can be read
    (:func:`opening_plan`): the longest run of them, ``(lowest, highest)``, or None."""
    zs = np.arange(zmin + step / 2, zmax, step)
    ok = []
    for z in zs:
        try:
            opening_plan(plan_cut(vertices, faces, z), centre, thickness, resolution)
            ok.append(True)
        except ValueError as exc:  # an opening in a wall of which one face is on the model is an opening still
            ok.append("thickness of the wall must be given" in str(exc) or "thin wall" in str(exc))
        except np.linalg.LinAlgError:
            ok.append(False)
    best, k = None, 0
    while k < len(ok):
        if ok[k]:
            j = k
            while j + 1 < len(ok) and ok[j + 1]:
                j += 1
            if best is None or j - k > best[1] - best[0]:
                best = (k, j)
            k = j + 1
        else:
            k += 1
    return None if best is None else (float(zs[best[0]]), float(zs[best[1]]))


def measure_opening(vertices, faces, centre, height: float | None = None, thickness: float | None = None,
                    top_limit: float | None = None, sill: float | None = None, lintel: float | None = None,
                    resolution: float | None = None, search: float = FACE_SEARCH, outward=None,
                    apply_thickness: bool = True) -> dict:
    """Everything the model says of a door or a window around a plan point ``centre`` inside it: the height of the
    cut (``height``, else the middle of the heights at which its plan can be read, below ``top_limit`` when an arch
    closes it), the plan of the void (:func:`opening_plan`; the outer face there is the near one), the sill and the
    lintel from the vertical section across the wall through the point (the first lines met downwards and upwards;
    ``sill`` and ``lintel`` given take their place) and the sill's slope across the wall, the widths near the sill and
    near the lintel (cuts at a tenth of the height from them, at both faces: ``widths``). ``outward``: a plan direction
    out of the wall that makes the face on its side the outer one (else the near face is); ``apply_thickness``: a
    given ``thickness`` replaces a measured one (the inner face moved). Coordinates as the mesh's. Raises
    ValueError."""
    v = np.asarray(vertices, np.float64)
    c = np.asarray(centre, float)[:2]
    r = search + 0.5
    sv, sf = select_faces(v, faces, box=(c[0] - r, c[1] - r, -np.inf, c[0] + r, c[1] + r, np.inf))
    if len(sf) < 10:
        raise ValueError("the model is not around the point")
    res = mesh_resolution(sv, sf) if resolution is None else float(resolution)
    found = None
    if height is None:
        z0, z1 = float(sv[:, 2].min()), float(sv[:, 2].max())
        if top_limit is not None:
            z1 = min(z1, float(top_limit))
        found = opening_heights(sv, sf, c, z0, z1, max(0.02, min(0.05, (z1 - z0) / 60)), thickness, res)
        if found is None:
            raise ValueError("the point is not inside an opening at any height (no gap in a wall face around it)")
        height = 0.5 * (found[0] + found[1])
        plan = opening_plan(plan_cut(sv, sf, height), c, thickness, res, search)
    else:
        try:
            plan = opening_plan(plan_cut(sv, sf, height), c, thickness, res, search)
        except ValueError as exc:
            raise ValueError(f"at the height given: {exc}") from exc
    if outward is not None and plan.axis @ np.asarray(outward, float)[:2] < 0:
        plan = plan.flipped()
    if thickness and apply_thickness and plan.measured:
        plan = with_thickness(plan, thickness)
    u_range = (_along_axis(plan, plan.outer), _along_axis(plan, plan.inner))
    down, up, section, us = vertical_hits(sv, sf, c, plan.axis, height, u_range)
    sill_m, sill_spread = robust_level(down)
    lintel_m, lintel_spread = robust_level(up)
    slope = level_slope(us, down)
    bottom = float(sill) if sill is not None else sill_m
    top = float(lintel) if lintel is not None else lintel_m
    if bottom is None:
        raise ValueError("nothing is met under the point inside the opening (the sill or the floor): give the sill")
    if top is None and top_limit is None:
        raise ValueError("nothing is met over the point inside the opening (the lintel): give the lintel")
    widths = {}
    upper = float(top_limit) if top_limit is not None else top
    for name, z in (("sill", bottom + 0.1 * (upper - bottom)), ("lintel", upper - 0.1 * (upper - bottom))):
        try:
            other = opening_plan(plan_cut(sv, sf, z), c, plan.thickness if not plan.measured else None, res, search)
            widths[name] = {}
            for face, line in (("outer", plan.outer), ("inner", plan.inner)):
                a = intersect_lines(*other.left, *line)
                b = intersect_lines(*other.right, *line)
                widths[name][face] = float(np.linalg.norm(b - a))
        except (ValueError, np.linalg.LinAlgError):
            widths[name] = {"outer": None, "inner": None}
    return {"plan": plan, "height": float(height), "heights_found": found, "sill": bottom, "lintel": top,
            "sill_measured": sill_m, "lintel_measured": lintel_m, "sill_spread": sill_spread,
            "lintel_spread": lintel_spread, "sill_slope": slope, "width_at_sill": widths["sill"]["outer"],
            "width_at_lintel": widths["lintel"]["outer"], "widths": widths, "section": section, "resolution": res,
            "u_range": u_range,
            "cut": [p for p in plan_cut(sv, sf, height) if len(p) > 1]}


def arch_void_outline(fit: ArchFit, below: float = 0.0) -> dict:
    """The void under an arch's intrados, closed by its springing line (``below`` > 0: by a line that much under it,
    the void reaching down into the opening's part below the springing line, so that the two make one solid): points
    and segments as :func:`arch_outline` (lines, three-point arcs), counter-clockwise in the arch's plane (u, v)."""
    p, vs = fit.params, fit.springing
    if fit.family == "circle":
        uc, vc, R = p["uc"], p["vc"], p["R"]
        half = np.sqrt(max(R * R - (vs - vc) ** 2, 0.0))
        pts = [[uc + half, vs], [uc, vc + R], [uc - half, vs]]
        segs = [["A", 0, 1, 2]]
    elif fit.family == "pointed":
        um, s, R = p["um"], p["s"], p["R"]
        cL, cR = um - s + R, um + s - R
        apex = vs + np.sqrt(max(R * R - (R - s) ** 2, 0.0))
        top = np.arctan2(apex - vs, um - cL)
        a_r, a_l = 0.5 * (np.pi - top), 0.5 * (top + np.pi)
        pts = [[um + s, vs], [cR + R * np.cos(a_r), vs + R * np.sin(a_r)], [um, apex],
               [cL + R * np.cos(a_l), vs + R * np.sin(a_l)], [um - s, vs]]
        segs = [["A", 0, 1, 2], ["A", 2, 3, 4]]
    else:
        curve = arch_curve(fit, 0.0)
        curve = curve[curve[:, 1] >= vs - 1e-9]
        if curve[0, 0] < curve[-1, 0]:
            curve = curve[::-1]  # right to left over the top: counter-clockwise with the springing line under it
        pts = drop_repeated(np.vstack([[curve[0, 0], vs], curve, [curve[-1, 0], vs]]), 1e-7).tolist()
        segs = [["L", i, i + 1] for i in range(len(pts) - 1)]
    last = len(pts) - 1
    if below > 0:
        pts += [[pts[last][0], vs - below], [pts[0][0], vs - below]]
        segs += [["L", last, last + 1], ["L", last + 1, last + 2], ["L", last + 2, 0]]
    else:
        segs.append(["L", last, 0])
    return {"points": [list(map(float, q)) for q in pts], "segments": segs}


def opening_volume(plan, bottom: float, top: float, void_area: float = 0.0, void_depth: float = 0.0) -> float:
    """The volume of an opening's solid: the plan extruded from ``bottom`` to ``top`` and the arch's void above."""
    return abs(polygon_area(plan)) * (float(top) - float(bottom)) + float(void_area) * float(void_depth)


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
