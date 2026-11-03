"""A known hull for the tests: half-breadth b(x) = B (1 - (2x/L - 1)^2), depth D, sections
z(x, y) = D (y^2 / b(x)^2 - 1), z = 0 on the sheer. A mesh of 2 cm cells, in a chunk whose internal frame
is not the world (scale, rotation, an offset of UTM size)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import pyarchinit_photogrammetry as ppg
from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

L, B, D = 12.0, 2.0, 1.5
UTM = np.array([500123.25, 4500456.75, 12.0])


def half_breadth(x):
    return B * (1.0 - (2.0 * np.asarray(x, float) / L - 1.0) ** 2)


def depth(x, y):
    b = half_breadth(x)
    return D * (np.asarray(y, float) ** 2 / b**2 - 1.0)


def hull_vertices(cell: float = 0.02) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (ship: x from one end, y, z) and faces of the hull, x in [0.2, L - 0.2]."""
    xs = np.linspace(0.2, L - 0.2, int((L - 0.4) / cell) + 1)
    ts = np.linspace(-1.0, 1.0, int(2 * B / cell) + 1)
    x, u = np.meshgrid(xs, ts, indexing="ij")
    y = half_breadth(x) * u
    z = depth(x, y)
    n, m = x.shape
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    a = (i * m + j).ravel()
    b, c, d = a + 1, a + m, a + m + 1
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()]), faces


def transform(heading_deg: float = 0.0, scale: float = 2.0, offset=UTM) -> np.ndarray:
    """Internal -> world: the hull's x along ``heading_deg`` clockwise from north, its y to the left of
    x, ``scale`` world metres per internal unit."""
    h = np.radians(heading_deg)
    T = np.eye(4)
    T[:3, :3] = scale * np.array([[np.sin(h), -np.cos(h), 0.0], [np.cos(h), np.sin(h), 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = offset
    return T


def project(tmp_path: Path, heading_deg: float = 0.0, offset=UTM, crs: str = "EPSG:32633", scale: float = 2.0):
    """A project with the hull as the model of its chunk. Returns the document, the chunk and the two
    ends of the keel line in world coordinates."""
    return _project(tmp_path, *hull_vertices(), heading_deg, offset, crs, scale)


def _project(tmp_path: Path, vertices, faces, heading_deg: float, offset, crs: str, scale: float):
    doc = ppg.Document()
    ch = doc.add_chunk("Scafo")
    doc.save(tmp_path / "relitto.ppg")
    ch.crs = crs
    ch.transform = transform(heading_deg, scale, offset)
    write_mesh_ply(ch.data_dir / "model.ply", vertices / scale, faces)
    ch._set_product("model", ch.data_dir / "model.ply")
    doc.save()
    ends = ch.to_world(np.array([[0.0, 0.0, 0.0], [L / scale, 0.0, 0.0]]))
    return doc, ch, ends


# ------------------------------------------------------------------------------- a clinker hull
def _arc(y, a):
    """Arc length of z = D (y^2 / b^2 - 1) from the keel to ``y`` (``a`` = 2 D / b^2)."""
    y = np.asarray(y, float)
    return 0.5 * (y * np.sqrt(1.0 + (a * y) ** 2) + np.arcsinh(a * y) / a)


def _y_at(s, b):
    """The half-breadth ``y`` at arc ``s`` from the keel on the smooth section of half-breadth ``b``."""
    a = 2.0 * D / b**2
    table = np.linspace(0.0, b, 4001)
    y = np.interp(s, _arc(table, a), table)
    for _ in range(4):  # Newton on the analytic arc
        y = y - (_arc(y, a) - s) / np.sqrt(1.0 + (a * y) ** 2)
    return y


def girth(x) -> float:
    """The arc of the smooth half-section at ``x``, keel to sheer."""
    b = float(half_breadth(x))
    return float(_arc(b, 2.0 * D / b**2))


def _lands(x, strakes: int, stealer) -> tuple[np.ndarray, np.ndarray]:
    """Arc positions (centres) of every land column of the hull at ``x`` (the stealer's too) and
    whether each one steps at this ``x``."""
    widths = 0.14 + 0.03 * np.arange(strakes)  # 0.14, 0.17, 0.20, 0.23, 0.26 m of arc at midship
    cum = np.cumsum(widths)
    centres = list(cum[:-1])
    active = [True] * len(centres)
    if stealer is not None:
        centres.append(cum[-1])  # one strake more above the last land
        active.append(stealer[0] < x < stealer[1])
    return np.asarray(centres) * girth(x) / girth(L / 2.0), np.asarray(active, bool)


def _side(x, cell, strakes, land, lap, stealer):
    """The arc positions of one side's columns at ``x`` and the outward offset at each: flats between
    the laps, a vertex at each end of every lap (the step is a straight ramp over ``lap``)."""
    centres, active = _lands(x, strakes, stealer)
    g = girth(x)
    edges = [0.0]
    for c in centres:
        edges += [c - lap / 2.0, c + lap / 2.0]
    edges.append(g)
    mid_edges = [0.0]
    for c in _lands(L / 2.0, strakes, stealer)[0]:
        mid_edges += [c - lap / 2.0, c + lap / 2.0]
    mid_edges.append(girth(L / 2.0))
    s = [np.array([0.0])]
    for k in range(0, len(edges) - 1, 2):  # a flat from edges[k] to edges[k + 1], then the lap
        n = max(2, int(np.ceil((mid_edges[k + 1] - mid_edges[k]) / cell)))
        s.append(np.linspace(edges[k], edges[k + 1], n + 1)[1:])
        if k + 2 < len(edges):
            s.append(np.array([edges[k + 2]]))
    s = np.concatenate(s)
    offset = land * np.clip((s[:, None] - centres[None]) / lap + 0.5, 0.0, 1.0)[:, active].sum(axis=1)
    return s, offset


def _section_points(x, s, offset, sign):
    """The smooth section's point at arc ``s`` on the side ``sign`` (+1: y > 0), moved by ``offset``
    along the outward normal."""
    b = float(half_breadth(x))
    a = 2.0 * D / b**2
    y = _y_at(s, b)
    z = D * (y**2 / b**2 - 1.0)
    norm = np.sqrt(1.0 + (a * y) ** 2)
    return sign * (y + offset * a * y / norm), z - offset / norm


def clinker_hull(cell: float = 0.02, strakes: int = 5, land: float = 0.015, lap: float = 0.02,
                 stealer=None) -> tuple[np.ndarray, np.ndarray]:
    """The hull with ``strakes`` strakes per side, widths growing from the keel (0.14, 0.17, ... m of
    arc at midship, scaled by the local girth), each land a step of ``land`` outward along the
    section's normal over ``lap`` of arc (the lower edge of the strake above lies outside the one
    below); ``stealer=(x0, x1)``: one strake more above the last land between those x."""
    xs = np.linspace(0.2, L - 0.2, int((L - 0.4) / cell) + 1)
    rows = []
    for x in xs:
        s, off = _side(x, cell, strakes, land, lap, stealer)
        yp, zp = _section_points(x, s, off, 1.0)
        ys, zs = _section_points(x, s[1:], off[1:], -1.0)
        y = np.concatenate([ys[::-1], yp])
        z = np.concatenate([zs[::-1], zp])
        rows.append(np.column_stack([np.full(len(y), x), y, z]))
    m = len(rows[0])
    n = len(rows)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    a = (i * m + j).ravel()
    b, c, d = a + 1, a + m, a + m + 1
    faces = np.vstack([np.column_stack([a, b, c]), np.column_stack([b, d, c])])
    return np.vstack(rows), faces


def clinker_truth(x, strakes: int = 5, land: float = 0.015, lap: float = 0.02, stealer=None) -> list[float]:
    """The arc positions of the lands at ``x`` (the centre of each lap), along the clinker section
    from the keel - the same on either side. Integrated on the analytic section, 1e-5 m apart."""
    centres, active = _lands(x, strakes, stealer)
    centres = centres[active]
    s = np.linspace(0.0, girth(x), 300001)
    offset = land * np.clip((s[:, None] - centres[None]) / lap + 0.5, 0.0, 1.0).sum(axis=1)
    y, z = _section_points(x, s, offset, 1.0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(y), np.diff(z)))])
    return [float(v) for v in np.interp(centres, s, arc)]


def clinker_project(tmp_path: Path, heading_deg: float = 0.0, offset=UTM, crs: str = "EPSG:32633",
                    scale: float = 2.0, hole=None, **hull):
    """As :func:`project`, with :func:`clinker_hull` (``hull``: its arguments). ``hole``: ``(x0, x1,
    y0, y1)`` of the hull: the faces whose centre lies inside are taken away (a cut in the mesh)."""
    vertices, faces = clinker_hull(**hull)
    if hole is not None:
        centre = vertices[faces].mean(axis=1)
        x0, x1, y0, y1 = hole
        inside = (centre[:, 0] > x0) & (centre[:, 0] < x1) & (centre[:, 1] > y0) & (centre[:, 1] < y1)
        faces = faces[~inside]
    return _project(tmp_path, vertices, faces, heading_deg, offset, crs, scale)


def clinker_section(x, cell: float = 0.02, strakes: int = 5, land: float = 0.015, lap: float = 0.02,
                    stealer=None) -> np.ndarray:
    """The clinker hull's section at ``x`` as one polyline ``(N, 2)`` (y, z), starboard sheer to port sheer, the
    columns of :func:`clinker_hull` (for tests that make a section by hand: closed, over a seabed, with a stray piece)."""
    s, off = _side(x, cell, strakes, land, lap, stealer)
    yp, zp = _section_points(x, s, off, 1.0)
    ys, zs = _section_points(x, s[1:], off[1:], -1.0)
    return np.column_stack([np.concatenate([ys[::-1], yp]), np.concatenate([zs[::-1], zp])])


def clinker_widths(x, strakes: int = 5, land: float = 0.015, lap: float = 0.02, stealer=None) -> list[float]:
    """The widths of the strakes at ``x`` from the keel (the arcs between the lands' centres, the last to the sheer)
    along the clinker section, as :func:`clinker_truth` measures them."""
    centres, active = _lands(x, strakes, stealer)
    centres = centres[active]
    s = np.linspace(0.0, girth(x), 300001)
    offset = land * np.clip((s[:, None] - centres[None]) / lap + 0.5, 0.0, 1.0).sum(axis=1)
    y, z = _section_points(x, s, offset, 1.0)
    arc = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(y), np.diff(z)))])
    edges = np.r_[0.0, np.interp(centres, s, arc), arc[-1]]
    return [float(v) for v in np.diff(edges)]


# ------------------------------------------------------------------------- a framed clinker hull
FRAMES = [0.9 + 0.45 * k for k in range(23)]
FRAME_ROOM, FRAME_MOULDED = 0.08, 0.06  # the frames: x across, height above the planking
KEELSON_HALF, KEELSON_X, KEELSON_HEIGHT = 0.06, (1.0, 11.0), 0.12  # the keelson: |y|, its x, height above the planking
_WALL = 5e-5  # half the run of a timber's side: two copies of the edge 0.1 mm apart, the inner one raised


def _columns(cell: float, frames, keelson: bool):
    """The x of the hull's columns: the regular grid and both copies of every edge of a frame or of the keelson;
    for each column whether it lies inside a frame and inside the keelson's x."""
    xs = list(np.linspace(0.2, L - 0.2, int((L - 0.4) / cell) + 1))
    edges = [e for f in frames for e in (f - FRAME_ROOM / 2.0, f + FRAME_ROOM / 2.0)]
    if keelson:
        edges += list(KEELSON_X)
    for e in edges:
        xs += [e - _WALL, e + _WALL]
    xs = np.unique(np.round(xs, 9))
    in_frame = np.zeros(len(xs), bool)
    for f in frames:
        in_frame |= np.abs(xs - f) < FRAME_ROOM / 2.0
    in_keel = (xs > KEELSON_X[0]) & (xs < KEELSON_X[1]) if keelson else np.zeros(len(xs), bool)
    return xs, in_frame, in_keel


LOOSE = (0.3, 0.08, 0.05)  # a loose timber lying on the hull: length, width, height above the hull at its centre


def _boxes(vertices: np.ndarray, loose) -> tuple[np.ndarray, np.ndarray]:
    """Closed boxes lying on the hull, ``loose`` = [(x, y, degrees from x)] in the hull's coordinates: :data:`LOOSE`
    boxes with a level top :data:`LOOSE` [2] above the hull's highest vertex within 2 cm of the centre; or
    [(x, y, degrees, length, width, thickness)]: a plank whose top follows the hull, each corner ``thickness`` above
    the hull's highest vertex within 2 cm of it. The bottoms are sunk 0.1 m."""
    out_v, out_f = [], []
    quad = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                     [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])

    def top_at(px, py):
        near = np.hypot(vertices[:, 0] - px, vertices[:, 1] - py) < 0.02
        return float(vertices[near, 2].max())

    for item in loose:
        x, y, deg = item[:3]
        length, width, height = item[3:] if len(item) == 6 else LOOSE
        a = np.radians(deg)
        corners = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * [length / 2.0, width / 2.0]
        plan = corners @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]]) + [x, y]
        if len(item) == 6:
            tops = np.array([top_at(*c) for c in plan]) + height
        else:
            tops = np.full(4, top_at(x, y) + height)
        box = np.vstack([np.column_stack([plan, tops - height - 0.1]), np.column_stack([plan, tops])])
        out_f.append(quad + len(vertices) + sum(len(v) for v in out_v))
        out_v.append(box)
    return np.vstack(out_v), np.vstack(out_f)


def framed_hull(cell: float = 0.02, strakes: int = 5, frames=FRAMES, keelson: bool = True, missing=None,
                x_range=None, land: float = 0.015, lap: float = 0.02, loose=None) -> tuple[np.ndarray, np.ndarray]:
    """The clinker hull of :func:`clinker_hull` with timbers on it, as vertical raises with sides 0.1 mm wide: a frame
    at each x of ``frames`` (room :data:`FRAME_ROOM` along x, moulded :data:`FRAME_MOULDED` above the planking, across
    the whole girth) and, with ``keelson``, the keelson (|y| < :data:`KEELSON_HALF`, x in :data:`KEELSON_X`,
    :data:`KEELSON_HEIGHT` above the planking, over the frames it crosses). ``missing=(x0, x1, side)``: the faces of
    ``side`` ("port" or "starboard" of the frame the tests set, bow at the start: port = the hull's y < 0) between
    those x taken away; ``x_range=(x0, x1)``: only the part of the hull between those x is kept; ``loose``: boxes
    lying on the hull (:func:`_boxes`)."""
    xs, in_frame, in_keel = _columns(cell, frames, keelson)
    rows = []
    for x, framed, keeled in zip(xs, in_frame, in_keel):
        s, off = _side(x, cell, strakes, land, lap, None)
        b = float(half_breadth(x))
        wall = float(_arc(KEELSON_HALF, 2.0 * D / b**2))
        s_all = np.unique(np.r_[s, wall - _WALL, wall + _WALL]) if keelson else s
        off_all = np.interp(s_all, s, off)
        raise_ = np.full(len(s_all), FRAME_MOULDED if framed else 0.0)
        if keelson and keeled:
            raise_ = np.where(s_all < wall, np.maximum(raise_, KEELSON_HEIGHT), raise_)
        yp, zp = _section_points(x, s_all, off_all, 1.0)
        ys, zs = _section_points(x, s_all[1:], off_all[1:], -1.0)
        y = np.concatenate([ys[::-1], yp])
        z = np.concatenate([(zs + raise_[1:])[::-1], zp + raise_])
        rows.append(np.column_stack([np.full(len(y), x), y, z]))
    m, n = len(rows[0]), len(rows)
    i, j = np.meshgrid(np.arange(n - 1), np.arange(m - 1), indexing="ij")
    a = (i * m + j).ravel()
    b_, c, d = a + 1, a + m, a + m + 1
    faces = np.vstack([np.column_stack([a, b_, c]), np.column_stack([b_, d, c])])
    vertices = np.vstack(rows)
    centre = vertices[faces].mean(axis=1)
    keep = np.ones(len(faces), bool)
    if missing is not None:
        x0, x1, side = missing
        on_side = centre[:, 1] < 0 if side == "port" else centre[:, 1] > 0
        keep &= ~((centre[:, 0] > x0) & (centre[:, 0] < x1) & on_side)
    if x_range is not None:
        keep &= (centre[:, 0] > x_range[0]) & (centre[:, 0] < x_range[1])
    faces = faces[keep]
    if loose:
        bv, bf = _boxes(vertices, loose)
        vertices, faces = np.vstack([vertices, bv]), np.vstack([faces, bf])
    return vertices, faces


def framed_project(tmp_path: Path, heading_deg: float = 0.0, offset=UTM, crs: str = "EPSG:32633", scale: float = 2.0,
                   **hull):
    """As :func:`project`, with :func:`framed_hull` (``hull``: its arguments)."""
    return _project(tmp_path, *framed_hull(**hull), heading_deg, offset, crs, scale)


def seam_y(x, strakes: int = 5, land: float = 0.015) -> list[float]:
    """The plan y of the centre of each land of the clinker hull at ``x``, from the keel (the port side's: the
    starboard's are the same with the sign changed)."""
    centres, _active = _lands(x, strakes, None)
    offsets = land * (np.arange(len(centres)) + 0.5)  # the lands below, and half of this one
    return [float(v) for v in _section_points(x, centres, offsets, 1.0)[0]]


def plan_half_breadth(x, strakes: int = 5, land: float = 0.015, lap: float = 0.02) -> float:
    """The plan half-breadth of the clinker hull at ``x``: the farthest y of its section."""
    s, off = _side(x, 0.001, strakes, land, lap, None)
    return float(_section_points(x, s, off, 1.0)[0].max())


def strake_widths(x, strakes: int = 5) -> list[float]:
    """The widths in plan of the strakes at ``x``, from the keel line (y 0) to the first land, between the lands, from
    the last land to the plan half-breadth."""
    return [float(v) for v in np.diff(np.r_[0.0, seam_y(x, strakes), plan_half_breadth(x, strakes)])]


def framed_truth(strakes: int = 5, frames=FRAMES, keelson: bool = True) -> dict:
    """What :func:`framed_hull` holds: ``frames`` [(x, room)], ``volumes`` {"frames": [...], "keelson"} in cubic metres,
    integrated in plan on a 1 mm grid along x (a raise times the hull's plan breadth; where a frame passes under the
    keelson its volume counts to the keelson, whose height is taken from the planking), ``strake_widths`` the function
    :func:`strake_widths`."""
    step = 0.001
    volumes = []
    for f in frames:
        xs = f - FRAME_ROOM / 2.0 + step * (np.arange(int(round(FRAME_ROOM / step))) + 0.5)
        width = np.array([2.0 * plan_half_breadth(x, strakes) for x in xs])
        if keelson:
            width -= np.where((xs > KEELSON_X[0]) & (xs < KEELSON_X[1]), 2.0 * KEELSON_HALF, 0.0)
        volumes.append(float(FRAME_MOULDED * width.sum() * step))
    keel = KEELSON_HEIGHT * 2.0 * KEELSON_HALF * (KEELSON_X[1] - KEELSON_X[0]) if keelson else 0.0
    return {"frames": [(float(f), FRAME_ROOM) for f in frames], "volumes": {"frames": volumes, "keelson": float(keel)},
            "strake_widths": lambda x: strake_widths(x, strakes)}
