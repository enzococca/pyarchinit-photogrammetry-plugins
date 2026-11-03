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
