"""Synthetic scenes of the tests, with their truths: a wall leaning 2 degrees towards the viewer with a Gaussian bulge
of
3 cm, a round tower of radius 3 m leaning 1 degree with an out-of-round of 1 cm, a hexagonal tower, two surveys of a
wall 2 cm apart. Everything in world coordinates (a projected CRS far from the origin); :func:`project` puts the meshes
in a project whose chunk has a rotated and scaled transform, so the model.ply holds small internal coordinates."""
from __future__ import annotations

from pathlib import Path

import numpy as np

WORLD = np.array([712345.678, 4567890.123, 312.5])
CRS = "EPSG:32633"
LEAN_DEG = 2.0
BULGE = 0.03  # metres
BULGE_AT = (3.0, 1.5)  # (along the wall, height above the base)
BULGE_SIGMA = 0.45
WALL_LENGTH, WALL_HEIGHT = 6.0, 3.0
TOWER_RADIUS, TOWER_HEIGHT, TOWER_LEAN_DEG, TOWER_LEAN_AZIMUTH = 3.0, 6.0, 1.0, 60.0
OUT_OF_ROUND = 0.01  # peak to valley of the radius
HEX_RADIUS, HEX_HEIGHT = 2.5, 4.0
STEP = 0.02


def bulge_truth(u, h):
    """The bulge (metres, towards the viewer) at ``u`` along the wall and ``h`` above its base."""
    return BULGE * np.exp(-((u - BULGE_AT[0]) ** 2 + (h - BULGE_AT[1]) ** 2) / (2 * BULGE_SIGMA ** 2))


def _grid(f, nu: int, nv: int):
    """A mesh of the surface ``f(s, t)`` (s, t in [0, 1]), (nu x nv) vertices, two triangles per cell."""
    s, t = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    verts = f(s.ravel(), t.ravel())
    idx = np.arange(nu * nv).reshape(nu, nv)
    a, b, c, d = idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return verts, faces


def wall(base=None, offset: float = 0.0, lean: bool = True, bulge: bool = True):
    """The face of a wall along +x from ``base`` (world), seen from -y; the face leans ``LEAN_DEG`` towards the viewer
    (-y) and bulges towards it; ``offset`` moves it all towards the viewer. Returns (vertices, faces)."""
    base = WORLD if base is None else np.asarray(base, np.float64)
    k = np.tan(np.radians(LEAN_DEG)) if lean else 0.0

    def f(s, t):
        u, h = s * WALL_LENGTH, t * WALL_HEIGHT
        y = -h * k - (bulge_truth(u, h) if bulge else 0.0) - offset
        return base + np.column_stack([u, y, h])

    return _grid(f, 241, 121)


def tower_axis():
    """The unit axis of the round tower (leaning ``TOWER_LEAN_DEG`` towards the azimuth ``TOWER_LEAN_AZIMUTH``)."""
    a, z = np.radians(TOWER_LEAN_AZIMUTH), np.radians(TOWER_LEAN_DEG)
    return np.array([np.sin(a) * np.sin(z), np.cos(a) * np.sin(z), np.cos(z)])


def round_tower(base=None):
    """An open tube: radius ``TOWER_RADIUS`` + ``OUT_OF_ROUND``/2 cos 2(phi - 20 degrees) across the axis."""
    base = WORLD if base is None else np.asarray(base, np.float64)
    axis = tower_axis()
    e1 = np.cross(axis, [0.0, 0.0, 1.0])
    e1 = e1 / np.linalg.norm(e1) if np.linalg.norm(e1) > 1e-9 else np.array([1.0, 0.0, 0.0])
    e2 = np.cross(axis, e1)

    def f(s, t):
        phi = s * 2 * np.pi
        r = TOWER_RADIUS + OUT_OF_ROUND / 2 * np.cos(2 * (phi - np.radians(20.0)))
        h = t * TOWER_HEIGHT / axis[2]
        return base + h[:, None] * axis + (r * np.cos(phi))[:, None] * e1 + (r * np.sin(phi))[:, None] * e2

    v, faces = _grid(f, 721, 61)
    # the seam: the last column of vertices is the first one
    n = 61
    v = v[:-n]
    faces = np.where(faces >= len(v), faces - len(v), faces)
    return v, faces


def hex_tower(base=None):
    """A hexagonal prism (open top and bottom), circumradius ``HEX_RADIUS``, a corner at azimuth 0 (+y)."""
    base = WORLD if base is None else np.asarray(base, np.float64)
    corners = [np.array([np.sin(a), np.cos(a)]) * HEX_RADIUS for a in np.radians(np.arange(0, 360, 60))]
    verts, faces = [], []
    for i in range(6):
        p, q = corners[i], corners[(i + 1) % 6]

        def f(s, t, p=p, q=q):
            xy = p + s[:, None] * (q - p)
            return base + np.column_stack([xy, t * HEX_HEIGHT])

        v, fc = _grid(f, 51, 41)
        faces.append(fc + sum(len(x) for x in verts))
        verts.append(v)
    return np.concatenate(verts), np.concatenate(faces)


def hex_normals():
    """The outward normals (x, y) of the six sides."""
    mids = [np.radians(a + 30.0) for a in range(0, 360, 60)]
    return [np.array([np.sin(a), np.cos(a)]) for a in mids]


def transform():
    from pyarchinit_photogrammetry.core.geometry import compose_similarity

    a, b = np.radians(30.0), np.radians(2.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    rx = np.array([[1, 0, 0], [0, np.cos(b), -np.sin(b)], [0, np.sin(b), np.cos(b)]])
    return compose_similarity(2.5, rz @ rx, WORLD)


def add_chunk(doc, label: str, mesh):
    """A chunk of ``doc`` holding ``mesh`` (world vertices, faces) as its model."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    ch = doc.add_chunk(label)
    ch.crs = CRS
    ch.transform = transform()
    verts, faces = mesh
    internal = ch.to_internal(np.asarray(verts, np.float64))
    path = write_mesh_ply(ch.data_dir / "model.ply", internal, np.asarray(faces), double=True)
    ch._set_product("model", path)
    ch.meta["metric"] = True
    return ch


def project(folder: Path, mesh, label: str = "Muro"):
    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    doc.save(Path(folder) / "murature.ppg")
    # the default chunk of a new document, if any, is replaced by ours
    ch = add_chunk(doc, label, mesh)
    doc.chunk = ch
    doc.save()
    return doc, ch


# ------------------------------------------------------------------ stones and units drawn on a face
COURSES, PITCH, JOINT_H, JOINT_V, CHAMFER = 5, 0.35, 0.025, 0.025, 0.02
LENGTHS = (0.42, 0.55, 0.48, 0.60, 0.38, 0.51, 0.45)
STONE_WALL = 6.0


def _stone(u0, v0, u1, v1, c=CHAMFER):
    from shapely.geometry import Polygon

    return Polygon([(u0 + c, v0), (u1 - c, v0), (u1, v0 + c), (u1, v1 - c), (u1 - c, v1), (u0 + c, v1), (u0, v1 - c),
                    (u0, v0 + c)])


def stone_pattern(v_base: float, hole=None):
    """Stones in ``COURSES`` courses of pitch ``PITCH`` from ``v_base`` along ``STONE_WALL`` metres (``u`` from 0),
    joints of ``JOINT_H`` and ``JOINT_V``, the lengths of ``LENGTHS`` in turn; none whose centre falls in ``hole`` (a
    shapely polygon). Returns ``(polygons, truths)``: ``truths`` = per stone (width, height, course centre)."""
    from shapely.geometry import Point

    polys, truths = [], []
    k = 0
    for course in range(COURSES):
        v0 = v_base + course * PITCH + JOINT_H / 2
        v1 = v0 + PITCH - JOINT_H
        u = JOINT_V / 2  # the joints stagger: the lengths go on from the course below
        while True:
            length = LENGTHS[k % len(LENGTHS)]
            k += 1
            room = STONE_WALL - JOINT_V / 2 - u
            if room < 0.25:
                break
            length = min(length, room)  # the last stone of a course cut to the end of the wall
            p = _stone(u, v0, u + length, v1)
            if hole is None or not hole.contains(Point(p.centroid.x, p.centroid.y)):
                polys.append(p)
                truths.append((length, v1 - v0, (v0 + v1) / 2))
            u += length + JOINT_V
    return polys, truths


def irregular_stones(v_base: float, n: int = 60, seed: int = 3):
    """Stones scattered without courses: ellipse-like outlines of random size (0.12-0.4 m) and rotation."""
    from shapely import affinity
    from shapely.geometry import Point

    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        a, b = rng.uniform(0.08, 0.2), rng.uniform(0.03, 0.08)
        p = affinity.rotate(affinity.scale(Point(0, 0).buffer(1.0, 8), a, b), rng.uniform(0, 180))
        p = affinity.translate(p, rng.uniform(0.3, STONE_WALL - 0.3), v_base + rng.uniform(0.3, 1.5))
        if all(not p.intersects(q.buffer(0.01)) for q in out):
            out.append(p)
    return out


def usm_layout(v_base: float):
    """Four USM on a face (``u`` from 0): USM 10 the coursed wall with a notch, USM 12 the patch in the notch (it cuts
    the courses of 10), USM 11 a block on top of 10, USM 13 lying on the top of 11 and wider than it."""
    from shapely.geometry import box

    top = v_base + COURSES * PITCH
    notch = box(4.0, v_base + 0.5, 5.0, top)
    u10 = box(0.0, v_base, STONE_WALL, top).difference(notch)
    u11 = box(0.5, top, 2.5, top + 0.8)
    u13 = box(0.0, top + 0.8, 3.0, top + 1.4)
    return {"10": u10, "11": u11, "12": notch, "13": u13}, notch


def put_drawings(ch, eid: str, stones=(), units=None, sito: str = "Saggio"):
    """The stones and the units drawn on the elevation ``eid`` as the window keeps them."""
    frames = ch.meta.setdefault("drawings", {}).setdefault("frames", {})
    data = frames.setdefault(eid, {})
    data["stones"] = {eid: [p.wkt for p in stones]}
    if units is not None:
        data["units"] = [{"wkt": g.wkt, "us": us, "tipo": "USM", "area": "1", "sito": sito,
                          "attributes": {"d_stratigrafica": f"Muratura {us}"}} for us, g in units.items()]


def broken_tower(keep_deg=None, gaps_deg=((30, 70), (120, 150), (200, 260), (300, 320)), base=None):
    """The round tower with its ring broken: the triangles whose direction from the axis (degrees from +x,
    anticlockwise) falls in ``gaps_deg`` taken away - or, with ``keep_deg`` (from, to), only those within it kept."""
    v, f = round_tower(base)
    b = WORLD if base is None else np.asarray(base, np.float64)
    c = v[f].mean(axis=1)
    axis = tower_axis()
    t = (c - b) @ axis
    radial = c - b - t[:, None] * axis
    ang = np.degrees(np.arctan2(radial[:, 1], radial[:, 0])) % 360.0
    if keep_deg is not None:
        ok = (ang >= keep_deg[0]) & (ang <= keep_deg[1])
    else:
        ok = np.ones(len(f), bool)
        for a0, a1 in gaps_deg:
            ok &= ~((ang >= a0) & (ang <= a1))
    return v, f[ok]


def ellipse_tower(a: float = 3.3, b: float = 2.7, height: float = 6.0, base=None):
    """An upright tower of elliptic plan (semi-axes ``a`` along x, ``b`` along y): not a cylinder."""
    base = WORLD if base is None else np.asarray(base, np.float64)

    def f(s, t):
        phi = s * 2 * np.pi
        return base + np.column_stack([a * np.cos(phi), b * np.sin(phi), t * height])

    v, faces = _grid(f, 361, 41)
    n = 41
    v = v[:-n]
    return v, np.where(faces >= len(v), faces - len(v), faces)


def cone_tower(r0: float = 3.4, r1: float = 2.6, height: float = 6.0, base=None):
    """An upright battered tower: radius ``r0`` at its foot, ``r1`` at its top."""
    base = WORLD if base is None else np.asarray(base, np.float64)

    def f(s, t):
        phi = s * 2 * np.pi
        r = r0 + (r1 - r0) * t
        return base + np.column_stack([r * np.cos(phi), r * np.sin(phi), t * height])

    v, faces = _grid(f, 361, 41)
    n = 41
    v = v[:-n]
    return v, np.where(faces >= len(v), faces - len(v), faces)
