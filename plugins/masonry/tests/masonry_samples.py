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
