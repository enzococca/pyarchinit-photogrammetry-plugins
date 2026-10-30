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
    vertices, faces = hull_vertices()
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
