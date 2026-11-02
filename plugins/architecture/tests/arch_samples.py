"""Synthetic truths for the tests: a column with entasis, a cornice along an L-shaped path, arches with their piers
in a section, meshed as photogrammetry would mesh them (triangles, no parameters)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HEIGHT = 3.0


def column_radius(z):
    """A shaft 3 m high: 0.30 m at the base, 0.26 at the top, swelling about 19 mm over the chord from base to top
    (the entasis)."""
    z = np.asarray(z, float)
    return 0.30 - 0.04 * (z / HEIGHT) ** 2 + 0.009 * np.sin(np.pi * z / HEIGHT)


def column_mesh(segments: int = 240, rings: int = 150, tilt_deg: float = 2.0, origin=(0.0, 0.0, 0.0)):
    """The column as a closed mesh, its axis tilted by ``tilt_deg`` towards x and its base at ``origin``.
    Returns ``(mesh, point, direction)``; the vertices lie on the true surface."""
    import trimesh

    a = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    z = np.linspace(0, HEIGHT, rings)
    r = column_radius(z)
    v = np.concatenate([np.column_stack([r[i] * np.cos(a), r[i] * np.sin(a), np.full(segments, z[i])])
                        for i in range(rings)] + [[[0, 0, 0], [0, 0, HEIGHT]]])
    f = []
    for i in range(rings - 1):
        for j in range(segments):
            a0, a1 = i * segments + j, i * segments + (j + 1) % segments
            f += [[a0, a1, a1 + segments], [a0, a1 + segments, a0 + segments]]
    b, t = rings * segments, rings * segments + 1
    for j in range(segments):
        f += [[b, (j + 1) % segments, j], [t, (rings - 1) * segments + j, (rings - 1) * segments + (j + 1) % segments]]
    th = np.radians(tilt_deg)
    rot = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    v = v @ rot.T + np.asarray(origin, float)
    return trimesh.Trimesh(v, np.asarray(f), process=False), np.asarray(origin, float), rot @ [0, 0, 1.0]


def column_volume(n: int = 200001) -> float:
    z = np.linspace(0, HEIGHT, n)
    return float(np.trapezoid(np.pi * column_radius(z) ** 2, z))


# a cornice: a cyma over a fillet, projecting 0.18 m from the wall, 0.25 m high; u out of the wall, v up
CORNICE = np.array([[0.0, 0.0], [0.04, 0.0], [0.04, 0.03], [0.06, 0.05], [0.10, 0.08], [0.13, 0.13], [0.16, 0.18],
                    [0.18, 0.21], [0.18, 0.25], [0.0, 0.25]])
L_PATH = np.array([[0.0, 0.0, 2.0], [4.0, 0.0, 2.0], [4.0, 3.0, 2.0]])


def cornice_mesh(depth: float = 0.3):
    """A wall corner with the cornice along it: the section (with ``depth`` m of wall behind it) swept along
    ``L_PATH`` with mitred corners. The wall face is the path; the cornice projects to the right of the travel
    (towards -u of the plugin's frame, u = up x tangent). Returns ``(mesh, section)``, the section in the plugin's
    frame (u, v) with v = 0 at the path's height."""
    engine = _engine()
    sec = np.vstack([CORNICE[:-1], [[-depth, 0.25], [-depth, 0.0]]])
    plug = np.column_stack([-sec[:, 0], sec[:, 1]])  # projecting to -u
    mesh = engine.sweep_mesh(L_PATH, plug)
    return mesh, plug


def _engine():
    """The plugin's engine, loaded from its file (the samples work without the plugin installed)."""
    import importlib.util

    name = "arch_engine_for_samples"
    if name not in sys.modules:
        here = Path(__file__).resolve().parent.parent / "ppg_architecture" / "engine.py"
        spec = importlib.util.spec_from_file_location(name, here)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def arch_points(kind: str, n: int = 400):
    """The intrados of an arch with its jambs, as (u, v), and its truth: span 2 m, springing at v = 1.5 m."""
    s, vs = 1.0, 1.5
    if kind == "semicircular":
        a = np.linspace(np.pi, 0, n)
        arc = np.column_stack([s * np.cos(a), vs + s * np.sin(a)])
        truth = {"span": 2 * s, "rise": s, "radius": s}
    elif kind == "segmental":
        R = 1.6
        vc = vs - np.sqrt(R * R - s * s)
        a0 = np.arcsin((vs - vc) / R)
        a = np.linspace(np.pi - a0, a0, n)
        arc = np.column_stack([R * np.cos(a), vc + R * np.sin(a)])
        truth = {"span": 2 * s, "rise": vc + R - vs, "radius": R}
    elif kind == "pointed":
        R = 2 * s  # equilateral
        cL, cR = -s + R, s - R
        top = np.arccos((0 - cL) / R)
        aL = np.linspace(np.pi, top, n // 2)
        aR = np.linspace(np.pi - top, 0, n // 2)
        arc = np.vstack([np.column_stack([cL + R * np.cos(aL), vs + R * np.sin(aL)]),
                         np.column_stack([cR + R * np.cos(aR), vs + R * np.sin(aR)])])
        truth = {"span": 2 * s, "rise": np.sqrt(R * R - (R - s) ** 2), "radius": R}
    elif kind == "elliptical":
        b = 0.6
        a = np.linspace(np.pi, 0, n)
        arc = np.column_stack([s * np.cos(a), vs + b * np.sin(a)])
        truth = {"span": 2 * s, "rise": b, "radius": None}
    else:
        raise ValueError(kind)
    jl = np.column_stack([np.full(60, -s), np.linspace(0.3, vs, 60, endpoint=False)])
    jr = np.column_stack([np.full(60, s), np.linspace(vs, 0.3, 60, endpoint=False)[::-1]])
    return np.vstack([jl, arc, jr[::-1]]), truth


def arch_mesh(kind: str, thickness: float = 0.4, depth: float = 0.8, ground: float = 0.0):
    """An arch with its piers (a doorway of a free-standing wall), meshed as a closed solid: the opening's outline
    (piers from the ground, the arch) offset by ``thickness`` outside, extruded by ``depth`` along -y: the plugin
    cuts it with the plane y = -depth / 2. Returns ``(mesh, truth)``; the section plane is x = u, z = v."""
    import trimesh
    from shapely.geometry import LineString, Polygon

    pts, truth = arch_points(kind)
    inner = np.vstack([[pts[0, 0], ground], pts, [pts[-1, 0], ground]])
    line = LineString(inner)
    outer = line.buffer(thickness, cap_style="flat", join_style="round", quad_segs=32)
    # keep the outside half: the region between the intrados and its offset outside
    opening = Polygon(np.vstack([inner, [[inner[-1, 0], ground - 1], [inner[0, 0], ground - 1]]]))
    ring = outer.difference(opening).buffer(0)
    if ring.geom_type != "Polygon":
        ring = max(ring.geoms, key=lambda g: g.area)
    ring = ring.intersection(Polygon([(-50, ground), (50, ground), (50, 50), (-50, 50)]))
    if ring.geom_type != "Polygon":
        ring = max(ring.geoms, key=lambda g: g.area)
    mesh = _engine().prism_mesh(np.asarray(ring.exterior.coords), depth)  # z from 0 to depth
    # x = u, y = v (up), z = depth -> world: x = u, y = -depth, z = v (a rotation of 90 degrees about x)
    mesh.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    v, f = trimesh.remesh.subdivide_to_size(mesh.vertices, mesh.faces, max_edge=0.05, max_iter=20)
    mesh = trimesh.Trimesh(v, f, process=False)
    truth = dict(truth, thickness=thickness, depth=depth)
    return mesh, truth


# openings: a wall along x, 5 m long and 3 m high, its outer face (towards the photos, -y) at y = 0, its inner face at
# y = WALL_T; the opening centred at x = 0
WALL_T = 0.5
OPENINGS = {
    # a door: 0.9 m wide, from the ground (z = 0) to 2.1 m, square reveals
    "door": {"width": 0.9, "sill": 0.0, "lintel": 2.1, "splay": 0.0, "top": "flat"},
    # a window splayed 10 degrees on both sides (wider at the inner face: an embrasure), 0.8 m wide at the outer face
    "splayed": {"width": 0.8, "sill": 1.0, "lintel": 2.0, "splay": 10.0, "top": "flat"},
    # a window with a semicircular top: 0.8 m wide, the springing line at 1.8 m, the crown at 2.2 m
    "arched": {"width": 0.8, "sill": 1.0, "lintel": 2.2, "springing": 1.8, "splay": 0.0, "top": "arch"},
}


def opening_wall(kind: str, both_faces: bool = True, ground: bool = True, edge: float = 0.04):
    """A wall with one opening of :data:`OPENINGS` as a photogrammetric mesh would be (triangles of ``edge`` metres),
    standing on a ground plane: built from horizontal slabs (plan polygons extruded between two heights) and, for an
    arched top, the part above the springing line as an elevation extruded across the wall. ``both_faces`` False: the
    inner face (y = WALL_T) is not there, as when only the outer face was photographed. Returns ``(mesh, truth)``;
    the truth also holds ``inner_width`` and ``plan`` (the void's corners in plan, outer face first)."""
    import trimesh
    from shapely.geometry import Point, box

    eng = _engine()
    o = OPENINGS[kind]
    x0, x1, top, T = -2.5, 2.5, 3.0, WALL_T
    half = o["width"] / 2
    grow = T * np.tan(np.radians(o["splay"]))
    plan = np.array([[-half, 0.0], [half, 0.0], [half + grow, T], [-half - grow, T]])
    parts = []

    def slab(poly, z0, z1):
        m = eng.prism_mesh(np.asarray(poly, float), z1 - z0)
        m.apply_translation([0.0, 0.0, z0])
        parts.append(m)

    full = [(x0, 0.0), (x1, 0.0), (x1, T), (x0, T)]
    left = [(x0, 0.0), (-half, 0.0), (-half - grow, T), (x0, T)]
    right = [(half, 0.0), (x1, 0.0), (x1, T), (half + grow, T)]
    if o["sill"] > 0:
        slab(full, 0.0, o["sill"])
    upto = o["springing"] if o["top"] == "arch" else o["lintel"]
    slab(left, o["sill"], upto)
    slab(right, o["sill"], upto)
    if o["top"] == "arch":
        wall = box(x0, upto, x1, top).difference(Point(0.0, upto).buffer(half, quad_segs=64))
        m = eng.prism_mesh(np.asarray(wall.exterior.coords), T)  # (x, z, y)
        v = np.asarray(m.vertices)[:, [0, 2, 1]]
        parts.append(trimesh.Trimesh(v, np.asarray(m.faces)[:, ::-1], process=False))  # y and z swapped: turned back
    else:
        slab(full, o["lintel"], top)
    if ground:
        g = np.array([[x0 - 0.5, -2.0, 0.0], [x1 + 0.5, -2.0, 0.0], [x1 + 0.5, T + 2.0, 0.0], [x0 - 0.5, T + 2.0, 0.0]])
        parts.append(trimesh.Trimesh(g, [[0, 1, 2], [0, 2, 3]], process=False))
    out = []
    for m in parts:
        v, f = trimesh.remesh.subdivide_to_size(m.vertices, m.faces, max_edge=edge, max_iter=20)
        out.append(trimesh.Trimesh(v, f, process=False))
    mesh = trimesh.util.concatenate(out)
    if not both_faces:
        tri = np.asarray(mesh.vertices)[np.asarray(mesh.faces)]
        back = np.abs(tri[:, :, 1] - T).max(axis=1) < 1e-6  # the faces lying in the plane of the inner face
        mesh = trimesh.Trimesh(mesh.vertices, np.asarray(mesh.faces)[~back], process=False)
    truth = dict(o, thickness=T, inner_width=o["width"] + 2 * grow, plan=plan)
    return mesh, truth


def cadquery_python() -> Path | None:
    """The Python of an environment with CadQuery for the worker's tests (``PPG_TEST_CADQUERY``)."""
    p = os.environ.get("PPG_TEST_CADQUERY")
    if p and Path(p).exists():
        return Path(p)
    return None

