"""The engine on synthetic truths (no CadQuery, no program window): a column with entasis meshed with triangles, a
cornice along an L-shaped path, arches of the four families with their piers, the fit of a mesh to a solid."""
from __future__ import annotations

import numpy as np
import pytest

from arch_samples import (CORNICE, HEIGHT, L_PATH, arch_mesh, arch_points, column_mesh, column_radius, column_volume,
                          cornice_mesh)
from ppg_architecture import engine


def _line_distance(p, point, direction):
    q = np.asarray(p) - point
    return float(np.linalg.norm(q - (q @ direction) * direction))


@pytest.fixture(scope="module")
def column():
    mesh, base, d = column_mesh(origin=(0.31, -0.27, 0.12), tilt_deg=2.0)
    p, dd, rms = engine.column_axis(mesh.vertices, mesh.faces)
    meas = engine.column_profile(mesh.vertices, mesh.faces, p, dd, rms=rms)
    return mesh, base, d, meas


def test_the_axis_of_a_tilted_column(column):
    _mesh, base, d, meas = column
    assert np.degrees(np.arccos(min(abs(meas.direction @ d), 1.0))) < 0.1
    assert meas.direction[2] > 0  # up
    assert _line_distance(meas.point, base, d) < 0.0005
    assert abs((meas.point - base) @ d) < 0.0005  # the base where the column starts
    assert abs(meas.height - HEIGHT) < 0.001


def test_knots_volume_entasis_and_fit_of_the_column(column):
    mesh, _base, _d, meas = column
    zk, rk = engine.fit_knots(meas.heights, meas.radii, 12)
    assert len(zk) == 12 and zk[0] == 0.0 and abs(zk[-1] - HEIGHT) < 0.001
    assert np.abs(rk - column_radius(zk)).max() < 0.001
    curve = engine.profile_curve(zk, rk)
    assert abs(engine.revolved_volume(curve) / column_volume() - 1) < 0.005
    z = np.linspace(0, HEIGHT, 2001)
    r = column_radius(z)
    truth = (r - (r[0] + (r[-1] - r[0]) * z / HEIGHT)).max()
    assert abs(engine.entasis(curve) - truth) < 0.001
    solid = engine.revolved_mesh(curve)
    local = engine.apply(np.linalg.inv(engine.column_frame(meas.point, meas.direction)), mesh.vertices)
    fit = engine.fit_report(engine.distances_to_mesh(local, solid.vertices, solid.faces))
    assert fit["median"] < 0.0005 and fit["points"] == len(mesh.vertices)


def test_the_column_as_measured_and_with_utm_coordinates():
    mesh, base, d = column_mesh(origin=(712345.0, 4567890.0, 312.0), tilt_deg=0.0)
    o = np.floor(mesh.vertices.min(axis=0))
    v = np.asarray(mesh.vertices, np.float64) - o  # the plugin works in the export frame
    p, dd, rms = engine.column_axis(v, mesh.faces)
    meas = engine.column_profile(v, mesh.faces, p, dd, rms=rms)
    assert _line_distance(meas.point + o, base, d) < 0.0005
    assert np.abs(meas.radii - column_radius(meas.heights)).max() < 0.001  # the measured profile itself


def test_points_in_a_plan_polygon_select_a_part():
    mesh, _base, _d = column_mesh(tilt_deg=0.0)
    poly = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    v, f = engine.select_faces(mesh.vertices, mesh.faces, poly, zmin=1.0, zmax=2.0)
    assert len(f) and v[:, 2].min() >= 1.0 and v[:, 2].max() <= 2.0
    v2, _f2 = engine.select_faces(mesh.vertices, mesh.faces, box=(-1, -1, 1.0, 1, 1, 2.0))
    assert len(v2) == len(v)


def test_the_cornice_section_and_its_sweep_back():
    mesh, plug = cornice_mesh()
    assert mesh.is_watertight
    for station in (1.5, 5.0):  # on each leg of the L
        point, tangent = engine.point_at(L_PATH, station)
        o, u, v, _n = engine.section_frame(tangent, point)
        lines = engine.mesh_section(mesh.vertices, mesh.faces, o, u, v)
        sec = engine.moulding_section(lines, (-0.6, -0.2, 0.6, 0.5), out=-1, depth=0.1)
        assert abs(abs(engine.polygon_area(sec)) - abs(engine.polygon_area(plug))) < 1e-6
        volume = engine.sweep_volume(L_PATH, sec)
        assert abs(volume / abs(mesh.volume) - 1) < 0.01


def test_a_cornice_against_the_wall_is_closed_by_the_given_depth():
    # only the visible profile: an open line from the wall face out and back
    profile = np.column_stack([-CORNICE[:-1, 0], CORNICE[:-1, 1]])
    sec = engine.moulding_section([profile], (-0.5, -0.1, 0.2, 0.4), out=-1, depth=0.05)
    assert sec[:, 0].max() == pytest.approx(0.05)
    area = abs(engine.polygon_area(np.column_stack([CORNICE[:, 0], CORNICE[:, 1]])))
    assert abs(engine.polygon_area(sec)) == pytest.approx(area + 0.05 * 0.25, rel=1e-6)
    simple = engine.moulding_section([profile], (-0.5, -0.1, 0.2, 0.4), out=-1, depth=0.05, simplify=0.02)
    assert len(simple) < len(sec)


@pytest.mark.parametrize("kind", ["semicircular", "segmental", "pointed", "elliptical"])
def test_arches_are_recognised_and_measured(kind):
    mesh, truth = arch_mesh(kind)
    lines = engine.mesh_section(mesh.vertices, mesh.faces, (0.0, -truth["depth"] / 2, 0.0), (1, 0, 0), (0, 0, 1))
    pick = engine.guess_pick(lines)
    assert -1.0 < pick[0] < 1.0 and 0.0 < pick[1] < 2.5
    near, far = engine.visible_points(lines, pick)
    fit = engine.fit_arch(near)
    assert fit.type == kind
    assert abs(fit.span - truth["span"]) < 0.002 and abs(fit.rise - truth["rise"]) < 0.002
    if truth["radius"] is not None:
        assert abs(fit.params["R"] - truth["radius"]) < 0.002
    assert abs(engine.arch_thickness(fit, far) - truth["thickness"]) < 0.002
    outline = engine.arch_outline(fit, truth["thickness"])
    ring = engine.outline_polygon(outline, arc_points=400)
    from shapely.geometry import Polygon, box

    uvd = np.column_stack([mesh.vertices[:, 0], mesh.vertices[:, 2], -mesh.vertices[:, 1]])
    lo, hi = engine.footprint_depth(uvd, ring)
    assert abs((hi - lo) - truth["depth"]) < 0.02
    cut = Polygon(ring).intersection(box(-5, 1.5, 5, 5))
    assert abs(cut.area - abs(engine.polygon_area(ring))) < 1e-3


def test_the_type_can_be_asked_and_measured_keeps_the_points():
    pts, truth = arch_points("pointed")
    fit = engine.fit_arch(pts, "semicircular")
    assert fit.family == "circle" and fit.rms > 0.01  # a circle on a pointed arch: the fit says it
    measured = engine.fit_arch(pts, "measured")
    assert measured.type == "measured" and len(measured.measured) > 100
    outline = engine.arch_outline(measured, 0.3)
    assert abs(engine.polygon_area(outline["points"])) > 0.3
    assert abs(measured.span - truth["span"]) < 0.002


def test_distances_to_a_mesh_are_exact():
    cube = engine.prism_mesh(np.array([[0, 0], [1, 0], [1, 1], [0, 1.0]]), 1.0)
    p = np.array([[0.5, 0.5, 2.0], [2.0, 2.0, 0.5], [0.5, 0.5, 0.5], [-1.0, 0.5, -1.0]])
    d = engine.distances_to_mesh(p, cube.vertices, cube.faces)
    np.testing.assert_allclose(d, [1.0, np.sqrt(2.0), 0.5, np.sqrt(2.0)], atol=1e-9)
    assert abs(cube.volume - 1.0) < 1e-12


def test_the_frames_of_a_path():
    fr = engine.path_frames(L_PATH)
    np.testing.assert_allclose(fr["u"][0], [0, 1, 0], atol=1e-12)  # up x +x
    np.testing.assert_allclose(fr["v"][0], [0, 0, 1], atol=1e-12)
    assert fr["mitre"][1] == pytest.approx(np.sqrt(2.0))
    assert engine.is_straight(L_PATH[:2]) and not engine.is_straight(L_PATH)
    assert engine.path_length(L_PATH) == pytest.approx(7.0)


def test_what_lies_behind_the_back_of_a_section():
    path = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])  # u = +y, v = up
    pts = np.array([[1.0, -0.1, 0.1], [1.0, 0.05, 0.1], [1.0, 0.2, 0.1], [2.0, 0.099, 0.0]])
    behind = engine.behind_back(pts, path, back=0.1, out=-1, margin=0.005)  # projects to -u: the wall is towards +u
    assert behind.tolist() == [False, False, True, True]
    np.testing.assert_allclose(engine.path_coordinates(pts[:1], path), [[1.0, -0.1, 0.1]])
    assert engine.behind_back(pts, path, back=0.0, out=1, margin=0.0).tolist() == [True, False, False, False]


def test_a_section_closed_on_its_last_point_has_no_repeated_point():
    # the back line starts where the profile ends: the corner is not written twice (OpenCascade refuses it)
    profile = np.array([[-0.001, 0.3], [-0.18, 0.25], [-0.04, 0.0], [-0.001, -0.03]])
    sec = engine.moulding_section([profile], (-0.4, -0.03, 0.02, 0.4), out=-1, depth=0.0)
    edges = np.linalg.norm(np.diff(np.vstack([sec, sec[:1]]), axis=0), axis=1)
    assert edges.min() > 1e-7 and len(sec) == 4


def test_a_column_with_the_fins_of_a_photogrammetric_mesh():
    """Flying points at the silhouette make thin radial fins around a shaft: they spoil the normals the first axis
    comes from and pull a median radius outwards. The vertical start, the trimmed circles and the densest radius
    per height keep the column."""
    import trimesh

    mesh, base, d = column_mesh(tilt_deg=0.0)
    rng = np.random.default_rng(3)
    fins = []
    for _ in range(400):  # thin slivers at 1-1.6 m, from the surface out to 0.2 m beyond it
        a, z = rng.uniform(0, 2 * np.pi), rng.uniform(1.0, 1.6)
        r0 = column_radius(z)
        e = np.array([np.cos(a), np.sin(a), 0.0])
        t = np.array([-np.sin(a), np.cos(a), 0.0]) * 0.002
        p0 = e * r0 + [0, 0, z]
        p1 = e * (r0 + rng.uniform(0.05, 0.2)) + [0, 0, z + rng.uniform(-0.15, 0.15)]
        fins.append(trimesh.Trimesh([p0, p1, p0 + t + [0, 0, 0.05]], [[0, 1, 2]], process=False))
    m = trimesh.util.concatenate([mesh] + fins)
    p, dd, rms = engine.column_axis(m.vertices, m.faces)
    assert np.degrees(np.arccos(min(abs(dd @ d), 1.0))) < 0.1 and _line_distance(p, base, d) < 0.001
    meas = engine.column_profile(m.vertices, m.faces, p, dd, rms=rms)
    zk, rk = engine.fit_knots(meas.heights, meas.radii, 12)
    assert np.abs(rk - column_radius(zk)).max() < 0.002
