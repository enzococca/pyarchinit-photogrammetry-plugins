"""Openings (doors and windows) in the engine, on synthetic walls meshed as photogrammetry would mesh them: a door with
square reveals, a window splayed 10 degrees on both sides, a window with a semicircular top; with both faces of the
wall on the model and with the outer face only (the thickness then given)."""
from __future__ import annotations

import numpy as np
import pytest

from arch_samples import WALL_T, opening_wall
from ppg_architecture import engine

INSIDE = (0.0, 0.25)  # a point in the middle of the opening, in plan
OUT = (0.0, -1.0)  # the photos are in front of the outer face


def _check(r, truth, measured=True):
    p = r["plan"]
    assert abs(p.width_outer - truth["width"]) < 0.002
    assert abs(p.width_inner - truth["inner_width"]) < 0.002
    assert abs(p.thickness - WALL_T) < 0.002 and p.measured is measured
    assert abs(p.splay_left - truth["splay"]) < 0.2 and abs(p.splay_right - truth["splay"]) < 0.2
    assert abs(r["sill"] - truth["sill"]) < 0.002
    assert p.axis @ OUT > 0.999  # across the wall, towards the photos
    poly = np.asarray(p.polygon)
    assert abs(abs(engine.polygon_area(poly)) - abs(engine.polygon_area(truth["plan"]))) < 0.002
    for corner in truth["plan"]:
        assert np.min(np.linalg.norm(poly - corner, axis=1)) < 0.003


@pytest.mark.parametrize("kind", ["door", "splayed"])
@pytest.mark.parametrize("both", [True, False], ids=["both-faces", "outer-face-only"])
def test_a_door_and_a_splayed_window(kind, both):
    mesh, truth = opening_wall(kind, both_faces=both)
    r = engine.measure_opening(mesh.vertices, mesh.faces, INSIDE, outward=OUT, thickness=None if both else WALL_T)
    _check(r, truth, measured=both)
    assert abs(r["lintel"] - truth["lintel"]) < 0.002
    assert truth["sill"] - 0.01 < r["heights_found"][0] and r["heights_found"][1] < truth["lintel"] + 0.01
    assert abs(r["height"] - (truth["sill"] + truth["lintel"]) / 2) < 0.05  # the cut in the middle of the opening
    assert abs(r["width_at_sill"] - truth["width"]) < 0.002 and abs(r["width_at_lintel"] - truth["width"]) < 0.002


def test_only_one_face_asks_for_the_thickness():
    mesh, _truth = opening_wall("door", both_faces=False)
    with pytest.raises(ValueError, match="thickness"):
        engine.measure_opening(mesh.vertices, mesh.faces, INSIDE, outward=OUT)


def test_a_given_thickness_and_a_given_height_take_the_place_of_the_measured_ones():
    mesh, truth = opening_wall("splayed")
    r = engine.measure_opening(mesh.vertices, mesh.faces, INSIDE, outward=OUT, thickness=0.4, height=1.3)
    p = r["plan"]
    assert r["height"] == 1.3 and r["heights_found"] is None
    assert abs(p.thickness - 0.4) < 1e-9 and not p.measured
    assert abs(p.width_inner - (truth["width"] + 2 * 0.4 * np.tan(np.radians(10)))) < 0.002


def test_a_point_not_in_an_opening_is_said():
    mesh, _truth = opening_wall("door")
    with pytest.raises(ValueError, match="not inside an opening"):
        engine.measure_opening(mesh.vertices, mesh.faces, (1.5, 0.25), outward=OUT)


def test_a_window_with_an_arch_top():
    """The cut below the springing line (the arch closes the opening above it): the reveals, the sill; the void
    under the intrados as lines and a true arc, its area that of the half disc."""
    mesh, truth = opening_wall("arched")
    r = engine.measure_opening(mesh.vertices, mesh.faces, INSIDE, outward=OUT, top_limit=truth["springing"])
    _check(r, truth)
    assert r["heights_found"][1] < truth["springing"] + 0.01
    assert abs(r["lintel_measured"] - truth["lintel"]) < 0.003  # what the point sees above it: the crown
    # the intrados on a section of the wall (the plane y = 0.25, u = x, v = z) and the void under it
    lines = engine.mesh_section(mesh.vertices, mesh.faces, (0.0, 0.25, 0.0), (1, 0, 0), (0, 0, 1))
    near, _far = engine.visible_points(lines, (0.0, 1.4))
    fit = engine.fit_arch(near, "semicircular")
    assert abs(fit.springing - truth["springing"]) < 0.003 and abs(fit.span - truth["width"]) < 0.003
    void = engine.arch_void_outline(fit)
    assert [s[0] for s in void["segments"]] == ["A", "L"]
    area = abs(engine.polygon_area(engine.outline_polygon(void, arc_points=2000)))
    assert abs(area - np.pi * 0.4 ** 2 / 2) < 0.002
    for kind in ("pointed", "elliptical", "measured"):
        other = engine.fit_arch(near, kind)
        poly = engine.outline_polygon(engine.arch_void_outline(other), arc_points=400)
        assert engine.polygon_area(poly) > 0 and abs(abs(engine.polygon_area(poly)) - area) < 0.03


def test_a_noisy_model():
    """A millimetre of noise on every vertex (as a dense model has): widths, thickness, splays and sill still right
    within a few millimetres."""
    mesh, truth = opening_wall("splayed")
    rng = np.random.default_rng(3)
    v = np.asarray(mesh.vertices) + rng.normal(0.0, 0.001, (len(mesh.vertices), 3))
    r = engine.measure_opening(v, mesh.faces, INSIDE, outward=OUT)
    p = r["plan"]
    assert abs(p.width_outer - truth["width"]) < 0.004 and abs(p.width_inner - truth["inner_width"]) < 0.004
    assert abs(p.thickness - WALL_T) < 0.004 and abs(p.splay_left - 10) < 0.5 and abs(p.splay_right - 10) < 0.5
    assert abs(r["sill"] - truth["sill"]) < 0.004 and abs(r["lintel"] - truth["lintel"]) < 0.004
    assert p.reveal_rms < 0.003


def test_the_volume_of_an_opening():
    plan = np.array([[0, 0], [1, 0], [1, 0.5], [0, 0.5]], float)
    assert engine.opening_volume(plan, 1.0, 3.0) == pytest.approx(1.0)
    assert engine.opening_volume(plan, 1.0, 3.0, 0.25, 0.5) == pytest.approx(1.125)
