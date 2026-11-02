"""The real worker with CadQuery (``PPG_TEST_CADQUERY``: the Python of an environment with CadQuery, as the plugin
installs it): the solids against the engine's own geometry, the placement, the STEP and the STL, the environment's
probe. Skipped without ``PPG_TEST_CADQUERY``. (The elements of a project through it and their IFC: test_plugin.py.)"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arch_samples import L_PATH, cadquery_python

CADQUERY = cadquery_python()
pytestmark = pytest.mark.skipif(CADQUERY is None, reason="no Python with CadQuery (PPG_TEST_CADQUERY)")
pytest.importorskip("pyarchinit_photogrammetry", reason="the runner needs the program")

from ppg_architecture import engine, environment, runner  # noqa: E402

WORKER = Path(runner.__file__).with_name("worker.py")


def _run(tmp_path, kind, spec, placement=None):
    job = {"protocol": 1, "kind": kind, "output": str(tmp_path / kind), kind: spec, "tolerance": 0.0005,
           "placement": (np.eye(4) if placement is None else placement).tolist()}
    return runner.run(job, tmp_path / f"work-{kind}", python=CADQUERY, script=WORKER)


def _stl(result):
    import trimesh

    m = trimesh.load(str(Path(result["output"]) / "solid.stl"), process=True, force="mesh")
    return m


def test_the_probe_of_the_environment():
    info = environment.probe(CADQUERY)
    assert environment.usable(info), info
    assert info["cadquery"].startswith("2.")


def test_a_column_by_revolution(tmp_path):
    z = np.linspace(0, 3, 12)
    r = 0.30 - 0.04 * (z / 3) ** 2 + 0.009 * np.sin(np.pi * z / 3)
    curve = engine.profile_curve(z, r)
    placement = engine.column_frame([10.0, 20.0, 30.0], [np.sin(0.05), 0.0, np.cos(0.05)])
    res = _run(tmp_path, "column", {"profile": curve.tolist(), "smooth": True}, placement)
    assert res["valid"] and abs(res["volume"] / engine.revolved_volume(curve) - 1) < 0.002
    out = Path(res["output"])
    assert (out / "solid.step").read_text(errors="replace").startswith("ISO-10303-21")
    step = (out / "solid.step").read_text(errors="replace")
    assert "AUTOMOTIVE_DESIGN" in step[:2000]  # AP214
    # metres: the file says so and the numbers are those of the solid (CadQuery's exporter would say millimetres)
    import re

    units = re.findall(r"LENGTH_UNIT\(\)[^;]*", step)
    assert units and "SI_UNIT($,.METRE.)" in units[0].replace(" ", ""), units
    points = np.array([[float(x) for x in m.split(",")] for m in re.findall(r"CARTESIAN_POINT\('',\(([^)]*)\)", step)
                       if m.count(",") == 2])
    assert np.abs(points - [10.0, 20.0, 30.0]).min(axis=0).max() < 0.5  # the solid's own coordinates, not x 1000
    stl = _stl(res)
    assert stl.is_watertight and abs(stl.volume / res["volume"] - 1) < 0.002
    centre = stl.center_mass
    assert np.hypot(centre[0] - 10.0 - 1.5 * np.sin(0.05), centre[1] - 20.0) < 0.01  # placed on the tilted axis
    poly = _run(tmp_path / "p", "column", {"profile": curve.tolist(), "smooth": False})
    assert abs(poly["volume"] / engine.revolved_volume(curve) - 1) < 0.001


def test_a_cornice_swept_along_an_l_and_a_straight_one(tmp_path):
    sec = np.array([[0.0, 0.0], [-0.04, 0.0], [-0.04, 0.03], [-0.10, 0.08], [-0.16, 0.18], [-0.18, 0.25],
                    [0.3, 0.25], [0.3, 0.0]])
    if engine.polygon_area(sec) < 0:
        sec = sec[::-1]
    res = _run(tmp_path, "moulding", {"path": L_PATH.tolist(), "section": sec.tolist()})
    assert res["valid"] and abs(res["volume"] / engine.sweep_volume(L_PATH, sec) - 1) < 0.001
    bb = res["bbox"]
    assert bb[2] == pytest.approx(2.0, abs=1e-6) and bb[5] == pytest.approx(2.25, abs=1e-6)
    straight = _run(tmp_path / "s", "moulding", {"path": L_PATH[:2].tolist(), "section": sec.tolist()})
    assert abs(straight["volume"] - abs(engine.polygon_area(sec)) * 4.0) < 1e-6


def test_an_arch_with_true_arcs(tmp_path):
    from arch_samples import arch_points

    pts, _truth = arch_points("pointed")
    fit = engine.fit_arch(pts)
    outline = engine.arch_outline(fit, 0.4)
    placement = engine.arch_frame([1.0, 2.0, 0.0], [1, 0, 0], [0, 0, 1])
    res = _run(tmp_path, "arch", {"outline": outline, "depth": 0.8}, placement)
    area = abs(engine.polygon_area(engine.outline_polygon(outline, arc_points=4000)))
    assert res["valid"] and abs(res["volume"] / (area * 0.8) - 1) < 1e-4
    bb = res["bbox"]
    assert bb[1] == pytest.approx(2.0 - 0.8, abs=1e-6) and bb[4] == pytest.approx(2.0, abs=1e-6)  # depth along -y


RAKING = [[[0, 0, 0], [4, 0, 1.5]], [[0, 0, 0], [3, 0, 1], [3, 3, 2]], [[0, 0, 0], [2, 0, 1], [4, 0, 0]]]


@pytest.mark.parametrize("path", RAKING, ids=["raking", "raking-turning", "gable"])
def test_a_raking_moulding_in_the_ifc_is_cadquerys_solid(tmp_path, path):
    """A sloped path: CadQuery sweeps with the corrected Frenet frame, the IFC is a fixed-reference sweep (up);
    ifcopenshell.geom gives back the same volume and bounding box, so the IFC stays parametric."""
    pytest.importorskip("ifcopenshell")
    from ppg_architecture import ifc

    sec = np.array([[0, 0], [-0.18, 0], [-0.18, 0.1], [-0.1, 0.18], [0, 0.25]], float)
    if engine.polygon_area(sec) < 0:
        sec = sec[::-1]
    res = _run(tmp_path, "moulding", {"path": path, "section": sec.tolist()})
    e = {"kind": "moulding", "name": "M", "params": {}, "solid": {"volume": res["volume"]},
         "geometry": {"path": path, "section": sec.tolist(), "placement": np.eye(4).tolist()}}
    volume, bbox = ifc.read_volumes(ifc.write(tmp_path / "m.ifc", [e], [0, 0, 0], "LOCAL"))["M"]
    assert abs(volume / res["volume"] - 1) < 0.01
    np.testing.assert_allclose(bbox, res["bbox"], atol=0.005)


def test_an_opening_and_one_closed_by_an_arch(tmp_path):
    """A splayed plan extruded from the sill to the lintel; and a plan up to the springing line united with the void
    under a semicircular arch whose plane crosses the wall: one solid, the volumes of the two parts added."""
    from arch_samples import arch_points

    t = np.tan(np.radians(10))
    plan = np.array([[-0.4, 0.0], [-0.4 - 0.5 * t, 0.5], [0.4 + 0.5 * t, 0.5], [0.4, 0.0]])[::-1]
    flat = _run(tmp_path, "opening", {"plan": plan.tolist(), "bottom": 1.0, "top": 2.0})
    assert flat["valid"] and abs(flat["volume"] - engine.opening_volume(plan, 1.0, 2.0)) < 1e-6
    assert flat["bbox"][2] == pytest.approx(1.0, abs=1e-6) and flat["bbox"][5] == pytest.approx(2.0, abs=1e-6)
    rect = np.array([[-0.5, 0.0], [0.5, 0.0], [0.5, 0.5], [-0.5, 0.5]])
    pts, _truth = arch_points("semicircular")  # span 2, springing 1.5: half as wide, the plan's width (an ellipse)
    fit = engine.fit_arch(np.column_stack([pts[:, 0] * 0.5, pts[:, 1]]), "elliptical")
    void = engine.arch_void_outline(fit, below=0.001)
    # the arch's plane: u = x, v = up, its normal -y; the void extruded from y = 0.5 to y = 0 (along the normal)
    placement = engine.arch_frame([0.0, 0.5, 0.0], [1, 0, 0], [0, 0, 1])
    arched = _run(tmp_path / "a", "opening", {"plan": rect.tolist(), "bottom": 1.0, "top": 1.5,
                                              "void": {"outline": void, "placement": placement.tolist(), "depth": 0.5}})
    area = abs(engine.polygon_area(engine.outline_polygon(void, arc_points=4000)))
    # the void reaches 1 mm into the part below: inside it, the union's volume is that of the two parts
    expected = engine.opening_volume(rect, 1.0, 1.5, area - 0.001, 0.5)
    assert arched["valid"] and abs(arched["volume"] - expected) < 1e-4
    assert arched["bbox"][5] == pytest.approx(2.5, abs=0.002)  # the crown: semi-axes 0.5 across and 1 up
    stl = _stl(arched)
    assert stl.is_watertight and len(stl.split(only_watertight=False)) == 1
