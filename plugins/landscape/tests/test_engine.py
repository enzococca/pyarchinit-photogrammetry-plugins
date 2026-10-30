"""The simulation itself (``engine.py`` and ``worker.py`` loaded by their path, without the program):
the pure helpers run everywhere; the simulations need Landlab in this Python and are skipped where
it is not (the program's own venv) - run them with a Python that has Landlab::

    <env>/bin/python -m pytest plugins/landscape/tests -q
"""
from __future__ import annotations

import importlib.util
import json
import math

import numpy as np
import pytest

from landscape_samples import CELL, load, notched_slope, tell

engine = load("engine")
HAS_LANDLAB = importlib.util.find_spec("landlab") is not None
landlab = pytest.mark.skipif(not HAS_LANDLAB, reason="Landlab is not installed for this Python")
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning"),
              pytest.mark.filterwarnings("ignore::DeprecationWarning")]


# ---------------------------------------------------------------------- without Landlab
def test_default_cell_and_limits():
    assert engine.default_cell((400, 300), 0.05) == 0.05
    assert engine.default_cell((1200, 800), 0.05) == pytest.approx(0.15)  # 1200 / 3 = 400 cells
    assert engine.default_cell((501, 10), 1.0) == 2.0
    assert engine.diffusion_dt_limit(0.5, 0.01) == pytest.approx(0.14 * 0.25 / 0.01)
    assert engine.diffusion_dt_limit(0.5, 0.0) == math.inf
    # n = 1: the celerity K Q^m does not depend on the slope
    assert engine.fluvial_dt_limit(1.0, 1e-3, [100.0, 400.0], [0.1, 0.2], 0.5, 1.0) == pytest.approx(0.3 / (1e-3 * 20))
    assert engine.fluvial_dt_limit(1.0, 1e-3, [0.0], [0.1], 0.5, 1.0) == math.inf


def test_targets_land_on_the_snapshots():
    assert engine.plan_targets(1000, [10, 100, 1000, 2000, 0, -5]) == [10.0, 100.0, 1000.0]
    assert engine.plan_targets(50) == [50.0]
    with pytest.raises(ValueError):
        engine.plan_targets(0)
    assert engine.snapshot_name(10) == "snap_10.npy" and engine.snapshot_name(2.5) == "snap_2.5.npy"


def test_the_edge_of_the_data():
    valid = np.ones((6, 7), bool)
    valid[0, :3] = False  # nodata reaching the edge
    valid[3, 3] = False  # a hole inside
    edge = engine.perimeter(valid)
    assert edge[1, :3].all() and edge[0, 3] and edge[-1].all() and edge[:, -1].all()
    assert not edge[2, 3] and not edge[3, 2] and not edge[3, 4]  # beside the hole: not an edge
    assert not edge[~valid].any()
    raster = np.arange(12.0).reshape(3, 4)
    assert engine.to_nodes(raster)[0] == raster[-1, 0]  # Landlab's node 0 is the south-west corner
    assert np.array_equal(engine.to_raster(engine.to_nodes(raster), raster.shape), raster)


def test_processes_and_their_defaults():
    diffusion, fluvial = engine.merged({"diffusion": {"D": 0.02}, "fluvial": {"K_sed": 0.005}})
    assert diffusion == {"D": 0.02}
    assert fluvial["K_sed"] == 0.005 and fluvial["K_br"] == engine.DEFAULTS["fluvial"]["K_br"]
    assert engine.merged({"diffusion": {"D": 0.0}, "fluvial": {}})[0] is None
    with pytest.raises(ValueError, match="no process"):
        engine.merged({"diffusion": None, "fluvial": None})
    for preset in engine.PRESETS.values():
        assert set(preset["fluvial"]) == set(engine.DEFAULTS["fluvial"]) and preset["diffusion"]["D"] > 0


def test_the_estimate_grows_with_the_years_and_the_cells():
    params = {"diffusion": {"D": 0.01}, "fluvial": {}}
    by_years = [engine.estimate((200, 200), 0.1, y, params)["seconds"] for y in (10, 100, 1000)]
    by_cells = [engine.estimate((n, n), 0.1, 100, params)["seconds"] for n in (100, 200, 400)]
    assert by_years == sorted(by_years) and by_years[0] < by_years[-1]
    assert by_cells == sorted(by_cells) and by_cells[0] < by_cells[-1]
    small = engine.estimate((100, 100), 0.5, 100, params)
    assert small["warning"] == "" and not small["refuse"] and small["diffusion_substeps"] >= small["steps"]
    # the "mudbrick tell" on a fine grid: long, said, refused above two hours
    tell_preset = dict(engine.PRESETS["mudbrick_tell"])
    big = engine.estimate((500, 500), 0.05, 1000, tell_preset)
    assert big["diffusion_substeps"] == math.ceil(1000 / engine.diffusion_dt_limit(0.05, 0.01) - 1e-9)
    assert "circa" in big["warning"]
    huge = engine.estimate((500, 500), 0.02, 1000, tell_preset, lang="en")
    assert huge["refuse"] and "hours" in huge["warning"]
    only_creep = engine.estimate((100, 100), 0.5, 100, {"diffusion": {"D": 0.01}, "fluvial": None})
    assert only_creep["fluvial_steps"] == 0 and only_creep["steps"] == engine.STEPS


def test_bad_steps_and_routers_are_refused():
    z, _r = tell(n=10)
    for dt in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="dt must be"):
            engine.run({"cell": CELL, "years": 10, "dt": dt, "diffusion": {"D": 0.01}, "fluvial": None}, z)
    with pytest.raises(ValueError, match="router"):
        engine.run({"cell": CELL, "years": 10, "router": "d16", "diffusion": {"D": 0.01}, "fluvial": None}, z)


# ---------------------------------------------------------------------- with Landlab
@landlab
def test_diffusion_lowers_the_summit_and_raises_the_foot():
    z, r = tell()
    res = engine.run({"cell": CELL, "years": 200, "boundary": "closed", "diffusion": {"D": 0.01}, "fluvial": None}, z)
    summit = r == r.min()
    assert (res.dz[summit] < -0.5).all()  # the point of the cone goes down
    foot = (r > 10.0) & (r < 11.5)
    assert (res.dz[foot] > 0.01).all()  # just outside the old base, the colluvium
    far = r > 18
    assert np.abs(res.dz[far]).max() < 1e-4  # next to nothing reaches the plane far away in 200 years
    assert res.balance["eroded_m3"] > 1 and res.balance["router"] == "priority_flood"


@landlab
def test_the_volume_is_kept_with_closed_boundaries():
    z, _r = tell()
    cone = (z - 10.0).sum() * CELL * CELL
    res = engine.run({"cell": CELL, "years": 2000, "boundary": "closed", "diffusion": {"D": 0.02}, "fluvial": None}, z)
    b = res.balance
    assert abs(b["net_m3"]) / cone < 0.01
    assert b["eroded_m3"] / cone > 0.2  # it moved a lot, and kept it
    assert b["outlet_node"] is None and abs(b["diffusion_out_m3"]) / cone < 0.01
    assert abs(b["closure_percent"]) < 1 and b["eroded_solid_m3"] == b["eroded_m3"]  # creep alone: solid = bulk


@landlab
def test_running_water_incises_the_notch_and_deposits_downstream():
    z, notch, north, cols = notched_slope()
    res = engine.run({"cell": CELL, "years": 100, "boundary": "open", "diffusion": None,
                      "fluvial": {"K_sed": 0.01, "K_br": 0.005, "runoff": 1.0, "soil_depth": 1.0}}, z)
    dz = res.dz
    band = (north > 20) & (north < 45)
    in_notch = dz[band & notch]
    beside = dz[band & (np.abs(cols - 50) > 8)]
    assert (in_notch < 0).all() and in_notch.mean() < beside.mean() - 0.05  # the notch deepens more than the slope
    apron = (north > 1) & (north < 14) & (np.abs(cols - 50) < 6)
    assert dz[apron].max() > 0.03  # where the slope flattens the sediment stays
    b = res.balance
    assert b["eroder"] == "space" and b["deposited_m3"] > 1 and b["fluvial_out_m3"] > 0
    assert b["smallest_step"] < 1.0  # the CFL limit of the incision shortened the steps


FLUVIAL = {"K_sed": 0.01, "K_br": 0.005, "runoff": 1.0, "soil_depth": 1.0}


@landlab
@pytest.mark.parametrize("boundary", ["open", "closed"])
@pytest.mark.parametrize("phi", [0.0, 0.3])
@pytest.mark.parametrize("F_f", [0.0, 0.5])
def test_the_balance_closes_on_the_solid_volumes(F_f, phi, boundary):
    z, *_ = notched_slope(n=60)
    res = engine.run({"cell": CELL, "years": 50, "boundary": boundary, "diffusion": None,
                      "fluvial": dict(FLUVIAL, F_f=F_f, phi=phi)}, z)
    b = res.balance
    out = b["bedload_out_m3"] + b["wash_load_m3"]
    assert b["fluvial_out_m3"] == pytest.approx(out) and out > 0
    assert abs(b["eroded_solid_m3"] - b["deposited_solid_m3"] - out) < 0.01 * b["eroded_solid_m3"]
    assert abs(b["closure_percent"]) < 1 and b["closure_m3"] == pytest.approx(
        b["eroded_solid_m3"] - b["deposited_solid_m3"] - out)
    assert (b["wash_load_m3"] > 0) == (F_f > 0)
    if phi > 0:  # the soil is porous: what the map shows (bulk) is more than the solid sediment
        assert b["eroded_m3"] > b["eroded_solid_m3"]
    else:
        assert b["eroded_m3"] == pytest.approx(b["eroded_solid_m3"])


@landlab
def test_the_wash_load_does_not_depend_on_the_step():
    z, *_ = notched_slope(n=60)
    job = {"cell": CELL, "years": 20, "boundary": "open", "diffusion": None, "fluvial": dict(FLUVIAL, F_f=0.5)}
    coarse, fine = engine.run(job, z).balance, engine.run(dict(job, dt=0.1), z).balance
    for b in (coarse, fine):
        assert abs(b["closure_percent"]) < 1
    assert fine["wash_load_m3"] == pytest.approx(coarse["wash_load_m3"], rel=0.1)


@landlab
def test_with_creep_too_the_balance_closes():
    z, *_ = notched_slope(n=60)
    res = engine.run({"cell": CELL, "years": 50, "boundary": "open", "diffusion": {"D": 0.01},
                      "fluvial": dict(FLUVIAL, F_f=0.5, phi=0.3)}, z)
    assert abs(res.balance["closure_percent"]) < 1 and res.balance["diffusion_out_solid_m3"] is not None


@landlab
def test_a_closed_boundary_with_running_water_has_one_outlet():
    z, notch, north, cols = notched_slope(n=60)
    res = engine.run({"cell": CELL, "years": 50, "boundary": "closed", "diffusion": {"D": 0.01},
                      "fluvial": dict(FLUVIAL, F_f=0.5)}, z)
    b = res.balance
    outlet = b["outlet_node"]
    rows, cols_n = z.shape
    r, c = divmod(outlet, cols_n)  # Landlab's node: row from the south
    assert r == 0 and 0 < c < cols_n - 1  # on the lowest (south) edge, not a corner
    edge = np.zeros(z.shape, bool)
    edge[0], edge[-1], edge[:, 0], edge[:, -1] = True, True, True, True
    assert (res.dz[edge] == 0).all()  # closed edge cells (and the fixed outlet) do not move
    assert b["fluvial_out_m3"] > 0 and abs(b["closure_percent"]) < 1  # the sediment leaves through the outlet
    assert (res.dz[(north > 10) & notch] < 0).any()


@landlab
@pytest.mark.parametrize("missing, eroder, router", [
    (("SpaceLargeScaleEroder",), "space_classic", "priority_flood"),
    (("SpaceLargeScaleEroder", "Space"), "erosion_deposition", "priority_flood"),
    (("PriorityFloodFlowRouter",), "space", "depression_finder"),
])
def test_the_fallbacks_incise_the_notch(monkeypatch, missing, eroder, router):
    import landlab.components as lc

    for name in missing:
        monkeypatch.delattr(lc, name)
    z, notch, north, cols = notched_slope(n=60)
    res = engine.run({"cell": CELL, "years": 50, "boundary": "open", "diffusion": None, "fluvial": FLUVIAL}, z)
    b = res.balance
    assert b["eroder"] == eroder and b["router"] == router
    band = (north > 17) & (north < 28)  # the slope (the apron, north < 15, takes the sediment)
    in_notch, beside = res.dz[band & notch], res.dz[band & (np.abs(cols - 30) > 6)]
    assert (in_notch < 0).all() and in_notch.mean() < beside.mean()
    if eroder != "space":
        assert b["closure_m3"] is None and b["fluvial_out_m3"] is None  # they do not say what leaves


@landlab
def test_the_bar_moves_inside_a_long_step(monkeypatch):
    clock = iter(range(10_000_000))
    monkeypatch.setattr(engine.time, "monotonic", lambda: float(next(clock)))
    z, _r = tell(n=30)
    seen = []
    engine.run({"cell": CELL, "years": 100, "dt": 100.0, "diffusion": {"D": 0.05}, "fluvial": None}, z,
               progress=lambda f, phase, **info: seen.append((f, phase, info.get("year"))))
    years = [y for _f, phase, y in seen if phase == "run"]
    assert len(years) > 10 and years == sorted(years) and years[-1] == 100  # one step, many sub-steps, many reports
    fractions = [f for f, _p, _y in seen]
    assert fractions == sorted(fractions)


@landlab
def test_snapshots_at_the_times_asked():
    z, _r = tell(n=40)
    job = {"cell": CELL, "years": 30, "dt": 1.0, "snapshots": [5, 20, 30, 50], "boundary": "closed",
           "diffusion": {"D": 0.01}, "fluvial": None}
    res = engine.run(job, z)
    assert sorted(res.snapshots) == [5.0, 20.0, 30.0]
    assert np.array_equal(res.snapshots[30.0], res.final)
    short = engine.run(dict(job, years=5, snapshots=[]), z)
    assert np.array_equal(res.snapshots[5.0], short.final)  # the same steps up to year 5
    assert res.balance["steps"] == 30


@landlab
def test_a_step_too_long_for_the_diffusion_is_split():
    # LinearDiffuser would split it by itself as well (at 0.15 dx^2 / D): what this checks is the engine's own
    # split - the count it reports, and a stable result (the soil bookkeeping and the bar follow its sub-steps)
    z, _r = tell(n=40)
    D = 0.05
    res = engine.run({"cell": CELL, "years": 100, "dt": 50.0, "boundary": "closed", "diffusion": {"D": D},
                      "fluvial": None}, z)
    b = res.balance
    limit = engine.diffusion_dt_limit(CELL, D)
    assert b["steps"] == 2 and b["diffusion_substeps"] == 2 * math.ceil(50.0 / limit)
    assert np.isfinite(res.final).all()
    assert res.final.max() <= z.max() + 1e-9 and res.final.min() >= z.min() - 1e-9  # no overshoot: stable


@landlab
def test_nodata_stays_nodata_and_the_edge_is_open():
    z, _r = tell(n=40)
    z[:5, :5] = np.nan
    res = engine.run({"cell": CELL, "years": 50, "boundary": "open", "diffusion": {"D": 0.01},
                      "fluvial": {"runoff": 0.5}}, z)
    for array in (res.final, res.dz, res.area):
        assert np.isnan(array[:5, :5]).all() and np.isfinite(array[5:, 5:]).all()
    assert res.dz[5, 2] == 0.0 and res.dz[2, 5] == 0.0  # beside the nodata reaching the edge: fixed values
    assert res.dz[6, 6] != 0.0
    assert np.nanmax(res.area) > 20 * CELL * CELL  # the water gathers


@landlab
def test_the_worker_writes_the_files_and_dz_is_final_minus_initial(tmp_path, capsys):
    worker = load("worker")
    z, _r = tell(n=40)
    np.save(tmp_path / "elevation.npy", z)
    out = tmp_path / "run"
    job = {"protocol": 1, "elevation": str(tmp_path / "elevation.npy"), "output": str(out), "cell": CELL,
           "years": 20, "snapshots": [10], "boundary": "open", "diffusion": {"D": 0.01}, "fluvial": {}}
    (tmp_path / "job.json").write_text(json.dumps(job))
    assert worker.main([str(tmp_path / "job.json")]) == 0
    said = [json.loads(line[len(worker.PREFIX):]) for line in capsys.readouterr().out.splitlines()
            if line.startswith(worker.PREFIX)]
    assert said[0]["type"] == "hello" and said[0]["versions"]["landlab"]
    assert said[-1]["type"] == "result" and any(m["type"] == "progress" and m["phase"] == "run" for m in said)
    fractions = [m["fraction"] for m in said if m["type"] == "progress"]
    assert fractions == sorted(fractions) and fractions[-1] < 1.0
    assert sorted(p.name for p in out.iterdir()) == ["area.npy", "balance.json", "dz.npy", "final.npy", "initial.npy",
                                                     "snap_10.npy"]
    final, dz, initial = (np.load(out / f"{k}.npy") for k in ("final", "dz", "initial"))
    assert np.array_equal(dz, final - z) and np.array_equal(initial, z)
    balance = json.loads((out / "balance.json").read_text())
    assert balance["eroder"] == "space" and balance["years"] == 20
    assert not out.with_name("run.partial").exists()
    # the output exists: refused, nothing touched
    assert worker.main([str(tmp_path / "job.json")]) == 1
    assert "exists already" in capsys.readouterr().out and np.array_equal(np.load(out / "final.npy"), final)
