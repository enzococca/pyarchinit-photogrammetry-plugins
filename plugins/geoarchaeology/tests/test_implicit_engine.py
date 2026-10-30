"""The implicit model against synthetic truths (``implicit_engine.py`` loaded by its path, without the
program). The pure helpers run everywhere; the models need LoopStructural in this Python and are
skipped where it is not (the program's own venv) - run them with a Python that has it::

    <env>/bin/python -m pytest plugins/geoarchaeology/tests/test_implicit_engine.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from implicit_samples import Ground, bowl, circle, has_loopstructural, layers_job, load, polygon_mask, stub_pyvista

engine = load("implicit_engine")
loop = pytest.mark.skipif(not has_loopstructural(), reason="LoopStructural is not installed for this Python")
pytestmark = [pytest.mark.filterwarnings("ignore::RuntimeWarning"),
              pytest.mark.filterwarnings("ignore::DeprecationWarning"),
              pytest.mark.filterwarnings("ignore::FutureWarning")]
if has_loopstructural():
    stub_pyvista()
    import logging

    logging.disable(logging.WARNING)  # the library's messages: it says a lot


# ---------------------------------------------------------------------- without LoopStructural
def test_the_vertical_scale_gives_the_thinnest_unit_three_elements():
    extent = (300.0, 300.0, 30.0)
    k = engine.vertical_scale(2.0, extent, 20000)
    h = (extent[0] * extent[1] * extent[2] * k / 20000) ** (1 / 3)
    assert 2.0 * k == pytest.approx(3 * h)
    assert engine.vertical_scale(5.0, (10, 10, 10), 20000) == 1.0  # clamped from below
    assert engine.vertical_scale(0.001, extent, 20000) == 40.0  # and from above
    assert engine.vertical_scale(None, extent) == 1.0


def test_normals_go_by_the_inverse_transpose():
    dip = math.radians(30)
    n = np.array([math.sin(dip), 0.0, math.cos(dip)])
    k = 8.0
    scaled = engine.scaled_normal(n, k)[0]
    tangent = np.array([math.cos(dip), 0.0, -math.sin(dip)]) * [1, 1, k]  # a line of the plane, scaled like points
    assert float(scaled @ tangent) == pytest.approx(0.0, abs=1e-12)
    assert math.degrees(math.atan2(scaled[0], scaled[2])) == pytest.approx(math.degrees(math.atan(k * math.tan(dip))))
    # on a slope: a plane parallel to the ground is horizontal in depth coordinates
    slope = (0.2, -0.1)
    parallel = np.array([-slope[0], -slope[1], 1.0])
    assert np.allclose(engine.scaled_normal(parallel, 3.0, slope)[0], [0, 0, 1])


def test_the_plane_of_a_contact():
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 50, (12, 2))
    pts = np.column_stack([xy, 3 + 0.2 * xy[:, 0] - 0.1 * xy[:, 1]])
    n = engine.plane_normal(pts)
    assert n[2] > 0 and np.allclose(n / n[2], [-0.2, 0.1, 1.0])
    plane, is_plane = engine.fit_plane(pts)
    assert is_plane and np.allclose(plane, [0.2, -0.1, 3.0])
    assert np.allclose(engine.plane_normal(pts[:1]), [0, 0, 1])  # one position: the vertical
    line = np.column_stack([np.arange(6.0), 2 * np.arange(6.0), np.arange(6.0)])
    plane, is_plane = engine.fit_plane(line)
    n = engine.plane_normal(line)
    assert not is_plane and np.allclose(plane, [0, 0, 2.5]) and np.allclose(n, [0, 0, 1])  # along a line: horizontal
    flat = np.column_stack([np.arange(6.0), np.zeros(6), np.full(6, 2.0)])
    assert np.allclose(engine.plane_normal(flat), [0, 0, 1])


def test_the_values_of_contacts_come_from_the_holes_that_see_two():
    # the mean of B is above that of A only because B is seen updip
    x = np.array([0.0, 0.0, 10.0, 10.0, 100.0])
    z = np.array([50.0, 48.0, 49.0, 47.0, 40.0])
    unit = np.array(["A", "B", "A", "B", "A"])
    hole = np.array(["1", "1", "2", "2", "3"])
    planes = {"A": np.array([-0.1, 0.0, 50.0]), "B": np.array([-0.1, 0.0, 48.0])}
    values, apart = engine.contact_values(["A", "B"], unit, z, hole, planes, {"A": (30, 0), "B": (5, 0)})
    assert values["A"] - values["B"] == pytest.approx(2.0) and apart == []
    # a contact that shares no hole keeps its height above the plane of the main contact, at its own place
    unit = np.array(["A", "B", "A", "C", "A"])
    hole = np.array(["1", "1", "2", "4", "3"])
    z = np.array([50.0, 48.0, 49.0, 45.0, 40.0])
    centres = {"A": (30, 0), "B": (0, 0), "C": (10, 0)}
    values, apart = engine.contact_values(["A", "B", "C"], unit, z, hole, planes, centres)
    assert apart == ["C"]
    assert values["A"] - values["C"] == pytest.approx(4.0)  # the plane of A is 49 at x = 10


def test_the_thin_unit_under_the_ground_counts():
    reader = engine.Dem(engine.Grid(Ground(10, 1.0).grid), np.full((10, 10), 10.0))
    column = [("T", "layer"), ("A", "layer"), ("B", "layer")]
    contacts = [("A", 5.0, 5.0, 9.8, 1.0, "S1"), ("B", 5.0, 5.0, 6.0, 1.0, "S1")]
    assert engine.thinnest_unit(contacts, column) == pytest.approx(3.8)
    assert engine.thinnest_unit(contacts, column, reader) == pytest.approx(0.2)


def test_a_hole_against_the_column_is_counted():
    column = [("A", "layer"), ("B", "layer")]
    rows = [("A", 0.0, 0.0, 5.0, 1.0, "1"), ("B", 0.0, 0.0, 3.0, 1.0, "1"),
            ("A", 9.0, 0.0, 3.0, 1.0, "2"), ("B", 9.0, 0.0, 5.0, 1.0, "2")]
    assert engine.column_contradictions(rows, column) == 1


def test_the_dem_is_read_between_its_cells_and_beyond_its_mask():
    ground = Ground(10, 2.0)
    dem = 5 + 0.5 * ground.X - 0.25 * ground.Y
    reader = engine.Dem(engine.Grid(ground.grid), dem)
    assert float(reader.at(np.array([7.3]), np.array([11.1]))[0]) == pytest.approx(5 + 3.65 - 2.775)
    dzdx, dzdy = reader.slope(np.array([9.0]), np.array([9.0]))
    assert float(dzdx[0]) == pytest.approx(0.5) and float(dzdy[0]) == pytest.approx(-0.25)
    dem[:3, :3] = np.nan
    reader = engine.Dem(engine.Grid(ground.grid), dem)
    assert np.isfinite(reader.at(np.array([1.0]), np.array([19.0]))[0])


def test_the_column_is_checked():
    good = {"column": [{"unit": "F", "kind": "fill"}, {"unit": "C", "kind": "cut"}, {"unit": "A"}], "fills": {"F": "C"}}
    assert engine.check_column(good) == [("F", "fill"), ("C", "cut"), ("A", "layer")]
    with pytest.raises(ValueError, match="under its cut"):
        engine.check_column({"column": [{"unit": "A"}, {"unit": "C", "kind": "cut"}, {"unit": "F", "kind": "fill"}],
                             "fills": {"F": "C"}})
    with pytest.raises(ValueError, match="top of the column"):
        engine.check_column({"column": [{"unit": "C", "kind": "cut"}, {"unit": "A"}]})
    with pytest.raises(ValueError, match="no cut"):
        engine.check_column({"column": [{"unit": "F", "kind": "fill"}, {"unit": "A"}], "fills": {}})


def test_groups_of_boreholes_are_shuffled_with_a_seed():
    holes = [f"S{i}" for i in range(30)]
    groups = engine.fold_groups(holes, 12, seed=4)
    assert len(groups) == 12 and sorted(sum(groups, [])) == sorted(holes)
    assert max(map(len, groups)) - min(map(len, groups)) <= 1
    assert groups == engine.fold_groups(list(reversed(holes)), 12, seed=4)
    assert groups != engine.fold_groups(holes, 12, seed=5)
    assert len(engine.fold_groups(holes[:5], 12)) == 5 and engine.fold_groups(holes[:1], 12) == []


def test_the_jackknife_of_a_raster():
    stack = [np.full((2, 2), v, dtype=float) for v in (1.0, 2.0, 3.0, 4.0)]
    expected = 3 / 4 * sum((v - 2.5) ** 2 for v in (1.0, 2.0, 3.0, 4.0))
    assert np.allclose(engine.jackknife(stack), expected)
    stack[0][0, 0] = np.nan
    stack[1][0, 0] = np.nan
    stack[2][0, 0] = np.nan
    assert np.isnan(engine.jackknife(stack)[0, 0])


def test_a_plan_nobody_drew_is_estimated():
    ground = Ground(40, 1.0)
    grid = engine.Grid(ground.grid)
    inside = np.array([[20.0, 20.0], [26.0, 20.0]])
    outside = np.array([[14.0, 20.0], [32.0, 20.0]])
    mask, limit = engine.estimated_plan(inside, outside, grid, 2.0)
    assert limit == pytest.approx(6.0)  # the spacing of the holes inside
    near = np.minimum(np.hypot(ground.X - 20, ground.Y - 20), np.hypot(ground.X - 26, ground.Y - 20))
    assert near[mask].max() <= 6.0
    assert np.all(ground.X[mask] > 17) and np.all(ground.X[mask] < 29)  # nearer to the holes inside
    mask, limit = engine.estimated_plan(inside, np.zeros((0, 2)), grid, 2.0)
    assert limit == pytest.approx(6.0) and mask[ground.Y == 19.5][20]  # no hole outside: discs
    assert near[mask].max() <= 6.0 and near[~mask].min() > 5.0
    mask, limit = engine.estimated_plan(inside[:1], np.zeros((0, 2)), grid, 2.5)
    assert limit == 2.5 and np.hypot(ground.X - 20, ground.Y - 20)[mask].max() <= 2.5  # a single hole
    assert engine.estimated_plan(inside[:1], outside, grid, 2.5)[1] == pytest.approx(6.0)


def test_the_profiles():
    t = np.array([0.0, 0.125, 0.25, 0.5, 1.0])
    assert np.allclose(engine.PROFILES["flat"](t), [0.0, 0.75, 1.0, 1.0, 1.0])
    assert np.allclose(engine.PROFILES["bowl"](t), [0.0, 1 - 0.875 ** 3, 1 - 0.75 ** 3, 0.875, 1.0])
    assert np.allclose(engine.PROFILES["v"](t), t)


def _bar(rows: int, cols: int, top: int, bottom: int) -> np.ndarray:
    mask = np.zeros((rows, cols), dtype=bool)
    mask[top:bottom, 4:-4] = True
    return mask


@pytest.mark.parametrize("cells", [4, 5])
def test_the_half_width_of_a_ditch_with_an_even_or_odd_number_of_cells(cells):
    distance, width = engine.edge_and_width(_bar(20, 40, 8, 8 + cells), 0.5)
    middle = (2 * 8 + cells, 2 * 20 + 1)
    assert distance[middle] == pytest.approx(cells * 0.25)  # the axis, between two rows of cells when even
    assert width[middle] == pytest.approx(cells * 0.25)
    assert width[2 * 8 + 1, 2 * 20 + 1] == pytest.approx(cells * 0.25)  # at the border cells too
    assert distance[2 * 8 + 1, 2 * 20 + 1] == pytest.approx(0.25)  # half a cell: never 0


def test_an_elbow_does_not_set_the_width_of_the_ditch():
    mask = np.zeros((60, 60), dtype=bool)
    mask[50:54, 10:50] = True
    mask[10:54, 10:14] = True
    plan = engine.Plan.of(mask, 0.5, "drawn")
    _distance, width = engine.edge_and_width(mask, 0.5)
    assert width[2 * 52, 2 * 12] > 1.03  # the elbow is wider
    assert width[2 * 52, 2 * 40] == pytest.approx(1.0) and width[2 * 30, 2 * 12] == pytest.approx(1.0)
    assert plan.cell_t[52, 40] == pytest.approx(0.75)


def test_a_hand_drawn_outline_keeps_its_width():
    ground = Ground(160, 0.25)
    rng = np.random.default_rng(1)
    angle = 2 * np.pi * np.arange(120) / 120
    radius = 8 + 0.3 * rng.standard_normal(120)
    outline = np.column_stack([20 + radius * np.cos(angle), 20 + radius * np.sin(angle)])
    mask = polygon_mask(ground, outline)
    t = engine.Plan.of(mask, 0.25, "drawn").cell_t[mask]
    truth = np.clip(1 - np.hypot(ground.X - 20, ground.Y - 20)[mask] / 8, 0, 1)
    bowl_profile = engine.PROFILES["bowl"]
    # its small bumps are ridges too: with the nearest ridge this was 1.6, and 2.6 for the v
    assert float(bowl_profile(t).mean() / bowl_profile(truth).mean()) == pytest.approx(1.0, abs=0.05)
    assert float(t.mean() / truth.mean()) == pytest.approx(1.0, abs=0.08)


def test_the_border_of_the_raster_is_not_the_edge_of_a_plan():
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, :6] = True
    distance, _width = engine.edge_and_width(mask, 1.0)
    assert distance[2 * 10, 1] == pytest.approx(5.0)  # from the sides, not from the west border of the raster
    distance, width = engine.edge_and_width(np.ones((4, 4), dtype=bool), 1.0)
    assert np.all(distance / width == 1.0)


def _floor_on(mask: np.ndarray, base: np.ndarray, ground: Ground, level: float = 10.0):
    grid = engine.Grid(ground.grid)
    dem = engine.Dem(grid, np.full(mask.shape, level))
    plan = engine.Plan.of(mask, grid.cell, "drawn")
    table = engine.base_rows(base, plan, grid, dem)
    depths, held = engine.floors(plan, table, grid, dem, base)
    return plan, table, depths, held


def test_a_floor_passes_through_its_measures():
    ground = Ground(40, 1.0)
    r = np.hypot(ground.X - 20, ground.Y - 20)
    mask = r < 12
    base = np.array([[20.5, 20.5, 7.0], [25.5, 20.5, 8.0], [14.5, 22.5, 8.8]])
    _plan, table, depths, held = _floor_on(mask, base, ground)
    for profile in engine.PROFILES:
        assert not held[profile].any()
        assert np.allclose(depths[profile][table.rows, table.cols], 10.0 - base[:, 2])
        assert depths[profile][~mask].max() == 0 and depths[profile][mask].min() > 0


def test_a_hole_by_the_edge_is_held_by_the_limits():
    ground = Ground(40, 1.0)
    mask = np.hypot(ground.X - 20, ground.Y - 20) < 12
    base = np.array([[31.5, 20.5, 9.5]])  # half a metre deep beside the edge
    _plan, table, depths, held = _floor_on(mask, base, ground)
    assert held["bowl"].all() and held["v"].all()
    # P is under P_MIN here: the floor in the middle is at most four times the depth measured beside the edge
    assert depths["bowl"].max() == pytest.approx(2.0, abs=1e-3)
    for profile in engine.PROFILES:
        assert depths[profile].max() <= 2.0 + 1e-6


def test_p_min_acts_alone_when_a_deep_hole_is_elsewhere():
    ground = Ground(40, 1.0)
    mask = np.hypot(ground.X - 20, ground.Y - 20) < 12
    base = np.array([[31.5, 20.5, 9.0], [20.5, 20.5, 6.0]])
    plan, table, depths, held = _floor_on(mask, base, ground)
    assert list(held["v"]) == [True, False]
    wanted = table.depth[0] / engine.P_MIN
    row, col = table.rows[0], table.cols[0]
    assert depths["v"][row, col] == pytest.approx(wanted * plan.cell_t[row, col], rel=0.1)


def test_the_wall_of_a_plan_is_on_the_edge_of_its_cells():
    ground = Ground(10, 1.0)
    frame = engine.Frame({"grid": ground.grid, "zmin": 0.0, "zmax": 12.0, "dz": 1.0}, np.full((10, 10), 10.0))
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    plan = engine.Plan.of(mask, 1.0, "drawn")
    cube = frame.volume(engine.plan_function(frame, plan, np.where(mask, 5.0, 0.0)))
    # half a cell on both sides of the wall, so a solid has it there and not on a slant to the next column
    assert cube[4, 3, 8] == pytest.approx(0.5) and cube[4, 2, 8] == pytest.approx(-0.5)
    assert cube[4, 4, 3] < 0  # under the floor


class _Positive:
    def at(self, x, _y, _z):
        return np.ones(len(x))


def test_a_voxel_under_the_box_of_a_field_is_never_inside():
    ground = Ground(10, 1.0)
    frame = engine.Frame({"grid": ground.grid, "zmin": 0.0, "zmax": 12.0, "dz": 1.0}, np.full((10, 10), 10.0))
    box = engine.Box([0.0, 0.0, 5.0], [10.0, 10.0, 20.0], 1.0, 1000)
    function = engine.wall_function(frame, _Positive(), box, np.array([[5.5, 5.5]]), frame.points())
    cube = frame.volume(function)
    assert cube[:, :, :5].max() < 0 and cube[:, :, 5:9].min() > 0


def test_a_row_past_the_outline_counts_at_the_nearest_cell():
    ground = Ground(40, 1.0)
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 5:35] = True
    base = np.array([[10.5, 20.5, 8.0], [20.5, 25.5, 8.0], [30.5, 20.5, 8.0]])
    plan, table, depths, _held = _floor_on(mask, base, ground)
    assert table.past[1] == pytest.approx(3.5) and table.past[0] == 0.0 and table.past[2] == 0.0
    assert (table.rows[1], table.cols[1]) == (18, 20)
    assert table.y[1] == pytest.approx(ground.ys[18])
    assert np.all(depths["flat"][mask] == pytest.approx(2.0))  # it changes nothing: same depth, flat floor


class _Linear:
    def evaluate_value(self, points):
        return points[:, 2] * 2.0


def test_a_field_goes_on_past_its_box():
    reader = engine.Dem(engine.Grid(Ground(4, 1.0).grid), np.zeros((4, 4)))
    fitted = engine.Fitted(_Linear(), "elevation", 1.0, reader, 0.0, 10.0, gain=2.0)
    values = fitted.evaluate(np.full(3, 1.0), np.full(3, 1.0), np.array([-5.0, 5.0, 15.0]))
    assert np.allclose(values, [-10.0, 10.0, 30.0])
    fitted.gain = 0.0
    assert np.allclose(fitted.evaluate(np.full(1, 1.0), np.full(1, 1.0), np.array([-5.0])), [0.0])


def test_too_many_voxels_are_refused_before_any_work():
    ground = Ground(1000, 1.0)
    job = {"grid": ground.grid, "zmin": 0.0, "zmax": 100.0, "dz": 0.1}
    with pytest.raises(MemoryError, match="Unable to allocate"):
        engine.Frame(job, np.full((1000, 1000), 50.0))


# ---------------------------------------------------------------------- the truths
def _closed(faces: np.ndarray) -> bool:
    """Every edge in an even number of faces (two, or four where a solid touches itself along an edge)."""
    edges = np.sort(np.vstack([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _unique, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(np.all(counts % 2 == 0))


def _idw_loo(xy: np.ndarray, z: np.ndarray) -> float:
    """Leave-one-out error of inverse distance weighting among the points themselves."""
    errors = []
    for i in range(len(xy)):
        d = np.hypot(*(np.delete(xy, i, axis=0) - xy[i]).T)
        w = 1 / np.maximum(d, 1e-6) ** 2
        errors.append(np.sum(w * np.delete(z, i)) / np.sum(w) - z[i])
    return float(np.sqrt(np.mean(np.square(errors))))


@pytest.fixture(scope="module")
def layered():
    """Three contacts following one wavy dipping form; the middle one seen in 6 boreholes of 36."""
    ground = Ground(80, 1.0)
    dem = np.full((80, 80), 14.0)
    form = lambda x, y: 0.05 * x + 0.6 * np.sin(x / 11.0) * np.cos(y / 13.0)  # noqa: E731
    holes = ground.holes(6, 0.08, jitter=1.5)
    contacts = []
    for name, (x, y) in enumerate(holes):
        contacts.append(["A", x, y, 11.0 + form(x, y), 1.0, f"S{name}"])
        contacts.append(["C", x, y, 5.0 + form(x, y), 1.0, f"S{name}"])
    rare = holes[[3, 8, 15, 20, 26, 33]]
    rows = [(x, y, 8.0 + form(x, y)) for x, y in rare]
    contacts += [["B", x, y, z, 1.0, f"R{i}"] for i, (x, y, z) in enumerate(rows)]
    units = ["T", "A", "B", "C"]
    job = layers_job(ground, dem, contacts, units, zmin=0.0, zmax=16.0, dz=0.25, folds=6)
    job["series"][0]["units"] = ["A", "B", "C"]
    return job, dem, np.array(rows), engine.run(job, dem)


@loop
def test_a_rare_contact_takes_its_shape_from_the_layers(layered):
    _job, _dem, rare, result = layered
    rmse = result.loo["B"]["rmse"]
    assert result.loo["B"]["nan"] == 0 and result.loo["B"]["n"] == 6
    assert rmse < _idw_loo(rare[:, :2], rare[:, 2])
    assert result.loo["A"]["rmse"] < 0.5 and result.loo["C"]["rmse"] < 0.5


@loop
def test_solids_are_closed_and_units_share_the_voxels(layered):
    job, dem, _rare, result = layered
    voxel = job["grid"]["cell"] ** 2 * job["dz"]
    under = int(result.info["voxels_under_dem"])
    assert sum(int((result.labels == i).sum()) for i in range(len(result.units))) == under
    assert int((result.labels >= 0).sum()) == under
    assert sum(result.volumes.values()) == pytest.approx(under * voxel)
    for unit, (vertices, faces) in result.solids.items():
        assert _closed(faces), unit
        volume = engine.mesh_volume(vertices, faces)
        assert volume > 0
        assert volume == pytest.approx(result.volumes[unit], rel=0.02)


@loop
def test_neighbouring_solids_meet_without_a_gap(layered):
    from scipy.spatial import cKDTree

    job, _dem, _rare, result = layered
    upper, lower = result.solids["A"][0], result.solids["B"][0]
    rows, cols = (80 - lower[:, 1]).astype(int).clip(0, 79), lower[:, 0].astype(int).clip(0, 79)
    on_top = np.abs(lower[:, 2] - result.tops["B"][rows, cols]) < 0.3  # the top of B, not its bottom or sides
    inside = (lower[:, 0] > 2) & (lower[:, 0] < 78) & (lower[:, 1] > 2) & (lower[:, 1] < 78)
    distance = cKDTree(upper).query(lower[on_top & inside])[0]
    assert on_top.sum() > 1000
    assert np.quantile(distance, 0.99) < 1e-3  # the very same vertices


@loop
def test_a_series_hung_from_the_dem_follows_a_mound():
    ground = Ground(70, 1.0)
    r = np.hypot(ground.X - 35, ground.Y - 35)
    dem = 10 + 6 * np.exp(-(r / 14) ** 2)
    depth_a = lambda x, y: 1.0 + 0.004 * x  # noqa: E731
    depth_b = lambda x, y: 2.2 + 0.004 * y  # noqa: E731
    reader = engine.Dem(engine.Grid(ground.grid), dem)
    contacts = []
    holes = ground.holes(6, 0.08, jitter=1.0)
    for i, (x, y) in enumerate(holes):
        z = float(reader.at(np.array([x]), np.array([y]))[0])
        contacts.append(["A", x, y, z - depth_a(x, y), 1.0, f"S{i}"])
        contacts.append(["B", x, y, z - depth_b(x, y), 1.0, f"S{i}"])
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], spaces={"ground": ("depth", ["A", "B"])},
                     zmin=3.0, zmax=18.0, dz=0.25, folds=5)
    result = engine.run(job, dem)
    voxel = job["dz"]
    for unit, depth in (("A", depth_a), ("B", depth_b)):
        top = result.tops[unit]
        truth = dem - depth(ground.X, ground.Y)
        at_holes = [abs(float(top[int((70 - y)), int(x)] - float(reader.at(np.array([x]), np.array([y]))[0])
                              + depth(x, y))) for x, y in holes]
        assert np.median(at_holes) < 2 * voxel
        holdout = np.abs(top - truth)[::3, ::3]
        assert np.nanmedian(holdout) < 2 * voxel, unit
        assert result.loo[unit]["rmse"] < 0.5
        assert result.variance[unit].shape == dem.shape


def _pit_job(ground, dem, tops, holes_in, holes_out, base_rows, outline, **options):
    contacts = []
    for i, (x, y) in enumerate(holes_out):
        for unit, z in tops.items():
            contacts.append([unit, x, y, z, 1.0, f"S{i}"])
    units = list(tops)
    return {"grid": ground.grid, "zmin": 2.0, "zmax": 12.0, "dz": 0.2,
            "series": [{"name": "layers", "space": "elevation", "units": units[1:]}],
            "contacts": contacts, "orientations": [],
            "column": [{"unit": "F", "kind": "fill"}, {"unit": "P", "kind": "cut"}] +
                      [{"unit": u, "kind": "layer"} for u in units],
            "fills": {"F": "P"},
            "cuts": {"P": {"base": base_rows, "outline": outline, "inside": holes_in, "outside": holes_out}},
            "options": options}


@pytest.fixture(scope="module")
def pit():
    """A bowl 2.8 m deep through the layers A and B into C, filled by F, on flat ground."""
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 10.0)
    centre = (30.0, 30.0)
    radius, depth = 8.0, 2.8
    tops = {"A": 10.0, "B": 9.0, "C": 8.0}
    inside = [[30.0, 30.0], [34.0, 30.0], [26.0, 30.0], [30.0, 34.0], [30.0, 26.0], [33.0, 33.0], [27.0, 27.0],
              [33.0, 27.0], [27.0, 33.0], [36.5, 30.0], [23.5, 30.0], [30.0, 36.5], [30.0, 23.5]]
    base = [[x, y, 10.0 - float(bowl(np.hypot(x - 30, y - 30), radius, depth))] for x, y in inside]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.1) if np.hypot(x - 30, y - 30) > radius + 1]
    job = _pit_job(ground, dem, tops, inside, outside, base, polygon_mask(ground, circle(centre, radius, 10.0)))
    return ground, dem, job, engine.run(job, dem)


@loop
def test_the_fill_of_a_pit_has_its_volume(pit):
    ground, _dem, job, result = pit
    fine = np.linspace(0, 8, 8001)
    truth = float(np.sum(2 * np.pi * fine * bowl(fine, 8.0, 2.8)) * (fine[1] - fine[0]))
    assert result.volumes["F"] == pytest.approx(truth, rel=0.10)
    assert result.info["cuts"]["P"]["edge"] == "drawn"
    assert not [w for w in result.info["warnings"] if w["code"] == "edge_estimated"]


@loop
def test_a_pit_leaves_the_layers_around_it_untouched(pit):
    ground, _dem, job, result = pit
    index = {u: i for i, u in enumerate(result.units)}
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    truth = np.where(zc > 9.0, index["A"], np.where(zc > 8.0, index["B"], index["C"]))
    r = np.hypot(ground.X - 30, ground.Y - 30)
    away = r > 8.0 + 2 * job["grid"]["cell"]
    under = zc < 10.0
    # a voxel at a contact can go either way: the layers themselves are checked away from it
    clear = under & (np.abs(zc - 9.0) > 0.25) & (np.abs(zc - 8.0) > 0.25)
    assert np.all(result.labels[away][:, clear] == truth[clear])
    # below the bottom of the pit the layers go on as if it were not there
    below = zc < 10.0 - 2.8 - 2 * job["dz"]
    middle = r < 6.0
    clear = below & (np.abs(zc - 8.0) > 0.25)
    assert np.all(result.labels[middle][:, clear] == truth[clear])
    assert result.tops["F"][30, 30] == pytest.approx(10.0)
    assert result.tops["C"][30, 30] == pytest.approx(7.2, abs=0.4)  # the top of what is left: the bottom of the pit
    assert result.tops["C"][5, 5] == pytest.approx(8.0, abs=0.15)
    for unit, (_vertices, faces) in result.solids.items():
        assert _closed(faces), unit


@loop
def test_an_edge_nobody_drew_is_estimated_and_said(pit):
    ground, dem, job, _result = pit
    job = {**job, "cuts": {"P": {**job["cuts"]["P"], "outline": None}}}
    result = engine.run(job, dem)
    assert result.info["cuts"]["P"]["edge"] == "estimated"
    assert [w for w in result.info["warnings"] if w["code"] == "edge_estimated"][0]["unit"] == "P"
    assert result.volumes["F"] > 0
    assert any(w.get("effect") == "the volume may be overstated" for w in result.info["warnings"])


@loop
def test_several_fills_are_ordered_inside_their_cut(pit):
    ground, dem, job, _result = pit
    inside = job["cuts"]["P"]["inside"]
    contacts = job["contacts"] + [["F2", x, y, 8.8, 1.0, f"I{i}"] for i, (x, y) in enumerate(inside[:9])]
    column = [{"unit": "F1", "kind": "fill"}, {"unit": "F2", "kind": "fill"}] + job["column"][1:]
    job = {**job, "contacts": contacts, "column": column, "fills": {"F1": "P", "F2": "P"}}
    result = engine.run(job, dem)
    index = {u: i for i, u in enumerate(result.units)}
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    middle = result.labels[30, 30]
    assert middle[np.argmin(np.abs(zc - 9.5))] == index["F1"]
    assert middle[np.argmin(np.abs(zc - 8.0))] == index["F2"]
    assert result.tops["F2"][30, 30] == pytest.approx(8.8, abs=0.3)
    assert result.volumes["F1"] > 0 and result.volumes["F2"] > 0


@loop
def test_a_layer_can_cover_a_filled_pit(pit):
    ground, dem, job, _result = pit
    inside = job["cuts"]["P"]["inside"]
    contacts = [c for c in job["contacts"]] + [["F", x, y, 9.3, 1.0, f"I{i}"] for i, (x, y) in enumerate(inside)]
    column = [{"unit": "Y", "kind": "layer"}] + job["column"]
    series = [{"name": "layers", "space": "elevation", "units": ["A", "B", "C"]}]
    contacts += [["A", x, y, 9.6, 1.0, f"S{i}"] for i, (x, y) in enumerate(job["cuts"]["P"]["outside"])]
    contacts = [c for c in contacts if not (c[0] == "A" and c[3] == 10.0)]
    job = {**job, "contacts": contacts, "column": column, "series": series}
    result = engine.run(job, dem)
    index = {u: i for i, u in enumerate(result.units)}
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    middle = result.labels[30, 30]
    assert middle[np.argmin(np.abs(zc - 9.7))] == index["Y"]  # over the pit
    assert middle[np.argmin(np.abs(zc - 8.9))] == index["F"]
    assert result.labels[8, 8][np.argmin(np.abs(zc - 9.7))] == index["Y"]


def _solved(job, dem):
    """The names of the fields solved by a run, then its result."""
    names = []
    original = engine._solve

    def spy(name, *args):
        names.append(name)
        return original(name, *args)

    engine._solve = spy
    try:
        names.append(engine.run(job, dem))
    finally:
        engine._solve = original
    return names


def _bottle_job(ground, dem, tops, zmin=2.0):
    # a mouth of 4 m at the ground, 7 m of radius at 2 m down: a bottle pit, seen on a section of its wall
    a = 2 * np.pi * np.arange(24) / 24
    wall = [[25 + 7 * math.cos(t), 25 + 7 * math.sin(t), 8.0, "wall"] for t in a]
    shoulder = [[25 + 5.5 * math.cos(t), 25 + 5.5 * math.sin(t), 9.1, "wall"] for t in a]
    lower = [[25 + 5.5 * math.cos(t), 25 + 5.5 * math.sin(t), 6.8, "wall"] for t in a]
    sides = [[25 + 5.5 * math.cos(t), 25 + 5.5 * math.sin(t), 6.5, "wall"] for t in a[::4]]
    floor = [[25 + r * math.cos(t), 25 + r * math.sin(t), 6.5] for r in (0.0, 3.0) for t in a[::4]]
    inside = [[25.0, 25.0], [27.0, 25.0], [23.0, 25.0], [25.0, 27.0], [25.0, 23.0]]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.06) if np.hypot(x - 25, y - 25) > 10]
    outline = polygon_mask(ground, circle((25.0, 25.0), 4.0, 10.0))
    job = _pit_job(ground, dem, tops, inside, outside, wall + shoulder + floor + sides + lower, outline)
    job["dz"] = 0.2
    job["zmin"] = zmin
    return job


@loop
def test_an_undercut_wall_stays_in_the_model():
    ground = Ground(50, 1.0)
    dem = np.full((50, 50), 10.0)
    tops = {"A": 10.0, "B": 8.0}
    job = _bottle_job(ground, dem, tops)
    calls = _solved(job, dem)
    result = calls.pop()
    assert any(name.startswith("cut ") for name in calls)  # walls: the only cut with a field
    assert result.info["cuts"]["P"]["field"] is True
    index = {u: i for i, u in enumerate(result.units)}
    labels = result.labels[25, 30]  # a column under the shoulder, 5.5 m from the middle
    fill = np.nonzero(labels == index["F"])[0]
    layer = np.nonzero((labels == index["A"]) | (labels == index["B"]))[0]
    assert len(fill) and len(layer)
    assert layer.max() > fill.max()  # layer above the fill
    assert layer.min() < fill.min()  # and layer again below it
    vertices, faces = result.solids["F"]
    assert _closed(faces)
    near = np.hypot(vertices[:, 0] - 30.5, vertices[:, 1] - 24.5) < 1.0
    assert vertices[near, 2].max() < 9.8  # the solid has a roof there, under the ground
    assert np.isfinite(result.tops["F"][25, 30])  # and the raster keeps the highest crossing


@loop
def test_a_field_stays_inside_its_box():
    ground = Ground(50, 1.0)
    dem = np.full((50, 50), 10.0)
    job = _bottle_job(ground, dem, {"A": 10.0, "B": 8.0}, zmin=-30.0)
    result = engine.run(job, dem)
    index = result.units.index("F")
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    deepest = float(zc[np.nonzero((result.labels == index).any(axis=(0, 1)))[0].min()])
    assert deepest > 4.0  # about 2 m under the floor at most, nothing down to the bottom of the job


@loop
def test_a_dip_measured_on_a_section_comes_out_in_the_world():
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 30.0)
    dip = math.radians(25)
    normal = [math.sin(dip), 0.0, math.cos(dip)]  # the plane falls towards the east
    plane = lambda x: 20.0 - math.tan(dip) * (x - 30.0)  # noqa: E731
    # the points lie along the strike: they alone say nothing about the dip
    contacts = [["A", 30.0, y, plane(30.0), 1.0, f"S{i}"] for i, y in enumerate((10.0, 25.0, 40.0, 50.0))]
    orientations = [["A", 30.0, 30.0, plane(30.0), *normal]]
    job = layers_job(ground, dem, contacts, ["T", "A"], zmin=0.0, zmax=35.0, dz=0.25, vertical=6.0)
    job["series"][0]["units"] = ["A"]
    job["orientations"] = orientations
    result = engine.run(job, dem)
    top = result.tops["A"]
    east = np.array([top[30, c] for c in range(15, 46)], dtype=np.float64)
    slope = np.polyfit(ground.xs[15:46], east, 1)[0]
    assert math.degrees(math.atan(-slope)) == pytest.approx(25.0, abs=2.0)


# ---------------------------------------------------------------------- where things are in the world
def _flat_layers(z_a: float, z_b: float):
    ground = Ground(30, 2.0)
    dem = np.full((30, 30), 12.0)
    contacts = []
    for i, (x, y) in enumerate(ground.holes(4, 0.1)):
        contacts += [["A", x, y, z_a, 1.0, f"S{i}"], ["B", x, y, z_b, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=1.0, zmax=14.0, dz=0.5)
    job["series"][0]["units"] = ["A", "B"]
    return ground, job, dem


@loop
def test_solids_are_where_the_ground_is():
    ground, job, dem = _flat_layers(7.13, 4.31)
    result = engine.run(job, dem)
    for unit in ("T", "A", "B"):
        vertices = result.solids[unit][0]
        assert vertices[:, 0].min() == pytest.approx(0.0) and vertices[:, 0].max() == pytest.approx(60.0)
        assert vertices[:, 1].min() == pytest.approx(0.0) and vertices[:, 1].max() == pytest.approx(60.0)
    assert result.solids["B"][0][:, 2].min() == pytest.approx(1.0)  # the floor of the box
    assert result.solids["T"][0][:, 2].max() == pytest.approx(12.0, abs=0.03)  # the DEM
    inner = result.solids["A"][0]
    middle = (inner[:, 0] > 5) & (inner[:, 0] < 55) & (inner[:, 1] > 5) & (inner[:, 1] < 55)
    heights = inner[middle, 2]
    assert np.abs(heights[heights > 6] - 7.13).max() < 0.05 and np.abs(heights[heights < 6] - 4.31).max() < 0.05


@loop
def test_a_top_is_refined_under_a_tenth_of_a_voxel():
    ground = Ground(50, 1.0)
    dem = np.full((50, 50), 20.0)
    plane = lambda x, y: 11.0 + 0.031 * x - 0.017 * y  # noqa: E731
    holes = ground.holes(5, 0.1)
    contacts = []
    for i, (x, y) in enumerate(holes):
        contacts += [["A", x, y, plane(x, y), 1.0, f"S{i}"], ["B", x, y, plane(x, y) - 4, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=2.0, zmax=22.0, dz=0.5)
    job["series"][0]["units"] = ["A", "B"]
    result = engine.run(job, dem)
    error = np.abs(result.tops["A"] - plane(ground.X, ground.Y))[8:42, 8:42]
    assert float(np.median(error)) < 0.05 and float(error.max()) < 0.1


@loop
def test_the_jackknife_raster_is_not_upside_down():
    ground = Ground(50, 1.0)
    dem = np.full((50, 50), 20.0)
    rng = np.random.default_rng(5)
    xy = np.column_stack([rng.uniform(3, 47, 16), rng.uniform(32, 47, 16)])  # all in the north
    noise = rng.normal(0, 0.15, (16, 1))
    contacts = []
    for i, (x, y) in enumerate(xy):
        z = 10.0 + 0.02 * y + float(noise[i, 0])
        contacts += [["A", x, y, z, 1.0, f"S{i}"], ["B", x, y, z - 4, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=22.0, dz=0.5, folds=8)
    job["series"][0]["units"] = ["A", "B"]
    variance = engine.run(job, dem).variance["A"]
    assert np.nanmedian(variance[40:, :]) > 3 * np.nanmedian(variance[:10, :])  # row 0 is north


@loop
def test_an_orientation_of_a_depth_series_counts_the_slope_of_the_dem():
    ground = Ground(60, 1.0)
    dem = 26 - 0.2 * (ground.X - 30)
    dip = math.radians(15)
    normal = [math.sin(dip), 0.0, math.cos(dip)]
    plane = lambda x: 16.0 - math.tan(dip) * (x - 30.0)  # noqa: E731
    contacts = [["A", 30.0, y, plane(30.0), 1.0, f"S{i}"] for i, y in enumerate((10.0, 25.0, 40.0, 50.0))]
    job = layers_job(ground, dem, contacts, ["T", "A"], spaces={"d": ("depth", ["A"])}, zmin=-10.0, zmax=40.0,
                     dz=0.25, vertical=6.0)
    job["orientations"] = [["A", 30.0, 30.0, plane(30.0), *normal]]
    top = engine.run(job, dem).tops["A"]
    slope = np.polyfit(ground.xs[15:46], np.array([top[30, c] for c in range(15, 46)], dtype=float), 1)[0]
    assert math.degrees(math.atan(-slope)) == pytest.approx(15.0, abs=1.5)


@loop
def test_an_orientation_counts_in_a_series_of_several_contacts():
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 30.0)
    dip = math.radians(25)
    normal = [math.sin(dip), 0.0, math.cos(dip)]
    plane = lambda x, z0: z0 - math.tan(dip) * (x - 30.0)  # noqa: E731
    contacts = []
    for i, y in enumerate((10.0, 25.0, 40.0, 50.0)):  # along the strike: nothing else gives the dip
        contacts += [["A", 30.0, y, plane(30.0, 20.0), 1.0, f"S{i}"], ["B", 30.0, y, plane(30.0, 14.0), 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=35.0, dz=0.25, vertical=6.0)
    job["series"][0]["units"] = ["A", "B"]
    job["orientations"] = [["A", 30.0, 30.0, 20.0, *normal], ["B", 30.0, 30.0, 14.0, *normal]]
    result = engine.run(job, dem)
    for unit, z0 in (("A", 20.0), ("B", 14.0)):
        top = np.array([result.tops[unit][30, c] for c in range(15, 46)], dtype=float)
        assert math.degrees(math.atan(-np.polyfit(ground.xs[15:46], top, 1)[0])) == pytest.approx(25.0, abs=2.0)
    assert not [w for w in result.info["warnings"] if w["code"] == "few_holes"]


@loop
def test_an_older_unit_keeps_its_ground_where_a_younger_contact_runs_under_it():
    ground = Ground(40, 1.0)
    dem = np.full((40, 40), 12.0)
    contacts = []
    for i, (x, y) in enumerate(ground.holes(5, 0.1)):
        contacts += [["A", x, y, 7.0, 1.0, f"S{i}"], ["B", x, y, 3.0 + 0.2 * x, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=14.0, dz=0.25)
    job["series"] = [{"name": "a", "space": "elevation", "units": ["A"]},
                     {"name": "b", "space": "elevation", "units": ["B"]}]
    result = engine.run(job, dem)
    index = {u: i for i, u in enumerate(result.units)}
    zc = (np.arange(result.labels.shape[2]) + 0.5) * 0.25
    east, west = result.labels[20, 35], result.labels[20, 5]
    assert east[np.argmin(np.abs(zc - 6.0))] == index["B"]  # B stands above where A would be: A is gone here
    assert not (east == index["A"]).any()
    assert west[np.argmin(np.abs(zc - 6.0))] == index["A"] and west[np.argmin(np.abs(zc - 2.0))] == index["B"]
    # the top of "A and what is under it" is the top of B there: the cumulative set keeps the older unit
    assert result.tops["A"][20, 30] == pytest.approx(3.0 + 0.2 * 30.5, abs=0.1)
    assert result.tops["A"][20, 5] == pytest.approx(7.0, abs=0.1)


# ---------------------------------------------------------------------- the review's cases
def _ditch(outline, inside, depth=1.5, profile="flat", n=200, dz=0.25):
    ground = Ground(n, 0.5)
    dem = np.full((n, n), 10.0)
    contacts = [[u, float(x), float(y), z, 1.0, f"S{i}"] for i, (x, y) in enumerate(ground.holes(6, 0.08))
                for u, z in (("B", 9.0), ("C", 7.0))]
    mask = polygon_mask(ground, outline)
    job = {"grid": ground.grid, "zmin": 3.0, "zmax": 12.0, "dz": dz,
           "series": [{"name": "l", "space": "elevation", "units": ["B", "C"]}], "contacts": contacts,
           "orientations": [], "column": [{"unit": "F", "kind": "fill"}, {"unit": "P", "kind": "cut"}] +
                     [{"unit": u, "kind": "layer"} for u in "ABC"],
           "fills": {"F": "P"}, "options": {},
           "cuts": {"P": {"base": [[x, y, 10.0 - depth] for x, y in inside], "outline": mask, "inside": inside,
                          "outside": [], "profile": profile}}}
    result = engine.run(job, dem)
    return result, (result.labels == result.units.index("F")).any(axis=2), mask


STRAIGHT = [[10, 49], [90, 49], [90, 51], [10, 51]]
_ARC = np.linspace(0, np.pi, 40)
CURVED = [[50 + 31.5 * np.cos(a), 50 + 31.5 * np.sin(a)] for a in _ARC] + \
         [[50 + 28.5 * np.cos(a), 50 + 28.5 * np.sin(a)] for a in _ARC[::-1]]
BENT = [[20, 20], [72, 20], [72, 22], [22, 22], [22, 72], [20, 72]]
TEE = [[10, 49], [90, 49], [90, 51], [51, 51], [51, 80], [49, 80], [49, 51], [10, 51]]
AXIS = [[20.0, 50.0], [50.0, 50.0], [80.0, 50.0]]


@loop
@pytest.mark.parametrize("outline, holes, truth", [
    (STRAIGHT, AXIS, 240.0),
    (CURVED, [[50 + 30 * np.cos(a), 50 + 30 * np.sin(a)] for a in (0.4, 1.57, 2.7)],
     1.5 * 0.5 * np.pi * (31.5 ** 2 - 28.5 ** 2)),
    (BENT, [[21.0, 60.0], [21.0, 21.0], [60.0, 21.0]], 306.0),
    (TEE, [[20.0, 50.0], [80.0, 50.0], [50.0, 70.0]], 327.0)], ids=["straight", "curved", "bent", "tee"])
def test_a_ditch_with_a_flat_floor_fills_its_plan(outline, holes, truth):
    result, cells, mask = _ditch(outline, holes)
    assert result.volumes["F"] == pytest.approx(truth, rel=0.10)
    assert np.array_equal(cells, mask)  # the footprint is the raster, cell by cell
    assert result.info["cuts"]["P"]["field"] is False
    assert result.info["cuts"]["P"]["plan_area"] == pytest.approx(mask.sum() * 0.25)
    assert max(result.info["cuts"]["P"]["base_miss"]) < 1e-9


@loop
def test_a_diagonal_ditch_follows_its_raster():
    a = math.pi / 4
    along, across, start = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)]), np.array([15.0, 15.0])
    outline = [start + across, start + along * 120 + across, start + along * 120 - across, start - across]
    result, cells, mask = _ditch(outline, [list(start + along * s) for s in (20, 60, 100)], n=300)
    assert np.array_equal(cells, mask)
    # the raster of a band at 45 degrees loses a strip: the volume is the mask's
    assert result.volumes["F"] == pytest.approx(mask.sum() * 0.25 * 1.5, rel=0.01)


@loop
def test_a_v_ditch_needs_its_profile():
    result, cells, mask = _ditch(STRAIGHT, AXIS, profile="v", dz=0.1)
    assert result.volumes["F"] == pytest.approx(120.0, rel=0.10)
    assert result.info["cuts"]["P"]["volume_range"]["v"] == pytest.approx(120.0, rel=0.02)
    assert np.array_equal(cells, mask)


@loop
@pytest.mark.parametrize("truth", [240.0, 180.0, 120.0], ids=["flat", "bowl", "v"])
def test_the_truth_lies_inside_the_range_whatever_the_profile(truth):
    # the three holes on the axis read the same depth under any of the three shapes
    result, _cells, _mask = _ditch(STRAIGHT, AXIS, profile="bowl")
    span = result.info["cuts"]["P"]["volume_range"]
    assert span["v"] * 0.99 <= truth <= span["flat"] * 1.01


@loop
def test_a_hole_past_the_side_is_said_and_leaves_the_floor_on():
    result, cells, mask = _ditch(STRAIGHT, [[20.0, 50.0], [50.0, 52.2], [80.0, 50.0]])
    assert result.volumes["F"] == pytest.approx(240.0, rel=0.10)
    assert np.array_equal(cells, mask)
    said = [w for w in result.info["warnings"] if w["code"] == "base_outside_outline"]
    assert len(said) == 1 and said[0]["distance"] == pytest.approx(1.2, abs=0.3)


@loop
def test_no_cut_runs_loopstructural_without_walls():
    _ground, dem, job, _base = _pit_fixture(3)
    names = _solved(job, dem)
    names.pop()
    assert not [n for n in names if n.startswith("cut ") or n.startswith("fills ")]


@loop
def test_a_contact_deepening_past_the_boreholes_stays_a_contact():
    ground = Ground(100, 1.0)
    dem = np.full((100, 100), 20.0)
    a = lambda x: 17 - 0.06 * (x - 50)  # noqa: E731
    contacts = []
    for i, (x, y) in enumerate((x, y) for x in (40.0, 47.0, 53.0, 60.0) for y in (30.0, 50.0, 70.0)):
        contacts += [["A", x, y, a(x), 1.0, f"S{i}"], ["B", x, y, a(x) - 2, 1.0, f"S{i}"]]
    results = {}
    for space in ("depth", "elevation"):
        job = layers_job(ground, dem, contacts, ["T", "A", "B"], spaces={"s": (space, ["A", "B"])}, zmin=0.0, zmax=22.0)
        results[space] = engine.run(job, dem)
    for unit in ("T", "A", "B"):
        assert results["depth"].volumes[unit] == pytest.approx(results["elevation"].volumes[unit], rel=0.15)
    assert not np.isnan(results["depth"].tops["B"]).any()
    assert float(np.median(np.abs(results["depth"].tops["A"] - a(ground.X)))) < 0.25


@loop
def test_a_voxel_past_the_box_stays_on_its_side():
    ground = Ground(40, 1.0)
    dem = np.full((40, 40), 40.0)
    contacts = []
    for i, (x, y) in enumerate(ground.holes(4, 0.1)):
        contacts += [["A", x, y, 38.0, 1.0, f"S{i}"], ["B", x, y, 36.0, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], spaces={"s": ("depth", ["A", "B"])}, zmin=0.0, zmax=42.0)
    result = engine.run(job, dem)
    assert result.volumes["B"] == pytest.approx(36.0 * 1600, rel=0.02)  # 36 m down, far below any data


@loop
def test_a_contact_that_leaves_the_range_of_the_job_is_said():
    ground = Ground(40, 1.0)
    dem = np.full((40, 40), 20.0)
    contacts = []
    for i, (x, y) in enumerate(ground.holes(4, 0.3)):
        contacts += [["A", x, y, 2.5 - 0.3 * (x - 20), 1.0, f"S{i}"], ["B", x, y, 12 - 0.3 * (x - 20), 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "B", "A"], zmin=0.0, zmax=22.0, dz=0.5)
    job["series"][0]["units"] = ["B", "A"]
    result = engine.run(job, dem)
    assert {"code": "contact_leaves_box", "unit": "A"} in result.info["warnings"]


@loop
def test_a_rare_contact_on_one_side_keeps_its_distance():
    ground = Ground(60, 2.0)
    dem = np.full((60, 60), 40.0)
    a = lambda x: 38 - 0.12 * x  # noqa: E731
    contacts = []
    for i, (x, y) in enumerate(ground.holes(7, 0.06, jitter=2.0)):
        contacts += [["A", x, y, a(x), 1.0, f"S{i}"], ["C", x, y, a(x) - 6, 1.0, f"S{i}"]]
        if x < 35:
            contacts.append(["B", x, y, a(x) - 1.5, 1.0, f"S{i}"])
    job = layers_job(ground, dem, contacts, ["T", "A", "B", "C"], zmin=10.0, zmax=42.0, dz=0.25)
    job["series"][0]["units"] = ["A", "B", "C"]
    result = engine.run(job, dem)
    seen = ground.X < 35
    assert np.median(np.abs(result.tops["B"] - (a(ground.X) - 1.5))[seen]) < 0.3
    assert result.volumes["A"] == pytest.approx(1.5 * 120 * 120, rel=0.15)
    assert result.volumes["B"] == pytest.approx(4.5 * 120 * 120, rel=0.15)


@loop
def test_a_contact_sharing_no_hole_is_said():
    ground = Ground(40, 1.0)
    dem = np.full((40, 40), 20.0)
    contacts = []
    for i, (x, y) in enumerate(ground.holes(5, 0.1)):
        contacts += [["A", x, y, 16.0, 1.0, f"S{i}"], ["C", x, y, 8.0, 1.0, f"S{i}"]]
    contacts += [["B", 10.0 + i, 20.0 + 3 * i, 12.0, 1.0, f"R{i}"] for i in range(4)]
    job = layers_job(ground, dem, contacts, ["T", "A", "B", "C"], zmin=0.0, zmax=22.0, dz=0.5)
    job["series"][0]["units"] = ["A", "B", "C"]
    result = engine.run(job, dem)
    assert {"code": "contact_shares_no_hole", "unit": "B"} in result.info["warnings"]
    assert float(np.nanmedian(result.tops["B"])) == pytest.approx(12.0, abs=0.5)


@loop
def test_two_cuts_keep_their_fills():
    ground, dem, job, _result = _pit_fixture(13)
    pit_q = [[12.0, 12.0], [14.0, 12.0], [10.0, 12.0], [12.0, 14.0], [12.0, 10.0]]
    contacts = job["contacts"] + [[u, x, y, z, 1.0, f"Q{i}"] for i, (x, y) in enumerate(pit_q)
                                  for u, z in (("G1", 9.5), ("G2", 8.8))]
    column = job["column"][:2] + [{"unit": "G1", "kind": "fill"}, {"unit": "G2", "kind": "fill"},
                                  {"unit": "Q", "kind": "cut"}] + job["column"][2:]
    cuts = dict(job["cuts"])
    cuts["Q"] = {"base": [[x, y, 7.5] for x, y in pit_q], "outline": polygon_mask(ground, circle((12, 12), 5.0, 10.0)), "inside": pit_q,
                 "outside": job["cuts"]["P"]["outside"]}
    job = {**job, "contacts": contacts, "column": column, "fills": {"F": "P", "G1": "Q", "G2": "Q"}, "cuts": cuts}
    result = engine.run(job, dem)
    index = {u: i for i, u in enumerate(result.units)}
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    middle = result.labels[47, 12]
    assert not (middle == index["F"]).any()  # the fill of the other pit stays there
    assert middle[np.argmin(np.abs(zc - 9.8))] == index["G1"]  # up to the ground
    assert middle[np.argmin(np.abs(zc - 8.0))] == index["G2"]
    cells = (result.labels == index["F"]).any(axis=2)
    assert float(np.hypot(ground.X - 30, ground.Y - 30)[cells].max()) <= 8.0 + 1.5


def _pit_fixture(holes: int):
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 10.0)
    inside = [[30.0, 30.0], [34.0, 30.0], [26.0, 30.0], [30.0, 34.0], [30.0, 26.0], [33.0, 33.0], [27.0, 27.0],
              [33.0, 27.0], [27.0, 33.0], [36.5, 30.0], [23.5, 30.0], [30.0, 36.5], [30.0, 23.5]][:holes]
    base = [[x, y, 10.0 - float(bowl(np.hypot(x - 30, y - 30), 8.0, 2.8))] for x, y in inside]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.1) if np.hypot(x - 30, y - 30) > 9.0]
    job = _pit_job(ground, dem, {"A": 10.0, "B": 9.0, "C": 8.0}, inside, outside, base,
                   polygon_mask(ground, circle((30.0, 30.0), 8.0, 10.0)))
    return ground, dem, job, base


def _pit_volume(profile: str) -> float:
    r = np.linspace(0, 8, 8001)
    return float(np.sum(2 * np.pi * r * 2.8 * engine.PROFILES[profile]((8 - r) / 8)) * (r[1] - r[0]))


def _profile_pit(profile: str, positions: list):
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 10.0)
    base = [[30 + a, 30 + b, 10 - 2.8 * float(engine.PROFILES[profile](np.array(1 - math.hypot(a, b) / 8)))]
            for a, b in positions]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.1) if np.hypot(x - 30, y - 30) > 9.0]
    job = _pit_job(ground, dem, {"A": 10.0, "B": 9.0, "C": 8.0}, [b[:2] for b in base], outside, base,
                   polygon_mask(ground, circle((30.0, 30.0), 8.0, 10.0)))
    job["cuts"]["P"]["profile"] = profile
    return dem, job, base


HOLES = [(0, 0), (4, 0), (-4, 0), (0, 4), (0, -4), (3, 3), (-3, -3), (3, -3), (-3, 3), (6.5, 0), (-6.5, 0), (0, 6.5), (0, -6.5)]


@loop
@pytest.mark.parametrize("profile", list(engine.PROFILES))
@pytest.mark.parametrize("holes", [1, 3, 13])
def test_a_pit_of_a_known_profile(profile, holes):
    dem, job, base = _profile_pit(profile, HOLES[:holes])
    result = engine.run(job, dem)
    truth = _pit_volume(profile)
    assert result.volumes["F"] == pytest.approx(truth, rel=0.20 if holes == 1 else 0.10)
    cut = result.info["cuts"]["P"]
    assert not any(n for n in cut["base_capped"]) or profile == "v"
    assert max(m for m, held in zip(cut["base_miss"], cut["base_capped"]) if not held) < 1e-9
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    fill = result.units.index("F")
    for (x, y, z), held in zip(base, cut["base_capped"]):
        column = np.nonzero(result.labels[int(60 - y), int(x)] == fill)[0]
        assert held or abs(zc[column.min()] - 0.5 * job["dz"] - z) <= job["dz"] + 1e-6
    assert cut["volume_range"]["v"] * 0.97 <= truth <= cut["volume_range"]["flat"] * 1.03


@loop
@pytest.mark.parametrize("profile", ["flat", "bowl", "v"])
def test_a_hole_by_the_rim_alone_says_that_the_floor_was_not_measured(profile):
    dem, job, base = _profile_pit(profile, [(7.5, 0.0)])
    job["zmin"] = 0.0  # the old field went down to here
    result = engine.run(job, dem)
    if profile != "v":  # beside the edge of a V the measure says nothing of the middle: only the warning can tell
        assert result.volumes["F"] == pytest.approx(_pit_volume(profile), rel=0.40)
    deepest = 10.0 - base[0][2]
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    cells = (result.labels == result.units.index("F")).any(axis=(0, 1))
    assert float(zc[np.nonzero(cells)[0].min()]) - 0.5 * job["dz"] >= 10.0 - 4.0 * deepest - job["dz"]
    assert {"code": "floor_unmeasured", "unit": "P"} in result.info["warnings"]


@loop
def test_a_rim_hole_beside_a_deep_one_is_met_or_said():
    dem, job, base = _profile_pit("bowl", [(0.0, 0.0), (7.5, 0.0)])
    result = engine.run(job, dem)
    cut = result.info["cuts"]["P"]
    assert cut["base_capped"] == [False, True]  # P under P_MIN at the rim
    assert result.volumes["F"] == pytest.approx(_pit_volume("bowl"), rel=0.20)
    assert any(w["code"] == "base_near_edge" for w in result.info["warnings"])


@loop
def test_a_floor_stops_at_the_bottom_of_the_job():
    dem, job, _base = _profile_pit("flat", [(0.0, 0.0)])
    job["zmin"] = 8.5
    job["series"][0]["units"] = ["B"]
    job["contacts"] = [c for c in job["contacts"] if c[0] == "B"]
    job["column"] = job["column"][:-1]
    result = engine.run(job, dem)
    assert {"code": "cut_reaches_zmin", "unit": "P"} in result.info["warnings"]
    cut = result.info["cuts"]["P"]
    assert 0.9 * 1.5 * cut["plan_area"] < cut["volume_range"]["flat"] <= 1.5 * cut["plan_area"]  # not 2.8 m
    assert int((result.labels >= 0).sum()) == result.info["voxels_under_dem"]


@loop
def test_a_pit_on_a_slope_hangs_from_the_ground():
    ground = Ground(60, 1.0)
    dem = 10 + 0.25 * (ground.X - 30)
    reader = engine.Dem(engine.Grid(ground.grid), dem)
    holes = [[30.5, 30.5], [34.5, 30.5], [26.5, 30.5]]
    base = [[x, y, float(reader.at(np.array([x]), np.array([y]))[0]) - 2.0] for x, y in holes]
    outside = [[5.0, 5.0], [55.0, 55.0]]
    job = _pit_job(ground, dem, {"A": 10.0, "B": 4.0, "C": 2.0}, holes, outside, base,
                   polygon_mask(ground, circle((30.0, 30.0), 8.0, 10.0)))
    job["zmin"], job["zmax"] = -4.0, 20.0
    job["cuts"]["P"]["profile"] = "flat"
    result = engine.run(job, dem)
    assert max(result.info["cuts"]["P"]["base_miss"]) < 1e-9
    assert result.volumes["F"] == pytest.approx(result.info["cuts"]["P"]["volume_range"]["flat"], rel=0.05)
    assert int((result.labels >= 0).sum()) == result.info["voxels_under_dem"]
    for _unit, (_vertices, faces) in result.solids.items():
        assert _closed(faces)


@loop
def test_a_cut_nobody_drew_with_one_hole_is_a_disc():
    dem, job, _base = _profile_pit("bowl", [(0.0, 0.0)])
    job["cuts"]["P"].update(outline=None, outside=[])
    result = engine.run(job, dem)
    said = [w for w in result.info["warnings"] if w["code"] == "edge_estimated"]
    assert said[0]["limit"] == pytest.approx(2.8) and "overstated" in said[0]["effect"]
    assert result.info["cuts"]["P"]["edge"] == "estimated"
    cells = (result.labels == result.units.index("F")).any(axis=2)
    assert 0 < cells.sum() <= math.pi * 2.8 ** 2 * 1.3


@loop
def test_an_older_fill_without_a_contact_is_said(pit):
    ground, dem, job, _result = pit
    inside = job["cuts"]["P"]["inside"]
    contacts = job["contacts"] + [["F1", x, y, 9.0, 1.0, f"I{i}"] for i, (x, y) in enumerate(inside)]
    column = [{"unit": "F1", "kind": "fill"}, {"unit": "F2", "kind": "fill"}] + job["column"][1:]
    result = engine.run({**job, "contacts": contacts, "column": column, "fills": {"F1": "P", "F2": "P"}}, dem)
    assert {"code": "fill_without_contact", "unit": "F2", "cut": "P"} in result.info["warnings"]


@loop
@pytest.mark.parametrize("a, b", [((14.85, 15.15), (9.9, 10.1)), ((14.7, 15.3), (10.3, 9.7))], ids=["same way", "opposite"])
def test_two_close_holes_do_not_tilt_the_site(a, b):
    ground = Ground(60, 2.0)
    dem = np.full((60, 60), 20.0)
    holes = [(57.0, 60.0), (63.0, 60.0)]
    contacts = []
    for i, (x, y) in enumerate(holes):
        contacts += [["A", x, y, a[i], 1.0, f"S{i}"], ["B", x, y, b[i], 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=22.0)
    job["series"][0]["units"] = ["A", "B"]
    result = engine.run(job, dem)
    for (x, y), za, zb in zip(holes, a, b):
        row, col = int((120 - y) / 2), int(x / 2)
        assert abs(result.tops["A"][row, col] - za) <= 0.27 and abs(result.tops["B"][row, col] - zb) <= 0.27
    assert result.volumes["T"] == pytest.approx(5 * 14400, rel=0.01)
    assert result.volumes["A"] == pytest.approx(5 * 14400, rel=0.01)
    assert {"code": "few_holes", "series": "layers"} in result.info["warnings"]


@loop
@pytest.mark.parametrize("line", [60.0, 50.0], ids=["through the middle", "off the middle"])
def test_a_transect_is_not_held_by_pins_on_its_line(line):
    ground = Ground(60, 2.0)
    dem = np.full((60, 60), 20.0)
    top = lambda x: 14 + 2.5 * math.exp(-((x - 60) / 15) ** 2)  # noqa: E731
    contacts = []
    for i, x in enumerate(np.linspace(10, 110, 8)):
        x = float(x)
        contacts += [["A", x, line, top(x), 1.0, f"S{i}"], ["B", x, line, top(x) - 3, 1.0, f"S{i}"]]
    job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=22.0, folds=8 if line == 60.0 else None)
    job["series"][0]["units"] = ["A", "B"]
    result = engine.run(job, dem)
    assert line != 60.0 or result.loo["A"]["rmse"] < 1.1 * 0.59  # the old pins gave 0.94 m
    tops = result.tops["A"]
    assert abs(tops[int((120 - 5) / 2), 30] - tops[int((120 - 115) / 2), 30]) < 0.3  # no tilt across the line


def test_pins_keep_away_from_the_holes():
    ground = Ground(10, 10.0)
    frame = engine.Frame({"grid": ground.grid, "zmin": 0.0, "zmax": 10.0, "dz": 1.0}, np.full((10, 10), 8.0))
    seen = []
    original, engine._solve = engine._solve, lambda _n, rows, *_a: seen.extend(rows) or _Linear()
    try:
        contacts = [("A", 50.0 + i, 50.0, 4.0, 1.0, f"S{i}") for i in range(2)] + \
                   [("B", 50.0 + i, 50.0, 2.0, 1.0, f"S{i}") for i in range(2)]
        engine.fit_series(frame, {"name": "s", "units": ["A", "B"]}, contacts, [], ["A", "B"], 1.0, 1000)
    finally:
        engine._solve = original
    pins = [(r["X"], r["Y"]) for r in seen if "val" in r and (r["X"], r["Y"]) not in ((50.0, 50.0), (51.0, 50.0))]
    assert pins and all(math.hypot(x - 50, y - 50) >= 25 for x, y in pins)
    assert all(abs(r["nz"] - 1) < 1e-9 for r in seen if "nz" in r)  # one horizontal plane


@loop
def test_one_borehole_leaves_the_contacts_flat():
    ground = Ground(60, 2.0)
    dem = np.full((60, 60), 10.0)
    for x, y in ((60.0, 60.0), (8.0, 8.0)):
        contacts = [["A", x, y, 7.0, 1.0, "S1"], ["B", x, y, 4.0, 1.0, "S1"]]
        job = layers_job(ground, dem, contacts, ["T", "A", "B"], zmin=0.0, zmax=12.0, dz=0.25)
        job["series"][0]["units"] = ["A", "B"]
        result = engine.run(job, dem)
        assert np.abs(result.tops["A"] - 7.0).max() <= 0.25
        assert np.abs(result.tops["B"] - 4.0).max() <= 0.25
        assert {"code": "few_holes", "series": "layers"} in result.info["warnings"]


@loop
def test_a_normal_pointing_down_is_turned_up():
    ground = Ground(30, 1.0)
    dem = np.full((30, 30), 10.0)
    contacts = [["A", x, y, 6.0, 1.0, f"S{i}"] for i, (x, y) in enumerate(ground.holes(3, 0.2))]
    job = layers_job(ground, dem, contacts, ["T", "A"], zmin=0.0, zmax=12.0, dz=0.25)
    job["series"][0]["units"] = ["A"]
    job["orientations"] = [["A", 15.0, 15.0, 6.0, 0.0, 0.0, -1.0]]
    result = engine.run(job, dem)
    assert {"code": "normal_turned_up", "unit": "A"} in result.info["warnings"]
    index = {u: i for i, u in enumerate(result.units)}
    assert result.labels[15, 15, 4] == index["A"] and result.labels[15, 15, 30] == index["T"]


@loop
def test_a_cut_of_the_column_without_data_is_said(pit):
    _ground, dem, job, _result = pit
    result = engine.run({**job, "cuts": {}}, dem)
    assert {"code": "cut_without_data", "unit": "P"} in result.info["warnings"]
    assert result.volumes["F"] == 0


@loop
def test_a_column_against_the_boreholes_is_said(layered):
    job, dem, _rare, _result = layered
    column = [job["column"][0], job["column"][3], job["column"][2], job["column"][1]]
    result = engine.run({**job, "column": column, "options": {}}, dem)
    assert [w for w in result.info["warnings"] if w["code"] == "column_contradicted"][0]["holes"] > 0


# ---------------------------------------------------------------------- a dip against the contact, a covered pit
@loop
def test_a_dip_against_the_contact_never_turns_the_model_over():
    ground = Ground(50, 2.0)
    dem = 104.0 + 0.01 * ground.X + 0.01 * ground.Y
    contacts = []
    for i, (x, y) in enumerate(ground.holes(4, 0.1)):
        collar = 104.0 + 0.01 * x + 0.01 * y
        contacts += [["A", x, y, collar - 2.0, 1.0, f"S{i}"], ["B", x, y, 100.0 + 0.012 * x + 0.008 * y, 1.0, f"S{i}"]]
    spaces = {"a": ("elevation", ["A"]), "b": ("elevation", ["B"])}
    base = layers_job(ground, dem, contacts, ["T", "A", "B"], spaces=spaces, zmin=90.0, zmax=107.0, dz=0.25, vertical=30.0)
    plain = engine.run(base, dem)
    d, a = math.radians(10.0), math.radians(90.0)
    normal = [math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)]
    x, y, z = contacts[1][1:4]
    dipped = engine.run({**base, "orientations": [["B", x, y, z, *normal]]}, dem)
    for unit in ("T", "A", "B"):
        assert dipped.volumes[unit] == pytest.approx(plain.volumes[unit], rel=0.05)
    assert np.nanmedian(dipped.tops["B"]) < np.nanmedian(dipped.tops["A"]) < np.nanmedian(dem)


def _covered_pit(profile: str, truth: str):
    """A pit of radius 8 under 0.4 m of topsoil, floor of shape ``truth``, built with ``profile``: its
    fill is seen from 0.4 m down to the floor, at three holes."""
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 10.0)
    r = np.hypot(ground.X - 30, ground.Y - 30)
    mask = r < 8.0
    plan = engine.Plan.of(mask, 1.0, "drawn")
    depth = 2.5 * np.where(mask, engine.PROFILES[truth](plan.cell_t), 0.0)
    holes = [(30.0, 30.0), (33.0, 30.0), (27.0, 28.0)]
    contacts = [["L", x, y, 8.5, 1.0, f"O{i}"] for i, (x, y) in enumerate(ground.holes(4, 0.1))]
    base = []
    rows = lambda x, y: (int(60 - y), int(x))  # noqa: E731
    for i, (x, y) in enumerate(holes):
        contacts.append(["F", x, y, 9.6, 1.0, f"H{i}"])
        base.append([x, y, 10.0 - float(depth[rows(x, y)])])
    job = {"grid": ground.grid, "zmin": 2.0, "zmax": 11.0, "dz": 0.05, "contacts": contacts, "orientations": [],
           "series": [{"name": "l", "space": "elevation", "units": ["L"]}],
           "column": [{"unit": "T", "kind": "layer"}, {"unit": "F", "kind": "fill"}, {"unit": "P", "kind": "cut"},
                      {"unit": "L", "kind": "layer"}, {"unit": "N", "kind": "layer"}],
           "fills": {"F": "P"}, "cuts": {"P": {"base": base, "outline": mask, "profile": profile, "inside": holes}},
           "options": {}}
    job["contacts"] += [["N", x, y, 4.0, 1.0, f"O{i}"] for i, (x, y) in enumerate(ground.holes(4, 0.1))]
    job["series"].append({"name": "n", "space": "elevation", "units": ["N"]})
    return job, dem, float((np.maximum(depth - 0.4, 0.0)).sum())


@loop
@pytest.mark.parametrize("profile", ["flat", "bowl", "v"])
def test_the_range_of_a_covered_fill_holds_the_fill(profile):
    job, dem, truth = _covered_pit(profile, profile)
    result = engine.run(job, dem)
    cut = result.info["cuts"]["P"]
    fill = result.volumes["F"]
    span = cut["volume_range"]
    assert min(span.values()) <= fill <= max(span.values())
    assert span[profile] == pytest.approx(fill)
    assert fill == pytest.approx(truth, rel=0.15)  # three holes do not tell a V to better than that
    assert cut["cut_volume_range"][profile] > span[profile] + 0.5 * cut["plan_area"] * 0.4  # the topsoil is not in it
    assert max(span.values()) < max(cut["cut_volume_range"].values())


@loop
def test_the_range_of_an_open_pit_is_that_of_the_cut():
    job, dem, _truth = _covered_pit("flat", "flat")
    job["column"] = [{"unit": "F", "kind": "fill"}, {"unit": "P", "kind": "cut"}, {"unit": "T", "kind": "layer"},
                     {"unit": "L", "kind": "layer"}, {"unit": "N", "kind": "layer"}]
    cut = engine.run(job, dem).info["cuts"]["P"]
    for p in ("bowl", "v"):
        assert cut["volume_range"][p] == pytest.approx(cut["cut_volume_range"][p])


@loop
def test_a_field_turned_over_by_a_dip_is_fitted_again_without_it(monkeypatch):
    ground = Ground(40, 2.0)
    dem = np.full((40, 40), 20.0)
    contacts = [["A", x, y, 12.0, 1.0, f"S{i}"] for i, (x, y) in enumerate(ground.holes(4, 0.1))]
    job = layers_job(ground, dem, contacts, ["T", "A"], zmin=5.0, zmax=22.0, dz=0.5)
    job["series"][0]["units"] = ["A"]
    job["orientations"] = [["A", 40.0, 40.0, 12.0, 0.0, 0.0, 1.0]]
    original = engine._solve

    class Reversed:
        def __init__(self, feature):
            self.feature = feature

        def evaluate_value(self, points):
            return -self.feature.evaluate_value(points)

    def spy(name, rows, *args):  # a measure in the data: the field comes out the wrong way round
        feature = original(name, rows, *args)
        return Reversed(feature) if sum(1 for r in rows if r.get("nz") is not None) > 1 else feature

    monkeypatch.setattr(engine, "_solve", spy)
    result = engine.run(job, dem)
    assert {"code": "orientation_dropped", "unit": "A", "x": 40.0, "y": 40.0} in result.info["warnings"]
    assert sum(w["code"] == "orientation_dropped" for w in result.info["warnings"]) == 1
    assert np.nanmedian(result.tops["A"]) == pytest.approx(12.0, abs=0.3)
