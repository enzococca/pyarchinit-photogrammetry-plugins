"""The implicit model against synthetic truths (``implicit_engine.py`` loaded by its path, without the
program). The pure helpers run everywhere; the models need LoopStructural in this Python and are
skipped where it is not (the program's own venv) - run them with a Python that has it::

    <env>/bin/python -m pytest plugins/geoarchaeology/tests/test_implicit_engine.py -q
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from implicit_samples import Ground, bowl, circle, has_loopstructural, layers_job, load, stub_pyvista

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
    assert not is_plane and n[2] > 0.5  # along a line: the least tilted plane through it
    assert float(n @ np.array([1.0, 2.0, 1.0])) == pytest.approx(0.0, abs=1e-12)
    assert float(n @ np.array([-2.0, 1.0, 0.0])) == pytest.approx(0.0, abs=1e-12)
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


def test_an_estimated_edge_runs_halfway_between_the_holes():
    ground = Ground(40, 1.0)
    reader = engine.Dem(engine.Grid(ground.grid), np.full((40, 40), 10.0))
    inside = np.array([[20.0, 20.0]])
    outside = np.array([[28.0, 20.0], [12.0, 20.0], [20.0, 28.0], [20.0, 12.0]])
    edge = engine.estimated_edge(inside, outside, reader, 0.25)
    r = np.hypot(edge[:, 0] - 20, edge[:, 1] - 20)
    assert np.all((r > 3.5) & (r < 6.0))  # 4 m along the axes, more between them
    assert r[0] == pytest.approx(4.0, abs=0.3) and np.all(edge[:, 2] == 10.0)


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


def _pit_job(ground, dem, centre, radius, depth, tops, holes_in, holes_out, base_rows, edge_rows, **options):
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
            "cuts": {"P": {"base": base_rows, "edge": edge_rows, "inside": holes_in, "outside": holes_out}},
            "options": options}


@pytest.fixture(scope="module")
def pit():
    """A flat-bottomed pit 2.8 m deep through the layers A and B into C, filled by F, on flat ground."""
    ground = Ground(60, 1.0)
    dem = np.full((60, 60), 10.0)
    centre = (30.0, 30.0)
    radius, depth = 8.0, 2.8
    tops = {"A": 10.0, "B": 9.0, "C": 8.0}
    inside = [[30.0, 30.0], [34.0, 30.0], [26.0, 30.0], [30.0, 34.0], [30.0, 26.0], [33.0, 33.0], [27.0, 27.0],
              [33.0, 27.0], [27.0, 33.0], [36.5, 30.0], [23.5, 30.0], [30.0, 36.5], [30.0, 23.5]]
    base = [[x, y, 10.0 - float(bowl(np.hypot(x - 30, y - 30), radius, depth))] for x, y in inside]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.1) if np.hypot(x - 30, y - 30) > radius + 1]
    job = _pit_job(ground, dem, centre, radius, depth, tops, inside, outside, base, circle(centre, radius, 10.0))
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
    job = {**job, "cuts": {"P": {**job["cuts"]["P"], "edge": []}}}
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


@loop
def test_an_undercut_wall_stays_in_the_model():
    ground = Ground(50, 1.0)
    dem = np.full((50, 50), 10.0)
    centre = (25.0, 25.0)
    tops = {"A": 10.0, "B": 8.0}
    # a mouth of 4 m at the ground, 7 m of radius at 2 m down: a bottle pit, seen on a section of its wall
    a = 2 * np.pi * np.arange(24) / 24
    wall = [[25 + 7 * math.cos(t), 25 + 7 * math.sin(t), 8.0] for t in a]
    shoulder = [[25 + 5.5 * math.cos(t), 25 + 5.5 * math.sin(t), 9.1] for t in a]
    floor = [[25 + r * math.cos(t), 25 + r * math.sin(t), 6.5] for r in (0.0, 3.0, 5.5) for t in a[::4]]
    lower = [[25 + 5.5 * math.cos(t), 25 + 5.5 * math.sin(t), 6.8] for t in a]
    base = wall + shoulder + floor + lower
    inside = [[25.0, 25.0], [27.0, 25.0], [23.0, 25.0], [25.0, 27.0], [25.0, 23.0]]
    outside = [[float(x), float(y)] for x, y in ground.holes(5, 0.06) if np.hypot(x - 25, y - 25) > 10]
    job = _pit_job(ground, dem, centre, 7.0, 3.5, tops, inside, outside, base, circle(centre, 4.0, 10.0))
    job["dz"] = 0.2
    result = engine.run(job, dem)
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


# ---------------------------------------------------------------------- the review's cases
def _ditch(outline, inside, depth=1.5):
    ground = Ground(200, 0.5)
    dem = np.full((200, 200), 10.0)
    contacts = [[u, float(x), float(y), z, 1.0, f"S{i}"] for i, (x, y) in enumerate(ground.holes(6, 0.08))
                for u, z in (("B", 9.0), ("C", 7.0))]
    job = {"grid": ground.grid, "zmin": 3.0, "zmax": 12.0, "dz": 0.25,
           "series": [{"name": "l", "space": "elevation", "units": ["B", "C"]}], "contacts": contacts,
           "orientations": [], "column": [{"unit": "F", "kind": "fill"}, {"unit": "P", "kind": "cut"}] +
                     [{"unit": u, "kind": "layer"} for u in "ABC"],
           "fills": {"F": "P"}, "options": {},
           "cuts": {"P": {"base": [[x, y, 10.0 - depth] for x, y in inside], "edge": [[x, y, 10.0] for x, y in outline],
                          "inside": inside, "outside": []}}}
    result = engine.run(job, dem)
    return ground, result, (result.labels == result.units.index("F")).any(axis=2)


@loop
def test_a_straight_ditch_stays_in_its_outline():
    outline = [[10, 49], [90, 49], [90, 51], [10, 51]]
    ground, result, cells = _ditch(outline, [[20.0, 50.0], [50.0, 50.0], [80.0, 50.0]])
    assert result.volumes["F"] == pytest.approx(240.0, rel=0.15)
    far = np.maximum(np.abs(ground.X - 50) - 40, np.abs(ground.Y - 50) - 1)
    assert float(far[cells].max()) <= 1.5 * 0.5
    assert cells.sum() >= 0.95 * 640
    assert result.info["cuts"]["P"]["base_miss"]["max"] < 0.25


@loop
def test_a_curved_ditch_leaves_the_inside_of_its_bend_alone():
    t = np.linspace(0, np.pi, 40)
    outline = [[50 + 31.5 * np.cos(a), 50 + 31.5 * np.sin(a)] for a in t]
    outline += [[50 + 28.5 * np.cos(a), 50 + 28.5 * np.sin(a)] for a in t[::-1]]
    inside = [[50 + 30 * np.cos(a), 50 + 30 * np.sin(a)] for a in (0.4, 1.57, 2.7)]
    ground, result, cells = _ditch(outline, inside)
    truth = 1.5 * 0.5 * np.pi * (31.5 ** 2 - 28.5 ** 2)
    assert result.volumes["F"] == pytest.approx(truth, rel=0.15)
    r = np.hypot(ground.X - 50, ground.Y - 50)
    apart = np.maximum(np.maximum(r - 31.5, 28.5 - r), 50 - ground.Y)
    assert float(apart[cells].max()) <= 1.5 * 0.5
    assert not cells[r < 27].any()


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
    cuts["Q"] = {"base": [[x, y, 7.5] for x, y in pit_q], "edge": circle((12, 12), 5.0, 10.0), "inside": pit_q,
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
    job = _pit_job(ground, dem, (30.0, 30.0), 8.0, 2.8, {"A": 10.0, "B": 9.0, "C": 8.0}, inside, outside, base,
                   circle((30.0, 30.0), 8.0, 10.0))
    return ground, dem, job, base


PIT_VOLUME = float(np.sum(2 * np.pi * np.linspace(0, 8, 8001) * bowl(np.linspace(0, 8, 8001), 8.0, 2.8)) * 8 / 8000)


@loop
@pytest.mark.parametrize("holes", [1, 3, 5])
def test_a_pit_met_by_few_boreholes(holes):
    ground, dem, job, base = _pit_fixture(holes)
    result = engine.run(job, dem)
    assert result.volumes["F"] == pytest.approx(PIT_VOLUME, rel=0.20 if holes == 1 else 0.15)
    zc = job["zmin"] + (np.arange(result.labels.shape[2]) + 0.5) * job["dz"]
    fill = result.units.index("F")
    for x, y, z in base:
        column = np.nonzero(result.labels[int(60 - y), int(x)] == fill)[0]
        assert abs(zc[column.min()] - 0.5 * job["dz"] - z) <= job["dz"] + 1e-6  # within a voxel of the measure
    assert result.info["cuts"]["P"]["base_miss"]["none"] == 0
    assert result.info["cuts"]["P"]["base_miss"]["max"] < job["dz"]


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
