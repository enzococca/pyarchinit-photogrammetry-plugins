"""ERT profiles in the project: the line placed from the coordinates of the file (and refused when they
are not in the survey), the spacing along the ground, the files that follow the program's Undo (the
electrodes in the record, inversions and sections under names of their own, a removed profile in a
trash folder until the project is opened again), the DEM sampled without reading its whole box, the
worker stopped after a time limit."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg_geophysics import ert, ert_ops, ert_runner, ert_section
from test_ert import FAKE, _project, write_ohm

DATA = Path(__file__).resolve().parent / "data"
LINE = [[500010.0, 4500050.0], [500040.0, 4500050.0]]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ert_ops, "WORKER", FAKE)
    monkeypatch.setattr(ert_ops, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.delenv("PPG_ERT_FAKE", raising=False)


def _steep_dem(ch, slope: float = 0.3):
    import rasterio

    with rasterio.open(ch.dem_path, "r+") as dst:
        w, h = dst.width, dst.height
        cols = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)[0]
        dst.write((100.0 + slope * cols * 0.5).astype(np.float32), 1)  # cells of 0.5 m


def test_the_line_from_world_coordinates_of_the_file(tmp_path):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, DATA / "gimli_world.dat", name="W")
    assert e["world"]
    e = ert.place(ch, e["id"], None, heights="file")
    prof = ert.load(ch, e["id"])
    assert e["placement"]["method"] == "file" and np.allclose(prof.elec["elec_x"], prof.file_x)
    assert np.allclose(prof.elec["elec_z"], 120 + 0.1 * np.arange(8))
    # coordinates of a local frame of the instrument (near 0, 0): never placed near the origin of the CRS
    f = tmp_path / "local.dat"
    f.write_text("4\n# x y z\n0 0 10\n1 0.5 10.1\n2 1 10.2\n3 1.5 10.3\n1\n# a b m n rhoa\n1 2 3 4 10\n0\n")
    e = ert_ops.import_profile(ch, f, name="L")
    assert e["world"]
    with pytest.raises(ValueError, match="lontane dal rilievo"):
        ert.place(ch, e["id"], None)
    assert "placement" not in ert.entry_of(ch, e["id"])
    # a profile of distances alone needs a line
    e = ert_ops.import_profile(ch, DATA / "gimli_topo.dat", name="G")
    with pytest.raises(ValueError):
        ert.place(ch, e["id"], None)


def test_the_spacing_along_the_ground(tmp_path):
    doc, ch = _project(tmp_path)
    _steep_dem(ch, 0.3)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm", spacing=2.0), name="P")
    e = ert.place(ch, e["id"], LINE, along="ground", heights="dem")
    prof = ert.load(ch, e["id"])
    du = np.diff(prof.elec["elec_u"])
    dz = np.diff(prof.elec["elec_z"])
    assert np.allclose(du, 2.0 / np.sqrt(1 + 0.09), atol=2e-3)  # shorter in plan
    assert np.allclose(np.hypot(du, dz), 2.0, atol=2e-3)  # 2 m along the slope, as the cable
    assert e["placement"]["along"] == "ground" and "lungo il terreno" in e["placement"]["notes"][0]
    e = ert.place(ch, e["id"], LINE, along="plan", heights="dem")
    assert np.allclose(np.diff(ert.load(ch, e["id"]).elec["elec_u"]), 2.0)
    # the heights of the file: the slope between neighbours (8 % in pyGIMLi's file)
    g = ert_ops.import_profile(ch, DATA / "gimli_topo.dat", name="G")
    g = ert.place(ch, g["id"], LINE, along="ground", heights="file")
    assert np.allclose(np.diff(ert.load(ch, g["id"]).elec["elec_u"]), np.sqrt(1 - 0.08 ** 2), atol=2e-4)


def test_the_dem_read_point_by_point_gives_the_same_heights(tmp_path, monkeypatch):
    doc, ch = _project(tmp_path)
    xy = np.column_stack([np.linspace(500001, 500099, 37), np.linspace(4500099, 4500001, 37)])
    whole = ert.sample_dem(ch.dem_path, xy)
    monkeypatch.setattr(ert, "WINDOW_CELLS", 1)
    by_point = ert.sample_dem(ch.dem_path, xy)
    assert np.isfinite(whole).all() and np.allclose(whole, by_point, atol=1e-4)
    assert np.isnan(ert.sample_dem(ch.dem_path, [[0.0, 0.0]])).all()


def test_the_files_follow_the_undo_of_the_program(tmp_path, fake):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P")
    pid = e["id"]
    npz = ert.folder(ch, pid) / "profile.npz"
    stamp = npz.stat().st_mtime_ns
    # a placement lives in the record: taking the record back takes the electrodes back
    ert.place(ch, pid, LINE, along="plan")
    first = copy.deepcopy(ch.meta)
    ert.place(ch, pid, [[500010.0, 4500060.0], [500040.0, 4500060.0]], along="plan")
    assert np.allclose(ert.load(ch, pid).elec["elec_y"], 4500060.0)
    assert npz.stat().st_mtime_ns == stamp  # the readings' file is never written again
    ch.meta.clear()
    ch.meta.update(copy.deepcopy(first))
    assert np.allclose(ert.load(ch, pid).elec["elec_y"], 4500050.0)
    # every inversion and every drawing of its section under names of their own
    ert_ops.invert_profile(ch, pid, {"lam": 10})
    inverted = copy.deepcopy(ch.meta)
    files1 = ert.run_files(ert.entry_of(ch, pid))
    ert_section.write(ch, pid, palette="viridis")
    ert_ops.invert_profile(ch, pid, {"lam": 30})
    files2 = ert.run_files(ert.entry_of(ch, pid))
    assert not files1 & {f for f in files2 if "section" in f}
    assert all((ch.data_dir / f).exists() for f in files1 | files2)
    ch.meta.clear()
    ch.meta.update(copy.deepcopy(inverted))  # Undo of the second inversion
    assert ert_section.view_path(ch, pid) == ch.data_dir / ert.entry_of(ch, pid)["inversion"]["files"]["view"]
    assert ert_section.load_result(ch, pid)["fit"]["lam"] == 10
    # removed: the folder waits in the trash; Undo brings back the record, the files follow
    before = copy.deepcopy(ch.meta)
    ert.remove(ch, pid)
    assert not ert.folder(ch, pid).exists() and (ert.trash(ch) / pid).is_dir()
    ch.meta.clear()
    ch.meta.update(copy.deepcopy(before))
    assert ert.load(ch, pid).count == e["readings"] and ert.folder(ch, pid).is_dir()
    # the project opened again: the files no record uses are deleted
    out = ert.settle(ch)
    assert out["files"] >= 4 and all((ch.data_dir / f).exists() for f in ert.run_files(ert.entry_of(ch, pid)))
    ert.remove(ch, pid)
    out = ert.settle(ch)  # removed and the project as saved without it
    assert out["deleted"] == 1 and not ert.trash(ch).exists()


def test_the_worker_stopped_after_its_time(tmp_path, fake, monkeypatch):
    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"))
    monkeypatch.setenv("PPG_ERT_FAKE", "sleep")
    with pytest.raises(ert_runner.WorkerError, match="limite di tempo"):
        ert_ops.invert_profile(ch, e["id"], timeout=1.0)
    assert not list(ert.folder(ch, e["id"]).glob("inversion*"))


def _model(ch, n: int = 40):
    """A model of the chunk: a grid of n x n cells on the DEM's plane, written as the program's model."""
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply

    xs, ys = np.meshgrid(np.linspace(500000, 500100, n + 1), np.linspace(4500000, 4500100, n + 1))
    world = np.column_stack([xs.ravel(), ys.ravel(), 100 + 0.02 * (xs.ravel() - 500000)])
    idx = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    q = np.stack([idx[:-1, :-1], idx[:-1, 1:], idx[1:, 1:], idx[1:, :-1]], -1).reshape(-1, 4)
    faces = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    p = ch.data_dir / "model.ply"
    write_mesh_ply(p, ch.to_internal(world), faces, np.full((len(world), 3), 120, np.uint8), double=True)
    ch._set_product("model", p)
    return p


def test_the_3d_scene_leaves_the_programs_model_preview_alone(tmp_path, fake, monkeypatch):
    doc, ch = _project(tmp_path)
    model = _model(ch)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P")
    ert.place(ch, e["id"], LINE, along="plan")
    ert_ops.invert_profile(ch, e["id"])
    monkeypatch.setattr(ert_section, "MAX_TERRAIN_FACES", 500)  # the model is "large": decimated in memory
    path, what = ert_section.scene(ch, terrain="model")
    assert what["terrain"] and path is not None
    assert not (model.parent / "model_preview.ply").exists()


def test_the_proxies_of_the_units_in_the_3d_scene(tmp_path, fake):
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh, write_mesh_ply

    doc, ch = _project(tmp_path)
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "line.ohm"), name="P")
    ert.place(ch, e["id"], LINE, along="plan")
    ert_ops.invert_profile(ch, e["id"])
    box = np.array([[500020, 4500048, 99], [500024, 4500048, 99], [500024, 4500052, 99], [500020, 4500052, 99],
                    [500020, 4500048, 100], [500024, 4500048, 100], [500024, 4500052, 100], [500020, 4500052, 100]], float)
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7], [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    (ch.data_dir / "proxies").mkdir(parents=True, exist_ok=True)
    write_mesh_ply(ch.data_dir / "proxies" / "US_1.ply", box, faces, double=True)
    ch.meta["proxies"] = [{"name": "US 1", "us": "1", "file": "proxies/US_1.ply"},
                          {"name": "US 9", "us": "9", "file": "proxies/missing.ply"}]
    path, what = ert_section.scene(ch, terrain="none")
    assert what["proxies"] == 1
    m = read_mesh(path)
    world = ch.to_world(np.asarray(m.xyz, float))
    near = np.all(np.isclose(world[:, None, :], box[None], atol=1e-6), axis=2).any(axis=1)
    assert near.sum() >= 8 and (np.asarray(m.colors)[near][:, :3] == [230, 25, 75]).all()
    path, what = ert_section.scene(ch, terrain="none", units=False)
    assert what["proxies"] == 0


def test_the_cores_in_the_colours_of_geoarchaeology(tmp_path):
    doc, ch = _project(tmp_path)
    ch.meta["geoarchaeology"] = {
        "units": [{"code": "1"}, {"code": "2", "colour": "#3366cc"}],
        "boreholes": [{"id": "S1", "x": 500020.0, "y": 4500052.0, "z": 100.4,
                       "intervals": [{"from": 0, "to": 1, "unit": "1"}, {"from": 1, "to": 2, "unit": "2"},
                                     {"from": 2, "to": 3, "unit": "2", "colour": "#ff0000"},
                                     {"from": 3, "to": 4, "unit": "2", "colour": "10YR 4/3"}]}]}
    v, f, c = ert_section._cores(ch, [LINE], 50.0, 0.1)
    rgb = [tuple(int(x) for x in c[8 * k]) for k in range(4)]
    assert rgb == [(0xd9, 0xb7, 0x7e), (0x33, 0x66, 0xcc), (255, 0, 0), (0x33, 0x66, 0xcc)]
