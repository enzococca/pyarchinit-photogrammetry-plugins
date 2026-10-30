"""The implicit model on the side of the program: what goes in the job, what comes back and where it goes,
in the worker's own process. With a fake worker always (:mod:`implicit_plugin_fake_worker`); with the real
one when a Python with LoopStructural is known (``PPG_TEST_LOOP``)."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
from implicit_samples import PACKAGE, loop_python
from test_geoarchaeology import CELL, X0, Y0, project
from test_tables_and_display import write

import pyarchinit_photogrammetry as ppg
from ppg_geoarchaeology import boreholes as bh
from ppg_geoarchaeology import export, implicit, model, section
from ppg_geoarchaeology import surfaces as sf

FAKE = Path(__file__).with_name("implicit_plugin_fake_worker.py")
CENTRE = (X0 + 50.0, Y0 + 40.0)  # the middle of the pit
RADIUS, PIT = 8.0, 2.5
TOP, SECOND = 0.4, 1.5  # the base of US 1 and of US 2 outside the pit


def logs(tmp_path: Path, name: str = "t.csv") -> Path:
    """Boreholes on a grid of 20 m: US 1, US 2, US 3 down to 5 m; the two in the pit (a fill US 7 from
    under the topsoil down to the floor, the rest of the pit is US 3), and a hole ending inside the pit."""
    rows = ["sondaggio;x;y;da;a;us"]
    k = 0
    for dx in (-40.0, -20.0, 20.0, 40.0):
        for dy in (-30.0, -15.0, 15.0, 30.0):
            k += 1
            rows += [f"S{k};{CENTRE[0] + dx};{CENTRE[1] + dy};0;{TOP};US 1",
                     f"S{k};{CENTRE[0] + dx};{CENTRE[1] + dy};{TOP};{SECOND};US 2",
                     f"S{k};{CENTRE[0] + dx};{CENTRE[1] + dy};{SECOND};5;US 3"]
    for name_, dx in (("P1", -3.0), ("P2", 3.0)):
        rows += [f"{name_};{CENTRE[0] + dx};{CENTRE[1]};0;{TOP};US 1",
                 f"{name_};{CENTRE[0] + dx};{CENTRE[1]};{TOP};{PIT};US 7",
                 f"{name_};{CENTRE[0] + dx};{CENTRE[1]};{PIT};5;US 3"]
    return write(tmp_path, name, "\n".join(rows) + "\n")


def draw(chunk, us: str, radius: float = RADIUS, centre=CENTRE) -> None:
    from shapely.geometry import Point

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    d = chunk.meta.setdefault("drawings", {"units": [], "quote": []})
    d["units"].append(UnitPolygon(Point(*centre).buffer(radius, 32), us=us).to_dict())


def pit_site(tmp_path: Path, surfaces: bool = True):
    doc, ch = project(tmp_path)
    bh.import_files(ch, [logs(tmp_path)])
    if surfaces:
        sf.build(ch)
    implicit.mark_fill(ch, "US 7", "T1")
    return doc, ch


def job_of(ch, tmp_path: Path, **kw) -> tuple[dict, implicit.Plan]:
    return implicit.prepare(ch, tmp_path / "job", **kw)


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(implicit, "WORKER", (Path(sys.executable), FAKE))


# ------------------------------------------------------------------ the job
def test_the_base_of_a_fill_goes_to_the_cut_and_not_to_the_layer_it_rests_on(tmp_path):
    doc, ch = pit_site(tmp_path)
    draw(ch, "7")
    job, _plan = job_of(ch, tmp_path)
    rows = job["contacts"]
    at_p1 = [r for r in rows if r[5] == "P1"]
    assert {r[0] for r in at_p1} == {"US 1", "US 7"}  # no US 3: its top is not there, the cut took it away
    base = job["cuts"]["T1"]["base"]
    assert len(base) == 2 and {tuple(np.round(b[:2], 3)) for b in base} == {(CENTRE[0] - 3, CENTRE[1]), (CENTRE[0] + 3, CENTRE[1])}
    collar = bh.get(ch, "P1").z
    assert min(b[2] for b in base) == pytest.approx(collar - PIT, abs=0.1)
    assert sorted(tuple(p) for p in job["cuts"]["T1"]["inside"]) == [(CENTRE[0] - 3, CENTRE[1]), (CENTRE[0] + 3, CENTRE[1])]
    assert len(job["cuts"]["T1"]["outside"]) == 16
    assert [e["unit"] for e in job["column"]] == ["US 1", "US 7", "T1", "US 2", "US 3"]
    assert job["fills"] == {"US 7": "T1"} and {e["unit"]: e["kind"] for e in job["column"]}["T1"] == "cut"
    assert all(u != "US 7" and u != "T1" for s in job["series"] for u in s["units"])
    # the US 2 is above the fill in the order of the logs, or below: the pit cuts it all the same
    bh.set_order(ch, ["US 1", "US 2", "US 7", "US 3"])
    job, _plan = job_of(ch, tmp_path / "again")
    assert [e["unit"] for e in job["column"]] == ["US 1", "US 7", "T1", "US 2", "US 3"]


def test_a_hole_ending_in_the_fill_has_not_found_the_floor(tmp_path):
    doc, ch = project(tmp_path)
    text = logs(tmp_path).read_text().rstrip() + f"\nP3;{CENTRE[0]};{CENTRE[1] + 3};0;{TOP};US 1\nP3;{CENTRE[0]};{CENTRE[1] + 3};{TOP};1;US 7\n"
    bh.import_files(ch, [write(tmp_path, "u.csv", text)])
    implicit.mark_fill(ch, "US 7", "T1")
    job, plan = job_of(ch, tmp_path)
    assert len(job["cuts"]["T1"]["base"]) == 2
    assert any(n["code"] == "base_at_hole_end" and "P3" in n["holes"] for n in plan.notes)


def test_series_from_the_modes_and_by_hand(tmp_path):
    doc, ch = pit_site(tmp_path)
    ch.meta["geoarchaeology"]["surfaces"]["modes"] = {"US 1": "ground", "US 2": "depth", "US 3": "elevation"}
    series = implicit.series_of(ch)
    assert series == [{"name": "US 3", "space": "elevation", "units": ["US 3"]},
                      {"name": "ground", "space": "depth", "units": ["US 1", "US 2"]}]
    series = implicit.series_of(ch, {"us2": "quota"})
    assert {"name": "US 2", "space": "elevation", "units": ["US 2"]} in series
    assert not any(s["space"] == "depth" for s in series)  # the top alone is no series: it takes what is left
    (tmp_path / "x").mkdir()
    doc2, ch2 = pit_site(tmp_path / "x", surfaces=False)
    assert not sf.registry(ch2)
    names = {u for s in implicit.series_of(ch2) for u in s["units"]}  # no surfaces yet: choose_mode on the logs
    assert {"US 2", "US 3"} <= names


def test_the_outline_and_its_narrowest_part(tmp_path):
    doc, ch = pit_site(tmp_path)
    draw(ch, "7")
    job, plan = job_of(ch, tmp_path, cell=1.0)
    mask = np.load(job["cuts"]["T1"]["outline"])
    g = plan.grid
    assert mask.shape == (g.height, g.width) and mask.dtype == bool
    assert mask.sum() * g.cell ** 2 == pytest.approx(np.pi * RADIUS ** 2, rel=0.05)
    xy = g.centres()
    inside = np.hypot(xy[:, 0] - CENTRE[0], xy[:, 1] - CENTRE[1]) < RADIUS
    assert (mask.ravel() == inside).mean() > 0.998  # a cell belongs to it by its centre
    assert not any(n["code"] == "outline_narrow" for n in plan.notes)
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    d = ch.meta["drawings"]
    d["units"] = [UnitPolygon(box(X0 + 20, Y0 + 39.2, X0 + 80, Y0 + 40.2), us="7").to_dict()]  # a ditch 1 m wide
    job, plan = job_of(ch, tmp_path / "ditch", cell=1.0)
    note = next(n for n in plan.notes if n["code"] == "outline_narrow")
    assert note["unit"] == "T1" and note["cells"] == 1.0 and note["use"] == pytest.approx(1.0 / 3)
    assert "m" in implicit.warning_text(note)
    # a bump on a wide pit says nothing of its width
    wide = np.zeros((60, 60), bool)
    rr, cc = np.ogrid[:60, :60]
    wide[(rr - 30) ** 2 + (cc - 30) ** 2 <= 400] = True
    wide[9, 30] = True
    assert implicit._narrowest(wide) > 30
    two = np.zeros((20, 40), bool)
    two[9:11, 2:38] = True  # an even width keeps its cells
    assert implicit._narrowest(two) == 2.0
    three = np.zeros((20, 40), bool)
    three[9:12, 2:38] = True
    assert implicit._narrowest(three) == 3.0


def test_no_drawing_no_outline(tmp_path):
    doc, ch = pit_site(tmp_path)
    job, _plan = job_of(ch, tmp_path)
    assert job["cuts"]["T1"]["outline"] is None


def test_the_voxel_rule():
    notes = []
    assert implicit.voxel_step(1.0, 2.0, 10.0, notes) == pytest.approx(0.1) and not notes
    assert implicit.voxel_step(1.0, 0.6, 10.0, notes) == pytest.approx(0.06) and not notes
    dz = implicit.voxel_step(0.02, 2.0, 10.0, notes)
    assert dz == pytest.approx(10.0 / (implicit.MAX_LEVELS - 1))
    assert notes == [{"code": "voxels_clamped", "wanted": pytest.approx(0.002), "used": dz, "levels": implicit.MAX_LEVELS}]


def test_the_frame_of_the_job(tmp_path):
    doc, ch = pit_site(tmp_path)
    job, plan = job_of(ch, tmp_path)
    reg = sf.registry(ch)
    assert plan.grid.width * plan.grid.height <= implicit.MAX_COLUMNS
    assert plan.grid.x0 == reg["grid"]["x0"] and plan.grid.y0 == reg["grid"]["y0"]
    assert plan.grid.cell == pytest.approx(reg["grid"]["cell"] * plan.factor)
    assert job["dz"] <= 0.1 * TOP + 1e-9  # the thinnest unit is US 1
    n = int(np.ceil((job["zmax"] - job["zmin"]) / job["dz"] - 1e-9))
    assert n <= implicit.MAX_LEVELS
    bottoms = [b.z - b.bottom for b in bh.load(ch)]
    assert job["zmin"] < min(bottoms) + 1e-9 and job["zmax"] > max(b.z for b in bh.load(ch))
    dem = np.load(job["dem"])
    assert dem.shape == (plan.grid.height, plan.grid.width) and np.isfinite(dem).any()
    # no surfaces yet: a grid of its own
    (tmp_path / "s").mkdir()
    doc2, ch2 = pit_site(tmp_path / "s", surfaces=False)
    job2, plan2 = implicit.prepare(ch2, tmp_path / "job2")
    assert not plan2.on_surfaces and job2["grid"]["width"] > 1


def test_contacts_orientations_and_spot_heights(tmp_path):
    doc, ch = pit_site(tmp_path)
    contacts = write(tmp_path, "contatti.csv", "US;Est;Nord;Quota;Tipo\n"
                                               f"us 2;{X0 + 10};{Y0 + 10};100,5;\n"
                                               f"T1;{CENTRE[0]};{CENTRE[1]};99,0;\n"
                                               f"T1;{CENTRE[0] + 7};{CENTRE[1]};100,0;wall\n"
                                               f"US 99;{X0};{Y0};1,0;\n"
                                               f"US 2;;{Y0};1,0;\n")
    orientations = write(tmp_path, "giaciture.csv", "unit;x;y;z;dip;dip direction\n"
                                                    f"US 3;{X0 + 5};{Y0 + 5};98,5;30;90\n"
                                                    f"US 3;{X0 + 5};{Y0 + 6};98,5;200;90\n"
                                                    f"T1;{CENTRE[0] + 9};{CENTRE[1]};100;80;90\n"
                                                    f"US 7;{CENTRE[0]};{CENTRE[1]};100;10;0\n")
    report = implicit.set_data(ch, contacts, orientations)
    assert report["contacts"] == 4 and report["orientations"] == 3
    assert report["unknown"] == ["US 99"]
    assert any("riga 6" in n for n in report["notes"])
    assert (implicit.folder(ch) / "data" / "contacts.csv").exists()
    assert (implicit.folder(ch) / "data" / "orientations.csv").exists()
    ch.meta.setdefault("drawings", {"units": [], "quote": []})["quote"] = [
        {"x": X0 + 12, "y": Y0 + 12, "quota_q": 100.25, "us_q": "2", "unita_tipo_q": "US"},
        {"x": X0 + 13, "y": Y0 + 13, "quota_q": 100.0, "us_q": "", "unita_tipo_q": "US"},
        {"x": X0 + 14, "y": Y0 + 14, "quota_q": 100.0, "us_q": "55", "unita_tipo_q": "US"}]
    job, plan = job_of(ch, tmp_path)
    file_rows = [r for r in job["contacts"] if r[5] is None]
    assert [X0 + 10, Y0 + 10] in [[r[1], r[2]] for r in file_rows if r[0] == "US 2"]
    assert any(r[0] == "US 2" and r[3] == pytest.approx(100.25) for r in file_rows)
    assert [CENTRE[0], CENTRE[1], 99.0] in job["cuts"]["T1"]["base"]
    assert [CENTRE[0] + 7, CENTRE[1], 100.0, "wall"] in job["cuts"]["T1"]["base"]
    note = next(n for n in plan.notes if n["code"] == "unknown_units")
    assert note["units"] == "US 99" and plan.spots_ignored == 1  # a spot height on no unit is not data of the model
    us3 = next(o for o in job["orientations"] if o[0] == "US 3")
    assert us3[4:] == pytest.approx([np.sin(np.radians(30)), 0.0, np.cos(np.radians(30))], abs=1e-9)  # dips east
    wall = next(o for o in job["orientations"] if o[0] == "T1")
    assert wall[4] < 0  # into the cut, which lies west of the point
    assert not any(o[0] == "US 7" for o in job["orientations"])
    assert any(n["code"] == "orientation_ignored" for n in plan.notes)


def test_a_dip_direction_is_clockwise_from_north():
    assert implicit.dip_normal(90, 0) == pytest.approx([0, 1, 0], abs=1e-12)
    assert implicit.dip_normal(90, 90) == pytest.approx([1, 0, 0], abs=1e-12)
    assert implicit.dip_normal(45, 180) == pytest.approx([0, -np.sqrt(0.5), np.sqrt(0.5)], abs=1e-12)


def test_marks(tmp_path):
    doc, ch = pit_site(tmp_path)
    assert implicit.cuts_of(ch)["T1"] == {"kind": "cut", "fills": ["US 7"], "profile": "bowl"}
    assert implicit.cuts_of(ch)["US 7"] == {"kind": "fill", "fills": "T1", "profile": None}
    implicit.set_profile(ch, "T1", "flat")
    assert implicit.cuts_of(ch)["T1"]["profile"] == "flat"
    with pytest.raises(ValueError):
        implicit.set_profile(ch, "T1", "cone")
    with pytest.raises(ValueError):
        implicit.mark_fill(ch, "US 99", "T1")
    # a unit that is a fill already: the floor of its cut, as many times as it is asked
    assert implicit.mark_cut(ch, "US 7", "v") == "T1" and implicit.mark_cut(ch, "us7", "bowl") == "T1"
    assert implicit.cuts_of(ch)["T1"]["profile"] == "bowl"
    implicit.unmark(ch, "T1")
    assert implicit.cuts_of(ch) == {}
    # a unit of the logs marked as a cut: a cut has no material, the unit is its fill
    assert implicit.mark_cut(ch, "us 7", "v") == "Taglio US 7"
    assert implicit.cuts_of(ch)["Taglio US 7"] == {"kind": "cut", "fills": ["US 7"], "profile": "v"}
    implicit.unmark(ch, "US 7")
    assert implicit.cuts_of(ch)["Taglio US 7"]["fills"] == []
    # one cut in two spellings is one cut
    implicit.unmark(ch, "Taglio US 7")
    implicit.mark_cut(ch, "US 8", "flat")
    implicit.mark_fill(ch, "US 7", "us8")
    assert implicit.cuts_of(ch)["US 8"]["fills"] == ["US 7"] and "us8" not in implicit.cuts_of(ch)


def test_a_cut_named_in_the_language_of_the_interface(tmp_path, monkeypatch):
    doc, ch = pit_site(tmp_path)
    implicit.unmark(ch, "T1")
    monkeypatch.setattr(implicit, "t", lambda it, en: en)
    assert implicit.mark_cut(ch, "US 7") == "Cut US 7"


def test_small_rules_of_the_data(tmp_path):
    doc, ch = pit_site(tmp_path)
    # "auto" is no choice: the mode of the surfaces stays
    assert implicit.series_of(ch, {"US 2": "auto"}) == implicit.series_of(ch)
    # a USM spot height is not the US with that number; a US that is no unit is not said
    ch.meta["drawings"] = {"units": [], "quote": [
        {"x": X0 + 1, "y": Y0 + 1, "quota_q": 99.0, "us_q": "3", "unita_tipo_q": "USM"},
        {"x": X0 + 2, "y": Y0 + 2, "quota_q": 99.0, "us_q": "3", "unita_tipo_q": "US"},
        {"x": X0 + 3, "y": Y0 + 3, "quota_q": 99.0, "us_q": "40", "unita_tipo_q": "US"}]}
    job, plan = job_of(ch, tmp_path)
    spots = [r for r in job["contacts"] if r[5] is None]
    assert [(r[1], r[0]) for r in spots] == [(X0 + 2, "US 3")]
    assert plan.spots_ignored == 2 and not any(n["code"] == "unknown_units" for n in plan.notes)
    # an orientation on a cut without walls does nothing, and says so
    or_file = write(tmp_path, "o.csv", f"unit;x;y;z;dip;dip direction\nT1;{X0 + 50};{Y0 + 40};100;80;90\n")
    implicit.set_data(ch, orientations=or_file)
    job, plan = job_of(ch, tmp_path / "o")
    assert not job["orientations"] and {"code": "orientation_ignored", "unit": "T1"} in plan.notes
    # at most 12 groups
    assert job_of(ch, tmp_path / "f", folds=50)[0]["options"]["folds"] == 12


def test_a_hole_without_elevation_through_a_fill_is_inside(tmp_path):
    doc, ch = project(tmp_path, dem=False)
    text = logs(tmp_path).read_text()
    collars = "sondaggio;x;y;z\n" + "".join(sorted({
        f"{r[0]};{r[1]};{r[2]};100\n" for r in (ln.split(";") for ln in text.splitlines()[1:]) if r[0] != "P1"}))
    collars += f"P1;{CENTRE[0] - 3};{CENTRE[1]};\n"
    intervals = "\n".join(";".join(c for k, c in enumerate(ln.split(";")) if k not in (1, 2)) for ln in text.splitlines())
    bh.import_files(ch, [write(tmp_path, "c.csv", collars), write(tmp_path, "i.csv", intervals + "\n")])
    assert bh.get(ch, "P1").z is None
    implicit.mark_fill(ch, "US 7", "T1")
    data = implicit.gather(bh.load(ch), implicit.cuts_of(ch))
    assert [CENTRE[0] - 3, CENTRE[1]] in data.inside["T1"]
    assert [CENTRE[0] - 3, CENTRE[1]] not in data.outside["T1"] and len(data.base["T1"]) == 1


def test_a_drawing_or_a_spot_height_changed_makes_the_model_old(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    draw(ch, "7")
    implicit.build(ch)
    assert not implicit.is_stale(ch) and "cambiati" not in implicit.summary_text(ch)
    draw(ch, "7", 3.0, (CENTRE[0] + 20, CENTRE[1]))
    assert implicit.is_stale(ch) and "cambiati" in implicit.summary_text(ch)
    implicit.build(ch)
    assert not implicit.is_stale(ch)
    ch.meta["drawings"]["quote"].append({"x": X0, "y": Y0, "quota_q": 99.0, "us_q": "2", "unita_tipo_q": "US"})
    assert implicit.is_stale(ch)


def test_an_outline_past_the_grid(tmp_path):
    from shapely.geometry import box

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon

    doc, ch = pit_site(tmp_path)  # surfaces: their grid stays
    g = sf.grid_of(ch)
    east = g.x0 + g.width * g.cell
    ch.meta["drawings"] = {"quote": [], "units": [UnitPolygon(box(east - 20, Y0 + 34, east + 60, Y0 + 46), us="7").to_dict()]}
    job, plan = job_of(ch, tmp_path)
    note = next(n for n in plan.notes if n["code"] == "outline_clipped")
    assert note["drawn"] == pytest.approx(960.0) and note["kept"] == pytest.approx(240.0, rel=0.05)
    assert "960" in implicit.warning_text(note)
    ch.meta["drawings"]["units"] = [UnitPolygon(box(east + 50, Y0, east + 70, Y0 + 20), us="7").to_dict()]
    job, plan = job_of(ch, tmp_path / "out")
    assert job["cuts"]["T1"]["outline"] is None and {"code": "outline_outside", "unit": "T1"} in plan.notes
    # no surfaces yet: the grid goes as far as the outline
    (tmp_path / "s").mkdir()
    doc2, ch2 = pit_site(tmp_path / "s", surfaces=False)
    ch2.meta["drawings"] = {"quote": [], "units": [UnitPolygon(box(east - 20, Y0 + 34, east + 60, Y0 + 46), us="7").to_dict()]}
    job, plan = job_of(ch2, tmp_path / "wide")
    assert plan.grid.x0 + plan.grid.width * plan.grid.cell >= east + 60
    assert np.load(job["cuts"]["T1"]["outline"]).sum() * plan.grid.cell ** 2 == pytest.approx(960.0, rel=0.05)
    assert not any(n["code"] == "outline_clipped" for n in plan.notes)


def test_a_cell_finer_than_the_surfaces_is_built_as_asked(tmp_path, fake):
    import rasterio

    doc, ch = project(tmp_path)
    bh.import_files(ch, [logs(tmp_path)])
    sf.build(ch, cell=2.0)
    implicit.mark_fill(ch, "US 7", "T1")
    g = sf.grid_of(ch)
    job, plan = job_of(ch, tmp_path, cell=0.5)
    assert plan.grid.cell == 0.5 and plan.factor is None and (plan.grid.x0, plan.grid.y0) == (g.x0, g.y0)
    assert plan.grid.width == 4 * g.width and plan.grid.height == 4 * g.height
    assert implicit.model_frame(ch, bh.load(ch), 4.0).factor == 2  # a whole factor stays a coarsening
    assert implicit.model_frame(ch, bh.load(ch), 3.0).factor is None
    reg = implicit.build(ch, cell=0.5, folds=None)
    e = next(e for e in reg["surfaces"] if e["unit"] == "US 2")
    with rasterio.open(ch.data_dir / e["file"]) as src:
        assert src.width == 4 * g.width and np.isfinite(src.read(3)).any()  # band 3 resampled onto it
    rows = {r["unit"]: r for r in model.registry(ch)["units"]}
    assert rows["US 2"]["kriging_m3"] == pytest.approx(next(  # the whole area of the kriging: it has the DEM's cells
        t["volume_m3"] for t in sf.registry(ch)["thickness"] if t["unit"] == "US 2"), rel=0.02)


def test_the_volumes_compare_with_the_kriging_on_the_cells_with_a_dem(tmp_path):
    doc, ch = pit_site(tmp_path)
    _job, plan = job_of(ch, tmp_path, cell=1.0)
    assert plan.has_dem.sum() * plan.grid.cell ** 2 == 100.0 * 80.0  # not a cell around it


def test_the_volumes_of_the_two_methods_are_on_the_same_cells(tmp_path, fake):
    """The kriging stops at its margin around the boreholes, the implicit model has every cell with a DEM: here
    the boreholes are in the western half of the DEM."""
    rows = ["sondaggio;x;y;da;a;us"]
    for i, (x, y) in enumerate((5.0 + 11 * a, 5.0 + 20 * b) for a in range(4) for b in range(4)):
        rows += [f"S{i};{X0 + x};{Y0 + y};0;1;US 1", f"S{i};{X0 + x};{Y0 + y};1;2;US 2", f"S{i};{X0 + x};{Y0 + y};2;4;US 3"]
    doc, ch = project(tmp_path)
    bh.import_files(ch, [write(tmp_path, "t.csv", "\n".join(rows) + "\n")])
    sf.build(ch, buffer=2.0)
    job, plan = job_of(ch, tmp_path, cell=2.0)
    common = implicit._common_cells(ch, plan)
    assert 0 < common.sum() < plan.has_dem.sum()
    volumes = implicit._kriging_volumes(ch, plan, common, job["zmin"])
    area = float(common.sum()) * plan.grid.cell ** 2
    assert volumes["US 2"] == pytest.approx(area, rel=0.05)  # 1 m thick
    assert volumes["US 3"] / area > 2.0  # the last one down to the base of this model
    implicit.build(ch, cell=2.0, folds=None)
    rows = {r["unit"]: r for r in model.registry(ch)["units"]}
    assert rows["US 2"]["kriging_m3"] == pytest.approx(volumes["US 2"])
    assert rows["US 2"]["same_cells_m3"] < rows["US 2"]["volume_m3"]  # the DEM goes on past the boreholes


def test_the_boreholes_vote_for_a_layer_and_a_cut_takes_no_vote(tmp_path):
    doc, ch = pit_site(tmp_path, surfaces=False)
    implicit.mark_fill(ch, "US 7", "T1")
    assert not job_of(ch, tmp_path)[0]["absent"]  # the holes in the pit do not have US 2: the cut took it away
    rows = (tmp_path / "t.csv").read_text(encoding="utf-8").splitlines()
    bh.import_files(ch, [write(tmp_path, "u.csv", "\n".join(r for r in rows if not (r.startswith("S1;") and "US 2" in r)) + "\n")],
                    replace=True)
    job, plan = job_of(ch, tmp_path)
    assert list(job["absent"]) == ["US 2"]
    mask = np.load(job["absent"]["US 2"], allow_pickle=False)
    s1 = bh.get(ch, "S1")
    assert mask[int((plan.grid.y0 - s1.y) / plan.grid.cell), int((s1.x - plan.grid.x0) / plan.grid.cell)]
    assert 0 < mask.mean() < 0.3  # around that hole only


def test_spot_heights_left_out_are_told(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    ch.meta.setdefault("drawings", {"units": [], "quote": []})["quote"] = [
        {"x": X0 + 5.0, "y": Y0 + 5.0, "quota_q": 99.0, "us_q": "99", "unita_tipo_q": "US"}]
    implicit.build(ch, folds=None)
    assert "Quote lasciate fuori, su US che non sono unità del modello: 1." in implicit.summary_text(ch)


# ------------------------------------------------------------------ what comes back
def test_the_results(tmp_path, fake):
    import rasterio

    doc, ch = pit_site(tmp_path)
    draw(ch, "7")
    kriged = dict(sf.tops(ch))
    reg = implicit.build(ch)
    assert not list(ch.data_dir.glob("geoarchaeology/implicit_job_*"))  # the temporary folder went
    assert reg["made"] and reg["stale"] is False and reg["grid"]["width"] > 0
    assert reg["versions"] == {"loopstructural": "0.0-fake", "python": "3"}
    assert {"T1", "US 7"} <= set(reg["cuts"]) and reg["cut_results"]["T1"]["edge"] == "drawn"
    assert {"code": "few_holes", "series": "ground"} in reg["warnings"]
    loo = {r["unit"]: r for r in reg["loo"]}
    assert loo["US 2"]["rmse"] == 0.25 and loo["US 2"]["n"] == 4
    assert loo["US 2"]["kriging_rmse"] == pytest.approx(next(
        e for e in sf.registry(ch)["contacts"] if e["unit"] == "US 2")["loocv_rmse"])
    # the surfaces: products, three bands
    entries = reg["surfaces"]
    assert [e["unit"] for e in entries] == ["US 1", "US 7", "US 2", "US 3"]
    e = next(e for e in entries if e["unit"] == "US 2")
    assert sf.entry(ch, e["id"]) == e and e["kind"] == "surface" and e["method"] == "implicit"
    path = ch.data_dir / e["file"]
    assert path.parent == implicit.folder(ch) and re.fullmatch(r"top_2_US_2\.tif", path.name)
    plan = sf.Grid.from_dict(reg["grid"])
    with rasterio.open(path) as src:
        assert src.count == 3 and src.dtypes == ("float32",) * 3 and src.crs is not None
        assert src.width == plan.width and np.isnan(src.nodata)
        assert src.transform.a == pytest.approx(plan.cell) and src.transform.c == pytest.approx(plan.x0)
        top, variance, difference = src.read(1), src.read(2), src.read(3)
    f = model.registry(ch)["factor"]
    other = model.coarsen(kriged["US 2"], f)
    ok = np.isfinite(difference)
    assert ok.any() and np.allclose(difference[ok], (top - other)[ok], atol=1e-3)
    assert np.isfinite(variance).any()
    # the solids: the registry of the kriging model
    m = model.registry(ch)
    assert m["method"] == "implicit" and model.has_model(ch) and model.method(ch) == "implicit"
    assert (ch.data_dir / m["files"]["obj"]).exists() and (ch.data_dir / m["files"]["display"]).exists()
    rows = {r["unit"]: r for r in m["units"]}
    assert rows["US 2"]["ply"] and (ch.data_dir / rows["US 2"]["ply"]).exists()
    assert rows["US 2"]["kriging_m3"] == pytest.approx(next(
        e for e in sf.registry(ch)["thickness"] if e["unit"] == "US 2")["volume_m3"], rel=0.02)
    assert rows["T1"]["kind"] == "cut" and not rows["T1"].get("ply")
    assert (rows["T1"]["range_min"], rows["T1"]["range_max"]) == (50.0, 100.0) and rows["T1"]["volume_m3"] == 80.0
    json.dumps(ch.meta["geoarchaeology"])  # all of it goes in the project file
    # the volumes, the new columns at the end
    with open(ch.data_dir / m["files"]["csv"], encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
        assert list(table[0])[-4:] == ["method", "kriging_m3", "range_min", "range_max"]
    assert list(table[0])[:len(model.VOLUME_COLUMNS)] == list(model.VOLUME_COLUMNS)
    row = next(r for r in table if r["unit"] == "T1")
    assert float(row["range_min"]) == 50.0 and float(row["range_max"]) == 100.0 and row["method"] == "implicit"
    # the rest of the plugin reads it as a model
    assert model.display_mesh(ch)[1] == 4
    new, report = model.to_chunk(doc, ch)
    assert report["faces"] > 0 and new.label.startswith("Unità geoarcheologiche")
    text = implicit.summary_text(ch)
    assert "interpretazione" in text and "T1, il riempimento: da 50,0 a 100,0 m³" in text and "kriging" in text
    assert "T1, il riempimento: da 50 a 100 m³" in sys.modules["ppg_geoarchaeology"].GeoarchaeologyPlugin.model_text(m)
    assert "implicit model" in export.summary(ch)


def test_a_failure_leaves_everything_as_it_was(tmp_path, fake, monkeypatch):
    doc, ch = pit_site(tmp_path)
    model.build(ch)
    implicit.build(ch)
    implicit.set_data(ch, write(tmp_path, "c.csv", f"us;x;y;z\nUS 2;{X0 + 10};{Y0 + 10};100\n"))

    def snapshot():
        return {p: p.read_bytes() for d in (model.folder(ch), implicit.folder(ch)) for p in sorted(d.rglob("*")) if p.is_file()}

    before = snapshot()
    meta = json.dumps(ch.meta["geoarchaeology"], sort_keys=True)
    monkeypatch.setenv("PPG_GEOARCH_FAKE", "fail")
    with pytest.raises(RuntimeError, match="does not solve"):
        implicit.build(ch)
    monkeypatch.delenv("PPG_GEOARCH_FAKE")
    assert snapshot() == before and json.dumps(ch.meta["geoarchaeology"], sort_keys=True) == meta
    assert not list((ch.data_dir / "geoarchaeology").glob("implicit_job_*"))
    # also when it fails after the worker, writing the results
    monkeypatch.setattr(implicit, "_model", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        implicit.build(ch)
    monkeypatch.undo()
    assert snapshot() == before and json.dumps(ch.meta["geoarchaeology"], sort_keys=True) == meta


def test_a_swap_that_fails_half_way_puts_the_folders_back(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    (a / "old").write_text("a")
    new_a = tmp_path / "new_a"
    new_a.mkdir()
    (new_a / "new").write_text("n")
    with pytest.raises(OSError):
        implicit._swap([(new_a, a), (tmp_path / "missing", b)])
    assert (a / "old").read_text() == "a" and not (a / "new").exists()


def test_the_data_files_survive_a_build(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    implicit.set_data(ch, write(tmp_path, "c.csv", f"us;x;y;z\nUS 2;{X0 + 10};{Y0 + 10};100\n"))
    implicit.build(ch)
    implicit.build(ch)
    assert (implicit.folder(ch) / "data" / "contacts.csv").exists()
    assert len(list(implicit.folder(ch).glob("*.tif"))) == 4


def test_a_kriging_model_after_it_replaces_it_and_the_surfaces_stay(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    implicit.build(ch)
    assert model.method(ch) == "implicit"
    assert section.profile(ch, [(X0 + 5, Y0 + 40), (X0 + 95, Y0 + 40)])["method"] == "implicit"
    model.build(ch)
    assert model.method(ch) == "kriging" and "method" not in model.registry(ch)
    assert implicit.registry(ch)["surfaces"]
    assert all((ch.data_dir / e["file"]).exists() for e in implicit.registry(ch)["surfaces"])
    assert "method" not in section.profile(ch, [(X0 + 5, Y0 + 40), (X0 + 95, Y0 + 40)])
    assert not implicit.active_surfaces(ch)
    # a kriging model built without any implicit one keeps its own volume columns
    with open(ch.data_dir / model.registry(ch)["files"]["csv"], encoding="utf-8") as fh:
        assert next(csv.reader(fh)) == list(model.VOLUME_COLUMNS)


def test_kriging_is_not_touched(tmp_path, fake):
    """The surfaces, the model and the sections of a project byte by byte with and without the data of
    an implicit model."""
    doc, ch = project(tmp_path)
    bh.import_files(ch, [logs(tmp_path)])
    sf.build(ch)
    model.build(ch)
    coords = [(X0 + 5, Y0 + 40), (X0 + 95, Y0 + 40)]
    profile = section.profile(ch, coords)
    entries = list(sf.entries(ch))
    tif = {e["id"]: (ch.data_dir / e["file"]).read_bytes() for e in entries}
    implicit.mark_fill(ch, "US 7", "T1")  # marks alone change nothing
    assert sf.entries(ch) == entries
    again = section.profile(ch, coords)
    assert set(again) == set(profile) and all(np.array_equal(a[1], b[1], equal_nan=True)
                                              for a, b in zip(again["tops"], profile["tops"]))
    implicit.build(ch)
    assert all((ch.data_dir / e["file"]).read_bytes() == tif[e["id"]] for e in entries)
    assert [e["id"] for e in sf.registry(ch)["contacts"] + sf.registry(ch)["thickness"]] == [e["id"] for e in entries]


def test_every_warning_of_the_engine_has_its_text():
    source = (PACKAGE / "implicit_engine.py").read_text(encoding="utf-8")
    codes = set(re.findall(r'"code": "(\w+)"', source))
    assert len(codes) >= 14 and codes <= set(implicit.WARNINGS)
    for code, (it, en) in implicit.WARNINGS.items():
        assert it and en and it != en, code
        assert sorted(re.findall(r"{(\w+)}", it)) == sorted(re.findall(r"{(\w+)}", en)), code
    text = implicit.warning_text({"code": "edge_estimated", "unit": "T1", "limit": 4.0, "effect": "x"})
    assert "sovrastimato o sottostimato" in text
    assert "ipotesi" in implicit.warning_text({"code": "floor_unmeasured", "unit": "T1"})
    assert implicit.warning_text({"code": "?"}) == "?"


def test_the_exports_have_the_implicit_surfaces(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    implicit.build(ch)
    out = tmp_path / "e" / "geo.gpkg"
    assert export.export_gpkg(ch, out) == out
    pid = implicit.registry(ch)["surfaces"][0]["id"]
    assert (tmp_path / "e" / f"geo_{pid}.tif").exists() and (tmp_path / "e" / "geo_volumi.csv").exists()
    rasters, _vectors = export.qgis_entries(ch)
    assert any("implicito" in r["name"] for r in rasters)
    assert export.export_raster(ch, pid, tmp_path / "one.tif").exists()
    assert sf.load(ch, pid, 2)[0].shape == sf.load(ch, pid, 1)[0].shape


def test_the_command_line(tmp_path, fake, capsys):
    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = project(tmp_path)
    plugin = GeoarchaeologyPlugin()
    plugin.activate(Context("geoarchaeology", tmp_path / "plugin"))
    p = str(doc.path)
    assert plugin.command("import", p, str(logs(tmp_path))) == 0
    assert plugin.command("surfaces", p) == 0
    contacts = write(tmp_path, "c.csv", f"us;x;y;z\nUS 2;{X0 + 10};{Y0 + 10};100\n")
    capsys.readouterr()
    assert plugin.command("implicit-data", p, "--contacts", str(contacts)) == 0
    assert "1 contatto letto" in capsys.readouterr().out
    assert plugin.command("implicit", p, "--cut", "T1=flat", "--fill", "us7=T1", "--folds", "4", "--series", "US2=depth") == 0
    out = capsys.readouterr().out
    assert "interpretazione" in out and "T1, il riempimento: da 50,0 a 100,0 m³" in out
    reg = ppg.Document.open(p).chunk.meta["geoarchaeology"]
    assert reg["implicit"]["cuts"]["T1"]["profile"] == "flat" and reg["model"]["method"] == "implicit"
    assert plugin.command("implicit", p, "--cut", "T1=cone") == 1
    assert plugin.command("list", p) == 0 and "implicit model" in capsys.readouterr().out
    plugin.command("implicit-status", p)
    assert "interpretazione" in capsys.readouterr().out
    assert plugin.command("remove", p, "implicit") == 0
    reg = ppg.Document.open(p).chunk.meta["geoarchaeology"]
    assert "surfaces" not in reg["implicit"] and "model" not in reg and reg["implicit"]["cuts"]["T1"]
    capsys.readouterr()
    assert plugin.command("help") == 0 and "implicit-data" in capsys.readouterr().out


def test_the_section_says_it_is_implicit(tmp_path, fake):
    doc, ch = pit_site(tmp_path)
    implicit.build(ch)
    e = section.build(ch, [(X0 + 5, Y0 + 40), (X0 + 95, Y0 + 40)], name="Prova")
    assert e["method"] == "implicit" and (ch.data_dir / e["files"]["png"]).exists()
    prof = section.profile(ch, [(X0 + 5, Y0 + 40), (X0 + 95, Y0 + 40)])
    assert [u for u, _v in prof["tops"]] == ["US 1", "US 7", "US 2", "US 3"]


# ------------------------------------------------------------------ the pane
def _window(tmp_path, monkeypatch, doc):
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    qapp = QApplication.instance() or QApplication([])
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    monkeypatch.setattr(w, "_warn", lambda text: None)
    monkeypatch.setattr(w, "_info", lambda title, text: None)
    return qapp, w


def test_the_pane(tmp_path, monkeypatch, fake):
    pytest.importorskip("PySide6")
    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = pit_site(tmp_path)
    draw(ch, "7")
    doc.save()
    qapp, w = _window(tmp_path, monkeypatch, doc)
    plugin = GeoarchaeologyPlugin()
    messages = []
    try:
        plugin.activate(Context("geoarchaeology", tmp_path / "plugin", window=w))
        pane = plugin.pane
        assert pane.tabs.indexOf(pane.t_implicit) >= 0
        assert pane.t_cuts.rowCount() == 2 and pane.marks == ["T1", "US 7"]
        assert pane.t_cuts.item(0, 3).text() == "US 7"
        combo = pane.t_cuts.cellWidget(0, 2)
        combo.setCurrentIndex(combo.findData("flat"))
        assert implicit.cuts_of(ch)["T1"]["profile"] == "flat"
        assert pane.b_implicit.isEnabled() and "non è ancora" in pane.l_implicit.toPlainText()
        monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
            lambda parent, title, fields, intro="": {"cell": None, "folds": 3, "series": "", "explode": None}))
        monkeypatch.setattr(plugin, "_run", lambda title, job, done=None: done(job(None)) if done else job(None))
        monkeypatch.setattr(plugin, "_message", lambda kind, text, *a: messages.append(text))
        plugin.show_pane()
        w.show_view(w.ortho_view)
        pane.b_implicit.click()
        assert model.method(ch) == "implicit" and "interpretazione" in messages[-1]
        assert "interpretazione" in pane.l_implicit.toPlainText() and "da 50,0 a 100,0" in pane.l_implicit.toPlainText()
        assert pane.b_model_view.isEnabled() and w.model_view.paths == {"model": str(ch.data_dir / model.registry(ch)["files"]["display"])}
        ids = [p["id"] for p in pane.products]
        pid = next(i for i in ids if implicit.is_product(i))
        row = ids.index(pid)
        pane.tabs.setCurrentWidget(pane.t_products)
        pane.t_products.selectRow(row)
        assert pane.t_products.item(row, 1).text().startswith("implicito")
        assert pane.c_band.itemText(2) == "Implicito meno kriging"
        w._show_raster("dem")
        pane.c_band.setCurrentIndex(2)
        pane.b_overlay.click()
        assert "implicito meno kriging" in pane.l_limits.text()
        pane.t_cuts.selectRow(1)
        assert pane.current_mark() == "US 7"
        pane.b_unmark.click()
        assert "US 7" not in implicit.cuts_of(ch)
        monkeypatch.setattr("pyarchinit_photogrammetry.gui.dialogs.choice", lambda *a, **k: 0)
        plugin.remove_dialog("product", pid)
        assert not implicit.registry(ch).get("surfaces") and sf.registry(ch)["contacts"]
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
        qapp.processEvents()


# ------------------------------------------------------------------ the real worker
@pytest.fixture
def real(monkeypatch):
    python = loop_python()
    if python is None or not python.exists():
        pytest.skip("no Python with LoopStructural (PPG_TEST_LOOP)")
    monkeypatch.setattr(implicit, "WORKER", (python, None))


def _tilted(tmp_path: Path):
    """16 holes, US 1 to 0.5 m and US 2 to 2 m under a ground rising to the north-east, so the contacts,
    taken by elevation, rise about half a degree to the east."""
    rows = ["sondaggio;x;y;da;a;us"]
    for i, (x, y) in enumerate((10.0 + 26 * a, 10.0 + 20 * b) for a in range(4) for b in range(4)):
        rows += [f"S{i};{X0 + x};{Y0 + y};0;0,5;US 1", f"S{i};{X0 + x};{Y0 + y};0,5;2;US 2",
                 f"S{i};{X0 + x};{Y0 + y};2;6;US 3"]
    doc, ch = project(tmp_path)
    bh.import_files(ch, [write(tmp_path, "t.csv", "\n".join(rows) + "\n")])
    sf.build(ch)
    return doc, ch


def test_a_rare_unit_covers_about_what_the_kriging_gives(tmp_path, real):
    """US 3 in 4 holes of 25, on one side of the site."""
    rows = ["sondaggio;x;y;da;a;us"]
    for i, (a, b) in enumerate((a, b) for a in range(5) for b in range(5)):
        x, y = X0 + 8 + 21 * a, Y0 + 6 + 17 * b
        rare = a < 2 and b < 2
        rows += [f"S{i};{x};{y};0;0,5;US 1"]
        rows += [f"S{i};{x};{y};0,5;1;US 2", f"S{i};{x};{y};1;1,5;US 3"] if rare else [f"S{i};{x};{y};0,5;1,5;US 2"]
        rows += [f"S{i};{x};{y};1,5;5;US 4"]
    doc, ch = project(tmp_path)
    bh.import_files(ch, [write(tmp_path, "t.csv", "\n".join(rows) + "\n")])
    sf.build(ch)
    reg = implicit.build(ch, cell=2.0, folds=None)
    rows = {r["unit"]: r for r in model.registry(ch)["units"]}
    thick = sf.read_band(ch.data_dir / next(e for e in sf.registry(ch)["thickness"] if e["unit"] == "US 3")["file"], 1)[0]
    grid = sf.grid_of(ch)
    kriging_area = float(((thick > 0.01) & np.isfinite(sf.dem_on_grid(ch, grid))).sum() * grid.cell ** 2)
    assert rows["US 3"]["area_m2"] == pytest.approx(kriging_area, rel=0.25)
    assert reg["absent"]["US 3"] == rows["US 3"]["absent_cells"] > 0
    assert "i sondaggi senza l'unità prevalgono" in implicit.summary_text(ch)


def test_a_dip_against_the_contact_does_not_turn_the_model_over(tmp_path, real):
    doc, ch = _tilted(tmp_path)
    series = {"US 2": "elevation", "US 3": "elevation"}
    implicit.build(ch, cell=4.0, folds=None, overrides=series)
    plain = {r["unit"]: r["volume_m3"] for r in model.registry(ch)["units"]}
    b = bh.get(ch, "S0")
    z = b.z - next(iv.top for iv in b.intervals if iv.unit == "US 3")
    implicit.set_data(ch, orientations=write(tmp_path, "o.csv", "unit;x;y;z;dip;dip direction\n"
                                                                f"US 3;{b.x};{b.y};{z};10;90\n"))
    reg = implicit.build(ch, cell=4.0, folds=None, overrides=series)
    rows = {r["unit"]: r["volume_m3"] for r in model.registry(ch)["units"]}
    for unit, volume in plain.items():
        assert rows[unit] == pytest.approx(volume, rel=0.05), unit
    tops = {e["unit"]: sf.read_band(ch.data_dir / e["file"], 1)[0] for e in reg["surfaces"]}
    assert np.nanmedian(tops["US 3"]) < np.nanmedian(tops["US 2"]) < np.nanmedian(tops["US 1"])


def test_a_dip_that_agrees_still_acts(tmp_path, real):
    """A line of holes west to east says nothing of the north-south tilt: the measure does."""
    rows = ["sondaggio;x;y;da;a;us"]
    for i, x in enumerate(np.linspace(10, 90, 6)):
        rows += [f"S{i};{X0 + x};{CENTRE[1]};0;1;US 1", f"S{i};{X0 + x};{CENTRE[1]};1;3;US 2", f"S{i};{X0 + x};{CENTRE[1]};3;6;US 3"]
    doc, ch = project(tmp_path)
    bh.import_files(ch, [write(tmp_path, "t.csv", "\n".join(rows) + "\n")])
    b = bh.get(ch, "S2")
    implicit.set_data(ch, orientations=write(tmp_path, "o.csv", "unit;x;y;z;dip;dip direction\n"
                                                                f"US 3;{b.x};{b.y};{b.z - 3};5;0\n"))
    reg = implicit.build(ch, cell=4.0, folds=None, overrides={"US 2": "elevation", "US 3": "elevation"})
    assert not any(w["code"] == "orientation_dropped" for w in reg["warnings"])
    e = next(e for e in reg["surfaces"] if e["unit"] == "US 3")
    north, south = (float(bh.raster_values(ch.data_dir / e["file"], b.x, b.y + d)[0]) for d in (12.0, -12.0))
    assert north - south == pytest.approx(-24.0 * np.tan(np.radians(5.0)), abs=0.8)


def test_contacts_and_spot_heights_shape_the_model(tmp_path, real):
    doc, ch = _tilted(tmp_path)
    series = {"US 2": "elevation", "US 3": "elevation"}
    x, y = X0 + 49.0, Y0 + 40.0  # between the holes
    before = implicit.build(ch, cell=2.0, folds=None, overrides=series)
    e = next(e for e in before["surfaces"] if e["unit"] == "US 3")
    level = float(bh.raster_values(ch.data_dir / e["file"], x, y)[0])
    spot = level - 0.8
    ch.meta["drawings"] = {"units": [], "quote": [{"x": x, "y": y, "quota_q": spot, "us_q": "3", "unita_tipo_q": "US"}]}
    implicit.set_data(ch, write(tmp_path, "c.csv", f"us;x;y;z\nUS 3;{x + 2};{y};{spot}\n"))
    after = implicit.build(ch, cell=2.0, folds=None, overrides=series)
    e = next(e for e in after["surfaces"] if e["unit"] == "US 3")
    moved = float(bh.raster_values(ch.data_dir / e["file"], x, y)[0])
    assert spot < moved < level - 0.1  # towards them: the field is smooth, two points do not make a dent of their own
    assert implicit.registry(ch)["spots_ignored"] == 0


def test_a_cut_with_walls_keeps_its_undercut(tmp_path, real):
    """A bottle pit: a mouth of 4 m, 7 m at 2 m down, the walls seen on a section."""
    from test_geoarchaeology import ground

    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    rows = ["sondaggio;x;y;da;a;us"]
    for i, (dx, dy) in enumerate((a, b) for a in (-30.0, -15.0, 15.0, 30.0) for b in (-25.0, 25.0)):
        rows += [f"S{i};{CENTRE[0] + dx};{CENTRE[1] + dy};0;2;US 1", f"S{i};{CENTRE[0] + dx};{CENTRE[1] + dy};2;6;US 2"]
    for i, (dx, dy) in enumerate(((0, 0), (2, 0), (-2, 0), (0, 2), (0, -2))):
        rows += [f"P{i};{CENTRE[0] + dx};{CENTRE[1] + dy};0;3,5;US 7", f"P{i};{CENTRE[0] + dx};{CENTRE[1] + dy};3,5;6;US 2"]
    doc, ch = project(tmp_path)
    bh.import_files(ch, [write(tmp_path, "t.csv", "\n".join(rows) + "\n")])
    implicit.mark_fill(ch, "US 7", "T1")
    draw(ch, "7", 4.0)
    a = 2 * np.pi * np.arange(24) / 24
    g = float(ground(CENTRE[0] - X0, CENTRE[1] - Y0))
    table = ["us;x;y;z;tipo"]
    for r, depth, step in ((7.0, 2.0, 1), (5.5, 0.9, 1), (5.5, 3.2, 1), (5.5, 3.5, 4)):
        table += [f"T1;{CENTRE[0] + r * np.cos(v)};{CENTRE[1] + r * np.sin(v)};{g - depth};wall" for v in a[::step]]
    implicit.set_data(ch, write(tmp_path, "w.csv", "\n".join(table).replace(".", ",") + "\n"))
    reg = implicit.build(ch, cell=1.0, folds=None)
    assert reg["cut_results"]["T1"]["field"] is True
    rows_ = {r["unit"]: r for r in model.registry(ch)["units"]}
    mesh = read_mesh(ch.data_dir / rows_["US 7"]["ply"])
    r = np.hypot(mesh.xyz[:, 0] - CENTRE[0], mesh.xyz[:, 1] - CENTRE[1])
    assert r.max() > 5.5  # past the mouth
    assert mesh.xyz[r > 5.0, 2].max() < g - 0.3  # under a roof: the layer stays over it


def test_a_drawn_pit_with_the_real_worker(tmp_path, real):
    doc, ch = pit_site(tmp_path)
    implicit.set_profile(ch, "T1", "flat")
    draw(ch, "7")
    reg = implicit.build(ch, cell=CELL * 2, folds=None)
    m = model.registry(ch)
    rows = {r["unit"]: r for r in m["units"]}
    fill = rows["US 7"]["volume_m3"]
    assert rows["T1"]["range_min"] <= fill <= rows["T1"]["range_max"]  # the range of the fill, under the topsoil
    assert rows["T1"]["range_max"] < max(rows["T1"]["cut_range"].values())
    truth = np.pi * RADIUS ** 2 * (PIT - TOP)
    assert fill == pytest.approx(truth, rel=0.25)
    assert reg["cut_results"]["T1"]["edge"] == "drawn" and reg["cut_results"]["T1"]["field"] is False
    assert model.has_model(ch) and rows["US 2"].get("ply")
    # the section across the pit shows the pit: the US 2 is not there, the floor is
    line = [(CENTRE[0] - 30, CENTRE[1]), (CENTRE[0] + 30, CENTRE[1])]
    prof = section.profile(ch, line)
    assert prof["method"] == "implicit"
    tops = dict(prof["tops"])
    mid = np.argmin(np.abs(prof["distance"] - 30.0))
    far = np.argmin(np.abs(prof["distance"] - 2.0))
    collar = float(prof["ground"][mid])
    assert tops["US 2"][mid] == pytest.approx(collar - PIT, abs=0.5)  # the top of what is under the fill: its floor
    assert tops["US 3"][far] == pytest.approx(prof["ground"][far] - SECOND, abs=0.4)
    assert tops["US 7"][mid] == pytest.approx(collar - TOP, abs=0.4)
    e = section.build(ch, line, name="Taglio")
    assert (ch.data_dir / e["files"]["png"]).exists()
    # a unit is a closed solid in the new chunk
    new, report = model.to_chunk(doc, ch)
    assert report["faces"] > 0
    out = tmp_path / "out" / "g.gpkg"
    export.export_gpkg(ch, out)
    assert (tmp_path / "out" / f"g_{reg['surfaces'][0]['id']}.tif").exists()
