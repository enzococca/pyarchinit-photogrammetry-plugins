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
    d["units"] = [UnitPolygon(box(X0 + 20, Y0 + 39.5, X0 + 80, Y0 + 40.5), us="7").to_dict()]  # a ditch 1 m wide
    job, plan = job_of(ch, tmp_path / "ditch", cell=1.0)
    note = next(n for n in plan.notes if n["code"] == "outline_narrow")
    assert note["unit"] == "T1" and note["use"] < 1.0 and note["width"] <= 1.0
    assert "m" in implicit.warning_text(note)
    # a bump on a wide pit says nothing of its width
    wide = np.zeros((60, 60), bool)
    rr, cc = np.ogrid[:60, :60]
    wide[(rr - 30) ** 2 + (cc - 30) ** 2 <= 400] = True
    wide[9, 30] = True
    assert implicit._narrowest(wide) > 30


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
    assert "US 99" in note["units"] and "55" in note["units"]
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
    implicit.unmark(ch, "T1")
    assert implicit.cuts_of(ch) == {}
    # a unit of the logs marked as a cut: a cut has no material, the unit is its fill
    assert implicit.mark_cut(ch, "us 7", "v") == "Taglio US 7"
    assert implicit.cuts_of(ch)["Taglio US 7"] == {"kind": "cut", "fills": ["US 7"], "profile": "v"}
    implicit.unmark(ch, "US 7")
    assert implicit.cuts_of(ch)["Taglio US 7"]["fills"] == []


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
        e for e in sf.registry(ch)["thickness"] if e["unit"] == "US 2")["volume_m3"])
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
    assert "interpretazione" in text and "T1: da 50,0 a 100,0 m³" in text and "kriging" in text
    assert "T1: da 50 a 100 m³" in sys.modules["ppg_geoarchaeology"].GeoarchaeologyPlugin.model_text(m)
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
    assert "interpretazione" in out and "T1: da 50,0 a 100,0 m³" in out
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
def test_a_drawn_pit_with_the_real_worker(tmp_path, monkeypatch):
    python = loop_python()
    if python is None or not python.exists():
        pytest.skip("no Python with LoopStructural (PPG_TEST_LOOP)")
    monkeypatch.setattr(implicit, "WORKER", (python, None))
    doc, ch = pit_site(tmp_path)
    implicit.set_profile(ch, "T1", "flat")
    draw(ch, "7")
    reg = implicit.build(ch, cell=CELL * 2, folds=None)
    m = model.registry(ch)
    rows = {r["unit"]: r for r in m["units"]}
    fill = rows["US 7"]["volume_m3"]
    assert rows["T1"]["range_min"] * 0.9 <= fill <= rows["T1"]["range_max"] * 1.1
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
