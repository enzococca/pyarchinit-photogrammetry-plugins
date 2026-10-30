"""The plugin's side of a scenario, with the fake worker (numpy diffusion, no Landlab): the DEM of a
chunk resampled to the simulation cell and the results back on the DEM's grid as GeoTIFFs, the
registry, the final surface as a version of the DEM (the active DEM untouched), the comparison with
the DEM of another chunk, the removal of a run alone, the estimate, the command line through the
program's ``ppg plugin``, the pane in the program's window. With ``PPG_TEST_LANDLAB`` one run goes
through the real worker and Landlab."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import pyarchinit_photogrammetry as ppg  # noqa: E402
from landscape_samples import landlab_python, tell  # noqa: E402
from ppg_landscape import landscape  # noqa: E402

FAKE = Path(__file__).with_name("fake_worker.py")
CRS = "EPSG:32633"
CELL = 0.5
N = 60
ORIGIN = (500000.0, 4500000.0)
LANDLAB = landlab_python()


@pytest.fixture
def fake(monkeypatch):
    """The fake worker for every run of the plugin."""
    monkeypatch.setenv("PPG_LANDSCAPE_FAKE", "")
    monkeypatch.setattr(landscape, "WORKER", (Path(sys.executable), FAKE))


def _write_dem(path: Path, z: np.ndarray, cell: float = CELL, origin=ORIGIN, crs: str | None = CRS) -> None:
    import rasterio
    from rasterio.transform import from_origin

    tr = from_origin(origin[0], origin[1] + z.shape[0] * cell, cell, cell)
    with rasterio.open(path, "w", driver="GTiff", height=z.shape[0], width=z.shape[1], count=1, dtype="float32",
                       crs=crs, transform=tr, nodata=float("nan")) as dst:
        dst.write(z.astype(np.float32), 1)


def _project(tmp_path: Path, later: bool = True):
    """A project with a chunk "Fase 1" whose DEM is a cone (a tell) on a plane, 60 x 60 cells of 0.5 m,
    the first two rows without a value; and a chunk "Fase 2" whose DEM is the same tell worn down."""
    z, _r = tell(n=N, cell=CELL)
    z[:2, :] = np.nan
    doc = ppg.Document()
    a = doc.add_chunk("Fase 1")
    doc.save(tmp_path / "scavo.ppg")
    a.crs = CRS
    _write_dem(a.data_dir / "dem.tif", z)
    a._set_product("dem", a.data_dir / "dem.tif")
    if later:
        b = doc.add_chunk("Fase 2")
        b.crs = CRS
        worn, _r = tell(n=N, cell=CELL, height=3.6, radius=11.0)
        _write_dem(b.data_dir / "dem.tif", worn)
        b._set_product("dem", b.data_dir / "dem.tif")
    doc.chunk = a
    doc.save()
    return doc, a


def _read(path: Path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), src.transform, src.crs, src.dtypes[0], src.nodata


def _digest(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


SCENARIO = {"preset": "custom", "years": 10, "snapshots": [5], "cell": 1.0, "boundary": "closed",
            "diffusion": {"D": 0.01}, "fluvial": None}


# ------------------------------------------------------------------ a run
def test_a_run_comes_back_onto_the_grid_of_the_dem(tmp_path, fake):
    doc, ch = _project(tmp_path)
    dem = ch.dem_path
    before = _digest(dem)
    seen = []
    e = landscape.run(ch, SCENARIO, lambda f, m: seen.append((f, m)))
    assert e["years"] == 10 and e["cell"] == 1.0 and e["grid"] == [30, 30] and e["preset"] == "custom"
    assert e["source"]["version"] == "active" and e["params"]["diffusion"] == {"D": 0.01}
    assert e["params"]["fluvial"] is None and e["balance"]["router"] == "fake" and e["versions"]["landlab"] == "0.0-fake"
    assert landscape.entry(ch, e["id"]) == e and ch.meta["landscape"]["runs"] == [e]
    assert set(e["files"]) == {"final", "dz", "area", "snapshots", "balance"} and list(e["files"]["snapshots"]) == ["5"]
    z, tr, crs, _dt, _nd = _read(dem)
    for layer in ("final", "dz", "area", "snap:5"):
        values, t2, c2, dtype, nodata = _read(landscape.path_of(ch, e, layer))
        assert values.shape == z.shape and t2 == tr and c2 == crs and dtype == "float32" and np.isnan(nodata), layer
        assert np.array_equal(np.isnan(values), np.isnan(z)), layer  # the cells without a value stay so
    final = _read(landscape.path_of(ch, e, "final"))[0]
    dz = _read(landscape.path_of(ch, e, "dz"))[0]
    ok = np.isfinite(z)
    np.testing.assert_allclose(final[ok] - z[ok], dz[ok], atol=1e-5)
    centre = N // 2
    assert dz[centre, centre] < -1e-3  # the summit wears down
    assert dz[ok].max() > 1e-4  # and the foot fills
    snap = _read(landscape.path_of(ch, e, "snap:5"))[0]
    assert z[centre, centre] > snap[centre, centre] > final[centre, centre]  # the summit half way down
    assert json.loads((ch.data_dir / e["files"]["balance"]).read_text())["router"] == "fake"
    assert _digest(dem) == before and ch.dem_path == dem  # the DEM is never touched
    assert sorted(p.name for p in (ch.data_dir / "landscape").iterdir()) == [e["id"]]  # no work folder left
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any("anno 5 di 10" in m for _f, m in seen)
    # on the DEM's own cell the change comes back as the simulation made it
    e2 = landscape.run(ch, {**SCENARIO, "cell": None, "snapshots": []})
    assert e2["cell"] == CELL and e2["grid"] == [N, N] and e2["id"] != e["id"]
    assert landscape.layers(e2) == ["final", "dz", "area"]


def test_the_scenario_is_checked_and_estimated(tmp_path):
    doc, ch = _project(tmp_path, later=False)
    p = landscape.job_params({"preset": "mudbrick_tell"})
    assert p["years"] == 1000 and p["diffusion"] == {"D": 0.01} and p["fluvial"]["K_sed"] == 5e-4
    assert p["fluvial"]["runoff"] == 0.2 and p["boundary"] == "open" and p["source"] == "active"
    p = landscape.job_params({"preset": "earth_slope", "fluvial": {"runoff": 0.5}, "snapshots": [10, 2000, 0]})
    assert p["fluvial"]["runoff"] == 0.5 and p["fluvial"]["K_sed"] == 1e-3 and p["snapshots"] == [10.0]
    with pytest.raises(ValueError, match="no process"):
        landscape.job_params({"preset": "custom", "diffusion": None, "fluvial": None})
    with pytest.raises(ValueError, match="boundary"):
        landscape.job_params({"boundary": "sometimes"})
    est = landscape.estimate(ch, SCENARIO)
    assert est["shape"] == [30, 30] and est["cell"] == 1.0 and not est["refuse"] and est["warning"] == ""
    auto = landscape.estimate(ch, {**SCENARIO, "cell": None})
    assert auto["cell"] == CELL and auto["shape"] == [N, N] and auto["seconds"] > est["seconds"]
    long = landscape.estimate(ch, {"preset": "mudbrick_tell", "years": 1e7, "cell": None}, lang="en")
    assert long["refuse"] and "hours" in long["warning"]
    with pytest.raises(ValueError, match="più di due ore"):
        landscape.run(ch, {"preset": "mudbrick_tell", "years": 1e7, "cell": None})
    assert not (ch.data_dir / "landscape").exists()  # refused before anything was written


# ------------------------------------------------------------------ versions, comparison, removal
def test_the_final_surface_as_a_version_of_the_dem(tmp_path, fake):
    from pyarchinit_photogrammetry.core import versions

    doc, ch = _project(tmp_path)
    dem, before = ch.dem_path, _digest(ch.dem_path)
    active = versions.active_info(ch, "dem")["name"]
    e = landscape.run(ch, SCENARIO, name="Tell, prova")
    v = landscape.add_as_dem_version(ch, e["id"])
    assert v["name"] == "Paesaggio dopo 10 anni (Tell, prova)" and v["params"]["run"] == e["id"]
    assert versions.versions(ch, "dem") == [v]
    assert _digest(versions.version_file(ch, "dem", v["id"])) == _digest(landscape.path_of(ch, e, "final"))
    assert ch.dem_path == dem and _digest(dem) == before and versions.active_info(ch, "dem")["name"] == active
    assert landscape.entry(ch, e["id"])["dem_version"] == v["id"]
    assert landscape.add_as_dem_version(ch, e["id"]) == v and len(versions.versions(ch, "dem")) == 1  # once
    assert [k for k, _n in landscape.dem_sources(ch)] == ["active", v["id"]]
    # a scenario that goes on from the version: it starts from the surface of the first one
    e2 = landscape.run(ch, {**SCENARIO, "source": v["id"], "snapshots": []})
    assert e2["source"] == {"version": v["id"], "name": v["name"], "file": v["files"]["dem"]}
    first = _read(landscape.path_of(ch, e, "final"))[0]
    dz2 = _read(landscape.path_of(ch, e2, "dz"))[0]
    np.testing.assert_allclose(_read(landscape.path_of(ch, e2, "final"))[0] - first, dz2, atol=1e-5, equal_nan=True)
    # made active by the user, the version is the DEM every tool uses; the run keeps its files
    versions.activate(ch, "dem", v["id"])
    assert _digest(ch.dem_path) == _digest(landscape.path_of(ch, e, "final"))
    assert landscape.path_of(ch, e, "final").exists()


def test_compare_with_the_dem_of_another_chunk(tmp_path, fake):
    doc, ch = _project(tmp_path)
    later = doc.chunks[1]
    e = landscape.run(ch, SCENARIO)
    c = landscape.compare(ch, e["id"], later)
    assert c["chunk"] == "Fase 2" and c["chunk_key"] == str(later.key) and c["source"] == "active"
    final = _read(landscape.path_of(ch, e, "final"))[0]
    other = _read(later.dem_path)[0]
    diff, tr, crs, _dt, _nd = _read(ch.data_dir / c["file"])
    assert tr == _read(ch.dem_path)[1] and crs == _read(ch.dem_path)[2]
    ok = np.isfinite(diff)
    np.testing.assert_allclose(diff[ok], (final - other)[ok], atol=1e-5)
    s = c["stats"]
    d = (final - other)[ok]
    assert s["cells"] == ok.sum() and s["median"] == pytest.approx(np.median(d), abs=1e-5)
    assert s["nmad"] == pytest.approx(1.4826 * np.median(np.abs(d - np.median(d))), abs=1e-5)
    assert s["p95"] == pytest.approx(np.percentile(np.abs(d), 95), abs=1e-5)
    z = _read(ch.dem_path)[0]
    base = (z - other)[ok]
    assert c["baseline"]["median"] == pytest.approx(np.median(base), abs=1e-5)
    # the simulation wore the summit down towards the later surface: the difference shrinks there
    centre = N // 2
    assert abs(diff[centre, centre]) < abs(z[centre, centre] - other[centre, centre])
    assert landscape.entry(ch, e["id"])["comparisons"] == [c] and f"compare:{later.key}" in landscape.layers(
        landscape.entry(ch, e["id"]))
    # again: replaced, not added; a later DEM on another grid is resampled onto the run's
    worn, _r = tell(n=40, cell=0.75, height=3.6, radius=11.0)
    _write_dem(later.data_dir / "dem.tif", worn, cell=0.75)
    c2 = landscape.compare(ch, e["id"], later)
    assert len(landscape.entry(ch, e["id"])["comparisons"]) == 1 and c2["stats"]["cells"] > 0.8 * ok.sum()
    assert _read(ch.data_dir / c2["file"])[0].shape == (N, N)


def test_remove_deletes_only_the_run(tmp_path, fake):
    from pyarchinit_photogrammetry.core import versions

    doc, ch = _project(tmp_path)
    a = landscape.run(ch, SCENARIO)
    b = landscape.run(ch, SCENARIO)
    v = landscape.add_as_dem_version(ch, a["id"])
    before = _digest(ch.dem_path)
    assert landscape.remove(ch, a["id"])
    assert not (ch.data_dir / "landscape" / a["id"]).exists() and (ch.data_dir / "landscape" / b["id"]).is_dir()
    assert [e["id"] for e in landscape.runs(ch)] == [b["id"]]
    assert versions.version_file(ch, "dem", v["id"]).exists() and _digest(ch.dem_path) == before
    assert not landscape.remove(ch, "nothing")
    (ch.data_dir / "keep").mkdir()
    ch.meta["landscape"]["runs"].append({"id": "../keep"})
    landscape.remove(ch, "../keep")
    assert (ch.data_dir / "keep").is_dir()  # never outside the plugin's folder


def test_export(tmp_path, fake):
    doc, ch = _project(tmp_path)
    e = landscape.run(ch, SCENARIO)
    one = landscape.export(ch, e["id"], tmp_path / "out" / "dz.tif", "dz")
    assert _digest(one) == _digest(landscape.path_of(ch, e, "dz"))
    import rasterio

    every = landscape.export(ch, e["id"], tmp_path / "out" / "all.tif")
    with rasterio.open(every) as src:
        assert src.count == 4 and src.descriptions == ("final", "dz", "area", "anni_5")
        np.testing.assert_array_equal(src.read(2), _read(landscape.path_of(ch, e, "dz"))[0].astype(np.float32))
    with pytest.raises(KeyError):
        landscape.export(ch, e["id"], tmp_path / "x.tif", "snap:7")


# ------------------------------------------------------------------ the command line
def _plugin_folder(root: Path) -> Path:
    """A plugin folder of the program with the plugin "installed" (its metadata; the package is on the path)."""
    folder = root / "plugins"
    dist = folder / "landscape" / "ppg_landscape-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-landscape\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\nlandscape = ppg_landscape:LandscapePlugin\n")
    return folder


def test_command_line(tmp_path, fake, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch = _project(tmp_path)
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    monkeypatch.setenv("PPG_PLUGINS_ENVIRONMENT", "0")
    cli = CliRunner()
    project = str(doc.path)
    r = cli.invoke(app, ["plugin", "landscape", "estimate", project, "--preset", "custom", "--years", "10",
                         "--no-water", "--cell", "1", "--json"])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output[r.output.index("{"):])["shape"] == [30, 30]
    r = cli.invoke(app, ["plugin", "landscape", "run", project, "--preset", "custom", "--years", "6", "--no-water",
                         "--D", "0.02", "--snapshots", "2,4", "--cell", "1", "--boundary", "closed", "--name", "Prova"])
    assert r.exit_code == 0, r.output
    assert "«Prova» pronto" in r.output and "[100.0%]" in r.output
    runs = landscape.runs(ppg.Document.open(doc.path).chunks[0])  # saved by the command
    assert len(runs) == 1 and runs[0]["params"]["diffusion"] == {"D": 0.02} and runs[0]["snapshots"] == [2.0, 4.0]
    rid = runs[0]["id"]
    r = cli.invoke(app, ["plugin", "landscape", "list", project])
    assert r.exit_code == 0 and json.loads(r.output[r.output.index("["):])[0]["id"] == rid
    r = cli.invoke(app, ["plugin", "landscape", "show", project, rid])
    assert r.exit_code == 0 and json.loads(r.output[r.output.index("{"):])["name"] == "Prova"
    r = cli.invoke(app, ["plugin", "landscape", "version", project, rid])
    assert r.exit_code == 0 and "Paesaggio dopo 6 anni (Prova)" in r.output
    r = cli.invoke(app, ["plugin", "landscape", "compare", project, rid, "--with", "Fase 2"])
    assert r.exit_code == 0 and '"nmad"' in r.output
    out = tmp_path / "dz.tif"
    r = cli.invoke(app, ["plugin", "landscape", "export", project, rid, str(out), "--layer", "dz"])
    assert r.exit_code == 0 and out.exists()
    r = cli.invoke(app, ["plugin", "landscape", "remove", project, rid])
    assert r.exit_code == 0 and landscape.runs(ppg.Document.open(doc.path).chunks[0]) == []
    r = cli.invoke(app, ["plugin", "landscape", "show", project, rid])
    assert r.exit_code == 1 and "no run" in r.output


# ------------------------------------------------------------------ the pane in the window
def test_the_pane_in_the_window(tmp_path, fake, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_landscape import LandscapePlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.core import versions
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch = _project(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    plugin = LandscapePlugin()
    try:
        plugin.activate(Context("landscape", tmp_path / "plugin", window=w))
        pane = plugin.pane
        dock = w.panes.groups["right"].dock(pane)
        assert dock is not None and dock.objectName().startswith("pane_plugin_landscape")
        assert pane.b_install.isVisibleTo(pane) == (plugin_status_can_install())
        assert [pane.c_source.itemData(i) for i in range(pane.c_source.count())] == ["active"]
        assert pane.has_dem and pane.b_run.isEnabled()
        # a preset fills the form, the estimate follows
        pane.c_preset.setCurrentIndex(pane.c_preset.findData("exposed_excavation"))
        assert pane.s_years.value() == 10 and pane.spins["D"].value() == pytest.approx(0.02)
        assert pane.spins["soil_depth"].value() == pytest.approx(0.2) and pane.preset() == "exposed_excavation"
        first = pane.l_estimate.text()
        assert "Griglia 60 × 60 celle" in first
        pane.c_preset.setCurrentIndex(pane.c_preset.findData("mudbrick_tell"))
        assert pane.s_years.value() == 1000 and pane.spins["K_sed"].value() == pytest.approx(5e-4)
        assert pane.l_estimate.text() != first
        pane.spins["D"].setValue(0.03)  # a value changed by hand: the scenario is custom now
        assert pane.preset() == "custom" and pane.spins["K_sed"].value() == pytest.approx(5e-4)
        pane.s_cell.setValue(0.1)
        pane.s_years.setValue(1e6)  # far too long: Run refused
        assert not pane.b_run.isEnabled() and "Troppo lunga" in pane.l_estimate.text()
        pane.s_years.setValue(10)
        pane.k_fluvial.setChecked(False)
        pane.s_cell.setValue(1.0)
        pane.e_snaps.setText("5")
        assert pane.b_run.isEnabled() and "Griglia 30 × 30 celle di 1 m" in pane.l_estimate.text()
        params = pane.params()
        assert params["fluvial"] is None and params["diffusion"] == {"D": 0.03} and params["snapshots"] == [5.0]
        # Run: in the background, then the run in the table, its dz over the DEM, in the tree
        pane.b_run.click()
        w.runner.wait()
        rs = landscape.runs(ch)
        assert len(rs) == 1 and pane.table.rowCount() == 1 and pane.current_id() == rs[0]["id"]
        rid = rs[0]["id"]
        assert plugin.overlay is not None and plugin.overlay.item is not None and plugin.overlay.key == (rid, "dz")
        assert "Erosione e deposito (dz)" in pane.l_limits.text() and "Eroso" in pane.l_summary.text()
        layers = [pane.c_layer.itemData(i) for i in range(pane.c_layer.count())]
        assert layers == ["final", "dz", "area", "snap:5"]
        assert w.isWindowModified()
        items = plugin._tree_items(ch)
        assert items[0]["label"] == "Paesaggio nel tempo" and items[0]["children"][0]["label"] == rs[0]["name"]
        pane.c_layer.setCurrentIndex(2)
        pane.b_show.click()
        assert plugin.overlay.key == (rid, "area") and "m²" in pane.l_limits.text()
        rasters, _v = plugin._qgis_entries()
        assert [r["ramp"] for r in rasters if "ramp" in r] == ["diverging", "grey"]
        assert any(r.get("hillshade") for r in rasters)
        # the final surface among the versions of the DEM, the comparison with the later phase
        pane.b_version.click()
        assert len(versions.versions(ch, "dem")) == 1 and ch.dem_path.name == "dem.tif"
        assert [pane.c_source.itemData(i) for i in range(pane.c_source.count())][1] == versions.versions(ch, "dem")[0]["id"]
        plugin.compare(rid, doc.chunks[1])
        w.runner.wait()
        assert "Con «Fase 2»" in pane.l_summary.text() and plugin.overlay.key == (rid, f"compare:{doc.chunks[1].key}")
        # Remove asks first; the box closed: nothing
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: None)
        assert not plugin.remove(rid) and landscape.entry(ch, rid) is not None
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
        assert plugin.remove(rid) and landscape.runs(ch) == [] and pane.table.rowCount() == 0
        assert plugin.overlay.item is None and plugin._tree_items(ch) == []
        plugin.deactivate()
        plugin.context.release()
        assert w.panes.dock(pane) is None
    finally:
        w.setWindowModified(False)
        w.close()


def plugin_status_can_install() -> bool:
    from ppg_landscape import environment

    return environment.unsupported() is None and environment.where()[0] is None


# ------------------------------------------------------------------ the real worker
@pytest.mark.skipif(LANDLAB is None, reason="no Python with Landlab (PPG_TEST_LANDLAB)")
def test_a_run_through_landlab(tmp_path):
    doc, ch = _project(tmp_path, later=False)
    e = landscape.run(ch, {"preset": "earth_slope", "years": 20, "cell": 1.0, "snapshots": [10]},
                      python=LANDLAB, script=Path(landscape.__file__).with_name("worker.py"))
    assert e["balance"]["eroder"] and e["balance"]["router"] and e["versions"]["landlab"]
    dz = _read(landscape.path_of(ch, e, "dz"))[0]
    z = _read(ch.dem_path)[0]
    ok = np.isfinite(z)
    np.testing.assert_allclose(_read(landscape.path_of(ch, e, "final"))[0][ok] - z[ok], dz[ok], atol=1e-5)
    assert dz[N // 2, N // 2] < 0
