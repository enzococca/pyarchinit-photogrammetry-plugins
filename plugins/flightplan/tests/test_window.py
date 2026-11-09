"""The plugin in an offscreen window: the pane on the right with its steps, the group of the tool rail, an area
drawn with the program's polygon tool and the plan designed in the background, the plan's layers on the 2D view
(again after another raster), the coverage shown and hidden, the exports, the undo of a removed plan with its
files, the view cleaned when the plugin is switched off, the English texts; and the carry-overs of task 6 (the
project marked modified when an export fails after refreshing the terrain, no coverage record for a plan gone, the
removed plans' files deleted when a project is opened)."""
from __future__ import annotations

import time

import numpy as np
import pytest
import synth

from ppg_flightplan import design, project
from ppg_flightplan.texts import LAYER_AREA, LAYER_LOW, LAYER_SHOTS, LAYER_STRIPS, LAYER_TAKEOFF, LAYERS

pytest.importorskip("PySide6")

AREA_W, AREA_H = 300.0, 200.0


def _wait(window, app_, seconds: float = 120.0) -> None:
    t0 = time.time()
    while window.runner.busy and time.time() - t0 < seconds:
        app_.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app_.processEvents()


@pytest.fixture
def gui(tmp_path, monkeypatch):
    """(window, plugin, chunk, app, said): a project with a DEM in EPSG:32632, the window offscreen, the plugin
    activated on it; the program's messages and choices collected instead of shown."""
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    from ppg_flightplan import FlightPlanPlugin

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    app_ = QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    said = []
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: said.append(a[3]))
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    doc, ch = synth.chunk(tmp_path, dem=lambda x, y: 100.0 + (x - synth.X0) * 0.05)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))

    def request(kind, on_finished, on_cancelled=None, hint=""):  # the user draws at once
        request.kind = kind
        on_finished(np.asarray(request.xy, float))
        return True

    request.xy = synth.rectangle_xy(AREA_W, AREA_H)
    monkeypatch.setattr(w, "request_geometry", request)
    plugin = FlightPlanPlugin()
    plugin.activate(Context("flightplan", tmp_path / "plugin", window=w))
    w._show_raster("dem")
    app_.processEvents()
    try:
        yield w, plugin, ch, app_, said, request
        assert not errors, errors
    finally:
        w.setWindowModified(False)
        w.close()


def _design(w, plugin, app_):
    plugin.draw_area()
    plugin.pane.b_design.click()
    _wait(w, app_)


def _expected(piano):
    from ppg_flightplan import plan as pl
    from ppg_flightplan.route import shot_points

    legs = pl.plan_legs(piano)
    return len(legs), sum(len(shot_points(leg.route)) for leg in legs)


def test_pane_and_rail_in_the_window(gui):
    w, plugin, ch, app_, said, _request = gui
    p = plugin.pane
    dock = w.panes.groups["right"].dock(p)
    assert dock is not None and dock.objectName() == "pane_plugin_flightplan"
    dock.close()
    plugin.show_pane()
    assert not dock.isHidden()
    groups = [g for g in plugin.context.contributions.rail_groups if g.key == "flightplan"]
    assert len(groups) == 1 and groups[0].title == "Piani di volo" and len(groups[0].actions) == 7
    assert [a.title for a in groups[0].actions][:3] == ["Pannello Piani di volo", "Disegna l'area del volo",
                                                        "Progetta il volo..."]
    assert plugin._qgis_entries in w.qgis_providers
    # no plan yet: the steps after the parameters say nothing, those needing a plan wait
    assert "Nessuna area" in p.states["area"].text() and "Mini 4 Pro" in p.states["camera"].text()
    for key in ("plan", "terrain", "coverage", "targets", "export"):
        assert p.states[key].text() == "", key
    assert p.b_draw.isEnabled() and p.b_design.isEnabled() and not p.b_export.isEnabled()
    assert [p.titles[k].text() for k in ("area", "after")] == ["<b>1. Area</b>", "<b>9. Dopo il volo</b>"]
    # grouped cameras: the group titles cannot be chosen
    model = p.c_camera.model()
    titles = [model.item(i).text() for i in range(model.rowCount()) if not model.item(i).isEnabled()]
    assert titles[0].startswith("DJI (Pilot") and "Altri droni" in titles


def test_area_drawn_designs_a_plan(gui):
    w, plugin, ch, app_, said, request = gui
    plugin.draw_area()
    assert request.kind == "polygon"
    assert "Area: 4 vertici" in plugin.pane.states["area"].text()
    assert len(w.ortho_view.features(LAYER_AREA)) == 1          # the area waits for the design, drawn already
    w.setWindowModified(False)
    plugin.pane.s_margin.setValue(20)
    plugin.pane.b_design.click()
    assert w.runner.busy and not plugin.pane.b_design.isEnabled()  # one step at a time
    _wait(w, app_)
    e = project.active_plan(ch)
    assert e is not None and w.isWindowModified()
    piano = e["piano"]
    # the polygon drawn is kept as drawn; the margin only in the blocks
    drawn = np.array(project.to_chunk(ch, [tuple(q) for q in piano["areaDisegnata"]]))
    assert np.abs(drawn - np.array(synth.rectangle_xy(AREA_W, AREA_H))).max() < 0.01
    block = project.to_chunk(ch, [tuple(q) for q in piano["blocchi"][0]["area"]])
    assert block[:, 0].max() - block[:, 0].min() > AREA_W + 30 and piano["crs"] == "EPSG:32632"
    legs, shots = _expected(piano)
    view = w.ortho_view
    assert len(view.features(LAYER_AREA)) == 2                 # the drawn area and the block with its margin
    assert len(view.features(LAYER_STRIPS)) == legs
    assert len(view.features(LAYER_SHOTS)) == shots and shots > 20
    assert {f["tratta"] for f in view.features(LAYER_STRIPS)} == {f["tratta"] for f in view.features(LAYER_SHOTS)}
    p = plugin.pane
    assert "scatti" in p.states["plan"].text() and p.t_plan.rowCount() >= 1 + legs
    assert p.c_plan.count() == 1 and p.b_export.isEnabled()
    assert "GSD" in p.states["params"].text()
    # a block switched off: the plan changes, its strips leave the view
    w.setWindowModified(False)
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Unchecked)
    app_.processEvents()
    assert project.active_plan(ch)["piano"]["blocchi"][0]["attivo"] is False and w.isWindowModified()
    assert view.features(LAYER_STRIPS) == []
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Checked)
    app_.processEvents()
    assert view.features(LAYER_STRIPS)
    # the takeoff on the view
    request.xy = [synth.rectangle_xy(AREA_W, AREA_H)[0]]
    p.b_takeoff.click()
    assert request.kind == "point" and len(view.features(LAYER_TAKEOFF)) == 1
    assert project.active_plan(ch)["piano"]["decollo"]
    # the takeoff of a leg: the leg chosen in the table
    leg_row = next(r for r in range(p.t_plan.rowCount()) if p.t_plan.item(r, 0).data(256)[0] == "leg")
    p.t_plan.selectRow(leg_row)
    p.b_leg_takeoff.click()
    leg_key = p.t_plan.item(leg_row, 0).data(256)[1]
    assert project.active_plan(ch)["piano"]["decolli"][leg_key]
    # design again on the same area: the same plan, changed
    p.s_front.setValue(70)
    p.b_design.click()
    _wait(w, app_)
    assert len(project.plans(ch)) == 1 and project.active_plan(ch)["piano"]["blocchi"][0]["avanti"] == 70
    assert project.active_plan(ch)["piano"]["decollo"]           # kept
    assert leg_key in project.active_plan(ch)["piano"]["decolli"]  # its leg is still there: kept too
    # a new area: a new plan
    request.xy = synth.rectangle_xy(150, 100, cx=synth.X0 + 400)
    _design(w, plugin, app_)
    assert len(project.plans(ch)) == 2 and p.c_plan.count() == 2


def test_layers_come_back_after_another_raster(gui):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    legs, shots = _expected(project.active_plan(ch)["piano"])
    assert plugin.context.show_raster("dem")
    app_.processEvents()
    view = w.ortho_view
    assert len(view.features(LAYER_STRIPS)) == legs and len(view.features(LAYER_SHOTS)) == shots
    assert len(view.features(LAYER_AREA)) == 1


def test_coverage_shown_and_hidden(gui):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    p = plugin.pane
    p.s_low.setValue(50)                                          # a high threshold: some cells are low
    p.b_coverage.click()
    _wait(w, app_)
    rec = project.coverage_record(ch, project.active_plan(ch)["id"])
    assert rec is not None and rec["low"] == 50
    rgba = ch.data_dir / rec["files"]["rgba"]
    assert w.ortho_view.path is not None and str(w.ortho_view.path) == str(rgba)
    assert w.ortho_view.features(LAYER_LOW) and w.ortho_view.features(LAYER_STRIPS)
    assert "Foto per cella" in p.states["coverage"].text()
    p.b_show_ortho.click()
    app_.processEvents()
    assert str(w.ortho_view.path) == str(ch.dem_path)              # no orthophoto: the DEM
    assert w.ortho_view.features(LAYER_LOW)                          # the low coverage stays on the plan
    p.b_show_coverage.click()
    app_.processEvents()
    assert str(w.ortho_view.path) == str(rgba)


def test_export_from_file_menu(gui, tmp_path, monkeypatch):
    import sqlite3
    import zipfile

    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    exporters = {e.suffix: e for e in plugin.context.contributions.exporters}
    assert set(exporters) == {".kmz", ".plan", ".csv", ".gpkg", ".xlsx"}       # registered once
    files = exporters[".kmz"].function(ch, tmp_path / "out" / "volo.kmz", None)
    assert files and all(zipfile.is_zipfile(f) for f in files)
    files = exporters[".gpkg"].function(ch, tmp_path / "out" / "volo.gpkg", None)
    with sqlite3.connect(files[0]) as con:
        assert con.execute("select count(*) from ppg_scatti").fetchone()[0] > 0
    # from the pane: a format chosen, a file asked, the export in the background, what was written said
    monkeypatch.setattr(plugin, "_save_path", lambda *a: str(tmp_path / "out" / "volo"))
    plugin.pane.c_format.setCurrentIndex(plugin.pane.c_format.findData("litchi"))
    plugin.pane.b_export.click()
    _wait(w, app_)
    assert said and "scritt" in said[-1] and list((tmp_path / "out").glob("*.csv"))


def test_undo_brings_the_plan_back(gui):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    plugin.pane.b_coverage.click()
    _wait(w, app_)
    id_ = project.active_plan(ch)["id"]
    folder = project.plan_folder(ch, id_)
    assert folder.is_dir()
    plugin.pane.b_remove_plan.click()
    assert project.entry(ch, id_) is None and not folder.exists()
    assert (project.folder(ch) / project.REMOVED / id_).is_dir()
    assert w.ortho_view.features(LAYER_STRIPS) == [] and plugin.pane.c_plan.count() == 0
    w.history.undo()
    app_.processEvents()
    assert project.entry(ch, id_) is not None and folder.is_dir()
    assert project.coverage_record(ch, id_) is not None             # its files are back
    assert w.ortho_view.features(LAYER_STRIPS) and plugin.pane.c_plan.count() == 1
    w.history.undo()                                               # the design itself: the plan and its folder go
    app_.processEvents()
    assert project.entry(ch, id_) is None and not folder.exists()
    assert w.ortho_view.features(LAYER_STRIPS) == []
    w.history.redo()
    app_.processEvents()
    assert project.entry(ch, id_) is not None and folder.is_dir()


def test_terrain_from_the_pane(gui):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    p = plugin.pane
    p.c_source.setCurrentIndex(p.c_source.findData("dem"))
    w.setWindowModified(False)
    p.b_terrain.click()
    _wait(w, app_)
    piano = project.active_plan(ch)["piano"]
    assert piano["blocchi"][0].get("quote") and piano["fonteQuote"]["sorgente"] == "dem" and w.isWindowModified()
    assert "segue il terreno" in p.states["terrain"].text()
    assert "quote punto per punto" in p.states["export"].text().lower()
    heights = {f["quota_decollo_m"] for f in w.ortho_view.features(LAYER_SHOTS)}
    assert len(heights) > 3                                       # the slope of the DEM in the shots' heights
    p.b_terrain_off.click()
    assert not project.active_plan(ch)["piano"]["blocchi"][0].get("quote")
    # the public tiles without the network: said in words
    p.c_source.setCurrentIndex(p.c_source.findData("tiles"))
    p.b_terrain.click()
    _wait(w, app_)
    assert said and "rete" in said[-1] and not project.active_plan(ch)["piano"]["blocchi"][0].get("quote")


def test_a_dem_without_georeference_is_said(tmp_path):
    from PySide6.QtWidgets import QApplication

    from ppg_flightplan import FlightPlanPlugin
    from ppg_flightplan.pane import FlightPlanPane

    QApplication.instance() or QApplication([])
    _, ch = synth.chunk(tmp_path)
    path = synth.write_dem(ch.data_dir / "dem.tif", 100.0, crs=None)
    ch._set_product("dem", path)
    plugin = FlightPlanPlugin()
    plugin.pane = FlightPlanPane()
    text = plugin._terrain_state(ch, {"blocchi": []})
    assert "nessuna georeferenza" in text


def test_plugin_disabled_cleans_the_view(gui):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    assert w.ortho_view.features(LAYER_SHOTS)
    pane = plugin.pane
    plugin.deactivate()
    plugin.context.release()
    assert all(name not in w.ortho_view.layers for name in LAYERS)
    assert plugin._qgis_entries not in w.qgis_providers and w.panes.dock(pane) is None
    w.refresh_all()                                               # nothing of the plugin draws again
    app_.processEvents()
    assert all(name not in w.ortho_view.layers for name in LAYERS)


def test_tree_and_targets(gui, tmp_path, monkeypatch):
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    items = plugin._tree_items(ch)
    assert items[0]["label"] == "Piani di volo" and items[0]["children"][0]["label"].endswith("(attivo)")
    # targets from a CSV in degrees with the columns in an uncertain order: said, and in red in the pane
    from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

    lat, lon = synth.area_latlon(40, 40)[0]
    csv = tmp_path / "target.csv"
    csv.write_text(f"T1,{lat:.7f},{lon:.7f},101.2\nT2,{lat + 0.0003:.7f},{lon + 0.0003:.7f},101.4\n")
    monkeypatch.setattr(plugin, "_open_path", lambda *a: str(csv))
    monkeypatch.setattr(ParamDialog, "ask", classmethod(lambda cls, *a, **k: {"epsg": ""}))
    plugin.pane.b_csv.click()
    piano = project.active_plan(ch)["piano"]
    assert len(piano["target"]["punti"]) == 2
    assert len(w.ortho_view.features("Piano di volo: target")) == 2
    assert "2 target" in plugin.pane.states["targets"].text()
    # no header, degrees, both columns valid latitudes (about 41.5 and 9.0): the plan's own place decides the order
    assert "ordine_incerto" not in (piano["target"].get("avvisi") or [])
    assert not plugin.pane.l_target_warning.isVisibleTo(plugin.pane)
    # without an active block the plan has no place to compare with: the order is a guess, said
    p = plugin.pane
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Unchecked)
    app_.processEvents()
    plugin.pane.b_csv.click()
    codes = project.active_plan(ch)["piano"]["target"].get("avvisi") or []
    assert "ordine_incerto" in codes
    assert plugin.pane.l_target_warning.isVisibleTo(plugin.pane)
    assert plugin.pane.l_target_warning.text().startswith("Ordine delle colonne incerto")
    assert any("Ordine delle colonne incerto" in m for m in said)
    plugin.pane.b_targets_off.click()
    assert "target" not in project.active_plan(ch)["piano"]


def test_english_texts(monkeypatch):
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.gui import i18n

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(i18n, "current", lambda: "en")
    from ppg_flightplan.pane import FlightPlanPane
    from ppg_flightplan.texts import format_name, name

    pane = FlightPlanPane()
    assert name() == "Flight plans"
    assert pane.titles["area"].text() == "<b>1. Area</b>" and "Parameters" in pane.titles["params"].text()
    assert pane.b_design.text() == "Design" and pane.b_draw.text() == "Draw on the view"
    assert format_name("litchi") == "Litchi (CSV)" and "After the flight" in pane.titles["after"].text()


# ------------------------------------------------------------------ carry-overs of task 6
def test_export_failing_after_the_refresh_still_marks_modified(tmp_path):
    """M1: a terrain-following plan whose heights are refreshed by the export, written where it cannot be: the
    plan in the meta changed, so the project is marked modified although the export failed."""
    from ppg_flightplan import FlightPlanPlugin

    _, ch = synth.chunk(tmp_path, dem=lambda x, y: 100.0 + (x - synth.X0) * 0.1)
    piano = design.design(synth.area_latlon(300, 200), "orto", 2.0, follow_terrain=True,
                          dem=project.chunk_dem(ch))["piano"]
    piano["blocchi"][0]["lato"] = 60                              # its heights are now stale: the export redoes them
    id_ = project.add_plan(ch, piano)
    before = project.entry(ch, id_)["stamp"]

    class Ctx:
        marked = 0

        def call_in_gui(self, function, *a, **k):
            function(*a, **k)

        def mark_modified(self):
            Ctx.marked += 1

    plugin = FlightPlanPlugin()
    plugin.context = Ctx()
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    with pytest.raises(OSError):
        plugin._exporter("plan")(ch, blocker / "volo.plan", None)
    assert project.entry(ch, id_)["stamp"] != before and Ctx.marked == 1


def test_no_coverage_record_for_a_plan_gone(tmp_path, monkeypatch):
    """M5: the plan leaves the meta while its coverage is computed (an undo of its addition): no record."""
    from ppg_flightplan import coverage

    _, ch = synth.chunk(tmp_path, dem=100.0)
    id_ = project.add_plan(ch, design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"])
    real = coverage.predict

    def predict(*a, **k):
        out = real(*a, **k)
        project.state(ch)["plans"] = []                          # the undo, meanwhile
        return out

    monkeypatch.setattr(coverage, "predict", predict)
    project.coverage_for(ch, id_)
    assert id_ not in project.state(ch)["coverage"]
    assert not project.plan_folder(ch, id_).exists()                 # its folder aside with the others


def test_removed_plans_purged_when_a_project_is_opened(tmp_path):
    """M3: the folders of removed plans are deleted when no undo can reach them: a project just opened (its
    meta is the project as saved) and the command line after saving."""
    import pyarchinit_photogrammetry as ppg

    doc, ch = synth.chunk(tmp_path, dem=100.0)
    a = project.add_plan(ch, design.design(synth.area_latlon(300, 200), "orto", 2.0)["piano"])
    b = project.add_plan(ch, design.design(synth.area_latlon(200, 200), "orto", 2.0)["piano"])
    project.coverage_for(ch, a)
    project.coverage_for(ch, b)
    project.remove_plan(ch, a)
    doc.save()
    removed = project.folder(ch) / project.REMOVED
    assert (removed / a).is_dir()
    reopened = ppg.Document.open(doc.path).chunk
    project.state(reopened)["plans"].append({"id": "pdeadbeef", "name": "x", "piano": {}, "stamp": ""})
    (removed / "pdeadbeef").mkdir()                                  # in the meta: kept for the files to follow
    assert project.purge_removed(reopened) == [a]
    assert not (removed / a).exists() and (removed / "pdeadbeef").is_dir()
    assert project.plan_folder(reopened, b).is_dir()


def test_the_window_purges_a_project_just_opened(gui, tmp_path):
    w, plugin, ch, app_, said, _request = gui
    stray = project.folder(ch) / project.REMOVED / "p0badc0d"
    stray.mkdir(parents=True)
    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document.open(w.doc.path)
    w.set_document(doc)
    app_.processEvents()
    assert not stray.exists()


def _user_mission() -> bytes:
    import io
    import zipfile

    from ppg_flightplan import export_dji
    from ppg_flightplan.camera import camera
    from ppg_flightplan.route import Route

    many = Route("utente", [(45.0 + k * 1e-4, 10.0) for k in range(5)], 50.0)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("wpmz/template.kml", export_dji.template_kml(many, camera("mini4pro"), 0, author="utente"))
        z.writestr("wpmz/waylines.wpml", export_dji.waylines_wpml(many, camera("mini4pro")))
    return buffer.getvalue()


def test_copy_into_djifly(gui, tmp_path, monkeypatch):
    from ppg_flightplan import djifly
    from ppg_flightplan import plan as pl

    w, plugin, ch, app_, said, _request = gui
    monkeypatch.setattr(djifly, "ONLY_VOLUMES", False)
    _design(w, plugin, app_)
    n = len(pl.plan_legs(project.active_plan(ch)["piano"]))
    waypoint = tmp_path / "telecomando" / "waypoint"
    user, *empty = [f"{k:08X}-0000-4000-8000-000000000000" for k in range(n + 1)]
    for s in [user, *empty]:
        (waypoint / s).mkdir(parents=True)
    mission = _user_mission()
    (waypoint / user / f"{user}.kmz").write_bytes(mission)
    asked = []
    monkeypatch.setattr(plugin, "_folder_path", lambda *a: str(waypoint))
    monkeypatch.setattr(plugin, "_choice", lambda text, options, kind="question": asked.append(text) or 0)
    plugin.pane.b_djifly.click()
    _wait(w, app_)
    # the slots said before copying; the mission of the user left alone, the legs in the empty slots
    assert asked and all(s in asked[0] for s in empty) and user not in asked[0]
    assert (waypoint / user / f"{user}.kmz").read_bytes() == mission
    assert all((waypoint / s / f"{s}.kmz").is_file() for s in empty)
    given = project.active_plan(ch)["piano"]["djiFly"]
    assert sorted(v["cartella"] for v in given.values()) == sorted(empty)
    # only the mission of the user left: the question, then the copy over it with a copy of it kept
    # one slot fewer than the legs among the safe ones (whatever their number): the user's mission is needed,
    # one question says it and asks to overwrite, and the copy goes over it with a copy of it kept
    import shutil

    shutil.rmtree(waypoint / empty[-1])
    asked.clear()
    plugin.pane.b_djifly.click()
    _wait(w, app_)
    assert len(asked) == 1 and "MISSIONE DELL'UTENTE" in asked[0] and user in asked[0]
    assert (waypoint / user / f"{user}.kmz").read_bytes() != mission
    backups = list((project.folder(ch) / "djifly_backup").rglob("*.kmz"))
    assert any(b.read_bytes() == mission for b in backups)      # the legs written before have their copies too


def test_a_stale_coverage_leaves_the_view(gui):
    """I1: the coverage raster shown goes away when the plan changes, when another plan becomes active and when the
    plan is removed: the orthophoto (here the DEM) comes back and the status says why."""
    w, plugin, ch, app_, said, request = gui
    view = w.ortho_view
    _design(w, plugin, app_)
    p = plugin.pane

    def shown_coverage():
        rec = project.coverage_record(ch, project.active_plan(ch)["id"])
        assert rec is not None and str(view.path) == str(ch.data_dir / rec["files"]["rgba"])

    p.b_coverage.click()
    _wait(w, app_)
    shown_coverage()
    # 1 the plan changed (a block off)
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Unchecked)
    app_.processEvents()
    assert str(view.path) == str(ch.dem_path) and "piano cambiato" in p.l_status.text()
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Checked)
    app_.processEvents()
    # 2 another plan active
    p.b_coverage.click()
    _wait(w, app_)
    shown_coverage()
    first = project.active_plan(ch)["id"]
    request.xy = synth.rectangle_xy(150, 100, cx=synth.X0 + 400)
    _design(w, plugin, app_)
    assert project.active_plan(ch)["id"] != first and str(view.path) == str(ch.dem_path)
    p.c_plan.setCurrentIndex(p.c_plan.findData(first))
    p.b_show_coverage.click()
    app_.processEvents()
    shown_coverage()
    second = next(q["id"] for q in project.plans(ch) if q["id"] != first)
    p.c_plan.setCurrentIndex(p.c_plan.findData(second))
    app_.processEvents()
    assert project.active_plan(ch)["id"] == second and str(view.path) == str(ch.dem_path)
    # 3 the plan removed
    p.c_plan.setCurrentIndex(p.c_plan.findData(first))
    p.b_show_coverage.click()
    app_.processEvents()
    shown_coverage()
    p.b_remove_plan.click()
    app_.processEvents()
    assert project.entry(ch, first) is None and str(view.path) == str(ch.dem_path)


def test_export_from_the_pane_failing_after_the_refresh_marks_modified(gui, tmp_path, monkeypatch):
    """m6: M1 through the pane: the terrain heights redone by the export, the file not writable: the project is
    marked modified all the same, and the error said."""
    w, plugin, ch, app_, said, _request = gui
    _design(w, plugin, app_)
    plugin.pane.c_source.setCurrentIndex(plugin.pane.c_source.findData("dem"))
    plugin.pane.b_terrain.click()
    _wait(w, app_)
    e = project.active_plan(ch)
    import copy

    piano = copy.deepcopy(e["piano"])
    piano["blocchi"][0]["lato"] = 60                              # its heights now stale: the export redoes them
    project.update_plan(ch, e["id"], piano)
    before = project.entry(ch, e["id"])["stamp"]
    blocker = tmp_path / "not_a_folder"
    blocker.write_text("x")
    monkeypatch.setattr(plugin, "_save_path", lambda *a: str(blocker / "volo.plan"))
    w.setWindowModified(False)
    plugin.export("plan")
    _wait(w, app_)
    assert project.entry(ch, e["id"])["stamp"] != before and w.isWindowModified()
    assert said and not (blocker / "volo.plan").exists()


def test_manual_camera_starts_from_the_list(gui, monkeypatch):
    """m3: the typed camera's fields start from the camera chosen in the list, never from invented values."""
    from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

    w, plugin, ch, app_, said, _request = gui
    seen = {}

    def ask(cls, parent, title, fields, intro=""):
        seen.update({f.name: f.default for f in fields})
        seen["intro"] = intro
        return {k: v for k, v in seen.items() if k != "intro"} | {"name": "Mia"}

    monkeypatch.setattr(ParamDialog, "ask", classmethod(ask))
    plugin.pane.set_camera("mavic3e")
    plugin.camera_by_hand()
    assert (seen["width"], seen["height"], seen["focal"], seen["pixels"]) == (17.3, 13.0, 12.29, 5280)
    assert "Mavic 3E" in seen["intro"] and plugin.pane.camera_id().startswith("manual-")


def test_the_project_dem_says_it_asks_the_tiles_where_it_ends():
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication

    from ppg_flightplan.pane import FlightPlanPane

    QApplication.instance() or QApplication([])
    p = FlightPlanPane()
    tip = p.c_source.itemData(p.c_source.findData("dem"), Qt.ItemDataRole.ToolTipRole)
    assert "tegole pubbliche" in tip and "non arriva" in tip
    assert "rete" in p.c_source.toolTip()


# ------------------------------------------------------------------ 0.1.1
def test_the_parameters_text_says_the_overlaps_used(gui):
    """Forward set to 75 %: step 3 says 75 % forward once, never the goal's 80 %."""
    w, plugin, ch, app_, said, _request = gui
    plugin.pane.s_front.setValue(75)
    _design(w, plugin, app_)
    text = plugin.pane.states["params"].text()
    assert "sovrapposizione 75% avanti e 70% laterale" in text
    assert text.count("% avanti") == 1 and "80%" not in text
    assert project.active_plan(ch)["piano"]["blocchi"][0]["avanti"] == 75


def test_high_resolution_off_for_new_plans_kept_for_old_ones(monkeypatch):
    from PySide6.QtWidgets import QApplication

    QApplication.instance() or QApplication([])
    from ppg_flightplan.pane import FlightPlanPane

    pane = FlightPlanPane()
    assert not pane.k_high.isChecked() and pane.parameters()["high"] is False
    pane.set_parameters({"camera": "mini4pro", "alta": True, "blocchi": []})      # a plan saved with it on
    assert pane.k_high.isChecked()
    pane.set_parameters({"camera": "mini4pro", "blocchi": []})                    # without the key: off, as read
    assert not pane.k_high.isChecked()


@pytest.mark.parametrize("lang", ["it", "en"])
def test_the_pane_fits_420_px(monkeypatch, lang):
    """The combos show their whole text and the table every column, the heights too, at 420 px."""
    from PySide6.QtWidgets import QApplication, QComboBox

    from pyarchinit_photogrammetry.gui import i18n

    app_ = QApplication.instance() or QApplication([])
    monkeypatch.setattr(i18n, "current", lambda: lang)
    from ppg_flightplan.pane import FlightPlanPane

    pane = FlightPlanPane()
    pane.resize(420, 1600)
    pane.show()
    pane.set_plan_table({"blocchi": [{"indice": 0, "nome": "A nadirale", "attivo": True, "scatti": 1234,
                                      "minuti": 45.3, "batterie": 2.6, "gsd": 3.0, "quote": "108.7–121.4",
                                      "tratte": [{"chiave": "k", "nome": "Piano_1_A_nadirale_tratta_12.kmz",
                                                  "scatti": 600, "minuti": 22.1, "quote": "108.7–121.4"}]}]})
    for _ in range(5):
        app_.processEvents()
    try:
        fm = pane.fontMetrics()
        for c in pane.findChildren(QComboBox):
            if c is pane.c_plan:
                continue
            longest = max(fm.horizontalAdvance(c.itemText(i)) for i in range(c.count()))
            assert c.width() >= longest + 16, (c.currentText(), c.width(), longest)
            assert c.mapTo(pane, c.rect().topRight()).x() <= pane.width(), c.currentText()
        table = pane.t_plan
        widths = [table.columnWidth(i) for i in range(table.columnCount())]
        assert sum(widths) <= table.viewport().width() and not table.horizontalScrollBar().isVisible()
        assert widths[-1] >= fm.horizontalAdvance("108.7–121.4")
        assert pane.c_direction.itemText(0) in ("Lungo il lato più lungo", "Along the longest side")
    finally:
        pane.close()
