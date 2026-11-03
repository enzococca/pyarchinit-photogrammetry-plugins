"""The clinker step of the plugin: the strake model refused where too few stations give a width, the files of an
earlier model set aside when the lands are found again, the stations kept in the mesh, the Houdini import's
catch-all groups, and the step in the pane (the window offscreen, the dialogs answered by the test) and on the
program's command line."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hulls import clinker_project, project

from ppg_shipwreck import clinker, houdini, lines, ship, timbers

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import pyarchinit_photogrammetry as ppg  # noqa: E402

STATIONS = [round(2.01 + k, 2) for k in range(9)]


def _ready(ch, ends, stations=STATIONS):
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "end"))
    ship.set_stations(ch, [{"name": f"S{i + 1}", "x": x} for i, x in enumerate(stations)])
    lines.build(ch, half_width=3.0)


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "shipwreck" / "ppg_shipwreck-0.2.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-shipwreck\nVersion: 0.2.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\nshipwreck = ppg_shipwreck:ShipwreckPlugin\n")
    return folder


# ------------------------------------------------------------------------------------ the strake model refused
def test_the_plain_hull_has_no_strake_model(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends)
    clinker.detect_lands(ch)
    with pytest.raises(ship.NotReady, match="liscio"):
        clinker.build_strakes(ch)


def test_too_few_stations_with_a_width_refuse_the_strake_model(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    clinker.detect_lands(ch)
    state = ship.state(ch)["clinker"]
    # the counts differ at every station but two on port, and a third one jumps
    state["uncertain"]["port"] = [f"S{i}" for i in range(1, 7)]
    state["jumps"]["port"] = ["S7"]
    with pytest.raises(ship.NotReady) as err:
        clinker.build_strakes(ch)
    text = str(err.value)
    assert "Babordo" in text and "2 stazioni su 9" in text
    assert "6 con un altro numero di gradini" in text and "1 con battura o torello che saltano" in text
    assert "almeno 3" in text
    assert not (ship.folder(ch) / "clinker" / "strakes.obj").exists()
    state["jumps"]["port"] = []
    clinker.build_strakes(ch)  # three stations: enough


# ------------------------------------------------------------------------------------ the old model's files
def test_finding_the_lands_again_sets_the_old_strake_files_aside(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    clinker.write_tables(ch)
    out = houdini.export(ch, tmp_path / "hda")
    assert out["strakes"].exists()
    folder = ship.folder(ch) / "clinker"
    clinker.detect_lands(ch, {"min_step": 0.009})
    for name in ("strakes.obj", "strakes.ply", "strakes.csv", "strakes.xlsx"):
        assert not (folder / name).exists() and (folder / f"{name}.old").exists(), name
    for name in ("strakes.obj", "strakes.csv", "strakes.xlsx", "lands.obj"):
        assert not (out["readme"].parent / name).exists() and (out["readme"].parent / f"{name}.old").exists(), name
    text = clinker.summary(ch)
    assert "modello delle fasce precedente" in text and "strakes.obj.old" in text
    # a second detection with nothing new to set aside leaves the .old as they are, and says nothing more
    clinker.detect_lands(ch)
    assert (folder / "strakes.obj.old").exists() and "modello delle fasce precedente" not in clinker.summary(ch)


# ------------------------------------------------------------------------------------ the mesh keeps the stations
def test_a_station_near_a_grid_point_stays_in_the_mesh():
    lower = np.array([[0.0, 0.5, 0.0], [0.5005, 0.6, 0.1], [1.0, 0.5, 0.0]])
    upper = lower + [0.0, 0.2, 0.3]
    lo, hi = clinker._rows_between(lower, upper, [0.0, 0.5005, 1.0])
    assert 0.5005 in lo[:, 0] and 0.5 not in lo[:, 0]
    assert np.allclose(lo[lo[:, 0] == 0.5005, 1:], [0.6, 0.1])
    assert np.all(np.diff(lo[:, 0]) > 1e-3)


# ------------------------------------------------------------------------------------ the Houdini import
def test_a_catch_all_group_without_the_loose_faces_is_dropped(tmp_path):
    path = tmp_path / "planks.obj"
    path.write_text("\n".join([
        "v 0 0 0", "v 1 0 0", "v 0 1 0", "v 0 0 1", "v 1 1 0", "v 1 0 1", "v 9 9 9", "v 9 8 9", "v 8 9 9",
        "g fascia_B1 tutte", "f 1 2 3", "g fascia_T1 tutte", "f 4 5 6", "g", "f 7 8 9"]) + "\n")
    planks = houdini.read_obj(path)
    assert sorted(planks) == ["fascia_B1", "fascia_T1"]


def test_an_import_saved_by_an_earlier_version_is_kept(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    out = houdini.export(ch, tmp_path / "hda")
    ship.state(ch)["houdini"] = {"file": "vecchio.obj", "imported": "2026-11-03T10:00:00", "planks": [{"id": "1"}]}
    houdini.import_planks(ch, out["strakes"], deviation=False)
    store = ship.state(ch)["houdini"]
    assert store["planks"]["vecchio.obj"] == [{"id": "1"}] and "strakes.obj" in store["planks"]


# ------------------------------------------------------------------------------------ the command line
def test_the_clinker_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    doc.save()
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()
    p = str(doc.path)
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "status", p])
    assert r.exit_code == 0 and "non ancora rilevate" in r.output, r.output
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "strakes", p])
    assert r.exit_code == 1 and "Rileva prima i gradini" in r.output, r.output
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "detect", p, "--min-step", "8", "--min-strake", "0.08"])
    assert r.exit_code == 0, r.output
    assert "Babordo: 5 fasce" in r.output and "Tribordo: 5 fasce" in r.output and "lands.dxf" in r.output
    assert "Stazioni incerte (un altro numero di gradini): nessuna" in r.output and "Lati non letti: nessuno" in r.output
    state = ship.state(ppg.Document.open(p).chunk)["clinker"]
    assert state["params"]["min_step"] == pytest.approx(0.008) and state["lines"]
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "strakes", p, "--lap", "25", "--catalogue"])
    assert r.exit_code == 0, r.output
    assert "strakes.obj" in r.output and "strakes.xlsx" in r.output and "10 fasce nel catalogo" in r.output
    ch = ppg.Document.open(p).chunk
    assert ship.state(ch)["clinker"]["strakes"][0]["lap_mm"] == 25.0
    assert sum(r["kind"] == "plank" for r in timbers.timbers_of(ch)) == 10
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "export", p, "--out", str(tmp_path / "hda")])
    assert r.exit_code == 0 and "hull.obj" in r.output, r.output
    obj = tmp_path / "hda" / "clinker_boat_tool" / "strakes.obj"
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "import", p, "--obj", str(obj)])
    assert r.exit_code == 0 and "10 fasce importate" in r.output and "scostamenti" in r.output, r.output
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "status", p])
    assert r.exit_code == 0 and "aggiornati" in r.output and "Babordo: 5 fasce" in r.output, r.output
    # the sections cut again with other stations: the lands are stale and the status says so
    doc2 = ppg.Document.open(p)
    ch = doc2.chunk
    ship.set_stations(ch, [{"name": f"S{i + 1}", "x": x} for i, x in enumerate(STATIONS[:-1])])
    lines.build(ch, half_width=3.0)
    doc2.save()
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "status", p])
    assert r.exit_code == 0 and "da rilevare di nuovo" in r.output, r.output
    r = cli.invoke(app, ["plugin", "shipwreck", "clinker", "export", p])
    assert r.exit_code == 1 and "--out" in r.output, r.output


# ------------------------------------------------------------------------------------ the pane
def test_the_clinker_step_in_the_pane(tmp_path, monkeypatch):
    from PySide6.QtWidgets import QApplication, QFileDialog

    from ppg_shipwreck import ShipwreckPlugin
    from ppg_shipwreck.pane import ShipwreckPane
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    errors, infos = [], []
    monkeypatch.setattr(w, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(w, "_info", lambda title, text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "_warn", lambda text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "choice", lambda *a, **k: 0, raising=False)

    def wait():
        w.runner.wait()
        QApplication.processEvents()
        assert not errors, "a task of the window failed:\n" + "\n".join(errors) + "\nsaid: " + "\n".join(infos)

    plugin = ShipwreckPlugin()
    try:
        plugin.activate(Context("shipwreck", tmp_path / "plugin", window=w))
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(f"{kind}: {text}"))
        pane = plugin.pane
        assert isinstance(pane, ShipwreckPane)
        names = ("clinker_detect", "clinker_strakes", "clinker_catalogue", "clinker_export", "clinker_import",
                 "clinker_open")
        assert all(n in pane.buttons for n in names)
        assert pane.buttons["clinker_detect"].text() == "Rileva i corsi..."
        assert pane.buttons["clinker_export"].text() == "Per Clinker Boat Tool..."
        assert "non ancora rilevate" in pane.states["clinker"].text()
        # the strakes before the lands: said, not raised
        plugin._requested("clinker_strakes")
        assert "Rileva prima i gradini" in said[-1]
        asked = []

        def answer(title, fields, intro=""):
            asked.append((title, [f.label for f in fields]))
            return {f.name: f.default for f in fields if f.kind != "heading"}

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._requested("clinker_detect")
        wait()
        title, labels = asked[-1]
        assert title == "Rileva i corsi" and "Gradino minimo (mm)" in labels and "Larghezza minima di fascia (m)" in labels
        assert "Babordo: 5 fasce" in pane.states["clinker"].text()
        assert (ship.folder(ch) / "clinker" / "lands.dxf").exists()
        # the land lines as a layer of the raster view, a colour per side (no orthophoto here: the model's frame)
        plugin._need_view(ch)
        plugin.draw_layer()
        view = w.ortho_view
        assert view.tr is not None
        layer = view.layers.get("Fasce")
        assert layer is not None and len(layer["features"]) == 8
        assert {f[2]["side"] for f in layer["features"]} == {"port", "starboard"}
        plugin._requested("clinker_strakes")
        wait()
        assert asked[-1][0] == "Modello delle fasce" and (ship.folder(ch) / "clinker" / "strakes.xlsx").exists()
        assert "ipotesi" in pane.states["clinker"].text()
        plugin._requested("clinker_catalogue")
        assert pane.table.rowCount() == 10
        items = plugin._tree_items(ch)[0]["children"]
        assert any(i["label"] == "Fasce (5 per lato)" for i in items)
        out = tmp_path / "hda"
        out.mkdir()
        monkeypatch.setattr(QFileDialog, "getExistingDirectory", lambda *a, **k: str(out))
        plugin._requested("clinker_export")
        wait()
        assert (out / "clinker_boat_tool" / "hull.obj").exists()
        monkeypatch.setattr(QFileDialog, "getOpenFileName",
                            lambda *a, **k: (str(out / "clinker_boat_tool" / "strakes.obj"), ""))
        plugin._requested("clinker_import")
        wait()
        assert "strakes.obj" in ship.state(ch)["houdini"]["planks"]
        assert "Da Houdini" in pane.states["clinker"].text()
        opened = []
        monkeypatch.setattr(plugin, "_open", lambda path: opened.append(Path(path)))
        plugin._requested("clinker_open")
        assert opened == [ship.folder(ch) / "clinker"]
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
        w.deleteLater()
        app.document = None
