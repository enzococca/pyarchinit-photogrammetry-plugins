"""The resistivity tomography in the program's window (offscreen) and on the command line, with the
stand-in worker: the ERT tab, a profile imported, its line drawn, the pseudosection with readings
deleted in the view, the inversion, the section in the 2D view with its legend, the 3D scene, the
line on the plan."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ppg_geophysics import ert, ert_ops, ert_section
from test_ert import FAKE, _project, write_ohm

LINE = [[500010.0, 4500050.0], [500040.0, 4500050.0]]


@pytest.fixture
def fake(monkeypatch):
    monkeypatch.setattr(ert_ops, "WORKER", FAKE)
    monkeypatch.setattr(ert_ops, "WORKER_PYTHON", Path(sys.executable))
    monkeypatch.delenv("PPG_ERT_FAKE", raising=False)


@pytest.fixture
def window(tmp_path, monkeypatch, fake):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geophysics import GeophysicsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    qapp = QApplication.instance() or QApplication([])
    doc, ch = _project(tmp_path)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))
    w.resize(1300, 850)
    w.show()
    qapp.processEvents()
    plugin = GeophysicsPlugin()
    plugin.activate(Context("geophysics", tmp_path / "plugin", window=w))
    messages = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text: messages.append((kind, text)))
    try:
        yield w, plugin, ch, qapp, errors, messages
    finally:
        plugin.deactivate()
        plugin.context.release()
        w.setWindowModified(False)
        w.close()


def _answer(monkeypatch, values: dict):
    """The plugin's ParamDialogs accepted with their defaults and ``values``."""
    from pyarchinit_photogrammetry.gui import dialogs

    def ask(parent, title, fields, intro=""):
        out = {f.name: f.default for f in fields if f.kind != "heading"}
        out.update(values)
        return out

    monkeypatch.setattr(dialogs.ParamDialog, "ask", ask)


def _run(w, qapp):
    w.runner.wait()
    for _ in range(5):
        qapp.processEvents()


def test_the_steps_in_the_window(window, monkeypatch, tmp_path):
    w, plugin, ch, qapp, errors, messages = window
    pane = plugin.pane.ert
    assert plugin.pane.tabs.count() == 2 and plugin.pane.tabs.tabText(1) == "ERT"
    assert pane.c_profile.count() == 0 and not pane.b_invert.isEnabled()
    # 1. import
    _answer(monkeypatch, {"file": str(write_ohm(tmp_path / "linea.ohm")), "name": "P1"})
    plugin.ert_import_dialog()
    _run(w, qapp)
    assert not errors and pane.c_profile.currentText() == "P1" and pane.b_invert.isEnabled()
    assert w.isWindowModified()
    pid = pane.selected_id()
    assert "24 elettrodi" in pane.steps[0].state.text()
    # 2. the line drawn on the DEM with the program's drawing lent to the plugin
    plugin.ert_draw_line(pid)
    assert w.ortho_view.kind == "dem" and w.ortho_view.mode == "req_line"
    _answer(monkeypatch, {"first": 0.0, "spacing": 1.0, "heights": "dem"})
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    v = w.ortho_view
    QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(*LINE[0])))
    QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(*LINE[1])))
    QTest.mouseDClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(*LINE[1])))
    qapp.processEvents()
    e = ert.entry_of(ch, pid)
    assert "placement" in e, (messages, errors, v.mode)
    assert e["placement"]["heights"] == "dem" and e["placement"]["method"] == "drawn"
    assert e["placement"]["length"] == pytest.approx(23.0) and "Linea di 23.0 m" in pane.steps[1].state.text()
    assert ert_section.number(e["placement"]["z_range"][0]) in pane.steps[1].state.text()
    assert "Profili ERT" in w.ortho_view.layers and "Elettrodi ERT" in w.ortho_view.layers
    # 3. the pseudosection: the readings as points; two deleted in the view leave the inversion
    plugin.ert_show_pseudo(pid)
    v = w.ortho_view
    assert v.kind == "frame" and len(v.features("Pseudosezione ERT")) == e["readings"]
    feats = [{"lettura": 3}, {"lettura": 4}]
    plugin._ert_readings_deleted(feats)
    e = ert.entry_of(ch, pid)
    assert e["filters"]["manual"] == [2, 3] and e["valid"] == e["readings"] - 2
    assert "tolte a mano 2" in pane.steps[2].state.text()
    # 4. the inversion: the section shown in the 2D view with the electrodes and the legend
    _answer(monkeypatch, {"lam": 15.0})
    plugin.ert_invert_dialog(pid)
    _run(w, qapp)
    assert not errors and not [m for m in messages if m[0] != "info"], (errors, messages)
    e = ert.entry_of(ch, pid)
    assert e["inversion"]["params"]["lam"] == 15.0 and e["inversion"]["fit"]["readings"] == e["readings"] - 2
    assert Path(v.path) == ert_section.view_path(ch, pid) and plugin._ert_shown == ("section", pid)
    assert "Elettrodi ERT" in v.layers and plugin._ert_legend is not None and plugin._ert_legend.scene() is v.scene()
    assert "χ² 1.00" in pane.steps[3].state.text() and pane.legend.isVisible()
    # the coordinates of the view are those of the section: u along the line, v the elevation
    x, y = v.world(v.sceneRect().topLeft())
    assert x <= 0.0 + 1e-6 and 100.0 < y < 103.0
    # another palette: the section written again and shown again
    plugin.ert_set_palette(pid, "viridis")
    assert ert.entry_of(ch, pid)["inversion"]["display"]["palette"] == "viridis"
    # 5. the 3D view: a file of the plugin with the section and the ground cut away
    assert plugin.ert_show_3d(pid)
    assert w.model_view.paths.get("model", "").endswith(ert_section.SCENE)
    # the line on the plan again
    plugin.ert_show_plan(pid)
    assert v.kind == "dem" and "Profili ERT" in v.layers and plugin._ert_shown is None
    # 6. export everything into a folder
    _answer(monkeypatch, {"folder": str(tmp_path / "esportati")})
    plugin.ert_export_dialog(pid)
    _run(w, qapp)
    assert messages and "Scritti" in messages[-1][1] and (tmp_path / "esportati" / "P1_sezione.tif").exists()


def test_remove_and_the_pane_follows(window, monkeypatch, tmp_path):
    w, plugin, ch, qapp, errors, messages = window
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "a.ohm"), name="A")
    ert_ops.import_profile(ch, write_ohm(tmp_path / "b.ohm"), name="B")
    plugin.refresh_pane()
    pane = plugin.pane.ert
    assert [pane.c_profile.itemText(i) for i in range(pane.c_profile.count())] == ["A", "B"]
    from pyarchinit_photogrammetry.gui import dialogs

    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    plugin.ert_remove_dialog(e["id"])
    assert [x["name"] for x in ert.entries(ch)] == ["B"] and pane.c_profile.count() == 1
    assert not ert.folder(ch, e["id"]).exists()


def test_the_command_line(tmp_path, fake, capsys):
    from ppg_geophysics import GeophysicsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = _project(tmp_path)
    doc.save()
    app.document = None
    plugin = GeophysicsPlugin()
    plugin.activate(Context("geophysics", tmp_path / "plugin"))
    project = str(tmp_path / "ert.ppg")
    src = write_ohm(tmp_path / "linea.ohm")
    assert plugin.command("ert", "import", project, str(src), "--name", "L1") == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["name"] == "L1" and out["array"] == "dipole-dipole"
    assert plugin.command("ert", "place", project, "L1", "--line", "500010,4500050;500040,4500050") == 0
    pl = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert pl["heights"] == "dem" and pl["length"] == pytest.approx(23.0)
    assert plugin.command("ert", "filter", project, "L1", "--spikes", "6", "--remove", "1,2") == 0
    f = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert f["manual"] == [0, 1] and f["rules"]["spikes"] == 6.0
    assert plugin.command("ert", "invert", project, "L1", "--lam", "30", "--palette", "earth") == 0
    inv = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert inv["display"]["palette"] == "earth" and inv["fit"]["chi2"] == 1.0
    assert plugin.command("ert", "export", project, "L1", str(tmp_path / "s.png")) == 0
    assert (tmp_path / "s.png").stat().st_size > 1000
    assert plugin.command("ert", "export", project, "L1", str(tmp_path / "tutto")) == 0
    assert (tmp_path / "tutto" / "L1_celle.csv").exists()
    assert plugin.command("ert", "scene", project, "--terrain", "dem") == 0
    assert plugin.command("ert", "list", project) == 0
    assert "L1: 24 elettrodi" in capsys.readouterr().out
    assert plugin.command("ert", "remove", project, "L1") == 0
    import pyarchinit_photogrammetry as ppg

    assert ert.entries(ppg.Document.open(project).chunk) == []
    assert plugin.command("ert", "place", project, "nessuno", "--line", "0,0;1,1") == 1


def test_undo_takes_a_change_of_the_profile_back(window, tmp_path):
    w, plugin, ch, qapp, errors, messages = window
    if w.history is None:
        pytest.skip("a program without Undo")
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "a.ohm"), name="A")
    plugin._ert_changed(e["id"])
    qapp.processEvents()
    plugin.ert_show_pseudo(e["id"])
    plugin._ert_readings_deleted([{"lettura": 1}, {"lettura": 2}, {"lettura": 3}])
    qapp.processEvents()
    assert ert.entry_of(w.chunk, e["id"])["filters"]["manual"] == [0, 1, 2]
    w.undo()
    qapp.processEvents()
    assert ert.entry_of(w.chunk, e["id"])["filters"].get("manual", []) == []
    assert "111 letture valide su 111" in plugin.pane.ert.steps[2].state.text()
