"""The resistivity tomography in the program's window (offscreen) and on the command line, with the
stand-in worker: the ERT tab, a profile imported, its line drawn, the pseudosection with readings
deleted in the view, the inversion, the section in the 2D view with its legend, the 3D scene, the
line on the plan."""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
    assert e["placement"]["length"] == pytest.approx(23.0, abs=0.01) and "Linea di 23.0 m" in pane.steps[1].state.text()
    assert ert_section.number(e["placement"]["z_range"][0]) in pane.steps[1].state.text()
    assert "Profili ERT" in w.ortho_view.layers and "Elettrodi ERT" in w.ortho_view.layers
    # 3. the pseudosection: the readings as one item of points (not a feature each); two inside an area drawn
    # on it with the program's drawing leave the inversion
    plugin.ert_show_pseudo(pid)
    v = w.ortho_view
    assert v.kind == "frame" and plugin._ert_readings.count == e["readings"] and not v.features("Pseudosezione ERT")
    assert plugin._ert_readings.scene() is v.scene()
    plugin.ert_remove_readings(pid)
    assert v.mode.startswith("req"), v.mode
    box = [(3.2, -0.42), (4.8, -0.42), (4.8, -0.6), (3.2, -0.6)]  # the readings 3 and 4 (pseudo-depth 0.51 m)
    for x, y in box:
        QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(x, y)))
    QTest.mouseDClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(*box[-1])))
    qapp.processEvents()
    e = ert.entry_of(ch, pid)
    assert e["filters"]["manual"] == [2, 3] and e["valid"] == e["readings"] - 2, messages
    assert "tolte a mano 2" in pane.steps[2].state.text()
    assert plugin._ert_readings is not None and plugin._ert_readings.scene() is v.scene()  # drawn again
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
    _run(w, qapp)
    assert ert.entry_of(ch, pid)["inversion"]["display"]["palette"] == "viridis"
    assert plugin._ert_shown == ("section", pid) and Path(v.path) == ert_section.view_path(ch, pid)
    # the blanking changed without inverting again
    fit_date = ert.entry_of(ch, pid)["inversion"]["date"]
    _answer(monkeypatch, {"method": "none"})
    plugin.ert_blank_dialog(pid)
    _run(w, qapp)
    inv = ert.entry_of(ch, pid)["inversion"]
    assert inv["display"]["blank"]["method"] == "none" and inv["date"] == fit_date and not errors
    # 5. the 3D view: a file of the plugin with the section and the ground cut away (built in the background)
    assert plugin.ert_show_3d(pid)
    _run(w, qapp)
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
    plugin._ert_changed()  # as the import in the window: a step of Undo
    qapp.processEvents()
    pane = plugin.pane.ert
    assert [pane.c_profile.itemText(i) for i in range(pane.c_profile.count())] == ["A", "B"]
    from pyarchinit_photogrammetry.gui import dialogs

    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    plugin.ert_remove_dialog(e["id"])
    assert [x["name"] for x in ert.entries(ch)] == ["B"] and pane.c_profile.count() == 1
    assert not ert.folder(ch, e["id"]).exists()
    if w.history is not None:  # Undo: the record back, and its files with it
        qapp.processEvents()
        w.undo()
        qapp.processEvents()
        assert [x["name"] for x in ert.entries(w.chunk)] == ["A", "B"] and pane.c_profile.count() == 2
        assert ert.folder(w.chunk, e["id"]).is_dir() and ert.load(w.chunk, e["id"]).count == e["readings"]


def test_the_line_from_the_coordinates_of_the_file_in_the_window(window, monkeypatch, tmp_path):
    w, plugin, ch, qapp, errors, messages = window
    data = Path(__file__).resolve().parent / "data"
    e = ert_ops.import_profile(ch, data / "gimli_world.dat", name="W")
    plugin.refresh_pane()
    _answer(monkeypatch, {"method": "file", "heights": "file"})
    plugin.ert_place_dialog(e["id"])
    assert ert.entry_of(ch, e["id"])["placement"]["method"] == "file", messages
    f = tmp_path / "local.dat"
    f.write_text("4\n# x y z\n0 0 10\n1 0.5 10.1\n2 1 10.2\n3 1.5 10.3\n1\n# a b m n rhoa\n1 2 3 4 10\n0\n")
    e = ert_ops.import_profile(ch, f, name="L")
    plugin.refresh_pane()
    plugin.ert_place_dialog(e["id"])
    assert "placement" not in ert.entry_of(ch, e["id"]) and "lontane dal rilievo" in messages[-1][1]


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
    assert pl["heights"] == "dem" and pl["length"] == pytest.approx(23.0) and pl["along"] == "plan"
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
    plugin._ert_remove(e["id"], [0, 1, 2])
    qapp.processEvents()
    assert ert.entry_of(w.chunk, e["id"])["filters"]["manual"] == [0, 1, 2]
    w.undo()
    qapp.processEvents()
    assert ert.entry_of(w.chunk, e["id"])["filters"].get("manual", []) == []
    assert "111 letture valide su 111" in plugin.pane.ert.steps[2].state.text()


def test_the_legend_is_one_image_made_once(window):
    from PySide6.QtGui import QColor

    from ppg_geophysics.ert_pane import legend_pixmap

    a = legend_pixmap(10.0, 1000.0, "rainbow", 300, "Ω·m")
    assert legend_pixmap(10.0, 1000.0, "rainbow", 300, "Ω·m").cacheKey() == a.cacheKey()  # cached
    img = a.toImage()
    near = lambda c, rgb: all(abs(a - b) <= 4 for a, b in zip(QColor(c).getRgb()[:3], rgb))  # noqa: E731
    assert img.width() == 300 and near(img.pixel(7, 12), (0, 0, 128))  # the first colour of the ramp
    assert near(img.pixel(292, 12), (120, 0, 90))


def test_another_frame_and_deactivate_leave_no_ert_drawing(window, monkeypatch, tmp_path):
    w, plugin, ch, qapp, errors, messages = window
    e = ert_ops.import_profile(ch, write_ohm(tmp_path / "a.ohm"), name="A")
    plugin.refresh_pane()
    plugin.ert_show_pseudo(e["id"])
    v = w.ortho_view
    assert plugin._ert_readings is not None and plugin._ert_legend is not None
    plugin.context.show_frame((0.0, 0.0, 50.0, 20.0), 0.05, "")  # another plugin's frame
    qapp.processEvents()
    assert plugin._ert_readings is None or _gone(plugin._ert_readings)
    plugin.ert_show_pseudo(e["id"])
    legend, readings = plugin._ert_legend, plugin._ert_readings
    plugin.deactivate()
    assert plugin._ert_legend is None and legend.scene() is None
    assert plugin._ert_readings is None and readings.scene() is None


def _gone(item) -> bool:
    try:
        return item.scene() is None
    except RuntimeError:  # deleted with the scene
        return True


def test_the_readings_item_paints_its_dots(window):
    import numpy as np
    from PySide6.QtCore import QRectF
    from PySide6.QtGui import QColor, QImage, QPainter
    from PySide6.QtWidgets import QGraphicsScene

    from ppg_geophysics import ert_view

    pts = np.array([[10.0, 10.0], [30.0, 10.0], [50.0, 10.0]])
    g = ert_view.groups(np.array([0.0, 1.0, 0.5]), lambda xs: ert_section.ramp(xs, "rainbow"),
                        np.array([True, True, False]))
    assert [len(i) for _rgb, i in g] == [1, 1, 1] and g[0][0] == ert_view.REMOVED
    scene = QGraphicsScene()
    item = ert_view.ReadingsItem(pts, g, electrodes=[[30.0, 30.0]])
    scene.addItem(item)
    img = QImage(64, 40, QImage.Format.Format_ARGB32)
    img.fill(QColor("white"))
    p = QPainter(img)
    scene.render(p, QRectF(0, 0, 64, 40), QRectF(0, 0, 64, 40))
    p.end()
    rgb = lambda x, y: QColor(img.pixel(x, y)).getRgb()[:3]  # noqa: E731
    assert rgb(10, 10)[2] > 100 and rgb(10, 10)[0] < 60  # blue: the low end
    assert rgb(50, 10) == ert_view.REMOVED and rgb(30, 30) == ert_view.ELECTRODE
    assert ert_view.inside(pts, [[0, 0], [40, 0], [40, 20], [0, 20]]).tolist() == [True, True, False]
