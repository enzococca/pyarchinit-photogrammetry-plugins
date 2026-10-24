"""The changes of 0.1.3, on the plugin API of the program 1.3.2: the units of the 3D model as the
model of a new chunk of the project, in the window and from the command line, back in their place in
the world (the offset of the OBJ added again), each a closed solid in its colour that the program's
cut closes on the plane; the window shows the new chunk and its model, and the project as modified.
A short section trace drawn with the drawing the window lends to plugins (a long one is traced in
``test_geoarchaeology.test_the_pane_in_the_window``)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from test_geoarchaeology import X0, Y0, project
from test_tables_and_display import write

import pyarchinit_photogrammetry as ppg
from ppg_geoarchaeology import boreholes as bh
from ppg_geoarchaeology import model
from ppg_geoarchaeology import surfaces as sf


def _program_before(version: tuple) -> bool:
    """True with a program older than ``version`` (its plugin API or a fix it needs)."""
    from pyarchinit_photogrammetry import __version__

    return tuple(int(p) for p in __version__.split(".")[:3]) < version


COLOURS = {(0x8B, 0x5A, 0x2B), (0xC9, 0xA5, 0x5C)}  # US 1 and US 2


def two_units(tmp_path: Path):
    """Three boreholes of two units (US 1 brown, US 2 ochre) on the DEM of the site, with the
    surfaces and the model of the units; the internal frame of the chunk is not the world (turned,
    scaled and moved), as in a photogrammetric chunk."""
    from pyarchinit_photogrammetry.core import geometry

    doc, ch = project(tmp_path)
    c, s = np.cos(np.radians(30.0)), np.sin(np.radians(30.0))
    T = geometry.identity()
    T[:3, :3] = 2.0 * np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T[:3, 3] = [500050.0, 4500040.0, 100.0]
    ch.transform = T
    t = write(tmp_path, "t.csv", "sondaggio;x;y;da;a;us;colore\n" + "".join(
        f"S{k};{500000 + 20 * k};4500040;{a};{b};{u};{col}\n" for k in (1, 2, 3)
        for a, b, u, col in ((0, 1, "US 1", "#8b5a2b"), (1, 2, "US 2", "#c9a55c"))))
    bh.import_files(ch, [t])
    sf.build(ch)
    model.build(ch)
    doc.save()
    return doc, ch


def farthest(new, ch) -> float:
    """How far (m) the vertices of the model of ``new`` lie, in the world, from the solids of the
    units of ``ch`` (their PLY files, in double precision)."""
    from scipy.spatial import cKDTree

    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    units = np.vstack([read_mesh(ch.data_dir / r["ply"]).xyz for r in model.registry(ch)["units"] if r.get("ply")])
    got = new.to_world(read_mesh(new.product_path("model")).xyz)
    return float(cKDTree(units).query(got)[0].max())


def test_the_units_as_the_model_of_a_new_chunk(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.engine import mesh_cut
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.io.mesh_import import closed_components
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")  # the window alone: the plugin is activated below
    qapp = QApplication.instance() or QApplication([])
    doc, ch = two_units(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    problems = []
    monkeypatch.setattr(w, "_warn", lambda text: problems.append(text))
    monkeypatch.setattr(w, "_error", lambda *a, **k: problems.append(a))
    plugin = GeoarchaeologyPlugin()
    context = Context("geoarchaeology", tmp_path / "plugin", window=w)
    try:
        plugin.activate(context)
        pane = plugin.pane
        assert pane.b_model_chunk.isEnabled() and pane.t_holes.rowCount() == 3
        if hasattr(context, "add_pane"):  # program 1.3.4: the pane is a dock of its own, on the right
            dock = w.panes.groups["right"].dock(pane)
            assert dock is not None and dock.objectName() == "pane_plugin_geoarchaeology"
            dock.close()
            plugin.show_pane()  # back also when the user closed it
            assert not dock.isHidden()
        w.setWindowModified(False)
        pane.b_model_chunk.click()  # in the background, as the window runs its steps
        w.runner.wait()
        qapp.processEvents()
        assert not problems
        assert [c.label for c in doc.chunks] == ["Carotaggi", "Unità geoarcheologiche - Carotaggi"]
        new = doc.chunks[1]
        assert w.chunk is new and new.crs == ch.crs and new.product_path("model").exists()
        # the 3D view shows the model of the new chunk, each unit in its colour
        assert w.tabs.currentWidget() is w.model_view and w.model_view.chunk is new and w.model_view.kind == "model"
        assert {tuple(int(v) for v in c) for c in w.model_view.sd.mesh_colors} == COLOURS
        assert "Taglia" in w.statusBar().currentMessage() and w.isWindowModified()
        assert pane.t_holes.rowCount() == 0  # the pane follows the active chunk
        # in their place in the world (the OBJ keeps the millimetres), each unit a closed solid
        assert farthest(new, ch) < 2e-3
        m = read_mesh(new.product_path("model"))
        assert (closed_components(m.xyz, m.faces) >= 0).all()
        # the program's cut (the Taglia button of the 3D view) closes both units on the plane, in their colours
        point, normal = mesh_cut.vertical((X0 + 30.25, Y0 + 20.0), (X0 + 30.25, Y0 + 60.0))
        report = new.cut_model(point, normal, keep="above", out=tmp_path / "ovest.ply")
        part = read_mesh(report["files"][0])
        assert (closed_components(part.xyz, part.faces) >= 0).all()
        assert {tuple(int(v) for v in c) for c in part.colors} == COLOURS
        assert part.xyz[:, 0].max() == pytest.approx(X0 + 30.25, abs=1e-3)  # the part west of the plane
        # back to the chunk of the boreholes: the pane shows them again
        w._activate_chunk(ch)
        assert pane.t_holes.rowCount() == 3
    finally:
        plugin.deactivate()
        context.release()
        if hasattr(w, "panes"):
            assert w.panes.dock(pane) is None  # the pane went with the plugin
        w.setWindowModified(False)
        w.close()


def test_the_units_in_a_new_chunk_from_the_command_line(tmp_path, capsys):
    from ppg_geoarchaeology import GeoarchaeologyPlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = two_units(tmp_path)
    plugin = GeoarchaeologyPlugin()
    plugin.activate(Context("geoarchaeology", tmp_path / "plugin"))
    p = str(doc.path)
    # the OBJ written around a round origin: the offset of its header puts the units back in their place
    assert plugin.command("model", p, "--offset", f"{X0},{Y0},100") == 0
    capsys.readouterr()
    assert plugin.command("model-chunk", p) == 0
    assert "Unità geoarcheologiche - Carotaggi: 2 units" in capsys.readouterr().out
    assert plugin.command("model-chunk", p, "--label", "Taglio") == 0
    assert plugin.command("model-chunk", p) == 0
    doc2 = ppg.Document.open(p)
    assert [c.label for c in doc2.chunks] == ["Carotaggi", "Unità geoarcheologiche - Carotaggi", "Taglio",
                                              "Unità geoarcheologiche - Carotaggi (2)"]
    assert doc2.chunk.label == "Carotaggi"  # the active chunk stays the same
    src, new = doc2.chunks[0], doc2.chunks[1]
    obj = src.data_dir / model.registry(src)["files"]["obj"]
    assert np.allclose(model.obj_offset(obj), [X0, Y0, 100.0])
    assert abs(float(obj.read_text().split("\nv ", 1)[1].split()[0])) <= 200  # small numbers in the file
    assert farthest(new, src) < 2e-3 and new.crs == src.crs
    # without a model: an error, and no chunk
    assert plugin.command("remove", p, "model") == 0
    capsys.readouterr()
    assert plugin.command("model-chunk", p) == 1 and "error" in capsys.readouterr().out
    assert len(ppg.Document.open(p).chunks) == 4


def test_the_offset_of_the_obj(tmp_path):
    obj = tmp_path / "unita.obj"
    obj.write_text("# pyarchinit-photogrammetry geoarchaeology\n# offset 600.000 800.000 -5.000\nmtllib unita.mtl\n")
    assert model.obj_offset(obj).tolist() == [600.0, 800.0, -5.0]
    obj.write_text("# another program\nv 0 0 0\n")
    assert model.obj_offset(obj).tolist() == [0.0, 0.0, 0.0]
    (tmp_path / "unita.obj.offset.txt").write_text("1.5 2.5 3.5\n")
    assert model.obj_offset(obj).tolist() == [1.5, 2.5, 3.5]


@pytest.mark.xfail(_program_before((1, 3, 3)), strict=False,
                   reason="before program 1.3.3 a vertex less than ~5 m (x) and ~45 m (y) from the previous one in UTM "
                          "coordinates counted as the same vertex: a short north-south trace could not be ended")
def test_a_short_north_south_trace(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication

    from ppg_geoarchaeology import GeoarchaeologyPlugin, section
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    qapp = QApplication.instance() or QApplication([])
    doc, ch = two_units(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    w.resize(1200, 800)
    w.show()
    plugin = GeoarchaeologyPlugin()
    try:
        plugin.activate(Context("geoarchaeology", tmp_path / "plugin", window=w))
        monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
            lambda parent, title, fields, intro="": {"name": "N-S", "distance": 30.0, "scale": None,
                                                     "exaggeration": None}))
        monkeypatch.setattr(plugin, "_run", lambda title, job, done=None: done(job(None)) if done else job(None))
        plugin.draw_section()
        view = w.ortho_view
        qapp.processEvents()
        for x, y in ((X0 + 40.0, Y0 + 25.0), (X0 + 40.0, Y0 + 55.0)):
            QTest.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier,
                             view.mapFromScene(view.scene_pt(x, y)))
        QTest.keyClick(view, Qt.Key.Key_Return)
        (s,) = section.entries(ch)
        assert s["name"] == "N-S" and s["length"] == pytest.approx(30.0, abs=0.3)
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()
