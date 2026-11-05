"""The plugin in the program's window (offscreen), on the fake server: the confirmation before the upload
(declined: nothing sent), the drawing as a layer "PhotoCAD: <name>" on the plan or on the elevation it
was made on, drawn again when the raster is loaded again, selectable; PhotoCAD polygons into US on the
plan and USM on the elevation (one undo entry); the key dialog."""
from __future__ import annotations

import os
import time

import fakecad
import pytest

from ppg_photocad import client


def _wait(w, qapp, seconds=60.0):
    t0 = time.time()
    while w.runner.busy and time.time() - t0 < seconds:
        qapp.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        qapp.processEvents()


@pytest.fixture
def window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_photocad import PhotoCADPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    monkeypatch.setenv("PHOTOCAD_API_KEY", fakecad.KEY)
    fake = fakecad.FakePhotoCAD(credits=9)
    monkeypatch.setattr(client, "_urlopen", fake)
    qapp = QApplication.instance() or QApplication([])
    doc, ch = fakecad.project(tmp_path / "p", square=[fakecad.SQUARE, fakecad.SQUARE2])
    entry = fakecad.add_elevation(ch, square=[fakecad.E_SQUARE, fakecad.E_SQUARE2])
    doc.meta["sito"] = "Saggio di prova"
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    plugin = PhotoCADPlugin()
    plugin.activate(Context("photocad", tmp_path / "plugin", window=w))
    try:
        yield w, plugin, ch, entry, fake, qapp, errors
    finally:
        plugin.deactivate()
        plugin.context.release()
        w.setWindowModified(False)
        w.close()


def _photocad_layers(view):
    return [n for n in view.layers if n.startswith("PhotoCAD: ") and not n.endswith(" (pietre)")]


def _stone_layers(view):
    return [n for n in view.layers if n.startswith("PhotoCAD: ") and n.endswith(" (pietre)")]


def test_plan_layer_units_and_elevation(window, monkeypatch):
    from pyarchinit_photogrammetry.gui import dialogs

    w, plugin, ch, entry, fake, qapp, errors = window
    v = w.ortho_view
    plugin.context.show_raster("ortho")
    answers = {"area": "all", "detailed": True, "remove_background": False, "three_d": False}
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda *a, **k: dict(answers))
    said = []
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: said.append(a[3]))
    # declined: nothing goes to photocad.ai
    texts = []
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: texts.append(a[3]) or None)
    plugin.convert_dialog("ortho")
    _wait(w, qapp)
    assert fake.calls("POST") == [] and "photocad.ai" in texts[0] and "Crediti: 9, conversioni gratuite: 5" in texts[0] and not errors
    # accepted
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    plugin.convert_dialog("ortho")
    _wait(w, qapp)
    assert not errors and len(fake.calls("POST")) == 1
    (layer,) = _photocad_layers(v)
    feats = v.features(layer)
    assert sorted(f["_geom"][0] for f in feats) == ["LineString", "Polygon", "Polygon"]
    assert _stone_layers(v) == [layer + " (pietre)"] and len(v.features(layer + " (pietre)")) == 2
    assert w.isWindowModified()
    # drawn again when the raster is loaded again (the DEM, then the orthophoto)
    plugin.context.show_raster("dem")
    qapp.processEvents()
    assert _photocad_layers(v) == [layer] and len(v.features(layer)) == 3
    v.set_layer_visible(layer, False)  # hidden in the Layer pane: stays hidden when the window refreshes
    plugin.draw_layers()
    qapp.processEvents()
    assert not v.layers[layer]["visible"]
    v.set_layer_visible(layer, True)
    # two polygons selected -> two US, one undo entry
    before, entries = len(w.units), w.history.index
    v.select_items([f[0] for f in v.layers[layer]["features"]])  # the line too: skipped and counted
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda *a, **k: {"which": "selected", "tipo": "US", "start": 3})
    plugin.units_dialog()
    assert len(w.units) == before + 2 and w.history.index == entries + 1
    assert sorted(u.us for u in w.units[-2:]) == ["3", "4"] and all(u.tipo == "US" for u in w.units[-2:])
    assert any("1" in s and ("aperte" in s or "chiuse" in s) for s in said)
    # the elevation: its own layer there, not on the plan; USM in its drawings
    w.show_elevation(entry)
    qapp.processEvents()
    assert _photocad_layers(v) == [] and _stone_layers(v) == []
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda *a, **k: dict(answers))
    plugin.convert_dialog("elevation")
    _wait(w, qapp)
    assert not errors
    (elayer,) = _photocad_layers(v)
    assert elayer != layer and len(v.features(elayer)) == 3
    rec = plugin.records()[-1]
    assert rec["frame"] == "e1"
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda *a, **k: {"which": "layer", "layer": rec["id"],
                                                                     "tipo": "USM", "start": 0, "from": "polygons"})
    n = len(w.frame_units)
    v.select_items([])
    plugin.units_dialog()
    assert len(w.frame_units) == n + 2 and all(u.tipo == "USM" for u in w.frame_units[-2:])
    w._store_drawings()
    assert len(ch.meta["drawings"]["frames"]["e1"]["units"]) == n + 2
    plugin.context.show_raster("ortho")
    qapp.processEvents()
    assert _photocad_layers(v) == [layer]
    # removed: the layer goes
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    plugin.remove_conversion(plugin.records()[0]["id"])
    assert _photocad_layers(v) == [] and _stone_layers(v) == [] and len(plugin.records()) == 1
    assert not errors


def test_the_key_dialog(window, _no_network_no_keychain, monkeypatch):
    from ppg_photocad.gui import KeyDialog

    w, plugin, ch, entry, fake, qapp, errors = window
    monkeypatch.delenv("PHOTOCAD_API_KEY")
    d = KeyDialog(w)
    assert "nessuna" in d.l_status.text().lower()
    d.e_key.setText("pcad_live_typed")
    d.b_save.click()
    assert _no_network_no_keychain.store == {("pyarchinit-photogrammetry", "photocad"): "pcad_live_typed"}
    assert "portachiavi" in d.l_status.text() and d.e_key.text() == ""
    d.b_forget.click()
    assert _no_network_no_keychain.store == {}
    d.close()


def _convert(window, monkeypatch, kind="ortho"):
    from pyarchinit_photogrammetry.gui import dialogs

    w, plugin, ch, entry, fake, qapp, errors = window
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda *a, **k: {"area": "all", "detailed": True,
                                                                     "remove_background": False, "three_d": False})
    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: None)
    plugin.convert_dialog(kind)
    _wait(w, qapp)
    assert not errors


def test_the_pane_is_docked_on_the_right_and_lists_the_conversions(window, monkeypatch):
    w, plugin, ch, entry, fake, qapp, errors = window
    pane = plugin.pane
    assert pane is not None
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QDockWidget

    dock = next(d for d in w.findChildren(QDockWidget) if d.widget() is pane or pane in d.findChildren(type(pane)))
    assert w.dockWidgetArea(dock) == Qt.DockWidgetArea.RightDockWidgetArea
    assert pane.table.rowCount() == 0 and not pane.l_empty.isHidden()
    plugin.context.show_raster("ortho")
    plugin.refresh_pane()
    assert "PHOTOCAD_API_KEY" in pane.l_key.text()
    assert pane.buttons["ortho"].isEnabled() and pane.buttons["dem"].isEnabled()
    assert not pane.buttons["elevation"].isEnabled()  # no elevation shown
    assert not pane.buttons["fetch"].isEnabled()  # no row selected
    _convert(window, monkeypatch)
    assert pane.table.rowCount() == 1 and pane.l_empty.isHidden()
    rec = plugin.records()[0]
    assert rec["name"] in pane.table.item(0, 0).text() and pane.table.item(0, 3).text() == "2"
    pane.table.selectRow(0)
    assert all(pane.buttons[n].isEnabled() for n in ("toggle", "fetch", "export", "remove"))
    # a conversion on the elevation: the elevation button is on, the row of that frame comes first
    w.show_elevation(entry)
    qapp.processEvents()
    assert pane.buttons["elevation"].isEnabled()
    _convert(window, monkeypatch, "elevation")
    assert pane.table.rowCount() == 2 and pane.table.item(0, 0).text().startswith("● ")
    assert pane.table.item(0, 0).data(Qt.ItemDataRole.UserRole) != rec["id"]


def test_a_row_toggles_its_layer_and_the_actions_get_its_id(window, monkeypatch):
    w, plugin, ch, entry, fake, qapp, errors = window
    pane, v = plugin.pane, w.ortho_view
    plugin.context.show_raster("ortho")
    _convert(window, monkeypatch)
    (layer,) = _photocad_layers(v)
    rid = plugin.records()[0]["id"]
    pane.table.selectRow(0)
    pane.buttons["toggle"].click()
    assert _photocad_layers(v) == [] and plugin.records()[0]["visible"] is False
    assert "nascosta" in pane.table.item(0, 0).text()
    pane.table.selectRow(0)
    pane.buttons["toggle"].click()
    assert _photocad_layers(v) == [layer] and plugin.records()[0]["visible"] is True
    calls = []
    monkeypatch.setattr(plugin, "fetch_dialog", lambda rid=None: calls.append(("fetch", rid)))
    monkeypatch.setattr(plugin, "export_dialog", lambda rid=None: calls.append(("export", rid)))
    monkeypatch.setattr(plugin, "free_drawings_dialog", lambda rid=None: calls.append(("free", rid)))
    monkeypatch.setattr(plugin, "units_dialog", lambda: calls.append(("units", None)))
    pane.table.selectRow(0)
    for n in ("fetch", "export", "units"):
        pane.buttons[n].click()
    if plugin.context is not None and not pane.buttons["free"].isHidden():
        pane.buttons["free"].click()
    assert calls[:3] == [("fetch", rid), ("export", rid), ("units", None)]
    assert len(calls) == 4 and calls[3] == ("free", rid)
    # remove acts on the row
    from pyarchinit_photogrammetry.gui import dialogs

    monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
    pane.table.selectRow(0)
    pane.buttons["remove"].click()
    assert plugin.records() == [] and pane.table.rowCount() == 0 and _photocad_layers(v) == []
    assert not errors


def test_the_menu_action_shows_the_pane(window):
    w, plugin, ch, entry, fake, qapp, errors = window
    shown = []
    plugin.context.show_pane = lambda p: shown.append(p)
    plugin.show_pane()
    assert shown == [plugin.pane]
