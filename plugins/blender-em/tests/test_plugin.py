"""The plugin in the program: what it adds, the command line, and the pane in the window - the em.json
linked, the EM project written, the live update, a selection that comes from EMtools."""
from __future__ import annotations

import json

import pytest
from fake_host import FakeHost, free_port
from samples import emjson, project

from ppg_blender_em import BlenderEmPlugin, report_text, rows_for, scene, sync
from ppg_blender_em import emjson as em


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("blender-em")
    assert info is not None and info.active, info and info.error
    titles = [a.title for a in manager.actions if a.plugin == "blender-em"]
    assert titles == ["Pannello Blender ed EMtools", "Collega l'em.json del sito...", "Esporta nel progetto EM",
                      "Apri in Blender (aggiornato dal vivo)", "Importa i proxy da Blender",
                      "Scegli il programma Blender..."]
    assert "blender-em" in manager.commands


def test_the_version_is_the_distribution_s():
    import tomllib
    from pathlib import Path

    import ppg_blender_em

    data = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    assert ppg_blender_em.__version__ == data["project"]["version"]


def test_rows_and_report(tmp_path):
    doc = project(tmp_path, model=False)
    em_path = tmp_path / "EM" / "Scavo.em.json"
    emjson(em_path, units=[("1", "US", "1"), ("1", "USM", "2"), ("1", "US", "4")])
    rows = rows_for(doc.chunk, em.load(em_path))
    assert [(r["name"], r["epoch"], bool(r["proxy"]), r["node_id"] is not None) for r in rows] == [
        ("1.US1", "Tardoantico", True, True), ("1.USM2", "Età romana", True, True), ("1.US4", "Età romana", False, True),
        ("US3", "", True, False)]
    assert rows[3]["proxy"].endswith("non nella matrice") and "m³" in rows[0]["proxy"]
    text = report_text({"written": ["1.US1"], "not_in_matrix": ["US3"], "shift": [300000.0, 4500000.0, 99.0]})
    assert "Scritti 1: 1.US1" in text and "Non nella matrice 1" in text and "300000, 4500000, 99" in text


def test_the_command_line(tmp_path, capsys):
    doc = project(tmp_path)
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    emjson(em_path)
    plugin = BlenderEmPlugin()
    assert plugin.command("status", str(doc.path), "--emjson", str(em_path)) == 0
    assert "1.USM2\tUSM\tEtà romana" in capsys.readouterr().out
    assert plugin.command("export", str(doc.path), "--emjson", str(em_path), "--no-survey") == 0
    out = capsys.readouterr().out
    assert "Scritti 3: 1.US1, 1.USM2, 1.US3" in out
    assert sorted(p.name for p in (tmp_path / "Scavo" / "EM" / "proxies").glob("*.glb")) == [
        "1.US1.glb", "1.US3.glb", "1.USM2.glb"]
    assert plugin.command("export", str(doc.path)) == 2  # nowhere to write: said
    assert "--out FOLDER" in capsys.readouterr().out
    assert plugin.command("export", str(doc.path), "--out", str(tmp_path / "Altro")) == 0
    manifest = scene.read_manifest(tmp_path / "Altro")
    assert [o["name"] for o in manifest["objects"]] == ["US1", "USM2", "US3", "Rilievo Chunk 1"]


@pytest.fixture
def window(tmp_path, monkeypatch):
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")  # the window alone: the plugin is activated by the test
    qapp = QApplication.instance() or QApplication([])
    doc = project(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    problems = []
    monkeypatch.setattr(w, "_warn", lambda text: problems.append(text))
    monkeypatch.setattr(w, "_error", lambda *a, **k: problems.append(a))
    yield w, qapp, doc, problems
    w.setWindowModified(False)
    w.close()


def _wait(qapp, predicate, timeout=10.0):
    import time

    end = time.monotonic() + timeout
    while time.monotonic() < end:
        qapp.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


def test_the_pane_in_the_window(tmp_path, window, monkeypatch):
    from pyarchinit_photogrammetry.plugins import Context

    w, qapp, doc, problems = window
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    ids = emjson(em_path)
    plugin = BlenderEmPlugin()
    context = Context("blender-em", tmp_path / "plugin", window=w)
    messages = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text, details="": messages.append((kind, text, details)))
    try:
        plugin.activate(context)
        pane = plugin.pane
        assert pane.table.rowCount() == 3 and pane.table.item(0, 0).text() == "US1"  # no em.json: the program's names
        # the em.json linked (as the dialog does it), kept with the chunk relative to the project too
        plugin._store(doc.chunk, emjson=em_path, root=None)
        plugin.refresh_pane()
        stored = doc.chunk.meta["blender_em"]
        assert stored["emjson"] == str(em_path.resolve()) and stored["emjson_rel"] == "Scavo/EM/Scavo.em.json"
        assert [pane.table.item(r, 0).text() for r in range(pane.table.rowCount())] == ["1.US1", "1.USM2", "1.US3"]
        assert "3 unità, 2 epoche" in pane.l_matrix.text()
        # the EM project written, as the button does it
        pane.b_export.click()
        w.runner.wait()
        qapp.processEvents()
        assert not problems and messages and messages[-1][1] == "Progetto EM aggiornato."
        proxies = tmp_path / "Scavo" / "EM" / "proxies"
        assert sorted(p.name for p in proxies.glob("*.glb")) == ["1.US1.glb", "1.US3.glb", "1.USM2.glb"]
        assert (tmp_path / "Scavo" / "RB" / "Chunk 1" / "Chunk 1.glb").exists()
        # live: a proxy built again is written again by itself, without a progress window
        plugin.set_live(True)
        assert context.settings["live"] and plugin._timer.isActive()
        import trimesh

        entry = doc.chunk.meta["proxies"][0]
        f = doc.chunk.data_dir / entry["file"]
        mesh = trimesh.load(str(f), process=False, force="mesh")
        mesh.vertices[:, 2] -= 0.3
        mesh.export(str(f))
        plugin._live_tick()
        assert _wait(qapp, lambda: not plugin._live_busy)
        manifest = scene.read_manifest(tmp_path / "Scavo")
        assert manifest["revision"] == 2 and "Scritti 1: 1.US1" in pane.l_status.text()
        # the sync with a host of EMtools: a proxy picked in Blender selects its unit in the pane
        port = free_port()
        host = FakeHost(port)
        context.settings["port"] = port
        plugin.set_sync(True)
        try:
            assert _wait(qapp, lambda: host.kinds()[:2] == ["client_info", "request_snapshot"])
            assert host.received[0]["payload"]["graph_id"] == "Scavo"
            usm2 = next(k for k, v in ids.items() if v == ("1", "USM", "2"))
            host.send(sync.envelope("select", {"node_id": usm2}, source="emtools"))
            assert _wait(qapp, lambda: pane.node_ids() == [usm2])
            assert pane.l_unit.text().startswith("In Blender: 1.USM2 (USM, Età romana)")
            # and a unit picked in the pane selects its proxy in Blender
            pane.table.selectRow(2)
            us3 = next(k for k, v in ids.items() if v == ("1", "US", "3"))
            assert _wait(qapp, lambda: host.received[-1]["type"] == "select")
            assert host.received[-1]["payload"] == {"node_id": us3}
        finally:
            plugin.set_sync(False)
            host.close()
        assert not context.settings["sync"] and plugin.client is None
        assert json.loads((tmp_path / "plugin" / "settings.json").read_text())["live"] is True
    finally:
        plugin.deactivate()
        context.release()


def test_back_from_blender_and_proxies_made_by_emtools(tmp_path, window, monkeypatch):
    from shapely.geometry import box
    from test_inbox import _box, _send

    from pyarchinit_photogrammetry.archaeo.stratigraphy import UnitPolygon
    from pyarchinit_photogrammetry.plugins import Context
    from samples import ORIGIN

    w, qapp, doc, problems = window
    em_path = tmp_path / "Scavo" / "EM" / "Scavo.em.json"
    ids = emjson(em_path, units=[("1", "US", "1"), ("1", "USM", "2"), ("1", "US", "3"), ("1", "US", "4")])
    plugin = BlenderEmPlugin()
    context = Context("blender-em", tmp_path / "plugin", window=w)
    messages = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text, details="": messages.append((kind, text, details)))
    x0, y0, z0 = ORIGIN
    try:
        plugin.activate(context)
        pane = plugin.pane
        plugin._store(doc.chunk, emjson=em_path, root=None)
        plugin.set_cameras(False)
        report = scene.export_project(doc.chunk, plugin.folders(doc.chunk), plugin.matrix(doc.chunk), survey=False)
        shift = report["shift"]
        # USM 2 modelled in Blender and sent: Importa da Blender brings it into the chunk and the window
        folders = plugin.folders(doc.chunk)
        _send(folders, "1.USM2", _box(x0 + 1.8, y0 + 0.5, z0 - 0.4, x0 + 2.9, y0 + 2.5, z0 + 0.2), shift)
        pane.b_take.click()
        entry = next(e for e in doc.chunk.meta["proxies"] if e["name"] == "USM2")
        assert entry["base"] == "blender" and entry["volume_m3"] == pytest.approx(1.32)
        assert messages[-1][1] == "Proxy da Blender" and "Da Blender 1: USM2" in messages[-1][2]
        assert w.proxies_pane.table.item(1, 3).text() == "1.320" and w.isWindowModified()
        assert "1.320 m³" in [pane.table.item(r, 3).text() for r in range(pane.table.rowCount())]
        # US 4 is drawn but has no proxy: EMtools is asked to start one, as wide as the unit, on the DEM
        w.units.append(UnitPolygon(box(x0 + 3.2, y0 + 0.4, x0 + 4.2, y0 + 1.0), us="4", area="1", tipo="US",
                                   attributes={"_layer": "Unita' stratigrafiche"}, sito="Scavo"))
        port = free_port()
        host = FakeHost(port)
        context.settings["port"] = port
        plugin.set_sync(True)
        try:
            assert _wait(qapp, lambda: host.kinds()[:2] == ["client_info", "request_snapshot"])
            host.send(sync.envelope("host_info", {"tool": "Blender · EMtools", "accepts_commands": True},
                                    source="emtools"))
            assert _wait(qapp, lambda: pane.b_proxy.isEnabled())
            us4 = next(k for k, v in ids.items() if v == ("1", "US", "4"))
            row = next(r for r in range(pane.table.rowCount()) if pane.table.item(r, 0).text() == "1.US4")
            pane.table.selectRow(row)
            pane.b_proxy.click()
            assert _wait(qapp, lambda: host.kinds()[-1:] == ["command"])
            command = host.received[-1]["payload"]
            assert command["verb"] == "create_proxy_for_unit" and command["target"] == us4
            params = command["params"]
            assert params["size"] == pytest.approx(1.0)
            assert params["location"][:2] == pytest.approx([x0 + 3.7 - shift[0], y0 + 0.7 - shift[1]], abs=1e-3)
            assert params["location"][2] == pytest.approx(z0 + 0.2 - 0.5 - shift[2], abs=0.06)  # top on the DEM
            assert command["cmd_id"] == sync.make_cmd_id("create_proxy_for_unit", us4, params)
            host.send(sync.envelope("command_result", {"ok": True, "cmd_id": command["cmd_id"],
                                                       "info": {"proxy_object": "1.US4"}}, source="emtools"))
            assert _wait(qapp, lambda: "Proxy di 1.US4 creato in Blender" in pane.l_status.text())
            # EMtools that does not accept commands: said before anything is sent
            host.send(sync.envelope("host_info", {"accepts_commands": False}, source="emtools"))
            assert _wait(qapp, lambda: not pane.b_proxy.isEnabled())
            plugin.proxies_in_blender()
            assert "Accept commands from EMStudio" in messages[-1][1]
        finally:
            plugin.set_sync(False)
            host.close()
        assert not problems
    finally:
        plugin.deactivate()
        context.release()
