"""The plugin's side: the command line through the program's ``ppg plugin masonry``, the pane in the program's window
(offscreen, the preferences in a temporary folder) with a face made through the context and its shape shown over the
elevation, the numbers typed in a dialog."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import importlib.util  # noqa: E402

import pyarchinit_photogrammetry as ppg  # noqa: E402

HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None

import masonry_samples as S  # noqa: E402
from ppg_masonry import parse_typed, walls  # noqa: E402

Z0 = S.WORLD[2]


def _plugin_folder(root: Path) -> Path:
    folder = root / "plugins"
    dist = folder / "masonry" / "ppg_masonry-0.1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: ppg-masonry\nVersion: 0.1.0\n")
    (dist / "entry_points.txt").write_text("[pyarchinit_photogrammetry.plugins]\nmasonry = ppg_masonry:MasonryPlugin\n")
    return folder


class FakePyArchInit:
    """What the plugin asks of the program's bridge, recorded."""

    def __init__(self):
        self.forms, self.rels, self.media = {}, [], []

    def us_record(self, sito, area, us, unita_tipo=None):
        return self.forms.get(str(us))

    def create_us(self, sito, area, us, unita_tipo="US", **fields):
        self.forms[str(us)] = {"sito": sito, "area": area, "unita_tipo": unita_tipo, **fields}
        return len(self.forms)

    def add_relationships(self, sito, area, us, rels, unita_tipo=None):
        self.rels += [(us, *r) for r in rels]
        return len(rels)

    def register_elevation_media(self, chunk, entry, paths, sito=None, link_us=None):
        self.media.append((paths["dxf"], paths["svg"], link_us))
        return [1, 2]


def test_typed_numbers():
    assert parse_typed("313,500; 314,250") == [313.5, 314.25]
    assert parse_typed("1.234,5 2,5") == [1234.5, 2.5]
    assert parse_typed("1,234.5; 2.5") == [1234.5, 2.5]
    assert parse_typed("") == []


def test_command_line(tmp_path, monkeypatch):
    from typer.testing import CliRunner

    from pyarchinit_photogrammetry.cli import app

    doc, ch = S.project(tmp_path, S.wall())
    other = S.add_chunk(doc, "Rilievo 2027", S.wall(offset=S.STEP))
    doc.save()
    monkeypatch.setenv("PPG_PLUGINS_DIR", str(_plugin_folder(tmp_path)))
    cli = CliRunner()
    project = str(doc.path)
    x0, y0 = S.WORLD[:2]
    line = f"{x0},{y0},{x0 + S.WALL_LENGTH},{y0}"
    r = cli.invoke(app, ["plugin", "masonry", "face", project, "Muro nord", "--line", line, "--side",
                         f"{x0 + 3},{y0 - 5}", "--resolution", "0.03", "--name", "Faccia sud"])
    assert r.exit_code == 0, r.output
    assert "Faccia sud" in r.output and "30,0 mm/px" in r.output
    r = cli.invoke(app, ["plugin", "masonry", "shape", project, "Muro nord"])
    assert r.exit_code == 0, r.output
    assert "Strapiombo 2,0°" in r.output and "bulge.tif" in r.output
    r = cli.invoke(app, ["plugin", "masonry", "shape", project, "Muro nord", "Faccia sud", "--sections",
                         f"{Z0 + 0.5},{Z0 + 2.5}", "--stations", "1,5"])
    assert r.exit_code == 0, r.output
    assert "+7,0 cm" in r.output and "sezioni.dxf" in r.output
    r = cli.invoke(app, ["plugin", "masonry", "shape", project, "Muro nord", "--compare", "Rilievo 2027"])
    assert r.exit_code == 0, r.output
    assert "mediana +2,0 cm" in r.output
    r = cli.invoke(app, ["plugin", "masonry", "list", project])
    assert r.exit_code == 0 and "Muro nord" in r.output and "Faccia sud" in r.output and "Strapiombo" in r.output
    ch = ppg.Document.open(project).chunk
    w = walls.need(ch, "Muro nord")
    res = walls.results(ch, w["id"], w["faces"][0])
    assert set(res) == {"shape", "sections", "surveys"} and "Rilievo 2027" in res["surveys"]
    r = cli.invoke(app, ["plugin", "masonry", "remove", project, "Muro nord", "--elevations"])
    assert r.exit_code == 0, r.output
    ch = ppg.Document.open(project).chunk
    assert walls.walls(ch) == [] and not ch.meta.get("elevations")
    r = cli.invoke(app, ["plugin", "masonry", "shape", project, "Nessuno"])
    assert r.exit_code != 0 and "Nessun muro" in r.output
    assert other.label == "Rilievo 2027"
    # stones and USM on a face of a new wall
    r = cli.invoke(app, ["plugin", "masonry", "face", project, "Muro est", "--line", line, "--side",
                         f"{x0 + 3},{y0 - 5}", "--resolution", "0.03"])
    assert r.exit_code == 0, r.output
    doc = ppg.Document.open(project)
    ch = doc.chunk
    eid = walls.need(ch, "Muro est")["faces"][0]
    units, notch = S.usm_layout(Z0)
    S.put_drawings(ch, eid, S.stone_pattern(Z0, hole=notch)[0], units)
    doc.save()
    r = cli.invoke(app, ["plugin", "masonry", "stones", project, "Muro est"])
    assert r.exit_code == 0, r.output
    assert "5 filari" in r.output and "pietre.xlsx" in r.output
    r = cli.invoke(app, ["plugin", "masonry", "usm", project, "Muro est", "--accept", "1,3"])
    assert r.exit_code == 0, r.output
    assert "3. USM" in r.output and "Accettati: 2" in r.output
    ch = ppg.Document.open(project).chunk
    accepted = walls.results(ch, walls.need(ch, "Muro est")["id"], eid)["usm"]["accepted_full"]
    assert len(accepted) == 2
    r = cli.invoke(app, ["plugin", "masonry", "plate", project, "Muro est", "--out", str(tmp_path / "tav")])
    assert r.exit_code == 0 and (tmp_path / "tav" / "tavola.pdf").exists(), r.output
    r = cli.invoke(app, ["plugin", "masonry", "export", project, "Muro est", "--out", str(tmp_path / "exp")])
    assert r.exit_code == 0, r.output
    assert any(p.name.endswith("_muratura.dxf") for p in (tmp_path / "exp").iterdir())
    r = cli.invoke(app, ["plugin", "masonry", "ifc", project, "Muro est", "--out", str(tmp_path / "m.ifc")])
    if HAS_IFC:
        assert r.exit_code != 0 and "spessore" in r.output
        r = cli.invoke(app, ["plugin", "masonry", "ifc", project, "Muro est", "--out", str(tmp_path / "m.ifc"),
                             "--thickness", "0.6"])
        assert r.exit_code == 0 and "2 relationships" in r.output, r.output


def test_the_pane_in_the_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_masonry import MasonryPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch = S.project(tmp_path, S.wall())
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False, check_updates=False)
    w.runner.show_dialog = False
    # the window's dialogs answered by the test: a failed task is said with its message, never a modal box
    errors, infos = [], []
    monkeypatch.setattr(w, "_error", lambda summary, details="", logged=True: errors.append(f"{summary}\n{details}"))
    monkeypatch.setattr(w, "_info", lambda title, text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "_warn", lambda text, *a, **k: infos.append(text))
    monkeypatch.setattr(w, "choice", lambda *a, **k: 0, raising=False)

    def wait():
        w.runner.wait()
        assert not errors, "a task of the window failed:\n" + "\n".join(errors) + "\nsaid: " + "\n".join(infos)
    plugin = MasonryPlugin()
    try:
        plugin.activate(Context("masonry", tmp_path / "plugin", window=w))
        monkeypatch.setattr(plugin, "_message", lambda kind, text: infos.append(f"{kind}: {text}"))
        pane = plugin.pane
        dock = w.panes.groups["right"].dock(pane)
        assert dock is not None and dock.objectName().startswith("pane_plugin_masonry")
        assert dock.windowTitle() == "Murature"
        assert pane.walls.count() == 0 and "Nessun muro" in pane.l_state.text()
        assert pane.minimumSizeHint().width() <= 340
        # a face traced on the plan through the context, the dialog answered with its defaults
        x0, y0 = S.WORLD[:2]
        monkeypatch.setattr(plugin.context, "request_line",
                            lambda done, hint="": done(np.array([[x0, y0], [x0 + S.WALL_LENGTH, y0]])))
        asked = []

        def answer(title, fields, intro=""):
            asked.append(title)
            values = {f.name: f.default for f in fields if f.kind != "heading"}
            if "resolution" in values:
                values["resolution"] = 0.03
            if "side" in values:
                values["side"] = "right"  # walking from x0 to x0 + 6 the viewer (-y) is on the right
            return values

        monkeypatch.setattr(plugin, "_ask", answer)
        plugin._requested("plane")
        wait()
        assert asked == ["Faccia su un piano"]
        rec = walls.walls(ch)[0]
        assert rec["name"] == "Muro 1" and len(rec["faces"]) == 1 and w.isWindowModified()
        assert pane.walls.count() == 1 and pane.table.rowCount() == 1
        assert pane.table.item(0, 2).text() == "6,000 m" and pane.table.item(0, 4).text() == "30,0 mm/px"
        assert w._raster_version == ("elevation", rec["faces"][0])  # the new face is shown
        entry = walls.elevation(ch, rec["faces"][0])
        assert np.allclose(np.cross(entry["frame"]["u_axis"], entry["frame"]["v_axis"]), [0, -1, 0], atol=1e-9)
        # its shape: the bulge over the elevation with its legend, the summary in the pane
        plugin._requested("shape")
        wait()
        assert plugin.overlay is not None and plugin.overlay.shown() and plugin.overlay.legend is not None
        assert "Strapiombo 2,0°" in pane.summary.toPlainText()
        plugin._requested("bulge")
        assert not plugin.overlay.shown()
        plugin._requested("bulge")
        assert plugin.overlay.shown()
        plugin._requested("sections")
        wait()
        assert asked[-1] == "Sezioni orizzontali" and "Sezioni a 4 quote" in pane.summary.toPlainText()
        items = plugin._tree_items(ch)
        assert items[0]["label"] == "Murature" and items[0]["children"][0]["children"][0]["label"] == entry["name"]
        # stones and USM drawn on the face (put in the project while the window shows the plan)
        plugin._need_plan()
        units, notch = S.usm_layout(Z0)
        S.put_drawings(ch, entry["id"], S.stone_pattern(Z0, hole=notch)[0], units)
        plugin._requested("measure_stones")
        wait()
        assert pane.stones_table.rowCount() == len(S.stone_pattern(Z0, hole=notch)[0])
        assert "Le misure suggeriscono" in pane.summary.toPlainText()
        offered = []
        monkeypatch.setattr(plugin, "_choose_relations", lambda ps: offered.append(ps) or ps[:2])
        plugin._requested("propose")
        assert len(offered[0]) == 3 and "Rapporti accettati" in pane.summary.toPlainText()
        said = []
        monkeypatch.setattr(plugin, "_message", lambda kind, text: said.append(text))
        plugin._requested("save_pyarchinit")
        assert "Collegati prima a PyArchInit" in said[-1]
        fake = FakePyArchInit()
        w.pa = fake
        plugin._requested("save_pyarchinit")
        wait()
        assert sorted(fake.forms) == ["10", "11", "12", "13"] and len(fake.rels) == 4 and fake.media
        assert "In PyArchInit: 4 schede nuove, 4 rapporti" in pane.summary.toPlainText()
        # plates and exports
        plugin._requested("plate")
        wait()
        face_dir = walls.face_folder(ch, rec["id"], entry["id"])
        assert (face_dir / "tavola.pdf").exists() and "Tavola in scala 1:" in said[-1]
        from pyarchinit_photogrammetry.io import qgis_link
        from PySide6.QtWidgets import QFileDialog

        monkeypatch.setattr(qgis_link, "find_qgis", lambda: None)
        plugin._requested("qgis")
        assert "QGIS non trovato" in said[-1]
        built = []
        monkeypatch.setattr(qgis_link, "find_qgis", lambda: object())
        monkeypatch.setattr(qgis_link, "build_project", lambda project, crs, layers, rasters, **k: built.append(
            (rasters, k["extra_vectors"])))
        monkeypatch.setattr(qgis_link, "open_in_qgis", lambda project, q: None)
        plugin._requested("qgis")
        wait()
        rasters, vectors = built[-1]
        assert {r["name"] for r in rasters} >= {entry["name"], "Spanciamento (cm)"}
        assert "pietre_misure" in {v["table"] for v in vectors}
        assert plugin._qgis_entries()[1][0]["table"] == "pietre_misure" and plugin._qgis_entries in w.qgis_providers
        monkeypatch.setattr(plugin, "_ask", lambda title, fields, intro="": {"thickness": 0.6})
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "muro.ifc"), ""))
        plugin._requested("ifc")
        wait()
        if HAS_IFC:
            assert (tmp_path / "muro.ifc").exists() and "IFC del muro" in said[-1]
        # the face taken out, the elevation kept
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
        plugin._requested("remove_face")
        assert walls.walls(ch)[0]["faces"] == [] and walls.elevation(ch, entry["id"]) is not None
        assert pane.table.rowCount() == 0
        plugin.deactivate()
        plugin.context.release()
        assert w.panes.dock(pane) is None
    finally:
        w.setWindowModified(False)
        w.close()
