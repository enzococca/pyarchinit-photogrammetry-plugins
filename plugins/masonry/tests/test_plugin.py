"""The plugin's side: the command line through the program's ``ppg plugin masonry``, the pane in the program's window
(offscreen, the preferences in a temporary folder) with a face made through the context and its shape shown over the
elevation, the numbers typed in a dialog."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the plugin side needs the program")

import pyarchinit_photogrammetry as ppg  # noqa: E402

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
    plugin = MasonryPlugin()
    try:
        plugin.activate(Context("masonry", tmp_path / "plugin", window=w))
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
        w.runner.wait()
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
        w.runner.wait()
        assert plugin.overlay is not None and plugin.overlay.shown() and plugin.overlay.legend is not None
        assert "Strapiombo 2,0°" in pane.summary.toPlainText()
        plugin._requested("bulge")
        assert not plugin.overlay.shown()
        plugin._requested("bulge")
        assert plugin.overlay.shown()
        plugin._requested("sections")
        w.runner.wait()
        assert asked[-1] == "Sezioni orizzontali" and "Sezioni a 4 quote" in pane.summary.toPlainText()
        items = plugin._tree_items(ch)
        assert items[0]["label"] == "Murature" and items[0]["children"][0]["children"][0]["label"] == entry["name"]
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
