"""The plugin in the program: found by its entry point with its actions, command and export
formats; the command line on a project (search, fetch, series, composite, crop marks, list,
export, remove; a local chunk refused); the pane in an offscreen window (search and reading in
the background, the chart, the overlay on the DEM and on the empty frame of the area, a crop-mark
map, an area drawn with the program's polygon tool, a product removed after a question)."""
from __future__ import annotations

import time

import pytest
import s2synth

from ppg_sentinel2 import project, stac


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("sentinel2")
    assert info is not None and info.active, info and info.error
    titles = [a.title for a in manager.actions if a.plugin == "sentinel2"]
    assert titles[:3] == ["Pannello Sentinel-2", "Cerca immagini Sentinel-2...", "Disegna l'area di Sentinel-2"]
    assert "sentinel2" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "sentinel2"] == [".csv", ".tif"]
    manager.unload()


def _plugin(tmp_path, window=None):
    from ppg_sentinel2 import Sentinel2Plugin
    from pyarchinit_photogrammetry.plugins import Context

    plugin = Sentinel2Plugin()
    plugin.activate(Context("sentinel2", tmp_path / "plugin", window=window))
    return plugin


def test_command(tmp_path, capsys, monkeypatch):
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.app import app

    items = s2synth.world(tmp_path / "cogs")
    monkeypatch.setattr(stac, "_urlopen", s2synth.FakeCatalog(items))
    doc, ch = s2synth.project(tmp_path)
    app.document = None
    plugin = _plugin(tmp_path)
    path = str(doc.path)
    assert plugin.command("search", path, "--start", "2023-01-01", "--end", "2023-12-31", "--catalog",
                          s2synth.CATALOG) == 0
    out = capsys.readouterr().out
    assert '"acquisitions": 6' in out and "20230614_S2B" in out and "32TMR,32TNR" in out
    assert plugin.command("fetch", path, "--workers", "2") == 0
    assert '"read": 5' in capsys.readouterr().out
    assert plugin.command("series", path, "--index", "NDVI") == 0
    out = capsys.readouterr().out
    assert out.startswith("# Contains modified Copernicus Sentinel data 2023") and "2023-06-14  NDVI" in out
    assert plugin.command("series", path, "--monthly") == 0 and "2023-07" in capsys.readouterr().out
    assert plugin.command("cropmarks", path, "--period", "2023-summer", "--background", "60") == 0
    assert '"cropmarks_NDVI_2023_summer_60m"' in capsys.readouterr().out
    assert plugin.command("composite", path, "--index", "NDMI", "--period", "2023-06") == 0
    assert plugin.command("cropmarks", path, "--period", "midsummer") == 1
    assert "unknown period" in capsys.readouterr().out
    assert plugin.command("list", path) == 0 and '"composite_NDMI_2023_06"' in capsys.readouterr().out
    assert plugin.command("export", path, str(tmp_path / "out")) == 0
    assert (tmp_path / "out" / "sentinel2_stack.tif").exists() and (tmp_path / "out" / "sentinel2_series.csv").exists()
    assert plugin.command("export", path, str(tmp_path / "stack.tif")) == 0 and (tmp_path / "stack.tif").exists()
    assert plugin.command("export", path, str(tmp_path / "serie.csv")) == 0 and (tmp_path / "serie_monthly.csv").exists()
    assert plugin.command("remove", path, "composite_NDMI_2023_06") == 0
    assert plugin.command("remove", path, "composite_NDMI_2023_06") == 1
    capsys.readouterr()
    back = ppg.Document.open(doc.path).chunk  # every command saved the project
    assert [p["id"] for p in project.products(back)] == ["cropmarks_NDVI_2023_summer_60m"]
    # a local chunk: said, not a traceback
    local = ppg.Document()
    local.add_chunk("Locale")
    local.save(tmp_path / "locale.ppg")
    assert plugin.command("search", str(local.path)) == 1
    assert "LOCAL" in capsys.readouterr().out
    assert plugin.command("fetch", str(local.path)) == 1 and "cerca prima" in capsys.readouterr().out


def _wait(window, app_, seconds: float = 60.0) -> None:
    t0 = time.time()
    while window.runner.busy and time.time() - t0 < seconds:
        app_.processEvents()
        time.sleep(0.01)
    for _ in range(5):
        app_.processEvents()


def test_the_pane_in_the_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QDate, QSettings
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    app_ = QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    items = s2synth.world(tmp_path / "cogs")
    monkeypatch.setattr(stac, "_urlopen", s2synth.FakeCatalog(items))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")  # only the instance of the test: not the one installed too
    said = []
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: said.append(a[3]))
    doc, ch = s2synth.project(tmp_path)
    app.document = doc
    w_ = mwmod.MainWindow(doc, autoconnect=False)
    w_.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w_, "_error", lambda *a, **k: errors.append(a))
    plugin = _plugin(tmp_path, w_)
    try:
        p = plugin.pane
        assert p is not None and w_.left_tabs.indexOf(p) >= 0 and plugin._qgis_entries in w_.qgis_providers
        plugin.show_pane()
        assert w_.left_tabs.currentWidget() is p
        # search: the area from the DEM, the acquisitions in the table
        p.d_start.setDate(QDate(2023, 1, 1))
        p.d_end.setDate(QDate(2023, 12, 31))
        p.c_catalog.setCurrentText(s2synth.CATALOG)
        p.b_search.click()
        assert w_.runner.busy and not p.b_search.isEnabled() and not p.b_area.isEnabled()  # one step at a time
        _wait(w_, app_)
        assert p.b_search.isEnabled() and p.b_fetch.isEnabled()
        assert not errors and p.scenes.rowCount() == 6 and w_.isWindowModified()
        assert "Area: 4" in p.l_area.text() and "EPSG:32632" in p.l_area.text()
        # what is being typed survives the refreshes of the window
        p.d_start.setDate(QDate(2022, 5, 1))
        w_.refresh_all()
        assert p.d_start.date() == QDate(2022, 5, 1) and p.scenes.rowCount() == 6
        # read and compute: the chart, the dates and periods, an overlay of NDVI on the DEM
        p.b_fetch.click()
        _wait(w_, app_)
        assert not errors and p.chart.has_image() and "5 date serene" in p.l_status.text()
        periods = [p.c_period.itemData(i) for i in range(p.c_period.count())]
        assert periods[:4] == ["2023-spring", "2023-summer", "2023-autumn", "all"] and "20230614_S2B" in periods
        assert plugin.overlay is not None and plugin.overlay.item is not None
        assert w_.ortho_view.tr is not None and "Sentinel-2: area" in w_.ortho_view.layers
        p.s_opacity.setValue(40)
        assert plugin.overlay.item.opacity() == pytest.approx(0.4)
        # natural colour of a date
        p.c_index.setCurrentIndex(p.c_index.findData("RGB"))
        p.c_period.setCurrentIndex(p.c_period.findData("20230803_S2B"))
        p.b_overlay.click()
        assert plugin.overlay.spec["index"] == "RGB" and "Colori naturali" in p.l_legend.text()
        assert not p.b_cropmarks.isEnabled()
        # a crop-mark map of the summer: a product, shown
        p.c_index.setCurrentIndex(p.c_index.findData("NDVI"))
        p.c_period.setCurrentIndex(p.c_period.findData("2023-summer"))
        p.b_cropmarks.click()
        _wait(w_, app_)
        assert not errors and p.products.rowCount() == 1 and plugin.overlay.spec["what"] == "product"
        assert "sotto -2" in p.l_legend.text() and "sotto -2" in w_.statusBar().currentMessage()
        rasters, _v = plugin._qgis_entries()
        assert rasters and rasters[0]["ramp"] == "diverging"
        # the menu actions ask their parameters in the program's dialog
        asked = {}

        def ask(parent, title, fields, intro=""):
            asked[title] = [f.name for f in fields]
            return answers.pop(0)

        monkeypatch.setattr(dialogs.ParamDialog, "ask", ask)
        answers = [{"index": "NDRE", "period": "20230614_S2B", "background": 40.0, "method": "median"}]
        plugin.cropmarks_dialog()
        _wait(w_, app_)
        assert asked["Mappa dei crop mark"] == ["index", "period", "background", "method"]
        assert project.product(ch, "cropmarks_NDRE_20230614_S2B_40m_median") is not None and p.products.rowCount() == 2
        answers = [{"start": "2023-13-01", "end": "2023-12-31", "max_cloud": 30.0, "margin": 300.0,
                    "catalog": s2synth.CATALOG}, None]
        plugin.search_dialog()
        assert said and "AAAA-MM-GG" in said[-1] and not w_.runner.busy
        plugin.search_dialog()  # closed: nothing
        assert not w_.runner.busy and p.scenes.rowCount() == 6
        pid_ndre = "cropmarks_NDRE_20230614_S2B_40m_median"
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
        assert plugin.remove(pid_ndre) and plugin.overlay.spec is None  # it was the one shown
        p.products.selectRow(0)
        p.b_p_overlay.click()  # the summer map again, from the table
        assert plugin.overlay.spec["id"] == "cropmarks_NDVI_2023_summer_50m"
        # the overlay and the area follow the view onto another raster: the empty frame of the area
        p.b_frame.click()
        app_.processEvents()
        assert w_.ortho_view.kind == "frame" and plugin.overlay.item is not None
        assert "Sentinel-2: area" in w_.ortho_view.layers
        # an area drawn with the program's tool
        plugin.draw_area()
        assert w_.ortho_view.mode == "req_polygon"
        b = project.grid_of_chunk(ch)
        from rasterio.warp import transform

        xs, ys = transform(s2synth.UTM, s2synth.CHUNK_CRS, [501050, 501350, 501350, 501050], [5003550, 5003550, 5003750, 5003750])
        w_._request_drawn([[x, y] for x, y in zip(xs, ys)])
        assert project.area_of(ch).source == "polygon" and project.grid_of_chunk(ch) != b
        assert "disegnata" in p.l_area.text() and {r["status"] for r in project.scene_rows(ch)} == {"found"}
        # a product removed after a question (closed: nothing; Remove: gone)
        pid = project.products(ch)[0]["id"]
        asked = []
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: asked.append(a[3]))
        assert not plugin.remove(pid) and asked and project.product(ch, pid) is not None
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)
        assert plugin.remove(pid) and project.product(ch, pid) is None and p.products.rowCount() == 0
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        # another chunk becomes the active one: the overlay of this one is not drawn on it
        p.c_index.setCurrentIndex(p.c_index.findData("NDVI"))
        p.c_period.setCurrentIndex(p.c_period.findData("20230614_S2B"))
        p.b_overlay.click()
        assert plugin.overlay.item is not None and plugin.overlay.spec["chunk"] == ch.key
        other = doc.add_chunk("Altro")
        w_._activate_chunk(other)
        app_.processEvents()
        assert plugin.overlay.spec is None and p.scenes.rowCount() == 0
        w_._activate_chunk(ch)
        app_.processEvents()
        assert p.scenes.rowCount() == 6 and p.d_start.date() == QDate(2023, 1, 1)  # the chunk's own search again
        assert not errors, errors
        plugin.deactivate()
        assert w_.left_tabs.indexOf(p) < 0 and plugin._qgis_entries not in w_.qgis_providers
        assert w_._sentinel2_refresh == [] and "Sentinel-2: area" not in w_.ortho_view.layers
        w_.refresh_all()  # the wrapper stays, with nothing to call; the view drawn again without the area
        plugin.context.release()  # what the program does after deactivate()
        assert "Sentinel-2: area" not in w_.ortho_view.layers
    finally:
        w_.setWindowModified(False)
        w_.close()


def test_the_pane_in_english(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    from PySide6.QtWidgets import QApplication

    from pyarchinit_photogrammetry.gui import i18n

    QApplication.instance() or QApplication([])
    monkeypatch.setattr(i18n, "current", lambda: "en")
    from ppg_sentinel2.pane import Sentinel2Pane
    from ppg_sentinel2.texts import period_name, product_name

    pane = Sentinel2Pane()
    assert pane.b_fetch.text() == "Read and compute" and pane.c_index.itemText(0) == "NDVI (vegetation vigour)"
    assert period_name("2023-summer") == "summer 2023" and period_name("2023-06") == "June 2023"
    assert product_name({"kind": "cropmarks", "index": "NDVI", "period": "summer", "params": {"background_m": 50}}) == \
        "Crop marks NDVI summer (every year), background 50 m"
