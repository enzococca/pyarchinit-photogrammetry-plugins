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
    # acquisitions out of the cache: one, then all (the crop marks of the summer stay, marked)
    capsys.readouterr()
    assert plugin.command("remove-scene", path, "20230709_S2B") == 0
    out = capsys.readouterr().out
    assert '"removed": [' in out and '"20230709_S2B"' in out and '"products_marked": [' in out
    assert plugin.command("remove-scene", path, "20230709_S2B") == 1 and "not in the cache" in capsys.readouterr().out
    assert plugin.command("remove-scene", path) == 1 and "--all" in capsys.readouterr().out
    assert plugin.command("list", path) == 0 and '"20230709_S2B"' not in capsys.readouterr().out.split('"cache"')[1].split("]")[0]
    assert plugin.command("remove-scene", path, "--all") == 0
    capsys.readouterr()
    back = ppg.Document.open(doc.path).chunk  # every command saved the project
    assert [p["id"] for p in project.products(back)] == ["cropmarks_NDVI_2023_summer_60m"]
    assert project.cached_ids(back) == [] and project.products(back)[0]["missing"] == [
        "20230614_S2B", "20230709_S2B", "20230803_S2B"]
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
        assert p.l_legend.text().startswith("z da -3 (rosso) a 3 (verde)")  # as in QGIS
        rasters, _v = plugin._qgis_entries()
        assert rasters and rasters[0]["ramp"] == "index" and rasters[0]["limits"] == [-3.0, 3.0]
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
        plugin.context.release()  # what the program does after deactivate()
        assert w_.left_tabs.indexOf(p) < 0 and plugin._qgis_entries not in w_.qgis_providers
        assert "Sentinel-2: area" not in w_.ortho_view.layers
        w_.refresh_all()  # nothing of the plugin told any more; the view drawn again without the area
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


class TreeContext:
    """A context of the program with the two methods of the project's tree of program 1.3.4
    (``add_tree_items(provider)``, ``refresh_tree()``) on top of the real ones."""

    def __new__(cls, *args, **kwargs):
        from pyarchinit_photogrammetry.plugins import Context

        class _Context(Context):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.providers, self.refreshed = [], 0

            def add_tree_items(self, provider):
                self.providers.append(provider)

            def refresh_tree(self):
                self.refreshed += 1

        return _Context(*args, **kwargs)


def _labels(items) -> list[str]:
    return [i["label"] for i in items]


def _find(items, start: str) -> dict:
    return next(i for i in items if i["label"].startswith(start))


def _action(item, text: str):
    return next(f for label, f in item["actions"] if label.startswith(text))


def test_the_products_in_the_projects_tree(tmp_path, monkeypatch):
    """The plugin gives the project's tree (program 1.3.4) a *Sentinel-2* node per chunk: acquisitions
    read, indices by date, series, composites, crop marks; the double clicks and the actions call what
    they say; every change of products or cache fills the tree again. On a program without the tree
    (1.3.3) nothing is registered and nothing breaks."""
    import pyarchinit_photogrammetry as ppg
    from ppg_sentinel2 import Sentinel2Plugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    from ppg_sentinel2 import area as area_mod

    items = s2synth.world(tmp_path / "cogs")
    doc, ch = s2synth.project(tmp_path)
    project.search(ch, area_mod.area_of_chunk(ch), "2023-01-01", "2023-12-31", 30, s2synth.CATALOG,
                   opener=s2synth.FakeCatalog(items))
    project.fetch(ch)
    marks = project.cropmarks(ch, "NDVI", "2023-summer", 50.0)
    april = project.composite(ch, "NDVI", "2023-04")
    other = doc.add_chunk("Senza Sentinel-2")
    app.document = doc
    # program 1.3.3: no tree, no error
    plain = Sentinel2Plugin()
    plain.activate(Context("sentinel2", tmp_path / "plain"))
    plain._refresh_tree()
    ctx = TreeContext("sentinel2", tmp_path / "plugin")
    plugin = Sentinel2Plugin()
    called = []
    for name in ("show_acquisition", "show_overlay", "show_product", "open_chart", "export_product",
                 "export_date_dialog", "export_acquisition_dialog", "export_series_dialog", "open_qgis", "show_pane",
                 "show_frame", "hide_overlay", "export_dialog"):
        monkeypatch.setattr(plugin, name, lambda *a, _n=name, **k: called.append((_n, a)))
    plugin.activate(ctx)
    assert ctx.providers == [plugin._tree_items]
    provider = ctx.providers[0]
    assert provider(other) == []
    (root,) = provider(ch)
    assert root["label"] == "Sentinel-2" and root["icon"] == "globe" and root["expanded"]
    assert "Area: 4" in root["tooltip"] and "Ricerca: 2023-01-01 - 2023-12-31" in root["tooltip"]
    assert _labels(root["children"]) == ["Acquisizioni lette (5)", "Indici per data (5 date)", "Serie temporale",
                                         "Compositi (1)", "Crop mark (1)"]
    acqs = root["children"][0]
    assert _labels(acqs["children"]) == ["2023-09-12 S2B, sereno 100 %", "2023-08-03 S2B, sereno 100 %",
                                         "2023-07-09 S2B, sereno 76 %", "2023-06-14 S2B, sereno 100 %",
                                         "2023-04-10 S2B, sereno 100 %"]
    assert {i["icon"] for i in acqs["children"]} == {"ortho"} and "32TMR" in acqs["children"][0]["tooltip"]
    indices = root["children"][1]
    assert _labels(indices["children"]) == ["NDVI (vigore della vegetazione)", "NDRE (clorofilla, bordo del rosso)",
                                            "NDMI (acqua nella vegetazione)"]
    ndvi = indices["children"][0]
    assert len(ndvi["children"]) == 5 and ndvi["children"][0]["label"].startswith("2023-09-12")
    crop = _find(root["children"], "Crop mark")["children"][0]
    assert crop["label"] == "Crop mark NDVI estate 2023, sfondo 50 m" and "% dell'area sotto -2" in crop["tooltip"]
    assert [a for a, _f in crop["actions"]] == ["Mostra", "Nascondi", "Esporta...", "Apri in QGIS", "Rimuovi"]
    assert [a for a, _f in acqs["children"][0]["actions"]] == ["Mostra (colori naturali)", "Nascondi", "Esporta...",
                                                              "Rimuovi dalla cache"]
    # double clicks and actions call what they say, on that chunk
    acqs["children"][1]["activate"]()
    ndvi["children"][1]["activate"]()
    crop["activate"]()
    _find(root["children"], "Serie")["activate"]()
    _action(crop, "Esporta")()
    _action(ndvi["children"][0], "Esporta")()
    _action(acqs["children"][0], "Esporta")()
    _action(crop, "Apri in QGIS")()
    root["activate"]()
    _action(root, "Mostra l'area intera")()
    _action(crop, "Nascondi")()
    assert called[:5] == [("show_acquisition", ("20230803_S2B",)),
                          ("show_overlay", ({"index": "NDVI", "period": "20230803_S2B", "palette": "rdylgn",
                                             "opacity": 0.75},)),
                          ("show_product", (marks["id"],)), ("open_chart", (ch,)), ("export_product", (marks["id"], ch))]
    assert called[5:] == [("export_date_dialog", ("NDVI", "20230912_S2B", ch)),
                          ("export_acquisition_dialog", ("20230912_S2B", ch)), ("open_qgis", ()), ("show_pane", ()),
                          ("show_frame", ()), ("hide_overlay", ())]
    # removing: from the cache (no window: no question, products kept and marked), a product; the tree
    # is filled again each time
    seen = ctx.refreshed
    _action(acqs["children"][2], "Rimuovi dalla cache")()  # 9 July
    assert ctx.refreshed > seen and "20230709_S2B" not in project.cached_ids(ch)
    (root,) = provider(ch)
    assert _labels(root["children"])[:2] == ["Acquisizioni lette (4)", "Indici per data (4 date)"]
    crop = _find(root["children"], "Crop mark")["children"][0]
    assert crop["label"].endswith("(date non più in cache)") and "20230709_S2B" in crop["tooltip"]
    seen = ctx.refreshed
    _action(_find(root["children"], "Compositi")["children"][0], "Rimuovi")()
    assert ctx.refreshed > seen and project.product(ch, april["id"]) is None
    _action(root, "Svuota la cache")()
    (root,) = provider(ch)
    assert _labels(root["children"]) == ["Acquisizioni lette (0)", "Crop mark (1)"]
    assert root["children"][0]["actions"] == []
    assert ppg is not None


def test_cache_and_tree_in_the_window(tmp_path, monkeypatch):
    """In the window: the acquisitions selected out of the cache after the question of the program
    (the products kept, marked), the context menu of the table, the whole cache emptied with its
    products; the overlay that showed a removed acquisition goes; the tree's double clicks show over
    the view and the pane follows them."""
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_sentinel2 import Sentinel2Plugin
    from ppg_sentinel2 import area as area_mod
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    app_ = QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    items = s2synth.world(tmp_path / "cogs")
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    said = []
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: said.append(a[3]))
    doc, ch = s2synth.project(tmp_path)
    project.search(ch, area_mod.area_of_chunk(ch), "2023-01-01", "2023-12-31", 30, s2synth.CATALOG,
                   opener=s2synth.FakeCatalog(items))
    project.fetch(ch)
    marks = project.cropmarks(ch, "NDVI", "2023-summer", 50.0)
    app.document = doc
    w_ = mwmod.MainWindow(doc, autoconnect=False)
    w_.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w_, "_error", lambda *a, **k: errors.append(a))
    ctx = TreeContext("sentinel2", tmp_path / "plugin", window=w_)
    plugin = Sentinel2Plugin()
    try:
        plugin.activate(ctx)
        p = plugin.pane
        assert p.scenes.rowCount() == 6 and p.b_clear_cache.isEnabled() and not p.b_uncache.isEnabled()
        # Ricalcola takes the least clear part of the pane, and saves it: 9 July (76 %) out at 90 %
        p.s_clear.setValue(90)
        p.b_compute.click()
        _wait(w_, app_)
        assert project.settings(ch)["min_clear"] == 0.9 and len(project.dates(ch)) == 4
        p.s_clear.setValue(50)
        p.b_compute.click()
        _wait(w_, app_)
        assert project.settings(ch)["min_clear"] == 0.5 and len(project.dates(ch)) == 5 and not errors
        # the tree: a date of NDVI shown over the view, the pane follows
        (root,) = ctx.providers[0](ch)
        leaf = root["children"][1]["children"][0]["children"][2]  # NDVI, 9 July
        leaf["activate"]()
        assert plugin.overlay.item is not None and plugin.overlay.spec["period"] == "20230709_S2B"
        assert p.c_period.currentData() == "20230709_S2B" and p.c_index.currentData() == "NDVI"
        # 9 July selected and taken out: the question names the products made from it, kept here
        row = next(r for r in range(p.scenes.rowCount()) if p.scenes.item(r, 0).data(mwmod.Qt.ItemDataRole.UserRole)
                   == "20230709_S2B")
        p.scenes.selectRow(row)
        assert p.selected_cached() == ["20230709_S2B"] and p.b_uncache.isEnabled()
        asked = []

        def choice(parent, kind, title, text, options, *a, **k):
            asked.append((title, text, options))
            return answer[0]

        answer = [1]
        monkeypatch.setattr(dialogs, "choice", choice)
        p.b_uncache.click()
        _wait(w_, app_)
        assert not errors, errors
        title, text, options = asked[-1]
        assert title == "Rimuovi dalla cache" and "Crop mark NDVI estate 2023" in text
        assert options == ["Rimuovi anche i prodotti", "Tieni i prodotti"]
        assert plugin.overlay.spec is None  # it showed the date removed
        assert "20230709_S2B" not in project.cached_ids(ch) and project.product(ch, marks["id"])["missing"]
        assert p.products.item(0, 0).text().endswith("(date non più in cache)")
        status = {p.scenes.item(r, 0).data(mwmod.Qt.ItemDataRole.UserRole): p.scenes.item(r, 5).text()
                  for r in range(p.scenes.rowCount())}
        assert status["20230709_S2B"] == "da leggere" and ctx.refreshed > 0
        # closed without answering: nothing happens
        answer[0] = None
        p.scenes.selectRow(0)
        assert not plugin.remove_scenes(p.selected_cached()) and len(project.cached_ids(ch)) == 4
        # the context menu: natural colour of the row, out of the cache, the whole cache
        menu = p.scenes_menu(0)
        texts = [a.text() for a in menu.actions() if not a.isSeparator()]
        assert texts == ["Mostra (colori naturali)", "Rimuovi dalla cache (1)", "Svuota la cache..."]
        menu.actions()[0].trigger()
        assert plugin.overlay.spec["what"] == "rgb" and plugin.overlay.item is not None
        answer[0] = 0  # remove the products too
        [a for a in menu.actions() if a.text() == "Svuota la cache..."][0].trigger()
        _wait(w_, app_)
        assert not errors, errors
        assert asked[-1][0] == "Svuota la cache" and "MB" in asked[-1][1]
        assert project.cached_ids(ch) == [] and project.products(ch) == [] and plugin.overlay.spec is None
        assert not p.b_clear_cache.isEnabled() and p.products.rowCount() == 0
        plugin.deactivate()
        ctx.release()
    finally:
        w_.setWindowModified(False)
        w_.close()


def test_command_line_boxes(tmp_path, monkeypatch, capsys):
    """``--bbox`` with a negative longitude, in every form: the value glued to its option by the
    plugin, ``--bbox=...``, four numbers; ``--polygon`` beginning with a minus too."""
    from ppg_sentinel2 import Sentinel2Plugin
    from pyarchinit_photogrammetry.app import app

    glue = Sentinel2Plugin.glue_values
    assert glue(["search", "--bbox", "-1.83,51.17,-1.82,51.18", "--margin", "0"]) == [
        "search", "--bbox=-1.83,51.17,-1.82,51.18", "--margin", "0"]
    assert glue(["--bbox", "-1.83", "51.17", "-1.82", "51.18"]) == ["--bbox", "-1.83", "51.17", "-1.82", "51.18"]
    assert glue(["--polygon", "-10,5;-12,7;-11,9"]) == ["--polygon=-10,5;-12,7;-11,9"]
    assert glue(["--bbox", "--margin", "0"]) == ["--bbox", "--margin", "0"]
    assert Sentinel2Plugin.bbox_values(["-1.83", "51.17", "-1.82", "51.18"]) == (-1.83, 51.17, -1.82, 51.18)
    with pytest.raises(ValueError):
        Sentinel2Plugin.bbox_values(["-1.83,51.17"])
    items = s2synth.world(tmp_path / "cogs")
    monkeypatch.setattr(stac, "_urlopen", s2synth.FakeCatalog(items))
    doc, ch = s2synth.project(tmp_path)
    app.document = None
    plugin = _plugin(tmp_path)
    path = str(doc.path)
    import pyarchinit_photogrammetry as ppg

    for form in (["--bbox", "-1.83,51.17,-1.82,51.18"], ["--bbox=-1.83,51.17,-1.82,51.18"],
                 ["--bbox", "-1.83", "51.17", "-1.82", "51.18"]):
        assert plugin.command("search", path, *form, "--margin", "0", "--catalog", s2synth.CATALOG) == 0, form
        a = project.area_of(ppg.Document.open(doc.path).chunk)
        assert a.source == "bbox" and a.margin_m == 0
        assert a.lonlat_bbox(margin=False) == pytest.approx((-1.83, 51.17, -1.82, 51.18), abs=1e-6)
    assert plugin.command("search", path, "--bbox", "-1.83,51.17", "--catalog", s2synth.CATALOG) == 1
    assert "four numbers" in capsys.readouterr().out
    # a polygon in the chunk's coordinates (EPSG:3035 here) beginning with a minus: read, not taken for an option
    assert plugin.command("search", path, "--polygon", "-10,5;-12,7;-11,9", "--catalog", s2synth.CATALOG) == 0
    a = project.area_of(ppg.Document.open(doc.path).chunk)
    assert a.source == "polygon" and a.ring == [[-10.0, 5.0], [-12.0, 7.0], [-11.0, 9.0]]
    capsys.readouterr()
