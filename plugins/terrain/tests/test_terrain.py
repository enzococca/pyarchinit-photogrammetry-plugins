"""The terrain plugin on a synthetic DEM: a tilted plain with a valley, a pit and a mound; the
hydrology finds the valley, the depressions find the pit with its depth and volume, the mounds
find the mound, the metrics have the right sign, the viewshed sees from the top, open ground is in
daylight all the time; the progress of a tool while it runs (a fake WhiteboxTools), the products
removed without losing a GeoPackage still used, the names in English, the exports, the command
line, the plugin and its pane in the program. Skipped when WhiteboxTools cannot be had."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import pyarchinit_photogrammetry as ppg
from ppg_terrain import depressions, export, overlay, terrain, vectors, wbt

CRS = "EPSG:32633"
CELL = 0.5
PIT = (30.0, 40.0, 4.0, 2.0)  # x, y (m from the origin), radius, depth
MOUND = (80.0, 60.0, 5.0, 3.0)
VALLEY_X = 50.0


def _have_wbt(tmp_path) -> bool:
    """WhiteboxTools for the tests: the computer's, else downloaded once into a folder the next
    tests find too through ``PPG_WBT`` (the plugin looks only in its own folder, which the tests
    do not configure: on CI every tool said it had no folder to install WhiteboxTools into)."""
    if wbt.binary() is not None:
        return True
    folder = tmp_path / "wbt"
    try:
        wbt.install(folder=folder)
    except Exception as exc:  # noqa: BLE001 - no network
        pytest.skip(f"WhiteboxTools not available: {exc}")
    os.environ["PPG_WBT"] = str(folder)
    return True


def _dem(tmp_path: Path):
    """A chunk with the synthetic DEM as its product (EPSG:32633, 0.5 m cells, 100 x 120 m)."""
    import rasterio
    from rasterio.transform import from_origin

    ncols, nrows = 200, 240
    x = (np.arange(ncols) + 0.5) * CELL
    y = (np.arange(nrows) + 0.5) * CELL
    X, Y = np.meshgrid(x, y)
    rng = np.random.default_rng(3)
    z = 100.0 + 0.02 * (nrows * CELL - Y) + 0.002 * rng.standard_normal((nrows, ncols))  # sloping south
    z -= 1.5 * np.exp(-((X - VALLEY_X) / 6.0) ** 2)  # a valley along y at x = 50
    px, py, pr, pd = PIT
    z -= pd * np.clip(1 - np.hypot(X - px, Y - py) / pr, 0, 1)  # a cone-shaped pit
    mx, my, mr, mh = MOUND
    z += mh * np.clip(1 - np.hypot(X - mx, Y - my) / mr, 0, 1)  # a cone-shaped mound
    z[:3, :] = np.nan  # a strip without data
    doc = ppg.Document()
    ch = doc.add_chunk("Campo")
    doc.save(tmp_path / "campo.ppg")
    ch.crs = CRS
    path = ch.data_dir / "dem.tif"
    tr = from_origin(500000.0, 4500000.0 + nrows * CELL, CELL, CELL)
    with rasterio.open(path, "w", driver="GTiff", height=nrows, width=ncols, count=1, dtype="float32", crs=CRS,
                       transform=tr, nodata=float("nan")) as dst:
        dst.write(z.astype(np.float32), 1)
    ch._set_product("dem", path)
    doc.save()
    return doc, ch, tr


def _world(tr, xm: float, ym: float) -> tuple[float, float]:
    """World coordinates of a point of the synthetic DEM given in metres from its top-left corner
    (the ``X``, ``Y`` grids of :func:`_dem` count from the first row)."""
    return tr.c + xm, tr.f - ym


def test_hydrology_finds_the_valley_and_the_basins(tmp_path):
    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)
    r = terrain.hydrology(ch, threshold_m2=20.0, breach_m=5.0)
    assert r["links"] >= 1 and r["basins"] >= 1
    ids = {e["id"] for e in terrain.entries(ch)}
    assert {"flow_accumulation", "streams", "basins"} <= ids
    from pyarchinit_photogrammetry.io import gpkg

    lines = gpkg.read_table(ch.data_dir / "terrain" / "hydrology.gpkg", "reticolo")
    assert lines
    main = max(lines, key=lambda row: row[1]["cells"])  # the link draining most: the valley's end
    xs = np.array([c[0] for c in main[0].coords]) - tr.c
    assert abs(np.median(xs) - VALLEY_X) < 3.0
    assert main[1]["cells"] > 1000 and sum(ln[1]["length_m"] for ln in lines) > 100.0
    basins = gpkg.read_table(ch.data_dir / "terrain" / "hydrology.gpkg", "bacini")
    assert sum(b[1]["area_m2"] for b in basins) > 0.8 * 100 * 118
    values, _t, _c = terrain.load(ch, "flow_accumulation")
    assert np.nanmax(values) > 3  # log10: thousands of cells drain into the valley's end
    assert terrain.entry(ch, "streams")["integer"] and terrain.entry(ch, "streams")["vector"] == "terrain/hydrology.gpkg"


def test_depressions_and_mounds(tmp_path):
    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)
    r = depressions.find(ch, min_area_m2=2.0, min_depth=0.2, interval=0.0)
    assert r["sinks"] == 1 and r["nodes"] >= 5
    rows = depressions.read_nodes(ch)
    outer = min(rows, key=lambda row: row[1]["rank"])[1]
    px, py, pr, pd = PIT
    assert outer["depth_max"] == pytest.approx(pd, abs=0.3)
    assert outer["area_m2"] == pytest.approx(np.pi * pr * pr, rel=0.35)
    assert outer["volume_m3"] == pytest.approx(np.pi * pr * pr * pd / 3.0, rel=0.4)  # a cone
    assert 0.8 < outer["elongation"] < 1.4
    wx, wy = _world(tr, px, py)
    assert abs(outer["x"] - wx) < 1.0 and abs(outer["y"] - wy) < 1.0
    inner = max(rows, key=lambda row: row[1]["rank"])[1]
    assert inner["parent"] > 0 and inner["area_m2"] < outer["area_m2"] and inner["level"] < outer["level"]
    depth, _t, _c = terrain.load(ch, "depth")
    assert np.nanmax(depth) == pytest.approx(pd, abs=0.3)
    levels, _t, _c = terrain.load(ch, "depression_levels")
    assert np.nanmax(levels) == r["nodes"] or np.nanmax(levels) >= 5
    # the mounds: the same on the inverted DEM
    r2 = depressions.find(ch, min_area_m2=2.0, min_depth=0.5, mounds=True)
    assert r2["sinks"] >= 1 and r2["table"] == "rilievi"
    top = min(depressions.read_nodes(ch, mounds=True), key=lambda row: row[1]["rank"])[1]
    mx, my, mr, mh = MOUND
    assert top["depth_max"] == pytest.approx(mh, abs=0.4)
    assert top["level"] < 0 or top["level"] > 0  # the sign of the level flipped back to heights
    wx, wy = _world(tr, mx, my)
    assert abs(top["x"] - wx) < 1.5 and abs(top["y"] - wy) < 1.5
    out = export.export_gpkg(ch, tmp_path / "terreno.gpkg")
    from pyarchinit_photogrammetry.io import gpkg

    assert len(gpkg.read_table(out, "depressioni")) == r["nodes"] and gpkg.read_table(out, "rilievi")
    assert gpkg.read_table(out, "depressioni")[0][1]["volume_m3"] > 0


def test_metrics_viewshed_and_exports(tmp_path):
    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)
    e = terrain.geomorphometry(ch, "tpi", size_m=10.0)
    values, t, _c = terrain.load(ch, e["id"])
    import rasterio

    px, py, _pr, _pd = PIT
    mx, my, _mr, _mh = MOUND
    r_pit, c_pit = rasterio.transform.rowcol(t, *_world(tr, px, py))
    r_m, c_m = rasterio.transform.rowcol(t, *_world(tr, mx, my))
    assert values[r_pit, c_pit] < -0.3 and values[r_m, c_m] > 0.3
    g = terrain.geomorphometry(ch, "geomorphons", size_m=10.0, cell_m=1.0)  # computed at 1 m, back on the 0.5 m grid
    forms, _t, _c = terrain.load(ch, g["id"])
    assert forms.shape == values.shape and g["params"]["cell_m"] == 1.0
    assert g["integer"] and set(np.unique(forms[np.isfinite(forms)]).astype(int)) <= set(range(1, 11))
    assert int(forms[r_pit, c_pit]) in (10, 9, 7) and int(forms[r_m, c_m]) in (2, 3)  # pit, peak
    day = terrain.geomorphometry(ch, "daylight")
    dl, _t, _c = terrain.load(ch, day["id"])
    assert 0.0 <= np.nanmin(dl) and np.nanmax(dl) <= 1.0
    v = terrain.viewshed(ch, *_world(tr, mx, my), height_m=1.7)
    assert v["params"]["visible_fraction"] > 0.5  # from the top of the mound most of the plain
    seen, _t, _c = terrain.load(ch, v["id"])
    assert set(np.unique(seen[np.isfinite(seen)])) <= {0.0, 1.0}
    out = export.export_raster(ch, e["id"], tmp_path / "tpi.tif")
    with rasterio.open(out) as src:
        assert src.crs.to_string() == CRS and src.dtypes[0] == "float32"
    rasters, vecs = export.qgis_entries(ch)
    assert any(r["ramp"] == "diverging" for r in rasters) and vecs == []
    lo, hi = overlay.limits(values, "diverging")
    assert lo == -hi and hi > 0
    rgba = overlay.render(values, lo, hi, "diverging")
    assert rgba[r_pit, c_pit, 2] > rgba[r_pit, c_pit, 0]  # blue where low, red where high
    assert rgba[r_m, c_m, 0] > rgba[r_m, c_m, 2]
    cl = overlay.render(np.array([[1.0, 2.0, np.nan]]), 1, 2, overlay.CATEGORICAL)
    assert cl[0, 2, 3] == 0 and not np.array_equal(cl[0, 0, :3], cl[0, 1, :3])
    terrain.remove(ch, e["id"])
    assert terrain.entry(ch, e["id"]) is None and not (ch.data_dir / e["file"]).exists()


def test_point_shapefile_and_stream_tracing(tmp_path):
    shp = vectors.write_point_shapefile(tmp_path / "pt.shp", [(10.0, 20.0), (30.0, 40.0)], CRS)
    assert shp.exists() and shp.with_suffix(".dbf").exists() and shp.with_suffix(".shx").stat().st_size == 100 + 2 * 8
    raw = shp.read_bytes()
    assert raw[:4] == b"\x00\x00\x27\x0a" and len(raw) == 100 + 2 * (8 + 20)
    # a straight link of three cells flowing east, then into another link
    from rasterio.transform import from_origin

    links = np.full((3, 5), np.nan)
    links[1, :3] = 1
    links[1, 3:] = 2
    d8 = np.full((3, 5), np.nan)
    d8[1, :] = 2  # east
    facc = np.array([[np.nan] * 5, [1, 2, 3, 4, 5], [np.nan] * 5], float)
    lines = vectors.stream_lines(links, d8, facc, from_origin(0, 3, 1, 1))
    one = next(ln for ln in lines if ln["link"] == 1)
    assert one["coords"][0] == (0.5, 1.5) and one["coords"][-1] == (3.5, 1.5) and one["length_m"] == 3.0
    assert one["cells"] == 3.0


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("terrain")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "terrain"][:4] == [
        "Idrologia: reticolo e bacini...", "Geomorfometria...", "Depressioni e rilievi...", "Visibilità da un punto..."]
    assert "terrain" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "terrain"] == [".gpkg"]


def test_command(tmp_path, capsys):
    _have_wbt(tmp_path)
    from ppg_terrain import TerrainPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch, tr = _dem(tmp_path)
    app.document = doc
    plugin = TerrainPlugin()
    plugin.activate(Context("terrain", tmp_path / "plugin"))
    assert plugin.command("geomorph", "ruggedness") == 0
    assert plugin.command("list") == 0
    out = capsys.readouterr().out
    assert '"ruggedness"' in out
    assert plugin.command("export", "ruggedness", str(tmp_path / "tri.tif")) == 0
    assert (tmp_path / "tri.tif").exists()
    assert plugin.command("remove", "ruggedness") == 0 and terrain.entries(ch) == []
    # from the command line: the project given after the subcommand, opened, worked on and saved
    doc.save()
    app.document = None
    assert plugin.command("geomorph", str(doc.path), "ruggedness") == 0
    assert plugin.command("list", str(doc.path)) == 0 and '"ruggedness"' in capsys.readouterr().out
    import pyarchinit_photogrammetry as ppg

    assert [e["id"] for e in terrain.entries(ppg.Document.open(doc.path).chunk)] == ["ruggedness"]


def test_the_pane_and_the_overlay_in_the_window(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    _have_wbt(tmp_path)
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_terrain import TerrainPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch, tr = _dem(tmp_path)
    app.document = doc
    w_ = mwmod.MainWindow(doc, autoconnect=False)
    w_.runner.show_dialog = False
    plugin = TerrainPlugin()
    try:
        plugin.activate(Context("terrain", tmp_path / "plugin", window=w_))
        assert plugin.pane is not None and w_.left_tabs.indexOf(plugin.pane) >= 0
        r = terrain.hydrology(ch, threshold_m2=20.0)
        depressions.find(ch, min_area_m2=2.0, min_depth=0.2)
        plugin.refresh_pane()
        ids = [plugin.pane.table.item(i, 0).data(mwmod.Qt.ItemDataRole.UserRole) for i in range(plugin.pane.table.rowCount())]
        assert {"flow_accumulation", "streams", "basins", "depth", "depression_levels"} <= set(ids)
        w_._show_raster("dem")
        plugin.show_overlay("depth", "", 0.6)
        assert plugin.overlay is not None and plugin.overlay.item is not None and plugin.overlay.item.opacity() == pytest.approx(0.6)
        assert "Profondità" in plugin.pane.l_limits.text()
        plugin.layer("streams")
        assert "Terreno: reticolo" in w_.line_layers and len(w_.line_layers["Terreno: reticolo"]) == r["links"]
        assert "Terreno: bacini" in w_.ortho_view.layers
        plugin.layer("depth")
        assert "Terreno: depressioni" in w_.ortho_view.layers
        rasters, vecs = plugin._qgis_entries()
        assert rasters and {v["table"] for v in vecs} == {"reticolo", "bacini", "depressioni"}
        # Remove asks first (a non-native box); a GeoPackage another product still uses stays
        from pyarchinit_photogrammetry.gui import dialogs

        asked = []
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: asked.append(a[3]))  # the box closed: nothing
        assert not plugin.remove("streams") and terrain.entry(ch, "streams") is not None
        assert "hydrology.gpkg" in asked[0] and "«Bacini»" in asked[0]
        monkeypatch.setattr(dialogs, "choice", lambda *a, **k: 0)  # Rimuovi
        assert plugin.remove("streams") and terrain.entry(ch, "streams") is None
        assert (ch.data_dir / "terrain" / "hydrology.gpkg").exists() and "Terreno: bacini" in w_.ortho_view.layers
        assert plugin.remove("basins") and not (ch.data_dir / "terrain" / "hydrology.gpkg").exists()
        assert "Terreno: bacini" not in w_.ortho_view.layers and "Terreno: reticolo" not in w_.line_layers
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        plugin.deactivate()
        assert w_.left_tabs.indexOf(plugin.pane or w_) < 0 or plugin.pane is None
    finally:
        w_.setWindowModified(False)
        w_.close()


# ------------------------------------------------------------------ removing, cells, names
def test_the_pane_follows_the_project_after_the_plugin_is_turned_on_again(tmp_path):
    """A plugin turned off and on again is a new instance: the window refreshes its pane, not the old
    one's - with the program's notice (1.3.3) and with the wrapper of older programs."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_terrain import TerrainPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    class Counting(TerrainPlugin):
        def refresh_pane(self):
            self.calls = getattr(self, "calls", 0) + 1
            super().refresh_pane()

    class OldContext(Context):  # a program before 1.3.3
        @property
        def on_refresh(self):
            raise AttributeError("on_refresh")

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, _ch, _tr = _dem(tmp_path)
    app.document = doc
    for context_class in (Context, OldContext):
        w_ = mwmod.MainWindow(doc, autoconnect=False)
        w_.runner.show_dialog = False
        try:
            first = Counting()
            first.activate(context_class("terrain", tmp_path / "plugin", window=w_))
            first.deactivate()
            first.context.release()  # what the program does after deactivate()
            second = Counting()
            second.activate(context_class("terrain", tmp_path / "plugin", window=w_))
            before = (first.calls, second.calls)
            w_.refresh_all()
            assert (first.calls, second.calls) == (before[0], before[1] + 1), context_class.__name__
            second.deactivate()
            second.context.release()
        finally:
            w_.setWindowModified(False)
            w_.close()


def test_remove_keeps_a_geopackage_another_product_uses(tmp_path):
    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)
    terrain.hydrology(ch, threshold_m2=20.0)
    gpkg = ch.data_dir / "terrain" / "hydrology.gpkg"
    assert [x["id"] for x in terrain.sharing(ch, "streams")] == ["basins"]
    gone = terrain.remove(ch, "streams")
    assert [Path(g).name for g in gone] == ["streams.tif"] and gpkg.exists()  # the basins still use it
    assert {table for _p, table, *_ in export.vector_files(ch)} == {"reticolo", "bacini"}
    gone = terrain.remove(ch, "basins")
    assert {Path(g).name for g in gone} == {"basins.tif", "hydrology.gpkg"} and not gpkg.exists()
    assert terrain.entry(ch, "flow_accumulation") is not None and (ch.data_dir / "terrain" / "flow_accumulation.tif").exists()
    depressions.find(ch, min_area_m2=2.0, min_depth=0.2)
    terrain.remove(ch, "depression_levels")
    assert (ch.data_dir / "terrain" / "depressioni.gpkg").exists() and depressions.read_nodes(ch)
    terrain.remove(ch, "depth")
    assert not (ch.data_dir / "terrain" / "depressioni.gpkg").exists()
    assert terrain.remove(ch, "nothing") == []


def test_the_cell_of_the_computation(tmp_path):
    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)  # cells of 0.5 m
    a = terrain.geomorphometry(ch, "ruggedness", cell_m=0.75)  # larger than the DEM's cell, though not 1.5 times
    assert a["id"] == "ruggedness_c0_75" and a["params"]["coarse"] and a["params"]["cell_m"] == 0.75
    b = terrain.geomorphometry(ch, "ruggedness", cell_m=1.0)
    assert b["id"] == "ruggedness_c1" and b["name"] == "Indice di rugosità (TRI), cella 1 m"
    assert terrain.entry(ch, "ruggedness_c0_75") is not None  # the two cells side by side
    values, _t, _c = terrain.load(ch, "ruggedness_c1")
    assert values.shape == (240, 200)  # back on the grid of the DEM
    c = terrain.geomorphometry(ch, "ruggedness", cell_m=0.25)  # finer than the DEM: its own cell, and said so
    assert c["id"] == "ruggedness" and not c["params"]["coarse"] and c["params"]["cell_m"] == 0.5
    assert c["params"]["cell_asked_m"] == 0.25 and "0.25 m" in c["note"] and "0.5 m" in c["note"]
    g = terrain.geomorphometry(ch, "geomorphons", size_m=10.0)
    g1 = terrain.geomorphometry(ch, "geomorphons", size_m=10.0, cell_m=1.0)
    assert (g["id"], g1["id"]) == ("geomorphons_10", "geomorphons_10_c1") and terrain.entry(ch, "geomorphons_10")
    assert g1["name"] == "Geomorphons (forme del terreno) 10 m, cella 1 m"


def test_names_in_the_language_of_the_interface(tmp_path):
    from pyarchinit_photogrammetry.gui import i18n

    from ppg_terrain.texts import layer_name

    _have_wbt(tmp_path)
    doc, ch, tr = _dem(tmp_path)
    terrain.hydrology(ch, threshold_m2=20.0)
    e = terrain.geomorphometry(ch, "tpi", size_m=10.0, cell_m=1.0)
    v = terrain.viewshed(ch, *_world(tr, *MOUND[:2]), height_m=1.7)
    before = i18n.current()
    try:
        i18n.set_current("en")
        assert terrain.display_name(e) == "Relative topographic position (TPI) 10 m, cell 1 m"
        assert terrain.display_name(terrain.entry(ch, "streams")) == "Stream network (links)"
        assert terrain.display_name(v).startswith("Viewshed from (") and v["name"].startswith("Visibilità da (")
        rasters, vecs = export.qgis_entries(ch)
        names = {r["name"] for r in rasters} | {x["name"] for x in vecs}
        assert {"Terrain: Flow accumulation (log10 cells)", "Terrain: Stream network", "Terrain: Basins"} <= names
        assert layer_name("reticolo") == "Terrain: streams"
        assert export.limits_text(e, -1, 1).startswith("Relative topographic position")
    finally:
        i18n.set_current(before)
    # stored in Italian, as data: the project reads the same in both languages
    assert e["name"] == "Posizione topografica relativa (TPI) 10 m, cella 1 m" and terrain.display_name(e) == e["name"]
    assert layer_name("reticolo") == "Terreno: reticolo"


# ------------------------------------------------------------------ time in daylight
def test_daylight_is_one_on_open_ground(tmp_path):
    alt, az = terrain.solar_track(40.6)
    assert alt.max() == pytest.approx(90.0 - 40.6 + 23.44, abs=0.2)  # noon of the summer solstice
    assert 0.49 < len(alt) / (365 * 24 * 6) < 0.52  # the sun is up about half the year
    assert 55.0 < az.min() < 62.0 and 298.0 < az.max() < 305.0  # never in the north, at 40 degrees
    _have_wbt(tmp_path)
    import rasterio
    from rasterio.transform import from_origin

    flat = tmp_path / "flat.tif"
    with rasterio.open(flat, "w", driver="GTiff", height=60, width=60, count=1, dtype="float32", crs=CRS,
                       transform=from_origin(500000.0, 4500000.0, CELL, CELL), nodata=-32767.0) as dst:
        dst.write(np.full((60, 60), 100.0, np.float32), 1)
    lit = terrain.read_result(terrain.daylight(flat, tmp_path / "day.tif", 40.6))
    assert np.nanmin(lit) == 1.0  # WhiteboxTools' own TimeInDaylight 2.4 gave 0.79 here
    doc, ch, tr = _dem(tmp_path)
    e = terrain.geomorphometry(ch, "daylight")
    assert e["params"]["lat"] == pytest.approx(40.65, abs=0.01) and e["params"]["az_fraction"] == 15.0
    day, t, _c = terrain.load(ch, e["id"])

    def at(xm, ym):
        r, c = rasterio.transform.rowcol(t, *_world(tr, xm, ym))
        return day[r, c]

    mx, my, mr, _mh = MOUND
    assert at(10.0, 100.0) > 0.98 and at(90.0, 20.0) > 0.98  # the open plain
    assert at(PIT[0], PIT[1]) < 0.7  # the bottom of the pit sees the sun only when it is high
    assert at(mx, my - 0.7 * mr) < at(mx, my + 0.7 * mr) - 0.1  # the north side of the mound is in its shadow


# ------------------------------------------------------------------ WhiteboxTools while it runs
FAKE_WBT = """#!{python}
import pathlib, sys, time
args = dict(a[2:].split("=", 1) for a in sys.argv[1:] if a.startswith("--") and "=" in a)
tool = args.get("run")
print("* Welcome to " + str(tool) + " *", flush=True)
if tool == "Slow":
    for phase in ("First phase", "Second phase"):
        for pct in range(0, 101, 20):
            print(f"{{phase}}: {{pct}}%", flush=True)
            time.sleep(0.06)
elif tool == "Panic" and "--max_procs=1" not in sys.argv:
    print("thread 'main' panicked at src/tools/fill.rs:657:27:", flush=True)
    print("Error unwrapping 'output'", flush=True)
    sys.exit(0)
elif tool == "AlwaysPanic":
    pathlib.Path(args["output"]).write_text("x")
    print("thread 'main' panicked at src/tools/fill.rs:657:27:", flush=True)
    print("Error unwrapping 'output'", flush=True)
    sys.exit(0)
elif tool == "Sleepy":
    print("Progress: 1%", flush=True)
    time.sleep(30)
pathlib.Path(args["output"]).write_text("x")
print("Output file written", flush=True)
"""


def _fake_wbt(tmp_path: Path) -> Path:
    exe = tmp_path / "fake_wbt" / "whitebox_tools"
    exe.parent.mkdir()
    exe.write_text(FAKE_WBT.format(python=sys.executable))
    exe.chmod(0o755)
    return exe


def test_progress_comes_while_the_tool_runs(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Cancelled

    monkeypatch.setenv("PPG_WBT", str(_fake_wbt(tmp_path)))
    seen = []
    out = tmp_path / "out.tif"
    lines = wbt.run("Slow", progress=lambda f, m: seen.append((time.monotonic(), f, m)), phases=2, output=str(out))
    end = time.monotonic()
    assert out.exists() and "Second phase: 100%" in lines
    reports = [s for s in seen if "phase" in s[2]]
    assert reports and reports[0][0] < end - 0.3  # read while the tool was still running, not after
    fractions = [f for _t, f, _m in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0  # one bar that only grows
    assert any(f == pytest.approx(0.5) and m == "Slow: First phase 100%" for _t, f, m in seen)  # a phase of two
    # a tool that panics (exit code 0) runs again on one thread; an output left over never passes
    out2 = tmp_path / "out2.tif"
    wbt.run("Panic", output=str(out2))
    assert out2.exists()
    with pytest.raises(wbt.ToolError, match="Error unwrapping"):
        wbt.run("AlwaysPanic", output=str(tmp_path / "out3.tif"))
    # cancelled while it runs: the tool is stopped, not waited for

    def cancel(_f, m):
        if "Progress 1%" in m:
            raise Cancelled("stop")

    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        wbt.run("Sleepy", progress=cancel, output=str(tmp_path / "out4.tif"))
    assert time.monotonic() - t0 < 10.0
    phases = wbt.Phases(3)
    assert [round(phases.update(lbl, p), 3) for lbl, p in (("a", 50), ("b", 0), ("b", 100), ("c", 100), ("d", 10))] == [
        0.167, 0.333, 0.667, 1.0, 1.0]
    assert wbt.parse_progress("Progress (Az=60): 35%") == ("Progress (Az=60)", 35)
    assert wbt.parse_progress("Flats increment value: 0.000001") is None
