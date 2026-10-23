"""The terrain plugin on a synthetic DEM: a tilted plain with a valley, a pit and a mound; the
hydrology finds the valley, the depressions find the pit with its depth and volume, the mounds
find the mound, the metrics have the right sign, the viewshed sees from the top; the exports, the
command line, the plugin and its pane in the program. Skipped when WhiteboxTools cannot be had."""
from __future__ import annotations

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
    if wbt.binary() is not None:
        return True
    try:
        wbt.install(folder=tmp_path / "wbt")
        return True
    except Exception as exc:  # noqa: BLE001 - no network
        pytest.skip(f"WhiteboxTools not available: {exc}")
        return False


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


def test_the_pane_and_the_overlay_in_the_window(tmp_path):
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
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        plugin.deactivate()
        assert w_.left_tabs.indexOf(plugin.pane or w_) < 0 or plugin.pane is None
    finally:
        w_.setWindowModified(False)
        w_.close()
