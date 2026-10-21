"""The geophysics plugin on synthetic grids: a magnetometry survey with dipole anomalies,
stripes between the traverses and spikes; the formats read and written, the filters, the
georeferencing, the overlay, the exports, the plugin in the program."""
from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

import pyarchinit_photogrammetry as ppg
from ppg_geophysics import export, filters, georef, grids, overlay

CRS = "EPSG:32633"


def synthetic(rows: int = 80, cols: int = 60, cell: float = 0.5, seed: int = 1):
    """nT: a wall (positive linear anomaly), a pit (dipole), noise, a stripe per traverse (the
    rows), a few spikes."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:rows, 0:cols].astype(float)
    v = 0.3 * rng.standard_normal((rows, cols))
    v += 4.0 * np.exp(-((x - 20) ** 2) / 4.0)  # a wall along y at column 20
    v += 6.0 * np.exp(-((x - 45) ** 2 + (y - 30) ** 2) / 12.0) - 3.0 * np.exp(-((x - 45) ** 2 + (y - 36) ** 2) / 12.0)
    stripes = rng.uniform(-2, 2, rows)[:, None]
    v_striped = v + stripes
    spikes = v_striped.copy()
    for r, c in ((10, 10), (50, 50), (70, 5)):
        if r < rows and c < cols:
            spikes[r, c] += 60.0
    g = grids.Grid("Area 1", spikes[None].astype(np.float32), cell, (0.0, rows * cell), "LOCAL", 0.0, "magnetometry", "nT")
    return g, v.astype(np.float32), stripes


def test_grd_round_trip_and_surfer7(tmp_path):
    g, _, _ = synthetic()
    for binary in (True, False):
        p = tmp_path / f"a_{int(binary)}.grd"
        grids.write_grd(g, p, binary=binary)
        back = grids.read_grd(p)
        assert back.shape == g.shape and back.cell == pytest.approx(g.cell)
        assert np.allclose(np.nan_to_num(back.values), np.nan_to_num(g.values), atol=1e-3)
        assert back.origin[0] == pytest.approx(g.origin[0]) and back.origin[1] == pytest.approx(g.origin[1])
    # a Surfer 7 file written by hand: header, GRID and DATA sections, a blank
    nx, ny = 4, 3
    data = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    data[0, 0] = 1.70141e38
    p7 = tmp_path / "s7.grd"
    with open(p7, "wb") as f:
        f.write(b"DSRB" + struct.pack("<ii", 4, 2))
        grid = struct.pack("<ii", ny, nx) + struct.pack("<8d", 10.0, 20.0, 2.0, 2.0, 0.0, 11.0, 0.0, 1.70141e38)
        f.write(b"GRID" + struct.pack("<i", len(grid)) + grid)
        raw = data.astype("<f8").tobytes()
        f.write(b"DATA" + struct.pack("<i", len(raw)) + raw)
    s7 = grids.read_grd(p7)
    assert s7.shape == (3, 4) and s7.cell == 2.0 and np.isnan(s7.values[0, -1, 0]) and s7.values[0, 0, 3] == 11.0
    assert s7.origin == (9.0, 25.0)


def test_xyz_import_and_rasters(tmp_path):
    g, _, _ = synthetic(rows=10, cols=12, cell=0.5)
    lines = ["x,y,nT"]
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            x, y = g.origin[0] + (c + 0.5) * g.cell, g.origin[1] - (r + 0.5) * g.cell
            if (r, c) != (3, 3):
                lines.append(f"{x:.3f},{y:.3f},{g.values[0, r, c]:.4f}")
    p = tmp_path / "survey.csv"
    p.write_text("\n".join(lines) + "\n")
    x = grids.read_xyz(p)
    assert x.shape == g.shape and x.cell == pytest.approx(0.5) and np.isnan(x.values[0, 3, 3])
    both = np.isfinite(x.values) & np.isfinite(g.values)
    assert both.sum() == x.values.size - 1 and np.allclose(x.values[both], g.values[both], atol=1e-3)
    tif = grids.write_geotiff(g, tmp_path / "g.tif", CRS)
    r = grids.read_raster(tif, "magnetometry")
    assert r.crs.endswith("32633") and r.shape == g.shape and r.cell == pytest.approx(0.5)
    assert np.allclose(np.nan_to_num(r.values), np.nan_to_num(g.values))
    # slices: three rasters of a folder become three bands with labels
    folder = tmp_path / "slices"
    folder.mkdir()
    for k, depth in enumerate((0.2, 0.4, 0.6)):
        gk = grids.Grid(f"s{k}", (g.values + k)[:1], g.cell, g.origin, CRS)
        grids.write_geotiff(gk, folder / f"slice_{depth:.1f}m.tif", CRS)
    s = grids.read_slices(folder, labels=["0.2 m", "0.4 m", "0.6 m"])
    assert s.bands == 3 and s.slices == ["0.2 m", "0.4 m", "0.6 m"] and np.allclose(s.values[2] - s.values[0], 2.0, equal_nan=True)
    assert grids.read_any(folder, kind="gpr").bands == 3


def test_filters_remove_spikes_stripes_and_the_trend():
    g, clean, stripes = synthetic()
    v = g.values[0]
    d = filters.despike(v.astype(np.float64), size=3, k=4.0)
    assert abs(v[10, 10] - d[10, 10]) > 50 and abs(d[10, 10] - (clean[10, 10] + stripes[10, 0])) < 3.0  # the stripes of the rows around blur the local median
    assert abs(filters.despike(filters.destripe(v.astype(np.float64)))[10, 10] - clean[10, 10]) < 1.0
    s = filters.destripe(d, axis="rows", stat="median")
    # the stripes gone: the row medians are all alike now
    assert np.nanstd(np.nanmedian(s, axis=1)) < 0.3
    hp = filters.highpass(s, sigma=8.0)
    assert abs(np.nanmean(hp)) < 0.2 and hp[30, 45] > 3.0  # the pit's dipole stays, the mean level goes
    c = filters.clip(v, 1, 99)
    assert c.max() < v.max()
    with_gaps = v.copy()
    with_gaps[20:23, 20:23] = np.nan
    f = filters.fill(with_gaps, distance=3)
    assert np.isfinite(f[21, 21])
    out = filters.apply(g, "resample", cell=1.0)
    assert out.shape == (40, 30) and out.cell == 1.0 and out.filters[-1]["name"] == "resample"
    chain = filters.apply(filters.apply(g, "despike"), "destripe")
    assert [f["name"] for f in chain.filters] == ["despike", "destripe"] and chain.values.shape == g.values.shape
    with pytest.raises(ValueError):
        filters.apply(g, "nothing")


def test_georeferencing_by_pairs_origin_and_markers():
    g, _, _ = synthetic(rows=40, cols=60, cell=0.5)  # 30 x 20 m, origin lower-left
    georef.local_to_corner_frame(g)
    corners = georef.corners_local(g)
    assert corners["upper-right"] == (30.0, 20.0)
    # the pegs measured by GPS: the grid turned by 35 degrees and moved
    a = np.radians(35.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = np.array([438100.0, 4553800.0])
    world = {k: tuple(R @ np.array(v) + t) for k, v in corners.items()}
    fit = georef.similarity([corners[k] for k in ("lower-left", "lower-right", "upper-left")],
                            [world[k] for k in ("lower-left", "lower-right", "upper-left")])
    assert fit["rotation"] == pytest.approx(35.0, abs=1e-6) and fit["scale"] == 1.0 and fit["rmse"] < 1e-6
    placed = georef.place(grids.Grid("x", g.values, g.cell, g.origin, "LOCAL"), fit, CRS)
    c = placed.corners()  # top-left, top-right, bottom-right, bottom-left
    assert np.allclose(c[3], world["lower-left"], atol=1e-6) and np.allclose(c[1], world["upper-right"], atol=1e-6)
    assert placed.crs == CRS and placed.rotation == pytest.approx(35.0)
    # the same by the origin and the azimuth (the x axis 35 degrees north of east = azimuth 55)
    by_origin = georef.place_by_origin(grids.Grid("y", g.values, g.cell, g.origin, "LOCAL"), *world["lower-left"], 55.0, CRS)
    assert np.allclose(by_origin.corners(), placed.corners(), atol=1e-6)
    # a scaled survey (the pegs 2 % longer than nominal) with a free scale
    fit2 = georef.similarity([corners[k] for k in corners], [tuple(1.02 * (R @ np.array(corners[k])) + t) for k in corners],
                             fix_scale=False)
    assert fit2["scale"] == pytest.approx(1.02, abs=1e-6)
    with pytest.raises(ValueError):
        georef.similarity([(0, 0)], [(1, 1)])


def test_overlay_colours_and_resampling():
    g, _, _ = synthetic()
    lo, hi = overlay.limits(g.values[0], "magnetometry")
    assert lo < 0 < hi and hi - lo < 30  # robust: the spikes do not stretch the ramp
    rgba = overlay.render(g.values[0], lo, hi, "grey")
    assert rgba.shape == (80, 60, 4) and rgba[..., 3].min() == 255
    assert rgba[10, 10, 0] == 255  # a spike is white at the top of the ramp
    from rasterio.transform import from_origin

    georef.local_to_corner_frame(g)
    georef.place_by_origin(g, 100.0, 200.0, 90.0, CRS)
    target = from_origin(99.0, 241.0, 0.25, 0.25)  # a finer frame around the grid
    v = overlay.resample_to(g, 0, target, (170, 130))
    assert np.isnan(v[0, 0]) and np.isfinite(v[85, 65])
    assert abs(np.nanmedian(v) - np.nanmedian(g.values[0])) < 0.5


def _project(tmp_path: Path):
    doc = ppg.Document()
    ch = doc.add_chunk("Saggio")
    doc.save(tmp_path / "geo.ppg")
    ch.crs = CRS
    return doc, ch


def test_grids_in_the_project_and_exports(tmp_path):
    from pyarchinit_photogrammetry.io import gpkg

    doc, ch = _project(tmp_path)
    g, _, _ = synthetic()
    georef.local_to_corner_frame(g)
    e = grids.save_grid(ch, g)
    assert e["crs"] == "LOCAL" and (ch.data_dir / e["file"]).exists() and grids.entries(ch)[0]["id"] == e["id"]
    back = grids.load_grid(ch, e["id"])
    assert back.name == "Area 1" and back.kind == "magnetometry" and back.shape == g.shape
    georef.place_by_origin(back, 438100.0, 4553800.0, 95.0, CRS)
    grids.save_grid(ch, back)
    assert len(grids.entries(ch)) == 1 and grids.entries(ch)[0]["crs"] == CRS
    filtered = filters.apply(back, "destripe")
    filtered.id, filtered.name = "", "Area 1 destripe"
    grids.save_grid(ch, filtered)
    assert [x["name"] for x in grids.entries(ch)] == ["Area 1", "Area 1 destripe"]
    out = export.export_grid(ch, "Area 1 destripe", tmp_path / "out.tif")
    assert grids.read_raster(out).crs.endswith("32633")
    assert export.export_grid(ch, "Area 1", tmp_path / "out.grd").exists()
    gp = export.export_gpkg(ch, tmp_path / "geo.gpkg")
    rows = gpkg.read_table(gp, "griglie")
    assert len(rows) == 2 and rows[0][0].area == pytest.approx(30 * 40, rel=1e-6) and rows[1][1]["filtri"] == "destripe"
    rasters, vectors = export.qgis_entries(ch)
    assert len(rasters) == 2 and rasters[0]["ramp"] == "diverging" and rasters[0]["limits"][0] < rasters[0]["limits"][1]
    assert "Area 1" in export.summary(ch)
    doc.save()
    doc2 = ppg.Document.open(doc.path)
    assert len(grids.entries(doc2.chunks[0])) == 2
    grids.remove_grid(ch, "Area 1")
    assert len(grids.entries(ch)) == 1 and not (ch.data_dir / e["file"]).exists()


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("geophysics")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "geophysics"] == [
        "Importa griglia geofisica...", "Griglie geofisiche...", "Sovrapponi la griglia alla vista", "Nascondi la griglia"]
    assert "geophysics" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "geophysics"] == [".gpkg"]


def test_command(tmp_path, capsys):
    from ppg_geophysics import GeophysicsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch = _project(tmp_path)
    app.document = doc
    g, _, _ = synthetic()
    grids.write_grd(g, tmp_path / "a.grd")
    plugin = GeophysicsPlugin()
    plugin.activate(Context("geophysics", tmp_path / "plugin"))
    assert plugin.command("import", str(tmp_path / "a.grd"), "--kind", "magnetometry", "--name", "A") == 0
    assert plugin.command("georef", "A", "--origin", "438100,4553800", "--azimuth", "90") == 0
    assert plugin.command("filter", "A", "destripe", "--copy") == 0
    assert plugin.command("list") == 0
    out = capsys.readouterr().out
    assert '"crs": "EPSG:32633"' in out and "A destripe" in out
    assert plugin.command("export", "A destripe", str(tmp_path / "a_out.tif")) == 0
    assert (tmp_path / "a_out.tif").exists()


def test_the_pane_and_the_overlay_in_the_window(tmp_path):
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geophysics import GeophysicsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path))
    qapp = QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch = _project(tmp_path)
    # a DEM to overlay on, 40 x 30 m around the grid
    from pyarchinit_photogrammetry.engine.dem import write_geotiff

    z = np.zeros((120, 160), np.float32) + 100.0
    dem = write_geotiff(ch.data_dir / "dem.tif", z, (438090.0, 4553845.0), 0.25, CRS)
    ch._set_product("dem", dem)
    g, _, _ = synthetic()
    georef.local_to_corner_frame(g)
    georef.place_by_origin(g, 438100.0, 4553800.0, 90.0, CRS)
    e = grids.save_grid(ch, g)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    try:
        plugin = GeophysicsPlugin()
        plugin.activate(Context("geophysics", tmp_path / "plugin", window=w))
        assert plugin.pane is not None and w.left_tabs.indexOf(plugin.pane) >= 0
        assert plugin.pane.table.rowCount() == 1 and plugin.pane.selected_id() == e["id"]
        w._load_raster("dem")
        qapp.processEvents()
        plugin.show_overlay(e["id"], 0, "grey", 0.6)
        assert plugin.overlay is not None and plugin.overlay.item is not None and plugin.overlay.item.opacity() == pytest.approx(0.6)
        assert plugin.overlay.item in w.ortho_view.overlays
        rasters, vectors = plugin._qgis_entries()
        assert len(rasters) == 1 and any(p == plugin._qgis_entries for p in w.qgis_providers)
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        plugin.deactivate()
        assert w.left_tabs.indexOf(plugin.pane) < 0 if plugin.pane is not None else True
    finally:
        w.setWindowModified(False)
        w.close()
