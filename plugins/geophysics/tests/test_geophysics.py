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
    t = np.array([500100.0, 4500800.0])
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
    georef.place_by_origin(back, 500100.0, 4500800.0, 95.0, CRS)
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
        "Importa griglia geofisica...", "Griglie geofisiche...", "Sovrapponi la griglia alla vista", "Nascondi la griglia",
        "Disegna un'anomalia"]
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
    assert plugin.command("georef", "A", "--origin", "500100,4500800", "--azimuth", "90") == 0
    assert plugin.command("filter", "A", "destripe", "--copy") == 0
    assert plugin.command("list") == 0
    out = capsys.readouterr().out
    assert '"crs": "EPSG:32633"' in out and "A destripe" in out
    assert plugin.command("export", "A destripe", str(tmp_path / "a_out.tif")) == 0
    assert (tmp_path / "a_out.tif").exists()
    # from the command line: the project given after the subcommand, opened and saved
    doc.save()
    app.document = None
    assert plugin.command("import", str(doc.path), str(tmp_path / "a.grd"), "--name", "B") == 0
    assert plugin.command("list", "--project", str(doc.path)) == 0 and "B" in capsys.readouterr().out


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
    dem = write_geotiff(ch.data_dir / "dem.tif", z, (500090.0, 4500845.0), 0.25, CRS)
    ch._set_product("dem", dem)
    g, _, _ = synthetic()
    georef.local_to_corner_frame(g)
    georef.place_by_origin(g, 500100.0, 4500800.0, 90.0, CRS)
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
        # not one of the view's drawings: it stays when the program draws its layers again
        assert plugin.overlay.shown() and plugin.overlay.item not in w.ortho_view.overlays
        w._redraw_overlays()
        assert plugin.overlay.shown()
        rasters, vectors = plugin._qgis_entries()
        assert len(rasters) == 1 and any(p == plugin._qgis_entries for p in w.qgis_providers)
        plugin.hide_overlay()
        assert plugin.overlay.item is None
        plugin.deactivate()
        assert w.left_tabs.indexOf(plugin.pane) < 0 if plugin.pane is not None else True
    finally:
        w.setWindowModified(False)
        w.close()


# ------------------------------------------------------------------ XYZ tables
def _traverses(path: Path, sep: str = " ", comma: bool = False, skip=(), extra: list[str] | None = None) -> Path:
    """A magnetometry table as a survey writes it: readings every 0.125 m along x on traverses 0.5 m
    apart, centred in their cells (x from 0.0625, y from 0.25: a survey of 30 x 6 m); the value a plane
    0.5 x + 2 y, which linear interpolation between the traverses gives back exactly."""
    lines = [sep.join(("x", "y", "nT"))]
    for j in range(12):
        for i in range(240):
            if (j, i) in skip:
                continue
            x, y = 0.0625 + 0.125 * i, 0.25 + 0.5 * j
            row = sep.join(f"{v:.4f}" for v in (x, y, 0.5 * x + 2.0 * y))
            lines.append(row.replace(".", ",") if comma else row)
    path.write_text("\n".join(lines + (extra or [])) + "\n")
    return path


def _centres(g) -> tuple[np.ndarray, np.ndarray]:
    t = g.transform()
    rows, cols = np.mgrid[0:g.shape[0], 0:g.shape[1]] + 0.5
    return t.a * cols + t.b * rows + t.c, t.d * cols + t.e * rows + t.f


def test_xyz_traverses_are_interpolated_between_them(tmp_path):
    g = grids.read_xyz(_traverses(tmp_path / "survey.xyz"))
    assert g.cell == pytest.approx(0.125) and g.shape == (45, 240)  # fine cell; from the first to the last traverse
    assert g.notes["interpolated"] == "rows" and g.notes["traverse_spacing"] == pytest.approx(0.5)
    x, y = _centres(g)
    assert x[0, 0] == pytest.approx(0.0625) and y[0, 0] == pytest.approx(5.75) and y[-1, 0] == pytest.approx(0.25)
    assert np.isfinite(g.values).all() and np.allclose(g.values[0], 0.5 * x + 2.0 * y, atol=2e-3)
    # a reading missing: the lines around it stay empty, nothing is made up across it
    holed = grids.read_xyz(_traverses(tmp_path / "holed.xyz", skip={(5, 10)}))
    assert np.isnan(holed.values[0, :, 10]).sum() == 7 and np.isfinite(holed.values[0, :, 11]).all()
    # the cell of the traverses: the readings of a cell averaged, the cells on the survey (60 x 12 of 0.5 m)
    c = grids.read_xyz(tmp_path / "survey.xyz", cell=0.5)
    assert c.shape == (12, 60) and c.origin == pytest.approx((0.0, 6.0)) and c.notes.get("averaged")
    x, y = _centres(c)
    assert np.allclose(c.values[0], 0.5 * x + 2.0 * y, atol=2e-3)  # each centre the mean of its four readings


def test_xyz_decimal_commas_and_dummies(tmp_path):
    plain = grids.read_xyz(_traverses(tmp_path / "a.xyz"))
    for name, sep in (("semi.csv", ";"), ("tabs.txt", "\t"), ("spaces.dat", " ")):
        g = grids.read_xyz(_traverses(tmp_path / name, sep=sep, comma=True))
        assert g.shape == plain.shape and np.allclose(g.values, plain.values) and g.notes.get("decimal_comma"), name
    spaced = tmp_path / "commas.csv"  # "x, y, v": commas between the columns, decimal points
    spaced.write_text("x, y, v\n" + "\n".join(f"{x}, {y}, {x + y}" for y in (0.5, 1.0) for x in (0.5, 1.0, 1.5)) + "\n")
    g = grids.read_xyz(spaced)
    assert g.shape == (2, 3) and not g.notes.get("decimal_comma") and g.values[0, 0, 0] == pytest.approx(1.5)
    # dummy readings are no readings: Geoscan's, the usual no-data numbers, Surfer's blank, one of yours
    rows = ["x y v"]
    dummies = {(0, 0): 2047.5, (1, 1): -9999, (2, 2): 1.70141e38, (3, 3): 32767, (4, 4): -32768, (5, 5): -1.5}
    for r in range(8):
        for col in range(10):
            rows.append(f"{0.25 + 0.5 * col} {0.25 + 0.5 * r} {dummies.get((r, col), r + col * 0.1)}")
    p = tmp_path / "dummies.xyz"
    p.write_text("\n".join(rows) + "\n")
    g = grids.read_xyz(p, dummy=-1.5)
    assert g.shape == (8, 10) and g.notes["dummies"] == 6
    v = g.values[0][::-1]  # rows from the bottom, as written
    assert all(np.isnan(v[r, col]) for r, col in dummies) and np.isfinite(v).sum() == 80 - 6
    assert np.isfinite(grids.read_xyz(p).values[0][::-1][5, 5])  # -1.5 is a reading unless said otherwise


# ------------------------------------------------------------------ georeferencing again, GRD, slices
def test_placing_again_is_the_same_as_placing_once(tmp_path):
    g, _, _ = synthetic(rows=40, cols=60, cell=0.5)
    georef.local_to_corner_frame(g)
    corners = georef.corners_local(g)
    a = np.radians(35.0)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    t = np.array([500100.0, 4500800.0])
    world = {k: R @ np.array(v) + t for k, v in corners.items()}
    names = ("lower-left", "lower-right", "upper-left")
    fit = georef.similarity([corners[k] for k in names], [world[k] for k in names])
    once = georef.place(grids.Grid("x", g.values, g.cell, g.origin, "LOCAL"), fit, CRS).corners()
    placed = georef.place(georef.place(grids.Grid("y", g.values, g.cell, g.origin, "LOCAL"), fit, CRS), fit, CRS)
    assert np.allclose(placed.corners(), once, atol=1e-6)  # was thrown kilometres away
    # placed again with other pegs: where those pegs say
    moved = georef.similarity([corners[k] for k in names], [world[k] + (10.0, -5.0) for k in names])
    assert np.allclose(georef.place(placed, moved, CRS).corners()[3], world["lower-left"] + (10.0, -5.0), atol=1e-6)
    # a free scale applied twice does not grow the grid twice; the local frame stays the survey's
    scaled = {k: 1.02 * (R @ np.array(v)) + t for k, v in corners.items()}
    fit_s = georef.similarity(list(corners.values()), [scaled[k] for k in corners], fix_scale=False)
    s1 = georef.place(grids.Grid("z", g.values, g.cell, g.origin, "LOCAL"), fit_s, CRS)
    c1, cell1 = s1.corners().copy(), s1.cell
    s2 = georef.place(s1, fit_s, CRS)
    assert cell1 == pytest.approx(0.51) and s2.cell == pytest.approx(cell1) and np.allclose(s2.corners(), c1, atol=1e-6)
    assert georef.corners_local(s2)["upper-right"] == (30.0, 20.0)
    by_origin = georef.place_by_origin(s2, *world["lower-left"], 55.0, CRS)  # the survey's own size again
    assert by_origin.cell == pytest.approx(0.5) and np.allclose(by_origin.corners(), once, atol=1e-6)
    # the survey cell goes into the project with the grid, and comes back
    doc, ch = _project(tmp_path)
    e = grids.save_grid(ch, georef.place(grids.Grid("w", g.values, g.cell, g.origin, "LOCAL"), fit_s, CRS))
    back = grids.load_grid(ch, e["id"])
    assert back.local_cell == pytest.approx(0.5) and np.allclose(georef.place(back, fit_s, CRS).corners(), c1, atol=1e-6)
    # a grid of 0.1.0 has no survey cell: the cell, divided by the scale of its last placement
    assert grids._survey_cell_of({"cell": 0.51, "filters": [{"name": "georef", "scale": 1.02}]}) == pytest.approx(0.5)


def test_a_turned_grid_is_written_north_up_in_surfer(tmp_path):
    from ppg_geophysics import overlay as overlay_mod

    g, _, _ = synthetic(rows=40, cols=60, cell=0.5)
    georef.local_to_corner_frame(g)
    georef.place_by_origin(g, 500100.0, 4500800.0, 60.0, CRS)  # the rows 30 degrees north of east
    back = grids.read_grd(grids.write_grd(g, tmp_path / "turned.grd"))
    x0, y0, x1, y1 = g.bounds()
    assert back.cell == pytest.approx(0.5) and back.origin == pytest.approx((x0, y1))
    assert back.shape == (int(np.ceil((y1 - y0) / 0.5 - 1e-6)), int(np.ceil((x1 - x0) / 0.5 - 1e-6)))
    v = back.values[0]
    assert np.isnan(v[0, 0]) and np.isnan(v[-1, -1])  # the corners of the box: outside the turned grid, blank
    ref = overlay_mod.resample_to(g, 0, back.transform(), back.shape)  # the grid on the same north-up cells
    both = np.isfinite(ref) & np.isfinite(v)
    assert both.sum() > 0.4 * v.size and np.allclose(v[both], ref[both], atol=1e-3)
    assert grids.north_up(grids.Grid("n", g.values, 0.5, (0.0, 20.0))).rotation == 0.0  # north up: as it is


def test_slices_in_the_natural_order_of_their_names(tmp_path):
    g, _, _ = synthetic(rows=10, cols=12)
    folder = tmp_path / "gpr"
    folder.mkdir()
    for k in (10, 2, 1):
        grids.write_geotiff(grids.Grid(f"s{k}", np.full((1, 10, 12), float(k), np.float32), g.cell, g.origin, CRS),
                            folder / f"slice_{k}.tif", CRS)
    s = grids.read_slices(folder)
    assert s.slices == ["slice_1", "slice_2", "slice_10"] and [float(b[0, 0]) for b in s.values] == [1.0, 2.0, 10.0]


def test_the_defaults_of_the_filters_and_their_window():
    assert filters.FILTERS["lowpass"][1]["sigma"] == 1.0 and filters.FILTERS["highpass"][1]["sigma"] == 10.0
    assert filters.from_window("clip", {"low": 2.0, "high": 98.0, "sigma_low": 1.0, "copy": True}) == {
        "low": 2.0, "high": 98.0, "absolute": False}
    assert filters.from_window("highpass", {"sigma_low": 1.0, "sigma_high": 12.0}) == {"sigma": 12.0}
    assert filters.from_window("lowpass", {"sigma_low": 2.0, "sigma_high": 12.0}) == {"sigma": 2.0}


def test_the_summary_in_the_language_of_the_interface(tmp_path):
    from pyarchinit_photogrammetry.gui import i18n

    from ppg_geophysics import anomalies

    doc, ch = _project(tmp_path)
    g = grids.read_any(_traverses(tmp_path / "survey.xyz"), kind="magnetometry")
    georef.local_to_corner_frame(g)
    grids.save_grid(ch, g)
    anomalies.add(ch, [(0, 0), (1, 0), (1, 1)], grids.entries(ch)[0], "1", "un muro")
    text = export.summary(ch)
    assert "magnetometria nT, 240 x 45 celle di 0.125 m" in text and "righe interpolate fra traverse a 0.5 m" in text
    assert "Anomalie disegnate: 1" in text
    before = i18n.current()
    try:
        i18n.set_current("en")
        text = export.summary(ch)
        assert "magnetometry nT, 240 x 45 cells of 0.125 m, 1 band, local" in text and "celle" not in text
        assert "rows interpolated between traverses 0.5 m apart (readings every 0.125 m)" in text
        assert "Anomalies drawn: 1" in text
    finally:
        i18n.set_current(before)


# ------------------------------------------------------------------ in the window
@pytest.fixture
def window(tmp_path, monkeypatch):
    """The program's window on a project with a georeferenced grid and no orthomosaic nor DEM."""
    pytest.importorskip("PySide6")
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_geophysics import GeophysicsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    qapp = QApplication.instance() or QApplication([])
    doc, ch = _project(tmp_path)
    g, _, _ = synthetic()  # 30 x 40 m of 0.5 m cells
    georef.local_to_corner_frame(g)
    georef.place_by_origin(g, 500100.0, 4500800.0, 90.0, CRS)
    e = grids.save_grid(ch, g)
    doc.save()
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))  # a modal error box would stop the test
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    plugin = GeophysicsPlugin()
    plugin.activate(Context("geophysics", tmp_path / "plugin", window=w))
    try:
        yield w, plugin, ch, e, qapp, errors
    finally:
        plugin.deactivate()
        w.setWindowModified(False)
        w.close()


def test_clip_from_the_filters_window(window, monkeypatch):
    from pyarchinit_photogrammetry.gui import dialogs

    w, plugin, ch, e, qapp, errors = window
    seen = {}

    def ask(parent, title, fields, intro=""):  # the real window, accepted with Clip chosen
        seen.update({f.name: f.default for f in fields})
        values = dialogs.ParamDialog(title, fields, parent, intro).values()
        values["filter"] = "clip"
        return values

    monkeypatch.setattr(dialogs.ParamDialog, "ask", ask)
    plugin.filter_dialog(e["id"])
    w.runner.wait()
    qapp.processEvents()
    assert not errors and [x["name"] for x in grids.entries(ch)] == ["Area 1", "Area 1 clip"]
    src, clipped = grids.load_grid(ch, "Area 1"), grids.load_grid(ch, "Area 1 clip")
    lo, hi = np.nanpercentile(src.values[0], [1, 99])
    assert np.nanmax(clipped.values) == pytest.approx(hi, abs=1e-3) and np.nanmin(clipped.values) == pytest.approx(lo, abs=1e-3)
    assert clipped.filters[-1] == {"name": "clip", "low": 1.0, "high": 99.0, "absolute": False}
    assert seen["sigma_low"] == filters.FILTERS["lowpass"][1]["sigma"]  # the window has the defaults of the command line
    assert seen["sigma_high"] == filters.FILTERS["highpass"][1]["sigma"]


def test_the_grid_shows_on_an_empty_view(window):
    w, plugin, ch, e, qapp, errors = window
    v = w.ortho_view
    assert v.tr is None  # no orthomosaic, no DEM
    plugin.show_overlay(e["id"], 0, "grey", 0.7)
    assert v.tr is not None and plugin.overlay.shown()
    x0, y0, x1, y1 = grids.load_grid(ch, e["id"]).bounds()
    (fx0, fy1), (fx1, fy0) = v.world(v.sceneRect().topLeft()), v.world(v.sceneRect().bottomRight())
    assert fx0 < x0 < x1 < fx1 and fy0 < y0 < y1 < fy1  # the frame holds the grid
    img = plugin.overlay.item.pixmap().toImage()
    assert img.pixelColor(img.width() // 2, img.height() // 2).alpha() == 255  # drawn in the middle


def test_draw_an_anomaly(window, monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    from ppg_geophysics import anomalies
    from pyarchinit_photogrammetry.engine.dem import write_geotiff
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.io import gpkg

    w, plugin, ch, e, qapp, errors = window
    v = w.ortho_view
    monkeypatch.setattr(dialogs.ParamDialog, "ask", lambda parent, title, fields, intro="": {"label": "A1", "note": "fossa?"})

    def click(x, y):
        QTest.mouseClick(v.viewport(), Qt.MouseButton.LeftButton, pos=v.mapFromScene(v.scene_pt(x, y)))

    plugin.pane.table.selectRow(0)
    plugin.draw_anomaly()  # the view is empty: the grid is shown first, on a frame of its own
    assert v.tr is not None and v.mode == "geophysics_anomaly" and plugin._tool.active
    for x, y in ((500110.0, 4500810.0), (500114.0, 4500810.0), (500114.0, 4500814.0), (500110.0, 4500814.0)):
        click(x, y)
    QTest.keyClick(v, Qt.Key.Key_Return)
    assert not plugin._tool.active and v.mode == "pan"
    (a,) = anomalies.entries(ch)
    assert (a["etichetta"], a["griglia"], a["banda"], a["note"]) == ("A1", "Area 1", "1", "fossa?")
    assert a["area_m2"] == pytest.approx(16.0, rel=0.03)
    assert len(v.features(anomalies.LAYER)) == 1 and v.features(anomalies.LAYER)[0]["note"] == "fossa?"
    # Esc cancels; another tool of the program ends the drawing
    plugin.draw_anomaly()
    click(500120.0, 4500820.0)
    QTest.keyClick(v, Qt.Key.Key_Escape)
    assert not plugin._tool.active and len(anomalies.entries(ch)) == 1
    plugin.draw_anomaly()
    w.set_draw_mode("polygon")
    QTest.mouseMove(v.viewport(), v.mapFromScene(v.scene_pt(500120.0, 4500820.0)))
    assert not plugin._tool.active and v.mode == "polygon"
    w.set_draw_mode("pan")
    # in the GeoPackage of File > Export, from the project
    rows = gpkg.read_table(export.export_gpkg(ch, tmp_path / "geo.gpkg"), "anomalie")
    assert len(rows) == 1 and rows[0][1]["note"] == "fossa?" and rows[0][0].area == pytest.approx(16.0, rel=0.03)
    # another raster in the view: the layer is drawn again on it
    dem = write_geotiff(ch.data_dir / "dem.tif", np.full((200, 200), 100.0, np.float32), (500090.0, 4500850.0), 0.25, CRS)
    ch._set_product("dem", dem)
    w._load_raster("dem")
    for _ in range(3):
        qapp.processEvents()
    assert v.kind == "dem" and len(v.features(anomalies.LAYER)) == 1
    # deleted in the view with the program's Select tool: out of the project too
    v.select_items([v.layers[anomalies.LAYER]["features"][0][0]])
    w.delete_selected(confirm=False)
    for _ in range(3):
        qapp.processEvents()
    assert anomalies.entries(ch) == [] and not errors
