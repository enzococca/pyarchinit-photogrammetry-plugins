"""After the flight: the planned shots against the photos of a chunk (references with noise, shots missing, photos far
from the plan, the height above the takeoff from the XMP), the coverage obtained against the predicted one, the files
written, the step of the pane with its layer, the sheet's page and the command line."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest
import synth

from ppg_flightplan import compare, coverage, project

CRS = "EPSG:32632"


def grid_plan(quota=60.0) -> dict:
    area = synth.area_latlon(300.0, 200.0)
    return {"nome": "Prova", "camera": "mini4pro", "alta": True, "minutiBatteria": 18,
            "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in area], "quota": quota,
                         "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0, "velocita": 5,
                         "modo": "per_punto", "alta": True}]}


def _xy(latlons) -> np.ndarray:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    a = np.asarray(latlons, float)
    x, y = tr.transform(a[:, 1], a[:, 0])
    return np.column_stack([x, y])


def _add_photos(ch, xy, heights=None, folder=None, relative=None) -> list:
    """Cameras whose references are ``xy`` (chunk's system); with ``folder`` a JPEG for each, with ``relative`` its
    ``drone-dji:RelativeAltitude`` in the XMP."""
    from PIL import Image

    from pyarchinit_photogrammetry.core.camera import Camera, Reference

    start = len(ch.cameras)
    cams = []
    for k, (x, y) in enumerate(xy):
        path = f"/nonexistent/DJI_{start + k:04d}.JPG"
        if folder is not None:
            path = str(folder / f"DJI_{start + k:04d}.JPG")
            xmp = None
            if relative is not None:
                xmp = ('<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-'
                       'syntax-ns#"><rdf:Description xmlns:drone-dji="http://www.dji.com/drone-dji/1.0/" '
                       f'drone-dji:RelativeAltitude="{relative[k]:+.2f}"/></rdf:RDF></x:xmpmeta>').encode()
            Image.new("RGB", (16, 12), (90, 90, 90)).save(path, "JPEG", **({"xmp": xmp} if xmp else {}))
        z = 160.0 if heights is None else heights[k]
        cam = Camera(key=start + k, label=f"DJI_{start + k:04d}", path=path, sensor_key=0,
                     reference=Reference(location=[float(x), float(y), float(z)], accuracy=[5.0, 5.0, 10.0]))
        ch.cameras.append(cam)
        cams.append(cam)
    return cams


def _noisy(xy, metres=0.5, seed=1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * math.pi, len(xy))
    return np.asarray(xy) + metres * np.column_stack([np.cos(angle), np.sin(angle)])


def test_photos_on_the_plan(tmp_path):
    doc, ch = synth.chunk(tmp_path)
    piano = grid_plan()
    shots = compare.planned_shots(piano)
    assert len(shots) > 60 and len({s["leg"] for s in shots}) >= 1
    gone = list(range(5, len(shots), len(shots) // 10))[:10]
    assert len(gone) == 10
    kept = [s for i, s in enumerate(shots) if i not in gone]
    _add_photos(ch, _noisy(_xy([s["latlon"] for s in kept])))
    out = compare.compare(piano, ch)
    st = out["stats"]
    assert st["planned"] == len(shots) and st["taken"] == len(kept)
    assert st["matched"] == len(shots) - 10 and st["missing"] == 10 and st["extra"] == 0
    assert [(m["leg"], m["n"]) for m in out["missing"]] == [(shots[i]["leg"], shots[i]["n"]) for i in gone]
    assert all(m["tratta"] == shots[i]["tratta"] for m, i in zip(out["missing"], gone))
    assert st["median_offset_m"] == pytest.approx(0.5, abs=0.3) and st["p90_offset_m"] < 1.0
    assert st["median_dh_m"] is None and "coverage" not in out          # no XMP, not aligned
    assert sum(g["missing"] for g in out["legs"]) == 10
    assert sum(g["planned"] for g in out["legs"]) == len(shots)


def test_relative_height_from_xmp(tmp_path):
    doc, ch = synth.chunk(tmp_path)
    piano = grid_plan(60.0)
    shots = compare.planned_shots(piano)
    assert {round(s["height"], 3) for s in shots} == {60.0}
    folder = tmp_path / "foto"
    folder.mkdir()
    _add_photos(ch, _xy([s["latlon"] for s in shots]), folder=folder, relative=[59.80] * len(shots))
    assert compare.relative_height(folder / "DJI_0000.JPG") == pytest.approx(59.8)
    out = compare.compare(piano, ch)
    assert out["stats"]["median_dh_m"] == pytest.approx(-0.2, abs=0.01)
    assert out["stats"]["with_height"] == len(shots)
    # the element form of the XMP is read as well, and a photo without one says nothing
    (tmp_path / "e.xmp").write_bytes(b"<drone-dji:RelativeAltitude>+12.5</drone-dji:RelativeAltitude>")
    assert compare.relative_height(tmp_path / "e.xmp") == 12.5
    assert compare.relative_height(tmp_path / "missing.jpg") is None


def test_exif_gps_when_there_is_no_reference(tmp_path, monkeypatch):
    doc, ch = synth.chunk(tmp_path)
    cams = _add_photos(ch, [(synth.X0, synth.Y0)])
    cams[0].reference.location = None
    from pyarchinit_photogrammetry.io import exif

    monkeypatch.setattr(exif, "read_exif", lambda p: {"gps": (9.0, 41.5, 100.0)})
    (p,) = compare.photo_positions(ch)
    assert p["latlon"] == (41.5, 9.0) and p["source"] == "exif" and p["aligned"] is False


def test_extra_photos(tmp_path):
    doc, ch = synth.chunk(tmp_path)
    piano = grid_plan()
    shots = compare.planned_shots(piano)
    _add_photos(ch, _xy([s["latlon"] for s in shots]))
    _add_photos(ch, [(synth.X0 + 350.0, synth.Y0), (synth.X0, synth.Y0 - 300.0)])    # 200 m beyond the area
    _add_photos(ch, _xy([shots[3]["latlon"]]) + 0.3)                                    # the same shot twice
    out = compare.compare(piano, ch)
    assert out["stats"]["extra"] == 2 and len(out["extra"]) == 2 and out["stats"]["repeats"] == 1
    assert out["stats"]["missing"] == 0 and all(x["distanza_m"] > 150 for x in out["extra"])
    assert "1 foto ripetuta" in compare.summary_line(out) and "2 foto lontane dal piano" in compare.summary_line(out)


def test_nothing_to_compare(tmp_path):
    doc, ch = synth.chunk(tmp_path)
    with pytest.raises(compare.CompareError):
        compare.compare(grid_plan(), ch)
    piano = grid_plan()
    piano["blocchi"][0]["attivo"] = False
    _add_photos(ch, [(synth.X0, synth.Y0)])
    with pytest.raises(compare.CompareError):
        compare.compare(piano, ch)


def _aligned(monkeypatch, ch):
    """The chunk said aligned (its class's properties for the time of the test)."""
    cls = type(ch)
    monkeypatch.setattr(cls, "aligned_cameras", property(lambda self: list(self.cameras)))
    monkeypatch.setattr(cls, "tie_points", property(lambda self: np.zeros((50, 3))))


def test_obtained_coverage(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.engine import fieldcheck

    doc, ch = synth.chunk(tmp_path)
    ch.crs = CRS
    piano = grid_plan()
    shots = compare.planned_shots(piano)
    _add_photos(ch, _xy([s["latlon"] for s in shots]))
    predicted = coverage.predict(piano, CRS)
    h, w = predicted.counts.shape
    hole = np.zeros((h, w), bool)
    hole[h // 3:h // 3 + 12, w // 3:w // 3 + 15] = True
    assert (hole & predicted.inside & (predicted.counts >= predicted.low)).sum() == 12 * 15
    asked = {}

    def fake(chunk, resolution=None, progress=None, used=None):
        asked["resolution"] = resolution
        counts = np.where(hole, 0, predicted.counts).astype(np.int32)
        return counts, predicted.transform, None

    monkeypatch.setattr(fieldcheck, "coverage", fake)
    _aligned(monkeypatch, ch)
    out = compare.compare(piano, ch, predicted=predicted)
    c = out["coverage"]
    assert asked["resolution"] == pytest.approx(predicted.cell)
    assert c["cells_worse"] == 12 * 15
    assert c["area_worse_m2"] == pytest.approx(12 * 15 * predicted.cell ** 2, rel=1e-3)
    assert c["obtained_low_fraction"] > c["predicted_low_fraction"]
    assert c["obtained_low_fraction"] - c["predicted_low_fraction"] == pytest.approx(
        12 * 15 / predicted.inside.sum(), abs=1e-3)
    # the files: the JSON without the grids, the missing shots, the rasters on the predicted grid
    paths = compare.write(out, tmp_path / "out", "x", crs=CRS)
    assert set(paths) == {"json", "gpkg", "count", "rgba"}
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    assert "_raster" not in data and data["coverage"]["cells_worse"] == 12 * 15
    import rasterio

    with rasterio.open(paths["count"]) as ds:
        got = ds.read(1)
        assert ds.transform == predicted.transform and ds.crs is not None
    assert (got[hole] == 0).all() and (got[~predicted.inside] == -1).all()


def test_compare_for_keeps_the_record(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.io import gpkg

    doc, ch = synth.chunk(tmp_path)
    id_ = project.add_plan(ch, grid_plan())
    piano = project.entry(ch, id_)["piano"]
    shots = compare.planned_shots(piano)
    _add_photos(ch, _xy([s["latlon"] for s in shots[3:]]))
    rec = compare.compare_for(ch, id_)
    assert project.peek(ch)["comparison"][id_] is rec
    assert compare.comparison_record(ch, id_) is rec and rec["stats"]["missing"] == 3
    rows = gpkg.read_table(ch.data_dir / rec["files"]["gpkg"], compare.MISSING_TABLE)
    assert len(rows) == 3
    xy = np.array([[g.x, g.y] for g, _a in rows])
    assert np.abs(xy - _xy([s["latlon"] for s in shots[:3]])).max() < 0.01        # in the chunk's system
    line = compare.summary_line(rec)
    assert line.startswith(f"{len(shots) - 3} di {len(shots)} scatti fatti; scarto mediano 0 m; 3 mancanti "
                           "nella tratta 1")
    sheet = compare.for_sheet(rec)
    assert sheet["Scatti mancanti"] and sheet["Scatti previsti"] == len(shots)
    # the sheet of the project gets it
    from openpyxl import load_workbook
    import io

    out = project.export(ch, id_, "sheet-xlsx", tmp_path / "scheda.xlsx")
    wb = load_workbook(io.BytesIO(out[0].read_bytes()))
    values = [c for row in wb["Confronto"].iter_rows(values_only=True) for c in row if c is not None]
    assert "Scatti mancanti" in values and len(shots) in values
    # a photo moved: the comparison is old
    ch.cameras[0].reference.location[0] += 5.0
    assert compare.comparison_record(ch, id_) is None
    # the plan removed: its comparison goes too
    project.remove_plan(ch, id_)
    assert id_ not in project.peek(ch)["comparison"]


def test_command_compare(tmp_path, capsys):
    from test_project_command import _plugin

    doc, ch = synth.chunk(tmp_path)
    id_ = project.add_plan(ch, grid_plan())
    shots = compare.planned_shots(project.entry(ch, id_)["piano"])
    _add_photos(ch, _xy([s["latlon"] for s in shots[:-4]]))
    doc.save()
    plugin = _plugin(tmp_path)
    assert plugin.command("compare", str(doc.path), "--max-distance", "3") == 0
    text = capsys.readouterr().out
    data = json.loads(text[text.index("{\n"):])
    assert "missing" in data and len(data["missing"]) == 4 and data["id"] == id_
    assert data["max_distance_m"] == 3.0 and "4 mancanti" in data["riassunto"]
    import pyarchinit_photogrammetry as ppg

    again = ppg.Document.open(doc.path)
    assert again.chunk.meta["flightplan"]["comparison"][id_]["stats"]["missing"] == 4
    assert plugin.command("compare", str(doc.path), "--plan-id", "pnothere0") == 1
    assert plugin.command("compare", str(doc.path), "--max-distance", "-1") == 1


def test_a_reference_in_degrees_moved_by_a_metre_makes_it_old(tmp_path):
    doc, ch = synth.chunk(tmp_path)
    id_ = project.add_plan(ch, grid_plan())
    shots = compare.planned_shots(project.entry(ch, id_)["piano"])
    cams = _add_photos(ch, _xy([s["latlon"] for s in shots]))
    for c, s in zip(cams, shots):                       # the EXIF GPS as the program keeps it: lon, lat in degrees
        c.reference.location = [s["latlon"][1], s["latlon"][0], 160.0]
        c.meta["reference_crs"] = "EPSG:4326"
    rec = compare.compare_for(ch, id_)
    assert rec["stats"]["matched"] == len(shots) and compare.comparison_record(ch, id_) is rec
    cams[0].reference.location[1] += 1e-5               # about 1 m north
    assert compare.comparison_record(ch, id_) is None


def test_one_system_that_cannot_be_read_leaves_the_others(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.markers import gcp

    doc, ch = synth.chunk(tmp_path)
    cams = _add_photos(ch, [(synth.X0, synth.Y0), (synth.X0 + 10, synth.Y0)])
    cams[1].meta["reference_crs"] = "EPSG:9999"
    real = gcp.camera_references

    def refs(chunk, group, crs=None, relative=False):
        if any(c.meta.get("reference_crs") == "EPSG:9999" for c in group):
            raise ValueError("a system that cannot be read")
        return real(chunk, group, crs=crs, relative=relative)

    monkeypatch.setattr(gcp, "camera_references", refs)
    from pyarchinit_photogrammetry.io import exif

    monkeypatch.setattr(exif, "read_exif", lambda p: {"gps": (9.0, 41.5, 100.0)})
    a, b = compare.photo_positions(ch)
    assert a["source"] == "reference" and b["source"] == "exif" and b["latlon"] == (41.5, 9.0)


def test_alignment_georeference_or_dem_changed_make_the_coverage_old(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.engine import fieldcheck

    doc, ch = synth.chunk(tmp_path, dem=100.0)
    id_ = project.add_plan(ch, grid_plan())
    shots = compare.planned_shots(project.entry(ch, id_)["piano"])

    def fake(chunk, resolution=None, progress=None, used=None):
        pred = project.read_coverage(chunk, id_)
        return pred.counts.astype(np.int32), pred.transform, None

    monkeypatch.setattr(fieldcheck, "coverage", fake)
    _aligned(monkeypatch, ch)
    # nothing to compare: refused before any predicted coverage is computed
    with pytest.raises(compare.CompareError):
        compare.compare_for(ch, id_)
    assert id_ not in project.peek(ch).get("coverage", {})
    _add_photos(ch, _xy([s["latlon"] for s in shots]))
    rec = compare.compare_for(ch, id_)
    assert rec["coverage"] and rec["basis"] and compare.comparison_record(ch, id_) is rec
    old = ch.transform.copy()
    ch.transform = old.copy()
    ch.transform[0, 3] += 0.01                          # georeferenced again (or re-aligned): another frame
    assert compare.comparison_record(ch, id_) is None
    ch.transform = old
    assert compare.comparison_record(ch, id_) is rec
    cams = ch.cameras
    cams[0].transform = np.eye(4)                       # a pose changed by an optimisation
    assert compare.comparison_record(ch, id_) is None
    cams[0].transform = None
    assert compare.comparison_record(ch, id_) is rec
    synth.write_dem(ch.data_dir / "dem.tif", 101.0)     # a new DEM
    import os
    import time

    st = (ch.data_dir / "dem.tif").stat()
    os.utime(ch.data_dir / "dem.tif", (time.time() + 5, st.st_mtime + 5))
    assert compare.comparison_record(ch, id_) is None


def test_obtained_grid_shifted_and_smaller(tmp_path, monkeypatch):
    """The field check's grid has another origin (0.37 cell off) and stops 3 cells inside the area on every side
    (-1 beyond): the reprojection puts the hole where it is, and the cells it does not reach are not
    measured."""
    from rasterio.transform import Affine

    from pyarchinit_photogrammetry.engine import fieldcheck

    doc, ch = synth.chunk(tmp_path)
    piano = grid_plan()
    shots = compare.planned_shots(piano)
    _add_photos(ch, _xy([s["latlon"] for s in shots]))
    predicted = coverage.predict(piano, CRS)
    cell = predicted.cell
    h, w = predicted.counts.shape
    pt = predicted.transform
    shift = 0.37 * cell
    hole = np.zeros((h, w), bool)
    hole[h // 3:h // 3 + 12, w // 3:w // 3 + 15] = True
    rows, cols = np.nonzero(predicted.inside)
    kept = np.zeros((h, w), bool)                       # the area less 3 cells on every side: beyond, -1
    kept[rows.min() + 3:rows.max() - 2, cols.min() + 3:cols.max() - 2] = True

    def fake(chunk, resolution=None, progress=None, used=None):
        # the source grid: the predicted cells shifted east and south by 0.37 cell; a -1 margin of 3 cells
        src = np.where(hole, 0, predicted.counts).astype(np.int32)
        src[~kept] = -1
        return src, Affine(pt.a, 0, pt.c + shift, 0, pt.e, pt.f - shift), None

    monkeypatch.setattr(fieldcheck, "coverage", fake)
    _aligned(monkeypatch, ch)
    out = compare.compare(piano, ch, predicted=predicted)
    c = out["coverage"]
    # nearest: a shift under half a cell keeps every cell where it was
    assert c["cells_worse"] == 12 * 15
    margin = ~kept
    assert c["outside_reconstruction"] == int((margin & predicted.inside).sum()) > 0
    assert c["cells_measured"] == int((~margin & predicted.inside).sum())
    raster = out["_raster"]
    assert (raster["worse"] == (hole & predicted.inside & (predicted.counts >= predicted.low))).all()
    # the maps agree with the numbers: the cells outside the reconstruction are nodata / transparent, not 0 photos
    import rasterio

    paths = compare.write(out, tmp_path / "out", "x", crs=CRS)
    outside = margin & predicted.inside
    measured = ~margin & predicted.inside
    with rasterio.open(paths["count"]) as ds:
        got = ds.read(1)
    assert (got[outside] == -1).all() and (got[measured] >= 0).all() and (got[hole & measured] == 0).all()
    with rasterio.open(paths["rgba"]) as ds:
        alpha = ds.read(4)
    assert (alpha[outside] == 0).all() and (alpha[hole & measured] > 0).all()


# ------------------------------------------------------------------ the window
from test_window import _design, _wait, gui  # noqa: E402,F401 - the window of the plugin's tests


def test_pane_step_and_layer(gui, monkeypatch):  # noqa: F811
    from ppg_flightplan.texts import LAYER_MISSING

    w, plugin, ch, app_, said, _request = gui
    p = plugin.pane
    _design(w, plugin, app_)
    e = project.active_plan(ch)
    assert "aggiungi al chunk le foto" in p.states["after"].text()
    assert p.b_compare.isEnabled() and not p.b_show_obtained.isVisible()
    shots = compare.planned_shots(e["piano"])
    gone = {2, 7, 11}
    _add_photos(ch, _noisy(_xy([s["latlon"] for i, s in enumerate(shots) if i not in gone]), 0.4))
    plugin.refresh_pane()
    assert "Pronto" in p.states["after"].text()
    w.setWindowModified(False)
    p.b_compare.click()
    _wait(w, app_)
    assert not said, said
    rec = compare.comparison_record(ch, e["id"])
    assert rec is not None and w.isWindowModified()
    text = p.states["after"].text()
    assert text.startswith(f"{len(shots) - 3} di {len(shots)} scatti fatti; scarto mediano 0,4 m; 3 mancanti")
    missing = w.ortho_view.features(LAYER_MISSING)
    assert len(missing) == 3
    assert sorted(f["scatto"] for f in missing) == sorted(shots[i]["n"] for i in gone)
    # in the project's tree
    items = plugin._tree_items(ch)
    labels = [k["label"] for plan in items[0]["children"] for k in plan.get("children") or []]
    assert "Confronto con il volo" in labels
    # the plan changed: the comparison is old, its layer leaves the view
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Unchecked)
    app_.processEvents()
    assert w.ortho_view.features(LAYER_MISSING) == [] and "Da rifare" in p.states["after"].text()
    # the rail has the action
    groups = [g for g in plugin.context.contributions.rail_groups if g.key == "flightplan"]
    assert "Confronta con il volo" in [a.title for a in groups[0].actions]


def test_obtained_coverage_on_the_view(gui, monkeypatch):  # noqa: F811
    from pathlib import Path

    from pyarchinit_photogrammetry.engine import fieldcheck

    w, plugin, ch, app_, said, _request = gui
    p = plugin.pane
    _design(w, plugin, app_)
    e = project.active_plan(ch)
    shots = compare.planned_shots(e["piano"])
    _add_photos(ch, _xy([s["latlon"] for s in shots]))

    def fake(chunk, resolution=None, progress=None, used=None):
        pred = project.read_coverage(chunk, e["id"])
        return pred.counts.astype(np.int32) // 2, pred.transform, None

    monkeypatch.setattr(fieldcheck, "coverage", fake)
    _aligned(monkeypatch, ch)
    p.b_compare.click()
    _wait(w, app_)
    assert not said, said
    rec = compare.comparison_record(ch, e["id"])
    assert rec["coverage"]["cells_worse"] > 0 and "Copertura ottenuta" in p.states["after"].text()
    assert not p.b_show_obtained.isHidden()
    p.b_show_obtained.click()
    app_.processEvents()
    assert Path(w.ortho_view.path).resolve() == (ch.data_dir / rec["files"]["rgba"]).resolve()
    # the plan changed: the coverage obtained shown belonged to it and leaves the view
    p.t_plan.item(0, 0).setCheckState(p.t_plan.item(0, 0).checkState().__class__.Unchecked)
    app_.processEvents()
    shown = getattr(w.ortho_view, "path", None)
    assert not shown or Path(shown).resolve() != (ch.data_dir / rec["files"]["rgba"]).resolve()
    assert p.b_show_obtained.isHidden()
