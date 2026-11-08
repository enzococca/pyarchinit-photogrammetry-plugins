"""A conversion end to end on the fake server, which behaves as photocad.ai did on the real test: the page of the
drawing is the box of the SUBJECT, in millimetres at a natural size whatever ``units`` says. The plugin downloads it
in mm and places it by registration (the page on the image's content box, then the offset and the scale refined on
the image's edges), in the coordinates of the chunk (plan) or of the elevation (u, v; and in 3D on its surface);
long images go as tiles at native resolution, one conversion each, merged; DXF, GeoPackage and SVG written; the key
never written anywhere."""
from __future__ import annotations

import json

import fakecad
import numpy as np
import pytest

from ppg_photocad import client, project, source

PX = fakecad.RES / 2  # the 600 x 400 orthophoto at 1 cm goes up twice (short side >= 600 px): 5 mm a pixel


@pytest.fixture
def fake(monkeypatch):
    f = fakecad.FakePhotoCAD(credits=7)
    monkeypatch.setattr(client, "_urlopen", f)
    return f


def _polygons(chunk, rec):
    return [g for g, a in project.features(chunk, rec) if g.geom_type == "Polygon"]


def _close(bounds, truth, tol):
    return np.abs(np.asarray(bounds) - np.asarray(truth)).max() <= tol


def test_the_plan_comes_back_at_its_world_coordinates(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    asked = []
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: asked.append(info) or True)
    (info,) = asked
    assert info["credits"] == 7 and info["free_conversions"] == 5 and "conversioni gratuite: 5" in info["text"]
    assert info["tiles"] == 1 and info["width"] == 2 * fakecad.W and info["height"] == 2 * fakecad.H
    assert rec["source"] == "ortho" and rec["frame"] is None and rec["crs"] == fakecad.CRS and rec["status"] == "done"
    (tile,) = rec["tiles"]
    assert tile["image"]["pixel"] == pytest.approx(PX) and tile["image"]["zoom"] == 2
    # downloaded at its natural size in mm: never relying on units/longest for the placement
    for q in fake.calls("GET", "/conversions/conv_0001/files/"):
        assert "longest" not in q
    reg = tile["registration"]
    assert reg["ok"] and reg["residual_px"] < 1.0 and abs(reg["aspect_ratio"] - 1) < 0.02
    (square,) = _polygons(ch, rec)
    assert _close(square.bounds, fakecad.square_world(), 1.5 * PX)
    (line,) = [g for g, a in project.features(ch, rec) if g.geom_type == "LineString"]
    x0, y0, x1, y1 = fakecad.square_world()
    assert abs(line.bounds[3] - y1) <= 1.5 * PX and abs(line.bounds[0] - x0) <= 1.5 * PX
    # the files: DXF in world metres with the PhotoCAD layers, GeoPackage in the CRS of the chunk, SVG + its place
    import sqlite3

    import ezdxf

    from pyarchinit_photogrammetry.io import gpkg

    folder = ch.data_dir / rec["folder"]
    dxf = ezdxf.readfile(folder / rec["files"]["dxf"])
    polys = dxf.modelspace().query("LWPOLYLINE")
    outline = next(p for p in polys if p.dxf.layer == "OUTLINE")
    assert dxf.header["$INSUNITS"] == 6
    xs, ys = zip(*[(p[0], p[1]) for p in outline.get_points()])
    assert _close((min(xs), min(ys), max(xs), max(ys)), fakecad.square_world(), 1.5 * PX)
    rows = gpkg.read_table(folder / rec["files"]["gpkg"], "photocad")
    assert sorted(a["layer"] for _g, a in rows) == ["DETAIL", "OUTLINE"]
    con = sqlite3.connect(folder / rec["files"]["gpkg"])
    assert con.execute("SELECT srs_id FROM gpkg_geometry_columns").fetchone()[0] == 32633
    con.close()
    assert (folder / rec["files"]["svg"]).read_bytes().startswith(b"<svg")
    note = json.loads((folder / rec["files"]["svg_georef"]).read_text())
    assert note["crs"] == fakecad.CRS and note["metres_per_mm"] > 0
    # conversion.json, the record in the project, the key nowhere
    meta = json.loads((folder / "conversion.json").read_text())
    assert meta["id"] == rec["id"] and meta["tiles"][0]["photocad_id"] == "conv_0001" and meta["credits_used"] == 1
    assert meta["tiles"][0]["idempotency_key"] and meta["account_before"] == {"credits": 7, "free_conversions": 5}
    assert project.records(ch)[0]["id"] == rec["id"]
    for path in list(folder.rglob("*")) + [doc.path]:
        if path.is_file():
            assert fakecad.KEY.encode() not in path.read_bytes(), path


def test_registration_corrects_a_subject_box_larger_than_the_content(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(page_pad=(5, 0, 0, 3))  # the subject box a little larger than the blobs
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    reg = rec["tiles"][0]["registration"]
    assert reg["ok"] and reg["residual_px"] < 1.0
    (square,) = _polygons(ch, rec)
    assert _close(square.bounds, fakecad.square_world(), 1.5 * PX)


def test_a_drawing_that_is_not_the_image_is_refused(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(garbage=True)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    with pytest.raises(project.FrameMismatch) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "conv_0001" in str(err.value)
    (rec,) = project.records(ch)
    assert rec["status"] == "mismatch" and not rec["tiles"][0]["registration"]["ok"]
    folder = project.folder_of(ch, rec)
    assert (folder / "t0" / "photocad_mm.dxf").exists() and not (folder / "drawing.gpkg").exists()
    # a page with other proportions than the content: refused too
    fake = fakecad.FakePhotoCAD(page_pad=(80, 0, 0, 0))
    monkeypatch.setattr(client, "_urlopen", fake)
    with pytest.raises(project.FrameMismatch) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "proporzioni" in str(err.value)


def test_a_soft_image_refused_only_for_its_residual_can_be_accepted_after_a_look(tmp_path, monkeypatch):
    """Under water the edges of the image are soft: PhotoCAD's lines lie a few pixels beside them although the
    drawing is in its place (Lagoi 2026-11-08: 6 px, peak 1.8-2.0). The refusal says so, leaves an overlay to check,
    and fetch(accept=True) places it - free, nothing sent again - marked as accepted after a look."""
    fake = fakecad.FakePhotoCAD(soft_px=4)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(project.FrameMismatch) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    (rec,) = project.records(ch)
    reg = rec["tiles"][0]["registration"]
    assert not reg["ok"] and reg["checks"] == {"border": False, "peak": False, "residual": True}
    overlay = project.folder_of(ch, rec) / "t0" / "overlay.jpg"
    assert overlay.exists() and str(overlay) in str(err.value) and "--accept" in str(err.value)
    posts = sum(1 for m, *_ in fake.requests if m == "POST")
    rec = project.fetch(ch, rec["id"], fakecad.KEY, accept=True)
    assert rec["status"] == "done" and rec["tiles"][0]["registration"]["accepted"] == "visual"
    assert sum(1 for m, *_ in fake.requests if m == "POST") == posts
    (square,) = _polygons(ch, rec)
    centre = np.asarray(fakecad.square_world()).reshape(2, 2).mean(axis=0)
    assert np.hypot(*(np.asarray(square.centroid.coords[0]) - centre)) <= 2 * PX


def test_accept_never_places_a_drawing_with_other_problems(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(garbage=True)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    with pytest.raises(project.FrameMismatch) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "--accept" not in str(err.value)
    (rec,) = project.records(ch)
    with pytest.raises(project.FrameMismatch):
        project.fetch(ch, rec["id"], fakecad.KEY, accept=True)
    assert project.records(ch)[0]["status"] == "mismatch"


def test_declined_nothing_is_sent(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    with pytest.raises(project.Declined):
        project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: False)
    assert fake.calls("POST") == [] and project.records(ch) == []


def test_the_confirmation_can_be_turned_off_for_a_project(tmp_path):
    doc, ch = fakecad.project(tmp_path)
    assert project.confirmation_wanted(doc)
    project.set_confirmation_wanted(doc, False)
    doc.save()
    import pyarchinit_photogrammetry as ppg

    assert not project.confirmation_wanted(ppg.Document.open(doc.path))


def test_a_long_raster_goes_as_tiles_merged(tmp_path, monkeypatch):
    # 6000 x 400 px at 1 cm: up twice for the short side, 6 tiles of 1200 native px (2400 sent) with 10 % overlap
    fake = fakecad.FakePhotoCAD(credits=7, free=0)
    monkeypatch.setattr(client, "_urlopen", fake)
    sq = (3001, 40, 299)  # inside the third tile, its right part in the overlap of the fourth
    doc, ch = fakecad.project(tmp_path, w=6000, h=400, square=sq)
    asked = []
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: asked.append(info) or True)
    assert asked[0]["tiles"] == 6 and "6 conversioni" in asked[0]["text"]
    assert len(rec["tiles"]) == 6 and len(fake.calls("POST")) == 6
    assert all(t["image"]["width"] == 2400 and t["image"]["height"] == 800 for t in rec["tiles"])
    placed = [t for t in rec["tiles"] if t["status"] == "placed"]
    assert len(placed) == 2 and rec["status"] == "done"  # the empty tiles failed (refunded)
    (square,) = _polygons(ch, rec)  # the copy in the overlap removed
    assert _close(square.bounds, fakecad.square_world(sq), 1.5 * PX)
    assert rec["counts"]["tiles_failed"] == 4


def test_tiles_need_enough_credits_and_a_guard(tmp_path, monkeypatch):
    fake = fakecad.FakePhotoCAD(credits=1, free=2)
    monkeypatch.setattr(client, "_urlopen", fake)
    doc, ch = fakecad.project(tmp_path, w=6000, h=400, square=(3001, 40, 299))
    with pytest.raises(project.NotEnough) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "6" in str(err.value) and fake.calls("POST") == []
    with pytest.raises(ValueError) as err:
        project.convert(ch, "ortho", fakecad.KEY, max_tiles=3)
    assert "--max-tiles" in str(err.value) and fake.calls("POST") == []
    # a part: one tile
    x0, y0, x1, y1 = fakecad.square_world((3001, 40, 299))
    rec = project.convert(ch, "ortho", fakecad.KEY, bbox=(x0 - 0.5, y0 - 0.4, x1 + 0.7, y1 + 0.3))
    assert len(rec["tiles"]) == 1
    (square,) = _polygons(ch, rec)
    assert _close(square.bounds, (x0, y0, x1, y1), 1.5 * PX)


def test_too_thin_is_said_not_sent(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path, w=3000, h=100, square=(1000, 20, 50))
    with pytest.raises(ValueError) as err:
        project.convert(ch, "ortho", fakecad.KEY)
    assert "600" in str(err.value) and fake.calls("POST") == []


def test_render_limits_the_size(tmp_path):
    doc, ch = fakecad.project(tmp_path, w=2000, h=1500)
    img = source.render(ch, "ortho", window=(0, 0, 2000, 1500), max_bytes=60_000, zoom=1)
    assert len(img.data) <= 60_000 and img.fmt == "jpeg"
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(img.data)).size == (img.width, img.height) == (2000, 1500)
    with pytest.raises(ValueError):
        source.tiles(ch, "ortho", bbox=(0, 0, 1, 1))  # outside the raster


def test_the_dem_as_a_hillshade(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    (t,) = source.tiles(ch, "dem")
    assert t["zoom"] == 3
    img = source.render(ch, "dem", window=t["window"], zoom=t["zoom"])
    assert img.kind == "dem" and img.pixel == pytest.approx(2 * fakecad.RES / 3) and img.height == 600


def test_an_elevation_in_u_v_and_in_3d(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    entry = fakecad.add_elevation(ch)
    doc.save()
    rec = project.convert(ch, "elevation:e1", fakecad.KEY, three_d=True)
    assert rec["frame"] == "e1" and rec["crs"] is None and rec["source"] == "elevation:e1"
    epx = fakecad.E_RES / 2
    (square,) = _polygons(ch, rec)
    assert _close(square.bounds, fakecad.elevation_square(), 1.5 * epx)
    folder = ch.data_dir / rec["folder"]
    import sqlite3

    con = sqlite3.connect(folder / rec["files"]["gpkg"])
    srs = dict(con.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns").fetchall())
    con.close()
    assert srs["photocad"] == -1 and srs["photocad_3d"] == 32633
    import ezdxf

    from pyarchinit_photogrammetry.engine.elevation import world_of

    dxf3 = ezdxf.readfile(folder / rec["files"]["dxf_3d"])
    pts = np.array([list(v.dxf.location) for e in dxf3.modelspace().query("POLYLINE") if e.dxf.layer == "OUTLINE"
                    for v in e.vertices])
    u0, v0, u1, v1 = fakecad.elevation_square()
    expect = world_of(entry, u0, v0, None, chunk=ch)
    assert np.abs(pts.min(axis=0) - expect).max() <= 1.5 * epx
    assert expect[1] == pytest.approx(fakecad.E_ORIGIN[1] - fakecad.E_DEPTH)  # the plane faces south (-y)


def test_export_and_remove(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY)
    for suffix in (".dxf", ".gpkg", ".svg"):
        out = project.export(ch, rec["id"], tmp_path / f"out{suffix}")
        assert out.exists() and out.stat().st_size > 0
    with pytest.raises(ValueError):
        project.export(ch, rec["id"], tmp_path / "out.xyz")
    folder = ch.data_dir / rec["folder"]
    assert project.remove(ch, rec["id"]) and not folder.exists() and project.records(ch) == []
    assert not project.remove(ch, rec["id"])


def test_units_from_closed_outlines(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path, square=[fakecad.SQUARE, fakecad.SQUARE2])
    rec = project.convert(ch, "ortho", fakecad.KEY)
    polygons, attributes, skipped = project.unit_polygons(project.features(ch, rec), tipo="USM", start=10)
    assert len(polygons) == 2 and skipped == 1  # the open line
    assert [a["us"] for a in attributes] == ["10", "11"] and all(a["tipo"] == "USM" for a in attributes)
    assert all(a["photocad_layer"] == "OUTLINE" for a in attributes)
    _p, attributes, _s = project.unit_polygons(project.features(ch, rec), tipo="US")
    assert all("us" not in a for a in attributes)


def test_the_real_khutm_frame():
    """The numbers of the real test (2026-11-04): a 2400 x 1198 image whose content is rows 55-981; PhotoCAD's page
    300 x 115.172 mm = the subject box; its DXF in mm whatever ``units`` asked. The page on the content box: aspect
    within 2 %, about 8 px a mm."""
    from ppg_photocad import register

    aspect = (2400 / (981 - 55 + 1)) / (300 / 115.172)
    assert abs(aspect - 1) < 0.02
    assert register.unit_to_mm(4) == 1.0 and register.unit_to_mm(6) == 1000.0 and register.unit_to_mm(0) == 1.0
