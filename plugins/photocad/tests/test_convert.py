"""A conversion end to end on the fake server: the image rendered from the orthophoto, the DEM or an
elevation, the confirmation before the upload, the drawing brought back in the coordinates of the chunk
(plan) or of the elevation (u, v; and in 3D on its surface), DXF, GeoPackage and SVG written and readable,
the key never written anywhere."""
from __future__ import annotations

import json

import fakecad
import numpy as np
import pytest

from ppg_photocad import client, project, source


@pytest.fixture
def fake(monkeypatch):
    f = fakecad.FakePhotoCAD(credits=7)
    monkeypatch.setattr(client, "_urlopen", f)
    return f


def _polygons(chunk, rec):
    return [g for g, a in project.features(chunk, rec) if g.geom_type == "Polygon"]


def test_the_plan_comes_back_at_its_world_coordinates(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    asked = []
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: asked.append(info) or True)
    (info,) = asked
    assert info["credits"] == 7 and info["width"] == fakecad.W and info["height"] == fakecad.H and info["bytes"] > 0
    assert rec["source"] == "ortho" and rec["frame"] is None and rec["crs"] == fakecad.CRS
    assert rec["credits_remaining"] == 6 and rec["image"]["pixel"] == pytest.approx(fakecad.RES)
    # the DXF asked in metres, without margin, the long side of the image in metres
    (q,) = [c for c in fake.calls("GET", "/conversions/conv_0001/files/dxf") if "units=m&" in c]
    assert "units=m" in q and "margin=0" in q and f"longest={fakecad.W * fakecad.RES:g}" in q
    (square,) = _polygons(ch, rec)
    assert np.allclose(square.bounds, fakecad.square_world(), rtol=0, atol=fakecad.RES / 2)
    lines = [g for g, a in project.features(ch, rec) if g.geom_type == "LineString"]
    assert len(lines) == 1 and np.allclose(lines[0].bounds, (fakecad.X0, fakecad.Y1 - fakecad.H * fakecad.RES,
                                                             fakecad.X0 + fakecad.W * fakecad.RES, fakecad.Y1),
                                                rtol=0, atol=1e-6)
    # the files: DXF in world metres with the PhotoCAD layers, GeoPackage in the CRS of the chunk, SVG + note
    import ezdxf

    from pyarchinit_photogrammetry.io import gpkg

    folder = ch.data_dir / rec["folder"]
    dxf = ezdxf.readfile(folder / rec["files"]["dxf"])
    (poly,) = dxf.modelspace().query("LWPOLYLINE")
    assert poly.dxf.layer == "OUTLINE" and dxf.header["$INSUNITS"] == 6
    xs, ys = zip(*[(p[0], p[1]) for p in poly.get_points()])
    assert np.allclose((min(xs), min(ys), max(xs), max(ys)), fakecad.square_world(), rtol=0, atol=fakecad.RES / 2)
    rows = gpkg.read_table(folder / rec["files"]["gpkg"], "photocad")
    assert sorted(a["layer"] for _g, a in rows) == ["DETAIL", "OUTLINE"]
    import sqlite3

    con = sqlite3.connect(folder / rec["files"]["gpkg"])
    assert con.execute("SELECT srs_id FROM gpkg_geometry_columns").fetchone()[0] == 32633
    con.close()
    assert (folder / rec["files"]["svg"]).read_bytes().startswith(b"<svg")
    note = json.loads((folder / rec["files"]["svg_georef"]).read_text())
    assert note["lower_left"] == pytest.approx([fakecad.X0, fakecad.Y1 - fakecad.H * fakecad.RES])
    assert note["crs"] == fakecad.CRS
    # conversion.json, the record in the project, the key nowhere
    meta = json.loads((folder / "conversion.json").read_text())
    assert meta["id"] == rec["id"] and meta["photocad_id"] == "conv_0001" and meta["credits_used"] == 1
    assert project.records(ch)[0]["id"] == rec["id"]
    for path in list(folder.rglob("*")) + [doc.path]:
        if path.is_file():
            assert fakecad.KEY.encode() not in path.read_bytes(), path


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


def test_a_part_and_a_large_raster(tmp_path, fake):
    # 6000 x 400 pixels: rendered at a third (2000 x 134), the square still within half a pixel of the image
    sq = (3001, 40, 299)  # not on the 3-pixel grid: the edges fall inside pixels of the image
    doc, ch = fakecad.project(tmp_path, w=6000, h=400, square=sq)
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: True)
    assert rec["image"]["width"] == 2000 and rec["image"]["pixel"] == pytest.approx(3 * fakecad.RES)
    (square,) = _polygons(ch, rec)
    assert np.allclose(square.bounds, fakecad.square_world(sq), rtol=0, atol=3 * fakecad.RES / 2)
    # a part: a box around the square, the drawing still where it was
    x0, y0, x1, y1 = fakecad.square_world(sq)
    rec = project.convert(ch, "ortho", fakecad.KEY, bbox=(x0 - 0.5, y0 - 0.4, x1 + 0.7, y1 + 0.3),
                          confirm=lambda info: True)
    assert rec["image"]["pixel"] == pytest.approx(fakecad.RES) and rec["image"]["width"] <= 2400
    (square,) = _polygons(ch, rec)
    assert np.allclose(square.bounds, (x0, y0, x1, y1), rtol=0, atol=fakecad.RES / 2)


def test_render_limits_the_size(tmp_path):
    doc, ch = fakecad.project(tmp_path, w=3000, h=2000)
    img = source.render(ch, "ortho", max_bytes=60_000)
    assert len(img.data) <= 60_000 and img.fmt == "jpeg" and max(img.width, img.height) <= 2400
    from PIL import Image
    import io

    assert Image.open(io.BytesIO(img.data)).size == (img.width, img.height)
    with pytest.raises(ValueError):
        source.render(ch, "ortho", bbox=(0, 0, 1, 1))  # outside the raster


def test_the_dem_as_a_hillshade(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    img = source.render(ch, "dem")
    assert img.kind == "dem" and img.pixel == pytest.approx(4 * fakecad.RES)
    rec = project.convert(ch, "dem", fakecad.KEY, confirm=lambda info: True)
    assert rec["source"] == "dem" and rec["frame"] is None
    for g, _a in project.features(ch, rec):  # the drawing within the DEM
        b = g.bounds
        assert fakecad.X0 - 1e-6 <= b[0] and b[2] <= fakecad.X0 + fakecad.W * fakecad.RES + 1e-6


def test_an_elevation_in_u_v_and_in_3d(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    entry = fakecad.add_elevation(ch)
    doc.save()
    rec = project.convert(ch, "elevation:e1", fakecad.KEY, three_d=True, confirm=lambda info: True)
    assert rec["frame"] == "e1" and rec["crs"] is None and rec["source"] == "elevation:e1"
    (square,) = _polygons(ch, rec)
    assert np.allclose(square.bounds, fakecad.elevation_square(), rtol=0, atol=fakecad.E_RES / 2)
    folder = ch.data_dir / rec["folder"]
    import sqlite3

    con = sqlite3.connect(folder / rec["files"]["gpkg"])
    srs = dict(con.execute("SELECT table_name, srs_id FROM gpkg_geometry_columns").fetchall())
    con.close()
    assert srs["photocad"] == -1 and srs["photocad_3d"] == 32633
    # 3D: on the surface of the elevation, the plane moved by the relief's depth towards the viewer
    import ezdxf

    from pyarchinit_photogrammetry.engine.elevation import world_of

    dxf3 = ezdxf.readfile(folder / rec["files"]["dxf_3d"])
    pts = np.array([list(v.dxf.location) for e in dxf3.modelspace().query("POLYLINE") if e.dxf.layer == "OUTLINE"
                    for v in e.vertices])
    u0, v0, u1, v1 = fakecad.elevation_square()
    expect = world_of(entry, u0, v0, None, chunk=ch)
    assert np.allclose(pts.min(axis=0), expect, rtol=0, atol=fakecad.E_RES)
    assert expect[1] == pytest.approx(fakecad.E_ORIGIN[1] - fakecad.E_DEPTH)  # the plane faces south (-y)
    assert pts[:, 2].max() == pytest.approx(world_of(entry, u0, v1, None, chunk=ch)[2], abs=fakecad.E_RES)


def test_export_and_remove(tmp_path, fake):
    doc, ch = fakecad.project(tmp_path)
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: True)
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
    rec = project.convert(ch, "ortho", fakecad.KEY, confirm=lambda info: True)
    polygons, attributes, skipped = project.unit_polygons(project.features(ch, rec), tipo="USM", start=10)
    assert len(polygons) == 2 and skipped == 1  # the open diagonal
    assert [a["us"] for a in attributes] == ["10", "11"] and all(a["tipo"] == "USM" for a in attributes)
    assert all(a["photocad_layer"] == "OUTLINE" for a in attributes)
    _p, attributes, _s = project.unit_polygons(project.features(ch, rec), tipo="US")
    assert all("us" not in a for a in attributes)
