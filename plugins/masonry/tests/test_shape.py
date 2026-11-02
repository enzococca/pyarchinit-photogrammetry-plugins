"""The shape of a face against the truth of the synthetic scenes: the lean of a wall (2 degrees towards the viewer) and
its bulge (3 cm), the horizontal sections stacked with their offsets, the axis of a round tower (1 degree) and its
out-of-round (1 cm), the sides of a hexagonal tower upright, two surveys 2 cm apart."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, shape, walls  # noqa: E402

Z0 = S.WORLD[2]
VIEWER = S.WORLD[:2] + [3.0, -5.0]


def _wall_face(tmp_path, resolution=0.02):
    doc, ch = S.project(tmp_path, S.wall())
    w = walls.new_wall(ch, "Muro nord")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.WALL_LENGTH, 0.0], side=VIEWER,
                             resolution=resolution)
    return doc, ch, w, entry


def _raster(path):
    import rasterio

    with rasterio.open(path) as src:
        values = src.read(1)
        tr = src.transform
        tags = src.tags()
    rows, cols = np.indices(values.shape)
    u = tr.c + (cols + 0.5) * tr.a
    v = tr.f + (rows + 0.5) * tr.e
    return values, u, v, tags


def test_lean_and_bulge_of_a_wall(tmp_path):
    doc, ch, w, entry = _wall_face(tmp_path)
    r = shape.face_shape(ch, w["id"], entry["id"])
    assert r["surface"] == "plane" and r["points"] > 10000
    assert abs(r["lean_deg"] - S.LEAN_DEG) < 0.1, r["lean_deg"]
    assert abs(r["lean_cm_per_m"] - 100 * np.tan(np.radians(S.LEAN_DEG))) < 0.2
    assert r["lean_toward"] == "out" and abs(r["lean_azimuth"] - 180.0) < 1.0
    path = walls.face_folder(ch, w["id"], entry["id"]) / "bulge.tif"
    assert r["bulge"] == str(path.relative_to(ch.data_dir)) and path.exists()
    values, u, v, tags = _raster(path)
    assert tags.get("units") == "cm" and values.dtype == np.float32
    ok = np.isfinite(values)
    assert ok.mean() > 0.95
    truth = 100 * S.bulge_truth(u, v - Z0) * np.cos(np.radians(S.LEAN_DEG))
    err = np.abs(values - truth)[ok]
    assert err.max() < 0.5, (err.max(), np.percentile(err, 99))  # within 5 mm everywhere
    assert abs(r["bulge_max_cm"] - 3.0) < 0.5 and r["bulge_min_cm"] > -0.5
    # the summary of the wall
    data = json.loads((walls.wall_folder(ch, w["id"]) / "wall.json").read_text())
    assert abs(data["faces"][0]["shape"]["lean_deg"] - S.LEAN_DEG) < 0.1
    text = shape.summary_text(r)
    assert "Strapiombo 2,0°" in text and "3,5 cm/m" in text and "verso chi guarda" in text
    assert "Spanciamento" in text


def test_sections_stacked_with_their_offsets(tmp_path):
    import ezdxf

    doc, ch, w, entry = _wall_face(tmp_path, resolution=0.05)
    heights = [Z0 + 0.5, Z0 + 1.5, Z0 + 2.5]
    r = shape.sections(ch, w["id"], entry["id"], heights=heights, stations=[1.0, 3.0, 5.0])
    k = np.tan(np.radians(S.LEAN_DEG))
    for st in r["stations"]:
        offs = {round(o["height"] - Z0, 1): o for o in st["offsets"]}
        assert offs[0.5]["offset"] == 0.0
        expected = (2.0 * k + S.bulge_truth(st["u"], 2.5) - S.bulge_truth(st["u"], 0.5))
        assert abs(offs[2.5]["offset"] - expected) < 0.003, (st, expected)
    mid = r["stations"][1]
    mid_offs = {round(o["height"] - Z0, 1): o["offset"] for o in mid["offsets"]}
    assert abs(mid_offs[1.5] - (1.0 * k + S.BULGE)) < 0.003  # the bulge stands out at mid height
    doc_dxf = ezdxf.readfile(ch.data_dir / r["dxf"])
    layers = {e.dxf.layer for e in doc_dxf.modelspace()}
    assert {"SEZIONE_0", "SEZIONE_1", "SEZIONE_2", "STAZIONI"} <= layers
    zs = sorted({round(e.points().__next__()[2], 3) for e in doc_dxf.modelspace().query("POLYLINE")
                 if e.dxf.layer.startswith("SEZIONE")})
    assert np.allclose(zs, heights, atol=1e-3)
    svg = (ch.data_dir / r["svg"]).read_text()
    assert svg.startswith("<?xml") and svg.count("<polyline") >= 3 and "+7,0 cm" in svg


def test_axis_and_out_of_round_of_a_round_tower(tmp_path):
    doc, ch = S.project(tmp_path, S.round_tower(), "Torre")
    w = walls.new_wall(ch, "Torre", kind="tower")
    entry = faces.cylinder_face(ch, w["id"], height=Z0 + 3.0, resolution=0.04)
    r = shape.face_shape(ch, w["id"], entry["id"])
    assert r["surface"] == "cylinder"
    assert abs(r["axis_deg"] - S.TOWER_LEAN_DEG) < 0.1, r["axis_deg"]
    assert abs(r["axis_azimuth"] - S.TOWER_LEAN_AZIMUTH) < 3.0
    assert abs(r["radius"] - S.TOWER_RADIUS) < 0.003
    values, _u, _v, _tags = _raster(ch.data_dir / r["bulge"])
    assert np.isfinite(values).mean() > 0.9
    assert abs(np.nanpercentile(values, 99) - 0.5) < 0.2 and abs(np.nanpercentile(values, 1) + 0.5) < 0.2
    a = shape.tower_axis(ch, w["id"], heights=[Z0 + h for h in (0.5, 1.5, 2.5, 3.5, 4.5, 5.5)])
    assert abs(a["axis_deg"] - S.TOWER_LEAN_DEG) < 0.1, a["axis_deg"]
    assert abs(a["axis_azimuth"] - S.TOWER_LEAN_AZIMUTH) < 3.0
    assert len(a["levels"]) == 6
    for level in a["levels"]:
        assert abs(level["out_of_round"] - S.OUT_OF_ROUND) < 0.003, level
        assert abs(level["radius"] - S.TOWER_RADIUS) < 0.003
    assert abs(a["lean_cm_per_m"] - 100 * np.tan(np.radians(S.TOWER_LEAN_DEG))) < 0.2
    assert walls.need(ch, w["id"])["axis"]["axis_deg"] == a["axis_deg"]
    text = shape.axis_text(a)
    assert "Asse" in text and "fuori tondo" in text


def test_the_sides_of_a_polygonal_tower_stand_upright(tmp_path):
    doc, ch = S.project(tmp_path, S.hex_tower(), "Torre")
    w = walls.new_wall(ch, "Torre", kind="tower")
    for e in faces.polygon_faces(ch, w["id"], Z0 + 2.0, resolution=0.05):
        r = shape.face_shape(ch, w["id"], e["id"])
        assert r["lean_deg"] < 0.1 and abs(r["bulge_max_cm"]) < 0.2 and abs(r["bulge_min_cm"]) < 0.2


def test_two_surveys_of_a_wall(tmp_path):
    doc, ch, w, entry = _wall_face(tmp_path, resolution=0.04)
    other = S.add_chunk(doc, "Muro 2027", S.wall(offset=S.STEP))
    doc.save()
    r = shape.compare_surveys(ch, w["id"], entry["id"], other)
    assert abs(r["median_cm"] - 100 * S.STEP) < 0.3, r
    assert abs(r["p90_cm"] - 100 * S.STEP) < 0.3 and r["cells"] > 5000
    values, *_ = _raster(ch.data_dir / r["raster"])
    assert abs(np.nanmedian(values) - 2.0) < 0.3
    assert r["other"] == other.label
    assert "2,0 cm" in shape.compare_text(r)
    with pytest.raises(walls.NotReady):
        shape.compare_surveys(ch, w["id"], entry["id"], ch)
