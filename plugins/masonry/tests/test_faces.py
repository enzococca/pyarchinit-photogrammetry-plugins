"""Walls and their faces: the records in ``chunk.meta["masonry"]``, faces as the program's elevations on a plane from
two points of the plan, on a cylinder (three points, or the circle of a horizontal slice), one plane per side of a
polygonal tower; their size, resolution and drawings; a face taken out, an elevation removed by the program."""
from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, walls  # noqa: E402

Z0 = S.WORLD[2]


def test_a_wall_and_its_face_on_a_plane(tmp_path):
    doc, ch = S.project(tmp_path, S.wall())
    w = walls.new_wall(ch, "Muro nord")
    assert w["kind"] == "wall" and w["faces"] == [] and w["made"]
    assert walls.new_wall(ch)["name"] == "Muro 1"
    with pytest.raises(ValueError):
        walls.new_wall(ch, kind="castle")
    p1, p2 = S.WORLD[:2] + [0.5, 0.0], S.WORLD[:2] + [5.5, 0.0]
    entry = faces.plane_face(ch, w["id"], p1, p2, side=S.WORLD[:2] + [3.0, -5.0], resolution=0.02)
    assert walls.need(ch, "Muro nord")["faces"] == [entry["id"]]
    assert entry in ch.meta["elevations"] and entry["v_is_height"]
    frame = entry["frame"]
    assert np.allclose(frame["u_axis"], [1, 0, 0], atol=1e-9) and np.allclose(frame["v_axis"], [0, 0, 1])
    info = walls.face_info(ch, entry["id"])
    assert abs(info["width"] - 5.0) < 0.03 and abs(info["height"] - S.WALL_HEIGHT) < 0.03
    assert abs(info["bounds"][1] - Z0) < 0.03 and info["resolution"] == 0.02 and info["surface"] == "plane"
    assert info["drawings"] == {"units": 0, "quote": 0, "points": 0, "stones": 0}
    # drawings kept with the elevation (the program's frames) are counted
    eid = entry["id"]
    ch.meta.setdefault("drawings", {}).setdefault("frames", {})[eid] = {
        "units": [{"wkt": "POLYGON((1 313, 2 313, 2 314, 1 314, 1 313))", "us": "1", "unita_tipo": "USM"}],
        "stones": {eid: ["POLYGON((0 312.6, 0.2 312.6, 0.2 312.8, 0 312.8, 0 312.6))"] * 3}}
    assert walls.face_info(ch, eid)["drawings"] == {"units": 1, "quote": 0, "points": 0, "stones": 3}
    path = walls.write_summary(ch, w["id"])
    data = json.loads(path.read_text())
    assert data["name"] == "Muro nord" and data["faces"][0]["id"] == eid
    assert path.parent == ch.data_dir / "masonry" / w["id"]
    # the two points coincide; a trace far from the model
    with pytest.raises(walls.NotReady):
        faces.plane_face(ch, w["id"], p1, p1)
    with pytest.raises(walls.NotReady):
        faces.plane_face(ch, w["id"], p1 + [0, 20], p2 + [0, 20], resolution=0.05)


def test_faces_of_a_round_tower(tmp_path):
    doc, ch = S.project(tmp_path, S.round_tower(), "Torre")
    w = walls.new_wall(ch, "Torre tonda", kind="tower")
    c = faces.slice_circle(ch, Z0 + 3.0)
    axis = S.tower_axis()
    centre = S.WORLD + axis * (3.0 / axis[2])
    assert np.linalg.norm(np.array(c["centre"]) - centre[:2]) < 0.005
    assert abs(c["radius"] - S.TOWER_RADIUS) < 0.005
    entry = faces.cylinder_face(ch, w["id"], height=Z0 + 3.0, resolution=0.05)
    assert entry["frame"]["kind"] == "cylinder" and not entry["frame"]["inside"]
    info = walls.face_info(ch, entry["id"])
    assert abs(info["width"] - 2 * np.pi * c["radius"]) < 0.2
    assert abs(info["height"] - S.TOWER_HEIGHT) < 0.15  # the rims lean
    angles = np.radians([10, 130, 250])
    pts = centre[:2] + S.TOWER_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])
    inner = faces.cylinder_face(ch, w["id"], points=pts, inside=True, resolution=0.05)
    assert inner["frame"]["inside"] and walls.need(ch, w["id"])["faces"] == [entry["id"], inner["id"]]


def test_one_face_per_side_of_a_polygonal_tower(tmp_path):
    doc, ch = S.project(tmp_path, S.hex_tower(), "Torre")
    w = walls.new_wall(ch, "Torre esagonale", kind="tower")
    poly = faces.slice_polygon(ch, Z0 + 2.0)
    assert len(poly.exterior.coords) - 1 == 6
    entries = faces.polygon_faces(ch, w["id"], Z0 + 2.0, min_side=0.5, resolution=0.05)
    assert len(entries) == 6 and walls.need(ch, w["id"])["faces"] == [e["id"] for e in entries]
    truth = S.hex_normals()
    for e in entries:
        u = np.asarray(e["frame"]["u_axis"])
        normal = np.cross(u, e["frame"]["v_axis"])[:2]  # towards the viewer: outwards
        best = max(float(normal @ n) for n in truth)
        assert best > np.cos(np.radians(0.5)), normal
        info = walls.face_info(ch, e["id"])
        assert abs(info["width"] - S.HEX_RADIUS) < 0.05 and abs(info["height"] - S.HEX_HEIGHT) < 0.1


def test_faces_taken_out_and_elevations_removed(tmp_path):
    from pyarchinit_photogrammetry.engine.elevation import remove_elevation

    doc, ch = S.project(tmp_path, S.wall())
    w = walls.new_wall(ch, "Muro")
    a = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [3, 0], resolution=0.05)
    b = faces.plane_face(ch, w["id"], S.WORLD[:2] + [3, 0], S.WORLD[:2] + [6, 0], resolution=0.05)
    walls.face_folder(ch, w["id"], a["id"]).mkdir(parents=True)
    assert walls.remove_face(ch, w["id"], a["id"])
    assert not walls.face_folder(ch, w["id"], a["id"]).exists()
    assert walls.elevation(ch, a["id"]) is not None  # the program's elevation stays
    remove_elevation(ch, b["id"])  # the user removes an elevation in the program
    assert walls.prune(ch) == [b["id"]] and walls.need(ch, w["id"])["faces"] == []
    c = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [3, 0], resolution=0.05)
    assert walls.remove_wall(ch, w["id"], elevations_too=True)
    assert walls.walls(ch) == [] and walls.elevation(ch, c["id"]) is None
    assert not walls.wall_folder(ch, w["id"]).exists()


def test_an_elevation_of_the_program_becomes_a_face_as_it_is(tmp_path):
    """A prospect the program made itself (the Elevation button, ``ppg elevation``) is adopted as a face: nothing is
    rebuilt, its entry is read like those of the faces the plugin makes; one wall at a time; sections and elevations
    without an orthophoto are refused."""
    from pyarchinit_photogrammetry.engine import elevation as E

    doc, ch = S.project(tmp_path, S.wall())
    p1, p2 = S.WORLD[:2] + [0.5, 0.0], S.WORLD[:2] + [5.5, 0.0]
    own = E.build_elevation(ch, E.with_heights(E.plane_from_line(ch, p1, p2, side=S.WORLD[:2] + [3.0, -5.0])),
                            resolution=0.02, name="Parete del programma")
    assert walls.walls(ch) == [] and [e["id"] for e in walls.free_elevations(ch)] == [own["id"]]
    w = walls.new_wall(ch, "Muro nord")
    entry = walls.adopt_face(ch, w["id"], "Parete del programma")
    assert entry is own and walls.need(ch, w["id"])["faces"] == [own["id"]]
    assert walls.free_elevations(ch) == [] and walls.wall_of_face(ch, own["id"])["id"] == w["id"]
    info = walls.face_info(ch, own["id"])
    assert info["surface"] == "plane" and 5.5 < info["width"] < 6.5 and info["resolution"] == 0.02  # the whole wall
    assert json.loads(walls.write_summary(ch, w["id"]).read_text())["faces"][0]["id"] == own["id"]
    # the same elevation again is no change; another wall cannot take it
    assert walls.adopt_face(ch, w["id"], own["id"]) is own and walls.need(ch, w["id"])["faces"] == [own["id"]]
    other = walls.new_wall(ch, "Muro sud")
    with pytest.raises(walls.NotReady, match="Muro nord"):
        walls.adopt_face(ch, other["id"], own["id"])
    with pytest.raises(walls.NotReady, match="Nessun prospetto"):
        walls.adopt_face(ch, other["id"], "non esiste")
    # a section (no plane or cylinder frame) and an elevation without its orthophoto are not faces
    ch.meta["elevations"].append({"id": "sez1", "name": "Sezione", "kind": "section", "frame": {"kind": "section"},
                                  "orthophoto": own["orthophoto"]})
    ch.meta["elevations"].append({"id": "gone", "name": "Senza file", "frame": {"kind": "plane"},
                                  "orthophoto": "elevations/gone/elevation.tif"})
    assert walls.free_elevations(ch) == []
    with pytest.raises(walls.NotReady, match="piano"):
        walls.adopt_face(ch, other["id"], "sez1")
    with pytest.raises(walls.NotReady, match="ortofoto"):
        walls.adopt_face(ch, other["id"], "gone")
