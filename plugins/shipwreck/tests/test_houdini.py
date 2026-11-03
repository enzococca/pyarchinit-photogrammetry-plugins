"""The file link with Fröhlich & Nielsen's Clinker Boat Tool (a Houdini Digital Asset): the export folder in the
ship frame turned to Y up, and the HDA's planks read back as timber records with their deviation from the scan."""
from __future__ import annotations

import json
import sys

import numpy as np
import pytest
from hulls import clinker_project, half_breadth, project

from ppg_shipwreck import clinker, houdini, lines, ship, timbers

STATIONS = [round(2.01 + k, 2) for k in range(9)]


def _ready(ch, ends, stations=STATIONS, bow="end"):
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], bow))
    ship.set_stations(ch, [{"name": f"S{i + 1}", "x": x} for i, x in enumerate(stations)])
    lines.build(ch, half_width=3.0)


def _clinker(tmp_path, bow="end"):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends, bow=bow)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    return ch


def _obj(path):
    """Vertices, and per object (``o``) the number of its ``v`` lines and its element lines."""
    v, objects, current = [], {}, None
    for ln in path.read_text(encoding="utf-8").splitlines():
        if ln.startswith("o "):
            current = ln[2:].strip()
            objects[current] = {"v": 0, "l": 0, "f": 0}
        elif ln.startswith("v "):
            v.append([float(a) for a in ln.split()[1:4]])
            if current is not None:
                objects[current]["v"] += 1
        elif ln[:2] in ("l ", "f ") and current is not None:
            objects[current][ln[0]] += 1
    return np.asarray(v), objects


def _strakes_point_outward(path):
    """The Y-up strakes' faces point away from the keel line (the X axis), on both sides."""
    text = path.read_text().splitlines()
    v = np.array([[float(a) for a in ln.split()[1:4]] for ln in text if ln.startswith("v ")])
    f = np.array([[int(a) - 1 for a in ln.split()[1:4]] for ln in text if ln.startswith("f ")])
    p = v[f]
    normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
    centre = p.mean(axis=1)
    centre[:, 0] = 0.0
    return float(((normal * centre).sum(axis=1) > 0).mean())


@pytest.mark.parametrize("sign", [1, -1])
def test_y_up_and_back_are_exact_inverses(sign):
    rng = np.random.default_rng(3)
    xyz = rng.normal(size=(50, 3)) * 10.0
    up = houdini.to_y_up(xyz, sign)
    assert np.allclose(up[:, 0], xyz[:, 0]) and np.allclose(up[:, 1], xyz[:, 2])
    assert np.allclose(up[:, 2], -xyz[:, 1] * sign)
    assert np.array_equal(houdini.from_y_up(up, sign), xyz)
    assert houdini.Y_UP == "X = x, Y = z, Z = -y * sign"


@pytest.mark.parametrize("bow", ["start", "end"])
def test_world_to_y_up_stays_right_handed(bow):
    frame = ship.Frame((500000.0, 4500000.0, 3.0), (500006.0, 4500008.0, 3.2), bow)
    world = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]) + frame.start
    up = houdini.to_y_up(frame.to_ship(world), frame.sign)
    e = up[1:] - up[0]
    assert np.dot(np.cross(e[0], e[1]), e[2]) == pytest.approx(1.0)


def test_the_export_folder(tmp_path):
    ch = _clinker(tmp_path)
    frame = ship.need_frame(ch)
    out = houdini.export(ch, tmp_path / "out")
    folder = tmp_path / "out" / "clinker_boat_tool"
    for name in ("hull.obj", "sections.obj", "stations.json", "frame.json", "lands.obj", "strakes.obj", "README.txt"):
        assert (folder / name).exists(), name
        assert any(p == folder / name for p in out.values()), name
    mesh = lines.ship_mesh(ch, frame)
    v, _objects = _obj(folder / "hull.obj")
    assert len(v) == len(mesh.vertices)
    assert np.abs(houdini.from_y_up(v, frame.sign) - mesh.vertices).max() < 1e-5
    faces = [ln for ln in (folder / "hull.obj").read_text().splitlines() if ln.startswith("f ")]
    assert len(faces) == len(mesh.faces)
    _v, objects = _obj(folder / "sections.obj")
    stations = lines.sections_of(ch, "station")
    assert list(objects) == [s["name"] for s in stations]
    for s in stations:
        assert objects[s["name"]]["v"] == sum(len(p) for p in s["lines"])
        assert objects[s["name"]]["l"] == len(s["lines"])
    d = json.loads((folder / "frame.json").read_text(encoding="utf-8"))
    for key in ("origin_world", "heading", "z0", "x0", "bow", "sign", "y_up", "crs"):
        assert key in d, key
    assert d["y_up"] == houdini.Y_UP and d["sign"] == frame.sign and d["crs"] == ch.crs
    assert d["origin_world"] == pytest.approx(list(frame.start))
    st = json.loads((folder / "stations.json").read_text(encoding="utf-8"))
    assert [s["name"] for s in st["stations"]] == [f"S{i + 1}" for i in range(len(STATIONS))]
    _v, lands = _obj(folder / "lands.obj")
    assert len(lands) == 8  # four land lines a side
    _v, strakes = _obj(folder / "strakes.obj")
    assert sorted(strakes) == sorted(f"fascia_{s}{k}" for s in "BT" for k in range(1, 6))
    assert _strakes_point_outward(folder / "strakes.obj") == 1.0
    assert (folder / "strakes.csv").exists() and (folder / "strakes.xlsx").exists()
    readme = (folder / "README.txt").read_text(encoding="utf-8")
    assert "10.5281/zenodo.23422398" in readme and "10.5281/zenodo.17488549" in readme and "CC BY 4.0" in readme


def test_a_plain_hull_exports_no_lands_nor_strakes(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    out = houdini.export(ch, tmp_path / "out")
    folder = tmp_path / "out" / "clinker_boat_tool"
    assert (folder / "hull.obj").exists() and (folder / "sections.obj").exists()
    assert not (folder / "lands.obj").exists() and not (folder / "strakes.obj").exists()
    assert "lands" not in out and "strakes" not in out
    assert not (folder / "strakes.csv").exists() and not (folder / "strakes.xlsx").exists()


def test_the_export_needs_the_sections(tmp_path):
    _doc, ch, ends = project(tmp_path)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "end"))
    with pytest.raises(ship.NotReady):
        houdini.export(ch, tmp_path / "out")


def test_the_strakes_read_back_as_planks(tmp_path):
    ch = _clinker(tmp_path)
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("point", [[0.0, 0.0]]))
    houdini.export(ch, tmp_path / "out")
    path = tmp_path / "out" / "clinker_boat_tool" / "strakes.obj"
    result = houdini.import_planks(ch, path)
    assert result["planks"] == 10
    planks = [r for r in timbers.timbers_of(ch) if r["kind"] == "plank"]
    assert len(planks) == 10
    assert sorted(r["name"] for r in planks) == sorted(f"fascia_{s}{k}" for s in "BT" for k in range(1, 6))
    frame = ship.need_frame(ch)
    for r in planks:
        assert r["side"] == ("port" if "_B" in r["name"] else "starboard"), r["name"]
        assert r["geometry"]["type"] == "polygon" and "strakes.obj" in r["notes"]
        xy = np.asarray(r["geometry"]["xy"])
        y = frame.to_ship(np.column_stack([xy, np.zeros(len(xy))]))[:, 1]
        assert ((y > -0.01) if r["side"] == "port" else (y < 0.01)).all()
    rows = result["rows"]
    assert len(rows) == 10
    for row in rows:
        assert row["median_mm"] < 10.0 and row["p90_mm"] >= row["median_mm"], row
    assert ch.meta["shipwreck"]["houdini"]["planks"]["strakes.obj"] == rows
    assert result["csv"].exists() and result["csv"].parent == ch.data_dir / "shipwreck" / "clinker"
    assert result["xlsx"].exists()
    again = houdini.import_planks(ch, path)  # the same file again: replaced, never doubled; the user's record stays
    assert again["planks"] == 10 and again["replaced"] == 10
    records = timbers.timbers_of(ch)
    assert len(records) == 11 and sum(r["kind"] == "frame" for r in records) == 1
    planks = [r for r in records if r["kind"] == "plank"]
    assert all(r["status"] == "reconstructed:imported" and not timbers.is_measured(r) for r in planks)
    for r in planks:  # a project of 0.2.0: no status, the import note in either language
        old_record = {k: v for k, v in r.items() if k != "status"}
        assert timbers.status_of(old_record) == "reconstructed:imported"
        assert timbers.status_of({**old_record, "notes": houdini.NOTE[1].format("x.obj")}) == "reconstructed:imported"
    assert timbers.status_title("reconstructed:imported") == "ricostruito: importato dal Clinker Boat Tool"
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    for r in planks:  # an imported plank named like a section strake blocks nothing
        r["name"] = "fascia " + r["name"].split("_")[-1]
    assert clinker.to_catalogue(ch) == 10


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    return path


PLANK = [(2.0, 0.6, -1.0), (6.0, 0.6, -1.0), (6.0, 0.8, -0.8), (2.0, 0.8, -0.8)]


def _y_up_lines(points, sign):
    return [f"v {a:.6f} {b:.6f} {c:.6f}" for a, b, c in houdini.to_y_up(np.asarray(points), sign)]


def test_groups_only_are_planks_too(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    sign = ship.need_frame(ch).sign
    starboard = [(x, -y, z) for x, y, z in PLANK]
    text = ["g plank_1", *_y_up_lines(PLANK, sign), "f 1 2 3", "f 1 3 4",
            "g plank_2", *_y_up_lines(starboard, sign), "f 5 6 7", "f 5 7 8"]
    result = houdini.import_planks(ch, _write(tmp_path / "planks.obj", "\n".join(text) + "\n"), deviation=False)
    assert result["planks"] == 2
    records = {r["name"]: r for r in timbers.timbers_of(ch)}
    assert records["plank_1"]["side"] == "port" and records["plank_2"]["side"] == "starboard"
    assert all(row["median_mm"] is None for row in result["rows"])


def test_a_file_with_neither_objects_nor_groups_is_refused(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    path = _write(tmp_path / "loose.obj", "\n".join(_y_up_lines(PLANK, 1) + ["f 1 2 3"]) + "\n")
    with pytest.raises(ship.NotReady, match="nessun oggetto o gruppo nel file"):
        houdini.import_planks(ch, path)
    assert timbers.timbers_of(ch) == []


def test_the_mirror_frame(tmp_path):
    """With the bow at the start (sign -1) the Y-up file is the same hull: the other bow changes nothing in it, and
    the planks come back on the side they lie on."""
    ch = _clinker(tmp_path, bow="end")
    houdini.export(ch, tmp_path / "a")
    ship.set_frame(ch, ship.need_frame(ch).other_bow())
    assert ship.need_frame(ch).sign == -1
    lines.build(ch, half_width=3.0)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    houdini.export(ch, tmp_path / "b")
    assert _strakes_point_outward(tmp_path / "b" / "clinker_boat_tool" / "strakes.obj") == 1.0
    va, _ = _obj(tmp_path / "a" / "clinker_boat_tool" / "hull.obj")
    vb, _ = _obj(tmp_path / "b" / "clinker_boat_tool" / "hull.obj")
    assert np.abs(va - vb).max() < 1e-5
    result = houdini.import_planks(ch, tmp_path / "b" / "clinker_boat_tool" / "strakes.obj")
    assert result["planks"] == 10
    frame = ship.need_frame(ch)
    for r in timbers.timbers_of(ch):
        xy = np.asarray(r["geometry"]["xy"])
        y = frame.to_ship(np.column_stack([xy, np.zeros(len(xy))]))[:, 1].mean()
        assert r["side"] == ("port" if y > 0 else "starboard")
        assert r["side"] == ("port" if "_B" in r["name"] else "starboard")
    assert all(row["median_mm"] < 10.0 for row in result["rows"])


def test_the_side_from_the_name_and_the_vertices_off_the_model(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    sign = ship.need_frame(ch).sign
    far = [(x, y + 40.0, z) for x, y, z in PLANK]  # off the model: no height to compare with
    text = ["o fascia_T2", *_y_up_lines(PLANK, sign), "f 1 2 3 4", "o lontana", *_y_up_lines(far, sign), "l 5 6 7 8"]
    result = houdini.import_planks(ch, _write(tmp_path / "p.obj", "\n".join(text) + "\n"), side_from="name")
    records = {r["name"]: r for r in timbers.timbers_of(ch)}
    assert records["fascia_T2"]["side"] == "starboard" and records["lontana"]["side"] == "port"
    rows = {r["name"]: r for r in result["rows"]}
    assert rows["lontana"]["missing"] == 4 and rows["lontana"]["median_mm"] is None
    assert rows["fascia_T2"]["missing"] == 0 and rows["fascia_T2"]["median_mm"] is not None


@pytest.mark.parametrize("backend", ["open3d", "trimesh"])
def test_a_plank_off_a_steep_side_reports_its_distance(tmp_path, monkeypatch, backend):
    """The distance to the scan, not a height: a plank 10 mm off the steep side of the hull reports 10 mm, with
    Open3D's raycasting scene and without it (trimesh: Open3D is the program's optional ``mesh`` extra)."""
    if backend == "open3d":
        pytest.importorskip("open3d")
    else:
        monkeypatch.setitem(sys.modules, "open3d", None)  # ``import open3d`` raises ImportError
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    frame = ship.need_frame(ch)
    mesh = lines.ship_mesh(ch, frame)
    normals = np.asarray(mesh.vertex_normals)
    v = mesh.vertices
    inside = np.abs(v[:, 1]) < 0.95 * half_breadth(v[:, 0])  # away from the sheer, the mesh's edge
    steep = np.flatnonzero((np.abs(normals[:, 2]) < 0.65) & (np.abs(v[:, 0] - 6.0) < 2.0) & inside)  # over 49 degrees
    pick = steep[np.linspace(0, len(steep) - 1, 200).astype(int)]
    moved = mesh.vertices[pick] + 0.010 * normals[pick]
    text = ["o lato", *_y_up_lines(moved, frame.sign), "p " + " ".join(str(i + 1) for i in range(len(moved)))]
    result = houdini.import_planks(ch, _write(tmp_path / "steep.obj", "\n".join(text) + "\n"))
    row = result["rows"][0]
    assert row["missing"] == 0
    assert row["median_mm"] == pytest.approx(10.0, abs=0.5) and row["p90_mm"] < 11.0, row
    assert result["backend"] == backend
    assert ship.state(ch)["houdini"]["imports"]["steep.obj"]["backend"] == backend


def test_both_distance_backends_agree(monkeypatch):
    pytest.importorskip("open3d")
    import trimesh

    from hulls import hull_vertices

    v, f = hull_vertices()
    mesh = trimesh.Trimesh(v, f, process=False)
    rng = np.random.default_rng(3)
    points = v[rng.choice(len(v), 300, replace=False)] + rng.normal(0.0, 0.05, (300, 3))
    fast = houdini.Distance(mesh)
    monkeypatch.setitem(sys.modules, "open3d", None)
    slow = houdini.Distance(mesh)
    assert (fast.backend, slow.backend) == ("open3d", "trimesh")
    assert np.abs(fast.to(points) - slow.to(points)).max() < 0.001


def _two_planks(sign):
    starboard = [(x, -y, z) for x, y, z in PLANK]
    return _y_up_lines(PLANK, sign), _y_up_lines(starboard, sign)


def test_one_object_wrapping_a_group_per_plank(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    a, b = _two_planks(ship.need_frame(ch).sign)
    text = ["o Planks", "g fascia_B1", *a, "f 1 2 3 4", "g fascia_T1", *b, "f 5 6 7 8"]
    result = houdini.import_planks(ch, _write(tmp_path / "w.obj", "\n".join(text) + "\n"), deviation=False)
    assert result["planks"] == 2
    assert sorted(r["name"] for r in timbers.timbers_of(ch)) == ["fascia_B1", "fascia_T1"]


def test_several_groups_on_a_line_and_tabs(tmp_path):
    """``g a b``: in both groups; a group holding every element (a catch-all) is not a plank; tabs separate too."""
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    a, b = _two_planks(ship.need_frame(ch).sign)
    text = ["g\tfascia_B1 planks", *[ln.replace(" ", "\t", 1) for ln in a], "f\t1 2 3 4",
            "g fascia_T1 planks", *b, "f 5 6 7 8"]
    result = houdini.import_planks(ch, _write(tmp_path / "m.obj", "\n".join(text) + "\n"), deviation=False)
    assert result["planks"] == 2
    records = {r["name"]: r for r in timbers.timbers_of(ch)}
    assert sorted(records) == ["fascia_B1", "fascia_T1"]
    assert records["fascia_B1"]["side"] == "port" and records["fascia_T1"]["side"] == "starboard"


@pytest.mark.parametrize("bad", ["f 1 x 3", "v 1.0 2.0"])
def test_an_unreadable_line_is_refused_with_its_number(tmp_path, bad):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    a, _b = _two_planks(1)
    text = ["o p", *a, bad]
    with pytest.raises(ship.NotReady, match="Riga 6"):
        houdini.import_planks(ch, _write(tmp_path / "bad.obj", "\n".join(text) + "\n"))


def test_each_file_keeps_its_rows(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    a, b = _two_planks(ship.need_frame(ch).sign)
    houdini.import_planks(ch, _write(tmp_path / "one.obj", "\n".join(["o x", *a, "f 1 2 3 4"]) + "\n"))
    houdini.import_planks(ch, _write(tmp_path / "two.obj", "\n".join(["o y", *b, "f 1 2 3 4"]) + "\n"))
    store = ch.meta["shipwreck"]["houdini"]
    assert sorted(store["planks"]) == ["one.obj", "two.obj"] and sorted(store["imports"]) == ["one.obj", "two.obj"]
    assert store["planks"]["one.obj"][0]["name"] == "x" and len(timbers.timbers_of(ch)) == 2
