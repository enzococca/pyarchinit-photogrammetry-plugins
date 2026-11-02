"""The cases of the review of 0.1.0: stones drawn twice, a structure behind the wall, the thickness of a tower in IFC,
a trace on the wall's axis, an opening blocked in a wall, the wall's area of a second run; sloping courses, long
ashlars, the share of a contact, the IFC exporter, a fragment among the stones, the plate on A3; and the minor ones."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the faces are elevations of the program")

import masonry_samples as S  # noqa: E402
from ppg_masonry import faces, outputs, shape, stones, usm, walls  # noqa: E402

Z0 = S.WORLD[2]
VIEWER = S.WORLD[:2] + [3.0, -5.0]
HAS_IFC = importlib.util.find_spec("ifcopenshell") is not None


def _face(tmp_path, mesh=None, resolution=0.02, **kw):
    doc, ch = S.project(tmp_path, mesh or S.wall(lean=False, bulge=False))
    doc.meta["sito"] = "Saggio"
    w = walls.new_wall(ch, "Muro")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [S.STONE_WALL, 0.0], side=VIEWER,
                             resolution=resolution, **kw)
    return doc, ch, w, entry


def _two(a, b):
    return np.vstack([a[0], b[0]]), np.vstack([a[1], b[1] + len(a[0])])


# ------------------------------------------------------------------ Important
def test_stones_drawn_twice_count_once(tmp_path):
    from shapely import affinity
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    polys, _truth = S.stone_pattern(Z0)
    region = box(0.0, Z0, S.STONE_WALL, Z0 + S.COURSES * S.PITCH)
    truth = 1.0 - sum(p.area for p in polys) / region.area
    again = [affinity.translate(p, 0.005, 0.0) for p in polys]  # a second run of SAM over the same face
    S.put_drawings(ch, entry["id"], polys + again)
    r = stones.measure_face(ch, w["id"], entry["id"], region=region)
    assert r["stones"] == len(polys) and len(r["merged"]) == len(polys)
    assert abs(r["mortar"] - truth) < 0.01 and 0.0 <= r["mortar"] <= 1.0
    assert {row["n"] for row in r["rows"]} == set(range(len(polys) + 1, 2 * len(polys) + 1))  # the later kept
    assert "disegnate due volte" in stones.stones_text(walls.results(ch, w["id"], entry["id"])["stones"])
    # one stone corrected by hand, overlapping its first outline by less than half: the union counts once
    p0 = polys[0]
    shifted = affinity.translate(p0, 0.6 * (p0.bounds[2] - p0.bounds[0]), 0.0)
    S.put_drawings(ch, entry["id"], polys + [shifted])
    r = stones.measure_face(ch, w["id"], entry["id"], region=region)
    union_truth = 1.0 - (sum(p.area for p in polys) + shifted.difference(polys[0]).area
                         - sum(shifted.difference(polys[0]).intersection(q).area for q in polys[1:])) / region.area
    assert abs(r["mortar"] - union_truth) < 0.005


def test_a_structure_behind_the_wall_is_left_out(tmp_path):
    front = S._grid(lambda s, t: S.WORLD + np.column_stack([s * 6.0, np.zeros_like(s), t * 2.0]), 241, 81)
    back = S._grid(lambda s, t: S.WORLD + np.column_stack([s * 6.0, np.full_like(s, 0.25), t * 3.0]), 241, 121)
    doc, ch, w, entry = _face(tmp_path, _two(front, back), resolution=0.03)
    r = shape.face_shape(ch, w["id"], entry["id"])
    assert r["lean_deg"] < 0.2, r["lean_deg"]
    b = r["behind"]
    assert abs(b["gap"] - 0.25) < 0.03 and 0.2 < b["share"] < 0.45
    assert abs(r["bulge_max_cm"]) < 0.3 and abs(r["bulge_min_cm"]) < 0.3
    assert "seconda superficie 25 cm più indietro" in shape.summary_text(r)


@pytest.mark.skipif(not HAS_IFC, reason="the program's bim extra (ifcopenshell)")
def test_the_thickness_of_a_tower_is_never_measured_across_it(tmp_path):
    import ifcopenshell

    from ppg_masonry import ifc

    doc, ch = S.project(tmp_path, S.hex_tower(), "Torre")
    w = walls.new_wall(ch, "Torre", kind="tower")
    faces.polygon_faces(ch, w["id"], Z0 + 2.0, resolution=0.05)
    with pytest.raises(walls.NotReady, match="spessore") as exc:
        ifc.wall_solids(ch, w["id"])
    assert "Torre - lato 1" in str(exc.value)
    r = ifc.export_wall(ch, w["id"], tmp_path / "t.ifc", thickness=0.8)
    assert r["walls"] == 6 and all(s_.startswith("dato") for s_ in r["sources"])
    model = ifcopenshell.open(r["path"])
    assert len(model.by_type("IfcWall")) == 6
    import ifcopenshell.util.element

    pset = ifcopenshell.util.element.get_psets(model.by_type("IfcWall")[0])["Pset_PyArchInit_Masonry"]
    assert pset["Spessore"] == 0.8 and pset["OrigineSpessore"].startswith("dato")
    # a building's two facades 8 m apart are not the two sides of one wall
    front = S.wall(lean=False, bulge=False)
    back = S.wall(base=S.WORLD + [0.0, 8.0, 0.0], lean=False, bulge=False)
    doc, ch = S.project(tmp_path / "b", _two(front, back))
    w = walls.new_wall(ch, "Edificio", kind="building")
    faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [6, 0], side=S.WORLD[:2] + [3, -5], resolution=0.05)
    faces.plane_face(ch, w["id"], S.WORLD[:2] + [0, 8], S.WORLD[:2] + [6, 8], side=S.WORLD[:2] + [3, 13],
                     resolution=0.05)
    with pytest.raises(walls.NotReady, match="spessore"):
        ifc.wall_solids(ch, w["id"])


def test_sections_on_the_face_when_the_trace_is_on_the_wall_axis(tmp_path):
    front = S.wall(bulge=False)  # leaning 2 degrees towards the viewer
    back = S.wall(base=S.WORLD + [0.0, 0.6, 0.0], lean=False, bulge=False)
    doc, ch = S.project(tmp_path, _two(front, back))
    w = walls.new_wall(ch, "Muro")
    entry = faces.plane_face(ch, w["id"], S.WORLD[:2] + [0, 0.35], S.WORLD[:2] + [6.0, 0.35], side=VIEWER,
                             resolution=0.04, front=0.6, back=0.5)
    r = shape.sections(ch, w["id"], entry["id"], heights=[Z0 + 0.5, Z0 + 2.5], stations=[1.0, 5.0])
    expected = 2.0 * np.tan(np.radians(S.LEAN_DEG))
    for st in r["stations"]:
        assert abs(st["offsets"][1]["offset"] - expected) < 0.005, st  # the front leans; the back is plumb
        assert st["offsets"][0]["d"] > 0.3  # the front face, 35 cm in front of the trace
    assert r["warnings"] and "non è sulla faccia" in r["warnings"][0]
    assert "non è sulla faccia" in shape.sections_text(r)


def test_an_opening_blocked_in_a_wall_fills_it(tmp_path):
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    hole = box(2.5, Z0 + 0.5, 3.5, Z0 + 1.4)
    wall = box(0.0, Z0, S.STONE_WALL, Z0 + S.COURSES * S.PITCH).difference(hole)
    S.put_drawings(ch, entry["id"], S.stone_pattern(Z0, hole=hole)[0], {"10": wall, "12": hole})
    stones.measure_face(ch, w["id"], entry["id"])
    found = usm.propose(ch, w["id"], entry["id"])
    assert [(p["unit"], p["relation"], p["other"]) for p in found] == [("USM 12", "Riempie", "USM 10")]
    assert "apertura o una lacuna tamponata" in found[0]["evidence"]
    # the wall drawn over the whole face, the blocked opening on top of it
    S.put_drawings(ch, entry["id"], S.stone_pattern(Z0, hole=hole)[0],
                   {"10": box(0.0, Z0, S.STONE_WALL, Z0 + S.COURSES * S.PITCH), "12": hole})
    found = usm.propose(ch, w["id"], entry["id"])
    assert [(p["unit"], p["relation"], p["other"]) for p in found] == [("USM 12", "Riempie", "USM 10")]
    assert usm.INVERSE["Riempie"] == "Riempito da"


def test_the_wall_area_follows_its_source_at_every_run(tmp_path):
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    S.put_drawings(ch, entry["id"], S.stone_pattern(Z0)[0])
    first = stones.measure_face(ch, w["id"], entry["id"])
    second = stones.measure_face(ch, w["id"], entry["id"])
    assert first["region_kind"] == second["region_kind"] == "bounds" and "region" not in second
    assert "limiti della faccia" in stones.stones_text(walls.results(ch, w["id"], entry["id"])["stones"])
    assert stones.drawn_region(ch, w["id"], entry["id"]) is None
    stones.set_region(ch, w["id"], entry["id"], box(0.0, Z0, 6.0, Z0 + 1.75))
    third = stones.measure_face(ch, w["id"], entry["id"])
    assert third["region_kind"] == "drawn" and abs(third["wall_area"] - 6.0 * 1.75) < 1e-9
    assert stones.drawn_region(ch, w["id"], entry["id"]).area == pytest.approx(6.0 * 1.75)


# ------------------------------------------------------------------ Medium
def test_sloping_courses_are_courses(tmp_path):
    from shapely import affinity

    k = np.tan(np.radians(5.0))
    polys = [affinity.translate(p, 0.0, k * p.centroid.x) for p in S.stone_pattern(0.0)[0][:]]
    rows = [stones.measure(p, i + 1) for i, p in enumerate(polys)]
    courses = [c for c in stones.find_courses(rows) if c["counts"]]
    assert len(courses) == S.COURSES
    for c in courses:
        assert abs(c["slope_deg"] - 5.0) < 0.7 and not c["horizontal"]
    # one course at 4 degrees among horizontal ones keeps its stones
    polys = S.stone_pattern(0.0)[0]
    rows = []
    for i, p in enumerate(polys):
        if abs(p.centroid.y - 2.5 * S.PITCH) < 0.1:
            p = affinity.translate(p, 0.0, np.tan(np.radians(4.0)) * (p.centroid.x - 3.0))
        rows.append(stones.measure(p, i + 1))
    courses = [c for c in stones.find_courses(rows) if c["counts"]]
    assert len(courses) == S.COURSES
    assert sum(1 for c in courses if c["horizontal"]) == S.COURSES - 1


def test_long_ashlars_are_squared_and_elongated():
    from shapely.geometry import box

    polys = [box(x, y, x + 0.9, y + 0.4) for y in (0.0, 0.42, 0.84) for x in np.arange(0, 5.5, 0.92)]
    rows = [stones.measure(p, i + 1) for i, p in enumerate(polys)]
    typ = stones.typology(rows, stones.find_courses(rows))
    assert typ["squared"] and typ["elongated"]
    assert "a conci squadrati allungati" in typ["line"]


def test_the_share_of_a_contact_is_not_inflated_by_the_tolerance():
    from shapely.geometry import box

    b = box(0.0, 0.0, 0.38, 1.0)
    a = box(0.0, 1.0, 2.0, 2.0)  # 2 m wide, touching B along 0.38 m: 19 %
    assert usm._rests_on(a, b, 0.06) is None
    b = box(0.0, 0.0, 0.42, 1.0)
    share, _h = usm._rests_on(a, b, 0.06)
    assert abs(share - 0.21) < 1e-6


def test_the_ifc_exporter_never_writes_a_thickness_not_given(tmp_path):
    pytest.importorskip("ifcopenshell")
    from ppg_masonry import MasonryPlugin
    from pyarchinit_photogrammetry.plugins import Context

    doc, ch, w, entry = _face(tmp_path)
    plugin = MasonryPlugin()
    plugin.activate(Context("masonry", tmp_path / "plugin"))
    with pytest.raises(walls.NotReady, match="IFC del muro"):
        plugin._export_ifc(ch, tmp_path / "w.ifc")
    assert not (tmp_path / "w.ifc").exists()
    empty = walls.new_wall(ch, "Vuoto")

    class Pane:  # the pane with the wall without faces chosen
        def wall_id(self):
            return empty["id"]

        def face_id(self):
            return None

    plugin.pane = Pane()
    with pytest.raises(walls.NotReady, match="fai prima una faccia"):
        plugin._export_plate(ch, tmp_path / "t.pdf")


def test_a_fragment_among_the_stones_is_left_out_and_named(tmp_path):
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    polys = S.stone_pattern(Z0)[0]
    S.put_drawings(ch, entry["id"], polys + [box(3.0, Z0 + 0.01, 3.03, Z0 + 0.03)])
    r = stones.measure_face(ch, w["id"], entry["id"])
    assert r["stones"] == len(polys) and [x["n"] for x in r["too_small"]] == [len(polys) + 1]
    assert "più piccole di quattro pixel" in " ".join(r["warnings"])


def test_the_plate_fits_an_a3_sheet(tmp_path):
    for width, height in ((7.6, 4.2), (6.0, 3.0), (30.0, 3.0), (4.0, 9.0)):
        n = outputs.choose_scale(width, height)
        page_w = max(width * 1000 / n, 200.0) + 2 * outputs.MARGIN_MM
        page_h = height * 1000 / n + outputs.TITLE_MM + 2 * outputs.MARGIN_MM
        assert page_w <= 420.0 and page_h <= 297.0, (width, height, n)


# ------------------------------------------------------------------ Minor
def test_minor_ones(tmp_path):
    from shapely.geometry import MultiPolygon, Polygon, box

    from ppg_masonry.texts import azimuth

    assert azimuth(359.6) == "0°" and azimuth(180.2) == "180°"
    # spread 0.2 exactly: not regular (the rule is under 20 %)
    courses = [{"counts": True, "horizontal": True, "stones": 3, "thickness": x} for x in (0.40, 0.48, 0.32, 0.48,
                                                                                            0.32)]
    rows = [{"elongation": 1.5, "rectangularity": 0.9}] * 15
    assert stones.typology(rows, courses)["regular"] is False
    # a stone drawn in two parts: its largest part, its number in the program's table
    doc, ch, w, entry = _face(tmp_path)
    polys = S.stone_pattern(Z0)[0][:6]
    two = MultiPolygon([polys[2], box(5.9, Z0 + 1.7, 5.92, Z0 + 1.72)])
    S.put_drawings(ch, entry["id"], polys[:2] + [two] + polys[3:])
    items, aside = stones.stone_items(ch, entry["id"])
    assert [n for n, _p in items] == [1, 2, 3, 4, 5, 6] and aside["multipart"] == [3]
    assert items[2][1].equals(polys[2])
    # the ellipse of a stone with a hole
    ring = Polygon(box(0, 0, 1, 1).exterior, [box(0.2, 0.2, 0.8, 0.8).exterior.coords[::-1]])
    major, minor, _ang = stones.ellipse(ring)
    assert abs(major - np.sqrt(1.36)) < 1e-6 and abs(minor - np.sqrt(1.36)) < 1e-6
    # the plate's words for a plumb wall and one leaning back
    ctx = {"results": {"shape": {"surface": "plane", "lean_deg": 0.01, "lean_cm_per_m": 0.0, "lean_toward": "in"}},
           "units": []}
    assert outputs.measures_lines(ctx)[0].startswith("A piombo")
    ctx["results"]["shape"].update(lean_deg=1.0, lean_cm_per_m=1.7)
    assert outputs.measures_lines(ctx)[0].startswith("Fuori piombo all'indietro 1,0°")


def test_a_face_refused_leaves_no_empty_wall(tmp_path):
    from ppg_masonry import MasonryPlugin

    doc, ch = S.project(tmp_path, S.wall())
    x0, y0 = S.WORLD[:2]
    code = MasonryPlugin().command("face", str(doc.path), "Nuovo", "--line", f"{x0},{y0 + 30},{x0 + 6},{y0 + 30}")
    assert code == 1
    import pyarchinit_photogrammetry as ppg

    assert walls.walls(ppg.Document.open(doc.path).chunk) == []


# ------------------------------------------------------------------ round 2
def test_an_obstacle_in_front_is_not_the_face(tmp_path):
    for lean in (0.0, S.LEAN_DEG):
        k = np.tan(np.radians(lean))
        wall = S._grid(lambda s, t: S.WORLD + np.column_stack([s * 6.0, -t * 3.0 * k, t * 3.0]), 241, 121)
        board = S._grid(lambda s, t: S.WORLD + np.column_stack([1.0 + s * 1.8, np.full_like(s, -0.2),
                                                                0.3 + t * 1.8]), 73, 73)  # 18 % of the face
        doc, ch, w, entry = _face(tmp_path / f"l{lean}", _two(wall, board), resolution=0.03)
        r = shape.face_shape(ch, w["id"], entry["id"])
        assert abs(r["lean_deg"] - lean) < 0.1, (lean, r["lean_deg"])
        b = r["behind"]
        assert b["side"] == "front" and 0.12 < b["share"] < 0.25 and not b["comparable"]
        assert "davanti alla faccia" in shape.summary_text(r) and "più avanti" in shape.summary_text(r)
        assert abs(r["bulge_max_cm"]) < 0.5 and abs(r["bulge_min_cm"]) < 0.5  # the bulge is the wall's


def test_a_missing_stone_does_not_split_a_course():
    polys, truths = S.stone_pattern(0.0)
    heights = sorted({round(c, 4) for _w, _h, c in truths})

    def course_of(i):
        return min(range(len(heights)), key=lambda k: abs(truths[i][2] - heights[k]))

    def middle(k):  # the stones of course k ordered along the face, the one in the middle
        idx = sorted((i for i in range(len(polys)) if course_of(i) == k), key=lambda i: polys[i].centroid.x)
        return idx

    gone = {middle(0)[len(middle(0)) // 2], middle(2)[len(middle(2)) // 2]}
    rows = [stones.measure(p, i + 1) for i, p in enumerate(polys) if i not in gone]
    courses = [c for c in stones.find_courses(rows) if c["counts"]]
    assert len(courses) == S.COURSES and np.allclose([c["height"] for c in courses], heights, atol=0.002)
    two = middle(1)
    gone = {two[3], two[6]}
    rows = [stones.measure(p, i + 1) for i, p in enumerate(polys) if i not in gone]
    assert len([c for c in stones.find_courses(rows) if c["counts"]]) == S.COURSES


@pytest.mark.skipif(not HAS_IFC, reason="the program's bim extra (ifcopenshell)")
def test_a_small_tower_is_never_paired_across_it(tmp_path, monkeypatch):
    from ppg_masonry import ifc

    monkeypatch.setattr(S, "HEX_RADIUS", 1.2)
    doc, ch = S.project(tmp_path, S.hex_tower(), "Torre")
    w = walls.new_wall(ch, "Pilastro", kind="wall")  # even as a wall: the faces of one slice are never paired
    entries = faces.polygon_faces(ch, w["id"], Z0 + 2.0, min_side=0.3, resolution=0.05)
    assert len(entries) == 6
    r = ifc.export_wall(ch, w["id"], tmp_path / "t.ifc", thickness=0.5)
    assert r["walls"] == 6 and all(x.startswith("dato") for x in r["sources"])
    assert all(abs(s_["thickness"] - 0.5) < 1e-9 for s_ in ifc.wall_solids(ch, w["id"], 0.5))
    # two faces of a building with another wall between them are not one wall
    front = S.wall(lean=False, bulge=False)
    middle_wall = S.wall(base=S.WORLD + [0.0, 1.2, 0.0], lean=False, bulge=False)
    back = S.wall(base=S.WORLD + [0.0, 2.5, 0.0], lean=False, bulge=False)
    mesh = _two(_two(front, middle_wall), back)
    doc, ch = S.project(tmp_path / "b", mesh)
    w = walls.new_wall(ch, "Vano", kind="building")
    faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [6, 0], side=S.WORLD[:2] + [3, -5], resolution=0.05,
                     back=0.2, front=0.2)
    faces.plane_face(ch, w["id"], S.WORLD[:2] + [0, 2.5], S.WORLD[:2] + [6, 2.5], side=S.WORLD[:2] + [3, 7],
                     resolution=0.05, back=0.2, front=0.2)
    with pytest.raises(walls.NotReady, match="altro fra loro"):
        ifc.wall_solids(ch, w["id"])
    # a wall 0.6 m thick, the slice crossing only at its faces: paired
    back = S.wall(base=S.WORLD + [0.0, 0.6, 0.0], lean=False, bulge=False)
    doc, ch = S.project(tmp_path / "c", _two(front, back))
    w = walls.new_wall(ch, "Muro", kind="wall")
    faces.plane_face(ch, w["id"], S.WORLD[:2], S.WORLD[:2] + [6, 0], side=S.WORLD[:2] + [3, -5], resolution=0.05,
                     back=0.2, front=0.2)
    faces.plane_face(ch, w["id"], S.WORLD[:2] + [0, 0.6], S.WORLD[:2] + [6, 0.6], side=S.WORLD[:2] + [3, 5],
                     resolution=0.05, back=0.2, front=0.2)
    solids = ifc.wall_solids(ch, w["id"])
    assert len(solids) == 1 and abs(solids[0]["thickness"] - 0.6) < 0.005
    assert solids[0]["source"].startswith("misurato fra le facce")


def test_a_stone_drawn_inside_another_is_left_out(tmp_path):
    from shapely.geometry import box

    doc, ch, w, entry = _face(tmp_path)
    polys = S.stone_pattern(Z0)[0]
    inner = polys[5].centroid.buffer(0.05)
    S.put_drawings(ch, entry["id"], polys + [inner])
    r = stones.measure_face(ch, w["id"], entry["id"], region=box(0.0, Z0, 6.0, Z0 + 1.75))
    assert r["stones"] == len(polys) and r["inside"] == [len(polys) + 1]
    assert "dentro un'altra pietra" in " ".join(r["warnings"])
