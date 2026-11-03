"""The lands of a clinker hull found on the sections and joined into land lines: within 5 mm of arc and 2 mm
of step of the hull's truth; a smooth hull gives none and says so; a cut side gives none there and the
lines skip it; a stealer is flagged and the other lands are matched through it by arc position."""
from __future__ import annotations

import numpy as np
import pytest
from hulls import clinker_project, clinker_truth, project

from ppg_shipwreck import clinker, lines, ship

STATIONS = [round(2.01 + k, 2) for k in range(9)]  # 2.01 ... 10.01: off the rows of the mesh


def _ready(ch, ends, stations=STATIONS):
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "end"))  # x = the hull's x, port = the hull's y > 0
    ship.set_stations(ch, [{"name": f"S{i + 1}", "x": x} for i, x in enumerate(stations)])
    lines.build(ch, half_width=3.0)


def _arc_of(points, y, z):
    """Arc along a profile up to its point nearest (y, z)."""
    p, s = points
    return s[np.argmin(np.hypot(p[:, 0] - y, p[:, 1] - z))]


def test_the_lands_of_a_clinker_hull(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    result = clinker.detect_lands(ch)
    assert result["mode"] == {"port": 4, "starboard": 4}
    assert result["uncertain"] == {"port": [], "starboard": []} and not result["skipped"]
    errors, steps = [], []
    for section in lines.sections_of(ch, "station"):
        truth = clinker_truth(section["value"])
        for side in ("port", "starboard"):
            found = sorted((d for d in result["lands"] if d["station"] == section["name"] and d["side"] == side),
                           key=lambda d: d["s"])
            assert len(found) == 4, (section["name"], side, [d["s"] for d in found])
            errors += [d["s"] - t for d, t in zip(found, truth)]
            steps += [d["step_mm"] for d in found]
            assert all((d["y"] > 0) == (side == "port") for d in found)
    assert np.abs(errors).max() < 0.005, np.abs(errors).max()
    assert np.abs(np.asarray(steps) - 15.0).max() < 2.0, (min(steps), max(steps))
    meta = ch.meta["shipwreck"]["clinker"]
    assert meta["params"] == clinker.DEFAULTS and meta["lands"] == result["lands"]
    # the land lines: 4 per side, ordered by x, through the lands of every station, in ship coordinates
    found = clinker.match_lines(ch)
    assert sorted((d["side"], d["k"]) for d in found) == [(s, k) for s in ("port", "starboard") for k in (1, 2, 3, 4)]
    for line in found:
        p = np.asarray(line["points"])
        assert len(p) == len(STATIONS) and (np.diff(p[:, 0]) > 0).all()
        assert ((p[:, 1] > 0) == (line["side"] == "port")).all()
        assert not line["uncertain_stations"] and not line["unmatched"]
    assert ch.meta["shipwreck"]["clinker"]["lines"] == found
    text = clinker.summary(ch)
    assert "5 fasce" in text and "incert" not in text
    assert not clinker.is_stale(ch)


def test_a_smooth_hull_gives_no_lands_and_says_so(tmp_path):
    _doc, ch, ends = project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    result = clinker.detect_lands(ch)
    assert result["lands"] == [] and result["mode"] == {"port": 0, "starboard": 0}
    assert clinker.match_lines(ch) == []
    assert "nessun gradino di fasciame trovato" in clinker.summary(ch).lower()


def test_a_cut_side_gives_no_lands_there_and_the_lines_skip_it(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, hole=(5.9, 6.1, 0.5, 0.9))  # port, through the station S5
    _ready(ch, ends)
    result = clinker.detect_lands(ch)
    assert [(d["station"], d["side"]) for d in result["skipped"]] == [("S5", "port")]
    assert result["skipped"][0]["reason"]
    assert not [d for d in result["lands"] if d["station"] == "S5" and d["side"] == "port"]
    assert len([d for d in result["lands"] if d["station"] == "S5" and d["side"] == "starboard"]) == 4
    assert result["mode"] == {"port": 4, "starboard": 4} and result["uncertain"] == {"port": [], "starboard": []}
    for line in clinker.match_lines(ch):
        x = np.asarray(line["points"])[:, 0]
        expected = [v for v in STATIONS if not (line["side"] == "port" and v == 6.01)]
        assert x.tolist() == pytest.approx(expected)
    assert "S5" in clinker.summary(ch)
    # the cut profile is refused by itself
    section = next(s for s in lines.sections_of(ch, "station") if s["name"] == "S5")
    assert clinker.profile(section, "port") is None and clinker.profile(section, "starboard") is not None


def test_a_stealer_is_flagged_and_the_lines_go_through_it(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, stealer=(4.5, 7.5))
    _ready(ch, ends)
    result = clinker.detect_lands(ch)
    assert result["mode"] == {"port": 4, "starboard": 4}
    assert result["uncertain"] == {"port": ["S4", "S5", "S6"], "starboard": ["S4", "S5", "S6"]}
    found = clinker.match_lines(ch)
    assert len(found) == 8
    sections = {s["name"]: s for s in lines.sections_of(ch, "station")}
    for line in found:
        assert line["uncertain_stations"] == ["S4", "S5", "S6"] and not line["unmatched"]
        assert len(line["points"]) == len(STATIONS)
        for name in ("S4", "S5", "S6"):  # the line's point there is the k-th land of the truth
            section = sections[name]
            point = next(p for p in line["points"] if p[0] == pytest.approx(section["value"]))
            arc = _arc_of(clinker.profile(section, line["side"]), point[1], point[2])
            truth = clinker_truth(section["value"], stealer=(4.5, 7.5))
            assert arc == pytest.approx(truth[line["k"] - 1], abs=0.005)
    extra = ch.meta["shipwreck"]["clinker"]["unmatched"]
    assert sorted((d["station"], d["side"]) for d in extra) == sorted(
        (n, s) for n in ("S4", "S5", "S6") for s in ("port", "starboard"))
    for d in extra:
        assert d["s"] == pytest.approx(clinker_truth(sections[d["station"]]["value"], stealer=(4.5, 7.5))[4], abs=0.005)
    text = clinker.summary(ch)
    assert "S4" in text and "S6" in text


def test_lands_are_stale_when_the_sections_change(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    clinker.detect_lands(ch)
    assert not clinker.is_stale(ch)
    lines.build(ch, half_width=3.0)  # the same sections again: the lands still hold
    assert not clinker.is_stale(ch)
    ship.set_stations(ch, [{"name": "S1", "x": 2.01}, {"name": "S2", "x": 3.51}])
    lines.build(ch, half_width=3.0)
    assert clinker.is_stale(ch)
    assert "rileva" in clinker.summary(ch).lower()


def test_the_land_lines_written_as_dxf_and_geopackage(tmp_path):
    import ezdxf
    import shapely

    from pyarchinit_photogrammetry.io import gpkg

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=123.0)
    _ready(ch, ends, STATIONS[:4])
    clinker.detect_lands(ch)
    files = clinker.write_lines(ch)
    assert files["dxf"].parent == ch.data_dir / "shipwreck" / "clinker"
    doc = ezdxf.readfile(files["dxf"])
    polylines = list(doc.modelspace().query("POLYLINE"))
    assert len(polylines) == 8
    assert {p.dxf.layer for p in polylines} == {"FASCE_BABORDO", "FASCE_TRIBORDO"}
    first = np.array([v.dxf.location for v in polylines[0].vertices])
    assert first.shape == (4, 3) and first[:, 2].min() < -0.5  # ship coordinates, z below the sheer
    rows = gpkg.read_table(files["gpkg"], "fasce_linee")
    assert len(rows) == 8
    frame = ship.need_frame(ch)
    lines_ship = {(d["side"], d["k"]): np.asarray(d["points"]) for d in ch.meta["shipwreck"]["clinker"]["lines"]}
    for geom, values in rows:
        assert geom.has_z
        xyz = np.asarray(shapely.get_coordinates(geom, include_z=True))
        assert np.abs(xyz - frame.to_world(lines_ship[(values["side"], values["k"])])).max() < 1e-6
        assert values["uncertain"] == 0
    con = gpkg.connect(files["gpkg"])
    try:
        srs, has_z = con.execute("SELECT srs_id, z FROM gpkg_geometry_columns WHERE table_name = 'fasce_linee'").fetchone()
    finally:
        con.close()
    assert srs == 32633 and has_z == 1


# ------------------------------------------------------------------------- fix round 1
def test_an_eroded_station_keeps_its_lands_numbered_from_the_keel(tmp_path):
    """A station with fewer lands (the sheer eroded on port): its lands are the garboard's and the next, by
    arc from the keel, not by the fraction of a shorter profile; the upper lines miss it, nothing jumps."""
    _doc, ch, ends = clinker_project(tmp_path, hole=(4.95, 5.07, 0.42, 3.0))  # port, above y = 0.42 at S4
    _ready(ch, ends)
    result = clinker.detect_lands(ch)
    assert result["uncertain"] == {"port": ["S4"], "starboard": []} and not result["skipped"]
    section = next(s for s in lines.sections_of(ch, "station") if s["name"] == "S4")
    truth = clinker_truth(section["value"])
    port = {d["k"]: d for d in clinker.match_lines(ch) if d["side"] == "port"}
    for k in (1, 2):
        point = next(p for p in port[k]["points"] if p[0] == pytest.approx(section["value"]))
        assert _arc_of(clinker.profile(section, "port"), point[1], point[2]) == pytest.approx(truth[k - 1], abs=0.005)
        assert not port[k]["unmatched"]
    for k in (3, 4):
        assert port[k]["unmatched"] == ["S4"] and len(port[k]["points"]) == len(STATIONS) - 1
    assert not ch.meta["shipwreck"]["clinker"]["unmatched"]
    assert "S4" in clinker.summary(ch)


def _one_station(tmp_path, polylines):
    """A project whose one station's section is replaced by ``polylines`` (the sections file kept valid)."""
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, [6.01])
    data = lines.load(ch)
    data["sections"][0]["lines"] = [np.round(p, 5).tolist() for p in polylines]
    lines._save(ch, data)
    return ch


def _check_four_lands(result, side):
    found = sorted((d for d in result["lands"] if d["side"] == side), key=lambda d: d["s"])
    assert len(found) == 4, [d["s"] for d in found]
    assert np.abs(np.array([d["s"] for d in found]) - clinker_truth(6.01)).max() < 0.005
    assert np.abs(np.array([d["step_mm"] for d in found]) - 15.0).max() < 2.0


def test_a_closed_section_ends_at_the_sheer(tmp_path):
    """Outer face, round the sheer, the inner face back down: the profile turns back and stops there."""
    from hulls import clinker_section

    outer = clinker_section(6.01)
    centre = np.array([0.0, outer[:, 1].max() + 0.5])
    inner = outer + 0.02 * (centre - outer) / np.linalg.norm(centre - outer, axis=1)[:, None]  # 2 cm inside
    loop = np.vstack([outer, inner[::-1], outer[:1]])
    ch = _one_station(tmp_path, [loop])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        _check_four_lands(result, side)
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["end"] == "turn"


def test_a_section_running_into_the_seabed_ends_at_the_sheer(tmp_path):
    from hulls import clinker_section

    hull = clinker_section(6.01)
    b, top = hull[-1, 0], hull[-1, 1]
    port = np.array([[b + 0.05, top - 0.2], [b + 1.5, top - 0.2]])
    starboard = port * [-1.0, 1.0]
    ch = _one_station(tmp_path, [np.vstack([starboard[::-1], hull, port])])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        _check_four_lands(result, side)
        assert next(st for st in result["stations"] if st["side"] == side)["end"] == "bend"


def test_a_stray_piece_and_a_keel_off_the_centreline(tmp_path):
    """A loose timber 30 cm inside the hull is listed apart, not a gap; a hull 4 cm off the centreline is read
    from its lowest point, the arcs unchanged."""
    from hulls import clinker_section

    hull = clinker_section(6.01) + [0.04, 0.0]
    y = np.linspace(0.5, 0.6, 6)
    timber = np.column_stack([y + 0.04, 1.5 * (y**2 / 1.99**2 - 1.0) + 0.3])  # above the bottom, inside
    ch = _one_station(tmp_path, [hull, timber])
    result = clinker.detect_lands(ch)
    assert not result["skipped"]
    for side in ("port", "starboard"):
        _check_four_lands(result, side)
    assert next(st for st in result["stations"] if st["side"] == "port")["stray"] == 1
    assert next(st for st in result["stations"] if st["side"] == "starboard")["stray"] == 0
    # a piece that goes on beyond the chain's end is a gap, not a stray
    upper = hull[hull[:, 0] > 1.0]
    lower = hull[hull[:, 0] < 0.8]
    (tmp_path / "gap").mkdir()
    ch = _one_station(tmp_path / "gap", [lower, upper])
    assert [(d["side"], d["reason"]) for d in clinker.detect_lands(ch)["skipped"]] == [("port", "gap")]


# ------------------------------------------------------------------------- fix round 2
def _built(segments, lands, land=0.015, lap=0.02, step=0.002, keel=None, deadrise=0.0, lean=0.0, heel=0.0):
    """A section built by hand, both sides alike: a base curve from the keel (``segments``: (length, curvature)
    from a start at ``deadrise`` degrees), each land a step of ``land`` outward over ``lap`` at the arcs ``lands``;
    ``keel``: (half width, depth) of a keel below the rabbet, where the planking starts; ``lean``: degrees from
    vertical of the keel's sides, the rabbet kept at the half width and the bottom narrower (a tapered keel; a
    pointed one when the bottom's width reaches zero); ``heel``: the whole section rotated by these degrees about
    the rabbet's level (port down for a positive heel). Returns the polyline (starboard sheer to port sheer) and the
    truth: the arcs of the lands along the stepped profile from the rabbet (a rotation keeps them)."""
    angle = np.radians(deadrise)
    u = keel[0] if keel else 0.0
    z, base = 0.0, [(u, 0.0, angle)]
    for length, curvature in segments:
        for _ in range(int(round(length / step))):
            angle += curvature * step
            u, z = u + step * np.cos(angle - curvature * step / 2), z + step * np.sin(angle - curvature * step / 2)
            base.append((u, z, angle))
    base = np.array(base)
    s = np.arange(len(base)) * step
    offset = land * np.clip((s[:, None] - np.asarray(lands)[None]) / lap + 0.5, 0.0, 1.0).sum(axis=1)
    side = base[:, :2] + offset[:, None] * np.column_stack([np.sin(base[:, 2]), -np.cos(base[:, 2])])
    arc = np.r_[0.0, np.cumsum(np.hypot(*np.diff(side, axis=0).T))]
    if keel:
        w, depth = keel
        wb = w - depth * np.tan(np.radians(lean))  # the bottom's half width
        wb = 0.0 if wb < 1e-9 else wb
        n = int(depth / step)
        wall = np.column_stack([np.linspace(wb, w, n, endpoint=False), np.linspace(-depth, 0.0, n, endpoint=False)])
        bottom = np.column_stack([np.arange(0.0, wb, step), np.full(len(np.arange(0.0, wb, step)), -depth)])
        side = np.vstack([bottom, wall, side])
    first = 1 if side[0, 0] == 0.0 else 0  # the centre point once
    polyline = np.vstack([(side[first:] * [-1.0, 1.0])[::-1], side])
    if heel:
        c, sn = np.cos(np.radians(heel)), np.sin(np.radians(heel))
        polyline = polyline @ np.array([[c, -sn], [sn, c]])  # (y, z) turned by heel: port (y > 0) goes down
    return polyline, np.interp(lands, s, arc)


def _lands_on(ch, side):
    return sorted((d for d in ch.meta["shipwreck"]["clinker"]["lands"] if d["side"] == side), key=lambda d: d["s"])


@pytest.mark.parametrize("level", [0.0, 0.05, -0.05])
def test_sediment_at_the_sheer_ends_the_profile(tmp_path, level):
    """Sediment level with the sheer, 5 cm above or below it, with 3 mm of noise: the four lands, none beyond."""
    from hulls import clinker_section

    hull = clinker_section(6.01)
    b, top = hull[-1]
    rng = np.random.default_rng(7)
    y = np.arange(b + 0.05, b + 1.5, 0.02)
    port = np.vstack([[b + 0.03, top + level], np.column_stack([y, top + level + rng.normal(0.0, 0.003, len(y))])])
    starboard = port * [-1.0, 1.0]
    ch = _one_station(tmp_path, [np.vstack([starboard[::-1], hull, port])])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        _check_four_lands(result, side)
        assert next(st for st in result["stations"] if st["side"] == side)["end"] == "bend"


def test_a_level_bottom_stepping_down_is_read_whole(tmp_path):
    """A level floor with 8 outward lands (the bottom 12 cm below the keel at the bilge), then the bilge and a
    side: nothing ends it early."""
    lands = [0.12 * k for k in range(1, 9)]
    section, truth = _built([(1.0, 0.0), (0.3 * np.pi / 2, 1 / 0.3), (0.5, 0.0)], lands)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert not result["skipped"]
    for side in ("port", "starboard"):
        found = _lands_on(ch, side)
        assert len(found) == 8 and np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005
        assert next(st for st in result["stations"] if st["side"] == side)["end"] == "line"


def test_tumblehome_is_read_whole(tmp_path):
    """15 cm of tumblehome over the top strake: the side leans in, the profile goes on to its end."""
    lands = [0.2, 0.5, 0.8, 1.1]
    section, truth = _built([(0.3, 0.0), (0.3 * np.pi / 2, 1 / 0.3), (0.5, 0.0), (0.5, 1.2)], lands)
    side_u = section[-1, 0]
    assert section[:, 0].max() - side_u > 0.14  # the tumblehome is there
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        found = _lands_on(ch, side)
        assert len(found) == 4 and np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["end"] == "line"


def test_the_keel_is_the_hulls_not_a_trench_floor_below(tmp_path):
    from hulls import clinker_section

    hull = clinker_section(6.01)
    floor = np.column_stack([np.linspace(-3.0, 3.0, 301), np.full(301, hull[:, 1].min() - 0.3)])
    ch = _one_station(tmp_path, [floor, hull])
    result = clinker.detect_lands(ch)
    assert not result["skipped"]
    for side in ("port", "starboard"):
        _check_four_lands(result, side)
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["keel"] == pytest.approx(0.0, abs=0.005) and record["stray"] == 1


def test_a_trench_wall_beyond_an_eroded_sheer_is_a_stray(tmp_path):
    """The hull eroded below its sheer, sediment out from there, the trench wall cut as its own polyline farther
    out and higher: the side is read (its 4 lands), the wall left apart."""
    from hulls import clinker_section

    hull = clinker_section(6.01)
    port = hull[(hull[:, 0] >= 0) & (hull[:, 0] < 1.6)]
    top = port[-1]
    y = np.arange(top[0] + 0.02, top[0] + 0.6, 0.02)
    sediment = np.column_stack([y, np.full(len(y), top[1])])
    wall = np.column_stack([np.full(30, y[-1] + 0.03), top[1] + np.linspace(0.0, 0.6, 30)])
    ch = _one_station(tmp_path, [np.vstack([hull[hull[:, 0] < 0], port, sediment]), wall])
    result = clinker.detect_lands(ch)
    assert not result["skipped"]
    record = next(st for st in result["stations"] if st["side"] == "port")
    assert record["end"] == "bend" and record["stray"] == 1
    found = _lands_on(ch, "port")
    assert len(found) == 4
    assert np.abs(np.array([d["s"] for d in found]) - clinker_truth(6.01)).max() < 0.005



# ------------------------------------------------------------------------- fix round 3
@pytest.mark.parametrize("depth, deadrise", [(0.15, 10.0), (0.25, 20.0), (0.06, 15.0)])
def test_a_keel_below_the_garboards(tmp_path, depth, deadrise):
    """A keel 6-25 cm deep, the garboards leaving its top at 10-20 degrees: the profile starts at the rabbet,
    the arcs from there, four lands, no land at the keel's corner, read to its end."""
    lands = [0.15, 0.32, 0.5, 0.7]
    section, truth = _built([(0.6, 0.0), (0.3 * np.pi / 2, 1 / 0.3), (0.4, 0.0)], lands, keel=(0.05, depth),
                            deadrise=deadrise)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert not result["skipped"]
    for side in ("port", "starboard"):
        found = _lands_on(ch, side)
        assert len(found) == 4, [d["s"] for d in found]
        assert np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["end"] == "line" and record["rabbet"][1] == pytest.approx(0.0, abs=0.003)


def test_a_hollow_garboard_is_read_whole(tmp_path):
    """A wineglass section: the garboard leaves the keel at 50 degrees and flattens to level (a fall of 50
    degrees near the keel) before the bilge: not taken for sediment."""
    lands = [0.3, 0.45, 0.62, 0.8]
    section, truth = _built([(0.25, -np.radians(50.0) / 0.25), (0.3, 0.0), (0.3 * np.pi / 2, 1 / 0.3), (0.4, 0.0)],
                            lands, keel=(0.05, 0.2), deadrise=50.0)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        found = _lands_on(ch, side)
        assert len(found) == 4 and np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005
        assert next(st for st in result["stations"] if st["side"] == side)["end"] == "line"


# ------------------------------------------------------------------------- fix round 4
@pytest.mark.parametrize("keel", [None, (0.05, 0.15)])
def test_a_steep_v_section_is_not_a_keel(tmp_path, keel):
    """A V section at 70 degrees (an end station near the stem or the stern), alone or above a 15 cm keel: the V is
    planking, not keel; four lands from the keel's bottom, or from the keel's top."""
    lands = [0.15, 0.32, 0.5, 0.68]
    section, truth = _built([(0.8, 0.0)], lands, keel=keel, deadrise=70.0)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    for side in ("port", "starboard"):
        found = _lands_on(ch, side)
        assert len(found) == 4, [d["s"] for d in found]
        assert np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["rabbet"][1] == pytest.approx(0.0, abs=0.003)  # the apex of the V, or the keel's top


# ------------------------------------------------------------------------- fix round 5
_KEEL_HULL = [(0.6, 0.0), (0.3 * np.pi / 2, 1 / 0.3), (0.4, 0.0)]  # 60 cm of floor, the bilge, 40 cm of side


def _check_lands(ch, side, truth):
    found = _lands_on(ch, side)
    assert len(found) == 4, [d["s"] for d in found]
    assert np.abs(np.array([d["s"] for d in found]) - truth).max() < 0.005


@pytest.mark.parametrize("lean", [15.0, 20.0])
def test_a_keel_with_leaning_sides(tmp_path, lean):
    """A tapered keel, its sides leaning 15-20 degrees from vertical (the bottom narrower than the rabbet), on a
    sole: the sides are the keel's, the profile starts at the rabbet; four lands within 5 mm of the arcs from
    there, none at the keel's corner."""
    section, truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.08, 0.15), deadrise=15.0, lean=lean)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert not result["skipped"] and result["mode"] == {"port": 4, "starboard": 4}
    for side in ("port", "starboard"):
        _check_lands(ch, side, truth)
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["end"] == "line" and record["rabbet"][1] == pytest.approx(0.0, abs=0.003)
        assert abs(record["rabbet"][0]) == pytest.approx(0.08, abs=0.003)
        assert record["sole"] == pytest.approx(2 * (0.08 - 0.15 * np.tan(np.radians(lean))), abs=0.005)


@pytest.mark.parametrize("heel", [12.0, 15.0])
def test_a_heeled_wreck_is_read_alike_on_both_sides(tmp_path, heel):
    """A wreck lying heeled by 12-15 degrees (a 15 cm keel, 15 degrees of deadrise): the low side's keel face
    leans, the high side's overhangs; both are read as the keel's side, four lands each within 5 mm of the arcs
    from the rabbet, the same arcs on the two sides."""
    section, truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.05, 0.15), deadrise=15.0, heel=heel)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert not result["skipped"] and result["mode"] == {"port": 4, "starboard": 4}
    c, sn = np.cos(np.radians(heel)), np.sin(np.radians(heel))
    for side, sign in (("port", 1.0), ("starboard", -1.0)):
        _check_lands(ch, side, truth)
        record = next(st for st in result["stations"] if st["side"] == side)
        rabbet = np.array([sign * 0.05 * c, -sign * 0.05 * sn])  # (+-0.05, 0) turned by the heel
        assert np.hypot(*(np.array(record["rabbet"]) - rabbet)) < 0.003 and record["end"] == "line"
    port = np.array([d["s"] for d in _lands_on(ch, "port")])
    starboard = np.array([d["s"] for d in _lands_on(ch, "starboard")])
    assert np.abs(port - starboard).max() < 0.002


@pytest.mark.parametrize("lean", [np.degrees(np.arctan(0.05 / 0.15)), 15.0])
def test_a_steep_start_with_no_sole_is_keel_unclear(tmp_path, lean):
    """A pointed keel (its sides at 72 degrees meeting below) or one whose sole is under 2 cm (sides at 75
    degrees): steep from the start, within the half width, the garboards flat above it (not a V of planking,
    whose steepness returns after a land). Keel or planking? Both sides are skipped with the reason and listed;
    no land is measured from a wrong origin."""
    section, _truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.05, 0.15), deadrise=15.0, lean=lean)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert [(d["station"], d["side"], d["reason"]) for d in result["skipped"]] == [
        ("S1", "port", "keel_unclear"), ("S1", "starboard", "keel_unclear")]
    assert result["lands"] == [] and result["mode"] == {"port": 0, "starboard": 0} and not result["lines"]
    station = next(iter(lines.sections_of(ch, "station")))
    assert clinker.profile(station, "port") is None and clinker.profile(station, "starboard") is None
    text = clinker.summary(ch)
    assert "chiglia o fasciame" in text and "S1 babordo" in text and "S1 tribordo" in text


# ------------------------------------------------------------------------- task 1b: scan noise, steep Vs
NOISE = 0.0005  # half a millimetre, the least a photogrammetric model gives
SPACING = 0.005  # the vertices of a section cut from a mesh


def _noisy(polyline, sigma=NOISE, seed=1, spacing=SPACING):
    """``polyline`` resampled every ``spacing`` along its arc, Gaussian noise of ``sigma`` on both coordinates:
    a section cut from a photogrammetric model, not a clean polyline."""
    s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(polyline, axis=0).T))]
    su = np.arange(0.0, s[-1], spacing)
    p = np.column_stack([np.interp(su, s, polyline[:, 0]), np.interp(su, s, polyline[:, 1])])
    return p + np.random.default_rng(seed).normal(0.0, sigma, p.shape)


def _both_sides(ch, result, truth, rabbet_z=None, tol=0.005):
    assert not result["skipped"] and result["mode"] == {"port": 4, "starboard": 4}
    for side in ("port", "starboard"):
        _check_lands(ch, side, truth)
        record = next(st for st in result["stations"] if st["side"] == side)
        assert record["end"] == "line"
        if rabbet_z is not None:
            assert record["rabbet"][1] == pytest.approx(rabbet_z, abs=tol)


def test_a_noisy_section_of_the_clinker_hull(tmp_path):
    """The hull's section with the vertices every 5 mm and half a millimetre of noise: the four lands within 5 mm
    of the arcs of the clean section (the arc is read on the smoothed profile, which noise does not lengthen)."""
    from hulls import clinker_section

    ch = _one_station(tmp_path, [_noisy(clinker_section(6.01))])
    result = clinker.detect_lands(ch)
    _both_sides(ch, result, clinker_truth(6.01))


def test_a_noisy_keel_with_leaning_sides(tmp_path):
    section, truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.08, 0.15), deadrise=15.0, lean=20.0)
    ch = _one_station(tmp_path, [_noisy(section, seed=2)])
    result = clinker.detect_lands(ch)
    _both_sides(ch, result, truth, rabbet_z=0.0)
    for side in ("port", "starboard"):
        record = next(st for st in result["stations"] if st["side"] == side)
        assert abs(record["rabbet"][0]) == pytest.approx(0.08, abs=0.005)


def test_a_noisy_heeled_wreck(tmp_path):
    section, truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.05, 0.15), deadrise=15.0, heel=12.0)
    ch = _one_station(tmp_path, [_noisy(section, seed=3)])
    result = clinker.detect_lands(ch)
    _both_sides(ch, result, truth)


def test_a_noisy_v_above_a_keel(tmp_path):
    section, truth = _built([(0.8, 0.0)], [0.15, 0.32, 0.5, 0.68], keel=(0.05, 0.15), deadrise=70.0)
    ch = _one_station(tmp_path, [_noisy(section, seed=4)])
    result = clinker.detect_lands(ch)
    _both_sides(ch, result, truth, rabbet_z=0.0)


@pytest.mark.parametrize("deadrise", [80.0, 85.0])
def test_a_near_vertical_v_is_planking(tmp_path, deadrise):
    """A V of planking at 80-85 degrees with no keel (the stem and stern stations of a fine-ended hull): its
    steepness comes back after every land, so no run of it is a keel's side; four lands from the apex."""
    section, truth = _built([(0.8, 0.0)], [0.15, 0.32, 0.5, 0.68], deadrise=deadrise)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    _both_sides(ch, result, truth, rabbet_z=0.0, tol=0.003)


def test_a_near_vertical_v_above_a_keel_is_keel_unclear(tmp_path):
    """A V at 85 degrees above a 15 cm keel on a sole: the keel's side and the planking are 5 degrees apart, no
    corner; the run rising from the sole is refused (its steepness returns): the side is skipped, never read
    from the keel's bottom with the keel's top as a fifth land."""
    section, _truth = _built([(0.8, 0.0)], [0.15, 0.32, 0.5, 0.68], keel=(0.05, 0.15), deadrise=85.0)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert [(d["side"], d["reason"]) for d in result["skipped"]] == [("port", "keel_unclear"),
                                                                     ("starboard", "keel_unclear")]
    assert result["lands"] == [] and result["mode"] == {"port": 0, "starboard": 0}


def test_a_keel_side_collinear_with_the_planking_is_keel_unclear(tmp_path):
    """A keel on a sole whose sides lean 20 degrees and whose garboards go on at the same 70 degrees: the rabbet
    is invisible; the run leaves the keel's half width with no corner: skipped with the reason, not read from
    the keel's bottom."""
    section, _truth = _built([(0.8, 0.0)], [0.15, 0.32, 0.5, 0.68], keel=(0.08, 0.15), deadrise=70.0, lean=20.0)
    ch = _one_station(tmp_path, [section])
    result = clinker.detect_lands(ch)
    assert [(d["side"], d["reason"]) for d in result["skipped"]] == [("port", "keel_unclear"),
                                                                     ("starboard", "keel_unclear")]
    assert result["lands"] == []
    assert "chiglia o fasciame" in clinker.summary(ch)


def test_a_narrow_sole_is_read_with_a_lower_sole_width(tmp_path):
    """A worn keel whose bottom is under 2 cm wide (sides at 75 degrees) is keel_unclear with the default; with
    ``sole_width`` lowered to 1.5 cm its bottom is a sole, the sides are the keel's and the lands are read from
    the rabbet."""
    assert clinker.DEFAULTS["sole_width"] == 0.02
    section, truth = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7], keel=(0.05, 0.15), deadrise=15.0, lean=15.0)
    ch = _one_station(tmp_path, [section])
    assert [d["reason"] for d in clinker.detect_lands(ch)["skipped"]] == ["keel_unclear", "keel_unclear"]
    result = clinker.detect_lands(ch, {"sole_width": 0.015})
    _both_sides(ch, result, truth, rabbet_z=0.0, tol=0.003)
    assert ch.meta["shipwreck"]["clinker"]["params"]["sole_width"] == 0.015


# ------------------------------------------------------------------------- task 2: the strake model
def _point_to_polyline(p, line):
    """The distance of the points ``p`` (N, 2) from the polyline ``line`` (M, 2)."""
    a, b = line[:-1], line[1:]
    ab = b - a
    t = np.clip(((p[:, None] - a[None]) * ab[None]).sum(-1) / np.maximum((ab**2).sum(-1), 1e-18)[None], 0.0, 1.0)
    q = a[None] + t[..., None] * ab[None]
    return np.hypot(*(p[:, None] - q).transpose(2, 0, 1)).min(axis=1)


def _outward(xyz, faces, inside):
    """The share of the faces whose normal points away from ``inside(centroids)``, a point within the hull."""
    v = xyz[faces]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    centre = v.mean(axis=1)
    return float(((normal * (centre - inside(centre))).sum(axis=1) > 0).mean())


def _faces_point_outward(result, frame):
    """The PLY's faces (world) and the OBJ's (ship coordinates) point away from the keel line, on both sides."""
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    mesh = read_mesh(result["ply"])
    on_axis = lambda c: frame.to_world(np.column_stack([frame.to_ship(c)[:, 0], np.zeros((len(c), 2))]))  # noqa: E731
    for s in (0, 1):
        mine = mesh.faces[(mesh.field("side")[mesh.faces] == s).all(axis=1)]
        assert len(mine) and _outward(mesh.xyz, mine, on_axis) == 1.0, s
    text = result["obj"].read_text().splitlines()
    v = np.array([[float(a) for a in ln.split()[1:]] for ln in text if ln.startswith("v ")])
    f = np.array([[int(a) - 1 for a in ln.split()[1:]] for ln in text if ln.startswith("f ")])
    assert _outward(v, f, lambda c: np.column_stack([c[:, 0], np.zeros((len(c), 2))])) == 1.0


def _objects(path):
    return [ln.split()[1] for ln in path.read_text().splitlines() if ln.startswith("o ")]


def test_the_strake_model_of_a_clinker_hull(tmp_path):
    from hulls import clinker_section, clinker_widths

    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends)
    clinker.detect_lands(ch)
    result = clinker.build_strakes(ch)
    strakes = {(d["side"], d["k"]): d for d in result["strakes"]}
    assert sorted(strakes) == [(s, k) for s in ("port", "starboard") for k in range(1, 6)]
    truth = clinker_widths(6.01)
    for (side, k), d in strakes.items():
        assert d["widths"]["S5"] == pytest.approx(truth[k - 1], rel=0.05), (side, k, d["widths"]["S5"], truth[k - 1])
        assert set(d["widths"]) == {f"S{i + 1}" for i in range(len(STATIONS))}
        assert d["uncertainty_mm"]["S5"] > 0.0
        assert abs(d["thickness_mm"] - 15.0) < 2.0 and d["lap_mm"] == pytest.approx(20.0)
        assert d["length"] > 7.0
    assert ch.meta["shipwreck"]["clinker"]["strakes"] == result["strakes"]
    folder = ch.data_dir / "shipwreck" / "clinker"
    assert result["obj"] == folder / "strakes.obj" and result["ply"] == folder / "strakes.ply"
    names = _objects(result["obj"])
    assert sorted(names) == sorted([f"fascia_{s}{k}" for s in "BT" for k in range(1, 6)])
    text = result["obj"].read_text()
    assert "ipotesi" in text and "# frame" in text
    mesh = read_mesh(result["ply"])
    assert mesh.vertices.dtype["x"] == np.float64
    assert any("ipotesi" in c for c in mesh.comments)
    k, side = mesh.field("strake"), mesh.field("side")
    assert set(np.unique(k)) == {1, 2, 3, 4, 5} and set(np.unique(side)) == {0, 1}
    frame = ship.need_frame(ch)
    xyz = frame.to_ship(mesh.xyz)
    assert (xyz[side == 0, 1] > -1e-6).all() and (xyz[side == 1, 1] < 1e-6).all()  # 0: port (B), y > 0
    worst = 0.0
    for x in np.unique(np.round(xyz[:, 0], 6)):
        p = xyz[np.abs(xyz[:, 0] - x) < 1e-6][:, 1:]
        worst = max(worst, _point_to_polyline(p, clinker_section(x, cell=0.005)).max())
    assert worst < 0.01, worst
    _faces_point_outward(result, frame)


def test_the_other_bow_swaps_the_names_of_the_sides_and_nothing_else(tmp_path):
    from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=37.0)
    _ready(ch, ends, STATIONS[2:7])
    clinker.detect_lands(ch)
    first = clinker.build_strakes(ch)
    before = read_mesh(first["ply"])
    widths = {(d["side"], d["k"]): d["widths"] for d in first["strakes"]}
    ship.set_frame(ch, ship.need_frame(ch).other_bow())
    lines.build(ch, half_width=3.0)
    clinker.detect_lands(ch)
    second = clinker.build_strakes(ch)
    after = read_mesh(second["ply"])
    other = {"port": "starboard", "starboard": "port"}
    for d in second["strakes"]:
        assert d["widths"] == pytest.approx(widths[(other[d["side"]], d["k"])], abs=1e-4)
    for k in range(1, 6):
        for s in (0, 1):
            a = before.xyz[(before.field("strake") == k) & (before.field("side") == s)]
            b = after.xyz[(after.field("strake") == k) & (after.field("side") == 1 - s)]
            assert a.shape == b.shape and np.abs(a - b).max() < 1e-4
    names = _objects(second["obj"])
    assert sorted(names) == sorted(_objects(first["obj"]))
    _faces_point_outward(second, ship.need_frame(ch))


def test_the_strakes_in_the_catalogue(tmp_path):
    import shapely

    from ppg_shipwreck import timbers

    _doc, ch, ends = clinker_project(tmp_path)
    _ready(ch, ends, STATIONS[2:7])
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    timbers.add(ch, "frame", "port", "F1", timbers.geometry("point", [[0.0, 0.0]]))
    assert clinker.to_catalogue(ch) == 10
    assert clinker.to_catalogue(ch) == 10
    records = timbers.timbers_of(ch)
    planks = [r for r in records if r["kind"] == "plank"]
    assert len(planks) == 10 and len(records) == 11
    assert sorted(r["name"] for r in planks) == sorted(f"fascia {s}{k}" for s in "BT" for k in range(1, 6))
    frame = ship.need_frame(ch)
    for r in planks:
        assert r["notes"] == "modello delle fasce (ipotesi)" and r["station"] == "" and r["geometry"]["type"] == "polygon"
        assert r["side"] == ("port" if r["name"].endswith(("B1", "B2", "B3", "B4", "B5")) else "starboard")
        xy = np.asarray(r["geometry"]["xy"])
        assert shapely.Polygon(xy).area > 0.1
        y = frame.to_ship(np.column_stack([xy, np.zeros(len(xy))]))[:, 1]
        assert ((y > -1e-6) if r["side"] == "port" else (y < 1e-6)).all()  # the garboard's edge at the rabbet, y = 0


def test_the_strake_tables(tmp_path):
    from openpyxl import load_workbook

    _doc, ch, ends = clinker_project(tmp_path, stealer=(4.5, 7.5))
    _ready(ch, ends)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    rows = clinker.strake_rows(ch)
    assert len(rows) == 10
    for row in rows:  # the stealer's stations are uncertain: no width measured there, nothing interpolated
        for name in ("S4", "S5", "S6"):
            assert row["widths"][name] is None and row["uncertainty_mm"][name] is None
        assert row["widths"]["S1"] is not None and row["uncertainty_mm"]["S1"] > 0
    files = clinker.write_tables(ch)
    assert files["csv"].parent == ch.data_dir / "shipwreck" / "clinker"
    text = files["csv"].read_text(encoding="utf-8")
    assert "ipotesi dalle sezioni" in text and "±" in text
    book = load_workbook(files["xlsx"])
    sheet = book.worksheets[0]
    assert "ipotesi dalle sezioni" in sheet.title
    header = [c.value for c in sheet[1]]
    assert header[:6] == ["fascia", "lato", "k", "lunghezza (m)", "spessore (mm)", "sovrapposizione (mm)"]
    assert "S1 (m)" in header and "S1 ± mm" in header
    assert sheet.max_row == 11
    s4 = header.index("S4 (m)") + 1
    assert all(sheet.cell(row=r, column=s4).value is None for r in range(2, 12))


def test_a_plain_hull_has_no_strakes(tmp_path):
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:3])
    clinker.detect_lands(ch)
    with pytest.raises(ship.NotReady, match="(?i)nessun gradino"):
        clinker.build_strakes(ch)


def _stations(tmp_path, polylines):
    """A project whose stations' sections are replaced by ``polylines``, one per station, 1 m apart."""
    _doc, ch, ends = project(tmp_path)
    _ready(ch, ends, STATIONS[:len(polylines)])
    data = lines.load(ch)
    for section, p in zip(data["sections"], polylines):
        section["lines"] = [np.round(p, 5).tolist()]
    lines._save(ch, data)
    return ch


@pytest.mark.parametrize("move", ["garboard", "rabbet"])
def test_a_station_whose_rabbet_or_garboard_jumps_is_uncertain(tmp_path, move):
    """Five stations with the same section, the third with its lands 3 cm farther from the keel (the garboard
    wider) or the whole section 3 cm higher (the rabbet higher): the count is right, but the station is uncertain on
    both sides, its widths left empty; the other stations are not."""
    lands = [0.15, 0.32, 0.5, 0.7]
    plain, _truth = _built(_KEEL_HULL, lands)
    moved, _ = _built(_KEEL_HULL, [v + 0.03 for v in lands]) if move == "garboard" else (plain + [0.0, 0.03], None)
    ch = _stations(tmp_path, [plain, plain, moved, plain, plain])
    result = clinker.detect_lands(ch)
    assert result["uncertain"] == {"port": [], "starboard": []}
    assert result["jumps"] == {"port": ["S3"], "starboard": ["S3"]}
    for line in result["lines"]:
        assert line["uncertain_stations"] == ["S3"]
    assert "S3" in clinker.summary(ch)
    strakes = clinker.build_strakes(ch)["strakes"]
    for d in strakes:
        assert d["widths"]["S3"] is None and d["widths"]["S2"] is not None


def test_a_hull_with_no_jump_has_none(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path)
    _ready(ch, ends)
    assert clinker.detect_lands(ch)["jumps"] == {"port": [], "starboard": []}


def test_widths_carry_their_uncertainty(tmp_path):
    """Noisy sections (half a millimetre on vertices 5 mm apart) at five stations: the noise is read from the
    sections, every width carries an uncertainty of at least twice the noise plus half the step, and the true
    width lies within it."""
    from hulls import clinker_section, clinker_widths

    xs = STATIONS[2:7]
    ch = _stations(tmp_path, [_noisy(clinker_section(x), seed=10 + i) for i, x in enumerate(xs)])
    result = clinker.detect_lands(ch)
    for st in result["stations"]:
        assert st["noise_mm"] == pytest.approx(0.5, abs=0.15), st
    for d in clinker.build_strakes(ch)["strakes"]:
        for i, x in enumerate(xs):
            name = f"S{i + 1}"
            w, u = d["widths"][name], d["uncertainty_mm"][name]
            assert u >= 2 * 0.4 + 0.5 * 13.0
            assert abs(w - clinker_widths(x)[d["k"] - 1]) * 1000.0 <= u, (d["side"], d["k"], name, w, u)


def _rocker(c, jump_at=(), count=7):
    """``count`` sections 1 m apart, the same keel hull raised along z = c (x - xm)^2 (a keel's rocker), the stations
    ``jump_at`` (their indexes) 3 cm higher still."""
    lands = [0.15, 0.32, 0.5, 0.7]
    plain, _truth = _built(_KEEL_HULL, lands)
    xs = np.asarray(STATIONS[:count])
    rise = c * (xs - xs.mean()) ** 2
    rise[list(jump_at)] += 0.03
    return [plain + [0.0, r] for r in rise]


def test_a_curved_rabbet_is_not_a_jump(tmp_path):
    """A keel with rocker (9 cm over 3 m, a radius of 50 m) along seven stations 1 m apart: no station jumps,
    every width measured."""
    ch = _stations(tmp_path, _rocker(0.01))
    result = clinker.detect_lands(ch)
    assert result["jumps"] == {"port": [], "starboard": []} and result["uncertain"] == {"port": [], "starboard": []}
    for d in clinker.build_strakes(ch)["strakes"]:
        assert all(w is not None for w in d["widths"].values())


def test_a_jump_on_a_curved_rabbet_is_the_only_one(tmp_path):
    ch = _stations(tmp_path, _rocker(0.01, jump_at=[2]))
    result = clinker.detect_lands(ch)
    assert result["jumps"] == {"port": ["S3"], "starboard": ["S3"]}


def test_the_catalogue_follows_a_model_with_fewer_strakes(tmp_path):
    """Five strakes a side in the catalogue, then the sections read again with one land fewer: the catalogue has
    the four strakes a side of the new model, no orphan ``fascia B5``, and a second call keeps the count."""
    from ppg_shipwreck import timbers

    five, _ = _built(_KEEL_HULL, [0.15, 0.32, 0.5, 0.7])
    four, _ = _built(_KEEL_HULL, [0.15, 0.32, 0.5])
    ch = _stations(tmp_path, [five] * 4)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    assert clinker.to_catalogue(ch) == 10
    timbers.add(ch, "plank", "port", "fascia B9", timbers.geometry("point", [[0.0, 0.0]]), notes="misurata in campo")
    data = lines.load(ch)
    for section in data["sections"]:
        section["lines"] = [np.round(four, 5).tolist()]
    lines._save(ch, data)
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    for _ in range(2):
        assert clinker.to_catalogue(ch) == 8
        names = sorted(r["name"] for r in timbers.timbers_of(ch))
        assert names == sorted([f"fascia {s}{k}" for s in "BT" for k in range(1, 5)] + ["fascia B9"])


@pytest.mark.parametrize("c", [0.0, 0.01])
def test_two_adjacent_jumps_are_both_found_and_no_clean_station(tmp_path, c):
    """Nine stations, the fourth and the fifth both 3 cm higher (a plateau two stations long): those two are the
    jumps on both sides; the clean end station beyond them is not."""
    ch = _stations(tmp_path, _rocker(c, jump_at=[3, 4], count=9))
    result = clinker.detect_lands(ch)
    assert result["jumps"] == {"port": ["S4", "S5"], "starboard": ["S4", "S5"]}


def test_the_catalogue_never_takes_a_users_record_with_a_reused_id(tmp_path):
    """The strakes written, then deleted by the user, who records a real plank: it gets a freed id; building the
    catalogue again keeps it."""
    from ppg_shipwreck import timbers

    _doc, ch, ends = clinker_project(tmp_path)
    _ready(ch, ends, STATIONS[2:7])
    clinker.detect_lands(ch)
    clinker.build_strakes(ch)
    clinker.to_catalogue(ch)
    written = list(ch.meta["shipwreck"]["clinker"]["catalogue"])
    timbers.remove(ch, written)
    mine = timbers.add(ch, "plank", "port", "corso 3", timbers.geometry("point", [[0.0, 0.0]]), notes="rilevato a mano")
    assert mine["id"] in written  # the id is handed out again
    assert clinker.to_catalogue(ch) == 10
    assert any(r["id"] == mine["id"] and r["name"] == "corso 3" for r in timbers.timbers_of(ch))
    assert len(timbers.timbers_of(ch)) == 11
