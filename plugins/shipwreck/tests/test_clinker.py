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
