"""The core on a known hull: sections within 2 mm of the truth, a frame invariant to the world's
rotation, UTM-sized coordinates kept exact, the water correction exact on a flat bottom."""
from __future__ import annotations

import json

import numpy as np
import pytest
from hulls import D, L, half_breadth, project

from ppg_shipwreck import lines, ship


def _frame(ch, ends, bow="end"):
    return ship.set_frame(ch, ship.Frame(ends[0], ends[1], bow))


def test_sections_are_within_2_mm_of_the_truth(tmp_path):
    _doc, ch, ends = project(tmp_path, heading_deg=37.0)
    _frame(ch, ends)
    ship.set_stations(ch, ship.regular(2.0, 5, start=2.0))
    data = lines.build(ch, longitudinals=[0.0, 0.5], horizontals=[-0.5], half_width=3.0)
    for s in lines.sections_of(ch, "station"):
        assert s["axes"] == ["y", "z"]
        p = np.vstack(s["lines"])
        truth = D * (p[:, 0] ** 2 / half_breadth(s["value"]) ** 2 - 1.0)
        assert np.abs(p[:, 1] - truth).max() < 0.002
        assert np.ptp(p[:, 0]) == pytest.approx(2 * half_breadth(s["value"]), abs=0.01)
    keel = next(s for s in lines.sections_of(ch, "longitudinal") if s["value"] == 0.0)
    p = np.vstack(keel["lines"])
    assert np.abs(p[:, 1] + D).max() < 0.002
    horizontal = lines.sections_of(ch, "horizontal")[0]
    p = np.vstack(horizontal["lines"])
    assert np.abs(np.abs(p[:, 1]) - half_breadth(p[:, 0]) * np.sqrt(1 - 0.5 / D)).max() < 0.002
    assert (ch.data_dir / "shipwreck" / "sections.json").exists()
    assert json.loads((ch.data_dir / "shipwreck" / "sections.json").read_text())["faces"] == data["faces"]


def test_the_frame_does_not_depend_on_the_heading_of_the_world(tmp_path):
    results = []
    for k, heading in enumerate((0.0, 37.0, 201.5)):
        (tmp_path / str(k)).mkdir()
        _doc, ch, ends = project(tmp_path / str(k), heading_deg=heading)
        _frame(ch, ends)
        ship.set_stations(ch, [{"name": "F30", "x": 6.0}])
        lines.build(ch, half_width=3.0)
        p = np.vstack(lines.sections_of(ch, "station")[0]["lines"])
        results.append(p[np.lexsort(p.T)])
    assert np.abs(results[0] - results[1]).max() < 2e-4
    assert np.abs(results[0] - results[2]).max() < 2e-4


def test_coordinates_of_utm_size_are_kept_exact(tmp_path):
    _doc, ch, ends = project(tmp_path, heading_deg=123.4)
    frame = _frame(ch, ends)
    again = ship.Frame.from_dict(json.loads(json.dumps(ch.meta["shipwreck"]["frame"])))
    points = np.array([[500130.123456789, 4500460.987654321, 11.5], [500125.0, 4500450.0, 10.0]])
    assert np.abs(again.to_world(again.to_ship(points)) - points).max() < 1e-8
    assert np.abs(frame.to_ship(ends) - [[0.0, 0.0, 0.0], [L, 0.0, 0.0]]).max() < 1e-9
    assert np.abs(frame.to_ship(points) - lines.geometry.apply(frame.matrix, points)).max() < 1e-8
    near, far = tmp_path / "near", tmp_path / "far"
    near.mkdir(), far.mkdir()
    out = []
    for folder, offset in ((near, np.zeros(3)), (far, np.array([6.5e5, 4.9e6, 0.0]))):
        _doc, ch, ends = project(folder, heading_deg=123.4, offset=offset)
        _frame(ch, ends)
        ship.set_stations(ch, [{"name": "S", "x": 3.0}])
        lines.build(ch, half_width=3.0)
        p = np.vstack(lines.sections_of(ch, "station")[0]["lines"])
        out.append(p[np.lexsort(p.T)])
    assert np.abs(out[0] - out[1]).max() <= 1e-4 + 1e-9  # the rounding of the file, no more


def test_port_is_port(tmp_path):
    """x from the bow: port on the right of who walks along x; x from the stern: on the left. The
    hull of the test is given a mark on one side."""
    _doc, ch, ends = project(tmp_path, heading_deg=90.0)  # x to the east
    east, north = 1.0, 2.0
    left = ends[0] + [east, north, 0.0]  # north of the keel: the left of who walks east
    assert ship.Frame(ends[0], ends[1], bow="end").to_ship(left)[0, 1] == pytest.approx(north)
    assert ship.Frame(ends[0], ends[1], bow="start").to_ship(left)[0, 1] == pytest.approx(-north)
    frame = ship.Frame(ends[0], ends[1], bow="start")
    flipped = frame.flipped()
    assert flipped.start == frame.end and flipped.to_ship(ends[1])[0, 0] == pytest.approx(0.0)
    assert flipped.z0 == frame.z0


def test_the_keel_is_levelled_and_its_slope_kept(tmp_path):
    frame = ship.Frame((100.0, 200.0, 5.0), (100.0, 210.0, 5.5))
    assert frame.slope == pytest.approx(0.05)
    assert frame.length == pytest.approx(10.0)
    assert frame.to_ship([100.0, 205.0, 5.0])[0, 2] == pytest.approx(0.0)  # z is the world's height over the datum
    assert frame.heading == pytest.approx(0.0)


def test_a_chunk_in_degrees_is_refused(tmp_path):
    _doc, ch, ends = project(tmp_path, crs="EPSG:4326")
    with pytest.raises(ship.NotReady, match="proiettato"):
        ship.frame_from_picks(ch, ends[0][:2], ends[1][:2])
    ch.crs = "LOCAL"  # a local frame in metres is fine
    middle = ch.to_world(np.array([[1.0, 0.0, 0.0], [5.0, 0.0, 0.0]]))  # x = 2 and 10 m of the hull: the model is there
    frame = ship.frame_from_picks(ch, middle[0][:2], middle[1][:2])
    assert frame.start[2] == pytest.approx(ends[0][2] - D, abs=0.05)


def test_the_water_correction_is_exact_on_a_flat_bottom(tmp_path):
    z = np.array([0.5, 0.0, -0.3, -1.0])  # level 0: above, on it, 30 cm and 1 m under
    assert lines.corrected(z, 0.0, 1.34) == pytest.approx([0.5, 0.0, -0.402, -1.34])
    _doc, ch, ends = project(tmp_path)
    _frame(ch, ends)
    ship.set_stations(ch, [{"name": "S", "x": 6.0}])
    lines.build(ch, half_width=3.0)
    raw = lines.sections_of(ch, "station")[0]
    assert "corrected" not in raw
    lines.set_water(ch, -0.5, on=True)
    s = lines.sections_of(ch, "station")[0]
    assert s["lines"] == raw["lines"]  # the raw line is kept
    a, b = np.vstack(s["lines"]), np.vstack(s["corrected"])
    under = a[:, 1] < -0.5
    assert under.any() and (~under).any()
    assert np.abs(b[under, 1] - (-0.5 - (-0.5 - a[under, 1]) * 1.34)).max() < 2e-4
    assert np.array_equal(b[~under, 1], a[~under, 1])
    assert "1,34" in lines.note(lines.load(ch))
    lines.set_water(ch, -0.5, on=False)
    assert "corrected" not in lines.sections_of(ch, "station")[0]


def test_sections_out_of_date_are_said(tmp_path):
    _doc, ch, ends = project(tmp_path)
    frame = _frame(ch, ends)
    ship.set_stations(ch, [{"name": "S", "x": 6.0}])
    assert lines.is_stale(ch)
    lines.build(ch)
    assert not lines.is_stale(ch)
    ship.set_frame(ch, frame.flipped())
    assert lines.is_stale(ch)


def test_a_table_of_stations_keeps_the_names_of_the_field(tmp_path):
    table = tmp_path / "st.csv"
    table.write_text("nome;x\nF30;12,4\nF31;13\n", encoding="utf-8")
    assert ship.stations_from_table(table) == [{"name": "F30", "x": 12.4}, {"name": "F31", "x": 13.0}]
    table.write_text("F30,12.4\nF31,13\n", encoding="utf-8")
    assert ship.stations_from_table(table)[0] == {"name": "F30", "x": 12.4}


def test_the_model_and_the_chunk_are_not_changed(tmp_path):
    _doc, ch, ends = project(tmp_path)
    before = (ch.data_dir / "model.ply").read_bytes(), ch.transform.copy(), dict(ch.products)
    _frame(ch, ends)
    ship.set_stations(ch, [{"name": "S", "x": 6.0}])
    lines.build(ch)
    assert (ch.data_dir / "model.ply").read_bytes() == before[0]
    assert np.array_equal(ch.transform, before[1]) and ch.products == before[2]
