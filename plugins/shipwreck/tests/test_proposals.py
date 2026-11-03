"""Proposals read on the relief of the hull: the seams of the planking, the frames, the keelson and the strakes on a
framed clinker hull; nothing on a smooth one; the catalogue's records and the rejected geometries never touched."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import shapely
from hulls import FRAMES, framed_project, project, seam_y, strake_widths

from ppg_shipwreck import proposals, relief, ship, timbers
from ppg_shipwreck.ship import NotReady

CELL = 0.005  # the plugin's default cell: the lands are told apart to about a metre from the ends


def _setup(tmp_path, bow="start", **hull):
    _doc, ch, ends = framed_project(tmp_path, heading_deg=35, **hull)
    frame = ship.set_frame(ch, ship.Frame(ends[0], ends[1], bow))
    relief.build(ch, cell=CELL)
    return ch, frame


@pytest.fixture(scope="module")
def framed(tmp_path_factory):
    ch, frame = _setup(tmp_path_factory.mktemp("framed"))
    prop = proposals.build(ch)
    return ch, frame, prop


def _of(prop, kind):
    return [i for i in prop["items"] if i["kind"] == kind]


def _centroid(item):
    return shapely.Polygon(item["ship_xy"]).centroid


def test_frames_found_with_room_and_space(framed):
    _ch, _frame, prop = framed
    frames = sorted(_of(prop, "frame"), key=lambda f: _centroid(f).x)
    assert len(frames) == len(FRAMES) == 23
    xs = np.array([_centroid(f).x for f in frames])
    assert np.abs(xs - np.asarray(FRAMES)).max() < 0.01
    rooms = np.array([f["room"] for f in frames])
    assert np.abs(rooms - 0.08).max() < 0.01
    assert abs(float(np.median(np.diff(xs))) - 0.45) < 0.01
    # across the hull: each frame reaches both sides, cut by nothing
    assert all(f["side"] == "centre" for f in frames)
    assert all(shapely.Polygon(f["ship_xy"]).bounds[1] < -1.0 < 1.0 < shapely.Polygon(f["ship_xy"]).bounds[3]
               for f in frames[2:-2])
    # named from the bow (at the start of the keel line: x 0)
    assert frames[0]["name"] == "ordinata 1" and frames[-1]["name"] == "ordinata 23"
    assert len({f["name"] for f in frames}) == 23


def test_seams_follow_the_lands(framed):
    _ch, _frame, prop = framed
    seams = _of(prop, "seam")
    grid = np.round(np.arange(0.2, 11.81, 0.02), 2)
    truth = {x: seam_y(x) for x in grid}
    for side, s in (("port", 1.0), ("starboard", -1.0)):
        mine = [i for i in seams if i["side"] == side]
        assert len(mine) == 4, [len(i["ship_xy"]) for i in mine]
        lands = []
        for seam in mine:
            xy = np.asarray(seam["ship_xy"])
            assert np.all(np.sign(xy[:, 1]) == s)
            xs = grid[(grid >= xy[:, 0].min()) & (grid <= xy[:, 0].max())]
            y = np.interp(xs, xy[:, 0], xy[:, 1]) * s
            t = np.array([truth[x] for x in xs])  # (n, 4): the lands' y at each x
            k = int(np.argmin(np.median(np.abs(t - y[:, None]), axis=0)))
            lands.append(k)
            assert xy[:, 0].max() - xy[:, 0].min() > 8.0  # along most of the hull
            assert (np.abs(t[:, k] - y) < 0.01).mean() >= 0.9
            assert 8.0 < seam["step_mm"] < 25.0
        assert sorted(lands) == [0, 1, 2, 3]


def test_keelson_is_one_longitudinal_on_the_centre(framed):
    _ch, _frame, prop = framed
    keelsons = _of(prop, "keelson")
    assert len(keelsons) == 1 and not _of(prop, "stringer")
    k = keelsons[0]
    x0, y0, x1, y1 = shapely.Polygon(k["ship_xy"]).bounds
    assert k["side"] == "centre" and k["name"] == "paramezzale"
    assert abs(x0 - 1.0) < 0.02 and abs(x1 - 11.0) < 0.02
    assert abs(k["room"] - 0.12) < 0.01 and abs(y0 + y1) < 0.01


def test_strakes_count_and_widths(framed):
    _ch, _frame, prop = framed
    widths = strake_widths(6.0)
    cut = shapely.LineString([(6.0, -5.0), (6.0, 5.0)])
    for side, code in (("port", "B"), ("starboard", "T")):
        planks = [i for i in _of(prop, "plank") if i["side"] == side]
        assert sorted(p["name"] for p in planks) == [f"fascia {code}{k}" for k in range(1, 6)]
        for k in (2, 3, 4):
            plank = next(p for p in planks if p["name"] == f"fascia {code}{k}")
            width = shapely.Polygon(plank["ship_xy"]).intersection(cut).length
            assert width == pytest.approx(widths[k - 1], rel=0.05)
        first = next(p for p in planks if p["name"] == f"fascia {code}1")
        assert np.abs(np.asarray(first["ship_xy"])[:, 1]).min() == pytest.approx(0.10, abs=1e-9)


def test_plain_hull_has_no_seams(tmp_path):
    _doc, ch, ends = project(tmp_path, heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.02)
    prop = proposals.build(ch)
    assert not _of(prop, "seam") and not _of(prop, "plank") and not _of(prop, "frame")
    assert "nessun gradino di fasciame trovato" in proposals.summary(ch)


def test_proposals_need_a_fresh_relief(tmp_path):
    _doc, ch, ends = framed_project(tmp_path, heading_deg=35, x_range=(4.0, 8.0))
    frame = ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    with pytest.raises(NotReady):
        proposals.build(ch)
    relief.build(ch, cell=CELL)
    ship.set_frame(ch, frame.other_bow())
    with pytest.raises(NotReady):
        proposals.build(ch)


def test_mirror_frame_sides(tmp_path):
    ch, frame = _setup(tmp_path, x_range=(3.0, 9.0))
    names = {}
    for f in (frame, frame.other_bow()):
        ship.set_frame(ch, f)
        relief.build(ch, cell=CELL)
        prop = proposals.build(ch)
        for item in prop["items"]:
            if item["kind"] in ("seam", "plank"):
                y = np.asarray(item["ship_xy"])[:, 1]
                assert item["side"] == ("port" if np.median(y) > 0 else "starboard")
                assert item["name"].split()[-1][0] == ("B" if item["side"] == "port" else "T")
        names[f.sign] = sorted((i["kind"], i["name"]) for i in prop["items"])
        frames = sorted(_of(prop, "frame"), key=lambda i: abs(_centroid(i).x - f.bow_x))
        assert frames[0]["name"] == "ordinata 1"
    assert names[-1] == names[1]
    assert sum(1 for k, _n in names[1] if k == "seam") == 8


def test_records_and_rejected_never_touched(framed, tmp_path):
    """The catalogue's records written by hand and the rejected geometries survive a new reading, and nothing is
    proposed again over them."""
    ch, frame = _setup(tmp_path, x_range=(3.0, 9.0))
    first = proposals.build(ch)
    fr = sorted(_of(first, "frame"), key=lambda i: _centroid(i).x)[2]
    xy = np.asarray(fr["ship_xy"])
    world = frame.to_world(np.column_stack([xy, np.zeros(len(xy))]))[:, :2]
    timbers.add(ch, "frame", "centre", "F12", timbers.geometry("polygon", world))
    seam = _of(first, "seam")[0]
    ship.state(ch)["proposals"]["rejected"].append(shapely.LineString(seam["ship_xy"]).wkt)
    records = copy.deepcopy(timbers.timbers_of(ch))
    rejected = list(ship.state(ch)["proposals"]["rejected"])
    last = max(int(i["id"][1:]) for i in first["items"])
    again = proposals.build(ch)
    assert timbers.timbers_of(ch) == records
    assert again["rejected"] == rejected
    assert all(abs(_centroid(i).x - _centroid(fr).x) > 0.04 for i in _of(again, "frame"))
    assert len(_of(again, "frame")) == len(_of(first, "frame")) - 1
    rej = shapely.LineString(seam["ship_xy"]).buffer(0.02)
    assert all(shapely.LineString(i["ship_xy"]).intersection(rej).length < 0.5 * shapely.LineString(i["ship_xy"]).length
               for i in _of(again, "seam"))
    assert len(_of(again, "seam")) == len(_of(first, "seam")) - 1
    assert min(int(i["id"][1:]) for i in again["items"]) > last  # ids never given twice


def test_rebuild_keeps_accepted_and_rejected(tmp_path):
    tracing = pytest.importorskip("ppg_shipwreck.tracing")
    ch, _frame = _setup(tmp_path, x_range=(3.0, 9.0))
    first = proposals.build(ch)
    frame_item = _of(first, "frame")[0]
    seam_item = _of(first, "seam")[0]
    plank_item = _of(first, "plank")[0]
    tracing.accept(ch, [frame_item["id"], seam_item["id"]])
    tracing.reject(ch, [plank_item["id"]])
    records = copy.deepcopy(timbers.timbers_of(ch))
    rejected = list(ship.state(ch)["proposals"]["rejected"])
    again = proposals.build(ch)
    assert timbers.timbers_of(ch) == records and again["rejected"] == rejected
    for gone in (frame_item, seam_item, plank_item):
        assert not any(i["kind"] == gone["kind"] and i["name"] == gone["name"] for i in again["items"])
