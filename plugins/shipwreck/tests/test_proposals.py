"""Proposals read on the relief of the hull: the seams of the planking, the frames, the keelson and the strakes on a
framed clinker hull; nothing on a smooth one; the catalogue's records and the rejected geometries never touched."""
from __future__ import annotations

import copy
from types import SimpleNamespace

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


def test_records_and_rejected_never_touched(tmp_path):
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


def _planks(prop, side):
    return [i for i in _of(prop, "plank") if i["side"] == side]


def test_strakes_at_one_centimetre(tmp_path):
    """At 1 cm the bow leaves a short stray piece of a seam: it joins the first land, the strakes stay five."""
    _doc, ch, ends = framed_project(tmp_path, heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    prop = proposals.build(ch)
    widths = strake_widths(6.0)
    cut = shapely.LineString([(6.0, -5.0), (6.0, 5.0)])
    for side, code in (("port", "B"), ("starboard", "T")):
        assert sorted({p["name"] for p in _planks(prop, side)}) == [f"fascia {code}{k}" for k in range(1, 6)]
        for k in (2, 3, 4):
            width = sum(shapely.Polygon(p["ship_xy"]).intersection(cut).length
                        for p in _planks(prop, side) if p["name"] == f"fascia {code}{k}")
            assert width == pytest.approx(widths[k - 1], rel=0.05)


def test_hole_keeps_the_names_fore_and_aft(tmp_path):
    """A hole on starboard longer than ``gap``: each land and each strake in two pieces of one name, fore and aft; the
    keelson's wall standing over the hole is no seam; the keelson in two numbered pieces."""
    ch, _frame = _setup(tmp_path, missing=(6.0, 6.6, "starboard"))
    prop = proposals.build(ch)
    seams = _of(prop, "seam")
    assert all(s["step_mm"] <= 40.0 for s in seams)
    port = [s for s in seams if s["side"] == "port"]
    assert sorted(s["name"] for s in port) == [f"comento B{k}" for k in range(1, 5)]
    starboard = [s for s in seams if s["side"] == "starboard"]
    assert sorted(s["name"] for s in starboard) == sorted([f"comento T{k}" for k in range(1, 5)] * 2)
    for k in range(1, 6):
        pieces = [shapely.Polygon(p["ship_xy"]).bounds for p in _planks(prop, "starboard") if p["name"] == f"fascia T{k}"]
        assert len(pieces) == 2
        assert min(b[2] for b in pieces) < 6.05 and max(b[0] for b in pieces) > 6.55  # one fore, one aft
        assert len([p for p in _planks(prop, "port") if p["name"] == f"fascia B{k}"]) == 1
    assert sorted(k["name"] for k in _of(prop, "keelson")) == ["paramezzale 1", "paramezzale 2"]


def test_a_stray_seam_piece_changes_nothing(framed):
    ch, frame, _prop = framed
    z, tr = relief.load(ch, "height")
    seam_list = proposals.seams(z, tr, frame.sign)
    edges = proposals._Grid(z, tr, frame.sign).edges()
    before = proposals.strakes(seam_list, frame.sign, edges=edges)
    y = 0.5 * (np.asarray(seam_y(5.3)[1]) + np.asarray(seam_y(5.3)[2]))  # between the second and third land
    stray = {"side": "port", "ship_xy": [[5.0, y], [5.6, y + 0.005]], "step_mm": 12.0, "length": 0.6, "score": 1.0}
    after = proposals.strakes(seam_list + [stray], frame.sign, edges=edges)
    assert proposals.lands(seam_list + [stray])[-1] == 0  # a stray, proposed as comento B?
    assert [(b["name"], b["ship_xy"]) for b in after] == [(b["name"], b["ship_xy"]) for b in before]


def test_frames_on_two_sides_apart_share_a_number():
    """Port and starboard pieces of each frame 2 cm apart: one number each, the space 0.45 m."""
    frames = [{"x": x, "side": "port"} for x in FRAMES] + [{"x": x + 0.02, "side": "starboard"} for x in FRAMES]
    assert proposals.frame_space(frames) == pytest.approx(0.45)
    proposals._name_frames(frames, 0.0)
    assert len({f["name"] for f in frames}) == 23
    for a, b in zip(frames[:23], frames[23:]):
        assert a["name"] == b["name"]
    square = lambda x: [[x - 0.04, 0.2], [x + 0.04, 0.2], [x + 0.04, 2.0], [x - 0.04, 2.0]]  # noqa: E731
    items = [{"kind": "frame", "name": f["name"], "side": f["side"], "ship_xy": square(f["x"])} for f in frames]
    chunk = SimpleNamespace(meta={"shipwreck": {"proposals": {"items": items}}})
    text = proposals.summary(chunk)
    assert "23 ordinate" in text and "0,45 m" in text


def test_loose_timbers_are_other(tmp_path):
    """Four loose timbers lying on the hull at 10, 30, 60 and 85 degrees from x, between the frames: neither frames
    nor stringers, and the frames keep their numbers."""
    boxes = [(3.375, -0.35, 10), (4.275, -0.35, 30), (5.175, -0.35, 60), (6.975, -0.35, 85)]
    ch, _frame = _setup(tmp_path, loose=boxes)
    prop = proposals.build(ch)
    frames = sorted(_of(prop, "frame"), key=lambda f: _centroid(f).x)
    assert len(frames) == 23 and not _of(prop, "stringer")
    assert [f["name"] for f in frames] == [f"ordinata {k}" for k in range(1, 24)]
    others = _of(prop, "other")
    assert len(others) == 4
    assert sorted(round(_centroid(o).x, 1) for o in others) == [3.4, 4.3, 5.2, 7.0]


def _cut(seam, x0, x1):
    """The piece of a seam between x0 and x1."""
    xy = np.asarray(seam["ship_xy"])
    xs = np.r_[x0, xy[(xy[:, 0] > x0) & (xy[:, 0] < x1), 0], x1]
    return {**seam, "ship_xy": np.column_stack([xs, np.interp(xs, xy[:, 0], xy[:, 1])]).tolist(), "length": x1 - x0}


@pytest.mark.parametrize("case", ["land 1 fore of 4 m", "land 1 fore of 6.5 m", "land 2 only at the ends",
                                  "land 2 only 0.8 m", "land 2 only 0.5 m at each end", "land 4 only 6-7 m"])
def test_a_land_partly_lost_keeps_every_name(framed, case):
    """A land under sediment over part of the hull: every piece keeps its land, the strakes beside the lost stretch
    keep their names, and the strake whose seam is lost stops there."""
    ch, frame, _prop = framed
    z, tr = relief.load(ch, "height")
    seam_list = proposals.seams(z, tr, frame.sign)
    ranks = proposals.lands(seam_list)
    port = {r: s for s, r in zip(seam_list, ranks) if s["side"] == "port"}
    assert sorted(port) == [1, 2, 3, 4]
    if case == "land 1 fore of 4 m":
        pieces, truth = [_cut(port[1], 1.0, 4.0), port[2], port[3], port[4]], [1, 2, 3, 4]
    elif case == "land 1 fore of 6.5 m":
        pieces, truth = [_cut(port[1], 1.0, 6.5), port[2], port[3], port[4]], [1, 2, 3, 4]
    elif case == "land 2 only at the ends":
        pieces, truth = [port[1], _cut(port[2], 1.0, 2.4), _cut(port[2], 9.6, 11.0), port[3], port[4]], [1, 2, 2, 3, 4]
    elif case == "land 4 only 6-7 m":
        pieces, truth = [port[1], port[2], port[3], _cut(port[4], 6.0, 7.0)], [1, 2, 3, 4]
    elif case == "land 2 only 0.8 m":
        pieces, truth = [port[1], _cut(port[2], 2.0, 2.8), port[3], port[4]], [1, 2, 3, 4]
    else:
        pieces, truth = [port[1], _cut(port[2], 1.5, 2.0), _cut(port[2], 10.0, 10.5), port[3], port[4]], [1, 2, 2, 3, 4]
    assert proposals.lands(pieces) == truth
    bands = proposals.strakes(pieces, frame.sign)
    widths = strake_widths(8.0)
    cut = shapely.LineString([(8.0, -5.0), (8.0, 5.0)])
    at8 = {b["name"]: shapely.Polygon(b["ship_xy"]).intersection(cut).length for b in bands}
    if case.startswith("land 4"):  # land 4 only at 6-7: B4 there, B1-B3 at x 8, nothing beyond land 3 at x 8
        at65 = {b["name"]: shapely.Polygon(b["ship_xy"]).intersection(
            shapely.LineString([(6.5, -5.0), (6.5, 5.0)])).length for b in bands}
        assert at65["fascia B4"] == pytest.approx(strake_widths(6.5)[3], rel=0.05)
        assert at8.get("fascia B4", 0.0) == 0.0
        for k in (2, 3):
            assert at8[f"fascia B{k}"] == pytest.approx(widths[k - 1], rel=0.05)
    elif case.startswith("land 1"):  # at x 8 land 1 is gone: no B1, no B2; B3, B4 where they were
        assert at8.get("fascia B1", 0.0) == 0.0 and at8.get("fascia B2", 0.0) == 0.0
        for k in (3, 4):
            assert at8[f"fascia B{k}"] == pytest.approx(widths[k - 1], rel=0.05)
    else:  # land 2 lost at x 8: no B2, no B3 there; B1 and B4 untouched; B2 and B3 at the ends
        assert at8.get("fascia B2", 0.0) == 0.0 and at8.get("fascia B3", 0.0) == 0.0
        assert at8["fascia B4"] == pytest.approx(widths[3], rel=0.05)
        assert at8["fascia B1"] > 0.0
        assert {b["name"] for b in bands} == {f"fascia B{k}" for k in range(1, 5)}


def test_a_loose_plank_in_the_outermost_strake_renames_nothing(tmp_path):
    """A loose plank 1.5 x 0.08 x 0.02 m along x in the outermost strake, near the sheer: its edges are strays and the
    band from the outermost land to the model's edge runs the whole hull."""
    ch, _frame = _setup(tmp_path, loose=[(6.075, -0.85, 0.0, 1.5, 0.08, 0.02)])  # hull y < 0: port
    prop = proposals.build(ch)
    port = [s for s in _of(prop, "seam") if s["side"] == "port"]
    assert sorted(s["name"] for s in port if not s["name"].endswith("?")) == [f"comento B{k}" for k in range(1, 5)]
    assert any(s["name"] == "comento B?" for s in port)
    assert sorted({b["name"] for b in _planks(prop, "port")}) == [f"fascia B{k}" for k in range(1, 6)]
    for x in (4.0, 9.0):
        widths = strake_widths(x)
        cut = shapely.LineString([(x, -5.0), (x, 5.0)])
        for k in range(2, 6):
            width = sum(shapely.Polygon(b["ship_xy"]).intersection(cut).length
                        for b in _planks(prop, "port") if b["name"] == f"fascia B{k}")
            assert width == pytest.approx(widths[k - 1], rel=0.05)


def test_a_loose_plank_along_the_hull_renames_nothing(tmp_path):
    """A loose plank 1.5 x 0.08 x 0.02 m lying along x in the third strake: its edges are seams of 1.5 m, strays
    proposed as comento B? with a low score; the lands keep ranks 1-4 and the strakes B1-B5 run the whole hull."""
    ch, _frame = _setup(tmp_path, loose=[(6.075, -0.41, 0.0, 1.5, 0.08, 0.02)])  # hull y < 0: port
    prop = proposals.build(ch)
    port = [s for s in _of(prop, "seam") if s["side"] == "port"]
    named = sorted(s["name"] for s in port if not s["name"].endswith("?"))
    assert named == [f"comento B{k}" for k in range(1, 5)]
    strays = [s for s in port if s["name"].endswith("?")]
    assert strays and all(s["name"] == "comento B?" and s["score"] <= 0.1 for s in strays)
    assert all(5.2 < np.mean(np.asarray(s["ship_xy"])[:, 0]) < 7.0 for s in strays)
    for x in (4.0, 9.0):
        widths = strake_widths(x)
        cut = shapely.LineString([(x, -5.0), (x, 5.0)])
        for k in range(2, 6):
            width = sum(shapely.Polygon(b["ship_xy"]).intersection(cut).length
                        for b in _planks(prop, "port") if b["name"] == f"fascia B{k}")
            assert width == pytest.approx(widths[k - 1], rel=0.05)
    assert sorted({b["name"] for b in _planks(prop, "port")}) == [f"fascia B{k}" for k in range(1, 6)]
