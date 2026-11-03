"""The traced timbers: proposals accepted into the catalogue with their geometry in the chunk's coordinates and in the
ship plan, rejected, traced and corrected by hand on the relief, renamed from a table, their visible parts cut by the
frames; the status codes, and the records of 0.2.0 read as they were."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import shapely
from hulls import FRAMES, framed_project, project

from ppg_shipwreck import clinker, proposals, relief, ship, timbers, tracing
from ppg_shipwreck.ship import NotReady


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("tracing"), heading_deg=35, x_range=(3.0, 9.0))
    frame = ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.005)
    proposals.build(ch)
    return ch, frame, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def hull(built):
    """The chunk with its proposals as first read (each test starts from them)."""
    ch, frame, snapshot = built
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch, frame


def _items(ch, kind):
    return [i for i in ship.state(ch)["proposals"]["items"] if i["kind"] == kind]


def _ship(frame, record):
    w = np.asarray(record["geometry"]["xy"], float)
    return frame.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2]


def test_accept_writes_geometry_and_ship_xy(hull):
    ch, frame = hull
    fr = sorted(_items(ch, "frame"), key=lambda i: np.mean(np.asarray(i["ship_xy"])[:, 0]))[3]
    seam = _items(ch, "seam")[0]
    records = tracing.accept(ch, [fr["id"], seam["id"]], names={fr["id"]: "F12"})
    assert [r["kind"] for r in records] == ["frame", "seam"]
    a, b = records
    assert a["name"] == "F12" and b["name"] == seam["name"]
    assert a["geometry"]["type"] == "polygon" and b["geometry"]["type"] == "line"
    for r, item in ((a, fr), (b, seam)):
        assert np.abs(_ship(frame, r) - np.asarray(r["ship_xy"])).max() < 1e-9
        assert np.asarray(r["ship_xy"]) == pytest.approx(np.asarray(item["ship_xy"]), abs=1e-12)
        assert r["status"] == "measured" and timbers.is_measured(r)
        assert r["chunk"] == ch.key and r["source"] == item["id"]
        assert r["elevation"] == relief.entry(ch)["id"]
    assert b["thickness_mm"] == pytest.approx(seam["step_mm"])
    left = {i["id"] for i in ship.state(ch)["proposals"]["items"]}
    assert fr["id"] not in left and seam["id"] not in left
    row = next(r for r in timbers.measure(ch) if r["id"] == a["id"])
    assert row["width"] == pytest.approx(0.08, abs=0.01)
    assert row["status"] == "measured" and row["chunk"] == ch.key and row["loose"] == 0
    with pytest.raises(NotReady):
        tracing.accept(ch, [fr["id"]])  # already taken


def test_reject_keeps_the_geometry_and_nothing_comes_back(hull):
    ch, _frame = hull
    plank = _items(ch, "plank")[0]
    assert tracing.reject(ch, [plank["id"]]) == 1
    rejected = ship.state(ch)["proposals"]["rejected"]
    assert shapely.from_wkt(rejected[-1]).equals_exact(shapely.Polygon(plank["ship_xy"]), 1e-9)
    assert plank["id"] not in {i["id"] for i in ship.state(ch)["proposals"]["items"]}
    again = proposals.build(ch)
    assert not any(i["kind"] == "plank" and i["name"] == plank["name"] and
                   shapely.Polygon(i["ship_xy"]).intersection(shapely.Polygon(plank["ship_xy"])).area >
                   0.5 * shapely.Polygon(i["ship_xy"]).area for i in again["items"])


def test_mirror_frame_traced_polygon_is_port(tmp_path):
    """Bow at the start (the frame a mirror image, ``sign`` -1): v = -y, so a polygon traced at v < 0 on the relief
    lies at y > 0 in the ship plan, on port; corrected, it keeps its id."""
    _doc, ch, ends = project(tmp_path, heading_deg=35)
    frame = ship.set_frame(ch, ship.Frame(ends[0], ends[1], "end").other_bow())
    assert frame.sign == -1
    uv = [[5.0, -0.6], [6.0, -0.6], [6.0, -0.4], [5.0, -0.4]]
    r = tracing.add_traced(ch, "plank", "auto", "fascia B3", uv, "polygon")
    assert r["side"] == "port" and r["status"] == "measured" and r["chunk"] == ch.key
    xy = np.asarray(r["ship_xy"])
    assert (xy[:, 1] > 0).all()
    assert np.abs(xy - relief.ship_of_uv(uv, frame.sign)).max() < 1e-12
    assert np.abs(_ship(frame, r) - xy).max() < 1e-9
    assert tracing.add_traced(ch, "frame", "auto", "F1", [[4.0, -0.02], [4.0, 0.03]], "line")["side"] == "centre"
    assert tracing.add_traced(ch, "frame", "auto", "F2", [[4.5, 0.2], [4.5, 0.9]], "line")["side"] == "starboard"
    fixed = tracing.correct(ch, r["id"], [[5.0, -0.62], [6.1, -0.62], [6.1, -0.4], [5.0, -0.4]])
    assert fixed["id"] == r["id"] and fixed["corrected"] is True and fixed["name"] == "fascia B3"
    assert np.asarray(fixed["ship_xy"])[:, 0].max() == pytest.approx(6.1)
    assert sum(1 for x in timbers.timbers_of(ch) if x["id"] == r["id"]) == 1
    with pytest.raises(NotReady):
        tracing.correct(ch, "T99", uv)


def test_rename_frames_from_a_table(hull, tmp_path):
    ch, _frame = hull
    tracing.accept(ch, [i["id"] for i in _items(ch, "frame")])
    frames = sorted((r for r in timbers.timbers_of(ch) if r["kind"] == "frame"),
                    key=lambda r: np.mean(np.asarray(r["ship_xy"])[:, 0]))
    xs = [float(np.mean(np.asarray(r["ship_xy"])[:, 0])) for r in frames]
    assert len(frames) == sum(1 for f in FRAMES if 3.0 < f < 9.0)
    table = tmp_path / "ordinate.csv"
    rows = ["nome;x"] + [f"F{20 + k};{str(round(x + 0.03, 3)).replace('.', ',')}" for k, x in enumerate(xs[:5])]
    rows.append("F99;50,0")  # far from every frame
    table.write_text("\n".join(rows) + "\n", encoding="utf-8")
    assert tracing.rename_frames(ch, table) == (5, 1)
    named = {r["id"]: r["name"] for r in timbers.timbers_of(ch)}
    assert [named[r["id"]] for r in frames[:5]] == [f"F{20 + k}" for k in range(5)]
    assert all(named[r["id"]] == r["name"] for r in frames[5:])


def test_visible_parts_cut_by_frames(hull):
    """A strake proposed runs under the frames: its visible parts are its polygon minus the frames', keelson's and
    stringers' polygons (rings oriented, the signed areas add up)."""
    ch, _frame = hull
    plank = next(i for i in _items(ch, "plank") if i["name"].endswith("B3"))
    cutters = _items(ch, "frame") + _items(ch, "keelson")
    tracing.accept(ch, [plank["id"]] + [i["id"] for i in cutters])
    assert tracing.update_visible(ch) == 1
    record = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank")
    rings = record["visible_xy"]
    assert len(rings) > 5  # cut in pieces by the frames
    area = sum(shapely.Polygon(r).area * (1 if shapely.LinearRing(r).is_ccw else -1) for r in rings)
    truth = shapely.Polygon(plank["ship_xy"]).difference(shapely.union_all([shapely.Polygon(i["ship_xy"]) for i in cutters]))
    assert area == pytest.approx(truth.area, rel=0.02)
    assert area < 0.95 * shapely.Polygon(plank["ship_xy"]).area
    assert tracing.loose(ch, [record["id"]]) == 1
    flagged = next(r for r in timbers.timbers_of(ch) if r["id"] == record["id"])
    assert flagged["kind"] == "other" and flagged["loose"] is True
    assert tracing.loose(ch, [record["id"]], flag=False) == 1
    assert next(r for r in timbers.timbers_of(ch) if r["id"] == record["id"])["loose"] is False


def test_old_catalogue_records_read_as_before(tmp_path):
    """Records of 0.2.0 (no status): the section strakes read as a hypothesis from the sections, the others as
    measured; measuring the catalogue rewrites nothing."""
    _doc, ch, ends = project(tmp_path, heading_deg=35)
    frame = ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    world = frame.to_world(np.array([[3.0, 0.1, 0.0], [5.0, 0.1, 0.0], [5.0, 0.4, 0.0], [3.0, 0.4, 0.0]]))[:, :2]
    old = [{"id": "T1", "kind": "plank", "side": "port", "name": "fascia B2", "station": "",
            "geometry": timbers.geometry("polygon", world), "notes": clinker.CATALOGUE_NOTES[0]},
           {"id": "T2", "kind": "plank", "side": "port", "name": "fascia B2", "station": "",
            "geometry": timbers.geometry("polygon", world), "notes": clinker.CATALOGUE_NOTES[1]},
           {"id": "T3", "kind": "frame", "side": "port", "name": "F1", "station": "S1",
            "geometry": timbers.geometry("point", world[:1]), "notes": "misurata"}]
    ship.state(ch)["timbers"] = copy.deepcopy(old)
    assert [timbers.status_of(r) for r in old] == ["reconstructed:sections", "reconstructed:sections", "measured"]
    assert [timbers.is_measured(r) for r in old] == [False, False, True]
    rows = timbers.measure(ch)
    assert [r["status"] for r in rows] == ["reconstructed:sections", "reconstructed:sections", "measured"]
    assert [r["chunk"] for r in rows] == ["", "", ""] and [r["loose"] for r in rows] == [0, 0, 0]
    out = timbers.export_csv(ch, tmp_path / "legni.csv")
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {"status", "chunk", "loose"} <= set(header)
    assert timbers.timbers_of(ch) == old
    assert timbers.status_title("reconstructed:sections") == "ipotesi dalle sezioni"
    assert set(timbers.STATUS) == {"measured", "reconstructed:given", "reconstructed:sections", "reconstructed:mirror",
                                   "reconstructed:extension", "reconstructed:spacing", "reconstructed:fairing"}


def test_add_keeps_the_new_fields_and_refuses_unknown_ones(tmp_path):
    _doc, ch, ends = project(tmp_path, heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    g = timbers.geometry("point", [[0.0, 0.0]])
    r = timbers.add(ch, "keel", "centre", "K", g, status="reconstructed:given", thickness_mm=12.0, sources=["a", "b"])
    assert r["status"] == "reconstructed:given" and r["thickness_mm"] == 12.0 and r["sources"] == ["a", "b"]
    old = timbers.add(ch, "frame", "port", "F1", g, "S1", "nota")  # the calls of 0.2.0
    assert "status" not in old and old["station"] == "S1" and old["notes"] == "nota"
    with pytest.raises(ValueError):
        timbers.add(ch, "frame", "port", "F2", g, colour="red")
    with pytest.raises(NotReady):
        timbers.add(ch, "frame", "port", "F2", g, status="guessed")
