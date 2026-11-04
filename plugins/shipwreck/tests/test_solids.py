"""Every traced timber a closed 3D object: the strakes as shells of their thickness with the spans under the frames
interpolated and flagged hidden, the frames and the keelson as solids from the hull under them up to their top, the
keel from typed siding and moulding as a stated hypothesis, one PLY per record; the records never touched beyond
their solid, volume and typed thickness."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import shapely
import trimesh
from hulls import FRAMES, KEELSON_HALF, KEELSON_HEIGHT, clinker_hull, depth, framed_project, \
    framed_truth, half_breadth

from ppg_shipwreck import clinker, relief, ship, solids, timbers, tracing
from ppg_shipwreck import proposals as proposals_mod
from ppg_shipwreck.ship import NotReady

X_RANGE = (3.0, 9.0)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("solids"), heading_deg=35, x_range=X_RANGE)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.005)
    proposals_mod.build(ch)
    items = ship.state(ch)["proposals"]["items"]
    tracing.accept(ch, [i["id"] for i in items])
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def hull(built):
    ch, snapshot = built
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _records(ch, kind):
    return [r for r in timbers.timbers_of(ch) if r["kind"] == kind]


def _named(ch, name):
    return next(r for r in timbers.timbers_of(ch) if r["name"] == name)


def _strake_truth(ch, record, thickness_mm):
    """Thickness times the area of the smooth hull's surface over the strake's polygon (a 2 mm grid in plan)."""
    poly = shapely.Polygon(tracing.ship_xy_of(ch, record))
    x0, y0, x1, y1 = poly.bounds
    step = 0.002
    gx, gy = np.meshgrid(np.arange(x0, x1, step) + step / 2, np.arange(y0, y1, step) + step / 2)
    inside = shapely.contains_xy(poly, gx, gy) & (np.abs(gy) < half_breadth(gx))
    x, y, h = gx[inside], gy[inside], 1e-5
    zx = (depth(x + h, y) - depth(x - h, y)) / (2 * h)
    zy = (depth(x, y + h) - depth(x, y - h)) / (2 * h)
    area = float(np.sqrt(1.0 + zx**2 + zy**2).sum() * step * step)
    return area * thickness_mm / 1000.0


def test_strake_shells_closed_and_thick(hull):
    ch = hull
    planks = _records(ch, "plank")
    assert len(planks) == 10
    for r in planks:
        mm, source = solids.thickness_of(ch, r)
        assert source == "record" and 13.0 < mm < 18.0
        s = solids.strake_shell(ch, r, mm)
        mesh = trimesh.Trimesh(s.vertices, s.faces, process=False)
        assert mesh.is_watertight, r["name"]
        assert mesh.volume == pytest.approx(s.volume, rel=1e-6) and s.volume > 0
        assert s.thickness_mm == pytest.approx(mm)
        assert s.kind == "plank" and s.name == r["name"] and s.status == "measured"
        truth = _strake_truth(ch, r, mm)
        assert s.volume == pytest.approx(truth, rel=0.05), (r["name"], s.volume, truth)


def test_hidden_span_under_a_frame(hull):
    ch = hull
    r = _named(ch, "fascia B3")
    mm, _ = solids.thickness_of(ch, r)
    s = solids.strake_shell(ch, r, mm)
    assert s.hidden.sum() > 0
    frames = [shapely.Polygon(tracing.ship_xy_of(ch, f)) for f in _records(ch, "frame")]
    under = shapely.union_all(frames).buffer(0.005 * 2 + mm / 1000.0)
    centres = s.vertices[s.faces[s.hidden]].mean(axis=1)
    assert shapely.contains_xy(under, centres[:, 0], centres[:, 1]).all()
    # the inner surface at a frame's x: the planking's height (the hull without timbers), not the frame's top
    from scipy.interpolate import griddata

    f = 6.3
    assert any(abs(x - f) < 1e-9 for x in FRAMES)
    v, _faces = clinker_hull()
    near = np.abs(v[:, 0] - f) < 0.05
    tri = s.vertices[s.faces[s.hidden]]
    up = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] > 0  # the inner face (outward = up)
    corners = s.vertices[np.unique(s.faces[s.hidden][up].ravel())]
    corners = corners[np.abs(corners[:, 0] - f) < 0.03]
    planking = griddata(np.column_stack([v[near, 0], -v[near, 1]]), v[near, 2], corners[:, :2])  # ship y = -hull y
    d = corners[:, 2] - planking
    inner = np.isfinite(d)  # every corner of the hidden inner faces: a sag is not filtered out
    assert inner.sum() > 10
    assert np.abs(d[inner]).max() < 0.005, np.abs(d[inner]).max()


def test_frame_and_keelson_volumes(hull):
    ch = hull
    truth = framed_truth()
    by_x = dict(zip(FRAMES, truth["volumes"]["frames"]))
    frames = _records(ch, "frame")
    assert len(frames) == 13
    for r in frames:
        s = solids.timber_solid(ch, r)
        assert trimesh.Trimesh(s.vertices, s.faces, process=False).is_watertight
        x = float(np.mean(tracing.ship_xy_of(ch, r)[:, 0]))
        expected = by_x[min(by_x, key=lambda f: abs(f - x))]
        assert s.volume == pytest.approx(expected, rel=0.05), (r["name"], s.volume, expected)
        assert not s.hidden.any() and s.thickness_mm is None
    (keelson,) = _records(ch, "keelson")
    s = solids.timber_solid(ch, keelson)
    assert trimesh.Trimesh(s.vertices, s.faces, process=False).is_watertight
    # the model holds the keelson between X_RANGE only (framed_truth's is the whole keelson, 1 to 11 m)
    expected = KEELSON_HEIGHT * 2.0 * KEELSON_HALF * (X_RANGE[1] - X_RANGE[0])
    assert truth["volumes"]["keelson"] > expected
    assert s.volume == pytest.approx(expected, rel=0.05)


def test_thickness_order(hull, monkeypatch):
    ch = hull
    r = _named(ch, "fascia B3")
    assert solids.thickness_of(ch, r, given=20.0) == (20.0, "given")
    assert solids.thickness_of(ch, r) == (pytest.approx(r["thickness_mm"]), "record")
    bare = {k: v for k, v in r.items() if k != "thickness_mm"}
    monkeypatch.setattr(clinker, "strake_rows", lambda chunk: [
        {"strake": "B3", "side": "port", "thickness_mm": 17.0}, {"strake": "T3", "side": "starboard",
                                                                 "thickness_mm": 19.0}])
    assert solids.thickness_of(ch, bare) == (17.0, "clinker")

    def no_model(chunk):
        raise NotReady("no strake model")

    monkeypatch.setattr(clinker, "strake_rows", no_model)
    mm, source = solids.thickness_of(ch, bare)
    steps = [_named(ch, n)["thickness_mm"] for n in ("comento B2", "comento B3")]
    assert source == "seams" and mm == pytest.approx(float(np.median(steps)))
    for seam in _records(ch, "seam"):
        seam.pop("thickness_mm", None)
    with pytest.raises(NotReady, match="spessore sconosciuto: scrivilo"):
        solids.thickness_of(ch, bare)


def test_keel_from_typed_dimensions(hull):
    ch = hull
    assert solids.keel_solid(ch) is None
    keel = solids.set_keel(ch, 0.12, 0.15)
    (keelson,) = _records(ch, "keelson")
    kx = tracing.ship_xy_of(ch, keelson)[:, 0]
    assert keel["x0"] == pytest.approx(kx.min()) and keel["x1"] == pytest.approx(kx.max())
    s = solids.keel_solid(ch)
    assert s.status == "reconstructed:given" and s.name == "chiglia" and s.kind == "keel"
    assert trimesh.Trimesh(s.vertices, s.faces, process=False).is_watertight
    assert s.volume == pytest.approx(0.12 * 0.15 * (keel["x1"] - keel["x0"]), rel=0.01)
    k = solids.timber_solid(ch, keelson)
    line = k.vertices[np.abs(k.vertices[:, 1]) < 0.003]  # the keelson's corners on the keel line
    xs = np.unique(np.round(line[:, 0], 6))
    lowest = np.array([line[np.round(line[:, 0], 6) == x, 2].min() for x in xs])
    top = s.vertices[s.vertices[:, 2] > s.vertices[:, 2].min() + 0.15 / 2]  # the keel's upper corners
    assert (top[:, 2] <= np.interp(top[:, 0], xs, lowest) + 1e-6).all()  # under the keelson's bottom
    mid = (top[:, 0] > 3.2) & (top[:, 0] < 8.8)  # away from the frame cut at the model's end (x 9)
    assert top[mid, 2].max() < -1.49  # the hull's keel line (z -1.5), not the keelson's top (-1.38)
    assert solids.set_keel(ch, 0.12, 0.15, x0=4.0, x1=8.0)["x0"] == 4.0
    assert solids.keel_solid(ch).volume == pytest.approx(0.12 * 0.15 * 4.0, rel=0.01)
    assert solids.set_keel(ch, None, None) is None and "keel" not in ship.state(ch)
    assert solids.keel_solid(ch) is None
    solids.set_keel(ch, 0.12, 0.15)
    timbers.add(ch, "keel", "centre", "chiglia misurata", timbers.geometry(
        "polygon", keelson["geometry"]["xy"]), status="measured")
    assert solids.keel_solid(ch) is None  # a measured keel replaces the typed one


def test_rebuild_keeps_records(hull):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    ch = hull
    transform, model = np.array(ch.transform), ch.product_path("model")
    seam = _records(ch, "seam")[0]
    b2 = _named(ch, "fascia B2")
    tracing.correct(ch, b2["id"], relief.uv_of_ship(tracing.ship_xy_of(ch, b2) + [0.0, 0.002],
                                                    ship.frame_of(ch).sign))
    b4 = _named(ch, "fascia B4")
    b4.pop("thickness_mm")
    for s in _records(ch, "seam"):
        s.pop("thickness_mm", None)
    before = copy.deepcopy(timbers.timbers_of(ch))
    out = solids.build(ch)
    assert {r["id"] for r in out["refused"]} == {b4["id"]}
    assert "spessore sconosciuto" in out["refused"][0]["reason"]
    assert {r["id"] for r in out["without_solid"]} >= {seam["id"]}
    assert out["keel"] is None
    built = {r["id"] for r in out["built"]}
    assert len(built) == len([r for r in before if r["geometry"]["type"] == "polygon"]) - 1
    for r in timbers.timbers_of(ch):
        if r["id"] in built:
            path = ch.data_dir / r["solid"]
            assert path.exists() and r["volume"] > 0
            mesh = PlyMesh(path)
            assert mesh.n_faces > 0 and "hidden" in mesh.face_dtype.names
            if r["kind"] == "plank":
                assert mesh._faces["hidden"].sum() > 0
        else:
            assert "solid" not in r and "volume" not in r

    def clean(records):
        return [{k: v for k, v in r.items() if k not in ("solid", "volume")} for r in records]

    assert clean(timbers.timbers_of(ch)) == clean(before)
    solids.set_keel(ch, 0.12, 0.15)
    again = solids.build(ch, thickness=16.0)
    assert not again["refused"] and again["keel"]["volume"] > 0
    assert (ch.data_dir / ship.state(ch)["keel"]["solid"]).exists()
    after = timbers.timbers_of(ch)
    assert next(r for r in after if r["id"] == b4["id"])["thickness_mm"] == 16.0
    assert next(r for r in after if r["id"] == b2["id"])["thickness_mm"] == pytest.approx(
        next(r for r in before if r["id"] == b2["id"])["thickness_mm"])  # a record's own thickness is never overwritten
    typed = {b["id"] for b in again["built"] if b["kind"] == "plank" and b["thickness_mm"] == 16.0}
    assert typed == {b4["id"]}  # the typed thickness only for the strake without one
    typed = {k: v for k, v in next(r for r in after if r["id"] == b4["id"]).items() if k != "thickness_mm"}
    assert clean([typed]) == clean([next(r for r in before if r["id"] == b4["id"])])
    assert clean([r for r in after if r["id"] != b4["id"]]) == clean([r for r in before if r["id"] != b4["id"]])
    timbers.remove(ch, [b2["id"]])
    solids.build(ch, thickness=16.0)
    assert not (ch.data_dir / "shipwreck" / "timbers" / f"{b2['id']}.ply").exists()
    assert np.array_equal(np.array(ch.transform), transform) and ch.product_path("model") == model


def test_a_collinear_ring_and_a_failing_record_do_not_stop_the_build(hull, monkeypatch):
    ch = hull
    frame = ship.frame_of(ch)
    sign = frame.sign
    # a thin timber at the model's end (x 9): the hull beyond is off the model and another timber takes the ring aft
    # of it but one column: its ring is one row of cells (qhull refuses a flat set)
    thin = tracing.add_traced(ch, "other", "auto", "listello", relief.uv_of_ship(
        [[8.985, 0.30], [8.995, 0.30], [8.995, 0.50], [8.985, 0.50]], sign), "polygon")
    tracing.add_traced(ch, "other", "auto", "tavola", relief.uv_of_ship(
        [[8.95, 0.30], [8.972, 0.30], [8.972, 0.50], [8.95, 0.50]], sign), "polygon")
    s = solids.timber_solid(ch, thin)
    assert trimesh.Trimesh(s.vertices, s.faces, process=False).is_watertight and np.isfinite(s.vertices).all()
    line = np.column_stack([np.full(5, 1.0), np.linspace(0.0, 1.0, 5)])
    at = np.array([[1.0, 0.25], [1.0, 2.0], [3.0, 0.5]])
    assert solids._scattered(line, line[:, 1] * 2.0, at) == pytest.approx([0.5, 2.0, 1.0])
    # any error of one record: that record refused with the reason, the others built, the state written
    victim = _named(ch, "ordinata 5")
    real = solids.timber_solid

    def failing(chunk, record):
        if record["id"] == victim["id"]:
            raise RuntimeError("boom")
        return real(chunk, record)

    solids.build(ch)
    assert (ch.data_dir / victim["solid"]).exists()
    monkeypatch.setattr(solids, "timber_solid", failing)
    ship.state(ch).pop("solids", None)
    out = solids.build(ch, thickness=15.0)
    refused = {r["id"]: r["reason"] for r in out["refused"]}
    assert refused == {victim["id"]: "RuntimeError: boom"}
    assert "solid" not in victim and not (ch.data_dir / "shipwreck" / "timbers" / f"{victim['id']}.ply").exists()
    assert ship.state(ch)["solids"]["refused"] == [victim["id"]]
    assert len(out["built"]) == len([r for r in timbers.timbers_of(ch) if r["geometry"]["type"] == "polygon"]) - 1
    with pytest.raises(NotReady, match="Spessore non valido"):
        solids.build(ch, thickness=0.0)
    with pytest.raises(NotReady, match="Spessore non valido"):
        solids.thickness_of(ch, _named(ch, "fascia B3"), given=-2.0)


def test_a_name_with_a_line_break_keeps_the_header(hull, tmp_path):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    s = solids.strake_shell(hull, _named(hull, "fascia B1"), 15.0)
    s.name = "fascia\nB1\r\nend_header"
    mesh = PlyMesh(solids.write_ply(tmp_path / "x.ply", s))
    assert mesh.n_faces == len(s.faces) and mesh._faces["hidden"].sum() == s.hidden.sum()


def test_every_frame_of_the_whole_hull_within_five_percent(tmp_path):
    """The end frames stand on a floor rising toward stem and stern: their footprint is read whole (proposals) and
    their bottom follows the floor's curve across the timber (:func:`solids._across`), not a chord."""
    from hulls import framed_truth

    _doc, ch, ends = framed_project(tmp_path, heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if i["kind"] in ("frame", "keelson")])
    out = solids.build(ch)
    assert not out["refused"]
    by_x = dict(zip(FRAMES, framed_truth()["volumes"]["frames"]))
    frames = _records(ch, "frame")
    assert len(frames) == len(FRAMES)
    for r in frames:
        x = float(np.mean(tracing.ship_xy_of(ch, r)[:, 0]))
        expected = by_x[min(by_x, key=lambda f: abs(f - x))]
        assert r["volume"] == pytest.approx(expected, rel=0.05), (r["name"], r["volume"], expected)


def test_the_bottom_across_a_timber_follows_a_curved_floor():
    w = np.array([-0.06, -0.055, -0.05, 0.05, 0.055, 0.06])
    z = 10.0 * w**2  # a floor curving up on both sides
    near, far = w < 0, w > 0
    at = np.linspace(-0.04, 0.04, 9)
    got = solids._across(w, z, near, far, at, 0.005)
    assert np.abs(got - 10.0 * at**2).max() < 5e-4  # the chord is up to 36 mm off
    flat = solids._across(w, np.zeros(6), near, far, at, 0.005)
    assert np.abs(flat).max() < 1e-12
