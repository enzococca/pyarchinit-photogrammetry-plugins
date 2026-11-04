"""Several chunks of the same wreck: two campaigns of the framed hull (x 0.2-7.0 and 6.0-11.8), the second placed in
the world by another transform, aligned to the first with the program's *Allinea chunk* on four markers of the same
names; the ship frame shared, the records merged in the ship plan (one per strake, the overlap's outline from the
finer chunk), the reconstruction over the whole surveyed length, only the main chunk's state written."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import shapely
from hulls import FRAMES, UTM, depth, framed_hull, transform

import pyarchinit_photogrammetry as ppg
from pyarchinit_photogrammetry.io.mesh_ply import write_mesh_ply
from ppg_shipwreck import chunks, lines, reconstruct, relief, ship, timbers, tracing
from ppg_shipwreck import proposals as proposals_mod

MARKERS = {"M1": (6.2, 0.6), "M2": (6.8, -0.9), "M3": (6.5, 1.3), "M4": (6.4, -0.3)}  # on the hull, in the overlap
KEEL = {"K1": (6.1, 0.0), "K2": (6.9, 0.0)}  # on the keel line
STATIONS = np.arange(1.125, 11.0, 0.45)


def _hull_point(x, y):
    return np.array([x, y, float(depth(x, y))])


def _add_chunk(doc, label, x_range, heading, offset, scale, markers=MARKERS):
    """A chunk of ``doc`` with the framed hull's part ``x_range`` as its model, placed by ``transform(heading, scale,
    offset)``, and ``markers`` at the hull's points (internal coordinates: the same world point in every chunk once
    the chunks are aligned)."""
    ch = doc.add_chunk(label)
    ch.crs = "EPSG:32633"
    ch.transform = transform(heading, scale, offset)
    v, f = framed_hull(x_range=x_range)
    write_mesh_ply(ch.data_dir / "model.ply", v / scale, f)
    ch._set_product("model", ch.data_dir / "model.ply")
    for name, (x, y) in markers.items():
        ch.add_marker(name, position=_hull_point(x, y) / scale)
    return ch


def _frame_names(tmp_path):
    path = tmp_path / "ordinate.csv"
    path.write_text("name,x\n" + "".join(f"ordinata {k + 1},{x:.3f}\n" for k, x in enumerate(FRAMES)))
    return path


def _survey(ch, cell, names):
    relief.build(ch, cell=cell)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    tracing.rename_frames(ch, names)  # the field's numbers: each chunk numbers its own frames from 1


@pytest.fixture(scope="module")
def wreck(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("chunks")
    doc = ppg.Document()
    doc.save(tmp / "relitto.ppg")
    a = _add_chunk(doc, "Campagna A", (0.0, 7.0), 35.0, UTM, 2.0)
    b = _add_chunk(doc, "Campagna B", (6.0, 12.0), 80.0, UTM + [37.5, -12.25, 3.0], 1.6)
    report = doc.align_chunks(a, [b], method="markers")
    assert report["chunks"][0]["rmse"] < 1e-6
    ends = a.to_world(np.array([[0.0, 0.0, 0.0], [12.0 / 2.0, 0.0, 0.0]]))
    ship.set_frame(a, ship.Frame(ends[0], ends[1], "start"))
    chunks.use_frame_of(b, a)
    names = _frame_names(tmp)
    _survey(a, 0.01, names)
    _survey(b, 0.008, names)
    ship.set_stations(a, [{"name": f"S{k}", "x": float(x)} for k, x in enumerate(STATIONS)])
    lines.build(a)
    return doc, a, b


@pytest.fixture(scope="module")
def full(tmp_path_factory):
    """The whole hull surveyed as one chunk, in A's world: the strakes a single campaign would give."""
    tmp = tmp_path_factory.mktemp("full")
    doc = ppg.Document()
    doc.save(tmp / "relitto.ppg")
    ch = _add_chunk(doc, "Tutto", (0.0, 12.0), 35.0, UTM, 2.0)
    ends = ch.to_world(np.array([[0.0, 0.0, 0.0], [12.0 / 2.0, 0.0, 0.0]]))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    _survey(ch, 0.01, _frame_names(tmp))
    return ch


def _poly(r):
    return shapely.Polygon(r["ship_xy"])


def test_the_frame_of_an_aligned_chunk(wreck):
    _doc, a, b = wreck
    assert b.meta["chunk_alignment"]["reference_key"] == a.key
    fa, fb = ship.state(a)["frame"], ship.state(b)["frame"]
    assert np.allclose(fa["start"], fb["start"]) and np.allclose(fa["end"], fb["end"]) and fa["bow"] == fb["bow"]
    assert chunks.compatible(a, b) == (True, "") and chunks.compatible(b, a)[0]
    # the same hull point read in either chunk's ship frame
    p = _hull_point(6.5, 1.3)
    wa = a.to_world((p / 2.0)[None])
    wb = b.to_world((p / 1.6)[None])
    assert np.abs(ship.frame_of(a).to_ship(wa) - ship.frame_of(b).to_ship(wb)).max() < 1e-6


def test_members_and_merged_records(wreck, full):
    doc, a, b = wreck
    members, refused = chunks.members(doc, a)
    assert [m["chunk"] for m in members] == [a, b] and refused == []
    assert members[1]["cell"] == pytest.approx(0.008) and members[0]["cell"] == pytest.approx(0.01)
    assert all(0.4 < m["coverage"] < 0.7 for m in members), [m["coverage"] for m in members]
    a_before, b_before = copy.deepcopy(timbers.timbers_of(a)), copy.deepcopy(ship.state(b))
    out = chunks.merged(doc, a)
    assert timbers.timbers_of(a) == a_before and ship.state(b) == b_before  # read only
    own = {k: [r for r in timbers.timbers_of(c) if r["kind"] == "plank" and timbers.is_measured(r)] for k, c in
           (("a", a), ("b", b))}
    planks = [r for r in out if r["kind"] == "plank"]
    names = sorted(r["name"] for r in planks)
    assert names == sorted({r["name"] for r in own["a"]} | {r["name"] for r in own["b"]}), names
    assert len(names) == len(set(names))  # one record per strake
    for r in planks:
        assert {s["chunk"] for s in r["sources"]} == {a.key, b.key}, (r["name"], r["sources"])
        poly = _poly(r)
        assert poly.is_valid and poly.geom_type == "Polygon"
        (ra,) = [x for x in own["a"] if x["name"] == r["name"]]
        (rb,) = [x for x in own["b"] if x["name"] == r["name"]]
        lo, hi = _poly(ra).bounds[0], _poly(rb).bounds[2]
        assert abs(poly.bounds[0] - lo) < 0.02 and abs(poly.bounds[2] - hi) < 0.02, (r["name"], poly.bounds, lo, hi)
        assert abs((poly.bounds[2] - poly.bounds[0]) - (hi - lo)) < 0.02  # the full strake: A's start to B's end
        # the strake of the whole hull surveyed at once: the same area (its narrow ends are read where each survey
        # finds them, decimetres apart), the same breadth across the overlap
        (whole,) = [x for x in timbers.timbers_of(full) if x["kind"] == "plank" and x["name"] == r["name"]]
        assert abs(poly.area - _poly(whole).area) < 0.02 * _poly(whole).area, (r["name"], poly.area, _poly(whole).area)
        for x in (6.2, 6.5, 6.8):
            cut = [shapely.get_coordinates(g.intersection(shapely.LineString([(x, -5.0), (x, 5.0)])))[:, 1]
                   for g in (poly, _poly(whole))]
            assert np.abs(np.sort(np.abs(cut[0]))[[0, -1]] - np.sort(np.abs(cut[1]))[[0, -1]]).max() < 0.01, \
                (r["name"], x, cut)
        # the geometry is the record in the main chunk's coordinates
        back = ship.frame_of(a).to_ship(np.column_stack([np.asarray(r["geometry"]["xy"]), np.zeros(len(r["ship_xy"]))]))
        assert np.abs(back[:, :2] - np.asarray(r["ship_xy"])).max() < 1e-6
        assert r["id"] == ra["id"]  # the main's id
    # the frames of the overlap: the finer chunk's outline
    frames = [r for r in out if r["kind"] == "frame" and 6.0 < _poly(r).centroid.x < 7.0]
    assert frames
    for r in frames:
        assert len(r["sources"]) == 2 and r["sources"][0]["chunk"] == b.key, r["sources"]
        (rb,) = [x for x in timbers.timbers_of(b) if x["kind"] == "frame" and x["name"] == r["name"]
                 and x["side"] == r["side"]]
        assert _poly(r).symmetric_difference(_poly(rb)).area < 0.02 * _poly(rb).area, r["name"]


def test_the_reconstruction_over_the_merged_wreck(wreck):
    doc, a, b = wreck
    b_meta = json.dumps(ship.state(b), sort_keys=True)
    b_files = {p.relative_to(b.data_dir): p.read_bytes() for p in b.data_dir.rglob("*") if p.is_file()}
    a_records = copy.deepcopy(timbers.timbers_of(a))
    out = reconstruct.build(a)
    assert json.dumps(ship.state(b), sort_keys=True) == b_meta and "reconstruction" not in ship.state(b)
    assert {p.relative_to(b.data_dir): p.read_bytes() for p in b.data_dir.rglob("*") if p.is_file()} == b_files
    assert timbers.timbers_of(a) == a_records
    st = reconstruct.state(a)
    assert [m["chunk"] for m in st["members"]["chunks"]] == [a.key, b.key]
    assert st["members"]["refused"] == [] and out["members"]["chunks"] == st["members"]["chunks"]
    assert reconstruct.height_path(a) != relief.path_of(a, "height") and reconstruct.height_path(a).exists()
    lo, hi = reconstruct.surveyed_range(a)
    assert abs(lo - 0.2) < 0.03 and abs(hi - 11.8) < 0.03, (lo, hi)
    assert st["extension"]["surveyed"] == pytest.approx([lo, hi])
    overlap = shapely.box(6.0, -5.0, 7.0, 5.0)
    extended = [r for r in reconstruct.records_of(a) if r["status"] == reconstruct.EXTENSION]
    assert not [r["name"] for r in extended if _poly(r).intersects(overlap)]
    # the merged raster: the finer cell, B's heights where B is, A's elsewhere
    z, af = reconstruct.height_load(a)
    assert abs(af.a) == pytest.approx(0.008)
    zb, bf = relief.load(b, "height")
    row, col, ok = relief.cells_at(zb.shape, bf, ship.frame_of(b).sign, np.array([8.0]), np.array([0.3]))
    rm, cm, okm = relief.cells_at(z.shape, af, ship.frame_of(a).sign, np.array([8.0]), np.array([0.3]))
    assert ok[0] and okm[0] and z[rm[0], cm[0]] == pytest.approx(zb[row[0], col[0]], abs=1e-6)


def test_frames_from_markers_of_the_same_names(tmp_path):
    """Two chunks never aligned, the ship frame of each from its own markers K1, K2: the frames coincide by
    construction (the markers are the same physical points) and the chunks are accepted."""
    doc = ppg.Document()
    doc.save(tmp_path / "relitto.ppg")
    markers = {**MARKERS, **KEEL}
    c = _add_chunk(doc, "Uno", (5.5, 7.5), 10.0, UTM, 2.0, markers)
    d = _add_chunk(doc, "Due", (5.5, 7.5), 200.0, UTM + [100.0, 50.0, -2.0], 1.25, markers)
    assert chunks.compatible(c, d)[0] is False
    for ch in (c, d):
        ship.set_frame(ch, ship.frame_from_markers(ch, "K1", "K2", "start", None, 6.1), markers=["K1", "K2"])
    assert ship.state(c)["frame_markers"] == ["K1", "K2"]
    assert chunks.compatible(c, d) == (True, "")
    p = _hull_point(6.5, 1.3)
    sc = ship.frame_of(c).to_ship(c.to_world((p / 2.0)[None]))
    sd = ship.frame_of(d).to_ship(d.to_world((p / 1.25)[None]))
    assert np.abs(sc - sd).max() < 1e-6
    # use_frame_of from markers: a third chunk gets the frame from its own markers
    e = _add_chunk(doc, "Tre", (5.5, 7.5), 300.0, UTM + [-20.0, 9.0, 1.0], 3.0, markers)
    frame = chunks.use_frame_of(e, c)
    assert ship.state(e)["frame_markers"] == ["K1", "K2"] and frame == ship.state(e)["frame"]
    assert chunks.compatible(c, e) == (True, "")
    # a frame set again by hand forgets its markers
    ship.set_frame(c, ship.frame_of(c).flipped())
    assert "frame_markers" not in ship.state(c) and chunks.compatible(c, d)[0] is False


def test_a_third_chunk_refused_until_aligned_then_merged_without_touching_the_others(wreck):
    """An unaligned chunk is refused with the reason; aligned, framed and surveyed it joins the next build, and the
    main's catalogue (a record corrected by hand among it) and the other members are as they were."""
    doc, a, b = wreck
    c = _add_chunk(doc, "Campagna C", (3.0, 4.6), 150.0, UTM + [-41.0, 22.0, -1.0], 2.5)
    ok, reason = chunks.compatible(a, c)
    assert not ok and reason == ("Allinea prima il chunk «Campagna C» a «Campagna A» (Flusso di lavoro > Allinea "
                                 "chunk) o definisci il sistema della nave con gli stessi marker")
    with pytest.raises(ship.NotReady):
        chunks.use_frame_of(c, a)
    _members, refused = chunks.members(doc, a)
    assert [r["chunk"] for r in refused] == [c] and refused[0]["reason"] == reason
    reconstruct.build(a)
    assert reconstruct.state(a)["members"]["refused"] == [{"chunk": c.key, "label": "Campagna C", "reason": reason}]
    # a record corrected by hand in the main chunk
    frame = ship.frame_of(a)
    strake = next(r for r in timbers.timbers_of(a) if r["kind"] == "plank" and r["side"] == "starboard")
    xy = tracing.ship_xy_of(a, strake, frame) + [0.0, -0.003]
    tracing.correct(a, strake["id"], relief.uv_of_ship(xy, frame.sign))
    doc.align_chunks(a, [c], method="markers")
    chunks.use_frame_of(c, a)
    _survey(c, 0.01, _frame_names(doc.path.parent))
    snapshots = {ch.key: json.dumps({k: v for k, v in ship.state(ch).items() if k != "reconstruction"}, sort_keys=True)
                 for ch in (a, b, c)}
    a_records = copy.deepcopy(timbers.timbers_of(a))
    files = {ch.key: {p.relative_to(ch.data_dir): p.read_bytes() for p in ch.data_dir.rglob("*") if p.is_file()}
             for ch in (b, c)}
    out = reconstruct.build(a)
    assert [m["chunk"] for m in out["members"]["chunks"]] == [a.key, b.key, c.key]
    assert timbers.timbers_of(a) == a_records
    for ch in (a, b, c):
        assert json.dumps({k: v for k, v in ship.state(ch).items() if k != "reconstruction"},
                          sort_keys=True) == snapshots[ch.key], ch.label
    for ch in (b, c):
        assert "reconstruction" not in ship.state(ch)
        assert {p.relative_to(ch.data_dir): p.read_bytes() for p in ch.data_dir.rglob("*") if p.is_file()} \
            == files[ch.key], ch.label
    # the corrected record keeps its id among the merged ones
    merged = {r["id"]: r for r in chunks.merged(doc, a)}
    assert strake["id"] in merged
    # a member gone: the build goes on with the others and says so
    doc.chunks.remove(c)
    out = reconstruct.build(a)
    assert [m["chunk"] for m in out["members"]["chunks"]] == [a.key, b.key]
    assert any("Campagna C" in w for w in out["warnings"]), out["warnings"]
