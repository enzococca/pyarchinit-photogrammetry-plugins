"""The mirror of the preserved side: a measured timber of one side reflected across the keel plane where the other
side has neither a record nor the model, as a reconstruction record (``reconstructed:mirror``) with its solid
reflected; the disagreement of the two sides per station where both are measured; the measured records untouched."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import shapely
import trimesh
from hulls import FRAME_ROOM, FRAMES, _project, clinker_hull, framed_hull, framed_project, seam_y

from ppg_shipwreck import reconstruct, relief, ship, solids, timbers, tracing
from ppg_shipwreck import proposals as proposals_mod

HOLE = 6.5  # the port side of the hull is lost from here to the stern end


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("mirror"), heading_deg=35,
                                    missing=(HOLE, 11.8, "port"))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    items = ship.state(ch)["proposals"]["items"]
    tracing.accept(ch, [i["id"] for i in items if i["side"] == "starboard" and "?" not in i["name"]])
    solids.build(ch)
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def hull(built):
    ch, snapshot = built
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _named(records, name):
    return [r for r in records if r["name"] == name]


def test_strakes_mirrored_where_port_is_lost(hull):
    ch = hull
    before = copy.deepcopy(timbers.timbers_of(ch))
    out = reconstruct.mirror(ch)
    assert out and all(r["status"] == "reconstructed:mirror" for r in out)
    assert ship.state(ch)["reconstruction"]["timbers"] == out
    assert timbers.timbers_of(ch) == before  # the measured records never touched
    assert all(r["id"].startswith("R") for r in out) and len({r["id"] for r in out}) == len(out)
    by_id = {r["id"]: r for r in before}
    planks = [r for r in out if r["kind"] == "plank"]
    assert {r["name"] for r in planks} == {"fascia B1", "fascia B2", "fascia B3", "fascia B4", "fascia B5"}
    for r in out:
        source = by_id[r["source"]]
        assert source["side"] == "starboard" and r["side"] == "port"
        xy = np.asarray(r["ship_xy"])
        assert xy[:, 0].min() > HOLE - 0.03, (r["name"], xy[:, 0].min())  # nothing over the preserved port side
        assert (xy[:, 1] > 0).all()
        # the geometry is the record in the chunk's coordinates
        frame = ship.frame_of(ch)
        back = frame.to_ship(np.column_stack([np.asarray(r["geometry"]["xy"]), np.zeros(len(xy))]))[:, :2]
        assert np.abs(back - xy).max() < 1e-6
    # the edges of the mirrored strakes on the seams of the lost side
    for k in range(2, 6):
        (r,) = _named(planks, f"fascia B{k}")
        poly = shapely.Polygon(r["ship_xy"])
        for x in (7.0, 8.0, 9.0):
            if not (poly.bounds[0] + 0.05 < x < poly.bounds[2] - 0.05):
                continue
            cut = poly.intersection(shapely.LineString([(x, 0.0), (x, 3.0)]))
            ys = np.asarray(shapely.get_coordinates(cut))[:, 1]
            inner = seam_y(x)[k - 2]
            assert abs(ys.min() - inner) < 0.01, (r["name"], x, ys.min(), inner)
            if k < 5:
                assert abs(ys.max() - seam_y(x)[k - 1]) < 0.01, (r["name"], x, ys.max())


def test_mirrored_solids_on_the_lost_planking(hull):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh
    from scipy.interpolate import griddata

    ch = hull
    out = reconstruct.mirror(ch)
    v, _f = clinker_hull()
    assert any(not r["clipped"] for r in out) and any(r["clipped"] for r in out)
    for r in out:
        path = ch.data_dir / r["solid"]
        assert path.exists() and path.parent.name == "reconstruction" and path.name == f"{r['id']}.ply"
        mesh = PlyMesh(path)
        verts = mesh.xyz()
        faces = np.concatenate(list(mesh.face_chunks()))
        hidden = np.asarray(mesh._faces["hidden"]).astype(bool)
        del mesh
        tm = trimesh.Trimesh(verts, faces, process=False)
        assert tm.is_watertight and tm.volume > 0, r["name"]  # outward normals after the reflection
        assert tm.volume == pytest.approx(r["volume"], rel=1e-6)
        assert (verts[:, 1] > 0).all()
        if not r["clipped"]:  # a whole timber: its solid is the original's, reflected
            source = next(x for x in timbers.timbers_of(ch) if x["id"] == r["source"])
            original = PlyMesh(ch.data_dir / source["solid"])
            assert np.allclose(verts, original.xyz() * [1.0, -1.0, 1.0])
            assert np.array_equal(faces, np.concatenate(list(original.face_chunks()))[:, ::-1])
            del original
        if r["kind"] != "plank":
            continue
        tri = verts[faces[~hidden]]
        up = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])[:, 2] > 0  # the inner face
        corners = verts[np.unique(faces[~hidden][up].ravel())]
        inner = shapely.Polygon(r["ship_xy"]).buffer(-0.012)  # off the laps at the edges (ramps of 2 cm, steps of 15 mm)
        corners = corners[shapely.contains_xy(inner, corners[:, 0], corners[:, 1])]
        walls = np.abs(corners[:, 0, None] - np.asarray(FRAMES)[None]).min(axis=1) > FRAME_ROOM / 2 + 0.02
        corners = corners[walls]  # the cells at a frame's sides read part of its wall, on either side alike
        near = (np.abs(v[:, 0] - corners[:, 0].mean()) < (np.ptp(corners[:, 0]) / 2 + 0.05)) & (v[:, 1] < 0)
        truth = griddata(np.column_stack([v[near, 0], -v[near, 1]]), v[near, 2], corners[:, :2])  # ship y = -hull y
        ok = np.isfinite(truth)
        assert ok.sum() > 10
        assert np.abs(corners[ok, 2] - truth[ok]).max() < 0.01, (r["name"], np.abs(corners[ok, 2] - truth[ok]).max())


def test_nothing_mirrored_over_the_model(hull):
    ch = hull
    frame = ship.frame_of(ch)
    # a starboard frame where port is preserved: its reflection lies on the model
    standing = tracing.add_traced(ch, "frame", "starboard", "ordinata prova", relief.uv_of_ship(
        [[3.56, -1.7], [3.64, -1.7], [3.64, -0.07], [3.56, -0.07]], frame.sign), "polygon")
    # a loose timber in the hole's mirror: never mirrored
    loose = tracing.add_traced(ch, "frame", "starboard", "legno sciolto", relief.uv_of_ship(
        [[8.25, -1.2], [8.40, -1.2], [8.40, -0.9], [8.25, -0.9]], frame.sign), "polygon")
    tracing.loose(ch, [loose["id"]])
    out = reconstruct.mirror(ch)
    sources = {r["source"] for r in out}
    assert standing["id"] not in sources and loose["id"] not in sources
    assert not any(r["side"] == "centre" for r in out)
    centre = {r["id"] for r in timbers.timbers_of(ch) if r["side"] == "centre"}
    assert not (centre & sources)
    frames = [r for r in out if r["kind"] == "frame"]
    assert len(frames) == 10  # ordinata 14 to 23, aft of the hole's start
    assert {r["name"] for r in frames} == {f"ordinata {n}" for n in range(14, 24)}  # frames keep their names
    assert all(np.asarray(r["ship_xy"])[:, 0].min() > HOLE for r in frames)


def test_a_record_of_the_other_side_stops_the_mirror(hull):
    ch = hull
    frame = ship.frame_of(ch)
    t20 = next(r for r in timbers.timbers_of(ch) if r["name"] == "ordinata 20")
    xy = tracing.ship_xy_of(ch, t20) * [1.0, -1.0]  # its port half traced by hand in the hole
    tracing.add_traced(ch, "frame", "port", "ordinata 20", relief.uv_of_ship(xy, frame.sign), "polygon")
    out = reconstruct.mirror(ch)
    assert t20["id"] not in {r["source"] for r in out}
    # mirroring again replaces the earlier mirror, files included
    first = reconstruct.mirror(ch)
    again = reconstruct.mirror(ch)
    assert [r["name"] for r in again] == [r["name"] for r in first]
    files = sorted(p.name for p in (ship.folder(ch) / "reconstruction").glob("*.ply"))
    assert files == sorted(f"{r['id']}.ply" for r in again)


def test_names_of_the_other_side():
    assert reconstruct.other_name("fascia B3") == "fascia T3"
    assert reconstruct.other_name("fascia T3") == "fascia B3"
    assert reconstruct.other_name("comento T12") == "comento B12"
    assert reconstruct.other_name("ordinata 7") == "ordinata 7"
    assert reconstruct.other_name("Bordo") == "Bordo"


def _wider_port(tmp_path, side: str = "port", amount: float = 0.03):
    """The framed hull with its ``side`` (port: the hull's y < 0) ``amount`` wider in plan between x 2 and 4 (ramps
    of 0.3 m)."""
    v, f = framed_hull()
    x = v[:, 0]
    w = np.clip(np.minimum(x - 1.7, 4.3 - x) / 0.3, 0.0, 1.0)
    v = v.copy()
    if side == "port":
        v[:, 1] -= amount * w * (v[:, 1] < 0)
    else:
        v[:, 1] += amount * w * (v[:, 1] > 0)
    from hulls import UTM

    return _project(tmp_path, v, f, 35.0, UTM, "EPSG:32633", 2.0)


def test_disagreement_where_both_sides_are_measured(tmp_path):
    _doc, ch, ends = _wider_port(tmp_path)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"]
                        if i["kind"] == "seam" and "?" not in i["name"]])
    ship.set_stations(ch, [{"name": f"S{k}", "x": x} for k, x in enumerate((2.5, 3.0, 3.5, 6.0, 8.0))])
    before = copy.deepcopy(timbers.timbers_of(ch))
    rows = reconstruct.disagreement(ch)
    assert ship.state(ch)["reconstruction"]["table"]["mirror"] == rows
    assert timbers.timbers_of(ch) == before
    assert rows and {"station", "x", "rank", "port_y", "starboard_y", "difference_mm"} <= set(rows[0])
    for row in rows:
        assert row["difference_mm"] == pytest.approx((row["port_y"] - row["starboard_y"]) * 1000.0)
        if 2.4 < row["x"] < 3.6:
            assert 25.0 < row["difference_mm"] < 35.0, row
        else:
            assert abs(row["difference_mm"]) < 5.0, row
    assert {r["station"] for r in rows} >= {"S0", "S1", "S2", "S3", "S4"}
    assert {r["rank"] for r in rows if r["station"] == "S3"} == {1, 2, 3, 4}


@pytest.mark.parametrize("amount", [0.03, 0.08])
def test_a_deformed_wreck_is_a_disagreement_not_a_mirror(tmp_path, amount):
    """Starboard read 3 or 8 cm wider than port over x 2-4: the strip of starboard's reflection beside the measured
    port timbers is the same timbers read wider, listed in the table, never a reconstructed timber."""
    _doc, ch, ends = _wider_port(tmp_path, "starboard", amount)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    ship.set_stations(ch, [{"name": f"S{k}", "x": x} for k, x in enumerate((2.5, 3.0, 3.5, 6.0, 8.0))])
    assert reconstruct.mirror(ch) == []
    rows = reconstruct.disagreement(ch)
    inside = [r for r in rows if 2.4 < r["x"] < 3.6]
    mm = 1000.0 * amount
    assert inside and all(-mm - 5.0 < r["difference_mm"] < -mm + 5.0 for r in inside), inside


def test_a_whole_lost_side_is_mirrored_whole(tmp_path):
    _doc, ch, ends = framed_project(tmp_path, heading_deg=35, missing=(0.0, 12.0, "port"))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"]
                        if i["side"] == "starboard" and "?" not in i["name"]])
    solids.build(ch)
    out = reconstruct.mirror(ch)
    assert out and not any(r["clipped"] for r in out)
    by_id = {r["id"]: r for r in timbers.timbers_of(ch)}
    for r in out:
        source = np.asarray(tracing.ship_xy_of(ch, by_id[r["source"]]))
        xy = np.asarray(r["ship_xy"])
        assert np.allclose(np.sort(xy[:, 0]), np.sort(source[:, 0]))  # the same vertices, the ends kept
        assert np.allclose(np.sort(xy[:, 1]), np.sort(-source[:, 1]))
        assert shapely.Polygon(xy).exterior.is_ccw
    (b1,) = _named(out, "fascia B1")
    t1 = by_id[b1["source"]]
    assert np.ptp(np.asarray(b1["ship_xy"])[:, 0]) == pytest.approx(np.ptp(tracing.ship_xy_of(ch, t1)[:, 0]))


def test_a_strake_measured_up_to_the_loss_goes_on_mirrored(hull):
    """Port B5 measured up to the lost part: the mirror of T5 meets it end to end and goes on aft of it."""
    ch = hull
    items = ship.state(ch)["proposals"]["items"]
    (b5,) = [i["id"] for i in items if i["name"] == "fascia B5"]
    tracing.accept(ch, [b5])
    out = reconstruct.mirror(ch)
    (r,) = _named(out, "fascia B5")
    xy = np.asarray(r["ship_xy"])
    assert xy[:, 0].min() > HOLE - 0.03 and r["clipped"]


def test_a_mirror_stopped_halfway_keeps_the_earlier_one(hull, monkeypatch):
    ch = hull
    first = reconstruct.mirror(ch)
    files = sorted(p.name for p in reconstruct.folder(ch).glob("*.ply"))
    kept = copy.deepcopy(reconstruct.records_of(ch))

    def broken(*args, **kwargs):
        raise RuntimeError("stopped")

    monkeypatch.setattr(reconstruct, "_others", broken)
    with pytest.raises(RuntimeError):
        reconstruct.mirror(ch)
    assert reconstruct.records_of(ch) == kept and len(kept) == len(first)
    assert sorted(p.name for p in reconstruct.folder(ch).glob("*.ply")) == files
    assert not (ship.folder(ch) / "reconstruction.new").exists()


def _lost_project(tmp_path, lost):
    """The framed hull with the port faces (the hull's y < 0) whose centre ``lost(x, |y|)`` says taken away."""
    from hulls import UTM

    v, f = framed_hull()
    c = v[f].mean(axis=1)
    drop = (c[:, 1] < 0) & lost(c[:, 0], np.abs(c[:, 1]))
    _doc, ch, ends = _project(tmp_path, v, f[~drop], 35.0, UTM, "EPSG:32633", 2.0)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    solids.build(ch)
    return ch


def test_an_eroded_sheer_is_mirrored(tmp_path):
    """Port's upper part of the sheer strake gone over x 3-9 (beyond seam 4 + 0.12 m), its lower part measured: the
    lost part, wide, alongside the measured one, is mirrored."""
    from hulls import plan_half_breadth

    def lost(x, y):
        edge = np.array([seam_y(v)[3] for v in x]) + 0.12
        return (x > 3.0) & (x < 9.0) & (y > edge)

    ch = _lost_project(tmp_path, lost)
    out = reconstruct.mirror(ch)
    (b5,) = _named(out, "fascia B5")
    assert [r["name"] for r in out] == ["fascia B5"]  # the frames are centre timbers, B1-B4 are whole on port
    poly = shapely.Polygon(b5["ship_xy"])
    x0, _y0, x1, _y1 = poly.bounds
    assert abs(x0 - 3.0) < 0.05 and abs(x1 - 9.0) < 0.05, (x0, x1)
    cut = np.asarray(shapely.get_coordinates(poly.intersection(shapely.LineString([(6.0, 0.0), (6.0, 3.0)]))))
    assert abs(cut[:, 1].min() - (seam_y(6.0)[3] + 0.12)) < 0.03, cut
    assert abs(cut[:, 1].max() - plan_half_breadth(6.0)) < 0.03, cut
    assert b5["clipped"] and timbers.timbers_of(ch)  # the measured lower part stays as it is


def test_an_oblique_loss_is_mirrored(tmp_path):
    """Port lost beyond x > 5.5 + 1.5 |y|, a boundary oblique to the strakes: every strake goes on mirrored, the
    narrow garboard too."""
    ch = _lost_project(tmp_path, lambda x, y: x > 5.5 + 1.5 * y)
    out = reconstruct.mirror(ch)
    names = {r["name"] for r in out if r["kind"] == "plank"}
    assert names == {f"fascia B{k}" for k in range(1, 6)}, names
    (b1,) = _named(out, "fascia B1")
    xy = np.asarray(b1["ship_xy"])
    assert abs(xy[:, 0].min() - (5.5 + 1.5 * 0.12)) < 0.1 and xy[:, 0].max() > 9.3, (xy[:, 0].min(), xy[:, 0].max())
    for r in out:
        xy = np.asarray(r["ship_xy"])
        assert (xy[:, 0] > 5.5 + 1.5 * xy[:, 1] - 0.05).all(), r["name"]  # nothing over the preserved port side
