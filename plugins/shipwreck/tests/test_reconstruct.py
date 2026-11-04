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
    assert ship.state(ch)["reconstruction"]["disagreement"] == rows
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


# ----------------------------------------------------------------------------- extension and frame spacing
def _seam_z(x, k: int) -> float:
    """The ship z of the land ``k`` (0 from the keel) of the framed hull at ``x``: its centre on the clinker section."""
    from hulls import _lands, _section_points

    centres, _active = _lands(x, 5, None)
    offsets = 0.015 * (np.arange(len(centres)) + 0.5)
    return float(_section_points(x, centres, offsets, 1.0)[1][k])


def _land_lines(stations):
    """The 0.2.0 land lines as the sections would read them on this hull: the lands' centres at ``stations``."""
    out = []
    for side, sign in (("port", 1.0), ("starboard", -1.0)):
        for k in range(1, 5):
            out.append({"side": side, "k": k, "uncertain_stations": [], "unmatched": [],
                        "points": [[x, sign * seam_y(x)[k - 1], _seam_z(x, k - 1)] for x in stations]})
    return out


def _clip(ch, x0: float, x1: float, kinds=("plank", "seam")) -> None:
    """The accepted records of ``kinds`` traced again on x0 <= x <= x1 only (as if the rest were never traced)."""
    frame = ship.frame_of(ch)
    box = shapely.box(x0, -5.0, x1, 5.0)
    for r in list(timbers.timbers_of(ch)):
        if r["kind"] not in kinds:
            continue
        xy = tracing.ship_xy_of(ch, r, frame)
        shape = (shapely.LineString(xy) if r["geometry"]["type"] == "line" else shapely.Polygon(xy)).intersection(box)
        parts = [g for g in getattr(shape, "geoms", [shape]) if not g.is_empty and g.geom_type in ("LineString", "Polygon")]
        if not parts:
            timbers.remove(ch, [r["id"]])
            continue
        best = max(parts, key=lambda g: g.length if g.geom_type == "LineString" else g.area)
        pts = np.asarray(best.coords if best.geom_type == "LineString" else best.exterior.coords[:-1], float)[:, :2]
        tracing.correct(ch, r["id"], relief.uv_of_ship(pts, frame.sign))


@pytest.fixture(scope="module")
def framed(tmp_path_factory):
    """The whole framed hull, everything proposed accepted."""
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("framed"), heading_deg=35)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def whole(framed):
    ch, snapshot = framed
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _extended(out, name):
    return sorted((r for r in out if r["name"] == name), key=lambda r: np.asarray(r["ship_xy"])[:, 0].min())


def test_strakes_extended_to_the_surveyed_ends(whole):
    """Strakes and seams traced only on x 2-9, the relief on 0.2-11.8, the land lines read at the stations: each
    strake between two seams goes on to both ends of the relief, its edges on the hull's lands (2 cm, z too)."""
    ch = whole
    _clip(ch, 2.0, 9.0)
    ship.state(ch)["clinker"] = {"lines": _land_lines(np.arange(0.25, 11.76, 0.5))}
    lo, hi = reconstruct.surveyed_range(ch)
    assert abs(lo - 0.2) < 0.02 and abs(hi - 11.8) < 0.02, (lo, hi)
    before = copy.deepcopy(timbers.timbers_of(ch))
    out = reconstruct.extend(ch)
    assert timbers.timbers_of(ch) == before  # the measured records never touched
    assert out and all(r["status"] == "reconstructed:extension" for r in out)
    assert reconstruct.records_of(ch) == out
    names = {r["name"] for r in out}
    assert names == {f"fascia {c}{k}" for c in "BT" for k in (2, 3, 4)}, names
    measured = {r["id"]: r for r in before}
    for name in names:
        pieces = _extended(out, name)
        assert len(pieces) == 2, (name, [np.ptp(np.asarray(p["ship_xy"])[:, 0]) for p in pieces])
        k = int(name[-1])
        sign = 1.0 if name.split()[-1][0] == "B" else -1.0
        for piece, end in zip(pieces, (lo, hi)):
            assert piece["source"] in measured and measured[piece["source"]]["name"] == name
            assert piece["trend"] == "land line" and piece["seams"] == [k - 1, k] and not piece["uncertain"]
            xy = np.asarray(piece["ship_xy"])
            assert (np.sign(xy[:, 1]) == sign).all() or np.abs(xy[:, 1]).min() < 1e-9
            assert shapely.Polygon(xy).is_valid and shapely.Polygon(xy).exterior.is_ccw
            assert min(abs(xy[:, 0] - end)) < 0.02 and (xy[:, 0] > lo - 0.02).all() and (xy[:, 0] < hi + 0.02).all()
            for edge, rank in (("lower", k - 1), ("upper", k)):
                e = np.asarray(piece["edges"][edge], float)
                far = e[np.argmin(np.abs(e[:, 0] - end))]
                assert abs(far[0] - end) < 0.02
                assert abs(abs(far[1]) - seam_y(far[0])[rank - 1]) < 0.02, (name, edge, far, seam_y(far[0])[rank - 1])
                assert abs(far[2] - _seam_z(far[0], rank - 1)) < 0.02, (name, edge, far, _seam_z(far[0], rank - 1))
    # again: the same records, the same ids
    again = reconstruct.extend(ch)
    assert [r["id"] for r in again] == [r["id"] for r in out]
    assert reconstruct.records_of(ch) == again and timbers.timbers_of(ch) == before


@pytest.fixture(scope="module")
def cut_at_nine(tmp_path_factory):
    """The framed hull's model only on x < 9.0 (the rest of the wreck not surveyed)."""
    _doc, ch, ends = framed_project(tmp_path_factory.mktemp("cut"), heading_deg=35, x_range=(0.0, 9.0))
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    _clip(ch, 2.0, 9.0)
    return ch, copy.deepcopy(ch.meta[ship.KEY])


def _straight(x, rank):
    """The hull's z of the land ``rank`` along the last 0.5 m before x 9.0, continued straight."""
    xs = np.linspace(8.5, 8.95, 10)
    c = np.polyfit(xs, [_seam_z(v, rank - 1) for v in xs], 1)
    return float(np.polyval(c, x))


def _beyond_z_ok(out, x_from: float, truth, tolerance: float = 0.05) -> None:
    """Every edge point past ``x_from`` within ``tolerance`` of ``truth(x, rank)`` in z."""
    for r in out:
        for edge, rank in zip(("lower", "upper"), r["seams"]):
            e = np.asarray([p for p in r["edges"][edge] if p[2] is not None], float)
            far = e[e[:, 0] > x_from]
            for x, _y, z in far:
                assert abs(z - truth(x, rank)) < tolerance, (r["name"], edge, x, z, truth(x, rank))


def test_nothing_past_the_surveyed_length_unless_the_ends_are_given(cut_at_nine):
    ch, snapshot = cut_at_nine
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    _lo, hi = reconstruct.surveyed_range(ch)
    assert abs(hi - 9.0) < 0.02, hi
    out = reconstruct.extend(ch)
    assert out
    assert max(np.asarray(r["ship_xy"])[:, 0].max() for r in out) < 9.0 + 0.02
    assert not any(r["beyond"] for r in out)
    # the stem typed at 11.5, the land lines read on the surveyed part: the strakes go on to it, not past it, and
    # their z follows the hull's (the model's open border at 9.0 does not lift it)
    ship.state(ch)["clinker"] = {"lines": _land_lines(np.arange(0.25, 8.76, 0.5))}
    assert reconstruct.set_ends(ch, stem=11.5) == {"stem": {"x": 11.5}}
    out = reconstruct.extend(ch)
    far = max(np.asarray(r["ship_xy"])[:, 0].max() for r in out)
    assert abs(far - 11.5) < 1e-6, far
    beyond = [r for r in out if r["beyond"]]
    assert beyond and {r["name"] for r in beyond} == {f"fascia {c}{k}" for c in "BT" for k in (2, 3, 4)}
    assert not any(r["uncertain"] for r in out if not r["beyond"])  # past the land lines' reach: uncertain (N1)
    _beyond_z_ok(beyond, 9.0, _straight)  # no section past the model: the last 0.5 m of the hull continued
    assert any("prua" in w for w in reconstruct.state(ch)["extension"]["warnings"])
    # a stem traced as a line in the plan, slanting: nothing past it, the edges cut on it
    line = [[10.5, -2.0], [11.0, 0.0], [11.5, 2.0]]
    reconstruct.set_ends(ch, stem={"line": line})
    out = reconstruct.extend(ch)
    stem = shapely.LineString(line)
    for r in out:
        xy = np.asarray(r["ship_xy"])
        limit = np.interp(xy[:, 1], [-2.0, 0.0, 2.0], [10.5, 11.0, 11.5])
        assert (xy[:, 0] <= limit + 1e-6).all(), r["name"]
        for edge in ("lower", "upper"):
            e = np.asarray([p[:2] for p in r["edges"][edge]], float)
            assert (e[:, 0] <= np.interp(e[:, 1], [-2.0, 0.0, 2.0], [10.5, 11.0, 11.5]) + 1e-6).all()
    assert any(shapely.Polygon(r["ship_xy"]).distance(stem) < 1e-6 for r in out)
    # no land line, no sections: past the model a seam goes on 1 m at most, uncertain, z straight on
    ship.state(ch).pop("clinker")
    reconstruct.set_ends(ch, stem=11.5)
    out = reconstruct.extend(ch)
    beyond = [r for r in out if r["beyond"]]
    assert beyond and all(r["uncertain"] and r["trend"] == "none" for r in beyond)
    assert max(np.asarray(r["ship_xy"])[:, 0].max() for r in beyond) < 9.0 + 1.0 + 0.05
    assert reconstruct.state(ch)["extension"]["warnings"]
    _beyond_z_ok(beyond, 9.0, _straight)
    # None removes the ends
    assert reconstruct.set_ends(ch) == {}
    out = reconstruct.extend(ch)
    assert max(np.asarray(r["ship_xy"])[:, 0].max() for r in out) < 9.0 + 0.02


def test_a_mirrored_strake_is_extended_along_its_edges(hull):
    """Port lost from 6.5, starboard traced: the mirrored port strakes go on forward of the loss along their own
    edges (the mirror has no seams) - with no land line and no sections 1 m at most, uncertain, y and z on the
    hull's lands all along; the measured records and the mirror untouched; the mirror built again keeps its ids, a
    catalogue changed after it stops the extension."""
    ch = hull
    mirrored = reconstruct.mirror(ch)
    before = copy.deepcopy(timbers.timbers_of(ch))
    out = reconstruct.extend(ch)
    assert reconstruct.records_of(ch) == mirrored + out
    assert timbers.timbers_of(ch) == before
    by_id = {r["id"]: r for r in mirrored}
    for k in (2, 3, 4):
        pieces = [r for r in out if r["name"] == f"fascia B{k}"]
        (piece,) = pieces
        xy = np.asarray(piece["ship_xy"])
        assert abs(xy[:, 0].max() - HOLE) < 0.1 and abs(xy[:, 0].min() - (HOLE - 1.0)) < 0.1, xy[:, 0]
        assert piece["source"] in by_id and piece["side"] == "port" and piece["uncertain"]
        for edge, rank in (("lower", k - 1), ("upper", k)):
            for x, y, z in piece["edges"][edge]:
                assert abs(y - seam_y(x)[rank - 1]) < 0.02, (k, edge, x, y, seam_y(x)[rank - 1])
                assert z is not None and abs(z - _seam_z(x, rank - 1)) < 0.02, (k, edge, x, z, _seam_z(x, rank - 1))
    # the mirror built again: the same ids, the extension's sources still there
    again = reconstruct.mirror(ch)
    assert [r["id"] for r in again] == [r["id"] for r in mirrored]
    ids = {r["id"] for r in reconstruct.records_of(ch)} | {r["id"] for r in timbers.timbers_of(ch)}
    assert all(r["source"] in ids for r in out)
    assert reconstruct.extend(ch)[0]["id"] == out[0]["id"]
    assert reconstruct.state(ch)["extension"]["mirror_built"] == reconstruct.state(ch)["mirror"]["built"]
    # a record traced after the mirror: the mirror is stale, the extension refused
    frame = ship.frame_of(ch)
    tracing.add_traced(ch, "frame", "port", "ordinata nuova", relief.uv_of_ship(
        [[3.30, 0.2], [3.38, 0.2], [3.38, 1.5], [3.30, 1.5]], frame.sign), "polygon")
    with pytest.raises(ship.NotReady):
        reconstruct.extend(ch)
    with pytest.raises(ship.NotReady):
        reconstruct.frames_by_spacing(ch)


def test_extension_and_mirror_in_either_order(tmp_path):
    """Port lost aft of 6.5, everything proposed accepted: the extension never covers the mirror's ground, so
    extension then mirror gives what mirror then extension gives, without overlaps."""
    ch = _lost_project(tmp_path, lambda x, y: x > HOLE)
    snapshot = copy.deepcopy(ch.meta[ship.KEY])

    def shapes(records):
        return sorted((r["name"], r["side"], tuple(np.round(shapely.Polygon(r["ship_xy"]).bounds, 4)))
                      for r in records)

    m1 = reconstruct.mirror(ch)
    e1 = reconstruct.extend(ch)
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    e2 = reconstruct.extend(ch)
    m2 = reconstruct.mirror(ch)
    assert m1 and shapes(m1) == shapes(m2)
    assert shapes(e1) == shapes(e2)
    for a, b in zip(sorted(e1, key=lambda r: (r["name"], r["position"])), sorted(e2, key=lambda r: (r["name"], r["position"]))):
        assert shapely.Polygon(a["ship_xy"]).symmetric_difference(shapely.Polygon(b["ship_xy"])).area < 1e-8
    over = shapely.union_all([shapely.Polygon(r["ship_xy"]) for r in m2 if r["kind"] == "plank"])
    for r in e2:
        assert shapely.Polygon(r["ship_xy"]).intersection(over).area < 1e-4, r["name"]
        assert np.asarray(r["ship_xy"])[:, 0].max() < HOLE + 0.1 or r["side"] == "starboard", r["name"]


def _frame_x(r):
    return float(np.asarray(r["ship_xy"])[:, 0].mean())


def test_frames_added_at_the_measured_spacing(whole):
    """Ordinata 6-8 and 15 removed, a loose frame lying between two others: four frames added at their places
    with the measured room, numbered after the last, as centre frames across both sides."""
    ch = whole
    frame = ship.frame_of(ch)
    removed = {r["name"]: r for r in timbers.timbers_of(ch) if r["name"] in
               ("ordinata 6", "ordinata 7", "ordinata 8", "ordinata 15")}
    timbers.remove(ch, [r["id"] for r in removed.values()])
    loose = tracing.add_traced(ch, "frame", "centre", "legno sciolto", relief.uv_of_ship(
        [[5.10, -0.8], [5.18, -0.8], [5.18, 0.8], [5.10, 0.8]], frame.sign), "polygon")
    tracing.loose(ch, [loose["id"]])
    before = copy.deepcopy(timbers.timbers_of(ch))
    out = reconstruct.frames_by_spacing(ch)
    assert timbers.timbers_of(ch) == before
    assert reconstruct.records_of(ch) == out
    assert all(r["status"] == "reconstructed:spacing" and r["kind"] == "frame" for r in out)
    assert len(out) == 4, [(r["name"], _frame_x(r), r["side"]) for r in out]
    xs = sorted(_frame_x(r) for r in out)
    truth = sorted(_frame_x(r) for r in removed.values())
    assert np.abs(np.asarray(xs) - truth).max() < 0.01, (xs, truth)
    assert {r["name"] for r in out} == {f"ordinata {n}" for n in range(24, 28)}
    for r in sorted(out, key=_frame_x):
        poly = shapely.Polygon(r["ship_xy"])
        x0, y0, x1, y1 = poly.bounds
        assert abs((x1 - x0) - FRAME_ROOM) < 0.01, (r["name"], x1 - x0)
        assert r["side"] == "centre" and y0 < -0.5 and y1 > 0.5
        assert r["notes"]  # where it was put
    # again: the same ids; a loose frame never counts
    again = reconstruct.frames_by_spacing(ch)
    assert [r["id"] for r in again] == [r["id"] for r in out]
    assert reconstruct.state(ch)["spacing"]["median"] == pytest.approx(0.45, abs=0.005)


def test_a_trend_held_past_its_reach_is_uncertain(cut_at_nine):
    """N1 of the review of Task 10: land lines ending at 8.75, the stem at 11.5: past the trend's points and the 1 m of
    its bending the extension only holds a slope, and says so."""
    ch, snapshot = cut_at_nine
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    ship.state(ch)["clinker"] = {"lines": _land_lines(np.arange(0.25, 8.76, 0.5))}
    reconstruct.set_ends(ch, stem=11.5)
    out = reconstruct.extend(ch)
    beyond = [r for r in out if r["beyond"]]
    assert beyond and all(r["uncertain"] for r in beyond)
    assert not any(r["uncertain"] for r in out if not r["beyond"])
    warnings = reconstruct.state(ch)["extension"]["warnings"]
    assert any(r["name"] in w and "oltre" in w for r in beyond for w in warnings), warnings


# ----------------------------------------------------------------------------- the loft, the build, the table
LOST_BOTH = (8.4, 9.3)  # both sides of the hull lost between these x
STATIONS = np.arange(1.125, 11.0, 0.45)  # halfway between the frames


def _wreck_project(tmp_path, port_from=None):
    """The framed hull with both sides lost over :data:`LOST_BOTH` (and, with ``port_from``, port lost aft of it),
    everything proposed accepted, the solids built, stations halfway between the frames cut from the model."""
    from hulls import UTM

    from ppg_shipwreck import lines

    v, f = framed_hull()
    c = v[f].mean(axis=1)
    drop = (c[:, 0] > LOST_BOTH[0]) & (c[:, 0] < LOST_BOTH[1])
    if port_from is not None:
        drop |= (c[:, 1] < 0) & (c[:, 0] > port_from)
    _doc, ch, ends = _project(tmp_path, v, f[~drop], 35.0, UTM, "EPSG:32633", 2.0)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    # the keelson traced over all of its length (the proposals stop it where a side is lost): what lies over the
    # planking is left out of the sections the hull is lofted through
    from hulls import KEELSON_HALF, KEELSON_X

    timbers.remove(ch, [r["id"] for r in timbers.timbers_of(ch) if r["kind"] == "keelson"])
    half = KEELSON_HALF + 0.005
    keelson = [[KEELSON_X[0], -half], [KEELSON_X[1], -half], [KEELSON_X[1], half], [KEELSON_X[0], half]]
    tracing.add_traced(ch, "keelson", "centre", "paramezzale", relief.uv_of_ship(keelson, ship.frame_of(ch).sign),
                       "polygon")
    solids.build(ch)
    ship.set_stations(ch, [{"name": f"S{k}", "x": float(x)} for k, x in enumerate(STATIONS)])
    lines.build(ch)
    return ch


@pytest.fixture(scope="module")
def wreck(tmp_path_factory):
    ch = _wreck_project(tmp_path_factory.mktemp("wreck"), port_from=HOLE)
    return ch, copy.deepcopy(ch.meta[ship.KEY])


@pytest.fixture
def lost(wreck):
    ch, snapshot = wreck
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    return ch


def _relief_at(ch, xy):
    z, a = relief.load(ch, "height")
    row, col, ok = relief.cells_at(z.shape, a, ship.frame_of(ch).sign, xy[:, 0], xy[:, 1])
    out = np.full(len(xy), np.nan)
    out[ok] = z[row[ok], col[ok]]
    return out


def _mesh(path):
    from pyarchinit_photogrammetry.io.mesh_ply import PlyMesh

    mesh = PlyMesh(path)
    v, f = np.array(mesh.xyz(), float), np.concatenate(list(mesh.face_chunks()))
    del mesh
    return v, f


def test_the_hull_lofted_where_neither_side_exists(lost):
    """Both sides lost over x 8.4-9.3, port also aft of 6.5 (mirrored): the loft spans the stations, keeps faces only
    where neither the model nor a mirrored timber is, lies on the hull's clinker section (p95 < 2.5 cm), and, with no
    stem or stern given, leaves both ends open; given, it closes on them."""
    from hulls import clinker_section, half_breadth

    ch = lost
    mirrored = reconstruct.mirror(ch)
    solid = reconstruct.loft(ch)
    assert solid is not None and solid.status == "reconstructed:fairing" and solid.kind == "hull"
    assert solid.name == "carenatura" and solid.volume == 0.0 and not solid.hidden.any()
    st = reconstruct.state(ch)["loft"]
    path = ch.data_dir / st["file"]
    assert path.name == "carenatura.ply" and path.parent.name == "reconstruction" and path.exists()
    assert set(st["open_ends"]) == {"stem", "stern"} and len(st["open_ends"]) == 2
    v, f = _mesh(path)
    assert np.allclose(v, solid.vertices) and np.array_equal(f, solid.faces)
    centres = v[f].mean(axis=1)
    # no face over the model, none over a mirrored timber
    assert np.isnan(_relief_at(ch, centres[:, :2])).all()
    over = shapely.union_all([shapely.Polygon(r["ship_xy"]) for r in mirrored])
    assert not shapely.contains_xy(over, centres[:, 0], centres[:, 1]).any()
    # the span between the two sides' loss covered, both sides out to the sheer
    for x in np.arange(8.5, 9.25, 0.1):
        near = np.abs(v[:, 0] - x) < 0.051
        for sign in (1.0, -1.0):
            ys = sign * v[near, 1]
            assert ys.max() > 0.9 * half_breadth(x), (x, sign, ys.max())
    # on the hull: z against the clinker section at the vertex's |y|
    span = (v[:, 0] > LOST_BOTH[0] + 0.05) & (v[:, 0] < LOST_BOTH[1] - 0.05)
    errors = []
    for x, y, z in v[span]:
        sec = clinker_section(x)
        port = sec[sec[:, 0] >= 0]
        errors.append(abs(z - np.interp(abs(y), port[:, 0], port[:, 1])))
    errors = np.asarray(errors)
    assert len(errors) > 100 and np.percentile(errors, 95) < 0.025, np.percentile(errors, [50, 95, 100])
    # the ends given: the loft goes on to them, closing to the keel line
    reconstruct.set_ends(ch, stem=-0.3, stern=12.3)  # past the model's ends (0.2, 11.8): the tips are shown
    solid = reconstruct.loft(ch)
    assert reconstruct.state(ch)["loft"]["open_ends"] == []
    v = solid.vertices
    for end in (-0.3, 12.3):
        tip = v[np.abs(v[:, 0] - end) < 1e-9]
        assert len(tip) and np.abs(tip[:, 1]).max() < 1e-9, end
    assert len(np.unique(v, axis=0)) == len(v)  # the two sides share the tips' vertices


def test_the_end_taper_follows_the_last_slope():
    """Past the last station each sample's |y| goes to 0 at the end on a quadratic in x: the last breadth and its slope
    along x at the start, 0 at the end (its slope there free: a stem meets the keel at an angle), never wider than the
    last breadth, never rising again, never below 0; either way along x."""
    from ppg_shipwreck import loft as loft_mod

    breadth = np.array([0.8, 0.5])  # two samples' |y| at the last station
    for x_last, x_end, slope in ((10.5, 12.0, -0.5), (1.0, -0.5, 0.4), (10.5, 12.0, 0.3), (10.5, 12.0, -2.0)):
        xs, ys = loft_mod._taper(x_last, x_end, 0.1, breadth, breadth * slope / 0.8)
        assert xs[-1] == x_end and np.all(np.abs(ys[-1]) < 1e-12)
        L = x_end - x_last
        r = (xs - x_last) / L
        for k, b in enumerate(breadth):
            m = b * slope / 0.8 * L
            quadratic = b + m * r - (b + m) * r**2
            expected = np.clip(np.minimum.accumulate(quadratic), 0.0, b)
            assert np.allclose(ys[:, k], expected), (x_last, x_end, slope, k)
            assert ys[:, k].max() <= b + 1e-12 and ys[:, k].min() >= 0.0
            assert np.all(np.diff(ys[:, k]) <= 1e-12)  # non-increasing towards the end
    # a start steeper than -2 b: the quadratic would dip below 0 and come back; it stays at 0 instead
    xs, ys = loft_mod._taper(10.5, 12.0, 0.1, breadth, breadth * -2.0 / 0.8)
    zero = np.flatnonzero(ys[:, 0] == 0.0)
    assert len(zero) > 1 and np.all(ys[zero[0]:, 0] == 0.0)


def test_build_the_whole_reconstruction(lost):
    """mirror -> extension -> frames by spacing -> loft -> solids, in one build: the extended strakes and the spaced
    frames closed and on the hull, the measured records and their files untouched, the same table and the same files
    built twice, the measured totals those of the catalogue's measures."""
    import trimesh

    ch = lost
    before = copy.deepcopy(timbers.timbers_of(ch))
    files = {p.name: p.read_bytes() for p in (ship.folder(ch) / solids.FOLDER).glob("*.ply")}
    out = reconstruct.build(ch)
    assert timbers.timbers_of(ch) == before  # the catalogue never touched
    assert {p.name: p.read_bytes() for p in (ship.folder(ch) / solids.FOLDER).glob("*.ply")} == files
    st = reconstruct.state(ch)
    assert {"params", "built", "timbers", "mirror", "extension", "spacing", "loft", "table", "open_ends"} <= set(st)
    assert st["params"] == {"step": 0.10, "samples": 41, "steps": ["mirror", "extension", "frames", "loft"]}
    assert out["loft"] and out["mirrored"] and out["extended"] and out["spaced"]
    assert out["table"] == st["table"] and set(out["open_ends"]) == {"stem", "stern"}
    records = reconstruct.records_of(ch)
    statuses = {r["status"] for r in records}
    assert statuses == {"reconstructed:mirror", "reconstructed:extension", "reconstructed:spacing"}
    built = {r["id"] for r in records if r.get("solid")}
    for kind, status in (("plank", "reconstructed:extension"), ("frame", "reconstructed:spacing")):
        done = [r for r in records if r["status"] == status and r["id"] in built]
        assert done, (status, out["refused"])
        for r in done:
            assert r["kind"] == kind
            v, f = _mesh(ch.data_dir / r["solid"])
            tm = trimesh.Trimesh(v, f, process=False)
            assert tm.is_watertight and tm.volume > 0, r["name"]
            assert tm.volume == pytest.approx(r["volume"], rel=1e-6)
    # the frames put back in the lost span stand on the loft
    spaced = [r for r in records if r["status"] == "reconstructed:spacing" and r["id"] in built]
    in_span = [r for r in spaced if LOST_BOTH[0] < _frame_x(r) < LOST_BOTH[1]]
    assert in_span, [(r["name"], _frame_x(r)) for r in spaced]
    refused = {r["id"] for r in out["refused"]}
    assert refused == {r["id"] for r in records if not r.get("solid") and r["status"] != "reconstructed:mirror"}
    names = sorted(p.name for p in reconstruct.folder(ch).glob("*.ply"))
    assert names == sorted(["carenatura.ply"] + [f"{r['id']}.ply" for r in records if r.get("solid")])
    first = {p.name: p.read_bytes() for p in reconstruct.folder(ch).glob("*.ply")}
    table = copy.deepcopy(st["table"])
    again = reconstruct.build(ch)
    assert again["table"] == table
    assert {p.name: p.read_bytes() for p in reconstruct.folder(ch).glob("*.ply")} == first
    assert timbers.timbers_of(ch) == before


def test_the_table_of_what_is_measured_and_what_is_reconstructed(lost, tmp_path):
    import csv

    from openpyxl import load_workbook

    ch = lost
    reconstruct.build(ch)
    tab = reconstruct.table(ch)
    assert tab == reconstruct.state(ch)["table"]
    rows = tab["rows"]
    catalogue, recon = timbers.timbers_of(ch), reconstruct.records_of(ch)
    assert [r["id"] for r in rows][:len(catalogue)] == [r["id"] for r in catalogue]
    assert {r["id"] for r in recon} <= {r["id"] for r in rows}
    measure = timbers.measure(ch)
    measured = [m for m in measure if m["status"] == "measured" and m["length"] is not None]
    assert tab["totals"]["measured"]["length"] == pytest.approx(sum(m["length"] for m in measured), abs=1e-3)
    frame = ship.frame_of(ch)
    area = sum(shapely.Polygon(tracing.ship_xy_of(ch, r, frame)).area for r in catalogue
               if timbers.is_measured(r) and r["geometry"]["type"] == "polygon")
    assert tab["totals"]["measured"]["area"] == pytest.approx(area, abs=1e-4)
    for row in rows:
        if row["status"] == "measured":
            assert row["reconstructed_length"] is None and row["reconstructed_area"] is None
        else:
            assert row["measured_length"] is None and row["measured_area"] is None
    by_kind = tab["totals"]["kind"]
    assert by_kind["plank"]["reconstructed_area"] > 0 and by_kind["frame"]["reconstructed_area"] > 0
    assert sum(tab["counts"].values()) == len(rows)
    assert tab["counts"]["reconstructed:mirror"] == len([r for r in recon if r["status"] == "reconstructed:mirror"])
    assert tab["mirror"]["pieces"] == tab["counts"]["reconstructed:mirror"]
    assert isinstance(tab["mirror"]["disagreements"], list)
    paths = reconstruct.write_table(ch, tmp_path)
    with open(paths["csv"], newline="", encoding="utf-8") as f:
        read = list(csv.reader(f))
    assert read[0][:5] == ["id", "name", "kind", "side", "status"] and len(read) == len(rows) + 1
    book = load_workbook(paths["xlsx"])
    assert book.sheetnames == ["misurato e ricostruito"] and book.properties.title == "misurato e ricostruito (ipotesi)"
    assert book.active.max_row == len(rows) + 1
    default = reconstruct.write_table(ch)
    assert default["csv"] == reconstruct.folder(ch) / "tabella.csv" and default["xlsx"].exists()


def test_surface_for_reads_relief_then_mirror_then_loft(lost):
    from hulls import clinker_section

    ch = lost
    reconstruct.build(ch)
    poly = shapely.box(5.0, -1.5, 10.0, 1.5)
    surface = solids.surface_for(ch, poly)
    xs = np.array([5.5, 7.5, 8.8, 8.8])
    ys = np.array([-0.5, 0.5, 0.5, -0.5])  # starboard on the model, port mirrored, both sides lofted
    z = surface(xs, ys)
    assert np.isfinite(z).all(), z
    assert z[0] == pytest.approx(_relief_at(ch, np.array([[5.5, -0.5]]))[0])
    for k in range(1, 4):
        sec = clinker_section(xs[k])
        port = sec[sec[:, 0] >= 0]
        assert abs(z[k] - np.interp(abs(ys[k]), port[:, 0], port[:, 1])) < 0.02, (k, z[k])
    assert np.isnan(surface(np.array([20.0]), np.array([0.0]))).all()


def test_a_build_stopped_halfway_keeps_the_earlier_reconstruction(lost, monkeypatch):
    ch = lost
    reconstruct.build(ch)
    kept = copy.deepcopy(ship.state(ch)["reconstruction"])
    files = {p.name: p.read_bytes() for p in reconstruct.folder(ch).glob("*.ply")}

    def broken(*args, **kwargs):
        raise ship.NotReady("stopped")

    monkeypatch.setattr(reconstruct, "frames_by_spacing", broken)
    with pytest.raises(ship.NotReady):
        reconstruct.build(ch)
    assert ship.state(ch)["reconstruction"] == kept
    assert {p.name: p.read_bytes() for p in reconstruct.folder(ch).glob("*.ply")} == files


def _clinker_errors(v, x0, x1, side=None):
    from hulls import clinker_section

    span = (v[:, 0] > x0) & (v[:, 0] < x1)
    if side is not None:
        span &= side * v[:, 1] >= 0
    out = []
    for x, y, z in v[span]:
        sec = clinker_section(x)
        port = sec[sec[:, 0] >= 0]
        out.append(abs(z - np.interp(abs(y), port[:, 0], port[:, 1])))
    return np.asarray(out)


def test_the_loft_refuses_sections_missing_or_out_of_date(lost):
    from ppg_shipwreck import lines

    ch = lost
    kept = copy.deepcopy(ship.state(ch).get("reconstruction"))
    data = lines.load(ch)
    stale = copy.deepcopy(data)
    stale["source"]["model_size"] = -1  # cut from another model
    try:
        lines._save(ch, stale)
        with pytest.raises(ship.NotReady, match="tagliale di nuovo"):
            reconstruct.loft(ch)
        with pytest.raises(ship.NotReady, match="tagliale di nuovo"):
            reconstruct.build(ch)
        assert ship.state(ch).get("reconstruction") == kept
        lines._path(ch).unlink()
        with pytest.raises(ship.NotReady, match="Taglia prima le sezioni"):
            reconstruct.loft(ch)
        with pytest.raises(ship.NotReady, match="Taglia prima le sezioni"):
            reconstruct.build(ch)
        assert ship.state(ch).get("reconstruction") == kept
    finally:
        lines._save(ch, data)


def test_an_untraced_keelson_is_told_and_kept_out_of_the_keel(hull):
    """Port lost aft of 6.5, starboard and the keelson accepted as the plugin proposes them (the keelson stops where
    port is lost): the stations aft of it read the keelson's top in the section; the keel is taken from the
    section's lowest point there, the faired port side stays on the planking, and the stations are named."""
    from ppg_shipwreck import lines

    ch = hull
    items = ship.state(ch)["proposals"]["items"]
    tracing.accept(ch, [i["id"] for i in items if i["kind"] == "keelson" and "?" not in i["name"]])
    keelsons = [r for r in timbers.timbers_of(ch) if r["kind"] == "keelson"]
    assert keelsons and max(np.asarray(tracing.ship_xy_of(ch, r))[:, 0].max() for r in keelsons) < 9.0
    ship.set_stations(ch, [{"name": f"S{k}", "x": float(x)} for k, x in enumerate(STATIONS)])
    lines.build(ch)
    solid = reconstruct.loft(ch)
    errors = _clinker_errors(solid.vertices, 8.45, 9.25)
    assert len(errors) > 100 and np.percentile(errors, 95) < 0.025, np.percentile(errors, [50, 95, 100])
    warnings = reconstruct.state(ch)["loft"]["warnings"]
    import re

    assert warnings and any(re.search(r"\bS16\b", w) for w in warnings), warnings  # x 8.325, no keelson over it
    assert not any(re.search(r"\bS2\b", w) for w in warnings), warnings  # x 2.025, under the keelson
    assert len(np.unique(solid.vertices, axis=0)) == len(solid.vertices)  # one vertex per place, tips too
    out = reconstruct.build(ch)  # the build says it too
    assert any(re.search(r"\bS16\b", w) for w in out["warnings"]), out["warnings"]


def test_build_minor_guards(lost, monkeypatch):
    """A file system without hard links still builds; each 3D file is read once per build; the disagreement
    of the two sides never overwrites the table."""
    from ppg_shipwreck import rebuild

    ch = lost

    def no_links(*args, **kwargs):
        raise OSError("hard links not supported")

    monkeypatch.setattr(rebuild.os, "link", no_links)
    reads: dict = {}
    real = reconstruct._read_solid

    def counted(path):
        reads[str(path)] = reads.get(str(path), 0) + 1
        return real(path)

    monkeypatch.setattr(reconstruct, "_read_solid", counted)
    reconstruct.build(ch)
    reconstruct.build(ch)
    assert reads and max(reads.values()) <= 2, sorted(reads.values())[-3:]  # once per build
    table = copy.deepcopy(reconstruct.state(ch)["table"])
    rows = reconstruct.disagreement(ch)
    assert reconstruct.state(ch)["table"] == table
    reconstruct.build(ch)  # the rows kept where the docstring says
    assert reconstruct.state(ch)["disagreement"] == rows


def test_a_typed_keel_standing_proud_is_the_keel_not_a_keelson(tmp_path, monkeypatch):
    """The keel a narrow ridge (7 cm wide, 5 cm proud) along the whole keel line, the frames running over it, no
    keelson, the keel typed with its siding: the loft follows the planking, with a neutral note, no warning to
    trace a keelson."""
    import hulls

    from ppg_shipwreck import lines

    monkeypatch.setattr(hulls, "KEELSON_HALF", 0.035)
    monkeypatch.setattr(hulls, "KEELSON_X", (0.25, 11.75))
    monkeypatch.setattr(hulls, "KEELSON_HEIGHT", 0.05)
    v, f = hulls.framed_hull()
    c = v[f].mean(axis=1)
    drop = (c[:, 0] > LOST_BOTH[0]) & (c[:, 0] < LOST_BOTH[1])
    _doc, ch, ends = _project(tmp_path, v, f[~drop], 35.0, hulls.UTM, "EPSG:32633", 2.0)
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start"))
    relief.build(ch, cell=0.01)
    proposals_mod.build(ch)
    tracing.accept(ch, [i["id"] for i in ship.state(ch)["proposals"]["items"] if "?" not in i["name"]])
    timbers.remove(ch, [r["id"] for r in timbers.timbers_of(ch) if r["kind"] == "keelson"])
    solids.set_keel(ch, 0.07, 0.15)
    ship.set_stations(ch, [{"name": f"S{k}", "x": float(x)} for k, x in enumerate(STATIONS)])
    lines.build(ch)
    solid = reconstruct.loft(ch)
    st = reconstruct.state(ch)["loft"]
    assert not any("paramezzale" in w for w in st["warnings"]), st["warnings"]
    assert any("la chiglia è il suo solido" in n for n in st["notes"]), st
    errors = _clinker_errors(solid.vertices, 8.45, 9.25)
    assert len(errors) > 100 and np.percentile(errors, 95) < 0.025, np.percentile(errors, [50, 95, 100])


def test_a_self_crossing_outline_is_mirrored_by_its_parts(hull):
    """A strake traced with its outline crossing itself (a bow tie, as a hand tracing can make): its valid parts are
    mirrored, the others as before; nothing raises."""
    ch = hull
    frame = ship.frame_of(ch)
    expected = {reconstruct._base_name(r["name"]) for r in reconstruct.mirror(ch)}
    plank = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and r["side"] == "starboard")
    xy = np.asarray(tracing.ship_xy_of(ch, plank, frame), float)
    poly = shapely.Polygon(xy)
    x0, y0, x1, y1 = poly.bounds
    xm = 0.5 * (x0 + x1)
    bowtie = poly.intersection(shapely.box(x0, -5, xm, 5)).union(poly.intersection(shapely.box(xm, -5, x1, 5)))
    left = np.asarray(shapely.geometry.polygon.orient(poly.intersection(shapely.box(x0, -5, xm, 5)), 1.0).exterior.coords)[:-1]
    right = np.asarray(shapely.geometry.polygon.orient(poly.intersection(shapely.box(xm, -5, x1, 5)), 1.0).exterior.coords)[:-1]
    crossing = np.vstack([left[::-1], right])  # two lobes joined by crossing edges
    assert not shapely.Polygon(crossing).is_valid and bowtie.area > 0
    plank["ship_xy"] = crossing.tolist()
    plank["geometry"] = timbers.geometry("polygon", frame.to_world(np.column_stack([crossing, np.zeros(len(crossing))]))[:, :2])
    out = reconstruct.mirror(ch)
    assert {reconstruct._base_name(r["name"]) for r in out} == expected


def test_an_extension_that_fans_out_is_cut_and_uncertain(cut_at_nine):
    """A noisy land line bent outwards past the surveyed length (the outer seam of a strake diverging from the inner
    one): the extended piece stops where its width passes 1.5 times the strake's measured width or its outer edge
    passes the hull's half-breadth, is flagged uncertain and named in a warning; the other strakes go on to the end."""
    ch, snapshot = cut_at_nine
    ch.meta[ship.KEY] = copy.deepcopy(snapshot)
    stations = np.arange(0.25, 11.76, 0.5)
    lines_ = _land_lines(stations)
    rng = np.random.default_rng(3)
    for ln in lines_:
        if ln["side"] == "port" and ln["k"] == 3:  # the upper seam of fascia B3: outwards past x 9, with noise
            for p in ln["points"]:
                if p[0] > 9.0:
                    p[1] += 0.6 * (p[0] - 9.0) ** 1.5 + rng.normal(0.0, 0.01)
    ship.state(ch)["clinker"] = {"lines": lines_}
    reconstruct.set_ends(ch, stern=11.8)
    out = reconstruct.extend(ch)
    frame = ship.frame_of(ch)
    measured = [shapely.Polygon(tracing.ship_xy_of(ch, r, frame)) for r in timbers.timbers_of(ch)
                if r["name"] == "fascia B3" and r["kind"] == "plank"]
    width = float(np.median([p.area / (p.bounds[2] - p.bounds[0]) for p in measured]))
    b3 = [r for r in out if r["name"] == "fascia B3" and np.asarray(r["ship_xy"])[:, 0].max() > 9.0]
    assert b3 and all(r["uncertain"] for r in b3)
    for r in b3:
        lo, up = np.asarray(r["edges"]["lower"], float), np.asarray(r["edges"]["upper"], float)
        xs = np.intersect1d(np.round(lo[:, 0], 4), np.round(up[:, 0], 4))
        w = [np.interp(x, up[:, 0], np.abs(up[:, 1])) - np.interp(x, lo[:, 0], np.abs(lo[:, 1])) for x in xs]
        assert max(w) <= 1.5 * width + 0.02, (max(w), width)
        assert np.asarray(r["ship_xy"])[:, 0].max() < 11.0  # stopped before the end
    warnings = reconstruct.state(ch)["extension"]["warnings"]
    assert any("fascia B3" in w and "tagliat" in w for w in warnings), warnings
    b2 = [r for r in out if r["name"] == "fascia B2"]
    assert max(np.asarray(r["ship_xy"])[:, 0].max() for r in b2) > 11.5  # an untouched strake reaches the end


def test_a_lost_side_takes_the_other_sides_half_breadth(lost):
    """Port lost aft of 6.5: where port has no relief the guard reads starboard's half-breadth at that x (the symmetry
    the mirror assumes), never the midship breadth held to the end. No extended piece leaves the hull by more than its
    strake's width; fascia B4's tip (x 11.2-11.8) is cut."""
    from hulls import plan_half_breadth

    ch = lost
    reconstruct.mirror(ch)
    heights = reconstruct._Heights(ch, ship.frame_of(ch), timbers.timbers_of(ch))
    xs = np.array([11.0, 11.5, 11.7])
    port, starboard = heights.half_breadth("port", xs), heights.half_breadth("starboard", xs)
    assert np.allclose(port, starboard) and np.all(port < 0.7), (port, starboard)
    out = reconstruct.extend(ch)
    for r in out:
        if r["side"] != "port":
            continue
        e = np.asarray(r["edges"]["upper"], float)
        assert np.all(np.abs(e[:, 1]) <= [plan_half_breadth(x) + 0.25 for x in e[:, 0]]), r["name"]
    b4 = [r for r in out if r["name"] == "fascia B4"]
    assert all(np.asarray(r["ship_xy"])[:, 0].max() < 11.6 for r in b4), [np.asarray(r["ship_xy"])[:, 0].max() for r in b4]
    assert any("fascia B4" in w and "tagliat" in w for w in reconstruct.state(ch)["extension"]["warnings"])


# ----------------------------------------------------------------------------- a stored reconstruction checked when used
def test_a_strake_corrected_after_the_build_makes_the_reconstruction_stale(lost, tmp_path):
    """After the build a traced strake is corrected by hand: the reconstruction says it is to be built again, with
    the reason, in the pane's state, in the table (CSV column, workbook title), on the plan drawn with it and in the
    3D export's note; built again it is current; the ends changed after the build make it stale too."""
    import csv

    from openpyxl import load_workbook

    import ppg_shipwreck as plugin
    from ppg_shipwreck import drawings

    ch = lost
    reconstruct.build(ch)
    assert reconstruct.stale(ch) is None
    assert "ricostruire" not in plugin.reconstruction_state(ch)
    frame = ship.frame_of(ch)
    strake = next(r for r in timbers.timbers_of(ch) if r["kind"] == "plank" and r["side"] == "starboard"
                  and r["geometry"]["type"] == "polygon")
    xy = tracing.ship_xy_of(ch, strake, frame) + [0.0, 0.05]
    tracing.correct(ch, strake["id"], relief.uv_of_ship(xy, frame.sign))
    reason = reconstruct.stale(ch)
    assert reason and "catalogo" in reason, reason
    state = plugin.reconstruction_state(ch)
    assert "Da ricostruire: " + reason in state, state
    tab = reconstruct.table(ch)
    assert tab["stale"] == reason
    out = reconstruct.write_table(ch, tmp_path)
    assert out["stale"] == reason
    with open(out["csv"], newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert all(r["stale"] == (reason if r["status"] != "measured" else "") for r in rows)
    assert "da rifare" in load_workbook(out["xlsx"]).properties.title
    d = drawings.make(ch, "wreck_plan", reconstruction=True)
    assert any(reason in n for n in d.notes), d.notes
    assert not any(reason in n for n in drawings.make(ch, "wreck_plan").notes)
    exported = plugin.export_3d(ch, tmp_path / "r.obj", tables=False, reconstruction=True)
    assert reason in exported["note"]
    reconstruct.build(ch)
    assert reconstruct.stale(ch) is None and reconstruct.table(ch)["stale"] is None
    reconstruct.set_ends(ch, stern=12.3)
    assert "estremit" in (reconstruct.stale(ch) or "")


def test_after_a_bow_swap_the_reconstruction_is_refused_in_3d(lost, tmp_path):
    """The bow swapped after the build (y and the sides exchanged): the reconstruction's solids lie in the old ship
    frame, so the 3D files refuse them (the catalogue's alone still go) and say why; the plan leaves the faired hull
    out and says the reconstruction is to be built again."""
    import ppg_shipwreck as plugin
    from ppg_shipwreck import drawings, models3d

    ch = lost
    reconstruct.build(ch)
    frame = ship.set_frame(ch, ship.need_frame(ch).other_bow())
    ship.state(ch)["solids"]["frame_key"] = relief.frame_key(frame)  # as if the catalogue's solids were built again
    reason = reconstruct.stale(ch)
    assert reason and "sistema della nave" in reason, reason
    assert models3d.items(ch, "ship", reconstruction=False)
    with pytest.raises(ship.NotReady, match="sistema della nave"):
        models3d.items(ch, "ship", reconstruction=True)
    with pytest.raises(ship.NotReady, match="sistema della nave"):
        plugin.export_3d(ch, tmp_path / "r.obj", tables=False, reconstruction=True)
    assert drawings._loft_mesh(ch) is None


def test_the_reconstructed_total_leaves_out_the_faired_hull_and_counts_overlaps_once(lost):
    """The faired hull (3.9 m² here) is given apart and the mirrored frames lie over the mirrored strakes: the
    reconstructed total is the area of the union of the reconstructed timbers in the ship plan, not the sum of the
    rows (10.4 m² on this hull); the rows, the totals per kind and per status keep their own areas."""
    from ppg_shipwreck import drawings
    from ppg_shipwreck.texts import num

    ch = lost
    reconstruct.build(ch)
    tab = reconstruct.table(ch)
    frame = ship.frame_of(ch)
    rows = [r for r in tab["rows"] if r["status"] != "measured"]
    summed = sum(r["reconstructed_area"] or 0.0 for r in rows)
    hull = next(r for r in rows if r["kind"] == "hull")["reconstructed_area"]
    assert summed == pytest.approx(10.42, abs=0.1) and hull == pytest.approx(3.89, abs=0.1)
    union = shapely.union_all([solids._polygon(tracing.ship_xy_of(ch, r, frame)) for r in reconstruct.records_of(ch)
                               if r["geometry"]["type"] == "polygon"]).area
    total = tab["totals"]["reconstructed"]["area"]
    assert total == pytest.approx(union, abs=1e-4)
    assert total < summed - hull - 0.3, (total, summed, hull)  # what lies over other reconstructed timbers once
    assert tab["totals"]["status"]["reconstructed:fairing"]["area"] == pytest.approx(hull)
    assert tab["totals"]["kind"]["hull"]["reconstructed_area"] == pytest.approx(hull)
    assert tab["totals"]["reconstructed"]["length"] == pytest.approx(
        sum(r["reconstructed_length"] or 0.0 for r in rows if r["kind"] != "hull"), abs=1e-4)
    d = drawings.make(ch, "wreck_plan", reconstruction=True)
    assert any(f"ricostruito: {num(total, 2)} m² senza la carenatura" in n for n in d.notes), d.notes
