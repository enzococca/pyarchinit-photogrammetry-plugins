"""The relief of the hull from above: an elevation of the program on a horizontal plane in the ship frame (u = x,
v = sign * y, d = ship z), its hillshade and local relief model, the copy north-up in the world."""
from __future__ import annotations

import os

import numpy as np
import pytest
import rasterio
from hulls import clinker_project, clinker_section

from pyarchinit_photogrammetry.archaeo import breaklines

from ppg_shipwreck import relief, ship
from ppg_shipwreck.ship import NotReady


def _value(arr, transform, uv):
    """The cell of ``arr`` under the point ``uv`` (u, v) of its grid."""
    col, row = ~transform @ (float(uv[0]), float(uv[1]))
    return float(arr[int(np.floor(row)), int(np.floor(col))])


def _set(ch, ends, bow="start"):
    return ship.set_frame(ch, ship.Frame(ends[0], ends[1], bow))


def test_relief_is_an_elevation_in_the_ship_frame(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    frame = _set(ch, ends)
    info = relief.build(ch, cell=0.02)
    e = relief.entry(ch)
    assert e is not None and e["name"] == relief.NAME[0]
    assert e["id"] == info["elevation"] == ship.state(ch)["relief"]["elevation"]
    plane = relief.plane_of(frame)
    assert plane.normal[2] == pytest.approx(1.0)
    z, tr = relief.load(ch, "height")
    uv = relief.uv_of_ship(np.array([[6.0, 0.8]]), frame.sign)[0]
    # the hull at y 0.8: its height within 2 cm of the clinker section there (four lands of 1.5 cm below it put the
    # surface 6 cm under the smooth hull's)
    section = clinker_section(6.0)
    truth = np.interp(0.8, section[:, 0], section[:, 1])
    assert abs(_value(z, tr, uv) - truth) < 0.02
    for kind in ("hillshade", "slrm"):
        arr, t2 = relief.load(ch, kind)
        assert arr.shape == z.shape and t2 == tr and arr.dtype == np.float32
        assert not (np.isfinite(arr) & ~np.isfinite(z)).any()  # nothing off the model
        assert np.isfinite(arr).sum() > 0.95 * np.isfinite(z).sum()  # the hillshade misses the rim's cells
        path = ch.data_dir / ship.state(ch)["relief"]["files"][kind]
        assert path.parent == ch.data_dir / "shipwreck" / "relief"
        with rasterio.open(path) as src:
            tags = src.tags()
        assert tags["ppg_v"] == "sign*y" and "ppg_ship_frame" in tags
    hs, _ = relief.load(ch, "hillshade")
    assert 0.0 <= np.nanmin(hs) and np.nanmax(hs) <= 1.0
    # built again: the previous elevation is replaced, not added
    relief.build(ch, cell=0.02)
    names = [x["name"] for x in ch.meta.get("elevations", [])]
    assert names.count(relief.NAME[0]) == 1


def test_mirror_frame_v_is_minus_y(tmp_path):
    # a hole on the hull's left (its y > 0, the physical left of who walks from the first end to the other)
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35, hole=(5.5, 6.5, 0.5, 1.1))
    for bow in ("start", "end"):
        frame = _set(ch, ends, bow)
        if bow == "end":
            assert frame.sign == +1
        else:
            assert frame.sign == -1
        relief.build(ch, cell=0.02)
        z, tr = relief.load(ch, "height")
        hole = frame.to_ship(ch.to_world(np.array([[6.0 / 2.0, 0.8 / 2.0, 0.0]])))[0]  # the hull's (6, 0.8)
        assert hole[1] == pytest.approx(0.8 * frame.sign, abs=1e-6)  # on the mirror frame (sign -1) the left is -y
        uv = relief.uv_of_ship(hole[None, :2], frame.sign)[0]
        assert uv[1] == pytest.approx(0.8, abs=1e-6)  # v: always the left of x
        assert np.isnan(_value(z, tr, uv))
        assert np.isfinite(_value(z, tr, (uv[0], -uv[1])))
        xy = np.random.default_rng(1).normal(size=(50, 2)) * 1e3
        back = relief.ship_of_uv(relief.uv_of_ship(xy, frame.sign), frame.sign)
        assert np.abs(back - xy).max() < 1e-12


def test_too_many_cells_refused_with_the_cell_that_fits(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    _set(ch, ends)
    with pytest.raises(NotReady, match="mm"):
        relief.check_size(ch, 1e-5)
    h, w = relief.check_size(ch, 0.02)
    assert 0 < h * w < relief.MAX_CELLS
    assert relief.default_cell(ch) == pytest.approx(0.02, rel=0.25)


def test_world_copy_is_north_up(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    frame = _set(ch, ends)
    relief.build(ch, cell=0.02)
    out = tmp_path / "world.tif"
    relief.export_world(ch, "height", out)
    with rasterio.open(out) as src:
        t, arr = src.transform, src.read(1)
        assert t.b == 0 and t.d == 0 and t.a == pytest.approx(0.02) and t.e == pytest.approx(-0.02)
        assert src.crs.to_epsg() == 32633
        assert src.tags()["ppg_heights"] == "world"
    point = np.array([[6.0, 0.8, 0.0]])
    world = frame.to_world(point)[0]
    z, tr = relief.load(ch, "height")
    d = _value(z, tr, relief.uv_of_ship(point[:, :2], frame.sign)[0])
    expected = frame.to_world(np.array([[6.0, 0.8, d]]))[0, 2]  # the chunk's height of the surface there
    assert expected == pytest.approx(d + frame.z0)
    slope = 0.6  # dz/dy of the hull there: 2 D y / b^2
    assert abs(_value(arr, t, world[:2]) - expected) < (slope + 0.2) * 0.02 * 1.5
    relief.export_world(ch, "slrm", tmp_path / "slrm.tif")
    with rasterio.open(tmp_path / "slrm.tif") as src:
        assert src.tags()["ppg_heights"] == "relative" and abs(np.nanmedian(src.read(1))) < 0.05
    ship_copy = tmp_path / "ship.tif"
    relief.export_ship(ch, "slrm", ship_copy)
    with rasterio.open(ship_copy) as src:
        assert src.tags()["ppg_v"] == "sign*y" and src.transform == tr


def test_stale_after_a_new_frame(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    frame = _set(ch, ends)
    assert not relief.is_stale(ch)
    relief.build(ch, cell=0.02)
    assert not relief.is_stale(ch)
    ship.set_frame(ch, frame.flipped())
    assert relief.is_stale(ch)


def test_stale_after_a_new_model(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    _set(ch, ends)
    relief.build(ch, cell=0.02)
    model = ch.product_path("model")
    st = model.stat()
    os.utime(model, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert relief.is_stale(ch)


def _draw(ch, eid):
    """Drawings on the elevation ``eid`` as the window keeps them, in (u, v)."""
    ch.meta.setdefault("drawings", {}).setdefault("frames", {})[eid] = {
        "units": [{"us": "1", "wkt": "POLYGON ((5 0.5, 6 0.5, 6 1, 5 1, 5 0.5))"}],
        "points": [{"u": 6.0, "v": 0.8, "name": "p"}],
        "lines": [breaklines.new_record("breakline", [(4.0, -0.5, None), (7.0, -0.5, None)])],  # as the window
        "stones": {eid: ["POLYGON ((3 0, 3.2 0, 3.2 0.2, 3 0))"]}}


def test_drawings_survive_a_rebuild(tmp_path):
    from shapely import wkt

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    _set(ch, ends)
    old = relief.build(ch, cell=0.02)["elevation"]
    _draw(ch, old)
    st = relief.build(ch, cell=0.03)
    new = st["elevation"]
    assert new != old and st["drawings"] == {"records": 4, "moved": False}
    frames = ch.meta["drawings"]["frames"]
    assert old not in frames and old not in [e["id"] for e in ch.meta["elevations"]]
    data = frames[new]
    assert data["points"][0]["u"] == 6.0 and data["points"][0]["v"] == 0.8
    assert wkt.loads(data["units"][0]["wkt"]).equals(wkt.loads("POLYGON ((5 0.5, 6 0.5, 6 1, 5 1, 5 0.5))"))
    assert list(data["stones"]) == [new]


def test_drawings_follow_a_new_frame(tmp_path):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    frame = _set(ch, ends)
    old = relief.build(ch, cell=0.02)["elevation"]
    _draw(ch, old)
    world = relief.plane_of(frame).point(np.array([6.0]), np.array([0.8]), np.array([0.0]))[0]
    moved = ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start", z0=frame.z0 + 0.5, x0=2.0))
    st = relief.build(ch, cell=0.02)
    assert st["drawings"] == {"records": 4, "moved": True}
    p = ch.meta["drawings"]["frames"][st["elevation"]]["points"][0]
    again = relief.plane_of(moved).point(np.array([p["u"]]), np.array([p["v"]]), np.array([0.0]))[0]
    assert np.abs(again[:2] - world[:2]).max() < 1e-9
    assert p["u"] == pytest.approx(8.0)
    line = ch.meta["drawings"]["frames"][st["elevation"]]["lines"][0]["coords"]
    assert line[0][:2] == pytest.approx([6.0, -0.5]) and line[0][2] is None


def test_a_failed_rebuild_keeps_the_relief(tmp_path, monkeypatch):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    _set(ch, ends)
    old = relief.build(ch, cell=0.02)
    _draw(ch, old["elevation"])

    def broken(*args, **kwargs):
        raise RuntimeError("cancelled")

    monkeypatch.setattr(relief, "_derive", broken)
    with pytest.raises(RuntimeError):
        relief.build(ch, cell=0.03)
    assert relief.entry(ch)["id"] == old["elevation"]
    assert [e["id"] for e in ch.meta["elevations"]] == [old["elevation"]]
    frames = ch.meta["drawings"]["frames"]
    assert list(frames) == [old["elevation"]] and frames[old["elevation"]]["points"][0]["u"] == 6.0
    assert relief.load(ch, "slrm")[0].shape == relief.load(ch, "height")[0].shape
    assert not (ch.data_dir / "shipwreck" / "relief.new").exists()


def test_tiles_give_the_whole_raster_result(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.archaeo.relief import compute

    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    _set(ch, ends)
    monkeypatch.setattr(relief, "TILE", 64)
    relief.build(ch, cell=0.02)
    z, _ = relief.load(ch, "height")
    for kind, (method, extra) in (("hillshade", ("multi_hillshade", {"directions": 4})), ("slrm", ("slrm", {}))):
        arr, _ = relief.load(ch, kind)
        whole = compute(z.astype(np.float64), 0.02, method, **extra).astype(np.float32)
        np.testing.assert_allclose(arr, whole, atol=1e-6, equal_nan=True)


def test_a_rebuild_failing_late_keeps_the_relief_and_its_drawings(tmp_path, monkeypatch):
    _doc, ch, ends = clinker_project(tmp_path, heading_deg=35)
    frame = _set(ch, ends)
    old = relief.build(ch, cell=0.02)
    _draw(ch, old["elevation"])
    before = {k: dict(v) for k, v in ch.meta["drawings"]["frames"].items()}
    ship.set_frame(ch, ship.Frame(ends[0], ends[1], "start", z0=frame.z0 + 0.5, x0=2.0))

    def broken(*args, **kwargs):
        raise RuntimeError("cancelled")

    monkeypatch.setattr(relief, "_move_drawings", broken)
    with pytest.raises(RuntimeError):
        relief.build(ch, cell=0.02)
    assert ch.meta["drawings"]["frames"] == before
    assert [e["id"] for e in ch.meta["elevations"]] == [old["elevation"]]
    assert relief.entry(ch)["id"] == old["elevation"]
    assert not (ch.data_dir / "shipwreck" / "relief.new").exists()
    assert (ch.data_dir / old["files"]["slrm"]).exists()
