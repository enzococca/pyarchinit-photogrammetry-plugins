"""The relief of the hull from above: an elevation of the program on a horizontal plane in the ship frame (u = x,
v = sign * y, d = ship z), its hillshade and local relief model, the copy north-up in the world."""
from __future__ import annotations

import numpy as np
import pytest
import rasterio
from hulls import clinker_project, clinker_section

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
    point = np.array([[6.0, 0.8, 0.0]])
    world = frame.to_world(point)[0]
    z, tr = relief.load(ch, "height")
    expected = _value(z, tr, relief.uv_of_ship(point[:, :2], frame.sign)[0])
    slope = 0.6  # dz/dy of the hull there: 2 D y / b^2
    assert abs(_value(arr, t, world[:2]) - expected) < (slope + 0.2) * 0.02 * 1.5
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
