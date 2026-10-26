"""Depth logs read in their many forms, matched to the photos by time; the chunk levelled and
scaled by the depths of its cameras, the time offset found, known positions placing it."""
from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pytest

import samples
from ppg_underwater import depthlog, levelling


def _utc(*args) -> float:
    return datetime(*args, tzinfo=timezone.utc).timestamp()


# ------------------------------------------------------------------ reading logs
def test_the_depth_file_of_aqualoc(tmp_path):
    p = tmp_path / "depth_sequence_6.csv"
    p.write_text("#timestamp [ns], depth [m]\n1542883311752336025,387.48\n1542883311765773057,387.39\n"
                 "1542883312765773057,388.40\n")
    log = depthlog.read_table(p)
    assert log.absolute and len(log) == 3 and log.times[0] == pytest.approx(1542883311.752336, abs=1e-5)
    assert log.depth_at([1542883312.2655], window=0)[0] == pytest.approx(387.895, abs=0.01)
    assert np.isnan(log.depth_at([1542883400.0])[0])  # beyond the log


def test_iso_times_and_a_pressure_column(tmp_path):
    p = tmp_path / "rov.csv"
    p.write_text("time;pressure (mbar)\n2024-06-01T10:00:00Z;1013.25\n2024-06-01T10:00:10Z;2018.0\n"
                 "2024-06-01T10:00:20Z;3023.0\n")
    log = depthlog.read_table(p)
    assert log.kind == "pressure" and log.absolute
    assert log.times[0] == _utc(2024, 6, 1, 10, 0, 0)
    assert log.depths[0] == pytest.approx(0.0, abs=1e-6)
    assert log.depths[1] == pytest.approx((2018.0 - 1013.25) * 100 / (1025 * 9.80665), rel=1e-6)
    fresh = depthlog.read_table(p, density=depthlog.FRESH_WATER)
    assert fresh.depths[1] > log.depths[1]


def test_a_dive_computer_export(tmp_path):
    p = tmp_path / "subsurface.csv"
    p.write_text('"dive number","date","time","sample time (min)","sample depth (m)","sample temperature (C)"\n'
                 '"1","2024-06-01","10:00:00","0:10","1.5","22"\n"1","2024-06-01","10:00:00","0:20","3.0","21"\n'
                 '"1","2024-06-01","10:00:00","1:00","6.0","20"\n')
    log = depthlog.read_table(p)
    assert log.absolute and log.times[0] == _utc(2024, 6, 1, 10, 0, 10) and log.times[-1] == _utc(2024, 6, 1, 10, 1, 0)
    assert list(log.depths) == [1.5, 3.0, 6.0]


def test_tables_without_header_in_feet_and_heights(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("0 1.0\n1 1.5\n2 2.0\n")
    log = depthlog.read_table(p)
    assert not log.absolute and list(log.depths) == [1.0, 1.5, 2.0]
    q = tmp_path / "b.csv"
    q.write_text("seconds,depth (ft)\n0,10\n5,20\n")
    assert depthlog.read_table(q).depths[1] == pytest.approx(6.096)
    r = tmp_path / "c.csv"
    r.write_text("t,z\n0,-3.0\n1,-3.5\n")
    assert list(depthlog.read_table(r).depths) == [3.0, 3.5]


def test_uddf(tmp_path):
    p = tmp_path / "log.uddf"
    p.write_text("""<?xml version="1.0" encoding="utf-8"?>
<uddf xmlns="http://www.streit.cc/uddf/3.2/" version="3.2.0"><profiledata><repetitiongroup id="rg">
<dive id="d1"><informationbeforedive><datetime>2024-06-01T10:00:00</datetime></informationbeforedive>
<samples><waypoint><divetime>0</divetime><depth>0.0</depth></waypoint>
<waypoint><divetime>30</divetime><depth>4.5</depth><temperature>294.2</temperature></waypoint>
<waypoint><divetime>60</divetime><depth>9.0</depth></waypoint></samples></dive>
</repetitiongroup></profiledata></uddf>""")
    log = depthlog.read_log(p)
    assert log.absolute and len(log) == 3 and log.times[1] == _utc(2024, 6, 1, 10, 0, 30) and log.depths[2] == 9.0


def test_the_time_of_a_photo(tmp_path):
    from PIL import Image

    assert depthlog.time_from_name("1542883311752336025.png") == pytest.approx(1542883311.752336)
    assert depthlog.time_from_name("frame000012.png") is None
    assert depthlog.time_from_name("GOPR_20190312_142355_123.jpg") == pytest.approx(_utc(2019, 3, 12, 14, 23, 55) + 0.123)
    assert depthlog.time_from_name("2019-03-12_14-23-55.jpg") == _utc(2019, 3, 12, 14, 23, 55)
    path = tmp_path / "IMG_0001.JPG"
    exif = Image.Exif()
    sub = exif.get_ifd(0x8769)
    sub[36867] = "2024:06:01 12:00:05"
    sub[37521] = "25"
    sub[36881] = "+02:00"
    Image.new("RGB", (8, 8)).save(path, exif=exif.tobytes())
    assert depthlog.time_from_exif(path) == pytest.approx(_utc(2024, 6, 1, 10, 0, 5) + 0.25)


def test_photos_matched_with_an_offset():
    class Cam:
        def __init__(self, key, t):
            self.key, self.path, self.label = key, f"/x/{key}.jpg", str(key)
            self.meta = {"underwater": {"time": t}}

    log = depthlog.DepthLog(np.arange(0, 100.0), np.linspace(0, 10, 100), absolute=True)
    cams = [Cam(0, 1000.0), Cam(1, 1050.0)]
    assert depthlog.match(cams, log) == {}  # another clock
    m = depthlog.match(cams, log, offset=-990.0, window=0)
    assert m[0] == pytest.approx(10 * 10 / 99) and m[1] == pytest.approx(10 * 60 / 99)
    rel = depthlog.DepthLog(np.arange(0, 100.0), np.linspace(0, 10, 100), absolute=False)
    m = depthlog.match(cams, rel, window=0)  # a log without dates starts with the first photo
    assert m[0] == pytest.approx(0.0) and m[1] == pytest.approx(10 * 50 / 99)


# ------------------------------------------------------------------ levelling
def _chunk(tmp_path, world_centres, depths=None, s=0.3, seed=1, heading=0.4):
    """A chunk whose aligned cameras look down from ``world_centres`` (metres, Z = -depth), in an
    internal frame scaled by ``s``, turned and moved; the depths of the cameras (with noise)."""
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.core.camera import Camera, Sensor

    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "p.ppg")
    chunk.sensors[0] = Sensor(0, "cam", 400, 300)
    rot = samples.random_rotation(seed)
    shift = np.array([3.0, -7.0, 11.0])
    to_internal = np.eye(4)
    to_internal[:3, :3] = s * rot
    to_internal[:3, 3] = shift
    rs = np.random.default_rng(seed)
    for k, c in enumerate(np.asarray(world_centres, float)):
        w = np.eye(4)
        w[:3, :3] = samples.looking_down(heading + 0.1 * k)
        w[:3, 3] = c
        internal = to_internal @ w
        internal[:3, :3] /= s  # a rotation again (world_from_cam in the internal frame)
        cam = Camera(key=k, label=f"c{k}", path=str(tmp_path / f"c{k}.jpg"), sensor_key=0, transform=internal)
        d = -c[2] + (0.02 * rs.standard_normal() if depths is None else 0.0)
        cam.meta["underwater"] = {"depth": float(d if depths is None else depths[k]), "time": 1000.0 + 2.0 * k}
        chunk.cameras.append(cam)
    return doc, chunk, to_internal


def _survey(n=40, seed=0):
    rs = np.random.default_rng(seed)
    xy = rs.uniform(-10, 10, (n, 2))
    z = -12.0 + 0.15 * xy[:, 0] + 0.5 * np.sin(xy[:, 1] / 3.0) + rs.uniform(-0.3, 0.3, n)  # following a sloping bottom
    return np.column_stack([xy, z])


def test_levelled_and_scaled_by_the_depths(tmp_path):
    world = _survey()
    doc, chunk, to_internal = _chunk(tmp_path, world)
    chunk.cameras[5].meta["underwater"]["depth"] += 3.0  # a wrong match of time
    rep = levelling.level(chunk)
    from pyarchinit_photogrammetry.core import crs as crs_mod

    assert chunk.crs == crs_mod.LOCAL and chunk.meta["metric"] and levelling.levelled(chunk)
    assert rep["scale"] == pytest.approx(1 / 0.3, rel=0.01)
    assert rep["outliers"] >= 1 and rep["rms"] < 0.03 and rep["residuals"]["5"] == pytest.approx(3.0, abs=0.1)
    heights = chunk.to_world(np.array([c.center for c in chunk.cameras]))[:, 2]
    good = [k for k in range(len(world)) if k != 5]
    assert np.sqrt(np.mean((heights[good] - world[good, 2]) ** 2)) < 0.03
    # the vertical: the real world's up taken to the new frame is up again
    up_internal = to_internal[:3, :3] @ np.array([0, 0, 1.0])
    up_new = chunk.transform[:3, :3] @ up_internal
    assert math.degrees(math.acos(np.clip(up_new[2] / np.linalg.norm(up_new), -1, 1))) < 0.3
    # an alignment done again resets the transform: no longer levelled
    chunk.transform = np.eye(4)
    assert not levelling.levelled(chunk)


def test_a_survey_at_constant_depth_needs_the_scale(tmp_path):
    rs = np.random.default_rng(3)
    world = np.column_stack([rs.uniform(-8, 8, (30, 2)), np.full(30, -15.0)])
    doc, chunk, to_internal = _chunk(tmp_path, world, depths=np.full(30, 15.0))
    with pytest.raises(ValueError):
        levelling.level(chunk)
    rep = levelling.level(chunk, scale=1 / 0.3)
    heights = chunk.to_world(np.array([c.center for c in chunk.cameras]))[:, 2]
    assert np.allclose(heights, -15.0, atol=1e-6) and rep["scale"] == pytest.approx(1 / 0.3)
    up_new = chunk.transform[:3, :3] @ (to_internal[:3, :3] @ np.array([0, 0, 1.0]))
    assert up_new[2] / np.linalg.norm(up_new) > 0.9999  # up, not down: the cameras look down


def test_the_time_offset_is_found(tmp_path):
    t = np.arange(0, 200.0, 0.1)
    depth = 12.0 + 2.0 * np.sin(t / 25.0) + 0.5 * np.sin(t / 7.0)
    log = depthlog.DepthLog(t + 5000.0, depth, absolute=True)
    times = np.linspace(20, 180, 40)
    world = np.column_stack([np.cos(times / 30.0) * 8, np.sin(times / 30.0) * 8,
                             -(12.0 + 2.0 * np.sin(times / 25.0) + 0.5 * np.sin(times / 7.0))])
    doc, chunk, _ = _chunk(tmp_path, world)
    for cam, tt in zip(chunk.cameras, times):
        cam.meta["underwater"]["time"] = 5000.0 + tt + 12.5  # the camera clock 12.5 s ahead
    res = levelling.offset_search(chunk, log, span=60.0)
    assert res["offset"] == pytest.approx(-12.5, abs=0.2) and res["rms"] < 0.02


def test_known_positions_place_the_chunk(tmp_path):
    world = _survey(30, seed=4)
    doc, chunk, _ = _chunk(tmp_path, world)
    heading = math.radians(37.0)
    rot = np.array([[math.cos(heading), -math.sin(heading)], [math.sin(heading), math.cos(heading)]])
    utm = world[:, :2] @ rot.T + np.array([512000.0, 4650000.0])
    positions = {k: tuple(utm[k]) for k in (0, 3, 7, 12, 20)}
    rep = levelling.level(chunk, positions=positions, positions_crs="EPSG:32633")
    assert chunk.crs == "EPSG:32633" and rep["positions"]["cameras"] == 5 and rep["positions"]["rms"] < 0.05
    placed = chunk.to_world(np.array([c.center for c in chunk.cameras]))
    assert np.sqrt(np.mean(np.sum((placed[:, :2] - utm) ** 2, axis=1))) < 0.05
    assert np.sqrt(np.mean((placed[:, 2] - world[:, 2]) ** 2)) < 0.03


def test_positions_by_photo_name_or_time(tmp_path):
    class Cam:
        def __init__(self, key, t):
            self.key, self.path, self.label = key, f"/x/IMG_{key}.JPG", f"IMG_{key}.JPG"
            self.meta = {"underwater": {"time": t}}

    a = tmp_path / "a.csv"
    a.write_text("photo,easting,northing\nIMG_0.JPG,10,20\nIMG_1.JPG,11,21\n")
    cams = [Cam(0, 1.7e9 + 100.0), Cam(1, 1.7e9 + 110.0)]
    assert depthlog.match_positions(cams, a) == {0: (10.0, 20.0), 1: (11.0, 21.0)}
    b = tmp_path / "b.csv"
    b.write_text("time,x,y\n1700000090,0,0\n1700000120,30,60\n")
    m = depthlog.match_positions(cams, b)
    assert m[0] == pytest.approx((10.0, 20.0)) and m[1] == pytest.approx((20.0, 40.0))
