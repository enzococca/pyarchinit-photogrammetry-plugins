"""The reference block: 846 x 822 m at 110 m with the Mini 4 Pro, the planner's own test case (18 strips
47 m apart, about 650 shots). If the port finds those numbers again, the arithmetic is the planner's."""
import math

import pytest

from ppg_flightplan import geometry as g
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import camera
from ppg_flightplan.route import MAX_POINTS_PER_LEG, Route, Stop, grid, ring, shot_points
from planfixtures import CENTRE, block_a, rectangle, reference_plan

MINI4 = camera("mini4pro")


def test_the_reference_grid_finds_its_18_strips():
    shots = grid(rectangle(846, 822), 110, 0.80, 0.70, MINI4)
    assert 620 <= len(shots) <= 660
    assert MINI4.footprint(110) * 0.30 == pytest.approx(47.3, abs=0.5)
    assert MINI4.footprint_short(110) * 0.20 == pytest.approx(23.6, abs=0.5)
    assert MINI4.gsd_cm(110, True) == pytest.approx(1.95, abs=0.02)
    assert MINI4.gsd_cm(110, False) == pytest.approx(3.91, abs=0.03)
    turns = g.simplify(shots, 1.0)
    assert len(turns) == 36
    assert g.length(turns) / 1000 == pytest.approx(15.6, abs=0.6)


def test_interval_5_seconds_at_48_mp():
    p, perimeter = pl.route_of_block(block_a(), {"camera": "mini4pro"})
    assert len(perimeter) == 4
    assert p.interval == 5.0
    assert p.effective_speed == pytest.approx(4.72, abs=0.05)
    assert 600 <= p.shots() <= 700
    assert 48 <= p.minutes() <= 65
    warnings = p.warnings(18)
    assert "velocita_ridotta" in warnings and "piu_tratte" in warnings and "quota_oltre_120" not in warnings
    loop = Route("anello", p.points, 75, photo_step=11.0, speed=4.0, min_interval=2.0)
    assert loop.interval == 3.0
    assert loop.effective_speed == pytest.approx(3.67, abs=0.05)


def test_legs_fit_one_battery():
    p, _ = pl.route_of_block(block_a(), {"camera": "mini4pro"})
    legs = p.legs(15)
    assert 3 <= len(legs) <= 6
    for leg in legs:
        assert leg.route.minutes() <= 15.5
        assert len(leg.route.points) <= MAX_POINTS_PER_LEG
        assert leg.route.points == p.points[leg.start:leg.end + 1]
    assert legs[0].start == 0 and legs[-1].end == len(p.points) - 1
    for a, b in zip(legs, legs[1:]):
        assert a.end == b.start                      # by interval: the next leg starts on the last point
    by_point, _ = pl.route_of_block(block_a("per_punto", False), {"camera": "mini4pro"})
    legs2 = by_point.legs(15)
    assert all(len(leg.route.points) <= MAX_POINTS_PER_LEG for leg in legs2)
    for a, b in zip(legs2, legs2[1:]):
        assert a.end + 1 == b.start                  # by point: from the point after
    assert sum(len(leg.route.points) for leg in legs2) == len(by_point.points)


def test_the_ring_looks_at_the_centre():
    plan = reference_plan()
    p, _ = pl.route_of_block(plan["blocchi"][1], plan)
    assert p.poi == CENTRE
    assert len(p.points) == 37 and p.points[0] == p.points[-1]
    assert p.length == pytest.approx(2 * math.pi * 200, rel=0.01)
    assert p.interval == 5.0                         # 16 m step at 4 m/s -> 4 s -> 5 s DJI, 3.2 m/s
    assert ring(CENTRE, 200)[0][0] > CENTRE[0]       # from the north, clockwise
    assert ring(CENTRE, 200)[9][1] > CENTRE[1]


def test_a_hand_route_can_close_or_aim():
    outline = [(44.9993, 9.9987), (44.9995, 10.0012), (45.0007, 10.0015), (45.0011, 9.9992), (45.0001, 9.9982)]
    loop = {"tipo": "percorso", "nome": "Contorno", "punti": outline, "chiuso": True, "versoCentro": True,
            "quota": 60, "inclinazione": -40, "avanti": 80, "velocita": 4}
    p, _ = pl.route_of_block(loop, {"camera": "mini4pro"})
    assert p.points[0] == p.points[-1] and len(p.points) == 6            # closed on the first point
    assert p.poi is not None and g.distance(p.poi, g.centroid(outline)) < 0.01
    road = {"tipo": "percorso", "nome": "Strada", "punti": outline[:3], "quota": 50, "inclinazione": -90,
            "poi": [45.0023, 10.0012]}
    q, _ = pl.route_of_block(road, {"camera": "mini4pro"})
    assert q.points == outline[:3] and q.poi == (45.0023, 10.0012)


def test_terrain_heights_per_point():
    p = Route(name="salita", points=[(45.0, 10.0), (45.001, 10.0), (45.002, 10.0)], height=60.0,
              heights=[60.0, 72.0, 84.0], mode="per_punto")
    assert p.height_at(1) == 72.0
    assert p.follows_terrain and p.max_height == 84.0
    assert Route(name="piano", points=p.points, height=60.0).height_at(1) == 60.0


def test_geometry_helpers():
    a, b = (45.0, 10.0), (45.001, 10.0)
    assert g.distance(a, b) == pytest.approx(111.32, abs=1e-6)
    assert g.bearing(a, b) == pytest.approx(0.0)
    assert g.bearing(a, (45.0, 10.001)) == pytest.approx(90.0)
    east = g.offset(a, 90.0, 50.0)
    assert g.distance(a, east) == pytest.approx(50.0, abs=1e-6) and east[0] == a[0]
    t, d = g.project_on_segment((45.0005, 10.0001), a, b)
    assert t == pytest.approx(0.5) and d == pytest.approx(0.1 * g.metres_per_degree_lon(45.0) / 1000, rel=1e-3)
    line = [a, g.interpolate(a, b, 0.5), b]
    assert g.simplify(line, 0.01) == [a, b]
    assert len(g.densify([a, b], 10.0)) == 13        # 0, 10, ..., 110 and the end
    assert g.length([a, b, a]) == pytest.approx(222.64, abs=1e-6)


def test_shot_points():
    p, _ = pl.route_of_block(block_a("per_punto", False), {"camera": "mini4pro"})
    shots = shot_points(p)
    assert len(shots) == len(p.points) and all(h == 110.0 for _, h in shots)
    q, _ = pl.route_of_block(block_a(), {"camera": "mini4pro"})
    shots = shot_points(q)
    assert abs(len(shots) - q.shots()) <= 1
    assert all(h == 110.0 for _, h in shots)
    straight = [g.distance(a[0], b[0]) for a, b in zip(shots, shots[1:])]
    # on the strips (not across a turn) the shots are one effective step apart
    on_strips = [d for d in straight if d > q.effective_step - 1.0]
    assert len(on_strips) > 0.9 * len(straight)
    assert all(d == pytest.approx(q.effective_step, abs=0.5) for d in on_strips)
    # heights follow the waypoints linearly, as the drone flies between them
    rising = Route(name="salita", points=[(45.0, 10.0), (45.001, 10.0)], height=60.0, heights=[60.0, 80.0],
                   photo_step=20.0, speed=5.0)
    up = shot_points(rising)
    assert up[0][1] == pytest.approx(60.0) and up[-1][1] > up[0][1]
    for (pt, h) in up:
        assert h == pytest.approx(60.0 + 20.0 * g.distance(rising.points[0], pt) / rising.length, abs=1e-6)
    # a special photo is taken at its stop
    special = Route(name="speciale", points=p.points[:3], height=110.0, mode="per_punto",
                    stops=[None, Stop(actions=("foto",), survey=False, kind="foto"), None])
    sp = shot_points(special)
    assert len(sp) == special.shots() == 3 and [pt for pt, _ in sp] == special.points


def test_the_planner_metres_against_the_ellipsoid():
    """The planner's flat-earth metres differ from the true ones by a few parts per thousand, in either
    direction: north-south its metres are longer than real below about 55 degrees (more overlap than nominal),
    east-west they are shorter than real off the equator (strips across longitude a little farther apart: less side
    overlap, 0.17 % of the spacing at 45 degrees). The coverage computed with pyproj is the truth."""
    from pyproj import Geod

    from ppg_flightplan import geometry

    geod = Geod(ellps="WGS84")
    for lat in (0.0, 30.0, 45.0, 60.0, 75.0):
        a, north, east = (lat, 10.0), (lat + 0.01, 10.0), (lat, 10.01)
        true_ns = geod.inv(10.0, lat, 10.0, lat + 0.01)[2]
        true_ew = geod.inv(10.0, lat, 10.01, lat)[2]
        if lat:
            assert geometry.distance(a, east) < true_ew                   # east-west: short off the equator
        else:
            assert geometry.distance(a, east) == pytest.approx(true_ew, rel=1e-5)    # (equal on it)
        assert (geometry.distance(a, north) > true_ns) == (lat < 55.0)     # north-south: long below ~55 degrees
    ratio = geometry.distance((45.0, 10.0), (45.0, 10.01)) / geod.inv(10.0, 45.0, 10.01, 45.0)[2]
    assert 1 - ratio == pytest.approx(0.0017, abs=0.0002)
