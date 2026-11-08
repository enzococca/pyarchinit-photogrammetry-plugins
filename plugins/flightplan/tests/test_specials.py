"""Special points of a block: detail photos, panoramas, video, the descents and the ladder of photos.

The first five tests are the planner's own (assertions unchanged), moved to the neutral place of the tests
(45.0 N 10.0 E): the takeoff at the bottom of the area, the point 550 m north of it."""
from ppg_flightplan import design, specials, tiles
from ppg_flightplan import plan as pl
from ppg_flightplan.geometry import distance, offset
from ppg_flightplan.route import Route
from terrainfixtures import AREA, tile_rising_north


# ---- the terrain for the special points, also when no block follows it

def _flat_plan_with_a_special_point(descent, takeoff=(44.9970, 10.0000), where=(45.0020, 10.0000)):
    return {"nome": "campo", "camera": "mini4pro", "alta": False, "minutiBatteria": 18, "decollo": list(takeoff),
            "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in AREA], "quota": 60,
                         "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0, "velocita": 5,
                         "modo": "per_punto",
                         "speciali": [{"id": "s1", "nome": "Target", "tipo": "foto", "punto": list(where),
                                       "aggancio": False, "discesa": descent}]}]}


def test_a_special_point_that_descends_has_the_ground_measured_even_in_a_flat_flight(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _flat_plan_with_a_special_point({"modo": "suolo", "metri": 12})
    assert pl.special_outcomes(piano["blocchi"][0], piano)[0]["avvisi"] == ["senza_terreno"]
    assert design.takeoffs_to_measure(piano) is True
    assert design.refresh_terrain(piano).get("suolo") is True
    assert design.takeoffs_to_measure(piano) is False
    # the block did NOT become a terrain-following flight: it only learned where the ground is
    assert not piano["blocchi"][0].get("seguiTerreno") and not piano["blocchi"][0].get("quote")
    outcome = pl.special_outcomes(piano["blocchi"][0], piano)[0]
    assert outcome["avvisi"] == [] and abs(outcome["dalSuolo"] - 12.0) < 0.1
    # the point is north, higher than the takeoff: twelve metres from the ground are more above the takeoff
    assert outcome["quotaBassa"] > 15.0
    flown = pl.plan_routes(piano)[0]
    assert flown.flat_heights and not flown.follows_terrain


def test_a_special_point_put_outside_the_known_terrain_has_it_measured_again(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _flat_plan_with_a_special_point({"modo": "suolo", "metri": 12})
    design.refresh_terrain(piano)
    piano["blocchi"][0]["speciali"][0]["punto"] = [45.0130, 10.0000]        # a kilometre farther north
    assert design.takeoffs_to_measure(piano) is True
    design.refresh_terrain(piano)
    assert design.takeoffs_to_measure(piano) is False


def test_in_a_flat_flight_the_takeoff_of_the_leg_moves_only_the_heights_counted_from_the_ground(monkeypatch):
    """A flight at a fixed height holds from any takeoff; the descent "twelve metres from the ground" does
    not: taking off from the top, higher, that height must come down by the difference."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _flat_plan_with_a_special_point({"modo": "suolo", "metri": 12})
    design.refresh_terrain(piano)
    before = pl.plan_legs(piano)[-1]
    piano["decolli"] = {before.key: [45.0030, 10.0000]}
    design.refresh_terrain(piano)
    after = pl.plan_legs(piano)[-1]
    rise = piano["suoloDecolli"][before.key]["suolo"] - piano["quotaDecollo"]
    assert rise > 3.0
    for i, (a, b) in enumerate(zip(before.route.heights, after.route.heights)):
        s = after.route.stop(i)
        if s and s.from_ground:
            assert abs((a - b) - rise) < 0.2
        else:
            assert a == b == 60.0


def test_switching_the_terrain_off_keeps_what_the_special_points_need(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _flat_plan_with_a_special_point({"modo": "suolo", "metri": 12})
    piano["blocchi"][0]["seguiTerreno"] = True
    design.apply_terrain(piano)
    design.remove_terrain(piano)
    assert piano.get("terreno") and not piano["blocchi"][0].get("quote")


def test_taking_off_lower_the_descent_above_the_takeoff_does_not_end_in_the_ground(monkeypatch):
    """"Descend to 12 m above the takeoff" is counted from the real takeoff: if the leg takes off lower than
    planned that point comes down with it, and must be raised to five metres from the ground."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    # designed taking off from the top (north), with the special point near it: 12 m above the takeoff is fine
    piano = _flat_plan_with_a_special_point({"modo": "decollo", "metri": 12}, takeoff=(45.0030, 10.0000),
                                            where=(45.0025, 10.0000))
    design.refresh_terrain(piano)
    before = pl.plan_legs(piano)[-1]
    outcome = pl.special_outcomes(piano["blocchi"][0], piano)[0]
    assert outcome["quotaBassa"] == 12.0 and "alzato" not in outcome["avvisi"]
    # ...but that leg then takes off from the bottom, some metres lower
    piano["decolli"] = {before.key: [44.9970, 10.0000]}
    design.refresh_terrain(piano)
    after = pl.plan_legs(piano)[-1].route
    ground = specials.ground_of_grid(piano["terreno"])
    k = next(i for i, s in enumerate(after.stops) if s and s.low)
    from_ground = after.heights[k] - (ground(after.points[k]) - piano["suoloDecolli"][before.key]["suolo"])
    assert after.heights[k] > 12.0 and abs(from_ground - 5.0) < 0.3
    assert all(q == 60.0 for i, q in enumerate(after.heights) if after.stop(i) is None)


# ---- with_specials directly

CENTRE = (45.0, 10.0)


def _line(n=6, step=40.0, height=60.0, mode="per_punto") -> Route:
    """A straight route to the east, ``n`` waypoints ``step`` metres apart."""
    return Route(name="b", points=[offset(CENTRE, 90.0, step * i) for i in range(n)], height=height, mode=mode)


def _flat(level=100.0):
    return lambda p: level


def _free_photo(**extra) -> dict:
    sp = {"id": "f", "nome": "F", "tipo": "foto", "punto": list(offset(CENTRE, 0.0, 60.0)), "aggancio": False,
          "discesa": {"modo": "suolo", "metri": 10}}
    sp.update(extra)
    return sp


def test_the_kinds_and_constants_are_the_planners():
    assert specials.SPECIAL_KINDS == ("foto", "panoramica", "giro_video", "video_inizio", "video_fine")
    assert (specials.MIN_DISTANCE_FROM_GROUND, specials.MIN_SPECIAL_HEIGHT, specials.STEP_BETWEEN_WAYPOINTS,
            specials.TRANSFER_STEP, specials.VIDEO_RING_RADIUS, specials.VIDEO_RING_POINTS,
            specials.VIDEO_RING_SPEED, specials.LADDER_RATIO, specials.PANORAMA_PITCH,
            specials.VIDEO_RING_PITCH) == (5.0, 5.0, 1.0, 25.0, 5.0, 12, 1.0, 0.6, -10.0, -15.0)


def test_no_special_points_leave_the_route_as_it_is():
    p = _line()
    assert specials.with_specials(p, []) == (p, [])
    off = _free_photo(attivo=False)
    assert specials.with_specials(p, [off, {"tipo": "inventato", "punto": [45, 10]}]) == (p, [])


def test_a_free_photo_descends_vertically_and_climbs_back():
    p = _line()
    route, outcomes = specials.with_specials(p, [_free_photo()], _flat(100.0), 100.0)
    assert len(outcomes) == 1 and outcomes[0]["waypoint"] is None and outcomes[0]["avvisi"] == []
    assert outcomes[0]["quotaBassa"] == 10.0 and outcomes[0]["dalSuolo"] == 10.0
    added = list(zip(route.points, route.heights, route.stops))[len(p.points):]
    assert [z for _, z, _ in added] == [60.0, 10.0, 60.0]
    (top, _, s_top), (low, _, s_low), (back, _, _) = added
    target = tuple(_free_photo()["punto"])
    assert distance(low, target) < 0.1                       # the low waypoint is on the point
    assert distance(top, low) < 1.1 and distance(back, low) < 1.1   # vertical: a metre at most in plan
    assert "foto" in s_low.actions and s_low.low and s_low.from_ground
    assert s_top.head and not s_top.survey
    assert route.flat_heights and not route.follows_terrain
    assert route.pitch == -90


def test_with_the_ladder_each_step_is_six_tenths_of_the_previous():
    p = _line(height=60.0)
    route, outcomes = specials.with_specials(p, [_free_photo(scaletta=True, inclinazione=-90)], _flat(100.0), 100.0)
    steps = outcomes[0]["gradini"]
    assert steps == [36.0, 21.6, 13.0]
    assert outcomes[0]["quotaBassa"] == 10.0
    heights = route.heights[len(p.points):]
    assert heights == [60.0, 36.0, 21.6, 13.0, 10.0, 60.0]
    for a, b in zip([60.0] + steps, steps):
        assert abs(b - a * specials.LADDER_RATIO) < 0.1
    shooting = [s for s in route.stops[len(p.points):] if s and "foto" in s.actions]
    assert len(shooting) == 4                                # three steps and the low point


def test_never_below_five_metres_from_the_ground():
    p = _line(height=60.0)
    route, outcomes = specials.with_specials(p, [_free_photo(discesa={"modo": "suolo", "metri": 1})],
                                             _flat(100.0), 100.0)
    assert outcomes[0]["quotaBassa"] == 5.0 and outcomes[0]["dalSuolo"] == 5.0
    # above the takeoff, over ground 20 m higher than the takeoff: raised to five metres from it
    route, outcomes = specials.with_specials(p, [_free_photo(discesa={"modo": "decollo", "metri": 10})],
                                             _flat(120.0), 100.0)
    assert outcomes[0]["quotaBassa"] == 25.0 and "alzato" in outcomes[0]["avvisi"]
    assert min(route.heights) >= 20.0 + specials.MIN_DISTANCE_FROM_GROUND


def test_a_hooked_point_is_taken_on_the_nearest_waypoint():
    p = _line()
    near_third = offset(p.points[3], 0.0, 7.0)
    sp = {"id": "h", "nome": "H", "tipo": "foto", "punto": list(near_third), "inclinazione": -45}
    route, outcomes = specials.with_specials(p, [sp])
    assert outcomes[0]["waypoint"] == 3
    k = next(i for i, s in enumerate(route.stops) if s)
    assert route.points[k] == p.points[3] and route.stops[k].head and route.stops[k].survey
    assert "camera:-45" in route.stops[k].actions
    assert route.points[:3] == p.points[:3] and route.points[-2:] == p.points[-2:]


def test_without_ground_the_descent_from_the_ground_is_not_made():
    route, outcomes = specials.with_specials(_line(), [_free_photo()])
    assert outcomes[0]["avvisi"] == ["senza_terreno"] and outcomes[0]["quotaBassa"] is None


def test_a_panorama_is_a_ring_of_photos_and_the_video_ring_starts_and_stops():
    p = _line()
    pano = {"id": "p", "nome": "P", "tipo": "panoramica", "punto": list(p.points[2]), "scatti": 8}
    route, _ = specials.with_specials(p, [pano])
    photos = [s for s in route.stops if s and "foto" in s.actions]
    assert len(photos) == 8 and all(s.pitch == specials.PANORAMA_PITCH for s in photos)
    video = {"id": "v", "nome": "V", "tipo": "giro_video", "punto": list(p.points[2])}
    route, _ = specials.with_specials(p, [video])
    ring = [s for s in route.stops if s and s.direction is not None]
    assert len(ring) == specials.VIDEO_RING_POINTS + 1
    assert ring[0].actions == ("video_inizio",) and ring[-1].actions == ("video_fine",)


def test_wpml_degrees_and_numbers():
    assert specials.wpml_degrees(190.0) == -170.0 and specials.wpml_degrees(-30.0) == -30.0
    assert specials.num(60.0) == "60" and specials.num(-45.5) == "-45.5"
