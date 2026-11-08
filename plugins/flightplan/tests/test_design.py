"""From the result wanted to the plan: height, overlaps, obliques and the flight following the terrain.

Ported from the planner's tests, moved to the neutral place of the tests (45.0 N 10.0 E); then the new parts
of the plugin: margin, double grid, obliques forced, GSD of every shot and the design warnings."""
import numpy as np
import pytest

from ppg_flightplan import design, tiles
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import camera
from ppg_flightplan.dem import RasterDem
from ppg_flightplan.geometry import distance
from ppg_flightplan.route import Route
from planfixtures import rectangle
from terrainfixtures import AREA, fake_tile, tile_rising_north, write_dem

SOUTH, NORTH = (44.9970, 10.0000), (45.0030, 10.0000)
FAR = [36.0, 20.0]                 # more than 1000 km away, at sea: a takeoff left there by another plan


def test_the_menu_of_goals_is_complete_and_coherent():
    ids = [o["id"] for o in design.GOALS]
    assert ids == ["orto", "dem", "modello3d", "ispezione", "ricognizione", "gsd"]
    for o in design.GOALS:
        assert o["minimo"] < o["valore"] < o["massimo"]
        assert o["nome"] and o["descrizione"] and o["unita"]
    with pytest.raises(ValueError):
        design.goal("inventato")


def test_the_height_is_the_inverse_of_the_gsd():
    cam = camera("mini4pro")
    q = design.height_for_gsd(cam, 2.0, high=True)
    assert cam.gsd_cm(q, high=True) == pytest.approx(2.0, abs=0.01)


def test_a_dem_at_5_cm_asks_a_gsd_finer_than_the_cell():
    assert design.gsd_for_goal("dem", 5.0) == pytest.approx(2.0)
    assert design.gsd_for_goal("orto", 2.0) == 2.0
    with pytest.raises(ValueError):
        design.gsd_for_goal("orto", 0)


def test_the_orthophoto_is_one_nadir_block():
    r = design.design(AREA, "orto", 2.0)
    blocks = r["piano"]["blocchi"]
    assert len(blocks) == 1
    assert blocks[0]["inclinazione"] == -90
    assert r["gsd"] == pytest.approx(2.0, abs=0.05)
    assert r["conti"]["totale"]["scatti"] > 0
    assert any("GSD" in m for m in r["motivi"])


def test_the_3d_model_adds_two_crossed_oblique_passes():
    r = design.design(AREA, "modello3d", 2.0)
    blocks = r["piano"]["blocchi"]
    assert len(blocks) == 3
    assert [b["inclinazione"] for b in blocks] == [-90, -45, -45]
    assert blocks[1]["orientamento"] != blocks[2]["orientamento"]
    assert any("oblique" in m for m in r["motivi"])


def test_an_impossible_detail_does_not_go_beyond_120_metres():
    r = design.design(AREA, "orto", 10.0)          # 10 cm/px would ask more than 500 m
    assert r["quota"] == design.MAX_HEIGHT
    assert any("120" in m for m in r["motivi"])
    r2 = design.design(AREA, "ispezione", 0.1)     # nor below the minimum
    assert r2["quota"] == design.MIN_HEIGHT


def test_the_area_needs_at_least_three_vertices():
    with pytest.raises(ValueError):
        design.design([(44.99, 9.99), (45.0, 10.0)], "orto", 2.0)


def test_the_strips_follow_the_long_side():
    narrow = [(44.9970, 9.9970), (44.9970, 9.9990), (45.0070, 9.9990), (45.0070, 9.9970)]
    r = design.design(narrow, "orto", 3.0)        # the long side is north-south
    assert r["piano"]["blocchi"][0]["orientamento"] == 0


def test_the_height_of_every_point_keeps_the_distance_from_the_ground():
    """Where the terrain rises 30 m the drone rises 30 m: the GSD stays the one wanted."""
    heights = design.terrain_heights([(44.99, 9.99)] * 3, agl=80.0, ground_takeoff=250.0,
                                     grounds=[250.0, 265.0, 280.0])
    assert heights == [80.0, 95.0, 110.0]
    # without terrain data the nominal height stays, nothing is made up
    assert design.terrain_heights([(0, 0)], 80.0, 250.0, [None]) == [80.0]
    # The terrain rises 50 m: to stay 100 m FROM THE GROUND 150 m above the takeoff are needed, and given.
    # Cutting at 120 would mean flying 70 m from the ground instead of 100 - and on a steeper slope ending
    # below the ground. Below, it stays above the minimum: flying higher than wanted does no harm.
    high = design.terrain_heights([(0, 0), (0, 0)], 100.0, 250.0, [300.0, 100.0])
    assert high == [150.0, design.MIN_HEIGHT]
    # the only ceiling is the technical one of the drone
    assert design.terrain_heights([(0, 0)], 100.0, 0.0, [900.0]) == [design.CEILING_ABOVE_TAKEOFF]


def test_the_plan_following_the_terrain_gives_a_height_to_every_waypoint(monkeypatch):
    def fake(z, x, y):
        return fake_tile(250.0 + (y % 5) * 10)      # different tiles, different heights
    monkeypatch.setattr(tiles, "_tile_from_network", fake)
    r = design.design(AREA, "orto", 3.0, follow_terrain=True)
    piano = r["piano"]
    assert piano["seguiTerreno"] is True and "quotaDecollo" in piano
    heights = piano["blocchi"][0]["quote"]
    route, _ = pl.route_of_block(piano["blocchi"][0], piano)
    assert len(heights) == len(route.points)
    assert route.heights == heights
    assert any("segue il terreno" in m for m in r["motivi"])
    # and one can go back
    design.remove_terrain(piano)
    assert "quote" not in piano["blocchi"][0] and "seguiTerreno" not in piano


def test_without_network_the_plan_comes_out_anyway_at_a_constant_height(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: None)
    r = design.design(AREA, "orto", 3.0, follow_terrain=True)
    assert "quote" not in r["piano"]["blocchi"][0]
    assert any("non disponibili" in m for m in r["motivi"])
    assert r["conti"]["totale"]["scatti"] > 0


def test_the_heights_follow_the_points_when_the_route_splits_into_legs():
    """The defect one would pay in flight: from the second battery on, the heights must be those of the
    points of that leg, not the first of the list."""
    points = [(44.99 + i * 0.0005, 10.0) for i in range(40)]
    p = Route(name="lunga", points=points, height=60.0, heights=[60.0 + i for i in range(40)],
              mode="per_punto", speed=3.0, photo_step=10.0)
    legs = p.legs(battery_minutes=1.0)
    assert len(legs) > 1
    for leg in legs:
        assert len(leg.route.heights) == len(leg.route.points)
        assert leg.route.heights[0] == 60.0 + leg.start
        assert leg.route.height_at(0) == 60.0 + leg.start
    resumed = p.from_resume(10)
    assert resumed.heights[0] == 70.0 and len(resumed.heights) == len(resumed.points)


def test_the_numbers_of_the_block_say_how_high_it_climbs():
    p = Route(name="salita", points=[(44.99, 10.0), (45.0, 10.0)], height=60.0, heights=[60.0, 130.0])
    data = p.to_json(18.0, camera("mini4pro"), high=True)
    assert data["seguiTerreno"] is True and data["quotaMin"] == 60.0 and data["quotaMax"] == 130.0
    # following the terrain the distance from the ground is the one wanted: the warning is not "beyond the
    # law" but "the stock drone does not climb that high"
    assert "terreno_oltre_120" in data["avvisi"] and "quota_oltre_120" not in data["avvisi"]


# ---- per-leg takeoffs and the terrain

def _plan(**extra) -> dict:
    piano = {"nome": "rilievo", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in AREA], "quota": 60,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "per_punto", "seguiTerreno": True}]}
    piano.update(extra)
    return piano


def test_the_leg_taking_off_from_the_top_flies_lower_from_its_takeoff(monkeypatch):
    """The heights of the file are measured from where the drone really takes off. If a leg starts from the
    top of the mound and the plan referred to the plain, its heights go down by the drop, or it would fly
    higher than wanted - and the other way round, lower."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan(decollo=list(SOUTH))
    design.apply_terrain(piano)
    base = pl.plan_legs(piano)[0]
    # now the same leg takes off from the top, to the north
    piano["decolli"] = {base.key: list(NORTH)}
    assert design.takeoffs_to_measure(piano) is True
    design.refresh_terrain(piano)
    assert design.takeoffs_to_measure(piano) is False
    from_top = pl.plan_legs(piano)[0]
    drop = piano["suoloDecolli"][base.key]["suolo"] - piano["quotaDecollo"]
    assert drop > 3.0                          # the top is higher
    differences = [a - b for a, b in zip(base.route.heights, from_top.route.heights)]
    inside = [d for d, a, b in zip(differences, base.route.heights, from_top.route.heights)
              if 10.0 < b < 120.0 and 10.0 < a < 120.0]
    assert inside and all(abs(d - drop) < 0.2 for d in inside)


def test_without_its_own_takeoff_the_leg_stays_as_it_is(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan()
    design.apply_terrain(piano)
    leg = pl.plan_legs(piano)[0]
    assert leg.route.heights == piano["blocchi"][0]["quote"][:len(leg.route.points)]
    assert design.takeoffs_to_measure(piano) is False


def test_a_takeoff_moved_afterwards_is_not_used_until_measured_again(monkeypatch):
    """If the saved point is no longer that one, the correction is not applied: better none than a wrong
    one, and it is measured again when saving."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan(decollo=list(SOUTH), decolli={"0-1": list(NORTH)})
    design.apply_terrain(piano)
    corrected = pl.plan_legs(piano)[0].route.heights
    piano["decolli"]["0-1"] = [45.0000, 10.0000]          # moved halfway up
    untouched = pl.plan_legs(piano)[0].route.heights
    assert untouched == piano["blocchi"][0]["quote"][:len(untouched)]
    assert untouched != corrected
    assert design.takeoffs_to_measure(piano) is True


def test_if_the_plan_climbs_above_the_stock_limit_it_is_said_with_the_numbers():
    """A stock DJI stops at 120 m over the takeoff. If the plan asks more it must be said, and how much
    would be left from the ground leaving the limit as it is."""
    # terrain rising 80 m, distance wanted 60: 140 m over the takeoff are needed
    warnings = design.height_warnings([60.0, 100.0, 140.0], [250.0, 290.0, 330.0], 250.0)
    assert len(warnings) == 1
    a = warnings[0]
    assert a["tipo"] == "oltre_limite_di_serie" and a["quotaMassima"] == 140.0
    assert a["dalSuoloSeNonAlzi"] == 40.0          # at 120 m over the takeoff, up there 40 m from the ground
    assert a["pericolo"] is False
    # a climb of 130 m: at 120 over the takeoff one would be UNDER the terrain
    serious = design.height_warnings([60.0, 190.0], [250.0, 380.0], 250.0)[0]
    assert serious["dalSuoloSeNonAlzi"] == -10.0 and serious["pericolo"] is True
    # below the limit nothing is said
    assert design.height_warnings([60.0, 95.0], [250.0, 285.0], 250.0) == []
    assert design.height_warnings([], [], 250.0) == []


# ---- the plan carries the terrain with it

def test_the_plan_carries_the_terrain_with_it(monkeypatch):
    """The heights are counted from where the drone really takes off. With the grid inside the plan the
    ground under any takeoff can be read without network, and the heights brought back to that point.
    (The planner's check that the grid travels inside the KML belongs to the KML writer.)"""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan(decollo=list(SOUTH))
    design.apply_terrain(piano)
    g = piano["terreno"]
    assert g["righe"] * g["colonne"] == len(g["quote"]) <= design.NODI_MASSIMI
    assert g["riferimento"] == [44.997, 10.0] and g["suoloRiferimento"] == piano["quotaDecollo"]
    # it covers the area with a margin, so one can take off a little outside too
    assert g["sud"] < min(p[0] for p in AREA) and g["ovest"] < min(p[1] for p in AREA)
    assert g["sud"] + g["righe"] * g["passoLat"] > max(p[0] for p in AREA)


def test_a_large_area_widens_the_mesh_instead_of_swelling_the_file(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    large = [(44.98, 9.97), (44.98, 10.03), (45.03, 10.03), (45.03, 9.97)]        # ~5 x 5 km
    piano = {"nome": "grande", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in large], "quota": 110,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "intervallo", "seguiTerreno": True}]}
    design.apply_terrain(piano)
    g = piano["terreno"]
    assert g["righe"] * g["colonne"] <= design.NODI_MASSIMI
    assert g["passoM"] > design.PASSO_GRIGLIA_M


def test_switching_the_terrain_off_the_plan_leaves_it_at_home(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan()
    design.apply_terrain(piano)
    assert "terreno" in piano
    design.remove_terrain(piano)
    assert "terreno" not in piano and "suoloDecolli" not in piano


def test_a_plan_saved_before_gets_the_terrain_at_the_first_save(monkeypatch):
    """Heights in order, but no grid (a plan of an older planner): nothing was "old", and the plan stayed
    without terrain."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan()
    design.apply_terrain(piano)
    heights = list(piano["blocchi"][0]["quote"])
    for key in ("terreno", "puntoDecollo", "suoloDecolli"):
        piano.pop(key, None)
    assert design.blocks_with_old_heights(piano) == []
    assert design.takeoffs_to_measure(piano) is True
    outcome = design.refresh_terrain(piano)
    assert outcome["rifatti"] == [0] and "terreno" in piano
    assert piano["blocchi"][0]["quote"] == heights                     # the heights stay those
    assert design.takeoffs_to_measure(piano) is False


# ---- a takeoff left in another region

def _plan_with_the_takeoff_elsewhere(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = _plan(decollo=list(FAR))
    piano["blocchi"][0]["quota"] = 40
    return piano


def test_a_takeoff_kilometres_from_the_flight_is_not_the_reference(monkeypatch):
    """A field case of the planner: the takeoff had stayed far away, from another plan, and much lower.
    Every waypoint ended at the 500 m ceiling, whatever height was written."""
    piano = _plan_with_the_takeoff_elsewhere(monkeypatch)
    assert pl.takeoff_far(piano) > 1_000_000
    design.apply_terrain(piano)
    heights = piano["blocchi"][0]["quote"]
    assert max(heights) < 80 and min(heights) >= 10                 # heights of a survey, not the ceiling
    assert pl.compute(piano)["decolloLontanoM"] > 1_000_000
    # the declared takeoff is remembered: "was it moved?" must not always answer yes
    assert piano["puntoDecollo"] == piano["decollo"] and design.takeoffs_to_measure(piano) is False
    assert distance(tuple(piano["riferimentoQuote"]), tuple(piano["blocchi"][0]["area"][0])) < 2000
    # and the grid stays around the site, with its fine mesh
    assert piano["terreno"]["passoM"] <= 25.0


def test_with_the_takeoff_in_its_place_nothing_changes(monkeypatch):
    piano = _plan_with_the_takeoff_elsewhere(monkeypatch)
    piano["decollo"] = list(SOUTH)
    assert pl.takeoff_far(piano) is None and pl.compute(piano)["decolloLontanoM"] is None
    design.apply_terrain(piano)
    assert "riferimentoQuote" not in piano and piano["puntoDecollo"] == piano["decollo"]


def test_heights_all_at_the_ceiling_are_a_danger_to_say():
    warnings = design.height_warnings([500.0] * 12, [594.0] * 12, 85.0)
    assert warnings[0]["alTetto"] is True and warnings[0]["pericolo"] is True


def test_a_far_leg_takeoff_does_not_move_the_heights(monkeypatch):
    piano = _plan_with_the_takeoff_elsewhere(monkeypatch)
    piano["decollo"] = list(SOUTH)
    design.apply_terrain(piano)
    before = pl.plan_legs(piano)[0]
    piano["decolli"] = {before.key: list(FAR)}
    design.refresh_terrain(piano)
    assert pl.plan_legs(piano)[0].route.heights == before.route.heights
    assert piano["terreno"]["passoM"] <= 25.0


# ------------------------------------------------------------------ new in the plugin

def _bbox_m(area):
    lats, lons = [p[0] for p in area], [p[1] for p in area]
    return (distance((min(lats), min(lons)), (min(lats), max(lons))),
            distance((min(lats), min(lons)), (max(lats), min(lons))))


def test_margin_enlarges_the_area():
    rect = rectangle(300, 200)
    plain = design.design(rect, "orto", 2.0)
    wide = design.design(rect, "orto", 2.0, margin_m=20)
    w0, h0 = _bbox_m(plain["piano"]["blocchi"][0]["area"])
    w1, h1 = _bbox_m(wide["piano"]["blocchi"][0]["area"])
    assert w1 - w0 == pytest.approx(40, abs=2) and h1 - h0 == pytest.approx(40, abs=2)
    assert wide["conti"]["totale"]["scatti"] > plain["conti"]["totale"]["scatti"]
    assert wide["piano"]["margine"] == 20.0 and "margine" not in plain["piano"]
    assert any("20 m" in m for m in wide["motivi"])


def test_double_grid_adds_the_crossed_block():
    r = design.design(AREA, "orto", 2.0, double_grid=True)
    a, a2 = r["piano"]["blocchi"]
    assert (a2["orientamento"] - a["orientamento"]) % 180 == 90
    assert a2["quota"] == a["quota"] and a2["inclinazione"] == -90 and a2["nome"].startswith("A2 nadirale")
    assert a2["area"] == a["area"] and a2["avanti"] == a["avanti"] and a2["lato"] == a["lato"]
    assert any("incrociata" in m for m in r["motivi"])


def test_obliques_forced():
    r = design.design(AREA, "orto", 2.0, obliques=True)
    blocks = r["piano"]["blocchi"]
    assert len(blocks) == 3 and [b["inclinazione"] for b in blocks[1:]] == [-45, -45]
    assert any("su richiesta" in m for m in r["motivi"])
    off = design.design(AREA, "modello3d", 2.0, obliques=False)
    assert len(off["piano"]["blocchi"]) == 1
    assert any("tolte" in m for m in off["motivi"])


def test_custom_camera_goes_into_the_plan():
    from ppg_flightplan.camera import manual_camera

    cam = manual_camera("Mia camera", 13.2, 8.8, 8.8, 5472)
    r = design.design(AREA, "orto", 2.0, custom_camera=cam, high=False)
    assert r["piano"]["camera"] == cam.id and r["piano"]["cameraPersonale"] == cam.to_json()
    assert r["gsd"] == pytest.approx(cam.gsd_cm(r["quota"], False), abs=0.01)
    assert pl.plan_camera(r["piano"]) == cam


def _sloping_dem(tmp_path, rise=30.0):
    """A DEM in degrees rising ``rise`` metres from the south edge to the north edge of AREA."""
    step = 0.00002
    south, north = AREA[0][0], AREA[2][0]
    lats = 45.004 - (np.arange(400) + 0.5) * step              # cell centres, from the north
    row = 200.0 + rise * (lats - south) / (north - south)
    heights = np.repeat(row[:, None], 400, axis=1)
    path = write_dem(tmp_path / "pendio.tif", heights, x0=9.996, y0=45.004, step=step)
    return RasterDem(path), (lambda lat: 200.0 + rise * (lat - south) / (north - south))


def test_shot_report_gsd_with_terrain(tmp_path):
    dem, ground_of = _sloping_dem(tmp_path)
    r = design.design(AREA, "orto", 2.0, takeoff=SOUTH)
    report = design.shot_report(r["piano"], dem=dem)
    assert len(report) == 1
    shots = report[0]["shots"]
    assert len(shots) == r["conti"]["totale"]["scatti"]
    assert report[0]["nominal_gsd_cm"] == pytest.approx(r["gsd"], abs=0.01)
    low = min(shots, key=lambda s: s["ground"])
    high = max(shots, key=lambda s: s["ground"])
    expected_rise = ground_of(high["latlon"][0]) - ground_of(low["latlon"][0])
    assert expected_rise > 25.0                     # shots sit half a step inside the 30 m rise of the area
    assert low["agl"] - high["agl"] == pytest.approx(expected_rise, abs=1.0)
    assert high["gsd_cm"] < low["gsd_cm"]
    assert high["gsd_cm"] / low["gsd_cm"] == pytest.approx(high["agl"] / low["agl"], rel=1e-9)
    # following the terrain: every shot at the nominal distance from the ground, every GSD the designed one
    f = design.design(AREA, "orto", 2.0, takeoff=SOUTH, follow_terrain=True, dem=dem)
    assert "quote" in f["piano"]["blocchi"][0]
    for s in design.shot_report(f["piano"], dem=dem)[0]["shots"]:
        assert s["agl"] == pytest.approx(f["quota"], abs=1.0)
        assert s["gsd_cm"] == pytest.approx(f["gsd"], rel=0.02)
    # without any ground the report says it does not know
    blind = design.shot_report(r["piano"])
    assert all(s["ground"] is None and s["agl"] is None and s["gsd_cm"] is None for s in blind[0]["shots"])


def test_design_warnings(tmp_path):
    dem, _ = _sloping_dem(tmp_path)
    cam = camera("mini4pro")
    # a constant 40 m over a terrain rising 30 m, taking off at the bottom: the overlap falls up there
    r = design.design(AREA, "orto", 2.0, takeoff=SOUTH)
    r["piano"]["blocchi"][0]["quota"] = 40
    codes = {w["tipo"]: w for w in design.design_warnings(r["piano"], dem=dem)}
    assert "sovrapposizione_calata" in codes and "gsd_oltre" not in codes
    lowest = min(s["agl"] for s in design.shot_report(r["piano"], dem=dem)[0]["shots"])
    assert codes["sovrapposizione_calata"]["peggiore"] == pytest.approx(
        1 - cam.footprint(40) * 0.3 / cam.footprint(lowest), abs=0.001)
    # the same taking off at the top: the ground falls away, the GSD gets coarser, the overlap grows
    r["piano"]["decollo"] = list(NORTH)
    codes = {w["tipo"]: w for w in design.design_warnings(r["piano"], dem=dem)}
    assert "gsd_oltre" in codes and "sovrapposizione_calata" not in codes
    assert codes["gsd_oltre"]["massimo"] > 1.25 * codes["gsd_oltre"]["nominale"]
    # a sliver: one strip of one shot
    sliver = design.design(rectangle(20, 30), "orto", 2.0)
    codes = {w["tipo"]: w for w in design.design_warnings(sliver["piano"])}
    assert codes["strisciata_corta"]["scattiMinimi"] < 3
    # and the plain survey on flat ground says nothing
    assert design.design_warnings(design.design(AREA, "orto", 2.0)["piano"]) == []


def test_too_close_to_the_ground_is_said(tmp_path):
    dem, _ = _sloping_dem(tmp_path, rise=40.0)
    r = design.design(AREA, "orto", 2.0, takeoff=SOUTH)
    r["piano"]["blocchi"][0]["quota"] = 40
    codes = {w["tipo"]: w for w in design.design_warnings(r["piano"], dem=dem)}
    assert codes["troppo_vicino_al_suolo"]["minimo"] < 10.0


# ------------------------------------------------------------------ fix round 1

def _true_strips(b, piano):
    """The shots of every strip as route.grid makes them (its own arithmetic, in its own frame)."""
    import math

    from ppg_flightplan.geometry import METRES_PER_DEGREE, centroid, metres_per_degree_lon
    from ppg_flightplan.route import _spans

    cam = pl.plan_camera(piano)
    area = [tuple(p) for p in b["area"]]
    c = centroid(area)
    k = metres_per_degree_lon(c[0])
    a = math.radians(b["orientamento"])
    ca, sa = math.cos(a), math.sin(a)
    edge = []
    for p in area:
        x, y = (p[1] - c[1]) * k, (p[0] - c[0]) * METRES_PER_DEGREE
        edge.append((x * ca + y * sa, -x * sa + y * ca))
    min_x, max_x = min(e[0] for e in edge), max(e[0] for e in edge)
    width = max_x - min_x
    step = cam.footprint(b["quota"]) * (1 - b["lato"] / 100)
    photo = cam.footprint_short(b["quota"]) * (1 - b["avanti"] / 100)
    strips = max(1, round(width / max(step, 1.0)))
    out = []
    for s in range(strips):
        x = min_x + width * (s + 0.5) / strips
        out.append(sum(max(1, round((e - st) / max(photo, 0.5))) for st, e in _spans(x, edge)))
    return out


@pytest.mark.parametrize("heading", [0, 30, 60, 90, 135])
@pytest.mark.parametrize("mode", ["per_punto", "intervallo"])
def test_strips_are_found_on_any_heading(heading, mode):
    """The shots fall in the strips of the grid whatever its heading: no strip split in two, no two strips
    merged (the corner strips of a rotated rectangle are really short, and only those are said short)."""
    area = rectangle(630, 670)
    r = design.design(area, "orto", 2.0, mode=mode)
    b = r["piano"]["blocchi"][0]
    b["orientamento"] = heading
    true = _true_strips(b, r["piano"])
    shots = design.shot_report(r["piano"])[0]["shots"]
    strips = design._strips(b, r["piano"], shots)
    assert len(strips) == len(true)
    if mode == "per_punto":
        assert [len(s) for s in strips] == true
    else:                   # by interval: about the same shots per strip, the turns left out
        assert all(abs(len(s) - n) <= 2 for s, n in zip(strips, true))
    short = [w for w in design.design_warnings(r["piano"]) if w["tipo"] == "strisciata_corta"]
    expected_short = sum(1 for n in true if n < 3)
    if mode == "per_punto":
        assert (short[0]["strisciate"] if short else 0) == expected_short


def _tiles_with_a_hole_at(centre, radius_m, height_m=100.0):
    """Tiles all at ``height_m`` but without data (black pixels) within ``radius_m`` of ``centre``."""
    import math

    from PIL import Image

    from ppg_flightplan.geometry import distance

    def fake(z, x, y):
        img = fake_tile(height_m)
        px = img.load()
        n = 2 ** z * 256
        reach = radius_m + tiles.metres_per_pixel(z, centre[0])      # the pixel holding the centre, at any zoom
        for r in range(256):
            lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y * 256 + r + 0.5) / n))))
            for c in range(256):
                lon = (x * 256 + c + 0.5) / n * 360.0 - 180.0
                if distance((lat, lon), centre) < reach:
                    px[c, r] = (0, 0, 0)
        return img
    return fake


def test_a_dem_that_never_meets_the_tiles_is_not_mixed_with_them(tmp_path, monkeypatch):
    """A small project DEM (a trench) under the takeoff, ellipsoidal heights 47 m above the tiles, none of
    the route on it, and the tiles without data right there: no offset can be measured, so the two references
    are not mixed - the plan is the tiles' plan, and it says the DEM was left out."""
    monkeypatch.setattr(tiles, "_tile_from_network", _tiles_with_a_hole_at(SOUTH, 30.0))
    dem = RasterDem(write_dem(tmp_path / "trincea.tif", [[147.0] * 10] * 10,
                              x0=SOUTH[1] - 0.00005, y0=SOUTH[0] + 0.00005, step=0.00001))
    assert dem.height(*SOUTH) == 147.0 and tiles.height(*SOUTH) is None
    takeoff = (SOUTH[0] + 0.00003, SOUTH[1])                 # on the DEM, in the hole of the tiles
    with_dem = design.design(AREA, "orto", 2.0, takeoff=takeoff, follow_terrain=True, dem=dem)
    piano = with_dem["piano"]
    # before: takeoff 147 from the DEM, route 100 from the tiles: every point 47 m lower (65.6 m)
    assert "quote" not in piano["blocchi"][0] or set(piano["blocchi"][0]["quote"]) != {65.6}
    assert any("non si mescolano" in m for m in with_dem["motivi"])
    assert piano.get("fonteQuote", {}).get("sorgente") != "dem"
    codes = {w["tipo"] for w in design.design_warnings(piano, dem=dem)}
    assert "scarto_non_misurato" in codes


def test_a_dem_in_a_hole_of_the_tiles_is_left_out_and_said(tmp_path, monkeypatch):
    """The DEM under a point of the route where the tiles have no data, the takeoff on the tiles: the DEM
    never meets them, so the tiles are used for everything and the outcome says the DEM was left out."""
    route, _ = pl.route_of_block(_plan()["blocchi"][0], _plan())
    spot = route.points[len(route.points) // 2]
    monkeypatch.setattr(tiles, "_tile_from_network", _tiles_with_a_hole_at(spot, 10.0))
    dem = RasterDem(write_dem(tmp_path / "trincea.tif", [[147.0] * 10] * 10,
                              x0=spot[1] - 0.00005, y0=spot[0] + 0.00005, step=0.00001))
    piano = _plan(decollo=list(SOUTH))
    outcome = design.apply_terrain(piano, dem=dem)
    assert outcome["avvisi"][0] == {"tipo": "scarto_non_misurato", "dem": str(dem.path)}
    assert "dem" not in outcome or outcome.get("scartoM") is None
    assert piano["fonteQuote"]["sorgente"] == "tegole" and piano["quotaDecollo"] == 100.0
    assert max(piano["blocchi"][0]["quote"]) == 60.0          # nothing 47 m off anywhere


def test_a_dem_around_the_takeoff_only_measures_its_offset_there(tmp_path, monkeypatch):
    """The reviewer's case: tiles at 100 m everywhere, a DEM of 147 m around the takeoff only. The planner
    took the takeoff from the DEM and the route from the tiles without correction: 65.6 m instead of 112.6 m
    over the takeoff. Now the offset is measured at the takeoff and the flight is the tiles-only one."""
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: fake_tile(100.0))
    dem = RasterDem(write_dem(tmp_path / "trincea.tif", [[147.0] * 10] * 10,
                              x0=SOUTH[1] - 0.00005, y0=SOUTH[0] + 0.00005, step=0.00001))
    with_dem = design.design(AREA, "orto", 2.0, takeoff=SOUTH, follow_terrain=True, dem=dem)
    tiles_only = design.design(AREA, "orto", 2.0, takeoff=SOUTH, follow_terrain=True)
    heights = with_dem["piano"]["blocchi"][0]["quote"]
    assert set(heights) == {with_dem["quota"]} == {112.6}
    assert heights == tiles_only["piano"]["blocchi"][0]["quote"]
    assert with_dem["piano"]["fonteQuote"] == {"sorgente": "dem", "file": str(dem.path), "epsg": 4326,
                                               "scartoM": 47.0}
    assert tiles_only["piano"]["fonteQuote"]["sorgente"] == "tegole"
    assert design.design_warnings(with_dem["piano"], dem=dem) == []


def test_the_offset_is_measured_at_the_takeoff_too(tmp_path, monkeypatch):
    """The same trench, but the tiles reach it: the offset is measured there (47 m) and the DEM rules."""
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: fake_tile(100.0))
    dem = RasterDem(write_dem(tmp_path / "trincea.tif", [[147.0] * 10] * 10,
                              x0=SOUTH[1] - 0.00005, y0=SOUTH[0] + 0.00005, step=0.00001))
    piano = _plan(decollo=list(SOUTH))
    piano["blocchi"][0]["quota"] = 112.6
    outcome = design.apply_terrain(piano, dem=dem)
    assert outcome["scartoM"] == pytest.approx(47.0, abs=0.1)
    assert set(piano["blocchi"][0]["quote"]) == {112.6}
    assert piano["quotaDecollo"] == 147.0 and piano["fonteQuote"]["sorgente"] == "dem"
    assert piano["fonteQuote"]["scartoM"] == pytest.approx(47.0, abs=0.1)
