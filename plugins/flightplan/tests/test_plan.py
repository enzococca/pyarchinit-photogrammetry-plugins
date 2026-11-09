"""The whole plan: blocks, totals, legs with their keys and takeoffs, the drone profiles, the resume."""
import pytest

from ppg_flightplan import plan as pl
from ppg_flightplan.camera import CAMERAS, camera
from ppg_flightplan.route import Route
from planfixtures import CENTRE, reference_plan


def test_compute_the_plan(plan):
    r = pl.compute(plan)
    assert r["camera"]["id"] == "mini4pro"
    assert [b["tipo"] for b in r["blocchi"]] == ["griglia", "anello"]
    a = r["blocchi"][0]
    assert a["waypoint"] == 36 and 600 <= a["scatti"] <= 700 and len(a["perimetro"]) == 4
    assert a["intervallo"] == 5.0 and a["velocitaEffettiva"] == pytest.approx(4.72, abs=0.05)
    assert a["gsdCm"] == pytest.approx(1.95, abs=0.02) and len(a["tratte"]) >= 3
    assert "piu_tratte" in a["avvisi"]
    d = r["blocchi"][1]
    assert d["poi"] == list(CENTRE) and d["gsdCm"] == pytest.approx(2.67, abs=0.03)
    assert r["totale"]["scatti"] == a["scatti"] + d["scatti"]
    assert r["totale"]["tratte"] == len(pl.plan_legs(plan))
    assert r["decolloLontanoM"] is None
    names = [leg.file_name for leg in pl.plan_legs(plan)]
    assert any("tratta1" in n and "a-nadirale" in n for n in names)
    assert all(n.startswith("campo-di-prova-") for n in names)


def test_an_inactive_block_stays_in_the_plan_but_does_not_count(plan):
    plan["blocchi"][1]["attivo"] = False
    r = pl.compute(plan)
    assert [b["attivo"] for b in r["blocchi"]] == [True, False]
    assert r["blocchi"][1]["scatti"] > 0                     # its numbers are shown all the same
    assert r["totale"]["scatti"] == r["blocchi"][0]["scatti"] and r["totale"]["blocchi"] == 1
    assert len(pl.plan_routes(plan)) == 1 and len(pl.plan_routes(plan, only_active=False)) == 2
    assert len(plan["blocchi"]) == 2
    assert not any("anello" in leg.file_name for leg in pl.plan_legs(plan))
    assert all(leg.block_index == 0 for leg in pl.plan_legs(plan))
    assert pl.active_areas(plan) == [[tuple(p) for p in plan["blocchi"][0]["area"]]]


def test_the_profiles_of_non_dji_drones():
    others = [c for c in CAMERAS if c.group == "altro"]
    assert [c.id for c in others] == ["autel_evo2pro", "autel_liteplus", "anafi", "anafi_ai", "skydio2", "m2ea", "p4m"]
    assert all(c.app and not c.enterprise for c in others)
    assert camera("anafi_ai").to_json()["gruppo"] == "altro" and camera("mini5pro").to_json()["gruppo"] == "fly"
    assert camera("mini4pro").to_json()["gruppo"] == "sdk" and camera("mavic3e").to_json()["gruppo"] == "sdk"
    # 29 mm equivalent: a narrower footprint than the Mini 4 Pro's at the same height
    assert camera("autel_evo2pro").footprint(100) < camera("mini4pro").footprint(100)
    # the GSDs DJI declares: H/18.9 cm/px for the P4 Multispectral, about H/55 for the 48 MP camera of the M2EA
    assert camera("p4m").gsd_cm(100, False) == pytest.approx(100 / 18.9, rel=0.01)
    assert camera("m2ea").gsd_cm(100, False) == pytest.approx(100 / 55.5, rel=0.02)


def test_the_drones_flown_only_from_dji_fly():
    fly_only = [c for c in CAMERAS if c.fly_only]
    assert [c.id for c in fly_only] == ["mini5pro", "air3", "air3s", "mavic3", "mavic4pro"]
    assert all(c.drone_enum == 68 and not c.enterprise and c.payload is None for c in fly_only)
    assert CAMERAS[0].id == "mini4pro" and not CAMERAS[0].fly_only
    mini4 = camera("mini4pro")
    for id_ in ("mini5pro", "air3", "air3s"):
        assert abs(camera(id_).footprint(110) / mini4.footprint(110) - 1) < 0.02
    mini5 = camera("mini5pro")
    assert mini5.gsd_cm(110, False) < mini4.gsd_cm(110, False) < 4.0
    assert mini5.gsd_cm(110, True) < 2.0 and mini5.min_interval_for(True) == 5.0
    m4 = camera("mavic4pro")
    assert m4.footprint(110) < camera("mavic3").footprint(110) and m4.gsd_cm(110, True) < 1.5
    j = mini5.to_json()
    assert j["soloFly"] is True and j["megapixelAlta"] == 50 and j["altaRisoluzione"] is True
    assert camera("mavic3").to_json()["altaRisoluzione"] is False


def test_the_legs_of_the_plan_have_a_key_and_a_takeoff(plan):
    legs = pl.plan_legs(plan)
    assert len(legs) >= 2 and all(leg.takeoff is None for leg in legs)
    keys = [leg.key for leg in legs]
    assert keys[0] == "0-1" and all(k.startswith(("0-", "1-")) for k in keys) and len(set(keys)) == len(keys)
    assert len({leg.file_name for leg in legs}) == len(legs)
    # the general takeoff holds for all, the one of a leg wins
    plan["decollo"] = [45.0001, 10.0001]
    plan["decolli"] = {keys[1]: [45.0011, 10.0011]}
    legs = pl.plan_legs(plan)
    assert legs[0].takeoff == (45.0001, 10.0001) and legs[1].takeoff == (45.0011, 10.0011)
    assert legs[1].to_json()["decollo"] == [45.0011, 10.0011] and legs[1].to_json()["chiave"] == keys[1]
    takeoffs = pl.plan_takeoffs(plan)
    assert takeoffs[0] == ("Decollo", (45.0001, 10.0001))
    assert ("Decollo " + legs[1].file_name, (45.0011, 10.0011)) in takeoffs and len(takeoffs) == 2
    # an inactive block has no legs
    plan["blocchi"][0]["attivo"] = False
    assert all(leg.block_index == 1 for leg in pl.plan_legs(plan))


def test_a_takeoff_far_away_is_said(plan):
    assert pl.takeoff_distance(plan) is None and pl.takeoff_far(plan) is None
    plan["decollo"] = [45.0, 10.0]
    assert pl.takeoff_distance(plan) < 200 and pl.takeoff_far(plan) is None
    plan["decollo"] = [46.0, 10.0]                           # 111 km to the north
    assert pl.takeoff_far(plan) > pl.FAR_TAKEOFF_M
    assert pl.compute(plan)["decolloLontanoM"] == round(pl.takeoff_far(plan))


def test_a_leg_with_its_own_takeoff_gets_heights_from_there():
    plan = reference_plan()
    block = plan["blocchi"][0]
    p, _ = pl.route_of_block(block, plan)
    block["quote"] = [110.0 + i for i in range(len(p.points))]
    plan["minutiBatteria"] = 15
    plan["quotaDecollo"] = 100.0
    legs = pl.plan_legs(plan)
    key = legs[1].key
    own = legs[1].route.points[0]
    plan["decolli"] = {key: list(own)}
    plan["suoloDecolli"] = {key: {"punto": list(own), "suolo": 120.0}}      # 20 m higher than the general one
    moved = pl.plan_legs(plan)
    before = [h for h in legs[1].route.heights]
    assert [h + 20.0 for h in moved[1].route.heights] == pytest.approx(before)
    assert moved[0].route.heights == legs[0].route.heights               # the others keep theirs


def test_the_resume_starts_from_the_last_waypoint_reached():
    points = [(45.0, 10.0), (45.001, 10.0), (45.001, 10.001), (45.0, 10.001)]
    p = Route(name="prova", points=points, height=60.0, speed=5.0, pitch=-90.0, photo_step=12.0, min_interval=2.0)
    assert p.from_resume(2).points == p.points[2:] and p.from_resume(0).points == p.points
    assert p.from_resume(-3).points == p.points
    assert len(p.from_resume(99).points) == 2 and p.from_resume(99).points == p.points[-2:]
    assert p.from_resume(1).height == p.height and p.from_resume(1).mode == p.mode


def test_slug():
    assert pl.slug("Campo di prova") == "campo-di-prova"
    assert pl.slug("***") == "piano"


def test_the_warnings_have_words():
    from ppg_flightplan import texts

    assert texts.lang() == "it"
    for code in ("velocita_ridotta", "quota_oltre_120", "terreno_oltre_120", "piu_tratte", "punti_pochi"):
        it, en = texts.WARNINGS[code]
        assert it and en and texts.warning_text(code) == it
    assert texts.warning_text("sconosciuto") == "sconosciuto"
    assert texts.plural(1, "{0} scatto", "{0} scatti") == "1 scatto"
    assert texts.plural(3, "{0} scatto", "{0} scatti") == "3 scatti"


def test_the_plugin_is_declared():
    import ppg_flightplan

    p = ppg_flightplan.FlightPlanPlugin()
    assert ppg_flightplan.__version__ == p.version == "0.1.0"
    assert p.title == "Piani di volo" and p.description
