"""The plans for the drones that do not read DJI's KMZ: QGroundControl, Mission Planner, Parrot and Litchi,
written leg by leg, with one height per point above the takeoff.

Ported from the planner's tests on the neutral place of the tests (45.0 N 10.0 E); the per-point heights on a
sloping terrain and the Mission Planner file are the plugin's."""
import csv
import io
import json
import zipfile

import pytest

from ppg_flightplan import design, tiles
from ppg_flightplan import export_formats as fv
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import CAMERAS, camera
from ppg_flightplan.route import Route
from terrainfixtures import AREA, tile_rising_north

POINTS = [(45.0, 10.0), (45.001, 10.0), (45.001, 10.001), (45.0, 10.001)]
POI = (45.0005, 10.0005)


def route(mode="intervallo", poi=None, heights=None):
    return Route(name="prova", points=list(POINTS), height=60.0, speed=5.0, pitch=-90.0, mode=mode,
                 photo_step=12.0, min_interval=2.0, poi=poi, heights=heights)


def test_the_course_in_degrees():
    assert fv.course_degrees((45.0, 10.0), (46.0, 10.0)) == pytest.approx(0.0)
    assert fv.course_degrees((45.0, 10.0), (45.0, 11.0)) == pytest.approx(90.0, abs=0.5)
    assert fv.course_degrees((45.0, 10.0), (44.0, 10.0)) == pytest.approx(180.0)


def test_the_qgroundcontrol_plan_has_takeoff_distance_trigger_waypoints_and_return():
    d = json.loads(fv.qgc_plan(route(), takeoff=(44.9993, 9.9992), name="prova"))
    assert d["fileType"] == "Plan" and d["mission"]["plannedHomePosition"][:2] == [44.9993, 9.9992]
    commands = [v["command"] for v in d["mission"]["items"]]
    assert commands[:3] == [22, 178, 205]                    # takeoff, speed, camera
    assert commands.count(16) == 4 and commands[-1] == 20     # the four waypoints and the return
    assert commands.count(206) == 2                           # trigger: on at the first point, off at the end
    trigger = [v for v in d["mission"]["items"] if v["command"] == 206]
    assert trigger[0]["params"][0] == pytest.approx(12.0, abs=0.5) and trigger[1]["params"][0] == 0
    wp = [v for v in d["mission"]["items"] if v["command"] == 16][0]
    assert wp["frame"] == 3 and wp["Altitude"] == 60 and wp["params"][4:] == [45.0, 10.0, 60.0]
    assert [v["doJumpId"] for v in d["mission"]["items"]] == list(range(1, len(commands) + 1))
    assert d["mission"]["globalPlanAltitudeMode"] == 1
    assert all(v["AltitudeMode"] == 1 for v in d["mission"]["items"] if v["command"] == 16)
    # per point: a photo at each waypoint, no distance trigger; with an aim there is the ROI
    d2 = json.loads(fv.qgc_plan(route("per_punto", poi=POI)))
    c2 = [v["command"] for v in d2["mission"]["items"]]
    assert c2.count(2000) == 4 and 206 not in c2 and 195 in c2
    assert d2["mission"]["plannedHomePosition"][:2] == [45.0, 10.0]   # without a takeoff: the first point


def test_the_parrot_mavlink_is_in_the_groundsdk_dialect():
    text = fv.parrot_mavlink(route())
    rows = text.splitlines()
    assert rows[0] == "QGC WPL 120"
    fields = [r.split("\t") for r in rows[1:]]
    assert [c[0] for c in fields] == [str(i) for i in range(len(fields))]     # indices in a row
    assert all(c[1] == "0" and c[2] == "3" and c[-1] == "1" for c in fields)  # current, frame, autocontinue
    commands = [int(c[3]) for c in fields]
    assert commands[:3] == [22, 178, 205] and commands[-2:] == [2001, 20]
    assert commands.count(16) == 4 and 50001 in commands and commands.count(2000) == 1
    gpslapse = fields[commands.index(50001)]
    assert float(gpslapse[4]) == 1.0 and float(gpslapse[5]) == pytest.approx(12.0, abs=0.5)
    wp = fields[commands.index(16)]
    assert float(wp[8]) == pytest.approx(45.0) and float(wp[9]) == pytest.approx(10.0) and float(wp[10]) == 60.0
    assert float(wp[7]) == pytest.approx(0.0, abs=0.5)      # nose along the route: north
    # with an aim: ROI and nose towards the centre; per point: a photo at each waypoint
    c2 = [int(r.split("\t")[3]) for r in fv.parrot_mavlink(route("per_punto", poi=POI)).splitlines()[1:]]
    assert 201 in c2 and c2.count(2000) == 4 and 50001 not in c2


def test_the_litchi_csv_has_the_columns_of_the_mission_hub():
    rows = list(csv.DictReader(io.StringIO(fv.litchi_csv(route()))))
    assert len(rows) == 4
    assert list(rows[0].keys())[:8] == ["latitude", "longitude", "altitude(m)", "heading(deg)", "curvesize(m)",
                                        "rotationdir", "gimbalmode", "gimbalpitchangle"]
    assert list(rows[0].keys())[-2:] == ["photo_timeinterval", "photo_distinterval"]
    assert rows[0]["altitude(m)"] == "60.0" and rows[0]["gimbalmode"] == "2" and rows[0]["gimbalpitchangle"] == "-90"
    assert rows[0]["actiontype1"] == "-1" and float(rows[0]["photo_distinterval"]) == pytest.approx(12.0, abs=0.5)
    assert rows[0]["altitudemode"] == "0" and float(rows[0]["speed(m/s)"]) > 0
    per_point = list(csv.DictReader(io.StringIO(fv.litchi_csv(route("per_punto", poi=POI)))))
    assert per_point[0]["actiontype1"] == "1" and per_point[0]["photo_distinterval"] == "-1"
    assert per_point[0]["gimbalmode"] == "1" and per_point[0]["poi_latitude"] == "45.0005000"


def test_the_legs_of_the_plan_have_a_key_and_a_takeoff(plan):
    legs = pl.plan_legs(plan)
    assert len(legs) >= 2 and all(leg.takeoff is None for leg in legs)
    keys = [leg.key for leg in legs]
    assert keys[0] == "0-1" and all(k.startswith(("0-", "1-")) for k in keys) and len(set(keys)) == len(keys)
    plan["decollo"] = [44.99, 9.99]
    plan["decolli"] = {keys[1]: [45.01, 10.01]}
    legs = pl.plan_legs(plan, specials=False)
    assert legs[0].takeoff == (44.99, 9.99) and legs[1].takeoff == (45.01, 10.01)
    # the takeoff of the leg is the home of its QGroundControl file
    files = dict(fv.files_per_leg(plan, "plan"))
    second = json.loads(files[f"{legs[1].file_name}.plan"])
    assert second["mission"]["plannedHomePosition"][:2] == [45.01, 10.01]
    plan["blocchi"][0]["attivo"] = False
    assert all(leg.block_index == 1 for leg in pl.plan_legs(plan))


def test_the_files_per_leg_and_the_zips(plan):
    for fmt, (extension, _, _) in fv.FORMATS.items():
        files = fv.files_per_leg(plan, fmt)
        assert len(files) == len(pl.plan_legs(plan))
        assert all(n.endswith("." + extension) for n, _ in files)
        with zipfile.ZipFile(io.BytesIO(fv.zip_format(plan, fmt))) as z:
            assert sorted(z.namelist()) == sorted([n for n, _ in files] + ["LEGGIMI.txt"])
            assert z.read("LEGGIMI.txt").decode("utf-8") == fv.README_TEXT[fmt]
    with pytest.raises(ValueError):
        fv.files_per_leg(plan, "docx")
    with pytest.raises(ValueError):
        fv.leg_file(route(), "docx")


def test_the_profiles_of_non_dji_drones():
    others = [c for c in CAMERAS if c.group == "altro"]
    assert [c.id for c in others] == ["autel_evo2pro", "autel_liteplus", "anafi", "anafi_ai", "skydio2",
                                      "m2ea", "p4m"]
    assert all(c.app and not c.enterprise for c in others)
    assert camera("autel_evo2pro").footprint(100) < camera("mini4pro").footprint(100)
    assert camera("p4m").gsd_cm(100, False) == pytest.approx(100 / 18.9, rel=0.01)
    assert camera("m2ea").gsd_cm(100, False) == pytest.approx(100 / 55.5, rel=0.02)


def test_the_resume_starts_from_the_last_waypoint_reached():
    p = route()
    assert p.from_resume(2).points == p.points[2:] and p.from_resume(0).points == p.points
    rows = fv.parrot_mavlink(p.from_resume(2)).splitlines()
    assert sum(1 for r in rows[1:] if r.split("\t")[3] == "16") == 2


# ------------------------------------------------------------------ Mission Planner

def _rows(text):
    return [r.split("\t") for r in text.splitlines()[1:]]


def test_mission_planner_waypoints():
    # the file is for ArduPilot: the yaw left "as it is" (NaN in QGroundControl) is written 0, which PX4 would obey
    assert "ArduPilot" in fv.README_TEXT["waypoints"] and "PX4" in fv.README_TEXT["waypoints"]
    for p, takeoff in ((route(), (44.9993, 9.9992)), (route("per_punto", poi=POI), None)):
        text = fv.mission_planner_waypoints(p, takeoff)
        assert text.splitlines()[0] == "QGC WPL 110" and text.endswith("\n")
        rows = _rows(text)
        home = rows[0]
        home_at = takeoff or p.points[0]
        assert home[:4] == ["0", "1", "0", "16"] and home[-1] == "1"
        assert float(home[8]) == pytest.approx(home_at[0], abs=1e-8)
        assert float(home[9]) == pytest.approx(home_at[1], abs=1e-8) and float(home[10]) == 0
        assert all(len(r) == 12 for r in rows)
        assert [r[0] for r in rows] == [str(i) for i in range(len(rows))]
        assert all(r[1] == "0" and r[2] == "3" and r[11] == "1" for r in rows[1:])
        commands = [int(r[3]) for r in rows[1:]]
        qgc = [v["command"] for v in json.loads(fv.qgc_plan(p, takeoff))["mission"]["items"]]
        assert commands == [201 if c == 195 else c for c in qgc]   # the ROI in ArduPilot's form
        wps = [r for r in rows[1:] if r[3] == "16"]
        assert len(wps) == len(p.points)
        for r, (lat, lon), i in zip(wps, p.points, range(len(p.points))):
            assert float(r[8]) == pytest.approx(lat, abs=1e-8) and float(r[9]) == pytest.approx(lon, abs=1e-8)
            assert float(r[10]) == pytest.approx(p.height_at(i))
            assert len(r[8].split(".")[1]) == 8
        if p.mode == "per_punto":
            for k, c in enumerate(commands):
                if c == 16:
                    assert commands[k + 1] == 2000
            roi = rows[1 + commands.index(201)]
            assert float(roi[8]) == pytest.approx(POI[0], abs=1e-8)
        else:
            assert commands.count(206) == 2


# ------------------------------------------------------------------ one height per point

def test_every_format_writes_one_height_per_point():
    p = route("per_punto", heights=[60.0, 72.5, 84.0, 66.0])
    qgc = [v for v in json.loads(fv.qgc_plan(p))["mission"]["items"] if v["command"] == 16]
    assert [v["Altitude"] for v in qgc] == [60.0, 72.5, 84.0, 66.0]
    assert [v["params"][6] for v in qgc] == [60.0, 72.5, 84.0, 66.0]
    parrot = [float(r[10]) for r in _rows(fv.parrot_mavlink(p)) if r[3] == "16"]
    assert parrot == [60.0, 72.5, 84.0, 66.0]
    litchi = [float(r["altitude(m)"]) for r in csv.DictReader(io.StringIO(fv.litchi_csv(p)))]
    assert litchi == [60.0, 72.5, 84.0, 66.0]
    mp = [float(r[10]) for r in _rows(fv.mission_planner_waypoints(p))[1:] if r[3] == "16"]
    assert mp == [60.0, 72.5, 84.0, 66.0]


def _terrain_plan(monkeypatch, mode="per_punto"):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = {"nome": "pendio", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "decollo": [44.9970, 10.0],
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(q) for q in AREA], "quota": 60,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": mode, "seguiTerreno": True}]}
    design.apply_terrain(piano)
    return piano


@pytest.mark.parametrize("mode", ["per_punto", "intervallo"])
def test_on_a_sloping_terrain_the_files_follow_it_point_by_point(monkeypatch, mode):
    piano = _terrain_plan(monkeypatch, mode)
    legs = pl.plan_legs(piano, specials=False)
    heights = [h for leg in legs for h in leg.route.heights]
    assert max(heights) - min(heights) > 5, "the test terrain rises about ten metres: the heights must too"
    for fmt in fv.FORMATS:
        for (name, data), leg in zip(fv.files_per_leg(piano, fmt), legs):
            text = data.decode("utf-8")
            if fmt == "plan":
                got = [v["Altitude"] for v in json.loads(text)["mission"]["items"] if v["command"] == 16]
                frames = {v["frame"] for v in json.loads(text)["mission"]["items"] if v["command"] == 16}
                assert frames == {3}
            elif fmt == "litchi":
                rows = list(csv.DictReader(io.StringIO(text)))
                got = [float(r["altitude(m)"]) for r in rows]
                assert {r["altitudemode"] for r in rows} == {"0"}
            else:
                rows = _rows(text)[1:] if fmt == "waypoints" else _rows(text)      # row 0 of Mission Planner: home
                got = [float(r[10]) for r in rows if r[3] == "16"]
            assert got == pytest.approx([round(h, 1) for h in leg.route.heights], abs=0.051), (fmt, name)
            if fmt in ("plan", "waypoints"):
                # the takeoff climbs straight to the height of the first waypoint, not to the nominal one
                if fmt == "plan":
                    climb = [v["Altitude"] for v in json.loads(text)["mission"]["items"] if v["command"] == 22]
                else:
                    climb = [float(r[10]) for r in _rows(text)[1:] if r[3] == "22"]
                assert climb == [round(leg.route.heights[0], 1)], (fmt, name)
    # a leg taking off from a point of its own gets heights from there: the files follow it
    key = legs[0].key
    takeoff = [44.9975, 10.0005]
    piano["decolli"] = {key: takeoff}
    design.apply_terrain(piano)
    moved = pl.plan_legs(piano, specials=False)[0]
    shift = piano["quotaDecollo"] - piano["suoloDecolli"][key]["suolo"]
    data = json.loads(dict(fv.files_per_leg(piano, "plan"))[f"{moved.file_name}.plan"])
    got = [v["Altitude"] for v in data["mission"]["items"] if v["command"] == 16]
    expected = [round(min(500, max(10, h + shift)), 1) for h in legs[0].route.heights]
    assert got == pytest.approx(expected, abs=0.051)


def test_the_terrain_note(monkeypatch, plan):
    assert fv.terrain_note(plan) is None
    note = fv.terrain_note(_terrain_plan(monkeypatch))
    assert note and "punto per punto" in note and "prova" in note
    switched_off = _terrain_plan(monkeypatch)
    switched_off["blocchi"][0]["attivo"] = False
    assert fv.terrain_note(switched_off) is None


# ------------------------------------------------------------------ heights computed for another block

@pytest.mark.parametrize("change", [{"lato": 60}, {"orientamento": 180}, {"quota": 80}])
def test_heights_of_a_block_changed_after_the_terrain_are_never_exported(monkeypatch, change, tmp_path):
    """The block changed after the heights were computed: the side overlap (other points), the heading (the same
    number of points in the reverse order) or the height above the ground. No file may fly the old heights, nor
    one height, while the note says the terrain is followed: the export refuses until the terrain is redone."""
    from ppg_flightplan import export_dji, gis

    piano = _terrain_plan(monkeypatch)
    piano["blocchi"][0].update(change)
    for fmt in fv.FORMATS:
        with pytest.raises(design.StaleHeights, match="ricalcola il terreno"):
            fv.files_per_leg(piano, fmt)
        with pytest.raises(design.StaleHeights):
            fv.zip_format(piano, fmt)
    with pytest.raises(design.StaleHeights, match="«A»"):
        export_dji.kmz_per_leg(piano)
    with pytest.raises(design.StaleHeights):
        export_dji.zip_kmz(piano)
    with pytest.raises(design.StaleHeights):
        gis.plan_kml(piano)
    with pytest.raises(design.StaleHeights):
        gis.write_gpkg(piano, tmp_path / "piano.gpkg", "EPSG:32632")
    assert not (tmp_path / "piano.gpkg").exists()
    assert fv.terrain_note(piano) is None
    assert design.blocks_with_old_heights(piano) == [0]
    # redone, the files follow the terrain again
    design.refresh_terrain(piano)
    assert design.blocks_with_old_heights(piano) == []
    legs = pl.plan_legs(piano, specials=False)
    got = [v["Altitude"] for v in json.loads(fv.files_per_leg(piano, "plan")[0][1])["mission"]["items"]
           if v["command"] == 16]
    assert got == pytest.approx([round(h, 1) for h in legs[0].route.heights], abs=0.051)
    assert len(set(got)) > 1 and fv.terrain_note(piano)


def test_a_switched_off_block_with_old_heights_does_not_stop_the_export(monkeypatch):
    piano = _terrain_plan(monkeypatch)
    piano["blocchi"].append(dict(piano["blocchi"][0], nome="B", lato=60, attivo=False))
    assert fv.files_per_leg(piano, "plan")


def test_a_plan_of_the_planner_without_the_signature_is_checked_by_the_count(monkeypatch):
    piano = _terrain_plan(monkeypatch)
    piano["blocchi"][0].pop("firmaQuote")
    assert fv.files_per_leg(piano, "litchi")          # heights as many as the points: taken as they are
    piano["blocchi"][0]["lato"] = 60
    with pytest.raises(design.StaleHeights):
        fv.files_per_leg(piano, "litchi")


def test_a_plan_without_the_signature_is_signed_when_it_comes_in(monkeypatch):
    """A plan of the planner (or saved before the signature) is taken as it is when it comes in - its heights as
    many as its points - and signed then: a later change of the block that keeps the number of points (the strips
    flown the other way round) is caught like in a plan made here."""
    piano = _terrain_plan(monkeypatch)
    piano["blocchi"][0].pop("firmaQuote")
    assert design.adopt_heights(piano) == [0]
    assert piano["blocchi"][0]["firmaQuote"] == design.heights_signature(piano["blocchi"][0], piano)
    assert design.adopt_heights(piano) == []          # already signed: nothing to do
    piano["blocchi"][0]["orientamento"] = 180
    with pytest.raises(design.StaleHeights):
        fv.files_per_leg(piano, "litchi")


def test_a_plan_whose_heights_do_not_match_its_points_is_not_signed(monkeypatch):
    piano = _terrain_plan(monkeypatch)
    piano["blocchi"][0].pop("firmaQuote")
    piano["blocchi"][0]["quote"] = piano["blocchi"][0]["quote"][:-1]
    assert design.adopt_heights(piano) == []
    assert "firmaQuote" not in piano["blocchi"][0]
    with pytest.raises(design.StaleHeights):
        fv.files_per_leg(piano, "litchi")
