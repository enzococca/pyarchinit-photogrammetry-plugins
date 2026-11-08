"""The DJI files: the KMZ as DJI Fly writes it for the Mini 4 Pro, the archive of the plan, and the readers.

Ported from the planner's tests on the neutral place of the tests (45.0 N 10.0 E); the data files are the
planner's, moved there with both coordinates (local metres kept)."""
import html
import io
import json
import math
import re
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from ppg_flightplan import design, gis, tiles
from ppg_flightplan import export_dji as dji
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import CAMERAS, camera
from ppg_flightplan.geometry import distance
from ppg_flightplan.route import Route
from planfixtures import CENTRE, block_a, reference_plan
from terrainfixtures import AREA, tile_rising_north

DATA = Path(__file__).parent / "data"
MINI4 = camera("mini4pro")


def local(el):
    return el.tag.rsplit("}", 1)[-1]


def values(root, name):
    return [e.text.strip() for e in root.iter() if local(e) == name and e.text]


def waylines(data: bytes) -> ET.Element:
    return ET.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("wpmz/waylines.wpml"))


def template(data: bytes) -> ET.Element:
    return ET.fromstring(zipfile.ZipFile(io.BytesIO(data)).read("wpmz/template.kml"))


def test_the_kmz_has_dji_s_two_files_and_the_mini_code():
    p, _ = pl.route_of_block(block_a(), {"camera": "mini4pro"})
    data = dji.kmz(p, MINI4, now_ms=1_700_000_000_000)
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        assert sorted(z.namelist()) == ["wpmz/template.kml", "wpmz/waylines.wpml"]
        model = ET.fromstring(z.read("wpmz/template.kml"))
        lines = ET.fromstring(z.read("wpmz/waylines.wpml"))
    assert values(model, "droneEnumValue") == ["68"]
    assert values(model, "createTime") == ["1700000000000"]
    # the template has the same points: DJI Fly rebuilds the mission from there when it saves it again
    assert values(model, "templateType") == ["waypoint"] and values(model, "globalHeight") == ["110"]
    assert values(model, "gimbalPitchMode") == ["usePointSetting"]
    assert len(values(model, "index")) == 36 and values(model, "index") == values(lines, "index")
    assert values(model, "coordinates") == values(lines, "coordinates")
    assert values(model, "height") == [] and values(model, "globalWaypointTurnMode") == []
    assert len(values(lines, "index")) == 36
    assert values(lines, "executeHeight")[0] == "110"
    assert values(lines, "waypointHeadingMode")[0] == "followWayline"
    assert values(lines, "waypointTurnMode")[0] == "toPointAndStopWithDiscontinuityCurvature"
    assert values(lines, "actionActuatorFunc") == ["gimbalRotate"]
    assert values(lines, "gimbalPitchRotateAngle") == ["-90"]
    assert values(lines, "takeOffSecurityHeight") == []
    # it reads again and gives the same route
    read = dji.read_file("prova.kmz", data, MINI4).routes[0]
    assert len(read.points) == 36 and read.height == 110 and read.pitch == -90
    assert read.mode == "intervallo" and read.poi is None
    assert all(distance(a, b) < 0.01 for a, b in zip(read.points, p.points))


def test_the_author_is_the_program():
    p, _ = pl.route_of_block(block_a(), {"camera": "mini4pro"})
    assert values(template(dji.kmz(p, MINI4)), "author") == [dji.AUTHOR] == ["pyarchinit-photogrammetry"]


@pytest.mark.parametrize("cam", [c for c in CAMERAS if c.group in ("sdk", "fly")], ids=lambda c: c.id)
def test_every_dji_camera_declares_heights_above_the_takeoff(cam):
    p, _ = pl.route_of_block(block_a(), {"camera": cam.id})
    data = dji.kmz(p, cam)
    model, lines = template(data), waylines(data)
    assert values(model, "heightMode") == ["relativeToStartPoint"]
    assert values(model, "coordinateMode") == ["WGS84"]
    assert values(lines, "executeHeightMode") == ["relativeToStartPoint"]
    assert len(values(lines, "executeHeight")) == len(p.points)


def test_drones_that_fly_only_from_dji_fly():
    """Mini 5 Pro, Air 3/3S, Mavic 3 and 4 Pro: the Mobile SDK does not know them, but DJI Fly flies the KMZ of
    the Mini 4 Pro (code 68). Same file, only the camera profile changes."""
    only_fly = [c for c in CAMERAS if c.fly_only]
    assert [c.id for c in only_fly] == ["mini5pro", "air3", "air3s", "mavic3", "mavic4pro"]
    assert all(c.drone_enum == 68 and not c.enterprise and c.payload is None for c in only_fly)
    mini5 = camera("mini5pro")
    p, _ = pl.route_of_block(block_a(), {"camera": "mini5pro"})
    lines = waylines(dji.kmz(p, mini5))
    assert values(lines, "droneEnumValue") == ["68"] and values(lines, "payloadEnumValue") == []
    assert values(lines, "takeOffSecurityHeight") == [] and values(lines, "actionGroupMode")[0] == "parallel"


def test_per_point_and_enterprise():
    p, _ = pl.route_of_block(block_a("per_punto", False), {"camera": "mini4pro"})
    leg = p.legs(15)[0]
    lines = waylines(dji.kmz(leg.route, MINI4))
    functions = values(lines, "actionActuatorFunc")
    assert functions.count("takePhoto") == len(leg.route.points)
    assert functions.count("gimbalRotate") == 1
    assert values(lines, "waypointTurnMode")[0] == "toPointAndPassWithContinuityCurvature"
    m3e = camera("mavic3e")
    l2 = waylines(dji.kmz(leg.route, m3e))
    assert values(l2, "takeOffSecurityHeight") == ["20"]
    assert values(l2, "payloadEnumValue") == ["66"] and values(l2, "droneEnumValue") == ["77"]
    assert values(l2, "actionGroupMode")[0] == "sequence"
    assert values(l2, "fileSuffix")
    # in the template the Enterprise also have the height and the global parameters of the WPML specification
    t2 = template(dji.kmz(leg.route, m3e))
    assert len(values(t2, "height")) == len(leg.route.points) == len(values(t2, "executeHeight"))
    assert values(t2, "globalWaypointTurnMode") == [values(l2, "waypointTurnMode")[0]]
    assert values(t2, "globalUseStraightLine") == ["0"] and values(t2, "templateType") == ["waypoint"]


def test_the_ring_looks_at_the_centre():
    piano = reference_plan()
    p, _ = pl.route_of_block(piano["blocchi"][1], piano)
    assert p.poi == CENTRE
    lines = waylines(dji.kmz(p, MINI4))
    assert values(lines, "waypointHeadingMode")[0] == "towardPOI"
    assert values(lines, "waypointPoiPoint")[0].startswith("45.0000")
    assert values(lines, "waypointTurnMode")[0] == "toPointAndPassWithContinuityCurvature"
    assert values(lines, "waypointGimbalPitchAngle")[0] == "-35"


def test_the_reference_kml_comes_in_with_its_five_blocks():
    read = dji.read_file("plan_mini4pro.kml", (DATA / "plan_mini4pro.kml").read_bytes(), MINI4, high=True)
    assert len(read.routes) == 5 and read.areas == []
    a = read.routes[0]
    assert a.name == "A  nadirale" and a.height == 110 and a.pitch == -90 and a.declared_shots == 648
    assert len(a.points) == 36 and a.poi is None
    assert a.length / 1000 == pytest.approx(15.6, abs=0.4)
    assert abs(a.shots() - 648) <= 70
    e = read.routes[4]
    assert e.height == 55 and e.pitch == -25 and e.poi is not None and 12 <= len(e.points) <= 100


def test_the_geojson_carries_speed_and_shots_and_the_polygon_becomes_an_area():
    read = dji.read_file("piano.geojson", (DATA / "plan_mini4pro.geojson").read_bytes(), MINI4)
    assert len(read.routes) == 5
    b = read.routes[1]
    assert b.name == "B  obliquo N-S" and b.speed == 4.5 and b.pitch == -45 and b.declared_shots == 325
    assert read.routes[3].poi is not None
    geo = json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"name": "campo"},
           "geometry": {"type": "Polygon", "coordinates": [[[9.99, 44.99], [10.0, 44.99], [10.0, 45.0],
                                                            [9.99, 45.0], [9.99, 44.99]]]}}]})
    area = dji.read_file("campo.geojson", geo.encode(), MINI4)
    assert area.routes == [] and len(area.areas) == 1 and len(area.areas[0]) == 4
    kml_area = ('<kml xmlns="http://www.opengis.net/kml/2.2"><Document><Placemark><name>campo</name><Polygon>'
                '<outerBoundaryIs><LinearRing><coordinates>9.99,44.99,0 10.0,44.99,0 10.0,45.0,0 9.99,45.0,0 '
                '9.99,44.99,0</coordinates></LinearRing></outerBoundaryIs></Polygon></Placemark></Document></kml>')
    assert len(dji.read_file("area.kml", kml_area.encode(), MINI4).areas[0]) == 4


def test_kml_and_geojson_exported_are_read_again():
    piano = reference_plan()
    routes = pl.plan_routes(piano)
    back = dji.read_file("campo.kml", gis.kml(routes, "Campo").encode(), MINI4, high=True).routes
    assert len(back) == 2
    assert len(back[0].points) == 36 and back[0].height == 110 and back[0].pitch == -90
    assert back[0].speed == pytest.approx(routes[0].effective_speed, abs=0.01)
    assert back[0].declared_shots == routes[0].shots()
    assert back[1].poi is not None and back[1].height == 75 and back[1].pitch == -35
    geo = dji.read_file("campo.geojson", gis.geojson(routes, MINI4, True, [piano["blocchi"][0]["area"]]).encode(),
                        MINI4).routes
    assert len(geo) == 2 and geo[1].poi is not None
    assert geo[0].speed == pytest.approx(routes[0].effective_speed, abs=0.01)


def test_a_hand_route_keeps_its_aim_through_kml_and_geojson():
    contour = [(44.9993, 9.9983), (44.9995, 10.0012), (45.0007, 10.0016), (45.0011, 9.9990), (45.0001, 9.9978)]
    ring_ = {"tipo": "percorso", "nome": "Contorno", "punti": contour, "chiuso": True, "versoCentro": True,
             "quota": 60, "inclinazione": -40, "avanti": 80, "velocita": 4}
    p, _ = pl.route_of_block(ring_, {"camera": "mini4pro"})
    road = {"tipo": "percorso", "nome": "Strada", "punti": contour[:3], "quota": 50, "inclinazione": -90,
            "poi": [45.0023, 10.0012]}
    q, _ = pl.route_of_block(road, {"camera": "mini4pro"})
    read = dji.read_file("mano.kml", gis.kml([p, q], "mano").encode(), MINI4).routes
    assert distance(read[0].poi, p.poi) < 0.01 and read[1].poi == (45.0023, 10.0012)
    geo = dji.read_file("mano.geojson", gis.geojson([p, q], MINI4, False).encode(), MINI4).routes
    assert geo[1].poi == (45.0023, 10.0012) and distance(geo[0].poi, p.poi) < 0.01
    assert values(waylines(dji.kmz(q, MINI4)), "waypointHeadingMode")[0] == "towardPOI"


def test_terrain_heights_per_point():
    """The per-point heights end in the KMZ: that is how the GSD stays constant uphill."""
    p = Route(name="salita", points=[(45.0, 10.0), (45.001, 10.0), (45.002, 10.0)],
              height=60.0, heights=[60.0, 72.0, 84.0], mode="per_punto")
    lines = zipfile.ZipFile(io.BytesIO(dji.kmz(p, MINI4))).read("wpmz/waylines.wpml").decode()
    assert re.findall(r"<wpml:executeHeight>([0-9.]+)</wpml:executeHeight>", lines) == ["60", "72", "84"]
    # and they come back from the KMZ
    back = dji.read_file("salita.kmz", dji.kmz(p, MINI4), MINI4).routes[0]
    assert back.heights == [60.0, 72.0, 84.0]


def test_zip_kmz_of_the_plan():
    piano = reference_plan()
    r = pl.compute(piano)
    with zipfile.ZipFile(io.BytesIO(dji.zip_kmz(piano))) as z:
        names = z.namelist()
        plan_kml = z.read("campo-di-prova.kml").decode()
    assert "campo-di-prova.kml" in names
    assert len([n for n in names if n.endswith(".kmz")]) == r["totale"]["tratte"]
    assert any("tratta1" in n and "a-nadirale" in n for n in names)
    ET.fromstring(plan_kml)
    piano["blocchi"][1]["attivo"] = False
    with zipfile.ZipFile(io.BytesIO(dji.zip_kmz(piano))) as z:
        assert not any("anello" in n for n in z.namelist())


def test_kmz_per_leg_follows_the_legs_of_the_plan():
    piano = reference_plan()
    legs = pl.plan_legs(piano)
    files = dji.kmz_per_leg(piano)
    assert [n for n, _ in files] == [f"{leg.file_name}.kmz" for leg in legs]
    for (_, data), leg in zip(files, legs):
        assert len(values(waylines(data), "index")) == len(leg.route.points)


# ------------------------------------------------------------------ KML heights: absolute and declared

def test_the_kml_writes_absolute_heights_and_says_from_what():
    piano = reference_plan()
    routes = pl.plan_routes(piano)
    text = gis.kml(routes, "Campo", ground_takeoff=250.0, height_source={"sorgente": "tegole"})
    root = ET.fromstring(text)
    assert set(values(root, "altitudeMode")) == {"absolute"}
    first = values(root, "coordinates")
    lines = [c for c in first if len(c.split()) > 2]
    assert {float(t.split(",")[2]) for t in lines[0].split()} == {360.0}          # 250 + 110
    data = {d.get("name"): d for d in root.iter() if local(d) == "Data"}
    assert values(data[dji.DATA_TAKEOFF_HEIGHT], "value") == ["250"]
    assert "250" in values(data[dji.DATA_HEIGHT_REFERENCE], "value")[0]
    assert "mare" in values(root, "description")[0]
    # read again: back to the heights above the takeoff
    back = dji.read_file("campo.kml", text.encode(), MINI4).routes
    assert back[0].height == 110 and back[1].height == 75


def test_without_a_takeoff_height_the_kml_writes_zero_and_says_so():
    routes = pl.plan_routes(reference_plan())
    root = ET.fromstring(gis.kml(routes, "Campo"))
    data = {d.get("name"): d for d in root.iter() if local(d) == "Data"}
    assert values(data[dji.DATA_TAKEOFF_HEIGHT], "value") == ["0"]
    assert "sconosciuta" in values(root, "description")[0]
    assert set(values(root, "altitudeMode")) == {"absolute"}


def test_the_plan_kml_carries_the_terrain_and_reads_back_per_point(monkeypatch):
    """The grid travels inside the KML without breaking the file for the others, and the terrain-following
    routes come back with their heights above the takeoff (the planner's check, moved here with the writer)."""
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = {"nome": "rilievo", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "decollo": [44.9970, 10.0],
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in AREA], "quota": 60,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "per_punto", "seguiTerreno": True}]}
    design.apply_terrain(piano)
    g = piano["terreno"]
    text = gis.plan_kml(piano)
    m = re.search(r'<Data name="ppg:terreno"><value>(.*?)</value>', text, re.S)
    assert m and json.loads(html.unescape(m.group(1)))["righe"] == g["righe"]
    ET.fromstring(text)
    read = dji.read_kml("rilievo", text, MINI4)
    original = pl.plan_routes(piano)[0]
    assert read.routes[0].heights == pytest.approx(original.heights, abs=0.01)
    # absolute in the file: the takeoff ground added to every height
    coords = [c for c in values(ET.fromstring(text), "coordinates") if len(c.split()) > 2][0]
    z = [float(t.split(",")[2]) for t in coords.split()]
    assert z == pytest.approx([piano["quotaDecollo"] + h for h in original.heights], abs=0.01)
    design.remove_terrain(piano)
    assert dji.DATA_TERRAIN not in gis.plan_kml(piano)


def test_a_kml_written_by_the_planner_still_reads_its_stops():
    stops = [{"i": 1, "azioni": ["foto"], "bassa": True, "speciale": "T1", "tipo": "foto"}]
    text = ('<?xml version="1.0" encoding="UTF-8"?>\n<kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
            '<name>vecchio</name><ExtendedData><Data name="photo3d:terreno"><value>{}</value></Data></ExtendedData>'
            '<Placemark><name>A</name><description>quota 60 m, camera -90 gradi, 3 scatti, 1 min, velocita 5 m/s, '
            'uno scatto per punto, 1 punti speciali</description>'
            f'<ExtendedData><Data name="photo3d:soste"><value>{html.escape(json.dumps(stops))}</value></Data>'
            '</ExtendedData><LineString><altitudeMode>relativeToGround</altitudeMode><coordinates>'
            '10.0,45.0,60 10.0,45.0001,20 10.0,45.0002,60 </coordinates></LineString></Placemark></Document></kml>')
    route = dji.read_kml("vecchio", text, MINI4).routes[0]
    assert route.stops is not None and route.stops[1].actions == ("foto",) and route.stops[1].low
    assert route.heights == [60.0, 20.0, 60.0] and route.height == 60


def test_the_geojson_says_its_heights_are_above_the_takeoff():
    routes = pl.plan_routes(reference_plan())
    data = json.loads(gis.geojson(routes, MINI4, True))
    lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
    assert all(f["properties"]["riferimento_quote"] == "decollo" for f in lines)
    assert lines[0]["geometry"]["coordinates"][0][2] == 110


def test_special_points_survive_the_kmz():
    """A special photo with its descent: the actions, the low waypoint and the head come back from the KMZ."""
    from ppg_flightplan.specials import with_specials

    p = Route(name="giro", points=[(45.0, 10.0), (45.0005, 10.0), (45.001, 10.0)], height=60.0, mode="per_punto")
    new, outcomes = with_specials(p, [{"tipo": "foto", "punto": [45.0005, 10.0],
                                       "discesa": {"modo": "decollo", "metri": 20}}])
    assert any(s and s.low for s in new.stops)
    back = dji.read_file("giro.kmz", dji.kmz(new, MINI4), MINI4).routes[0]
    assert back.stops is not None
    assert [bool(s and s.low) for s in back.stops] == [bool(s and s.low) for s in new.stops]
    assert math.isclose(back.height, 60.0)
