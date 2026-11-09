"""The plan for the GIS: the areas of the grids for the DJI apps that make their own strips, and the GeoPackage in
the chunk's system with heights declared.

The area tests are ported from the planner's (on the neutral place of the tests, 45.0 N 10.0 E); the GeoPackage is
the plugin's."""
import copy
import sqlite3
import xml.etree.ElementTree as ET

import pytest
from pyproj import Transformer

from pyarchinit_photogrammetry.io import gpkg

from ppg_flightplan import design, gis, tiles
from ppg_flightplan import plan as pl
from ppg_flightplan.export_dji import NS_KML
from ppg_flightplan.route import shot_points
from terrainfixtures import AREA, tile_rising_north

NS = {"k": NS_KML}


def _ring(text: str) -> list[tuple[float, float]]:
    root = ET.fromstring(text)
    coords = root.find(".//k:Polygon/k:outerBoundaryIs/k:LinearRing/k:coordinates", NS).text.split()
    return [(float(c.split(",")[0]), float(c.split(",")[1])) for c in coords]


def test_the_area_is_a_closed_anticlockwise_polygon_with_the_values_to_set_again(plan):
    piano = dict(plan, camera="m2ea", alta=False)
    files = gis.areas_kml(piano)
    assert [n for n, _ in files] == ["a-nadirale-area.kml"], "the ring has no area: it stays out"
    text = files[0][1]
    ring_ = _ring(text)
    assert len(ring_) == 5 and ring_[0] == ring_[-1]
    double = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(ring_, ring_[1:]))
    assert double > 0, "the outer boundary goes anticlockwise"
    sides = {(round(lat, 7), round(lon, 7)) for lat, lon in piano["blocchi"][0]["area"]}
    assert {(round(lat, 7), round(lon, 7)) for lon, lat in ring_} == sides
    description = ET.fromstring(text).find(".//k:Placemark/k:description", NS).text
    assert "Quota 110 m" in description and "frontale 80%" in description and "laterale 70%" in description
    assert "Mavic 2 Enterprise Advanced" in description and "SPERIMENTALE" in description
    # a clockwise area comes out anticlockwise all the same
    clockwise = copy.deepcopy(piano)
    clockwise["blocchi"][0]["area"] = list(reversed(clockwise["blocchi"][0]["area"]))
    r2 = _ring(gis.area_kml(clockwise["blocchi"][0], clockwise))
    assert sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(r2, r2[1:])) > 0
    # two grids with the same name: the second file does not cover the first
    two = copy.deepcopy(plan)
    two["blocchi"].append(copy.deepcopy(two["blocchi"][0]))
    assert [n for n, _ in gis.areas_kml(two)] == ["a-nadirale-area.kml", "a-nadirale-2-area.kml"]


def test_without_active_grids_there_is_no_area(plan):
    piano = copy.deepcopy(plan)
    piano["blocchi"][0]["attivo"] = False
    with pytest.raises(ValueError, match="griglia"):
        gis.areas_kml(piano)
    with pytest.raises(ValueError):
        gis.area_kml({"tipo": "griglia", "nome": "x", "area": [[45.0, 10.0]]}, piano)


def test_the_kml_names_the_dem_file_never_its_folder():
    piano = {"nome": "p", "camera": "mini4pro", "blocchi": [], "quotaDecollo": 120.0,
             "fonteQuote": {"sorgente": "dem", "file": "/Users/someone/private/rilievo/dem.tif", "epsg": 32632}}
    text = gis.plan_kml(piano)
    assert "dem.tif" in text and "/Users/someone" not in text and "private" not in text


# ------------------------------------------------------------------ GeoPackage

CRS = "EPSG:32632"


def _terrain_plan(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", tile_rising_north)
    piano = {"nome": "pendio", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "decollo": [44.9970, 10.0],
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(q) for q in AREA], "quota": 60,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "per_punto", "seguiTerreno": True}],
             "target": {"crs": "EPSG:4326", "punti": [
                 {"nome": "T1", "codice": 1, "lat": 45.0001, "lon": 10.0001, "quota": 250.0},
                 {"nome": "M2", "codice": 2, "lat": 45.0002, "lon": 9.9998, "quota": 251.0, "stimato": True}]}}
    design.apply_terrain(piano)
    return piano


def _tables(path):
    con = sqlite3.connect(path)
    try:
        return {r[0]: (r[1], r[2]) for r in con.execute(
            "SELECT table_name, geometry_type_name, z FROM gpkg_geometry_columns")}
    finally:
        con.close()


def test_gpkg(tmp_path, monkeypatch):
    piano = _terrain_plan(monkeypatch)
    ground = piano["quotaDecollo"]
    path = tmp_path / "piano.gpkg"
    gis.write_gpkg(piano, path, CRS)
    assert gpkg.is_geopackage(path)
    tables = _tables(path)
    assert set(tables) == {"ppg_area", "ppg_strisciate", "ppg_scatti", "ppg_decolli", "ppg_target"}
    assert tables["ppg_strisciate"] == ("LINESTRING", 1) and tables["ppg_scatti"] == ("POINT", 1)
    assert tables["ppg_area"] == ("POLYGON", 0)
    con = sqlite3.connect(path)
    assert {r[0] for r in con.execute("SELECT DISTINCT srs_id FROM gpkg_geometry_columns")} == {32632}
    con.close()

    legs = pl.plan_legs(piano)
    shots = [s for leg in legs for s in shot_points(leg.route)]
    rows = gpkg.read_table(path, "ppg_scatti")
    assert len(rows) == len(shots)
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    for (geom, values), ((lat, lon), h) in zip(rows, shots):
        x, y = to_utm.transform(lon, lat)
        assert geom.x == pytest.approx(x, abs=0.01) and geom.y == pytest.approx(y, abs=0.01)
        assert values["quota_decollo_m"] == pytest.approx(h, abs=0.01)
        assert geom.z == pytest.approx(ground + values["quota_decollo_m"], abs=0.01)
        assert values["agl_m"] is not None and values["gsd_cm"] > 0
        assert "decollo" in values["riferimento_quote"]
    # the ground rises about ten metres to the north: the shots' heights follow it, the distance from the ground stays
    agl = [v["agl_m"] for _, v in rows]
    assert max(agl) - min(agl) < 2.0
    assert max(v["quota_decollo_m"] for _, v in rows) - min(v["quota_decollo_m"] for _, v in rows) > 5

    strips = gpkg.read_table(path, "ppg_strisciate")
    assert len(strips) == len(legs)
    geom, values = strips[0]
    import shapely
    zs = shapely.get_coordinates(geom, include_z=True)[:, 2]
    assert list(zs) == pytest.approx([ground + h for h in legs[0].route.heights], abs=0.01)
    assert values["file"] == legs[0].file_name and values["scatti"] == legs[0].route.shots()
    assert values["quota_m"] == 60 and values["camera_gradi"] == -90

    targets = gpkg.read_table(path, "ppg_target")
    assert [v["nome"] for _, v in targets] == ["T1", "M2"]
    assert [v["stimato"] for _, v in targets] == [0, 1]
    x, y = to_utm.transform(10.0001, 45.0001)
    assert targets[0][0].x == pytest.approx(x, abs=0.01) and targets[0][0].y == pytest.approx(y, abs=0.01)
    takeoffs = gpkg.read_table(path, "ppg_decolli")
    assert [v["nome"] for _, v in takeoffs] == ["Decollo"]
    area = gpkg.read_table(path, "ppg_area")
    assert len(area) == 1 and area[0][0].area > 100_000          # about 470 x 670 m
    # written again over itself: still five tables, rows replaced
    gis.write_gpkg(piano, path, CRS)
    assert len(gpkg.read_table(path, "ppg_scatti")) == len(shots)
    before = [g.z for g, v in gpkg.read_table(path, "ppg_scatti") if v["tratta"] == 1]

    # a leg taking off from a point of its own: heights above THAT takeoff, Z from its ground - the same place
    key = legs[0].key
    piano["decolli"] = {key: [44.9975, 10.0005]}
    design.apply_terrain(piano)
    own = piano["suoloDecolli"][key]["suolo"]
    assert abs(own - piano["quotaDecollo"]) > 0.5
    gis.write_gpkg(piano, path, CRS)
    first = [(g, v) for g, v in gpkg.read_table(path, "ppg_scatti") if v["blocco"] == 0 and v["tratta"] == 1]
    for g, v in first:
        assert g.z == pytest.approx(own + v["quota_decollo_m"], abs=0.01)
    assert [g.z for g, _ in first] == pytest.approx(before, abs=0.06)
    assert str(round(own, 1)).rstrip("0").rstrip(".") in first[0][1]["riferimento_quote"]


def test_gpkg_heights_without_a_known_takeoff(tmp_path, plan):
    path = tmp_path / "piatto.gpkg"
    gis.write_gpkg(plan, path, CRS)
    rows = gpkg.read_table(path, "ppg_scatti")
    assert {round(g.z, 3) for g, v in rows if v["blocco"] == 0} == {110.0}
    assert all(v["agl_m"] is None for _, v in rows)
    assert "sconosciuta" in rows[0][1]["riferimento_quote"]
    # with the DEM's height at the takeoff given by the caller: absolute in the DEM's system, the file named
    gis.write_gpkg(plan, path, CRS, ground_takeoff=300.0, dem_name="/Volumes/dati/dem.tif")
    rows = gpkg.read_table(path, "ppg_scatti")
    assert {round(g.z, 3) for g, v in rows if v["blocco"] == 0} == {410.0}
    assert "dem.tif" in rows[0][1]["riferimento_quote"] and "/Volumes" not in rows[0][1]["riferimento_quote"]
    # a DEM value without its name: still said to be the project DEM's, never the plan's other source
    tiles_plan = dict(plan, fonteQuote={"sorgente": "tegole"})
    gis.write_gpkg(tiles_plan, path, CRS, ground_takeoff=300.0)
    reference = gpkg.read_table(path, "ppg_scatti")[0][1]["riferimento_quote"]
    assert "DEM del progetto" in reference and "Terrain Tiles" not in reference


def test_gpkg_target_codes_are_integers_or_nothing(tmp_path, plan):
    piano = dict(plan, target={"crs": "EPSG:4326", "punti": [
        {"nome": "T1", "codice": "12", "lat": 45.0, "lon": 10.0, "quota": None},
        {"nome": "T2", "codice": "GCP-A", "lat": 45.0001, "lon": 10.0, "quota": None},
        {"nome": "T3", "codice": 7.0, "lat": 45.0002, "lon": 10.0, "quota": "251.5"}]})
    path = tmp_path / "t.gpkg"
    gis.write_gpkg(piano, path, CRS)
    con = sqlite3.connect(path)
    rows = con.execute("SELECT codice, typeof(codice), quota, typeof(quota) FROM ppg_target ORDER BY fid").fetchall()
    con.close()
    assert rows == [(12, "integer", None, "null"), (None, "null", None, "null"), (7, "integer", 251.5, "real")]


def test_gpkg_refuses_a_file_that_is_not_a_geopackage(tmp_path, plan):
    other = tmp_path / "relazione.gpkg"
    other.write_bytes(b"not a database")
    with pytest.raises(FileExistsError):
        gis.write_gpkg(plan, other, CRS)
    assert other.read_bytes() == b"not a database"
    empty = tmp_path / "vuoto.gpkg"
    empty.write_bytes(b"")
    gis.write_gpkg(plan, empty, CRS)
    assert gpkg.is_geopackage(empty)
    with pytest.raises(ValueError):
        gis.write_gpkg(plan, tmp_path / "locale.gpkg", None)
