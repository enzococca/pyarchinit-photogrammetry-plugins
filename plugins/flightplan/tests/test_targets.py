"""GNSS targets on the ground: the file of the GNSS (or the markers of the chunk) and the special points they
make. The planner's tests, moved to the neutral place of the tests (45.0 N 10.0 E, UTM zone 32)."""
import numpy as np
import pytest
from pyproj import Transformer

from ppg_flightplan import targets
from ppg_flightplan.geometry import distance, offset

SITE = (45.0, 10.0)


def _to_utm(lat, lon, zone, north=True):
    epsg = (32600 if north else 32700) + zone
    return Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True).transform(lon, lat)


def test_utm_to_degrees_and_back():
    for lat, lon in ((45.0, 10.0), (12.0, -60.0), (-33.0, 20.0), (60.0, 25.0)):
        zone = targets.utm_zone(lon)
        e, n = _to_utm(lat, lon, zone, lat >= 0)
        lat2, lon2 = targets.latlon_from_utm(e, n, zone, lat >= 0)
        assert distance((lat, lon), (lat2, lon2)) < 0.01          # a centimetre


def test_the_target_number_comes_from_the_name():
    assert [targets.code_of_name(n) for n in ("T12", "target 3", "GCP_07", "5", "cima")] == [12, 3, 7, 5, None]


def test_a_utm_file_without_header_is_understood_from_the_numbers():
    rows = []
    for k, (dlat, dlon) in enumerate(((0.0, 0.0), (0.001, 0.0), (0.0, 0.001)), 1):
        e, n = _to_utm(SITE[0] + dlat, SITE[1] + dlon, 32)
        rows.append(f"T{k},{n:.3f},{e:.3f},{310 + k}")               # north BEFORE east, as many GNSS do
    read = targets.read_targets_csv("\n".join(rows), near=SITE)
    assert read["crs"] == "EPSG:32632" and "zona_dedotta" in read["avvisi"]
    assert [t["codice"] for t in read["punti"]] == [1, 2, 3]
    assert distance((read["punti"][0]["lat"], read["punti"][0]["lon"]), SITE) < 0.05
    # the measured coordinates stay those of the file, to the millimetre: they are the GCPs
    e, n = _to_utm(*SITE, 32)
    assert abs(read["punti"][0]["est"] - e) < 0.001 and abs(read["punti"][0]["nord"] - n) < 0.001


def test_in_degrees_the_order_is_chosen_by_the_site():
    """45 and 10 can both be a latitude or a longitude: the order that puts the points on the site wins."""
    lat_lon = "A1;45,0000;10,0000;312,4\nA2;45,0013;10,0013;318,0"
    lon_lat = "A1;10,0000;45,0000;312,4\nA2;10,0013;45,0013;318,0"
    for text in (lat_lon, lon_lat):
        read = targets.read_targets_csv(text, near=SITE)
        assert read["crs"] == "EPSG:4326"
        assert distance((read["punti"][0]["lat"], read["punti"][0]["lon"]), SITE) < 1.0
        assert read["punti"][0]["quota"] == 312.4                     # the decimal comma, with the semicolon
    assert "ordine_incerto" in targets.read_targets_csv(lat_lon)["avvisi"]
    swapped = targets.read_targets_csv(lat_lon, order="lonlat")
    assert swapped["punti"][0]["lat"] == 10.0                         # told the order: nothing guessed


def test_with_a_header_nothing_is_guessed():
    read = targets.read_targets_csv("name,longitude,latitude,elevation,code\ncima,10.0000,45.0000,330.5,4\n"
                                    "fondo,10.0012,44.9983,305.0,9\n")
    assert [(t["nome"], t["codice"], t["quota"]) for t in read["punti"]] == [("cima", 4, 330.5), ("fondo", 9, 305.0)]
    assert read["punti"][0]["lat"] == 45.0 and read["avvisi"] == []


def test_a_wrong_file_is_said():
    with pytest.raises(targets.TargetError):
        targets.read_targets_csv("")
    with pytest.raises(targets.TargetError):
        targets.read_targets_csv("T1,4000000,300000,10")              # metres, and nobody says of which zone
    with pytest.raises(targets.TargetError):
        targets.read_targets_csv("T1,4000000,300000,10", epsg=999999)  # a system nobody knows
    assert "senza_quota" in targets.read_targets_csv("T1,45.0000,10.0000", near=SITE)["avvisi"]


@pytest.mark.parametrize("epsg", [3003, 3004, 6707])
def test_a_non_utm_epsg_is_converted(epsg):
    """Gauss-Boaga (3003, 3004) and RDN2008 / UTM 32N (6707) are read through pyproj, which the planner refused."""
    points = [SITE, (45.0010, 10.0005), (44.9995, 10.0012)]
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    back = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    rows = []
    for k, (lat, lon) in enumerate(points, 1):
        e, n = forward.transform(lon, lat)
        rows.append(f"T{k},{e:.4f},{n:.4f},{100 + k}")
    for text in ("\n".join(rows), "name,east,north,height\n" + "\n".join(rows)):
        read = targets.read_targets_csv(text, epsg=epsg)
        assert read["crs"] == f"EPSG:{epsg}"
        for row, t in zip(rows, read["punti"]):
            e, n = (float(x) for x in row.split(",")[1:3])
            lon, lat = back.transform(e, n)
            assert abs(t["lat"] - lat) < 1e-6 and abs(t["lon"] - lon) < 1e-6
            assert t["est"] == e and t["nord"] == n                   # as measured
        assert distance((read["punti"][0]["lat"], read["punti"][0]["lon"]), SITE) < 0.5


def _plan():
    route = [list(offset(SITE, 90.0, 40.0 * i)) for i in range(6)]
    return {"nome": "campo", "camera": "mini4pro", "minutiBatteria": 18,
            "blocchi": [{"nome": "b", "tipo": "percorso", "quota": 60, "punti": route}]}


def test_a_special_point_with_ladder_on_each_target():
    read = targets.read_targets_csv("T1,45.0003,10.0002,312\nT2,45.0008,10.0012,318\nT3,44.9993,10.0007,309",
                                    near=SITE)
    piano = targets.put_targets(_plan(), read, metres=10)
    specials = piano["blocchi"][0]["speciali"]
    assert [sp["nome"] for sp in specials] == ["T1", "T2", "T3"]
    assert all(sp["scaletta"] and not sp["aggancio"] and sp["inclinazione"] == -90 for sp in specials)
    assert specials[0]["discesa"] == {"modo": "suolo", "metri": 10.0} and specials[0]["id"] == "target-t1"
    # putting them again does not double them, and removing them leaves the plan as it was
    targets.put_targets(piano, read)
    assert len(piano["blocchi"][0]["speciali"]) == 3
    targets.remove_targets(piano)
    assert "speciali" not in piano["blocchi"][0] and "target" not in piano


def test_the_targets_fly_in_the_route():
    from ppg_flightplan import plan as pl
    read = targets.read_targets_csv("T1,45.0003,10.0002,312", near=SITE)
    piano = targets.put_targets(_plan(), read)
    outcome = pl.special_outcomes(piano["blocchi"][0], piano)[0]
    assert outcome["nome"] == "T1" and outcome["waypoint"] is None
    assert outcome["avvisi"] == ["senza_terreno"]                     # no ground known: no descent, and it says so


def _chunk(crs="EPSG:32632"):
    from pyarchinit_photogrammetry.core.camera import Reference
    from pyarchinit_photogrammetry.core.chunk import Chunk
    from pyarchinit_photogrammetry.core.marker import Marker

    forward = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    e1, n1 = forward.transform(10.0, 45.0)
    e2, n2 = forward.transform(10.0010, 45.0007)
    markers = [Marker(key=0, label="target 1", reference=Reference(location=[e1, n1, 101.25])),
               Marker(key=1, label="cima", code=7, reference=Reference(location=[e2, n2, 99.5])),
               Marker(key=2, label="T3", position=np.array([e1 + 5.0, n1, 100.0]))]
    return Chunk(key=0, label="rilievo", crs=crs, markers=markers), [(e1, n1), (e2, n2)]


def test_markers_become_targets():
    chunk, measured = _chunk()
    read = targets.targets_from_markers(chunk)
    assert read["crs"] == "EPSG:32632"
    assert [t["nome"] for t in read["punti"]] == ["target 1", "cima"]
    assert [t["codice"] for t in read["punti"]] == [1, 7]              # from the name, or the marker's own code
    back = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    for t, (e, n) in zip(read["punti"], measured):
        lon, lat = back.transform(e, n)
        assert abs(t["lat"] - lat) < 1e-7 and abs(t["lon"] - lon) < 1e-7
        assert (t["est"], t["nord"]) == (e, n)
    assert read["punti"][0]["quota"] == 101.25
    # they go in the plan as the targets of a file do
    piano = targets.put_targets(_plan(), read)
    assert [sp["nome"] for sp in piano["blocchi"][0]["speciali"]] == ["target 1", "cima"]


def test_markers_estimated_only_when_asked():
    chunk, _ = _chunk()
    read = targets.targets_from_markers(chunk, only_with_reference=False)
    assert [t["nome"] for t in read["punti"]] == ["target 1", "cima", "T3"]
    assert "stimati" in read["avvisi"]


def test_markers_of_a_local_chunk_are_refused():
    chunk, _ = _chunk()
    chunk.crs = "LOCAL"
    with pytest.raises(targets.TargetError):
        targets.targets_from_markers(chunk)
    chunk, _ = _chunk()
    chunk.markers = [m for m in chunk.markers if m.reference.location is None]
    with pytest.raises(targets.TargetError):
        targets.targets_from_markers(chunk)


# ---- fix round 1

def _grid_rows(epsg, points, north_first=False):
    forward = Transformer.from_crs("EPSG:4326", f"EPSG:{epsg}", always_xy=True)
    rows = []
    for k, (lat, lon) in enumerate(points, 1):
        e, n = forward.transform(lon, lat)
        a, b = (n, e) if north_first else (e, n)
        rows.append(f"T{k},{a:.3f},{b:.3f},{400 + k}")
    return "\n".join(rows)


def _on(read, points, tol_m=0.05):
    return all(distance((t["lat"], t["lon"]), p) < tol_m for t, p in zip(read["punti"], points))


def test_a_grid_where_the_east_is_larger_is_not_swapped():
    """Swiss LV95: east 2.6 M, north 1.2 M. The UTM rule (north the larger) put these points in the Atlantic;
    either order is now told by where the system is used."""
    points = [(46.5, 8.0), (46.5008, 8.0010), (46.4995, 8.0012)]
    for north_first in (False, True):
        read = targets.read_targets_csv(_grid_rows(2056, points, north_first), epsg=2056)
        assert _on(read, points) and "ordine_incerto" not in read["avvisi"]


def test_british_grid_order_from_the_site_the_order_or_the_system():
    """The British grid: both orders fall inside Great Britain, so the area of use cannot tell. The site does,
    or ``order``; with neither, the system's own axis order (east, north) is taken and the doubt is said."""
    points = [(51.0, -1.5), (51.0008, -1.4990), (50.9995, -1.4988)]
    read = targets.read_targets_csv(_grid_rows(27700, points), epsg=27700)
    assert _on(read, points) and "ordine_incerto" in read["avvisi"]
    north_first = _grid_rows(27700, points, north_first=True)
    assert _on(targets.read_targets_csv(north_first, epsg=27700, near=(51.0, -1.5)), points)
    assert _on(targets.read_targets_csv(north_first, epsg=27700, order="latlon"), points)
    assert _on(targets.read_targets_csv(_grid_rows(27700, points), epsg=27700, order="lonlat"), points)


def test_utm_keeps_the_rule_of_the_larger_north():
    points = [SITE, (45.0010, 10.0005)]
    read = targets.read_targets_csv(_grid_rows(32632, points, north_first=True), epsg=32632)
    assert _on(read, points, 0.01) and read["avvisi"] == []


def test_a_geographic_epsg_is_converted_and_a_projected_one_with_degrees_refused():
    lat, lon = 45.0, 10.0
    lon_m, lat_m = Transformer.from_crs("EPSG:4326", "EPSG:4806", always_xy=True).transform(lon, lat)
    read = targets.read_targets_csv(f"name,lon,lat\nA,{lon_m:.8f},{lat_m:.8f}\n", epsg=4806)
    assert read["crs"] == "EPSG:4806"
    assert distance((read["punti"][0]["lat"], read["punti"][0]["lon"]), (lat, lon)) < 0.01
    assert read["punti"][0]["est"] == round(lon_m, 8)                # as written in the file
    with pytest.raises(targets.TargetError):
        targets.read_targets_csv("A,45.0,10.0,100", epsg=32632)

