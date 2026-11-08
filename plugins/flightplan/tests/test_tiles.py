"""Heights of the ground from the public tiles: decoding, cutting on the polygon, statistics, cache."""
import io

import pytest
from PIL import Image

from ppg_flightplan import tiles
from terrainfixtures import fake_tile


def test_reads_the_height_from_the_tile(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: fake_tile(246.5))
    assert tiles.height(45.0, 10.0) == pytest.approx(246.5, abs=0.01)


def test_without_network_says_none(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: None)
    assert tiles.height(45.0, 10.0) is None
    with pytest.raises(tiles.TerrainError):
        tiles.statistics([(45.0, 10.0), (45.0, 10.01), (45.01, 10.01)], step_m=50)


def test_tiles_are_reused(monkeypatch):
    asked = []
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: asked.append((z, x, y)) or fake_tile(100.0))
    tiles.height(45.0, 10.0)
    tiles.height(45.0001, 10.0001)
    assert len(asked) == 1


def test_statistics_on_an_inclined_terrain(monkeypatch):
    # one tile per height: farther north, higher
    def fake(z, x, y):
        return fake_tile(200.0 + y % 7)
    monkeypatch.setattr(tiles, "_tile_from_network", fake)
    polygon = [(44.990, 9.990), (44.990, 10.000), (45.000, 10.000), (45.000, 9.990)]
    s = tiles.statistics(polygon, step_m=50)
    assert s["campioni"] > 10
    assert s["minima"] <= s["mediana"] <= s["massima"]
    assert s["dislivello"] == round(s["massima"] - s["minima"], 1)
    assert s["ettari"] > 0 and s["riquadroM"][0] > 900          # ~1.1 km north-south
    assert s["fonte"].startswith("Quote:")


def test_point_heights(monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: fake_tile(310.0))
    q = tiles.point_heights([(44.99, 9.99), (45.0, 10.0)])
    assert q == [pytest.approx(310.0, abs=0.01), pytest.approx(310.0, abs=0.01)]


def test_an_empty_tile_is_not_terrain_at_minus_32768(monkeypatch):
    """A hole in the data is an all-black image: it must be recognised, or the flight following the terrain
    would start from absurd heights."""
    asked = []

    def fake(z, x, y):
        asked.append(z)
        return Image.new("RGB", (256, 256), (0, 0, 0)) if z == tiles.ZOOM else fake_tile(248.2)

    monkeypatch.setattr(tiles, "_tile_from_network", fake)
    q = tiles.height(45.0, 10.0)
    assert q == pytest.approx(248.2, abs=0.01)
    assert asked[0] == tiles.ZOOM and len(asked) > 1      # it tried another zoom
    # and if the data is there at no zoom, it says so instead of making it up
    tiles._cache.clear()
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: Image.new("RGB", (256, 256), (0, 0, 0)))
    assert tiles.height(45.0, 10.0) is None


def test_the_slope_between_neighbour_nodes():
    """Sorting the heights would only say how scattered they are: the slope is between those that touch."""
    step = 20.0
    flat = {(i, j): (0.0, 0.0, 100.0) for i in range(5) for j in range(5)}
    assert tiles.mean_slope(flat, step) == 0.0
    # a ramp rising 20 m every 20 m of step: 45 degrees
    ramp = {(i, j): (0.0, 0.0, 100.0 + i * 20.0) for i in range(5) for j in range(5)}
    assert tiles.mean_slope(ramp, step) == pytest.approx(45.0, abs=1.5)
    assert tiles.mean_slope({}, step) == 0.0


def test_details_say_how_fine():
    """Normal, fine, finest: different zooms, different mesh. They do not make up terrain where the data has
    none, but they take away the steps between one waypoint and the next."""
    assert tiles.DETAILS == {"normale": 13, "fine": 14, "finissimo": 15}
    assert tiles.zoom_for_detail("fine") == 14
    assert tiles.zoom_for_detail("FINISSIMO") == 15
    assert tiles.zoom_for_detail(None) == tiles.ZOOM
    assert tiles.zoom_for_detail("inventato") == tiles.ZOOM
    # at 45 degrees: 14 m, 7 m, 3 m of mesh
    meshes = [round(tiles.metres_per_pixel(z, 45.0)) for z in tiles.DETAILS.values()]
    assert meshes == [14, 7, 3]
    # nearer the equator the mesh is wider
    assert tiles.metres_per_pixel(13, 0.0) > tiles.metres_per_pixel(13, 60.0)


def test_the_detail_changes_the_tiles_asked(monkeypatch):
    asked = []
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: asked.append(z) or fake_tile(250.0))
    tiles.height(45.0, 10.0, tiles.zoom_for_detail("normale"))
    tiles.height(45.0, 10.0, tiles.zoom_for_detail("finissimo"))
    assert asked == [13, 15]


def test_disk_cache(tmp_path, monkeypatch):
    """A tile fetched once stays on disk: the next plan of the same site needs no network."""
    from conftest import REAL_TILE_FROM_NETWORK

    png = io.BytesIO()
    fake_tile(123.4).save(png, format="PNG")
    requests = []

    class Response:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def opener(request, timeout):
        requests.append((request.full_url, request.get_header("User-agent"), timeout))
        return Response(png.getvalue())

    monkeypatch.setattr(tiles.urllib.request, "urlopen", opener)
    monkeypatch.setattr(tiles, "_tile_from_network", REAL_TILE_FROM_NETWORK)
    tiles.set_cache_dir(tmp_path / "tiles")
    assert tiles.height(45.0, 10.0) == pytest.approx(123.4, abs=0.01)
    assert len(requests) == 1
    url, agent, timeout = requests[0]
    assert url.startswith("https://s3.amazonaws.com/elevation-tiles-prod/terrarium/13/")
    assert agent == "pyarchinit-photogrammetry flightplan" and timeout == 10.0
    assert list((tmp_path / "tiles").rglob("*.png"))

    def no_network(z, x, y):
        raise AssertionError("the network was asked again")

    tiles._cache.clear()
    monkeypatch.setattr(tiles, "_tile_from_network", no_network)
    assert tiles.height(45.0, 10.0) == pytest.approx(123.4, abs=0.01)
