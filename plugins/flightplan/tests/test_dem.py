"""The project's DEM, read and queried by latitude and longitude: what lets the next flight follow the
terrain as it really is, instead of at thirty metres of mesh. Any coordinate system."""
import numpy as np
import pytest
import rasterio.errors  # noqa: F401 - for the warning filter
from pyproj import Transformer

from ppg_flightplan import design, tiles
from ppg_flightplan.dem import DemError, RasterDem, chunk_dem
from terrainfixtures import fake_tile, write_dem

pytestmark = pytest.mark.filterwarnings("ignore::rasterio.errors.NotGeoreferencedWarning")

STEP = 0.000001                      # ~0.11 m in latitude


def test_reads_heights_in_degrees(tmp_path):
    # a little square at 45 N 10 E: ten centimetres of mesh
    heights = [[250.0 + x + 10 * y for x in range(10)] for y in range(8)]
    dem = RasterDem(write_dem(tmp_path / "dem.tif", heights, x0=10.0, y0=45.0, step=STEP))
    assert dem.in_degrees and dem.epsg == 4326
    assert dem.height(45.0, 10.0) == pytest.approx(250.0)                    # top-left corner
    # the centre of the cell three to the east and two to the south
    assert dem.height(45.0 - 2.5 * STEP, 10.0 + 3.5 * STEP) == pytest.approx(273.0)
    assert dem.to_json()["magliaM"] == pytest.approx(0.111, abs=0.01)
    assert dem.heights([(45.0 - 0.5 * STEP, 10.0 + 0.5 * STEP), (45.0 - 7.5 * STEP, 10.0 + 9.5 * STEP)]) \
        == [pytest.approx(250.0), pytest.approx(329.0)]


def test_outside_the_raster_says_none(tmp_path):
    """The DEM of the survey covers the site, not the world: outside, the free heights come back."""
    dem = RasterDem(write_dem(tmp_path / "dem.tif", [[100.0] * 4] * 4, x0=10.0, y0=45.0, step=0.0001))
    assert dem.height(44.99995, 10.00005) == 100.0
    assert dem.height(45.10, 10.0) is None          # too far north
    assert dem.height(45.0, 10.30) is None          # too far east
    assert dem.heights([(45.10, 10.0), (44.99995, 10.00005)]) == [None, 100.0]


def test_missing_data_is_not_a_height(tmp_path):
    """ODM writes -9999 and Metashape -32767 where they do not know: they are not heights."""
    heights = [[100.0, -9999.0], [-32767.0, float("nan")]]
    dem = RasterDem(write_dem(tmp_path / "dem.tif", heights, x0=10.0, y0=45.0, step=0.0001))
    assert dem.height(44.99995, 10.00005) == 100.0
    assert dem.height(44.99995, 10.00015) is None
    assert dem.height(44.99985, 10.00005) is None
    assert dem.height(44.99985, 10.00015) is None
    # and the nodata of the file, whatever it is, is not a height either
    dem2 = RasterDem(write_dem(tmp_path / "dem2.tif", [[100.0, 55.0]], x0=10.0, y0=45.0, step=0.0001, nodata=55.0))
    assert dem2.height(44.99995, 10.00015) is None


def test_reads_a_utm_dem(tmp_path):
    """ODM writes in UTM: the coordinates are projected before looking for the cell."""
    lat, lon = 45.0, 10.0
    east, north = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(lon, lat)
    heights = [[300.0 + x + 5 * y for x in range(6)] for y in range(6)]
    dem = RasterDem(write_dem(tmp_path / "dem_utm.tif", heights, x0=east - 0.5, y0=north + 0.5, step=1.0,
                              crs="EPSG:32632"))
    assert not dem.in_degrees and dem.epsg == 32632
    assert dem.height(lat, lon) == pytest.approx(300.0)
    # two metres east: two cells farther
    back = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
    lon2, lat2 = back.transform(east + 2.0, north)
    assert dem.height(lat2, lon2) == pytest.approx(302.0)
    assert dem.to_json()["magliaM"] == pytest.approx(1.0)


@pytest.mark.parametrize("crs", ["EPSG:3004", "EPSG:3857"])
def test_reads_a_dem_in_any_crs(tmp_path, crs):
    """Any coordinate system pyproj knows (Photo3D took only degrees and WGS84 UTM)."""
    lat, lon = 45.0, 10.0
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(lon, lat)
    heights = np.arange(100, dtype="float32").reshape(10, 10) + 50.0
    dem = RasterDem(write_dem(tmp_path / "dem.tif", heights, x0=x - 2.5, y0=y + 2.5, step=1.0, crs=crs))
    assert dem.height(lat, lon) == pytest.approx(heights[2, 2])
    back = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon2, lat2 = back.transform(x + 3.0, y - 4.0)
    assert dem.height(lat2, lon2) == pytest.approx(heights[6, 5])


def test_a_file_without_georeference_is_refused(tmp_path):
    import rasterio

    f = tmp_path / "senza.tif"
    with rasterio.open(f, "w", driver="GTiff", width=4, height=4, count=1, dtype="float32") as ds:
        ds.write(np.zeros((1, 4, 4), dtype="float32"))
    with pytest.raises(DemError, match="dove sta nel mondo"):
        RasterDem(f)
    with pytest.raises(DemError, match="non trovato"):
        RasterDem(tmp_path / "manca.tif")
    (tmp_path / "rotto.tif").write_bytes(b"not a tiff")
    with pytest.raises(DemError, match="non leggibile"):
        RasterDem(tmp_path / "rotto.tif")


def test_the_dem_of_the_chunk(tmp_path):
    class Chunk:
        dem_path = None

    c = Chunk()
    assert chunk_dem(c) is None
    c.dem_path = tmp_path / "manca.tif"
    assert chunk_dem(c) is None
    c.dem_path = write_dem(tmp_path / "dem.tif", [[1.0]], x0=10.0, y0=45.0, step=0.001)
    assert chunk_dem(c).height(44.9995, 10.0005) == 1.0
    import rasterio

    local = tmp_path / "locale.tif"            # a chunk in a local system: no latitude and longitude
    with rasterio.open(local, "w", driver="GTiff", width=2, height=2, count=1, dtype="float32") as ds:
        ds.write(np.ones((1, 2, 2), dtype="float32"))
    c.dem_path = local
    assert chunk_dem(c) is None


# ---- the DEM of the survey inside the plan

def test_your_dem_wins_over_the_tiles_and_the_rest_aligns(tmp_path, monkeypatch):
    """Where the survey's DEM reaches it rules; outside, the free tiles are used, moved by the measured
    offset - the tiles give heights above sea level, the survey's DEM may take them from the GNSS."""
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: fake_tile(200.0))
    # a small DEM around the first corner of the area, twenty metres higher than the tiles
    step = 0.0002
    heights = [[220.0 for _ in range(20)] for _ in range(20)]
    dem = RasterDem(write_dem(tmp_path / "dem.tif", heights, x0=9.9970, y0=44.9990, step=step))
    area = [(44.9970, 9.9970), (44.9970, 10.0030), (45.0030, 10.0030), (45.0030, 9.9970)]
    piano = {"nome": "prova", "camera": "mini4pro", "alta": False, "minutiBatteria": 18,
             "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in area], "quota": 30,
                          "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0,
                          "velocita": 5, "modo": "per_punto", "seguiTerreno": True}]}
    outcome = design.apply_terrain(piano, dem=dem)
    assert outcome["dem"]["file"].endswith("dem.tif")
    assert outcome["scartoM"] == pytest.approx(20.0, abs=0.5)      # the jump between the two sources
    # flat terrain in both sources, once the offset is taken away: all the heights equal
    q = piano["blocchi"][0]["quote"]
    assert max(q) - min(q) < 0.6


def test_without_overlap_no_offset_is_invented(tmp_path, monkeypatch):
    monkeypatch.setattr(tiles, "_tile_from_network", lambda z, x, y: None)   # no network
    dem = RasterDem(write_dem(tmp_path / "dem.tif", [[100.0] * 4] * 4, x0=10.0, y0=45.0, step=0.0001))
    points = [(44.99995, 10.00005), (44.99985, 10.00015)]
    assert design.source_offset(points, dem, 13) is None
    # and the ground heights stay those of the DEM where it is, None where there is nothing
    heights = design.ground_heights(points + [(45.10, 10.0)], dem, 13, None)
    assert heights[0] == 100.0 and heights[1] == 100.0 and heights[2] is None
