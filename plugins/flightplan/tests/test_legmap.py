"""The map of a leg: the chunk's orthophoto underneath (grey without one), the route of that leg alone on top.
Ported from the planner's tests; its satellite tiles are replaced by the orthophoto, so the tests on tiles
became tests on a synthetic orthophoto."""
import io
import zipfile
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from ppg_flightplan import legmap
from ppg_flightplan import plan as pl
from planfixtures import reference_plan

# The four quadrants of the synthetic orthophoto, around the centre of the reference plan.
NW, NE, SW, SE = (200, 40, 40), (40, 200, 40), (40, 40, 200), (200, 200, 40)


@pytest.fixture
def test_plan():
    return reference_plan()


def _pixels(img):
    return list(img.get_flattened_data()) if hasattr(img, "get_flattened_data") else list(img.getdata())


def _orthophoto(path, centre=(45.0, 10.0), half_m=1500.0, cell=2.0, overviews=True):
    """A 3-band uint8 GeoTIFF in EPSG:32632 around ``centre``, one colour per quadrant."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(centre[1], centre[0])
    size = int(2 * half_m / cell)
    a = np.zeros((3, size, size), dtype=np.uint8)
    h = size // 2
    for colour, rows, cols in ((NW, slice(0, h), slice(0, h)), (NE, slice(0, h), slice(h, None)),
                               (SW, slice(h, None), slice(0, h)), (SE, slice(h, None), slice(h, None))):
        for b in range(3):
            a[b, rows, cols] = colour[b]
    profile = dict(driver="GTiff", width=size, height=size, count=3, dtype="uint8", crs="EPSG:32632",
                   transform=from_origin(e - half_m, n + half_m, cell, cell), tiled=True)
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(a)
        if overviews:
            ds.build_overviews([2, 4, 8])
    return path


def test_the_map_of_the_leg_is_a_jpeg_of_the_right_size(test_plan):
    data = legmap.leg_image(test_plan, 0)
    img = Image.open(io.BytesIO(data))
    assert img.format == "JPEG" and img.size == (legmap.WIDTH, legmap.HEIGHT)
    # the route is drawn: the colour of the block (red) appears
    assert any(r > 200 and g < 80 and b < 80 for r, g, b in _pixels(img)[::50])
    png = Image.open(io.BytesIO(legmap.leg_image(test_plan, 0, fmt="PNG")))
    assert png.format == "PNG"


def test_without_a_background_the_map_is_grey_but_with_the_route(test_plan):
    for background in (None, legmap.grey_background):
        img = Image.open(io.BytesIO(legmap.leg_image(test_plan, 1, background=background)))
        most = max(img.getcolors(maxcolors=1 << 20))[1]
        assert all(abs(a - b) <= 3 for a, b in zip(most, legmap.GREY))


def test_a_leg_that_does_not_exist(test_plan):
    with pytest.raises(IndexError):
        legmap.leg_image(test_plan, 99)


def test_the_caption_says_who_the_leg_is_and_its_numbers(test_plan):
    leg = pl.plan_legs(test_plan)[0]
    line1, line2 = legmap.caption(test_plan, leg)
    assert line1 == "Campo di prova · A nadirale · tratta 1 di " + str(leg.total)
    assert "quota 110 m" in line2 and "camera -90°" in line2 and f"{leg.route.shots()} scatti" in line2
    assert "GSD" in line2
    test_plan["djiFly"] = {leg.key: {"nomeDji": "01/02/2026 10:00:00"}}
    assert legmap.caption(test_plan, leg)[0].endswith("· DJI Fly 01/02/2026 10:00:00")


def test_the_zip_has_one_map_per_leg_with_the_file_name_of_the_leg(test_plan):
    legs = pl.plan_legs(test_plan)
    with zipfile.ZipFile(io.BytesIO(legmap.zip_images(test_plan))) as z:
        assert z.namelist() == [f"{leg.file_name}.jpg" for leg in legs]
        assert Image.open(io.BytesIO(z.read(z.namelist()[0]))).format == "JPEG"


def test_the_orthophoto_is_read_under_the_leg_with_north_up(tmp_path, test_plan):
    chunk = SimpleNamespace(orthomosaic_path=_orthophoto(tmp_path / "ortho.tif"))
    background = legmap.orthophoto_background(chunk)
    assert background is not None and background.attribution
    # a box of 400 m around the centre: each quarter of the image is its quadrant of the orthophoto
    d_lat, d_lon = 200 / 111_320.0, 200 / (111_320.0 * np.cos(np.radians(45.0)))
    img = background((10.0 - d_lon, 45.0 - d_lat, 10.0 + d_lon, 45.0 + d_lat), 200, 160)
    assert img.size == (200, 160)
    for (x, y), colour in (((50, 40), NW), ((150, 40), NE), ((50, 120), SW), ((150, 120), SE)):
        assert all(abs(a - b) <= 10 for a, b in zip(img.getpixel((x, y)), colour)), (x, y, img.getpixel((x, y)))
    # the map of the ring (centred on the centre of the plan): its middle is where the four quadrants meet
    ring = legmap.draw_leg(test_plan, len(pl.plan_legs(test_plan)) - 1, background=background)
    colours = {c for _, c in ring.getcolors(maxcolors=1 << 20) if _ > 5000}
    assert any(all(abs(a - b) <= 10 for a, b in zip(c, q)) for c in colours for q in (NW, NE, SW, SE))


def test_the_centre_pixel_has_the_colour_of_the_orthophoto_there(tmp_path):
    """A uniform orthophoto under a leg: the background's centre pixel equals the GeoTIFF colour (within 10)."""
    import rasterio
    from pyproj import Transformer
    from rasterio.transform import from_origin

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(10.0, 45.0)
    a = np.empty((3, 400, 400), dtype=np.uint8)
    a[0], a[1], a[2] = 37, 151, 222
    path = tmp_path / "uniform.tif"
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=3, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(e - 400, n + 400, 2.0, 2.0)) as ds:
        ds.write(a)
    background = legmap.orthophoto_background(SimpleNamespace(orthomosaic_path=path))
    d = 0.001
    img = background((10.0 - d, 45.0 - d, 10.0 + d, 45.0 + d), 101, 101)
    assert all(abs(x - y) <= 10 for x, y in zip(img.getpixel((50, 50)), (37, 151, 222)))


def test_no_orthophoto_no_background(tmp_path):
    assert legmap.orthophoto_background(SimpleNamespace(orthomosaic_path=None)) is None
    assert legmap.orthophoto_background(SimpleNamespace(orthomosaic_path=tmp_path / "manca.tif")) is None
    assert legmap.orthophoto_background(SimpleNamespace()) is None
    # an orthophoto elsewhere: nothing under the leg, grey
    chunk = SimpleNamespace(orthomosaic_path=_orthophoto(tmp_path / "altrove.tif", centre=(45.5, 9.5),
                                                         half_m=200.0, overviews=False))
    background = legmap.orthophoto_background(chunk)
    assert background((9.99, 44.99, 10.01, 45.01), 50, 50) is None


def test_the_preview_for_dji_fly_is_a_1920x1080_jpeg_with_the_number(test_plan, tmp_path):
    img = Image.open(io.BytesIO(legmap.dji_preview(test_plan, 0, 7)))
    assert img.format == "JPEG" and img.size == (1920, 1080)
    chunk = SimpleNamespace(orthomosaic_path=_orthophoto(tmp_path / "ortho.tif"))
    img = Image.open(io.BytesIO(legmap.dji_preview(test_plan, 0, background=legmap.orthophoto_background(chunk))))
    assert img.size == (1920, 1080)


def test_a_tiny_area_is_enlarged_but_not_without_end():
    side = 40 / 111_320.0   # a square of 40 m
    piano = {"nome": "Piccolo", "camera": "mini4pro", "minutiBatteria": 18,
             "blocchi": [{"tipo": "griglia", "nome": "G", "quota": 10, "inclinazione": -90, "avanti": 80, "lato": 65,
                          "orientamento": 0, "velocita": 3, "modo": "intervallo",
                          "area": [[45.0, 10.0], [45.0, 10.0 + side], [45.0 + side, 10.0 + side],
                                   [45.0 + side, 10.0]]}]}
    img = Image.open(io.BytesIO(legmap.leg_image(piano, 0)))
    assert img.size == (1400, 1000)


def test_an_rgba_orthophoto_leaves_its_transparent_cells_grey(tmp_path):
    """The program's orthophotos are RGBA: where alpha is 0 the map stays grey."""
    import rasterio
    from pyproj import Transformer
    from rasterio.enums import ColorInterp
    from rasterio.transform import from_origin

    e, n = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform(10.0, 45.0)
    a = np.empty((4, 400, 400), dtype=np.uint8)
    a[0], a[1], a[2], a[3] = 37, 151, 222, 255
    a[3, :, :200] = 0                                       # the west half transparent
    path = tmp_path / "rgba.tif"
    with rasterio.open(path, "w", driver="GTiff", width=400, height=400, count=4, dtype="uint8",
                       crs="EPSG:32632", transform=from_origin(e - 400, n + 400, 2.0, 2.0)) as ds:
        ds.write(a)
        ds.colorinterp = [ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha]
    background = legmap.orthophoto_background(SimpleNamespace(orthomosaic_path=path))
    d = 0.002
    img = background((10.0 - d, 45.0 - d, 10.0 + d, 45.0 + d), 101, 101)
    assert all(abs(x - y) <= 10 for x, y in zip(img.getpixel((80, 50)), (37, 151, 222)))
    assert img.getpixel((20, 50)) == legmap.GREY


def test_no_test_reaches_the_network():
    import socket

    with pytest.raises(RuntimeError, match="network"):
        socket.create_connection(("192.0.2.1", 80), timeout=0.1)


def test_an_orthophoto_that_cannot_be_read_leaves_the_map_grey(tmp_path, test_plan, caplog):
    broken = tmp_path / "rotta.tif"
    broken.write_bytes(b"II*\x00 non sono un GeoTIFF")
    background = legmap.orthophoto_background(SimpleNamespace(orthomosaic_path=broken))
    assert background is not None
    assert background((9.99, 44.99, 10.01, 45.01), 50, 50) is None
    img = Image.open(io.BytesIO(legmap.leg_image(test_plan, 1, background=background)))
    most = max(img.getcolors(maxcolors=1 << 20))[1]
    assert all(abs(a - b) <= 3 for a, b in zip(most, legmap.GREY))
    assert "rotta.tif" in caplog.text
