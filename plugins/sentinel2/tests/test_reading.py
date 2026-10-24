"""The area and its grid, and the window of the area read from the band files: the footprint of
the DEM in the chunk's system, a local chunk refused with an explanation, the grid in UTM with its
corners on 20 m; a 10 m band copied as it is, a 20 m band interpolated, the classes taken from the
nearest pixel, a tile of another zone warped, nothing outside a tile; the scene classification
masking clouds, shadows and a buffer, a cloudy acquisition left out before its bands are read."""
from __future__ import annotations

import numpy as np
import pytest
import s2synth

from ppg_sentinel2 import area as area_mod
from ppg_sentinel2 import reader, stac


@pytest.fixture(scope="module")
def cogs(tmp_path_factory):
    folder = tmp_path_factory.mktemp("cogs")
    return folder, s2synth.world(folder)


def _grid(margin=300.0, tmp=None):
    doc, ch = s2synth.project(tmp)
    a = area_mod.area_of_chunk(ch, margin_m=margin)
    return doc, ch, a, area_mod.grid_of(a)


def _tile(path):
    import rasterio

    with rasterio.open(path) as src:
        return src.read(1)


def test_area_and_grid(tmp_path):
    doc, ch, a, g = _grid(tmp=tmp_path)
    assert a.source == "dem" and a.crs == s2synth.CHUNK_CRS and len(a.ring) >= 4
    # the footprint, back in UTM, is the site
    x0, y0, x1, y1 = a.in_crs(s2synth.UTM).bounds
    assert abs(x0 - s2synth.SITE[0]) < 10 and abs(y1 - s2synth.SITE[3]) < 10 and abs(x1 - s2synth.SITE[2]) < 10
    w, s, e, n = a.lonlat_bbox()
    assert 8.99 < w < e < 9.03 and 45.17 < s < n < 45.20
    assert g.crs == "EPSG:32632" and g.cell == 10.0
    assert g.x0 % 20 == 0 and g.y1 % 20 == 0 and (g.width * g.cell) % 20 == 0 and (g.height * g.cell) % 20 == 0
    gx0, gy0, gx1, gy1 = g.bounds
    assert gx0 <= s2synth.SITE[0] - 300 and gx1 >= s2synth.SITE[2] + 300 and gy1 >= s2synth.SITE[3] + 300
    mask = area_mod.area_mask(a, g)
    assert 0.12 < mask.mean() < 0.2 and mask.sum() > 1000
    # a box of longitudes and latitudes, a polygon drawn
    b = area_mod.area_of_bbox(ch, 9.01, 45.18, 9.02, 45.19, 100)
    assert b.source == "bbox" and b.crs == s2synth.CHUNK_CRS
    p = area_mod.area_of_polygon(ch, [[0, 0], [10, 10]], 50)  # two points: their box
    assert len(p.ring) == 4 and p.margin_m == 50
    assert area_mod.Grid.from_dict(g.to_dict()) == g


def test_a_local_chunk_cannot_be_placed(tmp_path):
    import pyarchinit_photogrammetry as ppg

    doc = ppg.Document()
    ch = doc.add_chunk("Locale")
    doc.save(tmp_path / "locale.ppg")
    with pytest.raises(area_mod.NotPlaced, match="LOCAL"):
        area_mod.area_of_chunk(ch)
    with pytest.raises(area_mod.NotPlaced):
        area_mod.area_of_polygon(ch, [[0, 0], [1, 0], [1, 1]])
    # with a box of longitudes and latitudes it can be searched, in WGS84
    assert area_mod.area_of_bbox(ch, 9.01, 45.18, 9.02, 45.19).crs == "EPSG:4326"


def test_bands_on_the_grid(cogs, tmp_path):
    folder, _items = cogs
    _doc, _ch, _a, g = _grid(tmp=tmp_path)
    r0, c0 = int((s2synth.TILE_Y1 - g.y1) / 10), int((g.x0 - s2synth.TILE_X0) / 10)
    # a 10 m band: the very numbers of the file (the west tile of 14 June: nothing east of 501200)
    west = reader.read_band(str(folder / "2023-06-14_32TMR_B04.tif"), g)
    tile = _tile(folder / "2023-06-14_32TMR_B04.tif")
    assert np.array_equal(west, tile[r0:r0 + g.height, c0:c0 + g.width]) and (west == 0).any() and (west > 0).any()
    red = reader.read_band(str(folder / "2023-08-03_32TMR_B04.tif"), g)
    tile = _tile(folder / "2023-08-03_32TMR_B04.tif")
    assert red.dtype == np.uint16 and red.shape == g.shape
    assert np.array_equal(red, tile[r0:r0 + g.height, c0:c0 + g.width])
    # a 20 m band: bilinear between the centres of its pixels (a ramp stays a ramp, half the step)
    import rasterio
    from rasterio.transform import from_origin

    ramp = np.tile((1000 + 10 * np.arange(150)).astype(np.uint16), (150, 1))
    path = tmp_path / "ramp_B8A.tif"
    with rasterio.open(path, "w", driver="GTiff", width=150, height=150, count=1, dtype="uint16", crs=s2synth.UTM,
                       transform=from_origin(s2synth.TILE_X0, s2synth.TILE_Y1, 20, 20), nodata=0) as dst:
        dst.write(ramp, 1)
    up = reader.read_band(str(path), g).astype(float)
    cols = np.arange(g.width) + c0  # columns of the 10 m cells in the tile
    expected = 997.5 + 5.0 * cols
    assert np.abs(up[5:-5, 5:-5] - expected[None, 5:-5]).max() <= 1.0
    # the classes: the nearest pixel, a 20 m pixel is four cells
    scl = reader.read_band(str(folder / "2023-07-09_32TMR_SCL.tif"), g, classes=True)
    tile_scl = _tile(folder / "2023-07-09_32TMR_SCL.tif")
    assert np.array_equal(scl[::2, ::2], tile_scl[r0 // 2:r0 // 2 + g.height // 2, c0 // 2:c0 // 2 + g.width // 2])
    assert set(np.unique(scl)) <= {3, 4, 9} and (scl == 9).any() and (scl == 3).any()
    # a tile in another UTM zone: warped onto the grid
    from rasterio.warp import Resampling, calculate_default_transform, reproject

    with rasterio.open(folder / "2023-08-03_32TMR_B04.tif") as src:
        tr, w, h = calculate_default_transform(src.crs, "EPSG:32633", src.width, src.height, *src.bounds, resolution=10)
        other = np.zeros((h, w), np.uint16)
        reproject(rasterio.band(src, 1), other, dst_transform=tr, dst_crs="EPSG:32633", resampling=Resampling.nearest)
    path33 = tmp_path / "zone33_B04.tif"
    with rasterio.open(path33, "w", driver="GTiff", width=w, height=h, count=1, dtype="uint16", crs="EPSG:32633",
                       transform=tr, nodata=0) as dst:
        dst.write(other, 1)
    warped = reader.read_band(str(path33), g).astype(float)
    inner = (slice(5, -5), slice(5, -5))
    assert (warped[inner] > 0).all()
    assert np.corrcoef(warped[inner].ravel(), red[inner].astype(float).ravel())[0, 1] > 0.6
    # a grid beyond the tile: nothing there
    far = area_mod.Grid(s2synth.UTM, s2synth.TILE_X0 + 2900, g.y1, 10.0, 30, 10)
    out = reader.read_band(str(folder / "2023-08-03_32TMR_B04.tif"), far)
    assert (out[:, :10] > 0).all() and (out[:, 10:] == 0).all()


def test_clear_mask_and_buffer():
    scl = np.full((20, 20), 4, np.uint8)
    scl[10, 10] = 9
    scl[0, 0] = 0
    scl[19, 19] = 11
    scl[0, 19] = 5
    ok = reader.clear_mask(scl, reader.CLEAR, 20.0, 10.0)
    assert not ok[10, 10] and not ok[10, 12] and not ok[8, 10] and not ok[11, 11] and ok[10, 13] and ok[12, 12]
    assert not ok[0, 0] and not ok[19, 19] and ok[0, 19]
    assert reader.clear_mask(scl, reader.CLEAR, 0.0, 10.0).sum() == 400 - 3
    assert reader.clear_mask(scl, (4,), 0.0).sum() == 400 - 4


def test_acquisitions_read_masked_and_left_out(cogs, tmp_path):
    folder, items = cogs
    _doc, _ch, a, g = _grid(tmp=tmp_path)
    acqs = stac.acquisitions(stac.keep_clear(items, 30))
    by_id = {x["id"]: x for x in acqs}
    mask = area_mod.area_mask(a, g)
    # two tiles of one datatake fill each other
    june = reader.read_acquisition(by_id["20230614_S2B"], g, mask)
    assert june["status"] == "read" and june["clear"] == 1.0
    assert all((june["bands"][b] > 0).all() for b in stac.BANDS) and (june["scl"] == 4).all()
    # a cloud and its shadow, with 20 m around them
    july = reader.read_acquisition(by_id["20230709_S2B"], g, mask)
    assert july["status"] == "read" and 0.5 < july["clear"] < 0.95
    # the cloudy one: its bands are never read (here they are not even there)
    cloudy = by_id["20230520_S2B"]
    missing = {**cloudy, "items": [{**it, "assets": {**it["assets"], **{b: {**it["assets"][b], "href": str(tmp_path / "gone.tif")}
                                                                       for b in stac.BANDS}}} for it in cloudy["items"]]}
    r = reader.read_acquisition(missing, g, mask)
    assert r == {"status": "cloudy", "clear": 0.0}
    # numbers harmonised although the catalog lists the offset: noticed, offset 0
    sept = reader.read_acquisition(by_id["20230912_S2B"], g, mask)
    assert sept["offset_checked"] and set(sept["offset"].values()) == {0.0}
    assert abs(np.median(sept["bands"]["B04"]) * 1e-4 - 0.32 * (1 - 0.40) / (1 + 0.40)) < 0.005
    assert not june["offset_checked"] and june["offset"]["B04"] == -0.1
    # the cache: numbers, scale and offset, the grid checked
    path = reader.write_cache(tmp_path / "c" / "x.tif", g, by_id["20230614_S2B"], june, "notice")
    back = reader.read_cache(path, g)
    assert back is not None and back["tags"]["ATTRIBUTION"] == "notice"
    nir = june["bands"]["B08"].astype(np.float32) * np.float32(1e-4) - np.float32(0.1)
    assert np.allclose(back["reflectance"]["B08"], nir, atol=1e-6) and np.allclose(back["reflectance"]["B08"], 0.32, atol=1e-4)
    other = area_mod.Grid(g.crs, g.x0 + 20, g.y1, g.cell, g.width, g.height)
    assert reader.read_cache(path, other) is None and not reader.cache_valid(path, other) and reader.cache_valid(path, g)


def test_fetch_in_parallel_with_cancel(cogs, tmp_path):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    folder, items = cogs
    _doc, _ch, a, g = _grid(tmp=tmp_path)
    acqs = stac.acquisitions(stac.keep_clear(items, 30))
    mask = area_mod.area_mask(a, g)
    seen = []
    res = reader.fetch(acqs, g, mask, tmp_path / "cache", workers=3, attribution="x", progress=lambda f, m: seen.append(m))
    assert reader.describe(res) == {"read": 5, "cached": 0, "cloudy": 1, "failed": 0}
    assert sorted(p.name for p in (tmp_path / "cache").glob("*.tif")) == [f"{x['id']}.tif" for x in acqs if x["id"] != "20230520_S2B"]
    again = reader.fetch(acqs, g, mask, tmp_path / "cache", progress=lambda f, m: None)
    # those in the cache are not read again; the cloudy one is asked again here (the project skips it)
    assert reader.describe(again) == {"read": 0, "cached": 5, "cloudy": 1, "failed": 0}
    # cancelled at the first answer: what is under way stops, nothing half written
    progress = Progress(lambda f, m: None)
    calls = {"n": 0}

    def cancel_soon(f, m):
        calls["n"] += 1
        if calls["n"] > 1:
            progress.cancel()

    progress.callback = cancel_soon
    with pytest.raises(Cancelled):
        reader.fetch(acqs, g, mask, tmp_path / "cache2", workers=2, progress=progress)
    assert not list((tmp_path / "cache2").glob("*.part.tif"))
