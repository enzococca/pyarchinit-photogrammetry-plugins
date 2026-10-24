"""What the plugin makes on the synthetic season: the indices of each clear date (equal to the
truth of the fields), the series per date and per month with the attribution of the data, the
chart, the stacks, the composites, the crop-mark map that finds the buried wall (on the grid of
the scenes and on the chunk's, in another coordinate system), a second look without network, the
registry in the project, the exports and the QGIS entries."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import s2synth

from ppg_sentinel2 import analysis, project
from ppg_sentinel2 import area as area_mod

NOTICE = "Contains modified Copernicus Sentinel data 2023"


@pytest.fixture()
def fetched(tmp_path):
    """A project whose acquisitions were searched (fake catalog) and read (local files)."""
    items = s2synth.world(tmp_path / "cogs")
    doc, ch = s2synth.project(tmp_path)
    a = area_mod.area_of_chunk(ch, margin_m=300)
    project.search(ch, a, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=s2synth.FakeCatalog(items))
    counts = project.fetch(ch, workers=2)
    doc.save()
    return doc, ch, counts, tmp_path


def _centres(transform, shape):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]] + 0.5
    a, b, c, d, e, f = tuple(transform)[:6]
    return a * cols + b * rows + c, d * cols + e * rows + f


def test_indices_and_series(fetched):
    import rasterio

    doc, ch, counts, _tmp = fetched
    assert counts["read"] == 5 and counts["cloudy"] == 1 and counts["computed"]["dates"] == 5
    assert [d["id"] for d in project.dates(ch)] == ["20230410_S2B", "20230614_S2B", "20230709_S2B", "20230803_S2B",
                                                     "20230912_S2B"]
    rows = {r["id"]: r for r in project.scene_rows(ch)}
    assert rows["20230520_S2B"]["status"] == "cloudy" and rows["20230520_S2B"]["clear"] == 0.0
    assert rows["20230709_S2B"]["status"] == "cached" and 0.5 < rows["20230709_S2B"]["clear"] < 0.95
    # the NDVI of 14 June (two tiles) is the truth of the fields, cell by cell
    values, ids, tr, crs = project.read_stack(ch, "NDVI", ["20230614_S2B"])
    xs, ys = _centres(tr, values.shape[1:])
    x10, y10 = s2synth.centres(10.0)
    truth_tile = s2synth.ndvi_truth("2023-06-14", x10, y10)
    r0, c0 = int((s2synth.TILE_Y1 - tr.f) / 10), int((tr.c - s2synth.TILE_X0) / 10)
    truth = truth_tile[r0:r0 + values.shape[1], c0:c0 + values.shape[2]]
    assert np.isfinite(values[0]).all() and np.abs(values[0] - truth).max() < 2e-3
    # 9 July: the cloud and the shadow (and 20 m around them) are empty
    july, _ids, _tr, _crs = project.read_stack(ch, "NDVI", ["20230709_S2B"])
    for spot in (s2synth.CLOUD_SPOT, s2synth.SHADOW_SPOT):
        near = np.hypot(xs - spot[0], ys - spot[1]) < spot[2]
        assert np.isnan(july[0][near]).all()
    assert np.isfinite(july[0]).mean() > 0.8
    # every date at its own level, whatever the offset of its numbers (old, harmonised, mislabelled)
    ndvi_median = {r["acquisition"]: r["median"] for r in project.series(ch) if r["index"] == "NDVI"}
    for acq, level in (("20230410_S2B", 0.45), ("20230803_S2B", 0.55), ("20230912_S2B", 0.40)):
        assert abs(ndvi_median[acq] - level) < 0.02, (acq, ndvi_median[acq])
    with rasterio.open(project.folder(ch) / "scenes" / "20230912_S2B.tif") as src:
        assert src.tags()["OFFSET_CORRECTED"] == "YES" and src.offsets[0] == 0.0
    # NDRE and NDMI from the 20 m bands
    ndre, _i, _t, _c = project.read_stack(ch, "NDRE", ["20230614_S2B"])
    assert abs(np.nanmedian(ndre) - 0.6 * 0.72) < 0.02
    # the stacks: a band per date, named, with the notice
    with rasterio.open(project.stack_path(ch, "NDMI")) as src:
        assert src.count == 5 and list(src.descriptions)[1] == "20230614_S2B" and src.crs.to_epsg() == 32632
        assert src.tags()["ATTRIBUTION"] == NOTICE and src.tags()["FORMULA"] == "(B8A - B11) / (B8A + B11)"
    # the series over the area per date: the median of the truth inside the area
    series = project.series(ch)
    assert len(series) == 15 and {r["index"] for r in series} == {"NDVI", "NDRE", "NDMI"}
    mask = area_mod.area_mask(project.area_of(ch), project.grid_of_chunk(ch))
    june = next(r for r in series if r["acquisition"] == "20230614_S2B" and r["index"] == "NDVI")
    assert june["cells"] == mask.sum() and abs(june["median"] - np.median(truth[mask])) < 2e-3
    assert june["p10"] < june["p25"] < june["median"] < june["p75"] < june["p90"]
    july_row = next(r for r in series if r["acquisition"] == "20230709_S2B" and r["index"] == "NDVI")
    assert july_row["cells"] < mask.sum() and 0.5 < july_row["clear_fraction"] < 0.95
    monthly = project.series(ch, monthly=True)
    assert [r["month"] for r in monthly if r["index"] == "NDVI"] == ["2023-04", "2023-06", "2023-07", "2023-08", "2023-09"]
    # the notice heads the CSVs and is written into the chart
    base = project.folder(ch)
    assert (base / "series.csv").read_text(encoding="utf-8").startswith(f"# {NOTICE}\n")
    assert (base / "series_monthly.csv").read_text(encoding="utf-8").startswith(f"# {NOTICE}\n")
    from PIL import Image

    with Image.open(base / "series.png") as im:
        assert im.info.get("Copyright") == NOTICE and im.size[0] > 600


def test_cropmarks_find_the_buried_wall(fetched):
    import rasterio
    from rasterio.warp import transform

    doc, ch, _counts, _tmp = fetched
    e = project.cropmarks(ch, "NDVI", "2023-summer", 50.0)
    assert e["id"] == "cropmarks_NDVI_2023_summer_50m" and e["dates"] == ["20230614_S2B", "20230709_S2B", "20230803_S2B"]
    assert e["attribution"] == NOTICE and e["name"] == "Crop mark NDVI estate 2023, sfondo 50 m"
    z, tr, crs = project.read_product(ch, e["id"])
    xs, ys = _centres(tr, z.shape)
    d = s2synth.wall_distance(xs, ys)
    x0, y0, x1, y1 = s2synth.SITE
    site = (xs > x0) & (xs < x1) & (ys > y0) & (ys < y1) & np.isfinite(z)
    on, off = site & (d <= 5), site & (d >= 40)
    assert np.median(z[on]) < -2.5 and abs(np.median(z[off])) < 0.5 and (np.abs(z[off]) > 2).mean() < 0.08
    # the cells under -2: most of them on the wall, a line across the site
    low = site & (z < -2)
    assert (d[low] <= 15).mean() > 0.6
    assert e["stats"]["low"] > 0.05
    # spring has no mark: nothing stands out along the wall
    spring = project.cropmarks(ch, "NDVI", "2023-spring", 50.0)
    zs, _t, _c = project.read_product(ch, spring["id"])
    assert abs(np.median(zs[on])) < 1.0
    # the copy on the chunk's grid (EPSG:3035, cells of the DEM's lattice): the wall where it is
    with rasterio.open(project.absolute(ch, e["chunk_file"])) as src:
        assert src.crs.to_string() == s2synth.CHUNK_CRS and abs(src.transform.a) <= 3.0
        assert src.tags()["ATTRIBUTION"] == NOTICE
        zc = src.read(1)
        cx, cy = _centres(src.transform, zc.shape)
        ux, uy = transform(s2synth.CHUNK_CRS, s2synth.UTM, cx.ravel(), cy.ravel())
        ux, uy = np.reshape(ux, cx.shape), np.reshape(uy, cy.shape)
        with rasterio.open(ch.dem_path) as dem:
            assert ((src.transform.c - dem.transform.c) / dem.transform.a) % 1 == 0  # on the DEM's lattice
    dc = s2synth.wall_distance(ux, uy)
    inside = (ux > x0) & (ux < x1) & (uy > y0) & (uy < y1) & np.isfinite(zc)
    assert np.median(zc[inside & (dc <= 4)]) < -2.5 and abs(np.median(zc[inside & (dc >= 40)])) < 0.5
    # the median background finds it too
    med = project.cropmarks(ch, "NDVI", "2023-summer", 50.0, "median")
    zm, _t, _c = project.read_product(ch, med["id"])
    assert med["id"].endswith("_median") and np.median(zm[on]) < -2.0


def test_composites_and_periods(fetched):
    doc, ch, _counts, _tmp = fetched
    assert project.periods(ch) == ["2023-spring", "2023-summer", "2023-autumn", "all"]
    assert project.period_ids(ch, "2023-06") == ["20230614_S2B"]
    assert project.period_ids(ch, "2023-06-01/2023-07-31") == ["20230614_S2B", "20230709_S2B"]
    assert project.period_ids(ch, "summer") == project.period_ids(ch, "2023-summer")
    assert project.period_ids(ch, "2023-winter") == [] and project.period_ids(ch, "20230803_S2B") == ["20230803_S2B"]
    with pytest.raises(ValueError):
        project.period_ids(ch, "midsummer")
    with pytest.raises(project.NoData):
        project.composite(ch, "NDVI", "2023-winter")
    e = project.composite(ch, "NDVI", "2023-summer")
    import rasterio

    with rasterio.open(project.absolute(ch, e["file"])) as src:
        med, count = src.read(1), src.read(2)
    assert list(np.unique(count[np.isfinite(med)])) in ([2, 3], [1, 2, 3])
    assert abs(e["stats"]["median"] - np.median([0.72, 0.65, 0.55])) < 0.03
    assert analysis.check_period("2023-04-01/2023-06-30") == "2023-04-01/2023-06-30"
    assert analysis.in_period(__import__("datetime").date(2022, 12, 20), "", "2023-winter")


def test_a_second_look_needs_no_network(fetched, monkeypatch):
    from ppg_sentinel2 import reader

    doc, ch, _counts, tmp = fetched
    shutil.rmtree(tmp / "cogs")  # the "COGs" are gone: everything must come from the cache
    counts = project.fetch(ch)
    assert counts["read"] == 0 and counts["failed"] == 0 and counts["cached"] == 5 and counts["skipped_cloudy"] == 1
    assert counts["computed"]["dates"] == 5
    # masking again with other classes and a lower threshold, still offline
    assert project.compute(ch)["dates"] == 5
    counts = project.fetch(ch, min_clear=0.9)
    assert counts["computed"]["dates"] == 4 and project.settings(ch)["min_clear"] == 0.9
    # another area: the cache belongs to the old grid, the scenes are read again (and fail here)
    doc.save()
    a = project.area_of(ch)
    a.margin_m = 200.0
    project.set_area(ch, a)
    assert {r["status"] for r in project.scene_rows(ch)} == {"found"}
    monkeypatch.setattr(reader, "WORKERS", 1)
    counts = project.fetch(ch, min_clear=0.5)
    assert counts["failed"] == 6 and counts["computed"]["dates"] == 0
    assert all("error" in r for r in project.scene_rows(ch))


def test_registry_in_the_project_and_exports(fetched, tmp_path):
    import rasterio

    import pyarchinit_photogrammetry as ppg

    doc, ch, _counts, _tmp = fetched
    e = project.cropmarks(ch, "NDRE", "2023-summer", 60.0)
    c = project.composite(ch, "NDVI", "2023-06-14")
    doc.save()
    back = ppg.Document.open(doc.path).chunk
    st = back.meta["sentinel2"]
    assert [p["id"] for p in st["products"]] == [e["id"], c["id"]]
    assert st["area"]["source"] == "dem" and st["grid"]["crs"] == "EPSG:32632" and len(st["scenes"]) == 6
    assert st["series"]["attribution"] == NOTICE and len(st["dates"]) == 5
    assert project.summary(back)["attribution"] == NOTICE
    # QGIS: the copies on the chunk's grid, crop marks diverging between -3 and 3
    rasters, vectors = project.qgis_entries(back)
    assert vectors == [] and [r["ramp"] for r in rasters] == ["diverging", "index"]
    assert rasters[0]["limits"] == [-3.0, 3.0] and rasters[0]["path"].endswith("_chunk.tif")
    # the exports
    files = project.export_all(back, tmp_path / "out")
    names = {p.name for p in files}
    assert {"sentinel2_stack.tif", "sentinel2_NDVI.tif", "sentinel2_series.csv", "sentinel2_series_monthly.csv",
            "sentinel2_series.png", f"{e['id']}.tif", f"{e['id']}_chunk.tif", "sentinel2_README.txt"} <= names
    with rasterio.open(tmp_path / "out" / "sentinel2_stack.tif") as src:
        assert src.count == 15 and src.descriptions[0] == "NDVI 2023-04-10 20230410_S2B"
        assert src.tags()["ATTRIBUTION"] == NOTICE
    assert NOTICE in (tmp_path / "out" / "sentinel2_README.txt").read_text(encoding="utf-8")
    # a product out: its files gone
    gone = project.remove(back, e["id"])
    assert len(gone) == 2 and not (project.folder(back) / "products" / f"{e['id']}.tif").exists()
    assert project.product(back, e["id"]) is None and project.remove(back, "nothing") == []


def test_a_local_chunk_searched_with_a_box(tmp_path):
    """A chunk without coordinates can still look at a box of longitudes and latitudes: the
    products stay in UTM, nothing goes on the chunk's grid."""
    import pyarchinit_photogrammetry as ppg

    items = s2synth.world(tmp_path / "cogs")
    doc = ppg.Document()
    ch = doc.add_chunk("Locale")
    doc.save(tmp_path / "locale.ppg")
    a = area_mod.area_of_bbox(ch, 9.0145, 45.1855, 9.0185, 45.1875, 300)
    project.search(ch, a, "2023-01-01", "2023-12-31", 30, s2synth.CATALOG, opener=s2synth.FakeCatalog(items))
    project.fetch(ch)
    e = project.cropmarks(ch, "NDVI", "2023-summer")
    assert e["chunk_file"] is None and project.absolute(ch, e["file"]).exists()
    assert project.qgis_entries(ch) == ([], [])
