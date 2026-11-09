"""The coverage predicted on the ground: photos per cell and GSD over a flat field and a slope, an oblique shot,
the low-coverage polygons, the rasters written, the refusals and the cancel."""
from __future__ import annotations

import math

import numpy as np
import pytest
import synth
from scipy import ndimage

from ppg_flightplan import coverage, design
from ppg_flightplan import plan as pl
from ppg_flightplan.camera import camera
from ppg_flightplan.dem import RasterDem
from ppg_flightplan.route import shot_points

CRS = "EPSG:32632"
MINI = camera("mini4pro")


def grid_plan(width=300.0, height=200.0, quota=60.0, mode="per_punto") -> dict:
    area = synth.area_latlon(width, height)
    return {"nome": "Prova", "camera": "mini4pro", "alta": True, "minutiBatteria": 18,
            "blocchi": [{"tipo": "griglia", "nome": "A", "area": [list(p) for p in area], "quota": quota,
                         "inclinazione": -90, "avanti": 80, "lato": 70, "orientamento": 0, "velocita": 5,
                         "modo": mode, "alta": True}]}


def interior(cov, metres: float) -> np.ndarray:
    """The area's cells farther than ``metres`` from its border."""
    return ndimage.binary_erosion(cov.inside, iterations=max(1, int(math.ceil(metres / cov.cell))))


def test_the_shots_are_those_of_the_route():
    piano = grid_plan(mode="intervallo")
    for b in piano["blocchi"]:
        route, _ = pl.flown_route(b, piano)
        mine = coverage.route_shots(route, True)
        theirs = shot_points(route)
        assert [(s.position, s.height) for s in mine] == theirs


def test_flat_nadir_grid():
    piano = grid_plan()
    cov = coverage.predict(piano, CRS)
    inner = interior(cov, MINI.footprint(60))
    counts = cov.counts[inner]
    assert 14 <= np.median(counts) <= 19                  # 5 x 3.33 = 16.7 in theory
    assert cov.stats["gsd_median_cm"] == pytest.approx(MINI.gsd_cm(60, True), rel=0.02)
    assert (counts < coverage.LOW).mean() == 0
    assert cov.surface == "flat" and cov.stats["occlusion"] is False
    assert cov.counts.dtype == np.int16 and cov.gsd_cm.dtype == np.float32


def test_slope_and_terrain_following(tmp_path):
    # the ground rises 30 m from the west to the east edge of the area (300 m wide): across the north-south strips
    def ramp(x, y):
        return 100.0 + np.clip((x - (synth.X0 - 150)) / 300.0, 0, 1) * 30.0

    dem_path = synth.write_dem(tmp_path / "rampa.tif", ramp)
    piano = grid_plan()
    piano["decollo"] = list(synth.area_latlon(300, 200)[0])          # the takeoff at the low (west) corner
    with RasterDem(dem_path) as dem:
        ground = dem.height(*piano["decollo"])
        assert ground == pytest.approx(100.0, abs=0.5)
        flat = coverage.predict(piano, CRS, dem=dem, ground_takeoff=ground)
        assert flat.surface == "dem"
        inner = interior(flat, MINI.footprint(60))
        cols = np.nonzero(inner.any(axis=0))[0]
        third = (cols.max() - cols.min()) // 3
        low_side = inner.copy()
        low_side[:, cols.min() + third:] = False
        high_side = inner.copy()
        high_side[:, :cols.max() - third] = False
        assert np.median(flat.counts[high_side]) < 0.75 * np.median(flat.counts[low_side])

        design.apply_terrain(piano, dem=dem)
        assert piano["blocchi"][0]["seguiTerreno"] and piano["quotaDecollo"] == pytest.approx(ground, abs=0.5)
        h, w = flat.counts.shape
        same = (flat.transform.c, flat.transform.f - h * flat.cell, flat.transform.c + w * flat.cell, flat.transform.f)
        follow = coverage.predict(piano, CRS, dem=dem, ground_takeoff=piano["quotaDecollo"], bounds=same,
                                  cell=flat.cell)
    a, b = np.median(follow.counts[low_side]), np.median(follow.counts[high_side])
    assert abs(a - b) <= 0.15 * max(a, b)
    nominal = MINI.gsd_cm(60, True)
    for side in (low_side, high_side):
        assert np.nanmedian(follow.gsd_cm[side]) == pytest.approx(nominal, rel=0.05)


def test_an_oblique_shot_looks_ahead():
    lat, lon = synth.area_latlon(1, 1)[0]
    piano = {"camera": "mini4pro", "alta": True, "blocchi": []}
    shot = coverage.Shot((lat, lon), 50.0, -45.0, None, ((lat, lon), (lat + 0.001, lon)), None, True, None)
    cov = coverage.predict(piano, CRS, ground_takeoff=0.0, shots=[shot], cell=1.0)
    seen = cov.counts > 0
    rows, cols = np.nonzero(seen)
    xs = cov.transform.c + (cols + 0.5) * cov.cell
    ys = cov.transform.f - (rows + 0.5) * cov.cell
    from pyproj import Transformer

    x0, y0 = Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(lon, lat)
    # the image's centre falls 50 m ahead (45 degrees from 50 m); the frame spans from the near edge, 45 degrees
    # less half the short side's field, to the far one, 45 degrees more: all of it ahead, none behind or aside
    half = math.degrees(math.atan(MINI.height_mm / (2 * MINI.focal_mm)))
    near, far = 50 * math.tan(math.radians(45 - half)), 50 * math.tan(math.radians(45 + half))
    assert (ys - y0).min() == pytest.approx(near, abs=2 * cov.cell)
    assert (ys - y0).max() == pytest.approx(far, abs=3 * cov.cell)
    assert abs(xs.mean() - x0) < 2.0
    col, row = (int(v) for v in ~cov.transform @ (x0, y0 + 50.0))
    assert seen[row, col]
    # its GSD there is the one of a distance of 50 x sqrt(2) along the optical axis
    expected = 50 * math.sqrt(2) * MINI.width_mm / MINI.pixels_for(True) / MINI.focal_mm * 100
    assert cov.gsd_cm[row, col] == pytest.approx(expected, rel=0.02)
    assert cov.shots == 1


def test_low_coverage_polygons():
    import shapely
    from pyproj import Transformer

    piano = grid_plan()
    cov = coverage.predict(piano, CRS)
    threshold = 12
    polygons = coverage.low_polygons(cov, threshold=threshold)
    assert polygons
    tr = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    area = shapely.Polygon([tr.transform(lon, lat) for lat, lon in piano["blocchi"][0]["area"]])
    inner = area.buffer(-MINI.footprint(60))
    for poly in polygons:
        assert poly.intersects(area.exterior.buffer(cov.cell * 1.5))     # they touch the border of the area
        assert poly.intersection(inner).area < cov.cell ** 2              # never inside the eroded interior


def test_written_rasters(tmp_path):
    import rasterio

    piano = grid_plan()
    cov = coverage.predict(piano, CRS)
    paths = coverage.write(cov, tmp_path / "out", "copertura")
    assert set(paths) == {"count", "gsd", "rgba"} and all(p.is_file() for p in paths.values())
    with rasterio.open(paths["count"]) as c, rasterio.open(paths["gsd"]) as g, rasterio.open(paths["rgba"]) as r:
        assert c.transform == g.transform == r.transform
        assert c.crs == g.crs == r.crs and c.crs.to_epsg() == 32632
        assert c.nodata == -1 and c.dtypes[0] == "int16"
        counts = c.read(1)
        assert (counts[~cov.inside] == -1).all() and (counts[cov.inside] >= 0).all()
        assert g.dtypes[0] == "float32" and np.isnan(g.read(1)[~cov.inside]).all()
        assert r.count == 4
        alpha = r.read(4)
        assert (alpha[~cov.inside] == 0).all() and (alpha[cov.inside] > 0).all()


def test_rgba_ramp():
    counts = np.array([[0, 3, 7, 12]], dtype=np.int16)
    out = coverage.rgba(counts, np.array([[True, True, True, False]]), low=5)
    assert tuple(out[:3, 0, 0]) == coverage.RAMP[0]
    assert tuple(out[:3, 0, 1]) == coverage.RAMP[1]
    assert tuple(out[:3, 0, 2]) == coverage.RAMP[2]
    assert tuple(out[:3, 0, 3]) == coverage.RAMP[3]
    assert list(out[3, 0]) == [coverage.ALPHA] * 3 + [0]


def test_geographic_crs_refused():
    with pytest.raises(coverage.CoverageError):
        coverage.predict(grid_plan(), "EPSG:4326")
    with pytest.raises(coverage.CoverageError):
        coverage.predict(grid_plan(), "LOCAL")


def test_progress_and_cancel():
    from pyarchinit_photogrammetry.core.progress import Cancelled

    calls = []

    def progress(fraction, text=""):
        calls.append(fraction)
        if len(calls) == 2:
            raise Cancelled()

    with pytest.raises(Cancelled):
        coverage.predict(grid_plan(), CRS, progress=progress)
    assert len(calls) == 2
