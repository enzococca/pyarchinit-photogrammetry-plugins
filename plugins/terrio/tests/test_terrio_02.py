"""The steps of 0.2.0 on the synthetic survey of test_terrio: cleaning (LOF, DBSCAN), colours and
values from rasters, heights between the ellipsoid and a geoid (a small geoid grid written by the
test, no network), reference rasters with their heights moved, the Planetary Computer searches
(the address only; the real catalogue with PPG_TEST_NETWORK=1), the terrio of the plugin folder."""
from __future__ import annotations

import logging
import os
import struct
import sys
import warnings

import numpy as np
import pytest

terrio = pytest.importorskip("terrio")
if tuple(int(v) for v in terrio.__version__.split(".")[:2]) < (0, 5):
    pytest.skip("the steps of 0.2.0 need terrio 0.5", allow_module_level=True)

from ppg_terrio import cleaning, cloudfile, colour, environment, heights, rasters, transforms  # noqa: E402

from test_terrio import CRS, LOGGER, ORIGIN, SIZE, _project  # noqa: E402

NETWORK = os.environ.get("PPG_TEST_NETWORK") == "1"


def _with_outliers(doc):
    """The dense cloud plus 12 isolated points in the air and a floating group of 40 points (a bird)."""
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    chunk = doc.chunk
    pc = chunk.point_cloud
    rng = np.random.default_rng(3)
    iso = np.column_stack([rng.uniform(2, SIZE - 2, 12), rng.uniform(2, SIZE - 2, 12), rng.uniform(6, 9, 12)])
    bird = np.column_stack([rng.normal(20, 0.05, 40), rng.normal(5, 0.05, 40), rng.normal(12, 0.05, 40)])
    n = len(pc)
    xyz = np.vstack([pc.xyz, iso, bird])
    rgb = np.vstack([pc.rgb, np.full((52, 3), 50, np.uint8)])
    chunk.point_cloud = PointCloud(xyz, rgb)
    return n, np.arange(n, n + 12), np.arange(n + 12, n + 52)


def test_cleaning_counts_first_then_classes_or_removes(tmp_path):
    from pyarchinit_photogrammetry.ai.classify import set_class
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    n, iso, bird = _with_outliers(doc)
    hand = np.zeros(n + 52, bool)
    hand[iso[0]] = True  # an isolated point labelled by hand: it stays
    set_class(chunk, hand, 64, manual=True)
    cloudfile.add_properties(chunk.product_path("point_cloud"), {"temperature": np.arange(n + 52, dtype=np.float32)})
    before = chunk.point_cloud.classification.copy()
    found = cleaning.analyse(chunk)
    assert np.array_equal(chunk.point_cloud.classification, before), "the analysis changes nothing"
    assert found["isolated"][iso[1:]].all() and not found["isolated"][iso[0]] and found["manual_kept"] >= 1
    assert found["detached"][bird].all() and found["n_detached"] == 40 and found["groups"] == [40]
    assert found["isolated"][:n].mean() < 0.01 and not found["detached"][:n].any(), "the survey stays"
    assert found["spacing"] > 0 and found["eps"] == pytest.approx(cleaning.EPS_FACTOR * found["spacing"])
    assert "Gruppi staccati" in cleaning.summary(found) and "40" in cleaning.summary(found)
    out = cleaning.apply(chunk, found)
    codes = chunk.point_cloud.classification
    assert (codes[iso[1:]] == 7).all() and codes[iso[0]] == 64 and (codes[bird] == 18).all()
    assert out["points_after"] == n + 52 and len(versions.versions(chunk, "point_cloud")) == 1
    assert cloudfile.read_extras(chunk.product_path("point_cloud"))["temperature"][-1] == n + 51, "values go along"
    # removed: the version keeps the cloud of before, the values follow the points kept
    again = cleaning.analyse(chunk, lof=False)  # the noise classes are left out: nothing detached any more
    assert again["n_detached"] == 0
    doc2 = _project(tmp_path / "b")
    _with_outliers(doc2)
    cloudfile.add_properties(doc2.chunk.product_path("point_cloud"),
                             {"temperature": np.arange(len(doc2.chunk.point_cloud), dtype=np.float32)})
    found, out = cleaning.clean(doc2.chunk, detached_action="remove")
    assert out["removed"] == 40 and len(doc2.chunk.point_cloud) == n + 12
    assert len(cloudfile.read_extras(doc2.chunk.product_path("point_cloud"))["temperature"]) == n + 12
    with pytest.raises(RuntimeError):
        cleaning.apply(doc2.chunk, found)  # the cloud changed after the analysis


def test_colours_from_the_orthomosaic_and_another_crs(tmp_path):
    import rasterio
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    with rasterio.open(chunk.orthomosaic_path) as s:
        rgba, transform = s.read(), s.transform
    pc = chunk.point_cloud
    world = chunk.to_world(pc.xyz)
    rows, cols = rasterio.transform.rowcol(transform, world[:, 0], world[:, 1])
    rows, cols = np.clip(rows, 0, rgba.shape[1] - 1), np.clip(cols, 0, rgba.shape[2] - 1)
    expected = rgba[:3, rows, cols].T
    out = colour.colorize(chunk, "orthomosaic")
    assert out["mode"] == "rgb" and out["with_value"] == len(pc)
    got = chunk.point_cloud.rgb
    inner = (world[:, 0] - ORIGIN[0] > 0.01) & (world[:, 1] - ORIGIN[1] > 0.01)  # not on the edge of a pixel
    assert (got[inner] == expected[inner]).mean() > 0.999
    assert len(versions.versions(chunk, "point_cloud")) == 1, "the colours of before are a version"
    # a raster in geographic coordinates (a gradient of colours by longitude and latitude): warped
    # into the chunk system tile by tile, onto a cloud without colours (a lidar)
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    lon, lat = terrio.transform_points(world[:, 0], world[:, 1], CRS, "EPSG:4326")[:2]
    x0, y1, step, n = lon.min() - 1e-4, lat.max() + 1e-4, 2e-6, 300
    cols_, rows_ = np.meshgrid(np.arange(n), np.arange(n))
    grad = np.stack([cols_ * 255 / (n - 1), rows_ * 255 / (n - 1), np.full((n, n), 100.0)]).round().astype(np.uint8)
    geo = tmp_path / "gradiente_wgs84.tif"
    terrio.Raster.from_array(grad, (x0, step, 0.0, y1, 0.0, -step), crs="EPSG:4326").write(str(geo))
    chunk.point_cloud = PointCloud(pc.xyz, None)
    out = colour.colorize(chunk, str(geo), mode="rgb", keep_version=False)
    assert out["with_value"] == len(pc)
    want = np.column_stack([(lon - x0) / step - 0.5, (y1 - lat) / step - 0.5]) * 255 / (n - 1)
    got = chunk.point_cloud.rgb.astype(float)
    assert np.median(np.abs(got[:, :2] - want)) < 1.5 and (got[:, 2] == 100).all()


def test_values_as_a_dimension_kept_and_exported(tmp_path):
    from ppg_terrio import clouds

    doc = _project(tmp_path)
    chunk = doc.chunk
    out = colour.colorize(chunk, "dem")
    assert out["mode"] == "value" and out["name"] == "height" and out["with_value"] == len(chunk.point_cloud)
    world = chunk.to_world(chunk.point_cloud.xyz)
    v = cloudfile.read_extras(chunk.product_path("point_cloud"))["height"]
    plane = ORIGIN[2] + 0.02 * (np.floor((world[:, 0] - ORIGIN[0]) / 0.5) * 0.5)
    assert np.abs(v - plane).max() < 0.011, "the DEM cell of each point"
    assert set(colour.values(chunk)) == {"height"}
    clouds.ground(chunk, "smrf")
    assert "height" in cloudfile.extra_names(chunk.product_path("point_cloud")), "the plugin's steps keep it"
    las = clouds.export_cloud(chunk, tmp_path / "n.laz")
    assert np.allclose(terrio.read_points(str(las)).to_dict()["height"], v, atol=1e-3)
    info = colour.preview_values(chunk, "height", tmp_path / "p.ply", "viridis")
    assert (tmp_path / "p.ply").exists() and info["limits"][1] > info["limits"][0]
    # a value outside the raster is NaN; the program writing the cloud again loses the dimension
    small = tmp_path / "small.tif"
    terrio.Raster.from_array(np.full((10, 10), 5.0, np.float32), (ORIGIN[0], 1.0, 0.0, ORIGIN[1] + 10, 0.0, -1.0),
                             crs=CRS, nodata=-9999.0).write(str(small))
    out = colour.colorize(chunk, str(small), mode="value", name="Prova (°C)")
    assert out["name"] == "prova_c" and 0 < out["with_value"] < len(chunk.point_cloud)
    vals = cloudfile.read_extras(chunk.product_path("point_cloud"))["prova_c"]
    assert np.isnan(vals).sum() == len(vals) - out["with_value"] and np.nanmax(vals) == 5.0
    colour.remove_value(chunk, "prova_c")
    assert set(colour.values(chunk)) == {"height"}
    chunk.point_cloud = chunk.point_cloud  # the program writes the cloud
    assert colour.values(chunk) == {} and "height" not in (chunk.meta.get("terrio_values") or {})
    with pytest.raises(ValueError):
        cloudfile.dimension_name("classification")


def _small_raster(tmp_path):
    """A raster of 50 x 50 cells of 0.5 m, all 7, in the system of the synthetic survey: its file, its bounds."""
    path = tmp_path / "piccolo.tif"
    terrio.Raster.from_array(np.full((3, 50, 50), 7, np.uint8), (1000.0, 0.5, 0.0, 2050.0, 0.0, -0.5),
                             crs=CRS).write(str(path))
    return path, (1000.0, 2025.0, 1025.0, 2050.0)


def test_a_tile_that_reaches_the_raster_by_less_than_a_cell_is_left_out_by_its_geometry(tmp_path, monkeypatch):
    """The window of a tile overlaps the edge of the raster by a hair: terrio refuses such a window as empty (in
    words that never held "outside"); the points of that tile are beyond the raster, they get no value, and the
    tiles that cover the raster are read."""
    from types import SimpleNamespace

    path, (left, bottom, right, top) = _small_raster(tmp_path)
    monkeypatch.setattr(colour, "TILE", 20)  # tiles of 10 m, windows padded by 1 m (two cells)
    origin = left - 11 + 1e-7  # the padded window of the first tile ends 1e-7 m inside the raster
    world = np.array([[origin, 2030.0, 0.0], [1010.0, 2030.0, 0.0], [1020.0, 2040.0, 0.0]])
    out = colour.sample(SimpleNamespace(crs=CRS, data_dir=tmp_path), str(path), (1, 2, 3), world)
    assert np.isnan(out[0]).all() and (out[1:] == 7).all(), out


def test_a_tile_beside_the_edge_of_a_raster_of_tall_cells_is_read(tmp_path, monkeypatch):
    """Cells taller than they are wide: a tile whose points are inside the raster, its window crossing the edge, is
    read (it was left out when the window was padded by the width of a cell and compared with its height)."""
    from types import SimpleNamespace

    path = tmp_path / "alto.tif"
    terrio.Raster.from_array(np.full((3, 20, 100), 7, np.uint8), (1000.0, 0.1, 0.0, 2010.0, 0.0, -0.5),
                             crs=CRS).write(str(path))  # 10 m by 10 m, cells 0.1 m wide and 0.5 m tall
    monkeypatch.setattr(colour, "TILE", 20)  # tiles of 2 m (twenty cells of the width)
    # the tile of the first point starts at 2009.75: its window, padded by two widths, reaches 0.45 m into the
    # raster - less than a height of a cell - while the point is 0.2 m inside the top edge
    world = np.array([[1005.0, 2009.8, 0.0], [1005.0, 2001.75, 0.0]])
    out = colour.sample(SimpleNamespace(crs=CRS, data_dir=tmp_path), str(path), (1, 2, 3), world)
    assert (out == 7).all(), out


@pytest.mark.parametrize("text", ["point outside the velocity grid", "no pixel of the raster could be read"])
def test_an_error_while_reading_a_tile_stops_the_step_whatever_its_words(tmp_path, monkeypatch, text):
    """Only the geometry leaves a tile out: an error of terrio while a tile inside the raster is read stops the step
    and is said, even when its text holds "outside" or "no pixel" (it was taken for a tile outside the raster, and
    its points silently got no value)."""
    from types import SimpleNamespace

    path, _ = _small_raster(tmp_path)

    def refuses(*a, **k):
        raise terrio.errors.InvalidDataError(text)

    monkeypatch.setattr(terrio, "translate", refuses)
    with pytest.raises(terrio.errors.InvalidDataError, match=text):
        colour.sample(SimpleNamespace(crs=CRS, data_dir=tmp_path), str(path), (1, 2, 3), np.array([[1010.0, 2030.0, 0.0]]))


def _memory():
    """What terrio raises when a limit of memory is passed: its ResourceLimitError (0.7 and later, a MemoryError),
    a MemoryError before."""
    limit = getattr(terrio.errors, "ResourceLimitError", MemoryError)
    return limit("the raster needs 12 GiB, over the limit of 4 GiB")


def test_a_refusal_for_memory_is_never_taken_for_a_file_that_cannot_be_read(tmp_path, monkeypatch):
    """The three places that pass over a file they cannot read (the bands of the reflectance, the system of a
    reference raster, the comparison of two systems) let a refusal for memory through: it is said as what it is,
    not taken for "not readable" (no bands, no system, not the same system)."""
    from types import SimpleNamespace

    import pyarchinit_photogrammetry.core.crs as crs_mod
    import rasterio
    from pyarchinit_photogrammetry.core import versions

    from ppg_terrio import clouds

    refusal = _memory()

    def refuses(*a, **k):
        raise refusal

    chunk = SimpleNamespace(orthomosaic_path=None, dem_path=None, data_dir=tmp_path,
                            meta={"multispectral": {"reflectance": "riflettanza.tif"}})
    monkeypatch.setattr(versions, "versions", lambda chunk, kind: [])
    monkeypatch.setattr(rasterio, "open", refuses)
    with pytest.raises(MemoryError, match="over the limit"):
        colour.chunk_rasters(chunk)
    with pytest.raises(MemoryError, match="over the limit"):
        rasters._source_crs(SimpleNamespace(open_raster=refuses), tmp_path / "riferimento.tif")
    monkeypatch.setattr(crs_mod, "to_pyproj", refuses)
    with pytest.raises(MemoryError, match="over the limit"):
        clouds._same_crs("EPSG:32633", "EPSG:6708")
    # what is not a refusal for memory is passed over as before
    broken = OSError("not a raster")

    def unreadable(*a, **k):
        raise broken

    monkeypatch.setattr(rasterio, "open", unreadable)
    monkeypatch.setattr(crs_mod, "to_pyproj", unreadable)
    assert colour.chunk_rasters(chunk) == []
    assert rasters._source_crs(SimpleNamespace(open_raster=unreadable), tmp_path / "riferimento.tif") is None
    assert clouds._same_crs("EPSG:32633", "EPSG:6708") is False


def _geoid_grid(folder, name="egm08_25.gtx", base=20.0):
    """A small GTX geoid around the synthetic survey: N = base + 0.4 per row + 0.1 per column of
    0.25 degrees; returns the undulation at the origin of the survey."""
    lon, lat = terrio.transform_points([ORIGIN[0]], [ORIGIN[1]], CRS, "EPSG:4326")[:2]
    lat0, lon0, step, n = float(np.floor(lat[0])) - 1, float(np.floor(lon[0])) - 1, 0.25, 17
    vals = np.fromfunction(lambda r, c: base + r * 0.4 + c * 0.1, (n, n)).astype(">f4")
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "wb") as f:
        f.write(struct.pack(">ddddii", lat0, lon0, step, step, n, n))
        f.write(vals.tobytes())
    return base + (lat[0] - lat0) / step * 0.4 + (lon[0] - lon0) / step * 0.1


def test_heights_of_the_dem_and_the_cloud(tmp_path):
    import rasterio
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    grids = tmp_path / "griglie"
    undulation = _geoid_grid(grids)
    opts = {"network": False, "folders": [grids], "cache": tmp_path / "cache", "environment_paths": False}
    out = heights.convert_dem(chunk, "ellipsoid", "EPSG:3855", **opts)
    with rasterio.open(chunk.dem_path) as a, rasterio.open(out["path"]) as b:
        dz = b.read(1) - a.read(1)
        assert b.crs == a.crs and b.transform == a.transform
    assert np.abs(dz + undulation).max() < 0.01, "orthometric = ellipsoidal - N"
    assert out["active"] is False and versions.versions(chunk, "dem")[-1]["params"]["heights"] == "EPSG:3855"
    assert "EGM2008" in out["name"] and chunk.dem_path.parent == chunk.data_dir, "the active DEM is untouched"
    # and back, from the version, made active
    back = heights.convert_dem(chunk, "EPSG:3855", "ellipsoid", version=out["version"], activate=True, **opts)
    with rasterio.open(chunk.dem_path) as a, rasterio.open(versions.version_file(chunk, "dem", out["version"])) as b:
        assert np.abs(a.read(1) - b.read(1) - undulation).max() < 0.01
    assert back["active"]
    # the cloud, with its values
    colour.colorize(chunk, "dem")
    z0 = chunk.to_world(chunk.point_cloud.xyz)[:, 2]
    out = heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", activate=True, **opts)
    z1 = chunk.to_world(chunk.point_cloud.xyz)[:, 2]
    assert np.abs(z1 - z0 + undulation).max() < 0.01 and len(versions.versions(chunk, "point_cloud")) == 1
    assert "height" in cloudfile.extra_names(chunk.product_path("point_cloud"))
    assert out["not_transformed"] == 0 and out["points"] == len(z0), "the grid has every point: none left out"
    assert chunk.meta["dense"]["heights"] == "EPSG:3855" and "not_transformed" not in chunk.meta["dense"]
    # a geoid of a file (a national geoid): its name and folder are enough
    own = _geoid_grid(tmp_path / "con spazi", "nazionale.gtx", base=40.0)
    out = heights.convert_dem(chunk, "ellipsoid", str(tmp_path / "con spazi" / "nazionale.gtx"), **opts)
    assert out["shift"][1] == pytest.approx(-own, abs=0.05) and "nazionale.gtx" in out["name"]


def test_heights_never_copied_silently(tmp_path):
    doc = _project(tmp_path)
    opts = {"network": False, "cache": tmp_path / "cache", "environment_paths": False}
    with pytest.raises(heights.HeightsUnavailable):
        heights.convert_dem(doc.chunk, "ellipsoid", "EPSG:5773", **opts)
    assert not (doc.chunk.data_dir / "versions").exists()
    assert heights.normalise("3855") == heights.normalise("EPSG:4326+3855") == "EPSG:3855"
    assert heights.normalise("Ellissoide") == "ellipsoid" and heights.short_label("EPSG:5773") == "EGM96"
    with pytest.raises(FileNotFoundError):
        heights.normalise(str(tmp_path / "manca.gtx"))
    local = doc.add_chunk("locale")
    with pytest.raises(RuntimeError):
        heights.vertical_shift(local.crs, [0.0], [0.0], "ellipsoid", "EPSG:3855", **opts)


def _no_shift(tmp_path):
    """A shift from the ellipsoid to EGM96 at the origin of the survey, whose grid is nowhere, asked twice: the two
    refusals and what was warned of meanwhile (but for the objects the interpreter clears when it comes to them)."""
    refusals = []
    with warnings.catch_warnings(record=True) as let_out:
        warnings.simplefilter("always")
        for _ in range(2):
            with pytest.raises(heights.HeightsUnavailable) as exc:
                heights.vertical_shift(CRS, [ORIGIN[0]], [ORIGIN[1]], "ellipsoid", "EPSG:5773", network=False,
                                       cache=tmp_path / "cache", environment_paths=False)
            refusals.append(exc.value)
    return refusals, [str(w.message) for w in let_out if not issubclass(w.category, ResourceWarning)]


def test_a_refusal_of_heights_says_what_terrio_said_of_the_grid(tmp_path, monkeypatch):
    """What terrio says in a warning of a grid it did not read goes into the refusal of the heights, after the
    plugin's own sentence, the same every time; when terrio says nothing more than its error (a grid that is not
    there) the message is the one of always. Nothing is shown twice, nothing is swallowed."""
    refusals, let_out = _no_shift(tmp_path)
    plain = str(refusals[0])
    assert plain.endswith("non copre il rilievo.") and str(refusals[1]) == plain and not let_out
    assert isinstance(refusals[0].__cause__, transforms.TransformUnavailable)
    real = terrio.transform_points
    too_large = "grid file is not read: larger than 2147483648 bytes (/griglie/geoide.gtx)"  # terrio 0.9's words

    def refusing(xs, ys, src_crs, dst_crs, zs=None, only_best=False):
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        warnings.warn(too_large, stacklevel=1)
        raise ValueError("coordinate reference system: cannot transform (39.5, 36, 0)")

    monkeypatch.setattr(terrio, "transform_points", refusing)
    refusals, let_out = _no_shift(tmp_path)
    assert str(refusals[0]) == f"{plain} terrio: {too_large}" and str(refusals[1]) == str(refusals[0])
    assert isinstance(refusals[0].__cause__, transforms.TransformUnavailable) and not let_out

    def older(xs, ys, src_crs, dst_crs, zs=None):  # no only_best: the heights come back as they are, with a warning
        return real(xs, ys, src_crs, dst_crs) if zs is None else real(xs, ys, src_crs, dst_crs, zs=zs)

    monkeypatch.setattr(terrio, "transform_points", older)
    refusals, let_out = _no_shift(tmp_path)
    assert str(refusals[0]).startswith(f"{plain} terrio: geoid model did not apply (its grid is missing")
    assert str(refusals[1]) == str(refusals[0]) and refusals[0].__cause__ is None and not let_out

    def warning(xs, ys, src_crs, dst_crs, zs=None, only_best=False):  # the heights moved: its warning is for others
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        warnings.warn("grid read from the network", stacklevel=1)
        return xs, ys, np.asarray(zs, np.float64) - 30.0

    monkeypatch.setattr(terrio, "transform_points", warning)
    with pytest.warns(UserWarning, match="grid read from the network"):
        shift = heights.vertical_shift(CRS, [ORIGIN[0]], [ORIGIN[1]], "ellipsoid", "EPSG:5773", network=False,
                                       cache=tmp_path / "cache", environment_paths=False)
    assert shift[0] == -30.0


def _edge_grid(folder, east=30.0, north=None, name="egm08_25.gtx", base=20.0):
    """A GTX geoid like the one of :func:`_geoid_grid` that ends inside the synthetic survey: its eastern edge is the
    meridian through the point ``east`` metres east of the origin, half way up the survey (a little slanted on the
    survey: half a metre from bottom to top), and with ``north`` its northern edge is the parallel through the point
    that many metres north of the origin. Beyond its edges no grid has the points. Returns the undulation at the
    longitudes and latitudes given."""
    lon, lat = terrio.transform_points([ORIGIN[0] + east], [ORIGIN[1] + (SIZE / 2 if north is None else north)], CRS,
                                       "EPSG:4326")[:2]
    step, n = 0.25, 17
    lon0 = float(lon[0]) - (n - 1) * step
    lat0 = float(np.floor(lat[0])) - 1 if north is None else float(lat[0]) - (n - 1) * step
    vals = np.fromfunction(lambda r, c: base + r * 0.4 + c * 0.1, (n, n)).astype(">f4")
    folder.mkdir(parents=True, exist_ok=True)
    with open(folder / name, "wb") as f:
        f.write(struct.pack(">ddddii", lat0, lon0, step, step, n, n))
        f.write(vals.tobytes())
    return lambda lons, lats: base + (np.asarray(lats) - lat0) / step * 0.4 + (np.asarray(lons) - lon0) / step * 0.1


def _across_the_edge(doc, keep=None):
    """The dense cloud of the synthetic survey without its points from 25 to 35 metres east of the origin, where the
    edge of :func:`_edge_grid` runs (or with the points ``keep(x, y)`` says), from west to east, with what the
    program keeps along the points - colours, normals, classes, confidence, hand labels - and a dimension of the
    plugin that numbers them. Returns which points lie west of the edge."""
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud

    chunk = doc.chunk
    pc = chunk.point_cloud
    x, y = pc.xyz[:, 0], pc.xyz[:, 1]
    pick = np.nonzero((x <= 25.0) | (x >= 35.0) if keep is None else keep(x, y))[0]
    pick = pick[np.argsort(x[pick], kind="stable")]
    i = np.arange(len(pick))
    normals = np.column_stack([np.cos(i), np.sin(i), np.zeros(len(i))])
    chunk.point_cloud = PointCloud(pc.xyz[pick], pc.rgb[pick], normals, classification=(i % 5 + 2).astype(np.uint8),
                                   confidence=(i % 200 + 1).astype(np.uint8), manual=i % 3 == 0)
    cloudfile.add_properties(chunk.product_path("point_cloud"), {"number": i.astype(np.int32)})
    return pc.xyz[pick, 0] <= 25.0


def _edge_options(tmp_path) -> dict:
    return {"network": False, "folders": [tmp_path / "griglie"], "cache": tmp_path / "cache",
            "environment_paths": False}


def test_the_shift_at_the_points_no_grid_covers_is_no_number_when_asked_so(tmp_path, monkeypatch):
    """``vertical_shift`` with ``partial`` and a terrio that tells the points it cannot move: NaN at the points beyond
    the edge of the grid, the shift at the others. Without ``partial``, with an older terrio, and when no point is
    covered, the refusal of always, for all the points. What tells a point that was not moved is any of its
    coordinates without a number: a height that came back beside a position at infinity is not taken."""
    undulation = _edge_grid(tmp_path / "griglie")
    opts = _edge_options(tmp_path)
    xs = ORIGIN[0] + np.array([0.0, 20.0, 29.0, 31.0, 40.0])
    ys = ORIGIN[1] + np.array([0.0, 20.0, 20.0, 20.0, 40.0])
    beyond = np.array([False, False, False, True, True])
    lon, lat = terrio.transform_points(xs, ys, CRS, "EPSG:4326")[:2]
    with pytest.raises(heights.HeightsUnavailable, match="in 5 punti su 5"):
        heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", **opts)
    if transforms.has_errcheck(terrio):
        dz = heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", partial=True, **opts)
        assert (np.isnan(dz) == beyond).all() and np.abs(dz[~beyond] + undulation(lon, lat)[~beyond]).max() < 1e-3
    else:
        with pytest.raises(heights.HeightsUnavailable, match="in 5 punti su 5"):
            heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", partial=True, **opts)
    with pytest.raises(heights.HeightsUnavailable, match="in 2 punti su 2") as none:
        heights.vertical_shift(CRS, xs[beyond], ys[beyond], "ellipsoid", "EPSG:3855", partial=True, **opts)
    assert isinstance(none.value.__cause__, transforms.TransformUnavailable)
    whole = heights.vertical_shift(CRS, xs[~beyond], ys[~beyond], "ellipsoid", "EPSG:3855", partial=True, **opts)
    assert np.abs(whole + undulation(lon, lat)[~beyond]).max() < 1e-3
    real, edge = terrio.transform_points, float(lon[2] + lon[3]) / 2

    def in_part(xs, ys, src_crs, dst_crs, zs=None, only_best=False, errcheck=True):
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        x, y = np.array(xs, np.float64), np.array(ys, np.float64)
        out = x > edge
        if errcheck and out.any():
            raise ValueError("coordinate reference system: cannot transform")
        x[out], y[out] = np.inf, np.inf
        return x, y, np.asarray(zs, np.float64) - 30.0  # a height also where the position is at infinity

    monkeypatch.setattr(terrio, "transform_points", in_part)
    dz = heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", partial=True, **opts)
    assert (np.isnan(dz) == beyond).all() and (dz[~beyond] == -30.0).all()

    def as_they_were(xs, ys, src_crs, dst_crs, zs=None, only_best=False, errcheck=True):
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        x, y = np.array(xs, np.float64), np.array(ys, np.float64)
        z = np.asarray(zs, np.float64) - 30.0
        z[x > edge] = np.asarray(zs, np.float64)[x > edge]  # a position with numbers, the height not moved
        return x, y, z

    monkeypatch.setattr(terrio, "transform_points", as_they_were)
    dz = heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", partial=True, **opts)
    assert (np.isnan(dz) == beyond).all() and (dz[~beyond] == -30.0).all(), "a height not moved is never copied"


def test_a_cloud_across_the_edge_of_a_geoid_leaves_out_and_counts_the_points_beyond_it(tmp_path, monkeypatch, caplog):
    """A cloud that crosses the edge of the grid of the geoid: with a terrio that tells the points it cannot move
    (0.9, ``errcheck``) the new version has the points the grid covers, each with its colour, normal, class,
    confidence, hand label and values, and how many were left out is in the log, in the parameters of the version and
    in the summary the window and the command line show; no point without numbers is written. An older terrio refuses
    the whole cloud, as before."""
    from ppg_terrio import summary_heights
    from ppg_terrio.texts import num
    from pyarchinit_photogrammetry.core import versions
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud
    from pyarchinit_photogrammetry.gui import i18n

    doc = _project(tmp_path)
    chunk = doc.chunk
    west = _across_the_edge(doc)
    undulation = _edge_grid(tmp_path / "griglie")
    before = chunk.point_cloud
    world = chunk.to_world(before.xyz)
    total, beyond = len(west), int((~west).sum())
    assert 500 < beyond < total - 500
    monkeypatch.setattr(heights, "BLOCK", 500)  # the points beside the edge asked in several blocks
    if not transforms.has_errcheck(terrio):
        with pytest.raises(heights.HeightsUnavailable, match="in 9 punti su 9"):
            heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
        assert not (chunk.data_dir / "versions").exists() and not versions.versions(chunk, "point_cloud")
        return
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        out = heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
    assert out["points"] == total - beyond and out["not_transformed"] == beyond and not out["active"]
    made = PointCloud.load(out["path"])
    assert len(made) == total - beyond and np.isfinite(made.xyz).all()
    lon, lat = terrio.transform_points(world[west, 0], world[west, 1], CRS, "EPSG:4326")[:2]
    moved = chunk.to_world(made.xyz)
    assert np.abs(moved[:, :2] - world[west, :2]).max() < 1e-6, "the points the grid covers, where they were"
    assert np.abs(moved[:, 2] - world[west, 2] + undulation(lon, lat)).max() < 0.01, "orthometric = ellipsoidal - N"
    assert (made.rgb == before.rgb[west]).all() and np.allclose(made.normals, before.normals[west])
    assert (made.classification == before.classification[west]).all()
    assert (made.confidence == before.confidence[west]).all() and (made.manual == before.manual[west]).all()
    assert (cloudfile.read_extras(out["path"])["number"] == np.nonzero(west)[0]).all()
    params = versions.versions(chunk, "point_cloud")[-1]["params"]
    assert params["not_transformed"] == beyond and np.isfinite(params["height_shift"]).all()
    lo, mean, hi = out["shift"]
    assert -undulation(lon, lat).max() - 0.01 < lo <= mean <= hi < -undulation(lon, lat).min() + 0.01
    said = [r.getMessage() for r in caplog.records if r.name == LOGGER]
    assert len(said) == 1 and f"{beyond} points of {total} not transformed" in said[0]
    assert f"{num(beyond)} punti su {num(total)} non trasformati" in summary_heights(out)
    try:
        i18n.set_current("en")
        assert (f"{num(beyond)} points of {num(total)} not transformed (no grid of the geoid covers them)"
                in summary_heights(out))
    finally:
        i18n.set_current(None)
    assert len(chunk.point_cloud) == total and np.array_equal(chunk.point_cloud.xyz, before.xyz), "the active one stays"
    # made the active version: the cloud of before, with every point, is among the versions
    out = heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", activate=True, **_edge_options(tmp_path))
    assert out["active"] and len(chunk.point_cloud) == total - beyond
    assert (cloudfile.read_extras(chunk.product_path("point_cloud"))["number"] == np.nonzero(west)[0]).all()
    kept = [PointCloud.read_column(versions.version_file(chunk, "point_cloud", e["id"]), "classification")
            for e in versions.versions(chunk, "point_cloud")]
    assert sorted(len(k) for k in kept) == [total - beyond, total]
    assert chunk.meta["dense"]["not_transformed"] == beyond, "the parameters of the active cloud say it"
    # and back: the grid has every point of the cloud that is left, the count of before is not said again
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        back = heights.convert_cloud(chunk, "EPSG:3855", "ellipsoid", **_edge_options(tmp_path))
    assert back["points"] == total - beyond and back["not_transformed"] == 0
    assert not [r for r in caplog.records if r.name == LOGGER]
    assert "not_transformed" not in versions.versions(chunk, "point_cloud")[-1]["params"]
    assert "non trasformati" not in summary_heights(back)
    again = chunk.to_world(PointCloud.load(back["path"]).xyz)
    assert np.abs(again - world[west]).max() < 0.01, "there and back: the points the grid covers, where they were"


def test_what_terrio_said_of_a_grid_comes_with_the_count_of_the_points_left_out(tmp_path, monkeypatch, caplog):
    """Points left out while terrio refused a grid for them (a model of several grids, one of them not read): what it
    warned of meanwhile comes with their count, once, in the log and in the summary - not in the parameters of the
    version, which go where the project goes and may hold a path of the computer - and is not issued again. A warning
    of a conversion that left no point out is for others, as it was, and so are those heard before an error that is
    not of the heights. terrio's answers are made up here: the real one says nothing of a point beyond the edge of
    the one grid of the other tests."""
    from ppg_terrio import summary_heights
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    west = _across_the_edge(doc)
    total, beyond = len(west), int((~west).sum())
    real = terrio.transform_points
    edge = float(real([ORIGIN[0] + 30.0], [ORIGIN[1] + SIZE / 2], CRS, "EPSG:4326")[0][0])
    refused = "grid file is not read: outside the folders of grids (/griglie/est.gtx)"
    fetched = "grid read from the web"
    state = {"edge": edge, "calls": 0, "fail_at": 0}

    def two_grids(xs, ys, src_crs, dst_crs, zs=None, only_best=False, errcheck=True):
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        state["calls"] += 1
        if state["calls"] == state["fail_at"]:
            raise OSError("the grid cannot be read")
        x, y, z = np.array(xs, np.float64), np.array(ys, np.float64), np.asarray(zs, np.float64) - 30.0
        east = x > state["edge"]
        if east.any() or state["edge"] == np.inf:
            warnings.warn(refused if east.any() else fetched, stacklevel=1)
        if east.any() and errcheck:
            raise ValueError("coordinate reference system: cannot transform (39.5, 36, 0)")
        x[east], y[east], z[east] = np.inf, np.inf, np.inf
        return x, y, z

    def convert():
        with caplog.at_level(logging.WARNING, logger=LOGGER), warnings.catch_warnings(record=True) as let_out:
            warnings.simplefilter("always")
            try:
                out = heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
            except OSError as exc:
                out = exc
        said = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        caplog.clear()
        return out, said, sorted(str(w.message) for w in let_out if not issubclass(w.category, ResourceWarning))

    monkeypatch.setattr(terrio, "transform_points", two_grids)
    monkeypatch.setattr(heights, "BLOCK", 500)  # blocks of points all beyond the edge, and one with both
    out, said, let_out = convert()
    assert out["points"] == total - beyond and out["not_transformed"] == beyond and out["reason"] == refused
    assert out["shift"] == pytest.approx((-30.0, -30.0, -30.0)), "of the points written"
    assert len(said) == 1 and said[0].endswith(f"terrio: {refused}")
    assert summary_heights(out).splitlines()[2].endswith(f"(terrio: {refused})")
    params = versions.versions(chunk, "point_cloud")[-1]["params"]
    assert params["not_transformed"] == beyond and refused not in str(params)
    assert let_out == [], "said with the count: not a second time"
    monkeypatch.setattr(heights, "BLOCK", 1_000_000)  # the points beside the edge in one call, some moved and some not
    state["calls"] = 0
    out, said, let_out = convert()
    assert out["not_transformed"] == beyond and out["reason"] == refused and let_out == []
    assert len(said) == 1 and said[0].endswith(f"terrio: {refused}")
    calls = state["calls"]
    state.update(calls=0, fail_at=calls)  # the last call of terrio fails for a reason of its own
    out, said, let_out = convert()
    assert isinstance(out, OSError) and said == [] and let_out == [refused] * (calls - 1), "heard before: let out"
    assert len(versions.versions(chunk, "point_cloud")) == 2
    state.update(edge=np.inf, calls=0, fail_at=0)  # every point moved, and a warning with them
    out, said, let_out = convert()
    assert out["points"] == total and out["not_transformed"] == 0 and out["reason"] == "" and said == []
    assert let_out == [fetched] and "not_transformed" not in versions.versions(chunk, "point_cloud")[-1]["params"]
    assert "non trasformati" not in summary_heights(out)


def test_a_cloud_that_no_grid_covers_is_refused_as_before(tmp_path):
    """No point of the cloud can be moved - the grid of the geoid ends west of the survey, or it has a corner of the
    survey where the cloud has no point -: the refusal of always, the one a DEM over the same ground has, and nothing
    is written. With any terrio."""
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    _edge_grid(tmp_path / "griglie", east=-10.0)
    with pytest.raises(heights.HeightsUnavailable, match="in 9 punti su 9") as cloud:
        heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
    with pytest.raises(heights.HeightsUnavailable) as dem:
        heights.convert_dem(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
    assert str(cloud.value) == str(dem.value) and isinstance(cloud.value.__cause__, transforms.TransformUnavailable)
    # the grid has the south-west corner of the survey, the cloud has no point there
    _across_the_edge(doc, keep=lambda x, y: (x >= 35.0) | (y >= 25.0))
    _edge_grid(tmp_path / "griglie", east=30.0, north=20.0)
    with pytest.raises(heights.HeightsUnavailable, match="in 9 punti su 9") as cloud:
        heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
    with pytest.raises(heights.HeightsUnavailable) as dem:
        heights.convert_dem(chunk, "ellipsoid", "EPSG:3855", **_edge_options(tmp_path))
    assert str(cloud.value) == str(dem.value)
    assert not (chunk.data_dir / "versions").exists() and not versions.versions(chunk, "point_cloud")


def test_a_terrio_that_does_not_tell_its_points_refuses_a_cloud_across_a_grid_as_before(tmp_path, monkeypatch):
    """A terrio without ``errcheck`` does not tell the points it could not move: a height that came back as it was
    (a terrio before ``only_best``, which took a rougher operation there) refuses all the points, also when a partial
    result is asked for, and the cloud across the edge of a grid is refused whole, nothing written - as before. The
    answers of that terrio are made up here: beyond the edge the heights as they are."""
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    _edge_grid(tmp_path / "griglie")
    opts = _edge_options(tmp_path)
    xs = ORIGIN[0] + np.array([0.0, 20.0, 29.0, 31.0, 40.0])
    ys = ORIGIN[1] + np.array([0.0, 20.0, 20.0, 20.0, 40.0])
    real = terrio.transform_points
    lon = real(xs, ys, CRS, "EPSG:4326")[0]
    edge = float(lon[2] + lon[3]) / 2

    def older(xs, ys, src_crs, dst_crs, zs=None):  # neither only_best nor errcheck
        if zs is None:
            return real(xs, ys, src_crs, dst_crs)
        x, z = np.asarray(xs, np.float64), np.array(zs, np.float64)
        z[x <= edge] -= 30.0
        return x, np.asarray(ys, np.float64), z

    monkeypatch.setattr(terrio, "transform_points", older)
    assert not transforms.has_errcheck(terrio)
    for partial in (False, True):
        with pytest.raises(heights.HeightsUnavailable, match="in 2 punti su 5"):
            heights.vertical_shift(CRS, xs, ys, "ellipsoid", "EPSG:3855", partial=partial, **opts)
    with pytest.raises(heights.HeightsUnavailable, match="in 3 punti su 9"):
        heights.convert_cloud(chunk, "ellipsoid", "EPSG:3855", **opts)
    assert not (chunk.data_dir / "versions").exists() and not versions.versions(chunk, "point_cloud")


def test_the_window_and_the_command_line_say_how_many_points_the_heights_left_out(tmp_path, monkeypatch, capsys):
    """The count of the points left out of a conversion of heights, where the user reads it: the command line prints
    it and the window shows it as a warning; the project keeps it in the parameters of the version. With an older
    terrio both say the refusal of before."""
    import pyarchinit_photogrammetry as ppg
    from ppg_terrio import TerrioPlugin
    from ppg_terrio.texts import num
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    doc = _project(tmp_path)
    west = _across_the_edge(doc)
    doc.save()
    grids = tmp_path / "griglie"
    _edge_grid(grids)
    total, beyond = len(west), int((~west).sum())
    line = f"{num(beyond)} punti su {num(total)} non trasformati"
    monkeypatch.setattr(heights, "_environment_grid_paths", lambda: [])  # no grid of the computer the test runs on
    shown = []

    class Window:
        def run_task(self, title, job, done=None, refresh=True, save=True):
            result = job(lambda *a: None)
            if done is not None:
                done(result)

    monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
        lambda parent, title, fields, intro="": {f.name: f.default for f in fields} | {
            "product": "point_cloud", "network": False, "folder": str(grids)}))
    monkeypatch.setattr(dialogs, "message", lambda parent, kind, title, text, **k: shown.append((kind, text)))
    plugin = TerrioPlugin()
    plugin.activate(Context("terrio", tmp_path / "_data", window=Window()))
    command = ("heights", str(doc.path), "--product", "point_cloud", "--from", "ellipsoid", "--to", "EPSG:3855",
               "--no-network", "--grids", str(grids))
    previous, app.document = app.document, doc
    try:
        if not transforms.has_errcheck(terrio):
            assert plugin.command(*command) == 1 and "error: Nessun modello ha spostato" in capsys.readouterr().out
            with pytest.raises(heights.HeightsUnavailable):
                plugin.heights_dialog()
            assert not shown
            return
        assert plugin.command(*command) == 0
        assert line in capsys.readouterr().out
        saved = ppg.Document.open(doc.path).chunk.meta["versions"]["point_cloud"]
        assert len(saved) == 1 and saved[0]["params"]["not_transformed"] == beyond
        plugin.heights_dialog()
    finally:
        app.document = previous
    assert len(shown) == 1 and shown[0][0] == "warning" and line in shown[0][1]
    assert "non sono nella nuova versione" in shown[0][1]


def test_reference_raster_with_its_heights_moved(tmp_path):
    import rasterio

    doc = _project(tmp_path)
    chunk = doc.chunk
    grids = tmp_path / "griglie"
    undulation = _geoid_grid(grids)
    big = tmp_path / "copernicus.tif"
    terrio.Raster.from_array(np.full((200, 200), 50.0, np.float32), (ORIGIN[0] - 150, 2.0, 0.0, ORIGIN[1] + 250, 0.0, -2.0),
                             crs=CRS, nodata=-9999.0).write(str(big), driver="COG")
    report = rasters.reference_raster(chunk, big, margin=10.0, resolution=1.0, heights_from="EPSG:3855",
                                      heights_to="ellipsoid", grids={"network": False, "folders": [grids],
                                                                     "cache": tmp_path / "cache",
                                                                     "environment_paths": False})
    with rasterio.open(report["path"]) as s:
        assert np.abs(s.read(1) - 50.0 - undulation).max() < 0.01
    kept = [e for e in chunk.meta["versions"]["dem"] if e["id"] == report["version"]][0]
    assert kept["params"]["heights"] == "ellipsoid" and "ellissoidiche" in kept["name"]
    with pytest.raises(ValueError):
        rasters.reference_raster(chunk, big, product="orthomosaic", heights_from="EPSG:3855", heights_to="ellipsoid")


def test_planetary_computer_search():
    src = rasters.planetary_source((14.26, 41.13, 14.27, 41.14), "cop-dem-glo-30")
    assert src.startswith('STACIT:"https://planetarycomputer.microsoft.com/api/stac/v1/search?collections=cop-dem-glo-30')
    assert src.endswith(":asset=data,method=POST") and "bbox=14.260000%2C41.130000" in src
    s2 = rasters.planetary_source((14.26, 41.13, 14.27, 41.14), "sentinel-2-l2a", datetime_range="2024-06-01/2024-09-30")
    assert "asset=visual" in s2 and "max_items=1" in s2 and "eo%3Acloud_cover" in s2 and "sortby=-datetime" in s2
    with pytest.raises(ValueError):
        rasters.planetary_source((0, 0, 1, 1), "an-unknown-collection")
    assert rasters.planetary_source((0, 0, 1, 1), "esa-worldcover", asset="map").endswith("asset=map,method=POST")


@pytest.mark.skipif(not NETWORK, reason="the Planetary Computer and the PROJ CDN: PPG_TEST_NETWORK=1")
def test_copernicus_dem_from_the_planetary_computer_in_ellipsoidal_heights(tmp_path):
    import rasterio

    doc = _project(tmp_path)
    report = rasters.reference_raster(doc.chunk, collection="cop-dem-glo-30", margin=50.0, heights_to="ellipsoid",
                                      grids={"cache": tmp_path / "grids"})
    assert report["shift"] is not None and report["width"] > 0
    with rasterio.open(report["path"]) as s:
        assert np.isfinite(s.read(1)).any()


def test_the_terrio_of_the_plugin_folder_when_newer(tmp_path, monkeypatch):
    folder = tmp_path / "lib"
    (folder / "terrio-9.9.9.dist-info").mkdir(parents=True)
    monkeypatch.setattr(environment, "_LIB_DIR", folder)
    assert environment.folder_version(folder) == "9.9.9"
    saved, module = list(sys.path), sys.modules.pop("terrio")
    try:
        environment._prefer_newest()
        assert sys.path[0] == str(folder), "a newer terrio in the plugin folder comes first"
    finally:
        sys.path[:] = saved
        sys.modules["terrio"] = module
    monkeypatch.setattr(terrio, "__version__", "0.4.2")
    with pytest.raises(environment.TerrioTooOld) as exc:
        environment.load(environment.FEATURES_SINCE, "La pulizia")
    assert "La pulizia" in str(exc.value) and "0.4.2" in str(exc.value) and exc.value.wanted == "0.5"
    assert environment.available() == "0.4.2" and environment.available(environment.FEATURES_SINCE) == ""


def test_commands(tmp_path, capsys):
    from ppg_terrio import TerrioPlugin

    import pyarchinit_photogrammetry as ppg

    doc = _project(tmp_path)
    _with_outliers(doc)
    doc.save()
    plugin = TerrioPlugin()
    assert plugin.command("clean", str(doc.path), "--dry-run") == 0
    assert "Gruppi staccati" in capsys.readouterr().out
    assert ppg.Document.open(doc.path).chunk.meta.get("terrio_clean") is None, "a dry run changes nothing"
    assert plugin.command("clean", str(doc.path), "--remove") == 0
    assert "rimossi" in capsys.readouterr().out
    assert plugin.command("colour", str(doc.path), "--list") == 0
    out = capsys.readouterr().out
    assert "orthomosaic" in out and "dem" in out
    assert plugin.command("colour", str(doc.path), "dem", "--name", "quota") == 0
    assert plugin.command("values", str(doc.path)) == 0
    assert "quota" in capsys.readouterr().out
    grids = tmp_path / "griglie"
    _geoid_grid(grids)
    assert plugin.command("heights", str(doc.path), "--from", "ellipsoid", "--to", "EPSG:3855", "--no-network",
                          "--grids", str(grids)) == 0
    assert "EGM2008" in capsys.readouterr().out
    assert plugin.command("heights", str(doc.path), "--from", "ellipsoid", "--to", "EPSG:5773", "--no-network") == 1
    assert plugin.command("catalogue") == 0 and "cop-dem-glo-30" in capsys.readouterr().out
    assert plugin.command("version") == 0 and "grids:" in capsys.readouterr().out


def test_clean_dialog_counts_then_applies(tmp_path, monkeypatch):
    from ppg_terrio import TerrioPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    doc = _project(tmp_path)
    n, iso, bird = _with_outliers(doc)
    asked, shown = [], []

    class Window:
        def run_task(self, title, job, done=None, refresh=True, save=True):
            result = job(lambda *a: None)
            if done is not None:
                done(result)

    class Ctx(Context):
        def run(self, title, function, done=None):
            result = function(lambda *a: None)
            if done is not None:
                done(result)

        def show_model(self, kind=None, chunk=None, path=None, title=""):
            shown.append((kind, path))
            return True

    monkeypatch.setattr(dialogs.ParamDialog, "ask", staticmethod(
        lambda parent, title, fields, intro="": {f.name: f.default for f in fields} | {"detached": "remove"}))
    monkeypatch.setattr(dialogs, "message", lambda *a, **k: None)
    monkeypatch.setattr(dialogs, "choice", lambda parent, kind, title, text, options, **k: asked.append(text) or 0)
    plugin = TerrioPlugin()
    plugin.activate(Ctx("terrio", tmp_path / "_data", window=Window()))
    previous, app.document = app.document, doc
    try:
        plugin.clean_dialog()
    finally:
        app.document = previous
    assert asked and "40" in asked[0], "the counts are asked about first"
    assert shown[0][1] is not None and shown[-1] == ("point_cloud", None), "the preview, then the cloud again"
    assert not shown[0][1].exists(), "the preview file goes away"
    assert len(doc.chunk.point_cloud) == n + 12 and (doc.chunk.point_cloud.classification[n:] == 7).all()
