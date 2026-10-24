"""The steps of 0.2.0 on the synthetic survey of test_terrio: cleaning (LOF, DBSCAN), colours and
values from rasters, heights between the ellipsoid and a geoid (a small geoid grid written by the
test, no network), reference rasters with their heights moved, the Planetary Computer searches
(the address only; the real catalogue with PPG_TEST_NETWORK=1), the terrio of the plugin folder."""
from __future__ import annotations

import os
import struct
import sys

import numpy as np
import pytest

terrio = pytest.importorskip("terrio")
if tuple(int(v) for v in terrio.__version__.split(".")[:2]) < (0, 5):
    pytest.skip("the steps of 0.2.0 need terrio 0.5", allow_module_level=True)

from ppg_terrio import cleaning, cloudfile, colour, environment, heights, rasters  # noqa: E402

from test_terrio import CRS, ORIGIN, SIZE, _project  # noqa: E402

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
