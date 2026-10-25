"""Physical raster alignment, unknown pixels and reproducible channel statistics."""
from __future__ import annotations

import json

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from shapely.geometry import box

from ppg_geoai import dataset
from ppg_geoai.inputs import RasterStack, channel_count, normalise


def write(path, image, *, transform=None, crs="EPSG:32632", nodata=None, unit=None, scales=None, offsets=None):
    if image.ndim == 2:
        image = image[None]
    with rasterio.open(path, "w", driver="GTiff", width=image.shape[2], height=image.shape[1],
                       count=image.shape[0], dtype=image.dtype,
                       transform=transform or from_origin(0, 64, 1, 1), crs=crs, nodata=nodata) as out:
        out.write(image)
        if unit:
            for b in range(1, image.shape[0] + 1):
                out.set_band_unit(b, unit)
        if scales is not None:
            out.scales = scales
        if offsets is not None:
            out.offsets = offsets
    return path


def descriptor(path, name="elevation", kind="dem", bands=None, **extra):
    return {"name": name, "kind": kind, "path": str(path), "bands": bands or [1],
            "resampling": "nearest", **extra}


def drawings():
    return [{"wkt": box(0, 40, 256, 56).wkt, "tipo": "USM"}]


def prepare(path, target, sources):
    return dataset.prepare(path, drawings(), "tipo", {"USM": "wall"}, box(0, 0, 256, 64).wkt,
                           target, tile_size=32, inputs=sources)


def test_dem_only_keeps_physical_values_and_learns_stats_from_training_only(tmp_path):
    values = np.broadcast_to(np.arange(256, dtype=np.float32), (64, 256)).copy()
    values[:, 192:] += 10000  # validation distribution must not leak into normalisation
    values[0, 2] = np.nan
    raster = write(tmp_path / "elevation.tif", values, nodata=np.nan, unit="m")
    folder = tmp_path / "dataset"
    meta = prepare(raster, folder, [descriptor(raster)])
    assert meta["format"] == 2 and meta["channels"] == 1
    assert meta["inputs"][0]["units"] == ["m"]
    train = []
    for row in meta["samples"]:
        with np.load(folder / row["file"]) as tile:
            assert tile["image"].dtype == np.float32
            if row["split"] == "train":
                train.append(tile["image"][:, tile["label"] != dataset.IGNORE])
    train = np.concatenate(train, axis=1).astype(np.float64)
    assert meta["normalisation"]["mean"] == pytest.approx(train.mean(axis=1))
    assert meta["normalisation"]["std"] == pytest.approx(train.std(axis=1))
    assert meta["normalisation"]["mean"][0] < 100
    with np.load(folder / "train_00000_00000.npz") as tile:
        assert not tile["valid"][0, 2] and tile["label"][0, 2] == dataset.IGNORE
    assert dataset.read_manifest(folder) == meta


def test_combined_inputs_align_coarse_shifted_grids_and_apply_scale_offset(tmp_path):
    rgb = write(tmp_path / "rgb.tif", np.full((3, 64, 256), 90, np.uint8))
    dem = write(tmp_path / "dem.tif", np.full((32, 127), 12.25, np.float32),
                transform=from_origin(2, 64, 2, 2), nodata=-9999, unit="m")
    temperatures = np.full((64, 256), 2500, np.int16)
    temperatures[4, 4] = -9999
    thermal = write(tmp_path / "thermal.tif", temperatures, nodata=-9999,
                    scales=(0.01,), offsets=(-10,), unit="C")
    sources = [descriptor(rgb, "photo", "rgb", [1, 2, 3]), descriptor(dem),
               descriptor(thermal, "heat", "thermal")]
    with RasterStack(rgb, sources) as stack:
        image, valid = stack.read(Window(-2, -2, 12, 12))
        assert image.shape == (5, 12, 12) and stack.shape == (64, 256)
        assert not valid[:2].any() and not valid[:, :4].any()
        assert image[:, 2 + 5, 2 + 5] == pytest.approx([90, 90, 90, 12.25, 15])
        assert not valid[2 + 4, 2 + 4]
        assert (image[:, ~valid] == 0).all()
        schema = stack.schema
    with RasterStack(rgb, sources, schema=schema) as stack:
        assert stack.channels == 5
    folder = tmp_path / "combined"
    meta = prepare(rgb, folder, sources)
    assert channel_count(meta) == 5
    assert meta["normalisation"]["std"] == [1] * 5


def test_normalisation_reuses_training_stats_on_another_project(tmp_path):
    train_path = write(tmp_path / "train-dem.tif", np.full((64, 256), 10, np.float32), unit="m")
    meta = prepare(train_path, tmp_path / "dataset", [descriptor(train_path)])
    target = write(tmp_path / "another-dem.tif", np.full((64, 256), 12, np.float32), unit="m")
    with RasterStack(target, [descriptor(target)], schema=meta["inputs"]) as stack:
        image, valid = stack.read(Window(0, 0, 32, 32))
        transformed = normalise(image, meta)
        transformed[:, ~valid] = 0
        assert (transformed == 2).all()  # never centre anew on the new survey
    rgb = np.full((3, 4, 4), 255, np.uint8)
    assert (normalise(rgb, {"format": 1}) == 1).all()


def test_explicit_crs_and_local_coordinates_are_not_mixed(tmp_path):
    known = write(tmp_path / "known.tif", np.ones((64, 256), np.float32))
    local = write(tmp_path / "local.tif", np.ones((64, 256), np.float32), crs=None)
    with pytest.raises(ValueError, match="both have a CRS"):
        with RasterStack(known, [descriptor(local)]):
            pass
    with RasterStack(local, [descriptor(local)]) as stack:
        _, valid = stack.read(Window(0, 0, 32, 32))
        assert valid.all() and stack.crs is None


def test_source_internal_masks_and_reprojection_are_preserved(tmp_path):
    from rasterio.transform import from_bounds
    from rasterio.warp import transform_bounds

    transform = from_origin(500000, 5100064, 1, 1)
    ref = write(tmp_path / "reference.tif", np.ones((64, 256), np.float32), transform=transform)
    geographic_bounds = transform_bounds("EPSG:32632", "EPSG:4326", 500000, 5100000, 500256, 5100064)
    source = write(tmp_path / "geographic.tif", np.full((64, 256), 7, np.float32), crs="EPSG:4326",
                   transform=from_bounds(*geographic_bounds, 256, 64))
    with rasterio.open(source, "r+") as out:
        mask = np.full((64, 256), 255, np.uint8)
        mask[20:40, 40:60] = 0
        out.write_mask(mask)
    with RasterStack(ref, [descriptor(source)]) as stack:
        image, valid = stack.read(Window(0, 0, 256, 64))
        assert valid[10, 100] and image[0, 10, 100] == 7
        assert not valid[30, 50]


def test_multispectral_band_order_names_and_units_must_match_model(tmp_path):
    raster = write(tmp_path / "spectrum.tif", np.full((4, 64, 256), 1000, np.uint16), unit="reflectance")
    source = descriptor(raster, "red-nir", "multispectral", [3, 4])
    with RasterStack(raster, [source]) as stack:
        schema = stack.schema
    for changed in ({**source, "bands": [4, 3]}, {**source, "name": "different"},
                    {**source, "kind": "thermal"}, {**source, "resampling": "bilinear"}):
        with pytest.raises(ValueError, match="match the model"):
            with RasterStack(raster, [changed], schema=schema):
                pass
    with pytest.raises(ValueError, match="units disagree"):
        with RasterStack(raster, [{**source, "unit": "C"}]):
            pass


@pytest.mark.parametrize("change", [
    {"channels": 2}, {"normalisation": {"method": "standard", "mean": [float("nan")], "std": [1]}},
    {"normalisation": {"method": "standard", "mean": [1], "std": [0]}}, {"bands": ["elevation:2"]},
    {"samples": []}, {"samples": [{"file": "../outside.npz", "split": "train"}]},
])
def test_inconsistent_multiband_manifests_are_refused(tmp_path, change):
    raster = write(tmp_path / "dem.tif", np.ones((64, 256), np.float32))
    folder = tmp_path / "dataset"
    meta = prepare(raster, folder, [descriptor(raster)])
    meta.update(change)
    (folder / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        dataset.read_manifest(folder)


def test_invalid_source_bands_and_complex_values_refused(tmp_path):
    raster = write(tmp_path / "real.tif", np.ones((64, 256), np.float32))
    complex_raster = write(tmp_path / "complex.tif", np.ones((64, 256), np.complex64))
    for source in (descriptor(raster, bands=[0]), descriptor(raster, bands=[2]), descriptor(raster, bands=[1, 1]),
                   descriptor(complex_raster)):
        with pytest.raises(ValueError):
            with RasterStack(raster, [source]):
                pass
