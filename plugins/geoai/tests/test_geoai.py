from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ppg_geoai import GeoaiPlugin, client, dataset, results


@pytest.fixture
def sample(tmp_path):
    raster = tmp_path / "ortho.tif"
    image = np.full((3, 64, 256), 50, np.uint8)
    image[:, 8:24, :] = np.array([220, 40, 30])[:, None, None]
    image[:, 40:56, :] = np.array([30, 220, 40])[:, None, None]
    # A nodata strip must never become training background or prediction proposals.
    image[:, :, :2] = 0
    with rasterio.open(raster, "w", driver="GTiff", width=256, height=64, count=3, dtype="uint8",
                       transform=from_origin(0, 64, 1, 1), crs="EPSG:32632", nodata=0) as out:
        out.write(image)
    units = [{"wkt": box(0, 40, 256, 56).wkt, "tipo": "USM", "attributes": {"material": "stone"}},
             {"wkt": box(0, 8, 256, 24).wkt, "tipo": "US", "attributes": {"material": "soil"}}]
    return raster, units


def prepare(sample, tmp_path, name="dataset"):
    raster, units = sample
    folder = tmp_path / name
    meta = dataset.prepare(raster, units, "material", {"stone": "wall", "soil": "ground"},
                           box(0, 0, 256, 64).wkt, folder, tile_size=32, kinds={"wall": "USM"})
    return folder, meta


def test_dataset_spatial_split_nodata_and_classes(sample, tmp_path):
    folder, meta = prepare(sample, tmp_path)
    train = [s for s in meta["samples"] if s["split"] == "train"]
    val = [s for s in meta["samples"] if s["split"] == "validation"]
    assert max(s["col"] for s in train) + 2 == min(s["col"] for s in val)
    for a in train:
        for b in val:
            assert not box(*a["bounds"]).intersects(box(*b["bounds"]))
    with np.load(folder / "train_00000_00000.npz") as tile:
        assert (tile["label"][:, :2] == 255).all()
        assert set(np.unique(tile["label"])) == {0, 2, 255}
    assert meta["classes"][2] == {"id": 2, "name": "wall", "tipo": "USM"}
    assert dataset.read_manifest(folder) == meta
    assert all(all(count > 0 for count in cs) for cs in meta["class_pixels"].values())


def test_prepare_reads_the_programs_saved_unit_drawings(sample, tmp_path):
    from shapely.geometry import mapping

    raster, units = sample
    drawings = [{**u, "geometry": mapping(__import__("shapely").wkt.loads(u["wkt"]))} for u in units]
    for unit in drawings:
        unit.pop("wkt")
    folder = tmp_path / "drawn-units"
    manifest = dataset.prepare(raster, drawings, "material", {"stone": "wall", "soil": "ground"},
                               box(0, 0, 256, 64).wkt, folder, tile_size=32, kinds={"wall": "USM"})
    assert manifest["field"] == "material" and len(manifest["samples"]) > 0


def test_unknown_outside_reviewed_area_and_unmapped_units(sample, tmp_path):
    raster, units = sample
    units.append({"wkt": box(10, 28, 240, 32).wkt, "attributes": {"material": "unknown"}})
    folder = tmp_path / "ds"
    dataset.prepare(raster, units, "material", {"stone": "wall", "soil": "ground"},
                    box(0, 4, 256, 60).wkt, folder, tile_size=32)
    with np.load(folder / "train_00000_00001.npz") as tile:
        assert (tile["label"][:4] == 255).all()
    with np.load(folder / "train_00001_00001.npz") as tile:
        assert (tile["label"][:4] == 255).all()


def test_conflicting_classes_are_ignored(sample, tmp_path):
    raster, units = sample
    units.append({"wkt": box(8, 42, 16, 50).wkt, "attributes": {"material": "soil"}})
    folder, _ = prepare((raster, units), tmp_path)
    with np.load(folder / "train_00000_00000.npz") as tile:
        assert (tile["label"][14:22, 8:16] == 255).all()


def test_small_area_and_missing_class_fail_without_ready_manifest(sample, tmp_path):
    raster, units = sample
    out = tmp_path / "small"
    with pytest.raises(ValueError, match="four tile"):
        dataset.prepare(raster, units, "material", {"stone": "wall"}, box(0, 0, 32, 64).wkt, out, tile_size=32)
    assert not (out / "dataset.json").exists()
    out = tmp_path / "missing"
    with pytest.raises(ValueError, match="lacks labelled"):
        dataset.prepare(raster, units, "material", {"stone": "wall", "absent": "missing"},
                        box(0, 0, 256, 64).wkt, out, tile_size=32)
    assert not (out / "dataset.json").exists()


def test_manifest_path_traversal_refused(sample, tmp_path):
    folder, meta = prepare(sample, tmp_path)
    meta["samples"][0]["file"] = "../outside.npz"
    (folder / "dataset.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="sample path"):
        dataset.read_manifest(folder)


def test_fake_train_apply_review_export_and_unit_attributes(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("PPG_GEOAI_FAKE", "1")
    folder, meta = prepare(sample, tmp_path)
    model = tmp_path / "model.ppgseg"
    trained = client.run({"op": "train", "dataset": str(folder), "output": str(model)}, python=sys.executable)
    assert trained["backend"] == "fake-colour-centroids"
    assert trained["metrics"]["mean_iou"] == pytest.approx(1)
    assert "torch" not in sys.modules
    chunk = SimpleNamespace(data_dir=tmp_path / "chunk", meta={}, crs="EPSG:32632")
    run = results.predict(chunk, sample[0], model, python=sys.executable)
    results.add(chunk, run)
    assert run["count"] == 2  # each strip remains joined across tile seams
    with rasterio.open(results.path(chunk, run, "mask")) as src:
        assert src.nodata == 255
        assert (src.read(1)[:, :2] == 255).all()
    geoms, attrs = results.unit_rows(chunk, run)
    assert {a["tipo"] for a in attrs} == {"US", "USM"}
    assert {a["geoai_class"] for a in attrs} == {"wall", "ground"}
    results.remove_proposals(chunk, run, {1})
    assert run["count"] == 1
    assert len(results.unit_rows(chunk, run)[0]) == 1
    assert len(geoms) == 2


def test_test_model_refused_in_real_mode(sample, tmp_path, monkeypatch):
    monkeypatch.setenv("PPG_GEOAI_FAKE", "1")
    folder, _ = prepare(sample, tmp_path)
    model = tmp_path / "test.ppgseg"
    client.run({"op": "train", "dataset": str(folder), "output": str(model)}, python=sys.executable)
    monkeypatch.delenv("PPG_GEOAI_FAKE")
    with pytest.raises(RuntimeError, match="test model"):
        client.run({"op": "predict", "model": str(model), "raster": str(sample[0]),
                    "output": str(tmp_path / "mask.tif")}, python=sys.executable)


def test_registration_without_loading_ml(tmp_path):
    from pyarchinit_photogrammetry.plugins.api import Context

    plugin = GeoaiPlugin()
    ctx = Context("geoai", tmp_path)
    plugin.activate(ctx)
    assert "geoai" in ctx.contributions.commands
    assert len(ctx.contributions.actions) == 2
    assert "torch" not in sys.modules and "geoai" not in sys.modules


def test_pane_preserves_class_mapping_and_resets_reviewed_area_on_chunk_change(sample, monkeypatch, qt_app):
    from ppg_geoai.pane import GeoaiPane

    app = qt_app
    chunk = SimpleNamespace(meta={"drawings": {"units": sample[1]}})
    active = [chunk]
    plugin = SimpleNamespace(context=SimpleNamespace(window=SimpleNamespace(runner=None)), chunk=lambda: active[0])
    pane = GeoaiPane(plugin)
    pane.refresh()
    pane.field.setCurrentText("material")
    assert pane.classes.rowCount() == 2
    pane.classes.item(0, 1).setText("edited class")
    pane.area = box(0, 0, 256, 64).wkt
    pane.refresh()
    assert pane.classes.item(0, 1).text() == "edited class"
    assert pane.area is not None
    active[0] = SimpleNamespace(meta={"drawings": {"units": sample[1]}})
    pane.refresh()
    assert pane.area is None
    pane.close()
    app.processEvents()
