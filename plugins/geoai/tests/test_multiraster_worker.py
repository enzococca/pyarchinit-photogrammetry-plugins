"""Optional bands survive isolated training, inference and review without RGB assumptions."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ppg_geoai import client, dataset, results


def inputs_for(tmp_path, *, rgb=True):
    image = np.full((64, 256), 10, np.float32)
    image[8:24] = 20
    image[40:56] = 30
    files = []
    for name, values in [("elevation", image), ("temperature", image + 270)]:
        file = tmp_path / f"{name}.tif"
        with rasterio.open(file, "w", driver="GTiff", width=256, height=64, count=1,
                           dtype="float32", crs="EPSG:32632", transform=from_origin(0, 64, 1, 1),
                           nodata=-9999) as dst:
            values = values.copy()
            values[:, :2] = -9999
            dst.write(values, 1)
        files.append({"name": name, "kind": "dem" if name == "elevation" else "thermal",
                      "path": str(file), "bands": [1], "resampling": "nearest",
                      "unit": "m" if name == "elevation" else "K"})
    if rgb:
        file = tmp_path / "colour.tif"
        with rasterio.open(file, "w", driver="GTiff", width=256, height=64, count=3,
                           dtype="uint8", crs="EPSG:32632", transform=from_origin(0, 64, 1, 1)) as dst:
            dst.write(np.stack([image, image * 2, image * 3]).astype(np.uint8))
        files.insert(0, {"name": "colour", "kind": "rgb", "path": str(file), "bands": [1, 2, 3],
                         "resampling": "nearest"})
    else:
        files = files[:1]
    return files


def numeric_dataset(tmp_path, *, rgb=True):
    inputs = inputs_for(tmp_path, rgb=rgb)
    units = [{"wkt": box(0, 40, 256, 56).wkt, "tipo": "USM"},
             {"wkt": box(0, 8, 256, 24).wkt, "tipo": "US"}]
    folder = tmp_path / "dataset"
    meta = dataset.prepare(inputs[0]["path"], units, "tipo", {"USM": "wall", "US": "ground"},
                           box(0, 0, 256, 64).wkt, folder, tile_size=32, inputs=inputs,
                           kinds={"wall": "USM"})
    return inputs, folder, meta


@pytest.mark.parametrize("rgb,channels", [(True, 5), (False, 1)])
def test_numeric_train_predict_and_proposals(tmp_path, monkeypatch, rgb, channels):
    monkeypatch.setenv("PPG_GEOAI_FAKE", "1")
    sources, folder, meta = numeric_dataset(tmp_path, rgb=rgb)
    model = tmp_path / "numeric.ppgseg"
    trained = client.run({"op": "train", "dataset": str(folder), "output": str(model)}, python=sys.executable)
    assert trained["channels"] == channels
    assert trained["inputs"] == meta["inputs"]
    assert trained["normalisation"] == meta["normalisation"]
    assert trained["encoder_weights"] is None
    assert trained["metrics"]["mean_iou"] == pytest.approx(1)
    chunk = SimpleNamespace(data_dir=tmp_path / "chunk", meta={}, crs="EPSG:32632")
    run = results.predict(chunk, sources[0]["path"], model, inputs=sources, python=sys.executable)
    assert run["count"] == 2
    assert run["channels"] == channels and run["inputs"] == meta["inputs"]
    assert run["normalisation"] == meta["normalisation"]
    assert {a["geoai_class"] for a in results.unit_rows(chunk, run)[1]} == {"wall", "ground"}
    with rasterio.open(results.path(chunk, run, "mask")) as src:
        assert (src.read(1)[:, :2] == dataset.IGNORE).all()
    assert "torch" not in sys.modules


def test_model_requires_new_compatible_sources_and_rejects_missing_input(tmp_path, monkeypatch):
    monkeypatch.setenv("PPG_GEOAI_FAKE", "1")
    sources, folder, _ = numeric_dataset(tmp_path)
    model = tmp_path / "model.ppgseg"
    trained = client.run({"op": "train", "dataset": str(folder), "output": str(model)}, python=sys.executable)
    request = {"op": "predict", "model": str(model), "raster": sources[0]["path"],
               "output": str(tmp_path / "out.tif")}
    with pytest.raises(RuntimeError, match="Choose the input"):
        client.run(request, python=sys.executable)
    with pytest.raises(RuntimeError, match="do not match"):
        client.run({**request, "inputs": sources[:-1]}, python=sys.executable)
    with pytest.raises(RuntimeError, match="do not match"):
        client.run({**request, "inputs": sources[::-1]}, python=sys.executable)
    # New survey paths are chosen explicitly; they do not enter the semantic channel schema.
    target = tmp_path / "new-survey"
    target.mkdir()
    new_sources = inputs_for(target)
    result = client.run({**request, "inputs": new_sources}, python=sys.executable)
    assert result["normalisation"] == trained["normalisation"]
    assert json.loads((model / "model.json").read_text()) == trained


def test_reference_must_match_unit_coordinates(tmp_path):
    sources = inputs_for(tmp_path)
    with pytest.raises(ValueError, match="chunk's CRS"):
        results.check_reference(SimpleNamespace(crs="EPSG:32633"), sources[0]["path"])
    with pytest.raises(ValueError, match="chunk's CRS"):
        results.check_reference(SimpleNamespace(crs="LOCAL"), sources[0]["path"])


def test_real_training_rejects_tiles_too_small_for_batchnorm(tmp_path, monkeypatch):
    monkeypatch.delenv("PPG_GEOAI_FAKE", raising=False)
    _, folder, _ = numeric_dataset(tmp_path, rgb=False)
    with pytest.raises(RuntimeError, match="at least 64 pixels"):
        client.run({"op": "train", "dataset": str(folder), "output": str(tmp_path / "model")},
                   python=sys.executable)
    assert not (tmp_path / "model").exists()

