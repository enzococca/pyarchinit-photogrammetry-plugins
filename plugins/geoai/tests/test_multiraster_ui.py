"""Optional raster selection, explicit remapping and CLI portability without loading ML."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from ppg_geoai import GeoaiPlugin, dataset, input_config, results


@pytest.fixture
def rasters(tmp_path):
    files = []
    for name, bands in [("ortho", 3), ("dem", 1), ("thermal", 1)]:
        path = tmp_path / f"{name}.tif"
        with rasterio.open(path, "w", driver="GTiff", width=8, height=8, count=bands,
                           dtype="float32", crs="EPSG:32632", transform=from_origin(0, 8, 1, 1)) as out:
            out.write(np.ones((bands, 8, 8), np.float32))
        files.append({"name": name, "label": name, "path": str(path)})
    return files


@pytest.fixture
def qt_app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def pane_for(chunk, rasters):
    from ppg_geoai.pane import GeoaiPane

    context = SimpleNamespace(window=SimpleNamespace(runner=None), rasters=lambda ch: rasters,
                              store_drawings=lambda: None)
    plugin = SimpleNamespace(context=context, chunk=lambda: chunk)
    pane = GeoaiPane(plugin)
    pane.refresh()
    return pane, plugin


def test_optional_inputs_preserve_rgb_default_and_reset_between_chunks(rasters, qt_app):
    chunk = SimpleNamespace(meta={}, orthomosaic_path=Path(rasters[0]["path"]))
    pane, plugin = pane_for(chunk, rasters)
    assert not pane.custom_inputs.isChecked()
    assert pane.selected_inputs() == (chunk.orthomosaic_path, None)
    pane.custom_inputs.setChecked(True)
    pane.input_sources = [{"name": "height", "kind": "dem", "path": rasters[1]["path"], "bands": [1]}]
    reference, selected = pane.selected_inputs()
    assert reference == rasters[1]["path"] and len(selected) == 1
    selected[0]["bands"].append(2)
    assert pane.input_sources[0]["bands"] == [1]
    plugin.chunk = lambda: SimpleNamespace(meta={}, orthomosaic_path=None)
    pane.refresh()
    assert not pane.custom_inputs.isChecked() and not pane.input_sources
    pane.close()


def test_model_source_dialog_requires_new_files_and_preserves_semantics(rasters, qt_app):
    from ppg_geoai.source_dialog import SourceDialog

    schema = [{"name": "height", "kind": "dem", "bands": [1], "resampling": "bilinear", "units": ["m"]}]
    dialog = SourceDialog(rasters, schema=schema)
    assert dialog.table.cellWidget(0, 2).path() == ""
    with pytest.raises(ValueError, match="height"):
        dialog.sources()
    dialog.table.cellWidget(0, 2).combo.setCurrentText(rasters[1]["path"])
    assert dialog.sources() == [{**schema[0], "path": rasters[1]["path"]}]
    assert not dialog.table.cellWidget(0, 1).isEnabled()
    dialog.close()


def test_source_order_can_change_reference_and_invalid_bands_are_rejected(rasters, qt_app):
    from ppg_geoai.source_dialog import SourceDialog

    sources = [{"name": "height", "kind": "dem", "path": rasters[1]["path"], "bands": [1]},
               {"name": "heat", "kind": "thermal", "path": rasters[2]["path"], "bands": [1], "unit": "K"}]
    dialog = SourceDialog(rasters, sources=sources)
    dialog.table.setCurrentCell(1, 0)
    dialog.move_source(-1)
    selected = dialog.sources()
    assert selected[0]["name"] == "heat" and selected[0]["unit"] == "K"
    dialog.table.item(0, 3).setText("2")
    with pytest.raises(ValueError):
        dialog.sources()
    dialog.close()


def test_prepare_dem_only_forwards_selected_sources(rasters, tmp_path, monkeypatch, qt_app):
    from PySide6.QtWidgets import QFileDialog, QMessageBox

    chunk = SimpleNamespace(meta={"drawings": {"units": [{"tipo": "USM"}]}}, orthomosaic_path=None)
    pane, plugin = pane_for(chunk, rasters)
    pane.custom_inputs.setChecked(True)
    pane.input_sources = [{"name": "height", "kind": "dem", "path": rasters[1]["path"], "bands": [1]}]
    pane.area = "POLYGON ((0 0,8 0,8 8,0 8,0 0))"
    captured = {}

    def prepare(reference, *args, **kwargs):
        captured.update(reference=reference, **kwargs)
        return {"samples": []}

    monkeypatch.setattr(dataset, "prepare", prepare)
    monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *a, **k: (str(tmp_path / "dataset"), ""))
    monkeypatch.setattr(QMessageBox, "information", lambda *a: None)
    plugin.context.run = lambda name, fn, done: done(fn(None))
    pane.prepare()
    assert captured["reference"] == rasters[1]["path"]
    assert captured["inputs"] == pane.input_sources
    pane.close()


def test_input_config_resolves_paths_from_json_and_rejects_empty_sources(tmp_path):
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({"inputs": [{"name": "height", "path": "dem.tif", "bands": [1]}]}))
    _, reference, sources = input_config(config)
    assert reference == sources[0]["path"] == str(tmp_path / "dem.tif")
    config.write_text('{"inputs": []}')
    with pytest.raises(ValueError, match="non-empty list"):
        input_config(config)


def test_cli_prepare_and_apply_dem_only_use_explicit_input_mapping(rasters, tmp_path, monkeypatch, capsys):
    import pyarchinit_photogrammetry as ppg

    chunk = SimpleNamespace(meta={}, orthomosaic_path=None)
    saved = []
    document = SimpleNamespace(chunk=chunk, save=lambda: saved.append(True))
    monkeypatch.setattr(ppg.Document, "open", lambda *args: document)
    capture = []

    def prepare(reference, *args, **kwargs):
        capture.append((reference, kwargs["inputs"]))
        return {"samples": []}

    monkeypatch.setattr(dataset, "prepare", prepare)
    config = tmp_path / "inputs.json"
    config.write_text(json.dumps({"field": "tipo", "mapping": {"USM": "wall"}, "reviewed_area": "unused",
                                 "inputs": [{"name": "height", "kind": "dem", "path": "dem.tif", "bands": [1]}]}))
    plugin = GeoaiPlugin()
    assert plugin.command("prepare", "project.ppg", config, tmp_path / "dataset") == 0
    assert capture[-1][0] == rasters[1]["path"]
    monkeypatch.setattr(dataset, "read_manifest", lambda *args: {"format": 2})
    monkeypatch.setattr(results, "predict", lambda chunk, reference, model, *args, **kwargs:
                        capture.append((reference, kwargs["inputs"])) or {"id": "test", "count": 0})
    assert plugin.command("apply", "project.ppg", "model.ppgseg") == 1
    assert "--inputs" in capsys.readouterr().out
    assert plugin.command("apply", "project.ppg", "model.ppgseg", "--inputs", config) == 0
    assert capture[-1][0] == rasters[1]["path"]
    assert capture[-1][1][0]["name"] == "height" and saved == [True]


def test_cli_legacy_apply_keeps_rgb_defaults(rasters, monkeypatch):
    import pyarchinit_photogrammetry as ppg

    chunk = SimpleNamespace(meta={}, orthomosaic_path=Path(rasters[0]["path"]))
    document = SimpleNamespace(chunk=chunk, save=lambda: None)
    monkeypatch.setattr(ppg.Document, "open", lambda *args: document)
    monkeypatch.setattr(dataset, "read_manifest", lambda *args: {"format": 1})
    captured = []
    monkeypatch.setattr(results, "predict", lambda chunk, reference, model, *args, **kwargs:
                        captured.append((reference, kwargs["inputs"])) or {"id": "test", "count": 0})
    assert GeoaiPlugin().command("apply", "project.ppg", "legacy.ppgseg") == 0
    assert captured == [(chunk.orthomosaic_path, None)]
