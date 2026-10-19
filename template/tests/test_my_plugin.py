"""The plugin found by its entry point (installed with ``pip install -e .``) and what it adds; its
functions on a small project."""
from __future__ import annotations

import numpy as np
import pytest

from ppg_my_plugin import dem_statistics, summary


def _project(tmp_path):
    import pyarchinit_photogrammetry as ppg
    import rasterio
    from rasterio.transform import from_origin

    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    doc.save(tmp_path / "scavo.ppg")
    chunk.crs = "EPSG:32637"
    z = np.linspace(100.0, 101.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    z[0, 0] = -9999  # no data
    path = chunk.data_dir / "dem.tif"
    with rasterio.open(path, "w", driver="GTiff", width=64, height=64, count=1, dtype="float32", crs="EPSG:32637",
                       transform=from_origin(300000, 3960000, 0.05, 0.05), nodata=-9999) as dst:
        dst.write(z, 1)
    chunk._set_product("dem", path)
    doc.save()
    return doc


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("my-plugin")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "my-plugin"] == ["Riepilogo del chunk",
                                                                              "Statistiche del DEM"]
    assert "chunk-summary" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "my-plugin"] == [".md"]


def test_summary_and_export(tmp_path):
    doc = _project(tmp_path)
    text = summary(doc.chunk)
    assert text.startswith(f"# {doc.chunk.label}") and "foto: 0" in text and "dem" in text
    from ppg_my_plugin import MyPlugin

    out = tmp_path / "riepilogo.md"
    MyPlugin().export(doc.chunk, out, lambda *a: None)
    assert out.read_text(encoding="utf-8") == text


def test_dem_statistics(tmp_path):
    doc = _project(tmp_path)
    seen = []
    stats = dem_statistics(doc.chunk.dem_path, lambda f, m: seen.append(f))
    assert stats["cells"] == 64 * 64 - 1 and seen[-1] == 1.0
    assert stats["min"] == pytest.approx(100.0, abs=1e-3) and stats["max"] == pytest.approx(101.0)


def test_command(tmp_path, capsys):
    from ppg_my_plugin import MyPlugin

    doc = _project(tmp_path)
    assert MyPlugin().command(str(doc.path)) == 0
    assert "allineate: 0" in capsys.readouterr().out
    assert MyPlugin().command(str(doc.path), "--chunk", "nessuno") == 2
