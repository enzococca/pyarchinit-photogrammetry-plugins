"""The plugin in the program: what it adds, the command line on a small project, the pane and the
workspace in the window; with the real datasets when they are on this computer."""
from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import pytest

import samples


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("underwater")
    assert info is not None and info.active, info and info.error
    titles = [a.title for a in manager.actions if a.plugin == "underwater"]
    assert titles[0] == "Pannello Subacqueo" and "Valuta le foto" in titles and len(titles) == 7
    assert "underwater" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "underwater"] == [".csv"]
    manager.unload()


def _project(tmp_path, n=6):
    """Photos of a green-blue water on disk, aligned looking down at known depths."""
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.core.camera import Camera, Sensor

    folder = tmp_path / "foto"
    folder.mkdir()
    doc = ppg.Document()
    chunk = doc.add_chunk()
    doc.chunk = chunk
    chunk.sensors[0] = Sensor(0, "cam", 160, 120)
    rs = np.random.default_rng(0)
    rows = ["time,depth"]
    for k in range(n):
        img = samples.srgb(samples.underwater(samples.texture(120, 160, seed=k), np.full((120, 160), 4.0, np.float32)))
        name = f"{1700000000 + 10 * k}.jpg"  # the time in the name
        cv2.imwrite(str(folder / name), img)
        w = np.eye(4)
        w[:3, :3] = samples.looking_down(0.2 * k)
        w[:3, 3] = [rs.uniform(-5, 5), rs.uniform(-5, 5), -10.0 - 0.3 * k + 0.1 * rs.standard_normal()]
        chunk.cameras.append(Camera(key=k, label=name, path=str(folder / name), sensor_key=0, transform=w))
    for k in range(0, 10 * n, 2):
        # the log of the depths of the cameras (between photos: the line between them)
        rows.append(f"{1700000000 + k},{10.0 + 0.3 * k / 10:.4f}")
    (tmp_path / "depth.csv").write_text("\n".join(rows) + "\n")
    for k, cam in enumerate(chunk.cameras):  # the depth the camera was at
        cam.transform[2, 3] = -(10.0 + 0.3 * k)
    doc.save(tmp_path / "p.ppg")
    return doc, chunk


def test_the_command_line(tmp_path, capsys):
    from ppg_underwater import UnderwaterPlugin

    doc, chunk = _project(tmp_path)
    plugin = UnderwaterPlugin()
    project = str(doc.path)
    assert plugin.command("score", project) == 0
    assert "6 photos scored" in capsys.readouterr().out
    assert plugin.command("quick", project, "--use") == 0
    out = capsys.readouterr().out
    assert "6 photos corrected" in out and "'switched': 6" in out
    assert plugin.command("depth", project, str(tmp_path / "depth.csv")) == 0
    assert "6 photos of 6 with a depth" in capsys.readouterr().out
    assert plugin.command("level", project) == 0
    assert '"cameras": 6' in capsys.readouterr().out
    assert plugin.command("status", project) == 0
    out = capsys.readouterr().out
    assert "6 foto corrette" in out and "In bolla su 6 camere" in out
    assert plugin.command("use", project, "original") == 0
    assert plugin.command("csv", project, str(tmp_path / "foto.csv")) == 0
    import pyarchinit_photogrammetry as ppg

    again = ppg.Document.open(doc.path).chunk
    assert again.meta["underwater"]["levelling"]["cameras"] == 6
    assert all(Path(c.path).parent == tmp_path / "foto" for c in again.cameras)
    assert again.cameras[2].meta["underwater"]["depth"] == pytest.approx(10.6, abs=0.01)
    heights = again.to_world(np.array([c.center for c in again.cameras]))[:, 2]
    assert np.allclose(heights, [-(10.0 + 0.3 * k) for k in range(6)], atol=0.01)


@pytest.fixture
def window(tmp_path, monkeypatch):
    """The program's window on a small project, with the plugin."""
    pytest.importorskip("PySide6")
    os.environ.setdefault("PPG_RENDERER", "software")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_underwater import UnderwaterPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import main_window as mwmod
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    monkeypatch.setenv("PPG_NO_PLUGINS", "1")
    qapp = QApplication.instance() or QApplication([])
    doc, chunk = _project(tmp_path)
    app.document = doc
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    errors = []
    monkeypatch.setattr(w, "_error", lambda *a, **k: errors.append(a))
    w.resize(1200, 800)
    w.show()
    qapp.processEvents()
    plugin = UnderwaterPlugin()
    plugin.activate(Context("underwater", tmp_path / "plugin", window=w))
    messages = []
    monkeypatch.setattr(plugin, "_message", lambda kind, text, details="": messages.append((kind, text)))
    try:
        yield w, plugin, chunk, qapp, errors, messages
    finally:
        plugin.deactivate()
        plugin.context.release()
        w.setWindowModified(False)
        w.close()


def _wait(w, qapp):
    w.runner.wait()
    for _ in range(3):
        qapp.processEvents()


def test_the_pane_in_the_window(window, tmp_path, monkeypatch):
    from PySide6.QtWidgets import QDockWidget

    w, plugin, chunk, qapp, errors, messages = window
    pane = plugin.pane
    assert w.findChild(QDockWidget, "pane_plugin_underwater") is not None
    assert pane.c_photo.count() == 6 and "non ancora valutate" in pane.l_photos.text()
    assert plugin._tree_items(chunk) == []  # nothing yet
    # 1. scored, the flags in the table
    plugin.score(pane.limits())
    _wait(w, qapp)
    assert "6 foto valutate su 6" in pane.l_photos.text() and not errors
    # 2. the preview of the quick correction, the photos corrected and used
    pane.c_method.setCurrentIndex(pane.c_method.findData("quick"))
    plugin.preview(pane.colour_values())
    _wait(w, qapp)
    assert pane.l_before.pixmap() is not None and not pane.l_after.pixmap().isNull()
    plugin.correct(pane.colour_values())
    _wait(w, qapp)
    assert "6 foto corrette" in pane.l_colour.text() and "in uso: le originali" in pane.l_colour.text()
    plugin.use("corrected")
    assert "in uso: le corrette" in pane.l_colour.text() and w.isWindowModified()
    # the restoration wants depth maps
    pane.c_method.setCurrentIndex(pane.c_method.findData("seathru"))
    plugin.correct(pane.colour_values())
    assert messages and "mappe di profondità" in messages[-1][1]
    # 3. the depth log (the file dialog answered), matched, the chunk levelled
    monkeypatch.setattr(plugin, "_open_file", lambda *a: str(tmp_path / "depth.csv"))
    plugin.import_log(pane.depth_values())
    assert "6 foto su 6 con la profondità" in pane.l_log.text()
    plugin.level(pane.depth_values())
    assert "In bolla su 6 camere" in pane.l_level.text() and messages[-1][0] == "info"
    (node,) = plugin._tree_items(chunk)
    labels = [c["label"] for c in node["children"]]
    assert node["label"] == "Subacqueo" and labels[0] == "Foto corrette: 6 (in uso)" and labels[-1].startswith("In bolla")
    assert not errors


AQUALOC = Path("/Volumes/Extreme Pro/verifica_ppg/dati_pubblici/underwater/aqualoc")


@pytest.mark.skipif(not (AQUALOC / "raw_data" / "img_sequence_6.csv").exists(), reason="AQUALOC is not on this computer")
def test_aqualoc_frames_and_depths():
    """The first 400 frames of the archaeological sequence 6 of AQUALOC (Ferrera et al., IJRR 2019)."""
    from ppg_underwater import depthlog, frames, sources

    found = sources.find_dataset(AQUALOC / "raw_data")
    imgs = frames.list_images(found["images"])[:400]
    times = frames.read_frame_times(found["times"], imgs)
    res = frames.select_sequence(imgs, times, overlap=0.85)
    assert 5 <= len(res["frames"]) <= 60 and all(f["time"] is not None for f in res["frames"])
    log = depthlog.read_log(found["depth"])
    d = log.depth_at([f["time"] for f in res["frames"]])
    assert np.all(np.isfinite(d)) and np.all((d > 380) & (d < 395))
    calib = sources.read_kalibr(found["calibration"])
    assert calib["model"] == "OPENCV" and (calib["width"], calib["height"]) == (968, 608)
