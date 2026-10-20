"""The Spirula plugin on a small aligned project (three photos, a COLMAP text model written by
hand, a mask, a dense cloud) with a stand-in for the spirula executable: the dataset as Spirula
reads it, training with its progress lines, failure, cancellation, meshing as a model version,
the exporter, the command line, the plugin in the program."""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest
from ppg_spirula import runner

FAKE = r'''
import json, os, sys, time
args = sys.argv[1:]
if "--lang" in args:
    i = args.index("--lang"); del args[i:i + 2]
log = os.environ.get("FAKE_SPIRULA_LOG")
if log:
    open(os.path.join(log, "calls.txt"), "a").write(" ".join(args) + "\n")
if not args or args[0] in ("--help", "-h"):
    print("usage: spirula <command> ..."); sys.exit(0)
if args[0] == "train":
    preset = "3dgs"
    rest = args[1:]
    if rest and not rest[0].startswith("-"):
        preset, rest = rest[0], rest[1:]
    opts = dict(zip(rest[0::2], rest[1::2]))
    data = opts["--data"]
    assert os.path.isdir(os.path.join(data, "sparse", "0")), "no sparse/0"
    assert os.path.exists(os.path.join(data, "images", "foto_1.jpg")), "no images"
    out = os.path.join(opts["--output-dir-prefix"], opts["--output-dir-name"])
    os.makedirs(out, exist_ok=True)
    json.dump({"preset": preset, **opts}, open(os.path.join(out, "config.json"), "w"))
    print("Output directory: " + os.path.abspath(out), flush=True)
    if os.environ.get("FAKE_SPIRULA_FAIL"):
        print("Error: no Vulkan device", flush=True); sys.exit(1)
    n = int(opts.get("--num-iterations", {"low": 15000, "medium": 30000, "high": 50000, "ultra": 80000}[opts.get("--quality", "medium")]))
    for step in (100, n // 2, n):
        print("step %6d/%d (%3d%%)  splats %d  [elapsed 0:01 | ETA 0:01]  rgb_loss=0.05" % (step, n, 100 * step // n, 1000 + step), flush=True)
        time.sleep(float(os.environ.get("FAKE_SPIRULA_SLEEP", "0")))
    with open(os.path.join(out, "splat.ply"), "wb") as f:
        f.write(b"ply\nformat binary_little_endian 1.0\nelement vertex 0\nend_header\n")
    print("Training complete. Steps: %d   Time: 1.0s" % n, flush=True)
elif args[0] == "mesh":
    run = args[1]
    opts = dict(zip(args[2::2], args[3::2]))
    assert os.path.isdir(os.path.join(opts["--data"], "sparse", "0"))
    for fmt in opts.get("--format", "ply").split(","):
        with open(os.path.join(run, "mesh." + fmt), "w") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex 0\nelement face 0\nend_header\n")
    print("mesh 50%", flush=True); print("mesh 100%", flush=True)
else:
    if log:
        open(os.path.join(log, "opened.txt"), "w").write(args[0])
'''


@pytest.fixture
def fake_spirula(tmp_path, monkeypatch):
    """A stand-in for the spirula executable (a shell wrapper around a Python script)."""
    if sys.platform == "win32":
        pytest.skip("the stand-in is a shell script")
    folder = tmp_path / "spirula-studio"
    folder.mkdir()
    script = folder / "fake_spirula.py"
    script.write_text(FAKE)
    exe = folder / "spirula"
    exe.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n')
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "log"
    log.mkdir()
    monkeypatch.setenv("FAKE_SPIRULA_LOG", str(log))
    monkeypatch.delenv("FAKE_SPIRULA_FAIL", raising=False)
    return exe


def _project(tmp_path):
    """A chunk with three small photos, their poses, a COLMAP text model, a mask and a dense cloud."""
    import cv2
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.core.pointcloud import PointCloud
    from pyarchinit_photogrammetry.engine.masks import write_mask

    photos = tmp_path / "foto"
    photos.mkdir()
    rng = np.random.default_rng(0)
    for i in range(1, 4):
        cv2.imwrite(str(photos / f"foto_{i}.jpg"), rng.integers(0, 255, (48, 64, 3)).astype(np.uint8))
    doc = ppg.Document()
    chunk = doc.add_chunk("rilievo")
    doc.chunk = chunk
    doc.save(tmp_path / "scavo.ppg")
    cams = chunk.add_photos(sorted(photos.glob("*.jpg")))
    assert len(cams) == 3
    sensor = chunk.sensors[cams[0].sensor_key]
    sensor.model, sensor.params = "PINHOLE", [50.0, 50.0, 32.0, 24.0]
    model = chunk.data_dir / "sfm" / "model"
    model.mkdir(parents=True)
    names = {}
    with open(model / "cameras.txt", "w") as f:
        f.write("1 PINHOLE 64 48 50 50 32 24\n")
    with open(model / "images.txt", "w") as f:
        for i, cam in enumerate(cams, 1):
            cam.transform = np.eye(4)
            cam.transform[:3, 3] = [i - 2.0, 0.0, 0.0]  # cameras along x, looking down +z
            names[str(cam.key)] = f"foto_{i}.jpg"
            f.write(f"{i} 1 0 0 0 {-(i - 2.0)} 0 0 1 foto_{i}.jpg\n")
            f.write("32 24 1 40 30 2\n")
    with open(model / "points3D.txt", "w") as f:
        f.write("1 0 0 5 200 100 50 0.5 1 0 2 0 3 0\n2 0.5 0.3 5 100 100 100 0.5 1 1 2 1 3 1\n")
    chunk._set_product("reconstruction", model)
    chunk.meta["sfm"] = {"image_root": str(photos), "names": names}
    keep = np.ones((48, 64), bool)
    keep[:10] = False
    write_mask(chunk, cams[0], keep)
    chunk.point_cloud = PointCloud(rng.uniform(-1, 1, (500, 3)) + [0, 0, 5], rng.integers(0, 255, (500, 3)).astype(np.uint8))
    doc.save()
    return doc


def test_dataset_as_spirula_reads_it(tmp_path):
    doc = _project(tmp_path)
    chunk = doc.chunk
    seen = []
    ds = runner.prepare_dataset(chunk, progress=lambda f, m: seen.append(f))
    folder = ds["folder"]
    assert folder == chunk.data_dir / "spirula" / "dataset" and seen[-1] == 1.0
    assert (folder / "sparse" / "0" / "cameras.txt").read_text().startswith("# Camera list") or \
        (folder / "sparse" / "0" / "cameras.txt").exists()
    assert "foto_2.jpg" in (folder / "sparse" / "0" / "images.txt").read_text()
    assert (folder / "images" / "foto_1.jpg").exists() and ds["images"] == 3 and ds["images_link"] in ("symlink", "junction", "copy")
    assert ds["masks"] and (folder / "masks" / "foto_1.jpg.png").exists() and (folder / "masks" / "foto_2.jpg.png").exists()
    assert ds["seed"] == folder / "seed.ply" and ds["seed_points"] == 500
    head = (folder / "seed.ply").read_bytes()[:200]
    assert b"element vertex 500" in head and b"property uchar red" in head
    sparse = runner.prepare_dataset(chunk, masks=False, seed="sparse")
    assert not sparse["masks"] and sparse["seed"] is None and not (folder / "seed.ply").exists()
    assert runner.suggested_divisor(chunk) == 1
    sensor = chunk.sensors[chunk.cameras[0].sensor_key]
    sensor.width, sensor.height = 8064, 4536
    assert runner.suggested_divisor(chunk) == 3


def test_train_mesh_and_open(tmp_path, fake_spirula):
    from pyarchinit_photogrammetry.core import versions

    doc = _project(tmp_path)
    chunk = doc.chunk
    seen = []
    report = runner.train(chunk, fake_spirula, preset="in-the-wild", quality="low", divisor=2, sh_degree=2,
                          name="prova", progress=lambda f, m: seen.append((f, m)))
    splat = report["path"]
    assert splat == chunk.data_dir / "spirula" / "runs" / "prova" / "splat.ply" and splat.exists()
    assert chunk.product_path("splat") == splat and chunk.meta["splat"]["steps"] == 15000
    assert chunk.meta["splat"]["splats"] == 16000 and chunk.meta["splat"]["preset"] == "in-the-wild"
    assert chunk.meta["splat"]["seed"] == "dense" and chunk.meta["splat"]["masks"] and chunk.meta["splat"]["photos"] == 3
    fractions = [f for f, _ in seen]
    assert fractions == sorted(fractions) and fractions[-1] == 1.0 and any("step 7,500" in m for _, m in seen)
    calls = (tmp_path / "log" / "calls.txt").read_text()
    assert "train in-the-wild --data" in calls and "--quality low" in calls and "--sh-degree 2" in calls
    assert "--train-resolution-divisor 2" in calls and "--seed-pointcloud seed.ply" in calls and "--disable-viewer 1" in calls
    assert "--load-masks" not in calls
    # the mesh: beside the splat, the PLY among the versions of the model
    out = runner.mesh(chunk, fake_spirula, formats=["ply", "glb"], colour="vertex")
    assert [p.name for p in out["files"]] == ["mesh.glb", "mesh.ply"]
    kept = versions.versions(chunk, "model")
    assert len(kept) == 1 and kept[0]["name"].startswith("Mesh dallo splat") and out["version"] == kept[0]
    assert versions.version_file(chunk, "model", kept[0]["id"]).exists()
    assert chunk.meta["splat"]["mesh"] == ["spirula/runs/prova/mesh.glb", "spirula/runs/prova/mesh.ply"]
    runner.open_viewer(fake_spirula, splat)
    import time

    for _ in range(50):
        if (tmp_path / "log" / "opened.txt").exists():
            break
        time.sleep(0.1)
    assert (tmp_path / "log" / "opened.txt").read_text() == str(splat)


def test_train_failure_and_cancellation(tmp_path, fake_spirula, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    doc = _project(tmp_path)
    chunk = doc.chunk
    monkeypatch.setenv("FAKE_SPIRULA_FAIL", "1")
    with pytest.raises(RuntimeError, match="no Vulkan device"):
        runner.train(chunk, fake_spirula, quality="low")
    assert "splat" not in chunk.products
    monkeypatch.delenv("FAKE_SPIRULA_FAIL")
    monkeypatch.setenv("FAKE_SPIRULA_SLEEP", "2")
    progress = Progress(lambda f, m: progress.cancel() if "step" in m else None)
    started = __import__("time").time()
    with pytest.raises(Cancelled):
        runner.train(chunk, fake_spirula, quality="low", progress=progress)
    assert __import__("time").time() - started < 8, "the process was ended at the first step line"
    assert "splat" not in chunk.products
    with pytest.raises(ValueError):
        runner.train(chunk, fake_spirula, preset="nope")
    with pytest.raises(runner.SpirulaMissing):
        runner.train(chunk, tmp_path / "nowhere", quality="low")
    with pytest.raises(RuntimeError, match="train the splats first"):
        runner.mesh(chunk, fake_spirula)


def test_find_spirula(tmp_path, fake_spirula, monkeypatch):
    monkeypatch.delenv("PPG_SPIRULA", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    assert runner.find_spirula(str(fake_spirula)) == fake_spirula
    assert runner.find_spirula(str(fake_spirula.parent)) == fake_spirula, "the folder of the release"
    assert runner.resolve("/Applications/Spirula Studio.app") == Path("/Applications/Spirula Studio.app/Contents/MacOS/spirula")
    monkeypatch.setenv("PPG_SPIRULA", str(fake_spirula))
    assert runner.find_spirula() == fake_spirula
    monkeypatch.delenv("PPG_SPIRULA")
    monkeypatch.setattr(runner, "candidates", lambda: [])
    assert runner.find_spirula() is None
    with pytest.raises(runner.SpirulaMissing):
        runner.require()


def test_exporter_and_plugin_in_the_program(tmp_path, fake_spirula):
    from ppg_spirula import SpirulaPlugin
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("spirula")
    assert info is not None and info.active, info and info.error
    assert [a.title for a in manager.actions if a.plugin == "spirula"] == [
        "Addestra Gaussian splatting...", "Mesh dallo splat...", "Apri lo splat in Spirula Studio",
        "Installa Spirula Studio...", "Dove sta Spirula Studio..."]
    assert "spirula" in manager.commands
    assert [e.suffix for e in manager.exporters if e.plugin == "spirula"] == [".ply"]
    doc = _project(tmp_path)
    plugin = SpirulaPlugin()
    with pytest.raises(RuntimeError):
        plugin.export_splat(doc.chunk, tmp_path / "s.ply", lambda *a: None)
    runner.train(doc.chunk, fake_spirula, quality="low", name="r1")
    plugin.export_splat(doc.chunk, tmp_path / "s.ply", lambda *a: None)
    assert (tmp_path / "s.ply").read_bytes().startswith(b"ply")


def test_command(tmp_path, fake_spirula, capsys, monkeypatch):
    from ppg_spirula import SpirulaPlugin

    doc = _project(tmp_path)
    plugin = SpirulaPlugin()
    monkeypatch.setenv("PPG_SPIRULA", str(fake_spirula))
    assert plugin.command("where") == 0 and str(fake_spirula) in capsys.readouterr().out
    assert plugin.command("train", str(doc.path), "--quality", "low", "--name", "cli", "--no-masks") == 0
    out = capsys.readouterr().out
    assert "Splat:" in out and "15,000 passi" in out
    assert "--load-masks 0" in (tmp_path / "log" / "calls.txt").read_text()
    assert plugin.command("mesh", str(doc.path), "--format", "ply,obj") == 0
    assert "mesh.obj" in capsys.readouterr().out
    assert plugin.command("open", str(doc.path)) == 0
    assert plugin.command("train", str(doc.path), "--chunk", "nessuno") == 2
    monkeypatch.setenv("FAKE_SPIRULA_FAIL", "1")
    assert plugin.command("train", str(doc.path), "--quality", "low") == 1
    assert "error" in capsys.readouterr().out
    monkeypatch.delenv("PPG_SPIRULA")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(runner, "candidates", lambda: [])
    assert plugin.command("where") == 1
    assert os.environ.get("PPG_LANG") == "it"
