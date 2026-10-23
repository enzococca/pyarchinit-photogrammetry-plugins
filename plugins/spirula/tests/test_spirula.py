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
from ppg_spirula import runner, splat

FAKE = r'''
import json, os, struct, sys, time
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
    if opts.get("--disable-viewer") == "0":
        p = opts.get("--viewer-port", "7007")
        print("Viewer at http://0.0.0.0:%s/ (forward the port for remote boxes: ssh -L %s:localhost:%s <host>)" % (p, p, p), flush=True)
    if os.environ.get("FAKE_SPIRULA_FAIL"):
        print("Error: no Vulkan device", flush=True); sys.exit(1)
    n = int(opts.get("--num-iterations", {"low": 15000, "medium": 30000, "high": 50000, "ultra": 80000}[opts.get("--quality", "medium")]))
    for step in (100, n // 2, n):
        print("step %6d/%d (%3d%%)  splats %d  [elapsed 0:01 | ETA 0:01]  rgb_loss=0.05" % (step, n, 100 * step // n, 1000 + step), flush=True)
        time.sleep(float(os.environ.get("FAKE_SPIRULA_SLEEP", "0")))
    deg = int(opts.get("--sh-degree", "3"))
    k = (deg + 1) ** 2 - 1
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2"] + ["f_rest_%d" % i for i in range(3 * k)] + \
        ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    rows = []
    for i in range(4):  # four splats, as Spirula writes them: the 3DGS PLY, in the frame of the training
        rows += [0.1 * i, -0.2 * i, 0.3, 0, 0, 0, 0.5, 0.4, 0.3] + [0.01 * (j + 1) for j in range(3 * k)] + \
            [0.8, -3.0, -3.5, -4.0, 1.0, 0.1 * i, 0.0, 0.0]
    with open(os.path.join(out, "splat.ply"), "wb") as f:
        f.write(("ply\nformat binary_little_endian 1.0\nelement vertex 4\n" + "".join("property float %s\n" % m for m in names)
                 + "end_header\n").encode("ascii") + struct.pack("<%df" % len(rows), *rows))
    json.dump({"world_from_train": {"matrix_4x4": [[0, -2, 0, 1], [2, 0, 0, 2], [0, 0, 2, 3], [0, 0, 0, 1]]}},
              open(os.path.join(out, "scene_transform.json"), "w"))
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
    assert fractions == sorted(fractions) and fractions[-1] == 1.0
    assert any(m == "Spirula: passo 7.500 di 15.000, 8.500 splat" for _, m in seen), "in the language of the program"
    assert chunk.meta["splat"]["world_from_train"][0] == [0.0, -2.0, 0.0, 1.0]
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


def test_web_viewer_opens_in_the_browser(tmp_path, fake_spirula, monkeypatch):
    doc = _project(tmp_path)
    chunk = doc.chunk
    opened = []
    monkeypatch.setattr(runner, "open_url", lambda url: opened.append(url) or True)
    seen = []
    report = runner.train(chunk, fake_spirula, quality="low", viewer=True, port=7117, name="web",
                          progress=lambda f, m: seen.append(m))
    assert opened == ["http://localhost:7117/"] and report["viewer"] == "http://localhost:7117/"
    assert any(m == "Spirula: visualizzatore web su http://localhost:7117/" for m in seen)
    calls = (tmp_path / "log" / "calls.txt").read_text()
    assert "--disable-viewer 0" in calls and "--viewer-port 7117" in calls
    # the browser declined, or no viewer at all: nothing opens
    opened.clear()
    quiet = runner.train(chunk, fake_spirula, quality="low", viewer=True, port=7117, open_browser=False, name="quiet")
    none = runner.train(chunk, fake_spirula, quality="low", name="none")
    assert opened == [] and quiet["viewer"] == "http://localhost:7117/" and none["viewer"] is None
    assert runner._local_url("http://0.0.0.0:7007/") == "http://localhost:7007/"
    assert runner._local_url("http://[::]:7007/") == "http://localhost:7007/"
    assert runner._local_url("http://localhost:7007/") == "http://localhost:7007/"
    if sys.platform == "darwin":
        assert runner.browser_command("http://localhost:7007/") == ["/usr/bin/open", "http://localhost:7007/"]


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
    progress = Progress(lambda f, m: progress.cancel() if "passo" in m else None)  # the first step line
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
    assert [e.suffix for e in manager.exporters if e.plugin == "spirula"] == [".ply", ".ply"]
    assert [e.title for e in manager.exporters if e.plugin == "spirula"] == [
        "Splat (PLY 3D Gaussian Splatting)", "Splat (PLY, sistema dell'addestramento)"]
    doc = _project(tmp_path)
    plugin = SpirulaPlugin()
    with pytest.raises(RuntimeError):
        plugin.export_splat(doc.chunk, tmp_path / "s.ply", lambda *a: None)
    report = runner.train(doc.chunk, fake_spirula, quality="low", name="r1")
    plugin.export_splat(doc.chunk, tmp_path / "s.ply", lambda *a: None)
    plugin.export_splat_training(doc.chunk, tmp_path / "t.ply", lambda *a: None)
    assert (tmp_path / "t.ply").read_bytes() == report["path"].read_bytes(), "the frame of the training: a copy"
    _, elements = splat.read_ply(tmp_path / "s.ply")
    moved = elements[0][1]
    # world_from_train of the stand-in: x' = 2 (-y) + 1, y' = 2 x + 2, z' = 2 z + 3; the chunk transform is the identity
    assert np.allclose([moved["x"][1], moved["y"][1], moved["z"][1]], [1.4, 2.2, 3.6], atol=1e-6)
    assert np.allclose(moved["scale_0"], -3.0 + np.log(2.0), atol=1e-6)


def test_command(tmp_path, fake_spirula, capsys, monkeypatch):
    from ppg_spirula import SpirulaPlugin

    doc = _project(tmp_path)
    plugin = SpirulaPlugin()
    monkeypatch.setenv("PPG_SPIRULA", str(fake_spirula))
    assert plugin.command("where") == 0 and str(fake_spirula) in capsys.readouterr().out
    assert plugin.command("train", str(doc.path), "--quality", "low", "--name", "cli", "--no-masks") == 0
    out = capsys.readouterr().out
    assert "Splat:" in out and "15.000 passi" in out
    assert "--load-masks 0" in (tmp_path / "log" / "calls.txt").read_text()
    assert plugin.command("mesh", str(doc.path), "--format", "ply,obj") == 0
    assert "mesh.obj" in capsys.readouterr().out
    assert plugin.command("open", str(doc.path)) == 0
    assert plugin.command("export", str(doc.path), str(tmp_path / "rilievo.ply")) == 0
    assert "coordinate del rilievo" in capsys.readouterr().out and (tmp_path / "rilievo.ply").exists()
    assert plugin.command("export", str(doc.path), str(tmp_path / "addestramento.ply"), "--frame", "training") == 0
    capsys.readouterr()
    assert plugin.command("train", str(doc.path), "--chunk", "nessuno") == 2
    monkeypatch.setenv("FAKE_SPIRULA_FAIL", "1")
    assert plugin.command("train", str(doc.path), "--quality", "low") == 1
    assert "error" in capsys.readouterr().out
    monkeypatch.delenv("PPG_SPIRULA")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(runner, "candidates", lambda: [])
    assert plugin.command("where") == 1
    assert os.environ.get("PPG_LANG") == "it"


def _splat_ply(path, n=40, degree=3, seed=0):
    """A 3DGS PLY of ``n`` random splats with harmonics up to ``degree``; returns its vertex array."""
    rng = np.random.default_rng(seed)
    k = (degree + 1) ** 2 - 1
    names = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2", *[f"f_rest_{i}" for i in range(3 * k)],
             "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    arr = np.zeros(n, [(m, "<f4") for m in names])
    for m in names:
        arr[m] = rng.normal(size=n)
    arr["nx"], arr["ny"], arr["nz"] = 0.0, 0.0, 0.0
    splat.write_ply(path, ["comment made by the tests"], [("vertex", arr)])
    return arr


def _colours(arr, dirs, degree):
    """The colour of every splat seen along ``dirs`` (one direction per splat), as 3DGS renders it."""
    k = (degree + 1) ** 2 - 1
    basis = splat.sh_basis(dirs, degree)
    out = []
    for c in range(3):
        coeffs = np.column_stack([arr[f"f_dc_{c}"]] + [arr[f"f_rest_{c * k + i}"] for i in range(k)]).astype(np.float64)
        out.append(np.einsum("nk,nk->n", basis, coeffs) + 0.5)
    return np.column_stack(out)


def test_splat_in_survey_coordinates(tmp_path):
    """The splat moved from the frame of the training to the survey coordinates of the chunk:
    positions, rotations, scales and the colour seen from every direction follow the similarity."""
    import shutil

    from scipy.spatial.transform import Rotation

    doc = _project(tmp_path)
    chunk = doc.chunk
    degree = 3
    src = tmp_path / "addestrato.ply"
    old = _splat_ply(src, degree=degree)
    run = chunk.data_dir / "spirula" / "runs" / "prova"
    run.mkdir(parents=True)
    shutil.copy2(src, run / "splat.ply")
    chunk._set_product("splat", run / "splat.ply")
    rot = Rotation.from_euler("zyx", [35.0, -20.0, 70.0], degrees=True).as_matrix()
    world_from_train = np.eye(4)
    world_from_train[:3, :3], world_from_train[:3, 3] = 0.8 * rot, [1.0, -2.0, 0.5]
    chunk.transform = np.eye(4)
    chunk.transform[:3, :3] = 3.0 * Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    chunk.transform[:3, 3] = [300000.0, 3960000.0, 100.0]
    chunk.crs = "EPSG:32637"
    chunk.meta["splat"] = {"world_from_train": world_from_train.tolist(), "run": "spirula/runs/prova"}
    m = chunk.transform @ world_from_train
    s, r = 2.4, m[:3, :3] / 2.4
    out = splat.export_splat(chunk, tmp_path / "rilievo.ply")
    assert out["frame"] == "survey" and out["splats"] == 40 and out["double"], "UTM positions need doubles"
    comments, elements = splat.read_ply(tmp_path / "rilievo.ply")
    new = elements[0][1]
    assert new.dtype["x"] == np.float64 and new.dtype["f_rest_0"] == np.float32
    assert any("EPSG:32637" in c for c in comments)
    xyz = np.column_stack([old["x"], old["y"], old["z"]]).astype(np.float64)
    assert np.allclose(np.column_stack([new["x"], new["y"], new["z"]]), xyz @ m[:3, :3].T + m[:3, 3], atol=1e-6)
    for i in range(3):
        assert np.allclose(new[f"scale_{i}"], old[f"scale_{i}"].astype(np.float64) + np.log(s), atol=1e-5)
    q_old = np.column_stack([old[f"rot_{i}"] for i in range(4)])
    q_new = np.column_stack([new[f"rot_{i}"] for i in range(4)])
    assert np.allclose(splat.quaternion_matrix(q_new), r @ splat.quaternion_matrix(q_old), atol=1e-5)
    assert np.allclose(np.linalg.norm(q_new, axis=1), np.linalg.norm(q_old, axis=1), rtol=1e-5)
    for name in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        assert np.array_equal(new[name], old[name])
    rng = np.random.default_rng(7)
    for _ in range(3):
        d = rng.normal(size=(len(old), 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        assert np.allclose(_colours(new, d @ r.T, degree), _colours(old, d, degree), atol=1e-4), \
            "the colour seen along R d in the survey is the one seen along d in the training"
    # the frame of the training: the file as Spirula wrote it; a small local frame keeps 32-bit positions
    assert splat.export_splat(chunk, tmp_path / "copia.ply", "training")["frame"] == "training"
    assert (tmp_path / "copia.ply").read_bytes() == src.read_bytes()
    chunk.transform = np.eye(4)
    local = splat.export_splat(chunk, tmp_path / "locale.ply")
    assert not local["double"] and splat.read_ply(tmp_path / "locale.ply")[1][0][1].dtype["x"] == np.float32
    # degrees 0 to 2 too, and a splat without a frame of its own cannot go to the survey
    for deg in (0, 1, 2):
        arr = _splat_ply(src, degree=deg, seed=deg)
        splat.transform_splat(src, tmp_path / f"g{deg}.ply", m)
        moved = splat.read_ply(tmp_path / f"g{deg}.ply")[1][0][1]
        d = rng.normal(size=(len(arr), 3))
        d /= np.linalg.norm(d, axis=1)[:, None]
        assert np.allclose(_colours(moved, d @ r.T, deg), _colours(arr, d, deg), atol=1e-4), deg
    chunk.meta["splat"] = {"run": "spirula/runs/prova"}
    with pytest.raises(RuntimeError, match="scene_transform"):
        splat.export_splat(chunk, tmp_path / "x.ply")
    with pytest.raises(ValueError):
        splat.similarity(np.diag([1.0, 2.0, 1.0, 1.0]))


def test_progress_in_english(tmp_path, fake_spirula):
    """The progress of the training follows the language of the program."""
    from pyarchinit_photogrammetry.gui import i18n

    doc = _project(tmp_path)
    seen = []
    try:
        i18n.set_current("en")
        runner.train(doc.chunk, fake_spirula, quality="low", name="en", progress=lambda f, m: seen.append(m))
    finally:
        i18n.set_current(None)
    assert "Spirula: step 7,500 of 15,000, 8,500 splats" in seen and "Dataset: 3 photos" in seen


def test_training_needs_aligned_photos(tmp_path, fake_spirula, capsys, monkeypatch):
    """The command line, the window and the training itself refuse a chunk whose photos are not
    aligned, before looking for Spirula."""
    from ppg_spirula import SpirulaPlugin

    doc = _project(tmp_path)
    for cam in doc.chunk.cameras:
        cam.transform = None
    doc.save()
    with pytest.raises(RuntimeError, match="non sono allineate"):
        runner.train(doc.chunk, fake_spirula, quality="low")
    monkeypatch.delenv("PPG_SPIRULA", raising=False)
    monkeypatch.setattr(runner, "candidates", lambda: [])
    assert SpirulaPlugin().command("train", str(doc.path), "--quality", "low") == 1
    out = capsys.readouterr().out
    assert "non sono allineate" in out and "Allinea foto" in out and "Spirula Studio not found" not in out
