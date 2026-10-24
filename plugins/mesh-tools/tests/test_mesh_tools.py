"""The plugin with a fake worker (``fake_worker.py``: the protocol of the real worker, numpy in
place of pymeshlab): the messages and the progress while it runs, failures, crashes and
cancelling, the new versions of the model (named, the one before kept, put back when storing
fails), the texture kept or left out, the vertex properties carried over, the borders kept by
the smoothing, the measures in metres, the parameters, the environment of MeshLab and its
installation with uv, the command line, the pane in the window."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from mesh_samples import SCALE, grid_mesh, project
from ppg_mesh_tools import environment, model, runner
from pyarchinit_photogrammetry.core import versions
from pyarchinit_photogrammetry.io.mesh_ply import read_mesh

FAKE = Path(__file__).with_name("fake_worker.py")


@pytest.fixture
def fake(monkeypatch):
    """Every run of the plugin goes to the fake worker, with this Python (as if pymeshlab were there)."""
    monkeypatch.setattr(runner, "WORKER", FAKE)
    monkeypatch.setattr(environment, "worker_python", lambda: Path(sys.executable))
    monkeypatch.setattr(environment, "where", lambda: ("local", "0.0-fake"))
    monkeypatch.setattr(environment, "unsupported", lambda *a: None)
    monkeypatch.setattr(environment, "known_usable", lambda: True)
    monkeypatch.delenv("PPG_MESH_TOOLS_FAKE", raising=False)
    return monkeypatch


def _job(tmp_path, mesh, ops, texture=False, measure=True):
    from pyarchinit_photogrammetry.io.mesh_ply import write_mesh

    write_mesh(tmp_path / "in.ply", mesh)
    return {"protocol": 1, "input": str(tmp_path / "in.ply"), "output": str(tmp_path / "out.ply"), "operations": ops,
            "measure": measure, "texture": texture, "colors": True}


def test_the_plugin_in_the_program(tmp_path):
    from pyarchinit_photogrammetry.plugins import Manager

    manager = Manager(tmp_path / "plugins")
    manager.load()
    info = manager.get("mesh-tools")
    assert info is not None and info.active, info and info.error
    titles = [a.title for a in manager.actions if a.plugin == "mesh-tools"]
    assert titles[:6] == ["Decima il modello...", "Rimaglia il modello (isotropa)...", "Leviga il modello...",
                          "Chiudi i buchi del modello...", "Ripara la topologia del modello...", "Misura il modello"]
    assert "mesh-tools" in manager.commands
    manager.unload()


def test_the_worker_speaks_while_it_runs(tmp_path, fake):
    fake.setenv("PPG_MESH_TOOLS_FAKE", "noise")
    seen = []
    job = _job(tmp_path, grid_mesh(), [{"op": "decimate", "target_faces": 500, "texture": True}], texture=True)
    result = runner.run(job, tmp_path, lambda f, m: seen.append((f, m)))
    assert result["faces_before"] > 500 and result["faces"] == 500 and result["version"] == "0.0-fake"
    assert result["steps"][0]["op"] == "decimate" and result["steps"][0]["faces"] == 500
    assert result["measures"]["faces"] == 500 and result["measures"]["boundary_edges"] > 0
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    assert any(m == "MeshLab: Decimazione" for _f, m in seen) and seen[0][1] == "MeshLab: avvio"
    assert [m for _f, m in seen][-1] == "MeshLab: scrittura del risultato"
    out = read_mesh(tmp_path / "out.ply")
    assert len(out.faces) == 500 and out.texcoords is not None and out.colors is not None


def test_failures_and_crashes_come_back_with_their_reason(tmp_path, fake):
    fake.setenv("PPG_MESH_TOOLS_FAKE", "fail")
    with pytest.raises(runner.WorkerError) as err:
        runner.run(_job(tmp_path, grid_mesh(), [{"op": "close_holes", "max_hole_edges": 30}]), tmp_path)
    assert "Chiusura dei buchi" in str(err.value) and "2-manifold" in str(err.value)
    assert "Ripara la topologia" in str(err.value) and err.value.op == "close_holes" and "Traceback" in err.value.details
    fake.setenv("PPG_MESH_TOOLS_FAKE", "crash")
    with pytest.raises(runner.WorkerError) as err:
        runner.run(_job(tmp_path, grid_mesh(), [{"op": "remesh", "edge_length": 0.05}]), tmp_path)
    assert "si è chiuso inaspettatamente durante «Rimaglia isotropa»" in str(err.value)
    # the model of a chunk stays as it was, no version is added
    doc, ch = project(tmp_path / "p")
    before = (ch.data_dir / "model.ply").read_bytes()
    with pytest.raises(runner.WorkerError):
        model.apply(ch, "remesh", {"edge_length": 0.1})
    assert (ch.data_dir / "model.ply").read_bytes() == before and versions.versions(ch, "model") == []
    assert not list(ch.data_dir.glob(model.TEMP_PREFIX + "*"))  # the work folder went with it


def test_cancelling_stops_the_worker_at_once(tmp_path, fake):
    from pyarchinit_photogrammetry.core.progress import Cancelled, Progress

    fake.setenv("PPG_MESH_TOOLS_FAKE", "sleep")
    doc, ch = project(tmp_path)
    before = (ch.data_dir / "model.ply").read_bytes()
    progress = Progress()
    threading.Timer(1.0, progress.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        model.apply(ch, "decimate", {"target_faces": 300}, progress)
    assert time.monotonic() - t0 < 15.0  # not the minute of the operation
    assert (ch.data_dir / "model.ply").read_bytes() == before and versions.versions(ch, "model") == []


def test_decimation_keeps_the_texture_and_the_properties_in_a_new_version(tmp_path, fake):
    doc, ch = project(tmp_path)
    old = read_mesh(ch.product_path("model"))
    first_name = versions.active_info(ch, "model")["name"]
    report = model.apply(ch, "decimate", {"target_faces": 1000})
    assert report["changed"] and report["faces"] == 1000 and report["texture_kept"]
    assert report["name"] == "MeshLab: decimazione a 1.000 facce (texture conservata)"
    assert versions.active_info(ch, "model")["name"] == report["name"]
    kept = versions.versions(ch, "model")
    assert [v["name"] for v in kept] == [first_name] and kept[0]["id"] == report["version"]
    assert len(read_mesh(versions.version_file(ch, "model", kept[0]["id"])).faces) == len(old.faces)
    new = read_mesh(ch.product_path("model"))
    assert len(new.faces) == 1000 and new.texture_files == ["model_texture.jpg"]
    assert ch.product_path("model_textured") == ch.data_dir / "model_texture.jpg"
    # the texture coordinates are still those of the corners (x, y), the confidence that of the vertex
    corners = new.xyz[new.faces][:, :, :2].reshape(len(new.faces), 6)
    assert np.allclose(new.texcoords, corners, atol=1e-5)
    lookup = {tuple(np.round(p, 5)): c for p, c in zip(old.xyz, old.field("confidence"))}
    assert all(lookup[tuple(np.round(p, 5))] == c for p, c in zip(new.xyz, new.field("confidence")))
    assert set(new.vertices.dtype.names) == set(old.vertices.dtype.names)  # MeshLab's alpha did not come in
    assert ch.meta["model"]["meshlab"][-1]["operation"] == "decimate"
    assert "Modello decimato" in model.report_text(report) and report["measures"]["faces"] == 1000
    with pytest.raises(ValueError, match="niente da decimare"):
        model.apply(ch, "decimate", {"target_faces": 5000})


def test_remeshing_takes_the_properties_from_the_nearest_vertex_and_leaves_the_texture(tmp_path, fake):
    doc, ch = project(tmp_path)
    old = read_mesh(ch.product_path("model"))
    report = model.apply(ch, "remesh", {"edge_length": 0.05}, name="Rimagliato per la stampa")
    assert report["changed"] and not report["texture_kept"] and report["name"] == "Rimagliato per la stampa"
    assert "la texture va ricostruita" in model.report_text(report)
    new = read_mesh(ch.product_path("model"))
    assert new.texcoords is None and new.texture_files == [] and ch.product_path("model_textured") is None
    # the fake reversed the order of the vertices: each one found its own properties again
    assert np.allclose(new.xyz, old.xyz[::-1]) and np.array_equal(new.field("confidence"), old.field("confidence")[::-1])
    assert np.array_equal(new.colors, old.colors[::-1])
    # the edge given in metres reaches the worker in the units of the model (scale 2)
    assert report["params"]["edge_length"] == pytest.approx(0.05)
    # the version kept still has its texture page
    kept = versions.versions(ch, "model")[-1]
    assert "texture" in kept["files"] and (ch.data_dir / kept["files"]["texture"]).exists()


def test_smoothing_keeps_the_borders_and_the_texture(tmp_path, fake):
    doc, ch = project(tmp_path, grid_mesh(duplicate_seam=True))
    old = read_mesh(ch.product_path("model"))
    report = model.apply(ch, "smooth", {"method": "hc"})
    assert report["params"]["steps"] == 1 and report["name"] == "MeshLab: levigatura HC laplaciana, 1 passo"
    assert report["welded"] == len(old.vertices) - len(np.unique(old.xyz, axis=0)) > 20  # the seam counted once
    new = read_mesh(ch.product_path("model"))
    assert report["texture_kept"] and new.texcoords is not None and len(new.faces) == len(old.faces)
    e = np.sort(np.concatenate([new.faces[:, [0, 1]], new.faces[:, [1, 2]], new.faces[:, [2, 0]]]), axis=1)
    uniq, count = np.unique(e, axis=0, return_counts=True)
    border = np.zeros(len(new.vertices), bool)
    border[uniq[count == 1].ravel()] = True
    assert border.sum() > 100
    old_z = {tuple(np.round(p[:2], 5)): p[2] for p in old.xyz}
    dz = np.array([z - old_z[tuple(np.round(p[:2], 5))] for p, z in zip(new.xyz, new.xyz[:, 2])])
    assert np.allclose(dz[border], 0.0, atol=1e-6) and np.allclose(dz[~border], 0.01, atol=1e-6)
    # the borders move too when asked
    model.apply(ch, "smooth", {"method": "taubin", "steps": 3, "keep_borders": False})
    again = read_mesh(ch.product_path("model"))
    assert np.allclose(again.xyz[:, 2] - new.xyz[:, 2], 0.01, atol=1e-6)
    assert versions.active_info(ch, "model")["name"] == "MeshLab: levigatura Taubin, 3 passi"
    # the history of the operations belongs to each version: the one kept before does not get the new ones
    assert [e["operation"] for e in ch.meta["model"]["meshlab"]] == ["smooth", "smooth"]
    assert [len(v["params"].get("meshlab", [])) for v in versions.versions(ch, "model")] == [0, 1]


def test_nothing_to_do_leaves_the_model_as_it_was(tmp_path, fake):
    doc, ch = project(tmp_path)
    before = (ch.data_dir / "model.ply").read_bytes()
    report = model.apply(ch, "close_holes", {"max_hole_edges": 5})
    assert not report["changed"] and report["version"] is None and versions.versions(ch, "model") == []
    assert (ch.data_dir / "model.ply").read_bytes() == before
    assert "il modello non è cambiato" in model.report_text(report)
    report = model.apply(ch, "close_holes", {"max_hole_edges": 40})
    assert report["changed"] and report["details"]["closed_holes"] == 1 and not report["texture_kept"]
    assert model.report_text(report).startswith("Chiuso 1 buco con 1 faccia")


def test_the_outlines_stay_open(tmp_path, fake):
    from pyarchinit_photogrammetry.io.mesh_ply import MeshData

    grid = grid_mesh(texture=False)
    mask, holes, outlines = model.hole_faces(grid)
    centre = np.hypot(*(grid.xyz[grid.faces].mean(axis=1)[:, :2] - 0.5).T)
    assert (holes, outlines) == (1, 1) and mask.any() and centre[mask].max() < 0.2  # the faces around the hole only
    other = grid.vertices.copy()
    other["x"] += 2.0  # a second part, beside the first
    two = MeshData(np.concatenate([grid.vertices, other]), np.vstack([grid.faces, grid.faces + len(grid.vertices)]))
    assert model.hole_faces(two)[1:] == (2, 2)
    plain = grid_mesh(hole=False, texture=False)
    assert model.hole_faces(plain)[1:] == (0, 1)
    # nothing but the outline: nothing to close, MeshLab is not even started (here it would crash)
    doc, ch = project(tmp_path, plain, texture=False)
    fake.setenv("PPG_MESH_TOOLS_FAKE", "crash")
    report = model.apply(ch, "close_holes", {"max_hole_edges": 100000})
    assert not report["changed"] and report["details"]["holes"] == 0 and versions.versions(ch, "model") == []


def test_a_failure_while_storing_puts_the_model_back(tmp_path, fake):
    from pyarchinit_photogrammetry.engine import cleaning

    doc, ch = project(tmp_path)
    before = (ch.data_dir / "model.ply").read_bytes()
    name = versions.active_info(ch, "model")["name"]
    real = cleaning.store_model

    def broken(chunk, mesh, texture_kept=True):
        real(chunk, mesh, texture_kept)  # the new model written, the texture left out ...
        raise OSError("disco pieno")  # ... then the failure

    fake.setattr(cleaning, "store_model", broken)
    with pytest.raises(OSError, match="disco pieno"):
        model.apply(ch, "remesh", {"edge_length": 0.05})
    assert (ch.data_dir / "model.ply").read_bytes() == before
    assert ch.product_path("model_textured") == ch.data_dir / "model_texture.jpg"
    assert versions.versions(ch, "model") == [] and versions.active_info(ch, "model")["name"] == name
    assert not (ch.data_dir / "versions" / "model").exists() or not any((ch.data_dir / "versions" / "model").iterdir())


def test_measures_in_metres(tmp_path, fake):
    mesh = grid_mesh(texture=False)
    doc, ch = project(tmp_path, mesh, texture=False)
    m = model.measure(ch)
    tri = mesh.xyz[mesh.faces]
    area = np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum() / 2
    assert m["area"] == pytest.approx(area * SCALE ** 2, rel=1e-6) and m["unit"] == "m"
    assert m["volume"] is None and not m["watertight"] and m["boundary_edges"] > 0
    assert m["size"][0] == pytest.approx(SCALE, rel=1e-6) and m["bbox_min"][0] == pytest.approx(500000.0)
    rows = dict(model.measure_rows(m))
    assert rows["Volume"] == "– (il modello non è chiuso)" and rows["Chiuso (watertight)"] == "no"
    assert rows["Area della superficie"].endswith(" m²") and "," in rows["Area della superficie"]
    # a chunk without a scale: units of the model
    doc2, ch2 = project(tmp_path / "local", grid_mesh(texture=False), texture=False, scale=1.0)
    ch2.transform = np.eye(4)
    assert model.measure(ch2)["unit"] == "unità"


def test_parameters_and_names(tmp_path):
    info = {"faces": 400_000, "scale": 2.0, "textured": True, "mean_edge": 0.012}
    p, w = model.resolve("decimate", {"fraction": 0.5}, info)
    assert p["target_faces"] == w["target_faces"] == 200_000 and w["texture"] and w["preserve_boundary"]
    assert model.version_name("decimate", p) == "MeshLab: decimazione a 200.000 facce (texture conservata)"
    p, w = model.resolve("decimate", {"target_faces": 1000, "texture": False}, {**info, "textured": True})
    assert not w["texture"] and model.version_name("decimate", p) == "MeshLab: decimazione a 1.000 facce"
    with pytest.raises(ValueError):
        model.resolve("decimate", {"target_faces": 500_000}, info)
    with pytest.raises(ValueError):
        model.resolve("decimate", {"fraction": 1.5}, info)
    p, w = model.resolve("remesh", {}, info)  # the mean edge of the model, in metres, to the units of the file
    assert p["edge_length"] == pytest.approx(0.012) and w["edge_length"] == pytest.approx(0.006)
    assert model.version_name("remesh", {"edge_length": 0.02}) == "MeshLab: rimaglia isotropa, lato 0,02 m"
    p, w = model.resolve("smooth", {"method": "taubin"}, info)
    assert w["steps"] == 10 and model.version_name("smooth", p) == "MeshLab: levigatura Taubin, 10 passi"
    p, w = model.resolve("close_holes", {"max_hole_edges": 100, "refine_edge": 0.02}, info)
    assert w["refine_edge"] == pytest.approx(0.01) and w["repair_first"]
    assert model.version_name("close_holes", p) == "MeshLab: buchi chiusi fino a 100 lati"
    p, w = model.resolve("close_holes", {"max_hole_edges": 5000, "keep_outline": False}, info)
    assert not w["select_marked"]
    assert model.version_name("close_holes", p) == "MeshLab: buchi chiusi fino a 5.000 lati, contorno compreso"
    p, w = model.resolve("repair", {"non_manifold_method": "split", "null_faces": False}, info)
    assert w["non_manifold_method"] == "split" and not w["null_faces"] and w["duplicate_faces"]
    with pytest.raises(ValueError):
        model.resolve("poisson", {}, info)


def test_where_meshlab_runs(tmp_path, monkeypatch):
    assert "Windows su ARM" in environment.unsupported("win32", "ARM64")
    assert environment.unsupported("darwin", "arm64") is None and environment.unsupported("win32", "AMD64") is None
    assert environment.unsupported("linux", "aarch64") is None and environment.unsupported("linux", "armv7l")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not environment.local_available()  # the packaged application is not a Python
    monkeypatch.delattr(sys, "frozen")
    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setenv(environment.ENV_VAR, str(tmp_path / "env"))
    assert environment.where() == (None, "") and "non è installato" in environment.status_text()
    with pytest.raises(environment.MeshLabMissing, match="non è installato"):
        environment.worker_python()
    monkeypatch.setattr(environment, "unsupported", lambda *a: "niente per questo computer")
    with pytest.raises(environment.MeshLabMissing, match="niente per questo computer"):
        environment.worker_python()


ANSWER_OK = {"python": "3.12.12", "pymeshlab": "2099.1", "numpy": "2.0", "missing_filters": [], "reads_ply": True,
             "warning": ""}
ANSWER_NO_GL = {"python": "3.12.12", "pymeshlab": "2099.1", "numpy": "2.0", "missing_filters": [], "reads_ply": False,
                "error": "Unknown format for load: ply",
                "warning": "Warning:\nUnable to load the following plugins:\n\n\t/x/libio_base.so: Cannot load library "
                           "/x/libio_base.so: (libOpenGL.so.0: cannot open shared object file: No such file or directory)"}


def _fake_python(path: Path) -> Path:
    """A "Python" that answers the check of pymeshlab with the JSON of the file ``$ANSWER``."""
    path.write_text('#!/bin/sh\ncat "$ANSWER"\n')
    path.chmod(0o755)
    return path


@pytest.mark.skipif(os.name == "nt", reason="the fake uv is a shell script")
def test_the_environment_installed_with_uv(tmp_path, monkeypatch):
    from pyarchinit_photogrammetry.core.progress import Progress

    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setattr(environment, "unsupported", lambda *a: None)
    monkeypatch.setattr(environment, "_CHECKED", {})
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER_OK))
    monkeypatch.setenv("ANSWER", str(answer))
    uv = tmp_path / "uv"
    uv.write_text("#!/bin/sh\n"
                  "if [ \"$1\" = venv ]; then\n"
                  "  for last; do :; done\n"
                  "  mkdir -p \"$last/bin\"\n"
                  "  printf '#!/bin/sh\\ncat \"$ANSWER\"\\n' > \"$last/bin/python\"\n"
                  "  chmod +x \"$last/bin/python\"\n"
                  "  echo \"Creating virtual environment at: $last\"\n"
                  "  exit 0\n"
                  "fi\n"
                  "if [ \"$1\" = pip ]; then echo 'Resolved 2 packages'; echo ' + pymeshlab==2099.1'; "
                  "[ -n \"$FAIL\" ] && exit 1; exit 0; fi\n"
                  "exit 2\n")
    uv.chmod(0o755)
    env = tmp_path / "env"
    monkeypatch.setenv(environment.ENV_VAR, str(env))
    seen = []
    assert environment.install(Progress(lambda f, m: seen.append((f, m))), uv=uv) == env
    assert environment.installed_info()["pymeshlab"] == "2099.1" and environment.is_installed()
    assert environment.worker_python() == environment.env_python(env) and environment.known_usable()
    assert environment.where() == ("environment", "2099.1") and "2099.1" in environment.status_text()
    assert seen[-1] == (1.0, "MeshLab pronto (pymeshlab 2099.1)")
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    # a failure leaves no environment that looks ready
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError, match="Installazione non riuscita"):
        environment.install(uv=uv)
    assert not environment.is_installed()
    # installed on a Linux without libOpenGL: the environment stays, the installation says what to add
    monkeypatch.delenv("FAIL")
    answer.write_text(json.dumps(ANSWER_NO_GL))
    monkeypatch.setattr(environment, "is_linux", lambda: True)
    with pytest.raises(environment.MeshLabMissing, match="sudo apt install libopengl0"):
        environment.install(uv=uv)
    assert environment.is_installed() and not environment.known_usable()
    assert "libglvnd-opengl" in environment.status_text()  # the pane says it too


@pytest.mark.skipif(os.name == "nt", reason="the fake Python is a shell script")
def test_the_check_says_what_linux_lacks(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "_CHECKED", {})
    monkeypatch.setattr(environment, "unsupported", lambda *a: None)
    python = _fake_python(tmp_path / "python")
    monkeypatch.setattr(environment, "_python", lambda: python)
    answer = tmp_path / "answer.json"
    monkeypatch.setenv("ANSWER", str(answer))
    answer.write_text(json.dumps(ANSWER_NO_GL))
    assert environment.probe(python)["error"] == "Unknown format for load: ply"
    assert environment.missing_libraries(ANSWER_NO_GL["warning"]) == ["libOpenGL.so.0"]
    text = environment.problem_text(ANSWER_NO_GL, linux=True)
    assert "non legge i modelli" in text and "(libOpenGL.so.0)" in text
    assert "«sudo apt install libopengl0»" in text and "«sudo dnf install libglvnd-opengl»" in text
    nothing_named = environment.problem_text({**ANSWER_NO_GL, "warning": ""}, linux=True)
    assert "sudo apt install libopengl0 libgl1" in nothing_named and "libglvnd-opengl libglvnd-glx" in nothing_named
    assert environment.problem_text(ANSWER_NO_GL, linux=False).endswith("Librerie mancanti: libOpenGL.so.0.")
    lacking = {**ANSWER_OK, "missing_filters": ["meshing_close_holes"]}
    assert not environment.usable(lacking) and "meshing_close_holes" in environment.problem_text(lacking, linux=False)
    # before an operation: a failure stops it with the hint, and is looked at again the next time
    calls = []
    real = environment.probe
    monkeypatch.setattr(environment, "probe", lambda p: calls.append(p) or real(p))
    monkeypatch.setattr(environment, "is_linux", lambda: True)
    with pytest.raises(environment.MeshLabMissing, match="sudo apt install libopengl0"):
        environment.worker_python()
    answer.write_text(json.dumps(ANSWER_OK))  # libopengl0 installed meanwhile
    assert environment.worker_python() == python and environment.worker_python() == python
    assert len(calls) == 2 and environment.known_usable()  # a working Python is checked once a session
    # a Python that dies before answering
    (tmp_path / "broken").write_text("#!/bin/sh\necho 'Segmentation fault' >&2\nexit 139\n")
    (tmp_path / "broken").chmod(0o755)
    broken = environment.probe(tmp_path / "broken")
    assert "codice 139" in broken["error"] and "Segmentation fault" in broken["warning"]
    assert not environment.usable(broken)


def test_meshlab_without_its_modules_says_what_to_install(tmp_path, fake):
    fake.setenv("PPG_MESH_TOOLS_FAKE", "nomodules")
    fake.setattr(environment, "is_linux", lambda: True)
    with pytest.raises(runner.WorkerError) as err:
        runner.run(_job(tmp_path, grid_mesh(), [{"op": "decimate", "target_faces": 500}]), tmp_path)
    assert "Unknown format for load: ply" in str(err.value)
    assert "(libOpenGL.so.0)" in str(err.value) and "sudo apt install libopengl0" in str(err.value)
    assert "libOpenGL.so.0: cannot open shared object file" in err.value.details  # MeshLab's lines in the details
    fake.setattr(environment, "is_linux", lambda: False)
    assert runner.hint_for("Unknown format for load: ply") == ("MeshLab non ha caricato i suoi moduli: installa di "
                                                               "nuovo MeshLab (pymeshlab).")
    fake.setattr(environment, "is_linux", lambda: True)  # a filter of a module that did not load
    assert "libopengl0 libgl1" in runner.hint_for("'MeshSet' object has no attribute 'meshing_close_holes'")
    assert "Ripara la topologia" in runner.hint_for("Mesh has some not 2-manifold edges")


def test_command(tmp_path, fake, capsys):
    import pyarchinit_photogrammetry as ppg
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.plugins import Context

    from ppg_mesh_tools import MeshToolsPlugin

    doc, ch = project(tmp_path)
    app.document = None
    plugin = MeshToolsPlugin()
    plugin.activate(Context("mesh-tools", tmp_path / "plugin"))
    try:
        assert plugin.command("decimate", str(doc.path), "--faces", "800") == 0
        out = capsys.readouterr().out
        assert "Modello decimato" in out and "Nuova versione: MeshLab: decimazione a 800 facce (texture conservata)" in out
        reopened = ppg.Document.open(doc.path).chunk  # the project was saved with the new version
        assert versions.active_info(reopened, "model")["name"] == "MeshLab: decimazione a 800 facce (texture conservata)"
        assert len(versions.versions(reopened, "model")) == 1
        assert plugin.command("measure", str(doc.path), "--json") == 0
        text = capsys.readouterr().out
        measures = json.loads(text[text.index("{\n"):])
        assert measures["faces"] == 800 and measures["unit"] == "m"
        assert plugin.command("measure", str(doc.path)) == 0 and "Area della superficie" in capsys.readouterr().out
        assert plugin.command("decimate", str(doc.path), "--faces", "99999") == 1
        assert "niente da decimare" in capsys.readouterr().out
        assert plugin.command("smooth", str(doc.path), "--method", "hc", "--overwrite", "--json") == 0
        text = capsys.readouterr().out
        report = json.loads(text[text.index("{\n"):])
        assert report["operation"] == "smooth" and report["version"] is None
        assert len(versions.versions(ppg.Document.open(doc.path).chunk, "model")) == 1
        assert plugin.command("repair", str(doc.path)) == 0
        assert "Nessun difetto di topologia" in capsys.readouterr().out
        app.document = None
        assert plugin.command("decimate") == 1 and "no project" in capsys.readouterr().out
        assert plugin.command("status") in (0, 1) and "MeshLab" in capsys.readouterr().out
    finally:
        app.document = None


def test_the_pane_in_the_window(tmp_path, fake):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from ppg_mesh_tools import MeshToolsPlugin
    from pyarchinit_photogrammetry.app import app
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, str(tmp_path / "settings"))
    qapp = QApplication.instance() or QApplication([])
    from pyarchinit_photogrammetry.gui import main_window as mwmod

    doc, ch = project(tmp_path)
    app.document = doc
    fake.setenv("PPG_NO_PLUGINS", "1")  # only the instance of the test, not the one installed in this Python
    w = mwmod.MainWindow(doc, autoconnect=False)
    w.runner.show_dialog = False
    plugin = MeshToolsPlugin()
    asked = []
    fake.setattr(dialogs.ParamDialog, "ask", classmethod(
        lambda cls, parent, title, fields, intro="": asked.append((title, [f.name for f in fields], intro))
        or {"target_faces": 700, "fraction": 0.5, "texture": True, "preserve_boundary": True, "preserve_topology": True,
            "quality": 0.3, "planar": False, "preserve_normal": False, "keep": "keep", "version_name": ""}))
    try:
        plugin.activate(Context("mesh-tools", tmp_path / "plugin", window=w))
        pane = plugin.pane
        assert pane is not None and w.left_tabs.indexOf(pane) >= 0
        assert "facce" in pane.l_model.text() and "texture (1 pagina)" in pane.l_model.text()
        pane.buttons["decimate"].click()
        assert w.runner.busy and not pane.buttons["measure"].isEnabledTo(pane)  # one step at a time
        w.runner.wait()
        qapp.processEvents()
        assert pane.buttons["measure"].isEnabledTo(pane)
        assert asked and asked[0][0] == "MeshLab: Decimazione" and "texture" in asked[0][1] and "keep" in asked[0][1]
        assert len(read_mesh(ch.product_path("model")).faces) == 700
        assert versions.active_info(ch, "model")["name"] == "MeshLab: decimazione a 700 facce (texture conservata)"
        assert w.model_view.kind == "model_textured" and w.tabs.currentWidget() is w.model_view
        assert "Modello decimato" in pane.l_last.text() and pane.table.rowCount() > 5
        import pyarchinit_photogrammetry as ppg

        assert len(versions.versions(ppg.Document.open(doc.path).chunk, "model")) == 1  # saved with its versions
        assert "700 facce" in pane.l_model.text()
        pane.buttons["measure"].click()
        w.runner.wait()
        qapp.processEvents()
        assert pane.table.item(2, 0).text() == "Area della superficie"
        # every dialog built for real and accepted as it opens: the defaults go through to the worker
        said = []
        fake.setattr(dialogs, "message", lambda parent, kind, title, text, *a, **k: said.append(text))

        def accept(cls, parent, title, fields, intro=""):
            d = cls(title, fields, parent, intro)
            try:
                return d.values()
            finally:
                d.deleteLater()

        fake.setattr(dialogs.ParamDialog, "ask", classmethod(accept))
        for op in ("remesh", "smooth", "close_holes", "repair"):
            pane.buttons[op].click()
            w.runner.wait()
            qapp.processEvents()
        names = [v["name"] for v in versions.versions(ch, "model")] + [versions.active_info(ch, "model")["name"]]
        assert names[1] == "MeshLab: decimazione a 700 facce (texture conservata)"
        assert names[2].startswith("MeshLab: rimaglia isotropa, lato 0,0") and names[2].endswith(" m")  # the mean edge
        assert names[3:] == ["MeshLab: levigatura Taubin, 10 passi"]
        # the fake decimation kept a strip of faces without the hole: only its outline, which stays open; nothing to
        # repair: said, no version
        assert said == ["Nessun buco fino a 30 lati da chiudere: il modello non è cambiato.",
                        "Nessun difetto di topologia: il modello non è cambiato."]
        # the first operation of a session checks MeshLab first: on a Linux without libOpenGL it stops there
        fake.setattr(environment, "known_usable", lambda: False)
        fake.setattr(environment, "check", lambda progress=None, **k: ANSWER_NO_GL)
        fake.setattr(environment, "is_linux", lambda: True)
        said.clear()
        pane.buttons["decimate"].click()
        w.runner.wait()
        qapp.processEvents()
        assert len(said) == 1 and "sudo apt install libopengl0" in said[0]
        assert versions.active_info(ch, "model")["name"] == "MeshLab: levigatura Taubin, 10 passi"  # nothing ran
        # MeshLab missing: the installation is offered first; refused, nothing runs
        fake.setattr(environment, "where", lambda: (None, ""))
        offers = []
        fake.setattr(dialogs, "choice", lambda *a, **k: offers.append(a[3]))
        pane.buttons["repair"].click()
        assert offers and "non è installato" in offers[0] and not w.runner.busy
        plugin.deactivate()
        assert w.left_tabs.indexOf(pane) < 0
    finally:
        app.document = None
        w.setWindowModified(False)
        w.close()


def test_texts_in_the_language_of_the_interface(tmp_path, fake):
    from pyarchinit_photogrammetry.gui import i18n

    doc, ch = project(tmp_path)
    before = i18n.current()
    try:
        i18n.set_current("en")
        report = model.apply(ch, "decimate", {"target_faces": 1200})
        text = model.report_text(report)
        assert text.startswith("Model decimated: ") and text.endswith("→ 1,200 faces; texture kept")
        rows = dict(model.measure_rows(report["measures"]))
        assert rows["Closed (watertight)"] == "no" and rows["Faces"] == "1,200"
        assert "is not installed" in environment.status_text() or "MeshLab: pymeshlab" in environment.status_text()
    finally:
        i18n.set_current(before)
    # the name of the version is stored in Italian, as data: the project reads the same in both languages
    assert report["name"] == "MeshLab: decimazione a 1.200 facce (texture conservata)"
    assert model.report_text(report).endswith("→ 1.200 facce; texture conservata")
