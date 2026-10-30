"""The worker in a process of its own through the runner: the protocol, the progress in the language
of the interface, failures and crashes with their reason, cancelling - and in every one of those
cases no output left behind. With the fake worker (numpy) always; with the real one and Landlab when
a Python with Landlab is known (``PPG_TEST_LANDLAB``, or this Python when Landlab is installed here).
And where Landlab runs: the environment installed with uv (a fake uv), the check of a real one."""
from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the runner needs the program")

from landscape_samples import CELL, landlab_python, tell  # noqa: E402
from ppg_landscape import environment, runner  # noqa: E402
from pyarchinit_photogrammetry.core.progress import Cancelled, Progress  # noqa: E402

FAKE = Path(__file__).with_name("fake_worker.py")
LANDLAB = landlab_python()
real = pytest.mark.skipif(LANDLAB is None, reason="no Python with Landlab (PPG_TEST_LANDLAB)")


def _job(tmp_path: Path, n: int = 30, **extra) -> dict:
    z, _r = tell(n=n)
    np.save(tmp_path / "elevation.npy", z)
    job = {"protocol": 1, "elevation": str(tmp_path / "elevation.npy"), "output": str(tmp_path / "run"),
           "cell": CELL, "years": 10, "snapshots": [5], "boundary": "closed", "diffusion": {"D": 0.01}, "fluvial": None}
    job.update(extra)
    return job


def _nothing_left(tmp_path: Path) -> None:
    assert not (tmp_path / "run").exists() and not (tmp_path / "run.partial").exists()


@pytest.fixture
def fake(monkeypatch):
    def mode(name: str = "") -> None:
        monkeypatch.setenv("PPG_LANDSCAPE_FAKE", name)

    mode("")
    return mode


def test_the_worker_speaks_while_it_runs(tmp_path, fake):
    fake("noise")
    seen = []
    result = runner.run(_job(tmp_path), tmp_path / "work", Progress(lambda f, m: seen.append((f, m))),
                        python=Path(sys.executable), script=FAKE)
    out = tmp_path / "run"
    assert result["output"] == str(out) and result["versions"]["landlab"] == "0.0-fake"
    assert result["files"]["snapshots"] == {"5": "snap_5.npy"} and (out / "snap_5.npy").exists()
    final, dz = np.load(out / "final.npy"), np.load(out / "dz.npy")
    assert np.array_equal(dz, final - np.load(tmp_path / "elevation.npy"))
    messages = [m for _f, m in seen]
    assert messages[0] == "Landlab: avvio" and "Landlab: anno 5 di 10" in messages
    assert "Landlab: scrittura dei risultati" in messages
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    assert json.loads((tmp_path / "work" / "job.json").read_text())["years"] == 10


def test_failures_and_crashes_leave_nothing(tmp_path, fake):
    fake("fail")
    with pytest.raises(runner.WorkerError, match="riduci il passo") as err:
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    assert "became NaN" in str(err.value) and "FloatingPointError" in err.value.details
    _nothing_left(tmp_path)
    fake("crash")
    with pytest.raises(runner.WorkerError, match="chiuso inaspettatamente durante «anno 5 di 10»"):
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    _nothing_left(tmp_path)
    fake("crash_save")
    with pytest.raises(runner.WorkerError, match="nessun risultato"):
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    _nothing_left(tmp_path)  # the partial folder the worker left is gone too


def test_a_lake_without_an_outlet_stops_the_run(tmp_path, fake):
    import time

    fake("lake")
    t0 = time.monotonic()
    with pytest.raises(runner.WorkerError, match="non trova uno sbocco") as err:
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    assert time.monotonic() - t0 < 30 and runner.STUCK_LAKE in err.value.details
    _nothing_left(tmp_path)


def test_an_existing_output_is_refused_and_kept(tmp_path, fake):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "mine.txt").write_text("keep")
    with pytest.raises(runner.WorkerError, match="exists already"):
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    assert (tmp_path / "run" / "mine.txt").read_text() == "keep"


def test_cancelling_stops_the_worker_at_once(tmp_path, fake):
    import time

    fake("sleep")
    progress = Progress(lambda f, m: None)
    threading.Timer(1.5, progress.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        runner.run(_job(tmp_path, years=60), tmp_path / "work", progress, python=Path(sys.executable), script=FAKE)
    assert time.monotonic() - t0 < 10
    _nothing_left(tmp_path)


def test_hints():
    assert "cella più grande" in runner.hint_for("MemoryError: Unable to allocate 3.1 GiB")
    assert "installa di nuovo" in runner.hint_for("No module named 'landlab'")
    assert runner.hint_for("something else") == ""
    assert runner.phase_text({"phase": "run", "year": 2.5, "years": 1000}) == "Landlab: anno 2,5 di 1.000"


# ---------------------------------------------------------------------- the real worker
@real
def test_the_real_worker_through_the_runner(tmp_path):
    seen = []
    job = _job(tmp_path, n=40, years=20, boundary="open", fluvial={"runoff": 0.5})
    result = runner.run(job, tmp_path / "work", Progress(lambda f, m: seen.append(m)), python=LANDLAB)
    out = tmp_path / "run"
    major, minor = (int(x) for x in result["versions"]["landlab"].split(".")[:2])
    assert (major, minor) >= (2, 9) and result["balance"]["eroder"] == "space"
    assert sorted(p.name for p in out.iterdir()) == ["area.npy", "balance.json", "dz.npy", "final.npy", "initial.npy",
                                                     "snap_5.npy"]
    z = np.load(tmp_path / "elevation.npy")
    assert np.array_equal(np.load(out / "dz.npy"), np.load(out / "final.npy") - z)
    assert any(m.startswith("Landlab: anno ") for m in seen) and "Landlab: area drenata" in seen


@real
def test_cancelling_the_real_worker_leaves_no_output(tmp_path):
    started = threading.Event()

    def watch(f, m):
        if m.startswith("Landlab: anno"):
            started.set()

    progress = Progress(watch)
    threading.Thread(target=lambda: started.wait(60) and progress.cancel(), daemon=True).start()
    job = _job(tmp_path, n=100, years=5000, dt=1.0, boundary="open", fluvial={"runoff": 0.5})
    with pytest.raises(Cancelled):
        runner.run(job, tmp_path / "work", progress, python=LANDLAB)
    assert started.is_set()
    _nothing_left(tmp_path)


@real
def test_the_check_of_a_real_landlab():
    info = environment.probe(LANDLAB)
    assert environment.usable(info), environment.problem_text(info)
    assert info["eroders"][0] == "SpaceLargeScaleEroder" and info["richdem"]


# ---------------------------------------------------------------------- where Landlab runs
def test_where_landlab_runs(tmp_path, monkeypatch):
    assert "Windows su ARM" in environment.unsupported("win32", "ARM64")
    assert environment.unsupported("darwin", "arm64") is None and environment.unsupported("win32", "AMD64") is None
    assert environment.unsupported("linux", "aarch64") is None and environment.unsupported("linux", "armv7l")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not environment.local_available()  # the packaged application is not a Python
    monkeypatch.delattr(sys, "frozen")
    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setenv(environment.ENV_VAR, str(tmp_path / "env"))
    assert environment.where() == (None, "") and "non è installato" in environment.status_text()
    with pytest.raises(environment.LandlabMissing, match="non è installato"):
        environment.worker_python()
    monkeypatch.setattr(environment, "unsupported", lambda *a: "niente per questo computer")
    with pytest.raises(environment.LandlabMissing, match="niente per questo computer"):
        environment.worker_python()


ANSWER_OK = {"python": "3.12.12", "landlab": "2.99.0", "numpy": "2.0", "scipy": "1.0", "missing": [],
             "eroders": ["SpaceLargeScaleEroder", "Space", "ErosionDeposition"], "richdem": "2.2.0"}
ANSWER_OLD = {"python": "3.12.12", "landlab": "2.0.0", "missing": ["DepressionFinderAndRouter"], "eroders": []}


@pytest.mark.skipif(os.name == "nt", reason="the fake uv is a shell script")
def test_the_environment_installed_with_uv(tmp_path, monkeypatch):
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
                  "if [ \"$1\" = pip ]; then echo 'Resolved 40 packages'; echo ' + landlab==2.99.0'; "
                  "[ -n \"$FAIL\" ] && exit 1; exit 0; fi\n"
                  "exit 2\n")
    uv.chmod(0o755)
    env = tmp_path / "env"
    monkeypatch.setenv(environment.ENV_VAR, str(env))
    seen = []
    assert environment.install(Progress(lambda f, m: seen.append((f, m))), uv=uv) == env
    assert environment.installed_info()["landlab"] == "2.99.0" and environment.is_installed()
    assert environment.worker_python() == environment.env_python(env) and environment.known_usable()
    assert environment.where() == ("environment", "2.99.0") and "2.99.0" in environment.status_text()
    assert seen[-1] == (1.0, "Landlab pronto (2.99.0)")
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    # a failure leaves no environment that looks ready
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError, match="Installazione non riuscita"):
        environment.install(uv=uv)
    assert not environment.is_installed()
    # a Landlab too old: installed, and the pane says what is wrong
    monkeypatch.delenv("FAIL")
    answer.write_text(json.dumps(ANSWER_OLD))
    with pytest.raises(environment.LandlabMissing, match="Landlab 2.9 o successivo"):
        environment.install(uv=uv)
    assert environment.is_installed() and not environment.known_usable()
    assert "DepressionFinderAndRouter" in environment.status_text()


def test_intel_macs_before_macos_15():
    assert "macOS 15" in environment.unsupported("darwin", "x86_64", "14.6")
    assert environment.unsupported("darwin", "x86_64", "15.1") is None
    assert environment.unsupported("darwin", "arm64", "14.6") is None


@pytest.mark.skipif(os.name == "nt", reason="the fake uv is a shell script")
def test_uv_without_its_cache_tries_again_with_the_plugins(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setattr(environment, "unsupported", lambda *a: None)
    monkeypatch.setattr(environment, "_CHECKED", {})
    monkeypatch.setattr(environment, "_DATA", tmp_path / "data")
    monkeypatch.delenv("UV_CACHE_DIR", raising=False)
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER_OK))
    monkeypatch.setenv("ANSWER", str(answer))
    uv = tmp_path / "uv"
    # uv as on a computer where a program run as root left files in ~/.cache/uv
    uv.write_text("#!/bin/sh\n"
                  "if [ \"$1\" = venv ]; then\n"
                  "  for last; do :; done\n"
                  "  mkdir -p \"$last/bin\"\n"
                  "  printf '#!/bin/sh\\ncat \"$ANSWER\"\\n' > \"$last/bin/python\"\n"
                  "  chmod +x \"$last/bin/python\"\n"
                  "  exit 0\n"
                  "fi\n"
                  "if [ -z \"$UV_CACHE_DIR\" ]; then\n"
                  "  echo 'error: Failed to write to the client cache'\n"
                  "  echo '  Caused by: Permission denied (os error 13) at path \"/Users/x/.cache/uv/wheels\"'\n"
                  "  exit 2\n"
                  "fi\n"
                  "mkdir -p \"$UV_CACHE_DIR\" && echo cached > \"$UV_CACHE_DIR/wheel\"\n"
                  "echo ' + landlab==2.99.0'\n"
                  "exit 0\n")
    uv.chmod(0o755)
    env = tmp_path / "env"
    monkeypatch.setenv(environment.ENV_VAR, str(env))
    seen = []
    assert environment.install(Progress(lambda f, m: seen.append((f, m))), uv=uv) == env
    said = [m for _f, m in seen]
    assert any("permesso negato" in m and str(tmp_path / "data" / "uv-cache") in m for m in said)
    assert environment.is_installed() and not (tmp_path / "data" / "uv-cache").exists()  # used, then removed
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    # another failure is not retried
    uv.write_text("#!/bin/sh\necho 'error: no matching distribution'\nexit 1\n")
    with pytest.raises(RuntimeError, match="no matching distribution"):
        environment.install(uv=uv)
