"""The worker in a process of its own through the runner, with the fake worker (the engine's meshes instead of
CadQuery): the protocol, the progress in the language of the interface, failures and crashes with their reason,
cancelling - and in every one of those cases no output left behind. Where CadQuery runs: the environment's rules."""
from __future__ import annotations

import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the runner needs the program")

from ppg_architecture import engine, environment, runner  # noqa: E402
from pyarchinit_photogrammetry.core.progress import Cancelled, Progress  # noqa: E402

FAKE = Path(__file__).with_name("fake_worker.py")


def _job(tmp_path: Path) -> dict:
    curve = np.column_stack([np.linspace(0.3, 0.25, 20), np.linspace(0, 2, 20)])
    return {"protocol": 1, "kind": "column", "output": str(tmp_path / "solid"), "tolerance": 0.0005,
            "placement": engine.frame_matrix([1.0, 2.0, 3.0], [1, 0, 0], [0, 1, 0], [0, 0, 1]).tolist(),
            "column": {"profile": curve.tolist(), "smooth": True}}


def _nothing_left(tmp_path: Path) -> None:
    assert not (tmp_path / "solid").exists() and not (tmp_path / "solid.partial").exists()


@pytest.fixture
def fake(monkeypatch):
    def mode(name: str = "") -> None:
        monkeypatch.setenv("PPG_ARCHITECTURE_FAKE", name)

    mode("")
    return mode


def test_the_worker_speaks_while_it_runs(tmp_path, fake):
    fake("noise")
    seen = []
    result = runner.run(_job(tmp_path), tmp_path / "work", Progress(lambda f, m: seen.append((f, m))),
                        python=Path(sys.executable), script=FAKE)
    out = tmp_path / "solid"
    assert result["output"] == str(out) and result["versions"]["cadquery"] == "0.0-fake" and result["valid"]
    assert (out / "solid.step").exists() and (out / "solid.stl").exists()
    curve = np.asarray(_job(tmp_path)["column"]["profile"])
    assert result["volume"] == pytest.approx(engine.revolved_volume(curve), rel=0.001)
    assert result["bbox"][2] == pytest.approx(3.0) and result["bbox"][5] == pytest.approx(5.0)  # placed
    messages = [m for _f, m in seen]
    assert messages[0] == "CadQuery: avvio" and "CadQuery: scrittura dello STEP" in messages


def test_failures_and_crashes_leave_nothing(tmp_path, fake):
    fake("fail")
    with pytest.raises(runner.WorkerError, match="non è valido|not valid") as err:
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    assert "La sezione o il percorso si intrecciano" in str(err.value)
    _nothing_left(tmp_path)
    fake("crash")
    with pytest.raises(runner.WorkerError, match="chiuso inaspettatamente durante «scrittura dello STEP»"):
        runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable), script=FAKE)
    _nothing_left(tmp_path)


def test_cancelling_kills_the_worker_and_leaves_nothing(tmp_path, fake):
    fake("sleep")
    progress = Progress(lambda f, m: None)
    threading.Timer(1.5, progress.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        runner.run(_job(tmp_path), tmp_path / "work", progress, python=Path(sys.executable), script=FAKE)
    assert time.monotonic() - t0 < 10
    _nothing_left(tmp_path)


def test_where_cadquery_may_run():
    assert environment.unsupported("win32", "ARM64") and "Windows su ARM" in environment.unsupported("win32", "ARM64")
    assert environment.unsupported("darwin", "arm64") is None
    assert environment.unsupported("linux", "aarch64") is None
    assert environment.unsupported("win32", "AMD64") is None
    cmds = environment.commands(Path("/uv"), Path("/env/bin/python"))
    assert any("--no-deps" in c for c in cmds)
    flat = " ".join(" ".join(c) for c in cmds)
    # OpenCascade's module links VTK's libraries: VTK stays; casadi, numba, scipy, matplotlib, trame do not come
    assert "vtk==9.6.2" in flat and "cadquery-ocp==7.9.3.1.1" in flat
    assert not any(name in flat for name in ("casadi", "numba", "scipy", "matplotlib", "trame"))


def test_the_check_of_a_python_without_cadquery():
    if environment.local_available():
        pytest.skip("CadQuery is installed in this Python")
    info = environment.probe(Path(sys.executable))
    assert not environment.usable(info) and "CadQuery non si importa" in environment.problem_text(info)
