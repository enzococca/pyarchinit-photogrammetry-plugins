"""The worker of the implicit model in a process of its own through the runner: the protocol, the
progress in the language of the interface, failures and crashes with their reason, cancelling - and
in every one of those cases no output left behind. With the fake worker (numpy) always; with the
real one when a Python with LoopStructural is known (``PPG_TEST_LOOP``, or this Python when it has
it). And where LoopStructural runs: the environment installed with uv (a fake uv), the check of a real one."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyarchinit_photogrammetry", reason="the runner needs the program")

from implicit_samples import Ground, loop_python  # noqa: E402
from ppg_geoarchaeology import environment, implicit_runner as runner  # noqa: E402
from pyarchinit_photogrammetry.core.progress import Cancelled, Progress  # noqa: E402

FAKE = Path(__file__).with_name("implicit_fake_worker.py")
LOOP = loop_python()
real = pytest.mark.skipif(LOOP is None, reason="no Python with LoopStructural (PPG_TEST_LOOP)")
PACKAGE = Path(runner.__file__).resolve().parent


def _job(tmp_path: Path, **extra) -> dict:
    ground = Ground(20, 1.0)
    np.save(tmp_path / "dem.npy", np.full((20, 20), 10.0))
    job = {"protocol": 1, "dem": str(tmp_path / "dem.npy"), "output": str(tmp_path / "run"), "grid": ground.grid,
           "zmin": 0.0, "zmax": 12.0, "dz": 0.5, "series": [{"name": "s", "space": "elevation", "units": ["A"]}],
           "contacts": [["A", 5.0, 5.0, 6.0, 1.0, "S1"]], "orientations": [],
           "column": [{"unit": "T", "kind": "layer"}, {"unit": "A", "kind": "layer"}], "options": {}}
    job.update(extra)
    return job


def _nothing_left(tmp_path: Path) -> None:
    assert not (tmp_path / "run").exists() and not (tmp_path / "run.partial").exists()


@pytest.fixture
def fake(monkeypatch):
    def mode(name: str = "") -> None:
        monkeypatch.setenv("PPG_GEOARCH_FAKE", name)

    mode("")
    return mode


def _run(tmp_path, progress=None, **extra):
    return runner.run(_job(tmp_path, **extra), tmp_path / "work", progress, python=Path(sys.executable), script=FAKE)


def test_a_job_goes_there_and_comes_back(tmp_path, fake):
    fake("noise")
    seen = []
    result = _run(tmp_path, Progress(lambda f, m: seen.append((f, m))))
    out = tmp_path / "run"
    assert result["output"] == str(out) and result["versions"]["loopstructural"] == "0.0-fake"
    assert result["units"] == ["A"] and result["volumes"] == {"A": 1.5} and result["loo"]["A"]["n"] == 4
    assert result["info"]["warnings"][0]["code"] == "edge_estimated" and result["info"]["nan"] is None
    assert result["files"]["tops"] == {"A": "top_0.npy"} and result["files"]["variance"] == {"A": "variance_0.npy"}
    assert np.load(out / "labels.npy").shape == (20, 20, 24)
    assert np.load(out / result["files"]["solids"]["A"][0]).shape == (4, 3)
    assert np.load(out / result["files"]["solids"]["A"][1]).dtype == np.int32
    assert json.loads((out / "model.json").read_text())["units"] == ["A"]
    messages = [m for _f, m in seen]
    assert messages[0] == "LoopStructural: avvio" and "LoopStructural: calcolo dei campi" in messages
    assert "LoopStructural: scrittura dei risultati" in messages
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    assert json.loads((tmp_path / "work" / "job.json").read_text())["dz"] == 0.5


def test_failures_and_crashes_leave_nothing(tmp_path, fake):
    fake("fail")
    with pytest.raises(runner.WorkerError, match="the column has no unit") as err:
        _run(tmp_path)
    assert "ValueError" in err.value.details and err.value.phase == "load"
    _nothing_left(tmp_path)
    fake("memory")
    with pytest.raises(runner.WorkerError, match="voxel più grandi"):
        _run(tmp_path)
    _nothing_left(tmp_path)
    fake("crash")
    with pytest.raises(runner.WorkerError, match="chiuso inaspettatamente durante «calcolo dei campi»"):
        _run(tmp_path)
    _nothing_left(tmp_path)
    fake("crash_save")
    with pytest.raises(runner.WorkerError, match="nessun risultato"):
        _run(tmp_path)
    _nothing_left(tmp_path)  # the partial folder the worker left is gone too


def test_an_existing_output_is_refused_and_kept(tmp_path, fake):
    (tmp_path / "run").mkdir()
    (tmp_path / "run" / "mine.txt").write_text("keep")
    with pytest.raises(runner.WorkerError, match="exists already"):
        _run(tmp_path)
    assert (tmp_path / "run" / "mine.txt").read_text() == "keep"


def test_cancelling_stops_the_worker_at_once(tmp_path, fake):
    import time

    fake("sleep")
    progress = Progress(lambda f, m: None)
    threading.Timer(1.5, progress.cancel).start()
    t0 = time.monotonic()
    with pytest.raises(Cancelled):
        _run(tmp_path, progress)
    assert time.monotonic() - t0 < 10
    _nothing_left(tmp_path)


def test_hints():
    assert "voxel più grandi" in runner.hint_for("MemoryError: Unable to allocate 3.1 GiB")
    assert "installa di nuovo" in runner.hint_for("No module named 'LoopStructural'")
    assert runner.hint_for("something else") == ""
    assert runner.parse("nothing here") is None and runner.parse('@ppg-geoarch {"type": "hello"}') == {"type": "hello"}


def test_the_stub_of_pyvista_comes_before_the_engine(tmp_path):
    """The worker beside an engine that reports what it finds when it is imported."""
    shutil.copy(PACKAGE / "implicit_worker.py", tmp_path / "implicit_worker.py")
    (tmp_path / "implicit_engine.py").write_text(
        "import sys\n"
        "seen = sys.modules.get('pyvista')\n"
        "stub = seen is not None and getattr(seen, '__file__', None) is None and not hasattr(seen, 'Plotter')\n"
        "def versions():\n"
        "    return {'stub': stub}\n"
        "class R:\n"
        "    units, tops, variance, solids, volumes, loo, info = [], {}, {}, {}, {}, {}, {}\n"
        "    labels = None\n"
        "def run(job, dem, progress):\n"
        "    import numpy as np\n"
        "    r = R()\n"
        "    r.labels = np.zeros((1, 1, 1), dtype=np.int16)\n"
        "    return r\n", encoding="utf-8")
    result = runner.run(_job(tmp_path), tmp_path / "work", python=Path(sys.executable),
                        script=tmp_path / "implicit_worker.py")
    assert result["versions"]["stub"] is True
    source = (PACKAGE / "implicit_worker.py").read_text(encoding="utf-8")
    assert source.index('sys.modules["pyvista"]') < source.index("import importlib.util")


def test_the_program_never_imports_the_implicit_modules():
    code = ("import sys, ppg_geoarchaeology\n"
            "print(ppg_geoarchaeology.__file__)\n"
            "bad = [m for m in sys.modules if 'implicit' in m"
            " or m.split('.')[0] in ('LoopStructural', 'loop_common', 'loop_interpolation')]\n"
            "print(bad)\n")
    env = dict(os.environ, PYTHONPATH=str(PACKAGE.parent))
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, close_fds=False)
    assert out.returncode == 0, out.stderr
    lines = out.stdout.strip().splitlines()
    assert lines[0] == str(PACKAGE / "__init__.py") and lines[-1] == "[]"


# ---------------------------------------------------------------------- the real worker
def _real_job(tmp_path: Path, **options) -> dict:
    ground = Ground(40, 1.0)
    dem = np.full((40, 40), 10.0)
    np.save(tmp_path / "dem.npy", dem)
    holes = ground.holes(4, 0.1)
    contacts = []
    for i, (x, y) in enumerate(holes):
        contacts.append(["B", x, y, 7.0 + 0.02 * x, 1.0, f"S{i}"])
        contacts.append(["C", x, y, 4.0 + 0.02 * x, 1.0, f"S{i}"])
    return {"protocol": 1, "dem": str(tmp_path / "dem.npy"), "output": str(tmp_path / "run"), "grid": ground.grid,
            "zmin": 0.0, "zmax": 12.0, "dz": 0.5, "series": [{"name": "s", "space": "elevation", "units": ["B", "C"]}],
            "contacts": contacts, "orientations": [],
            "column": [{"unit": u, "kind": "layer"} for u in ("A", "B", "C")], "options": options}


@real
def test_the_real_worker_through_the_runner(tmp_path):
    seen = []
    result = runner.run(_real_job(tmp_path, nelements=4000, folds=3), tmp_path / "work",
                        Progress(lambda f, m: seen.append(m)), python=LOOP)
    out = tmp_path / "run"
    assert result["versions"]["loopstructural"].startswith("1.10") and result["units"] == ["A", "B", "C"]
    assert (out / "model.json").exists() and sorted(result["loo"]) == ["B", "C"]
    labels = np.load(out / "labels.npy")
    assert labels.shape == (40, 40, 24) and set(np.unique(labels)) == {-1, 0, 1, 2}
    assert np.load(out / result["files"]["tops"]["B"]).shape == (40, 40)
    assert "LoopStructural: validazione incrociata" in seen and "LoopStructural: solidi delle unità" in seen


@real
def test_cancelling_the_real_worker_leaves_no_output(tmp_path):
    started = threading.Event()

    def watch(f, m):
        if m.endswith("calcolo dei campi"):
            started.set()

    progress = Progress(watch)
    threading.Thread(target=lambda: started.wait(60) and progress.cancel(), daemon=True).start()
    with pytest.raises(Cancelled):
        runner.run(_real_job(tmp_path, nelements=60000, folds=12), tmp_path / "work", progress, python=LOOP)
    assert started.is_set()
    _nothing_left(tmp_path)


@real
def test_a_real_pit_through_the_worker(tmp_path):
    job = _real_job(tmp_path, nelements=8000)
    job["column"].insert(0, {"unit": "F", "kind": "fill"})
    job["column"].insert(1, {"unit": "P", "kind": "cut"})
    job["fills"] = {"F": "P"}
    inside = [[20.0, 20.0], [22.0, 20.0], [18.0, 20.0]]
    grid = job["grid"]
    xs = grid["x0"] + (np.arange(grid["width"]) + 0.5) * grid["cell"]
    ys = grid["y0"] - (np.arange(grid["height"]) + 0.5) * grid["cell"]
    outline = tmp_path / "outline.npy"
    np.save(outline, np.hypot(*np.meshgrid(xs - 20, ys - 20)) < 5.0, allow_pickle=False)
    job["cuts"] = {"P": {"base": [[x, y, 6.0] for x, y in inside], "outline": str(outline),
                         "inside": inside, "outside": [[5.0, 5.0], [35.0, 35.0]]}}
    result = runner.run(job, tmp_path / "work", python=LOOP)
    cut = result["info"]["cuts"]["P"]
    assert result["volumes"]["F"] > 0 and cut["edge"] == "drawn" and cut["profile"] == "bowl"
    assert set(cut["volume_range"]) == {"flat", "bowl", "v"}


@real
def test_the_check_of_a_real_loopstructural():
    info = environment.probe(LOOP)
    assert environment.usable(info), environment.problem_text(info)
    assert info["loopstructural"].startswith("1.10")


# ---------------------------------------------------------------------- where LoopStructural runs
def test_where_loopstructural_runs(tmp_path, monkeypatch):
    for system, machine in (("darwin", "arm64"), ("darwin", "x86_64"), ("win32", "AMD64"), ("win32", "ARM64"),
                            ("linux", "x86_64"), ("linux", "aarch64")):
        assert environment.unsupported(system, machine) is None
    assert environment.unsupported("linux", "armv7l")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert not environment.local_available()  # the packaged application is not a Python
    monkeypatch.delattr(sys, "frozen")
    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setenv(environment.ENV_VAR, str(tmp_path / "env"))
    assert environment.where() == (None, "") and "300 MB" in environment.status_text()
    with pytest.raises(environment.LoopMissing, match="non è installato"):
        environment.worker_python()
    monkeypatch.setattr(environment, "unsupported", lambda *a: "niente per questo computer")
    with pytest.raises(environment.LoopMissing, match="niente per questo computer"):
        environment.worker_python()


def test_the_environment_goes_in_the_plugins_folder(tmp_path, monkeypatch):
    class Context:
        data_dir = tmp_path / "data"

    monkeypatch.delenv(environment.ENV_VAR, raising=False)
    monkeypatch.setattr(environment, "_DATA", None)
    with pytest.raises(environment.LoopMissing):
        environment.env_dir()
    environment.configure(Context())
    assert environment.env_dir() == tmp_path / "data" / "loop-env"
    monkeypatch.setenv(environment.ENV_VAR, str(tmp_path / "there"))
    assert environment.env_dir() == tmp_path / "there"


def test_the_install_commands_keep_vtk_out():
    libraries, loop = environment.commands(Path("/uv"), Path("/env/bin/python"))
    assert "--no-deps" not in libraries and "--no-deps" in loop
    assert set(environment.PACKAGES) <= set(libraries)
    assert set(environment.PACKAGES) == {"numpy", "scipy", "pandas", "scikit-image", "scikit-learn", "pydantic",
                                         "pyyaml"}
    assert libraries[:3] == ["/uv", "pip", "install"] and "/env/bin/python" in loop
    assert loop[-3:] == ["LoopStructural>=1.10,<1.11", "loop-common<0.1", "loop-interpolation<0.1"]
    assert not any("vtk" in c.lower() or "pyvista" in c.lower() for c in libraries + loop)


ANSWER_OK = {"python": "3.12.12", "loopstructural": "1.10.9", "numpy": "2.0", "pandas": "3.0", "scipy": "1.0",
             "scikit-image": "0.26"}
ANSWER_BAD = {"python": "3.12.12", "error": "ModuleNotFoundError: No module named 'loop_common'"}
FAKE_UV = ("#!/bin/sh\n"
           "if [ \"$1\" = venv ]; then\n"
           "  for last; do :; done\n"
           "  mkdir -p \"$last/bin\"\n"
           "  printf '#!/bin/sh\\ncat \"$ANSWER\"\\n' > \"$last/bin/python\"\n"
           "  chmod +x \"$last/bin/python\"\n"
           "  echo \"Creating virtual environment at: $last\"\n"
           "  exit 0\n"
           "fi\n")


@pytest.mark.skipif(os.name == "nt", reason="the fake uv is a shell script")
def test_the_environment_installed_with_uv(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "local_available", lambda: False)
    monkeypatch.setattr(environment, "unsupported", lambda *a: None)
    monkeypatch.setattr(environment, "_CHECKED", {})
    answer = tmp_path / "answer.json"
    answer.write_text(json.dumps(ANSWER_OK))
    monkeypatch.setenv("ANSWER", str(answer))
    log = tmp_path / "uv.log"
    monkeypatch.setenv("UVLOG", str(log))
    uv = tmp_path / "uv"
    uv.write_text(FAKE_UV + "if [ \"$1\" = pip ]; then echo \"$@\" >> \"$UVLOG\"; echo ' + loopstructural==1.10.9'; "
                  "[ -n \"$FAIL\" ] && exit 1; exit 0; fi\nexit 2\n")
    uv.chmod(0o755)
    env = tmp_path / "env"
    monkeypatch.setenv(environment.ENV_VAR, str(env))
    seen = []
    assert environment.install(Progress(lambda f, m: seen.append((f, m))), uv=uv) == env
    calls = log.read_text().strip().splitlines()
    assert len(calls) == 2 and "--no-deps" not in calls[0] and "--no-deps" in calls[1]
    assert (env / environment.MARKER).exists() and environment.installed_info()["loopstructural"] == "1.10.9"
    assert environment.worker_python() == environment.env_python(env) and environment.known_usable()
    assert environment.where() == ("environment", "1.10.9") and "1.10.9" in environment.status_text()
    assert seen[-1] == (1.0, "LoopStructural pronto (1.10.9)")
    fractions = [f for f, _m in seen]
    assert fractions == sorted(fractions)
    # a failure leaves no environment that looks ready
    monkeypatch.setenv("FAIL", "1")
    with pytest.raises(RuntimeError, match="Installazione non riuscita"):
        environment.install(uv=uv)
    assert not environment.is_installed()
    # installed but it does not import: no marker, and what is wrong is said
    monkeypatch.delenv("FAIL")
    answer.write_text(json.dumps(ANSWER_BAD))
    with pytest.raises(environment.LoopMissing, match="loop_common"):
        environment.install(uv=uv)
    assert not environment.is_installed()


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
    uv.write_text(FAKE_UV +
                  "if [ -z \"$UV_CACHE_DIR\" ]; then\n"
                  "  echo 'error: Failed to write to the client cache'\n"
                  "  echo '  Caused by: Permission denied (os error 13) at path \"/Users/x/.cache/uv/wheels\"'\n"
                  "  exit 2\n"
                  "fi\n"
                  "mkdir -p \"$UV_CACHE_DIR\" && echo cached > \"$UV_CACHE_DIR/wheel\"\n"
                  "echo ' + loopstructural==1.10.9'\n"
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
