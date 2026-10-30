"""Where LoopStructural runs: a Python that has it, never the program's process.

* the program's own Python, when LoopStructural is installed beside it with what the engine needs:
  looked up with :func:`importlib.util.find_spec`, never imported here - it pulls in pandas and a good
  part of the scientific stack, which stay out of the window's process;
* otherwise an environment the plugin installs on request with ``uv`` (Python 3.12, about 240 MB)
  in ``<plugin data>/loop-env`` (``PPG_GEOARCH_LOOP_ENV`` to put it elsewhere): the packaged
  application, which is not a Python and has no pip, works this way.

The environment comes in two commands. The libraries first, with their dependencies; then
LoopStructural and its two companions with ``--no-deps``: their declared dependencies would bring
VTK (650 MB more), and the only thing they use it for is drawing. The worker puts a stub in the
place of pyvista. All of it has wheels for macOS (Apple Silicon and Intel), Linux (x86-64 and
aarch64) and Windows (x64 and ARM64).

Before the first run of a session :func:`check` runs a probe in a process of its own: with the stub
in place it imports ``GeologicalModel``, pandas and scikit-image and reports the versions; a failure
stops there with what to do (:func:`problem_text`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .texts import t

# the wheels, with their dependencies
PACKAGES = ("numpy", "scipy", "pandas", "scikit-image", "scikit-learn", "pydantic", "pyyaml")
# with --no-deps: the versions the engine was written on (the API of LoopStructural is moving)
PINNED = ("LoopStructural>=1.10,<1.11", "loop-common<0.1", "loop-interpolation<0.1")
PYTHON = "3.12"
MARKER = "ppg-geoarch-loop.json"
ENV_VAR = "PPG_GEOARCH_LOOP_ENV"
_DATA: Path | None = None  # the plugin's folder (configure())
# what a Python with LoopStructural answers (one line of JSON): the versions
CHECK = r"""
import importlib.metadata as md, json, logging, platform, sys, types
sys.modules["pyvista"] = types.ModuleType("pyvista")
logging.disable(logging.INFO)
out = {"python": platform.python_version()}
try:
    from LoopStructural import GeologicalModel
    import pandas, scipy.ndimage, skimage.measure
    out["loopstructural"] = md.version("LoopStructural")
    for name in ("numpy", "pandas", "scipy", "scikit-image"):
        out[name] = md.version(name)
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out))
"""
_CHECKED: dict[str, dict] = {}  # Python -> what check() found (a failure is looked at again next time)


class LoopMissing(RuntimeError):
    """No Python with LoopStructural for the worker: the message says why and what to do."""


def configure(context) -> None:
    """Remember the plugin's folder (``context.data_dir``), where the environment goes."""
    global _DATA
    _DATA = Path(context.data_dir)


def unsupported(system: str | None = None, machine: str | None = None) -> str | None:
    """Why LoopStructural cannot run on this computer, or None. ``system`` (``sys.platform``) and
    ``machine`` (``platform.machine()``) for the tests."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    ok = {"darwin": ("arm64", "x86_64"), "win32": ("amd64", "x86_64", "arm64", "aarch64"),
          "linux": ("x86_64", "amd64", "aarch64", "arm64")}
    family = "linux" if system.startswith("linux") else system
    if machine not in ok.get(family, ()):
        return t("LoopStructural non ha una versione per questo computer ({0} {1}).",
                 "LoopStructural has no build for this computer ({0} {1}).").format(system, machine)
    return None


def local_available() -> bool:
    """LoopStructural and what the engine needs installed for this Python (looked up, not imported).
    Never in the packaged application: its executable is not a Python."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return all(importlib.util.find_spec(name) is not None for name in ("LoopStructural", "pandas", "skimage"))
    except (ImportError, ValueError):
        return False


def local_version() -> str:
    try:
        from importlib.metadata import version

        return version("LoopStructural")
    except Exception:  # noqa: BLE001
        return ""


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    if _DATA is None:
        raise LoopMissing("the plugin has no folder for the environment of LoopStructural (configure() first)")
    return _DATA / "loop-env"


def env_python(env: Path | None = None) -> Path:
    env = Path(env) if env is not None else env_dir()
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_info(env: Path | None = None) -> dict:
    """What the environment was installed with (empty when it is not)."""
    try:
        env = Path(env) if env is not None else env_dir()
        if not env_python(env).exists():
            return {}
        return json.loads((env / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, LoopMissing):
        return {}


def is_installed(env: Path | None = None) -> bool:
    return bool(installed_info(env))


def where() -> tuple[str | None, str]:
    """``("local", version)`` (this Python), ``("environment", version)`` or ``(None, "")``."""
    if unsupported():
        return None, ""
    if local_available():
        return "local", local_version()
    info = installed_info()
    if info:
        return "environment", str(info.get("loopstructural", ""))
    return None, ""


def status_text() -> str:
    """Where LoopStructural runs, in a line for the pane and the command line (with what the last
    check of this session found wrong, if anything)."""
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind is not None:
        try:
            known = _CHECKED.get(str(_python()))
        except LoopMissing:
            known = None
        if known is not None and not usable(known):
            return problem_text(known)
    if kind == "local":
        return t("LoopStructural {0} di questo Python, in un processo separato.",
                 "LoopStructural {0} of this Python, in a process of its own.").format(version)
    if kind == "environment":
        return t("LoopStructural {0} nell'ambiente del plugin ({1}).",
                 "LoopStructural {0} in the environment of the plugin ({1}).").format(version, env_dir())
    return t("LoopStructural non è installato: Installa LoopStructural lo scarica in un ambiente del plugin "
             "(circa 240 MB).",
             "LoopStructural is not installed: Install LoopStructural downloads it into an environment of the "
             "plugin (about 240 MB).")


def _python() -> Path:
    """The Python that has LoopStructural (:class:`LoopMissing` when there is none), not checked."""
    problem = unsupported()
    if problem:
        raise LoopMissing(problem)
    if local_available():
        return Path(sys.executable)
    if is_installed():
        return env_python()
    raise LoopMissing(t("LoopStructural non è installato: installalo dal pannello Geoarcheologia o con "
                        "«ppg plugin geoarchaeology implicit-install».",
                        "LoopStructural is not installed: install it from the Geoarchaeology pane or with "
                        "'ppg plugin geoarchaeology implicit-install'."))


def worker_python() -> Path:
    """The Python the worker runs with, checked (:func:`check`) once a session: :class:`LoopMissing`
    when there is none or LoopStructural does not work there, with what to do."""
    python = _python()
    info = check(python=python)
    if not usable(info):
        raise LoopMissing(problem_text(info))
    return python


def probe(python: Path) -> dict:
    """Run :data:`CHECK` with ``python`` in a process of its own (isolated: neither the program's
    variables nor its folder) and return what it answered; ``error`` when it did not answer."""
    try:
        out = subprocess.run([str(python), "-I", "-c", CHECK], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=clean_environment(), close_fds=False, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    for line in reversed((out.stdout or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    said = ((out.stderr or "") + "\n" + (out.stdout or "")).strip()
    return {"error": t("il controllo di LoopStructural si è chiuso (codice {0})",
                       "the check of LoopStructural closed (code {0})").format(out.returncode),
            "warning": said[-4000:]}


def usable(info: dict) -> bool:
    """LoopStructural and what the engine needs import."""
    return bool(info.get("loopstructural")) and not info.get("error")


def check(progress=None, python: Path | None = None, again: bool = False) -> dict:
    """Whether LoopStructural works with ``python`` (the worker's by default): :data:`CHECK` run once
    a session when it works; a failure is looked at again at the next call (the user may have
    installed what was missing meanwhile)."""
    python = Path(python) if python is not None else _python()
    key = str(python)
    known = _CHECKED.get(key)
    if known is not None and usable(known) and not again:
        return known
    if progress is not None:
        progress(0.1, t("Prova di LoopStructural", "Trying LoopStructural"))
    info = probe(python)
    _CHECKED[key] = info
    if progress is not None:
        progress(1.0, t("LoopStructural funziona", "LoopStructural works") if usable(info) else problem_text(info))
    return info


def known_usable() -> bool:
    """LoopStructural was found working in this session (no check needed before a run)."""
    try:
        known = _CHECKED.get(str(_python()))
    except LoopMissing:
        return False
    return known is not None and usable(known)


def problem_text(info: dict) -> str:
    """Why LoopStructural does not work, from what :func:`check` found, and what to do."""
    said = (info.get("error") or "?").strip().splitlines() or ["?"]
    return t("LoopStructural non si importa ({0}). Installa di nuovo LoopStructural.",
             "LoopStructural does not import ({0}). Install LoopStructural again.").format(said[-1][:300])


def clean_environment(for_uv: bool = False) -> dict:
    """The variables of this process without those of its Python (the worker must not take the
    program's paths for its own; ``for_uv``: nor the conda environment, where uv would install)."""
    drop = ("PYTHON", "VIRTUAL_ENV", "__PYVENV") + (("CONDA",) if for_uv else ())
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def find_uv(progress=None) -> Path:
    """``uv``: the program's (on the PATH, in its data folder, or downloaded from its official
    release, as for the SAM environment)."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    from pyarchinit_photogrammetry.ai.sam_env import find_uv as program_uv

    return Path(program_uv(progress))


class CommandFailed(RuntimeError):
    """A command of the installation failed: ``lines`` holds what it printed, ``fraction`` where the
    bar was."""

    def __init__(self, message: str, lines: list[str], fraction: float = 0.0):
        super().__init__(message)
        self.lines, self.fraction = lines, fraction


def cache_refused(lines) -> bool:
    """uv could not write in its cache (a cache with files of another user, made by a program run as root)."""
    text = "\n".join(lines).lower()
    return "permission denied" in text and "cache" in text


def _run(cmd: list[str], env: dict, progress, start: float, end: float) -> list[str]:
    """``cmd`` (absolute path, the file descriptors inherited, no working directory: posix_spawn,
    as every process the program starts), its lines moving the bar from ``start`` to ``end``;
    cancelling stops it."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env, close_fds=False)
    lines: list[str] = []
    fraction = start
    try:
        for k, raw in enumerate(proc.stdout, 1):
            line = raw.rstrip()
            lines.append(line)
            fraction = min(end, start + (end - start) * (1 - 1 / (1 + k / 20)))
            progress(fraction, line[-120:] or "...")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdout.close()
    if proc.wait() != 0:
        raise CommandFailed(t("Installazione non riuscita:", "Installation failed:") + "\n" + "\n".join(lines[-25:]),
                            lines, fraction)
    return lines


def _uv(cmd: list[str], env: dict, progress, start: float, end: float, own_cache: Path) -> None:
    """A uv command; when uv cannot write in its usual cache, once more with a cache of the plugin
    (``UV_CACHE_DIR`` = ``own_cache``, kept in ``env`` for the commands after), said in the log."""
    try:
        _run(cmd, env, progress, start, end)
    except CommandFailed as exc:
        if "UV_CACHE_DIR" in env or not cache_refused(exc.lines):
            raise
        progress(exc.fraction, t("uv non può scrivere nella sua cache (permesso negato): riprovo con una cache del "
                                 "plugin in {0}", "uv cannot write in its cache (permission denied): trying again "
                                 "with a cache of the plugin in {0}").format(own_cache))
        env["UV_CACHE_DIR"] = str(own_cache)
        _run(cmd, env, progress, exc.fraction, end)


def commands(uv: Path, python: Path) -> list[list[str]]:
    """The two commands that fill the environment: the libraries, then LoopStructural without VTK."""
    return [[str(uv), "pip", "install", "--upgrade", "--python", str(python), *PACKAGES],
            [str(uv), "pip", "install", "--upgrade", "--no-deps", "--python", str(python), *PINNED]]


def install(progress=None, env: Path | None = None, uv: Path | None = None) -> Path:
    """Create the environment of the plugin: Python 3.12 and LoopStructural from PyPI, with ``uv``
    (``uv``: another one, for the tests). A few minutes, about 240 MB; returns the folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    problem = unsupported()
    if problem:
        raise LoopMissing(problem)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():  # a failure leaves no environment that looks ready
        (env / MARKER).unlink(missing_ok=True)
    own_cache = (_DATA or env.parent) / "uv-cache"
    progress(0.05, t("Creazione dell'ambiente Python per LoopStructural",
                     "Creating the Python environment for LoopStructural"))
    _uv([str(uv), "venv", "--python", PYTHON, "--allow-existing", str(env)], clean, progress, 0.05, 0.2, own_cache)
    progress(0.2, t("Installazione di LoopStructural (alcuni minuti)", "Installing LoopStructural (a few minutes)"))
    libraries, loop = commands(uv, env_python(env))
    _uv(libraries, clean, progress, 0.2, 0.8, own_cache)
    _uv(loop, clean, progress, 0.8, 0.9, own_cache)
    if clean.get("UV_CACHE_DIR") == str(own_cache):  # the environment does not need the cache any more
        shutil.rmtree(own_cache, ignore_errors=True)
    progress(0.92, t("Prova di LoopStructural", "Trying LoopStructural"))
    info = probe(env_python(env))
    _CHECKED[str(env_python(env))] = info
    if not usable(info):
        raise LoopMissing(problem_text(info))
    (env / MARKER).write_text(json.dumps({k: info.get(k) for k in ("loopstructural", "numpy", "pandas", "python")}),
                              encoding="utf-8")
    progress(1.0, t("LoopStructural pronto ({0})", "LoopStructural ready ({0})").format(info.get("loopstructural", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
