"""Where CadQuery runs: a Python that has it, never the program's process.

* the program's own Python, when CadQuery is installed beside it: looked up with
  :func:`importlib.util.find_spec`, never imported here - OpenCascade (OCP) is hundreds of MB of compiled
  libraries and CadQuery pulls in casadi (LGPL-3), nlopt and more: all of it stays out of the window's process;
* otherwise an environment the plugin installs on request with ``uv`` (Python 3.12) in ``<plugin data>/env``
  (``PPG_ARCHITECTURE_ENV`` to put it elsewhere): the packaged application, which is not a Python and has no pip,
  works this way.

The environment comes in two commands (measured on macOS arm64, 2026-11-02): the small libraries CadQuery needs to
import, with their dependencies (numpy, ezdxf with fontTools, pyparsing, multimethod, runtype, nlopt,
typing-extensions); then CadQuery, its OpenCascade and VTK with ``--no-deps``, at the versions OpenCascade was built
against. VTK cannot be left out: the compiled OpenCascade module links VTK's libraries and does not load without
them. What ``--no-deps`` leaves out is casadi (LGPL-3, 183 MB: CadQuery's assembly solver), numba and llvmlite, scipy,
matplotlib and trame (display and notebooks): the worker puts a stub in the place of casadi, whose every name fails
if called. About 890 MB instead of the 1.3 GB of ``pip install cadquery``.

CadQuery's OpenCascade (``cadquery-ocp``) has wheels for macOS (Apple Silicon and Intel), Linux (x86-64 and
aarch64) and Windows x64, none for Windows on ARM: there the plugin says it is not available.

Before the first run of a session :func:`check` runs a probe in a process of its own: it imports CadQuery, makes a
small solid of revolution and reports the versions; a failure stops there with what to do (:func:`problem_text`).
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

# the libraries CadQuery needs to import, with their dependencies
PACKAGES = ("numpy", "ezdxf>=1.3", "pyparsing>=3", "multimethod>=1.11,<2", "runtype", "nlopt>=2.9,<3",
            "typing-extensions")
# with --no-deps: CadQuery, its OpenCascade and the VTK OpenCascade was linked against (its libraries are loaded by
# OpenCascade's module); without casadi, numba, scipy, matplotlib, trame
PINNED = ("cadquery>=2.8,<2.9", "cadquery-ocp==7.9.3.1.1", "cadquery-ocp-proxy==7.9.3.1.1", "vtk==9.6.2")
PYTHON = "3.12"
MARKER = "ppg-architecture.json"
ENV_VAR = "PPG_ARCHITECTURE_ENV"
_DATA: Path | None = None  # the plugin's folder (configure())
# what a Python with CadQuery answers (one line of JSON): the versions, and whether a revolve, a sweep and the
# STEP writer work there
CHECK = r"""
import importlib.metadata as md, json, os, platform, sys, tempfile
out = {"python": platform.python_version()}
try:
    import importlib.util, types
    if importlib.util.find_spec("casadi") is None:  # as the worker does: a stub for CadQuery's assembly solver
        stub = types.ModuleType("casadi")
        stub.__getattr__ = lambda name: None
        sys.modules["casadi"] = stub
    import cadquery as cq
    out["cadquery"] = md.version("cadquery")
    for name in ("cadquery-ocp", "numpy"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = ""
    solid = cq.Workplane("XZ").moveTo(0, 0).lineTo(1, 0).lineTo(1, 1).lineTo(0, 1).close().revolve(360, (0, 0, 0), (0, 1, 0))
    out["volume"] = solid.val().Volume()
    with tempfile.TemporaryDirectory() as d:
        cq.exporters.export(solid, os.path.join(d, "x.step"))
        out["step"] = os.path.getsize(os.path.join(d, "x.step")) > 0
    out["vtk"] = "vtkmodules" in sys.modules
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out))
"""
_CHECKED: dict[str, dict] = {}  # Python -> what check() found (a failure is looked at again next time)


class CadQueryMissing(RuntimeError):
    """No Python with CadQuery for the worker: the message says why and what to do."""


def configure(context) -> None:
    """Remember the plugin's folder (``context.data_dir``), where the environment goes."""
    global _DATA
    _DATA = Path(context.data_dir)


def unsupported(system: str | None = None, machine: str | None = None) -> str | None:
    """Why CadQuery cannot run on this computer, or None. ``system`` (``sys.platform``) and
    ``machine`` (``platform.machine()``) for the tests."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    if system == "win32" and machine in ("arm64", "aarch64"):
        return t("CadQuery (OpenCascade) non ha una versione per Windows su ARM: gli elementi architettonici non sono "
                 "disponibili su questo computer.",
                 "CadQuery (OpenCascade) has no build for Windows on ARM: the architectural elements are not available "
                 "on this computer.")
    ok = {"darwin": ("arm64", "x86_64"), "win32": ("amd64", "x86_64"), "linux": ("x86_64", "amd64", "aarch64", "arm64")}
    family = "linux" if system.startswith("linux") else system
    if machine not in ok.get(family, ()):
        return t("CadQuery non ha una versione per questo computer ({0} {1}).",
                 "CadQuery has no build for this computer ({0} {1}).").format(system, machine)
    return None


def local_available() -> bool:
    """CadQuery and what the engine needs installed for this Python (looked up, not imported).
    Never in the packaged application: its executable is not a Python."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return importlib.util.find_spec("cadquery") is not None
    except (ImportError, ValueError):
        return False


def local_version() -> str:
    try:
        from importlib.metadata import version

        return version("cadquery")
    except Exception:  # noqa: BLE001
        return ""


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    if _DATA is None:
        raise CadQueryMissing("the plugin has no folder for the environment of CadQuery (configure() first)")
    return _DATA / "env"


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
    except (OSError, ValueError, CadQueryMissing):
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
        return "environment", str(info.get("cadquery", ""))
    return None, ""


def status_text() -> str:
    """Where CadQuery runs, in a line for the pane and the command line (with what the last
    check of this session found wrong, if anything)."""
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind is not None:
        try:
            known = _CHECKED.get(str(_python()))
        except CadQueryMissing:
            known = None
        if known is not None and not usable(known):
            return problem_text(known)
    if kind == "local":
        return t("CadQuery {0} di questo Python, in un processo separato.",
                 "CadQuery {0} of this Python, in a process of its own.").format(version)
    if kind == "environment":
        return t("CadQuery {0} nell'ambiente del plugin ({1}).",
                 "CadQuery {0} in the environment of the plugin ({1}).").format(version, env_dir())
    return t("CadQuery non è installato: Installa CadQuery lo scarica in un ambiente del plugin "
             "(circa 900 MB).",
             "CadQuery is not installed: Install CadQuery downloads it into an environment of the "
             "plugin (about 900 MB).")


def _python() -> Path:
    """The Python that has CadQuery (:class:`CadQueryMissing` when there is none), not checked."""
    problem = unsupported()
    if problem:
        raise CadQueryMissing(problem)
    if local_available():
        return Path(sys.executable)
    if is_installed():
        return env_python()
    raise CadQueryMissing(t("CadQuery non è installato: installalo dal pannello Elementi architettonici o con "
                        "«ppg plugin architecture install».",
                        "CadQuery is not installed: install it from the Architectural elements pane or with "
                        "'ppg plugin architecture install'."))


def worker_python() -> Path:
    """The Python the worker runs with, checked (:func:`check`) once a session: :class:`CadQueryMissing`
    when there is none or CadQuery does not work there, with what to do."""
    python = _python()
    info = check(python=python)
    if not usable(info):
        raise CadQueryMissing(problem_text(info))
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
    return {"error": t("il controllo di CadQuery si è chiuso (codice {0})",
                       "the check of CadQuery closed (code {0})").format(out.returncode),
            "warning": said[-4000:]}


def usable(info: dict) -> bool:
    """CadQuery imports, makes a solid (a cylinder of radius and height 1: volume pi) and writes a STEP."""
    return bool(info.get("cadquery")) and not info.get("error") and bool(info.get("step")) and \
        abs(float(info.get("volume") or 0.0) - 3.141592653589793) < 1e-3


def check(progress=None, python: Path | None = None, again: bool = False) -> dict:
    """Whether CadQuery works with ``python`` (the worker's by default): :data:`CHECK` run once
    a session when it works; a failure is looked at again at the next call (the user may have
    installed what was missing meanwhile)."""
    python = Path(python) if python is not None else _python()
    key = str(python)
    known = _CHECKED.get(key)
    if known is not None and usable(known) and not again:
        return known
    if progress is not None:
        progress(0.1, t("Prova di CadQuery", "Trying CadQuery"))
    info = probe(python)
    _CHECKED[key] = info
    if progress is not None:
        progress(1.0, t("CadQuery funziona", "CadQuery works") if usable(info) else problem_text(info))
    return info


def known_usable() -> bool:
    """CadQuery was found working in this session (no check needed before a run)."""
    try:
        known = _CHECKED.get(str(_python()))
    except CadQueryMissing:
        return False
    return known is not None and usable(known)


def problem_text(info: dict) -> str:
    """Why CadQuery does not work, from what :func:`check` found, and what to do."""
    said = (info.get("error") or "?").strip().splitlines() or ["?"]
    return t("CadQuery non si importa ({0}). Installa di nuovo CadQuery.",
             "CadQuery does not import ({0}). Install CadQuery again.").format(said[-1][:300])


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
    """The two commands that fill the environment: the libraries, then CadQuery, OpenCascade and VTK without what they
    declare beside them."""
    return [[str(uv), "pip", "install", "--upgrade", "--python", str(python), *PACKAGES],
            [str(uv), "pip", "install", "--upgrade", "--no-deps", "--python", str(python), *PINNED]]


def install(progress=None, env: Path | None = None, uv: Path | None = None) -> Path:
    """Create the environment of the plugin: Python 3.12 and CadQuery from PyPI, with ``uv``
    (``uv``: another one, for the tests). A few minutes, about 900 MB; returns the folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    problem = unsupported()
    if problem:
        raise CadQueryMissing(problem)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():  # a failure leaves no environment that looks ready
        (env / MARKER).unlink(missing_ok=True)
    own_cache = (_DATA or env.parent) / "uv-cache"
    progress(0.05, t("Creazione dell'ambiente Python per CadQuery",
                     "Creating the Python environment for CadQuery"))
    _uv([str(uv), "venv", "--python", PYTHON, "--allow-existing", str(env)], clean, progress, 0.05, 0.2, own_cache)
    progress(0.2, t("Installazione di CadQuery (alcuni minuti)", "Installing CadQuery (a few minutes)"))
    libraries, cadquery = commands(uv, env_python(env))
    _uv(libraries, clean, progress, 0.2, 0.4, own_cache)
    _uv(cadquery, clean, progress, 0.4, 0.9, own_cache)
    if clean.get("UV_CACHE_DIR") == str(own_cache):  # the environment does not need the cache any more
        shutil.rmtree(own_cache, ignore_errors=True)
    progress(0.92, t("Prova di CadQuery", "Trying CadQuery"))
    info = probe(env_python(env))
    _CHECKED[str(env_python(env))] = info
    if not usable(info):
        raise CadQueryMissing(problem_text(info))
    (env / MARKER).write_text(json.dumps({k: info.get(k) for k in ("cadquery", "cadquery-ocp", "numpy", "python")}),
                              encoding="utf-8")
    progress(1.0, t("CadQuery pronto ({0})", "CadQuery ready ({0})").format(info.get("cadquery", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
