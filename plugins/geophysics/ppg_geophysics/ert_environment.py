"""Where pyGIMLi runs: a Python that has it, never the program's process.

* the program's own Python, when pyGIMLi is installed beside it (``pip install pygimli``): looked
  up with :func:`importlib.util.find_spec`, never imported here - its core (``pgcore``) brings
  compiled libraries and an OpenMP runtime of its own, which must not meet the program's;
* otherwise an environment the plugin installs on request with ``uv`` (a Python and pyGIMLi from
  PyPI, about 100 MB) in ``<plugin data>/ert-env`` (``PPG_GEOPHYSICS_ENV`` to put it elsewhere): the
  packaged application, which is not a Python and has no pip, works this way.

pyGIMLi 1.6's core has wheels for Windows x64 (Python 3.12-3.14), Linux x86_64 (3.10-3.14), macOS on
Apple Silicon (Python 3.12) and on Intel (Python 3.13), both for macOS 26 or later; none for Linux
or Windows on ARM: there the tomography says it is not available. The Python of the environment is
chosen by the computer (:data:`PYTHONS`).

The macOS wheel of pgcore 1.6.0 lacks ``libblas.3.dylib``, which two of its libraries ask for
while it carries OpenBLAS: :func:`repair` links the name to the OpenBLAS it carries.

pyGIMLi writes a configuration file and a cache into the user's folders when it is imported: the
worker runs with its own home folder, ``<plugin data>/ert-home`` (:func:`worker_environment`).
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

PYGIMLI = ("pygimli>=1.6.1,<1.7", "pgcore>=1.6,<1.7")
PYTHONS = {("darwin", "arm64"): "3.12", ("darwin", "x86_64"): "3.13", ("win32", "amd64"): "3.12",
           ("win32", "x86_64"): "3.12", ("linux", "x86_64"): "3.12", ("linux", "amd64"): "3.12"}
MARKER = "ppg-geophysics-ert.json"
ENV_VAR = "PPG_GEOPHYSICS_ENV"
SIZE_MB = 80  # the environment (pyGIMLi, pgcore, numpy, scipy, matplotlib)
PYTHON_MB = 60  # the Python uv downloads when it has not that version yet
_DATA: Path | None = None
CHECK = r"""
import importlib, importlib.metadata as md, json, platform
out = {"python": platform.python_version()}
try:
    import pygimli
    from pygimli.physics import ert
    import pygimli.meshtools
    out["pygimli"] = md.version("pygimli")
    out["pgcore"] = md.version("pgcore")
    out["numpy"] = md.version("numpy")
    out["manager"] = hasattr(ert, "ERTManager")
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out))
"""
_CHECKED: dict[str, dict] = {}


class GimliMissing(RuntimeError):
    """No Python with pyGIMLi for the worker: the message says why and what to do."""


def configure(context) -> None:
    global _DATA
    _DATA = Path(context.data_dir)


def _machine(machine: str | None) -> str:
    return (machine or platform.machine() or "").lower()


def python_for(system: str | None = None, machine: str | None = None) -> str | None:
    system = system or sys.platform
    family = "linux" if system.startswith("linux") else system
    return PYTHONS.get((family, _machine(machine)))


def unsupported(system: str | None = None, machine: str | None = None, mac_release: str | None = None) -> str | None:
    """Why pyGIMLi cannot run on this computer, or None (``system``, ``machine``, ``mac_release``
    for the tests)."""
    system = system or sys.platform
    machine = _machine(machine)
    if python_for(system, machine) is None:
        if machine in ("arm64", "aarch64"):
            return t("pyGIMLi non ha una versione per {0} su ARM: la tomografia elettrica non è disponibile su "
                     "questo computer.",
                     "pyGIMLi has no build for {0} on ARM: the resistivity tomography is not available on this "
                     "computer.").format("Windows" if system == "win32" else "Linux")
        return t("pyGIMLi non ha una versione per questo computer ({0} {1}).",
                 "pyGIMLi has no build for this computer ({0} {1}).").format(system, machine)
    if system == "darwin":
        release = mac_release if mac_release is not None else platform.mac_ver()[0]
        try:
            major = int(str(release).split(".")[0])
        except ValueError:
            major = 0
        if 0 < major < 26:
            return t("pyGIMLi su macOS richiede macOS 26 o successivo (il suo nucleo non ha una versione per "
                     "macOS {0}).",
                     "pyGIMLi on macOS needs macOS 26 or later (its core has no build for macOS {0}).").format(release)
    return None


def local_available() -> bool:
    if getattr(sys, "frozen", False):
        return False
    try:
        return importlib.util.find_spec("pygimli") is not None and importlib.util.find_spec("pgcore") is not None
    except (ImportError, ValueError):
        return False


def local_version() -> str:
    try:
        from importlib.metadata import version

        return version("pygimli")
    except Exception:  # noqa: BLE001
        return ""


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    if _DATA is None:
        raise GimliMissing("the plugin has no folder for the environment of pyGIMLi (configure() first)")
    return _DATA / "ert-env"


def home_dir() -> Path:
    """The home folder of the worker (pyGIMLi's configuration and cache)."""
    if _DATA is not None:
        return _DATA / "ert-home"
    try:
        return env_dir().parent / "ert-home"
    except GimliMissing:
        import tempfile

        return Path(tempfile.gettempdir()) / "ppg-geophysics-ert-home"


def env_python(env: Path | None = None) -> Path:
    env = Path(env) if env is not None else env_dir()
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_info(env: Path | None = None) -> dict:
    try:
        env = Path(env) if env is not None else env_dir()
        if not env_python(env).exists():
            return {}
        return json.loads((env / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError, GimliMissing):
        return {}


def is_installed(env: Path | None = None) -> bool:
    return bool(installed_info(env))


def where() -> tuple[str | None, str]:
    """``("local", version)``, ``("environment", version)`` or ``(None, "")``."""
    if unsupported():
        return None, ""
    if local_available():
        return "local", local_version()
    info = installed_info()
    if info:
        return "environment", str(info.get("pygimli", ""))
    return None, ""


def status_text() -> str:
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind is not None:
        try:
            known = _CHECKED.get(str(_python()))
        except GimliMissing:
            known = None
        if known is not None and not usable(known):
            return problem_text(known)
    if kind == "local":
        return t("pyGIMLi {0} di questo Python, in un processo separato.",
                 "pyGIMLi {0} of this Python, in a process of its own.").format(version)
    if kind == "environment":
        return t("pyGIMLi {0} nell'ambiente del plugin.", "pyGIMLi {0} in the environment of the plugin.").format(version)
    return t("pyGIMLi non è installato: Installa pyGIMLi lo scarica in un ambiente del plugin (circa {0} MB, più "
             "circa {1} MB per Python se uv non lo ha già).",
             "pyGIMLi is not installed: Install pyGIMLi downloads it into an environment of the plugin "
             "(about {0} MB, plus about {1} MB for Python when uv has not it yet).").format(SIZE_MB, PYTHON_MB)


def _python() -> Path:
    problem = unsupported()
    if problem:
        raise GimliMissing(problem)
    if local_available():
        return Path(sys.executable)
    if is_installed():
        return env_python()
    raise GimliMissing(t("pyGIMLi non è installato: installalo dal pannello Geofisica (scheda ERT) o con "
                         "«ppg plugin geophysics ert install».",
                         "pyGIMLi is not installed: install it from the Geofisica pane (ERT tab) or with "
                         "'ppg plugin geophysics ert install'."))


def worker_python() -> Path:
    python = _python()
    info = check(python=python)
    if not usable(info):
        raise GimliMissing(problem_text(info))
    return python


def clean_environment(for_uv: bool = False) -> dict:
    drop = ("PYTHON", "VIRTUAL_ENV", "__PYVENV") + (("CONDA",) if for_uv else ())
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def worker_environment() -> dict:
    """The variables of the worker: those of :func:`clean_environment` with a home folder of the
    plugin (pyGIMLi writes its configuration and cache there, not into the user's folders) and
    Matplotlib without a display. The threads are pyGIMLi's own choice (its OpenMP runtime lives only
    in the worker process)."""
    env = clean_environment()
    home = home_dir()
    for sub in ("Library/Caches", "Library/Preferences", ".cache", ".config", "AppData/Roaming", "AppData/Local"):
        try:
            (home / sub).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    env.update({"HOME": str(home), "USERPROFILE": str(home), "APPDATA": str(home / "AppData/Roaming"),
                "LOCALAPPDATA": str(home / "AppData/Local"), "XDG_CONFIG_HOME": str(home / ".config"),
                "XDG_CACHE_HOME": str(home / ".cache"), "MPLBACKEND": "Agg"})
    return env


def probe(python: Path) -> dict:
    try:
        out = subprocess.run([str(python), "-I", "-c", CHECK], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=worker_environment(), close_fds=False, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    for line in reversed((out.stdout or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    said = ((out.stderr or "") + "\n" + (out.stdout or "")).strip()
    return {"error": t("il controllo di pyGIMLi si è chiuso (codice {0})", "the check of pyGIMLi closed (code {0})")
            .format(out.returncode), "warning": said[-4000:]}


def usable(info: dict) -> bool:
    return bool(info.get("pygimli")) and bool(info.get("manager"))


def check(progress=None, python: Path | None = None, again: bool = False) -> dict:
    python = Path(python) if python is not None else _python()
    key = str(python)
    known = _CHECKED.get(key)
    if known is not None and usable(known) and not again:
        return known
    if progress is not None:
        progress(0.1, t("Prova di pyGIMLi", "Trying pyGIMLi"))
    info = probe(python)
    _CHECKED[key] = info
    if progress is not None:
        progress(1.0, t("pyGIMLi funziona", "pyGIMLi works") if usable(info) else problem_text(info))
    return info


def known_usable() -> bool:
    try:
        known = _CHECKED.get(str(_python()))
    except GimliMissing:
        return False
    return known is not None and usable(known)


def problem_text(info: dict) -> str:
    if not info.get("pygimli"):
        said = (info.get("error") or "?").strip().splitlines() or ["?"]
        return t("pyGIMLi non si importa ({0}). Installa di nuovo pyGIMLi.",
                 "pyGIMLi does not import ({0}). Install pyGIMLi again.").format(said[-1][:300])
    return t("pyGIMLi {0} non ha il gestore ERT: serve pyGIMLi 1.6. Installa di nuovo pyGIMLi.",
             "pyGIMLi {0} lacks the ERT manager: pyGIMLi 1.6 is needed. Install pyGIMLi again.").format(info["pygimli"])


def find_uv(progress=None) -> Path:
    found = shutil.which("uv")
    if found:
        return Path(found)
    from pyarchinit_photogrammetry.ai.sam_env import find_uv as program_uv

    return Path(program_uv(progress))


class CommandFailed(RuntimeError):
    def __init__(self, message: str, lines: list[str], fraction: float = 0.0):
        super().__init__(message)
        self.lines, self.fraction = lines, fraction


def cache_refused(lines) -> bool:
    text = "\n".join(lines).lower()
    return "permission denied" in text and "cache" in text


def _run(cmd: list[str], env: dict, progress, start: float, end: float) -> list[str]:
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


def site_packages(env: Path) -> list[Path]:
    if os.name == "nt":
        return [env / "Lib" / "site-packages"]
    return sorted((env / "lib").glob("python3*/site-packages"))


def repair(env: Path) -> list[Path]:
    """The macOS wheel of pgcore 1.6.0 asks for ``@rpath/libblas.3.dylib`` and carries OpenBLAS as
    ``libopenblas.0.dylib``: the name linked to it beside the other libraries of the wheel (nothing
    when the wheel carries a BLAS of that name, or on another system). Returns the links made."""
    made = []
    if sys.platform != "darwin":
        return made
    for site in site_packages(Path(env)):
        libs = site / "pgcore" / ".dylibs"
        blas = libs / "libblas.3.dylib"
        openblas = libs / "libopenblas.0.dylib"
        if libs.is_dir() and openblas.exists() and not blas.exists():
            os.symlink(openblas.name, blas)
            made.append(blas)
    return made


def install(progress=None, env: Path | None = None, uv: Path | None = None) -> Path:
    """Create the environment of the plugin: a Python (by the computer) and pyGIMLi from PyPI, with
    ``uv``. A minute or two; returns the folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    problem = unsupported()
    if problem:
        raise GimliMissing(problem)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():
        (env / MARKER).unlink(missing_ok=True)
    own_cache = (_DATA or env.parent) / "uv-cache"
    python = python_for()
    progress(0.05, t("Creazione dell'ambiente Python {0} per pyGIMLi", "Creating the Python {0} environment for pyGIMLi")
             .format(python))
    _uv([str(uv), "venv", "--python", python, "--allow-existing", str(env)], clean, progress, 0.05, 0.2, own_cache)
    progress(0.2, t("Installazione di pyGIMLi", "Installing pyGIMLi"))
    _uv([str(uv), "pip", "install", "--upgrade", "--python", str(env_python(env)), *PYGIMLI], clean, progress,
        0.2, 0.88, own_cache)
    if clean.get("UV_CACHE_DIR") == str(own_cache):
        shutil.rmtree(own_cache, ignore_errors=True)
    links = repair(env)
    if links:
        progress(0.9, t("Libreria BLAS collegata a OpenBLAS nel nucleo di pyGIMLi",
                        "BLAS library linked to the OpenBLAS of pyGIMLi's core"))
    progress(0.92, t("Prova di pyGIMLi", "Trying pyGIMLi"))
    info = probe(env_python(env))
    _CHECKED[str(env_python(env))] = info
    if not info.get("pygimli"):
        raise RuntimeError(t("pyGIMLi installato ma non si importa:", "pyGIMLi installed but it does not import:")
                           + "\n" + f"{info.get('error') or ''}\n{info.get('warning') or ''}".strip()[-1500:])
    (env / MARKER).write_text(json.dumps({k: info.get(k) for k in ("pygimli", "pgcore", "numpy", "python")}),
                              encoding="utf-8")
    if not usable(info):
        raise GimliMissing(problem_text(info))
    progress(1.0, t("pyGIMLi pronto ({0})", "pyGIMLi ready ({0})").format(info.get("pygimli", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
