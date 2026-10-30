"""Where Landlab runs: a Python that has it, never the program's process.

* the program's own Python, when Landlab is installed beside it (``pip install landlab``): looked
  up with :func:`importlib.util.find_spec`, never imported here - its compiled helpers (richdem,
  netCDF) and its many dependencies stay out of the window's process;
* otherwise an environment the plugin installs on request with ``uv`` (Python 3.12 and Landlab from
  PyPI with its dependencies, about 500 MB) in ``<plugin data>/env`` (``PPG_LANDSCAPE_ENV`` to put it
  elsewhere): the packaged application, which is not a Python and has no pip, works this way.

Landlab and its compiled dependencies have wheels for macOS (Apple Silicon and Intel), Linux and
Windows x64, not for Windows on ARM: there the plugin says it is not available.

Before the first run of a session :func:`check` runs Landlab in a process of its own: it imports
the components the worker uses and reports the versions, and whether richdem (the fast flow
router) loads; a failure stops there with what to do (:func:`problem_text`).
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

LANDLAB = "landlab>=2.9,<3"  # the components and the parameters the engine uses
PYTHON = "3.12"
MARKER = "ppg-landscape.json"
ENV_VAR = "PPG_LANDSCAPE_ENV"
_DATA: Path | None = None  # the plugin's folder (configure())
# the components the engine needs (FlowAccumulator and LinearDiffuser are a must, an eroder of the three)
REQUIRED = ("LinearDiffuser", "FlowAccumulator", "DepressionFinderAndRouter")
ERODERS = ("SpaceLargeScaleEroder", "Space", "ErosionDeposition")
# what a Python with Landlab answers (one line of JSON): the versions, the components missing, richdem
CHECK = r"""
import importlib.metadata as md, json, platform
out = {"python": platform.python_version()}
try:
    import landlab, landlab.components as lc
    out["landlab"] = md.version("landlab")
    for name in ("numpy", "scipy"):
        try:
            out[name] = md.version(name)
        except Exception:
            out[name] = ""
    out["missing"] = [c for c in __REQUIRED__ if not hasattr(lc, c)]
    out["eroders"] = [c for c in __ERODERS__ if hasattr(lc, c)]
    try:
        lc.PriorityFloodFlowRouter.load_richdem()
        out["richdem"] = md.version("py-richdem")
    except Exception as exc:
        out["richdem"] = ""
        out["richdem_error"] = str(exc)[:300]
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
print(json.dumps(out))
""".replace("__REQUIRED__", json.dumps(list(REQUIRED))).replace("__ERODERS__", json.dumps(list(ERODERS)))
_CHECKED: dict[str, dict] = {}  # Python -> what check() found (a failure is looked at again next time)


class LandlabMissing(RuntimeError):
    """No Python with Landlab for the worker: the message says why and what to do."""


def configure(context) -> None:
    """Remember the plugin's folder (``context.data_dir``), where the environment goes."""
    global _DATA
    _DATA = Path(context.data_dir)


def unsupported(system: str | None = None, machine: str | None = None, mac_release: str | None = None) -> str | None:
    """Why Landlab cannot run on this computer, or None. ``system`` (``sys.platform``),
    ``machine`` (``platform.machine()``) and ``mac_release`` (``platform.mac_ver()[0]``) for the tests.

    On Intel Macs netCDF4, a dependency of Landlab 2.11, has wheels only for macOS 15 and later: on an
    older system uv would try to compile it and fail after a long download."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    if system == "darwin" and machine == "x86_64":
        release = mac_release if mac_release is not None else platform.mac_ver()[0]
        try:
            major = int(str(release).split(".")[0])
        except ValueError:
            major = 0
        if 0 < major < 15:
            return t("Landlab su un Mac Intel richiede macOS 15 o successivo (le sue librerie non hanno una versione "
                     "per macOS {0}).",
                     "Landlab on an Intel Mac needs macOS 15 or later (its libraries have no build for "
                     "macOS {0}).").format(release)
    if system == "win32" and machine in ("arm64", "aarch64"):
        return t("Landlab non ha una versione per Windows su ARM: le simulazioni del paesaggio non sono disponibili "
                 "su questo computer.",
                 "Landlab has no build for Windows on ARM: the landscape simulations are not available on this "
                 "computer.")
    ok = {"darwin": ("arm64", "x86_64"), "win32": ("amd64", "x86_64"), "linux": ("x86_64", "amd64", "aarch64", "arm64")}
    family = "linux" if system.startswith("linux") else system
    if machine not in ok.get(family, ()):
        return t("Landlab non ha una versione per questo computer ({0} {1}).",
                 "Landlab has no build for this computer ({0} {1}).").format(system, machine)
    return None


def local_available() -> bool:
    """Landlab installed for this Python (looked up, not imported). Never in the packaged
    application: its executable is not a Python."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return importlib.util.find_spec("landlab") is not None
    except (ImportError, ValueError):
        return False


def local_version() -> str:
    try:
        from importlib.metadata import version

        return version("landlab")
    except Exception:  # noqa: BLE001
        return ""


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    if _DATA is None:
        raise LandlabMissing("the plugin has no folder for the environment of Landlab (configure() first)")
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
    except (OSError, ValueError, LandlabMissing):
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
        return "environment", str(info.get("landlab", ""))
    return None, ""


def status_text() -> str:
    """Where Landlab runs, in a line for the pane and ``ppg plugin landscape status`` (with what the
    last check of this session found wrong, if anything)."""
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind is not None:
        try:
            known = _CHECKED.get(str(_python()))
        except LandlabMissing:
            known = None
        if known is not None and not usable(known):
            return problem_text(known)
    if kind == "local":
        return t("Landlab {0} di questo Python, in un processo separato.",
                 "Landlab {0} of this Python, in a process of its own.").format(version)
    if kind == "environment":
        return t("Landlab {0} nell'ambiente del plugin ({1}).",
                 "Landlab {0} in the environment of the plugin ({1}).").format(version, env_dir())
    return t("Landlab non è installato: Installa Landlab lo scarica in un ambiente del plugin (circa 500 MB).",
             "Landlab is not installed: Install Landlab downloads it into an environment of the plugin "
             "(about 500 MB).")


def _python() -> Path:
    """The Python that has Landlab (:class:`LandlabMissing` when there is none), not checked."""
    problem = unsupported()
    if problem:
        raise LandlabMissing(problem)
    if local_available():
        return Path(sys.executable)
    if is_installed():
        return env_python()
    raise LandlabMissing(t("Landlab non è installato: installalo dal pannello Paesaggio o con "
                           "«ppg plugin landscape install».",
                           "Landlab is not installed: install it from the Landscape pane or with "
                           "'ppg plugin landscape install'."))


def worker_python() -> Path:
    """The Python the worker runs with, checked (:func:`check`) once a session: :class:`LandlabMissing`
    when there is none or Landlab does not work there, with what to do."""
    python = _python()
    info = check(python=python)
    if not usable(info):
        raise LandlabMissing(problem_text(info))
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
    return {"error": t("il controllo di Landlab si è chiuso (codice {0})", "the check of Landlab closed (code {0})")
            .format(out.returncode), "warning": said[-4000:]}


def usable(info: dict) -> bool:
    """Landlab imports and has the components the engine needs (one eroder at least)."""
    return bool(info.get("landlab")) and not info.get("missing") and bool(info.get("eroders"))


def check(progress=None, python: Path | None = None, again: bool = False) -> dict:
    """Whether Landlab works with ``python`` (the worker's by default): :data:`CHECK` run once a
    session when it works; a failure is looked at again at the next call (the user may have
    installed what was missing meanwhile)."""
    python = Path(python) if python is not None else _python()
    key = str(python)
    known = _CHECKED.get(key)
    if known is not None and usable(known) and not again:
        return known
    if progress is not None:
        progress(0.1, t("Prova di Landlab", "Trying Landlab"))
    info = probe(python)
    _CHECKED[key] = info
    if progress is not None:
        progress(1.0, t("Landlab funziona", "Landlab works") if usable(info) else problem_text(info))
    return info


def known_usable() -> bool:
    """Landlab was found working in this session (no check needed before a run)."""
    try:
        known = _CHECKED.get(str(_python()))
    except LandlabMissing:
        return False
    return known is not None and usable(known)


def problem_text(info: dict) -> str:
    """Why Landlab does not work, from what :func:`check` found, and what to do."""
    version = info.get("landlab")
    if not version:
        said = (info.get("error") or "?").strip().splitlines() or ["?"]
        return t("Landlab non si importa ({0}). Installa di nuovo Landlab.",
                 "Landlab does not import ({0}). Install Landlab again.").format(said[-1][:300])
    missing = list(info.get("missing") or []) + ([] if info.get("eroders") else list(ERODERS))
    return t("A Landlab {0} mancano dei componenti ({1}): serve Landlab 2.9 o successivo. Installa di nuovo Landlab.",
             "Landlab {0} lacks some components ({1}): Landlab 2.9 or later is needed. Install Landlab again.").format(
        version, ", ".join(missing))


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
                          "plugin in {0}", "uv cannot write in its cache (permission denied): trying again with a "
                          "cache of the plugin in {0}").format(own_cache))
        env["UV_CACHE_DIR"] = str(own_cache)
        _run(cmd, env, progress, exc.fraction, end)


def install(progress=None, env: Path | None = None, uv: Path | None = None) -> Path:
    """Create the environment of the plugin: Python 3.12 and Landlab from PyPI, with ``uv``
    (``uv``: another one, for the tests). A few minutes; returns the folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    problem = unsupported()
    if problem:
        raise LandlabMissing(problem)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():  # a failure leaves no environment that looks ready
        (env / MARKER).unlink(missing_ok=True)
    own_cache = (_DATA or env.parent) / "uv-cache"
    progress(0.05, t("Creazione dell'ambiente Python per Landlab", "Creating the Python environment for Landlab"))
    _uv([str(uv), "venv", "--python", PYTHON, "--allow-existing", str(env)], clean, progress, 0.05, 0.2, own_cache)
    progress(0.2, t("Installazione di Landlab (alcuni minuti)", "Installing Landlab (a few minutes)"))
    _uv([str(uv), "pip", "install", "--upgrade", "--python", str(env_python(env)), LANDLAB, "numpy"], clean, progress,
        0.2, 0.9, own_cache)
    if clean.get("UV_CACHE_DIR") == str(own_cache):  # the environment does not need the cache any more
        shutil.rmtree(own_cache, ignore_errors=True)
    progress(0.92, t("Prova di Landlab", "Trying Landlab"))
    info = probe(env_python(env))
    _CHECKED[str(env_python(env))] = info
    if not info.get("landlab"):
        raise RuntimeError(t("Landlab installato ma non si importa:", "Landlab installed but it does not import:")
                           + "\n" + f"{info.get('error') or ''}\n{info.get('warning') or ''}".strip()[-1500:])
    (env / MARKER).write_text(json.dumps({k: info.get(k) for k in ("landlab", "numpy", "richdem", "python")}),
                              encoding="utf-8")
    if not usable(info):
        raise LandlabMissing(problem_text(info))
    progress(1.0, t("Landlab pronto ({0})", "Landlab ready ({0})").format(info.get("landlab", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
