"""Where pymeshlab runs: a Python that has it, never the program's process.

* the program's own Python, when pymeshlab is installed beside it (``pip install pymeshlab``):
  looked up with :func:`importlib.util.find_spec`, never imported here - its Qt libraries would
  meet the PySide6 of the window;
* otherwise an environment the plugin installs on request with ``uv`` (Python 3.12 and
  pymeshlab from PyPI, about 300 MB) in ``<plugin data>/env`` (``PPG_MESH_TOOLS_ENV`` to put it
  elsewhere): the packaged application, which is not a Python and has no pip, works this way.

pymeshlab has wheels for macOS (Apple Silicon and Intel), Linux (x86_64 and aarch64) and Windows
x64, not for Windows on ARM: there the plugin says it is not available.
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

PYMESHLAB = "pymeshlab>=2022.2.post3,<2027"  # the filters and their parameters the worker uses
PYTHON = "3.12"
MARKER = "ppg-mesh-tools.json"
ENV_VAR = "PPG_MESH_TOOLS_ENV"
_DATA: Path | None = None  # the plugin's folder (configure())
# what the environment of the plugin answers: the versions it has
CHECK = ("import json, platform, importlib.metadata as md, pymeshlab; "
         "print(json.dumps({'pymeshlab': md.version('pymeshlab'), 'numpy': md.version('numpy'), "
         "'python': platform.python_version()}))")


class MeshLabMissing(RuntimeError):
    """No Python with pymeshlab for the worker: the message says why and what to do."""


def configure(context) -> None:
    """Remember the plugin's folder (``context.data_dir``), where the environment goes."""
    global _DATA
    _DATA = Path(context.data_dir)


def unsupported(system: str | None = None, machine: str | None = None) -> str | None:
    """Why pymeshlab cannot run on this computer, or None. ``system`` (``sys.platform``) and
    ``machine`` (``platform.machine()``) for the tests."""
    system = system or sys.platform
    machine = (machine or platform.machine() or "").lower()
    if system == "win32" and machine in ("arm64", "aarch64"):
        return t("MeshLab (pymeshlab) non ha una versione per Windows su ARM: gli strumenti MeshLab non sono "
                 "disponibili su questo computer.",
                 "MeshLab (pymeshlab) has no build for Windows on ARM: the MeshLab tools are not available on "
                 "this computer.")
    ok = {"darwin": ("arm64", "x86_64"), "win32": ("amd64", "x86_64"), "linux": ("x86_64", "amd64", "aarch64", "arm64")}
    family = "linux" if system.startswith("linux") else system
    if machine not in ok.get(family, ()):
        return t("MeshLab (pymeshlab) non ha una versione per questo computer ({0} {1}).",
                 "MeshLab (pymeshlab) has no build for this computer ({0} {1}).").format(system, machine)
    return None


def local_available() -> bool:
    """pymeshlab installed for this Python (looked up, not imported). Never in the packaged
    application: its executable is not a Python."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return importlib.util.find_spec("pymeshlab") is not None
    except (ImportError, ValueError):
        return False


def local_version() -> str:
    try:
        from importlib.metadata import version

        return version("pymeshlab")
    except Exception:  # noqa: BLE001
        return ""


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    if _DATA is None:
        raise MeshLabMissing("the plugin has no folder for the environment of MeshLab (configure() first)")
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
    except (OSError, ValueError, MeshLabMissing):
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
        return "environment", str(info.get("pymeshlab", ""))
    return None, ""


def status_text() -> str:
    """Where MeshLab runs, in a line for the pane and ``ppg plugin mesh-tools status``."""
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind == "local":
        return t("MeshLab: pymeshlab {0} di questo Python, in un processo separato.",
                 "MeshLab: pymeshlab {0} of this Python, in a process of its own.").format(version)
    if kind == "environment":
        return t("MeshLab: pymeshlab {0} nell'ambiente del plugin ({1}).",
                 "MeshLab: pymeshlab {0} in the environment of the plugin ({1}).").format(version, env_dir())
    return t("MeshLab (pymeshlab) non è installato: Installa MeshLab lo scarica in un ambiente del plugin "
             "(circa 300 MB).",
             "MeshLab (pymeshlab) is not installed: Install MeshLab downloads it into an environment of the "
             "plugin (about 300 MB).")


def worker_python() -> Path:
    """The Python the worker runs with (:class:`MeshLabMissing` when there is none)."""
    problem = unsupported()
    if problem:
        raise MeshLabMissing(problem)
    if local_available():
        return Path(sys.executable)
    if is_installed():
        return env_python()
    raise MeshLabMissing(t("MeshLab (pymeshlab) non è installato: installalo dal pannello Mesh (MeshLab) o con "
                           "«ppg plugin mesh-tools install».",
                           "MeshLab (pymeshlab) is not installed: install it from the Mesh (MeshLab) pane or with "
                           "'ppg plugin mesh-tools install'."))


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


def _run(cmd: list[str], env: dict, progress, start: float, end: float) -> list[str]:
    """``cmd`` (absolute path, the file descriptors inherited, no working directory: posix_spawn,
    as every process the program starts), its lines moving the bar from ``start`` to ``end``;
    cancelling stops it."""
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", env=env, close_fds=False)
    lines: list[str] = []
    try:
        for k, raw in enumerate(proc.stdout, 1):
            line = raw.rstrip()
            lines.append(line)
            progress(min(end, start + (end - start) * (1 - 1 / (1 + k / 20))), line[-120:] or "...")
    except BaseException:
        proc.kill()
        proc.wait()
        raise
    finally:
        proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError(t("Installazione non riuscita:", "Installation failed:") + "\n" + "\n".join(lines[-25:]))
    return lines


def install(progress=None, env: Path | None = None, uv: Path | None = None) -> Path:
    """Create the environment of the plugin: Python 3.12 and pymeshlab from PyPI, with ``uv``
    (``uv``: another one, for the tests). A few minutes; returns the folder."""
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    problem = unsupported()
    if problem:
        raise MeshLabMissing(problem)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():  # a failure leaves no environment that looks ready
        (env / MARKER).unlink(missing_ok=True)
    progress(0.05, t("Creazione dell'ambiente Python per MeshLab", "Creating the Python environment for MeshLab"))
    _run([str(uv), "venv", "--python", PYTHON, "--allow-existing", str(env)], clean, progress, 0.05, 0.2)
    progress(0.2, t("Installazione di pymeshlab (alcuni minuti)", "Installing pymeshlab (a few minutes)"))
    _run([str(uv), "pip", "install", "--upgrade", "--python", str(env_python(env)), PYMESHLAB, "numpy"], clean, progress,
         0.2, 0.9)
    progress(0.92, t("Prova di pymeshlab", "Trying pymeshlab"))
    out = subprocess.run([str(env_python(env)), "-c", CHECK], capture_output=True, text=True, env=clean,
                         close_fds=False, timeout=600)
    if out.returncode != 0:
        raise RuntimeError(t("pymeshlab installato ma non si importa:", "pymeshlab installed but it does not import:")
                           + "\n" + (out.stderr or out.stdout)[-1500:])
    info = json.loads(out.stdout.strip().splitlines()[-1])
    (env / MARKER).write_text(json.dumps(info), encoding="utf-8")
    progress(1.0, t("MeshLab pronto (pymeshlab {0})", "MeshLab ready (pymeshlab {0})").format(info.get("pymeshlab", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
