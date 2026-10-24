"""Where pymeshlab runs: a Python that has it, never the program's process.

* the program's own Python, when pymeshlab is installed beside it (``pip install pymeshlab``):
  looked up with :func:`importlib.util.find_spec`, never imported here - its Qt libraries would
  meet the PySide6 of the window;
* otherwise an environment the plugin installs on request with ``uv`` (Python 3.12 and
  pymeshlab from PyPI, about 300 MB) in ``<plugin data>/env`` (``PPG_MESH_TOOLS_ENV`` to put it
  elsewhere): the packaged application, which is not a Python and has no pip, works this way.

pymeshlab has wheels for macOS (Apple Silicon and Intel), Linux (x86_64 and aarch64) and Windows
x64, not for Windows on ARM: there the plugin says it is not available.

A Python that has pymeshlab is not enough: on Linux its wheels need system libraries they do not
carry - ``libOpenGL.so.0`` above all, which the GLU they bundle links to and which MeshLab's PLY
reader and meshing filters need: without it pymeshlab imports but knows no file format ("Unknown
format for load: ply"). Before the first operation of a session :func:`check` runs pymeshlab in a
process of its own on a triangle written as PLY and asks for the filters the worker uses; a
failure stops there, naming the libraries missing and the packages of Debian/Ubuntu and Fedora that
have them (:func:`problem_text`).
"""
from __future__ import annotations

import importlib.util
import json
import os
import platform
import re
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
# the filters the worker calls: a plugin of MeshLab that did not load leaves some of them out
FILTERS = ("meshing_decimation_quadric_edge_collapse", "meshing_decimation_quadric_edge_collapse_with_texture",
           "meshing_isotropic_explicit_remeshing", "apply_coord_taubin_smoothing", "apply_coord_hc_laplacian_smoothing",
           "meshing_close_holes", "compute_selection_by_condition_per_face", "meshing_remove_duplicate_vertices",
           "meshing_remove_duplicate_faces", "meshing_remove_null_faces", "meshing_repair_non_manifold_edges",
           "meshing_repair_non_manifold_vertices", "meshing_remove_unreferenced_vertices", "get_geometric_measures",
           "get_topological_measures")
# what a Python with pymeshlab answers (one line of JSON): the versions, whether it reads a PLY (MeshLab's
# PLY reader is a plugin that needs system libraries on Linux), the filters it lacks, what pymeshlab printed
# when its plugins did not load
CHECK = r"""
import contextlib, importlib.metadata as md, io, json, os, platform, tempfile
out = {"python": platform.python_version()}
said = io.StringIO()
try:
    with contextlib.redirect_stdout(said):
        import pymeshlab
    out["pymeshlab"] = md.version("pymeshlab")
    try:
        out["numpy"] = md.version("numpy")
    except Exception:
        out["numpy"] = ""
    known = set(pymeshlab.filter_list())
    out["missing_filters"] = [f for f in __FILTERS__ if f not in known]
    folder = tempfile.mkdtemp(prefix="ppg-mesh-tools-check-")
    path = os.path.join(folder, "triangle.ply")
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\n"
                "element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 2\n")
    try:
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(path)
        out["reads_ply"] = ms.current_mesh().face_number() == 1
    except Exception as exc:
        out["reads_ply"] = False
        out["error"] = str(exc)
    finally:
        os.remove(path)
        os.rmdir(folder)
except Exception as exc:
    out["error"] = type(exc).__name__ + ": " + str(exc)
out["warning"] = said.getvalue().strip()[-4000:]
print(json.dumps(out))
""".replace("__FILTERS__", json.dumps(list(FILTERS)))
# the libraries pymeshlab's Linux wheels need and do not carry: (Debian/Ubuntu package, Fedora package)
LINUX_PACKAGES = {
    "libOpenGL.so.0": ("libopengl0", "libglvnd-opengl"),
    "libGL.so.1": ("libgl1", "libglvnd-glx"),
    "libGLU.so.1": ("libglu1-mesa", "mesa-libGLU"),
    "libgomp.so.1": ("libgomp1", "libgomp"),
    "libX11.so.6": ("libx11-6", "libX11"),
    "libxcb.so.1": ("libxcb1", "libxcb"),
    "libfontconfig.so.1": ("libfontconfig1", "fontconfig"),
    "libfreetype.so.6": ("libfreetype6", "freetype"),
    "libz.so.1": ("zlib1g", "zlib"),
}
LINUX_USUAL = ("libOpenGL.so.0", "libGL.so.1")  # what a Linux without a desktop lacks, when nothing names it
_MISSING = re.compile(r"([A-Za-z0-9_.+-]+\.so(?:\.\d+)*): cannot open shared object file")
_CHECKED: dict[str, dict] = {}  # Python -> what check() found (a failure is looked at again next time)


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
    """Where MeshLab runs, in a line for the pane and ``ppg plugin mesh-tools status`` (with what the
    last check of this session found wrong, if anything)."""
    problem = unsupported()
    if problem:
        return problem
    kind, version = where()
    if kind is not None:
        try:
            known = _CHECKED.get(str(_python()))
        except MeshLabMissing:
            known = None
        if known is not None and not usable(known):
            return problem_text(known)
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


def _python() -> Path:
    """The Python that has pymeshlab (:class:`MeshLabMissing` when there is none), not checked."""
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


def worker_python() -> Path:
    """The Python the worker runs with, checked (:func:`check`) once a session: :class:`MeshLabMissing`
    when there is none or pymeshlab does not work there, with what to do."""
    python = _python()
    info = check(python=python)
    if not usable(info):
        raise MeshLabMissing(problem_text(info))
    return python


def is_linux() -> bool:
    return sys.platform.startswith("linux")


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
    return {"error": t("il controllo di pymeshlab si è chiuso (codice {0})", "the check of pymeshlab closed (code {0})")
            .format(out.returncode), "warning": said[-4000:]}


def usable(info: dict) -> bool:
    """pymeshlab imports, reads a PLY and has every filter the worker calls."""
    return bool(info.get("pymeshlab")) and bool(info.get("reads_ply")) and not info.get("missing_filters")


def check(progress=None, python: Path | None = None, again: bool = False) -> dict:
    """Whether pymeshlab works with ``python`` (the worker's by default): :data:`CHECK` run once a
    session when it works; a failure is looked at again at the next call (the user may have installed
    what was missing meanwhile)."""
    python = Path(python) if python is not None else _python()
    key = str(python)
    known = _CHECKED.get(key)
    if known is not None and usable(known) and not again:
        return known
    if progress is not None:
        progress(0.1, t("Prova di MeshLab", "Trying MeshLab"))
    info = probe(python)
    _CHECKED[key] = info
    if progress is not None:
        progress(1.0, t("MeshLab funziona", "MeshLab works") if usable(info) else problem_text(info))
    return info


def known_usable() -> bool:
    """MeshLab was found working in this session (no check needed before an operation)."""
    try:
        known = _CHECKED.get(str(_python()))
    except MeshLabMissing:
        return False
    return known is not None and usable(known)


def missing_libraries(text: str) -> list[str]:
    """The shared libraries a text says could not be opened (``libOpenGL.so.0``), each once."""
    out: list[str] = []
    for name in _MISSING.findall(text or ""):
        if name not in out:
            out.append(name)
    return out


def linux_hint(libraries=()) -> str:
    """What to install on Linux: the packages of Debian/Ubuntu and Fedora with the libraries given
    (those a Linux without a desktop lacks when none is named)."""
    named = list(libraries)
    libs = named or list(LINUX_USUAL)
    deb = [LINUX_PACKAGES[lib][0] for lib in libs if lib in LINUX_PACKAGES] or [LINUX_PACKAGES[x][0] for x in LINUX_USUAL]
    rpm = [LINUX_PACKAGES[lib][1] for lib in libs if lib in LINUX_PACKAGES] or [LINUX_PACKAGES[x][1] for x in LINUX_USUAL]
    which = f" ({', '.join(named)})" if named else ""
    return t("Su Linux pymeshlab ha bisogno di librerie di sistema che non porta con sé{0}: su Debian e Ubuntu "
             "«sudo apt install {1}», su Fedora «sudo dnf install {2}». Poi riprova.",
             "On Linux pymeshlab needs system libraries it does not carry{0}: on Debian and Ubuntu "
             "'sudo apt install {1}', on Fedora 'sudo dnf install {2}'. Then try again.").format(
        which, " ".join(dict.fromkeys(deb)), " ".join(dict.fromkeys(rpm)))


def problem_text(info: dict, linux: bool | None = None) -> str:
    """Why pymeshlab does not work, from what :func:`check` found, and what to do (``linux``: the
    hint of Linux, by default when this is a Linux)."""
    linux = is_linux() if linux is None else linux
    version = info.get("pymeshlab")
    said = f"{info.get('error') or ''}\n{info.get('warning') or ''}"
    libs = missing_libraries(said)
    if not version:
        head = t("MeshLab: pymeshlab non si importa ({0}).", "MeshLab: pymeshlab does not import ({0}).").format(
            (info.get("error") or "?").strip().splitlines()[-1][:300])
    elif not info.get("reads_ply"):
        head = t("MeshLab: pymeshlab {0} non legge i modelli: i suoi moduli non si caricano.",
                 "MeshLab: pymeshlab {0} reads no model: its modules do not load.").format(version)
    else:
        missing = list(info.get("missing_filters") or [])
        head = t("MeshLab: a pymeshlab {0} mancano dei filtri ({1}): i suoi moduli non si caricano tutti.",
                 "MeshLab: pymeshlab {0} lacks some filters ({1}): not all of its modules load.").format(
            version, ", ".join(missing[:3]) + (", ..." if len(missing) > 3 else ""))
    if linux:
        return head + " " + linux_hint(libs)
    if libs:
        return head + " " + t("Librerie mancanti: {0}.", "Libraries missing: {0}.").format(", ".join(libs))
    return head + " " + t("Installa di nuovo MeshLab (pymeshlab).", "Install MeshLab (pymeshlab) again.")


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
    info = probe(env_python(env))
    _CHECKED[str(env_python(env))] = info
    if not info.get("pymeshlab"):
        raise RuntimeError(t("pymeshlab installato ma non si importa:", "pymeshlab installed but it does not import:")
                           + "\n" + f"{info.get('error') or ''}\n{info.get('warning') or ''}".strip()[-1500:])
    # installed: the environment stays (no 300 MB again), also when the computer still lacks a library
    (env / MARKER).write_text(json.dumps({k: info.get(k) for k in ("pymeshlab", "numpy", "python")}), encoding="utf-8")
    if not usable(info):
        raise MeshLabMissing(problem_text(info))
    progress(1.0, t("MeshLab pronto (pymeshlab {0})", "MeshLab ready (pymeshlab {0})").format(info.get("pymeshlab", "")))
    return env


def remove(env: Path | None = None) -> None:
    shutil.rmtree(Path(env) if env is not None else env_dir(), ignore_errors=True)
