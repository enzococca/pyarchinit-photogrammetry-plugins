"""Where the models run: a Python with PyTorch, never the program's process.

* the program's own Python, when PyTorch, transformers and sam2 are installed beside it (a
  developer's ``pip install``): looked up with :func:`importlib.util.find_spec`, never imported
  here - PyTorch brings its own OpenMP runtime, and with pycolmap's in the same process the second
  one to start aborts it;
* otherwise the environment of the AI plugins, ``<app data>/pyarchinit-photogrammetry/geoai-env``
  (``PPG_GEOAI_ENV`` to put it elsewhere), shared with the plugin geoai so that PyTorch (about 2 GB)
  is there once: made on request with the program's ``uv`` (Python 3.12, the PyTorch build for the
  graphics card chosen by ``uv --torch-backend auto``), each plugin adding its packages.
  ``ppg-env.json`` records what is there - the versions, the device found (NVIDIA, Apple GPU or the
  processor) and the features installed - and is written last: a failed installation never looks
  ready.

The packaged application, which is not a Python and has no pip, always uses the environment.
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

PYTHON = "3.12"
MARKER = "ppg-env.json"
ENV_VAR = "PPG_GEOAI_ENV"
# the packages of each feature; the modules that must import for it to work
FEATURES = {
    "samgeo": {"packages": ["torch", "torchvision", "transformers", "sam2", "huggingface_hub", "pillow", "numpy"],
               "modules": ["torch", "transformers", "sam2"]},
}
# what a Python of the environment answers (one line of JSON): the versions and the device
CHECK = r"""
import importlib.metadata as md, json, platform
out = {"python": platform.python_version(), "versions": {}}
for name in __MODULES__:
    try:
        __import__(name)
        out["versions"][name] = md.version(name)
    except Exception as exc:
        out.setdefault("errors", {})[name] = type(exc).__name__ + ": " + str(exc)
try:
    import torch
    mps = getattr(torch.backends, "mps", None)
    out["device"] = "cuda" if torch.cuda.is_available() else "mps" if mps is not None and mps.is_available() else "cpu"
except Exception:
    pass
print(json.dumps(out))
"""
DEVICE_NAMES = {"cuda": ("GPU NVIDIA (CUDA)", "NVIDIA GPU (CUDA)"), "mps": ("GPU Apple (Metal)", "Apple GPU (Metal)"),
                "cpu": ("processore", "processor")}


class EnvironmentMissing(RuntimeError):
    """No Python with what a feature needs: the message says why and what to do."""


def data_dir() -> Path:
    """The folder of the program's data (where its own SAM environment and ``uv`` are)."""
    from pyarchinit_photogrammetry.ai.sam_env import data_dir as program_data

    return Path(program_data())


def env_dir() -> Path:
    if os.environ.get(ENV_VAR):
        return Path(os.environ[ENV_VAR]).expanduser().absolute()
    return data_dir() / "geoai-env"


def env_python(env: Path | None = None) -> Path:
    env = Path(env) if env is not None else env_dir()
    return env / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed_info(env: Path | None = None) -> dict:
    """What the environment holds (empty when it is not there): ``python``, ``versions``,
    ``device``, ``features``."""
    try:
        env = Path(env) if env is not None else env_dir()
        if not env_python(env).exists():
            return {}
        return json.loads((env / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def local_available(feature: str = "samgeo") -> bool:
    """The modules of ``feature`` installed for this Python (looked up, not imported). Never in the
    packaged application: its executable is not a Python."""
    if getattr(sys, "frozen", False):
        return False
    try:
        return all(importlib.util.find_spec(m) is not None for m in FEATURES[feature]["modules"])
    except (ImportError, ValueError):
        return False


def where(feature: str = "samgeo") -> str | None:
    """``"local"`` (this Python), ``"environment"`` or None."""
    if local_available(feature):
        return "local"
    if feature in (installed_info().get("features") or []):
        return "environment"
    return None


def device_name(device: str | None) -> str:
    return t(*DEVICE_NAMES.get(device or "cpu", DEVICE_NAMES["cpu"]))


def status_text(feature: str = "samgeo") -> str:
    """Where the models run, in a line for the pane and ``ppg plugin samgeo status``."""
    kind = where(feature)
    if kind == "local":
        return t("Modelli: PyTorch di questo Python, in un processo separato.",
                 "Models: the PyTorch of this Python, in a process of its own.")
    if kind == "environment":
        info = installed_info()
        return t("Modelli: PyTorch {0} nell'ambiente dei plugin AI, su {1}.",
                 "Models: PyTorch {0} in the environment of the AI plugins, on the {1}.").format(
            (info.get("versions") or {}).get("torch", "?"), device_name(info.get("device")))
    return t("PyTorch e i modelli non sono installati: «Installa i modelli» li scarica in un ambiente dei plugin "
             "AI (circa 2 GB, condiviso con geoai); i pesi dei modelli (circa 1 GB) arrivano al primo uso.",
             "PyTorch and the models are not installed: 'Install the models' downloads them into an environment "
             "of the AI plugins (about 2 GB, shared with geoai); the weights of the models (about 1 GB) come at "
             "the first use.")


def worker_python(feature: str = "samgeo") -> Path:
    """The Python the worker runs with: :class:`EnvironmentMissing` when there is none."""
    kind = where(feature)
    if kind == "local":
        return Path(sys.executable)
    if kind == "environment":
        return env_python()
    raise EnvironmentMissing(t("PyTorch e i modelli non sono installati: installali dal pannello Samgeo o con "
                               "«ppg plugin samgeo install».",
                               "PyTorch and the models are not installed: install them from the Samgeo pane or with "
                               "'ppg plugin samgeo install'."))


def clean_environment(for_uv: bool = False) -> dict:
    """The variables of this process without those of its Python (the worker must not take the
    program's paths for its own; ``for_uv``: nor the conda environment, where uv would install)."""
    drop = ("PYTHON", "VIRTUAL_ENV", "__PYVENV") + (("CONDA",) if for_uv else ())
    env = {k: v for k, v in os.environ.items() if not k.startswith(drop)}
    env["PYTHONNOUSERSITE"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def find_uv(progress=None) -> Path:
    """``uv``: on the PATH, in the program's data folder, or downloaded there from its official
    release (as for the program's SAM environment)."""
    found = shutil.which("uv")
    if found:
        return Path(found)
    from pyarchinit_photogrammetry.ai.sam_env import find_uv as program_uv

    return Path(program_uv(progress))


def _run(cmd: list[str], env: dict, progress, start: float, end: float) -> list[str]:
    """``cmd`` (absolute path, the file descriptors inherited: posix_spawn, as every process the
    program starts), its lines moving the bar from ``start`` to ``end``; cancelling stops it."""
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


def probe(python: Path, modules) -> dict:
    """Run :data:`CHECK` for ``modules`` with ``python`` in a process of its own (isolated from the
    program's variables and folder) and return what it answered."""
    code = CHECK.replace("__MODULES__", json.dumps(list(modules)))
    try:
        out = subprocess.run([str(python), "-I", "-c", code], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", env=clean_environment(), close_fds=False, timeout=900)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"errors": {"python": f"{type(exc).__name__}: {exc}"}}
    for line in reversed((out.stdout or "").strip().splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    return {"errors": {"python": ((out.stderr or "") + (out.stdout or "")).strip()[-1500:] or f"code {out.returncode}"}}


def install(progress=None, features=("samgeo",), env: Path | None = None, uv: Path | None = None) -> Path:
    """Make the environment of the AI plugins (when it is not there) and install the packages of
    ``features`` in it, with those it already had; ``uv``: another one, for the tests. Several
    minutes and about 2 GB the first time; returns the folder."""
    from pyarchinit_photogrammetry.ai.sam_env import torch_backend_args
    from pyarchinit_photogrammetry.core.progress import as_progress

    progress = as_progress(progress)
    env = Path(env) if env is not None else env_dir()
    uv = Path(uv) if uv is not None else find_uv(progress)
    before = installed_info(env)
    wanted = list(dict.fromkeys(list(before.get("features") or []) + [f for f in features if f in FEATURES]))
    packages = list(dict.fromkeys(p for f in wanted for p in FEATURES[f]["packages"]))
    modules = list(dict.fromkeys(m for f in wanted for m in FEATURES[f]["modules"]))
    env.parent.mkdir(parents=True, exist_ok=True)
    clean = clean_environment(for_uv=True)
    if env.exists():  # a failure leaves no environment that looks ready
        (env / MARKER).unlink(missing_ok=True)
    progress(0.03, t("Creazione dell'ambiente Python dei plugin AI", "Creating the Python environment of the AI plugins"))
    _run([str(uv), "venv", "--python", PYTHON, "--allow-existing", str(env)], clean, progress, 0.03, 0.1)
    progress(0.1, t("Installazione di PyTorch e dei modelli (alcuni minuti)", "Installing PyTorch and the models (a few minutes)"))
    python = str(env_python(env))
    args = torch_backend_args(uv, clean)
    if args and sys.platform != "darwin" and before.get("device") == "cpu":
        args += ["--reinstall-package", "torch"]  # an older install got the CPU build: look for the GPU again
    if sys.platform == "win32" and platform.machine().upper() == "ARM64":
        # PyTorch for Windows on ARM exists only on its own index, and sam2 (a source archive) wants it
        # while it builds: first PyTorch, then the rest built against it
        _run([str(uv), "pip", "install", "--python", python] + args + ["torch", "torchvision"], clean, progress, 0.1, 0.6)
        _run([str(uv), "pip", "install", "--python", python, "--no-build-isolation"]
             + [p for p in packages if p not in ("torch", "torchvision")], clean, progress, 0.6, 0.9)
    else:
        _run([str(uv), "pip", "install", "--python", python] + args + packages, clean, progress, 0.1, 0.9)
    progress(0.92, t("Prova di PyTorch e dei modelli", "Trying PyTorch and the models"))
    info = probe(env_python(env), modules)
    if info.get("errors"):
        raise RuntimeError(t("Installati, ma non si importano:", "Installed, but they do not import:") + "\n"
                           + "\n".join(f"{k}: {v}" for k, v in info["errors"].items())[-1500:])
    record = {"python": info.get("python"), "versions": info.get("versions") or {}, "device": info.get("device"),
              "features": wanted}
    (env / MARKER).write_text(json.dumps(record, indent=1), encoding="utf-8")
    progress(1.0, t("Modelli pronti (PyTorch {0}, {1})", "Models ready (PyTorch {0}, {1})").format(
        record["versions"].get("torch", "?"), device_name(record["device"])))
    return env

