"""Blender found and started on the EM project, with the live script.

Blender is found where it is installed (``/Applications/Blender*.app``, ``C:\\Program Files\\Blender
Foundation\\Blender *``, ``blender`` on the PATH, ``/snap/bin``, ``~/blender*``), or where
``PPG_BLENDER`` or the settings of the plugin say. *Apri in Blender* starts it as::

    blender [FILE.blend] --python blender_live.py -- ppg_scene.json [--emtools]

and the script (:mod:`.blender_live`, run by Blender's own Python) imports the objects of the
manifest and follows it.
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

LIVE_SCRIPT = Path(__file__).with_name("blender_live.py")


def _version(path: str) -> tuple:
    """The version in a path (``Blender 4.2``, ``blender-5.1.0-linux-x64``), for the newest first."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", path)
    return tuple(int(g or 0) for g in m.groups()) if m else ()


def candidates() -> list[str]:
    """The Blender executables of this computer, newest first."""
    found = []
    if sys.platform == "darwin":
        for app in sorted(glob.glob("/Applications/Blender*.app") + glob.glob(os.path.expanduser("~/Applications/Blender*.app")),
                          key=_version, reverse=True):
            found.append(os.path.join(app, "Contents", "MacOS", "Blender"))
    elif os.name == "nt":
        roots = [os.environ.get("PROGRAMFILES", r"C:\Program Files"), os.environ.get("LOCALAPPDATA", "")]
        for root in roots:
            if root:
                found += sorted(glob.glob(os.path.join(root, "Blender Foundation", "Blender*", "blender.exe")),
                                key=_version, reverse=True)
        steam = os.path.join(os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"), "Steam", "steamapps",
                             "common", "Blender", "blender.exe")
        found.append(steam)
    else:
        found += sorted(glob.glob(os.path.expanduser("~/blender*/blender")) + glob.glob("/opt/blender*/blender"),
                        key=_version, reverse=True)
        found += ["/snap/bin/blender", "/usr/bin/blender", "/usr/local/bin/blender"]
    on_path = shutil.which("blender")
    if on_path:
        found.insert(0 if os.name != "nt" else len(found), on_path)
    out = []
    for f in found:
        if f and f not in out and os.path.isfile(f):
            out.append(f)
    return out


def find_blender(configured: str | None = None) -> str | None:
    """The Blender to start: ``configured`` (the settings of the plugin), ``PPG_BLENDER``, else the
    newest found."""
    for path in (configured, os.environ.get("PPG_BLENDER")):
        if path and os.path.isfile(path):
            return path
    found = candidates()
    return found[0] if found else None


def command(blender: str, manifest: str | Path, blend: str | Path | None = None, emtools: bool = False) -> list[str]:
    """The command line that opens ``blend`` (or a new scene) with the live script on ``manifest``."""
    cmd = [blender]
    if blend:
        cmd.append(str(blend))
    cmd += ["--python", str(LIVE_SCRIPT), "--", str(Path(manifest).resolve())]
    if emtools:
        cmd.append("--emtools")
    return cmd


def environment(env: dict | None = None) -> dict:
    """The environment Blender starts in: without the Python and Qt of the program (Blender has its own),
    and with the library paths the packaged application changed given back as they were (PyInstaller
    keeps them in ``*_ORIG``)."""
    env = dict(os.environ if env is None else env)
    for key in [k for k in env if k.startswith(("PYTHON", "QT_"))]:
        del env[key]
    for key in ("LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH"):
        if key + "_ORIG" in env:
            env[key] = env.pop(key + "_ORIG")
        elif getattr(sys, "frozen", False):
            env.pop(key, None)
    return env


def launch(manifest: str | Path, blender: str | None = None, blend: str | Path | None = None,
           emtools: bool = False) -> subprocess.Popen:
    """Blender on the EM project; raises ``RuntimeError`` when there is no Blender."""
    blender = find_blender(blender)
    if blender is None:
        raise RuntimeError("Blender non trovato: installalo (blender.org) o indica il programma nelle impostazioni "
                           "del plugin (o con PPG_BLENDER)")
    env = environment()
    return subprocess.Popen(command(blender, manifest, blend, emtools), env=env, close_fds=False,  # posix_spawn
                            cwd=str(Path(manifest).resolve().parent))


__all__ = ["LIVE_SCRIPT", "candidates", "find_blender", "command", "launch"]
