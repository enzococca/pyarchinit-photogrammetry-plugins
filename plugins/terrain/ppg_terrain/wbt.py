"""WhiteboxTools (John Lindsay, University of Guelph, MIT), run as a program: the binary of this
computer is downloaded once into the plugin's folder (or found in the ``whitebox`` Python package,
on the PATH, or where ``PPG_WBT`` says) and every tool is a call of ``whitebox_tools --run=Tool``
with its parameters. The tool's output is read for its progress and for its errors: the program
prints them and exits with 0, so the lines are what tells.

Files are given with absolute paths (the working directory of the binary is not used: with it the
tools did not find their inputs) and must be GeoTIFFs with GeoKeys: a DEM of a local project has no
CRS, so :mod:`.terrain` writes the copies the tools read.
"""
from __future__ import annotations

import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    ("Darwin", "arm64"): "https://www.whiteboxgeo.com/WBT_Darwin/WhiteboxTools_darwin_m_series.zip",
    ("Darwin", "x86_64"): "https://www.whiteboxgeo.com/WBT_Darwin/WhiteboxTools_darwin_amd64.zip",
    ("Windows", "AMD64"): "https://www.whiteboxgeo.com/WBT_Windows/WhiteboxTools_win_amd64.zip",
    ("Linux", "x86_64"): "https://www.whiteboxgeo.com/WBT_Linux/WhiteboxTools_linux_amd64.zip",
}
USER_AGENT = "pyarchinit-photogrammetry plugin terrain"
_DIR: Path | None = None  # the plugin's folder (configure())
_PROGRESS = re.compile(r"(\d{1,3})%")


class ToolError(RuntimeError):
    """A tool of WhiteboxTools failed: the message holds what it said."""


class WhiteboxMissing(RuntimeError):
    """No binary of WhiteboxTools and none could be downloaded."""


def configure(context) -> None:
    """Remember the plugin's folder, where the binary is kept (``context.data_dir / "wbt"``)."""
    global _DIR
    _DIR = Path(context.data_dir) / "wbt"


def exe_name() -> str:
    return "whitebox_tools.exe" if platform.system() == "Windows" else "whitebox_tools"


def binary() -> Path | None:
    """The binary of WhiteboxTools, or None: ``PPG_WBT`` (the file or its folder), the plugin's
    folder, the ``whitebox`` Python package when installed, the PATH."""
    name = exe_name()
    env = os.environ.get("PPG_WBT")
    if env:
        p = Path(env)
        for c in (p, p / name, p / "WBT" / name):
            if c.is_file():
                return c
    if _DIR is not None:
        for c in (_DIR / "WBT" / name, _DIR / name):
            if c.is_file():
                return c
    try:
        import whitebox  # the pip package carries the binary beside its module once used

        pkg = Path(whitebox.__file__).parent
        for c in (pkg / name, pkg / "WBT" / name):
            if c.is_file():
                return c
    except Exception:  # noqa: BLE001
        pass
    found = shutil.which("whitebox_tools")
    return Path(found) if found else None


def download_url() -> str:
    system, machine = platform.system(), platform.machine()
    if system == "Darwin" and machine in ("arm64", "aarch64"):
        return URLS[("Darwin", "arm64")]
    if system == "Darwin":
        return URLS[("Darwin", "x86_64")]
    if system == "Windows":
        return URLS[("Windows", "AMD64")]
    if system == "Linux" and machine in ("x86_64", "AMD64"):
        return URLS[("Linux", "x86_64")]
    raise WhiteboxMissing(f"WhiteboxTools has no build for {system} {machine}: install it yourself and point "
                          "PPG_WBT to the binary")


def install(progress=None, folder: Path | None = None) -> Path:
    """Download the binary of this computer into the plugin's folder (``folder`` to choose
    another) and return it."""
    folder = Path(folder) if folder is not None else _DIR
    if folder is None:
        raise WhiteboxMissing("the plugin has no folder to install WhiteboxTools into")
    url = download_url()
    folder.mkdir(parents=True, exist_ok=True)
    if progress:
        progress(0.0, "Downloading WhiteboxTools")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with tempfile.TemporaryDirectory() as tmp:
        zip_path = Path(tmp) / "wbt.zip"
        with urllib.request.urlopen(req, timeout=120) as r, open(zip_path, "wb") as f:
            total = int(r.headers.get("Content-Length") or 0)
            done = 0
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if progress and total:
                    progress(0.9 * done / total, f"Downloading WhiteboxTools: {done / 1e6:.0f} of {total / 1e6:.0f} MB")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(folder)
    exe = None
    for c in folder.rglob(exe_name()):
        exe = c
        break
    if exe is None:
        raise WhiteboxMissing(f"the archive of WhiteboxTools from {url} holds no {exe_name()}")
    if platform.system() != "Windows":
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    # the binary at a fixed place: <folder>/WBT/whitebox_tools
    target = folder / "WBT" / exe_name()
    if exe.resolve() != target.resolve():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(exe.parent, target.parent, dirs_exist_ok=True)
    if progress:
        progress(1.0, "WhiteboxTools ready")
    return target


def ensure(progress=None) -> Path:
    """The binary, downloaded when missing."""
    exe = binary()
    if exe is None:
        exe = install(progress)
    return exe


def available() -> bool:
    return binary() is not None


def camel(tool: str) -> str:
    """``fill_depressions`` -> ``FillDepressions`` (a name already in CamelCase stays)."""
    if "_" not in tool and tool[:1].isupper():
        return tool
    return "".join(part[:1].upper() + part[1:] for part in tool.split("_"))


def _spawn(args: list[str]) -> subprocess.CompletedProcess:
    # posix_spawn with an absolute path and the file descriptors inherited: a fork after
    # SpatiaLite was loaded can die in the child (see the program's notes)
    return subprocess.run(args, capture_output=True, text=True, close_fds=False)


def run(tool: str, progress=None, **params) -> list[str]:
    """Run ``tool`` with ``params`` (``dem="/a/dem.tif"``, ``fix_flats=True``; booleans become
    flags) and return its lines; the fractions it prints go to ``progress``; a failure raises
    :class:`ToolError`."""
    exe = ensure(progress)
    args = [str(exe), f"--run={camel(tool)}", "-v"]
    for k, v in params.items():
        if v is None or v is False:
            continue
        args.append(f"--{k}" if v is True else f"--{k}={v}")
    outputs = [str(v) for k, v in params.items() if k in ("output", "out_mag") and isinstance(v, str)]
    proc = lines = None
    for attempt in range(3):
        # a tool that fails on its threads is run again, single-threaded: FillDepressions 2.4 dies
        # about once in six runs ("Error unwrapping 'output'") with many threads, never with one
        extra = ["--max_procs=1"] if attempt and "--max_procs" not in " ".join(args) else []
        proc = _spawn(args + extra)
        lines = ([ln.rstrip() for ln in (proc.stdout or "").splitlines()]
                 + [ln.rstrip() for ln in (proc.stderr or "").splitlines()])
        failed = proc.returncode != 0 or any("panicked" in ln for ln in lines) or any(not Path(o).exists() for o in outputs)
        if not failed:
            break
    if progress:
        for ln in lines:
            m = _PROGRESS.search(ln)
            if m and ("rogress" in ln or ln.strip().endswith("%")):
                try:
                    progress(min(1.0, int(m.group(1)) / 100.0), f"{camel(tool)}: {ln.strip()[:60]}")
                except Exception:  # noqa: BLE001 - a cancelled step: the tool already ran
                    break
    bad = [ln for ln in lines if "panicked" in ln or ln.startswith("Error") or "error:" in ln.lower()]
    if proc.returncode != 0 or bad:
        detail = bad[0] if bad else (lines[-1] if lines else f"exit code {proc.returncode}")
        if bad and bad[0].rstrip().endswith(":") and lines.index(bad[0]) + 1 < len(lines):
            detail += " " + lines[lines.index(bad[0]) + 1]  # the panic's message is on the next line
        raise ToolError(f"{camel(tool)}: {detail.strip()}")
    for o in outputs:
        if not Path(o).exists():
            raise ToolError(f"{camel(tool)} wrote no {Path(o).name}: " + " ".join(lines[-3:])[:300])
    return lines


def version() -> str:
    exe = binary()
    if exe is None:
        return ""
    proc = _spawn([str(exe), "--version"])
    return (proc.stdout or "").strip().splitlines()[0] if proc.stdout else ""


def tool_parameters(tool: str) -> dict:
    """The parameters of a tool as WhiteboxTools describes them (``{"parameters": [...]}``)."""
    import json

    exe = ensure()
    proc = _spawn([str(exe), f"--toolparameters={camel(tool)}"])
    return json.loads(proc.stdout or "{}")


def python_version_note() -> str:  # pragma: no cover - for the doctor line
    return f"Python {sys.version_info.major}.{sys.version_info.minor}"
