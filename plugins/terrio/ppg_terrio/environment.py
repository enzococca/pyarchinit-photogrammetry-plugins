"""terrio itself: imported from the Python environment, or installed by the plugin into its own
folder from PyPI when the program has no pip (the packaged application).

terrio is a compiled module (pure Rust, no C library to install) published as one ``abi3`` wheel
per platform: macOS (Apple Silicon and Intel), Windows (x64 and ARM64), Linux (x86_64 and
aarch64, glibc and musl). :func:`install` picks the wheel of this computer from the PyPI
listing of :data:`TERRIO_VERSION`, checks its SHA-256 against the one PyPI publishes, unpacks it
into ``<plugin data>/lib`` and puts that folder on ``sys.path``; :func:`load` imports terrio from
the environment first, then from there.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import shutil
import sys
import sysconfig
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

TERRIO_VERSION = "0.4.0"  # the version the plugin is written and tested with
OLDEST, NEWEST = (0, 3, 2), (0, 5)  # accepted: from 0.3.2 up to (not including) 0.5
PYPI = "https://pypi.org/pypi/terrio/{version}/json"
USER_AGENT = "pyarchinit-photogrammetry plugin terrio"

_LIB_DIR: Path | None = None  # the plugin folder where install() puts terrio (configure())


class TerrioMissing(RuntimeError):
    """terrio cannot be imported; the message says why and how to get it."""


def configure(context) -> None:
    """Remember the plugin folder (``context.data_dir / "lib"``) where terrio may be installed."""
    global _LIB_DIR
    _LIB_DIR = Path(context.data_dir) / "lib"


def lib_dir() -> Path | None:
    return _LIB_DIR


def _key(version: str) -> tuple:
    out = []
    for part in str(version).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _import():
    import terrio

    version = getattr(terrio, "__version__", "0")
    if not (OLDEST <= _key(version) < NEWEST):
        raise TerrioMissing(f"terrio {version} is installed, this plugin needs terrio {TERRIO_VERSION} "
                            f"(0.3.2 up to the 0.4 releases): pip install 'terrio>=0.3.2,<0.5'")
    return terrio


def load():
    """The terrio module (version checked), from the environment or from the plugin folder."""
    try:
        return _import()
    except ImportError as first:
        error = first
    folder = _LIB_DIR
    if folder is not None and folder.is_dir():
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
            importlib.invalidate_caches()
        try:
            return _import()
        except ImportError as exc:
            error = exc
    raise TerrioMissing(f"terrio is not installed ({error}): pip install terrio=={TERRIO_VERSION}, "
                        "or Plugin > terrio > Installa terrio in the window") from error


def available() -> str:
    """The version of terrio the plugin can use, "" when none."""
    try:
        return str(load().__version__)
    except TerrioMissing:
        return ""


def platform_tags() -> list[str]:
    """The wheel platform tags this interpreter accepts, the most specific first (``packaging``
    when the program carries it, else the tags of the terrio wheels for this system)."""
    try:
        from packaging import tags

        seen, out = set(), []
        for t in tags.sys_tags():
            if t.platform not in seen:
                seen.add(t.platform)
                out.append(t.platform)
        if out:
            return out
    except Exception:  # packaging missing or unusual platform
        pass
    machine = platform.machine().lower()
    arm = machine in ("arm64", "aarch64")
    if sys.platform == "darwin":
        return ["macosx_11_0_arm64"] if arm else ["macosx_10_12_x86_64"]
    if sys.platform == "win32":
        return ["win_arm64"] if arm else ["win_amd64"]
    arch = "aarch64" if arm else "x86_64"
    if "musl" in (sysconfig.get_config_var("HOST_GNU_TYPE") or ""):
        return [f"musllinux_1_2_{arch}"]
    return [f"manylinux_2_28_{arch}", f"manylinux_2_17_{arch}", f"manylinux2014_{arch}"]


def choose_wheel(files: list[dict], accepted: list[str] | None = None, python=None) -> dict | None:
    """The file of a PyPI listing (``urls`` of the JSON) that is an ``abi3`` wheel for this
    Python and one of the ``accepted`` platform tags (the first accepted tag wins)."""
    accepted = accepted or platform_tags()
    python = python or sys.version_info[:2]
    rank = {t: i for i, t in enumerate(accepted)}
    best = None
    for f in files:
        name = str(f.get("filename", ""))
        if not name.endswith(".whl"):
            continue
        parts = name[:-4].split("-")
        if len(parts) != 5:
            continue
        tag, abi, plat = parts[2:]
        if abi != "abi3" or not tag.startswith("cp3") or not tag[3:].isdigit():
            continue
        if tuple(python) < (3, int(tag[3:])):
            continue
        score = min((rank[p] for p in plat.split(".") if p in rank), default=None)
        if score is not None and (best is None or score < best[0]):
            best = (score, f)
    return None if best is None else best[1]


def _get(url: str, timeout: float):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=timeout)


def install(progress=None, version: str = TERRIO_VERSION, files: list[dict] | None = None) -> Path:
    """Download the wheel of terrio for this computer from PyPI (its SHA-256 checked against the
    one PyPI publishes), unpack it into the plugin folder and make it importable at once.
    ``files``: the PyPI listing, when already read. Returns the folder."""
    if _LIB_DIR is None:
        raise TerrioMissing("the plugin has no folder to install terrio into (configure() first)")
    progress = progress or (lambda *a: None)
    progress(0.0, f"terrio {version}: PyPI")
    if files is None:
        with _get(PYPI.format(version=version), 30) as r:
            files = json.load(r).get("urls", [])
    wheel = choose_wheel(files)
    if wheel is None:
        raise TerrioMissing(f"no wheel of terrio {version} for this computer ({platform.system()} "
                            f"{platform.machine()}, Python {sys.version_info[0]}.{sys.version_info[1]})")
    folder = _LIB_DIR
    folder.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".terrio-", dir=folder.parent))
    try:
        path = work / str(wheel["filename"])
        digest = hashlib.sha256()
        with _get(str(wheel["url"]), 120) as r, open(path, "wb") as out:
            total = int(r.headers.get("Content-Length") or wheel.get("size") or 0)
            done = 0
            while chunk := r.read(1 << 16):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    progress(0.05 + 0.85 * done / total, f"terrio {version}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
        expected = str((wheel.get("digests") or {}).get("sha256", "")).lower()
        if expected and digest.hexdigest() != expected:
            raise TerrioMissing("the wheel downloaded does not match its SHA-256 on PyPI")
        unpacked = work / "unpacked"
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                p = PurePosixPath(member)
                if p.is_absolute() or ".." in p.parts:
                    raise TerrioMissing(f"the wheel holds an unsafe path: {member}")
            z.extractall(unpacked)
        if folder.exists():
            shutil.rmtree(folder)
        unpacked.rename(folder)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    for name in [m for m in sys.modules if m == "terrio" or m.startswith("terrio.")]:
        del sys.modules[name]
    if str(folder) in sys.path:
        sys.path.remove(str(folder))
    sys.path.insert(0, str(folder))
    importlib.invalidate_caches()
    progress(1.0, f"terrio {version} installed in {folder}")
    return folder
