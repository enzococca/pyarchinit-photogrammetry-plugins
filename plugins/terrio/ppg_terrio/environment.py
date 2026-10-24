"""terrio itself: imported from the Python environment, or installed by the plugin into its own
folder from PyPI when the program has no pip (the packaged application).

terrio is a compiled module (pure Rust, no C library to install) published as one ``abi3`` wheel
per platform: macOS (Apple Silicon and Intel), Windows (x64 and ARM64), Linux (x86_64 and
aarch64, glibc and musl). :func:`install` picks the wheel of this computer from the PyPI
listing of :data:`TERRIO_VERSION`, checks its SHA-256 against the one PyPI publishes, unpacks it
into ``<plugin data>/lib`` and puts that folder on ``sys.path``; :func:`load` imports the newer
of the two: the terrio of the environment or the one of the plugin folder (a terrio downloaded to
update the one the application carries wins over it).

The plugin works with terrio from 0.3.2 up to the 0.5 releases; the steps added in 0.2.0 of the
plugin (cleaning, colours from rasters, heights, Planetary Computer) need terrio 0.5
(:data:`FEATURES_SINCE`) and say so when an older one is installed (:class:`TerrioTooOld`).
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

TERRIO_VERSION = "0.5.0"  # the version the plugin is written and tested with
OLDEST, NEWEST = (0, 3, 2), (0, 6)  # accepted: from 0.3.2 up to (not including) 0.6
FEATURES_SINCE = (0, 5)  # LOF, DBSCAN, colorization, heights through geoid grids, Planetary Computer signing
PYPI = "https://pypi.org/pypi/terrio/{version}/json"
USER_AGENT = "pyarchinit-photogrammetry plugin terrio"

_DATA_DIR: Path | None = None  # the folder of the plugin's files (configure())
_LIB_DIR: Path | None = None  # the plugin folder where install() puts terrio


class TerrioMissing(RuntimeError):
    """terrio cannot be imported; the message says why and how to get it."""


class TerrioTooOld(TerrioMissing):
    """terrio is there but older than a step needs (``found``, ``wanted``)."""

    def __init__(self, message: str, found: str = "", wanted: str = ""):
        super().__init__(message)
        self.found, self.wanted = found, wanted


def configure(context) -> None:
    """Remember the plugin folder (``context.data_dir``): terrio may be installed in its ``lib``,
    the geoid and datum grids are kept in its ``grids``."""
    global _DATA_DIR, _LIB_DIR
    _DATA_DIR = Path(context.data_dir)
    _LIB_DIR = _DATA_DIR / "lib"


def lib_dir() -> Path | None:
    return _LIB_DIR


def grids_dir() -> Path | None:
    """Where the plugin keeps the PROJ grids (geoid models, datum shifts) it downloads:
    ``<plugin data>/grids``; ``None`` before :func:`configure` (terrio's own cache is used then)."""
    return None if _DATA_DIR is None else _DATA_DIR / "grids"


def _key(version: str) -> tuple:
    out = []
    for part in str(version).split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _text(it: str, en: str) -> str:
    from .texts import t

    return t(it, en)


def _import(minimum: tuple | None = None, feature: str = ""):
    import terrio

    version = getattr(terrio, "__version__", "0")
    if not (OLDEST <= _key(version) < NEWEST):
        raise TerrioMissing(f"terrio {version} is installed, this plugin needs terrio {TERRIO_VERSION} "
                            f"(0.3.2 up to the 0.5 releases): pip install 'terrio>=0.5,<0.6'")
    if minimum is not None and _key(version) < tuple(minimum):
        from .texts import needs_newer

        wanted = ".".join(str(v) for v in minimum)
        raise TerrioTooOld(needs_newer(feature or _text("Questo passo", "This step"), version, wanted),
                           found=version, wanted=wanted)
    return terrio


def folder_version(folder: Path | None) -> str:
    """The version of the terrio unpacked in ``folder`` (its ``.dist-info``), "" when none."""
    if folder is None or not folder.is_dir():
        return ""
    found = sorted(folder.glob("terrio-*.dist-info"))
    return found[-1].name[len("terrio-"):-len(".dist-info")] if found else ""


def _environment_version(folder: Path | None) -> str:
    """The version of the terrio of the Python environment (the plugin folder left out)."""
    from importlib import metadata

    skip = str(folder) if folder is not None else None
    for dist in metadata.distributions(path=[p for p in sys.path if p != skip], name="terrio"):
        return str(dist.version)
    return ""


def _prefer_newest() -> None:
    """The plugin folder first on ``sys.path`` when its terrio is newer than the environment's
    (a terrio downloaded to update the one the application carries), before the first import."""
    folder = _LIB_DIR
    mine = folder_version(folder)
    if not mine or "terrio" in sys.modules:
        return
    theirs = _environment_version(folder)
    if not theirs or _key(mine) > _key(theirs):
        if str(folder) in sys.path:
            sys.path.remove(str(folder))
        sys.path.insert(0, str(folder))
        importlib.invalidate_caches()


def load(minimum: tuple | None = None, feature: str = ""):
    """The terrio module (version checked), the newer of the environment's and the plugin
    folder's. ``minimum``: the version a step needs (:class:`TerrioTooOld` otherwise, the message
    naming ``feature``)."""
    _prefer_newest()
    try:
        return _import(minimum, feature)
    except ImportError as first:
        error = first
    folder = _LIB_DIR
    if folder is not None and folder.is_dir():
        if str(folder) not in sys.path:
            sys.path.insert(0, str(folder))
            importlib.invalidate_caches()
        try:
            return _import(minimum, feature)
        except ImportError as exc:
            error = exc
    raise TerrioMissing(f"terrio is not installed ({error}): pip install terrio=={TERRIO_VERSION}, "
                        "or Plugin > terrio > Installa terrio in the window") from error


def load_features(feature: str = ""):
    """terrio for the steps that need :data:`FEATURES_SINCE` (0.5)."""
    return load(FEATURES_SINCE, feature)


def available(minimum: tuple | None = None) -> str:
    """The version of terrio the plugin can use (at least ``minimum``), "" when none."""
    try:
        return str(load(minimum).__version__)
    except TerrioMissing:
        return ""


def installed() -> str:
    """The version of the terrio found, even when too old or too new for the plugin; "" when none."""
    try:
        return str(load().__version__)
    except TerrioMissing:
        pass
    try:
        import terrio

        return str(getattr(terrio, "__version__", ""))
    except ImportError:
        return folder_version(_LIB_DIR)


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
