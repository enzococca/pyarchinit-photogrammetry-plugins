"""Spirula Studio installed by the plugin: the latest release of GitHub for this computer, its
SHA-256 checked against the digest GitHub publishes, put where the plugin looks for it.

The releases (https://github.com/harry7557558/spirula-studio/releases) carry three builds:
``spirula-<version>-macos-vulkan-arm64.dmg`` (a disk image with ``Spirula Studio.app``),
``spirula-<version>-windows-vulkan-x86_64.zip`` and ``spirula-<version>-ubuntu-vulkan-x86_64.zip``
(a zip with the one executable). The application goes into ``/Applications`` (or
``~/Applications`` when that cannot be written), the executable into
``%LOCALAPPDATA%\\Programs\\Spirula Studio`` on Windows and ``~/.local/share/spirula-studio`` on
Linux: the folders :func:`runner.candidates` searches. Intel Macs, Windows on ARM and Linux on
ARM have no build.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

RELEASES_API = "https://api.github.com/repos/harry7557558/spirula-studio/releases/latest"
RELEASES_PAGE = "https://github.com/harry7557558/spirula-studio/releases"
USER_AGENT = "pyarchinit-photogrammetry plugin spirula"
EXE = "spirula.exe" if sys.platform == "win32" else "spirula"


class InstallError(RuntimeError):
    """Spirula Studio could not be installed; the message says why."""


def _ssl_context():
    try:
        import ssl

        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return None


def _open(url: str, timeout: float, accept: str = "application/vnd.github+json"):
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": accept})
    return urllib.request.urlopen(request, timeout=timeout, context=_ssl_context())


def latest_release(url: str = RELEASES_API, timeout: float = 30.0) -> dict:
    """The latest release: ``{"tag", "assets": [{"name", "size", "url", "sha256"}]}``."""
    try:
        with _open(url, timeout) as r:
            data = json.load(r)
    except urllib.error.HTTPError as exc:
        if exc.code in (403, 429):
            raise InstallError("GitHub refused the request (too many requests from this network): try later") from exc
        raise InstallError(f"GitHub answered HTTP {exc.code} for the releases of Spirula Studio") from exc
    except (urllib.error.URLError, OSError) as exc:
        raise InstallError(f"GitHub could not be reached: {getattr(exc, 'reason', exc)}") from exc
    assets = []
    for a in data.get("assets", []):
        digest = str(a.get("digest") or "")
        assets.append({"name": str(a.get("name", "")), "size": int(a.get("size") or 0),
                       "url": str(a.get("browser_download_url", "")),
                       "sha256": digest[7:].lower() if digest.startswith("sha256:") else ""})
    return {"tag": str(data.get("tag_name", "")), "assets": assets}


def platform_key(system: str | None = None, machine: str | None = None) -> tuple[str, str] | None:
    """``(os, arch)`` as the release files are named, ``None`` when Spirula has no build for it."""
    system = system or sys.platform
    machine = (machine or platform.machine()).lower()
    arm = machine in ("arm64", "aarch64")
    x86 = machine in ("x86_64", "amd64")
    if system == "darwin" and arm:
        return "macos", "arm64"
    if system == "win32" and x86:
        return "windows", "x86_64"
    if system.startswith("linux") and x86:
        return "ubuntu", "x86_64"
    return None


def choose_asset(assets: list[dict], system: str | None = None, machine: str | None = None) -> dict | None:
    """The release file for this computer (the Vulkan build), ``None`` without one."""
    key = platform_key(system, machine)
    if key is None:
        return None
    os_name, arch = key
    suffix = ".dmg" if os_name == "macos" else ".zip"
    found = [a for a in assets if os_name in a["name"] and arch in a["name"] and a["name"].endswith(suffix)]
    found.sort(key=lambda a: "vulkan" not in a["name"])
    return found[0] if found else None


def default_folder(system: str | None = None) -> Path:
    """Where the release goes: the applications folder on macOS, the user's programs on Windows,
    ``~/.local/share/spirula-studio`` on Linux."""
    system = system or sys.platform
    home = Path.home()
    if system == "darwin":
        apps = Path("/Applications")
        return apps if os.access(apps, os.W_OK) else home / "Applications"
    if system == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(base) / "Programs" / "Spirula Studio"
    return home / ".local" / "share" / "spirula-studio"


def download(url: str, path: Path, progress=None, sha256: str = "", size: int = 0, timeout: float = 120.0) -> Path:
    """``url`` streamed into ``path``; its SHA-256 must match ``sha256`` when given."""
    progress = progress or (lambda *a: None)
    digest = hashlib.sha256()
    try:
        with _open(url, timeout, accept="application/octet-stream") as r, open(path, "wb") as out:
            total = int(r.headers.get("Content-Length") or size or 0)
            done = 0
            while chunk := r.read(1 << 16):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if total:
                    progress(min(1.0, done / total), f"{path.name}: {done / 1e6:.0f} / {total / 1e6:.0f} MB")
    except (urllib.error.URLError, OSError) as exc:
        raise InstallError(f"{path.name} could not be downloaded: {getattr(exc, 'reason', exc)}") from exc
    if sha256 and digest.hexdigest() != sha256.lower():
        raise InstallError(f"{path.name} does not match the SHA-256 GitHub publishes for it")
    return path


def _find_executable(folder: Path) -> Path | None:
    direct = folder / EXE
    if direct.is_file():
        return direct
    found = sorted(p for p in folder.rglob(EXE) if p.is_file())
    return found[0] if found else None


def install_zip(archive: Path, folder: Path, progress=None) -> Path:
    """The zip of a Windows or Linux release unpacked into ``folder``; the executable found."""
    progress = progress or (lambda *a: None)
    folder.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive) as z:
            for member in z.namelist():
                p = PurePosixPath(member)
                if p.is_absolute() or ".." in p.parts:
                    raise InstallError(f"the archive holds an unsafe path: {member}")
            progress(0.5, "Unpacking")
            z.extractall(folder)
    except zipfile.BadZipFile as exc:
        raise InstallError(f"{archive.name} is not a valid archive") from exc
    exe = _find_executable(folder)
    if exe is None:
        raise InstallError(f"no {EXE} in {archive.name}")
    if sys.platform != "win32":
        exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def _run(cmd: list[str], timeout: float = 600.0) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, close_fds=False)


def install_dmg(dmg: Path, folder: Path, progress=None) -> Path:
    """The application of a macOS disk image copied into ``folder``; the executable inside it."""
    progress = progress or (lambda *a: None)
    folder.mkdir(parents=True, exist_ok=True)
    mount = Path(tempfile.mkdtemp(prefix="spirula-dmg-"))
    progress(0.3, "Opening the disk image")
    r = _run(["/usr/bin/hdiutil", "attach", "-nobrowse", "-readonly", "-noautoopen", "-mountpoint", str(mount), str(dmg)])
    if r.returncode != 0:
        shutil.rmtree(mount, ignore_errors=True)
        raise InstallError(f"the disk image could not be opened: {(r.stderr or r.stdout).strip()}")
    try:
        apps = sorted(p for p in mount.glob("*.app") if p.is_dir())
        if not apps:
            raise InstallError("the disk image holds no application")
        target = folder / apps[0].name
        progress(0.6, f"Copying {apps[0].name}")
        if target.exists():
            shutil.rmtree(target)
        r = _run(["/usr/bin/ditto", str(apps[0]), str(target)])
        if r.returncode != 0:
            shutil.rmtree(target, ignore_errors=True)
            raise InstallError(f"the application could not be copied: {(r.stderr or r.stdout).strip()}")
    finally:
        if _run(["/usr/bin/hdiutil", "detach", "-quiet", str(mount)], timeout=120).returncode != 0:
            _run(["/usr/bin/hdiutil", "detach", "-force", "-quiet", str(mount)], timeout=120)
        shutil.rmtree(mount, ignore_errors=True)
    exe = target / "Contents" / "MacOS" / "spirula"
    if not exe.is_file():
        raise InstallError(f"{target.name} holds no spirula executable")
    return exe


def install(folder=None, progress=None, release: dict | None = None, fetch=None) -> dict:
    """Download the Spirula Studio of this computer and install it into ``folder`` (the default of
    the system). ``release`` and ``fetch(url, path, progress, sha256, size)`` replace GitHub in the
    tests. Returns the executable, the version and the file installed."""
    progress = progress or (lambda *a: None)
    progress(0.0, "Spirula Studio: the latest release")
    release = release or latest_release()
    asset = choose_asset(release["assets"])
    if asset is None:
        raise InstallError(f"no Spirula Studio build for this computer ({platform.system()} {platform.machine()}): "
                           f"the releases have macOS on Apple Silicon, Windows x86_64 and Linux x86_64 ({RELEASES_PAGE})")
    folder = Path(folder).expanduser() if folder else default_folder()
    work = Path(tempfile.mkdtemp(prefix="spirula-install-"))
    try:
        archive = work / asset["name"]
        progress(0.02, f"Spirula Studio {release['tag']}: {asset['size'] / 1e6:.0f} MB")
        (fetch or download)(asset["url"], archive, lambda f, m: progress(0.02 + 0.78 * f, m), asset.get("sha256", ""),
                            asset["size"])
        progress(0.82, "Installing")
        if archive.suffix.lower() == ".dmg":
            exe = install_dmg(archive, folder, lambda f, m: progress(0.82 + 0.15 * f, m))
        else:
            exe = install_zip(archive, folder, lambda f, m: progress(0.82 + 0.15 * f, m))
    finally:
        shutil.rmtree(work, ignore_errors=True)
    progress(1.0, f"Spirula Studio {release['tag']}: {exe}")
    return {"executable": exe, "version": release["tag"], "asset": asset["name"], "folder": folder}
