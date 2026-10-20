"""Spirula Studio installed by the plugin: the release file chosen for the computer, the zip of
Windows and Linux unpacked (a stand-in executable), the disk image of macOS copied (on a Mac),
the SHA-256 checked, the folders the plugin then searches."""
from __future__ import annotations

import hashlib
import shutil
import stat
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
from ppg_spirula import installer, runner

ASSETS = [{"name": "spirula-2026.10.16-macos-vulkan-arm64.dmg", "size": 59534616, "url": "u1", "sha256": "a"},
          {"name": "spirula-2026.10.16-ubuntu-vulkan-x86_64.zip", "size": 52931848, "url": "u2", "sha256": "b"},
          {"name": "spirula-2026.10.16-windows-vulkan-x86_64.zip", "size": 49168988, "url": "u3", "sha256": "c"},
          {"name": "spirula-2026.10.16-windows-cuda-x86_64.zip", "size": 1, "url": "u4", "sha256": "d"}]


def test_choose_asset_by_platform():
    pick = lambda s, m: (installer.choose_asset(ASSETS, s, m) or {}).get("name", None)  # noqa: E731
    assert pick("darwin", "arm64") == "spirula-2026.10.16-macos-vulkan-arm64.dmg"
    assert pick("win32", "AMD64") == "spirula-2026.10.16-windows-vulkan-x86_64.zip", "Vulkan before CUDA"
    assert pick("linux", "x86_64") == "spirula-2026.10.16-ubuntu-vulkan-x86_64.zip"
    assert pick("darwin", "x86_64") is None and pick("win32", "ARM64") is None and pick("linux", "aarch64") is None
    assert installer.platform_key("linux2", "amd64") == ("ubuntu", "x86_64")
    assert installer.default_folder("win32").name == "Spirula Studio"
    assert installer.default_folder("linux") == Path.home() / ".local" / "share" / "spirula-studio"
    assert installer.default_folder("darwin").name == "Applications"


def _fake_zip(path: Path, name: str) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(name, "#!/bin/sh\necho 'usage: spirula'\n")
    return path


def _fetch_from(source: Path):
    def fetch(url, path, progress, sha256, size):
        progress(1.0, "copied")
        shutil.copy2(source, path)
        if sha256 and hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise installer.InstallError("sha mismatch")
    return fetch


def test_install_zip_release(tmp_path, monkeypatch):
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    archive = _fake_zip(tmp_path / "spirula-1-ubuntu-vulkan-x86_64.zip", "spirula")
    release = {"tag": "v1", "assets": [{"name": archive.name, "size": archive.stat().st_size, "url": "file",
                                        "sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}]}
    seen = []
    report = installer.install(tmp_path / "dest", lambda f, m: seen.append(f), release=release, fetch=_fetch_from(archive))
    exe = report["executable"]
    assert exe == tmp_path / "dest" / "spirula" and exe.is_file() and report["version"] == "v1"
    assert seen == sorted(seen) and seen[-1] == 1.0
    if sys.platform != "win32":
        assert exe.stat().st_mode & stat.S_IXUSR
        assert "usage: spirula" in subprocess.run([str(exe)], capture_output=True, text=True).stdout
    # the SHA-256 of GitHub is checked
    release["assets"][0]["sha256"] = "0" * 64
    with pytest.raises(installer.InstallError):
        installer.install(tmp_path / "dest2", release=release, fetch=_fetch_from(archive))
    # an archive with a path outside its folder is refused
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(bad, "w") as z:
        z.writestr("../spirula", "x")
    with pytest.raises(installer.InstallError, match="unsafe"):
        installer.install_zip(bad, tmp_path / "dest3")
    # no build for this computer
    monkeypatch.setattr(installer.platform, "machine", lambda: "aarch64")
    with pytest.raises(installer.InstallError, match="no Spirula Studio build"):
        installer.install(tmp_path / "dest4", release=release, fetch=_fetch_from(archive))


@pytest.mark.skipif(sys.platform != "darwin", reason="a disk image needs macOS")
def test_install_dmg_release(tmp_path):
    src = tmp_path / "image"
    app = src / "Spirula Studio.app" / "Contents" / "MacOS"
    app.mkdir(parents=True)
    exe = app / "spirula"
    exe.write_text("#!/bin/sh\necho 'usage: spirula'\n")
    exe.chmod(0o755)
    dmg = tmp_path / "spirula-1-macos-vulkan-arm64.dmg"
    r = subprocess.run(["/usr/bin/hdiutil", "create", "-quiet", "-srcfolder", str(src), "-volname", "Spirula Studio",
                        "-format", "UDZO", str(dmg)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    release = {"tag": "v1", "assets": [{"name": dmg.name, "size": dmg.stat().st_size, "url": "file",
                                        "sha256": hashlib.sha256(dmg.read_bytes()).hexdigest()}]}
    if installer.platform_key() != ("macos", "arm64"):
        pytest.skip("an Intel Mac has no build")
    report = installer.install(tmp_path / "apps", release=release, fetch=_fetch_from(dmg))
    installed = report["executable"]
    assert installed == tmp_path / "apps" / "Spirula Studio.app" / "Contents" / "MacOS" / "spirula" and installed.is_file()
    assert "usage: spirula" in subprocess.run([str(installed)], capture_output=True, text=True).stdout
    assert runner.find_spirula(str(tmp_path / "apps" / "Spirula Studio.app")) == installed


def test_latest_release_shape(monkeypatch):
    import io
    import json

    class Reply(io.BytesIO):
        headers = {}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    data = {"tag_name": "v2026.10.16", "assets": [{"name": "spirula-2026.10.16-ubuntu-vulkan-x86_64.zip", "size": 5,
                                                   "browser_download_url": "https://x/y.zip", "digest": "sha256:ABCD"}]}
    monkeypatch.setattr(installer, "_open", lambda url, timeout, accept="": Reply(json.dumps(data).encode()))
    release = installer.latest_release()
    assert release["tag"] == "v2026.10.16" and release["assets"][0]["sha256"] == "abcd"
    assert release["assets"][0]["url"] == "https://x/y.zip" and release["assets"][0]["size"] == 5


def test_command_install(tmp_path, monkeypatch, capsys):
    from ppg_spirula import SpirulaPlugin

    archive = _fake_zip(tmp_path / "spirula-1-ubuntu-vulkan-x86_64.zip", "spirula")
    release = {"tag": "v1", "assets": [{"name": archive.name, "size": 1, "url": "file", "sha256": ""}]}
    monkeypatch.setattr(installer, "latest_release", lambda *a, **k: release)
    monkeypatch.setattr(installer, "download", _fetch_from(archive))
    monkeypatch.setattr(installer.sys, "platform", "linux")
    monkeypatch.setattr(installer.platform, "machine", lambda: "x86_64")
    plugin = SpirulaPlugin()
    assert plugin.command("install", "--dir", str(tmp_path / "here")) == 0
    out = capsys.readouterr().out
    assert str(tmp_path / "here" / "spirula") in out and "v1" in out
