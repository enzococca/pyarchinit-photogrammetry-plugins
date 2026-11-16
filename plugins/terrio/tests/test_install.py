"""terrio in the plugin folder of the application: the first download, the update of an older terrio (at once
when it is not in use, beside it until the next start when it is), what a failure leaves behind and what the
window says. The wheels are small stand-ins written here - a package ``terrio`` of two modules with its
``.dist-info`` - served by a stub in place of the download: nothing reaches the network, and for these tests
the only terrio is the one of the plugin folder."""
from __future__ import annotations

import hashlib
import importlib
import io
import json
import logging
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

import pytest

from ppg_terrio import MIB, environment

LOGGER = "pyarchinit_photogrammetry.plugins.terrio"  # where the plugin writes its log lines


def _wheel(version: str, members: dict | None = None) -> tuple[str, bytes]:
    """A stand-in for the wheel of terrio ``version`` for this computer (its name, its bytes); ``_native``
    stands for the compiled module, ``members`` are more files of it."""
    data = io.BytesIO()
    with zipfile.ZipFile(data, "w") as z:
        z.writestr("terrio/__init__.py", f'__version__ = "{version}"\nfrom . import _native  # noqa: F401\n')
        z.writestr("terrio/_native.py", f'BUILD = "{version}"\n')
        z.writestr(f"terrio-{version}.dist-info/METADATA", f"Metadata-Version: 2.1\nName: terrio\nVersion: {version}\n")
        for name, text in (members or {}).items():
            z.writestr(name, text)
    return f"terrio-{version}-cp310-abi3-{environment.platform_tags()[0]}.whl", data.getvalue()


class _Served:
    """What the download gives: the bytes of a file; ``cut``: the connection drops half way."""

    def __init__(self, data: bytes, cut: bool = False):
        self.data, self.cut, self.reads = io.BytesIO(data), cut, 0
        self.headers = {"Content-Length": str(len(data))}

    def read(self, n: int = -1) -> bytes:
        self.reads += 1
        if self.cut:
            if self.reads > 1:
                raise ConnectionResetError(54, "Connection reset by peer")
            return self.data.read(len(self.data.getvalue()) // 2)
        return self.data.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _PyPI:
    """PyPI for these tests, out of memory: the listing of each version given to :meth:`add` and its wheel;
    ``asked`` keeps the addresses read."""

    def __init__(self):
        self.listings, self.files, self.asked = {}, {}, []

    def add(self, version: str, digest: str = "right", members: dict | None = None, cut: bool = False) -> None:
        name, data = _wheel(version, members)
        entry = {"filename": name, "url": f"https://files.invalid/{name}", "size": len(data)}
        if digest != "none":
            entry["digests"] = {"sha256": hashlib.sha256(data).hexdigest() if digest == "right" else "0" * 64}
        self.listings[environment.PYPI.format(version=version)] = [
            {"filename": f"terrio-{version}.tar.gz", "url": "https://files.invalid/source"}, entry]
        self.files[entry["url"]] = (data, cut)

    def __call__(self, url: str, timeout: float):
        self.asked.append(url)
        if url in self.listings:
            return _Served(json.dumps({"urls": self.listings[url]}).encode())
        return _Served(*self.files[url])


class _OnlyTheFolder:
    """The import of terrio finds nothing while the plugin folder holds none (the tests have a terrio of their
    own in the environment, an application without pip has not)."""

    def __init__(self, lib: Path):
        self.lib = lib

    def find_spec(self, name, path=None, target=None):
        if name == "terrio" and not (self.lib / "terrio").is_dir():
            raise ModuleNotFoundError("No module named 'terrio'", name="terrio")
        return None


def _loaded() -> dict:
    return {name: module for name, module in sys.modules.items() if name == "terrio" or name.startswith("terrio.")}


def _no_network(*a, **k):
    raise AssertionError(f"a test of the installation reached for the network: {a[:1]}")


@pytest.fixture
def folder(tmp_path, monkeypatch):
    """The plugin folder of an application that carries no terrio, in a process that has loaded none; what the
    test loads and puts on the path goes away with it, and the terrio of the other tests comes back."""
    data = tmp_path / "_data"
    data.mkdir()
    loaded, path, guard = _loaded(), list(sys.path), _OnlyTheFolder(data / "lib")
    for name in loaded:
        del sys.modules[name]
    sys.meta_path.insert(0, guard)
    monkeypatch.setattr(environment, "_DATA_DIR", data)
    monkeypatch.setattr(environment, "_LIB_DIR", data / "lib")
    monkeypatch.setattr(environment, "_environment_version", lambda folder=None: "")
    monkeypatch.setattr(environment, "_SAID", "", raising=False)
    monkeypatch.setattr(urllib.request, "urlopen", _no_network)
    try:
        yield data
    finally:
        sys.meta_path.remove(guard)
        for name in _loaded():
            del sys.modules[name]
        sys.modules.update(loaded)
        sys.path[:] = path
        importlib.invalidate_caches()


@pytest.fixture
def pypi(monkeypatch):
    server = _PyPI()
    monkeypatch.setattr(environment, "_get", server)
    return server


def _restart(folder: Path) -> None:
    """The program closed and started again, as far as terrio goes: none of its modules loaded, the plugin
    folder not on the path yet."""
    for name in _loaded():
        del sys.modules[name]
    while str(folder / "lib") in sys.path:
        sys.path.remove(str(folder / "lib"))
    importlib.invalidate_caches()


def _files(root: Path) -> dict:
    """Every file under ``root`` with its bytes (the bytecode Python writes beside the modules left out)."""
    return {p.relative_to(root).as_posix(): p.read_bytes() for p in sorted(root.rglob("*"))
            if p.is_file() and "__pycache__" not in p.parts}


def _beside(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir())


def _stuck(name: str, rename=Path.rename):
    """A ``Path.rename`` that cannot move the folder called ``name``: another program has it in use (what
    Windows answers for the folder of a compiled module that is loaded, or of files a scanner has open)."""
    def refuse(self, target):
        if self.name == name:
            raise PermissionError(13, "in use by another program", str(self))
        return rename(self, target)

    return refuse


def _warnings(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == logging.WARNING]


def _in_use(folder: Path, pypi, version: str = "0.6.1"):
    """terrio ``version`` in the plugin folder, loaded: the module."""
    pypi.add(version)
    environment.install(version=version)
    module = environment.load()
    assert module.__version__ == version and Path(module.__file__).parent == folder / "lib" / "terrio"
    return module


def test_the_first_terrio_goes_into_the_plugin_folder_and_is_used_at_once(folder, pypi):
    pypi.add("0.9.0")
    said = []
    with pytest.raises(environment.TerrioMissing, match="terrio is not installed"):
        environment.load()
    out = environment.install(lambda fraction, text: said.append((fraction, text)))
    assert out == folder / "lib" == environment.lib_dir() and environment.folder_version(out) == "0.9.0"
    module = environment.load()
    assert module.__version__ == "0.9.0" and Path(module.__file__).parent == out / "terrio"
    assert _beside(folder) == ["lib"], "nothing else is left in the plugin folder"
    assert said[0][0] == 0.0 and said[-1][0] == 1.0 and str(out) in said[-1][1]


def test_an_update_of_the_terrio_in_use_waits_beside_it_for_the_next_start(folder, pypi):
    """A compiled module stays in the process that loaded it (and on Windows its file cannot be removed): the
    new terrio is made complete beside the one in use and takes its place at the next start, before the import."""
    lib = folder / "lib"
    old = _in_use(folder, pypi)
    before = _files(lib)
    pypi.add("0.9.0")
    out = environment.install()
    assert _files(lib) == before, "the terrio in use is where it was, whole"
    assert environment.load() is old and old._native.BUILD == "0.6.1", "and it is still the one the plugin uses"
    assert out == folder / "lib.new" == environment.pending_dir() and environment.pending_version() == "0.9.0"
    assert set(_files(out)) == {"terrio/__init__.py", "terrio/_native.py", "terrio-0.9.0.dist-info/METADATA"}
    assert _beside(folder) == ["lib", "lib.new"]
    assert environment.load() is old, "nothing changes however many times terrio is asked for"
    _restart(folder)
    new = environment.load()
    assert new.__version__ == "0.9.0" and Path(new.__file__).parent == lib / "terrio"
    assert new._native.BUILD == "0.9.0", "every module is of the new terrio"
    assert _beside(folder) == ["lib"] and environment.pending_version() == ""
    assert environment.folder_version(lib) == "0.9.0" and "terrio-0.6.1.dist-info/METADATA" not in _files(lib)


def test_an_update_of_a_terrio_that_is_not_in_use_takes_its_place_at_once(folder, pypi):
    lib = folder / "lib"
    _in_use(folder, pypi)
    _restart(folder)  # the terrio of the folder is there, this process has not loaded it
    pypi.add("0.9.0")
    assert environment.install() == lib
    assert environment.load().__version__ == "0.9.0" and _beside(folder) == ["lib"]
    assert set(_files(lib)) == {"terrio/__init__.py", "terrio/_native.py", "terrio-0.9.0.dist-info/METADATA"}


def test_a_second_download_takes_the_place_of_the_one_that_waits(folder, pypi):
    old = _in_use(folder, pypi)
    pypi.add("0.8.0")
    environment.install(version="0.8.0")
    assert environment.pending_version() == "0.8.0"
    pypi.add("0.9.0")
    assert environment.install() == folder / "lib.new"
    assert environment.pending_version() == "0.9.0" and environment.load() is old
    assert set(_files(folder / "lib.new")) == {"terrio/__init__.py", "terrio/_native.py",
                                               "terrio-0.9.0.dist-info/METADATA"}, "none of the files of 0.8.0"
    assert _beside(folder) == ["lib", "lib.new"]


@pytest.mark.parametrize("stuck", ["lib", "lib.new"])
def test_a_swap_that_cannot_be_done_leaves_the_terrio_that_works(folder, pypi, monkeypatch, caplog, stuck):
    """At the next start a folder cannot be moved (another window of the program has it in use, on Windows):
    the terrio of before is whole in its place and is the one used, the new one still waits, the log says why."""
    lib = folder / "lib"
    _in_use(folder, pypi)
    pypi.add("0.9.0")
    environment.install()
    before, waiting, rename = _files(lib), _files(folder / "lib.new"), Path.rename
    _restart(folder)
    monkeypatch.setattr(Path, "rename", _stuck(stuck))
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        module = environment.load()
    assert module.__version__ == "0.6.1" and Path(module.__file__).parent == lib / "terrio"
    assert _files(lib) == before and _files(folder / "lib.new") == waiting
    assert _beside(folder) == ["lib", "lib.new"] and environment.pending_version() == "0.9.0"
    lines = _warnings(caplog)
    assert len(lines) == 1 and "0.9.0" in lines[0] and "in use by another program" in lines[0], lines
    environment.load()
    assert len([r for r in caplog.records if r.name == LOGGER]) == 1, "said once, not at every step"
    monkeypatch.setattr(Path, "rename", rename)
    _restart(folder)
    assert environment.load().__version__ == "0.9.0" and _beside(folder) == ["lib"], "done at the start after"


@pytest.mark.parametrize("left", ["between the moves", "before the removal", "an older leftover", "only the one aside"])
def test_what_a_stopped_swap_left_is_put_right_at_the_next_start(folder, pypi, left):
    """The program stopped during a swap (or the computer was switched off): whatever folders it left, the next
    start ends with one whole terrio in ``lib`` - the new one when it is there - and nothing beside it."""
    lib, aside, waiting = folder / "lib", folder / "lib.old", folder / "lib.new"
    _in_use(folder, pypi)
    pypi.add("0.9.0")
    environment.install()
    wanted = "0.9.0"
    if left == "between the moves":  # the terrio of before set aside, the new one not moved in
        lib.rename(aside)
    elif left == "before the removal":  # the new one in place, the one of before not removed
        lib.rename(aside)
        waiting.rename(lib)
    elif left == "an older leftover":  # what an earlier swap could not remove, and a new terrio that waits
        shutil.copytree(lib, aside)
    else:  # the terrio of before set aside and the new one gone (removed by hand)
        lib.rename(aside)
        shutil.rmtree(waiting)
        wanted = "0.6.1"
    _restart(folder)
    module = environment.load()
    assert module.__version__ == wanted == module._native.BUILD and Path(module.__file__).parent == lib / "terrio"
    assert _beside(folder) == ["lib"] and environment.folder_version(lib) == wanted


def test_an_update_that_cannot_be_put_in_place_says_to_close_the_program_and_waits(folder, pypi, monkeypatch):
    """This process has not loaded the terrio of the folder, yet the folder cannot be moved (another window of the
    program has it loaded, on Windows): the download is kept beside it and the error says what to do."""
    lib = folder / "lib"
    _in_use(folder, pypi)
    _restart(folder)
    before, rename = _files(lib), Path.rename
    pypi.add("0.9.0")
    monkeypatch.setattr(Path, "rename", _stuck("lib"))
    with pytest.raises(environment.TerrioMissing, match="chiudi il programma") as error:
        environment.install()
    assert "terrio 0.9.0" in str(error.value) and "in use by another program" in str(error.value)
    assert _files(lib) == before and _beside(folder) == ["lib", "lib.new"]
    assert environment.pending_version() == "0.9.0" and environment.load().__version__ == "0.6.1"
    monkeypatch.setattr(Path, "rename", rename)
    _restart(folder)
    assert environment.load().__version__ == "0.9.0" and _beside(folder) == ["lib"]


@pytest.mark.parametrize("failure", ["digest", "no digest", "path", "cut", "empty"])
def test_a_download_that_fails_leaves_the_terrio_that_works_where_it_was(folder, pypi, failure):
    lib = folder / "lib"
    old = _in_use(folder, pypi)
    before = _files(lib)
    pypi.add("0.9.0", digest={"digest": "wrong", "no digest": "none"}.get(failure, "right"), cut=failure == "cut",
             members={"../outside.py": "x = 1\n"} if failure == "path" else None)
    if failure == "empty":  # a wheel of something else under the name of terrio
        name, _ = _wheel("0.9.0")
        data = io.BytesIO()
        with zipfile.ZipFile(data, "w") as z:
            z.writestr("other/__init__.py", "")
        pypi.files[f"https://files.invalid/{name}"] = (data.getvalue(), False)
        pypi.listings[environment.PYPI.format(version="0.9.0")][1]["digests"] = {
            "sha256": hashlib.sha256(data.getvalue()).hexdigest()}
    wanted = {"digest": (environment.TerrioMissing, "does not match its SHA-256"),
              "no digest": (environment.TerrioMissing, "no SHA-256"), "path": (environment.TerrioMissing, "unsafe path"),
              "cut": (ConnectionResetError, "reset"), "empty": (environment.TerrioMissing, "no terrio")}[failure]
    with pytest.raises(wanted[0], match=wanted[1]):
        environment.install()
    assert _files(lib) == before and environment.load() is old
    assert _beside(folder) == ["lib"], "nothing waits and nothing of the download is left"


def _window(folder: Path, monkeypatch, answer=0):
    """The plugin in a window that runs its tasks at once: the plugin, what it asked and what it said."""
    from ppg_terrio import TerrioPlugin
    from pyarchinit_photogrammetry.gui import dialogs
    from pyarchinit_photogrammetry.plugins import Context

    asked, said = [], []

    class Window:
        def run_task(self, title, job, done=None, refresh=True, save=True):
            result = job(lambda *a: None)
            if done is not None:
                done(result)

    monkeypatch.setattr(dialogs, "message", lambda parent, kind, title, text, **k: said.append(text))
    monkeypatch.setattr(dialogs, "choice",
                        lambda parent, kind, title, text, options, **k: asked.append((text, list(options))) or answer)
    plugin = TerrioPlugin()
    plugin.activate(Context("terrio", folder, window=Window()))
    return plugin, asked, said


def test_the_window_offers_to_update_an_older_terrio_and_says_to_reopen_the_program(folder, pypi, monkeypatch):
    _in_use(folder, pypi)
    pypi.add("0.9.0")
    plugin, asked, said = _window(folder, monkeypatch)
    plugin.install_dialog()
    assert len(asked) == 1, f"the update is offered: {said}"
    text, options = asked[0]
    assert "terrio 0.6.1" in text and "terrio 0.9.0" in text and f"circa {MIB} MB" in text and options == ["Aggiorna"]
    assert environment.pending_version() == "0.9.0" and environment.load().__version__ == "0.6.1"
    assert len(said) == 1 and "chiudi e riapri il programma" in said[0], said
    assert "terrio 0.9.0" in said[0] and "resta in uso terrio 0.6.1" in said[0]
    downloads = len(pypi.asked)
    plugin.install_dialog()
    assert len(asked) == 1 and len(pypi.asked) == downloads, "the terrio that waits is not downloaded again"
    assert len(said) == 2 and "chiudi e riapri il programma" in said[1] and "terrio 0.9.0" in said[1]
    _restart(folder)
    plugin.install_dialog()
    assert said[2] == "terrio 0.9.0 e' gia' disponibile." and len(asked) == 1 and len(pypi.asked) == downloads


def test_an_update_that_is_not_wanted_downloads_nothing(folder, pypi, monkeypatch):
    _in_use(folder, pypi)
    pypi.add("0.9.0")
    downloads = len(pypi.asked)
    plugin, asked, said = _window(folder, monkeypatch, answer=None)
    plugin.install_dialog()
    assert len(asked) == 1 and not said and len(pypi.asked) == downloads
    assert _beside(folder) == ["lib"] and environment.pending_version() == ""


def test_a_step_does_not_offer_again_the_terrio_that_waits_for_the_next_start(folder, pypi, monkeypatch):
    """A step that needs a newer terrio than the one in use offers the download once; asked again before the
    program is closed, it says that the new terrio waits."""
    _in_use(folder, pypi, "0.4.2")
    pypi.add("0.9.0")
    plugin, asked, said = _window(folder, monkeypatch)
    assert plugin._ready(environment.FEATURES_SINCE, "La pulizia") is False
    assert len(asked) == 1 and "La pulizia richiede terrio 0.5" in asked[0][0] and asked[0][1] == ["Scarica"]
    assert environment.pending_version() == "0.9.0" and "chiudi e riapri il programma" in said[-1]
    downloads = len(pypi.asked)
    assert plugin._ready(environment.FEATURES_SINCE, "La pulizia") is False
    assert len(asked) == 1 and len(pypi.asked) == downloads, "not offered and not downloaded a second time"
    assert len(said) == 2 and "terrio 0.9.0" in said[1] and "chiudi e riapri il programma" in said[1]
    assert said[1].startswith("La pulizia richiede terrio 0.5 o successivo.") and "resta in uso terrio 0.4.2" in said[1]
    _restart(folder)
    assert plugin._ready(environment.FEATURES_SINCE, "La pulizia") is True and len(asked) == 1 and len(said) == 2


def test_the_first_download_from_the_window_is_ready_at_once(folder, pypi, monkeypatch):
    pypi.add("0.9.0")
    plugin, asked, said = _window(folder, monkeypatch)
    assert plugin._ready() is False
    assert len(asked) == 1 and "terrio non e' installato" in asked[0][0] and asked[0][1] == ["Scarica"]
    assert len(said) == 1 and "terrio 0.9.0 installato in" in said[0] and "Avvia di nuovo il passo" in said[0]
    assert plugin._ready() is True and _beside(folder) == ["lib"]


def test_a_first_terrio_that_cannot_be_put_in_place_is_not_downloaded_twice(folder, pypi, monkeypatch, caplog):
    """The first download is whole but its folder cannot be moved into place (a scanner has its files open, on
    Windows): the window does not offer the download again, the log says why once, and the step after - the
    folder free again - the terrio is put in place and used."""
    rename = Path.rename
    pypi.add("0.9.0")
    plugin, asked, said = _window(folder, monkeypatch)
    monkeypatch.setattr(Path, "rename", _stuck("lib.new"))
    with pytest.raises(environment.TerrioMissing, match="chiudi il programma"):  # the window shows it as an error
        plugin._ready()
    downloads = len(pypi.asked)
    assert len(asked) == 1 and not said and _beside(folder) == ["lib.new"]
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert plugin._ready() is False
    assert len(asked) == 1 and len(pypi.asked) == downloads, "not offered and not downloaded a second time"
    assert said == ["terrio 0.9.0 e' gia' scaricato nella cartella del plugin: chiudi e riapri il programma per "
                    "usarlo."], "and nothing of a terrio in use: there is none"
    lines = _warnings(caplog)
    assert len(lines) == 1 and "0.9.0" in lines[0] and "in use by another program" in lines[0], lines
    monkeypatch.setattr(Path, "rename", rename)
    assert plugin._ready() is True and _beside(folder) == ["lib"] and len(asked) == 1 and len(said) == 1
    assert environment.load().__version__ == "0.9.0"


@pytest.mark.parametrize("found", ["0.10.0", "0.3.0"])
def test_a_terrio_outside_the_accepted_range_is_named_with_the_range(folder, pypi, monkeypatch, found):
    """A terrio is there but the plugin does not accept its version: the window says which version it found and
    which it accepts (not that terrio is missing), and offers the download that can replace it."""
    pypi.add(found)
    environment.install(version=found)  # in the plugin folder: what the plugin downloads takes its place
    downloads = len(pypi.asked)
    plugin, asked, said = _window(folder, monkeypatch, answer=None)
    assert plugin._ready() is False
    assert len(asked) == 1 and not said and len(pypi.asked) == downloads
    text, options = asked[0]
    assert "non e' installato" not in text
    assert f"terrio {found}" in text and "da 0.3.2" in text and "0.10 escluso" in text, text
    assert "terrio 0.9.0" in text and options == ["Scarica"]


def test_a_newer_terrio_of_the_environment_is_named_and_pip_is_said(folder, pypi, monkeypatch):
    """The terrio of the Python environment is newer than the plugin accepts: a terrio downloaded into the plugin
    folder would not be used (the newer one wins), so nothing is offered: the window says what pip is asked for."""
    monkeypatch.setattr(environment, "available", lambda minimum=None: "")
    monkeypatch.setattr(environment, "installed", lambda: "0.10.1")
    monkeypatch.setattr(environment, "_environment_version", lambda folder=None: "0.10.1")
    plugin, asked, said = _window(folder, monkeypatch)
    assert plugin._ready() is False
    assert not asked and len(pypi.asked) == 0 and len(said) == 1
    assert "terrio 0.10.1" in said[0] and "da 0.3.2" in said[0] and "0.10 escluso" in said[0], said
    assert "pip install" in said[0] and environment.requirement() in said[0]
