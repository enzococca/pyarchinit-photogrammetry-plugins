"""Checks of index.json, run on every pull request.

* always: the format (schema/index.schema.json, with jsonschema when installed), names unique,
  versions readable;
* ``--download``: each wheel is downloaded and checked - its SHA-256, a pure-Python wheel of the
  version given, declaring the plugin by its name in the entry point group, nothing outside its
  folder;
* ``--load`` (with pyarchinit-photogrammetry installed): each plugin is unpacked into a plugin
  folder of its own and started the way the program starts it; what it adds is listed.

    python scripts/check_index.py [--download] [--load] [--only NAME ...] [--index index.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
GROUP = "pyarchinit_photogrammetry.plugins"
WHEEL = re.compile(r"^(?P<dist>[A-Za-z0-9_.]+)-(?P<version>[^-]+)-py3-none-any\.whl$")


def _schema_problems(index: dict) -> list[str]:
    schema = json.loads((ROOT / "schema" / "index.schema.json").read_text(encoding="utf-8"))
    try:
        import jsonschema
    except ImportError:
        print("note: jsonschema not installed, only the basic checks of the format")
        problems = []
        if not isinstance(index.get("plugins"), list):
            return ["'plugins' must be a list"]
        required = schema["$defs"]["plugin"]["required"]
        for i, entry in enumerate(index["plugins"]):
            missing = [k for k in required if k not in entry]
            if missing:
                problems.append(f"plugins[{i}]: missing {', '.join(missing)}")
        return problems
    validator = jsonschema.Draft202012Validator(schema)
    return [f"{'/'.join(str(p) for p in e.absolute_path) or '(index)'}: {e.message}"
            for e in sorted(validator.iter_errors(index), key=lambda e: list(e.absolute_path))]


def _download(url: str, target: Path) -> str:
    """Download ``url`` into ``target``; its SHA-256."""
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "pyarchinit-photogrammetry-plugins checks"})
    with urllib.request.urlopen(request, timeout=120) as r, open(target, "wb") as out:
        while chunk := r.read(1 << 16):
            out.write(chunk)
            digest.update(chunk)
    return digest.hexdigest()


def check_wheel(entry: dict, work: Path) -> tuple[list[str], Path | None]:
    """Problems of the wheel of ``entry``, and the folder it was unpacked into."""
    name = PurePosixPath(entry["wheel"]).name
    m = WHEEL.match(name)
    if not m:
        return [f"{name}: not a pure-Python wheel (-py3-none-any.whl)"], None
    problems = []
    if m.group("version") != entry["version"]:
        problems.append(f"{name}: the file name says version {m.group('version')}, the index {entry['version']}")
    wheel = work / name
    try:
        sha = _download(entry["wheel"], wheel)
    except OSError as exc:
        return [f"{entry['wheel']}: could not be downloaded ({getattr(exc, 'reason', exc)})"], None
    if sha != entry["sha256"]:
        problems.append(f"{name}: SHA-256 {sha}, the index says {entry['sha256']}")
    unpacked = work / "unpacked" / entry["name"]
    try:
        with zipfile.ZipFile(wheel) as z:
            unsafe = [n for n in z.namelist() if PurePosixPath(n).is_absolute() or ".." in PurePosixPath(n).parts]
            if unsafe:
                return problems + [f"{name}: paths outside its folder: {unsafe[:3]}"], None
            z.extractall(unpacked)
    except zipfile.BadZipFile:
        return problems + [f"{name}: not a valid archive"], None
    import importlib.metadata

    dists = list(importlib.metadata.distributions(path=[str(unpacked)]))
    if len(dists) != 1:
        return problems + [f"{name}: {len(dists)} distributions inside, one expected"], None
    dist = dists[0]
    if dist.version != entry["version"]:
        problems.append(f"{name}: its metadata say version {dist.version}, the index {entry['version']}")
    plugins = [ep.name for ep in dist.entry_points.select(group=GROUP)]
    if entry["name"] not in plugins:
        problems.append(f"{name}: no entry point {entry['name']!r} in the group {GROUP} (it has {plugins})")
    return problems, unpacked


def load_plugin(entry: dict, unpacked: Path) -> list[str]:
    """Start the plugin as the program does, from a plugin folder of its own."""
    os.environ["PPG_PLUGINS_ENVIRONMENT"] = "0"  # only the plugin under test
    from pyarchinit_photogrammetry.plugins import index as plugin_index
    from pyarchinit_photogrammetry.plugins import Manager

    problem = plugin_index.Entry.from_dict(entry).problem(app_version=entry["min_app"])
    if problem:
        return [f"{entry['name']}: {problem}"]
    manager = Manager(unpacked.parent)
    manager.load()
    info = manager.get(entry["name"])
    if info is None:
        return [f"{entry['name']}: not found by the program"]
    if not info.active:
        return [f"{entry['name']}: did not start: {info.error}"]
    actions = [a.title for a in manager.actions if a.plugin == entry["name"]]
    commands = sorted(c for c, command in manager.commands.items() if command.plugin == entry["name"])
    exporters = [e.title for e in manager.exporters if e.plugin == entry["name"]]
    print(f"  started: actions {actions}, commands {commands}, export formats {exporters}")
    manager.unload()
    return []


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--index", default=str(ROOT / "index.json"))
    parser.add_argument("--download", action="store_true", help="download and check the wheels")
    parser.add_argument("--load", action="store_true", help="start each plugin with the program (implies --download)")
    parser.add_argument("--only", nargs="*", default=None, help="the plugins to check (all by default)")
    a = parser.parse_args(argv)
    index = json.loads(Path(a.index).read_text(encoding="utf-8"))
    problems = _schema_problems(index)
    entries = index.get("plugins", []) if isinstance(index.get("plugins"), list) else []
    names = [e.get("name") for e in entries]
    problems += [f"{n}: listed {names.count(n)} times" for n in sorted(set(names)) if names.count(n) > 1]
    if names != sorted(names):
        problems.append("the plugins are not in the order of their names")
    if not problems and (a.download or a.load):
        with tempfile.TemporaryDirectory(prefix="ppg-index-") as tmp:
            for entry in entries:
                if a.only and entry["name"] not in a.only:
                    continue
                print(f"{entry['name']} {entry['version']}")
                work = Path(tmp) / entry["name"]
                work.mkdir()
                found, unpacked = check_wheel(entry, work)
                problems += found
                if a.load and unpacked is not None and not found:
                    problems += load_plugin(entry, unpacked)
    for p in problems:
        print(f"::error::{p}" if os.environ.get("GITHUB_ACTIONS") else f"error: {p}")
    print(f"{len(entries)} plugins, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
