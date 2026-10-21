"""The index entry of a plugin wheel: name from the entry point, version, description, author,
licence and homepage from its metadata, SHA-256 of the file. What the index already says of the
plugin (title, description, requires, min_app, api) is kept; a plugin of this repository gives
its texts in ``plugins/<name>/entry.json`` (title, description, i18n, requires, min_app, api),
which win over the index; the options given here win over both.

    python scripts/make_entry.py WHEEL --url URL [--write] [--title ...] [--min-app 1.1.0]

WHEEL is a file or an HTTPS URL (then ``--url`` defaults to it); ``--write`` puts the entry into
index.json (in place of the one of the same name). The workflow ``release`` runs it for every
release of a plugin of this repository and commits index.json on main.
"""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
GROUP = "pyarchinit_photogrammetry.plugins"
FIELDS = ["name", "version", "title", "description", "i18n", "author", "homepage", "license", "api", "min_app",
          "requires", "wheel", "sha256"]


def _metadata(wheel: Path) -> tuple[email.message.Message, str]:
    """The METADATA of the wheel and the name of its plugin."""
    import configparser

    with zipfile.ZipFile(wheel) as z:
        info = next(n for n in z.namelist() if n.count("/") == 1 and n.split("/")[0].endswith(".dist-info"))
        folder = info.split("/")[0]
        meta = email.message_from_bytes(z.read(f"{folder}/METADATA"))
        points = configparser.ConfigParser(delimiters=("=",))
        points.optionxform = str
        points.read_string(z.read(f"{folder}/entry_points.txt").decode())
    names = list(points[GROUP]) if points.has_section(GROUP) else []
    if len(names) != 1:
        sys.exit(f"{wheel.name}: {len(names)} plugins in the group {GROUP}, one expected")
    return meta, names[0]


def texts_of(name: str) -> dict:
    """What ``plugins/<name>/entry.json`` says of a plugin kept in this repository: the title,
    the description, ``i18n``, ``requires``, ``min_app`` and ``api`` the wheel cannot carry."""
    path = ROOT / "plugins" / name / "entry.json"
    if not path.exists():
        return {}
    texts = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in texts.items() if k in ("title", "description", "i18n", "requires", "min_app", "api")}


def entry_for(wheel: Path, url: str, old: dict | None = None, **given) -> dict:
    meta, name = _metadata(wheel)
    old = dict(old or {})
    author = meta.get("Author") or (meta.get("Author-email") or "").split("<")[0].strip().strip('"')
    homepage = meta.get("Home-page") or ""
    for value in meta.get_all("Project-URL") or []:
        label, _, link = value.partition(",")
        if label.strip().lower() in ("homepage", "home", "source", "repository") and not homepage:
            homepage = link.strip()
    entry = {
        "name": name, "version": meta["Version"],
        "title": old.get("title") or name.replace("-", " ").capitalize(),
        "description": old.get("description") or meta.get("Summary") or name, "i18n": old.get("i18n") or {},
        "author": old.get("author") or author, "homepage": homepage or old.get("homepage", ""),
        "license": meta.get("License-Expression") or meta.get("License") or old.get("license", ""),
        "api": old.get("api", 1), "min_app": old.get("min_app", "1.1.0"), "requires": old.get("requires", []),
        "wheel": url, "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}
    entry.update(texts_of(name))
    entry.update({k: v for k, v in given.items() if v is not None})
    for key in ("homepage", "i18n"):
        if not entry[key]:
            del entry[key]
    return {k: entry[k] for k in FIELDS if k in entry}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("wheel", help="the wheel: a file or an https:// URL")
    parser.add_argument("--url", help="where the program downloads it (the URL of WHEEL by default)")
    parser.add_argument("--index", default=str(ROOT / "index.json"))
    parser.add_argument("--write", action="store_true", help="put the entry into the index")
    parser.add_argument("--title")
    parser.add_argument("--description")
    parser.add_argument("--min-app", dest="min_app")
    parser.add_argument("--api", type=int)
    parser.add_argument("--requires", help="modules beyond those of the program, separated by commas")
    a = parser.parse_args(argv)
    index_path = Path(a.index)
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"format": 1, "plugins": []}
    with tempfile.TemporaryDirectory() as tmp:
        if a.wheel.startswith("https://"):
            wheel = Path(tmp) / PurePosixPath(a.wheel).name
            request = urllib.request.Request(a.wheel, headers={"User-Agent": "pyarchinit-photogrammetry-plugins"})
            with urllib.request.urlopen(request, timeout=120) as r:
                wheel.write_bytes(r.read())
            url = a.url or a.wheel
        else:
            wheel = Path(a.wheel)
            if not a.url:
                parser.error("--url is needed for a local wheel: where the program will download it")
            url = a.url
        _meta, name = _metadata(wheel)
        old = next((e for e in index["plugins"] if e.get("name") == name), None)
        requires = None if a.requires is None else [m.strip() for m in a.requires.split(",") if m.strip()]
        entry = entry_for(wheel, url, old, title=a.title, description=a.description, min_app=a.min_app, api=a.api,
                          requires=requires)
    print(json.dumps(entry, indent=2, ensure_ascii=False))
    if a.write:
        plugins = [e for e in index["plugins"] if e.get("name") != name] + [entry]
        index["plugins"] = sorted(plugins, key=lambda e: e["name"])
        index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"written into {index_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
