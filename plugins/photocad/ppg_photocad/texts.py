"""Texts of the plugin in the language of the interface: ``t(it, en)``.

The program's catalogue of English texts belongs to the program, so the plugin keeps its own pairs.
"""
from __future__ import annotations


def lang() -> str:
    """``en`` when the interface speaks English, else ``it`` (also without the interface)."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return "en" if i18n.current() == "en" else "it"
    except Exception:  # noqa: BLE001 - the program without its interface
        return "it"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    return en if lang() == "en" else it


LAYER_PREFIX = "PhotoCAD: "  # the layers of the 2D view: "PhotoCAD: <name of the conversion>"


def layer_name(record: dict) -> str:
    return LAYER_PREFIX + str(record.get("name") or record.get("id"))
