"""Texts of the plugin in the language of the interface.

The program's catalogue of English texts belongs to the program, so the plugin keeps its own small
Italian-English pairs, ``t(it, en)``, as the other plugins of the repository do.
"""
from __future__ import annotations


def lang() -> str:
    """``en`` when the interface speaks English, else ``it`` (also without the interface)."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return "en" if i18n.current() == "en" else "it"
    except Exception:  # the program without its interface
        return "it"


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    return en if lang() == "en" else it


def plural(count: int, one: str, many: str) -> str:
    return one if int(count) == 1 else many
