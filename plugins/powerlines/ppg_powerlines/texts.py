"""Texts of the plugin in the language of the interface."""
from __future__ import annotations


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:  # the program without its interface
        return it
