"""Texts of the plugin in the language of the program."""
from __future__ import annotations


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return en if i18n.current() == "en" else it
    except Exception:  # the program without its interface
        return it


def num(value) -> str:
    """A whole number as the program writes it: 1.234 in Italian, 1,234 in English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.number(value)
    except Exception:
        return f"{value:,}"
