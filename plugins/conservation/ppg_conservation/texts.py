"""Texts of the plugin in the language of the interface, and its numbers as the interface writes them (decimal comma
in Italian: the program's ``gui.i18n.number``). The names stored in the project are the user's; everything the pane,
the dialogs and the messages show follows the language of the interface."""
from __future__ import annotations

NAME = ("Conservazione", "Conservation")


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


def num(value, decimals: int = 2) -> str:
    """``value`` as the interface writes it: ``1.234,56`` in Italian, ``1,234.56`` in English."""
    value = float(value)
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.number(value, decimals)
    except Exception:  # the program without its interface: the Italian way
        if round(value, decimals) == 0:
            value = 0.0
        return f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def plural(n: int, it: tuple, en: tuple) -> str:
    """``n`` with its noun, ``(one, many)`` in each language."""
    one, many = it if lang() == "it" else en
    return f"{n} {one if n == 1 else many}"
