"""Texts of the plugin in the language of the interface, and its numbers as the interface writes
them (decimal comma in Italian: the program's ``gui.i18n.number``)."""
from __future__ import annotations

NAME = ("Relitto", "Shipwreck")


def _italian() -> bool:
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.current() != "en"
    except Exception:  # the program without its interface
        return True


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    return it if _italian() else en


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
