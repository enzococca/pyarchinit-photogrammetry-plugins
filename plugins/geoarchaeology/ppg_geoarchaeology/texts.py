"""Texts of the plugin in the language of the interface, and its numbers as the interface writes
them (decimal comma in Italian: the program's ``gui.i18n.number``)."""
from __future__ import annotations


def _italian() -> bool:
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.current() != "en"
    except Exception:  # the program without its interface
        return True


def t(it: str, en: str) -> str:
    """The Italian text, or the English one when the interface speaks English."""
    return it if _italian() else en


def num(value, decimals: int | None = None) -> str:
    """``value`` as the interface writes it: ``1.234,56`` in Italian, ``1,234.56`` in English;
    ``decimals`` None: at most three, without the trailing zeros (``2,5`` for 2.5, ``8`` for 8.0)."""
    value = float(value)
    if decimals is None:
        text = f"{value:.3f}".rstrip("0").rstrip(".")
        if text in ("-0", ""):
            text = "0"
        return text.replace(".", ",") if _italian() else text
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.number(value, decimals)
    except Exception:  # the program without its interface: the Italian way
        if round(value, decimals) == 0:
            value = 0.0
        return f"{value:,.{decimals}f}".replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def signed(value, decimals: int) -> str:
    """:func:`num` with its sign, ``+0,07`` / ``-0,01``."""
    text = num(value, decimals)
    return text if text.startswith("-") else "+" + text
