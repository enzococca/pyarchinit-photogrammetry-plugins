"""Texts of the plugin in the language of the interface, and numbers as the interface writes them.

What is stored in the project (``chunk.meta["underwater"]``, ``camera.meta["underwater"]``) is data:
keys and values never change with the language. The program's catalogue of English texts belongs to
the program, so the plugin keeps its own Italian-English pairs, ``t(it, en)``.
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


def number(value: float, decimals: int = 0) -> str:
    """A number as the interface writes it: 1.234,5 in Italian, 1,234.5 in English."""
    value = float(value)
    if round(value, decimals) == 0:
        value = 0.0  # never "-0,00"
    text = f"{value:,.{decimals}f}"
    if lang() == "it":
        text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return text


def plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many
