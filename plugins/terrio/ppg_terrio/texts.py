"""Texts of the plugin in the language of the interface.

What is stored in the project and in the files is data: the names of the extra dimensions of the
points (``temperature``, ``ndvi``), the height systems (``ellipsoid``, ``EPSG:3855``) and the
catalogue keys (``cop-dem-glo-30``) never change with the language; the windows and the summaries
show them through these functions, in the language of the interface. The program's catalogue of
English texts belongs to the program, so the plugin keeps its own texts beside them.
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


def num(value) -> str:
    """A whole number as the program writes it: 1.234 in Italian, 1,234 in English."""
    try:
        from pyarchinit_photogrammetry.gui import i18n

        return i18n.number(value)
    except Exception:
        return f"{value:,}"


def dec(value, digits: int = 2) -> str:
    """A decimal number with ``digits`` decimals, with the decimal comma in Italian."""
    text = f"{float(value):.{digits}f}"
    return text.replace(".", ",") if lang() == "it" else text


def signed(value, digits: int = 2) -> str:
    """``+25,79`` / ``-0,41``: a difference, with its sign."""
    return ("+" if float(value) >= 0 else "-") + dec(abs(float(value)), digits)


def class_label(code: int, labels: dict | None = None) -> str:
    """The name of a class of points in the language of the program (the one the 3D view shows);
    ``labels``: the names the project gives (``chunk.meta["classes"]``)."""
    try:
        from pyarchinit_photogrammetry.gui.scene import class_name

        return class_name(int(code), labels)
    except Exception:
        from pyarchinit_photogrammetry.ai.classify import class_names

        return class_names().get(int(code), "")


def needs_newer(feature: str, found: str, wanted: str) -> str:
    """The message of a step that needs a newer terrio than the one installed."""
    return t("{0} richiede terrio {1} o successivo, ma e' installato terrio {2}: Plugin > terrio > Installa "
             "terrio... lo aggiorna (oppure pip install -U \"terrio>={1},<0.6\").",
             "{0} needs terrio {1} or later, but terrio {2} is installed: Plugin > terrio > Install terrio... "
             "updates it (or pip install -U \"terrio>={1},<0.6\").").format(feature, wanted, found)
