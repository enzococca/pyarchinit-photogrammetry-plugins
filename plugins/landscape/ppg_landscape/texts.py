"""Texts of the plugin in the language of the interface, and the numbers as Italian writes them.

The names stored in the project (versions of the DEM, runs) are in Italian, as the program's own;
everything the pane, the dialogs and the messages show follows the language of the interface. The
program's catalogue of English texts belongs to the program, so the plugin keeps its own small
Italian-English pairs, ``t(it, en)``.
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


def italian(value: float, decimals: int = 0) -> str:
    """A number as Italian writes it, whatever the interface: 200.000, 0,25 (for the stored names)."""
    if round(value, decimals) == 0:
        value = 0.0  # never "-0,00"
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")


def number(value: float, decimals: int = 0) -> str:
    """A number as the interface writes it: 1.234,5 in Italian, 1,234.5 in English."""
    if lang() == "it":
        return italian(value, decimals)
    if round(value, decimals) == 0:
        value = 0.0
    return f"{value:,.{decimals}f}"


def years(value: float) -> str:
    """A number of years without useless decimals: 1.000, 2,5."""
    value = float(value)
    return number(value, 0 if value == int(value) else (1 if value * 10 == int(value * 10) else 2))


def plural(count: int, one: str, many: str) -> str:
    return one if int(count) == 1 else many


# what the worker is doing (phase of its progress)
PHASES = {
    "start": ("avvio", "starting"),
    "load": ("lettura del DEM", "reading the DEM"),
    "grid": ("preparazione della griglia", "building the grid"),
    "run": ("anno {0} di {1}", "year {0} of {1}"),
    "area": ("area drenata", "drainage area"),
    "save": ("scrittura dei risultati", "writing the results"),
    "done": ("fatto", "done"),
}
# the scenarios of the pane (engine.PRESETS)
PRESET_TITLES = {
    "mudbrick_tell": ("Tell in mattoni crudi", "Mudbrick tell"),
    "earth_slope": ("Versante in terra", "Earth slope"),
    "exposed_excavation": ("Scavo esposto", "Exposed excavation"),
    "custom": ("Personalizzato", "Custom"),
}
NAME = ("Paesaggio nel tempo (Landlab)", "Landscape through time (Landlab)")


def phase(name: str) -> str:
    return t(*PHASES.get(name, (name, name)))


def preset_title(key: str) -> str:
    return t(*PRESET_TITLES.get(key, (key, key)))
