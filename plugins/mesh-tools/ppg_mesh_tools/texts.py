"""Texts of the plugin in the language of the interface, and the numbers as Italian writes them.

The names of the versions of the model are stored in the project in Italian (they are data, as
the names the program gives to its own versions); everything the pane, the dialogs and the
messages show follows the language of the interface. The program's catalogue of English texts
belongs to the program, so the plugin keeps its own small Italian-English pairs, ``t(it, en)``.
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


def length_decimals(value: float) -> int:
    """Decimals that show a length with three significant digits (0.0234 -> 4, 12.3 -> 1)."""
    import math

    value = abs(float(value))
    if value == 0 or not math.isfinite(value):
        return 0
    return max(0, min(6, 2 - int(math.floor(math.log10(value)))))


def short(value: float, stored: bool = False) -> str:
    """A length with three significant digits and no trailing zeros: 0,02 - 0,0692 - 12,3 (Italian
    in a stored name, ``stored``; else in the language of the interface)."""
    text = f"{float(value):.{length_decimals(value)}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text.replace(".", ",") if stored or lang() == "it" else text


def plural(count: int, one: str, many: str) -> str:
    return one if int(count) == 1 else many


# the operations: (Italian, English) title, as the menus and the pane show them
TITLES = {
    "decimate": ("Decimazione", "Decimation"),
    "remesh": ("Rimaglia isotropa", "Isotropic remeshing"),
    "smooth": ("Levigatura", "Smoothing"),
    "close_holes": ("Chiusura dei buchi", "Closing holes"),
    "repair": ("Riparazione della topologia", "Topology repair"),
    "measure": ("Misure del modello", "Measures of the model"),
}
SMOOTHING = {"taubin": ("Taubin", "Taubin"), "hc": ("HC laplaciana", "HC Laplacian")}
# what the worker is doing (phase of its progress)
PHASES = {
    "start": ("avvio", "starting"),
    "load": ("lettura del modello", "reading the model"),
    "run": ("{0} ({1} di {2})", "{0} ({1} of {2})"),
    "measure": ("misure", "measures"),
    "save": ("scrittura del risultato", "writing the result"),
}
# the steps of the repair, as the report counts them
REPAIR_STEPS = {
    "duplicate_vertices": ("vertici doppi tolti", "duplicate vertices removed"),
    "duplicate_faces": ("facce doppie tolte", "duplicate faces removed"),
    "null_faces": ("facce senza area tolte", "faces without area removed"),
    "non_manifold_edges": ("facce tolte sugli spigoli non-manifold", "faces removed on non-manifold edges"),
    "non_manifold_edges_split": ("vertici divisi sugli spigoli non-manifold", "vertices split on non-manifold edges"),
    "non_manifold_vertices": ("vertici non-manifold divisi", "non-manifold vertices split"),
    "unreferenced_vertices": ("vertici non usati tolti", "unused vertices removed"),
}


def title(op: str) -> str:
    return t(*TITLES.get(op, (op, op)))
