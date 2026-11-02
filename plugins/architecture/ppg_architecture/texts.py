"""Texts of the plugin in the language of the interface, and its numbers as the interface writes them (decimal comma
in Italian: the program's ``gui.i18n.number``). The names stored in the project are the user's; everything the pane,
the dialogs and the messages show follows the language of the interface."""
from __future__ import annotations

NAME = ("Elementi architettonici", "Architectural elements")
KINDS = {"column": ("Colonna", "Column"), "moulding": ("Modanatura", "Moulding"), "arch": ("Arco", "Arch"),
         "vault": ("Volta", "Vault"), "opening": ("Apertura", "Opening"), "door": ("Porta", "Door"),
         "window": ("Finestra", "Window")}
OPENING_TYPES = {"opening": ("solo apertura", "opening only"), "door": ("porta", "door"),
                 "window": ("finestra", "window")}
ARCH_TYPES = {"semicircular": ("a tutto sesto", "semicircular"), "segmental": ("ribassato (a segmento)", "segmental"),
              "pointed": ("a sesto acuto", "pointed"), "elliptical": ("ellittico", "elliptical"),
              "measured": ("come misurato", "as measured")}
PHASES = {"start": ("avvio", "starting"), "solid": ("costruzione del solido", "building the solid"),
          "check": ("controllo del solido", "checking the solid"), "step": ("scrittura dello STEP", "writing the STEP"),
          "stl": ("scrittura dell'STL", "writing the STL"), "done": ("fatto", "done")}


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


def metres(value, small: float = 0.1) -> str:
    """A length for the user: in metres (``1,234 m``), in millimetres under ``small`` metres (``3,2 mm``)."""
    if value is None:
        return "-"
    value = float(value)
    if abs(value) < small:
        return f"{num(value * 1000.0, 1)} mm"
    return f"{num(value, 3)} m"


def degrees(value) -> str:
    """An angle for the user: ``10,0°``."""
    return "-" if value is None else f"{num(value, 1)}°"


def plural(n: int, it: tuple, en: tuple) -> str:
    """``n`` with its noun, ``(one, many)`` in each language."""
    one, many = it if lang() == "it" else en
    return f"{n} {one if n == 1 else many}"


def kind_title(kind: str) -> str:
    return t(*KINDS.get(kind, (kind, kind)))


def arch_title(kind: str) -> str:
    return t(*ARCH_TYPES.get(kind, (kind, kind)))


def phase(name: str) -> str:
    return t(*PHASES.get(name, (name, name)))
