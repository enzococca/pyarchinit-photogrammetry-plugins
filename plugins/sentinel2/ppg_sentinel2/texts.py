"""Texts of the plugin in the language of the interface, and the names of what it makes.

What is stored in the project (``chunk.meta["sentinel2"]``) and in the files is data: index
names, period keys (``2023-summer``), acquisition ids (``20230614_S2B``) never change with the
language; the pane, the chart and the QGIS project show them through :func:`period_name`,
:func:`index_name` and :func:`product_name`, in the language of the interface. The program's
catalogue of English texts belongs to the program, so the plugin keeps its own small table.
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


def plural(n, one: str, many: str) -> str:
    """``one`` or ``many`` (already in the language of the interface, see :func:`t`) for ``n``
    things, ``{0}`` the number: ``plural(1, "{0} data", "{0} date")`` is "1 data"."""
    text = one if abs(float(n)) == 1 else many
    return text.format(n) if "{0}" in text else text


def clear_dates_left(n: int) -> str:
    """``1 data serena resta``, ``3 date serene restano`` (after the cache was emptied)."""
    return plural(n, t("{0} data serena resta", "{0} clear date left"), t("{0} date serene restano", "{0} clear dates left"))


INDEX_NAMES = {
    "NDVI": ("NDVI (vigore della vegetazione)", "NDVI (vegetation vigour)"),
    "NDRE": ("NDRE (clorofilla, bordo del rosso)", "NDRE (chlorophyll, red edge)"),
    "NDMI": ("NDMI (acqua nella vegetazione)", "NDMI (water in the vegetation)"),
    "RGB": ("Colori naturali (B04, B03, B02)", "Natural colour (B04, B03, B02)"),
}
SEASONS = {"spring": ("primavera", "spring"), "summer": ("estate", "summer"), "autumn": ("autunno", "autumn"),
           "winter": ("inverno", "winter")}
MONTHS = (("gennaio", "January"), ("febbraio", "February"), ("marzo", "March"), ("aprile", "April"),
          ("maggio", "May"), ("giugno", "June"), ("luglio", "July"), ("agosto", "August"),
          ("settembre", "September"), ("ottobre", "October"), ("novembre", "November"), ("dicembre", "December"))
STATUS = {"found": ("da leggere", "to read"), "cached": ("letta", "read"), "cloudy": ("nuvolosa", "cloudy"),
          "failed": ("errore", "error")}
KINDS = {"composite": ("Composito", "Composite"), "cropmarks": ("Crop mark", "Crop marks")}
LAYER = ("Sentinel-2: area", "Sentinel-2: area")  # the layer of the raster view with the area


def index_name(name: str, short: bool = False) -> str:
    """``NDVI (vigore della vegetazione)``; ``short``: just ``NDVI`` (``Colori naturali`` for RGB)."""
    if short and name != "RGB":
        return name
    pair = INDEX_NAMES.get(name)
    if pair is None:
        return name
    text = t(*pair)
    return text.split(" (")[0] if short else text


def period_name(period: str, language: str | None = None) -> str:
    """A period key in words: ``2023-summer`` -> ``estate 2023``, ``summer`` -> ``estate (ogni
    anno)``, ``2023-06`` -> ``giugno 2023``, ``all`` -> ``tutto il periodo``; dates, ranges and
    acquisition ids as they are."""
    en = (language or lang()) == "en"
    p = str(period)
    if p == "all":
        return "whole period" if en else "tutto il periodo"
    if p in SEASONS:
        return f"{SEASONS[p][1]} (every year)" if en else f"{SEASONS[p][0]} (ogni anno)"
    head, _, tail = p.partition("-")
    if tail in SEASONS and head.isdigit():
        return f"{SEASONS[tail][1 if en else 0]} {head}"
    if len(p) == 7 and head.isdigit() and tail.isdigit() and 1 <= int(tail) <= 12:
        return f"{MONTHS[int(tail) - 1][1 if en else 0]} {head}"
    return p.replace("/", " - ")


def status_name(status: str) -> str:
    pair = STATUS.get(status)
    return t(*pair) if pair else status


def product_name(entry: dict, language: str | None = None) -> str:
    """The name of a product (composite or crop-mark map) in the language of the interface."""
    en = (language or lang()) == "en"
    kind = KINDS.get(entry.get("kind", ""), (entry.get("kind", ""), entry.get("kind", "")))[1 if en else 0]
    text = f"{kind} {entry.get('index', '')} {period_name(entry.get('period', ''), 'en' if en else 'it')}"
    params = entry.get("params") or {}
    if entry.get("kind") == "cropmarks" and params.get("background_m"):
        text += (", background " if en else ", sfondo ") + f"{float(params['background_m']):g} m"
    return text


def layer_name() -> str:
    return t(*LAYER)
