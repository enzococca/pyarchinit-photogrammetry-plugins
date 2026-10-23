"""Texts of the plugin in the language of the interface, and the names of its products.

The names stored in the project (``chunk.meta["terrain"]``) and in the GeoPackages are Italian:
they are data, as the names of the program's own layers. The pane, the layers of the view and the
QGIS project show them in the language of the interface; the program's catalogue of English texts
belongs to the program, so the plugin keeps its own small Italian-English table here.
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


# product id: (Italian name, as stored in the project; English)
PRODUCTS = {
    "flow_accumulation": ("Accumulo di flusso (log10 celle)", "Flow accumulation (log10 cells)"),
    "streams": ("Reticolo (tratti)", "Stream network (links)"),
    "basins": ("Bacini", "Basins"),
    "depth": ("Profondità delle depressioni", "Depth of the depressions"),
    "depression_levels": ("Livelli delle depressioni", "Levels of the depressions"),
    "mound_height": ("Altezza dei rilievi", "Height of the mounds"),
    "mound_levels": ("Livelli dei rilievi", "Levels of the mounds"),
}
# table of a GeoPackage: (Italian title, the identifier written in the file; English)
TABLES = {
    "reticolo": ("Reticolo idrografico", "Stream network"),
    "bacini": ("Bacini", "Basins"),
    "depressioni": ("Depressioni", "Depressions"),
    "rilievi": ("Rilievi", "Mounds"),
}
# the layers of the raster view of the program (shown in its Layers pane, never saved)
LAYERS = {
    "reticolo": ("Terreno: reticolo", "Terrain: streams"),
    "bacini": ("Terreno: bacini", "Terrain: basins"),
    "depressioni": ("Terreno: depressioni", "Terrain: depressions"),
    "rilievi": ("Terreno: rilievi", "Terrain: mounds"),
}


def layer_name(table: str) -> str:
    """The layer of the raster view that shows a vector table, in the language of the interface."""
    return t(*LAYERS[table])


def table_title(table: str) -> str:
    """The title of a vector table in the language of the interface."""
    return t(*TABLES[table]) if table in TABLES else table
