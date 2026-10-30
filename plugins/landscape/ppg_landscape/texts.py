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


# the parameters of the scenario in the pane: key -> (label it, label en, tooltip it, tooltip en, minimum, maximum,
# decimals); the values are those of engine.DEFAULTS / engine.PRESETS (metres, years)
PARAMETERS = {
    "D": ("Diffusività del versante D (m²/anno)", "Hillslope diffusivity D (m²/yr)",
          "Quanto in fretta il suolo scivola giù per il pendio (creep): il flusso è D per la pendenza. Valori tipici "
          "1e-3 - 2e-2 m²/anno; mattoni crudi e terra smossa stanno in alto.",
          "How fast soil creeps down the slope: the flux is D times the slope. Typical values 1e-3 - 2e-2 m²/yr; "
          "mudbrick and loose earth are at the top.", 0.0, 10.0, 5),
    "K_sed": ("Erodibilità del suolo K_sed (1/anno)", "Erodibility of the soil K_sed (1/yr)",
              "Quanto l'acqua che scorre porta via il suolo sciolto (SPACE). Più alto = canali più rapidi.",
              "How readily running water entrains the loose soil (SPACE). Higher = faster channels.", 0.0, 1.0, 6),
    "K_br": ("Erodibilità del substrato K_br (1/anno)", "Erodibility of the substrate K_br (1/yr)",
             "Quanto l'acqua erode il materiale sotto il suolo (il corpo del tell, il terreno compatto).",
             "How readily water erodes the material under the soil (the body of the tell, the compact ground).",
             0.0, 1.0, 6),
    "runoff": ("Ruscellamento (m/anno)", "Runoff (m/yr)",
               "La pioggia che scorre in superficie in un anno, uniforme: la portata è area drenata per questo valore.",
               "The rain running off the surface in a year, uniform: the discharge is the drained area times this.",
               0.0, 10.0, 3),
    "soil_depth": ("Spessore del suolo sciolto (m)", "Thickness of the loose soil (m)",
                   "Il suolo mobile sopra il substrato all'inizio.", "The mobile soil above the substrate at the start.",
                   0.0, 100.0, 3),
    "F_f": ("Frazione fine persa F_f (0-1)", "Fine fraction lost F_f (0-1)",
            "La parte del substrato eroso così fine da andare via in sospensione senza depositarsi.",
            "The part of the eroded substrate fine enough to leave in suspension without settling.", 0.0, 1.0, 3),
    "phi": ("Porosità del sedimento (0-1)", "Porosity of the sediment (0-1)",
            "I vuoti del sedimento depositato.", "The voids of the deposited sediment.", 0.0, 0.95, 3),
    "H_star": ("Scala dello spessore H* (m)", "Thickness scale H* (m)",
               "Sotto questo spessore di suolo l'acqua comincia a intaccare il substrato.",
               "Below this thickness of soil the water starts to cut into the substrate.", 0.001, 100.0, 3),
    "v_s": ("Velocità di deposito v_s (m/anno)", "Settling velocity v_s (m/yr)",
            "Quanto in fretta il sedimento si deposita: più alta = depositi più vicini all'erosione.",
            "How fast the sediment settles: higher = deposits closer to where it eroded.", 0.0, 1000.0, 3),
    "m_sp": ("Esponente della portata m", "Discharge exponent m",
             "Esponente dell'area drenata nella legge di erosione (di solito 0,5).",
             "Exponent of the drained area in the erosion law (usually 0.5).", 0.0, 3.0, 3),
    "n_sp": ("Esponente della pendenza n", "Slope exponent n",
             "Esponente della pendenza nella legge di erosione (di solito 1).",
             "Exponent of the slope in the erosion law (usually 1).", 0.1, 3.0, 3),
}
FLUVIAL_KEYS = ("K_sed", "K_br", "runoff", "soil_depth", "F_f", "phi", "H_star", "v_s", "m_sp", "n_sp")
# the layers of a run, as the pane and the tree name them
LAYER_TITLES = {
    "final": ("Superficie finale", "Final surface"),
    "dz": ("Erosione e deposito (dz)", "Erosion and deposition (dz)"),
    "area": ("Area drenata", "Drainage area"),
    "snap": ("Superficie dopo {0} anni", "Surface after {0} years"),
    "compare": ("Differenza con il DEM di «{0}»", "Difference with the DEM of '{0}'"),
}
BOUNDARY_TITLES = {"open": ("Bordo aperto (acqua e sedimento escono)", "Open edge (water and sediment leave)"),
                   "closed": ("Bordo chiuso (esce solo dal punto più basso)",
                              "Closed edge (out only at the lowest point)")}
SCENARIO_NOTE = ("Sono scenari: il risultato dipende dai parametri, non è una previsione. Confrontalo con il DEM "
                 "di una fase successiva per tarare i parametri.",
                 "These are scenarios: the result depends on the parameters, it is not a prediction. Compare it "
                 "with the DEM of a later phase to calibrate the parameters.")


def layer_title(layer: str, entry: dict | None = None) -> str:
    """The name of a layer of a run in the language of the interface (``snap:<year>``, ``compare:<key>``)."""
    if layer.startswith("snap:"):
        return t(*LAYER_TITLES["snap"]).format(years(float(layer[5:])))
    if layer.startswith("compare:"):
        key = layer[8:]
        found = next((c for c in (entry or {}).get("comparisons") or [] if str(c.get("chunk_key")) == key), None)
        label = found.get("chunk", key) if found else key
        if found and found.get("source") not in (None, "", "active"):
            label = f"{label}: {found.get('source_name', found['source'])}"
        return t(*LAYER_TITLES["compare"]).format(label)
    return t(*LAYER_TITLES.get(layer, (layer, layer)))
