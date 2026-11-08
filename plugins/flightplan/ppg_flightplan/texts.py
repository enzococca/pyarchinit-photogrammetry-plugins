"""Texts of the plugin in the language of the interface.

What is stored in the plan (``chunk.meta["flightplan"]``) is data: warning codes (``velocita_ridotta``,
``quota_oltre_120``...), the shooting mode (``intervallo``, ``per_punto``), the block kinds never change with
the language; the pane and the files show them through :func:`warning_text` in the language of the interface.
The program's catalogue of English texts belongs to the program, so the plugin keeps its own small table.
"""
from __future__ import annotations

NAME = ("Piani di volo", "Flight plans")


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
    """``one`` or ``many`` (already in the language of the interface, see :func:`t`) for ``n`` things,
    ``{0}`` the number: ``plural(1, "{0} scatto", "{0} scatti")`` is "1 scatto"."""
    text = one if abs(float(n)) == 1 else many
    return text.format(n) if "{0}" in text else text


# Every warning code of the planner, in words. The codes are stored in the plans: never rename them.
WARNINGS: dict[str, tuple[str, str]] = {
    # the route (Route.warnings)
    "quota_oltre_120": ("Quota oltre 120 m sul decollo: un drone di serie non sale oltre",
                        "Height above 120 m over the takeoff: a stock drone does not climb higher"),
    "terreno_oltre_120": ("Seguendo il terreno alcuni punti superano i 120 m sul decollo",
                          "Following the terrain some points go above 120 m over the takeoff"),
    "velocita_ridotta": ("Velocità ridotta: alla velocità voluta la camera non scatta abbastanza spesso",
                         "Speed reduced: at the speed asked the camera cannot shoot often enough"),
    "piu_tratte": ("Il volo non sta in una batteria: diviso in più tratte",
                   "The flight does not fit one battery: split into several legs"),
    "punti_pochi": ("Meno di due punti: non è una rotta", "Fewer than two points: not a route"),
    # the special points
    "senza_terreno": ("Suolo sconosciuto: la discesa «dal suolo» non si fa",
                      "Ground unknown: the descent 'from the ground' is not made"),
    "suolo_sconosciuto": ("Si scende senza sapere dov'è il suolo: controlla tu cosa c'è sotto",
                          "Descending without knowing where the ground is: check what is below"),
    "alzato": ("Discesa alzata: sarebbe finita a meno di 5 m dal suolo",
               "Descent raised: it would have ended less than 5 m from the ground"),
    "sotto_il_decollo": ("Discesa alzata a 5 m sul decollo", "Descent raised to 5 m above the takeoff"),
    "gia_bassa": ("La rotta è già a quella quota: nessuna discesa", "The route is already at that height: no descent"),
    "nessun_effetto": ("Quel waypoint fa già la foto", "That waypoint already takes the photo"),
    # the terrain
    "oltre_limite_di_serie": ("Il piano sale oltre il limite di serie di 120 m sul decollo",
                              "The plan climbs above the stock limit of 120 m over the takeoff"),
    # the GNSS targets
    "codice_doppio": ("Due target con lo stesso numero", "Two targets with the same number"),
    "lontani_dal_sito": ("I target sono a più di 5 km dal sito", "The targets are more than 5 km from the site"),
    "ordine_incerto": ("Ordine delle colonne incerto: controlla latitudine e longitudine",
                       "Order of the columns uncertain: check latitude and longitude"),
    "senza_codice": ("Alcuni target senza numero", "Some targets without a number"),
    "senza_quota": ("Alcuni target senza quota: vanno bene per volarci sopra, non come GCP",
                    "Some targets without a height: fine to fly over, not as GCPs"),
    "zona_dedotta": ("Zona UTM dedotta dal sito", "UTM zone deduced from the site"),
}


def warning_text(code: str) -> str:
    """A warning code in words, in the language of the interface; an unknown code as it is."""
    pair = WARNINGS.get(code)
    return t(*pair) if pair else code


def name() -> str:
    return t(*NAME)
