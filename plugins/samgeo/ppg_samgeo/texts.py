"""Texts of the plugin in the language of the interface, the numbers as the interface writes them,
and what to look for: the ready entries of the pane, in Italian and English, with the text the models
read (in English: Grounding DINO and SAM 3 were trained on English captions).

The program's catalogue of English texts belongs to the program, so the plugin keeps its own small
Italian-English pairs, ``t(it, en)``.
"""
from __future__ import annotations

import re
import unicodedata

# (Italian, English shown, text given to the models): what an excavation or a survey looks for
PRESETS = (
    ("pietre", "stones", "stone"),
    ("muri", "walls", "stone wall"),
    ("mattoni", "bricks", "brick"),
    ("tegole e coppi", "roof tiles", "roof tile"),
    ("frammenti ceramici", "pottery sherds", "pottery sherd"),
    ("ossa", "bones", "bone"),
    ("tessere di mosaico", "mosaic tesserae", "mosaic tile"),
    ("alberi", "trees", "tree"),
    ("cespugli", "bushes", "bush"),
    ("persone", "people", "person"),
    ("attrezzi e secchi", "tools and buckets", "bucket. shovel. wheelbarrow"),
    ("teli e coperture", "tarpaulins", "tarpaulin"),
)


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
    if round(value, decimals) == 0:
        value = 0.0  # never "-0,00"
    text = f"{value:,.{decimals}f}"
    if lang() == "it":
        text = text.replace(",", "\x00").replace(".", ",").replace("\x00", ".")
    return text


def plural(n: int, one: str, many: str) -> str:
    return one if n == 1 else many


def presets() -> list[tuple[str, str]]:
    """The ready entries of the pane: ``(text shown, text given to the models)``."""
    return [(it if lang() == "it" else en, model) for it, en, model in PRESETS]


def model_text(text: str) -> str:
    """What the models read for the text the user wrote: a ready entry (in Italian or English)
    becomes its English text, anything else is kept as written (English works best). Phrases
    separated by commas, semicolons or full stops become the phrases of Grounding DINO, lower case,
    each closed by a full stop: ``"pietre, mattoni"`` -> ``"stone. brick."``."""
    known = {}
    for it, en, model in PRESETS:
        for key in (it, en, model):
            known[key.lower()] = model
    phrases = []
    whole = known.get(" ".join(text.lower().split()))
    for part in [whole] if whole else re.split(r"[,;.\n]+", text):
        part = " ".join(part.lower().split())
        if not part:
            continue
        for p in (known.get(part, part)).split("."):
            p = p.strip()
            if p and p not in phrases:
                phrases.append(p)
    return " ".join(p + "." for p in phrases)


def slug(text: str, fallback: str = "samgeo") -> str:
    """A name for files and formulas: ASCII letters, digits and ``_`` (``tegole e coppi`` ->
    ``tegole_e_coppi``)."""
    plain = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()
    plain = re.sub(r"[^a-z0-9]+", "_", plain).strip("_")[:40].strip("_")
    return plain or fallback
