"""The vocabulary of the plugin: the terms of four glossaries shipped as JSON (``thesaurus/``) and the local terms of a
project.

* ``iscs.json`` - ICOMOS-ISCS, *Illustrated glossary on stone deterioration patterns* (2008): its terms are the
  plugin's codes (``ISCS.CRACK``, sub-types under their term: ``ISCS.CRACK.HAIR_CRACK``; a sub-type is found also by
  its short form, ``ISCS.BLACK_CRUST`` = ``ISCS.CRUST.BLACK_CRUST``);
* ``uni11182.json`` - the 27 terms of UNI 11182:2006, each mapped onto the ICOMOS-ISCS term or terms it corresponds
  to (the mapping is the plugin's: no official table exists), the Italian face of the codes;
* ``masonry.json`` - masonry and structural symptoms, in neither glossary: ``LOCAL`` terms, flagged as such;
* ``interventions.json`` - the kinds of intervention after the families of EN 15898.

Names are the glossaries' own; definitions are the plugin's one-line paraphrases with the source cited; colours and
hatches are the plugin's defaults (no standard prescribes them), changed per project in
``chunk.meta["conservation"]["legend"]``. A project's own terms (``LOCAL.<SLUG>``) are kept in
``chunk.meta["conservation"]["local_terms"]`` and loaded with :func:`load` of that chunk.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path
from xml.sax.saxutils import quoteattr

THESAURUS = Path(__file__).resolve().parent / "thesaurus"
FILES = ("iscs.json", "uni11182.json", "masonry.json", "interventions.json")
FAMILIES = {
    "general": ("Termini generali", "General terms"),
    "crack_deformation": ("Fessurazione e deformazione", "Crack and deformation"),
    "detachment": ("Distacco", "Detachment"),
    "material_loss": ("Forme indotte da perdita di materiale", "Features induced by material loss"),
    "discoloration_deposit": ("Alterazione cromatica e deposito", "Discoloration and deposit"),
    "biological": ("Colonizzazione biologica", "Biological colonization"),
    "masonry": ("Murature (termini locali)", "Masonry (local terms)"),
    "intervention": ("Interventi", "Interventions"),
}
KINDS = ("decay", "crack", "treatment")
GEOMETRIES = ("area", "line", "point")
GEOMETRY_TYPES = {"point": 0, "line": 1, "area": 2}  # QGIS
COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
LOCAL_COLOURS = ("#8c564b", "#bcbd22", "#e377c2", "#7f7f7f", "#17becf", "#aec7e8", "#ffbb78", "#98df8a")
LOCAL_HATCHES = (("\\\\", "ANSI32"), ("//", "ANSI31"), ("xx", "ANSI37"), ("..", "DOTS"), ("++", "CROSS"))


@dataclass
class Term:
    """A term of the vocabulary. ``uni``: the UNI 11182 code it corresponds to (a UNI term: its own code);
    ``iscs``: for UNI terms, the ICOMOS-ISCS codes it is mapped onto; ``local``: not from a standard."""

    code: str
    family: str
    kind: str
    names: dict
    definition: dict
    geometry: str = "area"
    colour: str = "#808080"
    hatch: dict = field(default_factory=lambda: {"mpl": "//", "dxf": "ANSI31"})
    uni: str | None = None
    iscs: list | None = None
    local: bool = False
    parent: str | None = None
    aliases: tuple = ()
    source: str = ""
    group: str | None = None
    deterioration: bool = True  # False: a change "without visible deterioration" (ICOMOS-ISCS alteration, patina)

    @classmethod
    def from_dict(cls, d: dict) -> "Term":
        names = dict(d.get("names") or {})
        names.setdefault("en", names.get("it", d["code"]))
        names.setdefault("it", names["en"])
        names.setdefault("fr", names["en"])
        definition = dict(d.get("definition") or {})
        definition.setdefault("en", definition.get("it", ""))
        definition.setdefault("it", definition["en"])
        return cls(code=d["code"], family=d["family"], kind=d["kind"], names=names, definition=definition,
                   geometry=d.get("geometry", "area"), colour=d.get("colour", "#808080"),
                   hatch=dict(d.get("hatch") or {"mpl": "//", "dxf": "ANSI31"}), uni=d.get("uni"),
                   iscs=list(d["iscs"]) if d.get("iscs") else None, local=bool(d.get("local", False)),
                   parent=d.get("parent"), aliases=tuple(d.get("aliases") or ()), source=d.get("source", ""),
                   group=d.get("group"), deterioration=bool(d.get("deterioration", True)))

    def to_dict(self) -> dict:
        out = {"code": self.code, "family": self.family, "kind": self.kind, "geometry": self.geometry,
               "names": dict(self.names), "definition": dict(self.definition), "colour": self.colour,
               "hatch": dict(self.hatch), "local": self.local, "source": self.source}
        for key in ("uni", "parent", "group"):
            if getattr(self, key):
                out[key] = getattr(self, key)
        if self.iscs:
            out["iscs"] = list(self.iscs)
        if self.aliases:
            out["aliases"] = list(self.aliases)
        if not self.deterioration:
            out["deterioration"] = False
        return out

    def copy(self) -> "Term":
        return replace(self, names=dict(self.names), definition=dict(self.definition), hatch=dict(self.hatch),
                       iscs=list(self.iscs) if self.iscs else None)


def _fold(text: str) -> str:
    """Lower case without accents: what a search compares."""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


class Vocabulary:
    """The terms, in the order of the files (ISCS, UNI, local masonry terms, interventions, the project's own)."""

    def __init__(self, terms):
        self._terms: dict[str, Term] = {}
        for term in terms:
            self._terms[term.code] = term
        short: dict[str, list[str]] = {}
        for code, term in self._terms.items():
            if term.parent and code.count(".") >= 2:
                short.setdefault(f"{code.split('.')[0]}.{code.rsplit('.', 1)[1]}", []).append(code)
        self._short = {k: v[0] for k, v in short.items() if len(v) == 1 and k not in self._terms}

    def __contains__(self, code) -> bool:
        return code in self._terms or code in self._short

    def terms(self, kind: str | None = None, family: str | None = None) -> list[Term]:
        return [t.copy() for t in self._terms.values() if (kind is None or t.kind == kind)
                and (family is None or t.family == family)]

    def term(self, code: str) -> Term:
        """The term of ``code`` (a sub-type also by its short form); KeyError naming the known codes."""
        if code in self._terms:
            return self._terms[code].copy()
        if code in self._short:
            return self._terms[self._short[code]].copy()
        raise KeyError(f"unknown term {code!r}; the known codes: {', '.join(self._terms)}")

    def families(self, lang: str = "it") -> list[tuple[str, str]]:
        i = 1 if lang == "en" else 0
        return [(f, titles[i]) for f, titles in FAMILIES.items()]

    def name(self, code: str, lang: str = "it") -> str:
        """The name the interface shows: in Italian the UNI 11182 name first when the term has one, with the
        ICOMOS-ISCS name in brackets (``fessurazione (crack)``); in English the ICOMOS-ISCS name (a UNI term: the
        ISCS name of its first mapping, with the UNI name in brackets)."""
        term = self.term(code)
        if lang == "en":
            if term.iscs:
                return f"{self.term(term.iscs[0]).names['en']} (UNI: {term.names['it']})"
            return term.names["en"]
        if not term.code.startswith("ISCS."):
            return term.names["it"]
        uni = self.term(term.uni) if term.uni and term.uni in self else None
        if uni is not None and term.parent and len(uni.iscs or ()) > 1 and \
                _fold(uni.names["it"]) != _fold(term.names["it"]):
            return f"{term.names['it']} (UNI: {uni.names['it']})"  # a sub-type of a wider UNI term
        base = uni.names["it"] if uni is not None else term.names["it"]
        english = term.names["en"].lower()
        return base if _fold(base) == _fold(english) else f"{base} ({english})"

    def search(self, text: str, lang: str = "it") -> list[Term]:
        """The terms whose names (Italian, English, French), aliases or code contain ``text`` (no matter the case or
        the accents); the names starting with it in ``lang`` first."""
        needle = _fold(text).strip()
        if not needle:
            return []
        ranked = []
        for i, term in enumerate(self._terms.values()):
            own = [_fold(term.names.get(lang, "")), _fold(self.name(term.code, lang))]
            every = own + [_fold(n) for n in term.names.values()] + [_fold(a) for a in term.aliases] + \
                [_fold(term.code)]
            if not any(needle in n for n in every):
                continue
            rank = 0 if any(n.startswith(needle) for n in own) else 1 if any(needle in n for n in own) else 2
            ranked.append((rank, i, term))
        return [t for _r, _i, t in sorted(ranked, key=lambda x: (x[0], x[1]))]

    def interventions(self) -> list[Term]:
        return self.terms(kind="treatment")


@lru_cache(maxsize=1)
def _base() -> tuple[Term, ...]:
    terms = []
    for name in FILES:
        data = json.loads((THESAURUS / name).read_text(encoding="utf-8"))
        terms.extend(Term.from_dict(d) for d in data["terms"])
    return tuple(terms)


@lru_cache(maxsize=1)
def _base_vocabulary() -> Vocabulary:
    return Vocabulary(_base())


def load(chunk=None) -> Vocabulary:
    """The vocabulary (cached); with ``chunk``, also that project's local terms."""
    if chunk is None:
        return _base_vocabulary()
    local = ((getattr(chunk, "meta", None) or {}).get("conservation") or {}).get("local_terms") or []
    if not local:
        return _base_vocabulary()
    return Vocabulary(list(_base()) + [Term.from_dict(d) for d in local])


def deteriorates(vocab: Vocabulary, term: Term) -> bool:
    """Whether a term is a deterioration pattern: not a general term, not a change "without visible deterioration"
    (nor a sub-type of one, nor a UNI term mapped onto one)."""
    if term.iscs:
        term = vocab.term(term.iscs[0])
    while True:
        if term.family == "general" or not term.deterioration:
            return False
        if not term.parent or term.parent not in vocab:
            return True
        term = vocab.term(term.parent)


def legend(chunk) -> dict[str, dict]:
    """``{code: {"colour", "hatch"}}`` of every term: the defaults, changed by the project's legend."""
    own = ((getattr(chunk, "meta", None) or {}).get("conservation") or {}).get("legend") or {}
    out = {}
    for term in load(chunk).terms():
        entry = {"colour": term.colour, "hatch": dict(term.hatch)}
        change = own.get(term.code) or {}
        if change.get("colour"):
            entry["colour"] = change["colour"]
        if change.get("hatch"):
            entry["hatch"].update(change["hatch"])
        out[term.code] = entry
    return out


def set_legend(chunk, code: str, colour: str | None = None, hatch: dict | None = None) -> dict:
    """The colour (``#rrggbb``) and/or the hatch (``{"mpl", "dxf"}``) of a term in this project; returns the entry
    of the project's legend."""
    from .records import store

    code = load(chunk).term(code).code
    if colour is not None and not COLOUR.match(str(colour)):
        raise ValueError(f"a colour is written #rrggbb, not {colour!r}")
    if hatch is not None and (not isinstance(hatch, dict) or not set(hatch) <= {"mpl", "dxf"}):
        raise ValueError("a hatch is {'mpl': ..., 'dxf': ...}")
    entry = store(chunk)["legend"].setdefault(code, {})
    if colour is not None:
        entry["colour"] = str(colour).lower()
    if hatch is not None:
        entry.setdefault("hatch", {}).update({k: str(v) for k, v in hatch.items()})
    return entry


def _slug(text: str) -> str:
    slug = re.sub(r"[^A-Z0-9]+", "_", _fold(text).upper()).strip("_")
    if not slug:
        raise ValueError("a local term needs a name made of letters or digits")
    return slug


def add_local(chunk, slug: str, names: dict, kind: str, geometry: str, definition: dict) -> Term:
    """A term of this project (``LOCAL.<SLUG>``), flagged as not from a standard; ``names`` and ``definition`` in
    Italian and/or English. ValueError for a wrong kind or geometry, or a code already taken."""
    from .records import store

    if kind not in KINDS:
        raise ValueError(f"the kind of a term is one of {', '.join(KINDS)}, not {kind!r}")
    if geometry not in GEOMETRIES:
        raise ValueError(f"the geometry of a term is one of {', '.join(GEOMETRIES)}, not {geometry!r}")
    if not (names or {}).get("it") and not (names or {}).get("en"):
        raise ValueError("a local term needs a name in Italian or English")
    code = f"LOCAL.{_slug(slug)}"
    if code in load(chunk):
        raise ValueError(f"the term {code} exists already")
    taken = store(chunk)["local_terms"]
    n = len(taken)
    mpl, dxf = LOCAL_HATCHES[n % len(LOCAL_HATCHES)]
    term = Term.from_dict({"code": code, "family": "intervention" if kind == "treatment" else "masonry",
                           "kind": kind, "geometry": geometry, "names": dict(names),
                           "definition": dict(definition or {}), "colour": LOCAL_COLOURS[n % len(LOCAL_COLOURS)],
                           "hatch": {"mpl": mpl, "dxf": dxf}, "local": True, "source": "local term of the project"})
    taken.append(term.to_dict())
    return term


# ---------------------------------------------------------------- QGIS styles
def _rgb(colour: str) -> tuple[int, int, int]:
    c = colour.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _hatch_layers(hatch: str, colour: str, index: int = 0) -> list[str]:
    """QGIS symbol layers drawing a matplotlib hatch: lines at its angles, points for dots and circles; the
    repetition of a character makes it denser."""
    angles = []
    for ch, a in (("/", 45), ("\\", 135), ("|", 90), ("-", 0)):
        if ch in hatch:
            angles.append(a)
    if "x" in hatch.lower() and not angles:
        angles = [45, 135]
    if "+" in hatch:
        angles += [a for a in (0, 90) if a not in angles]
    count = max(1, max((hatch.count(c) for c in set(hatch)), default=1))
    distance = round(3.0 / count, 2)
    layers = []
    for a in angles:
        k = len(layers) + 1  # the simple fill is layer 0
        layers.append(
            '<layer class="LinePatternFill" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="angle" type="QString" value="{a}"/>'
            f'<Option name="distance" type="QString" value="{distance}"/>'
            '<Option name="distance_unit" type="QString" value="MM"/>'
            f'<Option name="color" type="QString" value="{colour},255"/>'
            '<Option name="line_width" type="QString" value="0.2"/></Option>'
            f'<symbol type="line" name="@{index}@{k}" alpha="1" clip_to_extent="1" force_rhr="0">'
            '<layer class="SimpleLine" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="line_color" type="QString" value="{colour},255"/>'
            '<Option name="line_width" type="QString" value="0.2"/>'
            '<Option name="line_width_unit" type="QString" value="MM"/></Option></layer></symbol></layer>')
    if not angles and any(c in hatch for c in ".oO*"):
        size = 0.6 if "." in hatch else 1.2
        layers.append(
            '<layer class="PointPatternFill" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="distance_x" type="QString" value="{distance}"/>'
            f'<Option name="distance_y" type="QString" value="{distance}"/>'
            '<Option name="distance_x_unit" type="QString" value="MM"/>'
            '<Option name="distance_y_unit" type="QString" value="MM"/></Option>'
            f'<symbol type="marker" name="@{index}@{len(layers) + 1}" alpha="1" clip_to_extent="1" force_rhr="0">'
            '<layer class="SimpleMarker" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="name" type="QString" value="{"star" if "*" in hatch else "circle"}"/>'
            f'<Option name="color" type="QString" value="{colour},{255 if "." in hatch else 0}"/>'
            f'<Option name="outline_color" type="QString" value="{colour},255"/>'
            f'<Option name="size" type="QString" value="{size}"/></Option></layer></symbol></layer>')
    return layers


def _symbol(index: int, colour: str, hatch: str, geometry: str) -> str:
    rgb = ",".join(str(c) for c in _rgb(colour))
    head = f'<symbol type="{{}}" name="{index}" alpha="1" clip_to_extent="1" force_rhr="0">'
    if geometry == "line":
        return (head.format("line") + '<layer class="SimpleLine" enabled="1" locked="0" pass="0"><Option type="Map">'
                f'<Option name="line_color" type="QString" value="{rgb},255"/>'
                '<Option name="line_width" type="QString" value="0.5"/>'
                '<Option name="line_width_unit" type="QString" value="MM"/></Option></layer></symbol>')
    if geometry == "point":
        return (head.format("marker") + '<layer class="SimpleMarker" enabled="1" locked="0" pass="0">'
                '<Option type="Map"><Option name="name" type="QString" value="circle"/>'
                f'<Option name="color" type="QString" value="{rgb},255"/>'
                '<Option name="outline_color" type="QString" value="35,35,35,255"/>'
                '<Option name="size" type="QString" value="2.4"/></Option></layer></symbol>')
    fill = ('<layer class="SimpleFill" enabled="1" locked="0" pass="0"><Option type="Map">'
            f'<Option name="color" type="QString" value="{rgb},70"/>'
            '<Option name="style" type="QString" value="solid"/>'
            f'<Option name="outline_color" type="QString" value="{rgb},255"/>'
            '<Option name="outline_style" type="QString" value="solid"/>'
            '<Option name="outline_width" type="QString" value="0.4"/>'
            '<Option name="outline_width_unit" type="QString" value="MM"/></Option></layer>')
    return head.format("fill") + fill + "".join(_hatch_layers(hatch, rgb, index)) + "</symbol>"


def write_qml(path, codes, legend: dict, geometry: str = "area", labels: dict | None = None,
              chunk=None) -> Path:
    """A QGIS 3 style (``.qml``): a categorized renderer on the column ``code``, one category per code with the
    legend's colour (a translucent fill and the outline) and its hatch drawn as line (or point) pattern fills;
    ``geometry`` "line" or "point" for tables of lines or points. ``labels``: ``{code: label}`` (default: the names
    of the vocabulary - with ``chunk``, also its local terms - in the interface's language, else the code)."""
    from .texts import lang

    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry is one of {', '.join(GEOMETRIES)}")
    vocab = load(chunk)
    categories, symbols = [], []
    for i, code in enumerate(dict.fromkeys(codes)):
        style = legend.get(code) or {"colour": "#808080", "hatch": {"mpl": "", "dxf": "SOLID"}}
        if labels and code in labels:
            label = labels[code]
        else:
            label = f"{code} - {vocab.name(code, lang())}" if code in vocab else code
        categories.append(f'<category render="true" symbol="{i}" value={quoteattr(code)} label={quoteattr(label)}/>')
        symbols.append(_symbol(i, style["colour"], (style.get("hatch") or {}).get("mpl", ""), geometry))
    text = ("<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>\n"
            '<qgis version="3.34.0" styleCategories="Symbology">\n'
            '<renderer-v2 type="categorizedSymbol" attr="code" symbollevels="0" enableorderby="0" forceraster="0">\n'
            "<categories>\n" + "\n".join(categories) + "\n</categories>\n"
            "<symbols>\n" + "\n".join(symbols) + "\n</symbols>\n"
            "</renderer-v2>\n"
            f"<layerGeometryType>{GEOMETRY_TYPES[geometry]}</layerGeometryType>\n"
            "</qgis>\n")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


__all__ = ["Term", "Vocabulary", "load", "legend", "set_legend", "add_local", "write_qml", "FAMILIES", "KINDS",
           "GEOMETRIES", "THESAURUS"]
