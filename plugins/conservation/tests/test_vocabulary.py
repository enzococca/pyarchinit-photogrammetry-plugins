"""The vocabulary: ICOMOS-ISCS (the codes), UNI 11182 mapped onto it, the plugin's local masonry terms, the
interventions; names in the interface's language, search, the legend of a project, local terms, QGIS styles."""
from __future__ import annotations

import json
import re

import pytest

from ppg_conservation import vocabulary
from ppg_conservation.vocabulary import FAMILIES, KINDS

# the ICOMOS-ISCS overview (design/conservation-research/standards.md, section 1): main terms and sub-types per family
ISCS_MAIN = {
    "general": ["ALTERATION", "DAMAGE", "DECAY", "DEGRADATION", "DETERIORATION", "WEATHERING"],
    "crack_deformation": ["CRACK", "DEFORMATION"],
    "detachment": ["BLISTERING", "BURSTING", "DELAMINATION", "DISINTEGRATION", "FRAGMENTATION", "PEELING",
                   "SCALING"],
    "material_loss": ["ALVEOLIZATION", "EROSION", "MECHANICAL_DAMAGE", "MICROKARST", "MISSING_PART", "PERFORATION",
                      "PITTING"],
    "discoloration_deposit": ["CRUST", "DEPOSIT", "DISCOLOURATION", "EFFLORESCENCE", "ENCRUSTATION", "FILM",
                              "GLOSSY_ASPECT", "GRAFFITI", "PATINA", "SOILING", "SUBFLORESCENCE"],
    "biological": ["BIOLOGICAL_COLONIZATION", "ALGA", "LICHEN", "MOSS", "MOULD", "PLANT"],
}
ISCS_SUB = {
    "CRACK": ["FRACTURE", "STAR_CRACK", "HAIR_CRACK", "CRAQUELE", "SPLITTING"],
    "DELAMINATION": ["EXFOLIATION"],
    "DISINTEGRATION": ["CRUMBLING", "GRANULAR_DISINTEGRATION", "POWDERING", "SANDING", "SUGARING"],
    "FRAGMENTATION": ["SPLINTERING", "CHIPPING"],
    "SCALING": ["FLAKING", "CONTOUR_SCALING"],
    "ALVEOLIZATION": ["COVING"],
    "EROSION": ["DIFFERENTIAL_EROSION", "LOSS_OF_COMPONENTS", "LOSS_OF_MATRIX", "ROUNDING", "ROUGHENING"],
    "MECHANICAL_DAMAGE": ["IMPACT_DAMAGE", "CUT", "SCRATCH", "ABRASION", "KEYING"],
    "MISSING_PART": ["GAP"],
    "CRUST": ["BLACK_CRUST", "SALT_CRUST"],
    "DISCOLOURATION": ["COLOURATION", "BLEACHING", "MOIST_AREA", "STAINING"],
    "ENCRUSTATION": ["CONCRETION"],
    "PATINA": ["IRON_RICH_PATINA", "OXALATE_PATINA"],
}
UNI_TERMS = ["alterazione cromatica", "alveolizzazione", "colatura", "colonizzazione biologica", "crosta",
             "deformazione", "degradazione differenziale", "deposito superficiale", "disgregazione", "distacco",
             "efflorescenza", "erosione", "esfoliazione", "fessurazione", "fronte di risalita", "graffito vandalico",
             "incrostazione", "lacuna", "macchia", "mancanza", "patina", "patina biologica", "pellicola", "pitting",
             "presenza di vegetazione", "rigonfiamento", "scagliatura"]
HEX = re.compile(r"^#[0-9a-f]{6}$")


def test_iscs_terms_and_sub_types():
    v = vocabulary.load()
    iscs = [t for t in v.terms() if t.code.startswith("ISCS.")]
    main = [t for t in iscs if t.parent is None]
    # the glossary's overview lists 39 main terms (6 general + 33 patterns): see the report of the task
    assert len(main) == sum(len(x) for x in ISCS_MAIN.values()) == 39
    for family, keys in ISCS_MAIN.items():
        assert {t.code for t in main if t.family == family} == {f"ISCS.{k}" for k in keys}
    subs = [t for t in iscs if t.parent is not None]
    assert len(subs) == sum(len(x) for x in ISCS_SUB.values())
    for parent, keys in ISCS_SUB.items():
        for k in keys:
            term = v.term(f"ISCS.{parent}.{k}")
            assert term.family == v.term(f"ISCS.{parent}").family
    for t in iscs:
        assert t.names["en"] and t.names["fr"] and t.names["it"] and t.definition["en"] and t.definition["it"]
        assert HEX.match(t.colour) and t.hatch["mpl"] and t.hatch["dxf"] and t.geometry in ("area", "line", "point")
        assert t.kind in KINDS and t.family in FAMILIES and not t.local
    # the examples of the design: stable keys, a sub-type reachable by its short form too
    assert v.term("ISCS.CRACK").kind == "crack" and v.term("ISCS.CRACK").geometry == "line"
    assert v.term("ISCS.CRACK.HAIR_CRACK").parent == "ISCS.CRACK"
    assert v.term("ISCS.ALVEOLIZATION").family == "material_loss"
    assert v.term("ISCS.BLACK_CRUST").code == "ISCS.CRUST.BLACK_CRUST"
    with pytest.raises(KeyError, match="ISCS.CRACK"):
        v.term("ISCS.NOTHING")


def test_uni_terms_map_onto_iscs():
    v = vocabulary.load()
    uni = [t for t in v.terms() if t.code.startswith("UNI.")]
    assert sorted(t.names["it"] for t in uni) == sorted(UNI_TERMS)
    codes = {t.code for t in v.terms()}
    for t in uni:
        assert t.iscs and all(c in codes and c.startswith("ISCS.") for c in t.iscs), t.code
        assert t.uni == t.code and t.definition["it"] and t.definition["en"]
    assert v.term("UNI.DISTACCO").iscs == ["ISCS.DELAMINATION", "ISCS.BLISTERING", "ISCS.SCALING"]
    assert v.term("UNI.FESSURAZIONE").kind == "crack"
    # ISCS terms point back to their UNI term
    assert v.term("ISCS.CRACK").uni == "UNI.FESSURAZIONE"
    assert v.term("ISCS.CRUST.BLACK_CRUST").uni == "UNI.CROSTA"
    assert v.term("ISCS.MICROKARST").uni is None


def test_local_masonry_terms_and_interventions():
    v = vocabulary.load()
    local = v.terms(family="masonry")
    assert {t.code for t in local} == {"LOCAL.OUT_OF_PLUMB", "LOCAL.BULGING", "LOCAL.LOSS_OF_MORTAR",
                                       "LOCAL.DISPLACED_STONES", "LOCAL.OPEN_JOINT"}
    assert all(t.local for t in local)
    ints = v.interventions()
    assert {"INT.PREVENTIVE", "INT.CLEANING", "INT.CONSOLIDATION", "INT.DESALINATION", "INT.BIOCIDE",
            "INT.FILLING", "INT.GROUTING", "INT.REPOINTING", "INT.PROTECTION", "INT.REINTEGRATION",
            "INT.REPLACEMENT", "INT.REMOVAL", "INT.RESTORATION"} <= {t.code for t in ints}
    assert all(t.kind == "treatment" and t.family == "intervention" for t in ints)
    assert v.terms(kind="treatment") == ints
    assert [f for f, _ in v.families("it")] == list(FAMILIES)
    assert dict(v.families("en"))["biological"] == "Biological colonization"


def test_names_in_the_two_languages():
    v = vocabulary.load()
    assert v.name("ISCS.CRACK", "it").startswith("fessurazione")
    assert v.name("ISCS.CRACK", "it") == "fessurazione (crack)"
    assert v.name("ISCS.CRACK", "en") == "Crack"
    assert v.name("ISCS.CRACK.HAIR_CRACK", "it") == "microfessura (hair crack)"
    assert v.name("ISCS.PITTING", "it") == "pitting"
    assert v.name("UNI.DISTACCO", "it") == "distacco"


def test_search_both_languages():
    v = vocabulary.load()
    found = [t.code for t in v.search("crosta", "it")]
    assert "ISCS.CRUST.BLACK_CRUST" in found and "UNI.CROSTA" in found
    assert v.term("ISCS.BLACK_CRUST").code in found
    assert "ISCS.CRUST.BLACK_CRUST" in [t.code for t in v.search("black crust", "it")]
    assert "ISCS.ALGA" in [t.code for t in v.search("ALGUE", "en")]  # French and capitals
    assert "UNI.FESSURAZIONE" in [t.code for t in v.search("fratturazione", "en")]  # an alias
    assert v.search("", "it") == []


class FakeChunk:
    def __init__(self):
        self.meta = {}


def test_legend_and_local_terms_of_a_chunk(tmp_path):
    ch = FakeChunk()
    leg = vocabulary.legend(ch)
    assert leg["ISCS.CRACK"] == {"colour": vocabulary.load().term("ISCS.CRACK").colour,
                                 "hatch": vocabulary.load().term("ISCS.CRACK").hatch}
    vocabulary.set_legend(ch, "ISCS.CRACK", colour="#123456")
    assert vocabulary.legend(ch)["ISCS.CRACK"]["colour"] == "#123456"
    assert vocabulary.legend(ch)["ISCS.CRACK"]["hatch"] == vocabulary.load().term("ISCS.CRACK").hatch
    vocabulary.set_legend(ch, "ISCS.CRACK", hatch={"mpl": "..", "dxf": "DOTS"})
    assert vocabulary.legend(ch)["ISCS.CRACK"] == {"colour": "#123456", "hatch": {"mpl": "..", "dxf": "DOTS"}}
    with pytest.raises(ValueError):
        vocabulary.set_legend(ch, "ISCS.CRACK", colour="red")
    with pytest.raises(KeyError):
        vocabulary.set_legend(ch, "ISCS.NOTHING", colour="#000000")
    term = vocabulary.add_local(ch, "Malta cementizia", {"it": "malta cementizia", "en": "cement mortar"}, "decay",
                                "area", {"it": "integrazione in malta di cemento", "en": "a cement mortar patch"})
    assert term.code == "LOCAL.MALTA_CEMENTIZIA" and term.local and term.family == "masonry"
    assert ch.meta["conservation"]["local_terms"][0]["code"] == term.code
    with_chunk = vocabulary.load(ch)
    assert with_chunk.term(term.code).local and with_chunk.term(term.code).names["fr"] == "cement mortar"
    assert term.code in vocabulary.legend(ch)
    with pytest.raises(KeyError):
        vocabulary.load().term(term.code)  # only in that chunk's vocabulary
    with pytest.raises(ValueError):
        vocabulary.add_local(ch, "x", {"it": "x", "en": "x"}, "decay", "volume", {})
    with pytest.raises(ValueError):
        vocabulary.add_local(ch, "malta cementizia", {"it": "x", "en": "x"}, "decay", "area", {})  # taken


def test_write_qml(tmp_path):
    ch = FakeChunk()
    vocabulary.set_legend(ch, "ISCS.CRUST.BLACK_CRUST", colour="#101010")
    codes = ["ISCS.CRUST.BLACK_CRUST", "ISCS.ALGA", "INT.CLEANING"]
    path = vocabulary.write_qml(tmp_path / "degrado.qml", codes, vocabulary.legend(ch))
    text = path.read_text(encoding="utf-8")
    assert text.startswith("<!DOCTYPE qgis") and 'type="categorizedSymbol"' in text and 'attr="code"' in text
    for code in codes:
        assert f'value="{code}"' in text
    assert "16,16,16" in text and "LinePatternFill" in text
    import xml.etree.ElementTree as ET

    ET.fromstring(text.split("\n", 1)[1])  # well formed


def test_vocabulary_files_are_complete():
    for name in ("iscs.json", "uni11182.json", "masonry.json", "interventions.json"):
        data = json.loads((vocabulary.THESAURUS / name).read_text(encoding="utf-8"))
        assert data["note"] and data["terms"]
        for t in data["terms"]:
            assert {"code", "family", "kind", "geometry", "names", "definition", "colour", "hatch",
                    "source"} <= set(t)
