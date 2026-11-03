"""The EN 16096 assessment of a campaign: the wordings of the condition (CC), urgency (UC) and recommendation (RC)
classes, the plugin's *suggestions* for CC and UC with the rule that gave them (the classes are the surveyor's: the
plugin never fills one), the usual interventions of a pattern (a table of the plugin, in ``interventions.json``), the
lines of the condition report in the order of EN 16096 and the word of a condition class for PyArchInit.

The thresholds of the CC suggestion are the plugin's defaults (:data:`DEFAULTS`), changed per project in
``chunk.meta["conservation"]["thresholds"]``: decay on more than ``cc1_share`` of the face gives CC 2, on more than
``cc2_share`` CC 3; a crack wider than ``crack_cc2_mm`` CC 2, than ``crack_cc3_mm`` CC 3; a loss deeper than
``loss_cc2_cm`` CC 2, than ``loss_cc3_cm`` CC 3; patterns of the detachment family on more than
``detachment_cc3_share`` of the face CC 3; any other symptom CC 1, nothing mapped CC 0."""
from __future__ import annotations

import json
from functools import lru_cache

from .texts import num, t

DEFAULTS = {"cc1_share": 0.10, "cc2_share": 0.30, "crack_cc2_mm": 1.0, "crack_cc3_mm": 5.0,
            "loss_cc2_cm": 2.0, "loss_cc3_cm": 5.0, "detachment_cc3_share": 0.05}

# EN 16096:2012 class wordings (as the standard is quoted; the Italian ones are the plugin's translation)
LABELS = {
    "en": {"cc": ("No symptoms", "Minor symptoms", "Moderately strong symptoms", "Major symptoms"),
           "uc": ("Long term", "Intermediate term", "Short term", "Urgent and immediate"),
           "rc": ("No measures", "Maintenance / preventive conservation",
                  "Moderate repair and/or further investigation", "Major interventions based on diagnosis")},
    "it": {"cc": ("Nessun sintomo", "Sintomi lievi", "Sintomi moderatamente forti", "Sintomi gravi"),
           "uc": ("Lungo termine", "Medio termine", "Breve termine", "Urgente e immediato"),
           "rc": ("Nessuna misura", "Manutenzione / conservazione preventiva",
                  "Riparazione moderata e/o ulteriori indagini", "Interventi importanti basati sulla diagnosi")},
}
CONDITION_WORDS = {"it": ("buono", "discreto", "mediocre", "cattivo"), "en": ("good", "fair", "poor", "bad")}
SUGGESTION = ("suggerimento del plugin, soglie modificabili", "suggestion of the plugin, thresholds can be changed")


def labels(lang: str = "it") -> dict[str, list[str]]:
    """The EN 16096 wordings of the classes 0-3: ``{"cc": [...], "uc": [...], "rc": [...]}``."""
    table = LABELS["en" if lang == "en" else "it"]
    return {k: list(v) for k, v in table.items()}


def class_text(kind: str, n, lang: str = "it") -> str:
    """``CC 1 - Sintomi lievi``; ``-`` when the class is not given."""
    if n is None or n == "":
        return "-"
    return f"{kind.upper()} {int(n)} - {labels(lang)[kind.lower()][int(n)]}"


def active_thresholds(chunk, given: dict | None = None) -> dict:
    """The thresholds of the CC suggestion: the defaults, the project's, then ``given``."""
    out = dict(DEFAULTS)
    own = ((getattr(chunk, "meta", None) or {}).get("conservation") or {}).get("thresholds") or {}
    for source in (own, given or {}):
        for key, value in source.items():
            if key in DEFAULTS and value is not None:
                out[key] = float(value)
    return out


def _pct(share: float) -> str:
    return f"{num(share * 100, 1)} %"


def suggest_cc(chunk, cid: str, thresholds=None) -> tuple[int, str]:
    """The condition class the mapped extents suggest for the campaign ``cid`` and the rule that gave it, in the
    language of the interface (accepted features only; the general terms and a patina are no symptom)."""
    from shapely import wkt
    from shapely.ops import unary_union

    from . import records
    from .vocabulary import deteriorates, load

    th = active_thresholds(chunk, thresholds)
    vocab = load(chunk)
    symptoms, detached = [], []
    for f in records.features(chunk, cid, status="accepted"):
        if f["kind"] not in ("decay", "crack"):
            continue
        term = vocab.term(f["code"]) if f["code"] in vocab else None
        if term is not None and not deteriorates(vocab, term):
            continue
        symptoms.append(f)
        family = term.family if term is not None else None
        if term is not None and term.iscs and term.iscs[0] in vocab:
            family = vocab.term(term.iscs[0]).family
        if f["kind"] == "decay" and family == "detachment":
            detached.append(wkt.loads(f["geometry"]))
    if not symptoms:
        return 0, t("CC 0: nessun sintomo mappato sulla faccia.", "CC 0: no symptom mapped on the face.")
    tot = records.totals(chunk, cid)
    rules = []
    share, face = tot.get("share_of_face"), tot.get("face_area")
    if share is not None:
        if share > th["cc2_share"]:
            rules.append((3, t(f"CC 3: degrado su {_pct(share)} della faccia (oltre {_pct(th['cc2_share'])})",
                               f"CC 3: decay on {_pct(share)} of the face (more than {_pct(th['cc2_share'])})")))
        elif share > th["cc1_share"]:
            rules.append((2, t(f"CC 2: degrado su {_pct(share)} della faccia (fra {_pct(th['cc1_share'])} e "
                               f"{_pct(th['cc2_share'])})", f"CC 2: decay on {_pct(share)} of the face (between "
                               f"{_pct(th['cc1_share'])} and {_pct(th['cc2_share'])})")))
        else:
            rules.append((1, t(f"CC 1: degrado su {_pct(share)} della faccia (fino a {_pct(th['cc1_share'])})",
                               f"CC 1: decay on {_pct(share)} of the face (up to {_pct(th['cc1_share'])})")))
    else:
        rules.append((1, t("CC 1: sintomi mappati (la faccia non ha un rilievo per misurarne la quota)",
                           "CC 1: symptoms mapped (the face has no relief to measure their share)")))
    width = tot.get("crack_max_width_mm")
    if width is not None:
        if width > th["crack_cc3_mm"]:
            rules.append((3, t(f"CC 3: fessura di {num(width, 1)} mm (oltre {num(th['crack_cc3_mm'], 1)} mm)",
                               f"CC 3: crack of {num(width, 1)} mm (wider than {num(th['crack_cc3_mm'], 1)} mm)")))
        elif width > th["crack_cc2_mm"]:
            rules.append((2, t(f"CC 2: fessura di {num(width, 1)} mm (fra {num(th['crack_cc2_mm'], 1)} e "
                               f"{num(th['crack_cc3_mm'], 1)} mm)", f"CC 2: crack of {num(width, 1)} mm (between "
                               f"{num(th['crack_cc2_mm'], 1)} and {num(th['crack_cc3_mm'], 1)} mm)")))
    depth = tot.get("loss_max_depth_mm")
    if depth is not None:
        cm = depth / 10.0
        if cm > th["loss_cc3_cm"]:
            rules.append((3, t(f"CC 3: perdita di materiale profonda {num(cm, 1)} cm (oltre "
                               f"{num(th['loss_cc3_cm'], 1)} cm)",
                               f"CC 3: loss of material {num(cm, 1)} cm deep (deeper than "
                               f"{num(th['loss_cc3_cm'], 1)} cm)")))
        elif cm > th["loss_cc2_cm"]:
            rules.append((2, t(f"CC 2: perdita di materiale profonda {num(cm, 1)} cm (fra "
                               f"{num(th['loss_cc2_cm'], 1)} e {num(th['loss_cc3_cm'], 1)} cm)",
                               f"CC 2: loss of material {num(cm, 1)} cm deep "
                               f"(between {num(th['loss_cc2_cm'], 1)} and {num(th['loss_cc3_cm'], 1)} cm)")))
    if detached and face:
        part = min(1.0, unary_union(detached).area / face)
        if part > th["detachment_cc3_share"]:
            rules.append((3, t(f"CC 3: distacco su {_pct(part)} della faccia (oltre "
                               f"{_pct(th['detachment_cc3_share'])})",
                               f"CC 3: detachment on {_pct(part)} of the face (more than "
                               f"{_pct(th['detachment_cc3_share'])})")))
    cc = max(r[0] for r in rules)
    return cc, "; ".join(text for n, text in rules if n == cc) + "."


def suggest_uc(chunk, cid: str) -> tuple[int | None, str]:
    """The highest urgency given to a feature of the campaign (accepted ones), and how it was found; None when no
    feature has one."""
    from . import records

    given = [int(f["urgency"]) for f in records.features(chunk, cid, status="accepted") if f.get("urgency") is not None]
    if not given:
        return None, t("Nessun elemento ha una classe di urgenza: UC si sceglie dalla valutazione del rischio.",
                       "No feature has an urgency class: UC is chosen from the risk assessment.")
    uc = max(given)
    n = sum(1 for g in given if g == uc)
    return uc, t(f"UC {uc}: la più alta fra le urgenze date agli elementi ({n} con UC {uc}).",
                 f"UC {uc}: the highest of the urgencies given to the features ({n} with UC {uc}).")


@lru_cache(maxsize=1)
def _usual_table() -> dict:
    from .vocabulary import THESAURUS

    data = json.loads((THESAURUS / "interventions.json").read_text(encoding="utf-8"))
    return data.get("usual_treatments") or {"by_code": {}, "by_family": {}}


def usual_treatments(code: str, chunk=None) -> list[str]:
    """The intervention codes usually applied to the pattern ``code`` (the term, else its parent, else its first
    ICOMOS-ISCS code, else its family); a suggestion, printed as such."""
    from .vocabulary import load

    vocab = load(chunk)
    table = _usual_table()
    term = vocab.term(code)
    seen, todo = set(), [term]
    while todo:
        tm = todo.pop(0)
        if tm.code in seen:
            continue
        seen.add(tm.code)
        if tm.code in table["by_code"]:
            return list(table["by_code"][tm.code])
        for nxt in ([tm.parent] if tm.parent else []) + list(tm.iscs or []):
            if nxt in vocab:
                todo.append(vocab.term(nxt))
    family = term.family
    if term.iscs and term.iscs[0] in vocab:
        family = vocab.term(term.iscs[0]).family
    return list(table["by_family"].get(family, []))


def condition_word(cc, lang: str = "it", thesaurus=None) -> str:
    """The word of a condition class for a form (``stato_di_conservazione``): buono / discreto / mediocre / cattivo
    for CC 0-3; with ``thesaurus`` (the database's values, strings or ``(sigla, value)`` pairs) the value that is
    that word, else the database's value of the same rank when it has four of them, else the plugin's word."""
    n = int(cc)
    word = CONDITION_WORDS["en" if lang == "en" else "it"][n]
    if thesaurus:
        values = [v[1] if isinstance(v, (tuple, list)) else v for v in thesaurus]
        values = [str(v) for v in values if v not in (None, "")]
        for v in values:
            if v.strip().casefold() == word.casefold():
                return v
        if len(values) == 4:
            return values[n]
    return word


def report_lines(chunk, cid: str, lang: str | None = None) -> list[str]:
    """The text of the condition report of a campaign in the order of EN 16096: object, sources, survey, the component
    (the face) with its symptoms and classes, risk and recommendation, summary."""
    from . import records, texts
    from .vocabulary import load

    lang = lang or texts.lang()

    def tr(it, en):
        return en if lang == "en" else it

    def n(value, d=2):
        return num(value, d)

    c = records.campaign(chunk, cid)
    entry = next((e for e in chunk.meta.get("elevations") or [] if e.get("id") == c["elevation"]), {})
    vocab = load(chunk)
    tot = records.totals(chunk, cid)
    feats = records.features(chunk, cid, status="accepted")
    classes = c.get("classes") or {}
    rc = records.assessment(chunk, c["date"])
    doc = getattr(chunk, "document", None)
    site = str(((getattr(doc, "meta", None) or {}).get("sito")) or "") if doc is not None else ""
    lines = [tr("1. Oggetto", "1. Object"),
             tr(f"Chunk: {chunk.label}", f"Chunk: {chunk.label}")]
    if site:
        lines.append(tr(f"Sito: {site}", f"Site: {site}"))
    lines.append(tr(f"Componente (faccia): {entry.get('name', c['elevation'])}",
                    f"Component (face): {entry.get('name', c['elevation'])}"))
    if tot.get("face_area"):
        lines.append(tr(f"Superficie della faccia: {n(tot['face_area'])} m²",
                        f"Surface of the face: {n(tot['face_area'])} m²"))
    lines += ["", tr("2. Fonti", "2. Sources")]
    if entry.get("resolution"):
        lines.append(tr(f"Prospetto fotogrammetrico a {n(entry['resolution'] * 1000, 1)} mm/px",
                        f"Photogrammetric elevation at {n(entry['resolution'] * 1000, 1)} mm/px"))
    lines.append(tr("Lessico: ICOMOS-ISCS 2008, UNI 11182:2006; classi: EN 16096:2012",
                    "Vocabulary: ICOMOS-ISCS 2008, UNI 11182:2006; classes: EN 16096:2012"))
    lines += ["", tr("3. Rilievo", "3. Survey"),
              tr(f"Campagna: {c['name']} del {c['date']}", f"Campaign: {c['name']} of {c['date']}")]
    for key, it, en in (("surveyor", "Rilevatore", "Surveyor"), ("method", "Metodo", "Method"),
                        ("weather", "Condizioni meteo", "Weather"), ("notes", "Note", "Notes")):
        if c.get(key):
            lines.append(f"{tr(it, en)}: {c[key]}")
    lines += ["", tr("4. Componente, sintomi e classi", "4. Component, symptoms and classes")]
    by_code = {}
    for f in feats:
        if f["kind"] in ("decay", "crack"):
            by_code.setdefault(f["code"], []).append(f)
    if not by_code:
        lines.append(tr("Nessun sintomo mappato.", "No symptom mapped."))
    for code, group in by_code.items():
        name = vocab.name(code, lang) if code in vocab else code
        if group[0]["kind"] == "crack":
            length = sum(float(f["extent"] or 0) for f in group)
            widths = [f["width_mm"] for f in group if f.get("width_mm") is not None]
            extra = tr(f", apertura fino a {n(max(widths), 1)} mm", f", opening up to {n(max(widths), 1)} mm") \
                if widths else ""
            lines.append(f"- {name}: {len(group)} × {n(length)} m{extra}")
        else:
            area = sum(float(f["extent"] or 0) for f in group)
            lines.append(f"- {name}: {len(group)} × {n(area, 3)} m²")
    if tot.get("share_of_face") is not None:
        lines.append(tr(f"Quota della faccia con degrado: {_pct(tot['share_of_face'])}",
                        f"Share of the face with decay: {_pct(tot['share_of_face'])}"))
    lines.append(tr("Classe di condizione: ", "Condition class: ") + class_text("cc", classes.get("cc"), lang))
    if classes.get("reason"):
        lines.append(tr(f"Motivo: {classes['reason']}", f"Reason: {classes['reason']}"))
    lines += ["", tr("5. Rischio e raccomandazioni", "5. Risk and recommendations"),
              tr("Classe di urgenza: ", "Urgency class: ") + class_text("uc", classes.get("uc"), lang),
              tr("Classe di raccomandazione dell'oggetto: ", "Recommendation class of the object: ")
              + class_text("rc", rc.get("rc") if rc else None, lang)]
    if rc and rc.get("reason"):
        lines.append(tr(f"Motivo: {rc['reason']}", f"Reason: {rc['reason']}"))
    treatments = [f for f in feats if f["kind"] == "treatment"]
    if treatments:
        done = sum(1 for f in treatments if (f.get("treatment") or {}).get("done"))
        lines.append(tr(f"Interventi mappati: {len(treatments)} ({done} eseguiti)",
                        f"Interventions mapped: {len(treatments)} ({done} done)"))
    def cls(value):
        return "-" if value is None else str(value)

    summary = f"CC {cls(classes.get('cc'))}, UC {cls(classes.get('uc'))}, RC {cls(rc.get('rc') if rc else None)}"
    lines += ["", tr("6. Sintesi", "6. Summary"),
              tr(f"{len(feats)} elementi; {summary}", f"{len(feats)} features; {summary}")]
    return lines
