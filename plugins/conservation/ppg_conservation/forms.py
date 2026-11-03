"""The plugin's forms: a campaign, a feature, a treatment, the EN 16096 classes and recommendation.

Each function asks through ``plugin._ask`` (a :class:`ParamDialog`, or the feature form with its photos) and returns
the values to write, or None when the form is cancelled. Nothing is written here."""
from __future__ import annotations

import dataclasses

from .texts import num, t

MEMORY = 12  # the materials and methods remembered
INTENSITIES = (("low", ("bassa", "low")), ("medium", ("media", "medium")), ("high", ("alta", "high")))
NUMBERS = {"width_mm": ("Apertura (mm)", "Opening (mm)"), "depth_mm": ("Profondità (mm)", "Depth (mm)")}


def number(value) -> float | None:
    """A number typed in a form (``3,5`` or ``3.5``; empty = None); ValueError when it is no number."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    text = str(value).strip().replace(",", ".")
    return float(text) if text else None


def _text(value) -> str:
    return "" if value is None else f"{value:g}"


def campaign_fields(plugin, chunk, c: dict | None = None) -> list:
    from pyarchinit_photogrammetry.gui.dialogs import Field

    if c is None:
        from datetime import date

        from pyarchinit_photogrammetry.archaeo.timeline import phase_date

        when = phase_date(chunk)
        day = when.date().isoformat() if when is not None else date.today().isoformat()
        c = {"date": day, "name": "", "surveyor": plugin.context.settings.get("surveyor", ""),
             "method": "", "weather": "", "notes": ""}
    methods = [(m, m) for m in plugin._settings_list("methods")]
    return [Field("date", t("Data (AAAA-MM-GG)", "Date (YYYY-MM-DD)"), c["date"], kind="text"),
            Field("name", t("Nome (vuoto: Campagna e la data)", "Name (empty: Campaign and the date)"), c["name"],
                  kind="text"),
            Field("surveyor", t("Rilevatore", "Surveyor"), c.get("surveyor", ""), kind="text"),
            Field("method", t("Metodo", "Method"), c.get("method", ""), choices=methods, kind="combo_text",
                  tip=t("Per esempio: rilievo fotogrammetrico da drone, ispezione diretta",
                        "For instance: photogrammetric survey by drone, direct inspection")),
            Field("weather", t("Condizioni meteo", "Weather"), c.get("weather", ""), kind="text"),
            Field("notes", t("Note", "Notes"), c.get("notes", ""), kind="text")]


def remember_campaign(plugin, v: dict) -> None:
    if v.get("surveyor"):
        plugin.context.settings["surveyor"] = v["surveyor"]
        plugin.context.save_settings()
    plugin._remember("methods", (v.get("method") or "").strip(), MEMORY)


def feature_form(plugin, chunk, rec: dict) -> dict | None:
    """The form of a feature: term (of the same kind), intensity, opening and depth, cause, urgency, notes, the
    photos that see it. A number that cannot be read asks the form again with what was typed. Returns the fields for
    ``update_feature`` or None."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import assess, records, texts
    from .vocabulary import load

    lang = texts.lang()
    vocab = load(chunk)
    term = vocab.term(rec["code"])
    c = records.campaign(chunk, rec["campaign"])
    entry = plugin._entry(chunk, c["elevation"])
    if rec["kind"] == "note":
        same = [tm for tm in vocab.terms() if tm.kind != "treatment"]
    else:
        same = [tm for tm in vocab.terms() if tm.kind == rec["kind"]]
    terms = [(tm.code, vocab.name(tm.code, lang)) for tm in same]
    fields = [Field("code", t("Termine", "Term"), rec["code"], choices=terms),
              Field("intensity", t("Intensità", "Intensity"), rec.get("intensity"),
                    choices=[(None, "-")] + [(k, t(*v)) for k, v in INTENSITIES])]
    if rec["kind"] == "crack":
        fields.append(Field("width_mm", t(*NUMBERS["width_mm"]), _text(rec.get("width_mm")), kind="text"))
    family = vocab.term(term.iscs[0]).family if term.iscs and term.iscs[0] in vocab else term.family
    if rec["kind"] == "decay" and family == "material_loss":
        fields.append(Field("depth_mm", t(*NUMBERS["depth_mm"]), _text(rec.get("depth_mm")), kind="text"))
    urgencies = [(None, "-")] + [(n, assess.class_text("uc", n, lang)) for n in range(4)]
    fields += [Field("cause", t("Causa", "Cause"), rec.get("cause") or "", kind="text"),
               Field("urgency", t("Urgenza (EN 16096)", "Urgency (EN 16096)"), rec.get("urgency"),
                     choices=urgencies),
               Field("notes", t("Note", "Notes"), rec.get("notes") or "", kind="text")]
    photos = plugin._photos(chunk, entry, rec)
    if photos:
        kept = [p["key"] for p in rec.get("photos") or []] or [p["key"] for p in photos[:2]]
        fields.append(Field("photos", t("Foto tenute", "Photos kept"), kept, kind="checklist",
                            choices=[(p["key"], f"{p['label']} ({num(p['distance'], 1)} m, "
                                                f"{num(p['angle'], 0)}°)") for p in photos]))
    if rec["extent"] is None:
        extent = ""
    elif rec["geometry"].startswith(("POLYGON", "MULTIPOLYGON")):
        extent = f" - {num(rec['extent'], 3)} m²"
    else:
        extent = f" - {num(rec['extent'], 2)} m"
    intro = f"{vocab.name(rec['code'], lang)}{extent}\n{term.definition.get(lang, '')}"
    warning = ""
    while True:
        v = plugin._ask(t("Elemento", "Feature"), fields, intro=warning + intro, photos=photos, rec=rec)
        if v is None:
            return None
        bad = []
        for name in NUMBERS:
            if name in v:
                try:
                    v[name] = number(v[name])
                except ValueError:
                    bad.append(t(*NUMBERS[name]))
        if not bad:
            break
        # asked again with what was typed: the drawing is not lost
        fields = [dataclasses.replace(f, default=v[f.name]) if f.name in v else f for f in fields]
        warning = t(f"Non è un numero: {', '.join(bad)}. Correggi e conferma.\n\n",
                    f"Not a number: {', '.join(bad)}. Correct it and confirm.\n\n")
    out = {k: v[k] for k in ("code", "intensity", "cause", "urgency", "notes", "width_mm", "depth_mm") if k in v}
    if "photos" in v:
        keep = set(v["photos"] or [])
        out["photos"] = [{"key": p["key"], "label": p["label"], "px": p["px"], "py": p["py"]}
                         for p in photos if p["key"] in keep]
    return out


def treatment_form(plugin, chunk, c: dict, treatment: dict, code: str | None, rec: dict | None = None,
                   intro: str = "") -> dict | None:
    """The form of a treatment: technique (the interventions' vocabulary), materials (with a memory of those
    typed), date, done or to do, by whom, notes. Returns the fields for ``add_feature``/``update_feature``."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import texts
    from .vocabulary import load

    lang = texts.lang()
    vocab = load(chunk)
    techniques = [(tm.code, vocab.name(tm.code, lang)) for tm in vocab.interventions()]
    default = code or plugin.context.settings.get("technique") or techniques[0][0]
    materials = [(m, m) for m in plugin._settings_list("materials")]
    fields = [Field("technique", t("Tecnica (EN 15898)", "Technique (EN 15898)"), default, choices=techniques),
              Field("materials", t("Materiali e prodotti", "Materials and products"),
                    treatment.get("materials", ""), choices=materials, kind="combo_text"),
              Field("date", t("Data (AAAA-MM-GG)", "Date (YYYY-MM-DD)"), treatment.get("date") or c["date"],
                    kind="text"),
              Field("done", t("Eseguito (altrimenti da eseguire)", "Done (else to do)"),
                    bool(treatment.get("done", False))),
              Field("by", t("Eseguito da", "Done by"), treatment.get("by") or c.get("surveyor", ""), kind="text"),
              Field("notes", t("Note", "Notes"), (rec or {}).get("notes") or "", kind="text")]
    v = plugin._ask(t("Intervento", "Treatment"), fields, intro=intro)
    if v is None:
        return None
    materials_text = (v.get("materials") or "").strip()
    plugin._remember("materials", materials_text, MEMORY)
    plugin.context.settings["technique"] = v["technique"]
    plugin.context.save_settings()
    return {"code": v["technique"], "notes": v.get("notes") or "",
            "treatment": {"technique": v["technique"], "materials": materials_text,
                          "date": (v.get("date") or "").strip(), "done": bool(v.get("done")),
                          "by": (v.get("by") or "").strip()}}


def classes_form(plugin, chunk, c: dict) -> dict | None:
    """The EN 16096 condition and urgency classes of a campaign, the plugin's suggestion and its rule in the intro;
    the class is the surveyor's. Returns ``{cc, uc, reason}`` or None."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import assess, texts

    lang = texts.lang()
    cc, rule = assess.suggest_cc(chunk, c["id"])
    _uc, urule = assess.suggest_uc(chunk, c["id"])
    th = assess.active_thresholds(chunk)
    limits = t(f"Soglie: degrado {num(th['cc1_share'] * 100, 0)} % / {num(th['cc2_share'] * 100, 0)} % della "
               f"faccia, fessure {num(th['crack_cc2_mm'], 1)} / {num(th['crack_cc3_mm'], 1)} mm, perdite "
               f"{num(th['loss_cc2_cm'], 1)} / {num(th['loss_cc3_cm'], 1)} cm, distacco "
               f"{num(th['detachment_cc3_share'] * 100, 0)} %.",
               f"Thresholds: decay {num(th['cc1_share'] * 100, 0)} % / {num(th['cc2_share'] * 100, 0)} % of the "
               f"face, cracks {num(th['crack_cc2_mm'], 1)} / {num(th['crack_cc3_mm'], 1)} mm, losses "
               f"{num(th['loss_cc2_cm'], 1)} / {num(th['loss_cc3_cm'], 1)} cm, detachment "
               f"{num(th['detachment_cc3_share'] * 100, 0)} %.")
    intro = t(f"Le classi le sceglie chi rileva (EN 16096). Il plugin suggerisce ({assess.SUGGESTION[0]}):\n"
              f"condizione {assess.class_text('cc', cc, lang)}: {rule}\nurgenza: {urule}\n{limits}",
              f"The classes are the surveyor's choice (EN 16096). The plugin suggests ({assess.SUGGESTION[1]}):\n"
              f"condition {assess.class_text('cc', cc, lang)}: {rule}\nurgency: {urule}\n{limits}")
    classes = c.get("classes") or {}
    fields = [Field("cc", t("Classe di condizione (CC)", "Condition class (CC)"), classes.get("cc"),
                    choices=[(None, "-")] + [(n, assess.class_text("cc", n, lang)) for n in range(4)]),
              Field("uc", t("Classe di urgenza (UC)", "Urgency class (UC)"), classes.get("uc"),
                    choices=[(None, "-")] + [(n, assess.class_text("uc", n, lang)) for n in range(4)]),
              Field("reason", t("Motivo", "Reason"), classes.get("reason") or "", kind="text")]
    v = plugin._ask(t("Classi EN 16096", "EN 16096 classes"), fields, intro=intro)
    return None if v is None else {"cc": v["cc"], "uc": v["uc"], "reason": v["reason"]}


def assessment_form(plugin, chunk, c: dict) -> dict | None:
    """The EN 16096 recommendation class of the object surveyed on the campaign's date (not suggested). Returns
    ``{rc, reason, surveyor}`` or None."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import assess, records, texts

    lang = texts.lang()
    rec = records.assessment(chunk, c["date"]) or {}
    fields = [Field("rc", t("Classe di raccomandazione (RC)", "Recommendation class (RC)"), rec.get("rc"),
                    choices=[(None, "-")] + [(n, assess.class_text("rc", n, lang)) for n in range(4)]),
              Field("reason", t("Motivo", "Reason"), rec.get("reason") or "", kind="text"),
              Field("surveyor", t("Rilevatore", "Surveyor"), rec.get("surveyor") or c.get("surveyor", ""),
                    kind="text")]
    intro = t(f"La raccomandazione vale per tutto l'oggetto rilevato il {c['date']}: una per data, mostrata "
              "in ogni campagna di quella data. Il plugin non la suggerisce: pesano di più le parti portanti e "
              "quelle più significative.",
              f"The recommendation is for the whole object surveyed on {c['date']}: one per date, shown in "
              "every campaign of that date. The plugin does not suggest it: load-bearing and more significant "
              "parts weigh more.")
    v = plugin._ask(t("Raccomandazione EN 16096", "EN 16096 recommendation"), fields, intro=intro)
    return None if v is None else {"rc": v["rc"], "reason": v["reason"], "surveyor": v["surveyor"]}
