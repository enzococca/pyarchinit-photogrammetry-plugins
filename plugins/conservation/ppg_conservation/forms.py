"""The plugin's forms: a campaign, a feature, a treatment, the EN 16096 classes and recommendation.

Each function asks through ``plugin._ask`` (a :class:`ParamDialog`, or the feature form with its photos) and returns
the values to write, or None when the form is cancelled. Nothing is written here."""
from __future__ import annotations

import dataclasses
import math

from . import texts
from .texts import num, t

MEMORY = 12  # the materials and methods remembered
INTENSITIES = (("low", ("bassa", "low")), ("medium", ("media", "medium")), ("high", ("alta", "high")))
NUMBERS = {"width_mm": ("Apertura (mm)", "Opening (mm)"), "depth_mm": ("Profondità (mm)", "Depth (mm)")}


def number(value, positive: bool = False) -> float | None:
    """A number typed in a form (``3,5`` or ``3.5``; empty = None); ValueError when it is no number, not finite, or
    negative where ``positive`` (a measure: an opening, a depth, a pixel)."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        out = float(value)
    else:
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        out = float(text)
    if not math.isfinite(out):
        raise ValueError(t(f"{value!r} non è un numero finito", f"{value!r} is not a finite number"))
    if positive and out < 0:
        raise ValueError(t(f"{value!r} non può essere negativo", f"{value!r} cannot be negative"))
    return out


def _text(value) -> str:
    """A number as the form shows it to be typed again: no thousands, the decimal comma in Italian."""
    if value is None:
        return ""
    text = f"{float(value):g}"
    return text.replace(".", ",") if texts.lang() == "it" else text


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
    from .vocabulary import UNNAMED, load

    lang = texts.lang()
    vocab = load(chunk)
    term = vocab.term(rec["code"])
    c = records.campaign(chunk, rec["campaign"])
    entry = plugin._entry(chunk, c["elevation"])
    if rec["kind"] == "note":
        same = [tm for tm in vocab.terms() if tm.kind != "treatment"]
    else:
        same = [tm for tm in vocab.terms() if tm.kind == rec["kind"]]
    same = [tm for tm in same if tm.code != UNNAMED or tm.code == rec["code"]]  # the working term only while carried
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
                    v[name] = number(v[name], positive=True)
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


DETECTOR_TITLES = {"cracks": ("Lesioni (valli del rilievo e linee scure)", "Cracks (valleys of the relief and dark "
                              "lines)"),
                   "loss": ("Mancanze (depressioni sotto il muro intorno)", "Losses (depressions behind the wall "
                            "around them)"),
                   "colours": ("Classi di colore (le nomini tu)", "Colour classes (you name them)"),
                   "sam": ("Aree di SAM 2.1 (da nominare)", "SAM 2.1 areas (to be named)"),
                   "like": ("Aree di SAM 2.1 come l'area scelta (Come questo)", "SAM 2.1 areas like the chosen area "
                            "(Like this)")}


def proposals_form(plugin, selected: dict | None, term_name: str | None) -> dict | None:
    """The detector, the window (the area chosen in the table, a window drawn on the view or the whole face) and
    the parameters of the proposals. Returns ``{detector, window, use_term, thoroughness, resolution_mm,
    min_length_cm, max_width_mm, min_depth_cm, min_area_m2, k, radius}`` or None (``use_term`` is never
    remembered: off at every opening)."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    s = plugin.context.settings.get("proposals") or {}
    windows = [("face", t("tutto il prospetto", "the whole face")),
               ("draw", t("disegna una finestra", "draw a window"))]
    if selected is not None:
        windows.insert(0, ("selected", t("l'area scelta nella tabella", "the area chosen in the table")))
    detectors = [(k, t(*v)) for k, v in DETECTOR_TITLES.items()]
    th = [(x, x) for x in ("normale", "alta", "massima")]
    fields = [Field("detector", t("Cosa proporre", "What to propose"), s.get("detector", "cracks"),
                    choices=detectors),
              Field("window", t("Finestra", "Window"), windows[0][0], choices=windows),
              Field("h_relief", t("Lesioni e mancanze", "Cracks and losses"), kind="heading"),
              Field("min_length_cm", t("Lunghezza minima di una lesione (cm)", "Least length of a crack (cm)"),
                    float(s.get("min_length_cm", 10.0)), minimum=0.5, maximum=1000.0, decimals=1),
              Field("max_width_mm", t("Apertura massima di una lesione (mm)", "Largest opening of a crack (mm)"),
                    float(s.get("max_width_mm", 12.0)), minimum=0.0, maximum=1000.0, decimals=1,
                    tip=t("Le linee più larghe sono giunti fra le pietre, non lesioni: tolte e contate (0: nessun "
                          "limite)", "Wider lines are joints between the stones, not cracks: dropped and counted "
                          "(0: no limit)")),
              Field("min_depth_cm", t("Profondità minima di una mancanza (cm)", "Least depth of a loss (cm)"),
                    float(s.get("min_depth_cm", 2.0)), minimum=0.1, maximum=100.0, decimals=1),
              Field("min_area_m2", t("Area minima di una mancanza (m²)", "Least area of a loss (m²)"),
                    float(s.get("min_area_m2", 0.0025)), minimum=0.0001, maximum=100.0, decimals=4,
                    tip=t("0,0025 m² = 5 x 5 cm, perché si vedano anche gli alveoli; il rilevatore da solo usa "
                          "0,01 m²", "0.0025 m² = 5 x 5 cm, so that alveoli show too; the detector by itself uses "
                          "0.01 m²")),
              Field("h_image", t("Colori e SAM", "Colours and SAM"), kind="heading"),
              Field("k", t("Classi di colore", "Colour classes"), int(s.get("k", 5)), minimum=2, maximum=12),
              Field("thoroughness", t("Accuratezza di SAM", "Thoroughness of SAM"), s.get("thoroughness", "normale"),
                    choices=th, tip=t("Sulle facce oltre 4 m² SAM lavora a 5 mm per pixel, salvo una risoluzione "
                                      "data", "On faces over 4 m² SAM works at 5 mm per pixel unless a "
                                      "resolution is given")),
              Field("use_term", t("Usa il termine scelto per tutti i segmenti di SAM", "Use the chosen term for "
                                  "all of SAM's segments"), False,
                    tip=t("Spento: i segmenti restano da nominare (Modifica..., poi Come questo)",
                          "Off: the segments stay to be named (Edit..., then Like this)")),
              Field("resolution_mm", t("Pixel di SAM (mm, vuoto: automatico)", "SAM's pixel (mm, empty: "
                                       "automatic)"), s.get("resolution_mm"), kind="optional_float"),
              Field("radius", t("Come questo: distanza massima (in dispersioni)", "Like this: largest distance "
                                "(in spreads)"), float(s.get("radius", 1.0)), minimum=0.1, maximum=10.0,
                    decimals=2)]
    chosen = t(f"Termine scelto nell'elenco: {term_name}.", f"Term chosen in the list: {term_name}.") if term_name \
        else t("Nessun termine scelto nell'elenco.", "No term chosen in the list.")
    intro = t("Le proposte non entrano nella campagna finché non le accetti. Le aree di SAM e le mancanze senza "
              "termine restano \"da nominare\": dai un termine a una con Modifica... e Come questo lo dà alle "
              "simili. ", "The proposals do not join the campaign until you accept them. SAM's areas and the losses "
              "with no term stay \"to be named\": give one a term with Edit... and Like this gives it to the "
              "similar ones. ") + chosen
    v = plugin._ask(t("Proposte", "Proposals"), fields, intro=intro)
    if v is None:
        return None
    keep = {k: v.get(k) for k in ("detector", "min_length_cm", "max_width_mm", "min_depth_cm", "min_area_m2", "k",
                                  "thoroughness", "resolution_mm", "radius")}
    plugin.context.settings["proposals"] = keep
    plugin.context.save_settings()
    return {**keep, "window": v.get("window"), "use_term": bool(v.get("use_term"))}


def colour_classes_form(plugin, chunk, classes: list[dict]) -> dict | None:
    """The user names each colour class with a term (or none: the class is left out); the plugin puts first the
    terms it suggests for green, bright and dark classes, never chosen by default. Returns ``{class: code}`` of the
    classes named, or None."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    from . import texts
    from .vocabulary import UNNAMED, load

    vocab = load(chunk)
    lang = texts.lang()
    decay = [term for term in vocab.terms("decay") if term.code != UNNAMED]
    looks = {"green": ("verde", "green"), "bright": ("chiara", "bright"), "dark": ("scura", "dark")}
    fields = []
    for c in classes:
        suggested = [code for code in c.get("suggest") or [] if _known(vocab, code)]
        rest = sorted((term.code for term in decay if term.code not in suggested), key=lambda x: vocab.name(x, lang))
        choices = [("", t("(nessuno: lasciala fuori)", "(none: leave it out)"))] + \
            [(code, ("* " if code in suggested else "") + vocab.name(code, lang)) for code in suggested + rest]
        L, a, b = c["lab_mean"]
        what = t(*looks[c["looks"]]) if c.get("looks") in looks else t("altro", "other")
        label = t(f"Classe {c['class'] + 1}: {what}, {num(c['share'] * 100, 1)} % (L {num(L, 0)}, a {num(a, 0)}, "
                  f"b {num(b, 0)})", f"Class {c['class'] + 1}: {what}, {num(c['share'] * 100, 1)} % (L {num(L, 0)}, "
                  f"a {num(a, 0)}, b {num(b, 0)})")
        fields.append(Field(f"class_{c['class']}", label, "", choices=choices))
    intro = t("Dai un termine alle classi che vuoi proporre; con * i termini che il plugin suggerisce per il colore "
              "(verde: colonizzazione biologica; chiaro: efflorescenza o deposito; scuro: crosta nera o macchia). "
              "Le classi senza termine restano fuori.",
              "Give a term to the classes you want to propose; marked * the terms the plugin suggests for the colour "
              "(green: biological colonization; bright: efflorescence or deposit; dark: black crust or staining). "
              "The classes with no term are left out.")
    v = plugin._ask(t("Classi di colore", "Colour classes"), fields, intro=intro)
    if v is None:
        return None
    return {c["class"]: v.get(f"class_{c['class']}") for c in classes if v.get(f"class_{c['class']}")}


def _known(vocab, code: str) -> bool:
    try:
        vocab.term(code)
        return True
    except KeyError:
        return False
