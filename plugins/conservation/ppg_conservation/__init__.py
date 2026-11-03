"""Conservation (Conservazione) - a plugin of pyarchinit-photogrammetry: decay mapping on the program's elevations.

* the vocabulary of the standards (:mod:`.vocabulary`): ICOMOS-ISCS 2008 terms as the codes, UNI 11182:2006 as their
  Italian face, the plugin's local masonry terms, the interventions after EN 15898; colours and hatches the plugin's,
  changed per project;
* campaigns and features on a face (:mod:`.records`), kept in ``chunk.meta["conservation"]``; the EN 16096 classes
  chosen by the surveyor, with the plugin's suggestions (:mod:`.assess`); the world geometry of a feature lifted on the
  relief (:mod:`.geometry`);
* the *Conservazione* pane (on the right, :mod:`.pane`): campaigns, mapping with the glossaries, interventions,
  assessment; the features drawn on the elevation as layers (not kept by the program: drawn again whenever the view
  loads the elevation), deleted with the window's *Seleziona* + Canc; every change one entry of *Modifica > Annulla*;
* ``ppg plugin conservation list``, the plate of a campaign among the exports.

The plugin writes only ``chunk.meta["conservation"]`` and the files under ``<chunk data>/conservation/<eid>/``.
"""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, num, t

__version__ = "0.1.0"

from .layers import FILL_ALPHA, LAYERS, PROPOSED_ALPHA  # noqa: F401 - the plugin's layers, named here too
from .vocabulary import UNNAMED  # noqa: E402 - the working term of a proposal not named yet

RECENT = 10  # the last terms used, at the top of the chooser


class _Step:
    """One change of the plugin's records as an entry of the window's *Modifica > Annulla*: the records before and
    after (the program's undo takes no snapshot of a plugin's meta by itself)."""

    def __init__(self, plugin, chunk, before, after):
        self.plugin, self.chunk, self.before, self.after = plugin, chunk, before, after

    def _put(self, state) -> None:
        if state is None:
            self.chunk.meta.pop("conservation", None)
        else:
            self.chunk.meta["conservation"] = copy.deepcopy(state)
        self.plugin.context.mark_modified()
        self.plugin.context.refresh_tree()
        self.plugin.refresh_pane()

    def undo(self) -> None:
        self._put(self.before)

    def redo(self) -> None:
        self._put(self.after)


def _parts(geom, kind: str):
    """The parts of ``geom`` (a proposal clipped to its window) a feature of ``kind`` can have: polygons for an area,
    lines for a crack; None when nothing of them is left."""
    import shapely
    from shapely.geometry import MultiLineString, MultiPolygon

    want = "Polygon" if kind == "decay" else "LineString"
    flat = [p for g in shapely.get_parts(geom) for p in shapely.get_parts(g)]  # a collection may hold multi-parts
    parts = [p for p in flat if p.geom_type == want and not p.is_empty]
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return MultiPolygon(parts) if kind == "decay" else MultiLineString(parts)


class ConservationPlugin(Plugin):
    name = "conservation"
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Mappatura del degrado sui prospetti: lessico ICOMOS-ISCS e UNI 11182, campagne, "
                             "classi EN 16096, interventi", "Decay mapping on elevations: ICOMOS-ISCS and UNI 11182 "
                             "vocabulary, campaigns, EN 16096 classes, interventions")
        self.context = None
        self.pane = None
        self.drawn: tuple[str, str] | None = None  # (eid, cid) whose features are on the view
        self._terms_key: str | None = None  # what the term chooser was filled with

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Conservazione...", "Conservation..."), self.show_pane,
                           tip=t("Il pannello della conservazione", "The conservation pane"))
        context.add_command("conservation", self.command,
                            help="decay mapping on elevations: list the campaigns and their features")
        context.add_exporter(t("Tavola della campagna (PDF)", "Plate of the campaign (PDF)"), ".pdf",
                             self._export_plate)
        if context.window is not None:
            from .pane import ConservationPane

            self.pane = ConservationPane()
            self.pane.setObjectName("conservation_pane")
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            self.pane.requested.connect(self._requested)
            self.pane.elevation_changed.connect(lambda _e: self.refresh_pane())
            self.pane.campaign_changed.connect(lambda _c: self.refresh_pane())
            self.pane.search.returnPressed.connect(self._first_found)
            context.on_refresh(self.refresh_pane)
            context.on_raster_loaded(self._raster_loaded)
            for layer in LAYERS.values():
                context.on_features_deleted(layer, self._on_deleted)
            self.refresh_pane()

    def deactivate(self) -> None:
        self._drop_layers()
        self.pane = None

    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    # ------------------------------------------------------------------ helpers
    def _chunk(self):
        window = self.context.window if self.context else None
        chunk = getattr(window, "chunk", None) if window is not None else None
        if chunk is not None:
            return chunk
        doc = self.context.app.document if self.context else None
        return None if doc is None else doc.chunk

    def _need_chunk(self):
        from .records import NotReady

        chunk = self._chunk()
        if chunk is None:
            raise NotReady(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _message(self, kind: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text)

    def _confirm(self, text: str) -> bool:
        from pyarchinit_photogrammetry.gui import dialogs

        return dialogs.choice(self.context.window, "question", self.title, text, [t("Rimuovi", "Remove")]) == 0

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        from .records import NotReady

        try:
            function(*args)
        except (NotReady, ValueError, RuntimeError, KeyError) as exc:
            self._message("warning", str(exc.args[0]) if isinstance(exc, KeyError) and exc.args else str(exc))

    def _ask(self, title: str, fields: list, intro: str = "", photos=None, rec: dict | None = None) -> dict | None:
        """A form; with ``photos`` the feature form, whose *apri* opens the photo at the feature ``rec``."""
        if photos:
            from . import pane

            return pane.feature_dialog(self.context.window, title, fields, intro, photos,
                                       lambda photo: self._open_photo(photo, rec))
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        return ParamDialog.ask(self.context.window, title, fields, intro=intro)

    @staticmethod
    def _snapshot(chunk):
        return copy.deepcopy(chunk.meta.get("conservation"))

    def _changed(self, chunk, before, text: str) -> None:
        """After a change of the records: an entry of *Modifica > Annulla* (``text``), the project modified, the
        workspace and the pane shown again, the layers drawn again."""
        after = self._snapshot(chunk)
        window = self.context.window
        history = getattr(window, "history", None) if window is not None else None
        scripted = bool(getattr(window, "_script_running", False))  # a script's own entry covers it
        if history is not None and not scripted and before != after and chunk is getattr(window, "chunk", None):
            history.push(t(f"Conservazione: {text}", f"Conservation: {text}"), _Step(self, chunk, before, after),
                         document=False)
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane()

    def _settings_list(self, key: str) -> list:
        value = self.context.settings.get(key)
        return list(value) if isinstance(value, list) else []

    def _remember(self, key: str, value, size: int) -> None:
        if value in (None, ""):
            return
        items = [x for x in self._settings_list(key) if x != value]
        self.context.settings[key] = [value] + items[:size - 1]
        self.context.save_settings()

    # ------------------------------------------------------------------ elevations, campaigns, the view
    @staticmethod
    def _elevations(chunk) -> list[dict]:
        return list(chunk.meta.get("elevations") or []) if chunk is not None else []

    def _entry(self, chunk, eid: str) -> dict:
        from .records import NotReady

        for e in self._elevations(chunk):
            if e.get("id") == eid:
                return e
        raise NotReady(t("Il prospetto non esiste più in questo chunk.", "The elevation is no longer in this chunk."))

    @staticmethod
    def _elevation_label(chunk, entry: dict) -> str:
        """The name of an elevation, after the masonry wall it is a face of."""
        name = entry.get("name") or entry["id"]
        for wall in ((chunk.meta.get("masonry") or {}).get("walls") or []):
            if entry["id"] in (wall.get("faces") or []):
                return f"{wall.get('name', '')} - {name}"
        if entry.get("kind") == "section":
            return f"{name} ({t('sezione', 'section')})"
        return name

    def _shown_eid(self) -> str | None:
        """The elevation the view shows, None on the plan."""
        from .layers import shown_eid

        return shown_eid(self.context.window if self.context else None)

    def _need_campaign(self) -> tuple:
        from . import records

        chunk = self._need_chunk()
        cid = self.pane.campaign_id() if self.pane is not None else None
        if cid is None:
            raise records.NotReady(t("Scegli una campagna (o fanne una nuova con Nuova campagna...).",
                                     "Choose a campaign (or make one with New campaign...)."))
        c = records.campaign(chunk, cid)
        return chunk, self._entry(chunk, c["elevation"]), c

    def _show(self, entry: dict) -> None:
        """The elevation in the view (the drawing requests then give its (u, v)) with the campaign's features."""
        window = self.context.window
        if window is None or not hasattr(window, "show_elevation"):
            return
        if self._shown_eid() != entry["id"]:
            window.show_elevation(entry)
        self._redraw()

    def _raster_loaded(self, *_args) -> None:
        self.drawn = None
        self._redraw()

    def _drop_layers(self) -> None:
        from .layers import drop

        drop(getattr(self.context.window, "ortho_view", None) if self.context else None)
        self.drawn = None

    def _redraw(self) -> None:
        """The features of the campaign chosen drawn on its elevation when the view shows it, else none."""
        from . import records

        chunk = self._chunk()
        eid = self._shown_eid()
        cid = self.pane.campaign_id() if self.pane is not None else None
        if chunk is None or eid is None or cid is None:
            self._drop_layers()
            return
        try:
            c = records.campaign(chunk, cid)
            entry = self._entry(chunk, c["elevation"])
        except records.NotReady:
            self._drop_layers()
            return
        if c["elevation"] != eid:
            self._drop_layers()
            return
        self.redraw(chunk, entry, cid)

    def redraw(self, chunk, entry: dict, cid: str) -> None:
        """The layers of the campaign ``cid`` on the view, which shows ``entry`` (see :mod:`.layers`)."""
        from .layers import draw

        self.drawn = (entry["id"], cid) if draw(getattr(self.context.window, "ortho_view", None), chunk, cid) \
            else None

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self, *_args) -> None:
        if self.pane is None:
            return
        from . import records

        chunk = self._chunk()
        entries = self._elevations(chunk)
        current = self.pane.elevation_id() or self._shown_eid()
        self.pane.set_elevations([{"id": e["id"], "label": self._elevation_label(chunk, e)} for e in entries],
                                 current)
        eid = self.pane.elevation_id()
        rows = records.campaigns(chunk, eid) if chunk is not None and eid else []
        self.pane.set_campaigns([{"id": c["id"], "date": c["date"], "name": c["name"]} for c in rows])
        self._fill_terms(chunk)
        self._fill_features(chunk)
        if not entries:
            self.pane.set_state(t("Nessun prospetto nel chunk: fai un prospetto (pulsante Prospetto della vista 3D) e "
                                  "torna qui.", "No elevation in the chunk: make one (the Elevation button of the 3D "
                                  "view) and come back here."))
        elif not rows:
            self.pane.set_state(t("Nessuna campagna su questo prospetto: Nuova campagna...",
                                  "No campaign on this elevation: New campaign..."))
        else:
            self.pane.set_state("")
        self._redraw()

    def _fill_terms(self, chunk) -> None:
        from . import texts
        from .pane import fold
        from .vocabulary import FAMILIES, load

        lang = texts.lang()
        vocab = load(chunk)

        def row(term, children=()):
            names = list(term.names.values()) + list(term.aliases) + [term.code, vocab.name(term.code, lang)]
            tip = f"{term.definition.get(lang, '')}\n{term.code} - {term.source}".strip()
            return {"code": term.code, "label": vocab.name(term.code, lang), "tip": tip,
                    "search": " ".join(fold(n) for n in names), "children": list(children)}

        mappable = [tm for tm in vocab.terms() if tm.kind in ("decay", "crack") and tm.code != UNNAMED]
        known = {tm.code for tm in mappable}
        favs = [c for c in self._settings_list("favourites") if c in known]
        recent = [c for c in self._settings_list("recent") if c in known]
        groups = [(t("Preferiti", "Favourites"), [row(vocab.term(c)) for c in favs], bool(favs)),
                  (t("Ultimi usati", "Last used"), [row(vocab.term(c)) for c in recent], bool(recent))]
        iscs = [tm for tm in mappable if tm.code.startswith("ISCS.")]
        for family, titles in FAMILIES.items():
            mains = [tm for tm in iscs if tm.family == family and not tm.parent]
            if mains:
                items = [row(tm, [row(s) for s in iscs if s.parent == tm.code]) for tm in mains]
                groups.append((f"ICOMOS-ISCS - {titles[1 if lang == 'en' else 0]}", items, False))
        uni = sorted((tm for tm in mappable if tm.code.startswith("UNI.")), key=lambda tm: fold(tm.names["it"]))
        groups.append(("UNI 11182", [row(tm) for tm in uni], False))
        local = [tm for tm in mappable if not tm.code.startswith(("ISCS.", "UNI."))]
        groups.append((t("Termini locali", "Local terms"), [row(tm) for tm in local], False))
        key = json.dumps(groups, sort_keys=True, default=str)
        if key == self._terms_key:  # the same terms: the tree stays as the user opened it
            return
        self._terms_key = key
        self.pane.set_terms(groups)
        self.pane.set_families([(k, v[1 if lang == "en" else 0]) for k, v in FAMILIES.items()])

    def _first_found(self) -> None:
        for item in self.pane._leaves():
            if not item.isHidden():
                self.pane.terms.setCurrentItem(item)
                return

    def _fill_features(self, chunk) -> None:
        from . import records, texts
        from .vocabulary import load

        cid = self.pane.campaign_id()
        if chunk is None or cid is None:
            self.pane.set_features([])
            self.pane.set_summary("")
            return
        lang = texts.lang()
        vocab = load(chunk)
        rows = []
        for f in records.features(chunk, cid):
            term = vocab.term(f["code"]) if f["code"] in vocab else None
            family = term.family if term is not None else ""
            if term is not None and term.iscs and term.iscs[0] in vocab:
                family = vocab.term(term.iscs[0]).family
            unit = "m²" if f["geometry"].startswith(("POLYGON", "MULTIPOLYGON")) else "m"
            rows.append({"id": f["id"], "term": vocab.name(f["code"], lang) if term is not None else f["code"],
                         "kind": f["kind"], "extent": f["extent"], "unit": unit, "intensity": f.get("intensity"),
                         "status": f["status"], "photos": len(f.get("photos") or []), "notes": f.get("notes") or "",
                         "family": family})
        self.pane.set_features(rows)
        try:
            self.pane.set_summary(self._summary(chunk, cid, lang))
        except Exception as exc:  # noqa: BLE001 - a summary never stops the pane
            self.pane.set_summary(str(exc))

    def _summary(self, chunk, cid: str, lang: str) -> str:
        from . import assess, records
        from .vocabulary import FAMILIES

        c = records.campaign(chunk, cid)
        tot = records.totals(chunk, cid)
        classes = c.get("classes") or {}
        rc = records.assessment(chunk, c["date"])
        lines = [f"{c['name']} - {c['date']}" + (f" - {c['surveyor']}" if c.get("surveyor") else "")]
        proposed = len(records.features(chunk, cid, status="proposed"))
        lines.append(t(f"{tot['features']} elementi accettati, {proposed} proposti",
                       f"{tot['features']} features accepted, {proposed} proposed"))
        for family, area in sorted(tot["area_by_family"].items(), key=lambda x: -x[1]):
            name = FAMILIES.get(family, (family, family))[1 if lang == "en" else 0]
            lines.append(f"  {name}: {num(area, 3)} m²")
        if tot["length_by_code"]:
            length = num(sum(tot["length_by_code"].values()), 2)
            lines.append(t(f"Lesioni: {length} m", f"Cracks: {length} m"))
        if tot.get("share_of_face") is not None:
            share = num(tot["share_of_face"] * 100, 1)
            lines.append(t(f"Degrado su {share} % della faccia", f"Decay on {share} % of the face"))
        lines.append(assess.class_text("cc", classes.get("cc"), lang) + "    "
                     + assess.class_text("uc", classes.get("uc"), lang))
        lines.append(t("Raccomandazione dell'oggetto: ", "Recommendation of the object: ")
                     + assess.class_text("rc", rc.get("rc") if rc else None, lang))
        _cc, rule = assess.suggest_cc(chunk, cid)
        lines.append(t(f"Suggerimento ({assess.SUGGESTION[0]}): {rule}",
                       f"Suggestion ({assess.SUGGESTION[1]}): {rule}"))
        return "\n".join(lines)

    def _requested(self, name: str) -> None:
        self._guarded(getattr(self, "_do_" + name))

    # ------------------------------------------------------------------ campaigns
    def _do_new_campaign(self) -> None:
        from . import forms, records

        chunk = self._need_chunk()
        eid = self.pane.elevation_id() if self.pane is not None else None
        if eid is None:
            raise records.NotReady(t("Il chunk non ha prospetti: fai prima un prospetto.",
                                     "The chunk has no elevation: make one first."))
        entry = self._entry(chunk, eid)
        label = self._elevation_label(chunk, entry)
        v = self._ask(t("Nuova campagna", "New campaign"), forms.campaign_fields(self, chunk),
                      intro=t(f"Un rilievo del degrado di «{label}» in una data.",
                              f"A survey of the decay of “{label}” on a date."))
        if v is None:
            return
        before = self._snapshot(chunk)
        c = records.new_campaign(chunk, eid, date=(v["date"] or "").strip() or None,
                                 name=(v["name"] or "").strip() or None, surveyor=v["surveyor"], method=v["method"],
                                 weather=v["weather"], notes=v["notes"])
        forms.remember_campaign(self, v)
        self.pane.set_campaigns([{"id": x["id"], "date": x["date"], "name": x["name"]}
                                 for x in records.campaigns(chunk, eid)], c["id"])
        self._changed(chunk, before, t("nuova campagna", "new campaign"))
        self._show(entry)

    def _do_edit_campaign(self) -> None:
        from . import forms, records

        chunk, entry, c = self._need_campaign()
        v = self._ask(t("Campagna", "Campaign"), forms.campaign_fields(self, chunk, c))
        if v is None:
            return
        before = self._snapshot(chunk)
        records.set_campaign(chunk, c["id"], date=v["date"], name=(v["name"] or "").strip() or c["name"],
                             surveyor=v["surveyor"], method=v["method"], weather=v["weather"], notes=v["notes"])
        forms.remember_campaign(self, v)
        self._changed(chunk, before, t("campagna modificata", "campaign changed"))

    def _do_remove_campaign(self) -> None:
        from . import records

        chunk, entry, c = self._need_campaign()
        n = len(records.features(chunk, c["id"]))
        if not self._confirm(t(f"Rimuovere la campagna «{c['name']}» con i suoi {n} elementi? Modifica > Annulla la "
                               "riporta. I file già esportati restano nella cartella della campagna.",
                               f"Remove the campaign “{c['name']}” with its {n} features? Edit > Undo brings it back. "
                               "The files already exported stay in the campaign's folder.")):
            return
        before = self._snapshot(chunk)
        records.remove_campaign(chunk, c["id"], files=False)
        self._changed(chunk, before, t("campagna rimossa", "campaign removed"))

    def _do_show(self) -> None:
        from .records import NotReady

        chunk = self._need_chunk()
        eid = self.pane.elevation_id()
        if eid is None:
            raise NotReady(t("Il chunk non ha prospetti.", "The chunk has no elevation."))
        self._show(self._entry(chunk, eid))

    # ------------------------------------------------------------------ mapping
    def _need_term(self):
        from .records import NotReady
        from .vocabulary import load

        code = self.pane.term_code() if self.pane is not None else None
        if not code:
            raise NotReady(t("Scegli un termine nell'elenco della mappatura (cerca per nome, in italiano o in "
                             "inglese).", "Choose a term in the list of the mapping (search it by name, in Italian or "
                             "English)."))
        return load(self._chunk()).term(code)

    def _request(self, kind: str, entry: dict, done, hint: str) -> None:
        self._show(entry)
        method = getattr(self.context, "request_" + kind)
        method(lambda xy: self._guarded(done, np.asarray(xy, float)), hint=hint)

    @staticmethod
    def _geometry(kind: str, xy):
        from shapely.geometry import LineString, Point, Polygon

        from .records import NotReady

        xy = np.asarray(xy, float).reshape(-1, 2)
        if kind == "polygon":
            if len(xy) < 3:
                raise NotReady(t("Un'area ha almeno tre vertici.", "An area has at least three vertices."))
            return Polygon(xy)
        if kind == "line":
            if len(xy) < 2:
                raise NotReady(t("Una linea ha almeno due vertici.", "A line has at least two vertices."))
            return LineString(xy)
        return Point(xy[0])

    def _do_area(self) -> None:
        from .records import NotReady

        _chunk, entry, c = self._need_campaign()
        term = self._need_term()
        if term.kind != "decay":
            raise NotReady(t("Una lesione si disegna con Linea.", "A crack is drawn with Line."))
        self._request("polygon", entry, lambda xy: self._drawn("decay", term.code, c["id"], self._geometry(
            "polygon", xy)), t("Area del degrado: clic per i vertici, doppio clic o Invio per chiudere; Esc annulla",
                               "Area of the decay: click the vertices, double click or Enter to close; Esc cancels"))

    def _do_line(self) -> None:
        from .records import NotReady

        _chunk, entry, c = self._need_campaign()
        term = self._need_term()
        if term.kind != "crack":
            raise NotReady(t("Questo termine è un'area: si disegna con Area.", "This term is an area: draw it with "
                             "Area."))
        self._request("line", entry, lambda xy: self._drawn("crack", term.code, c["id"], self._geometry("line", xy)),
                      t("Lesione: clic per i vertici, doppio clic o Invio per finire; Esc annulla",
                        "Crack: click the vertices, double click or Enter to end; Esc cancels"))

    def _do_point(self) -> None:
        _chunk, entry, c = self._need_campaign()
        term = self._need_term()
        self._request("point", entry, lambda xy: self._drawn("note", term.code, c["id"], self._geometry("point", xy)),
                      t("Nota: clic sul punto; Esc annulla", "Note: click the point; Esc cancels"))

    def _drawn(self, kind: str, code: str, cid: str, geom) -> None:
        """A feature drawn: added, its form asked (cancel = not added), one entry of the undo."""
        from . import forms, records

        chunk = self._need_chunk()
        before = self._snapshot(chunk)
        rec = records.add_feature(chunk, cid, kind, code, geom)
        try:
            values = forms.feature_form(self, chunk, rec)
            if values is not None:
                records.update_feature(chunk, rec["id"], **values)
        except Exception:
            records.remove_feature(chunk, rec["id"])
            raise
        if values is None:
            records.remove_feature(chunk, rec["id"])
            return
        self._remember("recent", code, RECENT)
        self._changed(chunk, before, t("elemento disegnato", "feature drawn"))
        self.pane.select(rec["id"])

    @staticmethod
    def _photos(chunk, entry, rec, limit=6) -> list[dict]:
        from . import records

        try:
            return records.photos_of(chunk, entry, rec, limit=limit)
        except Exception:  # noqa: BLE001 - no photos is not an error of the form
            return []

    def _selected(self, chunk, one: bool = True):
        from . import records

        ids = [i for i in (self.pane.selected_ids() if self.pane is not None else []) if i]
        if not ids:
            raise records.NotReady(t("Scegli un elemento nella tabella.", "Choose a feature in the table."))
        found = [records.feature(chunk, i) for i in ids]
        return found[0] if one else found

    def _do_edit(self) -> None:
        from . import forms, records

        chunk, _entry, c = self._need_campaign()
        rec = self._selected(chunk)
        if rec["kind"] == "treatment":
            v = forms.treatment_form(self, chunk, c, rec.get("treatment") or {}, rec["code"], rec)
        else:
            v = forms.feature_form(self, chunk, rec)
        if v is None:
            return
        before = self._snapshot(chunk)
        records.update_feature(chunk, rec["id"], **v)
        self._changed(chunk, before, t("elemento modificato", "feature changed"))

    def _open_photo(self, photo: dict, rec: dict | None = None) -> None:
        """A photo of a feature in the window's photo view, centred on the feature when it is given."""
        window = self.context.window
        chunk = self._need_chunk()
        camera = next((cam for cam in chunk.cameras if cam.key == photo["key"]), None)
        if camera is None or window is None:
            return
        focus = None
        if rec is not None:
            from shapely import wkt

            from pyarchinit_photogrammetry.engine.elevation import world_of

            from . import records

            g = wkt.loads(rec["geometry"])
            centre = g if g.geom_type == "Point" else g.representative_point()
            entry = self._entry(chunk, records.campaign(chunk, rec["campaign"])["elevation"])
            world = world_of(entry, centre.x, centre.y, None, chunk)
            if world is None:
                world = world_of(entry, centre.x, centre.y, 0.0)
            focus = chunk.to_internal(np.asarray(world, float)[None])[0]
        window.open_photo(camera, focus)

    def _do_photos(self) -> None:
        from . import records

        chunk, entry, _c = self._need_campaign()
        rec = self._selected(chunk)
        photos = rec.get("photos") or self._photos(chunk, entry, rec, limit=1)
        if not photos:
            raise records.NotReady(t("Nessuna foto allineata vede questo elemento.",
                                     "No aligned photo sees this feature."))
        self._open_photo(photos[0], rec)

    def _segmenter(self):
        """The program's SAM 2.1 worker; NotReady with the way to install it when it is missing."""
        from .records import NotReady

        window = self.context.window
        if window is not None and hasattr(window, "_segmenter"):
            try:
                return window._segmenter()
            except ImportError:
                pass
        try:
            from pyarchinit_photogrammetry.ai.sam_env import make_segmenter

            return make_segmenter()
        except ImportError:
            raise NotReady(t("SAM 2.1 non è installato: installalo dagli strumenti SAM del programma (per esempio "
                             "Pietre (SAM)), poi riprova.", "SAM 2.1 is not installed: install it from the program's "
                             "SAM tools (for instance Stones (SAM)), then try again.")) from None

    def _chosen_area(self, chunk):
        """The one polygon feature chosen in the table, or None."""
        from . import records

        ids = [i for i in (self.pane.selected_ids() if self.pane is not None else []) if i]
        if len(ids) != 1:
            return None
        rec = records.feature(chunk, ids[0])
        return rec if rec["geometry"].upper().startswith(("POLYGON", "MULTIPOLYGON")) else None

    def _do_propose(self) -> None:
        """The *Proposte* dialog: a detector of :mod:`.detect` on the area chosen in the table or on the whole face,
        in the background; its results join the campaign as proposals (``source`` ``proposed:<detector>``)."""
        from shapely import wkt

        from . import detect, forms, texts
        from .records import NotReady
        from .vocabulary import load

        chunk, entry, c = self._need_campaign()
        selected = self._chosen_area(chunk)
        code = self.pane.term_code() if self.pane is not None else None
        vocab = load(chunk)
        term = vocab.term(code) if code else None
        v = forms.proposals_form(self, selected, None if term is None else vocab.name(term.code, texts.lang()))
        if v is None:
            return
        name = v["detector"]
        # what a proposal with no term of its own gets: the working term "da nominare", or - for SAM's segments only,
        # when the user asked for it - the term chosen in the list
        default = UNNAMED
        if name == "sam" and v.get("use_term") and term is not None and term.kind == "decay":
            default = term.code
        reference = None
        if name == "like":
            if selected is None:
                raise NotReady(t("Come questo: scegli nella tabella l'area che fa da esempio.",
                                 "Like this: choose in the table the area that is the example."))
            if selected["code"] == UNNAMED:
                raise NotReady(t("Come questo: l'area d'esempio è ancora da nominare; dalle prima un termine.",
                                 "Like this: the example area is still to be named; give it a term first."))
            reference, default = wkt.loads(selected["geometry"]), selected["code"]
        area = None
        if v["window"] == "selected" and selected is not None and name != "like":  # like: the example is inside
            area = wkt.loads(selected["geometry"])
        bounds = None if area is None else area.bounds
        segmenter = self._segmenter() if name in ("sam", "like") else None
        res = forms.number(v.get("resolution_mm"))
        res = None if res is None else res / 1000.0

        def work(progress):
            win = detect.read_window(chunk, entry, bounds)
            if win is None or win.empty:
                return None
            if name == "cracks":
                return detect.cracks(chunk, entry, bounds, min_length=float(v["min_length_cm"]) / 100.0,
                                     progress=progress)
            if name == "loss":
                return detect.material_loss(chunk, entry, bounds, min_depth=float(v["min_depth_cm"]) / 100.0,
                                            min_area=float(v["min_area_m2"]), progress=progress)
            if name == "colours":
                return detect.colour_classes(chunk, entry, bounds, k=int(v["k"]), progress=progress)
            found = detect.sam_areas(chunk, entry, bounds, thoroughness=v.get("thoroughness") or "normale",
                                     resolution=res, progress=progress, segmenter=segmenter)
            if name == "like":
                radius = float(v["radius"])
                keep = detect.like_this(chunk, entry, [f["geometry"] for f in found], reference, radius)
                found = [found[i] for i in keep]
                for f in found:
                    f["notes"] += t(f"; come l'area {selected['id']} (distanza massima {num(radius, 2)})",
                                    f"; like the area {selected['id']} (largest distance {num(radius, 2)})")
            return found

        self.context.run(t("Conservazione: proposte", "Conservation: proposals"), work,
                         lambda result: self._guarded(self._proposed, chunk, c, name, area, default, result))

    def _restore(self, chunk, before) -> None:
        """The records as they were before a change that failed half-way."""
        if before is None:
            chunk.meta.pop("conservation", None)
        else:
            chunk.meta["conservation"] = copy.deepcopy(before)

    def _proposed(self, chunk, c: dict, name: str, area, default: str, result) -> None:
        """The detector's results as proposals of the campaign ``c``: clipped to the window ``area``, those with no
        term given ``default`` (the working term "da nominare" unless the user chose otherwise), the colour classes
        named by the user first. A failure half-way leaves the records as they were."""
        from . import forms, records

        if result is None:
            self._message("information", t("La finestra non ha superficie: niente da proporre.",
                                           "The window has no surface: nothing to propose."))
            return
        source = "proposed:" + {"like": "sam", "loss": "material_loss"}.get(name, name)
        found = result
        if name == "colours":
            named = forms.colour_classes_form(self, chunk, result) if result else {}
            if named is None:
                return
            found = [{"kind": "decay", "code": named[cl["class"]], "geometry": g, "width_mm": None, "depth_mm": None,
                      "notes": cl["notes"], "coverage": cl["coverage"]}
                     for cl in result if cl["class"] in named for g in cl["polygons"]]
        before = self._snapshot(chunk)
        added, unnamed = 0, 0
        try:
            for p in found:
                geom = p["geometry"] if area is None else _parts(p["geometry"].intersection(area), p["kind"])
                if geom is None:
                    continue
                code = p["code"] or default
                unnamed += code == UNNAMED
                cover = float(p.get("coverage") or 0.0)
                notes = p["notes"] + t(f"; rilievo sul {num(cover * 100, 0)} % della finestra",
                                       f"; relief on {num(cover * 100, 0)} % of the window")
                records.add_feature(chunk, c["id"], p["kind"], code, geom, status="proposed", source=source,
                                    width_mm=None if p.get("width_mm") is None else round(float(p["width_mm"]), 1),
                                    depth_mm=p.get("depth_mm"), notes=notes)
                added += 1
        except Exception:
            self._restore(chunk, before)
            raise
        if added:
            self._changed(chunk, before, t("proposte aggiunte", "proposals added"))
        text = t(f"{added} proposte aggiunte alla campagna (layer Proposte): Accetta o Scarta.",
                 f"{added} proposals added to the campaign (layer Proposte): Accept or Discard.") if added else \
            t("Nessuna proposta nella finestra.", "No proposal in the window.")
        if unnamed:
            text += " " + t(f"{unnamed} sono da nominare: scegline una, dalle un termine con Modifica... e usa Come "
                            "questo per le simili.", f"{unnamed} are to be named: choose one, give it a term with "
                            "Edit... and use Like this for the similar ones.")
        self._message("information", text)

    def _do_like(self) -> None:
        """*Come questo*: the campaign's proposed areas from SAM (and those still to be named) that look like the
        area chosen in the table take its term (:func:`.detect.like_this`, in the background; one entry of the
        undo)."""
        from shapely import wkt

        from . import detect, records

        chunk, entry, c = self._need_campaign()
        ref = self._chosen_area(chunk)
        if ref is None or ref["code"] == UNNAMED:
            raise records.NotReady(t("Come questo: scegli nella tabella un'area con il termine giusto.",
                                     "Like this: choose in the table an area with the right term."))
        others = [f for f in records.features(chunk, c["id"], status="proposed")
                  if f["id"] != ref["id"] and f["kind"] == "decay"
                  and (str(f.get("source", "")).startswith("proposed:sam") or f["code"] == UNNAMED)]
        if not others:
            raise records.NotReady(t("La campagna non ha aree proposte da SAM o da nominare: usa prima Proposte...",
                                     "The campaign has no area proposed by SAM or to be named: use Proposals... "
                                     "first."))
        radius = float((self.context.settings.get("proposals") or {}).get("radius", 1.0))
        shapes = [wkt.loads(f["geometry"]) for f in others]
        reference = wkt.loads(ref["geometry"])

        def work(_progress):
            return detect.like_this(chunk, entry, shapes, reference, radius)

        self.context.run(t("Conservazione: come questo", "Conservation: like this"), work,
                         lambda keep: self._guarded(self._liked, chunk, ref, others, keep, radius))

    def _liked(self, chunk, ref: dict, others: list, keep: list, radius: float) -> None:
        from . import records

        known = {f["id"] for f in records.features(chunk)}
        changed = [others[i] for i in keep if others[i]["code"] != ref["code"] and others[i]["id"] in known]
        if not changed:
            self._message("information", t("Nessun'altra area proposta somiglia a quella scelta.",
                                           "No other proposed area looks like the chosen one."))
            return
        before = self._snapshot(chunk)
        try:
            for f in changed:
                records.update_feature(chunk, f["id"], code=ref["code"], notes=(f.get("notes") or "") + t(
                    f"; termine come l'area {ref['id']} (distanza massima {num(radius, 2)})",
                    f"; term as the area {ref['id']} (largest distance {num(radius, 2)})"))
        except Exception:
            self._restore(chunk, before)
            raise
        self._changed(chunk, before, t("termine come questo", "term like this"))
        self._message("information", t(f"{len(changed)} aree proposte prendono il termine dell'area scelta.",
                                       f"{len(changed)} proposed areas take the term of the chosen area."))

    def _do_accept(self) -> None:
        from . import records

        chunk, _entry, _c = self._need_campaign()
        found = [f for f in self._selected(chunk, one=False) if f["status"] == "proposed"]
        if not found:
            raise records.NotReady(t("Gli elementi scelti non sono proposte.", "The features chosen are no proposals."))
        before = self._snapshot(chunk)
        records.accept(chunk, [f["id"] for f in found])
        self._changed(chunk, before, t("proposte accettate", "proposals accepted"))

    def _do_accept_all(self) -> None:
        from . import records

        chunk, _entry, c = self._need_campaign()
        found = records.features(chunk, c["id"], status="proposed")
        if not found:
            raise records.NotReady(t("La campagna non ha proposte.", "The campaign has no proposals."))
        before = self._snapshot(chunk)
        records.accept(chunk, [f["id"] for f in found])
        self._changed(chunk, before, t("proposte accettate", "proposals accepted"))

    def _do_discard(self) -> None:
        from . import records

        chunk, _entry, _c = self._need_campaign()
        found = self._selected(chunk, one=False)
        accepted = [f for f in found if f["status"] != "proposed"]
        if accepted and not self._confirm(
                t(f"Fra gli elementi scelti {len(accepted)} non sono proposte ma elementi del rilievo: rimuoverli "
                  "comunque? Modifica > Annulla li riporta.",
                  f"Among the features chosen {len(accepted)} are no proposals but features of the survey: remove "
                  "them anyway? Edit > Undo brings them back.")):
            return
        before = self._snapshot(chunk)
        records.discard(chunk, [f["id"] for f in found])
        self._changed(chunk, before, t("elementi scartati", "features discarded"))

    def _do_favourite(self) -> None:
        term = self._need_term()
        favs = self._settings_list("favourites")
        self.context.settings["favourites"] = [c for c in favs if c != term.code] if term.code in favs \
            else favs + [term.code]
        self.context.save_settings()
        self.refresh_pane()

    def _on_deleted(self, features: list) -> None:
        """Features deleted on the view with the window's tools: their records go too."""
        from . import records

        chunk = self._chunk()
        if chunk is None:
            return
        known = {f["id"] for f in records.features(chunk)}
        fids = [a.get("fid") for a in features if a.get("fid") in known]
        if not fids:
            return
        before = self._snapshot(chunk)
        records.discard(chunk, fids)
        self._changed(chunk, before, t("elementi eliminati", "features deleted"))

    # ------------------------------------------------------------------ interventions
    def _treated(self, cid: str, geom, code: str | None = None, intro: str = "") -> None:
        from . import forms, records

        chunk = self._need_chunk()
        c = records.campaign(chunk, cid)
        v = forms.treatment_form(self, chunk, c, {}, code, intro=intro)
        if v is None:
            return
        before = self._snapshot(chunk)
        rec = records.add_feature(chunk, cid, "treatment", v["code"], geom, notes=v["notes"],
                                  treatment=v["treatment"])
        self._changed(chunk, before, t("intervento", "treatment"))
        self.pane.select(rec["id"])

    def _do_treat_area(self) -> None:
        _chunk, entry, c = self._need_campaign()
        self._request("polygon", entry, lambda xy: self._treated(c["id"], self._geometry("polygon", xy)),
                      t("Area dell'intervento: clic per i vertici, doppio clic o Invio per chiudere; Esc annulla",
                        "Area of the treatment: click the vertices, double click or Enter to close; Esc cancels"))

    def _do_treat_line(self) -> None:
        _chunk, entry, c = self._need_campaign()
        self._request("line", entry, lambda xy: self._treated(c["id"], self._geometry("line", xy)),
                      t("Lesione trattata: clic per i vertici, doppio clic o Invio per finire; Esc annulla",
                        "Crack treated: click the vertices, double click or Enter to end; Esc cancels"))

    def _do_to_treat(self) -> None:
        from . import assess, records, texts
        from .vocabulary import load

        chunk, _entry, c = self._need_campaign()
        rec = self._selected(chunk)
        if rec["kind"] not in ("decay", "crack"):
            raise records.NotReady(t("Scegli un elemento di degrado o una lesione.",
                                     "Choose a decay feature or a crack."))
        lang = texts.lang()
        vocab = load(chunk)
        usual = assess.usual_treatments(rec["code"], chunk)
        names = ", ".join(vocab.name(u, lang) for u in usual) or "-"
        term = vocab.name(rec["code"], lang)
        intro = t(f"Per «{term}» di solito: {names} (suggerimento del plugin). La geometria è quella dell'elemento.",
                  f"For “{term}” usually: {names} (suggestion of the plugin). The geometry is the feature's.")
        self._treated(c["id"], rec["geometry"], usual[0] if usual else None, intro=intro)

    # ------------------------------------------------------------------ assessment
    def _do_classes(self) -> None:
        from . import forms, records

        chunk, _entry, c = self._need_campaign()
        v = forms.classes_form(self, chunk, c)
        if v is None:
            return
        before = self._snapshot(chunk)
        records.set_campaign(chunk, c["id"], classes=v)
        self._changed(chunk, before, t("classi della campagna", "classes of the campaign"))

    def _do_assessment(self) -> None:
        from . import forms, records

        chunk, _entry, c = self._need_campaign()
        v = forms.assessment_form(self, chunk, c)
        if v is None:
            return
        before = self._snapshot(chunk)
        records.set_assessment(chunk, c["date"], v["rc"], reason=v["reason"], surveyor=v["surveyor"])
        self._changed(chunk, before, t("raccomandazione", "recommendation"))

    # ------------------------------------------------------------------ outputs (the next steps of the plugin)
    def _coming(self) -> None:
        from .records import NotReady

        raise NotReady(t("Questa uscita è in arrivo in questa versione del plugin.",
                         "This output is coming in this version of the plugin."))

    _do_plate = _do_tables = _do_drawings = _do_qgis = _do_pyarchinit = _do_ifc = _do_compare = _do_folder = _coming

    # ------------------------------------------------------------------ exports
    def _export_plate(self, chunk, path, progress=None):
        """*File > Esporta > Tavola della campagna (PDF)*."""
        from .records import NotReady

        raise NotReady(t("La tavola della campagna non è ancora disponibile in questa versione del plugin.",
                         "The plate of the campaign is not available yet in this version of the plugin."))

    # ------------------------------------------------------------------ command line
    def command(self, *args) -> int:
        import pyarchinit_photogrammetry as ppg

        from . import records, texts
        from .vocabulary import load

        parser = argparse.ArgumentParser(prog="ppg plugin conservation", description="decay mapping on elevations")
        parser.add_argument("what", choices=["list"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        a = parser.parse_args(list(args))
        doc = ppg.Document.open(a.project)
        chunk = next((c for c in doc.chunks if c.label == a.chunk), None) if a.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {a.chunk!r}")
            return 2
        vocab = load(chunk)
        lang = texts.lang()
        names = {e["id"]: e.get("name", e["id"]) for e in chunk.meta.get("elevations") or []}
        found = records.campaigns(chunk)
        if not found:
            print(t("Nessuna campagna.", "No campaign."))
        for c in found:
            feats = records.features(chunk, c["id"])
            print(f"{c['id']}  {c['date']}  {c['name']}  [{names.get(c['elevation'], c['elevation'])}]  "
                  f"{len(feats)}")
            for f in feats:
                label = vocab.name(f["code"], lang) if f["code"] in vocab else f["code"]
                extent = "" if f["extent"] is None else f"  {f['extent']:.3f}"
                print(f"    {f['id']}  {f['kind']:<9} {f['status']:<8} {label}{extent}")
        return 0
