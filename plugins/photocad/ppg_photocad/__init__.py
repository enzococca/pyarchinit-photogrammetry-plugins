"""PhotoCAD - a plugin of pyarchinit-photogrammetry: the orthophoto, the DEM (as a hillshade) or the orthophoto of an
elevation sent to the PhotoCAD API (photocad.ai, an external paid service) to be vectorised, the drawing brought
back in place - the coordinates of the chunk on the plan, the ``(u, v)`` of the elevation (and the world on its
surface) - as DXF, GeoPackage and SVG.

* *Plugin > PhotoCAD*: *Vettorializza l'ortofoto...*, *... il DEM...*, *... il prospetto mostrato...* (all of it,
  the part on the screen, or an area drawn), *Converti in US / USM...*, *Importa come disegni liberi* (program
  1.7.2), *Recupera una conversione...*, *Esporta una conversione...*, *Rimuovi una conversione...*, *Chiave API PhotoCAD...*, *Credito PhotoCAD*;
* every conversion a layer "PhotoCAD: <name>" of the 2D view, on the plan or on its elevation, drawn again with
  the raster;
* ``ppg plugin photocad key set|forget|status``, ``credit``, ``convert PROJECT --source ortho|dem|elevation:<id>
  ...``, ``fetch PROJECT --id ID``, ``list``, ``units``, ``export``, ``remove``.

Before every upload the user confirms (the credits left, the size of the image), unless they chose not to be
asked again for the project. The API key: the environment variable ``PHOTOCAD_API_KEY``, else the system keychain
(:mod:`.keys`); never anywhere else.
"""
from __future__ import annotations

import argparse
import json

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import LAYER_PREFIX, layer_name, t

__version__ = "0.1.2"

POLYGON_COLOUR = "#e040fb"
LINE_COLOUR = "#ff4081"
STONE_COLOUR = "#ffab00"


def _has_free_drawings() -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec("pyarchinit_photogrammetry.core.free_drawings") is not None
    except (ImportError, ValueError):
        return False


def _open_file(path) -> None:
    """The file in the system's viewer (macOS open, Windows startfile, xdg-open)."""
    import os
    import subprocess
    import sys

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["/usr/bin/open", str(path)], close_fds=False)  # posix_spawn: no fork after SpatiaLite
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # noqa: S606
        else:
            subprocess.Popen(["xdg-open", str(path)], close_fds=False)
    except OSError:
        pass


class PhotoCADPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = "PhotoCAD"
        self.description = t("Vettorializza ortofoto, DEM e prospetti con PhotoCAD (photocad.ai): il disegno torna "
                             "georeferenziato come DXF, GeoPackage e SVG",
                             "Vectorise orthophotos, DEMs and elevations with PhotoCAD (photocad.ai): the drawing "
                             "comes back georeferenced as DXF, GeoPackage and SVG")
        self.context = None
        self.pane = None

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        add = context.add_action
        if context.window is not None and hasattr(context, "add_pane"):
            add(t("PhotoCAD...", "PhotoCAD..."), self.show_pane,
                tip=t("Il pannello di PhotoCAD", "The PhotoCAD pane"))
        add(t("Vettorializza l'ortofoto...", "Vectorise the orthophoto..."), lambda: self.convert_dialog("ortho"),
            tip=t("Invia l'ortofoto (o una parte) a PhotoCAD", "Send the orthophoto (or a part) to PhotoCAD"))
        add(t("Vettorializza il DEM...", "Vectorise the DEM..."), lambda: self.convert_dialog("dem"),
            tip=t("Il DEM come ombreggiatura", "The DEM as a hillshade"))
        add(t("Vettorializza il prospetto mostrato...", "Vectorise the elevation shown..."),
            lambda: self.convert_dialog("elevation"))
        add(t("Converti in US / USM...", "Turn into US / USM..."), self.units_dialog,
            tip=t("I poligoni PhotoCAD selezionati (o un layer) diventano unità stratigrafiche",
                  "The PhotoCAD polygons selected (or a layer) become stratigraphic units"))
        if _has_free_drawings():
            add(t("Importa come disegni liberi...", "Import as free drawings..."), self.free_drawings_dialog)
        add(t("Recupera una conversione...", "Fetch a conversion..."), self.fetch_dialog,
            tip=t("Scarica di nuovo una conversione già pagata (finita, non scaricata, o da rifare)",
                  "Download again a conversion already paid for (finished, not downloaded, or to redo)"))
        add(t("Esporta una conversione...", "Export a conversion..."), self.export_dialog)
        add(t("Rimuovi una conversione...", "Remove a conversion..."), lambda: self.remove_conversion(None))
        add(t("Chiave API PhotoCAD...", "PhotoCAD API key..."), self.key_dialog)
        add(t("Credito PhotoCAD", "PhotoCAD credit"), self.credit)
        context.add_command("photocad", self.command,
                            help="PhotoCAD vectorisation: key, credit, convert, list, units, export, remove")
        if context.window is not None:
            context.on_raster_loaded(self._raster_loaded)
            context.on_refresh(self._raster_loaded)
            if hasattr(context, "add_pane"):
                from .pane import PhotoCADPane

                self.pane = PhotoCADPane()
                context.add_pane(self.pane, "PhotoCAD", "plugin", area="right")
                self.pane.requested.connect(self._requested)
                self.refresh_pane()

    def deactivate(self) -> None:
        self.pane = None
        view = self._view()
        if view is not None:
            for name in [n for n in list(getattr(view, "layers", {})) if n.startswith(LAYER_PREFIX)]:
                view.remove_layer(name)

    # ------------------------------------------------------------------ the pane
    def show_pane(self) -> None:
        if self.pane is not None:
            self.context.show_pane(self.pane)

    def _raster_loaded(self, *_args) -> None:
        self.draw_layers()
        self.refresh_pane()

    def _source_available(self, src: str) -> bool:
        from . import source

        chunk = self._chunk()
        if chunk is None:
            return False
        try:
            source.raster_of(chunk, src)
        except Exception:  # noqa: BLE001 - a cheap check: any failure means "not there"
            return False
        return True

    def refresh_pane(self) -> None:
        """The pane from the chunk: the key, the buttons that can be used, the table of the conversions."""
        if self.pane is None:
            return
        from . import keys, project

        w = self._window()
        frame = getattr(w, "drawings_frame", None)
        chunk = self._chunk()
        self.pane.set_state(
            key_text=self._key_status(keys),
            ortho=self._source_available("ortho"), dem=self._source_available("dem"),
            elevation=frame is not None and self._source_available(f"elevation:{frame}"),
            records=[] if chunk is None else [{**r, "_summary": project.summary(r)} for r in project.records(chunk)],
            frame=frame, free=_has_free_drawings(), units=hasattr(self.context, "add_units"))

    def _key_status(self, keys) -> str:
        """The key line of the pane, read from the keychain once and again only after the key dialog or an action
        that used the key (a locked keychain could otherwise hold the window at every refresh)."""
        if getattr(self, "_key_text", None) is None:
            self._key_text = keys.status_text()
        return self._key_text

    def _requested(self, action: str, arg=None) -> None:
        simple = {"key": self.key_dialog, "credit": self.credit, "units": self.units_dialog,
                  "ortho": lambda: self.convert_dialog("ortho"), "dem": lambda: self.convert_dialog("dem"),
                  "elevation": lambda: self.convert_dialog("elevation")}
        if action in simple:
            simple[action]()
        elif action == "toggle":
            self.toggle_visible(arg)
        elif action == "fetch":
            self.fetch_dialog(arg)
        elif action == "export":
            self.export_dialog(arg)
        elif action == "remove":
            self.remove_conversion(arg)
        elif action == "free":
            self.free_drawings_dialog(arg)
        self.refresh_pane()

    def toggle_visible(self, rid: str | None) -> None:
        """Show or hide the layers of one conversion (the flag is kept in its record) and draw them again."""
        from . import project

        chunk = self._chunk()
        rec = project.record(chunk, rid) if chunk is not None and rid else None
        if rec is None:
            return
        rec["visible"] = not rec.get("visible", True)
        project._write_json(chunk, rec)
        self.context.mark_modified()
        # a layer hidden by hand in the Layer pane stays hidden unless the user shows it with this button
        hidden = getattr(self, "_hidden", set())
        for n in (layer_name(rec), layer_name(rec) + t(" (pietre)", " (stones)")):
            hidden.discard(n)
        self._hidden = hidden
        self.draw_layers()

    # ------------------------------------------------------------------ helpers
    def _window(self):
        return self.context.window if self.context is not None else None

    def _view(self):
        w = self._window()
        return getattr(w, "ortho_view", None) if w is not None else None

    def _chunk(self):
        w = self._window()
        chunk = getattr(w, "chunk", None) if w is not None else None
        if chunk is not None:
            return chunk
        doc = self.context.app.document if self.context is not None else None
        return getattr(doc, "chunk", None)

    def _doc(self):
        w = self._window()
        return getattr(w, "doc", None) if w is not None and getattr(w, "doc", None) is not None \
            else self.context.app.document

    def _say(self, text: str, kind: str = "info") -> None:
        w = self._window()
        if w is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui import dialogs

        log = getattr(w, "log_pane", None)
        if log is not None:
            (log.warning if kind == "warning" else log.info)(text)
        dialogs.message(w, kind, "PhotoCAD", text)

    def records(self, chunk=None) -> list[dict]:
        from . import project

        chunk = chunk or self._chunk()
        return [] if chunk is None else project.records(chunk)

    # ------------------------------------------------------------------ key and credit
    def key_dialog(self) -> None:
        from .gui import KeyDialog

        KeyDialog(self._window()).exec()
        self._key_text = None  # the pane reads the keychain again

    def _key(self) -> str | None:
        from . import keys

        key = keys.get_key()
        if key is None and self._window() is not None:
            self._say(t("Serve la chiave API di PhotoCAD: inseriscila nella finestra che si apre.",
                        "The PhotoCAD API key is needed: type it in the window that opens."), "warning")
            self.key_dialog()
            key = keys.get_key()
        return key

    def credit(self) -> None:
        from . import client

        key = self._key()
        if key is None:
            return
        try:
            text = client.account_text(client.Client(key).account())
        except client.PhotoCADError as exc:
            self._say(str(exc), "warning")
            return
        self._say("PhotoCAD - " + text)

    # ------------------------------------------------------------------ the conversion in the window
    def convert_dialog(self, kind: str) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog

        from . import source

        w, chunk = self._window(), self._chunk()
        if w is None or chunk is None:
            return
        if kind == "elevation":
            eid = getattr(w, "drawings_frame", None)
            if eid is None:
                self._say(t("Mostra prima un prospetto o una sezione nella vista 2D (dall'area di lavoro).",
                            "Show an elevation or a section in the 2D view first (from the workspace)."), "warning")
                return
            src = f"elevation:{eid}"
        else:
            src = kind
        try:
            source.raster_of(chunk, src)
        except (source.NoRaster, ValueError) as exc:
            self._say(str(exc), "warning")
            return
        key = self._key()
        if key is None:
            return
        view = self._view()
        on_it = view is not None and view.tr is not None and (
            (kind == "elevation") or (getattr(w, "drawings_frame", None) is None and view.kind in ("ortho", "dem")))
        areas = [("all", t("Tutto il raster", "The whole raster"))]
        if on_it:
            areas.append(("visible", t("La parte visibile nella vista", "The part visible in the view")))
            areas.append(("draw", t("Un'area da disegnare", "An area to draw")))
        fields = [Field("area", t("Area", "Area"), "all", choices=areas),
                  Field("detailed", t("Disegno dettagliato", "Detailed drawing"), True),
                  Field("remove_background", t("Togli lo sfondo", "Remove the background"), False)]
        if kind == "elevation":
            fields.append(Field("three_d", t("Anche in 3D sulla superficie del prospetto",
                                             "Also in 3D on the surface of the elevation"), False))
        v = ParamDialog.ask(w, t("Vettorializza con PhotoCAD", "Vectorise with PhotoCAD"), fields, intro=t(
            "L'immagine (al massimo 2400 px sul lato lungo) va a photocad.ai; il disegno torna al suo posto come "
            "layer «PhotoCAD: ...» e in DXF, GeoPackage e SVG. Prima dell'invio si chiede conferma.",
            "The image (at most 2400 px on its long side) goes to photocad.ai; the drawing comes back in place as a "
            "layer 'PhotoCAD: ...' and as DXF, GeoPackage and SVG. A confirmation is asked before sending."))
        if not v:
            return
        opts = {"detailed": bool(v.get("detailed", True)), "remove_background": bool(v.get("remove_background")),
                "three_d": bool(v.get("three_d"))}
        if v.get("area") == "visible" and view is not None:
            self._prepare(chunk, src, key, view.visible_world_bounds(), opts)
        elif v.get("area") == "draw":
            def drawn(xy):
                import numpy as np

                xy = np.asarray(xy, float)
                self._prepare(chunk, src, key, (*xy.min(axis=0), *xy.max(axis=0)), opts)

            self.context.request_polygon(drawn, hint=t(
                "Area per PhotoCAD: trascina un riquadro, oppure clic per i vertici e doppio clic o Invio; Esc annulla",
                "Area for PhotoCAD: drag a box, or click the vertices and double click or Enter; Esc cancels"))
        else:
            self._prepare(chunk, src, key, None, opts)

    def _prepare(self, chunk, src, key, bbox, opts) -> None:
        from . import client, project, source

        def work(progress):
            progress(0.1, t("Preparo l'immagine per PhotoCAD", "Preparing the image for PhotoCAD"))
            try:
                return project.prepare(chunk, src, key, bbox)
            except (client.PhotoCADError, source.NoRaster, ValueError) as exc:
                return exc

        def done(prep):
            if isinstance(prep, Exception):
                self._say(str(prep), "warning")
                return
            if not self._confirm(prep):
                w = self._window()
                if w is not None and hasattr(w, "statusBar"):
                    w.statusBar().showMessage(t("PhotoCAD: nulla è stato inviato.", "PhotoCAD: nothing was sent."),
                                              8000)
                return
            self._upload(chunk, prep, key, opts)

        self.context.run(t("PhotoCAD: immagine", "PhotoCAD: image"), work, done)

    def _confirm(self, prep) -> bool:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import project

        doc = self._doc()
        if doc is not None and not project.confirmation_wanted(doc):
            return True
        info = project.confirm_info(prep)
        answer = dialogs.choice(self._window(), "question", t("Invio a PhotoCAD", "Send to PhotoCAD"), info["text"],
                                [t("Invia", "Send"),
                                 t("Invia e non chiedere più per questo progetto", "Send and do not ask again for this "
                                                                                    "project")],
                                default=0, cancel=t("Annulla", "Cancel"))
        if answer is None:
            return False
        if answer == 1 and doc is not None:
            project.set_confirmation_wanted(doc, False)
            self.context.mark_modified()
        return True

    def _offer_accept(self, chunk, exc) -> bool:
        """After a refusal for the residual alone (a soft image: the search found a clear match): the overlay
        shown, and the placement accepted on the user's word - free, nothing sent again. True when asked."""
        from pyarchinit_photogrammetry.gui import dialogs

        from . import project, register

        if not isinstance(exc, project.FrameMismatch):
            return False
        bad = [(rec, tl) for rec in project.records(chunk) if rec.get("status") == "mismatch"
               for tl in rec.get("tiles") or [] if tl.get("status") == "mismatch"]
        if not bad or not all(register.only_residual(tl["registration"]) for _, tl in bad):
            return False
        rec = bad[-1][0]
        looks = [project.folder_of(chunk, rec) / tl["folder"] / "overlay.jpg" for r, tl in bad if r is rec]
        for path in looks:
            if path.exists():
                _open_file(path)
        answer = dialogs.choice(self._window(), "question", "PhotoCAD", str(exc) + "\n\n" + t(
            "Nella sovrapposizione (aperta) i tratti rossi stanno sui contorni dell'immagine?",
            "In the overlay (opened) do the red lines sit on the outlines of the image?"),
            [t("Accetta dopo il controllo", "Accept after a look")], cancel=t("Lascia rifiutato", "Leave refused"))
        if answer is None:
            return True
        self.fetch_dialog(rec["id"], accept=True)
        return True

    def _upload(self, chunk, prep, key, opts) -> None:
        from . import client, project

        def work(progress):
            try:
                return project.upload(chunk, prep, key, opts["detailed"], opts["remove_background"],
                                      opts.get("three_d", False), progress=progress)
            except client.PhotoCADError as exc:
                return exc

        def done(rec):
            self.context.mark_modified()  # also a failure: the paid conversion is kept in the project
            if isinstance(rec, Exception):
                if not self._offer_accept(chunk, rec):
                    self._say(str(rec), "warning")
                return
            self.draw_layers()
            self.refresh_pane()
            w = self._window()
            if w is not None and hasattr(w, "log_pane"):
                w.log_pane.info("PhotoCAD: " + project.summary(rec))
            if w is not None and hasattr(w, "statusBar"):
                w.statusBar().showMessage("PhotoCAD: " + project.summary(rec), 12000)

        self.context.run(t("PhotoCAD: vettorializzazione", "PhotoCAD: vectorisation"), work, done)

    def fetch_dialog(self, rid: str | None = None, accept: bool = False) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import client, project

        chunk = self._chunk()
        recs = self.records()
        if chunk is None or not recs:
            self._say(t("Nessuna conversione PhotoCAD nel chunk.", "No PhotoCAD conversion in the chunk."), "warning")
            return
        recs = sorted(recs, key=lambda r: r.get("status", "done") == "done")  # those not done first
        if rid is not None:
            k = next((i for i, r in enumerate(recs) if r["id"] == rid), None)
        else:
            k = dialogs.choice(self._window(), "question", t("Recupera una conversione", "Fetch a conversion"),
                               t("Quale conversione riprendere da PhotoCAD? Le tessere già inviate si scaricano di nuovo senza addebito.",
                                 "Which conversion to take up again from PhotoCAD? The tiles already sent are downloaded "
                                 "again at no charge."),
                               [project.summary(r) for r in recs])
        if k is None:
            return
        todo = project.pending_uploads(recs[k])
        if todo and dialogs.choice(self._window(), "question", "PhotoCAD", t(
                "{0} tessere di questa conversione non sono ancora state inviate: saranno inviate ora (una conversione "
                "ciascuna).", "{0} tiles of this conversion have not been sent yet: they will be sent now (one "
                "conversion each).").format(todo), [t("Invia", "Send")]) is None:
            return
        key = self._key()
        if key is None:
            return
        rid = recs[k]["id"]

        def work(progress):
            try:
                return project.fetch(chunk, rid, key, progress, accept=accept)
            except (client.PhotoCADError, ValueError) as exc:
                return exc

        def done(rec):
            self.context.mark_modified()
            if isinstance(rec, Exception):
                if accept or not self._offer_accept(chunk, rec):
                    self._say(str(rec), "warning")
                return
            self.draw_layers()
            self.refresh_pane()
            self._say("PhotoCAD: " + project.summary(rec))

        self.context.run(t("PhotoCAD: recupero", "PhotoCAD: fetch"), work, done)

    # ------------------------------------------------------------------ the layers
    def draw_layers(self) -> None:
        """Every conversion of the frame shown (the plan, or the elevation shown) as a layer of the 2D view."""
        from . import project

        view, w, chunk = self._view(), self._window(), self._chunk()
        if view is None:
            return
        hidden = getattr(self, "_hidden", set())
        for name in [n for n in list(view.layers) if n.startswith(LAYER_PREFIX)]:
            if not view.layers[name].get("visible", True):
                hidden.add(name)
            else:
                hidden.discard(name)
            view.remove_layer(name)
        self._hidden = hidden
        if chunk is None or view.tr is None:
            return
        frame = getattr(w, "drawings_frame", None)
        with view.batch():
            for rec in project.records(chunk):
                if rec.get("frame") != frame or not rec.get("visible", True):
                    continue
                name = layer_name(rec)
                for i, (g, a) in enumerate(project.features(chunk, rec)):
                    attrs = {"photocad": rec["id"], "n": i, "photocad_layer": a.get("layer") or "",
                             "entity": a.get("entity") or ""}
                    if g.geom_type == "Polygon":
                        view.add_polygon([list(p) for p in list(g.exterior.coords)[:-1]], "", POLYGON_COLOUR, 25,
                                         layer=name, attrs=attrs)
                    else:
                        view.add_line(list(g.coords), LINE_COLOUR, "", layer=name, attrs=attrs, width=1.5)
                stones = name + t(" (pietre)", " (stones)")
                for i, (g, a) in enumerate(project.stone_features(chunk, rec)):
                    view.add_polygon([list(p) for p in list(g.exterior.coords)[:-1]], "", STONE_COLOUR, 60,
                                     layer=stones, attrs={"photocad": rec["id"], "n": i, "photocad_layer": "pietre",
                                                          "area_m2": round(float(a.get("area_m2") or 0), 4)})
                for n in (name, stones):
                    if n in hidden and n in view.layers:  # hidden in the Layer pane: stays hidden on a refresh
                        view.set_layer_visible(n, False)

    # ------------------------------------------------------------------ units
    def units_dialog(self) -> None:
        from pyarchinit_photogrammetry.archaeo.stratigraphy import US_TYPES
        from pyarchinit_photogrammetry.gui.dialogs import Field, ParamDialog
        from shapely.geometry import LineString, Polygon

        from . import project

        w, view, chunk = self._window(), self._view(), self._chunk()
        if w is None or chunk is None:
            return
        if not hasattr(self.context, "add_units"):
            self._say(t("Per fare le US serve pyarchinit-photogrammetry 1.5.0 o successivo.",
                        "Making units needs pyarchinit-photogrammetry 1.5.0 or later."), "warning")
            return
        frame = getattr(w, "drawings_frame", None)
        here = [r for r in project.records(chunk) if r.get("frame") == frame]
        selected = []
        for layer, _i, attrs in (view.selected_features() if view is not None else []):
            if not str(layer).startswith(LAYER_PREFIX) or "_geom" not in attrs:
                continue
            kind, coords = attrs["_geom"]
            geom = Polygon(coords) if kind == "Polygon" else LineString(coords)
            if attrs.get("photocad_layer") == "pietre" and attrs.get("photocad"):  # the stone itself, with its holes
                rec_s = project.record(chunk, attrs["photocad"])
                stones = project.stone_features(chunk, rec_s) if rec_s is not None else []
                k = int(attrs.get("n", -1))
                if 0 <= k < len(stones):
                    geom = stones[k][0]
            selected.append((geom, {"layer": attrs.get("photocad_layer")}))
        if not selected and not here:
            self._say(t("Nessun disegno PhotoCAD qui: vettorializza prima l'ortofoto, il DEM o il prospetto.",
                        "No PhotoCAD drawing here: vectorise the orthophoto, the DEM or the elevation first."), "warning")
            return
        which = []
        if selected:
            which.append(("selected", t("I {0} elementi selezionati", "The {0} features selected").format(len(selected))))
        if here:
            which.append(("layer", t("Tutto un layer PhotoCAD", "A whole PhotoCAD layer")))
        fields = [Field("which", t("Quali", "Which"), which[0][0], choices=which)]
        if here:
            fields.append(Field("layer", t("Layer", "Layer"), here[-1]["id"],
                                choices=[(r["id"], layer_name(r)) for r in here]))
        if any(project.stone_features(chunk, r) for r in here):
            fields.append(Field("from", t("Dal layer", "From the layer"), "polygons", choices=[
                ("polygons", t("le pietre (poligoni ricavati dalle linee)", "the stones (polygons made from the lines)")),
                ("lines", t("il disegno di PhotoCAD (solo i contorni chiusi)",
                            "PhotoCAD's drawing (closed outlines only)"))]))
        fields += [Field("tipo", t("Tipo", "Kind"), "USM" if frame is not None else "US",
                         choices=[(k, k) for k in US_TYPES]),
                   Field("start", t("Primo numero (0: senza numero)", "First number (0: no number)"), 0,
                         minimum=0, maximum=1e7)]
        v = ParamDialog.ask(w, t("PhotoCAD: US / USM", "PhotoCAD: US / USM"), fields, intro=t(
            "Ogni contorno chiuso diventa un'US del chunk attivo (sul prospetto mostrato, se è un prospetto); le linee "
            "aperte si saltano. Modifica > Annulla le toglie tutte insieme.",
            "Every closed outline becomes a unit of the active chunk (on the elevation shown, if it is one); open "
            "lines are skipped. Edit > Undo takes them all away together."))
        if not v:
            return
        if v.get("which") == "layer":
            rec = project.record(chunk, v.get("layer"))
            feats = project.unit_features(chunk, rec, v.get("from")) if rec is not None else []
        else:
            feats = selected
        polygons, attrs, skipped = project.unit_polygons(feats, v.get("tipo") or "US", int(v.get("start") or 0) or None)
        made = self.context.add_units(polygons, tipo=v.get("tipo") or "US", attributes=attrs) if polygons else 0
        self._say(t("{0} unità aggiunte ({1}); {2} linee aperte (non chiuse) saltate.",
                    "{0} units added ({1}); {2} open lines (not closed) skipped.").format(
            made, v.get("tipo") or "US", skipped))

    def make_units(self, chunk, rec: dict, tipo: str = "US", start: int | None = None,
                   source: str | None = None) -> tuple[int, int]:
        """The closed outlines of a conversion into units without the window (command line): on the plan through
        ``context.add_units``; on an elevation into its drawings (``chunk.meta["drawings"]["frames"][id]``).
        ``(made, skipped)``."""
        from . import project

        polygons, attrs, skipped = project.unit_polygons(project.unit_features(chunk, rec, source), tipo, start)
        if not polygons:
            return 0, skipped
        if rec.get("frame") is None:
            return self.context.add_units(polygons, tipo=tipo, attributes=attrs, chunk=chunk), skipped
        from pyarchinit_photogrammetry.archaeo.stratigraphy import units_from_polygons

        doc = chunk.document
        units = units_from_polygons(polygons, tipo, attrs, sito=str(doc.meta.get("sito") or "") or None)
        frame = chunk.meta.setdefault("drawings", {}).setdefault("frames", {}).setdefault(rec["frame"], {})
        frame.setdefault("units", []).extend({**u.to_dict(), "layer": None} for u in units)
        frame.setdefault("quote", [])
        return len(units), skipped

    # ------------------------------------------------------------------ free drawings (program 1.7.2)
    def free_drawings_dialog(self, rid: str | None = None) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        chunk = self._chunk()
        recs = self.records()
        if chunk is None or not recs:
            self._say(t("Nessuna conversione PhotoCAD nel chunk.", "No PhotoCAD conversion in the chunk."), "warning")
            return
        if rid is not None:
            k = next((i for i, r in enumerate(recs) if r["id"] == rid), None)
        else:
            k = dialogs.choice(self._window(), "question", t("Importa come disegni liberi", "Import as free drawings"),
                               t("Quale conversione? I suoi elementi vanno nel layer di disegni liberi «PhotoCAD».",
                                 "Which conversion? Its features go into the free-drawing layer 'PhotoCAD'."),
                               [r["name"] for r in recs])
        if k is None:
            return
        n = self.import_free_drawings(chunk, recs[k])
        self.context.mark_modified()
        if hasattr(self.context, "refresh_tree"):
            self.context.refresh_tree()
        self._say(t("{0} disegni liberi aggiunti nel layer «PhotoCAD».", "{0} free drawings added to the layer "
                    "'PhotoCAD'.").format(n))

    @staticmethod
    def import_free_drawings(chunk, rec: dict) -> int:
        """The features of a conversion as free drawings (layer ``PhotoCAD``) in 3D: on the plan at the height of
        the DEM (0 without one), on an elevation on its surface."""
        import numpy as np

        from pyarchinit_photogrammetry.core import free_drawings

        from . import drawing, project, source

        feats = project.features(chunk, rec)
        items = []
        for g, a in feats:
            xy = np.asarray(g.exterior.coords if g.geom_type == "Polygon" else g.coords, float)[:, :2]
            if g.geom_type == "Polygon":
                xy = xy[:-1]
            items.append({"layer": a.get("layer"), "entity": a.get("entity"), "closed": g.geom_type == "Polygon",
                          "xy": xy})
        if rec.get("frame") is not None:
            items = drawing.world_items(chunk, source.elevation_entry(chunk, rec["frame"]), items)
        else:
            z = _dem_heights(chunk, items)
            items = [{**it, "xyz": np.column_stack([it["xy"], zz])} for it, zz in zip(items, z)]
        n = 0
        for it in items:
            free_drawings.add(chunk, "polygon" if it["closed"] else "line", it["xyz"], layer="PhotoCAD",
                              attributes={"photocad": rec["id"], "photocad_layer": it["layer"] or ""})
            n += 1
        return n

    # ------------------------------------------------------------------ export, remove
    def export_dialog(self, rid: str | None = None) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui import dialogs

        from . import project

        chunk, recs = self._chunk(), self.records()
        if chunk is None or not recs:
            self._say(t("Nessuna conversione PhotoCAD nel chunk.", "No PhotoCAD conversion in the chunk."), "warning")
            return
        if rid is not None:
            k = next((i for i, r in enumerate(recs) if r["id"] == rid), None)
        else:
            k = dialogs.choice(self._window(), "question", "PhotoCAD", t("Quale conversione esportare?",
                                                                         "Which conversion to export?"),
                               [r["name"] for r in recs])
        if k is None:
            return
        path, _f = QFileDialog.getSaveFileName(self._window(), t("Esporta il disegno PhotoCAD", "Export the PhotoCAD "
                                                                 "drawing"), recs[k]["name"] + ".dxf",
                                               "DXF (*.dxf);;GeoPackage (*.gpkg);;SVG (*.svg)")
        if not path:
            return
        try:
            out = project.export(chunk, recs[k]["id"], path)
        except (ValueError, OSError) as exc:
            self._say(str(exc), "warning")
            return
        self._say(t("Esportato: {0}", "Exported: {0}").format(out))

    def remove_conversion(self, rid: str | None) -> None:
        from pyarchinit_photogrammetry.gui import dialogs

        from . import project

        chunk, recs = self._chunk(), self.records()
        if chunk is None or not recs:
            return
        if rid is None:
            k = dialogs.choice(self._window(), "question", "PhotoCAD", t("Quale conversione togliere dal progetto "
                                                                         "(con i suoi file)?", "Which conversion to "
                                                                         "remove from the project (with its files)?"),
                               [r["name"] for r in recs])
            if k is None:
                return
            rid = recs[k]["id"]
        else:
            rec = project.record(chunk, rid)
            if rec is None or dialogs.choice(self._window(), "question", "PhotoCAD", t(
                    "Togliere «{0}» dal progetto, con i suoi file? Le US fatte dai suoi poligoni restano.",
                    "Remove '{0}' from the project, with its files? The units made from its polygons stay.")
                    .format(rec["name"]), [t("Togli", "Remove")]) is None:
                return
        project.remove(chunk, rid)
        self.context.mark_modified()
        self.draw_layers()
        self.refresh_pane()

    # ------------------------------------------------------------------ command line
    @staticmethod
    def _parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(prog="ppg plugin photocad",
                                         description="PhotoCAD (photocad.ai, an external paid service): vectorise an "
                                                     "orthophoto, a DEM or an elevation; the drawing georeferenced.")
        sub = parser.add_subparsers(dest="what", required=True)
        p = sub.add_parser("key", help="the API key in the system keychain (typed without echo)")
        p.add_argument("action", choices=("set", "forget", "status"))
        sub.add_parser("credit", help="the credits left on the account")
        p = sub.add_parser("convert", help="send an image to PhotoCAD (one credit) and bring the drawing back")
        p.add_argument("project")
        p.add_argument("--chunk", help="the chunk by name (the active one by default)")
        p.add_argument("--source", default="ortho", help="ortho, dem or elevation:<id> (see list --elevations)")
        p.add_argument("--bbox", help="xmin,ymin,xmax,ymax: a part (coordinates of the chunk, or (u, v) of the "
                                      "elevation)")
        p.add_argument("--simplified", action="store_true", help="a simplified drawing (detailed=false)")
        p.add_argument("--remove-background", action="store_true")
        p.add_argument("--3d", dest="three_d", action="store_true", help="an elevation also in 3D on its surface")
        p.add_argument("--pdf", action="store_true", help="also PhotoCAD's PDF")
        p.add_argument("--name", help="name of the conversion (its layer)")
        p.add_argument("--out", help="a copy of the drawing: FILE.dxf, .gpkg, .svg or .pdf")
        p.add_argument("--max-tiles", type=int, default=6,
                       help="most tiles (one conversion each) a long image may take (default 6)")
        p.add_argument("--yes", action="store_true", help="no confirmation prompt")
        p = sub.add_parser("fetch", help="download and georeference again a conversion already paid for (no upload)")
        p.add_argument("project")
        p.add_argument("--id", required=True, help="the id of the conversion, or its PhotoCAD id")
        p.add_argument("--chunk")
        p.add_argument("--out", help="a copy of the drawing: FILE.dxf, .gpkg, .svg or .pdf")
        p.add_argument("--search-px", type=float, default=None,
                       help="offset searched by the registration, image pixels (default 24; widened once by itself)")
        p.add_argument("--yes", action="store_true", help="send tiles not sent yet without asking")
        p.add_argument("--accept", action="store_true",
                       help="place a drawing refused only for its residual (a soft image, a clear match), after "
                            "looking at the overlay.jpg of its tiles")
        p = sub.add_parser("list", help="the conversions of the chunk (and its elevations)")
        p.add_argument("project")
        p.add_argument("--chunk")
        p.add_argument("--json", action="store_true")
        p = sub.add_parser("units", help="the closed outlines of a conversion into stratigraphic units")
        p.add_argument("project")
        p.add_argument("id")
        p.add_argument("--chunk")
        p.add_argument("--tipo", default="US", help="US, USM, ...")
        p.add_argument("--start", type=int, default=0, help="first number (0: no numbers)")
        p.add_argument("--from", dest="source", choices=("lines", "polygons"), default=None,
                       help="the stones made from the lines (default when there are some) or PhotoCAD's own outlines")
        p = sub.add_parser("polygonize", help="make the stones of a conversion again from its lines")
        p.add_argument("project")
        p.add_argument("id")
        p.add_argument("--chunk")
        p.add_argument("--snap-px", type=float, default=2.0, help="ends this close (image pixels) are joined")
        p.add_argument("--min-area-px", type=float, default=50.0, help="smaller faces are left out (image pixels)")
        p = sub.add_parser("export", help="a conversion as FILE.dxf, .gpkg, .svg or .pdf")
        p.add_argument("project")
        p.add_argument("id")
        p.add_argument("out")
        p.add_argument("--chunk")
        p = sub.add_parser("remove", help="a conversion out of the project, with its files")
        p.add_argument("project")
        p.add_argument("id")
        p.add_argument("--chunk")
        return parser

    @staticmethod
    def glue_values(args: list[str]) -> list[str]:
        """``--bbox -1.5,...``: a value beginning with a dash glued to its option."""
        out, i = [], 0
        while i < len(args):
            if args[i] == "--bbox" and i + 1 < len(args) and args[i + 1].startswith("-"):
                out.append(f"--bbox={args[i + 1]}")
                i += 2
                continue
            out.append(args[i])
            i += 1
        return out

    def _open(self, path: str, label: str | None):
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(path)
        if self.context is not None:
            self.context.app.document = doc
        if label:
            return doc, next((c for c in doc.chunks if c.label == label), None)
        return doc, doc.chunk

    def command(self, *args) -> int:
        from . import client, keys, project, source

        ns = self._parser().parse_args(self.glue_values([str(a) for a in args]))
        if ns.what == "key":
            if ns.action == "status":
                print(keys.status_text())
                return 0
            if ns.action == "forget":
                keys.forget_key()
                print(t("Chiave tolta dal portachiavi.", "Key removed from the keychain."))
                return 0
            import getpass

            typed = getpass.getpass(t("Chiave API PhotoCAD (non si vede): ", "PhotoCAD API key (hidden): "))
            try:
                keys.set_key(typed)
            except (keys.KeyringUnavailable, ValueError) as exc:
                print(keys.mask(str(exc), typed))
                return 1
            print(keys.status_text())
            return 0
        if ns.what == "credit":
            key = keys.get_key()
            if not key:
                print(keys.status_text())
                return 1
            try:
                print("PhotoCAD - " + client.account_text(client.Client(key).account()))
            except client.PhotoCADError as exc:
                print(f"error: {exc}")
                return 1
            return 0
        doc, chunk = self._open(ns.project, ns.chunk)
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project" if ns.chunk else "the project has no chunk")
            return 1
        if ns.what == "list":
            recs = project.records(chunk)
            if ns.json:
                print(json.dumps(recs, indent=1, ensure_ascii=False))
                return 0
            for e in chunk.meta.get("elevations") or []:
                print(f"elevation:{e.get('id')}  {e.get('name')} ({e.get('kind', 'elevation')})")
            for r in recs:
                print(f"{r['id']}  {r['source']:<16} {project.summary(r)}")
            return 0
        if ns.what == "export":
            try:
                print(project.export(chunk, ns.id, ns.out))
            except (ValueError, OSError) as exc:
                print(f"error: {exc}")
                return 1
            return 0
        if ns.what == "remove":
            if not project.remove(chunk, ns.id):
                print(t("Nessuna conversione «{0}».", "No conversion '{0}'.").format(ns.id))
                return 1
            doc.save()
            return 0
        if ns.what == "fetch":
            key = keys.get_key()
            if not key:
                print(keys.status_text())
                return 1
            found = project.find(chunk, ns.id)
            todo = project.pending_uploads(found) if found is not None else 0
            if todo and not ns.yes:
                print(t("{0} tessere non sono ancora state inviate: saranno inviate ora (una conversione ciascuna).",
                        "{0} tiles have not been sent yet: they will be sent now (one conversion each).").format(todo))
                try:
                    answer = input(t("Inviare? [s/N] ", "Send? [y/N] ")).strip().lower()
                except EOFError:
                    answer = ""
                if answer not in ("s", "si", "sì", "y", "yes"):
                    return 1
            try:
                run = (self.context.run if self.context is not None else (lambda title, f, done=None: f(None)))
                rec = run("PhotoCAD fetch", lambda p: project.fetch(chunk, ns.id, key, p, ns.search_px, ns.accept))
            except (client.PhotoCADError, ValueError) as exc:
                print(f"error: {exc}")
                doc.save()
                return 1
            doc.save()
            print(project.summary(rec))
            if ns.out:
                print(project.export(chunk, rec["id"], ns.out))
            return 0
        if ns.what == "polygonize":
            rec = project.record(chunk, ns.id)
            if rec is None or rec.get("status", "done") != "done":
                print(t("Nessuna conversione finita «{0}».", "No finished conversion '{0}'.").format(ns.id))
                return 1
            n = project.polygonize(chunk, rec, ns.snap_px, ns.min_area_px)
            doc.save()
            print(json.dumps(rec["polygonize"], indent=1))
            print(t("{0} pietre in {1}", "{0} stones in {1}").format(n, project.path_of(chunk, rec, "polygons")))
            return 0
        if ns.what == "units":
            rec = project.record(chunk, ns.id)
            if rec is None:
                print(t("Nessuna conversione «{0}».", "No conversion '{0}'.").format(ns.id))
                return 1
            made, skipped = self.make_units(chunk, rec, ns.tipo, ns.start or None, ns.source)
            doc.save()
            print(t("{0} unità aggiunte ({1}); {2} linee aperte (non chiuse) saltate.",
                    "{0} units added ({1}); {2} open lines (not closed) skipped.").format(made, ns.tipo, skipped))
            return 0
        # convert
        key = keys.get_key()
        if not key:
            print(keys.status_text())
            return 1
        bbox = None
        if ns.bbox:
            parts = [float(v) for v in ns.bbox.replace(" ", ",").split(",") if v]
            if len(parts) != 4:
                print("--bbox needs four numbers: xmin,ymin,xmax,ymax")
                return 1
            bbox = parts
        ask = not ns.yes and project.confirmation_wanted(doc)

        def confirm(info):
            if not ask:
                return True
            print(info["text"])
            try:
                answer = input(t("Inviare? [s/N] ", "Send? [y/N] ")).strip().lower()
            except EOFError:  # no terminal (a pipe, cron): no
                answer = ""
            return answer in ("s", "si", "sì", "y", "yes")

        def run(progress):
            return project.convert(chunk, ns.source, key, bbox, not ns.simplified, ns.remove_background, ns.three_d,
                                   ns.pdf, confirm, progress, ns.name, max_tiles=ns.max_tiles)

        try:
            rec = self.context.run("PhotoCAD", run) if self.context is not None else run(None)
        except project.Declined as exc:
            print(exc)
            return 1
        except (client.PhotoCADError, source.NoRaster, ValueError) as exc:
            print(f"error: {exc}")
            doc.save()  # a paid conversion that failed afterwards is kept in the project
            return 1
        doc.save()
        print(project.summary(rec))
        print(f"id: {rec['id']}  folder: {project.folder_of(chunk, rec)}")
        if ns.out:
            try:
                print(project.export(chunk, rec["id"], ns.out))
            except (ValueError, OSError) as exc:
                print(f"error: {exc}")
                return 1
        return 0


def _dem_heights(chunk, items):
    """The heights of the DEM under the vertices of ``items`` (0 where there is no DEM or no value)."""
    import numpy as np

    path = chunk.dem_path
    if path is None or not path.exists():
        return [np.zeros(len(it["xy"])) for it in items]
    import rasterio

    out = []
    with rasterio.open(path) as src:
        for it in items:
            z = np.array([v[0] for v in src.sample([tuple(map(float, p)) for p in it["xy"]])], float)
            if src.nodata is not None:
                z[z == src.nodata] = np.nan
            missing = int((~np.isfinite(z)).sum())
            if missing:
                import logging

                logging.getLogger("pyarchinit_photogrammetry.plugins.photocad").warning(
                    "PhotoCAD free drawings: %d vertices without a DEM value, placed at z = 0", missing)
            out.append(np.where(np.isfinite(z), z, 0.0))
    return out
