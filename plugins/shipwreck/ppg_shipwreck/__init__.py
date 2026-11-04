"""Shipwreck - a plugin of pyarchinit-photogrammetry: the technical drawing of an excavated hull from its 3D model.

* the *Relitto* pane (on the right), step by step: the ship frame (x along the keel, y to port, z up), the stations
  (regular, from a table with the field's names, from the frames of the catalogue), the lines cut from the model, the
  water (a level, the refraction taken out where the user says so), the drawings (body plan, half-breadth plan, profile,
  plan over the orthophoto; SVG, PDF, PNG, DXF), the catalogue of the timbers with their measures, the clinker
  strakes (the lands found on the sections, the land lines, a strake model labelled a hypothesis, the file link with
  Fröhlich & Nielsen's Clinker Boat Tool for Houdini), the relief of the hull from above (an elevation of the
  program in the ship frame, its hillshade or local relief in grey over it) and the tracing on it (the proposals
  accepted, rejected, corrected, added by hand, named from a table, flagged loose; each change a step of *Modifica >
  Annulla*), a keel from given dimensions (a hypothesis), every traced timber a closed 3D object (OBJ, GLB, IFC with
  the tables) and the plan of the wreck with every timber outlined, coloured by kind and named; the reconstruction,
  every part of it a stated hypothesis (step 9: the stem and the stern, the mirror, the extension of the strakes, the
  frames at the measured spacing, the faired hull, over the other chunks of the same wreck too; the table of what is
  measured and what is reconstructed, the 3D files and the plan with it; each build a step of *Modifica > Annulla*);
* the products in the workspace tree, the timbers and the land lines in layers of the raster view on the plan, the
  proposals, the traced timbers and the reconstruction (hatched) in layers of the relief while it is shown;
* *File > Esporta > Relitto: catalogo dei legni*;
* ``ppg plugin shipwreck frame|stations|sections|water|drawings|timbers|export|status``,
  ``ppg plugin shipwreck clinker detect|strakes|export|import|status`` and
  ``ppg plugin shipwreck relief|propose|trace|keel|solids|export3d|plan`` and
  ``ppg plugin shipwreck ends|frame-of|chunks|reconstruct``.

It measures what is there; what it adds (the keel of given dimensions, the strake model of the sections) is a stated
hypothesis, apart from the measures. It reads the model: the transform, the model and the products of the chunk are
never changed, and nothing is written outside ``<chunk data>/shipwreck/`` but what the user exports.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from pyarchinit_photogrammetry.plugins import Plugin

from .texts import NAME, num, plural, t

__version__ = "0.4.0"
NEGATIVE_VALUES = ("--start", "--end", "--longitudinals", "--horizontals")  # may begin with a minus: "-0.2;-0.5", "-1.5,2"
LAYER = "Legni del relitto"
CLINKER_LAYER = "Fasce"  # the land lines, in the chunk's x, y
SIDE_COLOURS = {"port": "#e04040", "starboard": "#30a040", "centre": "#ffd000"}
PROPOSALS_LAYER = "Proposte (relitto)"  # on the relief, in its (u, v): dashed
TRACED_LAYER = "Legni ricalcati"  # on the relief, in its (u, v): solid, labelled with the name
RECONSTRUCTION_LAYER = "Ricostruzione (ipotesi)"  # on the relief, in its (u, v): hatched, a hatch per way
REC_HATCHES = {"reconstructed:mirror": "BDiagPattern", "reconstructed:extension": "FDiagPattern",
               "reconstructed:spacing": "DiagCrossPattern"}  # as on the plan of the wreck: ////, \\\\, xxxx
UNDO_FOLDER = "reconstruction.undo"  # the reconstruction's files before and after each build, for Annulla
_UNDO_DIRS: set = set()  # the copies the steps of Annulla of this process hold (module level: a plugin turned off
_LAST_AFTER: dict = {}   # and on again keeps them); per chunk key: (the copy after the last build, its meta)
_HELD: dict = {}  # per copy, how many steps of Annulla hold it (the "before" of a build is the "after" of the one before)
KIND_COLOURS = {"seam": "#00e5ff", "frame": "#ff8c00", "keelson": "#d040ff", "keel": "#8d6e63", "stringer": "#ffd000",
                "plank": "#30c060", "ceiling": "#9ccc65", "fastening": "#ff3030", "other": "#b0b0b0"}
UNDO_KEYS = ("timbers", "proposals", "keel")  # what a step of the tracing changes, kept for *Modifica > Annulla*
OVERLAY_TITLES = {"height": ("Altezze", "Heights"), "hillshade": ("Ombreggiatura", "Hillshade"),
                  "slrm": ("Rilievo locale", "Local relief")}


class _Step:
    """One change of the catalogue or the proposals as an entry of the window's *Modifica > Annulla*: both before and
    after (the program's undo takes no snapshot of a plugin's meta by itself)."""

    def __init__(self, plugin, chunk, before: dict, after: dict):
        self.plugin, self.chunk, self.before, self.after = plugin, chunk, before, after

    def _put(self, state: dict) -> None:
        from . import ship

        st = ship.state(self.chunk)
        for key, value in state.items():
            if value is None:
                st.pop(key, None)
            else:
                st[key] = copy.deepcopy(value)
        self.plugin._changed()

    def undo(self) -> None:
        self._put(self.before)

    def redo(self) -> None:
        self._put(self.after)


class _RebuildStep:
    """A build of the reconstruction (or its ends set) as an entry of *Modifica > Annulla*: the reconstruction's
    entry of the meta before and after and, for a build, its folder before and after (copies by hard links: the files
    are replaced, never written over in place)."""

    def __init__(self, plugin, chunk, before, after, before_dir=None, after_dir=None):
        self.plugin, self.chunk = plugin, chunk
        self.before, self.after, self.before_dir, self.after_dir = before, after, before_dir, after_dir
        for d in {before_dir, after_dir} - {None}:
            _HELD[d] = _HELD.get(d, 0) + 1

    def drop(self) -> None:
        """The history forgets this step (trimmed, or cut by a new step after an undo): its copies go when no other
        step holds them and the next build will not reuse them."""
        for d in {self.before_dir, self.after_dir} - {None}:
            _HELD[d] = _HELD.get(d, 1) - 1
            if _HELD[d] <= 0:
                _HELD.pop(d, None)
                _forget_copy(self.chunk, d)
        self.before_dir = self.after_dir = None

    def _put(self, state, folder) -> None:
        from . import ship

        if folder is not None:
            if not folder.exists():  # never an empty folder under a meta that lists files
                self.plugin._message("warning", t("La copia per Annulla della ricostruzione non c'è più: ricostruisci.",
                                                  "The copy of the reconstruction for Undo is gone: reconstruct."))
                return
            _restore_folder(folder, _rec_folder(self.chunk))
        root = ship.state(self.chunk)
        if state is None:
            root.pop("reconstruction", None)
        else:
            root["reconstruction"] = copy.deepcopy(state)
        self.plugin._changed()

    def undo(self) -> None:
        self._put(self.before, self.before_dir)

    def redo(self) -> None:
        self._put(self.after, self.after_dir)


class ShipwreckPlugin(Plugin):
    version = __version__

    def __init__(self):
        self.title = t(*NAME)
        self.description = t("Il disegno tecnico di uno scafo scavato dal suo modello 3D: sistema della nave, sezioni, "
                             "piani, catalogo dei legni, fasce del fasciame a clinker, rilievo dello scafo e ricalco, "
                             "legni in 3D, pianta del relitto, ricostruzione (ipotesi)", "The technical drawing of an "
                             "excavated hull from its 3D model: ship frame, sections, plans, timber catalogue, clinker "
                             "strakes, relief of the hull and tracing, 3D timbers, plan of the wreck, reconstruction "
                             "(hypothesis)")
        self.context = None
        self.pane = None
        self.overlay = None  # the grey raster over the relief (:class:`overlay.Overlay`)
        self.overlay_kind = "hillshade"  # which one: a kind of :data:`relief.KINDS`, or "photo" for none
        self._deleting = None  # (step, layer group) of a deletion on the view while the event is being handled

    # ------------------------------------------------------------------ activation
    def activate(self, context) -> None:
        self.context = context
        context.add_action(t("Relitto...", "Shipwreck..."), self.show_pane,
                           tip=t("Il pannello del disegno dello scafo", "The pane of the drawing of the hull"))
        context.add_command("shipwreck", self.command,
                            help="the technical drawing of a hull: frame, stations, sections, water, drawings, "
                                 "timbers, export, status, clinker, relief ... plan, ends, frame-of, chunks, reconstruct")
        context.add_exporter(t("Relitto: catalogo dei legni (CSV)", "Shipwreck: timber catalogue (CSV)"), ".csv", self._export)
        context.add_exporter(t("Relitto: catalogo dei legni (Excel)", "Shipwreck: timber catalogue (Excel)"), ".xlsx", self._export)
        context.add_exporter(t("Relitto: catalogo dei legni (GeoPackage)", "Shipwreck: timber catalogue (GeoPackage)"), ".gpkg",
                             self._export)
        if context.window is not None:
            from .pane import ShipwreckPane

            self.pane = ShipwreckPane()
            context.add_pane(self.pane, t(*NAME), "plugin", area="right")
            self.pane.requested.connect(self._requested)
            context.on_refresh(self.refresh_pane)
            context.on_raster_loaded(self.draw_layer)
            context.on_features_deleted(LAYER, self._features_deleted)
            context.on_features_deleted(PROPOSALS_LAYER, self._proposals_deleted)
            context.on_features_deleted(TRACED_LAYER, self._features_deleted)
            context.on_features_deleted(RECONSTRUCTION_LAYER, self._reconstruction_deleted)
            self._clean_undo_copies(getattr(context.window, "doc", None) or context.app.document)
            context.add_tree_items(self._tree_items)
            self.refresh_pane()

    def deactivate(self) -> None:
        if self.overlay is not None:
            self.overlay.remove()
        self.overlay = None
        view = getattr(self.context.window, "ortho_view", None) if self.context else None
        if view is not None:
            for name in (LAYER, CLINKER_LAYER, PROPOSALS_LAYER, TRACED_LAYER, RECONSTRUCTION_LAYER):
                view.remove_layer(name)
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

    def _message(self, kind: str, text: str) -> None:
        window = self.context.window if self.context else None
        if window is None:
            print(text)
            return
        from pyarchinit_photogrammetry.gui.dialogs import message

        message(window, kind, self.title, text)

    def _status(self, text: str) -> None:
        window = self.context.window
        if window is None:
            print(text)
        elif hasattr(window, "statusBar"):
            window.statusBar().showMessage(text, 15000)

    def _changed(self) -> None:
        self.context.mark_modified()
        self.context.refresh_tree()
        self.refresh_pane()
        self.draw_layer()

    @staticmethod
    def _snapshot(chunk) -> dict:
        from . import ship

        st = chunk.meta.get(ship.KEY) or {}
        return copy.deepcopy({key: st.get(key) for key in UNDO_KEYS})

    def _step(self, chunk, before: dict, text: str, deleted: str | None = None) -> None:
        """After a change of the catalogue or the proposals: an entry of *Modifica > Annulla* (``text``), then the
        project modified and the pane, the tree and the layers shown again. ``deleted``: the group of layers ("timbers",
        "proposals") of a deletion on the view; the window tells the plugin layer by layer, so the second group of the
        same deletion joins the first's entry (one *Annulla* brings both back)."""
        window = self.context.window
        history = getattr(window, "history", None) if window is not None else None
        scripted = bool(getattr(window, "_script_running", False))  # a script's own entry covers it
        after = self._snapshot(chunk)
        changed = _without_caches(before) != _without_caches(after)  # a plan cache refreshed is no change
        if history is not None and not scripted and changed and chunk is getattr(window, "chunk", None):
            joined = self._deleting
            if (deleted and joined and joined[1] != deleted and history.entries
                    and history.index == len(history.entries) and history.entries[-1].action is joined[0]):
                joined[0].after = copy.deepcopy(after)
                history.entries[-1].text = t("Relitto: proposte e legni eliminati", "Shipwreck: proposals and timbers "
                                             "deleted")
                if getattr(history, "on_change", None) is not None:  # the menu says the new text
                    history.on_change()
                self._deleting = None
            else:
                step = _Step(self, chunk, before, after)
                history.push(t(f"Relitto: {text}", f"Shipwreck: {text}"), step, document=False)
                if deleted:
                    from PySide6.QtCore import QTimer

                    self._deleting = (step, deleted)
                    QTimer.singleShot(0, self._deletion_handled)
        self._changed()

    def _deletion_handled(self) -> None:
        self._deleting = None

    def _guarded(self, function, *args) -> None:
        """A step of the user: what it needs and is missing is said, not raised."""
        from .ship import NotReady

        try:
            function(*args)
        except NotReady as exc:
            self._message("warning", str(exc))

    def _run(self, title: str, job, done) -> None:
        window = self.context.window
        if window is not None and hasattr(window, "run_task"):
            window.run_task(title, job, done, refresh=False, save=True)
        else:
            self.context.run(title, job, done)

    def _ask(self, title: str, fields: list, intro: str = "") -> dict | None:
        from pyarchinit_photogrammetry.gui.dialogs import ParamDialog

        return ParamDialog.ask(self.context.window, title, fields, intro=intro)

    def _choose(self, title: str, text: str, options: list[str]) -> int | None:
        """The index of the option pressed, None for *Annulla* (a non-native box)."""
        from pyarchinit_photogrammetry.gui.dialogs import choice

        return choice(self.context.window, "question", title, text, options)

    # ------------------------------------------------------------------ the pane
    def refresh_pane(self) -> None:
        if self.pane is None:
            return
        chunk = self._chunk()
        self.pane.set_states(self.states(chunk) if chunk is not None else {k: "" for k in self.pane.states})
        self.pane.set_rows(self._rows(chunk) if chunk is not None else [])
        self.pane.set_notes(rebuild_notes(chunk))

    def _rows(self, chunk) -> list:
        from . import ship, timbers

        try:
            return timbers.measure(chunk)
        except ship.NotReady:
            return []

    @staticmethod
    def states(chunk) -> dict:
        """The state of each step, in words."""
        from . import drawings, lines, ship, timbers

        frame = ship.frame_of(chunk)
        stations = ship.stations_of(chunk)
        data = lines.load(chunk) if frame is not None else None
        out = {"frame": t("Non impostato.", "Not set."), "stations": t("Nessuna stazione.", "No stations."),
               "sections": t("Non calcolate.", "Not computed."), "water": t("Correzione spenta.", "Correction off."),
               "drawings": t("Nessun disegno.", "No drawings."), "timbers": t("Catalogo vuoto.", "The catalogue is empty."),
               "clinker": clinker_state(chunk), "relief": relief_state(chunk),
               "reconstruction": reconstruction_state(chunk)}
        if frame is not None:
            bow = t(f"prua a x = {num(frame.bow_x)}", f"bow at x = {num(frame.bow_x)}")
            out["frame"] = t(f"Chiglia {num(frame.length)} m, rotta {num(frame.heading, 1)}°, pendenza "
                             f"{num(frame.slope * 100, 1)} %, {bow}.",
                             f"Keel {num(frame.length)} m, heading {num(frame.heading, 1)}°, slope "
                             f"{num(frame.slope * 100, 1)} %, {bow}.")
            if chunk.meta.get(ship.KEY, {}).get("frame", {}).get("crs") != chunk.crs:
                out["frame"] += " " + t("Il sistema del chunk è cambiato: rifallo.", "The system of the chunk changed: set it again.")
        if stations:
            count = plural(len(stations), ("stazione", "stazioni"), ("station", "stations"))
            out["stations"] = t(f"{count}, da {stations[0]['name']} a {stations[-1]['name']}.",
                                f"{count}, from {stations[0]['name']} to {stations[-1]['name']}.")
        if data is not None:
            out["sections"] = lines.state_text(data)
            if lines.is_stale(chunk, data):
                out["sections"] += " " + t("Da ricalcolare: sistema della nave, modello o georeferenziazione cambiati.",
                                           "To compute again: ship frame, model or georeference changed.")
        water = lines.water_of(chunk)
        if water:
            on = t("correzione accesa", "correction on") if water["on"] else t("correzione spenta", "correction off")
            out["water"] = t(f"Livello z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.",
                             f"Level z {num(water['level'], 3)} m, n {num(water['n'], 2)}, {on}.")
        made = chunk.meta.get(ship.KEY, {}).get("drawings")
        if made:
            count = plural(len(made), ("disegno", "disegni"), ("drawing", "drawings"))
            out["drawings"] = t(f"{count} in shipwreck/drawings.", f"{count} in shipwreck/drawings.")
            old = drawings.out_of_date(chunk)
            if old:
                out["drawings"] += " " + t(f"Da rifare, non corrispondono più alle sezioni, all'acqua o ai legni: {', '.join(old)}.",
                                           f"To make again, they no longer match the sections, the water or the timbers: {', '.join(old)}.")
        records = timbers.timbers_of(chunk)
        if records:
            out["timbers"] = plural(len(records), ("legno", "legni"), ("timber", "timbers")) + "."
            try:
                wrong = [r for r in timbers.measure(chunk) if r["warning"]]
            except ship.NotReady:
                wrong = []
            if wrong:
                out["timbers"] += " " + t(f"Lato in disaccordo con y: {', '.join(r['name'] or r['id'] for r in wrong)}.",
                                          f"Side against y: {', '.join(r['name'] or r['id'] for r in wrong)}.")
        return out

    def _requested(self, name: str) -> None:
        self._guarded(getattr(self, "_do_" + name))

    # ------------------------------------------------------------------ 1 the frame
    def _do_frame_pick(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.check_metric(chunk)

        def finished(xy):
            self._guarded(self._set_picked, chunk, xy)

        self._need_view(chunk)
        self.context.request_line(finished, hint=t("Traccia la chiglia dal primo estremo (x = 0) all'altro; doppio clic o "
                                                   "Invio per finire", "Trace the keel from its first end (x = 0) to the "
                                                   "other; double click or Enter ends"))

    def _set_picked(self, chunk, xy) -> None:
        from . import ship

        old = ship.frame_of(chunk)
        new = ship.frame_from_picks(chunk, xy[0], xy[-1], old.bow if old else "start", old.z0 if old else None)
        ship.set_frame(chunk, ship.keeping_bow(old, new))
        self._changed()

    def _do_frame_markers(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        names = [(m.label, m.label) for m in chunk.markers if m.position is not None]
        if len(names) < 2:
            raise ship.NotReady(t("Servono due marker con una posizione (anche messi sul modello nella vista 3D).",
                                  "Two markers with a position are needed (also placed on the model in the 3D view)."))
        v = self._ask(t("Chiglia da due marker", "Keel from two markers"), [
            Field("a", t("Primo estremo (x = 0)", "First end (x = 0)"), names[0][0], choices=names),
            Field("b", t("Altro estremo", "Other end"), names[1][0], choices=names)])
        if v:
            old = ship.frame_of(chunk)
            new = ship.frame_from_markers(chunk, v["a"], v["b"], old.bow if old else "start", old.z0 if old else None)
            ship.set_frame(chunk, ship.keeping_bow(old, new),
                           markers=[ship.marker_label(chunk, v["a"]), ship.marker_label(chunk, v["b"])])
            self._changed()

    def _do_frame_options(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        v = self._ask(t("Sistema della nave", "Ship frame"), [
            Field("x0", t("x del primo estremo (m)", "x of the first end (m)"), frame.x0, minimum=-1e4, maximum=1e4, decimals=3,
                  tip=t("0 alla prua o a una stazione scelta", "0 at the stem or at a chosen station")),
            Field("z0", t("Quota del datum z = 0 (quota del chunk, m)", "Height of the datum z = 0 (height of the chunk, m)"),
                  frame.z0, minimum=-1e4, maximum=1e5, decimals=3)])
        if v:
            ship.set_frame(chunk, ship.Frame(frame.start, frame.end, frame.bow, v["z0"], v["x0"]),
                           markers=ship.frame_markers(chunk))
            self._changed()

    def _do_frame_flip(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.set_frame(chunk, ship.need_frame(chunk).flipped(), markers=ship.frame_markers(chunk, reverse=True))
        self._changed()

    def _do_frame_bow(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.set_frame(chunk, ship.need_frame(chunk).other_bow(), markers=ship.frame_markers(chunk))
        self._status(t("Babordo e tribordo scambiati: rifai sezioni e disegni.",
                       "Port and starboard exchanged: make the sections and drawings again."))
        self._changed()

    # ------------------------------------------------------------------ 2 the stations
    def _do_stations_regular(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        v = self._ask(t("Stazioni regolari", "Regular stations"), [
            Field("spacing", t("Distanza (m)", "Spacing (m)"), 1.0, minimum=0.01, maximum=1000.0, decimals=3),
            Field("count", t("Quante", "How many"), int(frame.length) + 1, minimum=1, maximum=1000, decimals=0),
            Field("start", t("Prima stazione, x (m)", "First station, x (m)"), frame.x0, minimum=-1e4, maximum=1e4, decimals=3)])
        if v:
            ship.set_stations(chunk, ship.regular(v["spacing"], int(v["count"]), v["start"]))
            self._changed()

    def _do_stations_table(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import ship

        chunk = self._need_chunk()
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Stazioni", "Stations"), "", "CSV (*.csv *.txt *.tsv)")
        if path:
            ship.set_stations(chunk, ship.stations_from_table(path))
            self._changed()

    def _do_stations_catalogue(self) -> None:
        from . import ship, timbers

        chunk = self._need_chunk()
        ship.set_stations(chunk, timbers.stations_from_catalogue(chunk))
        self._changed()

    # ------------------------------------------------------------------ 3 the sections
    def _do_sections(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        v = self._ask(t("Sezioni", "Sections"), [
            Field("longitudinals", t("Longitudinali, y (m)", "Longitudinals, y (m)"), "0", kind="text"),
            Field("horizontals", t("Orizzontali, z (m)", "Horizontals, z (m)"), "", kind="text"),
            Field("half_width", t("Metà larghezza (m, 0 = fin dove arriva il modello)", "Half width (m, 0 = as far as the model goes)"),
                  0.0, minimum=0.0, maximum=1e4, decimals=2)],
            intro=t("Valori separati da spazi o punto e virgola.", "Values separated by spaces or semicolons."))
        if v:
            self.sections(chunk, parse_values(v["longitudinals"]), parse_values(v["horizontals"]), v["half_width"])

    def sections(self, chunk, longitudinals, horizontals, half_width=None) -> None:
        from . import lines

        def done(data):
            self._status(t(f"{len(data['sections'])} sezioni calcolate", f"{len(data['sections'])} sections computed"))
            self._changed()

        self._run(t("Sezioni dello scafo", "Sections of the hull"),
                  lambda p: lines.build(chunk, longitudinals, horizontals, half_width, progress=p), done)

    # ------------------------------------------------------------------ 4 the water
    def _do_water(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import lines, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        water = lines.water_of(chunk) or {"level": 0.0, "n": lines.REFRACTION, "on": False}
        v = self._ask(t("Acqua", "Water"), [
            Field("level", t("Livello, z della nave (m)", "Level, ship z (m)"), water["level"], minimum=-1e4, maximum=1e4, decimals=3),
            Field("n", t("Indice di rifrazione", "Refractive index"), water["n"], minimum=1.0, maximum=2.0, decimals=3),
            Field("on", t("Correggi le quote sotto il livello", "Correct the heights under the level"), water["on"])],
            intro=t("Sotto il livello il fondo visto attraverso l'acqua è apparente: in vista nadirale la profondità vera è "
                    "circa quella apparente per n. È una prima approssimazione, dichiarata sui disegni.",
                    "Under the level the bottom seen through water is apparent: in a nadir view the true depth is about the "
                    "apparent one times n. A first approximation, stated on the drawings."))
        if v:
            lines.set_water(chunk, v["level"], v["n"], v["on"])
            self._changed()

    def _do_water_pick(self) -> None:
        from . import lines, ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        self._need_view(chunk)

        def finished(xy):
            ship_xy = frame.to_ship(np.r_[xy[0], 0.0])
            level = ship.Surface(lines.ship_mesh(chunk, frame).vertices).at(ship_xy[:, :2])[0]
            if not np.isfinite(level):
                self._message("warning", t("Il punto è fuori dal modello.", "The point lies off the model."))
                return
            water = lines.water_of(chunk) or {"n": lines.REFRACTION, "on": False}
            lines.set_water(chunk, float(level), water["n"], water["on"])
            self._changed()

        self.context.request_point(finished, hint=t("Clic su un punto del pelo dell'acqua", "Click a point of the waterline"))

    # ------------------------------------------------------------------ 5 the drawings
    def _do_drawings(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import drawings

        chunk = self._need_chunk()
        if drawings._traced(chunk) and not self._relief_still_right(chunk, t("Disegni", "Drawings")):
            return
        f = drawings.title_fields(chunk)
        v = self._ask(t("Disegni", "Drawings"), [
            Field("scale", t("Scala 1: (0 = automatica)", "Scale 1: (0 = automatic)"), 0, minimum=0, maximum=100000, decimals=0),
            Field("site", t("Sito", "Site"), f["site"], kind="text"),
            Field("vessel", t("Relitto", "Wreck"), f["vessel"], kind="text"),
            Field("surveyed", t("Rilievo", "Survey"), f["surveyed"], kind="text"),
            Field("drawn", t("Disegno", "Drawn by"), f["drawn"], kind="text"),
            Field("date", t("Data", "Date"), f["date"], kind="text"),
            Field("notes", t("Note", "Notes"), f["notes"], kind="text")] + _with_reconstruction(chunk, False))
        if not v:
            return
        scale = int(v.pop("scale")) or None
        reconstruction = bool(v.pop("reconstruction", False))
        self.drawings(chunk, scale, v, reconstruction)

    def drawings(self, chunk, scale=None, title=None, reconstruction: bool = False) -> None:
        """The drawings of the hull in the background; ``reconstruction``: with the reconstruction (a hypothesis) on
        the plan of the wreck, the body plan and the profile."""
        from . import drawings, ship

        if title:
            ship.state(chunk)["title"] = title
        if reconstruction:
            self._warn_stale(chunk)

        def done(paths):
            self._status(t(f"{len(paths)} file dei disegni in {paths[0].parent}", f"{len(paths)} drawing files in {paths[0].parent}"))
            self._changed()

        self._run(t("Disegni dello scafo", "Drawings of the hull"),
                  lambda p: drawings.write_all(chunk, scale=scale, progress=p, reconstruction=reconstruction), done)

    def _warn_stale(self, chunk) -> None:
        """A drawing, a table or an export with the reconstruction: said when it is out of date."""
        warning = reconstruction_warning(chunk)
        if warning:
            self._message("warning", warning)

    def _do_drawings_open(self) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        from . import ship

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(ship.folder(self._need_chunk()) / "drawings")))

    # ------------------------------------------------------------------ 6 the timbers
    def _timber(self, geometry: str, xy) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import timbers

        chunk = self._chunk()
        v = self._ask(t("Legno", "Timber"), [
            Field("kind", t("Tipo", "Kind"), "frame",
                  choices=[(k, timbers.kind_title(k)) for k in timbers.KINDS if k != "seam"]),  # seams: proposals
            Field("side", t("Lato", "Side"), "port", choices=[(k, timbers.side_title(k)) for k in timbers.SIDES]),
            Field("name", t("Nome", "Name"), "", kind="text", tip=t("Il nome del cantiere (F36)", "The name of the field (F36)")),
            Field("station", t("Stazione", "Station"), "", kind="text"), Field("notes", t("Note", "Notes"), "", kind="text")])
        if v:
            timbers.add(chunk, v["kind"], v["side"], v["name"], timbers.geometry(geometry, xy), v["station"], v["notes"])
            self._changed()

    def _do_timber_line(self) -> None:
        self._draw_timber("line", self.context.request_line)

    def _do_timber_polygon(self) -> None:
        self._draw_timber("polygon", self.context.request_polygon)

    def _do_timber_point(self) -> None:
        self._draw_timber("point", self.context.request_point)

    def _draw_timber(self, geometry: str, request) -> None:
        from . import ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._need_view(chunk)
        request(lambda xy: self._guarded(self._timber, geometry, xy), hint=t("Disegna il legno", "Draw the timber"))

    def _do_timber_import(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import timbers

        chunk = self._need_chunk()
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Legni", "Timbers"), "", "CSV (*.csv *.txt *.tsv)")
        if path:
            count = timbers.import_csv(chunk, path)
            self._status(t(f"Legni importati: {count}", f"Timbers imported: {count}"))
            self._changed()

    def _do_timber_remove(self) -> None:
        from . import ship, timbers, tracing

        ids = self.pane.selected_ids()
        if not ids:
            return
        chunk = self._need_chunk()
        before = self._snapshot(chunk)
        if timbers.remove(chunk, ids):
            try:
                tracing.update_visible(chunk)
            except ship.NotReady:
                pass
            self._step(chunk, before, t("legni rimossi", "timbers removed"))

    def _do_timber_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import timbers

        chunk = self._need_chunk()
        path, _ = QFileDialog.getSaveFileName(self.context.window, t("Esporta i legni", "Export the timbers"), "legni.xlsx",
                                              "Excel (*.xlsx);;CSV (*.csv);;GeoPackage (*.gpkg)")
        if path:
            self._status(str(timbers.export_any(chunk, path)))

    def _export(self, chunk, path, progress=None):
        from . import timbers

        return timbers.export_any(chunk, path)

    def _features_deleted(self, features) -> None:
        """Timbers deleted on the view (on the plan or on the relief): out of the catalogue, one undo step."""
        from . import ship, timbers, tracing

        chunk = self._chunk()
        if chunk is None:
            return
        ids = [f.get("ppg_id") or f.get("id") for f in features]
        before = self._snapshot(chunk)
        if timbers.remove(chunk, [i for i in ids if i]):
            try:
                tracing.update_visible(chunk)
            except ship.NotReady:
                pass
            self._step(chunk, before, t("legni eliminati", "timbers deleted"), deleted="timbers")

    # ------------------------------------------------------------------ 7 the clinker strakes
    def _do_clinker_detect(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import clinker, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        p = {**clinker.DEFAULTS, **(((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}).get("params") or {})}
        v = self._ask(t("Rileva i corsi", "Find the strakes"), [
            Field("min_step", t("Gradino minimo (mm)", "Least step (mm)"), p["min_step"] * 1000.0, minimum=1.0,
                  maximum=100.0, decimals=1, tip=t("Il salto più piccolo dove una fascia si sovrappone a quella sotto",
                                                   "The smallest step where a strake laps over the one below")),
            Field("min_strake", t("Larghezza minima di fascia (m)", "Least strake width (m)"), p["min_strake"],
                  minimum=0.01, maximum=2.0, decimals=3, tip=t("Due gradini più vicini di così sono uno solo",
                                                               "Two lands nearer than this are one")),
            Field("keel_half_width", t("Metà larghezza della chiglia (m)", "Half width of the keel (m)"),
                  p["keel_half_width"], minimum=0.01, maximum=1.0, decimals=3),
            Field("sole_width", t("Larghezza minima della suola (m)", "Least width of the sole (m)"), p["sole_width"],
                  minimum=0.001, maximum=0.5, decimals=3,
                  tip=t("Abbassala se una chiglia stretta e consumata dà lati non letti a ogni stazione",
                        "Lower it when a narrow worn keel leaves sides not read at every station"))],
            intro=t("I gradini del fasciame letti sulle sezioni alle stazioni (quelle grezze, senza la correzione "
                    "dell'acqua), dalla battura verso l'orlo, e uniti in linee lungo lo scafo. I gradini sono misurati; "
                    "le linee e il modello delle fasce sono un'ipotesi.",
                    "The plank lands read on the sections at the stations (the raw ones, without the water correction), "
                    "from the rabbet out to the sheer, and joined into lines along the hull. The lands are measured; the "
                    "lines and the strake model are a hypothesis."))
        if v:
            params = {k: float(v[k]) for k in ("min_strake", "keel_half_width", "sole_width")}
            params["min_step"] = float(v["min_step"]) / 1000.0
            self.detect(chunk, params)

    def detect(self, chunk, params=None) -> None:
        from . import clinker

        def job(p):
            result = clinker.detect_lands(chunk, params, progress=p)
            return result, lines_files(chunk)

        def done(found):
            result, files = found
            count = sum(1 for line in result["lines"] if len(line["points"]) > 1)
            self._status(t(f"Linee dei gradini: {count}", f"Land lines: {count}")
                         + (f" ({files['dxf'].parent})" if files else ""))
            self._changed()

        self._run(t("Fasce del fasciame", "Clinker strakes"), job, done)

    def _do_clinker_strakes(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import clinker

        chunk = self._need_chunk()
        state = clinker.check_strakes(chunk)
        before = (state.get("strakes") or [{}])[0]
        v = self._ask(t("Modello delle fasce", "Strake model"), [
            Field("lap", t("Sovrapposizione (mm)", "Lap (mm)"), before.get("lap_mm", 20.0), minimum=1.0, maximum=200.0,
                  decimals=1, tip=t("Quanto una fascia copre quella sotto", "How much a strake covers the one below")),
            Field("thickness", t("Spessore (mm, 0 = il gradino misurato)", "Thickness (mm, 0 = the step measured)"), 0.0,
                  minimum=0.0, maximum=200.0, decimals=1),
            Field("catalogue", t("Anche nel catalogo dei legni", "Into the timber catalogue too"), False)],
            intro=t("Fra due linee di gradini una fascia: larghezza a ogni stazione, lunghezza, spessore e "
                    "sovrapposizione, una mesh per fascia (OBJ e PLY) e le tabelle. È un'ipotesi dalle sezioni, non una "
                    "misura: è scritto in ogni file.",
                    "Between two land lines a strake: width at every station, length, thickness and lap, a mesh per "
                    "strake (OBJ and PLY) and the tables. It is a hypothesis from the sections, not a measure: every "
                    "file says so."))
        if v:
            self.strakes(chunk, float(v["lap"]) / 1000.0, float(v["thickness"]) / 1000.0 or None, bool(v["catalogue"]))

    def strakes(self, chunk, lap=0.02, thickness=None, catalogue=False) -> None:
        from . import clinker

        def job(p):
            result = clinker.build_strakes(chunk, lap, thickness, progress=p)
            result.update(clinker.write_tables(chunk))
            if catalogue:
                result["catalogue"] = clinker.to_catalogue(chunk)
            return result

        def done(result):
            count = plural(len(result["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
            self._status(t(f"Modello delle fasce (ipotesi): {count} in {result['obj'].parent}",
                           f"Strake model (hypothesis): {count} in {result['obj'].parent}"))
            self._changed()

        self._run(t("Modello delle fasce", "Strake model"), job, done)

    def _do_clinker_catalogue(self) -> None:
        from . import clinker

        count = clinker.to_catalogue(self._need_chunk())
        self._status(t(f"Fasce nel catalogo dei legni: {count}", f"Strakes in the timber catalogue: {count}"))
        self._changed()

    def _do_clinker_export(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import houdini, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        last = ((chunk.meta.get(ship.KEY) or {}).get("houdini") or {}).get("export")
        folder = QFileDialog.getExistingDirectory(self.context.window, t("Cartella per il Clinker Boat Tool",
                                                                         "Folder for the Clinker Boat Tool"),
                                                  str(Path(last).parent) if last else "")
        if not folder:
            return

        def done(out):
            self._status(t(f"File per il Clinker Boat Tool in {out['readme'].parent}",
                           f"Files for the Clinker Boat Tool in {out['readme'].parent}"))
            self._changed()

        self._run(t("File per il Clinker Boat Tool", "Files for the Clinker Boat Tool"),
                  lambda p: houdini.export(chunk, folder, progress=p), done)

    def _do_clinker_import(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import houdini, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        last = ((chunk.meta.get(ship.KEY) or {}).get("houdini") or {}).get("export") or ""
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Fasce da Houdini", "Planks from Houdini"), last,
                                              "OBJ (*.obj)")
        if not path:
            return
        v = self._ask(t("Fasce da Houdini", "Planks from Houdini"), [
            Field("side_from", t("Lato", "Side"), "y", choices=[
                ("y", t("Dalla y media dei vertici", "From the mean y of the vertices")),
                ("name", t("Dal nome (B/T, babordo/tribordo)", "From the name (B/T, port/starboard)"))]),
            Field("deviation", t("Distanza dalla scansione", "Distance from the scan"), True)],
            intro=t("Un oggetto (o un gruppo) del file per tavola, Y in alto come le esporta il Clinker Boat Tool. "
                    "Le tavole di un'importazione precedente dello stesso file vengono sostituite. Salva le tavole "
                    "dello strumento con un nome tuo, non strakes.obj o lands.obj: quelli il plugin li mette da parte "
                    "a ogni nuovo rilevamento.",
                    "One object (or group) of the file per plank, Y up as the Clinker Boat Tool exports them. The "
                    "planks of an earlier import of the same file are replaced. Save the tool's planks under a name "
                    "of your own, not strakes.obj or lands.obj: the plugin sets those aside at every new detection."))
        if not v:
            return

        def done(result):
            self._status(t(f"Tavole importate: {result['planks']} ({result['csv'].name})",
                           f"Planks imported: {result['planks']} ({result['csv'].name})"))
            self._changed()

        self._run(t("Fasce da Houdini", "Planks from Houdini"),
                  lambda p: houdini.import_planks(chunk, path, v["side_from"], bool(v["deviation"]), progress=p), done)

    def _do_clinker_open(self) -> None:
        from . import ship

        folder = ship.folder(self._need_chunk()) / "clinker"
        folder.mkdir(parents=True, exist_ok=True)
        self._open(folder)


    # ------------------------------------------------------------------ 8 the relief and the tracing
    def _relief_entry(self, chunk) -> dict:
        from . import relief, ship

        e = relief.entry(chunk)
        if e is None:
            raise ship.NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
        return e

    def _relief_shown(self, chunk) -> bool:
        from . import relief

        window = self.context.window if self.context else None
        e = relief.entry(chunk) if chunk is not None else None
        view = getattr(window, "ortho_view", None)
        return (e is not None and view is not None and view.tr is not None
                and getattr(window, "drawings_frame", None) == e["id"])

    def _leave_relief(self, chunk) -> None:
        """The plan in the view when the relief is shown: the drawings made on it stored first (a new relief carries
        them over to its new entry, so none may stay under the old one in the window's memory)."""
        if not self._relief_shown(chunk):
            return
        self.context.store_drawings()
        self._need_view(chunk)

    def _show_relief(self, chunk, kind: str | None = None) -> None:
        """The relief of the hull in the raster view (the program's elevation), with the grey raster ``kind`` over it
        (None: the one chosen last; "photo": none)."""
        from . import relief

        e = self._relief_entry(chunk)
        if kind is not None:
            self.overlay_kind = kind
        if not self._relief_shown(chunk):
            self.context.window.show_elevation(e)  # the layers and the overlay follow (on_raster_loaded)
        else:
            self.draw_layer()
        if relief.is_stale(chunk):
            self._status(t("Il rilievo dello scafo è di un altro sistema della nave o di un altro modello: rifallo.",
                           "The relief of the hull is of another ship frame or another model: build it again."))

    def _need_relief_view(self, chunk) -> None:
        if not self._relief_shown(chunk):
            self._show_relief(chunk)

    def _apply_overlay(self, chunk) -> None:
        from . import relief
        from .overlay import Overlay

        view = self.context.window.ortho_view
        if self.overlay is None or self.overlay.view is not view:
            self.overlay = Overlay(view)
        if self.overlay_kind not in relief.KINDS:
            self.overlay.remove()
            return
        self.overlay.show(relief.path_of(chunk, self.overlay_kind), self.overlay_kind, t(*OVERLAY_TITLES[self.overlay_kind]))

    def _do_relief_build(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import relief, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        ship.check_metric(chunk)
        old = (ship.state(chunk).get("relief") or {}).get("cell")
        cell = float(old or relief.default_cell(chunk))
        v = self._ask(t("Rilievo dello scafo", "Relief of the hull"), [
            Field("cell", t("Cella (mm)", "Cell (mm)"), round(cell * 1000.0, 1), minimum=0.5, maximum=1000.0, decimals=1,
                  tip=t("Dalla densità del modello; più piccola, più fine e più pesante",
                        "From the model's density; smaller, finer and heavier"))],
            intro=t("Lo scafo visto dall'alto nel sistema della nave (u = x, v = y verso sinistra), come un prospetto "
                    "del programma: le sue altezze, l'ombreggiatura e il rilievo locale. Si disegna sopra con gli "
                    "strumenti del programma; i disegni restano con il rilievo anche quando lo rifai.",
                    "The hull seen from above in the ship frame (u = x, v = y to the left), as an elevation of the "
                    "program: its heights, hillshade and local relief. Draw on it with the program's tools; the "
                    "drawings stay with the relief when you build it again."))
        if v:
            self.build_relief(chunk, float(v["cell"]) / 1000.0)

    def build_relief(self, chunk, cell=None) -> None:
        from . import relief

        if cell:
            relief.check_size(chunk, cell)  # too many cells: said before the work starts
        self._leave_relief(chunk)

        def done(st):
            self._status(t(f"Rilievo dello scafo: celle di {num(st['cell'] * 1000.0, 1)} mm",
                           f"Relief of the hull: cells of {num(st['cell'] * 1000.0, 1)} mm"))
            self._changed()
            self._guarded(self._show_relief, chunk)  # the NEW entry: the old one is gone

        self._run(t("Rilievo dello scafo", "Relief of the hull"), lambda p: relief.build(chunk, cell, progress=p), done)

    def _do_relief_show(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        chunk = self._need_chunk()
        self._relief_entry(chunk)
        kinds = [("photo", t("Solo l'immagine del rilievo", "The relief's image alone")),
                 ("height", t("Altezze", "Heights")), ("hillshade", t("Ombreggiatura", "Hillshade")),
                 ("slrm", t("Rilievo locale", "Local relief"))]
        v = self._ask(t("Mostra il rilievo", "Show the relief"), [
            Field("kind", t("Sopra", "Over it"), self.overlay_kind, choices=kinds)],
            intro=t("In grigio, dal 2 al 98 percentile, sopra l'immagine del rilievo e sotto i disegni.",
                    "In grey, from the 2nd to the 98th percentile, over the relief's image and under the drawings."))
        if v:
            self._show_relief(chunk, v["kind"])

    def _do_proposals_build(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import proposals, relief, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._relief_entry(chunk)
        if relief.is_stale(chunk):
            raise ship.NotReady(t("Il rilievo dello scafo è di un altro sistema della nave o di un altro modello: "
                                  "costruiscilo di nuovo.",
                                  "The relief of the hull is of another ship frame or another model: build it again."))
        p = {**proposals.DEFAULTS, **((ship.state(chunk).get("proposals") or {}).get("params") or {})}
        v = self._ask(t("Proponi", "Propose"), [
            Field("view", t("Il modello mostra", "The model shows"), p["view"], choices=[
                ("inside", t("L'interno dello scafo (relitto sul fondo)", "The inside of the hull (a wreck on its bottom)")),
                ("outside", t("L'esterno dello scafo (scafo sollevato)", "The outside of the hull (a lifted hull)"))],
                tip=t("Decide il verso del gradino di un comento: con il lato sbagliato ogni comento è un dubbio",
                      "It decides the sign of a land's step: with the wrong face every land is a stray")),
            Field("min_step", t("Gradino minimo dei comenti (mm)", "Least step of a seam (mm)"), p["min_step"] * 1000.0,
                  minimum=0.5, maximum=100.0, decimals=1),
            Field("opening", t("Apertura (m)", "Opening (m)"), p["opening"], minimum=0.02, maximum=5.0, decimals=2,
                  tip=t("Più lunga del vano fra due ordinate, più corta del fasciame fra due ordinate",
                        "Longer than the room between two frames, shorter than the planking between two frames")),
            Field("top_hat", t("Altezza minima di un legno sopra il fasciame (mm)", "Least height of a timber over the "
                               "planking (mm)"), p["top_hat"] * 1000.0, minimum=1.0, maximum=500.0, decimals=1)],
            intro=t("Comenti, ordinate, paramezzale, serrette e fasce letti sul rilievo: proposte da accettare, "
                    "correggere o scartare. Una nuova lettura non tocca i legni del catalogo e non ripropone ciò che "
                    "è stato accettato o scartato.",
                    "Seams, frames, keelson, stringers and strakes read on the relief: proposals to accept, correct "
                    "or reject. A new reading never touches the catalogue's timbers and never proposes again what "
                    "was accepted or rejected."))
        if not v:
            return
        params = {"view": v["view"], "min_step": float(v["min_step"]) / 1000.0, "opening": float(v["opening"]),
                  "top_hat": float(v["top_hat"]) / 1000.0}
        self.propose(chunk, params)

    def propose(self, chunk, params=None) -> None:
        from . import proposals

        before = self._snapshot(chunk)

        def done(_prop):
            self._status(proposals.summary(chunk))
            self._step(chunk, before, t("proposte lette", "proposals read"))
            self._guarded(self._need_relief_view, chunk)

        self._run(t("Proposte dal rilievo", "Proposals from the relief"),
                  lambda p: proposals.build(chunk, params, progress=p), done)

    def _selected(self, *layers) -> list[str]:
        """The ids of the features selected on the view in ``layers`` (``ppg_id``, the plan's ``id``)."""
        view = getattr(self.context.window, "ortho_view", None)
        if view is None:
            return []
        out = []
        for layer, _idx, attrs in view.selected_features():
            key = attrs.get("ppg_id") or attrs.get("id")
            if layer in layers and key and key not in out:
                out.append(key)
        return out

    def _selected_timbers(self) -> list[str]:
        """The timbers selected on the view; when none, those selected in the pane's table (a row left selected
        there never adds to what the user points at on the view)."""
        ids = self._selected(TRACED_LAYER, LAYER)
        if ids:
            return ids
        return self.pane.selected_ids() if self.pane is not None else []

    def _do_trace_accept(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import proposals, ship, timbers, tracing

        chunk = self._need_chunk()
        prop = ship.state(chunk).get("proposals") or {}
        if not prop.get("items"):
            raise ship.NotReady(t("Nessuna proposta da accettare: leggi prima le proposte sul rilievo.",
                                  "No proposal to accept: read the proposals on the relief first."))
        ids = self._selected(PROPOSALS_LAYER)
        if not ids:
            kinds = [("all", t("Tutte", "All"))] + [(k, timbers.kind_title(k)) for k in proposals.KINDS]
            v = self._ask(t("Accetta le proposte", "Accept the proposals"), [
                Field("kind", t("Tipo", "Kind"), "all", choices=kinds),
                Field("score", t("Punteggio minimo", "Least score"), 0.5, minimum=0.0, maximum=1.0, decimals=2,
                      tip=t("I comenti dubbi (comento B?, T?) hanno 0,1 o meno", "The doubtful seams (B?, T?) have "
                            "0.1 or less"))],
                intro=t("Nessuna proposta selezionata sulla vista: quali accettare?",
                        "No proposal selected on the view: which ones to accept?"))
            if not v:
                return
            ids = [i["id"] for i in prop["items"] if v["kind"] in ("all", i["kind"])
                   and float(i.get("score") or 0.0) >= float(v["score"])]
            if not ids:
                raise ship.NotReady(t("Nessuna proposta così.", "No such proposal."))
        before = self._snapshot(chunk)
        records = tracing.accept(chunk, ids)
        tracing.update_visible(chunk)
        self._status(plural(len(records), ("legno accettato", "legni accettati"), ("timber accepted", "timbers accepted")))
        self._step(chunk, before, t("proposte accettate", "proposals accepted"))

    def _do_trace_reject(self) -> None:
        from . import ship

        chunk = self._need_chunk()
        ids = self._selected(PROPOSALS_LAYER)
        if not ids:
            raise ship.NotReady(t("Seleziona sulla vista le proposte da scartare (strumento Seleziona).",
                                  "Select on the view the proposals to reject (the Select tool)."))
        self.reject(chunk, ids)

    def reject(self, chunk, ids, deleted: str | None = None) -> None:
        from . import tracing

        before = self._snapshot(chunk)
        count = tracing.reject(chunk, ids)
        self._status(plural(count, ("proposta scartata", "proposte scartate"), ("proposal rejected", "proposals rejected")))
        self._step(chunk, before, t("proposte scartate", "proposals rejected"), deleted=deleted)

    def _proposals_deleted(self, features) -> None:
        """Proposals deleted on the view: rejected (a stale relief or frame refuses it: said, and they come back)."""
        from . import ship

        chunk = self._chunk()
        ids = [f.get("ppg_id") for f in features if f.get("ppg_id")]
        if chunk is None or not ids:
            return
        known = {i["id"] for i in (ship.state(chunk).get("proposals") or {}).get("items", [])}
        ids = [i for i in ids if i in known]
        try:
            if ids:
                self.reject(chunk, ids, deleted="proposals")
        except ship.NotReady as exc:
            self._message("warning", str(exc))
            self.draw_layer()

    def _request(self, geometry: str):
        return {"polygon": self.context.request_polygon, "line": self.context.request_line,
                "point": self.context.request_point}[geometry]

    def _do_trace_correct(self) -> None:
        from . import ship, tracing

        chunk = self._need_chunk()
        ids = self._selected_timbers()
        if len(ids) != 1:
            raise ship.NotReady(t("Seleziona un legno (sulla vista o nella tabella) da ridisegnare.",
                                  "Select one timber (on the view or in the table) to trace again."))
        record = tracing.record(chunk, ids[0])
        tracing.need_fresh_relief(chunk)
        self._need_relief_view(chunk)
        gtype = record["geometry"]["type"]

        def finished(uv):
            before = self._snapshot(chunk)
            tracing.correct(chunk, record["id"], uv)
            self._step(chunk, before, t(f"{record['name'] or record['id']} corretto",
                                        f"{record['name'] or record['id']} corrected"))

        name = record["name"] or record["id"]
        self._request(gtype)(lambda uv: self._guarded(finished, uv),
                             hint=t(f"Ridisegna {name} sul rilievo", f"Trace {name} again on the relief"))

    def _do_trace_add(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship, timbers, tracing

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._relief_entry(chunk)
        tracing.need_fresh_relief(chunk)
        v = self._ask(t("Legno ricalcato", "Traced timber"), [
            Field("kind", t("Tipo", "Kind"), "frame", choices=[(k, timbers.kind_title(k)) for k in timbers.KINDS]),
            Field("side", t("Lato", "Side"), "auto", choices=[("auto", t("Dalla posizione", "From where it lies"))]
                  + [(k, timbers.side_title(k)) for k in timbers.SIDES]),
            Field("name", t("Nome", "Name"), "", kind="text", tip=t("Il nome del cantiere (F36)", "The name of the field (F36)")),
            Field("geometry", t("Disegno", "Drawing"), "polygon", choices=[
                ("polygon", t("Poligono", "Polygon")), ("line", t("Linea", "Line")), ("point", t("Punto", "Point"))],
                tip=t("Un comento è sempre una linea", "A seam is always a line"))])
        if not v:
            return
        gtype = "line" if v["kind"] == "seam" else v["geometry"]
        self._need_relief_view(chunk)

        def finished(uv):
            before = self._snapshot(chunk)
            record = tracing.add_traced(chunk, v["kind"], v["side"], v["name"], uv, gtype)
            tracing.update_visible(chunk)
            self._step(chunk, before, t(f"{record['name'] or record['id']} ricalcato",
                                        f"{record['name'] or record['id']} traced"))

        self._request(gtype)(lambda uv: self._guarded(finished, uv),
                             hint=t("Disegna il legno sul rilievo", "Trace the timber on the relief"))

    def _do_trace_names(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import tracing

        chunk = self._need_chunk()
        path, _ = QFileDialog.getOpenFileName(self.context.window, t("Nomi delle ordinate", "Names of the frames"), "",
                                              "CSV (*.csv *.txt *.tsv)")
        if not path:
            return
        before = self._snapshot(chunk)
        named, unmatched = tracing.rename_frames(chunk, path)
        self._status(t(f"Ordinate rinominate: {named}; righe senza ordinata: {unmatched}",
                       f"Frames renamed: {named}; rows matching no frame: {unmatched}"))
        self._step(chunk, before, t("nomi delle ordinate", "names of the frames"))

    def _do_trace_loose(self) -> None:
        from . import ship, timbers, tracing

        chunk = self._need_chunk()
        ids = self._selected_timbers()
        if not ids:
            raise ship.NotReady(t("Seleziona i legni (sulla vista o nella tabella).",
                                  "Select the timbers (on the view or in the table)."))
        records = [r for r in timbers.timbers_of(chunk) if r["id"] in set(ids)]
        flag = not all(r.get("loose") for r in records)
        before = self._snapshot(chunk)
        tracing.loose(chunk, ids, flag)
        tracing.update_visible(chunk)
        self._step(chunk, before, t("legni sciolti", "loose timbers") if flag else
                   t("legni non più sciolti", "timbers no longer loose"))

    # ------------------------------------------------------------------ 8 the keel, the timbers in 3D, the plan
    def _do_keel_set(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship, solids

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        keel = ship.state(chunk).get("keel") or {}
        v = self._ask(t("Chiglia (dimensioni)", "Keel (dimensions)"), [
            Field("siding", t("Larghezza (mm)", "Siding (mm)"), round(keel.get("siding", 0.15) * 1000.0, 1),
                  minimum=1.0, maximum=2000.0, decimals=1, tip=t("Di traverso", "Across")),
            Field("moulding", t("Altezza (mm)", "Moulding (mm)"), round(keel.get("moulding", 0.20) * 1000.0, 1),
                  minimum=1.0, maximum=2000.0, decimals=1, tip=t("In verticale, sotto il paramezzale",
                                                                 "Upright, under the keelson")),
            Field("auto", t("Lungo tutto il paramezzale (senza: tutto il rilievo)", "Along the whole keelson (without "
                            "one: the whole relief)"), not keel),
            Field("x0", t("Altrimenti da x (m)", "Otherwise from x (m)"), float(keel.get("x0", 0.0)), minimum=-1e4,
                  maximum=1e4, decimals=3),
            Field("x1", t("a x (m)", "to x (m)"), float(keel.get("x1", 0.0)), minimum=-1e4, maximum=1e4, decimals=3),
            Field("remove", t("Nessuna chiglia (toglie quella data)", "No keel (removes the given one)"), False)],
            intro=t("Il modello mostra l'interno dello scafo: la chiglia non si vede. Con larghezza e altezza date "
                    "diventa un parallelepipedo lungo la linea di chiglia sotto il paramezzale, stato «ricostruito: "
                    "dimensioni date»: un'ipotesi, campita nei disegni e trasparente nei file 3D. Un legno di chiglia "
                    "ricalcato (un chunk che mostra l'esterno) la sostituisce. Arriva con i prossimi Legni 3D.",
                    "The model shows the inside of the hull: the keel is not seen. With a given siding and moulding it "
                    "becomes a box along the keel line under the keelson, status 'reconstructed: given dimensions': a "
                    "hypothesis, hatched in the drawings and translucent in the 3D files. A traced keel record (a chunk "
                    "showing the outside) replaces it. It comes with the next 3D timbers."))
        if not v:
            return
        before = self._snapshot(chunk)
        if v["remove"]:
            solids.set_keel(chunk, None, None)
            text = t("chiglia tolta", "keel removed")
        else:
            x0, x1 = (None, None) if v["auto"] else (float(v["x0"]), float(v["x1"]))
            solids.set_keel(chunk, float(v["siding"]) / 1000.0, float(v["moulding"]) / 1000.0, x0, x1)
            text = t("chiglia data", "keel given")
        self._step(chunk, before, text)

    def _do_solids_build(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship, tracing

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._relief_entry(chunk)
        tracing.need_fresh_relief(chunk)
        v = self._ask(t("Legni in 3D", "Timbers in 3D"), [
            Field("thickness", t("Spessore delle fasce che non ne hanno uno (mm, 0 = nessuno)", "Thickness of the "
                                 "strakes that have none (mm, 0 = none)"), 0.0, minimum=0.0, maximum=500.0, decimals=1,
                  tip=t("Lo spessore di una fascia viene dal suo record, dal modello delle fasce o dai gradini dei "
                        "comenti; questo vale solo per quelle senza", "A strake's thickness comes from its record, the "
                        "strake model or the steps of the seams; this one only for those without"))],
            intro=t("Ogni legno ricalcato come poligono diventa un oggetto 3D chiuso nel sistema della nave: una "
                    "fascia è la superficie del modello con il suo spessore (sotto le ordinate interpolata e segnata "
                    "nascosta), un'ordinata o un legno longitudinale un solido dal fasciame sotto di lui alla sua "
                    "cima. Ricalca prima tutti i legni che toccano quello che costruisci: uno non ricalcato finisce "
                    "nell'anello da cui si legge il fasciame sotto.",
                    "Every timber traced as a polygon becomes a closed 3D object in the ship frame: a strake is the "
                    "model's surface with its thickness (interpolated under the frames and flagged hidden), a frame or "
                    "a longitudinal timber a solid from the planking under it to its top. Trace first every timber "
                    "touching the one you build: an untraced one ends up in the ring the planking under it is read "
                    "from."))
        if v:
            self.build_solids(chunk, float(v["thickness"]) or None)

    def build_solids(self, chunk, thickness=None) -> None:
        from . import solids

        def done(out):
            self._status(solids_summary(out))
            if out["refused"]:
                self._message("warning", solids_summary(out) + "\n" + "\n".join(
                    f"{r['name']}: {r['reason']}" for r in out["refused"]))
            self._changed()

        def job(p):
            try:
                return solids.build(chunk, thickness, progress=p)
            finally:  # also cancelled: files and meta changed, the pane and the tree say what was built ("interrotto")
                self.context.call_in_gui(self.context.mark_modified)
                self.context.call_in_gui(self.refresh_pane)
                self.context.call_in_gui(self.context.refresh_tree)

        self._run(t("Legni in 3D", "Timbers in 3D"), job, done)

    def _do_model_export(self, reconstruction: bool = False) -> None:
        from PySide6.QtWidgets import QFileDialog

        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import reconstruct, ship

        chunk = self._need_chunk()
        if not ship.state(chunk).get("solids"):
            raise ship.NotReady(t("Costruisci prima i legni in 3D.", "Build the timbers in 3D first."))
        if reconstruction and not _has_reconstruction(chunk):
            raise ship.NotReady(t("Ricostruisci prima (passo 9).", "Reconstruct first (step 9)."))
        title = (t("Esporta 3D con la ricostruzione", "Export 3D with the reconstruction") if reconstruction
                 else t("Esporta 3D", "Export 3D"))
        v = self._ask(title, [
            Field("system", t("Coordinate", "Coordinates"), "ship", choices=[
                ("ship", t("Della nave (x lungo la chiglia, z in alto dal datum)", "Of the ship (x along the keel, "
                                                                                    "z up from the datum)")),
                ("world", t("Del chunk (per GIS e BIM)", "Of the chunk (for GIS and BIM)"))]),
            Field("tables", t("Anche le tabelle dei legni in 3D (CSV, Excel)", "The tables of the timbers in 3D too "
                              "(CSV, Excel)"), True)] + _with_reconstruction(chunk, reconstruction),
            intro=t("Un oggetto per legno con il suo nome: OBJ (un materiale per tipo), GLB (un nodo per legno), IFC "
                    "(un elemento per legno con Pset_PyArchInit_Timber). Le ipotesi finiscono in _ric, più chiare e "
                    "trasparenti. Un IFC di uno scafo lungo può richiedere decine di secondi.",
                    "One object per timber with its name: OBJ (a material per kind), GLB (a node per timber), IFC (an "
                    "element per timber with Pset_PyArchInit_Timber). Hypotheses end in _ric, lighter and translucent. "
                    "An IFC of a long hull may take tens of seconds.")
            + (t(" Con la ricostruzione: i legni ricostruiti e la carenatura, e la tabella del misurato e del "
                 "ricostruito accanto.", " With the reconstruction: the reconstructed timbers and the faired hull, and "
                 "the table of the measured and the reconstructed beside.") if reconstruction else ""))
        if not v:
            return
        reconstruction = bool(v.get("reconstruction", reconstruction))
        last = ship.state(chunk).get("model_export") or (str(reconstruct.folder(chunk).parent / "ricostruzione.obj")
                                                         if reconstruction else "legni.obj")
        path, _ = QFileDialog.getSaveFileName(self.context.window, title, str(last),
                                              "OBJ (*.obj);;GLB (*.glb);;IFC (*.ifc)")
        if path:
            self.export_model(chunk, path, v["system"], bool(v["tables"]), reconstruction)

    def export_model(self, chunk, path, system: str = "ship", tables: bool = True, reconstruction: bool = False) -> None:
        path = Path(path)
        check_export_path(path)
        warning = solids_warning(chunk)
        if warning:
            self._message("warning", warning)
        if reconstruction:
            from . import models3d

            models3d.check_reconstruction(chunk)
            self._warn_stale(chunk)
        fmt = path.suffix[1:].upper()

        def job(p):
            p(0.0, t(f"Scrittura {fmt}...", f"Writing {fmt}..."))  # the writers have no progress of their own
            out = export_3d(chunk, path, system, tables, remember=False,  # the chunk's meta: in done, not here
                            reconstruction=reconstruction)
            p(1.0, str(out["path"]))
            return out

        def done(out):
            from . import ship

            ship.state(chunk)["model_export"] = str(out["path"])
            self._status(t(f"Legni in 3D: {out['path']}", f"Timbers in 3D: {out['path']}"))
            if out.get("note"):
                self._message("information", out["note"])
            self._changed()

        self._run(t(f"Legni in 3D: {fmt}", f"Timbers in 3D: {fmt}"), job, done)

    def _relief_still_right(self, chunk, title: str) -> bool:
        """A stale relief before a drawing of the traced timbers: the user rebuilds it (False: the drawing waits),
        draws anyway (True) or cancels (False)."""
        from . import relief

        if relief.entry(chunk) is None or not relief.is_stale(chunk):
            return True
        k = self._choose(title, t("Il rilievo dello scafo è di un altro sistema della nave o di un altro modello: lo "
                                  "sfondo sarebbe sbagliato e sul profilo mancherebbero paramezzale e chiglia. Rifarlo "
                                  "prima?",
                                  "The relief of the hull is of another ship frame or another model: the background "
                                  "would be wrong and the profile would lack the keelson and the keel. Build it again "
                                  "first?"),
                         [t("Rifai il rilievo", "Build the relief again"), t("Disegna lo stesso", "Draw anyway")])
        if k == 0:
            self._do_relief_build()
            self._message("information", t("Il disegno non è stato fatto: rifallo quando il rilievo è pronto.",
                                           "The drawing was not made: make it again when the relief is ready."))
        return k == 1

    def _do_wreck_plan(self, reconstruction: bool = False) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import drawings, ship

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        if reconstruction and not _has_reconstruction(chunk):
            raise ship.NotReady(t("Ricostruisci prima (passo 9).", "Reconstruct first (step 9)."))
        title = (t("Pianta con ricostruzione", "Plan with reconstruction") if reconstruction
                 else t("Pianta del relitto", "Plan of the wreck"))
        if not self._relief_still_right(chunk, title):
            return
        v = self._ask(title, [
            Field("background", t("Sotto i legni", "Under the timbers"), "relief", choices=[
                ("relief", t("L'ombreggiatura del rilievo dello scafo", "The hillshade of the relief of the hull")),
                ("ortho", t("L'ortofoto", "The orthophoto")), ("none", t("Niente", "Nothing"))]),
            Field("scale", t("Scala 1: (0 = automatica)", "Scale 1: (0 = automatic)"), 20, minimum=0, maximum=100000,
                  decimals=0)] + _with_reconstruction(chunk, reconstruction),
            intro=t("Ogni legno del catalogo contornato, colorato per tipo e nominato; le ipotesi campite a righe "
                    "oblique; legenda, scala, nord, cartiglio. SVG, PDF, PNG e DXF (un layer per tipo) in "
                    "shipwreck/drawings.",
                    "Every timber of the catalogue outlined, coloured by kind and named; the hypotheses hatched; "
                    "legend, scale, north, title block. SVG, PDF, PNG and DXF (a layer per kind) in shipwreck/drawings.")
            + (t(" Con la ricostruzione: una campitura per modo (simmetria ////, prolungamento \\\\, ordinate al "
                 "passo xxxx, tavole importate ----), la carenatura a trattini.", " With the reconstruction: a hatch "
                 "per way (mirror ////, extension \\\\, frames at the spacing xxxx, imported planks ----), the faired "
                 "hull dashed.") if reconstruction else ""))
        if not v:
            return
        scale = int(v["scale"]) or None
        reconstruction = bool(v.get("reconstruction", reconstruction))
        if reconstruction:
            self._warn_stale(chunk)

        def done(paths):
            self._status(t(f"Pianta del relitto: {paths[0].parent}", f"Plan of the wreck: {paths[0].parent}"))
            self._changed()

        self._run(title, lambda p: drawings.write_all(chunk, kinds=("wreck_plan",), scale=scale, progress=p,
                                                      background=v["background"], reconstruction=reconstruction), done)

    # ------------------------------------------------------------------ 9 the reconstruction
    def _rec_step(self, chunk, before, text: str, before_dir=None) -> None:
        """After a build or the ends set: an entry of *Modifica > Annulla* (meta and, for a build, the folder before
        and after), then the project modified and the pane, the tree and the layers shown again."""
        from . import ship

        window = self.context.window
        history = getattr(window, "history", None) if window is not None else None
        after = copy.deepcopy(ship.state(chunk).get("reconstruction"))
        if (history is not None and not getattr(window, "_script_running", False) and after != before
                and chunk is getattr(window, "chunk", None)):
            after_dir = self._keep_folder(chunk) if before_dir is not None else None
            if after_dir is not None:
                _LAST_AFTER[chunk.key] = (after_dir, copy.deepcopy(after))
            history.push(t(f"Relitto: {text}", f"Shipwreck: {text}"),
                         _RebuildStep(self, chunk, before, after, before_dir, after_dir), document=False)
        else:
            _forget_copy(chunk, before_dir)  # no step holds it
        self._changed()

    def _keep_folder(self, chunk, reuse: bool = False):
        """A copy of the reconstruction's folder (hard links) for a step of Annulla. ``reuse``: the copy taken after
        the last build when the reconstruction has not changed since (the "before" of a build is the "after" of the
        one before it: one copy, not two)."""
        import uuid

        from . import ship

        last = _LAST_AFTER.get(chunk.key)
        if (reuse and last is not None and last[0].exists()
                and last[1] == ship.state(chunk).get("reconstruction")):
            return last[0]
        root = _rec_folder(chunk).parent / UNDO_FOLDER
        root.mkdir(parents=True, exist_ok=True)
        for old in root.iterdir():  # an earlier session's (the project may be opened after the plugin started)
            if old not in _UNDO_DIRS:
                _remove(old)
        dst = root / uuid.uuid4().hex
        _copy_folder(_rec_folder(chunk), dst)
        _UNDO_DIRS.add(dst)
        return dst

    @staticmethod
    def _clean_undo_copies(doc) -> None:
        """The copies for Annulla left by an earlier session (or a build that failed) removed: only those the steps
        of this process hold stay."""
        for chunk in getattr(doc, "chunks", None) or []:
            root = chunk.data_dir / "shipwreck" / UNDO_FOLDER
            if root.exists():
                for old in root.iterdir():
                    if old not in _UNDO_DIRS:
                        _remove(old)

    def _reconstruction_deleted(self, features) -> None:
        """The reconstruction is not edited by hand on the view: it comes back, and the user is told how."""
        self._message("information", t("La ricostruzione non si modifica a mano: rimuovila dal pannello (Ricostruisci, "
                                       "senza quel passo) o con Modifica > Annulla, oppure correggi il ricalco.",
                                       "The reconstruction is not edited by hand: remove it from the pane (Reconstruct, "
                                       "without that step) or with Edit > Undo, or correct the tracing."))
        from PySide6.QtCore import QTimer

        QTimer.singleShot(0, self.draw_layer)

    def _do_rec_ends(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import reconstruct, ship

        chunk = self._need_chunk()
        frame = ship.need_frame(chunk)
        ends = (ship.state(chunk).get("reconstruction") or {}).get("ends") or {}
        try:
            lo, hi = reconstruct.surveyed_range(chunk)
        except ship.NotReady:
            lo, hi = sorted((frame.x0, frame.x0 + frame.length))
        near_lo = abs(frame.bow_x - lo) <= abs(frame.bow_x - hi)  # the bow's end: the stem
        stem0, stern0 = (lo - 0.5, hi + 0.5) if near_lo else (hi + 0.5, lo - 0.5)

        def x_of(key, default):
            e = ends.get(key) or {}
            if "x" in e:
                return float(e["x"])
            if "line" in e:
                return float(np.mean(np.asarray(e["line"], float)[:, 0]))
            return default

        v = self._ask(t("Prua e poppa", "Stem and stern"), [
            Field("stem_on", t("Prua data", "Stem given"), "stem" in ends),
            Field("stem", t("x della prua (m)", "x of the stem (m)"), round(x_of("stem", stem0), 3), minimum=-1e4,
                  maximum=1e4, decimals=3),
            Field("stern_on", t("Poppa data", "Stern given"), "stern" in ends),
            Field("stern", t("x della poppa (m)", "x of the stern (m)"), round(x_of("stern", stern0), 3),
                  minimum=-1e4, maximum=1e4, decimals=3)],
            intro=t(f"Senza estremità le fasce si prolungano solo entro la lunghezza rilevata (x {num(lo)}-{num(hi)} m) "
                    "e la carenatura resta aperta all'ultima stazione. Un'estremità data (una x della nave) le lascia "
                    "andare fin lì: un'ipotesi, mai una misura.",
                    f"Without ends the strakes are extended only within the surveyed length (x {num(lo)}-{num(hi)} m) "
                    "and the faired hull stays open at the last station. An end given (a ship x) lets them go so far: "
                    "a hypothesis, never a measure."))
        if not v:
            return
        before = copy.deepcopy(ship.state(chunk).get("reconstruction"))
        reconstruct.set_ends(chunk, float(v["stem"]) if v["stem_on"] else None,
                             float(v["stern"]) if v["stern_on"] else None)
        if not ship.state(chunk).get("reconstruction"):
            ship.state(chunk).pop("reconstruction", None)
        self._rec_step(chunk, before, t("prua e poppa", "stem and stern"))

    def _do_rec_build(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import ship, tracing

        chunk = self._need_chunk()
        ship.need_frame(chunk)
        self._relief_entry(chunk)
        tracing.need_fresh_relief(chunk)
        done = (ship.state(chunk).get("reconstruction") or {}).get("params", {}).get("steps") or REBUILD_STEPS
        v = self._ask(t("Ricostruisci", "Reconstruct"), [
            Field("mirror", t("Simmetria del lato conservato", "Mirror of the preserved side"), "mirror" in done),
            Field("extension", t("Prolungamento delle fasce", "Extension of the strakes"), "extension" in done),
            Field("frames", t("Ordinate al passo misurato", "Frames at the measured spacing"), "frames" in done),
            Field("loft", t("Carenatura (dalle sezioni alle stazioni)", "Faired hull (from the stations' sections)"),
                  "loft" in done)],
            intro=t("Ogni parte è un'ipotesi, campita nei disegni e trasparente nei file 3D, mai una misura: il "
                    "catalogo non cambia. Gli altri chunk dello stesso relitto (allineati o con marker degli stessi "
                    "nomi) entrano da soli. Si può annullare (Modifica > Annulla).",
                    "Every part is a hypothesis, hatched in the drawings and translucent in the 3D files, never a "
                    "measure: the catalogue does not change. The other chunks of the same wreck (aligned, or with "
                    "markers of the same names) come in by themselves. It can be undone (Edit > Undo)."))
        if not v:
            return
        steps = [s for s in REBUILD_STEPS if v.get(s)]
        self.rebuild(chunk, steps)

    def rebuild(self, chunk, steps=None) -> None:
        """The reconstruction built in the background (:func:`rebuild.build`), a step of Annulla when it is done."""
        from . import rebuild, ship

        before = copy.deepcopy(ship.state(chunk).get("reconstruction"))
        window = self.context.window
        before_dir = self._keep_folder(chunk, reuse=True) if getattr(window, "history", None) is not None else None
        doc = getattr(chunk, "document", None)

        def done(out):
            text = reconstruction_summary(out)
            self._status(text.splitlines()[0])
            if out["refused"] or out["warnings"]:
                self._message("warning", text)
            self._rec_step(chunk, before, t("ricostruzione", "reconstruction"), before_dir)

        def job(p):
            try:
                return rebuild.build(chunk, progress=p, doc=doc, steps=steps or REBUILD_STEPS)
            except BaseException:  # a build that failed or was cancelled: its "before" copy is held by nothing
                _forget_copy(chunk, before_dir)
                raise

        self._run(t("Ricostruzione (ipotesi)", "Reconstruction (hypothesis)"), job, done)

    def _do_rec_frame_of(self) -> None:
        from pyarchinit_photogrammetry.gui.dialogs import Field

        from . import chunks, ship

        chunk = self._need_chunk()
        doc = getattr(chunk, "document", None)
        others = [c for c in (doc.chunks if doc is not None else []) if c is not chunk and c.key != chunk.key]
        if not others:
            raise ship.NotReady(t("Nessun altro chunk nel progetto da cui prendere il sistema della nave.",
                                  "No other chunk in the project to take the ship frame from."))
        v = self._ask(t("Usa il sistema nave di", "Use the ship frame of"), [
            Field("source", t("Chunk", "Chunk"), others[0].key, choices=[(c.key, c.label) for c in others])],
            intro=t("Il sistema della nave dell'altro chunk, con le stazioni; le sezioni rifatte su questo modello. "
                    "Serve che i due chunk siano allineati (Flusso di lavoro > Allinea chunk) o che il sistema venga da "
                    "marker degli stessi nomi presenti in entrambi.",
                    "The ship frame of the other chunk, with the stations; the sections cut again on this model. The "
                    "two chunks must be aligned (Workflow > Align chunks) or the frame must come from markers of the "
                    "same names present in both."))
        if not v:
            return
        source = next(c for c in others if c.key == v["source"])
        if self._choose(t("Usa il sistema nave di", "Use the ship frame of"),
                        t(f"Il sistema della nave e le stazioni di «{chunk.label}» saranno sostituiti da quelli di "
                          f"«{source.label}», e le sezioni rifatte. Continuare?",
                          f"The ship frame and the stations of «{chunk.label}» will be replaced by those of "
                          f"«{source.label}», and the sections cut again. Go on?"),
                        [t("Sostituisci", "Replace")]) != 0:
            return
        chunks.use_frame_of(chunk, source)
        self._status(t(f"Sistema della nave di «{source.label}»: rifai il rilievo dello scafo.",
                       f"Ship frame of «{source.label}»: build the relief of the hull again."))
        self._changed()

    def _do_rec_table(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        from . import rebuild, reconstruct

        chunk = self._need_chunk()
        folder = QFileDialog.getExistingDirectory(self.context.window, t("Tabella: misurato e ricostruito",
                                                                         "Table: measured and reconstructed"),
                                                  str(reconstruct.folder(chunk)))
        if not folder:
            return
        out = rebuild.write_table(chunk, folder)
        self._status(t(f"Tabella: {out['csv']}", f"Table: {out['csv']}"))
        self._warn_stale(chunk)

    def _do_rec_export(self) -> None:
        self._do_model_export(reconstruction=True)

    def _do_rec_plan(self) -> None:
        self._do_wreck_plan(reconstruction=True)

    def _draw_relief(self, view, chunk) -> None:
        """The proposals (dashed, a colour per kind) and the traced timbers (solid, labelled) in the relief's (u, v):
        through the frame each was read in to the chunk's coordinates, then into the frame the relief was built in,
        so that they lie where the relief shows them even when the frame changed since."""
        from PySide6.QtCore import Qt

        from . import relief, ship, timbers

        st = ship.state(chunk)
        built_in = ship.Frame.from_dict(st["relief"]["frame"])

        def uv_of(world_xy):
            w = np.asarray(world_xy, float).reshape(-1, 2)
            return relief.uv_of_ship(built_in.to_ship(np.column_stack([w, np.zeros(len(w))]))[:, :2], built_in.sign)

        prop = st.get("proposals") or {}
        read_in = ship.Frame.from_dict(prop["frame"]) if prop.get("frame") else ship.frame_of(chunk)
        with view.batch():
            for item in prop.get("items", []) if read_in is not None else []:
                xy = np.asarray(item["ship_xy"], float).reshape(-1, 2)
                uv = uv_of(read_in.to_world(np.column_stack([xy, np.zeros(len(xy))]))[:, :2])
                colour = KIND_COLOURS.get(item["kind"], KIND_COLOURS["other"])
                attrs = {"ppg_id": item["id"], "kind": item["kind"], "side": item["side"], "name": item["name"],
                         "score": round(float(item.get("score") or 0.0), 3)}
                if item["kind"] == "seam" or len(uv) < 3:
                    view.add_line(uv, colour, "", layer=PROPOSALS_LAYER, attrs=attrs, width=1.5, dash=True)
                else:
                    shape = view.add_polygon(uv, "", colour, fill_alpha=25, layer=PROPOSALS_LAYER, attrs=attrs)
                    pen = shape.pen()
                    pen.setStyle(Qt.PenStyle.DashLine)
                    shape.setPen(pen)
            for r in timbers.timbers_of(chunk):
                geom = r.get("geometry") or {}
                if not geom.get("xy") or not timbers.is_measured(r):
                    continue
                uv = uv_of(geom["xy"])
                colour = KIND_COLOURS.get(r["kind"], KIND_COLOURS["other"])
                attrs = {"ppg_id": r["id"], "kind": r["kind"], "side": r["side"], "name": r["name"],
                         "loose": int(bool(r.get("loose")))}
                label = r["name"] or r["id"]
                if geom["type"] == "point":
                    view.add_point(uv[0, 0], uv[0, 1], colour, label, layer=TRACED_LAYER, attrs=attrs)
                elif geom["type"] == "polygon":
                    view.add_polygon(uv, label, colour, fill_alpha=60, layer=TRACED_LAYER, attrs=attrs)
                else:
                    view.add_line(uv, colour, label, layer=TRACED_LAYER, attrs=attrs)
            self._draw_reconstruction(view, chunk, uv_of)

    @staticmethod
    def _draw_reconstruction(view, chunk, uv_of) -> None:
        """The reconstruction's polygons on the relief, hatched as on the plan of the wreck (a hatch per way) and
        outlined dashed: a hypothesis, never taken for a traced timber."""
        from PySide6.QtCore import Qt
        from PySide6.QtGui import QBrush, QColor

        from . import reconstruct

        for r in reconstruct.records_of(chunk):
            geom = r.get("geometry") or {}
            if geom.get("type") != "polygon" or len(geom.get("xy") or []) < 3:
                continue
            colour = KIND_COLOURS.get(r["kind"], KIND_COLOURS["other"])
            attrs = {"ppg_id": r["id"], "kind": r["kind"], "side": r.get("side") or "", "name": r.get("name") or "",
                     "status": r.get("status") or ""}
            shape = view.add_polygon(uv_of(geom["xy"]), "", colour, fill_alpha=0, layer=RECONSTRUCTION_LAYER,
                                     attrs=attrs)
            style = getattr(Qt.BrushStyle, REC_HATCHES.get(r.get("status"), "BDiagPattern"))
            shape.setBrush(QBrush(QColor(colour), style))
            pen = shape.pen()
            pen.setStyle(Qt.PenStyle.DashLine)
            shape.setPen(pen)

    # ------------------------------------------------------------------ the view
    def _need_chunk(self):
        from . import ship

        chunk = self._chunk()
        if chunk is None:
            raise ship.NotReady(t("Apri un progetto con un chunk.", "Open a project with a chunk."))
        return chunk

    def _need_view(self, chunk) -> None:
        """Something to draw on: the orthophoto or the DEM, else an empty frame around the model."""
        view = getattr(self.context.window, "ortho_view", None)
        on_plan = getattr(self.context.window, "drawings_frame", None) is None  # an elevation has (u, v), not x, y
        if view is not None and view.tr is not None and on_plan:
            return
        if self.context.show_raster(None):
            return
        from . import ship

        xyz = ship.world_vertices(chunk)
        lo, hi = xyz[:, :2].min(axis=0), xyz[:, :2].max(axis=0)
        self.context.show_frame((lo[0], lo[1], hi[0], hi[1]), max(hi - lo) / 1500.0, t(*NAME))

    def draw_layer(self) -> None:
        """The layers of the plugin on the raster view: on the plan the timbers and the land lines in the chunk's
        x, y; on the relief of the hull the proposals and the traced timbers in its (u, v), and the grey raster over
        it; on anything else none."""
        from . import ship

        view = getattr(self.context.window, "ortho_view", None) if self.context else None
        chunk = self._chunk()
        if view is None:
            return
        for name in (LAYER, CLINKER_LAYER, PROPOSALS_LAYER, TRACED_LAYER, RECONSTRUCTION_LAYER):
            view.remove_layer(name)
        if chunk is None or view.tr is None:
            return
        shown = getattr(self.context.window, "drawings_frame", None)
        relief_shown = shown is not None and self._relief_shown(chunk)
        if not relief_shown and self.overlay is not None:
            self.overlay.remove()  # a relief removed without another raster shown
        if shown is None:
            self._draw_plan(view, chunk)
        elif relief_shown:
            self._draw_relief(view, chunk)
            try:
                self._apply_overlay(chunk)
            except (RuntimeError, ship.NotReady) as exc:  # a raster of the relief missing: said, the view stays as it is
                self._status(str(exc))

    def _draw_plan(self, view, chunk) -> None:
        """The timbers of the chunk and the land lines on the plan."""
        from . import timbers

        for r in timbers.timbers_of(chunk):
            xy = np.asarray(r["geometry"]["xy"], float)
            colour = SIDE_COLOURS.get(r["side"], "#ffd000")
            attrs = {"id": r["id"], "kind": r["kind"], "side": r["side"], "name": r["name"]}
            label = r["name"] or r["id"]
            if r["geometry"]["type"] == "point":
                view.add_point(xy[0, 0], xy[0, 1], colour, label, layer=LAYER, attrs=attrs)
            elif r["geometry"]["type"] == "polygon":
                view.add_polygon(xy, label, colour, layer=LAYER, attrs=attrs)
            else:
                view.add_line(xy, colour, label, layer=LAYER, attrs=attrs)
        self._draw_lands(view, chunk)

    @staticmethod
    def _draw_lands(view, chunk) -> None:
        """The land lines in the chunk's x, y, a colour per side, dashed when the sections changed since."""
        from . import clinker, ship

        view.remove_layer(CLINKER_LAYER)
        frame = ship.frame_of(chunk)
        found = [ln for ln in ((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}).get("lines") or [] if len(ln["points"]) > 1]
        if frame is None or not found:
            return
        stale = clinker.is_stale(chunk)
        for line in found:
            xy = frame.to_world(np.asarray(line["points"], float))[:, :2]
            name = f"{clinker.SIDE_CODES[line['side']][0]}{line['k']}"
            view.add_line(xy, SIDE_COLOURS[line["side"]], name, layer=CLINKER_LAYER, width=1.5, dash=stale,
                          attrs={"side": line["side"], "k": line["k"], "uncertain": len(line["uncertain_stations"])})

    # ------------------------------------------------------------------ the workspace tree
    def _tree_items(self, chunk) -> list:
        from . import lines, ship, timbers

        children = []
        frame = ship.frame_of(chunk)
        if frame is not None:
            children.append({"label": t(f"Sistema della nave ({num(frame.length)} m)", f"Ship frame ({num(frame.length)} m)"),
                             "icon": "plugin", "activate": self.show_pane})
        data = lines.load(chunk) if frame is not None else None
        if data is not None:
            children.append({"label": t(f"Sezioni ({len(data['sections'])})", f"Sections ({len(data['sections'])})"),
                             "icon": "plugin", "activate": self.show_pane})
        made = chunk.meta.get(ship.KEY, {}).get("drawings") or {}
        if made:
            from .drawings import KIND_TITLES, out_of_date

            folder = ship.folder(chunk) / "drawings"
            old = out_of_date(chunk)
            kids = [{"label": t(*KIND_TITLES[k]) + (t(" (da rifare)", " (to make again)") if k in old else ""), "icon": "plugin",
                     "activate": lambda f=folder / f"{k}.pdf": self._open(f)} for k in made]
            children.append({"label": t("Disegni", "Drawings"), "icon": "plugin", "children": kids})
        children += self._relief_items(chunk)
        children += self._solids_items(chunk)
        children += self._reconstruction_items(chunk)
        records = timbers.timbers_of(chunk)
        if records:
            count: dict[str, int] = {}
            for r in records:
                count[r["kind"]] = count.get(r["kind"], 0) + 1
            kinds = ", ".join(f"{timbers.kind_title(k)} {n}" for k, n in sorted(count.items(), key=lambda kv: -kv[1]))
            children.append({"label": t(f"Legni ({len(records)}): {kinds}", f"Timbers ({len(records)}): {kinds}"),
                             "icon": "plugin", "activate": self.show_pane})
        strakes = strake_counts(chunk)
        if strakes:
            if len(set(strakes.values())) == 1 and len(strakes) == 2:
                count = t(f"{strakes['port']} per lato", f"{strakes['port']} per side")
            else:
                from .clinker import _side_name

                count = ", ".join(f"{n} {_side_name(side)}" for side, n in strakes.items())
            children.append({"label": t(f"Fasce ({count})", f"Strakes ({count})"), "icon": "plugin",
                             "activate": self.show_pane})
        if not children:
            return []
        return [{"label": t(*NAME), "icon": "plugin", "children": children}]

    def _relief_items(self, chunk) -> list:
        """"Relitto: rilievo" (shows it) and "Proposte (n)" in the workspace tree."""
        from . import relief, ship

        if relief.entry(chunk) is None:
            return []

        def show():
            self._guarded(self._show_relief, self._need_chunk())

        label = t("Relitto: rilievo", "Shipwreck: relief")
        if relief.is_stale(chunk):
            label += t(" (da rifare)", " (to build again)")
        out = [{"label": label, "icon": "plugin", "activate": show,
                "tooltip": t("Lo scafo dall'alto nel sistema della nave", "The hull from above in the ship frame")}]
        prop = ship.state(chunk).get("proposals")
        if prop:
            n = len(prop.get("items", []))
            out.append({"label": t(f"Proposte ({n})", f"Proposals ({n})"), "icon": "plugin", "activate": show})
        return out

    def _solids_items(self, chunk) -> list:
        """"Legni 3D (n)" in the workspace tree, "(interrotto)" for a cancelled build, "(da rifare)" for a stale one;
        it opens the folder of the files."""
        from . import ship, solids

        done = ship.state(chunk).get("solids")
        if not done:
            return []
        label = t(f"Legni 3D ({done.get('count', 0)})", f"3D timbers ({done.get('count', 0)})")
        if done.get("cancelled"):
            label += t(" (interrotto)", " (interrupted)")
        elif solids.is_stale(chunk):
            label += t(" (da rifare)", " (to build again)")
        folder = ship.folder(chunk) / solids.FOLDER
        return [{"label": label, "icon": "plugin", "activate": lambda: self._open(folder),
                 "tooltip": t("Un PLY per legno in shipwreck/timbers", "A PLY per timber in shipwreck/timbers")}]

    def _reconstruction_items(self, chunk) -> list:
        """"Ricostruzione (n, ipotesi)" in the workspace tree: the pane's step 9."""
        from . import reconstruct

        if not _has_reconstruction(chunk):
            return []
        n = len(reconstruct.records_of(chunk))
        stale = reconstruct.stale(chunk)
        label = (t(f"Ricostruzione ({n}, ipotesi, da rifare)", f"Reconstruction ({n}, hypothesis, to build again)")
                 if stale else t(f"Ricostruzione ({n}, ipotesi)", f"Reconstruction ({n}, hypothesis)"))
        tip = t("Simmetria, prolungamenti, ordinate al passo, carenatura: ipotesi, mai misure",
                "Mirror, extensions, frames at the spacing, faired hull: hypotheses, never measures")
        return [{"label": label, "icon": "plugin", "activate": self.show_pane,
                 "tooltip": tip + (" " + t("Da ricostruire: ", "To reconstruct again: ") + stale if stale else "")}]

    def _open(self, path) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    # ------------------------------------------------------------------ the command line
    def command(self, *args) -> int:
        from . import drawings, lines, ship, timbers

        if args and str(args[0]) == "clinker":
            return self._clinker_command([str(a) for a in args[1:]])
        if args and str(args[0]) in RECONSTRUCTION:
            return self._reconstruction_command([str(a) for a in args])
        parser = argparse.ArgumentParser(prog="ppg plugin shipwreck", description="the technical drawing of a hull from its 3D model")
        parser.add_argument("what", choices=["frame", "stations", "sections", "water", "drawings", "timbers", "export", "status",
                                             "clinker", *RECONSTRUCTION],
                            help="clinker, relief, propose, trace, keel, solids, export3d, plan: see "
                                 "'ppg plugin shipwreck <what> --help'")
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--start", help="frame: first end of the keel, X,Y[,Z] world, or a marker with --markers")
        parser.add_argument("--end", help="frame: other end of the keel")
        parser.add_argument("--markers", action="store_true", help="frame: --start and --end are markers")
        parser.add_argument("--bow", choices=["start", "end"], default="start", help="frame: the end where the bow is")
        parser.add_argument("--flip", action="store_true", help="frame: x from the other end; the bow, y and the sides stay")
        parser.add_argument("--other-bow", action="store_true", help="frame: the bow is the other end; port and starboard are exchanged")
        parser.add_argument("--z0", type=float, help="frame: the world height of ship z = 0")
        parser.add_argument("--x0", type=float, default=0.0, help="frame: ship x of the first end")
        parser.add_argument("--spacing", type=float, help="stations: regular, metres")
        parser.add_argument("--count", type=int, help="stations: regular, how many")
        parser.add_argument("--first", type=float, default=0.0, help="stations: x of the first")
        parser.add_argument("--table", help="stations: a table name, x; timbers: a table to import")
        parser.add_argument("--from-catalogue", action="store_true", help="stations: the frames of the catalogue")
        parser.add_argument("--longitudinals", default="0", help="sections: y values, as 0;0.5")
        parser.add_argument("--horizontals", default="", help="sections: z values, as -0.5;-1")
        parser.add_argument("--half-width", type=float, help="sections: half width drawn, metres")
        parser.add_argument("--level", type=float, help="water: level in ship z")
        parser.add_argument("--n", type=float, default=lines.REFRACTION, help="water: refractive index")
        parser.add_argument("--on", action="store_true", help="water: correct the heights under the level")
        parser.add_argument("--scale", type=int, help="drawings: 1:SCALE (default: automatic)")
        parser.add_argument("--out", help="drawings: a folder; export: the file (.csv, .xlsx, .gpkg)")
        parser.add_argument("--reconstruction", action="store_true",
                            help="drawings: with the reconstruction (a hypothesis) on the plan of the wreck, the body "
                                 "plan and the profile")
        ns = parser.parse_args(joined_values([str(a) for a in args]))
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        try:
            if ns.what == "frame":
                if ns.flip:
                    ship.set_frame(chunk, ship.need_frame(chunk).flipped(),
                                   markers=ship.frame_markers(chunk, reverse=True))
                elif ns.other_bow:
                    ship.set_frame(chunk, ship.need_frame(chunk).other_bow(), markers=ship.frame_markers(chunk))
                elif ns.markers:
                    ship.set_frame(chunk, ship.frame_from_markers(chunk, ns.start, ns.end, ns.bow, ns.z0, ns.x0),
                                   markers=[ship.marker_label(chunk, ns.start), ship.marker_label(chunk, ns.end)])
                elif ns.start and ns.end:
                    a, b = (np.array([float(v) for v in s.split(",")]) for s in (ns.start, ns.end))
                    ship.set_frame(chunk, ship.frame_from_picks(chunk, a, b, ns.bow, ns.z0, ns.x0))
                else:
                    raise ship.NotReady("--start and --end, --flip or --other-bow")
            elif ns.what == "stations":
                if ns.table:
                    stations = ship.stations_from_table(ns.table)
                elif ns.from_catalogue:
                    stations = timbers.stations_from_catalogue(chunk)
                elif ns.spacing and ns.count:
                    stations = ship.regular(ns.spacing, ns.count, ns.first)
                else:
                    raise ship.NotReady("--spacing and --count, --table or --from-catalogue")
                ship.set_stations(chunk, stations)
            elif ns.what == "sections":
                data = lines.build(chunk, parse_values(ns.longitudinals), parse_values(ns.horizontals), ns.half_width,
                                   progress=_print_progress)
                print(f"{len(data['sections'])} sections")
            elif ns.what == "water":
                if ns.level is None:
                    raise ship.NotReady("--level")
                lines.set_water(chunk, ns.level, ns.n, ns.on)
            elif ns.what == "drawings":
                from . import relief

                if drawings._traced(chunk) and relief.entry(chunk) is not None and relief.is_stale(chunk):
                    print(t("Il rilievo dello scafo è da rifare: sul profilo mancano paramezzale e chiglia.",
                            "The relief of the hull is to be built again: the profile lacks the keelson and the keel."))
                if ns.reconstruction and reconstruction_warning(chunk):
                    print(reconstruction_warning(chunk))
                for path in drawings.write_all(chunk, scale=ns.scale, folder=ns.out, progress=_print_progress,
                                               reconstruction=ns.reconstruction):
                    print(path)
            elif ns.what == "timbers":
                if ns.table:
                    print(f"{timbers.import_csv(chunk, ns.table)} timbers imported")
                print(json.dumps(timbers.measure(chunk), indent=1, ensure_ascii=False))
            elif ns.what == "export":
                if not ns.out:
                    raise ship.NotReady("--out FILE (.csv, .xlsx, .gpkg)")
                print(timbers.export_any(chunk, ns.out))
            else:
                print(json.dumps(self.states(chunk), indent=1, ensure_ascii=False))
                return 0
        except ship.NotReady as exc:
            print(exc)
            return 1
        doc.save()
        return 0


    def _reconstruction_command(self, args: list[str]) -> int:
        """``relief build|export``, ``propose``, ``trace list|accept|reject|names``, ``keel``, ``solids``,
        ``export3d``, ``plan`` (0.3.0); ``ends``, ``frame-of``, ``chunks``, ``reconstruct [table]`` (0.4.0)."""
        from . import drawings, proposals, relief, ship, solids, tracing

        what, rest = args[0], args[1:]
        action = None
        if what == "reconstruct" and rest[:1] == ["table"]:
            action, rest = "table", rest[1:]
        parser = argparse.ArgumentParser(prog=f"ppg plugin shipwreck {what}", description=RECONSTRUCTION[what])
        if what in ("relief", "trace"):
            actions = {"relief": ["build", "export"], "trace": ["list", "accept", "reject", "names"]}[what]
            parser.add_argument("action", choices=actions)
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        if what == "relief":
            parser.add_argument("--cell", type=float, help="build: the cell, mm (default: from the model's density)")
            parser.add_argument("--kind", choices=list(relief.KINDS), default="hillshade", help="export: which raster")
            parser.add_argument("--out", help="export: the GeoTIFF")
            parser.add_argument("--world", action="store_true", help="export: north up in the chunk's system (default: "
                                                                     "ship metres, the frame in the tags)")
        elif what == "propose":
            parser.add_argument("--view", choices=list(proposals.VIEWS), help="the model shows the inside (default) or "
                                                                               "the outside of the hull")
            parser.add_argument("--min-step", type=float, help="least step of a seam, mm (8)")
            parser.add_argument("--opening", type=float, help="the opening, m (0.3)")
            parser.add_argument("--top-hat", type=float, help="least height of a timber over the planking, mm (20)")
        elif what == "trace":
            parser.add_argument("ids", nargs="*", help="accept, reject: the proposals' ids (P12 ...)")
            parser.add_argument("--kind", choices=list(proposals.KINDS), help="accept: every proposal of this kind")
            parser.add_argument("--min-score", type=float, default=0.5, help="accept --kind: least score (0.5)")
            parser.add_argument("--name", help="accept: the name of the timber (one id)")
            parser.add_argument("--table", help="names: a table nome, x (name, x) for the frames")
        elif what == "keel":
            parser.add_argument("--siding", type=float, help="across, mm")
            parser.add_argument("--moulding", type=float, help="upright, mm")
            parser.add_argument("--from", dest="x0", type=float, help="ship x of one end, m (default: the keelson's)")
            parser.add_argument("--to", dest="x1", type=float, help="ship x of the other end, m")
            parser.add_argument("--none", action="store_true", help="no keel: the given one removed")
        elif what == "solids":
            parser.add_argument("--thickness", type=float, help="mm, for the strakes that have none of their own")
        elif what == "export3d":
            parser.add_argument("--out", help="the file: .obj, .glb or .ifc")
            parser.add_argument("--world", action="store_true", help="the chunk's coordinates (default: the ship's)")
            parser.add_argument("--no-tables", action="store_true", help="without timbers_3d.csv and .xlsx beside it")
            parser.add_argument("--reconstruction", action="store_true", help="the reconstruction too (_ric) and its "
                                                                              "table beside")
        elif what == "plan":
            parser.add_argument("--background", choices=list(drawings.BACKGROUNDS), default="relief")
            parser.add_argument("--scale", type=int, help="1:SCALE (default: automatic)")
            parser.add_argument("--out", help="a folder (default: shipwreck/drawings of the chunk)")
            parser.add_argument("--reconstruction", action="store_true", help="the reconstruction hatched on it")
        elif what == "ends":
            parser.add_argument("--stem", type=float, help="the ship x of the stem, m")
            parser.add_argument("--stern", type=float, help="the ship x of the stern, m")
            parser.add_argument("--none", action="store_true", help="no ends: within the surveyed length only")
        elif what == "frame-of":
            parser.add_argument("source", help="the label of the chunk whose ship frame is taken")
        elif what == "reconstruct":
            for step in ("mirror", "extension", "frames", "loft"):
                parser.add_argument(f"--no-{step}", action="store_true", help=f"leave the {step} out")
            parser.add_argument("--out", help="table: the folder (default: shipwreck/reconstruction of the chunk)")
        ns = parser.parse_args(joined_values(rest))
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        try:
            if what in ("ends", "frame-of", "chunks", "reconstruct"):
                code = _rebuild_command(doc, chunk, what, action, ns)
                if code:
                    return code
            elif what == "relief" and ns.action == "build":
                relief.build(chunk, ns.cell / 1000.0 if ns.cell else None, progress=_print_progress)
                print(relief_state(chunk))
            elif what == "relief":
                if not ns.out:
                    raise ship.NotReady("--out FILE.tif")
                if relief.entry(chunk) is None:
                    raise ship.NotReady(t("Costruisci prima il rilievo dello scafo.", "Build the relief of the hull first."))
                print((relief.export_world if ns.world else relief.export_ship)(chunk, ns.kind, ns.out))
                return 0
            elif what == "propose":
                params = {}
                if ns.view:
                    params["view"] = ns.view
                if ns.min_step is not None:
                    params["min_step"] = ns.min_step / 1000.0
                if ns.opening is not None:
                    params["opening"] = ns.opening
                if ns.top_hat is not None:
                    params["top_hat"] = ns.top_hat / 1000.0
                proposals.build(chunk, params or None, progress=_print_progress)
                print(proposals.summary(chunk))
            elif what == "trace" and ns.action == "list":
                for i in (ship.state(chunk).get("proposals") or {}).get("items", []):
                    xs = np.asarray(i["ship_xy"], float)[:, 0]
                    print(f"{i['id']}\t{i['kind']}\t{i['side']}\t{i['name']}\tscore {i.get('score') or 0:.2f}\t"
                          f"x {xs.min():.2f}-{xs.max():.2f}")
                print(proposals.summary(chunk))
                return 0
            elif what == "trace" and ns.action == "accept":
                ids = list(ns.ids)
                if ns.kind:
                    ids += [i["id"] for i in (ship.state(chunk).get("proposals") or {}).get("items", [])
                            if i["kind"] == ns.kind and float(i.get("score") or 0.0) >= ns.min_score and i["id"] not in ids]
                if not ids:
                    raise ship.NotReady(t("Nessuna proposta da accettare: dai gli id o --kind.",
                                          "No proposal to accept: give the ids or --kind."))
                if ns.name and len(ids) != 1:
                    raise ship.NotReady("--name: one id")
                records = tracing.accept(chunk, ids, [ns.name] if ns.name else None)
                tracing.update_visible(chunk)
                for r in records:
                    print(f"{r['id']}\t{r['kind']}\t{r['name']}")
                print(plural(len(records), ("legno accettato", "legni accettati"), ("timber accepted", "timbers accepted")))
            elif what == "trace" and ns.action == "reject":
                if not ns.ids:
                    raise ship.NotReady(t("Dai gli id delle proposte da scartare.", "Give the ids of the proposals to reject."))
                count = tracing.reject(chunk, ns.ids)
                print(plural(count, ("proposta scartata", "proposte scartate"), ("proposal rejected", "proposals rejected")))
            elif what == "trace":
                if not ns.table:
                    raise ship.NotReady("--table FILE")
                named, unmatched = tracing.rename_frames(chunk, ns.table)
                print(t(f"Ordinate rinominate: {named}; righe senza ordinata: {unmatched}",
                        f"Frames renamed: {named}; rows matching no frame: {unmatched}"))
            elif what == "keel":
                if ns.none:
                    solids.set_keel(chunk, None, None)
                    print(t("Nessuna chiglia.", "No keel."))
                else:
                    if ns.siding is None or ns.moulding is None:
                        raise ship.NotReady("--siding MM --moulding MM, or --none")
                    keel = solids.set_keel(chunk, ns.siding / 1000.0, ns.moulding / 1000.0, ns.x0, ns.x1)
                    print(t(f"Chiglia data (ipotesi): {num(ns.siding, 0)} x {num(ns.moulding, 0)} mm, x da "
                            f"{num(keel['x0'])} a {num(keel['x1'])} m",
                            f"Given keel (hypothesis): {num(ns.siding, 0)} x {num(ns.moulding, 0)} mm, x from "
                            f"{num(keel['x0'])} to {num(keel['x1'])} m"))
            elif what == "solids":
                out = solids.build(chunk, ns.thickness, progress=_print_progress)
                print(solids_summary(out))
                for r in out["refused"]:
                    print(f"  {r['name']}: {r['reason']}")
            elif what == "export3d":
                if not ns.out:
                    raise ship.NotReady("--out FILE (.obj, .glb, .ifc)")
                check_export_path(ns.out)
                warning = solids_warning(chunk)
                if warning:
                    print(warning)
                out = export_3d(chunk, ns.out, "world" if ns.world else "ship", not ns.no_tables,
                                reconstruction=ns.reconstruction)
                for key in ("path", "csv", "xlsx", "table"):
                    if out.get(key):
                        print(out[key])
                if out.get("note"):
                    print(out["note"])
            elif what == "plan":
                if relief.entry(chunk) is not None and relief.is_stale(chunk):
                    print(t("Il rilievo dello scafo è da rifare: lo sfondo è quello vecchio.",
                            "The relief of the hull is to be built again: the background is the old one."))
                if ns.reconstruction and reconstruction_warning(chunk):
                    print(reconstruction_warning(chunk))
                for path in drawings.write_all(chunk, kinds=("wreck_plan",), scale=ns.scale, folder=ns.out,
                                               progress=_print_progress, background=ns.background,
                                               reconstruction=ns.reconstruction):
                    print(path)
        except ship.NotReady as exc:
            print(exc)
            return 1
        doc.save()
        return 0

    def _clinker_command(self, args: list[str]) -> int:
        from . import clinker, houdini, ship

        parser = argparse.ArgumentParser(prog="ppg plugin shipwreck clinker",
                                         description="clinker strakes: the lands on the sections, the land lines, the "
                                                     "strake model (a hypothesis), the Clinker Boat Tool's files")
        parser.add_argument("action", choices=["detect", "strakes", "export", "import", "status"])
        parser.add_argument("project", help="the .ppg file")
        parser.add_argument("--chunk", help="the label of the chunk (default: the active one)")
        parser.add_argument("--min-step", type=float, help="detect: least step of a land, mm (8)")
        parser.add_argument("--min-strake", type=float, help="detect: least width of a strake, m (0.08)")
        parser.add_argument("--keel-half-width", type=float, help="detect: half width of the keel, m (0.10)")
        parser.add_argument("--sole-width", type=float, help="detect: least width of a sole, m (0.02)")
        parser.add_argument("--lap", type=float, default=20.0, help="strakes: the lap, mm (20)")
        parser.add_argument("--thickness", type=float, help="strakes: mm (default: the step measured)")
        parser.add_argument("--catalogue", action="store_true", help="strakes: a plank of the catalogue per strake")
        parser.add_argument("--out", help="export: the folder (clinker_boat_tool/ is made in it)")
        parser.add_argument("--obj", help="import: the OBJ of the planks exported by the Clinker Boat Tool")
        parser.add_argument("--side-from", choices=["y", "name"], default="y", help="import: the side of a plank")
        parser.add_argument("--no-deviation", action="store_true", help="import: skip the distance from the scan")
        ns = parser.parse_args(args)
        import pyarchinit_photogrammetry as ppg

        doc = ppg.Document.open(ns.project)
        chunk = next((c for c in doc.chunks if c.label == ns.chunk), None) if ns.chunk else doc.chunk
        if chunk is None:
            print(f"no chunk {ns.chunk!r} in the project")
            return 1
        try:
            if ns.action == "detect":
                params = {key: getattr(ns, key) for key in ("min_strake", "keel_half_width", "sole_width")
                          if getattr(ns, key) is not None}
                if ns.min_step is not None:
                    params["min_step"] = ns.min_step / 1000.0
                result = clinker.detect_lands(chunk, params, progress=_print_progress)
                print(clinker.summary(chunk))
                for line in detection_lines(chunk, result):
                    print(line)
                for path in lines_files(chunk).values():
                    print(path)
            elif ns.action == "strakes":
                result = clinker.build_strakes(chunk, ns.lap / 1000.0,
                                               ns.thickness / 1000.0 if ns.thickness else None, progress=_print_progress)
                result.update(clinker.write_tables(chunk))
                count = plural(len(result["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
                print(t(f"Modello delle fasce (ipotesi): {count}", f"Strake model (hypothesis): {count}"))
                for key in ("obj", "ply", "csv", "xlsx"):
                    print(result[key])
                if ns.catalogue:
                    count = clinker.to_catalogue(chunk)
                    print(plural(count, ("fascia", "fasce"), ("strake", "strakes")) + t(" nel catalogo dei legni",
                                                                                        " in the timber catalogue"))
            elif ns.action == "export":
                if not ns.out:
                    raise ship.NotReady("--out FOLDER")
                for path in houdini.export(chunk, ns.out, progress=_print_progress).values():
                    print(path)
            elif ns.action == "import":
                if not ns.obj:
                    raise ship.NotReady("--obj FILE")
                result = houdini.import_planks(chunk, ns.obj, ns.side_from, not ns.no_deviation, progress=_print_progress)
                n = result["planks"]
                print(t("1 fascia importata" if n == 1 else f"{n} fasce importate",
                        "1 plank imported" if n == 1 else f"{n} planks imported"))
                for r in result["rows"]:
                    median = "-" if r["median_mm"] is None else num(r["median_mm"], 1)
                    print(t(f"  {r['name']}: distanza dalla scansione, mediana {median} mm",
                            f"  {r['name']}: distance from the scan, median {median} mm"))
                print(result["csv"])
                print(result["xlsx"])
            else:
                print(clinker_state(chunk))
                if "lands" in ((chunk.meta.get(ship.KEY) or {}).get("clinker") or {}):
                    print(t("Gradini da rilevare di nuovo: le sezioni sono cambiate." if clinker.is_stale(chunk)
                            else "Gradini aggiornati rispetto alle sezioni.",
                            "Lands to find again: the sections changed." if clinker.is_stale(chunk)
                            else "Lands up to date with the sections."))
                return 0
        except ship.NotReady as exc:
            print(exc)
            return 1
        doc.save()
        return 0


def clinker_state(chunk) -> str:
    """The clinker step's state in words: the summary of the lands, the strake model, the catalogue, the imports."""
    from statistics import median

    from . import clinker, houdini, ship

    text = clinker.summary(chunk)
    state = (chunk.meta.get(ship.KEY) or {}).get("clinker") or {}
    if state.get("strakes"):
        count = plural(len(state["strakes"]), ("fascia", "fasce"), ("strake", "strakes"))
        text += " " + t(f"Modello delle fasce (ipotesi dalle sezioni): {count} in shipwreck/clinker.",
                        f"Strake model (hypothesis from the sections): {count} in shipwreck/clinker.")
        if state.get("catalogue"):
            text += " " + t(f"Nel catalogo: {len(state['catalogue'])}.", f"In the catalogue: {len(state['catalogue'])}.")
    for name, rows in houdini.imports(chunk).items():
        medians = [r["median_mm"] for r in rows if r.get("median_mm") is not None]
        count = plural(len(rows), ("tavola", "tavole"), ("plank", "planks"))
        far = (t(f", distanza dalla scansione mediana {num(median(medians), 1)} mm",
                 f", distance from the scan median {num(median(medians), 1)} mm") if medians else "")
        text += " " + t(f"Da Houdini, {name}: {count}{far}.", f"From Houdini, {name}: {count}{far}.")
    return text


RECONSTRUCTION = {
    "relief": "the relief of the hull from above: build it (--cell MM), export a raster (--kind, --out, --world)",
    "propose": "read the proposals on the relief: seams, frames, keelson, stringers, strakes",
    "trace": "the proposals: list, accept (ids or --kind, --name), reject (ids), names of the frames (--table)",
    "keel": "a keel from a given siding and moulding (a hypothesis): --siding MM --moulding MM [--from X --to X] | --none",
    "solids": "every traced polygon a closed 3D object (--thickness MM for the strakes without one)",
    "export3d": "the timbers in 3D: --out FILE.obj|.glb|.ifc [--world]",
    "plan": "the plan of the wreck: [--background relief|ortho|none] [--scale N] [--out FOLDER] [--reconstruction]",
    "ends": "the stem and the stern the reconstruction may reach: [--stem X] [--stern X] | --none (ship x, m)",
    "frame-of": "the ship frame of another chunk of the wreck (aligned, or same-name markers): frame-of PROJECT CHUNK",
    "chunks": "the chunks of the wreck feeding the reconstruction, and why the others do not",
    "reconstruct": "the reconstruction (hypothesis): [--no-mirror] [--no-extension] [--no-frames] [--no-loft]; "
                   "'reconstruct table PROJECT --out FOLDER' writes the table of the measured and the reconstructed",
}
REBUILD_STEPS = ("mirror", "extension", "frames", "loft")  # :data:`rebuild.STEPS`, without importing it here


def _rebuild_command(doc, chunk, what: str, action, ns) -> int:
    """The 0.4.0 commands (see :data:`RECONSTRUCTION`); 1 when what is asked cannot be done (said)."""
    from . import chunks, rebuild, reconstruct, ship

    if what == "ends":
        if ns.none:
            reconstruct.set_ends(chunk)
        elif ns.stem is None and ns.stern is None:
            raise ship.NotReady("--stem X, --stern X or --none")
        else:
            reconstruct.set_ends(chunk, ns.stem, ns.stern)
        if not ship.state(chunk).get("reconstruction"):
            ship.state(chunk).pop("reconstruction", None)
        print(ends_text(chunk))
    elif what == "frame-of":
        source = next((c for c in doc.chunks if c.label == ns.source), None)
        if source is None or source is chunk:
            raise ship.NotReady(t(f"Nessun altro chunk «{ns.source}» nel progetto.",
                                  f"No other chunk «{ns.source}» in the project."))
        frame = chunks.use_frame_of(chunk, source)
        print(t(f"Sistema della nave di «{source.label}» per «{chunk.label}»: rifai il rilievo dello scafo.",
                f"Ship frame of «{source.label}» for «{chunk.label}»: build the relief of the hull again."))
        print(json.dumps({k: frame[k] for k in ("start", "end", "bow") if k in frame}))
    elif what == "chunks":
        found, refused = chunks.members(doc, chunk)
        for m in found:
            print(t(f"{m['chunk'].label}: cella {num(m['cell'] * 1000.0, 1)} mm, copertura {num(m['coverage'] * 100, 0)} %",
                    f"{m['chunk'].label}: cell {num(m['cell'] * 1000.0, 1)} mm, coverage {num(m['coverage'] * 100, 0)} %"))
        for r in refused:
            print(t(f"Escluso «{r['chunk'].label}»: {r['reason']}", f"Left out «{r['chunk'].label}»: {r['reason']}"))
        return 1 if refused and len(found) < 2 else 0  # every other chunk left out: nothing to merge
    elif action == "table":
        out = rebuild.write_table(chunk, ns.out, doc=doc)
        print(out["csv"])
        print(out["xlsx"])
        if out.get("stale"):
            print(reconstruction_warning(chunk))
    else:
        steps = [s for s in REBUILD_STEPS if not getattr(ns, f"no_{s}")]
        out = rebuild.build(chunk, progress=_print_progress, doc=doc, steps=steps)
        print(reconstruction_summary(out))
    return 0


def _has_reconstruction(chunk) -> bool:
    st = ((chunk.meta.get("shipwreck") or {}).get("reconstruction") or {})
    return bool(st.get("timbers") or (st.get("loft") or {}).get("file"))


def ends_text(chunk) -> str:
    """The stem and the stern given, in words."""
    ends = (((chunk.meta.get("shipwreck") or {}).get("reconstruction") or {}).get("ends")) or {}
    words = []
    for key, title in (("stem", ("prua", "stem")), ("stern", ("poppa", "stern"))):
        e = ends.get(key)
        if e is None:
            continue
        where = (f"x {num(e['x'])} m" if "x" in e else t("linea tracciata", "traced line"))
        words.append(f"{t(*title)} {where}")
    if not words:
        return t("Nessuna estremità data: solo entro la lunghezza rilevata.",
                 "No ends given: within the surveyed length only.")
    return t("Estremità date (ipotesi): ", "Ends given (hypothesis): ") + ", ".join(words) + "."


def reconstruction_state(chunk) -> str:
    """The reconstruction step's state in words: what was reconstructed and how, the faired hull, the chunks, the
    ends, what is out of date."""
    from . import reconstruct

    st = ((chunk.meta.get("shipwreck") or {}).get("reconstruction") or {})
    ends = ends_text(chunk) if st.get("ends") else ""
    if not _has_reconstruction(chunk):
        return " ".join(x for x in (t("Nessuna ricostruzione.", "No reconstruction."), ends) if x)
    count: dict[str, int] = {}
    for r in st.get("timbers") or []:
        count[r.get("status")] = count.get(r.get("status"), 0) + 1
    parts = [t(f"{count.get(reconstruct.STATUS, 0)} per simmetria", f"{count.get(reconstruct.STATUS, 0)} mirrored"),
             t(f"{count.get(reconstruct.EXTENSION, 0)} prolungamenti", f"{count.get(reconstruct.EXTENSION, 0)} extensions"),
             t(f"{count.get(reconstruct.SPACING, 0)} ordinate al passo", f"{count.get(reconstruct.SPACING, 0)} frames "
               "at the spacing")]
    n = sum(count.values())
    text = t(f"Ricostruzione (ipotesi): {plural(n, ('legno', 'legni'), ('timber', 'timbers'))} ({', '.join(parts)}).",
             f"Reconstruction (hypothesis): {plural(n, ('legno', 'legni'), ('timber', 'timbers'))} ({', '.join(parts)}).")
    loft = st.get("loft") or {}
    if loft.get("file"):
        names = {"stem": t("prua", "stem"), "stern": t("poppa", "stern")}
        opened = [names.get(e, e) for e in st.get("open_ends") or []]
        text += " " + (t(f"Carenatura aperta a {' e '.join(opened)}.", f"Faired hull open at the {' and '.join(opened)}.")
                       if opened else t("Carenatura chiusa alle estremità date.", "Faired hull closed at the ends given."))
    members = st.get("members") or {}
    if len(members.get("chunks") or []) > 1:
        labels = ", ".join(str(m.get("label") or m.get("chunk")) for m in members["chunks"])
        text += " " + t(f"Dai chunk: {labels}.", f"From the chunks: {labels}.")
    if members.get("refused"):
        text += " " + t(f"Chunk esclusi: {len(members['refused'])}.", f"Chunks left out: {len(members['refused'])}.")
    stale = reconstruct.stale(chunk)
    if stale:
        text += " " + t("Da ricostruire: ", "To reconstruct again: ") + stale
    return " ".join(x for x in (text, ends) if x)


def _with_reconstruction(chunk, default: bool) -> list:
    """The checkbox "con la ricostruzione (ipotesi)" of a dialog of drawings or of a 3D export, when the chunk has a
    reconstruction (else no field: the 0.3.0 output)."""
    from pyarchinit_photogrammetry.gui.dialogs import Field

    if not _has_reconstruction(chunk):
        return []
    return [Field("reconstruction", t("Con la ricostruzione (ipotesi)", "With the reconstruction (hypothesis)"),
                  bool(default), tip=t("Le parti ricostruite campite (nei disegni) o trasparenti _ric (nei file 3D)",
                                       "The reconstructed parts hatched (drawings) or translucent _ric (3D files)"))]


def rebuild_notes(chunk) -> dict:
    """The notes under step 9's buttons: "Da ricostruire: <why>" on the build, the table, the export and the plan
    while the stored reconstruction is out of date (:func:`reconstruct.stale`); "" otherwise."""
    from . import reconstruct

    reason = reconstruct.stale(chunk) if chunk is not None else None
    text = t("Da ricostruire: ", "To reconstruct again: ") + reason if reason else ""
    return {name: text for name in ("rec_build", "rec_table", "rec_export", "rec_plan")}


def reconstruction_warning(chunk) -> str:
    """What a table, a drawing or a 3D export with the reconstruction should say: why it is out of date
    (:func:`reconstruct.stale`), "" while it is current or there is none."""
    from . import reconstruct

    reason = reconstruct.stale(chunk)
    return t(f"La ricostruzione è da rifare (Ricostruisci, passo 9): {reason}",
             f"The reconstruction is to be built again (Reconstruct, step 9): {reason}") if reason else ""


def reconstruction_summary(out: dict) -> str:
    """A build of the reconstruction in lines: the counts, the faired hull, the chunks, then the warnings, the notes
    and the refused pieces."""
    lines_ = [t(f"Ricostruzione (ipotesi): {out['mirrored']} per simmetria, {out['extended']} prolungamenti, "
                f"{out['spaced']} ordinate al passo, {len(out['built'])} solidi, {len(out['refused'])} rifiutati",
                f"Reconstruction (hypothesis): {out['mirrored']} mirrored, {out['extended']} extended, {out['spaced']} "
                f"frames at the spacing, {len(out['built'])} solids, {len(out['refused'])} refused")]
    if out.get("loft"):
        names = {"stem": t("prua", "stem"), "stern": t("poppa", "stern")}
        opened = [names.get(e, e) for e in out.get("open_ends") or []]
        lines_.append(t("Carenatura" + (f": aperta a {' e '.join(opened)}" if opened else ": chiusa alle estremità"),
                        "Faired hull" + (f": open at the {' and '.join(opened)}" if opened else ": closed at the ends")))
    members = out.get("members") or {}
    if len(members.get("chunks") or []) > 1:
        lines_.append(t("Chunk: ", "Chunks: ") + ", ".join(str(m.get("label")) for m in members["chunks"]))
    for r in members.get("refused") or []:
        lines_.append(t(f"Escluso «{r['label']}»: {r['reason']}", f"Left out «{r['label']}»: {r['reason']}"))
    lines_ += [t("Avviso: ", "Warning: ") + w for w in out.get("warnings") or []]
    lines_ += [t("Nota: ", "Note: ") + n for n in out.get("notes") or []]
    lines_ += [f"  {r['name']}: {r['reason']}" for r in out.get("refused") or []]
    return "\n".join(lines_)


def _forget_copy(chunk, folder) -> None:
    """A copy for Annulla no step holds removed, unless the next build of ``chunk`` reuses it as its "before"."""
    last = _LAST_AFTER.get(chunk.key)
    if folder is None or _HELD.get(folder) or (last is not None and last[0] == folder):
        return
    _UNDO_DIRS.discard(folder)
    _remove(folder)


def _rec_folder(chunk) -> Path:
    """The reconstruction's folder (``<chunk data>/shipwreck/reconstruction``), not created."""
    from . import reconstruct, ship

    return ship.folder(chunk) / reconstruct.FOLDER


def _remove(path: Path) -> None:
    import shutil

    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)


def _copy_folder(src: Path, dst: Path) -> None:
    """``src`` copied to ``dst`` by hard links (a plain copy where the file system has none); an empty ``dst`` when
    ``src`` does not exist."""
    import os
    import shutil

    _remove(dst)
    if not src.exists():
        dst.mkdir(parents=True)
        return
    try:
        shutil.copytree(src, dst, copy_function=os.link)
    except OSError:
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(src, dst, copy_function=shutil.copy2)


def _restore_folder(kept: Path, folder: Path) -> None:
    """The reconstruction's ``folder`` as ``kept`` was (``kept`` stays, for a redo)."""
    tmp = folder.with_name(folder.name + ".undo-tmp")
    _copy_folder(kept, tmp)
    _remove(folder)
    tmp.replace(folder)
CACHES = ("ship_xy", "frame_key", "visible_xy")  # the fields of a record that are its plan in a frame, a cache


def _without_caches(snapshot: dict) -> dict:
    """A snapshot of :data:`UNDO_KEYS` without the plan caches of the records: what an undo step compares."""
    out = dict(snapshot)
    if out.get("timbers"):
        out["timbers"] = [{k: v for k, v in r.items() if k not in CACHES} for r in out["timbers"]]
    return out


def relief_state(chunk) -> str:
    """The relief and tracing step's state in words: the relief (cell, size, stale), the proposals, the traced timbers."""
    from . import proposals, relief, ship, timbers

    st = (chunk.meta.get(ship.KEY) or {}).get("relief")
    e = relief.entry(chunk)
    if not st or e is None:
        text = t("Non costruito.", "Not built.")
        if st:
            text = t("Il rilievo è stato tolto nel programma: costruiscilo di nuovo.",
                     "The relief was removed in the program: build it again.")
        return " ".join([text] + _solids_texts(chunk))
    size = e.get("size") or ("?", "?")
    text = t(f"Rilievo: {size[0]} x {size[1]} celle di {num(st['cell'] * 1000.0, 1)} mm.",
             f"Relief: {size[0]} x {size[1]} cells of {num(st['cell'] * 1000.0, 1)} mm.")
    if relief.is_stale(chunk):
        text += " " + t("Di un altro sistema della nave o di un altro modello: rifallo.",
                        "Of another ship frame or another model: build it again.")
    if (chunk.meta.get(ship.KEY) or {}).get("proposals"):
        text += " " + proposals.summary(chunk)
    traced = [r for r in timbers.timbers_of(chunk) if r.get("elevation") or r.get("source")]
    if traced:
        text += " " + t("Ricalcati: ", "Traced: ") + plural(len(traced), ("legno", "legni"), ("timber", "timbers")) + "."
    return " ".join([text] + _solids_texts(chunk))


def _solids_texts(chunk) -> list[str]:
    """The typed keel and the timbers in 3D in words, for the state of step 8."""
    from . import ship, solids

    st = chunk.meta.get(ship.KEY) or {}
    out = []
    keel = st.get("keel")
    if keel:
        out.append(t(f"Chiglia data {num(keel['siding'] * 1000.0, 0)} x {num(keel['moulding'] * 1000.0, 0)} mm "
                     f"(ipotesi).", f"Given keel {num(keel['siding'] * 1000.0, 0)} x {num(keel['moulding'] * 1000.0, 0)} "
                     f"mm (hypothesis)."))
    done = st.get("solids")
    if done:
        text = t(f"Legni 3D: {done.get('count', 0)}", f"3D timbers: {done.get('count', 0)}")
        if done.get("refused"):
            text += t(f", rifiutati {len(done['refused'])}", f", refused {len(done['refused'])}")
        if done.get("cancelled"):
            text += t(" (interrotto: parziali)", " (interrupted: partial)")
        elif solids.is_stale(chunk):
            text += t(" (da rifare: il catalogo, la chiglia o il rilievo sono cambiati)",
                      " (to build again: the catalogue, the keel or the relief changed)")
        out.append(text + ".")
    return out


def solids_summary(out: dict) -> str:
    """A line on a build of the timbers in 3D: built, refused, the volume."""
    volume = sum(b["volume"] for b in out["built"]) + (out["keel"]["volume"] if out.get("keel") else 0.0)
    return t(f"Legni in 3D: {len(out['built'])} costruiti, {len(out['refused'])} rifiutati, {num(volume, 3)} m³"
             + (", chiglia data" if out.get("keel") else ""),
             f"Timbers in 3D: {len(out['built'])} built, {len(out['refused'])} refused, {num(volume, 3)} m³"
             + (", given keel" if out.get("keel") else ""))


def solids_warning(chunk) -> str:
    """What an export of the timbers in 3D should know: a cancelled build, solids older than the records."""
    from . import ship, solids

    done = ship.state(chunk).get("solids")
    if not done:
        return ""
    if done.get("cancelled"):
        return t("La costruzione dei legni in 3D è stata interrotta: i file sono parziali. Costruiscili di nuovo.",
                 "The build of the timbers in 3D was interrupted: the files are partial. Build them again.")
    if solids.is_stale(chunk):
        return t("I legni in 3D sono più vecchi del catalogo, della chiglia o del rilievo: i file esportati dicono ciò "
                 "che il catalogo diceva allora. Costruiscili di nuovo per avere quello di adesso.",
                 "The timbers in 3D are older than the catalogue, the keel or the relief: the files exported say what "
                 "the catalogue said then. Build them again for what it says now.")
    return ""


EXPORT_3D = (".obj", ".glb", ".ifc")


def check_export_path(path) -> None:
    from .ship import NotReady

    if Path(path).suffix.lower() not in EXPORT_3D:
        raise NotReady(t("Esporta in .obj, .glb o .ifc.", "Export to .obj, .glb or .ifc."))


def export_3d(chunk, path, system: str = "ship", tables: bool = True, remember: bool = True,
              reconstruction: bool = False) -> dict:
    """The timbers in 3D as ``path`` (.obj, .glb, .ifc by its suffix) in ``system``, with the tables beside it.
    ``reconstruction``: the reconstruction too (``_ric``, translucent; with the tables, ``tabella.csv`` and
    ``.xlsx`` of the measured and the reconstructed beside; refused in another ship frame, and ``note`` says when it
    is out of date: :func:`reconstruction_warning`). Returns ``{"path", "note", "csv"?, "xlsx"?, "table"?}``;
    with ``remember`` the path is kept in the chunk's meta for the next export (False from a worker thread: the
    caller keeps it in the interface thread)."""
    from . import models3d, rebuild, ship

    path = Path(path)
    check_export_path(path)
    out = {"path": path, "note": ""}
    warning = reconstruction_warning(chunk) if reconstruction else ""
    suffix = path.suffix.lower()
    if suffix == ".obj":
        models3d.write_obj(chunk, path, system, reconstruction=reconstruction)
    elif suffix == ".glb":
        models3d.write_glb(chunk, path, system, reconstruction=reconstruction)
    else:
        result = models3d.write_ifc(chunk, path, system, reconstruction=reconstruction)
        out["note"] = result.get("note") or ""
        out["elements"] = result.get("elements")
    if tables:
        out.update({k: v for k, v in models3d.write_tables(chunk, path.parent).items() if k in ("csv", "xlsx")})
        if reconstruction:
            out["table"] = rebuild.write_table(chunk, path.parent)["csv"]
    if warning:
        out["note"] = " ".join(x for x in (out["note"], warning) if x)
    if remember:
        ship.state(chunk)["model_export"] = str(path)
    return out


def strake_counts(chunk) -> dict:
    """The strakes found per side (the land lines plus one), for the sides with lands."""
    from . import clinker, ship

    state = (chunk.meta.get(ship.KEY) or {}).get("clinker") or {}
    return {side: int(state["mode"][side]) + 1 for side in clinker.SIDES if int((state.get("mode") or {}).get(side, 0))}


def detection_lines(chunk, result: dict) -> list[str]:
    """The stations a detection was unsure of, one list per reason, for the command line."""
    from .clinker import _reason_text, _side_name

    def listed(names):
        return ", ".join(names) or t("nessuna", "none")

    other = [f"{n} {_side_name(side)}" for side, names in result["uncertain"].items() for n in names]
    jumped = [f"{n} {_side_name(side)}" for side, names in result["jumps"].items() for n in names]
    skipped = [f"{d['station']} {_side_name(d['side'])} ({_reason_text(d['reason'])})" for d in result["skipped"]]
    return [t(f"Stazioni incerte (un altro numero di gradini): {listed(other)}",
              f"Uncertain stations (another count of lands): {listed(other)}"),
            t(f"Stazioni incerte (battura o torello che saltano): {listed(jumped)}",
              f"Uncertain stations (the rabbet or the garboard jumping): {listed(jumped)}"),
            t(f"Lati non letti: {'; '.join(skipped) or 'nessuno'}", f"Sides not read: {'; '.join(skipped) or 'none'}")]


def lines_files(chunk) -> dict:
    """The land lines written as DXF and GeoPackage (:func:`clinker.write_lines`); with no land line those of an
    earlier detection are removed, as they no longer hold."""
    from . import clinker, ship

    try:
        return clinker.write_lines(chunk)
    except ship.NotReady:
        for name in ("lands.dxf", "lands.gpkg"):
            (ship.folder(chunk) / "clinker" / name).unlink(missing_ok=True)
        return {}


def joined_values(args: list[str]) -> list[str]:
    """``--horizontals -0.2;-0.5`` as ``--horizontals=-0.2;-0.5``: argparse takes for an option anything that begins with
    a minus and is not one negative number."""
    out: list[str] = []
    for a in args:
        if out and out[-1] in NEGATIVE_VALUES and a[:1] == "-" and a[1:2] in tuple("0123456789."):
            out[-1] = f"{out[-1]}={a}"
        else:
            out.append(a)
    return out


def parse_values(text) -> list[float]:
    """Numbers separated by spaces or ``;``, decimal point or comma."""
    return [float(v.replace(",", ".")) for v in str(text).replace(";", " ").split()]


def _print_progress(fraction: float, message: str) -> None:
    print(f"{fraction * 100:5.1f} % {message}", flush=True)
